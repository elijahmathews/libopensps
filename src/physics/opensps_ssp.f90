module opensps_ssp
    !> @brief Core module for Simple Stellar Population (SSP) generation.
    !>
    !> @details
    !> This module calculates the time-evolution of a Single Stellar Population (SSP),
    !> which is the fundamental building block of stellar population synthesis.
    !> It combines:
    !> 1.  **Isochrone Data:** Stellar evolution tracks (Mass, L, T, g vs Age).
    !> 2.  **Spectral Libraries:** Flux distributions for individual stars.
    !> 3.  **IMF Weights:** Initial Mass Function weighting.
    !> 4.  **Stellar Physics:** Horizontal Branch, Blue Stragglers, AGB modifications.
    !>
    !> **Key Architecture:**
    !> This module replaces the legacy `SSP_GEN` subroutine. It uses a 
    !> "Recyclable Buffer" pattern (`isochrone_buffer_t`) to process one time-step 
    !> at a time using contiguous 1D arrays. This significantly improves L1/L2 
    !> cache locality compared to the legacy strided access on `(NT, NM)` global arrays.
    !>
    !> **Optimization Strategy:**
    !> - **Contiguous Memory:** Data for the current age is gathered into a compact buffer.
    !> - **Cache Blocking:** Spectral summation uses a block-loop (Batch Size 32) 
    !>   to keep the accumulator in L1 cache, maximizing vectorization throughput.
    !> - **Vectorization:** Critical loops are marked `contiguous` to enable SIMD.

    use opensps_precision, only: WP, KP
    use opensps_constants, only: NM, M_SOL, L_SOL, VERBOSE, BHB_SBS_TIME, TIME_RES_INCR, SAFE_FLOOR
    use opensps_types, only: params
    use opensps_context_types, only: opensps_context_t, opensps_context_sync_device_settings
    
    !> Physics Modules
    use opensps_imf, only: compute_imf_weights
    use opensps_stellar_modifications, only: modify_horizontal_branch, &
                                          apply_blue_stragglers, &
                                          modify_giant_branch, &
                                          add_remnant_mass, &
                                          add_xray_binaries
    use opensps_gas, only: apply_nebular_emission
    use opensps_smoothing, only: apply_smoothing
    use opensps_spectral_library, only: get_stellar_spectrum
    use opensps_photometry, only: compute_magnitudes

    !> Math Modules
    use opensps_interpolation, only: find_interval
    use opensps_integration, only: integrate_trapezoid_array

    implicit none
    private

    public :: generate_ssp_grid
    public :: isochrone_buffer_t
    public :: ssp_per_star, opensps_ssp_set_per_star

    !> Sum the SSP over individual star spectra (accumulate_spectrum) instead of through the
    !> template operator (opensps_ssp_operator): the reference the operator is tested against
    logical :: ssp_per_star = .false.

    !> Libraries of the small-template rows (small_template_index; L_MAIN: the main library)
    integer, parameter :: L_MAIN = 0, L_WMB = 1, L_PAGB = 2, L_WRN = 3, L_WRC = 4, L_CAR = 5, L_AGBO = 6, L_AGBC = 7
    public :: init_isochrone_buffer, reset_buffer, free_isochrone_buffer
    public :: load_timestep_data
    public :: apply_isochrone_physics, apply_isochrone_modifications
    public :: finish_ssp_grid, small_template_index, L_MAIN, L_WMB, L_PAGB, L_WRN, L_WRC, L_CAR, L_AGBO, L_AGBC
    public :: compute_integrated_properties, accumulate_spectrum
    public :: interpolate_time_grid
    public :: configure_imf_parameters
    public :: compute_interpolated_ssp
    public :: compute_surface_brightness_fluctuations

    !> generate_ssp_grid is in the submodule opensps_ssp_generate (src/physics/opensps_ssp_generate.f90):
    !> it uses the template operator (opensps_ssp_operator), which uses this module
    interface
        module subroutine generate_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)
            type(opensps_context_t), target, intent(inout) :: ctx
            type(params), intent(in)            :: pset
            real(WP), intent(out), contiguous   :: mass_grid(:)
            real(WP), intent(out), contiguous   :: lbol_grid(:)
            real(WP), intent(out), contiguous   :: spec_grid(:,:)
        end subroutine generate_ssp_grid
    end interface

    ! ------------------------------------------------------------------------
    ! CONSTANTS
    ! ------------------------------------------------------------------------
    real(WP), parameter :: MIN_WEIGHT_CUTOFF = 0.0_wp     ! Skip stars with negligible weight
    real(WP), parameter :: SMOOTHING_SIGMA_FLAG = 99.0_wp ! Flag to use variable sigma from LSF
    real(WP), parameter :: LN10 = log(10.0_wp)            ! Natural log of 10
    integer, parameter :: BATCH_SIZE = 32                 ! Process stars in chunks for L1 cache locality

    ! ------------------------------------------------------------------------
    ! DERIVED TYPES
    ! ------------------------------------------------------------------------
    
    !> @brief A recyclable buffer for processing a single isochrone time-step.
    !> @details
    !> Holds all properties for the stars alive at a specific age. 
    !> These are 1D CONTIGUOUS arrays of size NM (max stars).
    !> We allocate this ONCE and reuse it to avoid heap fragmentation.
    type :: isochrone_buffer_t
        integer :: n_stars
        !> Number of stars loaded from the isochrone itself (set by load_timestep_data), before
        !> apply_isochrone_physics appends blue stragglers / blue HB stars; 0 = all stars native
        integer :: n_native = 0
        
        ! Changed to POINTER to allow them to be targets for 2D views.
        ! We manually manage their memory in init/free routines.
        real(WP), pointer :: initial_mass(:) => null()
        real(WP), pointer :: current_mass(:) => null()
        real(WP), pointer :: log_lum(:)      => null()
        real(WP), pointer :: log_teff(:)     => null()
        real(WP), pointer :: log_g(:)        => null()
        real(WP), pointer :: phase(:)        => null()
        real(WP), pointer :: co_ratio(:)     => null()
        real(WP), pointer :: log_mdot(:)     => null()
        real(WP), pointer :: weights(:)      => null()
    end type isochrone_buffer_t

contains

    ! ------------------------------------------------------------------------
    ! MAIN DRIVER
    ! ------------------------------------------------------------------------


    !> @brief Post-processing of the SSP grid: time interpolation (TIME_RES_INCR > 1), nebular
    !> emission (add_neb_emission = 2), X-ray binaries and instrumental smoothing. The last
    !> step of generate_ssp_grid, shared with the batched SSP operator (opensps_ssp_operator).
    subroutine finish_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)
        type(opensps_context_t), target, intent(inout) :: ctx
        type(params), intent(in)            :: pset
        real(WP), intent(inout), contiguous :: mass_grid(:)
        real(WP), intent(inout), contiguous :: lbol_grid(:)
        real(WP), intent(inout), contiguous :: spec_grid(:,:)
        real(WP), allocatable               :: temp_spec_grid(:,:)
        real(KP), allocatable               :: kp_in(:,:), kp_out(:,:)

        ! A. TIME INTERPOLATION
        !    If the calculation was done on a coarse grid (TIME_RES_INCR > 1),
        !    interpolate to fill the gaps.
        if (TIME_RES_INCR > 1) then
            call interpolate_time_grid(ctx, pset%zmet, mass_grid, lbol_grid, spec_grid)
        end if

        ! B. NEBULAR EMISSION
        if (ctx%add_neb_emission_val == 2) then
            ! Kernel-precision copies in and out (also avoids aliasing the arguments)
            allocate(kp_in(size(spec_grid,1), size(spec_grid,2)), kp_out(size(spec_grid,1), size(spec_grid,2)))
            kp_in = real(spec_grid, KP)
            kp_out = 0.0_KP
            call apply_nebular_emission(ctx, pset, kp_in, kp_out)
            spec_grid = real(kp_out, WP)
            deallocate(kp_in, kp_out)
        end if

        ! C. X-RAY BINARIES
        if (ctx%add_xrb_emission_val == 1) then
            ! Note: Calls into opensps_stellar_modifications
            ! Use temp buffer to prevent Aliasing (In/Out same array)
            allocate(temp_spec_grid(size(spec_grid,1), size(spec_grid,2)))
            
            call add_xray_binaries(ctx, pset, spec_grid, temp_spec_grid)
            spec_grid = temp_spec_grid
            
            deallocate(temp_spec_grid)
        end if

        ! D. INSTRUMENTAL SMOOTHING
        if (ctx%smooth_lsf_val == 1) then
             call apply_lsf_smoothing_grid(ctx, spec_grid)
        end if

    end subroutine finish_ssp_grid

    ! ------------------------------------------------------------------------
    ! BUFFER MANAGEMENT
    ! ------------------------------------------------------------------------

    !> @brief Allocates the recyclable isochrone buffer.
    subroutine init_isochrone_buffer(buf)
        type(isochrone_buffer_t), intent(out) :: buf

        if (associated(buf%initial_mass)) return ! Already initialized
        
        ! Allocate to maximum size NM defined in opensps_constants.
        ! We do this once per SSP generation run.
        allocate(buf%initial_mass(NM))
        allocate(buf%current_mass(NM))
        allocate(buf%log_lum(NM))
        allocate(buf%log_teff(NM))
        allocate(buf%log_g(NM))
        allocate(buf%phase(NM))
        allocate(buf%co_ratio(NM))
        allocate(buf%log_mdot(NM))
        allocate(buf%weights(NM))
        
        call reset_buffer(buf)
    end subroutine init_isochrone_buffer

    !> @brief Resets scalars and zeroes arrays (soft clear).
    subroutine reset_buffer(buf)
        type(isochrone_buffer_t), intent(inout) :: buf
        buf%n_stars = 0
        buf%n_native = 0
        if (associated(buf%initial_mass)) buf%initial_mass = 0.0_wp
        if (associated(buf%current_mass)) buf%current_mass = 0.0_wp
        if (associated(buf%log_lum))      buf%log_lum = SAFE_FLOOR
        if (associated(buf%log_teff))     buf%log_teff = 0.0_wp
        if (associated(buf%log_g))        buf%log_g = 0.0_wp
        if (associated(buf%phase))        buf%phase = 0.0_wp
        if (associated(buf%co_ratio))     buf%co_ratio = 0.0_wp
        if (associated(buf%log_mdot))     buf%log_mdot = 0.0_wp
        if (associated(buf%weights))      buf%weights = 0.0_wp
    end subroutine reset_buffer

    !> @brief Deallocates the isochrone buffer.
    subroutine free_isochrone_buffer(buf)
        type(isochrone_buffer_t), intent(inout) :: buf
        
        ! Pointers use 'associated', not 'allocated'
        if (associated(buf%initial_mass)) deallocate(buf%initial_mass)
        if (associated(buf%current_mass)) deallocate(buf%current_mass)
        if (associated(buf%log_lum))      deallocate(buf%log_lum)
        if (associated(buf%log_teff))     deallocate(buf%log_teff)
        if (associated(buf%log_g))        deallocate(buf%log_g)
        if (associated(buf%phase))        deallocate(buf%phase)
        if (associated(buf%co_ratio))     deallocate(buf%co_ratio)
        if (associated(buf%log_mdot))     deallocate(buf%log_mdot)
        if (associated(buf%weights))      deallocate(buf%weights)
        
        ! Good practice to nullify after deallocation
        buf%initial_mass => null()
        buf%current_mass => null()
        buf%log_lum      => null()
        buf%log_teff     => null()
        buf%log_g        => null()
        buf%phase        => null()
        buf%co_ratio     => null()
        buf%log_mdot     => null()
        buf%weights      => null()
    end subroutine free_isochrone_buffer


    !> @brief Copies strided global data into the contiguous local buffer.
    !> @details 
    !> Optimizes memory access by gathering data from the strided `ctx` structure
    !> into the hot L1/L2 cache-friendly `buf` once per timestep.
    subroutine load_timestep_data(ctx, z_idx, t_idx, buf)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: z_idx, t_idx
        type(isochrone_buffer_t), intent(inout) :: buf
        
        integer :: n, i
        
        ! Get number of mass points, clamped to max NM
        ! Inside load_timestep_data
        if (ctx%state%nmass_isoc(z_idx, t_idx) > NM) then
            if (VERBOSE == 1) &
                write(*,*) 'SSP_GEN WARNING: Isochrone truncated at NM stars. Increase NM.'
        endif
        n = min(ctx%state%nmass_isoc(z_idx, t_idx), NM)
        buf%n_stars = n
        buf%n_native = n

        ! Vectorized Copy: Global(z, t, 1:n) -> Buffer(1:n)
        ! This pays the "stride tax" exactly once per timestep.
        ! Executed on HOST to prepare for apply_isochrone_physics (also HOST).
        do i = 1, n
            buf%initial_mass(i) = ctx%state%mini_isoc(z_idx, t_idx, i)
            buf%current_mass(i) = ctx%state%mact_isoc(z_idx, t_idx, i)
            buf%log_lum(i)      = ctx%state%logl_isoc(z_idx, t_idx, i)
            buf%log_teff(i)     = ctx%state%logt_isoc(z_idx, t_idx, i)
            buf%log_g(i)        = ctx%state%logg_isoc(z_idx, t_idx, i)
            buf%phase(i)        = ctx%state%phase_isoc(z_idx, t_idx, i)
            buf%co_ratio(i)     = ctx%state%ffco_isoc(z_idx, t_idx, i)
            buf%log_mdot(i)     = ctx%state%lmdot_isoc(z_idx, t_idx, i)
            
            ! Weights are calculated fresh every step, but zeroing is safe
            buf%weights(i)      = 0.0_wp
        end do

    end subroutine load_timestep_data

    ! ------------------------------------------------------------------------
    ! PHYSICS ORCHESTRATION
    ! ------------------------------------------------------------------------

    !> @brief Applies all stellar population modifications to the buffer.
    !>
    !> @details
    !> Transforms the raw isochrone data in the buffer by:
    !> 1. Computing IMF weights.
    !> 2. Redistributing Horizontal Branch stars (Blue HB).
    !> 3. Adding Blue Straggler stars.
    !> 4. Shifting and scaling Giant Branch (AGB/RGB) properties.
    !>
    !> @note
    !> Uses rank-remapping (get_2d_view) to interface the 1D buffer with 
    !> legacy physics routines that expect 2D arrays.
    !>
    !> @param[in,out] ctx          Simulation context.
    !> @param[in]     pset         User settings.
    !> @param[in]     time_log_yr  Current age in log(years).
    !> @param[in,out] buf          The working buffer (modified in place).
    subroutine apply_isochrone_physics(ctx, pset, time_log_yr, buf)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        real(WP), intent(in)                :: time_log_yr
        type(isochrone_buffer_t), intent(inout) :: buf

        ! 1. Calculate IMF Weights
        call compute_imf_weights(ctx, buf%initial_mass(1:buf%n_stars), &
                                 buf%weights(1:buf%n_stars), buf%n_stars)

        ! 2-4. Horizontal branch, blue stragglers, giant branch
        call apply_isochrone_modifications(ctx, pset, time_log_yr, buf)

    end subroutine apply_isochrone_physics

    !> @brief The isochrone modifications of apply_isochrone_physics (blue HB, blue stragglers,
    !> giant-branch shifts and weights), applied to buf%weights as set by the caller.
    !>
    !> @details
    !> All of them are linear in the weights and do not depend on them otherwise (which stars
    !> are added or changed depends only on the isochrone and pset), which the batched SSP
    !> operator (opensps_ssp_operator) uses: with unit weights it reads off each star's weight
    !> factor. The optional outputs are its bookkeeping: hb_flag(j) for the stars summed into
    !> the HB weight (whose sum sets the blue-straggler weights), parent(i) = j for blue HB
    !> stars i split from star j, n_after_hb = number of stars before the blue stragglers.
    subroutine apply_isochrone_modifications(ctx, pset, time_log_yr, buf, hb_flag, parent, n_after_hb)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        real(WP), intent(in)                :: time_log_yr
        type(isochrone_buffer_t), intent(inout) :: buf
        logical, intent(inout), optional    :: hb_flag(:)
        integer, intent(inout), optional    :: parent(:)
        integer, intent(out), optional      :: n_after_hb

        ! Local variables
        real(WP) :: hb_weight
        
        ! Wrapper for n_stars to satisfy legacy array interfaces
        integer :: n_stars_wrap(1)
        
        ! Pointers for 2D views (Rank-2)
        real(WP), pointer :: p_mini(:,:), p_mact(:,:)
        real(WP), pointer :: p_logl(:,:), p_logt(:,:), p_logg(:,:)
        real(WP), pointer :: p_phase(:,:)

        ! Initialize wrapper with current count
        n_stars_wrap(1) = buf%n_stars
        hb_weight = 0.0_wp

        ! --------------------------------------------------------------------
        ! PREPARE 2D VIEWS (Inline Rank Remapping)
        ! --------------------------------------------------------------------
        ! We map the 1D buffer (1:NM) to a 2D pointer (1:1, 1:NM).
        ! This is safe because buf%components are pointers (implicit targets).
        p_mini(1:1, 1:NM)  => buf%initial_mass
        p_mact(1:1, 1:NM)  => buf%current_mass
        p_logl(1:1, 1:NM)  => buf%log_lum
        p_logt(1:1, 1:NM)  => buf%log_teff
        p_logg(1:1, 1:NM)  => buf%log_g
        p_phase(1:1, 1:NM) => buf%phase

        ! --------------------------------------------------------------------
        ! 2. MODIFY HORIZONTAL BRANCH (HB)
        ! --------------------------------------------------------------------
        if (pset%fbhb > 0.0_wp .or. pset%sbss > 1.0e-3_wp) then
            
            ! We pass '1' as the time index because our view is (1, NM).
            ! n_stars_wrap is modified in-place if new stars are added.
            call modify_horizontal_branch( &
                ctx, 1, pset%fbhb, time_log_yr, hb_weight, &
                n_stars_wrap, &
                p_mini, p_mact, p_logl, p_logt, p_logg, p_phase, &
                buf%weights, hb_flag, parent) ! weights is 1D in the interface
        end if
        if (present(n_after_hb)) n_after_hb = n_stars_wrap(1)

        ! --------------------------------------------------------------------
        ! 3. ADD BLUE STRAGGLERS (BS)
        ! --------------------------------------------------------------------
        if (time_log_yr >= BHB_SBS_TIME .and. pset%sbss > 1.0e-3_wp) then
            
            call apply_blue_stragglers( &
                ctx, 1, pset%zmet, pset%sbss, hb_weight, &
                n_stars_wrap, &
                p_mini, p_mact, p_logl, p_logt, p_logg, p_phase, &
                buf%weights)
        end if

        ! Sync buffer count with the wrapper (in case stars were added)
        buf%n_stars = n_stars_wrap(1)

        ! --------------------------------------------------------------------
        ! 4. MODIFY GIANT BRANCH (AGB / RGB / PAGB)
        ! --------------------------------------------------------------------
        ! This routine does NOT add stars, so we pass buf%n_stars directly.
        ! We must pass pset%zmet (metallicity index) from settings.
        
        call modify_giant_branch( &
            ctx, 1, pset%zmet, time_log_yr, buf%n_stars, &
            pset%delt, pset%dell, pset%pagb, pset%redgb, pset%agb, &
            p_logl, p_logt, p_phase, buf%weights)

    end subroutine apply_isochrone_modifications

    ! ------------------------------------------------------------------------
    ! INTEGRATION KERNELS
    ! ------------------------------------------------------------------------

    !> @brief Computes integrated scalar properties (Mass, Lbol) for the population.
    !>
    !> @details
    !> 1. Sums the mass of living stars.
    !> 2. Adds the mass of stellar remnants (BH, NS, WD) based on the turn-off mass.
    !> 3. Sums the bolometric luminosity of all stars.
    !>
    !> @param[in,out] ctx      Simulation context.
    !> @param[in]     buf      The isochrone buffer (contains weights and properties).
    !> @param[out]    tot_mass Total Mass of the SSP [M_sol].
    !> @param[out]    tot_lbol Total Bolometric Luminosity [log10(L_sol)].
    subroutine compute_integrated_properties(ctx, buf, tot_mass, tot_lbol)
        type(opensps_context_t), intent(inout) :: ctx
        type(isochrone_buffer_t), intent(in) :: buf
        real(WP), intent(out) :: tot_mass, tot_lbol

        integer :: n, i, n_native
        real(WP) :: max_living_mass, local_mass, local_linear_lum

        n = buf%n_stars
        ! Stars appended after the native isochrone (blue stragglers, blue HB) do not define the
        ! turn-off: stragglers carry ZAMS masses above it, which would hide the remnants of stars
        ! that have already died
        n_native = buf%n_native
        if (n_native <= 0 .or. n_native > n) n_native = n
        
        ! Handle empty buffer edge case
        if (n <= 0) then
            tot_mass = 0.0_wp
            tot_lbol = -99.0_wp ! Arbitrary low value
            return
        end if

        ! 1. Calculate Mass of Living Stars
        !    Vectorized dot product: sum(weights * current_mass)
        local_mass = 0.0_wp
        ! On the host in every build and for both spectrum paths: a few thousand terms in a fixed
        ! order (a device reduction's order varies from run to run)
        do i = 1, n
            local_mass = local_mass + buf%weights(i) * buf%current_mass(i)
        end do
        
        ! 2. Add Remnant Mass (Black Holes, Neutron Stars, White Dwarfs)
        if (ctx%add_stellar_remnants_val == 1) then
             
             ! The turn-off mass is approximately the maximum initial mass
             ! of stars still present in the native isochrone.
             max_living_mass = -1.0_wp
             do i = 1, n_native
                 if (buf%initial_mass(i) > max_living_mass) max_living_mass = buf%initial_mass(i)
             end do
             
             ! The IMF integral of the dead stars (host)
             call add_remnant_mass(ctx, local_mass, max_living_mass)
        end if

           tot_mass = local_mass

        ! 3. Calculate Total Bolometric Luminosity
        !    L_tot = sum( weight * 10^logL )
        !    We compute the linear sum first, then take log10.
        local_linear_lum = 0.0_wp
        do i = 1, n
            local_linear_lum = local_linear_lum + buf%weights(i) * exp(buf%log_lum(i) * LN10)
        end do
        
        ! Prevent log(0)
        tot_lbol = log10(max(local_linear_lum, SAFE_FLOOR))

    end subroutine compute_integrated_properties

    !> @brief Sums the spectra of all stars in the buffer into the SSP spectrum.
    !>
    !> @details
    !> Iterates over every star in the isochrone buffer:
    !> 1. Checks filters (Weight > cutoff, EVTYPE, MASSCUT).
    !> 2. Generates the specific stellar spectrum for that star.
    !> 3. Accumulates the weighted spectrum into the total grid.
    !>
    !> @param[in,out] ctx       Simulation context.
    !> @param[in]     pset      User settings (for EVTYPE/MASSCUT).
    !> @param[in]     buf       The isochrone buffer.
    !> @param[in,out] spec_out  The output spectrum accumulator (L_sol/Hz).
    subroutine accumulate_spectrum(ctx, pset, buf, spec_out)
        type(opensps_context_t), target, intent(inout) :: ctx
        type(params), intent(in)            :: pset
        type(isochrone_buffer_t), intent(in):: buf
        real(WP), intent(inout), contiguous :: spec_out(:)

        ! Local variables
        integer :: j, i_spec
        integer :: nspec, nstars, n_active, ia
        real(WP) :: linear_lbol, current_weight
        integer, allocatable :: active_idx(:)
        real(KP), allocatable :: active_w(:)
        real(KP) :: acc, comp, term, tsum
        real(KP), allocatable :: acc_v(:), comp_v(:)
        
        nspec  = size(spec_out)
        nstars = buf%n_stars
        
        if (nstars == 0) return

        ! Grow context-owned workspace only when needed.
        if ((.not. allocated(ctx%state%ssp_temp_grid)) .or. &
            (ctx%state%ssp_temp_nspec < nspec) .or. &
            (ctx%state%ssp_temp_nstars < nstars)) then
            if (allocated(ctx%state%ssp_temp_grid)) then
                deallocate(ctx%state%ssp_temp_grid)
            end if
            allocate(ctx%state%ssp_temp_grid(nspec, nstars))
            ctx%state%ssp_temp_nspec = nspec
            ctx%state%ssp_temp_nstars = nstars
        end if

        allocate(active_idx(nstars), active_w(nstars))

        ! Build compact list of active stars once.
        n_active = 0
        do j = 1, nstars
            current_weight = buf%weights(j)
            if (current_weight <= MIN_WEIGHT_CUTOFF .or. &
                (pset%evtype /= -1 .and. int(buf%phase(j)) /= pset%evtype) .or. &
                (buf%initial_mass(j) >= pset%masscut)) then
                cycle
            end if

            n_active = n_active + 1
            active_idx(n_active) = j
            active_w(n_active) = real(current_weight, KP)
        end do

        if (n_active == 0) then
            spec_out = 0.0_wp
        else

            ! Initialize output
            spec_out = 0.0_wp

            ! ----------------------------------------------------------------
            ! PHASE 1: PARALLEL GENERATION (Gang over Stars)
            ! ----------------------------------------------------------------
            do ia = 1, n_active
                j = active_idx(ia)

                ! Convert LogL -> Linear L
                linear_lbol = exp(buf%log_lum(j) * LN10)

                ! Generate Spectrum into temp_grid column
                ! get_stellar_spectrum must be '!$acc routine seq'
                call get_stellar_spectrum( &
                    ctx, &
                    pset, &
                    buf%current_mass(j), &
                    buf%log_teff(j), &
                    linear_lbol, &
                    buf%log_g(j), &
                    buf%phase(j), &
                    buf%co_ratio(j), &
                    buf%log_mdot(j), &
                    ctx%state%ssp_temp_grid(:, ia))

            end do

            ! ----------------------------------------------------------------
            ! PHASE 2: REDUCTION
            ! ----------------------------------------------------------------
            ! Weighted sum over stars at kernel precision (a matrix-vector product). With FP32
            ! kernels the sum over thousands of stars is compensated (Kahan): a plain FP32 sum
            ! loses up to 1e-5 of the result, the compensated one stays at FP32 rounding (~1e-7).
            ! KP == WP is a compile-time constant, so each build keeps one loop.
            ! Stars outer, pixels inner: contiguous and vectorizable, with one accumulator (and
            ! compensation) per pixel; each pixel still sums the stars in the same order
            allocate(acc_v(nspec))
            acc_v = 0.0_KP
            if (KP == WP) then
                do ia = 1, n_active
                    do i_spec = 1, nspec
                        acc_v(i_spec) = acc_v(i_spec) + ctx%state%ssp_temp_grid(i_spec, ia) * active_w(ia)
                    end do
                end do
            else
                allocate(comp_v(nspec))
                comp_v = 0.0_KP
                do ia = 1, n_active
                    do i_spec = 1, nspec
                        term = ctx%state%ssp_temp_grid(i_spec, ia) * active_w(ia) - comp_v(i_spec)
                        tsum = acc_v(i_spec) + term
                        comp_v(i_spec) = (tsum - acc_v(i_spec)) - term
                        acc_v(i_spec) = tsum
                    end do
                end do
                deallocate(comp_v)
            end if
            do i_spec = 1, nspec
                spec_out(i_spec) = max(real(acc_v(i_spec), WP), SAFE_FLOOR)
            end do
            deallocate(acc_v)
        end if

        deallocate(active_idx, active_w)

    end subroutine accumulate_spectrum


    !> Library and template indices of small-template row r (1-based, after the main library):
    !> WMBasic (a, b), post-AGB (a, b), WN, WC, C-rich AGB
    !> (carbon stars), O-rich and C-rich AGB.
    subroutine small_template_index(ctx, r, lib, a, b)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)  :: r
        integer, intent(out) :: lib, a, b
        integer :: q, n1, n

        q = r
        b = 1
        n1 = size(ctx%state%wmb_spec, 3)
        n = n1 * size(ctx%state%wmb_spec, 4)
        if (q <= n) then
            lib = L_WMB; a = mod(q - 1, n1) + 1; b = (q - 1) / n1 + 1; return
        end if
        q = q - n
        n1 = size(ctx%state%pagb_spec, 2)
        n = n1 * size(ctx%state%pagb_spec, 3)
        if (q <= n) then
            lib = L_PAGB; a = mod(q - 1, n1) + 1; b = (q - 1) / n1 + 1; return
        end if
        q = q - n
        n = size(ctx%state%wrn_spec, 2)
        if (q <= n) then
            lib = L_WRN; a = q; return
        end if
        q = q - n
        n = size(ctx%state%wrc_spec, 2)
        if (q <= n) then
            lib = L_WRC; a = q; return
        end if
        q = q - n
        n = size(ctx%state%agb_spec_car, 2)
        if (q <= n) then
            lib = L_CAR; a = q; return
        end if
        q = q - n
        n = size(ctx%state%agb_spec_o, 2)
        if (q <= n) then
            lib = L_AGBO; a = q; return
        end if
        lib = L_AGBC; a = q - n
    end subroutine small_template_index

    !> Selects the per-star SSP sum (true) or the template sum (false, default). Callers outside
    !> the library must use this rather than assign ssp_per_star: with nvfortran, a program that
    !> assigns the module variable of a shared library changes its own copy, not the library's.
    subroutine opensps_ssp_set_per_star(flag)
        logical, intent(in) :: flag
        ssp_per_star = flag
    end subroutine opensps_ssp_set_per_star



    ! ------------------------------------------------------------------------
    ! POST-PROCESSING HELPERS
    ! ------------------------------------------------------------------------

    !> @brief Interpolates the sparse calculated time grid onto the full high-res grid.
    !>
    !> @details
    !> When `TIME_RES_INCR > 1`, the SSP generation only computes every Nth time step.
    !> This routine fills in the intermediate steps using interpolation.
    !>
    !> **Interpolation Logic:**
    !> - **Spectrum:** Log-linear interpolation (power law).
    !>   \f$ S_{new} = 10^{( (1-dt) \log S_1 + dt \log S_2 )} \f$
    !> - **Mass/Lbol:** Linear interpolation.
    !>   \f$ M_{new} = (1-dt) M_1 + dt M_2 \f$
    !>
    !> @param[in]     ctx       Simulation context (contains time arrays).
    !> @param[in]     z_idx     Metallicity index (to access correct time array).
    !> @param[in,out] mass_grid Integrated mass history (modified in place).
    !> @param[in,out] lbol_grid Integrated luminosity history (modified in place).
    !> @param[in,out] spec_grid Spectral grid (modified in place).
    subroutine interpolate_time_grid(ctx, z_idx, mass_grid, lbol_grid, spec_grid)
        type(opensps_context_t), intent(in), target :: ctx
        integer, intent(in)                 :: z_idx
        real(WP), intent(inout), contiguous :: mass_grid(:)
        real(WP), intent(inout), contiguous :: lbol_grid(:)
        real(WP), intent(inout), contiguous :: spec_grid(:,:)

        ! Local variables
        integer :: j
        integer :: idx_low, idx_high
        integer :: raw_idx_low
        real(WP) :: dt_weight
        real(WP) :: t_target, t1, t2
        integer :: nt, nt_full

        real(WP), pointer :: time_computed(:)
        real(WP), pointer :: time_full(:)

        time_computed => ctx%state%timestep_isoc(z_idx, :)
        time_full     => ctx%state%time_full

        nt      = ctx%state%nt
        nt_full = ctx%state%ntfull
        
        ! Optimization: Initialize "Hunter" index
        raw_idx_low = 1

        ! Iterate over the FULL high-res time grid
        do j = 1, nt_full
            
            if (mod(j - 1, TIME_RES_INCR) == 0) cycle

            t_target = time_full(j)
            
            ! Optimization: Sequential Hunt
            ! Since t_target increases monotonically, we only need to move forward.
            ! Check if we need to advance the index
            do while (raw_idx_low < nt - 1)
                if (time_computed(raw_idx_low + 1) > t_target) exit
                raw_idx_low = raw_idx_low + 1
            end do
            
            ! 2. Calculate Interpolation Weight (dt)
            t1 = time_computed(raw_idx_low)
            t2 = time_computed(raw_idx_low + 1)
            
            if (abs(t2 - t1) > tiny(0.0_wp)) then
                dt_weight = (t_target - t1) / (t2 - t1)
            else
                dt_weight = 0.0_wp
            end if

            ! 3. Map Sparse Indices
            idx_low  = 1 + (raw_idx_low - 1) * TIME_RES_INCR
            idx_high = idx_low + TIME_RES_INCR

            ! 4. Interpolate Scalar Properties
            mass_grid(j) = (1.0_wp - dt_weight) * mass_grid(idx_low) + &
                           (dt_weight)          * mass_grid(idx_high)

            lbol_grid(j) = (1.0_wp - dt_weight) * lbol_grid(idx_low) + &
                           (dt_weight)          * lbol_grid(idx_high)

            ! 5. Interpolate Spectrum
            spec_grid(:, j) = exp( &
                LN10 * (1.0_wp - dt_weight) * log10(max(spec_grid(:, idx_low), SAFE_FLOOR)) + &
                LN10 * (dt_weight)          * log10(max(spec_grid(:, idx_high), SAFE_FLOOR)) )

        end do

    end subroutine interpolate_time_grid
    
    !> @brief Configures IMF parameters and handles user-defined IMF files.
    !>
    !> @details
    !> Maps the user parameters (pset) into the global context state.
    !> If a user-defined IMF is selected (imf_type=5), this routine handles
    !> opening and parsing the definition file.
    !>
    !> @param[in,out] ctx   Simulation context.
    !> @param[in]     pset  User settings.
    subroutine configure_imf_parameters(ctx, pset)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset

        ! Local variables
        integer :: imf_type
        integer :: unit_imf, io_status
        integer :: i
        character(len=1024) :: filepath

        imf_type = ctx%imf_type_val

        ! 1. Validate IMF Type
        !    Supported types: 0=Salpeter, 1=Chabrier, 2=Kroupa, 
        !                     3=van Dokkum, 4=Dave, 5=User-defined
        if (imf_type < 0 .or. imf_type > 5) then
            write(*,*) 'SSP_GEN ERROR: IMF type outside of range [0-5]:', imf_type
            stop
        end if

        ! 2. Transfer Scalar Parameters
        ctx%state%imf_alpha(1) = pset%imf1
        ctx%state%imf_alpha(2) = pset%imf2
        ctx%state%imf_alpha(3) = pset%imf3
        ctx%state%imf_vdmc     = pset%vdmc
        ctx%state%imf_mdave    = pset%mdave

        ! Runtime IMF mass limits (imf_lower_limit / imf_upper_limit parameters). A user IMF file
        ! (type 5) overrides them below. The first-bin edge follows the lower limit, except for
        ! Geneva isochrones whose edge is fixed at setup from the lowest tabulated mass.
        ctx%state%imf_lower_limit = ctx%imf_lower_limit_val
        ctx%state%imf_upper_limit = ctx%imf_upper_limit_val
        if (trim(ctx%state%isoc_type) /= 'gnva') ctx%state%imf_lower_bound = ctx%state%imf_lower_limit

        ! 3. Handle User-Defined IMF (Type 5)
        if (imf_type == 5) then
            
            ! Determine filename
            if (len_trim(pset%imf_filename) == 0) then
                filepath = trim(ctx%sps_home) // '/data/imf.dat'
            else
                filepath = trim(ctx%sps_home) // '/data/' // trim(pset%imf_filename)
            end if

            ! Open file
            open(newunit=unit_imf, file=trim(filepath), status='old', &
                 action='read', iostat=io_status)
            
            if (io_status /= 0) then
                write(*,*) 'SSP_GEN ERROR: Could not open IMF file: ', trim(filepath)
                stop
            end if

            ! Read Loop
            i = 0
            do
                ! Read into row i+1
                read(unit_imf, *, iostat=io_status) &
                     ctx%state%imf_user_alpha(1, i+1), & ! Mass Lower
                     ctx%state%imf_user_alpha(2, i+1), & ! Mass Upper
                     ctx%state%imf_user_alpha(3, i+1)    ! Slope
                
                if (io_status /= 0) exit ! EOF or Error
                
                i = i + 1
                
                ! Safety check for array bounds (100 is typical hard limit in legacy)
                if (i >= 100) then
                    write(*,*) 'SSP_GEN WARNING: Reached max segments (100) in user IMF.'
                    exit
                end if
            end do
            
            close(unit_imf)

            if (i == 0) then
                write(*,*) 'SSP_GEN ERROR: User IMF file was empty or invalid.'
                stop
            end if

            ! Store count and limits
            ctx%state%n_user_imf = i
            
            ! Define limits based on first lower bound and last upper bound
            ctx%state%imf_lower_limit = ctx%state%imf_user_alpha(1, 1)
            ctx%state%imf_upper_limit = ctx%state%imf_user_alpha(2, i)

        end if
        
        ! Verbose Logging (Optional, matching legacy behavior)
        if (VERBOSE == 1) then
            write(*,*) ''
            write(*,'("   Log(Z/Zsol): ",F6.3)') log10(ctx%state%zlegend(pset%zmet)/0.019_wp)
            write(*,'("   Fraction of blue HB stars: ",F6.3)') pset%fbhb
            write(*,'("   Ratio of BS to HB stars  : ",F6.3)') pset%sbss
            write(*,'("   Shift to TP-AGB [log(Teff),log(Lbol)]: ",F5.2,1x,F5.2)') &
                 pset%delt, pset%dell
            
            select case (imf_type)
            case (2)
                write(*,'("   IMF: ",I1,", slopes= ",3F4.1)') imf_type, ctx%state%imf_alpha
            case (3)
                write(*,'("   IMF: ",I1,", cut-off= ",F4.2)') imf_type, ctx%state%imf_vdmc
            case default
                write(*,'("   IMF: ",I1)') imf_type
            end select
        end if

    end subroutine configure_imf_parameters
    
    !> @brief Applies instrumental Line Spread Function (LSF) smoothing to the grid.
    !>
    !> @details
    !> Iterates over every time step in the spectral grid and convolves the 
    !> spectrum with the user-provided LSF (if loaded).
    !>
    !> @param[in]     ctx       Simulation context (contains LSF data).
    !> @param[in,out] spec_grid The full spectral grid (Lambda, Time).
    subroutine apply_lsf_smoothing_grid(ctx, spec_grid)
        type(opensps_context_t), intent(in)    :: ctx
        real(WP), intent(inout), contiguous :: spec_grid(:,:)

        integer :: j, n_times

        n_times = size(spec_grid, 2)

        ! Loop over time columns
        do j = 1, n_times
            
            ! Call the smoothing kernel for this specific time step.
            ! Note: The '99.0_wp' is a special flag in opensps_smoothing indicating
            ! that we should use the tabulated LSF kernel rather than a 
            ! simple Gaussian velocity dispersion.
            
            call apply_smoothing( &
                ctx, &
                ctx%state%spec_lambda, &      ! Wavelength grid
                spec_grid(:, j), &            ! Spectrum (in/out)
                SMOOTHING_SIGMA_FLAG, &       ! Sigma flag
                ctx%state%lsfinfo%minlam, &   ! LSF range min
                ctx%state%lsfinfo%maxlam, &   ! LSF range max
                ctx%state%lsfinfo%lsf)        ! LSF kernel array

        end do

    end subroutine apply_lsf_smoothing_grid

    ! ------------------------------------------------------------------------
    ! INTERPOLATION & QUERY UTILITIES
    ! ------------------------------------------------------------------------

    !> @brief Interpolates or integrates the SSP grid over Metallicity (Z) and Age (T).
    !>
    !> @details
    !> Performs one of three operations based on arguments:
    !> 1. **Point Interpolation:** Returns SSP properties at specific `z_pos` and `t_pos`.
    !> 2. **Isochrone Interpolation:** Returns full time-grid for a specific `z_pos`.
    !> 3. **MDF Integration:** Integrates over a Metallicity Distribution Function (MDF) 
    !>    defined by `z_pos` (mean) and `z_width_or_power` (width or power law).
    !>
    !> @param[in]  ctx       FSPS Context containing the `_ssp_zz` grids.
    !> @param[in]  z_pos     Target log(Z/Zsol).
    !> @param[out] mass_out  Output integrated mass.
    !> @param[out] lbol_out  Output integrated bolometric luminosity.
    !> @param[out] spec_out  Output spectrum.
    !> @param[in]  t_pos     (Optional) Target Age in log(years). If present, output is scalar (time dimension collapsed).
    !> @param[in]  z_param   (Optional) If < 0, treats as smoothing sigma. If > 0, treats as power-law index for MDF.
    subroutine compute_interpolated_ssp(ctx, z_pos, mass_out, lbol_out, spec_out, t_pos, z_param)
        type(opensps_context_t), intent(in), target :: ctx
        real(WP), intent(in)             :: z_pos
        real(WP), intent(out)            :: mass_out(:), lbol_out(:)
        real(WP), intent(out)            :: spec_out(:,:)
        real(WP), intent(in), optional   :: t_pos
        real(WP), intent(in), optional   :: z_param

        ! Local variables
        integer :: nz, nt, nspec
        integer :: z_lo, z_hi, t_lo
        integer :: i
        real(WP) :: dt, dz, z0
        real(WP) :: weight_lo, weight_hi
        real(WP), allocatable :: mdf_weights(:)
        
        ! Pointers to context arrays for readability
        real(WP), pointer :: grid_mass(:,:), grid_lbol(:,:), grid_spec(:,:,:)
        real(WP), pointer :: z_legend(:), t_full(:)
        real(WP) :: z_sol

        ! Validate Context
        if (.not. allocated(ctx%state%mass_ssp_zz)) then
             error stop "compute_interpolated_ssp: SSP Grid not generated yet."
        end if

        ! Bind pointers
        nz        = ctx%state%nz
        nt        = ctx%state%ntfull
        nspec     = ctx%state%nspec
        z_legend  => ctx%state%zlegend
        z_sol     = ctx%state%zsol
        t_full    => ctx%state%time_full
        grid_mass => ctx%state%mass_ssp_zz
        grid_lbol => ctx%state%lbol_ssp_zz
        grid_spec => ctx%state%spec_ssp_zz

        ! --------------------------------------------------------------------
        ! CASE 1: POINT INTERPOLATION (Specific Z, Specific T)
        ! --------------------------------------------------------------------
        if (present(t_pos)) then
            
            ! Validation
            if (present(z_param)) then
                error stop "compute_interpolated_ssp: Cannot specify both Age (t_pos) and MDF (z_param)."
            end if
            if (size(mass_out) > 1) then
                error stop "compute_interpolated_ssp: t_pos specified, but output arrays are arrays, not scalars."
            end if

            ! 1. Find Time Interval
            t_lo = max(1, min(find_interval(t_full, t_pos), nt - 1))
            dt   = (t_pos - t_full(t_lo)) / (t_full(t_lo+1) - t_full(t_lo))

            ! 2. Find Z Interval
            z_lo = max(1, min(find_interval(log10(z_legend/z_sol), z_pos), nz - 1))
            dz   = (z_pos - log10(z_legend(z_lo)/z_sol)) / &
                   (log10(z_legend(z_lo+1)/z_sol) - log10(z_legend(z_lo)/z_sol))

            ! 3. Bilinear Interpolation
            !    f(z,t) = (1-dz)(1-dt)*00 + dz(1-dt)*10 + (1-dz)dt*01 + dz*dt*11
            
            ! Precompute weights
            weight_lo = 1.0_wp - dz
            weight_hi = dz

            ! Interpolate Mass
            mass_out(1) = (1.0_wp - dt) * (weight_lo * grid_mass(t_lo, z_lo)   + weight_hi * grid_mass(t_lo, z_lo+1)) + &
                          (dt)          * (weight_lo * grid_mass(t_lo+1, z_lo) + weight_hi * grid_mass(t_lo+1, z_lo+1))

            ! Interpolate Lbol
            lbol_out(1) = (1.0_wp - dt) * (weight_lo * grid_lbol(t_lo, z_lo)   + weight_hi * grid_lbol(t_lo, z_lo+1)) + &
                          (dt)          * (weight_lo * grid_lbol(t_lo+1, z_lo) + weight_hi * grid_lbol(t_lo+1, z_lo+1))

            ! Interpolate Spectrum (Vectorized)
            spec_out(:,1) = (1.0_wp - dt) * (weight_lo * grid_spec(:, t_lo, z_lo)   + weight_hi * grid_spec(:, t_lo, z_lo+1)) + &
                            (dt)          * (weight_lo * grid_spec(:, t_lo+1, z_lo) + weight_hi * grid_spec(:, t_lo+1, z_lo+1))

            return
        end if

        ! --------------------------------------------------------------------
        ! CASE 2 & 3: GRID OUTPUT (Full Time History)
        ! --------------------------------------------------------------------
        
        if (present(z_param)) then
            ! --- MDF INTEGRATION ---
            allocate(mdf_weights(nz))
            mdf_weights = 0.0_wp

            z_lo = max(1, min(find_interval(log10(z_legend/z_sol), z_pos), nz - 1))
            dz   = (z_pos - log10(z_legend(z_lo)/z_sol)) / &
                   (log10(z_legend(z_lo+1)/z_sol) - log10(z_legend(z_lo)/z_sol))

            if (z_param < 0.0_wp) then
                ! Triangular Smoothing Kernel (Legacy Logic)
                ! w1=0.25, w2=0.5, w3=0.25 implicit in logic below
                ! This smooths neighboring metallicity points.
                
                ! Center weights
                mdf_weights(z_lo)   = mdf_weights(z_lo)   + 0.5_wp*(1.0_wp - dz) + 0.25_wp*dz
                mdf_weights(z_lo+1) = mdf_weights(z_lo+1) + 0.25_wp*(1.0_wp - dz) + 0.5_wp*dz
                
                ! Wings (handling boundaries)
                if (z_lo > 1)  mdf_weights(z_lo-1) = 0.25_wp * (1.0_wp - dz)
                if (z_lo+2 <= nz) mdf_weights(z_lo+2) = 0.25_wp * dz
                
                ! Set Loop bounds for optimization
                z_lo = max(1, z_lo - 1)
                z_hi = min(nz, z_lo + 3) ! Just scan the local area

            else
                ! Power Law MDF: dN/dZ ~ (Z/Z0)^pow * exp(-Z/Z0)
                z0  = (10.0_wp**z_pos) * z_sol
                
                ! Calculate weights for all Z
                mdf_weights = (z_legend / z0 * exp(-z_legend / z0))**z_param
                
                z_lo = 1
                z_hi = nz
            end if

            if (sum(mdf_weights) > tiny(0.0_wp)) then
                mdf_weights = mdf_weights / sum(mdf_weights) ! Normalize
            end if

            ! Perform Integration
            mass_out = 0.0_wp
            lbol_out = 0.0_wp
            spec_out = 0.0_wp

            do i = z_lo, z_hi
                if (mdf_weights(i) <= tiny(0.0_wp)) cycle
                
                ! Vectorized accumulation
                mass_out = mass_out + mdf_weights(i) * grid_mass(:, i)
                lbol_out = lbol_out + mdf_weights(i) * grid_lbol(:, i)
                
                ! Loop order for spectrum: Spectrum is (Lambda, Time, Z)
                ! We are summing over Z, so we add (Lambda, Time) slices.
                spec_out = spec_out + mdf_weights(i) * grid_spec(:, :, i)
            end do

        else
            ! --- SIMPLE Z INTERPOLATION (No Age Interpolation) ---
            z_lo = max(1, min(find_interval(log10(z_legend/z_sol), z_pos), nz - 1))
            dz   = (z_pos - log10(z_legend(z_lo)/z_sol)) / &
                   (log10(z_legend(z_lo+1)/z_sol) - log10(z_legend(z_lo)/z_sol))

            weight_lo = 1.0_wp - dz
            weight_hi = dz

            ! Linear Interpolation between two Z planes
            mass_out = weight_lo * grid_mass(:, z_lo)   + weight_hi * grid_mass(:, z_lo+1)
            lbol_out = weight_lo * grid_lbol(:, z_lo)   + weight_hi * grid_lbol(:, z_lo+1)
            spec_out = weight_lo * grid_spec(:, :, z_lo) + weight_hi * grid_spec(:, :, z_lo+1)
        end if

    end subroutine compute_interpolated_ssp

    !> @brief Compute surface brightness fluctuation (SBF) magnitudes.
    !>
    !> @details
    !> Computes SBF magnitudes by summing the first and second moments of the
    !> stellar luminosity function for each isochrone time step.
    !>
    !> @param[inout] ctx     The FSPS context (must be initialized).
    !> @param[in]    pset    Parameter set (uses `zmet`, `fbhb`, `sbss`, etc.).
    !> @param[in]    outfile Output filename stem (written to OUTPUTS/<name>.mags).
    subroutine compute_surface_brightness_fluctuations(ctx, pset, outfile)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in) :: pset
        character(len=*), intent(in) :: outfile

        integer :: i, j
        character(34) :: fmt
        real(WP) :: zero, hb_wght
        real(WP), dimension(NM) :: wght
        real(KP), allocatable :: tspec(:)
        real(WP), allocatable :: tspec2(:), spec1(:), spec2(:)
        real(WP), allocatable :: mags(:)
        real(WP), allocatable :: mini(:, :), mact(:, :), logl(:, :), logt(:, :), logg(:, :)
        real(WP), allocatable :: ffco(:, :), phase(:, :), lmdot(:, :)
        integer, allocatable :: nmass(:)
        real(WP), allocatable :: time(:)

        zero = 0.0_wp

        associate( &
            nbands => ctx%state%nbands, nspec => ctx%state%nspec, nt => ctx%state%nt, &
            output_home => ctx%output_home, &
            mini_isoc => ctx%state%mini_isoc, mact_isoc => ctx%state%mact_isoc, &
            logl_isoc => ctx%state%logl_isoc, logt_isoc => ctx%state%logt_isoc, &
            logg_isoc => ctx%state%logg_isoc, ffco_isoc => ctx%state%ffco_isoc, &
            lmdot_isoc => ctx%state%lmdot_isoc, phase_isoc => ctx%state%phase_isoc, &
            nmass_isoc => ctx%state%nmass_isoc, timestep_isoc => ctx%state%timestep_isoc )

            allocate (tspec(nspec), tspec2(nspec), spec1(nspec), spec2(nspec))
            allocate (mags(nbands))
            allocate (mini(nt, NM), mact(nt, NM), logl(nt, NM), logt(nt, NM), logg(nt, NM))
            allocate (ffco(nt, NM), phase(nt, NM), lmdot(nt, NM))
            allocate (nmass(nt), time(nt))

            fmt = '(F7.4,1x,3(F8.4,1x),000(F7.3,1x))'
            write (fmt(21:23), '(I3,1x,I4)') nbands

            hb_wght = 0.0_wp
            wght = 0.0_wp

            open (56, file=trim(output_home)//'/OUTPUTS/'//trim(outfile)//'.mags', status='replace')
            do i = 1, 8
                write (56, *) '#'
            end do

            mini = mini_isoc(pset%zmet, :, :)
            mact = mact_isoc(pset%zmet, :, :)
            logl = logl_isoc(pset%zmet, :, :)
            logt = logt_isoc(pset%zmet, :, :)
            logg = logg_isoc(pset%zmet, :, :)
            ffco = ffco_isoc(pset%zmet, :, :)
            lmdot = lmdot_isoc(pset%zmet, :, :)
            phase = phase_isoc(pset%zmet, :, :)
            nmass = nmass_isoc(pset%zmet, :)
            time = timestep_isoc(pset%zmet, :)

            do i = 1, nt
                call compute_imf_weights(ctx, mini(i, :), wght, nmass(i))

                if (pset%fbhb > 0.0_wp .or. pset%sbss > 1.0e-3_wp) then
                    call modify_horizontal_branch(ctx, i, pset%fbhb, time(i), hb_wght, nmass, &
                        mini, mact, logl, logt, logg, phase, wght)
                end if

                if (time(i) >= BHB_SBS_TIME .and. pset%sbss > 1.0e-3_wp) then
                    call apply_blue_stragglers(ctx, i, pset%zmet, pset%sbss, hb_wght, nmass, &
                        mini, mact, logl, logt, logg, phase, wght)
                end if

                call modify_giant_branch(ctx, i, pset%zmet, time(i), nmass(i), pset%delt, pset%dell, pset%pagb, &
                    pset%redgb, pset%agb, logl, logt, phase, wght)

                spec1 = 0.0_wp
                spec2 = 0.0_wp
                do j = 1, nmass(i)
                    call get_stellar_spectrum(ctx, pset, mact(i, j), logt(i, j), 10.0_wp**logl(i, j), logg(i, j), &
                        phase(i, j), ffco(i, j), lmdot(i, j), tspec)
                    spec2 = spec2 + wght(j)*real(tspec, WP)**2
                    spec1 = spec1 + wght(j)*real(tspec, WP)
                end do

                tspec2 = spec2/spec1
                call compute_magnitudes(ctx, zero, tspec2, mags)
                write (56, fmt) time(i), 0.0_wp, 0.0_wp, 0.0_wp, mags
            end do

            close (56)

            deallocate (tspec, tspec2, spec1, spec2, mags)
            deallocate (mini, mact, logl, logt, logg, ffco, phase, lmdot)
            deallocate (nmass, time)
        end associate
    end subroutine compute_surface_brightness_fluctuations

end module opensps_ssp