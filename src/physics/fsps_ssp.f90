module fsps_ssp
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

    use fsps_precision, only: WP, KP
    use fsps_constants, only: NM, M_SOL, L_SOL, VERBOSE, BHB_SBS_TIME, TIME_RES_INCR, SAFE_FLOOR
    use fsps_types, only: params
    use fsps_context_types, only: fsps_context_t, fsps_context_sync_device_settings
    
    !> Physics Modules
    use fsps_imf, only: compute_imf_weights
    use fsps_stellar_modifications, only: modify_horizontal_branch, &
                                          apply_blue_stragglers, &
                                          modify_giant_branch, &
                                          add_remnant_mass, &
                                          add_xray_binaries
    use fsps_gas, only: apply_nebular_emission
    use fsps_smoothing, only: apply_smoothing
    use fsps_spectral_library, only: get_stellar_spectrum, get_stellar_terms, SRC_MAIN_LIB, SRC_WMBASIC, &
                                     SRC_AGB_O, SRC_AGB_C, SRC_PAGB, SRC_WRN, SRC_WRC, SRC_AGB_CAR
    use fsps_photometry, only: compute_magnitudes

    !> Math Modules
    use fsps_interpolation, only: find_interval
    use fsps_dust, only: get_agb_dust_terms
    use fsps_constants, only: C_LIGHT
    use fsps_integration, only: integrate_trapezoid_array

    implicit none
    private

    public :: generate_ssp_grid
    public :: isochrone_buffer_t
    public :: ssp_per_star, fsps_ssp_set_per_star

    !> Sum the SSP over individual star spectra instead of over library templates (the two are
    !> equal; the per-star form is kept for the OpenACC build and to test the template form)
    logical :: ssp_per_star = .false.

    !> Exact power of two keeping the O/C-rich AGB sums (before lambda^2) clear of FP32 subnormals;
    !> 1 with FP64 kernels, which do not need it (with it, gfortran FP64 SSP grids ran 25% slower)
    real(WP), parameter :: AGB_SCALE = merge(2.0_wp**64, 1.0_wp, KP /= WP)
    !> Libraries of the deferred template weights (OpenACC builds: main library, then these)
    integer, parameter :: L_MAIN = 0, L_WMB = 1, L_PAGB = 2, L_WRN = 3, L_WRC = 4, L_CAR = 5, L_AGBO = 6, L_AGBC = 7

    !> Dust-shell AGB corners of all ages, deferred to the device product (OpenACC builds): per
    !> corner ("slot") its age, transfer-function indices (flux_dagb) and template weights
    !> (library, a, b, weight), stored as consecutive entries slot_ptr(s) .. slot_ptr(s+1) - 1.
    type :: dusty_defer_t
        integer :: cur_age = 0, n_slots = 0, n_ent = 0
        integer,  allocatable :: slot_age(:), slot_key(:,:), slot_ptr(:)
        integer,  allocatable :: ent_lib(:), ent_a(:), ent_b(:)
        real(WP), allocatable :: ent_w(:)
    end type dusty_defer_t
    public :: init_isochrone_buffer, reset_buffer, free_isochrone_buffer
    public :: load_timestep_data
    public :: apply_isochrone_physics, apply_isochrone_modifications
    public :: sum_small_templates, sum_dusty_slots, add_main_library_templates, finish_ssp_grid
    public :: add_column, add_column_sp
    public :: compute_integrated_properties, accumulate_spectrum
    public :: interpolate_time_grid
    public :: configure_imf_parameters
    public :: compute_interpolated_ssp
    public :: compute_surface_brightness_fluctuations

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

    !> @brief Main driver routine to generate the SSP grid.
    !>
    !> @details
    !> Orchestrates the extraction of isochrones, application of stellar physics,
    !> IMF weighting, spectral integration, and post-processing.
    !>
    !> **Pipeline:**
    !> 1. **Setup:** Initialize buffers and handle special cases (BPASS).
    !> 2. **Evolution Loop:** For each time step:
    !>    - Load isochrone data into contiguous buffer.
    !>    - Apply stellar physics (HB, BS, AGB).
    !>    - Sum integrated properties (Mass, Lbol).
    !>    - Accumulate Spectra.
    !> 3. **Post-Processing:**
    !>    - Interpolate time grid (if high-res).
    !>    - Add Nebular Emission.
    !>    - Add X-Ray Binaries.
    !>    - Apply Instrumental Smoothing (LSF).
    !>
    !> @param[in,out] ctx       The main FSPS context.
    !> @param[in]     pset      User parameter set.
    !> @param[out]    mass_grid Output mass history (time dependent).
    !> @param[out]    lbol_grid Output bolometric luminosity history.
    !> @param[out]    spec_grid Output spectral grid (wavelength, time).
    subroutine generate_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)
        type(fsps_context_t), target, intent(inout) :: ctx
        type(params), intent(in)            :: pset
        real(WP), intent(out), contiguous   :: mass_grid(:)
        real(WP), intent(out), contiguous   :: lbol_grid(:)
        real(WP), intent(out), contiguous   :: spec_grid(:,:)

        ! Local variables
        type(isochrone_buffer_t) :: buf
        integer :: n_curr, n_times, i_time, out_idx
        real(WP) :: time_log_yr
        ! Main-library template weights of every age (templates x ages), summed after the age
        ! loop as one blocked product (CPU template-sum path)
        real(WP), allocatable :: w_main_all(:,:)
        type(dusty_defer_t) :: dusty

        ! --------------------------------------------------------------------
        ! 1. INITIALIZATION
        ! --------------------------------------------------------------------
        
        call fsps_context_sync_device_settings(ctx)
        ! Initialize outputs to zero
        mass_grid = 0.0_wp
        lbol_grid = 0.0_wp
        spec_grid = 0.0_wp

        ! Validation: Check for setup flag
        if (ctx%state%check_sps_setup == 0) then
            write(*,*) 'SSP_GEN ERROR: SPS_SETUP must be run before SSP_GEN.'
            stop
        end if

        ! Validation: Check metallicity index
        if (pset%zmet < 1 .or. pset%zmet > ctx%state%nz) then
            write(*,*) 'SSP_GEN ERROR: Metallicity index out of range:', pset%zmet
            stop
        end if

        ! --------------------------------------------------------------------
        ! 2. SPECIAL CASE: BPASS MODELS
        ! --------------------------------------------------------------------
        ! BPASS models are pre-computed and stored in bpass_* arrays.
        ! We do not synthesize them star-by-star.
        if (trim(ctx%state%isoc_type) == 'bpss') then
            
            ! Copy pre-computed data for the requested metallicity
            spec_grid = ctx%state%bpass_spec_ssp(:, :, pset%zmet)
            mass_grid = ctx%state%bpass_mass_ssp(:, pset%zmet)

            ! BPASS ships no isochrones, so L_bol is the integral of the SSP spectrum
            ! (Lsun/Hz over Hz). Legacy FSPS leaves it unset (log L = 0 at every age).
            do i_time = 1, size(lbol_grid)
                lbol_grid(i_time) = log10(max(integrate_trapezoid_array(ctx%state%spec_nu(1:size(spec_grid, 1)), &
                                                                        spec_grid(:, i_time)), tiny(1.0_wp)))
            end do
            
            ! Push the assigned arrays to the device so downstream device
            ! kernels (like CSP) see the valid data.
            !$acc update device(spec_grid, lbol_grid, mass_grid) if_present
            
            return ! Exit immediately
        end if

        ! --------------------------------------------------------------------
        ! 3. SETUP & CONFIGURATION
        ! --------------------------------------------------------------------

        ! Configure IMF parameters (maps pset -> context global vars)
        ! Also handles legacy file reading for user-defined IMFs.
        call configure_imf_parameters(ctx, pset)

        ! Initialize the reusable memory buffer
        call init_isochrone_buffer(buf)
        
        ! Move buffer to device
        !$acc enter data copyin(buf)
        !$acc enter data create(buf%initial_mass, buf%current_mass, buf%log_lum, buf%log_teff)
        !$acc enter data create(buf%log_g, buf%phase, buf%co_ratio, buf%log_mdot, buf%weights)
        !$acc enter data attach(buf%initial_mass)
        !$acc enter data attach(buf%current_mass)
        !$acc enter data attach(buf%log_lum)
        !$acc enter data attach(buf%log_teff)
        !$acc enter data attach(buf%log_g)
        !$acc enter data attach(buf%phase)
        !$acc enter data attach(buf%co_ratio)
        !$acc enter data attach(buf%log_mdot)
        !$acc enter data attach(buf%weights)

        ! --------------------------------------------------------------------
        ! 4. EVOLUTION LOOP
        ! --------------------------------------------------------------------
        
        n_times = ctx%state%nt
        if (.not. ssp_per_star) then
            ! OpenACC builds also defer the other libraries' template weights (extra rows, summed
            ! with the main library on the device); CPU builds sum those per age
#ifdef _OPENACC
            allocate(w_main_all(size(ctx%state%speclib, 3) * size(ctx%state%speclib, 4) + n_small_templates(ctx), n_times))
#else
            allocate(w_main_all(size(ctx%state%speclib, 3) * size(ctx%state%speclib, 4), n_times))
#endif
            w_main_all = 0.0_wp
        end if

        ! The per-star kernel builds each age's spectrum on the device; the template path
        ! aggregates weights on the host and adds the main library for all ages at the end
        !$acc data present(ctx, buf) copy(spec_grid) if(ssp_per_star)
        do i_time = 1, n_times
            
            ! Optimization: Skip ages not requested by the user
            if (pset%ssp_gen_age(i_time) == 0) cycle

            ! Map internal time index to output grid index
            ! (Supports legacy spacing logic via TIME_RES_INCR)
            out_idx = 1 + (i_time - 1) * TIME_RES_INCR

            ! Get current Age (log years)
            time_log_yr = ctx%state%timestep_isoc(pset%zmet, i_time)

            ! A. LOAD DATA
            !    Copy strided global data into our contiguous buffer
            call load_timestep_data(ctx, pset%zmet, i_time, buf)

            ! B. APPLY PHYSICS
            !    IMF, Horizontal Branch, Blue Stragglers, Giant Branch modifications
            call apply_isochrone_physics(ctx, pset, time_log_yr, buf)

            ! Sync buffer to device for the per-star kernel (the template path and
            ! compute_integrated_properties then run on the host)
            n_curr = buf%n_stars
            if (ssp_per_star) then
                !$acc update device(buf%n_stars)
                if (n_curr > 0) then
                    !$acc update device(buf%initial_mass(1:n_curr), buf%current_mass(1:n_curr), buf%log_lum(1:n_curr))
                    !$acc update device(buf%log_teff(1:n_curr), buf%log_g(1:n_curr), buf%phase(1:n_curr))
                    !$acc update device(buf%co_ratio(1:n_curr), buf%log_mdot(1:n_curr), buf%weights(1:n_curr))
                end if
            end if

            ! C. COMPUTE INTEGRATED PROPERTIES
            !    Mass and Bolometric Luminosity
            call compute_integrated_properties(ctx, buf, mass_grid(out_idx), lbol_grid(out_idx))

            ! Prevent host/device desync if the caller mapped these to the device
            !$acc update device(mass_grid(out_idx), lbol_grid(out_idx)) if_present

            ! D. ACCUMULATE SPECTRA
            !    Sum individual stellar spectra into the grid
            if (allocated(w_main_all)) then
#ifdef _OPENACC
                dusty%cur_age = i_time
                call accumulate_spectrum(ctx, pset, buf, spec_grid(:, out_idx), w_main_all(:, i_time), dusty)
#else
                call accumulate_spectrum(ctx, pset, buf, spec_grid(:, out_idx), w_main_all(:, i_time))
#endif
            else
                call accumulate_spectrum(ctx, pset, buf, spec_grid(:, out_idx))
            end if

        end do
        !$acc end data
        ! The per-star kernel left the spectra on the device; if the caller mapped spec_grid
        ! (copy above was then a no-op), bring them back for the host post-processing below
        if (ssp_per_star) then
            !$acc update self(spec_grid) if_present
        end if

        if (allocated(w_main_all)) then
#ifdef _OPENACC
            call add_main_library_templates(ctx, pset, w_main_all, spec_grid, dusty)
#else
            call add_main_library_templates(ctx, pset, w_main_all, spec_grid)
#endif
            deallocate(w_main_all)
        end if

        ! Release buffer memory from device
        !$acc exit data delete(buf%initial_mass, buf%current_mass, buf%log_lum, buf%log_teff)
        !$acc exit data delete(buf%log_g, buf%phase, buf%co_ratio, buf%log_mdot, buf%weights)
        !$acc exit data delete(buf)

        call free_isochrone_buffer(buf)

        ! --------------------------------------------------------------------
        ! 5. POST-PROCESSING
        ! --------------------------------------------------------------------

        call finish_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)

        ! Outputs are final on the host; refresh device copies the caller may have mapped
        !$acc update device(spec_grid, mass_grid, lbol_grid) if_present

    end subroutine generate_ssp_grid

    !> @brief Post-processing of the SSP grid: time interpolation (TIME_RES_INCR > 1), nebular
    !> emission (add_neb_emission = 2), X-ray binaries and instrumental smoothing. The last
    !> step of generate_ssp_grid, shared with the batched SSP operator (fsps_ssp_operator).
    subroutine finish_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)
        type(fsps_context_t), target, intent(inout) :: ctx
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
            ! Note: Calls into fsps_stellar_modifications
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
        
        ! Allocate to maximum size NM defined in fsps_constants.
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
        type(fsps_context_t), intent(in) :: ctx
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
        type(fsps_context_t), intent(inout) :: ctx
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
    !> operator (fsps_ssp_operator) uses: with unit weights it reads off each star's weight
    !> factor. The optional outputs are its bookkeeping: hb_flag(j) for the stars summed into
    !> the HB weight (whose sum sets the blue-straggler weights), parent(i) = j for blue HB
    !> stars i split from star j, n_after_hb = number of stars before the blue stragglers.
    subroutine apply_isochrone_modifications(ctx, pset, time_log_yr, buf, hb_flag, parent, n_after_hb)
        type(fsps_context_t), intent(inout) :: ctx
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
        type(fsps_context_t), intent(inout) :: ctx
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
        ! On the device only for the per-star kernel (buffer already there): otherwise a few
        ! thousand terms, cheaper on the host than the uploads and launches (56 ms per SSP grid)
        !$acc parallel loop reduction(+:local_mass) present(buf) if(ssp_per_star)
        do i = 1, n
            local_mass = local_mass + buf%weights(i) * buf%current_mass(i)
        end do
        
        ! 2. Add Remnant Mass (Black Holes, Neutron Stars, White Dwarfs)
        if (ctx%add_stellar_remnants_val == 1) then
             
             ! The turn-off mass is approximately the maximum initial mass
             ! of stars still present in the native isochrone.
             max_living_mass = -1.0_wp
             !$acc parallel loop reduction(max:max_living_mass) present(buf) if(ssp_per_star)
             do i = 1, n_native
                 if (buf%initial_mass(i) > max_living_mass) max_living_mass = buf%initial_mass(i)
             end do
             
             ! This routine integrates the IMF for dead stars and adds to tot_mass
             ! This routine needs to be device-compatible or run on host with scalar update?
             ! add_remnant_mass uses simple math. Assuming it's routine seq.
               call add_remnant_mass(ctx, local_mass, max_living_mass)
        end if

           tot_mass = local_mass

        ! 3. Calculate Total Bolometric Luminosity
        !    L_tot = sum( weight * 10^logL )
        !    We compute the linear sum first, then take log10.
        local_linear_lum = 0.0_wp
        !$acc parallel loop reduction(+:local_linear_lum) present(buf) if(ssp_per_star)
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
    subroutine accumulate_spectrum(ctx, pset, buf, spec_out, w_main_out, dusty)
        type(fsps_context_t), target, intent(inout) :: ctx
        type(params), intent(in)            :: pset
        type(isochrone_buffer_t), intent(in):: buf
        real(WP), intent(inout), contiguous :: spec_out(:)
        !> If present (template path), the main-library template weights are returned here instead
        !> of summed, and spec_out holds the rest, unfloored (see add_main_library_templates)
        real(WP), intent(inout), optional   :: w_main_out(:)
        !> If present (OpenACC template path), dust-shell AGB corners are appended here
        type(dusty_defer_t), intent(inout), optional :: dusty

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
                !$acc exit data delete(ctx%state%ssp_temp_grid)
                deallocate(ctx%state%ssp_temp_grid)
            end if
            allocate(ctx%state%ssp_temp_grid(nspec, nstars))
            !$acc enter data create(ctx%state%ssp_temp_grid)
            !$acc enter data attach(ctx%state%ssp_temp_grid)
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

        ! Template-sum form (on the host; the main library is added on the device in OpenACC
        ! builds, see add_main_library_templates); the per-star loop below is the reference
        if (n_active > 0 .and. .not. ssp_per_star) then
            call accumulate_by_templates(ctx, pset, buf, active_idx(1:n_active), spec_out, w_main_out, dusty)
            deallocate(active_idx, active_w)
            return
        end if

        !$acc data create(active_idx, active_w) pcopy(spec_out)
        if (n_active == 0) then
            spec_out = 0.0_wp
        else
            !$acc update device(active_idx, active_w)

            ! Initialize output
            !$acc kernels present(spec_out)
            spec_out = 0.0_wp
            !$acc end kernels

            ! ----------------------------------------------------------------
            ! PHASE 1: PARALLEL GENERATION (Gang over Stars)
            ! ----------------------------------------------------------------
            !$acc parallel loop gang vector collapse(1) present(ctx, buf)
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
#ifdef _OPENACC
            ! One thread per pixel, summing over the stars
            if (KP == WP) then
                !$acc parallel loop gang vector private(ia, acc) present(ctx, spec_out)
                do i_spec = 1, nspec
                    acc = 0.0_KP
                    do ia = 1, n_active
                        acc = acc + ctx%state%ssp_temp_grid(i_spec, ia) * active_w(ia)
                    end do
                    spec_out(i_spec) = max(real(acc, WP), SAFE_FLOOR)
                end do
            else
                !$acc parallel loop gang vector private(ia, acc, comp, term, tsum) present(ctx, spec_out)
                do i_spec = 1, nspec
                    acc = 0.0_KP
                    comp = 0.0_KP
                    do ia = 1, n_active
                        term = ctx%state%ssp_temp_grid(i_spec, ia) * active_w(ia) - comp
                        tsum = acc + term
                        comp = (tsum - acc) - term
                        acc = tsum
                    end do
                    spec_out(i_spec) = max(real(acc, WP), SAFE_FLOOR)
                end do
            end if
#else
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
#endif
        end if

        !$acc end data
        deallocate(active_idx, active_w)

    end subroutine accumulate_spectrum

    !> @brief SSP spectrum of one isochrone as a sum over library templates.
    !>
    !> @details
    !> Every star's spectrum is a linear combination of at most four templates of one library
    !> (get_stellar_terms), so the IMF-weighted sum over stars equals the sum over templates of
    !> (total weight of the template) x (template). The weights are aggregated in WP; the
    !> template sum touches each used template once, instead of up to four times per star.
    !> Stars that are not such a combination (AGB stars with a dust shell, degenerate cases) are
    !> added from get_stellar_spectrum.
    subroutine accumulate_by_templates(ctx, pset, buf, idx, spec_out, w_main_out, ddefer)
        type(fsps_context_t), target, intent(inout) :: ctx
        type(params), intent(in)                    :: pset
        type(isochrone_buffer_t), intent(in)        :: buf
        integer, intent(in)                         :: idx(:)
        real(WP), intent(inout), contiguous         :: spec_out(:)
        real(WP), intent(inout), optional           :: w_main_out(:)
        type(dusty_defer_t), intent(inout), optional :: ddefer

        real(WP), allocatable :: w_main(:,:), w_wmb(:,:), w_pagb(:,:), w_wrn(:), w_wrc(:), &
                                 w_agbo(:), w_agbc(:), w_car(:)
        real(KP), allocatable :: acc(:), comp(:), star(:)
        integer  :: ia, j, n, nspec, nz, src, nterm, i1(4), i2(4), a, b
        real(WP) :: coef(4), w
        logical  :: linear, dusty
        ! Dust-shell AGB stars: (template sum) x (dust transfer sum) is bilinear, so their template
        ! weights are aggregated per dust-grid corner (key: C-rich flag, Teff index, tau index)
        integer,  allocatable :: dkey(:,:)
        real(WP), allocatable :: dw_main(:,:,:), dw_o(:,:), dw_c(:,:), dw_car(:,:)
        integer  :: nslot, slot, d, rich, it, itau, cdi(4), cdj(4)
        real(WP) :: cd(4)
        logical  :: active, smooth, defer_small

        nspec = size(spec_out)
        nz = pset%zmet
        allocate(acc(nspec), comp(nspec), star(nspec))
        acc = 0.0_KP; comp = 0.0_KP
        allocate(w_main(size(ctx%state%speclib, 3), size(ctx%state%speclib, 4)))
        allocate(w_wmb(size(ctx%state%wmb_spec, 3), size(ctx%state%wmb_spec, 4)))
        allocate(w_pagb(size(ctx%state%pagb_spec, 2), size(ctx%state%pagb_spec, 3)))
        allocate(w_wrn(size(ctx%state%wrn_spec, 2)), w_wrc(size(ctx%state%wrc_spec, 2)))
        allocate(w_agbo(size(ctx%state%agb_spec_o, 2)), w_agbc(size(ctx%state%agb_spec_c, 2)), &
                 w_car(size(ctx%state%agb_spec_car, 2)))
        w_main = 0.0_wp; w_wmb = 0.0_wp; w_pagb = 0.0_wp; w_wrn = 0.0_wp; w_wrc = 0.0_wp
        w_agbo = 0.0_wp; w_agbc = 0.0_wp; w_car = 0.0_wp
        nslot = 0
        cdi = [0, 1, 1, 0]; cdj = [0, 0, 1, 1]   ! dust corners, in the order of get_agb_dust_terms

        ! 1. Aggregate template weights (stars that are not a template combination go direct)
        do ia = 1, size(idx)
            j = idx(ia)
            w = buf%weights(j)
            call get_stellar_terms(ctx, pset, buf%current_mass(j), buf%log_teff(j), &
                                   exp(buf%log_lum(j) * LN10), buf%log_g(j), buf%phase(j), &
                                   buf%co_ratio(j), buf%log_mdot(j), src, i1, i2, coef, nterm, linear, &
                                   allow_dust=associated(ctx%state%speclib_agb_smooth), dusty=dusty)
            if (linear .and. dusty) then
                call get_agb_dust_terms(ctx, pset%agb_dust, buf%current_mass(j), buf%log_teff(j), &
                                        buf%log_lum(j), buf%log_g(j), buf%co_ratio(j), buf%log_mdot(j), &
                                        rich, it, itau, cd, active)
                if (active .and. (src == SRC_MAIN_LIB .or. src == SRC_AGB_O .or. src == SRC_AGB_C .or. &
                                  src == SRC_AGB_CAR)) then
                    do d = 1, 4
                        if (cd(d) == 0.0_wp) cycle
                        slot = find_slot(rich, it + cdi(d), itau + cdj(d))
                        do n = 1, nterm
                            select case (src)
                            case (SRC_MAIN_LIB)
                                dw_main(i1(n), i2(n), slot) = dw_main(i1(n), i2(n), slot) + w * coef(n) * cd(d)
                            case (SRC_AGB_O)
                                dw_o(i1(n), slot) = dw_o(i1(n), slot) + w * coef(n) * cd(d)
                            case (SRC_AGB_C)
                                dw_c(i1(n), slot) = dw_c(i1(n), slot) + w * coef(n) * cd(d)
                            case (SRC_AGB_CAR)
                                dw_car(i1(n), slot) = dw_car(i1(n), slot) + w * coef(n) * cd(d)
                            end select
                        end do
                    end do
                    cycle
                else if (active) then
                    linear = .false.   ! not expected (AGB stars use these libraries); per star
                end if
                ! no optical depth: no shell and no smoothed templates, a plain linear star
            end if
            if (buf%log_lum(j) * LN10 < log(tiny(0.0_wp))) linear = .false.   ! zero luminosity: floored
            if (.not. linear) then
                call get_stellar_spectrum(ctx, pset, buf%current_mass(j), buf%log_teff(j), &
                                          exp(buf%log_lum(j) * LN10), buf%log_g(j), buf%phase(j), &
                                          buf%co_ratio(j), buf%log_mdot(j), star)
                call add_column(acc, comp, real(w, KP), star)
                cycle
            end if
            do n = 1, nterm
                select case (src)
                case (SRC_MAIN_LIB); w_main(i1(n), i2(n)) = w_main(i1(n), i2(n)) + w * coef(n)
                case (SRC_WMBASIC);  w_wmb(i1(n), i2(n))  = w_wmb(i1(n), i2(n))  + w * coef(n)
                case (SRC_PAGB);     w_pagb(i1(n), i2(n)) = w_pagb(i1(n), i2(n)) + w * coef(n)
                case (SRC_WRN);      w_wrn(i1(n))  = w_wrn(i1(n))  + w * coef(n)
                case (SRC_WRC);      w_wrc(i1(n))  = w_wrc(i1(n))  + w * coef(n)
                case (SRC_AGB_O);    w_agbo(i1(n)) = w_agbo(i1(n)) + w * coef(n)
                case (SRC_AGB_C);    w_agbc(i1(n)) = w_agbc(i1(n)) + w * coef(n)
                case (SRC_AGB_CAR);  w_car(i1(n))  = w_car(i1(n))  + w * coef(n)
                end select
            end do
        end do

        ! 2. Sum the templates with non-zero weight (the main library after the age loop, if batched;
        !    with OpenACC the other libraries too, in the rows after the main library's)
        defer_small = .false.
        if (present(w_main_out)) defer_small = size(w_main_out) > size(w_main)
        if (present(w_main_out)) then
            n = size(w_main)
            w_main_out(1:n) = w_main_out(1:n) + reshape(w_main, [n])
            if (defer_small) then
                call add_rows(reshape(w_wmb, [size(w_wmb)]))
                call add_rows(reshape(w_pagb, [size(w_pagb)]))
                call add_rows(w_wrn)
                call add_rows(w_wrc)
                call add_rows(w_car)
                call add_rows(w_agbo)
                call add_rows(w_agbc)
            end if
        else
            do b = 1, size(w_main, 2)
                do a = 1, size(w_main, 1)
                    if (w_main(a, b) /= 0.0_wp) &
                        call add_column_sp(acc, comp, real(w_main(a, b), KP), ctx%state%speclib(1:nspec, nz, a, b))
                end do
            end do
        end if
        if (.not. defer_small) call sum_small_templates(ctx, nz, w_wmb, w_pagb, w_wrn, w_wrc, w_car, w_agbo, &
                                                        w_agbc, acc, comp)

        ! 3. Dust-shell AGB stars: per dust corner, the template sum (smoothed beyond 3 um, as in
        !    get_stellar_spectrum) times that corner's transfer function
        if (nslot > 0 .and. present(ddefer)) then
            do slot = 1, nslot
                call defer_slot(slot)
            end do
        else if (nslot > 0) then
            call sum_dusty_slots(ctx, nz, nslot, dkey, dw_main, dw_o, dw_c, dw_car, acc, comp)
        end if

        if (present(w_main_out)) then
            spec_out = real(acc, WP)   ! floored after the main-library templates are added
        else
            spec_out = max(real(acc, WP), SAFE_FLOOR)
        end if

    contains

        !> Appends dust corner slot (of age dusty%cur_age) and its non-zero template weights.
        subroutine defer_slot(sl)
            integer, intent(in) :: sl
            integer :: aa, bb
            call dusty_push_slot(ddefer, ddefer%cur_age, [dkey(1, sl) + 1, dkey(2, sl), dkey(3, sl)])
            do bb = 1, size(dw_main, 2)
                do aa = 1, size(dw_main, 1)
                    if (dw_main(aa, bb, sl) /= 0.0_wp) call dusty_push_entry(ddefer, L_MAIN, aa, bb, dw_main(aa, bb, sl))
                end do
            end do
            do aa = 1, size(dw_car, 1)
                if (dw_car(aa, sl) /= 0.0_wp) call dusty_push_entry(ddefer, L_CAR, aa, 1, dw_car(aa, sl))
            end do
            do aa = 1, size(dw_o, 1)
                if (dw_o(aa, sl) /= 0.0_wp) call dusty_push_entry(ddefer, L_AGBO, aa, 1, dw_o(aa, sl))
            end do
            do aa = 1, size(dw_c, 1)
                if (dw_c(aa, sl) /= 0.0_wp) call dusty_push_entry(ddefer, L_AGBC, aa, 1, dw_c(aa, sl))
            end do
        end subroutine defer_slot

        !> Appends a library's template weights to the deferred rows (n: rows filled so far).
        subroutine add_rows(wl)
            real(WP), intent(in) :: wl(:)
            w_main_out(n+1:n+size(wl)) = w_main_out(n+1:n+size(wl)) + wl
            n = n + size(wl)
        end subroutine add_rows

        !> Slot of a dust-grid corner, created (and the slot arrays grown) on first use.
        integer function find_slot(r, kt, ktau)
            integer, intent(in) :: r, kt, ktau
            integer :: k, cap
            integer,  allocatable :: k2(:,:)
            real(WP), allocatable :: m2(:,:,:), o2(:,:), c2(:,:), car2(:,:)
            do k = 1, nslot
                if (dkey(1, k) == r .and. dkey(2, k) == kt .and. dkey(3, k) == ktau) then
                    find_slot = k
                    return
                end if
            end do
            if (.not. allocated(dkey)) then
                cap = 16
                allocate(dkey(3, cap), dw_main(size(w_main, 1), size(w_main, 2), cap), &
                         dw_o(size(w_agbo), cap), dw_c(size(w_agbc), cap), dw_car(size(w_car), cap))
                dw_main = 0.0_wp; dw_o = 0.0_wp; dw_c = 0.0_wp; dw_car = 0.0_wp
            else if (nslot == size(dkey, 2)) then
                cap = 2 * nslot
                allocate(k2(3, cap), m2(size(w_main, 1), size(w_main, 2), cap), o2(size(w_agbo), cap), &
                         c2(size(w_agbc), cap), car2(size(w_car), cap))
                m2 = 0.0_wp; o2 = 0.0_wp; c2 = 0.0_wp; car2 = 0.0_wp
                k2(:, 1:nslot) = dkey(:, 1:nslot)
                m2(:, :, 1:nslot) = dw_main(:, :, 1:nslot)
                o2(:, 1:nslot) = dw_o(:, 1:nslot); c2(:, 1:nslot) = dw_c(:, 1:nslot)
                car2(:, 1:nslot) = dw_car(:, 1:nslot)
                call move_alloc(k2, dkey); call move_alloc(m2, dw_main); call move_alloc(o2, dw_o)
                call move_alloc(c2, dw_c); call move_alloc(car2, dw_car)
            end if
            nslot = nslot + 1
            dkey(:, nslot) = [r, kt, ktau]
            find_slot = nslot
        end function find_slot

    end subroutine accumulate_by_templates

    !> Adds the templates of the libraries other than the main one (WMBasic, post-AGB, WN, WC,
    !> carbon-star, O- and C-rich AGB) with weights w_*, to acc (compensated with comp at FP32).
    !> Step 2 of accumulate_by_templates, also used by the batched SSP operator.
    subroutine sum_small_templates(ctx, nz, w_wmb, w_pagb, w_wrn, w_wrc, w_car, w_agbo, w_agbc, acc, comp)
        type(fsps_context_t), intent(in) :: ctx
        integer, intent(in)  :: nz
        real(WP), intent(in) :: w_wmb(:,:), w_pagb(:,:), w_wrn(:), w_wrc(:), w_car(:), w_agbo(:), w_agbc(:)
        real(KP), intent(inout) :: acc(:), comp(:)
        real(KP), allocatable :: lam_sum(:), lam_comp(:)
        integer :: a, b, nspec

        nspec = size(acc)
        do b = 1, size(w_wmb, 2)
            do a = 1, size(w_wmb, 1)
                if (w_wmb(a, b) /= 0.0_wp) &
                    call add_column_sp(acc, comp, real(w_wmb(a, b), KP), ctx%state%wmb_spec(1:nspec, nz, a, b))
            end do
        end do
        do b = 1, size(w_pagb, 2)
            do a = 1, size(w_pagb, 1)
                if (w_pagb(a, b) /= 0.0_wp) &
                    call add_column(acc, comp, real(w_pagb(a, b), KP), ctx%state%pagb_spec(1:nspec, a, b))
            end do
        end do
        do a = 1, size(w_wrn)
            if (w_wrn(a) /= 0.0_wp) call add_column(acc, comp, real(w_wrn(a), KP), ctx%state%wrn_spec(1:nspec, a, nz))
        end do
        do a = 1, size(w_wrc)
            if (w_wrc(a) /= 0.0_wp) call add_column(acc, comp, real(w_wrc(a), KP), ctx%state%wrc_spec(1:nspec, a, nz))
        end do
        do a = 1, size(w_car)
            if (w_car(a) /= 0.0_wp) call add_column(acc, comp, real(w_car(a), KP), ctx%state%agb_spec_car(1:nspec, a))
        end do
        ! O- and C-rich AGB templates carry lambda^2 (per unit frequency), applied once per library.
        ! Before lambda^2 (up to 7e14 A^2 in the far IR) the weighted sum is ~1e-37 there, at the
        ! bottom of the FP32 range, where subnormal handling (kept by gfortran, flushed by
        ! nvfortran and GPUs) changed the result by 1%: the sum carries an exact factor AGB_SCALE
        if (any(w_agbo /= 0.0_wp) .or. any(w_agbc /= 0.0_wp)) then
            allocate(lam_sum(nspec), lam_comp(nspec))
            lam_sum = 0.0_KP; lam_comp = 0.0_KP
            do a = 1, size(w_agbo)
                if (w_agbo(a) /= 0.0_wp) call add_column(lam_sum, lam_comp, real(w_agbo(a) * AGB_SCALE, KP), &
                                                         ctx%state%agb_spec_o(1:nspec, a))
            end do
            do a = 1, size(w_agbc)
                if (w_agbc(a) /= 0.0_wp) call add_column(lam_sum, lam_comp, real(w_agbc(a) * AGB_SCALE, KP), &
                                                         ctx%state%agb_spec_c(1:nspec, a))
            end do
            call add_column(acc, comp, real(1.0_wp / AGB_SCALE, KP), &
                            real(ctx%state%spec_lambda(1:nspec), KP)**2 * lam_sum)
        end if
    end subroutine sum_small_templates

    !> Adds the dust-shell AGB corners 1..nslot (key dkey(:, slot): C-rich flag, Teff and tau
    !> indices of flux_dagb; template weights dw_*(:, slot)): per corner the template sum (smoothed
    !> beyond 3 um) times its transfer function. Step 3 of accumulate_by_templates, also used by
    !> the batched SSP operator.
    subroutine sum_dusty_slots(ctx, nz, nslot, dkey, dw_main, dw_o, dw_c, dw_car, acc, comp)
        type(fsps_context_t), intent(in) :: ctx
        integer, intent(in)  :: nz, nslot, dkey(:,:)
        real(WP), intent(in) :: dw_main(:,:,:), dw_o(:,:), dw_c(:,:), dw_car(:,:)
        real(KP), intent(inout) :: acc(:), comp(:)
        real(KP), allocatable :: tmp(:), tmp_comp(:), low(:), low_comp(:)
        integer :: a, b, slot, i0, nspec

        nspec = size(acc)
        i0 = nspec - size(ctx%state%speclib_agb_smooth, 1) + 1
        allocate(tmp(nspec), tmp_comp(nspec), low(nspec), low_comp(nspec))
        do slot = 1, nslot
            tmp = 0.0_KP; tmp_comp = 0.0_KP; low = 0.0_KP; low_comp = 0.0_KP
            do b = 1, size(dw_main, 2)
                do a = 1, size(dw_main, 1)
                    if (dw_main(a, b, slot) == 0.0_wp) cycle
                    call add_column_sp(tmp(1:i0-1), tmp_comp(1:i0-1), real(dw_main(a, b, slot), KP), &
                                       ctx%state%speclib(1:i0-1, nz, a, b))
                    call add_column(tmp(i0:), tmp_comp(i0:), real(dw_main(a, b, slot), KP), &
                                    ctx%state%speclib_agb_smooth(:, nz, a, b))
                end do
            end do
            do a = 1, size(dw_car, 1)
                if (dw_car(a, slot) == 0.0_wp) cycle
                call add_column(tmp(1:i0-1), tmp_comp(1:i0-1), real(dw_car(a, slot), KP), &
                                ctx%state%agb_spec_car(1:i0-1, a))
                call add_column(tmp(i0:), tmp_comp(i0:), real(dw_car(a, slot), KP), &
                                ctx%state%agb_spec_car_smooth(:, a))
            end do
            ! O/C-rich: lambda^2 / c below 3 um; the smoothed templates already include it
            ! (low carries the factor AGB_SCALE, as lam_sum above)
            do a = 1, size(dw_o, 1)
                if (dw_o(a, slot) == 0.0_wp) cycle
                call add_column(low(1:i0-1), low_comp(1:i0-1), real(dw_o(a, slot) * AGB_SCALE, KP), &
                                ctx%state%agb_spec_o(1:i0-1, a))
                call add_column(tmp(i0:), tmp_comp(i0:), real(dw_o(a, slot) * C_LIGHT, KP), &
                                ctx%state%agb_spec_o_smooth(:, a))
            end do
            do a = 1, size(dw_c, 1)
                if (dw_c(a, slot) == 0.0_wp) cycle
                call add_column(low(1:i0-1), low_comp(1:i0-1), real(dw_c(a, slot) * AGB_SCALE, KP), &
                                ctx%state%agb_spec_c(1:i0-1, a))
                call add_column(tmp(i0:), tmp_comp(i0:), real(dw_c(a, slot) * C_LIGHT, KP), &
                                ctx%state%agb_spec_c_smooth(:, a))
            end do
            tmp(1:i0-1) = tmp(1:i0-1) + real(ctx%state%spec_lambda(1:i0-1), KP)**2 * low(1:i0-1) * &
                          real(1.0_wp / AGB_SCALE, KP)
            call add_column(acc, comp, 1.0_KP, tmp * ctx%state%flux_dagb(1:nspec, dkey(1, slot) + 1, &
                                                                        dkey(2, slot), dkey(3, slot)))
        end do
    end subroutine sum_dusty_slots

    !> acc += a * x, compensated when the kernels are narrower than WP (templates stored in FP32).
    subroutine add_column_sp(acc, comp, a, x)
        real(KP), intent(inout) :: acc(:), comp(:)
        real(KP), intent(in)    :: a
        real(kind(1.0)), intent(in) :: x(:)
        integer  :: i
        real(KP) :: term, tsum
        if (KP == WP) then
            do i = 1, size(acc)
                acc(i) = acc(i) + a * x(i)
            end do
        else
            do i = 1, size(acc)
                term = a * x(i) - comp(i)
                tsum = acc(i) + term
                comp(i) = (tsum - acc(i)) - term
                acc(i) = tsum
            end do
        end if
    end subroutine add_column_sp

    !> acc += a * x, compensated when the kernels are narrower than WP (kernel-precision arrays).
    subroutine add_column(acc, comp, a, x)
        real(KP), intent(inout) :: acc(:), comp(:)
        real(KP), intent(in)    :: a
        real(KP), intent(in) :: x(:)
        integer  :: i
        real(KP) :: term, tsum
        if (KP == WP) then
            do i = 1, size(acc)
                acc(i) = acc(i) + a * x(i)
            end do
        else
            do i = 1, size(acc)
                term = a * x(i) - comp(i)
                tsum = acc(i) + term
                comp(i) = (tsum - acc(i)) - term
                acc(i) = tsum
            end do
        end if
    end subroutine add_column


    !> Starts a new deferred dust corner (slot) of age t with transfer-function indices key.
    subroutine dusty_push_slot(d, t, key)
        type(dusty_defer_t), intent(inout) :: d
        integer, intent(in) :: t, key(3)
        integer, allocatable :: ia(:), ik(:,:)
        integer :: cap

        if (.not. allocated(d%slot_age)) then
            allocate(d%slot_age(64), d%slot_key(3, 64), d%slot_ptr(65))
            d%slot_ptr(1) = 1
        else if (d%n_slots == size(d%slot_age)) then
            cap = 2 * size(d%slot_age)
            allocate(ia(cap)); ia(1:d%n_slots) = d%slot_age(1:d%n_slots); call move_alloc(ia, d%slot_age)
            allocate(ik(3, cap)); ik(:, 1:d%n_slots) = d%slot_key(:, 1:d%n_slots); call move_alloc(ik, d%slot_key)
            allocate(ia(cap + 1)); ia(1:d%n_slots + 1) = d%slot_ptr(1:d%n_slots + 1); call move_alloc(ia, d%slot_ptr)
        end if
        d%n_slots = d%n_slots + 1
        d%slot_age(d%n_slots) = t
        d%slot_key(:, d%n_slots) = key
        d%slot_ptr(d%n_slots + 1) = d%n_ent + 1
    end subroutine dusty_push_slot

    !> Appends a template weight to the current deferred dust corner.
    subroutine dusty_push_entry(d, lib, a, b, w)
        type(dusty_defer_t), intent(inout) :: d
        integer, intent(in)  :: lib, a, b
        real(WP), intent(in) :: w
        integer, allocatable  :: ia(:)
        real(WP), allocatable :: wa(:)
        integer :: cap

        if (.not. allocated(d%ent_lib)) then
            allocate(d%ent_lib(1024), d%ent_a(1024), d%ent_b(1024), d%ent_w(1024))
        else if (d%n_ent == size(d%ent_lib)) then
            cap = 2 * size(d%ent_lib)
            allocate(ia(cap)); ia(1:d%n_ent) = d%ent_lib(1:d%n_ent); call move_alloc(ia, d%ent_lib)
            allocate(ia(cap)); ia(1:d%n_ent) = d%ent_a(1:d%n_ent); call move_alloc(ia, d%ent_a)
            allocate(ia(cap)); ia(1:d%n_ent) = d%ent_b(1:d%n_ent); call move_alloc(ia, d%ent_b)
            allocate(wa(cap)); wa(1:d%n_ent) = d%ent_w(1:d%n_ent); call move_alloc(wa, d%ent_w)
        end if
        d%n_ent = d%n_ent + 1
        d%ent_lib(d%n_ent) = lib
        d%ent_a(d%n_ent) = a
        d%ent_b(d%n_ent) = b
        d%ent_w(d%n_ent) = w
        d%slot_ptr(d%n_slots + 1) = d%n_ent + 1
    end subroutine dusty_push_entry

    !> Number of templates of the libraries other than the main one (deferred rows, OpenACC).
    integer function n_small_templates(ctx)
        type(fsps_context_t), intent(in) :: ctx
        n_small_templates = size(ctx%state%wmb_spec, 3) * size(ctx%state%wmb_spec, 4) + &
                            size(ctx%state%pagb_spec, 2) * size(ctx%state%pagb_spec, 3) + &
                            size(ctx%state%wrn_spec, 2) + size(ctx%state%wrc_spec, 2) + &
                            size(ctx%state%agb_spec_car, 2) + size(ctx%state%agb_spec_o, 2) + &
                            size(ctx%state%agb_spec_c, 2)
    end function n_small_templates

    !> Library and template indices of deferred row r (1-based, after the main library), in the
    !> order of accumulate_by_templates: WMBasic (a, b), post-AGB (a, b), WN, WC, C-rich AGB
    !> (carbon stars), O-rich and C-rich AGB.
    subroutine small_template_index(ctx, r, lib, a, b)
        type(fsps_context_t), intent(in) :: ctx
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
    subroutine fsps_ssp_set_per_star(flag)
        logical, intent(in) :: flag
        ssp_per_star = flag
    end subroutine fsps_ssp_set_per_star

    !> @brief Adds the main-library templates of all ages: spec_grid += templates . W.
    !>
    !> @details
    !> One matrix product (pixels x templates) . (templates x ages) instead of one sum per age,
    !> blocked over pixels so that a block of each template and the accumulators of all ages stay
    !> in cache: each template is read once per SSP grid. Compensated at FP32 kernel precision.
    !> Ages with no stars are left at zero; computed ages are floored as in the per-age path.
    subroutine add_main_library_templates(ctx, pset, w, spec_grid, dusty)
        type(fsps_context_t), intent(in)    :: ctx
        type(params), intent(in)            :: pset
        real(WP), intent(in)                :: w(:,:)          ! (templates, ages)
        real(WP), intent(inout), contiguous :: spec_grid(:,:)  ! (pixels, output ages)
        !> Deferred dust-shell AGB corners (OpenACC template path)
        type(dusty_defer_t), intent(in), optional :: dusty
        integer, parameter :: PIX_BLOCK = 512
        real(KP), allocatable :: acc(:,:), comp(:,:)
        integer, allocatable :: used(:)
        logical, allocatable :: has_stars(:)
        integer :: nspec, nt_lib, n_used, t, a, b, i_time, out_idx, p0, p1, i, k
        real(KP) :: wk, term, tsum
#ifdef _OPENACC
        real(KP), allocatable :: w_used(:,:), t_sum(:,:)
        integer, allocatable :: a_used(:), b_used(:), lib_used(:)
        real(KP) :: acc1, comp1, lacc, lcomp, x, inv_scale, ta, tc, la, lc, tmpv
        integer :: zmet, n_ages, n_main, r, lib, n2, nsl, sl, e, j, i0
        logical :: any_agb, low
        real(WP) :: wv
        integer, allocatable  :: age_first(:), age_last(:), sl_key(:,:), sl_ptr(:), e_lib(:), e_a(:), e_b(:)
        real(WP), allocatable :: e_w(:)
#endif

        nspec = size(spec_grid, 1)
        nt_lib = size(ctx%state%speclib, 3)
        used = pack([(t, t = 1, size(w, 1))], any(w /= 0.0_wp, dim=2))
        n_used = size(used)
        allocate(acc(PIX_BLOCK, size(w, 2)), comp(PIX_BLOCK, size(w, 2)), has_stars(size(w, 2)))
        ! Ages without stars stay at zero, as in the per-age path
        do i_time = 1, size(w, 2)
            out_idx = 1 + (i_time - 1) * TIME_RES_INCR
            has_stars(i_time) = pset%ssp_gen_age(i_time) /= 0 .and. &
                                (any(w(:, i_time) /= 0.0_wp) .or. any(spec_grid(:, out_idx) /= 0.0_wp))
        end do
#ifdef _OPENACC
        ! Device: one thread per (pixel, age) sums the used templates in the same order as the
        ! host loop below (compensated at FP32), reading the resident library. The sums come
        ! back in t_sum and are added on the host (spec_grid may be mapped by the caller).
        n_ages = size(w, 2)
        zmet = pset%zmet
        ! Rows after the main library's: the other libraries (accumulate_by_templates order)
        n_main = nt_lib * size(ctx%state%speclib, 4)
        allocate(w_used(n_used, n_ages), a_used(n_used), b_used(n_used), lib_used(n_used), t_sum(nspec, n_ages))
        do k = 1, n_used
            r = used(k)
            if (r <= n_main) then
                lib_used(k) = L_MAIN
                a_used(k) = mod(r - 1, nt_lib) + 1
                b_used(k) = (r - 1) / nt_lib + 1
            else
                call small_template_index(ctx, r - n_main, lib_used(k), a_used(k), b_used(k))
            end if
            if (lib_used(k) == L_AGBO .or. lib_used(k) == L_AGBC) then
                w_used(k, :) = real(w(r, :) * AGB_SCALE, KP)   ! as lam_sum in accumulate_by_templates
            else
                w_used(k, :) = real(w(r, :), KP)
            end if
        end do
        any_agb = any(lib_used == L_AGBO .or. lib_used == L_AGBC)
        inv_scale = real(1.0_wp / AGB_SCALE, KP)

        ! Dust-shell AGB corners: the slots of age t are age_first(t) .. age_last(t) (appended in
        ! age order); empty placeholders keep the data clauses valid without any
        nsl = 0
        if (present(dusty)) nsl = dusty%n_slots
        allocate(age_first(n_ages), age_last(n_ages))
        age_first = 1
        age_last = 0
        i0 = nspec + 1
        if (nsl > 0) then
            ! Explicit allocations (nvfortran: no reallocation on assignment by default)
            allocate(sl_key(3, nsl), sl_ptr(nsl + 1), e_lib(max(dusty%n_ent, 1)), e_a(max(dusty%n_ent, 1)), &
                     e_b(max(dusty%n_ent, 1)), e_w(max(dusty%n_ent, 1)))
            sl_key = dusty%slot_key(:, 1:nsl)
            sl_ptr = dusty%slot_ptr(1:nsl + 1)
            e_lib(1:dusty%n_ent) = dusty%ent_lib(1:dusty%n_ent)
            e_a(1:dusty%n_ent) = dusty%ent_a(1:dusty%n_ent)
            e_b(1:dusty%n_ent) = dusty%ent_b(1:dusty%n_ent)
            e_w(1:dusty%n_ent) = dusty%ent_w(1:dusty%n_ent)
            do sl = nsl, 1, -1
                age_first(dusty%slot_age(sl)) = sl
            end do
            do sl = 1, nsl
                age_last(dusty%slot_age(sl)) = sl
                has_stars(dusty%slot_age(sl)) = pset%ssp_gen_age(dusty%slot_age(sl)) /= 0
            end do
            i0 = nspec - size(ctx%state%speclib_agb_smooth, 1) + 1
        else
            allocate(sl_key(3, 1), sl_ptr(2), e_lib(1), e_a(1), e_b(1), e_w(1))
            sl_key = 1; sl_ptr = 1; e_lib = 0; e_a = 1; e_b = 1; e_w = 0.0_wp
        end if
        !$acc parallel loop gang vector collapse(2) present(ctx) &
        !$acc private(acc1, comp1, lacc, lcomp, x, lib, wk, term, tsum, k, ta, tc, la, lc, tmpv, sl, e, j, wv, low) &
        !$acc copyin(w_used, a_used, b_used, lib_used, has_stars, age_first, age_last, sl_key, sl_ptr, e_lib, e_a, &
        !$acc        e_b, e_w) copyout(t_sum)
        do i_time = 1, n_ages
            do i = 1, nspec
                if (has_stars(i_time)) then
                    acc1 = 0.0_KP
                    comp1 = 0.0_KP
                    lacc = 0.0_KP
                    lcomp = 0.0_KP
                    !$acc loop seq
                    do k = 1, n_used
                        wk = w_used(k, i_time)
                        if (wk == 0.0_KP) cycle
                        lib = lib_used(k)
                        select case (lib)
                        case (L_MAIN)
                            x = ctx%state%speclib(i, zmet, a_used(k), b_used(k))
                        case (L_WMB)
                            x = ctx%state%wmb_spec(i, zmet, a_used(k), b_used(k))
                        case (L_PAGB)
                            x = ctx%state%pagb_spec(i, a_used(k), b_used(k))
                        case (L_WRN)
                            x = ctx%state%wrn_spec(i, a_used(k), zmet)
                        case (L_WRC)
                            x = ctx%state%wrc_spec(i, a_used(k), zmet)
                        case (L_CAR)
                            x = ctx%state%agb_spec_car(i, a_used(k))
                        case (L_AGBO)
                            x = ctx%state%agb_spec_o(i, a_used(k))
                        case default
                            x = ctx%state%agb_spec_c(i, a_used(k))
                        end select
                        if (lib == L_AGBO .or. lib == L_AGBC) then
                            ! O/C-rich AGB: summed apart, times lambda^2 below
                            term = wk * x - lcomp
                            tsum = lacc + term
                            lcomp = (tsum - lacc) - term
                            lacc = tsum
                        else if (KP == WP) then
                            acc1 = acc1 + wk * x
                        else
                            term = wk * x - comp1
                            tsum = acc1 + term
                            comp1 = (tsum - acc1) - term
                            acc1 = tsum
                        end if
                    end do
                    if (any_agb) then
                        term = inv_scale * (real(ctx%state%spec_lambda(i), KP)**2 * lacc) - comp1
                        tsum = acc1 + term
                        comp1 = (tsum - acc1) - term
                        acc1 = tsum
                    end if
                    ! Dust-shell AGB corners of this age, as step 3 of accumulate_by_templates: the
                    ! template sum (smoothed beyond 3 um; O/C-rich below it times lambda^2, with the
                    ! factor AGB_SCALE) times the corner's transfer function
                    !$acc loop seq
                    do sl = age_first(i_time), age_last(i_time)
                        ta = 0.0_KP; tc = 0.0_KP; la = 0.0_KP; lc = 0.0_KP
                        !$acc loop seq
                        do e = sl_ptr(sl), sl_ptr(sl + 1) - 1
                            wv = e_w(e)
                            low = .false.
                            if (i < i0) then
                                select case (e_lib(e))
                                case (L_MAIN)
                                    x = ctx%state%speclib(i, zmet, e_a(e), e_b(e)); wk = real(wv, KP)
                                case (L_CAR)
                                    x = ctx%state%agb_spec_car(i, e_a(e)); wk = real(wv, KP)
                                case (L_AGBO)
                                    x = ctx%state%agb_spec_o(i, e_a(e)); wk = real(wv * AGB_SCALE, KP); low = .true.
                                case default
                                    x = ctx%state%agb_spec_c(i, e_a(e)); wk = real(wv * AGB_SCALE, KP); low = .true.
                                end select
                            else
                                j = i - i0 + 1
                                select case (e_lib(e))
                                case (L_MAIN)
                                    x = ctx%state%speclib_agb_smooth(j, zmet, e_a(e), e_b(e)); wk = real(wv, KP)
                                case (L_CAR)
                                    x = ctx%state%agb_spec_car_smooth(j, e_a(e)); wk = real(wv, KP)
                                case (L_AGBO)
                                    x = ctx%state%agb_spec_o_smooth(j, e_a(e)); wk = real(wv * C_LIGHT, KP)
                                case default
                                    x = ctx%state%agb_spec_c_smooth(j, e_a(e)); wk = real(wv * C_LIGHT, KP)
                                end select
                            end if
                            if (KP == WP) then
                                if (low) then
                                    la = la + wk * x
                                else
                                    ta = ta + wk * x
                                end if
                            else if (low) then
                                term = wk * x - lc
                                tsum = la + term
                                lc = (tsum - la) - term
                                la = tsum
                            else
                                term = wk * x - tc
                                tsum = ta + term
                                tc = (tsum - ta) - term
                                ta = tsum
                            end if
                        end do
                        if (i < i0) then
                            tmpv = ta + real(ctx%state%spec_lambda(i), KP)**2 * la * inv_scale
                        else
                            tmpv = ta
                        end if
                        if (KP == WP) then
                            acc1 = acc1 + tmpv * ctx%state%flux_dagb(i, sl_key(1, sl), sl_key(2, sl), sl_key(3, sl))
                        else
                            term = tmpv * ctx%state%flux_dagb(i, sl_key(1, sl), sl_key(2, sl), sl_key(3, sl)) - comp1
                            tsum = acc1 + term
                            comp1 = (tsum - acc1) - term
                            acc1 = tsum
                        end if
                    end do
                    t_sum(i, i_time) = acc1
                else
                    t_sum(i, i_time) = 0.0_KP
                end if
            end do
        end do
        do i_time = 1, n_ages
            if (.not. has_stars(i_time)) cycle
            out_idx = 1 + (i_time - 1) * TIME_RES_INCR
            spec_grid(:, out_idx) = max(spec_grid(:, out_idx) + real(t_sum(:, i_time), WP), SAFE_FLOOR)
        end do
        return
#endif
#ifdef OMP_TARGET
        call main_templates_omp(ctx, pset%zmet, w, used, has_stars, spec_grid)
        return
#endif
        do p0 = 1, nspec, PIX_BLOCK
            p1 = min(p0 + PIX_BLOCK - 1, nspec)
            acc = 0.0_KP; comp = 0.0_KP
            do k = 1, n_used
                t = used(k)
                a = mod(t - 1, nt_lib) + 1
                b = (t - 1) / nt_lib + 1
                do i_time = 1, size(w, 2)
                    if (w(t, i_time) == 0.0_wp) cycle
                    wk = real(w(t, i_time), KP)
                    if (KP == WP) then
                        do i = p0, p1
                            acc(i - p0 + 1, i_time) = acc(i - p0 + 1, i_time) + wk * ctx%state%speclib(i, pset%zmet, a, b)
                        end do
                    else
                        do i = p0, p1
                            term = wk * ctx%state%speclib(i, pset%zmet, a, b) - comp(i - p0 + 1, i_time)
                            tsum = acc(i - p0 + 1, i_time) + term
                            comp(i - p0 + 1, i_time) = (tsum - acc(i - p0 + 1, i_time)) - term
                            acc(i - p0 + 1, i_time) = tsum
                        end do
                    end if
                end do
            end do
            do i_time = 1, size(w, 2)
                if (.not. has_stars(i_time)) cycle
                out_idx = 1 + (i_time - 1) * TIME_RES_INCR
                spec_grid(p0:p1, out_idx) = max(spec_grid(p0:p1, out_idx) + real(acc(1:p1 - p0 + 1, i_time), WP), &
                                                SAFE_FLOOR)
            end do
        end do
    end subroutine add_main_library_templates

#ifdef OMP_TARGET
    !> add_main_library_templates with OpenMP target offload (-Daccel=omp_gpu): the kernel of the
    !> OpenACC build, one thread per (pixel, age) summing the used templates in the host order,
    !> compensated at FP32. The library slice is copied and mapped for the call.
    subroutine main_templates_omp(ctx, zmet, w, used, has_stars, spec_grid)
        type(fsps_context_t), intent(in)    :: ctx
        integer, intent(in)                 :: zmet, used(:)
        real(WP), intent(in)                :: w(:,:)
        logical, intent(in)                 :: has_stars(:)
        real(WP), intent(inout), contiguous :: spec_grid(:,:)

        real(kind(1.0)), allocatable :: lib(:,:,:)
        real(KP), allocatable :: w_used(:,:), t_sum(:,:)
        integer, allocatable  :: a_used(:), b_used(:)
        integer  :: nspec, nt_lib, n_used, n_ages, k, i, i_time, out_idx
        real(KP) :: acc1, comp1, wk, term, tsum

        nspec = size(spec_grid, 1)
        nt_lib = size(ctx%state%speclib, 3)
        n_used = size(used)
        n_ages = size(w, 2)
        allocate(lib(nspec, nt_lib, size(ctx%state%speclib, 4)))
        lib = ctx%state%speclib(1:nspec, zmet, :, :)
        allocate(w_used(n_used, n_ages), a_used(n_used), b_used(n_used), t_sum(nspec, n_ages))
        do k = 1, n_used
            a_used(k) = mod(used(k) - 1, nt_lib) + 1
            b_used(k) = (used(k) - 1) / nt_lib + 1
            w_used(k, :) = real(w(used(k), :), KP)
        end do
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2) private(acc1, comp1, wk, term, tsum, k) &
        !$omp map(to: lib, w_used, a_used, b_used, has_stars) map(from: t_sum)
#endif
        do i_time = 1, n_ages
            do i = 1, nspec
                acc1 = 0.0_KP
                comp1 = 0.0_KP
                if (has_stars(i_time)) then
                    do k = 1, n_used
                        wk = w_used(k, i_time)
                        if (wk == 0.0_KP) cycle
                        if (KP == WP) then
                            acc1 = acc1 + wk * lib(i, a_used(k), b_used(k))
                        else
                            term = wk * lib(i, a_used(k), b_used(k)) - comp1
                            tsum = acc1 + term
                            comp1 = (tsum - acc1) - term
                            acc1 = tsum
                        end if
                    end do
                end if
                t_sum(i, i_time) = acc1
            end do
        end do
        do i_time = 1, n_ages
            if (.not. has_stars(i_time)) cycle
            out_idx = 1 + (i_time - 1) * TIME_RES_INCR
            spec_grid(:, out_idx) = max(spec_grid(:, out_idx) + real(t_sum(:, i_time), WP), SAFE_FLOOR)
        end do
    end subroutine main_templates_omp
#endif

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
        type(fsps_context_t), intent(in), target :: ctx
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
        type(fsps_context_t), intent(inout) :: ctx
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
        type(fsps_context_t), intent(in)    :: ctx
        real(WP), intent(inout), contiguous :: spec_grid(:,:)

        integer :: j, n_times

        n_times = size(spec_grid, 2)

        ! Loop over time columns
        do j = 1, n_times
            
            ! Call the smoothing kernel for this specific time step.
            ! Note: The '99.0_wp' is a special flag in fsps_smoothing indicating
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
        type(fsps_context_t), intent(in), target :: ctx
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
        type(fsps_context_t), intent(inout) :: ctx
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

end module fsps_ssp