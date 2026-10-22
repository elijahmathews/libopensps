module opensps_gas
    !> @brief
    !> Handles nebular emission physics (gas phase).
    !>
    !> @details
    !> This module processes the reprocessing of ionizing stellar radiation by
    !> the Interstellar Medium (ISM). It includes routines to:
    !> 1. Calculate the rate of ionizing photons (Q_H) produced by the stellar population.
    !> 2. Attenuate the stellar UV flux due to absorption by neutral hydrogen.
    !> 3. Interpolate pre-computed CLOUDY photoionization grid models for
    !>    continuum and line emission.
    !> 4. Add smoothed emission lines and nebular continuum to the output spectrum.
    !>
    !> AD_NOTE: This module contains table lookups and discrete index finding
    !> (`find_interval`), which create discontinuities. Special care is required
    !> for Automatic Differentiation (AD).

    use opensps_precision, only: WP, KP, kahan_add
    use opensps_constants, only: NEMLINE, NEBNAGE, NEBNZ, NEBNIP, C_LIGHT, &
                              H_PLANCK, L_SOL, SAFE_FLOOR, PI
    use opensps_types, only: params
    use opensps_context_types, only: opensps_context_t
    use opensps_integration, only: integrate_trapezoid_array
    use opensps_interpolation, only: find_interval
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan

    implicit none
    private

    ! Public Interface
    public :: apply_nebular_emission
    public :: process_ionizing_radiation
    public :: interpolate_zu_slice_point
    public :: get_grid_indices_weights
    public :: prepare_line_gaussians

    ! ------------------------------------------------------------------------
    ! CONSTANTS
    ! ------------------------------------------------------------------------
    real(WP), parameter :: SQRT_2_PI  = sqrt(2.0_wp * PI)
    real(WP), parameter :: KM_S_TO_ANGSTROM_FACTOR = 1.0e13_wp / C_LIGHT

contains

    !> @brief
    !> Main driver for adding nebular emission to the stellar spectrum.
    !>
    !> @details
    !> Iterates over the age of the stellar population. For each age step:
    !> 1. Calculates the number of ionizing photons (Q) that are absorbed by gas.
    !> 2. Interpolates the nebular grid (CLOUDY) for the given metallicity, 
    !>    ionization parameter, and age.
    !> 3. Adds the scaled nebular continuum and emission lines to the spectrum.
    !>
    !> @param[inout] ctx         FSPS context (contains CLOUDY grids and wavelength arrays).
    !> @param[in]    pset        FSPS parameter structure (gas_logz, gas_logu, etc.).
    !> @param[in]    sspi        Input Stellar Spectrum (L_sol/Hz).
    !> @param[inout] sspo        Output Spectrum (Stellar + Nebular).
    !> @param[out]   nebemline   (Optional) Output array for individual line luminosities.
    subroutine apply_nebular_emission(ctx, pset, sspi, sspo, nebemline)

        type(opensps_context_t), intent(inout)       :: ctx
        type(params), intent(in)                  :: pset
        real(KP), dimension(:,:), intent(in)      :: sspi
        real(KP), dimension(:,:), intent(inout)   :: sspo
        real(KP), dimension(:,:), intent(inout), optional :: nebemline

        ! Loop variables
        integer :: t, k, max_neb_time_idx
        integer :: nspec
        
        ! Grid Interpolation Indices & Weights
        integer :: idx_z, idx_u, idx_a
        real(WP) :: w_z, w_u, w_a
        
        ! Physics variables
        real(WP) :: q_ionizing
        logical :: use_xrb_grid
        logical :: calc_lines, calc_cont

        nspec = size(sspi, 1)

        ! sspi is either mapped by the caller (the SSP basis in compute_csp_scenario, current on the
        ! device) or copied in here; no update (it re-sent the whole basis on every call)

        ! 0. Initialization & Validation
        ! ------------------------------
        sspo = sspi
        
        if (present(nebemline)) then
            nebemline = 0.0_wp
        end if

        ! Determine what needs calculating
        calc_cont = (ctx%add_neb_continuum_val == 1)
        calc_lines = (ctx%nebemlineinspec_val == 1 .or. present(nebemline))
        
        ! Logic flag for XRB (BPSS models)
        use_xrb_grid = (ctx%state%isoc_type == 'bpss') .and. (ctx%add_xrb_emission_val == 1)

        call prepare_line_gaussians(ctx, pset)

        ! 1. Pre-calculate Interpolation Weights for Z and U
        ! --------------------------------------------------
        ! These are constant for the entire subroutine call.
        call get_grid_indices_weights(ctx%state%nebem_logz, pset%gas_logz, NEBNZ, idx_z, w_z)
        call get_grid_indices_weights(ctx%state%nebem_logu, pset%gas_logu, NEBNIP, idx_u, w_u)


        ! 2. Dimensionality Reduction
        ! ----------------------------------------------

        ! (On the host: SSP generation with add_neb_emission = 2; the batch interpolates the grids
        ! on the device itself, nebular_kernels)
        if (calc_cont) then
            do k = 1, NEBNAGE
                do t = 1, nspec
                    if (use_xrb_grid) then
                       ctx%state%gas_neb_cont_reduced(t, k) = interpolate_zu_slice_point(ctx%state%xnebem_cont, &
                            t, k, idx_z, idx_u, w_z, w_u)
                    else
                        ctx%state%gas_neb_cont_reduced(t, k) = interpolate_zu_slice_point(ctx%state%nebem_cont, &
                            t, k, idx_z, idx_u, w_z, w_u)
                    end if
                end do
            end do
        end if

        if (calc_lines) then
            do k = 1, NEBNAGE
                do t = 1, NEMLINE
                    if (use_xrb_grid) then
                       ctx%state%gas_neb_line_reduced(t, k) = interpolate_zu_slice_point(ctx%state%xnebem_line, &
                            t, k, idx_z, idx_u, w_z, w_u)
                    else
                       ctx%state%gas_neb_line_reduced(t, k) = interpolate_zu_slice_point(ctx%state%nebem_line, &
                            t, k, idx_z, idx_u, w_z, w_u)
                    end if
                end do
            end do
        end if
        
        ! 3. Main Time Loop
        ! -----------------
        max_neb_time_idx = find_interval(ctx%state%time_full, ctx%state%nebem_age(NEBNAGE))

        ! Scratch arrays initialized directly on the device
        ctx%state%gas_current_step_cont = 0.0_wp
        ctx%state%gas_current_step_lines = log10(SAFE_FLOOR)

        do t = 1, max_neb_time_idx
            
            ! A. Calculate Ionizing Photons
            call process_ionizing_radiation(ctx, pset, sspi, sspo, t, q_ionizing)

            ! Optimization: Skip expensive math if there is no ionizing radiation
            if (q_ionizing <= SAFE_FLOOR) cycle

            ! B. Interpolate Age (1D)
            ! -----------------------
            call get_grid_indices_weights(ctx%state%nebem_age, ctx%state%time_full(t), &
                                          NEBNAGE, idx_a, w_a)

            ! C. Add Continuum
            ! ----------------
            if (calc_cont) then
                do k = 1, nspec
                    ctx%state%gas_current_step_cont(k) = (1.0_wp - w_a) * ctx%state%gas_neb_cont_reduced(k, idx_a) + &
                                                         (         w_a) * ctx%state%gas_neb_cont_reduced(k, idx_a + 1)

                    sspo(k,t) = sspo(k,t) + (10.0_wp**ctx%state%gas_current_step_cont(k)) * q_ionizing
                end do
            end if

            ! D. Add Lines
            ! ------------
            if (calc_lines) then
                do k = 1, NEMLINE
                    ctx%state%gas_current_step_lines(k) = (1.0_wp - w_a) * ctx%state%gas_neb_line_reduced(k, idx_a) + &
                                                          (         w_a) * ctx%state%gas_neb_line_reduced(k, idx_a + 1)
                end do

                if (present(nebemline)) then
                    do k = 1, NEMLINE
                        nebemline(k,t) = (10.0_wp**ctx%state%gas_current_step_lines(k)) * q_ionizing
                    end do
                end if

                if (ctx%nebemlineinspec_val == 1) then
                    call add_lines_to_spectrum(ctx, sspo, t, ctx%state%gas_current_step_lines, q_ionizing)
                end if
            end if

        end do


    end subroutine apply_nebular_emission

    !> @brief Line profiles for pset (gaussnebarr, also on the device): per-call kernels depend
    !> only on sigma_smooth and smooth_velocity, so they are rebuilt (about 1e6 exponentials) only
    !> when those change.
    subroutine prepare_line_gaussians(ctx, pset)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset

        if (ctx%setup_nebular_gaussians_val == 0 .and. ctx%nebemlineinspec_val == 1) then
            if (pset%sigma_smooth /= ctx%state%gauss_built_sigma .or. &
                ctx%smooth_velocity_val /= ctx%state%gauss_built_velocity) then
                call compute_line_gaussians(ctx, pset)
#ifdef OMP_TARGET
                !$omp target update if(ctx%on_device) to(ctx%state%gaussnebarr)
#endif
                ctx%state%gauss_built_sigma = pset%sigma_smooth
                ctx%state%gauss_built_velocity = ctx%smooth_velocity_val
            end if
        end if
    end subroutine prepare_line_gaussians


    ! ------------------------------------------------------------------------
    ! PRIVATE HELPER ROUTINES
    ! ------------------------------------------------------------------------

    !> @brief
    !> Calculates the number of ionizing photons and attenuates the stellar UV.
    !>
    !> @details
    !> 1. Attenuates the output spectrum (`spec_out`) below the Lyman limit 
    !>    based on `frac_obrun` (fraction of photons that escape the nebula).
    !> 2. Integrates the *input* spectrum (`spec_in`) to find the total 
    !>    ionizing flux available.
    !> 3. Returns `q_val`, the number of photons *absorbed* by the gas.
    !>
    !> @param[in]  ctx       FSPS context (wavelength arrays).
    !> @param[in]  pset      Parameters (frac_obrun).
    !> @param[in]  spec_in   Intrinsic stellar spectrum grid.
    !> @param[out] spec_out  Attenuated stellar spectrum grid.
    !> @param[in]  t_idx     Time index into spectral grids.
    !> @param[out] q_val     Number of ionizing photons absorbed.
    subroutine process_ionizing_radiation(ctx, pset, spec_in, spec_out, t_idx, q_val)
        type(opensps_context_t), intent(in)    :: ctx
        type(params), intent(in)            :: pset
        real(KP), dimension(:,:), intent(in)  :: spec_in
        real(KP), dimension(:,:), intent(inout) :: spec_out
        integer, intent(in)                 :: t_idx
        real(WP), intent(out)               :: q_val
        
        integer :: whlylim
        real(KP) :: integral_flux, comp, wt   ! integral at kernel precision
        integer :: i
        real(WP) :: frac_obrun_clamped
        
        whlylim = ctx%state%whlylim
        frac_obrun_clamped = max(0.0_wp, min(pset%frac_obrun, 1.0_wp))

        ! spec_in is mapped by the caller or copied in here; an update here re-sent the whole SSP
        ! grid for every young age (~50 x 5 MB per nebular CSP call)

        ! 1. Attenuate Output Spectrum (EUV < 912 A)
        if (whlylim > 0) then
            do i = 1, whlylim
                spec_out(i, t_idx) = spec_in(i, t_idx) * real(frac_obrun_clamped, KP)
            end do
        end if

        ! 2. Calculate Total Ionizing Photons (Q)
        if (whlylim < 2) then
            q_val = 0.0_wp
        else
            ! Photon-number integral of L_nu / nu at kernel precision (the h and L_sun scaling,
            ! ~6e59, stays in WP): a plain reduction on the GPU, compensated on the CPU
            integral_flux = 0.0_KP
            comp = 0.0_KP
            do i = 1, whlylim - 1
                wt = real(0.5_wp * abs(ctx%state%spec_nu(i+1) - ctx%state%spec_nu(i)), KP)
                if (KP == WP) then   ! compile-time constant: plain sum, no call
                    integral_flux = integral_flux + wt * (spec_in(i, t_idx) / real(ctx%state%spec_nu(i), KP) + &
                                                          spec_in(i+1, t_idx) / real(ctx%state%spec_nu(i+1), KP))
                else
                    call kahan_add(integral_flux, comp, wt * (spec_in(i, t_idx) / real(ctx%state%spec_nu(i), KP) + &
                                                              spec_in(i+1, t_idx) / real(ctx%state%spec_nu(i+1), KP)))
                end if
            end do
            
            ! We check NaN on host? Or device?
            ! Can't check IEEE NaN easily on device across reduction.
            ! Assuming logic holds.
            q_val = abs((real(integral_flux, WP) / H_PLANCK * L_SOL) * (1.0_wp - pset%frac_obrun))
        end if


    end subroutine process_ionizing_radiation

    !> @brief
    !> Pre-computes Gaussian profiles for emission lines.
    !>
    !> @details
    !> Computes the normalized Gaussian profile `gaussnebarr` for each emission line,
    !> broadened by `sigma_smooth`.
    !>
    !> AD_NOTE: This modifies `ctx%state`. In a purely functional AD context, 
    !> this might need to return a local array instead.
    subroutine compute_line_gaussians(ctx, pset)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        
        integer :: i
        real(WP) :: sigma_angstroms, norm_factor
        real(WP), allocatable, dimension(:) :: lambda_grid

        allocate(lambda_grid(size(ctx%state%spec_lambda)))
        lambda_grid = 0.0_wp
        lambda_grid = ctx%state%spec_lambda

        do i = 1, NEMLINE
            ! Determine width in Angstroms
            if (ctx%smooth_velocity_val == 1) then
                ! velocity smoothing (sigma_smooth in km/s)
                sigma_angstroms = ctx%state%nebem_line_pos(i) * pset%sigma_smooth * KM_S_TO_ANGSTROM_FACTOR
            else
                ! wavelength smoothing (sigma_smooth in Angstroms)
                sigma_angstroms = pset%sigma_smooth
            end if

            ! Enforce minimum resolution (Nyquist sampling or instrumental limit)
            ! The factor of 2 ensures we don't alias on the grid.
            sigma_angstroms = max(sigma_angstroms, ctx%state%neb_res_min(i) * 2.0_wp)

            ! Gaussian line profile normalized by 1/(sqrt(2*pi)*sigma).
            ! The lambda^2 / c factor converts L_lambda to L_nu so that line
            ! luminosities and spectrum units remain consistent downstream.
            norm_factor = (1.0_wp / (SQRT_2_PI * sigma_angstroms)) * &
                          (ctx%state%nebem_line_pos(i)**2 / C_LIGHT)

            ctx%state%gaussnebarr(:,i) = norm_factor * &
                exp( -0.5_wp * ((lambda_grid - ctx%state%nebem_line_pos(i)) / sigma_angstroms)**2 )
        end do

        deallocate(lambda_grid)
    end subroutine compute_line_gaussians

    !> @brief
    !> Adds emission lines to the spectrum.
    subroutine add_lines_to_spectrum(ctx, spectrum, t_idx, line_lum_log, q_val)
        type(opensps_context_t), intent(in)        :: ctx
        real(KP), dimension(:,:), intent(inout) :: spectrum
        integer, intent(in)                     :: t_idx
        real(WP), dimension(:), intent(in)      :: line_lum_log
        real(WP), intent(in)                    :: q_val

        integer :: i, j
        real(KP) :: sum_val
        real(WP) :: amp(NEMLINE)   ! Line luminosities (10**log * Q), computed once per call
        real(WP), parameter :: LINE_PROFILE_CUT = 1.0e-14_wp
        integer :: n, jc, j_lo, j_hi
        real(WP) :: cut

        ! Line amplitudes once per call: in the wavelength loop below, 10** would be evaluated for
        ! every line at every pixel (~1e6 times per SSP age), which dominated nebular CSP calls
        do i = 1, NEMLINE
            amp(i) = (10.0_wp**line_lum_log(i)) * q_val
        end do

        ! On the CPU, add each line only where its profile is non-negligible: a line spans tens of
        ! the ~6000 pixels, so the dense sum is ~99% multiplications by (nearly) zero. The window
        ! extends from the pixel nearest the line centre until the profile falls below
        ! LINE_PROFILE_CUT of its peak (~8 sigma; the neglected tails are ~1e-15 of the line).
        n = size(spectrum, 1)
        do i = 1, NEMLINE
            if (amp(i) == 0.0_wp) cycle
            jc = max(1, min(find_interval(ctx%state%spec_lambda(1:n), ctx%state%nebem_line_pos(i)), n))
            j_hi = min(jc + 1, n)
            cut = LINE_PROFILE_CUT * max(ctx%state%gaussnebarr(jc, i), ctx%state%gaussnebarr(j_hi, i))
            if (cut <= 0.0_wp) cycle
            j_lo = jc
            do while (j_lo > 1)
                if (ctx%state%gaussnebarr(j_lo - 1, i) <= cut) exit
                j_lo = j_lo - 1
            end do
            do while (j_hi < n)
                if (ctx%state%gaussnebarr(j_hi + 1, i) <= cut) exit
                j_hi = j_hi + 1
            end do
            spectrum(j_lo:j_hi, t_idx) = spectrum(j_lo:j_hi, t_idx) + real(amp(i), KP) * ctx%state%gaussnebarr(j_lo:j_hi, i)
        end do
    end subroutine add_lines_to_spectrum

    !> @brief
    !> Finds grid index and linear interpolation weight for a single value.
    !>
    !> @details
    !> AD_NOTE: `find_interval` introduces a discontinuity in the derivative 
    !> of the index w.r.t the input value. The weight `w` is differentiable.
    subroutine get_grid_indices_weights(grid, val, n_grid, idx, weight)
        real(WP), dimension(:), intent(in) :: grid
        real(WP), intent(in)               :: val
        integer, intent(in)                :: n_grid
        integer, intent(out)               :: idx
        real(WP), intent(out)              :: weight
        
        ! Find index in sorted array
        idx = find_interval(grid, val)
        
        ! Clamp to valid range [1, N-1] for interpolation safety
        idx = max(1, min(idx, n_grid - 1))
        
        ! Calculate weight (0.0 to 1.0)
        weight = (val - grid(idx)) / (grid(idx+1) - grid(idx))
        
        ! Clamp weight to avoid extrapolation
        weight = max(0.0_wp, min(weight, 1.0_wp))
    end subroutine get_grid_indices_weights

    !> @brief
    !> Scalar version of interpolate_zu_slice for use inside parallel loops.
    !> Returns single value at index `i_wave`.
    pure function interpolate_zu_slice_point(grid, i_wave, idx_age, idx_z, idx_u, w_z, w_u) result(val)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), dimension(:,:,:,:), intent(in) :: grid
        integer, intent(in)  :: i_wave, idx_age, idx_z, idx_u
        real(WP), intent(in) :: w_z, w_u
        real(WP) :: val
        
        real(WP) :: c00, c01, c10, c11

        c00 = (1.0_wp - w_z) * (1.0_wp - w_u)
        c01 = (1.0_wp - w_z) * (         w_u)
        c10 = (         w_z) * (1.0_wp - w_u)
        c11 = (         w_z) * (         w_u)

        val = c00 * grid(i_wave, idx_z,   idx_age, idx_u  ) + &
              c01 * grid(i_wave, idx_z,   idx_age, idx_u+1) + &
              c10 * grid(i_wave, idx_z+1, idx_age, idx_u  ) + &
              c11 * grid(i_wave, idx_z+1, idx_age, idx_u+1)

    end function interpolate_zu_slice_point

end module opensps_gas
