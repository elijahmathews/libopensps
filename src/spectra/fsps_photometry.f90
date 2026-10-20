
module fsps_photometry
    !> @brief
    !> Module for calculating photometric magnitudes (AB or Vega) from spectra.
    !>
    !> @details
    !> This module replaces the legacy `spec_mags.f90`. It handles:
    !> 1. Redshifting the rest-frame spectrum to the observed frame.
    !> 2. Applying cosmological distance dimming (Distance Modulus).
    !> 3. Integrating the spectrum over filter transmission curves.
    !> 4. Converting fluxes to AB or Vega magnitudes.

    use fsps_precision, only: WP, KP
    use fsps_constants, only: SAFE_FLOOR, ABS_MAG_ZEROPOINT_LOG
    use fsps_context_types, only: fsps_context_t
    use fsps_interpolation, only: interpolate_linear, find_interval
    use fsps_special_functions, only: mag_from_flux
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan

    implicit none
    private

    public :: compute_magnitudes

    ! ------------------------------------------------------------------------
    ! CONSTANTS
    ! ------------------------------------------------------------------------
    real(WP), parameter :: AB_ZEROPOINT   = 48.60_wp   !< CGS to AB Mag constant
    real(WP), parameter :: LOG10_FACTOR   = 2.5_wp     !< Magnitude scaling factor
    integer, parameter  :: IDX_V_BAND     = 1          !< Index of V-band (for Vega norm)

contains

    !> @brief
    !> Calculates magnitudes for a given spectrum and redshift.
    !>
    !> @details
    !> The routine first shifts the input spectrum (Lsun/Hz) to the observed frame
    !> if z > 0. It then integrates over the transmission curves defined in
    !> `ctx%state%bands`.
    !>
    !> @param[in]    ctx          The FSPS context (contains filter curves, cosmology).
    !> @param[in]    zred         Redshift of the galaxy.
    !> @param[in]    spec         Rest-frame spectrum (Lsun/Hz).
    !> @param[out]   mags         Output magnitudes array.
    !> @param[in]    mag_compute  (Optional) Integer mask (1=compute, 0=skip) for bands.
    subroutine compute_magnitudes(ctx, zred, spec, mags, mag_compute)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: zred
        real(WP), dimension(:), intent(in), contiguous :: spec
        real(WP), dimension(:), intent(out), contiguous :: mags
        integer, dimension(:), intent(in), optional :: mag_compute

        ! Local variables
        integer :: i, j, n_spec, n_bands
        integer, allocatable :: work_flags(:)
        
        ! Scratch arrays allocated on host.
        real(WP), allocatable :: obs_frame_spec(:)
        real(KP), allocatable :: flux_over_lambda(:)
        
        real(WP) :: dist_mod_term, integrated_flux
        real(KP) :: s_band, c_band, t_band, u_band   ! filter integral at kernel precision, compensated
        logical :: do_vega, do_light_ages

        ! 1. Setup and Validation
        n_spec = size(spec)
        n_bands = size(mags)

        if (n_spec < 2) then
            mags = get_quiet_nan()
            return
        end if
        
        ! Initialize output
        mags = get_quiet_nan()

        ! Parse Context Flags
        do_vega = (ctx%compute_vega_mags_val == 1)
        do_light_ages = (ctx%compute_light_ages_val == 1)

        ! 2. Determine which bands to compute
        ! NOTE: Allocation on host is fine if we copy it, but Rule 3 says NO heap alloc in kernels.
        ! If this routine is a kernel, we can't allocate.
        ! But work_flags is integer mask. We can use a simpler approach.
        ! For now, let's assume this part runs on host or we map it?
        ! Actually, if 'mag_compute' is present, it's on host?
        ! This routine is tricky if it bridges host/device. 
        ! Assuming 'spec' and 'mags' are device resident. 'mag_compute' might be host.
        ! We will create work_flags on device.
        allocate(work_flags(n_bands))
        allocate(obs_frame_spec(n_spec), flux_over_lambda(n_spec))
        
        if (present(mag_compute)) then
            work_flags = mag_compute
            if (do_vega) work_flags(IDX_V_BAND) = 1
        else
            work_flags = 1 
        end if
        
        ! 3. Prepare Spectrum (Redshift & Distance Modulus)
        call prepare_observed_spectrum(ctx, zred, spec, obs_frame_spec, dist_mod_term, n_spec)

        ! 4. Pre-calculate terms for Integration
        do i = 1, n_spec
            if (ctx%state%spec_lambda(i) > tiny(0.0_wp)) then
                flux_over_lambda(i) = real(obs_frame_spec(i) / ctx%state%spec_lambda(i), KP)
            else
                flux_over_lambda(i) = 0.0_KP
            end if
        end do

        ! 5. Filter Integration Loop
        do i = 1, n_bands
            if (work_flags(i) == 0) cycle

            s_band = 0.0_KP
            c_band = 0.0_KP
            do j = 1, n_spec - 1
                ! 0.5 |dlambda| depends only on the grid (precomputable); the products and the
                ! sum run at kernel precision
                if (KP == WP) then   ! compile-time constant: plain sum, no call
                    s_band = s_band + real(0.5_wp * abs(ctx%state%spec_lambda(j + 1) - ctx%state%spec_lambda(j)), KP) * &
                             (flux_over_lambda(j) * real(ctx%state%bands(j, i), KP) + &
                              flux_over_lambda(j + 1) * real(ctx%state%bands(j + 1, i), KP))
                else
                    ! Compensated sum, inline (this loop runs bands x pixels times per spectrum)
                    t_band = real(0.5_wp * abs(ctx%state%spec_lambda(j + 1) - ctx%state%spec_lambda(j)), KP) * &
                             (flux_over_lambda(j) * real(ctx%state%bands(j, i), KP) + &
                              flux_over_lambda(j + 1) * real(ctx%state%bands(j + 1, i), KP)) - c_band
                    u_band = s_band + t_band
                    c_band = (u_band - s_band) - t_band
                    s_band = u_band
                end if
            end do
            integrated_flux = real(s_band, WP)

            if (.not. do_light_ages) then
                mags(i) = mag_from_flux(integrated_flux) - &
                            AB_ZEROPOINT - &
                            (LOG10_FACTOR * ABS_MAG_ZEROPOINT_LOG) + &
                            dist_mod_term
            else
                if (integrated_flux > SAFE_FLOOR) then
                    mags(i) = integrated_flux
                end if
            end if
        end do

        ! 6. Vega System Correction
        if (do_vega .and. .not. do_light_ages) then
            do i = 2, n_bands
                if (mags(IDX_V_BAND) == mags(IDX_V_BAND)) then
                    mags(i) = mags(i) - ctx%state%magvega(i) + ctx%state%magvega(IDX_V_BAND)
                else
                    mags(i) = get_quiet_nan()
                end if
            end do
        end if

        ! Cleanup
        deallocate(work_flags)
        deallocate(obs_frame_spec, flux_over_lambda)

    end subroutine compute_magnitudes

    ! ------------------------------------------------------------------------
    ! PRIVATE HELPERS
    ! ------------------------------------------------------------------------

    !> @brief
    !> Handles redshifting and cosmology calculations.
    !> 
    !> @details
    !> If z > 0, interpolates the spectrum to the observed frame and calculates
    !> the distance modulus term.
    subroutine prepare_observed_spectrum(ctx, z, spec_rest, spec_obs, dist_term, n)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: z
        real(WP), dimension(:), intent(in) :: spec_rest
        real(WP), dimension(:), intent(out) :: spec_obs
        real(WP), intent(out) :: dist_term
        integer, intent(in) :: n

        real(WP) :: dm, z_factor, target_lam_rest
        integer :: k, idx

        if (abs(z) > SAFE_FLOOR) then
            ! --- High Redshift Case ---
            z_factor = 1.0_wp + z
            
            do k = 1, n
                ! The rest-frame wavelength corresponding to this observed grid point
                target_lam_rest = ctx%state%spec_lambda(k) / z_factor

                ! Binary search (safer for parallel execution than hunting with shared idx)
                ! Assuming find_interval is !acc routine seq
                idx = find_interval(ctx%state%spec_lambda, target_lam_rest)
                idx = max(1, min(idx, n - 1))

                ! Manual Linear Interpolation
                spec_obs(k) = spec_rest(idx) + &
                              (target_lam_rest - ctx%state%spec_lambda(idx)) * &
                              (spec_rest(idx+1) - spec_rest(idx)) / &
                              (ctx%state%spec_lambda(idx+1) - ctx%state%spec_lambda(idx))
                
                ! Physical Constraints
                if (spec_obs(k) /= spec_obs(k)) spec_obs(k) = 0.0_wp
                spec_obs(k) = max(spec_obs(k), 0.0_wp)
            end do

            ! 2. Cosmology / Distance Modulus
            dm = interpolate_linear(ctx%state%cosmospl(:,1), &
                                    ctx%state%cosmospl(:,3), &
                                    z)

            if (dm /= dm .or. dm <= SAFE_FLOOR) then
                dist_term = 0.0_wp
            else
                dist_term = (5.0_wp * log10(dm / 10.0_wp)) + & 
                            mag_from_flux(z_factor) ! Bandwidth stretching
            end if

        else
            ! --- Zero Redshift Case ---
            do k = 1, n
                spec_obs(k) = spec_rest(k)
            end do
            dist_term = 0.0_wp
        end if

    end subroutine prepare_observed_spectrum

    !> @brief Helper to generate a Quiet NaN
    pure function get_quiet_nan() result(val)
        real(WP) :: val
        val = ieee_value(0.0_wp, ieee_quiet_nan)
    end function get_quiet_nan

end module fsps_photometry