module fsps_spectral_indices
    !> @brief
    !> Module for calculating spectral indices (equivalent widths, magnitudes,
    !> and breaks) from stellar population spectra.
    !>
    !> @details
    !> This module replaces the legacy `spec_indices.f90`. It handles the
    !> integration of spectra over defined bandpasses and computes indices
    !> based on the definitions in the FSPS data files.
    !>
    !> **Key Features:**
    !> - Trapezoidal integration with precise sub-pixel interpolation at bounds.
    !> - Support for Equivalent Widths, Magnitudes, Dn4000, and Flux Ratios.
    !> - Robust handling of continuum fitting (blue/red sidebands).

    use fsps_precision, only: WP, KP
    use fsps_constants, only: SAFE_FLOOR
    use fsps_context_types, only: fsps_context_t
    use fsps_integration, only: integrate_trapezoid_array
    use fsps_interpolation, only: find_interval
    use fsps_special_functions, only: mag_from_flux

    implicit none
    private

    public :: compute_spectral_indices, compute_spectral_indices_columns
    public :: integrate_interval, integrate_ratio_interval

    ! ------------------------------------------------------------------------
    ! CONSTANTS: Index Definition Map (row indices for ctx%indexdefined)
    ! ------------------------------------------------------------------------
    integer, parameter :: IDX_FEAT_START = 1 !< Feature integration start
    integer, parameter :: IDX_FEAT_END   = 2 !< Feature integration end
    integer, parameter :: IDX_BLUE_START = 3 !< Blue continuum start
    integer, parameter :: IDX_BLUE_END   = 4 !< Blue continuum end
    integer, parameter :: IDX_RED_START  = 5 !< Red continuum start
    integer, parameter :: IDX_RED_END    = 6 !< Red continuum end
    integer, parameter :: IDX_T          = 7 !< Index unit type identifier

    ! ------------------------------------------------------------------------
    ! CONSTANTS: Unit Types (values found in indexdefined(7, :))
    ! ------------------------------------------------------------------------
    real(WP), parameter :: UNIT_TYPE_MAG        = 1.0_wp !< Magnitude: -2.5 * log10(Flux)
    real(WP), parameter :: UNIT_TYPE_EW         = 2.0_wp !< Equivalent Width (Angstroms)
    real(WP), parameter :: UNIT_TYPE_DN4000     = 3.0_wp !< Break Strength (Red/Blue ratio)
    real(WP), parameter :: UNIT_TYPE_FLUX_RATIO = 4.0_wp !< Mag from Flux Ratio

    ! ------------------------------------------------------------------------
    ! CONSTANTS: Sentinel Values
    ! ------------------------------------------------------------------------
    real(WP), parameter :: IND_UNDEFINED = 999.0_wp

contains

    !> @brief
    !> Main routine to calculate all defined spectral indices for a given spectrum.
    !>
    !> @details
    !> Iterates through the indices defined in the FSPS context (`ctx%state%indexdefined`).
    !> Calculates blue/red continua and integrates the feature based on the
    !> specific type of index (EW, Mag, etc).
    !>
    !> @param[in]    ctx      The FSPS context containing index definitions.
    !> @param[in]    lambda   Wavelength array (Angstroms).
    !> @param[in]    spec     Flux/Spectrum array (Lsun/Hz or similar).
    !> @param[inout] indices  Output array of calculated indices.
    subroutine compute_spectral_indices(ctx, lambda, spec, indices)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(:), intent(in), contiguous :: lambda, spec
        real(WP), dimension(:), intent(inout), contiguous :: indices

        integer :: j, n_idx

        ! Local copies for speed/readability
        n_idx = ctx%state%nindx

        !$acc data pcopyin(lambda, spec) pcopy(indices)
        !$acc update device(spec, lambda)

        ! Initialize with sentinel
        !$acc kernels present(spec, lambda, indices)
        indices = IND_UNDEFINED
        !$acc end kernels

        ! Iterate over all defined indices
        ! Parallelize over indices (gang vector)
        !$acc parallel loop gang vector present(spec, lambda, indices, ctx)
        do j = 1, n_idx
            indices(j) = spectral_index_value(ctx, lambda, spec, j)
        end do

        !$acc end data

    end subroutine compute_spectral_indices


    !> @brief Value of spectral index j of one spectrum (the body of compute_spectral_indices).
    function spectral_index_value(ctx, lambda, spec, j) result(val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(:), intent(in), contiguous :: lambda, spec
        integer, intent(in) :: j
        real(WP) :: val

        real(WP) :: idx_feat_lo, idx_feat_hi, idx_type
        real(WP) :: idx_blue_lo, idx_blue_hi
        real(WP) :: idx_red_lo, idx_red_hi
        real(WP) :: val_blue_cont, val_red_cont
        real(WP) :: lambda_blue_cen, lambda_red_cen
        real(WP) :: integrated_flux, feature_width
        real(WP) :: cont_slope, cont_intercept, denom_width

        val = IND_UNDEFINED
        
        ! 1. Extract Definitions
        idx_feat_lo = ctx%state%indexdefined(IDX_FEAT_START, j)
        idx_feat_hi = ctx%state%indexdefined(IDX_FEAT_END, j)
        idx_blue_lo = ctx%state%indexdefined(IDX_BLUE_START, j)
        idx_blue_hi = ctx%state%indexdefined(IDX_BLUE_END, j)
        idx_red_lo  = ctx%state%indexdefined(IDX_RED_START, j)
        idx_red_hi  = ctx%state%indexdefined(IDX_RED_END, j)
        idx_type    = ctx%state%indexdefined(IDX_T, j)

        ! 2. Check Bounds
        ! If the feature definition is outside the provided wavelength grid, skip
        if (idx_blue_lo < lambda(1) .or. idx_red_hi > lambda(size(lambda))) then
            val = IND_UNDEFINED
            return
        end if

        ! 3. Compute Blue Continuum
        ! Integrate flux in blue band, divide by width to get mean level
        val_blue_cont = integrate_interval(lambda, spec, idx_blue_lo, idx_blue_hi)
        val_blue_cont = val_blue_cont / max(idx_blue_hi - idx_blue_lo, tiny(0.0_wp))
        lambda_blue_cen = (idx_blue_lo + idx_blue_hi) * 0.5_wp

        ! 4. Branch based on Index Type
        if (abs(idx_type - UNIT_TYPE_FLUX_RATIO) > tiny(0.0_wp)) then
            ! --------------------------------------------------------
            ! Standard Indices (Mag, EW, Dn4000) - require Red Continuum
            ! --------------------------------------------------------
            
            ! Compute Red Continuum
            val_red_cont = integrate_interval(lambda, spec, idx_red_lo, idx_red_hi)
            val_red_cont = val_red_cont / max(idx_red_hi - idx_red_lo, tiny(0.0_wp))
            lambda_red_cen = (idx_red_lo + idx_red_hi) * 0.5_wp

            ! Compute Integral of (Flux / Continuum)
            ! Here, continuum is a linear interpolation between Blue and Red centers
            denom_width = max(lambda_red_cen - lambda_blue_cen, tiny(0.0_wp))
            cont_slope  = (val_red_cont - val_blue_cont) / denom_width
            cont_intercept = val_blue_cont
            
            ! Integrate (Spec / Linear_Continuum)
            ! Helper handles the division internally over the specific range
            integrated_flux = integrate_ratio_interval(lambda, spec, &
                              idx_feat_lo, idx_feat_hi, &
                              cont_slope, cont_intercept, lambda_blue_cen)

        else
            ! --------------------------------------------------------
            ! Flux Ratio Type - does not normalize by linear continuum
            ! --------------------------------------------------------
            integrated_flux = integrate_interval(lambda, spec, idx_feat_lo, idx_feat_hi)
            integrated_flux = integrated_flux / max(idx_feat_hi - idx_feat_lo, tiny(0.0_wp))
        end if

        ! 5. Calculate Final Index Value
        feature_width = idx_feat_hi - idx_feat_lo

        if (abs(idx_type - UNIT_TYPE_MAG) < tiny(0.0_wp)) then
            ! Magnitude
            val = mag_from_flux(integrated_flux / feature_width)

        else if (abs(idx_type - UNIT_TYPE_EW) < tiny(0.0_wp)) then
            ! Equivalent Width
            val = feature_width - integrated_flux

        else if (abs(idx_type - UNIT_TYPE_DN4000) < tiny(0.0_wp)) then
            ! Dn4000 (Red / Blue)
            ! Note: Logic relies on pre-computed continua. 
            ! Dn4000 uses the 'cr' and 'cb' values directly.
            if (val_blue_cont > 0.0_wp) then
                val = val_red_cont / val_blue_cont
            else
                val = IND_UNDEFINED
            end if

        else if (abs(idx_type - UNIT_TYPE_FLUX_RATIO) < tiny(0.0_wp)) then
            ! Flux Ratio Magnitude
            if (integrated_flux > 0.0_wp .and. val_blue_cont > 0.0_wp) then
                val = mag_from_flux(integrated_flux / val_blue_cont)
            else
                val = IND_UNDEFINED
            end if
        end if
    end function spectral_index_value

    !> @brief compute_spectral_indices for the columns spec(:, 1:n_col) of a spectrum array
    !> resident on the device at kernel precision (e.g. the CSP outputs): one thread per (index,
    !> spectrum), on WP copies of the spectra, as the per-spectrum routine.
    subroutine compute_spectral_indices_columns(ctx, lambda, spec, n_col, indices)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(:), intent(in), contiguous, target :: lambda
        real(KP), dimension(:,:), intent(in) :: spec
        integer, intent(in) :: n_col
        real(WP), dimension(:,:), intent(out) :: indices   ! (nindx, n_col)

        real(WP), allocatable :: specw(:,:), ind(:,:)
        real(WP), allocatable, target :: lam(:)
        real(WP), pointer, contiguous :: lp(:)
        integer :: j, m, i, n_idx, nspec

        n_idx = ctx%state%nindx
        nspec = size(lambda)
        ! Results in a local array: indices may have more columns than n_col, and mapping a
        ! section of it while the kernel refers to the whole dummy is "partially present"
        allocate(specw(nspec, n_col), ind(n_idx, n_col))
#ifdef OMP_TARGET
        ! A local copy of the wavelengths for the kernel: with OpenMP target offload, nvfortran 25.1
        ! fails (illegal address) when a device function receives an assumed-shape dummy of the
        ! enclosing host routine; local arrays and their sections work
        lam = lambda
        lp => lam
#else
        ! OpenACC: the wavelengths as given (the context's, already on the device: copyin finds
        ! them there instead of uploading 48 kB per call)
        lp => lambda
#endif
        !$acc data present(spec) copyin(lp) create(specw) copyout(ind)
#ifdef OMP_TARGET
        !$omp target data map(to: lam) map(alloc: specw) map(from: ind)
#endif
        !$acc parallel loop collapse(2) present(spec, specw)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2)
#endif
        do m = 1, n_col
            do i = 1, nspec
                specw(i, m) = real(spec(i, m), WP)
            end do
        end do
        !$acc parallel loop gang vector collapse(2) present(ctx, lp, specw, ind)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2)
#endif
        do m = 1, n_col
            do j = 1, n_idx
                ind(j, m) = spectral_index_value(ctx, lp, specw(:, m), j)
            end do
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
        !$acc end data
        indices(1:n_idx, 1:n_col) = ind
    end subroutine compute_spectral_indices_columns

    ! ------------------------------------------------------------------------
    ! PRIVATE HELPERS
    ! ------------------------------------------------------------------------

    !> @brief
    !> Integrates a spectrum over a specific wavelength range.
    !>
    !> @details
    !> Replaces legacy `INTIND`. Finds array indices corresponding to `lo` and `hi`,
    !> linearly interpolates the flux at exact endpoints, and performs trapezoidal
    !> integration between them.
    pure function integrate_interval(lam, func, lo, hi) result(area)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), dimension(:), intent(in), contiguous :: lam, func
        real(WP), intent(in) :: lo, hi
        real(WP) :: area

        integer :: l1, l2
        real(WP) :: f1, f2
        real(WP) :: term1, term2, term_trapz

        ! 1. Locate indices (clamp to array bounds for safety)
        l1 = max(min(find_interval(lam, lo), size(lam) - 1), 1)
        l2 = max(min(find_interval(lam, hi), size(lam) - 1), 1)

        ! 2. Interpolate flux values at exact 'lo' and 'hi'
        f1 = interpolate_linear_point(lam(l1), func(l1), lam(l1+1), func(l1+1), lo)
        f2 = interpolate_linear_point(lam(l2), func(l2), lam(l2+1), func(l2+1), hi)

        ! 3. Integrate
        if (l1 == l2) then
            ! Sub-pixel integration: range is contained within one bin
            area = (f1 + f2) * 0.5_wp * (hi - lo)
        else
            ! Sum full trapezoids between l1+1 and l2
            ! Interior grid points l1+1..l2; a single point (l2 = l1 + 1) contributes no area
            if (l2 > l1 + 1) then
                term_trapz = integrate_trapezoid_array(lam(l1+1:l2), func(l1+1:l2))
            else
                term_trapz = 0.0_wp
            end if
            
            ! Add fractional start triangle
            term1 = (lam(l1+1) - lo) * (f1 + func(l1+1)) * 0.5_wp
            
            ! Add fractional end triangle
            term2 = (hi - lam(l2)) * (f2 + func(l2)) * 0.5_wp
            
            area = term_trapz + term1 + term2
        end if

    end function integrate_interval

    !> @brief
    !> Integrates (Flux / LinearContinuum) over a specific range.
    !>
    !> @details
    !> Used for EW and Mag indices where the spectrum is normalized by a pseudo-continuum
    !> defined by the blue and red sidebands.
    pure function integrate_ratio_interval(lam, func, lo, hi, slope, intercept, x_ref) result(area)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), dimension(:), intent(in), contiguous :: lam, func
        real(WP), intent(in) :: lo, hi
        real(WP), intent(in) :: slope, intercept, x_ref
        real(WP) :: area

        integer :: l1, l2, k
        real(WP) :: r_lo, r_hi
        real(WP) :: current_lam, next_lam, current_flux, next_flux
        real(WP) :: current_cont, next_cont, term_val, next_term_val
        
        ! 1. Locate indices
        l1 = max(min(find_interval(lam, lo), size(lam) - 1), 1)
        l2 = max(min(find_interval(lam, hi), size(lam) - 1), 1)
        
        ! 2-3. Flux/continuum at the endpoints, interpolated from the ratio at the neighbouring grid
        !      points as legacy FSPS (INTIND on spec/fc) does. Interpolating the flux and dividing by
        !      the continuum at the endpoint is no more accurate (neither matches the exact integral of
        !      the piecewise-linear spectrum better) and differed from legacy by up to 0.03 A in
        !      HgammaA on the 20 A BaSeL grid.
        r_lo = interpolate_linear_point(lam(l1), func(l1) / (slope * (lam(l1) - x_ref) + intercept), &
                                        lam(l1+1), func(l1+1) / (slope * (lam(l1+1) - x_ref) + intercept), lo)
        r_hi = interpolate_linear_point(lam(l2), func(l2) / (slope * (lam(l2) - x_ref) + intercept), &
                                        lam(l2+1), func(l2+1) / (slope * (lam(l2+1) - x_ref) + intercept), hi)

        ! 4. Handle sub-pixel vs multi-pixel case
        if (l1 == l2) then
            ! Ratio at lo and hi
            area = (r_lo + r_hi) * 0.5_wp * (hi - lo)
        else
            ! 1. Start with the partial start triangle
            ! Continuum and Flux at grid point l1+1
            current_lam = lam(l1+1)
            current_flux = func(l1+1)
            current_cont = slope * (current_lam - x_ref) + intercept
            term_val = current_flux / current_cont
            
            ! Add the "left" tip
            area = (current_lam - lo) * (r_lo + term_val) * 0.5_wp
            
            ! 2. Accumulate central trapezoids
            ! Iterate through the full grid points inside the range
            do k = l1 + 1, l2 - 1
                next_lam = lam(k+1)
                next_flux = func(k+1)
                next_cont = slope * (next_lam - x_ref) + intercept
                next_term_val = next_flux / next_cont
                
                area = area + (next_lam - current_lam) * (term_val + next_term_val) * 0.5_wp
                
                ! Shift for next iteration
                current_lam = next_lam
                term_val = next_term_val
            end do
            
            ! 3. Add the "right" tip
            area = area + (hi - lam(l2)) * (r_hi + term_val) * 0.5_wp
        end if

    end function integrate_ratio_interval

    !> @brief
    !> Helper: Linear interpolation for a single point.
    pure function interpolate_linear_point(x1, y1, x2, y2, x_target) result(y_target)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: x1, y1, x2, y2, x_target
        real(WP) :: y_target
        
        y_target = (y2 - y1) / (x2 - x1) * (x_target - x1) + y1
    end function interpolate_linear_point

end module fsps_spectral_indices