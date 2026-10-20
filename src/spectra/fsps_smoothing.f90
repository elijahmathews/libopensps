module fsps_smoothing
    !> @brief
    !> Routines for velocity and wavelength smoothing of stellar spectra.
    !>
    !> @details
    !> This module handles the convolution of spectra with velocity dispersion
    !> (Gaussian broadening) or instrumental resolution curves. It supports
    !> three distinct smoothing modes:
    !> 1. Log-Linear Convolution: High accuracy for constant velocity dispersion.
    !> 2. Variable Resolution: Direct integration for wavelength-dependent broadening.
    !> 3. Constant Wavelength: Simple Gaussian smoothing in linear wavelength space.

    use fsps_precision, only: WP
    use fsps_constants, only: SAFE_FLOOR, C_LIGHT, PI
    use fsps_context_types, only: fsps_context_t

    implicit none
    private

    public :: apply_smoothing
    public :: smooth_velocity
    public :: velocity_kernel_t, build_velocity_kernel, apply_velocity_kernel

    ! --------------------------------------------------------------------------
    ! CONSTANTS
    ! --------------------------------------------------------------------------

    real(WP), parameter :: N_SIGMA_RANGE = 4.0_wp !> Range of kernel in sigmas
    real(WP), parameter :: INV_SQRT_2PI = 1.0_wp / sqrt(2.0_wp * PI) !> 1/sqrt(2π)

    !> Sparse, fixed-sigma velocity smoothing operator on a wavelength grid: row i (if active) is
    !> smoothed(i) = sum(coef(off(i):off(i)+hi(i)-lo(i)) * spec(lo(i):hi(i))). The weights depend
    !> only on the grid, so applying the operator to many spectra avoids recomputing them.
    type :: velocity_kernel_t
        integer, allocatable :: lo(:), hi(:), off(:)   ! lo(i) = 0: pixel i is left unchanged
        real(WP), allocatable :: coef(:)
    end type velocity_kernel_t

contains

    !> @brief
    !> Main entry point for spectral smoothing.
    !>
    !> @details
    !> Dispatches the calculation to the appropriate private kernel based on
    !> the context settings and input arguments.
    !>
    !> **Logic Flow:**
    !> 1. If `smooth_velocity` is false: Uses Constant Wavelength (Linear) smoothing.
    !> 2. If `smooth_velocity` is true:
    !>    a. If `ires` is present OR `smoothspec_fast` is true: Uses Variable Resolution.
    !>    b. Otherwise: Uses Log-Linear (FFT-like) convolution.
    !>
    !> @param[in]     ctx    FSPS context containing smoothing configuration flags.
    !> @param[in]     lambda Wavelength grid (Angstroms).
    !> @param[in,out] spec   Flux array (L_sun/Hz or L_sun/Ang). Modified in-place.
    !> @param[in]     sigma  Smoothing width (km/s if velocity mode, Angstroms otherwise).
    !> @param[in]     minl   Minimum wavelength to smooth.
    !> @param[in]     maxl   Maximum wavelength to smooth.
    !> @param[in]     ires   (Optional) Instrumental resolution (sigma in km/s per pixel).
    subroutine apply_smoothing(ctx, lambda, spec, sigma, minl, maxl, ires)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(:), intent(in), contiguous :: lambda
        real(WP), dimension(:), intent(inout), contiguous :: spec
        real(WP), intent(in) :: sigma
        real(WP), intent(in) :: minl
        real(WP), intent(in) :: maxl
        real(WP), dimension(:), intent(in), optional, contiguous :: ires
        
        ! Local flags
        logical :: use_velocity_smoothing
        logical :: use_fast_mode
        logical :: has_ires
        integer :: n

        ! ------------------------------------------------------------------------
        ! 1. PRELIMINARY CHECKS
        ! ------------------------------------------------------------------------
        n = size(lambda)
        if (n < 2) return

        has_ires = present(ires)

        ! Optimization: Legacy code returned if sigma <= SAFE_FLOOR.
        ! We only return if sigma is zero AND we don't have an ires array.
        ! If ires is present, we proceed (assuming ires values > 0).
        if (sigma <= SAFE_FLOOR .and. .not. has_ires) return

        ! Extract flags from context for readability
        ! Legacy mapping: smooth_velocity=1 -> True
        use_velocity_smoothing = (ctx%smooth_velocity_val == 1)
        use_fast_mode          = (ctx%smoothspec_fast_val == 1)

        ! ------------------------------------------------------------------------
        ! 2. DISPATCH LOGIC
        ! ------------------------------------------------------------------------
        
        if (use_velocity_smoothing) then
            
            ! Case A: Variable Resolution (Fast Mode OR IRES)
            ! We force this mode if IRES is present because log-linear cannot 
            ! handle wavelength-dependent sigma.
            if (use_fast_mode .or. has_ires) then
                
                call convolve_variable_resolution(lambda, spec, sigma, minl, maxl, ires)
            
            ! Case B: Constant Velocity (High Accuracy / Slow Mode)
            ! Uses log-linear resampling + constant kernel
            else
                
                call convolve_log_linear(lambda, spec, sigma, minl, maxl)
                
            end if

        else
            
            ! Case C: Constant Wavelength Smoothing
            ! Sigma is treated as Angstroms here, not km/s.
            call convolve_constant_lambda(lambda, spec, sigma, minl, maxl)

        end if

    end subroutine apply_smoothing

    !> @brief
    !> Gaussian smoothing in velocity (sigma in km/s) by direct integration.
    !>
    !> @details
    !> The velocity path of apply_smoothing (smooth_velocity = 1, smoothspec_fast = 1), for
    !> internal uses that must not depend on the user's output-smoothing settings. Only pixels
    !> with minl <= lambda <= maxl change.
    subroutine smooth_velocity(lambda, spec, sigma, minl, maxl)
        real(WP), dimension(:), intent(in), contiguous :: lambda
        real(WP), dimension(:), intent(inout), contiguous :: spec
        real(WP), intent(in) :: sigma, minl, maxl

        if (size(lambda) < 2 .or. sigma <= SAFE_FLOOR) return
        call convolve_variable_resolution(lambda, spec, sigma, minl, maxl)
    end subroutine smooth_velocity

    ! ------------------------------------------------------------------------
    ! PRIVATE HELPER KERNELS
    ! ------------------------------------------------------------------------

    !> @brief
    !> smooth_velocity=1 (Correct/Slow Mode)
    !>
    !> @details
    !> Resamples the spectrum onto a logarithmic wavelength grid where velocity
    !> shifts are constant, convolves with a Gaussian kernel, and interpolates back.
    !>
    !> **Optimizations:**
    !> 1. Uses "hunting" interpolation (sliding index) for resampling, exploiting
    !>    the sorted nature of wavelength arrays ($O(N)$ instead of $O(N \log N)$).
    !> 2. Avoids allocating the `lnlam` grid array, calculating coordinates on the fly.
    !> 3. Uses `dot_product` for SIMD-friendly convolution.
    !>
    !> @param[in]     lambda   Wavelength grid (Angstroms).
    !> @param[in,out] spec     Flux array.
    !> @param[in]     sigma    Velocity dispersion in km/s.
    !> @param[in]     minl     Min wavelength for smoothing window.
    !> @param[in]     maxl     Max wavelength for smoothing window.
    subroutine convolve_log_linear(lambda, spec, sigma, minl, maxl)
        real(WP), dimension(:), intent(in), contiguous :: lambda
        real(WP), dimension(:), intent(inout), contiguous :: spec
        real(WP), intent(in) :: sigma
        real(WP), intent(in) :: minl
        real(WP), intent(in) :: maxl

        ! Constants
        real(WP) :: c_kms
        
        ! Grid parameters
        integer :: n_pix, i
        real(WP) :: ln_min, ln_max, d_ln, inv_d_ln
        real(WP) :: ln_val

        ! Kernel parameters
        real(WP) :: sigma_ln, sigma_pix, w, inv_norm, sum_kernel
        integer :: k_rad, k_width, k_start, k_end
        real(WP), allocatable :: kernel(:)

        ! Data buffers
        real(WP), allocatable :: flux_log(:), conv_log(:)
        
        ! Interpolation state (Hunting indices)
        integer :: idx_hunt

        ! --------------------------------------------------------------------
        ! 1. SETUP & VALIDATION
        ! --------------------------------------------------------------------
        if (sigma <= SAFE_FLOOR) return
        
        n_pix = size(lambda)
        if (n_pix < 2) return

        ! Speed of light in km/s
        c_kms = C_LIGHT * 1.0e-13_wp

        ! Define Log-Linear Grid (Implicitly)
        ! We align the grid to the requested min/max range
        ln_min = log(max(minl, lambda(1)))
        ln_max = log(min(maxl, lambda(n_pix)))
        
        ! We keep the same number of pixels to maintain sampling density roughly
        d_ln = (ln_max - ln_min) / real(n_pix - 1, WP)
        inv_d_ln = 1.0_wp / d_ln

        ! --------------------------------------------------------------------
        ! 2. PREPARE KERNEL
        ! --------------------------------------------------------------------
        ! Sigma in log-space: d(lnlam) = dlam/lam = v/c
        sigma_ln = sigma / c_kms
        
        ! Sigma in pixels
        sigma_pix = sigma_ln * inv_d_ln
        
        ! Kernel radius (integer pixels)
        k_rad = ceiling(N_SIGMA_RANGE * sigma_pix)
        k_width = 2 * k_rad + 1
        
        allocate(kernel(-k_rad:k_rad))
        
        ! Construct Gaussian Kernel
        ! Exp(-0.5 * (x/sigma)^2)
        w = -0.5_wp / (sigma_pix**2)
        do i = -k_rad, k_rad
            kernel(i) = exp(w * real(i, WP)**2)
        end do
        
        ! Normalize kernel to conserve energy
        sum_kernel = sum(kernel)
        if (sum_kernel <= SAFE_FLOOR) then
            ! Fallback for numerical underflow (effectively a delta function)
            kernel = 0.0_wp
            kernel(0) = 1.0_wp
        else
            inv_norm = 1.0_wp / sum_kernel
            kernel = kernel * inv_norm
        endif

        ! --------------------------------------------------------------------
        ! 3. FORWARD INTERPOLATION (Linear -> Log)
        ! --------------------------------------------------------------------
        ! Resample input spec onto the uniform log grid.
        ! We use a "hunting" loop because both grids are sorted.
        
        allocate(flux_log(n_pix))
        
        idx_hunt = 1 ! Initialize hunter
        
        do i = 1, n_pix
            ln_val = ln_min + (i - 1) * d_ln
            
            ! Find bracket [idx_hunt, idx_hunt+1] for exp(ln_val)
            ! Since ln_val increases, idx_hunt only moves forward.
            call hunt_interval(lambda, exp(ln_val), idx_hunt)
            
            ! Linear Interpolation inline
            ! y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            flux_log(i) = interpolate_linear_scalar( &
                lambda(idx_hunt), lambda(idx_hunt+1), &
                spec(idx_hunt),   spec(idx_hunt+1),   &
                exp(ln_val))
        end do

        ! --------------------------------------------------------------------
        ! 4. CONVOLUTION
        ! --------------------------------------------------------------------
        allocate(conv_log(n_pix))
        conv_log = 0.0_wp ! Initialize
        
        ! Inner area where full kernel fits
        ! We use dot_product for vectorization
        do i = 1, n_pix
            
            ! Determine valid kernel overlap
            k_start = max(-k_rad, 1 - i)
            k_end   = min(k_rad, n_pix - i)
            
            ! If we are too close to edges (less than 1 sigma), just copy?
            ! Original code logic: "the ends are not smoothed"
            ! We will strictly convolve what we can.
            
            if (k_start > k_end) then
                conv_log(i) = flux_log(i)
            else
                ! The slice flux_log(i+k_start : i+k_end) aligns with kernel(k_start : k_end)
                conv_log(i) = dot_product( &
                    kernel(k_start:k_end), &
                    flux_log(i+k_start : i+k_end) &
                )
                
                ! Renormalize if edges were clipped (optional, depends on physics preference)
                ! Ideally, we re-normalize the weights used. 
                ! For strict FSPS matching, we often leave edges alone or accept flux loss.
                ! Here we adopt the robust approach: Re-normalize weights.
                if (k_end - k_start + 1 < k_width) then
                    conv_log(i) = conv_log(i) / sum(kernel(k_start:k_end))
                end if
            end if
        end do

        ! --------------------------------------------------------------------
        ! 5. BACKWARD INTERPOLATION (Log -> Linear) & UPDATE
        ! --------------------------------------------------------------------
        ! Interpolate from conv_log back to spec.
        ! Update spec ONLY within the minl/maxl range.
        
        idx_hunt = 1 ! Reset hunter
        
        do i = 1, n_pix
            ! Check bounds
            if (lambda(i) < minl .or. lambda(i) > maxl) cycle
            
            ! Map current lambda to log grid coordinate (continuous index)
            ! coord = 1 + (ln(lambda) - ln_min) / d_ln
            ln_val = log(lambda(i))
            
            ! We effectively hunt in the implicit log grid (which is just 1..N)
            ! But to be generic, we use the logic:
            ! grid pos = (ln_val - ln_min) * inv_d_ln + 1.0
            
            w = (ln_val - ln_min) * inv_d_ln + 1.0_wp

            ! Robust interpolation indices
            k_start = int(w) ! Lower index
            w = w - real(k_start, WP) ! Fraction
            
            ! Clamp to safe bounds [1, n_pix-1]
            if (k_start < 1) then 
                k_start = 1; w = 0.0_wp
            elseif (k_start >= n_pix) then 
                k_start = n_pix - 1; w = 1.0_wp
            endif
            
            spec(i) = (1.0_wp - w) * conv_log(k_start) + w * conv_log(k_start + 1)
        end do

    end subroutine convolve_log_linear

    !> @brief
    !> smooth_velocity=1 (Fast Mode or IRES)
    !>
    !> @details
    !> Performs a direct convolution integral allowing for wavelength-dependent
    !> broadening widths. This handles both the "Fast" velocity smoothing (where
    !> sigma_lambda ~ lambda) and explicit instrumental resolution (ires).
    !>
    !> **Optimizations:**
    !> 1. Allocates scratch buffers *once* to avoid allocation overhead inside the loop.
    !> 2. Uses a bi-directional "hunter" for integration bounds, adapting efficiently
    !>    to changing sigma widths ($O(N)$ for smooth variation).
    !> 3. Fixes legacy approximation: explicitly finds the left wavelength bound
    !>    instead of assuming symmetric indices (`2*i - ih`).
    !>
    !> @param[in]     lambda       Wavelength grid (Angstroms).
    !> @param[in,out] spec         Flux array.
    !> @param[in]     global_sigma Global velocity dispersion (km/s). Used if ires missing.
    !> @param[in]     minl         Min wavelength to smooth.
    !> @param[in]     maxl         Max wavelength to smooth.
    !> @param[in]     ires         (Optional) Velocity dispersion per pixel (km/s).
    subroutine convolve_variable_resolution(lambda, spec, global_sigma, minl, maxl, ires)
        real(WP), dimension(:), intent(in), contiguous :: lambda
        real(WP), dimension(:), intent(inout), contiguous :: spec
        real(WP), intent(in) :: global_sigma
        real(WP), intent(in) :: minl
        real(WP), intent(in) :: maxl
        real(WP), dimension(:), intent(in), optional, contiguous :: ires

        ! Constants
        real(WP) :: c_kms

        ! Loop & Index variables
        integer :: i, n, il, ih, n_window
        real(WP) :: val_l, val_h

        ! Physics variables
        real(WP) :: sig_kms, sig_beta, inv_2sigma2, sig_beta_global, inv_2sigma2_global
        real(WP) :: total_weight, total_flux

        ! Pre-computed bounds factors
        real(WP) :: bound_factor_blue, bound_factor_red

        ! Memory buffers
        real(WP), allocatable :: d_lam(:)
        real(WP), allocatable :: spec_smooth(:)
        real(WP), allocatable :: scratch_weights(:)
        real(WP), allocatable :: log_lambda(:)

        ! Flags
        logical :: use_ires

        ! --------------------------------------------------------------------
        ! 1. SETUP & ALLOCATION
        ! --------------------------------------------------------------------
        n = size(lambda)
        if (n < 2) return

        ! Inverse speed of light for converting km/s -> dimensionless z
        ! c_kms = 2.9979e5
        c_kms = C_LIGHT * 1.0e-13_wp

        use_ires = present(ires)

        ! Allocate Output & Pre-calc Grid Differences
        allocate(spec_smooth(n))
        spec_smooth = spec

        allocate(d_lam(n-1))
        d_lam = lambda(2:n) - lambda(1:n-1)

        ! Allocate SCRATCH buffer for the vectorization
        ! Size N guarantees we never overflow, avoiding re-allocation in loop.
        allocate(scratch_weights(n))

        ! Pre-calculate Log Lambda to speed up loop and match physics
        allocate(log_lambda(n))
        log_lambda = log(lambda)

        ! Initialize Window Indices
        il = 1
        ih = 1

        ! Pre-calculate global constant
        if (.not. use_ires) then
            sig_beta_global = global_sigma / c_kms
            inv_2sigma2_global = 0.5_wp * (1.0_wp / sig_beta_global)**2

            ! Pre-calculate bounds factors (Exp for log-space symmetry)
            ! range = 4 sigma. Log_shift = 4 * (v/c)
            bound_factor_blue = exp(-N_SIGMA_RANGE * sig_beta_global)
            bound_factor_red  = exp( N_SIGMA_RANGE * sig_beta_global)
        end if

        ! --------------------------------------------------------------------
        ! 2. CONVOLUTION LOOP
        ! --------------------------------------------------------------------
        do i = 1, n

            if (lambda(i) < minl .or. lambda(i) > maxl) cycle

            ! A. Determine Sigma
            if (use_ires) then
                sig_kms = ires(i)
                if (sig_kms <= SAFE_FLOOR) cycle
                
                sig_beta = sig_kms / c_kms
                inv_2sigma2 = 0.5_wp * (1.0_wp / sig_beta)**2

                bound_factor_blue = exp(-N_SIGMA_RANGE * sig_beta)
                bound_factor_red  = exp( N_SIGMA_RANGE * sig_beta)
            else
                ! Optimization: Uses pre-calculated globals
                sig_beta = sig_beta_global
                inv_2sigma2 = inv_2sigma2_global
            end if

            ! B. Define Relativistic Window Bounds
            ! In Log-Space: ln(lam') = ln(lam) +/- range
            ! In Linear:    lam'     = lam * exp(+/- range)
            val_l = lambda(i) * bound_factor_blue
            val_h = lambda(i) * bound_factor_red

            ! C. Update Indices (The Hunter)
            ! ----------------------------------------------
            ! Fix Lower Bound (il)
            do while (il > 1 .and. lambda(il) > val_l)
                il = il - 1
            end do
            do while (il < n .and. lambda(il) < val_l)
                il = il + 1
            end do
            if (il > 1 .and. lambda(il) > val_l) il = il - 1

            ! Fix Upper Bound (ih)
            if (ih < il) ih = il
            do while (ih > il .and. lambda(ih) > val_h)
                ih = ih - 1
            end do
            do while (ih < n .and. lambda(ih) < val_h)
                ih = ih + 1
            end do

            n_window = ih - il + 1
            if (n_window < 2) cycle

            ! 1. Calculate Weights (Log-Normal Shape)
            ! Exponent = -0.5 * ((ln(lam_i) - ln(lam_j)) / (v/c))^2
            ! Using pre-calculated subtraction is FASTER than division.
            
            scratch_weights(1:n_window) = &
                exp( -inv_2sigma2 * ((log_lambda(i) - log_lambda(il:ih))**2) )
            
            scratch_weights(1:n_window) = scratch_weights(1:n_window) / lambda(il:ih)

            ! 2. Integrate Weights
            total_weight = 0.5_wp * dot_product( &
                scratch_weights(1:n_window-1) + scratch_weights(2:n_window), &
                d_lam(il:ih-1) &
            )

            if (total_weight <= SAFE_FLOOR) cycle

            ! 3. Integrate Flux
            scratch_weights(1:n_window) = scratch_weights(1:n_window) * spec(il:ih)

            total_flux = 0.5_wp * dot_product( &
                scratch_weights(1:n_window-1) + scratch_weights(2:n_window), &
                d_lam(il:ih-1) &
            )

            spec_smooth(i) = total_flux / total_weight

        end do

        ! --------------------------------------------------------------------
        ! 3. FINALIZE
        ! --------------------------------------------------------------------
        spec = spec_smooth

    end subroutine convolve_variable_resolution

    !> @brief
    !> Builds the operator applied by convolve_variable_resolution for a fixed sigma (no ires):
    !> the same window (+-N_SIGMA_RANGE sigma in ln lambda), Gaussian weights in ln lambda divided
    !> by lambda, trapezoid rule over the grid, and normalization. apply_velocity_kernel then
    !> reproduces convolve_variable_resolution to rounding (tested in test_fsps_smoothing).
    subroutine build_velocity_kernel(lambda, sigma, minl, maxl, kernel)
        real(WP), dimension(:), intent(in), contiguous :: lambda
        real(WP), intent(in) :: sigma, minl, maxl
        type(velocity_kernel_t), intent(out) :: kernel

        integer :: i, m, n, il, ih, nw, n_total
        integer, allocatable :: win_lo(:), win_hi(:)
        real(WP) :: sig_beta, inv_2sigma2, factor_blue, factor_red, val_l, val_h, total_weight, d_left, d_right
        real(WP), allocatable :: w(:)

        n = size(lambda)
        allocate(kernel%lo(n), kernel%hi(n), kernel%off(n), win_lo(n), win_hi(n), w(n))
        kernel%lo = 0
        kernel%hi = -1
        kernel%off = 0
        win_lo = 0
        win_hi = -1
        if (n < 2 .or. sigma <= SAFE_FLOOR) then
            allocate(kernel%coef(0))
            return
        end if

        sig_beta = sigma / (C_LIGHT * 1.0e-13_wp)
        inv_2sigma2 = 0.5_wp * (1.0_wp / sig_beta)**2
        factor_blue = exp(-N_SIGMA_RANGE * sig_beta)
        factor_red = exp(N_SIGMA_RANGE * sig_beta)

        ! Pass 1: windows (the same hunting search as convolve_variable_resolution)
        il = 1
        ih = 1
        n_total = 0
        do i = 1, n
            if (lambda(i) < minl .or. lambda(i) > maxl) cycle
            val_l = lambda(i) * factor_blue
            val_h = lambda(i) * factor_red
            do while (il > 1 .and. lambda(il) > val_l)
                il = il - 1
            end do
            do while (il < n .and. lambda(il) < val_l)
                il = il + 1
            end do
            if (il > 1 .and. lambda(il) > val_l) il = il - 1
            if (ih < il) ih = il
            do while (ih > il .and. lambda(ih) > val_h)
                ih = ih - 1
            end do
            do while (ih < n .and. lambda(ih) < val_h)
                ih = ih + 1
            end do
            if (ih - il + 1 < 2) cycle
            win_lo(i) = il
            win_hi(i) = ih
            n_total = n_total + (ih - il + 1)
        end do

        ! Pass 2: normalized trapezoid coefficients
        allocate(kernel%coef(n_total))
        n_total = 0
        do i = 1, n
            if (win_lo(i) == 0) cycle
            il = win_lo(i)
            ih = win_hi(i)
            nw = ih - il + 1
            w(1:nw) = exp(-inv_2sigma2 * ((log(lambda(i)) - log(lambda(il:ih)))**2)) / lambda(il:ih)
            total_weight = 0.0_wp
            do m = 1, nw
                d_left = 0.0_wp
                d_right = 0.0_wp
                if (m > 1) d_left = lambda(il + m - 1) - lambda(il + m - 2)
                if (m < nw) d_right = lambda(il + m) - lambda(il + m - 1)
                w(m) = 0.5_wp * w(m) * (d_left + d_right)
                total_weight = total_weight + w(m)
            end do
            if (total_weight <= SAFE_FLOOR) cycle
            kernel%lo(i) = il
            kernel%hi(i) = ih
            kernel%off(i) = n_total + 1
            kernel%coef(n_total + 1:n_total + nw) = w(1:nw) / total_weight
            n_total = n_total + nw
        end do
    end subroutine build_velocity_kernel

    !> @brief Applies a velocity_kernel_t to the pixels i_first..size(spec) of a spectrum.
    subroutine apply_velocity_kernel(kernel, spec, i_first, smoothed)
        type(velocity_kernel_t), intent(in) :: kernel
        real(WP), dimension(:), intent(in) :: spec
        integer, intent(in) :: i_first
        real(WP), dimension(:), intent(out) :: smoothed   ! size(spec) - i_first + 1
        integer :: i

        do i = i_first, size(spec)
            if (kernel%lo(i) == 0) then
                smoothed(i - i_first + 1) = spec(i)
            else
                smoothed(i - i_first + 1) = dot_product(kernel%coef(kernel%off(i):kernel%off(i) + kernel%hi(i) - &
                                                                    kernel%lo(i)), spec(kernel%lo(i):kernel%hi(i)))
            end if
        end do
    end subroutine apply_velocity_kernel

    !> @brief
    !> smooth_velocity=0 (Constant Wavelength Dispersion)
    !>
    !> @details
    !> Smoothes the spectrum by a constant wavelength dispersion (sigma in Angstroms).
    !>
    !> **Optimizations:**
    !> 1. Uses a "sliding window" approach to find integration bounds, reducing
    !>    complexity from O(N log N) to O(N).
    !> 2. Inlines the trapezoidal integration to allow for compiler vectorization
    !>    and avoid function call overhead inside the loop.
    !> 3. Pre-calculates wavelength differences (d_lambda) to speed up integration.
    !>
    !> @param[in]     lambda   Wavelength grid (Angstroms).
    !> @param[in,out] spec     Flux array.
    !> @param[in]     sigma    Dispersion width in Angstroms.
    !> @param[in]     minl     Min wavelength to smooth.
    !> @param[in]     maxl     Max wavelength to smooth.
    subroutine convolve_constant_lambda(lambda, spec, sigma, minl, maxl)
        real(WP), dimension(:), intent(in), contiguous :: lambda
        real(WP), dimension(:), intent(inout), contiguous :: spec
        real(WP), intent(in) :: sigma
        real(WP), intent(in) :: minl
        real(WP), intent(in) :: maxl

        ! Loop variables
        integer :: i, n, il, ih, n_window
        real(WP) :: val_l, val_h

        ! Kernel variables
        real(WP) :: inv_sigma, inv_2sigma2
        real(WP) :: total_weight, total_flux
        
        ! Buffers
        real(WP), allocatable :: spec_smooth(:)
        real(WP), allocatable :: d_lam(:)
        real(WP), allocatable :: scratch_weights(:)

        ! --------------------------------------------------------------------
        ! 1. SETUP
        ! --------------------------------------------------------------------
        if (sigma <= SAFE_FLOOR) return
        n = size(lambda)
        if (n < 2) return

        allocate(spec_smooth(n))
        spec_smooth = spec ! Default: copy original

        ! Pre-calculate grid spacing for fast trapezoid integration
        ! d_lam(j) = lambda(j+1) - lambda(j)
        allocate(d_lam(n-1))
        d_lam = lambda(2:n) - lambda(1:n-1)

        ! Allocating once outside the loop
        allocate(scratch_weights(n))

        ! Pre-calculate kernel constants
        inv_sigma = 1.0_wp / sigma
        inv_2sigma2 = 0.5_wp * (inv_sigma**2)

        ! Initialize sliding window indices
        il = 1
        ih = 1
        
        ! --------------------------------------------------------------------
        ! 2. CONVOLUTION LOOP
        ! --------------------------------------------------------------------
        do i = 1, n
            
            ! Skip if outside requested range
            if (lambda(i) < minl .or. lambda(i) > maxl) cycle

            ! A. UPDATE SLIDING WINDOW (The "Hunter")
            ! Define bounds: lambda[i] +/- 4*sigma
            val_l = lambda(i) - N_SIGMA_RANGE * sigma
            val_h = lambda(i) + N_SIGMA_RANGE * sigma

            ! Advance 'il' until lambda(il) >= val_l
            do while (il < n .and. lambda(il) < val_l)
                il = il + 1
            end do
            ! Correct if we overshot (ensure we include the bracket)
            if (il > 1 .and. lambda(il) > val_l) il = max(1, il - 1)

            ! Advance 'ih' until lambda(ih) >= val_h
            do while (ih < n .and. lambda(ih) < val_h)
                ih = ih + 1
            end do
            
            n_window = ih - il + 1

            ! B. INTEGRATION
            ! If window is too small (e.g. edge of grid or sub-pixel sigma), skip
            if (n_window < 2) cycle

            ! 1. Calculate Gaussian Weights
            ! Vectorized Integration using Scratch Buffer
            scratch_weights(1:n_window) = exp( -((lambda(il:ih) - lambda(i))**2) * inv_2sigma2 )
            
            ! 2. Integrate Weights (Normalization Factor)
            ! Trapezoid: sum( 0.5 * (w(j) + w(j+1)) * d_lam(j) )
            ! Vectorized using dot_product
            total_weight = 0.5_wp * dot_product( &
                scratch_weights(1:n_window-1) + scratch_weights(2:n_window), &
                d_lam(il:ih-1) &
            )

            ! Handle underflow/zero weight
            if (total_weight <= SAFE_FLOOR) cycle

            ! 3. Integrate Flux * Weights
            ! flux_w = spec * w
            ! result = sum( 0.5 * (fw(j) + fw(j+1)) * d_lam(j) )
            
            scratch_weights(1:n_window) = scratch_weights(1:n_window) * spec(il:ih)
            
            total_flux = 0.5_wp * dot_product( &
                scratch_weights(1:n_window-1) + scratch_weights(2:n_window), &
                d_lam(il:ih-1) &
            )

            ! C. STORE RESULT
            spec_smooth(i) = total_flux / total_weight

        end do

        ! --------------------------------------------------------------------
        ! 3. FINALIZE
        ! --------------------------------------------------------------------
        spec = spec_smooth

    end subroutine convolve_constant_lambda

    !> @brief Finds interval in a sorted array such that arr(i) <= val < arr(i+1)
    !> @details Optimized "Hunter" that starts search from a guess index.
    pure subroutine hunt_interval(arr, val, idx)
        real(WP), dimension(:), intent(in) :: arr
        real(WP), intent(in) :: val
        integer, intent(inout) :: idx ! Input: guess, Output: result
        
        integer :: n
        n = size(arr)
        
        ! 1. Check if we can simply step forward (most common case in sequential access)
        if (idx < n) then
            if (val >= arr(idx) .and. val < arr(idx+1)) return
        endif
        
        ! 2. Step forward loop
        do while (idx < n - 1)
            if (val < arr(idx+1)) exit
            idx = idx + 1
        end do
        
        ! 3. Safety Check (if val went backwards or is out of bounds)
        ! If val is smaller than current, we might need to search backwards or reset.
        ! For this specific routine, we assume monotonic forward processing.
        ! But for safety:
        if (val < arr(idx)) then
           ! simple linear scan backwards (rare in this context)
           do while (idx > 1)
              if (val >= arr(idx)) exit
              idx = idx - 1
           end do
        endif
        
        ! Clamp
        if (idx < 1) idx = 1
        if (idx >= n) idx = n - 1
        
    end subroutine hunt_interval

    !> @brief Simple scalar linear interpolation
    pure function interpolate_linear_scalar(x1, x2, y1, y2, x) result(y)
        real(WP), intent(in) :: x1, x2, y1, y2, x
        real(WP) :: y
        y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    end function interpolate_linear_scalar

end module fsps_smoothing