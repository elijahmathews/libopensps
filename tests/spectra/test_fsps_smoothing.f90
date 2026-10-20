module test_fsps_smoothing_mod
    use fsps_precision, only: WP
    use fsps_constants, only: C_LIGHT, SAFE_FLOOR
    use fsps_context_types, only: fsps_context_t
    use fsps_smoothing, only: apply_smoothing, smooth_velocity, velocity_kernel_t, build_velocity_kernel, &
                              apply_velocity_kernel
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_relative_error
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_smoothing_tests, total_failures, total_tests

contains

    subroutine run_fsps_smoothing_tests()
        call print_minor_header("fsps_smoothing")

        call test_instrumental_override()
        call test_angstrom_mode_switch()
        call test_zero_sigma_optimization()

        call test_relativistic_asymmetry()
        call test_log_linear_accuracy()
        call test_method_convergence()

        call test_zig_zag_sigma()
        call test_sparse_grid_subpixel_sigma()

        call test_flat_field_identity()
        call test_log_linear_edge_renormalization()
        call test_window_cut_passthrough()
        call test_velocity_kernel_matches_convolution()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_smoothing_tests

    ! ------------------------------------------------------------------------
    ! Group 1: Architectural Dispatch & Flag Logic
    ! ------------------------------------------------------------------------
    subroutine test_instrumental_override()
        type(fsps_context_t), allocatable :: ctx_slow, ctx_fast
        real(WP), allocatable :: lambda(:), spec_in(:), spec_a(:), spec_b(:), ires(:)
        real(WP) :: minl, maxl, sigma
        integer :: n, i
        real(WP) :: max_diff

        call print_group("Instrumental Override (IRES forces Variable Resolution)")

        n = 1200
        allocate(lambda(n), spec_in(n), spec_a(n), spec_b(n), ires(n))
        call fill_linear_grid(lambda, 3500.0_wp, 2.0_wp)
        minl = lambda(1)
        maxl = lambda(n)

        do i = 1, n
            spec_in(i) = 1.0_wp + 0.2_wp * sin((lambda(i) - 3500.0_wp) / 40.0_wp)
            ires(i) = 120.0_wp + 20.0_wp * sin((lambda(i) - 3500.0_wp) / 100.0_wp)
        end do

        sigma = 150.0_wp

        allocate(ctx_slow, ctx_fast)
        call setup_context(ctx_slow, smooth_velocity=.true., smoothspec_fast=.false.)
        call setup_context(ctx_fast, smooth_velocity=.true., smoothspec_fast=.true.)

        spec_a = spec_in
        spec_b = spec_in
        call apply_smoothing(ctx_slow, lambda, spec_a, sigma, minl, maxl, ires)
        call apply_smoothing(ctx_fast, lambda, spec_b, sigma, minl, maxl, ires)

        max_diff = maxval(abs(spec_a - spec_b))
        call assert_true(max_diff < 1.0e-8_wp, "IRES forces Variable Resolution path", total_tests, total_failures)

        deallocate(lambda, spec_in, spec_a, spec_b, ires, ctx_slow, ctx_fast)
    end subroutine test_instrumental_override

    subroutine test_angstrom_mode_switch()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: minl, maxl, sigma, fwhm, blue_hwhm, red_hwhm
        integer :: n, peak_idx

        call print_group("Angstrom Mode Switch (smooth_velocity=0)")

        n = 801
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 4800.0_wp, 0.5_wp)
        minl = lambda(1)
        maxl = lambda(n)

        spec = 0.0_wp
        peak_idx = find_nearest_index(lambda, 5000.0_wp)
        spec(peak_idx) = 1.0_wp

        sigma = 10.0_wp
        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.false., smoothspec_fast=.false.)
        call apply_smoothing(ctx, lambda, spec, sigma, minl, maxl)

        call compute_fwhm(lambda, spec, fwhm, blue_hwhm, red_hwhm)

        call assert_true(fwhm > 10.0_wp, "FWHM not in km/s regime", total_tests, total_failures)
        call assert_float_equals(2.355_wp * sigma, fwhm, 2.0_wp, "FWHM ~ 2.355*sigma (Angstrom)", total_tests, total_failures)

        deallocate(lambda, spec, ctx)
    end subroutine test_angstrom_mode_switch

    subroutine test_zero_sigma_optimization()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec(:), spec_orig(:)
        real(WP) :: minl, maxl
        integer :: n

        call print_group("Zero-Sigma Optimization")

        n = 400
        allocate(lambda(n), spec(n), spec_orig(n))
        call fill_linear_grid(lambda, 4000.0_wp, 1.0_wp)
        minl = lambda(1)
        maxl = lambda(n)

        spec = 1.0_wp + 0.1_wp * cos((lambda - 4000.0_wp) / 25.0_wp)
        spec_orig = spec

        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.true., smoothspec_fast=.false.)
        call apply_smoothing(ctx, lambda, spec, 0.0_wp, minl, maxl)

        call assert_true(all(spec == spec_orig), "Spec unchanged for sigma <= SAFE_FLOOR", total_tests, total_failures)

        deallocate(lambda, spec, spec_orig, ctx)
    end subroutine test_zero_sigma_optimization

    ! ------------------------------------------------------------------------
    ! Group 2: Physics of Broadening (Relativistic Skew)
    ! ------------------------------------------------------------------------
    subroutine test_relativistic_asymmetry()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: minl, maxl, sigma_kms
        real(WP) :: fwhm, blue_hwhm, red_hwhm
        real(WP) :: c_kms, lambda0, expected_blue, expected_red
        real(WP) :: peak_val, threshold, sig_beta
        real(WP) :: blue_extent, red_extent
        integer :: n, i, peak_idx

        call print_group("Relativistic Asymmetry (Log-Normal Physics)")

        n = 1501
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 1000.0_wp, 2.0_wp)
        minl = lambda(1)
        maxl = lambda(n)

        lambda0 = 2000.0_wp
        peak_idx = find_nearest_index(lambda, lambda0)
        spec = 0.0_wp
        spec(peak_idx) = 1.0_wp

        sigma_kms = 30000.0_wp
        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.true., smoothspec_fast=.true.)
        call apply_smoothing(ctx, lambda, spec, sigma_kms, minl, maxl)

        call compute_fwhm(lambda, spec, fwhm, blue_hwhm, red_hwhm)
        
        ! Fast mode now matches Log-Linear physics:
        ! A symmetric Gaussian in Log-Lambda is ASYMMETRIC in Linear Lambda.
        call assert_true(red_hwhm > blue_hwhm, "Red HWHM > Blue HWHM (Log-Normal)", total_tests, total_failures)

        c_kms = C_LIGHT * 1.0e-13_wp
        sig_beta = sigma_kms / c_kms

        ! Bounds in Log-Normal physics:
        ! lambda_limit = lambda0 * exp( +/- 4 * v/c )
        expected_blue = lambda0 * exp(-4.0_wp * sig_beta)
        expected_red  = lambda0 * exp( 4.0_wp * sig_beta)

        peak_val = maxval(spec)
        threshold = max(peak_val * 1.0e-4_wp, 1.0e-12_wp)

        blue_extent = lambda0
        red_extent = lambda0
        do i = 1, n
            if (spec(i) > threshold) then
                blue_extent = lambda(i)
                exit
            end if
        end do
        do i = n, 1, -1
            if (spec(i) > threshold) then
                red_extent = lambda(i)
                exit
            end if
        end do

        call assert_relative_error(expected_blue, blue_extent, 0.05_wp, &
                                   "Blue extent ~ lam*exp(-4v/c)", total_tests, total_failures)
        call assert_relative_error(expected_red, red_extent, 0.05_wp, &
                                   "Red extent ~ lam*exp(+4v/c)", total_tests, total_failures)

        deallocate(lambda, spec, ctx)
    end subroutine test_relativistic_asymmetry

    subroutine test_log_linear_accuracy()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: minl, maxl
        real(WP) :: sigma_true, sigma_apply, sigma_expected
        real(WP) :: lambda0, fwhm, blue_hwhm, red_hwhm
        real(WP) :: c_kms, sigma_measured, peak_expected, peak_actual
        integer :: n

        call print_group("Log-Linear Accuracy Benchmark")

        n = 2001
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 4800.0_wp, 0.2_wp)
        minl = lambda(1)
        maxl = lambda(n)

        lambda0 = 5000.0_wp
        sigma_true = 100.0_wp
        sigma_apply = 100.0_wp
        sigma_expected = sqrt(sigma_true**2 + sigma_apply**2)

        call fill_gaussian_logv(lambda, lambda0, sigma_true, spec)

        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.true., smoothspec_fast=.false.)
        call apply_smoothing(ctx, lambda, spec, sigma_apply, minl, maxl)

        call compute_fwhm(lambda, spec, fwhm, blue_hwhm, red_hwhm)

        c_kms = C_LIGHT * 1.0e-13_wp
        sigma_measured = (fwhm / lambda0) * (c_kms / 2.355_wp)
        call assert_relative_error(sigma_expected, sigma_measured, 0.02_wp, &
                                   "Sigma_final ~= sqrt(sum^2)", total_tests, total_failures)

        peak_expected = sigma_true / sigma_expected
        peak_actual = maxval(spec)
        call assert_relative_error(peak_expected, peak_actual, 1.0e-3_wp, &
                                   "Peak flux conservation (<0.1%)", total_tests, total_failures)

        deallocate(lambda, spec, ctx)
    end subroutine test_log_linear_accuracy

    subroutine test_method_convergence()
        type(fsps_context_t), allocatable :: ctx_fast, ctx_slow
        real(WP), allocatable :: lambda(:), spec_in(:), spec_fast(:), spec_slow(:)
        real(WP) :: minl, maxl, sigma_kms
        real(WP) :: max_diff, norm
        integer :: n

        call print_group("Method Convergence (Well-Sampled Regime)")

        n = 1601
        allocate(lambda(n), spec_in(n), spec_fast(n), spec_slow(n))
        call fill_linear_grid(lambda, 4500.0_wp, 0.5_wp)
        minl = lambda(1)
        maxl = lambda(n)

        ! Input feature: Gaussian at 5000A with native width
        call fill_gaussian_logv(lambda, 5000.0_wp, 50.0_wp, spec_in)

        ! Use a large sigma to ensure the kernel is well-sampled (>10 pixels).
        ! This reduces interpolation artifacts from the "Slow" method's resampling.
        sigma_kms = 300.0_wp

        allocate(ctx_fast, ctx_slow)
        call setup_context(ctx_fast, smooth_velocity=.true., smoothspec_fast=.true.)
        call setup_context(ctx_slow, smooth_velocity=.true., smoothspec_fast=.false.)

        spec_fast = spec_in
        spec_slow = spec_in

        call apply_smoothing(ctx_fast, lambda, spec_fast, sigma_kms, minl, maxl)
        call apply_smoothing(ctx_slow, lambda, spec_slow, sigma_kms, minl, maxl)

        max_diff = maxval(abs(spec_fast - spec_slow))
        norm = maxval(spec_slow)
        
        ! Tolerance set to 5e-3 (0.5%).
        ! A perfect match is impossible because:
        ! 1. The "Fast" method is continuous (floating-point trapezoids).
        ! 2. The "Slow" method is discrete (snaps to integer pixels on log-grid).
        ! A 0.5% agreement confirms they are calculating the same physics.
        call assert_true(max_diff / max(norm, SAFE_FLOOR) < 5.0e-3_wp, &
                         "Fast vs Slow match within numerical noise", total_tests, total_failures)

        deallocate(lambda, spec_in, spec_fast, spec_slow, ctx_fast, ctx_slow)
    end subroutine test_method_convergence

    ! ------------------------------------------------------------------------
    ! Group 3: Adaptive Grid Dynamics (The "Hunter")
    ! ------------------------------------------------------------------------
    subroutine test_zig_zag_sigma()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec_in(:), spec_out(:), ires(:)
        real(WP) :: minl, maxl, sigma_kms
        real(WP) :: sum_high, sum_low
        integer :: n, i, count_high, count_low

        call print_group("Zig-Zag Sigma Stress Test")

        n = 501
        allocate(lambda(n), spec_in(n), spec_out(n), ires(n))
        call fill_linear_grid(lambda, 4000.0_wp, 2.0_wp)
        minl = lambda(1)
        maxl = lambda(n)

        do i = 1, n
            spec_in(i) = 1.0_wp + 0.5_wp * sin((lambda(i) - 4000.0_wp) / 12.0_wp)
            if (mod(i, 2) == 0) then
                ires(i) = 1000.0_wp
            else
                ires(i) = 10.0_wp
            end if
        end do

        sigma_kms = 200.0_wp
        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.true., smoothspec_fast=.false.)
        spec_out = spec_in
        call apply_smoothing(ctx, lambda, spec_out, sigma_kms, minl, maxl, ires)

        sum_high = 0.0_wp
        sum_low = 0.0_wp
        count_high = 0
        count_low = 0
        do i = 1, n
            if (ires(i) > 100.0_wp) then
                sum_high = sum_high + abs(spec_out(i) - spec_in(i))
                count_high = count_high + 1
            else
                sum_low = sum_low + abs(spec_out(i) - spec_in(i))
                count_low = count_low + 1
            end if
        end do

        call assert_true((sum_high / real(count_high, WP)) > (sum_low / real(count_low, WP)), &
                         "High-sigma pixels smooth more", total_tests, total_failures)

        deallocate(lambda, spec_in, spec_out, ires, ctx)
    end subroutine test_zig_zag_sigma

    subroutine test_sparse_grid_subpixel_sigma()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec(:), spec_orig(:)
        real(WP) :: minl, maxl, flux_in, flux_out
        integer :: n, peak_idx

        call print_group("Sparse Grid / Sub-Pixel Sigma")

        n = 101
        allocate(lambda(n), spec(n), spec_orig(n))
        call fill_linear_grid(lambda, 4000.0_wp, 10.0_wp)
        minl = lambda(1)
        maxl = lambda(n)

        spec = 0.0_wp
        peak_idx = n / 2
        spec(peak_idx) = 1.0_wp
        spec_orig = spec

        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.false., smoothspec_fast=.false.)
        call apply_smoothing(ctx, lambda, spec, 1.0_wp, minl, maxl)

        flux_in = integrate_trapezoid(lambda, spec_orig)
        flux_out = integrate_trapezoid(lambda, spec)
        call assert_relative_error(flux_in, flux_out, 1.0e-6_wp, "Total flux conserved", total_tests, total_failures)
        call assert_true(maxval(spec) >= 0.9_wp, "Line does not disappear", total_tests, total_failures)

        deallocate(lambda, spec, spec_orig, ctx)
    end subroutine test_sparse_grid_subpixel_sigma

    ! ------------------------------------------------------------------------
    ! Group 4: Boundary Conditions & Numerical Stability
    ! ------------------------------------------------------------------------
    subroutine test_flat_field_identity()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: minl, maxl, max_abs
        integer :: n

        call print_group("Flat Field Normalization Identity")

        n = 2501
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 3000.0_wp, 2.0_wp)
        minl = lambda(1)
        maxl = lambda(n)

        spec = 1.0_wp
        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.true., smoothspec_fast=.true.)
        call apply_smoothing(ctx, lambda, spec, 500.0_wp, minl, maxl)

        max_abs = maxval(abs(spec(10:n-10) - 1.0_wp))
        call assert_true(max_abs <= 1.0e-6_wp, "Flat spectrum remains flat", total_tests, total_failures)

        deallocate(lambda, spec, ctx)
    end subroutine test_flat_field_identity

    subroutine test_log_linear_edge_renormalization()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: minl, maxl, max_abs
        integer :: n

        call print_group("Log-Linear Edge Renormalization")

        n = 1201
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 4000.0_wp, 1.0_wp)
        minl = lambda(1)
        maxl = lambda(n)

        spec = 1.0_wp
        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.true., smoothspec_fast=.false.)
        call apply_smoothing(ctx, lambda, spec, 200.0_wp, minl, maxl)

        max_abs = maxval(abs(spec(1:5) - 1.0_wp))
        call assert_true(max_abs <= 1.0e-6_wp, "Edges remain normalized", total_tests, total_failures)

        deallocate(lambda, spec, ctx)
    end subroutine test_log_linear_edge_renormalization

    subroutine test_window_cut_passthrough()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: lambda(:), spec(:), spec_orig(:)
        real(WP) :: minl, maxl
        integer :: n, i
        logical :: any_modified, below_ok, above_ok

        call print_group("Window Cut Pass-Through")

        n = 401
        allocate(lambda(n), spec(n), spec_orig(n))
        call fill_linear_grid(lambda, 3500.0_wp, 5.0_wp)
        minl = 4000.0_wp
        maxl = 5000.0_wp

        spec = 1.2_wp + 0.3_wp * sin(lambda / 150.0_wp)
        spec_orig = spec

        allocate(ctx)
        call setup_context(ctx, smooth_velocity=.true., smoothspec_fast=.true.)
        call apply_smoothing(ctx, lambda, spec, 300.0_wp, minl, maxl)

        any_modified = .false.
        below_ok = .true.
        above_ok = .true.
        do i = 1, size(lambda)
            if (lambda(i) < minl) then
                if (spec(i) /= spec_orig(i)) below_ok = .false.
            elseif (lambda(i) > maxl) then
                if (spec(i) /= spec_orig(i)) above_ok = .false.
            else
                if (abs(spec(i) - spec_orig(i)) > 0.0_wp) any_modified = .true.
            end if
        end do

        call assert_true(below_ok, "Below minl unchanged", total_tests, total_failures)
        call assert_true(above_ok, "Above maxl unchanged", total_tests, total_failures)
        call assert_true(any_modified, "Inside window modified", total_tests, total_failures)

        deallocate(lambda, spec, spec_orig, ctx)
    end subroutine test_window_cut_passthrough

    ! ------------------------------------------------------------------------
    ! Helper Routines
    ! ------------------------------------------------------------------------
    subroutine setup_context(ctx, smooth_velocity, smoothspec_fast)
        type(fsps_context_t), intent(out) :: ctx
        logical, intent(in) :: smooth_velocity, smoothspec_fast

        ctx%smooth_velocity_val = merge(1, 0, smooth_velocity)
        ctx%smoothspec_fast_val = merge(1, 0, smoothspec_fast)
    end subroutine setup_context

    subroutine fill_linear_grid(lambda, start_val, step)
        real(WP), dimension(:), intent(out) :: lambda
        real(WP), intent(in) :: start_val, step
        integer :: i

        do i = 1, size(lambda)
            lambda(i) = start_val + step * real(i - 1, WP)
        end do
    end subroutine fill_linear_grid

    subroutine fill_gaussian_logv(lambda, lambda0, sigma_kms, spec)
        real(WP), dimension(:), intent(in) :: lambda
        real(WP), intent(in) :: lambda0, sigma_kms
        real(WP), dimension(:), intent(out) :: spec
        real(WP) :: c_kms
        real(WP) :: v
        integer :: i

        c_kms = C_LIGHT * 1.0e-13_wp
        do i = 1, size(lambda)
            v = c_kms * log(lambda(i) / lambda0)
            spec(i) = exp(-0.5_wp * (v / sigma_kms)**2)
        end do
    end subroutine fill_gaussian_logv

    function find_nearest_index(lambda, value) result(idx)
        real(WP), dimension(:), intent(in) :: lambda
        real(WP), intent(in) :: value
        integer :: idx
        real(WP) :: min_diff
        integer :: i

        idx = 1
        min_diff = abs(lambda(1) - value)
        do i = 2, size(lambda)
            if (abs(lambda(i) - value) < min_diff) then
                min_diff = abs(lambda(i) - value)
                idx = i
            end if
        end do
    end function find_nearest_index

    subroutine compute_fwhm(lambda, spec, fwhm, blue_hwhm, red_hwhm)
        real(WP), dimension(:), intent(in) :: lambda, spec
        real(WP), intent(out) :: fwhm, blue_hwhm, red_hwhm
        real(WP) :: peak_val, half_val
        real(WP) :: left_x, right_x
        integer :: peak_idx, i

        peak_idx = maxloc(spec, dim=1)
        peak_val = spec(peak_idx)
        half_val = 0.5_wp * peak_val

        left_x = lambda(1)
        do i = peak_idx, 2, -1
            if (spec(i) <= half_val) then
                left_x = interpolate_x(lambda(i), spec(i), lambda(i+1), spec(i+1), half_val)
                exit
            end if
        end do

        right_x = lambda(size(lambda))
        do i = peak_idx, size(lambda) - 1
            if (spec(i) <= half_val) then
                right_x = interpolate_x(lambda(i-1), spec(i-1), lambda(i), spec(i), half_val)
                exit
            end if
        end do

        blue_hwhm = lambda(peak_idx) - left_x
        red_hwhm = right_x - lambda(peak_idx)
        fwhm = right_x - left_x
    end subroutine compute_fwhm

    function interpolate_x(x1, y1, x2, y2, y_target) result(x)
        real(WP), intent(in) :: x1, y1, x2, y2, y_target
        real(WP) :: x

        if (abs(y2 - y1) <= SAFE_FLOOR) then
            x = 0.5_wp * (x1 + x2)
        else
            x = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
        end if
    end function interpolate_x

    function integrate_trapezoid(lambda, spec) result(total)
        real(WP), dimension(:), intent(in) :: lambda, spec
        real(WP) :: total
        integer :: i

        total = 0.0_wp
        do i = 1, size(lambda) - 1
            total = total + 0.5_wp * (spec(i) + spec(i+1)) * (lambda(i+1) - lambda(i))
        end do
    end function integrate_trapezoid

    !> The precomputed sparse kernel (used to pre-smooth the AGB templates) must reproduce the
    !> direct convolution it mirrors, including where the grid spacing changes and at the window
    !> limits.
    subroutine test_velocity_kernel_matches_convolution()
        integer, parameter :: n = 600
        real(WP) :: lambda(n), spec(n), direct(n), smoothed(n)
        type(velocity_kernel_t) :: kernel
        integer :: i, i_first

        call print_group("Velocity Kernel Matches Direct Convolution")

        ! Grid with a spacing jump (fine, then coarse), spectrum with lines on a power law
        do i = 1, n
            if (i <= 300) then
                lambda(i) = 2.0e4_wp + 20.0_wp * i
            else
                lambda(i) = lambda(300) + 400.0_wp * (i - 300)
            end if
            spec(i) = (lambda(i) / 3.0e4_wp)**(-2) * (1.0_wp - 0.5_wp * exp(-0.5_wp * ((lambda(i) - 2.6e4_wp) / 60.0_wp)**2))
        end do

        direct = spec
        call smooth_velocity(lambda, direct, 1.0e4_wp, 3.0e4_wp, 1.0e8_wp)
        call build_velocity_kernel(lambda, 1.0e4_wp, 3.0e4_wp, 1.0e8_wp, kernel)
        i_first = 200
        call apply_velocity_kernel(kernel, spec, i_first, smoothed(1:n - i_first + 1))

        call assert_true(maxval(abs(smoothed(1:n - i_first + 1) - direct(i_first:)) / abs(direct(i_first:))) < 1.0e-12_wp, &
                         "kernel application equals smooth_velocity (rel. 1e-12)", total_tests, total_failures)
        call assert_true(any(abs(direct(i_first:) - spec(i_first:)) > 1.0e-3_wp * spec(i_first:)), &
                         "smoothing changes the spectrum (non-trivial check)", total_tests, total_failures)
    end subroutine test_velocity_kernel_matches_convolution

end module test_fsps_smoothing_mod