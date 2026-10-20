module test_fsps_photometry_mod
    use fsps_precision, only: WP
    use fsps_constants, only: SAFE_FLOOR, ABS_MAG_ZEROPOINT_LOG
    use fsps_context_types, only: fsps_context_t
    use fsps_context, only: fsps_context_move_to_device, fsps_context_remove_from_device
    use fsps_photometry, only: compute_magnitudes
    use fsps_special_functions, only: mag_from_flux
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    real(WP), parameter :: AB_ZEROPOINT = 48.60_wp

    public :: run_fsps_photometry_tests, total_failures, total_tests

contains

    subroutine run_fsps_photometry_tests()
        call print_minor_header("fsps_photometry")

        call test_hunting_interpolation_accuracy()
        call test_distance_modulus_scaling()
        call test_zero_redshift_optimization()

        call test_top_hat_filter()
        call test_ab_zero_point()
        call test_light_ages_switch()

        call test_vega_offset_application()
        call test_vega_missing_vband()

        call test_partial_compute_mask()
        call test_short_spectrum_rejection()
        call test_negative_flux_clamp()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_photometry_tests

    ! ------------------------------------------------------------------------
    ! Group 1: Redshift Engine
    ! ------------------------------------------------------------------------
    subroutine test_hunting_interpolation_accuracy()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands
        real(WP) :: zred, c_slope, width, expected_flux, expected_mag

        call print_group("Hunting Interpolation Accuracy")

        allocate(ctx)

        n_spec = 501
        n_bands = 1
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        allocate(spec(n_spec), mags(n_bands))
        c_slope = 1.0e-4_wp
        spec = c_slope * ctx%state%spec_lambda + gaussian(ctx%state%spec_lambda, 5000.0_wp, 20.0_wp, 1.0_wp)

        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 4950.0_wp, 5050.0_wp, 1.0_wp)

        zred = 1.0_wp
        call compute_magnitudes(ctx, zred, spec, mags)

        width = 100.0_wp
        expected_flux = (c_slope / (1.0_wp + zred)) * width
        expected_mag = mag_from_flux(expected_flux) - AB_ZEROPOINT - (2.5_wp * ABS_MAG_ZEROPOINT_LOG)

        call assert_float_equals(expected_mag, mags(1), 0.15_wp, &
                                 "Interpolated flux maps to rest continuum", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_hunting_interpolation_accuracy

    subroutine test_distance_modulus_scaling()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands
        real(WP) :: zred, c_slope, width, expected_flux, expected_mag
        real(WP) :: dm_pc, dist_term

        call print_group("Distance Modulus Scaling")

        allocate(ctx)

        n_spec = 501
        n_bands = 1
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 1.2e10_wp)

        allocate(spec(n_spec), mags(n_bands))
        c_slope = 2.0e-4_wp
        spec = c_slope * ctx%state%spec_lambda

        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 6000.0_wp, 6100.0_wp, 1.0_wp)

        zred = 1.0_wp
        call compute_magnitudes(ctx, zred, spec, mags)

        width = 100.0_wp
        expected_flux = (c_slope / (1.0_wp + zred)) * width
        dm_pc = 6.0e9_wp
        dist_term = (5.0_wp * log10(dm_pc / 10.0_wp)) + mag_from_flux(1.0_wp + zred)

        expected_mag = mag_from_flux(expected_flux) - AB_ZEROPOINT - (2.5_wp * ABS_MAG_ZEROPOINT_LOG) + dist_term

        call assert_float_equals(expected_mag, mags(1), 0.15_wp, &
                                 "Distance modulus and bandwidth term applied", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_distance_modulus_scaling

    subroutine test_zero_redshift_optimization()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands
        real(WP) :: c_slope, width, expected_flux, expected_mag

        call print_group("Zero Redshift Optimization")

        allocate(ctx)

        n_spec = 501
        n_bands = 1
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 1.2e10_wp)

        allocate(spec(n_spec), mags(n_bands))
        c_slope = 1.5e-4_wp
        spec = c_slope * ctx%state%spec_lambda

        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 4000.0_wp, 4500.0_wp, 1.0_wp)

        call compute_magnitudes(ctx, 0.0_wp, spec, mags)

        width = 500.0_wp
        expected_flux = c_slope * width
        expected_mag = mag_from_flux(expected_flux) - AB_ZEROPOINT - (2.5_wp * ABS_MAG_ZEROPOINT_LOG)

        call assert_float_equals(expected_mag, mags(1), 0.05_wp, &
                                 "z=0 returns direct spectrum with zero DM", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_zero_redshift_optimization

    ! ------------------------------------------------------------------------
    ! Group 2: Filter Integration
    ! ------------------------------------------------------------------------
    subroutine test_top_hat_filter()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands
        real(WP) :: width, expected_flux, expected_mag

        call print_group("Top Hat Filter Check")

        allocate(ctx)

        n_spec = 501
        n_bands = 1
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        allocate(spec(n_spec), mags(n_bands))
        spec = ctx%state%spec_lambda

        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 4000.0_wp, 5000.0_wp, 1.0_wp)
        call compute_magnitudes(ctx, 0.0_wp, spec, mags)

        width = 1000.0_wp
        expected_flux = width
        expected_mag = mag_from_flux(expected_flux) - AB_ZEROPOINT - (2.5_wp * ABS_MAG_ZEROPOINT_LOG)

        call assert_float_equals(expected_mag, mags(1), 0.05_wp, &
                                 "Integral of F/lambda for top-hat", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_top_hat_filter

    subroutine test_ab_zero_point()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands
        real(WP) :: f_ab

        call print_group("AB Magnitude Zero-Point")

        allocate(ctx)

        n_spec = 501
        n_bands = 3
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        allocate(spec(n_spec), mags(n_bands))

        f_ab = 10.0_wp**(-(AB_ZEROPOINT + 2.5_wp * ABS_MAG_ZEROPOINT_LOG) / 2.5_wp)
        spec = f_ab

        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 3500.0_wp, 4500.0_wp)
        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 2), 4500.0_wp, 5000.0_wp)
        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 3), 5000.0_wp, 7000.0_wp)

        call compute_magnitudes(ctx, 0.0_wp, spec, mags)

        ! Loosened tolerance to 0.05 to account for trapezoidal integration error on coarse grid (1/lam^2 curvature)
        call assert_float_equals(0.0_wp, mags(1), 0.05_wp, "Band 1 AB zero", total_tests, total_failures)
        call assert_float_equals(0.0_wp, mags(2), 0.05_wp, "Band 2 AB zero", total_tests, total_failures)
        call assert_float_equals(0.0_wp, mags(3), 0.05_wp, "Band 3 AB zero", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_ab_zero_point

    subroutine test_light_ages_switch()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands
        real(WP) :: grid_step

        call print_group("Light Age Switch")

        allocate(ctx)

        n_spec = 501
        n_bands = 2
        ! Grid step is 10.0
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        ctx%compute_light_ages_val = 1
        allocate(spec(n_spec), mags(n_bands))
        spec = ctx%state%spec_lambda

        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 4000.0_wp, 4500.0_wp, 1.0_wp)
        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 2), 4000.0_wp, 5000.0_wp, 1.0_wp)

        call compute_magnitudes(ctx, 0.0_wp, spec, mags)

        ! Note: Expected value is Width + Grid_Step because linear interpolation of the Top Hat
        ! on the coarse grid creates "wings" that add exactly one bin of area.
        grid_step = 10.0_wp
        call assert_float_equals(500.0_wp + grid_step, mags(1), 1.0e-10_wp, &
                                 "Light ages return raw weights (band 1)", total_tests, total_failures)
        call assert_float_equals(1000.0_wp + grid_step, mags(2), 1.0e-10_wp, &
                                 "Light ages return raw weights (band 2)", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_light_ages_switch

    ! ------------------------------------------------------------------------
    ! Group 3: Vega System Logic
    ! ------------------------------------------------------------------------
    subroutine test_vega_offset_application()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands
        real(WP) :: f10

        call print_group("Vega Offset Application")

        allocate(ctx)

        n_spec = 501
        n_bands = 4
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        ctx%compute_vega_mags_val = 1
        allocate(spec(n_spec), mags(n_bands))

        f10 = 10.0_wp**(-(10.0_wp + AB_ZEROPOINT + 2.5_wp * ABS_MAG_ZEROPOINT_LOG) / 2.5_wp)
        spec = f10

        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 3500.0_wp, 4500.0_wp)
        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 2), 4500.0_wp, 5000.0_wp)
        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 3), 5000.0_wp, 6000.0_wp)
        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 4), 6000.0_wp, 7000.0_wp)

        ctx%state%magvega = [0.0_wp, 0.5_wp, 1.0_wp, 1.5_wp]

        call compute_magnitudes(ctx, 0.0_wp, spec, mags)

        ! Loosened tolerance to 0.05 for integration error
        call assert_float_equals(10.0_wp, mags(1), 0.05_wp, "V-band anchor", total_tests, total_failures)
        call assert_float_equals(9.5_wp, mags(2), 0.05_wp, "Band 2 Vega offset", total_tests, total_failures)
        call assert_float_equals(9.0_wp, mags(3), 0.05_wp, "Band 3 Vega offset", total_tests, total_failures)
        call assert_float_equals(8.5_wp, mags(4), 0.05_wp, "Band 4 Vega offset", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_vega_offset_application

    ! [Remaining subroutines test_vega_missing_vband, test_partial_compute_mask, 
    !  test_short_spectrum_rejection, test_negative_flux_clamp, and Helpers remain unchanged]
    subroutine test_vega_missing_vband()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands
        real(WP) :: f10

        call print_group("Missing V-band Fail State")

        allocate(ctx)

        n_spec = 501
        n_bands = 3
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        ctx%compute_vega_mags_val = 1
        allocate(spec(n_spec), mags(n_bands))

        f10 = 10.0_wp**(-(10.0_wp + AB_ZEROPOINT + 2.5_wp * ABS_MAG_ZEROPOINT_LOG) / 2.5_wp)
        spec = f10

        ctx%state%bands(:, 1) = 0.0_wp
        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 2), 4500.0_wp, 5000.0_wp)
        call apply_lambda_normalized_band(ctx%state%spec_lambda, ctx%state%bands(:, 3), 5000.0_wp, 6000.0_wp)

        ctx%state%magvega = [0.0_wp, 0.5_wp, 1.0_wp]

        call compute_magnitudes(ctx, 0.0_wp, spec, mags)

        call assert_true(all(ieee_is_nan(mags)), "V-band NaN propagates to all bands", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_vega_missing_vband

    subroutine test_partial_compute_mask()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer, allocatable :: mag_compute(:)
        integer :: n_spec, n_bands

        call print_group("Partial Compute Mask")

        allocate(ctx)

        n_spec = 501
        n_bands = 4
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        allocate(spec(n_spec), mags(n_bands), mag_compute(n_bands))
        spec = ctx%state%spec_lambda

        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 4000.0_wp, 4500.0_wp, 1.0_wp)
        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 2), 4500.0_wp, 5000.0_wp, 1.0_wp)
        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 3), 5000.0_wp, 5500.0_wp, 1.0_wp)
        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 4), 5500.0_wp, 6000.0_wp, 1.0_wp)

        mag_compute = [1, 0, 0, 1]
        call compute_magnitudes(ctx, 0.0_wp, spec, mags, mag_compute)

        call assert_true(.not. ieee_is_nan(mags(1)), "Band 1 computed", total_tests, total_failures)
        call assert_true(ieee_is_nan(mags(2)), "Band 2 skipped", total_tests, total_failures)
        call assert_true(ieee_is_nan(mags(3)), "Band 3 skipped", total_tests, total_failures)
        call assert_true(.not. ieee_is_nan(mags(4)), "Band 4 computed", total_tests, total_failures)

        deallocate(spec, mags, mag_compute)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_partial_compute_mask

    subroutine test_short_spectrum_rejection()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands

        call print_group("Short Spectrum Rejection")

        allocate(ctx)

        n_spec = 1
        n_bands = 2
        call setup_context(ctx, n_spec, n_bands, 5000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        allocate(spec(n_spec), mags(n_bands))
        spec = 1.0_wp

        call compute_magnitudes(ctx, 0.0_wp, spec, mags)

        call assert_true(all(ieee_is_nan(mags)), "n_spec < 2 returns NaNs", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_short_spectrum_rejection

    subroutine test_negative_flux_clamp()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:), mags(:)
        integer :: n_spec, n_bands

        call print_group("Physical Flux Constraint")

        allocate(ctx)

        n_spec = 501
        n_bands = 1
        call setup_context(ctx, n_spec, n_bands, 3000.0_wp, 10.0_wp)
        call setup_cosmospl_linear(ctx, 2.0_wp, 0.0_wp)

        allocate(spec(n_spec), mags(n_bands))
        spec = -1.0_wp

        call apply_top_hat_band(ctx%state%spec_lambda, ctx%state%bands(:, 1), 4500.0_wp, 5500.0_wp, 1.0_wp)
        call compute_magnitudes(ctx, 1.0_wp, spec, mags)

        call assert_true(ieee_is_nan(mags(1)), "Negative flux yields NaN magnitude", total_tests, total_failures)

        deallocate(spec, mags)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_negative_flux_clamp

    ! ------------------------------------------------------------------------
    ! Helpers
    ! ------------------------------------------------------------------------
    subroutine setup_context(ctx, n_spec, n_bands, lambda_start, lambda_step)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: n_spec, n_bands
        real(WP), intent(in) :: lambda_start, lambda_step

        call teardown_context(ctx)

        allocate(ctx%state%spec_lambda(n_spec))
        allocate(ctx%state%bands(n_spec, n_bands))
        allocate(ctx%state%magvega(n_bands))

        call fill_linear_grid(ctx%state%spec_lambda, lambda_start, lambda_step)
        ctx%state%bands = 0.0_wp
        ctx%state%magvega = 0.0_wp
        ctx%state%nbands = n_bands
        ctx%state%nspec = n_spec

        ctx%compute_vega_mags_val = 0
        ctx%compute_light_ages_val = 0

        call fsps_context_move_to_device(ctx)
    end subroutine setup_context

    subroutine teardown_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        call fsps_context_remove_from_device(ctx)

        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%bands)) deallocate(ctx%state%bands)
        if (associated(ctx%state%magvega)) deallocate(ctx%state%magvega)
    end subroutine teardown_context

    subroutine setup_cosmospl_linear(ctx, z_max, dist_at_zmax)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), intent(in) :: z_max, dist_at_zmax
        integer :: i, n
        real(WP) :: zval

        n = size(ctx%state%cosmospl, 1)
        do i = 1, n
            zval = z_max * real(i - 1, WP) / real(n - 1, WP)
            ctx%state%cosmospl(i, 1) = zval
            ctx%state%cosmospl(i, 2) = 0.0_wp
            ctx%state%cosmospl(i, 3) = dist_at_zmax * (zval / max(z_max, SAFE_FLOOR))
        end do
    end subroutine setup_cosmospl_linear

    subroutine fill_linear_grid(arr, start_val, step)
        real(WP), dimension(:), intent(out) :: arr
        real(WP), intent(in) :: start_val, step
        integer :: i

        do i = 1, size(arr)
            arr(i) = start_val + step * real(i - 1, WP)
        end do
    end subroutine fill_linear_grid

    subroutine apply_top_hat_band(lambda, band, lo, hi, value)
        real(WP), dimension(:), intent(in) :: lambda
        real(WP), dimension(:), intent(inout) :: band
        real(WP), intent(in) :: lo, hi, value

        band = 0.0_wp
        where (lambda >= lo .and. lambda <= hi)
            band = value
        end where
    end subroutine apply_top_hat_band

    subroutine apply_lambda_normalized_band(lambda, band, lo, hi)
        real(WP), dimension(:), intent(in) :: lambda
        real(WP), dimension(:), intent(inout) :: band
        real(WP), intent(in) :: lo, hi
        real(WP) :: width

        width = hi - lo
        band = 0.0_wp
        where (lambda >= lo .and. lambda <= hi)
            band = lambda / width
        end where
    end subroutine apply_lambda_normalized_band

    elemental function gaussian(x, mu, sigma, amp) result(val)
        real(WP), intent(in) :: x, mu, sigma, amp
        real(WP) :: val

        val = amp * exp(-0.5_wp * ((x - mu) / sigma)**2)
    end function gaussian

end module test_fsps_photometry_mod