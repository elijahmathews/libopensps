module test_fsps_interpolation_mod
    use fsps_precision, only: WP
    use fsps_interpolation
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_int_equals, assert_float_equals, assert_is_nan
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_interpolation_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for fsps_interpolation module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_fsps_interpolation_tests()

        call print_minor_header("fsps_interpolation")

        call test_find_interval()
        call test_linear_scalar()
        call test_linear_array()
        call test_robustness()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    
    end subroutine run_fsps_interpolation_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: FIND_INTERVAL
    ! ------------------------------------------------------------------------
    subroutine test_find_interval()
        real(WP), dimension(4) :: x_asc = [0.0_wp, 10.0_wp, 20.0_wp, 30.0_wp]
        real(WP), dimension(4) :: x_desc = [30.0_wp, 20.0_wp, 10.0_wp, 0.0_wp]
        integer :: idx

        call print_group("find_interval")

        ! Case 1.1: Standard Ascending Search (Value 15 is in interval 2: 10..20)
        idx = find_interval(x_asc, 15.0_wp)
        call assert_int_equals(2, idx, "Ascending Search (15.0 in 0..30)", total_tests, total_failures)

        ! Case 1.2: Standard Descending Search (Value 15 is in interval 2: 20..10)
        ! Note: fsps_interpolation handles monotonic decreasing automatically
        idx = find_interval(x_desc, 15.0_wp)
        call assert_int_equals(2, idx, "Descending Search (15.0 in 30..0)", total_tests, total_failures)

        ! Case 1.3: Exact Match (Lower Bound)
        ! Value 10.0 matches index 2. Should return 2.
        idx = find_interval(x_asc, 10.0_wp)
        call assert_int_equals(2, idx, "Exact Match Lower Bound (10.0)", total_tests, total_failures)

        ! Case 1.4: Low Out-of-Bounds (Ascending)
        ! Value -5.0 < 0.0. Should return 0.
        idx = find_interval(x_asc, -5.0_wp)
        call assert_int_equals(0, idx, "Low OOB Ascending (-5.0)", total_tests, total_failures)

        ! Case 1.5: High Out-of-Bounds (Ascending)
        ! Value 35.0 > 30.0. Should return N = 4.
        idx = find_interval(x_asc, 35.0_wp)
        call assert_int_equals(4, idx, "High OOB Ascending (35.0)", total_tests, total_failures)

    end subroutine test_find_interval

    ! ------------------------------------------------------------------------
    ! TEST SUITE: LINEAR INTERPOLATION (SCALAR)
    ! ------------------------------------------------------------------------
    subroutine test_linear_scalar()
        real(WP), dimension(3) :: x = [1.0_wp, 2.0_wp, 3.0_wp]
        real(WP), dimension(3) :: y = [1.0_wp, 2.0_wp, 3.0_wp]
        real(WP), dimension(2) :: x_short = [0.0_wp, 1.0_wp]
        real(WP), dimension(2) :: y_short = [0.0_wp, 2.0_wp] ! Slope = 2
        real(WP), dimension(3) :: y_flat = [5.0_wp, 5.0_wp, 5.0_wp]
        real(WP) :: res

        call print_group("interpolate_linear (Scalar)")

        ! Case 2.1: Identity Function
        res = interpolate_linear(x, y, 1.5_wp)
        call assert_float_equals(1.5_wp, res, 1.0e-5_wp, "Identity Interpolation", total_tests, total_failures)

        ! Case 2.2: Extrapolation
        ! Using slope=2 line defined on [0,1]. Value at 2.0 should be 4.0.
        res = interpolate_linear(x_short, y_short, 2.0_wp)
        call assert_float_equals(4.0_wp, res, 1.0e-5_wp, "Linear Extrapolation", total_tests, total_failures)

        ! Case 2.4: Constant Function
        res = interpolate_linear(x, y_flat, 1.5_wp)
        call assert_float_equals(5.0_wp, res, 1.0e-5_wp, "Constant Function", total_tests, total_failures)

    end subroutine test_linear_scalar

    ! ------------------------------------------------------------------------
    ! TEST SUITE: LINEAR INTERPOLATION (ARRAY)
    ! ------------------------------------------------------------------------
    subroutine test_linear_array()
        real(WP), dimension(3) :: x = [1.0_wp, 2.0_wp, 3.0_wp]
        real(WP), dimension(3) :: y = [2.0_wp, 4.0_wp, 6.0_wp] ! y = 2x
        real(WP), dimension(2) :: query = [1.5_wp, 2.5_wp]
        real(WP), dimension(2) :: expected = [3.0_wp, 5.0_wp]
        real(WP), dimension(2) :: res

        call print_group("interpolate_linear (Array)")

        ! Case 2.3: Array Interface
        res = interpolate_linear(x, y, query)
        
        call assert_float_equals(expected(1), res(1), 1.0e-5_wp, "Array Element 1", total_tests, total_failures)
        call assert_float_equals(expected(2), res(2), 1.0e-5_wp, "Array Element 2", total_tests, total_failures)

    end subroutine test_linear_array

    ! ------------------------------------------------------------------------
    ! TEST SUITE: ROBUSTNESS / ERRORS
    ! ------------------------------------------------------------------------
    subroutine test_robustness()
        real(WP), dimension(1) :: x_bad = [1.0_wp]
        real(WP), dimension(1) :: y_bad = [1.0_wp]
        real(WP), dimension(2) :: x_ok = [1.0_wp, 2.0_wp]
        real(WP), dimension(3) :: y_mismatch = [1.0_wp, 2.0_wp, 3.0_wp]
        real(WP) :: res_scalar
        real(WP), dimension(1) :: res_array
        
        call print_group("Robustness")

        ! Case 3.1: Insufficient Data (Size < 2)
        res_scalar = interpolate_linear(x_bad, y_bad, 1.5_wp)
        call assert_is_nan(res_scalar, "Size < 2 returns NaN (Scalar)", total_tests, total_failures)

        res_array = interpolate_linear(x_bad, y_bad, [1.5_wp])
        call assert_is_nan(res_array(1), "Size < 2 returns NaN (Array)", total_tests, total_failures)

        ! Case 3.2: Mismatched Dimensions
        res_scalar = interpolate_linear(x_ok, y_mismatch, 1.5_wp)
        call assert_is_nan(res_scalar, "Dimension Mismatch returns NaN", total_tests, total_failures)

    end subroutine test_robustness

end module test_fsps_interpolation_mod