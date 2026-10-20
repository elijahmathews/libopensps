! ------------------------------------------------------------------------
! AUXILIARY MODULE: TEST FUNCTIONS
! We define these in a module to avoid "Executable Stack" warnings.
! Passing internal procedures (from 'program contains') as arguments
! forces GFortran to create stack trampolines. Module procs do not.
! ------------------------------------------------------------------------
module integration_test_funcs
    use fsps_precision, only: WP
    use fsps_context_types, only: fsps_context_t
    implicit none

contains

    ! f(x) = x^4
    pure function func_poly4(x) result(res)
        real(WP), intent(in) :: x
        real(WP) :: res
        res = x**4
    end function func_poly4

    ! f(x) = exp(x)
    pure function func_exp(x) result(res)
        real(WP), intent(in) :: x
        real(WP) :: res
        res = exp(x)
    end function func_exp

    ! f(x) = 1.0
    pure function func_const(x) result(res)
        real(WP), intent(in) :: x
        real(WP) :: res
        res = 1.0_wp
    end function func_const

    ! f(ctx, x) = x^2 (ignores ctx)
    pure function func_ctx_poly2(ctx, x) result(res)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: x
        real(WP) :: res
        
        ! Explicitly ignore ctx to silence unused-dummy-argument warning
        associate (ignore => ctx)
        end associate
        
        res = x**2
    end function func_ctx_poly2

    ! f(x) = sin(1/x) (Pathological oscillation near 0)
    pure function func_pathological(x) result(res)
        real(WP), intent(in) :: x
        real(WP) :: res
        res = sin(1.0_wp / x)
    end function func_pathological

end module integration_test_funcs

! ------------------------------------------------------------------------
! MAIN TEST MODULE
! ------------------------------------------------------------------------
module test_fsps_integration_mod
    use fsps_precision, only: WP
    use fsps_context_types, only: fsps_context_t
    use fsps_integration
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_is_nan
    use integration_test_funcs ! Import the functions to test
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_integration_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for fsps_integration module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_fsps_integration_tests()

        call print_minor_header("fsps_integration")

        call test_trapezoid_array()
        call test_romberg_simple()
        call test_romberg_context()
        call test_romberg_robustness()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_integration_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: TRAPEZOIDAL ARRAY INTEGRATION
    ! ------------------------------------------------------------------------
    subroutine test_trapezoid_array()
        real(WP), dimension(2) :: x_box = [0.0_wp, 10.0_wp]
        real(WP), dimension(2) :: y_box = [5.0_wp, 5.0_wp]
        
        real(WP), dimension(2) :: x_tri = [0.0_wp, 1.0_wp]
        real(WP), dimension(2) :: y_tri = [0.0_wp, 2.0_wp]
        
        real(WP), dimension(2) :: x_rev = [10.0_wp, 0.0_wp]
        
        real(WP), dimension(3) :: x_irr = [0.0_wp, 0.1_wp, 1.0_wp]
        real(WP), dimension(3) :: y_irr ! y = x
        real(WP) :: res

        call print_group("integrate_trapezoid_array")

        ! Case 4.1: Constant Area (Box)
        ! Width=10, Height=5, Area=50
        res = integrate_trapezoid_array(x_box, y_box)
        call assert_float_equals(50.0_wp, res, 1.0e-5_wp, "Constant Area (Box)", total_tests, total_failures)

        ! Case 4.2: Triangle Area
        ! Width=1, Height=2, Area=0.5*1*2 = 1.0
        res = integrate_trapezoid_array(x_tri, y_tri)
        call assert_float_equals(1.0_wp, res, 1.0e-5_wp, "Triangle Area", total_tests, total_failures)

        ! Case 4.3: Unsorted/Reverse Order
        ! Checks that ABS(dx) logic correctly produces positive area
        res = integrate_trapezoid_array(x_rev, y_box)
        call assert_float_equals(50.0_wp, res, 1.0e-5_wp, "Reverse Order (ABS check)", total_tests, total_failures)

        ! Case 4.4: Irregular Grid
        ! x=[0, 0.1, 1], y=x. 
        ! Trapz 1: 0.5 * 0.1 * (0 + 0.1) = 0.005
        ! Trapz 2: 0.5 * 0.9 * (0.1 + 1.0) = 0.495
        ! Sum = 0.500
        y_irr = x_irr ! y = x
        res = integrate_trapezoid_array(x_irr, y_irr)
        call assert_float_equals(0.50_wp, res, 1.0e-5_wp, "Irregular Grid", total_tests, total_failures)

        ! Case 4.5: Error Handling (Size < 2)
        res = integrate_trapezoid_array([1.0_wp], [1.0_wp])
        call assert_is_nan(res, "Size < 2 returns NaN", total_tests, total_failures)

        ! Case 4.6: Error Handling (Mismatch)
        res = integrate_trapezoid_array(x_irr, y_box) ! Size 3 vs 2
        call assert_is_nan(res, "Dimension Mismatch returns NaN", total_tests, total_failures)

    end subroutine test_trapezoid_array

    ! ------------------------------------------------------------------------
    ! TEST SUITE: ROMBERG INTEGRATION (SIMPLE)
    ! ------------------------------------------------------------------------
    subroutine test_romberg_simple()
        real(WP) :: res
        real(WP), parameter :: pi = 3.14159265359_wp

        call print_group("integrate_romberg (Simple Function)")

        ! Case 5.1: Polynomial Exactness (x^4)
        ! Integral of x^4 from 0 to 1 is 1/5 = 0.2
        ! Romberg order 5 (K=5) should handle this with high precision.
        res = integrate_romberg(func_poly4, 0.0_wp, 1.0_wp)
        call assert_float_equals(0.2_wp, res, 1.0e-6_wp, "Polynomial Exactness (x^4)", total_tests, total_failures)

        ! Case 5.2: Transcendental Function (e^x)
        ! Integral of e^x from 0 to 1 is e^1 - e^0 = 1.71828...
        res = integrate_romberg(func_exp, 0.0_wp, 1.0_wp)
        call assert_float_equals(exp(1.0_wp) - 1.0_wp, res, 1.0e-6_wp, "Transcendental (e^x)", total_tests, total_failures)

        ! Case: Constant Function (Verify basic sanity)
        res = integrate_romberg(func_const, 0.0_wp, 10.0_wp)
        call assert_float_equals(10.0_wp, res, 1.0e-6_wp, "Constant Function", total_tests, total_failures)

    end subroutine test_romberg_simple

    ! ------------------------------------------------------------------------
    ! TEST SUITE: ROMBERG INTEGRATION (CONTEXT)
    ! ------------------------------------------------------------------------
    subroutine test_romberg_context()
        ! FIX 1: Use ALLOCATABLE to put large structure on heap, not stack
        type(fsps_context_t), allocatable :: dummy_ctx
        real(WP) :: res

        call print_group("integrate_romberg (Context)")

        allocate(dummy_ctx)

        ! Case 5.3: Context Passing
        ! Integral of x^2 from 0 to 1 is 1/3 ~ 0.333333...
        ! This verifies the generic interface routes to the context-aware function.
        res = integrate_romberg(dummy_ctx, func_ctx_poly2, 0.0_wp, 1.0_wp)
        call assert_float_equals(1.0_wp/3.0_wp, res, 1.0e-6_wp, "Context Interface (x^2)", total_tests, total_failures)

        deallocate(dummy_ctx)

    end subroutine test_romberg_context

    ! ------------------------------------------------------------------------
    ! TEST SUITE: ROBUSTNESS / CONVERGENCE FAILURE
    ! ------------------------------------------------------------------------
    subroutine test_romberg_robustness()
        real(WP) :: res
        
        call print_group("integrate_romberg (Robustness)")

        ! Case 5.4: Convergence Failure
        ! Function: sin(1/x) near zero behaves pathologically (infinite oscillation).
        ! Integrating from 0.0001 to 0.1 creates a scenario hard to resolve 
        ! within standard fixed steps, triggering the "max steps" fallback.
        ! Note: We use a range close to 0 where oscillation is rapid.
        res = integrate_romberg(func_pathological, 1.0e-5_wp, 1.0e-2_wp)
        
        ! We expect NaN because the hardcoded MAX_STEPS (20) in the module 
        ! is likely insufficient for this singularity without adaptive subdivision.
        call assert_is_nan(res, "Non-convergence returns NaN (Pathological func)", total_tests, total_failures)

    end subroutine test_romberg_robustness

end module test_fsps_integration_mod