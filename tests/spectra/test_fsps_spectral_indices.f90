module test_fsps_spectral_indices_mod
    use fsps_precision, only: WP
    use fsps_context_types, only: fsps_context_t
    use fsps_context, only: fsps_context_move_to_device, fsps_context_remove_from_device
    use fsps_spectral_indices, only: compute_spectral_indices, integrate_interval, integrate_ratio_interval
    use fsps_special_functions, only: mag_from_flux
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    real(WP), parameter :: IND_UNDEFINED = 999.0_wp

    public :: run_fsps_spectral_indices_tests, total_failures, total_tests

contains

    subroutine run_fsps_spectral_indices_tests()
        call print_minor_header("fsps_spectral_indices")

        call test_subpixel_precision()
        call test_scalar_accumulation_integrity()
        call test_sloped_continuum_normalization()
        call test_grid_gap_integration()
        call test_single_interior_point()
        call test_ratio_endpoints_legacy()

        call test_equivalent_width_identity()
        call test_magnitude_identity()
        call test_dn4000_ratio()
        call test_flux_ratio_magnitude()

        call test_seesaw_continuum()
        call test_zero_width_sideband()

        call test_out_of_bounds_sentinel()
        call test_zero_continuum_dn4000()
        call test_zero_continuum_flux_ratio()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_spectral_indices_tests

    ! ------------------------------------------------------------------------
    ! Group 1: Integration Kernels
    ! ------------------------------------------------------------------------
    subroutine test_subpixel_precision()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        integer :: n

        call print_group("Sub-Pixel Precision")

        n = 11
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 3000.0_wp, 1.0_wp)
        spec = 1.0_wp

        result = integrate_interval(lambda, spec, 3000.25_wp, 3000.75_wp)
        call assert_float_equals(0.5_wp, result, 1.0e-12_wp, &
                                 "Sub-pixel interval integrates to width", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_subpixel_precision

    subroutine test_scalar_accumulation_integrity()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        integer :: n

        call print_group("Scalar Accumulation Integrity")

        n = 11
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 4000.0_wp, 100.0_wp)
        spec = 2.0_wp

        result = integrate_ratio_interval(lambda, spec, 4000.0_wp, 5000.0_wp, &
                                          0.0_wp, 2.0_wp, 0.0_wp)
        call assert_float_equals(1000.0_wp, result, 1.0e-10_wp, &
                                 "Scalar loop matches full integral", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_scalar_accumulation_integrity

    subroutine test_sloped_continuum_normalization()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        integer :: n

        call print_group("Sloped Continuum Normalization")

        n = 11
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 10.0_wp, 1.0_wp)
        spec = lambda**2

        result = integrate_ratio_interval(lambda, spec, 10.0_wp, 20.0_wp, &
                                          1.0_wp, 0.0_wp, 0.0_wp)
        call assert_float_equals(150.0_wp, result, 1.0e-10_wp, &
                                 "Integral of lambda^2/lambda", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_sloped_continuum_normalization

    subroutine test_grid_gap_integration()
        real(WP), dimension(2) :: lambda, spec
        real(WP) :: result

        call print_group("Grid Gap Integration")

        lambda = [3000.0_wp, 4000.0_wp]
        spec = [1.0_wp, 3.0_wp]

        result = integrate_interval(lambda, spec, 3200.0_wp, 3800.0_wp)
        call assert_float_equals(1200.0_wp, result, 1.0e-10_wp, &
                                 "Trapezoid area across coarse grid", total_tests, total_failures)
    end subroutine test_grid_gap_integration

    !> A bandpass whose ends fall in adjacent grid intervals contains one grid point. The
    !> interior trapezoid sum is then empty (zero), not undefined; on 20 A grids such as
    !> BaSeL this is the case for most Lick indices (CN2, Ca4227, ...), which came out NaN.
    subroutine test_single_interior_point()
        real(WP), dimension(3) :: lambda, spec
        real(WP) :: result

        call print_group("Bandpass Containing One Grid Point")

        lambda = [4070.0_wp, 4090.0_wp, 4110.0_wp]
        spec = [1.0_wp, 3.0_wp, 5.0_wp]

        ! f is linear, f(l) = 1 + (l - 4070)/10, so the area is the mean value times the width
        result = integrate_interval(lambda, spec, 4083.875_wp, 4096.375_wp)
        call assert_float_equals(12.5_wp * (1.0_wp + (4090.125_wp - 4070.0_wp) / 10.0_wp), result, 1.0e-10_wp, &
                                 "area across one interior grid point", total_tests, total_failures)
    end subroutine test_single_interior_point

    !> Flux/continuum at a band edge is interpolated from the ratio at the neighbouring grid points,
    !> as legacy FSPS (INTIND on spec/fc) does, not from the interpolated flux divided by the
    !> continuum at the edge. On a coarse grid with a sloped continuum the two differ: here
    !> 52.946 (ratio) vs 53.373 (flux).
    subroutine test_ratio_endpoints_legacy()
        real(WP), dimension(4) :: lambda, spec
        real(WP) :: result

        call print_group("Ratio Integral Endpoints (legacy INTIND)")

        lambda = [4000.0_wp, 4020.0_wp, 4040.0_wp, 4060.0_wp]
        spec = [1.0_wp, 2.0_wp, 4.0_wp, 8.0_wp]

        ! Continuum 1 + 0.01 (lambda - 4000)
        result = integrate_ratio_interval(lambda, spec, 4005.0_wp, 4035.0_wp, 0.01_wp, 1.0_wp, 4000.0_wp)
        call assert_float_equals(52.94642857142857_wp, result, 1.0e-10_wp, &
                                 "endpoint ratios interpolated from grid-point ratios", total_tests, total_failures)
    end subroutine test_ratio_endpoints_legacy

    ! ------------------------------------------------------------------------
    ! Group 2: Index Types
    ! ------------------------------------------------------------------------
    subroutine test_equivalent_width_identity()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("Equivalent Width Identity (Type 2)")

        n = 201
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 4900.0_wp, 1.0_wp)
        spec = 10.0_wp
        call apply_top_hat(lambda, spec, 4950.0_wp, 4970.0_wp, 5.0_wp)

        indexdef = [4950.0_wp, 4970.0_wp, 4910.0_wp, 4930.0_wp, &
                    5070.0_wp, 5090.0_wp, 2.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        call assert_float_equals(10.0_wp, result, 1.0e-8_wp, &
                                 "EW matches width*(1 - depth)", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_equivalent_width_identity

    subroutine test_magnitude_identity()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("Magnitude Calculation (Type 1)")

        n = 101
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 4000.0_wp, 1.0_wp)
        spec = 1.0_wp

        indexdef = [4020.0_wp, 4040.0_wp, 4000.0_wp, 4010.0_wp, &
                    4090.0_wp, 4100.0_wp, 1.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        call assert_float_equals(0.0_wp, result, 1.0e-12_wp, &
                                 "Flat spectrum yields 0 mag", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_magnitude_identity

    subroutine test_dn4000_ratio()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("Dn4000 Break Strength (Type 3)")

        n = 401
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 3800.0_wp, 1.0_wp)
        spec = 1.0_wp
        where (lambda > 4000.0_wp)
            spec = 2.0_wp
        end where

        indexdef = [3950.0_wp, 4050.0_wp, 3850.0_wp, 3950.0_wp, &
                    4050.0_wp, 4150.0_wp, 3.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        call assert_float_equals(2.0_wp, result, 1.0e-8_wp, &
                                 "Dn4000 ratio equals red/blue", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_dn4000_ratio

    subroutine test_flux_ratio_magnitude()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result, expected
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("Flux Ratio Magnitude (Type 4)")

        n = 101
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 5000.0_wp, 1.0_wp)
        spec = 2.0_wp
        call apply_top_hat(lambda, spec, 5040.0_wp, 5060.0_wp, 4.0_wp)

        indexdef = [5040.0_wp, 5060.0_wp, 5000.0_wp, 5020.0_wp, &
                    5080.0_wp, 5100.0_wp, 4.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        expected = mag_from_flux(2.0_wp)
        call assert_float_equals(expected, result, 1.0e-8_wp, &
                                 "Flux ratio converted to magnitude", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_flux_ratio_magnitude

    ! ------------------------------------------------------------------------
    ! Group 3: Continuum Fitting
    ! ------------------------------------------------------------------------
    subroutine test_seesaw_continuum()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("See-Saw Continuum")

        n = 3001
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 3500.0_wp, 1.0_wp)
        spec = lambda

        indexdef = [4950.0_wp, 5050.0_wp, 3950.0_wp, 4050.0_wp, &
                    5950.0_wp, 6050.0_wp, 2.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        call assert_float_equals(0.0_wp, result, 1.0e-8_wp, &
                                 "Linear continuum yields zero EW", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_seesaw_continuum

    subroutine test_zero_width_sideband()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("Zero-Width Sideband")

        n = 41
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 3990.0_wp, 1.0_wp)
        spec = 3.0_wp

        indexdef = [4005.0_wp, 4015.0_wp, 4000.0_wp, 4000.0_wp + 1.0e-5_wp, &
                    4020.0_wp, 4030.0_wp, 2.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        call assert_true(.not. ieee_is_nan(result), "No NaN for tiny sideband", total_tests, total_failures)
        call assert_float_equals(0.0_wp, result, 1.0e-6_wp, &
                                 "Tiny sideband returns valid EW", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_zero_width_sideband

    ! ------------------------------------------------------------------------
    ! Group 4: Edge Cases & Stability
    ! ------------------------------------------------------------------------
    subroutine test_out_of_bounds_sentinel()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("Out of Bounds Sentinel")

        n = 601
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 3000.0_wp, 10.0_wp)
        spec = 1.0_wp

        indexdef = [4000.0_wp, 4100.0_wp, 2000.0_wp, 2100.0_wp, &
                    5000.0_wp, 5100.0_wp, 2.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        call assert_float_equals(IND_UNDEFINED, result, 1.0e-12_wp, &
                                 "Out-of-bounds returns IND_UNDEFINED", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_out_of_bounds_sentinel

    subroutine test_zero_continuum_dn4000()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("Zero Continuum (Dn4000)")

        n = 401
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 3800.0_wp, 1.0_wp)
        spec = 0.0_wp

        indexdef = [3950.0_wp, 4050.0_wp, 3850.0_wp, 3950.0_wp, &
                    4050.0_wp, 4150.0_wp, 3.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        call assert_float_equals(IND_UNDEFINED, result, 1.0e-12_wp, &
                                 "Zero blue continuum returns undefined", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_zero_continuum_dn4000

    subroutine test_zero_continuum_flux_ratio()
        real(WP), allocatable :: lambda(:), spec(:)
        real(WP) :: result
        real(WP), dimension(7) :: indexdef
        integer :: n

        call print_group("Zero Continuum (Flux Ratio)")

        n = 101
        allocate(lambda(n), spec(n))
        call fill_linear_grid(lambda, 5000.0_wp, 1.0_wp)
        spec = 0.0_wp

        indexdef = [5040.0_wp, 5060.0_wp, 5000.0_wp, 5020.0_wp, &
                    5080.0_wp, 5100.0_wp, 4.0_wp]

        call compute_single_index(lambda, spec, indexdef, result)
        call assert_float_equals(IND_UNDEFINED, result, 1.0e-12_wp, &
                                 "Zero denominator returns undefined", total_tests, total_failures)

        deallocate(lambda, spec)
    end subroutine test_zero_continuum_flux_ratio

    ! ------------------------------------------------------------------------
    ! Helpers
    ! ------------------------------------------------------------------------
    subroutine fill_linear_grid(arr, start_val, step)
        real(WP), dimension(:), intent(out) :: arr
        real(WP), intent(in) :: start_val, step
        integer :: i

        do i = 1, size(arr)
            arr(i) = start_val + step * real(i - 1, WP)
        end do
    end subroutine fill_linear_grid

    subroutine apply_top_hat(lambda, spec, lo, hi, value)
        real(WP), dimension(:), intent(in) :: lambda
        real(WP), dimension(:), intent(inout) :: spec
        real(WP), intent(in) :: lo, hi, value

        where (lambda >= lo .and. lambda <= hi)
            spec = value
        end where
    end subroutine apply_top_hat

    subroutine setup_context(ctx, indexdef)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), dimension(7), intent(in) :: indexdef

        if (associated(ctx%state%indexdefined)) then
            deallocate(ctx%state%indexdefined)
        end if

        allocate(ctx%state%indexdefined(7, 1))
        ctx%state%indexdefined(:, 1) = indexdef
        ctx%state%nindx = 1

        call fsps_context_move_to_device(ctx)
    end subroutine setup_context

    subroutine teardown_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        call fsps_context_remove_from_device(ctx)

        if (associated(ctx%state%indexdefined)) then
            deallocate(ctx%state%indexdefined)
        end if
    end subroutine teardown_context

    subroutine compute_single_index(lambda, spec, indexdef, result)
        real(WP), dimension(:), intent(in) :: lambda, spec
        real(WP), dimension(7), intent(in) :: indexdef
        real(WP), intent(out) :: result
        
        ! Make ctx allocatable
        type(fsps_context_t), allocatable :: ctx 
        real(WP), allocatable :: indices(:)

        ! Allocate on heap
        allocate(ctx) 
        
        call setup_context(ctx, indexdef)
        allocate(indices(1))
        call compute_spectral_indices(ctx, lambda, spec, indices)
        result = indices(1)

        deallocate(indices)
        call teardown_context(ctx)
        
        ! Deallocate
        deallocate(ctx) 
    end subroutine compute_single_index

end module test_fsps_spectral_indices_mod
