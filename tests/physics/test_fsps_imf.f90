module test_fsps_imf_mod
    use fsps_precision, only: WP
    use fsps_context_types, only: fsps_context_t
    use fsps_imf
    use fsps_types, only: params
    use fsps_ssp, only: configure_imf_parameters
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_relative_error
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_imf_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for fsps_imf module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_fsps_imf_tests()

        call print_minor_header("fsps_imf")

        call test_salpeter()
        call test_chabrier()
        call test_kroupa()
        call test_imf_weighting()
        call test_compute_weights_normalization()
        call test_analytic_integrals_match_quadrature()
        call test_weights_match_quadrature()
        call test_runtime_imf_limits_are_honoured()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_imf_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: SALPETER IMF (Type 0)
    ! ------------------------------------------------------------------------
    subroutine test_salpeter()
        type(fsps_context_t), allocatable :: ctx
        real(WP), dimension(3) :: m = [0.5_wp, 1.0_wp, 2.0_wp]
        real(WP), dimension(3) :: res
        real(WP) :: expected
        integer :: i
        
        call print_group("Salpeter IMF")

        allocate(ctx)
        ! Setup Context
        ctx%imf_type_val = 0
        ctx%state%salp_ind = 2.35_wp ! Standard Salpeter

        ! Test Number Density (dN/dM)
        do i = 1, size(m)
            res(i) = get_imf_value(ctx, m(i), mass_weighted=.false.)
        end do
        
        ! Check m=1.0 (should be 1.0^-2.35 = 1.0)
        call assert_float_equals(1.0_wp, res(2), 1.0e-6_wp, "Salpeter(1.0)", total_tests, total_failures)
        
        ! Check m=2.0 (should be 2.0^-2.35)
        expected = 2.0_wp**(-2.35_wp)
        call assert_float_equals(expected, res(3), 1.0e-6_wp, "Salpeter(2.0)", total_tests, total_failures)

        ! Test Mass Weighting (M * dN/dM)
        do i = 1, size(m)
            res(i) = get_imf_value(ctx, m(i), mass_weighted=.true.)
        end do
        expected = 2.0_wp * (2.0_wp**(-2.35_wp))
        call assert_float_equals(expected, res(3), 1.0e-6_wp, "Salpeter Mass-Weighted", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_salpeter

    ! ------------------------------------------------------------------------
    ! TEST SUITE: CHABRIER IMF (Type 1)
    ! ------------------------------------------------------------------------
    subroutine test_chabrier()
        type(fsps_context_t), allocatable :: ctx
        real(WP), dimension(2) :: m = [0.1_wp, 2.0_wp] ! Below and above 1.0 break
        real(WP), dimension(2) :: res
        real(WP) :: log_m, log_mc, term, expected
        integer :: i
        
        call print_group("Chabrier IMF")

        allocate(ctx)
        ctx%imf_type_val = 1
        
        ! Note: CHAB_MC, CHAB_SIGMA2, CHAB_IND are from fsps_types (constants)

        do i = 1, size(m)
            res(i) = get_imf_value(ctx, m(i), mass_weighted=.false.)
        end do

        ! Case 1: Low Mass (Log Normal) at m=0.1
        log_m = log10(0.1_wp)
        log_mc = log10(CHAB_MC)
        term = (log_m - log_mc)**2 / (2.0_wp * CHAB_SIGMA2)
        ! Formula: exp(-term) / m
        expected = exp(-term) / 0.1_wp
        call assert_float_equals(expected, res(1), 1.0e-5_wp, "Chabrier Low Mass (<1)", total_tests, total_failures)

        ! Case 2: High Mass (Power Law) at m=2.0
        ! Formula: exp(-term_at_mc) * m^(-ind) / m  (Continuity matching)
        term = log_mc**2 / (2.0_wp * CHAB_SIGMA2)
        expected = exp(-term) * 2.0_wp**(-CHAB_IND) / 2.0_wp
        call assert_float_equals(expected, res(2), 1.0e-5_wp, "Chabrier High Mass (>1)", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_chabrier

    ! ------------------------------------------------------------------------
    ! TEST SUITE: KROUPA IMF (Type 2)
    ! ------------------------------------------------------------------------
    subroutine test_kroupa()
        type(fsps_context_t), allocatable :: ctx
        real(WP), dimension(3) :: m = [0.2_wp, 0.7_wp, 2.0_wp] ! One in each regime
        real(WP), dimension(3) :: res
        real(WP) :: expected
        real(WP), dimension(3) :: alpha
        integer :: i
        
        call print_group("Kroupa IMF")

        allocate(ctx)
        ctx%imf_type_val = 2
        ! Standard Kroupa exponents
        alpha = [1.3_wp, 2.3_wp, 2.3_wp]
        ctx%state%imf_alpha = alpha

        do i = 1, size(m)
            res(i) = get_imf_value(ctx, m(i), mass_weighted=.false.)
        end do

        ! Regime 1: 0.08 <= m < 0.5 (m=0.2) -> m^-1.3
        expected = 0.2_wp**(-1.3_wp)
        call assert_float_equals(expected, res(1), 1.0e-5_wp, "Kroupa Low (0.2)", total_tests, total_failures)

        ! Regime 2: 0.5 <= m < 1.0 (m=0.7) -> C1 * m^-2.3
        ! C1 = 0.5^(-1.3 + 2.3) = 0.5^1.0 = 0.5
        expected = 0.5_wp * 0.7_wp**(-2.3_wp)
        call assert_float_equals(expected, res(2), 1.0e-5_wp, "Kroupa Mid (0.7)", total_tests, total_failures)

        ! Regime 3: m >= 1.0 (m=2.0) -> C2 * m^-2.3
        ! C2 = 0.5^(-1.3 + 2.3) = 0.5
        expected = 0.5_wp * 2.0_wp**(-2.3_wp)
        call assert_float_equals(expected, res(3), 1.0e-5_wp, "Kroupa High (2.0)", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_kroupa

    ! ------------------------------------------------------------------------
    ! TEST SUITE: COMPUTE_IMF_WEIGHTS (Normalization)
    ! ------------------------------------------------------------------------
    subroutine test_compute_weights_normalization()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: mini(:), weights(:)
        integer :: nmass
        
        call print_group("compute_imf_weights (Normalization)")

        allocate(ctx)
        
        ! Setup a simple Salpeter context
        ctx%imf_type_val = 0
        ctx%state%salp_ind = 2.35_wp
        
        ! Define limits covering the isochrone points
        ctx%state%imf_lower_limit = 0.1_wp
        ctx%state%imf_upper_limit = 100.0_wp
        ctx%state%imf_lower_bound = 0.1_wp

        ! Create a mock isochrone mass grid (log-spaced)
        nmass = 10
        allocate(mini(nmass))
        allocate(weights(nmass))
        
        ! Fill mini with values from 0.5 to 10.0
        ! (Well within the limits to avoid edge effect complexity in this test)
        mini = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0]

        ! Run computation
        call compute_imf_weights(ctx, mini, weights, nmass)
        
        call assert_true(all(weights > 0.0_wp), "All weights positive", total_tests, total_failures)
        call assert_true(weights(1) > weights(nmass), "Salpeter decreases with mass", total_tests, total_failures)

        deallocate(ctx, mini, weights)
    end subroutine test_compute_weights_normalization

    ! ------------------------------------------------------------------------
    ! TEST SUITE: MASS WEIGHTING FLAG
    ! ------------------------------------------------------------------------
    subroutine test_imf_weighting()
        type(fsps_context_t), allocatable :: ctx
        real(WP), dimension(1) :: m = [2.0_wp]
        real(WP), dimension(1) :: val_num, val_mass
        
        call print_group("Mass Weighting Flag")

        allocate(ctx)
        ctx%imf_type_val = 0 ! Salpeter
        ctx%state%salp_ind = 2.0_wp ! Simple square law

        ! Get dN/dM
        val_num(1) = get_imf_value(ctx, m(1), mass_weighted=.false.)
        ! Get M * dN/dM
        val_mass(1) = get_imf_value(ctx, m(1), mass_weighted=.true.)

        ! Check relationship
        call assert_float_equals(val_num(1) * m(1), val_mass(1), 1.0e-6_wp, &
                                 "Mass Weighted = Mass * Number Weighted", &
                                 total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_imf_weighting

    ! ------------------------------------------------------------------------
    ! ORACLE TESTS: analytic and Romberg IMF integrals vs dense quadrature
    ! ------------------------------------------------------------------------

    !> Configures ctx for IMF base type imf_type with standard slopes over [0.08, 120] Msun.
    subroutine setup_imf_context(ctx, imf_type)
        type(fsps_context_t), allocatable, intent(out) :: ctx
        integer, intent(in) :: imf_type

        allocate(ctx)
        ctx%imf_type_val = imf_type
        ctx%state%salp_ind = 2.35_wp
        ctx%state%imf_alpha = [1.3_wp, 2.3_wp, 2.3_wp]
        ctx%state%imf_mdave = 0.7_wp
        ctx%state%imf_vdmc = 0.3_wp
        ctx%state%imf_lower_limit = 0.08_wp
        ctx%state%imf_upper_limit = 120.0_wp
        ctx%state%imf_lower_bound = 0.08_wp
    end subroutine setup_imf_context

    !> Composite Simpson integral of get_imf_value over [lo, hi], in u = ln m (dm = m du).
    !> The step is small enough that the slope kinks at segment breakpoints contribute < 1e-10.
    function imf_quadrature(ctx, lo, hi, mass_weighted) result(total)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: lo, hi
        logical, intent(in) :: mass_weighted
        real(WP) :: total

        integer, parameter :: n_steps = 200000           ! even number of Simpson intervals
        real(WP) :: du, m, coeff
        integer :: i

        du = (log(hi) - log(lo)) / real(n_steps, WP)
        total = 0.0_wp
        do i = 0, n_steps
            ! Pin the end points exactly: exp(log(0.08)) rounds below the 0.08 Msun cutoff
            if (i == 0) then
                m = lo
            else if (i == n_steps) then
                m = hi
            else
                m = exp(log(lo) + du * real(i, WP))
            end if
            if (i == 0 .or. i == n_steps) then
                coeff = 1.0_wp
            else if (mod(i, 2) == 1) then
                coeff = 4.0_wp
            else
                coeff = 2.0_wp
            end if
            total = total + coeff * get_imf_value(ctx, m, mass_weighted) * m
        end do
        total = total * du / 3.0_wp
    end function imf_quadrature

    !> integrate_imf_interval_analytic must equal the integral of get_imf_value, for number and
    !> mass weighting, on intervals inside single segments and straddling the 0.5 / 1.0 Msun
    !> (Kroupa) and mdave (Dave) breakpoints. This pins the segment continuity constants.
    subroutine test_analytic_integrals_match_quadrature()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: types(3) = [0, 2, 4]
        real(WP), parameter :: lo(6) = [0.08_wp, 0.3_wp, 0.4_wp, 0.9_wp, 5.0_wp, 0.08_wp]
        real(WP), parameter :: hi(6) = [0.3_wp, 0.7_wp, 2.0_wp, 1.1_wp, 120.0_wp, 120.0_wp]
        logical :: mass_weighted
        integer :: k, j, w
        character(len=96) :: label

        call print_group("Analytic IMF Integrals vs Quadrature (oracle)")

        do k = 1, size(types)
            call setup_imf_context(ctx, types(k))
            do j = 1, size(lo)
                do w = 0, 1
                    mass_weighted = (w == 1)
                    write(label, '(A,I0,A,F5.2,A,F6.2,A,L1)') "type ", types(k), " [", lo(j), ",", hi(j), &
                                                             "] mass_weighted=", mass_weighted
                    call assert_relative_error(imf_quadrature(ctx, lo(j), hi(j), mass_weighted), &
                                               integrate_imf_interval_analytic(ctx, lo(j), hi(j), mass_weighted), &
                                               1.0e-8_wp, trim(label), total_tests, total_failures)
                end do
            end do
            deallocate(ctx)
        end do
    end subroutine test_analytic_integrals_match_quadrature

    !> compute_imf_weights must return, for every IMF type (analytic and Romberg), the number of
    !> stars in each isochrone mass bin per 1 Msun formed:
    !>     w_i = int_{m1_i}^{m2_i} phi dm / int_{lower}^{upper} m phi dm
    !> with bin edges at midpoints between neighbouring masses (first edge = imf_lower_bound, last
    !> edge = the last mass). The existing normalization test only checks positivity.
    subroutine test_weights_match_quadrature()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: nmass = 40
        integer, parameter :: types(5) = [0, 1, 2, 3, 4]
        real(WP) :: mini(nmass), weights(nmass), expected(nmass)
        real(WP) :: m1, m2, total_mass, tol, worst
        integer :: i, k
        character(len=96) :: label

        call print_group("compute_imf_weights vs Quadrature (oracle, all types)")

        ! Log-spaced isochrone-like mass grid strictly inside the IMF limits
        do i = 1, nmass
            mini(i) = 0.1_wp * (1000.0_wp)**(real(i - 1, WP) / real(nmass - 1, WP))
        end do

        do k = 1, size(types)
            call setup_imf_context(ctx, types(k))
            call compute_imf_weights(ctx, mini, weights, nmass)

            total_mass = imf_quadrature(ctx, 0.08_wp, 120.0_wp, .true.)
            do i = 1, nmass
                if (i == 1) then
                    m1 = ctx%state%imf_lower_bound
                else
                    m1 = 0.5_wp * (mini(i - 1) + mini(i))
                end if
                if (i == nmass) then
                    m2 = mini(i)
                else
                    m2 = 0.5_wp * (mini(i) + mini(i + 1))
                end if
                expected(i) = imf_quadrature(ctx, m1, m2, .false.) / total_mass
            end do

            ! Analytic types are exact. Romberg types (1, 3) converge to ~1e-6 on smooth bins, but the
            ! van Dokkum form jumps by +0.059% at m = 25 mc (the rounded constants 0.14 / 0.0443,
            ! shared with legacy FSPS), which slows Romberg convergence in the straddling bin to ~1e-5
            if (types(k) == 1 .or. types(k) == 3) then
                tol = 3.0e-5_wp
            else
                tol = 1.0e-8_wp
            end if

            worst = maxval(abs(weights / expected - 1.0_wp))
            write(label, '(A,I0,A,ES9.2,A)') "type ", types(k), " per-bin weights (max rel err ", worst, ")"
            call assert_true(worst < tol, trim(label), total_tests, total_failures)

            deallocate(ctx)
        end do
    end subroutine test_weights_match_quadrature

    !> The imf_lower_limit / imf_upper_limit parameters (fsps_set_param_float, C API) must reach the
    !> IMF weights: configure_imf_parameters runs before every SSP generation and has to copy them
    !> into the state used by compute_imf_weights and add_remnant_mass. Checks the propagated limits
    !> and lower bin edge, zero weight outside the limits, and per-bin weights normalized by the mass
    !> integral over the restricted range (quadrature oracle).
    subroutine test_runtime_imf_limits_are_honoured()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        integer, parameter :: nmass = 40
        real(WP), parameter :: lo_lim = 0.15_wp, hi_lim = 40.0_wp
        real(WP) :: mini(nmass), weights(nmass), expected(nmass)
        real(WP) :: m1, m2, total_mass
        integer :: i

        call print_group("Runtime IMF Limits Reach the Weights (oracle)")

        allocate(ctx)
        ctx%imf_type_val = 2
        ctx%imf_lower_limit_val = lo_lim
        ctx%imf_upper_limit_val = hi_lim
        ctx%state%isoc_type = 'mist'
        call configure_imf_parameters(ctx, pset)

        call assert_float_equals(lo_lim, ctx%state%imf_lower_limit, 1.0e-12_wp, &
                                 "state lower limit = imf_lower_limit parameter", total_tests, total_failures)
        call assert_float_equals(hi_lim, ctx%state%imf_upper_limit, 1.0e-12_wp, &
                                 "state upper limit = imf_upper_limit parameter", total_tests, total_failures)
        call assert_float_equals(lo_lim, ctx%state%imf_lower_bound, 1.0e-12_wp, &
                                 "first bin edge follows the lower limit (non-Geneva)", total_tests, total_failures)

        do i = 1, nmass
            mini(i) = 0.1_wp * (1000.0_wp)**(real(i - 1, WP) / real(nmass - 1, WP))
        end do
        call compute_imf_weights(ctx, mini, weights, nmass)

        ! Expected: midpoint bins (as in compute_imf_weights) for masses inside the limits, zero outside
        total_mass = imf_quadrature(ctx, lo_lim, hi_lim, .true.)
        do i = 1, nmass
            expected(i) = 0.0_wp
            if (mini(i) < lo_lim .or. mini(i) > hi_lim) cycle
            if (i == 1) then
                m1 = ctx%state%imf_lower_bound
            else
                m1 = 0.5_wp * (mini(i - 1) + mini(i))
            end if
            if (i == nmass) then
                m2 = mini(i)
            else
                m2 = 0.5_wp * (mini(i) + mini(i + 1))
            end if
            expected(i) = imf_quadrature(ctx, m1, m2, .false.) / total_mass
        end do

        call assert_true(all(weights(pack([(i, i = 1, nmass)], mini < lo_lim .or. mini > hi_lim)) == 0.0_wp), &
                         "zero weight outside [imf_lower_limit, imf_upper_limit]", total_tests, total_failures)
        call assert_true(maxval(abs(weights - expected), mask=expected > 0.0_wp) < &
                         1.0e-8_wp * maxval(expected), &
                         "weights normalized over the restricted mass range", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_runtime_imf_limits_are_honoured

end module test_fsps_imf_mod