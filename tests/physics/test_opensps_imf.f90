module test_opensps_imf_mod
    use opensps_precision, only: WP, KP
    use opensps_context_types, only: opensps_context_t
    use opensps_imf
    use opensps_types, only: params
    use opensps_ssp, only: configure_imf_parameters
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_relative_error
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_opensps_imf_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for opensps_imf module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_opensps_imf_tests()

        call print_minor_header("opensps_imf")

        call test_salpeter()
        call test_chabrier()
        call test_kroupa()
        call test_imf_weighting()
        call test_compute_weights_normalization()
        call test_analytic_integrals_match_quadrature()
        call test_weights_match_quadrature()
        call test_runtime_imf_limits_are_honoured()
        call test_bin_integral_equals_imf_integral()
        call test_lognormal_narrow_bins_additive()
        call test_powerlaw_bin_worst_cases()
        call test_bin_weights_tab_paths()
        call test_powerlaw_integral_worst_cases()
        call test_user_imf_segments()
        call test_bin_logs_edge_cases()
        call test_non_monotonic_mass_grid()
        call test_exp_reduced_kp()
        call test_expm1_kp()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_opensps_imf_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: SALPETER IMF (Type 0)
    ! ------------------------------------------------------------------------
    subroutine test_salpeter()
        type(opensps_context_t), allocatable :: ctx
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
        type(opensps_context_t), allocatable :: ctx
        real(WP), dimension(2) :: m = [0.1_wp, 2.0_wp] ! Below and above 1.0 break
        real(WP), dimension(2) :: res
        real(WP) :: log_m, log_mc, term, expected
        integer :: i
        
        call print_group("Chabrier IMF")

        allocate(ctx)
        ctx%imf_type_val = 1
        
        ! Note: CHAB_MC, CHAB_SIGMA2, CHAB_IND are from opensps_types (constants)

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
        type(opensps_context_t), allocatable :: ctx
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
        type(opensps_context_t), allocatable :: ctx
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
        type(opensps_context_t), allocatable :: ctx
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
        type(opensps_context_t), allocatable, intent(out) :: ctx
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
        ! User-defined IMF (type 5): three segments (lower mass, upper mass, slope)
        ctx%state%n_user_imf = 3
        ctx%state%imf_user_alpha(:, 1) = [0.08_wp, 0.5_wp, 1.3_wp]
        ctx%state%imf_user_alpha(:, 2) = [0.5_wp, 1.5_wp, 2.3_wp]
        ctx%state%imf_user_alpha(:, 3) = [1.5_wp, 150.0_wp, 2.7_wp]   ! (upper mass is exclusive: above the 120 of the tests)
    end subroutine setup_imf_context

    !> Composite Simpson integral of get_imf_value over [lo, hi], in u = ln m (dm = m du).
    !> The step is small enough that the slope kinks at segment breakpoints contribute < 1e-10.
    function imf_quadrature(ctx, lo, hi, mass_weighted) result(total)
        type(opensps_context_t), intent(in) :: ctx
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
    !> imf_bin_weights (the SSP operator's IMF weights, each shared edge's power computed once)
    !> equals imf_integral_p per bin, bit for bit: every analytic IMF type, consecutive bins from
    !> 0.07 to 150 M_sun (narrow and wide, across every segment boundary), in several runs.
    subroutine test_bin_integral_equals_imf_integral()
        integer, parameter :: NB = 4000
        integer, parameter :: RUNS(5) = [1, 777, 1554, 2331, 3108]
        real(WP) :: p(IMF_NPAR), edge(0:NB), mini(NB), m1(NB), m2(NB), w(NB), ref
        integer  :: first(NB), t, k, r, k0, k1, n_diff
        character(len=64) :: label

        call print_group("IMF bin weights with shared edges equal imf_integral")
        do k = 0, NB
            ! Log-spaced edges, with a few exact segment boundaries and a cluster of narrow bins
            edge(k) = 0.07_wp * (150.0_wp / 0.07_wp)**(real(k, WP) / NB)
        end do
        edge(1000) = 0.5_wp; edge(1300) = 1.0_wp; edge(2000) = 0.08_wp * 25.0_wp
        do k = 2500, 2600
            edge(k) = edge(2499) * (1.0_wp + 1.0e-7_wp * (k - 2499))
        end do
        do k = 1, NB
            m1(k) = edge(k - 1); m2(k) = edge(k); mini(k) = 0.5_wp * (edge(k - 1) + edge(k))
        end do
        first = 0
        do t = 0, 4
            p = 0.0_wp
            p(IMF_P_SALP) = 2.35_wp
            p(IMF_P_ALPHA:IMF_P_ALPHA + 2) = [1.1_wp, 2.1_wp, 2.6_wp]
            p(IMF_P_VDMC) = 0.08_wp
            p(IMF_P_MDAVE) = 0.7_wp
            p(IMF_P_LOWER_LIMIT) = 0.0_wp
            p(IMF_P_UPPER_LIMIT) = 1.0e3_wp
            do r = 1, size(RUNS)
                k0 = RUNS(r)
                k1 = NB
                if (r < size(RUNS)) k1 = RUNS(r + 1) - 1
                call imf_bin_weights(t, p, 1.0_wp, k1 - k0 + 1, mini(k0), m1(k0), m2(k0), first(k0), w(k0), 1)
            end do
            n_diff = 0
            do k = 1, NB
                ref = 0.0_wp
                if (edge(k) > edge(k - 1)) ref = imf_integral_p(t, p, edge(k - 1), edge(k), .false.)
                if (w(k) /= ref) n_diff = n_diff + 1
            end do
            write(label, '(A,I0)') "IMF type ", t
            call assert_true(n_diff == 0, trim(label), total_tests, total_failures)
        end do
    end subroutine test_bin_integral_equals_imf_integral

    !> The lognormal bin integral of a wide bin (erf difference) equals the sum of its narrow sub-bins
    !> (Gauss-Legendre) to rounding, for Chabrier and van Dokkum, number and mass weighted.
    subroutine test_lognormal_narrow_bins_additive()
        integer, parameter :: NS = 4000
        real(WP) :: p(IMF_NPAR), edge(0:NS), whole, parts, worst
        integer  :: t, k, mw, j
        ! (and far in both tails of a narrow lognormal, vdmc 0.13 with mass limits 30-100 and
        ! 0.0001-0.001 Msun, where an erf difference loses up to 3e-5 of the bin)
        integer, parameter :: NR = 5
        real(WP), parameter :: LO(NR) = [0.1_wp, 0.2_wp, 0.5_wp, 30.0_wp, 1.0e-4_wp], &
                               HI(NR) = [0.9_wp, 1.0_wp, 0.95_wp, 100.0_wp, 1.0e-3_wp]
        character(len=64) :: label

        call print_group("Lognormal IMF bins: wide bin = sum of narrow bins")
        p = 0.0_wp
        p(IMF_P_SALP) = 2.35_wp
        p(IMF_P_ALPHA:IMF_P_ALPHA + 2) = [1.1_wp, 2.1_wp, 2.6_wp]
        p(IMF_P_VDMC) = 0.13_wp
        p(IMF_P_MDAVE) = 0.7_wp
        do t = 1, 3, 2
            do mw = 0, 1
                worst = 0.0_wp
                do j = 1, NR
                    do k = 0, NS
                        edge(k) = LO(j) * (HI(j) / LO(j))**(real(k, WP) / NS)
                    end do
                    whole = imf_integral_p(t, p, edge(0), edge(NS), mw == 1)
                    parts = 0.0_wp
                    do k = 1, NS
                        parts = parts + imf_integral_p(t, p, edge(k - 1), edge(k), mw == 1)
                    end do
                    worst = max(worst, abs(parts / whole - 1.0_wp))
                end do
                write(label, '(A,I0,A,I0)') "IMF type ", t, ", mass weighted ", mw
                call assert_true(worst < 1.0e-13_wp, trim(label), total_tests, total_failures)
            end do
        end do
    end subroutine test_lognormal_narrow_bins_additive

    subroutine test_analytic_integrals_match_quadrature()
        type(opensps_context_t), allocatable :: ctx
        integer, parameter :: types(6) = [0, 1, 2, 3, 4, 5]
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
        type(opensps_context_t), allocatable :: ctx
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

    !> The imf_lower_limit / imf_upper_limit parameters (opensps_set_param_float, C API) must reach the
    !> IMF weights: configure_imf_parameters runs before every SSP generation and has to copy them
    !> into the state used by compute_imf_weights and add_remnant_mass. Checks the propagated limits
    !> and lower bin edge, zero weight outside the limits, and per-bin weights normalized by the mass
    !> integral over the restricted range (quadrature oracle).
    subroutine test_runtime_imf_limits_are_honoured()
        type(opensps_context_t), allocatable :: ctx
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

    !> Power-law bin weights where a difference of powers breaks, against 60-digit references
    !> (tests/data/imf_powerlaw_bins.txt, from tools/reference/imf_powerlaw_bins.py: Python decimal,
    !> independent of this code): bins down to a relative width of 1e-14 (the powers cancel), slopes
    !> within 1e-6 of 1 and exactly 1 (a cancellation divided by a small exponent; ln(hi/lo) of a
    !> rounded ratio), slopes 0.3 to 5 at 0.08 to 300 M_sun, bins up to a factor 10 wide, and either
    !> side of the switch between the series and exp(y) - 1 in imf_bin_weights_tab.
    !> imf_bin_weights_tab (the batched SSP's weights) must be within 1e-14 of the reference in
    !> every category; in FP32 builds its power-law bins are FP32 arithmetic (exponents reduced in
    !> FP64), within 16 ulp of FP32 (1.9e-6; 2.8e-7 on MIST).
    !> The difference of powers (the single SSP's weights before #40) is evaluated on the same bins
    !> and its worst error printed beside it: the failure modes the table form removes, measured.
    subroutine test_powerlaw_bin_worst_cases()
#if defined(KERNEL_SINGLE_PRECISION)
        real(WP), parameter :: TOL = 16.0_wp * real(epsilon(1.0_KP), WP)   ! FP32 IMF weights
#else
        real(WP), parameter :: TOL = 1.0e-14_wp
#endif
        integer, parameter :: NCAT = 6
        character(len=*), parameter :: CAT_NAME(NCAT) = [character(len=40) :: &
            "relative width 1e-14 to 1e-12", "relative width 1e-9 to 1e-6", "relative width 1e-3 to 0.1", &
            "wide bins (factor 2 to 10)", "slope within 1e-6 of 1 (or exactly 1)", "either side of the series switch"]
        character(len=4096) :: root
        character(len=256) :: line
        real(WP) :: p(IMF_NPAR), alpha, lo, hi, ref, llo, dl, w_new(1), w_old(1), mini(1), rw
        real(WP) :: worst_new(NCAT), worst_old(NCAT)
        integer :: u, n, i, ios, cat, cnt(NCAT), first(1)

        call print_group("Power-law IMF bins: worst cases against 60-digit references")
        call get_environment_variable('FSPS_TEST_SOURCE_ROOT', root)
        open(newunit=u, file=trim(root)//'/tests/data/imf_powerlaw_bins.txt', status='old', action='read', iostat=ios)
        call assert_true(ios == 0, "reference table tests/data/imf_powerlaw_bins.txt", total_tests, total_failures)
        if (ios /= 0) return
        read(u, '(A)') line
        read(u, *) n
        worst_new = 0.0_wp; worst_old = 0.0_wp; cnt = 0
        first = 0
        do i = 1, n
            read(u, *) alpha, lo, hi, ref
            rw = (hi - lo) / lo
            if (i > n - 12) then
                cat = 6
            else if (abs(1.0_wp - alpha) < 1.0e-5_wp) then
                cat = 5
            else if (rw < 1.0e-11_wp) then
                cat = 1
            else if (rw < 1.0e-5_wp) then
                cat = 2
            else if (rw < 0.5_wp) then
                cat = 3
            else
                cat = 4
            end if
            p = 0.0_wp
            p(IMF_P_SALP) = alpha
            p(IMF_P_LOWER_LIMIT) = 0.0_wp
            p(IMF_P_UPPER_LIMIT) = 1.0e5_wp      ! above the widest bins (300 to 3000 M_sun)
            mini(1) = 0.5_wp * (lo + hi)
            call imf_bin_logs(lo, hi, llo, dl)
            call imf_bin_weights_tab(0, p, 1.0_wp, 1, mini, [lo], [hi], [llo], [dl], first, w_new, 1)
            if (abs(1.0_wp - alpha) > 1.0e-12_wp) then   ! the difference form (before #40)
                w_old(1) = (hi**(1.0_wp - alpha) - lo**(1.0_wp - alpha)) / (1.0_wp - alpha)
            else
                w_old(1) = log(hi / lo)
            end if
            worst_new(cat) = max(worst_new(cat), abs(w_new(1) / ref - 1.0_wp))
            worst_old(cat) = max(worst_old(cat), abs(w_old(1) / ref - 1.0_wp))
            cnt(cat) = cnt(cat) + 1
        end do
        close(u)
        do cat = 1, NCAT
            write(line, '(A,A,A,I0,A,ES8.1,A,ES8.1)') '# ', trim(CAT_NAME(cat)), ' (', cnt(cat), ' bins): table form ', &
                worst_new(cat), ', difference form ', worst_old(cat)
            write(*, '(A)') trim(line)
            write(line, '(A,ES8.1)') ": table form within ", TOL
            call assert_true(worst_new(cat) <= TOL .and. cnt(cat) > 0, trim(CAT_NAME(cat)) // trim(line), &
                             total_tests, total_failures)
        end do
    end subroutine test_powerlaw_bin_worst_cases

    !> imf_bin_weights_tab takes the right path for every analytic IMF type: power-law bins (table
    !> form), bins across the segment boundaries (0.5 and 1 M_sun for Kroupa, mdave for Dave, the
    !> lognormal breaks of Chabrier and van Dokkum), lognormal bins, first bins of an age (lower edge
    !> from the IMF's lower bound) and points outside the IMF's mass limits (zero). Reference:
    !> imf_integral_p per bin, a difference of powers, whose own error is ~ eps / (|e| w) for a bin of
    !> relative width w and exponent e (here |e| >= 0.1: Kroupa's first slope 1.1): the tolerance is
    !> 1e-13 plus 4 eps / (0.1 w), so the narrow bins here (w = 1e-7) test the paths, and
    !> test_powerlaw_bin_worst_cases tests the accuracy.
    subroutine test_bin_weights_tab_paths()
        integer, parameter :: NB = 4000
        integer, parameter :: RUNS(5) = [1, 777, 1554, 2331, 3108]
        real(WP) :: p(IMF_NPAR), edge(0:NB), mini(NB), m1(NB), m2(NB), llo(NB), dl(NB), w(NB), ref, lo, tol, worst
        integer  :: first(NB), t, k, r, k0, k1, n_bad, n_zero_bad, lim
        character(len=64) :: label

        call print_group("IMF bin weights from the bins' logarithms: paths for every IMF type")
        do k = 0, NB
            edge(k) = 0.07_wp * (150.0_wp / 0.07_wp)**(real(k, WP) / NB)
        end do
        edge(1000) = 0.5_wp; edge(1300) = 1.0_wp; edge(2000) = 0.08_wp * 25.0_wp
        do k = 2500, 2600
            edge(k) = edge(2499) * (1.0_wp + 1.0e-7_wp * (k - 2499))
        end do
        do k = 1, NB
            m1(k) = edge(k - 1); m2(k) = edge(k); mini(k) = 0.5_wp * (edge(k - 1) + edge(k))
        end do
        call imf_bin_logs(m1, m2, llo, dl)
        first = 0
        first(RUNS) = 1      ! the first bin of each run is the first of its age
        do lim = 1, 2
            do t = 0, 4
                p = 0.0_wp
                p(IMF_P_SALP) = 2.35_wp
                p(IMF_P_ALPHA:IMF_P_ALPHA + 2) = [1.1_wp, 2.1_wp, 2.6_wp]
                p(IMF_P_VDMC) = 0.08_wp
                p(IMF_P_MDAVE) = 0.7_wp
                p(IMF_P_LOWER_BOUND) = 0.065_wp
                if (lim == 1) then
                    p(IMF_P_LOWER_LIMIT) = 0.0_wp
                    p(IMF_P_UPPER_LIMIT) = 1.0e3_wp
                else
                    ! Mass limits inside the bins: points outside must get zero
                    p(IMF_P_LOWER_LIMIT) = 0.3_wp
                    p(IMF_P_UPPER_LIMIT) = 40.0_wp
                end if
                do r = 1, size(RUNS)
                    k0 = RUNS(r)
                    k1 = NB
                    if (r < size(RUNS)) k1 = RUNS(r + 1) - 1
                    call imf_bin_weights_tab(t, p, 1.0_wp, k1 - k0 + 1, mini(k0), m1(k0), m2(k0), llo(k0), dl(k0), &
                                             first(k0), w(k0), 1)
                end do
                n_bad = 0; n_zero_bad = 0; worst = 0.0_wp
                do k = 1, NB
                    if (mini(k) < p(IMF_P_LOWER_LIMIT) .or. mini(k) > p(IMF_P_UPPER_LIMIT)) then
                        if (w(k) /= 0.0_wp) n_zero_bad = n_zero_bad + 1
                        cycle
                    end if
                    lo = m1(k)
                    if (first(k) == 1) lo = p(IMF_P_LOWER_BOUND)
                    ref = 0.0_wp
                    if (m2(k) > lo) ref = imf_integral_p(t, p, lo, m2(k), .false.)
                    tol = 1.0e-13_wp + 40.0_wp * epsilon(1.0_wp) * lo / (m2(k) - lo)
#if defined(KERNEL_SINGLE_PRECISION)
                    tol = tol + 16.0_wp * real(epsilon(1.0_KP), WP)   ! FP32 IMF weights
#endif
                    if (ref == 0.0_wp) then
                        if (w(k) /= 0.0_wp) n_bad = n_bad + 1
                    else
                        worst = max(worst, abs(w(k) / ref - 1.0_wp))
                        if (abs(w(k) / ref - 1.0_wp) > tol) n_bad = n_bad + 1
                    end if
                end do
                write(label, '(A,I0,A)') "IMF type ", t, merge(", wide limits     ", ", limits 0.3-40   ", lim == 1)
                call assert_true(n_bad == 0, trim(label) // ": every bin", total_tests, total_failures)
                call assert_true(n_zero_bad == 0, trim(label) // ": zero outside the limits", total_tests, total_failures)
            end do
        end do
    end subroutine test_bin_weights_tab_paths


    !> Power-law IMF integrals where a difference of powers breaks, against 60-digit references
    !> (tests/data/imf_powerlaw_bins.txt, from tools/reference/imf_powerlaw_bins.py: Python decimal):
    !> bins down to a relative width of 1e-14, slopes within 1e-6 of 1 and exactly 1, slopes 0.3 to 5
    !> at 0.08 to 300 M_sun, bins up to a factor 10 wide, and either side of the series switch.
    !> imf_integral_p (integrate_powerlaw, issue #37) must be within 1e-14 in every category; the
    !> difference form it replaced is evaluated beside it and its worst error printed.
    subroutine test_powerlaw_integral_worst_cases()
        real(WP), parameter :: TOL = 1.0e-14_wp
        integer, parameter :: NCAT = 6
        character(len=*), parameter :: CAT_NAME(NCAT) = [character(len=40) :: &
            "relative width 1e-14 to 1e-12", "relative width 1e-9 to 1e-6", "relative width 1e-3 to 0.1", &
            "wide bins (factor 2 to 10)", "slope within 1e-6 of 1 (or exactly 1)", "either side of the series switch"]
        character(len=4096) :: root
        character(len=256) :: line
        real(WP) :: p(IMF_NPAR), alpha, lo, hi, ref, w_new, w_old, rw, e, worst_new(NCAT), worst_old(NCAT)
        integer :: u, n, i, ios, cat, cnt(NCAT)

        call print_group("Power-law IMF integrals: worst cases against 60-digit references")
        call get_environment_variable('FSPS_TEST_SOURCE_ROOT', root)
        open(newunit=u, file=trim(root)//'/tests/data/imf_powerlaw_bins.txt', status='old', action='read', iostat=ios)
        call assert_true(ios == 0, "reference table tests/data/imf_powerlaw_bins.txt", total_tests, total_failures)
        if (ios /= 0) return
        read(u, '(A)') line
        read(u, *) n
        worst_new = 0.0_wp; worst_old = 0.0_wp; cnt = 0
        do i = 1, n
            read(u, *) alpha, lo, hi, ref
            rw = (hi - lo) / lo
            if (i > n - 12) then
                cat = 6
            else if (abs(1.0_wp - alpha) < 1.0e-5_wp) then
                cat = 5
            else if (rw < 1.0e-11_wp) then
                cat = 1
            else if (rw < 1.0e-5_wp) then
                cat = 2
            else if (rw < 0.5_wp) then
                cat = 3
            else
                cat = 4
            end if
            p = 0.0_wp
            p(IMF_P_SALP) = alpha
            p(IMF_P_LOWER_LIMIT) = 0.0_wp
            p(IMF_P_UPPER_LIMIT) = 1.0e5_wp
            w_new = imf_integral_p(0, p, lo, hi, .false.)
            e = 1.0_wp - alpha
            if (abs(e) > 1.0e-12_wp) then
                w_old = (hi**e - lo**e) / e        ! the difference form replaced in #37
            else
                w_old = log(hi / lo)
            end if
            worst_new(cat) = max(worst_new(cat), abs(w_new / ref - 1.0_wp))
            worst_old(cat) = max(worst_old(cat), abs(w_old / ref - 1.0_wp))
            cnt(cat) = cnt(cat) + 1
        end do
        close(u)
        do cat = 1, NCAT
            write(line, '(A,A,A,I0,A,ES8.1,A,ES8.1)') '# ', trim(CAT_NAME(cat)), ' (', cnt(cat), ' bins): expm1 form ', &
                worst_new(cat), ', difference form ', worst_old(cat)
            write(*, '(A)') trim(line)
            call assert_true(worst_new(cat) <= TOL .and. cnt(cat) > 0, trim(CAT_NAME(cat)) // ": within 1e-14", &
                             total_tests, total_failures)
        end do
    end subroutine test_powerlaw_integral_worst_cases

    ! ------------------------------------------------------------------------
    ! User-defined IMF, edge cases of the bin logarithms and mass grid, FP32 exponentials
    ! ------------------------------------------------------------------------

    !> The user IMF (type 5): the segments join continuously at their breakpoints (the cumulative
    !> factor carries the amplitude from one segment to the next), each segment is a power law of its
    !> own slope, and the IMF is zero outside [first lower mass, last upper mass).
    subroutine test_user_imf_segments()
        type(opensps_context_t), allocatable :: ctx
        real(WP), parameter :: EPS = 1.0e-12_wp
        real(WP) :: below, above, a1, a2, a3
        logical :: ok

        call print_group("User-defined IMF: segments join continuously, zero outside")
        call setup_imf_context(ctx, 5)
        a1 = 1.3_wp; a2 = 2.3_wp; a3 = 2.7_wp

        ! first segment: m**-a1, unit amplitude
        call assert_relative_error(get_imf_value(ctx, 0.2_wp, .false.), 0.2_wp**(-a1), EPS, &
                                   "segment 1 is m^-a1", total_tests, total_failures)
        ! continuity at 0.5 and 1.5: the limit from below equals the first value of the next segment
        below = 0.5_wp * (1.0_wp - 1.0e-13_wp); above = 0.5_wp
        call assert_relative_error(get_imf_value(ctx, above, .false.), get_imf_value(ctx, below, .false.), &
                                   1.0e-9_wp, "continuous at the first breakpoint", total_tests, total_failures)
        below = 1.5_wp * (1.0_wp - 1.0e-13_wp); above = 1.5_wp
        call assert_relative_error(get_imf_value(ctx, above, .false.), get_imf_value(ctx, below, .false.), &
                                   1.0e-9_wp, "continuous at the second breakpoint", total_tests, total_failures)
        ! segment 2 and 3 carry the amplitudes of the earlier segments
        call assert_relative_error(get_imf_value(ctx, 1.0_wp, .false.), 0.5_wp**(-a1 + a2) * 1.0_wp**(-a2), EPS, &
                                   "segment 2 amplitude", total_tests, total_failures)
        call assert_relative_error(get_imf_value(ctx, 10.0_wp, .false.), &
                                   0.5_wp**(-a1 + a2) * 1.5_wp**(-a2 + a3) * 10.0_wp**(-a3), EPS, &
                                   "segment 3 amplitude", total_tests, total_failures)
        ! mass weighting multiplies by the mass
        call assert_relative_error(get_imf_value(ctx, 10.0_wp, .true.), 10.0_wp * get_imf_value(ctx, 10.0_wp, .false.), &
                                   EPS, "mass weighting", total_tests, total_failures)
        ok = get_imf_value(ctx, 0.05_wp, .false.) == 0.0_wp .and. get_imf_value(ctx, 150.0_wp, .false.) == 0.0_wp .and. &
             get_imf_value(ctx, 500.0_wp, .false.) == 0.0_wp .and. get_imf_value(ctx, 149.9_wp, .false.) > 0.0_wp
        call assert_true(ok, "zero below the first segment and from the last upper mass on", total_tests, total_failures)
        ! an unknown base type gives zero
        ctx%imf_type_val = 9
        call assert_true(get_imf_value(ctx, 1.0_wp, .false.) == 0.0_wp, "unknown IMF type gives zero", &
                         total_tests, total_failures)
        deallocate(ctx)
    end subroutine test_user_imf_segments

    !> imf_bin_logs: ln m1 and ln(m2/m1) as log1p of the relative width. Empty and invalid bins give
    !> zero; a bin one rounding step wide keeps its width (the naive log(m2/m1) rounds to 0 or 1 ulp of
    !> 1); a wide bin gives the ordinary logarithm.
    subroutine test_bin_logs_edge_cases()
        real(WP) :: llo, dl, m1, m2

        call print_group("imf_bin_logs: empty, invalid, one-ulp and wide bins")
        call imf_bin_logs(1.0_wp, 1.0_wp, llo, dl)
        call assert_true(llo == 0.0_wp .and. dl == 0.0_wp, "empty bin: zero", total_tests, total_failures)
        call imf_bin_logs(2.0_wp, 1.0_wp, llo, dl)
        call assert_true(llo == 0.0_wp .and. dl == 0.0_wp, "reversed bin: zero", total_tests, total_failures)
        call imf_bin_logs(-1.0_wp, 1.0_wp, llo, dl)
        call assert_true(llo == 0.0_wp .and. dl == 0.0_wp, "non-positive lower edge: zero", total_tests, total_failures)

        ! The largest double below 1 and 1: the relative width is 2**-53, so 1 + x rounds to exactly 1
        m1 = nearest(1.0_wp, -1.0_wp); m2 = 1.0_wp
        call imf_bin_logs(m1, m2, llo, dl)
        call assert_relative_error(dl, 2.0_wp**(-53), 1.0e-3_wp, "one-ulp bin keeps its width (1 + x == 1 branch)", &
                                   total_tests, total_failures)
        call assert_true(dl > 0.0_wp, "one-ulp bin is not rounded to zero", total_tests, total_failures)
        call assert_relative_error(llo, log(m1), 1.0e-15_wp, "ln of the lower edge", total_tests, total_failures)

        m1 = 0.3_wp; m2 = 0.9_wp
        call imf_bin_logs(m1, m2, llo, dl)
        call assert_relative_error(dl, log(3.0_wp), 1.0e-14_wp, "wide bin: ln(m2/m1)", total_tests, total_failures)
    end subroutine test_bin_logs_edge_cases

    !> A mass grid that is not increasing yields a bin with its upper edge below its lower edge: the
    !> bin is skipped (weight 0) with a warning, and its neighbours are unaffected.
    subroutine test_non_monotonic_mass_grid()
        type(opensps_context_t), allocatable :: ctx, ref_ctx
        real(WP) :: mini(5), w(5), w_ok(5)
        logical :: ok

        call print_group("compute_imf_weights: a non-monotonic mass grid skips the bad bin")
        call setup_imf_context(ctx, 0)
        ! Bin i spans the midpoints to its neighbours: bin 3 runs from (0.8 + 0.3) / 2 down to (0.3 + 0.2) / 2
        mini = [0.1_wp, 0.8_wp, 0.3_wp, 0.2_wp, 5.0_wp]
        call compute_imf_weights(ctx, mini, w, 5)
        ok = w(3) == 0.0_wp .and. w(1) > 0.0_wp .and. w(5) > 0.0_wp
        call assert_true(ok, "the bin with m2 < m1 has zero weight, the others do not", total_tests, total_failures)
        ! the first bin does not see the bad grid point beyond its own upper edge
        call setup_imf_context(ref_ctx, 0)
        mini = [0.1_wp, 0.8_wp, 1.0_wp, 2.0_wp, 5.0_wp]
        call compute_imf_weights(ref_ctx, mini, w_ok, 5)
        call assert_relative_error(w(1), w_ok(1), 1.0e-14_wp, "bin 1 unaffected", total_tests, total_failures)
        deallocate(ctx, ref_ctx)
    end subroutine test_non_monotonic_mass_grid

    !> exp_reduced_kp(x): exp of an FP64 argument in kernel precision, by exponent reduction. Against
    !> exp(x) in FP64 across the range where the result is normal (a few ulp of KP), with the clamps:
    !> 0 below the smallest normal, huge above the largest, for any argument. Host evaluation: the
    !> library's device code is a separate image that a test executable's target region cannot call
    !> into (the link fails). The functions are called only by imf_bin_weights_tab in FP32 builds, so
    !> an FP64 build never runs them; this test checks the arithmetic in every build.
    subroutine test_exp_reduced_kp()
        integer, parameter :: N = 241
        real(WP) :: x(N + 6)
        real(KP) :: e(N + 6)
        real(WP) :: lnmin, lnmax, ref, worst
        integer :: i

        call print_group("exp_reduced_kp: accuracy, flush to zero, overflow to huge")
        lnmin = real(minexponent(1.0_KP), WP) * log(2.0_wp)       ! below this the result is not normal
        lnmax = real(maxexponent(1.0_KP), WP) * log(2.0_wp)
        do i = 1, N
            x(i) = (lnmin + 2.0_wp) + (lnmax - 2.0_wp - lnmin - 2.0_wp) * real(i - 1, WP) / real(N - 1, WP)
        end do
        x(N + 1) = 0.0_wp
        x(N + 2) = lnmin - 3.0_wp          ! flushes to zero
        x(N + 3) = lnmax + 3.0_wp          ! saturates to huge
        x(N + 4) = -1.0e30_wp              ! clamped argument
        x(N + 5) = 1.0e30_wp
        x(N + 6) = 1.0e-9_wp
        call eval_exp_reduced(size(x), x, e)

        ! The reduction x - n ln 2 uses one FP64 constant for ln 2 (error 2**-53 relative, times n) and
        ! rounds the product n ln 2 (half an ulp): the argument carries up to ~2e-16 |x| of error on top
        ! of the rounding to KP (negligible against FP32's 6e-8 over its range, visible when KP is FP64)
        worst = 0.0_wp
        do i = 1, N
            ref = exp(x(i))
            worst = max(worst, abs(real(e(i), WP) - ref) / ref - 2.5e-16_wp * abs(x(i)))
        end do
        write (*, '(A,ES10.2)') '# exp_reduced_kp: worst relative error beyond the reduction term ', worst
        call assert_true(worst <= 4.0_wp * real(epsilon(1.0_KP), WP), &
                         "within 4 ulp of exp(x) over the normal range (plus the argument reduction's own rounding)", &
                         total_tests, total_failures)
        call assert_true(e(N + 1) == 1.0_KP, "exp(0) = 1 exactly", total_tests, total_failures)
        call assert_true(e(N + 2) == 0.0_KP, "below the smallest normal: 0", total_tests, total_failures)
        call assert_true(e(N + 3) == huge(1.0_KP), "above the largest: huge", total_tests, total_failures)
        call assert_true(e(N + 4) == 0.0_KP .and. e(N + 5) == huge(1.0_KP), "arguments of 1e30 are clamped", &
                         total_tests, total_failures)
        call assert_relative_error(real(e(N + 6), WP), 1.0_wp + 1.0e-9_wp, 4.0_wp * real(epsilon(1.0_KP), WP), &
                                   "exp of a tiny argument", total_tests, total_failures)
    end subroutine test_exp_reduced_kp

    subroutine eval_exp_reduced(n, x, e)
        integer, intent(in) :: n
        real(WP), intent(in) :: x(n)
        real(KP), intent(out) :: e(n)
        integer :: i
        do i = 1, n
            e(i) = exp_reduced_kp(x(i))
        end do
    end subroutine eval_exp_reduced

    !> expm1_kp(y) = exp(y) - 1 without cancellation: against the Taylor sum of the exponential in
    !> FP64, over small arguments (the series branch, |y| < 1/4), at its switch, and for large ones
    !> (exp(y) - 1: loses at most a factor 4); the tiny-argument limit (expm1(y) = y to rounding), and
    !> the identity (1 + expm1(y)) (1 + expm1(-y)) = 1. Host evaluation (see test_exp_reduced_kp).
    subroutine test_expm1_kp()
        integer, parameter :: N = 81
        real(KP) :: y(N + 3), r(N + 3)
        real(WP) :: ref, term, worst_small, worst_large, prod
        integer :: i, k

        call print_group("expm1_kp: no cancellation for small arguments, exp(y) - 1 for large")
        do i = 1, N
            y(i) = real(-2.0_wp + 4.0_wp * real(i - 1, WP) / real(N - 1, WP), KP)
        end do
        y(N + 1) = 1.0e-6_KP
        y(N + 2) = -1.0e-6_KP
        y(N + 3) = 1.0e-30_KP
        call eval_expm1(size(y), y, r)

        worst_small = 0.0_wp
        worst_large = 0.0_wp
        do i = 1, N + 2
            ! Taylor sum in FP64 of the KP-rounded argument
            ref = 0.0_wp
            term = 1.0_wp
            do k = 1, 40
                term = term * real(y(i), WP) / real(k, WP)
                ref = ref + term
            end do
            if (ref == 0.0_wp) cycle
            if (abs(y(i)) < 0.25_KP) then
                worst_small = max(worst_small, abs(real(r(i), WP) - ref) / abs(ref))
            else
                worst_large = max(worst_large, abs(real(r(i), WP) - ref) / abs(ref))
            end if
        end do
        write (*, '(A,2ES10.2)') '# expm1_kp: worst relative error (series, large): ', worst_small, worst_large
        ! the series is truncated at y**7: 5e-10 relative (the FP32 build's 6e-8 is far above it)
        call assert_true(worst_small <= max(8.0_wp * real(epsilon(1.0_KP), WP), 1.0e-9_wp), &
                         "series branch: relative error at the truncation / rounding level", total_tests, total_failures)
        call assert_true(worst_large <= 8.0_wp * real(epsilon(1.0_KP), WP), &
                         "exp(y) - 1 branch: within a few ulp (cancellation <= 4)", total_tests, total_failures)
        call assert_relative_error(real(r(N + 3), WP), 1.0e-30_wp, 1.0e-6_wp, "expm1 of a tiny argument is the argument", &
                                   total_tests, total_failures)
        prod = (1.0_wp + real(r(N + 1), WP)) * (1.0_wp + real(r(N + 2), WP))
        call assert_relative_error(prod, 1.0_wp, 1.0e-9_wp, "(1 + expm1(y)) (1 + expm1(-y)) = 1", total_tests, total_failures)
        call assert_true(all(r(1:N) /= 0.0_KP .or. y(1:N) == 0.0_KP), "nonzero for every nonzero argument", &
                         total_tests, total_failures)
    end subroutine test_expm1_kp

    subroutine eval_expm1(n, y, r)
        integer, intent(in) :: n
        real(KP), intent(in) :: y(n)
        real(KP), intent(out) :: r(n)
        integer :: i
        do i = 1, n
            r(i) = expm1_kp(y(i))
        end do
    end subroutine eval_expm1

end module test_opensps_imf_mod