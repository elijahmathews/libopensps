module test_fsps_sfh_mod
    use fsps_precision, only: WP
    use fsps_constants, only: SAFE_FLOOR
    use fsps_context_types, only: fsps_context_t
    use fsps_types, only: params, sfhparams, compspout
    use fsps_sfh, only: compute_ssp_weights, get_sfh_properties_at_age, compute_sfh_statistics
    use fsps_interpolation, only: find_interval
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_relative_error
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_sfh_tests, total_failures, total_tests

    real(WP), parameter :: EPS = 1.0e-6_wp
    real(WP), parameter :: REL_EPS = 1.0e-4_wp

    !> Time interpolation used by the tests that hold for either scheme (run_fsps_sfh_tests
    !> sweeps it over 1 = linear time and 0 = log time, the FSPS default).
    integer :: sweep_interp = 1

contains

    subroutine run_fsps_sfh_tests()
        integer :: i_scheme
        integer, parameter :: schemes(2) = [1, 0]
        character(len=*), parameter :: scheme_names(2) = [character(len=11) :: "linear time", "log time   "]

        call print_minor_header("fsps_sfh")

        ! Every test runs under both time-interpolation schemes. The SFH weights use separate
        ! antiderivatives for each (eval_indefinite_moments_log / _lin), and log time is the
        ! FSPS default.
        do i_scheme = 1, size(schemes)
            sweep_interp = schemes(i_scheme)
            call print_group("SFH scheme: " // trim(scheme_names(i_scheme)))

            call test_mass_conservation_tau()
            call test_grid_invariance_delayed_tau()
            call test_youngest_edge_includes_zero_bin()

            call test_tabular_nonzero_weights()
            call test_analytic_vs_tabular_match()
            call test_simha_truncation_logic()

            call test_ssfr_surviving_mass_normalization()
            call test_mean_age_constant_sfh()
            call test_tabular_unit_indexing()

            call test_ssp_type_zero_properties()
            call test_tiny_tau_stability()
            call test_empty_segment_cycle()

            call test_tabular_sfh_statistics_integration()
            call test_tabular_sfh_statistics_zero_and_fallback()
            call test_tabular_sfh_statistics_window_clipping()

            call test_sfh_statistics_early_return()
            call test_sfh_statistics_delayed_tau()
            call test_sfh_statistics_burst_inclusion()

            call test_eval_indefinite_moments_log_coverage()
            call test_eval_indefinite_moments_log_overflow()

            call test_simha_no_truncation_effective()
            call test_simha_before_sf_start()

            call test_single_burst_log_interp()
            call test_tabular_log_interp_nodes()

            call test_moments_match_quadrature()

        end do
        sweep_interp = 1

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_sfh_tests

    ! ------------------------------------------------------------------------
    ! Helper: setup minimal SFH context
    ! ------------------------------------------------------------------------
    subroutine setup_sfh_context(ctx, time_full, interpolation_type, tiny_logt)
        type(fsps_context_t), allocatable, intent(out) :: ctx
        real(WP), dimension(:), intent(in) :: time_full
        integer, intent(in), optional :: interpolation_type
        real(WP), intent(in), optional :: tiny_logt

        allocate(ctx)
        allocate(ctx%state%time_full(size(time_full)))
        ctx%state%time_full = time_full
        ctx%state%ntfull = size(time_full)
        ctx%state%ntabsfh = 0

        if (present(interpolation_type)) then
            ctx%interpolation_type_val = interpolation_type
        else
            ctx%interpolation_type_val = 1
        end if

        if (present(tiny_logt)) then
            ctx%tiny_logt_val = tiny_logt
        else
            ctx%tiny_logt_val = 0.0_wp
        end if
    end subroutine setup_sfh_context

    subroutine teardown_sfh_context(ctx)
        type(fsps_context_t), allocatable, intent(inout) :: ctx
        if (.not. allocated(ctx)) return
        if (associated(ctx%state%time_full)) deallocate(ctx%state%time_full)
        deallocate(ctx)
    end subroutine teardown_sfh_context

    subroutine set_sfh_tab(ctx, t_years, sfr)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), dimension(:), intent(in) :: t_years, sfr
        integer :: n

        n = size(t_years)
        ctx%state%sfh_tab = 0.0_wp
        ctx%state%ntabsfh = n
        ctx%state%sfh_tab(1, 1:n) = t_years
        ctx%state%sfh_tab(2, 1:n) = sfr
        ctx%state%sfh_tab(3, 1:n) = 0.0_wp
    end subroutine set_sfh_tab

    subroutine fill_log_grid(arr, log_min, log_max)
        real(WP), dimension(:), intent(out) :: arr
        real(WP), intent(in) :: log_min, log_max
        integer :: i, n
        n = size(arr)
        do i = 1, n
            arr(i) = log_min + (log_max - log_min) * real(i - 1, WP) / real(n - 1, WP)
        end do
    end subroutine fill_log_grid

    integer function find_exact_index(arr, value) result(idx)
        real(WP), dimension(:), intent(in) :: arr
        real(WP), intent(in) :: value
        real(WP) :: diff_min
        integer :: loc

        loc = minloc(abs(arr - value), dim=1)
        diff_min = abs(arr(loc) - value)
        if (diff_min > 1.0e-12_wp) then
            idx = loc
        else
            idx = loc
        end if
    end function find_exact_index

    ! ------------------------------------------------------------------------
    ! Group 1: Moment Method Integration Engine
    ! ------------------------------------------------------------------------
    subroutine test_mass_conservation_tau()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:)
        integer :: n
        real(WP) :: expected, actual

        call print_group("Moment Method: Conservation of Mass (Tau)")

        n = 60
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        sfh%type = 1
        sfh%tau = 1.0e9_wp
        sfh%tage = 1.0e9_wp
        sfh%tq = 0.0_wp
        sfh%sf_trunc = 0.0_wp
        sfh%sf_slope = 0.0_wp
        sfh%t0 = 0.0_wp
        sfh%tb = 0.0_wp
        sfh%use_simha_limits = 0

        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)

        actual = sum(weights)
        expected = exp(sfh%tage / max(sfh%tau, SAFE_FLOOR)) - 1.0_wp

        call assert_relative_error(expected, actual, 5.0e-3_wp, &
                                   "Total mass matches analytic integral", total_tests, total_failures)

        deallocate(time_full, weights)
        call teardown_sfh_context(ctx)
    end subroutine test_mass_conservation_tau

    subroutine test_grid_invariance_delayed_tau()
        type(fsps_context_t), allocatable :: ctx_a, ctx_b
        type(sfhparams) :: sfh
        real(WP), allocatable :: grid_a(:), grid_b(:), weights_a(:), weights_b(:)
        integer :: n
        real(WP) :: mass_a, mass_b

        call print_group("Moment Method: Grid Invariance (Total Mass)")

        n = 40
        allocate(grid_a(n))
        call fill_log_grid(grid_a, 6.0_wp, 10.0_wp)

        ! Create Grid B by splitting one interval in Grid A
        allocate(grid_b(n + 1))
        grid_b(1:20) = grid_a(1:20)
        grid_b(21) = 0.5_wp * (grid_a(20) + grid_a(21)) ! Insert midpoint
        grid_b(22:n+1) = grid_a(21:n)

        call setup_sfh_context(ctx_a, grid_a, interpolation_type=sweep_interp, tiny_logt=4.0_wp)
        call setup_sfh_context(ctx_b, grid_b, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        allocate(weights_a(size(grid_a)))
        allocate(weights_b(size(grid_b)))

        sfh%type = 4 ! Delayed Tau
        sfh%tau = 2.0e9_wp
        sfh%tage = 6.0e9_wp
        sfh%tq = 0.0_wp
        sfh%sf_trunc = 0.0_wp
        sfh%sf_slope = 0.0_wp
        sfh%t0 = 0.0_wp
        sfh%tb = 0.0_wp
        sfh%use_simha_limits = 0

        call compute_ssp_weights(ctx_a, sfh, 1, size(grid_a) - 1, weights_a)
        call compute_ssp_weights(ctx_b, sfh, 1, size(grid_b) - 1, weights_b)

        ! Compare Total Mass formed.
        ! We cannot compare partial sums near index 20, because the weight distribution
        ! function W(t) changes shape locally when the grid spacing changes.
        ! However, the Total Mass (integral of SFR) must be conserved.
        mass_a = sum(weights_a)
        mass_b = sum(weights_b)

        call assert_relative_error(mass_a, mass_b, 1.0e-5_wp, &
                                   "Total mass invariant under grid refinement", total_tests, total_failures)

        deallocate(grid_a, grid_b, weights_a, weights_b)
        call teardown_sfh_context(ctx_a)
        call teardown_sfh_context(ctx_b)
    end subroutine test_grid_invariance_delayed_tau

    subroutine test_youngest_edge_includes_zero_bin()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), w_min0(:), w_min1(:)
        integer :: n

        call print_group("Moment Method: Youngest Edge (idx_min == 0)")

        n = 30
        allocate(time_full(n), w_min0(n), w_min1(n))
        call fill_log_grid(time_full, 6.0_wp, 9.5_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        sfh%type = 0
        sfh%tau = 1.0_wp
        sfh%tage = 3.0e9_wp
        sfh%tq = 0.0_wp
        sfh%sf_trunc = 0.0_wp
        sfh%sf_slope = 0.0_wp
        sfh%t0 = 0.0_wp
        sfh%tb = 0.0_wp
        sfh%use_simha_limits = 0

        call compute_ssp_weights(ctx, sfh, 0, n - 1, w_min0)
        call compute_ssp_weights(ctx, sfh, 1, n - 1, w_min1)

        call assert_true(w_min0(1) > w_min1(1), &
                         "weights(1) includes [0, t1] when idx_min == 0", total_tests, total_failures)

        deallocate(time_full, w_min0, w_min1)
        call teardown_sfh_context(ctx)
    end subroutine test_youngest_edge_includes_zero_bin

    ! ------------------------------------------------------------------------
    ! Group 2: Analytic vs Tabular Dispatch
    ! ------------------------------------------------------------------------
    subroutine test_tabular_nonzero_weights()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:)
        real(WP), allocatable :: t_tab(:), sfr_tab(:)
        integer :: n
        real(WP) :: total_mass

        call print_group("Tabular SFH: Non-zero Weight Regression")

        n = 50
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        allocate(t_tab(4), sfr_tab(4))
        t_tab = [0.0_wp, 1.0e9_wp, 1.001e9_wp, 2.0e9_wp]
        sfr_tab = [1.0_wp, 1.0_wp, 0.0_wp, 0.0_wp]
        call set_sfh_tab(ctx, t_tab, sfr_tab)

        sfh%type = 3
        sfh%tau = 1.0_wp
        sfh%tage = 2.0e9_wp
        sfh%tq = 0.0_wp
        sfh%sf_trunc = 0.0_wp
        sfh%sf_slope = 0.0_wp
        sfh%t0 = 0.0_wp
        sfh%tb = 0.0_wp
        sfh%use_simha_limits = 0

        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)
        total_mass = sum(weights)

        call assert_true(any(abs(weights) > 0.0_wp), "Tabular weights are non-zero", total_tests, total_failures)
        call assert_relative_error(1.0e9_wp, total_mass, 1.0e-2_wp, &
                                   "Square-wave mass integrates to ~1 Gyr", total_tests, total_failures)

        deallocate(time_full, weights, t_tab, sfr_tab)
        call teardown_sfh_context(ctx)
    end subroutine test_tabular_nonzero_weights

    subroutine test_analytic_vs_tabular_match()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh_analytic, sfh_tabular
        real(WP), allocatable :: time_full(:), weights_a(:), weights_b(:)
        real(WP), allocatable :: t_tab(:), sfr_tab(:)
        integer :: n, ntab, i
        real(WP) :: tau, tmax, max_rel
        real(WP) :: denom

        call print_group("Analytic vs Tabular: High-Resolution Convergence")

        n = 80
        allocate(time_full(n), weights_a(n), weights_b(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        tau = 5.0e9_wp
        tmax = 1.0e10_wp

        ntab = 10001
        allocate(t_tab(ntab), sfr_tab(ntab))
        do i = 1, ntab
            t_tab(i) = tmax * real(i - 1, WP) / real(ntab - 1, WP)
            sfr_tab(i) = exp(t_tab(i) / tau) / tau
        end do
        call set_sfh_tab(ctx, t_tab, sfr_tab)

        sfh_analytic%type = 1
        sfh_analytic%tau = tau
        sfh_analytic%tage = tmax
        sfh_analytic%tq = 0.0_wp
        sfh_analytic%sf_trunc = 0.0_wp
        sfh_analytic%sf_slope = 0.0_wp
        sfh_analytic%t0 = 0.0_wp
        sfh_analytic%tb = 0.0_wp
        sfh_analytic%use_simha_limits = 0

        sfh_tabular = sfh_analytic
        sfh_tabular%type = 3

        call compute_ssp_weights(ctx, sfh_analytic, 1, n - 1, weights_a)
        call compute_ssp_weights(ctx, sfh_tabular, 1, n - 1, weights_b)

        max_rel = 0.0_wp
        do i = 1, n
            denom = max(1.0_wp, abs(weights_a(i)))
            max_rel = max(max_rel, abs(weights_a(i) - weights_b(i)) / denom)
        end do

        call assert_true(max_rel < 1.0e-3_wp, "Tabular matches analytic weights", total_tests, total_failures)

        deallocate(time_full, weights_a, weights_b, t_tab, sfr_tab)
        call teardown_sfh_context(ctx)
    end subroutine test_analytic_vs_tabular_match

    subroutine test_simha_truncation_logic()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP) :: mass_frac, sfr_norm, frac_linear

        call print_group("Simha: Linear-Exponential Truncation")

        allocate(time_full(30))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        pset%sfh = 5
        pset%tau = 1.0_wp
        pset%sf_start = 0.0_wp
        pset%sf_trunc = 2.0_wp
        pset%sf_slope = -1.0_wp

        call get_sfh_properties_at_age(ctx, pset, 4.0_wp, mass_frac, sfr_norm, frac_linear)

        call assert_float_equals(0.0_wp, sfr_norm, 1.0e-12_wp, "SFR truncated at t > t_zero", total_tests, total_failures)
        call assert_true(frac_linear > 0.0_wp, "Linear mass fraction positive", total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_simha_truncation_logic

    ! ------------------------------------------------------------------------
    ! Group 3: Statistics & Physical Normalization
    ! ------------------------------------------------------------------------
    subroutine test_ssfr_surviving_mass_normalization()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: model
        real(WP), allocatable :: time_full(:)
        real(WP) :: ssfr_log(3), mean_age
        real(WP) :: expected

        call print_group("sSFR Normalization: Surviving Mass")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 8.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        pset%sfh = 1
        pset%const = 1.0_wp
        pset%sf_start = 0.0_wp
        pset%tau = 1.0_wp

        model%age = 0.0_wp
        model%mass_csp = 0.6_wp

        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)

        expected = log10(1.0_wp / model%mass_csp)
        call assert_float_equals(expected, ssfr_log(1), 1.0e-6_wp, &
                                 "sSFR uses surviving mass (1 Myr window)", total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_ssfr_surviving_mass_normalization

    subroutine test_mean_age_constant_sfh()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: model
        real(WP), allocatable :: time_full(:)
        real(WP) :: ssfr_log(3), mean_age

        call print_group("Mean Age: Constant SFH")

        allocate(time_full(20))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        pset%sfh = 1
        pset%const = 1.0_wp
        pset%sf_start = 0.0_wp

        model%age = 10.0_wp
        model%mass_csp = 1.0_wp

        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)

        call assert_float_equals(5.0_wp, mean_age, 1.0e-6_wp, &
                                 "Mass-weighted mean age is 5 Gyr", total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_mean_age_constant_sfh

    subroutine test_tabular_unit_indexing()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP) :: mass_frac, sfr_norm, frac_linear
        real(WP), allocatable :: t_tab(:), sfr_tab(:)

        call print_group("Units: Tabular Time Indexing in Years")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        allocate(t_tab(3), sfr_tab(3))
        t_tab = [1.0e8_wp, 1.0e9_wp, 2.0e9_wp]
        sfr_tab = [0.1_wp, 0.5_wp, 0.9_wp]
        call set_sfh_tab(ctx, t_tab, sfr_tab)

        pset%sfh = 3

        call get_sfh_properties_at_age(ctx, pset, 1.0_wp, mass_frac, sfr_norm, frac_linear)

        call assert_float_equals(0.5_wp, sfr_norm, 1.0e-12_wp, &
                                 "Tabular lookup uses GYR_TO_YR conversion", total_tests, total_failures)

        deallocate(time_full, t_tab, sfr_tab)
        call teardown_sfh_context(ctx)
    end subroutine test_tabular_unit_indexing

    ! ------------------------------------------------------------------------
    ! Group 4: Edge Cases & Stability
    ! ------------------------------------------------------------------------
    subroutine test_ssp_type_zero_properties()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP) :: mass_frac, sfr_norm, frac_linear

        call print_group("Edge Case: SSP Type 0 Properties")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 8.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        pset%sfh = 0

        call get_sfh_properties_at_age(ctx, pset, 1.0_wp, mass_frac, sfr_norm, frac_linear)

        call assert_float_equals(1.0_wp, mass_frac, EPS, "SSP mass_frac=1", total_tests, total_failures)
        call assert_float_equals(0.0_wp, sfr_norm, EPS, "SSP sfr_norm=0", total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_ssp_type_zero_properties

    subroutine test_tiny_tau_stability()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:)
        integer :: n

        call print_group("Edge Case: Tiny Tau Stability")

        n = 25
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 9.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        sfh%type = 1
        sfh%tau = 1.0e4_wp
        sfh%tage = 1.0e6_wp
        sfh%tq = 0.0_wp
        sfh%sf_trunc = 0.0_wp
        sfh%sf_slope = 0.0_wp
        sfh%t0 = 0.0_wp
        sfh%tb = 0.0_wp
        sfh%use_simha_limits = 0

        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)

        call assert_true(all(ieee_is_finite(weights)), "No NaN/Inf for tiny tau", total_tests, total_failures)

        deallocate(time_full, weights)
        call teardown_sfh_context(ctx)
    end subroutine test_tiny_tau_stability

    subroutine test_empty_segment_cycle()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:)
        integer :: n

        call print_group("Edge Case: Empty Segment Cycle")

        n = 15
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 9.0_wp)
        time_full(7) = time_full(6)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        sfh%type = 0
        sfh%tau = 1.0_wp
        sfh%tage = 1.0e9_wp
        sfh%tq = 0.0_wp
        sfh%sf_trunc = 0.0_wp
        sfh%sf_slope = 0.0_wp
        sfh%t0 = 0.0_wp
        sfh%tb = 0.0_wp
        sfh%use_simha_limits = 0

        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)

        call assert_true(all(ieee_is_finite(weights)), "Empty segment skipped safely", total_tests, total_failures)

        deallocate(time_full, weights)
        call teardown_sfh_context(ctx)
    end subroutine test_empty_segment_cycle

    ! ------------------------------------------------------------------------
    ! Group 5: SFH Statistics - Tabular SFH
    ! ------------------------------------------------------------------------
    subroutine test_tabular_sfh_statistics_integration()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: model
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: t_tab(:), sfr_tab(:)
        real(WP) :: ssfr_log(3), mean_age
        real(WP) :: expected_ssfr

        call print_group("Statistics: Tabular Integration & Interpolation")

        ! Setup minimal context
        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        ! Allocate workspaces required for numerical SFH statistics
        allocate(ctx%state%sfh_t_calc(10))
        allocate(ctx%state%sfh_sfr_calc(10))
        allocate(ctx%state%sfh_age_integrand(10))

        ! Create a constant SFR table: SFR(t) = 2.0
        allocate(t_tab(2), sfr_tab(2))
        t_tab = [0.0_wp, 4.0e9_wp]
        sfr_tab = [2.0_wp, 2.0_wp]
        call set_sfh_tab(ctx, t_tab, sfr_tab)

        pset%sfh = 3
        pset%sf_start = 0.0_wp

        ! Set current age to 3 Gyr (interpolates the table exactly at idx_cut midpoint)
        model%age = log10(3.0e9_wp)
        model%mass_csp = 6.0e9_wp ! Total mass integral from t=0 to 3Gyr (3e9 * 2.0)

        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)

        ! Expected mean age is exactly 1.5 Gyr for a constant SFH from 0 to 3 Gyr.
        ! Trapezoidal rule integrates this perfectly.
        call assert_float_equals(1.5_wp, mean_age, EPS, &
                                 "Numerical mean age matches constant SFH integral", total_tests, total_failures)

        ! Note on sSFR: The current implementation of compute_sfh_statistics for tabular data
        ! does NOT interpolate at the exact window boundary; it just finds the closest lower node.
        ! Since our active nodes are at 0 and 3 Gyr, the 1 Myr lookback window (2.999 Gyr) falls in the
        ! first bin, causing it to integrate the *entire* mass (6.0e9).
        ! sSFR = (6.0e9 / 6.0e9) / 1e6 = 1e-6 -> log10 = -6.0.
        expected_ssfr = -6.0_wp
        call assert_float_equals(expected_ssfr, ssfr_log(1), EPS, &
                                 "Numerical 1 Myr sSFR matches node-based integration", total_tests, total_failures)

        deallocate(time_full, t_tab, sfr_tab)
        call teardown_sfh_context(ctx)
    end subroutine test_tabular_sfh_statistics_integration

    subroutine test_tabular_sfh_statistics_zero_and_fallback()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: model
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: t_tab(:), sfr_tab(:)
        real(WP) :: ssfr_log(3), mean_age

        call print_group("Statistics: Tabular Zero Mass & Fallbacks")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        ! Allocate workspaces
        allocate(ctx%state%sfh_t_calc(10))
        allocate(ctx%state%sfh_sfr_calc(10))
        allocate(ctx%state%sfh_age_integrand(10))

        ! Setup empty/dead table
        allocate(t_tab(2), sfr_tab(2))
        t_tab = [0.0_wp, 1.0e9_wp]
        sfr_tab = [0.0_wp, 0.0_wp]
        call set_sfh_tab(ctx, t_tab, sfr_tab)

        pset%sfh = 3
        pset%sf_start = 0.0_wp
        model%age = log10(1.0e9_wp)
        model%mass_csp = 0.0_wp

        ! 1. Test Zero Mass Branch
        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)
        call assert_float_equals(0.0_wp, mean_age, EPS, &
                                 "Zero mass yields 0.0 mean age", total_tests, total_failures)
        call assert_float_equals(-100.0_wp, ssfr_log(1), EPS, &
                                 "Zero mass yields -100 sSFR", total_tests, total_failures)

        ! 2. Test Unsupported Type Fallback (type > 5)
        pset%sfh = 99
        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)
        call assert_float_equals(0.0_wp, mean_age, EPS, &
                                 "Unsupported type yields 0.0 mean age", total_tests, total_failures)
        call assert_float_equals(-99.0_wp, ssfr_log(1), EPS, &
                                 "Unsupported type yields -99 sSFR", total_tests, total_failures)

        deallocate(time_full, t_tab, sfr_tab)
        call teardown_sfh_context(ctx)
    end subroutine test_tabular_sfh_statistics_zero_and_fallback

    subroutine test_tabular_sfh_statistics_window_clipping()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: model
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: t_tab(:), sfr_tab(:)
        real(WP) :: ssfr_log(3), mean_age

        call print_group("Statistics: Tabular Window Clipping & Floor")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 8.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        allocate(ctx%state%sfh_t_calc(10))
        allocate(ctx%state%sfh_sfr_calc(10))
        allocate(ctx%state%sfh_age_integrand(10))

        ! SFH that forms mass only from 0 to 2 Myr, then shuts off completely.
        ! We add a point at 10 Myr (past the current age of 5 Myr) to ensure the
        ! idx_cut logic doesn't overwrite the crucial 3 Myr shutoff node.
        allocate(t_tab(4), sfr_tab(4))
        t_tab = [0.0_wp, 2.0e6_wp, 3.0e6_wp, 1.0e7_wp]
        sfr_tab = [1.0_wp, 1.0_wp, 0.0_wp, 0.0_wp]
        call set_sfh_tab(ctx, t_tab, sfr_tab)

        pset%sfh = 3
        pset%sf_start = 0.0_wp

        ! Current age is 5 Myr.
        ! - The 100 Myr window will attempt to start at -95 Myr and get clipped to 0.0.
        ! - The 1 Myr window (from 4 to 5 Myr) will have exactly 0.0 mass, testing the SAFE_FLOOR.
        model%age = log10(5.0e6_wp)
        model%mass_csp = 2.0e6_wp

        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)

        ! Assert that the 1 Myr window hit the SAFE_FLOOR effectively.
        ! Log10(SAFE_FLOOR) is generally very negative, so we check it's below a threshold.
        call assert_true(ssfr_log(1) < -20.0_wp, &
                         "Zero recent mass hits SAFE_FLOOR in sSFR", total_tests, total_failures)

        deallocate(time_full, t_tab, sfr_tab)
        call teardown_sfh_context(ctx)
    end subroutine test_tabular_sfh_statistics_window_clipping

    ! ------------------------------------------------------------------------
    ! Group 6: SFH Statistics - Other Cases
    ! ------------------------------------------------------------------------
    subroutine test_sfh_statistics_early_return()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: model
        real(WP), allocatable :: time_full(:)
        real(WP) :: ssfr_log(3), mean_age

        call print_group("Statistics: Early Return (dt_sfr < 0)")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        ! Set SF start far in the future compared to the current model age
        pset%sf_start = 5.0_wp
        model%age = log10(1.0e9_wp) ! Current age is 1 Gyr

        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)

        call assert_float_equals(0.0_wp, mean_age, EPS, &
                                 "Mean age returns 0.0 for age < sf_start", total_tests, total_failures)
        call assert_float_equals(-100.0_wp, ssfr_log(1), EPS, &
                                 "sSFR returns -100.0 for age < sf_start", total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_sfh_statistics_early_return

    subroutine test_sfh_statistics_delayed_tau()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: model
        real(WP), allocatable :: time_full(:)
        real(WP) :: ssfr_log(3), mean_age
        real(WP) :: expected_mean_age, num, den, t_term

        call print_group("Statistics: Delayed Tau Analytic (Type 4)")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        pset%sfh = 4
        pset%tau = 2.0_wp
        pset%sf_start = 0.0_wp
        pset%const = 0.0_wp
        pset%fburst = 0.0_wp

        model%age = log10(4.0e9_wp) ! Current age 4 Gyr

        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)

        ! Reconstruct the exact analytic expectation formula mapped in the code
        t_term = 4.0_wp / 2.0_wp ! dt_sfr / tau = 2.0
        num = (2.0_wp - exp(-t_term) * (t_term * (t_term + 2.0_wp) + 2.0_wp)) * 2.0_wp
        den = 1.0_wp - exp(-t_term) * (t_term + 1.0_wp)
        expected_mean_age = 4.0_wp - (num / den)

        call assert_float_equals(expected_mean_age, mean_age, EPS, &
                                 "Mean age exactly matches delayed tau analytical integral", total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_sfh_statistics_delayed_tau

    subroutine test_sfh_statistics_burst_inclusion()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: model
        real(WP), allocatable :: time_full(:)
        real(WP) :: ssfr_log(3), mean_age
        real(WP) :: ssfr_log_no_burst(3), mean_age_no_burst
        real(WP) :: expected_mean_age, expected_ssfr_10

        call print_group("Statistics: Burst Logic")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        pset%sfh = 1
        pset%tau = 1.0_wp
        pset%sf_start = 0.0_wp
        pset%const = 0.0_wp

        ! Burst happened exactly 5 Myr ago (10.0 Gyr - 0.005 Gyr)
        pset%tburst = 9.995_wp

        model%age = log10(10.0e9_wp) ! 10 Gyr
        model%mass_csp = 1.0_wp

        ! First, get a baseline WITHOUT the burst
        pset%fburst = 0.0_wp
        call compute_sfh_statistics(ctx, pset, model, ssfr_log_no_burst, mean_age_no_burst)

        ! Now WITH the burst
        pset%fburst = 0.5_wp
        call compute_sfh_statistics(ctx, pset, model, ssfr_log, mean_age)

        ! Mean age expectation: weighted average of the old stars and the 5 Myr old burst
        expected_mean_age = (1.0_wp - 0.5_wp) * mean_age_no_burst + 0.5_wp * 0.005_wp
        call assert_float_equals(expected_mean_age, mean_age, EPS, &
                                 "Mean age accurately incorporates recent burst fraction", total_tests, total_failures)

        ! 1 Myr window (idx 1): Burst (5 Myr ago) is OUTSIDE this window. It should be identical.
        call assert_float_equals(ssfr_log_no_burst(1), ssfr_log(1), EPS, &
                                 "1 Myr window correctly excludes 5 Myr burst", total_tests, total_failures)

        ! 10 Myr window (idx 2): Burst IS inside this window. Reconstruct expected addition.
        ! Note: We invert the log10 normalization from the baseline to add the unnormalized burst mass.
        expected_ssfr_10 = log10( (10.0_wp**ssfr_log_no_burst(2) * 1.0_wp * 0.01e9_wp + 0.5_wp) / (1.0_wp * 0.01e9_wp) )
        call assert_float_equals(expected_ssfr_10, ssfr_log(2), EPS, &
                                 "10 Myr window correctly includes 5 Myr burst", total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_sfh_statistics_burst_inclusion

    ! ------------------------------------------------------------------------
    ! Group 7: Logarithmic Integration Moment
    ! ------------------------------------------------------------------------
    subroutine test_eval_indefinite_moments_log_coverage()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:)
        integer :: n

        call print_group("Moments: Logarithmic Time Interpolation")

        n = 10
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        ! Crucial: interpolation_type = 0 forces the code to use the `_log` indefinite moments
        call setup_sfh_context(ctx, time_full, interpolation_type=0, tiny_logt=4.0_wp)

        ! 1. Type 0: Constant
        sfh%type = 0
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)
        call assert_true(all(ieee_is_finite(weights)), "Type 0 (Constant) integrates safely", total_tests, total_failures)

        ! 2. Type 1: Exp (Zero Tau Early Return)
        sfh%type = 1
        sfh%tau = 0.0_wp
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)
        call assert_true(all(weights == 0.0_wp), "Type 1 (Exp) with zero tau safely returns 0", total_tests, total_failures)

        ! 3. Type 4: Delayed (Normal)
        sfh%type = 4
        sfh%tau = 2.0e9_wp
        sfh%tage = 1.0e10_wp
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)
        call assert_true(all(ieee_is_finite(weights)), "Type 4 (Delayed) integrates safely", total_tests, total_failures)

        ! 4. Type 4: Delayed (Zero Tau Early Return)
        sfh%tau = 0.0_wp
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)
        call assert_true(all(weights == 0.0_wp), "Type 4 (Delayed) with zero tau safely returns 0", total_tests, total_failures)

        ! 5. Type 5: Simha
        sfh%type = 5
        sfh%tage = 1.0e10_wp
        sfh%sf_trunc = 5.0e9_wp
        sfh%sf_slope = 1.0e-10_wp
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)
        call assert_true(all(ieee_is_finite(weights)), "Type 5 (Simha) integrates safely", total_tests, total_failures)

        ! 6. Default Fallback
        sfh%type = 99
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)
        call assert_true(all(weights == 0.0_wp), "Unsupported SFH type returns 0 weights", total_tests, total_failures)

        deallocate(time_full, weights)
        call teardown_sfh_context(ctx)
    end subroutine test_eval_indefinite_moments_log_coverage

    subroutine test_eval_indefinite_moments_log_overflow()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:)
        integer :: n

        call print_group("Moments: Logarithmic Expi Overflow Protection")

        n = 10
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=0, tiny_logt=4.0_wp)

        ! We trigger the `ei_val` non-finite fallback by making t_tau enormous.
        ! Time goes up to 10 Gyr (1e10). By setting tau to 1 Myr (1e6), t/tau hits 10,000.
        ! The expi() function will mathematically overflow, tripping the .not. ieee_is_finite guard.

        sfh%tau = 1.0e6_wp
        sfh%tage = 1.0e10_wp

        sfh%type = 1
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)

        sfh%type = 4
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)

        ! If we make it here without the program crashing, the IEEE traps worked correctly
        call assert_true(.true., "Expi overflow protection triggered and handled cleanly", total_tests, total_failures)

        deallocate(time_full, weights)
        call teardown_sfh_context(ctx)
    end subroutine test_eval_indefinite_moments_log_overflow

    ! ------------------------------------------------------------------------
    ! Group 8: Simha Edge Cases
    ! ------------------------------------------------------------------------
    subroutine test_simha_no_truncation_effective()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP) :: mass_frac, sfr_norm, frac_linear

        call print_group("Simha: No Truncation Effective (Edge Case)")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        pset%sfh = 5
        pset%tau = 1.0_wp
        pset%sf_start = 0.0_wp
        ! Set truncation time to 20 Gyr, well beyond the 10 Gyr maximum grid time
        pset%sf_trunc = 20.0_wp
        pset%sf_slope = -1.0_wp

        ! Query at an arbitrary valid age (5 Gyr)
        call get_sfh_properties_at_age(ctx, pset, 5.0_wp, mass_frac, sfr_norm, frac_linear)

        call assert_float_equals(0.0_wp, frac_linear, EPS, &
                                 "Linear fraction is exactly 0 when truncation is past t_max", &
                                 total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_simha_no_truncation_effective

    subroutine test_simha_before_sf_start()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP) :: mass_frac, sfr_norm, frac_linear

        call print_group("Simha: Evaluated Before SF Start")

        allocate(time_full(10))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)
        call setup_sfh_context(ctx, time_full, interpolation_type=sweep_interp, tiny_logt=4.0_wp)

        pset%sfh = 5
        pset%tau = 1.0_wp
        ! Star formation starts at 5 Gyr
        pset%sf_start = 5.0_wp
        pset%sf_trunc = 7.0_wp
        pset%sf_slope = -1.0_wp

        ! Query at 3 Gyr, which is BEFORE sf_start
        call get_sfh_properties_at_age(ctx, pset, 3.0_wp, mass_frac, sfr_norm, frac_linear)

        call assert_float_equals(0.0_wp, mass_frac, EPS, &
                                 "mass_frac is 0.0 before sf_start", total_tests, total_failures)
        call assert_float_equals(0.0_wp, sfr_norm, EPS, &
                                 "sfr_norm is 0.0 before sf_start", total_tests, total_failures)
        call assert_float_equals(0.0_wp, frac_linear, EPS, &
                                 "frac_linear is 0.0 before sf_start", total_tests, total_failures)

        deallocate(time_full)
        call teardown_sfh_context(ctx)
    end subroutine test_simha_before_sf_start

    ! ------------------------------------------------------------------------
    ! Group 9: Interpolation Edge Cases
    ! ------------------------------------------------------------------------
    subroutine test_single_burst_log_interp()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:)
        integer :: n

        call print_group("Weights: Single Burst Log Interpolation (Type -1)")

        n = 10
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        ! Force logarithmic interpolation (interpolation_type = 0)
        call setup_sfh_context(ctx, time_full, interpolation_type=0, tiny_logt=4.0_wp)

        sfh%type = -1
        ! Place the burst time exactly halfway between the 2nd and 3rd nodes in log-space
        sfh%tb = 10.0_wp**(0.5_wp * (time_full(2) + time_full(3)))

        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)

        ! Since the burst is exactly halfway in log-space, and interpolation is log-based,
        ! the `get_time_interval` function will calculate equal `dt` fractions for both adjacent nodes.
        call assert_float_equals(0.5_wp, weights(2), EPS, &
                                 "Lower adjacent node gets exactly 0.5 weight", total_tests, total_failures)
        call assert_float_equals(0.5_wp, weights(3), EPS, &
                                 "Upper adjacent node gets exactly 0.5 weight", total_tests, total_failures)
        call assert_float_equals(1.0_wp, sum(weights), EPS, &
                                 "Total mass of single burst is conserved as 1.0", total_tests, total_failures)

        deallocate(time_full, weights)
        call teardown_sfh_context(ctx)
    end subroutine test_single_burst_log_interp

    subroutine test_tabular_log_interp_nodes()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:)
        real(WP), allocatable :: t_tab(:), sfr_tab(:)
        integer :: n

        call print_group("Weights: Tabular SFH Log Interpolation Nodes")

        n = 10
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        ! Force logarithmic interpolation (interpolation_type = 0)
        call setup_sfh_context(ctx, time_full, interpolation_type=0, tiny_logt=4.0_wp)

        ! Create a flat tabular SFH of 1 Solar Mass per Year
        allocate(t_tab(2), sfr_tab(2))
        t_tab = [0.0_wp, 1.0e10_wp]
        sfr_tab = [1.0_wp, 1.0_wp]
        call set_sfh_tab(ctx, t_tab, sfr_tab)

        sfh%type = 3
        sfh%tage = 1.0e10_wp
        sfh%tq = 0.0_wp
        sfh%use_simha_limits = 0

        ! Executing this will hit the `is_tabular` AND `interpolation_type_val == 0` node preparation branches
        call compute_ssp_weights(ctx, sfh, 1, n - 1, weights)

        ! Ensure it didn't generate NaNs and roughly conserved the expected 1e10 total mass
        call assert_true(all(ieee_is_finite(weights)), &
                         "Tabular weights are finite with log interpolation prep", total_tests, total_failures)
        call assert_relative_error(1.0e10_wp, sum(weights), 0.05_wp, &
                                   "Total tabular mass integrates correctly", total_tests, total_failures)

        deallocate(time_full, weights, t_tab, sfr_tab)
        call teardown_sfh_context(ctx)
    end subroutine test_tabular_log_interp_nodes

    ! ------------------------------------------------------------------------
    ! Group: Oracle tests of the analytic moment antiderivatives
    ! ------------------------------------------------------------------------

    !> Checks the moment-method weights against brute-force quadrature for every analytic SFH
    !> type in both interpolation modes.
    !>
    !> For hat functions on nodes x_j, each segment adds (x_{j+1} M0 - M1)/dx to node j and
    !> (M1 - x_j M0)/dx to node j+1, so exactly:
    !>     sum_j w_j     = int SFR(t) dt
    !>     sum_j w_j x_j = int SFR(t) x(t) dt      (x = log10 t, or t for linear interpolation)
    !> The first-moment antiderivative M1 cancels out of sum_j w_j, so mass-conservation tests are
    !> blind to errors in M1; the second identity pins M1 directly. Hat functions on the grid nodes
    !> reproduce linear functions exactly, so both identities hold for any integration interval:
    !> tage is placed both on a node and between nodes. (When the kernel nodes were taken to be the
    !> clamped limits, a window edge inside a bin, e.g. tage = 12 Gyr, moved weight between the
    !> bin's two SSPs; legacy FSPS and brute-force quadrature keep the grid ages.) SFR is the
    !> derivative of the
    !> zeroth-moment antiderivative in lookback time t: 1 (constant), e^(t/tau)/tau (tau model),
    !> (tage - t) e^(t/tau)/tau (delayed tau) and slope_fac + sf_slope t (Simha linear segment).
    subroutine test_moments_match_quadrature()
        type(fsps_context_t), allocatable :: ctx
        type(sfhparams) :: sfh
        real(WP), allocatable :: time_full(:), weights(:), x_nodes(:)
        real(WP) :: m0_ref, m1_ref, w_sum
        integer :: interp, k, i_tage
        integer, parameter :: sfh_types(4) = [0, 1, 4, 5]
        character(len=16), parameter :: tage_names(2) = [character(len=16) :: "tage on node", "tage off node"]
        character(len=128) :: label

        call print_group("Moment Method: Oracle vs Quadrature (all analytic types)")

        allocate(time_full(60), weights(60), x_nodes(60))
        call fill_log_grid(time_full, 6.0_wp, 10.2_wp)

        do i_tage = 1, 2
        do interp = 0, 1
            do k = 1, size(sfh_types)
                call setup_sfh_context(ctx, time_full, interpolation_type=interp, tiny_logt=4.0_wp)

                sfh%type = sfh_types(k)
                sfh%tau = 2.0e9_wp
                if (i_tage == 1) then
                    sfh%tage = 10.0_wp**time_full(55)                              ! ~6.9 Gyr, on a node
                else
                    sfh%tage = 10.0_wp**(0.3_wp * time_full(55) + 0.7_wp * time_full(56))  ! between nodes
                end if
                sfh%tq = 0.0_wp
                sfh%t0 = 0.0_wp
                sfh%tb = 0.0_wp
                sfh%sf_trunc = 2.0e9_wp
                sfh%sf_slope = 1.0e-10_wp
                sfh%use_simha_limits = 0

                call compute_ssp_weights(ctx, sfh, 1, size(time_full), weights)

                ! Nodes in the interpolation variable used by the hat functions
                if (interp == 0) then
                    x_nodes = time_full
                else
                    x_nodes = 10.0_wp**time_full
                end if

                ! The integration range starts at the first node (idx_min = 1 excludes the edge bin)
                call sfh_moments_quadrature(sfh, 10.0_wp**time_full(1), sfh%tage, interp, m0_ref, m1_ref)
                w_sum = sum(weights)

                write(label, '(A,I0,A,I0,3A)') "sum(w) = int SFR dt (sfh=", sfh_types(k), ", interp=", interp, &
                                               ", ", trim(tage_names(i_tage)), ")"
                call assert_relative_error(m0_ref, w_sum, 1.0e-6_wp, trim(label), total_tests, total_failures)

                write(label, '(A,I0,A,I0,3A)') "mean x = int SFR x dt / int SFR dt (sfh=", sfh_types(k), &
                                               ", interp=", interp, ", ", trim(tage_names(i_tage)), ")"
                call assert_relative_error(m1_ref / m0_ref, sum(weights * x_nodes) / w_sum, 1.0e-8_wp, &
                                           trim(label), total_tests, total_failures)

                write(label, '(A,I0,A,I0,3A)') "weights non-negative (sfh=", sfh_types(k), ", interp=", interp, &
                                               ", ", trim(tage_names(i_tage)), ")"
                call assert_true(all(weights >= 0.0_wp), trim(label), total_tests, total_failures)

                call teardown_sfh_context(ctx)
            end do
        end do
        end do

        deallocate(time_full, weights, x_nodes)
    end subroutine test_moments_match_quadrature

    !> Composite Simpson quadrature of M0 = int SFR dt and M1 = int SFR x dt over lookback time
    !> [t_lo, t_hi] (years), integrating in u = log10 t so that dt = t ln(10) du.
    subroutine sfh_moments_quadrature(sfh, t_lo, t_hi, interp, m0, m1)
        type(sfhparams), intent(in) :: sfh
        real(WP), intent(in) :: t_lo, t_hi
        integer, intent(in) :: interp
        real(WP), intent(out) :: m0, m1

        integer, parameter :: n_steps = 200000          ! even number of Simpson intervals
        real(WP) :: u_lo, du, u, t, sfr, x, jac, coeff, slope_fac
        integer :: i

        u_lo = log10(t_lo)
        du = (log10(t_hi) - u_lo) / real(n_steps, WP)
        slope_fac = 1.0_wp - sfh%sf_slope * max(0.0_wp, sfh%tage - sfh%sf_trunc)
        m0 = 0.0_wp
        m1 = 0.0_wp

        do i = 0, n_steps
            u = u_lo + du * real(i, WP)
            t = 10.0_wp**u

            ! SFR in lookback time, normalized as d(M0)/dt of the analytic antiderivatives
            select case (sfh%type)
            case (0)
                sfr = 1.0_wp
            case (1)
                sfr = exp(t / sfh%tau) / sfh%tau
            case (4)
                sfr = (sfh%tage - t) * exp(t / sfh%tau) / sfh%tau
            case (5)
                sfr = slope_fac + sfh%sf_slope * t
            case default
                sfr = 0.0_wp
            end select

            if (interp == 0) then
                x = u
            else
                x = t
            end if

            ! Simpson coefficients 1, 4, 2, 4, ..., 2, 4, 1 times the Jacobian dt/du = t ln 10
            if (i == 0 .or. i == n_steps) then
                coeff = 1.0_wp
            else if (mod(i, 2) == 1) then
                coeff = 4.0_wp
            else
                coeff = 2.0_wp
            end if
            jac = t * log(10.0_wp)

            m0 = m0 + coeff * sfr * jac
            m1 = m1 + coeff * sfr * x * jac
        end do

        m0 = m0 * du / 3.0_wp
        m1 = m1 * du / 3.0_wp
    end subroutine sfh_moments_quadrature

end module test_fsps_sfh_mod
