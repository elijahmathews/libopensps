program test_context_isolation
    !> Integration tests for contexts that share a setup cache entry.
    !>
    !> Contexts with identical library selections share one read-only setup cache (stellar
    !> libraries, isochrones, tables). Anything a context writes after setup, or that depends on a
    !> context-specific setting, must not leak into another context through that cache:
    !>   1. setup-time nebular line kernels built from nebular_smooth_init,
    !>   2. per-call nebular line kernels rebuilt from each context's sigma_smooth,
    !>   3. cache reference counting across create/destroy,
    !>   4. a context that hits an existing entry reads no data and reproduces the first context's
    !>      SSP and CSP output bit for bit, while settings the shared data depend on (cloudy_dust,
    !>      compute_vega_mags) select a separate entry.
    !> Requires the FSPS data tree (OPENSPS_DATA_HOME / FSPS_DATA_HOME), like the regression tests.

    use fsps_precision, only: WP, KP
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy, fsps_set_param_int, fsps_set_param_float
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset
    use fsps_context_types, only: fsps_context_t
    use fsps_initialization, only: fsps_shared_load_count
    use fsps_ssp, only: generate_ssp_grid
    use fsps_csp, only: compute_csp_scenario
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true, &
                              assert_int_equals
    implicit none

    real(WP), parameter :: SAME_RTOL = max(1.0e-10_wp, 100.0_wp * real(epsilon(1.0_KP), WP))

    integer, parameter :: ZMET = 10
    integer :: total_tests = 0
    integer :: total_failures = 0

    call print_major_header("Context Isolation (shared setup cache)")

    call test_setup_kernels_follow_own_smoothing()
    call test_per_call_kernels_are_private()
    call test_cache_refcount_bookkeeping()
    call test_cache_hit_reuses_shared_data()

    call print_summary_line("Context Isolation", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    !> Configures a context for nebular line emission and runs setup.
    subroutine setup_nebular_context(ctx, setup_gaussians, smooth_init)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: setup_gaussians
        real(WP), intent(in) :: smooth_init
        integer :: status

        call fsps_create(ctx)
        call fsps_set_param_int(ctx, 'add_neb_emission', 1, status)
        call fsps_set_param_int(ctx, 'setup_nebular_gaussians', setup_gaussians, status)
        call fsps_set_param_float(ctx, 'nebular_smooth_init', smooth_init, status)
        call fsps_setup(ctx, ZMET, 'mist', 'miles', 'DL07')
    end subroutine setup_nebular_context

    !> With setup_nebular_gaussians = 1 the line kernels are built once at setup from
    !> nebular_smooth_init. A second context with a different nebular_smooth_init must neither
    !> overwrite the first context's kernels nor inherit them.
    subroutine test_setup_kernels_follow_own_smoothing()
        type(fsps_context_t) :: ctx_a, ctx_b
        real(WP), allocatable :: kernels_a(:, :)

        call print_group("Setup-time line kernels follow each context's nebular_smooth_init")

        call setup_nebular_context(ctx_a, 1, 100.0_wp)
        kernels_a = ctx_a%state%gaussnebarr

        call setup_nebular_context(ctx_b, 1, 500.0_wp)

        call assert_true(all(ctx_a%state%gaussnebarr == kernels_a), &
                         "context A kernels unchanged by context B setup", total_tests, total_failures)
        call assert_true(any(ctx_b%state%gaussnebarr /= kernels_a), &
                         "context B kernels built from its own nebular_smooth_init", total_tests, total_failures)

        call fsps_destroy(ctx_b)
        call fsps_destroy(ctx_a)
    end subroutine test_setup_kernels_follow_own_smoothing

    !> With setup_nebular_gaussians = 0 (the default) every nebular call rebuilds the line kernels
    !> from that context's sigma_smooth, so the kernel array must be owned by the context; the
    !> read-only spectral library must still be shared.
    subroutine test_per_call_kernels_are_private()
        type(fsps_context_t) :: ctx_c, ctx_d

        call print_group("Per-call line kernels are private to each context")

        call setup_nebular_context(ctx_c, 0, 100.0_wp)
        call setup_nebular_context(ctx_d, 0, 100.0_wp)

        call assert_true(.not. associated(ctx_c%state%gaussnebarr, ctx_d%state%gaussnebarr), &
                         "per-call kernel arrays are distinct", total_tests, total_failures)
        call assert_true(associated(ctx_c%state%speclib, ctx_d%state%speclib), &
                         "read-only spectral library is still shared", total_tests, total_failures)

        call fsps_destroy(ctx_d)
        call assert_true(associated(ctx_c%state%gaussnebarr), &
                         "kernels survive destruction of the other context", total_tests, total_failures)
        call fsps_destroy(ctx_c)
    end subroutine test_per_call_kernels_are_private

    !> Contexts with the same key share one cache entry whose reference count tracks the number of
    !> live contexts; destroying one must leave the shared data intact for the others.
    subroutine test_cache_refcount_bookkeeping()
        type(fsps_context_t) :: ctx_e, ctx_f, ctx_g

        call print_group("Setup cache reference counting")

        call setup_nebular_context(ctx_e, 0, 100.0_wp)
        call setup_nebular_context(ctx_f, 0, 100.0_wp)
        call setup_nebular_context(ctx_g, 0, 100.0_wp)

        call assert_true(associated(ctx_e%setup_cache, ctx_g%setup_cache), &
                         "identical selections share one cache entry", total_tests, total_failures)
        call assert_int_equals(3, ctx_e%setup_cache%refcount, "refcount = 3 live contexts", &
                               total_tests, total_failures)

        call fsps_destroy(ctx_g)
        call assert_int_equals(2, ctx_e%setup_cache%refcount, "refcount = 2 after one destroy", &
                               total_tests, total_failures)
        call fsps_destroy(ctx_f)
        call assert_int_equals(1, ctx_e%setup_cache%refcount, "refcount = 1 after two destroys", &
                               total_tests, total_failures)
        call assert_true(associated(ctx_e%state%speclib), "surviving context keeps shared data", &
                         total_tests, total_failures)

        call fsps_destroy(ctx_e)
    end subroutine test_cache_refcount_bookkeeping

    !> A second context with the same library selection must reuse the cached data without reading
    !> any file (deterministic: the process-wide count of shared-data loads does not change, and the
    !> setup takes a small fraction of a cold setup), receive identical per-context tables, and
    !> produce the same SSP and CSP output (bit-identical in serial builds; to 1e-10 relative, since
    !> OpenACC multicore reductions sum in a run-dependent order). Settings the shared data depend on must select a
    !> separate cache entry rather than silently reusing (or overwriting) another context's data.
    subroutine test_cache_hit_reuses_shared_data()
        type(fsps_context_t) :: ctx_miss, ctx_hit, ctx_other
        integer :: loads_before
        integer(8) :: t0, t1, t2, rate
        real(WP) :: t_miss, t_hit
        real(WP), allocatable :: mass_a(:), lbol_a(:), spec_a(:, :), mass_b(:), lbol_b(:), spec_b(:, :)
        type(COMPSPOUT), allocatable :: csp_a(:), csp_b(:)

        call print_group("Cache hit reuses shared data without reloading")

        loads_before = fsps_shared_load_count()
        call system_clock(t0, rate)
        call setup_csp_context(ctx_miss, cloudy_dust=0, vega_mags=0)
        call system_clock(t1)
        call setup_csp_context(ctx_hit, cloudy_dust=0, vega_mags=0)
        call system_clock(t2)
        t_miss = real(t1 - t0, WP)/real(rate, WP)
        t_hit = real(t2 - t1, WP)/real(rate, WP)
        write (*, '(A,F8.3,A,F8.3,A)') '# setup time: miss ', t_miss, ' s, hit ', t_hit, ' s'

        call assert_true(associated(ctx_miss%setup_cache, ctx_hit%setup_cache), &
                         "second context hits the first context's cache entry", total_tests, total_failures)
        call assert_int_equals(loads_before + 1, fsps_shared_load_count(), &
                               "shared data loaded from disk once for two contexts", total_tests, total_failures)
        call assert_true(t_hit < 0.25_wp*t_miss, "cache-hit setup takes < 25% of a cold setup", &
                         total_tests, total_failures)

        call assert_true(all(ctx_hit%state%speclib_logt == ctx_miss%state%speclib_logt) .and. &
                         all(ctx_hit%state%wmb_logt == ctx_miss%state%wmb_logt) .and. &
                         all(ctx_hit%state%pagb_logt == ctx_miss%state%pagb_logt) .and. &
                         all(ctx_hit%state%teff_dagb == ctx_miss%state%teff_dagb) .and. &
                         all(ctx_hit%state%agndust_tau == ctx_miss%state%agndust_tau) .and. &
                         all(ctx_hit%state%nebem_line == ctx_miss%state%nebem_line) .and. &
                         all(ctx_hit%state%nebem_age == ctx_miss%state%nebem_age), &
                         "per-context tables copied from the cache entry", total_tests, total_failures)
        call assert_true(ctx_hit%state%imf_lower_bound == ctx_miss%state%imf_lower_bound .and. &
                         all(ctx_hit%state%cosmospl == ctx_miss%state%cosmospl) .and. &
                         ctx_hit%state%whlam5000 == ctx_miss%state%whlam5000, &
                         "context-local properties identical", total_tests, total_failures)

        call run_ssp(ctx_miss, mass_a, lbol_a, spec_a)
        call run_ssp(ctx_hit, mass_b, lbol_b, spec_b)
        call assert_true(same_2d(spec_a, spec_b) .and. same(mass_a, mass_b) .and. same(lbol_a, lbol_b), &
                         "SSP output identical on a cache hit", total_tests, total_failures)

        call run_csp(ctx_miss, mass_a, lbol_a, spec_a, csp_a)
        call run_csp(ctx_hit, mass_b, lbol_b, spec_b, csp_b)
        call assert_true(same(csp_a(1)%spec, csp_b(1)%spec) .and. same(csp_a(1)%mags, csp_b(1)%mags) .and. &
                         same(csp_a(1)%emlines, csp_b(1)%emlines) .and. &
                         same([csp_a(1)%mass_csp, csp_a(1)%lbol_csp], [csp_b(1)%mass_csp, csp_b(1)%lbol_csp]), &
                         "CSP output (spectrum, mags, lines) identical on a cache hit", &
                         total_tests, total_failures)
        call assert_true(any(csp_a(1)%emlines > 0.0_wp), "CSP exercises nebular line emission", &
                         total_tests, total_failures)

        ! cloudy_dust and compute_vega_mags change the shared nebular grid and solar magnitudes
        call setup_csp_context(ctx_other, cloudy_dust=1, vega_mags=1)
        call assert_true(.not. associated(ctx_other%setup_cache, ctx_miss%setup_cache), &
                         "cloudy_dust / compute_vega_mags select a separate cache entry", &
                         total_tests, total_failures)
        call assert_int_equals(loads_before + 2, fsps_shared_load_count(), &
                               "the separate entry is loaded from disk", total_tests, total_failures)
        call assert_true(any(ctx_other%state%nebem_cont /= ctx_miss%state%nebem_cont) .and. &
                         any(ctx_other%state%magsun /= ctx_miss%state%magsun), &
                         "its nebular grid and solar magnitudes follow its own settings", &
                         total_tests, total_failures)

        call fsps_destroy(ctx_other)
        call fsps_destroy(ctx_hit)
        call fsps_destroy(ctx_miss)
    end subroutine test_cache_hit_reuses_shared_data

    !> Element-wise equality to 1e-10 relative (a cache error gives order-unity differences).
    !> Parallel (OpenACC) reductions are not bitwise reproducible from run to run: they differ by
    !> a few ulps of the kernel precision, ~1e-16 in FP64 but ~1e-7 with FP32 kernels.
    logical function same(a, b)
        real(WP), intent(in) :: a(:), b(:)

        same = size(a) == size(b)
        if (same) same = all(abs(a - b) <= SAME_RTOL*max(abs(a), abs(b)))
    end function same

    !> same() for a 2-D array, without reshaping (ifx would put the 5 MB temporary on the stack).
    logical function same_2d(a, b)
        real(WP), intent(in) :: a(:, :), b(:, :)

        same_2d = all(shape(a) == shape(b))
        if (same_2d) same_2d = all(abs(a - b) <= SAME_RTOL*max(abs(a), abs(b)))
    end function same_2d

    !> Configures a context for a nebular CSP and runs setup.
    subroutine setup_csp_context(ctx, cloudy_dust, vega_mags)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: cloudy_dust, vega_mags
        integer :: status

        call fsps_create(ctx)
        call fsps_set_param_int(ctx, 'add_neb_emission', 1, status)
        call fsps_set_param_int(ctx, 'cloudy_dust', cloudy_dust, status)
        call fsps_set_param_int(ctx, 'compute_vega_mags', vega_mags, status)
        call fsps_setup(ctx, ZMET, 'mist', 'miles', 'DL07')
        call fsps_context_move_to_device(ctx)
        call fsps_set_param_int(ctx, 'zmet', ZMET, status)
        call fsps_set_param_int(ctx, 'sfh', 1, status)
        call fsps_set_param_int(ctx, 'dust_type', 0, status)
        call fsps_set_param_float(ctx, 'tau', 1.0_wp, status)
        call fsps_set_param_float(ctx, 'tage', 3.0_wp, status)
        call fsps_set_param_float(ctx, 'dust2', 0.3_wp, status)
    end subroutine setup_csp_context

    subroutine run_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), allocatable, intent(out) :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)
        type(PARAMS) :: pset

        call fsps_context_prepare_pset(ctx)
        call fsps_context_get_pset(ctx, pset)
        allocate (mass_ssp(ctx%state%ntfull), lbol_ssp(ctx%state%ntfull), &
                  spec_ssp(ctx%state%nspec, ctx%state%ntfull))
        call fsps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
    end subroutine run_ssp

    !> CSP on the context's SSP grid (mirrors test_physics_oracles).
    subroutine run_csp(ctx, mass_ssp, lbol_ssp, spec_ssp, results)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), allocatable, intent(inout) :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)
        type(COMPSPOUT), allocatable, intent(inout) :: results(:)
        type(PARAMS) :: pset
        integer :: status

        call run_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
        call fsps_context_get_pset(ctx, pset)
        call fsps_context_update_ssp_basis(ctx, reshape(spec_ssp, [size(spec_ssp, 1), size(spec_ssp, 2), 1]), &
                                           reshape(mass_ssp, [size(mass_ssp), 1]), &
                                           reshape(lbol_ssp, [size(lbol_ssp), 1]), 1)
        if (allocated(results)) deallocate (results)
        call compute_csp_scenario(ctx, pset, 1, results, status)
        if (status /= 0) then
            write (*, *) 'ERROR: compute_csp_scenario returned status ', status
            stop 1
        end if
    end subroutine run_csp

end program test_context_isolation
