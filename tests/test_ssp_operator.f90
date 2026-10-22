program test_ssp_operator
    !> The SSP operator (opensps_ssp_operator, which generate_ssp_grid runs) against the per-star
    !> sum: S is built once per isochrone set and non-IMF parameter set, then evaluated for many
    !> IMFs (all six types, changed slopes, characteristic and break masses, mass limits, a
    !> tabulated user IMF), and its spectra, masses and L_bol are compared with the SSP summed
    !> star by star (generate_ssp_grid with opensps_ssp_set_per_star) for each IMF.
    !> Also with the isochrone modifications on (blue HB, blue stragglers, TP-AGB and post-AGB
    !> weights and shifts), for MIST (HB tagged) and Padova (HB detected, 10 blue stars per HB
    !> point). Requires the FSPS data tree, like the regression tests.

    use opensps_precision, only: WP, KP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_ssp, only: generate_ssp_grid, configure_imf_parameters, opensps_ssp_set_per_star
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_build, ssp_operator_evaluate, ssp_operator_free
    use opensps_imf, only: imf_pack_params, IMF_NPAR
    use opensps_context, only: opensps_context_get_pset, opensps_context_prepare_pset, &
                            opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use test_utils_mod, only: tap_finish, skip_without_data, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    integer :: total_tests = 0
    integer :: total_failures = 0
    ! FP64: the operator sums the same products in another order (rounding only); FP32 kernels:
    ! 100 eps(KP), for pixels within 6 decades of the age's peak (below, near FP32 underflow, the
    ! two orders round differently; as test_physics_oracles)
    real(WP), parameter :: SPEC_RTOL = merge(1.0e-12_wp, 100.0_wp * real(epsilon(1.0_KP), WP), KP == WP)
    real(WP), parameter :: MASS_RTOL = 1.0e-13_wp
    real(WP), parameter :: LBOL_ATOL = 1.0e-13_wp
    type(opensps_context_t), allocatable :: ctx
    type(ssp_operator_t) :: op

    call skip_without_data()
    call print_major_header("SSP operator vs the per-star sum")

    ! ---------------- MIST + MILES ----------------
    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 10
    call opensps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call set_int('zmet', 10)
    call set_int('imf_type', 0)
    call build("MIST, default isochrone")
    call imf_cases("MIST")

    ! Isochrone modifications (S rebuilt; IMF still free)
    call set_float('fbhb', 0.3_wp)
    call set_float('sbss', 3.0_wp)
    call set_float('agb', 0.6_wp)
    call set_float('pagb', 2.0_wp)
    call set_float('redgb', 1.2_wp)
    call set_float('delt', 0.05_wp)
    call set_float('dell', -0.1_wp)
    call build("MIST, fbhb/sbss/agb/pagb/redgb/delt/dell")
    call set_int('imf_type', 1)
    call check("MIST modified, Chabrier")
    call set_int('imf_type', 2)
    call set_float('imf1', 0.9_wp); call set_float('imf2', 1.9_wp); call set_float('imf3', 2.8_wp)
    call check("MIST modified, Kroupa 0.9/1.9/2.8")
    call ssp_operator_free(op)
    call opensps_destroy(ctx)
    deallocate(ctx)

    ! ---------------- Padova + BaSeL ----------------
    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 15
    call opensps_setup(ctx, 15, isoc_type_in='pdva', spec_type_in='basel', dust_type_in='DL07')
    call set_int('zmet', 15)
    call set_int('imf_type', 0)
    call set_float('fbhb', 0.3_wp)
    call set_float('sbss', 2.0_wp)
    call set_float('agb', 0.7_wp)
    call build("Padova, fbhb/sbss/agb")
    call check("Padova modified, Salpeter")
    call set_int('imf_type', 1)
    call check("Padova modified, Chabrier")
    call set_int('imf_type', 3)
    call set_float('vdmc', 0.3_wp)
    call check("Padova modified, van Dokkum mc = 0.3")
    call ssp_operator_free(op)
    call opensps_destroy(ctx)
    deallocate(ctx)

    call print_summary_line("Batched SSP operator", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    !> IMF cases evaluated from one operator (built with Salpeter).
    subroutine imf_cases(tag)
        character(len=*), intent(in) :: tag
        call set_int('imf_type', 0)
        call check(tag // ", Salpeter")
        call set_int('imf_type', 1)
        call check(tag // ", Chabrier")
        call set_int('imf_type', 2)
        call check(tag // ", Kroupa")
        call set_float('imf1', 0.7_wp); call set_float('imf2', 1.8_wp); call set_float('imf3', 2.9_wp)
        call check(tag // ", Kroupa 0.7/1.8/2.9")
        call set_float('imf_lower_limit', 0.15_wp); call set_float('imf_upper_limit', 60.0_wp)
        call check(tag // ", Kroupa 0.7/1.8/2.9, 0.15-60 Msun")
        call set_float('imf_lower_limit', 0.08_wp); call set_float('imf_upper_limit', 120.0_wp)
        call set_float('imf1', 1.3_wp); call set_float('imf2', 2.3_wp); call set_float('imf3', 2.3_wp)
        call set_int('imf_type', 3)
        call set_float('vdmc', 0.2_wp)
        call check(tag // ", van Dokkum mc = 0.2")
        call set_int('imf_type', 4)
        call set_float('mdave', 1.5_wp)
        call check(tag // ", Dave mdave = 1.5")
        call set_int('imf_type', 0)
        call set_float('imf_upper_limit', 80.0_wp)
        call check(tag // ", Salpeter, upper limit 80 Msun")
        call set_float('imf_upper_limit', 120.0_wp)
        call set_int('imf_type', 5)
        call check(tag // ", user IMF (data/imf.dat)")
        call set_int('imf_type', 0)
    end subroutine imf_cases

    subroutine build(label)
        character(len=*), intent(in) :: label
        type(PARAMS) :: pset
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        call ssp_operator_build(ctx, pset, op)
        call print_group("S built: " // label)
        write(*, '(a, i0, a, i0, a, i0, a, i0, a, i0)') '#   points ', op%n_pts, ', rows ', op%n_rows, &
            ', entries ', op%nnz, ', dust corners ', op%n_slots, ', direct stars ', op%n_direct
    end subroutine build

    !> The per-star SSP and the operator (built once) for the context's current IMF.
    subroutine check(label)
        character(len=*), intent(in) :: label
        type(PARAMS) :: pset
        real(WP), allocatable :: m_ref(:), l_ref(:), s_ref(:,:), m_op(:), l_op(:), s_op(:,:)
        real(WP) :: p(IMF_NPAR), dspec, dmass, dlbol
        integer :: nt, ns, t, i
        character(len=200) :: msg

        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        nt = ctx%state%ntfull
        ns = ctx%state%nspec
        allocate(m_ref(nt), l_ref(nt), s_ref(ns, nt), m_op(nt), l_op(nt), s_op(ns, nt))
        call opensps_ssp_set_per_star(.true.)
        call generate_ssp_grid(ctx, pset, m_ref, l_ref, s_ref)
        call opensps_ssp_set_per_star(.false.)
        call configure_imf_parameters(ctx, pset)
        call imf_pack_params(ctx, p)
        call ssp_operator_evaluate(ctx, pset, op, ctx%imf_type_val, p, m_op, l_op, s_op)

        dspec = 0.0_wp; dmass = 0.0_wp; dlbol = 0.0_wp
        do t = 1, nt
            do i = 1, ns
                if (KP /= WP .and. s_ref(i, t) <= 1.0e-6_wp * maxval(s_ref(:, t))) cycle
                if (s_ref(i, t) > 0.0_wp) then
                    dspec = max(dspec, abs(s_op(i, t) / s_ref(i, t) - 1.0_wp))
                else
                    dspec = max(dspec, abs(s_op(i, t)))
                end if
            end do
            if (m_ref(t) > 0.0_wp) then
                dmass = max(dmass, abs(m_op(t) / m_ref(t) - 1.0_wp))
            else
                dmass = max(dmass, abs(m_op(t)))
            end if
            dlbol = max(dlbol, abs(l_op(t) - l_ref(t)))
        end do
        call print_group(label)
        write(msg, '(a, es9.2, a, es9.2, a, es9.2)') 'spectra max rel ', dspec, ', mass max rel ', dmass, &
            ', log L_bol max abs ', dlbol
        write(*, '(a, a)') '#   ', trim(msg)
        ! The IMF must matter: surviving mass and L_bol of the oldest age
        write(*, '(a, f10.6, a, f10.6)') '#   oldest age: mass ', m_ref(nt), ', log L_bol ', l_ref(nt)
        call assert_true(dspec <= SPEC_RTOL, label // ": spectra", total_tests, total_failures)
        call assert_true(dmass <= MASS_RTOL, label // ": masses", total_tests, total_failures)
        call assert_true(dlbol <= LBOL_ATOL, label // ": L_bol", total_tests, total_failures)
    end subroutine check

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer :: status
        call opensps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: status
        call opensps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_float

end program test_ssp_operator
