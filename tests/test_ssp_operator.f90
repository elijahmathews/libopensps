program test_ssp_operator
    !> The batched SSP operator (opensps_ssp_operator) against generate_ssp_grid: S is built once
    !> per isochrone set and non-IMF parameter set, then evaluated for many IMFs (all six types,
    !> changed slopes, characteristic and break masses, mass limits, a tabulated user IMF), and
    !> its spectra, masses and L_bol are compared with generate_ssp_grid run for each IMF.
    !> Also with the isochrone modifications on (blue HB, blue stragglers, TP-AGB and post-AGB
    !> weights and shifts), for MIST (HB tagged) and Padova (HB detected, 10 blue stars per HB
    !> point). Requires the FSPS data tree, like the regression tests.

    use opensps_precision, only: WP, KP
    use opensps_constants, only: SAFE_FLOOR
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_ssp, only: generate_ssp_grid, configure_imf_parameters
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_build, ssp_operator_evaluate, ssp_operator_free, &
                                    ssp_mod_coefficients, ssp_mod_coefficient_derivatives, MOD_NGRP
    use opensps_imf, only: imf_pack_params, IMF_NPAR
    use opensps_context, only: opensps_context_get_pset, opensps_context_prepare_pset, &
                            opensps_context_set_param_int, opensps_context_set_param_float, opensps_context_move_to_device
    use opensps_context_types, only: opensps_context_t
    use test_utils_mod, only: tap_finish, skip_without_data, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    integer :: total_tests = 0
    integer :: total_failures = 0
    ! FP64: the operator aggregates the same products in another order (rounding only); FP32
    ! kernels: the template weights round to FP32 at slightly different values, 100 eps(KP)
    real(WP), parameter :: SPEC_RTOL = merge(1.0e-12_wp, 100.0_wp * real(epsilon(1.0_KP), WP), KP == WP)
    real(WP), parameter :: MASS_RTOL = 1.0e-13_wp
    real(WP), parameter :: LBOL_ATOL = 1.0e-13_wp
    type(opensps_context_t), allocatable :: ctx
    type(ssp_operator_t) :: op

    call skip_without_data()
    call print_major_header("Batched SSP operator vs generate_ssp_grid")

    ! ---------------- MIST + MILES ----------------
    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 10
    call opensps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    ! (generate_ssp_grid's kernels read the context on the device in device builds)
    call opensps_context_move_to_device(ctx)
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

    ! The modification groups' operators, combined with the coefficients of the factors, against generate_ssp_grid
    ! (delt, dell are not factors: zero here)
    call set_float('delt', 0.0_wp); call set_float('dell', 0.0_wp)
    call set_int('imf_type', 1)
    call check_groups("MIST groups: fbhb 0.3, sbss 3, agb 0.6, pagb 2, redgb 1.2")
    call set_float('fbhb', 0.0_wp)
    call check_groups("MIST groups: no HB modification")
    call set_float('fbhb', 5.0e-5_wp)
    call check_groups("MIST groups: fbhb below its threshold")
    call set_float('fbhb', 1.0e-3_wp)
    call set_float('sbss', 0.0_wp)
    call check_groups("MIST groups: fbhb 1e-3, no stragglers")
    call set_float('sbss', 5.0e-4_wp)
    call check_groups("MIST groups: sbss below its threshold")
    call set_float('sbss', 12.0_wp)
    call set_float('agb', 0.0_wp)
    call check_groups("MIST groups: agb 0")
    call set_float('agb', 1.0_wp); call set_float('pagb', 1.0_wp); call set_float('redgb', 0.5_wp)
    call check_groups("MIST groups: redgb 0.5, sbss 12")
    call set_float('fbhb', 0.3_wp); call set_float('agb', 0.6_wp); call set_float('pagb', 2.0_wp)
    call set_int('imf_type', 2)
    call check_groups("MIST groups: Kroupa, all factors")
    ! The TP-AGB shifts delt and dell: the group of the TP-AGB stars built at them
    call set_float('delt', 0.03_wp); call set_float('dell', -0.08_wp)
    call check_groups("MIST groups: all factors, delt 0.03, dell -0.08")
    call set_float('delt', -0.12_wp); call set_float('dell', 0.0_wp)
    call check_groups("MIST groups: delt -0.12")
    call set_float('delt', 0.0_wp); call set_float('dell', 0.15_wp)
    call check_groups("MIST groups: dell 0.15")
    call set_float('delt', 0.0_wp); call set_float('dell', 0.0_wp)
    call check_coefficient_derivatives()
    call ssp_operator_free(op)
    call opensps_destroy(ctx)
    deallocate(ctx)

    ! ---------------- Padova + BaSeL ----------------
    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 15
    call opensps_setup(ctx, 15, isoc_type_in='pdva', spec_type_in='basel', dust_type_in='DL07')
    ! (generate_ssp_grid's kernels read the context on the device in device builds)
    call opensps_context_move_to_device(ctx)
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

    call set_int('imf_type', 1)
    call set_float('pagb', 1.5_wp); call set_float('redgb', 0.8_wp)
    call check_groups("Padova groups: fbhb 0.3, sbss 2, agb 0.7, pagb 1.5, redgb 0.8")
    call set_float('fbhb', 5.0e-4_wp)
    call check_groups("Padova groups: fbhb below its threshold (above MIST's)")
    call set_float('fbhb', 2.0e-3_wp)
    call check_groups("Padova groups: fbhb 2e-3")
    call set_float('fbhb', 0.0_wp); call set_float('sbss', 0.0_wp)
    call check_groups("Padova groups: factors only")
    call set_float('delt', 0.05_wp); call set_float('dell', -0.1_wp)
    call check_groups("Padova groups: delt 0.05, dell -0.1")
    call set_float('delt', 0.0_wp); call set_float('dell', 0.0_wp)
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

    !> generate_ssp_grid and the operator for the context's current IMF.
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
        call generate_ssp_grid(ctx, pset, m_ref, l_ref, s_ref)
        call configure_imf_parameters(ctx, pset)
        call imf_pack_params(ctx, p)
        call ssp_operator_evaluate(ctx, pset, op, ctx%imf_type_val, p, m_op, l_op, s_op)

        dspec = 0.0_wp; dmass = 0.0_wp; dlbol = 0.0_wp
        do t = 1, nt
            do i = 1, ns
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

    !> The operators of the modification groups (ssp_operator_build, group = 0 .. MOD_NGRP - 1), evaluated for the
    !> context's IMF and combined with the coefficients of the context's factors (ssp_mod_coefficients), against
    !> generate_ssp_grid with the factors on. Without remnants (the groups other than the plain one do not carry them).
    subroutine check_groups(label)
        character(len=*), intent(in) :: label
        type(PARAMS) :: pset
        type(ssp_operator_t) :: opg
        real(WP), allocatable :: m_ref(:), l_ref(:), s_ref(:,:), m_g(:), l_g(:), s_g(:,:), m_sum(:), l_sum(:), s_sum(:,:)
        real(WP) :: p(IMF_NPAR), c(0:MOD_NGRP - 1), dspec, dmass, dlbol
        real(WP), parameter :: FLOOR_REL = 1.0e7_wp
        integer :: nt, ns, t, i, g
        character(len=200) :: msg

        call set_int('add_stellar_remnants', 0)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        nt = ctx%state%ntfull
        ns = ctx%state%nspec
        allocate(m_ref(nt), l_ref(nt), s_ref(ns, nt), m_g(nt), l_g(nt), s_g(ns, nt), m_sum(nt), l_sum(nt), s_sum(ns, nt))
        call generate_ssp_grid(ctx, pset, m_ref, l_ref, s_ref)
        call configure_imf_parameters(ctx, pset)
        call imf_pack_params(ctx, p)
        c = ssp_mod_coefficients(pset, trim(ctx%state%isoc_type) == 'pdva')
        s_sum = 0.0_wp; m_sum = 0.0_wp; l_sum = 0.0_wp
        do g = 0, MOD_NGRP - 1
            call ssp_operator_build(ctx, pset, opg, group=g, shift_t=pset%delt, shift_l=pset%dell)
            call ssp_operator_evaluate(ctx, pset, opg, ctx%imf_type_val, p, m_g, l_g, s_g)
            s_sum = s_sum + c(g) * s_g
            m_sum = m_sum + c(g) * m_g
            l_sum = l_sum + c(g) * 10.0_wp**l_g
        end do
        call ssp_operator_free(opg)
        dspec = 0.0_wp; dmass = 0.0_wp; dlbol = 0.0_wp
        do t = 1, nt
            do i = 1, ns
                ! (every group's spectrum is floored at the smallest normal number of the kernel precision where it has no
                ! flux, so the sum over the MOD_NGRP groups carries up to MOD_NGRP * c of these floors, a relative error
                ! of ~1e-3 at a pixel of 1e3 floors, which kernel rounding near the flush threshold makes appear on one
                ! compiler and not another (nvfortran FP32): the pixels below 1e7 floors, where the floors exceed
                ! the tolerance, are not compared relatively but only required to be as small)
                if (s_ref(i, t) > FLOOR_REL * max(SAFE_FLOOR, real(tiny(1.0_KP), WP))) then
                    dspec = max(dspec, abs(s_sum(i, t) / s_ref(i, t) - 1.0_wp))
                else
                    dspec = max(dspec, merge(0.0_wp, abs(s_sum(i, t)), &
                                             abs(s_sum(i, t)) < 10.0_wp * FLOOR_REL * max(SAFE_FLOOR, real(tiny(1.0_KP), WP))))
                end if
            end do
            if (m_ref(t) > 0.0_wp) then
                dmass = max(dmass, abs(m_sum(t) / m_ref(t) - 1.0_wp))
            else
                dmass = max(dmass, abs(m_sum(t)))
            end if
            if (l_ref(t) > -90.0_wp) dlbol = max(dlbol, abs(log10(max(l_sum(t), 1.0e-300_wp)) - l_ref(t)))
        end do
        call print_group(label)
        write(msg, '(a, es9.2, a, es9.2, a, es9.2)') 'spectra max rel ', dspec, ', mass max rel ', dmass, &
            ', log L_bol max abs ', dlbol
        write(*, '(a, a)') '#   ', trim(msg)
        call assert_true(dspec <= SPEC_RTOL, label // ": spectra", total_tests, total_failures)
        call assert_true(dmass <= MASS_RTOL, label // ": masses", total_tests, total_failures)
        call assert_true(dlbol <= LBOL_ATOL, label // ": L_bol", total_tests, total_failures)
        call set_int('add_stellar_remnants', 1)
    end subroutine check_groups

    !> ssp_mod_coefficient_derivatives against central differences of ssp_mod_coefficients, for MIST/BaSTI and for Padova
    !> and on each side of the thresholds of fbhb (1e-4 for MIST, 1e-3 for Padova) and sbss (1e-3). The coefficients are
    !> polynomials of degree 2 in the factors, so the differences are exact to rounding.
    subroutine check_coefficient_derivatives()
        real(WP), parameter :: FB(4) = [0.3_wp, 5.0e-5_wp, 5.0e-4_wp, 0.0_wp], SB(4) = [4.0_wp, 5.0e-4_wp, 12.0_wp, 0.0_wp]
        real(WP), parameter :: H = 1.0e-6_wp
        type(PARAMS) :: p0, pp, pm
        real(WP) :: dc(0:MOD_NGRP - 1, 5), cp(0:MOD_NGRP - 1), cm(0:MOD_NGRP - 1), worst
        integer :: ic, k, ipd
        logical :: pdva

        call print_group("Isochrone modification groups: coefficients' derivatives against central differences")
        worst = 0.0_wp
        do ipd = 1, 2
            pdva = ipd == 2
            do ic = 1, 4
                p0 = PARAMS()
                p0%fbhb = FB(ic); p0%sbss = SB(ic); p0%agb = 0.7_wp; p0%pagb = 1.3_wp; p0%redgb = 0.9_wp
                dc = ssp_mod_coefficient_derivatives(p0, pdva)
                do k = 1, 5
                    pp = p0; pm = p0
                    select case (k)
                    case (1); pp%fbhb = p0%fbhb + H; pm%fbhb = p0%fbhb - H
                    case (2); pp%sbss = p0%sbss + H; pm%sbss = p0%sbss - H
                    case (3); pp%agb = p0%agb + H; pm%agb = p0%agb - H
                    case (4); pp%pagb = p0%pagb + H; pm%pagb = p0%pagb - H
                    case (5); pp%redgb = p0%redgb + H; pm%redgb = p0%redgb - H
                    end select
                    cp = ssp_mod_coefficients(pp, pdva)
                    cm = ssp_mod_coefficients(pm, pdva)
                    worst = max(worst, maxval(abs(dc(:, k) - (cp - cm) / (2.0_wp * H))))
                end do
            end do
        end do
        write(*, '(a, es9.2)') '#   largest difference ', worst
        call assert_true(worst <= 1.0e-8_wp, "coefficient derivatives against central differences (MIST and Padova)", &
                         total_tests, total_failures)
    end subroutine check_coefficient_derivatives

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
