module test_opensps_dust_mod
    use opensps_precision, only: WP, KP
    use opensps_constants, only: NEMLINE, C_LIGHT, GRAVITY_L_M_T_COEFF, NTEFF_DAGB, NTAU_DAGB, &
                              NAGNDUST, M_SOL, G_NEWTON, R_SOL, YEAR_TO_SECOND, SAFE_FLOOR, PI
    use opensps_types, only: params
    use opensps_context_types, only: opensps_context_t
    use opensps_context, only: opensps_context_move_to_device, opensps_context_remove_from_device, &
                            opensps_context_prepare_csp_workspace
    use opensps_dust
    use opensps_integration, only: integrate_trapezoid_array
    use opensps_interpolation, only: find_interval
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_int_equals, assert_relative_error
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_is_finite
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_opensps_dust_tests, total_failures, total_tests

    ! Local constants for expected-value calculations
    real(WP), parameter :: V_BAND_ANGSTROMS = 5500.0_wp
    real(WP), parameter :: UV_BUMP_CENTER = 2175.0_wp
    real(WP), parameter :: KC13_BUMP_WIDTH = 350.0_wp
    real(WP), parameter :: KC13_R_V_BASE = 4.05_wp

    real(WP), parameter :: CALZ_LAM_BREAK  = 6300.0_wp
    real(WP), parameter :: CALZ_R_V   = 4.05_wp
    real(WP), parameter :: CALZ_SCALE = 2.659_wp
    real(WP), parameter :: CALZ_UV_COEFFS(0:3) = [-2.156_wp, 1.509_wp, -0.198_wp, 0.011_wp]
    real(WP), parameter :: CALZ_OPT_COEFFS(0:1) = [-1.857_wp, 1.040_wp]

    real(WP), parameter :: EPS = 1.0e-6_wp
    real(WP), parameter :: SMALL_DELTA = 1.0e-4_wp
    real(WP), parameter :: LESS_SMALL_DELTA = 1.0e-2_wp
    ! Relative rounding floor of the kernel precision KP (2.2e-14 for FP64, 1.2e-5 for FP32); tolerances on
    ! KP-computed quantities use max(tol, KP_RTOL) so the FP64 build keeps its original tolerance
    real(WP), parameter :: KP_RTOL = 100.0_wp * real(epsilon(1.0_KP), WP)

    ! AGB / VW93 constants (copied from opensps_dust for test expectations)
    real(WP), parameter :: AGB_DELTA_C_RICH = 0.0025_wp
    real(WP), parameter :: AGB_DELTA_O_RICH = 0.01_wp
    real(WP), parameter :: AGB_KAPPA_C_RICH = 3200.0_wp
    real(WP), parameter :: AGB_KAPPA_O_RICH = 3000.0_wp
    real(WP), parameter :: AGB_RIN_FACTOR_C = 1.92E12_wp
    real(WP), parameter :: AGB_RIN_FACTOR_O = 4.74E12_wp
    real(WP), parameter :: AGB_PER_INTERCEPT = -2.07_wp
    real(WP), parameter :: AGB_PER_SLOPE_R   = 1.94_wp
    real(WP), parameter :: AGB_PER_SLOPE_M   = -0.9_wp
    real(WP), parameter :: AGB_VEXP_INTERCEPT = -13.5_wp
    real(WP), parameter :: AGB_VEXP_SLOPE     = 0.056_wp
    real(WP), parameter :: AGB_VEXP_MIN       = 3.0_wp
    real(WP), parameter :: AGB_VEXP_MAX       = 15.0_wp
    real(WP), parameter :: VW93_MDOT_LIMIT_ISO = 1.0e-4_wp
    real(WP), parameter :: VW93_PER_THRESH     = 500.0_wp
    real(WP), parameter :: VW93_MASS_THRESH    = 2.5_wp
    real(WP), parameter :: VW93_BASE_INTERCEPT = -11.4_wp
    real(WP), parameter :: VW93_BASE_SLOPE     = 0.0123_wp
    real(WP), parameter :: VW93_HIGH_SLOPE     = 0.0125_wp
    real(WP), parameter :: VW93_SUPERWIND_NORM = 1.93e3_wp
    real(WP), parameter :: AGB_DTG_VEL_NORM    = 225.0_wp
    real(WP), parameter :: AGB_DTG_LUM_NORM    = 1.0e4_wp
    real(WP), parameter :: AGB_DTG_LUM_EXP     = -0.6_wp

contains

    !> @brief Wrapper for compute_attenuation_curve (replaced by point version in core)
    function compute_attenuation_curve(wavelengths, dust_type_id, settings, ctx) result(attenuation_curve)
        real(WP), dimension(:), intent(in) :: wavelengths
        integer, intent(in)                :: dust_type_id
        type(params), intent(in)           :: settings
        type(opensps_context_t), intent(in)   :: ctx
        real(WP), dimension(size(wavelengths)) :: attenuation_curve
        integer :: i
        
        if (dust_type_id == 3) then
             attenuation_curve = ctx%state%wgdust(:, settings%wgp1, settings%wgp2, settings%wgp3)
             return
        end if
        if (dust_type_id == 5) then
             attenuation_curve = ctx%state%g03smcextn
             return
        end if
        
        do i = 1, size(wavelengths)
            attenuation_curve(i) = compute_attenuation_curve_point(wavelengths(i), i, dust_type_id, settings, ctx)
        end do
    end function compute_attenuation_curve

    !> @brief
    !> Unit test suite for opensps_dust module.
    subroutine run_opensps_dust_tests()

        call print_minor_header("opensps_dust")

        call test_power_law()
        call test_ccm89()
        call test_calzetti()
        call test_wg_smc()
        call test_kriek_conroy()
        call test_reddy()
        call test_circumstellar_optical_depth()
        call test_agb_dust_screen()
        call test_compute_attenuation_type5_point()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_opensps_dust_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: POWER LAW (Type 0)
    ! ------------------------------------------------------------------------
    subroutine test_power_law()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(2) :: w2
        real(WP), dimension(3) :: w3
        real(WP), dimension(1) :: w1
        real(WP), dimension(2) :: res2
        real(WP), dimension(3) :: res3
        real(WP), dimension(1) :: res1

        call print_group("Power Law (Type 0)")

        allocate(ctx)
        call opensps_context_move_to_device(ctx)

        ! Test 1.1: V-band normalization
        settings%dust_index = -0.7_wp
        w1 = [5500.0_wp]
        res1 = compute_attenuation_curve(w1, 0, settings, ctx)
        call assert_float_equals(1.0_wp, res1(1), 0.0_wp, "V-band normalization (5500A)", total_tests, total_failures)

        ! Test 1.2: Grey screen (index=0)
        settings%dust_index = 0.0_wp
        w3 = [1000.0_wp, 2000.0_wp, 5000.0_wp]
        res3 = compute_attenuation_curve(w3, 0, settings, ctx)
        call assert_true(all(abs(res3 - 1.0_wp) <= EPS), "Grey screen (index=0)", total_tests, total_failures)

        ! Test 1.3: Blue-tilted slope
        settings%dust_index = -1.0_wp
        w2 = [2750.0_wp, 5500.0_wp]
        res2 = compute_attenuation_curve(w2, 0, settings, ctx)
        call assert_float_equals(2.0_wp, res2(1), EPS, "Blue-tilted slope (2750A)", total_tests, total_failures)

        call opensps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_power_law

    ! ------------------------------------------------------------------------
    ! TEST SUITE: CCM89 Milky Way (Type 1)
    ! ------------------------------------------------------------------------
    subroutine test_ccm89()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(5) :: w_regions
        real(WP), dimension(5) :: res_regions
        real(WP), dimension(2) :: w_break
        real(WP), dimension(2) :: res_break
        real(WP), dimension(2) :: w_fuv
        real(WP), dimension(2) :: res_fuv
        real(WP), dimension(1) :: w_bump
        real(WP), dimension(1) :: res_uvb0, res_uvb1
        real(WP), dimension(3) :: w_ir, res_ir

        call print_group("CCM89 Milky Way (Type 1)")

        allocate(ctx)
        call opensps_context_move_to_device(ctx)

        settings%mwr = 3.1_wp
        settings%uvb = 1.0_wp

        ! Test 2.1: Region coverage (IR, Optical, NUV, MUV, FUV)
        w_regions = [1.0e4_wp/0.7_wp, 1.0e4_wp/2.0_wp, 1.0e4_wp/4.0_wp, &
                     1.0e4_wp/6.5_wp, 1.0e4_wp/9.0_wp]
        res_regions = compute_attenuation_curve(w_regions, 1, settings, ctx)
        call assert_true(all(ieee_is_finite(res_regions)) .and. all(res_regions > 0.0_wp), &
                         "Region coverage (finite and > 0)", total_tests, total_failures)

        ! Test 2.2: Smoothing hack continuity at 3030A
        w_break = [3030.30_wp, 3030.31_wp]
        res_break = compute_attenuation_curve(w_break, 1, settings, ctx)
        call assert_true(abs(res_break(1) - res_break(2)) <= SMALL_DELTA, &
                         "Smoothing continuity (3030A)", total_tests, total_failures)

        ! Test 2.3: Far-UV clamping (lambda < 833A)
        w_fuv = [800.0_wp, 700.0_wp]
        res_fuv = compute_attenuation_curve(w_fuv, 1, settings, ctx)
        call assert_float_equals(res_fuv(1), res_fuv(2), 0.0_wp, "FUV clamp equality", total_tests, total_failures)

        ! Test 2.4: UV bump scaling
        w_bump = [2175.0_wp]
        settings%uvb = 0.0_wp
        res_uvb0 = compute_attenuation_curve(w_bump, 1, settings, ctx)
        settings%uvb = 1.0_wp
        res_uvb1 = compute_attenuation_curve(w_bump, 1, settings, ctx)
        call assert_true(res_uvb1(1) > res_uvb0(1), "UV bump scaling (uvb 0 -> 1)", total_tests, total_failures)

        ! IR power law extended from 3.3 um to 10 um (x = 0.1), as in legacy FSPS; zero beyond
        w_ir = [3.8e4_wp, 8.0e4_wp, 1.2e5_wp]
        res_ir = compute_attenuation_curve(w_ir, 1, settings, ctx)
        call assert_true(res_ir(1) > res_ir(2) .and. res_ir(2) > 0.0_wp, &
                         "IR power law extended to 10 um (L' and 8 um attenuated)", total_tests, total_failures)
        call assert_true(res_ir(3) == 0.0_wp, "no attenuation beyond 10 um", total_tests, total_failures)

        call opensps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_ccm89
    ! ------------------------------------------------------------------------
    ! ------------------------------------------------------------------------
    ! TEST SUITE: Calzetti et al. (Type 2)
    ! ------------------------------------------------------------------------
    subroutine test_calzetti()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(2) :: w_break
        real(WP), dimension(2) :: res_break
        real(WP), dimension(1) :: w_v
        real(WP), dimension(1) :: res_v
        real(WP), dimension(6) :: w_ext, res_ext

        call print_group("Calzetti (Type 2)")

        allocate(ctx)
        call opensps_context_move_to_device(ctx)

        ! Test 3.1: 0.63um junction continuity
        w_break = [6300.000_wp, 6300.001_wp]
        res_break = compute_attenuation_curve(w_break, 2, settings, ctx)
        call assert_true(abs(res_break(1) - res_break(2)) <= LESS_SMALL_DELTA, &
                         "0.63um junction continuity", total_tests, total_failures)

        ! Test 3.2: Precision fix at 5500A
        w_v = [5500.0_wp]
        res_v = compute_attenuation_curve(w_v, 2, settings, ctx)
        call assert_float_equals(1.0_wp, res_v(1), 1.0e-3_wp, &
                                 "Calzetti V-band normalization", total_tests, total_failures)

        ! Test 3.3: eq. 4 is extended beyond 0.12-2.2 um (as in legacy FSPS), not cut to zero
        w_ext = [21999.0_wp, 22001.0_wp, 1000.0_wp, 25000.0_wp, 35000.0_wp, 1.0e5_wp]
        res_ext = compute_attenuation_curve(w_ext, 2, settings, ctx)
        call assert_true(abs(res_ext(1) - res_ext(2)) <= 1.0e-4_wp, "2.2um continuity (curve extended)", &
                         total_tests, total_failures)
        call assert_true(res_ext(3) > 2.0_wp, "far-UV (1000 A) attenuated by the extended polynomial", &
                         total_tests, total_failures)
        call assert_true(res_ext(4) > 0.0_wp, "2.5um attenuated by the extended linear law", &
                         total_tests, total_failures)
        call assert_true(res_ext(5) == 0.0_wp .and. res_ext(6) == 0.0_wp, &
                         "no attenuation beyond the zero crossing (~3.1um)", total_tests, total_failures)

        call opensps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_calzetti

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Witt & Gordon (Type 3) and SMC (Type 5)
    ! ------------------------------------------------------------------------
    subroutine test_wg_smc()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: settings
        integer, parameter :: nlam = 5
        real(WP), dimension(nlam) :: wavelengths
        real(WP), dimension(nlam) :: res

        call print_group("Witt & Gordon + SMC Tables (Types 3 & 5)")

        wavelengths = [1000.0_wp, 2000.0_wp, 3000.0_wp, 4000.0_wp, 5000.0_wp]
        call setup_mock_context(ctx, nlam)

        ! Test 4.1: W&G table mapping (index order)
        settings%wgp1 = 2
        settings%wgp2 = 3
        settings%wgp3 = 1
        res = compute_attenuation_curve(wavelengths, 3, settings, ctx)
        call assert_true(all(res == ctx%state%wgdust(:, 2, 3, 1)), "W&G table mapping", total_tests, total_failures)

        ! Test 4.2: SMC passthrough
        res = compute_attenuation_curve(wavelengths, 5, settings, ctx)
        call assert_true(all(res == ctx%state%g03smcextn), "SMC passthrough", total_tests, total_failures)

        call teardown_mock_context(ctx)

    end subroutine test_wg_smc

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Kriek & Conroy (Type 4)
    ! ------------------------------------------------------------------------
    subroutine test_kriek_conroy()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(3) :: wavelengths
        real(WP), dimension(3) :: res, expected
        real(WP), dimension(1) :: w_bump
        real(WP), dimension(1) :: res_bump, expected_no_bump
        real(WP), dimension(3) :: res_uvb0, res_uvb5

        call print_group("Kriek & Conroy (Type 4)")

        allocate(ctx)
        call opensps_context_move_to_device(ctx)

        wavelengths = [1500.0_wp, 2175.0_wp, 5500.0_wp]

        ! Test 5.1: No-tilt baseline (delta=0)
        settings%dust_index = 0.0_wp
        res = compute_attenuation_curve(wavelengths, 4, settings, ctx)
        expected = (calc_calzetti_curve(wavelengths) * KC13_R_V_BASE + &
                    calc_drude_profile(wavelengths, 0.85_wp)) / KC13_R_V_BASE
        call assert_true(all(abs(res - expected) <= EPS), "No-tilt baseline (delta=0)", total_tests, total_failures)

        ! Test 5.2: No-bump index (delta=0.447)
        settings%dust_index = 0.447_wp
        w_bump = [2175.0_wp]
        res_bump = compute_attenuation_curve(w_bump, 4, settings, ctx)
        expected_no_bump = calc_calzetti_curve(w_bump) * (w_bump / V_BAND_ANGSTROMS)**settings%dust_index
        call assert_float_equals(expected_no_bump(1), res_bump(1), 5.0e-4_wp, &
                                 "No-bump index (delta=0.447)", total_tests, total_failures)

        ! Test 5.3: UVB independence (ghost variable)
        settings%dust_index = -0.2_wp
        settings%uvb = 0.0_wp
        res_uvb0 = compute_attenuation_curve(wavelengths, 4, settings, ctx)
        settings%uvb = 5.0_wp
        res_uvb5 = compute_attenuation_curve(wavelengths, 4, settings, ctx)
        call assert_true(all(res_uvb0 == res_uvb5), "UVB independence", total_tests, total_failures)

        call opensps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_kriek_conroy

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Reddy et al. (Type 6)
    ! ------------------------------------------------------------------------
    subroutine test_reddy()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(2) :: w_uv
        real(WP), dimension(2) :: res_uv
        real(WP), dimension(2) :: w_break
        real(WP), dimension(2) :: res_break

        call print_group("Reddy et al. (Type 6)")

        allocate(ctx)
        call opensps_context_move_to_device(ctx)

        ! Test 6.1: UV extrapolation (lambda < 1500A constant)
        w_uv = [1400.0_wp, 1500.0_wp]
        res_uv = compute_attenuation_curve(w_uv, 6, settings, ctx)
        call assert_float_equals(res_uv(1), res_uv(2), 0.0_wp, "UV extrapolation clamp", total_tests, total_failures)

        ! Test 6.2: 6000A continuity
        w_break = [5999.999_wp, 6000.000_wp]
        res_break = compute_attenuation_curve(w_break, 6, settings, ctx)
        call assert_true(abs(res_break(1) - res_break(2)) <= SMALL_DELTA, "6000A continuity", total_tests, total_failures)

        call opensps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_reddy


    ! ------------------------------------------------------------------------
    ! TEST SUITE: Circumstellar Optical Depth
    ! ------------------------------------------------------------------------
    subroutine test_circumstellar_optical_depth()
        type(opensps_context_t), allocatable :: ctx
        real(WP) :: tau_c, tau_o, tau_expected_c, tau_expected_o
        real(WP) :: m_act, log_l, log_g, log_mdot_iso
        real(WP) :: period_days, vexp_unclamped
        real(WP) :: tau_clamped, tau_unclamped

        call print_group("Circumstellar Optical Depth")

        allocate(ctx)
        call opensps_context_move_to_device(ctx)
        ctx%use_isoc_mdot_val = 1

        m_act = 1.0_wp
        log_l = 3.0_wp
        log_g = 1.0_wp
        log_mdot_iso = -5.0_wp

        ! Test 2.1: Chemistry switching
        tau_c = compute_circumstellar_optical_depth(ctx, 1, m_act, log_l, log_g, log_mdot_iso)
        tau_o = compute_circumstellar_optical_depth(ctx, 0, m_act, log_l, log_g, log_mdot_iso)
        tau_expected_c = calc_tau_expected(1, m_act, log_l, log_g, 10.0_wp**log_mdot_iso, .true.)
        tau_expected_o = calc_tau_expected(0, m_act, log_l, log_g, 10.0_wp**log_mdot_iso, .true.)
        call assert_float_equals(tau_expected_c, tau_c, 1.0e-8_wp, "C-rich kappa path", total_tests, total_failures)
        call assert_float_equals(tau_expected_o, tau_o, 1.0e-8_wp, "O-rich kappa path", total_tests, total_failures)

        ! Test 2.2: Expansion velocity clamping (low)
        log_g = 6.0_wp
        period_days = calc_period_days(m_act, log_g)
        vexp_unclamped = AGB_VEXP_INTERCEPT + AGB_VEXP_SLOPE * period_days
        tau_clamped = calc_tau_expected(0, m_act, log_l, log_g, 10.0_wp**log_mdot_iso, .true.)
        tau_unclamped = calc_tau_expected(0, m_act, log_l, log_g, 10.0_wp**log_mdot_iso, .false.)
        tau_o = compute_circumstellar_optical_depth(ctx, 0, m_act, log_l, log_g, log_mdot_iso)
        
        call assert_true(vexp_unclamped < AGB_VEXP_MIN, "Velocity unclamped < 3 km/s", total_tests, total_failures)
        call assert_float_equals(tau_clamped, tau_o, 1.0e-8_wp, "Velocity clamped low", total_tests, total_failures)
        call assert_true(abs(tau_unclamped - tau_o) > 0.0_wp, "Velocity clamp changes tau", total_tests, total_failures)

        ! Test 2.2b: Expansion velocity clamping (high) & Superwind
        ! FIX: Use log_g = -1.0 to generate a physical star large enough (P ~ 1500 days)
        !      to trigger the high velocity clamp (>15 km/s) AND the superwind branch (>500 days).
        log_g = -1.0_wp 
        
        period_days = calc_period_days(m_act, log_g)
        vexp_unclamped = AGB_VEXP_INTERCEPT + AGB_VEXP_SLOPE * period_days
        
        tau_clamped = calc_tau_expected(0, m_act, log_l, log_g, 10.0_wp**log_mdot_iso, .true.)
        tau_unclamped = calc_tau_expected(0, m_act, log_l, log_g, 10.0_wp**log_mdot_iso, .false.)
        tau_o = compute_circumstellar_optical_depth(ctx, 0, m_act, log_l, log_g, log_mdot_iso)
        
        call assert_true(vexp_unclamped > AGB_VEXP_MAX, "Velocity unclamped > 15 km/s", total_tests, total_failures)
        call assert_float_equals(tau_clamped, tau_o, 1.0e-8_wp, "Velocity clamped high", total_tests, total_failures)
        call assert_true(abs(tau_unclamped - tau_o) > 0.0_wp, "Velocity clamp changes tau (high)", total_tests, total_failures)

        ! Test 2.3: Mass loss logic (isochrone override + clamp)
        ! Reset log_g to standard value for isolation
        log_g = 1.0_wp
        log_mdot_iso = -5.0_wp
        tau_o = compute_circumstellar_optical_depth(ctx, 0, m_act, log_l, log_g, log_mdot_iso)
        tau_expected_o = calc_tau_expected(0, m_act, log_l, log_g, 1.0e-5_wp, .true.)
        call assert_float_equals(tau_expected_o, tau_o, 1.0e-8_wp, "Isochrone mdot (1e-5)", total_tests, total_failures)

        log_mdot_iso = 0.0_wp
        tau_o = compute_circumstellar_optical_depth(ctx, 0, m_act, log_l, log_g, log_mdot_iso)
        tau_expected_o = calc_tau_expected(0, m_act, log_l, log_g, VW93_MDOT_LIMIT_ISO, .true.)
        call assert_float_equals(tau_expected_o, tau_o, 1.0e-8_wp, "Isochrone mdot clamp (1e-4)", total_tests, total_failures)

        ! Test 2.4: Superwind transition
        ctx%use_isoc_mdot_val = 0
        ! FIX: Use log_g = -1.0 again to strictly test the superwind branch logic
        log_g = -1.0_wp
        period_days = calc_period_days(m_act, log_g)
        call assert_true(period_days > VW93_PER_THRESH, "Superwind period > 500", total_tests, total_failures)
        
        tau_o = compute_circumstellar_optical_depth(ctx, 0, m_act, log_l, log_g, log_mdot_iso)
        tau_expected_o = calc_tau_expected_superwind(0, m_act, log_l, log_g)
        call assert_float_equals(tau_expected_o, tau_o, 1.0e-8_wp, "Superwind branch", total_tests, total_failures)

        call opensps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_circumstellar_optical_depth

    ! ------------------------------------------------------------------------
    ! TEST SUITE: AGB Dust Screen
    ! ------------------------------------------------------------------------
    subroutine test_agb_dust_screen()
        type(opensps_context_t), allocatable :: ctx
        integer, parameter :: nlam = 5
        real(KP), dimension(nlam) :: spectrum
        real(WP), dimension(nlam) :: expected
        real(WP) :: mass_act, log_t, log_l, log_g, log_mdot, c_o_ratio
        real(WP) :: log_g_local, tau_1um

        call print_group("AGB Dust Screen")

        call setup_physics_context(ctx, nlam)

        ! Test 3.1: Identity weight
        spectrum = 100.0_wp
        call apply_agb_dust_screen(ctx, 0.0_wp, spectrum, 1.0_wp, 3.6_wp, 3.0_wp, 0.0_wp, 1.5_wp, -5.0_wp)
        call assert_true(all(abs(spectrum - 100.0_wp) <= EPS), "AGB weight=0 no change", total_tests, total_failures)

        ! Test 3.2: BaSTI log_g calculation
        ctx%state%isoc_type = 'bsti'
        ctx%use_isoc_mdot_val = 1
        spectrum = 100.0_wp
        mass_act = 1.0_wp
        log_t = 3.6_wp
        log_l = 3.0_wp
        log_g = 0.0_wp
        log_mdot = -5.0_wp
        c_o_ratio = 1.5_wp

        call set_flux_dagb_pattern(ctx, nlam)

        log_g_local = log10(GRAVITY_L_M_T_COEFF * mass_act / (10.0_wp**log_l)) + 4.0_wp * log_t
        tau_1um = compute_circumstellar_optical_depth(ctx, 1, mass_act, log_l, log_g_local, log_mdot)
        expected = spectrum * calc_dusty_transfer(ctx, 1, log_t, tau_1um)

        call apply_agb_dust_screen(ctx, 1.0_wp, spectrum, mass_act, log_t, log_l, log_g, c_o_ratio, log_mdot)
        call assert_true(all(abs(spectrum - expected) <= 5.0e-6_wp), "BaSTI log_g fallback", total_tests, total_failures)

        ! Test 3.3: Extrapolation edge case
        spectrum = 100.0_wp
        call apply_agb_dust_screen(ctx, 1.0_wp, spectrum, 1.0_wp, 3.0_wp, 3.0_wp, 0.0_wp, 1.5_wp, -5.0_wp)
        call assert_true(all(ieee_is_finite(spectrum)) .and. all(spectrum > 0.0_wp), &
                         "AGB extrapolation safe", total_tests, total_failures)

        ! Test 3.4: Transfer function application (0.5)
        call set_flux_dagb_constant(ctx, nlam, 0.5_wp)
        spectrum = 100.0_wp
        call apply_agb_dust_screen(ctx, 1.0_wp, spectrum, 1.0_wp, 3.6_wp, 3.0_wp, 0.0_wp, 1.5_wp, -5.0_wp)
        call assert_true(all(abs(spectrum - 50.0_wp) <= LESS_SMALL_DELTA), "AGB transfer function", total_tests, total_failures)

        call teardown_physics_context(ctx)

    end subroutine test_agb_dust_screen










    subroutine test_compute_attenuation_type5_point()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP) :: res

        call print_group("Type 5 Attenuation Point (SMC direct)")
        call setup_mock_context(ctx, 5)

        ! Bypass the vectorized testing wrapper and hit the point subroutine natively
        ! For idx=3, our mock ctx%state%g03smcextn(3) is 3.0_wp
        res = compute_attenuation_curve_point(1000.0_wp, 3, 5, settings, ctx)

        call assert_float_equals(3.0_wp, res, EPS, "Type 5 directly via point function", total_tests, total_failures)

        call teardown_mock_context(ctx)
    end subroutine test_compute_attenuation_type5_point

    ! ------------------------------------------------------------------------
    ! Helper: mock context setup
    ! ------------------------------------------------------------------------
    subroutine setup_mock_context(ctx, nlam)
        type(opensps_context_t), allocatable, intent(out) :: ctx
        integer, intent(in) :: nlam
        integer :: i, j, k, l

        allocate(ctx)
        ctx%state%nspec = nlam
        ctx%state%ntfull = 1
        ctx%state%nz = 1
        allocate(ctx%state%wgdust(nlam, 3, 4, 2))
        allocate(ctx%state%g03smcextn(nlam))

        do l = 1, 2
            do k = 1, 4
                do j = 1, 3
                    do i = 1, nlam
                        ctx%state%wgdust(i, j, k, l) = real(i + 100*j + 10000*k + 1000000*l, WP)
                    end do
                end do
            end do
        end do

        do i = 1, nlam
            ctx%state%g03smcextn(i) = real(i, WP)
        end do

        call opensps_context_prepare_csp_workspace(ctx)
        call opensps_context_move_to_device(ctx)
    end subroutine setup_mock_context

    subroutine teardown_mock_context(ctx)
        type(opensps_context_t), allocatable, intent(inout) :: ctx
        
        call opensps_context_remove_from_device(ctx)
        
        if (associated(ctx%state%wgdust)) deallocate(ctx%state%wgdust)
        if (associated(ctx%state%g03smcextn)) deallocate(ctx%state%g03smcextn)
        deallocate(ctx)

    end subroutine teardown_mock_context

    ! ------------------------------------------------------------------------
    ! Helper: mock context setup for full dust physics
    ! ------------------------------------------------------------------------
    subroutine setup_physics_context(ctx, nlam)
        type(opensps_context_t), allocatable, intent(out) :: ctx
        integer, intent(in) :: nlam
        integer :: i, j, k
        real(WP) :: teff_min, teff_max, tau_min, tau_max, dteff, dtau

        allocate(ctx)
        ctx%state%nspec = nlam
        ctx%state%ntfull = 1
        ctx%state%nz = 1
        allocate(ctx%state%spec_lambda(nlam))
        allocate(ctx%state%wgdust(nlam, 3, 4, 2))
        allocate(ctx%state%g03smcextn(nlam))
        allocate(ctx%state%agndust_spec(nlam, NAGNDUST))
        allocate(ctx%state%qpaharr(3))
        allocate(ctx%state%uminarr(3))
        allocate(ctx%state%dustem2_dustem(nlam, 3, 6))
        allocate(ctx%state%flux_dagb(nlam, 2, NTEFF_DAGB, NTAU_DAGB))

        ctx%state%nqpah_dustem = 3
        ctx%state%numin_dustem = 3

        do i = 1, nlam
            ctx%state%spec_lambda(i) = 1000.0_wp * real(i, WP)
        end do

        ctx%state%nebem_line_pos = 5500.0_wp

        ctx%state%agndust_tau = [10.0_wp, 20.0_wp, 30.0_wp, 40.0_wp, 50.0_wp, 60.0_wp, 70.0_wp, 80.0_wp, 90.0_wp]
        ctx%state%agndust_spec(:, :) = 0.0_wp
        ctx%state%agndust_spec(:, 1) = 1.0_wp
        ctx%state%agndust_spec(:, 2) = 2.0_wp
        ctx%state%agndust_spec(:, 3) = 3.0_wp

        ctx%state%qpaharr = [1.0_wp, 2.0_wp, 3.0_wp]
        ctx%state%uminarr = [0.5_wp, 1.0_wp, 1.5_wp]

        ctx%state%dustem2_dustem(:, :, :) = 0.0_wp
        do j = 1, 3
            do k = 1, 3
                ctx%state%dustem2_dustem(:, j, 2*k - 1) = 1.0_wp
                ctx%state%dustem2_dustem(:, j, 2*k) = 10.0_wp
            end do
        end do

        ctx%state%wgdust(:, :, :, :) = 1.0_wp
        ctx%state%g03smcextn(:) = 1.0_wp

        teff_min = 3.5_wp
        teff_max = 4.0_wp
        dteff = (teff_max - teff_min) / real(NTEFF_DAGB - 1, WP)
        do i = 1, NTEFF_DAGB
            ctx%state%teff_dagb(1, i) = 10.0_wp**(teff_min + dteff * real(i - 1, WP))
            ctx%state%teff_dagb(2, i) = ctx%state%teff_dagb(1, i)
        end do

        tau_min = -2.0_wp
        tau_max = 1.0_wp
        dtau = (tau_max - tau_min) / real(NTAU_DAGB - 1, WP)
        do i = 1, NTAU_DAGB
            ctx%state%tau1_dagb(1, i) = tau_min + dtau * real(i - 1, WP)
            ctx%state%tau1_dagb(2, i) = ctx%state%tau1_dagb(1, i)
        end do

        ctx%state%flux_dagb(:, :, :, :) = 1.0_wp

        call opensps_context_prepare_csp_workspace(ctx)
        call opensps_context_move_to_device(ctx)
    end subroutine setup_physics_context

    subroutine teardown_physics_context(ctx)
        type(opensps_context_t), allocatable, intent(inout) :: ctx

        call opensps_context_remove_from_device(ctx)

        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%wgdust)) deallocate(ctx%state%wgdust)
        if (associated(ctx%state%g03smcextn)) deallocate(ctx%state%g03smcextn)
        if (associated(ctx%state%agndust_spec)) deallocate(ctx%state%agndust_spec)
        if (associated(ctx%state%qpaharr)) deallocate(ctx%state%qpaharr)
        if (associated(ctx%state%uminarr)) deallocate(ctx%state%uminarr)
        if (associated(ctx%state%dustem2_dustem)) deallocate(ctx%state%dustem2_dustem)
        if (associated(ctx%state%flux_dagb)) deallocate(ctx%state%flux_dagb)
        deallocate(ctx)

    end subroutine teardown_physics_context

    subroutine set_flux_dagb_constant(ctx, nlam, value)
        type(opensps_context_t), intent(inout) :: ctx
        integer, intent(in) :: nlam
        real(WP), intent(in) :: value

        ctx%state%flux_dagb(1:nlam, :, :, :) = value

    end subroutine set_flux_dagb_constant

    subroutine set_flux_dagb_pattern(ctx, nlam)
        type(opensps_context_t), intent(inout) :: ctx
        integer, intent(in) :: nlam
        integer :: i, j, k, l

        do l = 1, NTAU_DAGB
            do k = 1, NTEFF_DAGB
                do j = 1, 2
                    do i = 1, nlam
                        ctx%state%flux_dagb(i, j, k, l) = 0.1_wp * real(k, WP) + 0.01_wp * real(l, WP)
                    end do
                end do
            end do
        end do

    end subroutine set_flux_dagb_pattern

    pure function calc_period_days(m_act, log_g) result(period_days)
        real(WP), intent(in) :: m_act, log_g
        real(WP) :: period_days
        real(WP) :: radius_solar

        radius_solar = sqrt(m_act * M_SOL * G_NEWTON / (10.0_wp**log_g)) / R_SOL
        period_days = 10.0_wp**(AGB_PER_INTERCEPT + AGB_PER_SLOPE_R * log10(radius_solar) + &
                               AGB_PER_SLOPE_M * log10(m_act))
    end function calc_period_days

    pure function calc_tau_expected(c_rich_flag, m_act, log_l, log_g, mdot_sol_yr, clamp_velocity) result(tau)
        integer, intent(in) :: c_rich_flag
        real(WP), intent(in) :: m_act, log_l, log_g, mdot_sol_yr
        logical, intent(in) :: clamp_velocity
        real(WP) :: tau
        real(WP) :: period_days, velocity_exp
        real(WP) :: inner_radius_cm, dust_gas_ratio, kappa_eff

        if (c_rich_flag == 1) then
            kappa_eff = AGB_KAPPA_C_RICH
        else
            kappa_eff = AGB_KAPPA_O_RICH
        end if

        period_days = calc_period_days(m_act, log_g)
        velocity_exp = AGB_VEXP_INTERCEPT + AGB_VEXP_SLOPE * period_days
        if (clamp_velocity) then
            velocity_exp = max(min(velocity_exp, AGB_VEXP_MAX), AGB_VEXP_MIN)
        end if

        if (c_rich_flag == 1) then
            inner_radius_cm = AGB_RIN_FACTOR_C * (10.0_wp**log_l)**0.5_wp
            dust_gas_ratio = AGB_DELTA_C_RICH
        else
            inner_radius_cm = AGB_RIN_FACTOR_O * (10.0_wp**log_l)**0.5_wp
            dust_gas_ratio = AGB_DELTA_O_RICH
        end if

        dust_gas_ratio = dust_gas_ratio * (velocity_exp**2 / AGB_DTG_VEL_NORM) * &
                         ((10.0_wp**log_l / AGB_DTG_LUM_NORM)**AGB_DTG_LUM_EXP)

        tau = kappa_eff * dust_gas_ratio * (mdot_sol_yr * M_SOL / YEAR_TO_SECOND) / &
              inner_radius_cm / (4.0_wp * PI) / (velocity_exp * 1.0e5_wp)

    end function calc_tau_expected

    pure function calc_tau_expected_superwind(c_rich_flag, m_act, log_l, log_g) result(tau)
        integer, intent(in) :: c_rich_flag
        real(WP), intent(in) :: m_act, log_l, log_g
        real(WP) :: tau
        real(WP) :: period_days, velocity_exp, mdot_sol_yr

        period_days = calc_period_days(m_act, log_g)
        velocity_exp = AGB_VEXP_INTERCEPT + AGB_VEXP_SLOPE * period_days
        velocity_exp = max(min(velocity_exp, AGB_VEXP_MAX), AGB_VEXP_MIN)
        mdot_sol_yr = (10.0_wp**log_l) / velocity_exp * VW93_SUPERWIND_NORM * YEAR_TO_SECOND / C_LIGHT

        tau = calc_tau_expected(c_rich_flag, m_act, log_l, log_g, mdot_sol_yr, .true.)
    end function calc_tau_expected_superwind

    pure function calc_dusty_transfer(ctx, c_rich_flag, log_t, tau_1um) result(transfer)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in) :: c_rich_flag
        real(WP), intent(in) :: log_t, tau_1um
        real(WP), dimension(size(ctx%state%flux_dagb, 1)) :: transfer
        integer :: idx_teff, idx_tau, n_teff_grid, n_tau_grid
        real(WP) :: w_teff, w_tau

        n_teff_grid = size(ctx%state%teff_dagb, 2)
        n_tau_grid  = size(ctx%state%tau1_dagb, 2)

        idx_teff = find_interval(ctx%state%teff_dagb(c_rich_flag + 1, :), 10.0_wp**log_t)
        idx_teff = max(1, min(idx_teff, n_teff_grid - 1))

        w_teff = (10.0_wp**log_t - ctx%state%teff_dagb(c_rich_flag + 1, idx_teff)) / &
                 (ctx%state%teff_dagb(c_rich_flag + 1, idx_teff + 1) - ctx%state%teff_dagb(c_rich_flag + 1, idx_teff))
        w_teff = max(-1.0_wp, min(w_teff, 1.0_wp))

        idx_tau = find_interval(ctx%state%tau1_dagb(c_rich_flag + 1, :), log10(tau_1um))
        idx_tau = max(1, min(idx_tau, n_tau_grid - 1))

        w_tau = (log10(tau_1um) - ctx%state%tau1_dagb(c_rich_flag + 1, idx_tau)) / &
                (ctx%state%tau1_dagb(c_rich_flag + 1, idx_tau + 1) - ctx%state%tau1_dagb(c_rich_flag + 1, idx_tau))
        w_tau = max(-1.0_wp, min(w_tau, 1.0_wp))

        transfer = &
            (1.0_wp - w_teff) * (1.0_wp - w_tau) * ctx%state%flux_dagb(:, c_rich_flag + 1, idx_teff, idx_tau) + &
            w_teff            * (1.0_wp - w_tau) * ctx%state%flux_dagb(:, c_rich_flag + 1, idx_teff + 1, idx_tau) + &
            w_teff            * w_tau            * ctx%state%flux_dagb(:, c_rich_flag + 1, idx_teff + 1, idx_tau + 1) + &
            (1.0_wp - w_teff) * w_tau            * ctx%state%flux_dagb(:, c_rich_flag + 1, idx_teff, idx_tau + 1)

    end function calc_dusty_transfer

    ! ------------------------------------------------------------------------
    ! Helper: Calzetti curve (for expected values)
    ! ------------------------------------------------------------------------
    pure function calc_calzetti_curve(wavelengths) result(curve)
        real(WP), dimension(:), intent(in) :: wavelengths
        real(WP), dimension(size(wavelengths)) :: curve
        real(WP), dimension(size(wavelengths)) :: wavenumbers, extinction_k

        wavenumbers = 1.0e4_wp / wavelengths
        extinction_k = 0.0_wp

        ! Eq. 4 of Calzetti et al. (2000), extended beyond 0.12-2.2 um and clipped at zero
        where (wavelengths > CALZ_LAM_BREAK)
            extinction_k = CALZ_SCALE * (CALZ_OPT_COEFFS(0) + CALZ_OPT_COEFFS(1) * wavenumbers) + CALZ_R_V
        end where

        where (wavelengths <= CALZ_LAM_BREAK)
            extinction_k = CALZ_R_V + CALZ_SCALE * ( &
                CALZ_UV_COEFFS(0) + wavenumbers * ( &
                    CALZ_UV_COEFFS(1) + wavenumbers * ( &
                        CALZ_UV_COEFFS(2) + wavenumbers * CALZ_UV_COEFFS(3) &
                    ) &
                ) &
            )
        end where

        curve = max(extinction_k, 0.0_wp) / CALZ_R_V

    end function calc_calzetti_curve

    ! ------------------------------------------------------------------------
    ! Helper: Drude bump profile (for expected values)
    ! ------------------------------------------------------------------------
    pure function calc_drude_profile(wavelengths, amplitude) result(profile)
        real(WP), dimension(:), intent(in) :: wavelengths
        real(WP), intent(in) :: amplitude
        real(WP), dimension(size(wavelengths)) :: profile

        profile = amplitude * (wavelengths * KC13_BUMP_WIDTH)**2 / &
                  ((wavelengths**2 - UV_BUMP_CENTER**2)**2 + (wavelengths * KC13_BUMP_WIDTH)**2)

    end function calc_drude_profile

end module test_opensps_dust_mod
