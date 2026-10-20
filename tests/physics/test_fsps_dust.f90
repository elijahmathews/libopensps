module test_fsps_dust_mod
    use fsps_precision, only: WP, KP
    use fsps_constants, only: NEMLINE, C_LIGHT, GRAVITY_L_M_T_COEFF, NTEFF_DAGB, NTAU_DAGB, &
                              NAGNDUST, M_SOL, G_NEWTON, R_SOL, YEAR_TO_SECOND, SAFE_FLOOR, PI
    use fsps_types, only: params
    use fsps_context_types, only: fsps_context_t
    use fsps_context, only: fsps_context_move_to_device, fsps_context_remove_from_device, &
                            fsps_context_prepare_csp_workspace
    use fsps_dust
    use fsps_integration, only: integrate_trapezoid_array
    use fsps_interpolation, only: find_interval
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_int_equals, assert_relative_error
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_is_finite
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_dust_tests, total_failures, total_tests

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

    ! AGB / VW93 constants (copied from fsps_dust for test expectations)
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
        type(fsps_context_t), intent(in)   :: ctx
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
    !> Unit test suite for fsps_dust module.
    subroutine run_fsps_dust_tests()

        call print_minor_header("fsps_dust")

        call test_power_law()
        call test_ccm89()
        call test_calzetti()
        call test_wg_smc()
        call test_kriek_conroy()
        call test_reddy()
        call test_agn_dust_emission()
        call test_circumstellar_optical_depth()
        call test_agb_dust_screen()
        call test_energy_conservation()
        call test_differential_attenuation()
        call test_draine_li_energy_balance()
        call test_dust_self_absorption()
        call test_robustness()
        call test_dust_early_returns()
        call test_dust_branches_and_type3()
        call test_dust_emission_zero_norm()
        call test_compute_attenuation_type5_point()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_dust_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: POWER LAW (Type 0)
    ! ------------------------------------------------------------------------
    subroutine test_power_law()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(2) :: w2
        real(WP), dimension(3) :: w3
        real(WP), dimension(1) :: w1
        real(WP), dimension(2) :: res2
        real(WP), dimension(3) :: res3
        real(WP), dimension(1) :: res1

        call print_group("Power Law (Type 0)")

        allocate(ctx)
        call fsps_context_move_to_device(ctx)

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

        call fsps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_power_law

    ! ------------------------------------------------------------------------
    ! TEST SUITE: CCM89 Milky Way (Type 1)
    ! ------------------------------------------------------------------------
    subroutine test_ccm89()
        type(fsps_context_t), allocatable :: ctx
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
        call fsps_context_move_to_device(ctx)

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

        call fsps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_ccm89
    ! ------------------------------------------------------------------------
    ! ------------------------------------------------------------------------
    ! TEST SUITE: Calzetti et al. (Type 2)
    ! ------------------------------------------------------------------------
    subroutine test_calzetti()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(2) :: w_break
        real(WP), dimension(2) :: res_break
        real(WP), dimension(1) :: w_v
        real(WP), dimension(1) :: res_v
        real(WP), dimension(6) :: w_ext, res_ext

        call print_group("Calzetti (Type 2)")

        allocate(ctx)
        call fsps_context_move_to_device(ctx)

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

        call fsps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_calzetti

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Witt & Gordon (Type 3) and SMC (Type 5)
    ! ------------------------------------------------------------------------
    subroutine test_wg_smc()
        type(fsps_context_t), allocatable :: ctx
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
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(3) :: wavelengths
        real(WP), dimension(3) :: res, expected
        real(WP), dimension(1) :: w_bump
        real(WP), dimension(1) :: res_bump, expected_no_bump
        real(WP), dimension(3) :: res_uvb0, res_uvb5

        call print_group("Kriek & Conroy (Type 4)")

        allocate(ctx)
        call fsps_context_move_to_device(ctx)

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

        call fsps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_kriek_conroy

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Reddy et al. (Type 6)
    ! ------------------------------------------------------------------------
    subroutine test_reddy()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        real(WP), dimension(2) :: w_uv
        real(WP), dimension(2) :: res_uv
        real(WP), dimension(2) :: w_break
        real(WP), dimension(2) :: res_break

        call print_group("Reddy et al. (Type 6)")

        allocate(ctx)
        call fsps_context_move_to_device(ctx)

        ! Test 6.1: UV extrapolation (lambda < 1500A constant)
        w_uv = [1400.0_wp, 1500.0_wp]
        res_uv = compute_attenuation_curve(w_uv, 6, settings, ctx)
        call assert_float_equals(res_uv(1), res_uv(2), 0.0_wp, "UV extrapolation clamp", total_tests, total_failures)

        ! Test 6.2: 6000A continuity
        w_break = [5999.999_wp, 6000.000_wp]
        res_break = compute_attenuation_curve(w_break, 6, settings, ctx)
        call assert_true(abs(res_break(1) - res_break(2)) <= SMALL_DELTA, "6000A continuity", total_tests, total_failures)

        call fsps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_reddy

    ! ------------------------------------------------------------------------
    ! TEST SUITE: AGN Dust Emission
    ! ------------------------------------------------------------------------
    subroutine test_agn_dust_emission()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        integer, parameter :: nlam = 5
        integer :: tmp_dust_type
        real(WP), dimension(nlam) :: wavelengths
        real(KP), dimension(nlam, 1) :: spectrum
        real(WP), dimension(nlam) :: expected
        real(WP), dimension(1) :: test_log_lbol

        call print_group("AGN Dust Emission")

        call setup_physics_context(ctx, nlam)
        wavelengths = ctx%state%spec_lambda

        ctx%dust_type_val = 0
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        settings%dust_index = 0.0_wp

        ! Test 1.1: Early exit when fagn=0
        settings%fagn = 0.0_wp
        spectrum(:, 1) = 100.0_wp
        test_log_lbol(1) = 0.0_wp
        !$acc data copyin(wavelengths, spectrum, settings, test_log_lbol)
        !$acc update device(settings)
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        call apply_agn_dust_emission(ctx, settings, 1, wavelengths, test_log_lbol, spectrum)
        !$acc update self(spectrum)
        !$acc end data
        call assert_true(all(abs(spectrum(:, 1) - 100.0_wp) <= EPS), "AGN early exit (fagn=0)", total_tests, total_failures)

        ! Test 1.2: Torus interpolation exact grid point
        settings%fagn = 1.0_wp
        settings%dust2 = 0.0_wp
        settings%agn_tau = 20.0_wp
        spectrum(:, 1) = 0.0_wp
        test_log_lbol(1) = 0.0_wp
        !$acc data copyin(wavelengths, spectrum, settings, test_log_lbol)
        !$acc update device(settings)
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        call apply_agn_dust_emission(ctx, settings, 1, wavelengths, test_log_lbol, spectrum)
        !$acc update self(spectrum)
        !$acc end data
        expected = 2.0_wp
        call assert_true(all(abs(spectrum(:, 1) - expected) <= EPS), "AGN grid exact (tau=20)", total_tests, total_failures)

        ! Test 1.3: Torus interpolation linear
        settings%agn_tau = 15.0_wp
        spectrum(:, 1) = 0.0_wp
        test_log_lbol(1) = 0.0_wp
        !$acc data copyin(wavelengths, spectrum, settings, test_log_lbol)
        !$acc update device(settings)
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        call apply_agn_dust_emission(ctx, settings, 1, wavelengths, test_log_lbol, spectrum)
        !$acc update self(spectrum)
        !$acc end data
        expected = 1.5_wp
        call assert_true(all(abs(spectrum(:, 1) - expected) <= EPS), "AGN grid linear (tau=15)", total_tests, total_failures)

        ! Test 1.4: Galaxy attenuation (power-law, dust2=1)
        settings%agn_tau = 10.0_wp
        settings%dust2 = 1.0_wp
        spectrum(:, 1) = 0.0_wp
        test_log_lbol(1) = 0.0_wp
        !$acc data copyin(wavelengths, spectrum, settings, test_log_lbol)
        !$acc update device(settings)
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        call apply_agn_dust_emission(ctx, settings, 1, wavelengths, test_log_lbol, spectrum)
        !$acc update self(spectrum)
        !$acc end data
        expected = exp(-1.0_wp)
        call assert_true(all(abs(spectrum(:, 1) - expected) <= EPS), "AGN host attenuation", total_tests, total_failures)

        ! Test 1.5: Type 3 exception (no dust2 scaling)
        ctx%dust_type_val = 3
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        settings%dust2 = 50.0_wp
        spectrum(:, 1) = 0.0_wp
        test_log_lbol(1) = 0.0_wp
        !$acc data copyin(wavelengths, spectrum, settings, test_log_lbol)
        !$acc update device(settings)
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        call apply_agn_dust_emission(ctx, settings, 1, wavelengths, test_log_lbol, spectrum)
        !$acc update self(spectrum)
        !$acc end data
        expected = exp(-1.0_wp)
        call assert_true(all(abs(spectrum(:, 1) - expected) <= EPS), "AGN Type 3 exception", total_tests, total_failures)

        ! Test 1.6: Luminosity normalization
        ctx%dust_type_val = 0
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        settings%dust2 = 0.0_wp
        settings%agn_tau = 10.0_wp
        settings%fagn = 0.1_wp
        spectrum(:, 1) = 0.0_wp
        test_log_lbol(1) = 10.0_wp
        !$acc data copyin(wavelengths, spectrum, settings, test_log_lbol)
        !$acc update device(settings)
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        call apply_agn_dust_emission(ctx, settings, 1, wavelengths, test_log_lbol, spectrum)
        !$acc update self(spectrum)
        !$acc end data
        expected = 1.0e9_wp
        call assert_true(all(abs(spectrum(:, 1) - expected) <= EPS), "AGN luminosity normalization", total_tests, total_failures)

        call teardown_physics_context(ctx)

    end subroutine test_agn_dust_emission

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Circumstellar Optical Depth
    ! ------------------------------------------------------------------------
    subroutine test_circumstellar_optical_depth()
        type(fsps_context_t), allocatable :: ctx
        real(WP) :: tau_c, tau_o, tau_expected_c, tau_expected_o
        real(WP) :: m_act, log_l, log_g, log_mdot_iso
        real(WP) :: period_days, vexp_unclamped
        real(WP) :: tau_clamped, tau_unclamped

        call print_group("Circumstellar Optical Depth")

        allocate(ctx)
        call fsps_context_move_to_device(ctx)
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

        call fsps_context_remove_from_device(ctx)
        deallocate(ctx)

    end subroutine test_circumstellar_optical_depth

    ! ------------------------------------------------------------------------
    ! TEST SUITE: AGB Dust Screen
    ! ------------------------------------------------------------------------
    subroutine test_agb_dust_screen()
        type(fsps_context_t), allocatable :: ctx
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

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Energy Conservation (AGN + AGB)
    ! ------------------------------------------------------------------------
    subroutine test_energy_conservation()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        integer, parameter :: nlam = 5
        integer :: tmp_dust_type
        real(KP), dimension(nlam, 1) :: spectrum
        real(WP), dimension(nlam) :: wavelengths
        real(KP), dimension(nlam) :: spectrum_out
        real(WP), dimension(nlam) :: freqs
        real(WP), dimension(1) :: test_log_lbol
        real(WP) :: lbol_in, lbol_out
        real(WP) :: norm_template

        call print_group("Energy Conservation")

        call setup_physics_context(ctx, nlam)
        wavelengths = ctx%state%spec_lambda
        freqs = C_LIGHT / wavelengths

        ! Test 4.1: AGN energy conservation (template normalized)
        spectrum(:, 1) = 1.0_wp
        lbol_in = integrate_trapezoid_array(freqs, real(spectrum(:, 1), WP))

        norm_template = lbol_in
        ctx%state%agndust_spec(:, :) = 1.0_wp / norm_template

        settings%fagn = 0.1_wp
        settings%dust2 = 0.0_wp
        settings%agn_tau = 10.0_wp
        test_log_lbol(1) = log10(lbol_in)
        ctx%dust_type_val = 0
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        ctx%state%agndust_spec(:, :) = 1.0_wp / norm_template
        !$acc update device(ctx%dust_type_val, ctx%state%agndust_spec(:, :))

        !$acc data copyin(wavelengths, spectrum, settings, test_log_lbol)
        !$acc update device(settings)
        tmp_dust_type = ctx%dust_type_val
        ctx%dust_type_val = tmp_dust_type
        !$acc update device(ctx%dust_type_val)
        call apply_agn_dust_emission(ctx, settings, 1, wavelengths, test_log_lbol, spectrum)
        !$acc update self(spectrum)
        !$acc end data
        lbol_out = integrate_trapezoid_array(freqs, real(spectrum(:, 1), WP))
        call assert_relative_error(lbol_in * 1.1_wp, lbol_out, 1.0e-6_wp, "AGN energy conservation", total_tests, total_failures)

        ! Test 4.2: AGB energy conservation (attenuation only)
        call set_flux_dagb_constant(ctx, nlam, 0.5_wp)
        spectrum_out = 1.0_wp
        lbol_in = integrate_trapezoid_array(freqs, real(spectrum_out, WP))
        call apply_agb_dust_screen(ctx, 1.0_wp, spectrum_out, 1.0_wp, 3.6_wp, 3.0_wp, 0.0_wp, 1.5_wp, -5.0_wp)
        lbol_out = integrate_trapezoid_array(freqs, real(spectrum_out, WP))
        call assert_true(lbol_out < lbol_in, "AGB screen reduces flux", total_tests, total_failures)

        call teardown_physics_context(ctx)

    end subroutine test_energy_conservation

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Differential Attenuation
    ! ------------------------------------------------------------------------
    subroutine test_differential_attenuation()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        integer, parameter :: nlam = 5
        real(WP) :: tmp_nebem_line_pos1
        real(WP), dimension(nlam) :: spec_young, spec_old, spec_out
        real(WP), dimension(NEMLINE) :: neb_young, neb_old, neb_out
        real(WP) :: dust_mass

        call print_group("Differential Attenuation")

        call setup_physics_context(ctx, nlam)
        !$acc enter data create(settings)

        ctx%dust_type_val = 0
        ctx%add_dust_emission_val = 0
        ctx%nebemlineinspec_val = 1
        !$acc update device(ctx%dust_type_val, ctx%add_dust_emission_val, ctx%nebemlineinspec_val)

        settings%dust_index = 0.0_wp
        settings%dust1_index = 0.0_wp
        settings%frac_nodust = 0.0_wp
        settings%frac_obrun = 0.0_wp
        settings%dust3 = 0.0_wp

        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp

        ctx%state%spec_young(:, 1) = 0.0_wp
        ctx%state%spec_old(:, 1) = 100.0_wp
        settings%dust1 = 5.0_wp
        settings%dust2 = 1.0_wp
        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), ctx%state%csp_emlin_young(:, 1), &
        !$acc               ctx%state%csp_emlin_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1))

        call assert_true(all(abs(ctx%state%dust_spec_total_work(:, 1) - 100.0_wp * exp(-1.0_wp)) <= EPS), &
                         "Old stars diffuse only", total_tests, total_failures)

        ctx%state%spec_young(:, 1) = 100.0_wp
        ctx%state%spec_old(:, 1) = 0.0_wp
        settings%dust1 = 1.0_wp
        settings%dust2 = 1.0_wp
        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1))

        call assert_true(all(abs(ctx%state%dust_spec_total_work(:, 1) - 100.0_wp * exp(-2.0_wp)) <= EPS), &
                         "Young stars birth+diffuse", total_tests, total_failures)

        ctx%state%spec_young(:, 1) = 100.0_wp
        ctx%state%spec_old(:, 1) = 0.0_wp
        settings%dust1 = 100.0_wp
        settings%dust2 = 0.0_wp
        settings%frac_obrun = 0.25_wp
        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1))

        call assert_true(all(abs(ctx%state%dust_spec_total_work(:, 1) - 25.0_wp) <= LESS_SMALL_DELTA), &
                         "OB runaways", total_tests, total_failures)

        ctx%state%spec_young(:, 1) = 0.0_wp
        ctx%state%spec_old(:, 1) = 100.0_wp
        settings%frac_obrun = 0.0_wp
        settings%dust2 = 100.0_wp
        settings%frac_nodust = 0.10_wp
        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1))

        call assert_true(all(abs(ctx%state%dust_spec_total_work(:, 1) - 10.0_wp) <= LESS_SMALL_DELTA), &
                         "Patchy ISM", total_tests, total_failures)

        settings%frac_nodust = 0.0_wp
        settings%dust2 = 0.0_wp
        settings%dust1 = 1.0_wp
        settings%dust1_index = -1.0_wp
        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp
        ctx%state%csp_emlin_young(1, 1) = 1.0_wp
        ctx%state%nebem_line_pos = 5500.0_wp
        tmp_nebem_line_pos1 = ctx%state%nebem_line_pos(1)
        ctx%state%nebem_line_pos(1) = tmp_nebem_line_pos1
        !$acc update device(ctx%state%nebem_line_pos(1), ctx%state%csp_emlin_young(:, 1), ctx%state%csp_emlin_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%csp_emlin_young(:, 1), ctx%state%csp_emlin_old(:, 1))

        call assert_float_equals(exp(-1.0_wp), real(ctx%state%csp_emlin_young(1, 1) + ctx%state%csp_emlin_old(1, 1), WP), &
                                 1.0e-6_wp, &
                                 "Nebular line attenuation", total_tests, total_failures)

        !$acc exit data delete(settings)
        call fsps_context_remove_from_device(ctx)
        call teardown_physics_context(ctx)

    end subroutine test_differential_attenuation

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Draine & Li + Energy Balance
    ! ------------------------------------------------------------------------
    subroutine test_draine_li_energy_balance()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        integer, parameter :: nlam = 5
        real(WP), dimension(nlam) :: spec_young, spec_old, spec_out
        real(WP), dimension(nlam) :: freqs
        real(KP), dimension(nlam) :: dust_shape
        real(WP), dimension(NEMLINE) :: neb_young, neb_old, neb_out
        real(WP) :: dust_mass, lbol_in, lbol_out
        real(WP) :: lbol_atten, lbol_abs, emission_norm
        real(WP) :: expected_mass
        real(WP), dimension(nlam) :: spec_atten

        call print_group("Draine & Li Energy Balance")

        call setup_physics_context(ctx, nlam)
        !$acc enter data create(settings)

        ctx%dust_type_val = 0
        ctx%add_dust_emission_val = 1
        ctx%nebemlineinspec_val = 1
        !$acc update device(ctx%dust_type_val, ctx%add_dust_emission_val, ctx%nebemlineinspec_val)

        settings%dust_index = 0.0_wp
        settings%dust1_index = 0.0_wp
        settings%dust1 = 0.0_wp
        settings%dust2 = 1.0_wp
        settings%frac_nodust = 0.0_wp
        settings%frac_obrun = 0.0_wp
        settings%duste_gamma = 0.1_wp
        settings%duste_qpah = 1.0_wp
        settings%duste_umin = 0.5_wp

        ctx%state%spec_young(:, 1) = 100.0_wp
        ctx%state%spec_old(:, 1) = 0.0_wp
        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp

        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), ctx%state%csp_emlin_young(:, 1), &
        !$acc               ctx%state%csp_emlin_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1), ctx%state%sfh_w_tmp1(3:3, 1))

        dust_mass = ctx%state%sfh_w_tmp1(3, 1)

        freqs = C_LIGHT / ctx%state%spec_lambda
        lbol_in = integrate_trapezoid_array(freqs, real(ctx%state%spec_young(:, 1), WP))
        lbol_out = integrate_trapezoid_array(freqs, real(ctx%state%dust_spec_total_work(:, 1), WP))
        call assert_relative_error(lbol_in, lbol_out, 1.0e-6_wp, "Energy conservation", total_tests, total_failures)

        call interpolate_draine_li_dust_model(ctx, settings, dust_shape)
        call assert_true(all(abs(dust_shape - 1.9_wp) <= EPS), "Draine-Li gamma weighting", total_tests, total_failures)
        emission_norm = integrate_trapezoid_array(freqs, real(dust_shape, WP))
        spec_atten = 100.0_wp * exp(-1.0_wp)
        lbol_atten = integrate_trapezoid_array(freqs, spec_atten)
        lbol_abs = lbol_in - lbol_atten
        expected_mass = 3.21e-3_wp / (4.0_wp * PI) * (lbol_abs / emission_norm)
        call assert_float_equals(expected_mass, dust_mass, 1.0e-6_wp, "Dust mass scaling", total_tests, total_failures)

        !$acc exit data delete(settings)
        call fsps_context_remove_from_device(ctx)
        call teardown_physics_context(ctx)

    end subroutine test_draine_li_energy_balance

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Dust Self-Absorption
    ! ------------------------------------------------------------------------
    subroutine test_dust_self_absorption()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP) :: mdust
        integer :: nlam

        call print_group("Dust Self-Absorption")

        nlam = 100
        call setup_physics_context(ctx, nlam)
        !$acc enter data create(pset)

        ctx%state%spec_young(:, 1) = 1.0_wp
        ctx%state%spec_old(:, 1) = 0.0_wp
        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp
        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), &
        !$acc               ctx%state%csp_emlin_young(:, 1), ctx%state%csp_emlin_old(:, 1))

        ! Base settings to trigger emission
        ctx%dust_type_val = 0
        pset%dust1 = 0.0_wp
        pset%dust2 = 1.0_wp  ! Turn on diffuse dust to absorb energy
        pset%dust_index = 0.0_wp
        pset%frac_obrun = 0.0_wp
        pset%frac_nodust = 0.0_wp
        pset%duste_gamma = 0.0_wp
        pset%duste_qpah = 2.0_wp
        pset%duste_umin = 1.0_wp

        ctx%add_dust_emission_val = 1
        ctx%nebemlineinspec_val = 0

        ! Sync flags to the GPU so kernels use the updated logic
        !$acc update device(ctx%dust_type_val, ctx%add_dust_emission_val, ctx%nebemlineinspec_val)

        ! 1. Transparent Limit (Zero Absorption)
        ! If we set dust2 = 0, no energy is absorbed, so dust mass should be 0
        pset%dust2 = 0.0_wp
        !$acc update device(pset)
        call apply_dust_attenuation_and_emission(ctx, pset, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%sfh_w_tmp1(3:3, 1))
        mdust = ctx%state%sfh_w_tmp1(3, 1)

        call assert_float_equals(0.0_wp, mdust, EPS, &
                                 "Transparent limit yields zero dust mass", total_tests, total_failures)

        ! 2. Opaque Limit (Energy Conservation)
        ! Set dust2 high. Ensure emitted IR energy matches absorbed optical energy.
        pset%dust2 = 2.0_wp
        !$acc update device(pset)
        call apply_dust_attenuation_and_emission(ctx, pset, 1)
        !$acc update host(ctx%state%sfh_w_tmp1(3:3, 1))
        mdust = ctx%state%sfh_w_tmp1(3, 1)

        call assert_true(mdust > 0.0_wp, &
                         "Opaque limit yields positive dust mass", total_tests, total_failures)

        !$acc exit data delete(pset)
        call fsps_context_remove_from_device(ctx)
        call teardown_physics_context(ctx)
    end subroutine test_dust_self_absorption

    ! ------------------------------------------------------------------------
    ! TEST SUITE: Robustness
    ! ------------------------------------------------------------------------
    subroutine test_robustness()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        integer :: tmp_dust_type, tmp_add_dust, tmp_nebemline
        real(WP), dimension(:), allocatable :: w
        real(WP), dimension(:), allocatable :: res
        real(WP) :: dust_mass
        integer :: n
        integer :: i

        call print_group("Robustness")

        allocate(ctx)
        call fsps_context_move_to_device(ctx)

        ! Test 7.1: Invalid ID returns NaN
        allocate(w(3))
        w = [1000.0_wp, 2000.0_wp, 3000.0_wp]
        res = compute_attenuation_curve(w, 99, settings, ctx)
        call assert_true(all(ieee_is_nan(res)), "Invalid ID returns NaN", total_tests, total_failures)
        deallocate(w, res)

        call fsps_context_remove_from_device(ctx)
        deallocate(ctx)

        ! Test 8.1: No dust (output equals input, dust mass ~ 0)
        call setup_physics_context(ctx, 5)
        !$acc enter data create(settings)

        ctx%dust_type_val = 0
        ctx%add_dust_emission_val = 0
        ctx%nebemlineinspec_val = 1
        tmp_dust_type = ctx%dust_type_val
        tmp_add_dust = ctx%add_dust_emission_val
        tmp_nebemline = ctx%nebemlineinspec_val
        ctx%dust_type_val = tmp_dust_type
        ctx%add_dust_emission_val = tmp_add_dust
        ctx%nebemlineinspec_val = tmp_nebemline
        !$acc update device(ctx%dust_type_val, ctx%add_dust_emission_val, ctx%nebemlineinspec_val)

        settings%dust_index = 0.0_wp
        settings%dust1 = 0.0_wp
        settings%dust2 = 0.0_wp
        settings%dust3 = 0.0_wp
        settings%frac_nodust = 0.0_wp
        settings%frac_obrun = 0.0_wp

        n = size(ctx%state%spec_lambda)
        ctx%state%spec_young(:, 1) = 100.0_wp
        ctx%state%spec_old(:, 1) = 0.0_wp
        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp

        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), ctx%state%csp_emlin_young(:, 1), &
        !$acc               ctx%state%csp_emlin_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1), ctx%state%sfh_w_tmp1(3:3, 1))

        dust_mass = ctx%state%sfh_w_tmp1(3, 1)

        call assert_true(all(abs(ctx%state%dust_spec_total_work(:, 1) - 100.0_wp) <= EPS), &
                         "No dust spectrum unchanged", total_tests, total_failures)
        call assert_true(dust_mass <= SAFE_FLOOR, "No dust mass ~ 0", total_tests, total_failures)

        !$acc exit data delete(settings)
        call fsps_context_remove_from_device(ctx)
        call teardown_physics_context(ctx)

        ! Test 7.2: Array shape conformance
        allocate(ctx)
        call fsps_context_move_to_device(ctx)
        n = 1
        allocate(w(n), res(n))
        w = [5500.0_wp]
        res = compute_attenuation_curve(w, 0, settings, ctx)
        call assert_int_equals(size(w), size(res), "Array shape (size=1)", total_tests, total_failures)
        deallocate(w, res)

        n = 10
        allocate(w(n), res(n))
        w = [(real(i, WP), i=1, n)]
        res = compute_attenuation_curve(w, 0, settings, ctx)
        call assert_int_equals(size(w), size(res), "Array shape (size=10)", total_tests, total_failures)
        deallocate(w, res)

        call fsps_context_remove_from_device(ctx)
        deallocate(ctx)
        allocate(w(n), res(n))
        w = [(real(i, WP), i=1, n)]
        res = compute_attenuation_curve(w, 0, settings, ctx)
        call assert_int_equals(size(w), size(res), "Array shape (size=1000)", total_tests, total_failures)
        deallocate(w, res)

    end subroutine test_robustness


    ! ------------------------------------------------------------------------
    ! TEST SUITE: Edge Cases
    ! ------------------------------------------------------------------------
    subroutine test_dust_early_returns()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings

        call print_group("Dust Early Returns")
        call setup_physics_context(ctx, 5)
        !$acc enter data create(settings)

        ! Set up a recognizable baseline
        ctx%state%dust_spec_total_work(:, 1) = 99.0_wp
        !$acc update device(ctx%state%dust_spec_total_work(:, 1))

        ! 1. uvb < 0
        settings%uvb = -1.0_wp
        settings%wgp1 = 1
        settings%wgp2 = 1
        !$acc update device(settings)
        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1))
        call assert_float_equals(99.0_wp, real(ctx%state%dust_spec_total_work(1, 1), WP), EPS, &
                                 "Early return triggered for uvb < 0", total_tests, total_failures)

        ! 2. wgp1 < 1
        settings%uvb = 1.0_wp
        settings%wgp1 = 0
        settings%wgp2 = 1
        !$acc update device(settings)
        call apply_dust_attenuation_and_emission(ctx, settings, 1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1))
        call assert_float_equals(99.0_wp, real(ctx%state%dust_spec_total_work(1, 1), WP), EPS, &
                                 "Early return triggered for wgp1 < 1", total_tests, total_failures)

        !$acc exit data delete(settings)
        call fsps_context_remove_from_device(ctx)
        call teardown_physics_context(ctx)
    end subroutine test_dust_early_returns

    subroutine test_dust_branches_and_type3()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        integer :: tmp_dust_type, tmp_add_dust, tmp_nebemline

        call print_group("Dust Branches & Type 3")
        call setup_physics_context(ctx, 5)
        !$acc enter data create(settings)

        ! Activate Type 3
        ctx%dust_type_val = 3
        ctx%add_dust_emission_val = 0
        ctx%nebemlineinspec_val = 0

        tmp_dust_type = ctx%dust_type_val
        tmp_add_dust = ctx%add_dust_emission_val
        tmp_nebemline = ctx%nebemlineinspec_val
        ctx%dust_type_val = tmp_dust_type
        ctx%add_dust_emission_val = tmp_add_dust
        ctx%nebemlineinspec_val = tmp_nebemline
        !$acc update device(ctx%dust_type_val, ctx%add_dust_emission_val, ctx%nebemlineinspec_val)

        settings%uvb = 1.0_wp
        settings%wgp1 = 2
        settings%wgp2 = 3
        settings%wgp3 = 1

        ! Variables to trigger the missing if/else blocks:
        settings%dust3 = 2.0_wp       ! Hits trans_old = exp(-dust3 * curve)
        settings%dust1 = 5.0_wp
        settings%dust1_index = -1.0_wp
        settings%frac_obrun = 1.0_wp  ! Hits trans_birth = 1.0_wp
        settings%frac_nodust = 1.0_wp ! Hits the `one_minus_nodust <= SAFE_FLOOR` branch

        ctx%state%spec_young(:, 1) = 100.0_wp
        ctx%state%spec_old(:, 1) = 100.0_wp
        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp

        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), &
        !$acc               ctx%state%csp_emlin_young(:, 1), ctx%state%csp_emlin_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)

        !$acc wait(1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1))

        ! Because frac_nodust = 1.0, the calculation skips diffuse geometry completely.
        ! Output evaluates purely to: spec_young * trans_birth + spec_old * trans_old
        ! With obrun=1.0, trans_birth=1.0. With W&G, curve=1.0, so trans_old=exp(-2.0*1.0)
        call assert_float_equals(100.0_wp + 100.0_wp * exp(-2.0_wp), &
                                 real(ctx%state%dust_spec_total_work(1, 1), WP), &
                                 max(EPS, KP_RTOL * (100.0_wp + 100.0_wp * exp(-2.0_wp))), &
                                 "Evaluates correct fallback logic with nodust=1 and obrun=1", &
                                 total_tests, total_failures)

        !$acc exit data delete(settings)
        call fsps_context_remove_from_device(ctx)
        call teardown_physics_context(ctx)
    end subroutine test_dust_branches_and_type3

    subroutine test_dust_emission_zero_norm()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: settings
        integer :: tmp_dust_type, tmp_add_dust, tmp_nebemline

        call print_group("Dust Emission Zero Norm Edge Cases")
        call setup_physics_context(ctx, 5)
        !$acc enter data create(settings)

        ctx%dust_type_val = 0
        ctx%add_dust_emission_val = 1
        ctx%nebemlineinspec_val = 0

        tmp_dust_type = ctx%dust_type_val
        tmp_add_dust = ctx%add_dust_emission_val
        tmp_nebemline = ctx%nebemlineinspec_val
        ctx%dust_type_val = tmp_dust_type
        ctx%add_dust_emission_val = tmp_add_dust
        ctx%nebemlineinspec_val = tmp_nebemline

        ! Force df = 0 by making all wavelengths identical.
        ! This guarantees emission_norm_factor and lum_escaped_profile stay exactly 0.0.
        ctx%state%spec_lambda(:) = 5000.0_wp

        !$acc update device(ctx%dust_type_val, ctx%add_dust_emission_val, ctx%nebemlineinspec_val)
        !$acc update device(ctx%state%spec_lambda)

        settings%dust_index = 0.0_wp
        settings%dust1 = 0.0_wp
        settings%dust2 = 1.0_wp

        ! Critical: must be 1.0 to avoid divide-by-zero in the nebular diffuse interpolation
        ! since all wavelengths are now identically 5000.0
        settings%frac_nodust = 1.0_wp

        settings%frac_obrun = 0.0_wp
        settings%duste_gamma = 0.1_wp
        settings%duste_qpah = 1.0_wp
        settings%duste_umin = 0.5_wp

        ctx%state%spec_young(:, 1) = 100.0_wp
        ctx%state%spec_old(:, 1) = 0.0_wp
        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp

        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), &
        !$acc               ctx%state%csp_emlin_young(:, 1), ctx%state%csp_emlin_old(:, 1), settings)

        call apply_dust_attenuation_and_emission(ctx, settings, 1)

        !$acc wait(1)
        !$acc update host(ctx%state%sfh_w_tmp1(3:3, 1), ctx%state%dust_emission_final(:, 1))

        call assert_float_equals(SAFE_FLOOR, ctx%state%sfh_w_tmp1(3, 1), EPS, &
                                 "Zero emission norm cleanly falls back to SAFE_FLOOR mdust", total_tests, total_failures)
        call assert_true(all(ctx%state%dust_emission_final(:, 1) == 0.0_wp), &
                         "Zero escaped lum cleanly falls back to 0.0 emission", total_tests, total_failures)

        !$acc exit data delete(settings)
        call fsps_context_remove_from_device(ctx)
        call teardown_physics_context(ctx)
    end subroutine test_dust_emission_zero_norm

    subroutine test_compute_attenuation_type5_point()
        type(fsps_context_t), allocatable :: ctx
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
        type(fsps_context_t), allocatable, intent(out) :: ctx
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

        call fsps_context_prepare_csp_workspace(ctx)
        call fsps_context_move_to_device(ctx)
    end subroutine setup_mock_context

    subroutine teardown_mock_context(ctx)
        type(fsps_context_t), allocatable, intent(inout) :: ctx
        
        call fsps_context_remove_from_device(ctx)
        
        if (associated(ctx%state%wgdust)) deallocate(ctx%state%wgdust)
        if (associated(ctx%state%g03smcextn)) deallocate(ctx%state%g03smcextn)
        deallocate(ctx)

    end subroutine teardown_mock_context

    ! ------------------------------------------------------------------------
    ! Helper: mock context setup for full dust physics
    ! ------------------------------------------------------------------------
    subroutine setup_physics_context(ctx, nlam)
        type(fsps_context_t), allocatable, intent(out) :: ctx
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

        call fsps_context_prepare_csp_workspace(ctx)
        call fsps_context_move_to_device(ctx)
    end subroutine setup_physics_context

    subroutine teardown_physics_context(ctx)
        type(fsps_context_t), allocatable, intent(inout) :: ctx

        call fsps_context_remove_from_device(ctx)

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
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: nlam
        real(WP), intent(in) :: value

        ctx%state%flux_dagb(1:nlam, :, :, :) = value

    end subroutine set_flux_dagb_constant

    subroutine set_flux_dagb_pattern(ctx, nlam)
        type(fsps_context_t), intent(inout) :: ctx
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
        type(fsps_context_t), intent(in) :: ctx
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

end module test_fsps_dust_mod
