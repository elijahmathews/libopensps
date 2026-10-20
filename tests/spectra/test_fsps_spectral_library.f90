module test_fsps_spectral_library_mod
    use fsps_precision, only: WP, KP
    use fsps_constants, only: SAFE_FLOOR, PI, C_LIGHT, L_SOL, M_SOL, G_NEWTON, GRAVITY_L_M_T_COEFF, &
                              NDIM_LOGT, NDIM_LOGG, NDIM_WMB_LOGT, NDIM_WMB_LOGG, NDIM_WR, NDIM_PAGB, &
                              N_AGB_O, N_AGB_C, N_AGB_CAR, NTEFF_DAGB, NTAU_DAGB
    use fsps_context_types, only: fsps_context_t
    use fsps_types, only: params
    use fsps_spectral_library, only: get_stellar_spectrum
    use fsps_dust, only: compute_circumstellar_optical_depth
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_relative_error
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_spectral_library_tests, total_failures, total_tests

contains

    subroutine run_fsps_spectral_library_tests()
        call print_minor_header("fsps_spectral_library")

        call test_hot_star_handoff()
        call test_carbon_star_fork()
        call test_wr_enable_switch()

        call test_stefan_boltzmann_identity()
        call test_recalculated_gravity_check()
        call test_zero_luminosity_safety()

        call test_bilinear_plane()
        call test_partial_overlap_snap()
        call test_agb_o_stack_allocation()

        call test_wr_wind_temperature()
        call test_agb_dust_screen()

        call test_scaled_output_identity()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_spectral_library_tests

    ! --------------------------------------------------------------------
    ! Group 1: Architectural Dispatch & Library Selection
    ! --------------------------------------------------------------------
    subroutine test_hot_star_handoff()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: expected_main, scale_factor
        integer :: nlam

        call print_group("Hot Star Handoff (WMBasic vs Main)")

        nlam = 8
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 4.8_wp
        lbol = 1.0_wp
        logg = 4.0_wp
        phase = 0.0_wp
        ffco = 0.0_wp
        lmdot = -6.0_wp

        ctx%state%speclib = real(1.0_wp, kind=kind(ctx%state%speclib))
        ctx%state%wmb_spec = real(7.0_wp, kind=kind(ctx%state%wmb_spec))

        ctx%logt_wmb_hot_val = 4.7_wp
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        call assert_float_equals(7.0_wp, real(spec(1), WP), 1.0e-6_wp, &
                                 "WMBasic used when logt > cutoff", total_tests, total_failures)

        ctx%logt_wmb_hot_val = 4.9_wp
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)

        scale_factor = compute_scale_factor_main(mact, lbol, logt, logg)
        expected_main = scale_factor * 1.0_wp
        call assert_relative_error(expected_main, real(spec(1), WP), 1.0e-6_wp, &
                                   "Main lib used when cutoff higher", total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_hot_star_handoff

    subroutine test_carbon_star_fork()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: expected_o
        integer :: nlam

        call print_group("Carbon Star Fork (AGB O vs C)")

        nlam = 6
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 3.5_wp
        lbol = 1.0_wp
        logg = 0.0_wp
        phase = 5.0_wp
        lmdot = -6.0_wp

        ctx%state%agb_spec_o = 2.0_wp
        ctx%state%agb_spec_car = 20.0_wp

        ffco = 1.01_wp
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        call assert_float_equals(20.0_wp, real(spec(1), WP), 1.0e-6_wp, "C-rich uses AGB C library", total_tests, total_failures)

        ffco = 1.0_wp
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        expected_o = (ctx%state%spec_lambda(1)**2 / C_LIGHT) * 2.0_wp
        call assert_relative_error(expected_o, real(spec(1), WP), 1.0e-6_wp, &
                                   "O-rich uses AGB O library", total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_carbon_star_fork

    subroutine test_wr_enable_switch()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: expected_main, scale_factor
        integer :: nlam

        call print_group("Wolf-Rayet Enable Switch")

        nlam = 6
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 4.6_wp
        lbol = 1.0_wp
        logg = 4.0_wp
        phase = 9.0_wp
        ffco = 0.0_wp
        lmdot = -5.0_wp

        ctx%state%speclib = real(1.0_wp, kind=kind(ctx%state%speclib))
        ctx%state%wrn_spec = 33.0_wp

        ctx%use_wr_spectra_val = 0
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        scale_factor = compute_scale_factor_main(mact, lbol, logt, logg)
        expected_main = scale_factor * 1.0_wp
        call assert_relative_error(expected_main, real(spec(1), WP), 1.0e-6_wp, &
                                   "WR disabled -> Main library", total_tests, total_failures)

        ctx%use_wr_spectra_val = 1
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        call assert_float_equals(33.0_wp, real(spec(1), WP), 1.0e-6_wp, "WR enabled -> WR library", total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_wr_enable_switch

    ! --------------------------------------------------------------------
    ! Group 2: Physical Consistency & Scaling Laws
    ! --------------------------------------------------------------------
    subroutine test_stefan_boltzmann_identity()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP), allocatable :: nu(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: scale_factor, hnu, total_lbol
        integer :: nlam

        call print_group("Stefan-Boltzmann Identity (Surface Flux Scaling)")

        nlam = 200
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam), nu(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 3.7617_wp
        lbol = 1.0_wp
        logg = 4.0_wp
        phase = 0.0_wp
        ffco = 0.0_wp
        lmdot = -6.0_wp

        scale_factor = compute_scale_factor_main(mact, lbol, logt, logg)
        nu = C_LIGHT / ctx%state%spec_lambda
        hnu = lbol / (scale_factor * (maxval(nu) - minval(nu)))
        ctx%state%speclib = real(hnu, kind=kind(ctx%state%speclib))

        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        total_lbol = integrate_trapezoid(reverse_array(nu), reverse_array(real(spec, WP)))

        call assert_relative_error(lbol, total_lbol, 1.0e-6_wp, "Integrated Lbol matches L_sol", total_tests, total_failures)

        deallocate(spec, nu)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_stefan_boltzmann_identity

    subroutine test_recalculated_gravity_check()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg_in, phase, ffco, lmdot
        real(WP) :: logg_calc, scale_factor
        integer :: nlam, j, k

        call print_group("Recalculated Gravity Check")

        nlam = 5
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 3.7617_wp
        lbol = 1.0_wp
        logg_in = 6.0_wp
        phase = 0.0_wp
        ffco = 0.0_wp
        lmdot = -6.0_wp

        do j = 1, NDIM_LOGT
            do k = 1, NDIM_LOGG
                ctx%state%speclib(:, 1, j, k) = real(ctx%state%speclib_logg(k), kind=kind(ctx%state%speclib))
            end do
        end do

        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg_in, phase, ffco, lmdot, spec)

        logg_calc = compute_logg_calc(mact, lbol, logt, logg_in)
        scale_factor = compute_scale_factor_main(mact, lbol, logt, logg_in)

        call assert_relative_error(logg_calc, spec(1) / scale_factor, 1.0e-6_wp, &
                                   "Logg recalculated (ignores logg_in)", total_tests, total_failures)
        call assert_true(abs(spec(1) / scale_factor - logg_in) > 0.1_wp, &
                         "Logg differs from input", total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_recalculated_gravity_check

    subroutine test_zero_luminosity_safety()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        integer :: nlam

        call print_group("Zero-Luminosity Safety")

        nlam = 6
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 4.5_wp
        lbol = 0.0_wp
        logg = -99.0_wp
        phase = 9.0_wp
        ffco = 0.0_wp
        lmdot = -3.0_wp

        ctx%use_wr_spectra_val = 1
        ctx%state%wrn_spec = 5.0_wp

        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)

        call assert_true(all(.not. ieee_is_nan(spec)), "No NaNs for lbol=0", total_tests, total_failures)
        ! The floor is tiny(1.0_KP): SAFE_FLOOR in the default build
        call assert_float_equals(real(tiny(1.0_KP), WP), real(spec(1), WP), 0.0_wp, "SAFE_FLOOR output for lbol=0", &
                                 total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_zero_luminosity_safety

    ! --------------------------------------------------------------------
    ! Group 3: Interpolation Accuracy & Grid Dynamics
    ! --------------------------------------------------------------------
    subroutine test_bilinear_plane()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: scale_factor, expected, target_logg
        integer :: nlam, j, k

        call print_group("Bilinear Interpolation on Plane S(T,g)=T+g")

        nlam = 4
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 3.5_wp
        target_logg = 4.5_wp 
        
        ! Calculate the Lbol required to produce logg=4.5 physically
        lbol = compute_lbol_for_logg(mact, logt, target_logg)
        
        ! We still pass target_logg as input, though the module ignores it
        logg = target_logg 
        phase = 0.0_wp
        ffco = 0.0_wp
        lmdot = -6.0_wp

        ! Fill grid with Plane Equation: Value = T + g
        do j = 1, NDIM_LOGT
            do k = 1, NDIM_LOGG
                ctx%state%speclib(:, 1, j, k) = real(ctx%state%speclib_logt(j) + ctx%state%speclib_logg(k), &
                                                   kind=kind(ctx%state%speclib))
            end do
        end do

        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        
        ! Calculate scale factor (using the physically consistent parameters)
        scale_factor = compute_scale_factor_main(mact, lbol, logt, logg)
        
        expected = logt + target_logg

        call assert_relative_error(expected, spec(1) / scale_factor, 1.0e-6_wp, &
                                   "Bilinear plane returns exact midpoint", total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_bilinear_plane

    subroutine test_partial_overlap_snap()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: scale_factor, expected, target_logg
        real(WP) :: logt_lo, logg_lo, dlogt, dlogg
        integer :: nlam

        call print_group("Partial Overlap Snap (No Averaging with SAFE_FLOOR)")

        nlam = 4
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        phase = 0.0_wp
        ffco = 0.0_wp
        lmdot = -6.0_wp

        ctx%state%speclib = real(0.0_wp, kind=kind(ctx%state%speclib))

        logt_lo = ctx%state%speclib_logt(1)
        logg_lo = ctx%state%speclib_logg(1)
        dlogt = ctx%state%speclib_logt(2) - ctx%state%speclib_logt(1)
        dlogg = ctx%state%speclib_logg(2) - ctx%state%speclib_logg(1)

        ! Target a point inside the first grid cell
        logt = logt_lo + 0.1_wp * dlogt
        target_logg = logg_lo + 0.1_wp * dlogg
        
        ! Force physics engine to agree with our grid location
        lbol = compute_lbol_for_logg(mact, logt, target_logg)
        logg = target_logg

        ! Populate ONLY Corner 1 (Index 1,1) with valid data
        ctx%state%speclib(:, 1, 1, 1) = real(100.0_wp, kind=kind(ctx%state%speclib))

        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        
        scale_factor = compute_scale_factor_main(mact, lbol, logt, logg)
        expected = 100.0_wp * scale_factor

        call assert_relative_error(expected, real(spec(1), WP), 1.0e-6_wp, "Snaps to valid corner", total_tests, total_failures)

        ! With several valid corners, legacy FSPS keeps the last valid one in the order (j,k), (j+1,k),
        ! (j,k+1), (j+1,k+1), even when another has a larger bilinear weight (here (j,k) does)
        ctx%state%speclib(:, 1, 2, 2) = real(300.0_wp, kind=kind(ctx%state%speclib))
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        call assert_relative_error(300.0_wp * scale_factor, real(spec(1), WP), 1.0e-6_wp, &
                                   "Legacy corner order: (j+1,k+1) over (j,k)", total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_partial_overlap_snap

    subroutine test_agb_o_stack_allocation()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: expected
        integer :: nlam, i

        call print_group("AGB O Stack Allocation Stability")

        nlam = 6
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 3.4_wp
        lbol = 1.0_wp
        logg = 0.0_wp
        phase = 5.0_wp
        ffco = 0.8_wp
        lmdot = -6.0_wp

        ctx%state%agb_spec_o = 3.0_wp

        expected = (ctx%state%spec_lambda(1)**2 / C_LIGHT) * 3.0_wp
        do i = 1, 100
            call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        end do
        call assert_relative_error(expected, real(spec(1), WP), 1.0e-6_wp, "Repeated calls stable", total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_agb_o_stack_allocation

    ! --------------------------------------------------------------------
    ! Group 4: Specialized Physics Modules
    ! --------------------------------------------------------------------
    subroutine test_wr_wind_temperature()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        integer :: nlam, j

        call print_group("WR Wind Temperature (MIST vs Padova)")

        nlam = 5
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 4.8_wp
        lbol = 1.0_wp
        logg = 4.0_wp
        phase = 9.0_wp
        ffco = 0.0_wp
        lmdot = -2.0_wp

        ctx%use_wr_spectra_val = 1
        do j = 1, NDIM_WR
            ctx%state%wrn_spec(:, j, 1) = ctx%state%wrn_logt(j)
        end do

        ctx%state%isoc_type = 'mist'
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        call assert_true(spec(1) < logt, "MIST wind temperature lower than photosphere", total_tests, total_failures)

        ctx%state%isoc_type = 'padova'
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        call assert_relative_error(logt, real(spec(1), WP), 1.0e-6_wp, &
                                   "Padova uses photospheric temperature", total_tests, total_failures)

        deallocate(spec)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_wr_wind_temperature

    subroutine test_agb_dust_screen()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec_raw(:), spec_dust(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: tau1
        integer :: nlam, i

        call print_group("AGB Dust Screen Applied After Spectrum")

        nlam = 10
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec_raw(nlam), spec_dust(nlam))

        pset%zmet = 1
        pset%agb_dust = 1.0_wp
        mact = 1.0_wp
        logt = 3.4_wp
        lbol = 1.0_wp
        logg = 0.5_wp
        phase = 5.0_wp
        ffco = 1.2_wp
        lmdot = -4.0_wp

        ctx%state%agb_spec_car = 1.0_wp

        do i = 1, nlam
            if (i <= nlam / 2) then
                ctx%state%flux_dagb(i, 2, :, :) = 0.5_wp
            else
                ctx%state%flux_dagb(i, 2, :, :) = 0.9_wp
            end if
        end do

        ctx%add_agb_dust_model_val = 0
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec_raw)

        ctx%add_agb_dust_model_val = 1
        ctx%use_isoc_mdot_val = 1
        tau1 = compute_circumstellar_optical_depth(ctx, 1, mact, log10(lbol), logg, lmdot)
        ctx%state%tau1_dagb(2, :) = [(log10(tau1) - 1.0_wp + 0.1_wp * real(i - 1, WP), i = 1, NTAU_DAGB)]
        ctx%state%teff_dagb(2, :) = [(10.0_wp**logt - 500.0_wp + 100.0_wp * real(i - 1, WP), i = 1, NTEFF_DAGB)]

        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec_dust)

        call assert_true(spec_dust(1) < spec_raw(1), "Short wavelength dimming", total_tests, total_failures)
        call assert_true(spec_dust(nlam) <= spec_raw(nlam), "Long wavelength not brighter", total_tests, total_failures)

        deallocate(spec_raw, spec_dust)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_agb_dust_screen

    ! --------------------------------------------------------------------
    ! Group 5: Optimization Verification
    ! --------------------------------------------------------------------
    subroutine test_scaled_output_identity()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(KP), allocatable :: spec_fused(:), spec_raw(:)
        real(WP) :: mact, logt, lbol, logg, phase, ffco, lmdot
        real(WP) :: max_diff
        integer :: nlam, i

        call print_group("Scaled Output Identity (Loop Fusion)")

        nlam = 8
        allocate(ctx)
        call setup_context(ctx, nlam)
        allocate(spec_fused(nlam), spec_raw(nlam))

        pset%zmet = 1
        mact = 1.0_wp
        logt = 4.8_wp
        logg = 4.0_wp
        phase = 0.0_wp
        ffco = 0.0_wp
        lmdot = -6.0_wp

        do i = 1, nlam
            ctx%state%wmb_spec(i, 1, :, :) = real(i, kind=kind(ctx%state%wmb_spec))
        end do

        lbol = 2.5_wp
        ctx%logt_wmb_hot_val = 4.0_wp
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec_fused)

        lbol = 1.0_wp
        call get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec_raw)

        spec_raw = spec_raw * 2.5_wp
        max_diff = maxval(abs(spec_fused - spec_raw))
        call assert_true(max_diff < 1.0e-6_wp, "Fused scale matches legacy", total_tests, total_failures)

        deallocate(spec_fused, spec_raw)
        call teardown_context(ctx)
        deallocate(ctx)
    end subroutine test_scaled_output_identity

    ! --------------------------------------------------------------------
    ! Helper routines
    ! --------------------------------------------------------------------
    subroutine setup_context(ctx, nlam)
        type(fsps_context_t), intent(out) :: ctx
        integer, intent(in) :: nlam
        integer :: i

        allocate(ctx%state%spec_lambda(nlam))
        allocate(ctx%state%spec_nu(nlam))
        allocate(ctx%state%speclib(nlam, 1, NDIM_LOGT, NDIM_LOGG))
        allocate(ctx%state%wmb_spec(nlam, 1, NDIM_WMB_LOGT, NDIM_WMB_LOGG))
        allocate(ctx%state%agb_spec_o(nlam, N_AGB_O))
        allocate(ctx%state%agb_logt_o(1, N_AGB_O))
        allocate(ctx%state%agb_spec_c(nlam, N_AGB_C))
        allocate(ctx%state%agb_logt_c(N_AGB_C))
        allocate(ctx%state%agb_spec_car(nlam, N_AGB_CAR))
        allocate(ctx%state%pagb_spec(nlam, NDIM_PAGB, 2))
        allocate(ctx%state%wrn_spec(nlam, NDIM_WR, 1))
        allocate(ctx%state%wrc_spec(nlam, NDIM_WR, 1))
        allocate(ctx%state%flux_dagb(nlam, 2, NTEFF_DAGB, NTAU_DAGB))
        allocate(ctx%state%zlegend(1))

        do i = 1, nlam
            ctx%state%spec_lambda(i) = 1000.0_wp + 10.0_wp * real(i - 1, WP)
            ctx%state%spec_nu(i) = C_LIGHT / ctx%state%spec_lambda(i)
        end do

        call fill_linear_grid(ctx%state%speclib_logt, 3.0_wp, (5.0_wp - 3.0_wp) / real(NDIM_LOGT - 1, WP))
        call fill_linear_grid(ctx%state%speclib_logg, 0.0_wp, (6.0_wp - 0.0_wp) / real(NDIM_LOGG - 1, WP))

        call fill_linear_grid(ctx%state%wmb_logt, 4.0_wp, (5.0_wp - 4.0_wp) / real(NDIM_WMB_LOGT - 1, WP))
        call fill_linear_grid(ctx%state%wmb_logg, 3.0_wp, (5.0_wp - 3.0_wp) / real(NDIM_WMB_LOGG - 1, WP))

        call fill_linear_grid(ctx%state%pagb_logt, 4.6_wp, 0.05_wp)
        call fill_linear_grid(ctx%state%wrn_logt, 4.0_wp, (5.0_wp - 4.0_wp) / real(NDIM_WR - 1, WP))
        call fill_linear_grid(ctx%state%wrc_logt, 4.0_wp, (5.0_wp - 4.0_wp) / real(NDIM_WR - 1, WP))

        ctx%state%agb_logt_o = reshape([(3.2_wp + 0.05_wp * real(i - 1, WP), i = 1, N_AGB_O)], [1, N_AGB_O])
        ctx%state%agb_logt_c = [(3.2_wp + 0.05_wp * real(i - 1, WP), i = 1, N_AGB_C)]
        ctx%state%agb_logt_car = [(3.2_wp + 0.05_wp * real(i - 1, WP), i = 1, N_AGB_CAR)]

        ctx%state%speclib = real(0.0_wp, kind=kind(ctx%state%speclib))
        ctx%state%wmb_spec = real(0.0_wp, kind=kind(ctx%state%wmb_spec))
        ctx%state%agb_spec_o = SAFE_FLOOR
        ctx%state%agb_spec_c = SAFE_FLOOR
        ctx%state%agb_spec_car = SAFE_FLOOR
        ctx%state%pagb_spec = SAFE_FLOOR
        ctx%state%wrn_spec = SAFE_FLOOR
        ctx%state%wrc_spec = SAFE_FLOOR
        ctx%state%flux_dagb = 1.0_wp

        ctx%state%zlegend = 1.0_wp
        ctx%state%zsol = 1.0_wp
        ctx%state%whlam5000 = 1
        ctx%state%isoc_type = 'padova'
        ctx%logt_wmb_hot_val = 4.7_wp
        ctx%use_wr_spectra_val = 0
        ctx%add_agb_dust_model_val = 0
        ctx%use_isoc_mdot_val = 0
    end subroutine setup_context

    subroutine teardown_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%spec_nu)) deallocate(ctx%state%spec_nu)
        if (associated(ctx%state%speclib)) deallocate(ctx%state%speclib)
        if (associated(ctx%state%wmb_spec)) deallocate(ctx%state%wmb_spec)
        if (associated(ctx%state%agb_spec_o)) deallocate(ctx%state%agb_spec_o)
        if (associated(ctx%state%agb_logt_o)) deallocate(ctx%state%agb_logt_o)
        if (associated(ctx%state%agb_spec_c)) deallocate(ctx%state%agb_spec_c)
        if (associated(ctx%state%agb_logt_c)) deallocate(ctx%state%agb_logt_c)
        if (associated(ctx%state%agb_spec_car)) deallocate(ctx%state%agb_spec_car)
        if (associated(ctx%state%pagb_spec)) deallocate(ctx%state%pagb_spec)
        if (associated(ctx%state%wrn_spec)) deallocate(ctx%state%wrn_spec)
        if (associated(ctx%state%wrc_spec)) deallocate(ctx%state%wrc_spec)
        if (associated(ctx%state%flux_dagb)) deallocate(ctx%state%flux_dagb)
        if (associated(ctx%state%zlegend)) deallocate(ctx%state%zlegend)
    end subroutine teardown_context

    subroutine fill_linear_grid(arr, start_val, step)
        real(WP), dimension(:), intent(out) :: arr
        real(WP), intent(in) :: start_val, step
        integer :: i

        do i = 1, size(arr)
            arr(i) = start_val + step * real(i - 1, WP)
        end do
    end subroutine fill_linear_grid

    pure function compute_logg_calc(mact, lbol, logt, logg_in) result(logg_out)
        real(WP), intent(in) :: mact, lbol, logt, logg_in
        real(WP) :: logg_out

        if (lbol > tiny(0.0_wp)) then
            logg_out = log10(GRAVITY_L_M_T_COEFF * mact / lbol) + 4.0_wp * logt
        else
            logg_out = logg_in
        end if
    end function compute_logg_calc

    pure function compute_scale_factor_main(mact, lbol, logt, logg_in) result(scale_factor)
        real(WP), intent(in) :: mact, lbol, logt, logg_in
        real(WP) :: scale_factor
        real(WP) :: logg_out, gravity_cgs, r2

        logg_out = compute_logg_calc(mact, lbol, logt, logg_in)
        gravity_cgs = 10.0_wp**logg_out

        if (gravity_cgs > tiny(0.0_wp)) then
            r2 = (mact * M_SOL * G_NEWTON) / gravity_cgs
        else
            r2 = 0.0_wp
        end if

        scale_factor = (16.0_wp * PI * PI * r2) / L_SOL
    end function compute_scale_factor_main

    function integrate_trapezoid(x, y) result(total)
        real(WP), dimension(:), intent(in) :: x, y
        real(WP) :: total
        integer :: i

        total = 0.0_wp
        do i = 1, size(x) - 1
            total = total + 0.5_wp * (y(i) + y(i + 1)) * (x(i + 1) - x(i))
        end do
    end function integrate_trapezoid

    function reverse_array(arr) result(out)
        real(WP), dimension(:), intent(in) :: arr
        real(WP), dimension(size(arr)) :: out
        integer :: i, n

        n = size(arr)
        do i = 1, n
            out(i) = arr(n - i + 1)
        end do
    end function reverse_array

    pure function compute_lbol_for_logg(mact, logt, target_logg) result(lbol_req)
        real(WP), intent(in) :: mact, logt, target_logg
        real(WP) :: lbol_req
        real(WP) :: num, den

        ! Formula: logg = log10(Coeff * M / L) + 4*logT
        ! Invert for L:
        ! log10(L) = log10(Coeff * M) + 4*logT - logg
        
        num = GRAVITY_L_M_T_COEFF * mact
        den = 10.0_wp**(target_logg - 4.0_wp * logt)
        
        lbol_req = num / den
    end function compute_lbol_for_logg

end module test_fsps_spectral_library_mod