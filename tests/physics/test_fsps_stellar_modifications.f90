module test_fsps_stellar_modifications_mod
    use fsps_precision, only: WP
    use fsps_constants, only: NM, BHB_SBS_TIME
    use fsps_types, only: params
    use fsps_context_types, only: fsps_context_t
    use fsps_stellar_modifications
    use fsps_integration, only: integrate_romberg
    use fsps_imf, only: get_imf_value
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_int_equals, assert_relative_error
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    real(WP), parameter :: EPS = 1.0e-6_wp
    real(WP), parameter :: REL_EPS = 1.0e-6_wp

    public :: run_fsps_stellar_modifications_tests, total_failures, total_tests

contains

    subroutine run_fsps_stellar_modifications_tests()

        call print_minor_header("fsps_stellar_modifications")

        call test_bs_perfect_line_zams()
        call test_bs_cache_safety()
        call test_bs_distribution()
        call test_bs_zams_vs_current()
        call test_bs_zams_skips_pre_main_sequence()

        call test_gb_phase_selectivity()
        call test_gb_villaume_factor()
        call test_gb_conroy_gunn_lowz()

        call test_hb_cliff_padova()
        call test_hb_cliff_padova_duplicate_mass()
        call test_hb_mist_redistribution()

        call test_remnant_ghost_mass()
        call test_remnant_hierarchy()

        call test_xrb_bilinear_identity()
        call test_xrb_zero_fraction()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_stellar_modifications_tests

    ! --------------------------------------------------------------------
    ! GROUP 1: BLUE STRAGGLER TESTS
    ! --------------------------------------------------------------------
    subroutine test_bs_perfect_line_zams()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 2
        integer, parameter :: n_curr = 60
        integer, dimension(n_time) :: n_mass
        real(WP), allocatable :: mass_ini(:,:), mass_act(:,:), log_l(:,:), log_t(:,:), log_g(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        integer :: i
        real(WP) :: expected_logl

        call print_group("BS: Perfect Line ZAMS MSTO")

        allocate(ctx)
        call setup_zams_context(ctx)

        allocate(mass_ini(n_time, NM))
        allocate(mass_act(n_time, NM))
        allocate(log_l(n_time, NM))
        allocate(log_t(n_time, NM))
        allocate(log_g(n_time, NM))
        allocate(phase(n_time, NM))
        allocate(weights(NM))

        mass_ini = 0.0_wp
        mass_act = 0.0_wp
        log_l = 0.0_wp
        log_t = 0.0_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        ! Setup ZAMS: Steeper slope (0.1) so we hit L=3.5 at index 35 (safe within NM)
        do i = 1, NM
            log_t(1, i) = 0.1_wp * real(i, WP)
            log_l(1, i) = log_t(1, i)
            mass_ini(1, i) = log_l(1, i) + 1.0_wp
            mass_act(1, i) = mass_ini(1, i)
            
            ctx%state%logt_isoc(1, 1, i) = log_t(1, i)
            ctx%state%logl_isoc(1, 1, i) = log_l(1, i)
            ctx%state%mini_isoc(1, 1, i) = mass_ini(1, i)
        end do

        ! Setup Current Age: Same as ZAMS until i=50, then turn off
        do i = 1, n_curr
            log_t(2, i) = log_t(1, i)
            if (i <= 50) then
                log_l(2, i) = log_t(2, i)
            else
                ! Turn off: Brighter than ZAMS by 1.0 dex
                log_l(2, i) = log_t(2, i) + 1.0_wp
            end if
            mass_ini(2, i) = log_l(2, i) + 1.0_wp
            mass_act(2, i) = mass_ini(2, i)
            weights(i) = 1.0_wp
        end do

        n_mass = n_curr
        ctx%zin = 0

        ! MSTO should be detected at index 51. BS starts at index 50.
        ! log_l(50) = 0.1 * 50 = 5.0. 
        ! Wait, 5.0 > ZAMS_LUM_LIMIT (3.5). 
        ! The ZAMS cache cut-off will handle this, but MSTO detection logic 
        ! interpolates against the *cached* ZAMS. 
        ! If Current T=5.0, and Cache max T=3.5, interpolate_linear will extrapolate or use max.
        
        ! Let's adjust the turn off to be earlier to be safe.
        ! Turn off at 20.
        do i = 1, n_curr
            if (i <= 20) then
                log_l(2, i) = log_t(2, i)
            else
                log_l(2, i) = log_t(2, i) + 1.0_wp
            end if
        end do
        
        ! log_l(20) = 0.1 * 20 = 2.0.
        expected_logl = 2.0_wp + 0.2_wp + (0.75_wp * (1.0_wp/20.0_wp))

        call apply_blue_stragglers(ctx, 2, 1, 1.0_wp, 10.0_wp, n_mass, mass_ini, mass_act, log_l, log_t, log_g, phase, weights)

        call assert_int_equals(n_curr + 20, n_mass(2), "BS count updated", total_tests, total_failures)
        call assert_float_equals(expected_logl, log_l(2, n_curr + 1), EPS, "BS starts at MSTO-1", total_tests, total_failures)
        call assert_float_equals(7.0_wp, phase(2, n_curr + 1), EPS, "BS phase tag", total_tests, total_failures)

        call teardown_zams_context(ctx)
        deallocate(ctx)
        deallocate(mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
    end subroutine test_bs_perfect_line_zams

    subroutine test_bs_cache_safety()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 2
        integer, parameter :: n_curr = 40
        integer, dimension(n_time) :: n_mass
        real(WP), allocatable :: mass_ini(:,:), mass_act(:,:), log_l(:,:), log_t(:,:), log_g(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        integer :: i
        real(WP) :: logl_t1, logl_t2

        call print_group("BS: ZAMS Cache Safety")

        allocate(ctx)
        call setup_zams_context(ctx)

        allocate(mass_ini(n_time, NM))
        allocate(mass_act(n_time, NM))
        allocate(log_l(n_time, NM))
        allocate(log_t(n_time, NM))
        allocate(log_g(n_time, NM))
        allocate(phase(n_time, NM))
        allocate(weights(NM))

        mass_ini = 0.0_wp
        mass_act = 0.0_wp
        log_l = 0.0_wp
        log_t = 0.0_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        do i = 1, NM
            log_t(1, i) = 0.1_wp * real(i, WP) ! Steeper slope for safety
            log_l(1, i) = log_t(1, i)
            mass_ini(1, i) = log_l(1, i) + 0.5_wp
            mass_act(1, i) = mass_ini(1, i)
            
            ctx%state%logt_isoc(1, 1, i) = log_t(1, i)
            ctx%state%logl_isoc(1, 1, i) = log_l(1, i)
            ctx%state%mini_isoc(1, 1, i) = mass_ini(1, i)

            log_t(2, i) = log_t(1, i)
            log_l(2, i) = log_t(2, i)
            mass_ini(2, i) = log_l(2, i) + 0.5_wp
            mass_act(2, i) = mass_ini(2, i)
        end do

        ! Turn off at 20
        do i = 21, n_curr
             log_l(1, i) = log_t(1, i) + 1.0_wp
             log_l(2, i) = log_t(2, i) + 1.0_wp
        end do

        weights(1:n_curr) = 1.0_wp
        n_mass = n_curr
        ctx%zin = 0

        call apply_blue_stragglers(ctx, 1, 1, 1.0_wp, 2.0_wp, n_mass, mass_ini, mass_act, log_l, log_t, log_g, phase, weights)

        logl_t1 = log_l(1, n_curr + 1)
        call assert_int_equals(n_curr + 20, n_mass(1), "BS count time 1", total_tests, total_failures)

        ! Reset arrays but KEEP ZAMS context
        mass_ini = 0.0_wp
        mass_act = 0.0_wp
        log_l = 0.0_wp
        log_t = 0.0_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        ! Re-populate current age
        do i = 1, n_curr
            log_t(1, i) = 0.1_wp * real(i, WP)
            log_t(2, i) = log_t(1, i)
            if (i <= 20) then
                log_l(2, i) = log_t(2, i)
            else
                log_l(2, i) = log_t(2, i) + 1.0_wp
            end if
            mass_ini(2, i) = log_l(2, i) + 0.5_wp
            mass_act(2, i) = mass_ini(2, i)
        end do

        weights(1:n_curr) = 1.0_wp
        n_mass = n_curr
        call apply_blue_stragglers(ctx, 2, 1, 2.0_wp, 1.0_wp, n_mass, mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
        logl_t2 = log_l(2, n_curr + 1)

        call assert_float_equals(logl_t1, logl_t2, EPS, "Consistent BS logL across time", total_tests, total_failures)
        call assert_int_equals(n_curr + 20, n_mass(2), "BS count time 2", total_tests, total_failures)

        call teardown_zams_context(ctx)
        deallocate(ctx)
        deallocate(mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
    end subroutine test_bs_cache_safety

    subroutine test_bs_distribution()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 2
        integer, parameter :: n_curr = 55
        integer, dimension(n_time) :: n_mass
        real(WP), allocatable :: mass_ini(:,:), mass_act(:,:), log_l(:,:), log_t(:,:), log_g(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        integer :: i, k
        real(WP) :: expected_logl, expected_mass
        real(WP) :: inv_nbs
        character(len=64) :: msg

        call print_group("BS: Luminosity/Mass Distribution")

        allocate(ctx)
        call setup_zams_context(ctx)

        allocate(mass_ini(n_time, NM))
        allocate(mass_act(n_time, NM))
        allocate(log_l(n_time, NM))
        allocate(log_t(n_time, NM))
        allocate(log_g(n_time, NM))
        allocate(phase(n_time, NM))
        allocate(weights(NM))

        mass_ini = 0.0_wp
        mass_act = 0.0_wp
        log_l = 0.0_wp
        log_t = 0.0_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        do i = 1, NM
            log_t(1, i) = 0.1_wp * real(i, WP) ! Steeper slope
            log_l(1, i) = log_t(1, i)
            mass_ini(1, i) = log_l(1, i) + 1.0_wp
            mass_act(1, i) = mass_ini(1, i)
            
            ctx%state%logt_isoc(1, 1, i) = log_t(1, i)
            ctx%state%logl_isoc(1, 1, i) = log_l(1, i)
            ctx%state%mini_isoc(1, 1, i) = mass_ini(1, i)
        end do

        ! Turn off at 30
        do i = 1, n_curr
            log_t(2, i) = log_t(1, i)
            if (i <= 30) then
                log_l(2, i) = log_t(2, i)
            else
                log_l(2, i) = log_t(2, i) + 1.0_wp
            end if
            mass_ini(2, i) = log_l(2, i) + 1.0_wp
            mass_act(2, i) = mass_ini(2, i)
            weights(i) = 1.0_wp
        end do

        n_mass = n_curr
        ctx%zin = 0

        call apply_blue_stragglers(ctx, 2, 1, 1.0_wp, 1.0_wp, n_mass, mass_ini, mass_act, log_l, log_t, log_g, phase, weights)

        inv_nbs = 1.0_wp / 20.0_wp

        do k = 1, 20
            i = n_curr + k
            expected_logl = log_l(2, 30) + 0.2_wp + (0.75_wp * real(k, WP) * inv_nbs)
            expected_mass = expected_logl + 1.0_wp
            write(msg, '("BS logL distribution (iter ", I0, ")")') k
            call assert_float_equals(expected_logl, log_l(2, i), EPS, trim(msg), total_tests, total_failures)
            write(msg, '("BS mass from ZAMS (iter ", I0, ")")') k
            call assert_float_equals(expected_mass, mass_ini(2, i), EPS, trim(msg), total_tests, total_failures)
        end do

        call teardown_zams_context(ctx)
        deallocate(ctx)
        deallocate(mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
    end subroutine test_bs_distribution

    subroutine test_bs_zams_vs_current()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 2
        integer, parameter :: n_curr = 40
        integer, dimension(n_time) :: n_mass
        real(WP), allocatable :: mass_ini(:,:), mass_act(:,:), log_l(:,:), log_t(:,:), log_g(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        integer :: i

        call print_group("BS: ZAMS vs Current Distinction")

        allocate(ctx)
        call setup_zams_context(ctx)

        allocate(mass_ini(n_time, NM))
        allocate(mass_act(n_time, NM))
        allocate(log_l(n_time, NM))
        allocate(log_t(n_time, NM))
        allocate(log_g(n_time, NM))
        allocate(phase(n_time, NM))
        allocate(weights(NM))

        mass_ini = 0.0_wp
        mass_act = 0.0_wp
        log_l = 0.0_wp
        log_t = 0.0_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        ! FIX: ZAMS must start faint (<3.5) for the limit finder to work.
        do i = 1, NM
            log_t(1, i) = 3.0_wp + 0.01_wp * real(i, WP)
            log_l(1, i) = log_t(1, i) ! Crosses 3.5 around index 50
            mass_ini(1, i) = 1.0_wp + 0.001_wp * real(i, WP)
            mass_act(1, i) = mass_ini(1, i)
            
            ctx%state%logt_isoc(1, 1, i) = log_t(1, i)
            ctx%state%logl_isoc(1, 1, i) = log_l(1, i)
            ctx%state%mini_isoc(1, 1, i) = mass_ini(1, i)
        end do

        ! Current Age: Fainter, different slope
        do i = 1, n_curr
            log_t(2, i) = 3.0_wp + 0.01_wp * real(i, WP)
            if (i <= 30) then
                log_l(2, i) = log_t(2, i)
            else
                log_l(2, i) = log_t(2, i) + 1.0_wp
            end if
            mass_ini(2, i) = 1.0_wp + 0.001_wp * real(i, WP)
            mass_act(2, i) = mass_ini(2, i)
            weights(i) = 1.0_wp
        end do

        n_mass = n_curr
        ctx%zin = 0

        call apply_blue_stragglers(ctx, 2, 1, 1.0_wp, 2.0_wp, n_mass, mass_ini, mass_act, log_l, log_t, log_g, phase, weights)

        call assert_int_equals(n_curr + 20, n_mass(2), "BS added from ZAMS", total_tests, total_failures)
        
        ! LogT should be interpolated from ZAMS. 
        ! ZAMS relation is L=T.
        ! BS LogL range: 3.5 to 4.25.
        ! Current Age T relation: T = L - 1 (approx). If used, T would be ~2.5 to 3.25.
        ! So T > 3.4 proves ZAMS cache usage.
        call assert_true(log_t(2, n_curr + 1) > 3.4_wp, "BS logT uses ZAMS cache", total_tests, total_failures)

        call teardown_zams_context(ctx)
        deallocate(ctx)
        deallocate(mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
    end subroutine test_bs_zams_vs_current

    !> MIST-like isochrone set whose youngest isochrone is still on the pre-main sequence
    !> (phase -1) below some mass. The ZAMS reference must take those stars from the age at
    !> which they reach the MS, not from the first isochrone. (Using the first isochrone, the
    !> lowest-mass stars of an old isochrone already differ from the "ZAMS" by 1 dex, so no
    !> turn-off is found and sbss has no effect.)
    !>
    !> ZAMS line: log L = log T, M_ini = log L + 1, with log T = 0.05 i (i = 1..60).
    !> - Age 1: i <= 30 pre-MS (1 dex too bright), i > 30 on the MS.
    !> - Age 2: every point on the MS.
    !> - Age 3 (current): on the MS up to i = 20, 1 dex brighter beyond (turned off).
    subroutine test_bs_zams_skips_pre_main_sequence()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 3
        integer, parameter :: n_curr = 60
        integer, dimension(n_time) :: n_mass
        real(WP), allocatable :: mass_ini(:,:), mass_act(:,:), log_l(:,:), log_t(:,:), log_g(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        real(WP) :: expected_logl
        integer :: i, it

        call print_group("BS: ZAMS Skips Pre-Main Sequence")

        allocate(ctx)
        allocate(ctx%state%logl_isoc(1, n_time, NM), ctx%state%logt_isoc(1, n_time, NM), &
                 ctx%state%mini_isoc(1, n_time, NM), ctx%state%phase_isoc(1, n_time, NM), &
                 ctx%state%nmass_isoc(1, n_time))
        allocate(mass_ini(n_time, NM), mass_act(n_time, NM), log_l(n_time, NM), log_t(n_time, NM), &
                 log_g(n_time, NM), phase(n_time, NM), weights(NM))

        mass_ini = 0.0_wp
        mass_act = 0.0_wp
        log_l = 0.0_wp
        log_t = 0.0_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        do it = 1, n_time
            do i = 1, n_curr
                log_t(it, i) = 0.05_wp * real(i, WP)
                log_l(it, i) = log_t(it, i)
                mass_ini(it, i) = log_t(it, i) + 1.0_wp
                if (it == 1 .and. i <= 30) then
                    log_l(it, i) = log_l(it, i) + 1.0_wp
                    phase(it, i) = -1.0_wp
                end if
                if (it == 3 .and. i > 20) log_l(it, i) = log_l(it, i) + 1.0_wp
            end do
        end do
        mass_act = mass_ini
        weights(1:n_curr) = 1.0_wp
        n_mass = n_curr

        ctx%state%logl_isoc(1, :, :) = log_l
        ctx%state%logt_isoc(1, :, :) = log_t
        ctx%state%mini_isoc(1, :, :) = mass_ini
        ctx%state%phase_isoc(1, :, :) = phase
        ctx%state%nmass_isoc(1, :) = n_curr
        ctx%zin = 0

        call apply_blue_stragglers(ctx, 3, 1, 1.0_wp, 10.0_wp, n_mass, mass_ini, mass_act, &
                                   log_l, log_t, log_g, phase, weights)

        ! Turn-off at i = 20 (log L = 1.0); the first straggler sits 0.2 + 0.75/20 dex above it
        expected_logl = 1.0_wp + 0.2_wp + 0.75_wp / 20.0_wp
        call assert_int_equals(n_curr + 20, n_mass(3), "stragglers appended", total_tests, total_failures)
        call assert_float_equals(expected_logl, log_l(3, n_curr + 1), EPS, "first straggler log L", &
                                 total_tests, total_failures)
        call assert_float_equals(expected_logl + 1.0_wp, mass_ini(3, n_curr + 1), EPS, &
                                 "first straggler mass from ZAMS", total_tests, total_failures)
        call assert_float_equals(expected_logl, log_t(3, n_curr + 1), EPS, &
                                 "first straggler log T from ZAMS", total_tests, total_failures)

        deallocate(ctx%state%logl_isoc, ctx%state%logt_isoc, ctx%state%mini_isoc, ctx%state%phase_isoc, &
                   ctx%state%nmass_isoc)
        deallocate(ctx)
        deallocate(mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
    end subroutine test_bs_zams_skips_pre_main_sequence

    ! --------------------------------------------------------------------
    ! GROUP 2: GIANT BRANCH TESTS
    ! --------------------------------------------------------------------
    subroutine test_gb_phase_selectivity()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 1
        integer, parameter :: n_stars = 5
        real(WP), allocatable :: log_l(:,:), log_t(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        call print_group("GB: Phase Selectivity Mask")

        allocate(ctx)

        allocate(log_l(n_time, n_stars))
        allocate(log_t(n_time, n_stars))
        allocate(phase(n_time, n_stars))
        allocate(weights(n_stars))

        log_l = 0.0_wp
        log_t = 0.0_wp

        phase(1, :) = [1.0_wp, 3.0_wp, 5.0_wp, 6.0_wp, 7.0_wp]
        weights = 1.0_wp

        ctx%state%isoc_type = 'pdva'
        ctx%tpagb_norm_type_val = 0

        call modify_giant_branch(ctx, 1, 1, 9.0_wp, n_stars, 0.0_wp, 0.0_wp, 3.0_wp, 4.0_wp, 2.0_wp, &
                                 log_l, log_t, phase, weights)

        call assert_float_equals(1.0_wp, weights(1), EPS, "MS weight unchanged", total_tests, total_failures)
        call assert_float_equals(4.0_wp, weights(2), EPS, "HB weight scaled by RedGB", total_tests, total_failures)
        call assert_float_equals(8.0_wp, weights(3), EPS, "TP-AGB weight scaled by AGB*RedGB", total_tests, total_failures)
        call assert_float_equals(3.0_wp, weights(4), EPS, "Post-AGB weight scaled", total_tests, total_failures)
        call assert_float_equals(1.0_wp, weights(5), EPS, "BS weight unchanged", total_tests, total_failures)

        deallocate(ctx)
        deallocate(log_l, log_t, phase, weights)
    end subroutine test_gb_phase_selectivity

    subroutine test_gb_villaume_factor()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 1
        integer, parameter :: n_stars = 1
        real(WP), allocatable :: log_l(:,:), log_t(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        real(WP) :: expected_factor

        call print_group("GB: Villaume Factor Pre-calculation")

        allocate(ctx)

        allocate(log_l(n_time, n_stars))
        allocate(log_t(n_time, n_stars))
        allocate(phase(n_time, n_stars))
        allocate(weights(n_stars))

        log_l = 0.0_wp
        log_t = 0.0_wp
        phase(1, 1) = 5.0_wp
        weights(1) = 1.0_wp

        ctx%state%isoc_type = 'pdva'
        ctx%tpagb_norm_type_val = 2

        expected_factor = max(0.1_wp, 10.0_wp**(-1.0_wp + (10.5_wp - 8.0_wp)/2.5_wp))

        call modify_giant_branch(ctx, 1, 1, 10.5_wp, n_stars, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, &
                                 log_l, log_t, phase, weights)

        call assert_relative_error(expected_factor, weights(1), REL_EPS, "Villaume TP-AGB scaling", total_tests, total_failures)

        deallocate(ctx)
        deallocate(log_l, log_t, phase, weights)
    end subroutine test_gb_villaume_factor

    subroutine test_gb_conroy_gunn_lowz()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 1
        integer, parameter :: n_stars = 1
        real(WP), allocatable :: log_l(:,:), log_t(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        real(WP), allocatable, target :: zlegend(:)
        real(WP) :: expected_logl, expected_logt

        call print_group("GB: Conroy & Gunn Low-Z Shifts")

        allocate(ctx)

        allocate(log_l(n_time, n_stars))
        allocate(log_t(n_time, n_stars))
        allocate(phase(n_time, n_stars))
        allocate(weights(n_stars))
        allocate(zlegend(1))

        log_l = 1.0_wp
        log_t = 3.7_wp
        phase(1, 1) = 5.0_wp
        weights(1) = 1.0_wp

        zlegend(1) = 0.01_wp
        ctx%state%zsol = 0.02_wp
        ctx%state%zlegend => zlegend
        ctx%state%isoc_type = 'pdva'
        ctx%tpagb_norm_type_val = 1

        expected_logl = log_l(1, 1) + (-1.0_wp + (8.5_wp - 8.0_wp) / 1.5_wp)
        expected_logt = log_t(1, 1) + 0.10_wp

        call modify_giant_branch(ctx, 1, 1, 8.5_wp, n_stars, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, &
                                 log_l, log_t, phase, weights)

        call assert_float_equals(expected_logl, log_l(1, 1), EPS, "Conroy-Gunn logL shift", total_tests, total_failures)
        call assert_float_equals(expected_logt, log_t(1, 1), EPS, "Conroy-Gunn logT shift", total_tests, total_failures)

        deallocate(ctx)
        deallocate(log_l, log_t, phase, weights)
        deallocate(zlegend)
    end subroutine test_gb_conroy_gunn_lowz

    ! --------------------------------------------------------------------
    ! GROUP 3: HORIZONTAL BRANCH TESTS
    ! --------------------------------------------------------------------
    subroutine test_hb_cliff_padova()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 1
        integer, parameter :: n_curr = 3
        integer, dimension(n_time) :: n_mass
        real(WP), allocatable :: mass_ini(:,:), mass_act(:,:), log_l(:,:), log_t(:,:), log_g(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        real(WP) :: hb_total_weight

        call print_group("HB: Padova Cliff Gradient Detection")

        allocate(ctx)

        allocate(mass_ini(n_time, NM))
        allocate(mass_act(n_time, NM))
        allocate(log_l(n_time, NM))
        allocate(log_t(n_time, NM))
        allocate(log_g(n_time, NM))
        allocate(phase(n_time, NM))
        allocate(weights(NM))

        mass_ini = 0.0_wp
        mass_act = 0.0_wp
        log_l = 0.0_wp
        log_t = 0.0_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        mass_ini(1, 1) = 1.0_wp
        mass_ini(1, 2) = 1.001_wp
        mass_ini(1, 3) = 1.002_wp
        mass_act(1, 1:3) = mass_ini(1, 1:3)

        log_l(1, 1) = 3.0_wp
        log_l(1, 2) = 2.0_wp
        log_l(1, 3) = 2.2_wp
        log_t(1, 1:3) = 3.7_wp
        weights(1:3) = [1.0_wp, 2.0_wp, 1.0_wp]

        n_mass(1) = n_curr
        ctx%state%isoc_type = 'pdva'

        call modify_horizontal_branch(ctx, 1, 0.0_wp, 10.0_wp, hb_total_weight, n_mass, mass_ini, mass_act, &
                                      log_l, log_t, log_g, phase, weights)

        call assert_float_equals(2.0_wp, hb_total_weight, EPS, "HB weight detected", total_tests, total_failures)
        call assert_int_equals(n_curr, n_mass(1), "HB count unchanged for f_bhb=0", total_tests, total_failures)

        deallocate(ctx)
        deallocate(mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
    end subroutine test_hb_cliff_padova

    !> The Padova tables repeat the tip-of-RGB initial mass at the drop onto the HB, so the
    !> cliff has dM_ini = 0 (467 of the 753 Padova isochrones with an HB). Legacy MOD_HB
    !> divides by zero there, giving a -Inf gradient that passes the cliff test, and stops
    !> looking for the HB once it has left it. Points 2-4 are the HB; the later 3.0 -> 2.9 dip
    !> (an AGB glitch) must not be taken for a second HB.
    subroutine test_hb_cliff_padova_duplicate_mass()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 1
        integer, parameter :: n_curr = 6
        integer, dimension(n_time) :: n_mass
        real(WP), allocatable :: mass_ini(:,:), mass_act(:,:), log_l(:,:), log_t(:,:), log_g(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        real(WP) :: hb_total_weight

        call print_group("HB: Padova Cliff at Repeated Mass")

        allocate(ctx)
        allocate(mass_ini(n_time, NM), mass_act(n_time, NM), log_l(n_time, NM), log_t(n_time, NM), &
                 log_g(n_time, NM), phase(n_time, NM), weights(NM))

        mass_ini = 0.0_wp
        log_l = 0.0_wp
        log_t = 3.7_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        mass_ini(1, 1:n_curr) = [1.0_wp, 1.0_wp, 1.001_wp, 1.002_wp, 1.0021_wp, 1.00211_wp]
        mass_act = mass_ini
        log_l(1, 1:n_curr) = [3.3_wp, 1.6_wp, 1.62_wp, 1.65_wp, 3.0_wp, 2.9_wp]
        weights(1:n_curr) = [1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp]

        n_mass(1) = n_curr
        ctx%state%isoc_type = 'pdva'

        call modify_horizontal_branch(ctx, 1, 0.0_wp, 10.0_wp, hb_total_weight, n_mass, mass_ini, mass_act, &
                                      log_l, log_t, log_g, phase, weights)

        call assert_float_equals(9.0_wp, hb_total_weight, EPS, "HB found at repeated mass, later dip ignored", &
                                 total_tests, total_failures)

        deallocate(ctx)
        deallocate(mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
    end subroutine test_hb_cliff_padova_duplicate_mass

    subroutine test_hb_mist_redistribution()
        type(fsps_context_t), allocatable :: ctx
        integer, parameter :: n_time = 1
        integer, parameter :: n_curr = 5
        integer, dimension(n_time) :: n_mass
        real(WP), allocatable :: mass_ini(:,:), mass_act(:,:), log_l(:,:), log_t(:,:), log_g(:,:), phase(:,:)
        real(WP), allocatable :: weights(:)
        real(WP) :: hb_total_weight
        real(WP), dimension(n_curr) :: weights_init
        integer :: i, j
        real(WP) :: min_teff, expected_logt
        real(WP) :: inv_nhb
        character(len=64) :: msg

        call print_group("HB: MIST Phase Redistribution")

        allocate(ctx)

        allocate(mass_ini(n_time, NM))
        allocate(mass_act(n_time, NM))
        allocate(log_l(n_time, NM))
        allocate(log_t(n_time, NM))
        allocate(log_g(n_time, NM))
        allocate(phase(n_time, NM))
        allocate(weights(NM))

        mass_ini = 0.0_wp
        mass_act = 0.0_wp
        log_l = 0.0_wp
        log_t = 0.0_wp
        log_g = 0.0_wp
        phase = 0.0_wp
        weights = 0.0_wp

        do i = 1, n_curr
            mass_ini(1, i) = 1.0_wp + 0.1_wp * real(i - 1, WP)
            mass_act(1, i) = mass_ini(1, i)
            log_l(1, i) = 1.0_wp
            phase(1, i) = 3.0_wp
        end do

        log_t(1, 1:n_curr) = [3.8_wp, 3.7_wp, 3.6_wp, 3.65_wp, 3.75_wp]
        weights(1:n_curr) = 1.0_wp
        weights_init = weights(1:n_curr)

        n_mass(1) = n_curr
        ctx%state%isoc_type = 'mist'

        call modify_horizontal_branch(ctx, 1, 0.4_wp, BHB_SBS_TIME + 0.1_wp, hb_total_weight, n_mass, mass_ini, mass_act, &
                                      log_l, log_t, log_g, phase, weights)

        call assert_int_equals(n_curr + n_curr, n_mass(1), "MIST HB star count doubled", total_tests, total_failures)

        min_teff = 3.6_wp
        call assert_true(all(abs(log_t(1, 1:n_curr) - min_teff) <= EPS), &
                         "Red clump forced to min Teff", total_tests, total_failures)

        inv_nhb = 1.0_wp / real(n_curr, WP)
        do j = 1, n_curr
            expected_logt = min_teff + (4.5_wp - min_teff) * real(j, WP) * inv_nhb
            write(msg, '("Blue HB Teff distribution (iter ", I0, ")")') j
            call assert_float_equals(expected_logt, log_t(1, n_curr + j), EPS, trim(msg), total_tests, total_failures)
            write(msg, '("Blue HB weight (iter ", I0, ")")') j
            call assert_float_equals(0.4_wp * weights_init(j), weights(n_curr + j), EPS, trim(msg), total_tests, total_failures)
            write(msg, '("Red HB weight (iter ", I0, ")")') j
            call assert_float_equals(0.6_wp * weights_init(j), weights(j), EPS, trim(msg), total_tests, total_failures)
        end do

        deallocate(ctx)
        deallocate(mass_ini, mass_act, log_l, log_t, log_g, phase, weights)
    end subroutine test_hb_mist_redistribution

    ! --------------------------------------------------------------------
    ! GROUP 4: REMNANT MASS TESTS
    ! --------------------------------------------------------------------
    subroutine test_remnant_ghost_mass()
        type(fsps_context_t), allocatable :: ctx
        real(WP) :: current_mass
        real(WP) :: expected_mass

        call print_group("Remnants: Ghost Mass Fix")

        allocate(ctx)

        call setup_imf_context(ctx, 0.1_wp, 10.0_wp, 40.0_wp, 10.0_wp)

        current_mass = 0.0_wp
        call add_remnant_mass(ctx, current_mass, 9.0_wp)

        expected_mass = compute_expected_remnant_mass(ctx, 9.0_wp)

        call assert_true(current_mass > 0.0_wp, "WD integration executed", total_tests, total_failures)
        call assert_relative_error(expected_mass, current_mass, REL_EPS, "WD mass range [9,10]", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_remnant_ghost_mass

    subroutine test_remnant_hierarchy()
        type(fsps_context_t), allocatable :: ctx
        real(WP) :: current_mass
        real(WP) :: expected_mass

        call print_group("Remnants: Hierarchy of BH/NS/WD")

        allocate(ctx)

        call setup_imf_context(ctx, 0.1_wp, 120.0_wp, 40.0_wp, 8.5_wp)

        current_mass = 0.0_wp
        call add_remnant_mass(ctx, current_mass, 50.0_wp)
        expected_mass = compute_expected_remnant_mass(ctx, 50.0_wp)
        call assert_relative_error(expected_mass, current_mass, REL_EPS, "BH-only remnant", total_tests, total_failures)

        current_mass = 0.0_wp
        call add_remnant_mass(ctx, current_mass, 20.0_wp)
        expected_mass = compute_expected_remnant_mass(ctx, 20.0_wp)
        call assert_relative_error(expected_mass, current_mass, REL_EPS, "BH+NS remnant", total_tests, total_failures)

        current_mass = 0.0_wp
        call add_remnant_mass(ctx, current_mass, 1.0_wp)
        expected_mass = compute_expected_remnant_mass(ctx, 1.0_wp)
        call assert_relative_error(expected_mass, current_mass, REL_EPS, "BH+NS+WD remnant", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_remnant_hierarchy

    ! --------------------------------------------------------------------
    ! GROUP 5: XRB TESTS
    ! --------------------------------------------------------------------
    subroutine test_xrb_bilinear_identity()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: spec_in(:,:), spec_out(:,:)
        integer, parameter :: n_wave = 3
        integer, parameter :: n_time = 1
        integer, parameter :: n_age = 2
        integer, parameter :: n_z = 2
        real(WP), allocatable, target :: grid_wpec(:,:,:)
        real(WP), allocatable, target :: ages(:), zgrid(:), time_full(:), zlegend(:)

        call print_group("XRB: Bilinear Identity")

        allocate(ctx)

        allocate(spec_in(n_wave, n_time))
        allocate(spec_out(n_wave, n_time))
        allocate(grid_wpec(n_wave, n_age, n_z))
        allocate(ages(n_age))
        allocate(zgrid(n_z))
        allocate(time_full(n_time))
        allocate(zlegend(1))

        spec_in = 0.0_wp
        spec_out = 0.0_wp

        grid_wpec(:, :, :) = 0.0_wp
        grid_wpec(:, 1, 1) = [1.0_wp, 2.0_wp, 3.0_wp]
        grid_wpec(:, 2, 1) = [4.0_wp, 5.0_wp, 6.0_wp]
        grid_wpec(:, 1, 2) = [7.0_wp, 8.0_wp, 9.0_wp]
        grid_wpec(:, 2, 2) = [10.0_wp, 11.0_wp, 12.0_wp]

        ages = [1.0_wp, 2.0_wp]
        time_full = [1.0_wp]
        zlegend(1) = 0.02_wp
        zgrid = [0.0_wp, 0.5_wp]

        ctx%state%spec_xrb => grid_wpec
        ctx%state%ages_xrb => ages
        ctx%state%zmet_xrb => zgrid
        ctx%state%time_full => time_full
        ctx%state%zlegend => zlegend
        ctx%state%zsol = 0.02_wp
        ctx%state%nt = n_time
        ctx%state%nt_xrb = n_age
        ctx%state%nz_xrb = n_z

        pset%zmet = 1
        pset%frac_xrb = 1.0_wp

        call add_xray_binaries(ctx, pset, spec_in, spec_out)

        call assert_true(all(abs(spec_out(:, 1) - grid_wpec(:, 1, 1)) <= EPS), &
                         "Exact grid node interpolation", total_tests, total_failures)

        deallocate(ctx)
        deallocate(spec_in, spec_out, grid_wpec, ages, zgrid, time_full, zlegend)
    end subroutine test_xrb_bilinear_identity

    subroutine test_xrb_zero_fraction()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: spec_in(:,:), spec_out(:,:)
        integer, parameter :: n_wave = 2
        integer, parameter :: n_time = 1
        integer, parameter :: n_age = 2
        integer, parameter :: n_z = 2
        real(WP), allocatable, target :: grid_wpec(:,:,:)
        real(WP), allocatable, target :: ages(:), zgrid(:), time_full(:), zlegend(:)

        call print_group("XRB: Zero Fraction Shortcut")

        allocate(ctx)

        allocate(spec_in(n_wave, n_time))
        allocate(spec_out(n_wave, n_time))
        allocate(grid_wpec(n_wave, n_age, n_z))
        allocate(ages(n_age))
        allocate(zgrid(n_z))
        allocate(time_full(n_time))
        allocate(zlegend(1))

        spec_in(:, 1) = [1.1_wp, 2.2_wp]
        spec_out = 0.0_wp
        grid_wpec = 5.0_wp

        ages = [1.0_wp, 2.0_wp]
        time_full = [1.0_wp]
        zlegend(1) = 0.02_wp
        zgrid = [0.0_wp, 0.5_wp]

        ctx%state%spec_xrb => grid_wpec
        ctx%state%ages_xrb => ages
        ctx%state%zmet_xrb => zgrid
        ctx%state%time_full => time_full
        ctx%state%zlegend => zlegend
        ctx%state%zsol = 0.02_wp
        ctx%state%nt = n_time
        ctx%state%nt_xrb = n_age
        ctx%state%nz_xrb = n_z

        pset%zmet = 1
        pset%frac_xrb = 0.0_wp

        call add_xray_binaries(ctx, pset, spec_in, spec_out)

        call assert_true(all(spec_out == spec_in), "Spec unchanged for frac_xrb=0", total_tests, total_failures)

        deallocate(ctx)
        deallocate(spec_in, spec_out, grid_wpec, ages, zgrid, time_full, zlegend)
    end subroutine test_xrb_zero_fraction

    ! --------------------------------------------------------------------
    ! HELPERS
    ! --------------------------------------------------------------------
    subroutine setup_imf_context(ctx, lower_limit, upper_limit, mlim_bh, mlim_ns)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), intent(in) :: lower_limit, upper_limit, mlim_bh, mlim_ns

        ctx%imf_type_val = 0
        ctx%state%salp_ind = 2.35_wp
        ctx%state%imf_lower_limit = lower_limit
        ctx%state%imf_upper_limit = upper_limit
        ctx%state%mlim_bh = mlim_bh
        ctx%state%mlim_ns = mlim_ns
    end subroutine setup_imf_context

    pure function compute_expected_remnant_mass(ctx, max_living_mass) result(rem_mass)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: max_living_mass
        real(WP) :: rem_mass

        real(WP) :: imf_norm
        real(WP) :: integration_min, integration_max
        real(WP) :: term_bh, term_ns, term_wd_const, term_wd_linear
        real(WP) :: limit_low, limit_high, limit_bh, limit_ns

        limit_low = ctx%state%imf_lower_limit
        limit_high = ctx%state%imf_upper_limit
        limit_bh = ctx%state%mlim_bh
        limit_ns = ctx%state%mlim_ns

        rem_mass = 0.0_wp

        imf_norm = integrate_romberg(ctx, wrapper_imf_mass, limit_low, limit_high)
        if (imf_norm <= 0.0_wp) return

        integration_min = min(max(limit_bh, max_living_mass), limit_high)
        if (integration_min < limit_high) then
            term_bh = integrate_romberg(ctx, wrapper_imf_mass, integration_min, limit_high)
            rem_mass = rem_mass + (0.5_wp * term_bh / imf_norm)
        end if

        if (max_living_mass <= limit_bh) then
            integration_min = min(max(limit_ns, max_living_mass), limit_high)
            integration_max = min(limit_bh, limit_high)

            if (integration_min < integration_max) then
                term_ns = integrate_romberg(ctx, wrapper_imf_number, integration_min, integration_max)
                rem_mass = rem_mass + (1.4_wp * term_ns / imf_norm)
            end if
        end if

        if (max_living_mass <= limit_ns) then
            integration_min = min(max_living_mass, limit_high)
            integration_max = min(limit_ns, limit_high)

            if (integration_min < integration_max) then
                term_wd_const = integrate_romberg(ctx, wrapper_imf_number, integration_min, integration_max)
                term_wd_linear = integrate_romberg(ctx, wrapper_imf_mass, integration_min, integration_max)

                rem_mass = rem_mass + ((0.48_wp * term_wd_const) / imf_norm) + ((0.077_wp * term_wd_linear) / imf_norm)
            end if
        end if

    end function compute_expected_remnant_mass

    pure function wrapper_imf_number(ctx, x) result(res)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: x
        real(WP) :: res

        res = get_imf_value(ctx, x, mass_weighted=.false.)
    end function wrapper_imf_number

    pure function wrapper_imf_mass(ctx, x) result(res)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: x
        real(WP) :: res

        res = get_imf_value(ctx, x, mass_weighted=.true.)
    end function wrapper_imf_mass

    subroutine setup_zams_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        
        ! Allocate minimal state for ZAMS lookups (Metal 1, Time 1)
        if (.not. associated(ctx%state%logl_isoc)) allocate(ctx%state%logl_isoc(1, 1, NM))
        if (.not. associated(ctx%state%logt_isoc)) allocate(ctx%state%logt_isoc(1, 1, NM))
        if (.not. associated(ctx%state%mini_isoc)) allocate(ctx%state%mini_isoc(1, 1, NM))
        if (.not. associated(ctx%state%phase_isoc)) allocate(ctx%state%phase_isoc(1, 1, NM))
        if (.not. associated(ctx%state%nmass_isoc)) allocate(ctx%state%nmass_isoc(1, 1))
        
        ctx%state%logl_isoc = 0.0_wp
        ctx%state%logt_isoc = 0.0_wp
        ctx%state%mini_isoc = 0.0_wp
        ctx%state%phase_isoc = 0.0_wp
        ctx%state%nmass_isoc = NM
    end subroutine setup_zams_context

    subroutine teardown_zams_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        
        if (associated(ctx%state%logl_isoc)) deallocate(ctx%state%logl_isoc)
        if (associated(ctx%state%logt_isoc)) deallocate(ctx%state%logt_isoc)
        if (associated(ctx%state%mini_isoc)) deallocate(ctx%state%mini_isoc)
        if (associated(ctx%state%phase_isoc)) deallocate(ctx%state%phase_isoc)
        if (associated(ctx%state%nmass_isoc)) deallocate(ctx%state%nmass_isoc)
    end subroutine teardown_zams_context

end module test_fsps_stellar_modifications_mod
