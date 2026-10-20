module test_fsps_ssp_mod
    use fsps_precision, only: WP, KP
    use fsps_constants, only: NM, SAFE_FLOOR, TIME_RES_INCR, BHB_SBS_TIME, C_LIGHT
    use fsps_types, only: params
    use fsps_context_types, only: fsps_context_t
    use fsps_ssp, only: generate_ssp_grid, isochrone_buffer_t, init_isochrone_buffer, reset_buffer, &
                        free_isochrone_buffer, load_timestep_data, apply_isochrone_physics, &
                        compute_integrated_properties, accumulate_spectrum, interpolate_time_grid, &
                        configure_imf_parameters, compute_interpolated_ssp
    use fsps_spectral_library, only: get_stellar_spectrum
    use fsps_imf, only: compute_imf_weights
    use fsps_stellar_modifications, only: add_remnant_mass, modify_giant_branch
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_int_equals, assert_relative_error
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    real(WP), parameter :: EPS = 1.0e-10_wp
    real(WP), parameter :: REL_EPS = 1.0e-10_wp
    ! Relative rounding floor of the kernel precision KP (2.2e-14 for FP64, 1.2e-5 for FP32); tolerances on
    ! KP-computed quantities use max(tol, KP_RTOL) so the FP64 build keeps its original tolerance
    real(WP), parameter :: KP_RTOL = 100.0_wp * real(epsilon(1.0_KP), WP)

    public :: run_fsps_ssp_tests, total_failures, total_tests

contains

    subroutine run_fsps_ssp_tests()
        call print_minor_header("fsps_ssp")

        call test_strided_gather()
        call test_silent_truncation_guard()
        call test_fresh_start_zeroing()

        call test_cache_blocked_summation()
        call test_remnant_budgeting()
        call test_remnants_ignore_appended_stragglers()

        call test_sequential_hunter()
        call test_log_linear_interpolation()

        call test_bpass_short_circuit()
        call test_user_imf_file_parsing()

        call test_interp_point_bilinear()
        call test_interp_grid_z_only()
        call test_mdf_smoothing_kernel()
        call test_mdf_power_law()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_ssp_tests

    ! --------------------------------------------------------------------
    ! Helper: allocate minimal isochrone context
    ! --------------------------------------------------------------------
    subroutine setup_isochrone_context(ctx, n_z, n_t)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: n_z, n_t

        allocate(ctx%state%mini_isoc(n_z, n_t, NM))
        allocate(ctx%state%mact_isoc(n_z, n_t, NM))
        allocate(ctx%state%logl_isoc(n_z, n_t, NM))
        allocate(ctx%state%logt_isoc(n_z, n_t, NM))
        allocate(ctx%state%logg_isoc(n_z, n_t, NM))
        allocate(ctx%state%phase_isoc(n_z, n_t, NM))
        allocate(ctx%state%ffco_isoc(n_z, n_t, NM))
        allocate(ctx%state%lmdot_isoc(n_z, n_t, NM))
        allocate(ctx%state%nmass_isoc(n_z, n_t))
        allocate(ctx%state%timestep_isoc(n_z, n_t))
        allocate(ctx%state%zlegend(n_z))
        allocate(ctx%state%mass_ssp_zz(n_t, n_z))
        allocate(ctx%state%lbol_ssp_zz(n_t, n_z))
        allocate(ctx%state%spec_ssp_zz(1, n_t, n_z)) ! 1 wavelength for simplicity in tests
        allocate(ctx%state%time_full(n_t))

        if (allocated(ctx%state%ssp_temp_grid)) then
            !$acc exit data delete(ctx%state%ssp_temp_grid)
            deallocate(ctx%state%ssp_temp_grid)
        end if

        ctx%state%mini_isoc = 0.0_wp
        ctx%state%mact_isoc = 0.0_wp
        ctx%state%logl_isoc = 0.0_wp
        ctx%state%logt_isoc = 0.0_wp
        ctx%state%logg_isoc = 0.0_wp
        ctx%state%phase_isoc = 0.0_wp
        ctx%state%ffco_isoc = 0.0_wp
        ctx%state%lmdot_isoc = 0.0_wp
        ctx%state%nmass_isoc = 0
        ctx%state%timestep_isoc = 0.0_wp
        ctx%state%zlegend = 0.019_wp
        ctx%state%mass_ssp_zz = 0.0_wp
        ctx%state%lbol_ssp_zz = 0.0_wp
        ctx%state%spec_ssp_zz = 0.0_wp
        ctx%state%time_full = 0.0_wp
        ctx%state%ntfull = n_t
        ctx%state%nspec = 1

        ctx%state%nt = n_t
        ctx%state%nz = n_z
    end subroutine setup_isochrone_context

    subroutine teardown_isochrone_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (associated(ctx%state%mini_isoc)) deallocate(ctx%state%mini_isoc)
        if (associated(ctx%state%mact_isoc)) deallocate(ctx%state%mact_isoc)
        if (associated(ctx%state%logl_isoc)) deallocate(ctx%state%logl_isoc)
        if (associated(ctx%state%logt_isoc)) deallocate(ctx%state%logt_isoc)
        if (associated(ctx%state%logg_isoc)) deallocate(ctx%state%logg_isoc)
        if (associated(ctx%state%phase_isoc)) deallocate(ctx%state%phase_isoc)
        if (associated(ctx%state%ffco_isoc)) deallocate(ctx%state%ffco_isoc)
        if (associated(ctx%state%lmdot_isoc)) deallocate(ctx%state%lmdot_isoc)
        if (associated(ctx%state%nmass_isoc)) deallocate(ctx%state%nmass_isoc)
        if (associated(ctx%state%timestep_isoc)) deallocate(ctx%state%timestep_isoc)
        if (associated(ctx%state%zlegend)) deallocate(ctx%state%zlegend)
        if (allocated(ctx%state%mass_ssp_zz)) deallocate(ctx%state%mass_ssp_zz)
        if (allocated(ctx%state%lbol_ssp_zz)) deallocate(ctx%state%lbol_ssp_zz)
        if (allocated(ctx%state%spec_ssp_zz)) deallocate(ctx%state%spec_ssp_zz)
        if (associated(ctx%state%time_full)) deallocate(ctx%state%time_full)
    end subroutine teardown_isochrone_context

    ! --------------------------------------------------------------------
    ! Helper: allocate minimal spectral context for get_stellar_spectrum
    ! --------------------------------------------------------------------
    subroutine setup_spectral_context(ctx, n_wave, speclib_val)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: n_wave
        real(WP), intent(in) :: speclib_val

        integer :: i
        real(WP) :: logt_min, logt_step
        real(WP) :: logg_min, logg_step

        allocate(ctx%state%spec_lambda(n_wave))
        allocate(ctx%state%spec_nu(n_wave))
        allocate(ctx%state%speclib(n_wave, 1, size(ctx%state%speclib_logt), size(ctx%state%speclib_logg)))

        ctx%state%spec_lambda = [(1000.0_wp + 100.0_wp * real(i - 1, WP), i = 1, n_wave)]
        ctx%state%spec_nu = C_LIGHT / ctx%state%spec_lambda
        ctx%state%whlam5000 = 1

        logt_min = 3.0_wp
        logt_step = 0.02_wp
        do i = 1, size(ctx%state%speclib_logt)
            ctx%state%speclib_logt(i) = logt_min + logt_step * real(i - 1, WP)
        end do

        logg_min = 0.0_wp
        logg_step = 0.2_wp
        do i = 1, size(ctx%state%speclib_logg)
            ctx%state%speclib_logg(i) = logg_min + logg_step * real(i - 1, WP)
        end do

        ctx%state%speclib = real(speclib_val, kind=kind(1.0))

        ctx%use_wr_spectra_val = 0
        ctx%logt_wmb_hot_val = 10.0_wp
        ctx%state%wmb_logt = 10.0_wp
    end subroutine setup_spectral_context

    subroutine teardown_spectral_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%spec_nu)) deallocate(ctx%state%spec_nu)
        if (associated(ctx%state%speclib)) deallocate(ctx%state%speclib)

        if (allocated(ctx%state%ssp_temp_grid)) then
            !$acc exit data delete(ctx%state%ssp_temp_grid)
            deallocate(ctx%state%ssp_temp_grid)
        end if
    end subroutine teardown_spectral_context

    ! --------------------------------------------------------------------
    ! GROUP 1: Buffer Management & Memory Safety
    ! --------------------------------------------------------------------
    subroutine test_strided_gather()
        type(fsps_context_t), allocatable :: ctx
        type(isochrone_buffer_t) :: buf
        integer :: i

        call print_group("Buffer: Strided Gather")

        allocate(ctx)
        call setup_isochrone_context(ctx, 1, 6)
        call init_isochrone_buffer(buf)

        ctx%state%nmass_isoc(1, 5) = 10
        do i = 1, 10
            ctx%state%mini_isoc(1, 5, i) = real(i + 5000, WP)
            ctx%state%mact_isoc(1, 5, i) = 1.0_wp
            ctx%state%logl_isoc(1, 5, i) = 0.0_wp
            ctx%state%logt_isoc(1, 5, i) = 0.0_wp
            ctx%state%logg_isoc(1, 5, i) = 0.0_wp
            ctx%state%phase_isoc(1, 5, i) = 0.0_wp
            ctx%state%ffco_isoc(1, 5, i) = 0.0_wp
            ctx%state%lmdot_isoc(1, 5, i) = 0.0_wp
        end do

        call load_timestep_data(ctx, 1, 5, buf)

        do i = 1, 10
            call assert_float_equals(real(i + 5000, WP), buf%initial_mass(i), EPS, &
                                     "Strided gather idx " // trim(adjustl(to_str(i))), total_tests, total_failures)
        end do

        call free_isochrone_buffer(buf)
        call teardown_isochrone_context(ctx)
        deallocate(ctx)
    end subroutine test_strided_gather

    subroutine test_silent_truncation_guard()
        type(fsps_context_t), allocatable :: ctx
        type(isochrone_buffer_t) :: buf
        integer :: i

        call print_group("Buffer: Silent Truncation Guard")

        allocate(ctx)
        call setup_isochrone_context(ctx, 1, 1)
        call init_isochrone_buffer(buf)

        ctx%state%nmass_isoc(1, 1) = NM + 50
        do i = 1, NM
            ctx%state%mini_isoc(1, 1, i) = real(i, WP)
            ctx%state%mact_isoc(1, 1, i) = 1.0_wp
            ctx%state%logl_isoc(1, 1, i) = 0.0_wp
            ctx%state%logt_isoc(1, 1, i) = 0.0_wp
            ctx%state%logg_isoc(1, 1, i) = 0.0_wp
            ctx%state%phase_isoc(1, 1, i) = 0.0_wp
            ctx%state%ffco_isoc(1, 1, i) = 0.0_wp
            ctx%state%lmdot_isoc(1, 1, i) = 0.0_wp
        end do

        call load_timestep_data(ctx, 1, 1, buf)
        call assert_int_equals(NM, buf%n_stars, "Clamped to NM", total_tests, total_failures)

        call free_isochrone_buffer(buf)
        call teardown_isochrone_context(ctx)
        deallocate(ctx)
    end subroutine test_silent_truncation_guard

    subroutine test_fresh_start_zeroing()
        type(fsps_context_t), allocatable :: ctx
        type(isochrone_buffer_t) :: buf
        integer :: i

        call print_group("Buffer: Fresh Start Zeroing")

        allocate(ctx)
        call setup_isochrone_context(ctx, 1, 2)
        call init_isochrone_buffer(buf)

        ctx%state%nmass_isoc(1, 1) = 5
        ctx%state%nmass_isoc(1, 2) = 5
        do i = 1, 5
            ctx%state%mini_isoc(1, 1, i) = real(i, WP)
            ctx%state%mini_isoc(1, 2, i) = real(i, WP)
        end do

        call load_timestep_data(ctx, 1, 1, buf)
        buf%weights(1:buf%n_stars) = 1.0_wp

        call load_timestep_data(ctx, 1, 2, buf)
        do i = 1, buf%n_stars
            call assert_float_equals(0.0_wp, buf%weights(i), EPS, &
                                     "Weights reset idx " // trim(adjustl(to_str(i))), total_tests, total_failures)
        end do

        call free_isochrone_buffer(buf)
        call teardown_isochrone_context(ctx)
        deallocate(ctx)
    end subroutine test_fresh_start_zeroing

    ! --------------------------------------------------------------------
    ! GROUP 2: Integration & Spectral Accumulation
    ! --------------------------------------------------------------------

    subroutine test_cache_blocked_summation()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(isochrone_buffer_t) :: buf
        real(WP), allocatable :: spec_out(:)
        real(KP), allocatable :: ref_spec(:)
        real(WP) :: expected_sum
        integer :: i, n_wave

        call print_group("Accumulate: Cache-Blocked Summation")

        allocate(ctx)
        call setup_spectral_context(ctx, 4, 1.0e20_wp)

        n_wave = 4
        allocate(spec_out(n_wave))
        allocate(ref_spec(n_wave))
        spec_out = 0.0_wp

        pset%zmet = 1
        pset%evtype = -1
        pset%masscut = 1.0e9_wp

        call init_isochrone_buffer(buf)
        ! Use 65 stars to force 2 full batches (32+32) plus 1 leftover
        buf%n_stars = 65
        expected_sum = 0.0_wp
        
        do i = 1, buf%n_stars
            buf%initial_mass(i) = 1.0_wp
            buf%current_mass(i) = 1.0_wp
            buf%log_lum(i) = 0.0_wp
            buf%log_teff(i) = 4.0_wp
            buf%log_g(i) = 4.0_wp
            buf%phase(i) = 1.0_wp
            buf%co_ratio(i) = 0.0_wp
            buf%log_mdot(i) = 0.0_wp
            ! Assign varying weights to verify accumulation handles scaling
            buf%weights(i) = 0.01_wp * real(i, WP)
            expected_sum = expected_sum + buf%weights(i)
        end do

        ! Pre-calculate the spectrum for a single star (all stars are identical here)
        call get_stellar_spectrum(ctx, pset, 1.0_wp, 4.0_wp, 1.0_wp, 4.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, ref_spec)

        !$acc enter data copyin(ctx)
        !$acc enter data copyin(ctx%state)
        !$acc enter data copyin(ctx%state%spec_lambda, ctx%state%spec_nu, ctx%state%speclib)
        !$acc enter data attach(ctx%state%spec_lambda)
        !$acc enter data attach(ctx%state%spec_nu)
        !$acc enter data attach(ctx%state%speclib)

        !$acc enter data copyin(buf)
        !$acc enter data create(buf%initial_mass, buf%current_mass, buf%log_lum, buf%log_teff)
        !$acc enter data create(buf%log_g, buf%phase, buf%co_ratio, buf%log_mdot, buf%weights)
        !$acc enter data attach(buf%initial_mass)
        !$acc enter data attach(buf%current_mass)
        !$acc enter data attach(buf%log_lum)
        !$acc enter data attach(buf%log_teff)
        !$acc enter data attach(buf%log_g)
        !$acc enter data attach(buf%phase)
        !$acc enter data attach(buf%co_ratio)
        !$acc enter data attach(buf%log_mdot)
        !$acc enter data attach(buf%weights)
        !$acc update device(buf%weights, buf%current_mass, buf%log_lum, buf%initial_mass)
        !$acc update device(buf%log_teff, buf%log_g, buf%phase, buf%co_ratio, buf%log_mdot)

        ! Run accumulation (Unsorted)
        call accumulate_spectrum(ctx, pset, buf, spec_out)

        !$acc exit data delete(buf%initial_mass, buf%current_mass, buf%log_lum, buf%log_teff)
        !$acc exit data delete(buf%log_g, buf%phase, buf%co_ratio, buf%log_mdot, buf%weights)
        !$acc exit data delete(buf)

        !$acc exit data delete(ctx%state%spec_lambda, ctx%state%spec_nu, ctx%state%speclib)
        !$acc exit data delete(ctx%state)
        !$acc exit data delete(ctx)

        do i = 1, n_wave
            call assert_relative_error(ref_spec(i) * expected_sum, spec_out(i), max(1.0e-12_wp, KP_RTOL), &
                                       "Cache-block spec " // trim(adjustl(to_str(i))), total_tests, total_failures)
        end do

        deallocate(spec_out, ref_spec)
        call free_isochrone_buffer(buf)
        call teardown_spectral_context(ctx)
        deallocate(ctx)
    end subroutine test_cache_blocked_summation

    subroutine test_remnant_budgeting()
        type(fsps_context_t), allocatable :: ctx
        type(isochrone_buffer_t) :: buf
        real(WP) :: tot_mass, tot_lbol
        real(WP) :: expected_mass

        call print_group("Integrate: Remnant Budgeting")

        allocate(ctx)
        ctx%add_stellar_remnants_val = 1
        ctx%imf_type_val = 0
        ctx%state%imf_lower_limit = 0.1_wp
        ctx%state%imf_upper_limit = 100.0_wp
        ctx%state%imf_lower_bound = 0.08_wp

        call init_isochrone_buffer(buf)
        buf%n_stars = 3
        buf%initial_mass(1:3) = [0.5_wp, 0.6_wp, 0.8_wp]
        buf%current_mass(1:3) = [0.5_wp, 0.6_wp, 0.8_wp]
        buf%log_lum(1:3) = 0.0_wp
        buf%weights(1:3) = 1.0_wp

        expected_mass = sum(buf%weights(1:3) * buf%current_mass(1:3))
        call add_remnant_mass(ctx, expected_mass, maxval(buf%initial_mass(1:3)))

        !$acc enter data copyin(buf)
        !$acc enter data create(buf%initial_mass, buf%current_mass, buf%log_lum, buf%weights)
        !$acc enter data attach(buf%initial_mass)
        !$acc enter data attach(buf%current_mass)
        !$acc enter data attach(buf%log_lum)
        !$acc enter data attach(buf%weights)
        !$acc update device(buf%weights, buf%current_mass, buf%log_lum, buf%initial_mass)

        call compute_integrated_properties(ctx, buf, tot_mass, tot_lbol)

        !$acc exit data delete(buf%initial_mass, buf%current_mass, buf%log_lum, buf%weights)
        !$acc exit data delete(buf)
        call assert_relative_error(expected_mass, tot_mass, 1.0e-10_wp, &
                                   "Remnant mass added", total_tests, total_failures)

        call free_isochrone_buffer(buf)
        deallocate(ctx)
    end subroutine test_remnant_budgeting

    ! --------------------------------------------------------------------
    ! GROUP 3: Post-Processing & Interpolation
    ! --------------------------------------------------------------------
    subroutine test_sequential_hunter()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: mass_grid(:)
        real(WP), allocatable :: lbol_grid(:)
        real(WP), allocatable :: spec_grid(:,:)
        integer :: nt, nt_full

        call print_group("Interpolate: Sequential Hunter")

        if (TIME_RES_INCR /= 2) then
            call assert_true(.true., "TIME_RES_INCR /= 2 (skipped)", total_tests, total_failures)
            return
        end if

        allocate(ctx)
        nt = 4
        nt_full = 1 + (nt - 1) * TIME_RES_INCR

        ctx%state%nt = nt
        ctx%state%ntfull = nt_full

        allocate(ctx%state%timestep_isoc(1, nt))
        allocate(ctx%state%time_full(nt_full))

        ctx%state%timestep_isoc(1, :) = [6.0_wp, 7.0_wp, 8.0_wp, 9.0_wp]
        ctx%state%time_full = [6.0_wp, 6.5_wp, 7.0_wp, 7.5_wp, 8.0_wp, 8.5_wp, 9.0_wp]

        allocate(mass_grid(nt_full))
        allocate(lbol_grid(nt_full))
        allocate(spec_grid(1, nt_full))

        mass_grid = 0.0_wp
        lbol_grid = 0.0_wp
        spec_grid = 0.0_wp

        mass_grid(1) = 1.0_wp
        mass_grid(3) = 3.0_wp
        mass_grid(5) = 5.0_wp
        mass_grid(7) = 7.0_wp

        call interpolate_time_grid(ctx, 1, mass_grid, lbol_grid, spec_grid)
        call assert_float_equals(4.0_wp, mass_grid(4), 1.0e-12_wp, &
                                 "Sequential hunt midpoint", total_tests, total_failures)

        if (associated(ctx%state%timestep_isoc)) deallocate(ctx%state%timestep_isoc)
        if (associated(ctx%state%time_full)) deallocate(ctx%state%time_full)
        deallocate(mass_grid, lbol_grid, spec_grid)
        deallocate(ctx)
    end subroutine test_sequential_hunter

    subroutine test_log_linear_interpolation()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: mass_grid(:)
        real(WP), allocatable :: lbol_grid(:)
        real(WP), allocatable :: spec_grid(:,:)
        integer :: nt, nt_full

        call print_group("Interpolate: Log-Linear Spectra")

        if (TIME_RES_INCR /= 2) then
            call assert_true(.true., "TIME_RES_INCR /= 2 (skipped)", total_tests, total_failures)
            return
        end if

        allocate(ctx)
        nt = 2
        nt_full = 1 + (nt - 1) * TIME_RES_INCR

        ctx%state%nt = nt
        ctx%state%ntfull = nt_full

        allocate(ctx%state%timestep_isoc(1, nt))
        allocate(ctx%state%time_full(nt_full))

        ctx%state%timestep_isoc(1, :) = [1.0_wp, 2.0_wp]
        ctx%state%time_full = [1.0_wp, 1.5_wp, 2.0_wp]

        allocate(mass_grid(nt_full))
        allocate(lbol_grid(nt_full))
        allocate(spec_grid(1, nt_full))

        mass_grid = 0.0_wp
        lbol_grid = 0.0_wp
        spec_grid = 0.0_wp

        spec_grid(1, 1) = 10.0_wp
        spec_grid(1, 3) = 1000.0_wp

        call interpolate_time_grid(ctx, 1, mass_grid, lbol_grid, spec_grid)
        call assert_float_equals(100.0_wp, spec_grid(1, 2), 1.0e-10_wp, &
                                 "Log-linear midpoint", total_tests, total_failures)

        if (associated(ctx%state%timestep_isoc)) deallocate(ctx%state%timestep_isoc)
        if (associated(ctx%state%time_full)) deallocate(ctx%state%time_full)
        deallocate(mass_grid, lbol_grid, spec_grid)
        deallocate(ctx)
    end subroutine test_log_linear_interpolation

    ! --------------------------------------------------------------------
    ! GROUP 4: BPASS & Configuration
    ! --------------------------------------------------------------------
    subroutine test_bpass_short_circuit()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: mass_grid(:)
        real(WP), allocatable :: lbol_grid(:)
        real(WP), allocatable :: spec_grid(:,:)
        integer :: n_wave, n_time

        call print_group("BPASS: Short-Circuit")

        allocate(ctx)
        n_wave = 3
        n_time = 2
        allocate(ctx%state%bpass_spec_ssp(n_wave, n_time, 1))
        allocate(ctx%state%bpass_mass_ssp(n_time, 1))
        allocate(ctx%state%spec_nu(n_wave))
        ctx%state%spec_nu = [3.0_wp, 2.0_wp, 1.0_wp]

        ctx%state%check_sps_setup = 1
        ctx%state%nz = 1
        ctx%state%nt = n_time
        ctx%state%isoc_type = 'bpss'

        ctx%state%bpass_spec_ssp(:, :, 1) = reshape([1.0_wp, 2.0_wp, 3.0_wp, &
                                                    4.0_wp, 5.0_wp, 6.0_wp], &
                                                   [n_wave, n_time])
        ctx%state%bpass_mass_ssp(:, 1) = [2.5_wp, 3.5_wp]

        allocate(mass_grid(n_time))
        allocate(lbol_grid(n_time))
        allocate(spec_grid(n_wave, n_time))

        pset%zmet = 1

        call generate_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)

        call assert_float_equals(2.5_wp, mass_grid(1), EPS, "BPASS mass idx1", total_tests, total_failures)
        call assert_float_equals(3.5_wp, mass_grid(2), EPS, "BPASS mass idx2", total_tests, total_failures)
        call assert_float_equals(1.0_wp, spec_grid(1, 1), EPS, "BPASS spec (1,1)", total_tests, total_failures)
        call assert_float_equals(6.0_wp, spec_grid(3, 2), EPS, "BPASS spec (3,2)", total_tests, total_failures)
        ! L_bol is the frequency integral of the spectrum: (1+2)/2 + (2+3)/2 = 4 and (4+5)/2 + (5+6)/2 = 10
        call assert_float_equals(log10(4.0_wp), lbol_grid(1), EPS, "BPASS lbol from spectrum idx1", &
                                 total_tests, total_failures)
        call assert_float_equals(1.0_wp, lbol_grid(2), EPS, "BPASS lbol from spectrum idx2", total_tests, total_failures)

        deallocate(mass_grid, lbol_grid, spec_grid)
        if (associated(ctx%state%bpass_spec_ssp)) deallocate(ctx%state%bpass_spec_ssp)
        if (associated(ctx%state%bpass_mass_ssp)) deallocate(ctx%state%bpass_mass_ssp)
        if (associated(ctx%state%spec_nu)) deallocate(ctx%state%spec_nu)
        deallocate(ctx)
    end subroutine test_bpass_short_circuit

    subroutine test_user_imf_file_parsing()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        integer :: unit_imf
        character(len=256) :: filepath
        character(len=256) :: build_root
        character(len=256) :: data_dir
        integer :: env_len, env_status, cmd_status

        call print_group("IMF: User File Parsing")

        allocate(ctx)
        ctx%imf_type_val = 5
        pset%imf_filename = 'imf_test.dat'

        build_root = ''
        call get_environment_variable('MESON_BUILD_ROOT', build_root, env_len, env_status, .false.)
        if (env_status /= 0 .or. env_len == 0) then
            build_root = './tmp/test_fsps_ssp'
        else
            build_root = build_root(1:env_len)
        end if

        data_dir = trim(build_root) // '/data'
        call execute_command_line('mkdir -p ' // trim(data_dir), exitstat=cmd_status)

        ctx%sps_home = trim(build_root)
        filepath = trim(ctx%sps_home) // '/data/' // trim(pset%imf_filename)
        open(newunit=unit_imf, file=trim(filepath), status='replace', action='write')
        write(unit_imf, *) 0.1_wp, 1.0_wp, 1.3_wp
        write(unit_imf, *) 1.0_wp, 100.0_wp, 2.3_wp
        close(unit_imf)

        call configure_imf_parameters(ctx, pset)

        call assert_float_equals(0.1_wp, ctx%state%imf_user_alpha(1, 1), EPS, "IMF seg1 low", total_tests, total_failures)
        call assert_float_equals(1.0_wp, ctx%state%imf_user_alpha(2, 1), EPS, "IMF seg1 high", total_tests, total_failures)
        call assert_float_equals(1.3_wp, ctx%state%imf_user_alpha(3, 1), EPS, "IMF seg1 slope", total_tests, total_failures)
        call assert_float_equals(1.0_wp, ctx%state%imf_user_alpha(1, 2), EPS, "IMF seg2 low", total_tests, total_failures)
        call assert_float_equals(100.0_wp, ctx%state%imf_user_alpha(2, 2), EPS, "IMF seg2 high", total_tests, total_failures)
        call assert_float_equals(2.3_wp, ctx%state%imf_user_alpha(3, 2), EPS, "IMF seg2 slope", total_tests, total_failures)
        deallocate(ctx)
    end subroutine test_user_imf_file_parsing

    ! --------------------------------------------------------------------
    ! GROUP 5: Interpolated SSP Queries (compute_interpolated_ssp)
    ! --------------------------------------------------------------------

    subroutine test_interp_point_bilinear()
        !> Case 1: Specific Z and Specific T
        type(fsps_context_t), allocatable, target :: ctx
        real(WP) :: mass_out(1), lbol_out(1), spec_out(1,1)
        real(WP) :: z_target, t_target
        
        call print_group("Interp SSP: Point Bilinear")

        allocate(ctx)
        ! Setup 2x2 grid for manual verification
        ! Z = [-1.0, 0.0] (log solar)
        ! T = [6.0, 7.0] (log yr)
        call setup_isochrone_context(ctx, 2, 2)
        
        ! Mock Data: Mass = t_idx + 10 * z_idx
        ! (1,1)=11, (1,2)=12
        ! (2,1)=21, (2,2)=22
        ctx%state%mass_ssp_zz(1,1) = 11.0_wp; ctx%state%mass_ssp_zz(2,1) = 12.0_wp
        ctx%state%mass_ssp_zz(1,2) = 21.0_wp; ctx%state%mass_ssp_zz(2,2) = 22.0_wp
        
        ctx%state%time_full(1) = 6.0_wp; ctx%state%time_full(2) = 7.0_wp
        ctx%state%zlegend(1) = 0.019_wp * 0.1_wp; ctx%state%zlegend(2) = 0.019_wp * 1.0_wp
        ctx%state%zsol = 0.019_wp

        ! Target: Dead center (Z log=-0.5, T=6.5)
        ! Expected: Average of all 4 corners = (11+12+21+22)/4 = 16.5
        z_target = -0.5_wp
        t_target = 6.5_wp

        call compute_interpolated_ssp(ctx, z_target, mass_out, lbol_out, spec_out, t_pos=t_target)

        call assert_float_equals(16.5_wp, mass_out(1), EPS, &
                                 "Bilinear Mass Center", total_tests, total_failures)

        call teardown_isochrone_context(ctx)
        deallocate(ctx)
    end subroutine test_interp_point_bilinear

    subroutine test_interp_grid_z_only()
        !> Case 2: Specific Z, Full Time Grid
        type(fsps_context_t), allocatable, target :: ctx
        real(WP), allocatable :: mass_out(:), lbol_out(:), spec_out(:,:)
        real(WP) :: z_target
        integer :: nt
        
        call print_group("Interp SSP: Grid Z-Only")

        allocate(ctx)
        call setup_isochrone_context(ctx, 3, 2) ! 3 Metallicities, 2 Times
        nt = 2
        
        allocate(mass_out(nt), lbol_out(nt), spec_out(1, nt))

        ! Grid Z-values: -1.0, 0.0, +1.0
        ctx%state%zlegend(1) = 0.0019_wp
        ctx%state%zlegend(2) = 0.019_wp
        ctx%state%zlegend(3) = 0.19_wp
        ctx%state%zsol = 0.019_wp

        ! Set Mass to be purely Z-dependent
        ctx%state%mass_ssp_zz(:, 1) = 10.0_wp
        ctx%state%mass_ssp_zz(:, 2) = 20.0_wp
        ctx%state%mass_ssp_zz(:, 3) = 30.0_wp

        ! Target: Halfway between Z1 and Z2 (log Z = -0.5)
        z_target = -0.5_wp 
        
        call compute_interpolated_ssp(ctx, z_target, mass_out, lbol_out, spec_out)

        ! Expected: 0.5 * 10 + 0.5 * 20 = 15.0
        call assert_float_equals(15.0_wp, mass_out(1), EPS, "Z-Interp T1", total_tests, total_failures)
        call assert_float_equals(15.0_wp, mass_out(2), EPS, "Z-Interp T2", total_tests, total_failures)

        deallocate(mass_out, lbol_out, spec_out)
        call teardown_isochrone_context(ctx)
        deallocate(ctx)
    end subroutine test_interp_grid_z_only

    subroutine test_mdf_smoothing_kernel()
        !> Case 3b: MDF Smoothing (Triangular Kernel)
        type(fsps_context_t), allocatable, target :: ctx
        real(WP), allocatable :: mass_out(:), lbol_out(:), spec_out(:,:)
        real(WP) :: z_target, z_param_smooth
        integer :: nt
        
        call print_group("Interp SSP: MDF Smoothing")

        allocate(ctx)
        call setup_isochrone_context(ctx, 3, 1) ! 3 Metallicities
        nt = 1
        allocate(mass_out(nt), lbol_out(nt), spec_out(1, nt))
        
        ! Grid: Exactly aligned with Z values
        ctx%state%zlegend = [0.0019_wp, 0.019_wp, 0.19_wp]
        ctx%state%zsol = 0.019_wp
        
        ! Mass is a delta function at the center
        ctx%state%mass_ssp_zz(1, 1) = 0.0_wp
        ctx%state%mass_ssp_zz(1, 2) = 100.0_wp ! Center
        ctx%state%mass_ssp_zz(1, 3) = 0.0_wp

        ! Target exactly at center Z (idx 2)
        z_target = 0.0_wp 
        z_param_smooth = -1.0_wp ! Triggers smoothing

        call compute_interpolated_ssp(ctx, z_target, mass_out, lbol_out, spec_out, z_param=z_param_smooth)

        call assert_float_equals(50.0_wp, mass_out(1), EPS, "Smoothed Center Weight", total_tests, total_failures)

        deallocate(mass_out, lbol_out, spec_out)
        call teardown_isochrone_context(ctx)
        deallocate(ctx)
    end subroutine test_mdf_smoothing_kernel

    subroutine test_mdf_power_law()
        !> Case 3a: MDF Power Law Integration
        type(fsps_context_t), allocatable, target :: ctx
        real(WP), allocatable :: mass_out(:), lbol_out(:), spec_out(:,:)
        integer :: nt
        
        call print_group("Interp SSP: MDF Power Law")

        allocate(ctx)
        call setup_isochrone_context(ctx, 2, 1) 
        nt = 1
        allocate(mass_out(nt), lbol_out(nt), spec_out(1, nt))

        ctx%state%zlegend = [0.019_wp, 0.038_wp] ! Z and 2Z
        ctx%state%zsol = 0.019_wp
        ctx%state%mass_ssp_zz(1, :) = 1.0_wp ! Mass constant 1.0

        ! If Mass is constant 1.0, the integrated mass MUST be 1.0
        ! because the MDF weights are normalized to sum to 1.
        
        call compute_interpolated_ssp(ctx, 0.0_wp, mass_out, lbol_out, spec_out, z_param=1.0_wp)

        call assert_float_equals(1.0_wp, mass_out(1), EPS, "MDF Normalization", total_tests, total_failures)

        deallocate(mass_out, lbol_out, spec_out)
        call teardown_isochrone_context(ctx)
        deallocate(ctx)
    end subroutine test_mdf_power_law

    ! --------------------------------------------------------------------
    ! Utility: int to string
    ! --------------------------------------------------------------------
    pure function to_str(i) result(str)
        integer, intent(in) :: i
        character(len=16) :: str
        write(str, '(I0)') i
    end function to_str

    !> Blue stragglers (and blue HB stars) are appended to the isochrone buffer after the native
    !> isochrone points, with initial masses above the turn-off. The remnant budget must use the
    !> turn-off of the native isochrone: appending a straggler may only add that star's own mass,
    !> w * M_act, never reduce the remnant mass of stars that have died. (Legacy FSPS takes
    !> MAXVAL(mini) after ADD_BS and so undercounts remnants when sbss > 0.)
    subroutine test_remnants_ignore_appended_stragglers()
        type(fsps_context_t), allocatable :: ctx
        type(isochrone_buffer_t) :: buf
        real(WP) :: mass_native, mass_with_bs, lbol
        real(WP), parameter :: mini(3) = [0.5_wp, 0.6_wp, 0.8_wp]
        real(WP), parameter :: bs_mass = 1.2_wp, bs_weight = 0.01_wp

        call print_group("Integrate: Remnants Ignore Appended Blue Stragglers")

        allocate(ctx)
        ctx%add_stellar_remnants_val = 1
        ctx%imf_type_val = 0
        ctx%state%salp_ind = 2.35_wp
        ctx%state%imf_lower_limit = 0.08_wp
        ctx%state%imf_upper_limit = 120.0_wp
        ctx%state%imf_lower_bound = 0.08_wp

        ! One-age, one-metallicity synthetic isochrone with a turn-off at 0.8 Msun
        allocate(ctx%state%mini_isoc(1, 1, NM), ctx%state%mact_isoc(1, 1, NM), ctx%state%logl_isoc(1, 1, NM), &
                 ctx%state%logt_isoc(1, 1, NM), ctx%state%logg_isoc(1, 1, NM), ctx%state%phase_isoc(1, 1, NM), &
                 ctx%state%ffco_isoc(1, 1, NM), ctx%state%lmdot_isoc(1, 1, NM), ctx%state%nmass_isoc(1, 1))
        ctx%state%mini_isoc = 0.0_wp
        ctx%state%mact_isoc = 0.0_wp
        ctx%state%logl_isoc = 0.0_wp
        ctx%state%logt_isoc = 3.7_wp
        ctx%state%logg_isoc = 4.5_wp
        ctx%state%phase_isoc = 0.0_wp
        ctx%state%ffco_isoc = 0.0_wp
        ctx%state%lmdot_isoc = -99.0_wp
        ctx%state%mini_isoc(1, 1, 1:3) = mini
        ctx%state%mact_isoc(1, 1, 1:3) = mini
        ctx%state%nmass_isoc(1, 1) = 3

        call init_isochrone_buffer(buf)

        ! Native isochrone only
        call load_timestep_data(ctx, 1, 1, buf)
        buf%weights(1:3) = 1.0_wp
        ! compute_integrated_properties reads the buffer on the device in OpenACC builds
        !$acc enter data copyin(buf)
        !$acc enter data create(buf%initial_mass, buf%current_mass, buf%log_lum, buf%weights)
        !$acc enter data attach(buf%initial_mass)
        !$acc enter data attach(buf%current_mass)
        !$acc enter data attach(buf%log_lum)
        !$acc enter data attach(buf%weights)
        !$acc update device(buf%weights, buf%current_mass, buf%log_lum, buf%initial_mass)
        call compute_integrated_properties(ctx, buf, mass_native, lbol)

        ! Same isochrone plus one appended straggler above the turn-off (as apply_blue_stragglers does)
        call load_timestep_data(ctx, 1, 1, buf)
        buf%weights(1:3) = 1.0_wp
        buf%n_stars = 4
        buf%initial_mass(4) = bs_mass
        buf%current_mass(4) = bs_mass
        buf%log_lum(4) = 0.5_wp
        buf%phase(4) = 7.0_wp
        buf%weights(4) = bs_weight
        !$acc update device(buf%weights, buf%current_mass, buf%log_lum, buf%initial_mass)
        call compute_integrated_properties(ctx, buf, mass_with_bs, lbol)
        !$acc exit data delete(buf%initial_mass, buf%current_mass, buf%log_lum, buf%weights)
        !$acc exit data delete(buf)

        call assert_relative_error(mass_native + bs_weight * bs_mass, mass_with_bs, 1.0e-12_wp, &
                                   "straggler adds only its own mass (remnant turn-off unchanged)", &
                                   total_tests, total_failures)

        call free_isochrone_buffer(buf)
        deallocate(ctx%state%mini_isoc, ctx%state%mact_isoc, ctx%state%logl_isoc, ctx%state%logt_isoc, &
                   ctx%state%logg_isoc, ctx%state%phase_isoc, ctx%state%ffco_isoc, ctx%state%lmdot_isoc, &
                   ctx%state%nmass_isoc)
        deallocate(ctx)
    end subroutine test_remnants_ignore_appended_stragglers

end module test_fsps_ssp_mod
