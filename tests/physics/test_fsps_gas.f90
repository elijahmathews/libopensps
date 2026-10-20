module test_fsps_gas_mod
    use fsps_precision, only: WP, KP
    use fsps_constants, only: NEMLINE, NEBNAGE, NEBNZ, NEBNIP, C_LIGHT, H_PLANCK, L_SOL
    use fsps_types, only: params
    use fsps_context_types, only: fsps_context_t
    use fsps_context, only: fsps_context_move_to_device, fsps_context_remove_from_device, &
                            fsps_context_prepare_csp_workspace
    use fsps_gas
    use fsps_integration, only: integrate_trapezoid_array
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_int_equals, assert_relative_error
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_gas_tests, total_failures, total_tests

    real(WP), parameter :: EPS = 1.0e-6_wp
    real(WP), parameter :: REL_EPS = 1.0e-6_wp
    real(WP), parameter :: LYMAN_LIMIT = 912.0_wp

contains

    !> @brief Wrapper for interpolate_zu_slice (now scalar in core)
    subroutine interpolate_zu_slice(grid, res, idx_age, idx_z, idx_u, w_z, w_u)
        real(WP), dimension(:,:,:,:), intent(in) :: grid
        real(WP), dimension(:), intent(out)    :: res
        integer, intent(in)  :: idx_age, idx_z, idx_u
        real(WP), intent(in) :: w_z, w_u
        integer :: i
        
        do i = 1, size(res)
            res(i) = interpolate_zu_slice_point(grid, i, idx_age, idx_z, idx_u, w_z, w_u)
        end do
    end subroutine interpolate_zu_slice

    !> @brief
    !> Unit test suite for fsps_gas module.
    subroutine run_fsps_gas_tests()

        call print_minor_header("fsps_gas")

        call test_interpolate_identity()
        call test_interpolate_corners()
        call test_dimensionality_reduction()

        call test_dark_universe()
        call test_ionizing_conservation()
        call test_leaky_bucket()
        call test_line_profile_gaussian()

        call test_xrb_switch()
        call test_grid_clamping()
        call test_old_universe_clamp()
        call test_toggle_switches()

        call test_xrb_switch_lines()
        call test_no_ionizing_wavelengths()

        call test_weight_continuity()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_gas_tests

    ! ------------------------------------------------------------------------
    ! Helper: Setup a minimal FSPS context for gas tests
    ! ------------------------------------------------------------------------
    subroutine setup_gas_context(ctx, lambda, time_full)
        type(fsps_context_t), allocatable, intent(out) :: ctx
        real(WP), dimension(:), intent(in) :: lambda
        real(WP), dimension(:), intent(in) :: time_full

        integer :: i, n_wave, n_time

        allocate(ctx)
        n_wave = size(lambda)
        n_time = size(time_full)
        ctx%state%nspec = n_wave

        allocate(ctx%state%spec_lambda(n_wave))
        allocate(ctx%state%spec_nu(n_wave))
        allocate(ctx%state%nebem_cont(n_wave, NEBNZ, NEBNAGE, NEBNIP))
        allocate(ctx%state%xnebem_cont(n_wave, NEBNZ, NEBNAGE, NEBNIP))
        allocate(ctx%state%neb_res_min(NEMLINE))
        allocate(ctx%state%gaussnebarr(n_wave, NEMLINE))
        allocate(ctx%state%time_full(n_time))

        ctx%state%spec_lambda = lambda
        ctx%state%spec_nu = C_LIGHT / lambda
        ctx%state%time_full = time_full
        ctx%state%whlylim = count(lambda < LYMAN_LIMIT)

        do i = 1, NEBNZ
            ctx%state%nebem_logz(i) = real(i - 1, WP)
        end do
        do i = 1, NEBNIP
            ctx%state%nebem_logu(i) = real(i - 1, WP)
        end do
        do i = 1, NEBNAGE
            ctx%state%nebem_age(i) = real(i, WP)
        end do

        ctx%state%nebem_cont = 0.0_wp
        ctx%state%xnebem_cont = 0.0_wp
        ctx%state%neb_res_min = 1.0_wp
        ctx%state%gaussnebarr = 0.0_wp
        ctx%state%nebem_line_pos = 1500.0_wp
        ctx%state%nebem_line = 0.0_wp
        ctx%state%xnebem_line = 0.0_wp

        ctx%setup_nebular_gaussians_val = 0
        ctx%nebemlineinspec_val = 0
        ctx%add_neb_continuum_val = 0
        ctx%add_xrb_emission_val = 0
        ctx%smooth_velocity_val = 0

        call fsps_context_prepare_csp_workspace(ctx)
        call fsps_context_move_to_device(ctx)
    end subroutine setup_gas_context

    ! ------------------------------------------------------------------------
    ! Helper: Teardown a minimal FSPS context
    ! ------------------------------------------------------------------------
    subroutine teardown_gas_context(ctx)
        type(fsps_context_t), allocatable, intent(inout) :: ctx

        if (.not. allocated(ctx)) return

        call fsps_context_remove_from_device(ctx)

        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%spec_nu)) deallocate(ctx%state%spec_nu)
        if (associated(ctx%state%nebem_cont)) deallocate(ctx%state%nebem_cont)
        if (associated(ctx%state%xnebem_cont)) deallocate(ctx%state%xnebem_cont)
        if (associated(ctx%state%neb_res_min)) deallocate(ctx%state%neb_res_min)
        if (associated(ctx%state%gaussnebarr)) deallocate(ctx%state%gaussnebarr)
        if (associated(ctx%state%time_full)) deallocate(ctx%state%time_full)

        deallocate(ctx)
    end subroutine teardown_gas_context

    ! ------------------------------------------------------------------------
    ! Helper: Expected Q(H) for a given spectrum
    ! ------------------------------------------------------------------------
    pure function compute_expected_q(ctx, spec_in, frac_obrun) result(q_val)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(:), intent(in) :: spec_in
        real(WP), intent(in) :: frac_obrun
        real(WP) :: q_val

        integer :: whlylim
        real(WP) :: integral_flux

        whlylim = ctx%state%whlylim

        if (whlylim < 2) then
            q_val = 0.0_wp
        else
            integral_flux = integrate_trapezoid_array( &
                ctx%state%spec_nu(:whlylim), &
                spec_in(:whlylim) / ctx%state%spec_nu(:whlylim) &
            )
            q_val = (integral_flux / H_PLANCK * L_SOL) * (1.0_wp - frac_obrun)
        end if
    end function compute_expected_q

    ! ------------------------------------------------------------------------
    ! Helper: Compute FWHM for a line profile
    ! ------------------------------------------------------------------------
    pure subroutine compute_fwhm(lambda, profile, fwhm)
        real(WP), dimension(:), intent(in) :: lambda
        real(WP), dimension(:), intent(in) :: profile
        real(WP), intent(out) :: fwhm

        integer :: i, idx_peak
        real(WP) :: peak, half, left, right, frac

        idx_peak = maxloc(profile, dim=1)
        peak = profile(idx_peak)
        half = 0.5_wp * peak

        left = lambda(1)
        right = lambda(size(lambda))

        do i = idx_peak - 1, 1, -1
            if (profile(i) <= half) then
                frac = (half - profile(i)) / (profile(i+1) - profile(i))
                left = lambda(i) + frac * (lambda(i+1) - lambda(i))
                exit
            end if
        end do

        do i = idx_peak + 1, size(lambda)
            if (profile(i) <= half) then
                frac = (half - profile(i-1)) / (profile(i) - profile(i-1))
                right = lambda(i-1) + frac * (lambda(i) - lambda(i-1))
                exit
            end if
        end do

        fwhm = right - left
    end subroutine compute_fwhm

    ! ------------------------------------------------------------------------
    ! GROUP 1: Mathematical Precision & Interpolation Tests
    ! ------------------------------------------------------------------------
    subroutine test_interpolate_identity()
        integer, parameter :: n_wave = 5, n_z = 2, n_age = 2, n_u = 2
        real(WP), dimension(n_wave, n_z, n_age, n_u) :: grid
        real(WP), dimension(n_wave) :: res
        real(WP) :: w_z, w_u, expected
        integer :: iw, iz, ia, iu
        character(len=64) :: label

        call print_group("Interpolate Z/U Slice: Identity Grid")

        do iw = 1, n_wave
            do iz = 1, n_z
                do ia = 1, n_age
                    do iu = 1, n_u
                        grid(iw, iz, ia, iu) = real(iw + iz + ia + iu, WP)
                    end do
                end do
            end do
        end do

        w_z = 0.5_wp
        w_u = 0.25_wp
        call interpolate_zu_slice(grid, res, 1, 1, 1, w_z, w_u)

        do iw = 1, n_wave
            expected = real(iw, WP) + 3.0_wp + w_z + w_u
            write(label, '(A,I0)') "Identity grid wave ", iw
            call assert_float_equals(expected, res(iw), EPS, trim(label), total_tests, total_failures)
        end do
    end subroutine test_interpolate_identity

    subroutine test_interpolate_corners()
        integer, parameter :: n_wave = 4, n_z = 2, n_age = 2, n_u = 2
        real(WP), dimension(n_wave, n_z, n_age, n_u) :: grid
        real(WP), dimension(n_wave) :: res, expected
        integer :: iw, iz, ia, iu

        call print_group("Interpolate Z/U Slice: Corner Cases")

        do iw = 1, n_wave
            do iz = 1, n_z
                do ia = 1, n_age
                    do iu = 1, n_u
                        grid(iw, iz, ia, iu) = real(iw + iz + ia + iu, WP)
                    end do
                end do
            end do
        end do

        call interpolate_zu_slice(grid, res, 1, 1, 1, 0.0_wp, 0.0_wp)
        expected = [(real(iw, WP) + 3.0_wp, iw=1,n_wave)]
        call assert_true(all(abs(res - expected) <= EPS), "Corner (wz=0, wu=0)", total_tests, total_failures)

        call interpolate_zu_slice(grid, res, 1, 1, 1, 1.0_wp, 0.0_wp)
        expected = [(real(iw, WP) + 4.0_wp, iw=1,n_wave)]
        call assert_true(all(abs(res - expected) <= EPS), "Corner (wz=1, wu=0)", total_tests, total_failures)

        call interpolate_zu_slice(grid, res, 1, 1, 1, 0.0_wp, 1.0_wp)
        expected = [(real(iw, WP) + 4.0_wp, iw=1,n_wave)]
        call assert_true(all(abs(res - expected) <= EPS), "Corner (wz=0, wu=1)", total_tests, total_failures)

        call interpolate_zu_slice(grid, res, 1, 1, 1, 1.0_wp, 1.0_wp)
        expected = [(real(iw, WP) + 5.0_wp, iw=1,n_wave)]
        call assert_true(all(abs(res - expected) <= EPS), "Corner (wz=1, wu=1)", total_tests, total_failures)
    end subroutine test_interpolate_corners

    subroutine test_dimensionality_reduction()
        integer, parameter :: n_wave = 4, n_z = 2, n_age = 2, n_u = 2
        real(WP), dimension(n_wave, n_z, n_age, n_u) :: grid
        real(WP), dimension(n_wave) :: res_age1, res_age2
        real(WP) :: w_z, w_u, w_a
        real(WP) :: full_trilinear, reduced
        integer :: iw, iz, ia, iu

        call print_group("Dimensionality Reduction Consistency")

        do iw = 1, n_wave
            do iz = 1, n_z
                do ia = 1, n_age
                    do iu = 1, n_u
                        grid(iw, iz, ia, iu) = 10.0_wp + 7.0_wp * real(iw, WP) + &
                                               2.0_wp * real(iz, WP) + 3.0_wp * real(ia, WP) + &
                                               5.0_wp * real(iu, WP)
                    end do
                end do
            end do
        end do

        w_z = 0.4_wp
        w_u = 0.2_wp
        w_a = 0.6_wp
        iw = 2

        call interpolate_zu_slice(grid, res_age1, 1, 1, 1, w_z, w_u)
        call interpolate_zu_slice(grid, res_age2, 2, 1, 1, w_z, w_u)
        reduced = (1.0_wp - w_a) * res_age1(iw) + w_a * res_age2(iw)

        full_trilinear = &
            (1.0_wp - w_a) * (1.0_wp - w_z) * (1.0_wp - w_u) * grid(iw,1,1,1) + &
            (1.0_wp - w_a) * (1.0_wp - w_z) * (         w_u) * grid(iw,1,1,2) + &
            (1.0_wp - w_a) * (         w_z) * (1.0_wp - w_u) * grid(iw,2,1,1) + &
            (1.0_wp - w_a) * (         w_z) * (         w_u) * grid(iw,2,1,2) + &
            (         w_a) * (1.0_wp - w_z) * (1.0_wp - w_u) * grid(iw,1,2,1) + &
            (         w_a) * (1.0_wp - w_z) * (         w_u) * grid(iw,1,2,2) + &
            (         w_a) * (         w_z) * (1.0_wp - w_u) * grid(iw,2,2,1) + &
            (         w_a) * (         w_z) * (         w_u) * grid(iw,2,2,2)

        call assert_float_equals(full_trilinear, reduced, EPS, "Trilinear vs reduced", total_tests, total_failures)
    end subroutine test_dimensionality_reduction

    ! ------------------------------------------------------------------------
    ! GROUP 2: Physics Logic & Integration Tests
    ! ------------------------------------------------------------------------
    subroutine test_dark_universe()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), dimension(5) :: lambda
        real(WP), dimension(1) :: time_full
        real(KP), allocatable :: sspi(:,:), sspo(:,:), nebemline(:,:)
        integer :: tmp_add_neb_cont, tmp_nebemlineinspec
        real(WP) :: tmp_nebem_cont_val, tmp_nebem_line_val
        integer :: n_wave

        call print_group("Dark Universe (Zero Ionizing Flux)")

        lambda = [500.0_wp, 700.0_wp, 900.0_wp, 1200.0_wp, 1500.0_wp]
        time_full = [1.0_wp]
        call setup_gas_context(ctx, lambda, time_full)

        n_wave = size(lambda)
        allocate(sspi(n_wave,1), sspo(n_wave,1), nebemline(NEMLINE,1))
        sspo = 0.0_wp
        nebemline = 0.0_wp

        sspi(:,1) = [0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp]

        pset%frac_obrun = 1.0_wp
        pset%gas_logz = 0.0_wp
        pset%gas_logu = 0.0_wp

        ctx%add_neb_continuum_val = 1
        ctx%nebemlineinspec_val = 1
        ctx%state%nebem_cont = 0.5_wp
        ctx%state%nebem_line = 0.5_wp
        tmp_nebem_cont_val = ctx%state%nebem_cont(1,1,1,1)
        tmp_nebem_line_val = ctx%state%nebem_line(1,1,1,1)
        ctx%state%nebem_cont = tmp_nebem_cont_val
        ctx%state%nebem_line = tmp_nebem_line_val
        !$acc update device(ctx%state%nebem_cont, ctx%state%nebem_line)

        !$acc data copyin(sspi, pset) copy(sspo, nebemline)
        !$acc update device(sspo, nebemline)
        !$acc update device(pset)
        tmp_add_neb_cont = ctx%add_neb_continuum_val
        tmp_nebemlineinspec = ctx%nebemlineinspec_val
        ctx%add_neb_continuum_val = tmp_add_neb_cont
        ctx%nebemlineinspec_val = tmp_nebemlineinspec
        !$acc update device(ctx%add_neb_continuum_val, ctx%nebemlineinspec_val)
        call apply_nebular_emission(ctx, pset, sspi, sspo, nebemline)
        !$acc update self(sspo, nebemline)
        !$acc end data

        call assert_true(all(abs(sspo - sspi) <= EPS), "No nebular emission when Q=0", total_tests, total_failures)
        call assert_true(all(abs(nebemline) <= EPS), "No nebular lines when Q=0", total_tests, total_failures)

        deallocate(sspi, sspo, nebemline)
        call teardown_gas_context(ctx)
    end subroutine test_dark_universe

    subroutine test_ionizing_conservation()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), dimension(5) :: lambda
        real(WP), dimension(1) :: time_full
        real(KP), dimension(5,1) :: spec_in, spec_out
        real(WP) :: q_val, expected_q
        integer :: whlylim

        call print_group("Ionizing Photon Conservation (frac_obrun=0)")

        lambda = [500.0_wp, 700.0_wp, 900.0_wp, 1100.0_wp, 1300.0_wp]
        time_full = [1.0_wp]
        call setup_gas_context(ctx, lambda, time_full)

        spec_in = 2.0_wp
        spec_out = spec_in
        pset%frac_obrun = 0.0_wp

        call process_ionizing_radiation(ctx, pset, spec_in, spec_out, 1, q_val)
        expected_q = compute_expected_q(ctx, real(spec_in(:,1), WP), pset%frac_obrun)

        whlylim = ctx%state%whlylim
        call assert_true(all(abs(spec_out(1:whlylim,1)) <= EPS), "Ionizing flux fully absorbed", total_tests, total_failures)
        call assert_relative_error(expected_q, q_val, REL_EPS, "Q(H) matches trapezoid", total_tests, total_failures)

        call teardown_gas_context(ctx)
    end subroutine test_ionizing_conservation

    subroutine test_leaky_bucket()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), dimension(5) :: lambda
        real(WP), dimension(1) :: time_full
        real(KP), dimension(5,1) :: spec_in, spec_out
        real(WP) :: q_val, expected_q
        integer :: whlylim

        call print_group("Leaky Bucket (frac_obrun=0.3)")

        lambda = [500.0_wp, 700.0_wp, 900.0_wp, 1100.0_wp, 1300.0_wp]
        time_full = [1.0_wp]
        call setup_gas_context(ctx, lambda, time_full)

        spec_in = 1.5_wp
        spec_out = spec_in
        pset%frac_obrun = 0.3_wp

        call process_ionizing_radiation(ctx, pset, spec_in, spec_out, 1, q_val)
        expected_q = compute_expected_q(ctx, real(spec_in(:,1), WP), pset%frac_obrun)

        whlylim = ctx%state%whlylim
        call assert_true(all(abs(spec_out(1:whlylim,1) - 0.3_wp * spec_in(1:whlylim,1)) <= EPS), &
                         "Leakage scaling below 912A", total_tests, total_failures)
        call assert_relative_error(expected_q, q_val, REL_EPS, "Q(H) scales with (1-frac_obrun)", total_tests, total_failures)

        call teardown_gas_context(ctx)
    end subroutine test_leaky_bucket

    subroutine test_line_profile_gaussian()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: lambda(:)
        real(WP), dimension(1) :: time_full
        real(KP), allocatable :: sspi(:,:), sspo(:,:), nebemline(:,:)
        real(WP), allocatable :: line_wpec(:)
        real(WP) :: q_val, expected_fwhm, fwhm
        real(WP) :: sigma_angstroms
        integer :: tmp_add_neb_cont, tmp_nebemlineinspec, tmp_smooth_velocity
        real(WP) :: tmp_line_pos1
        integer :: n_line, n_wave, i

        call print_group("Line Profile Integrity (Gaussian Widths)")

        n_line = 101
        n_wave = 3 + n_line
        allocate(lambda(n_wave))
        lambda(1:3) = [700.0_wp, 800.0_wp, 900.0_wp]
        do i = 1, n_line
            lambda(3 + i) = 1450.0_wp + real(i - 1, WP)
        end do

        time_full = [1.0_wp]
        call setup_gas_context(ctx, lambda, time_full)

        allocate(sspi(n_wave,1), sspo(n_wave,1), nebemline(NEMLINE,1), line_wpec(n_wave))
        sspo = 0.0_wp
        nebemline = 0.0_wp
        line_wpec = 0.0_wp

        ! Physical scales (ionizing continuum ~1e-13 L_sun/Hz per M_sun; log L/Q ~ -45 for a strong
        ! line), so that the spectra stay within FP32 range in kernel-precision builds
        sspi(:,1) = 0.0_wp
        sspi(1:3,1) = 1.0e-13_wp

        pset%frac_obrun = 0.0_wp
        pset%gas_logz = 0.0_wp
        pset%gas_logu = 0.0_wp
        pset%sigma_smooth = 1000.0_wp

        ctx%add_neb_continuum_val = 0
        ctx%nebemlineinspec_val = 1
        ctx%smooth_velocity_val = 1
        ctx%state%nebem_line = -75.0_wp
        ctx%state%nebem_line(1,:,:,:) = -45.0_wp
        ctx%state%nebem_line_pos(1) = 1500.0_wp
        tmp_line_pos1 = ctx%state%nebem_line_pos(1)
        ctx%state%nebem_line = -75.0_wp
        ctx%state%nebem_line(1,:,:,:) = -45.0_wp
        ctx%state%nebem_line_pos(1) = tmp_line_pos1
        !$acc update device(ctx%state%nebem_line, ctx%state%nebem_line_pos)

        !$acc data copyin(sspi, pset) copy(sspo, nebemline, line_wpec)
        !$acc update device(sspo, nebemline, line_wpec)
        !$acc update device(pset)
        tmp_add_neb_cont = ctx%add_neb_continuum_val
        tmp_nebemlineinspec = ctx%nebemlineinspec_val
        tmp_smooth_velocity = ctx%smooth_velocity_val
        ctx%add_neb_continuum_val = tmp_add_neb_cont
        ctx%nebemlineinspec_val = tmp_nebemlineinspec
        ctx%smooth_velocity_val = tmp_smooth_velocity
        !$acc update device(ctx%add_neb_continuum_val, ctx%nebemlineinspec_val, ctx%smooth_velocity_val)
        call apply_nebular_emission(ctx, pset, sspi, sspo, nebemline)
        !$acc update self(sspo, nebemline)
        !$acc end data

        line_wpec = sspo(:,1) - sspi(:,1)
        line_wpec(1:3) = 0.0_wp   ! ionizing pixels (absorbed by the gas), not line flux
        q_val = nebemline(1,1)

        sigma_angstroms = ctx%state%nebem_line_pos(1) * pset%sigma_smooth * (1.0e13_wp / C_LIGHT)
        expected_fwhm = 2.0_wp * sqrt(2.0_wp * log(2.0_wp)) * sigma_angstroms
        call compute_fwhm(lambda, line_wpec, fwhm)

        call assert_relative_error(expected_fwhm, fwhm, 0.05_wp, "FWHM matches sigma_smooth", total_tests, total_failures)
        call assert_relative_error(q_val, integrate_trapezoid_array(ctx%state%spec_nu, line_wpec), 1.0e-3_wp, &
                                   "Line flux conserved", total_tests, total_failures)

        deallocate(lambda, sspi, sspo, nebemline, line_wpec)
        call teardown_gas_context(ctx)
    end subroutine test_line_profile_gaussian

    ! ------------------------------------------------------------------------
    ! GROUP 3: Edge Cases & Robustness
    ! ------------------------------------------------------------------------
    subroutine test_xrb_switch()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), dimension(4) :: lambda
        real(WP), dimension(1) :: time_full
        real(KP), allocatable :: sspi(:,:), sspo(:,:)
        integer :: tmp_add_neb_cont, tmp_nebemlineinspec, tmp_add_xrb
        real(WP) :: tmp_nebem_cont_val, tmp_xnebem_cont_val

        call print_group("XRB Switch (BPSS grid)")

        lambda = [500.0_wp, 700.0_wp, 900.0_wp, 1200.0_wp]
        time_full = [1.0_wp]
        call setup_gas_context(ctx, lambda, time_full)

        allocate(sspi(4,1), sspo(4,1))
        sspo = 0.0_wp
        sspi(:,1) = 0.0_wp
        sspi(1:3,1) = 1.0_wp

        pset%frac_obrun = 0.0_wp
        pset%gas_logz = 0.0_wp
        pset%gas_logu = 0.0_wp

        ctx%add_neb_continuum_val = 1
        ctx%nebemlineinspec_val = 0
        ctx%add_xrb_emission_val = 1
        ctx%state%isoc_type = 'bpss'

        ctx%state%nebem_cont = -30.0_wp
        ctx%state%xnebem_cont = 0.0_wp
        tmp_nebem_cont_val = ctx%state%nebem_cont(1,1,1,1)
        tmp_xnebem_cont_val = ctx%state%xnebem_cont(1,1,1,1)
        ctx%state%nebem_cont = tmp_nebem_cont_val
        ctx%state%xnebem_cont = tmp_xnebem_cont_val
        !$acc update device(ctx%state%nebem_cont, ctx%state%xnebem_cont)

        !$acc data copyin(sspi, pset) copy(sspo)
        !$acc update device(pset)
        tmp_add_neb_cont = ctx%add_neb_continuum_val
        tmp_nebemlineinspec = ctx%nebemlineinspec_val
        tmp_add_xrb = ctx%add_xrb_emission_val
        ctx%add_neb_continuum_val = tmp_add_neb_cont
        ctx%nebemlineinspec_val = tmp_nebemlineinspec
        ctx%add_xrb_emission_val = tmp_add_xrb
        !$acc update device(ctx%add_neb_continuum_val, ctx%nebemlineinspec_val, ctx%add_xrb_emission_val)
        call apply_nebular_emission(ctx, pset, sspi, sspo)
        !$acc update self(sspo)
        !$acc end data

        call assert_true(sspo(4,1) > 0.0_wp, "Uses XRB grid for continuum", total_tests, total_failures)

        deallocate(sspi, sspo)
        call teardown_gas_context(ctx)
    end subroutine test_xrb_switch

    subroutine test_grid_clamping()
        real(WP), dimension(4) :: grid
        integer :: idx
        real(WP) :: w

        call print_group("Grid Bounds Clamping")

        grid = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp]

        call get_grid_indices_weights(grid, 5.0_wp, size(grid), idx, w)
        call assert_int_equals(3, idx, "Clamp high: idx", total_tests, total_failures)
        call assert_float_equals(1.0_wp, w, EPS, "Clamp high: weight", total_tests, total_failures)

        call get_grid_indices_weights(grid, -2.0_wp, size(grid), idx, w)
        call assert_int_equals(1, idx, "Clamp low: idx", total_tests, total_failures)
        call assert_float_equals(0.0_wp, w, EPS, "Clamp low: weight", total_tests, total_failures)
    end subroutine test_grid_clamping

    subroutine test_old_universe_clamp()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), dimension(4) :: lambda
        real(WP), dimension(2) :: time_full
        real(KP), allocatable :: sspi(:,:), sspo(:,:)
        integer :: tmp_add_neb_cont, tmp_nebemlineinspec
        real(WP) :: tmp_nebem_cont_val

        call print_group("Old Universe (Age > Grid)")

        lambda = [500.0_wp, 700.0_wp, 900.0_wp, 1200.0_wp]
        time_full = [1.0e9_wp, 2.0e9_wp]
        call setup_gas_context(ctx, lambda, time_full)

        allocate(sspi(4,2), sspo(4,2))
        sspo = 0.0_wp
        sspi(:,1) = 0.0_wp
        sspi(1:3,1) = 1.0_wp
        sspi(:,2) = sspi(:,1)

        pset%frac_obrun = 0.0_wp
        pset%gas_logz = 0.0_wp
        pset%gas_logu = 0.0_wp

        ctx%add_neb_continuum_val = 1
        ctx%nebemlineinspec_val = 0
        ctx%state%nebem_cont = 1.0_wp
        tmp_nebem_cont_val = ctx%state%nebem_cont(1,1,1,1)
        ctx%state%nebem_cont = tmp_nebem_cont_val
        !$acc update device(ctx%state%nebem_cont)

        !$acc data copyin(sspi, pset) copy(sspo)
        !$acc update device(pset)
        tmp_add_neb_cont = ctx%add_neb_continuum_val
        tmp_nebemlineinspec = ctx%nebemlineinspec_val
        ctx%add_neb_continuum_val = tmp_add_neb_cont
        ctx%nebemlineinspec_val = tmp_nebemlineinspec
        !$acc update device(ctx%add_neb_continuum_val, ctx%nebemlineinspec_val)
        call apply_nebular_emission(ctx, pset, sspi, sspo)
        !$acc update self(sspo)
        !$acc end data

        call assert_true(all(abs(sspo - sspi) <= EPS), "No processing beyond nebular grid", total_tests, total_failures)

        deallocate(sspi, sspo)
        call teardown_gas_context(ctx)
    end subroutine test_old_universe_clamp

    subroutine test_toggle_switches()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), dimension(4) :: lambda
        real(WP), dimension(1) :: time_full
        real(KP), allocatable :: sspi(:,:), sspo(:,:), nebemline(:,:)
        integer :: tmp_add_neb_cont, tmp_nebemlineinspec
        real(WP) :: tmp_nebem_cont_val

        call print_group("Toggle Switches (Continuum/Lines)")

        lambda = [500.0_wp, 700.0_wp, 900.0_wp, 1200.0_wp]
        time_full = [1.0_wp]
        call setup_gas_context(ctx, lambda, time_full)

        allocate(sspi(4,1), sspo(4,1), nebemline(NEMLINE,1))
        sspo = 0.0_wp
        nebemline = 0.0_wp
        sspi(:,1) = 0.0_wp
        sspi(1:3,1) = 1.0_wp

        pset%frac_obrun = 0.0_wp
        pset%gas_logz = 0.0_wp
        pset%gas_logu = 0.0_wp

        ! Continuum disabled
        ctx%add_neb_continuum_val = 0
        ctx%nebemlineinspec_val = 0
        ctx%state%nebem_cont = 1.0_wp
        tmp_nebem_cont_val = ctx%state%nebem_cont(1,1,1,1)
        ctx%state%nebem_cont = tmp_nebem_cont_val
        !$acc update device(ctx%state%nebem_cont)
        !$acc data copyin(sspi, pset) copy(sspo)
        !$acc update device(pset)
        tmp_add_neb_cont = ctx%add_neb_continuum_val
        tmp_nebemlineinspec = ctx%nebemlineinspec_val
        ctx%add_neb_continuum_val = tmp_add_neb_cont
        ctx%nebemlineinspec_val = tmp_nebemlineinspec
        !$acc update device(ctx%add_neb_continuum_val, ctx%nebemlineinspec_val)
        call apply_nebular_emission(ctx, pset, sspi, sspo)
        !$acc update self(sspo)
        !$acc end data
        call assert_true(all(abs(sspo(4:4,1) - sspi(4:4,1)) <= EPS), "Continuum toggle off", total_tests, total_failures)

        ! Lines disabled in spectrum, but nebemline should populate
        ctx%add_neb_continuum_val = 0
        ctx%nebemlineinspec_val = 0
        ctx%state%nebem_line = -30.0_wp
        ctx%state%nebem_line(1,:,:,:) = 0.0_wp
        ctx%state%nebem_line = -30.0_wp
        ctx%state%nebem_line(1,:,:,:) = 0.0_wp
        !$acc update device(ctx%state%nebem_line)
        !$acc data copyin(sspi, pset) copy(sspo, nebemline)
        !$acc update device(sspo, nebemline)
        !$acc update device(pset)
        tmp_add_neb_cont = ctx%add_neb_continuum_val
        tmp_nebemlineinspec = ctx%nebemlineinspec_val
        ctx%add_neb_continuum_val = tmp_add_neb_cont
        ctx%nebemlineinspec_val = tmp_nebemlineinspec
        !$acc update device(ctx%add_neb_continuum_val, ctx%nebemlineinspec_val)
        call apply_nebular_emission(ctx, pset, sspi, sspo, nebemline)
        !$acc update self(sspo, nebemline)
        !$acc end data
        call assert_true(all(abs(sspo(4:4,1) - sspi(4:4,1)) <= EPS), "Lines disabled in spectrum", total_tests, total_failures)
        call assert_true(nebemline(1,1) > 0.0_wp, "Nebemline populated when requested", total_tests, total_failures)

        deallocate(sspi, sspo, nebemline)
        call teardown_gas_context(ctx)
    end subroutine test_toggle_switches


    ! ------------------------------------------------------------------------
    ! GROUP 4: Edge Cases
    ! ------------------------------------------------------------------------
    subroutine test_xrb_switch_lines()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), dimension(4) :: lambda
        real(WP), dimension(1) :: time_full
        real(KP), allocatable :: sspi(:,:), sspo(:,:), nebemline(:,:)
        integer :: tmp_add_neb_cont, tmp_nebemlineinspec, tmp_add_xrb

        call print_group("XRB Switch for Lines (BPSS grid)")

        lambda = [500.0_wp, 700.0_wp, 900.0_wp, 1200.0_wp]
        time_full = [1.0_wp]
        call setup_gas_context(ctx, lambda, time_full)

        allocate(sspi(4,1), sspo(4,1), nebemline(NEMLINE,1))
        sspo = 0.0_wp
        nebemline = 0.0_wp
        sspi(:,1) = 0.0_wp
        sspi(1:3,1) = 1.0_wp ! Provide ionizing flux

        pset%frac_obrun = 0.0_wp
        pset%gas_logz = 0.0_wp
        pset%gas_logu = 0.0_wp

        ! Request lines AND use BPSS XRB grid
        ctx%add_neb_continuum_val = 0
        ctx%nebemlineinspec_val = 1
        ctx%add_xrb_emission_val = 1
        ctx%state%isoc_type = 'bpss'

        ! Set xnebem_line to a distinct valid value to ensure it processes
        ctx%state%nebem_line = -100.0_wp
        ctx%state%xnebem_line = -20.0_wp

        !$acc update device(ctx%state%nebem_line, ctx%state%xnebem_line)

        !$acc data copyin(sspi, pset) copy(sspo, nebemline)
        !$acc update device(pset)
        tmp_add_neb_cont = ctx%add_neb_continuum_val
        tmp_nebemlineinspec = ctx%nebemlineinspec_val
        tmp_add_xrb = ctx%add_xrb_emission_val
        ctx%add_neb_continuum_val = tmp_add_neb_cont
        ctx%nebemlineinspec_val = tmp_nebemlineinspec
        ctx%add_xrb_emission_val = tmp_add_xrb
        !$acc update device(ctx%add_neb_continuum_val, ctx%nebemlineinspec_val, ctx%add_xrb_emission_val)

        call apply_nebular_emission(ctx, pset, sspi, sspo, nebemline)

        !$acc update self(sspo, nebemline)
        !$acc end data

        ! If the xnebem_line grid was accessed, we should get emission based on -20.0
        call assert_true(nebemline(1,1) > 0.0_wp, "Uses XRB grid for lines successfully", total_tests, total_failures)

        deallocate(sspi, sspo, nebemline)
        call teardown_gas_context(ctx)
    end subroutine test_xrb_switch_lines

    subroutine test_no_ionizing_wavelengths()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), dimension(3) :: lambda
        real(WP), dimension(1) :: time_full
        real(KP), dimension(3,1) :: spec_in, spec_out
        real(WP) :: q_val

        call print_group("No Ionizing Wavelengths (whlylim < 2)")

        ! All wavelengths are > 912 A (Lyman Limit)
        lambda = [1000.0_wp, 1500.0_wp, 2000.0_wp]
        time_full = [1.0_wp]
        call setup_gas_context(ctx, lambda, time_full)

        spec_in = 1.0_wp
        spec_out = spec_in
        pset%frac_obrun = 0.0_wp

        ! Process radiation; because there are no wavelengths < 912A, whlylim < 2
        call process_ionizing_radiation(ctx, pset, spec_in, spec_out, 1, q_val)

        call assert_float_equals(0.0_wp, q_val, EPS, "Q(H) strictly 0.0 when whlylim < 2", total_tests, total_failures)

        call teardown_gas_context(ctx)
    end subroutine test_no_ionizing_wavelengths

    ! ------------------------------------------------------------------------
    ! GROUP 5: Automatic Differentiation (AD) Safety
    ! ------------------------------------------------------------------------
    subroutine test_weight_continuity()
        real(WP), dimension(3) :: grid
        real(WP) :: v1, v2, w1, w2, res1, res2
        integer :: idx1, idx2

        call print_group("Weight Continuity")

        grid = [0.0_wp, 1.0_wp, 2.0_wp]
        v1 = 0.999_wp
        v2 = 1.001_wp

        call get_grid_indices_weights(grid, v1, size(grid), idx1, w1)
        call get_grid_indices_weights(grid, v2, size(grid), idx2, w2)

        res1 = (1.0_wp - w1) * grid(idx1) + w1 * grid(idx1 + 1)
        res2 = (1.0_wp - w2) * grid(idx2) + w2 * grid(idx2 + 1)

        call assert_float_equals(v1, res1, EPS, "Left of grid point", total_tests, total_failures)
        call assert_float_equals(v2, res2, EPS, "Right of grid point", total_tests, total_failures)
        call assert_true(abs(res2 - res1) <= 0.01_wp, "No jump across grid point", total_tests, total_failures)
    end subroutine test_weight_continuity

end module test_fsps_gas_mod
