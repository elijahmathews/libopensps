program test_filters
    !> Bandpass curves (doc/FILTERS_PLAN.md): SHA-256 against the FIPS test vectors; cleaning, content
    !> hashes, detector types, the two resamplings, coverage policy and calibration on synthetic grids,
    !> each against an analytic result; then, on a context of the small test library (FSPS_TEST_LIBRARY),
    !> installing filters after setup: the default list through the new path is bit-identical to the
    !> setup's, user curves give AB = 0 for a flat f_nu and Vega-consistent zero points, the batched and
    !> single-spectrum photometry agree on the new set, errors leave the context unchanged, and other
    !> contexts and a re-setup are not affected.
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: iso_fortran_env, only: real64
    use opensps_precision, only: WP, KP
    use opensps_constants, only: ABS_MAG_ZEROPOINT_LOG
    use opensps_integration, only: integrate_trapezoid_array
    use opensps_sha256, only: sha256_t, sha256_init, sha256_update_string, sha256_hex
    use opensps_filters
    use opensps_context_types, only: opensps_context_t
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_context, only: opensps_context_prepare_pset, opensps_context_move_to_device
    use opensps_context_filters
    use opensps_photometry, only: compute_magnitudes
    use opensps_batch_photometry, only: compute_magnitudes_columns
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true, &
                              assert_float_equals, assert_int_equals
    implicit none

    interface
        integer(c_int) function c_setenv(name, value, overwrite) bind(C, name='setenv')
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: name(*), value(*)
            integer(c_int), value :: overwrite
        end function c_setenv
    end interface

    integer :: total_tests = 0, total_failures = 0
    character(len=1024) :: root

    call print_major_header("Bandpass curves")
    call test_sha256()
    call test_clean_and_hash()
    call test_resampling()
    call test_detectors()
    call test_coverage_policy()
    call test_calibration()

    call get_environment_variable('FSPS_TEST_LIBRARY', root)
    if (len_trim(root) > 0) then
        if (c_setenv('OPENSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
        if (c_setenv('FSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
        if (c_setenv('SPS_HOME'//c_null_char, ''//c_null_char, 1_c_int) /= 0) stop 3
        call test_context()
    end if

    call print_summary_line("bandpass curves", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    function uniform(lo, hi, h) result(g)
        real(WP), intent(in) :: lo, hi, h
        real(WP), allocatable :: g(:)
        integer :: i, n
        n = nint((hi - lo)/h) + 1
        allocate (g(n))
        g = [(lo + h*real(i - 1, WP), i=1, n)]
    end function uniform

    function one_set(lam, resp, detector) result(set)
        real(WP), intent(in) :: lam(:), resp(:)
        integer, intent(in) :: detector
        type(filter_set_t) :: set
        integer :: st
        call filter_set_add(set, 'f', lam, resp, detector, .false., 0.0_wp, 0.0_wp, st)
    end function one_set

    function digest(str) result(hex)
        character(len=*), intent(in) :: str
        character(len=64) :: hex
        type(sha256_t) :: s
        call sha256_init(s)
        call sha256_update_string(s, str)
        hex = sha256_hex(s)
    end function digest

    ! ---------------------------------------------------------------------------------------------
    subroutine test_sha256()
        type(sha256_t) :: s
        integer :: i
        character(len=64) :: hex
        call print_group("SHA-256: FIPS 180-4 test vectors")
        call assert_true(digest('') == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', &
                         "empty message", total_tests, total_failures)
        call assert_true(digest('abc') == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad', &
                         "abc", total_tests, total_failures)
        call assert_true(digest('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq') == &
                         '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1', &
                         "448-bit message (padding spills into a second block)", total_tests, total_failures)
        call sha256_init(s)
        do i = 1, 1000
            call sha256_update_string(s, repeat('a', 1000))
        end do
        hex = sha256_hex(s)
        call assert_true(hex == 'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0', &
                         "one million 'a' (streamed)", total_tests, total_failures)
    end subroutine test_sha256

    ! ---------------------------------------------------------------------------------------------
    subroutine test_clean_and_hash()
        real(WP), allocatable :: lc(:), rc(:), l2(:), r2(:)
        integer :: nc, st, nc2, st2
        character(len=64) :: h1, h2, h3, h4, h5
        type(filter_set_t) :: set
        type(filter_info_t) :: info
        real(WP), allocatable :: bands(:, :), grid(:)
        integer :: bad
        real(WP) :: nan

        call print_group("cleaning and content hashes")
        nan = 0.0_wp
        nan = nan/nan
        call filter_clean([1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp], [0.1_wp, -0.5_wp, 0.7_wp, 0.2_wp], .true., lc, rc, nc, st)
        call assert_true(st == FILTER_OK .and. nc == 3, "a repeated wavelength is dropped (the first kept)", &
                         total_tests, total_failures)
        call assert_true(rc(2) == 0.0_wp, "negative response is clamped to zero", total_tests, total_failures)
        call filter_clean([1.0_wp, nan, 3.0_wp], [1.0_wp, 1.0_wp, 1.0_wp], .true., lc, rc, nc, st)
        call assert_int_equals(FILTER_ERR_NONFINITE, st, "NaN is rejected", total_tests, total_failures)
        call filter_clean([1.0_wp, 3.0_wp, 2.0_wp], [1.0_wp, 1.0_wp, 1.0_wp], .true., lc, rc, nc, st)
        call assert_int_equals(FILTER_ERR_ORDER, st, "non-monotonic wavelengths are rejected", total_tests, total_failures)
        call filter_clean([-1.0_wp, 3.0_wp], [1.0_wp, 1.0_wp], .true., lc, rc, nc, st)
        call assert_int_equals(FILTER_ERR_NONFINITE, st, "a wavelength <= 0 is rejected", total_tests, total_failures)
        call filter_clean([5.0_wp, 5.0_wp], [1.0_wp, 1.0_wp], .true., lc, rc, nc, st)
        call assert_int_equals(FILTER_ERR_FEW_POINTS, st, "one distinct wavelength is rejected", total_tests, total_failures)

        call filter_clean([1.0_wp, 2.0_wp, 3.0_wp], [0.1_wp, 0.5_wp, 0.2_wp], .true., lc, rc, nc, st)
        call filter_clean([3.0_wp, 2.0_wp, 1.0_wp], [0.2_wp, 0.5_wp, 0.1_wp], .true., l2, r2, nc2, st2)
        call assert_true(st2 == FILTER_OK .and. all(l2(1:3) == lc(1:3)) .and. all(r2(1:3) == rc(1:3)), &
                         "a decreasing curve is reversed", total_tests, total_failures)

        h1 = filter_hash(lc, rc, nc, DETECTOR_PHOTON, .false., 0.0_wp, 0.0_wp)
        h2 = filter_hash(lc, rc, nc, DETECTOR_PHOTON, .false., 0.0_wp, 0.0_wp)
        h3 = filter_hash(lc, rc, nc, DETECTOR_ENERGY, .false., 0.0_wp, 0.0_wp)
        h4 = filter_hash(lc, rc, nc, DETECTOR_PHOTON, .true., 35500.0_wp, -1.0_wp)
        rc(2) = rc(2)*(1.0_wp + 1.0e-15_wp)
        h5 = filter_hash(lc, rc, nc, DETECTOR_PHOTON, .false., 0.0_wp, 0.0_wp)
        call assert_true(h1 == h2 .and. len_trim(h1) == 64, "the hash is deterministic, 64 characters", &
                         total_tests, total_failures)
        call assert_true(h1 /= h3 .and. h1 /= h4 .and. h1 /= h5, &
                         "the detector, the calibration and one bit of the response change the hash", &
                         total_tests, total_failures)

        ! The hash is of the curve, not of its name; the reversed and the duplicate-laden tabulations of one curve agree
        grid = uniform(0.5_wp, 4.0_wp, 0.5_wp)
        call filter_set_add(set, 'one', [1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp], [0.1_wp, 0.5_wp, 0.5_wp, 0.2_wp], &
                            DETECTOR_PHOTON, .false., 0.0_wp, 0.0_wp, st)
        call filter_set_add(set, 'two', [3.0_wp, 2.0_wp, 1.0_wp], [0.2_wp, 0.5_wp, 0.1_wp], &
                            DETECTOR_PHOTON, .false., 0.0_wp, 0.0_wp, st)
        allocate (bands(size(grid), 2))
        call filters_build(set, grid, RESAMPLE_CONSERVING, GRID_POLICY_ALLOW, .true., bands, info, st, bad)
        call assert_true(st == FILTER_OK .and. info%hash(1) == info%hash(2) .and. info%hash(1) == h1 .and. &
                         info%name(1) /= info%name(2), "one curve, two names and tabulations: one hash", &
                         total_tests, total_failures)
        set%expect_hash(2) = h3
        call filters_build(set, grid, RESAMPLE_CONSERVING, GRID_POLICY_ALLOW, .true., bands, info, st, bad)
        call assert_true(st == FILTER_ERR_HASH .and. bad == 2, "a stored hash that differs is refused", &
                         total_tests, total_failures)
    end subroutine test_clean_and_hash

    ! ---------------------------------------------------------------------------------------------
    subroutine test_resampling()
        real(WP), parameter :: lc(6) = [4100.0_wp, 4800.0_wp, 5200.0_wp, 5600.0_wp, 6500.0_wp, 6900.0_wp]
        real(WP), parameter :: rc(6) = [0.0_wp, 0.7_wp, 1.0_wp, 0.9_wp, 0.4_wp, 0.0_wp]
        real(WP), allocatable :: grid(:), band(:), g(:), f(:), m(:)
        real(WP) :: exact, ws, hh(3)
        integer :: i, k, n
        type(filter_set_t) :: set
        type(filter_info_t) :: info
        real(WP), allocatable :: bands(:, :)
        integer :: st, bad

        call print_group("resampling: the hat projection conserves the integral of T on any grid")
        n = 900
        allocate (grid(n), band(n))
        do i = 1, n
            grid(i) = 3900.0_wp + 7.3_wp*real(i - 1, WP) + 2.0_wp*sin(real(i, WP))   ! uneven, increasing
        end do
        call resample_hat(lc, rc, 6, grid, band)
        ws = 0.0_wp
        ws = ws + band(1)*0.5_wp*(grid(2) - grid(1)) + band(n)*0.5_wp*(grid(n) - grid(n - 1))
        do i = 2, n - 1
            ws = ws + band(i)*0.5_wp*(grid(i + 1) - grid(i - 1))
        end do
        exact = 0.0_wp
        do k = 1, 5
            exact = exact + 0.5_wp*(lc(k + 1) - lc(k))*(rc(k) + rc(k + 1))
        end do
        call assert_float_equals(exact, ws, 1.0e-10_wp*exact, "sum w_i T_i = integral of T (uneven grid)", &
                                 total_tests, total_failures)

        ! The same curve on grids of 0.5, 1 and 4 A: the mean of a smooth spectrum agrees
        allocate (m(3))
        hh = [0.5_wp, 1.0_wp, 4.0_wp]
        do k = 1, 3
            g = uniform(3900.0_wp, 7100.0_wp, hh(k))
            set%n = 0
            call filter_set_clear(set)
            call filter_set_add(set, 'bell', lc, rc, DETECTOR_PHOTON, .false., 0.0_wp, 0.0_wp, st)
            if (allocated(bands)) deallocate (bands)
            allocate (bands(size(g), 1))
            call filters_build(set, g, RESAMPLE_CONSERVING, GRID_POLICY_ERROR, .true., bands, info, st, bad)
            f = 1.0_wp + (g/5500.0_wp)**(-2) + 0.3_wp*sin(g/700.0_wp)
            m(k) = integrate_trapezoid_array(g, f*bands(:, 1)/g)
        end do
        call assert_float_equals(m(1), m(2), 1.0e-6_wp*m(1), "mean over the band, 0.5 A vs 1 A grid", &
                                 total_tests, total_failures)
        call assert_float_equals(m(1), m(3), 2.0e-5_wp*m(1), "mean over the band, 0.5 A vs 4 A grid", &
                                 total_tests, total_failures)

        ! The legacy interpolation: exactly the piecewise-linear curve at the grid points
        g = uniform(3900.0_wp, 7100.0_wp, 10.0_wp)
        deallocate (band)
        allocate (band(size(g)))
        call resample_point(lc, rc, 6, g, band)
        call assert_float_equals(0.95_wp, band(nint((5400.0_wp - 3900.0_wp)/10.0_wp) + 1), 1.0e-12_wp, &
                                 "legacy point interpolation at 5400 A", total_tests, total_failures)
        call assert_true(band(1) == 0.0_wp .and. band(size(g)) == 0.0_wp, "zero outside the curve", &
                         total_tests, total_failures)
    end subroutine test_resampling

    ! ---------------------------------------------------------------------------------------------
    subroutine test_detectors()
        real(WP) :: l1, l2, expected, got
        real(WP), allocatable :: g(:), f(:), bands(:, :)
        type(filter_set_t) :: set
        type(filter_info_t) :: info
        integer :: st, bad

        call print_group("detector types against analytic means of f_nu = 1/lambda")
        l1 = 5000.0_wp
        l2 = 6000.0_wp
        g = uniform(4000.0_wp, 8000.0_wp, 0.25_wp)
        f = 1.0_wp/g
        allocate (bands(size(g), 2))
        call filter_set_add(set, 'photon', [l1, l2], [1.0_wp, 1.0_wp], DETECTOR_PHOTON, .false., 0.0_wp, 0.0_wp, st)
        call filter_set_add(set, 'energy', [l1, l2], [1.0_wp, 1.0_wp], DETECTOR_ENERGY, .false., 0.0_wp, 0.0_wp, st)
        call filters_build(set, g, RESAMPLE_CONSERVING, GRID_POLICY_ERROR, .true., bands, info, st, bad)
        call assert_int_equals(FILTER_OK, st, "build", total_tests, total_failures)
        ! photon counter: <f_nu> = int f_nu dlambda/lambda / int dlambda/lambda
        expected = (1.0_wp/l1 - 1.0_wp/l2)/log(l2/l1)
        got = integrate_trapezoid_array(g, f*bands(:, 1)/g)
        call assert_float_equals(expected, got, 1.0e-7_wp*expected, "photon counter", total_tests, total_failures)
        ! energy counter: <f_nu> = int f_nu dnu / int dnu = (1/l1 + 1/l2)/2 for f_nu = 1/lambda (T flat in lambda)
        expected = 0.5_wp*(1.0_wp/l1 + 1.0_wp/l2)
        got = integrate_trapezoid_array(g, f*bands(:, 2)/g)
        call assert_float_equals(expected, got, 1.0e-7_wp*expected, "energy counter", total_tests, total_failures)
        call assert_true(info%hash(1) /= info%hash(2), "the two detectors have different hashes", &
                         total_tests, total_failures)
        got = integrate_trapezoid_array(g, bands(:, 2)/g)
        call assert_float_equals(1.0_wp, got, 1.0e-12_wp, "both are normalized to integral T/lambda = 1", &
                                 total_tests, total_failures)
    end subroutine test_detectors

    ! ---------------------------------------------------------------------------------------------
    subroutine test_coverage_policy()
        real(WP), allocatable :: g(:), bands(:, :)
        type(filter_set_t) :: set
        type(filter_info_t) :: info
        integer :: st, bad
        real(WP) :: expected

        call print_group("coverage: filters beyond the grid, dead and undersampled filters")
        g = uniform(4000.0_wp, 8000.0_wp, 4.0_wp)
        allocate (bands(size(g), 3))
        call filter_set_add(set, 'half out', [3000.0_wp, 5000.0_wp], [1.0_wp, 1.0_wp], DETECTOR_PHOTON, &
                            .false., 0.0_wp, 0.0_wp, st)
        call filters_build(set, g, RESAMPLE_CONSERVING, GRID_POLICY_ERROR, .true., bands(:, 1:1), info, st, bad)
        call assert_true(st == FILTER_ERR_COVERAGE .and. bad == 1, "error policy: a filter half outside the grid", &
                         total_tests, total_failures)
        expected = log(5000.0_wp/4000.0_wp)/log(5000.0_wp/3000.0_wp)
        call assert_float_equals(expected, info%coverage(1), 1.0e-12_wp, "coverage = integral T/lambda inside / total", &
                                 total_tests, total_failures)
        call filters_build(set, g, RESAMPLE_CONSERVING, GRID_POLICY_WARN, .true., bands(:, 1:1), info, st, bad)
        call assert_true(st == FILTER_OK .and. iand(info%flags(1), FILTER_FLAG_TRUNCATED) /= 0, &
                         "warn policy: reported in the flags", total_tests, total_failures)

        call filter_set_clear(set)
        call filter_set_add(set, 'outside', [100.0_wp, 200.0_wp], [1.0_wp, 1.0_wp], DETECTOR_PHOTON, &
                            .false., 0.0_wp, 0.0_wp, st)
        call filters_build(set, g, RESAMPLE_CONSERVING, GRID_POLICY_ERROR, .true., bands(:, 1:1), info, st, bad)
        call assert_true(st == FILTER_ERR_DEAD, "error policy: a filter without throughput", total_tests, total_failures)
        call filters_build(set, g, RESAMPLE_CONSERVING, GRID_POLICY_ALLOW, .true., bands(:, 1:1), info, st, bad)
        call assert_true(st == FILTER_OK .and. all(bands(:, 1) == 0.0_wp) .and. &
                         iand(info%flags(1), FILTER_FLAG_DEAD) /= 0, "allow policy: a zero band, flagged", &
                         total_tests, total_failures)

        call filter_set_clear(set)
        call filter_set_add(set, 'narrow', [5000.0_wp, 5010.0_wp], [1.0_wp, 1.0_wp], DETECTOR_PHOTON, &
                            .false., 0.0_wp, 0.0_wp, st)
        call filter_set_add(set, 'wide', [5000.0_wp, 5400.0_wp], [1.0_wp, 1.0_wp], DETECTOR_PHOTON, &
                            .false., 0.0_wp, 0.0_wp, st)
        call filters_build(set, g, RESAMPLE_CONSERVING, GRID_POLICY_ERROR, .true., bands(:, 1:2), info, st, bad)
        call assert_true(st == FILTER_OK .and. iand(info%flags(1), FILTER_FLAG_UNDERSAMPLED) /= 0 .and. &
                         info%flags(2) == 0 .and. info%coverage(2) > 0.999999_wp, &
                         "a 10 A filter on a 4 A grid is undersampled (reported, not an error); a 400 A one is not", &
                         total_tests, total_failures)
        call assert_float_equals(1.0_wp, integrate_trapezoid_array(g, bands(:, 1)/g), 1.0e-12_wp, &
                                 "the undersampled filter is still normalized", total_tests, total_failures)
    end subroutine test_coverage_policy

    ! ---------------------------------------------------------------------------------------------
    subroutine test_calibration()
        real(WP), allocatable :: g(:), band(:)
        real(WP) :: d

        call print_group("calibration: flat nu f_nu and beta = 2 blackbody conventions")
        g = uniform(4000.0_wp, 8000.0_wp, 1.0_wp)
        allocate (band(size(g)))
        band = merge(1.0_wp, 0.0_wp, g >= 5000.0_wp .and. g <= 6000.0_wp)
        band = band/integrate_trapezoid_array(g, band/g)
        call filter_calibrate_band(g, band, 5500.0_wp, -1.0_wp)
        d = integrate_trapezoid_array(g, (g/5500.0_wp)**(-1.0_wp)*band/g)
        call assert_float_equals(1.0_wp, d, 1.0e-12_wp, "exponent -1", total_tests, total_failures)
        call filter_calibrate_band(g, band, 5500.0_wp, -2.0_wp)
        d = integrate_trapezoid_array(g, (g/5500.0_wp)**(-2.0_wp)*band/g)
        call assert_float_equals(1.0_wp, d, 1.0e-12_wp, "exponent -2", total_tests, total_failures)
        call filter_calibrate_band(g, band, 5500.0_wp, 0.5_wp)
        d = integrate_trapezoid_array(g, (g/5500.0_wp)**0.5_wp*band/g)
        call assert_float_equals(1.0_wp, d, 1.0e-12_wp, "any other exponent", total_tests, total_failures)
    end subroutine test_calibration

    ! ---------------------------------------------------------------------------------------------
    subroutine test_context()
        type(opensps_context_t), allocatable :: ctx, ctx2
        type(filter_set_t) :: set, bad_set
        real(WP), allocatable :: bands0(:, :), mv0(:), ms0(:), le0(:), lam(:), spec(:), mags(:), mags0(:), &
                                 mags_ab(:), mags_vg(:), vega_spec(:), col(:, :), mcol(:, :), mag_lin(:, :), mag_lin2(:, :)
        real(KP), allocatable :: speck(:, :)
        character(len=64), allocatable :: hash0(:)
        integer :: nb0, nspec, st, k, nb
        character(len=256) :: msg
        real(WP) :: f_ab, nan, dm1, dm2
        real(WP), parameter :: AB_ZEROPOINT = 48.60_wp
        !> Magnitude tolerance of the kernels' arithmetic: FP64, or 1e-6 (the FP32 requirement is 1e-5)
        real(WP), parameter :: tolm = merge(1.0e-9_wp, 1.0e-6_wp, KP == WP)

        call print_group("filters installed after setup (context of the small test library)")
        nan = 0.0_wp
        nan = nan/nan
        allocate (ctx, ctx2)
        call opensps_create(ctx)
        call opensps_setup(ctx, 0, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
        call opensps_create(ctx2)
        call opensps_setup(ctx2, 0, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
        call assert_true(ctx%compute_vega_mags_val == 0 .and. ctx%keep_maggies_val == 0, &
                         "defaults: AB magnitudes (not the Vega system), no linear fluxes kept", total_tests, total_failures)
        nspec = ctx%state%nspec
        nb0 = ctx%state%nbands
        lam = ctx%state%spec_lambda
        bands0 = ctx%state%bands
        mv0 = ctx%state%magvega
        ms0 = ctx%state%magsun
        le0 = ctx%state%filter_leff
        hash0 = ctx%state%filter_info%hash
        call assert_true(len_trim(hash0(1)) == 64 .and. all(hash0 /= ''), "the default filters carry content hashes", &
                         total_tests, total_failures)
        call assert_true(ctx%state%filter_info%resample == RESAMPLE_LINEAR .and. &
                         ctx%state%filter_info%grid_hash == grid_hash(lam), "the report names the resampling and the grid", &
                         total_tests, total_failures)
        allocate (spec(nspec), mags0(nb0))
        f_ab = 10.0_wp**(-(AB_ZEROPOINT + 2.5_wp*ABS_MAG_ZEROPOINT_LOG)/2.5_wp)
        spec = f_ab*(1.0_wp + 0.5_wp*sin(lam/900.0_wp))
        call opensps_context_prepare_pset(ctx)
        ! On an accelerated build the context lives on the device (the batched kernels need it there), so
        ! replacing the filters takes the device path: off the device, the new arrays, back on
        call opensps_context_move_to_device(ctx)
        call compute_magnitudes(ctx, 0.0_wp, spec, mags0)

        ! 1. The default list through the new path is bit-identical to the setup's
        call opensps_context_set_filters_library(ctx, 'allfilters', st, msg, RESAMPLE_LINEAR, GRID_POLICY_ALLOW, 1)
        call assert_int_equals(0, st, "set_filters_library('allfilters'): "//trim(msg), total_tests, total_failures)
        call assert_true(ctx%state%nbands == nb0 .and. ctx%state%filters_owned, "the context owns the arrays now", &
                         total_tests, total_failures)
        call assert_true(all(ctx%state%bands == bands0), "bands bit-identical to the setup's", total_tests, total_failures)
        call assert_true(all(ctx%state%magvega == mv0) .and. all(ctx%state%magsun == ms0), &
                         "Vega and Sun zero points bit-identical", total_tests, total_failures)
        call assert_true(all(ctx%state%filter_leff == le0), "pivot wavelengths bit-identical", total_tests, total_failures)
        call assert_true(all(ctx%state%filter_info%hash == hash0), "hashes identical", total_tests, total_failures)
        allocate (mags(nb0))
        call compute_magnitudes(ctx, 0.0_wp, spec, mags)
        call assert_true(all(mags == mags0), "magnitudes bit-identical", total_tests, total_failures)

        ! 2. User curves: two top hats (one per detector) and a smooth one
        call filter_set_clear(set)
        call filter_set_add(set, 'tophat_photon', [4500.0_wp, 5500.0_wp], [1.0_wp, 1.0_wp], DETECTOR_PHOTON, .false., &
                            0.0_wp, 0.0_wp, st)
        call filter_set_add(set, 'tophat_energy', [6000.0_wp, 7000.0_wp], [1.0_wp, 1.0_wp], DETECTOR_ENERGY, .false., &
                            0.0_wp, 0.0_wp, st)
        call filter_set_add(set, 'bell', [4000.0_wp, 4500.0_wp, 5000.0_wp, 5500.0_wp, 6000.0_wp], &
                            [0.0_wp, 0.4_wp, 1.0_wp, 0.4_wp, 0.0_wp], DETECTOR_PHOTON, .false., 0.0_wp, 0.0_wp, st)
        call opensps_context_set_filters(ctx, set, st, msg)
        call assert_int_equals(0, st, "set_filters (conserving, error): "//trim(msg), total_tests, total_failures)
        nb = 3
        call assert_true(ctx%state%nbands == nb .and. size(ctx%state%bands, 2) == nb .and. &
                         size(ctx%pset%mag_compute) == nb .and. all(ctx%pset%mag_compute == 1), &
                         "nbands, the band arrays and the band mask follow the new set", total_tests, total_failures)
        call assert_true(ctx%state%filter_info%resample == RESAMPLE_CONSERVING .and. &
                         ctx%state%filter_info%grid_policy == GRID_POLICY_ERROR .and. ctx%state%vega_ref_band == 0, &
                         "defaults: conserving, error, Vega = 0 in every band", total_tests, total_failures)
        deallocate (mags)
        allocate (mags(nb))
        spec = f_ab
        call compute_magnitudes(ctx, 0.0_wp, spec, mags)
        do k = 1, nb
            call assert_float_equals(0.0_wp, mags(k), tolm, "flat f_nu: AB = 0, band "//char(48 + k), &
                                     total_tests, total_failures)
        end do

        ! 3. Zero points: Vega's own spectrum gives the same offset from magvega in every band
        vega_spec = ctx%state%vega_spec    ! f_nu on the grid
        spec = vega_spec*1.0e-30_wp
        call compute_magnitudes(ctx, 0.0_wp, spec, mags)
        dm1 = mags(1) - ctx%state%magvega(1)
        dm2 = mags(3) - ctx%state%magvega(3)
        call assert_float_equals(dm1, dm2, tolm, "mag(Vega spectrum) - magvega is the same in every band", &
                                 total_tests, total_failures)
        call assert_float_equals(dm1, mags(2) - ctx%state%magvega(2), tolm, "(energy-counting band)", &
                                 total_tests, total_failures)

        ! 4. Vega system without a reference band: AB - magvega in every band; with band 2 as reference
        allocate (mags_ab(nb), mags_vg(nb))
        spec = f_ab*(1.0_wp + 0.5_wp*sin(lam/900.0_wp))
        ctx%compute_vega_mags_val = 0
        call compute_magnitudes(ctx, 0.0_wp, spec, mags_ab)
        ctx%compute_vega_mags_val = 1
        call compute_magnitudes(ctx, 0.0_wp, spec, mags_vg)
        call assert_true(all(abs((mags_ab - mags_vg) - ctx%state%magvega) < 1.0e-10_wp), &
                         "vega_ref_band = 0: Vega-system magnitude = AB - magvega in every band", total_tests, total_failures)
        call opensps_context_set_filters(ctx, set, st, msg, vega_ref_band=2)
        ctx%compute_vega_mags_val = 1
        call compute_magnitudes(ctx, 0.0_wp, spec, mags_vg)
        ctx%compute_vega_mags_val = 0
        call compute_magnitudes(ctx, 0.0_wp, spec, mags_ab)
        call assert_float_equals(mags_ab(2), mags_vg(2), 1.0e-12_wp, "vega_ref_band = 2: that band stays AB", &
                                 total_tests, total_failures)
        call assert_float_equals(mags_ab(1) - ctx%state%magvega(1) + ctx%state%magvega(2), mags_vg(1), 1.0e-10_wp, &
                                 "vega_ref_band = 2: the others are shifted relative to it", total_tests, total_failures)
        ctx%compute_vega_mags_val = 0

        ! 5. The batched photometry (rebuilt for the new set) agrees with the single-spectrum path
        allocate (col(nspec, 4), mcol(nb, 4), speck(nspec, 4), mag_lin(nb, 4), mag_lin2(nb, 4))
        do k = 1, 4
            col(:, k) = f_ab*(1.0_wp + 0.4_wp*sin(lam/(500.0_wp + 150.0_wp*k)))
        end do
        speck = real(col, KP)
        call batch_cols(ctx, speck, mcol, maggies_out=mag_lin)
        do k = 1, 4
            call compute_magnitudes(ctx, 0.3_wp, real(speck(:, k), WP), mags)
            call assert_true(all(abs(mags - mcol(:, k)) < merge(1.0e-9_wp, 1.0e-4_wp, KP == WP)), &
                             "batched = single-spectrum magnitudes, model "//char(48 + k), total_tests, total_failures)
        end do

        ! 5b. Linear fluxes: 10^(-0.4 m) of the AB magnitudes, 0 where the magnitude is NaN, NaN when masked
        call assert_true(all(abs(mag_lin - 10.0_wp**(-0.4_wp*mcol)) <= merge(1.0e-12_wp, 1.0e-5_wp, KP == WP)*mag_lin), &
                         "maggies = 10^(-0.4 AB magnitude), forms from the band flux", total_tests, total_failures)
        speck(:, 4) = -speck(:, 4)
        call batch_cols(ctx, speck, mcol, maggies_out=mag_lin2)
        call assert_true(all(mcol(:, 4) /= mcol(:, 4)) .and. all(mag_lin2(:, 4) == 0.0_wp), &
                         "a flux that is not positive (clamped to zero): the magnitude is NaN, the linear flux 0", &
                         total_tests, total_failures)
        call batch_cols(ctx, speck, mcol, mag_compute=[1, 0, 1], maggies_out=mag_lin2)
        call assert_true(all(mag_lin2(2, :) /= mag_lin2(2, :)) .and. all(mag_lin2(1, 1:3) == mag_lin(1, 1:3)), &
                         "bands masked out are NaN, the others unchanged", total_tests, total_failures)
        ctx%compute_vega_mags_val = 1
        call batch_cols(ctx, speck, mcol, maggies_out=mag_lin2)
        ctx%compute_vega_mags_val = 0
        call assert_true(all(mag_lin2(:, 1:3) == mag_lin(:, 1:3)), &
                         "the linear fluxes are AB whatever the Vega setting", total_tests, total_failures)

        ! 6. An error leaves the context as it was; so do other contexts and a re-setup
        call filter_set_add(bad_set, 'nan', [4000.0_wp, nan, 5000.0_wp], [1.0_wp, 1.0_wp, 1.0_wp], DETECTOR_PHOTON, &
                            .false., 0.0_wp, 0.0_wp, st)
        call opensps_context_set_filters(ctx, bad_set, st, msg)
        call assert_true(st == FILTERS_ERR_CURVE .and. index(msg, 'nan') > 0 .and. ctx%state%nbands == nb, &
                         "an unusable curve: status 320, the filter is named, the context keeps its filters: "// &
                         trim(msg), total_tests, total_failures)
        call filter_set_clear(bad_set)
        call filter_set_add(bad_set, 'far_ir', [1.0e11_wp, 2.0e11_wp], [1.0_wp, 1.0_wp], DETECTOR_PHOTON, .false., &
                            0.0_wp, 0.0_wp, st)
        call opensps_context_set_filters(ctx, bad_set, st, msg)
        call assert_true(st == FILTERS_ERR_DEAD .and. ctx%state%nbands == nb, &
                         "a filter beyond the model range: refused (322), unchanged", total_tests, total_failures)
        call opensps_context_set_filters(ctx, bad_set, st, msg, grid_policy=GRID_POLICY_ALLOW)
        call assert_true(st == 0 .and. ctx%state%nbands == 1 .and. ctx%state%filter_info%flags(1) /= 0, &
                         "with the allow policy it is installed as a zero band and flagged", total_tests, total_failures)
        call opensps_context_set_filters(ctx, set, st, msg, vega_ref_band=7)
        call assert_int_equals(FILTERS_ERR_ARGUMENT, st, "a reference band beyond the set is refused", &
                               total_tests, total_failures)
        call opensps_context_set_filters_library(ctx, 'nonesuch', st, msg)
        call assert_int_equals(FILTERS_ERR_LIBRARY, st, "no such library: 324", total_tests, total_failures)

        call assert_true(ctx2%state%nbands == nb0 .and. all(ctx2%state%bands == bands0) .and. &
                         .not. ctx2%state%filters_owned, "another context of the same setup is untouched", &
                         total_tests, total_failures)
        call opensps_setup(ctx, 0, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
        call assert_true(ctx%state%nbands == nb0 .and. all(ctx%state%bands == bands0) .and. &
                         .not. ctx%state%filters_owned .and. ctx%state%vega_ref_band == 1, &
                         "a new setup restores the default filters", total_tests, total_failures)
        call opensps_destroy(ctx)
        call opensps_destroy(ctx2)
    end subroutine test_context

    !> The batched magnitudes of 4 spectra at z = 0.3 (spec is device-resident for the kernels)
    subroutine batch_cols(ctx, speck, mcol, mag_compute, maggies_out)
        type(opensps_context_t), intent(inout) :: ctx
        real(KP), intent(in) :: speck(:, :)
        real(WP), intent(out) :: mcol(:, :)
        integer, intent(in), optional :: mag_compute(:)
        real(WP), intent(out), optional :: maggies_out(:, :)

        !$omp target enter data map(to: speck)
        call compute_magnitudes_columns(ctx, 0.3_wp, 0.0_wp, speck, size(speck, 2), mcol, mag_compute, &
                                        maggies_out=maggies_out)
        !$omp target exit data map(delete: speck)
    end subroutine batch_cols

end program test_filters
