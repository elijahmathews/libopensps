program test_context_params
    !> Context parameters: (1) a parameter the SSPs depend on, set to the value it already has, changes
    !> nothing (the basis and the bindings' SSP grids are kept: re-setting `imf_type` on every call made a
    !> single-model call ten times slower, #96), while a real change invalidates; every SSP-sensitive key is
    !> comparable (a key the comparison does not know would count as changed: safe but slow); (2) H0, om0 and
    !> ol0 set on a context change the cosmology table and so the distance modulus of the magnitudes and the
    !> age of the universe (#95: they were ignored after setup), checked against an independent integration;
    !> invalid cosmologies are refused. Uses the small test library (FSPS_TEST_LIBRARY); no other data.
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use opensps_precision, only: WP, KP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_ssp, only: generate_ssp_grid
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_set_pset, &
                               opensps_context_get_pset, opensps_context_prepare_pset, &
                               opensps_context_set_param_int, opensps_context_set_param_float, &
                               opensps_context_set_param_str, ssp_int_param_value, ssp_float_param_value, &
                               is_ssp_sensitive_int_param, is_ssp_sensitive_float_param
    use opensps_context_types, only: opensps_context_t
    use opensps_csp_batch, only: compute_csp_batch
    use opensps_cosmology, only: get_luminosity_distance, get_universe_age, refresh_cosmology
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    interface
        integer(c_int) function c_setenv(name, value, overwrite) bind(C, name='setenv')
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: name(*), value(*)
            integer(c_int), value :: overwrite
        end function c_setenv
    end interface

    character(len=*), parameter :: INT_KEYS(*) = [character(len=24) :: &
        'zmet', 'wgp1', 'wgp2', 'wgp3', 'evtype', 'imf_type', 'tpagb_norm_type', 'interpolation_type', &
        'use_wr_spectra', 'use_lw_tpagb', 'add_neb_emission', 'add_neb_continuum', 'add_xrb_emission', &
        'add_stellar_remnants', 'smooth_velocity', 'smooth_lsf', 'smoothspec_fast', 'vactoair_flag', &
        'use_isoc_mdot', 'add_agb_dust_model']
    character(len=*), parameter :: FLOAT_KEYS(*) = [character(len=24) :: &
        'tiny_logt', 'imf_upper_limit', 'imf_lower_limit', 'logt_wmb_hot', 'nebular_smooth_init', 'imf1', &
        'imf2', 'imf3', 'vdmc', 'mdave', 'dell', 'delt', 'sbss', 'fbhb', 'pagb', 'redgb', 'agb', 'masscut', &
        'fcstar', 'frac_xrb', 'sigma_smooth', 'min_wave_smooth', 'max_wave_smooth', 'gas_logu', 'gas_logz', &
        'agb_dust', 'frac_obrun']
    real(WP), parameter :: C_KMS = 299792.458_wp
    integer :: total_tests = 0, total_failures = 0
    character(len=1024) :: root
    type(opensps_context_t), allocatable :: ctx
    type(PARAMS) :: p1(1)
    real(WP), allocatable :: spec_z(:, :, :), mass_z(:, :), lbol_z(:, :)
    real(WP), allocatable :: mags0(:, :), mags1(:, :), spec1(:, :), mass1(:), lbol1(:)
    real(WP) :: nan, dmag_expected, dl_old, dl_new
    integer :: nspec, nbands, nt, nz, z, st, k
    logical :: found

    call get_environment_variable('FSPS_TEST_LIBRARY', root)
    if (len_trim(root) == 0) then
        write (*, '(A)') '1..0 # SKIP FSPS_TEST_LIBRARY is not set'
        stop
    end if
    if (c_setenv('OPENSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('FSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('SPS_HOME'//c_null_char, ''//c_null_char, 1_c_int) /= 0) stop 3
    nan = ieee_value(1.0_wp, ieee_quiet_nan)

    call print_major_header("Context parameters: unchanged sets, cosmology")
    allocate(ctx)
    call opensps_create(ctx)
    call opensps_setup(ctx, 0, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')

    ! ---------------------------------------------------------------------------------------------
    call print_group("a set to the same value changes nothing; a change invalidates (integers)")
    do k = 1, size(INT_KEYS)
        block
            integer :: cur, other
            character(len=24) :: key
            key = INT_KEYS(k)
            call assert_true(is_ssp_sensitive_int_param(key), trim(key)//": is SSP-sensitive", total_tests, total_failures)
            cur = ssp_int_param_value(ctx, key, found)
            call assert_true(found, trim(key)//": its value can be compared", total_tests, total_failures)
            ctx%state%ssp_basis_is_dirty = .false.
            call set_int(key, cur, st)
            call assert_true(st == 0 .and. .not. ctx%state%ssp_basis_is_dirty .and. .not. ctx%last_set_changed_ssp, &
                             trim(key)//": the same value keeps the basis", total_tests, total_failures)
            other = cur + 1
            call set_int(key, other, st)
            call assert_true(st == 0 .and. ctx%state%ssp_basis_is_dirty .and. ctx%last_set_changed_ssp, &
                             trim(key)//": another value invalidates it", total_tests, total_failures)
            call set_int(key, cur, st)
            call assert_true(ssp_int_param_value(ctx, key, found) == cur, trim(key)//": restored", &
                             total_tests, total_failures)
        end block
    end do

    call print_group("... (floating point, with NaN and strings)")
    do k = 1, size(FLOAT_KEYS)
        block
            real(WP) :: cur
            character(len=24) :: key
            key = FLOAT_KEYS(k)
            call assert_true(is_ssp_sensitive_float_param(key), trim(key)//": is SSP-sensitive", total_tests, total_failures)
            cur = ssp_float_param_value(ctx, key, found)
            call assert_true(found, trim(key)//": its value can be compared", total_tests, total_failures)
            ctx%state%ssp_basis_is_dirty = .false.
            call set_float(key, cur, st)
            call assert_true(st == 0 .and. .not. ctx%state%ssp_basis_is_dirty .and. .not. ctx%last_set_changed_ssp, &
                             trim(key)//": the same value keeps the basis", total_tests, total_failures)
            call set_float(key, cur + 0.5_wp, st)
            call assert_true(st == 0 .and. ctx%state%ssp_basis_is_dirty .and. ctx%last_set_changed_ssp, &
                             trim(key)//": another value invalidates it", total_tests, total_failures)
            call set_float(key, cur, st)
        end block
    end do
    ! A NaN is never equal to itself: always a change (the safe side)
    call set_float('imf1', nan, st)
    ctx%state%ssp_basis_is_dirty = .false.
    call set_float('imf1', nan, st)
    call assert_true(ctx%state%ssp_basis_is_dirty, "NaN set twice counts as a change", total_tests, total_failures)
    call set_float('imf1', -1.3_wp, st)
    ! strings
    call opensps_context_set_param_str(ctx, 'imf_filename', 'a.dat', st)
    ctx%state%ssp_basis_is_dirty = .false.
    call opensps_context_set_param_str(ctx, 'imf_filename', 'a.dat', st)
    call assert_true(.not. ctx%state%ssp_basis_is_dirty .and. .not. ctx%last_set_changed_ssp, &
                     "imf_filename: the same name keeps the basis", total_tests, total_failures)
    call opensps_context_set_param_str(ctx, 'imf_filename', 'b.dat', st)
    call assert_true(ctx%state%ssp_basis_is_dirty .and. ctx%last_set_changed_ssp, &
                     "imf_filename: another name invalidates it", total_tests, total_failures)
    call opensps_context_set_param_str(ctx, 'imf_filename', '', st)
    ! a key the SSPs do not depend on never invalidates, changed or not
    call print_group("parameters the SSPs do not depend on")
    do k = 1, 3
        ctx%state%ssp_basis_is_dirty = .false.
        select case (k)
        case (1); call set_float('dust2', 0.1_wp * k, st)
        case (2); call set_int('dust_type', k, st)
        case (3); call set_int('sfh', k, st)
        end select
        call assert_true(st == 0 .and. .not. ctx%state%ssp_basis_is_dirty .and. .not. ctx%last_set_changed_ssp, &
                         "dust2 / dust_type / sfh do not touch the basis", total_tests, total_failures)
    end do
    call set_int('dust_type', 0, st)
    call set_int('sfh', 0, st)
    call set_float('dust2', 0.0_wp, st)

    ! ---------------------------------------------------------------------------------------------
    call print_group("cosmology: the table and the distance of the magnitudes follow H0, om0 and ol0")
    nspec = ctx%state%nspec
    nbands = ctx%state%nbands
    nt = ctx%state%ntfull
    nz = ctx%state%nz
    allocate(spec_z(nspec, nt, nz), mass_z(nt, nz), lbol_z(nt, nz))
    do z = 1, nz
        call set_int('zmet', z, st)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, p1(1))
        call opensps_context_set_pset(ctx, p1(1))
        call generate_ssp_grid(ctx, p1(1), mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
    end do
    call opensps_context_update_ssp_basis(ctx, spec_z(:, :, 1:1), mass_z(:, 1:1), lbol_z(:, 1:1), 1)
    call set_int('imf_type', 1, st)
    call set_int('dust_type', 2, st)
    call set_int('add_igm_absorption', 0, st)
    call set_int('add_neb_emission', 0, st)
    call set_int('compute_indices', 0, st)
    call set_int('sfh', 1, st)
    call set_float('tau', 1.5_wp, st)
    call set_float('tage', 3.0_wp, st)
    call set_float('dust2', 0.2_wp, st)
    allocate(mags0(nbands, 1), mags1(nbands, 1), spec1(nspec, 1), mass1(1), lbol1(1))

    ! defaults: FSPS's H0 = 72, om0 = 0.27, ol0 = 0.73 at setup
    call assert_true(abs(ctx%H0_val - 72.0_wp) < 1.0e-12_wp .and. abs(ctx%om0_val - 0.27_wp) < 1.0e-12_wp, &
                     "default cosmology: H0 = 72, om0 = 0.27", total_tests, total_failures)
    call check_table("the table of the default cosmology")

    call batch_at(0.5_wp, mags0)
    ! one parameter at a time, then all: the table is rebuilt once, at the next computation
    call set_float('H0', 69.32_wp, st)
    call assert_true(st == 0 .and. ctx%cosmology_dirty, "H0 set: the table is marked stale", total_tests, total_failures)
    call set_float('om0', 0.2865_wp, st)
    call set_float('ol0', 0.7135_wp, st)
    call batch_at(0.5_wp, mags1)
    call assert_true(.not. ctx%cosmology_dirty, "the computation rebuilt the table", total_tests, total_failures)
    dl_old = dl_independent(0.5_wp, 72.0_wp, 0.27_wp, 0.73_wp)
    dl_new = dl_independent(0.5_wp, 69.32_wp, 0.2865_wp, 0.7135_wp)
    dmag_expected = 5.0_wp * log10(dl_new / dl_old)
    call check_shift(mags0, mags1, dmag_expected, "WMAP9 against the defaults at z = 0.5")
    call check_table("the table of the new cosmology")
    call assert_true(abs(ctx%state%tuniv / age_independent(69.32_wp, 0.2865_wp, 0.7135_wp) - 1.0_wp) < 5.0e-4_wp, &
                     "age of the universe at z = 0 follows (against an independent integral, 5e-4)", &
                     total_tests, total_failures)

    ! an unchanged value does not mark the table stale
    call set_float('H0', 69.32_wp, st)
    call assert_true(.not. ctx%cosmology_dirty, "the same H0: the table stays valid", total_tests, total_failures)

    ! Omega_m alone, at another redshift
    call set_float('om0', 0.31_wp, st)
    call set_float('ol0', 0.69_wp, st)
    call batch_at(2.0_wp, mags1)
    call set_float('H0', 72.0_wp, st)
    call set_float('om0', 0.27_wp, st)
    call set_float('ol0', 0.73_wp, st)
    call batch_at(2.0_wp, mags0)
    dmag_expected = 5.0_wp * log10(dl_independent(2.0_wp, 69.32_wp, 0.31_wp, 0.69_wp) / &
                                   dl_independent(2.0_wp, 72.0_wp, 0.27_wp, 0.73_wp))
    call check_shift(mags0, mags1, dmag_expected, "om0 = 0.31 at z = 2 (and back to the defaults)")

    ! invalid cosmologies are refused and leave the context unchanged
    call print_group("invalid cosmologies")
    call set_float('H0', 0.0_wp, st);  call assert_true(st /= 0, "H0 = 0 is refused", total_tests, total_failures)
    call set_float('H0', -10.0_wp, st); call assert_true(st /= 0, "H0 < 0 is refused", total_tests, total_failures)
    call set_float('H0', nan, st);      call assert_true(st /= 0, "H0 = NaN is refused", total_tests, total_failures)
    call set_float('om0', -0.1_wp, st); call assert_true(st /= 0, "om0 < 0 is refused", total_tests, total_failures)
    call assert_true(abs(ctx%H0_val - 72.0_wp) < 1.0e-12_wp .and. abs(ctx%om0_val - 0.27_wp) < 1.0e-12_wp .and. &
                     .not. ctx%cosmology_dirty, "a refused value changes nothing", total_tests, total_failures)

    call opensps_destroy(ctx)
    deallocate(ctx)
    call print_summary_line("context parameters", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    subroutine set_int(key, value, status)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer, intent(out) :: status
        call opensps_context_set_param_int(ctx, key, value, status)
    end subroutine set_int

    subroutine set_float(key, value, status)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer, intent(out) :: status
        call opensps_context_set_param_float(ctx, key, value, status)
    end subroutine set_float

    !> Magnitudes of one model at redshift z (a batch of one: the distance term is applied per model).
    subroutine batch_at(z, mags)
        real(WP), intent(in) :: z
        real(WP), intent(out) :: mags(:, :)
        integer :: status, s2

        call set_float('zred', z, s2)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, p1(1))
        call compute_csp_batch(ctx, p1, mags, status, spec=spec1, mass=mass1, lbol=lbol1)
        if (status /= 0) then
            write (*, '(A,I0)') 'compute_csp_batch status ', status
            stop 1
        end if
    end subroutine batch_at

    !> Every band moves by the same amount, the change of the distance modulus.
    subroutine check_shift(m0, m1, expected, label)
        real(WP), intent(in) :: m0(:, :), m1(:, :), expected
        character(len=*), intent(in) :: label
        real(WP) :: d(size(m0, 1))
        logical :: ok(size(m0, 1))

        ok = (m0(:, 1) == m0(:, 1)) .and. (m1(:, 1) == m1(:, 1)) .and. abs(m0(:, 1)) < 1.0e3_wp .and. abs(m1(:, 1)) < 1.0e3_wp
        d = m1(:, 1) - m0(:, 1)
        call assert_true(count(ok) >= 5, label//": bands to compare", total_tests, total_failures)
        ! (the tabulated distance is linear between 500 grid points: ~1e-4 mag at z = 2; FP32 magnitudes: 1e-5)
        call assert_true(all(abs(pack(d, ok) - expected) < 1.0e-3_wp), &
                         label//": the shift is the distance modulus (1e-3 mag)", total_tests, total_failures)
        write (*, '(A,A,ES11.3,A,ES11.3,A,ES9.2)') '# ', label, expected, '  expected;  measured', sum(pack(d, ok)) / &
            max(count(ok), 1), '  spread', maxval(pack(d, ok)) - minval(pack(d, ok))
    end subroutine check_shift

    !> The table (z, age, D_L) of the context against the independent integrals.
    subroutine check_table(label)
        character(len=*), intent(in) :: label
        integer :: rows(7), r
        real(WP) :: dl, age, errd, erra

        rows = [499, 450, 400, 300, 200, 100, 50]
        errd = 0.0_wp
        erra = 0.0_wp
        do r = 1, size(rows)
            dl = dl_independent(ctx%state%cosmospl(rows(r), 1), ctx%H0_val, ctx%om0_val, ctx%ol0_val)
            age = age_independent_z(ctx%state%cosmospl(rows(r), 1), ctx%H0_val, ctx%om0_val, ctx%ol0_val)
            if (dl > 0.0_wp) errd = max(errd, abs(ctx%state%cosmospl(rows(r), 3) / dl - 1.0_wp))
            erra = max(erra, abs(ctx%state%cosmospl(rows(r), 2) / age - 1.0_wp))
            write (*, '(A,F9.3,2ES11.2)') '# z, D_L and age relative errors of the table row: ', ctx%state%cosmospl(rows(r), 1), &
                abs(ctx%state%cosmospl(rows(r), 3) / max(dl, tiny(1.0_wp)) - 1.0_wp), &
                abs(ctx%state%cosmospl(rows(r), 2) / age - 1.0_wp)
        end do
        call assert_true(errd < 1.0e-5_wp, label//": D_L column against an independent integral (1e-5)", &
                         total_tests, total_failures)
        call assert_true(erra < 5.0e-4_wp, label//": age column against an independent integral (5e-4)", &
                         total_tests, total_failures)
    end subroutine check_table

    !> Luminosity distance in pc: (1 + z) c / H0 * integral_0^z dz' / E(z'), Simpson's rule, 20001 points.
    function dl_independent(z, h0, om0, ol0) result(dl)
        real(WP), intent(in) :: z, h0, om0, ol0
        real(WP) :: dl, hz, s, x
        integer :: i
        integer, parameter :: N = 20000

        dl = 0.0_wp
        if (z <= 0.0_wp) return
        hz = z / real(N, WP)
        s = 0.0_wp
        do i = 0, N
            x = real(i, WP) * hz
            s = s + merge(1.0_wp, merge(4.0_wp, 2.0_wp, mod(i, 2) == 1), i == 0 .or. i == N) / sqrt(om0 * (1.0_wp + x)**3 + ol0)
        end do
        dl = (1.0_wp + z) * (C_KMS / h0) * (s * hz / 3.0_wp) * 1.0e6_wp
    end function dl_independent

    !> Age of the universe at z in Gyr: (977.792 / H0) integral_z^inf dz' / ((1 + z') E(z')), in u = ln(1 + z'),
    !> Simpson's rule to z = 1e4 (the integrand is ~ exp(-1.5 u) there: the tail is below 1e-6).
    function age_independent_z(z, h0, om0, ol0) result(age)
        real(WP), intent(in) :: z, h0, om0, ol0
        real(WP) :: age, u0, u1, hu, s, u
        integer :: i
        integer, parameter :: N = 40000

        u0 = log(1.0_wp + z)
        u1 = log(1.0e4_wp)
        hu = (u1 - u0) / real(N, WP)
        s = 0.0_wp
        do i = 0, N
            u = u0 + real(i, WP) * hu
            s = s + merge(1.0_wp, merge(4.0_wp, 2.0_wp, mod(i, 2) == 1), i == 0 .or. i == N) / sqrt(om0 * exp(3.0_wp * u) + ol0)
        end do
        age = (977.792_wp / h0) * (s * hu / 3.0_wp)
    end function age_independent_z

    function age_independent(h0, om0, ol0) result(age)
        real(WP), intent(in) :: h0, om0, ol0
        real(WP) :: age
        age = age_independent_z(0.0_wp, h0, om0, ol0)
    end function age_independent

end program test_context_params
