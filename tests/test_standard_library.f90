program test_standard_library
    !> Setup and synthesis from the small standard-format test library, and [alpha/Fe] selection.
    !>
    !> Usage: test_standard_library TEST_ROOT [AFE_ROOT]
    !>
    !> 1. With TEST_ROOT (tests/data/fsps-test-data, coarse subsets committed in the repository):
    !>    a full setup (MIST + MILES + DL07, nebular and dust emission) reads only standard-format
    !>    libraries, and an SSP grid and a CSP are finite and positive. No other data needed.
    !> 2. With AFE_ROOT (tools/fsps_data/make_afe_variants.py): libraries with an [alpha/Fe] axis
    !>    load exactly like the axis-free library when the axis has one value or the default
    !>    (afe = 0) is taken, and 'name:afe=0.4' selects the other block.
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fsps_precision, only: WP
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy, fsps_set_param_int, fsps_set_param_float
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset
    use fsps_context_types, only: fsps_context_t
    use fsps_ssp, only: generate_ssp_grid
    use fsps_csp, only: compute_csp_scenario
    use fsps_json, only: json_doc_t, json_parse_string, json_child, json_item, json_len, json_string, json_real, &
                         json_int, json_is_integer
    use, intrinsic :: iso_fortran_env, only: int64
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    interface
        integer(c_int) function c_setenv(name, value, overwrite) bind(C, name='setenv')
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: name(*), value(*)
            integer(c_int), value :: overwrite
        end function c_setenv
    end interface

    integer :: total_tests = 0, total_failures = 0
    character(len=1024) :: test_root, afe_root

    if (command_argument_count() < 1) then
        write (*, '(A)') 'usage: test_standard_library TEST_ROOT [AFE_ROOT]'
        stop 2
    end if
    call get_command_argument(1, test_root)
    afe_root = ''
    if (command_argument_count() >= 2) call get_command_argument(2, afe_root)

    call print_major_header('Standard-format test library')
    if (c_setenv('SPS_HOME'//c_null_char, ''//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('FSPS_DATA_FORMAT'//c_null_char, 'standard'//c_null_char, 1_c_int) /= 0) stop 3

    call test_json_parser()
    call test_end_to_end()
    if (len_trim(afe_root) > 0) call test_afe_selection()

    call print_summary_line('Standard-format test library', total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    subroutine use_root(root)
        character(len=*), intent(in) :: root
        if (c_setenv('OPENSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
        if (c_setenv('FSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    end subroutine use_root

    subroutine test_json_parser()
        type(json_doc_t) :: doc
        logical :: ok
        character(len=256) :: msg
        character(len=*), parameter :: BS = achar(92)
        character(len=:), allocatable :: text
        integer :: arr

        call print_group('JSON reader (manifests)')
        text = '{"s": "a' // BS // '"b' // BS // BS // 'c' // BS // 'u00e9' // BS // 'n", ' // &
               '"x": [0.1, -2.5e-3, 1E2, 42, -0], "o": {"k": true, "n": null}}'
        call json_parse_string(doc, text, ok, msg)
        call assert_true(ok, 'a valid document parses', total_tests, total_failures)
        call assert_true(json_string(doc, json_child(doc, 1, 's')) == 'a"b' // BS // 'c' // achar(195) // achar(169) // &
                         achar(10), 'string escapes decoded (quote, backslash, U+00E9 as UTF-8, newline)', &
                         total_tests, total_failures)
        arr = json_child(doc, 1, 'x')
        call assert_true(json_len(doc, arr) == 5 .and. json_real(doc, json_item(doc, arr, 1)) == 0.1_wp .and. &
                         json_real(doc, json_item(doc, arr, 2)) == -2.5e-3_wp .and. &
                         json_real(doc, json_item(doc, arr, 3)) == 100.0_wp, &
                         'numbers read exactly (correctly rounded)', total_tests, total_failures)
        call assert_true(json_is_integer(doc, json_item(doc, arr, 4)) .and. json_int(doc, json_item(doc, arr, 4)) == 42_int64 &
                         .and. .not. json_is_integer(doc, json_item(doc, arr, 1)), 'integers told from reals', &
                         total_tests, total_failures)
        call json_parse_string(doc, '{"a": [1, 2,]}', ok, msg)
        call assert_true(.not. ok, 'a trailing comma is rejected', total_tests, total_failures)
        call json_parse_string(doc, '{"a": 01}', ok, msg)
        call assert_true(.not. ok, 'a leading zero is rejected', total_tests, total_failures)
    end subroutine test_json_parser

    subroutine test_end_to_end()
        type(fsps_context_t) :: ctx
        type(COMPSPOUT), allocatable :: res(:)
        type(PARAMS) :: pset
        real(WP), allocatable :: mass(:), lbol(:), spec(:, :)
        integer :: status

        call print_group('Setup, SSP and CSP from the test library alone')
        call use_root(test_root)
        call fsps_create(ctx)
        call fsps_set_param_int(ctx, 'add_neb_emission', 1, status)
        call fsps_set_param_int(ctx, 'add_dust_emission', 1, status)
        call fsps_setup(ctx, -1, 'mist', 'miles', 'DL07')
        call assert_true(ctx%data_format == 'standard' .and. ctx%state%nz == 2 .and. ctx%state%nzinit == 2, &
                         'library dimensions from the manifests', total_tests, total_failures)
        call fsps_context_move_to_device(ctx)
        call fsps_set_param_int(ctx, 'zmet', 2, status)
        call fsps_set_param_int(ctx, 'sfh', 1, status)
        call fsps_set_param_float(ctx, 'tau', 1.0_wp, status)
        call fsps_set_param_float(ctx, 'tage', 3.0_wp, status)
        call fsps_set_param_float(ctx, 'dust2', 0.3_wp, status)

        call fsps_context_prepare_pset(ctx)
        call fsps_context_get_pset(ctx, pset)
        allocate (mass(ctx%state%ntfull), lbol(ctx%state%ntfull), spec(ctx%state%nspec, ctx%state%ntfull))
        call fsps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass, lbol, spec)
        call assert_true(all(ieee_is_finite(spec)) .and. all(spec >= 0.0_wp) .and. any(spec > 0.0_wp), &
                         'SSP spectra finite and non-negative', total_tests, total_failures)
        call assert_true(all(mass > 0.0_wp) .and. all(mass <= 1.0_wp) .and. all(ieee_is_finite(lbol)), &
                         'SSP masses in (0, 1], luminosities finite', total_tests, total_failures)

        call fsps_context_update_ssp_basis(ctx, reshape(spec, [size(spec, 1), size(spec, 2), 1]), &
                                           reshape(mass, [size(mass), 1]), reshape(lbol, [size(lbol), 1]), 1)
        call compute_csp_scenario(ctx, pset, 1, res, status)
        call assert_true(status == 0, 'CSP computed', total_tests, total_failures)
        if (status == 0) then
            call assert_true(all(ieee_is_finite(res(1)%spec)) .and. any(res(1)%spec > 0.0_wp), &
                             'CSP spectrum finite and positive', total_tests, total_failures)
            call assert_true(ieee_is_finite(res(1)%mags(1)) .and. res(1)%mags(1) < 99.0_wp, &
                             'CSP V magnitude finite', total_tests, total_failures)
        end if
        call fsps_destroy(ctx)
    end subroutine test_end_to_end

    subroutine setup_ctx(ctx, isoc, spec)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: isoc, spec
        call fsps_create(ctx)
        call fsps_setup(ctx, -1, isoc, spec, 'DL07')
    end subroutine setup_ctx

    subroutine test_afe_selection()
        type(fsps_context_t) :: ref, one, def, sel
        real(WP) :: dt
        integer :: iz, it, m

        call print_group('[alpha/Fe] axis: one value or the default equals the axis-free library')
        call use_root(afe_root)
        call setup_ctx(ref, 'mist', 'miles')
        call setup_ctx(one, 'mist1', 'miles1')
        call setup_ctx(def, 'mist2', 'miles2')
        call setup_ctx(sel, 'mist2:afe=0.4', 'miles2:afe=0.4')

        call assert_true(same(ref, one), 'afe axis of one value: loaded data identical', total_tests, total_failures)
        call assert_true(same(ref, def), 'two afe values, default (afe = 0): identical', total_tests, total_failures)
        call assert_true(all(ref%state%zlegend == sel%state%zlegend) .and. &
                         all(ref%state%nmass_isoc == sel%state%nmass_isoc), &
                         'afe = 0.4 selection: same metallicity grid and isochrone sizes', total_tests, total_failures)
        dt = 0.0_wp
        do iz = 1, ref%state%nz
            do it = 1, ref%state%nt
                do m = 1, ref%state%nmass_isoc(iz, it)
                    dt = max(dt, abs(sel%state%logt_isoc(iz, it, m) - ref%state%logt_isoc(iz, it, m) - 0.01_wp))
                end do
            end do
        end do
        call assert_true(dt < 1.0e-12_wp, 'afe = 0.4 selection: isochrones of the 0.4 block', total_tests, total_failures)
        call assert_true(all(abs(sel%state%speclib - 2.0*ref%state%speclib) <= 1.0e-5*abs(ref%state%speclib) &
                             .or. ref%state%speclib < 1.0e-30), &
                         'afe = 0.4 selection: spectra of the 0.4 block', total_tests, total_failures)

        call fsps_destroy(ref)
        call fsps_destroy(one)
        call fsps_destroy(def)
        call fsps_destroy(sel)
    end subroutine test_afe_selection

    logical function same(a, b)
        type(fsps_context_t), intent(in) :: a, b
        same = all(a%state%zlegend == b%state%zlegend) .and. all(a%state%zlegendinit == b%state%zlegendinit) .and. &
               all(a%state%spec_lambda == b%state%spec_lambda) .and. all(a%state%spec_res == b%state%spec_res) .and. &
               all(a%state%speclib == b%state%speclib) .and. all(a%state%nmass_isoc == b%state%nmass_isoc) .and. &
               all(a%state%mini_isoc == b%state%mini_isoc) .and. all(a%state%logt_isoc == b%state%logt_isoc) .and. &
               all(a%state%logl_isoc == b%state%logl_isoc) .and. all(a%state%lmdot_isoc == b%state%lmdot_isoc) .and. &
               all(a%state%timestep_isoc == b%state%timestep_isoc) .and. a%state%zsol == b%state%zsol
    end function same

end program test_standard_library
