program test_param_regression
    !> Parameterized regression suite: the parity case matrix (tests/data/param_cases.psv, shared
    !> with the libopensps fixture generator) covers IMF types, stellar modifications, every SFH
    !> type, every dust type, nebular emission, IGM, AGN, smoothing and five library combinations.
    !> Each case is compared with a compact reference (tests/data/param_ref/<case>.opsf): the SSP
    !> mass and L_bol at every age, SSP spectra at five ages, the CSP scalars at every output, and
    !> the CSP spectra, magnitudes, indices and lines at up to five outputs.
    !>
    !> Usage:
    !>   test_param_regression CASES.psv REF_DIR [options]
    !> Options:
    !>   --generate          write the references instead of comparing (exit 1 on any mismatch)
    !>   --only PREFIX       run only cases whose name starts with PREFIX (used to shard the suite)
    !>   --skip-spec PREFIX  skip cases whose spectral library starts with PREFIX (e.g. c3k, when
    !>                       a checkout omits that library's data)
    !>
    !> Tolerance: FSPS_TEST_RTOL (default 1e-5), relative to the larger magnitude of the two values,
    !> with an absolute floor of 1e-10 times the array's largest reference value. Magnitudes and
    !> spectral indices are compared absolutely at the same tolerance.
    !>
    !> Each case gets a fresh context. One extra context per library combination is kept alive for
    !> the whole run so that the shared setup cache is not freed between cases.

    use iso_fortran_env, only: int32, real64
    use fsps_precision, only: WP
    use fsps_constants, only: NEMLINE
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
    use fsps_csp, only: compute_csp_scenario
    use fsps_ssp, only: generate_ssp_grid
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset, &
                            fsps_context_set_param_int, fsps_context_set_param_float
    use fsps_context_types, only: fsps_context_t
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    integer, parameter :: MAX_ARRAYS = 32
    integer, parameter :: MAX_LIBRARIES = 16
    integer, parameter :: N_SSP_AGES = 5
    real(WP), parameter :: SSP_LOG_AGES(N_SSP_AGES) = [6.0_wp, 7.0_wp, 8.0_wp, 9.0_wp, 10.0_wp]

    type :: named_array_t
        character(len=64) :: name = ''
        integer, allocatable :: dims(:)
        real(real64), allocatable :: values(:)
    end type named_array_t

    type(named_array_t) :: arrays(MAX_ARRAYS), refs(MAX_ARRAYS)
    integer :: n_arrays, n_refs
    character(len=1024) :: case_file, ref_dir, arg, line
    character(len=256) :: only_prefix = '', skip_spec = ''
    logical :: generate = .false.
    integer :: i_arg, n_skipped
    integer :: unit_cases, ios, n_cases
    real(WP) :: rtol
    ! Contexts are ~2.4 MB each, so they live on the heap rather than the stack
    type(fsps_context_t), allocatable :: pinned(:)
    character(len=128) :: pinned_key(MAX_LIBRARIES) = ''
    integer :: n_pinned = 0
    integer :: total_tests = 0
    integer :: total_failures = 0

    call get_command_argument(1, case_file)
    call get_command_argument(2, ref_dir)
    if (len_trim(case_file) == 0 .or. len_trim(ref_dir) == 0) then
        write(*, *) 'usage: test_param_regression CASES.psv REF_DIR [--generate] [--only PREFIX] [--skip-spec PREFIX]'
        stop 1
    end if
    i_arg = 3
    do while (i_arg <= command_argument_count())
        call get_command_argument(i_arg, arg)
        select case (trim(arg))
        case ('--generate')
            generate = .true.
        case ('--only')
            i_arg = i_arg + 1
            call get_command_argument(i_arg, only_prefix)
        case ('--skip-spec')
            i_arg = i_arg + 1
            call get_command_argument(i_arg, skip_spec)
        case default
            write(*, *) 'ERROR: unknown option ', trim(arg)
            stop 1
        end select
        i_arg = i_arg + 1
    end do
    rtol = env_real('FSPS_TEST_RTOL', 1.0e-5_wp)
    allocate(pinned(MAX_LIBRARIES))

    ! Without references (neither the data release's references/param nor tests/data/param_ref)
    ! the suite is skipped, not failed: probed with the first case of the table
    if (.not. generate) then
        if (.not. references_present()) then
            write(*, '(A)') '1..0 # SKIP regression references not found in ' // trim(ref_dir) // &
                ' (data release library references/param; set OPENSPS_DATA_HOME or -Ddata_home)'
            stop
        end if
    end if

    call print_major_header("Parameterized Regression (parity case matrix)")

    open(newunit=unit_cases, file=trim(case_file), status='old', action='read', iostat=ios)
    if (ios /= 0) then
        write(*, *) 'ERROR: cannot open ', trim(case_file)
        stop 1
    end if
    n_cases = 0
    n_skipped = 0
    do
        read(unit_cases, '(A)', iostat=ios) line
        if (ios /= 0) exit
        if (len_trim(line) == 0 .or. line(1:1) == '#') cycle
        if (.not. selected(trim(line))) then
            n_skipped = n_skipped + 1
            cycle
        end if
        n_cases = n_cases + 1
        call run_case(trim(line))
    end do
    write(*, '(A,I0,A,I0,A)') '# Ran ', n_cases, ' cases (', n_skipped, ' not selected)'
    close(unit_cases)
    do ios = 1, n_pinned
        call fsps_destroy(pinned(ios))
    end do
    deallocate(pinned)

    if (generate) then
        write(*, '(A,I0,2A)') '# Wrote ', n_cases, ' references to ', trim(ref_dir)
    else
        call print_summary_line("Parameterized Regression", total_tests - total_failures, total_tests)
        call tap_finish()
        if (total_failures > 0) stop 1
    end if

contains

    !> True if the reference of the first case of the table exists in ref_dir.
    logical function references_present()
        integer :: u, ios2, k
        character(len=1024) :: l
        logical :: ex
        references_present = .false.
        open(newunit=u, file=trim(case_file), status='old', action='read', iostat=ios2)
        if (ios2 /= 0) return
        do
            read(u, '(A)', iostat=ios2) l
            if (ios2 /= 0) exit
            if (len_trim(l) == 0 .or. l(1:1) == '#') cycle
            k = index(l, '|')
            if (k <= 1) exit
            inquire(file=trim(ref_dir) // '/' // trim(adjustl(l(1:k - 1))) // '.opsf', exist=ex)
            references_present = ex
            exit
        end do
        close(u)
    end function references_present

    !> True if the case passes the --only and --skip-spec filters.
    logical function selected(spec_line)
        character(len=*), intent(in) :: spec_line
        character(len=256) :: fields(6)

        call split_fields(spec_line, fields)
        selected = .true.
        if (len_trim(only_prefix) > 0) selected = index(fields(1), trim(only_prefix)) == 1
        if (len_trim(skip_spec) > 0) then
            if (index(fields(3), trim(skip_spec)) == 1) then
                selected = .false.
                if (len_trim(only_prefix) == 0 .or. index(fields(1), trim(only_prefix)) == 1) &
                    write(*, '(3A)') '# SKIP ', trim(fields(1)), ' (spectral library data not in this checkout)'
            end if
        end if
    end function selected

    !> Runs one case line: name|isoc|spec|dust|flags|key=value ...
    subroutine run_case(spec_line)
        character(len=*), intent(in) :: spec_line
        character(len=256) :: fields(6)
        type(fsps_context_t), allocatable :: ctx
        type(PARAMS) :: pset
        type(COMPSPOUT), allocatable :: results(:)
        real(WP), allocatable :: spec_ssp(:, :), mass_ssp(:), lbol_ssp(:)
        integer :: status, nspec, ntfull, n_out, k, j, i_age(N_SSP_AGES)
        integer, allocatable :: sel(:)
        logical :: csp_only

        call split_fields(spec_line, fields)
        csp_only = index(fields(5), 'csp') > 0
        call pin_libraries(fields(2), fields(3), fields(4))
        allocate(ctx)

        call fsps_create(ctx)
        ctx%pset%zmet = 10
        call apply_overrides(ctx, fields(6))
        call fsps_setup(ctx, ctx%pset%zmet, isoc_type_in=trim(fields(2)), spec_type_in=trim(fields(3)), &
                        dust_type_in=trim(fields(4)))
        call fsps_context_move_to_device(ctx)
        call apply_overrides(ctx, fields(6))
        call fsps_context_prepare_pset(ctx)
        call fsps_context_get_pset(ctx, pset)

        nspec = ctx%state%nspec
        ntfull = ctx%state%ntfull
        allocate(spec_ssp(nspec, ntfull), mass_ssp(ntfull), lbol_ssp(ntfull))
        call fsps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)

        n_arrays = 0
        call put('ssp/mass', mass_ssp, [ntfull])
        call put('ssp/lbol', lbol_ssp, [ntfull])
        if (.not. csp_only) then
            do j = 1, N_SSP_AGES
                i_age(j) = minloc(abs(ctx%state%time_full(1:ntfull) - SSP_LOG_AGES(j)), dim=1)
            end do
            call put('ssp/spec_sel', reshape(spec_ssp(:, i_age), [nspec * N_SSP_AGES]), [nspec, N_SSP_AGES])
        end if

        call fsps_context_update_ssp_basis(ctx, reshape(spec_ssp, [nspec, ntfull, 1]), &
                                           reshape(mass_ssp, [ntfull, 1]), reshape(lbol_ssp, [ntfull, 1]), 1)
        call compute_csp_scenario(ctx, pset, 1, results, status)
        if (status /= 0) then
            write(*, *) 'ERROR: compute_csp_scenario returned status ', status, ' for ', trim(fields(1))
            stop 1
        end if

        n_out = size(results)
        call put('csp/mass', [(results(k)%mass_csp, k = 1, n_out)], [n_out])
        call put('csp/lbol', [(results(k)%lbol_csp, k = 1, n_out)], [n_out])
        call put('csp/sfr', [(results(k)%sfr, k = 1, n_out)], [n_out])
        call put('csp/mdust', [(results(k)%mdust, k = 1, n_out)], [n_out])
        call put('csp/mformed', [(results(k)%mformed, k = 1, n_out)], [n_out])

        ! Spectra and derived quantities at up to N_SSP_AGES outputs
        if (n_out == 1) then
            sel = [1]
        else
            sel = [(minloc(abs(results(:)%age - SSP_LOG_AGES(j)), dim=1), j = 1, N_SSP_AGES)]
        end if
        call put('csp/spec_sel', [(results(sel(k))%spec(1:nspec), k = 1, size(sel))], [nspec, size(sel)])
        call put('csp/mags_sel', [(results(sel(k))%mags(1:ctx%state%nbands), k = 1, size(sel))], &
                 [ctx%state%nbands, size(sel)])
        call put('csp/indx_sel', [(results(sel(k))%indx(1:ctx%state%nindx), k = 1, size(sel))], &
                 [ctx%state%nindx, size(sel)])
        call put('csp/emlines_sel', [(results(sel(k))%emlines(1:NEMLINE), k = 1, size(sel))], [NEMLINE, size(sel)])
        call fsps_destroy(ctx)

        if (generate) then
            call write_opsf(trim(ref_dir) // '/' // trim(fields(1)) // '.opsf')
        else
            call compare_case(trim(fields(1)))
        end if
    end subroutine run_case

    !> Keeps one context per library combination alive so the shared setup cache persists.
    subroutine pin_libraries(isoc, spec, dust)
        character(len=*), intent(in) :: isoc, spec, dust
        character(len=128) :: key
        integer :: i

        key = trim(isoc) // '|' // trim(spec) // '|' // trim(dust)
        do i = 1, n_pinned
            if (pinned_key(i) == key) return
        end do
        if (n_pinned == MAX_LIBRARIES) return
        n_pinned = n_pinned + 1
        pinned_key(n_pinned) = key
        call fsps_create(pinned(n_pinned))
        call fsps_setup(pinned(n_pinned), 10, isoc_type_in=trim(isoc), spec_type_in=trim(spec), &
                        dust_type_in=trim(dust))
    end subroutine pin_libraries

    subroutine compare_case(name)
        character(len=*), intent(in) :: name
        integer :: i, j, k, n_bad
        logical :: absolute
        real(real64) :: floor, diff, worst, a, b

        call print_group(name)
        if (.not. read_opsf(trim(ref_dir) // '/' // name // '.opsf')) then
            call assert_true(.false., "reference file readable", total_tests, total_failures)
            return
        end if

        do i = 1, n_arrays
            j = find_ref(arrays(i)%name)
            if (j == 0 .or. size(arrays(i)%values) /= size(refs(max(j, 1))%values)) then
                call assert_true(.false., trim(arrays(i)%name) // " present with the reference shape", &
                                 total_tests, total_failures)
                cycle
            end if
            absolute = index(arrays(i)%name, 'mags') > 0 .or. index(arrays(i)%name, 'indx') > 0
            floor = 1.0e-10_real64 * maxval(abs(refs(j)%values), mask=refs(j)%values == refs(j)%values)
            n_bad = 0
            worst = 0.0_real64
            do k = 1, size(arrays(i)%values)
                a = arrays(i)%values(k)
                b = refs(j)%values(k)
                if ((a /= a) .neqv. (b /= b)) then
                    n_bad = n_bad + 1
                else if (a == a) then
                    if (absolute) then
                        diff = abs(a - b)
                        if (diff > rtol) n_bad = n_bad + 1
                    else
                        diff = abs(a - b) / max(abs(a), abs(b), floor, tiny(1.0_real64))
                        if (abs(a - b) > floor .and. diff > rtol) n_bad = n_bad + 1
                    end if
                    worst = max(worst, diff)
                end if
            end do
            write(*, '(3A,ES9.2,A,I0,A)') '# ', trim(arrays(i)%name), ": worst ", worst, " (", n_bad, &
                " outside tolerance)"
            call assert_true(n_bad == 0, trim(arrays(i)%name) // " within tolerance", total_tests, total_failures)
        end do
    end subroutine compare_case

    ! ------------------------------------------------------------------------
    ! OPSF containers (same format as generate_opensps_fixtures)
    ! ------------------------------------------------------------------------

    subroutine put(name, values, dims)
        character(len=*), intent(in) :: name
        real(WP), intent(in) :: values(:)
        integer, intent(in) :: dims(:)

        n_arrays = n_arrays + 1
        arrays(n_arrays)%name = name
        arrays(n_arrays)%dims = dims
        arrays(n_arrays)%values = real(values, real64)
    end subroutine put

    subroutine write_opsf(path)
        character(len=*), intent(in) :: path
        integer :: u, i

        open(newunit=u, file=path, status='replace', form='unformatted', access='stream')
        write(u) 'OPSF'
        write(u) int(1, int32)
        do i = 1, n_arrays
            write(u) int(len_trim(arrays(i)%name), int32), trim(arrays(i)%name)
            write(u) int(size(arrays(i)%dims), int32), int(arrays(i)%dims, int32)
            write(u) arrays(i)%values
        end do
        close(u)
    end subroutine write_opsf

    logical function read_opsf(path)
        character(len=*), intent(in) :: path
        integer :: u, ios
        integer(int32) :: version, name_len, ndim
        integer(int32), allocatable :: dims(:)
        character(len=4) :: magic

        read_opsf = .false.
        n_refs = 0
        open(newunit=u, file=path, status='old', form='unformatted', access='stream', iostat=ios)
        if (ios /= 0) return
        read(u, iostat=ios) magic, version
        if (ios /= 0 .or. magic /= 'OPSF') then
            close(u)
            return
        end if
        do
            read(u, iostat=ios) name_len
            if (ios /= 0) exit
            n_refs = n_refs + 1
            refs(n_refs)%name = ''
            read(u) refs(n_refs)%name(1:name_len)
            read(u) ndim
            allocate(dims(ndim))
            read(u) dims
            refs(n_refs)%dims = dims
            if (allocated(refs(n_refs)%values)) deallocate(refs(n_refs)%values)
            allocate(refs(n_refs)%values(product(dims)))
            read(u) refs(n_refs)%values
            deallocate(dims)
        end do
        close(u)
        read_opsf = .true.
    end function read_opsf

    integer function find_ref(name)
        character(len=*), intent(in) :: name
        integer :: j

        find_ref = 0
        do j = 1, n_refs
            if (trim(refs(j)%name) == trim(name)) then
                find_ref = j
                return
            end if
        end do
    end function find_ref

    ! ------------------------------------------------------------------------
    ! Case parsing
    ! ------------------------------------------------------------------------

    subroutine split_fields(text, fields)
        character(len=*), intent(in) :: text
        character(len=*), intent(out) :: fields(:)
        integer :: i, start, bar

        fields = ''
        start = 1
        do i = 1, size(fields)
            bar = index(text(start:), '|')
            if (bar == 0 .or. i == size(fields)) then
                fields(i) = adjustl(text(start:))
                exit
            end if
            fields(i) = adjustl(text(start:start + bar - 2))
            start = start + bar
        end do
    end subroutine split_fields

    !> Applies space-separated key=value pairs, trying the integer setter before the float one.
    subroutine apply_overrides(ctx, text)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: text
        character(len=256) :: token, key, valstr
        integer :: pos, sp, eq, ival, st, ios
        real(WP) :: rval

        pos = 1
        do while (pos <= len_trim(text))
            sp = index(text(pos:), ' ')
            if (sp == 0) then
                token = text(pos:)
                pos = len_trim(text) + 1
            else
                token = text(pos:pos + sp - 2)
                pos = pos + sp
            end if
            if (len_trim(token) == 0) cycle
            eq = index(token, '=')
            key = token(1:eq - 1)
            valstr = token(eq + 1:)
            st = 1
            if (scan(valstr, '.eE') == 0) then
                read(valstr, *, iostat=ios) ival
                if (ios == 0) call fsps_context_set_param_int(ctx, trim(key), ival, st)
            end if
            if (st /= 0) then
                read(valstr, *, iostat=ios) rval
                if (ios == 0) call fsps_context_set_param_float(ctx, trim(key), rval, st)
            end if
            if (st /= 0) then
                write(*, *) 'ERROR: cannot apply ', trim(token)
                stop 1
            end if
        end do
    end subroutine apply_overrides

    real(WP) function env_real(name, default)
        character(len=*), intent(in) :: name
        real(WP), intent(in) :: default
        character(len=64) :: buf
        integer :: stat, ios

        env_real = default
        call get_environment_variable(name, buf, status=stat)
        if (stat == 0 .and. len_trim(buf) > 0) then
            read(buf, *, iostat=ios) env_real
            if (ios /= 0) env_real = default
        end if
    end function env_real

end program test_param_regression
