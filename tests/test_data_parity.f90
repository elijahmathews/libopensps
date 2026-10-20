program test_data_parity
    !> Bit-for-bit parity between the legacy data layout and the standard format.
    !>
    !> Usage: test_data_parity LEGACY_ROOT STANDARD_ROOT [--skip-spec NAME ...]
    !>
    !> For each library combination, sets up one context from LEGACY_ROOT (FSPS_DATA_FORMAT=legacy)
    !> and one from STANDARD_ROOT (FSPS_DATA_FORMAT=standard), then requires every array the
    !> setup loads to be identical (values compared exactly; NaN matches NaN), and the SSP grid
    !> and a CSP (spectrum, magnitudes, indices, line luminosities) to be identical too.
    !> Skipped when STANDARD_ROOT has no opensps-data.json (or the older fsps-data.json).
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use fsps_precision, only: WP, KP
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy, fsps_set_param_int, fsps_set_param_float
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset
    use fsps_context_types, only: fsps_context_t
    use fsps_ssp, only: generate_ssp_grid
    use fsps_csp, only: compute_csp_scenario
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
    character(len=1024) :: legacy_root, std_root, arg
    character(len=64) :: skip_spec(8)
    integer :: nskip, i
    logical :: exists

    if (command_argument_count() < 2) then
        write (*, '(A)') 'usage: test_data_parity LEGACY_ROOT STANDARD_ROOT [--skip-spec NAME ...]'
        stop 2
    end if
    call get_command_argument(1, legacy_root)
    call get_command_argument(2, std_root)
    nskip = 0
    i = 3
    do while (i <= command_argument_count())
        call get_command_argument(i, arg)
        if (trim(arg) == '--skip-spec' .and. i < command_argument_count()) then
            nskip = nskip + 1
            call get_command_argument(i + 1, skip_spec(nskip))
            i = i + 2
        else
            i = i + 1
        end if
    end do
    inquire (file=trim(std_root)//'/opensps-data.json', exist=exists)
    if (.not. exists) inquire (file=trim(std_root)//'/fsps-data.json', exist=exists)
    if (.not. exists) then
        write (*, '(A)') '1..0 # SKIP no standard data root at '//trim(std_root)//' (tools/fsps_data/convert_legacy.py)'
        stop
    end if
    ! OPENSPS_DATA_HOME and SPS_HOME are searched before FSPS_DATA_HOME: set the first, clear the second
    if (c_setenv('SPS_HOME'//c_null_char, ''//c_null_char, 1_c_int) /= 0) stop 3

    call print_major_header('Data format parity: legacy layout vs standard format')

    call check_case('mist', 'miles', 'DL07', 10, full=.true.)
    call check_case('pdva', 'basel', 'THEMIS', 20, full=.true.)
    call check_case('prsc', 'miles', 'DL07', 12, full=.false.)
    call check_case('bsti', 'basel', 'DL07', 8, full=.false.)
    call check_case('gnva', 'basel', 'DL07', 4, full=.false.)
    if (.not. skipped('c3k_afe+0.0')) call check_case('mist', 'c3k_afe+0.0', 'DL07', 10, full=.false.)
    call check_case('bpss', 'bpass', 'DL07', 10, full=.true.)

    call print_summary_line('Data format parity', total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    logical function skipped(name)
        character(len=*), intent(in) :: name
        integer :: k
        skipped = .false.
        do k = 1, nskip
            if (trim(skip_spec(k)) == name) skipped = .true.
        end do
    end function skipped

    subroutine use_root(root, fmt)
        character(len=*), intent(in) :: root, fmt
        if (c_setenv('OPENSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
        if (c_setenv('FSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
        if (c_setenv('FSPS_DATA_FORMAT'//c_null_char, trim(fmt)//c_null_char, 1_c_int) /= 0) stop 3
    end subroutine use_root

    subroutine make_context(ctx, root, fmt, isoc, spec, dust, zmet, full)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: root, fmt, isoc, spec, dust
        integer, intent(in) :: zmet
        logical, intent(in) :: full
        integer :: status

        call use_root(root, fmt)
        call fsps_create(ctx)
        call fsps_set_param_int(ctx, 'add_neb_emission', 1, status)
        if (full) then
            call fsps_set_param_int(ctx, 'add_agb_dust_model', 1, status)
            call fsps_set_param_int(ctx, 'add_dust_emission', 1, status)
            call fsps_set_param_int(ctx, 'add_xrb_emission', 1, status)
            call fsps_set_param_int(ctx, 'cloudy_dust', 1, status)
        end if
        call fsps_setup(ctx, -1, isoc, spec, dust)
        call fsps_context_move_to_device(ctx)
        call fsps_set_param_int(ctx, 'zmet', zmet, status)
        call fsps_set_param_int(ctx, 'sfh', 1, status)
        call fsps_set_param_int(ctx, 'dust_type', 0, status)
        call fsps_set_param_float(ctx, 'tau', 1.0_wp, status)
        call fsps_set_param_float(ctx, 'tage', 3.0_wp, status)
        call fsps_set_param_float(ctx, 'dust2', 0.3_wp, status)
        if (full) then
            call fsps_set_param_float(ctx, 'fagn', 0.1_wp, status)
            call fsps_set_param_int(ctx, 'add_agn_dust', 1, status)
        end if
    end subroutine make_context

    subroutine check_case(isoc, spec, dust, zmet, full)
        character(len=*), intent(in) :: isoc, spec, dust
        integer, intent(in) :: zmet
        logical, intent(in) :: full
        type(fsps_context_t) :: a, b
        type(COMPSPOUT), allocatable :: ra(:), rb(:)
        real(WP), allocatable :: ma(:), la(:), sa(:, :), mb(:), lb(:), sb(:, :)
        integer :: k

        call print_group(isoc//' + '//spec//' + '//dust)
        call make_context(a, legacy_root, 'legacy', isoc, spec, dust, zmet, full)
        call make_context(b, std_root, 'standard', isoc, spec, dust, zmet, full)
        call assert_true(trim(a%sps_home) /= trim(b%sps_home), 'contexts read different roots', &
                         total_tests, total_failures)

        associate (x => a%state, y => b%state)
            call assert_true(x%nspec == y%nspec .and. x%nt == y%nt .and. x%nz == y%nz .and. &
                             x%nzinit == y%nzinit .and. x%nbands == y%nbands .and. &
                             x%zsol == y%zsol .and. x%zsol_spec == y%zsol_spec, &
                             'dimensions and solar metallicities', total_tests, total_failures)
            call cmp_wp('spec_lambda', x%spec_lambda, y%spec_lambda, &
                size(x%spec_lambda), size(y%spec_lambda))
            call cmp_wp('spec_nu', x%spec_nu, y%spec_nu, &
                size(x%spec_nu), size(y%spec_nu))
            call cmp_wp('spec_res', x%spec_res, y%spec_res, &
                size(x%spec_res), size(y%spec_res))
            call cmp_wp('zlegend', x%zlegend, y%zlegend, &
                size(x%zlegend), size(y%zlegend))
            call cmp_wp('zlegendinit', x%zlegendinit, y%zlegendinit, &
                size(x%zlegendinit), size(y%zlegendinit))
            call cmp_wp('time_full', x%time_full, y%time_full, &
                size(x%time_full), size(y%time_full))
            call cmp_wp('speclib_logt', x%speclib_logt, y%speclib_logt, &
                size(x%speclib_logt), size(y%speclib_logt))
            call cmp_wp('speclib_logg', x%speclib_logg, y%speclib_logg, &
                size(x%speclib_logg), size(y%speclib_logg))
            call cmp_r4('speclib', x%speclib, y%speclib, &
                size(x%speclib), size(y%speclib))
            call cmp_wp('mini_isoc', x%mini_isoc, y%mini_isoc, &
                size(x%mini_isoc), size(y%mini_isoc))
            call cmp_wp('mact_isoc', x%mact_isoc, y%mact_isoc, &
                size(x%mact_isoc), size(y%mact_isoc))
            call cmp_wp('logl_isoc', x%logl_isoc, y%logl_isoc, &
                size(x%logl_isoc), size(y%logl_isoc))
            call cmp_wp('logt_isoc', x%logt_isoc, y%logt_isoc, &
                size(x%logt_isoc), size(y%logt_isoc))
            call cmp_wp('logg_isoc', x%logg_isoc, y%logg_isoc, &
                size(x%logg_isoc), size(y%logg_isoc))
            call cmp_wp('ffco_isoc', x%ffco_isoc, y%ffco_isoc, &
                size(x%ffco_isoc), size(y%ffco_isoc))
            call cmp_wp('phase_isoc', x%phase_isoc, y%phase_isoc, &
                size(x%phase_isoc), size(y%phase_isoc))
            call cmp_wp('lmdot_isoc', x%lmdot_isoc, y%lmdot_isoc, &
                size(x%lmdot_isoc), size(y%lmdot_isoc))
            call cmp_wp('timestep_isoc', x%timestep_isoc, y%timestep_isoc, &
                size(x%timestep_isoc), size(y%timestep_isoc))
            call assert_true(all(x%nmass_isoc == y%nmass_isoc), 'nmass_isoc', total_tests, total_failures)
            call cmp_wp('bpass_spec_ssp', x%bpass_spec_ssp, y%bpass_spec_ssp, &
                size(x%bpass_spec_ssp), size(y%bpass_spec_ssp))
            call cmp_wp('bpass_mass_ssp', x%bpass_mass_ssp, y%bpass_mass_ssp, &
                size(x%bpass_mass_ssp), size(y%bpass_mass_ssp))
            call cmp_wp('wmb_logt', x%wmb_logt, y%wmb_logt, &
                size(x%wmb_logt), size(y%wmb_logt))
            call cmp_wp('wmb_logg', x%wmb_logg, y%wmb_logg, &
                size(x%wmb_logg), size(y%wmb_logg))
            call cmp_r4('wmb_spec', x%wmb_spec, y%wmb_spec, &
                size(x%wmb_spec), size(y%wmb_spec))
            call cmp_kp('agb_spec_o', x%agb_spec_o, y%agb_spec_o, &
                size(x%agb_spec_o), size(y%agb_spec_o))
            call cmp_kp('agb_spec_c', x%agb_spec_c, y%agb_spec_c, &
                size(x%agb_spec_c), size(y%agb_spec_c))
            call cmp_kp('agb_spec_car', x%agb_spec_car, y%agb_spec_car, &
                size(x%agb_spec_car), size(y%agb_spec_car))
            call cmp_wp('agb_logt_o', x%agb_logt_o, y%agb_logt_o, &
                size(x%agb_logt_o), size(y%agb_logt_o))
            call cmp_wp('agb_logt_c', x%agb_logt_c, y%agb_logt_c, &
                size(x%agb_logt_c), size(y%agb_logt_c))
            call cmp_wp('agb_logt_car', x%agb_logt_car, y%agb_logt_car, &
                size(x%agb_logt_car), size(y%agb_logt_car))
            call cmp_kp('pagb_spec', x%pagb_spec, y%pagb_spec, &
                size(x%pagb_spec), size(y%pagb_spec))
            call cmp_wp('pagb_logt', x%pagb_logt, y%pagb_logt, &
                size(x%pagb_logt), size(y%pagb_logt))
            call cmp_kp('wrn_spec', x%wrn_spec, y%wrn_spec, &
                size(x%wrn_spec), size(y%wrn_spec))
            call cmp_kp('wrc_spec', x%wrc_spec, y%wrc_spec, &
                size(x%wrc_spec), size(y%wrc_spec))
            call cmp_wp('wrn_logt', x%wrn_logt, y%wrn_logt, &
                size(x%wrn_logt), size(y%wrn_logt))
            call cmp_wp('wrc_logt', x%wrc_logt, y%wrc_logt, &
                size(x%wrc_logt), size(y%wrc_logt))
            call cmp_wp('qpaharr', x%qpaharr, y%qpaharr, &
                size(x%qpaharr), size(y%qpaharr))
            call cmp_wp('uminarr', x%uminarr, y%uminarr, &
                size(x%uminarr), size(y%uminarr))
            call cmp_wp('dustem2_dustem', x%dustem2_dustem, y%dustem2_dustem, &
                size(x%dustem2_dustem), size(y%dustem2_dustem))
            call cmp_kp('flux_dagb', x%flux_dagb, y%flux_dagb, &
                size(x%flux_dagb), size(y%flux_dagb))
            call cmp_wp('tau1_dagb', x%tau1_dagb, y%tau1_dagb, &
                size(x%tau1_dagb), size(y%tau1_dagb))
            call cmp_wp('teff_dagb', x%teff_dagb, y%teff_dagb, &
                size(x%teff_dagb), size(y%teff_dagb))
            call cmp_wp('nebem_line_pos', x%nebem_line_pos, y%nebem_line_pos, &
                size(x%nebem_line_pos), size(y%nebem_line_pos))
            call cmp_wp('nebem_line', x%nebem_line, y%nebem_line, &
                size(x%nebem_line), size(y%nebem_line))
            call cmp_wp('xnebem_line', x%xnebem_line, y%xnebem_line, &
                size(x%xnebem_line), size(y%xnebem_line))
            call cmp_wp('nebem_cont', x%nebem_cont, y%nebem_cont, &
                size(x%nebem_cont), size(y%nebem_cont))
            call cmp_wp('xnebem_cont', x%xnebem_cont, y%xnebem_cont, &
                size(x%xnebem_cont), size(y%xnebem_cont))
            call cmp_wp('nebem_logz', x%nebem_logz, y%nebem_logz, &
                size(x%nebem_logz), size(y%nebem_logz))
            call cmp_wp('nebem_age', x%nebem_age, y%nebem_age, &
                size(x%nebem_age), size(y%nebem_age))
            call cmp_wp('nebem_logu', x%nebem_logu, y%nebem_logu, &
                size(x%nebem_logu), size(y%nebem_logu))
            call cmp_kp('gaussnebarr', x%gaussnebarr, y%gaussnebarr, &
                size(x%gaussnebarr), size(y%gaussnebarr))
            call cmp_wp('agndust_tau', x%agndust_tau, y%agndust_tau, &
                size(x%agndust_tau), size(y%agndust_tau))
            call cmp_wp('agndust_spec', x%agndust_spec, y%agndust_spec, &
                size(x%agndust_spec), size(y%agndust_spec))
            call cmp_wp('bands', x%bands, y%bands, &
                size(x%bands), size(y%bands))
            call cmp_wp('filter_leff', x%filter_leff, y%filter_leff, &
                size(x%filter_leff), size(y%filter_leff))
            call cmp_wp('magsun', x%magsun, y%magsun, &
                size(x%magsun), size(y%magsun))
            call cmp_wp('magvega', x%magvega, y%magvega, &
                size(x%magvega), size(y%magvega))
            call cmp_wp('vega_spec', x%vega_spec, y%vega_spec, &
                size(x%vega_spec), size(y%vega_spec))
            call cmp_wp('sun_spec', x%sun_spec, y%sun_spec, &
                size(x%sun_spec), size(y%sun_spec))
            call cmp_wp('indexdefined', x%indexdefined, y%indexdefined, &
                size(x%indexdefined), size(y%indexdefined))
            call cmp_wp('wgdust', x%wgdust, y%wgdust, &
                size(x%wgdust), size(y%wgdust))
            call cmp_wp('g03smcextn', x%g03smcextn, y%g03smcextn, &
                size(x%g03smcextn), size(y%g03smcextn))
            call cmp_wp('spec_xrb', x%spec_xrb, y%spec_xrb, &
                size(x%spec_xrb), size(y%spec_xrb))
            call cmp_wp('lam_xrb', x%lam_xrb, y%lam_xrb, &
                size(x%lam_xrb), size(y%lam_xrb))
            call cmp_wp('ages_xrb', x%ages_xrb, y%ages_xrb, &
                size(x%ages_xrb), size(y%ages_xrb))
            call cmp_wp('zmet_xrb', x%zmet_xrb, y%zmet_xrb, &
                size(x%zmet_xrb), size(y%zmet_xrb))
            if (associated(x%speclib_agb_smooth) .or. associated(y%speclib_agb_smooth)) &
                call cmp_kp('speclib_agb_smooth', x%speclib_agb_smooth, y%speclib_agb_smooth, &
                size(x%speclib_agb_smooth), size(y%speclib_agb_smooth))
        end associate

        call run_csp(a, ma, la, sa, ra)
        call run_csp(b, mb, lb, sb, rb)
        call cmp_wp('SSP mass', ma, mb, &
                size(ma), size(mb))
        call cmp_wp('SSP lbol', la, lb, &
                size(la), size(lb))
        call cmp_wp('SSP spectra', sa, sb, &
                size(sa), size(sb))
        do k = 1, min(size(ra), size(rb))
            call cmp_wp('CSP spectrum', ra(k)%spec, rb(k)%spec, &
                size(ra(k)%spec), size(rb(k)%spec))
            call cmp_wp('CSP magnitudes', ra(k)%mags, rb(k)%mags, &
                size(ra(k)%mags), size(rb(k)%mags))
            if (allocated(ra(k)%indx)) call cmp_wp('CSP indices', ra(k)%indx, rb(k)%indx, &
                size(ra(k)%indx), size(rb(k)%indx))
            if (allocated(ra(k)%emlines)) call cmp_wp('CSP lines', ra(k)%emlines, rb(k)%emlines, &
                size(ra(k)%emlines), size(rb(k)%emlines))
            call assert_true(ra(k)%mass_csp == rb(k)%mass_csp .and. ra(k)%lbol_csp == rb(k)%lbol_csp, &
                             'CSP mass and luminosity', total_tests, total_failures)
        end do

        call fsps_destroy(a)
        call fsps_destroy(b)
    end subroutine check_case

    subroutine run_csp(ctx, mass_ssp, lbol_ssp, spec_ssp, results)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), allocatable, intent(inout) :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)
        type(COMPSPOUT), allocatable, intent(inout) :: results(:)
        type(PARAMS) :: pset
        integer :: status

        call fsps_context_prepare_pset(ctx)
        call fsps_context_get_pset(ctx, pset)
        if (allocated(mass_ssp)) deallocate (mass_ssp, lbol_ssp, spec_ssp)
        allocate (mass_ssp(ctx%state%ntfull), lbol_ssp(ctx%state%ntfull), &
                  spec_ssp(ctx%state%nspec, ctx%state%ntfull))
        call fsps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
        call fsps_context_update_ssp_basis(ctx, reshape(spec_ssp, [size(spec_ssp, 1), size(spec_ssp, 2), 1]), &
                                           reshape(mass_ssp, [size(mass_ssp), 1]), &
                                           reshape(lbol_ssp, [size(lbol_ssp), 1]), 1)
        if (allocated(results)) deallocate (results)
        call compute_csp_scenario(ctx, pset, 1, results, status)
        if (status /= 0) then
            write (*, *) 'ERROR: compute_csp_scenario returned status ', status
            stop 1
        end if
    end subroutine run_csp

    ! Exact comparisons through explicit-shape dummies (any rank, contiguous arrays)
    subroutine cmp_wp(label, x, y, nx, ny)
        character(len=*), intent(in) :: label
        integer, intent(in) :: nx, ny
        real(WP), intent(in) :: x(*), y(*)
        if (nx /= ny) then
            call report(label, .false., nx, 0)
        else
            call report(label, .true., nx, same_wp(x, y, nx))
        end if
    end subroutine cmp_wp

    subroutine cmp_kp(label, x, y, nx, ny)
        character(len=*), intent(in) :: label
        integer, intent(in) :: nx, ny
        real(KP), intent(in) :: x(*), y(*)
        if (nx /= ny) then
            call report(label, .false., nx, 0)
        else
            call report(label, .true., nx, same_kp(x, y, nx))
        end if
    end subroutine cmp_kp

    subroutine cmp_r4(label, x, y, nx, ny)
        character(len=*), intent(in) :: label
        integer, intent(in) :: nx, ny
        real(kind(1.0)), intent(in) :: x(*), y(*)
        if (nx /= ny) then
            call report(label, .false., nx, 0)
        else
            call report(label, .true., nx, same_r4(x, y, nx))
        end if
    end subroutine cmp_r4

    subroutine report(label, same_size, n, ndiff)
        character(len=*), intent(in) :: label
        logical, intent(in) :: same_size
        integer, intent(in) :: n, ndiff
        character(len=96) :: msg
        if (.not. same_size) then
            call assert_true(.false., label//': sizes differ', total_tests, total_failures)
            return
        end if
        if (ndiff == 0) then
            write (msg, '(A,": ",I0," values identical")') label, n
        else
            write (msg, '(A,": ",I0," of ",I0," values differ")') label, ndiff, n
        end if
        call assert_true(ndiff == 0, trim(msg), total_tests, total_failures)
    end subroutine report

    integer function same_wp(x, y, n)
        integer, intent(in) :: n
        real(WP), intent(in) :: x(n), y(n)
        same_wp = count(.not. (x == y .or. (x /= x .and. y /= y)))
    end function same_wp

    integer function same_kp(x, y, n)
        integer, intent(in) :: n
        real(KP), intent(in) :: x(n), y(n)
        same_kp = count(.not. (x == y .or. (x /= x .and. y /= y)))
    end function same_kp

    integer function same_r4(x, y, n)
        integer, intent(in) :: n
        real(kind(1.0)), intent(in) :: x(n), y(n)
        same_r4 = count(.not. (x == y .or. (x /= x .and. y /= y)))
    end function same_r4

end program test_data_parity
