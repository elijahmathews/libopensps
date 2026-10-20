program benchmark_phases
    !> Timing benchmark matched step for step to legacy FSPS (tools/benchmark_legacy/bench_legacy.f90):
    !> same libraries (MIST + MILES + DL07), parameters and work per call. Spectral indices are off,
    !> since legacy COMPSP only computes them when writing .indx files.
    !> Usage: benchmark_phases [n_ssp [n_csp]]. Prints one line per metric: name, seconds.

    use fsps_precision, only: WP
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
    use fsps_csp, only: compute_csp_scenario
    use fsps_ssp, only: generate_ssp_grid
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset, &
                            fsps_context_set_param_int, fsps_context_set_param_float
    use fsps_context_types, only: fsps_context_t
    implicit none

    type(fsps_context_t), allocatable :: ctx
    type(PARAMS) :: pset
    type(COMPSPOUT), allocatable :: results(:)
    real(WP), allocatable :: spec_ssp(:, :), mass_ssp(:), lbol_ssp(:)
    character(len=32) :: arg
    integer(8) :: t0, t1, rate
    integer :: i, n_ssp = 5, n_csp = 200, status, nspec, ntfull
    real(WP) :: t_setup, t_first, total

    call get_command_argument(1, arg)
    if (len_trim(arg) > 0) read(arg, *) n_ssp
    call get_command_argument(2, arg)
    if (len_trim(arg) > 0) read(arg, *) n_csp
    call system_clock(t0, rate)

    ! Common parameters: Chabrier, solar MIST, tau model observed at 10 Gyr, dust + dust emission
    allocate(ctx)
    call fsps_create(ctx)
    ctx%pset%zmet = 10
    call fsps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call fsps_context_move_to_device(ctx)
    call set_int('imf_type', 1)
    call set_int('zmet', 10)
    call set_int('sfh', 1)
    call set_float('tau', 1.0_wp)
    call set_float('tage', 10.0_wp)
    call set_float('dust2', 0.3_wp)
    call set_int('compute_indices', 0)
    call fsps_context_prepare_pset(ctx)
    call fsps_context_get_pset(ctx, pset)
    call system_clock(t1)
    t_setup = real(t1 - t0, WP) / rate

    ! 1. Time to first spectrum: setup + SSP grid + one CSP
    nspec = ctx%state%nspec
    ntfull = ctx%state%ntfull
    allocate(spec_ssp(nspec, ntfull), mass_ssp(ntfull), lbol_ssp(ntfull))
    call ssp()
    call basis()
    call csp()
    call system_clock(t1)
    t_first = real(t1 - t0, WP) / rate
    write(*, '(A,F12.6)') 'setup_s ', t_setup
    write(*, '(A,F12.6)') 'first_spectrum_s ', t_first

    ! 2. Per SSP call (full SSP grid regeneration)
    call system_clock(t0)
    do i = 1, n_ssp
        call ssp()
    end do
    call system_clock(t1)
    write(*, '(A,F12.6)') 'ssp_call_s ', real(t1 - t0, WP) / rate / n_ssp
    call basis()

    ! 3. Per CSP call (SSP basis cached; tau and dust vary)
    total = 0.0_wp
    call system_clock(t0)
    do i = 1, n_csp
        pset%tau = 0.5_wp + 0.01_wp * mod(i, 100)
        pset%dust2 = 0.1_wp + 0.002_wp * mod(i, 100)
        call csp()
        total = total + results(1)%spec(1000)
    end do
    call system_clock(t1)
    write(*, '(A,F12.6)') 'csp_call_s ', real(t1 - t0, WP) / rate / n_csp

    ! 4. Nebular emission: per SSP call and per CSP call (gas parameters vary)
    call set_int('add_neb_emission', 1)
    call fsps_context_prepare_pset(ctx)
    call system_clock(t0)
    do i = 1, n_ssp
        call ssp()
    end do
    call system_clock(t1)
    write(*, '(A,F12.6)') 'ssp_call_nebular_s ', real(t1 - t0, WP) / rate / n_ssp
    call basis()
    call system_clock(t0)
    do i = 1, n_csp
        pset%tau = 0.5_wp + 0.01_wp * mod(i, 100)
        pset%gas_logu = -3.0_wp + 0.01_wp * mod(i, 100)
        call csp()
        total = total + results(1)%spec(1000)
    end do
    call system_clock(t1)
    write(*, '(A,F12.6)') 'csp_call_nebular_s ', real(t1 - t0, WP) / rate / n_csp
    if (total < 0.0_wp) write(*, *) total   ! keep the results live

    call fsps_destroy(ctx)
    deallocate(ctx)

contains

    subroutine ssp()
        call fsps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
    end subroutine ssp

    subroutine basis()
        call fsps_context_update_ssp_basis(ctx, reshape(spec_ssp, [nspec, ntfull, 1]), &
                                           reshape(mass_ssp, [ntfull, 1]), reshape(lbol_ssp, [ntfull, 1]), 1)
    end subroutine basis

    subroutine csp()
        call compute_csp_scenario(ctx, pset, 1, results, status)
        if (status /= 0) stop 1
    end subroutine csp

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        call fsps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) stop 1
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        call fsps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) stop 1
    end subroutine set_float

end program benchmark_phases
