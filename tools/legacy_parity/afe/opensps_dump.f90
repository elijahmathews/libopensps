! This fork's counterpart of legacy_dump.f90: the SSP at the five [alpha/Fe] nodes of one
! metallicity index (each node is its own single-[alpha/Fe] library pair, as in legacy).
!
!   opensps_dump ZMET [EVTYPE [ADD_AGB_DUST]]
!
! Needs OPENSPS_DATA_HOME to point at a data root with the mist_afe and c3k_lr libraries.
! The options legacy v4 defaults to are set: use_lw_tpagb = 0, logt_wmb_hot = 10.
program opensps_dump
    use opensps_precision, only: WP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_context, only: opensps_context_set_pset, opensps_context_prepare_pset, opensps_context_get_pset, &
                               opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use opensps_ssp, only: generate_ssp_grid
    implicit none
    type(opensps_context_t), allocatable :: ctx
    type(PARAMS) :: pset
    real(WP), allocatable :: spec(:, :), mass(:), lbol(:), all(:, :, :)
    character(len=16) :: av(5) = [character(len=16) :: '-0.2', '0.0', '0.2', '0.4', '0.6']
    character(len=100) :: fn, arg
    integer :: ia, status, zmet, ev, agbdust, nspec, ntfull
    call get_command_argument(1, arg); read(arg, *) zmet
    ev = -1; agbdust = 1
    if (command_argument_count() >= 2) then; call get_command_argument(2, arg); read(arg, *) ev; end if
    if (command_argument_count() >= 3) then; call get_command_argument(3, arg); read(arg, *) agbdust; end if
    do ia = 1, 5
        allocate(ctx)
        call opensps_create(ctx)
        ctx%pset%zmet = zmet
        call opensps_setup(ctx, zmet, isoc_type_in='mist_afe:afe='//trim(av(ia)), &
                           spec_type_in='c3k_lr:afe='//trim(av(ia)), dust_type_in='DL07')
        call opensps_context_set_param_int(ctx, 'imf_type', 1, status)
        call opensps_context_set_param_int(ctx, 'zmet', zmet, status)
        call opensps_context_set_param_int(ctx, 'evtype', ev, status)
        call opensps_context_set_param_int(ctx, 'add_agb_dust_model', agbdust, status)
        call opensps_context_set_param_int(ctx, 'use_lw_tpagb', 0, status)
        call opensps_context_set_param_float(ctx, 'logt_wmb_hot', 10.0_wp, status)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        nspec = ctx%state%nspec; ntfull = ctx%state%ntfull
        if (.not. allocated(all)) allocate(all(nspec, ntfull, 5))
        allocate(spec(nspec, ntfull), mass(ntfull), lbol(ntfull))
        call opensps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass, lbol, spec)
        all(:, :, ia) = spec
        deallocate(spec, mass, lbol)
        call opensps_destroy(ctx)
        deallocate(ctx)
    end do
    write(fn, '(A,I2.2,A,I0,A)') 'opensps_z', zmet, '_ev', ev, '.bin'
    open(12, file=trim(fn), form='unformatted', access='stream', status='replace')
    write(12) int(ntfull), int(nspec)
    write(12) all
    close(12)
end program opensps_dump
