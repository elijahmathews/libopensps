program imf_batch_dump
    !> Dump batched-SSP (IMF per model) results for a set of extreme IMF parameters: arg1 imf_type, arg2 file
    use opensps_precision, only: WP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_csp_batch, only: compute_csp_batch
    use opensps_context, only: opensps_context_move_to_device, opensps_context_set_pset, opensps_context_get_pset, &
                               opensps_context_prepare_pset, opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_build
    implicit none
    type(opensps_context_t), allocatable :: ctx
    type(PARAMS) :: pset
    type(PARAMS), allocatable :: psets(:)
    type(ssp_operator_t) :: op
    real(WP), allocatable :: mags(:,:), spec(:,:), mass(:), lbol(:), mdust(:)
    character(len=256) :: arg, fname
    integer :: status, itype, i, n, nspec, u
    n = 64
    call get_command_argument(1, arg); read(arg, *) itype
    call get_command_argument(2, fname)
    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 10
    call opensps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call opensps_context_move_to_device(ctx)
    call si('imf_type', itype); call si('zmet', 10); call si('sfh', 1)
    call sf('tau', 1.0_wp); call sf('tage', 10.0_wp); call sf('dust2', 0.3_wp); call sf('dust1', 0.5_wp)
    call si('add_neb_emission', 1); call si('add_igm_absorption', 1); call sf('zred', 0.5_wp)
    call si('add_agn_dust', 1); call sf('fagn', 0.1_wp); call si('compute_indices', 0)
    call opensps_context_prepare_pset(ctx)
    call opensps_context_get_pset(ctx, pset)
    nspec = ctx%state%nspec
    call ssp_operator_build(ctx, pset, op)
    allocate(psets(n), mags(ctx%state%nbands, n), spec(nspec, n), mass(n), lbol(n), mdust(n))
    do i = 1, n
        psets(i) = pset
        psets(i)%tage = 0.5_wp + 0.2_wp * i
        psets(i)%tau = 0.3_wp + 0.05_wp * mod(i, 17)
        ! extremes: slopes 0.3 .. 3.9; vdmc and mdave varied
        psets(i)%imf1 = 0.3_wp + 0.057_wp * mod(7 * i, 64)
        psets(i)%imf2 = 0.3_wp + 0.057_wp * mod(11 * i, 64)
        psets(i)%imf3 = 0.3_wp + 0.057_wp * mod(13 * i, 64)
        psets(i)%vdmc = 0.02_wp + 0.004_wp * mod(i, 40)
        psets(i)%mdave = 0.1_wp + 0.03_wp * mod(i, 40)
    end do
    call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol, mdust=mdust, ssp_op=op)
    write(*, *) 'status', status
    if (status /= 0) stop 1
    open(newunit=u, file=trim(fname), form='unformatted', access='stream', status='replace')
    write(u) nspec, n, ctx%state%nbands
    write(u) spec, mags, mass, lbol
    close(u)
    call opensps_destroy(ctx)
contains
    subroutine si(k, v)
        character(len=*), intent(in) :: k
        integer, intent(in) :: v
        call opensps_context_set_param_int(ctx, k, v, status)
        if (status /= 0) stop 1
    end subroutine
    subroutine sf(k, v)
        character(len=*), intent(in) :: k
        real(WP), intent(in) :: v
        call opensps_context_set_param_float(ctx, k, v, status)
        if (status /= 0) stop 1
    end subroutine
end program imf_batch_dump
