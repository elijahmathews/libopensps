program imf_batch_dump
    !> Dump batched-SSP (IMF per model) results for a set of extreme IMF parameters: arg1 imf_type, arg2 file,
    !> imf_type -1: the fixed Chabrier IMF with the metallicity free per model (the per-metallicity basis, age_sums_z)
    !> optional arg3 containing 'cue': the cue nebular model (OPENSPS_CUE_DATA; not with a free IMF),
    !> optional arg3 'z': also a redshift per model (0 to 3, 0 included: the observed-frame path with one redshift per model)
    use opensps_precision, only: WP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_csp_batch, only: compute_csp_batch
    use opensps_context, only: opensps_context_move_to_device, opensps_context_set_pset, opensps_context_get_pset, &
                               opensps_context_prepare_pset, opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_build
    use opensps_ssp, only: generate_ssp_grid
    use opensps_context, only: opensps_context_update_ssp_basis
    use opensps_ssp_batch, only: ssp_opset_t, ssp_opset_build
    implicit none
    type(opensps_context_t), allocatable :: ctx
    type(PARAMS) :: pset
    type(PARAMS), allocatable :: psets(:)
    type(ssp_opset_t) :: opset
    type(ssp_operator_t), allocatable :: ops(:)
    real(WP), allocatable :: mags(:,:), spec(:,:), mass(:), lbol(:), mdust(:)
    character(len=256) :: arg, fname, zarg
    integer :: status, itype, i, n, nspec, u, z, ntfull
    real(WP), allocatable :: spec_z(:,:,:), mass_z(:,:), lbol_z(:,:)
    n = 64
    call get_command_argument(1, arg); read(arg, *) itype
    call get_command_argument(2, fname)
    call get_command_argument(3, zarg)
    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 10
    call opensps_setup(ctx, merge(0, 10, itype < 0), isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call opensps_context_move_to_device(ctx)
    call si('imf_type', merge(1, itype, itype < 0)); call si('zmet', 10); call si('sfh', 1)
    call sf('tau', 1.0_wp); call sf('tage', 10.0_wp); call sf('dust2', 0.3_wp); call sf('dust1', 0.5_wp)
    call si('add_neb_emission', 1); call si('add_igm_absorption', 1); call sf('zred', 0.5_wp)
    if (index(zarg, 'cue') > 0) then
        call si('nebular_model', 1); call si('cue_fp64', 0)
    end if
    call si('add_agn_dust', 1); call sf('fagn', 0.1_wp); call si('compute_indices', 0)
    call opensps_context_prepare_pset(ctx)
    call opensps_context_get_pset(ctx, pset)
    nspec = ctx%state%nspec
    if (itype < 0) then
        ! The per-metallicity bases of the fixed IMF (as benchmark_csp_batch): the metallicity free per model
        ntfull = ctx%state%ntfull
        allocate(spec_z(nspec, ntfull, ctx%state%nz), mass_z(ntfull, ctx%state%nz), lbol_z(ntfull, ctx%state%nz))
        call opensps_context_set_pset(ctx, pset)
        do z = 1, ctx%state%nz
            call si('zmet', z)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call opensps_context_set_pset(ctx, pset)
            call generate_ssp_grid(ctx, pset, mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
        end do
        call si('zmet', 10)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        call opensps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, ctx%state%nz)
    end if
    allocate(ops(1))
    if (itype >= 0) then
        call ssp_operator_build(ctx, pset, ops(1))
        call ssp_opset_build(ops, opset)
    end if
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
        if (index(zarg, 'z') > 0) psets(i)%zred = 0.05_wp * mod(i, 61)
        if (itype < 0) psets(i)%logzsol = -1.5_wp + 0.0267_wp * mod(5 * i, 64)
    end do
    if (itype >= 0) then
        call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol, mdust=mdust, ssp_ops=opset)
    else
        call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol, mdust=mdust)
    end if
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
