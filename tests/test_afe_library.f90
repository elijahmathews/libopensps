program test_afe_library
    !> The [alpha/Fe] axis through the whole stack: the stack built by opensps_build_afe_basis, a batch on it
    !> against single-plane contexts at the nodes, the linearity of the spectra between nodes (no dust, nebular
    !> emission or IGM), and the [alpha/Fe] gradient against central differences. Two uses, both through
    !> OPENSPS_AFE_DATA (a data root that holds the libraries and the others of the standard release; skipped
    !> without it):
    !>   - the real libraries (alpha-enhanced MIST v2.5 and C3K-LR from FSPS v4, converted with
    !>     tools/opensps_data/convert_legacy.py --upstream-mist/--upstream-c3k): `isochrones/mist_afe`,
    !>     `stellar_spectra/c3k_lr` (the defaults);
    !>   - the five-node variants derived from the small test library by tools/opensps_data/make_afe_variants.py
    !>     (`mist5`, `miles5`, named by OPENSPS_AFE_ISOC and OPENSPS_AFE_SPEC; tests/run_afe_derived_test.py):
    !>     no external data, the same checks (the library-specific ones aside).
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use opensps_precision, only: WP, KP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy, opensps_build_afe_basis
    use opensps_ssp, only: generate_ssp_grid
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_move_to_device, &
                               opensps_context_set_pset, opensps_context_get_pset, opensps_context_prepare_pset, &
                               opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use opensps_csp_batch, only: compute_csp_batch, BGRAD_AFE, BGRAD_LOGZSOL, NBGRAD
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    interface
        integer(c_int) function c_setenv(name, value, overwrite) bind(C, name='setenv')
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: name(*), value(*)
            integer(c_int), value :: overwrite
        end function c_setenv
    end interface

    integer, parameter :: N_MOD = 6
    real(WP), parameter :: AFE(5) = [-0.2_wp, 0.0_wp, 0.2_wp, 0.4_wp, 0.6_wp]
    ! Models 1-3 on the nodes -0.2, 0.2, 0.6; 4-6 between them
    real(WP), parameter :: AFEV(N_MOD) = [-0.2_wp, 0.2_wp, 0.6_wp, 0.07_wp, 0.31_wp, 0.52_wp]
    ! Metallicities of the models: inside the grid of each kind of library (real: -2.5 to 0.5 or so; derived from
    ! the test library: -1 to 0), models 4-6 away from its nodes
    real(WP), parameter :: LOGZ_REAL(N_MOD) = [-0.9_wp, -0.3_wp, 0.11_wp, 0.12_wp, -1.4_wp, -0.62_wp]
    real(WP), parameter :: LOGZ_SMALL(N_MOD) = [-0.9_wp, -0.3_wp, -0.05_wp, -0.12_wp, -0.7_wp, -0.52_wp]
    real(WP), parameter :: RTOL = merge(1.0e-10_wp, 1.0e-5_wp, KP == WP)
    integer :: total_tests = 0, total_failures = 0
    character(len=1024) :: root
    character(len=64) :: isoc_name, spec_name
    real(WP) :: LOGZ(N_MOD)
    logical :: real_libs
    type(opensps_context_t), allocatable :: ctx
    type(PARAMS) :: psets(N_MOD)
    real(WP), allocatable :: sp_no_neb(:, :)
    real(WP), allocatable :: spec(:, :), mags(:, :), ref_spec(:, :), ref_mags(:, :), spec_n(:, :, :)
    real(WP) :: d
    integer :: status, nz, ia, m, nspec, nbands

    call get_environment_variable('OPENSPS_AFE_DATA', root)
    if (len_trim(root) == 0) then
        write (*, '(A)') '1..0 # SKIP OPENSPS_AFE_DATA is not set (a data root with mist_afe and c3k_lr)'
        stop
    end if
    call get_environment_variable('OPENSPS_AFE_ISOC', isoc_name)
    if (len_trim(isoc_name) == 0) isoc_name = 'mist_afe'
    call get_environment_variable('OPENSPS_AFE_SPEC', spec_name)
    if (len_trim(spec_name) == 0) spec_name = 'c3k_lr'
    real_libs = trim(spec_name) == 'c3k_lr'
    LOGZ = merge(LOGZ_REAL, LOGZ_SMALL, real_libs)
    if (c_setenv('OPENSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('FSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('SPS_HOME'//c_null_char, ''//c_null_char, 1_c_int) /= 0) stop 3

    call print_major_header("[alpha/Fe] axis: "//trim(isoc_name)//" + "//trim(spec_name))

    ! ----- the stack
    call print_group("opensps_build_afe_basis")
    allocate(ctx)
    call opensps_create(ctx)
    ! (nebular emission is read at setup, the nebular grids by the isochrone library's name: the real [alpha/Fe]
    ! libraries have none yet)
    call set_int(ctx, 'add_neb_emission', merge(0, 1, real_libs))
    call opensps_setup(ctx, 0, isoc_type_in=trim(isoc_name)//':afe=0.0', &
                       spec_type_in=trim(spec_name)//':afe=0.0', dust_type_in='DL07')
    call set_int(ctx, 'imf_type', 1)
    call opensps_build_afe_basis(ctx, trim(isoc_name), trim(spec_name), AFE, status)
    call assert_true(status == 0, "status 0", total_tests, total_failures)
    if (status /= 0) stop 1
    nz = ctx%state%nz
    nspec = ctx%state%nspec
    nbands = ctx%state%nbands
    call assert_true(size(ctx%state%ssp_basis_spec, 3) == 5 * nz .and. allocated(ctx%state%afe_basis), &
                     "the stack holds 5 planes of every metallicity", total_tests, total_failures)
    if (real_libs) then
        call assert_true(ctx%state%nlogt == 80 .and. ctx%state%nlogg == 14 .and. nspec == 1936, &
                         "C3K-LR on its native 80 x 14 grid, 1936 pixels", total_tests, total_failures)
    end if

    ! ----- batches: no dust, nebular emission or IGM (the spectra are linear in the stack)
    call set_int(ctx, 'dust_type', 2)
    call set_int(ctx, 'add_igm_absorption', 0)
    call set_int(ctx, 'add_neb_emission', 0)
    call set_int(ctx, 'add_dust_emission', 0)
    call set_int(ctx, 'compute_indices', 0)
    call set_int(ctx, 'sfh', 1)
    call set_float(ctx, 'tau', 1.5_wp)
    call set_float(ctx, 'tage', 7.0_wp)
    call set_float(ctx, 'dust2', 0.0_wp)
    call set_float(ctx, 'dust1', 0.0_wp)
    do m = 1, N_MOD
        call set_float(ctx, 'logzsol', LOGZ(m))
        call set_float(ctx, 'afe', AFEV(m))
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, psets(m))
    end do
    allocate(spec(nspec, N_MOD), mags(nbands, N_MOD), ref_spec(nspec, N_MOD), ref_mags(nbands, N_MOD), &
             spec_n(nspec, N_MOD, 5))
    call compute_csp_batch(ctx, psets, mags, status, spec=spec)
    call assert_true(status == 0, "batch on the stack: status 0", total_tests, total_failures)
    if (status /= 0) stop 1

    ! ----- nodes: a single-plane context at each node (the libraries' slice, the metallicity stack)
    call print_group("a model on a node equals a single-plane context at that node")
    do ia = 1, 5
        call node_batch(AFE(ia), spec_n(:, :, ia))
    end do
    do m = 1, 3
        ia = merge(1, merge(3, 5, m == 2), m == 1)
        d = maxval(abs(spec(:, m) - spec_n(:, m, ia))) / maxval(abs(spec_n(:, m, ia)))
        write (*, '(A,I0,A,ES10.2)') '# model ', m, ': spectrum difference ', d
        call assert_true(d <= RTOL, "model on a node = the single-plane context", total_tests, total_failures)
    end do

    ! ----- between the nodes the spectrum of the (dust-free) batch is the blend of the node spectra
    call print_group("between nodes: the spectrum is the bilinear blend of the nodes' (no dust)")
    do m = 4, 6
        block
            integer :: lo
            real(WP) :: f, ref_d
            lo = max(1, min(count(AFE <= AFEV(m)), 4))
            f = (AFEV(m) - AFE(lo)) / (AFE(lo + 1) - AFE(lo))
            ! The nodes' batches use each model's own metallicity: the blend is in [alpha/Fe] at fixed logzsol
            ref_d = maxval(abs(spec(:, m) - ((1.0_wp - f) * spec_n(:, m, lo) + f * spec_n(:, m, lo + 1)))) / &
                    maxval(abs(spec(:, m)))
            write (*, '(A,I0,A,F5.2,A,ES10.2)') '# model ', m, ' at [alpha/Fe] = ', AFEV(m), ': ', ref_d
            call assert_true(ref_d <= RTOL, "blend of the bracketing nodes", total_tests, total_failures)
        end block
    end do

    ! ----- gradient (FP64 builds: central differences are not resolved in FP32)
    if (KP == WP) then
        call check_gradient(.false.)
        ! the same with nebular emission (derived libraries): the nebular continuum and lines follow the [alpha/Fe] blend of the ionizing
        ! photons, and the reverse pass differentiates through it
        if (.not. real_libs) call check_gradient(.true.)
    end if

    ! ----- the stack rebuilt with other nodes, and what the axis cannot be combined with
    call check_rebuild_and_limits()

    call opensps_destroy(ctx)
    deallocate(ctx)
    call print_summary_line("[alpha/Fe] axis", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    subroutine set_int(c, key, value)
        type(opensps_context_t), intent(inout) :: c
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer :: st
        call opensps_context_set_param_int(c, key, value, st)
        if (st /= 0) stop 2
    end subroutine set_int

    subroutine set_float(c, key, value)
        type(opensps_context_t), intent(inout) :: c
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: st
        call opensps_context_set_param_float(c, key, value, st)
        if (st /= 0) stop 2
    end subroutine set_float

    !> The batch of the models on a context set up at one [alpha/Fe] slice of the libraries, its metallicity
    !> stack the ordinary one (no [alpha/Fe] axis).
    subroutine node_batch(afe_node, spec_out)
        real(WP), intent(in) :: afe_node
        real(WP), intent(out) :: spec_out(:, :)
        type(opensps_context_t), allocatable :: c
        type(PARAMS) :: p, ps(N_MOD)
        real(WP), allocatable :: s(:, :, :), mass(:, :), lbol(:, :), mg(:, :)
        character(len=16) :: sel
        integer :: z, st

        write (sel, '(F8.3)') afe_node
        allocate(c)
        call opensps_create(c)
        call opensps_setup(c, 0, isoc_type_in=trim(isoc_name)//':afe='//trim(adjustl(sel)), &
                           spec_type_in=trim(spec_name)//':afe='//trim(adjustl(sel)), dust_type_in='DL07')
        call set_int(c, 'imf_type', 1)
        allocate(s(nspec, c%state%ntfull, nz), mass(c%state%ntfull, nz), lbol(c%state%ntfull, nz))
        do z = 1, nz
            call set_int(c, 'zmet', z)
            call opensps_context_prepare_pset(c)
            call opensps_context_get_pset(c, p)
            call opensps_context_set_pset(c, p)
            call generate_ssp_grid(c, p, mass(:, z), lbol(:, z), s(:, :, z))
        end do
        call set_int(c, 'zmet', min(10, nz))
        call set_int(c, 'dust_type', 2)
        call set_int(c, 'add_igm_absorption', 0)
        call set_int(c, 'add_neb_emission', 0)
        call set_int(c, 'add_dust_emission', 0)
        call set_int(c, 'compute_indices', 0)
        call set_int(c, 'sfh', 1)
        call set_float(c, 'tau', 1.5_wp)
        call set_float(c, 'tage', 7.0_wp)
        call set_float(c, 'dust2', 0.0_wp)
        call set_float(c, 'dust1', 0.0_wp)
        do z = 1, N_MOD
            call set_float(c, 'logzsol', LOGZ(z))
            call opensps_context_prepare_pset(c)
            call opensps_context_get_pset(c, ps(z))
        end do
        c%state%ssp_basis_is_dirty = .true.
        call opensps_context_update_ssp_basis(c, s, mass, lbol, nz)
        allocate(mg(c%state%nbands, N_MOD))
        call compute_csp_batch(c, ps, mg, st, spec=spec_out)
        if (st /= 0) stop 1
        call opensps_destroy(c)
        deallocate(c)
    end subroutine node_batch

    !> The stack rebuilt with fewer nodes and then with other values (the resident bases follow), and the
    !> combinations the axis does not support: weights over ages and metallicities (ssp_weights_z) give status 2,
    !> a stack of the wrong size status 3.
    subroutine check_rebuild_and_limits()
        real(WP), parameter :: AFE3(3) = [-0.2_wp, 0.2_wp, 0.6_wp], AFE3B(3) = [0.0_wp, 0.4_wp, 0.6_wp]
        real(WP), allocatable :: wz(:, :, :)
        type(PARAMS) :: p1(1)
        real(WP), allocatable :: mg1(:, :)
        integer :: st, nplanes

        call print_group("the stack rebuilt; limits of the axis")
        call opensps_build_afe_basis(ctx, trim(isoc_name), trim(spec_name), AFE3, st)
        call assert_true(st == 0 .and. size(ctx%state%afe_basis) == 3 .and. size(ctx%state%ssp_basis_spec, 3) == 3 * nz, &
                         "rebuilt with 3 nodes: the stack holds 3 planes of every metallicity", total_tests, total_failures)
        call opensps_build_afe_basis(ctx, trim(isoc_name), trim(spec_name), AFE3B, st)
        call assert_true(st == 0 .and. ctx%state%afe_basis(1) == 0.0_wp, &
                         "rebuilt with other values: the nodes follow", total_tests, total_failures)
        nplanes = size(AFE3B)
        ! weights over ages and metallicities with the axis: not supported
        allocate(wz(ctx%state%ntfull, nplanes * nz, 1), mg1(nbands, 1))
        wz = 1.0e-3_wp
        p1(1) = psets(1)
        call compute_csp_batch(ctx, p1, mg1, st, ssp_weights_z=wz)
        call assert_true(st == 2, "ssp_weights_z with the [alpha/Fe] axis: status 2", total_tests, total_failures)
        ! weights whose metallicity count does not match the stack
        deallocate(wz)
        allocate(wz(ctx%state%ntfull, nz + 1, 1))
        wz = 1.0e-3_wp
        call compute_csp_batch(ctx, p1, mg1, st, ssp_weights_z=wz)
        call assert_true(st == 3, "ssp_weights_z of the wrong metallicity count: status 3", total_tests, total_failures)
    end subroutine check_rebuild_and_limits

    !> d L / d afe (and logzsol) against central differences, with dust, for models between the nodes.
    subroutine check_gradient(neb)
        logical, intent(in) :: neb
        type(PARAMS) :: pp(N_MOD)
        real(WP), allocatable :: mb(:, :), sb(:, :), pbar(:, :), mg(:, :), sp(:, :), wbar(:, :)
        real(WP) :: mbr(N_MOD), lbr(N_MOD), ms(N_MOD), lb(N_MOD)
        real(WP) :: lp, lm, g, h, worst_a, worst_z, base
        integer :: st, k, mm

        call print_group("gradients: d L / d afe, d L / d logzsol against central differences (dust, FP64"// &
                         merge(", nebular emission", "                  ", neb)//")")
        call set_int(ctx, 'add_neb_emission', merge(1, 0, neb))
        call set_int(ctx, 'dust_type', 2)
        call set_float(ctx, 'dust2', 0.4_wp)
        call set_float(ctx, 'dust1', 0.3_wp)
        do mm = 1, N_MOD
            call set_float(ctx, 'logzsol', LOGZ(mm))
            call set_float(ctx, 'afe', AFEV(mm))
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, psets(mm))
        end do
        allocate(mb(nbands, N_MOD), sb(nspec, N_MOD), pbar(NBGRAD, N_MOD), mg(nbands, N_MOD), sp(nspec, N_MOD), &
                 wbar(ctx%state%ntfull, N_MOD))
        do mm = 1, N_MOD
            do k = 1, nbands
                mb(k, mm) = cos(0.7_wp * real(k + mm, WP))
            end do
            do k = 1, nspec
                sb(k, mm) = 1.0e-9_wp * sin(0.011_wp * real(k * mm, WP))
            end do
        end do
        ! (the mass and bolometric luminosity outputs enter the loss too, with their cotangents)
        mbr = [(0.3_wp + 0.1_wp * real(mm, WP), mm = 1, N_MOD)]
        lbr = [(0.2_wp - 0.05_wp * real(mm, WP), mm = 1, N_MOD)]
        call compute_csp_batch(ctx, psets, mg, st, spec=sp, mass=ms, lbol=lb, mags_bar=mb, spec_bar=sb, params_bar=pbar, &
                               mass_bar=mbr, lbol_bar=lbr)
        call assert_true(st == 0, "status 0 with gradients", total_tests, total_failures)
        if (st /= 0) return
        if (.not. neb) then
            sp_no_neb = sp
        else
            d = maxval(abs(sp - sp_no_neb)) / maxval(abs(sp))
            write (*, '(A,ES10.2)') '# nebular emission changes the spectra by (relative to the peak): ', d
            call assert_true(d > 1.0e-6_wp, "nebular emission is on (the spectra differ from the dust-only ones)", &
                             total_tests, total_failures)
        end if
        worst_a = 0.0_wp
        worst_z = 0.0_wp
        h = 1.0e-4_wp
        ! (models 4-6: inside the [alpha/Fe] and metallicity intervals, where the function is differentiable)
        do mm = 4, N_MOD
            pp = psets
            pp(mm)%afe = psets(mm)%afe + h
            call loss(pp, mb, sb, mbr, lbr, mg, sp, lp)
            pp(mm)%afe = psets(mm)%afe - h
            call loss(pp, mb, sb, mbr, lbr, mg, sp, lm)
            g = (lp - lm) / (2.0_wp * h)
            worst_a = max(worst_a, abs(pbar(BGRAD_AFE, mm) - g) / max(abs(g), 1.0e-3_wp * maxval(abs(pbar(:, mm)))))
            pp = psets
            pp(mm)%logzsol = psets(mm)%logzsol + h
            call loss(pp, mb, sb, mbr, lbr, mg, sp, lp)
            pp(mm)%logzsol = psets(mm)%logzsol - h
            call loss(pp, mb, sb, mbr, lbr, mg, sp, lm)
            g = (lp - lm) / (2.0_wp * h)
            worst_z = max(worst_z, abs(pbar(BGRAD_LOGZSOL, mm) - g) / max(abs(g), 1.0e-3_wp * maxval(abs(pbar(:, mm)))))
        end do
        base = maxval(abs(pbar(BGRAD_AFE, 4:)))
        write (*, '(A,2ES10.2,A,ES10.2)') '# worst relative difference (afe, logzsol): ', worst_a, worst_z, &
            '; largest d L / d afe ', base
        call assert_true(worst_a <= 1.0e-5_wp, "d L / d afe against finite differences", total_tests, total_failures)
        call assert_true(worst_z <= 1.0e-5_wp, "d L / d logzsol against finite differences", total_tests, total_failures)
    end subroutine check_gradient

    subroutine loss(ps, mb, sb, mbr, lbr, mg, sp, l)
        type(PARAMS), intent(in) :: ps(:)
        real(WP), intent(in) :: mb(:, :), sb(:, :), mbr(:), lbr(:)
        real(WP), intent(inout) :: mg(:, :), sp(:, :)
        real(WP), intent(out) :: l
        real(WP) :: ms(N_MOD), lb(N_MOD)
        integer :: s2

        call compute_csp_batch(ctx, ps, mg, s2, spec=sp, mass=ms, lbol=lb)
        if (s2 /= 0) stop 1
        l = sum(mb * mg) + sum(sb * sp) + sum(mbr * ms) + sum(lbr * lb)
    end subroutine loss

end program test_afe_library
