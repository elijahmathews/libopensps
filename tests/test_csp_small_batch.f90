program test_csp_small_batch
    !> compute_csp_batch with fewer than 16 models, the batches whose SFH weights and mass sums run on the
    !> host (HOST_SFH_MAX), at a single OpenMP thread (the test's meson environment sets OMP_NUM_THREADS=1): in
    !> an LLVM flang offload build a "teams distribute parallel do" with a false if clause crashed in libomp at
    !> one thread (#87). Each of the models of a 1-, 2- and 15-model batch equals the same model inside a
    !> 17-model batch (the device path in an offload build), from the SFH parameters, from the weight matrix and
    !> from weights over ages and metallicities. Uses the small test library (FSPS_TEST_LIBRARY); no other data.
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use opensps_precision, only: WP, KP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_csp, only: sfh_weights_core
    use opensps_ssp, only: generate_ssp_grid
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_set_pset, &
                               opensps_context_get_pset, opensps_context_prepare_pset, &
                               opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use opensps_csp_batch, only: compute_csp_batch
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    interface
        integer(c_int) function c_setenv(name, value, overwrite) bind(C, name='setenv')
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: name(*), value(*)
            integer(c_int), value :: overwrite
        end function c_setenv
    end interface

    integer, parameter :: NBIG = 17
    real(WP), parameter :: RTOL = merge(1.0e-10_wp, 1.0e-5_wp, KP == WP)
    integer :: total_tests = 0, total_failures = 0
    character(len=1024) :: root
    type(opensps_context_t), allocatable :: ctx
    type(PARAMS) :: pbig(NBIG)
    real(WP), allocatable :: spec_big(:, :), mags_big(:, :), mass_big(:), lbol_big(:)
    real(WP), allocatable :: wmat(:, :), wz_big(:, :, :), spec_z(:, :, :), mass_z(:, :), lbol_z(:, :)
    real(WP), allocatable :: t1(:), t2(:), w1(:, :)
    integer :: nspec, nbands, nt, nz, m, z, status, n

    call get_environment_variable('FSPS_TEST_LIBRARY', root)
    if (len_trim(root) == 0) then
        write (*, '(A)') '1..0 # SKIP FSPS_TEST_LIBRARY is not set'
        stop
    end if
    if (c_setenv('OPENSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('FSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('SPS_HOME'//c_null_char, ''//c_null_char, 1_c_int) /= 0) stop 3

    call print_major_header("compute_csp_batch with fewer than HOST_SFH_MAX models")
    allocate(ctx)
    call opensps_create(ctx)
    call opensps_setup(ctx, 0, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call set_int('imf_type', 1)
    call set_int('dust_type', 2)
    call set_int('add_igm_absorption', 0)
    call set_int('add_neb_emission', 0)
    call set_int('compute_indices', 0)
    nspec = ctx%state%nspec
    nbands = ctx%state%nbands
    nt = ctx%state%ntfull
    nz = ctx%state%nz
    allocate(spec_z(nspec, nt, nz), mass_z(nt, nz), lbol_z(nt, nz))
    do z = 1, nz
        call set_int('zmet', z)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pbig(1))
        call opensps_context_set_pset(ctx, pbig(1))
        call generate_ssp_grid(ctx, pbig(1), mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
    end do
    call opensps_context_update_ssp_basis(ctx, spec_z(:, :, 1:1), mass_z(:, 1:1), lbol_z(:, 1:1), 1)

    ! the models: exponential SFHs of several ages and e-folding times, with dust
    do m = 1, NBIG
        call set_int('sfh', 1)
        call set_float('tau', 0.4_wp + 0.3_wp * m)
        call set_float('tage', 1.0_wp + 0.7_wp * m)
        call set_float('dust2', 0.03_wp * mod(m, 7))
        call set_float('dust1', 0.02_wp * mod(m, 5))
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pbig(m))
    end do
    allocate(spec_big(nspec, NBIG), mags_big(nbands, NBIG), mass_big(NBIG), lbol_big(NBIG))
    call compute_csp_batch(ctx, pbig, mags_big, status, spec=spec_big, mass=mass_big, lbol=lbol_big)
    call assert_true(status == 0, "17 models: status 0", total_tests, total_failures)
    if (status /= 0) stop 1

    call print_group("SFH parameters")
    do n = 1, 3
        call small(merge(1, merge(2, 15, n == 2), n == 1), 'p')
    end do

    call print_group("the weight matrix")
    allocate(wmat(nt, NBIG), t1(nt), t2(nt), w1(nt, 1))
    do m = 1, NBIG
        call sfh_weights_core(ctx, pbig(m), pbig(m)%tage, 1, w1, t1, t2)
        wmat(:, m) = w1(:, 1)
    end do
    where (.not. (wmat > 0.0_wp)) wmat = 0.0_wp
    call compute_csp_batch(ctx, pbig, mags_big, status, spec=spec_big, mass=mass_big, lbol=lbol_big, ssp_weights=wmat)
    call assert_true(status == 0, "17 models, weight matrix: status 0", total_tests, total_failures)
    if (status /= 0) stop 1
    do n = 1, 3
        call small(merge(1, merge(2, 15, n == 2), n == 1), 'w')
    end do

    if (nz >= 2) then
        call print_group("weights over ages and metallicities")
        call opensps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, nz)
        allocate(wz_big(nt, nz, NBIG))
        wz_big = 0.0_wp
        do m = 1, NBIG
            wz_big(:, 1 + mod(m, nz), m) = wmat(:, m)
            wz_big(:, 1 + mod(m + 1, nz), m) = 0.5_wp * wmat(:, m)
        end do
        call compute_csp_batch(ctx, pbig, mags_big, status, spec=spec_big, mass=mass_big, lbol=lbol_big, &
                               ssp_weights_z=wz_big)
        call assert_true(status == 0, "17 models, weights over Z: status 0", total_tests, total_failures)
        if (status /= 0) stop 1
        do n = 1, 3
            call small(merge(1, merge(2, 15, n == 2), n == 1), 'z')
        end do
    end if

    call opensps_destroy(ctx)
    deallocate(ctx)
    deallocate(spec_big, mags_big, mass_big, lbol_big, wmat, t1, t2, w1, spec_z, mass_z, lbol_z)
    if (allocated(wz_big)) deallocate(wz_big)
    call print_summary_line("compute_csp_batch, small batches", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer :: st
        call opensps_context_set_param_int(ctx, key, value, st)
        if (st /= 0) stop 2
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: st
        call opensps_context_set_param_float(ctx, key, value, st)
        if (st /= 0) stop 2
    end subroutine set_float

    !> The first nsm models (a batch of nsm) against the same models in the batch of NBIG; kind 'p' from the
    !> parameters, 'w' with the weight matrix, 'z' with weights over metallicities.
    subroutine small(nsm, kind)
        integer, intent(in) :: nsm
        character(len=1), intent(in) :: kind
        real(WP), allocatable :: spec(:, :), mags(:, :), mass(:), lbol(:)
        real(WP) :: d_spec, d_mag, d_ml
        integer :: st

        allocate(spec(nspec, nsm), mags(nbands, nsm), mass(nsm), lbol(nsm))
        select case (kind)
        case ('w')
            call compute_csp_batch(ctx, pbig(1:nsm), mags, st, spec=spec, mass=mass, lbol=lbol, ssp_weights=wmat(:, 1:nsm))
        case ('z')
            call compute_csp_batch(ctx, pbig(1:nsm), mags, st, spec=spec, mass=mass, lbol=lbol, &
                                   ssp_weights_z=wz_big(:, :, 1:nsm))
        case default
            call compute_csp_batch(ctx, pbig(1:nsm), mags, st, spec=spec, mass=mass, lbol=lbol)
        end select
        call assert_true(st == 0, "status 0", total_tests, total_failures)
        if (st /= 0) return
        d_spec = maxval(abs(spec - spec_big(:, 1:nsm))) / maxval(abs(spec_big(:, 1:nsm)))
        d_mag = maxval(abs(mags - mags_big(:, 1:nsm)))
        d_ml = max(maxval(abs(mass / mass_big(1:nsm) - 1.0_wp)), maxval(abs(lbol - lbol_big(1:nsm))))
        write (*, '(A,I0,A,3ES10.2)') '# ', nsm, ' models: spectrum, magnitude, mass / L_bol differences ', d_spec, d_mag, d_ml
        call assert_true(d_spec <= RTOL .and. d_mag <= 10.0_wp * RTOL .and. d_ml <= 10.0_wp * RTOL, &
                         "a small batch equals the same models in a batch of 17", total_tests, total_failures)
    end subroutine small

end program test_csp_small_batch
