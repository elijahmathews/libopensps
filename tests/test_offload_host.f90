program test_offload_host
    !> What compute_csp_batch does in an offload build whose target regions run on the host (no device, or
    !> OMP_TARGET_OFFLOAD=DISABLED; #97), and that it changes nothing elsewhere: on a device, and in a CPU build,
    !> every call below succeeds. On the host, a forward batch runs (the native HIP kernels take device
    !> pointers, so the band fluxes and the nebular continuum take the OpenMP kernels), and what cannot run is
    !> refused with status 5 instead of returning wrong numbers: any call at fewer than two OpenMP threads
    !> with LLVM flang (libomp crashes in teams distribute parallel do, #87), and, in a build with the native
    !> HIP kernels, the gradients. The numbers of the host run against the CPU build are checked outside this
    !> test (tests/compare_builds). Uses the small test library (FSPS_TEST_LIBRARY); no other data.
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use opensps_precision, only: WP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_ssp, only: generate_ssp_grid
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_set_pset, &
                               opensps_context_get_pset, opensps_context_prepare_pset, &
                               opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use opensps_csp_batch, only: compute_csp_batch, NBGRAD
    use opensps_offload, only: offload_on_host, offload_host_unusable
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    interface
        integer(c_int) function c_setenv(name, value, overwrite) bind(C, name='setenv')
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: name(*), value(*)
            integer(c_int), value :: overwrite
        end function c_setenv
    end interface

    integer, parameter :: N = 6
    integer :: total_tests = 0, total_failures = 0
    character(len=1024) :: root
    type(opensps_context_t), allocatable :: ctx
    type(PARAMS) :: ps(N)
    real(WP), allocatable :: spec_z(:, :, :), mass_z(:, :), lbol_z(:, :), mags(:, :), mbar(:, :), pbar(:, :)
    integer :: nspec, nbands, nt, nz, z, m, st, expect_fwd, expect_grad
    logical :: on_host, hip

    call get_environment_variable('FSPS_TEST_LIBRARY', root)
    if (len_trim(root) == 0) then
        write (*, '(A)') '1..0 # SKIP FSPS_TEST_LIBRARY is not set'
        stop
    end if
    if (c_setenv('OPENSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('FSPS_DATA_HOME'//c_null_char, trim(root)//c_null_char, 1_c_int) /= 0) stop 3
    if (c_setenv('SPS_HOME'//c_null_char, ''//c_null_char, 1_c_int) /= 0) stop 3

    call print_major_header("Offload builds on the host: what runs and what is refused")
    allocate(ctx)
    call opensps_create(ctx)
    call opensps_setup(ctx, 0, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    nspec = ctx%state%nspec
    nbands = ctx%state%nbands
    nt = ctx%state%ntfull
    nz = ctx%state%nz
    allocate(spec_z(nspec, nt, nz), mass_z(nt, nz), lbol_z(nt, nz))
    do z = 1, nz
        call set_int('zmet', z)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, ps(1))
        call opensps_context_set_pset(ctx, ps(1))
        call generate_ssp_grid(ctx, ps(1), mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
    end do
    call opensps_context_update_ssp_basis(ctx, spec_z(:, :, 1:1), mass_z(:, 1:1), lbol_z(:, 1:1), 1)
    call set_int('imf_type', 1)
    call set_int('dust_type', 2)
    call set_int('add_igm_absorption', 1)
    call set_int('add_neb_emission', 0)
    call set_int('compute_indices', 0)
    call set_int('sfh', 1)
    do m = 1, N
        call set_float('tau', 0.5_wp + 0.4_wp * m)
        call set_float('tage', 1.0_wp + 1.1_wp * m)
        call set_float('dust2', 0.1_wp * m)
        call set_float('zred', 0.1_wp * m)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, ps(m))
    end do
    allocate(mags(nbands, N), mbar(nbands, N), pbar(NBGRAD, N))
    mbar = 1.0_wp

    on_host = offload_on_host()
    hip = .false.
#if defined(OPENSPS_HIP)
    hip = .true.
#endif
    expect_fwd = merge(5, 0, offload_host_unusable())
    expect_grad = merge(5, 0, offload_host_unusable() .or. (on_host .and. hip))
    write (*, '(A,L2,A,L2,A,I0,A,I0)') '# target regions on the host:', on_host, '  HIP kernels built:', hip, &
        '  expected status forward / gradient:', expect_fwd, ' /', expect_grad

    call print_group("a forward batch")
    call compute_csp_batch(ctx, ps, mags, st)
    call assert_true(st == expect_fwd, "forward batch: the expected status", total_tests, total_failures)
    if (st == 0) then
        ! (a wrong-numbers regression shows as garbage: NaN, or magnitudes far outside any galaxy's)
        call assert_true(count(mags /= mags) == 0 .and. all(abs(mags) < 100.0_wp), &
                         "forward batch: finite, physical magnitudes", total_tests, total_failures)
    end if

    call print_group("a gradient batch")
    pbar = -1.0_wp
    call compute_csp_batch(ctx, ps, mags, st, mags_bar=mbar, params_bar=pbar)
    call assert_true(st == expect_grad, "gradient batch: the expected status", total_tests, total_failures)
    if (st == 0) then
        call assert_true(count(pbar /= pbar) == 0 .and. any(pbar(1:NBGRAD, :) /= -1.0_wp), &
                         "gradient batch: the gradients were written", total_tests, total_failures)
    else
        call assert_true(all(pbar == -1.0_wp), "refused gradient batch: the gradients were not touched", &
                         total_tests, total_failures)
    end if

    call opensps_destroy(ctx)
    deallocate(ctx)
    call print_summary_line("offload builds on the host", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer :: s
        call opensps_context_set_param_int(ctx, key, value, s)
        if (s /= 0) stop 2
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: s
        call opensps_context_set_param_float(ctx, key, value, s)
        if (s /= 0) stop 2
    end subroutine set_float

end program test_offload_host
