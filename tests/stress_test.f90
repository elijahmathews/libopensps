program stress_test
    use fsps_precision, only: WP
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy, fsps_prepare_pset, fsps_compute_ssp, fsps_compute_csp
    use fsps_context_types, only: fsps_context_t
#ifdef _OPENMP
    use omp_lib, only: omp_set_num_threads
#endif

    implicit none

    integer, parameter :: DEFAULT_ITERS = 20
    integer, parameter :: DEFAULT_THREADS = 1

    integer :: iterations
    integer :: num_threads
    character(len=16) :: phases

    call read_env_int('FSPS_STRESS_ITERS', DEFAULT_ITERS, iterations)
    call read_env_int('OMP_NUM_THREADS', DEFAULT_THREADS, num_threads)
    num_threads = max(1, num_threads)

    ! FSPS_STRESS_PHASES selects phases by letter (default: all of ABCD)
    phases = 'ABCD'
    call get_environment_variable('FSPS_STRESS_PHASES', value=phases)
    if (len_trim(phases) == 0) phases = 'ABCD'

    write (*, *) '========================================='
    write (*, *) 'FSPS MEMORY + CONCURRENCY STRESS TEST'
    write (*, *) '========================================='
    write (*, *) 'FSPS_STRESS_ITERS = ', iterations
    write (*, *) 'num_threads       = ', num_threads
    write (*, *) 'phases            = ', trim(phases)

    if (index(phases, 'A') > 0) then
        call set_omp_threads(1)
        call phase_a_leak_check(iterations)
    end if
    if (index(phases, 'B') > 0) then
        call set_omp_threads(num_threads)
        call phase_b_concurrency_check(num_threads)
    end if
    if (index(phases, 'C') > 0) then
        call set_omp_threads(1)
        call phase_c_reallocation_thrash(iterations)
    end if
    if (index(phases, 'D') > 0) then
        call set_omp_threads(num_threads)
        call phase_d_setup_release_churn(num_threads, max(1, iterations / 4))
    end if

    write (*, *) 'Stress test PASS'

contains

    subroutine phase_a_leak_check(iter_count)
        integer, intent(in) :: iter_count

        type(fsps_context_t) :: ctx
        integer :: iter_idx

        write (*, *) '[Phase A] Leak check: create -> compute CSP -> destroy'
        do iter_idx = 1, iter_count
            call fsps_create(ctx)
            call run_lightweight_csp(ctx, iter_idx)
            call fsps_destroy(ctx)
        end do
    end subroutine phase_a_leak_check

    subroutine phase_b_concurrency_check(thread_count)
        integer, intent(in) :: thread_count

        type(fsps_context_t), allocatable :: contexts(:)
        integer :: thread_idx

        write (*, *) '[Phase B] Concurrency check: per-thread context SSP+CSP'

        allocate (contexts(thread_count))

        !$omp parallel do default(shared) private(thread_idx) schedule(static)
        do thread_idx = 1, thread_count
            call fsps_create(contexts(thread_idx))
            call run_heavy_workload(contexts(thread_idx), thread_idx)
            call fsps_destroy(contexts(thread_idx))
        end do
        !$omp end parallel do

        deallocate (contexts)
    end subroutine phase_b_concurrency_check

    subroutine phase_c_reallocation_thrash(iter_count)
        integer, intent(in) :: iter_count

        type(fsps_context_t) :: ctx
        integer :: iter_idx

        write (*, *) '[Phase C] Reallocation thrash via parameter mutations'

        call fsps_create(ctx)
        call fsps_setup(ctx, 1, 'mist', 'miles', 'DL07')
        call fsps_prepare_pset(ctx)

        do iter_idx = 1, iter_count
            call mutate_parameters(ctx, iter_idx)
            call execute_ssp_and_csp(ctx)
        end do

        call fsps_destroy(ctx)
    end subroutine phase_c_reallocation_thrash

    !> Every thread repeatedly creates, sets up and destroys a context with the same library
    !> selection, so setups (cache lookup, refcount increment, loaders writing the shared entry) and
    !> releases (refcount decrement, freeing the entry) of one shared cache entry interleave across
    !> threads. Phase B overlaps only one setup and one destroy per thread.
    subroutine phase_d_setup_release_churn(thread_count, iter_count)
        integer, intent(in) :: thread_count, iter_count

        type(fsps_context_t), allocatable :: contexts(:)
        integer :: thread_idx, iter_idx

        write (*, *) '[Phase D] Shared setup-cache churn: concurrent setup/destroy'

        allocate (contexts(thread_count))

        !$omp parallel do default(shared) private(thread_idx, iter_idx) schedule(static)
        do thread_idx = 1, thread_count
            do iter_idx = 1, iter_count
                call fsps_create(contexts(thread_idx))
                call fsps_setup(contexts(thread_idx), 1, 'mist', 'miles', 'DL07')
                call fsps_destroy(contexts(thread_idx))
            end do
        end do
        !$omp end parallel do

        deallocate (contexts)
    end subroutine phase_d_setup_release_churn

    subroutine run_lightweight_csp(ctx, iter_idx)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: iter_idx
        type(PARAMS) :: pset

        call fsps_setup(ctx, 1, 'mist', 'miles', 'DL07')
        call fsps_prepare_pset(ctx)

        pset = ctx%pset
        pset%zmet = 1
        pset%sfh = 1
        pset%tau = 2.0_wp
        pset%const = 0.1_wp
        pset%dust1 = 0.2_wp
        pset%dust2 = 0.2_wp
        pset%dust3 = 0.0_wp
        pset%tage = 0.0_wp
        pset%compute_mags = 1
        pset%compute_indices = 0

        ctx%imf_type_val = 1
        if (mod(iter_idx, 2) == 0) then
            ctx%add_neb_emission_val = 1
        else
            ctx%add_neb_emission_val = 0
        end if
        ctx%pset = pset

        call execute_ssp_and_csp(ctx)
    end subroutine run_lightweight_csp

    subroutine run_heavy_workload(ctx, thread_idx)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: thread_idx
        type(PARAMS) :: pset

        call fsps_setup(ctx, 1, 'mist', 'miles', 'DL07')
        call fsps_prepare_pset(ctx)

        pset = ctx%pset
        pset%zmet = 1
        pset%sfh = 1
        pset%tau = 2.0_wp
        pset%const = 0.2_wp
        pset%dust1 = 1.0_wp
        pset%dust2 = 0.3_wp
        pset%dust3 = 0.0_wp
        pset%zred = 0.0_wp
        pset%compute_mags = 1
        pset%compute_indices = 0

        ctx%imf_type_val = 1
        ctx%add_neb_emission_val = 1
        ctx%add_neb_continuum_val = 1
        ctx%add_dust_emission_val = 1
        ctx%add_agn_dust_val = 0
        ctx%pset = pset

        call execute_ssp_and_csp(ctx)
    end subroutine run_heavy_workload

    subroutine mutate_parameters(ctx, iter_idx)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: iter_idx
        type(PARAMS) :: pset

        pset = ctx%pset
        pset%zmet = 1
        pset%tage = 0.0_wp
        pset%compute_mags = 1
        pset%compute_indices = 0

        select case (mod(iter_idx - 1, 4))
        case (0)
            pset%sfh = 0
            pset%tau = 1.0_wp
            pset%const = 0.0_wp
            pset%fburst = 0.0_wp
            pset%dust1 = 0.0_wp
            pset%dust2 = 0.0_wp
            pset%dust3 = 0.0_wp
            pset%duste_gamma = 0.01_wp
            pset%duste_qpah = 2.5_wp
            pset%zred = 0.0_wp
        case (1)
            pset%sfh = 1
            pset%tau = 1.5_wp
            pset%const = 0.25_wp
            pset%fburst = 0.1_wp
            pset%tburst = 2.0_wp
            pset%dust1 = 0.7_wp
            pset%dust2 = 0.3_wp
            pset%dust3 = 0.0_wp
            pset%duste_gamma = 0.08_wp
            pset%duste_qpah = 4.0_wp
            pset%zred = 0.0_wp
        case (2)
            pset%sfh = 1
            pset%tau = 0.8_wp
            pset%const = 0.45_wp
            pset%fburst = 0.15_wp
            pset%tburst = 5.0_wp
            pset%dust1 = 1.2_wp
            pset%dust2 = 0.9_wp
            pset%dust3 = 0.0_wp
            pset%duste_gamma = 0.2_wp
            pset%duste_qpah = 6.0_wp
            pset%zred = 0.0_wp
        case default
            pset%sfh = 1
            pset%tau = 2.5_wp
            pset%const = 0.6_wp
            pset%fburst = 0.25_wp
            pset%tburst = 9.0_wp
            pset%dust1 = 1.8_wp
            pset%dust2 = 1.4_wp
            pset%dust3 = 0.0_wp
            pset%duste_gamma = 0.4_wp
            pset%duste_qpah = 1.0_wp
            pset%zred = 0.0_wp
        end select

        pset%imf1 = 1.3_wp + 0.05_wp*real(mod(iter_idx, 4), WP)
        pset%imf2 = 2.2_wp + 0.1_wp*real(mod(iter_idx, 3), WP)
        pset%imf3 = 2.3_wp + 0.1_wp*real(mod(iter_idx + 1, 4), WP)

        ctx%imf_type_val = 1
        ctx%add_neb_emission_val = merge(1, 0, mod(iter_idx, 2) == 0)
        ctx%add_dust_emission_val = 1
        ctx%add_agn_dust_val = 0
        ctx%pset = pset
    end subroutine mutate_parameters

    subroutine execute_ssp_and_csp(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        integer :: ntfull
        integer :: nspec
        real(WP), allocatable :: mass_ssp(:)
        real(WP), allocatable :: lbol_ssp(:)
        real(WP), allocatable :: spec_ssp(:,:)
        real(WP), allocatable :: mass_ssp_2d(:,:)
        real(WP), allocatable :: lbol_ssp_2d(:,:)
        real(WP), allocatable :: spec_ssp_3d(:,:,:)
        type(COMPSPOUT), allocatable :: csp_results(:)

        ntfull = ctx%state%ntfull
        nspec = ctx%state%nspec

        allocate (mass_ssp(ntfull), lbol_ssp(ntfull), spec_ssp(nspec, ntfull))
        call fsps_compute_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)

        allocate (mass_ssp_2d(ntfull, 1), lbol_ssp_2d(ntfull, 1), spec_ssp_3d(nspec, ntfull, 1))
        mass_ssp_2d(:, 1) = mass_ssp
        lbol_ssp_2d(:, 1) = lbol_ssp
        spec_ssp_3d(:, :, 1) = spec_ssp

        allocate (csp_results(ntfull))
        call fsps_compute_csp(ctx, 0, 1, '', mass_ssp_2d, lbol_ssp_2d, spec_ssp_3d, csp_results)
        call release_csp_results(csp_results)

        deallocate (mass_ssp, lbol_ssp, spec_ssp)
        deallocate (mass_ssp_2d, lbol_ssp_2d, spec_ssp_3d)
    end subroutine execute_ssp_and_csp

    subroutine release_csp_results(results)
        type(COMPSPOUT), allocatable, intent(inout) :: results(:)

        integer :: result_idx

        if (.not. allocated(results)) return

        do result_idx = 1, size(results)
            if (allocated(results(result_idx)%mags)) deallocate (results(result_idx)%mags)
            if (allocated(results(result_idx)%spec)) deallocate (results(result_idx)%spec)
            if (allocated(results(result_idx)%indx)) deallocate (results(result_idx)%indx)
            if (allocated(results(result_idx)%emlines)) deallocate (results(result_idx)%emlines)
        end do

        deallocate (results)
    end subroutine release_csp_results

    subroutine read_env_int(name, default_value, output_value)
        character(len=*), intent(in) :: name
        integer, intent(in) :: default_value
        integer, intent(out) :: output_value

        character(len=64) :: env_text
        integer :: env_status
        integer :: read_status

        output_value = default_value
        env_text = ''

        call get_environment_variable(trim(name), value=env_text, status=env_status)
        if (env_status /= 0) return

        read (env_text, *, iostat=read_status) output_value
        if (read_status /= 0) output_value = default_value
    end subroutine read_env_int

    subroutine set_omp_threads(thread_count)
        integer, intent(in) :: thread_count

#ifdef _OPENMP
        call omp_set_num_threads(max(1, thread_count))
#else
        if (thread_count < 0) then
            stop 1
        end if
#endif
    end subroutine set_omp_threads

end program stress_test