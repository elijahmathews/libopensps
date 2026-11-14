!> Where the OpenMP target regions of an offload build actually run.
!>
!> An offload build (`-Daccel=omp_gpu`) whose target regions fall back to the host (no usable device, or
!> `OMP_TARGET_OFFLOAD=DISABLED`) is not the build it was made to be: the native HIP kernels of an AMD
!> build receive host pointers where they expect device pointers (the band fluxes were never written:
!> wrong magnitudes, #97), and in an LLVM flang build libomp crashes in `teams distribute parallel do` at a
!> single OpenMP thread (#87). `offload_on_host` tells which case a process is in, once, so that the code
!> can take the host-safe kernel or refuse with a status instead of returning wrong numbers.
module opensps_offload
#ifdef OMP_TARGET
    use omp_lib, only: omp_is_initial_device, omp_get_max_threads
#endif
    implicit none
    private

    public :: offload_on_host, offload_host_unusable

    integer, save :: probe_state = -1   ! -1 not probed, 0 target regions run on a device, 1 on the host

contains

    !> True when the target regions of this process run on the host. Probed once with a one-thread target
    !> region (omp_is_initial_device); always false in builds without OMP_TARGET.
    logical function offload_on_host() result(on_host)
        logical :: host

        on_host = .false.
#ifdef OMP_TARGET
        if (probe_state < 0) then
            !$omp critical (opensps_offload_probe)
            if (probe_state < 0) then
                host = .true.
                !$omp target map(from: host)
                host = omp_is_initial_device()
                !$omp end target
                probe_state = merge(1, 0, host)
            end if
            !$omp end critical (opensps_offload_probe)
        end if
        on_host = probe_state == 1
#endif
    end function offload_on_host

    !> True when the batched kernels cannot run in this process: an LLVM flang offload build whose target
    !> regions run on the host with fewer than two OpenMP threads (libomp crashes there, #87).
    logical function offload_host_unusable() result(unusable)
        unusable = .false.
#if defined(OMP_TARGET) && defined(__flang__)
        unusable = offload_on_host() .and. omp_get_max_threads() < 2
#endif
    end function offload_host_unusable

end module opensps_offload
