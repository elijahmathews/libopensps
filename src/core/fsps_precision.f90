module fsps_precision
    !> @brief
    !> Defines the precision parameter for FSPS.
    !>
    !> @details
    !> This module serves as the root dependency for precision definitions (WP)
    !> and is used by almost all other modules.

    use, intrinsic :: iso_fortran_env, only: real32, real64

    implicit none
    private

    public :: WP, KP, kahan_add

! Default to double (64-bit) unless the flag is set.
#ifdef FORCE_SINGLE_PRECISION
    integer, parameter :: WP = real32
#else
    integer, parameter :: WP = real64
#endif

    !> Kernel precision: the per-pixel arithmetic of the expensive kernels (stellar templates,
    !> SSP accumulation) and the arrays they read. Setup, isochrones and weights stay in WP.
    !> FP64 by default; -DKERNEL_SINGLE_PRECISION selects FP32, the precision that consumer and
    !> workstation GPUs run at full rate.
#ifdef KERNEL_SINGLE_PRECISION
    integer, parameter :: KP = real32
#else
    integer, parameter :: KP = WP
#endif

contains

    !> Compensated (Kahan) accumulation at kernel precision: keeps long FP32 sums (over stars,
    !> pixels) at FP32 rounding instead of losing ~N*eps. A plain sum when KP is WP. Needs a compiler that does not
    !> reassociate floating point (no -ffast-math); GPU reductions use pairwise trees instead.
    pure subroutine kahan_add(acc, comp, x)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(KP), intent(inout) :: acc, comp
        real(KP), intent(in) :: x
        real(KP) :: term, tsum
        if (KP == WP) then   ! compile-time constant: the full-precision build keeps a plain sum
            acc = acc + x
            return
        end if
        term = x - comp
        tsum = acc + term
        comp = (tsum - acc) - term
        acc = tsum
    end subroutine kahan_add

end module fsps_precision
