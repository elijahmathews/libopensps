!> Hardware-level tuning parameters: the one place that holds them.
!>
!> The batched kernels carry sizes whose best value depends on the machine: how many models one chunk
!> holds, below which batch size the SFH weights are formed on the host, the pixels per block of a
!> split sum. They used to be literals scattered through the sources, each with a comment on the machine
!> it was measured on. They live here now, in two kinds:
!>
!> * Compile-time defaults (`TUNE_*` named constants): values that fix an array extent, a loop trip count
!>   or the order of a floating point sum inside a device kernel. Changing one of the summation
!>   parameters (`TUNE_NCH`, `TUNE_NCHG`, `TUNE_BAND_BLOCK`, `TUNE_PIX_BLOCK`, `TUNE_SEG_LEN`,
!>   `TUNE_PT_PIECE`) changes the rounding of the results, so each is a constant of the build that the
!>   tests of both precisions gate; none is overridable at run time.
!> * Run-time parameters (`tuning_get`): launch and chunking decisions that do not change what is
!>   computed beyond the gates (`TUNE_P_*` ids). Each starts at its default, may be set by the
!>   environment variable `OPENSPS_TUNE_<NAME>` (read once, at the first use), and may be changed by
!>   `tuning_set` (tests, programs that tune themselves). A value outside the legal range of its
!>   parameter, or not an integer, is refused with one message on stderr and the default is kept.
!>
!> `tuning_report` lists every parameter with its value, where the value came from and its range.
!> Device-specific profiles (a table by architecture) and a measuring tuner are the next steps;
!> every consumer already reads through this module, so they will not touch the kernels.
module opensps_tuning
    use, intrinsic :: iso_fortran_env, only: error_unit
    implicit none
    private

    ! ---- Compile-time defaults -------------------------------------------------------------------
    !> Entries per segment of an effective template's list (ssp_operator_t: seg_ptr). Measured on the A100
    !> only (#46: to sweep on other devices). Changes the summation order.
    integer, parameter, public :: TUNE_SEG_LEN = 256
    !> Points per piece of an age's points (ssp_operator_t: pc_ptr). A100-tuned guess (#46).
    !> Changes the summation order.
    integer, parameter, public :: TUNE_PT_PIECE = 32
    !> Pixels per block of the SSP operator and SSP band-flux sums (plain sum per block, blocks added in
    !> FP64). Changes the summation order.
    integer, parameter, public :: TUNE_PIX_BLOCK = 512
    !> Pixels per block of the split-K band product: each block is a plain FP32 sum of at most this many
    !> positive terms. Changes the summation order.
    integer, parameter, public :: TUNE_BAND_BLOCK = 256
    !> Fewest columns for which the BLAS band product is used instead of the compensated kernel
    !> (OPENSPS_ACCEL_BLAS builds). Changes which sum is taken.
    integer, parameter, public :: TUNE_GEMM_MIN_COLS = 16
    !> Partial sums per model of the band-derivative and nebular adjoint reductions: each takes every
    !> NCH-th pixel. Changes the summation order.
    integer, parameter, public :: TUNE_NCH = 64
    !> Pixel chunks per model of the attenuation adjoint (each thread takes every NCHG-th pixel).
    !> Changes the summation order.
    integer, parameter, public :: TUNE_NCHG = 64
    !> Models per gang of the band-flux kernel (the threads of a gang share one band: one trip count).
    integer, parameter, public :: TUNE_MBLK = 128
    !> Largest batch for which the band fluxes use a workgroup per band (small batches).
    integer, parameter, public :: TUNE_SMALL_N = 16
    !> Ages per thread of the tiled nebular continuum.
    integer, parameter, public :: TUNE_CSP_PT = 4

    ! ---- Run-time parameters ---------------------------------------------------------------------
    !> Models per chunk of the batched SSP.
    integer, parameter, public :: TUNE_P_SSP_BATCH_CHUNK = 1
    !> Below this batch size the SFH weights are computed on the host: on the device each model's SFH
    !> integral is one serial thread (~0.3 ms on the A100 and the P100, ~20 us per model on the host).
    !> The crossover depends on the host core and the GPU's single-thread FP64 speed (A100 node).
    !> 0 sends every batch to the device.
    integer, parameter, public :: TUNE_P_HOST_SFH_MAX = 2
    !> Bytes per chunk of the pinned host-device copies of a batch (OPENSPS_PINNED offload builds):
    !> ~16 MiB measured best on the A100 node, PCIe 3: large enough to reach the DMA rate, small enough to
    !> overlap most of the conversion.
    integer, parameter, public :: TUNE_P_PINNED_CHUNK_BYTES = 3

    integer, parameter :: N_TUNE = 3
    integer, parameter :: NAME_LEN = 24

    character(len=NAME_LEN), parameter :: names(N_TUNE) = [character(len=NAME_LEN) :: &
        'SSP_BATCH_CHUNK', 'HOST_SFH_MAX', 'PINNED_CHUNK_BYTES']
    integer, parameter :: defaults(N_TUNE) = [2048, 16, 16 * 1024 * 1024]
    integer, parameter :: lo_limit(N_TUNE) = [1, 0, 65536]
    integer, parameter :: hi_limit(N_TUNE) = [1000000, 1000000, 1024 * 1024 * 1024]

    integer, save :: values(N_TUNE) = defaults
    !> 0 default, 1 environment, 2 set by tuning_set
    integer, save :: source(N_TUNE) = 0
    logical, save :: env_read = .false.

    public :: tuning_get, tuning_set, tuning_reset, tuning_report, tuning_count, tuning_name, &
              tuning_parse, tuning_in_range

contains

    !> Number of run-time parameters.
    pure integer function tuning_count()
        tuning_count = N_TUNE
    end function tuning_count

    !> Environment-variable stem of a parameter (OPENSPS_TUNE_<name>); empty for an unknown id.
    function tuning_name(id) result(nm)
        integer, intent(in) :: id
        character(len=NAME_LEN) :: nm
        nm = ''
        if (id >= 1 .and. id <= N_TUNE) nm = names(id)
    end function tuning_name

    !> True when v is a legal value of parameter id.
    logical function tuning_in_range(id, v)
        integer, intent(in) :: id, v
        tuning_in_range = id >= 1 .and. id <= N_TUNE
        if (tuning_in_range) tuning_in_range = v >= lo_limit(id) .and. v <= hi_limit(id)
    end function tuning_in_range

    !> Reads an integer (optional sign, digits, optional K, M or G suffix for 1024^n) from text. ok is
    !> false for anything else, including an overflow of a default integer.
    subroutine tuning_parse(text, v, ok)
        character(len=*), intent(in)  :: text
        integer,          intent(out) :: v
        logical,          intent(out) :: ok
        character(len=:), allocatable :: s
        integer(8) :: acc, mult
        integer :: i, n, sgn, c
        character :: last

        v = 0
        ok = .false.
        s = trim(adjustl(text))
        n = len(s)
        if (n == 0) return
        mult = 1
        last = s(n:n)
        select case (last)
        case ('K', 'k'); mult = 1024_8; n = n - 1
        case ('M', 'm'); mult = 1024_8**2; n = n - 1
        case ('G', 'g'); mult = 1024_8**3; n = n - 1
        end select
        if (n == 0) return
        sgn = 1
        i = 1
        if (s(1:1) == '-') then
            sgn = -1; i = 2
        else if (s(1:1) == '+') then
            i = 2
        end if
        if (i > n) return
        acc = 0
        do while (i <= n)
            c = iachar(s(i:i)) - iachar('0')
            if (c < 0 .or. c > 9) return
            acc = acc * 10 + c
            if (acc > 4000000000_8) return
            i = i + 1
        end do
        acc = sgn * acc * mult
        if (abs(acc) > int(huge(v), 8)) return
        v = int(acc)
        ok = .true.
    end subroutine tuning_parse

    !> Value of a run-time parameter (default, environment, or tuning_set, in that order of precedence).
    integer function tuning_get(id) result(v)
        integer, intent(in) :: id

        if (.not. env_read) then
            !$omp critical (opensps_tuning_env)
            if (.not. env_read) then
                call read_environment()
                env_read = .true.
            end if
            !$omp end critical (opensps_tuning_env)
        end if
        v = 0
        if (id >= 1 .and. id <= N_TUNE) v = values(id)
    end function tuning_get

    !> Sets a run-time parameter. ok is false (and nothing changes) for an unknown id or a value outside
    !> the legal range. Not safe to call while batches run on other threads.
    subroutine tuning_set(id, v, ok)
        integer, intent(in)  :: id, v
        logical, intent(out) :: ok
        integer :: dummy

        dummy = tuning_get(id)      ! the environment is read first, so that a later set wins
        ok = tuning_in_range(id, v)
        if (.not. ok) return
        values(id) = v
        source(id) = 2
    end subroutine tuning_set

    !> Back to the defaults, and the environment is read again at the next use.
    subroutine tuning_reset()
        values = defaults
        source = 0
        env_read = .false.
    end subroutine tuning_reset

    subroutine read_environment()
        integer :: id, len_v, stat, v
        logical :: ok
        character(len=64) :: val
        character(len=NAME_LEN + 16) :: var

        do id = 1, N_TUNE
            var = 'OPENSPS_TUNE_' // trim(names(id))
            call get_environment_variable(trim(var), val, length=len_v, status=stat)
            if (stat /= 0 .or. len_v == 0) cycle
            call tuning_parse(val(1:min(len_v, len(val))), v, ok)
            if (ok) ok = tuning_in_range(id, v)
            if (ok) then
                values(id) = v
                source(id) = 1
            else
                write (error_unit, '(a,a,a,i0,a,i0,a,i0)') 'opensps: ', trim(var), ' ignored (not an integer in [', &
                    lo_limit(id), ', ', hi_limit(id), ']); using ', defaults(id)
            end if
        end do
    end subroutine read_environment

    !> Writes the table of run-time parameters (value, source, default, range) to a unit.
    subroutine tuning_report(unit)
        integer, intent(in) :: unit
        character(len=8), parameter :: src(0:2) = ['default ', 'env     ', 'set     ']
        integer :: id, v

        v = tuning_get(1)
        write (unit, '(a)') 'OpenSPS tuning (run-time parameters; set with OPENSPS_TUNE_<NAME>)'
        do id = 1, N_TUNE
            write (unit, '(2x,a24,i12,2x,a8,"default ",i0,"  range [",i0,", ",i0,"]")') names(id), values(id), &
                src(source(id)), defaults(id), lo_limit(id), hi_limit(id)
        end do
    end subroutine tuning_report

end module opensps_tuning
