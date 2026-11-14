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
!>   computed beyond the gates (`TUNE_P_*` ids). A value is resolved in layers, a later one
!>   overriding an earlier one: the default (right for hardware nobody has measured); the device
!>   profile (below); the environment variable `OPENSPS_TUNE_<NAME>` (read once, at the first use);
!>   `tuning_set` (tests, programs that tune themselves). A value outside the legal range of its
!>   parameter, or not an integer, is refused with one message on stderr and the layer below stays.
!>
!> Device profiles are the one place a number measured on one device lives. An entry is a (profile,
!> parameter, value) triple with a reference to the measurement that justifies it, listed with its data in
!> doc/TUNING.md ("Measured device data"): no measurement, no entry, and a device without entries
!> uses the defaults. The profile is that of the build's `-Dgpu_arch` when it names one target
!> (`cc80`, `gfx1201`); a build for several, or `OPENSPS_TUNE_PROFILE=<name>`, selects by hand (the
!> variable wins; an unknown name is refused with a message and no profile applies).
!>
!> `tuning_report` lists every parameter with its value, where the value came from and its range.
!> A measuring tuner, a C API and a device query are the next steps; every consumer already reads
!> through this module, so they will not touch the kernels.
module opensps_tuning
    use, intrinsic :: iso_fortran_env, only: error_unit
    use opensps_precision, only: WP, KP
    use opensps_build_arch, only: BUILD_GPU_ARCH
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

    !> Register tile of the age sums of the single-metallicity-cell kernel `age_sums_z4` (LLVM flang / AMD
    !> variant): models per tile and pixels per thread. Measured on the R9700 against `age_sums` (N = 1e4, 107 ages):
    !> 1.4x at 5994 pixels with MT = 8, PT = 2; the tile shape matters (register spills). Retune per device.
    integer, parameter, public :: TUNE_AGE_SUMS_Z4_FLANG_MT = 8
    integer, parameter, public :: TUNE_AGE_SUMS_Z4_FLANG_PT = 2
    !> Register tile of `age_sums` (models per tile, pixels per thread): chosen for the accumulators of one
    !> thread to fit the registers without spills; "retune on other GPUs, where registers per thread and the
    !> load / FMA balance differ" (the comment this replaces).
    integer, parameter, public :: TUNE_AGE_SUMS_MT = 8
    integer, parameter, public :: TUNE_AGE_SUMS_PT = 4
    !> Register tile of the batched SSP's template sums (`template_columns`): columns per tile, pixels per
    !> thread, pixels per block of the plain partial sum (blocks are added in FP64: changes the summation
    !> order), templates per step.
    integer, parameter, public :: TUNE_TEMPLATE_MT = 8
    integer, parameter, public :: TUNE_TEMPLATE_PT = 4
    integer, parameter, public :: TUNE_TEMPLATE_PIX_BLOCK = 256
    integer, parameter, public :: TUNE_TEMPLATE_KB = 4

    !> Register tile of the age sums of `age_sums_z` (models per tile, pixels per thread), LLVM flang / AMD
    !> variant: models of a tile share one pair of basis slabs. FP64 on the 780M: 4 pixels spill registers (890 ms
    !> at N = 10^4; 658 ms with 2; 8 models x 1-4 pixels is the best range, 16 or more models 6-8 s).
    integer, parameter, public :: TUNE_AGE_SUMS_Z_FLANG_MT = 8
    integer, parameter, public :: TUNE_AGE_SUMS_Z_FLANG_PT = merge(2, 4, KP == WP)
    !> Models per thread of `age_sums_z` and `age_sums_z4` in the nvfortran variants.
    integer, parameter, public :: TUNE_AGE_SUMS_Z_NVF_MT = 4
    integer, parameter, public :: TUNE_AGE_SUMS_Z4_NVF_MT = 4
    !> Templates per block of the split-K template product (`gemm_columns`, `gemm_columns_batched`): each block
    !> is a plain sum, the blocks are added in FP64 (changes the summation order).
    integer, parameter, public :: TUNE_GEMM_KB = 256

    !> Stars per chunk of the single-model SSP sum (host path: L1 cache locality).
    integer, parameter, public :: TUNE_SSP_STAR_BATCH = 32
    !> Column counts up to which the FP32 accelerated BLAS product uses the batched form (`gemm_columns_batched`);
    !> sets the size of its work array (pixels x NC_BATCHED x blocks, on the device once allocated).
    integer, parameter, public :: TUNE_GEMM_NC_BATCHED = 512
    !> Models per chunk of the IMF-gradient reduction of the batched SSP (`CHG`).
    integer, parameter, public :: TUNE_IMF_GRAD_CHG = 256
    !> Pixels per thread of the observed-frame band-flux kernel (`CH`).
    integer, parameter, public :: TUNE_BAND_PIX_PER_THREAD = 16

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

    ! ---- Device profiles (each entry cites its measurement in doc/TUNING.md, "Measured device data") ----
    integer, parameter :: N_PROFILE = 3
    integer, parameter :: PROFILE_LEN = 16
    character(len=PROFILE_LEN), parameter :: prof_arch(N_PROFILE) = [character(len=PROFILE_LEN) :: &
        'cc80', 'cc80', 'gfx1201']
    integer, parameter :: prof_id(N_PROFILE) = [TUNE_P_HOST_SFH_MAX, TUNE_P_PINNED_CHUNK_BYTES, TUNE_P_HOST_SFH_MAX]
    integer, parameter :: prof_val(N_PROFILE) = [16, 16 * 1024 * 1024, 160]
    !> The measurement of each entry (M-numbers of doc/TUNING.md)
    character(len=4), parameter :: prof_ref(N_PROFILE) = ['M1  ', 'M2  ', 'M3  ']

    integer, save :: values(N_TUNE) = defaults
    !> 0 default, 1 environment, 2 set by tuning_set, 3 device profile
    integer, save :: source(N_TUNE) = 0
    logical, save :: env_read = .false.
    character(len=PROFILE_LEN), save :: active_profile = ''

    public :: tuning_get, tuning_set, tuning_reset, tuning_report, tuning_count, tuning_name, &
              tuning_parse, tuning_in_range, tuning_profile, tuning_source, tuning_profile_known, &
              tuning_select_profile

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
        active_profile = ''
        env_read = .false.
    end subroutine tuning_reset

    !> True when `name` (any case) has entries in the profile table.
    logical function tuning_profile_known(name)
        character(len=*), intent(in) :: name
        character(len=PROFILE_LEN) :: key
        integer :: i

        tuning_profile_known = .false.
        key = lower(name)
        do i = 1, N_PROFILE
            if (prof_arch(i) == key) tuning_profile_known = .true.
        end do
    end function tuning_profile_known

    !> The profile in effect ('' when none): the one named by OPENSPS_TUNE_PROFILE, else the build's offload
    !> target when it is a single one.
    function tuning_profile() result(nm)
        character(len=PROFILE_LEN) :: nm
        integer :: dummy

        dummy = tuning_get(1)
        nm = active_profile
    end function tuning_profile

    !> Where the value of a parameter came from: 0 default, 1 environment, 2 tuning_set, 3 device profile.
    integer function tuning_source(id)
        integer, intent(in) :: id
        integer :: dummy

        dummy = tuning_get(1)
        tuning_source = -1
        if (id >= 1 .and. id <= N_TUNE) tuning_source = source(id)
    end function tuning_source

    !> Selects a profile by name (any case; '' for none) and applies it over the defaults, discarding what the
    !> environment and tuning_set had set. ok is false, and no profile applies, for an unknown name.
    subroutine tuning_select_profile(name, ok)
        character(len=*), intent(in) :: name
        logical, intent(out) :: ok
        integer :: dummy

        dummy = tuning_get(1)
        call apply_profile(name, ok)
    end subroutine tuning_select_profile

    subroutine apply_profile(name, ok)
        character(len=*), intent(in) :: name
        logical, intent(out) :: ok
        character(len=PROFILE_LEN) :: key
        integer :: i

        values = defaults
        source = 0
        active_profile = ''
        key = lower(name)
        ok = len_trim(key) == 0
        if (ok) return
        ok = tuning_profile_known(key)
        if (.not. ok) return
        active_profile = key
        do i = 1, N_PROFILE
            if (prof_arch(i) == key) then
                values(prof_id(i)) = prof_val(i)
                source(prof_id(i)) = 3
            end if
        end do
    end subroutine apply_profile

    pure function lower(text) result(r)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: r
        integer :: i, c

        r = text
        do i = 1, len(text)
            c = iachar(text(i:i))
            if (c >= iachar('A') .and. c <= iachar('Z')) r(i:i) = achar(c + 32)
        end do
        r = adjustl(r)
    end function lower

    !> The build's offload target when it names exactly one ('gfx90a:xnack+' counts as gfx90a), else ''.
    function build_profile() result(nm)
        character(len=PROFILE_LEN) :: nm
        integer :: c

        nm = ''
        if (len_trim(BUILD_GPU_ARCH) == 0 .or. index(BUILD_GPU_ARCH, ',') > 0) return
        c = index(BUILD_GPU_ARCH, ':')
        if (c == 0) c = len_trim(BUILD_GPU_ARCH) + 1
        if (c - 1 <= PROFILE_LEN) nm = lower(BUILD_GPU_ARCH(1:c - 1))
    end function build_profile

    subroutine read_environment()
        integer :: id, len_v, stat, v
        logical :: ok
        character(len=64) :: val
        character(len=NAME_LEN + 16) :: var
        character(len=PROFILE_LEN) :: want

        ! Device profile: OPENSPS_TUNE_PROFILE if set, else the build's single offload target
        want = build_profile()
        call get_environment_variable('OPENSPS_TUNE_PROFILE', val, length=len_v, status=stat)
        if (stat == 0 .and. len_v > 0) want = val(1:min(len_v, len(val)))
        call apply_profile(want, ok)
        if (.not. ok) then
            write (error_unit, '(a,a,a)') 'opensps: tuning profile "', trim(want), '" is unknown; the defaults apply'
        end if

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
                    lo_limit(id), ', ', hi_limit(id), ']); using ', values(id)
            end if
        end do
    end subroutine read_environment

    !> Writes the table of run-time parameters (value, source, default, range) to a unit.
    subroutine tuning_report(unit)
        integer, intent(in) :: unit
        character(len=8), parameter :: src(0:3) = ['default ', 'env     ', 'set     ', 'profile ']
        integer :: id, v

        v = tuning_get(1)
        write (unit, '(a)') 'OpenSPS tuning (run-time parameters; set with OPENSPS_TUNE_<NAME>)'
        if (len_trim(active_profile) > 0) then
            write (unit, '(2x,a,a)') 'device profile: ', trim(active_profile)
        else
            write (unit, '(2x,a)') 'device profile: (none)'
        end if
        do id = 1, N_TUNE
            write (unit, '(2x,a24,i12,2x,a8,"default ",i0,"  range [",i0,", ",i0,"]")') names(id), values(id), &
                src(source(id)), defaults(id), lo_limit(id), hi_limit(id)
        end do
    end subroutine tuning_report

end module opensps_tuning
