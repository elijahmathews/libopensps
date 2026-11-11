module cue_test_library
    !> Synthetic cue networks for the tests (no data needed): weights of cue's architecture (17
    !> networks of 12 -> 256 -> 256 -> 256 inputs and units) drawn at random with sizes that keep
    !> the outputs in cue's range (log L ~ 38 erg/s for lines, ~25 erg/s/Hz for the continuum),
    !> 43 lines in 16 groups, a 37-point continuum grid; and a writer of them as a converted
    !> library (manifest.json and raw little-endian FP64 arrays, as tools/cue/convert_cue.py
    !> lays them out), so the loader and the batch's cue path run in CI.
    use, intrinsic :: iso_fortran_env, only: int64
    use opensps_precision, only: WP, KP
    use opensps_cue, only: cue_weights_t, cue_device_put, CUE_NIN, CUE_NHID, CUE_NGROUP, CUE_NNET, CUE_NPCA_LINE
    implicit none
    private

    public :: cue_synthetic, cue_write_library

    integer, parameter :: CNT(CUE_NGROUP) = [5, 3, 1, 2, 4, 1, 3, 2, 2, 4, 5, 3, 1, 1, 2, 1]
    integer(int64) :: seed = 20261110_int64

contains

    real(WP) function urand()
        seed = iand(1103515245_int64 * seed + 12345_int64, 2147483647_int64)
        urand = real(seed, WP) / 2147483648.0_wp
    end function urand

    !> Uniform in [-s, s), rounded to KP (the hidden layers' weights are FP32 in cue)
    subroutine fill(a, n, s)
        integer, intent(in) :: n
        real(KP), intent(out) :: a(n)
        real(WP), intent(in) :: s
        integer :: i
        do i = 1, n
            a(i) = real(s * (2.0_wp * urand() - 1.0_wp), KP)
        end do
    end subroutine fill

    !> Uniform in [-s, s)
    subroutine fill_r(a, n, s)
        integer, intent(in) :: n
        real(WP), intent(out) :: a(n)
        real(WP), intent(in) :: s
        integer :: i
        do i = 1, n
            a(i) = s * (2.0_wp * urand() - 1.0_wp)
        end do
    end subroutine fill_r

    !> s (1 +- 10%)
    subroutine fill_near(a, n, s)
        integer, intent(in) :: n
        real(WP), intent(out) :: a(n)
        real(WP), intent(in) :: s
        integer :: i
        do i = 1, n
            a(i) = s * (1.0_wp + 0.1_wp * (2.0_wp * urand() - 1.0_wp))
        end do
    end subroutine fill_near

    !> Synthetic networks in w (put on the device in GPU builds). line_lam: the 43 lines'
    !> wavelengths (default: 1000 A to 10 microns, spread).
    subroutine cue_synthetic(w, line_lam)
        type(cue_weights_t), intent(inout) :: w
        real(WP), intent(in), optional :: line_lam(:)
        integer :: g, n, i

        w%nlmax = maxval(CNT)
        w%ncw = 37
        w%npc = 6
        allocate(w%w1(CUE_NHID, CUE_NIN, CUE_NNET), w%w2(CUE_NHID, CUE_NHID, CUE_NNET), &
                 w%w3(CUE_NHID, CUE_NHID, CUE_NNET), w%bias(CUE_NHID, CUE_NNET, 3), &
                 w%alpha(CUE_NHID, CUE_NNET, 3), w%beta(CUE_NHID, CUE_NNET, 3), &
                 w%in_shift(CUE_NIN, CUE_NNET), w%in_scale(CUE_NIN, CUE_NNET))
        call fill(w%w1, size(w%w1), 0.5_wp)
        call fill(w%w2, size(w%w2), 0.12_wp)
        call fill(w%w3, size(w%w3), 0.12_wp)
        call fill(w%bias, size(w%bias), 0.2_wp)
        call fill(w%alpha, size(w%alpha), 2.0_wp)
        call fill(w%beta, size(w%beta), 1.0_wp)
        w%beta = 0.5_KP + 0.5_KP * w%beta                     ! in (0, 1)
        call fill(w%in_shift, size(w%in_shift), 2.0_wp)
        call fill(w%in_scale, size(w%in_scale), 1.0_wp)
        w%in_scale = 1.0_KP + abs(w%in_scale)
        ! (inputs: the 12 of cue; n_H ~ 1e1-1e4 and log Q ~ 50 are scaled as cue does)
        w%in_shift(8, :) = 49.5_KP
        w%in_scale(8, :) = 1.2_KP
        w%in_shift(9, :) = 1500.0_KP
        w%in_scale(9, :) = 2300.0_KP
        allocate(w%npca(CUE_NGROUP), w%ncount(CUE_NGROUP), w%loff(CUE_NGROUP))
        w%ncount = CNT
        n = 0
        do g = 1, CUE_NGROUP
            w%npca(g) = 1 + mod(g, 4)
            w%loff(g) = n
            n = n + CNT(g)
        end do
        w%nlines = n
        allocate(w%lw4(CUE_NPCA_LINE, CUE_NHID, CUE_NGROUP), w%lb4(CUE_NPCA_LINE, CUE_NGROUP), &
                 w%lpsh(CUE_NPCA_LINE, CUE_NGROUP), w%lpsc(CUE_NPCA_LINE, CUE_NGROUP), &
                 w%lcmp(w%nlmax, CUE_NPCA_LINE, CUE_NGROUP), w%lmean(w%nlmax, CUE_NGROUP), &
                 w%lout_sh(w%nlmax, CUE_NGROUP), w%lout_sc(w%nlmax, CUE_NGROUP))
        call fill(w%lw4, size(w%lw4), 0.1_wp)
        call fill_r(w%lb4, size(w%lb4), 0.5_wp)
        call fill_r(w%lpsh, size(w%lpsh), 1.0_wp)
        call fill_r(w%lpsc, size(w%lpsc), 1.0_wp)
        call fill_r(w%lcmp, size(w%lcmp), 1.0_wp)
        call fill_r(w%lmean, size(w%lmean), 1.0_wp)
        ! Padding as the converter writes it: zeros beyond each group's PCA size
        do g = 1, CUE_NGROUP
            w%lw4(w%npca(g) + 1:, :, g) = 0.0_KP
            w%lb4(w%npca(g) + 1:, g) = 0.0_wp
            w%lpsh(w%npca(g) + 1:, g) = 0.0_wp
            w%lpsc(w%npca(g) + 1:, g) = 0.0_wp
            w%lcmp(:, w%npca(g) + 1:, g) = 0.0_wp
        end do
        call fill_near(w%lout_sh, size(w%lout_sh), 38.0_wp)
        call fill_near(w%lout_sc, size(w%lout_sc), 0.2_wp)
        w%lcmp_k = real(w%lcmp, KP)
        allocate(w%cw4(w%npc, CUE_NHID), w%cb4(w%npc), w%cpsh(w%npc), w%cpsc(w%npc), &
                 w%ccmp(w%ncw, w%npc), w%cmean(w%ncw), w%cout_sh(w%ncw), w%cout_sc(w%ncw))
        call fill(w%cw4, size(w%cw4), 0.1_wp)
        call fill_r(w%cb4, size(w%cb4), 0.5_wp)
        call fill_r(w%cpsh, size(w%cpsh), 1.0_wp)
        call fill_r(w%cpsc, size(w%cpsc), 1.0_wp)
        call fill_r(w%ccmp, size(w%ccmp), 1.0_wp)
        call fill_r(w%cmean, size(w%cmean), 1.0_wp)
        call fill_near(w%cout_sh, size(w%cout_sh), 25.0_wp)
        call fill_near(w%cout_sc, size(w%cout_sc), 0.2_wp)
        allocate(w%line_lam(w%nlines), w%cont_lam(w%ncw))
        if (present(line_lam)) then
            w%line_lam = line_lam(1:w%nlines)
        else
            do i = 1, w%nlines
                w%line_lam(i) = 1000.0_wp * 100.0_wp**(real(i - 1, WP) / real(w%nlines - 1, WP))
            end do
        end if
        do i = 1, w%ncw
            w%cont_lam(i) = 915.0_wp * (1.0e8_wp / 915.0_wp)**(real(i - 1, WP) / real(w%ncw - 1, WP))
        end do
        w%loaded = .true.
        call cue_device_put(w)
    end subroutine cue_synthetic

    !> Writes w as a converted cue library into the directory dir (created).
    subroutine cue_write_library(w, dir)
        type(cue_weights_t), intent(in) :: w
        character(len=*), intent(in) :: dir
        integer :: u, l, g, k
        real(WP), allocatable :: t3(:,:,:)

        call execute_command_line('mkdir -p "' // dir // '"')
        open(newunit=u, file=dir // '/manifest.json', status='replace', action='write')
        write(u, '(A)') '{"format": "opensps-data", "format_version": 1, "kind": "nebular_emulator", "name": "cue",'
        write(u, '(A)') ' "title": "synthetic cue networks (tests)", "metadata": {}, "axes": {'
        call label_axis(u, 'net', CUE_NNET, .false.)
        call label_axis(u, 'line_group', CUE_NGROUP, .false.)
        call label_axis(u, 'input', CUE_NIN, .false.)
        call index_axis(u, 'hidden', CUE_NHID)
        call index_axis(u, 'hidden_in', CUE_NHID)
        call index_axis(u, 'layer', 3)
        call index_axis(u, 'pca_line', CUE_NPCA_LINE)
        call index_axis(u, 'line_slot', w%nlmax)
        call index_axis(u, 'pca_cont', w%npc)
        write(u, '(A,I0,A)') '  "line": {"units": "angstrom", "file": "axis_line.bin", "dtype": "<f8", "shape": [', &
            w%nlines, ']},'
        write(u, '(A,I0,A)') '  "cont_wavelength": {"units": "angstrom", "file": "axis_cont.bin", "dtype": "<f8", "shape": [', &
            w%ncw, ']}'
        write(u, '(A)') ' }, "arrays": {'
        call array(u, 'w1', ['net   ', 'input ', 'hidden'], [CUE_NNET, CUE_NIN, CUE_NHID], .true.)
        call array(u, 'w2', ['net      ', 'hidden_in', 'hidden   '], [CUE_NNET, CUE_NHID, CUE_NHID], .true.)
        call array(u, 'w3', ['net      ', 'hidden_in', 'hidden   '], [CUE_NNET, CUE_NHID, CUE_NHID], .true.)
        do l = 1, 3
            call array(u, 'b' // achar(48 + l), ['net   ', 'hidden'], [CUE_NNET, CUE_NHID], .true.)
        end do
        call array(u, 'alpha', ['net   ', 'layer ', 'hidden'], [CUE_NNET, 3, CUE_NHID], .true.)
        call array(u, 'beta', ['net   ', 'layer ', 'hidden'], [CUE_NNET, 3, CUE_NHID], .true.)
        call array(u, 'input_shift', ['net  ', 'input'], [CUE_NNET, CUE_NIN], .true.)
        call array(u, 'input_scale', ['net  ', 'input'], [CUE_NNET, CUE_NIN], .true.)
        call array(u, 'line_w4', ['line_group', 'hidden_in ', 'pca_line  '], [CUE_NGROUP, CUE_NHID, CUE_NPCA_LINE], .true.)
        call array(u, 'line_b4', ['line_group', 'pca_line  '], [CUE_NGROUP, CUE_NPCA_LINE], .true.)
        call array(u, 'line_pca_shift', ['line_group', 'pca_line  '], [CUE_NGROUP, CUE_NPCA_LINE], .true.)
        call array(u, 'line_pca_scale', ['line_group', 'pca_line  '], [CUE_NGROUP, CUE_NPCA_LINE], .true.)
        call array(u, 'line_components', ['line_group', 'pca_line  ', 'line_slot '], &
                   [CUE_NGROUP, CUE_NPCA_LINE, w%nlmax], .true.)
        call array(u, 'line_mean', ['line_group', 'line_slot '], [CUE_NGROUP, w%nlmax], .true.)
        call array(u, 'line_out_shift', ['line_group', 'line_slot '], [CUE_NGROUP, w%nlmax], .true.)
        call array(u, 'line_out_scale', ['line_group', 'line_slot '], [CUE_NGROUP, w%nlmax], .true.)
        call array(u, 'line_npca', ['line_group'], [CUE_NGROUP], .true., '<i4')
        call array(u, 'line_count', ['line_group'], [CUE_NGROUP], .true., '<i4')
        call array(u, 'cont_w4', ['hidden_in', 'pca_cont '], [CUE_NHID, w%npc], .true.)
        call array(u, 'cont_b4', ['pca_cont'], [w%npc], .true.)
        call array(u, 'cont_pca_shift', ['pca_cont'], [w%npc], .true.)
        call array(u, 'cont_pca_scale', ['pca_cont'], [w%npc], .true.)
        call array(u, 'cont_components', ['pca_cont       ', 'cont_wavelength'], [w%npc, w%ncw], .true.)
        call array(u, 'cont_mean', ['cont_wavelength'], [w%ncw], .true.)
        call array(u, 'cont_out_shift', ['cont_wavelength'], [w%ncw], .true.)
        call array(u, 'cont_out_scale', ['cont_wavelength'], [w%ncw], .false.)
        write(u, '(A)') ' }}'
        close(u)

        ! The arrays: Fortran order is C order with the dimensions reversed
        call put(dir // '/w1.bin', real(w%w1, WP), size(w%w1))
        call put(dir // '/w2.bin', real(w%w2, WP), size(w%w2))
        call put(dir // '/w3.bin', real(w%w3, WP), size(w%w3))
        do l = 1, 3
            call put(dir // '/b' // achar(48 + l) // '.bin', real(w%bias(:, :, l), WP), CUE_NHID * CUE_NNET)
        end do
        ! alpha, beta: (hidden, layer, net) in the file, (hidden, net, layer) in memory
        allocate(t3(CUE_NHID, 3, CUE_NNET))
        do g = 1, CUE_NNET
            do k = 1, 3
                t3(:, k, g) = real(w%alpha(:, g, k), WP)
            end do
        end do
        call put(dir // '/alpha.bin', t3, size(t3))
        do g = 1, CUE_NNET
            do k = 1, 3
                t3(:, k, g) = real(w%beta(:, g, k), WP)
            end do
        end do
        call put(dir // '/beta.bin', t3, size(t3))
        call put(dir // '/input_shift.bin', real(w%in_shift, WP), size(w%in_shift))
        call put(dir // '/input_scale.bin', real(w%in_scale, WP), size(w%in_scale))
        call put(dir // '/line_w4.bin', real(w%lw4, WP), size(w%lw4))
        call put(dir // '/line_b4.bin', w%lb4, size(w%lb4))
        call put(dir // '/line_pca_shift.bin', w%lpsh, size(w%lpsh))
        call put(dir // '/line_pca_scale.bin', w%lpsc, size(w%lpsc))
        call put(dir // '/line_components.bin', w%lcmp, size(w%lcmp))
        call put(dir // '/line_mean.bin', w%lmean, size(w%lmean))
        call put(dir // '/line_out_shift.bin', w%lout_sh, size(w%lout_sh))
        call put(dir // '/line_out_scale.bin', w%lout_sc, size(w%lout_sc))
        call put_i(dir // '/line_npca.bin', w%npca)
        call put_i(dir // '/line_count.bin', w%ncount)
        call put(dir // '/cont_w4.bin', real(w%cw4, WP), size(w%cw4))
        call put(dir // '/cont_b4.bin', w%cb4, size(w%cb4))
        call put(dir // '/cont_pca_shift.bin', w%cpsh, size(w%cpsh))
        call put(dir // '/cont_pca_scale.bin', w%cpsc, size(w%cpsc))
        call put(dir // '/cont_components.bin', w%ccmp, size(w%ccmp))
        call put(dir // '/cont_mean.bin', w%cmean, size(w%cmean))
        call put(dir // '/cont_out_shift.bin', w%cout_sh, size(w%cout_sh))
        call put(dir // '/cont_out_scale.bin', w%cout_sc, size(w%cout_sc))
        call put(dir // '/axis_line.bin', w%line_lam, w%nlines)
        call put(dir // '/axis_cont.bin', w%cont_lam, w%ncw)
    end subroutine cue_write_library

    subroutine label_axis(u, name, n, last)
        integer, intent(in) :: u, n
        character(len=*), intent(in) :: name
        logical, intent(in) :: last
        integer :: i
        write(u, '(3A)', advance='no') '  "', name, '": {"units": "", "labels": ['
        do i = 1, n
            write(u, '(A,I0,A)', advance='no') '"', i, '"'
            if (i < n) write(u, '(A)', advance='no') ', '
        end do
        write(u, '(A)') ']}' // merge('  ', ', ', last)
    end subroutine label_axis

    subroutine index_axis(u, name, n)
        integer, intent(in) :: u, n
        character(len=*), intent(in) :: name
        integer :: i
        write(u, '(3A)', advance='no') '  "', name, '": {"units": "", "values": ['
        do i = 1, n
            write(u, '(I0)', advance='no') i - 1
            if (i < n) write(u, '(A)', advance='no') ', '
        end do
        write(u, '(A)') ']},'
    end subroutine index_axis

    !> An array's descriptor: shape and axes in C order
    subroutine array(u, name, axes, shape, comma, dtype)
        integer, intent(in) :: u, shape(:)
        character(len=*), intent(in) :: name, axes(:)
        logical, intent(in) :: comma
        character(len=*), intent(in), optional :: dtype
        integer :: i
        character(len=3) :: dt
        dt = '<f8'
        if (present(dtype)) dt = dtype
        write(u, '(5A)', advance='no') '  "', name, '": {"file": "', name, '.bin", "dtype": "' // dt // '", "shape": ['
        do i = 1, size(shape)
            write(u, '(I0)', advance='no') shape(i)
            if (i < size(shape)) write(u, '(A)', advance='no') ', '
        end do
        write(u, '(A)', advance='no') '], "axes": ['
        do i = 1, size(axes)
            write(u, '(3A)', advance='no') '"', trim(axes(i)), '"'
            if (i < size(axes)) write(u, '(A)', advance='no') ', '
        end do
        write(u, '(A)') ']}' // merge(', ', '  ', comma)
    end subroutine array

    subroutine put(path, a, n)
        character(len=*), intent(in) :: path
        integer, intent(in) :: n
        real(WP), intent(in) :: a(n)
        integer :: u
        open(newunit=u, file=path, access='stream', form='unformatted', status='replace', action='write')
        write(u) a
        close(u)
    end subroutine put

    subroutine put_i(path, a)
        character(len=*), intent(in) :: path
        integer, intent(in) :: a(:)
        integer :: u
        open(newunit=u, file=path, access='stream', form='unformatted', status='replace', action='write')
        write(u) a
        close(u)
    end subroutine put_i

end module cue_test_library
