module test_opensps_cue_mod
    !> The pieces of the cue nebular model (opensps_cue).
    !>
    !> Without data (always run):
    !> - the networks on synthetic weights against a direct FP64 evaluation of the same algebra
    !>   written here (speculator layers, PCA bases, output scale and shift), with the batch split
    !>   in chunks with a remainder and a register tile with a remainder;
    !> - cue's power-law fit: on spectra that are exact power laws per segment (slopes and
    !>   luminosity ratios recovered), and, on irregular spectra, the optimality of the fitted
    !>   segment (zero gradient of cue's loss, computed here by finite differences of an
    !>   independent implementation; no lower loss nearby);
    !> - the stable form of log10((b**beta - a**beta) / beta) against the direct one and its limit;
    !> - the not-a-knot spline: exact for cubics (inside and extrapolated), zero at and below 911.6 A;
    !> - the line matching between two line lists.
    !> With OPENSPS_CUE_DATA (converted with --reference by tools/cue/convert_cue.py; skipped
    !> otherwise): the networks against cue's algebra evaluated in FP64 by the converter, and the
    !> fit against cue's fit_4loglinear_ionparam with its loss minimized to convergence.
    use, intrinsic :: iso_fortran_env, only: int64
    use opensps_precision, only: WP, KP
    use opensps_cue
    use cue_test_library, only: cue_synthetic, cue_write_library
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use opensps_datalib, only: datalib_t, datalib_open_dir, datalib_has_array, datalib_read_real, &
                               datalib_axis_len, datalib_axis_values
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, assert_true
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_opensps_cue_tests, total_failures, total_tests

    integer(int64) :: seed = 20261110_int64
    real(WP), parameter :: HC = 6.626e-27_wp * 2.9979e18_wp

contains

    subroutine run_opensps_cue_tests()
        call print_minor_header("opensps_cue")
        call test_log_ratio_power()
        call test_fit_power_law()
        call test_fit_optimality()
        call test_spline()
        call test_line_map()
        call test_forward_synthetic()
        call test_loader()
        call test_with_data()
        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_opensps_cue_tests

    real(WP) function urand()
        seed = iand(1103515245_int64 * seed + 12345_int64, 2147483647_int64)
        urand = real(seed, WP) / 2147483648.0_wp
    end function urand

    ! ------------------------------------------------------------------------------------------

    subroutine test_log_ratio_power()
        real(WP) :: betas(7), a, b, v, ref, worst
        integer :: i
        call print_group("log10((b**beta - a**beta) / beta) without cancellation")
        betas = [-3.0_wp, -0.5_wp, 0.7_wp, 2.0_wp, 39.0_wp, 1.0e-2_wp, -2.0e-3_wp]
        a = 227.838_wp
        b = 353.02_wp
        worst = 0.0_wp
        do i = 1, size(betas)
            v = cue_log_ratio_power(betas(i), a, b)
            ref = log10(abs((b**betas(i) - a**betas(i)) / betas(i)))
            worst = max(worst, abs(v - ref))
        end do
        call assert_true(worst < 1.0e-11_wp, "matches the direct form away from beta = 0", total_tests, total_failures)
        ! beta -> 0: log10 ln(b / a), and continuity across a tiny beta
        v = cue_log_ratio_power(0.0_wp, a, b)
        call assert_true(abs(v - log10(log(b / a))) < 1.0e-14_wp, "beta = 0 gives log10 ln(b / a)", &
                         total_tests, total_failures)
        v = cue_log_ratio_power(1.0e-9_wp, a, b)
        ref = cue_log_ratio_power(2.0e-3_wp, a, b)
        call assert_true(abs(v - log10(log(b / a))) < 1.0e-8_wp .and. abs(ref - log10(abs((b**2.0e-3_wp - &
                         a**2.0e-3_wp) / 2.0e-3_wp))) < 1.0e-10_wp, "continuous through small beta", &
                         total_tests, total_failures)
        ! (from a = 1, as the first segment of cue's luminosities)
        v = cue_log_ratio_power(5.0_wp, 1.0_wp, 227.838_wp)
        call assert_true(abs(v - log10((227.838_wp**5 - 1.0_wp) / 5.0_wp)) < 1.0e-12_wp, &
                         "segment from 1 A", total_tests, total_failures)
    end subroutine test_log_ratio_power

    !> A grid like cue's tests: 90 A to 1200 A, log-spaced
    subroutine fit_grid(nw, lam)
        integer, intent(in) :: nw
        real(WP), intent(out) :: lam(nw)
        integer :: i
        do i = 1, nw
            lam(i) = 90.0_wp * (1200.0_wp / 90.0_wp)**(real(i - 1, WP) / real(nw - 1, WP))
        end do
    end subroutine fit_grid

    subroutine test_fit_power_law()
        integer, parameter :: NW = 800
        real(WP) :: lam(NW), al(4), la(4), ion(CUE_NION), edge(5), lnl(4), ratio(3), err_a, err_r
        real(KP) :: spec(NW)
        integer :: i, s, c
        call print_group("power-law fit: exact power laws are recovered")
        call fit_grid(NW, lam)
        edge = [1.0_wp, CUE_EDGE_HEII, CUE_EDGE_OII, CUE_EDGE_HEI, CUE_EDGE_HI]
        err_a = 0.0_wp
        err_r = 0.0_wp
        do c = 1, 4
            ! Slopes inside the training ranges, normalizations giving steps at the edges
            al = [6.0_wp + 3.0_wp * c, 2.0_wp + 0.5_wp * c, 0.5_wp * c, -1.0_wp + 0.4_wp * c]
            la = [-40.0_wp - 2.0_wp * c, -20.0_wp - 0.3_wp * c, -12.0_wp, -9.0_wp + 0.1_wp * c]
            do i = 1, NW
                if (lam(i) <= edge(2)) then
                    s = 1
                else if (lam(i) <= edge(3)) then
                    s = 2
                else if (lam(i) <= edge(4)) then
                    s = 3
                else
                    s = 4
                end if
                spec(i) = real(10.0_wp**(la(s) + al(s) * log10(lam(i))), KP)
            end do
            call cue_fit_ionizing(NW, lam, spec, ion)
            do s = 1, 4
                lnl(s) = la(s) + cue_log_ratio_power(al(s) - 1.0_wp, edge(s), edge(s + 1))
                err_a = max(err_a, abs(ion(s) - al(s)))
            end do
            do s = 1, 3
                ratio(s) = lnl(s + 1) - lnl(s)
                err_r = max(err_r, abs(ion(4 + s) - min(max(ratio(s), cue_ion_bound(4 + s, .false.)), &
                                                       cue_ion_bound(4 + s, .true.))))
            end do
        end do
        write (*, '(A,2ES10.2)') '# worst slope and ratio errors: ', err_a, err_r
        ! The photon-rate term compares the trapezoid on the grid with the exact integral, so the
        ! fit sits ~(spacing)^2 from the truth: 1e-3 on 800 pixels; FP32 data round at 6e-8
        call assert_true(err_a < 2.0e-3_wp, "slopes recovered", total_tests, total_failures)
        call assert_true(err_r < 2.0e-3_wp, "luminosity ratios recovered", total_tests, total_failures)
        ! No flux at all: the lower ends of the ranges
        spec = 0.0_KP
        call cue_fit_ionizing(NW, lam, spec, ion)
        call assert_true(all(ion == [(cue_ion_bound(s, .false.), s = 1, CUE_NION)]), &
                         "an empty spectrum gives the lower bounds", total_tests, total_failures)
        ! No flux below 228 A: the first slope at its lower bound, the first ratio at its upper
        call fit_grid(NW, lam)
        do i = 1, NW
            spec(i) = real(10.0_wp**(-10.0_wp + 0.5_wp * log10(lam(i))), KP)
            if (lam(i) <= CUE_EDGE_HEII) spec(i) = 0.0_KP
        end do
        call cue_fit_ionizing(NW, lam, spec, ion)
        call assert_true(ion(1) == cue_ion_bound(1, .false.) .and. ion(5) == cue_ion_bound(5, .true.), &
                         "an empty He II segment: index at its floor, ratio at its ceiling", total_tests, total_failures)
        ! Two empty segments next to each other: ratio 0 (cue: NaN)
        do i = 1, NW
            if (lam(i) <= CUE_EDGE_OII) spec(i) = 0.0_KP
        end do
        call cue_fit_ionizing(NW, lam, spec, ion)
        call assert_true(ion(5) == 0.0_wp .and. ion(6) == cue_ion_bound(6, .true.), &
                         "two empty segments: ratio 0, then the ceiling", total_tests, total_failures)
    end subroutine test_fit_power_law

    !> cue's loss of one segment, written out independently (cue.utils.customize_loss_funtion_*)
    real(WP) function cue_loss(n, lam, y, logq, al, la)
        integer, intent(in) :: n
        real(WP), intent(in) :: lam(n), y(n), logq, al, la
        integer :: i
        real(WP) :: r, qp
        cue_loss = 0.0_wp
        do i = 1, n
            r = la + al * log10(lam(i)) - y(i)
            cue_loss = cue_loss + 0.5_wp * r * r
        end do
        qp = la - log10(6.626e-27_wp) + log10((lam(n)**al - lam(1)**al) / al)
        cue_loss = cue_loss + 0.5_wp * real(n, WP) * (logq - qp)**2
    end function cue_loss

    subroutine test_fit_optimality()
        integer, parameter :: NW = 300
        real(WP) :: lam(NW), alpha, loga, y(NW), q, g1, g2, h, worst_g, l0, lmin, fnorm
        real(KP) :: spec(NW)
        integer :: c, i, i0, i1, k, j
        logical :: ok_min
        call print_group("power-law fit: cue's loss is minimal at the fitted segment")
        call fit_grid(NW, lam)
        worst_g = 0.0_wp
        ok_min = .true.
        do c = 1, 12
            ! A blackbody-like spectrum with noise and a step: not a power law
            do i = 1, NW
                spec(i) = real(1.0e-12_wp * (lam(i) / 500.0_wp)**(1.0_wp + 0.3_wp * c) * &
                               exp(-200.0_wp / lam(i)) * (1.0_wp + 0.1_wp * (urand() - 0.5_wp)), KP)
                if (lam(i) < 300.0_wp) spec(i) = spec(i) * 0.1_KP
            end do
            ! Segment: pixels in (353, 504] (cue's third)
            i0 = 0; i1 = 0
            do i = 1, NW
                if (lam(i) <= CUE_EDGE_OII) i0 = i
                if (lam(i) <= CUE_EDGE_HEI) i1 = i
            end do
            i0 = i0 + 1
            fnorm = 1.0e-18_wp / real(spec(i1 + 1), WP)
            call fit_segment(NW, lam, spec, i0, i1, fnorm, 1.0e-70_wp * fnorm, alpha, loga)
            ! The data of the loss, independently: Y and Q from the normalized spectrum
            k = i1 - i0 + 1
            do i = 1, k
                y(i) = log10(real(spec(i0 + i - 1), WP) * fnorm)
            end do
            q = 0.0_wp
            do i = i0, i1 - 1
                q = q + 0.5_wp * (real(spec(i), WP) * fnorm * lam(i) + real(spec(i + 1), WP) * fnorm * lam(i + 1)) / HC * &
                        (2.9979e18_wp / lam(i) - 2.9979e18_wp / lam(i + 1))
            end do
            q = log10(abs(q))
            h = 1.0e-6_wp
            g1 = (cue_loss(k, lam(i0:i1), y, q, alpha + h, loga) - cue_loss(k, lam(i0:i1), y, q, alpha - h, loga)) / (2 * h)
            g2 = (cue_loss(k, lam(i0:i1), y, q, alpha, loga + h) - cue_loss(k, lam(i0:i1), y, q, alpha, loga - h)) / (2 * h)
            worst_g = max(worst_g, abs(g1), abs(g2))
            l0 = cue_loss(k, lam(i0:i1), y, q, alpha, loga)
            lmin = huge(1.0_wp)
            do j = 0, 7
                lmin = min(lmin, cue_loss(k, lam(i0:i1), y, q, alpha + 1.0e-3_wp * cos(0.785_wp * j), &
                                          loga + 1.0e-3_wp * sin(0.785_wp * j)))
            end do
            ok_min = ok_min .and. l0 <= lmin
        end do
        write (*, '(A,ES10.2)') '# worst |gradient| of the loss at the fit: ', worst_g
        ! (finite differences of a loss of size ~1 at h = 1e-6: noise ~1e-10; FP32 data are fitted
        ! in FP64, so the bound holds in both builds)
        call assert_true(worst_g < 1.0e-8_wp, "zero gradient at the fit", total_tests, total_failures)
        call assert_true(ok_min, "no lower loss around the fit", total_tests, total_failures)
    end subroutine test_fit_optimality

    subroutine test_spline()
        integer, parameter :: NK = 40, NO = 300, NS = 3
        real(WP) :: x(NK), xo(NO), p, worst, s
        real(KP) :: v(NK, NS), mm(NK, NS), out(NO, NS)
        integer :: i, j, m
        logical :: ok_zero
        call print_group("not-a-knot spline: exact for cubics, zero at and below 911.6 A")
        x(1) = 1000.0_wp
        do i = 2, NK
            x(i) = x(i - 1) + 50.0_wp + 150.0_wp * urand()
        end do
        do j = 1, NO
            xo(j) = 800.0_wp + (x(NK) + 300.0_wp - 800.0_wp) * real(j - 1, WP) / real(NO - 1, WP)
        end do
        do m = 1, NS
            do i = 1, NK
                v(i, m) = real(cubic(x(i), m), KP)
            end do
        end do
        call cue_spline_build(cue_sp, NK, x, NO, xo)
        call cue_spline_build(cue_sp, NK, x, NO, xo)      ! (unchanged grid: kept)
        call cue_spline_apply(NK, NO, NS, cue_sp%h, cue_sp%cp, cue_sp%rden, cue_sp%aa, cue_sp%m0c, cue_sp%mnc, &
                              cue_sp%k, cue_sp%t, cue_sp%hh6, v, mm, out)
        worst = 0.0_wp
        ok_zero = .true.
        do m = 1, NS
            do j = 1, NO
                if (xo(j) <= 911.6_wp) then
                    ok_zero = ok_zero .and. out(j, m) == 0.0_KP
                else
                    p = cubic(xo(j), m)
                    s = maxval(abs(v(:, m)))
                    worst = max(worst, abs(real(out(j, m), WP) - p) / s)
                end if
            end do
        end do
        write (*, '(A,ES10.2)') '# worst error on cubics (relative to their range): ', worst
        call assert_true(worst < merge(1.0e-11_wp, 2.0e-5_wp, KP == WP), "cubics reproduced, also extrapolated", &
                         total_tests, total_failures)
        call assert_true(ok_zero, "no continuum at and below 911.6 A", total_tests, total_failures)
    contains
        real(WP) function cubic(xx, k)
            real(WP), intent(in) :: xx
            integer, intent(in) :: k
            real(WP) :: t
            t = (xx - 3000.0_wp) / 1000.0_wp
            cubic = 2.0_wp + 0.3_wp * k * t + 0.2_wp * t * t - (0.05_wp + 0.01_wp * k) * t * t * t
        end function cubic
    end subroutine test_spline

    subroutine test_line_map()
        type(cue_weights_t) :: w
        real(WP) :: lam(6)
        integer :: map(6)
        call print_group("line matching")
        w%nlines = 5
        allocate(w%line_lam(5))
        w%line_lam = [1215.67_wp, 4862.69_wp, 4863.5_wp, 6564.6_wp, 1750.0_wp]
        lam = [6564.61_wp, 4862.71_wp, 1215.67_wp, 5008.24_wp, 0.0_wp, 1749.0_wp]
        call cue_line_map(w, 6, lam, map)
        call assert_true(all(map == [4, 2, 1, 0, 0, 5]), "nearest within tolerance, closer wins a shared slot", &
                         total_tests, total_failures)
    end subroutine test_line_map

    ! ------------------------------------------------------------------------------------------
    ! The networks on synthetic weights
    ! ------------------------------------------------------------------------------------------

    subroutine test_forward_synthetic()
        type(cue_weights_t) :: w
        integer, parameter :: N = 9
        real(WP) :: theta(CUE_NIN, N), ll(1, 1), cl(1, 1)
        real(WP), allocatable :: line_log(:,:), cont_log(:,:), lref(:,:), cref(:,:), line2(:,:), cont2(:,:)
        real(WP) :: worst, tol
        integer :: m, k
        call print_group("networks: synthetic weights against a direct FP64 evaluation")
        call cue_synthetic(w)
        do m = 1, N
            do k = 1, CUE_NIN
                theta(k, m) = 2.0_wp * urand() - 1.0_wp + 2.0_wp
            end do
        end do
        allocate(line_log(w%nlines, N), cont_log(w%ncw, N), lref(w%nlines, N), cref(w%ncw, N), &
                 line2(w%nlines, N), cont2(w%ncw, N))
        call cue_forward(w, N, theta, line_log, cont_log)
        call cue_forward(w, N, theta, line2, cont2, chunk=4)     ! three passes, the last of one model
        call reference_forward(w, N, theta, lref, cref)
        worst = max(maxval(abs(line_log - lref)), maxval(abs(cont_log - cref)))
        write (*, '(A,ES10.2)') '# worst |log L| difference: ', worst
        ! FP32 networks: the hidden layers round at ~1e-7 of O(1) values, the output scale is ~2
        tol = merge(1.0e-11_wp, 1.0e-4_wp, KP == WP)
        call assert_true(worst < tol, "lines and continuum match the direct evaluation", total_tests, total_failures)
#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET)
        ! (a GPU BLAS chooses its kernels, so its order of summation, by the matrices' sizes)
        call assert_true(maxval(abs(line2 - line_log)) < tol .and. maxval(abs(cont2 - cont_log)) < tol, &
                         "the split into chunks changes nothing (to rounding: BLAS)", total_tests, total_failures)
#else
        call assert_true(all(line2 == line_log) .and. all(cont2 == cont_log), &
                         "the split into chunks changes nothing (bit-identical)", total_tests, total_failures)
#endif
        ! A batch of one (a tile of one model)
        call cue_forward(w, 1, theta(:, 5), line2(:, 1), cont2(:, 1))
#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET)
        call assert_true(maxval(abs(line2(:, 1) - line_log(:, 5))) < tol .and. &
                         maxval(abs(cont2(:, 1) - cont_log(:, 5))) < tol, &
                         "a model's result does not depend on the batch (to rounding: BLAS)", total_tests, total_failures)
#else
        call assert_true(all(line2(:, 1) == line_log(:, 5)) .and. all(cont2(:, 1) == cont_log(:, 5)), &
                         "a model's result does not depend on the batch", total_tests, total_failures)
#endif
        ! FP64 hidden layers (FP32 builds; the same path as the default in FP64 builds)
        call cue_forward(w, N, theta, line2, cont2, chunk=4, fp64=.true.)
        worst = max(maxval(abs(line2 - lref)), maxval(abs(cont2 - cref)))
        write (*, '(A,ES10.2)') '# worst |log L| difference, FP64 hidden layers: ', worst
        call assert_true(worst < 1.0e-11_wp, "FP64 hidden layers match the FP64 evaluation", total_tests, total_failures)
        call cue_forward(w, 0, theta, ll, cl)     ! (no models: returns)
        call cue_unload(w)
    end subroutine test_forward_synthetic

    !> The loader: a synthetic library written as tools/cue/convert_cue.py lays it out reads back
    !> exactly (the forward pass bit-identical), and OPENSPS_CUE_DATA is found as the library
    !> directory or as a data root; unset, cue_load reports it.
    subroutine test_loader()
        interface
            integer(c_int) function setenv(name, value, overwrite) bind(C, name='setenv')
                import :: c_char, c_int
                character(kind=c_char), intent(in) :: name(*), value(*)
                integer(c_int), value :: overwrite
            end function setenv
            integer(c_int) function unsetenv(name) bind(C, name='unsetenv')
                import :: c_char, c_int
                character(kind=c_char), intent(in) :: name(*)
            end function unsetenv
        end interface
        type(cue_weights_t) :: w, r
        character(len=*), parameter :: ROOT = 'cue_test_root', LIB = ROOT // '/libraries/nebular_emulator/cue'
        character(len=4096) :: saved
        integer :: nsaved, st, rc
        real(WP) :: theta(CUE_NIN, 3)
        real(WP), allocatable :: l1(:,:), c1(:,:), l2(:,:), c2(:,:)
        logical :: was_loaded

        call print_group("the loader and OPENSPS_CUE_DATA")
        call cue_synthetic(w)
        call cue_write_library(w, LIB)
        call cue_load_dir(r, LIB)
        call assert_true(r%loaded .and. r%nlines == w%nlines .and. r%ncw == w%ncw .and. r%npc == w%npc .and. &
                         all(r%w2 == w%w2) .and. all(r%alpha == w%alpha) .and. all(r%lcmp == w%lcmp) .and. &
                         all(r%ccmp == w%ccmp) .and. all(r%line_lam == w%line_lam) .and. &
                         all(r%loff == w%loff), "a written library reads back exactly", total_tests, total_failures)
        theta = 2.0_wp
        theta(8, :) = 49.6_wp
        theta(9, :) = 100.0_wp
        allocate(l1(w%nlines, 3), c1(w%ncw, 3), l2(w%nlines, 3), c2(w%ncw, 3))
        call cue_forward(w, 3, theta, l1, c1)
        call cue_forward(r, 3, theta, l2, c2)
        call assert_true(all(l1 == l2) .and. all(c1 == c2), "and gives the same outputs", total_tests, total_failures)
        call cue_unload(r)
        call assert_true(.not. r%loaded .and. .not. allocated(r%w1), "unloading frees it", total_tests, total_failures)

        ! The environment variable (restored afterwards; the module's weights are left as found)
        call get_environment_variable('OPENSPS_CUE_DATA', saved, nsaved)
        was_loaded = cue_w%loaded
        rc = setenv('OPENSPS_CUE_DATA' // c_null_char, LIB // c_null_char, 1)
        call assert_true(cue_data_dir() == LIB, "the library directory itself", total_tests, total_failures)
        rc = setenv('OPENSPS_CUE_DATA' // c_null_char, ROOT // c_null_char, 1)
        call assert_true(cue_data_dir() == LIB, "a data root holding it", total_tests, total_failures)
        rc = setenv('OPENSPS_CUE_DATA' // c_null_char, 'no/such/dir' // c_null_char, 1)
        call assert_true(cue_data_dir() == '', "a directory without it: not found", total_tests, total_failures)
        rc = unsetenv('OPENSPS_CUE_DATA' // c_null_char)
        if (.not. was_loaded) then
            call cue_load(st)
            call assert_true(st == 1 .and. .not. cue_w%loaded, "unset: cue_load reports not found", &
                             total_tests, total_failures)
            rc = setenv('OPENSPS_CUE_DATA' // c_null_char, ROOT // c_null_char, 1)
            call cue_load(st)
            call assert_true(st == 0 .and. cue_w%loaded .and. cue_w%nlines == w%nlines, "set: cue_load reads it once", &
                             total_tests, total_failures)
            call cue_load(st)
            call assert_true(st == 0, "a second cue_load returns at once", total_tests, total_failures)
            call cue_unload(cue_w)
        end if
        if (nsaved > 0) then
            rc = setenv('OPENSPS_CUE_DATA' // c_null_char, trim(saved) // c_null_char, 1)
        else
            rc = unsetenv('OPENSPS_CUE_DATA' // c_null_char)
        end if
        call cue_unload(w)
    end subroutine test_loader

    !> cue's algebra, written out in FP64 on the KP weights.
    subroutine reference_forward(w, n, theta, lref, cref)
        type(cue_weights_t), intent(in) :: w
        integer, intent(in) :: n
        real(WP), intent(in) :: theta(CUE_NIN, n)
        real(WP), intent(out) :: lref(:,:), cref(:,:)
        real(WP) :: x(CUE_NHID), y(CUE_NHID), pc(50), v
        integer :: m, g, l, u, k, p, s, i, nin
        do m = 1, n
            do g = 1, CUE_NNET
                do k = 1, CUE_NIN
                    x(k) = (theta(k, m) - real(w%in_shift(k, g), WP)) / real(w%in_scale(k, g), WP)
                end do
                nin = CUE_NIN
                do l = 1, 3
                    do u = 1, CUE_NHID
                        v = real(w%bias(u, g, l), WP)
                        do k = 1, nin
                            select case (l)
                            case (1); v = v + real(w%w1(u, k, g), WP) * x(k)
                            case (2); v = v + real(w%w2(u, k, g), WP) * x(k)
                            case default; v = v + real(w%w3(u, k, g), WP) * x(k)
                            end select
                        end do
                        y(u) = (real(w%beta(u, g, l), WP) + (1.0_wp - real(w%beta(u, g, l), WP)) / &
                                (1.0_wp + exp(-real(w%alpha(u, g, l), WP) * v))) * v
                    end do
                    x = y
                    nin = CUE_NHID
                end do
                if (g <= CUE_NGROUP) then
                    do p = 1, w%npca(g)
                        v = real(w%lb4(p, g), WP)
                        do k = 1, CUE_NHID
                            v = v + real(w%lw4(p, k, g), WP) * x(k)
                        end do
                        pc(p) = v * real(w%lpsc(p, g), WP) + real(w%lpsh(p, g), WP)
                    end do
                    do s = 1, w%ncount(g)
                        v = real(w%lmean(s, g), WP)
                        do p = 1, w%npca(g)
                            v = v + pc(p) * real(w%lcmp(s, p, g), WP)
                        end do
                        lref(w%loff(g) + s, m) = v * w%lout_sc(s, g) + w%lout_sh(s, g)
                    end do
                else
                    do p = 1, w%npc
                        v = real(w%cb4(p), WP)
                        do k = 1, CUE_NHID
                            v = v + real(w%cw4(p, k), WP) * x(k)
                        end do
                        pc(p) = v * real(w%cpsc(p), WP) + real(w%cpsh(p), WP)
                    end do
                    do i = 1, w%ncw
                        v = real(w%cmean(i), WP)
                        do p = 1, w%npc
                            v = v + pc(p) * real(w%ccmp(i, p), WP)
                        end do
                        cref(i, m) = v * w%cout_sc(i) + w%cout_sh(i)
                    end do
                end if
            end do
        end do
    end subroutine reference_forward

    ! ------------------------------------------------------------------------------------------
    ! Against cue (OPENSPS_CUE_DATA with references)
    ! ------------------------------------------------------------------------------------------

    subroutine test_with_data()
        type(datalib_t) :: lib
        character(len=:), allocatable :: dir
        integer :: st, nref, nfit, nw, m
        real(WP), allocatable :: th(:,:), lr(:,:), cr(:,:), ll(:,:), cl(:,:), fs(:,:), fp(:,:), fw(:), ion(:,:)
        real(KP), allocatable :: fsk(:,:)
        real(WP) :: dl, dc, df
        call print_group("against cue (OPENSPS_CUE_DATA)")
        dir = cue_data_dir()
        if (len(dir) == 0) then
            write (*, '(A)') '# OPENSPS_CUE_DATA is not set (or holds no cue library): skipped'
            return
        end if
        call cue_load(st)
        call assert_true(st == 0 .and. cue_w%loaded .and. cue_w%nlines == 138 .and. cue_w%ncw == 1841, &
                         "the converted library loads (138 lines, 1841 continuum wavelengths)", total_tests, total_failures)
        call datalib_open_dir(lib, dir, 'nebular_emulator', 'cue')
        if (.not. datalib_has_array(lib, 'reference_theta')) then
            write (*, '(A)') '# the library has no references (convert with --reference): skipped'
            return
        end if
        nref = datalib_axis_len(lib, 'reference')
        allocate(th(CUE_NIN, nref), lr(cue_w%nlines, nref), cr(cue_w%ncw, nref), ll(cue_w%nlines, nref), &
                 cl(cue_w%ncw, nref))
        call datalib_read_real(lib, 'reference_theta', int(CUE_NIN * nref, int64), th)
        call datalib_read_real(lib, 'reference_line_log', int(cue_w%nlines * nref, int64), lr)
        call datalib_read_real(lib, 'reference_cont_log', int(cue_w%ncw * nref, int64), cr)
        call cue_forward(cue_w, nref, th, ll, cl)
        dl = maxval(abs(ll - lr))
        dc = maxval(abs(cl - cr))
        call cue_forward(cue_w, nref, th, ll, cl, fp64=.true.)
        write (*, '(A,2ES10.2)') '# FP64 hidden layers (lines, continuum): ', maxval(abs(ll - lr)), maxval(abs(cl - cr))
        call assert_true(max(maxval(abs(ll - lr)), maxval(abs(cl - cr))) < 1.0e-9_wp, &
                         "FP64 hidden layers match cue's FP64 algebra", total_tests, total_failures)
        write (*, '(A,2ES10.2)') '# worst |log L| differences from cue (lines, continuum): ', dl, dc
        ! FP64: summation order only. FP32: the 256-term sums of the hidden layers in FP32 (the
        ! weights are FP32 in both); 1e-3 dex = 0.2%, against cue's own 1-5% accuracy
        call assert_true(dl < merge(1.0e-9_wp, 1.0e-3_wp, KP == WP), "lines match cue's algebra", &
                         total_tests, total_failures)
        call assert_true(dc < merge(1.0e-9_wp, 1.0e-3_wp, KP == WP), "continuum matches cue's algebra", &
                         total_tests, total_failures)
        if (.not. datalib_has_array(lib, 'reference_fit_params')) then
            write (*, '(A)') '# no reference of the power-law fit (SciPy and dill were missing): skipped'
            return
        end if
        nfit = datalib_axis_len(lib, 'fit_case')
        nw = datalib_axis_len(lib, 'fit_wavelength')
        allocate(fs(nw, nfit), fsk(nw, nfit), fp(CUE_NION, nfit), fw(nw), ion(CUE_NION, nfit))
        call datalib_axis_values(lib, 'fit_wavelength', fw)
        call datalib_read_real(lib, 'reference_fit_spec', int(nw * nfit, int64), fs)
        call datalib_read_real(lib, 'reference_fit_params', int(CUE_NION * nfit, int64), fp)
        fsk = real(fs, KP)
        call cue_ionizing_params(nfit, nw, fw, fsk, ion)
        df = 0.0_wp
        do m = 1, nfit
            df = max(df, maxval(abs(ion(:, m) - fp(:, m))))
        end do
        write (*, '(A,ES10.2)') '# worst difference from the minimum of cue''s loss: ', df
        ! The reference minimizes cue's own loss to convergence (cue's SLSQP stops early on some
        ! segments, by up to ~0.1 in a slope); FP32 spectra round the data at ~6e-8
        call assert_true(df < 1.0e-4_wp, "the fit is the minimum of cue's loss", total_tests, total_failures)
    end subroutine test_with_data

end module test_opensps_cue_mod
