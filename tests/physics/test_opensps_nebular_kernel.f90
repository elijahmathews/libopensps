module test_opensps_nebular_kernel_mod
    !> The nebular continuum and line kernel (nebular_continuum_tiled) on synthetic data, at batch sizes on both
    !> sides of the grouping of models by (Z, U) cell (up to 8 models share each thread's table reads in the HIP
    !> kernel), with models sharing one cell, each in its own cell, and mixed, and an age-interval sequence that
    !> skips an interval. In a GPU build the data are mapped to the device first, so the kernel runs; the tests
    !> need the LLVM flang offload build (the routine is not built otherwise: they are skipped with a message).
    !> Checks: the sums against a double-precision direct evaluation of C(t) 10**lg(t) (lg interpolated in Z, U
    !> and age), that a model's result does not depend on the other models of the batch (bit-identical), and that
    !> the kernel adds to what the output holds.
    use, intrinsic :: iso_fortran_env, only: int64
    use opensps_precision, only: WP, KP
    use opensps_constants, only: NEBNZ, NEBNAGE, NEBNIP
#if defined(OMP_TARGET) && defined(__flang__)
    use opensps_csp_batch, only: nebular_continuum_tiled
#endif
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, assert_true
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_opensps_nebular_kernel_tests, total_failures, total_tests

    integer, parameter :: NTAB = 300, NT = 29, NMAX = 120
    integer(int64) :: seed = 20261109_int64

contains

    subroutine run_opensps_nebular_kernel_tests()
        call print_minor_header("opensps_nebular_kernel")
#if defined(OMP_TARGET) && defined(__flang__)
        call test_kernel()
#else
        write (*, '(A)') '# nebular_continuum_tiled is built only with LLVM flang offload: skipped'
#endif
        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_opensps_nebular_kernel_tests

    real(WP) function urand()
        seed = iand(1103515245_int64 * seed + 12345_int64, 2147483647_int64)
        urand = real(seed, WP) / 2147483648.0_wp
    end function urand

#if defined(OMP_TARGET) && defined(__flang__)
    subroutine test_kernel()
        integer, parameter :: SIZES(8) = [1, 2, 7, 8, 9, 31, 64, NMAX]
        real(WP), parameter :: LN10 = log(10.0_wp)
        real(WP), allocatable :: tab(:, :, :, :)
        real(WP) :: wa(NT), dwa(NT), lns(NMAX), wz(NMAX), wu(NMAX), worst, ref_y, ref_o, ra, rb, lg, tol
        real(KP) :: cy(NT, NMAX), co(NT, NMAX)
        real(KP), allocatable :: yo(:, :), yo1(:, :), yo_full(:, :)
        integer :: ia(NT), iz(NMAX), iu(NMAX), mode, k, n, m, j, t, a, c, nbad
        logical :: newr(NT), ok_ref, ok_alone, ok_add
        character(len=64) :: label

        call print_group("nebular_continuum_tiled against a direct evaluation, and its identities")

        ! A log grid (log10 of a flux per Hz, ~1e-44) with structure in every index
        allocate(tab(NTAB, NEBNZ, NEBNAGE, NEBNIP))
        do c = 1, NEBNIP
            do a = 1, NEBNAGE
                do k = 1, NEBNZ
                    do j = 1, NTAB
                        tab(j, k, a, c) = -43.0_wp + 2.0_wp * urand() - 0.3_wp * a + 0.4_wp * c - 0.1_wp * k
                    end do
                end do
            end do
        end do
        ! Ages: intervals 1 1 1 1 2 2 2 3 3 ... with interval 5 skipped; the position in the interval advances by
        ! 1/4 (the geometric progression of the kernel), the step is flagged at the second age of an interval
        a = 1
        k = 0
        do t = 1, NT
            if (t > 1 .and. mod(t - 1, 4) == 0) then
                a = a + 1
                if (a == 5) a = 6
                k = 0
            end if
            ia(t) = a
            wa(t) = 0.25_wp * k
            dwa(t) = merge(0.25_wp, 0.0_wp, k > 0)
            newr(t) = k == 1
            k = k + 1
        end do
        ! Models: the weights (zero for some ages, young and old: one of the two per age), ln S, the cell
        do m = 1, NMAX
            do t = 1, NT
                cy(t, m) = 0.0_KP
                co(t, m) = 0.0_KP
                if (urand() < 0.85_wp) then
                    if (t <= 12) then
                        cy(t, m) = real(urand(), KP)
                    else
                        co(t, m) = real(urand(), KP)
                    end if
                end if
            end do
            lns(m) = 80.0_wp + 30.0_wp * urand()
            wz(m) = urand()
            wu(m) = urand()
        end do

        ! FP32 kernels: the exponent (~100) is reduced in FP64, the offsets in FP32 (< 3 in log10): ~1e-7 in the
        ! exponent; the FP64 kernels agree to the order of the sums
        tol = merge(1.0e-11_wp, 2.0e-6_wp, KP == WP)

        allocate(yo_full(NTAB, 2 * NMAX))
        do mode = 1, 3
            ! 1: one cell for all models; 2: every model in its own cell as far as possible; 3: mixed
            do m = 1, NMAX
                select case (mode)
                case (1)
                    iz(m) = 4
                    iu(m) = 3
                case (2)
                    iz(m) = 1 + mod(m - 1, NEBNZ - 1)
                    iu(m) = 1 + mod((m - 1) / (NEBNZ - 1), NEBNIP - 1)
                case default
                    iz(m) = 1 + int(urand() * 3.0_wp) + 2 * mod(m, 2)
                    iu(m) = 1 + int(urand() * 2.0_wp)
                end select
            end do

            ! The full batch once (the models alone are compared against it below)
            yo_full = 0.0_KP
            call run_kernel(NMAX, yo_full)

            do k = 1, size(SIZES)
                n = SIZES(k)
                allocate(yo(NTAB, 2 * n))
                yo = 0.0_KP
                call run_kernel(n, yo)

                ! Reference: C(t) 10**lg(t) in double precision, lg(t) = ra + wa (rb - ra) in the cell's interpolation
                worst = 0.0_wp
                nbad = 0
                do m = 1, n
                    do j = 1, NTAB
                        ref_y = 0.0_wp
                        ref_o = 0.0_wp
                        do t = 1, NT
                            a = ia(t)
                            ra = interp(j, iz(m), iu(m), a, wz(m), wu(m))
                            rb = interp(j, iz(m), iu(m), a + 1, wz(m), wu(m))
                            lg = (ra + wa(t) * (rb - ra)) * LN10 + lns(m)
                            ref_y = ref_y + real(cy(t, m), WP) * exp(lg)
                            ref_o = ref_o + real(co(t, m), WP) * exp(lg)
                        end do
                        call compare(real(yo(j, m), WP), ref_y, worst, nbad)
                        call compare(real(yo(j, n + m), WP), ref_o, worst, nbad)
                    end do
                end do
                ok_ref = worst <= tol
                ! The models alone or in a batch of any size give the same bits as in the full batch
                ok_alone = .true.
                do m = 1, n
                    ok_alone = ok_alone .and. all(yo(:, m) == yo_full(:, m)) .and. &
                               all(yo(:, n + m) == yo_full(:, NMAX + m))
                end do

                write (label, '(A,I0,A,I0,A)') "mode ", mode, ", N = ", n, ": "
                write (*, '(A,A,ES10.2)') '# ', trim(label) // "worst relative difference from the reference", worst
                call assert_true(ok_ref, trim(label) // "sums match the double-precision evaluation", &
                                 total_tests, total_failures)
                call assert_true(ok_alone, trim(label) // "a model's result does not depend on its batch (bit-identical)", &
                                 total_tests, total_failures)
                deallocate(yo)
            end do
        end do

        ! The kernel adds to the output (the EUV rows of the continuum are already there)
        allocate(yo(NTAB, 2 * NMAX), yo1(NTAB, 2 * NMAX))
        yo = 0.0_KP
        call run_kernel(NMAX, yo)
        yo1 = 1.0_KP
        call run_kernel(NMAX, yo1)
        ok_add = all(abs(yo1 - (yo + 1.0_KP)) <= 4.0_wp * epsilon(1.0_KP) * (yo + 1.0_KP))
        call assert_true(ok_add, "the kernel adds to the existing output", total_tests, total_failures)
        deallocate(yo, yo1, yo_full, tab)

    contains

        subroutine compare(got, ref, worst_, nbad_)
            real(WP), intent(in) :: got, ref
            real(WP), intent(inout) :: worst_
            integer, intent(inout) :: nbad_
            real(WP) :: rel
            ! Values many decades below the typical are not accurate in relative terms and do not matter: skip
            ! the references below 1e-30 of the exponent range (the largest are ~1e9)
            if (abs(ref) < 1.0e-25_wp) then
                worst_ = max(worst_, merge(0.0_wp, abs(got), abs(got) < 1.0e-25_wp))
                return
            end if
            rel = abs(got - ref) / abs(ref)
            worst_ = max(worst_, rel)
            if (rel > 1.0e-3_wp) nbad_ = nbad_ + 1
        end subroutine compare

        real(WP) function interp(j_, iz_, iu_, a_, wz_, wu_)
            integer, intent(in) :: j_, iz_, iu_, a_
            real(WP), intent(in) :: wz_, wu_
            interp = (1.0_wp - wz_) * (1.0_wp - wu_) * tab(j_, iz_, a_, iu_) + (1.0_wp - wz_) * wu_ * tab(j_, iz_, a_, iu_ + 1) &
                     + wz_ * (1.0_wp - wu_) * tab(j_, iz_ + 1, a_, iu_) + wz_ * wu_ * tab(j_, iz_ + 1, a_, iu_ + 1)
        end function interp

        !> One call on the first n models, the data resident on the device in a GPU build
        subroutine run_kernel(n_, out)
            integer, intent(in) :: n_
            real(KP), intent(inout) :: out(NTAB, 2 * n_)
            real(KP) :: cy_(NT, n_), co_(NT, n_), out_in(NTAB, 2 * n_)
            real(WP) :: lns_(n_), wz_(n_), wu_(n_)
            integer :: iz_(n_), iu_(n_)
            cy_ = cy(:, 1:n_)
            co_ = co(:, 1:n_)
            lns_ = lns(1:n_)
            wz_ = wz(1:n_)
            wu_ = wu(1:n_)
            iz_ = iz(1:n_)
            iu_ = iu(1:n_)
            !$omp target data map(to: tab, ia, wa, dwa, cy_, co_, lns_, iz_, wz_, iu_, wu_) map(tofrom: out)
            call nebular_continuum_tiled(n_, NTAB, NTAB, NT, tab, ia, wa, dwa, newr, cy_, co_, lns_, iz_, wz_, iu_, wu_, out)
            !$omp end target data
        end subroutine run_kernel

    end subroutine test_kernel
#endif

end module test_opensps_nebular_kernel_mod
