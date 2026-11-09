module test_opensps_batch_photometry_mod
    !> The band integrals of the batched photometry (band_fluxes) on synthetic data, at batch sizes
    !> on both sides of every switch between kernels (the NVIDIA chunked kernel below 16 models, the
    !> AMD workgroup kernel below 40, the generic loops above). In a GPU build the data are mapped
    !> to the device first, so the kernel that serves the batch runs; in a CPU build the host loop
    !> runs. Checks: the sums against a double-precision reference, bit-identical results on a
    !> repeat call, an exact scaling identity between models, and skipped bands that are exactly 0.
    use, intrinsic :: iso_fortran_env, only: int64
    use opensps_precision, only: WP, KP
    use opensps_batch_photometry, only: band_fluxes, observed_frame_z
    use opensps_context_types, only: opensps_context_t
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, assert_true
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_opensps_batch_photometry_tests, total_failures, total_tests

    integer, parameter :: NSPEC = 700, NBANDS = 12, NMAX = 100
    !> Band supports (1-based pixels; j + 1 is read, so the last is NSPEC - 1) and which are computed
    integer, parameter :: J_LO(NBANDS) = [1, 1, 699, 100, 250, 1, 1, 300, 10, 100, 2, 600]
    integer, parameter :: J_HI(NBANDS) = [699, 1, 699, 100, 260, 256, 257, 655, 400, 611, 40, 699]
    integer, parameter :: DO_BAND(NBANDS) = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
    integer(int64) :: seed = 20261107_int64

contains

    subroutine run_opensps_batch_photometry_tests()
        call print_minor_header("opensps_batch_photometry")
        call test_band_fluxes()
#if !defined(OMP_TARGET)
        call test_observed_frame_z()
#else
        write (*, '(A)') '# observed_frame_z: tested on host builds (it reads the context, whose grid the test does not map)'
#endif
        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_opensps_batch_photometry_tests

    !> Uniform in [0, 1); a 31-bit generator in 64-bit arithmetic, no overflow
    real(WP) function urand()
        seed = iand(1103515245_int64 * seed + 12345_int64, 2147483647_int64)
        urand = real(seed, WP) / 2147483648.0_wp
    end function urand

    subroutine test_band_fluxes()
        integer, parameter :: SIZES(10) = [1, 2, 7, 15, 16, 17, 39, 40, 41, NMAX]
        real(KP), allocatable :: bands_t(:, :), wt(:), fobs_all(:, :), fobs_t(:, :)
        real(WP), allocatable :: fband(:, :), fband2(:, :)
        real(WP) :: ref, worst, tol, rel
        integer :: k, n, b, m, j
        character(len=64) :: label
        logical :: ok_ref, ok_repeat, ok_scale, ok_zero, ok_skip

        call print_group("band_fluxes against a reference, and its identities")

        ! The data are drawn once; every batch takes the first n models. Model 2 is twice model 1 (an
        ! exact scaling in binary arithmetic), model 3 has no flux
        allocate(bands_t(NBANDS, NSPEC), wt(NSPEC), fobs_all(NMAX, NSPEC))
        do j = 1, NSPEC
            wt(j) = real(0.5_wp + urand(), KP)
            do b = 1, NBANDS
                bands_t(b, j) = real(urand(), KP)
            end do
            do m = 1, NMAX
                fobs_all(m, j) = real(2.0_wp * urand(), KP)
            end do
            fobs_all(2, j) = 2.0_KP * fobs_all(1, j)
            fobs_all(3, j) = 0.0_KP
        end do

        ! FP64 kernels differ from the reference only by the order of the sums; FP32 kernels sum
        ! compensated, the products round once (100 eps of KP, as the other tests)
        tol = merge(1.0e-11_wp, 100.0_wp * real(epsilon(1.0_KP), WP), KP == WP)

        do k = 1, size(SIZES)
            n = SIZES(k)
            allocate(fobs_t(n, NSPEC), fband(NBANDS, n), fband2(NBANDS, n))
            fobs_t = fobs_all(1:n, :)
            fband = -1.0_wp
            fband2 = -1.0_wp

            call run_once(n, bands_t, wt, fobs_t, fband)
            call run_once(n, bands_t, wt, fobs_t, fband2)

            ! Reference: the same trapezoid sums in double precision from the same stored values
            worst = 0.0_wp
            ok_skip = .true.
            do m = 1, n
                do b = 1, NBANDS
                    if (DO_BAND(b) == 0) then
                        ok_skip = ok_skip .and. fband(b, m) == 0.0_wp
                        cycle
                    end if
                    ref = 0.0_wp
                    do j = J_LO(b), J_HI(b)
                        ref = ref + real(wt(j), WP) * (real(fobs_t(m, j), WP) * real(bands_t(b, j), WP) + &
                                                       real(fobs_t(m, j + 1), WP) * real(bands_t(b, j + 1), WP))
                    end do
                    if (ref > 0.0_wp) then
                        rel = abs(fband(b, m) - ref) / ref
                    else
                        rel = abs(fband(b, m))
                    end if
                    worst = max(worst, rel)
                end do
            end do
            ok_ref = worst <= tol
            ok_repeat = all(fband == fband2)
            ok_scale = .true.
            ok_zero = .true.
            if (n >= 2) ok_scale = all(fband(:, 2) == 2.0_wp * fband(:, 1))
            if (n >= 3) ok_zero = all(fband(:, 3) == 0.0_wp)

            write (label, '(A,I0,A)') "N = ", n, ": "
            write (*, '(A,A,ES10.2)') '# ', trim(label) // "worst relative difference from the reference", worst
            call assert_true(ok_ref, trim(label) // "sums match the double-precision reference", &
                             total_tests, total_failures)
            call assert_true(ok_repeat, trim(label) // "a repeat call gives bit-identical results", &
                             total_tests, total_failures)
            call assert_true(ok_scale, trim(label) // "twice the flux gives exactly twice the band flux", &
                             total_tests, total_failures)
            call assert_true(ok_zero, trim(label) // "a model with no flux gives exactly 0", &
                             total_tests, total_failures)
            call assert_true(ok_skip, trim(label) // "skipped bands are exactly 0", total_tests, total_failures)

            deallocate(fobs_t, fband, fband2)
        end do
    end subroutine test_band_fluxes


#if !defined(OMP_TARGET)
    !> observed_frame_z against a direct evaluation: per (pixel, column) the rest-frame wavelength lambda / (1 + z),
    !> the interval it falls in (a bisection, as find_interval), the linear interpolation of the total there,
    !> NaN to 0, negatives to 0, times 1 / lambda weights; in both layouts (columns x pixels as band_fluxes reads
    !> it, and pixels x columns with zero padding rows), for redshifts below, at and above the grid's range
    !> (out-of-range pixels extrapolate from the end interval, as legacy locate does) and a rest-frame column
    subroutine test_observed_frame_z()
        integer, parameter :: NS = 70, NC = 9, NPAD = 80
        type(opensps_context_t) :: ctx
        real(WP), target :: lam(NS)
        real(WP) :: zc(NC), tg, t, v, ref
        real(KP) :: wl(NS), total(NS, NC), ft(NC, NS), fp(NPAD, NC), worst
        integer :: j, m, il, lo, hi, mid
        logical :: ok_t, ok_p, ok_pad

        call print_group("observed_frame_z against a direct evaluation, in both layouts")
        ! A non-uniform ascending grid, so that the search from the previous pixel's interval must move by
        ! different amounts (and stay put where the observed pixels are denser than the rest-frame ones)
        do j = 1, NS
            lam(j) = 900.0_wp * exp(0.045_wp * real(j - 1, WP)) * (1.0_wp + 0.01_wp * sin(real(j, WP)))
        end do
        ctx%state%spec_lambda => lam
        do j = 1, NS
            wl(j) = real(1.0_wp / lam(j), KP)
        end do
        do m = 1, NC
            do j = 1, NS
                total(j, m) = real(1.0_wp + urand(), KP)
            end do
        end do
        total(5, 3) = real(ieee_value(1.0_WP, ieee_quiet_nan), KP)   ! NaN gives 0
        total(9, 4) = -1.0_KP         ! negative gives 0
        zc = [0.0_wp, 0.05_wp, 0.4_wp, 1.0_wp, 2.5_wp, 6.0_wp, -0.3_wp, 1.0e-12_wp, 0.7_wp]
        ft = -1.0_KP
        fp = -1.0_KP

        call observed_frame_z(ctx, NC, NS, NS, .true., zc, wl, total, ft)
        call observed_frame_z(ctx, NC, NS, NPAD, .false., zc, wl, total, fp)

        ok_t = .true.
        ok_p = .true.
        worst = 0.0_KP
        do m = 1, NC
            do j = 1, NS
                if (abs(zc(m)) > tiny(1.0_WP)) then
                    tg = lam(j) / (1.0_wp + zc(m))
                    lo = 0
                    hi = NS + 1
                    do while (hi - lo > 1)
                        mid = (hi + lo) / 2
                        if (tg >= lam(mid)) then
                            lo = mid
                        else
                            hi = mid
                        end if
                    end do
                    il = max(1, min(lo, NS - 1))
                    t = (tg - lam(il)) / (lam(il + 1) - lam(il))
                    v = real(total(il, m), WP) + t * (real(total(il + 1, m), WP) - real(total(il, m), WP))
                else
                    v = real(total(j, m), WP)
                end if
                if (v /= v) v = 0.0_wp
                ref = real(real(max(v, 0.0_wp), KP) * wl(j), WP)
                if (abs(real(ft(m, j), WP) - ref) > 4.0_wp * epsilon(1.0_KP) * abs(ref)) ok_t = .false.
                if (abs(real(fp(j, m), WP) - ref) > 4.0_wp * epsilon(1.0_KP) * abs(ref)) ok_p = .false.
                worst = max(worst, abs(ft(m, j) - fp(j, m)))
            end do
        end do
        ok_pad = all(fp(NS + 1:NPAD, :) == 0.0_KP)
        write (*, '(A,ES10.2)') '# largest difference between the two layouts: ', real(worst, WP)
        call assert_true(ok_t, "transposed layout matches the direct evaluation", total_tests, total_failures)
        call assert_true(ok_p, "pixels x columns layout matches the direct evaluation", total_tests, total_failures)
        call assert_true(worst == 0.0_KP, "the two layouts agree bit for bit", total_tests, total_failures)
        call assert_true(ok_pad, "padding rows are exactly 0", total_tests, total_failures)
    end subroutine test_observed_frame_z
#endif

    !> One call with the data resident on the device in a GPU build (the kernels expect it)
    subroutine run_once(n, bands_t, wt, fobs_t, fband)
        integer, intent(in) :: n
        real(KP), intent(in) :: bands_t(NBANDS, NSPEC), wt(NSPEC), fobs_t(n, NSPEC)
        real(WP), intent(inout) :: fband(NBANDS, n)
        integer :: jl(NBANDS), jh(NBANDS), db(NBANDS)

        jl = J_LO
        jh = J_HI
        db = DO_BAND
#ifdef OMP_TARGET
        !$omp target data map(to: bands_t, wt, fobs_t, jl, jh, db) map(from: fband)
#endif
        call band_fluxes(n, NSPEC, NBANDS, bands_t, jl, jh, db, wt, fobs_t, fband)
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_once

end module test_opensps_batch_photometry_mod
