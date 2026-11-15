module opensps_filters
    !> @brief
    !> Bandpass (filter) curves: the filter-set object and the load-time pipeline that turns tabulated
    !> curves into the `bands` array the photometry kernels integrate (doc/FILTERS_PLAN.md).
    !>
    !> @details
    !> Everything here runs when filters are set, never in a kernel. The pipeline per filter is
    !> clean -> (energy detectors: T' = T/lambda) -> resample onto the model wavelength grid ->
    !> normalize to integral T'/lambda dlambda = 1 -> (optional) calibrate -> zero points.
    !>
    !> **Detectors.** The kernels evaluate the photon-counting form
    !> `<f_nu> = integral f_nu T dlambda/lambda / integral T dlambda/lambda`. For an energy-counting
    !> detector `<f_nu> = integral f_nu T dnu / integral T dnu = integral f_nu (T/lambda^2) dlambda /
    !> integral T/lambda^2 dlambda`, which is the photon form with T' = T/lambda (an identity).
    !>
    !> **Resampling.** `RESAMPLE_LINEAR` is the legacy FSPS point interpolation (kept bit-identical for
    !> the legacy list). `RESAMPLE_CONSERVING` projects the piecewise-linear curve on the hat functions
    !> of the grid, `T_i = integral T h_i dlambda / w_i` with `w_i` the trapezoid weight of node i, so
    !> that `sum_i w_i T_i` equals the integral of T over the grid range exactly, independent of the
    !> grid spacing (the kernels' trapezoid rule is `sum_i w_i f_i T_i`).
    !>
    !> **Content hash.** SHA-256 of the cleaned curve: a domain tag, detector, calibration, n,
    !> wavelengths and response (little-endian IEEE binary64 and int64). Names and provenance are
    !> not hashed. Cleaning (repeated wavelengths dropped, negative response clamped) precedes the
    !> hash, so it identifies what is used.
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use opensps_precision, only: WP
    use opensps_filter_types, only: filter_info_t, filter_info_clear, RESAMPLE_LINEAR, RESAMPLE_CONSERVING, &
                                    GRID_POLICY_ERROR, GRID_POLICY_WARN, GRID_POLICY_ALLOW
    use opensps_constants, only: SAFE_FLOOR
    use opensps_integration, only: integrate_trapezoid_array
    use opensps_interpolation, only: interpolate_linear, find_interval
    use opensps_sha256, only: sha256_t, sha256_init, sha256_update_int64, sha256_update_real64, &
                              sha256_update_string, sha256_hex
    implicit none
    private

    public :: filter_set_t, filter_info_t
    public :: filter_set_add, filter_set_init_bulk, filter_set_clear
    public :: filter_clean, filters_build, filter_calibrate_band, filter_pivot, filters_zero_points
    public :: filter_hash, grid_hash, filter_info_clear, resample_hat, resample_point
    public :: DETECTOR_PHOTON, DETECTOR_ENERGY, RESAMPLE_LINEAR, RESAMPLE_CONSERVING
    public :: GRID_POLICY_ERROR, GRID_POLICY_WARN, GRID_POLICY_ALLOW
    public :: FILTER_FLAG_TRUNCATED, FILTER_FLAG_UNDERSAMPLED, FILTER_FLAG_DEAD
    public :: FILTER_OK, FILTER_ERR_FEW_POINTS, FILTER_ERR_NONFINITE, FILTER_ERR_ORDER, FILTER_ERR_ARGUMENT, &
              FILTER_ERR_COVERAGE, FILTER_ERR_DEAD, FILTER_ERR_HASH
    public :: FILTER_MIN_COVERAGE, FILTER_MIN_FWHM_POINTS

    integer, parameter :: DETECTOR_PHOTON = 0       !< response per photon (FSPS, sedpy default, SVO "photon counter")
    integer, parameter :: DETECTOR_ENERGY = 1       !< response per unit energy ("energy counter")

    !> Bits of filter_info_t%flags
    integer, parameter :: FILTER_FLAG_TRUNCATED = 1    !< coverage below FILTER_MIN_COVERAGE
    integer, parameter :: FILTER_FLAG_UNDERSAMPLED = 2 !< fewer than FILTER_MIN_FWHM_POINTS grid points above half maximum
    integer, parameter :: FILTER_FLAG_DEAD = 4         !< no throughput on the grid: the band is zero

    integer, parameter :: FILTER_OK = 0
    integer, parameter :: FILTER_ERR_FEW_POINTS = 1   !< fewer than 2 distinct wavelengths
    integer, parameter :: FILTER_ERR_NONFINITE = 2    !< NaN/Inf, or a wavelength <= 0
    integer, parameter :: FILTER_ERR_ORDER = 3        !< wavelengths neither increasing nor decreasing
    integer, parameter :: FILTER_ERR_ARGUMENT = 4     !< bad detector, mismatched sizes, bad calibration
    integer, parameter :: FILTER_ERR_COVERAGE = 5     !< policy error: filter beyond the model grid
    integer, parameter :: FILTER_ERR_DEAD = 6         !< policy error: no throughput on the grid
    integer, parameter :: FILTER_ERR_HASH = 7         !< library hash does not match the curve

    !> A filter counts as covered if at least this fraction of its integral of T'/lambda lies on the
    !> grid. 1e-3 is below any plausible difference between curve tabulations of one filter and
    !> above the truncation of a curve whose tail merely reaches past the grid end.
    real(WP), parameter :: FILTER_MIN_COVERAGE = 0.999_wp
    !> Fewer grid points than this above half of the band's maximum cannot represent the shape
    !> (the trapezoid rule needs several nodes across the passband)
    integer, parameter :: FILTER_MIN_FWHM_POINTS = 5

    !> A set of tabulated filters (wavelengths in Angstrom; points concatenated, npts per filter)
    type :: filter_set_t
        integer :: n = 0
        character(len=64), allocatable :: name(:)
        integer, allocatable :: npts(:), detector(:)
        logical, allocatable :: has_cal(:)
        real(WP), allocatable :: cal_lref(:), cal_expo(:)
        real(WP), allocatable :: lam(:), resp(:)
        !> Optional (blank: none) hash each filter must have after cleaning, from a library manifest
        character(len=64), allocatable :: expect_hash(:)
    end type filter_set_t

contains

    subroutine filter_set_clear(set)
        type(filter_set_t), intent(inout) :: set
        set%n = 0
        if (allocated(set%name)) deallocate (set%name, set%npts, set%detector, set%has_cal, set%cal_lref, &
                                             set%cal_expo, set%lam, set%resp)
        if (allocated(set%expect_hash)) deallocate (set%expect_hash)
    end subroutine filter_set_clear

    !> Appends one filter. status FILTER_ERR_ARGUMENT for bad sizes, detector or calibration.
    subroutine filter_set_add(set, name, lam, resp, detector, has_cal, lref, expo, status)
        type(filter_set_t), intent(inout) :: set
        character(len=*), intent(in) :: name
        real(WP), intent(in) :: lam(:), resp(:)
        integer, intent(in) :: detector
        logical, intent(in) :: has_cal
        real(WP), intent(in) :: lref, expo
        integer, intent(out) :: status

        type(filter_set_t) :: new
        integer :: n, np
        integer, allocatable :: ntmp(:)

        status = FILTER_OK
        np = size(lam)
        if (np /= size(resp) .or. np < 2 .or. (detector /= DETECTOR_PHOTON .and. detector /= DETECTOR_ENERGY)) then
            status = FILTER_ERR_ARGUMENT
            return
        end if
        if (has_cal) then
            if (.not. (lref > 0.0_wp) .or. lref /= lref .or. expo /= expo) then
                status = FILTER_ERR_ARGUMENT
                return
            end if
        end if
        n = set%n + 1
        allocate (new%name(n), new%npts(n), new%detector(n), new%has_cal(n), new%cal_lref(n), new%cal_expo(n), &
                  new%expect_hash(n))
        new%expect_hash = ''
        if (set%n > 0) then
            if (allocated(set%expect_hash)) new%expect_hash(1:set%n) = set%expect_hash(1:set%n)
            new%name(1:set%n) = set%name(1:set%n)
            new%npts(1:set%n) = set%npts(1:set%n)
            new%detector(1:set%n) = set%detector(1:set%n)
            new%has_cal(1:set%n) = set%has_cal(1:set%n)
            new%cal_lref(1:set%n) = set%cal_lref(1:set%n)
            new%cal_expo(1:set%n) = set%cal_expo(1:set%n)
            allocate (new%lam(sum(set%npts) + np), new%resp(sum(set%npts) + np))
            new%lam(1:sum(set%npts)) = set%lam
            new%resp(1:sum(set%npts)) = set%resp
        else
            allocate (new%lam(np), new%resp(np))
        end if
        new%name(n) = name
        new%npts(n) = np
        new%detector(n) = detector
        new%has_cal(n) = has_cal
        new%cal_lref(n) = merge(lref, 1.0_wp, has_cal)
        new%cal_expo(n) = merge(expo, 0.0_wp, has_cal)
        ntmp = new%npts
        new%lam(sum(ntmp(1:n - 1)) + 1:sum(ntmp(1:n))) = lam
        new%resp(sum(ntmp(1:n - 1)) + 1:sum(ntmp(1:n))) = resp
        new%n = n
        call move_alloc(new%name, set%name); call move_alloc(new%npts, set%npts)
        call move_alloc(new%detector, set%detector); call move_alloc(new%has_cal, set%has_cal)
        call move_alloc(new%cal_lref, set%cal_lref); call move_alloc(new%cal_expo, set%cal_expo)
        call move_alloc(new%lam, set%lam); call move_alloc(new%resp, set%resp)
        call move_alloc(new%expect_hash, set%expect_hash)
        set%n = n
    end subroutine filter_set_add

    !> A set of n filters from concatenated curves (used by the readers of the legacy and standard
    !> layouts); detectors are photon counting, no calibration, names blank until set by the caller.
    subroutine filter_set_init_bulk(set, npts, lam, resp)
        type(filter_set_t), intent(inout) :: set
        integer, intent(in) :: npts(:)
        real(WP), intent(in) :: lam(:), resp(:)
        integer :: n
        call filter_set_clear(set)
        n = size(npts)
        set%n = n
        allocate (set%name(n), set%npts(n), set%detector(n), set%has_cal(n), set%cal_lref(n), set%cal_expo(n), &
                  set%expect_hash(n))
        set%expect_hash = ''
        set%name = ''
        set%npts = npts
        set%detector = DETECTOR_PHOTON
        set%has_cal = .false.
        set%cal_lref = 1.0_wp
        set%cal_expo = 0.0_wp
        allocate (set%lam(sum(npts)), set%resp(sum(npts)))
        set%lam = lam(1:sum(npts))
        set%resp = resp(1:sum(npts))
    end subroutine filter_set_init_bulk

    !> Cleans one tabulated curve: repeated wavelengths are dropped (the first kept), negative
    !> response is clamped to zero. Strict mode (user curves) also rejects NaN/Inf, wavelengths <= 0
    !> and non-monotonic wavelengths (a strictly decreasing curve is reversed), and needs two
    !> distinct points. Non-strict mode is the legacy reader's: no checks beyond the two edits.
    subroutine filter_clean(lam, resp, strict, lc, rc, nc, status)
        real(WP), intent(in) :: lam(:), resp(:)
        logical, intent(in) :: strict
        real(WP), allocatable, intent(out) :: lc(:), rc(:)
        integer, intent(out) :: nc, status

        real(WP), allocatable :: l(:), r(:)
        integer :: i, n

        status = FILTER_OK
        n = size(lam)
        allocate (lc(n), rc(n))
        nc = 0
        if (strict) then
            if (any(lam /= lam) .or. any(resp /= resp) .or. &
                any(abs(lam) > huge(1.0_wp)) .or. any(abs(resp) > huge(1.0_wp)) .or. any(lam <= 0.0_wp)) then
                status = FILTER_ERR_NONFINITE
                return
            end if
        end if
        allocate (l(n), r(n))
        l = lam
        r = resp
        if (strict .and. n > 1) then
            if (l(n) < l(1)) then
                l = l(n:1:-1)
                r = r(n:1:-1)
            end if
        end if
        do i = 1, n
            if (nc > 0) then
                if (abs(l(i) - lc(nc)) < SAFE_FLOOR) cycle
            end if
            nc = nc + 1
            lc(nc) = l(i)
            rc(nc) = max(r(i), 0.0_wp)
        end do
        if (strict) then
            if (any(lc(2:nc) <= lc(1:nc - 1))) then
                status = FILTER_ERR_ORDER
                return
            end if
            if (nc < 2) status = FILTER_ERR_FEW_POINTS
        end if
    end subroutine filter_clean

    !> SHA-256 (64 hex characters) of a cleaned curve with its detector and calibration.
    function filter_hash(lc, rc, nc, detector, has_cal, lref, expo) result(hex)
        real(WP), intent(in) :: lc(:), rc(:)
        integer, intent(in) :: nc, detector
        logical, intent(in) :: has_cal
        real(WP), intent(in) :: lref, expo
        character(len=64) :: hex
        type(sha256_t) :: s

        call sha256_init(s)
        call sha256_update_string(s, 'OPENSPS-FILTER-v1')
        call sha256_update_int64(s, int(detector, int64))
        call sha256_update_int64(s, merge(1_int64, 0_int64, has_cal))
        if (has_cal) call sha256_update_real64(s, [real(lref, real64), real(expo, real64)])
        call sha256_update_int64(s, int(nc, int64))
        call sha256_update_real64(s, real(lc(1:nc), real64))
        call sha256_update_real64(s, real(rc(1:nc), real64))
        hex = sha256_hex(s)
    end function filter_hash

    !> SHA-256 of the model wavelength grid
    function grid_hash(grid) result(hex)
        real(WP), intent(in) :: grid(:)
        character(len=64) :: hex
        type(sha256_t) :: s
        call sha256_init(s)
        call sha256_update_string(s, 'OPENSPS-GRID-v1')
        call sha256_update_int64(s, int(size(grid), int64))
        call sha256_update_real64(s, real(grid, real64))
        hex = sha256_hex(s)
    end function grid_hash

    !> Integral of r/lambda^p over [lo, hi] (p = 1: photon counting, p = 2: energy counting, the
    !> weights of <f_nu>) of the piecewise-linear curve (lc, r): exact, per segment r = alpha + beta
    !> lambda gives alpha ln(b/a) + beta (b - a) for p = 1 and alpha (1/a - 1/b) + beta ln(b/a) for p = 2.
    pure function curve_integral(lc, r, nc, lo, hi, p) result(s)
        real(WP), intent(in) :: lc(:), r(:), lo, hi
        integer, intent(in) :: nc, p
        real(WP) :: s
        integer :: k
        real(WP) :: a, b, ya, yb, t, beta, alpha

        s = 0.0_wp
        do k = 1, nc - 1
            a = max(lc(k), lo)
            b = min(lc(k + 1), hi)
            if (b <= a) cycle
            t = (a - lc(k))/(lc(k + 1) - lc(k))
            ya = r(k) + t*(r(k + 1) - r(k))
            t = (b - lc(k))/(lc(k + 1) - lc(k))
            yb = r(k) + t*(r(k + 1) - r(k))
            beta = (yb - ya)/(b - a)
            alpha = ya - beta*a
            if (p == 2) then
                s = s + alpha*(1.0_wp/a - 1.0_wp/b) + beta*log(b/a)
            else
                s = s + alpha*log(b/a) + beta*(b - a)
            end if
        end do
    end function curve_integral

    !> Index j with grid(j) <= x < grid(j+1) (1 below the grid, ng - 1 at or above its end)
    pure function cell_of(grid, x) result(j)
        real(WP), intent(in) :: grid(:), x
        integer :: j, lo, hi, mid
        lo = 1
        hi = size(grid)
        if (x <= grid(1)) then
            j = 1
            return
        end if
        if (x >= grid(hi)) then
            j = hi - 1
            return
        end if
        do while (hi - lo > 1)
            mid = (lo + hi)/2
            if (grid(mid) <= x) then
                lo = mid
            else
                hi = mid
            end if
        end do
        j = lo
    end function cell_of

    !> Hat-function projection of the piecewise-linear curve (lc, r) onto the grid (see module doc).
    subroutine resample_hat(lc, r, nc, grid, band)
        real(WP), intent(in) :: lc(:), r(:), grid(:)
        integer, intent(in) :: nc
        real(WP), intent(out) :: band(:)

        integer :: k, j, ng
        real(WP) :: x0, x1, y0, y1, a, b, ya, yb, dx, ym, ha, hb, hm, t, m
        real(WP) :: acc(size(grid))

        ng = size(grid)
        acc = 0.0_wp
        do k = 1, nc - 1
            x0 = max(lc(k), grid(1))
            x1 = min(lc(k + 1), grid(ng))
            if (x1 <= x0) cycle
            j = cell_of(grid, x0)
            do while (j <= ng - 1)
                if (grid(j) >= x1) exit
                a = max(x0, grid(j))
                b = min(x1, grid(j + 1))
                if (b > a) then
                    dx = grid(j + 1) - grid(j)
                    t = (a - lc(k))/(lc(k + 1) - lc(k)); ya = r(k) + t*(r(k + 1) - r(k))
                    t = (b - lc(k))/(lc(k + 1) - lc(k)); yb = r(k) + t*(r(k + 1) - r(k))
                    m = 0.5_wp*(a + b); ym = 0.5_wp*(ya + yb)
                    ! product of two linear functions: Simpson's rule is exact
                    ha = (grid(j + 1) - a)/dx; hb = (grid(j + 1) - b)/dx; hm = (grid(j + 1) - m)/dx
                    acc(j) = acc(j) + (b - a)/6.0_wp*(ya*ha + 4.0_wp*ym*hm + yb*hb)
                    ha = (a - grid(j))/dx; hb = (b - grid(j))/dx; hm = (m - grid(j))/dx
                    acc(j + 1) = acc(j + 1) + (b - a)/6.0_wp*(ya*ha + 4.0_wp*ym*hm + yb*hb)
                end if
                j = j + 1
            end do
        end do
        band(1) = acc(1)/(0.5_wp*(grid(2) - grid(1)))
        do j = 2, ng - 1
            band(j) = acc(j)/(0.5_wp*(grid(j + 1) - grid(j - 1)))
        end do
        band(ng) = acc(ng)/(0.5_wp*(grid(ng) - grid(ng - 1)))
        where (band < 0.0_wp) band = 0.0_wp
    end subroutine resample_hat

    !> Legacy point interpolation (FSPS): inside [curve start, curve end] on the grid, zero outside.
    subroutine resample_point(lc, r, nc, grid, band)
        real(WP), intent(in) :: lc(:), r(:), grid(:)
        integer, intent(in) :: nc
        real(WP), intent(out) :: band(:)
        integer :: si, ei, i, ng

        ng = size(grid)
        band = 0.0_wp
        if (nc <= 1) return
        ! The legacy bounds (opensps_io find_bounds, ascending grid)
        if (lc(1) <= grid(1)) then
            si = 1
        else
            si = max(find_interval(grid, lc(1)), 1)
        end if
        if (lc(nc) >= grid(ng)) then
            ei = ng
        else
            ei = min(find_interval(grid, lc(nc)), ng)
        end if
        if (ei > si) then
            do i = si, ei
                band(i) = interpolate_linear(lc(1:nc), r(1:nc), grid(i))
            end do
        end if
    end subroutine resample_point

    !> Builds the (normalized, uncalibrated) band of every filter of `set` on `grid`.
    !>
    !> @param[in]  strict  user curves (checked cleaning) or the legacy reader's lenient cleaning
    !> @param[out] bands   (size(grid), set%n), integral of T/lambda = 1; a dead filter is zero
    !> @param[out] info    names, hashes, coverage, points above half maximum, flags
    !> @param[out] status  FILTER_OK, a curve error, or a policy error; `bad` is the failing filter
    subroutine filters_build(set, grid, mode, policy, strict, bands, info, status, bad)
        type(filter_set_t), intent(in) :: set
        real(WP), intent(in) :: grid(:)
        integer, intent(in) :: mode, policy
        logical, intent(in) :: strict
        real(WP), intent(out) :: bands(:, :)
        type(filter_info_t), intent(inout) :: info
        integer, intent(out) :: status, bad

        real(WP), allocatable :: lc(:), rc(:)
        real(WP) :: norm
        integer :: i, first, nc, st, ng, k, p

        status = FILTER_OK
        bad = 0
        ng = size(grid)
        call filter_info_clear(info)
        allocate (info%name(set%n), info%hash(set%n), info%coverage(set%n), info%n_fwhm(set%n), &
                  info%flags(set%n), info%detector(set%n))
        info%resample = mode
        info%grid_policy = policy
        info%grid_hash = grid_hash(grid)
        info%name = set%name(1:set%n)
        info%detector = set%detector(1:set%n)
        info%coverage = 0.0_wp
        info%n_fwhm = 0
        info%flags = 0
        info%hash = ''
        bands = 0.0_wp

        first = 0
        do i = 1, set%n
            call filter_clean(set%lam(first + 1:first + set%npts(i)), set%resp(first + 1:first + set%npts(i)), &
                              strict, lc, rc, nc, st)
            first = first + set%npts(i)
            if (st /= FILTER_OK) then
                status = st
                bad = i
                return
            end if
            info%hash(i) = filter_hash(lc, rc, nc, set%detector(i), set%has_cal(i), set%cal_lref(i), set%cal_expo(i))
            if (allocated(set%expect_hash)) then
                if (len_trim(set%expect_hash(i)) > 0 .and. set%expect_hash(i) /= info%hash(i)) then
                    status = FILTER_ERR_HASH
                    bad = i
                    return
                end if
            end if
            if (nc < 2) then
                info%flags(i) = FILTER_FLAG_DEAD
                cycle
            end if

            ! Coverage: the part of the curve's weight (T/lambda for photon counting, T/lambda^2 for
            ! energy counting) that lies on the grid
            p = merge(2, 1, set%detector(i) == DETECTOR_ENERGY)
            norm = curve_integral(lc, rc, nc, lc(1), lc(nc), p)
            if (norm > 0.0_wp) then
                info%coverage(i) = curve_integral(lc, rc, nc, grid(1), grid(ng), p)/norm
            end if

            ! The response T on the grid (piecewise-linear curve)
            select case (mode)
            case (RESAMPLE_CONSERVING)
                call resample_hat(lc, rc, nc, grid, bands(:, i))
            case default
                call resample_point(lc, rc, nc, grid, bands(:, i))
            end select
            ! An energy counter's weight T/lambda^2 is the photon form with T' = T/lambda, taken at the
            ! grid points (the same sampling as the kernels' quadrature)
            if (set%detector(i) == DETECTOR_ENERGY) bands(:, i) = bands(:, i)/grid
            ! The first grid point can lie just below the curve's first wavelength, where the linear
            ! interpolation extrapolates the rising edge to a negative value
            where (bands(:, i) < 0.0_wp) bands(:, i) = 0.0_wp

            norm = integrate_trapezoid_array(grid, bands(:, i)/(grid + SAFE_FLOOR))
            if (norm > SAFE_FLOOR) then
                bands(:, i) = bands(:, i)/norm
                k = count(bands(:, i) >= 0.5_wp*maxval(bands(:, i)))
                info%n_fwhm(i) = k
                if (k < FILTER_MIN_FWHM_POINTS) info%flags(i) = ior(info%flags(i), FILTER_FLAG_UNDERSAMPLED)
            else
                bands(:, i) = 0.0_wp
                info%flags(i) = ior(info%flags(i), FILTER_FLAG_DEAD)
            end if
            if (info%coverage(i) < FILTER_MIN_COVERAGE) info%flags(i) = ior(info%flags(i), FILTER_FLAG_TRUNCATED)
        end do

        if (policy == GRID_POLICY_ERROR) then
            do i = 1, set%n
                if (iand(info%flags(i), FILTER_FLAG_DEAD) /= 0) then
                    status = FILTER_ERR_DEAD
                    bad = i
                    return
                else if (iand(info%flags(i), FILTER_FLAG_TRUNCATED) /= 0) then
                    status = FILTER_ERR_COVERAGE
                    bad = i
                    return
                end if
            end do
        end if
    end subroutine filters_build

    !> Calibrates a normalized band: division by integral (lambda/lam0)^expo T/lambda dlambda
    !> (expo = -1: flat nu f_nu, -2: beta = 2 blackbody). The exponents -1 and -2 are separate
    !> branches with literal powers, so that every layout gives bit-identical bands.
    subroutine filter_calibrate_band(grid, band, lam0, expo)
        real(WP), intent(in) :: grid(:), lam0, expo
        real(WP), intent(inout) :: band(:)
        real(WP) :: d

        if (expo == -1.0_wp) then
            d = integrate_trapezoid_array(grid, (grid/lam0)**(-1.0_wp)*band/(grid + SAFE_FLOOR))
        else if (expo == -2.0_wp) then
            d = integrate_trapezoid_array(grid, (grid/lam0)**(-2.0_wp)*band/(grid + SAFE_FLOOR))
        else
            d = integrate_trapezoid_array(grid, (grid/lam0)**expo*band/(grid + SAFE_FLOOR))
        end if
        if (d > SAFE_FLOOR) band = band/d
    end subroutine filter_calibrate_band

    !> Pivot wavelength sqrt(integral lambda T / integral T/lambda); 0 for a dead band
    function filter_pivot(grid, band) result(leff)
        real(WP), intent(in) :: grid(:), band(:)
        real(WP) :: leff, d
        leff = integrate_trapezoid_array(grid, grid*band)
        d = integrate_trapezoid_array(grid, band/grid)
        if (leff /= leff .or. d /= d .or. d <= SAFE_FLOOR) then
            leff = 0.0_wp
        else
            leff = sqrt(leff/d)
        end if
    end function filter_pivot

    !> AB magnitudes of two spectra (f_nu on the grid, e.g. Vega and the Sun) in every band:
    !> -2.5 log10(integral f_nu T/lambda) - 48.60, or 99 for no flux.
    subroutine filters_zero_points(grid, spec1, spec2, bands, mag1, mag2)
        real(WP), intent(in) :: grid(:), spec1(:), spec2(:), bands(:, :)
        real(WP), intent(out) :: mag1(:), mag2(:)
        integer :: i

        do i = 1, size(bands, 2)
            mag1(i) = integrate_trapezoid_array(grid, spec1*bands(:, i)/grid)
            if (mag1(i) > SAFE_FLOOR) then
                mag1(i) = -2.5_wp*log10(mag1(i)) - 48.60_wp
            else
                mag1(i) = 99.0_wp
            end if
            mag2(i) = integrate_trapezoid_array(grid, spec2*bands(:, i)/grid)
            if (mag2(i) > SAFE_FLOOR) then
                mag2(i) = -2.5_wp*log10(mag2(i)) - 48.60_wp
            else
                mag2(i) = 99.0_wp
            end if
        end do
    end subroutine filters_zero_points

end module opensps_filters
