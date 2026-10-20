module fsps_batch_photometry
    !> @brief Magnitudes of many spectra at once on the device (batched CSP, compute_csp_scenario).
    !>
    !> @details
    !> Same quantities as compute_magnitudes, for the columns of a spectrum array resident on the
    !> device: observed-frame interpolation, then the trapezoid filter integrals over each band's
    !> support, compensated at FP32 kernel precision (or, with -Dcublas=true, a split-K matrix
    !> product over 256-pixel blocks, the blocks summed in FP64). The grid-only data for a
    !> redshift (observed-frame map, distance term, band weights and supports, trapezoid weights,
    !> IGM transmission) are cached in ctx%state%batch and on the device (prepare_batch_grid).
    !> Light-weighted ages and redshifted colors are not handled (callers use compute_magnitudes).

    use fsps_precision, only: WP, KP
    use fsps_constants, only: SAFE_FLOOR, C_LIGHT, ABS_MAG_ZEROPOINT_LOG
    use fsps_context_types, only: fsps_context_t, batch_grid_release
    use fsps_cosmology, only: compute_igm_transmission
    use fsps_interpolation, only: find_interval, interpolate_linear
    use fsps_special_functions, only: mag_from_flux
    use fsps_smoothing, only: velocity_kernel_t, build_velocity_kernel
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
#ifdef FSPS_CUBLAS
    use cublas
    use openacc, only: acc_get_cuda_stream, acc_async_sync
#endif

    implicit none
    private

    public :: compute_magnitudes_columns, prepare_batch_grid
    public :: smoothing_on_device, prepare_smoothing_kernel, smooth_columns
#ifdef FSPS_CUBLAS
    public :: blas, blas_init
#endif

    real(WP), parameter :: AB_ZEROPOINT = 48.60_wp
    integer, parameter  :: IDX_V_BAND = 1
    !> Pixels per block of the split-K band product (cuBLAS path): each block is a plain FP32 sum
    !> of at most 256 positive terms; the blocks are added in FP64
    integer, parameter :: BAND_BLOCK = 256
#ifdef FSPS_CUBLAS
    type(cublasHandle), save :: blas
    logical, save :: blas_ready = .false.
#endif

contains

    !> @brief Magnitudes of spec(:, 1:n_col) (present on the device, L_sun/Hz) at redshift zred,
    !> as compute_magnitudes(ctx, zred, spec(:, i), mags(:, i), mag_compute) for each column; with
    !> zred_col, column i at redshift zred_col(i) (zred is then ignored).
    subroutine compute_magnitudes_columns(ctx, zred, igm_factor, spec, n_col, mags, mag_compute, zred_col)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), intent(in)                :: zred, igm_factor
        real(KP), intent(in)                :: spec(:,:)
        integer, intent(in)                 :: n_col
        real(WP), intent(out)               :: mags(:,:)
        integer, intent(in), optional       :: mag_compute(:)
        real(WP), intent(in), optional      :: zred_col(:)

        integer  :: nspec, nbands, b, m
        integer, allocatable  :: do_band(:)
        real(WP), allocatable :: fband(:,:), mags_l(:,:)
        ! Redshift and distance term per column: one array, one upload
        real(WP), allocatable, target :: zd(:)
        real(WP), pointer, contiguous :: zc(:), dist(:)
        logical  :: zvar
        real(KP), allocatable :: fobs(:,:)
        real(WP) :: flux
        logical  :: done
#ifdef FSPS_CUBLAS
        integer :: npad
        integer, parameter :: GEMM_MIN_COLS = 16
#endif

        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        zvar = present(zred_col)
        ! Per-column redshifts: the grid data of z = 0 (filters, trapezoid weights); the observed
        ! frame is then formed per column on the device, and the distance terms per column here
        if (zvar) then
            call prepare_batch_grid(ctx, 0.0_wp, igm_factor)
        else
            call prepare_batch_grid(ctx, zred, igm_factor)
        end if
        allocate(zd(2 * n_col))
        zc => zd(1:n_col)
        dist => zd(n_col + 1:2 * n_col)
        if (zvar) then
            zc = zred_col(1:n_col)
            do m = 1, n_col
                dist(m) = distance_term(ctx, zc(m))
            end do
        else
            zc = zred
            dist = ctx%state%batch%dist_term
        end if

        allocate(do_band(nbands), fband(nbands, n_col), mags_l(nbands, n_col))
        do_band = 1
        if (present(mag_compute)) then
            b = min(nbands, size(mag_compute))
            do_band(1:b) = mag_compute(1:b)
            if (ctx%compute_vega_mags_val == 1) do_band(IDX_V_BAND) = 1
        end if

        associate(g => ctx%state%batch)
        done = .false.
#ifdef FSPS_CUBLAS
        ! cuBLAS for many spectra; a few (single CSP calls) take the compensated kernels below,
        ! which are more accurate (no plain FP32 block sums) and as fast there
        if (n_col >= GEMM_MIN_COLS) then
            npad = BAND_BLOCK * ((nspec + BAND_BLOCK - 1) / BAND_BLOCK)
            allocate(fobs(npad, n_col))
            !$acc data present(spec) copyin(do_band, zd) create(fobs, fband) copyout(mags_l)
            if (zvar) then
                call observed_frame_z(ctx, n_col, nspec, npad, .false., zc, g%wl, spec, fobs)
            else
                call observed_frame(n_col, nspec, npad, g%i_lo, g%t_obs, g%wl, spec, fobs)
            end if
            call band_fluxes_gemm(n_col, npad, nbands, g%cmat, fobs, fband)
            call magnitudes_from_fluxes(ctx, n_col, nbands, do_band, dist, fband, mags_l)
            !$acc end data
            done = .true.
        end if
#endif
        if (.not. done) then
            allocate(fobs(n_col, nspec))
            !$acc data present(spec) copyin(do_band, zd) create(fobs, fband) copyout(mags_l)
#ifdef OMP_TARGET
            !$omp target data map(to: do_band, zd) map(alloc: fobs, fband) map(from: mags_l)
#endif
            if (zvar) then
                call observed_frame_z(ctx, n_col, nspec, nspec, .true., zc, g%wl, spec, fobs)
            else
                call observed_frame_t(n_col, nspec, g%i_lo, g%t_obs, g%wl, spec, fobs)
            end if
            call band_fluxes(n_col, nbands, g%bands_t, g%j_lo, g%j_hi, do_band, g%wtl, fobs, fband)
            call magnitudes_from_fluxes(ctx, n_col, nbands, do_band, dist, fband, mags_l)
#ifdef OMP_TARGET
            !$omp end target data
#endif
            !$acc end data
        end if
        end associate

        ! The cache's device copies are released with the context's (remove_from_device); a context
        ! not moved there by fsps_context_move_to_device would leave them behind, so drop them now
        if (.not. ctx%on_device) call batch_grid_release(ctx%state%batch)

        mags(1:nbands, 1:n_col) = mags_l
    end subroutine compute_magnitudes_columns

    !> Magnitudes from band fluxes (fband, present), as compute_magnitudes: AB, the distance term,
    !> NaN for bands not requested, and Vega magnitudes (relative to the V band) if requested.
    !> On the device (one thread per band and spectrum; the host loop was ~2 us per spectrum).
    subroutine magnitudes_from_fluxes(ctx, n_col, nbands, do_band, dist, fband, mags_l)
        type(fsps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_col, nbands, do_band(:)
        real(WP), intent(in)             :: dist(:)   ! distance term per column (present)
        real(WP), intent(in)             :: fband(:,:)
        real(WP), intent(out)            :: mags_l(:,:)

        integer  :: m, b
        real(WP) :: qnan, v, vv
        logical  :: vega_on

        qnan = ieee_value(1.0_wp, ieee_quiet_nan)
        vega_on = ctx%compute_vega_mags_val == 1

        ! (Vega magnitudes from the context's resident copy: no upload per call)
        !$acc parallel loop gang vector collapse(2) present(ctx, fband, do_band, dist, mags_l) private(v, vv)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2) private(v, vv)
#endif
        do m = 1, n_col
            do b = 1, nbands
                v = qnan
                if (do_band(b) /= 0) v = mag_from_flux(fband(b, m)) - AB_ZEROPOINT - 2.5_wp * ABS_MAG_ZEROPOINT_LOG + dist(m)
                if (vega_on .and. b >= 2) then
                    vv = qnan
                    if (do_band(IDX_V_BAND) /= 0) &
                        vv = mag_from_flux(fband(IDX_V_BAND, m)) - AB_ZEROPOINT - 2.5_wp * ABS_MAG_ZEROPOINT_LOG + dist(m)
                    if (vv == vv) then
                        v = v - ctx%state%magvega(b) + ctx%state%magvega(IDX_V_BAND)
                    else
                        v = qnan
                    end if
                end if
                mags_l(b, m) = v
            end do
        end do
    end subroutine magnitudes_from_fluxes

    !> Grid-only data for redshift zred (observed-frame map, distance term, band weights and
    !> supports, trapezoid weights, IGM transmission), cached in ctx%state%batch and on the device
    !> until zred or igm_factor change; released with the context's device data.
    subroutine prepare_batch_grid(ctx, zred, igm_factor)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), intent(in)                :: zred, igm_factor

        integer  :: nspec, nbands, j, b, ilo, ihi, npad
        real(WP) :: z_factor, target, dm, sw
        logical  :: igm

        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        igm = ctx%add_igm_absorption_val == 1 .and. zred > SAFE_FLOOR
        associate(g => ctx%state%batch)
        if (g%ready) then
            if (g%zred == zred .and. (g%igm .eqv. igm) .and. g%igm_factor == igm_factor .and. &
                size(g%wl) == nspec .and. size(g%j_lo) == nbands) return
        end if
        call batch_grid_release(g)
        allocate(g%i_lo(nspec), g%t_obs(nspec), g%wt(nspec), g%wtl(nspec), g%wl(nspec), g%igm_t(nspec), &
                 g%bands_t(nbands, nspec), g%j_lo(nbands), g%j_hi(nbands))
        g%zred = zred
        g%igm = igm
        g%igm_factor = igm_factor

        ! Trapezoid weights in frequency (dust emission) and wavelength (filters), formed in FP64
        do j = 1, nspec - 1
            g%wt(j) = real(0.5_wp * abs(C_LIGHT / ctx%state%spec_lambda(j+1) - C_LIGHT / ctx%state%spec_lambda(j)), KP)
            g%wtl(j) = real(0.5_wp * abs(ctx%state%spec_lambda(j + 1) - ctx%state%spec_lambda(j)), KP)
        end do
        g%wt(nspec) = 0.0_KP
        g%wtl(nspec) = 0.0_KP

        ! Observed-frame interpolation (prepare_observed_spectrum): per pixel, the rest-frame
        ! interval and position, and the distance term
        if (abs(zred) > SAFE_FLOOR) then
            z_factor = 1.0_wp + zred
            do j = 1, nspec
                target = ctx%state%spec_lambda(j) / z_factor
                g%i_lo(j) = max(1, min(find_interval(ctx%state%spec_lambda, target), nspec - 1))
                g%t_obs(j) = (target - ctx%state%spec_lambda(g%i_lo(j))) / &
                             (ctx%state%spec_lambda(g%i_lo(j)+1) - ctx%state%spec_lambda(g%i_lo(j)))
            end do
            g%dist_term = distance_term(ctx, zred)
        else
            g%i_lo = [(j, j = 1, nspec)]
            g%t_obs = 0.0_wp
            g%dist_term = 0.0_wp
        end if
        do j = 1, nspec
            if (ctx%state%spec_lambda(j) > tiny(0.0_wp)) then
                g%wl(j) = real(1.0_wp / ctx%state%spec_lambda(j), KP)
            else
                g%wl(j) = 0.0_KP
            end if
        end do

        ! Bands: transposed at kernel precision, and the support of each
        do b = 1, nbands
            g%bands_t(b, :) = real(ctx%state%bands(1:nspec, b), KP)
            ilo = nspec; ihi = 1
            do j = 1, nspec
                if (ctx%state%bands(j, b) /= 0.0_wp) then
                    ilo = min(ilo, j); ihi = max(ihi, j)
                end if
            end do
            ! Trapezoid intervals touching the support: j = ilo-1 .. ihi
            g%j_lo(b) = max(1, ilo - 1)
            g%j_hi(b) = min(nspec - 1, ihi)
            if (ilo > ihi) then
                g%j_lo(b) = 1; g%j_hi(b) = 0
            end if
        end do
        g%igm_t = 1.0_KP
        !$acc enter data copyin(g%i_lo, g%j_lo, g%j_hi, g%t_obs, g%wt, g%wtl, g%wl, g%bands_t, g%igm_t)
#ifdef OMP_TARGET
        !$omp target enter data map(to: g%i_lo, g%j_lo, g%j_hi, g%t_obs, g%wt, g%wtl, g%wl, g%bands_t, g%igm_t)
#endif
        if (igm) call compute_igm_transmission(ctx%state%spec_lambda, zred, igm_factor, g%igm_t)
        ! (with OpenMP target offload compute_igm_transmission runs on the host)
#ifdef OMP_TARGET
        !$omp target update to(g%igm_t)
#endif

#ifdef FSPS_CUBLAS
        ! Filter matrix of the split-K band product, padded to whole pixel blocks
        npad = BAND_BLOCK * ((nspec + BAND_BLOCK - 1) / BAND_BLOCK)
        allocate(g%cmat(npad, nbands))
        do b = 1, nbands
            do j = 1, npad
                if (j > nspec) then
                    g%cmat(j, b) = 0.0_KP
                else
                    sw = real(g%wtl(j), WP)
                    if (j > 1) sw = sw + real(g%wtl(j - 1), WP)
                    g%cmat(j, b) = real(sw * ctx%state%bands(j, b), KP)
                end if
            end do
        end do
        !$acc enter data copyin(g%cmat)
#ifdef OMP_TARGET
        !$omp target enter data map(to: g%cmat)
#endif
#endif
        g%ready = .true.
        end associate
    end subroutine prepare_batch_grid

    !> True when apply_smoothing for these settings is the velocity operator that
    !> smooth_columns applies (smooth_velocity = 1, smoothspec_fast = 1, no instrumental
    !> resolution).
    logical function smoothing_on_device(ctx)
        type(fsps_context_t), intent(in) :: ctx
        smoothing_on_device = ctx%smooth_velocity_val == 1 .and. ctx%smoothspec_fast_val == 1
    end function smoothing_on_device

    !> Velocity smoothing operator for (sigma, minl, maxl) (build_velocity_kernel: the
    !> convolve_variable_resolution of apply_smoothing as a sparse matrix), cached with the
    !> redshift grid data in ctx%state%batch and on the device. Call after prepare_batch_grid,
    !> which releases it when the redshift changes.
    subroutine prepare_smoothing_kernel(ctx, sigma, minl, maxl)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), intent(in)                :: sigma, minl, maxl
        type(velocity_kernel_t) :: kern

        associate(g => ctx%state%batch)
        if (g%sm_ready) then
            if (g%sm_sigma == sigma .and. g%sm_minl == minl .and. g%sm_maxl == maxl) return
            !$acc exit data delete(g%sm_lo, g%sm_hi, g%sm_off, g%sm_coef)
#ifdef OMP_TARGET
            !$omp target exit data map(delete: g%sm_lo, g%sm_hi, g%sm_off, g%sm_coef)
#endif
            deallocate(g%sm_lo, g%sm_hi, g%sm_off, g%sm_coef)
            g%sm_ready = .false.
        end if
        call build_velocity_kernel(ctx%state%spec_lambda(1:ctx%state%nspec), sigma, minl, maxl, kern)
        call move_alloc(kern%lo, g%sm_lo)
        call move_alloc(kern%hi, g%sm_hi)
        call move_alloc(kern%off, g%sm_off)
        if (size(kern%coef) == 0) then
            allocate(g%sm_coef(1)); g%sm_coef = 0.0_wp
        else
            call move_alloc(kern%coef, g%sm_coef)
        end if
        !$acc enter data copyin(g%sm_lo, g%sm_hi, g%sm_off, g%sm_coef)
#ifdef OMP_TARGET
        !$omp target enter data map(to: g%sm_lo, g%sm_hi, g%sm_off, g%sm_coef)
#endif
        g%sm_sigma = sigma
        g%sm_minl = minl
        g%sm_maxl = maxl
        g%sm_ready = .true.
        end associate
    end subroutine prepare_smoothing_kernel

    !> Smooths spec(:, 1:n_col) (present on the device) with the prepared operator: per pixel the
    !> normalized trapezoid weights of its window, summed in FP64 (apply_velocity_kernel).
    subroutine smooth_columns(ctx, spec, n_col)
        type(fsps_context_t), intent(inout) :: ctx
        real(KP), intent(inout)             :: spec(:,:)
        integer, intent(in)                 :: n_col

        ! The operator's arrays passed as arrays: device code does not refer to the cache's
        ! derived type, which is mapped only through its components
        call smooth_kernel(ctx%state%nspec, n_col, ctx%state%batch%sm_lo, ctx%state%batch%sm_hi, &
                           ctx%state%batch%sm_off, ctx%state%batch%sm_coef, spec)
    end subroutine smooth_columns

    subroutine smooth_kernel(nspec, n_col, sm_lo, sm_hi, sm_off, sm_coef, spec)
        integer, intent(in)                 :: nspec, n_col
        integer, intent(in), contiguous     :: sm_lo(:), sm_hi(:), sm_off(:)
        real(WP), intent(in), contiguous    :: sm_coef(:)
        real(KP), intent(inout)             :: spec(:,:)

        real(KP), allocatable :: sm(:,:)
        integer  :: m, i, k, lo, off
        real(WP) :: s

        allocate(sm(nspec, n_col))
        !$acc data present(spec, sm_lo, sm_hi, sm_off, sm_coef) create(sm)
#ifdef OMP_TARGET
        !$omp target data map(alloc: sm)
#endif
        !$acc parallel loop gang vector collapse(2) private(s, lo, off, k)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2) private(s, lo, off, k)
#endif
        do m = 1, n_col
            do i = 1, nspec
                lo = sm_lo(i)
                if (lo == 0) then
                    sm(i, m) = spec(i, m)
                else
                    off = sm_off(i) - lo
                    s = 0.0_wp
                    !$acc loop seq
                    do k = lo, sm_hi(i)
                        s = s + sm_coef(off + k) * real(spec(k, m), WP)
                    end do
                    sm(i, m) = real(s, KP)
                end if
            end do
        end do
        !$acc parallel loop gang vector collapse(2)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2)
#endif
        do m = 1, n_col
            do i = 1, nspec
                spec(i, m) = sm(i, m)
            end do
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
        !$acc end data
    end subroutine smooth_kernel

    !> Distance modulus and (1 + z) term of the magnitudes at redshift zred (0 at z = 0:
    !> absolute magnitudes), as prepare_observed_spectrum.
    function distance_term(ctx, zred) result(dist)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in)             :: zred
        real(WP) :: dist, dm

        dist = 0.0_wp
        if (abs(zred) <= SAFE_FLOOR) return
        dm = interpolate_linear(ctx%state%cosmospl(:,1), ctx%state%cosmospl(:,3), zred)
        if (dm /= dm .or. dm <= SAFE_FLOOR) return
        dist = 5.0_wp * log10(dm / 10.0_wp) + mag_from_flux(1.0_wp + zred)
    end function distance_term

    !> Observed-frame spectrum / lambda with a redshift per column (zc, present): per (pixel,
    !> column) the rest-frame interval and position as prepare_batch_grid finds them for one
    !> redshift. transposed: fobs(column, pixel) (band_fluxes), else fobs(pixel, column) with zero
    !> padding rows to npad (the cuBLAS product).
    subroutine observed_frame_z(ctx, n_col, nspec, npad, transposed, zc, wl, total, fobs)
        type(fsps_context_t), intent(in) :: ctx
        integer, intent(in)     :: n_col, nspec, npad
        logical, intent(in)     :: transposed
        real(WP), intent(in)    :: zc(:)
        real(KP), intent(in)    :: wl(:), total(:,:)
        real(KP), intent(out)   :: fobs(:,:)

        integer  :: m, j, il
        real(WP) :: v, target, t
        real(KP) :: f

        !$acc parallel loop gang vector collapse(2) present(ctx, zc, wl, total, fobs) private(il, v, target, t, f)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2) private(il, v, target, t, f)
#endif
        do m = 1, n_col
            do j = 1, npad
                f = 0.0_KP
                if (j <= nspec) then
                    if (abs(zc(m)) > SAFE_FLOOR) then
                        target = ctx%state%spec_lambda(j) / (1.0_wp + zc(m))
                        il = max(1, min(find_interval(ctx%state%spec_lambda, target), nspec - 1))
                        t = (target - ctx%state%spec_lambda(il)) / (ctx%state%spec_lambda(il + 1) - ctx%state%spec_lambda(il))
                        v = real(total(il, m), WP) + t * (real(total(il + 1, m), WP) - real(total(il, m), WP))
                    else
                        v = real(total(j, m), WP)
                    end if
                    if (v /= v) v = 0.0_wp
                    f = real(max(v, 0.0_wp), KP) * wl(j)
                end if
                if (transposed) then
                    if (j <= nspec) fobs(m, j) = f
                else
                    fobs(j, m) = f
                end if
            end do
        end do
    end subroutine observed_frame_z

    !> Observed-frame spectrum / lambda (prepare_observed_spectrum), zero in the padding rows.
    subroutine observed_frame(n_mod, nspec, npad, i_lo, t_obs, wl, total, fobs)
        integer, intent(in)     :: n_mod, nspec, npad, i_lo(:)
        real(WP), intent(in)    :: t_obs(:)
        real(KP), intent(in)    :: wl(:), total(:,:)
        real(KP), intent(out)   :: fobs(:,:)

        integer  :: m, j, il
        real(WP) :: v

        !$acc parallel loop gang vector collapse(2) present(i_lo, t_obs, wl, total, fobs) private(il, v)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2) private(il, v)
#endif
        do m = 1, n_mod
            do j = 1, npad
                if (j <= nspec) then
                    il = i_lo(j)
                    if (t_obs(j) == 0.0_wp) then   ! rest frame (i_lo = j: no right neighbour at j = nspec)
                        v = real(total(il, m), WP)
                    else
                        v = real(total(il, m), WP) + t_obs(j) * (real(total(il + 1, m), WP) - real(total(il, m), WP))
                    end if
                    if (v /= v) v = 0.0_wp
                    fobs(j, m) = real(max(v, 0.0_wp), KP) * wl(j)
                else
                    fobs(j, m) = 0.0_KP
                end if
            end do
        end do
    end subroutine observed_frame

    !> observed_frame, written transposed (models, pixels) for band_fluxes.
    subroutine observed_frame_t(n_mod, nspec, i_lo, t_obs, wl, total, fobs_t)
        integer, intent(in)     :: n_mod, nspec, i_lo(:)
        real(WP), intent(in)    :: t_obs(:)
        real(KP), intent(in)    :: wl(:), total(:,:)
        real(KP), intent(out)   :: fobs_t(:,:)

        integer  :: m, j, il
        real(WP) :: v

        !$acc parallel loop gang vector collapse(2) present(i_lo, t_obs, wl, total, fobs_t) private(il, v)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2) private(il, v)
#endif
        do j = 1, nspec
            do m = 1, n_mod
                il = i_lo(j)
                if (t_obs(j) == 0.0_wp) then
                    v = real(total(il, m), WP)
                else
                    v = real(total(il, m), WP) + t_obs(j) * (real(total(il + 1, m), WP) - real(total(il, m), WP))
                end if
                if (v /= v) v = 0.0_wp
                fobs_t(m, j) = real(max(v, 0.0_wp), KP) * wl(j)
            end do
        end do
    end subroutine observed_frame_t

#ifdef FSPS_CUBLAS
    !> Band fluxes as a split-K matrix product: per block of BAND_BLOCK pixels,
    !> partial = C^T . fobs with C(j, b) = (dl(j-1) + dl(j)) / 2 * band(j, b) (the trapezoid rule
    !> regrouped by pixel, formed in FP64; prepare_batch_grid); the blocks are summed in FP64.
    subroutine band_fluxes_gemm(n_mod, npad, nbands, cmat, fobs, fband)
        integer, intent(in)     :: n_mod, npad, nbands
        real(KP), intent(in)    :: cmat(:,:), fobs(:,:)
        real(WP), intent(out)   :: fband(:,:)

        real(KP), allocatable :: partial(:,:,:)
        integer  :: b, m, q, nkb, istat
        real(WP) :: sw

        nkb = npad / BAND_BLOCK
        allocate(partial(nbands, n_mod, nkb))
        call blas_init()
        !$acc data create(partial) present(cmat, fobs, fband)
        !$acc host_data use_device(cmat, fobs, partial)
        do q = 1, nkb
#ifdef KERNEL_SINGLE_PRECISION
            istat = cublasSgemm_v2(blas, CUBLAS_OP_T, CUBLAS_OP_N, nbands, n_mod, BAND_BLOCK, 1.0_KP, &
                                   cmat((q - 1) * BAND_BLOCK + 1, 1), npad, fobs((q - 1) * BAND_BLOCK + 1, 1), npad, &
                                   0.0_KP, partial(1, 1, q), nbands)
#else
            istat = cublasDgemm_v2(blas, CUBLAS_OP_T, CUBLAS_OP_N, nbands, n_mod, BAND_BLOCK, 1.0_KP, &
                                   cmat((q - 1) * BAND_BLOCK + 1, 1), npad, fobs((q - 1) * BAND_BLOCK + 1, 1), npad, &
                                   0.0_KP, partial(1, 1, q), nbands)
#endif
            if (istat /= 0) exit
        end do
        !$acc end host_data
        if (istat /= 0) error stop 'compute_csp_batch: cuBLAS GEMM failed'
        !$acc parallel loop gang vector collapse(2) present(partial, fband) private(sw, q)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(2) private(sw, q)
#endif
        do m = 1, n_mod
            do b = 1, nbands
                sw = 0.0_wp
                !$acc loop seq
                do q = 1, nkb
                    sw = sw + real(partial(b, m, q), WP)
                end do
                fband(b, m) = sw
            end do
        end do
        !$acc end data
    end subroutine band_fluxes_gemm

    !> The cuBLAS handle, on the OpenACC default stream. Experiment switch (NVIDIA only):
    !> FSPS_CUBLAS_MATH = tf32 lets the FP32 products use TF32 tensor cores (A100 and newer;
    !> 10-bit mantissa inputs), pedantic forbids tensor cores and reduced-precision paths (the
    !> FP64 products then run without the A100's FP64 tensor cores); unset: cuBLAS default math
    !> (FP32 on FP32 units, FP64 on FP64 tensor cores where available).
    subroutine blas_init()
        integer :: istat
        character(len=16) :: mode
        if (blas_ready) return
        istat = cublasCreate(blas)
        istat = cublasSetStream(blas, acc_get_cuda_stream(acc_async_sync))
        call get_environment_variable('FSPS_CUBLAS_MATH', mode)
        select case (trim(mode))
        case ('tf32')
            istat = cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH)
        case ('pedantic')
            istat = cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH)
        end select
        blas_ready = .true.
    end subroutine blas_init
#endif

    !> Band fluxes: the trapezoid filter integral of fobs over each band's support, compensated at
    !> FP32 kernel precision.
    subroutine band_fluxes(n_mod, nbands, bands_t, j_lo, j_hi, do_band, wt, fobs_t, fband)
        integer, intent(in)     :: n_mod, nbands, j_lo(:), j_hi(:), do_band(:)
        !> Observed spectrum / lambda, transposed (models, pixels): the threads of a gang read
        !> consecutive models
        real(KP), intent(in)    :: bands_t(:,:), wt(:), fobs_t(:,:)
        real(WP), intent(out)   :: fband(:,:)

        !> Models per gang: the threads of a gang share one band, hence one trip count (no
        !> divergence)
        integer, parameter :: MBLK = 128, SMALL_N = 16
        integer  :: m, j, b, mb, ml
        real(KP) :: s, c, term, tsum

#if defined(_OPENACC) || defined(OMP_TARGET)
        if (n_mod < SMALL_N) then
            ! Few spectra: a gang per (band, spectrum), the pixels reduced across its threads
            ! (a tree reduction, pairwise like the dust integrals in fsps_dust). Measured on the
            ! P100: 1 spectrum 0.65 -> 0.03 ms; slower than the kernel below from ~100 spectra
            !$acc parallel loop gang collapse(2) present(bands_t, j_lo, j_hi, do_band, wt, fobs_t, fband) private(s)
#ifdef OMP_TARGET
            !$omp target teams distribute collapse(2) private(s)
#endif
            do b = 1, nbands
                do m = 1, n_mod
                    s = 0.0_KP
                    if (do_band(b) /= 0) then
                        !$acc loop vector reduction(+:s)
#ifdef OMP_TARGET
                        !$omp parallel do reduction(+:s)
#endif
                        do j = j_lo(b), j_hi(b)
                            s = s + wt(j) * (fobs_t(m, j) * bands_t(b, j) + fobs_t(m, j + 1) * bands_t(b, j + 1))
                        end do
                    end if
                    fband(b, m) = real(s, WP)
                end do
            end do
            return
        end if
#endif
        ! OpenMP: one combined construct over the three loops (perfectly nested), consecutive threads
        ! on consecutive models of one band as with OpenACC; teams distribute over (band, block) with
        ! an inner parallel do ran 8x slower with nvfortran 25.1 on the A100 (56 vs 6.7 ms at N = 1e4)
        !$acc parallel loop gang collapse(2) vector_length(MBLK) present(bands_t, j_lo, j_hi, do_band, wt, fobs_t, fband)
#ifdef OMP_TARGET
        !$omp target teams distribute parallel do collapse(3) private(m, s, c, term, tsum, j)
#endif
        do b = 1, nbands
            do mb = 1, (n_mod + MBLK - 1) / MBLK
                !$acc loop vector private(m, s, c, term, tsum, j)
                do ml = 1, MBLK
                    m = (mb - 1) * MBLK + ml
                    if (m <= n_mod) then
                        s = 0.0_KP
                        c = 0.0_KP
                        if (do_band(b) /= 0) then
                            !$acc loop seq
                            do j = j_lo(b), j_hi(b)
                                if (KP == WP) then
                                    s = s + wt(j) * (fobs_t(m, j) * bands_t(b, j) + fobs_t(m, j + 1) * bands_t(b, j + 1))
                                else
                                    term = wt(j) * (fobs_t(m, j) * bands_t(b, j) + fobs_t(m, j + 1) * bands_t(b, j + 1)) - c
                                    tsum = s + term
                                    c = (tsum - s) - term
                                    s = tsum
                                end if
                            end do
                        end if
                        fband(b, m) = real(s, WP)
                    end if
                end do
            end do
        end do
    end subroutine band_fluxes

end module fsps_batch_photometry
