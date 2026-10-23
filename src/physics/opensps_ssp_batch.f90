module opensps_ssp_batch
    !> @brief The SSP stage of compute_csp_batch with the IMF free per model: the age sums of
    !> each model from the template operator (opensps_ssp_operator) instead of a resident SSP
    !> basis.
    !>
    !> @details
    !> With a basis, the young and old age sums of model m are basis . w_sfh(m) (basis = SSP
    !> spectra of all ages for the shared IMF: templates . B, B = S . w_IMF per age). With the IMF
    !> free per model the SFH is folded in first, so no SSP grid is built per model:
    !>
    !>   x(p, m) = w_IMF(p, m) w_sfh(age(p), m)               (each isochrone point p, FP64)
    !>   V(e, m) = sum_p S(e, p) x(p, m)                      (each effective template e, FP64)
    !>   young / old age sums = etpl . V_young / V_old         (kernel precision)
    !>
    !> split at the model's dust_tesc as the basis path splits the ages. Per age and model, the
    !> surviving mass (with the remnants), the linear L_bol and, for the nebular ages, the
    !> ionizing photon rate come from the same IMF weights through the operator's per-point
    !> coefficients; the nebular ages' EUV pixels are summed again with the escaping fraction
    !> (as add_nebular_batch does with a basis). Everything after the age sums is
    !> compute_csp_batch's.
    !>
    !> The models are processed in chunks (the IMF weights of a chunk, n_pts x chunk FP64, are
    !> the largest array). Every kernel runs on the device for an operator there (OpenMP target)
    !> and as OpenMP host loops otherwise, with the same arithmetic in the same order.

    use opensps_precision, only: WP, KP
    use opensps_constants, only: SAFE_FLOOR
    use opensps_imf, only: imf_integral_p, imf_bin_weights, IMF_P_LOWER_LIMIT, IMF_P_UPPER_LIMIT
    use opensps_stellar_modifications, only: add_remnant_mass_p
    implicit none
    private

    public :: ssp_batch_age_sums

contains

    !> Young and old age sums yo (pixels x 2N: young 1:N, old N+1:2N), and per age and model the
    !> surviving mass smass and linear L_bol slum (ages x N) and the ionizing photons q0 (nebular
    !> ages x N), for the IMF parameters pm(:, m) of each model and its SFH weights wsfh (N x
    !> ages, FP64, zero where the batch zeroes them).
    !>
    !> The operator arrays are those of ssp_operator_t (batch form); dev: they and the work arrays
    !> are on the device (OpenMP target). it_tesc: last young age per model; nt_neb: number of
    !> nebular ages (0: none), fesc: their escaping EUV fraction per model, whl: EUV pixels.
    subroutine ssp_batch_age_sums(dev, imf_type, remnants, n_mod, nt, nspec, n_pts, n_eff, pm, wsfh, it_tesc, &
                                  pt_ptr, pt_mini, pt_m1, pt_m2, pt_first, pt_cm, pt_cl, pt_q, age_mmax, &
                                  pt_age, eff_ptr, eff_pt, eff_val, k_age, n_seg, seg_ptr, seg_first, n_pc, pc_ptr, age_pc, etpl, &
                                  nt_neb, whl, fesc, &
                                  yo, smass, slum, q0)
        logical, intent(in)  :: dev, remnants
        integer, intent(in)  :: imf_type, n_mod, nt, nspec, n_pts, n_eff, nt_neb, whl, n_seg, n_pc
        real(WP), intent(in), contiguous :: pm(:,:), wsfh(:,:), fesc(:)
        integer, intent(in), contiguous  :: it_tesc(:), pt_ptr(:), pt_first(:), k_age(:), seg_ptr(:), seg_first(:)
        integer, intent(in), contiguous  :: pc_ptr(:), age_pc(:)
        integer, intent(in), contiguous  :: pt_age(:), eff_ptr(:), eff_pt(:)
        real(WP), intent(in), contiguous :: pt_mini(:), pt_m1(:), pt_m2(:), pt_cm(:), pt_cl(:), pt_q(:), age_mmax(:)
        real(WP), intent(in), contiguous :: eff_val(:)
        real(KP), intent(in), contiguous :: etpl(:,:)
        real(KP), intent(inout), contiguous :: yo(:,:)
        real(WP), intent(inout), contiguous :: smass(:,:), slum(:,:), q0(:,:)

        !> Models per chunk: bounded by the IMF weights (n_pts x chunk FP64, ~1.7 GB at 2048 models
        !> of MIST)
        integer, parameter :: CHUNK = 2048
        real(WP), allocatable :: w(:,:), tot(:), vp(:,:,:), pps(:,:,:)
        real(KP), allocatable :: wk(:,:), v(:,:), veuv(:,:)
        integer :: m0, nc, ky, neb_cols
        logical :: useg

        neb_cols = 0
        if (nt_neb > 0 .and. whl > 0) neb_cols = 1
        allocate(w(min(CHUNK, n_mod), n_pts), tot(min(CHUNK, n_mod)), v(n_eff, 2 * min(CHUNK, n_mod)))
        allocate(veuv(n_eff, max(1, neb_cols * 2 * min(CHUNK, n_mod))))
        ! The IMF weights at kernel precision for the fold (half the memory traffic with FP32 kernels;
        ! the mass, L_bol and photon sums take the FP64 weights)
        allocate(wk(min(CHUNK, n_mod), n_pts))
        ! Device: the fold by segments of the template lists (4 x n_seg x chunk FP64 partial sums,
        ! ~260 MB at 2048 models of MIST); faster than one thread per template at every batch size
        ! tried on the A100 (N = 1: 15 -> 10 ms, 100: 128 -> 85 us/model, 10^4: 28.0 -> 26.7 us/model)
        useg = dev
        if (useg) then
            allocate(vp(min(CHUNK, n_mod), 4, n_seg), pps(min(CHUNK, n_mod), 3, n_pc))
        else
            allocate(vp(1, 1, 1), pps(1, 1, 1))
        end if
#ifdef OMP_TARGET
        !$omp target data if(dev) map(alloc: w, wk, tot, v, veuv, vp, pps)
#endif
        do m0 = 1, n_mod, CHUNK
            nc = min(CHUNK, n_mod - m0 + 1)
            if (useg) then
                call imf_pieces_chunk(imf_type, remnants, m0, nc, nt, n_pc, nt_neb, pm, pt_ptr, pc_ptr, age_pc, pt_mini, &
                                      pt_m1, pt_m2, pt_first, pt_cm, pt_cl, pt_q, age_mmax, size(w, 1), n_pts, w, wk, &
                                      tot, pps, smass, slum, q0)
            else
                call imf_ages_chunk(dev, imf_type, remnants, m0, nc, nt, nt_neb, pm, pt_ptr, pt_mini, pt_m1, pt_m2, &
                                    pt_first, pt_cm, pt_cl, pt_q, age_mmax, size(w, 1), n_pts, w, wk, tot, smass, slum, q0)
            end if
            if (useg) then
                call fold_segments(m0, nc, n_eff, n_seg, nt_neb, neb_cols, wsfh, it_tesc, fesc, pt_age, eff_pt, eff_val, &
                                   seg_ptr, seg_first, wk, vp, v, veuv)
            else
                call fold_chunk(dev, m0, nc, n_eff, nt_neb, neb_cols, wsfh, it_tesc, fesc, pt_age, eff_ptr, eff_pt, &
                                eff_val, wk, v, veuv)
            end if
            ! Young sums: only the templates of the ages up to the chunk's last young age
            ky = k_age(max(1, min(nt, maxval(it_tesc(m0:m0 + nc - 1)))))
            call template_columns(dev, nspec, nspec, n_eff, ky, nc, etpl, v, 0, yo, m0 - 1)
            call template_columns(dev, nspec, nspec, n_eff, n_eff, nc, etpl, v, nc, yo, n_mod + m0 - 1)
            if (neb_cols > 0) then
                ! The EUV pixels again, the nebular ages' templates times the escaping fraction
                call template_columns(dev, whl, nspec, n_eff, ky, nc, etpl, veuv, 0, yo, m0 - 1)
                call template_columns(dev, whl, nspec, n_eff, n_eff, nc, etpl, veuv, nc, yo, n_mod + m0 - 1)
            end if
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine ssp_batch_age_sums

    !> IMF weights w(j, p) of models m = m0 .. m0+nc-1 (j = m - m0 + 1) at every isochrone point p
    !> (imf_bin_weights, as the SSP grid), and per age and model the surviving mass (with the
    !> remnants, as ssp_operator_evaluate), the linear L_bol and for the nebular ages the ionizing
    !> photon rate. One thread per (age, model): the age's points in order (each bin edge's power
    !> computed once), then their sums.
    subroutine imf_ages_chunk(dev, imf_type, remnants, m0, nc, nt, nt_neb, pm, pt_ptr, pt_mini, pt_m1, pt_m2, &
                              pt_first, pt_cm, pt_cl, pt_q, age_mmax, ldw, n_pts, w, wk, tot, smass, slum, q0)
        logical, intent(in) :: dev, remnants
        integer, intent(in) :: imf_type, m0, nc, nt, nt_neb
        real(WP), intent(in), contiguous :: pm(:,:), pt_mini(:), pt_m1(:), pt_m2(:), pt_cm(:), pt_cl(:), pt_q(:), &
                                            age_mmax(:)
        integer, intent(in), contiguous  :: pt_ptr(:), pt_first(:)
        integer, intent(in) :: ldw, n_pts
        real(WP), intent(inout) :: w(ldw, n_pts)
        real(KP), intent(inout) :: wk(ldw, n_pts)
        real(WP), intent(inout), contiguous :: tot(:), smass(:,:), slum(:,:), q0(:,:)
        integer  :: j, t, k, m, k0, n
        real(WP) :: sm, sl, sq

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do if(target: dev)
#elif defined(_OPENMP)
        !$omp parallel do
#endif
        do j = 1, nc
            tot(j) = imf_integral_p(imf_type, pm(:, m0 + j - 1), pm(IMF_P_LOWER_LIMIT, m0 + j - 1), &
                                    pm(IMF_P_UPPER_LIMIT, m0 + j - 1), .true.)
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, m, k0, n, sm, sl, sq) map(tofrom: q0) &
        !$omp if(target: dev)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, m, k0, n, sm, sl, sq)
#endif
        do t = 1, nt
            do j = 1, nc
                m = m0 + j - 1
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                k0 = pt_ptr(t)
                n = pt_ptr(t + 1) - k0
                if (n > 0) call imf_bin_weights(imf_type, pm(:, m), tot(j), n, pt_mini(k0:k0 + n - 1), &
                                                pt_m1(k0:k0 + n - 1), pt_m2(k0:k0 + n - 1), pt_first(k0:k0 + n - 1), &
                                                w(j, k0), ldw)
                do k = k0, k0 + n - 1
                    sm = sm + pt_cm(k) * w(j, k)
                    sl = sl + pt_cl(k) * w(j, k)
                    sq = sq + pt_q(k) * w(j, k)
                    wk(j, k) = real(w(j, k), KP)
                end do
                ! An empty isochrone has no mass (and no remnants), as ssp_operator_evaluate
                if (remnants .and. pt_ptr(t + 1) > pt_ptr(t)) call add_remnant_mass_p(imf_type, pm(:, m), sm, age_mmax(t))
                smass(t, m) = sm
                slum(t, m) = sl
                if (t <= nt_neb) q0(t, m) = sq
            end do
        end do
    end subroutine imf_ages_chunk

    !> imf_ages_chunk on the device with the points of each age cut into pieces (ssp_operator_t:
    !> pc_ptr, age_pc): one thread per (piece, model) makes the IMF weights of its points and their
    !> partial sums of mass, L_bol and photons, then one thread per (age, model) adds the pieces in
    !> order (and the remnants). The weights are the same as imf_ages_chunk's (a piece starts with
    !> both edges' powers computed, which are the values the previous bin would have handed over);
    !> the sums differ in FP64 rounding only.
    subroutine imf_pieces_chunk(imf_type, remnants, m0, nc, nt, n_pc, nt_neb, pm, pt_ptr, pc_ptr, age_pc, pt_mini, &
                                pt_m1, pt_m2, pt_first, pt_cm, pt_cl, pt_q, age_mmax, ldw, n_pts, w, wk, tot, pps, &
                                smass, slum, q0)
        logical, intent(in) :: remnants
        integer, intent(in) :: imf_type, m0, nc, nt, n_pc, nt_neb
        real(WP), intent(in), contiguous :: pm(:,:), pt_mini(:), pt_m1(:), pt_m2(:), pt_cm(:), pt_cl(:), pt_q(:), &
                                            age_mmax(:)
        integer, intent(in), contiguous  :: pt_ptr(:), pc_ptr(:), age_pc(:), pt_first(:)
        integer, intent(in) :: ldw, n_pts
        real(WP), intent(inout) :: w(ldw, n_pts)
        real(KP), intent(inout) :: wk(ldw, n_pts)
        real(WP), intent(inout), contiguous :: tot(:), pps(:,:,:), smass(:,:), slum(:,:), q0(:,:)
        integer  :: j, t, k, m, q, k0, n
        real(WP) :: sm, sl, sq

        !$omp target teams distribute parallel do
        do j = 1, nc
            tot(j) = imf_integral_p(imf_type, pm(:, m0 + j - 1), pm(IMF_P_LOWER_LIMIT, m0 + j - 1), &
                                    pm(IMF_P_UPPER_LIMIT, m0 + j - 1), .true.)
        end do
        !$omp target teams distribute parallel do collapse(2) private(k, m, k0, n, sm, sl, sq)
        do q = 1, n_pc
            do j = 1, nc
                m = m0 + j - 1
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                k0 = pc_ptr(q)
                n = pc_ptr(q + 1) - k0
                call imf_bin_weights(imf_type, pm(:, m), tot(j), n, pt_mini(k0:k0 + n - 1), &
                                     pt_m1(k0:k0 + n - 1), pt_m2(k0:k0 + n - 1), pt_first(k0:k0 + n - 1), &
                                     w(j, k0), ldw)
                do k = k0, k0 + n - 1
                    sm = sm + pt_cm(k) * w(j, k)
                    sl = sl + pt_cl(k) * w(j, k)
                    sq = sq + pt_q(k) * w(j, k)
                    wk(j, k) = real(w(j, k), KP)
                end do
                pps(j, 1, q) = sm
                pps(j, 2, q) = sl
                pps(j, 3, q) = sq
            end do
        end do
        !$omp target teams distribute parallel do collapse(2) private(q, sm, sl, sq)
        do t = 1, nt
            do j = 1, nc
                m = m0 + j - 1
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                do q = age_pc(t), age_pc(t + 1) - 1
                    sm = sm + pps(j, 1, q)
                    sl = sl + pps(j, 2, q)
                    sq = sq + pps(j, 3, q)
                end do
                ! An empty isochrone has no mass (and no remnants), as ssp_operator_evaluate
                if (remnants .and. pt_ptr(t + 1) > pt_ptr(t)) call add_remnant_mass_p(imf_type, pm(:, m), sm, age_mmax(t))
                smass(t, m) = sm
                slum(t, m) = sl
                if (t <= nt_neb) q0(t, m) = sq
            end do
        end do
    end subroutine imf_pieces_chunk

    !> Template weights of the chunk's models, the SFH folded in: v(e, j) young and v(e, nc + j)
    !> old sums over the points of template e of S(e, p) w(j, p) w_sfh(age(p), m); with nebular
    !> emission also veuv, the same with the nebular ages' terms times fesc(m). One thread per
    !> (template, model); the template's points are in age order: per age the sum of
    !> S(e, p) w(j, p) over its points, then times the model's SFH weight. FP64 sums, stored at
    !> kernel precision. (Memory-bound: w is read once per template entry.)
    subroutine fold_chunk(dev, m0, nc, n_eff, nt_neb, neb_cols, wsfh, it_tesc, fesc, pt_age, eff_ptr, eff_pt, &
                          eff_val, w, v, veuv)
        logical, intent(in) :: dev
        integer, intent(in) :: m0, nc, n_eff, nt_neb, neb_cols
        real(WP), intent(in), contiguous :: wsfh(:,:), fesc(:), eff_val(:)
        real(KP), intent(in), contiguous :: w(:,:)
        integer, intent(in), contiguous  :: it_tesc(:), pt_age(:), eff_ptr(:), eff_pt(:)
        real(KP), intent(inout), contiguous :: v(:,:), veuv(:,:)
        integer  :: e, j, k, t, m, tesc, tc
        real(WP) :: x, y, o, ye, oe, f, a, ae

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, t, m, tesc, tc, x, y, o, ye, oe, f, a, ae) &
        !$omp map(to: fesc) if(target: dev)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, t, m, tesc, tc, x, y, o, ye, oe, f, a, ae)
#endif
        do e = 1, n_eff
            do j = 1, nc
                m = m0 + j - 1
                tesc = it_tesc(m)
                f = fesc(m)
                y = 0.0_wp; o = 0.0_wp; ye = 0.0_wp; oe = 0.0_wp
                tc = 0
                a = 0.0_wp
                do k = eff_ptr(e), eff_ptr(e + 1)   ! (one past the end: flushes the last age)
                    if (k < eff_ptr(e + 1)) then
                        t = pt_age(eff_pt(k))
                    else
                        t = -1
                    end if
                    if (t /= tc) then
                        if (tc > 0) then
                            x = a * wsfh(m, tc)
                            ae = x
                            if (tc <= nt_neb) ae = f * x
                            if (tc <= tesc) then
                                y = y + x
                                ye = ye + ae
                            else
                                o = o + x
                                oe = oe + ae
                            end if
                        end if
                        tc = t
                        a = 0.0_wp
                    end if
                    if (t > 0) a = a + eff_val(k) * real(w(j, eff_pt(k)), WP)
                end do
                v(e, j) = real(y, KP)
                v(e, nc + j) = real(o, KP)
                if (neb_cols > 0) then
                    veuv(e, j) = real(ye, KP)
                    veuv(e, nc + j) = real(oe, KP)
                end if
            end do
        end do
    end subroutine fold_chunk

    !> fold_chunk with the lists of the templates cut into segments (ssp_operator_t: seg_ptr,
    !> seg_first), for small batches on the device: one thread per (segment, model) sums its ages
    !> (the segments end at age boundaries) into vp(model, 1:4, segment) = young, old, young and old
    !> with the escaping fraction; then one thread per (template, model) adds its segments in
    !> order. The same sums as fold_chunk with the ages grouped differently (FP64 rounding only).
    subroutine fold_segments(m0, nc, n_eff, n_seg, nt_neb, neb_cols, wsfh, it_tesc, fesc, pt_age, eff_pt, eff_val, &
                             seg_ptr, seg_first, w, vp, v, veuv)
        integer, intent(in) :: m0, nc, n_eff, n_seg, nt_neb, neb_cols
        real(WP), intent(in), contiguous :: wsfh(:,:), fesc(:), eff_val(:)
        real(KP), intent(in), contiguous :: w(:,:)
        integer, intent(in), contiguous  :: it_tesc(:), pt_age(:), eff_pt(:), seg_ptr(:), seg_first(:)
        real(WP), intent(inout), contiguous :: vp(:,:,:)
        real(KP), intent(inout), contiguous :: v(:,:), veuv(:,:)
        integer  :: s, e, j, k, t, m, tesc, tc
        real(WP) :: x, y, o, ye, oe, f, a, ae

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, t, m, tesc, tc, x, y, o, ye, oe, f, a, ae) &
        !$omp map(to: fesc)
#endif
        do s = 1, n_seg
            do j = 1, nc
                m = m0 + j - 1
                tesc = it_tesc(m)
                f = fesc(m)
                y = 0.0_wp; o = 0.0_wp; ye = 0.0_wp; oe = 0.0_wp
                tc = 0
                a = 0.0_wp
                do k = seg_ptr(s), seg_ptr(s + 1)   ! (one past the end: flushes the last age)
                    if (k < seg_ptr(s + 1)) then
                        t = pt_age(eff_pt(k))
                    else
                        t = -1
                    end if
                    if (t /= tc) then
                        if (tc > 0) then
                            x = a * wsfh(m, tc)
                            ae = x
                            if (tc <= nt_neb) ae = f * x
                            if (tc <= tesc) then
                                y = y + x
                                ye = ye + ae
                            else
                                o = o + x
                                oe = oe + ae
                            end if
                        end if
                        tc = t
                        a = 0.0_wp
                    end if
                    if (t > 0) a = a + eff_val(k) * real(w(j, eff_pt(k)), WP)
                end do
                vp(j, 1, s) = y
                vp(j, 2, s) = o
                vp(j, 3, s) = ye
                vp(j, 4, s) = oe
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(s, y, o, ye, oe)
#endif
        do e = 1, n_eff
            do j = 1, nc
                y = 0.0_wp; o = 0.0_wp; ye = 0.0_wp; oe = 0.0_wp
                do s = seg_first(e), seg_first(e + 1) - 1
                    y = y + vp(j, 1, s)
                    o = o + vp(j, 2, s)
                    ye = ye + vp(j, 3, s)
                    oe = oe + vp(j, 4, s)
                end do
                v(e, j) = real(y, KP)
                v(e, nc + j) = real(o, KP)
                if (neb_cols > 0) then
                    veuv(e, j) = real(ye, KP)
                    veuv(e, nc + j) = real(oe, KP)
                end if
            end do
        end do
    end subroutine fold_segments

    !> yo(1:np, c0 + j) = etpl(1:np, 1:k) . v(1:k, v0 + j), j = 1 .. nc: the age sums from the
    !> template weights, compensated (Kahan) with FP32 kernels.
    !> - cuBLAS builds (-Dcublas=true): GEMMs; with FP32 kernels over blocks of KB templates, each
    !>   block's product added to yo with compensation (a plain FP32 GEMM over ~2600 templates
    !>   loses ~1e-6).
    !> - Device: a register tile per thread, PT pixels (strided, so loads coalesce) x MT models,
    !>   the templates in order (the age_sums tile of compute_csp_batch; PT + MT loads per PT MT
    !>   multiply-adds instead of 2 per 1).
    !> - Host: per block of pixels, each template block read once for all the chunk's models.
    !> Each (pixel, column) sums the templates in order in the last two (the same results).
    subroutine template_columns(dev, np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
        logical, intent(in) :: dev
        integer, intent(in) :: np, nspec, n_eff, k, nc, v0, c0
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
        integer, parameter :: MT = 8, PT = 4, PIX_BLOCK = 256
        integer  :: i, j, e, mq, ip, npt, l, p, mm(MT), ii(PT), p0, p1, nb, q
        real(KP) :: bk(PT), wk(MT), acc(PT, MT), cmp(PT, MT), term, tsum
        real(KP), allocatable :: a(:,:), c(:,:)

        if (np < 1 .or. nc < 1) return
        if (k < 1) then
            if (dev) then
#ifdef OMP_TARGET
                !$omp target teams distribute parallel do collapse(2)
#endif
                do j = 1, nc
                    do i = 1, np
                        yo(i, c0 + j) = 0.0_KP
                    end do
                end do
            else
                yo(1:np, c0 + 1:c0 + nc) = 0.0_KP
            end if
            return
        end if

        if (dev) then
#ifdef FSPS_CUBLAS
            call gemm_columns(np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
            return
#endif
            npt = (np + PT - 1) / PT
#ifdef OMP_TARGET
            !$omp target teams loop collapse(2) private(mm, ii, bk, wk, acc, cmp, e, l, p, term, tsum)
#endif
            do mq = 1, (nc + MT - 1) / MT
                do ip = 1, npt
                    do l = 1, MT
                        ! Columns of this thread (clamped at the end; duplicates write the same value)
                        mm(l) = min((mq - 1) * MT + l, nc)
                    end do
                    do p = 1, PT
                        ii(p) = min(ip + (p - 1) * npt, np)
                    end do
                    acc = 0.0_KP
                    cmp = 0.0_KP
                    do e = 1, k
                        do l = 1, MT
                            wk(l) = v(e, v0 + mm(l))
                        end do
                        do p = 1, PT
                            bk(p) = etpl(ii(p), e)
                        end do
                        do l = 1, MT
                            if (wk(l) == 0.0_KP) cycle
                            do p = 1, PT
                                if (KP == WP) then
                                    acc(p, l) = acc(p, l) + wk(l) * bk(p)
                                else
                                    term = wk(l) * bk(p) - cmp(p, l)
                                    tsum = acc(p, l) + term
                                    cmp(p, l) = (tsum - acc(p, l)) - term
                                    acc(p, l) = tsum
                                end if
                            end do
                        end do
                    end do
                    do l = 1, MT
                        do p = 1, PT
                            yo(ii(p), c0 + mm(l)) = acc(p, l)
                        end do
                    end do
                end do
            end do
            return
        end if

        ! Host: blocks of pixels (threads), the templates in order, the columns, the pixels inner
        !$omp parallel do schedule(dynamic) private(p0, p1, nb, a, c, e, j, q, wk, term, tsum)
        do p0 = 1, np, PIX_BLOCK
            p1 = min(p0 + PIX_BLOCK - 1, np)
            nb = p1 - p0 + 1
            allocate(a(nb, nc), c(nb, nc))
            a = 0.0_KP
            c = 0.0_KP
            do e = 1, k
                do j = 1, nc
                    wk(1) = v(e, v0 + j)
                    if (wk(1) == 0.0_KP) cycle
                    if (KP == WP) then
                        do q = 1, nb
                            a(q, j) = a(q, j) + wk(1) * etpl(p0 + q - 1, e)
                        end do
                    else
                        do q = 1, nb
                            term = wk(1) * etpl(p0 + q - 1, e) - c(q, j)
                            tsum = a(q, j) + term
                            c(q, j) = (tsum - a(q, j)) - term
                            a(q, j) = tsum
                        end do
                    end if
                end do
            end do
            yo(p0:p1, c0 + 1:c0 + nc) = a
            deallocate(a, c)
        end do
    end subroutine template_columns

#ifdef FSPS_CUBLAS
    !> template_columns with cuBLAS (arrays on the device). FP64: one GEMM. FP32: a GEMM per
    !> block of KB templates into a work array, each added to yo with a compensation term.
    subroutine gemm_columns(np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
        use opensps_batch_photometry, only: blas, blas_init
        use cublas
        integer, intent(in) :: np, nspec, n_eff, k, nc, v0, c0
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
        integer, parameter :: KB = 256
        integer :: istat, e0, kb1, i, j
        real(KP), allocatable :: tmp(:,:), cmp(:,:)
        real(KP) :: term, tsum

        call blas_init()
#ifdef KERNEL_SINGLE_PRECISION
        allocate(tmp(np, nc), cmp(np, nc))
        !$omp target data map(alloc: tmp, cmp)
        !$omp target teams distribute parallel do collapse(2)
        do j = 1, nc
            do i = 1, np
                yo(i, c0 + j) = 0.0_KP
                cmp(i, j) = 0.0_KP
            end do
        end do
        do e0 = 1, k, KB
            kb1 = min(KB, k - e0 + 1)
            !$omp target data use_device_addr(etpl, v, tmp)
            istat = cublasSgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_N, np, nc, kb1, 1.0_KP, etpl(1, e0), nspec, &
                                   v(e0, v0 + 1), n_eff, 0.0_KP, tmp, np)
            !$omp end target data
            if (istat /= 0) error stop 'ssp batch: cuBLAS GEMM failed'
            !$omp target teams distribute parallel do collapse(2) private(term, tsum)
            do j = 1, nc
                do i = 1, np
                    term = tmp(i, j) - cmp(i, j)
                    tsum = yo(i, c0 + j) + term
                    cmp(i, j) = (tsum - yo(i, c0 + j)) - term
                    yo(i, c0 + j) = tsum
                end do
            end do
        end do
        !$omp end target data
#else
        !$omp target data use_device_addr(etpl, v, yo)
        istat = cublasDgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_N, np, nc, k, 1.0_KP, etpl, nspec, &
                               v(1, v0 + 1), n_eff, 0.0_KP, yo(1, c0 + 1), size(yo, 1))
        !$omp end target data
        if (istat /= 0) error stop 'ssp batch: cuBLAS GEMM failed'
#endif
    end subroutine gemm_columns
#endif

end module opensps_ssp_batch
