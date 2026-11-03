module opensps_sfh_tangent
    !> @brief Tangents of the SFH weights (forward mode), for the gradients of compute_csp_batch.
    !>
    !> @details
    !> sfh_weights_core gives each SSP age the mass formed against the hat kernel of that age:
    !> per age bin, w_j += (node_hi m0 - m1) / dt and w_j+1 += (m1 - node_lo m0) / dt, with
    !> m0, m1 differences of closed-form indefinite moments (eval_indefinite_moments_log / _lin) at
    !> limits clamped to the grid ages and to the star-forming window [tq, tage] (lookback time,
    !> years). Its derivative with respect to an SFH parameter is the same sums with the moments'
    !> total derivatives: their partial derivatives at fixed time (tau, tage enter the closed forms;
    !> d Ei(x) / dx = e^x / x) plus the moments' time derivative times the limit's, where a window
    !> edge lies inside a bin. Then the forward's normalization and mixture (constant, burst) are
    !> differentiated as written. The burst's two weights are linear in its age (log or linear, as
    !> the SSP interpolation).
    !>
    !> Supported: sfh = 0 (SSP), 1 (tau), 4 (delayed tau), with const and fburst, without
    !> truncation (sf_trunc outside (0, tage]); the Simha form (sfh = 5) and truncation are not
    !> (sfh_tangent_supported). Parameters, in the units of PARAMS (Gyr): tage, tau, const, fburst,
    !> tburst, sf_start (SFHT_*).
    !> Where tage (or the burst's age) sits exactly on a grid age the weights have a kink; the tangent
    !> there is the upward one (the derivative from above).

    use opensps_precision, only: WP
    use opensps_context_types, only: opensps_context_t
    use opensps_types, only: params_core, sfhparams
    use opensps_constants, only: SAFE_FLOOR
    use opensps_special_functions, only: expi
    use opensps_interpolation, only: find_interval
    use opensps_sfh, only: convert_sfhparams
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite

    implicit none
    private

    public :: sfh_weights_tangent, sfh_tangent_supported
    public :: SFHT_TAGE, SFHT_TAU, SFHT_CONST, SFHT_FBURST, SFHT_TBURST, SFHT_SF_START, NSFHT

    integer, parameter :: SFHT_TAGE = 1, SFHT_TAU = 2, SFHT_CONST = 3, SFHT_FBURST = 4, SFHT_TBURST = 5, &
                          SFHT_SF_START = 6, NSFHT = 6
    real(WP), parameter :: LOG_E = log10(exp(1.0_wp))
    real(WP), parameter :: LN10 = log(10.0_wp)
    real(WP), parameter :: GYR = 1.0e9_wp

contains

    !> True when sfh_weights_tangent covers this model's SFH.
    pure logical function sfh_tangent_supported(pl, tage)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(params_core), intent(in) :: pl
        real(WP), intent(in)          :: tage
        type(sfhparams) :: sfh

        sfh_tangent_supported = .false.
        if (pl%sfh /= 0 .and. pl%sfh /= 1 .and. pl%sfh /= 4 .and. pl%sfh /= 2 .and. pl%sfh /= 3) return
        if (pl%sfh == 1 .or. pl%sfh == 4) then
            call convert_sfhparams(pl, tage, sfh)
            if (sfh%tq /= 0.0_wp) return   ! truncation inside (0, tage]
        end if
        sfh_tangent_supported = .true.
    end function sfh_tangent_supported

    !> The SFH weights w (as sfh_weights_core, one metallicity) and their derivative dw with respect
    !> to parameter q (SFHT_*) of model pl at age tage (Gyr). Tabular SFHs (2, 3) have no
    !> parameters here: dw = 0. s: scratch (ages, 4).
    subroutine sfh_weights_tangent(ctx, pl, tage, q, w, dw, s)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(params_core), intent(in)       :: pl
        real(WP), intent(in)                :: tage
        integer, intent(in)                 :: q
        real(WP), intent(inout)             :: w(:), dw(:), s(:,:)

        type(sfhparams) :: sfh
        real(WP) :: dtg, dtau, dtb, mass1, dmass1, massc, dmassc, c, fb, wt, dwt, cn, dcn
        integer  :: nt, imax, i

        nt = ctx%state%ntfull
        do i = 1, nt
            w(i) = 0.0_wp
            dw(i) = 0.0_wp
        end do
        if (pl%sfh /= 0 .and. pl%sfh /= 1 .and. pl%sfh /= 4) return
        call convert_sfhparams(pl, tage, sfh)
        ! The internal parameters' tangents (years; lookback time): tage_int = (tage - sf_start) Gyr,
        ! tau, tb = tage_int - tburst_int = (tage - tburst) Gyr
        dtg = 0.0_wp; dtau = 0.0_wp; dtb = 0.0_wp
        select case (q)
        case (SFHT_TAGE)
            dtg = GYR; dtb = GYR
        case (SFHT_TAU)
            dtau = GYR
        case (SFHT_TBURST)
            dtb = -GYR
        case (SFHT_SF_START)
            if (pl%sfh /= 0) dtg = -GYR
        end select
        imax = min(max(find_interval(ctx%state%time_full, log10(sfh%tage)) + 2, 1), nt)

        if (pl%sfh == 0) then
            ! A burst at tb = tage
            sfh%type = -1
            sfh%tb = sfh%tage
            call ssp_weights_t(ctx, sfh, 0.0_wp, 0.0_wp, dtg, max(imax - 2, 1), imax, w, dw)
            return
        end if

        ! Tau or delayed tau, normalized to one solar mass
        sfh%type = pl%sfh
        call ssp_weights_t(ctx, sfh, dtg, dtau, 0.0_wp, 0, imax, w, dw)
        mass1 = 0.0_wp
        dmass1 = 0.0_wp
        do i = 1, imax
            mass1 = mass1 + w(i)
            dmass1 = dmass1 + dw(i)
        end do
        if (mass1 < SAFE_FLOOR) then
            mass1 = 1.0_wp
            dmass1 = 0.0_wp
        end if
        ! Constant and burst components (computed whatever const and fburst: the derivative with
        ! respect to them at 0 is the one-sided one)
        sfh%type = 0
        call ssp_weights_t(ctx, sfh, dtg, 0.0_wp, 0.0_wp, 0, imax, s(:, 1), s(:, 2))
        massc = 0.0_wp
        dmassc = 0.0_wp
        do i = 1, imax
            massc = massc + s(i, 1)
            dmassc = dmassc + s(i, 2)
        end do
        if (massc < SAFE_FLOOR) then
            massc = 1.0_wp
            dmassc = 0.0_wp
        end if
        do i = 1, nt
            s(i, 3) = 0.0_wp
            s(i, 4) = 0.0_wp
        end do
        fb = 0.0_wp
        if (sfh%tb >= 0.0_wp) then
            sfh%type = -1
            call ssp_weights_t(ctx, sfh, 0.0_wp, 0.0_wp, dtb, 0, imax, s(:, 3), s(:, 4))
            fb = pl%fburst
        end if
        c = pl%const
        if (.not. (pl%const > 0.0_wp .or. pl%fburst > SAFE_FLOOR)) then
            ! (the forward uses the tau part alone: the mixture with zero fractions, the same weights)
            c = 0.0_wp
            fb = 0.0_wp
        end if
        do i = 1, nt
            wt = w(i) / mass1
            dwt = (dw(i) - wt * dmass1) / mass1
            cn = s(i, 1) / massc
            dcn = (s(i, 2) - cn * dmassc) / massc
            w(i) = (1.0_wp - c - fb) * wt + c * cn + fb * s(i, 3)
            dw(i) = (1.0_wp - c - fb) * dwt + c * dcn + fb * s(i, 4)
            if (q == SFHT_CONST) dw(i) = dw(i) - wt + cn
            if (q == SFHT_FBURST .and. sfh%tb >= 0.0_wp) dw(i) = dw(i) - wt + s(i, 3)
        end do
    end subroutine sfh_weights_tangent

    !> compute_ssp_weights (types 0, 1, 4 and the burst, -1; no Simha limits) with tangents: dtg, dtau,
    !> dtb the tangents of sfh%tage, sfh%tau, sfh%tb (years). sfh%tq is constant (no truncation).
    subroutine ssp_weights_t(ctx, sfh, dtg, dtau, dtb, idx_min, idx_max, w, dw)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(sfhparams), intent(in)         :: sfh
        real(WP), intent(in)                :: dtg, dtau, dtb
        integer, intent(in)                 :: idx_min, idx_max
        real(WP), intent(inout)             :: w(:), dw(:)

        integer  :: j, nt, j_start, j_end
        real(WP) :: lim_min, lts, lte, dlte, l1, dl1, l2, dl2, lim_prev, inv_dt, node_lo, node_hi
        real(WP) :: m0p, dm0p, m1p, dm1p, m0c, dm0c, m1c, dm1c, m0, dm0, m1, dm1, ltb, dltb, bdt, tb
        real(WP) :: ma, dma, mb, dmb, m1a, dm1a, m1b, dm1b

        nt = ctx%state%ntfull
        do j = 1, nt
            w(j) = 0.0_wp
            dw(j) = 0.0_wp
        end do
        lim_min = 10.0_wp**ctx%tiny_logt_val

        if (sfh%type == -1) then
            ! The burst: linear in its age between the two bracketing grid ages
            if (sfh%tb < 0.0_wp) return
            tb = max(sfh%tb, lim_min)
            ltb = log10(tb)
            dltb = 0.0_wp
            if (sfh%tb > lim_min) dltb = dtb / (sfh%tb * LN10)
            if (ltb <= ctx%state%time_full(1)) then
                w(1) = 1.0_wp
                return
            end if
            j = min(max(find_interval(ctx%state%time_full, ltb), 1), nt - 1)
            if (ctx%interpolation_type_val == 0) then
                bdt = ctx%state%time_full(j + 1) - ctx%state%time_full(j)
                w(j) = (ctx%state%time_full(j + 1) - ltb) / bdt
                w(j + 1) = (ltb - ctx%state%time_full(j)) / bdt
                dw(j) = -dltb / bdt
                dw(j + 1) = dltb / bdt
            else
                bdt = 10.0_wp**ctx%state%time_full(j + 1) - 10.0_wp**ctx%state%time_full(j)
                w(j) = (10.0_wp**ctx%state%time_full(j + 1) - tb) / bdt
                w(j + 1) = (tb - 10.0_wp**ctx%state%time_full(j)) / bdt
                dw(j) = -tb * LN10 * dltb / bdt
                dw(j + 1) = tb * LN10 * dltb / bdt
            end if
            return
        end if

        ! Star-forming window [tq, tage] (log years); only its old end moves (tage)
        lts = log10(max(sfh%tq, lim_min))
        lte = log10(max(sfh%tage, lim_min))
        dlte = 0.0_wp
        if (sfh%tage > lim_min) dlte = dtg / (sfh%tage * LN10)
        j_start = find_interval(ctx%state%time_full, lts)
        j_end = find_interval(ctx%state%time_full, lte) + 1
        j_start = max(j_start, max(idx_min, 1))
        j_end = min(j_end, min(idx_max, nt - 1))

        call clamp_t(ctx%state%time_full(j_start), lts, lte, dlte, l1, dl1)
        lim_prev = l1
        call moments_t(ctx, l1, dl1, sfh, dtg, dtau, m0p, dm0p, m1p, dm1p)
        do j = j_start, j_end
            call clamp_t(ctx%state%time_full(j + 1), lts, lte, dlte, l2, dl2)
            ! (a segment of zero width is kept: it adds nothing to the weights, but where the window's
            ! end sits on a grid age its tangent is the one-sided one, upwards; the forward skips it)
            if (l2 < lim_prev) then
                lim_prev = l2
                cycle
            end if
            call moments_t(ctx, l2, dl2, sfh, dtg, dtau, m0c, dm0c, m1c, dm1c)
            m0 = m0c - m0p
            dm0 = dm0c - dm0p
            m1 = m1c - m1p
            dm1 = dm1c - dm1p
            m0p = m0c; dm0p = dm0c; m1p = m1c; dm1p = dm1c
            if (ctx%interpolation_type_val == 0) then
                node_lo = ctx%state%time_full(j)
                node_hi = ctx%state%time_full(j + 1)
            else
                node_lo = 10.0_wp**ctx%state%time_full(j)
                node_hi = 10.0_wp**ctx%state%time_full(j + 1)
            end if
            inv_dt = 1.0_wp / (node_hi - node_lo)
            w(j) = w(j) + (node_hi * m0 - m1) * inv_dt
            w(j + 1) = w(j + 1) + (m1 - node_lo * m0) * inv_dt
            dw(j) = dw(j) + (node_hi * dm0 - dm1) * inv_dt
            dw(j + 1) = dw(j + 1) + (dm1 - node_lo * dm0) * inv_dt
            lim_prev = l2
        end do

        ! The youngest edge: the mass formed before the first grid age goes to it
        if (idx_min == 0) then
            call clamp_t(ctx%tiny_logt_val, lts, lte, dlte, l1, dl1)
            call clamp_t(ctx%state%time_full(1), lts, lte, dlte, l2, dl2)
            if (l1 < l2) then
                call moments_t(ctx, l2, dl2, sfh, dtg, dtau, mb, dmb, m1b, dm1b)
                call moments_t(ctx, l1, dl1, sfh, dtg, dtau, ma, dma, m1a, dm1a)
                w(1) = w(1) + (mb - ma)
                dw(1) = dw(1) + (dmb - dma)
            end if
        end if
    end subroutine ssp_weights_t

    !> min(max(node, lts), lte) with its tangent (only lte moves).
    pure subroutine clamp_t(node, lts, lte, dlte, v, dv)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in)  :: node, lts, lte, dlte
        real(WP), intent(out) :: v, dv
        v = max(node, lts)
        dv = 0.0_wp
        if (lte < v) then
            v = lte
            dv = dlte
        end if
    end subroutine clamp_t

    !> The indefinite moments at log time lt (tangent dlt), as eval_indefinite_moments_log or _lin
    !> by the context's interpolation, with their total tangents (dtg, dtau: of sfh%tage, sfh%tau).
    pure subroutine moments_t(ctx, lt, dlt, sfh, dtg, dtau, m0, dm0, m1, dm1)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        real(WP), intent(in)  :: lt, dlt, dtg, dtau
        type(sfhparams), intent(in) :: sfh
        real(WP), intent(out) :: m0, dm0, m1, dm1
        real(WP) :: t, dt, x, dx, ex, ei, dei, a, da, tau, tg

        t = 10.0_wp**lt
        dt = t * LN10 * dlt
        tau = sfh%tau
        tg = sfh%tage
        m0 = 0.0_wp; dm0 = 0.0_wp; m1 = 0.0_wp; dm1 = 0.0_wp
        if (sfh%type == 1 .or. sfh%type == 4) then
            if (tau <= SAFE_FLOOR) return
        end if
        if (ctx%interpolation_type_val == 0) then
            select case (sfh%type)
            case (0)
                m0 = t
                dm0 = dt
                m1 = t * (lt - LOG_E)
                dm1 = dt * (lt - LOG_E) + t * dlt
            case (1, 4)
                x = t / tau
                dx = dt / tau - t * dtau / tau**2
                if (x < SAFE_FLOOR) then
                    x = SAFE_FLOOR
                    dx = 0.0_wp
                end if
                ex = exp(x)
                ei = expi(x)
                dei = ex / x * dx
                if (.not. ieee_is_finite(ei)) then
                    ei = 0.0_wp
                    dei = 0.0_wp
                end if
                if (sfh%type == 1) then
                    m0 = ex
                    dm0 = ex * dx
                    m1 = lt * m0 - LOG_E * ei
                    dm1 = dlt * m0 + lt * dm0 - LOG_E * dei
                else
                    a = t - tg - tau
                    da = dt - dtg - dtau
                    m0 = -a * ex
                    dm0 = -(da * ex + a * ex * dx)
                    m1 = -(a * lt - tau * LOG_E) * ex - (tg + tau) * LOG_E * ei
                    dm1 = -((da * lt + a * dlt - dtau * LOG_E) * ex + (a * lt - tau * LOG_E) * ex * dx) - &
                          (dtg + dtau) * LOG_E * ei - (tg + tau) * LOG_E * dei
                end if
            end select
        else
            select case (sfh%type)
            case (0)
                m0 = t
                dm0 = dt
                m1 = 0.5_wp * t**2
                dm1 = t * dt
            case (1)
                x = t / tau
                dx = dt / tau - t * dtau / tau**2
                ex = exp(x)
                m0 = ex
                dm0 = ex * dx
                m1 = (t - tau) * ex
                dm1 = (dt - dtau) * ex + m1 * dx
            case (4)
                x = t / tau
                dx = dt / tau - t * dtau / tau**2
                ex = exp(x)
                m0 = (tg - t + tau) * ex
                dm0 = (dtg - dt + dtau) * ex + m0 * dx
                m1 = (tg * (t - tau) - (t**2 - 2.0_wp * t * tau + 2.0_wp * tau**2)) * ex
                dm1 = (dtg * (t - tau) + tg * (dt - dtau) - &
                       (2.0_wp * t * dt - 2.0_wp * dt * tau - 2.0_wp * t * dtau + 4.0_wp * tau * dtau)) * ex + m1 * dx
            end select
        end if
    end subroutine moments_t

end module opensps_sfh_tangent
