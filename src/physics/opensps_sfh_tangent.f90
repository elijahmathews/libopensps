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
    !> Supported: sfh = 0 (SSP), 1 (tau), 4 (delayed tau), with const, fburst and truncation (the
    !> window's young end tq = tage - sf_trunc moves with both), and 5 (Simha: delayed tau before the
    !> truncation, linear after it, mixed by the linear part's mass fraction of
    !> get_sfh_properties_at_age, whose tangent is simha_frac_t). Parameters, in the units of PARAMS
    !> (Gyr): tage, tau, const, fburst, tburst, sf_start, sf_trunc, sf_slope (SFHT_*).
    !> Where tage (or the burst's age) sits exactly on a grid age the weights have a kink; the tangent
    !> there is the upward one (the derivative from above).

    use opensps_precision, only: WP
    use opensps_context_types, only: opensps_context_t
    use opensps_types, only: params_core, sfhparams
    use opensps_constants, only: SAFE_FLOOR
    use opensps_special_functions, only: expi, gammainc
    use opensps_interpolation, only: find_interval
    use opensps_sfh, only: convert_sfhparams
    use opensps_ieee_compat

    implicit none
    private

    public :: sfh_weights_tangent, sfh_tangent_supported
    public :: SFHT_TAGE, SFHT_TAU, SFHT_CONST, SFHT_FBURST, SFHT_TBURST, SFHT_SF_START, SFHT_SF_TRUNC, SFHT_SF_SLOPE, NSFHT

    integer, parameter :: SFHT_TAGE = 1, SFHT_TAU = 2, SFHT_CONST = 3, SFHT_FBURST = 4, SFHT_TBURST = 5, &
                          SFHT_SF_START = 6, SFHT_SF_TRUNC = 7, SFHT_SF_SLOPE = 8, NSFHT = 8
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

        sfh_tangent_supported = pl%sfh >= 0 .and. pl%sfh <= 5
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
        real(WP) :: dtg, dtau, dtb, dtq, ds, dt0, mass1, dmass1, massc, dmassc, c, fb, wt, dwt, cn, dcn
        real(WP) :: fl, dfl, mass2, dmass2
        integer  :: nt, imax, i

        nt = ctx%state%ntfull
        do i = 1, nt
            w(i) = 0.0_wp
            dw(i) = 0.0_wp
        end do
        if (pl%sfh /= 0 .and. pl%sfh /= 1 .and. pl%sfh /= 4 .and. pl%sfh /= 5) return
        call convert_sfhparams(pl, tage, sfh)
        ! The internal parameters' tangents (years; lookback time): tage_int = (tage - sf_start) Gyr,
        ! tau, tb = tage_int - tburst_int = (tage - tburst) Gyr, and with truncation (tq > 0)
        ! tq = tage_int - sf_trunc_int = (tage - sf_trunc) Gyr
        dtg = 0.0_wp; dtau = 0.0_wp; dtb = 0.0_wp; dtq = 0.0_wp; ds = 0.0_wp; dt0 = 0.0_wp
        select case (q)
        case (SFHT_SF_SLOPE)
            ds = -1.0_wp / GYR   ! sfh%sf_slope = -sf_slope / 1e9 (per year, increasing in lookback)
        case (SFHT_TAGE)
            dtg = GYR; dtb = GYR
            if (sfh%tq > 0.0_wp) dtq = GYR
        case (SFHT_SF_TRUNC)
            if (sfh%tq > 0.0_wp) dtq = -GYR
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
            call ssp_weights_t(ctx, sfh, 0.0_wp, 0.0_wp, dtg, 0.0_wp, 0.0_wp, 0.0_wp, max(imax - 2, 1), imax, w, dw)
            return
        end if

        if (pl%sfh == 5) then
            ! Simha: the delayed-tau part over [tq, tage] and the linear part over [t0, tq], each
            ! normalized, mixed by the linear part's mass fraction
            sfh%type = 4
            call ssp_weights_t(ctx, sfh, dtg, dtau, 0.0_wp, dtq, 0.0_wp, 0.0_wp, 0, imax, s(:, 1), s(:, 2))
            sfh%type = 5
            sfh%use_simha_limits = 1
            ! t0 = tq - 1 / slope (where the linear SFR reaches zero, if inside (0, tq])
            if (sfh%t0 > 0.0_wp) dt0 = dtq + ds / sfh%sf_slope**2
            call ssp_weights_t(ctx, sfh, dtg, 0.0_wp, 0.0_wp, dtq, dt0, ds, 0, imax, s(:, 3), s(:, 4))
            mass1 = 0.0_wp; dmass1 = 0.0_wp; mass2 = 0.0_wp; dmass2 = 0.0_wp
            do i = 1, imax
                mass1 = mass1 + s(i, 1); dmass1 = dmass1 + s(i, 2)
                mass2 = mass2 + s(i, 3); dmass2 = dmass2 + s(i, 4)
            end do
            if (mass1 < SAFE_FLOOR) then
                mass1 = 1.0_wp; dmass1 = 0.0_wp
            end if
            if (mass2 < SAFE_FLOOR) then
                mass2 = 1.0_wp; dmass2 = 0.0_wp
            end if
            call simha_frac_t(ctx, pl, tage, q, fl, dfl)
            do i = 1, nt
                wt = s(i, 1) / mass1
                dwt = (s(i, 2) - wt * dmass1) / mass1
                cn = s(i, 3) / mass2
                dcn = (s(i, 4) - cn * dmass2) / mass2
                w(i) = wt * (1.0_wp - fl) + cn * fl
                dw(i) = dwt * (1.0_wp - fl) + dcn * fl + (cn - wt) * dfl
            end do
            return
        end if

        ! Tau or delayed tau, normalized to one solar mass
        sfh%type = pl%sfh
        call ssp_weights_t(ctx, sfh, dtg, dtau, 0.0_wp, dtq, 0.0_wp, 0.0_wp, 0, imax, w, dw)
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
        call ssp_weights_t(ctx, sfh, dtg, 0.0_wp, 0.0_wp, dtq, 0.0_wp, 0.0_wp, 0, imax, s(:, 1), s(:, 2))
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
            call ssp_weights_t(ctx, sfh, 0.0_wp, 0.0_wp, dtb, 0.0_wp, 0.0_wp, 0.0_wp, 0, imax, s(:, 3), s(:, 4))
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

    !> compute_ssp_weights (types 0, 1, 4, 5 and the burst, -1) with tangents: dtg, dtau, dtb, dtq,
    !> dt0, ds the tangents of sfh%tage, sfh%tau, sfh%tb, sfh%tq, sfh%t0, sfh%sf_slope (years).
    subroutine ssp_weights_t(ctx, sfh, dtg, dtau, dtb, dtq, dt0, ds, idx_min, idx_max, w, dw)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(sfhparams), intent(in)         :: sfh
        real(WP), intent(in)                :: dtg, dtau, dtb, dtq, dt0, ds
        integer, intent(in)                 :: idx_min, idx_max
        real(WP), intent(inout)             :: w(:), dw(:)

        integer  :: j, nt, j_start, j_end
        real(WP) :: lim_min, lts, dlts, lte, dlte, l1, dl1, l2, dl2, lim_prev, inv_dt, node_lo, node_hi
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

        ! Star-forming window [tq, tage] (log years): the old end moves with tage, the young one with
        ! the truncation; Simha's linear part: [t0, tq]
        if (sfh%use_simha_limits == 1) then
            lts = log10(max(sfh%t0, lim_min))
            dlts = 0.0_wp
            if (sfh%t0 > lim_min) dlts = dt0 / (sfh%t0 * LN10)
            lte = log10(max(sfh%tq, lim_min))
            dlte = 0.0_wp
            if (sfh%tq > lim_min) dlte = dtq / (sfh%tq * LN10)
        else
            lts = log10(max(sfh%tq, lim_min))
            dlts = 0.0_wp
            if (sfh%tq > lim_min) dlts = dtq / (sfh%tq * LN10)
            lte = log10(max(sfh%tage, lim_min))
            dlte = 0.0_wp
            if (sfh%tage > lim_min) dlte = dtg / (sfh%tage * LN10)
        end if
        j_start = find_interval(ctx%state%time_full, lts)
        j_end = find_interval(ctx%state%time_full, lte) + 1
        j_start = max(j_start, max(idx_min, 1))
        j_end = min(j_end, min(idx_max, nt - 1))

        call clamp_t(ctx%state%time_full(j_start), lts, dlts, lte, dlte, l1, dl1)
        lim_prev = l1
        call moments_t(ctx, l1, dl1, sfh, dtg, dtau, dtq, ds, m0p, dm0p, m1p, dm1p)
        do j = j_start, j_end
            call clamp_t(ctx%state%time_full(j + 1), lts, dlts, lte, dlte, l2, dl2)
            ! (a segment of zero width is kept: it adds nothing to the weights, but where the window's
            ! end sits on a grid age its tangent is the one-sided one, upwards; the forward skips it)
            if (l2 < lim_prev) then
                lim_prev = l2
                cycle
            end if
            call moments_t(ctx, l2, dl2, sfh, dtg, dtau, dtq, ds, m0c, dm0c, m1c, dm1c)
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
            call clamp_t(ctx%tiny_logt_val, lts, dlts, lte, dlte, l1, dl1)
            call clamp_t(ctx%state%time_full(1), lts, dlts, lte, dlte, l2, dl2)
            if (l1 < l2) then
                call moments_t(ctx, l2, dl2, sfh, dtg, dtau, dtq, ds, mb, dmb, m1b, dm1b)
                call moments_t(ctx, l1, dl1, sfh, dtg, dtau, dtq, ds, ma, dma, m1a, dm1a)
                w(1) = w(1) + (mb - ma)
                dw(1) = dw(1) + (dmb - dma)
            end if
        end if
    end subroutine ssp_weights_t

    !> The Simha SFH's linear mass fraction (get_sfh_properties_at_age, sfh = 5, Gyr) and its
    !> tangent for parameter q: frac = m_lin / (m_lin + m_tau), m_tau = tau P(2, min(t', t_trunc) /
    !> tau) (d P(2, x) / dx = x e^-x), m_lin = sfr_trunc ((t_hi - t_tr) - s/2 (t_hi^2 + t_tr^2) +
    !> s t_hi t_tr), sfr_trunc = (t_tr / tau) e^(-t_tr / tau), s = -sf_slope, t_hi = min(t_zero, t').
    subroutine simha_frac_t(ctx, pl, tage, q, f, df)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(params_core), intent(in)       :: pl
        real(WP), intent(in)                :: tage
        integer, intent(in)                 :: q
        real(WP), intent(out)               :: f, df
        real(WP) :: tmax, dtmax, tpr, dtpr, ttr, dttr, tau, dtau, sm, dsm, tz, dtz, x, dx, mt, dmt
        real(WP) :: r, dr, sfr, dsfr, th, dth, bb, dbb, ml, dml, mn, dmn
        logical  :: trunc_on

        f = 0.0_wp
        df = 0.0_wp
        tmax = 10.0_wp**(ctx%state%time_full(ctx%state%ntfull) - 9.0_wp) - pl%sf_start
        tpr = tage - pl%sf_start
        trunc_on = .not. (pl%sf_trunc < SAFE_FLOOR .or. pl%sf_trunc < pl%sf_start)
        if (trunc_on) then
            ttr = pl%sf_trunc - pl%sf_start
        else
            ttr = tmax
        end if
        tau = max(pl%tau, SAFE_FLOOR)
        sm = -pl%sf_slope
        dtmax = 0.0_wp; dtpr = 0.0_wp; dttr = 0.0_wp; dtau = 0.0_wp; dsm = 0.0_wp
        select case (q)
        case (SFHT_TAGE)
            dtpr = 1.0_wp
        case (SFHT_TAU)
            if (pl%tau > SAFE_FLOOR) dtau = 1.0_wp
        case (SFHT_SF_START)
            dtmax = -1.0_wp
            dtpr = -1.0_wp
            dttr = -1.0_wp
        case (SFHT_SF_TRUNC)
            if (trunc_on) dttr = 1.0_wp
        case (SFHT_SF_SLOPE)
            dsm = -1.0_wp
        end select
        if (.not. trunc_on) dttr = dtmax
        if (tpr < 0.0_wp) return
        if (ttr <= 0.0_wp .or. ttr > tmax) return
        ! The delayed-tau mass formed by t'
        if (tpr <= ttr) then
            mn = tpr; dmn = dtpr
        else
            mn = ttr; dmn = dttr
        end if
        x = mn / tau
        dx = (dmn - x * dtau) / tau
        mt = tau * gammainc(2, x)
        dmt = dtau * gammainc(2, x) + tau * x * exp(-x) * dx
        ! The linear part's mass formed by t'
        if (tpr <= ttr) return
        if (sm > 0.0_wp) then
            tz = ttr + 1.0_wp / sm
            dtz = dttr - dsm / sm**2
        else
            tz = tmax
            dtz = dtmax
        end if
        if (tz <= tpr) then
            th = tz; dth = dtz
        else
            th = tpr; dth = dtpr
        end if
        r = ttr / pl%tau
        dr = (dttr - r * dtau) / pl%tau
        sfr = r * exp(-r)
        dsfr = exp(-r) * (1.0_wp - r) * dr
        bb = (th - ttr) - (sm / 2.0_wp) * (th**2 + ttr**2) + sm * th * ttr
        dbb = (dth - dttr) - (dsm / 2.0_wp) * (th**2 + ttr**2) - sm * (th * dth + ttr * dttr) + &
              dsm * th * ttr + sm * (dth * ttr + th * dttr)
        ml = sfr * bb
        dml = dsfr * bb + sfr * dbb
        if (ml + mt > SAFE_FLOOR) then
            f = ml / (ml + mt)
            df = (dml * mt - ml * dmt) / (ml + mt)**2
        end if
    end subroutine simha_frac_t

    !> min(max(node, lts), lte) with its tangent (both window ends move; at equality the upward
    !> derivative of the kept value).
    pure subroutine clamp_t(node, lts, dlts, lte, dlte, v, dv)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in)  :: node, lts, dlts, lte, dlte
        real(WP), intent(out) :: v, dv
        v = node
        dv = 0.0_wp
        if (lts > node) then
            v = lts
            dv = dlts
        end if
        if (lte < v) then
            v = lte
            dv = dlte
        end if
    end subroutine clamp_t

    !> The indefinite moments at log time lt (tangent dlt), as eval_indefinite_moments_log or _lin
    !> by the context's interpolation, with their total tangents (dtg, dtau: of sfh%tage, sfh%tau).
    pure subroutine moments_t(ctx, lt, dlt, sfh, dtg, dtau, dtq, ds, m0, dm0, m1, dm1)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        real(WP), intent(in)  :: lt, dlt, dtg, dtau, dtq, ds
        type(sfhparams), intent(in) :: sfh
        real(WP), intent(out) :: m0, dm0, m1, dm1
        real(WP) :: t, dt, x, dx, ex, ei, dei, a, da, tau, tg, tp, dtp, sf, dsf, sl

        t = 10.0_wp**lt
        dt = t * LN10 * dlt
        tau = sfh%tau
        tg = sfh%tage
        m0 = 0.0_wp; dm0 = 0.0_wp; m1 = 0.0_wp; dm1 = 0.0_wp
        if (sfh%type == 1 .or. sfh%type == 4) then
            if (tau <= SAFE_FLOOR) return
        end if
        ! Simha's linear part: slope factor 1 - s t', t' = max(0, tage - sf_trunc) (= tq, moving with it)
        sl = sfh%sf_slope
        tp = max(0.0_wp, tg - sfh%sf_trunc)
        dtp = 0.0_wp
        if (tg - sfh%sf_trunc > 0.0_wp) dtp = dtq
        sf = 1.0_wp - sl * tp
        dsf = -ds * tp - sl * dtp
        if (ctx%interpolation_type_val == 0) then
            select case (sfh%type)
            case (0)
                m0 = t
                dm0 = dt
                m1 = t * (lt - LOG_E)
                dm1 = dt * (lt - LOG_E) + t * dlt
            case (5)
                m0 = sf * t + sl * t**2 / 2.0_wp
                dm0 = dsf * t + sf * dt + ds * t**2 / 2.0_wp + sl * t * dt
                m1 = sf * t * (lt - LOG_E) + (sl * t**2 / 2.0_wp) * (lt - 0.5_wp * LOG_E)
                dm1 = dsf * t * (lt - LOG_E) + sf * (dt * (lt - LOG_E) + t * dlt) + &
                      (ds * t**2 / 2.0_wp + sl * t * dt) * (lt - 0.5_wp * LOG_E) + (sl * t**2 / 2.0_wp) * dlt
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
            case (5)
                m0 = sf * t + sl * 0.5_wp * t**2
                dm0 = dsf * t + sf * dt + ds * 0.5_wp * t**2 + sl * t * dt
                m1 = sf * 0.5_wp * t**2 + sl * t**3 / 3.0_wp
                dm1 = dsf * 0.5_wp * t**2 + sf * t * dt + ds * t**3 / 3.0_wp + sl * t**2 * dt
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
