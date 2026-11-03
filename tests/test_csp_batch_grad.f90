program test_csp_batch_grad
    !> Gradients of the batched CSP (compute_csp_batch with mags_bar / spec_bar and params_bar /
    !> weights_bar) against finite differences of the loss they differentiate,
    !>     L = sum(mags_bar * mags) + sum(spec_bar * spec),
    !> for the stages whose adjoints are written (observed frame and band integrals, AGN, IGM,
    !> attenuation, age sums):
    !> - d L / d fagn per model (central differences; one-sided at fagn = 0, where the AGN stage of
    !>   the forward model switches on);
    !> - d L / d (dust1, dust2, dust3, dust_index, dust1_index, frac_obrun, frac_nodust, mwr, uvb) per
    !>   model (one-sided, second order, where a parameter sits at its bound 0), and with dust emission
    !>   (energy balance) also duste_qpah, duste_umin, duste_gamma; with velocity smoothing; with
    !>   nebular emission (continuum and lines) also gas_logz and gas_logu; with every metallicity's
    !>   basis resident, logzsol;
    !> - d L / d(SSP weights) along a random direction (relative perturbations of the weights, so
    !>   they stay positive), and for single ages, including an age whose weight is zero;
    !> - the weights' gradient of a parametric SFH equals that of the same weights given as a matrix;
    !> - with SFH parameters, d L / d(tage, tau, const, fburst, tburst, sf_start, sf_trunc, sf_slope);
    !> - d L / d zred above z = 0 (resampling, distance, IGM);
    !> - with smoothing, d L / d sigma_smooth (shared by the batch: the sum of the models' rows);
    !> - unsupported stages return status 2.
    !> The spectra are linear in the weights and in fagn, so the differences are exact up to rounding
    !> and the curvature of the magnitudes (log10 of the band fluxes). FP64 builds are the
    !> validation (roadmap priority 4); FP32 builds check the same with differences sized for FP32.
    !> Requires the data tree, like the regression tests.

    use opensps_precision, only: WP, KP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup
    use opensps_csp, only: sfh_weights_core
    use opensps_csp_batch, only: compute_csp_batch, BGRAD_FAGN, BGRAD_DUST1, BGRAD_DUST2, BGRAD_DUST3, &
                                 BGRAD_DUST_INDEX, BGRAD_DUST1_INDEX, BGRAD_FRAC_OBRUN, BGRAD_FRAC_NODUST, &
                                 BGRAD_MWR, BGRAD_UVB, BGRAD_DUSTE_QPAH, BGRAD_DUSTE_UMIN, BGRAD_DUSTE_GAMMA, &
                                 BGRAD_GAS_LOGZ, BGRAD_GAS_LOGU, BGRAD_LOGZSOL, BGRAD_TAGE, BGRAD_TAU, BGRAD_CONST, &
                                 BGRAD_FBURST, BGRAD_TBURST, BGRAD_SF_START, BGRAD_ZRED, BGRAD_SIGMA_SMOOTH, BGRAD_SF_TRUNC, &
                                 BGRAD_SF_SLOPE, NBGRAD
    use opensps_ssp, only: generate_ssp_grid
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_move_to_device, &
                               opensps_context_set_pset, opensps_context_get_pset, opensps_context_prepare_pset, &
                               opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use test_utils_mod, only: tap_finish, skip_without_data, print_major_header, print_group, print_summary_line, &
                              assert_true
    implicit none

    integer, parameter :: N_MOD = 6
    integer :: total_tests = 0
    integer :: total_failures = 0
    !> Relative step of the finite differences, and the tolerance on the relative difference
    !> |g - g_fd| / max(|g_fd|, scale). FP64: central differences at h = 1e-4 leave the
    !> magnitudes' curvature (~h^2) and rounding (~1e-12); FP32: h = 1e-2, rounding of FP32 spectra
    !> and magnitudes (~1e-6 / h) dominates
    real(WP), parameter :: H_REL = merge(1.0e-4_wp, 1.0e-2_wp, KP == WP)
    real(WP), parameter :: GRAD_RTOL = merge(1.0e-6_wp, 1.0e-3_wp, KP == WP)
    !> FP32 builds check fagn and single ages with the spectrum part of the loss alone (linear in
    !> both), at steps H_LIN
    logical, parameter  :: LINEAR_ONLY = KP /= WP
    real(WP), parameter :: H_LIN = 0.5_wp
    !> Magnitude cotangents only on bands within MAG_RANGE of the model's brightest
    real(WP), parameter :: MAG_RANGE = 10.0_wp
    !> Dust parameters (nonlinear): steps and tolerances of their finite differences
    !> FP32: the loss's spectrum part is not linear in them, so FP32 builds check interior points
    !> only (central differences at h = 1e-2: the offload builds' host fallback sums the forward in
    !> plain FP32 trees, whose rounding dominated at 1e-3), relative to the larger of |g_fd| and 1e-2
    !> of the model's largest gradient; the bounds are validated in FP64
    real(WP), parameter :: H_DUST = 1.0e-5_wp, H_DUST32 = 1.0e-2_wp
    real(WP), parameter :: DUST_RTOL = 1.0e-5_wp, DUST_RTOL32 = 2.0e-2_wp
    !> Per-model redshifts, off round values: at z = 0.5 (or 2, 3) many observed pixels fall exactly on
    !> rest-frame pixels, where the resampled spectrum has a kink in z (the gradient there is
    !> one-sided, central differences average the two sides)
    real(WP), parameter :: ZLIST(N_MOD) = [0.0_wp, 0.113_wp, 0.507_wp, 1.213_wp, 2.017_wp, 3.011_wp]
    type(opensps_context_t), allocatable :: ctx
    integer :: seed = 12345
    logical :: zred_per_model = .false.
    real(WP) :: smooth_sigma = 0.0_wp   ! sigma_smooth of every model
    !> The cotangents of the current check (the loss L is formed with them)
    real(WP), allocatable :: lbar_m(:, :), lbar_s(:, :)

    call skip_without_data()
    call print_major_header("Batched CSP gradients")

    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 10
    call opensps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call opensps_context_move_to_device(ctx)
    call set_int('zmet', 10)
    call set_int('imf_type', 1)
    call set_int('add_dust_emission', 0)
    call set_int('add_neb_emission', 0)
    call build_basis()

    call check_grad(2, 0.0_wp, 0, 1, .true., "Calzetti, z = 0, AGN, weight matrix")
    call check_grad(0, 0.513_wp, 1, 1, .false., "power law, z = 0.513, IGM, AGN, SFH parameters")
    call check_grad(1, 0.0_wp, 0, 0, .true., "Cardelli et al., z = 0, weight matrix")
    zred_per_model = .true.
    call check_grad(4, 0.0_wp, 1, 1, .true., "Kriek & Conroy, per-model zred 0-3, IGM, AGN, weight matrix")
    zred_per_model = .false.
    ! Dust emission (energy balance), with AGN and IGM
    call set_int('add_dust_emission', 1)
    call check_grad(2, 0.0_wp, 0, 1, .true., "Calzetti, z = 0, dust emission, AGN, weight matrix")
    call check_grad(1, 0.513_wp, 1, 0, .false., "Cardelli et al., z = 0.513, IGM, dust emission, SFH parameters")
    ! Velocity smoothing (the device operator), with dust emission, AGN and IGM
    call set_int('smooth_velocity', 1)
    smooth_sigma = 150.0_wp
    call check_grad(0, 0.513_wp, 1, 1, .true., "power law, z = 0.513, IGM, dust emission, AGN, sigma_smooth = 150 km/s")
    smooth_sigma = 0.0_wp
    call set_int('add_dust_emission', 0)
    ! Nebular emission (continuum and lines in the spectrum)
    call set_int('add_neb_emission', 1)
    call check_grad(2, 0.0_wp, 0, 0, .true., "Calzetti, z = 0, nebular, weight matrix")
    call set_int('add_dust_emission', 1)
    call check_grad(1, 0.513_wp, 1, 1, .false., "Cardelli et al., z = 0.513, IGM, nebular, dust emission, AGN, SFH parameters")
    call set_int('add_dust_emission', 0)
    call set_int('add_neb_emission', 0)
    call check_unsupported()
    ! Per-model metallicity: every metallicity's basis resident (the SSPs interpolated in log Z)
    call opensps_setup(ctx, 0, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call opensps_context_move_to_device(ctx)
    call set_int('zmet', 10)
    call set_int('imf_type', 1)
    call build_basis_all()
    call set_int('add_neb_emission', 1)
    call set_int('add_dust_emission', 1)
    call check_grad(2, 0.513_wp, 1, 1, .true., "Calzetti, z = 0.513, per-model logzsol, IGM, nebular, dust emission, AGN")
    call set_int('add_dust_emission', 0)
    call set_int('add_neb_emission', 0)

    call print_summary_line("Batched CSP gradients", total_tests - total_failures, total_tests)
    call tap_finish()

contains

    subroutine build_basis()
        type(PARAMS) :: pset
        real(WP), allocatable :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)

        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        allocate(mass_ssp(ctx%state%ntfull), lbol_ssp(ctx%state%ntfull), &
                 spec_ssp(ctx%state%nspec, ctx%state%ntfull))
        call opensps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
        call opensps_context_update_ssp_basis(ctx, reshape(spec_ssp, [size(spec_ssp, 1), size(spec_ssp, 2), 1]), &
                                              reshape(mass_ssp, [size(mass_ssp), 1]), &
                                              reshape(lbol_ssp, [size(lbol_ssp), 1]), 1)
    end subroutine build_basis

    !> The bases of every metallicity of the context (per-model logzsol).
    subroutine build_basis_all()
        type(PARAMS) :: pset
        real(WP), allocatable :: spec_z(:, :, :), mass_z(:, :), lbol_z(:, :)
        integer :: z, nz

        nz = ctx%state%nz
        allocate(spec_z(ctx%state%nspec, ctx%state%ntfull, nz), mass_z(ctx%state%ntfull, nz), &
                 lbol_z(ctx%state%ntfull, nz))
        do z = 1, nz
            call set_int('zmet', z)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call opensps_context_set_pset(ctx, pset)
            call generate_ssp_grid(ctx, pset, mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
        end do
        call set_int('zmet', 10)
        ctx%state%ssp_basis_is_dirty = .true.
        call opensps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, nz)
    end subroutine build_basis_all

    !> Model m: SFH, dust and AGN parameters over their usual ranges (fagn = 0 for model 6).
    subroutine set_model(m, zred)
        integer, intent(in) :: m
        real(WP), intent(in) :: zred
        ! (SFH forms with parameter gradients: an SSP, tau and delayed tau with constant, burst,
        ! truncation and a late start, Simha)
        integer, parameter :: sfh(N_MOD) = [1, 4, 0, 1, 4, 5]
        ! (off the grid ages, where the weights have a kink in tage)
        real(WP), parameter :: tage(N_MOD) = [10.3_wp, 3.1_wp, 7.2_wp, 0.31_wp, 8.3_wp, 12.7_wp]
        real(WP), parameter :: tau(N_MOD) = [1.0_wp, 0.5_wp, 2.0_wp, 0.1_wp, 3.0_wp, 0.7_wp]
        real(WP), parameter :: dust1(N_MOD) = [0.0_wp, 0.6_wp, 1.0_wp, 1.5_wp, 0.0_wp, 0.4_wp]
        real(WP), parameter :: dust2(N_MOD) = [0.3_wp, 0.0_wp, 0.5_wp, 2.0_wp, 0.1_wp, 0.2_wp]
        real(WP), parameter :: dindex(N_MOD) = [-0.7_wp, -0.3_wp, -1.2_wp, -0.7_wp, 0.2_wp, -0.4_wp]
        real(WP), parameter :: fagn(N_MOD) = [0.05_wp, 0.2_wp, 0.01_wp, 0.5_wp, 0.1_wp, 0.0_wp]
        ! Inside the Draine & Li grid intervals (the derivative of the linear interpolation; at a
        ! node it is one-sided and differences straddle the kink)
        real(WP), parameter :: umin(N_MOD) = [1.1_wp, 4.3_wp, 13.0_wp, 0.55_wp, 2.2_wp, 0.9_wp]
        real(WP), parameter :: qpah(N_MOD) = [3.3_wp, 0.6_wp, 4.4_wp, 2.1_wp, 1.2_wp, 3.0_wp]
        real(WP), parameter :: gam(N_MOD) = [0.01_wp, 0.1_wp, 0.05_wp, 0.3_wp, 0.02_wp, 0.5_wp]
        ! Inside the CLOUDY grid intervals
        real(WP), parameter :: glogz(N_MOD) = [-1.37_wp, -0.62_wp, 0.13_wp, -0.29_wp, -1.81_wp, 0.37_wp]
        real(WP), parameter :: glogu(N_MOD) = [-2.3_wp, -3.15_wp, -1.7_wp, -3.6_wp, -2.6_wp, -1.35_wp]
        ! Inside the metallicity grid's intervals
        real(WP), parameter :: logz(N_MOD) = [-1.37_wp, -0.62_wp, 0.13_wp, -0.29_wp, -1.81_wp, 0.07_wp]

        call set_int('sfh', sfh(m))
        call set_float('tage', tage(m))
        call set_float('tau', tau(m))
        call set_float('sf_trunc', merge(0.71_wp * tage(m), 0.0_wp, m == 1 .or. m == 4 .or. m == 6))   ! (truncated)
        call set_float('sf_slope', merge(-1.3_wp, 0.0_wp, m == 6))
        call set_float('const', merge(0.15_wp, 0.0_wp, m == 2 .or. m == 6))
        call set_float('fburst', merge(0.2_wp, 0.0_wp, m == 1 .or. m == 5))
        call set_float('tburst', 0.63_wp * tage(m))
        call set_float('sf_start', merge(0.43_wp, 0.0_wp, m == 2 .or. m == 5))   ! (before tage)
        call set_float('dust1', dust1(m))
        call set_float('dust2', dust2(m))
        call set_float('dust_index', dindex(m))
        call set_float('dust_tesc', merge(7.5_wp, 7.0_wp, mod(m, 2) == 0))
        ! (interior values: at frac_nodust = 0 a band attenuated by ~10 mag has a curvature in
        ! frac_nodust that no finite difference resolves)
        call set_float('frac_obrun', 0.02_wp + 0.1_wp * mod(m, 3))
        call set_float('frac_nodust', 0.01_wp + 0.05_wp * mod(m, 4))
        call set_float('dust3', merge(0.3_wp, 0.0_wp, m == 3))
        call set_float('mwr', merge(4.0_wp, 3.1_wp, mod(m, 2) == 1))
        call set_float('uvb', merge(0.5_wp, 1.0_wp, mod(m, 3) == 2))
        call set_float('fagn', fagn(m))
        call set_float('duste_umin', umin(m))
        call set_float('duste_qpah', qpah(m))
        call set_float('duste_gamma', gam(m))
        call set_float('gas_logz', glogz(m))
        call set_float('gas_logu', glogu(m))
        call set_float('logzsol', logz(m))
        call set_float('agn_tau', 5.0_wp + 15.0_wp * mod(m, 8))
        call set_float('sigma_smooth', smooth_sigma)
        call set_float('zred', zred)
    end subroutine set_model

    subroutine check_grad(dust_type, zred, igm, agn, given_w, label)
        integer, intent(in) :: dust_type, igm, agn
        real(WP), intent(in) :: zred
        logical, intent(in) :: given_w
        character(len=*), intent(in) :: label
        type(PARAMS) :: psets(N_MOD), pp(N_MOD)
        real(WP), allocatable :: mags(:, :), spec(:, :), mbar(:, :), sbar(:, :), pbar(:, :), wbar(:, :), wbar2(:, :)
        real(WP), allocatable :: w(:, :), wp_(:, :), wm(:, :), dir(:, :), t1(:), t2(:)
        real(WP) :: lp, lm, l0, l3, g_fd, g_ad, h, worst, scale
        integer :: m, b, i, k, nspec, nbands, nt, status, k0

        call print_group("Gradients: " // label)
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        nt = ctx%state%ntfull
        allocate(mags(nbands, N_MOD), spec(nspec, N_MOD), mbar(nbands, N_MOD), sbar(nspec, N_MOD), &
                 pbar(NBGRAD, N_MOD), wbar(nt, N_MOD), wbar2(nt, N_MOD), w(nt, N_MOD), wp_(nt, N_MOD), &
                 wm(nt, N_MOD), dir(nt, N_MOD), t1(nt), t2(nt))
        call set_int('dust_type', dust_type)
        call set_int('add_igm_absorption', igm)
        call set_int('add_agn_dust', agn)
        do m = 1, N_MOD
            call set_model(m, merge(ZLIST(m), zred, zred_per_model))
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, psets(m))
            call sfh_weights_core(ctx, psets(m), psets(m)%tage, 1, w(:, m:m), t1, t2)
        end do

        ! The loss: magnitudes with O(1) cotangents, the spectrum with cotangents scaled by the
        ! model's peak, so that both parts of L are of order one
        if (allocated(lbar_m)) deallocate(lbar_m, lbar_s)
        call loss(psets, w, given_w, lp, mags, spec)
        do m = 1, N_MOD
            do b = 1, nbands
                mbar(b, m) = 0.0_wp
                if (mags(b, m) == mags(b, m)) mbar(b, m) = 2.0_wp * urand() - 1.0_wp
                ! Only bands within MAG_RANGE of the brightest: a band attenuated to nearly nothing has
                ! a derivative that is right but local (d mag / d frac_nodust ~ unattenuated / attenuated
                ! flux, 1e25 here), which finite differences cannot resolve; FP32 spectra of the
                ! IGM-blanketed bands at z = 3 also reach the denormal range
                if (mags(b, m) > minval(mags(:, m), mask=mags(:, m) == mags(:, m)) + MAG_RANGE) mbar(b, m) = 0.0_wp
            end do
            scale = maxval(abs(spec(:, m)))
            do i = 1, nspec
                sbar(i, m) = (2.0_wp * urand() - 1.0_wp) * 10.0_wp / (scale * nspec)
            end do
        end do
        lbar_m = mbar
        lbar_s = sbar
        if (given_w) then
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, ssp_weights=w, mags_bar=mbar, spec_bar=sbar, &
                                   params_bar=pbar, weights_bar=wbar)
        else
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, mags_bar=mbar, spec_bar=sbar, &
                                   params_bar=pbar, weights_bar=wbar)
        end if
        call assert_true(status == 0, "status 0 with gradients", total_tests, total_failures)
        if (status /= 0) return

        ! The weights along a random direction (relative perturbations, so they stay positive);
        ! with SFH parameters, the gradient with respect to the weights they give
        if (given_w) then
            do m = 1, N_MOD
                do k = 1, nt
                    dir(k, m) = w(k, m) * (2.0_wp * urand() - 1.0_wp)
                end do
            end do
            wp_ = w + H_REL * dir
            wm = w - H_REL * dir
            call loss(psets, wp_, .true., lp)
            call loss(psets, wm, .true., lm)
            g_fd = (lp - lm) / (2.0_wp * H_REL)
            g_ad = sum(wbar * dir)
            write(*, '(A,2ES16.8,A,ES10.2)') "# weights, random direction (adjoint, differences): ", g_ad, g_fd, &
                ", relative ", abs(g_ad - g_fd) / abs(g_fd)
            call assert_true(abs(g_ad - g_fd) <= GRAD_RTOL * abs(g_fd), "d L / d weights along a direction", &
                             total_tests, total_failures)
        else
            ! The gradient with respect to the weights the SFH parameters give equals that of the
            ! same weights given as a matrix
            call compute_csp_batch(ctx, psets, mags, status, ssp_weights=w, mags_bar=mbar, spec_bar=sbar, &
                                   weights_bar=wbar2)
            worst = maxval(abs(wbar - wbar2)) / maxval(abs(wbar2))
            write(*, '(A,ES10.2)') "# SFH parameters vs weight matrix: ", worst
            call assert_true(status == 0 .and. worst <= merge(1.0e-12_wp, 1.0e-5_wp, KP == WP), &
                             "weights gradient: SFH parameters = weight matrix", total_tests, total_failures)
        end if

        ! FP32 builds: finite differences of FP32 magnitudes are rounding noise at any step that
        ! keeps their curvature small, so the checks below take the spectrum part of the loss
        ! alone, which is linear in fagn and in the weights (any step is exact up to rounding)
        if (LINEAR_ONLY) then
            lbar_m = 0.0_wp
            if (given_w) then
                call compute_csp_batch(ctx, psets, mags, status, ssp_weights=w, mags_bar=lbar_m, spec_bar=sbar, &
                                       params_bar=pbar, weights_bar=wbar)
            else
                call compute_csp_batch(ctx, psets, mags, status, mags_bar=lbar_m, spec_bar=sbar, &
                                       params_bar=pbar, weights_bar=wbar)
            end if
        end if

        ! d L / d fagn, model by model
        if (agn == 1) then
            worst = 0.0_wp
            do m = 1, N_MOD
                pp = psets
                if (psets(m)%fagn > 0.0_wp) then
                    h = merge(H_LIN, H_REL, LINEAR_ONLY) * psets(m)%fagn
                    pp(m)%fagn = psets(m)%fagn + h
                    call loss(pp, w, given_w, lp)
                    pp(m)%fagn = psets(m)%fagn - h
                    call loss(pp, w, given_w, lm)
                    g_fd = (lp - lm) / (2.0_wp * h)
                else
                    ! One-sided at fagn = 0, third order (-11 L0 + 18 L1 - 9 L2 + 2 L3) / 6h (where the AGN
                    ! dominates a band the magnitudes' curvature in fagn is large)
                    h = merge(0.1_wp, 1.0e-6_wp, LINEAR_ONLY)
                    call loss(pp, w, given_w, l0)
                    pp(m)%fagn = h
                    call loss(pp, w, given_w, lp)
                    pp(m)%fagn = 2.0_wp * h
                    call loss(pp, w, given_w, lm)
                    pp(m)%fagn = 3.0_wp * h
                    call loss(pp, w, given_w, l3)
                    g_fd = (-11.0_wp * l0 + 18.0_wp * lp - 9.0_wp * lm + 2.0_wp * l3) / (6.0_wp * h)
                end if
                g_ad = pbar(BGRAD_FAGN, m)
                worst = max(worst, abs(g_ad - g_fd) / max(abs(g_fd), 1.0e-3_wp))
            end do
            write(*, '(A,ES10.2)') "# fagn: worst relative difference ", worst
            ! (FP64: the differences' rounding reaches ~1e-6 where an AGN dominates a band: 5e-6)
            call assert_true(worst <= merge(GRAD_RTOL, 5.0e-6_wp, LINEAR_ONLY), "d L / d fagn against finite differences", &
                             total_tests, total_failures)
        end if

        call check_dust(psets, w, given_w, pbar)

        ! sigma_smooth: one value for the batch, so the total derivative (all models' sigma moved)
        ! against the sum of the rows. (The forward jumps by ~3e-4 of a window's edge coefficient
        ! where a window edge crosses a pixel, at N_SIGMA_RANGE sigma; relative steps of 1e-5 balance
        ! those against rounding: 5e-6 measured)
        if (smooth_sigma > 0.0_wp) then
            h = merge(1.0e-2_wp, 1.0e-5_wp, LINEAR_ONLY) * smooth_sigma
            pp = psets
            pp(:)%sigma_smooth = smooth_sigma + h
            call loss(pp, w, given_w, lp)
            pp(:)%sigma_smooth = smooth_sigma - h
            call loss(pp, w, given_w, lm)
            g_fd = (lp - lm) / (2.0_wp * h)
            g_ad = sum(pbar(BGRAD_SIGMA_SMOOTH, :))
            write(*, '(A,2ES16.8,A,ES10.2)') "# sigma_smooth (adjoint, differences): ", g_ad, g_fd, ", relative ", &
                abs(g_ad - g_fd) / abs(g_fd)
            ! (FP32: an absolute floor of 1e-5, the spectrum part's FP32 rounding over the step being
            ! ~1e-7: its sigma gradient can be ~1e-9)
            call assert_true(abs(g_ad - g_fd) <= merge(2.0e-2_wp, 1.0e-4_wp, LINEAR_ONLY) * &
                             max(abs(g_fd), merge(1.0e-5_wp, 0.0_wp, LINEAR_ONLY)), &
                             "d L / d sigma_smooth against finite differences", total_tests, total_failures)
        end if

        ! Single ages of model 2: its oldest weighted age, its youngest, and an age with zero weight
        ! (one-sided there: weights are masses, >= 0)
        if (given_w) then
            worst = 0.0_wp
            m = 2
            do i = 1, 3
                select case (i)
                case (1)
                    k0 = maxloc(w(:, m), 1)
                case (2)
                    k0 = findloc(w(:, m) > 0.0_wp, .true., 1)
                case (3)
                    k0 = findloc(w(:, m) > 0.0_wp, .false., 1, back=.true.)
                end select
                if (k0 < 1) cycle
                ! (relative to the age's own weight: a young age that dominates a UV band makes its
                ! magnitude strongly curved in that weight)
                if (w(k0, m) > 0.0_wp) then
                    h = merge(H_LIN, 1.0e-5_wp, LINEAR_ONLY) * w(k0, m)
                else
                    h = merge(H_LIN, 1.0e-5_wp, LINEAR_ONLY) * maxval(w(:, m))
                end if
                wp_ = w
                wp_(k0, m) = w(k0, m) + h
                call loss(psets, wp_, .true., lp)
                if (w(k0, m) > h) then
                    wm = w
                    wm(k0, m) = w(k0, m) - h
                    call loss(psets, wm, .true., lm)
                    g_fd = (lp - lm) / (2.0_wp * h)
                else
                    call loss(psets, w, .true., l0)
                    wm = w
                    wm(k0, m) = w(k0, m) + 2.0_wp * h
                    call loss(psets, wm, .true., lm)
                    g_fd = (-3.0_wp * l0 + 4.0_wp * lp - lm) / (2.0_wp * h)
                end if
                worst = max(worst, abs(wbar(k0, m) - g_fd) / max(abs(g_fd), tiny(1.0_wp)))
            end do
            write(*, '(A,ES10.2)') "# single ages: worst relative difference ", worst
            ! (FP64: the step is relative to the age's own weight, so for the smallest weights the
            ! differences' rounding reaches ~1e-5: 5e-5. FP32: L changes by one age's share, ~1e-4 of L
            ! for a young age, whose FP32 rounding, ~1e-7 of L, is ~1e-3 of the difference: 5e-3)
            call assert_true(worst <= merge(5.0e-3_wp, 5.0e-5_wp, LINEAR_ONLY), "d L / d weight, single ages", &
                             total_tests, total_failures)
        end if
    end subroutine check_grad

    !> d L / d(dust parameter) of every model against finite differences of L.
    subroutine check_dust(psets, w, given_w, pbar)
        type(PARAMS), intent(in) :: psets(:)
        real(WP), intent(in) :: w(:, :), pbar(:, :)
        logical, intent(in) :: given_w
        integer, parameter :: NQ = 24
        integer, parameter :: ROWS(NQ) = [BGRAD_DUST1, BGRAD_DUST2, BGRAD_DUST3, BGRAD_DUST_INDEX, BGRAD_DUST1_INDEX, &
                                          BGRAD_FRAC_OBRUN, BGRAD_FRAC_NODUST, BGRAD_MWR, BGRAD_UVB, BGRAD_DUSTE_QPAH, &
                                          BGRAD_DUSTE_UMIN, BGRAD_DUSTE_GAMMA, BGRAD_GAS_LOGZ, BGRAD_GAS_LOGU, &
                                          BGRAD_LOGZSOL, BGRAD_TAGE, BGRAD_TAU, BGRAD_CONST, BGRAD_FBURST, BGRAD_TBURST, &
                                          BGRAD_SF_START, BGRAD_ZRED, BGRAD_SF_TRUNC, BGRAD_SF_SLOPE]
        character(len=12), parameter :: NAMES(NQ) = [character(len=12) :: 'dust1', 'dust2', 'dust3', 'dust_index', &
                                                     'dust1_index', 'frac_obrun', 'frac_nodust', 'mwr', 'uvb', &
                                                     'duste_qpah', 'duste_umin', 'duste_gamma', 'gas_logz', 'gas_logu', &
                                                     'logzsol', 'tage', 'tau', 'const', 'fburst', 'tburst', 'sf_start', &
                                                     'zred', 'sf_trunc', 'sf_slope']
        type(PARAMS) :: pp(size(psets))
        real(WP) :: g_fd(NQ, size(psets)), lp, lm, l0, l3, x, h, worst(NQ), gmax
        integer :: q, m, worst_m(NQ)
        logical :: bound

        do q = 1, NQ
            do m = 1, size(psets)
                pp = psets
                x = get_par(pp(m), q)
                ! Parameters at their bound 0 (masses of dust, fractions): one-sided, second order
                bound = q /= 4 .and. q /= 5 .and. q /= 8 .and. q /= 9 .and. q /= 15 .and. q /= 24 .and. x <= 0.0_wp
                ! (sf_trunc = 0: no truncation, not a bound of a continuous parameter; not checked)
                if (q == 23 .and. x <= 0.0_wp) then
                    g_fd(q, m) = pbar(ROWS(q), m)
                    cycle
                end if
                ! (zred: zero at z = 0 by definition, the rest frame; not checked there)
                if (q == 22 .and. x <= 0.0_wp) then
                    g_fd(q, m) = pbar(ROWS(q), m)
                    cycle
                end if
                if (bound .and. LINEAR_ONLY) then
                    g_fd(q, m) = pbar(ROWS(q), m)   ! (FP32: not checked at a bound)
                    cycle
                end if
                ! (the template's parameters: steps inside their grid interval)
                h = merge(H_DUST32, H_DUST, LINEAR_ONLY) * max(abs(x), 1.0_wp)
                if (q >= 10) h = merge(H_DUST32, H_DUST, LINEAR_ONLY) * max(abs(x), 0.01_wp)
                if (q >= 13) h = merge(H_DUST32, H_DUST, LINEAR_ONLY)
                ! (SFH: relative steps; times in Gyr)
                if (q >= 16) h = merge(H_DUST32, H_DUST, LINEAR_ONLY) * max(abs(x), 1.0_wp)
                ! (zred: the resampled spectrum is piecewise linear in z, with a kink at every pixel
                ! crossing, ~1e-4 apart in z for each pixel but dense over the spectrum: steps of 1e-9
                ! avoid them, and the differences' rounding is then ~1e-5 relative. FP32 builds check
                ! the spectrum part, smooth in z (the IGM))
                if (q == 22) h = merge(1.0e-4_wp, 1.0e-9_wp, LINEAR_ONLY) * (1.0_wp + x)
                if (bound) then
                    ! One-sided, third order: (-11 L0 + 18 L1 - 9 L2 + 2 L3) / 6h (at dust2 = 0 the UV
                    ! curve makes L strongly curved; second order needed h ~ 1e-6, below the
                    ! rounding floor of the smaller gradients)
                    call loss(pp, w, given_w, l0)
                    call set_par(pp(m), q, x + h)
                    call loss(pp, w, given_w, lp)
                    call set_par(pp(m), q, x + 2.0_wp * h)
                    call loss(pp, w, given_w, lm)
                    call set_par(pp(m), q, x + 3.0_wp * h)
                    call loss(pp, w, given_w, l3)
                    g_fd(q, m) = (-11.0_wp * l0 + 18.0_wp * lp - 9.0_wp * lm + 2.0_wp * l3) / (6.0_wp * h)
                else
                    call set_par(pp(m), q, x + h)
                    call loss(pp, w, given_w, lp)
                    call set_par(pp(m), q, x - h)
                    call loss(pp, w, given_w, lm)
                    g_fd(q, m) = (lp - lm) / (2.0_wp * h)
                end if
            end do
        end do
        ! Relative to the larger of |g_fd|, 1e-3 of the largest gradient of that model, and an
        ! absolute floor: the differences' rounding (L ~ 1-10, rounding ~1e-13 over h = 1e-5) is up to
        ! ~1e-7 absolute, so gradients below ~0.1 are checked to 1e-6 absolute (rows that do not act,
        ! mwr and uvb outside dust type 1, gas_logz and gas_logu without nebular emission, are zero)
        worst = 0.0_wp
        worst_m = 0
        do m = 1, size(psets)
            gmax = maxval(abs(g_fd(:, m)))
            do q = 1, NQ
                x = abs(pbar(ROWS(q), m) - g_fd(q, m)) / max(abs(g_fd(q, m)), &
                                                               merge(1.0e-2_wp, 1.0e-3_wp, LINEAR_ONLY) * gmax, &
                                                               merge(tiny(1.0_wp), 0.1_wp, LINEAR_ONLY))
                ! (zred: tolerance 1e-3 in FP64 and an absolute floor of 1e-3: its steps are 1e-9, the
                ! differences' rounding is ~1e-4 absolute, and over the spectrum's ~6000 pixels a pixel
                ! crossing, a kink of the resampled spectrum, falls inside a step now and then; measured
                ! 1e-7 - 2e-4)
                if (q == 22 .and. .not. LINEAR_ONLY) x = 0.01_wp * abs(pbar(ROWS(q), m) - g_fd(q, m)) / &
                                                          max(abs(g_fd(q, m)), 1.0_wp)
                if (x > worst(q)) then
                    worst(q) = x
                    worst_m(q) = m
                end if
            end do
        end do
        do q = 1, NQ
            write(*, '(A,A12,A,ES10.2,A,I0,A,2ES14.6)') "# ", NAMES(q), ": worst relative difference ", worst(q), &
                " (model ", worst_m(q), "): ", pbar(ROWS(q), max(worst_m(q), 1)), g_fd(q, max(worst_m(q), 1))
        end do
        call assert_true(maxval(worst) <= merge(DUST_RTOL32, DUST_RTOL, LINEAR_ONLY), &
                         "d L / d(dust parameters) against finite differences", total_tests, total_failures)
    end subroutine check_dust

    real(WP) function get_par(p, q)
        type(PARAMS), intent(in) :: p
        integer, intent(in) :: q
        select case (q)
        case (1); get_par = p%dust1
        case (2); get_par = p%dust2
        case (3); get_par = p%dust3
        case (4); get_par = p%dust_index
        case (5); get_par = p%dust1_index
        case (6); get_par = p%frac_obrun
        case (7); get_par = p%frac_nodust
        case (8); get_par = p%mwr
        case (9); get_par = p%uvb
        case (10); get_par = p%duste_qpah
        case (11); get_par = p%duste_umin
        case (12); get_par = p%duste_gamma
        case (13); get_par = p%gas_logz
        case (14); get_par = p%gas_logu
        case (15); get_par = p%logzsol
        case (16); get_par = p%tage
        case (17); get_par = p%tau
        case (18); get_par = p%const
        case (19); get_par = p%fburst
        case (20); get_par = p%tburst
        case (21); get_par = p%sf_start
        case (22); get_par = p%zred
        case (23); get_par = p%sf_trunc
        case default; get_par = p%sf_slope
        end select
    end function get_par

    subroutine set_par(p, q, x)
        type(PARAMS), intent(inout) :: p
        integer, intent(in) :: q
        real(WP), intent(in) :: x
        select case (q)
        case (1); p%dust1 = x
        case (2); p%dust2 = x
        case (3); p%dust3 = x
        case (4); p%dust_index = x
        case (5); p%dust1_index = x
        case (6); p%frac_obrun = x
        case (7); p%frac_nodust = x
        case (8); p%mwr = x
        case (9); p%uvb = x
        case (10); p%duste_qpah = x
        case (11); p%duste_umin = x
        case (12); p%duste_gamma = x
        case (13); p%gas_logz = x
        case (14); p%gas_logu = x
        case (15); p%logzsol = x
        case (16); p%tage = x
        case (17); p%tau = x
        case (18); p%const = x
        case (19); p%fburst = x
        case (20); p%tburst = x
        case (21); p%sf_start = x
        case (22); p%zred = x
        case (23); p%sf_trunc = x
        case default; p%sf_slope = x
        end select
    end subroutine set_par

    !> The loss L of a batch (and, on request, its magnitudes and spectra).
    subroutine loss(psets, w, given_w, l, mags_out, spec_out)
        type(PARAMS), intent(in) :: psets(:)
        real(WP), intent(in) :: w(:, :)
        logical, intent(in) :: given_w
        real(WP), intent(out) :: l
        real(WP), intent(out), optional :: mags_out(:, :), spec_out(:, :)
        real(WP), allocatable :: mags(:, :), spec(:, :)
        integer :: status

        allocate(mags(ctx%state%nbands, size(psets)), spec(ctx%state%nspec, size(psets)))
        if (given_w) then
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, ssp_weights=w)
        else
            call compute_csp_batch(ctx, psets, mags, status, spec=spec)
        end if
        if (status /= 0) then
            write(*, *) 'ERROR: compute_csp_batch returned status ', status
            stop 1
        end if
        if (present(mags_out)) mags_out = mags
        if (present(spec_out)) spec_out = spec
        l = loss_of(mags, spec)
    end subroutine loss

    !> L = sum(mags_bar * mags) + sum(spec_bar * spec) with the cotangents of the current check
    !> (zero before they are drawn); magnitudes not computed (NaN) have zero cotangent
    real(WP) function loss_of(mags, spec)
        real(WP), intent(in) :: mags(:, :), spec(:, :)
        loss_of = 0.0_wp
        if (allocated(lbar_m)) loss_of = sum(lbar_m * merge(mags, 0.0_wp, mags == mags)) + sum(lbar_s * spec)
    end function loss_of

    real(WP) function urand()
        seed = mod(1103515245 * seed + 12345, 2147483647)
        if (seed < 0) seed = seed + 2147483647
        urand = real(seed, WP) / 2147483647.0_wp
    end function urand

    !> Gradients with stages whose adjoints are not written: status 2.
    subroutine check_unsupported()
        type(PARAMS) :: psets(1)
        real(WP), allocatable :: mags(:, :), wbar(:, :)
        integer :: status

        call print_group("Gradients: unsupported stages")
        allocate(mags(ctx%state%nbands, 1), wbar(ctx%state%ntfull, 1))
        call set_model(1, 0.0_wp)
        ! Lines outside the spectrum with dust emission (their absorbed part joins the energy balance)
        call set_int('add_neb_emission', 1)
        call set_int('add_dust_emission', 1)
        call set_int('nebemlineinspec', 0)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, psets(1))
        call compute_csp_batch(ctx, psets, mags, status, mags_bar=mags, weights_bar=wbar)
        call assert_true(status == 2, "nebular lines outside the spectrum, dust emission: status 2", total_tests, &
                         total_failures)
        call set_int('nebemlineinspec', 1)
        call set_int('add_dust_emission', 0)
        call set_int('add_neb_emission', 0)
        ! Smoothing at constant width in wavelength (on the host): not yet
        call set_int('smooth_velocity', 0)
        call set_float('sigma_smooth', 5.0_wp)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, psets(1))
        call compute_csp_batch(ctx, psets, mags, status, mags_bar=mags, weights_bar=wbar)
        call assert_true(status == 2, "smoothing on the host: status 2", total_tests, total_failures)
        call set_int('smooth_velocity', 1)
        call set_float('sigma_smooth', 0.0_wp)
    end subroutine check_unsupported

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer :: status

        call opensps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: status

        call opensps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_float

end program test_csp_batch_grad
