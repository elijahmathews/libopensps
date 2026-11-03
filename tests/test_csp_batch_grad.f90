program test_csp_batch_grad
    !> Gradients of the batched CSP (compute_csp_batch with mags_bar / spec_bar and params_bar /
    !> weights_bar) against finite differences of the loss they differentiate,
    !>     L = sum(mags_bar * mags) + sum(spec_bar * spec),
    !> for the stages whose adjoints are written (observed frame and band integrals, AGN, IGM,
    !> attenuation, age sums):
    !> - d L / d fagn per model (central differences; one-sided at fagn = 0, where the AGN stage of
    !>   the forward model switches on);
    !> - d L / d(SSP weights) along a random direction (relative perturbations of the weights, so
    !>   they stay positive), and for single ages, including an age whose weight is zero;
    !> - the weights' gradient of a parametric SFH equals that of the same weights given as a matrix;
    !> - unsupported stages return status 2.
    !> The spectra are linear in the weights and in fagn, so the differences are exact up to rounding
    !> and the curvature of the magnitudes (log10 of the band fluxes). FP64 builds are the
    !> validation (roadmap priority 4); FP32 builds check the same with differences sized for FP32.
    !> Requires the data tree, like the regression tests.

    use opensps_precision, only: WP, KP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup
    use opensps_csp, only: sfh_weights_core
    use opensps_csp_batch, only: compute_csp_batch, BGRAD_FAGN, NBGRAD
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
    real(WP), parameter :: GRAD_RTOL = merge(1.0e-6_wp, 2.0e-2_wp, KP == WP)
    real(WP), parameter :: ZLIST(N_MOD) = [0.0_wp, 0.1_wp, 0.5_wp, 1.2_wp, 2.0_wp, 3.0_wp]
    type(opensps_context_t), allocatable :: ctx
    integer :: seed = 12345
    logical :: zred_per_model = .false.
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
    call check_grad(0, 0.5_wp, 1, 1, .false., "power law, z = 0.5, IGM, AGN, SFH parameters")
    call check_grad(1, 0.0_wp, 0, 0, .true., "Cardelli et al., z = 0, weight matrix")
    zred_per_model = .true.
    call check_grad(4, 0.0_wp, 1, 1, .true., "Kriek & Conroy, per-model zred 0-3, IGM, AGN, weight matrix")
    zred_per_model = .false.
    call check_unsupported()

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

    !> Model m: SFH, dust and AGN parameters over their usual ranges (fagn = 0 for model 6).
    subroutine set_model(m, zred)
        integer, intent(in) :: m
        real(WP), intent(in) :: zred
        integer, parameter :: sfh(N_MOD) = [1, 4, 5, 1, 4, 1]
        real(WP), parameter :: tage(N_MOD) = [10.0_wp, 3.0_wp, 7.0_wp, 0.3_wp, 8.0_wp, 13.0_wp]
        real(WP), parameter :: tau(N_MOD) = [1.0_wp, 0.5_wp, 2.0_wp, 0.1_wp, 3.0_wp, 0.7_wp]
        real(WP), parameter :: dust1(N_MOD) = [0.0_wp, 0.6_wp, 1.0_wp, 1.5_wp, 0.0_wp, 0.4_wp]
        real(WP), parameter :: dust2(N_MOD) = [0.3_wp, 0.0_wp, 0.5_wp, 2.0_wp, 0.1_wp, 0.2_wp]
        real(WP), parameter :: dindex(N_MOD) = [-0.7_wp, -0.3_wp, -1.2_wp, -0.7_wp, 0.2_wp, -0.4_wp]
        real(WP), parameter :: fagn(N_MOD) = [0.05_wp, 0.2_wp, 0.01_wp, 0.5_wp, 0.1_wp, 0.0_wp]

        call set_int('sfh', sfh(m))
        call set_float('tage', tage(m))
        call set_float('tau', tau(m))
        call set_float('sf_trunc', merge(0.7_wp * tage(m), 0.0_wp, sfh(m) == 5))
        call set_float('sf_slope', merge(-1.0_wp, 0.0_wp, sfh(m) == 5))
        call set_float('dust1', dust1(m))
        call set_float('dust2', dust2(m))
        call set_float('dust_index', dindex(m))
        call set_float('dust_tesc', merge(7.5_wp, 7.0_wp, mod(m, 2) == 0))
        call set_float('frac_obrun', merge(0.2_wp, 0.0_wp, mod(m, 3) == 1))
        call set_float('frac_nodust', merge(0.1_wp, 0.0_wp, mod(m, 4) == 2))
        call set_float('dust3', merge(0.3_wp, 0.0_wp, m == 3))
        call set_float('mwr', merge(4.0_wp, 3.1_wp, mod(m, 2) == 1))
        call set_float('uvb', merge(0.5_wp, 1.0_wp, mod(m, 3) == 2))
        call set_float('fagn', fagn(m))
        call set_float('agn_tau', 5.0_wp + 15.0_wp * mod(m, 8))
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
        real(WP) :: lp, lm, l0, g_fd, g_ad, h, worst, scale
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

        ! d L / d fagn, model by model
        if (agn == 1) then
            worst = 0.0_wp
            do m = 1, N_MOD
                pp = psets
                if (psets(m)%fagn > 0.0_wp) then
                    h = H_REL * psets(m)%fagn
                    pp(m)%fagn = psets(m)%fagn + h
                    call loss(pp, w, given_w, lp)
                    pp(m)%fagn = psets(m)%fagn - h
                    call loss(pp, w, given_w, lm)
                    g_fd = (lp - lm) / (2.0_wp * h)
                else
                    ! One-sided at fagn = 0, second order: (-3 L(0) + 4 L(h) - L(2h)) / 2h (where the
                    ! AGN dominates a band the magnitudes' curvature is large)
                    h = merge(1.0e-8_wp, 1.0e-3_wp, KP == WP)
                    call loss(pp, w, given_w, l0)
                    pp(m)%fagn = h
                    call loss(pp, w, given_w, lp)
                    pp(m)%fagn = 2.0_wp * h
                    call loss(pp, w, given_w, lm)
                    g_fd = (-3.0_wp * l0 + 4.0_wp * lp - lm) / (2.0_wp * h)
                end if
                g_ad = pbar(BGRAD_FAGN, m)
                worst = max(worst, abs(g_ad - g_fd) / max(abs(g_fd), 1.0e-3_wp))
            end do
            write(*, '(A,ES10.2)') "# fagn: worst relative difference ", worst
            call assert_true(worst <= GRAD_RTOL, "d L / d fagn against finite differences", total_tests, total_failures)
        end if

        ! The weights: along a random direction (relative perturbations), and single ages
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

            ! Single ages of model 2: its oldest weighted age, its youngest, and an age with zero
            ! weight (one-sided: weights are masses, >= 0)
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
                h = merge(1.0e-5_wp, H_REL, KP == WP) * maxval(w(:, m))
                wp_ = w
                wp_(k0, m) = w(k0, m) + h
                call loss(psets, wp_, .true., lp)
                if (w(k0, m) > h) then
                    wm = w
                    wm(k0, m) = w(k0, m) - h
                    call loss(psets, wm, .true., lm)
                    g_fd = (lp - lm) / (2.0_wp * h)
                else
                    ! One-sided, second order (weights are masses, >= 0)
                    call loss(psets, w, .true., l0)
                    wm = w
                    wm(k0, m) = w(k0, m) + 2.0_wp * h
                    call loss(psets, wm, .true., lm)
                    g_fd = (-3.0_wp * l0 + 4.0_wp * lp - lm) / (2.0_wp * h)
                end if
                worst = max(worst, abs(wbar(k0, m) - g_fd) / max(abs(g_fd), tiny(1.0_wp)))
            end do
            write(*, '(A,ES10.2)') "# single ages: worst relative difference ", worst
            call assert_true(worst <= merge(1.0e-5_wp, GRAD_RTOL, KP == WP), "d L / d weight, single ages", &
                             total_tests, total_failures)
        else
            ! SFH parameters: the gradient with respect to the weights they give equals that of the
            ! same weights given as a matrix
            call compute_csp_batch(ctx, psets, mags, status, ssp_weights=w, mags_bar=mbar, spec_bar=sbar, &
                                   weights_bar=wbar2)
            worst = maxval(abs(wbar - wbar2)) / maxval(abs(wbar2))
            write(*, '(A,ES10.2)') "# SFH parameters vs weight matrix: ", worst
            call assert_true(status == 0 .and. worst <= merge(1.0e-12_wp, 1.0e-5_wp, KP == WP), &
                             "weights gradient: SFH parameters = weight matrix", total_tests, total_failures)
        end if
    end subroutine check_grad

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
        call set_int('add_dust_emission', 1)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, psets(1))
        call compute_csp_batch(ctx, psets, mags, status, mags_bar=mags, weights_bar=wbar)
        call assert_true(status == 2, "dust emission: status 2", total_tests, total_failures)
        call set_int('add_dust_emission', 0)
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
