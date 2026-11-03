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
    !> - the loss includes the mass and lbol outputs (cotangents mass_bar, lbol_bar);
    !> - SSP-level nebular emission (add_neb_emission = 2, in the basis);
    !> - with the IMF free per model (ssp_ops), d L / d(imf1, imf2, imf3) and every other row through
    !>   the operator's reverse pass;
    !> - every smoothing mode (the device operator, the host modes);
    !> - weights over ages and metallicities (ssp_weights_z): d L / d(weights) along a random direction
    !>   and for single entries, and the dust, AGN and redshift rows;
    !> - with the IMF and the metallicity free per model (an operator per metallicity), the IMF rows,
    !>   logzsol and the mass output through the operator's reverse pass;
    !> - agn_tau and igm_factor;
    !> - van Dokkum's vdmc and Dave's mdave (and slopes) with those IMF types free.
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
                                 BGRAD_SF_SLOPE, BGRAD_IMF1, BGRAD_IMF2, BGRAD_IMF3, BGRAD_AGN_TAU, &
                                 BGRAD_IGM_FACTOR, BGRAD_VDMC, BGRAD_MDAVE, NBGRAD
    use opensps_ssp, only: generate_ssp_grid
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_build
    use opensps_ssp_batch, only: ssp_opset_t, ssp_opset_build, ssp_opset_free
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
    !> Absolute floor of the fagn checks' relative differences: the loss's rounding over a step of 1e-5
    !> is ~1e-8 absolute (measured: smaller steps make it larger), against fagn gradients down to ~1e-3
    real(WP), parameter :: FAGN_FLOOR = 1.0e-2_wp
    !> Dust parameters (nonlinear): steps and tolerances of their finite differences
    !> FP32: the loss's spectrum part is not linear in them, so FP32 builds check interior points
    !> only (central differences at h = 1e-2: the offload builds' host fallback sums the forward in
    !> plain FP32 trees, whose rounding dominated at 1e-3), relative to the larger of |g_fd| and 1e-2
    !> of the model's largest gradient; the bounds are validated in FP64
    real(WP), parameter :: H_DUST = 1.0e-5_wp, H_DUST32 = 1.0e-2_wp
    !> (FP64: 3e-5, the central differences' rounding being ~1e-6 absolute against the floor of 0.1)
    real(WP), parameter :: DUST_RTOL = 3.0e-5_wp, DUST_RTOL32 = 2.0e-2_wp
    !> (FP64, rows one-sided at their bound 0: see check_dust)
    real(WP), parameter :: DUST_RTOL_BOUND = 3.0e-4_wp
    !> Per-model redshifts, off round values: at z = 0.5 (or 2, 3) many observed pixels fall exactly on
    !> rest-frame pixels, where the resampled spectrum has a kink in z (the gradient there is
    !> one-sided, central differences average the two sides)
    real(WP), parameter :: ZLIST(N_MOD) = [0.0_wp, 0.113_wp, 0.507_wp, 1.213_wp, 2.017_wp, 3.011_wp]
    type(opensps_context_t), allocatable :: ctx
    integer :: seed = 12345
    logical :: zred_per_model = .false.
    real(WP) :: smooth_sigma = 0.0_wp   ! sigma_smooth of every model
    !> The IMF free per model: the batch takes the operator set opset (cb)
    logical :: use_ops = .false.
    type(ssp_opset_t) :: opset
    !> The cotangents of the current check (the loss L is formed with them)
    real(WP), allocatable :: lbar_m(:, :), lbar_s(:, :)
    !> ... and of the mass and lbol outputs; the last call's mass and lbol (cb)
    real(WP), allocatable :: lbar_mass(:), lbar_lbol(:)
    real(WP) :: last_mass(N_MOD), last_lbol(N_MOD)
    !> Weights over ages and metallicities: the batch takes wzg (ssp_weights_z) in place of the weight
    !> matrix, and returns d L / d wzg in wzg_bar (cb)
    logical :: use_wz = .false.
    real(WP), allocatable :: wzg(:, :, :), wzg_bar(:, :, :)

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
    ! The host smoothing modes: the exact log-linear velocity convolution, constant width in wavelength
    call set_int('smoothspec_fast', 0)
    smooth_sigma = 150.0_wp
    call check_grad(2, 0.513_wp, 1, 1, .true., "Calzetti, z = 0.513, IGM, dust emission, AGN, exact velocity smoothing")
    call set_int('smoothspec_fast', 1)
    call set_int('smooth_velocity', 0)
    ! (5.37 A, not 5: at 5 A the windows' edges, lambda +- 20 A, fall on pixels of the 5 A-spaced parts of
    ! the grid, where the smoothed spectrum jumps in sigma)
    smooth_sigma = 5.37_wp
    call check_grad(1, 0.0_wp, 0, 0, .false., "Cardelli et al., z = 0, dust emission, smoothing at 5.37 A, SFH")
    call set_int('smooth_velocity', 1)
    smooth_sigma = 0.0_wp
    call set_int('add_dust_emission', 0)
    ! Nebular emission (continuum and lines in the spectrum)
    call set_int('add_neb_emission', 1)
    call check_grad(2, 0.0_wp, 0, 0, .true., "Calzetti, z = 0, nebular, weight matrix")
    call set_int('add_dust_emission', 1)
    call check_grad(1, 0.513_wp, 1, 1, .false., "Cardelli et al., z = 0.513, IGM, nebular, dust emission, AGN, SFH parameters")
    call set_int('add_dust_emission', 0)
    call set_int('add_neb_emission', 0)
    ! Nebular lines outside the spectrum with dust emission: their absorbed part joins the energy balance
    call set_int('add_neb_emission', 1)
    call set_int('add_dust_emission', 1)
    call set_int('nebemlineinspec', 0)
    call check_grad(4, 0.513_wp, 1, 1, .false., "Kriek & Conroy, z = 0.513, IGM, nebular lines outside the spectrum, " // &
                    "dust emission, AGN, SFH")
    call check_grad(1, 0.0_wp, 0, 0, .true., "Cardelli et al., z = 0, nebular lines outside the spectrum, dust emission, " // &
                    "weight matrix")
    call set_int('nebemlineinspec', 1)
    call set_int('add_dust_emission', 0)
    call set_int('add_neb_emission', 0)
    ! SSP-level nebular emission (in the basis: the batch adds none; gas_logz, gas_logu do not act)
    call set_int('add_neb_emission', 2)
    call build_basis()
    call set_int('add_dust_emission', 1)
    call check_grad(2, 0.513_wp, 1, 1, .false., "Calzetti, z = 0.513, SSP-level nebular, dust emission, AGN, SFH")
    call set_int('add_dust_emission', 0)
    call set_int('add_neb_emission', 0)
    call build_basis()
    ! The IMF free per model (Kroupa slopes through the SSP operator), with full physics
    call set_int('imf_type', 2)
    call build_opset()
    use_ops = .true.
    call set_int('add_neb_emission', 1)
    call set_int('add_dust_emission', 1)
    call check_grad(2, 0.513_wp, 1, 1, .false., "Kroupa IMF free (ssp_ops), z = 0.513, IGM, nebular, dust emission, AGN, SFH")
    call check_grad(1, 0.0_wp, 0, 0, .true., "Kroupa IMF free (ssp_ops), z = 0, nebular, dust emission, weight matrix")
    call set_int('add_dust_emission', 0)
    call set_int('add_neb_emission', 0)
    call ssp_opset_free(opset)
    ! van Dokkum (vdmc) and Dave (imf1, imf2, mdave) through the operator: their rows against finite
    ! differences, and the rows each type does not read zero (the reverse pass skips their differences)
    call set_int('imf_type', 3)
    call build_opset()
    call check_grad(2, 0.0_wp, 0, 0, .true., "van Dokkum IMF free (ssp_ops), z = 0, weight matrix")
    call ssp_opset_free(opset)
    call set_int('imf_type', 4)
    call build_opset()
    call check_grad(1, 0.0_wp, 0, 0, .true., "Dave IMF free (ssp_ops), z = 0, weight matrix")
    use_ops = .false.
    call ssp_opset_free(opset)
    call set_int('imf_type', 1)
    ! Per-model metallicity: every metallicity's basis resident (the SSPs interpolated in log Z)
    call opensps_setup(ctx, 0, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call opensps_context_move_to_device(ctx)
    call set_int('zmet', 10)
    call set_int('imf_type', 1)
    call build_basis_all()
    call set_int('add_neb_emission', 1)
    call set_int('add_dust_emission', 1)
    call check_grad(2, 0.513_wp, 1, 1, .true., "Calzetti, z = 0.513, per-model logzsol, IGM, nebular, dust emission, AGN")
    call set_int('add_neb_emission', 0)
    ! Weights over ages and metallicities (an enrichment history per model), with dust emission
    call check_weights_z(4, 0.513_wp, "Kriek & Conroy, z = 0.513, IGM, dust emission, AGN")
    call set_int('add_dust_emission', 0)
    call check_weights_z(1, 0.0_wp, "Cardelli et al., z = 0, AGN")
    ! The IMF and the metallicity free per model: an operator per metallicity (Kroupa slopes, logzsol)
    call set_int('imf_type', 2)
    call build_opset_all()
    use_ops = .true.
    call set_int('add_neb_emission', 1)
    call set_int('add_dust_emission', 1)
    call check_grad(4, 0.513_wp, 1, 1, .false., "Kroupa IMF and logzsol free (ssp_ops), z = 0.513, IGM, nebular, " // &
                    "dust emission, AGN, SFH")
    call set_int('add_dust_emission', 0)
    call set_int('add_neb_emission', 0)
    use_ops = .false.
    call ssp_opset_free(opset)
    call set_int('imf_type', 1)

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
        ctx%state%ssp_basis_is_dirty = .true.
        call opensps_context_update_ssp_basis(ctx, reshape(spec_ssp, [size(spec_ssp, 1), size(spec_ssp, 2), 1]), &
                                              reshape(mass_ssp, [size(mass_ssp), 1]), &
                                              reshape(lbol_ssp, [size(lbol_ssp), 1]), 1)
    end subroutine build_basis

    !> The operator set of the context's metallicity (zmet 10) and IMF type, for the IMF free per model.
    subroutine build_opset()
        type(PARAMS) :: pset
        type(ssp_operator_t), allocatable :: ops(:)
        call set_model(1, 0.0_wp)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        allocate(ops(1))
        call ssp_operator_build(ctx, pset, ops(1))
        call ssp_opset_build(ops, opset)
    end subroutine build_opset

    !> An operator per metallicity of the context (the IMF and the metallicity free per model).
    subroutine build_opset_all()
        type(PARAMS) :: pset
        type(ssp_operator_t), allocatable :: ops(:)
        integer :: z
        call set_model(1, 0.0_wp)
        allocate(ops(ctx%state%nz))
        do z = 1, ctx%state%nz
            call set_int('zmet', z)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call opensps_context_set_pset(ctx, pset)
            call ssp_operator_build(ctx, pset, ops(z))
        end do
        call set_int('zmet', 10)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        call opensps_context_set_pset(ctx, pset)
        call ssp_opset_build(ops, opset)
    end subroutine build_opset_all

    !> compute_csp_batch, with the operator set when use_ops.
    subroutine cb(psets, mags, status, spec, ssp_weights, mags_bar, spec_bar, params_bar, weights_bar)
        type(PARAMS), intent(in) :: psets(:)
        real(WP), intent(out) :: mags(:, :)
        integer, intent(out) :: status
        real(WP), intent(out), optional :: spec(:, :), params_bar(:, :), weights_bar(:, :)
        real(WP), intent(in), optional :: ssp_weights(:, :), mags_bar(:, :), spec_bar(:, :)
        logical :: grad
        ! (the mass and lbol cotangents with the gradients, when drawn)
        grad = (present(params_bar) .or. present(weights_bar)) .and. allocated(lbar_mass)
        if (use_wz .and. grad) then
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, ssp_weights_z=wzg, mags_bar=mags_bar, &
                                   spec_bar=spec_bar, params_bar=params_bar, weights_z_bar=wzg_bar, &
                                   mass=last_mass, lbol=last_lbol, mass_bar=lbar_mass, lbol_bar=lbar_lbol)
        else if (use_wz) then
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, ssp_weights_z=wzg, mass=last_mass, lbol=last_lbol)
        else if (use_ops .and. grad) then
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, ssp_weights=ssp_weights, ssp_ops=opset, &
                                   mags_bar=mags_bar, spec_bar=spec_bar, params_bar=params_bar, weights_bar=weights_bar, &
                                   mass=last_mass, lbol=last_lbol, mass_bar=lbar_mass, lbol_bar=lbar_lbol)
        else if (use_ops) then
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, ssp_weights=ssp_weights, ssp_ops=opset, &
                                   mags_bar=mags_bar, spec_bar=spec_bar, params_bar=params_bar, weights_bar=weights_bar, &
                                   mass=last_mass, lbol=last_lbol)
        else if (grad) then
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, ssp_weights=ssp_weights, mags_bar=mags_bar, &
                                   spec_bar=spec_bar, params_bar=params_bar, weights_bar=weights_bar, &
                                   mass=last_mass, lbol=last_lbol, mass_bar=lbar_mass, lbol_bar=lbar_lbol)
        else
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, ssp_weights=ssp_weights, mags_bar=mags_bar, &
                                   spec_bar=spec_bar, params_bar=params_bar, weights_bar=weights_bar, &
                                   mass=last_mass, lbol=last_lbol)
        end if
    end subroutine cb

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
        ! (Kroupa slopes, used with the IMF free per model)
        call set_float('imf1', 1.1_wp + 0.07_wp * m)
        call set_float('imf2', 2.1_wp + 0.05_wp * m)
        call set_float('imf3', 2.2_wp + 0.09_wp * m)
        ! (van Dokkum's characteristic mass, Dave's break mass: used with those IMF types free)
        call set_float('vdmc', 0.25_wp + 0.05_wp * m)
        call set_float('mdave', 0.35_wp + 0.1_wp * m)
        ! (inside the torus grid's intervals, 5 10 20 30 40 60 80 100 150: 21.7 - 96.7)
        call set_float('agn_tau', 6.7_wp + 15.0_wp * mod(m, 8))
        ! (the IGM's factor: one value for the batch, as the common IGM table takes it; a finite
        ! difference of one model's factor runs the per-model IGM)
        call set_float('igm_factor', 1.3_wp)
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
        real(WP) :: lp, lm, l0, l3, g_fd, g_ad, h, worst, scale, gwa, gwf
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
        if (allocated(lbar_m)) deallocate(lbar_m, lbar_s, lbar_mass, lbar_lbol)
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
        ! (the mass and lbol outputs: O(1) cotangents; the mass's not with ssp_ops)
        allocate(lbar_mass(N_MOD), lbar_lbol(N_MOD))
        do m = 1, N_MOD
            lbar_mass(m) = 2.0_wp * urand() - 1.0_wp
            lbar_lbol(m) = 2.0_wp * urand() - 1.0_wp
        end do
        if (given_w) then
            call cb(psets, mags, status, spec=spec, ssp_weights=w, mags_bar=mbar, spec_bar=sbar, &
                                   params_bar=pbar, weights_bar=wbar)
        else
            call cb(psets, mags, status, spec=spec, mags_bar=mbar, spec_bar=sbar, &
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
            ! (FP32: the full loss, magnitudes and lbol, at h = 1e-2: 3e-3)
            call assert_true(abs(g_ad - g_fd) <= merge(3.0e-3_wp, GRAD_RTOL, LINEAR_ONLY) * abs(g_fd), &
                             "d L / d weights along a direction", &
                             total_tests, total_failures)
        else
            ! The gradient with respect to the weights the SFH parameters give equals that of the
            ! same weights given as a matrix
            call cb(psets, mags, status, ssp_weights=w, mags_bar=mbar, spec_bar=sbar, &
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
            lbar_mass = 0.0_wp
            lbar_lbol = 0.0_wp
            if (given_w) then
                call cb(psets, mags, status, ssp_weights=w, mags_bar=lbar_m, spec_bar=sbar, &
                                       params_bar=pbar, weights_bar=wbar)
            else
                call cb(psets, mags, status, mags_bar=lbar_m, spec_bar=sbar, &
                                       params_bar=pbar, weights_bar=wbar)
            end if
        end if

        ! d L / d fagn, model by model
        if (agn == 1) then
            worst = 0.0_wp; k0 = 0; gwa = 0.0_wp; gwf = 0.0_wp
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
                if (abs(g_ad - g_fd) / max(abs(g_fd), FAGN_FLOOR) > worst) then
                    worst = abs(g_ad - g_fd) / max(abs(g_fd), FAGN_FLOOR)
                    k0 = m
                    gwa = g_ad
                    gwf = g_fd
                end if
            end do
            write(*, '(A,ES10.2,A,I0,A,2ES16.8)') "# fagn: worst relative difference ", worst, " (model ", k0, "): ", gwa, gwf
            ! (FP64: the differences' rounding reaches ~1e-6 where an AGN dominates a band: 5e-6; the floor
            ! FAGN_FLOOR: the loss's rounding over the step, ~1e-8 absolute, against gradients down to ~1e-3)
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
            ! (An absolute floor of 1e-5. FP32: the spectrum part's FP32 rounding over the step is ~1e-7,
            ! and its sigma gradient can be ~1e-9. FP64: the window-edge jumps leave the differences
            ! ~1e-10 absolute; the random cotangents can make the batch's sigma gradient ~1e-6, as with
            ! LLVM flang's draws at 5.37 A: 2.4e-10 = 1e-4 relative; nvfortran 1.3e-9: the floor is 3e-5)
            call assert_true(abs(g_ad - g_fd) <= merge(2.0e-2_wp, 1.0e-4_wp, LINEAR_ONLY) * max(abs(g_fd), 3.0e-5_wp), &
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
                    ! (loops, not findloc: the nvfortran runtime has no findloc on logical arrays)
                    k0 = 0
                    do k = 1, size(w, 1)
                        if (w(k, m) > 0.0_wp) then
                            k0 = k
                            exit
                        end if
                    end do
                case (3)
                    k0 = 0
                    do k = size(w, 1), 1, -1
                        if (.not. (w(k, m) > 0.0_wp)) then
                            k0 = k
                            exit
                        end if
                    end do
                end select
                if (k0 < 1) cycle
                ! (relative to the age's own weight: a young age that dominates a UV band makes its
                ! magnitude strongly curved in that weight)
                if (w(k0, m) > 0.0_wp .and. .not. LINEAR_ONLY) then
                    h = 1.0e-5_wp * w(k0, m)
                else
                    ! (FP32: the check is linear, so any step is exact; relative to the model's largest
                    ! weight, L changes well above the FP32 rounding of the forward's sums, which a step of
                    ! a young age's own weight did not: 1.65e-2 in the offload builds' host fallback)
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
                ! (An absolute floor, a fraction of the model's largest weight gradient: the differences'
                ! rounding scales with L / h, not with the age's own gradient. FP32 1e-4: an old age with zero
                ! weight can have a gradient 1e-7 of the largest, below the FP32 rounding of L over any step.
                ! FP64 1e-6: a small age's step is relative to its own weight; 1.3e-4 relative in the -O0
                ! coverage build)
                worst = max(worst, abs(wbar(k0, m) - g_fd) / max(abs(g_fd), &
                                                               merge(1.0e-4_wp, 1.0e-6_wp, LINEAR_ONLY) * &
                                                               maxval(abs(wbar(:, m)))))
            end do
            write(*, '(A,ES10.2)') "# single ages: worst relative difference ", worst
            ! (FP64: the step is relative to the age's own weight, so for the smallest weights the
            ! differences' rounding reaches ~1e-5: 5e-5. FP32: L changes by one age's share, ~1e-4 of L
            ! for a young age, whose FP32 rounding, ~1e-7 of L, is ~1e-3 of the difference: 1e-2)
            call assert_true(worst <= merge(1.0e-2_wp, 5.0e-5_wp, LINEAR_ONLY), "d L / d weight, single ages", &
                             total_tests, total_failures)
        end if
    end subroutine check_grad

    !> Weights over ages and metallicities (ssp_weights_z): each model's SFH weights spread over an
    !> enrichment history (the metallicity rising from the lowest to near solar over its ages, each age
    !> split between the two bracketing metallicities), against finite differences of L: along a random
    !> direction (relative perturbations), single entries (the largest, and one with zero weight,
    !> one-sided), and the parameter rows (dust, AGN, zred; the SFH and metallicity rows, not used with
    !> given weights, are zero).
    subroutine check_weights_z(dust_type, zred, label)
        integer, intent(in) :: dust_type
        real(WP), intent(in) :: zred
        character(len=*), intent(in) :: label
        type(PARAMS) :: psets(N_MOD)
        real(WP), allocatable :: mags(:, :), spec(:, :), mbar(:, :), sbar(:, :), pbar(:, :), w(:, :), w0(:, :, :), &
                                 dir(:, :, :), t1(:), t2(:)
        real(WP) :: lp, lm, l0, g_fd, g_ad, h, worst, scale, x
        integer :: m, b, i, k, z, nspec, nbands, nt, nz, status, loc(3)

        call print_group("Gradients: ssp_weights_z, " // label)
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        nt = ctx%state%ntfull
        nz = size(ctx%state%ssp_basis_spec, 3)
        allocate(mags(nbands, N_MOD), spec(nspec, N_MOD), mbar(nbands, N_MOD), sbar(nspec, N_MOD), &
                 pbar(NBGRAD, N_MOD), w(nt, N_MOD), t1(nt), t2(nt))
        if (allocated(wzg)) deallocate(wzg, wzg_bar)
        allocate(wzg(nt, nz, N_MOD), wzg_bar(nt, nz, N_MOD), w0(nt, nz, N_MOD), dir(nt, nz, N_MOD))
        call set_int('dust_type', dust_type)
        call set_int('add_igm_absorption', merge(1, 0, zred > 0.0_wp))
        call set_int('add_agn_dust', 1)
        wzg = 0.0_wp
        do m = 1, N_MOD
            call set_model(m, zred)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, psets(m))
            call sfh_weights_core(ctx, psets(m), psets(m)%tage, 1, w(:, m:m), t1, t2)
            ! (metallicity index rising from 1 at the oldest age to nz - 1 - mod(m, 3) at the youngest)
            do k = 1, nt
                x = 1.0_wp + real(nz - 2 - mod(m, 3), WP) * real(nt - k, WP) / real(nt - 1, WP)
                z = min(int(x), nz - 1)
                wzg(k, z, m) = w(k, m) * (real(z + 1, WP) - x)
                wzg(k, z + 1, m) = w(k, m) * (x - real(z, WP))
            end do
        end do
        use_wz = .true.
        if (allocated(lbar_m)) deallocate(lbar_m, lbar_s, lbar_mass, lbar_lbol)
        call loss(psets, w, .true., lp, mags, spec)
        do m = 1, N_MOD
            do b = 1, nbands
                mbar(b, m) = 0.0_wp
                if (mags(b, m) == mags(b, m)) mbar(b, m) = 2.0_wp * urand() - 1.0_wp
                if (mags(b, m) > minval(mags(:, m), mask=mags(:, m) == mags(:, m)) + MAG_RANGE) mbar(b, m) = 0.0_wp
            end do
            scale = maxval(abs(spec(:, m)))
            do i = 1, nspec
                sbar(i, m) = (2.0_wp * urand() - 1.0_wp) * 10.0_wp / (scale * nspec)
            end do
        end do
        allocate(lbar_mass(N_MOD), lbar_lbol(N_MOD))
        do m = 1, N_MOD
            lbar_mass(m) = 2.0_wp * urand() - 1.0_wp
            lbar_lbol(m) = 2.0_wp * urand() - 1.0_wp
        end do
        ! (FP32: the spectrum part of the loss alone, linear in the weights)
        if (LINEAR_ONLY) then
            mbar = 0.0_wp
            lbar_mass = 0.0_wp
            lbar_lbol = 0.0_wp
        end if
        lbar_m = mbar
        lbar_s = sbar
        call cb(psets, mags, status, spec=spec, mags_bar=mbar, spec_bar=sbar, params_bar=pbar)
        call assert_true(status == 0, "status 0 with gradients (ssp_weights_z)", total_tests, total_failures)
        if (status /= 0) then
            use_wz = .false.
            return
        end if

        ! A random direction (relative perturbations, so the weights stay positive)
        w0 = wzg
        do m = 1, N_MOD
            do z = 1, nz
                do k = 1, nt
                    dir(k, z, m) = w0(k, z, m) * (2.0_wp * urand() - 1.0_wp)
                end do
            end do
        end do
        h = merge(H_LIN, H_REL, LINEAR_ONLY)
        wzg = w0 + h * dir
        call loss(psets, w, .true., lp)
        wzg = w0 - h * dir
        call loss(psets, w, .true., lm)
        wzg = w0
        g_fd = (lp - lm) / (2.0_wp * h)
        g_ad = sum(wzg_bar * dir)
        write(*, '(A,2ES16.8,A,ES10.2)') "# weights over ages and metallicities, random direction (adjoint, " // &
            "differences): ", g_ad, g_fd, ", relative ", abs(g_ad - g_fd) / abs(g_fd)
        call assert_true(abs(g_ad - g_fd) <= GRAD_RTOL * abs(g_fd), "d L / d ssp_weights_z along a direction", &
                         total_tests, total_failures)

        ! Single entries of model 2: the largest weight, and one with zero weight (one-sided, second
        ! order: weights are masses, >= 0) at a metallicity the model does not use
        worst = 0.0_wp
        m = 2
        do i = 1, 2
            if (i == 1) then
                loc = maxloc(w0(:, :, m:m))
            else
                loc = [maxloc(w(:, m), 1), nz, 1]
            end if
            k = loc(1)
            z = loc(2)
            h = merge(H_LIN, 1.0e-5_wp, LINEAR_ONLY) * maxval(w0(:, :, m))
            if (w0(k, z, m) > 0.0_wp) h = merge(H_LIN, 1.0e-5_wp, LINEAR_ONLY) * w0(k, z, m)
            wzg(k, z, m) = w0(k, z, m) + h
            call loss(psets, w, .true., lp)
            if (w0(k, z, m) > h) then
                wzg(k, z, m) = w0(k, z, m) - h
                call loss(psets, w, .true., lm)
                g_fd = (lp - lm) / (2.0_wp * h)
            else
                wzg(k, z, m) = w0(k, z, m)
                call loss(psets, w, .true., l0)
                wzg(k, z, m) = w0(k, z, m) + 2.0_wp * h
                call loss(psets, w, .true., lm)
                g_fd = (-3.0_wp * l0 + 4.0_wp * lp - lm) / (2.0_wp * h)
            end if
            wzg(k, z, m) = w0(k, z, m)
            worst = max(worst, abs(wzg_bar(k, z, m) - g_fd) / max(abs(g_fd), tiny(1.0_wp)))
        end do
        write(*, '(A,ES10.2)') "# single entries: worst relative difference ", worst
        call assert_true(worst <= merge(1.0e-2_wp, 5.0e-5_wp, LINEAR_ONLY), "d L / d ssp_weights_z, single entries", &
                         total_tests, total_failures)

        ! fagn and the dust, redshift and remaining rows
        worst = 0.0_wp
        do m = 1, N_MOD
            if (psets(m)%fagn <= 0.0_wp) cycle
            block
                type(PARAMS) :: pp(N_MOD)
                pp = psets
                h = merge(H_LIN, H_REL, LINEAR_ONLY) * psets(m)%fagn
                pp(m)%fagn = psets(m)%fagn + h
                call loss(pp, w, .true., lp)
                pp(m)%fagn = psets(m)%fagn - h
                call loss(pp, w, .true., lm)
            end block
            g_fd = (lp - lm) / (2.0_wp * h)
            worst = max(worst, abs(pbar(BGRAD_FAGN, m) - g_fd) / max(abs(g_fd), FAGN_FLOOR))
        end do
        write(*, '(A,ES10.2)') "# fagn: worst relative difference ", worst
        call assert_true(worst <= merge(GRAD_RTOL, 5.0e-6_wp, LINEAR_ONLY), "d L / d fagn (ssp_weights_z)", &
                         total_tests, total_failures)
        call check_dust(psets, w, .true., pbar)
        use_wz = .false.
    end subroutine check_weights_z

    !> d L / d(dust parameter) of every model against finite differences of L.
    subroutine check_dust(psets, w, given_w, pbar)
        type(PARAMS), intent(in) :: psets(:)
        real(WP), intent(in) :: w(:, :), pbar(:, :)
        logical, intent(in) :: given_w
        integer, parameter :: NQ = 31
        integer, parameter :: ROWS(NQ) = [BGRAD_DUST1, BGRAD_DUST2, BGRAD_DUST3, BGRAD_DUST_INDEX, BGRAD_DUST1_INDEX, &
                                          BGRAD_FRAC_OBRUN, BGRAD_FRAC_NODUST, BGRAD_MWR, BGRAD_UVB, BGRAD_DUSTE_QPAH, &
                                          BGRAD_DUSTE_UMIN, BGRAD_DUSTE_GAMMA, BGRAD_GAS_LOGZ, BGRAD_GAS_LOGU, &
                                          BGRAD_LOGZSOL, BGRAD_TAGE, BGRAD_TAU, BGRAD_CONST, BGRAD_FBURST, BGRAD_TBURST, &
                                          BGRAD_SF_START, BGRAD_ZRED, BGRAD_SF_TRUNC, BGRAD_SF_SLOPE, BGRAD_IMF1, &
                                          BGRAD_IMF2, BGRAD_IMF3, BGRAD_AGN_TAU, BGRAD_IGM_FACTOR, &
                                          BGRAD_VDMC, BGRAD_MDAVE]
        character(len=12), parameter :: NAMES(NQ) = [character(len=12) :: 'dust1', 'dust2', 'dust3', 'dust_index', &
                                                     'dust1_index', 'frac_obrun', 'frac_nodust', 'mwr', 'uvb', &
                                                     'duste_qpah', 'duste_umin', 'duste_gamma', 'gas_logz', 'gas_logu', &
                                                     'logzsol', 'tage', 'tau', 'const', 'fburst', 'tburst', 'sf_start', &
                                                     'zred', 'sf_trunc', 'sf_slope', 'imf1', 'imf2', 'imf3', 'agn_tau', &
                                                     'igm_factor', 'vdmc', 'mdave']
        type(PARAMS) :: pp(size(psets))
        real(WP) :: g_fd(NQ, size(psets)), lp, lm, l0, l3, x, h, worst(NQ), gmax
        integer :: q, m, worst_m(NQ)
        logical :: bound, bnd(NQ, size(psets))

        do q = 1, NQ
            do m = 1, size(psets)
                pp = psets
                x = get_par(pp(m), q)
                ! Parameters at their bound 0 (masses of dust, fractions): one-sided, second order
                bound = q /= 4 .and. q /= 5 .and. q /= 8 .and. q /= 9 .and. q /= 15 .and. q /= 24 .and. x <= 0.0_wp
                bnd(q, m) = bound
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
                ! (gas_logz, gas_logu: smooth in the CLOUDY grids' log space; 1e-4 keeps the rounding of
                ! their small gradients below the floor's 3e-6, truncation ~1e-8)
                if ((q == 13 .or. q == 14) .and. .not. LINEAR_ONLY) h = 1.0e-4_wp
                ! (SFH: relative steps; times in Gyr)
                if (q >= 16) h = merge(H_DUST32, H_DUST, LINEAR_ONLY) * max(abs(x), 1.0_wp)
                ! (zred: the resampled spectrum is piecewise linear in z, with a kink at every pixel
                ! crossing, ~1e-4 apart in z for each pixel but dense over the spectrum: steps of 1e-9
                ! avoid them, and the differences' rounding is then ~1e-5 relative. FP32 builds check
                ! the spectrum part, smooth in z (the IGM))
                if (q == 22) h = merge(1.0e-4_wp, 1.0e-9_wp, LINEAR_ONLY) * (1.0_wp + x)
                ! (vdmc: van Dokkum's break from the lognormal to the power law, at a mass proportional to vdmc,
                ! crosses isochrone mass points ~1e-5 apart in relative terms, each a kink of L; steps of
                ! 1e-6 relative resolve the local derivative. Measured: central differences -0.2770633 at
                ! 1e-6, -0.2770529 at 1e-5 and 1e-4, -0.2770666 at 1e-3, against the adjoint's -0.2770632)
                if (q == 30 .and. .not. LINEAR_ONLY) h = 1.0e-6_wp * max(abs(x), 1.0e-2_wp)
                if (bound) then
                    ! (h / 10: at dust2 or dust3 = 0 the curvature of the UV screen leaves the third-order
                    ! formula a truncation of up to ~2e-4 at h = 1e-5 (as h^3); the rounding floor is below)
                    h = h / 10.0_wp
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
                ! (One-sided at a bound 0, FP64: tolerance DUST_RTOL_BOUND. The attenuation curves extrapolated
                ! below ~100 A (Calzetti's polynomial, power laws) reach 1e5 - 1e7 per unit dust, so L, through
                ! the spectrum there and the energy balance, varies on dust2 and dust3 scales of ~1e-6 near 0.
                ! Measured with Kriek & Conroy, dust3 = 0: the third-order differences converge to the
                ! adjoint, 8.1074e-3 at h = 1e-6, 8.1241e-3 at 1e-7, against 8.1279e-3, until rounding takes
                ! over below 1e-8)
                if (bnd(q, m) .and. .not. LINEAR_ONLY) x = x * DUST_RTOL / DUST_RTOL_BOUND
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
        case (24); get_par = p%sf_slope
        case (25); get_par = p%imf1
        case (26); get_par = p%imf2
        case (27); get_par = p%imf3
        case (28); get_par = p%agn_tau
        case (29); get_par = p%igm_factor
        case (30); get_par = p%vdmc
        case default; get_par = p%mdave
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
        case (24); p%sf_slope = x
        case (25); p%imf1 = x
        case (26); p%imf2 = x
        case (27); p%imf3 = x
        case (28); p%agn_tau = x
        case (29); p%igm_factor = x
        case (30); p%vdmc = x
        case default; p%mdave = x
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
            call cb(psets, mags, status, spec=spec, ssp_weights=w)
        else
            call cb(psets, mags, status, spec=spec)
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
        if (allocated(lbar_m)) loss_of = sum(lbar_m * merge(mags, 0.0_wp, mags == mags)) + sum(lbar_s * spec) + &
                                         sum(lbar_mass * last_mass(1:size(lbar_mass))) + &
                                         sum(lbar_lbol * last_lbol(1:size(lbar_lbol)))
    end function loss_of

    real(WP) function urand()
        seed = mod(1103515245 * seed + 12345, 2147483647)
        if (seed < 0) seed = seed + 2147483647
        urand = real(seed, WP) / 2147483647.0_wp
    end function urand

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
