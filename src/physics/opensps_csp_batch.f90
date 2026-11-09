module opensps_csp_batch
    !> @brief Batched CSP evaluation: many parameter sets against resident SSP bases.
    !>
    !> @details
    !> The fitting workload: N models that share the libraries and the resident SSP bases (one
    !> metallicity, or all of them for per-model logzsol), and differ in everything else that
    !> acts in the CSP: star formation history (parameters or a weight matrix), metallicity,
    !> redshift and IGM, dust, dust emission, nebular and AGN parameters. Per model the work is
    !> the same as `compute_csp_scenario` with `tage > 0`; here each stage runs once for the
    !> whole batch:
    !>
    !> 1. SFH weights per model in FP64 (on the device, one thread per model; on the host below
    !>    16 models), split into young and old ages by each model's `dust_tesc`, cast once to
    !>    kernel precision: W_young, W_old (ages x N).
    !> 2. Age sums: young = basis . W_young, old = basis . W_old (matrix products with K = number
    !>    of ages). With the SSP bases of all metallicities resident (opensps_context_update_ssp_basis
    !>    with nz bases), each model's basis is interpolated in log Z at its logzsol between the
    !>    two bracketing metallicities (as compute_interpolated_ssp: linear in the spectra, the
    !>    masses and log L_bol, not clamped); mass, L_bol and the nebular emission (Q, EUV) use the
    !>    interpolated SSPs. This mode uses the portable kernels (also in cuBLAS builds).
    !> 3. Nebular emission (add_neb_emission = 1), as apply_nebular_emission does to each young SSP
    !>    before the age sum, but summed over ages per model: the EUV rows of the age sums are
    !>    redone with the escaping fraction (frac_obrun) for the nebular ages; the nebular
    !>    continuum (Q(t) times the CLOUDY grid interpolated in gas_logz, gas_logu and age, in
    !>    FP64 log space) is summed with the SFH weights per (pixel, model); line luminosities per
    !>    (line, model), then their profiles added over each pixel's overlapping lines.
    !> 4. Attenuation, the birth-cloud and diffuse dust of `opensps_dust`, and the absorbed light.
    !> 5. Dust emission (energy balance): per-model reductions over pixels, then the
    !>    Draine & Li template, interpolated per model from the resident grid, is added.
    !> 6. IGM transmission (per model when the redshifts or igm_factor differ), then AGN dust
    !>    emission (fagn, agn_tau per model), attenuated by the model's diffuse dust, as legacy
    !>    FSPS agn_dust.
    !> 7. Velocity smoothing (sigma_smooth, shared by the batch; smooth_velocity = 1 with the
    !>    default smoothspec_fast = 1), the sparse operator of apply_smoothing applied on the device;
    !>    the other smoothing modes on the host (apply_smoothing per model).
    !> 8. Magnitudes: per (band, model) filter integrals over each band's support, compensated
    !>    at FP32 kernel precision (each model resampled to the observed frame at its redshift);
    !>    optionally spectral indices.
    !>
    !> Every stage runs on the device with OpenMP target offload
    !> (-Daccel=omp_gpu), both directive sets on the same loops; the context must be on the
    !> device (moved there on the first call if it is not). The two matrix products (age sums,
    !> band fluxes) use cuBLAS when built with -Dcublas=true (NVIDIA only; a ceiling for the
    !> portable loops).
    !>
    !> Redshift and igm_factor may differ per model: each model's IGM transmission is computed on
    !> the device and its spectrum resampled to the (fixed) observed-frame grid for the shared
    !> filters, with its own distance term.
    !>
    !> Instead of SFH parameters, a precomputed SSP-weight matrix (ages x N, ssp_weights) can be
    !> given (non-parametric SFHs): the weights of each SSP age (mass formed in that bin, as
    !> sfh_weights_core gives them; spectra, mass and L_bol are per that mass), used as
    !> they are; tage is then not used, and SFR is not defined (NaN).
    !>
    !> With the SSP bases of all metallicities resident, a weight matrix over ages and
    !> metallicities (ssp_weights_z: ages x nz x N) sums every basis with its own weights instead
    !> of interpolating at logzsol: metallicity varying with age within a model (chemical
    !> enrichment, tabular SFHs with a metallicity column) and metallicity distributions. Not
    !> with nebular emission (status 2; compute_csp_scenario rejects that case too).
    !>
    !> Redshifted colours (redshift_colors = 1) need zred_mags, each model's magnitude redshift
    !> (compute_csp_scenario's all-ages mode gives the redshift of each output's age).
    !>
    !> IMF free per model (ssp_ops: the template operators, opensps_ssp_operator and
    !> opensps_ssp_batch): instead of a resident basis, each model's age sums come from the
    !> operators with its own imf1, imf2, imf3, vdmc and mdave (imf_type, the IMF mass limits and
    !> the isochrone modifications are the operators' and the context's, shared by the batch). One
    !> operator (the batch's metallicity) or one per metallicity of the context, in which case the
    !> metallicity is free per model too (logzsol; the SSP of a model is the blend of its two
    !> bracketing metallicities' SSPs). The rest of the batch is unchanged. Not with ssp_weights_z,
    !> SSP-level nebular emission (add_neb_emission = 2), X-ray binaries or the instrumental LSF (status 2).
    !>
    !> Not supported (status 2): redshifted colours without zred_mags, and models with tage <= 0
    !> without a weight matrix (the all-ages output of compute_csp_scenario is a batch with a
    !> weight matrix). Models must share the smoothing parameters (sigma_smooth,
    !> min/max_wave_smooth).
    !>
    !> Mass, L_bol and SFR follow `compute_csp_scenario` in FP64; spectra, magnitudes, indices,
    !> line luminosities and dust masses agree with it to the rounding of the kernel precision
    !> (summation order differs).

    use opensps_precision, only: WP, KP, kahan_add
    use opensps_constants, only: SAFE_FLOOR, PI, NEMLINE, NEBNAGE, NEBNZ, NEBNIP, H_PLANCK, L_SOL
    use opensps_types, only: params, params_core
    use opensps_context_types, only: opensps_context_t, opensps_context_sync_device_settings, batch_out_reserve, &
                                     batch_grid_release
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_loc, c_int, c_size_t, c_intptr_t, c_f_pointer
    use, intrinsic :: iso_fortran_env, only: int64
    use opensps_context, only: opensps_context_move_to_device
    use opensps_sfh, only: sfh_weights_core
    use opensps_sfh, only: get_sfh_properties_at_age
    use opensps_dust, only: compute_attenuation_curve_point, one_minus_trans
    use opensps_interpolation, only: find_interval, find_interval_with_hint
    use opensps_cosmology, only: igm_tau, igm_tau_dz, igm_tau_pixel_terms, igm_tau_z_terms, IGM_NTERM
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use opensps_gas, only: get_grid_indices_weights, prepare_line_gaussians
    use opensps_spectral_indices, only: compute_spectral_indices_columns
    use opensps_smoothing, only: apply_smoothing, apply_smoothing_adjoint
    use opensps_batch_photometry, only: compute_magnitudes_columns, prepare_batch_grid, smoothing_on_device, &
                                     prepare_smoothing_kernel, smooth_columns, smooth_columns_adjoint, distance_term_dz, &
                                     smooth_sigma_adjoint
    use opensps_ssp, only: configure_imf_parameters
    use opensps_imf, only: imf_pack_params, IMF_NPAR, IMF_P_ALPHA, IMF_P_VDMC, IMF_P_MDAVE
    use opensps_ssp_operator, only: ssp_op_mod_signature
    use opensps_ssp_batch, only: ssp_batch_age_sums_z, ssp_opset_t, ssp_batch_age_sums_adjoint, NIMFG, dot_lanes
    use opensps_sfh_tangent, only: sfh_weights_tangent, sfh_tangent_supported, NSFHT
#if defined(OPENSPS_ACCEL_BLAS)
    use opensps_accel_blas, only: accel_blas_init, accel_blas_gemm, ACCEL_BLAS_OP_N
    use, intrinsic :: iso_c_binding, only: c_loc
#endif
    use opensps_trace, only: opensps_trace_push, opensps_trace_pop

    implicit none
    private

    public :: compute_csp_batch, batch_spectra_device, batch_spectra_host
#if defined(OMP_TARGET) && defined(__flang__)
    public :: nebular_continuum_tiled   ! (the nebular kernel, for its unit test)
#endif
    public :: BGRAD_FAGN, BGRAD_DUST1, BGRAD_DUST2, BGRAD_DUST3, BGRAD_DUST_INDEX, BGRAD_DUST1_INDEX, &
              BGRAD_FRAC_OBRUN, BGRAD_FRAC_NODUST, BGRAD_MWR, BGRAD_UVB, BGRAD_DUSTE_QPAH, BGRAD_DUSTE_UMIN, &
              BGRAD_DUSTE_GAMMA, BGRAD_GAS_LOGZ, BGRAD_GAS_LOGU, BGRAD_LOGZSOL, BGRAD_TAGE, BGRAD_TAU, BGRAD_CONST, &
              BGRAD_FBURST, BGRAD_TBURST, BGRAD_SF_START, BGRAD_ZRED, BGRAD_SIGMA_SMOOTH, BGRAD_SF_TRUNC, &
              BGRAD_SF_SLOPE, BGRAD_IMF1, BGRAD_IMF2, BGRAD_IMF3, BGRAD_VDMC, BGRAD_MDAVE, BGRAD_AGN_TAU, &
              BGRAD_IGM_FACTOR, NBGRAD

    !> Rows of params_bar (compute_csp_batch's gradients): the derivative of the caller's loss with
    !> respect to each parameter of the model. The list grows as the adjoints of more stages are
    !> written; callers index it by these names and size it with NBGRAD.
    integer, parameter :: BGRAD_FAGN = 1
    !> The dust parameters of attenuate_batch (mwr and uvb act with dust_type 1 only; a caller with
    !> dust1 = dust_ratio * dust2 chains the two rows itself)
    integer, parameter :: BGRAD_DUST1 = 2, BGRAD_DUST2 = 3, BGRAD_DUST3 = 4, BGRAD_DUST_INDEX = 5, &
                          BGRAD_DUST1_INDEX = 6, BGRAD_FRAC_OBRUN = 7, BGRAD_FRAC_NODUST = 8, BGRAD_MWR = 9, &
                          BGRAD_UVB = 10
    !> Dust emission (Draine & Li template: q_PAH and U_min interpolated linearly in the grid, so
    !> their derivative is that of the bracketing interval, one-sided at the grid nodes)
    integer, parameter :: BGRAD_DUSTE_QPAH = 11, BGRAD_DUSTE_UMIN = 12, BGRAD_DUSTE_GAMMA = 13
    !> Nebular emission (CLOUDY grids interpolated linearly in gas_logz and gas_logu, in log space)
    integer, parameter :: BGRAD_GAS_LOGZ = 14, BGRAD_GAS_LOGU = 15
    !> Per-model metallicity (all metallicities' bases resident: the SSPs interpolated linearly in
    !> log Z, not clamped; zero with one basis, where logzsol does not act in the batch)
    integer, parameter :: BGRAD_LOGZSOL = 16
    !> The SFH parameters (models with SFH parameters, not a weight matrix: sfh = 0, 1, 4 with const,
    !> burst and truncation, 5; opensps_sfh_tangent). Zero with a weight matrix
    integer, parameter :: BGRAD_TAGE = 17, BGRAD_TAU = 18, BGRAD_CONST = 19, BGRAD_FBURST = 20, &
                          BGRAD_TBURST = 21, BGRAD_SF_START = 22
    !> The redshift: the observed-frame resampling and distance of the magnitudes (unless zred_mags
    !> gives their redshift), and the IGM absorption. At z = 0 (rest frame, no IGM) it is zero
    integer, parameter :: BGRAD_ZRED = 23
    !> sigma_smooth (the device smoothing operator), per model: the batch shares one sigma, so the
    !> row is that model's own sensitivity (a sampler with one sigma for all sums the rows)
    integer, parameter :: BGRAD_SIGMA_SMOOTH = 24
    !> The SFH's truncation (sf_trunc in (0, tage]: the young end of the star-forming window)
    integer, parameter :: BGRAD_SF_TRUNC = 25
    !> The Simha SFH's slope after the truncation (sfh = 5)
    integer, parameter :: BGRAD_SF_SLOPE = 26
    !> The IMF parameters with ssp_ops (the operator's reverse pass, ssp_batch_age_sums_adjoint)
    integer, parameter :: BGRAD_IMF1 = 27, BGRAD_IMF2 = 28, BGRAD_IMF3 = 29, BGRAD_VDMC = 30, BGRAD_MDAVE = 31
    !> The AGN torus optical depth (agn_tau: the templates interpolated linearly, so the derivative is
    !> that of the bracketing interval, one-sided at the nodes; zero outside the grid, where the
    !> forward clamps, and for fagn = 0)
    integer, parameter :: BGRAD_AGN_TAU = 32
    !> The IGM optical depth's factor (igm_factor), per model as sigma_smooth (zero at z = 0 or
    !> without IGM absorption)
    integer, parameter :: BGRAD_IGM_FACTOR = 33
    integer, parameter :: NBGRAD = 33
    !> The rows of the SFH tangents' parameters (opensps_sfh_tangent: SFHT_TAGE .. SFHT_SF_SLOPE)
    integer, parameter :: SFHT_ROWS(8) = [17, 18, 19, 20, 21, 22, 25, 26]
    !> Rows BGRAD_DUST1 .. BGRAD_DUSTE_GAMMA, summed per model by the attenuation adjoint (10-12 as
    !> derivatives with respect to the template's interpolation weights wq, wu, gamma), and (13) the
    !> IGM's part of d / d zred, (14) d / d igm_factor
    integer, parameter :: NDP = 14
    !> Nebular continuum adjoint: at most this many ages per run (ages in one CLOUDY interval)
    integer, parameter :: NEB_RUN = 4
    !> Pixel chunks per model of the attenuation adjoint (each thread takes every NCHG-th pixel)
    integer, parameter :: NCHG = 64

#if defined(__flang__)
#define PBUF_ARG pbuf
#else
#define PBUF_ARG psets
#endif

    !> Words per model of the packed parameters (PARAMS_CORE) that flang kernels read through c_f_pointer
    integer, parameter :: NPW = (storage_size(params_core()) + 63) / 64

#if (defined(OPENSPS_PINNED) || defined(FSPS_PINNED)) && defined(OMP_TARGET)
    !> Device-to-host DMA into a page-locked buffer on a given CUDA stream (src/core/opensps_pinned.c)
    interface
        integer(c_int) function opensps_memcpy_d2h_async(dst, src, bytes, stream) bind(C, name="opensps_memcpy_d2h_async")
            import :: c_int, c_ptr, c_size_t, c_intptr_t
            type(c_ptr), value :: dst, src
            integer(c_size_t), value :: bytes
            integer(c_intptr_t), value :: stream
        end function opensps_memcpy_d2h_async
#ifdef OMP_TARGET
        !> A CUDA stream for the copy-back, and the wait for it (OpenMP target: its own streams)
        type(c_ptr) function opensps_stream_create() bind(C, name="opensps_stream_create")
            import :: c_ptr
        end function opensps_stream_create
        integer(c_int) function opensps_stream_sync(stream) bind(C, name="opensps_stream_sync")
            import :: c_int, c_intptr_t
            integer(c_intptr_t), value :: stream
        end function opensps_stream_sync
#endif
    end interface
#endif
#if (defined(OPENSPS_PINNED) || defined(FSPS_PINNED)) && defined(OMP_TARGET)
    !> The two copy-back streams of the OpenMP target build (created on first use; 0: none)
    integer(c_intptr_t), save :: copy_stream(2) = 0
#endif

#if defined(OPENSPS_HIP)
    interface
        subroutine opensps_hip_nebular_continuum_sp(n_mod, nspec, ntab, nt_neb, tab, ia, wa, dwa, newr, &
                                                    cy, co, lns, iz, wz, iu, wu, yo, stream) &
            bind(C, name="opensps_hip_nebular_continuum_sp")
            import :: c_ptr, c_int
            integer(c_int), value :: n_mod, nspec, ntab, nt_neb
            type(c_ptr), value :: tab, ia, wa, dwa, newr, cy, co, lns, iz, wz, iu, wu, yo, stream
        end subroutine opensps_hip_nebular_continuum_sp

        subroutine opensps_hip_nebular_continuum_dp(n_mod, nspec, ntab, nt_neb, tab, ia, wa, dwa, newr, &
                                                    cy, co, lns, iz, wz, iu, wu, yo, stream) &
            bind(C, name="opensps_hip_nebular_continuum_dp")
            import :: c_ptr, c_int
            integer(c_int), value :: n_mod, nspec, ntab, nt_neb
            type(c_ptr), value :: tab, ia, wa, dwa, newr, cy, co, lns, iz, wz, iu, wu, yo, stream
        end subroutine opensps_hip_nebular_continuum_dp

        subroutine opensps_hip_adj_nebular_continuum_sp(n_mod, nspec, ntab, ntn, nrun, tab, gw, it_tesc, ia, wa, &
                                                        rs, rl, iz, wz, iu, wu, fabs, qzt, lref, csum, stream) &
            bind(C, name="opensps_hip_adj_nebular_continuum_sp")
            import :: c_ptr, c_int
            integer(c_int), value :: n_mod, nspec, ntab, ntn, nrun
            type(c_ptr), value :: tab, gw, it_tesc, ia, wa, rs, rl, iz, wz, iu, wu, fabs, qzt, lref, csum, stream
        end subroutine opensps_hip_adj_nebular_continuum_sp

        subroutine opensps_hip_adj_nebular_continuum_dp(n_mod, nspec, ntab, ntn, nrun, tab, gw, it_tesc, ia, wa, &
                                                        rs, rl, iz, wz, iu, wu, fabs, qzt, lref, csum, stream) &
            bind(C, name="opensps_hip_adj_nebular_continuum_dp")
            import :: c_ptr, c_int
            integer(c_int), value :: n_mod, nspec, ntab, ntn, nrun
            type(c_ptr), value :: tab, gw, it_tesc, ia, wa, rs, rl, iz, wz, iu, wu, fabs, qzt, lref, csum, stream
        end subroutine opensps_hip_adj_nebular_continuum_dp
    end interface
#endif


    real(WP), parameter :: V_BAND_ANGSTROMS = 5500.0_wp   ! as in opensps_dust

    !> Nebular emission data of a batch (nebular_setup)
    type :: nebular_setup_t
        integer :: nt_neb = 0         !< ages with nebular emission (1..nt_neb)
        integer :: whlylim = 0        !< pixels below the Lyman limit
        logical :: cont = .false., lines_in_spec = .false., xrb = .false.
        !> Host-filled inputs: sections of ipk, rpk, lpk (pack_nebular), one transfer per type
        integer,  allocatable :: ipk(:)
        real(WP), allocatable :: rpk(:)
        logical,  allocatable :: lpk(:)
        integer,  pointer, contiguous :: ia(:) => null()     !< per age: CLOUDY age interval and weight
        real(WP), pointer, contiguous :: wa(:) => null()
        real(WP), pointer, contiguous :: dwa(:) => null()    !< per age: wa(t) - wa(t-1) within an interval
        logical,  pointer, contiguous :: newr(:) => null()   !< per age: the step dwa differs from the previous one
        !> (ages, N): SFH weight * Q, young and old (continuum), divided by the model's largest one,
        !> whose logarithm is lns (per model): Q reaches ~1e53, out of the FP32 range (rule 5)
        real(KP), allocatable :: cy(:,:), co(:,:)
        real(WP), allocatable :: lns(:)
        real(WP), allocatable :: q0(:,:)        !< per age and basis: ionizing photons (frac_obrun = 0)
        integer,  pointer, contiguous :: iz(:) => null(), iu(:) => null()  !< per model: gas_logz and gas_logu intervals and weights
        real(WP), pointer, contiguous :: wz(:) => null(), wu(:) => null()
        real(WP), pointer, contiguous :: fesc(:) => null()   !< per model: escaping EUV fraction (clamped frac_obrun)
        real(WP), pointer, contiguous :: fabs(:) => null()   !< per model: 1 - frac_obrun (absorbed fraction in Q)
        real(KP), allocatable :: lum(:,:)       !< line luminosities (lines, 2N): young, then old
        integer,  pointer, contiguous :: lptr(:) => null(), lid(:) => null() !< per pixel (CSR): lines whose profile reaches it
        integer,  pointer, contiguous :: llo(:) => null()    !< per line: pixel interval of its wavelength
        integer :: nrun = 0
        integer,  pointer, contiguous :: rs(:) => null(), rl(:) => null(), rt(:) => null()
        integer,  pointer, contiguous :: lj_lo(:) => null(), lj_hi(:) => null()
    end type nebular_setup_t


contains


    !> @brief Evaluates a batch of single-age CSPs.
    !>
    !> @param[inout] ctx     Context with the SSP basis set (opensps_context_update_ssp_basis) and,
    !>                       in GPU builds, moved to the device.
    !> @param[in]    psets   Parameter sets, one per model (prepared: opensps_context_prepare_pset).
    !> @param[out]   mags    Magnitudes (nbands, N); bands not requested in psets(1)%mag_compute
    !>                       are NaN, as in compute_magnitudes.
    !> @param[out]   status  0 on success; 1 no setup or basis; 2 unsupported option; 3 bad shape.
    !> @param[out]   spec    Optional spectra (nspec, N), L_sun/Hz per solar mass formed.
    !> @param[out]   mass, lbol, mdust, sfr  Optional per-model scalars as in compspout.
    !> @param[out]   emlines Optional emission-line luminosities (NEMLINE, N), as compspout%emlines
    !>                       (zero without nebular emission).
    !> @param[out]   indx    Optional spectral indices (nindx, N), as compspout%indx.
    !> @param[in]    ssp_weights Optional SSP weights (ntfull, N) instead of the SFH parameters.
    !> @param[out]   spec_young, spec_old  Optional (all six together) intrinsic young and old
    !>                       spectra (nspec, N), L_sun/Hz per solar mass formed: the age sums split
    !>                       at dust_tesc (young: ages <= dust_tesc), with nebular emission included
    !>                       (it belongs to the young ages; their ionizing flux is removed as in the
    !>                       full model), before any dust, dust emission, AGN, IGM or smoothing. They
    !>                       are what the attenuation stage receives: without dust emission, AGN, IGM
    !>                       and smoothing, spec = (young (T_birth (1 - frac_obrun) + frac_obrun) +
    !>                       old T_dust3) (T_diffuse (1 - frac_nodust) + frac_nodust).
    !> @param[out]   mass_young, mass_old  Surviving stellar mass of the young and old ages.
    !> @param[out]   lbol_young, lbol_old  log10 L_bol [L_sun] of the young and old stellar ages
    !>                       (as lbol: stars only).
    !> @param[in]    ssp_weights_z Optional SSP weights (ntfull, nz, N) over ages and metallicities,
    !>                       instead of the SFH parameters (or ssp_weights), with the bases of all
    !>                       nz metallicities resident.
    !> @param[in]    zred_mags Optional redshift of each model's magnitudes (N), in place of its
    !>                       zred for the magnitudes only: the spectrum, its IGM absorption and
    !>                       the SFH stay at zred (compute_csp_scenario's redshift_colors = 1).
    !> @param[inout] ssp_ops Optional set of template operators (ssp_opset_build of ssp_operator_build
    !>                       operators, for the context's non-IMF parameters): the IMF free per model, no
    !>                       basis needed (moved to the device on first use in device builds). One operator
    !>                       (psets%zmet) or one per metallicity of the context (logzsol per model).
    !> @param[in]    mags_bar, spec_bar  Optional gradient of the caller's loss L with respect to the
    !>                       magnitudes (nbands, N) and the spectra (nspec, N); either or both. With
    !>                       params_bar or weights_bar, the batch runs its reverse pass (batch_backward)
    !>                       after the forward one. Bands whose magnitude is not computed (NaN) are
    !>                       ignored.
    !> @param[out]   params_bar Optional dL / d(parameter) (NBGRAD, N), rows BGRAD_*.
    !> @param[out]   weights_bar Optional dL / d(SSP weight) (ntfull, N): the derivative with respect to
    !>                       the mass formed in each SSP age (ssp_weights, or the weights the SFH
    !>                       parameters give), for every age, including ages whose weight is zero (the
    !>                       spectra are linear in the weights). Masses are per solar mass formed, so a
    !>                       mass scale is the caller's: d mag / d log10 M = -2.5.
    !>                       With SFH parameters, params_bar includes tage, tau, const, fburst,
    !>                       tburst, sf_start, sf_trunc and sf_slope for sfh = 0, 1, 4, 5.
    !> @param[in]    mass_bar, lbol_bar Optional dL / d(mass), dL / d(lbol) (N): cotangents of those
    !>                       outputs.
    !> @param[out]   weights_z_bar Optional dL / d(ssp_weights_z) (ntfull, nz, N), with ssp_weights_z
    !>                       (weights_bar is then zero). With
    !>                       ssp_ops, params_bar includes the IMF's imf1, imf2, imf3, vdmc and mdave.
    subroutine compute_csp_batch(ctx, psets, mags, status, spec, mass, lbol, mdust, sfr, emlines, indx, ssp_weights, &
                                 spec_young, spec_old, mass_young, mass_old, lbol_young, lbol_old, zred_mags, &
                                 ssp_weights_z, ssp_ops, mags_bar, spec_bar, params_bar, weights_bar, mass_bar, lbol_bar, &
                                 weights_z_bar)
        type(opensps_context_t), intent(inout) :: ctx
#if defined(__flang__)
        type(params), intent(in)            :: psets(:)
#else
        type(params), intent(in), target    :: psets(:)
#endif
        real(WP), intent(out)               :: mags(:,:)
        integer, intent(out)                :: status
        real(WP), intent(out), optional     :: spec(:,:)
        real(WP), intent(out), optional     :: mass(:), lbol(:), mdust(:), sfr(:)
        real(WP), intent(out), optional     :: emlines(:,:), indx(:,:)
        real(WP), intent(in), optional      :: ssp_weights(:,:)
        real(WP), intent(out), optional     :: spec_young(:,:), spec_old(:,:)
        real(WP), intent(out), optional     :: mass_young(:), mass_old(:), lbol_young(:), lbol_old(:)
        real(WP), intent(in), optional      :: zred_mags(:)
        real(WP), intent(in), optional      :: ssp_weights_z(:,:,:)
        type(ssp_opset_t), intent(inout), optional :: ssp_ops
        real(WP), intent(in), optional      :: mags_bar(:,:), spec_bar(:,:)
        real(WP), intent(out), optional     :: params_bar(:,:), weights_bar(:,:)
        real(WP), intent(in), optional      :: mass_bar(:), lbol_bar(:)
        real(WP), intent(out), optional     :: weights_z_bar(:,:,:)

        integer :: n_mod, nt, nspec, nbands, m, k
        real(WP) :: zred, dust_age_log, w, mass_frac, sfr_norm, frac_linear, mk, lk
        logical  :: igm, any_emis, neb, agn, smooth, zmode, zvar, given_w, host_w, parts, zw, host_m, imf, dev, zimf, grad
        ! Gradients: the band fluxes of the magnitudes, and each column's magnitude redshift
        real(WP), allocatable :: fbk(:,:), zcm(:), dchain(:,:)
        ! Gradients: dL / d(SSP weights) of every model, and the SFH weights' tangents (opensps_sfh_tangent)
        real(WP), allocatable :: wbl(:,:), tw(:,:), tdw(:,:), tsc(:,:,:), lbx(:), wzbl(:,:,:), mbx(:)
        integer :: q
        ! Gradients with dust emission: the age sums before attenuate_batch replaces them; with
        ! smoothing, the spectra before it
        real(KP), allocatable :: ysave(:,:), tsave(:,:)
        ! IMF per model (ssp_ops): IMF parameters (IMF_NPAR x N), per age and model the surviving mass
        ! and linear L_bol, the SFH weights (FP64, masked as w2), and a placeholder basis
        real(WP), allocatable :: pm(:,:), smass(:,:), slum(:,:), wsfh(:,:)
        real(KP), allocatable :: nobasis(:,:,:)
        !> Below this batch size the SFH weights are computed on the host: on the device each
        !> model's SFH integral is one serial thread (~0.3 ms on the A100, the same on the P100),
        !> on the host ~20 us per model. (Hardware note: the crossover depends on the host core
        !> and on the GPU's single-thread FP64 speed; measured on the A100 node.)
        integer, parameter :: HOST_SFH_MAX = 16
        real(WP), allocatable :: wh(:,:), t1h(:), t2h(:)
        integer  :: nzb, nw, nzw, z

        ! Per-model inputs and results: sections of one array per type (ipk, rpk, lpk in, rout
        ! out), so that each moves in one transfer. (At N = 1 the ~30 separate small pageable
        ! transfers and their array descriptors were most of a call: ~10-20 us each on the A100
        ! node.)
        integer,  allocatable, target :: ipk(:)
        real(WP), allocatable, target :: rpk(:), rout(:)
        logical,  allocatable, target :: lpk(:)
        !> The parameters of every model (PARAMS_CORE). LLVM flang: raw words (from transfer) that
        !> device code reads a row of through c_f_pointer, because its default mapper made a kernel
        !> launch cost ~10 ms per model for an array of derived type. Other compilers read psets itself
        !> (no copy; nvfortran rejects c_f_pointer in device code: NVFORTRAN-S-1058)
#if defined(__flang__)
        integer(int64), allocatable, target :: pbuf(:,:)
#endif
        type(params_core), pointer :: pl
        integer,  pointer, contiguous :: it_tesc(:), iq(:), iu(:), ita(:), zlo(:), zhi(:)
        real(WP), pointer, contiguous :: wq(:), wu(:), gam(:), wta(:), dz(:), zcol(:), lum_linear(:), wext(:,:)
        logical,  pointer, contiguous :: emis(:), agn_on(:)
        real(WP), pointer, contiguous :: s_abs(:), s_norm(:), s_esc(:), s_line(:), msum(:), lsum(:)
        real(WP), pointer, contiguous :: msy(:), mso(:), lsy(:), lso(:)

        ! Work arrays on the device
        real(KP), allocatable :: w2(:,:)   ! (ages, 2N): young weights, then old
        real(KP), allocatable :: wz(:,:,:) ! (ages, nz, N): the weights over ages and metallicities
        real(WP), allocatable :: weights(:,:,:), tmp1(:,:), tmp2(:,:)

        ! Nebular emission and AGN (per age, per model, per pixel line lists)
        type(nebular_setup_t), target :: nb
        real(KP), allocatable :: eml(:,:)

        ! Work arrays (pixels x models) at kernel precision
        real(KP), allocatable :: yo(:,:), absorbed(:,:), tdiff(:,:)

        ! Device builds: the batch runs on the device, so a context not yet there is moved
#if defined(OMP_TARGET)
        if (.not. ctx%on_device) call opensps_context_move_to_device(ctx)
#endif
        call opensps_context_sync_device_settings(ctx)
        status = 0
        n_mod = size(psets)
        nt = ctx%state%ntfull
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        if (n_mod == 0) return

        ! ------------------------------------------------------------------
        ! 0. Checks
        ! ------------------------------------------------------------------
        imf = present(ssp_ops)
        zimf = .false.
        if (imf) zimf = ssp_ops%nz > 1
        if (ctx%state%check_sps_setup == 0 .or. (.not. imf .and. .not. allocated(ctx%state%ssp_basis_spec))) then
            status = 1; return
        end if
        if (imf) then
            ! The operators of the SSPs (one metallicity: the batch's; several: one per metallicity of
            ! the context, the metallicity free per model)
            if (ssp_ops%nz < 1 .or. .not. allocated(ssp_ops%ops)) then
                status = 1; return
            end if
            if (zimf .and. ssp_ops%nz /= ctx%state%nz) then
                status = 3; return
            end if
            do m = 1, ssp_ops%nz
                if (ssp_ops%ops(m)%n_pts == 0 .or. .not. allocated(ssp_ops%ops(m)%pt_q)) then
                    status = 1; return
                end if
                if (ssp_ops%ops(m)%nspec /= nspec .or. ssp_ops%ops(m)%n_ages /= nt) then
                    status = 3; return
                end if
                if (zimf .and. ssp_ops%ops(m)%zmet /= m) then
                    status = 3; return
                end if
                if (.not. all(ssp_ops%ops(m)%age_on)) then
                    status = 2; return
                end if
            end do
            if (.not. zimf .and. ssp_ops%ops(1)%zmet /= psets(1)%zmet) then
                status = 3; return
            end if
            if (present(ssp_weights_z) .or. ctx%add_neb_emission_val == 2 .or. ctx%add_xrb_emission_val == 1 .or. &
                ctx%smooth_lsf_val == 1) then
                status = 2; return
            end if
        end if
        if (size(mags, 1) < nbands .or. size(mags, 2) < n_mod) then
            status = 3; return
        end if
        if (present(spec)) then
            if (size(spec, 1) < nspec .or. size(spec, 2) < n_mod) then
                status = 3; return
            end if
        end if
        if (present(emlines)) then
            if (size(emlines, 1) < NEMLINE .or. size(emlines, 2) < n_mod) then
                status = 3; return
            end if
        end if
        if (present(indx)) then
            if (size(indx, 1) < ctx%state%nindx .or. size(indx, 2) < n_mod) then
                status = 3; return
            end if
        end if
        if (ctx%redshift_colors_val /= 0 .and. .not. present(zred_mags)) then
            status = 2; return
        end if
        if (present(zred_mags)) then
            if (size(zred_mags) < n_mod) then
                status = 3; return
            end if
        end if
        parts = present(spec_young) .and. present(spec_old) .and. present(mass_young) .and. &
                present(mass_old) .and. present(lbol_young) .and. present(lbol_old)
        if (parts) then
            if (size(spec_young, 1) < nspec .or. size(spec_young, 2) < n_mod .or. &
                size(spec_old, 1) < nspec .or. size(spec_old, 2) < n_mod .or. size(mass_young) < n_mod .or. &
                size(mass_old) < n_mod .or. size(lbol_young) < n_mod .or. size(lbol_old) < n_mod) then
                status = 3; return
            end if
        end if
        grad = present(params_bar) .or. present(weights_bar) .or. present(weights_z_bar)
        if (grad) then
            if (present(mags_bar)) then
                if (size(mags_bar, 1) < nbands .or. size(mags_bar, 2) < n_mod) then
                    status = 3; return
                end if
            end if
            if (present(spec_bar)) then
                if (size(spec_bar, 1) < nspec .or. size(spec_bar, 2) < n_mod) then
                    status = 3; return
                end if
            end if
            if (present(params_bar)) then
                if (size(params_bar, 1) < NBGRAD .or. size(params_bar, 2) < n_mod) then
                    status = 3; return
                end if
            end if
            if (present(weights_bar)) then
                if (size(weights_bar, 1) < nt .or. size(weights_bar, 2) < n_mod) then
                    status = 3; return
                end if
            end if
        end if
        given_w = present(ssp_weights)
        if (given_w) then
            if (size(ssp_weights, 1) < nt .or. size(ssp_weights, 2) < n_mod) then
                status = 3; return
            end if
        end if
        ! Weights over ages and metallicities: every resident basis summed with its own weights
        nzb = 1
        if (.not. imf) nzb = size(ctx%state%ssp_basis_spec, 3)
        if (imf) nzb = ssp_ops%nz
        zw = present(ssp_weights_z)
        nzw = 1
        if (zw) then
            if (given_w .or. size(ssp_weights_z, 1) < nt .or. size(ssp_weights_z, 2) < nzb .or. &
                size(ssp_weights_z, 3) < n_mod) then
                status = 3; return
            end if
            ! Not with nebular emission: its kernels take the young/old weights of one basis (w2),
            ! which these weights do not fill (with one resident basis either)
            if (ctx%add_neb_emission_val == 1) then
                status = 2; return
            end if
            given_w = .true.
            nzw = nzb
        end if
        zred = psets(1)%zred
        zvar = .false.
        do m = 1, n_mod
            if ((psets(m)%tage <= 0.0_wp .and. .not. given_w) .or. psets(m)%sigma_smooth /= psets(1)%sigma_smooth .or. &
                psets(m)%min_wave_smooth /= psets(1)%min_wave_smooth .or. &
                psets(m)%max_wave_smooth /= psets(1)%max_wave_smooth .or. &
                psets(m)%uvb < 0.0_wp .or. psets(m)%wgp1 < 1 .or. psets(m)%wgp2 < 1) then
                status = 2; return
            end if
            if (psets(m)%zred /= zred .or. psets(m)%igm_factor /= psets(1)%igm_factor) zvar = .true.
        end do
        ! Gradients of the SFH parameters: the forms opensps_sfh_tangent covers
        if (present(params_bar) .and. .not. given_w) then
            do m = 1, n_mod
                if (.not. sfh_tangent_supported(psets(m)%params_core, psets(m)%tage)) then
                    status = 2; return
                end if
            end do
        end if
        host_w = .false.
#if defined(OMP_TARGET)
        host_w = (.not. given_w) .and. n_mod < HOST_SFH_MAX
#endif
        ! IMF per model: one metallicity (the operator's)
        if (imf) then
            do m = 1, n_mod
                if (.not. zimf) then
                    if (psets(m)%zmet /= psets(1)%zmet) then
                        status = 2; return
                    end if
                end if
                ! Isochrone modifications change which templates the stars use, so S is rebuilt for
                ! them: not supported per model (status 2), and not different from the build either
                if (any(ssp_op_mod_signature(psets(m)) /= ssp_ops%ops(1)%mod_sig)) then
                    status = 2; return
                end if
            end do
        end if
        nw = 1
        if (given_w .or. host_w) nw = nt * nzw * n_mod
        allocate(ipk(6 * n_mod), rpk(6 * n_mod + nt + nw), lpk(2 * n_mod), rout(10 * n_mod))
        it_tesc => ipk(1:n_mod)
        iq => ipk(n_mod + 1:2 * n_mod)
        iu => ipk(2 * n_mod + 1:3 * n_mod)
        ita => ipk(3 * n_mod + 1:4 * n_mod)
        zlo => ipk(4 * n_mod + 1:5 * n_mod)
        zhi => ipk(5 * n_mod + 1:6 * n_mod)
        wq => rpk(1:n_mod)
        wu => rpk(n_mod + 1:2 * n_mod)
        gam => rpk(2 * n_mod + 1:3 * n_mod)
        wta => rpk(3 * n_mod + 1:4 * n_mod)
        dz => rpk(4 * n_mod + 1:5 * n_mod)
        zcol => rpk(5 * n_mod + 1:6 * n_mod)
        lum_linear => rpk(6 * n_mod + 1:6 * n_mod + nt)
        if (nw > 1) then
            wext(1:nt * nzw, 1:n_mod) => rpk(6 * n_mod + nt + 1:6 * n_mod + nt + nw)
        else
            wext(1:1, 1:1) => rpk(6 * n_mod + nt + 1:6 * n_mod + nt + 1)
        end if
        emis => lpk(1:n_mod)
        agn_on => lpk(n_mod + 1:2 * n_mod)
        s_abs => rout(1:n_mod)
        s_norm => rout(n_mod + 1:2 * n_mod)
        s_esc => rout(2 * n_mod + 1:3 * n_mod)
        msum => rout(3 * n_mod + 1:4 * n_mod)
        lsum => rout(4 * n_mod + 1:5 * n_mod)
        s_line => rout(5 * n_mod + 1:6 * n_mod)
        msy => rout(6 * n_mod + 1:7 * n_mod)
        mso => rout(7 * n_mod + 1:8 * n_mod)
        lsy => rout(8 * n_mod + 1:9 * n_mod)
        lso => rout(9 * n_mod + 1:10 * n_mod)
        rpk = 0.0_wp

        zcol = psets(1:n_mod)%zred
        if (zw) then
            do m = 1, n_mod
                do z = 1, nzw
                    wext((z - 1) * nt + 1:z * nt, m) = ssp_weights_z(1:nt, z, m)
                end do
            end do
        else if (given_w) then
            wext = ssp_weights(1:nt, 1:n_mod)
        else if (host_w) then
            ! Few models: the SFH weights on the host (as compute_csp_scenario does for one output),
            ! then used as a given weight matrix
            allocate(wh(nt, 1), t1h(nt), t2h(nt))
            do m = 1, n_mod
                call sfh_weights_core(ctx, psets(m)%params_core, psets(m)%tage, 1, wh, t1h, t2h)
                wext(:, m) = wh(:, 1)
            end do
        end if

        ! Smoothing as apply_smoothing (which does nothing for sigma <= SAFE_FLOOR): the fast
        ! velocity kernel on the device, the other modes on the host (smooth_columns_host)
        smooth = psets(1)%sigma_smooth > SAFE_FLOOR

        ! Per-model metallicity: a basis per metallicity of the context (interpolated at logzsol,
        ! unless the weights cover the metallicities)
        zmode = nzb > 1 .and. .not. zw
        if ((zmode .or. (zw .and. nzb > 1)) .and. nzb /= ctx%state%nz) then
            status = 3; return
        end if
        zlo = 1; zhi = 1; dz = 0.0_wp
        if (zmode) then
            do m = 1, n_mod
                call metallicity_interval(ctx, psets(m)%logzsol, zlo(m), dz(m))
                zhi(m) = zlo(m) + 1
            end do
        end if

        ! ------------------------------------------------------------------
        ! 1. Host: per-model indices (birth-cloud age, dust emission grid, nebular grid, AGN
        !    template) and SFR
        ! ------------------------------------------------------------------
        allocate(w2(nt, 2 * n_mod))
        ! SFH scratch: sfh_weights_core uses ntfull entries (sized from the ages, not from the
        ! context's CSP workspace, which a context used only through the batch may not have)
        allocate(weights(nt, nzw, n_mod), tmp1(nt, n_mod), tmp2(nt, n_mod))
        if (zw) then
            allocate(wz(nt, nzw, n_mod))
        else
            allocate(wz(1, 1, 1))
        end if
        if (imf) then
            lum_linear = 0.0_wp
        else
            lum_linear = 10.0_wp**ctx%state%ssp_basis_lbol(1:nt, 1)
        end if
        do m = 1, n_mod
            if (psets(m)%dust_tesc > SAFE_FLOOR) then
                dust_age_log = psets(m)%dust_tesc
            else
                dust_age_log = 7.0_wp
            end if
            it_tesc(m) = max(1, min(find_interval(ctx%state%time_full, dust_age_log), nt))
            if (present(sfr) .and. given_w) then
                sfr(m) = ieee_value(1.0_wp, ieee_quiet_nan)
            else if (present(sfr)) then
                call get_sfh_properties_at_age(ctx, psets(m)%params_core, psets(m)%tage, mass_frac, sfr_norm, frac_linear)
                if (mass_frac > SAFE_FLOOR) sfr_norm = sfr_norm / mass_frac
                sfr(m) = sfr_norm
            end if

            emis(m) = ctx%add_dust_emission_val == 1 .and. &
                      (psets(m)%dust1 > SAFE_FLOOR .or. psets(m)%dust2 > SAFE_FLOOR)
            call dl07_indices(ctx, psets(m), iq(m), wq(m), iu(m), wu(m), gam(m))
            ! (With gradients the AGN stage runs for every model, so that d/d fagn exists at fagn = 0,
            ! where it adds exact zeros)
            call agn_indices(ctx, psets(m), grad, agn_on(m), ita(m), wta(m))
        end do
        any_emis = any(emis)
        agn = any(agn_on)
        ! Only with a nebular grid: isochrone sets without one (BaSTI, Geneva) get no nebular
        ! emission and zero lines, as in legacy FSPS (whose tables stay zero there). The setup
        ! leaves the grid's age axis at zero when it loads no grid (reset_setup_tables)
        neb = ctx%add_neb_emission_val == 1 .and. ctx%state%nebem_age(NEBNAGE) > 0.0_wp
        if (imf) then
            call nebular_setup(ctx, psets, neb, n_mod, nb)   ! ionizing photons per age and model
        else
            call nebular_setup(ctx, psets, neb, nzb, nb)
        end if

        ! IMF per model: the IMF parameters of each model (the context's IMF with its slopes,
        ! characteristic and break masses), the operator on the device
        dev = .false.
        if (imf) then
            call configure_imf_parameters(ctx, psets(1))
            allocate(pm(IMF_NPAR, n_mod), smass(nt, n_mod), slum(nt, n_mod), wsfh(n_mod, nt), nobasis(1, 1, 1))
            call imf_pack_params(ctx, pm(:, 1))
            do m = 1, n_mod
                pm(:, m) = pm(:, 1)
                pm(IMF_P_ALPHA, m) = psets(m)%imf1
                pm(IMF_P_ALPHA + 1, m) = psets(m)%imf2
                pm(IMF_P_ALPHA + 2, m) = psets(m)%imf3
                pm(IMF_P_VDMC, m) = psets(m)%vdmc
                pm(IMF_P_MDAVE, m) = psets(m)%mdave
            end do
            nobasis = 0.0_KP
#if defined(OMP_TARGET)
            dev = ctx%on_device
#endif
        end if
        if (neb) then
            allocate(eml(NEMLINE, n_mod))
        else
            allocate(eml(1, 1))
        end if

        ! Grid-only data for this redshift (cached in ctx%state%batch, on the device too)
        ! (with per-model redshifts: the z = 0 data, for the redshift-independent parts)
        if (zvar) then
            call prepare_batch_grid(ctx, 0.0_wp, psets(1)%igm_factor)
            igm = ctx%add_igm_absorption_val == 1 .and. any(zcol > SAFE_FLOOR)
        else
            call prepare_batch_grid(ctx, zred, psets(1)%igm_factor)
            igm = ctx%state%batch%igm
        end if
        if (smooth .and. smoothing_on_device(ctx)) call prepare_smoothing_kernel(ctx, psets(1)%sigma_smooth, &
                                                                             psets(1)%min_wave_smooth, &
                                                                             psets(1)%max_wave_smooth)

        ! ------------------------------------------------------------------
        ! 2-7. Device: age sums, nebular emission, attenuation, dust emission, IGM, AGN, smoothing
        ! ------------------------------------------------------------------
        allocate(yo(nspec, 2 * n_mod), absorbed(nspec, n_mod))
        if (grad .and. any_emis) then
            allocate(ysave(nspec, 2 * n_mod))
        else
            allocate(ysave(1, 1))
        end if
        if (grad .and. smooth) then
            allocate(tsave(nspec, n_mod))
        else
            allocate(tsave(1, 1))
        end if
        ! The spectra: the context's persistent batch output (on the device; columns 1:n_mod)
        call batch_out_reserve(ctx%state%batch_out, nspec, n_mod, ctx%on_device)
        ! Diffuse-dust transmission kept from the attenuation for the AGN emission
        if (agn) then
            allocate(tdiff(nspec, n_mod))
        else
            allocate(tdiff(1, 1))
        end if

        ! Nebular data only with nebular emission (its kernels run only then)
#if defined(__flang__)
        allocate(pbuf(NPW, n_mod))
        do m = 1, n_mod
            pbuf(:, m) = transfer(psets(m)%params_core, pbuf(:, m))
        end do
#endif
#ifdef OMP_TARGET
#if defined(__flang__)
        !$omp target data map(to: pbuf, ipk, rpk, lpk) &
#else
        !$omp target data map(to: psets, ipk, rpk, lpk) &
#endif
        !$omp map(alloc: w2, wz, weights, tmp1, tmp2, yo, absorbed, tdiff, ysave, tsave) &
        !$omp map(from: rout, eml)
#endif
#ifdef OMP_TARGET
        ! The IMF-per-model arrays are mapped only then (the mappings cost ~10 us each per call)
        !$omp target data if(imf) map(to: pm, nobasis) map(alloc: smass, slum, wsfh)
        !$omp target data if(neb) map(to: nb%ipk, nb%rpk, nb%lpk) map(alloc: nb%q0, nb%lum, nb%cy, nb%co, nb%lns)
#endif

        ! SFH weights (FP64), mass and L_bol per model, one thread per model; the weights are split
        ! into young and old ages at the model's dust_tesc and cast once to kernel precision
        ! (Two kernels: the device call's array sections cost six descriptor transfers per launch,
        ! paid only when the device computes the weights.) Fewer than HOST_SFH_MAX models with
        ! weights from the host: these loops run on the host (host_m), then w2 / wz and the
        ! results go up in one update each (one device thread per model took ~90 us at N = 1)
        host_m = .false.
#if defined(OMP_TARGET)
        host_m = (given_w .or. host_w) .and. n_mod < HOST_SFH_MAX .and. .not. imf
#endif
        if (given_w .or. host_w) then
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2) if(target: .not. host_m)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, n_mod
                do k = 1, nt
                    do z = 1, nzw
                        weights(k, z, m) = wext((z - 1) * nt + k, m)
                    end do
                end do
            end do
        else
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do private(pl) &
            !$omp map(OMP_MAP_PRESENT_ALLOC: ctx%state%time_full, ctx%state%zlegend, ctx%state%ntfull, ctx%state%nz, &
            !$omp     ctx%state%ntabsfh, ctx%state%sfh_tab, ctx%interpolation_type_val, ctx%tiny_logt_val)
#elif defined(_OPENMP)
            !$omp parallel do private(pl)
#endif
            do m = 1, n_mod
#if !defined(__flang__)
                pl => psets(m)%params_core
#else
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                call sfh_weights_core(ctx, pl, pl%tage, 1, weights(:, :, m), tmp1(:, m), tmp2(:, m))
            end do
        end if
        ! IMF per model: the young and old age sums, and per age and model the mass, L_bol and
        ! ionizing photons, from the operator (opensps_ssp_batch), with the SFH weights masked as
        ! below
        if (imf) then
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2) if(target: dev)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do k = 1, nt
                do m = 1, n_mod
                    if (weights(k, 1, m) > SAFE_FLOOR) then
                        wsfh(m, k) = weights(k, 1, m)
                    else
                        wsfh(m, k) = 0.0_wp
                    end if
                end do
            end do
            call ssp_batch_age_sums_z(ctx%imf_type_val, ctx%add_stellar_remnants_val == 1, n_mod, nt, nspec, &
                                      ssp_ops, zlo, zhi, dz, pm, wsfh, it_tesc, nb%nt_neb, nb%whlylim, nb%fesc, yo, &
                                      smass, slum, nb%q0)
        end if

        associate (bas_mass => ctx%state%ssp_basis_mass, bas_lbol => ctx%state%ssp_basis_lbol)
        if (zw) then
            ! Weights over ages and metallicities: mass and L_bol summed over both (metallicities
            ! outer, as compute_csp_scenario with nzin > 1), the weights cast once to KP
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do private(k, z, w, mk, lk) if(target: .not. host_m) &
            !$omp& map(OMP_MAP_PRESENT_ALLOC: bas_lbol, bas_mass)
#elif defined(_OPENMP)
            !$omp parallel do private(k, z, w, mk, lk)
#endif
            do m = 1, n_mod
                msum(m) = 0.0_wp
                lsum(m) = 0.0_wp
                msy(m) = 0.0_wp; mso(m) = 0.0_wp; lsy(m) = 0.0_wp; lso(m) = 0.0_wp
                do z = 1, nzw
                    do k = 1, nt
                        w = weights(k, z, m)
                        if (w > SAFE_FLOOR) then
                            mk = w * bas_mass(k, z)
                            lk = w * 10.0_wp**bas_lbol(k, z)
                            msum(m) = msum(m) + mk
                            lsum(m) = lsum(m) + lk
                            if (k <= it_tesc(m)) then
                                msy(m) = msy(m) + mk; lsy(m) = lsy(m) + lk
                            else
                                mso(m) = mso(m) + mk; lso(m) = lso(m) + lk
                            end if
                        else
                            w = 0.0_wp
                        end if
                        wz(k, z, m) = real(w, KP)
                    end do
                end do
            end do
        else if (imf) then
            ! The model's own SSPs (IMF per model): the surviving mass and L_bol per age and model
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do private(k, w, mk, lk) if(target: dev)
#elif defined(_OPENMP)
            !$omp parallel do private(k, w, mk, lk)
#endif
            do m = 1, n_mod
                msum(m) = 0.0_wp
                lsum(m) = 0.0_wp
                msy(m) = 0.0_wp; mso(m) = 0.0_wp; lsy(m) = 0.0_wp; lso(m) = 0.0_wp
                do k = 1, nt
                    w = weights(k, 1, m)
                    if (w > SAFE_FLOOR) then
                        mk = w * smass(k, m)
                        lk = w * slum(k, m)
                        msum(m) = msum(m) + mk
                        lsum(m) = lsum(m) + lk
                        if (k <= it_tesc(m)) then
                            msy(m) = msy(m) + mk; lsy(m) = lsy(m) + lk
                        else
                            mso(m) = mso(m) + mk; lso(m) = lso(m) + lk
                        end if
                    else
                        w = 0.0_wp
                    end if
                    if (k <= it_tesc(m)) then
                        w2(k, m) = real(w, KP); w2(k, n_mod + m) = 0.0_KP
                    else
                        w2(k, m) = 0.0_KP; w2(k, n_mod + m) = real(w, KP)
                    end if
                end do
            end do
        else
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do private(k, w, mk, lk) if(target: .not. host_m) &
            !$omp& map(OMP_MAP_PRESENT_ALLOC: bas_lbol, bas_mass)
#elif defined(_OPENMP)
            !$omp parallel do private(k, w, mk, lk)
#endif
            do m = 1, n_mod
                msum(m) = 0.0_wp
                lsum(m) = 0.0_wp
                msy(m) = 0.0_wp; mso(m) = 0.0_wp; lsy(m) = 0.0_wp; lso(m) = 0.0_wp
                do k = 1, nt
                    w = weights(k, 1, m)
                    if (w > SAFE_FLOOR) then
                        if (zmode) then
                            ! The interpolated SSP's mass and L_bol (compute_interpolated_ssp)
                            mk = w * ((1.0_wp - dz(m)) * bas_mass(k, zlo(m)) + &
                                      dz(m) * bas_mass(k, zhi(m)))
                            lk = w * 10.0_wp**((1.0_wp - dz(m)) * bas_lbol(k, zlo(m)) + &
                                               dz(m) * bas_lbol(k, zhi(m)))
                        else
                            mk = w * bas_mass(k, 1)
                            lk = w * lum_linear(k)
                        end if
                        msum(m) = msum(m) + mk
                        lsum(m) = lsum(m) + lk
                        ! Young (ages <= dust_tesc) and old parts, summed separately (no differences)
                        if (k <= it_tesc(m)) then
                            msy(m) = msy(m) + mk; lsy(m) = lsy(m) + lk
                        else
                            mso(m) = mso(m) + mk; lso(m) = lso(m) + lk
                        end if
                    else
                        w = 0.0_wp
                    end if
                    if (k <= it_tesc(m)) then
                        w2(k, m) = real(w, KP); w2(k, n_mod + m) = 0.0_KP
                    else
                        w2(k, m) = 0.0_KP; w2(k, n_mod + m) = real(w, KP)
                    end if
                end do
            end do

        end if
        end associate

        if (host_m) then
#ifdef OMP_TARGET
            !$omp target update to(w2, wz, rout)
#endif
        end if

        if (imf) then
            ! (done above, from the operator)
        else if (zw) then
            call age_sums_w(nspec, nt, nzw, n_mod, ctx%state%ssp_basis_spec, wz, it_tesc, yo)
        else if (zmode) then
            call age_sums_z(nspec, nt, size(ctx%state%ssp_basis_spec, 3), n_mod, ctx%state%ssp_basis_spec, w2, w2, &
                            it_tesc, zlo, zhi, dz, yo)
        else
            call age_sums(nspec, nt, n_mod, ctx%state%ssp_basis_spec(:, :, 1), w2, it_tesc, yo)
        end if
        if (neb .and. imf) then
            call add_nebular_batch(ctx, n_mod, nspec, nt, nzb, nobasis, w2, it_tesc, zlo, zhi, dz, nb, yo, eml, &
                                   q_model=.true.)
        else if (neb) then
            call add_nebular_batch(ctx, n_mod, nspec, nt, nzb, ctx%state%ssp_basis_spec, w2, it_tesc, &
                                   zlo, zhi, dz, nb, yo, eml, q_model=.false.)
        end if
        if (neb) call attenuate_lines(ctx, PBUF_ARG, n_mod, nb%lum, nb%llo, eml, s_line)
        ! Intrinsic young and old spectra (the input of the attenuation), on request
        if (parts) then
#ifdef OMP_TARGET
            !$omp target update from(yo)
#endif
            do m = 1, n_mod
                spec_young(1:nspec, m) = real(yo(:, m), WP)
                spec_old(1:nspec, m) = real(yo(:, n_mod + m), WP)
            end do
        end if
        associate(total => ctx%state%batch_out%spec)
        if (grad .and. any_emis) then
            ! (the reverse pass reads the age sums, which attenuate_batch replaces for dust emission)
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, 2 * n_mod
                do k = 1, nspec
                    ysave(k, m) = yo(k, m)
                end do
            end do
        end if
        call attenuation_tables_cached(ctx, nspec)
        call attenuate_batch(ctx, PBUF_ARG, n_mod, nspec, iq, iu, wq, wu, gam, emis, agn_on, yo, total, absorbed, tdiff)
        if (any_emis) then
            call dust_emission_reductions(n_mod, nspec, emis, ctx%state%batch%wt, yo, absorbed, s_abs, s_norm, s_esc)
            ! Lines not in the spectrum: their absorbed part (intrinsic - attenuated) is added to
            ! the absorbed luminosity (legacy add_dust)
            if (neb .and. ctx%nebemlineinspec_val == 0) then
#if defined(OMP_TARGET)
                !$omp target teams distribute parallel do
#elif defined(_OPENMP)
                !$omp parallel do
#endif
                do m = 1, n_mod
                    s_abs(m) = s_abs(m) + s_line(m)
                end do
            end if
            call add_dust_emission(n_mod, nspec, emis, yo, s_abs, s_esc, total)
        end if

        if (igm .and. zvar) then
            call apply_igm_z(ctx, PBUF_ARG, n_mod, nspec, absorbed, total)   ! absorbed: free by now
        else if (igm) then
            call apply_igm(n_mod, nspec, ctx%state%batch%igm_t, total)
        end if
        if (agn) call add_agn_batch(ctx, PBUF_ARG, n_mod, nspec, agn_on, ita, wta, lsum, tdiff, total)
        if (grad .and. smooth) then
            ! (the reverse pass of sigma_smooth needs the spectra before smoothing)
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, n_mod
                do k = 1, nspec
                    tsave(k, m) = total(k, m)
                end do
            end do
        end if
        if (smooth .and. smoothing_on_device(ctx)) then
            call smooth_columns(ctx, total, n_mod)
        else if (smooth) then
            call smooth_columns_host(ctx, psets(1), total(:, 1:n_mod))
        end if

        ! ------------------------------------------------------------------
        ! 8. Magnitudes and indices
        ! ------------------------------------------------------------------
        ! (fbk: the band fluxes, allocated only for the gradients; unallocated, the argument is absent)
        if (grad) allocate(fbk(nbands, n_mod))
        if (present(zred_mags) .and. allocated(psets(1)%mag_compute)) then
            call compute_magnitudes_columns(ctx, zred, psets(1)%igm_factor, total, n_mod, mags, psets(1)%mag_compute, &
                                            zred_col=zred_mags(1:n_mod), fband_out=fbk)
        else if (present(zred_mags)) then
            call compute_magnitudes_columns(ctx, zred, psets(1)%igm_factor, total, n_mod, mags, &
                                            zred_col=zred_mags(1:n_mod), fband_out=fbk)
        else if (zvar .and. allocated(psets(1)%mag_compute)) then
            call compute_magnitudes_columns(ctx, zred, psets(1)%igm_factor, total, n_mod, mags, psets(1)%mag_compute, &
                                            zred_col=zcol, fband_out=fbk)
        else if (zvar) then
            call compute_magnitudes_columns(ctx, zred, psets(1)%igm_factor, total, n_mod, mags, zred_col=zcol, fband_out=fbk)
        else if (allocated(psets(1)%mag_compute)) then
            call compute_magnitudes_columns(ctx, zred, psets(1)%igm_factor, total, n_mod, mags, psets(1)%mag_compute, fband_out=fbk)
        else
            call compute_magnitudes_columns(ctx, zred, psets(1)%igm_factor, total, n_mod, mags, fband_out=fbk)
        end if
        if (present(indx)) call compute_spectral_indices_columns(ctx, ctx%state%spec_lambda, total, n_mod, indx)
        ! ------------------------------------------------------------------
        ! 9. Gradients: the reverse pass
        ! ------------------------------------------------------------------
        if (grad) then
            ! Each column's magnitude redshift, as compute_magnitudes_columns took it
            allocate(zcm(n_mod))
            if (present(zred_mags)) then
                zcm = zred_mags(1:n_mod)
            else if (zvar) then
                zcm = zcol
            else
                zcm = zred
            end if
            ! (the forward's per-model sums, L_bol among them, on the host for the reverse pass)
#ifdef OMP_TARGET
            !$omp target update from(rout)
#endif
            allocate(wbl(nt, n_mod), lbx(n_mod), mbx(n_mod))
            if (zw) then
                allocate(wzbl(nt, nzw, n_mod))
            else
                allocate(wzbl(1, 1, 1))
            end if
            lbx = 0.0_wp
            if (present(lbol_bar)) lbx = lbol_bar(1:n_mod)
            ! (with ssp_ops the mass's cotangent goes through the operator's reverse pass, the IMF
            ! parameters acting on the surviving mass and the remnants)
            mbx = 0.0_wp
            if (present(mass_bar) .and. imf) mbx = mass_bar(1:n_mod)
            ! The template's interpolation weights against q_PAH, U_min and gamma (dl07_indices: linear in
            ! the bracketing interval, clamped to the grid; one-sided at the nodes)
            allocate(dchain(7, n_mod))
            dchain = 0.0_wp
            ! Per-model metallicity: dz / d logzsol (metallicity_interval, not clamped)
            if (zmode) then
                do m = 1, n_mod
                    dchain(7, m) = 1.0_wp / (log10(ctx%state%zlegend(zhi(m)) / ctx%state%zsol) - &
                                             log10(ctx%state%zlegend(zlo(m)) / ctx%state%zsol))
                end do
            end if
            ! Nebular: gas_logz and gas_logu as get_grid_indices_weights (linear in the interval,
            ! clamped to the grid), frac_obrun as the escaping fraction (clamped to [0, 1])
            if (neb) then
                do m = 1, n_mod
                    associate (za => ctx%state%nebem_logz, ua => ctx%state%nebem_logu, p => psets(m))
                    if (p%gas_logz >= za(nb%iz(m)) .and. p%gas_logz <= za(nb%iz(m) + 1)) &
                        dchain(4, m) = 1.0_wp / (za(nb%iz(m) + 1) - za(nb%iz(m)))
                    if (p%gas_logu >= ua(nb%iu(m)) .and. p%gas_logu <= ua(nb%iu(m) + 1)) &
                        dchain(5, m) = 1.0_wp / (ua(nb%iu(m) + 1) - ua(nb%iu(m)))
                    if (p%frac_obrun >= 0.0_wp .and. p%frac_obrun <= 1.0_wp) dchain(6, m) = 1.0_wp
                    end associate
                end do
            end if
            if (any_emis) then
                do m = 1, n_mod
                    associate (qa => ctx%state%qpaharr, ua => ctx%state%uminarr, p => psets(m))
                    if (p%duste_qpah >= qa(iq(m)) .and. p%duste_qpah <= qa(iq(m) + 1)) &
                        dchain(1, m) = 1.0_wp / (qa(iq(m) + 1) - qa(iq(m)))
                    if (p%duste_umin >= ua(iu(m)) .and. p%duste_umin <= ua(iu(m) + 1)) &
                        dchain(2, m) = 1.0_wp / (ua(iu(m) + 1) - ua(iu(m)))
                    if (p%duste_gamma >= 0.0_wp .and. p%duste_gamma <= 1.0_wp) dchain(3, m) = 1.0_wp
                    end associate
                end do
            end if
            if (any_emis) then
                call batch_backward(ctx, PBUF_ARG, n_mod, nspec, nt, nbands, zred, psets(1)%igm_factor, mags, fbk, zcm, &
                                    it_tesc, zlo, zhi, dz, zmode, igm, zvar, agn_on, ita, wta, lsum, lum_linear, tdiff, &
                                    absorbed, ysave, total, emis, iq, iu, wq, wu, gam, s_abs, s_esc, dchain, smooth, &
                                    neb, nb, w2, present(zred_mags), tsave, psets(1)%sigma_smooth, &
                                    psets(1)%min_wave_smooth, psets(1)%max_wave_smooth, &
                                    imf, zimf, pm, wsfh, slum, smass, ssp_ops, lbx, mbx, zw, wzbl, &
                                    mags_bar, spec_bar, params_bar, wbl)
            else
                call batch_backward(ctx, PBUF_ARG, n_mod, nspec, nt, nbands, zred, psets(1)%igm_factor, mags, fbk, zcm, &
                                    it_tesc, zlo, zhi, dz, zmode, igm, zvar, agn_on, ita, wta, lsum, lum_linear, tdiff, &
                                    absorbed, yo, total, emis, iq, iu, wq, wu, gam, s_abs, s_esc, dchain, smooth, &
                                    neb, nb, w2, present(zred_mags), tsave, psets(1)%sigma_smooth, &
                                    psets(1)%min_wave_smooth, psets(1)%max_wave_smooth, &
                                    imf, zimf, pm, wsfh, slum, smass, ssp_ops, lbx, mbx, zw, wzbl, &
                                    mags_bar, spec_bar, params_bar, wbl)
            end if
            ! The mass output's cotangent: mass = sum over ages of w_k M_k (M interpolated in log Z with
            ! per-model metallicity, which then adds to logzsol's row)
            if (present(mass_bar) .and. .not. zw .and. .not. imf) then
#ifdef OMP_TARGET
                ! (the SFH weights from the device, where it computed them; with few models the host
                ! did, and the device copy is not current)
                if (zmode .and. .not. host_m) then
                    !$omp target update from(weights)
                end if
#endif
                do m = 1, n_mod
                    do k = 1, nt
                        if (zmode) then
                            wbl(k, m) = wbl(k, m) + mass_bar(m) * ((1.0_wp - dz(m)) * ctx%state%ssp_basis_mass(k, zlo(m)) + &
                                                                   dz(m) * ctx%state%ssp_basis_mass(k, zhi(m)))
                        else
                            wbl(k, m) = wbl(k, m) + mass_bar(m) * ctx%state%ssp_basis_mass(k, 1)
                        end if
                    end do
                    if (zmode .and. present(params_bar)) then
                        w = 0.0_wp
                        do k = 1, nt
                            if (weights(k, 1, m) > SAFE_FLOOR) w = w + weights(k, 1, m) * &
                                (ctx%state%ssp_basis_mass(k, zhi(m)) - ctx%state%ssp_basis_mass(k, zlo(m)))
                        end do
                        params_bar(BGRAD_LOGZSOL, m) = params_bar(BGRAD_LOGZSOL, m) + mass_bar(m) * w * dchain(7, m)
                    end if
                end do
            end if
            if (zw) then
                ! (ssp_weights_z: the mass's cotangent per age and metallicity)
                if (present(mass_bar)) then
                    do m = 1, n_mod
                        do z = 1, nzw
                            wzbl(:, z, m) = wzbl(:, z, m) + mass_bar(m) * ctx%state%ssp_basis_mass(1:nt, z)
                        end do
                    end do
                end if
                if (present(weights_z_bar)) weights_z_bar(1:nt, 1:nzw, 1:n_mod) = wzbl
                wbl = 0.0_wp
            end if
            if (present(weights_bar)) weights_bar(1:nt, 1:n_mod) = wbl
            ! agn_tau: the template weight's slope, 1 / (the interval's width) inside the grid, zero where
            ! agn_indices clamps it
            if (present(params_bar)) then
                do m = 1, n_mod
                    params_bar(BGRAD_AGN_TAU, m) = params_bar(BGRAD_AGN_TAU, m) * agn_tau_slope(ctx, psets(m)%agn_tau)
                end do
            end if
            ! The SFH parameters: dL / d theta = sum over ages of dL / d w_k  d w_k / d theta, with the
            ! weights' tangents in FP64 on the host (one model per thread)
            if (present(params_bar) .and. .not. given_w) then
                allocate(tw(nt, n_mod), tdw(nt, n_mod), tsc(nt, 4, n_mod))
#if defined(_OPENMP)
                !$omp parallel do private(q)
#endif
                do m = 1, n_mod
                    do q = 1, NSFHT
                        call sfh_weights_tangent(ctx, psets(m)%params_core, psets(m)%tage, q, tw(:, m), tdw(:, m), &
                                                 tsc(:, :, m))
                        params_bar(SFHT_ROWS(q), m) = sum(wbl(:, m) * tdw(:, m))
                    end do
                end do
            end if
        end if
        end associate
        ctx%state%batch_out%n = n_mod
        if (present(spec)) call spectra_to_host(ctx%state%batch_out%spec, nspec, n_mod, spec, ctx%state%batch_out%pinned)
#ifdef OMP_TARGET
        !$omp end target data
#endif
#ifdef OMP_TARGET
        !$omp end target data
#endif
#ifdef OMP_TARGET
        !$omp end target data
#endif

        if (present(emlines)) then
            do m = 1, n_mod
                if (neb) then
                    emlines(1:NEMLINE, m) = real(eml(:, m), WP)
                else
                    emlines(1:NEMLINE, m) = 0.0_wp
                end if
            end do
        end if
        do m = 1, n_mod
            if (present(mass)) mass(m) = msum(m)
            if (parts) then
                mass_young(m) = msy(m)
                mass_old(m) = mso(m)
                lbol_young(m) = log10(max(lsy(m), tiny(1.0_wp)))
                lbol_old(m) = log10(max(lso(m), tiny(1.0_wp)))
            end if
            if (present(lbol)) then
                if (lsum(m) > SAFE_FLOOR) then
                    lbol(m) = log10(lsum(m))
                else
                    lbol(m) = 0.0_wp
                end if
            end if
            if (present(mdust)) then
                if (emis(m) .and. s_norm(m) > SAFE_FLOOR) then
                    mdust(m) = 3.21e-3_wp / (4.0_wp * PI) * (s_abs(m) / s_norm(m))
                else
                    mdust(m) = SAFE_FLOOR
                end if
            end if
        end do
    end subroutine compute_csp_batch

    !> Smoothing other than the fast velocity kernel (smooth_velocity = 0: constant width in
    !> wavelength; smoothspec_fast = 0: the exact log-linear velocity convolution), on the host
    !> with compute_csp_scenario's routine (apply_smoothing), one column at a time in FP64: sp
    !> (pixels x models, on the device in device builds) is copied to the host and back.
    subroutine smooth_columns_host(ctx, pset, sp)
        type(opensps_context_t), intent(in) :: ctx
        type(params), intent(in)         :: pset
        real(KP), intent(inout), contiguous :: sp(:,:)
        real(WP), allocatable :: col(:)
        integer :: m, nspec

        nspec = size(sp, 1)
        allocate(col(nspec))
#ifdef OMP_TARGET
        !$omp target update from(sp)
#endif
        do m = 1, size(sp, 2)
            col = real(sp(:, m), WP)
            call apply_smoothing(ctx, ctx%state%spec_lambda(1:nspec), col, pset%sigma_smooth, pset%min_wave_smooth, &
                                 pset%max_wave_smooth)
            sp(:, m) = real(col, KP)
        end do
#ifdef OMP_TARGET
        !$omp target update to(sp)
#endif
    end subroutine smooth_columns_host

    !> The transpose of smooth_columns_host (the host smoothing modes) for the gradients: per model,
    !> apply_smoothing_adjoint on the smoothed spectrum's cotangent (gw(:, N + m)) and the spectrum
    !> before smoothing (tsave), and d L / d sigma (dsg).
    subroutine smooth_columns_host_adjoint(ctx, nspec, n_mod, sigma, minl, maxl, tsave, gw, dsg)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)     :: nspec, n_mod
        real(WP), intent(in)    :: sigma, minl, maxl
        real(KP), intent(inout) :: tsave(:,:), gw(:,:)
        real(WP), intent(out)   :: dsg(n_mod)
        real(WP), allocatable :: col(:), gb(:)
        integer :: m

        allocate(col(nspec), gb(nspec))
#ifdef OMP_TARGET
        !$omp target update from(tsave, gw)
#endif
        do m = 1, n_mod
            col = real(tsave(1:nspec, m), WP)
            gb = real(gw(1:nspec, n_mod + m), WP)
            call apply_smoothing_adjoint(ctx, ctx%state%spec_lambda(1:nspec), col, gb, sigma, minl, maxl, dsg(m))
            gw(1:nspec, n_mod + m) = real(gb, KP)
        end do
#ifdef OMP_TARGET
        !$omp target update to(gw)
#endif
    end subroutine smooth_columns_host_adjoint

    !> Copies the batch's spectra (sp = the batch output, columns 1:n_mod, on the device) to spec
    !> at WP. NVIDIA GPU builds: in chunks of columns on two CUDA streams, each chunk converted on
    !> the host while the next one is in flight; sp's host copy is page-locked
    !> (batch_out_reserve), so each chunk moves at the PCIe rate.
    subroutine spectra_to_host(sp, nspec, n_mod, spec, pinned)
        integer, intent(in)              :: nspec, n_mod
        real(KP), intent(inout), target  :: sp(nspec, n_mod)
        real(WP), intent(inout)          :: spec(:,:)
        logical, intent(in)              :: pinned   !< sp's host copy is page-locked
        integer :: m
#if (defined(OPENSPS_PINNED) || defined(FSPS_PINNED)) && defined(OMP_TARGET)
        !> (The kernels are synchronous, so the data is ready when the copy starts; hardware note:
        !> ~16 MB chunks measured best on the A100 node, PCIe 3 host: large enough to reach the DMA
        !> rate, small enough to overlap most of the conversion)
        integer, parameter :: CHUNK_BYTES = 16 * 1024 * 1024
        integer :: cw, nch, c, q, kb, st
        type(c_ptr) :: dbase

        if (pinned .and. copy_stream(1) == 0) then
            do q = 1, 2
                copy_stream(q) = transfer(opensps_stream_create(), 0_c_intptr_t)
            end do
        end if
        if (pinned .and. all(copy_stream /= 0)) then
            !$omp target data use_device_addr(sp)
            dbase = c_loc(sp)
            !$omp end target data
            kb = storage_size(sp) / 8
            cw = max(1, CHUNK_BYTES / (nspec * kb))
            nch = (n_mod + cw - 1) / cw
            if (issue(1, min(cw, n_mod), 1)) then
                do c = 1, nch
                    if (c < nch) then
                        if (.not. issue(c * cw + 1, min((c + 1) * cw, n_mod), 1 + mod(c, 2))) exit
                    end if
                    q = 1 + mod(c - 1, 2)
                    if (opensps_stream_sync(copy_stream(q)) /= 0) exit
                    do m = (c - 1) * cw + 1, min(c * cw, n_mod)
                        spec(1:nspec, m) = real(sp(:, m), WP)
                    end do
                    if (c == nch) return
                end do
            end if
            ! (a failed copy: the whole update below)
            do q = 1, 2
                st = opensps_stream_sync(copy_stream(q))
            end do
        end if
#endif
#ifdef OMP_TARGET
        !$omp target update from(sp(:, 1:n_mod))
#endif
        do m = 1, n_mod
            spec(1:nspec, m) = real(sp(:, m), WP)
        end do
#if (defined(OPENSPS_PINNED) || defined(FSPS_PINNED)) && defined(OMP_TARGET)
    contains

        !> Starts the DMA of columns c1:c2 of sp to its page-locked host copy on stream q.
        logical function issue(c1, c2, q)
            integer, intent(in) :: c1, c2, q
            integer(c_intptr_t) :: dsrc
            dsrc = transfer(dbase, 0_c_intptr_t) + int(c1 - 1, c_intptr_t) * int(nspec, c_intptr_t) * int(kb, c_intptr_t)
            issue = opensps_memcpy_d2h_async(c_loc(sp(1, c1)), transfer(dsrc, dbase), &
                                             int(nspec, c_size_t) * int(c2 - c1 + 1, c_size_t) * int(kb, c_size_t), &
                                             copy_stream(q)) == 0
        end function issue
#endif
    end subroutine spectra_to_host


    !> Where the last batch's spectra are, for callers that keep working on the device: ptr is
    !> the device address of o%spec (kernel precision, column-major, leading dimension nspec;
    !> the host address in CPU builds), n the batch's models and kp_bytes the
    !> bytes per value (4 in FP32 builds, 8 in FP64). Valid until the next batch call or
    !> opensps_context_remove_from_device; ptr is null (n = 0) before the first batch.
    subroutine batch_spectra_device(ctx, ptr, nspec, n, kp_bytes)
        type(opensps_context_t), intent(inout), target :: ctx
        type(c_ptr), intent(out) :: ptr
        integer, intent(out)     :: nspec, n, kp_bytes
        real(KP), pointer, contiguous :: sp(:,:)

        ptr = c_null_ptr
        nspec = ctx%state%nspec
        n = 0
        kp_bytes = storage_size(1.0_KP) / 8
        if (.not. allocated(ctx%state%batch_out%spec) .or. ctx%state%batch_out%n == 0) return
        n = ctx%state%batch_out%n
        sp => ctx%state%batch_out%spec
#if defined(OMP_TARGET)
        !$omp target data use_device_addr(sp)
        ptr = c_loc(sp)
        !$omp end target data
#else
        ptr = c_loc(sp)
#endif
    end subroutine batch_spectra_device

    !> The last batch's spectra on the host at kernel precision, without conversion: copies them
    !> from the device into the batch output's page-locked host copy (one DMA at the PCIe rate in
    !> NVIDIA GPU builds) and returns its address (column-major, leading dimension nspec, n
    !> models, kp_bytes per value: 4 in FP32 builds, 8 in FP64). For callers that can use the
    !> spectra in kernel precision: it skips compute_csp_batch's conversion to WP into spec.
    !> Valid until the next batch call; ptr is null (n = 0) before the first batch.
    subroutine batch_spectra_host(ctx, ptr, nspec, n, kp_bytes)
        type(opensps_context_t), intent(inout), target :: ctx
        type(c_ptr), intent(out) :: ptr
        integer, intent(out)     :: nspec, n, kp_bytes

        ptr = c_null_ptr
        nspec = ctx%state%nspec
        n = 0
        kp_bytes = storage_size(1.0_KP) / 8
        if (.not. allocated(ctx%state%batch_out%spec) .or. ctx%state%batch_out%n == 0) return
        n = ctx%state%batch_out%n
        if (ctx%on_device) call spectra_to_pinned(ctx%state%batch_out%spec, nspec, size(ctx%state%batch_out%spec, 2), n, &
                                                  ctx%state%batch_out%pinned)
        ptr = c_loc(ctx%state%batch_out%spec)
    end subroutine batch_spectra_host

    !> Copies columns 1:n of the batch output sp (nspec x cap) from the device to its host copy.
    subroutine spectra_to_pinned(sp, nspec, cap, n, pinned)
        integer, intent(in)             :: nspec, cap, n
        real(KP), intent(inout), target :: sp(nspec, cap)
        logical, intent(in)             :: pinned
#if defined(OMP_TARGET)
        !$omp target update from(sp(:, 1:n))
#endif
    end subroutine spectra_to_pinned

    !> Host-side nebular data of a batch: young ages and their grid interval, per-model grid
    !> indices and escape fractions, and per-pixel lists of the lines whose profile reaches it.
    subroutine nebular_setup(ctx, psets, neb, nzb, nb)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: psets(:)
        logical, intent(in)                 :: neb
        integer, intent(in)                 :: nzb
        type(nebular_setup_t), intent(out), target :: nb

        real(WP), parameter :: LINE_PROFILE_CUT = 1.0e-14_wp   ! as add_lines_to_spectrum (CPU)
        integer  :: n_mod, nspec, t, m, i, j, jc, n, p
        integer, allocatable :: j_lo(:), j_hi(:), cnt(:)
        real(WP) :: cut

        n_mod = size(psets)
        nspec = ctx%state%nspec
        nb%nt_neb = 0
        if (neb) nb%nt_neb = max(0, min(find_interval(ctx%state%time_full, ctx%state%nebem_age(NEBNAGE)), &
                                        ctx%state%ntfull))
        nb%cont = neb .and. ctx%add_neb_continuum_val == 1
        nb%lines_in_spec = neb .and. ctx%nebemlineinspec_val == 1
        nb%xrb = (ctx%state%isoc_type == 'bpss') .and. (ctx%add_xrb_emission_val == 1)
        nb%whlylim = ctx%state%whlylim
        n = max(1, nb%nt_neb)
        allocate(nb%ia(n), nb%wa(n), nb%q0(n, nzb))
        allocate(nb%iz(n_mod), nb%wz(n_mod), nb%iu(n_mod), nb%wu(n_mod), nb%fesc(n_mod), nb%fabs(n_mod))
        nb%ia = 1; nb%wa = 0.0_wp
        do t = 1, nb%nt_neb
            call get_grid_indices_weights(ctx%state%nebem_age, ctx%state%time_full(t), NEBNAGE, nb%ia(t), nb%wa(t))
        end do
        allocate(nb%rs(n), nb%rl(n), nb%rt(n))
        nb%nrun = 0
        do t = 1, nb%nt_neb
            if (nb%nrun > 0) then
                if (nb%ia(t) == nb%ia(nb%rs(nb%nrun)) .and. nb%rl(nb%nrun) < NEB_RUN) then
                    nb%rl(nb%nrun) = nb%rl(nb%nrun) + 1
                    nb%rt(t) = nb%nrun
                    cycle
                end if
            end if
            nb%nrun = nb%nrun + 1
            nb%rs(nb%nrun) = t
            nb%rl(nb%nrun) = 1
            nb%rt(t) = nb%nrun
        end do
        if (nb%nrun == 0) then
            nb%nrun = 1
            nb%rs(1) = 1
            nb%rl(1) = 1
            nb%rt(1) = 1
        end if
        ! Steps of wa within each interval (constant on the uniform log-age grid, zero where wa is
        ! clamped), and where a new step starts
        allocate(nb%dwa(n), nb%newr(n))
        nb%dwa = 0.0_wp
        nb%newr = .false.
        do t = 2, nb%nt_neb
            if (nb%ia(t) /= nb%ia(t - 1)) cycle
            nb%dwa(t) = nb%wa(t) - nb%wa(t - 1)
            nb%newr(t) = nb%ia(t - 1) /= nb%ia(max(1, t - 2)) .or. t == 2 .or. nb%dwa(t) /= nb%dwa(t - 1)
        end do
        allocate(nb%cy(n, n_mod), nb%co(n, n_mod), nb%lns(n_mod))
        do m = 1, n_mod
            nb%iz(m) = 1; nb%wz(m) = 0.0_wp; nb%iu(m) = 1; nb%wu(m) = 0.0_wp
            if (neb) then
                call get_grid_indices_weights(ctx%state%nebem_logz, psets(m)%gas_logz, NEBNZ, nb%iz(m), nb%wz(m))
                call get_grid_indices_weights(ctx%state%nebem_logu, psets(m)%gas_logu, NEBNIP, nb%iu(m), nb%wu(m))
            end if
            nb%fesc(m) = max(0.0_wp, min(psets(m)%frac_obrun, 1.0_wp))
            nb%fabs(m) = 1.0_wp - psets(m)%frac_obrun
        end do
        if (neb) then
            allocate(nb%lum(NEMLINE, 2 * n_mod))
        else
            allocate(nb%lum(1, 1))
        end if
        ! Pixel interval of each line wavelength
        allocate(nb%llo(NEMLINE))
        do i = 1, NEMLINE
            jc = 1
            j = nspec + 1
            do while (j - jc > 1)
                p = (jc + j) / 2
                if (ctx%state%nebem_line_pos(i) >= ctx%state%spec_lambda(p)) then
                    jc = p
                else
                    j = p
                end if
            end do
            nb%llo(i) = max(1, min(jc, nspec - 1))
        end do

        ! Line windows (as the CPU add_lines_to_spectrum: from the pixel nearest the line centre
        ! until the profile falls below LINE_PROFILE_CUT of its peak), then per pixel the lines
        ! whose window contains it, in line order (CSR)
        allocate(nb%lptr(nspec + 1))
        nb%lptr = 1
        allocate(nb%lj_lo(NEMLINE), nb%lj_hi(NEMLINE))
        nb%lj_lo = 1
        nb%lj_hi = 0
        if (.not. nb%lines_in_spec) then
            allocate(nb%lid(1)); nb%lid = 1
            call pack_nebular(nb)
            return
        end if
        call prepare_line_gaussians(ctx, psets(1))
        allocate(j_lo(NEMLINE), j_hi(NEMLINE), cnt(nspec))
        cnt = 0
        do i = 1, NEMLINE
            j_lo(i) = 1; j_hi(i) = 0
            jc = max(1, min(find_interval(ctx%state%spec_lambda(1:nspec), ctx%state%nebem_line_pos(i)), nspec))
            j = min(jc + 1, nspec)
            cut = LINE_PROFILE_CUT * max(real(ctx%state%gaussnebarr(jc, i), WP), real(ctx%state%gaussnebarr(j, i), WP))
            if (cut <= 0.0_wp) cycle
            j_hi(i) = j
            j_lo(i) = jc
            do while (j_lo(i) > 1)
                if (ctx%state%gaussnebarr(j_lo(i) - 1, i) <= cut) exit
                j_lo(i) = j_lo(i) - 1
            end do
            do while (j_hi(i) < nspec)
                if (ctx%state%gaussnebarr(j_hi(i) + 1, i) <= cut) exit
                j_hi(i) = j_hi(i) + 1
            end do
            cnt(j_lo(i):j_hi(i)) = cnt(j_lo(i):j_hi(i)) + 1
        end do
        nb%lj_lo = j_lo
        nb%lj_hi = j_hi
        deallocate(j_lo, j_hi)
        do j = 1, nspec
            nb%lptr(j + 1) = nb%lptr(j) + cnt(j)
        end do
        allocate(nb%lid(max(1, nb%lptr(nspec + 1) - 1)))
        cnt = 0
        do i = 1, NEMLINE
            do j = nb%lj_lo(i), nb%lj_hi(i)
                p = nb%lptr(j) + cnt(j)
                nb%lid(p) = i
                cnt(j) = cnt(j) + 1
            end do
        end do
        call pack_nebular(nb)
    end subroutine nebular_setup

    !> Moves the host-filled arrays of nb into one array per type (nb%ipk, rpk, lpk) and points
    !> them there, so the batch uploads them in three transfers.
    subroutine pack_nebular(nb)
        type(nebular_setup_t), intent(inout), target :: nb
        integer :: oi, orr, ol

        allocate(nb%ipk(size(nb%ia) + size(nb%iz) + size(nb%iu) + size(nb%lptr) + size(nb%lid) + size(nb%llo) + &
                        size(nb%rs) + size(nb%rl) + size(nb%rt) + size(nb%lj_lo) + size(nb%lj_hi)))
        allocate(nb%rpk(size(nb%wa) + size(nb%dwa) + size(nb%wz) + size(nb%wu) + size(nb%fesc) + size(nb%fabs)))
        allocate(nb%lpk(size(nb%newr)))
        oi = 0; orr = 0; ol = 0
        call move_i(nb%ia); call move_i(nb%iz); call move_i(nb%iu)
        call move_i(nb%lptr); call move_i(nb%lid); call move_i(nb%llo)
        call move_i(nb%rs); call move_i(nb%rl); call move_i(nb%rt)
        call move_i(nb%lj_lo); call move_i(nb%lj_hi)
        call move_r(nb%wa); call move_r(nb%dwa); call move_r(nb%wz)
        call move_r(nb%wu); call move_r(nb%fesc); call move_r(nb%fabs)
        nb%lpk = nb%newr
        deallocate(nb%newr)
        nb%newr => nb%lpk

    contains

        subroutine move_i(p)
            integer, pointer, contiguous, intent(inout) :: p(:)
            integer :: n

            n = size(p)
            nb%ipk(oi + 1:oi + n) = p
            deallocate(p)
            p => nb%ipk(oi + 1:oi + n)
            oi = oi + n
        end subroutine move_i

        subroutine move_r(p)
            real(WP), pointer, contiguous, intent(inout) :: p(:)
            integer :: n

            n = size(p)
            nb%rpk(orr + 1:orr + n) = p
            deallocate(p)
            p => nb%rpk(orr + 1:orr + n)
            orr = orr + n
        end subroutine move_r
    end subroutine pack_nebular

    !> AGN template interval for pset (agn_tau, not extrapolated); on = AGN emission is added.
    subroutine agn_indices(ctx, pset, all_models, on, ita, wta)
        type(opensps_context_t), intent(in) :: ctx
        type(params), intent(in)         :: pset
        logical, intent(in)              :: all_models   ! also models with fagn = 0
        logical, intent(out)             :: on
        integer, intent(out)             :: ita
        real(WP), intent(out)            :: wta
        integer :: n

        on = ctx%add_agn_dust_val == 1 .and. (all_models .or. (pset%fagn > SAFE_FLOOR .and. pset%fagn > tiny(0.0_wp)))
        ita = 1; wta = 0.0_wp
        if (.not. on) return
        n = size(ctx%state%agndust_tau)
        ita = max(1, min(find_interval(ctx%state%agndust_tau, pset%agn_tau), n - 1))
        wta = (pset%agn_tau - ctx%state%agndust_tau(ita)) / (ctx%state%agndust_tau(ita + 1) - ctx%state%agndust_tau(ita))
        wta = max(0.0_wp, min(wta, 1.0_wp))
    end subroutine agn_indices

    !> d wta / d agn_tau of agn_indices: 1 / (the bracketing interval's width), zero outside the grid.
    real(WP) function agn_tau_slope(ctx, agn_tau)
        type(opensps_context_t), intent(in) :: ctx
        real(WP), intent(in)             :: agn_tau
        integer :: n, ita
        agn_tau_slope = 0.0_wp
        n = size(ctx%state%agndust_tau)
        if (n < 2) return
        if (agn_tau < ctx%state%agndust_tau(1) .or. agn_tau >= ctx%state%agndust_tau(n)) return
        ita = max(1, min(find_interval(ctx%state%agndust_tau, agn_tau), n - 1))
        agn_tau_slope = 1.0_wp / (ctx%state%agndust_tau(ita + 1) - ctx%state%agndust_tau(ita))
    end function agn_tau_slope

    !> Interpolation indices of the Draine & Li grid in q_PAH and U_min.
    subroutine dl07_indices(ctx, pset, iq, wq, iu, wu, gam)
        type(opensps_context_t), intent(in) :: ctx
        type(params), intent(in)         :: pset
        integer, intent(out)             :: iq, iu
        real(WP), intent(out)            :: wq, wu, gam

        iq = 1; iu = 1; wq = 0.0_wp; wu = 0.0_wp; gam = 0.0_wp
        if (ctx%add_dust_emission_val /= 1 .or. .not. associated(ctx%state%qpaharr)) return
        iq = max(1, min(find_interval(ctx%state%qpaharr, pset%duste_qpah), ctx%state%nqpah_dustem - 1))
        wq = (pset%duste_qpah - ctx%state%qpaharr(iq)) / (ctx%state%qpaharr(iq + 1) - ctx%state%qpaharr(iq))
        wq = max(0.0_wp, min(wq, 1.0_wp))
        iu = max(1, min(find_interval(ctx%state%uminarr, pset%duste_umin), ctx%state%numin_dustem - 1))
        wu = (pset%duste_umin - ctx%state%uminarr(iu)) / (ctx%state%uminarr(iu + 1) - ctx%state%uminarr(iu))
        wu = max(0.0_wp, min(wu, 1.0_wp))
        gam = max(0.0_wp, min(pset%duste_gamma, 1.0_wp))
    end subroutine dl07_indices

    !> Draine & Li emission template at one pixel, interpolated in q_PAH and U_min and mixed
    !> with gamma (legacy add_dust).
    function dl07_shape(ctx, i, iq, wq, iu, wu, gam) result(shape)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: i, iq, iu
        real(WP), intent(in)             :: wq, wu, gam
        real(KP) :: shape
        real(WP) :: lo, hi

        lo = (1-wq)*(1-wu) * ctx%state%dustem2_dustem(i, iq,   2*iu - 1) + &
             wq    *(1-wu) * ctx%state%dustem2_dustem(i, iq+1, 2*iu - 1) + &
             wq    *wu     * ctx%state%dustem2_dustem(i, iq+1, 2*(iu+1) - 1) + &
             (1-wq)*wu     * ctx%state%dustem2_dustem(i, iq,   2*(iu+1) - 1)
        hi = (1-wq)*(1-wu) * ctx%state%dustem2_dustem(i, iq,   2*iu) + &
             wq    *(1-wu) * ctx%state%dustem2_dustem(i, iq+1, 2*iu) + &
             wq    *wu     * ctx%state%dustem2_dustem(i, iq+1, 2*(iu+1)) + &
             (1-wq)*wu     * ctx%state%dustem2_dustem(i, iq,   2*(iu+1))
        shape = max(real((1.0_wp - gam) * lo + gam * hi, KP), tiny(1.0_KP))
    end function dl07_shape

    !> Derivatives of dl07_shape with respect to its interpolation weights wq, wu and the mix gamma
    !> (zero where the floor acts).
    subroutine dl07_shape_grad(ctx, i, iq, wq, iu, wu, gam, dsq, dsu, dsg)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: i, iq, iu
        real(WP), intent(in)             :: wq, wu, gam
        real(KP), intent(out)            :: dsq, dsu, dsg
        real(WP) :: lo, hi, lq, hq, lu, hu
        integer  :: kl, kh

        kl = 2 * iu - 1
        kh = 2 * (iu + 1) - 1
        associate (dd => ctx%state%dustem2_dustem)
        lo = (1-wq)*(1-wu) * dd(i, iq, kl) + wq*(1-wu) * dd(i, iq+1, kl) + wq*wu * dd(i, iq+1, kh) + (1-wq)*wu * dd(i, iq, kh)
        hi = (1-wq)*(1-wu) * dd(i, iq, kl+1) + wq*(1-wu) * dd(i, iq+1, kl+1) + wq*wu * dd(i, iq+1, kh+1) + &
             (1-wq)*wu * dd(i, iq, kh+1)
        lq = (1-wu) * (dd(i, iq+1, kl) - dd(i, iq, kl)) + wu * (dd(i, iq+1, kh) - dd(i, iq, kh))
        hq = (1-wu) * (dd(i, iq+1, kl+1) - dd(i, iq, kl+1)) + wu * (dd(i, iq+1, kh+1) - dd(i, iq, kh+1))
        lu = (1-wq) * (dd(i, iq, kh) - dd(i, iq, kl)) + wq * (dd(i, iq+1, kh) - dd(i, iq+1, kl))
        hu = (1-wq) * (dd(i, iq, kh+1) - dd(i, iq, kl+1)) + wq * (dd(i, iq+1, kh+1) - dd(i, iq+1, kl+1))
        end associate
        if ((1.0_wp - gam) * lo + gam * hi > real(tiny(1.0_KP), WP)) then
            dsq = real((1.0_wp - gam) * lq + gam * hq, KP)
            dsu = real((1.0_wp - gam) * lu + gam * hu, KP)
            dsg = real(hi - lo, KP)
        else
            dsq = 0.0_KP; dsu = 0.0_KP; dsg = 0.0_KP
        end if
    end subroutine dl07_shape_grad

    !> dl07_shape and its derivatives in one pass with a single set of table lookups.
    subroutine dl07_shape_and_grad(ctx, i, iq, wq, iu, wu, gam, shape, dsq, dsu, dsg)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: i, iq, iu
        real(WP), intent(in)             :: wq, wu, gam
        real(KP), intent(out)            :: shape, dsq, dsu, dsg
        real(WP) :: lo, hi, lq, hq, lu, hu, val
        integer  :: kl, kh
        real(WP) :: d00, d10, d11, d01, e00, e10, e11, e01

        kl = 2 * iu - 1
        kh = 2 * (iu + 1) - 1
        associate (dd => ctx%state%dustem2_dustem)
        d00 = dd(i, iq, kl)
        d10 = dd(i, iq+1, kl)
        d11 = dd(i, iq+1, kh)
        d01 = dd(i, iq, kh)
        e00 = dd(i, iq, kl+1)
        e10 = dd(i, iq+1, kl+1)
        e11 = dd(i, iq+1, kh+1)
        e01 = dd(i, iq, kh+1)
        end associate

        lo = (1.0_wp - wq)*(1.0_wp - wu) * d00 + wq*(1.0_wp - wu) * d10 + wq*wu * d11 + (1.0_wp - wq)*wu * d01
        hi = (1.0_wp - wq)*(1.0_wp - wu) * e00 + wq*(1.0_wp - wu) * e10 + wq*wu * e11 + (1.0_wp - wq)*wu * e01

        val = (1.0_wp - gam) * lo + gam * hi
        shape = max(real(val, KP), tiny(1.0_KP))

        if (val > real(tiny(1.0_KP), WP)) then
            lq = (1.0_wp - wu) * (d10 - d00) + wu * (d11 - d01)
            hq = (1.0_wp - wu) * (e10 - e00) + wu * (e11 - e01)
            lu = (1.0_wp - wq) * (d01 - d00) + wq * (d11 - d10)
            hu = (1.0_wp - wq) * (e01 - e00) + wq * (e11 - e10)
            dsq = real((1.0_wp - gam) * lq + gam * hq, KP)
            dsu = real((1.0_wp - gam) * lu + gam * hu, KP)
            dsg = real(hi - lo, KP)
        else
            dsq = 0.0_KP; dsu = 0.0_KP; dsg = 0.0_KP
        end if
    end subroutine dl07_shape_and_grad

    !> Diffuse-dust transmission of model pset at pixel i (and its optical depth).
    subroutine diffuse_transmission(ctx, pset, i, curve, tau_d, trans)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(params_core), intent(in)    :: pset
        integer, intent(in)              :: i
        real(WP), intent(out)            :: curve, tau_d, trans

        curve = compute_attenuation_curve_point(ctx%state%spec_lambda(i), i, ctx%dust_type_val, pset, ctx)
        if (ctx%dust_type_val == 3) then
            tau_d = curve
        else
            tau_d = pset%dust2 * curve
        end if
        trans = exp(-tau_d)
    end subroutine diffuse_transmission

    !> Age sums (young and old) and attenuation; absorbed light density for dust emission.

    !> Nebular emission of every model, added to its young and old age sums (yo), as
    !> apply_nebular_emission modifies each SSP of age t <= nt_neb before the age sum of
    !> compute_csp_scenario: (1) the EUV escapes with fraction fesc, (2) the continuum
    !> 10**log(t; Z, U) * Q(t), (3) the lines, whose luminosities (also returned in eml) are summed
    !> over ages first and then spread over their profiles.
    !> q_model: the ionizing photons per age are per model (nb%q0(t, m), the IMF per model) and
    !> the EUV pixels of yo already have the escaping fraction (opensps_ssp_batch); basis is
    !> then not read.
    subroutine add_nebular_batch(ctx, n_mod, nspec, nt, nzb, basis, w2, it_tesc, zlo, zhi, dz, nb, yo, eml, q_model)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_mod, nspec, nt, nzb, it_tesc(:), zlo(:), zhi(:)
        real(KP), intent(in), contiguous :: basis(:,:,:), w2(:,:)
        real(WP), intent(in)             :: dz(:)
        type(nebular_setup_t), intent(inout) :: nb
        real(KP), intent(inout)          :: yo(:,:)
        real(KP), intent(out)            :: eml(:,:)
        logical, intent(in)              :: q_model

        ! The kernels take the components as plain arrays: device code never refers to the
        ! derived type itself, which is mapped only through its components (mapping it whole and
        ! by component, or implicitly through a dummy, conflicts in LLVM's offload runtime)
        ! (the shapes of basis and nb%q0 as they are: with the IMF per model, basis is a placeholder
        ! and q0 has a column per model, not per metallicity)
        call nebular_kernels(ctx, n_mod, nspec, nt, nzb, size(basis, 1), size(basis, 2), size(basis, 3), &
                             size(nb%q0, 2), basis, w2, it_tesc, zlo, zhi, dz, &
                             nb%nt_neb, nb%whlylim, nb%cont, nb%lines_in_spec, nb%xrb, nb%ia, nb%wa, nb%dwa, &
                             nb%newr, nb%q0, nb%iz, nb%wz, nb%iu, nb%wu, nb%fesc, nb%fabs, nb%cy, nb%co, nb%lns, &
                             nb%lum, nb%lptr, nb%lid, yo, eml, q_model)
    end subroutine add_nebular_batch

    !> The kernels of add_nebular_batch, on the components of its nebular_setup_t.
    subroutine nebular_kernels(ctx, n_mod, nspec, nt, nzb, nb1, nb2, nb3, nq0, basis, w2, it_tesc, zlo, zhi, dz, &
                               nb_nt_neb, nb_whlylim, nb_cont, nb_lines_in_spec, nb_xrb, nb_ia, nb_wa, nb_dwa, &
                               nb_newr, nb_q0, nb_iz, nb_wz, nb_iu, nb_wu, nb_fesc, nb_fabs, nb_cy, nb_co, nb_lns, &
                               nb_lum, nb_lptr, nb_lid, yo, eml, q_model)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_mod, nspec, nt, nzb, nb_nt_neb, nb_whlylim
        ! The actual shapes of basis (nb1, nb2, nb3) and of nb_q0's second dimension (nq0): with the
        ! IMF per model basis is a 1 x 1 x 1 placeholder and q0 has n_mod columns while nzb counts
        ! the metallicities. Declared larger than the actual arrays, the dummies made device kernels
        ! map past them: on the A100 (nvfortran) a stale, partially present entry of nb%q0 then
        ! outlived its target data region, and later allocations landing on it were skipped
        ! (wrong spectra) or aborted ("partially present")
        integer, intent(in)              :: nb1, nb2, nb3, nq0
        ! Explicit-shape arguments (no descriptors to map per launch), except nb_lid, whose size is not
        ! known here (the pixels' line lists)
        integer, intent(in)              :: it_tesc(n_mod), zlo(n_mod), zhi(n_mod)
        real(KP), intent(in)             :: basis(nb1, nb2, nb3), w2(nt, 2 * n_mod)
        real(WP), intent(in)             :: dz(n_mod)
        logical, intent(in)              :: nb_cont, nb_lines_in_spec, nb_xrb
        integer, intent(in)              :: nb_ia(max(1, nb_nt_neb)), nb_iz(n_mod), nb_iu(n_mod), nb_lptr(nspec + 1)
        integer, intent(in), contiguous  :: nb_lid(:)
        real(WP), intent(in)             :: nb_wa(max(1, nb_nt_neb)), nb_dwa(max(1, nb_nt_neb)), nb_wz(n_mod)
        real(WP), intent(in)             :: nb_wu(n_mod), nb_fesc(n_mod), nb_fabs(n_mod)
        logical, intent(in)              :: nb_newr(max(1, nb_nt_neb))
        real(WP), intent(inout)          :: nb_q0(max(1, nb_nt_neb), nq0)
        real(KP), intent(inout)          :: nb_cy(max(1, nb_nt_neb), n_mod), nb_co(max(1, nb_nt_neb), n_mod)
        real(WP), intent(inout)          :: nb_lns(n_mod)
        real(KP), intent(inout)          :: nb_lum(NEMLINE, 2 * n_mod)
        real(KP), intent(inout)          :: yo(nspec, 2 * n_mod)
        real(KP), intent(out)            :: eml(NEMLINE, n_mod)
        logical, intent(in)              :: q_model

        integer  :: m, i, j, k, t, nt_neb, whlylim, a, a_prev, p, iz, iu
        real(KP) :: s, c, wt, bk, y, o, wk, v, dzk, f, r, sy, so
        real(WP) :: q, lg, ra, rb, dl, c00, c01, c10, c11, cq, cmax
        real(WP), parameter :: LN10 = log(10.0_wp)
        logical  :: xrb

        nt_neb = nb_nt_neb
        whlylim = nb_whlylim
        xrb = nb_xrb
        if (nt_neb < 1) then
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, n_mod
                do i = 1, NEMLINE
                    eml(i, m) = 0.0_KP
                end do
            end do
            return
        end if

        ! (IMF per model: steps 1 and 2 were done from the operator)
        if (.not. q_model) then

        ! 1. Ionizing photons of each young basis spectrum (the photon-number integral of
        !    process_ionizing_radiation, without the absorbed fraction)
        !    (linear in the spectrum, so interpolated SSPs take the interpolated Q)
        ! (LLVM flang / AMD: one thread per (age, metallicity) with a compensated serial sum over the
        ! pixels, ~120 of them: the result of a parallel do reduction nested in a teams distribute
        ! loop was lost there. nvfortran: gang per (age, metallicity), pixels reduced across its
        ! threads; the serial form was 3-5% slower on the A100 for a single call)
        associate (a_spec_nu => ctx%state%spec_nu)
#if defined(OMP_TARGET) && defined(__flang__)
        !$omp target teams distribute parallel do collapse(2) private(s, c, wt, v, i, y, wk) &
        !$omp map(OMP_MAP_PRESENT_ALLOC: a_spec_nu)
#elif defined(OMP_TARGET)
        !$omp target teams distribute collapse(2) private(s, c) &
        !$omp map(OMP_MAP_PRESENT_ALLOC: a_spec_nu)
#endif
        do iz = 1, nzb
        do t = 1, nt_neb
            s = 0.0_KP
            c = 0.0_KP
            if (whlylim >= 2) then
#if defined(OMP_TARGET) && !defined(__flang__)
                !$omp parallel do reduction(+:s) private(wt, v)
#endif
                do i = 1, whlylim - 1
                    wt = real(0.5_wp * abs(a_spec_nu(i+1) - a_spec_nu(i)), KP)
                    v = wt * (basis(i, t, iz) / real(a_spec_nu(i), KP) + &
                              basis(i+1, t, iz) / real(a_spec_nu(i+1), KP))
#if defined(OMP_TARGET) && defined(__flang__)
                    ! (kahan_add inline: it is not a device routine)
                    if (KP == WP) then
                        s = s + v
                    else
                        y = v - c
                        wk = s + y
                        c = (wk - s) - y
                        s = wk
                    end if
#elif defined(OMP_TARGET)
                    s = s + v
#else
                    if (KP == WP) then
                        s = s + v
                    else
                        call kahan_add(s, c, v)
                    end if
#endif
                end do
                nb_q0(t, iz) = real(s, WP) / H_PLANCK * L_SOL
            else
                nb_q0(t, iz) = 0.0_wp
            end if
        end do
        end do
        end associate

        ! 2. EUV rows: the age sums again, the nebular ages' spectra times the escaping fraction
        if (whlylim > 0) then
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2) private(y, o, bk, k, dzk)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2) private(y, o, bk, k, dzk)
#endif
            do m = 1, n_mod
                do i = 1, whlylim
                    y = 0.0_KP
                    o = 0.0_KP
                    dzk = real(dz(m), KP)
                    do k = 1, nt
                        if (nzb > 1) then
                            bk = (1.0_KP - dzk) * basis(i, k, zlo(m)) + dzk * basis(i, k, zhi(m))
                        else
                            bk = basis(i, k, 1)
                        end if
                        if (k <= nt_neb) bk = bk * real(nb_fesc(m), KP)
                        if (k <= it_tesc(m)) then
                            y = y + w2(k, m) * bk
                        else
                            o = o + w2(k, n_mod + m) * bk
                        end if
                    end do
                    yo(i, m) = y
                    yo(i, n_mod + m) = o
                end do
            end do
        end if

        end if

        ! 3. Continuum: per (pixel, model), sum over the nebular ages t of C(t) 10**lg(t), with
        !    C = SFH weight * Q (per model, young or old) and lg the log grid interpolated in Z, U
        !    (FP64) and age. Within one CLOUDY age interval [ra, rb], lg = ra + wa (rb - ra) and on
        !    the uniform log-age grid wa advances by a constant step, so 10**lg is a geometric
        !    progression: per interval the first age's value and the ratio, per age one multiply
        !    and two adds. Kernel precision for the per-age work, with the rules of
        !    doc/MIXED_PRECISION.md: C (up to ~1e53, rule 5) is stored divided by its largest value
        !    S per model, and ln S joins the exponent; the exponents (~100, rule 3) are formed in
        !    FP64 and reduced there before the exponential (exp_kp). FP64 remains in the grid
        !    interpolation and the argument reduction, a few operations per interval; on GPUs with
        !    slow FP64 (1/32-1/64 of FP32, most consumer and workstation cards) the FP64
        !    exponentials this replaces were most of the batch's time (R9700: 44%).
        if (nb_cont) then
            ! C = SFH weight * Q per (age, model), stored as C / max_t C in KP with ln(max_t C) in
            ! nb_lns: one model per thread, the first pass for the maximum
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do private(t, q, wk, cq, cmax)
#elif defined(_OPENMP)
            !$omp parallel do private(t, q, wk, cq, cmax)
#endif
            do m = 1, n_mod
                cmax = 0.0_wp
                do t = 1, nt_neb
                    if (q_model) then
                        q = abs(nb_q0(t, m) * nb_fabs(m))
                    else
                        q = abs(((1.0_wp - dz(m)) * nb_q0(t, zlo(m)) + dz(m) * nb_q0(t, zhi(m))) * nb_fabs(m))
                    end if
                    if (t <= it_tesc(m)) then
                        wk = w2(t, m)
                    else
                        wk = w2(t, n_mod + m)
                    end if
                    if (wk /= 0.0_KP .and. q > SAFE_FLOOR) cmax = max(cmax, abs(real(wk, WP) * q))
                end do
                nb_lns(m) = 0.0_wp
                if (cmax > 0.0_wp) nb_lns(m) = log(cmax)
                do t = 1, nt_neb
                    if (q_model) then
                        q = abs(nb_q0(t, m) * nb_fabs(m))
                    else
                        q = abs(((1.0_wp - dz(m)) * nb_q0(t, zlo(m)) + dz(m) * nb_q0(t, zhi(m))) * nb_fabs(m))
                    end if
                    if (t <= it_tesc(m)) then
                        wk = w2(t, m)
                    else
                        wk = w2(t, n_mod + m)
                    end if
                    nb_cy(t, m) = 0.0_KP
                    nb_co(t, m) = 0.0_KP
                    if (wk /= 0.0_KP .and. q > SAFE_FLOOR) then
                        cq = real(wk, WP) * q / cmax
                        if (t <= it_tesc(m)) then
                            nb_cy(t, m) = real(cq, KP)
                        else
                            nb_co(t, m) = real(cq, KP)
                        end if
                    end if
                end do
            end do

#if defined(OMP_TARGET) && defined(__flang__)
            if (xrb) then
                call nebular_continuum_tiled(n_mod, nspec, size(ctx%state%xnebem_cont, 1), nt_neb, &
                                             ctx%state%xnebem_cont, nb_ia, nb_wa, nb_dwa, nb_newr, &
                                             nb_cy, nb_co, nb_lns, nb_iz, nb_wz, nb_iu, nb_wu, yo)
            else
                call nebular_continuum_tiled(n_mod, nspec, size(ctx%state%nebem_cont, 1), nt_neb, &
                                             ctx%state%nebem_cont, nb_ia, nb_wa, nb_dwa, nb_newr, &
                                             nb_cy, nb_co, nb_lns, nb_iz, nb_wz, nb_iu, nb_wu, yo)
            end if
#else
            associate (a_nebem_cont => ctx%state%nebem_cont, a_xnebem_cont => ctx%state%xnebem_cont)
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2) private(y, o, sy, so, ra, rb, dl, f, r, &
            !$omp& a, a_prev, t, iz, iu, c00, c01, c10, c11) &
            !$omp& map(OMP_MAP_PRESENT_ALLOC: a_nebem_cont, a_xnebem_cont)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2) private(y, o, sy, so, ra, rb, dl, f, r, &
            !$omp& a, a_prev, t, iz, iu, c00, c01, c10, c11)
#endif
            do m = 1, n_mod
                do j = 1, nspec
                    y = 0.0_KP
                    o = 0.0_KP
                    sy = 0.0_KP
                    so = 0.0_KP
                    a_prev = -1
                    ra = 0.0_wp; rb = 0.0_wp
                    dl = 0.0_wp; f = 0.0_KP; r = 1.0_KP
                    iz = nb_iz(m)
                    iu = nb_iu(m)
                    ! As interpolate_zu_slice_point, written out: with nvfortran 25.1 (A100) a call
                    ! with the 4-d assumed-shape grid per interval and thread cost more than the
                    ! arithmetic (descriptor set-up in local memory); 53 -> 41 ms at N = 10000
                    c00 = (1.0_wp - nb_wz(m)) * (1.0_wp - nb_wu(m))
                    c01 = (1.0_wp - nb_wz(m)) * (         nb_wu(m))
                    c10 = (         nb_wz(m)) * (1.0_wp - nb_wu(m))
                    c11 = (         nb_wz(m)) * (         nb_wu(m))
                    do t = 1, nt_neb
                        a = nb_ia(t)
                        if (a /= a_prev) then
                            ! Close the previous interval
                            y = y + sy
                            o = o + so
                            sy = 0.0_KP
                            so = 0.0_KP
                            ! The interval only advances: its lower end is the previous upper end
                            if (a == a_prev + 1) then
                                ra = rb
                            else
                                if (xrb) then
                                    ra = c00 * a_xnebem_cont(j, iz, a, iu) + &
                                         c01 * a_xnebem_cont(j, iz, a, iu + 1) + &
                                         c10 * a_xnebem_cont(j, iz + 1, a, iu) + &
                                         c11 * a_xnebem_cont(j, iz + 1, a, iu + 1)
                                else
                                    ra = c00 * a_nebem_cont(j, iz, a, iu) + &
                                         c01 * a_nebem_cont(j, iz, a, iu + 1) + &
                                         c10 * a_nebem_cont(j, iz + 1, a, iu) + &
                                         c11 * a_nebem_cont(j, iz + 1, a, iu + 1)
                                end if
                            end if
                            if (xrb) then
                                rb = c00 * a_xnebem_cont(j, iz, a + 1, iu) + &
                                     c01 * a_xnebem_cont(j, iz, a + 1, iu + 1) + &
                                     c10 * a_xnebem_cont(j, iz + 1, a + 1, iu) + &
                                     c11 * a_xnebem_cont(j, iz + 1, a + 1, iu + 1)
                            else
                                rb = c00 * a_nebem_cont(j, iz, a + 1, iu) + &
                                     c01 * a_nebem_cont(j, iz, a + 1, iu + 1) + &
                                     c10 * a_nebem_cont(j, iz + 1, a + 1, iu) + &
                                     c11 * a_nebem_cont(j, iz + 1, a + 1, iu + 1)
                            end if
                            a_prev = a
                            dl = (rb - ra) * LN10
                            f = exp_kp(ra * LN10 + nb_wa(t) * dl + nb_lns(m))
                        else
                            if (nb_newr(t)) r = exp_kp(nb_dwa(t) * dl)
                            f = f * r
                        end if
                        sy = sy + nb_cy(t, m) * f
                        so = so + nb_co(t, m) * f
                    end do
                    y = y + sy
                    o = o + so
                    yo(j, m) = yo(j, m) + y
                    yo(j, n_mod + m) = yo(j, n_mod + m) + o
                end do
            end do
            end associate
#endif
        end if

        ! 4. Line luminosities per (line, model), young and old: sum over ages of the weight times
        !    10**(log grid interpolated) * Q
#if defined(OMP_TARGET) && defined(__flang__)
        if (nb_cont) then
            ! The same sum as the continuum's, with the line in place of the pixel: C(t) 10**lg(t) per
            ! line, the weights C (young, old) and ln S as computed above for the continuum
            !$omp target teams distribute parallel do collapse(2)
            do m = 1, 2 * n_mod
                do i = 1, NEMLINE
                    nb_lum(i, m) = 0.0_KP
                end do
            end do
            if (xrb) then
                call nebular_continuum_tiled(n_mod, NEMLINE, NEMLINE, nt_neb, ctx%state%xnebem_line, nb_ia, nb_wa, &
                                             nb_dwa, nb_newr, nb_cy, nb_co, nb_lns, nb_iz, nb_wz, nb_iu, nb_wu, nb_lum)
            else
                call nebular_continuum_tiled(n_mod, NEMLINE, NEMLINE, nt_neb, ctx%state%nebem_line, nb_ia, nb_wa, &
                                             nb_dwa, nb_newr, nb_cy, nb_co, nb_lns, nb_iz, nb_wz, nb_iu, nb_wu, nb_lum)
            end if
            !$omp target teams distribute parallel do collapse(2)
            do m = 1, n_mod
                do i = 1, NEMLINE
                    eml(i, m) = nb_lum(i, m) + nb_lum(i, n_mod + m)
                end do
            end do
        else
#endif
        associate (a_nebem_line => ctx%state%nebem_line, a_xnebem_line => ctx%state%xnebem_line)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(y, o, wk, q, lg, ra, rb, a, t) &
        !$omp map(OMP_MAP_PRESENT_ALLOC: a_nebem_line, a_xnebem_line)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(y, o, wk, q, lg, ra, rb, a, t)
#endif
        do m = 1, n_mod
            do i = 1, NEMLINE
                y = 0.0_KP
                o = 0.0_KP
                do t = 1, nt_neb
                    if (t <= it_tesc(m)) then
                        wk = w2(t, m)
                    else
                        wk = w2(t, n_mod + m)
                    end if
                    if (wk == 0.0_KP) cycle
                    if (q_model) then
                        q = abs(nb_q0(t, m) * nb_fabs(m))
                    else
                        q = abs(((1.0_wp - dz(m)) * nb_q0(t, zlo(m)) + dz(m) * nb_q0(t, zhi(m))) * nb_fabs(m))
                    end if
                    if (q <= SAFE_FLOOR) cycle
                    a = nb_ia(t)
                    if (xrb) then
                        ra = zu_point(a_xnebem_line, i, a, nb_iz(m), nb_iu(m), nb_wz(m), nb_wu(m))
                        rb = zu_point(a_xnebem_line, i, a + 1, nb_iz(m), nb_iu(m), nb_wz(m), nb_wu(m))
                    else
                        ra = zu_point(a_nebem_line, i, a, nb_iz(m), nb_iu(m), nb_wz(m), nb_wu(m))
                        rb = zu_point(a_nebem_line, i, a + 1, nb_iz(m), nb_iu(m), nb_wz(m), nb_wu(m))
                    end if
                    lg = (1.0_wp - nb_wa(t)) * ra + nb_wa(t) * rb
                    if (t <= it_tesc(m)) then
                        y = y + wk * real((10.0_wp**lg) * q, KP)
                    else
                        o = o + wk * real((10.0_wp**lg) * q, KP)
                    end if
                end do
                nb_lum(i, m) = y
                nb_lum(i, n_mod + m) = o
                eml(i, m) = y + o
            end do
        end do
        end associate
#if defined(OMP_TARGET) && defined(__flang__)
        end if
#endif

        ! 5. Line profiles: per pixel, the lines whose profile reaches it
        if (nb_lines_in_spec) then
            associate (a_gaussnebarr => ctx%state%gaussnebarr)
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2) private(s, p, i) &
            !$omp map(OMP_MAP_PRESENT_ALLOC: a_gaussnebarr)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2) private(s, p, i)
#endif
            do m = 1, 2 * n_mod
                do j = 1, nspec
                    s = 0.0_KP
                    do p = nb_lptr(j), nb_lptr(j + 1) - 1
                        i = nb_lid(p)
                        s = s + a_gaussnebarr(j, i) * nb_lum(i, m)
                    end do
                    yo(j, m) = yo(j, m) + s
                end do
            end do
            end associate
        end if
    end subroutine nebular_kernels

#if defined(OMP_TARGET) && defined(__flang__)
    !> The continuum kernel of nebular_kernels for LLVM flang / AMD: explicit-shape arguments (no
    !> array descriptors in the kernel), and each thread does PT pixels (strided, so the loads are
    !> coalesced) of one model, which shares the per-age loads (CLOUDY age interval, weights, step
    !> flags, SFH weight * Q) among them: with one pixel per thread the loop was ~90% address
    !> arithmetic and loads. The arithmetic per pixel is the one of the generic kernel, in the same
    !> order, so the results are identical bit for bit.
    subroutine nebular_continuum_tiled(n_mod, nspec, ntab, nt_neb, tab, ia, wa, dwa, newr, cy, co, lns, iz, wz, iu, wu, yo)
        use, intrinsic :: iso_c_binding, only: c_loc, c_null_ptr
        integer, intent(in)     :: n_mod, nspec, ntab, nt_neb
        integer, intent(in), target :: ia(nt_neb), iz(n_mod), iu(n_mod)
        real(WP), intent(in), target :: tab(ntab, NEBNZ, NEBNAGE, NEBNIP), wa(nt_neb), dwa(nt_neb)
        real(KP), intent(in), target :: cy(nt_neb, n_mod), co(nt_neb, n_mod)
        real(WP), intent(in), target :: lns(n_mod), wz(n_mod), wu(n_mod)
        logical, intent(in)     :: newr(nt_neb)
        real(KP), intent(inout), target :: yo(nspec, 2 * n_mod)

        integer, parameter :: PT = 4
        real(WP), parameter :: LN10 = log(10.0_wp)
        integer  :: m, ip, npt
#if defined(OPENSPS_HIP)
        integer, allocatable, target :: newr_int(:)
#endif

#if defined(OPENSPS_HIP)
        allocate(newr_int(nt_neb))
        newr_int = merge(1, 0, newr)
        !$omp target data map(to: newr_int) use_device_addr(tab, ia, wa, dwa, newr_int, cy, co, lns, iz, wz, iu, wu, yo)
        if (KP == WP) then
            call opensps_hip_nebular_continuum_dp(n_mod, nspec, ntab, nt_neb, &
                c_loc(tab), c_loc(ia), c_loc(wa), c_loc(dwa), c_loc(newr_int), &
                c_loc(cy), c_loc(co), c_loc(lns), c_loc(iz), c_loc(wz), c_loc(iu), c_loc(wu), &
                c_loc(yo), c_null_ptr)
        else
            call opensps_hip_nebular_continuum_sp(n_mod, nspec, ntab, nt_neb, &
                c_loc(tab), c_loc(ia), c_loc(wa), c_loc(dwa), c_loc(newr_int), &
                c_loc(cy), c_loc(co), c_loc(lns), c_loc(iz), c_loc(wz), c_loc(iu), c_loc(wu), &
                c_loc(yo), c_null_ptr)
        end if
        !$omp end target data
        deallocate(newr_int)
        return
#endif

        npt = (nspec + PT - 1) / PT
        ! (The small work arrays are declared in a block of the loop body: flang maps `private` arrays,
        ! one host-device copy each way per launch, a block-local array it does not)
        !$omp target teams distribute parallel do collapse(2) &
        !$omp map(OMP_MAP_PRESENT_ALLOC: tab)
        do m = 1, n_mod
            do ip = 1, npt
              block
                integer  :: p, t, a, a_prev, jz, ju, jj(PT)
                real(KP) :: y(PT), o(PT), sy(PT), so(PT), f(PT), r(PT), cyt, cot
                real(WP) :: ra(PT), rb(PT), dl(PT), c00, c01, c10, c11, wat, lsm
                logical  :: valid(PT)
                do p = 1, PT
                    jj(p) = min(ip + (p - 1) * npt, nspec)
                    valid(p) = ip + (p - 1) * npt <= nspec
                    y(p) = 0.0_KP
                    o(p) = 0.0_KP
                    sy(p) = 0.0_KP
                    so(p) = 0.0_KP
                    ra(p) = 0.0_wp
                    rb(p) = 0.0_wp
                    dl(p) = 0.0_wp
                    f(p) = 0.0_KP
                    r(p) = 1.0_KP
                end do
                lsm = lns(m)
                a_prev = -1
                jz = iz(m)
                ju = iu(m)
                c00 = (1.0_wp - wz(m)) * (1.0_wp - wu(m))
                c01 = (1.0_wp - wz(m)) * (         wu(m))
                c10 = (         wz(m)) * (1.0_wp - wu(m))
                c11 = (         wz(m)) * (         wu(m))
                do t = 1, nt_neb
                    a = ia(t)
                    wat = wa(t)
                    cyt = cy(t, m)
                    cot = co(t, m)
                    if (a /= a_prev) then
                        do p = 1, PT
                            ! Close the previous interval
                            y(p) = y(p) + sy(p)
                            o(p) = o(p) + so(p)
                            sy(p) = 0.0_KP
                            so(p) = 0.0_KP
                            ! The interval only advances: its lower end is the previous upper end
                            if (a == a_prev + 1) then
                                ra(p) = rb(p)
                            else
                                ra(p) = c00 * tab(jj(p), jz, a, ju) + c01 * tab(jj(p), jz, a, ju + 1) + &
                                        c10 * tab(jj(p), jz + 1, a, ju) + c11 * tab(jj(p), jz + 1, a, ju + 1)
                            end if
                            rb(p) = c00 * tab(jj(p), jz, a + 1, ju) + c01 * tab(jj(p), jz, a + 1, ju + 1) + &
                                    c10 * tab(jj(p), jz + 1, a + 1, ju) + c11 * tab(jj(p), jz + 1, a + 1, ju + 1)
                            dl(p) = (rb(p) - ra(p)) * LN10
                            f(p) = exp_kp(ra(p) * LN10 + wat * dl(p) + lsm)
                        end do
                        a_prev = a
                    else
                        do p = 1, PT
                            if (newr(t)) r(p) = exp_kp(dwa(t) * dl(p))
                            f(p) = f(p) * r(p)
                        end do
                    end if
                    do p = 1, PT
                        sy(p) = sy(p) + cyt * f(p)
                        so(p) = so(p) + cot * f(p)
                    end do
                end do
                do p = 1, PT
                    y(p) = y(p) + sy(p)
                    o(p) = o(p) + so(p)
                    if (valid(p)) then
                        yo(jj(p), m) = yo(jj(p), m) + y(p)
                        yo(jj(p), n_mod + m) = yo(jj(p), n_mod + m) + o(p)
                    end if
                end do
              end block
            end do
        end do
    end subroutine nebular_continuum_tiled
#endif

    !> Line luminosities after dust (legacy add_dust): young lines through
    !> the birth cloud, all through the diffuse dust at the line wavelength (the transmission
    !> interpolated between pixels, at KP as there). eml = attenuated luminosities; s_line =
    !> intrinsic - attenuated, per model.
    subroutine attenuate_lines(ctx, pbuf, n_mod, nb_lum, nb_llo, eml, s_line)
        type(opensps_context_t), intent(in)  :: ctx
        integer, intent(in)               :: n_mod
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        real(KP), intent(in)              :: nb_lum(NEMLINE, 2 * n_mod)   ! line luminosities (nebular_setup_t)
        integer, intent(in)               :: nb_llo(NEMLINE)              ! line pixel intervals
        real(KP), intent(out)             :: eml(NEMLINE, n_mod)
        real(WP), intent(out)             :: s_line(n_mod)

        integer  :: m, i
        real(WP) :: si, sa, ay, ao
        type(params_core), pointer :: pl
#if defined(OMP_TARGET) && defined(__flang__)
        real(WP), allocatable :: lw(:,:,:)

        ! (LLVM flang: the result of a parallel do reduction nested in a teams distribute loop is lost
        ! there, which left s_line wrong on AMD GPUs. One thread per (line, model), the lines' terms
        ! summed per model afterwards in line order: the serial sums, bit for bit)
        allocate(lw(2, NEMLINE, n_mod))
        !$omp target data map(alloc: lw)
        !$omp target teams distribute parallel do collapse(2) private(pl, ay, ao) &
        !$omp map(OMP_MAP_PRESENT_ALLOC: ctx%state%spec_lambda, ctx%state%nebem_line_pos)
        do m = 1, n_mod
            do i = 1, NEMLINE
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
                call line_attenuated(ctx, pl, i, nb_llo(i), nb_lum(i, m), nb_lum(i, n_mod + m), ay, ao)
                lw(1, i, m) = real(nb_lum(i, m) + nb_lum(i, n_mod + m), WP)   ! (summed at KP, as before)
                lw(2, i, m) = ay + ao
                eml(i, m) = real(ay, KP) + real(ao, KP)
            end do
        end do
        !$omp target teams distribute parallel do private(i, si, sa)
        do m = 1, n_mod
            si = 0.0_wp
            sa = 0.0_wp
            do i = 1, NEMLINE
                si = si + lw(1, i, m)
                sa = sa + lw(2, i, m)
            end do
            s_line(m) = si - sa
        end do
        !$omp end target data
#else
#if defined(OMP_TARGET)
        !$omp target teams distribute private(si, sa, pl) &
        !$omp map(OMP_MAP_PRESENT_ALLOC: ctx%state%spec_lambda, ctx%state%nebem_line_pos)
#endif
        do m = 1, n_mod
#if !defined(__flang__)
            pl => pbuf(m)%params_core
#else
            call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
            si = 0.0_wp
            sa = 0.0_wp
#if defined(OMP_TARGET)
            !$omp parallel do reduction(+:si, sa) private(ay, ao)
#endif
            do i = 1, NEMLINE
                call line_attenuated(ctx, pl, i, nb_llo(i), nb_lum(i, m), nb_lum(i, n_mod + m), ay, ao)
                si = si + (nb_lum(i, m) + nb_lum(i, n_mod + m))
                sa = sa + (ay + ao)
                eml(i, m) = real(ay, KP) + real(ao, KP)
            end do
            s_line(m) = si - sa
        end do
#endif
    end subroutine attenuate_lines

    !> One line of attenuate_lines: its young and old luminosities after the birth cloud and the
    !> diffuse dust (T_d interpolated between the line's two pixels).
    subroutine line_attenuated(ctx, pl, i, lo, ly, lo_, ay, ao)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(params_core), intent(in)    :: pl
        integer, intent(in)              :: i, lo
        real(KP), intent(in)             :: ly, lo_
        real(WP), intent(out)            :: ay, ao
        real(WP) :: t_lo, t_hi, curve, tau_d, trans, td, birth, fob, fnd, pos

        fob = pl%frac_obrun
        fnd = pl%frac_nodust
        pos = ctx%state%nebem_line_pos(i)
        if (1.0_wp - fnd <= SAFE_FLOOR) then
            td = 1.0_wp
        else
            call diffuse_transmission(ctx, pl, lo, curve, tau_d, trans)
            t_lo = real(real(trans, KP), WP)
            call diffuse_transmission(ctx, pl, lo + 1, curve, tau_d, trans)
            t_hi = real(real(trans, KP), WP)
            td = t_lo + (t_hi - t_lo) / (ctx%state%spec_lambda(lo + 1) - ctx%state%spec_lambda(lo)) * &
                 (pos - ctx%state%spec_lambda(lo))
        end if
        if (td /= td) td = 1.0_wp
        if (pl%dust1 <= SAFE_FLOOR .or. 1.0_wp - fob <= SAFE_FLOOR) then
            birth = 1.0_wp
        else
            birth = exp(-pl%dust1 * (pos / V_BAND_ANGSTROMS)**pl%dust1_index)
        end if
        ay = ly * (birth * (1.0_wp - fob) + fob)
        ao = lo_
        if (1.0_wp - fnd > SAFE_FLOOR) then
            ay = ay * (td * (1.0_wp - fnd) + fnd)
            ao = ao * (td * (1.0_wp - fnd) + fnd)
        end if
    end subroutine line_attenuated

    !> exp(x) in kernel precision for an FP64 argument of any size. x = n ln 2 + r (|r| <= ln 2 / 2)
    !> is reduced in FP64, so r keeps its digits; exp(r) is taken in KP and scaled by 2**n. The
    !> result is good to ~1 ulp of KP, where rounding x itself to FP32 would cost |x| eps (rule 3
    !> of doc/MIXED_PRECISION.md: the nebular exponents are ~100). The FP64 work is three
    !> operations instead of an FP64 exponential. Results below the smallest normal KP flush to 0.
    !> 2**n by binary exponentiation (exact for powers of two; |n| < 2**10, at most ten
    !> squarings): `scale`, `ieee_scalb` and `transfer` are calls into the Fortran runtime with LLVM
    !> flang, which device code cannot make, and `2.0**n` inlines a general integer-power routine.
    pure function exp_kp(x) result(e)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: x
        real(KP) :: e
        real(WP), parameter :: LN2 = log(2.0_wp), LOG2E = 1.0_wp / log(2.0_wp)
        real(WP) :: xc
        real(KP) :: b, p2
        integer  :: n, k

        if (kind(1.0_KP) == kind(1.0_wp)) then
            ! FP64 kernels: the plain exponential (the reduction serves FP32 only)
            e = real(exp(x), KP)
            return
        end if
        xc = max(-1.0e4_wp, min(1.0e4_wp, x))
        n = nint(xc * LOG2E)
        if (n < minexponent(1.0_KP)) then
            e = 0.0_KP
        else if (n >= maxexponent(1.0_KP)) then
            e = huge(1.0_KP)
        else
            b = merge(0.5_KP, 2.0_KP, n < 0)
            k = abs(n)
            p2 = 1.0_KP
            do while (k > 0)
                if (iand(k, 1) == 1) p2 = p2 * b
                b = b * b
                k = ishft(k, -1)
            end do
            e = exp(real(xc - n * LN2, KP)) * p2
        end if
    end function exp_kp

    !> 1 - t for t = exp(-x) in kernel precision without cancellation (rule 2 of
    !> doc/MIXED_PRECISION.md): for |x| < 1, Kahan's form (1 - t) x / (-ln t), good to a few ulp,
    !> where 1 - t alone loses eps / |x| (6e-5 at x = 1e-3 in FP32). x may be negative (curves
    !> that dip below zero, e.g. Reddy et al. in the near-IR).
    pure function one_minus_trans_kp(x, t) result(y)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(KP), intent(in) :: x, t
        real(KP) :: y

        if (t == 1.0_KP) then
            y = x
        else if (abs(x) < 1.0_KP) then
            y = (1.0_KP - t) * x / (-log(t))
        else
            y = 1.0_KP - t
        end if
    end function one_minus_trans_kp

    !> CLOUDY grid value at (i, age a) bilinear in (Z, U), as interpolate_zu_slice_point.
    pure function zu_point(grid, i, a, iz, iu, wz, wu) result(val)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: grid(:,:,:,:)
        integer, intent(in)  :: i, a, iz, iu
        real(WP), intent(in) :: wz, wu
        real(WP) :: val

        val = (1.0_wp - wz) * (1.0_wp - wu) * grid(i, iz,     a, iu    ) + &
              (1.0_wp - wz) * (         wu) * grid(i, iz,     a, iu + 1) + &
              (         wz) * (1.0_wp - wu) * grid(i, iz + 1, a, iu    ) + &
              (         wz) * (         wu) * grid(i, iz + 1, a, iu + 1)
    end function zu_point

    !> AGN dust emission (legacy agn_dust): fagn L_bol times the agn_tau template,
    !> attenuated by the model's diffuse dust; after the IGM transmission, as there.
    subroutine add_agn_batch(ctx, pbuf, n_mod, nspec, agn_on, ita, wta, lsum, tdiff, total)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_mod, nspec
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        integer, intent(in)              :: ita(n_mod)
        logical, intent(in)              :: agn_on(n_mod)
        real(WP), intent(in)             :: wta(n_mod), lsum(n_mod)
        real(KP), intent(in)             :: tdiff(:,:)   ! the diffuse transmission (attenuate_batch)
        real(KP), intent(inout)          :: total(nspec, n_mod)

        integer  :: m, i
        real(WP) :: lagn, tmpl, lb
        type(params_core), pointer :: pl
#if defined(OMP_TARGET) && defined(__flang__)
        real(WP), allocatable :: lag(:)

        ! LLVM flang / AMD: the AGN luminosity (a log10 and a power in FP64, one per model) is
        ! computed once per model, not per pixel (FP64 is 1/32 of FP32 on the 780M)
        allocate(lag(n_mod))
        !$omp target data map(alloc: lag)
        !$omp target teams distribute parallel do private(lb, pl)
        do m = 1, n_mod
#if !defined(__flang__)
            pl => pbuf(m)%params_core
#else
            call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
            lag(m) = 0.0_wp
            if (agn_on(m)) then
                if (lsum(m) > SAFE_FLOOR) then
                    lb = log10(lsum(m))
                else
                    lb = 0.0_wp
                end if
                lag(m) = (10.0_wp**lb) * pl%fagn
            end if
        end do
        associate (a_agndust_spec => ctx%state%agndust_spec)
        !$omp target teams distribute parallel do collapse(2) private(tmpl) &
        !$omp map(OMP_MAP_PRESENT_ALLOC: a_agndust_spec)
        do m = 1, n_mod
            do i = 1, nspec
                if (agn_on(m)) then
                    tmpl = (1.0_wp - wta(m)) * a_agndust_spec(i, ita(m)) + &
                           wta(m) * a_agndust_spec(i, ita(m) + 1)
                    tmpl = tmpl * real(tdiff(i, m), WP)
                    total(i, m) = total(i, m) + lag(m) * tmpl
                end if
            end do
        end do
        end associate
        !$omp end target data
        return
#endif

        associate (a_agndust_spec => ctx%state%agndust_spec)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(lagn, tmpl, lb, pl) &
        !$omp map(OMP_MAP_PRESENT_ALLOC: a_agndust_spec)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(lagn, tmpl, lb, pl)
#endif
        do m = 1, n_mod
            do i = 1, nspec
#if !defined(__flang__)
                pl => pbuf(m)%params_core
#else
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                if (agn_on(m)) then
                    ! L_bol through its logarithm, as compute_csp_scenario stores it
                    if (lsum(m) > SAFE_FLOOR) then
                        lb = log10(lsum(m))
                    else
                        lb = 0.0_wp
                    end if
                    lagn = (10.0_wp**lb) * pl%fagn
                    tmpl = (1.0_wp - wta(m)) * a_agndust_spec(i, ita(m)) + &
                           wta(m) * a_agndust_spec(i, ita(m) + 1)
                    ! exp(-dust2 curve) (exp(-curve) for dust_type 3): the transmission of attenuate_batch,
                    ! kept at KP as opensps_dust keeps it
                    tmpl = tmpl * real(tdiff(i, m), WP)
                    total(i, m) = total(i, m) + lagn * tmpl
                end if
            end do
        end do
        end associate
    end subroutine add_agn_batch

    !> Interval and weight of log Z = logzsol in the context's metallicities, as
    !> compute_interpolated_ssp (the weight is not clamped: outside the grid it extrapolates).
    subroutine metallicity_interval(ctx, logzsol, zlo, dz)
        type(opensps_context_t), intent(in) :: ctx
        real(WP), intent(in)             :: logzsol
        integer, intent(out)             :: zlo
        real(WP), intent(out)            :: dz
        integer :: nz

        nz = ctx%state%nz
        zlo = max(1, min(find_interval(log10(ctx%state%zlegend(1:nz) / ctx%state%zsol), logzsol), nz - 1))
        dz = (logzsol - log10(ctx%state%zlegend(zlo) / ctx%state%zsol)) / &
             (log10(ctx%state%zlegend(zlo + 1) / ctx%state%zsol) - log10(ctx%state%zlegend(zlo) / ctx%state%zsol))
    end subroutine metallicity_interval

    !> Age sums with weights over ages and metallicities: yo = sum over z of basis(:, :, z) .
    !> wz(:, z, :), young ages (<= it_tesc) into column m and old ones into N + m, summed in
    !> compute_csp_scenario's order (metallicities outer, ages inner) at kernel precision.
    subroutine age_sums_w(nspec, nt, nz, n_mod, basis, wz, it_tesc, yo)
        integer, intent(in)              :: nspec, nt, nz, n_mod, it_tesc(:)
        real(KP), intent(in), contiguous :: basis(:,:,:), wz(:,:,:)
        real(KP), intent(out)            :: yo(:,:)
        integer  :: m, i, k, z
        real(KP) :: wk, y, o
#if !defined(OMP_TARGET)
        integer  :: col

        ! CPU builds: one pass over the pixels (stride 1) per weight, as compute_csp_scenario
        do m = 1, 2 * n_mod
            do i = 1, nspec
                yo(i, m) = 0.0_KP
            end do
        end do
        ! (each model writes only its own two columns)
        !$omp parallel do private(z, k, i, wk, col)
        do m = 1, n_mod
            do z = 1, nz
                do k = 1, nt
                    wk = wz(k, z, m)
                    if (wk == 0.0_KP) cycle
                    col = merge(m, n_mod + m, k <= it_tesc(m))
                    do i = 1, nspec
                        yo(i, col) = yo(i, col) + wk * basis(i, k, z)
                    end do
                end do
            end do
        end do
#else
        ! One thread per (pixel, model)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, z, wk, y, o)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, z, wk, y, o)
#endif
        do m = 1, n_mod
            do i = 1, nspec
                y = 0.0_KP
                o = 0.0_KP
                do z = 1, nz
                    do k = 1, nt
                        wk = wz(k, z, m)
                        if (k <= it_tesc(m)) then
                            y = y + wk * basis(i, k, z)
                        else
                            o = o + wk * basis(i, k, z)
                        end if
                    end do
                end do
                yo(i, m) = y
                yo(i, n_mod + m) = o
            end do
        end do
#endif
    end subroutine age_sums_w

    !> Age sums with per-model metallicity: each model's basis is (1 - dz) basis(:, :, zlo) +
    !> dz basis(:, :, zhi), formed at kernel precision term by term (portable kernel; also used in
    !> cuBLAS builds). One pass over the ages (young or old per model), 4 models per thread.
    subroutine age_sums_z(nspec, nt, nz, n_mod, basis, w2, w2flat, it_tesc, zlo, zhi, dz, yo)
        ! Explicit-shape dummies (no array descriptors in the kernel)
        integer, intent(in)              :: nspec, nt, nz, n_mod, it_tesc(n_mod), zlo(n_mod), zhi(n_mod)
        real(KP), intent(in)             :: basis(nspec, nt, nz), w2(nt, 2 * n_mod), w2flat(nt * 2 * n_mod)
        real(WP), intent(in)             :: dz(n_mod)
        real(KP), intent(out)            :: yo(nspec, 2 * n_mod)
#if defined(OMP_TARGET) && defined(__flang__)
        ! LLVM flang / AMD. The models are visited sorted by their metallicity interval, so that a tile
        ! of MT consecutive models usually shares one pair of basis slabs; each thread does PT pixels
        ! (strided, so that the loads are coalesced) of the tile, which shares the per-model weight
        ! loads among the pixels. The per-model offsets are formed once, outside the age loop (inside
        ! it, address arithmetic and loads were ~97% of the instructions). The arithmetic per model
        ! is unchanged (same results bit for bit).
        ! Tile of PT pixels x MT models per thread. FP64 on the 780M: 4 pixels spill registers (890 ms at
        ! N = 10^4; 658 ms with 2; 8 models x 1-4 pixels is the best range, 16 or more models 6-8 s)
        integer, parameter :: MT = 8, PT = merge(2, 4, KP == WP)
        integer  :: m, mq, ip, npt, key, nzk
        integer, allocatable :: zord(:), starts(:)

        nzk = maxval(zhi(1:n_mod))
        allocate(zord(n_mod), starts(nzk + 1))
        starts = 0
        do m = 1, n_mod
            starts(zlo(m) + 1) = starts(zlo(m) + 1) + 1
        end do
        starts(1) = 1
        do key = 2, nzk + 1
            starts(key) = starts(key) + starts(key - 1)
        end do
        do m = 1, n_mod
            zord(starts(zlo(m))) = m
            starts(zlo(m)) = starts(zlo(m)) + 1
        end do

        npt = (nspec + PT - 1) / PT
        ! (Work arrays in a block of the loop body: flang maps `private` arrays, a copy each way per launch)
        !$omp target teams distribute parallel do collapse(2) map(to: zord)
        do mq = 1, (n_mod + MT - 1) / MT
            do ip = 1, npt
              block
                integer  :: i, k, l, p, mm(MT), ii(PT), zl, zh, it(MT), oy(MT), oo(MT)
                real(KP) :: b_lo(PT), b_hi(PT), bk, wy, y(PT, MT), o(PT, MT), d(MT)
                logical  :: same
                do l = 1, MT
                    mm(l) = zord(min((mq - 1) * MT + l, n_mod))
                    d(l) = real(dz(mm(l)), KP)
                    it(l) = it_tesc(mm(l))
                    oy(l) = (mm(l) - 1) * nt          ! offsets of the model's young and old weight columns
                    oo(l) = (n_mod + mm(l) - 1) * nt
                end do
                do p = 1, PT
                    ii(p) = min(ip + (p - 1) * npt, nspec)
                end do
                y = 0.0_KP
                o = 0.0_KP
                same = .true.
                do l = 2, MT
                    if (zlo(mm(l)) /= zlo(mm(1)) .or. zhi(mm(l)) /= zhi(mm(1))) same = .false.
                end do
                if (same) then
                    zl = zlo(mm(1))
                    zh = zhi(mm(1))
                    do k = 1, nt
                        do p = 1, PT
                            b_lo(p) = basis(ii(p), k, zl)
                            b_hi(p) = basis(ii(p), k, zh)
                        end do
                        do l = 1, MT
                            if (k <= it(l)) then
                                wy = w2flat(oy(l) + k)
                                do p = 1, PT
                                    bk = (1.0_KP - d(l)) * b_lo(p) + d(l) * b_hi(p)
                                    y(p, l) = y(p, l) + wy * bk
                                end do
                            else
                                wy = w2flat(oo(l) + k)
                                do p = 1, PT
                                    bk = (1.0_KP - d(l)) * b_lo(p) + d(l) * b_hi(p)
                                    o(p, l) = o(p, l) + wy * bk
                                end do
                            end if
                        end do
                    end do
                else
                    do k = 1, nt
                        do l = 1, MT
                            if (k <= it(l)) then
                                wy = w2flat(oy(l) + k)
                                do p = 1, PT
                                    bk = (1.0_KP - d(l)) * basis(ii(p), k, zlo(mm(l))) + d(l) * basis(ii(p), k, zhi(mm(l)))
                                    y(p, l) = y(p, l) + wy * bk
                                end do
                            else
                                wy = w2flat(oo(l) + k)
                                do p = 1, PT
                                    bk = (1.0_KP - d(l)) * basis(ii(p), k, zlo(mm(l))) + d(l) * basis(ii(p), k, zhi(mm(l)))
                                    o(p, l) = o(p, l) + wy * bk
                                end do
                            end if
                        end do
                    end do
                end if
                do l = 1, MT
                    do p = 1, PT
                        yo(ii(p), mm(l)) = y(p, l)
                        yo(ii(p), n_mod + mm(l)) = o(p, l)
                    end do
                end do
              end block
            end do
        end do
#else
        integer, parameter :: MT = 4
        integer  :: mq, i, k, l, mm(MT)
        real(KP) :: bk, y(MT), o(MT), d(MT)

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, l, bk, mm, y, o, d)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, l, bk, mm, y, o, d)
#endif
        do mq = 1, (n_mod + MT - 1) / MT
            do i = 1, nspec
                do l = 1, MT
                    mm(l) = min((mq - 1) * MT + l, n_mod)
                    d(l) = real(dz(mm(l)), KP)
                    y(l) = 0.0_KP
                    o(l) = 0.0_KP
                end do
                do k = 1, nt
                    do l = 1, MT
                        bk = (1.0_KP - d(l)) * basis(i, k, zlo(mm(l))) + d(l) * basis(i, k, zhi(mm(l)))
                        if (k <= it_tesc(mm(l))) then
                            y(l) = y(l) + w2(k, mm(l)) * bk
                        else
                            o(l) = o(l) + w2(k, n_mod + mm(l)) * bk
                        end if
                    end do
                end do
                do l = 1, MT
                    yo(i, mm(l)) = y(l)
                    yo(i, n_mod + mm(l)) = o(l)
                end do
            end do
        end do
#endif
    end subroutine age_sums_z

    !> IGM transmission per model (redshift and igm_factor per model), as compute_igm_transmission:
    !> the optical depth at KP, capped below the wavelength of its maximum, then exp(-tau factor).
    !> tau is work space (pixels x models).
    subroutine apply_igm_z(ctx, pbuf, n_mod, nspec, tau, total)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_mod, nspec
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        real(KP), intent(inout)          :: tau(nspec, n_mod), total(nspec, n_mod)
        integer  :: m, j, jmax, jtop
        real(WP) :: tmax
        real(WP), allocatable :: ta(:, :), tb(:, :)
        type(params_core), pointer :: pl

        ! Only the pixels below Lyman-alpha (the wavelengths are ascending) have an optical depth: the
        ! others keep their flux (exp(0) = 1), as before. The optical depth is the sum of IGM_NTERM terms,
        ! a wavelength factor (a table over these pixels, formed here) times a redshift factor (per model)
        jtop = 0
        do j = 1, nspec
            if (ctx%state%spec_lambda(j) < 1215.67_wp) jtop = j
        end do
        if (jtop == 0) return
        allocate(ta(IGM_NTERM, jtop), tb(IGM_NTERM, n_mod))
        ta = 0.0_wp
        tb = 0.0_wp

#if defined(OMP_TARGET)
        !$omp target data map(alloc: ta, tb) map(OMP_MAP_PRESENT_ALLOC: ctx%state%spec_lambda)
#endif
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do
#elif defined(_OPENMP)
        !$omp parallel do
#endif
        do j = 1, jtop
            call igm_tau_pixel_terms(ctx%state%spec_lambda(j), ta(:, j))
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(pl)
#elif defined(_OPENMP)
        !$omp parallel do private(pl)
#endif
        do m = 1, n_mod
#if !defined(__flang__)
            pl => pbuf(m)%params_core
#else
            call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
            if (pl%zred > SAFE_FLOOR) call igm_tau_z_terms(1.0_wp + pl%zred, tb(:, m))
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(pl)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(pl)
#endif
        do m = 1, n_mod
            do j = 1, jtop
#if !defined(__flang__)
                pl => pbuf(m)%params_core
#else
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                if (pl%zred > SAFE_FLOOR) tau(j, m) = real(ta(1, j) * tb(1, m) + ta(2, j) * tb(2, m) + &
                                                           ta(3, j) * tb(3, m) + ta(4, j) * tb(4, m) + &
                                                           ta(5, j) * tb(5, m) + ta(6, j) * tb(6, m), KP)
            end do
        end do
        ! The short-wavelength cap, one thread per model
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(tmax, jmax, j, pl)
#elif defined(_OPENMP)
        !$omp parallel do private(tmax, jmax, j, pl)
#endif
        do m = 1, n_mod
#if !defined(__flang__)
            pl => pbuf(m)%params_core
#else
            call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
            if (pl%zred > SAFE_FLOOR) then
                tmax = 0.0_wp
                jmax = 1
                do j = 1, jtop
                    if (tau(j, m) > tmax) then
                        tmax = tau(j, m)
                        jmax = j
                    end if
                end do
                if (jmax > 1) then
                    do j = 1, jmax
                        tau(j, m) = real(tmax, KP)
                    end do
                end if
            end if
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(pl)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(pl)
#endif
        do m = 1, n_mod
            do j = 1, jtop
#if !defined(__flang__)
                pl => pbuf(m)%params_core
#else
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                if (pl%zred > SAFE_FLOOR) &
                    total(j, m) = total(j, m) * real(exp(-real(tau(j, m), WP) * pl%igm_factor), KP)
            end do
        end do
#if defined(OMP_TARGET)
        !$omp end target data
#endif
    end subroutine apply_igm_z

    !> Common IGM transmission.
    subroutine apply_igm(n_mod, nspec, igm_t, total)
        integer, intent(in)     :: n_mod, nspec
        real(KP), intent(in)    :: igm_t(nspec)
        real(KP), intent(inout) :: total(nspec, n_mod)
        integer :: m, i

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do i = 1, nspec
                total(i, m) = total(i, m) * igm_t(i)
            end do
        end do
    end subroutine apply_igm

    !> Age sums of all models: yo = basis . w2 (pixels x ages . ages x 2N), plain sums over the ages
    !> at kernel precision as in compute_csp_scenario.
    subroutine age_sums(nspec, nt, n_mod, basis, w2, it_tesc, yo)
        integer, intent(in)                      :: nspec, nt, n_mod, it_tesc(:)
        real(KP), intent(in), contiguous, target :: basis(:,:), w2(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
#if defined(OPENSPS_ACCEL_BLAS)
        call accel_blas_init()
#ifdef OMP_TARGET
        !$omp target data use_device_addr(basis, w2, yo)
#endif
        call accel_blas_gemm(ACCEL_BLAS_OP_N, ACCEL_BLAS_OP_N, nspec, 2 * n_mod, nt, 1.0_KP, &
                             c_loc(basis), nspec, c_loc(w2), nt, 0.0_KP, c_loc(yo), nspec)
#ifdef OMP_TARGET
        !$omp end target data
#endif
#elif !defined(OMP_TARGET)
        ! CPU builds: per model, one pass over the pixels (stride 1, vectorized) for each age with
        ! a weight, as compute_csp_scenario; the register tile below would pad a batch of fewer
        ! than MT models with copies (8x the work at N = 1). The same sums in the same order:
        ! ages in order, the young ones (<= it_tesc) into column m, the old into N + m, so the
        ! results equal the tiled kernel's bit for bit.
        integer  :: m, k, i, col
        real(KP) :: wk

        do m = 1, 2 * n_mod
            do i = 1, nspec
                yo(i, m) = 0.0_KP
            end do
        end do
        ! (each model writes only its own two columns)
        !$omp parallel do private(k, i, wk, col)
        do m = 1, n_mod
            do k = 1, nt
                wk = w2(k, m) + w2(k, n_mod + m)   ! one of the two is zero
                if (wk == 0.0_KP) cycle
                col = merge(m, n_mod + m, k <= it_tesc(m))
                do i = 1, nspec
                    yo(i, col) = yo(i, col) + wk * basis(i, k)
                end do
            end do
        end do
#else
        ! Register tile: each thread does PT pixels (strided by npt, so each load is coalesced
        ! across the threads) x MT models, PT + 2 MT loads per PT * MT multiply-adds.
        ! Hardware note: MT = 8, PT = 4 were measured on the A100 (nvfortran 25.1, FP32: 64
        ! accumulators per thread, no spills); retune on other GPUs, where registers per thread
        ! and the load / FMA balance differ. Each age is young or old for a given model: one
        ! running sum per (pixel, model), stored as the young sum at the model's last young age
        ! (it_tesc) and restarted for the old ages. Same sums, in the same order, as two separate
        ! products.
        integer, parameter :: MT = 8, PT = 4
        integer  :: mq, ip, npt, i, k, l, p, mm(MT), it(MT), ii(PT)
        real(KP) :: bk(PT), wk(MT), acc(PT, MT), yy(PT, MT)

        npt = (nspec + PT - 1) / PT
#if defined(OMP_TARGET)
        ! (teams loop: 15% faster than teams distribute parallel do with nvfortran 25.1 on the A100)
        !$omp target teams loop collapse(2) private(mm, it, ii, bk, wk, acc, yy, k, l, p, i)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(mm, it, ii, bk, wk, acc, yy, k, l, p, i)
#endif
        do mq = 1, (n_mod + MT - 1) / MT
            do ip = 1, npt
                do l = 1, MT
                    ! Models of this thread (clamped at the end; duplicates are written twice, same value)
                    mm(l) = min((mq - 1) * MT + l, n_mod)
                    it(l) = it_tesc(mm(l))
                end do
                do p = 1, PT
                    ii(p) = min(ip + (p - 1) * npt, nspec)
                end do
                acc = 0.0_KP
                yy = 0.0_KP
                do k = 1, nt
                    do p = 1, PT
                        bk(p) = basis(ii(p), k)
                    end do
                    do l = 1, MT
                        wk(l) = w2(k, mm(l)) + w2(k, n_mod + mm(l))   ! one of the two is zero
                    end do
                    do l = 1, MT
                        do p = 1, PT
                            acc(p, l) = acc(p, l) + wk(l) * bk(p)
                        end do
                        if (k == it(l)) then
                            do p = 1, PT
                                yy(p, l) = acc(p, l)
                                acc(p, l) = 0.0_KP
                            end do
                        end if
                    end do
                end do
                do l = 1, MT
                    do p = 1, PT
                        yo(ii(p), mm(l)) = yy(p, l)
                        yo(ii(p), n_mod + mm(l)) = acc(p, l)
                    end do
                end do
            end do
        end do
#endif
    end subroutine age_sums

    !> Grid-only factors of the diffuse attenuation curve, per pixel (rule 4 of
    !> doc/MIXED_PRECISION.md: formed in FP64 once per context and dust type, stored in kernel
    !> precision, kept in the context: ctx%state%att_tab, att_ltab). Every
    !> dust type with a closed form is, per model m,
    !>     curve(i, m) = (T0(i) + s1 T1(i) + s2 T2(i) + s3 T3(i)) * exp(e L(i)),  L = ln(lambda / lambda_V)
    !> with model scalars (s1, s2, s3, e): power law (0): T0 = 1, e = dust_index; Cardelli et al. (1):
    !> bilinear in 1/R_V and the bump strength uvb (the NUV continuity term too), s = (1/R_V, uvb,
    !> uvb/R_V); Calzetti (2), Reddy (6): T0; Kriek & Conroy (4): linear in delta once the power law
    !> is factored out, s1 = e = delta. The tables come from the curve functions themselves at a few
    !> parameter values (no second copy of their constants). Types 3 and 5 are table lookups and are
    !> read directly. L stays in FP64 (it is exponentiated, rule 3).
    subroutine attenuation_tables_cached(ctx, nspec)
        type(opensps_context_t), intent(inout) :: ctx
        integer, intent(in) :: nspec
        integer :: i, dt
        logical :: fresh

        dt = ctx%dust_type_val
        fresh = .not. allocated(ctx%state%att_tab)
        if (fresh) allocate(ctx%state%att_tab(nspec, 0:3), ctx%state%att_ltab(nspec))
        if (fresh .or. ctx%state%att_dust_type /= dt) then
            ! On the host, once per dust type: nspec x (up to 4) curve evaluations, ~1 ms
            do i = 1, nspec
                call attenuation_table_row(ctx, nspec, i, dt, ctx%state%att_tab, ctx%state%att_ltab)
            end do
            ctx%state%att_dust_type = dt
        else if (.not. ctx%on_device .or. ctx%state%att_on_device) then
            return   ! current, and on the device whenever the context is
        end if
#if defined(OMP_TARGET)
        if (ctx%on_device) then
            if (ctx%state%att_on_device) then
                !$omp target update to(ctx%state%att_tab, ctx%state%att_ltab)
            else
                !$omp target enter data map(to: ctx%state%att_tab, ctx%state%att_ltab)
                ctx%state%att_on_device = .true.
            end if
        end if
#endif
    end subroutine attenuation_tables_cached

    !> One pixel of attenuation_tables.
    subroutine attenuation_table_row(ctx, nspec, i, dt, ttab, ltab)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)   :: nspec, i, dt
        real(KP), intent(inout) :: ttab(nspec, 0:3)
        real(WP), intent(inout) :: ltab(nspec)
        real(WP), parameter :: R_BIG = 1.0e30_wp   ! 1/R_V = 1e-30: the terms without 1/R_V
        type(params_core) :: pc
        real(WP) :: lam, f00, f10, f01, f11

            lam = ctx%state%spec_lambda(i)
            ltab(i) = log(lam / V_BAND_ANGSTROMS)
            ttab(i, :) = 0.0_KP
            select case (dt)
            case (0)
                ttab(i, 0) = 1.0_KP
            case (1)
                pc%mwr = R_BIG; pc%uvb = 0.0_wp
                f00 = compute_attenuation_curve_point(lam, i, dt, pc, ctx)
                pc%mwr = 1.0_wp
                f10 = compute_attenuation_curve_point(lam, i, dt, pc, ctx)
                pc%mwr = R_BIG; pc%uvb = 1.0_wp
                f01 = compute_attenuation_curve_point(lam, i, dt, pc, ctx)
                pc%mwr = 1.0_wp
                f11 = compute_attenuation_curve_point(lam, i, dt, pc, ctx)
                ttab(i, 0) = real(f00, KP)
                ttab(i, 1) = real(f10 - f00, KP)
                ttab(i, 2) = real(f01 - f00, KP)
                ttab(i, 3) = real(f11 - f10 - f01 + f00, KP)
            case (2, 6)
                ttab(i, 0) = real(compute_attenuation_curve_point(lam, i, dt, pc, ctx), KP)
            case (4)
                pc%dust_index = 0.0_wp
                f00 = compute_attenuation_curve_point(lam, i, dt, pc, ctx)
                pc%dust_index = 1.0_wp
                f10 = compute_attenuation_curve_point(lam, i, dt, pc, ctx) * (V_BAND_ANGSTROMS / lam)
                ttab(i, 0) = real(f00, KP)
                ttab(i, 1) = real(f10 - f00, KP)
            end select
    end subroutine attenuation_table_row

    !> Attenuation and absorbed light per (pixel, model). For models with dust emission the age
    !> sums are then replaced by what the emission stages need: yo(:, m) = diffuse transmission
    !> (stored at KP, as opensps_dust stores it), yo(:, N + m) = Draine & Li template.
    !> Kernel precision throughout (doc/MIXED_PRECISION.md): the curve from the grid-only tables of
    !> attenuation_tables, the power laws (lambda / lambda_V)**e as exp_kp(e L) with e L formed in
    !> FP64, the transmissions as KP exponentials, and the absorbed light from 1 - T without
    !> cancellation (one_minus_trans_kp). On GPUs with slow FP64 the FP64 form of this kernel was
    !> a fifth of the batch's time (R9700).
    subroutine attenuate_batch(ctx, pbuf, n_mod, nspec, iq, iu, wq, wu, gam, emis, agn_on, yo, total, absorbed, tdiff)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_mod, nspec
        ! Explicit-shape arguments (no array descriptors to map and re-read in every launch), except tdiff:
        ! allocated 1 x 1 without AGN
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        integer, intent(in)              :: iq(n_mod), iu(n_mod)
        real(WP), intent(in)             :: wq(n_mod), wu(n_mod), gam(n_mod)
        logical, intent(in)              :: emis(n_mod), agn_on(n_mod)
        real(KP), intent(inout)          :: yo(nspec, 2 * n_mod)   ! in: young (1:N), old (N+1:2N)
        real(KP), intent(out)            :: total(nspec, n_mod), absorbed(nspec, n_mod)
        real(KP), intent(inout)          :: tdiff(:,:)   ! diffuse transmission of AGN models

        integer  :: m, i, dt
        real(KP) :: young, old, tau_b, tau_o, tau_d, trans_birth, trans_old, trans_d, frac_obrun, frac_nodust
        real(KP) :: one_m_tb, one_m_to, one_m_td, a_young, spec_sum, curve
        type(params_core), pointer :: pl

        dt = ctx%dust_type_val
        ! The grid-only curve tables of the context (attenuation_tables_cached, built once per dust type)
        associate (ttab => ctx%state%att_tab, ltab => ctx%state%att_ltab)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(young, old, tau_b, tau_o, &
        !$omp& tau_d, trans_birth, trans_old, trans_d, frac_obrun, frac_nodust, one_m_tb, one_m_to, &
        !$omp& one_m_td, a_young, spec_sum, curve, pl) &
        !$omp& map(OMP_MAP_PRESENT_ALLOC: ctx%state%spec_lambda, ctx%state%dustem2_dustem, ttab, ltab)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(young, old, tau_b, tau_o, &
        !$omp& tau_d, trans_birth, trans_old, trans_d, frac_obrun, frac_nodust, one_m_tb, one_m_to, &
        !$omp& one_m_td, a_young, spec_sum, curve, pl)
#endif
        do m = 1, n_mod
            do i = 1, nspec
                young = yo(i, m)
                old = yo(i, n_mod + m)

#if !defined(__flang__)
                pl => pbuf(m)%params_core
#else
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                frac_obrun = real(pl%frac_obrun, KP)
                frac_nodust = real(pl%frac_nodust, KP)
                call pixel_transmissions(ctx, pl, nspec, i, dt, ttab, ltab, tau_d, trans_d, tau_b, trans_birth, &
                                         tau_o, trans_old, curve)
                if (agn_on(m)) tdiff(i, m) = trans_d

                spec_sum = (young * trans_birth * (1.0_KP - frac_obrun) + young * frac_obrun) + old * trans_old
                if (1.0_wp - pl%frac_nodust <= SAFE_FLOOR) then
                    total(i, m) = spec_sum
                else
                    total(i, m) = spec_sum * (trans_d * (1.0_KP - frac_nodust) + frac_nodust)
                end if

                if (emis(m)) then
                    one_m_tb = one_minus_trans_kp(tau_b, trans_birth)
                    one_m_to = one_minus_trans_kp(tau_o, trans_old)
                    if (1.0_wp - pl%frac_nodust <= SAFE_FLOOR) then
                        one_m_td = 0.0_KP
                    else
                        one_m_td = (1.0_KP - frac_nodust) * one_minus_trans_kp(tau_d, trans_d)
                    end if
                    a_young = trans_birth * (1.0_KP - frac_obrun) + frac_obrun
                    absorbed(i, m) = young * ((1.0_KP - frac_obrun) * one_m_tb + a_young * one_m_td) + &
                                     old * (one_m_to + trans_old * one_m_td)
                    yo(i, m) = trans_d
                    yo(i, n_mod + m) = dl07_shape(ctx, i, iq(m), wq(m), iu(m), wu(m), gam(m))
                else
                    absorbed(i, m) = 0.0_KP
                end if
            end do
        end do
        end associate
    end subroutine attenuate_batch

    !> The three transmissions of one pixel and model (attenuate_batch, and its adjoint in
    !> batch_backward): diffuse (tau_d, trans_d), birth cloud (tau_b, trans_birth) and the old
    !> stars' extra screen (tau_o, trans_old), at kernel precision from the grid-only tables
    !> (ttab, ltab: attenuation_tables_cached).
    subroutine pixel_transmissions(ctx, pl, nspec, i, dt, ttab, ltab, tau_d, trans_d, tau_b, trans_birth, tau_o, trans_old, &
                                   curve)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(params_core), intent(in)    :: pl
        integer, intent(in)              :: nspec, i, dt
        real(KP), intent(in)             :: ttab(nspec, 0:3)
        real(WP), intent(in)             :: ltab(nspec)
        real(KP), intent(out)            :: tau_d, trans_d, tau_b, trans_birth, tau_o, trans_old
        real(KP), intent(out)            :: curve   ! the diffuse curve (tau_d / dust2, or tau_d for dust type 3)
        real(KP) :: s1, s2, s3
        real(WP) :: ed

        ! Diffuse curve and optical depth (as diffuse_transmission)
        if (dt == 3 .or. dt == 5) then
            curve = real(compute_attenuation_curve_point(ctx%state%spec_lambda(i), i, dt, pl, ctx), KP)
        else
            s1 = 0.0_KP; s2 = 0.0_KP; s3 = 0.0_KP; ed = 0.0_wp
            if (dt == 0) then
                ed = pl%dust_index
            else if (dt == 1) then
                s1 = 1.0_KP / real(pl%mwr, KP)
                s2 = real(pl%uvb, KP)
                s3 = s2 * s1
            else if (dt == 4) then
                ed = pl%dust_index
                s1 = real(ed, KP)
            end if
            curve = ttab(i, 0) + s1 * ttab(i, 1) + s2 * ttab(i, 2) + s3 * ttab(i, 3)
            if (ed /= 0.0_wp) curve = curve * exp_kp(ed * ltab(i))
        end if
        if (dt == 3) then
            tau_d = curve
        else
            tau_d = real(pl%dust2, KP) * curve
        end if
        trans_d = exp(-tau_d)
        if (pl%dust1 <= SAFE_FLOOR .or. 1.0_wp - pl%frac_obrun <= SAFE_FLOOR) then
            tau_b = 0.0_KP
            trans_birth = 1.0_KP
        else
            tau_b = real(pl%dust1, KP) * exp_kp(pl%dust1_index * ltab(i))
            trans_birth = exp(-tau_b)
        end if
        if (abs(pl%dust3) <= SAFE_FLOOR) then
            tau_o = 0.0_KP
            trans_old = 1.0_KP
        else
            tau_o = real(pl%dust3, KP) * curve
            trans_old = exp(-tau_o)
        end if
    end subroutine pixel_transmissions

    !> Diffuse transmission at pixel i only (skips birth-cloud and old-stars screens).
    subroutine pixel_diffuse_transmission(ctx, pl, nspec, i, dt, ttab, ltab, trans_d)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(params_core), intent(in)    :: pl
        integer, intent(in)              :: nspec, i, dt
        real(KP), intent(in)             :: ttab(nspec, 0:3)
        real(WP), intent(in)             :: ltab(nspec)
        real(KP), intent(out)            :: trans_d
        real(KP) :: s1, s2, s3, curve, tau_d
        real(WP) :: ed

        if (dt == 3 .or. dt == 5) then
            curve = real(compute_attenuation_curve_point(ctx%state%spec_lambda(i), i, dt, pl, ctx), KP)
        else
            s1 = 0.0_KP; s2 = 0.0_KP; s3 = 0.0_KP; ed = 0.0_wp
            if (dt == 0) then
                ed = pl%dust_index
            else if (dt == 1) then
                s1 = 1.0_KP / real(pl%mwr, KP)
                s2 = real(pl%uvb, KP)
                s3 = s2 * s1
            else if (dt == 4) then
                ed = pl%dust_index
                s1 = real(ed, KP)
            end if
            curve = ttab(i, 0) + s1 * ttab(i, 1) + s2 * ttab(i, 2) + s3 * ttab(i, 3)
            if (ed /= 0.0_wp) curve = curve * exp_kp(ed * ltab(i))
        end if
        if (dt == 3) then
            tau_d = curve
        else
            tau_d = real(pl%dust2, KP) * curve
        end if
        trans_d = exp(-tau_d)
    end subroutine pixel_diffuse_transmission

    !> Per model: absorbed luminosity, template normalization and escaped template fraction.
    subroutine dust_emission_reductions(n_mod, nspec, emis, wt, yo, absorbed, s_abs, s_norm, s_esc)
        integer, intent(in)              :: n_mod, nspec
        logical, intent(in)              :: emis(n_mod)
        real(KP), intent(in)             :: wt(nspec), yo(nspec, 2 * n_mod), absorbed(nspec, n_mod)
        real(WP), intent(out)            :: s_abs(n_mod), s_norm(n_mod), s_esc(n_mod)

        integer  :: m, i
        real(KP) :: sa, sn, se, ca, cn, ce
#if defined(OMP_TARGET) && defined(__flang__)
        !> Partial sums per model: each takes every NCH-th pixel (consecutive threads, consecutive pixels)
        integer, parameter :: NCH = 64
        real(KP), allocatable :: part(:,:,:)
        integer  :: c

        ! LLVM flang rewrites the inner loop of a teams loop to a serial loop: one thread summed all
        ! pixels of a model (33 us per model on the 780M, ~10x its memory time). Here (model, chunk)
        ! pairs are threads and a second kernel adds the NCH partial sums
        allocate(part(3, NCH, n_mod))
        !$omp target data map(alloc: part)
        !$omp target teams distribute parallel do collapse(2) private(sa, sn, se, i)
        do m = 1, n_mod
            do c = 1, NCH
                sa = 0.0_KP; sn = 0.0_KP; se = 0.0_KP
                if (emis(m)) then
                    do i = c, nspec - 1, NCH
                        sa = sa + wt(i) * (absorbed(i, m) + absorbed(i + 1, m))
                        sn = sn + wt(i) * (yo(i, n_mod + m) + yo(i + 1, n_mod + m))
                        se = se + wt(i) * (yo(i, n_mod + m) * yo(i, m) + yo(i + 1, n_mod + m) * yo(i + 1, m))
                    end do
                end if
                part(1, c, m) = sa
                part(2, c, m) = sn
                part(3, c, m) = se
            end do
        end do
        !$omp target teams distribute parallel do private(sa, sn, se, c)
        do m = 1, n_mod
            sa = 0.0_KP; sn = 0.0_KP; se = 0.0_KP
            do c = 1, NCH
                sa = sa + part(1, c, m)
                sn = sn + part(2, c, m)
                se = se + part(3, c, m)
            end do
            s_abs(m) = real(sa, WP)
            s_norm(m) = real(sn, WP)
            s_esc(m) = real(se, WP)
        end do
        !$omp end target data
        return
#endif

#ifdef OMP_TARGET
        ! (teams loop + loop reduction: as fast as the former OpenACC gang / vector reduction with
        ! nvfortran 25.1 on the A100; teams distribute + parallel do reduction took 2.5x as long)
        !$omp target teams loop private(sa, sn, se, ca, cn, ce)
#endif
        do m = 1, n_mod
            sa = 0.0_KP; sn = 0.0_KP; se = 0.0_KP
            ca = 0.0_KP; cn = 0.0_KP; ce = 0.0_KP
            if (emis(m)) then
#ifdef OMP_TARGET
                !$omp loop reduction(+:sa, sn, se)
#endif
                do i = 1, nspec - 1
#if defined(OMP_TARGET)
                    sa = sa + wt(i) * (absorbed(i, m) + absorbed(i + 1, m))
                    sn = sn + wt(i) * (yo(i, n_mod + m) + yo(i + 1, n_mod + m))
                    se = se + wt(i) * (yo(i, n_mod + m) * yo(i, m) + yo(i + 1, n_mod + m) * yo(i + 1, m))
#else
                    call kadd(sa, ca, wt(i) * (absorbed(i, m) + absorbed(i + 1, m)))
                    call kadd(sn, cn, wt(i) * (yo(i, n_mod + m) + yo(i + 1, n_mod + m)))
                    call kadd(se, ce, wt(i) * (yo(i, n_mod + m) * yo(i, m) + yo(i + 1, n_mod + m) * yo(i + 1, m)))
#endif
                end do
            end if
            s_abs(m) = real(sa, WP)
            s_norm(m) = real(sn, WP)
            s_esc(m) = real(se, WP)
        end do
    end subroutine dust_emission_reductions

    !> Adds the dust emission, scaled to the absorbed luminosity (energy balance).
    subroutine add_dust_emission(n_mod, nspec, emis, yo, s_abs, s_esc, total)
        integer, intent(in)              :: n_mod, nspec
        logical, intent(in)              :: emis(n_mod)
        real(KP), intent(in)             :: yo(nspec, 2 * n_mod)
        real(WP), intent(in)             :: s_abs(n_mod), s_esc(n_mod)
        real(KP), intent(inout)          :: total(nspec, n_mod)

        integer  :: m, i

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do i = 1, nspec
                if (emis(m) .and. s_esc(m) > SAFE_FLOOR) then
                    ! As dust_emission_final in opensps_dust: shape * transmission (KP), then * norm (WP)
                    total(i, m) = total(i, m) + real(yo(i, n_mod + m) * yo(i, m) * (s_abs(m) / s_esc(m)), KP)
                end if
            end do
        end do
    end subroutine add_dust_emission




    !> The reverse pass of compute_csp_batch (gradients), for the stages whose adjoints are written:
    !> magnitudes and spectra -> observed frame -> AGN -> IGM -> attenuation -> age sums. Every stage
    !> before the magnitudes is linear in the spectrum it receives, so the adjoint is the transpose of
    !> each map, evaluated with the forward's own transmissions (pixel_transmissions, recomputed) and
    !> interpolation weights; the parameters enter through the transmissions:
    !>   F(b) = trapezoid integral of band(b) * fobs,  mag = -2.5 log10 F + const:  F_bar = -2.5 / (F ln 10) mag_bar
    !>   fobs(j) = max(v(j), 0) / lambda(j),  v(j) = (1 - t) S(il) + t S(il + 1):  scatter to S_bar
    !>   S = T_IGM A + L fagn tmpl T_d:  fagn_bar = L sum(S_bar tmpl T_d), L_bar = fagn sum(...)
    !>   A = (young a_young + old T_old) D,  a_young = T_birth (1 - frac_obrun) + frac_obrun,
    !>   D = T_d (1 - frac_nodust) + frac_nodust:  young_bar = A_bar a_young D, old_bar = A_bar T_old D,
    !>   and the dust parameters through T_d = exp(-dust2 curve), T_old = exp(-dust3 curve),
    !>   T_birth = exp(-dust1 (lambda / lambda_V)^dust1_index) (atten_adjoint_chunk)
    !>   young / old = basis . w (split at dust_tesc), L = sum(w L_age):  w_bar = basis^T y_bar + L_bar L_age
    !> Each model's cotangents are divided by their largest magnitude (cscale) before they are
    !> stored at kernel precision, and the results multiplied by it in FP64: the gradient of a
    !> magnitude of a nearly dark band (1 / F) overflows FP32 otherwise (rule 5 of
    !> doc/MIXED_PRECISION.md). Work space gw (pixels x 2N): the gradient of fobs (1:N), of the
    !> final spectrum (N+1:2N), then of the young (1:N) and old (N+1:2N) age sums; yo (the
    !> forward's age sums) is read.
    subroutine batch_backward(ctx, pbuf, n_mod, nspec, nt, nbands, zred, igm_factor, mags, fband, zcm, it_tesc, &
                              zlo, zhi, dz, zmode, igm, zvar, agn_on, ita, wta, lsum, lum_linear, tdiff, tau, yo, total, &
                              emis, iq, iu, wq, wu, gam, s_abs, s_esc, dchain, smooth, neb, nb, w2, lzm, tsave, &
                              sigma, minl, maxl, imf, zimf, pm, wsfh, slum, smass, ssp_ops, lbx, mbx, zw, wzb_out, mags_bar, &
                              spec_bar, params_bar, weights_bar)
        type(opensps_context_t), intent(inout) :: ctx
        integer, intent(in)              :: n_mod, nspec, nt, nbands
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        real(WP), intent(in)             :: zred, igm_factor   ! the forward's common redshift and IGM factor
        real(WP), intent(in)             :: mags(:,:), fband(:,:), zcm(n_mod)
        integer, intent(in)              :: it_tesc(n_mod), zlo(n_mod), zhi(n_mod), ita(n_mod)
        real(WP), intent(in)             :: dz(n_mod), wta(n_mod), lsum(n_mod), lum_linear(nt)
        logical, intent(in)              :: zmode, igm, zvar, agn_on(n_mod)
        real(KP), intent(in)             :: tdiff(:,:), tau(nspec, n_mod), total(nspec, n_mod)
        real(KP), intent(in)             :: yo(nspec, 2 * n_mod)
        ! Dust emission: per model, the flag, the template's interpolation (dl07_indices), the absorbed
        ! and escaping sums of the forward (dust_emission_reductions), d(wq, wu, gamma) / d(q_PAH,
        ! U_min, gamma)
        logical, intent(in)              :: emis(n_mod)
        integer, intent(in)              :: iq(n_mod), iu(n_mod)
        real(WP), intent(in)             :: wq(n_mod), wu(n_mod), gam(n_mod), s_abs(n_mod), s_esc(n_mod), dchain(7, n_mod)
        logical, intent(in)              :: smooth   ! the forward smoothed the spectra (smooth_columns)
        ! Nebular emission (add_neb_emission = 1): its setup, and the young / old SFH weights
        logical, intent(in)              :: neb
        type(nebular_setup_t), intent(in) :: nb
        real(KP), intent(in)             :: w2(nt, 2 * n_mod)
        logical, intent(in)              :: lzm   ! the magnitudes at zred_mags (not at zred)
        ! Smoothing: the spectra before it (nspec x N when smooth), and sigma_smooth
        real(KP), intent(inout), contiguous :: tsave(:,:)
        real(WP), intent(in)             :: sigma, minl, maxl   ! (the smoothing window: host modes)
        ! The IMF per model (ssp_ops): its flags, parameters, masked SFH weights, L_bol per age, operators
        logical, intent(in)              :: imf, zimf
        real(WP), intent(in), contiguous :: pm(:,:), wsfh(:,:), slum(:,:), smass(:,:)
        type(ssp_opset_t), intent(inout), optional :: ssp_ops
        !> dL / d(L_bol) from the lbol output's cotangent (lbol = log10 L_bol), unscaled
        real(WP), intent(in)             :: lbx(n_mod)
        !> dL / d(mass) from the mass output's cotangent with ssp_ops (zero otherwise), unscaled
        real(WP), intent(in)             :: mbx(n_mod)
        ! Weights over ages and metallicities (ssp_weights_z): d L / d wz (nt, nz, N; 1 x 1 x 1 otherwise)
        logical, intent(in)              :: zw
        real(WP), intent(out), contiguous :: wzb_out(:,:,:)
        real(WP), intent(in), optional   :: mags_bar(:,:), spec_bar(:,:)
        real(WP), intent(out), optional  :: params_bar(:,:), weights_bar(:,:)

        real(WP), allocatable :: fb_bar(:,:), fagn_bar(:), lsum_bar(:), w_bar(:,:), cscale(:), dpar(:,:), eb(:,:), tmbar(:)
        ! Nebular: per age and model the weights' cotangent, and per model gas_logz, gas_logu, the
        ! absorbed and escaping fractions' cotangents and the metallicity weight dz's (gneb: 5 x N)
        real(WP), allocatable :: nwb(:,:), gneb(:,:)
        real(KP), allocatable :: lbar(:,:)
        integer  :: ntn
        ! The redshift: the observed frame's part (dzo), the distance's (host), the IGM's pixel cap
        real(WP), allocatable :: dzo(:), dzd(:), dsg(:)
        ! The IMF's reverse pass: dL / dQ per nebular age, the SFH weights' cotangent, the IMF rows,
        ! fesc's and dz's cotangents
        real(WP), allocatable :: qb(:,:), wbi(:,:), imfb(:,:), fescb(:), zbi(:)
        logical :: lgiven
        integer, allocatable  :: jmx(:)
        real(WP) :: msum_bar
        real(KP), allocatable :: sbar(:,:), gw(:,:), part(:,:,:)
        ! (no AGN: a zero template of the right shape, for the explicit-shape dummies of the device
        ! routines; a smaller placeholder made LLVM flang add a runtime copy in device code)
        real(WP), allocatable :: noagn(:,:)
        integer  :: m, b, nzb, ntau, i
        real(WP) :: c
        logical  :: have_s, any_agn
        real(WP), parameter :: DMAG_DF = -2.5_wp / log(10.0_wp)

        ! Magnitudes -> band fluxes (host: nbands x N values); bands without a magnitude are skipped
        allocate(fb_bar(nbands, n_mod), fagn_bar(n_mod), lsum_bar(n_mod), w_bar(nt, n_mod), cscale(n_mod), tmbar(n_mod), &
                 dpar(NDP, n_mod), gw(nspec, 2 * n_mod), part(NDP, NCHG, n_mod), eb(3, n_mod))
        allocate(dzo(n_mod), dzd(n_mod), jmx(n_mod), dsg(n_mod))
        dsg = 0.0_wp
        ! The distance term (every magnitude of the model; not with zred_mags)
        dzd = 0.0_wp
        if (present(mags_bar) .and. .not. lzm) then
            do m = 1, n_mod
                msum_bar = 0.0_wp
                do b = 1, nbands
                    if (mags(b, m) == mags(b, m) .and. fband(b, m) > 0.0_wp) msum_bar = msum_bar + mags_bar(b, m)
                end do
                if (msum_bar /= 0.0_wp) dzd(m) = msum_bar * distance_term_dz(ctx, zcm(m))
            end do
        end if
        ntn = 0
        if (neb) ntn = nb%nt_neb
        allocate(nwb(max(1, ntn), n_mod), gneb(5, n_mod), qb(max(1, ntn), n_mod), wbi(nt, n_mod), imfb(NIMFG, n_mod), &
                 fescb(n_mod), zbi(n_mod))
        qb = 0.0_wp
        imfb = 0.0_wp
        ! (the lines' cotangents: from their profiles in the spectrum, or from their absorbed part in the
        ! dust emission's energy balance when outside it)
        lgiven = neb .and. .not. nb%lines_in_spec .and. any(emis)
        if (neb .and. (nb%lines_in_spec .or. lgiven)) then
            allocate(lbar(NEMLINE, 2 * n_mod))
        else
            allocate(lbar(1, 1))
        end if
        fb_bar = 0.0_wp
        if (present(mags_bar)) then
            do m = 1, n_mod
                do b = 1, nbands
                    if (mags(b, m) == mags(b, m) .and. fband(b, m) > 0.0_wp) &
                        fb_bar(b, m) = mags_bar(b, m) * DMAG_DF / fband(b, m)
                end do
            end do
        end if
        ! (explicit loops over the models in host threads: maxval(abs(section)) made a temporary per
        ! model, and one thread took 24 ms per call of 1000 models for these two passes)
        have_s = present(spec_bar)
        if (have_s) then
            allocate(sbar(nspec, n_mod))
        else
            allocate(sbar(1, 1))
        end if
        !$omp parallel do private(b, i, c)
        do m = 1, n_mod
            c = 0.0_wp
            do b = 1, nbands
                c = max(c, abs(fb_bar(b, m)))
            end do
            c = max(c, abs(lbx(m)) * max(lsum(m), 0.0_wp))
            if (have_s) then
                do i = 1, nspec
                    c = max(c, abs(spec_bar(i, m)))
                end do
            end if
            if (.not. (c > 0.0_wp) .or. c > huge(1.0_wp)) c = 1.0_wp
            cscale(m) = c
            do b = 1, nbands
                fb_bar(b, m) = fb_bar(b, m) / c
            end do
            if (have_s) then
                do i = 1, nspec
                    sbar(i, m) = real(spec_bar(i, m) / c, KP)
                end do
            end if
        end do
        ! The band weights and trapezoid weights (z-independent), and the common IGM transmission of
        ! the forward's redshift (the cache: compute_magnitudes_columns may have released it, or
        ! left the z = 0 grid with per-model redshifts, where the IGM is per model anyway)
        if (zvar) then
            call prepare_batch_grid(ctx, 0.0_wp, igm_factor)
        else
            call prepare_batch_grid(ctx, zred, igm_factor)
        end if
        if (imf) then
            nzb = n_mod   ! (the per-model ionizing photons of nebular_setup: q0(ages, models))
        else
            nzb = size(ctx%state%ssp_basis_spec, 3)
        end if
        any_agn = any(agn_on)
        ntau = 1
        if (any_agn) ntau = size(ctx%state%agndust_spec, 2)
        if (any_agn) then
            allocate(noagn(1, 1))
        else
            allocate(noagn(nspec, 1))
        end if
        noagn = 0.0_wp

#ifdef OMP_TARGET
        !$omp target data map(to: fb_bar, sbar, zcm, noagn, qb) &
        !$omp& map(alloc: fagn_bar, lsum_bar, tmbar, gw, part, eb, nwb, lbar, jmx) &
        !$omp& map(from: w_bar, dpar, gneb, dzo)
#endif
        associate (g => ctx%state%batch)
        call backward_observed_frame(n_mod, nspec, nbands, fb_bar, g%bands_t, g%j_lo, g%j_hi, g%wtl, g%wl, &
                                     ctx%state%spec_lambda, zcm, total, have_s, sbar, gw, .not. lzm, dzo, &
                                     g%bp, g%bi, g%bk, g%i_lo, g%t_obs, g%jr0, part)
        ! The IGM's short-wavelength cap: per model, the pixel whose optical depth caps the bluer ones
        if (igm) call igm_cap_pixels(n_mod, nspec, pbuf, ctx%state%spec_lambda, jmx)
        ! Velocity smoothing (the last stage of the spectrum): sigma_smooth's gradient (from the
        ! smoothed spectra's cotangents and the spectra before smoothing), then the transpose
        if (smooth .and. smoothing_on_device(ctx)) then
            call smooth_sigma_adjoint(ctx, sigma, tsave, gw(:, n_mod + 1:2 * n_mod), n_mod, dsg)
            call smooth_columns_adjoint(ctx, gw(:, n_mod + 1:2 * n_mod), n_mod)
        else if (smooth) then
            ! The host modes (exact log-linear velocity convolution, constant width in wavelength): their
            ! transpose in FP64 on the host, a column at a time (as smooth_columns_host)
            call smooth_columns_host_adjoint(ctx, nspec, n_mod, sigma, minl, maxl, tsave, gw, dsg)
        end if
        if (any_agn) then
            call backward_agn(n_mod, nspec, ntau, pbuf, agn_on, ita, wta, lsum, ctx%state%agndust_spec, tdiff, gw, &
                              fagn_bar, lsum_bar, tmbar)
        else
            fagn_bar = 0.0_wp
            lsum_bar = 0.0_wp
            tmbar = 0.0_wp
#ifdef OMP_TARGET
            !$omp target update to(fagn_bar, lsum_bar, tmbar)
#endif
        end if
        ! The lbol output's cotangent joins L_bol's (lbol = log10 L_bol; zero where the forward floors it)
        if (any(lbx /= 0.0_wp)) then
#ifdef OMP_TARGET
            !$omp target update from(lsum_bar)
#endif
            do m = 1, n_mod
                if (lsum(m) > SAFE_FLOOR) lsum_bar(m) = lsum_bar(m) + lbx(m) / (lsum(m) * log(10.0_wp)) / cscale(m)
            end do
#ifdef OMP_TARGET
            !$omp target update to(lsum_bar)
#endif
        end if
        ! Dust emission: the cotangents of its two sums (eb: s_abs_bar, s_esc_bar, R = s_abs / s_esc)
        call backward_emission_sums(ctx, pbuf, n_mod, nspec, igm, zvar, g%igm_t, tau, g%wt, emis, iq, iu, wq, wu, gam, &
                                    s_abs, s_esc, gw, part, eb)
        if (any_agn) then
            call backward_attenuation(ctx, pbuf, n_mod, nspec, ntau, igm, zvar, g%igm_t, tau, yo, agn_on, ita, wta, &
                                      lsum, ctx%state%agndust_spec, g%wt, emis, iq, iu, wq, wu, gam, eb, jmx, gw, part, dpar)
        else
            call backward_attenuation(ctx, pbuf, n_mod, nspec, 1, igm, zvar, g%igm_t, tau, yo, agn_on, ita, wta, &
                                      lsum, noagn, g%wt, emis, iq, iu, wq, wu, gam, eb, jmx, gw, part, dpar)
        end if
        ! Lines outside the spectrum with dust emission: their absorbed part joins s_abs
        if (lgiven) call backward_line_absorption(ctx, pbuf, n_mod, nspec, nb%lum, nb%llo, emis, eb, lbar, dpar)
        ! Nebular emission: the continuum and the lines (the weights, gas_logz, gas_logu and the
        ! absorbed fraction), before the age sums' adjoint, which takes the escaping EUV fraction
        if (neb) then
            ! (the line profiles exist only with the lines in the spectrum; lbar stands in otherwise)
            if (nb%xrb .and. nb%lines_in_spec) then
                call backward_nebular(n_mod, nspec, nt, nzb, ntn, nb%cont, .true., it_tesc, zlo, zhi, dz, &
                                      zmode .and. .not. imf, &
                                      nb%ia, nb%wa, nb%iz, nb%wz, nb%iu, nb%wu, nb%fabs, nb%q0, &
                                      nb%lj_lo, nb%lj_hi, nb%nrun, nb%rs, nb%rl, nb%rt, w2, &
                                      ctx%state%xnebem_cont, ctx%state%xnebem_line, ctx%state%gaussnebarr, gw, lbar, nwb, gneb, &
                                      imf, qb, lgiven)
            else if (nb%xrb) then
                call backward_nebular(n_mod, nspec, nt, nzb, ntn, nb%cont, lgiven, it_tesc, zlo, zhi, dz, &
                                      zmode .and. .not. imf, &
                                      nb%ia, nb%wa, nb%iz, nb%wz, nb%iu, nb%wu, nb%fabs, nb%q0, &
                                      nb%lj_lo, nb%lj_hi, nb%nrun, nb%rs, nb%rl, nb%rt, w2, &
                                      ctx%state%xnebem_cont, ctx%state%xnebem_line, lbar, gw, lbar, nwb, gneb, imf, qb, lgiven)
            else if (nb%lines_in_spec) then
                call backward_nebular(n_mod, nspec, nt, nzb, ntn, nb%cont, .true., it_tesc, zlo, zhi, dz, &
                                      zmode .and. .not. imf, &
                                      nb%ia, nb%wa, nb%iz, nb%wz, nb%iu, nb%wu, nb%fabs, nb%q0, &
                                      nb%lj_lo, nb%lj_hi, nb%nrun, nb%rs, nb%rl, nb%rt, w2, &
                                      ctx%state%nebem_cont, ctx%state%nebem_line, ctx%state%gaussnebarr, gw, lbar, nwb, gneb, &
                                      imf, qb, lgiven)
            else
                call backward_nebular(n_mod, nspec, nt, nzb, ntn, nb%cont, lgiven, it_tesc, zlo, zhi, dz, &
                                      zmode .and. .not. imf, &
                                      nb%ia, nb%wa, nb%iz, nb%wz, nb%iu, nb%wu, nb%fabs, nb%q0, &
                                      nb%lj_lo, nb%lj_hi, nb%nrun, nb%rs, nb%rl, nb%rt, w2, &
                                      ctx%state%nebem_cont, ctx%state%nebem_line, lbar, gw, lbar, nwb, gneb, imf, qb, lgiven)
            end if
            if (.not. imf) call backward_age_sums(n_mod, nspec, nt, nzb, ctx%state%ssp_basis_spec, &
                                                  ctx%state%ssp_basis_spec_t, &
                                                  ctx%state%ssp_basis_lbol, zmode, it_tesc, zlo, zhi, dz, lum_linear, &
                                                  lsum_bar, gw, w_bar, ntn, nb%whlylim, nb%fesc, w2, nwb, gneb)
        else
            gneb = 0.0_wp
            nwb = 0.0_wp
#ifdef OMP_TARGET
            !$omp target update to(nwb, gneb)
#endif
            if (zw) then
                call backward_age_sums_w(n_mod, nspec, nt, nzb, ctx%state%ssp_basis_spec, &
                                         ctx%state%ssp_basis_spec_t, ctx%state%ssp_basis_lbol, &
                                         it_tesc, lsum_bar, gw, wzb_out)
                w_bar = 0.0_wp
#ifdef OMP_TARGET
                !$omp target update to(w_bar)
#endif
            else if (.not. imf) then
                call backward_age_sums(n_mod, nspec, nt, nzb, ctx%state%ssp_basis_spec, &
                                       ctx%state%ssp_basis_spec_t, &
                                       ctx%state%ssp_basis_lbol, zmode, it_tesc, zlo, zhi, dz, lum_linear, &
                                       lsum_bar, gw, w_bar, 0, 0, nb%fesc, w2, nwb, gneb)
            end if
        end if
        ! The IMF per model: the operator's reverse pass (the SFH weights, the IMF parameters, the
        ! escaping fraction and, with the metallicity free, dz), then the nebular weights' part
        if (imf) then
#ifdef OMP_TARGET
            !$omp target update from(qb, nwb, gneb)
#endif
            call ssp_batch_age_sums_adjoint(ctx%imf_type_val, ctx%add_stellar_remnants_val == 1, n_mod, nt, nspec, &
                                            ssp_ops, zlo, zhi, dz, pm, wsfh, it_tesc, nb%nt_neb, nb%whlylim, nb%fesc, &
                                            gw, slum, lsum_bar, qb, wbi, imfb, fescb, zbi, smass, mbx / cscale)
            do m = 1, n_mod
                w_bar(:, m) = wbi(:, m)
                if (ntn > 0) w_bar(1:ntn, m) = w_bar(1:ntn, m) + nwb(1:ntn, m)
                gneb(4, m) = fescb(m)
                if (zimf) gneb(5, m) = gneb(5, m) + zbi(m)
            end do
#ifdef OMP_TARGET
            !$omp target update to(w_bar, gneb)
#endif
        end if
        end associate
#ifdef OMP_TARGET
        !$omp target update from(fagn_bar, tmbar)
        !$omp end target data
#endif
        if (.not. ctx%on_device) call batch_grid_release(ctx%state%batch)

        if (present(params_bar)) then
            do m = 1, n_mod
                params_bar(1:NBGRAD, m) = 0.0_wp
                params_bar(BGRAD_FAGN, m) = fagn_bar(m) * cscale(m)
                params_bar(BGRAD_DUST1:BGRAD_UVB, m) = dpar(1:9, m) * cscale(m)
                params_bar(BGRAD_DUSTE_QPAH, m) = dpar(10, m) * dchain(1, m) * cscale(m)
                params_bar(BGRAD_DUSTE_UMIN, m) = dpar(11, m) * dchain(2, m) * cscale(m)
                params_bar(BGRAD_DUSTE_GAMMA, m) = dpar(12, m) * dchain(3, m) * cscale(m)
                ! Nebular: frac_obrun also sets the escaping EUV fraction (gneb(4)) and the absorbed
                ! fraction of the ionizing photons, 1 - frac_obrun (gneb(3))
                params_bar(BGRAD_GAS_LOGZ, m) = gneb(1, m) * dchain(4, m) * cscale(m)
                params_bar(BGRAD_GAS_LOGU, m) = gneb(2, m) * dchain(5, m) * cscale(m)
                params_bar(BGRAD_FRAC_OBRUN, m) = params_bar(BGRAD_FRAC_OBRUN, m) + &
                                                  (gneb(4, m) * dchain(6, m) - gneb(3, m)) * cscale(m)
                params_bar(BGRAD_LOGZSOL, m) = gneb(5, m) * dchain(7, m) * cscale(m)
                params_bar(BGRAD_ZRED, m) = (dzo(m) + dpar(13, m)) * cscale(m) + dzd(m)
                params_bar(BGRAD_SIGMA_SMOOTH, m) = dsg(m) * cscale(m)
                ! (agn_tau: d L / d wta here; compute_csp_batch applies d wta / d agn_tau)
                params_bar(BGRAD_AGN_TAU, m) = tmbar(m) * cscale(m)
                params_bar(BGRAD_IGM_FACTOR, m) = dpar(14, m) * cscale(m)
                if (imf) params_bar(BGRAD_IMF1:BGRAD_MDAVE, m) = imfb(:, m) * cscale(m)
            end do
        end if
        if (present(weights_bar)) then
            do m = 1, n_mod
                weights_bar(1:nt, m) = w_bar(:, m) * cscale(m)
            end do
        end if
        if (zw) then
            do m = 1, n_mod
                wzb_out(:, :, m) = wzb_out(:, :, m) * cscale(m)
            end do
        end if
    end subroutine batch_backward

    !> Adjoint of the observed-frame interpolation and of the band integrals: gw(:, m) = d L /
    !> d fobs (per pixel j, the band weights of the trapezoid rule, as band_fluxes sums them), then
    !> gw(:, N + m) = spec_bar + the transposed interpolation (one thread per model, serial over the
    !> pixels: each rest-frame pixel collects from its observed-frame neighbours).
    subroutine backward_observed_frame(n_mod, nspec, nbands, fb_bar, bands_t, j_lo, j_hi, wtl, wl, lam, zcm, total, &
                                       have_s, sbar, gw, dzon, dzo, bp, bi, bk, g_i_lo, g_t_obs, g_jr0, part)
        integer, intent(in)     :: n_mod, nspec, nbands, j_lo(nbands), j_hi(nbands)
        real(WP), intent(in)    :: fb_bar(nbands, n_mod), lam(nspec), zcm(n_mod)
        real(KP), intent(in)    :: bands_t(nbands, nspec), wtl(nspec), wl(nspec), total(nspec, n_mod), sbar(:,:)
        logical, intent(in)     :: have_s
        real(KP), intent(inout) :: gw(nspec, 2 * n_mod)
        !> dzo: d L / d z of the observed-frame resampling, sum over pixels of fobs_bar
        !> (S(il + 1) - S(il)) dt / dz (computed when dzon)
        logical, intent(in)     :: dzon
        real(WP), intent(out)   :: dzo(n_mod)
        integer, intent(in)     :: bp(nspec + 1), bi(:)
        real(KP), intent(in)    :: bk(:)
        integer, intent(in)     :: g_i_lo(nspec), g_jr0(nspec)
        real(WP), intent(in)    :: g_t_obs(nspec)
        real(KP), intent(inout) :: part(NDP, NCHG, n_mod)

        integer, parameter :: NZC = 32
        integer  :: m, j, il, k, c, j0, j1, nbz, i, jlo, jhi, jm
        real(WP) :: t, sz
        real(KP) :: vk
        logical  :: zsame
        real(KP) :: gk, tk

        zsame = all(zcm(1:n_mod) == zcm(1))

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(gk, vk, tk, t, il, k)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(gk, vk, tk, t, il, k)
#endif
        do m = 1, n_mod
            do j = 1, nspec
                gk = 0.0_KP
                do k = bp(j), bp(j + 1) - 1
                    gk = gk + real(fb_bar(bi(k), m), KP) * bk(k)
                end do
                ! The clamp of the forward (max(v, 0), NaN -> 0): no gradient where it acts
                if (zsame) then
                    il = g_i_lo(j)
                    t = g_t_obs(j)
                else
                    call frame_point(nspec, lam, zcm(m), j, il, t)
                end if
                tk = real(t, KP)
                vk = (1.0_KP - tk) * total(il, m)
                if (t /= 0.0_wp) vk = vk + tk * total(il + 1, m)
                if (.not. (vk > 0.0_KP)) gk = 0.0_KP
                gw(j, m) = gk * real(wl(j), KP)
            end do
        end do

        ! d L / d z of the resampling (t = (lambda_j / (1 + z) - lambda_il) / (lambda_il+1 - lambda_il)):
        ! per model, partial sums over NZC contiguous blocks of pixels (one thread each), then the
        ! blocks in order. We reuse the scratch buffer 'part' (which is only needed in backward_attenuation later).
        nbz = (nspec + NZC - 1) / NZC
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(j, j0, j1, il, t, sz)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(j, j0, j1, il, t, sz)
#endif
        do m = 1, n_mod
            do c = 1, NZC
                sz = 0.0_wp
                if (dzon .and. abs(zcm(m)) > SAFE_FLOOR) then
                    j0 = (c - 1) * nbz + 1
                    j1 = min(c * nbz, nspec)
                    il = 1
                    do j = j0, j1
                        if (zsame) then
                            il = g_i_lo(j)
                            t = g_t_obs(j)
                        else
                            call frame_point_h(nspec, lam, zcm(m), j, il, t)
                        end if
                        sz = sz + real(gw(j, m), WP) * (real(total(il + 1, m), WP) - real(total(il, m), WP)) * &
                                  (-lam(j) / (1.0_wp + zcm(m))**2) / (lam(il + 1) - lam(il))
                    end do
                end if
                part(1, c, m) = real(sz, KP)
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(c, sz)
#elif defined(_OPENMP)
        !$omp parallel do private(c, sz)
#endif
        do m = 1, n_mod
            sz = 0.0_wp
            do c = 1, NZC
                sz = sz + real(part(1, c, m), WP)
            end do
            dzo(m) = sz
        end do

        ! The resampling's transpose into the rest frame (the old column, which also takes the
        ! spectrum's own cotangent): S_bar(il) += (1 - t) fobs_bar(j), S_bar(il + 1) += t fobs_bar(j)
#if defined(OMP_TARGET)
        ! GPUs: a gather, one thread per (rest pixel i, model). The interval il(j) does not decrease
        ! with j, so the observed pixels that reach i (il = i - 1 with t /= 0, then il = i) are one run,
        ! from the first with il >= i - 1 (a bisection over j): the same additions, in the same order,
        ! as the scatter below (bit for bit), without a serial pass per model
        !$omp target teams distribute parallel do collapse(2) private(j, jlo, jhi, jm, il, t, vk)
        do m = 1, n_mod
            do i = 1, nspec
                if (have_s) then
                    vk = sbar(i, m)
                else
                    vk = 0.0_KP
                end if
                ! (first j with il(j) >= i - 1)
                jlo = 1
                jhi = nspec + 1
                if (zsame) then
                    jlo = g_jr0(i)
                    jhi = jlo
                end if
                do while (jlo < jhi)
                    jm = (jlo + jhi) / 2
                    call frame_point(nspec, lam, zcm(m), jm, il, t)
                    if (il >= i - 1) then
                        jhi = jm
                    else
                        jlo = jm + 1
                    end if
                end do
                il = 1
                do j = jlo, nspec
                    if (zsame) then
                        il = g_i_lo(j)
                        t = g_t_obs(j)
                    else
                        call frame_point(nspec, lam, zcm(m), j, il, t)
                    end if
                    if (il > i) exit
                    if (il == i - 1) then
                        if (t /= 0.0_wp) vk = vk + real(t * real(gw(j, m), WP), KP)
                    else if (il == i) then
                        vk = vk + real((1.0_wp - t) * real(gw(j, m), WP), KP)
                    end if
                end do
                gw(i, n_mod + m) = vk
            end do
        end do
#else
        ! (CPU builds: the scatter, one model per thread, the pixels' intervals found from the previous one)
#if defined(_OPENMP)
        !$omp parallel do private(j, il, t)
#endif
        do m = 1, n_mod
            do j = 1, nspec
                if (have_s) then
                    gw(j, n_mod + m) = sbar(j, m)
                else
                    gw(j, n_mod + m) = 0.0_KP
                end if
            end do
            il = 1
            do j = 1, nspec
                if (zsame) then
                    il = g_i_lo(j)
                    t = g_t_obs(j)
                else
                    call frame_point_h(nspec, lam, zcm(m), j, il, t)
                end if
                gw(il, n_mod + m) = gw(il, n_mod + m) + real((1.0_wp - t) * real(gw(j, m), WP), KP)
                if (t /= 0.0_wp) gw(il + 1, n_mod + m) = gw(il + 1, n_mod + m) + real(t * real(gw(j, m), WP), KP)
            end do
        end do
#endif
    end subroutine backward_observed_frame

    !> Rest-frame interval il and position t of observed-frame pixel j at redshift z, as
    !> prepare_batch_grid and observed_frame_z find them (il = j, t = 0 at z = 0).
    subroutine frame_point(nspec, lam, z, j, il, t)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)   :: nspec, j
        real(WP), intent(in)  :: lam(nspec), z
        integer, intent(out)  :: il
        real(WP), intent(out) :: t
        real(WP) :: target

        if (abs(z) > SAFE_FLOOR) then
            target = lam(j) / (1.0_wp + z)
            il = max(1, min(find_interval(lam, target), nspec - 1))
            t = (target - lam(il)) / (lam(il + 1) - lam(il))
        else
            il = j
            t = 0.0_wp
        end if
    end subroutine frame_point

    !> frame_point with the previous pixel's interval as a hint (il: in, the hint; out, the
    !> interval): the pixels of one model in increasing order find theirs in a step or two.
    subroutine frame_point_h(nspec, lam, z, j, il, t)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)    :: nspec, j
        real(WP), intent(in)   :: lam(nspec), z
        integer, intent(inout) :: il
        real(WP), intent(out)  :: t
        real(WP) :: target

        if (abs(z) > SAFE_FLOOR) then
            target = lam(j) / (1.0_wp + z)
            il = max(1, min(find_interval_with_hint(lam, target, il), nspec - 1))
            t = (target - lam(il)) / (lam(il + 1) - lam(il))
        else
            il = j
            t = 0.0_wp
        end if
    end subroutine frame_point_h

    !> Per model, the pixel whose IGM optical depth caps the bluer pixels (apply_igm_z,
    !> compute_igm_transmission: the first maximum; 1 without a cap).
    subroutine igm_cap_pixels(n_mod, nspec, pbuf, lam, jmx)
        integer, intent(in)              :: n_mod, nspec
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        real(WP), intent(in)             :: lam(nspec)
        integer, intent(out)             :: jmx(n_mod)
        integer, parameter :: NCK = 32
        integer  :: m, j, c, jl, nb, j0, j1
        real(WP) :: tmax, tj
        real(WP), allocatable :: cmx(:,:)
        integer, allocatable  :: cjx(:,:)
        type(params_core), pointer :: pl

        ! The optical depth is zero at and above Ly-alpha whatever the redshift (it depends on the rest
        ! wavelength only through the lines' and the limit's thresholds), and the maximum is the first
        ! strict one: only the pixels below jl can hold it (the same pixel as a scan of all)
        jl = nspec
        do while (jl > 1)
            if (igm_tau(lam(jl), 2.0_wp) > 0.0_wp) exit
            jl = jl - 1
        end do
        ! Per model the first maximum over NCK contiguous blocks (one thread per block and model),
        ! then over the blocks in order
        nb = (jl + NCK - 1) / NCK
        allocate(cmx(NCK, n_mod), cjx(NCK, n_mod))
#if defined(OMP_TARGET)
        !$omp target data map(alloc: cmx, cjx)
        !$omp target teams distribute parallel do collapse(2) private(j, j0, j1, tmax, tj, pl)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(j, j0, j1, tmax, tj, pl)
#endif
        do m = 1, n_mod
            do c = 1, NCK
#if !defined(__flang__)
                pl => pbuf(m)%params_core
#else
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                tmax = 0.0_wp
                cjx(c, m) = 0
                if (pl%zred > SAFE_FLOOR) then
                    j0 = (c - 1) * nb + 1
                    j1 = min(c * nb, jl)
                    do j = j0, j1
                        tj = igm_tau(lam(j), 1.0_wp + pl%zred)
                        if (tj > tmax) then
                            tmax = tj
                            cjx(c, m) = j
                        end if
                    end do
                end if
                cmx(c, m) = tmax
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(c, tmax)
#elif defined(_OPENMP)
        !$omp parallel do private(c, tmax)
#endif
        do m = 1, n_mod
            jmx(m) = 1
            tmax = 0.0_wp
            do c = 1, NCK
                if (cmx(c, m) > tmax) then
                    tmax = cmx(c, m)
                    jmx(m) = cjx(c, m)
                end if
            end do
        end do
#if defined(OMP_TARGET)
        !$omp end target data
#endif
    end subroutine igm_cap_pixels

    !> Adjoint of add_agn_batch: per model, s = sum over pixels of S_bar tmpl T_diff (compensated
    !> at FP32 kernel precision); fagn_bar = L_bol s, lsum_bar = fagn s (zero where the forward
    !> floors L_bol). S_bar passes through unchanged.
    subroutine backward_agn(n_mod, nspec, ntau, pbuf, agn_on, ita, wta, lsum, agndust_spec, tdiff, gw, fagn_bar, lsum_bar, &
                            tmb)
        integer, intent(in)              :: n_mod, nspec, ntau
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        logical, intent(in)              :: agn_on(n_mod)
        integer, intent(in)              :: ita(n_mod)
        real(WP), intent(in)             :: wta(n_mod), lsum(n_mod), agndust_spec(nspec, ntau)
        real(KP), intent(in)             :: tdiff(:,:), gw(nspec, 2 * n_mod)
        real(WP), intent(out)            :: fagn_bar(n_mod), lsum_bar(n_mod)
        !> d L / d wta (the template's interpolation weight): L_bol fagn sum(S_bar (tmpl_hi - tmpl_lo) T_d)
        real(WP), intent(out)            :: tmb(n_mod)

        ! Per model the sums over NAC blocks of pixels (one thread each; compensated at kernel
        ! precision), then the blocks in order (FP64)
        integer, parameter :: NAC = 32
        integer  :: m, i, c, i0, i1, nbk
        real(KP) :: sk, ck, term, tsum, sd, cd
        real(WP) :: lb, s1, s2
        real(KP), allocatable :: ap(:,:,:)
        type(params_core), pointer :: pl

        nbk = (nspec + NAC - 1) / NAC
        allocate(ap(2, NAC, n_mod))
#if defined(OMP_TARGET)
        !$omp target data map(alloc: ap)
        !$omp target teams distribute parallel do collapse(2) private(i, i0, i1, sk, ck, term, tsum, sd, cd)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(i, i0, i1, sk, ck, term, tsum, sd, cd)
#endif
        do m = 1, n_mod
            do c = 1, NAC
                sk = 0.0_KP
                ck = 0.0_KP
                sd = 0.0_KP
                cd = 0.0_KP
                if (agn_on(m)) then
                    i0 = (c - 1) * nbk + 1
                    i1 = min(c * nbk, nspec)
                    do i = i0, i1
                        term = gw(i, n_mod + m) * real((1.0_wp - wta(m)) * agndust_spec(i, ita(m)) + &
                                                       wta(m) * agndust_spec(i, ita(m) + 1), KP) * tdiff(i, m) - ck
                        tsum = sk + term
                        ck = (tsum - sk) - term
                        sk = tsum
                        term = gw(i, n_mod + m) * real(agndust_spec(i, ita(m) + 1) - agndust_spec(i, ita(m)), KP) * &
                               tdiff(i, m) - cd
                        tsum = sd + term
                        cd = (tsum - sd) - term
                        sd = tsum
                    end do
                end if
                ap(1, c, m) = sk
                ap(2, c, m) = sd
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(c, s1, s2, lb, pl)
#elif defined(_OPENMP)
        !$omp parallel do private(c, s1, s2, lb, pl)
#endif
        do m = 1, n_mod
            fagn_bar(m) = 0.0_wp
            lsum_bar(m) = 0.0_wp
            tmb(m) = 0.0_wp
            if (agn_on(m)) then
#if !defined(__flang__)
                pl => pbuf(m)%params_core
#else
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                s1 = 0.0_wp
                s2 = 0.0_wp
                do c = 1, NAC
                    s1 = s1 + real(ap(1, c, m), WP)
                    s2 = s2 + real(ap(2, c, m), WP)
                end do
                if (lsum(m) > SAFE_FLOOR) then
                    lb = log10(lsum(m))
                    lsum_bar(m) = pl%fagn * s1
                else
                    lb = 0.0_wp
                end if
                fagn_bar(m) = 10.0_wp**lb * s1
                tmb(m) = 10.0_wp**lb * pl%fagn * s2
            end if
        end do
#if defined(OMP_TARGET)
        !$omp end target data
#endif
    end subroutine backward_agn

    !> Dust emission, first pass: the emission E = shape T_d R (R = s_abs / s_esc) is linear in R, so
    !> R_bar = sum over pixels of A_bar shape T_d (per model, NCHG chunks then FP64), and
    !> s_abs_bar = R_bar / s_esc, s_esc_bar = -R_bar s_abs / s_esc^2 (eb(1:2, m); eb(3, m) = R). Zero
    !> for models without dust emission (or with s_esc at the floor, where the forward adds none).
    subroutine backward_emission_sums(ctx, pbuf, n_mod, nspec, igm, zvar, igm_t, tau, wt, emis, iq, iu, wq, wu, gam, &
                                      s_abs, s_esc, gw, part, eb)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_mod, nspec
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        logical, intent(in)              :: igm, zvar, emis(n_mod)
        integer, intent(in)              :: iq(n_mod), iu(n_mod)
        real(WP), intent(in)             :: wq(n_mod), wu(n_mod), gam(n_mod), s_abs(n_mod), s_esc(n_mod)
        real(KP), intent(in)             :: igm_t(nspec), tau(nspec, n_mod), wt(nspec), gw(nspec, 2 * n_mod)
        real(KP), intent(out)            :: part(NDP, NCHG, n_mod)
        real(WP), intent(out)            :: eb(3, n_mod)

        integer  :: m, c, i, dt
        real(KP) :: sk, ck, term, tsum, ga, td
        real(WP) :: rb
        type(params_core), pointer :: pl

        dt = ctx%dust_type_val
        associate (ttab => ctx%state%att_tab, ltab => ctx%state%att_ltab)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(i, sk, ck, term, tsum, ga, td, pl) &
        !$omp& map(OMP_MAP_PRESENT_ALLOC: ctx%state%spec_lambda, ctx%state%dustem2_dustem, ttab, ltab)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(i, sk, ck, term, tsum, ga, td, pl)
#endif
        do m = 1, n_mod
            do c = 1, NCHG
                sk = 0.0_KP
                ck = 0.0_KP
                if (emis(m) .and. s_esc(m) > SAFE_FLOOR) then
#if !defined(__flang__)
                    pl => pbuf(m)%params_core
#else
                    call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                    do i = c, nspec, NCHG
                        ga = gw(i, n_mod + m)
                        if (igm .and. zvar) then
                            if (pl%zred > SAFE_FLOOR) ga = ga * real(exp(-real(tau(i, m), WP) * pl%igm_factor), KP)
                        else if (igm) then
                            ga = ga * igm_t(i)
                        end if
                        call pixel_diffuse_transmission(ctx, pl, nspec, i, dt, ttab, ltab, td)
                        term = ga * dl07_shape(ctx, i, iq(m), wq(m), iu(m), wu(m), gam(m)) * td - ck
                        tsum = sk + term
                        ck = (tsum - sk) - term
                        sk = tsum
                    end do
                end if
                part(1, c, m) = sk
            end do
        end do
        end associate
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(c, rb)
#elif defined(_OPENMP)
        !$omp parallel do private(c, rb)
#endif
        do m = 1, n_mod
            eb(1, m) = 0.0_wp
            eb(2, m) = 0.0_wp
            eb(3, m) = 0.0_wp
            if (emis(m) .and. s_esc(m) > SAFE_FLOOR) then
                rb = 0.0_wp
                do c = 1, NCHG
                    rb = rb + real(part(1, c, m), WP)
                end do
                eb(1, m) = rb / s_esc(m)
                eb(2, m) = -rb * s_abs(m) / s_esc(m)**2
                eb(3, m) = s_abs(m) / s_esc(m)
            end if
        end do
    end subroutine backward_emission_sums

    !> Adjoint of attenuate_lines' absorbed part (lines outside the spectrum with dust emission: s_abs
    !> += s_line, s_line = sum over lines of (L_y + L_o) - (L_y A_y D + L_o D), A_y = T_birth (1 -
    !> frac_obrun) + frac_obrun, D = T_d (1 - frac_nodust) + frac_nodust at the line's wavelength, T_d
    !> interpolated between its pixels): with s_line_bar = s_abs_bar (eb(1, m)), the lines'
    !> cotangents lbar (young: s_line_bar (1 - A_y D), old: s_line_bar (1 - D)) and the dust rows'
    !> parts added to dpar (T_d through the curve tables' derivatives, as atten_adjoint_chunk). One
    !> thread per model.
    subroutine backward_line_absorption(ctx, pbuf, n_mod, nspec, nb_lum, nb_llo, emis, eb, lbar, dpar)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_mod, nspec
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        real(KP), intent(in)             :: nb_lum(NEMLINE, 2 * n_mod)
        integer, intent(in)              :: nb_llo(NEMLINE)
        logical, intent(in)              :: emis(n_mod)
        real(WP), intent(in)             :: eb(3, n_mod)
        real(KP), intent(out)            :: lbar(NEMLINE, 2 * n_mod)
        real(WP), intent(inout)          :: dpar(NDP, n_mod)

        integer  :: m, dt
        type(params_core), pointer :: pl

        dt = ctx%dust_type_val
        associate (ttab => ctx%state%att_tab, ltab => ctx%state%att_ltab)
        ! (the model's work in a routine of its own: its arrays are then plain locals; LLVM flang maps
        ! private arrays of a target loop)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(pl) &
        !$omp& map(OMP_MAP_PRESENT_ALLOC: ctx%state%spec_lambda, ctx%state%nebem_line_pos, ttab, ltab)
#elif defined(_OPENMP)
        !$omp parallel do private(pl)
#endif
        do m = 1, n_mod
#if !defined(__flang__)
            pl => pbuf(m)%params_core
#else
            call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
            call line_absorption_model(ctx, pl, n_mod, nspec, m, dt, emis(m), eb(1, m), nb_lum, nb_llo, ttab, ltab, &
                                       lbar, dpar)
        end do
        end associate
    end subroutine backward_line_absorption

    !> One model of backward_line_absorption.
    subroutine line_absorption_model(ctx, pl, n_mod, nspec, m, dt, emis_m, sb, nb_lum, nb_llo, ttab, ltab, lbar, dpar)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(params_core), intent(in)    :: pl
        integer, intent(in)              :: n_mod, nspec, m, dt
        logical, intent(in)              :: emis_m
        real(WP), intent(in)             :: sb
        real(KP), intent(in)             :: nb_lum(NEMLINE, 2 * n_mod)
        integer, intent(in)              :: nb_llo(NEMLINE)
        real(KP), intent(in)             :: ttab(nspec, 0:3)
        real(WP), intent(in)             :: ltab(nspec)
        real(KP), intent(inout)          :: lbar(NEMLINE, 2 * n_mod)
        real(WP), intent(inout)          :: dpar(NDP, n_mod)

        integer  :: i, lo, k
        real(WP) :: pos, fob, fnd, fr, tpix(2), dtp(2, 4), td, dtd(4), birth, pb, ay, dfac, ly, lo_, lk
        real(WP) :: curve, tau_d, trans, dc(3), d1, d2, d4, d5, d6, d7, d8, d9, s1

        do i = 1, NEMLINE
            lbar(i, m) = 0.0_KP
            lbar(i, n_mod + m) = 0.0_KP
        end do
        if (.not. emis_m .or. sb == 0.0_wp) return
        fob = pl%frac_obrun
        fnd = pl%frac_nodust
        d1 = 0.0_wp; d2 = 0.0_wp; d4 = 0.0_wp; d5 = 0.0_wp; d6 = 0.0_wp; d7 = 0.0_wp; d8 = 0.0_wp; d9 = 0.0_wp
        do i = 1, NEMLINE
            pos = ctx%state%nebem_line_pos(i)
            ! T_d at the line (and its derivatives in dust2, dust_index, R_V, uvb: dtd(1:4))
            td = 1.0_wp
            dtd = 0.0_wp
            if (1.0_wp - fnd > SAFE_FLOOR) then
                lo = nb_llo(i)
                do k = 1, 2
                    call diffuse_transmission(ctx, pl, lo + k - 1, curve, tau_d, trans)
                    tpix(k) = real(real(trans, KP), WP)
                    ! The curve's derivatives from its tables (as atten_adjoint_chunk)
                    dc = 0.0_wp
                    lk = ltab(lo + k - 1)
                    if (dt == 0) then
                        dc(1) = curve * lk
                    else if (dt == 4) then
                        dc(1) = real(ttab(lo + k - 1, 1), WP) * exp(pl%dust_index * lk) + curve * lk
                    else if (dt == 1) then
                        s1 = 1.0_wp / pl%mwr
                        dc(3) = real(ttab(lo + k - 1, 2), WP) + s1 * real(ttab(lo + k - 1, 3), WP)
                        dc(2) = -(real(ttab(lo + k - 1, 1), WP) + pl%uvb * real(ttab(lo + k - 1, 3), WP)) * s1 * s1
                    end if
                    if (dt /= 3) then
                        dtp(k, 1) = -tpix(k) * curve
                        dtp(k, 2) = -tpix(k) * pl%dust2 * dc(1)
                        dtp(k, 3) = -tpix(k) * pl%dust2 * dc(2)
                        dtp(k, 4) = -tpix(k) * pl%dust2 * dc(3)
                    else
                        dtp(k, 1) = 0.0_wp; dtp(k, 2) = 0.0_wp; dtp(k, 3) = 0.0_wp; dtp(k, 4) = 0.0_wp
                    end if
                end do
                fr = (pos - ctx%state%spec_lambda(lo)) / (ctx%state%spec_lambda(lo + 1) - ctx%state%spec_lambda(lo))
                td = tpix(1) + (tpix(2) - tpix(1)) * fr
                do k = 1, 4
                    dtd(k) = dtp(1, k) * (1.0_wp - fr) + dtp(2, k) * fr
                end do
            end if
            if (td /= td) then
                td = 1.0_wp
                dtd = 0.0_wp
            end if
            pb = (pos / V_BAND_ANGSTROMS)**pl%dust1_index
            if (pl%dust1 <= SAFE_FLOOR .or. 1.0_wp - fob <= SAFE_FLOOR) then
                birth = 1.0_wp
            else
                birth = exp(-pl%dust1 * pb)
            end if
            ay = birth * (1.0_wp - fob) + fob
            dfac = 1.0_wp
            if (1.0_wp - fnd > SAFE_FLOOR) dfac = td * (1.0_wp - fnd) + fnd
            ly = real(nb_lum(i, m), WP)
            lo_ = real(nb_lum(i, n_mod + m), WP)
            lbar(i, m) = real(sb * (1.0_wp - ay * dfac), KP)
            lbar(i, n_mod + m) = real(sb * (1.0_wp - dfac), KP)
            ! s_line's dust derivatives (minus the attenuated lines')
            if (1.0_wp - fnd > SAFE_FLOOR) then
                d2 = d2 - sb * (ly * ay + lo_) * (1.0_wp - fnd) * dtd(1)
                d4 = d4 - sb * (ly * ay + lo_) * (1.0_wp - fnd) * dtd(2)
                d8 = d8 - sb * (ly * ay + lo_) * (1.0_wp - fnd) * dtd(3)
                d9 = d9 - sb * (ly * ay + lo_) * (1.0_wp - fnd) * dtd(4)
                d7 = d7 - sb * (ly * ay + lo_) * (1.0_wp - td)
            end if
            d6 = d6 - sb * ly * (1.0_wp - birth) * dfac
            if (pl%dust1 <= SAFE_FLOOR .or. 1.0_wp - fob <= SAFE_FLOOR) then
                d1 = d1 + sb * ly * (1.0_wp - fob) * pb * dfac
            else
                d1 = d1 + sb * ly * (1.0_wp - fob) * pb * birth * dfac
                d5 = d5 + sb * ly * (1.0_wp - fob) * pl%dust1 * pb * log(pos / V_BAND_ANGSTROMS) * birth * dfac
            end if
        end do
        dpar(1, m) = dpar(1, m) + d1
        dpar(2, m) = dpar(2, m) + d2
        dpar(4, m) = dpar(4, m) + d4
        dpar(5, m) = dpar(5, m) + d5
        dpar(6, m) = dpar(6, m) + d6
        dpar(7, m) = dpar(7, m) + d7
        dpar(8, m) = dpar(8, m) + d8
        dpar(9, m) = dpar(9, m) + d9
    end subroutine line_absorption_model

    !> Adjoint of the IGM transmission and of attenuate_batch (with the dust emission's second pass): from S_bar in
    !> gw(:, N + m), young_bar = S_bar T_IGM a_young D into gw(:, m) and old_bar = S_bar T_IGM T_old D
    !> into gw(:, N + m), and the dust parameters' gradients dpar (NDP x N), summed per model over
    !> NCHG chunks of pixels (atten_adjoint_chunk; one thread per model and chunk, consecutive
    !> threads on consecutive pixels), then over the chunks in FP64.
    subroutine backward_attenuation(ctx, pbuf, n_mod, nspec, ntau, igm, zvar, igm_t, tau, yo, agn_on, ita, wta, lsum, &
                                    agndust_spec, wt, emis, iq, iu, wq, wu, gam, eb, jmx, gw, part, dpar)
        type(opensps_context_t), intent(in) :: ctx
        integer, intent(in)              :: n_mod, nspec, ntau
#if !defined(__flang__)
        type(params), intent(in), target :: pbuf(n_mod)
#else
        integer(int64), intent(in), target :: pbuf(NPW, n_mod)
#endif
        logical, intent(in)              :: igm, zvar, agn_on(n_mod)
        integer, intent(in)              :: ita(n_mod)
        real(WP), intent(in)             :: wta(n_mod), lsum(n_mod), agndust_spec(nspec, ntau)
        real(KP), intent(in)             :: igm_t(nspec), tau(nspec, n_mod), yo(nspec, 2 * n_mod), wt(nspec)
        logical, intent(in)              :: emis(n_mod)
        integer, intent(in)              :: iq(n_mod), iu(n_mod)
        real(WP), intent(in)             :: wq(n_mod), wu(n_mod), gam(n_mod), eb(3, n_mod)
        integer, intent(in)              :: jmx(n_mod)
        real(KP), intent(inout)          :: gw(nspec, 2 * n_mod)
        real(KP), intent(out)            :: part(NDP, NCHG, n_mod)
        real(WP), intent(out)            :: dpar(NDP, n_mod)

        integer  :: m, c, dt, q
        real(WP) :: sw
        type(params_core), pointer :: pl

        dt = ctx%dust_type_val
        associate (ttab => ctx%state%att_tab, ltab => ctx%state%att_ltab)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(pl) &
        !$omp& map(OMP_MAP_PRESENT_ALLOC: ctx%state%spec_lambda, ctx%state%dustem2_dustem, ttab, ltab)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(pl)
#endif
        do m = 1, n_mod
            do c = 1, NCHG
#if !defined(__flang__)
                pl => pbuf(m)%params_core
#else
                call c_f_pointer(c_loc(pbuf(1, m)), pl)
#endif
                call atten_adjoint_chunk(ctx, pl, n_mod, nspec, ntau, m, c, dt, ttab, ltab, igm, zvar, igm_t, tau, yo, &
                                         agn_on(m), ita(m), wta(m), lsum(m), agndust_spec, wt, emis(m), iq(m), iu(m), &
                                         wq(m), wu(m), gam(m), eb, jmx(m), gw, part)
            end do
        end do
        end associate
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(c, sw)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(c, sw)
#endif
        do m = 1, n_mod
            do q = 1, NDP
                sw = 0.0_wp
                do c = 1, NCHG
                    sw = sw + real(part(q, c, m), WP)
                end do
                dpar(q, m) = sw
            end do
        end do
    end subroutine backward_attenuation

    !> One chunk of backward_attenuation: pixels c, c + NCHG, ... of model m. Per pixel, with
    !> A_bar = S_bar T_IGM:
    !>   D_bar = A_bar (young a_young + old T_old),  T_d_bar = D_bar (1 - frac_nodust) + S_bar L fagn tmpl,
    !>   T_birth_bar = A_bar young (1 - frac_obrun) D,  T_old_bar = A_bar old D,
    !>   frac_obrun: A_bar young (1 - T_birth) D,  frac_nodust: D_bar (1 - T_d),
    !> and through T = exp(-tau): d T / d theta = -T d tau / d theta, with d curve / d dust_index =
    !> curve L (power law), T1 exp(e L) + curve L (Kriek & Conroy); d curve / d uvb = T2 + T3 / R_V
    !> and d curve / d R_V = -(T1 + uvb T3) / R_V^2 (Cardelli et al.); tau_birth = dust1 p,
    !> p = exp(dust1_index L). At dust1 = 0 or dust3 = 0 (where the forward skips the screen) the
    !> derivative is the one-sided one. With dust emission (eb: s_abs_bar, s_esc_bar, R), the
    !> emission E = shape T_d R adds T_d_bar += A_bar shape R + s_esc_bar c shape and shape_bar =
    !> A_bar T_d R + s_esc_bar c T_d (c: the pixel's trapezoid weight in frequency, as
    !> dust_emission_reductions sums), and the absorbed light (s_abs = sum c absorbed) adds
    !> absorbed_bar = s_abs_bar c to the young and old spectra, the three transmissions and the two
    !> fractions; shape_bar is carried to the template's interpolation weights (dl07_shape_grad).
    !> Sums compensated at FP32 kernel precision.
    subroutine atten_adjoint_chunk(ctx, pl, n_mod, nspec, ntau, m, c, dt, ttab, ltab, igm, zvar, igm_t, tau, yo, &
                                   agn_m, ita_m, wta_m, lsum_m, agndust_spec, wt, emis_m, iq_m, iu_m, wq_m, wu_m, gam_m, &
                                   eb, jmx_m, gw, part)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(opensps_context_t), intent(in) :: ctx
        type(params_core), intent(in)    :: pl
        integer, intent(in)              :: n_mod, nspec, ntau, m, c, dt, ita_m
        real(KP), intent(in)             :: ttab(nspec, 0:3)
        real(WP), intent(in)             :: ltab(nspec), wta_m, lsum_m, agndust_spec(nspec, ntau)
        logical, intent(in)              :: igm, zvar, agn_m
        real(KP), intent(in)             :: igm_t(nspec), tau(nspec, n_mod), yo(nspec, 2 * n_mod), wt(nspec)
        logical, intent(in)              :: emis_m
        integer, intent(in)              :: iq_m, iu_m
        real(WP), intent(in)             :: wq_m, wu_m, gam_m
        real(WP), intent(in)             :: eb(3, n_mod)   ! (whole: a section would be copied at run time)
        integer, intent(in)              :: jmx_m          ! the IGM's cap pixel (igm_cap_pixels)
        real(KP), intent(inout)          :: gw(nspec, 2 * n_mod)
        real(KP), intent(inout)          :: part(NDP, NCHG, n_mod)

        integer  :: i, q
        real(KP) :: acc(NDP), cmp(NDP), d(NDP)
        real(KP) :: y, o, gs, ga, tau_d, td, tau_b, tb, tau_o, to, curve, fo, fn, dfac, ay, gd, gtd, gtb, gto
        real(KP) :: lk, pb, dc_idx, dc_mwr, dc_uvb, s1, dust2k, dust3k, term, tsum, tmpl, lagn
        real(KP) :: cwt, shape, gshape, gab, rk, sabk, sesk, omtd, gy, go, dsq, dsu, dsg, apre
        real(WP) :: dtz
        real(WP) :: lb
        logical  :: nodust, em

        acc = 0.0_KP
        cmp = 0.0_KP
        fo = real(pl%frac_obrun, KP)
        fn = real(pl%frac_nodust, KP)
        nodust = 1.0_wp - pl%frac_nodust > SAFE_FLOOR
        dust2k = real(pl%dust2, KP)
        if (dt == 3) dust2k = 1.0_KP   ! tau_d = curve
        dust3k = real(pl%dust3, KP)
        lagn = 0.0_KP
        if (agn_m) then
            lb = 0.0_wp
            if (lsum_m > SAFE_FLOOR) lb = log10(lsum_m)
            lagn = real(10.0_wp**lb * pl%fagn, KP)
        end if
        em = emis_m .and. (eb(3, m) /= 0.0_wp .or. eb(1, m) /= 0.0_wp .or. eb(2, m) /= 0.0_wp)
        sabk = real(eb(1, m), KP)
        sesk = real(eb(2, m), KP)
        rk = real(eb(3, m), KP)
        do i = c, nspec, NCHG
            y = yo(i, m)
            o = yo(i, n_mod + m)
            gs = gw(i, n_mod + m)
            ga = gs
            if (igm .and. zvar) then
                if (pl%zred > SAFE_FLOOR) ga = gs * real(exp(-real(tau(i, m), WP) * pl%igm_factor), KP)
            else if (igm) then
                ga = gs * igm_t(i)
            end if
            call pixel_transmissions(ctx, pl, nspec, i, dt, ttab, ltab, tau_d, td, tau_b, tb, tau_o, to, curve)
            if (nodust) then
                dfac = td * (1.0_KP - fn) + fn
            else
                dfac = 1.0_KP
            end if
            ay = tb * (1.0_KP - fo) + fo
            ! The age sums' gradients
            gy = ga * ay * dfac
            go = ga * to * dfac
            ! The transmissions' gradients
            gd = ga * (y * ay + o * to)
            gtd = 0.0_KP
            if (nodust) gtd = gd * (1.0_KP - fn)
            if (agn_m) then
                tmpl = real((1.0_wp - wta_m) * agndust_spec(i, ita_m) + wta_m * agndust_spec(i, ita_m + 1), KP)
                gtd = gtd + gs * lagn * tmpl
            end if
            gtb = ga * y * (1.0_KP - fo) * dfac
            gto = ga * o * dfac
            d = 0.0_KP
            if (em) then
                ! The pixel's weight in s_abs and s_esc (wt(i) (x_i + x_i+1) summed over i)
                cwt = wt(i)
                if (i > 1) cwt = cwt + wt(i - 1)
                call dl07_shape_and_grad(ctx, i, iq_m, wq_m, iu_m, wu_m, gam_m, shape, dsq, dsu, dsg)
                ! E = shape T_d R, and s_esc = sum c shape T_d
                gtd = gtd + ga * shape * rk + sesk * cwt * shape
                gshape = ga * td * rk + sesk * cwt * td
                d(10) = gshape * dsq
                d(11) = gshape * dsu
                d(12) = gshape * dsg
                ! s_abs = sum c absorbed, absorbed = young ((1 - fo)(1 - T_b) + a_young (1 - fn)(1 - T_d))
                !                                  + old ((1 - T_o) + T_o (1 - fn)(1 - T_d))
                gab = sabk * cwt
                omtd = 0.0_KP
                if (nodust) omtd = (1.0_KP - fn) * (1.0_KP - td)
                gy = gy + gab * ((1.0_KP - fo) * (1.0_KP - tb) + ay * omtd)
                go = go + gab * ((1.0_KP - to) + to * omtd)
                gtb = gtb - gab * y * (1.0_KP - fo) * (1.0_KP - omtd)
                gto = gto - gab * o * (1.0_KP - omtd)
                if (nodust) then
                    gtd = gtd - gab * (1.0_KP - fn) * (y * ay + o * to)
                    d(7) = -gab * (1.0_KP - td) * (y * ay + o * to)
                end if
                d(6) = -gab * y * (1.0_KP - tb) * (1.0_KP - omtd)
            end if
            gw(i, m) = gy
            gw(i, n_mod + m) = go
            ! The redshift through the IGM: S = T_IGM A, T_IGM = exp(-tau igm_factor), the capped pixels
            ! taking the cap pixel's d tau / dz
            if (igm .and. pl%zred > SAFE_FLOOR) then
                apre = (y * ay + o * to) * dfac
                if (em) apre = apre + shape * td * rk
                dtz = igm_tau_dz(ctx%state%spec_lambda(max(i, jmx_m)), 1.0_wp + pl%zred)
                d(13) = -ga * apre * real(pl%igm_factor * dtz, KP)
                ! igm_factor: d T_IGM / d factor = -tau T_IGM, tau as the forward (capped below jmx)
                d(14) = -ga * apre * real(igm_tau(ctx%state%spec_lambda(max(i, jmx_m)), 1.0_wp + pl%zred), KP)
            end if
            ! The curve's derivatives
            lk = real(ltab(i), KP)
            dc_idx = 0.0_KP; dc_mwr = 0.0_KP; dc_uvb = 0.0_KP
            if (dt == 0) then
                dc_idx = curve * lk
            else if (dt == 4) then
                dc_idx = ttab(i, 1) * exp_kp(pl%dust_index * ltab(i)) + curve * lk
            else if (dt == 1) then
                s1 = 1.0_KP / real(pl%mwr, KP)
                dc_uvb = ttab(i, 2) + s1 * ttab(i, 3)
                dc_mwr = -(ttab(i, 1) + real(pl%uvb, KP) * ttab(i, 3)) * s1 * s1
            end if
            ! dust1 and dust1_index (the birth-cloud screen)
            pb = exp_kp(pl%dust1_index * ltab(i))
            if (pl%dust1 <= SAFE_FLOOR .or. 1.0_wp - pl%frac_obrun <= SAFE_FLOOR) then
                d(1) = -gtb * pb
            else
                d(1) = -gtb * pb * tb
                d(5) = -gtb * real(pl%dust1, KP) * pb * lk * tb
            end if
            ! dust2 (diffuse; not with dust type 3, whose optical depth is the curve itself)
            if (dt /= 3) d(2) = -gtd * td * curve
            ! dust3 (the old stars' extra screen; one-sided at dust3 = 0, where T_old = 1)
            d(3) = -gto * to * curve
            ! dust_index, R_V and uvb, through the curve of both screens
            d(4) = -(gtd * td * dust2k + gto * to * dust3k) * dc_idx
            d(8) = -(gtd * td * dust2k + gto * to * dust3k) * dc_mwr
            d(9) = -(gtd * td * dust2k + gto * to * dust3k) * dc_uvb
            ! frac_obrun and frac_nodust
            d(6) = d(6) + ga * y * (1.0_KP - tb) * dfac
            if (nodust) d(7) = d(7) + gd * (1.0_KP - td)
            do q = 1, NDP
                term = d(q) - cmp(q)
                tsum = acc(q) + term
                cmp(q) = (tsum - acc(q)) - term
                acc(q) = tsum
            end do
        end do
        do q = 1, NDP
            part(q, c, m) = acc(q)
        end do
    end subroutine atten_adjoint_chunk

    !> Adjoint of age_sums_w (weights over ages and metallicities, ssp_weights_z) and of L_bol:
    !> wzb(k, z, m) = sum over pixels of basis(:, k, z) times young_bar (k <= dust_tesc) or old_bar,
    !> plus lsum_bar 10^lbol(k, z) (compensated at FP32 kernel precision).
    subroutine backward_age_sums_w(n_mod, nspec, nt, nz, basis, basis_t, blbol, it_tesc, lsum_bar, gw, wzb)
        integer, intent(in)     :: n_mod, nspec, nt, nz, it_tesc(n_mod)
        real(KP), intent(in)    :: basis(nspec, nt, nz), basis_t(nt, nspec, nz), gw(nspec, 2 * n_mod)
        real(WP), intent(in)    :: blbol(nt, nz), lsum_bar(n_mod)
        real(WP), intent(out)   :: wzb(nt, nz, n_mod)
        integer  :: m, k, z, i, col
        real(KP) :: sk, ck, term, tsum

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(i, col, sk, ck, term, tsum) map(from: wzb)
#elif defined(_OPENMP)
        !$omp parallel do collapse(3) private(i, col, sk, ck, term, tsum)
#endif
        do m = 1, n_mod
            do z = 1, nz
                do k = 1, nt
                    col = n_mod + m
                    if (k <= it_tesc(m)) col = m
                    sk = 0.0_KP
                    ck = 0.0_KP
                    do i = 1, nspec
#if defined(OMP_TARGET)
                        term = basis_t(k, i, z) * gw(i, col) - ck
#else
                        term = basis(i, k, z) * gw(i, col) - ck
#endif
                        tsum = sk + term
                        ck = (tsum - sk) - term
                        sk = tsum
                    end do
                    wzb(k, z, m) = real(sk, WP) + lsum_bar(m) * 10.0_wp**blbol(k, z)
                end do
            end do
        end do
    end subroutine backward_age_sums_w

    !> Adjoint of the age sums and of L_bol: w_bar(k, m) = sum over pixels of basis(:, k) times
    !> young_bar (k <= dust_tesc) or old_bar, plus lsum_bar L_bol(k) (compensated at FP32 kernel
    !> precision). Per-model metallicity: the basis interpolated as age_sums_z interpolates it.
    !> Nebular emission (ntn nebular ages): the EUV rows (pixels <= whlylim) of the nebular ages
    !> carry the escaping fraction fesc, as nebular_kernels redoes them; gneb(4, m) = d L / d fesc;
    !> nwb (the nebular continuum's and lines' part) is added for the nebular ages.
    subroutine backward_age_sums(n_mod, nspec, nt, nzb, basis, basis_t, blbol, zmode, it_tesc, zlo, zhi, dz, lum_linear, &
                                 lsum_bar, gw, w_bar, ntn, whlylim, fesc, w2, nwb, gneb)
        integer, intent(in)     :: n_mod, nspec, nt, nzb, it_tesc(n_mod), zlo(n_mod), zhi(n_mod), ntn, whlylim
        real(KP), intent(in)    :: basis(nspec, nt, nzb), basis_t(nt, nspec, nzb), gw(nspec, 2 * n_mod), w2(nt, 2 * n_mod)
        real(WP), intent(in)    :: blbol(nt, nzb), dz(n_mod), lum_linear(nt), lsum_bar(n_mod), fesc(n_mod)
        real(WP), intent(in)    :: nwb(max(1, ntn), n_mod)
        logical, intent(in)     :: zmode
        real(WP), intent(out)   :: w_bar(nt, n_mod)
        real(WP), intent(inout) :: gneb(5, n_mod)   ! (4): set here; (5): added to

        integer  :: m, k, i, col, zl, zh, nlim
        real(KP) :: sk, ck, term, tsum, d, bk, ek, ce, fk, dk, cd, dbk, bkl, bkh
        real(WP) :: lk, sf, sd
        real(WP) :: sr, se, hr, he, sw, ew, dw
        real(WP), allocatable :: euv(:,:), zd(:,:)

        ! zd(k, m): with per-model metallicity, d L / d dz through age k (the basis, its EUV rows and
        ! L_bol, interpolated in log Z)
        allocate(euv(max(1, ntn), n_mod), zd(nt, n_mod))
#ifdef OMP_TARGET
        !$omp target data map(alloc: euv, zd)
#endif
#if !defined(OMP_TARGET)
        ! CPUs: per (model, age) lane-vectorized dot products of the basis columns with the cotangents
        ! (dot_lanes): the non-EUV and EUV pixels, of the two metallicities with per-model metallicity
        !$omp parallel do collapse(2) private(col, zl, zh, nlim, fk, lk, sr, se, hr, he, sw, ew, dw)
        do m = 1, n_mod
            do k = 1, nt
                col = n_mod + m
                if (k <= it_tesc(m)) col = m
                nlim = 0
                if (k <= ntn) nlim = whlylim
                fk = 1.0_KP
                if (k <= ntn) fk = real(fesc(m), KP)
                if (zmode) then
                    zl = zlo(m); zh = zhi(m)
                    sr = dot_lanes(nspec - nlim, basis(nlim + 1, k, zl), gw(nlim + 1, col))
                    hr = dot_lanes(nspec - nlim, basis(nlim + 1, k, zh), gw(nlim + 1, col))
                    se = dot_lanes(nlim, basis(1, k, zl), gw(1, col))
                    he = dot_lanes(nlim, basis(1, k, zh), gw(1, col))
                    sw = (1.0_wp - dz(m)) * sr + dz(m) * hr
                    ew = (1.0_wp - dz(m)) * se + dz(m) * he
                    dw = (hr - sr) + real(fk, WP) * (he - se)
                    lk = 10.0_wp**((1.0_wp - dz(m)) * blbol(k, zl) + dz(m) * blbol(k, zh))
                else
                    zl = 1; zh = 1
                    sw = dot_lanes(nspec - nlim, basis(nlim + 1, k, 1), gw(nlim + 1, col))
                    ew = dot_lanes(nlim, basis(1, k, 1), gw(1, col))
                    dw = 0.0_wp
                    lk = lum_linear(k)
                end if
                if (k <= ntn) euv(k, m) = ew * real(w2(k, col), WP)
                w_bar(k, m) = sw + real(fk, WP) * ew + lsum_bar(m) * lk
                zd(k, m) = 0.0_wp
                if (zmode) zd(k, m) = real(w2(k, col), WP) * (dw + lsum_bar(m) * lk * log(10.0_wp) * &
                                                               (blbol(k, zh) - blbol(k, zl)))
                if (k <= ntn) w_bar(k, m) = w_bar(k, m) + nwb(k, m)
            end do
        end do
#else
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(i, col, zl, zh, sk, ck, term, tsum, d, bk, lk, &
        !$omp& ek, ce, fk, nlim, dk, cd, dbk, bkl, bkh)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(i, col, zl, zh, sk, ck, term, tsum, d, bk, lk, ek, ce, fk, nlim, dk, cd, dbk, bkl, bkh)
#endif
        do m = 1, n_mod
            do k = 1, nt
                col = n_mod + m
                if (k <= it_tesc(m)) col = m
                ! The EUV pixels of a nebular age: summed apart (ek), then weighted by fesc
                nlim = 0
                if (k <= ntn) nlim = whlylim
                sk = 0.0_KP
                ck = 0.0_KP
                ek = 0.0_KP
                ce = 0.0_KP
                dk = 0.0_KP
                cd = 0.0_KP
                fk = 1.0_KP
                if (k <= ntn) fk = real(fesc(m), KP)
                if (zmode) then
                    zl = zlo(m); zh = zhi(m)
                    d = real(dz(m), KP)
                    if (nlim > 0) then
                        do i = 1, nlim
                            bkl = basis_t(k, i, zl)
                            bkh = basis_t(k, i, zh)
                            bk = (1.0_KP - d) * bkl + d * bkh
                            dbk = (bkh - bkl) * fk
                            term = dbk * gw(i, col) - cd
                            tsum = dk + term
                            cd = (tsum - dk) - term
                            dk = tsum
                            term = bk * gw(i, col) - ce
                            tsum = ek + term
                            ce = (tsum - ek) - term
                            ek = tsum
                        end do
                    end if
                    do i = nlim + 1, nspec
                        bkl = basis_t(k, i, zl)
                        bkh = basis_t(k, i, zh)
                        bk = (1.0_KP - d) * bkl + d * bkh
                        dbk = bkh - bkl
                        term = dbk * gw(i, col) - cd
                        tsum = dk + term
                        cd = (tsum - dk) - term
                        dk = tsum
                        term = bk * gw(i, col) - ck
                        tsum = sk + term
                        ck = (tsum - sk) - term
                        sk = tsum
                    end do
                    lk = 10.0_wp**((1.0_wp - dz(m)) * blbol(k, zl) + dz(m) * blbol(k, zh))
                else
                    if (nlim > 0) then
                        do i = 1, nlim
                            bk = basis_t(k, i, 1)
                            term = bk * gw(i, col) - ce
                            tsum = ek + term
                            ce = (tsum - ek) - term
                            ek = tsum
                        end do
                    end if
                    do i = nlim + 1, nspec
                        bk = basis_t(k, i, 1)
                        term = bk * gw(i, col) - ck
                        tsum = sk + term
                        ck = (tsum - sk) - term
                        sk = tsum
                    end do
                    lk = lum_linear(k)
                end if
                if (k <= ntn) euv(k, m) = real(ek, WP) * real(w2(k, col), WP)
                w_bar(k, m) = real(sk, WP) + real(fk, WP) * real(ek, WP) + lsum_bar(m) * lk
                zd(k, m) = 0.0_wp
                if (zmode) zd(k, m) = real(w2(k, col), WP) * (real(dk, WP) + lsum_bar(m) * lk * log(10.0_wp) * &
                                                               (blbol(k, zh) - blbol(k, zl)))
                if (k <= ntn) w_bar(k, m) = w_bar(k, m) + nwb(k, m)
            end do
        end do
#endif
        ! d L / d fesc per model
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(k, sf, sd)
#elif defined(_OPENMP)
        !$omp parallel do private(k, sf, sd)
#endif
        do m = 1, n_mod
            sf = 0.0_wp
            do k = 1, ntn
                sf = sf + euv(k, m)
            end do
            gneb(4, m) = sf
            sd = 0.0_wp
            do k = 1, nt
                sd = sd + zd(k, m)
            end do
            gneb(5, m) = gneb(5, m) + sd
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine backward_age_sums

    !> Adjoint of nebular_kernels' continuum and lines (add_neb_emission = 1, not the IMF per model):
    !> the young (old) sums gain, per nebular age t of the model's young (old) ages,
    !>     w_t q_t 10^lg(pixel or line, t),  q_t = |Q0(t)| (1 - frac_obrun),
    !> lg the CLOUDY log grid interpolated bilinearly in (gas_logz, gas_logu) weights (wz, wu) and
    !> linearly in age, the lines spread over the pixels by their profiles. So with y_bar the
    !> young / old cotangents (gw) and, for the lines, l_bar = sum over pixels of profile y_bar (lbar):
    !>   nwb(t) = d L / d w_t = q_t sum(y_bar 10^lg),  d L / d wz = sum_t w_t q_t sum(y_bar 10^lg ln 10 d lg / d wz),
    !>   the same for wu, and d L / d(1 - frac_obrun) = sum_t w_t |Q0(t)| sum(y_bar 10^lg) (gneb(1:3)).
    !> One thread per (model, nebular age), serial over pixels and lines, sums compensated at FP32
    !> kernel precision; 10^lg q as exp_kp(lg ln 10 + ln q) (exponent formed in FP64, rule 3).
    subroutine backward_nebular(n_mod, nspec, nt, nzb, ntn, cont, lines, it_tesc, zlo, zhi, dz, zmode, ia, wa, iz, wz, &
                                iu, wu, fabs, q0, lj_lo, lj_hi, nrun, rs, rl, rt, w2, ctab, ltab, gauss, gw, lbar, &
                                nwb, gneb, qmod, qb, lgiven)
#if defined(OPENSPS_HIP)
        use, intrinsic :: iso_c_binding, only: c_loc, c_null_ptr, c_int
#endif
        integer, intent(in), target :: n_mod, nspec, nt, nzb, ntn, it_tesc(n_mod), zlo(n_mod), zhi(n_mod)
        logical, intent(in)     :: cont, lines, zmode
        integer, intent(in), target :: ia(max(1, ntn)), iz(n_mod), iu(n_mod)
        integer, intent(in)     :: lj_lo(NEMLINE), lj_hi(NEMLINE), nrun
        integer, intent(in), target :: rs(max(1, ntn)), rl(max(1, ntn)), rt(max(1, ntn))
        real(WP), intent(in), target :: dz(n_mod), wa(max(1, ntn)), wz(n_mod), wu(n_mod), fabs(n_mod), q0(max(1, ntn), nzb)
        real(KP), intent(in), target :: w2(nt, 2 * n_mod), gw(nspec, 2 * n_mod)
        real(WP), intent(in), target :: ctab(nspec, NEBNZ, NEBNAGE, NEBNIP), ltab(:,:,:,:)
        real(KP), intent(in)    :: gauss(:,:)
        real(KP), intent(inout) :: lbar(:,:)
        real(WP), intent(out)   :: nwb(max(1, ntn), n_mod)
        real(WP), intent(inout) :: gneb(5, n_mod)   ! (1:3) and (5) set here
        !> The IMF per model: Q per (age, model), q0(t, m); qb = d L / d Q (the operator's reverse pass)
        logical, intent(in)     :: qmod
        real(WP), intent(inout) :: qb(max(1, ntn), n_mod)
        !> lbar given (the lines' absorbed part, backward_line_absorption), not from the profiles
        logical, intent(in)     :: lgiven

        integer  :: m, t, j, i, p, c, col, a, z0, u0, r, k
        real(KP) :: sw, sz, su, cw, cz, cu, term, tsum, val, gv
        real(WP) :: qz, lnq, c00, c01, c10, c11, ra, rb, raz, rbz, rau, rbu, lg, w1, fa, qf, s1, s2, s3, s4, s5
        real(WP), parameter :: LN10 = log(10.0_wp)
        real(WP), allocatable :: tz(:,:), tu(:,:), tf(:,:), tq(:,:)
        ! The continuum (cont): runs of consecutive nebular ages in one CLOUDY age interval (at most
        ! NEB_RUN ages), run r = ages rs(r) .. rs(r) + rl(r) - 1 (rt(t): the run of age t); per
        ! (model, run, block of pixels) a thread sums the pixels (neb_run_block), partial sums cpart
        ! (5 per age and block); qzt(t, m) the ionizing photons Q of age t, lref(r, m) the log Q of
        ! the run's first age with emission (the sums are taken with it, then scaled by Q / Q_ref)
        ! (npb pixel blocks per (model, run): GPUs, enough threads to fill the device (~64k), at most 64
        ! blocks and at least 8; CPUs 8)
        integer :: npb
        real(WP), allocatable, target :: qzt(:,:), lref(:,:)
        real(KP), allocatable :: cpart(:,:,:,:)
#if defined(OPENSPS_HIP)
        real(KP), allocatable, target :: csum(:,:,:)
#endif

        ! (tq: per-model metallicity, d L / d dz through Q, interpolated in log Z)
        allocate(tz(max(1, ntn), n_mod), tu(max(1, ntn), n_mod), tf(max(1, ntn), n_mod), tq(max(1, ntn), n_mod))
        allocate(qzt(max(1, ntn), n_mod), lref(max(1, nrun), n_mod))
#if defined(OPENSPS_HIP)
        if (cont .and. ntn > 0) then
            allocate(csum(5, ntn, n_mod))
        else
            allocate(csum(5, 1, 1))
        end if
        allocate(cpart(5, 1, 1, 1))
        npb = 1
#elif defined(OMP_TARGET)
        npb = max(8, min(64, 65536 / max(1, n_mod * nrun)))
        if (cont .and. ntn > 0) then
            allocate(cpart(5, ntn, npb, n_mod))
        else
            allocate(cpart(5, 1, 1, 1))
        end if
#else
        npb = 8
        if (cont .and. ntn > 0) then
            allocate(cpart(5, ntn, npb, n_mod))
        else
            allocate(cpart(5, 1, 1, 1))
        end if
#endif
#ifdef OMP_TARGET
#if defined(OPENSPS_HIP)
        !$omp target data map(alloc: tz, tu, tf, tq, qzt, lref, csum)
#else
        !$omp target data map(alloc: tz, tu, tf, tq, qzt, lref, cpart)
#endif
#endif
        ! Q per (age, model)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do t = 1, ntn
                if (qmod) then
                    qzt(t, m) = abs(q0(t, m))
                else if (zmode) then
                    qzt(t, m) = abs((1.0_wp - dz(m)) * q0(t, zlo(m)) + dz(m) * q0(t, zhi(m)))
                else
                    qzt(t, m) = abs(q0(t, 1))
                end if
            end do
        end do
        ! The continuum's sums: one thread per (model, run, block of pixels)
        if (cont .and. ntn > 0) then
#if defined(OPENSPS_HIP)
            !$omp target data use_device_addr(ctab, gw, it_tesc, ia, wa, rs, rl, iz, wz, iu, wu, fabs, qzt, lref, csum)
            if (KP == WP) then
                call opensps_hip_adj_nebular_continuum_dp(int(n_mod, c_int), int(nspec, c_int), &
                    int(size(ctab, 1), c_int), int(ntn, c_int), int(nrun, c_int), &
                    c_loc(ctab), c_loc(gw), c_loc(it_tesc), c_loc(ia), c_loc(wa), &
                    c_loc(rs), c_loc(rl), c_loc(iz), c_loc(wz), c_loc(iu), c_loc(wu), &
                    c_loc(fabs), c_loc(qzt), c_loc(lref), c_loc(csum), c_null_ptr)
            else
                call opensps_hip_adj_nebular_continuum_sp(int(n_mod, c_int), int(nspec, c_int), &
                    int(size(ctab, 1), c_int), int(ntn, c_int), int(nrun, c_int), &
                    c_loc(ctab), c_loc(gw), c_loc(it_tesc), c_loc(ia), c_loc(wa), &
                    c_loc(rs), c_loc(rl), c_loc(iz), c_loc(wz), c_loc(iu), c_loc(wu), &
                    c_loc(fabs), c_loc(qzt), c_loc(lref), c_loc(csum), c_null_ptr)
            end if
            !$omp end target data
#else
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(3) private(t, lnq)
#elif defined(_OPENMP)
            !$omp parallel do collapse(3) private(t, lnq)
#endif
            do m = 1, n_mod
                do r = 1, nrun
                    do p = 1, npb
                        ! (the log Q of the run's first age with emission; none: the run adds nothing)
                        lnq = huge(1.0_wp)
                        do t = rs(r), rs(r) + rl(r) - 1
                            if (qzt(t, m) * abs(fabs(m)) > SAFE_FLOOR) then
                                lnq = log(qzt(t, m))
                                exit
                            end if
                        end do
                        if (p == 1) lref(r, m) = lnq
                        call neb_run_block(nspec, n_mod, ntn, npb, ctab, gw, wa, qzt, fabs, it_tesc, m, rs(r), rl(r), &
                                           ia(rs(r)), iz(m), iu(m), wz(m), wu(m), lnq, p, cpart)
                    end do
                end do
            end do
#endif
        end if
        ! The lines' cotangents (profiles transposed): one thread per (line, young / old column), over
        ! the pixels whose profile lists include the line (lj_lo(i) to lj_hi(i)), in
        ! pixel order as a pass over the pixels would add them
        if (lines .and. .not. lgiven) then
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2) private(j, val)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2) private(j, val)
#endif
            do c = 1, 2 * n_mod
                do i = 1, NEMLINE
                    val = 0.0_KP
                    do j = lj_lo(i), lj_hi(i)
                        val = val + gauss(j, i) * gw(j, c)
                    end do
                    lbar(i, c) = val
                end do
            end do
        end if

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(col, a, z0, u0, sw, sz, su, cw, cz, cu, term, &
        !$omp& tsum, val, gv, qz, lnq, c00, c01, c10, c11, ra, rb, raz, rbz, rau, rbu, lg, w1, fa, j, i, qf, s1, s2, &
        !$omp& s3, s4, s5, k)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(col, a, z0, u0, sw, sz, su, cw, cz, cu, term, tsum, val, gv, qz, lnq, &
        !$omp& c00, c01, c10, c11, ra, rb, raz, rbz, rau, rbu, lg, w1, fa, j, i, qf, s1, s2, s3, s4, s5, k)
#endif
        do m = 1, n_mod
            do t = 1, ntn
                col = n_mod + m
                if (t <= it_tesc(m)) col = m
                qz = qzt(t, m)
                fa = abs(fabs(m))
                nwb(t, m) = 0.0_wp
                tz(t, m) = 0.0_wp
                tu(t, m) = 0.0_wp
                tf(t, m) = 0.0_wp
                tq(t, m) = 0.0_wp
                if (qz * fa > SAFE_FLOOR) then
                    lnq = log(qz)
                    a = ia(t)
                    z0 = iz(m)
                    u0 = iu(m)
                    w1 = wa(t)
                    c00 = (1.0_wp - wz(m)) * (1.0_wp - wu(m))
                    c01 = (1.0_wp - wz(m)) * wu(m)
                    c10 = wz(m) * (1.0_wp - wu(m))
                    c11 = wz(m) * wu(m)
                    sw = 0.0_KP; sz = 0.0_KP; su = 0.0_KP
                    cw = 0.0_KP; cz = 0.0_KP; cu = 0.0_KP
                    if (lines) then
                        do i = 1, NEMLINE
                            ra = c00 * ltab(i, z0, a, u0) + c01 * ltab(i, z0, a, u0 + 1) + &
                                 c10 * ltab(i, z0 + 1, a, u0) + c11 * ltab(i, z0 + 1, a, u0 + 1)
                            rb = c00 * ltab(i, z0, a + 1, u0) + c01 * ltab(i, z0, a + 1, u0 + 1) + &
                                 c10 * ltab(i, z0 + 1, a + 1, u0) + c11 * ltab(i, z0 + 1, a + 1, u0 + 1)
                            raz = (1.0_wp - wu(m)) * (ltab(i, z0 + 1, a, u0) - ltab(i, z0, a, u0)) + &
                                  wu(m) * (ltab(i, z0 + 1, a, u0 + 1) - ltab(i, z0, a, u0 + 1))
                            rbz = (1.0_wp - wu(m)) * (ltab(i, z0 + 1, a + 1, u0) - ltab(i, z0, a + 1, u0)) + &
                                  wu(m) * (ltab(i, z0 + 1, a + 1, u0 + 1) - ltab(i, z0, a + 1, u0 + 1))
                            rau = (1.0_wp - wz(m)) * (ltab(i, z0, a, u0 + 1) - ltab(i, z0, a, u0)) + &
                                  wz(m) * (ltab(i, z0 + 1, a, u0 + 1) - ltab(i, z0 + 1, a, u0))
                            rbu = (1.0_wp - wz(m)) * (ltab(i, z0, a + 1, u0 + 1) - ltab(i, z0, a + 1, u0)) + &
                                  wz(m) * (ltab(i, z0 + 1, a + 1, u0 + 1) - ltab(i, z0 + 1, a + 1, u0))
                            lg = (1.0_wp - w1) * ra + w1 * rb
                            val = exp_kp(lg * LN10 + lnq)
                            gv = lbar(i, col) * val
                            term = gv - cw; tsum = sw + term; cw = (tsum - sw) - term; sw = tsum
                            term = gv * real(LN10 * ((1.0_wp - w1) * raz + w1 * rbz), KP) - cz
                            tsum = sz + term; cz = (tsum - sz) - term; sz = tsum
                            term = gv * real(LN10 * ((1.0_wp - w1) * rau + w1 * rbu), KP) - cu
                            tsum = su + term; cu = (tsum - su) - term; su = tsum
                        end do
                    end if
                    ! (the continuum: the blocks' sums, scaled by Q / Q_ref of the run; s1 .. s5 as in
                    ! neb_run_block)
                    s1 = real(sw, WP)
                    s2 = real(sz, WP)
                    s3 = real(su, WP)
                    if (cont) then
                        qf = exp(lnq - lref(rt(t), m))
                        s4 = 0.0_wp
                        s5 = 0.0_wp
#if defined(OPENSPS_HIP)
                        s1 = s1 + qf * real(csum(1, t, m), WP)
                        s4 = real(csum(2, t, m), WP) * (1.0_wp - w1) + real(csum(3, t, m), WP) * w1
                        s5 = real(csum(4, t, m), WP) * (1.0_wp - w1) + real(csum(5, t, m), WP) * w1
#else
                        do k = 1, npb
                            s1 = s1 + qf * real(cpart(1, t, k, m), WP)
                        end do
                        do k = 1, npb
                            s4 = s4 + real(cpart(2, t, k, m), WP) * (1.0_wp - w1) + real(cpart(3, t, k, m), WP) * w1
                            s5 = s5 + real(cpart(4, t, k, m), WP) * (1.0_wp - w1) + real(cpart(5, t, k, m), WP) * w1
                        end do
#endif
                        s2 = s2 + qf * LN10 * s4
                        s3 = s3 + qf * LN10 * s5
                    end if
                    nwb(t, m) = fa * s1
                    tz(t, m) = real(w2(t, col), WP) * fa * s2
                    tu(t, m) = real(w2(t, col), WP) * fa * s3
                    tf(t, m) = real(w2(t, col), WP) * s1
                    if (qmod) qb(t, m) = real(w2(t, col), WP) * fa * s1 / qz
                    if (zmode) tq(t, m) = real(w2(t, col), WP) * fa * s1 / qz * &
                                          sign(1.0_wp, (1.0_wp - dz(m)) * q0(t, zlo(m)) + dz(m) * q0(t, zhi(m))) * &
                                          (q0(t, zhi(m)) - q0(t, zlo(m)))
                end if
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(t)
#elif defined(_OPENMP)
        !$omp parallel do private(t)
#endif
        do m = 1, n_mod
            gneb(1, m) = 0.0_wp
            gneb(2, m) = 0.0_wp
            gneb(3, m) = 0.0_wp
            gneb(5, m) = 0.0_wp
            do t = 1, ntn
                gneb(1, m) = gneb(1, m) + tz(t, m)
                gneb(2, m) = gneb(2, m) + tu(t, m)
                gneb(3, m) = gneb(3, m) + tf(t, m)
                gneb(5, m) = gneb(5, m) + tq(t, m)
            end do
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine backward_nebular

    !> The nebular continuum's adjoint for one (model, run of ages, block of pixels): over the block's
    !> pixels, per age t of the run (at most NEB_RUN = 4), the sums of g = S_bar(young or old column)
    !> 10**lg(t) Q_ref and of g times the interval ends' z and U slopes (backward_nebular weights them
    !> by the age's position wa and scales by Q(t) / Q_ref). Within a CLOUDY age interval
    !> lg = ra + wa (rb - ra), so 10**lg advances by exp(dwa (rb - ra) ln 10) from age to age: one
    !> exponential per pixel and run (and one per change of the step), as the forward's continuum
    !> (nebular_kernels). Exponents in FP64 (exp_kp); FP32 kernels sum with compensation. The 20 sums
    !> are scalars (an array of them spilled to scratch memory on the R9700).
    subroutine neb_run_block(nspec, n_mod, ntn, npb, ctab, gw, wa, qzt, fabs, it_tesc, m, ts, tl, a, z0, u0, wzm, &
                             wum, lnq, p, cpart)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)     :: nspec, n_mod, ntn, npb, m, ts, tl, a, z0, u0, p
        integer, intent(in)     :: it_tesc(n_mod)
        real(WP), intent(in)    :: ctab(nspec, NEBNZ, NEBNAGE, NEBNIP), wa(max(1, ntn)), qzt(max(1, ntn), n_mod)
        real(WP), intent(in)    :: fabs(n_mod), wzm, wum, lnq
        real(KP), intent(in)    :: gw(nspec, 2 * n_mod)
        real(KP), intent(inout) :: cpart(5, ntn, npb, n_mod)
        real(WP), parameter :: LN10 = log(10.0_wp)
        real(KP) :: e, rr, gy, go, g, term, tsum, rz_a, rz_b, ru_a, ru_b
        real(KP) :: a11, a21, a31, a41, a51, a12, a22, a32, a42, a52, a13, a23, a33, a43, a53, a14, a24, a34, a44, a54
        real(KP) :: k11, k21, k31, k41, k51, k12, k22, k32, k42, k52, k13, k23, k33, k43, k53, k14, k24, k34, k44, k54
        real(WP) :: c00, c01, c10, c11, ra, rb, raz, rbz, rau, rbu, dl, dw, dwp
        logical  :: ok1, ok2, ok3, ok4
        integer  :: j, j0, j1, nbk

        ok1 = .false.; ok2 = .false.; ok3 = .false.; ok4 = .false.
        if (tl >= 1) ok1 = qzt(ts, m) * abs(fabs(m)) > SAFE_FLOOR
        if (tl >= 2) ok2 = qzt(ts + 1, m) * abs(fabs(m)) > SAFE_FLOOR
        if (tl >= 3) ok3 = qzt(ts + 2, m) * abs(fabs(m)) > SAFE_FLOOR
        if (tl >= 4) ok4 = qzt(ts + 3, m) * abs(fabs(m)) > SAFE_FLOOR
        a11 = 0.0_KP
        k11 = 0.0_KP
        a21 = 0.0_KP
        k21 = 0.0_KP
        a31 = 0.0_KP
        k31 = 0.0_KP
        a41 = 0.0_KP
        k41 = 0.0_KP
        a51 = 0.0_KP
        k51 = 0.0_KP
        a12 = 0.0_KP
        k12 = 0.0_KP
        a22 = 0.0_KP
        k22 = 0.0_KP
        a32 = 0.0_KP
        k32 = 0.0_KP
        a42 = 0.0_KP
        k42 = 0.0_KP
        a52 = 0.0_KP
        k52 = 0.0_KP
        a13 = 0.0_KP
        k13 = 0.0_KP
        a23 = 0.0_KP
        k23 = 0.0_KP
        a33 = 0.0_KP
        k33 = 0.0_KP
        a43 = 0.0_KP
        k43 = 0.0_KP
        a53 = 0.0_KP
        k53 = 0.0_KP
        a14 = 0.0_KP
        k14 = 0.0_KP
        a24 = 0.0_KP
        k24 = 0.0_KP
        a34 = 0.0_KP
        k34 = 0.0_KP
        a44 = 0.0_KP
        k44 = 0.0_KP
        a54 = 0.0_KP
        k54 = 0.0_KP
        if (lnq < huge(1.0_wp)) then
            c00 = (1.0_wp - wzm) * (1.0_wp - wum)
            c01 = (1.0_wp - wzm) * wum
            c10 = wzm * (1.0_wp - wum)
            c11 = wzm * wum
            nbk = (nspec + npb - 1) / npb
            j0 = (p - 1) * nbk + 1
            j1 = min(p * nbk, nspec)
            do j = j0, j1
                gy = gw(j, m)
                go = gw(j, n_mod + m)
                if (gy == 0.0_KP .and. go == 0.0_KP) cycle
                ra = c00 * ctab(j, z0, a, u0) + c01 * ctab(j, z0, a, u0 + 1) + &
                     c10 * ctab(j, z0 + 1, a, u0) + c11 * ctab(j, z0 + 1, a, u0 + 1)
                rb = c00 * ctab(j, z0, a + 1, u0) + c01 * ctab(j, z0, a + 1, u0 + 1) + &
                     c10 * ctab(j, z0 + 1, a + 1, u0) + c11 * ctab(j, z0 + 1, a + 1, u0 + 1)
                raz = (1.0_wp - wum) * (ctab(j, z0 + 1, a, u0) - ctab(j, z0, a, u0)) + &
                      wum * (ctab(j, z0 + 1, a, u0 + 1) - ctab(j, z0, a, u0 + 1))
                rbz = (1.0_wp - wum) * (ctab(j, z0 + 1, a + 1, u0) - ctab(j, z0, a + 1, u0)) + &
                      wum * (ctab(j, z0 + 1, a + 1, u0 + 1) - ctab(j, z0, a + 1, u0 + 1))
                rau = (1.0_wp - wzm) * (ctab(j, z0, a, u0 + 1) - ctab(j, z0, a, u0)) + &
                      wzm * (ctab(j, z0 + 1, a, u0 + 1) - ctab(j, z0 + 1, a, u0))
                rbu = (1.0_wp - wzm) * (ctab(j, z0, a + 1, u0 + 1) - ctab(j, z0, a + 1, u0)) + &
                      wzm * (ctab(j, z0 + 1, a + 1, u0 + 1) - ctab(j, z0 + 1, a + 1, u0))
                rz_a = real(raz, KP)
                rz_b = real(rbz, KP)
                ru_a = real(rau, KP)
                ru_b = real(rbu, KP)
                dl = (rb - ra) * LN10
                e = exp_kp(ra * LN10 + wa(ts) * dl + lnq)
                rr = 1.0_KP
                dwp = 0.0_wp
                ! (age 1 of the run)
                if (tl >= 1 .and. ok1) then
                    if (ts + 0 <= it_tesc(m)) then
                        g = gy * e
                    else
                        g = go * e
                    end if
                    if (KP == WP) then
                        a11 = a11 + g
                        a21 = a21 + g * rz_a
                        a31 = a31 + g * rz_b
                        a41 = a41 + g * ru_a
                        a51 = a51 + g * ru_b
                    else
                        term = g - k11
                        tsum = a11 + term
                        k11 = (tsum - a11) - term
                        a11 = tsum
                        term = g * rz_a - k21
                        tsum = a21 + term
                        k21 = (tsum - a21) - term
                        a21 = tsum
                        term = g * rz_b - k31
                        tsum = a31 + term
                        k31 = (tsum - a31) - term
                        a31 = tsum
                        term = g * ru_a - k41
                        tsum = a41 + term
                        k41 = (tsum - a41) - term
                        a41 = tsum
                        term = g * ru_b - k51
                        tsum = a51 + term
                        k51 = (tsum - a51) - term
                        a51 = tsum
                    end if
                end if
                ! (age 2 of the run)
                if (tl >= 2) then
                    dw = wa(ts + 1) - wa(ts + 0)
                    if (dw /= dwp) then
                        rr = exp_kp(dw * dl)
                        dwp = dw
                    end if
                    e = e * rr
                end if
                if (tl >= 2 .and. ok2) then
                    if (ts + 1 <= it_tesc(m)) then
                        g = gy * e
                    else
                        g = go * e
                    end if
                    if (KP == WP) then
                        a12 = a12 + g
                        a22 = a22 + g * rz_a
                        a32 = a32 + g * rz_b
                        a42 = a42 + g * ru_a
                        a52 = a52 + g * ru_b
                    else
                        term = g - k12
                        tsum = a12 + term
                        k12 = (tsum - a12) - term
                        a12 = tsum
                        term = g * rz_a - k22
                        tsum = a22 + term
                        k22 = (tsum - a22) - term
                        a22 = tsum
                        term = g * rz_b - k32
                        tsum = a32 + term
                        k32 = (tsum - a32) - term
                        a32 = tsum
                        term = g * ru_a - k42
                        tsum = a42 + term
                        k42 = (tsum - a42) - term
                        a42 = tsum
                        term = g * ru_b - k52
                        tsum = a52 + term
                        k52 = (tsum - a52) - term
                        a52 = tsum
                    end if
                end if
                ! (age 3 of the run)
                if (tl >= 3) then
                    dw = wa(ts + 2) - wa(ts + 1)
                    if (dw /= dwp) then
                        rr = exp_kp(dw * dl)
                        dwp = dw
                    end if
                    e = e * rr
                end if
                if (tl >= 3 .and. ok3) then
                    if (ts + 2 <= it_tesc(m)) then
                        g = gy * e
                    else
                        g = go * e
                    end if
                    if (KP == WP) then
                        a13 = a13 + g
                        a23 = a23 + g * rz_a
                        a33 = a33 + g * rz_b
                        a43 = a43 + g * ru_a
                        a53 = a53 + g * ru_b
                    else
                        term = g - k13
                        tsum = a13 + term
                        k13 = (tsum - a13) - term
                        a13 = tsum
                        term = g * rz_a - k23
                        tsum = a23 + term
                        k23 = (tsum - a23) - term
                        a23 = tsum
                        term = g * rz_b - k33
                        tsum = a33 + term
                        k33 = (tsum - a33) - term
                        a33 = tsum
                        term = g * ru_a - k43
                        tsum = a43 + term
                        k43 = (tsum - a43) - term
                        a43 = tsum
                        term = g * ru_b - k53
                        tsum = a53 + term
                        k53 = (tsum - a53) - term
                        a53 = tsum
                    end if
                end if
                ! (age 4 of the run)
                if (tl >= 4) then
                    dw = wa(ts + 3) - wa(ts + 2)
                    if (dw /= dwp) then
                        rr = exp_kp(dw * dl)
                        dwp = dw
                    end if
                    e = e * rr
                end if
                if (tl >= 4 .and. ok4) then
                    if (ts + 3 <= it_tesc(m)) then
                        g = gy * e
                    else
                        g = go * e
                    end if
                    if (KP == WP) then
                        a14 = a14 + g
                        a24 = a24 + g * rz_a
                        a34 = a34 + g * rz_b
                        a44 = a44 + g * ru_a
                        a54 = a54 + g * ru_b
                    else
                        term = g - k14
                        tsum = a14 + term
                        k14 = (tsum - a14) - term
                        a14 = tsum
                        term = g * rz_a - k24
                        tsum = a24 + term
                        k24 = (tsum - a24) - term
                        a24 = tsum
                        term = g * rz_b - k34
                        tsum = a34 + term
                        k34 = (tsum - a34) - term
                        a34 = tsum
                        term = g * ru_a - k44
                        tsum = a44 + term
                        k44 = (tsum - a44) - term
                        a44 = tsum
                        term = g * ru_b - k54
                        tsum = a54 + term
                        k54 = (tsum - a54) - term
                        a54 = tsum
                    end if
                end if
            end do
        end if
        if (tl >= 1) then
            cpart(1, ts + 0, p, m) = a11
            cpart(2, ts + 0, p, m) = a21
            cpart(3, ts + 0, p, m) = a31
            cpart(4, ts + 0, p, m) = a41
            cpart(5, ts + 0, p, m) = a51
        end if
        if (tl >= 2) then
            cpart(1, ts + 1, p, m) = a12
            cpart(2, ts + 1, p, m) = a22
            cpart(3, ts + 1, p, m) = a32
            cpart(4, ts + 1, p, m) = a42
            cpart(5, ts + 1, p, m) = a52
        end if
        if (tl >= 3) then
            cpart(1, ts + 2, p, m) = a13
            cpart(2, ts + 2, p, m) = a23
            cpart(3, ts + 2, p, m) = a33
            cpart(4, ts + 2, p, m) = a43
            cpart(5, ts + 2, p, m) = a53
        end if
        if (tl >= 4) then
            cpart(1, ts + 3, p, m) = a14
            cpart(2, ts + 3, p, m) = a24
            cpart(3, ts + 3, p, m) = a34
            cpart(4, ts + 3, p, m) = a44
            cpart(5, ts + 3, p, m) = a54
        end if
    end subroutine neb_run_block


    !> Compensated add (CPU builds at FP32 kernel precision).
    subroutine kadd(s, c, x)
        real(KP), intent(inout) :: s, c
        real(KP), intent(in)    :: x
        real(KP) :: t, y
        if (KP == WP) then
            s = s + x
        else
            y = x - c
            t = s + y
            c = (t - s) - y
            s = t
        end if
    end subroutine kadd

end module opensps_csp_batch
