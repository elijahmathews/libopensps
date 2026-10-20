module fsps_context_types
    !> @brief
    !> Core derived types that define FSPS contexts and shared state.
    !>
    !> @details
    !> Provides the `fsps_context_t` and `fsps_context_state_t` structures along
    !> with lifecycle helpers for releasing associated memory.

    use fsps_precision, only: WP, KP
    use fsps_constants, only: &
        NDIM_LOGT, NDIM_LOGG, NDIM_WMB_LOGT, NDIM_WMB_LOGG, &
        N_AGB_CAR, NDIM_PAGB, NDIM_WR, NTAU_DAGB, NTEFF_DAGB, &
        NEMLINE, NEBNZ, NEBNAGE, NEBNIP, NAGNDUST, NTABMAX
    use fsps_types, only: PARAMS, COMPSPOUT, TLSF, OBSDAT
    use fsps_cache, only: fsps_setup_cache_t
    use, intrinsic :: iso_c_binding, only: c_int, c_size_t, c_loc, c_ptr

    implicit none
    private

    public :: fsps_context_state_t
    public :: fsps_context_t
    public :: fsps_context_state_destroy
    public :: fsps_context_sync_device_settings
    public :: batch_grid_t, batch_grid_release
    public :: batch_out_t, batch_out_reserve, batch_out_release

    ! ---------------------------------------------------------------------
    ! Module constants
    ! ---------------------------------------------------------------------

    !> Grid-only data of the batched CSP (fsps_csp_batch) for one redshift: observed-frame map,
    !> band weights and supports, trapezoid weights, IGM transmission. Built on the first batch
    !> call for a redshift and kept (on the device too) until the redshift or IGM factor changes.
    type :: batch_grid_t
        logical  :: ready = .false.
        logical  :: igm = .false.
        real(WP) :: zred = 0.0_wp
        real(WP) :: igm_factor = 0.0_wp
        real(WP) :: dist_term = 0.0_wp
        integer,  allocatable :: i_lo(:), j_lo(:), j_hi(:)
        real(WP), allocatable :: t_obs(:)
        real(KP), allocatable :: wt(:), wtl(:), wl(:), bands_t(:,:), cmat(:,:), igm_t(:)
        !> Velocity smoothing operator (build_velocity_kernel) for sm_sigma, sm_minl, sm_maxl,
        !> on the device too (prepare_smoothing_kernel); released with the rest
        logical  :: sm_ready = .false.
        real(WP) :: sm_sigma = 0.0_wp, sm_minl = 0.0_wp, sm_maxl = 0.0_wp
        integer,  allocatable :: sm_lo(:), sm_hi(:), sm_off(:)
        real(WP), allocatable :: sm_coef(:)
    end type batch_grid_t

    !> Spectra of the last batched CSP (kernel precision, nspec x models), kept on the device
    !> between calls: the batch's working array, so it is allocated (and its host copy page-locked
    !> in NVIDIA GPU builds) only when a batch outgrows it, and callers working on the device can
    !> use the spectra there (batch_spectra_device) without copying them back.
#ifdef FSPS_PINNED
    !> Page-locking of host buffers (src/core/fsps_pinned.c)
    interface
        integer(c_int) function fsps_host_register(ptr, bytes) bind(C, name="fsps_host_register")
            import :: c_int, c_ptr, c_size_t
            type(c_ptr), value :: ptr
            integer(c_size_t), value :: bytes
        end function fsps_host_register
        integer(c_int) function fsps_host_unregister(ptr) bind(C, name="fsps_host_unregister")
            import :: c_int, c_ptr
            type(c_ptr), value :: ptr
        end function fsps_host_unregister
    end interface
#endif

    type :: batch_out_t
        real(KP), allocatable :: spec(:,:)
        integer :: n = 0             !< models of the last batch in spec
        logical :: pinned = .false.  !< host copy page-locked (fsps_host_register)
    end type batch_out_t

    type :: fsps_context_state_t
        real(WP) :: zsol = 0.0_wp
        real(WP) :: zsol_spec = 0.0_wp
        character(len=64) :: isoc_type = ''
        character(len=64) :: spec_type = ''
        !> Axis selections of the isochrone and spectral libraries (standard format), e.g.
        !> 'afe=0.4' from a library name 'mist:afe=0.4'; empty: each axis's default
        character(len=64) :: isoc_select = ''
        character(len=64) :: spec_select = ''
        integer :: nt = 0
        integer :: nz = 0
        integer :: nspec = 0
        integer :: nzinit = 0
        integer :: nbands = 0
        integer :: nindx = 0
        integer :: ntfull = 0
        integer :: nspec_xrb = 0
        integer :: nt_xrb = 0
        integer :: nz_xrb = 0
        integer :: check_sps_setup = 0
        real(WP) :: tuniv = 0.0_wp
        integer :: whlam5000 = 0
        integer :: whlylim = 0
        real(WP) :: zpow2 = 1.0_wp
        integer, dimension(6) :: mwdindex = 0
        real(WP), dimension(500, 3) :: cosmospl = 0.0_wp
        integer :: ntabsfh = 0
        real(WP), dimension(3, NTABMAX) :: sfh_tab = 0.0_wp
        real(WP), dimension(3) :: imf_alpha = 1.3_wp
        real(WP) :: imf_vdmc = 0.08_wp
        real(WP) :: imf_mdave = 0.5_wp
        integer :: n_user_imf = 0
        real(WP), dimension(3, 100) :: imf_user_alpha = 0.0_wp
        real(WP) :: salp_ind = 2.35_wp
        real(WP) :: imf_lower_limit = 0.08_wp
        real(WP) :: imf_upper_limit = 120.0_wp
        real(WP) :: imf_lower_bound = 0.0_wp
        real(WP) :: mlim_bh = 40.0_wp
        real(WP) :: mlim_ns = 8.5_wp
        character(len=30) :: alt_filter_file = ''
        real(WP), pointer, contiguous :: indexdefined(:, :) => null()
        real(WP), pointer, contiguous :: wgdust(:, :, :, :) => null()
        real(WP), pointer, contiguous :: g03smcextn(:) => null()
        real(WP), pointer, contiguous :: bands(:, :) => null()
        real(WP), pointer, contiguous :: magsun(:) => null()
        real(WP), pointer, contiguous :: magvega(:) => null()
        real(WP), pointer, contiguous :: filter_leff(:) => null()
        real(WP), pointer, contiguous :: vega_spec(:) => null()
        real(WP), pointer, contiguous :: sun_spec(:) => null()
        real(WP), pointer, contiguous :: spec_lambda(:) => null()
        real(WP), pointer, contiguous :: spec_nu(:) => null()
        real(WP), pointer, contiguous :: spec_res(:) => null()
        real(WP), dimension(NDIM_LOGT) :: speclib_logt = 0.0_wp
        real(WP), dimension(NDIM_LOGG) :: speclib_logg = 0.0_wp
        real(kind(1.0)), pointer, contiguous :: speclib(:, :, :, :) => null()
        real(WP), dimension(NDIM_WMB_LOGT) :: wmb_logt = 0.0_wp
        real(WP), dimension(NDIM_WMB_LOGG) :: wmb_logg = 0.0_wp
        real(kind(1.0)), pointer, contiguous :: wmb_spec(:, :, :, :) => null()
        real(KP), pointer, contiguous :: agb_spec_o(:, :) => null()
        real(WP), pointer, contiguous :: agb_logt_o(:, :) => null()
        real(KP), pointer, contiguous :: agb_spec_c(:, :) => null()
        real(WP), pointer, contiguous :: agb_logt_c(:) => null()
        real(WP), dimension(N_AGB_CAR) :: agb_logt_car = 0.0_wp
        real(KP), pointer, contiguous :: agb_spec_car(:, :) => null()
        !> Templates used for dusty AGB stars, smoothed as legacy ADD_AGB_DUST smooths each such star
        !> (10^4 km/s, lambda >= 3 um); only the smoothed segment lambda >= AGB_SMOOTH_LAMBDA_MIN is
        !> stored. O-rich and Lancon-Wood C-rich templates include their lambda^2/c factor.
        real(KP), pointer, contiguous :: speclib_agb_smooth(:, :, :, :) => null()
        real(KP), pointer, contiguous :: agb_spec_o_smooth(:, :) => null()
        real(KP), pointer, contiguous :: agb_spec_c_smooth(:, :) => null()
        real(KP), pointer, contiguous :: agb_spec_car_smooth(:, :) => null()
        real(KP), pointer, contiguous :: pagb_spec(:, :, :) => null()
        real(WP), dimension(NDIM_PAGB) :: pagb_logt = 0.0_wp
        real(KP), pointer, contiguous :: wrn_spec(:, :, :) => null()
        real(KP), pointer, contiguous :: wrc_spec(:, :, :) => null()
        real(WP), dimension(NDIM_WR) :: wrn_logt = 0.0_wp
        real(WP), dimension(NDIM_WR) :: wrc_logt = 0.0_wp
        integer :: ndim_dustem = 0
        integer :: numin_dustem = 0
        integer :: nqpah_dustem = 0
        character(len=6) :: str_dustem = 'DL07'
        real(WP), pointer, contiguous :: qpaharr(:) => null()
        real(WP), pointer, contiguous :: uminarr(:) => null()
        real(WP), pointer, contiguous :: lambda_dustem(:) => null()
        real(WP), pointer, contiguous :: dustem_dustem(:, :) => null()
        real(WP), pointer, contiguous :: dustem2_dustem(:, :, :) => null()
        real(KP), pointer, contiguous :: flux_dagb(:, :, :, :) => null()
        real(WP), dimension(2, NTAU_DAGB) :: tau1_dagb = 0.0_wp
        real(WP), dimension(2, NTEFF_DAGB) :: teff_dagb = 0.0_wp
        real(WP), dimension(NEMLINE) :: nebem_line_pos = 0.0_wp
        real(WP), dimension(NEMLINE, NEBNZ, NEBNAGE, NEBNIP) :: nebem_line = 0.0_wp
        real(WP), dimension(NEMLINE, NEBNZ, NEBNAGE, NEBNIP) :: xnebem_line = 0.0_wp
        real(WP), pointer, contiguous :: nebem_cont(:, :, :, :) => null()
        real(WP), pointer, contiguous :: xnebem_cont(:, :, :, :) => null()
        real(WP), dimension(NEBNZ) :: nebem_logz = 0.0_wp
        real(WP), dimension(NEBNAGE) :: nebem_age = 0.0_wp
        real(WP), dimension(NEBNIP) :: nebem_logu = 0.0_wp
        real(WP), pointer, contiguous :: neb_res_min(:) => null()
        real(KP), pointer, contiguous :: gaussnebarr(:, :) => null()
        !> True when gaussnebarr is a private per-context allocation (per-call kernels) rather than a
        !> pointer into the shared setup cache; it is then deallocated, not nullified, on destroy
        logical :: owns_gaussnebarr = .false.
        !> sigma_smooth and smooth_velocity the per-call line kernels (gaussnebarr) were last built
        !> for; compute_line_gaussians is skipped while they are unchanged (negative: not built)
        real(WP) :: gauss_built_sigma = -1.0_wp
        integer :: gauss_built_velocity = -1
        real(WP), dimension(NAGNDUST) :: agndust_tau = 0.0_wp
        real(WP), pointer, contiguous :: agndust_spec(:, :) => null()
        real(WP), pointer, contiguous :: mact_isoc(:, :, :) => null()
        real(WP), pointer, contiguous :: logl_isoc(:, :, :) => null()
        real(WP), pointer, contiguous :: logt_isoc(:, :, :) => null()
        real(WP), pointer, contiguous :: logg_isoc(:, :, :) => null()
        real(WP), pointer, contiguous :: ffco_isoc(:, :, :) => null()
        real(WP), pointer, contiguous :: phase_isoc(:, :, :) => null()
        real(WP), pointer, contiguous :: mini_isoc(:, :, :) => null()
        real(WP), pointer, contiguous :: lmdot_isoc(:, :, :) => null()
        integer, pointer, contiguous :: nmass_isoc(:, :) => null()
        real(WP), pointer, contiguous :: timestep_isoc(:, :) => null()
        real(WP), pointer, contiguous :: zlegend(:) => null()
        real(WP), pointer, contiguous :: zlegendinit(:) => null()
        real(WP), allocatable :: spec_ssp_zz(:, :, :)
        real(WP), allocatable :: mass_ssp_zz(:, :)
        real(WP), allocatable :: lbol_ssp_zz(:, :)
        real(WP), pointer, contiguous :: time_full(:) => null()
        real(WP), allocatable :: weight_ssp(:, :)
        real(KP), allocatable :: spec_young(:,:)
        real(KP), allocatable :: spec_old(:,:)
        real(KP), allocatable :: ssp_temp_grid(:, :)
        integer, allocatable :: ssp_active_idx(:)
        real(WP), allocatable :: ssp_active_w(:)
        integer :: ssp_temp_nspec = 0
        integer :: ssp_temp_nstars = 0
        real(WP), pointer, contiguous :: bpass_spec_ssp(:, :, :) => null()
        real(WP), pointer, contiguous :: bpass_mass_ssp(:, :) => null()
        real(WP), pointer, contiguous :: lam_xrb(:) => null()
        real(WP), pointer, contiguous :: spec_xrb(:, :, :) => null()
        real(WP), pointer, contiguous :: ages_xrb(:) => null()
        real(WP), pointer, contiguous :: zmet_xrb(:) => null()
        type(TLSF) :: lsfinfo
        type(OBSDAT) :: powell_data
        type(OBSDAT) :: sedfit_data

        ! --- Persistent CSP Workspace ---
        real(KP), allocatable :: ssp_basis_spec(:, :, :)
        real(WP), allocatable :: ssp_basis_mass(:, :)
        real(WP), allocatable :: ssp_basis_lbol(:, :)
        logical :: ssp_basis_is_dirty = .true.
        real(KP), allocatable :: csp_ssp_grid(:,:,:)
        real(KP), allocatable :: csp_emlin_grid(:,:,:)
        real(WP), allocatable :: csp_ssp_lum_linear(:,:)
        real(KP), allocatable :: csp_igm_transmission(:)
        real(WP), allocatable :: csp_spec_final(:)
        real(WP), allocatable :: csp_emlin_final(:)

        ! --- Replacements for csp_buffer_t ---
        real(WP), allocatable :: csp_weights(:,:,:)
        real(KP), allocatable :: csp_emlin_young(:,:)
        real(KP), allocatable :: csp_emlin_old(:,:)

        ! --- Persistent Physics Workspaces (Dust & Gas) ---
        real(KP), allocatable :: dust_transmission_diffuse(:)
        real(WP), allocatable :: dust_frequencies(:)
        real(KP), allocatable :: dust_spec_total_work(:,:)
        real(KP), allocatable :: dust_emission_shape(:)
        real(KP), allocatable :: dust_emission_final(:,:)
        real(WP), allocatable :: gas_neb_cont_reduced(:,:)   ! log10 values: kept in WP (an FP32 log loses ln(10)*|x|*eps ~ 5e-6 in 10**x)
        real(WP), allocatable :: gas_neb_line_reduced(:,:)
        real(WP), allocatable :: gas_current_step_cont(:)   ! log10 values: kept in WP (an FP32 log loses ln(10)*|x|*eps ~ 5e-6 in 10**x)
        real(WP), allocatable :: gas_current_step_lines(:)
        real(WP), allocatable :: scalar_reductions(:)
        real(WP), allocatable :: sfh_t_calc(:)
        real(WP), allocatable :: sfh_sfr_calc(:)
        real(WP), allocatable :: sfh_age_integrand(:)
        real(WP), allocatable :: sfh_w_tmp1(:,:)
        real(WP), allocatable :: sfh_w_tmp2(:,:)

        ! --- Fused Loop Output Buffers ---
        real(KP), allocatable :: out_csp_spec(:,:)     ! (nspec, n_outputs)
        real(WP), allocatable :: out_csp_emlin(:,:)    ! (nemline, n_outputs)
        real(WP), allocatable :: out_mass_csp(:)       ! (n_outputs)
        real(WP), allocatable :: out_lbol_csp(:)       ! (n_outputs)
        type(batch_grid_t) :: batch
        type(batch_out_t) :: batch_out
    end type fsps_context_state_t

    type :: fsps_context_t
        logical :: initialized = .false.
        !> True between fsps_context_move_to_device and fsps_context_remove_from_device: setup and
        !> destroy release the device copies first, so freed host arrays leave no stale mappings
        logical :: on_device = .false.
        !> True when a scalar setting (*_val, fast_mode) changed on the host after the context was
        !> moved to the device: device kernels read the device copy of ctx, so the compute entry
        !> points refresh it first (fsps_context_sync_device_settings)
        logical :: settings_dirty = .false.
        logical :: fast_mode = .false.
        integer :: zin = 0
        character(len=64) :: isoc_type_name = ''
        character(len=64) :: spec_type_name = ''
        character(len=64) :: dust_type_name = ''
        character(len=250) :: sps_home = ''
        character(len=250) :: data_home = ''
        character(len=250) :: output_home = ''
        !> Data format of the libraries (FSPS_DATA_FORMAT): 'auto', 'legacy' or 'standard'
        character(len=8) :: data_format = 'auto'
        type(fsps_setup_cache_t), pointer :: setup_cache => null()
        type(fsps_context_state_t) :: state
        real(WP) :: om0_val = 0.0_wp
        real(WP) :: ol0_val = 0.0_wp
        real(WP) :: H0_val = 0.0_wp
        real(WP) :: tiny_logt_val = 0.0_wp
        ! Same defaults as fsps_context_create: these now drive the IMF limits used by the physics
        real(WP) :: imf_upper_limit_val = 120.0_wp
        real(WP) :: imf_lower_limit_val = 0.08_wp
        real(WP) :: logt_wmb_hot_val = 0.0_wp
        real(WP) :: nebular_smooth_init_val = 0.0_wp
        integer :: imf_type_val = 0
        integer :: tpagb_norm_type_val = 0
        integer :: pzcon_val = 0
        integer :: interpolation_type_val = 0
        integer :: add_agb_dust_model_val = 0
        integer :: dust_type_val = 0
        integer :: add_dust_emission_val = 0
        integer :: compute_vega_mags_val = 0
        integer :: vactoair_flag_val = 0
        integer :: add_agn_dust_val = 0
        integer :: use_wr_spectra_val = 0
        integer :: add_neb_emission_val = 0
        integer :: add_neb_continuum_val = 0
        integer :: cloudy_dust_val = 0
        integer :: add_igm_absorption_val = 0
        integer :: nebemlineinspec_val = 0
        integer :: add_xrb_emission_val = 0
        integer :: add_stellar_remnants_val = 0
        integer :: smooth_velocity_val = 0
        integer :: smooth_lsf_val = 0
        integer :: smoothspec_fast_val = 0
        integer :: redshift_colors_val = 0
        integer :: compute_light_ages_val = 0
        integer :: use_isoc_mdot_val = 0
        integer :: setup_nebular_gaussians_val = 0
        type(PARAMS) :: pset
    end type fsps_context_t

contains

    !> @brief Copies the scalar settings of a device-resident context to its device copy, if any
    !> changed on the host since the last copy (settings_dirty). Called by the compute entry points.
    subroutine fsps_context_sync_device_settings(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (.not. (ctx%on_device .and. ctx%settings_dirty)) return
        !$acc update device(ctx%fast_mode, ctx%om0_val, ctx%ol0_val, ctx%H0_val, ctx%tiny_logt_val, &
        !$acc   ctx%imf_upper_limit_val, ctx%imf_lower_limit_val, ctx%logt_wmb_hot_val, &
        !$acc   ctx%nebular_smooth_init_val, ctx%imf_type_val, ctx%tpagb_norm_type_val, ctx%pzcon_val, &
        !$acc   ctx%interpolation_type_val, ctx%add_agb_dust_model_val, ctx%dust_type_val, &
        !$acc   ctx%add_dust_emission_val, ctx%compute_vega_mags_val, ctx%vactoair_flag_val, &
        !$acc   ctx%add_agn_dust_val, ctx%use_wr_spectra_val, ctx%add_neb_emission_val, &
        !$acc   ctx%add_neb_continuum_val, ctx%cloudy_dust_val, ctx%add_igm_absorption_val, &
        !$acc   ctx%nebemlineinspec_val, ctx%add_xrb_emission_val, ctx%add_stellar_remnants_val, &
        !$acc   ctx%smooth_velocity_val, ctx%smooth_lsf_val, ctx%smoothspec_fast_val, &
        !$acc   ctx%redshift_colors_val, ctx%compute_light_ages_val, ctx%use_isoc_mdot_val, &
        !$acc   ctx%setup_nebular_gaussians_val)
#ifdef OMP_TARGET
        !$omp target update to(ctx%fast_mode, ctx%om0_val, ctx%ol0_val, ctx%H0_val, ctx%tiny_logt_val, &
        !$omp   ctx%imf_upper_limit_val, ctx%imf_lower_limit_val, ctx%logt_wmb_hot_val, &
        !$omp   ctx%nebular_smooth_init_val, ctx%imf_type_val, ctx%tpagb_norm_type_val, ctx%pzcon_val, &
        !$omp   ctx%interpolation_type_val, ctx%add_agb_dust_model_val, ctx%dust_type_val, &
        !$omp   ctx%add_dust_emission_val, ctx%compute_vega_mags_val, ctx%vactoair_flag_val, &
        !$omp   ctx%add_agn_dust_val, ctx%use_wr_spectra_val, ctx%add_neb_emission_val, &
        !$omp   ctx%add_neb_continuum_val, ctx%cloudy_dust_val, ctx%add_igm_absorption_val, &
        !$omp   ctx%nebemlineinspec_val, ctx%add_xrb_emission_val, ctx%add_stellar_remnants_val, &
        !$omp   ctx%smooth_velocity_val, ctx%smooth_lsf_val, ctx%smoothspec_fast_val, &
        !$omp   ctx%redshift_colors_val, ctx%compute_light_ages_val, ctx%use_isoc_mdot_val, &
        !$omp   ctx%setup_nebular_gaussians_val)
#endif
        ctx%settings_dirty = .false.
    end subroutine fsps_context_sync_device_settings

    !> @brief Drops the batched-CSP grid cache (and its device copies, if mapped).
    subroutine batch_grid_release(g)
        type(batch_grid_t), intent(inout) :: g

        if (g%ready) then
            !$acc exit data delete(g%i_lo, g%j_lo, g%j_hi, g%t_obs, g%wt, g%wtl, g%wl, g%bands_t, g%igm_t)
#ifdef OMP_TARGET
            !$omp target exit data map(delete: g%i_lo, g%j_lo, g%j_hi, g%t_obs, g%wt, g%wtl, g%wl, g%bands_t, g%igm_t)
#endif
            if (allocated(g%cmat)) then
                !$acc exit data delete(g%cmat)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: g%cmat)
#endif
            end if
        end if
        if (allocated(g%i_lo)) deallocate(g%i_lo, g%j_lo, g%j_hi, g%t_obs, g%wt, g%wtl, g%wl, g%bands_t, g%igm_t)
        if (allocated(g%cmat)) deallocate(g%cmat)
        g%ready = .false.
        if (g%sm_ready) then
            !$acc exit data delete(g%sm_lo, g%sm_hi, g%sm_off, g%sm_coef)
#ifdef OMP_TARGET
            !$omp target exit data map(delete: g%sm_lo, g%sm_hi, g%sm_off, g%sm_coef)
#endif
        end if
        if (allocated(g%sm_lo)) deallocate(g%sm_lo, g%sm_hi, g%sm_off, g%sm_coef)
        g%sm_ready = .false.
    end subroutine batch_grid_release

    !> Room for n_mod spectra of nspec pixels in the batch output (o%spec, on the device when
    !> on_device); grows by at least half to avoid repeated reallocation and pinning.
    subroutine batch_out_reserve(o, nspec, n_mod, on_device)
        type(batch_out_t), intent(inout), target :: o
        integer, intent(in) :: nspec, n_mod
        logical, intent(in) :: on_device
        integer :: cap

        o%n = 0
        if (allocated(o%spec)) then
            if (size(o%spec, 1) == nspec .and. size(o%spec, 2) >= n_mod) return
            cap = max(n_mod, size(o%spec, 2) + size(o%spec, 2) / 2)
            if (size(o%spec, 1) /= nspec) cap = n_mod
        else
            cap = n_mod
        end if
        call batch_out_release(o, on_device)
        allocate(o%spec(nspec, cap))
#ifdef FSPS_PINNED
        ! Page-locked host copy: the copy-back runs at the PCIe rate instead of through the
        ! driver's staging buffers (~2.4 GB/s on the A100 node)
        o%pinned = fsps_host_register(c_loc(o%spec), int(storage_size(o%spec) / 8, c_size_t) * &
                                      int(size(o%spec), c_size_t)) == 0
#endif
        if (on_device) then
            !$acc enter data create(o%spec)
#ifdef OMP_TARGET
            !$omp target enter data map(alloc: o%spec)
#endif
        end if
    end subroutine batch_out_reserve

    !> Release the batch output (its device copy too if on_device).
    subroutine batch_out_release(o, on_device)
        type(batch_out_t), intent(inout), target :: o
        logical, intent(in) :: on_device

        if (.not. allocated(o%spec)) return
        if (on_device) then
            !$acc exit data delete(o%spec)
#ifdef OMP_TARGET
            !$omp target exit data map(delete: o%spec)
#endif
        end if
#ifdef FSPS_PINNED
        if (o%pinned) then
            if (fsps_host_unregister(c_loc(o%spec)) /= 0) continue
        end if
#endif
        o%pinned = .false.
        o%n = 0
        deallocate(o%spec)
    end subroutine batch_out_release

    !> @brief Release pointer members in a context state object.
    !> @param[inout] state State to detach from shared cache arrays.
    subroutine fsps_context_state_destroy(state)
        type(fsps_context_state_t), intent(inout) :: state

        call batch_grid_release(state%batch)
        call batch_out_release(state%batch_out, .false.)

        if (associated(state%indexdefined)) nullify (state%indexdefined)
        if (associated(state%wgdust)) nullify (state%wgdust)
        if (associated(state%g03smcextn)) nullify (state%g03smcextn)
        if (associated(state%bands)) nullify (state%bands)
        if (associated(state%magsun)) nullify (state%magsun)
        if (associated(state%magvega)) nullify (state%magvega)
        if (associated(state%filter_leff)) nullify (state%filter_leff)
        if (associated(state%vega_spec)) nullify (state%vega_spec)
        if (associated(state%sun_spec)) nullify (state%sun_spec)
        if (associated(state%spec_lambda)) nullify (state%spec_lambda)
        if (associated(state%spec_nu)) nullify (state%spec_nu)
        if (associated(state%spec_res)) nullify (state%spec_res)
        if (associated(state%speclib)) nullify (state%speclib)
        if (associated(state%wmb_spec)) nullify (state%wmb_spec)
        if (associated(state%agb_spec_o)) nullify (state%agb_spec_o)
        if (associated(state%agb_logt_o)) nullify (state%agb_logt_o)
        if (associated(state%agb_spec_c)) nullify (state%agb_spec_c)
        if (associated(state%agb_logt_c)) nullify (state%agb_logt_c)
        if (associated(state%agb_spec_car)) nullify (state%agb_spec_car)
        if (associated(state%speclib_agb_smooth)) nullify (state%speclib_agb_smooth)
        if (associated(state%agb_spec_o_smooth)) nullify (state%agb_spec_o_smooth)
        if (associated(state%agb_spec_c_smooth)) nullify (state%agb_spec_c_smooth)
        if (associated(state%agb_spec_car_smooth)) nullify (state%agb_spec_car_smooth)
        if (associated(state%pagb_spec)) nullify (state%pagb_spec)
        if (associated(state%wrn_spec)) nullify (state%wrn_spec)
        if (associated(state%wrc_spec)) nullify (state%wrc_spec)
        if (associated(state%qpaharr)) nullify (state%qpaharr)
        if (associated(state%uminarr)) nullify (state%uminarr)
        if (associated(state%lambda_dustem)) nullify (state%lambda_dustem)
        if (associated(state%dustem_dustem)) nullify (state%dustem_dustem)
        if (associated(state%dustem2_dustem)) nullify (state%dustem2_dustem)
        if (associated(state%flux_dagb)) nullify (state%flux_dagb)
        if (associated(state%nebem_cont)) nullify (state%nebem_cont)
        if (associated(state%xnebem_cont)) nullify (state%xnebem_cont)
        if (associated(state%neb_res_min)) nullify (state%neb_res_min)
        if (state%owns_gaussnebarr) then
            if (associated(state%gaussnebarr)) deallocate (state%gaussnebarr)
            state%owns_gaussnebarr = .false.
        else if (associated(state%gaussnebarr)) then
            nullify (state%gaussnebarr)
        end if
        if (associated(state%agndust_spec)) nullify (state%agndust_spec)
        if (associated(state%mact_isoc)) nullify (state%mact_isoc)
        if (associated(state%logl_isoc)) nullify (state%logl_isoc)
        if (associated(state%logt_isoc)) nullify (state%logt_isoc)
        if (associated(state%logg_isoc)) nullify (state%logg_isoc)
        if (associated(state%ffco_isoc)) nullify (state%ffco_isoc)
        if (associated(state%phase_isoc)) nullify (state%phase_isoc)
        if (associated(state%mini_isoc)) nullify (state%mini_isoc)
        if (associated(state%lmdot_isoc)) nullify (state%lmdot_isoc)
        if (associated(state%nmass_isoc)) nullify (state%nmass_isoc)
        if (associated(state%timestep_isoc)) nullify (state%timestep_isoc)
        if (associated(state%zlegend)) nullify (state%zlegend)
        if (associated(state%zlegendinit)) nullify (state%zlegendinit)
        if (allocated(state%spec_ssp_zz)) deallocate (state%spec_ssp_zz)
        if (allocated(state%mass_ssp_zz)) deallocate (state%mass_ssp_zz)
        if (allocated(state%lbol_ssp_zz)) deallocate (state%lbol_ssp_zz)
        if (associated(state%time_full)) nullify (state%time_full)
        if (allocated(state%weight_ssp)) deallocate (state%weight_ssp)
        if (allocated(state%spec_young)) deallocate (state%spec_young)
        if (allocated(state%spec_old)) deallocate (state%spec_old)
        if (allocated(state%ssp_temp_grid)) deallocate (state%ssp_temp_grid)
        if (allocated(state%ssp_active_idx)) deallocate (state%ssp_active_idx)
        if (allocated(state%ssp_active_w)) deallocate (state%ssp_active_w)
        if (associated(state%bpass_spec_ssp)) nullify (state%bpass_spec_ssp)
        if (associated(state%bpass_mass_ssp)) nullify (state%bpass_mass_ssp)
        if (associated(state%lam_xrb)) nullify (state%lam_xrb)
        if (associated(state%spec_xrb)) nullify (state%spec_xrb)
        if (associated(state%ages_xrb)) nullify (state%ages_xrb)
        if (associated(state%zmet_xrb)) nullify (state%zmet_xrb)
        if (allocated(state%lsfinfo%lsf)) deallocate (state%lsfinfo%lsf)
        if (allocated(state%powell_data%mags)) deallocate (state%powell_data%mags)
        if (allocated(state%powell_data%magerr)) deallocate (state%powell_data%magerr)
        if (allocated(state%powell_data%spec)) deallocate (state%powell_data%spec)
        if (allocated(state%powell_data%specerr)) deallocate (state%powell_data%specerr)
        if (allocated(state%sedfit_data%mags)) deallocate (state%sedfit_data%mags)
        if (allocated(state%sedfit_data%magerr)) deallocate (state%sedfit_data%magerr)
        if (allocated(state%sedfit_data%spec)) deallocate (state%sedfit_data%spec)
        if (allocated(state%sedfit_data%specerr)) deallocate (state%sedfit_data%specerr)

        ! --- Persistent CSP Workspace ---
        if (allocated(state%ssp_basis_spec)) deallocate (state%ssp_basis_spec)
        if (allocated(state%ssp_basis_mass)) deallocate (state%ssp_basis_mass)
        if (allocated(state%ssp_basis_lbol)) deallocate (state%ssp_basis_lbol)
        if (allocated(state%csp_ssp_grid)) deallocate (state%csp_ssp_grid)
        if (allocated(state%csp_emlin_grid)) deallocate (state%csp_emlin_grid)
        if (allocated(state%csp_ssp_lum_linear)) deallocate (state%csp_ssp_lum_linear)
        if (allocated(state%csp_igm_transmission)) deallocate (state%csp_igm_transmission)
        if (allocated(state%csp_spec_final)) deallocate (state%csp_spec_final)
        if (allocated(state%csp_emlin_final)) deallocate (state%csp_emlin_final)

        ! --- Replacements for csp_buffer_t ---
        if (allocated(state%csp_weights)) deallocate (state%csp_weights)
        if (allocated(state%csp_emlin_young)) deallocate (state%csp_emlin_young)
        if (allocated(state%csp_emlin_old)) deallocate (state%csp_emlin_old)

        ! --- Persistent Physics Workspaces (Dust & Gas) ---
        if (allocated(state%dust_transmission_diffuse)) deallocate(state%dust_transmission_diffuse)
        if (allocated(state%dust_frequencies)) deallocate(state%dust_frequencies)
        if (allocated(state%dust_spec_total_work)) deallocate(state%dust_spec_total_work)
        if (allocated(state%dust_emission_shape)) deallocate(state%dust_emission_shape)
        if (allocated(state%dust_emission_final)) deallocate(state%dust_emission_final)
        if (allocated(state%gas_neb_cont_reduced)) deallocate(state%gas_neb_cont_reduced)
        if (allocated(state%gas_neb_line_reduced)) deallocate(state%gas_neb_line_reduced)
        if (allocated(state%gas_current_step_cont)) deallocate(state%gas_current_step_cont)
        if (allocated(state%gas_current_step_lines)) deallocate(state%gas_current_step_lines)
        if (allocated(state%scalar_reductions)) deallocate(state%scalar_reductions)
        if (allocated(state%sfh_t_calc)) deallocate(state%sfh_t_calc)
        if (allocated(state%sfh_sfr_calc)) deallocate(state%sfh_sfr_calc)
        if (allocated(state%sfh_age_integrand)) deallocate(state%sfh_age_integrand)
        if (allocated(state%sfh_w_tmp1)) deallocate(state%sfh_w_tmp1)
        if (allocated(state%sfh_w_tmp2)) deallocate(state%sfh_w_tmp2)

        ! --- Fused Loop Output Buffers ---
        if (allocated(state%out_csp_spec)) deallocate(state%out_csp_spec)
        if (allocated(state%out_csp_emlin)) deallocate(state%out_csp_emlin)
        if (allocated(state%out_mass_csp)) deallocate(state%out_mass_csp)
        if (allocated(state%out_lbol_csp)) deallocate(state%out_lbol_csp)

        state%nt = 0
        state%nz = 0
        state%nspec = 0
        state%nzinit = 0
        state%nbands = 0
        state%nindx = 0
        state%ntfull = 0
        state%nspec_xrb = 0
        state%nt_xrb = 0
        state%nz_xrb = 0
        state%ssp_temp_nspec = 0
        state%ssp_temp_nstars = 0
        state%check_sps_setup = 0
        state%tuniv = 0.0_wp
        state%whlam5000 = 0
        state%whlylim = 0
        state%zsol = 0.0_wp
        state%zsol_spec = 0.0_wp
        state%zpow2 = 1.0_wp
        state%ntabsfh = 0
        state%lsfinfo%minlam = 0.0_wp
        state%lsfinfo%maxlam = 0.0_wp
        state%str_dustem = 'DL07'
        state%imf_alpha = 1.3_wp
        state%imf_vdmc = 0.08_wp
        state%imf_mdave = 0.5_wp
        state%n_user_imf = 0
        state%imf_user_alpha = 0.0_wp
        state%salp_ind = 2.35_wp
        state%imf_lower_limit = 0.08_wp
        state%imf_upper_limit = 120.0_wp
        state%imf_lower_bound = 0.0_wp
        state%mlim_bh = 40.0_wp
        state%mlim_ns = 8.5_wp
        state%alt_filter_file = ''
        state%powell_data%zred = 0.0_wp
        state%powell_data%logsmass = 0.0_wp
        state%sedfit_data%zred = 0.0_wp
        state%sedfit_data%logsmass = 0.0_wp
        state%ssp_basis_is_dirty = .true.
        
    end subroutine fsps_context_state_destroy

end module fsps_context_types
