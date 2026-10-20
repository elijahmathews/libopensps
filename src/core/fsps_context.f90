module fsps_context
    !> @brief
    !> Context lifecycle and parameter management for FSPS.
    !>
    !> @details
    !> Provides creation, setup, teardown, and parameter mutation for
    !> `fsps_context_t`, along with high-level SSP/CSP execution helpers.

    use fsps_precision, only: WP
    use fsps_constants, only: NEMLINE, NEBNAGE, NTABMAX
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_context_types, only: fsps_context_t, fsps_context_state_destroy, batch_grid_release, &
                                  batch_out_release
    use fsps_environment, only: fsps_cleanup
    use fsps_initialization, only: fsps_initialize_data
    use fsps_ssp, only: generate_ssp_grid
    use fsps_csp, only: compute_csp_scenario
    use fsps_io, only: load_tabular_sfh, write_csp_output_files

    implicit none
    private

    public :: fsps_context_create
    public :: fsps_context_setup
    public :: fsps_context_ensure_setup
    public :: fsps_context_destroy
    public :: fsps_context_set_pset
    public :: fsps_context_get_pset
    public :: fsps_context_set_param_int
    public :: fsps_context_set_param_float
    public :: fsps_context_set_param_str
    public :: fsps_context_set_fast_mode
    public :: fsps_context_get_paths
    public :: fsps_context_prepare_pset
    public :: is_ssp_sensitive_float_param
    public :: is_ssp_sensitive_int_param
    public :: is_ssp_sensitive_str_param
    public :: fsps_context_compute_ssp
    public :: fsps_context_compute_csp
    public :: fsps_context_update_ssp_basis
    public :: fsps_context_prepare_csp_workspace
    public :: fsps_context_move_to_device
    public :: fsps_context_remove_from_device

    ! ---------------------------------------------------------------------
    ! Module constants
    ! ---------------------------------------------------------------------

    integer, parameter :: FSPS_ERR_UNKNOWN_INT_PARAM = 101
    integer, parameter :: FSPS_ERR_UNKNOWN_FLOAT_PARAM = 102
    integer, parameter :: FSPS_ERR_UNKNOWN_STRING_PARAM = 103

contains

    !> @brief Create a new FSPS context with default values.
    !> @param[out] ctx Newly initialized context.
    subroutine fsps_context_create(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        ctx%initialized = .false.
        ctx%fast_mode = .false.
        ctx%zin = 0
        ctx%isoc_type_name = ''
        ctx%spec_type_name = ''
        ctx%dust_type_name = ''
        ctx%sps_home = ''
        ctx%data_home = ''
        ctx%output_home = ''
        ctx%om0_val = 0.27_wp
        ctx%ol0_val = 0.73_wp
        ctx%H0_val = 72.0_wp
        ctx%tpagb_norm_type_val = 2
        ctx%pzcon_val = 0
        ctx%interpolation_type_val = 0
        ctx%tiny_logt_val = 0.0_wp
        ctx%compute_light_ages_val = 0
        ctx%add_dust_emission_val = 1
        ctx%add_agn_dust_val = 1
        ctx%add_agb_dust_model_val = 1
        ctx%use_wr_spectra_val = 1
        ctx%logt_wmb_hot_val = 0.0_wp
        ctx%add_neb_emission_val = 0
        ctx%add_neb_continuum_val = 1
        ctx%cloudy_dust_val = 0
        ctx%add_igm_absorption_val = 0
        ctx%add_xrb_emission_val = 0
        ctx%add_stellar_remnants_val = 1
        ctx%smoothspec_fast_val = 1
        ctx%smooth_velocity_val = 1
        ctx%smooth_lsf_val = 0
        ctx%dust_type_val = 0
        ctx%imf_type_val = 2
        ctx%compute_vega_mags_val = 0
        ctx%vactoair_flag_val = 0
        ctx%redshift_colors_val = 0
        ctx%use_isoc_mdot_val = 0
        ctx%setup_nebular_gaussians_val = 0
        ctx%nebular_smooth_init_val = 100.0_wp
        ctx%nebemlineinspec_val = 1
        ctx%imf_lower_limit_val = 0.08_wp
        ctx%imf_upper_limit_val = 120.0_wp
    end subroutine fsps_context_create

    !> @brief Initialize a context by loading data and binding cache resources.
    !> @param[inout] ctx Context to initialize.
    !> @param[in] zin Metallicity index (-1 for all).
    !> @param[in] isoc_type_in Optional isochrone library name.
    !> @param[in] spec_type_in Optional spectral library name.
    !> @param[in] dust_type_in Optional dust model name.
    subroutine fsps_context_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin
        character(len=*), intent(in), optional :: isoc_type_in
        character(len=*), intent(in), optional :: spec_type_in
        character(len=*), intent(in), optional :: dust_type_in

        ! Setup repoints and reallocates the state arrays: drop the old device copies first
        if (ctx%on_device) call fsps_context_remove_from_device(ctx)
        ctx%zin = zin
        if (present(isoc_type_in)) ctx%isoc_type_name = trim(isoc_type_in)
        if (present(spec_type_in)) ctx%spec_type_name = trim(spec_type_in)
        if (present(dust_type_in)) ctx%dust_type_name = trim(dust_type_in)

        ! fsps_initialize_data serializes itself with the process-wide setup lock, which (unlike an
        ! OpenMP critical section) also covers non-OpenMP builds and fsps_context_ensure_setup
        call fsps_initialize_data(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)

        ctx%initialized = .true.
        if (len_trim(ctx%isoc_type_name) == 0) then
            ctx%isoc_type_name = trim(ctx%state%isoc_type)
        end if
        if (len_trim(ctx%spec_type_name) == 0) then
            ctx%spec_type_name = trim(ctx%state%spec_type)
        end if
        if (len_trim(ctx%dust_type_name) == 0) then
            ctx%dust_type_name = trim(ctx%state%str_dustem)
        end if
        call fsps_context_prepare_pset(ctx)
        ctx%state%ssp_basis_is_dirty = .true.
    end subroutine fsps_context_setup

    !> @brief Ensure the context is initialized with the current library selections.
    !> @param[inout] ctx Context to check and initialize if needed.
    subroutine fsps_context_ensure_setup(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        logical :: need_setup
        character(len=64) :: isoc_name
        character(len=64) :: spec_name
        character(len=64) :: dust_name

        need_setup = .false.
        isoc_name = trim(ctx%isoc_type_name)
        spec_name = trim(ctx%spec_type_name)
        dust_name = trim(ctx%dust_type_name)

        if (len_trim(isoc_name) /= 0 .and. trim(ctx%state%isoc_type) /= isoc_name) then
            need_setup = .true.
        end if

        if (len_trim(spec_name) /= 0 .and. trim(ctx%state%spec_type) /= spec_name) then
            need_setup = .true.
        end if

        if (.not. need_setup) then
            if (len_trim(isoc_name) == 0) then
                ctx%isoc_type_name = trim(ctx%state%isoc_type)
            end if
            if (len_trim(spec_name) == 0) then
                ctx%spec_type_name = trim(ctx%state%spec_type)
            end if
            if (len_trim(dust_name) == 0) then
                ctx%dust_type_name = trim(ctx%state%str_dustem)
            end if
            return
        end if

        if (len_trim(isoc_name) == 0 .and. len_trim(spec_name) == 0 .and. len_trim(dust_name) == 0) then
            call fsps_initialize_data(ctx, ctx%zin)
        else if (len_trim(spec_name) == 0 .and. len_trim(dust_name) == 0) then
            call fsps_initialize_data(ctx, ctx%zin, isoc_type_in=isoc_name)
        else if (len_trim(dust_name) == 0) then
            call fsps_initialize_data(ctx, ctx%zin, isoc_type_in=isoc_name, spec_type_in=spec_name)
        else if (len_trim(spec_name) == 0) then
            call fsps_initialize_data(ctx, ctx%zin, isoc_type_in=isoc_name, dust_type_in=dust_name)
        else
            call fsps_initialize_data(ctx, ctx%zin, isoc_type_in=isoc_name, spec_type_in=spec_name, &
                                      dust_type_in=dust_name)
        end if

        ctx%isoc_type_name = trim(ctx%state%isoc_type)
        ctx%spec_type_name = trim(ctx%state%spec_type)
        ctx%dust_type_name = trim(ctx%state%str_dustem)
    end subroutine fsps_context_ensure_setup

    !> @brief Release all resources owned by a context.
    !> @param[inout] ctx Context to destroy.
    subroutine fsps_context_destroy(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (ctx%on_device) call fsps_context_remove_from_device(ctx)
        if (ctx%initialized) then
            call fsps_cleanup(ctx)
            ctx%initialized = .false.
        end if
        call fsps_context_state_destroy(ctx%state)
    end subroutine fsps_context_destroy

    !> @brief Replace the context parameter set.
    !> @param[inout] ctx Context to update.
    !> @param[in] pset_in New parameter set.
    subroutine fsps_context_set_pset(ctx, pset_in)
        type(fsps_context_t), intent(inout) :: ctx
        type(PARAMS), intent(in) :: pset_in

        if (pset_changes_ssp_basis(ctx%pset, pset_in)) then
            ctx%state%ssp_basis_is_dirty = .true.
        end if
        ctx%pset = pset_in
    end subroutine fsps_context_set_pset

    !> @brief Retrieve the current context parameter set.
    !> @param[in] ctx Context to read.
    !> @param[out] pset_out Current parameter set.
    subroutine fsps_context_get_pset(ctx, pset_out)
        type(fsps_context_t), intent(in) :: ctx
        type(PARAMS), intent(out) :: pset_out

        pset_out = ctx%pset
    end subroutine fsps_context_get_pset

    !> @brief Set an integer parameter by key.
    !> @param[inout] ctx Context to update.
    !> @param[in] key Parameter name.
    !> @param[in] value Parameter value.
    !> @param[out] status Non-zero on unknown parameter.
    subroutine fsps_context_set_param_int(ctx, key, value, status)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer, intent(out) :: status
        character(len=64) :: key_norm

        status = 0
        key_norm = trim(key)
        select case (key_norm)
        case ('sfh')
            ctx%pset%sfh = value
        case ('zmet')
            ctx%pset%zmet = value
        case ('wgp1')
            ctx%pset%wgp1 = value
        case ('wgp2')
            ctx%pset%wgp2 = value
        case ('wgp3')
            ctx%pset%wgp3 = value
        case ('evtype')
            ctx%pset%evtype = value
        case ('imf_type')
            ctx%imf_type_val = value
            ctx%settings_dirty = .true.
        case ('tpagb_norm_type')
            ctx%tpagb_norm_type_val = value
            ctx%settings_dirty = .true.
        case ('pzcon')
            ctx%pzcon_val = value
            ctx%settings_dirty = .true.
        case ('interpolation_type')
            ctx%interpolation_type_val = value
            ctx%settings_dirty = .true.
        case ('add_agb_dust_model')
            ctx%add_agb_dust_model_val = value
            ctx%settings_dirty = .true.
        case ('dust_type')
            ctx%dust_type_val = value
            ctx%settings_dirty = .true.
        case ('add_dust_emission')
            ctx%add_dust_emission_val = value
            ctx%settings_dirty = .true.
        case ('compute_vega_mags')
            ctx%compute_vega_mags_val = value
            ctx%settings_dirty = .true.
        case ('vactoair_flag')
            ctx%vactoair_flag_val = value
            ctx%settings_dirty = .true.
        case ('add_agn_dust')
            ctx%add_agn_dust_val = value
            ctx%settings_dirty = .true.
        case ('use_wr_spectra')
            ctx%use_wr_spectra_val = value
            ctx%settings_dirty = .true.
        case ('add_neb_emission')
            ctx%add_neb_emission_val = value
            ctx%settings_dirty = .true.
        case ('add_neb_continuum')
            ctx%add_neb_continuum_val = value
            ctx%settings_dirty = .true.
        case ('cloudy_dust')
            ctx%cloudy_dust_val = value
            ctx%settings_dirty = .true.
        case ('add_igm_absorption')
            ctx%add_igm_absorption_val = value
            ctx%settings_dirty = .true.
        case ('nebemlineinspec')
            ctx%nebemlineinspec_val = value
            ctx%settings_dirty = .true.
        case ('add_xrb_emission')
            ctx%add_xrb_emission_val = value
            ctx%settings_dirty = .true.
        case ('add_stellar_remnants')
            ctx%add_stellar_remnants_val = value
            ctx%settings_dirty = .true.
        case ('smooth_velocity')
            ctx%smooth_velocity_val = value
            ctx%settings_dirty = .true.
        case ('smooth_lsf')
            ctx%smooth_lsf_val = value
            ctx%settings_dirty = .true.
        case ('smoothspec_fast')
            ctx%smoothspec_fast_val = value
            ctx%settings_dirty = .true.
        case ('redshift_colors')
            ctx%redshift_colors_val = value
            ctx%settings_dirty = .true.
        case ('compute_light_ages')
            ctx%compute_light_ages_val = value
            ctx%settings_dirty = .true.
        case ('use_isoc_mdot')
            ctx%use_isoc_mdot_val = value
            ctx%settings_dirty = .true.
        case ('setup_nebular_gaussians')
            ctx%setup_nebular_gaussians_val = value
            ctx%settings_dirty = .true.
        case ('compute_mags')
            ctx%pset%compute_mags = value
        case ('compute_indices')
            ctx%pset%compute_indices = value
        case ('fast_mode')
            call fsps_context_set_fast_mode(ctx, value /= 0)
        case default
            status = FSPS_ERR_UNKNOWN_INT_PARAM
        end select

        if (status == 0 .and. is_ssp_sensitive_int_param(key_norm)) then
            ctx%state%ssp_basis_is_dirty = .true.
        end if
    end subroutine fsps_context_set_param_int

    !> @brief Enable or disable CSP fast mode.
    !> @param[inout] ctx Context to update.
    !> @param[in] fast_mode_in Fast mode flag.
    subroutine fsps_context_set_fast_mode(ctx, fast_mode_in)
        type(fsps_context_t), intent(inout) :: ctx
        logical, intent(in) :: fast_mode_in

        ctx%fast_mode = fast_mode_in
        ctx%settings_dirty = .true.   ! device copy refreshed by the next compute call
    end subroutine fsps_context_set_fast_mode

    !> @brief Set a floating-point parameter by key.
    !> @param[inout] ctx Context to update.
    !> @param[in] key Parameter name.
    !> @param[in] value Parameter value.
    !> @param[out] status Non-zero on unknown parameter.
    subroutine fsps_context_set_param_float(ctx, key, value, status)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer, intent(out) :: status
        character(len=64) :: key_norm

        status = 0
        key_norm = trim(key)
        select case (key_norm)
        case ('om0')
            ctx%om0_val = value
            ctx%settings_dirty = .true.
        case ('ol0')
            ctx%ol0_val = value
            ctx%settings_dirty = .true.
        case ('H0')
            ctx%H0_val = value
            ctx%settings_dirty = .true.
        case ('tiny_logt')
            ctx%tiny_logt_val = value
            ctx%settings_dirty = .true.
        case ('imf_upper_limit')
            ctx%imf_upper_limit_val = value
            ctx%settings_dirty = .true.
        case ('imf_lower_limit')
            ctx%imf_lower_limit_val = value
            ctx%settings_dirty = .true.
        case ('logt_wmb_hot')
            ctx%logt_wmb_hot_val = value
            ctx%settings_dirty = .true.
        case ('nebular_smooth_init')
            ctx%nebular_smooth_init_val = value
            ctx%settings_dirty = .true.
        case ('imf1')
            ctx%pset%imf1 = value
        case ('imf2')
            ctx%pset%imf2 = value
        case ('imf3')
            ctx%pset%imf3 = value
        case ('vdmc')
            ctx%pset%vdmc = value
        case ('mdave')
            ctx%pset%mdave = value
        case ('dell')
            ctx%pset%dell = value
        case ('delt')
            ctx%pset%delt = value
        case ('sbss')
            ctx%pset%sbss = value
        case ('fbhb')
            ctx%pset%fbhb = value
        case ('pagb')
            ctx%pset%pagb = value
        case ('agb_dust')
            ctx%pset%agb_dust = value
        case ('redgb')
            ctx%pset%redgb = value
        case ('agb')
            ctx%pset%agb = value
        case ('masscut')
            ctx%pset%masscut = value
        case ('fcstar')
            ctx%pset%fcstar = value
        case ('frac_xrb')
            ctx%pset%frac_xrb = value
        case ('logzsol')
            ctx%pset%logzsol = value
        case ('tau')
            ctx%pset%tau = value
        case ('const')
            ctx%pset%const = value
        case ('tage')
            ctx%pset%tage = value
        case ('fburst')
            ctx%pset%fburst = value
        case ('tburst')
            ctx%pset%tburst = value
        case ('dust1')
            ctx%pset%dust1 = value
        case ('dust2')
            ctx%pset%dust2 = value
        case ('dust3')
            ctx%pset%dust3 = value
        case ('zred')
            ctx%pset%zred = value
        case ('pmetals')
            ctx%pset%pmetals = value
        case ('dust_clumps')
            ctx%pset%dust_clumps = value
        case ('frac_nodust')
            ctx%pset%frac_nodust = value
        case ('dust_index')
            ctx%pset%dust_index = value
        case ('dust_tesc')
            ctx%pset%dust_tesc = value
        case ('frac_obrun')
            ctx%pset%frac_obrun = value
        case ('uvb')
            ctx%pset%uvb = value
        case ('mwr')
            ctx%pset%mwr = value
        case ('dust1_index')
            ctx%pset%dust1_index = value
        case ('sf_start')
            ctx%pset%sf_start = value
        case ('sf_trunc')
            ctx%pset%sf_trunc = value
        case ('sf_slope')
            ctx%pset%sf_slope = value
        case ('duste_gamma')
            ctx%pset%duste_gamma = value
        case ('duste_umin')
            ctx%pset%duste_umin = value
        case ('duste_qpah')
            ctx%pset%duste_qpah = value
        case ('sigma_smooth')
            ctx%pset%sigma_smooth = value
        case ('min_wave_smooth')
            ctx%pset%min_wave_smooth = value
        case ('max_wave_smooth')
            ctx%pset%max_wave_smooth = value
        case ('gas_logu')
            ctx%pset%gas_logu = value
        case ('gas_logz')
            ctx%pset%gas_logz = value
        case ('igm_factor')
            ctx%pset%igm_factor = value
        case ('fagn')
            ctx%pset%fagn = value
        case ('agn_tau')
            ctx%pset%agn_tau = value
        case default
            status = FSPS_ERR_UNKNOWN_FLOAT_PARAM
        end select

        if (status == 0 .and. is_ssp_sensitive_float_param(key_norm)) then
            ctx%state%ssp_basis_is_dirty = .true.
        end if
    end subroutine fsps_context_set_param_float

    !> @brief Set a string parameter by key.
    !> @param[inout] ctx Context to update.
    !> @param[in] key Parameter name.
    !> @param[in] value Parameter value.
    !> @param[out] status Non-zero on unknown parameter.
    subroutine fsps_context_set_param_str(ctx, key, value, status)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        character(len=*), intent(in) :: value
        integer, intent(out) :: status
        character(len=64) :: key_norm

        status = 0
        key_norm = trim(key)
        select case (key_norm)
        case ('imf_filename')
            ctx%pset%imf_filename = trim(value)
        case ('sfh_filename')
            ctx%pset%sfh_filename = trim(value)
        case default
            status = FSPS_ERR_UNKNOWN_STRING_PARAM
        end select

        if (status == 0 .and. is_ssp_sensitive_str_param(key_norm)) then
            ctx%state%ssp_basis_is_dirty = .true.
        end if
    end subroutine fsps_context_set_param_str

    !> @brief Retrieve resolved FSPS data and output paths.
    !> @param[in] ctx Context to read.
    !> @param[out] sps_home_out FSPS root path.
    !> @param[out] data_home_out Data directory path.
    !> @param[out] output_home_out Output directory path.
    subroutine fsps_context_get_paths(ctx, sps_home_out, data_home_out, output_home_out)
        type(fsps_context_t), intent(in) :: ctx
        character(len=*), intent(out) :: sps_home_out
        character(len=*), intent(out) :: data_home_out
        character(len=*), intent(out) :: output_home_out

        sps_home_out = trim(ctx%sps_home)
        data_home_out = trim(ctx%data_home)
        output_home_out = trim(ctx%output_home)
    end subroutine fsps_context_get_paths

    !> @brief Compute an SSP grid for the current context and parameter set.
    !> @param[inout] ctx Context to use.
    !> @param[out] mass_ssp SSP mass history.
    !> @param[out] lbol_ssp SSP bolometric luminosity history.
    !> @param[out] spec_ssp SSP spectra.
    subroutine fsps_context_compute_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), dimension(:), intent(out) :: mass_ssp, lbol_ssp
        real(WP), dimension(:, :), intent(out) :: spec_ssp

        call fsps_context_ensure_setup(ctx)
        call fsps_context_prepare_pset(ctx)
        call generate_ssp_grid(ctx, ctx%pset, mass_ssp, lbol_ssp, spec_ssp)
    end subroutine fsps_context_compute_ssp

    !> @brief Compute CSP outputs for the current context.
    !> @param[inout] ctx Context to use.
    !> @param[in] write_compsp Output file flag.
    !> @param[in] nzin Number of metallicities.
    !> @param[in] outfile Output filename prefix.
    !> @param[in] mass_ssp SSP mass grid.
    !> @param[in] lbol_ssp SSP bolometric luminosity grid.
    !> @param[in] spec_ssp SSP spectra grid.
    !> @param[inout] ocompsp CSP output buffer.
    subroutine fsps_context_compute_csp(ctx, write_compsp, nzin, outfile, mass_ssp, lbol_ssp, spec_ssp, ocompsp)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: write_compsp, nzin
        character(len=*), intent(in) :: outfile
        real(WP), dimension(:, :), intent(in) :: mass_ssp, lbol_ssp
        real(WP), dimension(:, :, :), intent(in) :: spec_ssp
        type(COMPSPOUT), dimension(:), intent(inout) :: ocompsp
        type(COMPSPOUT), allocatable :: results(:)
        integer :: i, n_out, status

        call fsps_context_ensure_setup(ctx)
        call fsps_context_prepare_pset(ctx)

        if (ctx%pset%sfh == 2 .or. ctx%pset%sfh == 3) then
            call load_tabular_sfh(ctx, ctx%pset, nzin)
        end if

        call fsps_context_update_ssp_basis(ctx, spec_ssp, mass_ssp, lbol_ssp, nzin)

        call fsps_context_prepare_csp_workspace(ctx)
        call compute_csp_scenario(ctx, ctx%pset, nzin, results, status)
        if (status /= 0) then
            return
        end if

        if (write_compsp > 0 .and. .not. ctx%fast_mode) then
            call write_csp_output_files(ctx, ctx%pset, results, outfile, write_compsp)
        end if

        n_out = min(size(ocompsp), size(results))
        do i = 1, n_out
            ocompsp(i) = results(i)
        end do
        deallocate (results)
    end subroutine fsps_context_compute_csp

    !> @brief Update persistent SSP basis arrays used by CSP integration.
    !> @param[inout] ctx Context to update.
    !> @param[in] spec_ssp SSP spectra [nspec, nt, nz_in].
    !> @param[in] mass_ssp SSP mass [nt, nz_in].
    !> @param[in] lbol_ssp SSP bolometric luminosity [nt, nz_in].
    !> @param[in] nzin Number of metallicity bins actively provided.
    subroutine fsps_context_update_ssp_basis(ctx, spec_ssp, mass_ssp, lbol_ssp, nzin)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), intent(in) :: spec_ssp(:, :, :)
        real(WP), intent(in) :: mass_ssp(:, :)
        real(WP), intent(in) :: lbol_ssp(:, :)
        integer, intent(in) :: nzin
        logical :: mapped_new, needs_realloc

        mapped_new = .false.
        needs_realloc = .false.

        if (nzin <= 0) return
        if (size(spec_ssp, 3) < nzin .or. size(mass_ssp, 2) < nzin .or. size(lbol_ssp, 2) < nzin) return

        if (.not. allocated(ctx%state%ssp_basis_spec) .or. &
            .not. allocated(ctx%state%ssp_basis_mass) .or. &
            .not. allocated(ctx%state%ssp_basis_lbol)) then
            needs_realloc = .true.
        else
            ! Safe to check shape because we know they are all allocated
            if (any(shape(ctx%state%ssp_basis_spec) /= shape(spec_ssp)) .or. &
                any(shape(ctx%state%ssp_basis_mass) /= shape(mass_ssp)) .or. &
                any(shape(ctx%state%ssp_basis_lbol) /= shape(lbol_ssp))) then
                needs_realloc = .true.
            end if
        end if

        if (needs_realloc) then
                if (allocated(ctx%state%ssp_basis_spec)) then
                    !$acc exit data delete(ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
#ifdef OMP_TARGET
                    !$omp target exit data if(ctx%on_device) &
                    !$omp& map(delete: ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
#endif
                end if
                if (allocated(ctx%state%ssp_basis_spec)) deallocate(ctx%state%ssp_basis_spec)
                if (allocated(ctx%state%ssp_basis_mass)) deallocate(ctx%state%ssp_basis_mass)
                if (allocated(ctx%state%ssp_basis_lbol)) deallocate(ctx%state%ssp_basis_lbol)
            ctx%state%ssp_basis_is_dirty = .true.
        end if

        if (.not. allocated(ctx%state%ssp_basis_spec)) then
            allocate(ctx%state%ssp_basis_spec(size(spec_ssp, 1), size(spec_ssp, 2), size(spec_ssp, 3)))
            allocate(ctx%state%ssp_basis_mass(size(mass_ssp, 1), size(mass_ssp, 2)))
            allocate(ctx%state%ssp_basis_lbol(size(lbol_ssp, 1), size(lbol_ssp, 2)))
            mapped_new = .true.
        end if

        if (mapped_new) then
            !$acc enter data copyin(ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
#ifdef OMP_TARGET
            !$omp target enter data if(ctx%on_device) &
            !$omp& map(to: ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
#endif
            !$acc enter data attach(ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
            ctx%state%ssp_basis_is_dirty = .true.
        end if

        if (ctx%state%ssp_basis_is_dirty) then
            ctx%state%ssp_basis_spec(:, :, 1:nzin) = spec_ssp(:, :, 1:nzin)
            ctx%state%ssp_basis_mass(:, 1:nzin) = mass_ssp(:, 1:nzin)
            ctx%state%ssp_basis_lbol(:, 1:nzin) = lbol_ssp(:, 1:nzin)

            !$acc update device(ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
#ifdef OMP_TARGET
            !$omp target update if(ctx%on_device) &
            !$omp& to(ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
#endif
            ctx%state%ssp_basis_is_dirty = .false.
        end if
    end subroutine fsps_context_update_ssp_basis

    logical function is_ssp_sensitive_int_param(key) result(is_sensitive)
        character(len=*), intent(in) :: key

        is_sensitive = .false.
        select case (trim(key))
        case ('zmet', 'wgp1', 'wgp2', 'wgp3', 'evtype', 'imf_type', 'tpagb_norm_type', &
              'interpolation_type', 'use_wr_spectra', 'add_neb_emission', 'add_neb_continuum', &
              'add_xrb_emission', 'add_stellar_remnants', 'smooth_velocity', 'smooth_lsf', &
              'smoothspec_fast', 'vactoair_flag', 'use_isoc_mdot', 'add_agb_dust_model')
            is_sensitive = .true.
        end select
    end function is_ssp_sensitive_int_param

    logical function is_ssp_sensitive_float_param(key) result(is_sensitive)
        character(len=*), intent(in) :: key

        is_sensitive = .false.
        select case (trim(key))
        case ('tiny_logt', 'imf_upper_limit', 'imf_lower_limit', 'logt_wmb_hot', 'nebular_smooth_init', &
              'imf1', 'imf2', 'imf3', 'vdmc', 'mdave', 'dell', 'delt', 'sbss', 'fbhb', 'pagb', &
              'redgb', 'agb', 'masscut', 'fcstar', 'frac_xrb', 'sigma_smooth', 'min_wave_smooth', &
              'max_wave_smooth', 'gas_logu', 'gas_logz', 'agb_dust', 'frac_obrun')
            is_sensitive = .true.
        end select
    end function is_ssp_sensitive_float_param

    logical function is_ssp_sensitive_str_param(key) result(is_sensitive)
        character(len=*), intent(in) :: key

        is_sensitive = .false.
        select case (trim(key))
        case ('imf_filename')
            is_sensitive = .true.
        end select
    end function is_ssp_sensitive_str_param

    logical function pset_changes_ssp_basis(old_pset, new_pset) result(changed)
        type(PARAMS), intent(in) :: old_pset
        type(PARAMS), intent(in) :: new_pset

        changed = .false.

        if (old_pset%zmet /= new_pset%zmet) changed = .true.
        if (old_pset%wgp1 /= new_pset%wgp1) changed = .true.
        if (old_pset%wgp2 /= new_pset%wgp2) changed = .true.
        if (old_pset%wgp3 /= new_pset%wgp3) changed = .true.
        if (old_pset%evtype /= new_pset%evtype) changed = .true.

        if (old_pset%imf1 /= new_pset%imf1) changed = .true.
        if (old_pset%imf2 /= new_pset%imf2) changed = .true.
        if (old_pset%imf3 /= new_pset%imf3) changed = .true.
        if (old_pset%vdmc /= new_pset%vdmc) changed = .true.
        if (old_pset%mdave /= new_pset%mdave) changed = .true.
        if (old_pset%dell /= new_pset%dell) changed = .true.
        if (old_pset%delt /= new_pset%delt) changed = .true.
        if (old_pset%sbss /= new_pset%sbss) changed = .true.
        if (old_pset%fbhb /= new_pset%fbhb) changed = .true.
        if (old_pset%pagb /= new_pset%pagb) changed = .true.
        if (old_pset%redgb /= new_pset%redgb) changed = .true.
        if (old_pset%agb /= new_pset%agb) changed = .true.
        if (old_pset%masscut /= new_pset%masscut) changed = .true.
        if (old_pset%fcstar /= new_pset%fcstar) changed = .true.
        if (old_pset%frac_xrb /= new_pset%frac_xrb) changed = .true.
        if (old_pset%sigma_smooth /= new_pset%sigma_smooth) changed = .true.
        if (old_pset%min_wave_smooth /= new_pset%min_wave_smooth) changed = .true.
        if (old_pset%max_wave_smooth /= new_pset%max_wave_smooth) changed = .true.
        if (old_pset%gas_logu /= new_pset%gas_logu) changed = .true.
        if (old_pset%gas_logz /= new_pset%gas_logz) changed = .true.
        ! Circumstellar AGB dust screen (get_stellar_spectrum) and the ionizing-flux escape
        ! fraction used when nebular emission is added to the SSPs (add_neb_emission = 2)
        if (old_pset%agb_dust /= new_pset%agb_dust) changed = .true.
        if (old_pset%frac_obrun /= new_pset%frac_obrun) changed = .true.
        if (trim(old_pset%imf_filename) /= trim(new_pset%imf_filename)) changed = .true.

        if (allocated(old_pset%ssp_gen_age) .neqv. allocated(new_pset%ssp_gen_age)) then
            changed = .true.
        else if (allocated(old_pset%ssp_gen_age)) then
            if (size(old_pset%ssp_gen_age) /= size(new_pset%ssp_gen_age)) then
                changed = .true.
            else if (any(old_pset%ssp_gen_age /= new_pset%ssp_gen_age)) then
                changed = .true.
            end if
        end if
    end function pset_changes_ssp_basis

    !> @brief Set up workspace for the CSP hot path.
    !> @param[inout] ctx Context to prepare CSP workspace for.
    subroutine fsps_context_prepare_csp_workspace(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        integer :: nspec, nt, nz_max

        nspec = ctx%state%nspec
        nt = ctx%state%ntfull
        nz_max = ctx%state%nz

        ! Always allocate size-independent or constant-sized buffers
        if (.not. allocated(ctx%state%csp_emlin_old)) allocate(ctx%state%csp_emlin_old(NEMLINE, nt))
        if (.not. allocated(ctx%state%csp_emlin_young)) allocate(ctx%state%csp_emlin_young(NEMLINE, nt))
        if (.not. allocated(ctx%state%csp_emlin_final)) allocate(ctx%state%csp_emlin_final(NEMLINE))
        if (.not. allocated(ctx%state%gas_current_step_lines)) allocate(ctx%state%gas_current_step_lines(NEMLINE))
        if (.not. allocated(ctx%state%scalar_reductions)) allocate(ctx%state%scalar_reductions(2))
        if (.not. allocated(ctx%state%sfh_t_calc)) allocate(ctx%state%sfh_t_calc(NTABMAX))
        if (.not. allocated(ctx%state%sfh_sfr_calc)) allocate(ctx%state%sfh_sfr_calc(NTABMAX))
        if (.not. allocated(ctx%state%sfh_age_integrand)) allocate(ctx%state%sfh_age_integrand(NTABMAX))

        ! Allocate spectrum-dependent arrays
        if (nspec > 0) then
            ! Dust Workspace
            if (.not. allocated(ctx%state%dust_transmission_diffuse)) allocate(ctx%state%dust_transmission_diffuse(nspec))
            if (.not. allocated(ctx%state%dust_frequencies)) allocate(ctx%state%dust_frequencies(nspec))
            if (.not. allocated(ctx%state%dust_spec_total_work)) allocate(ctx%state%dust_spec_total_work(nspec, nt))
            if (.not. allocated(ctx%state%dust_emission_shape)) allocate(ctx%state%dust_emission_shape(nspec))
            if (.not. allocated(ctx%state%dust_emission_final)) allocate(ctx%state%dust_emission_final(nspec, nt))

            ! Gas Workspace
            if (.not. allocated(ctx%state%gas_neb_cont_reduced)) allocate(ctx%state%gas_neb_cont_reduced(nspec, NEBNAGE))
            if (.not. allocated(ctx%state%gas_neb_line_reduced)) allocate(ctx%state%gas_neb_line_reduced(NEMLINE, NEBNAGE))
            if (.not. allocated(ctx%state%gas_current_step_cont)) allocate(ctx%state%gas_current_step_cont(nspec))

            ! CSP Spectrum Workspace
            if (.not. allocated(ctx%state%csp_igm_transmission)) allocate(ctx%state%csp_igm_transmission(nspec))
            if (.not. allocated(ctx%state%csp_spec_final)) allocate(ctx%state%csp_spec_final(nspec))
            if (.not. allocated(ctx%state%spec_young)) allocate(ctx%state%spec_young(nspec, nt))
            if (.not. allocated(ctx%state%spec_old)) allocate(ctx%state%spec_old(nspec, nt))

            ! Fused Loop Output Buffers
            if (.not. allocated(ctx%state%out_csp_spec)) allocate(ctx%state%out_csp_spec(nspec, nt))
            if (.not. allocated(ctx%state%out_csp_emlin)) allocate(ctx%state%out_csp_emlin(NEMLINE, nt))
            if (.not. allocated(ctx%state%out_mass_csp)) allocate(ctx%state%out_mass_csp(nt))
            if (.not. allocated(ctx%state%out_lbol_csp)) allocate(ctx%state%out_lbol_csp(nt))
        end if

        ! Allocate age/metallicity dependent arrays
        if (nt > 0 .and. nz_max > 0) then
            if (.not. allocated(ctx%state%csp_weights)) allocate(ctx%state%csp_weights(nt, nz_max, nt))
            if (.not. allocated(ctx%state%sfh_w_tmp1)) allocate(ctx%state%sfh_w_tmp1(max(nt, 3), nt))
            if (.not. allocated(ctx%state%sfh_w_tmp2)) allocate(ctx%state%sfh_w_tmp2(max(nt, 3), nt))
            if (.not. allocated(ctx%state%csp_ssp_lum_linear)) allocate(ctx%state%csp_ssp_lum_linear(nt, nz_max))
            if (.not. allocated(ctx%state%csp_emlin_grid)) allocate(ctx%state%csp_emlin_grid(NEMLINE, nt, nz_max))

            if (nspec > 0) then
                if (.not. allocated(ctx%state%csp_ssp_grid)) allocate(ctx%state%csp_ssp_grid(nspec, nt, nz_max))
            end if
        end if
    end subroutine fsps_context_prepare_csp_workspace

    !> @brief Ensure allocatable parameter-set arrays are allocated.
    !> @param[inout] ctx Context to update.
    subroutine fsps_context_prepare_pset(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (.not. allocated(ctx%pset%mag_compute)) then
            allocate (ctx%pset%mag_compute(ctx%state%nbands))
            ctx%pset%mag_compute = 1
        end if
        if (.not. allocated(ctx%pset%ssp_gen_age)) then
            allocate (ctx%pset%ssp_gen_age(ctx%state%nt))
            ctx%pset%ssp_gen_age = 1
        end if
    end subroutine fsps_context_prepare_pset

    !> @brief Moves the context and all its associated data to the device.
    !> @details Performs a deep copy of the context state structure.
    !> @param[inout] ctx Context to move.
    subroutine fsps_context_move_to_device(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        ! A second call refreshes the device copies rather than stacking reference counts
        if (ctx%on_device) call fsps_context_remove_from_device(ctx)
        call batch_out_release(ctx%state%batch_out, .false.)   ! host-only output from before
        ctx%on_device = .true.
        ctx%settings_dirty = .false.   ! copied with ctx below

        ! Pre-allocate CSP workspace so it gets mapped correctly
        call fsps_context_prepare_csp_workspace(ctx)

        ! Copy the main structure
        !$acc enter data copyin(ctx)
#ifdef OMP_TARGET
        !$omp target enter data map(to: ctx)
#endif
        !$acc enter data copyin(ctx%state)
#ifdef OMP_TARGET
        !$omp target enter data map(to: ctx%state)
#endif
        ! OpenMP: the SSP basis of a context that had one before moving (update_ssp_basis maps it
        ! only for contexts on the device; OpenACC maps it there regardless)
        if (allocated(ctx%state%ssp_basis_spec)) then
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
#endif
        end if
        
        ! Copy allocatable parameter arrays in pset
        if (allocated(ctx%pset%mag_compute)) then
            !$acc enter data copyin(ctx%pset%mag_compute)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%pset%mag_compute)
#endif
            !$acc enter data attach(ctx%pset%mag_compute)
        end if
        if (allocated(ctx%pset%ssp_gen_age)) then
            !$acc enter data copyin(ctx%pset%ssp_gen_age)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%pset%ssp_gen_age)
#endif
            !$acc enter data attach(ctx%pset%ssp_gen_age)
        end if

        ! Copy pointer/allocatable components of state
        if (associated(ctx%state%indexdefined)) then
            !$acc enter data copyin(ctx%state%indexdefined)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%indexdefined)
#endif
            !$acc enter data attach(ctx%state%indexdefined)
        end if
        if (associated(ctx%state%wgdust)) then
            !$acc enter data copyin(ctx%state%wgdust)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%wgdust)
#endif
            !$acc enter data attach(ctx%state%wgdust)
        end if
        if (associated(ctx%state%g03smcextn)) then
            !$acc enter data copyin(ctx%state%g03smcextn)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%g03smcextn)
#endif
            !$acc enter data attach(ctx%state%g03smcextn)
        end if
        if (associated(ctx%state%bands)) then
            !$acc enter data copyin(ctx%state%bands)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%bands)
#endif
            !$acc enter data attach(ctx%state%bands)
        end if
        if (associated(ctx%state%magsun)) then
            !$acc enter data copyin(ctx%state%magsun)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%magsun)
#endif
            !$acc enter data attach(ctx%state%magsun)
        end if
        if (associated(ctx%state%magvega)) then
            !$acc enter data copyin(ctx%state%magvega)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%magvega)
#endif
            !$acc enter data attach(ctx%state%magvega)
        end if
        if (associated(ctx%state%filter_leff)) then
            !$acc enter data copyin(ctx%state%filter_leff)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%filter_leff)
#endif
            !$acc enter data attach(ctx%state%filter_leff)
        end if
        if (associated(ctx%state%vega_spec)) then
            !$acc enter data copyin(ctx%state%vega_spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%vega_spec)
#endif
            !$acc enter data attach(ctx%state%vega_spec)
        end if
        if (associated(ctx%state%sun_spec)) then
            !$acc enter data copyin(ctx%state%sun_spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sun_spec)
#endif
            !$acc enter data attach(ctx%state%sun_spec)
        end if
        if (associated(ctx%state%spec_lambda)) then
            !$acc enter data copyin(ctx%state%spec_lambda)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%spec_lambda)
#endif
            !$acc enter data attach(ctx%state%spec_lambda)
        end if
        if (associated(ctx%state%spec_nu)) then
            !$acc enter data copyin(ctx%state%spec_nu)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%spec_nu)
#endif
            !$acc enter data attach(ctx%state%spec_nu)
        end if
        if (associated(ctx%state%spec_res)) then
            !$acc enter data copyin(ctx%state%spec_res)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%spec_res)
#endif
            !$acc enter data attach(ctx%state%spec_res)
        end if
        if (associated(ctx%state%speclib)) then
            !$acc enter data copyin(ctx%state%speclib)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%speclib)
#endif
            !$acc enter data attach(ctx%state%speclib)
        end if
        if (associated(ctx%state%wmb_spec)) then
            !$acc enter data copyin(ctx%state%wmb_spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%wmb_spec)
#endif
            !$acc enter data attach(ctx%state%wmb_spec)
        end if
        if (associated(ctx%state%agb_spec_o)) then
            !$acc enter data copyin(ctx%state%agb_spec_o)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agb_spec_o)
#endif
            !$acc enter data attach(ctx%state%agb_spec_o)
        end if
        if (associated(ctx%state%agb_logt_o)) then
            !$acc enter data copyin(ctx%state%agb_logt_o)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agb_logt_o)
#endif
            !$acc enter data attach(ctx%state%agb_logt_o)
        end if
        if (associated(ctx%state%agb_spec_c)) then
            !$acc enter data copyin(ctx%state%agb_spec_c)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agb_spec_c)
#endif
            !$acc enter data attach(ctx%state%agb_spec_c)
        end if
        if (associated(ctx%state%agb_logt_c)) then
            !$acc enter data copyin(ctx%state%agb_logt_c)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agb_logt_c)
#endif
            !$acc enter data attach(ctx%state%agb_logt_c)
        end if
        if (associated(ctx%state%agb_spec_car)) then
            !$acc enter data copyin(ctx%state%agb_spec_car)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agb_spec_car)
#endif
            !$acc enter data attach(ctx%state%agb_spec_car)
        end if
        if (associated(ctx%state%speclib_agb_smooth)) then
            !$acc enter data copyin(ctx%state%speclib_agb_smooth)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%speclib_agb_smooth)
#endif
            !$acc enter data attach(ctx%state%speclib_agb_smooth)
        end if
        if (associated(ctx%state%agb_spec_o_smooth)) then
            !$acc enter data copyin(ctx%state%agb_spec_o_smooth)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agb_spec_o_smooth)
#endif
            !$acc enter data attach(ctx%state%agb_spec_o_smooth)
        end if
        if (associated(ctx%state%agb_spec_c_smooth)) then
            !$acc enter data copyin(ctx%state%agb_spec_c_smooth)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agb_spec_c_smooth)
#endif
            !$acc enter data attach(ctx%state%agb_spec_c_smooth)
        end if
        if (associated(ctx%state%agb_spec_car_smooth)) then
            !$acc enter data copyin(ctx%state%agb_spec_car_smooth)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agb_spec_car_smooth)
#endif
            !$acc enter data attach(ctx%state%agb_spec_car_smooth)
        end if
        if (associated(ctx%state%pagb_spec)) then
            !$acc enter data copyin(ctx%state%pagb_spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%pagb_spec)
#endif
            !$acc enter data attach(ctx%state%pagb_spec)
        end if
        if (associated(ctx%state%wrn_spec)) then
            !$acc enter data copyin(ctx%state%wrn_spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%wrn_spec)
#endif
            !$acc enter data attach(ctx%state%wrn_spec)
        end if
        if (associated(ctx%state%wrc_spec)) then
            !$acc enter data copyin(ctx%state%wrc_spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%wrc_spec)
#endif
            !$acc enter data attach(ctx%state%wrc_spec)
        end if
        if (associated(ctx%state%qpaharr)) then
            !$acc enter data copyin(ctx%state%qpaharr)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%qpaharr)
#endif
            !$acc enter data attach(ctx%state%qpaharr)
        end if
        if (associated(ctx%state%uminarr)) then
            !$acc enter data copyin(ctx%state%uminarr)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%uminarr)
#endif
            !$acc enter data attach(ctx%state%uminarr)
        end if
        if (associated(ctx%state%lambda_dustem)) then
            !$acc enter data copyin(ctx%state%lambda_dustem)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%lambda_dustem)
#endif
            !$acc enter data attach(ctx%state%lambda_dustem)
        end if
        if (associated(ctx%state%dustem_dustem)) then
            !$acc enter data copyin(ctx%state%dustem_dustem)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%dustem_dustem)
#endif
            !$acc enter data attach(ctx%state%dustem_dustem)
        end if
        if (associated(ctx%state%dustem2_dustem)) then
            !$acc enter data copyin(ctx%state%dustem2_dustem)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%dustem2_dustem)
#endif
            !$acc enter data attach(ctx%state%dustem2_dustem)
        end if
        if (associated(ctx%state%flux_dagb)) then
            !$acc enter data copyin(ctx%state%flux_dagb)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%flux_dagb)
#endif
            !$acc enter data attach(ctx%state%flux_dagb)
        end if
        if (associated(ctx%state%nebem_cont)) then
            !$acc enter data copyin(ctx%state%nebem_cont)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%nebem_cont)
#endif
            !$acc enter data attach(ctx%state%nebem_cont)
        end if
        if (associated(ctx%state%xnebem_cont)) then
            !$acc enter data copyin(ctx%state%xnebem_cont)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%xnebem_cont)
#endif
            !$acc enter data attach(ctx%state%xnebem_cont)
        end if
        if (associated(ctx%state%neb_res_min)) then
            !$acc enter data copyin(ctx%state%neb_res_min)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%neb_res_min)
#endif
            !$acc enter data attach(ctx%state%neb_res_min)
        end if
        if (associated(ctx%state%gaussnebarr)) then
            !$acc enter data copyin(ctx%state%gaussnebarr)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%gaussnebarr)
#endif
            !$acc enter data attach(ctx%state%gaussnebarr)
        end if
        if (associated(ctx%state%agndust_spec)) then
            !$acc enter data copyin(ctx%state%agndust_spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%agndust_spec)
#endif
            !$acc enter data attach(ctx%state%agndust_spec)
        end if
        if (associated(ctx%state%mact_isoc)) then
            !$acc enter data copyin(ctx%state%mact_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%mact_isoc)
#endif
            !$acc enter data attach(ctx%state%mact_isoc)
        end if
        if (associated(ctx%state%logl_isoc)) then
            !$acc enter data copyin(ctx%state%logl_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%logl_isoc)
#endif
            !$acc enter data attach(ctx%state%logl_isoc)
        end if
        if (associated(ctx%state%logt_isoc)) then
            !$acc enter data copyin(ctx%state%logt_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%logt_isoc)
#endif
            !$acc enter data attach(ctx%state%logt_isoc)
        end if
        if (associated(ctx%state%logg_isoc)) then
            !$acc enter data copyin(ctx%state%logg_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%logg_isoc)
#endif
            !$acc enter data attach(ctx%state%logg_isoc)
        end if
        if (associated(ctx%state%ffco_isoc)) then
            !$acc enter data copyin(ctx%state%ffco_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%ffco_isoc)
#endif
            !$acc enter data attach(ctx%state%ffco_isoc)
        end if
        if (associated(ctx%state%phase_isoc)) then
            !$acc enter data copyin(ctx%state%phase_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%phase_isoc)
#endif
            !$acc enter data attach(ctx%state%phase_isoc)
        end if
        if (associated(ctx%state%mini_isoc)) then
            !$acc enter data copyin(ctx%state%mini_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%mini_isoc)
#endif
            !$acc enter data attach(ctx%state%mini_isoc)
        end if
        if (associated(ctx%state%lmdot_isoc)) then
            !$acc enter data copyin(ctx%state%lmdot_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%lmdot_isoc)
#endif
            !$acc enter data attach(ctx%state%lmdot_isoc)
        end if
        if (associated(ctx%state%nmass_isoc)) then
            !$acc enter data copyin(ctx%state%nmass_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%nmass_isoc)
#endif
            !$acc enter data attach(ctx%state%nmass_isoc)
        end if
        if (associated(ctx%state%timestep_isoc)) then
            !$acc enter data copyin(ctx%state%timestep_isoc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%timestep_isoc)
#endif
            !$acc enter data attach(ctx%state%timestep_isoc)
        end if
        if (associated(ctx%state%zlegend)) then
            !$acc enter data copyin(ctx%state%zlegend)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%zlegend)
#endif
            !$acc enter data attach(ctx%state%zlegend)
        end if
        if (associated(ctx%state%zlegendinit)) then
            !$acc enter data copyin(ctx%state%zlegendinit)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%zlegendinit)
#endif
            !$acc enter data attach(ctx%state%zlegendinit)
        end if
        if (allocated(ctx%state%spec_ssp_zz)) then
            !$acc enter data copyin(ctx%state%spec_ssp_zz)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%spec_ssp_zz)
#endif
            !$acc enter data attach(ctx%state%spec_ssp_zz)
        end if
        if (allocated(ctx%state%mass_ssp_zz)) then
            !$acc enter data copyin(ctx%state%mass_ssp_zz)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%mass_ssp_zz)
#endif
            !$acc enter data attach(ctx%state%mass_ssp_zz)
        end if
        if (allocated(ctx%state%lbol_ssp_zz)) then
            !$acc enter data copyin(ctx%state%lbol_ssp_zz)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%lbol_ssp_zz)
#endif
            !$acc enter data attach(ctx%state%lbol_ssp_zz)
        end if
        if (associated(ctx%state%time_full)) then
            !$acc enter data copyin(ctx%state%time_full)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%time_full)
#endif
            !$acc enter data attach(ctx%state%time_full)
        end if
        if (allocated(ctx%state%weight_ssp)) then
            !$acc enter data copyin(ctx%state%weight_ssp)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%weight_ssp)
#endif
            !$acc enter data attach(ctx%state%weight_ssp)
        end if
        if (allocated(ctx%state%spec_young)) then
            !$acc enter data copyin(ctx%state%spec_young)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%spec_young)
#endif
            !$acc enter data attach(ctx%state%spec_young)
        end if
        if (allocated(ctx%state%spec_old)) then
            !$acc enter data copyin(ctx%state%spec_old)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%spec_old)
#endif
            !$acc enter data attach(ctx%state%spec_old)
        end if
        if (allocated(ctx%state%ssp_temp_grid)) then
            !$acc enter data copyin(ctx%state%ssp_temp_grid)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%ssp_temp_grid)
#endif
            !$acc enter data attach(ctx%state%ssp_temp_grid)
        end if
        if (allocated(ctx%state%ssp_active_idx)) then
            !$acc enter data copyin(ctx%state%ssp_active_idx)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%ssp_active_idx)
#endif
            !$acc enter data attach(ctx%state%ssp_active_idx)
        end if
        if (allocated(ctx%state%ssp_active_w)) then
            !$acc enter data copyin(ctx%state%ssp_active_w)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%ssp_active_w)
#endif
            !$acc enter data attach(ctx%state%ssp_active_w)
        end if
        if (associated(ctx%state%bpass_spec_ssp)) then
            !$acc enter data copyin(ctx%state%bpass_spec_ssp)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%bpass_spec_ssp)
#endif
            !$acc enter data attach(ctx%state%bpass_spec_ssp)
        end if
        if (associated(ctx%state%bpass_mass_ssp)) then
            !$acc enter data copyin(ctx%state%bpass_mass_ssp)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%bpass_mass_ssp)
#endif
            !$acc enter data attach(ctx%state%bpass_mass_ssp)
        end if
        if (associated(ctx%state%lam_xrb)) then
            !$acc enter data copyin(ctx%state%lam_xrb)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%lam_xrb)
#endif
            !$acc enter data attach(ctx%state%lam_xrb)
        end if
        if (associated(ctx%state%spec_xrb)) then
            !$acc enter data copyin(ctx%state%spec_xrb)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%spec_xrb)
#endif
            !$acc enter data attach(ctx%state%spec_xrb)
        end if
        if (associated(ctx%state%ages_xrb)) then
            !$acc enter data copyin(ctx%state%ages_xrb)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%ages_xrb)
#endif
            !$acc enter data attach(ctx%state%ages_xrb)
        end if
        if (associated(ctx%state%zmet_xrb)) then
            !$acc enter data copyin(ctx%state%zmet_xrb)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%zmet_xrb)
#endif
            !$acc enter data attach(ctx%state%zmet_xrb)
        end if
        if (allocated(ctx%state%lsfinfo%lsf)) then
            !$acc enter data copyin(ctx%state%lsfinfo%lsf)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%lsfinfo%lsf)
#endif
            !$acc enter data attach(ctx%state%lsfinfo%lsf)
        end if
        ! Powell and Sedfit data are usually observation data, typically not needed for simulation,
        ! but we include them to be safe if they are present.
        if (allocated(ctx%state%powell_data%mags)) then
            !$acc enter data copyin(ctx%state%powell_data%mags)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%powell_data%mags)
#endif
            !$acc enter data attach(ctx%state%powell_data%mags)
        end if
        if (allocated(ctx%state%powell_data%magerr)) then
            !$acc enter data copyin(ctx%state%powell_data%magerr)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%powell_data%magerr)
#endif
            !$acc enter data attach(ctx%state%powell_data%magerr)
        end if
        if (allocated(ctx%state%powell_data%spec)) then
            !$acc enter data copyin(ctx%state%powell_data%spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%powell_data%spec)
#endif
            !$acc enter data attach(ctx%state%powell_data%spec)
        end if
        if (allocated(ctx%state%powell_data%specerr)) then
            !$acc enter data copyin(ctx%state%powell_data%specerr)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%powell_data%specerr)
#endif
            !$acc enter data attach(ctx%state%powell_data%specerr)
        end if
        if (allocated(ctx%state%sedfit_data%mags)) then
            !$acc enter data copyin(ctx%state%sedfit_data%mags)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sedfit_data%mags)
#endif
            !$acc enter data attach(ctx%state%sedfit_data%mags)
        end if
        if (allocated(ctx%state%sedfit_data%magerr)) then
            !$acc enter data copyin(ctx%state%sedfit_data%magerr)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sedfit_data%magerr)
#endif
            !$acc enter data attach(ctx%state%sedfit_data%magerr)
        end if
        if (allocated(ctx%state%sedfit_data%spec)) then
            !$acc enter data copyin(ctx%state%sedfit_data%spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sedfit_data%spec)
#endif
            !$acc enter data attach(ctx%state%sedfit_data%spec)
        end if
        if (allocated(ctx%state%sedfit_data%specerr)) then
            !$acc enter data copyin(ctx%state%sedfit_data%specerr)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sedfit_data%specerr)
#endif
            !$acc enter data attach(ctx%state%sedfit_data%specerr)
        end if

        ! --- Permanent CSP Workspace ---
        if (allocated(ctx%state%csp_ssp_grid)) then
            !$acc enter data copyin(ctx%state%csp_ssp_grid)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_ssp_grid)
#endif
            !$acc enter data attach(ctx%state%csp_ssp_grid)
        end if
        if (allocated(ctx%state%csp_emlin_grid)) then
            !$acc enter data copyin(ctx%state%csp_emlin_grid)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_emlin_grid)
#endif
            !$acc enter data attach(ctx%state%csp_emlin_grid)
        end if
        if (allocated(ctx%state%csp_ssp_lum_linear)) then
            !$acc enter data copyin(ctx%state%csp_ssp_lum_linear)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_ssp_lum_linear)
#endif
            !$acc enter data attach(ctx%state%csp_ssp_lum_linear)
        end if
        if (allocated(ctx%state%csp_igm_transmission)) then
            !$acc enter data copyin(ctx%state%csp_igm_transmission)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_igm_transmission)
#endif
            !$acc enter data attach(ctx%state%csp_igm_transmission)
        end if
        if (allocated(ctx%state%csp_spec_final)) then
            !$acc enter data copyin(ctx%state%csp_spec_final)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_spec_final)
#endif
            !$acc enter data attach(ctx%state%csp_spec_final)
        end if
        if (allocated(ctx%state%csp_emlin_final)) then
            !$acc enter data copyin(ctx%state%csp_emlin_final)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_emlin_final)
#endif
            !$acc enter data attach(ctx%state%csp_emlin_final)
        end if
        if (allocated(ctx%state%csp_weights)) then
            !$acc enter data copyin(ctx%state%csp_weights)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_weights)
#endif
            !$acc enter data attach(ctx%state%csp_weights)
        end if
        if (allocated(ctx%state%csp_emlin_young)) then
            !$acc enter data copyin(ctx%state%csp_emlin_young)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_emlin_young)
#endif
            !$acc enter data attach(ctx%state%csp_emlin_young)
        end if
        if (allocated(ctx%state%csp_emlin_old)) then
            !$acc enter data copyin(ctx%state%csp_emlin_old)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%csp_emlin_old)
#endif
            !$acc enter data attach(ctx%state%csp_emlin_old)
        end if

        ! --- Persistent Physics Workspaces ---
        if (allocated(ctx%state%dust_transmission_diffuse)) then
            !$acc enter data copyin(ctx%state%dust_transmission_diffuse)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%dust_transmission_diffuse)
#endif
            !$acc enter data attach(ctx%state%dust_transmission_diffuse)
        end if
        if (allocated(ctx%state%dust_frequencies)) then
            !$acc enter data copyin(ctx%state%dust_frequencies)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%dust_frequencies)
#endif
            !$acc enter data attach(ctx%state%dust_frequencies)
        end if
        if (allocated(ctx%state%dust_spec_total_work)) then
            !$acc enter data copyin(ctx%state%dust_spec_total_work)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%dust_spec_total_work)
#endif
            !$acc enter data attach(ctx%state%dust_spec_total_work)
        end if
        if (allocated(ctx%state%dust_emission_shape)) then
            !$acc enter data copyin(ctx%state%dust_emission_shape)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%dust_emission_shape)
#endif
            !$acc enter data attach(ctx%state%dust_emission_shape)
        end if
        if (allocated(ctx%state%dust_emission_final)) then
            !$acc enter data copyin(ctx%state%dust_emission_final)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%dust_emission_final)
#endif
            !$acc enter data attach(ctx%state%dust_emission_final)
        end if
        if (allocated(ctx%state%gas_neb_cont_reduced)) then
            !$acc enter data copyin(ctx%state%gas_neb_cont_reduced)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%gas_neb_cont_reduced)
#endif
            !$acc enter data attach(ctx%state%gas_neb_cont_reduced)
        end if
        if (allocated(ctx%state%gas_neb_line_reduced)) then
            !$acc enter data copyin(ctx%state%gas_neb_line_reduced)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%gas_neb_line_reduced)
#endif
            !$acc enter data attach(ctx%state%gas_neb_line_reduced)
        end if
        if (allocated(ctx%state%gas_current_step_cont)) then
            !$acc enter data copyin(ctx%state%gas_current_step_cont)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%gas_current_step_cont)
#endif
            !$acc enter data attach(ctx%state%gas_current_step_cont)
        end if
        if (allocated(ctx%state%gas_current_step_lines)) then
            !$acc enter data copyin(ctx%state%gas_current_step_lines)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%gas_current_step_lines)
#endif
            !$acc enter data attach(ctx%state%gas_current_step_lines)
        end if
        if (allocated(ctx%state%scalar_reductions)) then
            !$acc enter data copyin(ctx%state%scalar_reductions)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%scalar_reductions)
#endif
            !$acc enter data attach(ctx%state%scalar_reductions)
        end if
        if (allocated(ctx%state%sfh_t_calc)) then
            !$acc enter data copyin(ctx%state%sfh_t_calc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sfh_t_calc)
#endif
            !$acc enter data attach(ctx%state%sfh_t_calc)
        end if
        if (allocated(ctx%state%sfh_sfr_calc)) then
            !$acc enter data copyin(ctx%state%sfh_sfr_calc)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sfh_sfr_calc)
#endif
            !$acc enter data attach(ctx%state%sfh_sfr_calc)
        end if
        if (allocated(ctx%state%sfh_age_integrand)) then
            !$acc enter data copyin(ctx%state%sfh_age_integrand)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sfh_age_integrand)
#endif
            !$acc enter data attach(ctx%state%sfh_age_integrand)
        end if
        if (allocated(ctx%state%sfh_w_tmp1)) then
            !$acc enter data copyin(ctx%state%sfh_w_tmp1)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sfh_w_tmp1)
#endif
            !$acc enter data attach(ctx%state%sfh_w_tmp1)
        end if
        if (allocated(ctx%state%sfh_w_tmp2)) then
            !$acc enter data copyin(ctx%state%sfh_w_tmp2)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%sfh_w_tmp2)
#endif
            !$acc enter data attach(ctx%state%sfh_w_tmp2)
        end if

        ! --- Fused Loop Output Buffers ---
        if (allocated(ctx%state%out_csp_spec)) then
            !$acc enter data copyin(ctx%state%out_csp_spec)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%out_csp_spec)
#endif
            !$acc enter data attach(ctx%state%out_csp_spec)
        end if
        if (allocated(ctx%state%out_csp_emlin)) then
            !$acc enter data copyin(ctx%state%out_csp_emlin)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%out_csp_emlin)
#endif
            !$acc enter data attach(ctx%state%out_csp_emlin)
        end if
        if (allocated(ctx%state%out_mass_csp)) then
            !$acc enter data copyin(ctx%state%out_mass_csp)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%out_mass_csp)
#endif
            !$acc enter data attach(ctx%state%out_mass_csp)
        end if
        if (allocated(ctx%state%out_lbol_csp)) then
            !$acc enter data copyin(ctx%state%out_lbol_csp)
#ifdef OMP_TARGET
            !$omp target enter data map(to: ctx%state%out_lbol_csp)
#endif
            !$acc enter data attach(ctx%state%out_lbol_csp)
        end if
    end subroutine fsps_context_move_to_device

    !> @brief Removes the context and its data from the device.
    !> @param[inout] ctx Context to remove.
    subroutine fsps_context_remove_from_device(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        ! Nothing mapped (also avoids an exit data before any OpenACC initialization, which
        ! crashes in the HPC SDK 25.1 runtime)
        if (.not. ctx%on_device) return
        if (allocated(ctx%state%ssp_basis_spec)) then
#ifdef OMP_TARGET
            !$omp target exit data map(delete: ctx%state%ssp_basis_spec, ctx%state%ssp_basis_mass, ctx%state%ssp_basis_lbol)
#endif
        end if

        associate(s => ctx%state)
            if (allocated(s%sedfit_data%specerr)) then
                !$acc exit data delete(s%sedfit_data%specerr)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sedfit_data%specerr)
#endif
            end if
            if (allocated(s%sedfit_data%spec)) then
                !$acc exit data delete(s%sedfit_data%spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sedfit_data%spec)
#endif
            end if
            if (allocated(s%sedfit_data%magerr)) then
                !$acc exit data delete(s%sedfit_data%magerr)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sedfit_data%magerr)
#endif
            end if
            if (allocated(s%sedfit_data%mags)) then
                !$acc exit data delete(s%sedfit_data%mags)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sedfit_data%mags)
#endif
            end if
            if (allocated(s%powell_data%specerr)) then
                !$acc exit data delete(s%powell_data%specerr)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%powell_data%specerr)
#endif
            end if
            if (allocated(s%powell_data%spec)) then
                !$acc exit data delete(s%powell_data%spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%powell_data%spec)
#endif
            end if
            if (allocated(s%powell_data%magerr)) then
                !$acc exit data delete(s%powell_data%magerr)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%powell_data%magerr)
#endif
            end if
            if (allocated(s%powell_data%mags)) then
                !$acc exit data delete(s%powell_data%mags)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%powell_data%mags)
#endif
            end if
            if (allocated(s%lsfinfo%lsf)) then
                !$acc exit data delete(s%lsfinfo%lsf)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%lsfinfo%lsf)
#endif
            end if
            if (associated(s%zmet_xrb)) then
                !$acc exit data delete(s%zmet_xrb)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%zmet_xrb)
#endif
            end if
            if (associated(s%ages_xrb)) then
                !$acc exit data delete(s%ages_xrb)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%ages_xrb)
#endif
            end if
            if (associated(s%spec_xrb)) then
                !$acc exit data delete(s%spec_xrb)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%spec_xrb)
#endif
            end if
            if (associated(s%lam_xrb)) then
                !$acc exit data delete(s%lam_xrb)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%lam_xrb)
#endif
            end if
            if (associated(s%bpass_mass_ssp)) then
                !$acc exit data delete(s%bpass_mass_ssp)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%bpass_mass_ssp)
#endif
            end if
            if (associated(s%bpass_spec_ssp)) then
                !$acc exit data delete(s%bpass_spec_ssp)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%bpass_spec_ssp)
#endif
            end if
            if (allocated(s%spec_old)) then
                !$acc exit data delete(s%spec_old)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%spec_old)
#endif
            end if
            if (allocated(s%ssp_temp_grid)) then
                !$acc exit data delete(s%ssp_temp_grid)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%ssp_temp_grid)
#endif
            end if
            if (allocated(s%ssp_active_idx)) then
                !$acc exit data delete(s%ssp_active_idx)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%ssp_active_idx)
#endif
            end if
            if (allocated(s%ssp_active_w)) then
                !$acc exit data delete(s%ssp_active_w)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%ssp_active_w)
#endif
            end if
            if (allocated(s%spec_young)) then
                !$acc exit data delete(s%spec_young)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%spec_young)
#endif
            end if
            if (allocated(s%weight_ssp)) then
                !$acc exit data delete(s%weight_ssp)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%weight_ssp)
#endif
            end if
            if (associated(s%time_full)) then
                !$acc exit data delete(s%time_full)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%time_full)
#endif
            end if
            if (allocated(s%lbol_ssp_zz)) then
                !$acc exit data delete(s%lbol_ssp_zz)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%lbol_ssp_zz)
#endif
            end if
            if (allocated(s%mass_ssp_zz)) then
                !$acc exit data delete(s%mass_ssp_zz)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%mass_ssp_zz)
#endif
            end if
            if (allocated(s%spec_ssp_zz)) then
                !$acc exit data delete(s%spec_ssp_zz)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%spec_ssp_zz)
#endif
            end if
            if (associated(s%zlegendinit)) then
                !$acc exit data delete(s%zlegendinit)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%zlegendinit)
#endif
            end if
            if (associated(s%zlegend)) then
                !$acc exit data delete(s%zlegend)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%zlegend)
#endif
            end if
            if (associated(s%timestep_isoc)) then
                !$acc exit data delete(s%timestep_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%timestep_isoc)
#endif
            end if
            if (associated(s%nmass_isoc)) then
                !$acc exit data delete(s%nmass_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%nmass_isoc)
#endif
            end if
            if (associated(s%lmdot_isoc)) then
                !$acc exit data delete(s%lmdot_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%lmdot_isoc)
#endif
            end if
            if (associated(s%mini_isoc)) then
                !$acc exit data delete(s%mini_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%mini_isoc)
#endif
            end if
            if (associated(s%phase_isoc)) then
                !$acc exit data delete(s%phase_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%phase_isoc)
#endif
            end if
            if (associated(s%ffco_isoc)) then
                !$acc exit data delete(s%ffco_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%ffco_isoc)
#endif
            end if
            if (associated(s%logg_isoc)) then
                !$acc exit data delete(s%logg_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%logg_isoc)
#endif
            end if
            if (associated(s%logt_isoc)) then
                !$acc exit data delete(s%logt_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%logt_isoc)
#endif
            end if
            if (associated(s%logl_isoc)) then
                !$acc exit data delete(s%logl_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%logl_isoc)
#endif
            end if
            if (associated(s%mact_isoc)) then
                !$acc exit data delete(s%mact_isoc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%mact_isoc)
#endif
            end if
            if (associated(s%agndust_spec)) then
                !$acc exit data delete(s%agndust_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agndust_spec)
#endif
            end if
            if (associated(s%gaussnebarr)) then
                !$acc exit data delete(s%gaussnebarr)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%gaussnebarr)
#endif
            end if
            if (associated(s%neb_res_min)) then
                !$acc exit data delete(s%neb_res_min)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%neb_res_min)
#endif
            end if
            if (associated(s%xnebem_cont)) then
                !$acc exit data delete(s%xnebem_cont)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%xnebem_cont)
#endif
            end if
            if (associated(s%nebem_cont)) then
                !$acc exit data delete(s%nebem_cont)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%nebem_cont)
#endif
            end if
            if (associated(s%flux_dagb)) then
                !$acc exit data delete(s%flux_dagb)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%flux_dagb)
#endif
            end if
            if (associated(s%dustem2_dustem)) then
                !$acc exit data delete(s%dustem2_dustem)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%dustem2_dustem)
#endif
            end if
            if (associated(s%dustem_dustem)) then
                !$acc exit data delete(s%dustem_dustem)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%dustem_dustem)
#endif
            end if
            if (associated(s%lambda_dustem)) then
                !$acc exit data delete(s%lambda_dustem)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%lambda_dustem)
#endif
            end if
            if (associated(s%uminarr)) then
                !$acc exit data delete(s%uminarr)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%uminarr)
#endif
            end if
            if (associated(s%qpaharr)) then
                !$acc exit data delete(s%qpaharr)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%qpaharr)
#endif
            end if
            if (associated(s%wrc_spec)) then
                !$acc exit data delete(s%wrc_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%wrc_spec)
#endif
            end if
            if (associated(s%wrn_spec)) then
                !$acc exit data delete(s%wrn_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%wrn_spec)
#endif
            end if
            if (associated(s%pagb_spec)) then
                !$acc exit data delete(s%pagb_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%pagb_spec)
#endif
            end if
            if (associated(s%agb_spec_car)) then
                !$acc exit data delete(s%agb_spec_car)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agb_spec_car)
#endif
            end if
            if (associated(s%speclib_agb_smooth)) then
                !$acc exit data delete(s%speclib_agb_smooth)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%speclib_agb_smooth)
#endif
            end if
            if (associated(s%agb_spec_o_smooth)) then
                !$acc exit data delete(s%agb_spec_o_smooth)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agb_spec_o_smooth)
#endif
            end if
            if (associated(s%agb_spec_c_smooth)) then
                !$acc exit data delete(s%agb_spec_c_smooth)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agb_spec_c_smooth)
#endif
            end if
            if (associated(s%agb_spec_car_smooth)) then
                !$acc exit data delete(s%agb_spec_car_smooth)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agb_spec_car_smooth)
#endif
            end if
            if (associated(s%agb_logt_c)) then
                !$acc exit data delete(s%agb_logt_c)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agb_logt_c)
#endif
            end if
            if (associated(s%agb_spec_c)) then
                !$acc exit data delete(s%agb_spec_c)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agb_spec_c)
#endif
            end if
            if (associated(s%agb_logt_o)) then
                !$acc exit data delete(s%agb_logt_o)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agb_logt_o)
#endif
            end if
            if (associated(s%agb_spec_o)) then
                !$acc exit data delete(s%agb_spec_o)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%agb_spec_o)
#endif
            end if
            if (associated(s%wmb_spec)) then
                !$acc exit data delete(s%wmb_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%wmb_spec)
#endif
            end if
            if (associated(s%speclib)) then
                !$acc exit data delete(s%speclib)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%speclib)
#endif
            end if
            if (associated(s%spec_res)) then
                !$acc exit data delete(s%spec_res)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%spec_res)
#endif
            end if
            if (associated(s%spec_nu)) then
                !$acc exit data delete(s%spec_nu)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%spec_nu)
#endif
            end if
            if (associated(s%spec_lambda)) then
                !$acc exit data delete(s%spec_lambda)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%spec_lambda)
#endif
            end if
            if (associated(s%sun_spec)) then
                !$acc exit data delete(s%sun_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sun_spec)
#endif
            end if
            if (associated(s%vega_spec)) then
                !$acc exit data delete(s%vega_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%vega_spec)
#endif
            end if
            if (associated(s%filter_leff)) then
                !$acc exit data delete(s%filter_leff)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%filter_leff)
#endif
            end if
            if (associated(s%magvega)) then
                !$acc exit data delete(s%magvega)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%magvega)
#endif
            end if
            if (associated(s%magsun)) then
                !$acc exit data delete(s%magsun)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%magsun)
#endif
            end if
            if (associated(s%bands)) then
                !$acc exit data delete(s%bands)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%bands)
#endif
            end if
            if (associated(s%g03smcextn)) then
                !$acc exit data delete(s%g03smcextn)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%g03smcextn)
#endif
            end if
            if (associated(s%wgdust)) then
                !$acc exit data delete(s%wgdust)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%wgdust)
#endif
            end if
            if (associated(s%indexdefined)) then
                !$acc exit data delete(s%indexdefined)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%indexdefined)
#endif
            end if

            ! --- Permanent CSP Workspace ---
            if (allocated(s%ssp_basis_spec)) then
                !$acc exit data delete(s%ssp_basis_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%ssp_basis_spec)
#endif
            end if
            if (allocated(s%ssp_basis_mass)) then
                !$acc exit data delete(s%ssp_basis_mass)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%ssp_basis_mass)
#endif
            end if
            if (allocated(s%ssp_basis_lbol)) then
                !$acc exit data delete(s%ssp_basis_lbol)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%ssp_basis_lbol)
#endif
            end if
            if (allocated(s%csp_ssp_grid)) then
                !$acc exit data delete(s%csp_ssp_grid)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_ssp_grid)
#endif
            end if
            if (allocated(s%csp_emlin_grid)) then
                !$acc exit data delete(s%csp_emlin_grid)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_emlin_grid)
#endif
            end if
            if (allocated(s%csp_ssp_lum_linear)) then
                !$acc exit data delete(s%csp_ssp_lum_linear)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_ssp_lum_linear)
#endif
            end if
            if (allocated(s%csp_igm_transmission)) then
                !$acc exit data delete(s%csp_igm_transmission)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_igm_transmission)
#endif
            end if
            if (allocated(s%csp_spec_final)) then
                !$acc exit data delete(s%csp_spec_final)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_spec_final)
#endif
            end if
            if (allocated(s%csp_emlin_final)) then
                !$acc exit data delete(s%csp_emlin_final)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_emlin_final)
#endif
            end if
            if (allocated(s%csp_weights)) then
                !$acc exit data delete(s%csp_weights)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_weights)
#endif
            end if
            if (allocated(s%csp_emlin_young)) then
                !$acc exit data delete(s%csp_emlin_young)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_emlin_young)
#endif
            end if
            if (allocated(s%csp_emlin_old)) then
                !$acc exit data delete(s%csp_emlin_old)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%csp_emlin_old)
#endif
            end if

            ! --- Persistent Physics Workspaces ---
            if (allocated(s%dust_transmission_diffuse)) then
                !$acc exit data delete(s%dust_transmission_diffuse)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%dust_transmission_diffuse)
#endif
            end if
            if (allocated(s%dust_frequencies)) then
                !$acc exit data delete(s%dust_frequencies)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%dust_frequencies)
#endif
            end if
            if (allocated(s%dust_spec_total_work)) then
                !$acc exit data delete(s%dust_spec_total_work)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%dust_spec_total_work)
#endif
            end if
            if (allocated(s%dust_emission_shape)) then
                !$acc exit data delete(s%dust_emission_shape)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%dust_emission_shape)
#endif
            end if
            if (allocated(s%dust_emission_final)) then
                !$acc exit data delete(s%dust_emission_final)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%dust_emission_final)
#endif
            end if
            if (allocated(s%gas_neb_cont_reduced)) then
                !$acc exit data delete(s%gas_neb_cont_reduced)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%gas_neb_cont_reduced)
#endif
            end if
            if (allocated(s%gas_neb_line_reduced)) then
                !$acc exit data delete(s%gas_neb_line_reduced)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%gas_neb_line_reduced)
#endif
            end if
            if (allocated(s%gas_current_step_cont)) then
                !$acc exit data delete(s%gas_current_step_cont)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%gas_current_step_cont)
#endif
            end if
            if (allocated(s%gas_current_step_lines)) then
                !$acc exit data delete(s%gas_current_step_lines)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%gas_current_step_lines)
#endif
            end if
            if (allocated(s%scalar_reductions)) then
                !$acc exit data delete(s%scalar_reductions)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%scalar_reductions)
#endif
            end if
            if (allocated(s%sfh_t_calc)) then
                !$acc exit data delete(s%sfh_t_calc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sfh_t_calc)
#endif
            end if
            if (allocated(s%sfh_sfr_calc)) then
                !$acc exit data delete(s%sfh_sfr_calc)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sfh_sfr_calc)
#endif
            end if
            if (allocated(s%sfh_age_integrand)) then
                !$acc exit data delete(s%sfh_age_integrand)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sfh_age_integrand)
#endif
            end if
            if (allocated(s%sfh_w_tmp1)) then
                !$acc exit data delete(s%sfh_w_tmp1)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sfh_w_tmp1)
#endif
            end if
            if (allocated(s%sfh_w_tmp2)) then
                !$acc exit data delete(s%sfh_w_tmp2)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%sfh_w_tmp2)
#endif
            end if

            ! --- Fused Loop Output Buffers ---
            if (allocated(s%out_csp_spec)) then
                !$acc exit data delete(s%out_csp_spec)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%out_csp_spec)
#endif
            end if
            if (allocated(s%out_csp_emlin)) then
                !$acc exit data delete(s%out_csp_emlin)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%out_csp_emlin)
#endif
            end if
            if (allocated(s%out_mass_csp)) then
                !$acc exit data delete(s%out_mass_csp)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%out_mass_csp)
#endif
            end if
            if (allocated(s%out_lbol_csp)) then
                !$acc exit data delete(s%out_lbol_csp)
#ifdef OMP_TARGET
                !$omp target exit data map(delete: s%out_lbol_csp)
#endif
            end if
        end associate

        if (allocated(ctx%pset%ssp_gen_age)) then
            !$acc exit data delete(ctx%pset%ssp_gen_age)
#ifdef OMP_TARGET
            !$omp target exit data map(delete: ctx%pset%ssp_gen_age)
#endif
        end if
        if (allocated(ctx%pset%mag_compute)) then
            !$acc exit data delete(ctx%pset%mag_compute)
#ifdef OMP_TARGET
            !$omp target exit data map(delete: ctx%pset%mag_compute)
#endif
        end if

        call batch_grid_release(ctx%state%batch)
        call batch_out_release(ctx%state%batch_out, .true.)
        !$acc exit data delete(ctx%state)
#ifdef OMP_TARGET
        !$omp target exit data map(delete: ctx%state)
#endif
        !$acc exit data delete(ctx)
#ifdef OMP_TARGET
        !$omp target exit data map(delete: ctx)
#endif
        ctx%on_device = .false.
    end subroutine fsps_context_remove_from_device

end module fsps_context
