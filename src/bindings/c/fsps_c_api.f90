module fsps_c_api
    !> @brief
    !> C/Fortran interoperability layer for FSPS.
    !>
    !> @details
    !> Exposes stable C bindings for driving FSPS from C/C++ and other languages.
    !> This module owns the legacy global state and handle-based context pool.

    use iso_c_binding
    use fsps_precision, only: WP, KP
    use fsps_constants, only: NEMLINE, NM, BHB_SBS_TIME, GRAVITY_L_M_T_COEFF
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy, &
                        fsps_set_param_int, fsps_set_param_float, fsps_set_param_str, &
                        fsps_set_fast_mode, fsps_get_paths, fsps_prepare_pset, &
                        api_compute_ssp => fsps_compute_ssp, api_compute_csp => fsps_compute_csp
    use fsps_csp, only: compute_csp_scenario
    use fsps_io, only: load_tabular_sfh, write_isochrone_cmd
    use fsps_ssp, only: generate_ssp_grid, compute_interpolated_ssp
    use fsps_spectral_library, only: get_stellar_spectrum
    use fsps_stellar_modifications, only: apply_blue_stragglers, modify_giant_branch, &
                                          modify_horizontal_branch
    use fsps_imf, only: compute_imf_weights
    use fsps_smoothing, only: apply_smoothing
    use fsps_cosmology, only: vacuum_to_air
    use fsps_interpolation, only: find_interval
    use fsps_spectral_indices, only: compute_spectral_indices
    use fsps_photometry, only: compute_magnitudes
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            is_ssp_sensitive_float_param, is_ssp_sensitive_int_param, &
                            is_ssp_sensitive_str_param, fsps_context_prepare_csp_workspace
    use fsps_csp_batch, only: compute_csp_batch, batch_spectra_device, batch_spectra_host
    use fsps_context_types, only: fsps_context_t

    implicit none

    ! ---------------------------------------------------------------------
    ! Module constants
    ! ---------------------------------------------------------------------

    integer, parameter :: fsps_driver_version_major = 1
    integer, parameter :: fsps_driver_version_minor = 2   ! = FSPS_API_VERSION_MINOR in fsps.h
    integer, parameter :: fsps_driver_version_patch = 0

    ! ---------------------------------------------------------------------
    ! Global state
    ! ---------------------------------------------------------------------
    type(PARAMS), pointer :: global_pset => NULL()
    type(COMPSPOUT), pointer :: global_ocompsp(:) => NULL()
    integer, allocatable :: has_ssp(:)
    integer, allocatable :: has_ssp_age(:, :)
    ! Context pool for handle-based API
    type(fsps_context_t), allocatable :: ctx_pool(:)
    logical, allocatable :: ctx_inuse(:)
    !> Per handle, for fsps_context_compute_csp: the SSP grids computed so far (one per
    !> metallicity, kept until an SSP parameter changes) and the last single CSP's outputs
    type :: ctx_store_t
        real(WP), allocatable :: spec(:, :, :), mass(:, :), lbol(:, :)
        logical, allocatable :: have(:)
        type(COMPSPOUT), allocatable :: results(:)
    end type ctx_store_t
    type(ctx_store_t), allocatable :: ctx_store(:)
    ! Driver error state
    integer :: fsps_last_status = 0
    character(LEN=256) :: fsps_last_error = ''
    integer :: fsps_debug = 0
    integer :: fsps_lock_state = 0
    type(fsps_context_t), save :: fsps_default_ctx
    logical :: fsps_default_ctx_ready = .false.
contains

    subroutine fsps_ensure_default_ctx()
        if (.not. fsps_default_ctx_ready) then
            call fsps_create(fsps_default_ctx)
            fsps_default_ctx_ready = .true.
        end if
    end subroutine fsps_ensure_default_ctx

    subroutine fsps_context_alloc_slot(slot)
        integer, intent(OUT) :: slot
        integer :: i, n
        type(fsps_context_t), allocatable :: new_pool(:)
        logical, allocatable :: new_inuse(:)
        type(ctx_store_t), allocatable :: new_store(:)

        if (.not. ALLOCATED(ctx_pool)) then
            allocate (ctx_pool(1))
            allocate (ctx_inuse(1))
            allocate (ctx_store(1))
            ctx_inuse = .false.
        end if

        slot = 0
        do i = 1, SIZE(ctx_pool)
            if (.not. ctx_inuse(i)) then
                slot = i
                exit
            end if
        end do

        if (slot == 0) then
            n = SIZE(ctx_pool)
            allocate (new_pool(n + 1))
            allocate (new_inuse(n + 1))
            new_pool(1:n) = ctx_pool
            new_inuse(1:n) = ctx_inuse
            new_inuse(n + 1) = .false.
            call MOVE_ALLOC(new_pool, ctx_pool)
            call MOVE_ALLOC(new_inuse, ctx_inuse)
            allocate (new_store(n + 1))
            do i = 1, n
                call move_store(ctx_store(i), new_store(i))
            end do
            call MOVE_ALLOC(new_store, ctx_store)
            slot = n + 1
        end if
    end subroutine fsps_context_alloc_slot

    subroutine fsps_ensure_legacy_state()
        integer :: i
        integer :: n_bands, n_t, n_tfull, n_spec, n_indx, n_z

        call fsps_ensure_default_ctx()
        n_bands = fsps_default_ctx%state%nbands
        n_t = fsps_default_ctx%state%nt
        n_tfull = fsps_default_ctx%state%ntfull
        n_spec = fsps_default_ctx%state%nspec
        n_indx = fsps_default_ctx%state%nindx
        n_z = fsps_default_ctx%state%nz

        if (.not. ASSOCIATED(global_pset)) then
            allocate (global_pset)
        end if

        if (.not. ALLOCATED(global_pset%mag_compute)) then
            allocate (global_pset%mag_compute(n_bands))
            global_pset%mag_compute = 1
        else if (SIZE(global_pset%mag_compute) /= n_bands) then
            deallocate (global_pset%mag_compute)
            allocate (global_pset%mag_compute(n_bands))
            global_pset%mag_compute = 1
        end if

        if (.not. ALLOCATED(global_pset%ssp_gen_age)) then
            allocate (global_pset%ssp_gen_age(n_t))
            global_pset%ssp_gen_age = 1
        else if (SIZE(global_pset%ssp_gen_age) /= n_t) then
            deallocate (global_pset%ssp_gen_age)
            allocate (global_pset%ssp_gen_age(n_t))
            global_pset%ssp_gen_age = 1
        end if

        if (ASSOCIATED(global_ocompsp)) then
            if (SIZE(global_ocompsp) /= n_tfull) then
                do i = 1, SIZE(global_ocompsp)
                    if (ALLOCATED(global_ocompsp(i)%mags)) deallocate (global_ocompsp(i)%mags)
                    if (ALLOCATED(global_ocompsp(i)%spec)) deallocate (global_ocompsp(i)%spec)
                    if (ALLOCATED(global_ocompsp(i)%indx)) deallocate (global_ocompsp(i)%indx)
                    if (ALLOCATED(global_ocompsp(i)%emlines)) deallocate (global_ocompsp(i)%emlines)
                end do
                deallocate (global_ocompsp)
            end if
        end if

        if (.not. ASSOCIATED(global_ocompsp)) then
            allocate (global_ocompsp(n_tfull))
            do i = 1, n_tfull
                allocate (global_ocompsp(i)%mags(n_bands))
                allocate (global_ocompsp(i)%spec(n_spec))
                allocate (global_ocompsp(i)%indx(n_indx))
                allocate (global_ocompsp(i)%emlines(NEMLINE))
            end do
        end if

        if (.not. ALLOCATED(has_ssp)) allocate (has_ssp(n_z))
        if (.not. ALLOCATED(has_ssp_age)) allocate (has_ssp_age(n_z, n_t))
        has_ssp = 0
        has_ssp_age = 0
    end subroutine fsps_ensure_legacy_state

    subroutine fsps_copy_pset_from_ctx(ctx)
        type(fsps_context_t), intent(IN) :: ctx
        global_pset = ctx%pset
    end subroutine fsps_copy_pset_from_ctx

    ! Record a driver error or warning.
    subroutine fsps_set_error(status, message)
        integer, intent(IN) :: status
        character(LEN=*), intent(IN) :: message
        fsps_last_status = status
        fsps_last_error = message
        if (fsps_debug /= 0) then
            write (*, *) TRIM(message)
        end if
    end subroutine fsps_set_error

    ! Copy CSP results into global_ocompsp and (optionally) an output buffer.
    subroutine fsps_store_results(results, f_spec)
        type(COMPSPOUT), intent(IN) :: results(:)
        real(WP), intent(INOUT), optional :: f_spec(:, :)
        integer :: i, n_out, n_spec, n_time

        n_out = SIZE(results)
        n_spec = SIZE(results(1)%spec)

        if (PRESENT(f_spec)) then
            n_time = SIZE(f_spec, 2)
            f_spec = 0.0_wp
        else
            n_time = n_out
        end if

        do i = 1, n_out
            global_ocompsp(i)%age = results(i)%age
            global_ocompsp(i)%mass_csp = results(i)%mass_csp
            global_ocompsp(i)%lbol_csp = results(i)%lbol_csp
            global_ocompsp(i)%sfr = results(i)%sfr
            global_ocompsp(i)%mdust = results(i)%mdust
            global_ocompsp(i)%mformed = results(i)%mformed
            global_ocompsp(i)%spec = results(i)%spec
            global_ocompsp(i)%emlines = results(i)%emlines

            if (allocated(results(i)%mags)) then
                global_ocompsp(i)%mags = results(i)%mags
            else
                if (allocated(global_ocompsp(i)%mags)) deallocate (global_ocompsp(i)%mags)
            end if

            if (allocated(results(i)%indx)) then
                global_ocompsp(i)%indx = results(i)%indx
            else
                if (allocated(global_ocompsp(i)%indx)) deallocate (global_ocompsp(i)%indx)
            end if

            if (PRESENT(f_spec)) then
                if (i <= n_time .and. n_spec == SIZE(f_spec, 1)) then
                    f_spec(:, i) = results(i)%spec
                end if
            end if
        end do
    end subroutine fsps_store_results

    ! Clear error state.
    subroutine fsps_clear_error() bind(C, name="fsps_clear_error")
        fsps_last_status = 0
        fsps_last_error = ''
    end subroutine fsps_clear_error

    ! Retrieve the last error message into a C buffer.
    subroutine fsps_get_last_error(status, c_msg, c_len) &
        bind(C, name="fsps_get_last_error")
        integer(c_int), intent(OUT) :: status
        character(KIND=c_char), dimension(*), intent(OUT) :: c_msg
        integer(c_int), value :: c_len

        status = fsps_last_status
        call f_to_c_string(fsps_last_error, c_msg, c_len)
    end subroutine fsps_get_last_error

    ! Enable or disable driver debug prints.
    subroutine fsps_set_debug(flag) bind(C, name="fsps_set_debug")
        integer(c_int), value :: flag
        fsps_debug = flag
    end subroutine fsps_set_debug

    subroutine fsps_lock(status) bind(C, name="fsps_lock")
        integer(c_int), intent(OUT) :: status
        if (fsps_lock_state /= 0) then
            status = 1
            call fsps_set_error(401, "[FSPS-C] Error: fsps_lock already held")
            return
        end if
        fsps_lock_state = 1
        status = 0
    end subroutine fsps_lock

    subroutine fsps_unlock(status) bind(C, name="fsps_unlock")
        integer(c_int), intent(OUT) :: status
        if (fsps_lock_state == 0) then
            status = 1
            call fsps_set_error(402, "[FSPS-C] Error: fsps_unlock without lock")
            return
        end if
        fsps_lock_state = 0
        status = 0
    end subroutine fsps_unlock

    subroutine fsps_get_driver_version(major, minor, patch) &
        bind(C, name="fsps_get_driver_version")
        integer(c_int), intent(OUT) :: major, minor, patch
        major = fsps_driver_version_major
        minor = fsps_driver_version_minor
        patch = fsps_driver_version_patch
    end subroutine fsps_get_driver_version

    ! -------------------------------------------------------------------------
    ! CONTEXT-BASED C API
    ! -------------------------------------------------------------------------
    subroutine fsps_context_create_handle(handle, status) &
        bind(C, name="fsps_context_create")
        integer(c_int), intent(OUT) :: handle
        integer(c_int), intent(OUT) :: status
        integer :: slot
        type(PARAMS) :: default_pset

        call fsps_context_alloc_slot(slot)
        call reset_store(ctx_store(slot))
        ! A reused slot keeps the previous context's parameter set (fsps_create resets only the
        ! context settings): start from the defaults
        ctx_pool(slot)%pset = default_pset
        call fsps_create(ctx_pool(slot))
        ctx_inuse(slot) = .true.
        handle = slot
        status = 0
    end subroutine fsps_context_create_handle

    subroutine fsps_context_destroy_handle(handle, status) &
        bind(C, name="fsps_context_destroy")
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: status

        status = 0
        if (.not. ALLOCATED(ctx_pool)) then
            status = 1
            return
        end if
        if (handle < 1 .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        call fsps_destroy(ctx_pool(handle))
        call reset_store(ctx_store(handle))
        ctx_inuse(handle) = .false.
    end subroutine fsps_context_destroy_handle

    subroutine fsps_context_setup_handle(zin, c_isoc, c_spec, c_dust, handle, status) &
        bind(C, name="fsps_context_setup")
        integer(c_int), value :: zin
        character(KIND=c_char), dimension(*), intent(IN) :: c_isoc
        character(KIND=c_char), dimension(*), intent(IN) :: c_spec
        character(KIND=c_char), dimension(*), intent(IN) :: c_dust
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: status
        character(LEN=64) :: isoc_type_in
        character(LEN=64) :: spec_type_in
        character(LEN=64) :: dust_type_in

        status = 0
        if (.not. ALLOCATED(ctx_pool)) then
            status = 1
            return
        end if
        if (handle < 1 .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        call c_to_f_string(c_isoc, isoc_type_in)
        call c_to_f_string(c_spec, spec_type_in)
        call c_to_f_string(c_dust, dust_type_in)
        call reset_store(ctx_store(handle))

        if (LEN_TRIM(isoc_type_in) == 0 .and. LEN_TRIM(spec_type_in) == 0 .and. &
            LEN_TRIM(dust_type_in) == 0) then
            call fsps_setup(ctx_pool(handle), zin)
        else if (LEN_TRIM(spec_type_in) == 0 .and. LEN_TRIM(dust_type_in) == 0) then
            call fsps_setup(ctx_pool(handle), zin, TRIM(isoc_type_in))
        else if (LEN_TRIM(dust_type_in) == 0) then
            call fsps_setup(ctx_pool(handle), zin, TRIM(isoc_type_in), TRIM(spec_type_in))
        else
            call fsps_setup(ctx_pool(handle), zin, TRIM(isoc_type_in), TRIM(spec_type_in), TRIM(dust_type_in))
        end if
    end subroutine fsps_context_setup_handle

    subroutine fsps_context_set_int_handle(handle, c_key, value, status) &
        bind(C, name="fsps_context_set_int")
        integer(c_int), value :: handle
        character(KIND=c_char), dimension(*), intent(IN) :: c_key
        integer(c_int), value :: value
        integer(c_int), intent(OUT) :: status
        character(LEN=64) :: key

        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        call c_to_f_string(c_key, key)
        call fsps_set_param_int(ctx_pool(handle), TRIM(key), value, status)
        ! zmet only selects among the stored grids
        if (status == 0 .and. TRIM(key) /= 'zmet' .and. is_ssp_sensitive_int_param(TRIM(key))) &
            call invalidate_ssps(ctx_store(handle))
    end subroutine fsps_context_set_int_handle

    subroutine fsps_context_set_float_handle(handle, c_key, value, status) &
        bind(C, name="fsps_context_set_float")
        integer(c_int), value :: handle
        character(KIND=c_char), dimension(*), intent(IN) :: c_key
        real(c_double), value :: value
        integer(c_int), intent(OUT) :: status
        character(LEN=64) :: key

        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        call c_to_f_string(c_key, key)
        call fsps_set_param_float(ctx_pool(handle), TRIM(key), real(value, WP), status)
        ! The nebular parameters change the SSPs only with nebular emission in them
        ! (add_neb_emission = 2), as in fsps_context_compute_csp_batch
        if (status == 0 .and. is_ssp_sensitive_float_param(TRIM(key))) then
            if (ctx_pool(handle)%add_neb_emission_val == 2 .or. .not. &
                any(TRIM(key) == [character(LEN=10) :: 'gas_logu', 'gas_logz', 'frac_obrun'])) &
                call invalidate_ssps(ctx_store(handle))
        end if
    end subroutine fsps_context_set_float_handle

    subroutine fsps_context_set_str_handle(handle, c_key, c_val, status) &
        bind(C, name="fsps_context_set_str")
        integer(c_int), value :: handle
        character(KIND=c_char), dimension(*), intent(IN) :: c_key
        character(KIND=c_char), dimension(*), intent(IN) :: c_val
        integer(c_int), intent(OUT) :: status
        character(LEN=64) :: key
        character(LEN=128) :: val

        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        call c_to_f_string(c_key, key)
        call c_to_f_string(c_val, val)
        call fsps_set_param_str(ctx_pool(handle), TRIM(key), TRIM(val), status)
        if (status == 0 .and. is_ssp_sensitive_str_param(TRIM(key))) call invalidate_ssps(ctx_store(handle))
    end subroutine fsps_context_set_str_handle

    subroutine fsps_context_set_fast_mode_handle(handle, fast_mode, status) &
        bind(C, name="fsps_context_set_fast_mode")
        integer(c_int), value :: handle
        integer(c_int), value :: fast_mode
        integer(c_int), intent(OUT) :: status

        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        call fsps_set_fast_mode(ctx_pool(handle), fast_mode /= 0)
    end subroutine fsps_context_set_fast_mode_handle

    !> @brief Batched CSPs from parameter arrays (fitting pipelines).
    !>
    !> @details
    !> n models: the context's current parameters, with n_params of them varied per model,
    !> names(i) (C strings) taking the n values values(i)(1:n) (one array per parameter). The
    !> parameters must be per-model ones of compute_csp_batch (SFH, dust, dust emission, nebular,
    !> AGN, logzsol, zred, igm_factor, ...; 'sfh' as an integer value): parameters that change the
    !> SSPs (IMF, isochrone options, ...) give status 2, unknown names status 3. ssp_weights
    !> (ntfull x n, column-major, or NULL) replaces the SFH parameters. The SSP basis is computed
    !> on the first call and when an SSP parameter changes; if 'logzsol' is among the names, the
    !> bases of all metallicities (per-model interpolation in log Z). Outputs are column-major
    !> (quantity x n); any but mags may be NULL: mags (nbands), spec (nspec), mass, lbol, mdust,
    !> sfr (1), emlines (nemline), indx (nindx). status: as compute_csp_batch, plus 1 for a bad
    !> handle, 2 / 3 above.
    subroutine fsps_context_compute_csp_batch_handle(handle, n, n_params, c_names, c_values, c_weights, &
                                                     c_mags, c_spec, c_mass, c_lbol, c_mdust, c_sfr, &
                                                     c_emlines, c_indx, status) &
        bind(C, name="fsps_context_compute_csp_batch")
        integer(c_int), value :: handle, n, n_params
        type(c_ptr), value :: c_names, c_values, c_weights
        type(c_ptr), value :: c_mags, c_spec, c_mass, c_lbol, c_mdust, c_sfr, c_emlines, c_indx
        integer(c_int), intent(OUT) :: status

        call batch_handle(handle, n, n_params, c_names, c_values, c_weights, c_mags, c_spec, c_mass, c_lbol, &
                          c_mdust, c_sfr, c_emlines, c_indx, c_null_ptr, c_null_ptr, c_null_ptr, c_null_ptr, &
                          c_null_ptr, c_null_ptr, status)
    end subroutine fsps_context_compute_csp_batch_handle

    !> @brief The batch with its intrinsic young and old components as well (emulator training
    !> sets: birth-cloud and diffuse dust applied afterwards).
    !>
    !> @details
    !> As fsps_context_compute_csp_batch, plus young and old (nspec x n, required): the intrinsic
    !> spectra of the ages <= dust_tesc and > dust_tesc, nebular emission included in them (it
    !> belongs to the young ages), before dust, dust emission, AGN, IGM and smoothing; and the
    !> surviving stellar masses and log10 L_bol of the two parts (any of the four may be NULL).
    subroutine fsps_context_compute_csp_batch_parts_handle(handle, n, n_params, c_names, c_values, c_weights, &
                                                           c_mags, c_spec, c_mass, c_lbol, c_mdust, c_sfr, &
                                                           c_emlines, c_indx, c_young, c_old, c_mass_young, &
                                                           c_mass_old, c_lbol_young, c_lbol_old, status) &
        bind(C, name="fsps_context_compute_csp_batch_parts")
        integer(c_int), value :: handle, n, n_params
        type(c_ptr), value :: c_names, c_values, c_weights
        type(c_ptr), value :: c_mags, c_spec, c_mass, c_lbol, c_mdust, c_sfr, c_emlines, c_indx
        type(c_ptr), value :: c_young, c_old, c_mass_young, c_mass_old, c_lbol_young, c_lbol_old
        integer(c_int), intent(OUT) :: status

        if (.not. c_associated(c_young) .or. .not. c_associated(c_old)) then
            status = 3
            call fsps_set_error(3, 'compute_csp_batch_parts: young and old are required')
            return
        end if
        call batch_handle(handle, n, n_params, c_names, c_values, c_weights, c_mags, c_spec, c_mass, c_lbol, &
                          c_mdust, c_sfr, c_emlines, c_indx, c_young, c_old, c_mass_young, c_mass_old, &
                          c_lbol_young, c_lbol_old, status)
    end subroutine fsps_context_compute_csp_batch_parts_handle

    !> The two batch entry points (parts requested when c_young is associated).
    subroutine batch_handle(handle, n, n_params, c_names, c_values, c_weights, c_mags, c_spec, c_mass, c_lbol, &
                            c_mdust, c_sfr, c_emlines, c_indx, c_young, c_old, c_mass_young, c_mass_old, &
                            c_lbol_young, c_lbol_old, status)
        integer(c_int), intent(IN) :: handle, n, n_params
        type(c_ptr), intent(IN) :: c_names, c_values, c_weights
        type(c_ptr), intent(IN) :: c_mags, c_spec, c_mass, c_lbol, c_mdust, c_sfr, c_emlines, c_indx
        type(c_ptr), intent(IN) :: c_young, c_old, c_mass_young, c_mass_old, c_lbol_young, c_lbol_old
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: young(:, :), old(:, :), pv(:)
        real(WP), allocatable, target :: spec_l(:, :)
        real(WP), allocatable :: my_l(:), mo_l(:), ly_l(:), lo_l(:)
        type(c_ptr), pointer :: name_ptrs(:), value_ptrs(:)
        real(c_double), pointer :: v(:)
        real(c_double), pointer :: mags(:, :), spec(:, :), emlines(:, :), indx(:, :), weights(:, :)
        real(c_double), pointer :: mass(:), lbol(:), mdust(:), sfr(:)
        character(LEN=64), allocatable :: names(:)
        type(PARAMS) :: base
        type(PARAMS), allocatable :: psets(:)
        integer :: i, m, st, nspec, nt, nbands
        logical :: zmode, dirty
        real(WP), allocatable :: mass_l(:), lbol_l(:), mdust_l(:), sfr_l(:), eml_l(:, :)
        character(LEN=10), parameter :: NEB_PARAMS(3) = [character(LEN=10) :: 'gas_logu', 'gas_logz', 'frac_obrun']

        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if
        if (n <= 0) return
        associate(ctx => ctx_pool(handle))
        nspec = ctx%state%nspec
        nt = ctx%state%ntfull
        nbands = ctx%state%nbands

        ! Parameter names; those that change the SSPs cannot vary per model
        allocate(names(max(0, n_params)))
        zmode = .false.
        if (n_params > 0) then
            call c_f_pointer(c_names, name_ptrs, [n_params])
            call c_f_pointer(c_values, value_ptrs, [n_params])
            do i = 1, n_params
                call c_ptr_to_f_string(name_ptrs(i), names(i))
                ! The nebular parameters are SSP parameters only with nebular emission in the SSPs
                ! (add_neb_emission = 2); otherwise they act in the CSP and may vary per model
                if (is_ssp_sensitive_float_param(trim(names(i))) .and. .not. &
                    (ctx%add_neb_emission_val /= 2 .and. any(trim(names(i)) == NEB_PARAMS))) then
                    status = 2
                    call fsps_set_error(2, 'compute_csp_batch: '//trim(names(i))//' changes the SSPs')
                    return
                end if
                if (trim(names(i)) == 'logzsol') zmode = .true.
            end do
        end if

        ! Per-model parameter sets: the context's, with the named parameters set
        call fsps_prepare_pset(ctx)
        base = ctx%pset
        ! The setters mark the SSP basis dirty for the nebular parameters; per-model values in a
        ! batch do not change it
        dirty = ctx%state%ssp_basis_is_dirty
        allocate(psets(n))
        do m = 1, n
            ctx%pset = base
            do i = 1, n_params
                call c_f_pointer(value_ptrs(i), v, [n])
                call fsps_set_param_float(ctx, trim(names(i)), real(v(m), WP), st)
                if (st /= 0 .and. trim(names(i)) == 'sfh') then
                    call fsps_set_param_int(ctx, 'sfh', nint(v(m)), st)
                end if
                if (st /= 0) then
                    ctx%pset = base
                    ctx%state%ssp_basis_is_dirty = dirty
                    status = 3
                    call fsps_set_error(3, 'compute_csp_batch: unknown parameter '//trim(names(i)))
                    return
                end if
            end do
            psets(m) = ctx%pset
        end do
        ctx%pset = base
        ctx%state%ssp_basis_is_dirty = dirty

        call ensure_batch_basis(ctx, zmode)

        call c_f_pointer(c_mags, mags, [nbands, n])
        if (c_associated(c_spec)) call c_f_pointer(c_spec, spec, [nspec, n])
        if (c_associated(c_indx)) call c_f_pointer(c_indx, indx, [ctx%state%nindx, n])
        if (c_associated(c_weights)) call c_f_pointer(c_weights, weights, [nt, n])
        ! Small outputs always computed; spectra, indices and weights passed only when given.
        ! (Not by passing disassociated pointers as absent arguments: with -acc=gpu, nvfortran
        ! 25.1 reports some of them as present.)
        allocate(mass_l(n), lbol_l(n), mdust_l(n), sfr_l(n), eml_l(NEMLINE, n))
        if (c_associated(c_young)) then
            ! With the young / old parts: the spectra always (the caller's array or a local one)
            call c_f_pointer(c_young, young, [nspec, n])
            call c_f_pointer(c_old, old, [nspec, n])
            allocate(my_l(n), mo_l(n), ly_l(n), lo_l(n))
            if (.not. c_associated(c_spec)) then
                allocate(spec_l(nspec, n))
                spec => spec_l
            end if
            if (c_associated(c_indx)) then
                if (c_associated(c_weights)) then
                    call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, indx=indx, ssp_weights=weights, &
                                           spec_young=young, spec_old=old, mass_young=my_l, mass_old=mo_l, &
                                           lbol_young=ly_l, lbol_old=lo_l)
                else
                    call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, indx=indx, &
                                           spec_young=young, spec_old=old, mass_young=my_l, mass_old=mo_l, &
                                           lbol_young=ly_l, lbol_old=lo_l)
                end if
            else
                if (c_associated(c_weights)) then
                    call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, ssp_weights=weights, &
                                           spec_young=young, spec_old=old, mass_young=my_l, mass_old=mo_l, &
                                           lbol_young=ly_l, lbol_old=lo_l)
                else
                    call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, &
                                           spec_young=young, spec_old=old, mass_young=my_l, mass_old=mo_l, &
                                           lbol_young=ly_l, lbol_old=lo_l)
                end if
            end if
            if (st == 0) then
                if (c_associated(c_mass_young)) then
                    call c_f_pointer(c_mass_young, pv, [n]); pv = my_l
                end if
                if (c_associated(c_mass_old)) then
                    call c_f_pointer(c_mass_old, pv, [n]); pv = mo_l
                end if
                if (c_associated(c_lbol_young)) then
                    call c_f_pointer(c_lbol_young, pv, [n]); pv = ly_l
                end if
                if (c_associated(c_lbol_old)) then
                    call c_f_pointer(c_lbol_old, pv, [n]); pv = lo_l
                end if
            end if
        else if (c_associated(c_spec)) then
            if (c_associated(c_indx)) then
                if (c_associated(c_weights)) then
                    call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, indx=indx, ssp_weights=weights)
                else
                    call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, indx=indx)
                end if
            else
                if (c_associated(c_weights)) then
                    call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, ssp_weights=weights)
                else
                    call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l)
                end if
            end if
        else
            if (c_associated(c_indx)) then
                if (c_associated(c_weights)) then
                    call compute_csp_batch(ctx, psets, mags, st, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, indx=indx, ssp_weights=weights)
                else
                    call compute_csp_batch(ctx, psets, mags, st, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, indx=indx)
                end if
            else
                if (c_associated(c_weights)) then
                    call compute_csp_batch(ctx, psets, mags, st, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l, ssp_weights=weights)
                else
                    call compute_csp_batch(ctx, psets, mags, st, mass=mass_l, lbol=lbol_l, mdust=mdust_l, &
                                           sfr=sfr_l, emlines=eml_l)
                end if
            end if
        end if
        if (st == 0) then
            if (c_associated(c_mass)) then
                call c_f_pointer(c_mass, mass, [n]); mass = mass_l
            end if
            if (c_associated(c_lbol)) then
                call c_f_pointer(c_lbol, lbol, [n]); lbol = lbol_l
            end if
            if (c_associated(c_mdust)) then
                call c_f_pointer(c_mdust, mdust, [n]); mdust = mdust_l
            end if
            if (c_associated(c_sfr)) then
                call c_f_pointer(c_sfr, sfr, [n]); sfr = sfr_l
            end if
            if (c_associated(c_emlines)) then
                call c_f_pointer(c_emlines, emlines, [NEMLINE, n]); emlines = eml_l
            end if
        end if
        status = st
        end associate
    end subroutine batch_handle

    !> Where the last batch's spectra are (see batch_spectra_device): the device address, nspec,
    !> the number of models and the bytes per value (4 or 8); status 1 for a bad handle.
    subroutine fsps_context_batch_spec_device_handle(handle, ptr, nspec, n, bytes, status) &
        bind(C, name="fsps_context_batch_spec_device")
        integer(c_int), value :: handle
        type(c_ptr), intent(OUT) :: ptr
        integer(c_int), intent(OUT) :: nspec, n, bytes, status
        integer :: ns, nm, nb

        ptr = c_null_ptr
        nspec = 0; n = 0; bytes = 0
        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if
        call batch_spectra_device(ctx_pool(handle), ptr, ns, nm, nb)
        nspec = ns; n = nm; bytes = nb
    end subroutine fsps_context_batch_spec_device_handle

    !> The last batch's spectra on the host at kernel precision (see batch_spectra_host): the
    !> page-locked host buffer's address, nspec, the number of models and the bytes per value;
    !> status 1 for a bad handle.
    subroutine fsps_context_batch_spec_host_handle(handle, ptr, nspec, n, bytes, status) &
        bind(C, name="fsps_context_batch_spec_host")
        integer(c_int), value :: handle
        type(c_ptr), intent(OUT) :: ptr
        integer(c_int), intent(OUT) :: nspec, n, bytes, status
        integer :: ns, nm, nb

        ptr = c_null_ptr
        nspec = 0; n = 0; bytes = 0
        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if
        call batch_spectra_host(ctx_pool(handle), ptr, ns, nm, nb)
        nspec = ns; n = nm; bytes = nb
    end subroutine fsps_context_batch_spec_host_handle

    !> The context's SSP basis for compute_csp_batch: on the device in OpenACC builds, for the
    !> current SSP parameters, one metallicity (zmet) or, with zmode, all of them.
    subroutine ensure_batch_basis(ctx, zmode)
        type(fsps_context_t), intent(inout) :: ctx
        logical, intent(in) :: zmode
        type(PARAMS) :: pset
        real(WP), allocatable :: spec(:, :, :), mass(:, :), lbol(:, :)
        integer :: nz, z, zmet

#if defined(_OPENACC) || defined(OMP_TARGET)
        if (.not. ctx%on_device) call fsps_context_move_to_device(ctx)
#endif
        nz = merge(ctx%state%nz, 1, zmode)
        if (allocated(ctx%state%ssp_basis_spec) .and. .not. ctx%state%ssp_basis_is_dirty) then
            if (size(ctx%state%ssp_basis_spec, 3) == nz) return
        end if
        allocate(spec(ctx%state%nspec, ctx%state%ntfull, nz), mass(ctx%state%ntfull, nz), lbol(ctx%state%ntfull, nz))
        zmet = ctx%pset%zmet
        do z = 1, nz
            pset = ctx%pset
            if (zmode) pset%zmet = z
            call generate_ssp_grid(ctx, pset, mass(:, z), lbol(:, z), spec(:, :, z))
        end do
        ctx%pset%zmet = zmet
        ctx%state%ssp_basis_is_dirty = .true.
        call fsps_context_update_ssp_basis(ctx, spec, mass, lbol, nz)
    end subroutine ensure_batch_basis

    !> A C string (char *) as a Fortran string.
    subroutine c_ptr_to_f_string(p, f_str)
        type(c_ptr), intent(IN) :: p
        character(LEN=*), intent(OUT) :: f_str
        character(KIND=c_char), pointer :: chars(:)

        call c_f_pointer(p, chars, [len(f_str) + 1])
        call c_to_f_string(chars, f_str)
    end subroutine c_ptr_to_f_string

    subroutine fsps_context_compute_ssp_handle(handle, c_spec, c_mass, c_lbol, status) &
        bind(C, name="fsps_context_compute_ssp")
        integer(c_int), value :: handle
        type(c_ptr), value :: c_spec
        type(c_ptr), value :: c_mass
        type(c_ptr), value :: c_lbol
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: spec_ptr(:, :)
        real(c_double), pointer :: mass_ptr(:)
        real(c_double), pointer :: lbol_ptr(:)
        real(WP), allocatable :: spec(:, :), mass(:), lbol(:)
        integer :: n_spec, n_time

        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        n_spec = ctx_pool(handle)%state%nspec
        n_time = ctx_pool(handle)%state%ntfull
        call c_f_pointer(c_spec, spec_ptr, [n_spec, n_time])
        call c_f_pointer(c_mass, mass_ptr, [n_time])
        call c_f_pointer(c_lbol, lbol_ptr, [n_time])
        ! Computed in the working precision, returned as double (the C API is double throughout)
        allocate (spec(n_spec, n_time), mass(n_time), lbol(n_time))
        call api_compute_ssp(ctx_pool(handle), mass, lbol, spec)
        spec_ptr = spec
        mass_ptr = mass
        lbol_ptr = lbol
    end subroutine fsps_context_compute_ssp_handle

    subroutine fsps_context_get_paths_handle(handle, c_sps, sps_len, c_data, data_len, c_out, out_len) &
        bind(C, name="fsps_context_get_paths")
        integer(c_int), value :: handle
        character(KIND=c_char), dimension(*), intent(OUT) :: c_sps
        integer(c_int), value :: sps_len
        character(KIND=c_char), dimension(*), intent(OUT) :: c_data
        integer(c_int), value :: data_len
        character(KIND=c_char), dimension(*), intent(OUT) :: c_out
        integer(c_int), value :: out_len
        integer(c_int) :: status
        character(LEN=250) :: sps_path, data_path, out_path

        status = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            call f_to_c_string('', c_sps, sps_len)
            call f_to_c_string('', c_data, data_len)
            call f_to_c_string('', c_out, out_len)
            return
        end if
        if (.not. ctx_inuse(handle)) then
            call f_to_c_string('', c_sps, sps_len)
            call f_to_c_string('', c_data, data_len)
            call f_to_c_string('', c_out, out_len)
            return
        end if

        call fsps_get_paths(ctx_pool(handle), sps_path, data_path, out_path)
        call f_to_c_string(TRIM(sps_path), c_sps, sps_len)
        call f_to_c_string(TRIM(data_path), c_data, data_len)
        call f_to_c_string(TRIM(out_path), c_out, out_len)
    end subroutine fsps_context_get_paths_handle

    subroutine fsps_context_get_dims_handle(handle, n_spec, n_time, status) &
        bind(C, name="fsps_context_get_dims")
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: n_spec, n_time
        integer(c_int), intent(OUT) :: status

        status = 0
        n_spec = 0
        n_time = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        n_spec = ctx_pool(handle)%state%nspec
        n_time = ctx_pool(handle)%state%ntfull
    end subroutine fsps_context_get_dims_handle

    subroutine fsps_context_get_nspec_handle(handle, n_spec, status) &
        bind(C, name="fsps_context_get_nspec")
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: n_spec
        integer(c_int), intent(OUT) :: status

        status = 0
        n_spec = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        n_spec = ctx_pool(handle)%state%nspec
    end subroutine fsps_context_get_nspec_handle

    subroutine fsps_context_get_ntfull_handle(handle, n_time, status) &
        bind(C, name="fsps_context_get_ntfull")
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: n_time
        integer(c_int), intent(OUT) :: status

        status = 0
        n_time = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        n_time = ctx_pool(handle)%state%ntfull
    end subroutine fsps_context_get_ntfull_handle

    subroutine fsps_context_get_nbands_handle(handle, n_bands, status) &
        bind(C, name="fsps_context_get_nbands")
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: n_bands
        integer(c_int), intent(OUT) :: status

        status = 0
        n_bands = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        n_bands = ctx_pool(handle)%state%nbands
    end subroutine fsps_context_get_nbands_handle

    subroutine fsps_context_get_nindx_handle(handle, n_indices, status) &
        bind(C, name="fsps_context_get_nindx")
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: n_indices
        integer(c_int), intent(OUT) :: status

        status = 0
        n_indices = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        n_indices = ctx_pool(handle)%state%nindx
    end subroutine fsps_context_get_nindx_handle

    subroutine fsps_context_get_nz_handle(handle, n_z, status) &
        bind(C, name="fsps_context_get_nz")
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: n_z
        integer(c_int), intent(OUT) :: status

        status = 0
        n_z = 0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        n_z = ctx_pool(handle)%state%nz
    end subroutine fsps_context_get_nz_handle

    subroutine fsps_context_get_nemline_handle(handle, n_line, status) &
        bind(C, name="fsps_context_get_nemline")
        integer(c_int), value :: handle
        integer(c_int), intent(OUT) :: n_line
        integer(c_int), intent(OUT) :: status

        status = 0
        n_line = NEMLINE
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            n_line = 0
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            n_line = 0
            return
        end if
    end subroutine fsps_context_get_nemline_handle

    subroutine fsps_context_get_zsol_handle(handle, z_sol, status) &
        bind(C, name="fsps_context_get_zsol")
        integer(c_int), value :: handle
        real(c_double), intent(OUT) :: z_sol
        integer(c_int), intent(OUT) :: status

        status = 0
        z_sol = 0.0
        if (handle < 1 .or. .not. ALLOCATED(ctx_pool) .or. handle > SIZE(ctx_pool)) then
            status = 1
            return
        end if
        if (.not. ctx_inuse(handle)) then
            status = 1
            return
        end if

        z_sol = ctx_pool(handle)%state%zsol
    end subroutine fsps_context_get_zsol_handle

    ! -------------------------------------------------------------------------
    ! Context grids and filter data (for language bindings)
    ! -------------------------------------------------------------------------

    !> True for a handle of a context in use.
    logical function handle_ok(handle)
        integer(c_int), intent(IN) :: handle
        handle_ok = .false.
        if (.not. ALLOCATED(ctx_pool)) return
        if (handle < 1 .or. handle > SIZE(ctx_pool)) return
        handle_ok = ctx_inuse(handle)
    end function handle_ok

    !> Wavelengths of the spectra [A], nspec values: vacuum, or air with vactoair_flag = 1.
    subroutine fsps_context_get_lambda_handle(handle, c_lambda, status) &
        bind(C, name="fsps_context_get_lambda")
        integer(c_int), value :: handle
        type(c_ptr), value :: c_lambda
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: f(:)

        status = 0
        if (.not. handle_ok(handle)) then
            status = 1
            return
        end if
        associate(ctx => ctx_pool(handle))
        if (.not. ASSOCIATED(ctx%state%spec_lambda)) then
            status = 301
            return
        end if
        call c_f_pointer(c_lambda, f, [ctx%state%nspec])
        if (ctx%vactoair_flag_val == 1) then
            f = vacuum_to_air(ctx%state%spec_lambda(1:ctx%state%nspec))
        else
            f = ctx%state%spec_lambda(1:ctx%state%nspec)
        end if
        end associate
    end subroutine fsps_context_get_lambda_handle

    !> Ages of the SSP grid, log10(age / yr), ntfull values.
    subroutine fsps_context_get_ages_handle(handle, c_ages, status) &
        bind(C, name="fsps_context_get_ages")
        integer(c_int), value :: handle
        type(c_ptr), value :: c_ages
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: f(:)

        status = 0
        if (.not. handle_ok(handle)) then
            status = 1
            return
        end if
        associate(ctx => ctx_pool(handle))
        if (.not. ASSOCIATED(ctx%state%time_full)) then
            status = 301
            return
        end if
        call c_f_pointer(c_ages, f, [ctx%state%ntfull])
        f = ctx%state%time_full(1:ctx%state%ntfull)
        end associate
    end subroutine fsps_context_get_ages_handle

    !> Metallicities Z (mass fraction) of the isochrone grid, nz values.
    subroutine fsps_context_get_zlegend_handle(handle, c_z, status) &
        bind(C, name="fsps_context_get_zlegend")
        integer(c_int), value :: handle
        type(c_ptr), value :: c_z
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: f(:)

        status = 0
        if (.not. handle_ok(handle)) then
            status = 1
            return
        end if
        associate(ctx => ctx_pool(handle))
        if (.not. ASSOCIATED(ctx%state%zlegend)) then
            status = 301
            return
        end if
        call c_f_pointer(c_z, f, [ctx%state%nz])
        f = ctx%state%zlegend(1:ctx%state%nz)
        end associate
    end subroutine fsps_context_get_zlegend_handle

    !> Emission-line wavelengths [A], nemline values: vacuum, or air with vactoair_flag = 1.
    subroutine fsps_context_get_emlambda_handle(handle, c_wave, status) &
        bind(C, name="fsps_context_get_emlambda")
        integer(c_int), value :: handle
        type(c_ptr), value :: c_wave
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: f(:)

        status = 0
        if (.not. handle_ok(handle)) then
            status = 1
            return
        end if
        associate(ctx => ctx_pool(handle))
        call c_f_pointer(c_wave, f, [NEMLINE])
        if (ctx%vactoair_flag_val == 1) then
            f = vacuum_to_air(ctx%state%nebem_line_pos)
        else
            f = ctx%state%nebem_line_pos
        end if
        end associate
    end subroutine fsps_context_get_emlambda_handle

    !> Filter data, nbands values each (any pointer may be NULL): effective wavelength [A], AB
    !> magnitude of Vega (AB - Vega offset), absolute AB magnitude of the Sun.
    subroutine fsps_context_get_filter_data_handle(handle, c_wave_eff, c_mag_vega, c_mag_sun, status) &
        bind(C, name="fsps_context_get_filter_data")
        integer(c_int), value :: handle
        type(c_ptr), value :: c_wave_eff, c_mag_vega, c_mag_sun
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: f(:)
        integer :: nb

        status = 0
        if (.not. handle_ok(handle)) then
            status = 1
            return
        end if
        associate(ctx => ctx_pool(handle))
        nb = ctx%state%nbands
        if (.not. ASSOCIATED(ctx%state%filter_leff)) then
            status = 301
            return
        end if
        if (c_associated(c_wave_eff)) then
            call c_f_pointer(c_wave_eff, f, [nb]); f = ctx%state%filter_leff(1:nb)
        end if
        if (c_associated(c_mag_vega)) then
            call c_f_pointer(c_mag_vega, f, [nb]); f = ctx%state%magvega(1:nb) - ctx%state%magvega(1)
        end if
        if (c_associated(c_mag_sun)) then
            call c_f_pointer(c_mag_sun, f, [nb]); f = ctx%state%magsun(1:nb)
        end if
        end associate
    end subroutine fsps_context_get_filter_data_handle

    !> Tabular SFH for sfh = 3: ntab points of age [Gyr since the start of star formation],
    !> SFR [M_sun/yr] and, used with zcontinuous = 3, metallicity Z (met may be NULL: zeros).
    !> status 1 bad handle, 3 ntab out of range (1 to NTABMAX).
    subroutine fsps_context_set_sfh_tab_handle(handle, ntab, c_age, c_sfr, c_met, status) &
        bind(C, name="fsps_context_set_sfh_tab")
        integer(c_int), value :: handle, ntab
        type(c_ptr), value :: c_age, c_sfr, c_met
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: f(:)

        status = 0
        if (.not. handle_ok(handle)) then
            status = 1
            return
        end if
        associate(ctx => ctx_pool(handle))
        if (ntab < 1 .or. ntab > SIZE(ctx%state%sfh_tab, 2)) then
            status = 3
            call fsps_set_error(3, 'fsps_context_set_sfh_tab: ntab out of range')
            return
        end if
        ctx%state%sfh_tab = 0.0_wp
        ctx%state%ntabsfh = ntab
        ! Stored in years, as load_tabular_sfh does for sfh = 2
        call c_f_pointer(c_age, f, [ntab]); ctx%state%sfh_tab(1, 1:ntab) = f * 1.0e9_wp
        call c_f_pointer(c_sfr, f, [ntab]); ctx%state%sfh_tab(2, 1:ntab) = f
        if (c_associated(c_met)) then
            call c_f_pointer(c_met, f, [ntab]); ctx%state%sfh_tab(3, 1:ntab) = f
        end if
        end associate
    end subroutine fsps_context_set_sfh_tab_handle

    ! -------------------------------------------------------------------------
    ! Single CSP (compute_csp_scenario) on a context
    ! -------------------------------------------------------------------------

    subroutine reset_store(st)
        type(ctx_store_t), intent(INOUT) :: st
        if (ALLOCATED(st%spec)) deallocate (st%spec)
        if (ALLOCATED(st%mass)) deallocate (st%mass)
        if (ALLOCATED(st%lbol)) deallocate (st%lbol)
        if (ALLOCATED(st%have)) deallocate (st%have)
        if (ALLOCATED(st%results)) deallocate (st%results)
    end subroutine reset_store

    subroutine move_store(from, to)
        type(ctx_store_t), intent(INOUT) :: from, to
        if (ALLOCATED(from%spec)) call MOVE_ALLOC(from%spec, to%spec)
        if (ALLOCATED(from%mass)) call MOVE_ALLOC(from%mass, to%mass)
        if (ALLOCATED(from%lbol)) call MOVE_ALLOC(from%lbol, to%lbol)
        if (ALLOCATED(from%have)) call MOVE_ALLOC(from%have, to%have)
        if (ALLOCATED(from%results)) call MOVE_ALLOC(from%results, to%results)
    end subroutine move_store

    !> An SSP parameter changed: the stored SSP grids are stale.
    subroutine invalidate_ssps(st)
        type(ctx_store_t), intent(INOUT) :: st
        if (ALLOCATED(st%have)) st%have = .false.
    end subroutine invalidate_ssps

    !> The SSP grid of metallicity index z in the handle's store, computed if missing.
    subroutine ensure_ssp(ctx, st, z)
        type(fsps_context_t), intent(INOUT) :: ctx
        type(ctx_store_t), intent(INOUT) :: st
        integer, intent(IN) :: z
        type(PARAMS) :: pset
        integer :: ns, nt, nz

        ns = ctx%state%nspec
        nt = ctx%state%ntfull
        nz = ctx%state%nz
        if (ALLOCATED(st%spec)) then
            if (any(shape(st%spec) /= [ns, nt, nz])) call reset_store(st)
        end if
        if (.not. ALLOCATED(st%spec)) then
            allocate (st%spec(ns, nt, nz), st%mass(nt, nz), st%lbol(nt, nz), st%have(nz))
            st%have = .false.
        end if
        if (st%have(z)) return
        pset = ctx%pset
        pset%zmet = z
        call generate_ssp_grid(ctx, pset, st%mass(:, z), st%lbol(:, z), st%spec(:, :, z))
        st%have(z) = .true.
    end subroutine ensure_ssp

    !> @brief One CSP for the context's current parameters (compute_csp_scenario).
    !>
    !> @details
    !> zcontinuous selects the SSPs as in python-fsps: 0 the grid at zmet; 1 the grids interpolated
    !> linearly in log Z at logzsol; 2 the grids weighted by a metallicity distribution (pmetals);
    !> 3 all grids, metallicity from the tabular SFH. SSP grids are computed per metallicity on
    !> first use and kept until an SSP parameter is set. With tage > 0 one output, otherwise one
    !> per SSP age (*n_out); fetch them with fsps_context_get_csp. status: 0, 1 bad handle, 301
    !> context not set up, 304 zmet out of range, 306 bad zcontinuous, 311 compute_csp_scenario
    !> failed.
    subroutine fsps_context_compute_csp_handle(handle, zcontinuous, n_out, status) &
        bind(C, name="fsps_context_compute_csp")
        integer(c_int), value :: handle, zcontinuous
        integer(c_int), intent(OUT) :: n_out, status
        real(WP), allocatable :: spec(:, :, :), mass(:, :), lbol(:, :)
        integer :: z, nz, nzin, zlo, st

        n_out = 0
        status = 0
        if (.not. handle_ok(handle)) then
            status = 1
            return
        end if
        associate(ctx => ctx_pool(handle), store => ctx_store(handle))
        if (.not. ctx%initialized .or. ctx%state%check_sps_setup == 0) then
            status = 301
            call fsps_set_error(301, 'fsps_context_compute_csp: context not set up')
            return
        end if
        if (ALLOCATED(store%results)) deallocate (store%results)
        call fsps_prepare_pset(ctx)
        nz = ctx%state%nz
        nzin = 1
        select case (zcontinuous)
        case (0)
            z = ctx%pset%zmet
            if (z < 1 .or. z > nz) then
                status = 304
                call fsps_set_error(304, 'fsps_context_compute_csp: zmet out of range')
                return
            end if
            call ensure_ssp(ctx, store, z)
            allocate (spec, source=store%spec(:, :, z:z))
            allocate (mass, source=store%mass(:, z:z))
            allocate (lbol, source=store%lbol(:, z:z))
        case (1, 2)
            if (zcontinuous == 1) then
                zlo = MAX(MIN(find_interval(LOG10(ctx%state%zlegend(1:nz) / ctx%state%zsol), &
                                            ctx%pset%logzsol), nz - 1), 1)
                call ensure_ssp(ctx, store, zlo)
                call ensure_ssp(ctx, store, zlo + 1)
            else
                do z = 1, nz
                    call ensure_ssp(ctx, store, z)
                end do
            end if
            allocate (spec(ctx%state%nspec, ctx%state%ntfull, 1), mass(ctx%state%ntfull, 1), &
                      lbol(ctx%state%ntfull, 1))
            ! compute_interpolated_ssp reads the context's spec/mass/lbol_ssp_zz: lend it the
            ! stored grids (no copy; grids not yet needed are unused)
            call MOVE_ALLOC(store%spec, ctx%state%spec_ssp_zz)
            call MOVE_ALLOC(store%mass, ctx%state%mass_ssp_zz)
            call MOVE_ALLOC(store%lbol, ctx%state%lbol_ssp_zz)
            if (zcontinuous == 1) then
                call compute_interpolated_ssp(ctx, ctx%pset%logzsol, mass(:, 1), lbol(:, 1), spec(:, :, 1))
            else
                call compute_interpolated_ssp(ctx, ctx%pset%logzsol, mass(:, 1), lbol(:, 1), spec(:, :, 1), &
                                              z_param=ctx%pset%pmetals)
            end if
            call MOVE_ALLOC(ctx%state%spec_ssp_zz, store%spec)
            call MOVE_ALLOC(ctx%state%mass_ssp_zz, store%mass)
            call MOVE_ALLOC(ctx%state%lbol_ssp_zz, store%lbol)
        case (3)
            do z = 1, nz
                call ensure_ssp(ctx, store, z)
            end do
            nzin = nz
            allocate (spec, source=store%spec)
            allocate (mass, source=store%mass)
            allocate (lbol, source=store%lbol)
        case default
            status = 306
            call fsps_set_error(306, 'fsps_context_compute_csp: zcontinuous must be 0 to 3')
            return
        end select

        if (ctx%pset%sfh == 2 .or. ctx%pset%sfh == 3) then
            if (ctx%pset%sfh == 3 .and. ctx%state%ntabsfh == 0) then
                status = 311
                call fsps_set_error(311, 'fsps_context_compute_csp: sfh = 3 without a tabular SFH')
                return
            end if
            call load_tabular_sfh(ctx, ctx%pset, nzin)
        end if
        ctx%state%ssp_basis_is_dirty = .true.
        call fsps_context_update_ssp_basis(ctx, spec, mass, lbol, nzin)
        call fsps_context_prepare_csp_workspace(ctx)
        call compute_csp_scenario(ctx, ctx%pset, nzin, store%results, st)
        ! The basis now holds this call's SSPs, not the batch's: the next batch rebuilds its own
        ctx%state%ssp_basis_is_dirty = .true.
        if (st /= 0) then
            status = 311
            call fsps_set_error(311, 'fsps_context_compute_csp: compute_csp_scenario failed')
            if (ALLOCATED(store%results)) deallocate (store%results)
            return
        end if
        n_out = SIZE(store%results)
        end associate
    end subroutine fsps_context_compute_csp_handle

    !> @brief Outputs of the last fsps_context_compute_csp on the handle, n_out of each (any
    !> pointer may be NULL), column-major: age (log10 yr), mass (surviving stellar mass, M_sun),
    !> lbol (log10 L_sun), sfr (M_sun/yr), mdust (M_sun), mformed (M_sun formed), spec (nspec x
    !> n_out, L_sun/Hz), mags (nbands x n_out), emlines (nemline x n_out, L_sun), indx (nindx x
    !> n_out). Masses and luminosities are per solar mass formed, as in FSPS. status 1 bad handle,
    !> 302 no CSP computed, 309 mags or indx requested but not computed (compute_mags = 0 or
    !> compute_indices = 0).
    subroutine fsps_context_get_csp_handle(handle, c_age, c_mass, c_lbol, c_sfr, c_mdust, c_mformed, &
                                           c_spec, c_mags, c_emlines, c_indx, status) &
        bind(C, name="fsps_context_get_csp")
        integer(c_int), value :: handle
        type(c_ptr), value :: c_age, c_mass, c_lbol, c_sfr, c_mdust, c_mformed
        type(c_ptr), value :: c_spec, c_mags, c_emlines, c_indx
        integer(c_int), intent(OUT) :: status
        real(c_double), pointer :: v(:), a(:, :)
        integer :: i, n, ns, nb, ni

        status = 0
        if (.not. handle_ok(handle)) then
            status = 1
            return
        end if
        if (.not. ALLOCATED(ctx_store(handle)%results)) then
            status = 302
            call fsps_set_error(302, 'fsps_context_get_csp: no CSP computed')
            return
        end if
        associate(ctx => ctx_pool(handle), r => ctx_store(handle)%results)
        n = SIZE(r)
        ns = ctx%state%nspec
        nb = ctx%state%nbands
        ni = ctx%state%nindx
        if (c_associated(c_mags)) then
            do i = 1, n
                if (.not. ALLOCATED(r(i)%mags)) status = 309
            end do
        end if
        if (c_associated(c_indx)) then
            do i = 1, n
                if (.not. ALLOCATED(r(i)%indx)) status = 309
            end do
        end if
        if (status /= 0) then
            call fsps_set_error(309, 'fsps_context_get_csp: magnitudes or indices not computed')
            return
        end if
        if (c_associated(c_age)) then
            call c_f_pointer(c_age, v, [n]); v = r(:)%age
        end if
        if (c_associated(c_mass)) then
            call c_f_pointer(c_mass, v, [n]); v = r(:)%mass_csp
        end if
        if (c_associated(c_lbol)) then
            call c_f_pointer(c_lbol, v, [n]); v = r(:)%lbol_csp
        end if
        if (c_associated(c_sfr)) then
            call c_f_pointer(c_sfr, v, [n]); v = r(:)%sfr
        end if
        if (c_associated(c_mdust)) then
            call c_f_pointer(c_mdust, v, [n]); v = r(:)%mdust
        end if
        if (c_associated(c_mformed)) then
            call c_f_pointer(c_mformed, v, [n]); v = r(:)%mformed
        end if
        if (c_associated(c_spec)) then
            call c_f_pointer(c_spec, a, [ns, n])
            do i = 1, n
                a(:, i) = r(i)%spec(1:ns)
            end do
        end if
        if (c_associated(c_mags)) then
            call c_f_pointer(c_mags, a, [nb, n])
            do i = 1, n
                a(:, i) = r(i)%mags(1:nb)
            end do
        end if
        if (c_associated(c_emlines)) then
            call c_f_pointer(c_emlines, a, [NEMLINE, n])
            do i = 1, n
                a(:, i) = r(i)%emlines(1:NEMLINE)
            end do
        end if
        if (c_associated(c_indx)) then
            call c_f_pointer(c_indx, a, [ni, n])
            do i = 1, n
                a(:, i) = r(i)%indx(1:ni)
            end do
        end if
        end associate
    end subroutine fsps_context_get_csp_handle

#ifdef FSPS_ENABLE_LEGACY

    ! -------------------------------------------------------------------------
    ! INITIALIZATION
    ! -------------------------------------------------------------------------
    ! Initialize with default libraries.
    subroutine fsps_initialize(zin) bind(C, name="fsps_initialize")
        integer(c_int), value :: zin

        ! Call standard FSPS setup
        call fsps_ensure_default_ctx()
        call fsps_setup(fsps_default_ctx, zin, 'mist', 'miles', 'DL07')
        call fsps_initialize_state(zin)
    end subroutine fsps_initialize

    ! Initialize with explicit library selections.
    subroutine fsps_initialize_full(zin, compute_vega_mags0, vactoair_flag0, &
                                    c_isoc, c_spec, c_dust) &
        bind(C, name="fsps_initialize_full")
        integer(c_int), value :: zin
        integer(c_int), value :: compute_vega_mags0
        integer(c_int), value :: vactoair_flag0
        character(KIND=c_char), dimension(*), intent(IN) :: c_isoc
        character(KIND=c_char), dimension(*), intent(IN) :: c_spec
        character(KIND=c_char), dimension(*), intent(IN) :: c_dust
        character(LEN=64) :: isoc_type_in
        character(LEN=64) :: spec_type_in
        character(LEN=64) :: dust_type_in

        call fsps_ensure_default_ctx()
        fsps_default_ctx%compute_vega_mags_val = compute_vega_mags0
        fsps_default_ctx%vactoair_flag_val = vactoair_flag0

        call c_to_f_string(c_isoc, isoc_type_in)
        call c_to_f_string(c_spec, spec_type_in)
        call c_to_f_string(c_dust, dust_type_in)

        if (LEN_TRIM(isoc_type_in) == 0 .and. LEN_TRIM(spec_type_in) == 0 .and. &
            LEN_TRIM(dust_type_in) == 0) then
            call fsps_setup(fsps_default_ctx, zin)
        else if (LEN_TRIM(spec_type_in) == 0 .and. LEN_TRIM(dust_type_in) == 0) then
            call fsps_setup(fsps_default_ctx, zin, TRIM(isoc_type_in))
        else if (LEN_TRIM(dust_type_in) == 0) then
            call fsps_setup(fsps_default_ctx, zin, TRIM(isoc_type_in), TRIM(spec_type_in))
        else
            call fsps_setup(fsps_default_ctx, zin, TRIM(isoc_type_in), TRIM(spec_type_in), TRIM(dust_type_in))
        end if

        call fsps_initialize_state(zin)
    end subroutine fsps_initialize_full

    ! -------------------------------------------------------------------------
    ! PARAMETER CONTROL
    ! -------------------------------------------------------------------------
    ! Set integer parameters by name.
    subroutine fsps_set_int(c_key, val) bind(C, name="fsps_set_int")
        character(KIND=c_char), dimension(*), intent(IN) :: c_key
        integer(c_int), value :: val

        character(LEN=64) :: key
        call c_to_f_string(c_key, key)
        call fsps_ensure_default_ctx()

        select case (TRIM(key))
            ! Globals
        case ('imf_type')
            fsps_default_ctx%imf_type_val = val
        case ('tpagb_norm_type')
            fsps_default_ctx%tpagb_norm_type_val = val
        case ('pzcon')
            fsps_default_ctx%pzcon_val = val
        case ('interpolation_type')
            fsps_default_ctx%interpolation_type_val = val
        case ('add_agb_dust_model')
            fsps_default_ctx%add_agb_dust_model_val = val
        case ('add_stellar_remnants')
            fsps_default_ctx%add_stellar_remnants_val = val
        case ('add_agn_dust')
            fsps_default_ctx%add_agn_dust_val = val
        case ('use_wr_spectra')
            fsps_default_ctx%use_wr_spectra_val = val
        case ('add_xrb_emission')
            fsps_default_ctx%add_xrb_emission_val = val
        case ('smooth_lsf')
            fsps_default_ctx%smooth_lsf_val = val
        case ('smoothspec_fast')
            fsps_default_ctx%smoothspec_fast_val = val
        case ('dust_type')
            fsps_default_ctx%dust_type_val = val
        case ('add_dust_emission')
            fsps_default_ctx%add_dust_emission_val = val
        case ('add_neb_emission')
            fsps_default_ctx%add_neb_emission_val = val
        case ('add_neb_continuum')
            fsps_default_ctx%add_neb_continuum_val = val
        case ('cloudy_dust')
            fsps_default_ctx%cloudy_dust_val = val
        case ('add_igm_absorption')
            fsps_default_ctx%add_igm_absorption_val = val
        case ('nebemlineinspec')
            fsps_default_ctx%nebemlineinspec_val = val
        case ('smooth_velocity')
            fsps_default_ctx%smooth_velocity_val = val
        case ('redshift_colors')
            fsps_default_ctx%redshift_colors_val = val
        case ('compute_light_ages')
            fsps_default_ctx%compute_light_ages_val = val
        case ('compute_vega_mags')
            fsps_default_ctx%compute_vega_mags_val = val
        case ('vactoair_flag')
            fsps_default_ctx%vactoair_flag_val = val
        case ('use_isoc_mdot')
            fsps_default_ctx%use_isoc_mdot_val = val
        case ('setup_nebular_gaussians')
            fsps_default_ctx%setup_nebular_gaussians_val = val
        case ('fast_mode')
            fsps_default_ctx%fast_mode = (val /= 0)

            ! PARAMS Members
        case ('evtype')
            global_pset%evtype = val
        case ('sfh')
            global_pset%sfh = val
        case ('zmet')
            global_pset%zmet = val
        case ('wgp1')
            global_pset%wgp1 = val
        case ('wgp2')
            global_pset%wgp2 = val
        case ('wgp3')
            global_pset%wgp3 = val
        case ('compute_mags')
            global_pset%compute_mags = val
        case ('compute_indices')
            global_pset%compute_indices = val

        case DEFAULT
            call fsps_set_error(101, "[FSPS-C] Warning: Unknown integer parameter: "//TRIM(key))
        end select
    end subroutine fsps_set_int

    ! Set floating-point parameters by name.
    subroutine fsps_set_float(c_key, val) bind(C, name="fsps_set_float")
        character(KIND=c_char), dimension(*), intent(IN) :: c_key
        real(c_double), value :: val

        character(LEN=64) :: key
        call c_to_f_string(c_key, key)

        select case (TRIM(key))
            ! Globals
        case ('om0')
            om0 = real(val, WP)
        case ('ol0')
            ol0 = real(val, WP)
        case ('H0')
            H0 = real(val, WP)
        case ('tiny_logt')
            tiny_logt = real(val, WP)
        case ('imf_upper_limit')
            imf_upper_limit = real(val, WP)
        case ('imf_lower_limit')
            imf_lower_limit = real(val, WP)
        case ('logt_wmb_hot')
            logt_wmb_hot = real(val, WP)
        case ('nebular_smooth_init')
            nebular_smooth_init = real(val, WP)

            ! PARAMS Members - SSP
        case ('imf1')
            global_pset%imf1 = real(val, WP)
        case ('imf2')
            global_pset%imf2 = real(val, WP)
        case ('imf3')
            global_pset%imf3 = real(val, WP)
        case ('vdmc')
            global_pset%vdmc = real(val, WP)
        case ('mdave')
            global_pset%mdave = real(val, WP)
        case ('dell')
            global_pset%dell = real(val, WP)
        case ('delt')
            global_pset%delt = real(val, WP)
        case ('sbss')
            global_pset%sbss = real(val, WP)
        case ('fbhb')
            global_pset%fbhb = real(val, WP)
        case ('pagb')
            global_pset%pagb = real(val, WP)
        case ('agb_dust')
            global_pset%agb_dust = real(val, WP)
        case ('redgb')
            global_pset%redgb = real(val, WP)
        case ('agb')
            global_pset%agb = real(val, WP)
        case ('masscut')
            global_pset%masscut = real(val, WP)
        case ('fcstar')
            global_pset%fcstar = real(val, WP)
        case ('frac_xrb')
            global_pset%frac_xrb = real(val, WP)

            ! PARAMS Members - CSP
        case ('logzsol')
            global_pset%logzsol = real(val, WP)
        case ('tau')
            global_pset%tau = real(val, WP)
        case ('const')
            global_pset%const = real(val, WP)
        case ('tage')
            global_pset%tage = real(val, WP)
        case ('fburst')
            global_pset%fburst = real(val, WP)
        case ('tburst')
            global_pset%tburst = real(val, WP)
        case ('dust1')
            global_pset%dust1 = real(val, WP)
        case ('dust2')
            global_pset%dust2 = real(val, WP)
        case ('dust3')
            global_pset%dust3 = real(val, WP)
        case ('zred')
            global_pset%zred = real(val, WP)
        case ('pmetals')
            global_pset%pmetals = real(val, WP)
        case ('dust_clumps')
            global_pset%dust_clumps = real(val, WP)
        case ('frac_nodust')
            global_pset%frac_nodust = real(val, WP)
        case ('dust_index')
            global_pset%dust_index = real(val, WP)
        case ('dust_tesc')
            global_pset%dust_tesc = real(val, WP)
        case ('frac_obrun')
            global_pset%frac_obrun = real(val, WP)
        case ('uvb')
            global_pset%uvb = real(val, WP)
        case ('mwr')
            global_pset%mwr = real(val, WP)
        case ('dust1_index')
            global_pset%dust1_index = real(val, WP)
        case ('sf_start')
            global_pset%sf_start = real(val, WP)
        case ('sf_trunc')
            global_pset%sf_trunc = real(val, WP)
        case ('sf_slope')
            global_pset%sf_slope = real(val, WP)
        case ('duste_gamma')
            global_pset%duste_gamma = real(val, WP)
        case ('duste_umin')
            global_pset%duste_umin = real(val, WP)
        case ('duste_qpah')
            global_pset%duste_qpah = real(val, WP)
        case ('sigma_smooth')
            global_pset%sigma_smooth = real(val, WP)
        case ('min_wave_smooth')
            global_pset%min_wave_smooth = real(val, WP)
        case ('max_wave_smooth')
            global_pset%max_wave_smooth = real(val, WP)
        case ('gas_logu')
            global_pset%gas_logu = real(val, WP)
        case ('gas_logz')
            global_pset%gas_logz = real(val, WP)
        case ('igm_factor')
            global_pset%igm_factor = real(val, WP)
        case ('fagn')
            global_pset%fagn = real(val, WP)
        case ('agn_tau')
            global_pset%agn_tau = real(val, WP)

        case DEFAULT
            call fsps_set_error(102, "[FSPS-C] Warning: Unknown float parameter: "//TRIM(key))
        end select
    end subroutine fsps_set_float

    ! Validate common parameter constraints.
    subroutine fsps_validate_params(status) bind(C, name="fsps_validate_params")
        integer(c_int), intent(OUT) :: status
        real(WP) :: sumcb

        status = 0
        call fsps_ensure_default_ctx()
        if (global_pset%zmet < 1 .or. global_pset%zmet > fsps_default_ctx%state%nz) then
            status = 1
            call fsps_set_error(201, "[FSPS-C] Warning: zmet out of range")
        end if
        if (fsps_default_ctx%dust_type_val < 0 .or. fsps_default_ctx%dust_type_val > 6) then
            status = 2
            call fsps_set_error(202, "[FSPS-C] Warning: dust_type out of range")
        end if
        if (fsps_default_ctx%imf_type_val < 0 .or. fsps_default_ctx%imf_type_val > 5) then
            status = 3
            call fsps_set_error(203, "[FSPS-C] Warning: imf_type out of range")
        end if
        if (global_pset%tage > 0.0 .and. global_pset%sf_start > global_pset%tage) then
            status = 4
            call fsps_set_error(204, "[FSPS-C] Warning: sf_start > tage")
        end if
        sumcb = global_pset%const + global_pset%fburst
        if (sumcb > 1.0) then
            status = 5
            call fsps_set_error(205, "[FSPS-C] Warning: const + fburst > 1")
        end if
    end subroutine fsps_validate_params

    ! Set string parameters by name.
    subroutine fsps_set_str(c_key, c_val) bind(C, name="fsps_set_str")
        character(KIND=c_char), dimension(*), intent(IN) :: c_key
        character(KIND=c_char), dimension(*), intent(IN) :: c_val
        character(LEN=64) :: key
        character(LEN=256) :: val

        call c_to_f_string(c_key, key)
        call c_to_f_string(c_val, val)

        select case (TRIM(key))
        case ('imf_filename')
            global_pset%imf_filename = TRIM(val)
        case ('sfh_filename')
            global_pset%sfh_filename = TRIM(val)
        case DEFAULT
            call fsps_set_error(103, "[FSPS-C] Warning: Unknown string parameter: "//TRIM(key))
        end select
    end subroutine fsps_set_str

    ! -------------------------------------------------------------------------
    ! COMPUTATION
    ! -------------------------------------------------------------------------
    ! Compute SSP or CSP depending on `sfh`.
    subroutine fsps_compute(c_spec) bind(C, name="fsps_compute")
        type(c_ptr), value :: c_spec
        real(WP), pointer :: f_spec(:, :)

        ! SSP Workspace
        real(WP), allocatable, target :: ssp_mass(:), ssp_lbol(:)
        real(WP), allocatable, target :: ssp_spec(:, :)
        real(WP), allocatable :: ssp_mass_zz(:, :), ssp_lbol_zz(:, :), ssp_spec_zz(:, :, :)

        integer :: i
        integer :: n_spec, n_time

        if (.not. ASSOCIATED(global_pset)) then
            call fsps_set_error(301, "[FSPS-C] Error: fsps_compute called before initialize!")
            return
        end if

        call fsps_ensure_default_ctx()
        call fsps_ensure_legacy_state()
        n_spec = fsps_default_ctx%state%nspec
        n_time = fsps_default_ctx%state%ntfull

        call C_F_POINTER(c_spec, f_spec, [n_spec, n_time])

        ! 1. Calculate the SSP for the current global_pset%zmet
        ! We allocate workspace because we might need to feed this into COMPSP
        allocate (ssp_mass(n_time))
        allocate (ssp_lbol(n_time))
        allocate (ssp_spec(n_spec, n_time))

        call generate_ssp_grid(fsps_default_ctx, global_pset, ssp_mass, ssp_lbol, ssp_spec)

        if (global_pset%sfh .eq. 0) then
            ! --- SSP Mode ---
            ! Copy directly to output buffer
            f_spec = ssp_spec

            ! Populate global_ocompsp so that get_mags works
            do i = 1, n_time
                global_ocompsp(i)%spec = ssp_spec(:, i)
                global_ocompsp(i)%mags = 0.0 ! Will be calc'd by get_mags
            end do
        else
            ! --- CSP Mode ---
            ! Use fsps_csp to handle CSP generation.
            allocate (ssp_mass_zz(n_time, 1))
            allocate (ssp_lbol_zz(n_time, 1))
            allocate (ssp_spec_zz(n_spec, n_time, 1))
            ssp_mass_zz(:, 1) = ssp_mass
            ssp_lbol_zz(:, 1) = ssp_lbol
            ssp_spec_zz(:, :, 1) = ssp_spec

            if (global_pset%sfh == 2 .or. global_pset%sfh == 3) then
                call load_tabular_sfh(fsps_default_ctx, global_pset, 1)
            end if

            block
                type(COMPSPOUT), allocatable :: results(:)
                integer :: status
                call fsps_context_update_ssp_basis(fsps_default_ctx, ssp_spec_zz, ssp_mass_zz, ssp_lbol_zz, 1)
                call compute_csp_scenario(fsps_default_ctx, global_pset, 1, results, status)
                if (status /= 0) then
                    call fsps_set_error(311, "[FSPS-C] Error: compute_csp_scenario failed")
                    deallocate (results)
                    return
                end if
                call fsps_store_results(results, f_spec)
                deallocate (results)
            end block
        end if

        deallocate (ssp_mass)
        deallocate (ssp_lbol)
        deallocate (ssp_spec)
        if (ALLOCATED(ssp_mass_zz)) deallocate (ssp_mass_zz)
        if (ALLOCATED(ssp_lbol_zz)) deallocate (ssp_lbol_zz)
        if (ALLOCATED(ssp_spec_zz)) deallocate (ssp_spec_zz)

    end subroutine fsps_compute

    ! Compute CSP with metallicity interpolation.
    subroutine fsps_compute_csp(zcontinuous) bind(C, name="fsps_compute_csp")
        integer(c_int), value :: zcontinuous

        if (.not. ASSOCIATED(global_pset)) then
            call fsps_set_error(302, "[FSPS-C] Error: fsps_compute_csp called before initialize!")
            return
        end if
        call fsps_ensure_default_ctx()
        if (zcontinuous == 3 .and. fsps_default_ctx%add_neb_emission_val /= 0) then
            call fsps_set_error(206, "[FSPS-C] Warning: zcontinuous=3 with nebular emission enabled")
        end if

        call fsps_compute_zdep(zcontinuous)
    end subroutine fsps_compute_csp

    ! Compute and cache a single SSP at a metallicity index.
    subroutine fsps_compute_ssp(zin) bind(C, name="fsps_compute_ssp")
        integer(c_int), value :: zin
        integer :: zidx
        integer :: old_z

        if (.not. ASSOCIATED(global_pset)) then
            call fsps_set_error(303, "[FSPS-C] Error: fsps_compute_ssp called before initialize!")
            return
        end if

        call fsps_ensure_default_ctx()
        zidx = zin
        if (zidx < 1 .or. zidx > fsps_default_ctx%state%nz) then
            call fsps_set_error(304, "[FSPS-C] Error: z index out of bounds in fsps_compute_ssp")
            return
        end if

        old_z = global_pset%zmet
        global_pset%zmet = zidx
        call generate_ssp_grid(fsps_default_ctx, global_pset, fsps_default_ctx%state%mass_ssp_zz(:, zidx), &
                               fsps_default_ctx%state%lbol_ssp_zz(:, zidx), fsps_default_ctx%state%spec_ssp_zz(:, :, zidx))
        has_ssp(zidx) = 1
        has_ssp_age(zidx, :) = global_pset%ssp_gen_age
        global_pset%zmet = old_z
    end subroutine fsps_compute_ssp

    ! Compute and cache SSPs across the full Z grid.
    subroutine fsps_compute_ssps() bind(C, name="fsps_compute_ssps")
        integer :: zidx
        call fsps_ensure_default_ctx()
        do zidx = 1, fsps_default_ctx%state%nz
            call fsps_compute_ssp(zidx)
        end do
    end subroutine fsps_compute_ssps

    ! Compute CSP with metallicity interpolation mode (0-3).
    subroutine fsps_compute_zdep(ztype) bind(C, name="fsps_compute_zdep")
        integer(c_int), value :: ztype
        real(WP), allocatable :: mass(:), lbol(:)
        real(WP), allocatable :: spec(:, :)
        real(WP), allocatable :: mass_zz(:, :), lbol_zz(:, :), spec_zz(:, :, :)
        real(WP) :: zpos
        integer :: zlo, zmet
        integer :: n_spec, n_time

        if (.not. ASSOCIATED(global_pset)) then
            call fsps_set_error(305, "[FSPS-C] Error: fsps_compute_zdep called before initialize!")
            return
        end if

        call fsps_ensure_default_ctx()
        call fsps_ensure_legacy_state()
        n_spec = fsps_default_ctx%state%nspec
        n_time = fsps_default_ctx%state%ntfull
        allocate (mass(n_time))
        allocate (lbol(n_time))
        allocate (spec(n_spec, n_time))
        allocate (mass_zz(n_time, 1))
        allocate (lbol_zz(n_time, 1))
        allocate (spec_zz(n_spec, n_time, 1))

        select case (ztype)
        case (0)
            zmet = global_pset%zmet
            if (has_ssp(zmet) == 0) call fsps_compute_ssp(zmet)
            if (global_pset%sfh == 2 .or. global_pset%sfh == 3) then
                call load_tabular_sfh(fsps_default_ctx, global_pset, 1)
            end if
            block
                type(COMPSPOUT), allocatable :: results(:)
                integer :: status
                call fsps_context_update_ssp_basis(fsps_default_ctx, &
                                                   fsps_default_ctx%state%spec_ssp_zz(:, :, zmet:zmet), &
                                                   fsps_default_ctx%state%mass_ssp_zz(:, zmet:zmet), &
                                                   fsps_default_ctx%state%lbol_ssp_zz(:, zmet:zmet), 1)
                call compute_csp_scenario(fsps_default_ctx, global_pset, 1, results, status)
                if (status /= 0) then
                    call fsps_set_error(312, "[FSPS-C] Error: compute_csp_scenario failed")
                    deallocate (results)
                    return
                end if
                call fsps_store_results(results)
                deallocate (results)
            end block
        case (1)
            zpos = global_pset%logzsol
            zlo = MAX(MIN(find_interval(LOG10(fsps_default_ctx%state%zlegend/fsps_default_ctx%state%zsol), zpos), &
                          fsps_default_ctx%state%nz - 1), 1)
            do zmet = zlo, zlo + 1
                if (has_ssp(zmet) == 0) call fsps_compute_ssp(zmet)
            end do
            call compute_interpolated_ssp(fsps_default_ctx, zpos, mass, lbol, spec)
            mass_zz(:, 1) = mass
            lbol_zz(:, 1) = lbol
            spec_zz(:, :, 1) = spec
            if (global_pset%sfh == 2 .or. global_pset%sfh == 3) then
                call load_tabular_sfh(fsps_default_ctx, global_pset, 1)
            end if
            block
                type(COMPSPOUT), allocatable :: results(:)
                integer :: status
                call fsps_context_update_ssp_basis(fsps_default_ctx, spec_zz, mass_zz, lbol_zz, 1)
                call compute_csp_scenario(fsps_default_ctx, global_pset, 1, results, status)
                if (status /= 0) then
                    call fsps_set_error(313, "[FSPS-C] Error: compute_csp_scenario failed")
                    deallocate (results)
                    return
                end if
                call fsps_store_results(results)
                deallocate (results)
            end block
        case (2)
            zpos = global_pset%logzsol
            do zmet = 1, fsps_default_ctx%state%nz
                if (has_ssp(zmet) == 0) call fsps_compute_ssp(zmet)
            end do
            call compute_interpolated_ssp(fsps_default_ctx, zpos, mass, lbol, spec, z_param=global_pset%pmetals)
            mass_zz(:, 1) = mass
            lbol_zz(:, 1) = lbol
            spec_zz(:, :, 1) = spec
            if (global_pset%sfh == 2 .or. global_pset%sfh == 3) then
                call load_tabular_sfh(fsps_default_ctx, global_pset, 1)
            end if
            block
                type(COMPSPOUT), allocatable :: results(:)
                integer :: status
                call fsps_context_update_ssp_basis(fsps_default_ctx, spec_zz, mass_zz, lbol_zz, 1)
                call compute_csp_scenario(fsps_default_ctx, global_pset, 1, results, status)
                if (status /= 0) then
                    call fsps_set_error(314, "[FSPS-C] Error: compute_csp_scenario failed")
                    deallocate (results)
                    return
                end if
                call fsps_store_results(results)
                deallocate (results)
            end block
        case (3)
            do zmet = 1, fsps_default_ctx%state%nz
                if (has_ssp(zmet) == 0) call fsps_compute_ssp(zmet)
            end do
            if (global_pset%sfh == 2 .or. global_pset%sfh == 3) then
                call load_tabular_sfh(fsps_default_ctx, global_pset, fsps_default_ctx%state%nz)
            end if
            block
                type(COMPSPOUT), allocatable :: results(:)
                integer :: status
                call fsps_context_update_ssp_basis(fsps_default_ctx, fsps_default_ctx%state%spec_ssp_zz, &
                                                   fsps_default_ctx%state%mass_ssp_zz, &
                                                   fsps_default_ctx%state%lbol_ssp_zz, fsps_default_ctx%state%nz)
                call compute_csp_scenario(fsps_default_ctx, global_pset, fsps_default_ctx%state%nz, results, status)
                if (status /= 0) then
                    call fsps_set_error(315, "[FSPS-C] Error: compute_csp_scenario failed")
                    deallocate (results)
                    return
                end if
                call fsps_store_results(results)
                deallocate (results)
            end block
        case DEFAULT
            call fsps_set_error(306, "[FSPS-C] Error: Unknown ztype in fsps_compute_zdep")
        end select

        deallocate (mass)
        deallocate (lbol)
        deallocate (spec)
        deallocate (mass_zz)
        deallocate (lbol_zz)
        deallocate (spec_zz)
    end subroutine fsps_compute_zdep

    ! Interpolate an SSP to a target metallicity and age.
    subroutine fsps_interp_ssp(zpos, tpos, c_spec, c_mass, c_lbol) &
        bind(C, name="fsps_interp_ssp")
        real(c_double), value :: zpos, tpos
        type(c_ptr), value :: c_spec, c_mass, c_lbol
        real(WP), pointer :: f_spec(:, :), f_mass(:), f_lbol(:)
        real(WP), allocatable :: time(:)
        integer :: zlo, zmet, tlo, n_spec, n_t

        if (.not. ASSOCIATED(global_pset)) then
            call fsps_set_error(307, "[FSPS-C] Error: fsps_interp_ssp called before initialize!")
            return
        end if

        call fsps_ensure_default_ctx()
        n_t = fsps_default_ctx%state%nt
        n_spec = fsps_default_ctx%state%nspec
        allocate (time(n_t))
        zlo = MAX(MIN(find_interval(LOG10(fsps_default_ctx%state%zlegend/fsps_default_ctx%state%zsol), &
                                    real(zpos, WP)), fsps_default_ctx%state%nz - 1), 1)
        time = fsps_default_ctx%state%timestep_isoc(zlo, :)
        tlo = MAX(MIN(find_interval(time, real(tpos, WP)), n_t - 1), 1)

        do zmet = zlo, zlo + 1
            if (has_ssp_age(zmet, tlo) == 0 .or. has_ssp_age(zmet, tlo + 1) == 0) then
                global_pset%ssp_gen_age = 0
                global_pset%ssp_gen_age(tlo:tlo + 1) = 1
                call fsps_compute_ssp(zmet)
                global_pset%ssp_gen_age = 1
            end if
        end do

        call C_F_POINTER(c_spec, f_spec, [n_spec, 1])
        call C_F_POINTER(c_mass, f_mass, [1])
        call C_F_POINTER(c_lbol, f_lbol, [1])
        call compute_interpolated_ssp(fsps_default_ctx, real(zpos, WP), f_mass, f_lbol, f_spec, t_pos=real(tpos, WP))
        deallocate (time)
    end subroutine fsps_interp_ssp

    ! -------------------------------------------------------------------------
    ! PHOTOMETRY
    ! -------------------------------------------------------------------------
    ! Compute magnitudes for all bands.
    subroutine fsps_get_mags(zred, c_mags) bind(C, name="fsps_get_mags")
        real(c_double), value :: zred
        type(c_ptr), value :: c_mags
        real(WP), pointer :: f_mags(:, :) ! (nbands, ntfull)

        integer :: i
        integer :: n_spec, n_bands, n_time
        real(WP), allocatable :: tspec(:)
        integer, allocatable :: all_bands(:)

        call fsps_ensure_default_ctx()
        n_spec = fsps_default_ctx%state%nspec
        n_bands = fsps_default_ctx%state%nbands
        n_time = fsps_default_ctx%state%ntfull
        call C_F_POINTER(c_mags, f_mags, [n_bands, n_time])

        allocate (tspec(n_spec))
        allocate (all_bands(n_bands))

        all_bands = 1

        do i = 1, n_time
            tspec = global_ocompsp(i)%spec
            call compute_magnitudes(fsps_default_ctx, real(zred, WP), tspec, f_mags(:, i), all_bands)
        end do

        deallocate (tspec)
        deallocate (all_bands)

    end subroutine fsps_get_mags

    ! Compute magnitudes for a band mask.
    subroutine fsps_get_mags_mask(zred, c_mags, c_mc) bind(C, name="fsps_get_mags_mask")
        real(c_double), value :: zred
        type(c_ptr), value :: c_mags
        type(c_ptr), value :: c_mc
        real(WP), pointer :: f_mags(:, :) ! (nbands, ntfull)
        integer(c_int), pointer :: f_mc(:)

        integer :: i
        integer :: n_spec, n_bands, n_time
        real(WP), allocatable :: tspec(:)

        call fsps_ensure_default_ctx()
        n_spec = fsps_default_ctx%state%nspec
        n_bands = fsps_default_ctx%state%nbands
        n_time = fsps_default_ctx%state%ntfull
        call C_F_POINTER(c_mags, f_mags, [n_bands, n_time])
        call C_F_POINTER(c_mc, f_mc, [n_bands])

        allocate (tspec(n_spec))

        do i = 1, n_time
            tspec = global_ocompsp(i)%spec
            call compute_magnitudes(fsps_default_ctx, real(zred, WP), tspec, f_mags(:, i), f_mc)
        end do

        deallocate (tspec)

    end subroutine fsps_get_mags_mask

    ! Return spectra from the compsp output buffer.
    subroutine fsps_get_spec(c_spec) bind(C, name="fsps_get_spec")
        type(c_ptr), value :: c_spec
        real(WP), pointer :: f_spec(:, :)
        integer :: i
        integer :: n_spec, n_time

        call fsps_ensure_default_ctx()
        n_spec = fsps_default_ctx%state%nspec
        n_time = fsps_default_ctx%state%ntfull
        call C_F_POINTER(c_spec, f_spec, [n_spec, n_time])

        do i = 1, n_time
            f_spec(:, i) = global_ocompsp(i)%spec
        end do
    end subroutine fsps_get_spec

    ! Return spectra converted to Lsun/Angstrom.
    subroutine fsps_get_spec_peraa(c_spec) bind(C, name="fsps_get_spec_peraa")
        type(c_ptr), value :: c_spec
        real(WP), pointer :: f_spec(:, :)
        real(WP) :: lam
        real(WP) :: lamarr(1)
        integer :: i, j
        integer :: n_spec, n_time

        call fsps_ensure_default_ctx()
        n_spec = fsps_default_ctx%state%nspec
        n_time = fsps_default_ctx%state%ntfull
        call C_F_POINTER(c_spec, f_spec, [n_spec, n_time])

        do j = 1, n_time
            do i = 1, n_spec
                if (fsps_default_ctx%vactoair_flag_val == 1) then
                    lamarr = vacuum_to_air(fsps_default_ctx%state%spec_lambda(i:i))
                    lam = lamarr(1)
                else
                    lam = fsps_default_ctx%state%spec_lambda(i)
                end if
                f_spec(i, j) = global_ocompsp(j)%spec(i)*(3.0e18_wp/(lam*lam))
            end do
        end do
    end subroutine fsps_get_spec_peraa

    ! Return CSP statistics and emission lines.
    subroutine fsps_get_stats(c_age, c_mass, c_lbol, c_sfr, c_mdust, c_mformed, &
                              c_emlines) bind(C, name="fsps_get_stats")
        type(c_ptr), value :: c_age, c_mass, c_lbol, c_sfr, c_mdust, c_mformed
        type(c_ptr), value :: c_emlines
        real(WP), pointer :: f_age(:), f_mass(:), f_lbol(:), f_sfr(:), f_mdust(:), f_mformed(:)
        real(WP), pointer :: f_emlines(:, :)
        integer :: i
        integer :: n_time

        call fsps_ensure_default_ctx()
        n_time = fsps_default_ctx%state%ntfull
        call C_F_POINTER(c_age, f_age, [n_time])
        call C_F_POINTER(c_mass, f_mass, [n_time])
        call C_F_POINTER(c_lbol, f_lbol, [n_time])
        call C_F_POINTER(c_sfr, f_sfr, [n_time])
        call C_F_POINTER(c_mdust, f_mdust, [n_time])
        call C_F_POINTER(c_mformed, f_mformed, [n_time])
        call C_F_POINTER(c_emlines, f_emlines, [NEMLINE, n_time])

        do i = 1, n_time
            f_age(i) = global_ocompsp(i)%age
            f_mass(i) = global_ocompsp(i)%mass_csp
            f_lbol(i) = global_ocompsp(i)%lbol_csp
            f_sfr(i) = global_ocompsp(i)%sfr
            f_mdust(i) = global_ocompsp(i)%mdust
            f_mformed(i) = global_ocompsp(i)%mformed
            f_emlines(:, i) = global_ocompsp(i)%emlines
        end do
    end subroutine fsps_get_stats

    subroutine fsps_get_indices(c_spec, c_indices) bind(C, name="fsps_get_indices")
        type(c_ptr), value :: c_spec
        type(c_ptr), value :: c_indices
        real(WP), pointer :: f_spec(:)
        real(WP), pointer :: f_indices(:)
        real(WP), allocatable :: lamarr(:)
        integer :: n_spec, n_indx

        call fsps_ensure_default_ctx()
        n_spec = fsps_default_ctx%state%nspec
        n_indx = fsps_default_ctx%state%nindx
        allocate (lamarr(n_spec))

        call C_F_POINTER(c_spec, f_spec, [n_spec])
        call C_F_POINTER(c_indices, f_indices, [n_indx])

        if (fsps_default_ctx%vactoair_flag_val == 1) then
            lamarr = vacuum_to_air(fsps_default_ctx%state%spec_lambda)
        else
            lamarr = fsps_default_ctx%state%spec_lambda
        end if

        call compute_spectral_indices(fsps_default_ctx, lamarr, f_spec, f_indices)
        deallocate (lamarr)
    end subroutine fsps_get_indices

    ! Return a stellar spectrum for stellar parameters.
    subroutine fsps_stellar_spectrum(mact, logt, lbol, logg, phase, ffco, lmdot, &
                                     wght, c_spec) bind(C, name="fsps_stellar_spectrum")
        real(c_double), value :: mact, logt, lbol, logg, phase, ffco, lmdot, wght
        type(c_ptr), value :: c_spec
        real(WP), pointer :: f_spec(:)
        real(KP), allocatable :: kspec(:)

        if (.not. ASSOCIATED(global_pset)) then
            call fsps_set_error(308, "[FSPS-C] Error: fsps_stellar_spectrum called before initialize!")
            return
        end if

        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_spec, f_spec, [fsps_default_ctx%state%nspec])
        allocate (kspec(size(f_spec)))
        call get_stellar_spectrum(fsps_default_ctx, global_pset, real(mact, WP), real(logt, WP), real(lbol, WP), &
                                  real(logg, WP), real(phase, WP), real(ffco, WP), real(lmdot, WP), kspec)
        f_spec = kspec
    end subroutine fsps_stellar_spectrum

    ! -------------------------------------------------------------------------
    ! TEARDOWN
    ! -------------------------------------------------------------------------
    subroutine fsps_finalize() bind(C, name="fsps_finalize")
        integer :: i

        if (ASSOCIATED(global_ocompsp)) then
            do i = 1, SIZE(global_ocompsp)
                if (ALLOCATED(global_ocompsp(i)%mags)) deallocate (global_ocompsp(i)%mags)
                if (ALLOCATED(global_ocompsp(i)%spec)) deallocate (global_ocompsp(i)%spec)
                if (ALLOCATED(global_ocompsp(i)%indx)) deallocate (global_ocompsp(i)%indx)
                if (ALLOCATED(global_ocompsp(i)%emlines)) deallocate (global_ocompsp(i)%emlines)
            end do
            deallocate (global_ocompsp)
            global_ocompsp => NULL()
        end if

        if (ASSOCIATED(global_pset)) then
            if (ALLOCATED(global_pset%mag_compute)) deallocate (global_pset%mag_compute)
            if (ALLOCATED(global_pset%ssp_gen_age)) deallocate (global_pset%ssp_gen_age)
            deallocate (global_pset)
            global_pset => NULL()
        end if

        if (ALLOCATED(has_ssp)) deallocate (has_ssp)
        if (ALLOCATED(has_ssp_age)) deallocate (has_ssp_age)

        if (fsps_default_ctx_ready) then
            call fsps_destroy(fsps_default_ctx)
        end if
    end subroutine fsps_finalize

    ! -------------------------------------------------------------------------
    ! UTILS
    ! -------------------------------------------------------------------------
    subroutine fsps_get_dims(n_spec, n_time) bind(C, name="fsps_get_dims")
        integer(c_int), intent(OUT) :: n_spec, n_time
        call fsps_ensure_default_ctx()
        n_spec = fsps_default_ctx%state%nspec
        n_time = fsps_default_ctx%state%ntfull
    end subroutine fsps_get_dims

    subroutine fsps_get_nbands(n_bands) bind(C, name="fsps_get_nbands")
        integer(c_int), intent(OUT) :: n_bands
        call fsps_ensure_default_ctx()
        n_bands = fsps_default_ctx%state%nbands
    end subroutine fsps_get_nbands

    subroutine fsps_get_nindx(n_indices) bind(C, name="fsps_get_nindx")
        integer(c_int), intent(OUT) :: n_indices
        call fsps_ensure_default_ctx()
        n_indices = fsps_default_ctx%state%nindx
    end subroutine fsps_get_nindx

    ! Dimension getters
    subroutine fsps_get_nspec(n_spec) bind(C, name="fsps_get_nspec")
        integer(c_int), intent(OUT) :: n_spec
        call fsps_ensure_default_ctx()
        n_spec = fsps_default_ctx%state%nspec
    end subroutine fsps_get_nspec

    subroutine fsps_get_ntfull(n_time) bind(C, name="fsps_get_ntfull")
        integer(c_int), intent(OUT) :: n_time
        call fsps_ensure_default_ctx()
        n_time = fsps_default_ctx%state%ntfull
    end subroutine fsps_get_ntfull

    subroutine fsps_get_nt(n_time) bind(C, name="fsps_get_nt")
        integer(c_int), intent(OUT) :: n_time
        call fsps_ensure_default_ctx()
        n_time = fsps_default_ctx%state%nt
    end subroutine fsps_get_nt

    subroutine fsps_get_nm(n_mass) bind(C, name="fsps_get_nm")
        integer(c_int), intent(OUT) :: n_mass
        n_mass = NM
    end subroutine fsps_get_nm

    subroutine fsps_get_ntabmax(n_tabmax) bind(C, name="fsps_get_ntabmax")
        integer(c_int), intent(OUT) :: n_tabmax
        n_tabmax = NTABMAX
    end subroutine fsps_get_ntabmax

    subroutine fsps_get_nz(n_z) bind(C, name="fsps_get_nz")
        integer(c_int), intent(OUT) :: n_z
        call fsps_ensure_default_ctx()
        n_z = fsps_default_ctx%state%nz
    end subroutine fsps_get_nz

    subroutine fsps_get_nemline(n_line) bind(C, name="fsps_get_nemline")
        integer(c_int), intent(OUT) :: n_line
        n_line = NEMLINE
    end subroutine fsps_get_nemline

    ! Isochrone metadata
    subroutine fsps_get_isochrone_dimensions(n_age, n_mass) &
        bind(C, name="fsps_get_isochrone_dimensions")
        integer(c_int), intent(OUT) :: n_age, n_mass
        call fsps_ensure_default_ctx()
        n_age = fsps_default_ctx%state%nt
        n_mass = NM
    end subroutine fsps_get_isochrone_dimensions

    subroutine fsps_get_nmass_isochrone(z_idx, t_idx, n_mass) &
        bind(C, name="fsps_get_nmass_isochrone")
        integer(c_int), value :: z_idx, t_idx
        integer(c_int), intent(OUT) :: n_mass
        call fsps_ensure_default_ctx()
        if (z_idx < 1 .or. z_idx > fsps_default_ctx%state%nz .or. t_idx < 1 .or. t_idx > fsps_default_ctx%state%nt) then
            n_mass = -1
            call fsps_set_error(207, "[FSPS-C] Warning: get_nmass_isochrone index out of range")
            return
        end if
        n_mass = fsps_default_ctx%state%nmass_isoc(z_idx, t_idx)
    end subroutine fsps_get_nmass_isochrone

    subroutine fsps_get_zsol(z_sol) bind(C, name="fsps_get_zsol")
        real(c_double), intent(OUT) :: z_sol
        call fsps_ensure_default_ctx()
        z_sol = fsps_default_ctx%state%zsol
    end subroutine fsps_get_zsol

    subroutine fsps_get_zlegend(c_zlegend) bind(C, name="fsps_get_zlegend")
        type(c_ptr), value :: c_zlegend
        real(WP), pointer :: f_zlegend(:)
        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_zlegend, f_zlegend, [fsps_default_ctx%state%nz])
        f_zlegend = fsps_default_ctx%state%zlegend
    end subroutine fsps_get_zlegend

    subroutine fsps_get_timefull(c_timefull) bind(C, name="fsps_get_timefull")
        type(c_ptr), value :: c_timefull
        real(WP), pointer :: f_timefull(:)
        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_timefull, f_timefull, [fsps_default_ctx%state%ntfull])
        f_timefull = fsps_default_ctx%state%time_full
    end subroutine fsps_get_timefull

    subroutine fsps_get_lambda(c_lambda) bind(C, name="fsps_get_lambda")
        type(c_ptr), value :: c_lambda
        real(WP), pointer :: f_lambda(:)
        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_lambda, f_lambda, [fsps_default_ctx%state%nspec])
        if (fsps_default_ctx%vactoair_flag_val == 1) then
            f_lambda = vacuum_to_air(fsps_default_ctx%state%spec_lambda)
        else
            f_lambda = fsps_default_ctx%state%spec_lambda
        end if
    end subroutine fsps_get_lambda

    subroutine fsps_get_emlambda(c_emlambda) bind(C, name="fsps_get_emlambda")
        type(c_ptr), value :: c_emlambda
        real(WP), pointer :: f_emlambda(:)
        call C_F_POINTER(c_emlambda, f_emlambda, [NEMLINE])
        call fsps_ensure_default_ctx()
        if (fsps_default_ctx%vactoair_flag_val == 1) then
            f_emlambda = vacuum_to_air(fsps_default_ctx%state%nebem_line_pos)
        else
            f_emlambda = fsps_default_ctx%state%nebem_line_pos
        end if
    end subroutine fsps_get_emlambda

    subroutine fsps_get_res(c_res) bind(C, name="fsps_get_res")
        type(c_ptr), value :: c_res
        real(WP), pointer :: f_res(:)
        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_res, f_res, [fsps_default_ctx%state%nspec])
        f_res = fsps_default_ctx%state%spec_res
    end subroutine fsps_get_res

    subroutine fsps_get_filter_data(c_wave_eff, c_mag_vega, c_mag_sun) &
        bind(C, name="fsps_get_filter_data")
        type(c_ptr), value :: c_wave_eff, c_mag_vega, c_mag_sun
        real(WP), pointer :: f_wave_eff(:), f_mag_vega(:), f_mag_sun(:)
        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_wave_eff, f_wave_eff, [fsps_default_ctx%state%nbands])
        call C_F_POINTER(c_mag_vega, f_mag_vega, [fsps_default_ctx%state%nbands])
        call C_F_POINTER(c_mag_sun, f_mag_sun, [fsps_default_ctx%state%nbands])
        f_wave_eff = fsps_default_ctx%state%filter_leff
        f_mag_vega = fsps_default_ctx%state%magvega - fsps_default_ctx%state%magvega(1)
        f_mag_sun = fsps_default_ctx%state%magsun
    end subroutine fsps_get_filter_data

    subroutine fsps_get_ssp_weights(c_wghts) bind(C, name="fsps_get_ssp_weights")
        type(c_ptr), value :: c_wghts
        real(WP), pointer :: f_wghts(:, :)
        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_wghts, f_wghts, [fsps_default_ctx%state%ntfull, fsps_default_ctx%state%nz])
        f_wghts = fsps_default_ctx%state%weight_ssp
    end subroutine fsps_get_ssp_weights

    subroutine fsps_get_csp_components(c_young, c_old) bind(C, name="fsps_get_csp_components")
        type(c_ptr), value :: c_young, c_old
        real(WP), pointer :: f_young(:), f_old(:)
        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_young, f_young, [fsps_default_ctx%state%nspec])
        call C_F_POINTER(c_old, f_old, [fsps_default_ctx%state%nspec])
        f_young = fsps_default_ctx%state%spec_young
        f_old = fsps_default_ctx%state%spec_old
    end subroutine fsps_get_csp_components

    subroutine fsps_get_ssp_spec(c_spec, c_mass, c_lbol) bind(C, name="fsps_get_ssp_spec")
        type(c_ptr), value :: c_spec, c_mass, c_lbol
        real(WP), pointer :: f_spec(:, :, :)
        real(WP), pointer :: f_mass(:, :), f_lbol(:, :)
        integer :: zidx

        call fsps_ensure_default_ctx()
        do zidx = 1, fsps_default_ctx%state%nz
            if (has_ssp(zidx) == 0) call fsps_compute_ssp(zidx)
        end do

        call C_F_POINTER(c_spec, f_spec, [fsps_default_ctx%state%nspec, fsps_default_ctx%state%ntfull, fsps_default_ctx%state%nz])
        call C_F_POINTER(c_mass, f_mass, [fsps_default_ctx%state%ntfull, fsps_default_ctx%state%nz])
        call C_F_POINTER(c_lbol, f_lbol, [fsps_default_ctx%state%ntfull, fsps_default_ctx%state%nz])
        f_spec = fsps_default_ctx%state%spec_ssp_zz
        f_mass = fsps_default_ctx%state%mass_ssp_zz
        f_lbol = fsps_default_ctx%state%lbol_ssp_zz
    end subroutine fsps_get_ssp_spec

    ! Set a tabular SFH.
    subroutine fsps_set_sfh_tab(ntab, c_age, c_sfr, c_met) bind(C, name="fsps_set_sfh_tab")
        integer(c_int), value :: ntab
        type(c_ptr), value :: c_age, c_sfr, c_met
        real(WP), pointer :: f_age(:), f_sfr(:), f_met(:)

        call C_F_POINTER(c_age, f_age, [ntab])
        call C_F_POINTER(c_sfr, f_sfr, [ntab])
        call C_F_POINTER(c_met, f_met, [ntab])

        call fsps_ensure_default_ctx()
        fsps_default_ctx%state%ntabsfh = ntab
        fsps_default_ctx%state%sfh_tab(1, 1:fsps_default_ctx%state%ntabsfh) = f_age
        fsps_default_ctx%state%sfh_tab(2, 1:fsps_default_ctx%state%ntabsfh) = f_sfr
        fsps_default_ctx%state%sfh_tab(3, 1:fsps_default_ctx%state%ntabsfh) = f_met
    end subroutine fsps_set_sfh_tab

    ! Set band computation mask for mags.
    subroutine fsps_set_mag_compute(n_bands, c_mask) bind(C, name="fsps_set_mag_compute")
        integer(c_int), value :: n_bands
        type(c_ptr), value :: c_mask
        integer(c_int), pointer :: f_mask(:)

        call C_F_POINTER(c_mask, f_mask, [n_bands])
        if (.not. ASSOCIATED(global_pset)) then
            call fsps_set_error(309, "[FSPS-C] Error: fsps_set_mag_compute called before initialize!")
            return
        end if
        call fsps_ensure_default_ctx()
        if (n_bands < 1 .or. n_bands > fsps_default_ctx%state%nbands) then
            call fsps_set_error(208, "[FSPS-C] Warning: fsps_set_mag_compute size out of range")
            return
        end if
        if (.not. ALLOCATED(global_pset%mag_compute)) then
            allocate (global_pset%mag_compute(fsps_default_ctx%state%nbands))
            global_pset%mag_compute = 0
        end if
        global_pset%mag_compute(1:n_bands) = f_mask(1:n_bands)
    end subroutine fsps_set_mag_compute

    ! Set SSP age computation mask.
    subroutine fsps_set_ssp_gen_age(n_age, c_mask) bind(C, name="fsps_set_ssp_gen_age")
        integer(c_int), value :: n_age
        type(c_ptr), value :: c_mask
        integer(c_int), pointer :: f_mask(:)

        call C_F_POINTER(c_mask, f_mask, [n_age])
        if (.not. ASSOCIATED(global_pset)) then
            call fsps_set_error(310, "[FSPS-C] Error: fsps_set_ssp_gen_age called before initialize!")
            return
        end if
        call fsps_ensure_default_ctx()
        if (n_age < 1 .or. n_age > fsps_default_ctx%state%nt) then
            call fsps_set_error(209, "[FSPS-C] Warning: fsps_set_ssp_gen_age size out of range")
            return
        end if
        if (.not. ALLOCATED(global_pset%ssp_gen_age)) then
            allocate (global_pset%ssp_gen_age(fsps_default_ctx%state%nt))
            global_pset%ssp_gen_age = 0
        end if
        global_pset%ssp_gen_age(1:n_age) = f_mask(1:n_age)
    end subroutine fsps_set_ssp_gen_age

    ! Provide a wavelength-dependent LSF (sigma in km/s).
    subroutine fsps_set_ssp_lsf(nsv, c_sigma, wlo, whi) bind(C, name="fsps_set_ssp_lsf")
        integer(c_int), value :: nsv
        type(c_ptr), value :: c_sigma
        real(c_double), value :: wlo, whi
        real(WP), pointer :: f_sigma(:)

        call C_F_POINTER(c_sigma, f_sigma, [nsv])

        call fsps_ensure_default_ctx()
        fsps_default_ctx%state%lsfinfo%minlam = real(wlo, WP)
        fsps_default_ctx%state%lsfinfo%maxlam = real(whi, WP)
        if (ALLOCATED(fsps_default_ctx%state%lsfinfo%lsf)) deallocate (fsps_default_ctx%state%lsfinfo%lsf)
        allocate (fsps_default_ctx%state%lsfinfo%lsf(nsv))
        fsps_default_ctx%state%lsfinfo%lsf = f_sigma
    end subroutine fsps_set_ssp_lsf

    ! Smooth a spectrum using a Gaussian kernel.
    subroutine fsps_smooth_spectrum(c_wave, c_spec, sigma_broad, minw, maxw) &
        bind(C, name="fsps_smooth_spectrum")
        type(c_ptr), value :: c_wave, c_spec
        real(c_double), value :: sigma_broad, minw, maxw
        real(WP), pointer :: f_wave(:), f_spec(:)

        call fsps_ensure_default_ctx()
        call C_F_POINTER(c_wave, f_wave, [fsps_default_ctx%state%nspec])
        call C_F_POINTER(c_spec, f_spec, [fsps_default_ctx%state%nspec])

        call apply_smoothing(fsps_default_ctx, f_wave, f_spec, real(sigma_broad, WP), &
                             real(minw, WP), real(maxw, WP))
    end subroutine fsps_smooth_spectrum

    ! Write isochrone data to a .cmd file.
    subroutine fsps_write_isochrone(c_outfile) bind(C, name="fsps_write_isochrone")
        character(KIND=c_char), dimension(*), intent(IN) :: c_outfile
        character(LEN=100) :: outfile
        integer :: i, tt, zz
        real(WP) :: dz, loggi, hb_wght
        real(WP), dimension(NM) :: wght
        real(WP), allocatable :: spec(:)
        real(KP), allocatable :: kspec(:)
        real(WP), allocatable :: mags_tmp(:)
        real(WP), allocatable :: time_grid(:)
        real(WP), allocatable :: mini(:, :), mact(:, :), logl(:, :), logt(:, :), logg(:, :), &
                                 ffco(:, :), phase(:, :), lmdot(:, :), weights(:, :)
        real(WP), allocatable :: mags(:, :, :)
        integer, allocatable :: nmass(:)

        call fsps_ensure_default_ctx()
        call c_to_f_string(c_outfile, outfile)

        associate ( &
            nbands => fsps_default_ctx%state%nbands, nspec => fsps_default_ctx%state%nspec, &
            nt => fsps_default_ctx%state%nt, &
            isoc_type => fsps_default_ctx%state%isoc_type, &
            mini_isoc => fsps_default_ctx%state%mini_isoc, &
            mact_isoc => fsps_default_ctx%state%mact_isoc, &
            logl_isoc => fsps_default_ctx%state%logl_isoc, &
            logt_isoc => fsps_default_ctx%state%logt_isoc, &
            logg_isoc => fsps_default_ctx%state%logg_isoc, &
            ffco_isoc => fsps_default_ctx%state%ffco_isoc, &
            lmdot_isoc => fsps_default_ctx%state%lmdot_isoc, &
            phase_isoc => fsps_default_ctx%state%phase_isoc, &
            nmass_isoc => fsps_default_ctx%state%nmass_isoc, &
            timestep_isoc => fsps_default_ctx%state%timestep_isoc, &
            mact_isoc_full => fsps_default_ctx%state%mact_isoc)

            allocate (spec(nspec), kspec(nspec))
            allocate (mags_tmp(nbands))
            allocate (mini(nt, NM), mact(nt, NM), logl(nt, NM), logt(nt, NM), logg(nt, NM))
            allocate (ffco(nt, NM), phase(nt, NM), lmdot(nt, NM))
            allocate (nmass(nt))
            allocate (time_grid(nt))
            allocate (weights(nt, NM))
            allocate (mags(nt, NM, nbands))

            dz = 0.0_wp
            hb_wght = 0.0_wp
            weights = 0.0_wp
            mags = 0.0_wp
            wght = 0.0_wp
            zz = global_pset%zmet

            mini = mini_isoc(zz, :, :)
            mact = mact_isoc(zz, :, :)
            logl = logl_isoc(zz, :, :)
            logt = logt_isoc(zz, :, :)
            logg = logg_isoc(zz, :, :)
            ffco = ffco_isoc(zz, :, :)
            lmdot = lmdot_isoc(zz, :, :)
            phase = phase_isoc(zz, :, :)
            nmass = nmass_isoc(zz, :)
            time_grid = timestep_isoc(zz, :)

            do tt = 1, nt
                wght = 0.0_wp
                call compute_imf_weights(fsps_default_ctx, mini(tt, :), wght, nmass(tt))

                if (global_pset%fbhb .gt. 0.0 .or. global_pset%sbss .gt. 1e-3) &
                    call modify_horizontal_branch(fsps_default_ctx, tt, global_pset%fbhb, timestep_isoc(zz, tt), hb_wght, nmass, &
                                                  mini, mact, logl, logt, logg, phase, wght)

                if (timestep_isoc(zz, tt) .ge. BHB_SBS_TIME .and. global_pset%sbss .gt. 1e-3) &
                    call apply_blue_stragglers(fsps_default_ctx, tt, zz, global_pset%sbss, hb_wght, nmass, &
                                               mini, mact, logl, logt, logg, phase, wght)

                call modify_giant_branch(fsps_default_ctx, tt, zz, timestep_isoc(zz, tt), nmass(tt), global_pset%delt, &
                                    global_pset%dell, global_pset%pagb, global_pset%redgb, global_pset%agb, logl, logt, phase, wght)

                do i = 1, nmass(tt)
                    call get_stellar_spectrum(fsps_default_ctx, global_pset, mact(tt, i), logt(tt, i), 10**logl(tt, i), &
                                              logg(tt, i), phase(tt, i), ffco(tt, i), lmdot(tt, i), kspec)
                    spec = kspec
                    call compute_magnitudes(fsps_default_ctx, dz, spec, mags_tmp)

                    if (isoc_type .eq. 'bsti') then
                        loggi = LOG10(GRAVITY_L_M_T_COEFF*mact_isoc_full(zz, tt, i)/logl(tt, i)) + 4*logt(tt, i)
                    else
                        loggi = logg(tt, i)
                    end if

                    logg(tt, i) = loggi
                    weights(tt, i) = wght(i)
                    mags(tt, i, :) = mags_tmp
                end do
            end do

            call write_isochrone_cmd(fsps_default_ctx, global_pset, TRIM(outfile), time_grid, nmass, &
                                     mini, mact, logl, logt, logg, phase, ffco, lmdot, weights, mags)

            deallocate (spec, mags_tmp, mini, mact, logl, logt, logg, ffco, phase, lmdot, nmass, time_grid, weights, mags)

        end associate
    end subroutine fsps_write_isochrone

    subroutine fsps_get_setup_vars(cvms, vta_flag) bind(C, name="fsps_get_setup_vars")
        integer(c_int), intent(OUT) :: cvms, vta_flag
        call fsps_ensure_default_ctx()
        cvms = fsps_default_ctx%compute_vega_mags_val
        vta_flag = fsps_default_ctx%vactoair_flag_val
    end subroutine fsps_get_setup_vars

    ! Return the currently-selected library names.
    subroutine fsps_get_libraries(c_isoc, c_isoc_len, c_spec, c_spec_len, &
                                  c_dust, c_dust_len) &
        bind(C, name="fsps_get_libraries")
        character(KIND=c_char), dimension(*), intent(OUT) :: c_isoc
        character(KIND=c_char), dimension(*), intent(OUT) :: c_spec
        character(KIND=c_char), dimension(*), intent(OUT) :: c_dust
        integer(c_int), value :: c_isoc_len, c_spec_len, c_dust_len

        call fsps_ensure_default_ctx()
        call f_to_c_string(fsps_default_ctx%state%isoc_type, c_isoc, c_isoc_len)
        call f_to_c_string(fsps_default_ctx%state%spec_type, c_spec, c_spec_len)
        call f_to_c_string(fsps_default_ctx%state%str_dustem, c_dust, c_dust_len)
    end subroutine fsps_get_libraries

#endif

    ! Convert a null-terminated C string into a Fortran fixed-length string.
    subroutine c_to_f_string(c_ptr, f_str)
        character(KIND=c_char), dimension(*), intent(IN) :: c_ptr
        character(LEN=*), intent(OUT) :: f_str
        integer :: i

        f_str = ''
        i = 1
        do while (c_ptr(i) /= c_null_char .and. i <= LEN(f_str))
            f_str(i:i) = c_ptr(i)
            i = i + 1
        end do
    end subroutine c_to_f_string

    ! Copy a Fortran string into a C buffer with null termination.
    subroutine f_to_c_string(f_str, c_ptr, c_len)
        character(LEN=*), intent(IN) :: f_str
        character(KIND=c_char), dimension(*), intent(OUT) :: c_ptr
        integer(c_int), value :: c_len
        integer :: i, ncopy

        ncopy = MIN(LEN_TRIM(f_str), c_len - 1)
        do i = 1, ncopy
            c_ptr(i) = f_str(i:i)
        end do
        if (ncopy + 1 <= c_len) c_ptr(ncopy + 1) = c_null_char
        if (ncopy + 2 <= c_len) c_ptr(ncopy + 2:c_len) = c_null_char
    end subroutine f_to_c_string

end module fsps_c_api
