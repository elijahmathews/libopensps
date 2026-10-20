module fsps_initialization
    !> @brief
    !> Handles the initialization, memory allocation, and data loading for FSPS contexts.
    !>
    !> @details
    !> This module replaces the legacy `sps_setup.f90`. It orchestrates the setup process:
    !> 1. Determines array dimensions based on selected physics models (MIST vs Padova, MILES vs BaSeL).
    !> 2. Manages the shared data cache (allocating new memory or linking to existing caches).
    !> 3. Calls `fsps_io` to load data from disk only if the cache is cold. A context that hits
    !>    an existing entry reads no files and never writes the shared arrays (other contexts
    !>    may be computing from them); it only copies the small per-context tables recorded in
    !>    the entry and computes its own context-local state.
    !> 4. Pre-calculates derived physical quantities (redshift grids, cosmology).

    use iso_fortran_env, only: error_unit, int64
    use fsps_precision, only: WP, KP
    use fsps_constants, only: VERBOSE, TIME_RES_INCR, SAFE_FLOOR, C_LIGHT, L_SOL, PI, &
                              NM, NDIM_LOGT, NDIM_LOGG, NDIM_WMB_LOGT, NDIM_WMB_LOGG, &
                              NDIM_PAGB, NDIM_WR, N_AGB_O, N_AGB_C, N_AGB_CAR, &
                              NTAU_DAGB, NTEFF_DAGB, NEMLINE, NEBNZ, NEBNAGE, NEBNIP, &
                              NAGNDUST, NTABMAX, NT_XRB, NZ_XRB, NSPEC_XRB, &
                              NT_MIST, NZ_MIST, NT_PADOVA, NZ_PADOVA, NT_PARSEC, NZ_PARSEC, &
                              NT_BASTI, NZ_BASTI, NT_GENEVA, NZ_GENEVA
    use fsps_strings, only: to_lower
    use fsps_context_types, only: fsps_context_t
    use fsps_cache, only: fsps_setup_cache_t, fsps_setup_tables_t, fsps_cache_get_setup, &
                          fsps_setup_lock, fsps_setup_unlock
    use fsps_environment, only: fsps_resolve_paths, fsps_cleanup
    use fsps_io, only: load_zlegend_file, load_wavelength_grid, load_spectral_resolution, &
                       read_isochrone_database, read_spectral_binary, read_bpass_data, &
                       load_dust_emission_table, load_nebular_grid, load_filter_definitions, &
                       load_standard_sed, load_index_definitions, load_wr_spectra, &
                       load_agb_spectra, load_post_agb_spectra, load_attenuation_curves, &
                       load_wmbasic_spectra, load_lsf_data, load_agn_dust_models, &
                       apply_legacy_filter_norm, isochrone_kind, load_spectral_grid_axes, &
                       use_standard_library, library_spec, metallicity_axis
    use fsps_datalib, only: datalib_t, datalib_available, datalib_open, datalib_meta_real, datalib_axis_len, &
                            datalib_axis_values, datalib_check_array, datalib_read_real
    use fsps_cosmology, only: get_universe_age, get_luminosity_distance
    use fsps_interpolation, only: find_interval, interpolate_linear
    use fsps_integration, only: integrate_trapezoid_array
    use fsps_smoothing, only: velocity_kernel_t, build_velocity_kernel, apply_velocity_kernel
    use fsps_dust, only: AGB_SMOOTH_SIGMA, AGB_SMOOTH_LAMBDA_MIN, AGB_SMOOTH_LAMBDA_MAX
    

    implicit none
    private

    ! Only the main driver is public (plus a read-only diagnostic counter for tests)
    public :: fsps_initialize_data
    public :: fsps_shared_load_count

    ! ---------------------------------------------------------------------
    ! Module state
    ! ---------------------------------------------------------------------

    !> Number of times the shared setup data has been loaded from disk (one per cache miss),
    !> process-wide. Only modified under fsps_setup_lock.
    integer, save :: shared_load_count = 0

contains

    !> @brief Number of cache misses that loaded shared setup data from disk in this process.
    !>
    !> @details
    !> A setup that hits an existing cache entry performs no file I/O for shared data and leaves
    !> this count unchanged; intended for tests and diagnostics.
    integer function fsps_shared_load_count()
        call fsps_setup_lock()
        fsps_shared_load_count = shared_load_count
        call fsps_setup_unlock()
    end function fsps_shared_load_count

    !> @brief Loads all data stored in the shared setup cache (called on a cache miss only).
    !>
    !> @details
    !> Every routine called here writes shared (cache-owned) arrays and/or the fixed-size
    !> per-context tables captured by `save_setup_tables`. Its results depend only on settings
    !> that are part of the cache key (see `build_cache_key`).
    !>
    !> @param[inout] ctx The context populating the new cache entry.
    !> @param[in]    zin Metallicity index to load (-1 / 0 for all).
    subroutine load_shared_data(ctx, zin)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin

        call load_stellar_data(ctx, zin)
        call load_interstellar_physics(ctx)
        call load_photometry_data(ctx)
        call compute_filter_leff(ctx)
        call load_attenuation_curves(ctx)
        call load_agn_dust_models(ctx)
        call load_index_definitions(ctx)
        call build_time_grid(ctx)
        call compute_frequency_grid(ctx)
        if (ctx%add_agb_dust_model_val == 1) call precompute_agb_smoothed_templates(ctx)
    end subroutine load_shared_data

    !> @brief Pre-smooths the templates that dusty AGB stars are built from.
    !>
    !> @details
    !> Legacy ADD_AGB_DUST smooths each dusty AGB star's spectrum (Gaussian of 10^4 km/s for
    !> 3 um <= lambda <= 1e8 A) before applying the DUSTY transfer function; without it the
    !> transfer function, about 6 at 65 um, amplifies the templates' far-IR pixel structure into
    !> ~10% jitter. A star's spectrum is a linear combination of templates times a
    !> wavelength-independent scale (and lambda^2/c for the O-rich and Lancon-Wood C-rich
    !> families), so smoothing commutes with it: smoothing the templates once here gives the same
    !> spectra as smoothing every dusty AGB star, which took ~90% of SSP generation in legacy
    !> (cconroy20/fsps#80). The smoothing is always in velocity, by direct integration, whatever
    !> smooth_velocity and smoothspec_fast are set to (they describe the output smoothing; legacy
    !> applies them here too, and with smooth_velocity = 0 smooths these stars by 1 um).
    !> Only the changed segment, lambda >= AGB_SMOOTH_LAMBDA_MIN, is stored.
    subroutine precompute_agb_smoothed_templates(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        type(fsps_setup_cache_t), pointer :: cache
        type(velocity_kernel_t) :: kernel
        real(WP), allocatable :: work(:), lam2c(:), sm(:)  ! smoothing runs in WP; results are stored as KP
        integer :: nspec, i0, n_ir, iz, it, ig, k

        cache => ctx%setup_cache
        if (.not. associated(cache)) return
        nspec = ctx%state%nspec
        ! First pixel at or above the smoothing range (a loop: nvfortran lacks logical FINDLOC)
        i0 = 0
        do k = 1, nspec
            if (ctx%state%spec_lambda(k) >= AGB_SMOOTH_LAMBDA_MIN) then
                i0 = k
                exit
            end if
        end do
        if (i0 == 0) return
        n_ir = nspec - i0 + 1
        allocate (work(nspec), sm(n_ir))
        lam2c = ctx%state%spec_lambda(1:nspec)**2 / C_LIGHT

        ! The smoothing weights depend only on the wavelength grid: build them once
        call build_velocity_kernel(ctx%state%spec_lambda(1:nspec), AGB_SMOOTH_SIGMA, AGB_SMOOTH_LAMBDA_MIN, &
                                   AGB_SMOOTH_LAMBDA_MAX, kernel)

        if (associated(ctx%state%speclib)) then
            allocate (cache%speclib_agb_smooth(n_ir, size(ctx%state%speclib, 2), size(ctx%state%speclib, 3), &
                                               size(ctx%state%speclib, 4)))
            do ig = 1, size(ctx%state%speclib, 4)
                do it = 1, size(ctx%state%speclib, 3)
                    do iz = 1, size(ctx%state%speclib, 2)
                        work = real(ctx%state%speclib(1:nspec, iz, it, ig), WP)
                        if (any(work(i0:) > 0.0_wp)) then
                            call apply_velocity_kernel(kernel, work, i0, sm)
                            cache%speclib_agb_smooth(:, iz, it, ig) = real(sm, KP)
                        else
                            cache%speclib_agb_smooth(:, iz, it, ig) = real(work(i0:), KP)
                        end if
                    end do
                end do
            end do
            ctx%state%speclib_agb_smooth => cache%speclib_agb_smooth
        end if

        if (associated(ctx%state%agb_spec_o)) then
            allocate (cache%agb_spec_o_smooth(n_ir, size(ctx%state%agb_spec_o, 2)))
            do k = 1, size(ctx%state%agb_spec_o, 2)
                work = lam2c * ctx%state%agb_spec_o(1:nspec, k)
                call apply_velocity_kernel(kernel, work, i0, sm)
                cache%agb_spec_o_smooth(:, k) = real(sm, KP)
            end do
            ctx%state%agb_spec_o_smooth => cache%agb_spec_o_smooth
        end if

        if (associated(ctx%state%agb_spec_c)) then
            allocate (cache%agb_spec_c_smooth(n_ir, size(ctx%state%agb_spec_c, 2)))
            do k = 1, size(ctx%state%agb_spec_c, 2)
                work = lam2c * ctx%state%agb_spec_c(1:nspec, k)
                call apply_velocity_kernel(kernel, work, i0, sm)
                cache%agb_spec_c_smooth(:, k) = real(sm, KP)
            end do
            ctx%state%agb_spec_c_smooth => cache%agb_spec_c_smooth
        end if

        if (associated(ctx%state%agb_spec_car)) then
            allocate (cache%agb_spec_car_smooth(n_ir, size(ctx%state%agb_spec_car, 2)))
            do k = 1, size(ctx%state%agb_spec_car, 2)
                work = ctx%state%agb_spec_car(1:nspec, k)
                call apply_velocity_kernel(kernel, work, i0, sm)
                cache%agb_spec_car_smooth(:, k) = real(sm, KP)
            end do
            ctx%state%agb_spec_car_smooth => cache%agb_spec_car_smooth
        end if

    end subroutine precompute_agb_smoothed_templates

    !> @brief Main entry point for initializing an FSPS context.
    !>
    !> @details
    !> This routine coordinates the entire setup lifecycle. It is the only routine
    !> that `fsps_api` needs to call.
    !>
    !> @param[inout] ctx             The context to initialize.
    !> @param[in]    zin             Specific metallicity index to load (-1 for all).
    !> @param[in]    isoc_type_in    (Optional) Isochrone library string.
    !> @param[in]    spec_type_in    (Optional) Spectral library string.
    !> @param[in]    dust_type_in    (Optional) Dust emission model string.
    subroutine fsps_initialize_data(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin
        character(len=*), intent(in), optional :: isoc_type_in
        character(len=*), intent(in), optional :: spec_type_in
        character(len=*), intent(in), optional :: dust_type_in
        logical :: cache_is_new
        character(len=512) :: cache_key
        type(fsps_setup_cache_t), pointer :: cache_ptr
        character(len=64) :: isoc_name, spec_name, dust_name

        ! Serialize the whole setup: cache lookup, population of a new entry and the loaders all
        ! write into memory shared with every context using the same entry (released below)
        call fsps_setup_lock()

        call fsps_cleanup(ctx)
        ctx%zin = zin

        call normalize_library_names(isoc_type_in, spec_type_in, dust_type_in, isoc_name, spec_name, dust_name)
        ctx%isoc_type_name = trim(isoc_name)
        ctx%spec_type_name = trim(spec_name)
        ctx%dust_type_name = trim(dust_name)
        call split_selection(isoc_name, ctx%state%isoc_type, ctx%state%isoc_select)
        call split_selection(spec_name, ctx%state%spec_type, ctx%state%spec_select)
        ctx%state%str_dustem = trim(dust_name)

        ! Paths first: with standard-format libraries the dimensions come from their manifests
        call fsps_resolve_paths(ctx)

        call set_simulation_dimensions(ctx)
        call configure_dust_model(ctx)
        if ((len_trim(ctx%state%isoc_select) > 0 .and. .not. uses_standard_isochrones(ctx)) .or. &
            (len_trim(ctx%state%spec_select) > 0 .and. .not. use_standard_library(ctx, 'stellar_spectra', &
                                                                                     ctx%state%spec_type))) then
            write (error_unit, '(A)') '[FSPS_INIT] Error: axis selections (name:axis=value) need a standard-format library'
            error stop 1
        end if

        if (zin > ctx%state%nz) then
            write (error_unit, '(A,1x,I0,1x,I0)') '[FSPS_INIT] Error: zin > nz', zin, ctx%state%nz
            error stop 1
        end if

        call build_cache_key(ctx, zin, cache_key)
        call fsps_cache_get_setup(cache_key, ctx%setup_cache, cache_is_new)
        cache_ptr => ctx%setup_cache

        if (cache_is_new) then
            call populate_cache_metadata(ctx, cache_ptr)
            call allocate_shared_memory(cache_ptr, ctx)
        end if

        call link_context_to_cache(ctx, cache_ptr)
        call allocate_local_workspace(ctx)

        if (cache_is_new) then
            ! Cache miss: read every data file once, filling the shared arrays of the new entry
            ! and recording the small per-context tables the loaders produce alongside them
            call initialize_cache_arrays(ctx)
            call reset_setup_tables(ctx)
            call load_shared_data(ctx, zin)
            call save_setup_tables(ctx, cache_ptr)
            shared_load_count = shared_load_count + 1
        else
            ! Cache hit: the shared arrays are complete and may be read concurrently by other
            ! contexts' computations, so they must not be written; no file is read
            call restore_setup_tables(ctx, cache_ptr)
        end if

        ! Context-local state that depends on settings outside the cache key
        call compute_imf_lower_bound(ctx, zin)
        if (ctx%smooth_lsf_val == 1) call load_lsf_data(ctx)
        call compute_local_properties(ctx)

        ctx%state%check_sps_setup = 1
        ctx%initialized = .true.

        call fsps_setup_unlock()

        if (VERBOSE == 1) then
            write (*, '(A)') '      ...done'
        end if

    end subroutine fsps_initialize_data

    ! ========================================================================
    ! PRIVATE HELPER SUBROUTINES
    ! ========================================================================

    !> @brief Determines grid dimensions (NT, NZ, NSPEC) based on library selection.
    !>
    !> @details
    !> Sets the fundamental array shapes for the simulation based on the selected
    !> Isochrone (`isoc_type`) and Spectral Library (`spec_type`).
    !>
    !> Logic corresponds to legacy `sps_setup` lines ~749-756.
    !>
    !> **Dimensions Calculated:**
    !> - `nt`: Number of age steps in the isochrone.
    !> - `nz`: Number of metallicity steps in the isochrone.
    !> - `nspec`: Number of wavelength points in the spectral library.
    !> - `ntfull`: The expanded time grid size (including interpolation).
    !>
    !> @param[inout] ctx The context to configure.
    subroutine set_simulation_dimensions(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        type(datalib_t) :: lib
        character(len=64) :: iso_str, spec_str

        iso_str = trim(ctx%state%isoc_type)
        spec_str = trim(ctx%state%spec_type)

        if (uses_standard_isochrones(ctx)) then
            call isochrone_dimensions_from_manifest(ctx)
        else
        select case (iso_str)
        case ('mist')
            ctx%state%zsol = 0.0142_wp
            ctx%state%nt = NT_MIST
            ctx%state%nz = NZ_MIST
        case ('pdva')
            ctx%state%zsol = 0.019_wp
            ctx%state%nt = NT_PADOVA
            ctx%state%nz = NZ_PADOVA
        case ('prsc')
            ctx%state%zsol = 0.01524_wp
            ctx%state%nt = NT_PARSEC
            ctx%state%nz = NZ_PARSEC
        case ('bsti')
            ctx%state%zsol = 0.020_wp
            ctx%state%nt = NT_BASTI
            ctx%state%nz = NZ_BASTI
        case ('gnva')
            ctx%state%zsol = 0.020_wp
            ctx%state%nt = NT_GENEVA
            ctx%state%nz = NZ_GENEVA
        case ('bpss')
            ctx%state%zsol = 0.020_wp
            ctx%state%nt = 43
            ctx%state%nz = 12
        case default
            write (error_unit, '(A,1x,A)') '[FSPS_INIT] Error: Unknown isoc_type', trim(iso_str)
            error stop 1
        end select
        end if

        if (iso_str /= 'bpss' .and. datalib_available(ctx%sps_home, 'stellar_spectra', spec_str, ctx%data_format)) then
            call spectral_dimensions_from_manifest(ctx)
        else
        select case (spec_str)
        case ('miles')
            ctx%state%zsol_spec = 0.019_wp
            ctx%state%nzinit = 5
            ctx%state%nspec = 5994
        case ('basel')
            ctx%state%zsol_spec = 0.020_wp
            ctx%state%nzinit = 6
            ctx%state%nspec = 1963
        case ('bpass')
            ctx%state%zsol_spec = 0.020_wp
            ctx%state%nzinit = 1
            ctx%state%nspec = 15000
        case default
            if (len_trim(spec_str) >= 3 .and. spec_str(1:3) == 'c3k') then
                ctx%state%zsol_spec = 0.0134_wp
                ctx%state%nzinit = 11
                ctx%state%nspec = 11149
            else
                write (error_unit, '(A,1x,A)') '[FSPS_INIT] Error: Unknown spec_type', trim(spec_str)
                error stop 1
            end if
        end select
        end if

        ctx%state%nbands = 159
        if (len_trim(ctx%state%alt_filter_file) == 0 .or. trim(ctx%state%alt_filter_file) == 'allfilters.dat') then
            if (use_standard_library(ctx, 'filters', 'allfilters')) then
                call datalib_open(lib, ctx%sps_home, 'filters', 'allfilters')
                ctx%state%nbands = datalib_axis_len(lib, 'filter')
            end if
        end if
        ctx%state%nindx = 30
        ctx%state%ntfull = TIME_RES_INCR*ctx%state%nt

        ctx%state%nspec_xrb = NSPEC_XRB
        ctx%state%nt_xrb = NT_XRB
        ctx%state%nz_xrb = NZ_XRB

        if (ctx%state%isoc_type == 'bpss' .and. TIME_RES_INCR /= 1) then
            write (error_unit, '(A)') '[FSPS_INIT] Error: TIME_RES_INCR>1 not supported for BPASS.'
            error stop 1
        end if

    end subroutine set_simulation_dimensions

    !> @brief Splits 'name:axis=value,...' into the library name and its axis selections.
    subroutine split_selection(full, name, sel)
        character(len=*), intent(in) :: full
        character(len=*), intent(out) :: name, sel
        integer :: k
        k = index(full, ':')
        if (k > 0) then
            name = full(1:k - 1)
            sel = full(k + 1:)
        else
            name = full
            sel = ''
        end if
    end subroutine split_selection

    !> @brief True if the isochrones (or, for BPASS, the SSP library) are read in the standard format.
    logical function uses_standard_isochrones(ctx)
        type(fsps_context_t), intent(in) :: ctx
        uses_standard_isochrones = datalib_available(ctx%sps_home, isochrone_kind(ctx%state%isoc_type), &
                                                     ctx%state%isoc_type, ctx%data_format)
    end function uses_standard_isochrones

    !> @brief nt, nz and the solar metallicity of a standard-format isochrone (or SSP) library.
    subroutine isochrone_dimensions_from_manifest(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        type(datalib_t) :: lib

        call datalib_open(lib, ctx%sps_home, isochrone_kind(ctx%state%isoc_type), &
                          library_spec(ctx, isochrone_kind(ctx%state%isoc_type), ctx%state%isoc_type))
        ctx%state%zsol = datalib_meta_real(lib, 'solar_metallicity')
        ctx%state%nz = datalib_axis_len(lib, metallicity_axis(lib))
        if (ctx%state%isoc_type == 'bpss') then
            ctx%state%nt = datalib_axis_len(lib, 'log_age')
            ! The SSP library is also the spectral library
            ctx%state%zsol_spec = ctx%state%zsol
            ctx%state%nzinit = 1
            ctx%state%nspec = datalib_axis_len(lib, 'wavelength')
        else
            ctx%state%nt = datalib_axis_len(lib, 'age_index')
        end if
    end subroutine isochrone_dimensions_from_manifest

    !> @brief nspec, nzinit and the solar metallicity of a standard-format spectral library.
    subroutine spectral_dimensions_from_manifest(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        type(datalib_t) :: lib

        call datalib_open(lib, ctx%sps_home, 'stellar_spectra', &
                          library_spec(ctx, 'stellar_spectra', ctx%state%spec_type))
        ctx%state%zsol_spec = datalib_meta_real(lib, 'solar_metallicity')
        ctx%state%nzinit = datalib_axis_len(lib, metallicity_axis(lib))
        ctx%state%nspec = datalib_axis_len(lib, 'wavelength')
        ! The (log T, log g) grid sizes are fixed by the context's lookup tables
        if (datalib_axis_len(lib, 'logT') /= NDIM_LOGT .or. datalib_axis_len(lib, 'logg') /= NDIM_LOGG) then
            write (error_unit, '(A,I0,A,I0,A)') '[FSPS_INIT] Error: spectral library ' // trim(ctx%state%spec_type) // &
                ' must be on the ', NDIM_LOGT, ' x ', NDIM_LOGG, ' (log T, log g) grid of this version'
            error stop 1
        end if
    end subroutine spectral_dimensions_from_manifest

    !> @brief Allocates the massive shared arrays within the cache object.
    !>
    !> @details
    !> Uses dimensions stored in `ctx%state` (NT, NZ, NSPEC) and global constants
    !> to allocate the shared physics grids.
    !>
    !> **Allocation Strategy:**
    !> 1. **Isochrones:** Allocated using `NT` and `NZ`.
    !> 2. **Auxiliary Stars:** Allocated using `NSPEC` and `fsps_constants` (WMB, AGB, WR).
    !> 3. **Interstellar Physics:** Allocated using `fsps_constants` (Nebular, Dust).
    !> 4. **Spectral Library:** The main `speclib` is *not* allocated here; it is
    !>    deferred to `read_spectral_binary` which reads the grid dimensions (T, g)
    !>    directly from the file header.
    !>
    !> @param[inout] cache The cache object to allocate.
    !> @param[in]    ctx   The context containing dimension info (nt, nz, nspec).
    subroutine allocate_shared_memory(cache, ctx)
        type(fsps_setup_cache_t), intent(inout) :: cache
        type(fsps_context_t), intent(in) :: ctx

        integer :: nt, nz, nspec, ntfull
        integer :: nspec_xrb, nt_xrb, nz_xrb
        integer :: ndim_dustem, numin_dustem, nqpah_dustem

        nt = ctx%state%nt
        nz = ctx%state%nz
        nspec = ctx%state%nspec
        ntfull = ctx%state%ntfull
        nspec_xrb = ctx%state%nspec_xrb
        nt_xrb = ctx%state%nt_xrb
        nz_xrb = ctx%state%nz_xrb
        ndim_dustem = ctx%state%ndim_dustem
        numin_dustem = ctx%state%numin_dustem
        nqpah_dustem = ctx%state%nqpah_dustem

        allocate (cache%indexdefined(7, ctx%state%nindx))
        allocate (cache%wgdust(nspec, 18, 6, 2))
        allocate (cache%g03smcextn(nspec))
        allocate (cache%bands(nspec, ctx%state%nbands))
        allocate (cache%magsun(ctx%state%nbands), cache%magvega(ctx%state%nbands), &
                  cache%filter_leff(ctx%state%nbands))
        allocate (cache%vega_spec(nspec), cache%sun_spec(nspec))
        allocate (cache%spec_lambda(nspec), cache%spec_nu(nspec))
        allocate (cache%spec_res(nspec))
        allocate (cache%speclib(nspec, nz, NDIM_LOGT, NDIM_LOGG))
        allocate (cache%wmb_spec(nspec, nz, NDIM_WMB_LOGT, NDIM_WMB_LOGG))
        allocate (cache%agb_spec_o(nspec, N_AGB_O))
        allocate (cache%agb_logt_o(nz, N_AGB_O))
        allocate (cache%agb_spec_c(nspec, N_AGB_C))
        allocate (cache%agb_logt_c(N_AGB_C))
        allocate (cache%agb_spec_car(nspec, N_AGB_CAR))
        allocate (cache%pagb_spec(nspec, NDIM_PAGB, 2))
        allocate (cache%wrn_spec(nspec, NDIM_WR, nz), cache%wrc_spec(nspec, NDIM_WR, nz))

        allocate (cache%qpaharr(nqpah_dustem))
        allocate (cache%uminarr(numin_dustem))
        allocate (cache%lambda_dustem(ndim_dustem))
        allocate (cache%dustem_dustem(ndim_dustem, numin_dustem*2))
        allocate (cache%dustem2_dustem(nspec, nqpah_dustem, numin_dustem*2))
        allocate (cache%flux_dagb(nspec, 2, NTEFF_DAGB, NTAU_DAGB))
        allocate (cache%nebem_cont(nspec, NEBNZ, NEBNAGE, NEBNIP), &
                  cache%xnebem_cont(nspec, NEBNZ, NEBNAGE, NEBNIP))
        allocate (cache%neb_res_min(nspec))
        allocate (cache%gaussnebarr(nspec, NEMLINE))
        allocate (cache%agndust_spec(nspec, NAGNDUST))
        allocate (cache%mact_isoc(nz, nt, NM), cache%logl_isoc(nz, nt, NM), &
                  cache%logt_isoc(nz, nt, NM), cache%logg_isoc(nz, nt, NM))
        allocate (cache%ffco_isoc(nz, nt, NM), cache%phase_isoc(nz, nt, NM), &
                  cache%mini_isoc(nz, nt, NM), cache%lmdot_isoc(nz, nt, NM))
        allocate (cache%nmass_isoc(nz, nt))
        allocate (cache%timestep_isoc(nz, nt))
        allocate (cache%zlegend(nz))
        allocate (cache%zlegendinit(ctx%state%nzinit))
        allocate (cache%time_full(ntfull))
        allocate (cache%bpass_spec_ssp(nspec, nt, nz))
        allocate (cache%bpass_mass_ssp(nt, nz))
        allocate (cache%lam_xrb(nspec_xrb))
        allocate (cache%spec_xrb(nspec, nt_xrb, nz_xrb))
        allocate (cache%ages_xrb(nt_xrb))
        allocate (cache%zmet_xrb(nz_xrb))

    end subroutine allocate_shared_memory

    !> @brief Orchestrates I/O for Isochrones, Spectral Libraries, and Auxiliary Stars.
    !>
    !> @details
    !> This is the core loading routine for stellar physics. It handles:
    !> 1. **Isochrones:** Evolution tracks (MIST, Padova, etc.) via `read_isochrone_database`.
    !> 2. **Spectral Libraries:** Base stellar spectra (MILES, BaSeL, C3K) via `read_spectral_binary`.
    !> 3. **Auxiliary Spectra:** Special handling for O-stars (WMBasic),
    !>    Wolf-Rayet stars (CMFGEN), and AGB stars (Lancon & Wood).
    !>
    !> @param[inout] ctx   The context containing path configuration and model selections.
    !> @param[inout] cache The cache object to populate.
    subroutine load_stellar_data(ctx, zin)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin
        integer :: z, zmin, zmax, i, i1, j, k, nzinit
        real(WP) :: dz, log_spec_val
        real(WP), parameter :: LIBRARY_FLOOR = 1.0e-70_wp  ! legacy FSPS tiny_number
        real(WP), allocatable :: speclibinit(:, :, :, :), speclib_slice(:, :, :)

        call load_zlegend_file(ctx, ctx%state%isoc_type, .false.)

        if (zin <= 0) then
            zmin = 1
            zmax = ctx%state%nz
        else
            zmin = zin
            zmax = zin
        end if

        if (ctx%state%isoc_type == 'bpss') then
            call read_bpass_data(ctx)
            return
        end if

        call load_wavelength_grid(ctx, ctx%state%spec_type)
        call load_spectral_resolution(ctx, ctx%state%spec_type)
        call load_zlegend_file(ctx, ctx%state%spec_type, .true.)

        call load_spectral_grid_axes(ctx, ctx%state%spec_type)

        nzinit = ctx%state%nzinit
        allocate (speclibinit(ctx%state%nspec, nzinit, NDIM_LOGT, NDIM_LOGG))
        allocate (speclib_slice(ctx%state%nspec, NDIM_LOGT, NDIM_LOGG))
        speclibinit = 0.0_wp

        do z = 1, nzinit
            call read_spectral_binary(ctx, ctx%state%spec_type, z, speclib_slice)
            speclibinit(:, z, :, :) = speclib_slice
        end do

        do z = 1, ctx%state%nz
            i1 = min(max(find_interval(log10(ctx%state%zlegendinit/ctx%state%zsol_spec), &
                                       log10(ctx%state%zlegend(z)/ctx%state%zsol)), 1), nzinit - 1)
            dz = (log10(ctx%state%zlegend(z)/ctx%state%zsol) - &
                  log10(ctx%state%zlegendinit(i1)/ctx%state%zsol_spec))/ &
                 (log10(ctx%state%zlegendinit(i1 + 1)/ctx%state%zsol_spec) - &
                  log10(ctx%state%zlegendinit(i1)/ctx%state%zsol_spec))
            dz = min(max(dz, 0.0_wp), 1.0_wp)

            ! Interpolate log flux in metallicity with legacy FSPS's 1e-70 floor (tiny_number), so that
            ! a template present at only one library metallicity becomes ~1e-68 here, not ~1e-150 as
            ! with a 1e-308 floor (which also sends pow into its slow underflow path).
            do k = 1, NDIM_LOGG
                do j = 1, NDIM_LOGT
                    do i = 1, ctx%state%nspec
                        if (dz <= 0.0_wp) then
                            ctx%state%speclib(i, z, j, k) = speclibinit(i, i1, j, k)
                        else if (dz >= 1.0_wp) then
                            ctx%state%speclib(i, z, j, k) = speclibinit(i, i1 + 1, j, k)
                        else
                            log_spec_val = (1.0_wp - dz) * log10(speclibinit(i, i1, j, k) + LIBRARY_FLOOR) + &
                                           dz * log10(speclibinit(i, i1 + 1, j, k) + LIBRARY_FLOOR)
                            ctx%state%speclib(i, z, j, k) = 10.0_wp**log_spec_val
                        end if
                    end do
                end do
            end do
        end do

        deallocate (speclib_slice)
        deallocate (speclibinit)

        call load_wmbasic_spectra(ctx)
        call load_agb_spectra(ctx)
        call load_post_agb_spectra(ctx)
        call load_wr_spectra(ctx)

        do z = zmin, zmax
            call read_isochrone_database(ctx, ctx%state%isoc_type, z)
        end do

    end subroutine load_stellar_data

    !> @brief Sets the context's lower IMF integration bound.
    !>
    !> @details
    !> Context-local: depends on `imf_lower_limit`, which is not part of the cache key, so it is
    !> evaluated on every setup (cache hit or miss) from the shared isochrones.
    !> BPASS leaves the bound untouched, as before.
    subroutine compute_imf_lower_bound(ctx, zin)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin
        integer :: zmin

        if (ctx%state%isoc_type == 'bpss') return

        zmin = 1
        if (zin > 0) zmin = zin

        if (ctx%state%isoc_type == 'gnva') then
            ctx%state%imf_lower_bound = minval(ctx%state%mini_isoc(zmin, 1, 1:ctx%state%nmass_isoc(zmin, 1)))*0.99_wp
        else
            ctx%state%imf_lower_bound = ctx%state%imf_lower_limit
        end if
    end subroutine compute_imf_lower_bound

    !> @brief Orchestrates I/O for Dust Emission, Attenuation, and Nebular grids.
    !>
    !> @details
    !> Calls `fsps_io` routines to populate the shared cache with:
    !> 1. **Dust Emission:** Draine & Li (2007) or THEMIS IR models.
    !> 2. **Nebular Emission:** CLOUDY model grids for HII regions (lines and continuum).
    !> 3. **AGN Dust:** Nenkova et al. (2008) CLUMPY torus models.
    !> 4. **Attenuation:** Tabulated extinction curves (Witt & Gordon, SMC).
    !> 5. **X-Ray Binaries:** Theoretical XRB spectral grids.
    !>
    !> @param[inout] ctx   The context containing path configuration and model selections.
    !> @param[inout] cache The cache object to populate.
    subroutine load_interstellar_physics(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        call load_dust_emission_table(ctx, ctx%state%str_dustem)
        call load_dusty_agb_spectra(ctx)

        if (ctx%state%isoc_type == 'mist' .or. ctx%state%isoc_type == 'pdva' .or. &
            ctx%state%isoc_type == 'prsc' .or. ctx%state%isoc_type == 'bpss' .or. &
            (ctx%data_format /= 'legacy' .and. &
             datalib_available(ctx%sps_home, 'nebular', trim(ctx%state%isoc_type)//nebular_suffix(ctx), 'auto'))) then
            call load_nebular_grid(ctx, ctx%state%isoc_type, ctx%cloudy_dust_val == 1)
            call compute_nebular_kernels(ctx)
        end if

        if (ctx%state%isoc_type == 'bpss') then
            ! The grid photoionized with X-ray binaries included (xnebem_*); read like the
            ! regular grid, whose axes it shares
            call load_nebular_grid(ctx, ctx%state%isoc_type, ctx%cloudy_dust_val == 1, xray=.true.)
        end if

        ! X-ray binary spectra (11 x 10 tables read as text, ~2 s) are only used with
        ! add_xrb_emission = 1, which is part of the setup-cache key
        if (ctx%add_xrb_emission_val == 1) call load_xrb_spectra(ctx)

    end subroutine load_interstellar_physics

    !> @brief Loads Filter Curves, Solar/Vega SEDs, and Spectral Indices.
    !>
    !> @details
    !> Orchestrates the loading of photometric and calibration data via `fsps_io`.
    !>
    !> **Data Dependency Note:**
    !> The Solar and Vega spectra are loaded *before* the filters. This allows
    !> `load_filter_definitions` (in `fsps_io`) to integrate the transmission curves
    !> against these standards immediately upon loading, populating `magsun` and
    !> `magvega` without requiring permanent storage of the full transmission curves
    !> in the cache.
    !>
    !> @param[inout] ctx   The context containing path configuration.
    !> @param[inout] cache The cache object to populate.
    subroutine load_photometry_data(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (len_trim(ctx%state%alt_filter_file) == 0 .or. trim(ctx%state%alt_filter_file) == 'allfilters.dat') then
            call load_filter_definitions(ctx)
        else
            call load_filter_definitions(ctx, trim(ctx%state%alt_filter_file))
        end if

        call load_standard_sed(ctx)

        if (ctx%compute_vega_mags_val == 1) then
            call convert_to_vega_system(ctx)
        end if

        if (len_trim(ctx%state%alt_filter_file) == 0 .or. trim(ctx%state%alt_filter_file) == 'allfilters.dat') then
            call apply_legacy_filter_norm(ctx)
        end if

    end subroutine load_photometry_data

    !> @brief Computes data that depends on loaded physics but isn't read from files.
    !>
    !> @details
    !> Populates shared cache arrays that are derived from the primary loaded data.
    !>
    !> 1. **Frequency Grid (`spec_nu`):** Calculated from `spec_lambda`.
    !> 2. **Expanded Time Grid (`time_full`):** Interpolates the isochrone age grid
    !>    if `TIME_RES_INCR > 1`.
    !>
    !> @note
    !> This routine populates *Cached* data. Context-specific derived data (like
    !> `cosmospl` or `mwdindex`) which are not pointers in the context state must
    !> be computed in a separate local initialization step.
    !>
    !> @param[inout] ctx   The context (used for accessing constants/state).
    !> @param[inout] cache The cache object to populate.
    subroutine compute_derived_quantities(ctx, cache)
        type(fsps_context_t), intent(inout) :: ctx
        type(fsps_setup_cache_t), intent(inout) :: cache

        integer :: i, j, k
        integer :: nt_iso, nt_full
        real(WP) :: dt, t_start

        ! --------------------------------------------------------------------
        ! 1. COMPUTE FREQUENCY GRID (spec_nu)
        ! --------------------------------------------------------------------
        ! c / lambda (converted to appropriate units if necessary, but FSPS usually
        ! keeps standard units). Lambda is in Angstroms.
        ! C_LIGHT is typically in Ang/s (2.99792458e18).

        where (cache%spec_lambda > SAFE_FLOOR)
            cache%spec_nu = C_LIGHT/cache%spec_lambda
        elsewhere
            cache%spec_nu = 0.0_wp
        end where

        ! --------------------------------------------------------------------
        ! 2. COMPUTE EXTENDED TIME GRID (time_full)
        ! --------------------------------------------------------------------
        ! If TIME_RES_INCR > 1, we interpolate between the native isochrone
        ! time steps to create a finer grid for the SSP generation.

        nt_iso = ctx%state%nt
        nt_full = ctx%state%ntfull

        ! Use the first available metallicity (z=1) as the reference time grid.
        ! Isochrone libraries (MIST, Padova) enforce consistent age grids across Z.

        if (TIME_RES_INCR == 1) then
            ! Simple copy if no oversampling
            cache%time_full(1:nt_iso) = cache%timestep_isoc(1, 1:nt_iso)
        else
            ! Interpolate
            do i = 1, nt_iso - 1

                ! The start time of this isochrone interval
                t_start = cache%timestep_isoc(1, i)

                ! The time step size (delta t) between oversampled points
                dt = (cache%timestep_isoc(1, i + 1) - t_start)/real(TIME_RES_INCR, kind=WP)

                ! Fill the sub-steps
                do j = 0, TIME_RES_INCR - 1
                    k = (i - 1)*TIME_RES_INCR + j + 1
                    if (k > nt_full) exit
                    cache%time_full(k) = t_start + real(j, kind=WP)*dt
                end do

            end do

            ! Handle the final point
            cache%time_full(nt_full) = cache%timestep_isoc(1, nt_iso)
        end if

    end subroutine compute_derived_quantities

    !> @brief Computes local physical properties specific to this context instance.
    !>
    !> @details
    !> These quantities are stored in `fsps_context_state_t` but are *not* pointers,
    !> meaning they must be re-calculated for every new context (they cannot be
    !> linked from the cache).
    !>
    !> Calculations:
    !> 1. **MW Dust Indices (`mwdindex`):** Defines wavelength breakpoints for the
    !>    Cardelli, Clayton, & Mathis (1989) extinction curve.
    !> 2. **Cosmology Grid (`cosmospl`):** A lookup table for (Redshift, Age, Distance)
    !>    used for fast interpolation during runtime.
    !> 3. **Universe Age (`tuniv`):** Age at z=0.
    !> 4. **Key Indices:** `whlam5000` (5000A) and `whlylim` (912A).
    !>
    !> @param[inout] ctx The context to populate.
    subroutine compute_local_properties(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        integer :: i, j
        real(WP) :: x_inv_micron, a_scale
        real(WP), dimension(:), pointer :: lam

        ! Alias for readability
        lam => ctx%state%spec_lambda

        ! --------------------------------------------------------------------
        ! 1. MW Dust Extinction Indices (CCM89 Breakpoints)
        ! --------------------------------------------------------------------
        ! x = 10000 / lambda(A)
        ! Breaks at x = [0.1, 1.1, 3.3, 5.9, 8.0, 12.0]

        ctx%state%mwdindex = 0

        do j = 1, ctx%state%nspec
            if (lam(j) > SAFE_FLOOR) then
                x_inv_micron = 1.0e4_wp/lam(j)

                ! Legacy logic finds the *last* index where the condition holds
                ! assuming lambda is increasing (and x is decreasing).
                if (x_inv_micron >= 0.1_wp) ctx%state%mwdindex(1) = j
                if (x_inv_micron >= 1.1_wp) ctx%state%mwdindex(2) = j
                if (x_inv_micron >= 3.3_wp) ctx%state%mwdindex(3) = j
                if (x_inv_micron >= 5.9_wp) ctx%state%mwdindex(4) = j
                if (x_inv_micron >= 8.0_wp) ctx%state%mwdindex(5) = j
                if (x_inv_micron > 12.0_wp) ctx%state%mwdindex(6) = j
            end if
        end do

        ! --------------------------------------------------------------------
        ! 2. Cosmology Lookup Table (cosmospl)
        ! --------------------------------------------------------------------
        ! Generates a grid of 500 points scaling factor 'a' from ~0.001 to 1.0.
        ! Stores: [Redshift, Age(Gyr), Luminosity Distance(pc)]

        do i = 1, 500
            ! Scale factor 'a' distribution (legacy formula)
            a_scale = real(i - 1, kind=WP)/499.0_wp*(1.0_wp - 1.0_wp/1001.0_wp) + &
                      1.0_wp/1001.0_wp

            ! Col 1: Redshift (z = 1/a - 1)
            ctx%state%cosmospl(i, 1) = (1.0_wp/a_scale) - 1.0_wp

            ! Col 2: Age of Universe at z
            ctx%state%cosmospl(i, 2) = get_universe_age(ctx, ctx%state%cosmospl(i, 1))

            ! Col 3: Luminosity Distance at z
            ctx%state%cosmospl(i, 3) = get_luminosity_distance(ctx, ctx%state%cosmospl(i, 1))
        end do

        ! --------------------------------------------------------------------
        ! 3. Age of the Universe (z=0)
        ! --------------------------------------------------------------------
        ctx%state%tuniv = get_universe_age(ctx, 0.0_wp)

        ! --------------------------------------------------------------------
        ! 4. Key Wavelength Indices
        ! --------------------------------------------------------------------
        ! Find index for 5000 Angstroms
        ctx%state%whlam5000 = find_interval(lam, 5000.0_wp)

        ! Find index for Lyman Limit (912 Angstroms)
        ctx%state%whlylim = find_interval(lam, 912.0_wp)

        ctx%state%sfh_tab = 0.0_wp
        ctx%state%ntabsfh = 0

    end subroutine compute_local_properties

    !> @brief Recomputes the shared frequency grid from the final wavelength grid.
    !>
    !> @details
    !> Writes the shared `spec_nu`, so it runs only when a cache entry is populated.
    subroutine compute_frequency_grid(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        ctx%state%spec_nu = 0.0_wp
        where (ctx%state%spec_lambda > SAFE_FLOOR)
            ctx%state%spec_nu = C_LIGHT/ctx%state%spec_lambda
        end where
    end subroutine compute_frequency_grid

    !> @brief Clears the fixed-size per-context tables written by the shared-data loaders.
    !>
    !> @details
    !> Called before populating a new cache entry so the recorded tables never carry values
    !> left over from an earlier setup of this context with other libraries (e.g. the nebular
    !> tables for an isochrone set without a nebular grid).
    subroutine reset_setup_tables(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        ! Assigned field by field: a temporary fsps_setup_tables_t (~2 MB) could overflow a
        ! thread stack
        ctx%state%speclib_logt = 0.0_wp
        ctx%state%speclib_logg = 0.0_wp
        ctx%state%wmb_logt = 0.0_wp
        ctx%state%wmb_logg = 0.0_wp
        ctx%state%agb_logt_car = 0.0_wp
        ctx%state%pagb_logt = 0.0_wp
        ctx%state%wrn_logt = 0.0_wp
        ctx%state%wrc_logt = 0.0_wp
        ctx%state%tau1_dagb = 0.0_wp
        ctx%state%teff_dagb = 0.0_wp
        ctx%state%nebem_line_pos = 0.0_wp
        ctx%state%nebem_line = 0.0_wp
        ctx%state%xnebem_line = 0.0_wp
        ctx%state%nebem_logz = 0.0_wp
        ctx%state%nebem_age = 0.0_wp
        ctx%state%nebem_logu = 0.0_wp
        ctx%state%agndust_tau = 0.0_wp
    end subroutine reset_setup_tables

    !> @brief Copies a nebular line table (2 MB with xnebem_line).
    !>
    !> @details
    !> Through explicit-shape dummies, which may not alias, so the compiler copies in place; a
    !> plain assignment between components of a context and a cache entry made gfortran stage it
    !> through a stack temporary, overflowing the stack of callers that hold a context there.
    subroutine copy_line_table(src, dst)
        real(WP), intent(in) :: src(NEMLINE, NEBNZ, NEBNAGE, NEBNIP)
        real(WP), intent(out) :: dst(NEMLINE, NEBNZ, NEBNAGE, NEBNIP)

        dst = src
    end subroutine copy_line_table

    !> @brief Records the per-context tables produced while populating a cache entry.
    subroutine save_setup_tables(ctx, cache)
        type(fsps_context_t), intent(in) :: ctx
        type(fsps_setup_cache_t), intent(inout) :: cache

        if (.not. allocated(cache%tables)) allocate (cache%tables)
        cache%tables%speclib_logt = ctx%state%speclib_logt
        cache%tables%speclib_logg = ctx%state%speclib_logg
        cache%tables%wmb_logt = ctx%state%wmb_logt
        cache%tables%wmb_logg = ctx%state%wmb_logg
        cache%tables%agb_logt_car = ctx%state%agb_logt_car
        cache%tables%pagb_logt = ctx%state%pagb_logt
        cache%tables%wrn_logt = ctx%state%wrn_logt
        cache%tables%wrc_logt = ctx%state%wrc_logt
        cache%tables%tau1_dagb = ctx%state%tau1_dagb
        cache%tables%teff_dagb = ctx%state%teff_dagb
        cache%tables%nebem_line_pos = ctx%state%nebem_line_pos
        call copy_line_table(ctx%state%nebem_line, cache%tables%nebem_line)
        call copy_line_table(ctx%state%xnebem_line, cache%tables%xnebem_line)
        cache%tables%nebem_logz = ctx%state%nebem_logz
        cache%tables%nebem_age = ctx%state%nebem_age
        cache%tables%nebem_logu = ctx%state%nebem_logu
        cache%tables%agndust_tau = ctx%state%agndust_tau
    end subroutine save_setup_tables

    !> @brief Copies the per-context tables recorded in a populated cache entry into a context.
    subroutine restore_setup_tables(ctx, cache)
        type(fsps_context_t), intent(inout) :: ctx
        type(fsps_setup_cache_t), target, intent(in) :: cache
        type(fsps_setup_tables_t), pointer :: tables

        if (.not. allocated(cache%tables)) then
            write (error_unit, '(A)') '[FSPS_INIT] Error: cache entry has no recorded setup tables'
            error stop 1
        end if
        tables => cache%tables

        ctx%state%speclib_logt = tables%speclib_logt
        ctx%state%speclib_logg = tables%speclib_logg
        ctx%state%wmb_logt = tables%wmb_logt
        ctx%state%wmb_logg = tables%wmb_logg
        ctx%state%agb_logt_car = tables%agb_logt_car
        ctx%state%pagb_logt = tables%pagb_logt
        ctx%state%wrn_logt = tables%wrn_logt
        ctx%state%wrc_logt = tables%wrc_logt
        ctx%state%tau1_dagb = tables%tau1_dagb
        ctx%state%teff_dagb = tables%teff_dagb
        ctx%state%nebem_line_pos = tables%nebem_line_pos
        call copy_line_table(tables%nebem_line, ctx%state%nebem_line)
        call copy_line_table(tables%xnebem_line, ctx%state%xnebem_line)
        ctx%state%nebem_logz = tables%nebem_logz
        ctx%state%nebem_age = tables%nebem_age
        ctx%state%nebem_logu = tables%nebem_logu
        ctx%state%agndust_tau = tables%agndust_tau
    end subroutine restore_setup_tables

    !> @brief Maps the pointers in `ctx%state` to the arrays in `cache`.
    !>
    !> @details
    !> Performs two types of linking:
    !> 1. **Pointer Association (`=>`):** For large dynamic arrays (spectra, isochrones).
    !>    This allows multiple contexts to share read-only memory.
    !> 2. **Value Copy (`=`):** For small fixed-size arrays (grid axes like `logt`, `logg`).
    !>    This ensures that contexts initialized from the cache still have valid
    !>    axis definitions, fixing a potential issue in the legacy architecture.
    !>
    !> @param[inout] ctx   The context to populate.
    !> @param[in]    cache The shared cache object containing the loaded data.
    subroutine link_context_to_cache(ctx, cache)
        type(fsps_context_t), intent(inout) :: ctx
        type(fsps_setup_cache_t), target, intent(in) :: cache

        ctx%state%indexdefined => cache%indexdefined
        ctx%state%wgdust => cache%wgdust
        ctx%state%g03smcextn => cache%g03smcextn
        ctx%state%bands => cache%bands
        ctx%state%magsun => cache%magsun
        ctx%state%magvega => cache%magvega
        ctx%state%filter_leff => cache%filter_leff
        ctx%state%vega_spec => cache%vega_spec
        ctx%state%sun_spec => cache%sun_spec
        ctx%state%spec_lambda => cache%spec_lambda
        ctx%state%spec_nu => cache%spec_nu
        ctx%state%spec_res => cache%spec_res
        ctx%state%speclib => cache%speclib
        ctx%state%wmb_spec => cache%wmb_spec
        ctx%state%agb_spec_o => cache%agb_spec_o
        ctx%state%agb_logt_o => cache%agb_logt_o
        ctx%state%agb_spec_c => cache%agb_spec_c
        ctx%state%agb_logt_c => cache%agb_logt_c
        ctx%state%agb_spec_car => cache%agb_spec_car
        if (associated(cache%speclib_agb_smooth)) ctx%state%speclib_agb_smooth => cache%speclib_agb_smooth
        if (associated(cache%agb_spec_o_smooth)) ctx%state%agb_spec_o_smooth => cache%agb_spec_o_smooth
        if (associated(cache%agb_spec_c_smooth)) ctx%state%agb_spec_c_smooth => cache%agb_spec_c_smooth
        if (associated(cache%agb_spec_car_smooth)) ctx%state%agb_spec_car_smooth => cache%agb_spec_car_smooth
        ctx%state%pagb_spec => cache%pagb_spec
        ctx%state%wrn_spec => cache%wrn_spec
        ctx%state%wrc_spec => cache%wrc_spec
        ctx%state%qpaharr => cache%qpaharr
        ctx%state%uminarr => cache%uminarr
        ctx%state%lambda_dustem => cache%lambda_dustem
        ctx%state%dustem_dustem => cache%dustem_dustem
        ctx%state%dustem2_dustem => cache%dustem2_dustem
        ctx%state%flux_dagb => cache%flux_dagb
        ctx%state%nebem_cont => cache%nebem_cont
        ctx%state%xnebem_cont => cache%xnebem_cont
        ctx%state%neb_res_min => cache%neb_res_min
        if (ctx%setup_nebular_gaussians_val == 1) then
            ! Kernels fixed at setup: read-only after setup, safe to share
            ctx%state%gaussnebarr => cache%gaussnebarr
        else
            ! Kernels rebuilt on every nebular call from this context's sigma_smooth: the array
            ! must be private, or concurrent contexts overwrite each other's line profiles
            allocate (ctx%state%gaussnebarr(size(cache%gaussnebarr, 1), size(cache%gaussnebarr, 2)))
            ctx%state%gaussnebarr = 0.0_wp
            ctx%state%owns_gaussnebarr = .true.
            ctx%state%gauss_built_sigma = -1.0_wp
        end if
        ctx%state%agndust_spec => cache%agndust_spec
        ctx%state%mact_isoc => cache%mact_isoc
        ctx%state%logl_isoc => cache%logl_isoc
        ctx%state%logt_isoc => cache%logt_isoc
        ctx%state%logg_isoc => cache%logg_isoc
        ctx%state%ffco_isoc => cache%ffco_isoc
        ctx%state%phase_isoc => cache%phase_isoc
        ctx%state%mini_isoc => cache%mini_isoc
        ctx%state%lmdot_isoc => cache%lmdot_isoc
        ctx%state%nmass_isoc => cache%nmass_isoc
        ctx%state%timestep_isoc => cache%timestep_isoc
        ctx%state%zlegend => cache%zlegend
        ctx%state%zlegendinit => cache%zlegendinit
        ctx%state%time_full => cache%time_full
        ctx%state%bpass_spec_ssp => cache%bpass_spec_ssp
        ctx%state%bpass_mass_ssp => cache%bpass_mass_ssp
        ctx%state%lam_xrb => cache%lam_xrb
        ctx%state%spec_xrb => cache%spec_xrb
        ctx%state%ages_xrb => cache%ages_xrb
        ctx%state%zmet_xrb => cache%zmet_xrb

    end subroutine link_context_to_cache

    !> @brief Allocates arrays that are specific to this context instance.
    !>
    !> @details
    !> Allocates working buffers for SSP generation, spectral smoothing, and intermediate results.
    !> Unlike the cached physics arrays, these arrays are mutable and specific to the
    !> current simulation state.
    !>
    !> Allocates:
    !> - `spec_ssp_zz`: The computed SSP grid [nspec, ntfull, nz].
    !> - `mass_ssp_zz`, `lbol_ssp_zz`: Stellar mass and luminosity grids.
    !> - `weight_ssp`: Weights for interpolation.
    !> - `spec_young`, `spec_old`: Buffers for separating young/old stellar populations.
    !> - `lsfinfo%lsf`: Buffer for the Line Spread Function.
    !>
    !> @param[inout] ctx The FSPS context to allocate.
    subroutine allocate_local_workspace(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        ! Clean up existing allocations if this context is being re-initialized
        if (allocated(ctx%state%spec_ssp_zz)) deallocate (ctx%state%spec_ssp_zz)
        if (allocated(ctx%state%mass_ssp_zz)) deallocate (ctx%state%mass_ssp_zz)
        if (allocated(ctx%state%lbol_ssp_zz)) deallocate (ctx%state%lbol_ssp_zz)
        if (allocated(ctx%state%weight_ssp)) deallocate (ctx%state%weight_ssp)
        if (allocated(ctx%state%spec_young)) deallocate (ctx%state%spec_young)
        if (allocated(ctx%state%spec_old)) deallocate (ctx%state%spec_old)
        if (allocated(ctx%state%lsfinfo%lsf)) deallocate (ctx%state%lsfinfo%lsf)

        ! Allocate SSP Grids
        ! Dimensions: (Wavelengths, Time Steps, Metallicity)
        allocate (ctx%state%spec_ssp_zz(ctx%state%nspec, ctx%state%ntfull, ctx%state%nz))
        allocate (ctx%state%mass_ssp_zz(ctx%state%ntfull, ctx%state%nz))
        allocate (ctx%state%lbol_ssp_zz(ctx%state%ntfull, ctx%state%nz))

        ! Allocate Weighting Buffer
        allocate (ctx%state%weight_ssp(ctx%state%ntfull, ctx%state%nz))

        ! Allocate Component Spectral Buffers
        allocate (ctx%state%spec_young(ctx%state%nspec, ctx%state%ntfull))
        allocate (ctx%state%spec_old(ctx%state%nspec, ctx%state%ntfull))

        ! Allocate Line Spread Function Buffer
        allocate (ctx%state%lsfinfo%lsf(ctx%state%nspec))

        ! Initialize to zero to prevent garbage data
        ctx%state%spec_ssp_zz = 0.0_wp
        ctx%state%mass_ssp_zz = 0.0_wp
        ctx%state%lbol_ssp_zz = 0.0_wp
        ctx%state%weight_ssp = 0.0_wp
        ctx%state%spec_young = 0.0_wp
        ctx%state%spec_old = 0.0_wp
        ctx%state%lsfinfo%lsf = 0.0_wp

    end subroutine allocate_local_workspace

    ! =====================================================================
    ! SUPPORT HELPERS
    ! =====================================================================

    subroutine normalize_library_names(isoc_in, spec_in, dust_in, isoc_out, spec_out, dust_out)
        character(len=*), intent(in), optional :: isoc_in
        character(len=*), intent(in), optional :: spec_in
        character(len=*), intent(in), optional :: dust_in
        character(len=64), intent(out) :: isoc_out
        character(len=64), intent(out) :: spec_out
        character(len=64), intent(out) :: dust_out
        character(len=64) :: iso, spec, dust

        iso = 'mist'
        spec = 'miles'
        dust = 'DL07'

        if (present(isoc_in)) then
            if (len_trim(isoc_in) > 0) iso = to_lower(trim(isoc_in))
        end if
        if (present(spec_in)) then
            if (len_trim(spec_in) > 0) spec = to_lower(trim(spec_in))
        end if
        if (present(dust_in)) then
            if (len_trim(dust_in) > 0) dust = trim(dust_in)
        end if

        select case (iso)
        case ('padova', 'padova2007', 'default')
            iso = 'pdva'
        case ('parsec')
            iso = 'prsc'
        case ('basti')
            iso = 'bsti'
        case ('geneva')
            iso = 'gnva'
        case ('bpass', 'bpss')
            iso = 'bpss'
        case default
            ! keep as-is
        end select

        if (iso == 'bpss') then
            spec = 'bpass'
        end if

        select case (spec)
        case ('basel', 'basel2.2', 'basel3.1')
            spec = 'basel'
        case ('miles')
            spec = 'miles'
        case ('bpass')
            spec = 'bpass'
        case default
            if (len_trim(spec) >= 3) then
                if (spec(1:3) == 'c3k') spec = trim(spec)
            end if
        end select

        if (trim(dust) == 'themis') dust = 'THEMIS'
        if (trim(dust) /= 'THEMIS') dust = 'DL07'

        isoc_out = iso
        spec_out = spec
        dust_out = dust
    end subroutine normalize_library_names

    subroutine configure_dust_model(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        type(datalib_t) :: lib

        if (use_standard_library(ctx, 'dust_emission', ctx%state%str_dustem)) then
            call datalib_open(lib, ctx%sps_home, 'dust_emission', ctx%state%str_dustem)
            ctx%state%ndim_dustem = datalib_axis_len(lib, 'wavelength')
            ctx%state%numin_dustem = datalib_axis_len(lib, 'umin')
            ctx%state%nqpah_dustem = datalib_axis_len(lib, 'qpah')
        else if (trim(ctx%state%str_dustem) == 'THEMIS') then
            ctx%state%ndim_dustem = 576
            ctx%state%numin_dustem = 37
            ctx%state%nqpah_dustem = 11
        else
            ctx%state%str_dustem = 'DL07'
            ctx%state%ndim_dustem = 1001
            ctx%state%numin_dustem = 22
            ctx%state%nqpah_dustem = 7
        end if
    end subroutine configure_dust_model

    subroutine build_cache_key(ctx, zin, cache_key)
        type(fsps_context_t), intent(in) :: ctx
        integer, intent(in) :: zin
        character(len=*), intent(out) :: cache_key

        ! The key must contain every setting the shared-data loaders depend on, because a context
        ! that hits an entry reuses its data without reloading:
        !   - nebular_smooth_init and smooth_velocity set the shared setup-time line kernels
        !     (setup_nebular_gaussians = 1);
        !   - cloudy_dust selects the nebular grid files (ZAU_WD_* vs ZAU_ND_*);
        !   - compute_vega_mags converts the shared magsun to the Vega system;
        !   - zin selects the isochrones loaded and the reference time grid.
        write (cache_key, '(A,"|",A,"|",A,"|",A,"|",A,"|",I0,"|",I0,"|",I0,"|",I0,"|",I0,"|",I0,"|",I0,"|",I0,"|",I0,"|",A, &
                           &"|",ES24.16,"|",I0,"|",I0)') &
            trim(ctx%sps_home), trim(ctx%data_format), trim(ctx%state%isoc_type)//':'//trim(ctx%state%isoc_select), &
            trim(ctx%state%spec_type)//':'//trim(ctx%state%spec_select), &
            trim(ctx%state%str_dustem), &
            zin, ctx%smooth_velocity_val, ctx%setup_nebular_gaussians_val, ctx%add_neb_emission_val, &
            ctx%add_neb_continuum_val, ctx%add_dust_emission_val, ctx%add_agn_dust_val, &
            ctx%add_xrb_emission_val, ctx%add_agb_dust_model_val, trim(ctx%state%alt_filter_file), &
            ctx%nebular_smooth_init_val, ctx%cloudy_dust_val, ctx%compute_vega_mags_val
    end subroutine build_cache_key

    subroutine populate_cache_metadata(ctx, cache)
        type(fsps_context_t), intent(in) :: ctx
        type(fsps_setup_cache_t), intent(inout) :: cache

        cache%isoc_type = trim(ctx%state%isoc_type)
        cache%spec_type = trim(ctx%state%spec_type)
        cache%dust_type = trim(ctx%state%str_dustem)
        cache%alt_filter_file = ctx%state%alt_filter_file
        cache%nz = ctx%state%nz
        cache%nt = ctx%state%nt
        cache%nspec = ctx%state%nspec
        cache%nzinit = ctx%state%nzinit
        cache%nbands = ctx%state%nbands
        cache%nindx = ctx%state%nindx
        cache%ntfull = ctx%state%ntfull
        cache%nspec_xrb = ctx%state%nspec_xrb
        cache%nt_xrb = ctx%state%nt_xrb
        cache%nz_xrb = ctx%state%nz_xrb
        cache%smooth_velocity = ctx%smooth_velocity_val
        cache%setup_nebular_gaussians = ctx%setup_nebular_gaussians_val
        cache%add_neb_emission = ctx%add_neb_emission_val
        cache%add_neb_continuum = ctx%add_neb_continuum_val
        cache%add_dust_emission = ctx%add_dust_emission_val
        cache%add_agn_dust = ctx%add_agn_dust_val
        cache%add_xrb_emission = ctx%add_xrb_emission_val
        cache%add_agb_dust_model = ctx%add_agb_dust_model_val
        cache%use_wr_spectra = ctx%use_wr_spectra_val
    end subroutine populate_cache_metadata

    subroutine initialize_cache_arrays(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        ctx%state%bands = 0.0_wp
        ctx%state%magsun = 0.0_wp
        ctx%state%magvega = 0.0_wp
        ctx%state%filter_leff = 0.0_wp
        ctx%state%vega_spec = 0.0_wp
        ctx%state%sun_spec = 0.0_wp
        ctx%state%spec_lambda = 0.0_wp
        ctx%state%spec_nu = 0.0_wp
        ctx%state%spec_res = 0.0_wp
        ctx%state%speclib = 0.0
        ctx%state%wmb_spec = 0.0
        ctx%state%agb_spec_o = 0.0_wp
        ctx%state%agb_logt_o = 0.0_wp
        ctx%state%agb_spec_c = 0.0_wp
        ctx%state%agb_logt_c = 0.0_wp
        ctx%state%agb_spec_car = 0.0_wp
        ctx%state%pagb_spec = 0.0_wp
        ctx%state%wrn_spec = 0.0_wp
        ctx%state%wrc_spec = 0.0_wp

        if (trim(ctx%state%str_dustem) == 'THEMIS') then
         ctx%state%qpaharr = (/0.02_wp, 0.06_wp, 0.10_wp, 0.14_wp, 0.17_wp, 0.20_wp, 0.24_wp, 0.28_wp, 0.32_wp, 0.36_wp, 0.40_wp/) &
                                /2.2_wp*100.0_wp
        ctx%state%uminarr = (/0.1_wp, 0.12_wp, 0.15_wp, 0.17_wp, 0.2_wp, 0.25_wp, 0.3_wp, 0.35_wp, 0.4_wp, 0.5_wp, 0.6_wp, 0.7_wp, &
                           0.8_wp, 1.0_wp, 1.2_wp, 1.5_wp, 1.7_wp, 2.0_wp, 2.5_wp, 3.0_wp, 3.5_wp, 4.0_wp, 5.0_wp, 6.0_wp, 7.0_wp, &
                         8.0_wp, 10.0_wp, 12.0_wp, 15.0_wp, 17.0_wp, 20.0_wp, 25.0_wp, 30.0_wp, 35.0_wp, 40.0_wp, 50.0_wp, 80.0_wp/)
        else
            ctx%state%qpaharr = (/0.47_wp, 1.12_wp, 1.77_wp, 2.50_wp, 3.19_wp, 3.90_wp, 4.58_wp/)
            ctx%state%uminarr = (/0.1_wp, 0.15_wp, 0.2_wp, 0.3_wp, 0.4_wp, 0.5_wp, 0.7_wp, 0.8_wp, 1.0_wp, 1.2_wp, 1.5_wp, 2.0_wp, &
                                  2.5_wp, 3.0_wp, 4.0_wp, 5.0_wp, 7.0_wp, 8.0_wp, 12.0_wp, 15.0_wp, 20.0_wp, 25.0_wp/)
        end if

        ctx%state%lambda_dustem = 0.0_wp
        ctx%state%dustem_dustem = 0.0_wp
        ctx%state%dustem2_dustem = 0.0_wp
        ctx%state%flux_dagb = 0.0_wp
        ctx%state%nebem_cont = 0.0_wp
        ctx%state%xnebem_cont = 0.0_wp
        ctx%state%neb_res_min = 0.0_wp
        ctx%state%gaussnebarr = 0.0_wp
        ctx%state%agndust_spec = 0.0_wp
        ctx%state%mact_isoc = 0.0_wp
        ctx%state%logl_isoc = 0.0_wp
        ctx%state%logt_isoc = 0.0_wp
        ctx%state%logg_isoc = 0.0_wp
        ctx%state%ffco_isoc = 0.0_wp
        ctx%state%phase_isoc = 0.0_wp
        ctx%state%mini_isoc = 0.0_wp
        ctx%state%lmdot_isoc = 0.0_wp
        ctx%state%nmass_isoc = 0
        ctx%state%timestep_isoc = 0.0_wp
        ctx%state%zlegend = -99.0_wp
        ctx%state%zlegendinit = -99.0_wp
        ctx%state%time_full = 0.0_wp
        ctx%state%bpass_spec_ssp = 0.0_wp
        ctx%state%bpass_mass_ssp = 0.0_wp
        ctx%state%lam_xrb = 0.0_wp
        ctx%state%spec_xrb = 0.0_wp
        ctx%state%ages_xrb = 0.0_wp
        ctx%state%zmet_xrb = 0.0_wp
        ctx%state%indexdefined = 0.0_wp
        ctx%state%wgdust = 0.0_wp
        ctx%state%g03smcextn = 0.0_wp
        ctx%state%sfh_tab = 0.0_wp
        ctx%state%ntabsfh = 0
    end subroutine initialize_cache_arrays

    subroutine compute_filter_leff(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        integer :: i
        real(WP) :: d

        do i = 1, ctx%state%nbands
            ctx%state%filter_leff(i) = integrate_trapezoid_array(ctx%state%spec_lambda, &
                                                                 ctx%state%spec_lambda*ctx%state%bands(:, i))
            d = integrate_trapezoid_array(ctx%state%spec_lambda, ctx%state%bands(:, i)/ctx%state%spec_lambda)
            if (ctx%state%filter_leff(i) /= ctx%state%filter_leff(i) .or. d /= d .or. d <= SAFE_FLOOR) then
                ctx%state%filter_leff(i) = 0.0_wp
            else
                ctx%state%filter_leff(i) = sqrt(ctx%state%filter_leff(i)/d)
            end if
        end do
    end subroutine compute_filter_leff

    subroutine build_time_grid(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        integer :: i
        real(WP) :: d1
        integer :: zref

        if (ctx%state%isoc_type == 'bpss') return

        zref = 1
        if (ctx%zin >= 1 .and. ctx%zin <= ctx%state%nz) then
            zref = ctx%zin
        end if
        do i = 1, ctx%state%ntfull
            if (mod(i - 1, TIME_RES_INCR) == 0) then
                ctx%state%time_full(i) = ctx%state%timestep_isoc(zref, (i - 1)/TIME_RES_INCR + 1)
            else
                if ((i - 1)/TIME_RES_INCR + 2 < ctx%state%nt) then
                    d1 = (ctx%state%timestep_isoc(zref, (i - 1)/TIME_RES_INCR + 2) - &
                          ctx%state%timestep_isoc(zref, (i - 1)/TIME_RES_INCR + 1))/TIME_RES_INCR
                end if
                ctx%state%time_full(i) = ctx%state%time_full(i - 1) + d1
            end if
        end do
    end subroutine build_time_grid

    subroutine compute_nebular_kernels(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        integer :: i, j
        real(WP) :: dlam

        do i = 1, NEMLINE
            j = min(max(find_interval(ctx%state%spec_lambda, ctx%state%nebem_line_pos(i)), 1), ctx%state%nspec - 1)
            ctx%state%neb_res_min(i) = ctx%state%spec_lambda(j + 1) - ctx%state%spec_lambda(j)
        end do

        if (ctx%setup_nebular_gaussians_val == 1) then
            do i = 1, NEMLINE
                if (ctx%smooth_velocity_val == 1) then
                    dlam = ctx%state%nebem_line_pos(i)*ctx%nebular_smooth_init_val/C_LIGHT*1.0e13_wp
                else
                    dlam = ctx%nebular_smooth_init_val
                end if
                dlam = max(dlam, ctx%state%neb_res_min(i)*2.0_wp)
                ctx%state%gaussnebarr(:, i) = 1.0_wp/sqrt(2.0_wp*PI)/dlam* &
                                              exp(-(ctx%state%spec_lambda - ctx%state%nebem_line_pos(i))**2/(2.0_wp*dlam**2))/ &
                                              C_LIGHT*ctx%state%nebem_line_pos(i)**2
            end do
        end if
    end subroutine compute_nebular_kernels

    subroutine load_dusty_agb_spectra(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        integer :: i, j, k, nlam, stat, u
        integer :: jj, i_spec
        real(WP), allocatable :: lambda_dagb(:), fluxin(:, :, :)   ! (lambda, tau, Teff)
        character(len=5), parameter :: CHEM(2) = ['orich', 'crich']
        character(len=5), parameter :: LEGACY(2) = ['Orich', 'Crich']
        type(datalib_t) :: lib
        integer(int64) :: off

        do k = 1, 2
            if (use_standard_library(ctx, 'dusty_agb', CHEM(k))) then
                call datalib_open(lib, ctx%sps_home, 'dusty_agb', CHEM(k))
                call datalib_axis_values(lib, 'wavelength', lambda_dagb)
                nlam = size(lambda_dagb)
                call standard_axis_values(lib, 'teff', ctx%state%teff_dagb(k, :))
                call standard_axis_values(lib, 'tau', ctx%state%tau1_dagb(k, :))
                allocate (fluxin(nlam, NTAU_DAGB, NTEFF_DAGB))
                call datalib_check_array(lib, 'transfer', [character(len=10) :: 'teff', 'tau', 'wavelength'], &
                                         [NTEFF_DAGB, NTAU_DAGB, nlam], off)
                call datalib_read_real(lib, 'transfer', size(fluxin, kind=int64), fluxin, off)
            else
                open (newunit=u, file=trim(ctx%sps_home)//'/data/dust/dusty/'//LEGACY(k)//'_dusty.spec', &
                      status='old', action='read', iostat=stat)
                if (stat /= 0) then
                    write (error_unit, '(A)') '[FSPS_INIT] Error: dusty '//LEGACY(k)//' models missing'
                    error stop 1
                end if
                read (u, *) nlam
                allocate (lambda_dagb(nlam), fluxin(nlam, NTAU_DAGB, NTEFF_DAGB))
                read (u, *) lambda_dagb
                do i = 1, NTEFF_DAGB
                    do j = 1, NTAU_DAGB
                        read (u, *) ctx%state%teff_dagb(k, i), ctx%state%tau1_dagb(k, j)
                        read (u, *) fluxin(:, j, i)
                    end do
                end do
                close (u)
            end if

            jj = max(find_interval(ctx%state%spec_lambda, lambda_dagb(1)), 1)
            do i = 1, NTEFF_DAGB
                do j = 1, NTAU_DAGB
                    do i_spec = jj, ctx%state%nspec
                        ctx%state%flux_dagb(i_spec, k, i, j) = interpolate_linear(lambda_dagb, fluxin(:, j, i), &
                                                                                  ctx%state%spec_lambda(i_spec))
                    end do
                end do
            end do
            deallocate (lambda_dagb, fluxin)
        end do

    end subroutine load_dusty_agb_spectra

    !> @brief Copies the coordinates of axis `axis` of an open library into `dest` (same length).
    subroutine standard_axis_values(lib, axis, dest)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: axis
        real(WP), intent(out) :: dest(:)
        real(WP), allocatable :: vals(:)
        call datalib_axis_values(lib, axis, vals)
        if (size(vals) /= size(dest)) then
            write (error_unit, '(A,I0,A,I0)') '[FSPS_INIT] Error: axis '//axis//' of '//lib%kind//'/'//lib%name// &
                ' has ', size(vals), ' values, expected ', size(dest)
            error stop 1
        end if
        dest = vals
    end subroutine standard_axis_values



    subroutine load_xrb_spectra(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        integer :: i, j, stat, i_spec, u
        character(len=5), allocatable :: zz_str(:)
        real(WP), allocatable :: tspec(:, :, :)     ! (lambda, age, [Fe/H])
        type(datalib_t) :: lib
        integer(int64) :: off

        if (use_standard_library(ctx, 'xrb', 'xsp')) then
            call datalib_open(lib, ctx%sps_home, 'xrb', 'xsp')
            call standard_axis_values(lib, 'wavelength', ctx%state%lam_xrb)
            call standard_axis_values(lib, 'log_age', ctx%state%ages_xrb)
            call standard_axis_values(lib, 'feh', ctx%state%zmet_xrb)
            allocate (tspec(ctx%state%nspec_xrb, ctx%state%nt_xrb, ctx%state%nz_xrb))
            call datalib_check_array(lib, 'flux', [character(len=10) :: 'feh', 'log_age', 'wavelength'], &
                                     [ctx%state%nz_xrb, ctx%state%nt_xrb, ctx%state%nspec_xrb], off)
            call datalib_read_real(lib, 'flux', size(tspec, kind=int64), tspec, off)
        else
            allocate (tspec(ctx%state%nspec_xrb, ctx%state%nt_xrb, ctx%state%nz_xrb))
            allocate (zz_str(ctx%state%nz_xrb))

            open (newunit=u, file=trim(ctx%sps_home)//'/data/spectra/xrb/xsp.lambda', status='old', action='read', &
                  iostat=stat)
            if (stat /= 0) then
                write (error_unit, '(A)') '[FSPS_INIT] Error: xrb lambda file missing'
                error stop 1
            end if
            do i = 1, ctx%state%nspec_xrb
                read (u, *) ctx%state%lam_xrb(i)
            end do
            close (u)

            ctx%state%ages_xrb = log10((/1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 8.0_wp, 10.0_wp, 12.6_wp, 16.0_wp, &
                                         20.0_wp/)) + 6.0_wp
            ctx%state%zmet_xrb = (/-1.3_wp, -1.0_wp, -0.8_wp, -0.7_wp, -0.5_wp, -0.4_wp, -0.3_wp, -0.2_wp, 0.0_wp, &
                                   0.2_wp, 0.3_wp/)
            zz_str = (/'-1.30', '-1.00', '-0.80', '-0.70', '-0.50', '-0.40', '-0.30', '-0.20', '+0.00', '+0.20', '+0.30'/)

            do j = 1, ctx%state%nz_xrb
                open (newunit=u, file=trim(ctx%sps_home)//'/data/spectra/xrb/xsp_feh'//zz_str(j)//'.spec', &
                      status='old', action='read', iostat=stat)
                if (stat /= 0) then
                    write (error_unit, '(A,A)') '[FSPS_INIT] Error: xrb spec missing for ', zz_str(j)
                    error stop 1
                end if
                do i = 1, ctx%state%nt_xrb
                    read (u, *) tspec(:, i, j)
                end do
                close (u)
            end do
        end if

        do j = 1, ctx%state%nz_xrb
            do i = 1, ctx%state%nt_xrb
                do i_spec = 1, ctx%state%nspec
                    ctx%state%spec_xrb(i_spec, i, j) = max(interpolate_linear(ctx%state%lam_xrb, tspec(:, i, j), &
                                                           ctx%state%spec_lambda(i_spec)), SAFE_FLOOR)
                end do
            end do
        end do

        ctx%state%spec_xrb = ctx%state%spec_xrb*L_SOL

    end subroutine load_xrb_spectra

    !> @brief Name suffix of the nebular library for the context's cloudy_dust setting.
    function nebular_suffix(ctx) result(sfx)
        type(fsps_context_t), intent(in) :: ctx
        character(len=:), allocatable :: sfx
        if (ctx%cloudy_dust_val == 1) then
            sfx = '_dust'
        else
            sfx = '_nodust'
        end if
    end function nebular_suffix

    subroutine convert_to_vega_system(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        integer :: i

        do i = 1, ctx%state%nbands
            if (ctx%state%magsun(i) /= 99.0_wp) then
                ctx%state%magsun(i) = (ctx%state%magsun(i) - ctx%state%magsun(1)) - &
                                      (ctx%state%magvega(i) - ctx%state%magvega(1)) + ctx%state%magsun(1)
            end if
        end do
    end subroutine convert_to_vega_system

end module fsps_initialization
