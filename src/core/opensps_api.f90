module opensps_api
    !> @brief
    !> Public, stable API entry point for FSPS.
    !>
    !> @details
    !> This module provides a single, high-level interface used by Fortran callers
    !> and the C bindings. It wraps the internal context lifecycle and runtime
    !> operations while delegating heavy lifting to core implementations.

    use opensps_precision, only: WP
    use opensps_types, only: compspout
    use opensps_context_types, only: opensps_context_t
    use opensps_ssp, only: generate_ssp_grid
    use opensps_types, only: params
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_create, &
        opensps_context_remove_from_device, opensps_context_setup, &
        opensps_context_destroy, &
                            opensps_context_set_param_int, opensps_context_set_param_float, &
                            opensps_context_set_param_str, opensps_context_get_paths, &
                            opensps_context_set_fast_mode, opensps_context_prepare_pset, opensps_context_compute_ssp, &
                            opensps_context_compute_csp
    use opensps_environment, only: opensps_resolve_paths, opensps_cleanup, opensps_print_env_info

    implicit none
    private

    public :: opensps_create
    public :: opensps_setup
    public :: opensps_build_afe_basis
    public :: opensps_destroy
    public :: sps_setup
    public :: sps_takedown
    public :: opensps_set_param_int
    public :: opensps_set_param_float
    public :: opensps_set_param_str
    public :: opensps_set_fast_mode
    public :: opensps_get_paths
    public :: opensps_prepare_pset
    public :: opensps_compute_ssp
    public :: opensps_compute_csp
    public :: opensps_resolve_paths
    public :: opensps_cleanup
    public :: opensps_print_env_info

    ! ---------------------------------------------------------------------
    ! Module constants
    ! ---------------------------------------------------------------------

contains

    !> @brief Create a new FSPS context with default parameters.
    subroutine opensps_create(ctx)
        type(opensps_context_t), intent(inout) :: ctx

        call opensps_context_create(ctx)
    end subroutine opensps_create

    !> @brief Initialize or reinitialize a context with selected libraries.
    !>
    !> @param[inout] ctx The context to initialize.
    !> @param[in]    zin Metallicity index to load (-1 for all).
    !> @param[in]    isoc_type_in Optional isochrone library name.
    !> @param[in]    spec_type_in Optional spectral library name.
    !> @param[in]    dust_type_in Optional dust emission model name.
    subroutine opensps_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
        type(opensps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin
        character(len=*), intent(in), optional :: isoc_type_in
        character(len=*), intent(in), optional :: spec_type_in
        character(len=*), intent(in), optional :: dust_type_in

        if (present(isoc_type_in) .and. present(spec_type_in) .and. present(dust_type_in)) then
            call opensps_context_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
        else if (present(isoc_type_in) .and. present(spec_type_in)) then
            call opensps_context_setup(ctx, zin, isoc_type_in, spec_type_in)
        else if (present(isoc_type_in) .and. present(dust_type_in)) then
            call opensps_context_setup(ctx, zin, isoc_type_in, dust_type_in=dust_type_in)
        else if (present(spec_type_in) .and. present(dust_type_in)) then
            call opensps_context_setup(ctx, zin, spec_type_in=spec_type_in, dust_type_in=dust_type_in)
        else if (present(isoc_type_in)) then
            call opensps_context_setup(ctx, zin, isoc_type_in)
        else if (present(spec_type_in)) then
            call opensps_context_setup(ctx, zin, spec_type_in=spec_type_in)
        else if (present(dust_type_in)) then
            call opensps_context_setup(ctx, zin, dust_type_in=dust_type_in)
        else
            call opensps_context_setup(ctx, zin)
        end if
    end subroutine opensps_setup

    !> @brief Builds the resident SSP bases over [alpha/Fe] and every metallicity of the libraries.
    !>
    !> @details For each value of afe_values (increasing; values of the libraries' [alpha/Fe] axis) the
    !> context is set up with the libraries `isoc_name:afe=value` and `spec_name:afe=value`, the SSP of every
    !> metallicity is generated with the context's current parameters, and the stack of nz * size(afe_values)
    !> bases is handed to opensps_context_update_ssp_basis (the metallicity fastest). The parameters that act
    !> on the SSPs (IMF, isochrone modifications, ...) are the context's at the call. Call it before moving the
    !> context to the device; afterwards `afe` is a per-model parameter of compute_csp_batch (with gradients).
    !> @param[in]  isoc_name   Isochrone library without selection (e.g. 'mist_afe').
    !> @param[in]  spec_name   Spectral library without selection (e.g. 'c3k_lr').
    !> @param[in]  afe_values  The [alpha/Fe] planes, increasing.
    !> @param[out] status      0 on success, 1 for fewer than two or unsorted values.
    subroutine opensps_build_afe_basis(ctx, isoc_name, spec_name, afe_values, status, dust_type_in)
        type(opensps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: isoc_name, spec_name
        real(WP), intent(in) :: afe_values(:)
        integer, intent(out) :: status
        character(len=*), intent(in), optional :: dust_type_in
        real(WP), allocatable :: spec(:, :, :), mass(:, :), lbol(:, :)
        type(params) :: pset
        character(len=16) :: sel
        character(len=32) :: dust
        integer :: ia, iz, nz, st, slab

        status = 1
        if (size(afe_values) < 2) return
        if (any(afe_values(2:) <= afe_values(:size(afe_values) - 1))) return
        dust = 'DL07'
        if (present(dust_type_in)) dust = dust_type_in
        do ia = 1, size(afe_values)
            write (sel, '(F8.3)') afe_values(ia)
            call opensps_setup(ctx, 0, isoc_type_in=trim(isoc_name)//':afe='//trim(adjustl(sel)), &
                               spec_type_in=trim(spec_name)//':afe='//trim(adjustl(sel)), dust_type_in=trim(dust))
            nz = ctx%state%nz
            if (.not. allocated(spec)) then
                allocate (spec(ctx%state%nspec, ctx%state%ntfull, nz * size(afe_values)), &
                          mass(ctx%state%ntfull, nz * size(afe_values)), lbol(ctx%state%ntfull, nz * size(afe_values)))
            end if
            do iz = 1, nz
                slab = iz + nz * (ia - 1)
                call opensps_context_set_param_int(ctx, 'zmet', iz, st)
                call opensps_context_prepare_pset(ctx)
                pset = ctx%pset
                call generate_ssp_grid(ctx, pset, mass(:, slab), lbol(:, slab), spec(:, :, slab))
            end do
        end do
        ctx%state%ssp_basis_is_dirty = .true.
        call opensps_context_update_ssp_basis(ctx, spec, mass, lbol, nz * size(afe_values), afe_values)
        status = 0
    end subroutine opensps_build_afe_basis

    !> @brief Legacy setup wrapper (kept for compatibility with older callers).
    subroutine sps_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
        type(opensps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin
        character(len=*), intent(in), optional :: isoc_type_in
        character(len=*), intent(in), optional :: spec_type_in
        character(len=*), intent(in), optional :: dust_type_in

        call opensps_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
    end subroutine sps_setup

    !> @brief Destroy a context and release associated resources.
    subroutine opensps_destroy(ctx)
        type(opensps_context_t), intent(inout) :: ctx

        call opensps_context_destroy(ctx)
    end subroutine opensps_destroy

    !> @brief Legacy teardown wrapper (kept for compatibility with older callers).
    subroutine sps_takedown(ctx)
        type(opensps_context_t), intent(inout) :: ctx

        if (ctx%on_device) call opensps_context_remove_from_device(ctx)
        call opensps_cleanup(ctx)
    end subroutine sps_takedown

    !> @brief Set integer parameter by name.
    subroutine opensps_set_param_int(ctx, key, value, status)
        type(opensps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer, intent(out) :: status

        call opensps_context_set_param_int(ctx, key, value, status)
    end subroutine opensps_set_param_int

    !> @brief Set floating-point parameter by name.
    subroutine opensps_set_param_float(ctx, key, value, status)
        type(opensps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer, intent(out) :: status

        call opensps_context_set_param_float(ctx, key, value, status)
    end subroutine opensps_set_param_float

    !> @brief Set string parameter by name.
    subroutine opensps_set_param_str(ctx, key, value, status)
        type(opensps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        character(len=*), intent(in) :: value
        integer, intent(out) :: status

        call opensps_context_set_param_str(ctx, key, value, status)
    end subroutine opensps_set_param_str

    !> @brief Enable or disable CSP fast mode.
    subroutine opensps_set_fast_mode(ctx, fast_mode)
        type(opensps_context_t), intent(inout) :: ctx
        logical, intent(in) :: fast_mode

        call opensps_context_set_fast_mode(ctx, fast_mode)
    end subroutine opensps_set_fast_mode

    !> @brief Retrieve context paths for data and outputs.
    subroutine opensps_get_paths(ctx, sps_home_out, data_home_out, output_home_out)
        type(opensps_context_t), intent(in) :: ctx
        character(len=*), intent(out) :: sps_home_out
        character(len=*), intent(out) :: data_home_out
        character(len=*), intent(out) :: output_home_out

        call opensps_context_get_paths(ctx, sps_home_out, data_home_out, output_home_out)
    end subroutine opensps_get_paths

    !> @brief Prepare context parameter-set buffers.
    subroutine opensps_prepare_pset(ctx)
        type(opensps_context_t), intent(inout) :: ctx

        call opensps_context_prepare_pset(ctx)
    end subroutine opensps_prepare_pset

    !> @brief Compute SSP outputs for the current context.
    subroutine opensps_compute_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
        type(opensps_context_t), intent(inout) :: ctx
        real(WP), dimension(:), intent(out) :: mass_ssp, lbol_ssp
        real(WP), dimension(:, :), intent(out) :: spec_ssp

        call opensps_context_compute_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
    end subroutine opensps_compute_ssp

    !> @brief Compute CSP outputs for the current context.
    subroutine opensps_compute_csp(ctx, write_compsp, nzin, outfile, mass_ssp, lbol_ssp, spec_ssp, ocompsp)
        type(opensps_context_t), intent(inout) :: ctx
        integer, intent(in) :: write_compsp, nzin
        character(len=*), intent(in) :: outfile
        real(WP), dimension(:, :), intent(in) :: mass_ssp, lbol_ssp
        real(WP), dimension(:, :, :), intent(in) :: spec_ssp
        type(compspout), dimension(:), intent(inout) :: ocompsp

        call opensps_context_compute_csp(ctx, write_compsp, nzin, outfile, mass_ssp, lbol_ssp, spec_ssp, ocompsp)
    end subroutine opensps_compute_csp

end module opensps_api
