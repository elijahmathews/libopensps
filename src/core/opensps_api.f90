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
    use opensps_context, only: opensps_context_create, opensps_context_remove_from_device, opensps_context_setup, &
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
