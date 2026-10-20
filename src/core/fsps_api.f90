module fsps_api
    !> @brief
    !> Public, stable API entry point for FSPS.
    !>
    !> @details
    !> This module provides a single, high-level interface used by Fortran callers
    !> and the C bindings. It wraps the internal context lifecycle and runtime
    !> operations while delegating heavy lifting to core implementations.

    use fsps_precision, only: WP
    use fsps_types, only: compspout
    use fsps_context_types, only: fsps_context_t
    use fsps_context, only: fsps_context_create, fsps_context_remove_from_device, fsps_context_setup, fsps_context_destroy, &
                            fsps_context_set_param_int, fsps_context_set_param_float, &
                            fsps_context_set_param_str, fsps_context_get_paths, &
                            fsps_context_set_fast_mode, fsps_context_prepare_pset, fsps_context_compute_ssp, &
                            fsps_context_compute_csp
    use fsps_environment, only: fsps_resolve_paths, fsps_cleanup, fsps_print_env_info

    implicit none
    private

    public :: fsps_create
    public :: fsps_setup
    public :: fsps_destroy
    public :: sps_setup
    public :: sps_takedown
    public :: fsps_set_param_int
    public :: fsps_set_param_float
    public :: fsps_set_param_str
    public :: fsps_set_fast_mode
    public :: fsps_get_paths
    public :: fsps_prepare_pset
    public :: fsps_compute_ssp
    public :: fsps_compute_csp
    public :: fsps_resolve_paths
    public :: fsps_cleanup
    public :: fsps_print_env_info

    ! ---------------------------------------------------------------------
    ! Module constants
    ! ---------------------------------------------------------------------

contains

    !> @brief Create a new FSPS context with default parameters.
    subroutine fsps_create(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        call fsps_context_create(ctx)
    end subroutine fsps_create

    !> @brief Initialize or reinitialize a context with selected libraries.
    !>
    !> @param[inout] ctx The context to initialize.
    !> @param[in]    zin Metallicity index to load (-1 for all).
    !> @param[in]    isoc_type_in Optional isochrone library name.
    !> @param[in]    spec_type_in Optional spectral library name.
    !> @param[in]    dust_type_in Optional dust emission model name.
    subroutine fsps_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin
        character(len=*), intent(in), optional :: isoc_type_in
        character(len=*), intent(in), optional :: spec_type_in
        character(len=*), intent(in), optional :: dust_type_in

        if (present(isoc_type_in) .and. present(spec_type_in) .and. present(dust_type_in)) then
            call fsps_context_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
        else if (present(isoc_type_in) .and. present(spec_type_in)) then
            call fsps_context_setup(ctx, zin, isoc_type_in, spec_type_in)
        else if (present(isoc_type_in) .and. present(dust_type_in)) then
            call fsps_context_setup(ctx, zin, isoc_type_in, dust_type_in=dust_type_in)
        else if (present(spec_type_in) .and. present(dust_type_in)) then
            call fsps_context_setup(ctx, zin, spec_type_in=spec_type_in, dust_type_in=dust_type_in)
        else if (present(isoc_type_in)) then
            call fsps_context_setup(ctx, zin, isoc_type_in)
        else if (present(spec_type_in)) then
            call fsps_context_setup(ctx, zin, spec_type_in=spec_type_in)
        else if (present(dust_type_in)) then
            call fsps_context_setup(ctx, zin, dust_type_in=dust_type_in)
        else
            call fsps_context_setup(ctx, zin)
        end if
    end subroutine fsps_setup

    !> @brief Legacy setup wrapper (kept for compatibility with older callers).
    subroutine sps_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: zin
        character(len=*), intent(in), optional :: isoc_type_in
        character(len=*), intent(in), optional :: spec_type_in
        character(len=*), intent(in), optional :: dust_type_in

        call fsps_setup(ctx, zin, isoc_type_in, spec_type_in, dust_type_in)
    end subroutine sps_setup

    !> @brief Destroy a context and release associated resources.
    subroutine fsps_destroy(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        call fsps_context_destroy(ctx)
    end subroutine fsps_destroy

    !> @brief Legacy teardown wrapper (kept for compatibility with older callers).
    subroutine sps_takedown(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (ctx%on_device) call fsps_context_remove_from_device(ctx)
        call fsps_cleanup(ctx)
    end subroutine sps_takedown

    !> @brief Set integer parameter by name.
    subroutine fsps_set_param_int(ctx, key, value, status)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer, intent(out) :: status

        call fsps_context_set_param_int(ctx, key, value, status)
    end subroutine fsps_set_param_int

    !> @brief Set floating-point parameter by name.
    subroutine fsps_set_param_float(ctx, key, value, status)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer, intent(out) :: status

        call fsps_context_set_param_float(ctx, key, value, status)
    end subroutine fsps_set_param_float

    !> @brief Set string parameter by name.
    subroutine fsps_set_param_str(ctx, key, value, status)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        character(len=*), intent(in) :: value
        integer, intent(out) :: status

        call fsps_context_set_param_str(ctx, key, value, status)
    end subroutine fsps_set_param_str

    !> @brief Enable or disable CSP fast mode.
    subroutine fsps_set_fast_mode(ctx, fast_mode)
        type(fsps_context_t), intent(inout) :: ctx
        logical, intent(in) :: fast_mode

        call fsps_context_set_fast_mode(ctx, fast_mode)
    end subroutine fsps_set_fast_mode

    !> @brief Retrieve context paths for data and outputs.
    subroutine fsps_get_paths(ctx, sps_home_out, data_home_out, output_home_out)
        type(fsps_context_t), intent(in) :: ctx
        character(len=*), intent(out) :: sps_home_out
        character(len=*), intent(out) :: data_home_out
        character(len=*), intent(out) :: output_home_out

        call fsps_context_get_paths(ctx, sps_home_out, data_home_out, output_home_out)
    end subroutine fsps_get_paths

    !> @brief Prepare context parameter-set buffers.
    subroutine fsps_prepare_pset(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        call fsps_context_prepare_pset(ctx)
    end subroutine fsps_prepare_pset

    !> @brief Compute SSP outputs for the current context.
    subroutine fsps_compute_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), dimension(:), intent(out) :: mass_ssp, lbol_ssp
        real(WP), dimension(:, :), intent(out) :: spec_ssp

        call fsps_context_compute_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
    end subroutine fsps_compute_ssp

    !> @brief Compute CSP outputs for the current context.
    subroutine fsps_compute_csp(ctx, write_compsp, nzin, outfile, mass_ssp, lbol_ssp, spec_ssp, ocompsp)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: write_compsp, nzin
        character(len=*), intent(in) :: outfile
        real(WP), dimension(:, :), intent(in) :: mass_ssp, lbol_ssp
        real(WP), dimension(:, :, :), intent(in) :: spec_ssp
        type(compspout), dimension(:), intent(inout) :: ocompsp

        call fsps_context_compute_csp(ctx, write_compsp, nzin, outfile, mass_ssp, lbol_ssp, spec_ssp, ocompsp)
    end subroutine fsps_compute_csp

end module fsps_api
