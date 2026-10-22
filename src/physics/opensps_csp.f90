module opensps_csp
    !> @brief Composite Stellar Populations (CSP): the single-parameter-set driver.
    !>
    !> @details
    !> `compute_csp_scenario` (the legacy `COMPSP`) returns the CSP of one parameter set at one
    !> age (`tage > 0`), at every age of the SSP grid (`tage <= 0`), or at the last age of a
    !> tabular SFH (`tage = -99`, `sfh = 2, 3`). It runs as one call of the batched CSP
    !> (`compute_csp_batch`, src/physics/opensps_csp_batch.f90), the only implementation of the
    !> CSP physics, with one model per output age (src/physics/opensps_scenario_batch.f90).

    use opensps_types, only: params, compspout
    use opensps_context_types, only: opensps_context_t, opensps_context_sync_device_settings

    !> Re-exported for the callers and tests that compute SFH weights directly
    use opensps_sfh, only: sfh_weights_core, convert_sfhparams

    implicit none
    private

    public :: compute_csp_scenario
    public :: sfh_weights_core, convert_sfhparams

    ! The batch call is in the submodule opensps_scenario_batch (src/physics/opensps_scenario_batch.f90):
    ! it uses opensps_csp_batch, which uses opensps_context, which uses this module
    interface
        module function opensps_scenario_as_batch(ctx, pset, nzin, results) result(st)
            type(opensps_context_t), intent(inout)         :: ctx
            type(params), intent(in)                    :: pset
            integer, intent(in)                         :: nzin
            type(compspout), allocatable, intent(inout) :: results(:)
            integer :: st
        end function opensps_scenario_as_batch
    end interface

contains

    !> @brief CSP outputs for one parameter set (the legacy `COMPSP`).
    !>
    !> @details
    !> One output for `tage > 0` (at tage) and for `tage = -99` with a tabular SFH (at the last
    !> age of the table); otherwise one output per age of the SSP grid, each per unit mass formed
    !> over the full history (mformed = the fraction formed by then), as legacy FSPS. Spectra,
    !> magnitudes, indices and line luminosities come from `compute_csp_batch`.
    !>
    !> @param[in,out] ctx       The context, with the SSP basis set (opensps_context_update_ssp_basis).
    !> @param[in]     pset      User parameter set.
    !> @param[in]     nzin      Number of metallicities summed: 1, or ctx%state%nz for a tabular SFH
    !>                          with a metallicity column; the resident basis must have nzin.
    !> @param[out]    results   Allocatable array of output structures.
    !> @param[out]    status    (Optional) 0 on success; 1 no setup; 2 unsupported option
    !>                          (nebular emission with nzin > 1, invalid dust settings: uvb < 0,
    !>                          wgp1/2 < 1); 3 no SSP basis, or one that does not have nzin
    !>                          metallicities.
    subroutine compute_csp_scenario(ctx, pset, nzin, results, status)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        integer, intent(in)                 :: nzin
        type(compspout), allocatable, intent(out) :: results(:)
        integer, intent(out), optional      :: status

        integer :: st

        call opensps_context_sync_device_settings(ctx)
        if (present(status)) status = 0

        if (ctx%state%check_sps_setup == 0) then
            if (present(status)) status = 1
            return
        end if

        if (.not. allocated(ctx%state%ssp_basis_spec) .or. &
            .not. allocated(ctx%state%ssp_basis_mass) .or. &
            .not. allocated(ctx%state%ssp_basis_lbol)) then
            if (present(status)) status = 3
            return
        end if

        ! Nebular emission is added to the young SSPs of one metallicity
        if (ctx%add_neb_emission_val == 1 .and. nzin > 1) then
            if (present(status)) status = 2
            return
        end if

        st = opensps_scenario_as_batch(ctx, pset, nzin, results)
        if (present(status)) status = st
    end subroutine compute_csp_scenario

end module opensps_csp
