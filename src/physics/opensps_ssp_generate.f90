!> generate_ssp_grid, as a submodule of opensps_ssp so that it can use the template operator
!> (opensps_ssp_operator), which uses opensps_ssp.
submodule (opensps_ssp) opensps_ssp_generate
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_build, ssp_operator_evaluate, ssp_operator_free
    use opensps_imf, only: imf_pack_params, IMF_NPAR
    implicit none

contains

    !> @brief Generates the SSP grid (all ages of one metallicity): the legacy `SSP_GEN`.
    !>
    !> @details
    !> **Pipeline:**
    !> 1. BPASS: the tabulated SSPs (no isochrones).
    !> 2. Otherwise the template operator of opensps_ssp_operator, the one implementation of the
    !>    SSP sum: S (isochrone points to template weights per age, independent of the IMF) is
    !>    built from the modified isochrones (blue HB, blue stragglers, AGB and post-AGB
    !>    weights), then evaluated for the IMF of pset: IMF weights of all points, mass and L_bol
    !>    per age, and spec = templates . (S . w_IMF) for all ages in one product.
    !> 3. Post-processing (finish_ssp_grid): time interpolation (TIME_RES_INCR > 1), nebular
    !>    emission (add_neb_emission = 2), X-ray binaries, instrumental smoothing.
    !>
    !> With opensps_ssp_set_per_star(.true.) the spectra are instead summed star by star
    !> (accumulate_spectrum, get_stellar_spectrum per star): the reference the operator is tested
    !> against (test_physics_oracles, test_ssp_operator).
    !>
    !> @param[in,out] ctx       The main FSPS context.
    !> @param[in]     pset      User parameter set.
    !> @param[out]    mass_grid Output mass history (time dependent).
    !> @param[out]    lbol_grid Output bolometric luminosity history.
    !> @param[out]    spec_grid Output spectral grid (wavelength, time).
    module subroutine generate_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)
        type(opensps_context_t), target, intent(inout) :: ctx
        type(params), intent(in)            :: pset
        real(WP), intent(out), contiguous   :: mass_grid(:)
        real(WP), intent(out), contiguous   :: lbol_grid(:)
        real(WP), intent(out), contiguous   :: spec_grid(:,:)

        type(isochrone_buffer_t) :: buf
        type(ssp_operator_t) :: op
        real(WP) :: p(IMF_NPAR)
        integer :: i_time, out_idx
        real(WP) :: time_log_yr

        call opensps_context_sync_device_settings(ctx)
        mass_grid = 0.0_wp
        lbol_grid = 0.0_wp
        spec_grid = 0.0_wp

        if (ctx%state%check_sps_setup == 0) then
            write(*,*) 'SSP_GEN ERROR: SPS_SETUP must be run before SSP_GEN.'
            stop
        end if
        if (pset%zmet < 1 .or. pset%zmet > ctx%state%nz) then
            write(*,*) 'SSP_GEN ERROR: Metallicity index out of range:', pset%zmet
            stop
        end if

        ! BPASS models are tabulated SSPs: no isochrones to sum
        if (trim(ctx%state%isoc_type) == 'bpss') then
            spec_grid = ctx%state%bpass_spec_ssp(:, :, pset%zmet)
            mass_grid = ctx%state%bpass_mass_ssp(:, pset%zmet)
            ! BPASS ships no isochrones, so L_bol is the integral of the SSP spectrum
            ! (Lsun/Hz over Hz). Legacy FSPS leaves it unset (log L = 0 at every age).
            do i_time = 1, size(lbol_grid)
                lbol_grid(i_time) = log10(max(integrate_trapezoid_array(ctx%state%spec_nu(1:size(spec_grid, 1)), &
                                                                        spec_grid(:, i_time)), tiny(1.0_wp)))
            end do
            return
        end if

        ! The IMF of pset (maps pset to the context; reads a user IMF file)
        call configure_imf_parameters(ctx, pset)

        if (.not. ssp_per_star) then
            call ssp_operator_build(ctx, pset, op)
            call imf_pack_params(ctx, p)
            call ssp_operator_evaluate(ctx, pset, op, ctx%imf_type_val, p, mass_grid, lbol_grid, spec_grid)
            call ssp_operator_free(op)
            return
        end if

        ! Reference: star by star, age by age
        call init_isochrone_buffer(buf)
        do i_time = 1, ctx%state%nt
            if (pset%ssp_gen_age(i_time) == 0) cycle
            out_idx = 1 + (i_time - 1) * TIME_RES_INCR
            time_log_yr = ctx%state%timestep_isoc(pset%zmet, i_time)
            call load_timestep_data(ctx, pset%zmet, i_time, buf)
            call apply_isochrone_physics(ctx, pset, time_log_yr, buf)
            call compute_integrated_properties(ctx, buf, mass_grid(out_idx), lbol_grid(out_idx))
            call accumulate_spectrum(ctx, pset, buf, spec_grid(:, out_idx))
        end do
        call free_isochrone_buffer(buf)
        call finish_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)
    end subroutine generate_ssp_grid

end submodule opensps_ssp_generate
