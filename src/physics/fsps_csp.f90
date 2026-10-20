module fsps_csp
    !> @brief Core module for Composite Stellar Population (CSP) synthesis.
    !>
    !> @details
    !> This module calculates the spectral energy distribution of a composite
    !> population by integrating Simple Stellar Populations (SSPs) over a 
    !> user-defined Star Formation History (SFH).
    !>
    !> **Key Architecture:**
    !> 1. **Driver (`compute_csp_scenario`):** Orchestrates the loop over requested 
    !>    output ages and manages high-level post-processing (IGM, AGN, Smoothing).
    !> 2. **Kernel (`integrate_csp_step`):** The vectorization engine. It computes 
    !>    SFH weights and performs the linear combination of SSP spectra.
    !>    It separates flux into "Young" and "Old" components to handle 
    !>    age-dependent dust attenuation efficiently.
    !> 3. **Helpers:** Dedicated routines for SFH weight calculation and 
    !>    complex physics corrections (Dust, Nebular).
    !>
    !> **Optimization Strategy:**
    !> - **Matrix Operations:** Where possible, spectral summation is treated as 
    !>   matrix multiplication (Weights x SSP_Grid).
    !> - **Pre-calculation:** SFH weights are computed into a contiguous buffer 
    !>   before spectral integration to maximize memory locality.

    use fsps_precision, only: WP, KP
    use fsps_constants, only: NEMLINE, SAFE_FLOOR, C_LIGHT, VERBOSE
    use fsps_types, only: params, sfhparams, compspout
    use fsps_context_types, only: fsps_context_t, fsps_context_sync_device_settings

    !> Physics Modules
    use fsps_sfh, only: get_sfh_properties_at_age, compute_ssp_weights, sfh_weights_core, convert_sfhparams
    use fsps_dust, only: apply_dust_attenuation_and_emission, apply_agn_dust_emission
    use fsps_gas, only: apply_nebular_emission
    use fsps_smoothing, only: apply_smoothing
    use fsps_cosmology, only: compute_igm_transmission

    !> Math Modules
    use fsps_interpolation, only: find_interval, interpolate_linear

    !> Spectra Modules
    use fsps_photometry, only: compute_magnitudes
    use fsps_batch_photometry, only: compute_magnitudes_columns, prepare_batch_grid, smoothing_on_device, &
                                     prepare_smoothing_kernel, smooth_columns
    use fsps_spectral_indices, only: compute_spectral_indices, compute_spectral_indices_columns

    implicit none
    private

    !> Public API
    public :: compute_csp_scenario, set_scenario_batch

    !> One code path (doc/ROADMAP.md, priority 2): FSPS_SCENARIO_BATCH=1 runs eligible single
    !> calls as a batch of one (scenario_batch_of_one); read once (-1: not yet read)
    integer, save :: scenario_batch_mode = -1

    ! The batch of one is in the submodule fsps_scenario_batch (src/physics/fsps_scenario_batch.f90):
    ! it uses fsps_csp_batch, which uses fsps_context, which uses this module
    interface
        module function fsps_scenario_batch_of_one(ctx, pset, nzin, results) result(done)
            type(fsps_context_t), intent(inout)         :: ctx
            type(params), intent(in)                    :: pset
            integer, intent(in)                         :: nzin
            type(compspout), allocatable, intent(inout) :: results(:)
            logical :: done
        end function fsps_scenario_batch_of_one
    end interface
    ! Expose internal kernels for unit testing
    public :: compute_sfh_weights, sfh_weights_core
    public :: convert_sfhparams
    public :: integrate_csp_step
    public :: apply_dust_physics
    public :: apply_post_processing

contains

    ! ------------------------------------------------------------------------
    ! MAIN DRIVER
    ! ------------------------------------------------------------------------

    !> @brief Main driver to generate CSP properties for a set of parameters.
    !>
    !> @details
    !> This routine replaces the legacy `COMPSP`. It performs the following steps:
    !> 1. **Setup:** Creates local copies of the SSP grids and adds Nebular Emission 
    !>    (if enabled) to the SSPs *before* integration.
    !> 2. **Loop:** Iterates over the requested output ages.
    !>    - If `pset%tage > 0`: Computes a single snapshot at that specific age.
    !>    - If `pset%tage <= 0`: Computes the full time evolution history.
    !> 3. **Integration:** Calls `integrate_csp_step` to sum the SSPs.
    !> 4. **Physics:** Calls `apply_dust_physics` and `apply_post_processing`.
    !> 5. **Output:** Returns an allocatable array of `compspout` structures.
    !>
    !> @param[in,out] ctx       The main FSPS context.
    !> @param[in]     pset      User parameter set.
    !> @param[in]     nzin      Number of input metallicities to use (usually 1 or ctx%nz).
    !> @param[out]    results   Allocatable array of output structures.
    !> @param[out]    status    (Optional) Error status (0 = Success).
    subroutine compute_csp_scenario(ctx, pset, nzin, results, status)
        type(fsps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        integer, intent(in)                 :: nzin
        type(compspout), allocatable, intent(out) :: results(:)
        integer, intent(out), optional      :: status

        ! Local variables
        real(WP) :: target_age
        real(WP) :: mass_frac, sfr_norm, frac_linear, z_effective

        integer :: i, nt, nspec, n_outputs, start_idx
        integer :: k, i_spec, i_em
        logical :: dev_mags, tabular
        real(WP), allocatable :: mags_all(:,:), indx_all(:,:)

        call fsps_context_sync_device_settings(ctx)
        if (present(status)) status = 0

        ! 1. SETUP
        nt    = ctx%state%ntfull
        nspec = ctx%state%nspec
        
        if (ctx%state%check_sps_setup == 0) then
            if (present(status)) status = 1; return
        end if

        if (.not. allocated(ctx%state%ssp_basis_spec) .or. &
            .not. allocated(ctx%state%ssp_basis_mass) .or. &
            .not. allocated(ctx%state%ssp_basis_lbol)) then
            if (present(status)) status = 3
            return
        end if

        ! Early exit before device work if bad nebular params
        if (ctx%add_neb_emission_val == 1 .and. nzin > 1) then
             if (present(status)) status = 2
             return
        end if

        ! One code path: a single age with the common options as a batch of one; any other case,
        ! or a batch status other than 0, continues on this path
        if (scenario_batch_enabled() .and. pset%tage > 0.0_wp) then
            if (fsps_scenario_batch_of_one(ctx, pset, nzin, results)) return
        end if

        ! The parameter set is read by most kernels below (and by the dust and nebular routines it
        ! is passed to): map it once instead of an implicit copy at every launch
        !$acc enter data copyin(pset)

        ! Initialize working grids from SSP inputs natively on the device (Explicit loops to bypass NVHPC kernels bug)
        nt = ctx%state%ntfull
        nspec = ctx%state%nspec

        !$acc parallel loop collapse(3) present(ctx) private(i, i_spec)
        do k = 1, nzin
            do i = 1, nt
                do i_spec = 1, nspec
                    ctx%state%csp_ssp_grid(i_spec, i, k) = ctx%state%ssp_basis_spec(i_spec, i, k)
                end do
            end do
        end do

        !$acc parallel loop collapse(3) present(ctx) private(i, i_em)
        do k = 1, nzin
            do i = 1, nt
                do i_em = 1, NEMLINE
                    ctx%state%csp_emlin_grid(i_em, i, k) = 0.0_wp
                end do
            end do
        end do

        if (ctx%add_neb_emission_val == 1) then
            ! (nzin > 1 was rejected above)
            call apply_nebular_emission(ctx, pset, ctx%state%ssp_basis_spec(:,:,1), &
                                        ctx%state%csp_ssp_grid(:,:,1), ctx%state%csp_emlin_grid(:,:,1))
        end if
        
        ! 2. OPTIMIZATIONS (Pre-calculations)
        ! -----------------------------------
        
        ! A. Linearize Luminosity (Avoids 10**x inside hot loops)
        !$acc kernels present(ctx)
        ctx%state%csp_ssp_lum_linear(:,1:nzin) = 10.0_wp**ctx%state%ssp_basis_lbol(:,1:nzin)
        !$acc end kernels

        ! B. Pre-calculate IGM Transmission (Constant for this PSET)
        if (ctx%add_igm_absorption_val == 1 .and. pset%zred > SAFE_FLOOR) then
            call compute_igm_transmission(ctx%state%spec_lambda, &
                                          pset%zred, pset%igm_factor, &
                                          ctx%state%csp_igm_transmission)
        end if

        if (pset%tage > 0.0_wp) then
            n_outputs = 1; start_idx = 0 
        elseif (pset%tage == -99.0_wp .and. (pset%sfh == 2 .or. pset%sfh == 3)) then
            n_outputs = 1; start_idx = -99
        else
            n_outputs = nt; start_idx = 1
        end if
        
        allocate(results(n_outputs))

        ! 3. PRE-CALCULATE TARGET AGES
        ! -----------------------
        !$acc data present(ctx)

        ! Store target ages in a scratch array so the GPU can access them in the fused kernels
        do i = 1, n_outputs
            if (pset%tage > 0.0_wp) then
                ctx%state%sfh_t_calc(i) = pset%tage
            elseif (start_idx == -99) then
                ctx%state%sfh_t_calc(i) = maxval(ctx%state%sfh_tab(1, 1:ctx%state%ntabsfh)) / 1.0e9_wp
            else
                ctx%state%sfh_t_calc(i) = 10.0_wp**(ctx%state%time_full(i) - 9.0_wp)
            end if
        end do
        !$acc update device(ctx%state%sfh_t_calc(1:n_outputs)) async(1)

        ! 4. MASSIVE FUSED GENERATION (100% Device Resident)
        ! -----------------------

        ! Fused Integration Kernel
        call integrate_csp_step(ctx, pset, n_outputs, nzin, &
                                ctx%state%csp_ssp_grid, ctx%state%csp_emlin_grid, &
                                ctx%state%ssp_basis_mass, ctx%state%csp_ssp_lum_linear)

        ! Fused Dust Physics
        call apply_dust_physics(ctx, pset, n_outputs)

        ! Fused Post-Processing (Writes directly to ctx%state%out_* buffers)
        call apply_post_processing(ctx, pset, n_outputs, ctx%state%csp_igm_transmission)

        ! 5. SINGLE BULK PCIE TRANSFER
        ! -----------------------
        !$acc update host(ctx%state%out_csp_spec(:, 1:n_outputs), &
        !$acc             ctx%state%out_csp_emlin(:, 1:n_outputs), &
        !$acc             ctx%state%out_mass_csp(1:n_outputs), &
        !$acc             ctx%state%out_lbol_csp(1:n_outputs), &
        !$acc             ctx%state%sfh_w_tmp1(:, 1:n_outputs)) async(1)

        ! Block host here until the entire device pipeline finishes
        !$acc wait(1)

        !$acc end data

        ! 6. POPULATE DERIVED TYPES ON HOST CPU (Cache-Coherent)
        ! -----------------------
        ! On the GPU the magnitudes of all outputs are computed at once from the device spectra
        ! (fsps_batch_photometry), when no step between here and compute_magnitudes changes them
        ! (smoothing) or the redshift per output (redshift_colors), and without light-weighted ages
        dev_mags = .false.
#ifdef _OPENACC
        dev_mags = (.not. ctx%fast_mode) .and. pset%compute_mags == 1 .and. &
                   (pset%sigma_smooth <= 0.0_wp .or. smoothing_on_device(ctx)) .and. &
                   ctx%redshift_colors_val == 0 .and. ctx%compute_light_ages_val == 0
#endif
        if (dev_mags) then
            ! Velocity smoothing of all outputs on the device (the operator of apply_smoothing)
            if (pset%sigma_smooth > SAFE_FLOOR) then
                call prepare_batch_grid(ctx, pset%zred, pset%igm_factor)
                call prepare_smoothing_kernel(ctx, pset%sigma_smooth, pset%min_wave_smooth, pset%max_wave_smooth)
                call smooth_columns(ctx, ctx%state%out_csp_spec, n_outputs)
                !$acc update host(ctx%state%out_csp_spec(:, 1:n_outputs))
            end if
            allocate(mags_all(ctx%state%nbands, n_outputs))
            if (allocated(pset%mag_compute)) then
                call compute_magnitudes_columns(ctx, pset%zred, pset%igm_factor, ctx%state%out_csp_spec, n_outputs, &
                                                mags_all, pset%mag_compute)
            else
                call compute_magnitudes_columns(ctx, pset%zred, pset%igm_factor, ctx%state%out_csp_spec, n_outputs, &
                                                mags_all)
            end if
            if (pset%compute_indices == 1) then
                allocate(indx_all(ctx%state%nindx, n_outputs))
                call compute_spectral_indices_columns(ctx, ctx%state%spec_lambda, ctx%state%out_csp_spec, n_outputs, &
                                                      indx_all)
            end if
        end if

        ! Tabular SFHs (sfh = 2, 3) are not normalized: their SSP weights are the solar masses formed
        ! per SSP age (from the SFR table), and the spectrum, mass and L_bol are absolute. The mass
        ! formed is then the sum of the weights (it was reported as 1). The weights of more than one
        ! output are computed on the device in OpenACC GPU builds.
        tabular = pset%sfh == 2 .or. pset%sfh == 3
#if defined(_OPENACC) && !defined(ACC_MULTICORE)
        if (tabular .and. n_outputs > 1) then
            !$acc update host(ctx%state%csp_weights(:, 1:nzin, 1:n_outputs))
        end if
#endif

        do i = 1, n_outputs
            if (.not. allocated(results(i)%spec)) allocate(results(i)%spec(nspec))
            if (.not. allocated(results(i)%emlines)) allocate(results(i)%emlines(NEMLINE))

            ! Fast scalar CPU packing for arrays
            results(i)%spec(:)    = ctx%state%out_csp_spec(:, i)
            results(i)%emlines(:) = ctx%state%out_csp_emlin(:, i)

            ! Recalculate scalar properties on CPU to pack into struct
            target_age = ctx%state%sfh_t_calc(i)
            call get_sfh_properties_at_age(ctx, pset, target_age, mass_frac, sfr_norm, frac_linear)

            if (pset%tage <= 0.0_wp) then
                ! All-ages mode: outputs are per unit mass formed over the full history, so each
                ! age is rescaled by the fraction formed by then (the spectrum is rescaled in
                ! apply_post_processing).
                results(i)%lbol_csp = ctx%state%out_lbol_csp(i) + log10(max(mass_frac, SAFE_FLOOR))
                results(i)%mformed  = mass_frac
                results(i)%mass_csp = ctx%state%out_mass_csp(i) * mass_frac
                results(i)%mdust    = ctx%state%sfh_w_tmp1(3, i) * mass_frac
            else
                ! Single-age mode: the SFH weights are already normalized to 1 Msun formed by tage,
                ! exactly as for the spectrum and lbol, so mass and dust mass are not rescaled.
                results(i)%lbol_csp = ctx%state%out_lbol_csp(i)
                if (mass_frac > SAFE_FLOOR) sfr_norm = sfr_norm / mass_frac
                results(i)%mformed  = 1.0_wp
                results(i)%mass_csp = ctx%state%out_mass_csp(i)
                results(i)%mdust    = ctx%state%sfh_w_tmp1(3, i)
            end if

            if (tabular) results(i)%mformed = sum(ctx%state%csp_weights(1:nt, 1:nzin, i))

            results(i)%age      = log10(target_age * 1.0e9_wp)
            results(i)%sfr      = sfr_norm

            ! Perform heavy post-processing only if not in fast_mode
            if (.not. ctx%fast_mode) then
                if (.not. allocated(results(i)%mags)) allocate(results(i)%mags(ctx%state%nbands))
                if (.not. allocated(results(i)%indx)) allocate(results(i)%indx(ctx%state%nindx))

                if (dev_mags) then
                    results(i)%mags = mags_all(:, i)
                    if (pset%compute_indices == 1) then
                        results(i)%indx = indx_all(:, i)
                    else
                        results(i)%indx = -99.0_wp
                    end if
                    cycle
                end if

                ! Safely route the spectrum into the mapped device scratchpad
                ctx%state%csp_spec_final(:) = results(i)%spec(:)
                !$acc update device(ctx%state%csp_spec_final)

                ! Instrumental Smoothing
                if (pset%sigma_smooth > 0.0_wp) then
                    call apply_smoothing(ctx, ctx%state%spec_lambda, ctx%state%csp_spec_final, &
                                         pset%sigma_smooth, pset%min_wave_smooth, pset%max_wave_smooth)
                end if

                ! Redshift for magnitudes calculation
                if (ctx%redshift_colors_val == 1) then
                    z_effective = interpolate_linear(ctx%state%cosmospl(:,2), ctx%state%cosmospl(:,1), target_age)
                    z_effective = min(max(z_effective, 0.0_wp), 20.0_wp)
                else
                    z_effective = pset%zred
                end if

                ! Compute Magnitudes
                if (pset%compute_mags == 1) then
                    call compute_magnitudes(ctx, z_effective, ctx%state%csp_spec_final, results(i)%mags, pset%mag_compute)
                else
                    results(i)%mags = -99.0_wp
                end if

                ! Compute Spectral Indices
                if (pset%compute_indices == 1) then
                    call compute_spectral_indices(ctx, ctx%state%spec_lambda, ctx%state%csp_spec_final, results(i)%indx)
                else
                    results(i)%indx = -99.0_wp
                end if

                ! Pull the modified spectrum back to the host struct
                !$acc update host(ctx%state%csp_spec_final)
                results(i)%spec(:) = ctx%state%csp_spec_final(:)
            end if
        end do

        !$acc exit data delete(pset)

    end subroutine compute_csp_scenario

    !> Selects the single-call path explicitly, overriding FSPS_SCENARIO_BATCH: tests that inspect
    !> the old path's internal state (its SFH weights, IGM transmission) or use it as the
    !> reference for the batch call this with .false.
    subroutine set_scenario_batch(on)
        logical, intent(in) :: on

        scenario_batch_mode = merge(1, 0, on)
    end subroutine set_scenario_batch

    !> True if FSPS_SCENARIO_BATCH is set to a value other than 0 (read once).
    logical function scenario_batch_enabled()
        character(len=8) :: v
        integer :: st

        if (scenario_batch_mode < 0) then
            call get_environment_variable('FSPS_SCENARIO_BATCH', v, status=st)
            scenario_batch_mode = 0
            if (st == 0 .and. len_trim(v) > 0 .and. v /= '0') scenario_batch_mode = 1
        end if
        scenario_batch_enabled = scenario_batch_mode == 1
    end function scenario_batch_enabled


    ! ------------------------------------------------------------------------
    ! CORE INTEGRATION KERNEL
    ! ------------------------------------------------------------------------

    !> @brief Integrates SSPs to produce raw Young/Old composite spectra.
    !>
    !> @details
    !> 1. Calculates SFH weights.
    !> 2. Determines the age index `i_tesc` that separates "Young" (birth cloud)
    !>    from "Old" (diffuse ISM) populations based on `pset%dust_tesc`.
    !> 3. Accumulates the weighted sum of SSP spectra and emission lines into 
    !>    the buffer, preserving the Young/Old separation.
    !> 4. Computes total Stellar Mass and Bolometric Luminosity.
    !>
    !> @param[inout]  ctx        Context.
    !> @param[in]     pset       User parameters.
    !> @param[in]     tage       Target age of the galaxy [Gyr].
    !> @param[in]     nzin       Number of metallicities.
    !> @param[in]     ssp_grid   Input SSP grid [Lambda, Age, Z].
    !> @param[in]     emlin_grid Input Emission Line grid [Line, Age, Z].
    !> @param[out]    mass_csp   Total stellar mass formed (normalized).
    !> @param[out]    lbol_csp   Total bolometric luminosity [log10(L_sol)].
    subroutine integrate_csp_step(ctx, pset, n_outputs, nzin, ssp_grid, emlin_grid, &
                                  mass_ssp, ssp_lum_linear)
        type(fsps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        integer, intent(in)                 :: n_outputs, nzin
        real(KP), intent(in), contiguous    :: ssp_grid(:,:,:)
        real(KP), intent(in), contiguous    :: emlin_grid(:,:,:)
        real(WP), intent(in), contiguous    :: mass_ssp(:,:)
        real(WP), intent(in), contiguous    :: ssp_lum_linear(:,:)

        integer :: i, k, nt, i_tesc, i_out
        integer :: i_spec, i_em
        integer :: nspec, nem
        real(WP) :: dust_age_log
        real(WP) :: linear_lbol_sum
        real(WP) :: weight_ik
        real(KP) :: sum_young, sum_old
        real(KP) :: sum_em_young, sum_em_old

        nt    = ctx%state%ntfull
        nspec = size(ssp_grid, 1)
        nem   = size(emlin_grid, 1)

        ! 0. Determine Dust Separation Index (Host-Side)
        if (pset%dust_tesc > SAFE_FLOOR) then
            dust_age_log = pset%dust_tesc
        else
            dust_age_log = 7.0_wp
        end if
        i_tesc = max(1, min(find_interval(ctx%state%time_full, dust_age_log), nt))

        ! 1. Clear Accumulators (Fused)
        !$acc parallel loop collapse(2) present(ctx) private(i_spec) async(1)
        do i_out = 1, n_outputs
            do i_spec = 1, nspec
                ctx%state%spec_young(i_spec, i_out) = 0.0_wp
                ctx%state%spec_old(i_spec, i_out)   = 0.0_wp
            end do
        end do
        !$acc parallel loop collapse(2) present(ctx) private(i_em) async(1)
        do i_out = 1, n_outputs
            do i_em = 1, nem
                ctx%state%csp_emlin_young(i_em, i_out) = 0.0_wp
                ctx%state%csp_emlin_old(i_em, i_out)   = 0.0_wp
            end do
        end do

        ! 2. Compute SFH Weights (Gang Parallel over Time)
#if defined(_OPENACC) && !defined(ACC_MULTICORE)
        if (n_outputs == 1) then
            ! One output: a single device thread would run the whole SFH integral (~0.7 ms on a
            ! P100); the host does it in ~20 us
            call compute_sfh_weights(ctx, pset, ctx%state%sfh_t_calc(1), nzin, 1, ctx%state%csp_weights)
            !$acc update device(ctx%state%csp_weights(:, :, 1))
        else
            !$acc parallel loop gang present(ctx) async(1)
            do i_out = 1, n_outputs
                call compute_sfh_weights(ctx, pset, ctx%state%sfh_t_calc(i_out), nzin, i_out, ctx%state%csp_weights)
            end do
        end if
#else
        do i_out = 1, n_outputs
            call compute_sfh_weights(ctx, pset, ctx%state%sfh_t_calc(i_out), nzin, i_out, ctx%state%csp_weights)
        end do
#endif

        ! 3. Integration Loop for Scalars (Vector Reduction mapped to out buffers)
        if (n_outputs == 1) then
            i_out = 1

            ! Run sequentially on the device to avoid async reduction hazards on host scalars
            !$acc serial present(ctx, mass_ssp, ssp_lum_linear) private(linear_lbol_sum, weight_ik, i, k) async(1)
            ctx%state%out_mass_csp(i_out) = 0.0_wp
            linear_lbol_sum               = 0.0_wp

            do k = 1, nzin
                do i = 1, nt
                    weight_ik = ctx%state%csp_weights(i, k, i_out)
                    if (weight_ik > SAFE_FLOOR) then
                        ctx%state%out_mass_csp(i_out) = ctx%state%out_mass_csp(i_out) + (weight_ik * mass_ssp(i, k))
                        linear_lbol_sum               = linear_lbol_sum + (weight_ik * ssp_lum_linear(i, k))
                    end if
                end do
            end do

            if (linear_lbol_sum > SAFE_FLOOR) then
                ctx%state%out_lbol_csp(i_out) = log10(linear_lbol_sum)
            else
                ctx%state%out_lbol_csp(i_out) = 0.0_wp
            end if
            !$acc end serial

        else
            !$acc parallel loop gang present(ctx, mass_ssp, ssp_lum_linear) private(linear_lbol_sum, k, i, weight_ik) async(1)
            do i_out = 1, n_outputs
                ctx%state%out_mass_csp(i_out) = 0.0_wp
                linear_lbol_sum               = 0.0_wp
                !$acc loop seq
                do k = 1, nzin
                    !$acc loop seq
                    do i = 1, nt
                        weight_ik = ctx%state%csp_weights(i, k, i_out)
                        if (weight_ik > SAFE_FLOOR) then
                            ctx%state%out_mass_csp(i_out) = ctx%state%out_mass_csp(i_out) + (weight_ik * mass_ssp(i, k))
                            linear_lbol_sum               = linear_lbol_sum + (weight_ik * ssp_lum_linear(i, k))
                        end if
                    end do
                end do

                if (linear_lbol_sum > SAFE_FLOOR) then
                    ctx%state%out_lbol_csp(i_out) = log10(linear_lbol_sum)
                else
                    ctx%state%out_lbol_csp(i_out) = 0.0_wp
                end if
            end do
        end if

        ! 4. Integration Loop for Spectra
#ifdef _OPENACC
        ! =====================================================================
        ! GPU OPTIMIZED PATH (Coalesced Stride-1 Memory)
        ! =====================================================================
        ! One thread per (output, pixel), summing the ages in order (young up to i_tesc, then old)
        !$acc parallel loop gang vector collapse(2) present(ctx, ssp_grid) private(sum_young, sum_old, weight_ik, k, i) &
        !$acc async(1)
        do i_out = 1, n_outputs
            do i_spec = 1, nspec
                sum_young = 0.0_wp
                sum_old   = 0.0_wp

                !$acc loop seq
                do k = 1, nzin
                    ! Young stars
                    !$acc loop seq
                    do i = 1, i_tesc
                        weight_ik = ctx%state%csp_weights(i, k, i_out)
                        if (weight_ik > SAFE_FLOOR) then
                            sum_young = sum_young + (real(weight_ik, KP) * ssp_grid(i_spec, i, k))
                        end if
                    end do
                    ! Old stars
                    !$acc loop seq
                    do i = i_tesc + 1, nt
                        weight_ik = ctx%state%csp_weights(i, k, i_out)
                        if (weight_ik > SAFE_FLOOR) then
                            sum_old = sum_old + (real(weight_ik, KP) * ssp_grid(i_spec, i, k))
                        end if
                    end do
                end do

                ! Write out to global memory exactly once per wavelength bin
                ctx%state%spec_young(i_spec, i_out) = sum_young
                ctx%state%spec_old(i_spec, i_out)   = sum_old
            end do
        end do
#else
        ! =====================================================================
        ! CPU OPTIMIZED PATH (Strict Stride-1 Cache Locality, Vectorized)
        ! =====================================================================
        do i_out = 1, n_outputs
            do k = 1, nzin
                ! Young stars
                do i = 1, i_tesc
                    weight_ik = ctx%state%csp_weights(i, k, i_out)
                    if (weight_ik > SAFE_FLOOR) then
                        ! Stride-1 inner loop for CPU vectorization/cache
                        do i_spec = 1, nspec
                            ctx%state%spec_young(i_spec, i_out) = ctx%state%spec_young(i_spec, i_out) + &
                                (real(weight_ik, KP) * ssp_grid(i_spec, i, k))
                        end do
                    end if
                end do
                ! Old stars
                do i = i_tesc + 1, nt
                    weight_ik = ctx%state%csp_weights(i, k, i_out)
                    if (weight_ik > SAFE_FLOOR) then
                        do i_spec = 1, nspec
                            ctx%state%spec_old(i_spec, i_out) = ctx%state%spec_old(i_spec, i_out) + &
                                (real(weight_ik, KP) * ssp_grid(i_spec, i, k))
                        end do
                    end if
                end do
            end do
        end do
#endif
        
        ! 5. Integration Loop for Emission Lines
#ifdef _OPENACC
        ! =====================================================================
        ! GPU OPTIMIZED PATH (Coalesced Stride-1 Memory)
        ! =====================================================================
        !$acc parallel loop gang vector collapse(2) present(ctx, emlin_grid) &
        !$acc private(sum_em_young, sum_em_old, weight_ik, k, i) async(1)
        do i_out = 1, n_outputs
            do i_em = 1, nem
                sum_em_young = 0.0_wp
                sum_em_old   = 0.0_wp

                !$acc loop seq
                do k = 1, nzin
                    ! Young stars
                    !$acc loop seq
                    do i = 1, i_tesc
                        weight_ik = ctx%state%csp_weights(i, k, i_out)
                        if (weight_ik > SAFE_FLOOR) then
                            sum_em_young = sum_em_young + (real(weight_ik, KP) * emlin_grid(i_em, i, k))
                        end if
                    end do
                    ! Old stars
                    !$acc loop seq
                    do i = i_tesc + 1, nt
                        weight_ik = ctx%state%csp_weights(i, k, i_out)
                        if (weight_ik > SAFE_FLOOR) then
                            sum_em_old = sum_em_old + (real(weight_ik, KP) * emlin_grid(i_em, i, k))
                        end if
                    end do
                end do

                ctx%state%csp_emlin_young(i_em, i_out) = sum_em_young
                ctx%state%csp_emlin_old(i_em, i_out)   = sum_em_old
            end do
        end do
#else
        ! =====================================================================
        ! CPU OPTIMIZED PATH (Strict Stride-1 Cache Locality, Vectorized)
        ! =====================================================================
        do i_out = 1, n_outputs
            do k = 1, nzin
                ! Young stars
                do i = 1, i_tesc
                    weight_ik = ctx%state%csp_weights(i, k, i_out)
                    if (weight_ik > SAFE_FLOOR) then
                        ! Stride-1 inner loop for CPU vectorization/cache
                        do i_em = 1, nem
                            ctx%state%csp_emlin_young(i_em, i_out) = ctx%state%csp_emlin_young(i_em, i_out) + &
                                (real(weight_ik, KP) * emlin_grid(i_em, i, k))
                        end do
                    end if
                end do
                ! Old stars
                do i = i_tesc + 1, nt
                    weight_ik = ctx%state%csp_weights(i, k, i_out)
                    if (weight_ik > SAFE_FLOOR) then
                        do i_em = 1, nem
                            ctx%state%csp_emlin_old(i_em, i_out) = ctx%state%csp_emlin_old(i_em, i_out) + &
                                (real(weight_ik, KP) * emlin_grid(i_em, i, k))
                        end do
                    end if
                end do
            end do
        end do
#endif

    end subroutine integrate_csp_step

    ! ------------------------------------------------------------------------
    ! PHYSICS LAYERS
    ! ------------------------------------------------------------------------

    !> @brief Applies dust attenuation and IR emission models.
    !>
    !> @details
    !> Wraps the `fsps_dust` module to process the separated Young/Old spectra.
    !> 1. Applies birth cloud attenuation to `spec_young`.
    !> 2. Applies diffuse ISM attenuation to both `spec_young` and `spec_old`.
    !> 3. Computes IR dust re-emission via energy balance.
    !>
    !> @param[inout] ctx           Context.
    !> @param[in]    pset          User parameters.
    !> @param[in]    n_outputs     Number of outputs.
    subroutine apply_dust_physics(ctx, pset, n_outputs)
        type(fsps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        integer, intent(in)                 :: n_outputs

        ! Call the massively fused dust kernel. It will process all ages and wavelengths,
        ! writing directly to ctx%state%dust_spec_total_work.
        call apply_dust_attenuation_and_emission(ctx, pset, n_outputs)
    end subroutine apply_dust_physics

    !> @brief Applies final physical corrections and populates output.
    !>
    !> @details
    !> Performs the following operations in order:
    !> 1. **Mass Renormalization:** Calculates the "Surviving Mass" fraction if needed.
    !> 2. **Smoothing:** Convolves with LSF or velocity dispersion.
    !> 3. **IGM:** Applies Madau absorption if redshift > 0.
    !> 4. **AGN:** Adds AGN dust torus emission.
    !> 5. **Output:** Packages everything into the `compspout` structure.
    !>
    !> @param[inout] ctx              Context.
    !> @param[in]    pset             User parameters.
    !> @param[in]    n_outputs        Number of outputs.
    !> @param[in]    igm_transmission TODO: Add description
    subroutine apply_post_processing(ctx, pset, n_outputs, igm_transmission)
        type(fsps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        integer, intent(in)                 :: n_outputs
        real(KP), intent(in), optional      :: igm_transmission(:)

        integer  :: i_out, i_spec, i_em
        integer  :: nspec, nem
        real(WP) :: tage, mass_frac, sfr_norm, frac_linear

        nspec = ctx%state%nspec
        nem   = NEMLINE

        ! 1. Post-process Spectra (Mass Renormalization & IGM)
        !$acc parallel loop collapse(2) present(ctx, igm_transmission) &
        !$acc private(tage, mass_frac, sfr_norm, frac_linear, i_spec) async(1)
        do i_out = 1, n_outputs
            do i_spec = 1, nspec
                ! Recalculate scalars locally on device to bypass temp arrays
                tage = ctx%state%sfh_t_calc(i_out)
                call get_sfh_properties_at_age(ctx, pset, tage, mass_frac, sfr_norm, frac_linear)

                ctx%state%out_csp_spec(i_spec, i_out) = ctx%state%dust_spec_total_work(i_spec, i_out)

                if (pset%tage <= 0.0_wp) then
                    ctx%state%out_csp_spec(i_spec, i_out) = ctx%state%out_csp_spec(i_spec, i_out) * mass_frac
                end if

                if (ctx%add_igm_absorption_val == 1 .and. pset%zred > SAFE_FLOOR .and. present(igm_transmission)) then
                    ctx%state%out_csp_spec(i_spec, i_out) = ctx%state%out_csp_spec(i_spec, i_out) * igm_transmission(i_spec)
                end if
            end do
        end do

        if (ctx%add_agn_dust_val == 1 .and. pset%fagn > SAFE_FLOOR) then
            call apply_agn_dust_emission(ctx, pset, n_outputs, ctx%state%spec_lambda, &
                                         ctx%state%out_lbol_csp, ctx%state%out_csp_spec)
        end if

        ! 2. Post-process Emission Lines (Combination & Mass Renormalization)
        !$acc parallel loop collapse(2) present(ctx) &
        !$acc private(tage, mass_frac, sfr_norm, frac_linear, i_em) async(1)
        do i_out = 1, n_outputs
            do i_em = 1, nem
                tage = ctx%state%sfh_t_calc(i_out)
                call get_sfh_properties_at_age(ctx, pset, tage, mass_frac, sfr_norm, frac_linear)

                ctx%state%out_csp_emlin(i_em, i_out) = ctx%state%csp_emlin_young(i_em, i_out) + &
                                                       ctx%state%csp_emlin_old(i_em, i_out)

                if (pset%tage <= 0.0_wp) then
                    ctx%state%out_csp_emlin(i_em, i_out) = ctx%state%out_csp_emlin(i_em, i_out) * mass_frac
                end if
            end do
        end do
    end subroutine apply_post_processing

    ! ------------------------------------------------------------------------
    ! SFH LOGIC
    ! ------------------------------------------------------------------------

    !> @brief Computes the mass weights for each SSP bin for the given SFH.
    !>
    !> @details
    !> Orchestrates the calculation of weights for different SFH types:
    !> - **SSP (0):** Single burst.
    !> - **Tau/Delayed (1,4):** Exponential models + Constant + Burst.
    !> - **Simha (5):** Mixed Linear + Delayed-Tau.
    !> - **Tabular (2,3):** Sum of linear segments from a table.
    !>
    !> Ensures that the final weights sum to 1.0 M_sol formed (or appropriate 
    !> mass fraction) so that the resulting spectrum is per unit mass formed.
    !>
    !> @param[inout] ctx     Context.
    !> @param[in]    pset    User parameters.
    !> @param[in]    tage    Age of the galaxy [Gyr].
    !> @param[in]    nzin    Number of metallicity bins.
    !> @param[in]    i_out   Output age index.
    !> @param[inout] weights Weights [ntfull, nzin].
    subroutine compute_sfh_weights(ctx, pset, tage, nzin, i_out, weights)
        type(fsps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        real(WP), intent(in)                :: tage
        integer, intent(in)                 :: nzin, i_out
        real(WP), intent(inout), contiguous :: weights(:,:,:)

        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif

        call sfh_weights_core(ctx, pset, tage, nzin, weights(:, :, i_out), ctx%state%sfh_w_tmp1(:, i_out), &
                              ctx%state%sfh_w_tmp2(:, i_out))
    end subroutine compute_sfh_weights

end module fsps_csp
