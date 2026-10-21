!> One code path (doc/ROADMAP.md, priority 2): compute_csp_scenario as a batch, as a submodule of
!> opensps_csp so that opensps_csp (used by opensps_context) need not use opensps_csp_batch (which
!> uses opensps_context): the submodule is compiled after both. Called by compute_csp_scenario when
!> FSPS_SCENARIO_BATCH=1.
submodule (opensps_csp) opensps_scenario_batch
    implicit none
contains
    !> compute_csp_scenario as one batch call (compute_csp_batch) with one model per output age,
    !> packed into results(:) as the scenario path does:
    !> - tage > 0: one model at tage;
    !> - tage <= 0: one model per age of the grid (ntfull outputs); tage = -99 with a tabular SFH
    !>   (sfh = 2, 3): one model at the last age of the table.
    !> The SFH weights of each output are computed on the host (sfh_weights_core, as the scenario
    !> path does for one output) and given to the batch as an SSP-weight matrix. In all-ages mode
    !> they are scaled by the fraction of the mass formed by that age (mass_frac), so that every
    !> output is per unit mass formed over the full history, as legacy FSPS's compsp does: all
    !> outputs are linear in the weights, and the AGN torus emission (fagn x L_bol) follows the
    !> scaled L_bol, as in legacy.
    !> With redshift_colors = 1 the magnitudes of each output are at the redshift of its age (from
    !> the cosmology table, clamped to 0-20, as the scenario path), the spectrum at zred.
    !> Several metallicities (nzin > 1: tabular SFHs with a metallicity column, the bases of all
    !> metallicities resident) are summed with weights over ages and metallicities.
    !> Returns .false. (results untouched) for the options the batch does not cover yet: several
    !> metallicities without all bases resident, light-weighted ages, OpenACC multicore builds, and
    !> anything the batch rejects (invalid dust settings: status /= 0).
    !> Design note: doc/GPU_RESULTS.md, "compute_csp_scenario as a batch of one".
    module function opensps_scenario_as_batch(ctx, pset, nzin, results) result(done)
        use opensps_constants, only: NEMLINE
        use opensps_csp_batch, only: compute_csp_batch
        use opensps_interpolation, only: interpolate_linear
        type(opensps_context_t), intent(inout)       :: ctx
        type(params), intent(in)                  :: pset
        integer, intent(in)                       :: nzin
        type(compspout), allocatable, intent(inout) :: results(:)
        logical :: done

        type(params), allocatable :: psets(:)
        real(WP), allocatable :: mags(:,:), spec(:,:), eml(:,:), indx(:,:), w(:,:), wh(:,:), t1(:), t2(:), wz(:,:,:)
        real(WP), allocatable :: mass(:), lbol(:), mdust(:), tage(:), mformed(:), sfr(:), frac(:), zmag(:)
        real(WP) :: mass_frac, sfr_norm, frac_linear, scale
        integer :: st, nspec, nbands, nindx, nt, n, i
        logical :: all_ages, tabular, grid_ages

        done = .false.
#ifdef ACC_MULTICORE
        ! OpenACC multicore keeps the old path: its FP32 reductions are plain per-core sums (~2e-6 of
        ! the peak here, against 4e-8 on the GPU and the CPU), which fails the single-age vs all-ages
        ! oracle (1e-6 of the peak, test_physics_oracles); multicore is to be replaced by OpenMP
        ! host threads (doc/ROADMAP.md)
        return
#endif
        ! One metallicity, or several (tabular SFHs with a metallicity column) with the bases of all
        ! of the context's metallicities resident: the batch then sums them with weights over ages
        ! and metallicities
        if (size(ctx%state%ssp_basis_spec, 3) /= nzin) return
        if (nzin > 1 .and. nzin /= ctx%state%nz) return
        if (ctx%compute_light_ages_val /= 0) return
        nt = ctx%state%ntfull
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        nindx = ctx%state%nindx
        tabular = pset%sfh == 2 .or. pset%sfh == 3
        all_ages = pset%tage <= 0.0_wp
        grid_ages = .false.

        ! Output ages [Gyr], as compute_csp_scenario
        if (pset%tage > 0.0_wp) then
            n = 1
            allocate(tage(1))
            tage(1) = pset%tage
        else if (pset%tage == -99.0_wp .and. tabular) then
            n = 1
            allocate(tage(1))
            tage(1) = maxval(ctx%state%sfh_tab(1, 1:ctx%state%ntabsfh)) / 1.0e9_wp
        else
            n = nt
            grid_ages = .true.
            allocate(tage(n))
            ! (set in the loop below)
        end if

        ! SFH weights per output (normalized to 1 Msun formed by that age; tabular SFHs: Msun
        ! formed per SSP age), mass formed and SFR
        allocate(psets(n), wh(nt, nzin), t1(nt), t2(nt), mformed(n), sfr(n), frac(n))
        if (nzin == 1) then
            allocate(w(nt, n))
        else
            allocate(wz(nt, nzin, n))
        end if
        psets = pset
        do i = 1, n
            ! The grid ages in the scalar form of compute_csp_scenario (in a loop with calls, so
            ! never vectorized): a vectorized power can differ by an ulp, and the SFH weights of
            ! truncated and Simha SFHs amplify a 1-ulp change of the age to ~1e-8 (differences
            ! of the indefinite moments)
            if (grid_ages) tage(i) = 10.0_wp**(ctx%state%time_full(i) - 9.0_wp)
            call sfh_weights_core(ctx, pset, tage(i), nzin, wh, t1, t2)
            call get_sfh_properties_at_age(ctx, pset, tage(i), mass_frac, sfr_norm, frac_linear)
            frac(i) = mass_frac
            if (all_ages) then
                scale = mass_frac
                mformed(i) = mass_frac
                sfr(i) = sfr_norm
            else
                scale = 1.0_wp
                mformed(i) = 1.0_wp
                if (mass_frac > SAFE_FLOOR) sfr_norm = sfr_norm / mass_frac
                sfr(i) = sfr_norm
            end if
            if (tabular) mformed(i) = sum(wh)
            ! The scenario's age sums skip weights that are not above SAFE_FLOOR (zero, or NaN
            ! before sf_start), so the matrix holds zero there
            where (.not. (wh > SAFE_FLOOR)) wh = 0.0_wp
            if (nzin == 1) then
                w(:, i) = wh(:, 1) * scale
            else
                wz(:, :, i) = wh * scale
            end if
            ! The weights stay readable where the scenario path leaves them (host copy)
            if (allocated(ctx%state%csp_weights)) then
                if (size(ctx%state%csp_weights, 1) >= nt .and. size(ctx%state%csp_weights, 2) >= nzin .and. &
                    size(ctx%state%csp_weights, 3) >= i) ctx%state%csp_weights(1:nt, 1:nzin, i) = wh
            end if
        end do

        ! The optional outputs and inputs are allocated only when used: an unallocated actual argument
        ! is an absent optional argument of compute_csp_batch (Fortran 2008)
        allocate(mags(nbands, n), spec(nspec, n), eml(NEMLINE, n), mass(n), lbol(n), mdust(n))
        ! (indices of the spectrum, and of no output in fast mode)
        if (pset%compute_indices == 1 .and. .not. ctx%fast_mode) allocate(indx(max(nindx, 1), n))
        if (ctx%redshift_colors_val == 1) then
            allocate(zmag(n))
            do i = 1, n
                zmag(i) = interpolate_linear(ctx%state%cosmospl(:, 2), ctx%state%cosmospl(:, 1), tage(i))
                zmag(i) = min(max(zmag(i), 0.0_wp), 20.0_wp)
            end do
        end if
        call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass, lbol=lbol, mdust=mdust, emlines=eml, &
                               indx=indx, ssp_weights=w, ssp_weights_z=wz, zred_mags=zmag)
        if (st /= 0) return

        if (allocated(results)) then
            if (size(results) /= n) deallocate(results)
        end if
        if (.not. allocated(results)) allocate(results(n))
        do i = 1, n
            if (.not. allocated(results(i)%spec)) allocate(results(i)%spec(nspec))
            if (.not. allocated(results(i)%emlines)) allocate(results(i)%emlines(NEMLINE))
            results(i)%spec = spec(:, i)
            results(i)%emlines = eml(:, i)
            if (.not. ctx%fast_mode) then
                if (.not. allocated(results(i)%mags)) allocate(results(i)%mags(nbands))
                if (.not. allocated(results(i)%indx)) allocate(results(i)%indx(nindx))
                if (pset%compute_mags == 1) then
                    results(i)%mags = mags(:, i)
                else
                    results(i)%mags = -99.0_wp
                end if
                if (allocated(indx)) then
                    results(i)%indx = indx(1:nindx, i)
                else
                    results(i)%indx = -99.0_wp
                end if
            end if
            results(i)%mass_csp = mass(i)
            results(i)%lbol_csp = lbol(i)
            ! No mass formed yet (before sf_start): the scenario path's log10 of the SAFE_FLOOR
            ! fraction
            if (all_ages .and. frac(i) <= SAFE_FLOOR) results(i)%lbol_csp = lbol(i) + log10(SAFE_FLOOR)
            results(i)%mdust = mdust(i)
            results(i)%sfr = sfr(i)
            results(i)%mformed = mformed(i)
            results(i)%age = log10(tage(i) * 1.0e9_wp)
        end do
        done = .true.
    end function opensps_scenario_as_batch

end submodule opensps_scenario_batch
