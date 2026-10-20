!> One code path (doc/ROADMAP.md, priority 2): compute_csp_scenario's batch of one, as a
!> submodule of fsps_csp so that fsps_csp (used by fsps_context) need not use fsps_csp_batch (which
!> uses fsps_context): the submodule is compiled after both. Called by compute_csp_scenario when
!> FSPS_SCENARIO_BATCH=1.
submodule (fsps_csp) fsps_scenario_batch
    implicit none
contains
    !> compute_csp_scenario for one age (pset%tage > 0) as a batch of one model
    !> (compute_csp_batch), packed into results(1) as the scenario path does. Returns .false.
    !> (results untouched) for the options the batch does not cover yet: several metallicities,
    !> redshift colours, light-weighted ages, fast mode (no magnitudes), OpenACC multicore builds,
    !> and anything the batch
    !> rejects (non-device smoothing, invalid dust settings: status /= 0). Design note:
    !> doc/GPU_RESULTS.md, "compute_csp_scenario as a batch of one".
    module function fsps_scenario_batch_of_one(ctx, pset, nzin, results) result(done)
        use fsps_constants, only: NEMLINE
        use fsps_csp_batch, only: compute_csp_batch
        type(fsps_context_t), intent(inout)       :: ctx
        type(params), intent(in)                  :: pset
        integer, intent(in)                       :: nzin
        type(compspout), allocatable, intent(inout) :: results(:)
        logical :: done

        type(params) :: psets(1)
        real(WP), allocatable :: mags(:,:), spec(:,:), eml(:,:), indx(:,:)
        real(WP) :: mass(1), lbol(1), mdust(1), sfr(1)
        integer :: st, nspec, nbands, nindx

        done = .false.
#ifdef ACC_MULTICORE
        ! OpenACC multicore keeps the old path: its FP32 reductions are plain per-core sums (~2e-6 of
        ! the peak here, against 4e-8 on the GPU and the CPU), which fails the single-age vs all-ages
        ! oracle (1e-6 of the peak, test_physics_oracles); multicore is to be replaced by OpenMP
        ! host threads (doc/ROADMAP.md)
        return
#endif
        if (nzin /= 1 .or. size(ctx%state%ssp_basis_spec, 3) /= 1) return
        if (ctx%redshift_colors_val /= 0 .or. ctx%compute_light_ages_val /= 0 .or. ctx%fast_mode) return
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        nindx = ctx%state%nindx
        psets(1) = pset
        allocate(mags(nbands, 1), spec(nspec, 1), eml(NEMLINE, 1), indx(max(nindx, 1), 1))
        if (pset%compute_indices == 1) then
            call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass, lbol=lbol, mdust=mdust, sfr=sfr, &
                                   emlines=eml, indx=indx)
        else
            call compute_csp_batch(ctx, psets, mags, st, spec=spec, mass=mass, lbol=lbol, mdust=mdust, sfr=sfr, &
                                   emlines=eml)
        end if
        if (st /= 0) return

        if (allocated(results)) then
            if (size(results) /= 1) deallocate(results)
        end if
        if (.not. allocated(results)) allocate(results(1))
        if (.not. allocated(results(1)%spec)) allocate(results(1)%spec(nspec))
        if (.not. allocated(results(1)%emlines)) allocate(results(1)%emlines(NEMLINE))
        if (.not. allocated(results(1)%mags)) allocate(results(1)%mags(nbands))
        if (.not. allocated(results(1)%indx)) allocate(results(1)%indx(nindx))
        results(1)%spec = spec(:, 1)
        results(1)%emlines = eml(:, 1)
        if (pset%compute_mags == 1) then
            results(1)%mags = mags(:, 1)
        else
            results(1)%mags = -99.0_wp
        end if
        if (pset%compute_indices == 1) then
            results(1)%indx = indx(1:nindx, 1)
        else
            results(1)%indx = -99.0_wp
        end if
        results(1)%mass_csp = mass(1)
        results(1)%lbol_csp = lbol(1)
        results(1)%mdust = mdust(1)
        results(1)%sfr = sfr(1)
        results(1)%mformed = 1.0_wp
        results(1)%age = log10(pset%tage * 1.0e9_wp)
        done = .true.
    end function fsps_scenario_batch_of_one

end submodule fsps_scenario_batch
