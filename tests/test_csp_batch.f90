program test_csp_batch
    !> The batched CSP (compute_csp_batch) against compute_csp_scenario, model by model: spectra,
    !> magnitudes, spectral indices, mass, L_bol, dust mass and SFR, for SFH types 0/1/4/5,
    !> birth-cloud and diffuse dust (dust types 0, 1, 2, 4), dust emission (Draine & Li parameters
    !> varied per model), nebular emission (continuum and lines; gas_logu, gas_logz and the line
    !> luminosities compared too), AGN dust emission, and redshifts 0, 0.5 (with IGM absorption)
    !> and 1.2; and per-model metallicity (logzsol varied, against compute_csp_scenario on the
    !> SSP interpolated in log Z the same way). Requires the FSPS data tree, like the regression
    !> tests.

    use fsps_precision, only: WP, KP
    use fsps_constants, only: NEMLINE
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
    use fsps_csp, only: compute_csp_scenario, set_scenario_batch
    use fsps_csp_batch, only: compute_csp_batch, batch_spectra_device, batch_spectra_host
    use, intrinsic :: iso_c_binding, only: c_ptr, c_associated, c_loc, c_size_t, c_f_pointer
#if defined(OMP_TARGET)
    use omp_lib, only: omp_target_memcpy, omp_get_initial_device, omp_get_default_device
#endif
    use fsps_ssp, only: generate_ssp_grid
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset, &
                            fsps_context_set_param_int, fsps_context_set_param_float
    use fsps_context_types, only: fsps_context_t
    use fsps_interpolation, only: find_interval
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    integer, parameter :: N_MOD = 12
    integer :: total_tests = 0
    integer :: total_failures = 0
    ! Spectra and magnitudes: the batched path sums in a different order (and integrates the
    ! filters over each band's support), so they agree to the rounding of the kernel precision
    ! FP32 kernels: 100 eps(KP), as elsewhere; OpenACC multicore runs the FP32 reductions as
    ! plain per-core sums (~4e-6), the GPU as trees (~3e-7)
    real(WP), parameter :: SPEC_RTOL = merge(1.0e-10_wp, 100.0_wp * real(epsilon(1.0_KP), WP), KP == WP)
    real(WP), parameter :: MAG_TOL = merge(1.0e-9_wp, 1.0e-5_wp, KP == WP)
    ! Spectral indices (absolute, as tools/compare_builds)
    real(WP), parameter :: INDX_TOL = merge(1.0e-8_wp, 1.0e-4_wp, KP == WP)
    type(fsps_context_t), allocatable :: ctx
    real(WP) :: smooth_sigma = 0.0_wp   ! sigma_smooth of every model
    logical  :: zred_per_model = .false.  ! models at the redshifts ZLIST instead of one
    real(WP), parameter :: ZLIST(N_MOD) = [0.0_wp, 0.1_wp, 0.5_wp, 1.2_wp, 2.0_wp, 3.0_wp, &
                                           0.02_wp, 0.8_wp, 1.6_wp, 2.5_wp, 0.3_wp, 0.0_wp]

    call print_major_header("Batched CSP vs compute_csp_scenario")

    allocate(ctx)
    call fsps_create(ctx)
    ! The reference is the old single-call path (an independent implementation) and its SFH
    ! weights, whatever FSPS_SCENARIO_BATCH says
    call set_scenario_batch(.false.)
    ctx%pset%zmet = 10
    call fsps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call fsps_context_move_to_device(ctx)
    call set_int('zmet', 10)
    call set_int('imf_type', 1)
    call set_int('add_dust_emission', 1)
    call set_int('compute_indices', 1)
    call build_basis()

    call check_batch(2, 0.0_wp, 0, 0, 0, "Calzetti, z = 0")
    call check_batch(0, 0.5_wp, 1, 0, 0, "power law, z = 0.5, IGM")
    call check_batch(1, 0.0_wp, 0, 0, 0, "Cardelli et al., z = 0")
    call check_batch(4, 1.2_wp, 0, 0, 0, "Kriek & Conroy, z = 1.2")
    call check_batch(2, 0.0_wp, 0, 1, 0, "Calzetti, z = 0, nebular")
    call check_batch(0, 0.5_wp, 1, 1, 1, "power law, z = 0.5, IGM, nebular, AGN")
    call check_batch(4, 1.2_wp, 0, 0, 1, "Kriek & Conroy, z = 1.2, AGN")
    ! Lines not in the spectrum: their absorbed luminosity enters the dust energy balance
    call set_int('nebemlineinspec', 0)
    call check_batch(1, 0.0_wp, 0, 1, 0, "Cardelli et al., z = 0, nebular, lines not in the spectrum")
    call set_int('nebemlineinspec', 1)
    ! Velocity smoothing (sigma_smooth = 150 km/s), with nebular emission and AGN
    call set_int('smooth_velocity', 1)
    smooth_sigma = 150.0_wp
    call check_batch(2, 0.5_wp, 1, 1, 1, "Calzetti, z = 0.5, IGM, nebular, AGN, sigma_smooth = 150 km/s")
    smooth_sigma = 0.0_wp
    ! Per-model redshift (IGM, resampling to the observed frame, distance per model)
    zred_per_model = .true.
    call check_batch(2, 0.0_wp, 1, 1, 1, "Calzetti, per-model zred 0-3, IGM, nebular, AGN")
    zred_per_model = .false.
    ! Per-model metallicity: all bases resident, logzsol varied
    call check_batch_z()

    call fsps_destroy(ctx)
    deallocate(ctx)
    call print_summary_line("Batched CSP", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    !> SSP basis for the default (Chabrier, solar) parameters; the batch varies only CSP parameters.
    subroutine build_basis()
        type(PARAMS) :: pset
        real(WP), allocatable :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)

        call fsps_context_prepare_pset(ctx)
        call fsps_context_get_pset(ctx, pset)
        allocate(mass_ssp(ctx%state%ntfull), lbol_ssp(ctx%state%ntfull), &
                 spec_ssp(ctx%state%nspec, ctx%state%ntfull))
        call fsps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
        call fsps_context_update_ssp_basis(ctx, reshape(spec_ssp, [size(spec_ssp, 1), size(spec_ssp, 2), 1]), &
                                           reshape(mass_ssp, [size(mass_ssp), 1]), &
                                           reshape(lbol_ssp, [size(lbol_ssp), 1]), 1)
    end subroutine build_basis

    !> Model m of the batch: SFH and dust parameters varied over their usual ranges.
    subroutine set_model(m, zred)
        integer, intent(in) :: m
        real(WP), intent(in) :: zred
        integer, parameter :: sfh(N_MOD) = [1, 4, 5, 0, 1, 4, 5, 1, 4, 1, 5, 4]
        real(WP), parameter :: tage(N_MOD) = [10.0_wp, 3.0_wp, 7.0_wp, 0.05_wp, 1.0_wp, 12.0_wp, &
                                              5.0_wp, 0.3_wp, 8.0_wp, 13.0_wp, 2.0_wp, 6.0_wp]
        real(WP), parameter :: tau(N_MOD) = [1.0_wp, 0.5_wp, 2.0_wp, 1.0_wp, 0.3_wp, 5.0_wp, &
                                             1.5_wp, 0.1_wp, 3.0_wp, 0.7_wp, 0.4_wp, 1.0_wp]
        real(WP), parameter :: dust1(N_MOD) = [0.0_wp, 0.6_wp, 1.0_wp, 0.3_wp, 0.0_wp, 0.8_wp, &
                                               0.2_wp, 1.5_wp, 0.0_wp, 0.4_wp, 0.9_wp, 0.1_wp]
        real(WP), parameter :: dust2(N_MOD) = [0.3_wp, 0.0_wp, 0.5_wp, 1.0_wp, 0.05_wp, 0.3_wp, &
                                               0.8_wp, 2.0_wp, 0.0_wp, 0.2_wp, 0.6_wp, 0.01_wp]
        real(WP), parameter :: dindex(N_MOD) = [-0.7_wp, -0.3_wp, -1.2_wp, 0.0_wp, -0.5_wp, -0.9_wp, &
                                                0.2_wp, -0.7_wp, -1.0_wp, -0.4_wp, -0.6_wp, -0.8_wp]
        real(WP), parameter :: umin(N_MOD) = [1.0_wp, 5.0_wp, 15.0_wp, 0.5_wp, 2.0_wp, 1.0_wp, &
                                              8.0_wp, 20.0_wp, 1.0_wp, 3.0_wp, 0.2_wp, 10.0_wp]
        real(WP), parameter :: qpah(N_MOD) = [3.5_wp, 0.5_wp, 4.5_wp, 2.0_wp, 1.0_wp, 3.5_wp, &
                                              0.1_wp, 3.0_wp, 2.5_wp, 4.0_wp, 1.5_wp, 3.5_wp]
        real(WP), parameter :: gam(N_MOD) = [0.01_wp, 0.1_wp, 0.0_wp, 0.3_wp, 0.01_wp, 0.05_wp, &
                                             0.5_wp, 0.01_wp, 0.2_wp, 0.0_wp, 0.02_wp, 1.0_wp]

        call set_int('sfh', sfh(m))
        call set_float('tage', tage(m))
        call set_float('tau', tau(m))
        call set_float('const', merge(0.2_wp, 0.0_wp, mod(m, 3) == 0))
        call set_float('fburst', merge(0.1_wp, 0.0_wp, mod(m, 4) == 1 .and. sfh(m) == 1))
        call set_float('tburst', 0.5_wp * tage(m))
        call set_float('sf_start', merge(0.5_wp, 0.0_wp, m == 6))
        call set_float('sf_trunc', merge(0.7_wp * tage(m), 0.0_wp, sfh(m) == 5))
        call set_float('sf_slope', merge(-1.0_wp, 0.0_wp, sfh(m) == 5))
        call set_float('dust1', dust1(m))
        call set_float('dust2', dust2(m))
        call set_float('dust_index', dindex(m))
        call set_float('dust1_index', merge(-1.0_wp, -0.7_wp, mod(m, 2) == 0))
        call set_float('dust_tesc', merge(7.5_wp, 7.0_wp, mod(m, 5) == 0))
        call set_float('frac_obrun', merge(0.2_wp, 0.0_wp, mod(m, 3) == 1))
        call set_float('frac_nodust', merge(0.1_wp, 0.0_wp, mod(m, 4) == 2))
        call set_float('mwr', merge(4.0_wp, 3.1_wp, mod(m, 2) == 1))
        call set_float('uvb', merge(0.5_wp, 1.0_wp, mod(m, 3) == 2))
        call set_float('duste_umin', umin(m))
        call set_float('duste_qpah', qpah(m))
        call set_float('duste_gamma', gam(m))
        call set_float('gas_logu', -3.5_wp + 0.25_wp * mod(m, 9))
        call set_float('gas_logz', -1.0_wp + 0.2_wp * mod(m, 7))
        call set_float('fagn', merge(0.0_wp, 0.05_wp * mod(m, 5), mod(m, 6) == 0))
        call set_float('agn_tau', 5.0_wp + 15.0_wp * mod(m, 8))
        call set_float('sigma_smooth', smooth_sigma)
        call set_float('zred', zred)
    end subroutine set_model

    subroutine check_batch(dust_type, zred, igm, neb, agn, label)
        integer, intent(in) :: dust_type, igm, neb, agn
        real(WP), intent(in) :: zred
        character(len=*), intent(in) :: label
        type(PARAMS) :: psets(N_MOD)
        type(COMPSPOUT), allocatable :: results(:)
        real(WP), allocatable :: spec(:, :), mags(:, :), ref_spec(:, :), ref_mags(:, :)
        real(WP) :: mass(N_MOD), lbol(N_MOD), mdust(N_MOD), sfr(N_MOD)
        real(WP) :: ref_mass(N_MOD), ref_lbol(N_MOD), ref_mdust(N_MOD), ref_sfr(N_MOD)
        real(WP), allocatable :: spec2(:, :), mags2(:, :)
        real(WP), allocatable :: eml(:, :), ref_eml(:, :), indx(:, :), ref_indx(:, :), ref_w(:, :)
        real(WP) :: d_spec, d_mag, d_scal, d_line, d_indx
        integer :: m, status, nspec, nbands

        call print_group("Batched CSP: " // label)
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        allocate(spec(nspec, N_MOD), mags(nbands, N_MOD), ref_spec(nspec, N_MOD), ref_mags(nbands, N_MOD))
        allocate(eml(NEMLINE, N_MOD), ref_eml(NEMLINE, N_MOD), indx(ctx%state%nindx, N_MOD), &
                 ref_indx(ctx%state%nindx, N_MOD), ref_w(ctx%state%ntfull, N_MOD))
        call set_int('dust_type', dust_type)
        call set_int('add_igm_absorption', igm)
        call set_int('add_neb_emission', neb)
        call set_int('add_agn_dust', agn)

        do m = 1, N_MOD
            call set_model(m, merge(ZLIST(m), zred, zred_per_model))
            call fsps_context_prepare_pset(ctx)
            call fsps_context_get_pset(ctx, psets(m))
            call compute_csp_scenario(ctx, psets(m), 1, results, status)
            if (status /= 0) then
                write(*, *) 'ERROR: compute_csp_scenario returned status ', status
                stop 1
            end if
            ref_spec(:, m) = results(1)%spec(1:nspec)
            ref_mags(:, m) = results(1)%mags(1:nbands)
            ref_eml(:, m) = results(1)%emlines(1:NEMLINE)
            ref_indx(:, m) = results(1)%indx(1:ctx%state%nindx)
            ref_mass(m) = results(1)%mass_csp
            ref_lbol(m) = results(1)%lbol_csp
            ref_mdust(m) = results(1)%mdust
            ref_sfr(m) = results(1)%sfr
            ref_w(:, m) = ctx%state%csp_weights(1:ctx%state%ntfull, 1, 1)
        end do

        call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol, mdust=mdust, sfr=sfr, &
                               emlines=eml, indx=indx)
        call assert_true(status == 0, "status 0", total_tests, total_failures)
        if (status /= 0) return

        call max_differences(spec, mags, ref_spec, ref_mags, d_spec, d_mag)
        d_scal = max(maxval(abs(mass / ref_mass - 1.0_wp)), maxval(abs(lbol - ref_lbol)), &
                     maxval(abs(sfr - ref_sfr) / max(abs(ref_sfr), tiny(1.0_wp))))
        write(*, '(A,4ES10.2)') "# max difference (spectrum, mag, mass/lbol/sfr, mdust): ", d_spec, d_mag, d_scal, &
            maxval(abs(mdust / ref_mdust - 1.0_wp))
        ! Lines above 1e-3 of the strongest (as tools/compare_builds), indices in absolute terms
        d_line = 0.0_wp
        if (maxval(ref_eml) > 0.0_wp) d_line = maxval(abs(eml / ref_eml - 1.0_wp), &
                                                      mask=ref_eml > 1.0e-3_wp * maxval(ref_eml))
        if (neb == 0 .and. any(eml /= 0.0_wp)) d_line = huge(1.0_wp)
        d_indx = maxval(abs(indx - ref_indx))
        write(*, '(A,2ES10.2)') "# max difference (lines, indices): ", d_line, d_indx
        call assert_true(d_line <= SPEC_RTOL, "emission lines", total_tests, total_failures)
        call assert_true(d_indx <= INDX_TOL, "spectral indices", total_tests, total_failures)
        call assert_true(d_spec <= SPEC_RTOL, "spectra", total_tests, total_failures)
        call assert_true(d_mag <= MAG_TOL, "magnitudes", total_tests, total_failures)
        call assert_true(d_scal <= 1.0e-12_wp, "mass, L_bol, SFR", total_tests, total_failures)
        call assert_true(maxval(abs(mdust / ref_mdust - 1.0_wp)) <= SPEC_RTOL, "dust mass", &
                         total_tests, total_failures)
        call check_spectra_device(spec)

        ! The SFH weights given as a matrix (non-parametric SFH input) instead of the parameters
        call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol, ssp_weights=ref_w)
        call assert_true(status == 0, "status 0 (weight matrix)", total_tests, total_failures)
        if (status /= 0) return
        call max_differences(spec, mags, ref_spec, ref_mags, d_spec, d_mag)
        d_scal = max(maxval(abs(mass / ref_mass - 1.0_wp)), maxval(abs(lbol - ref_lbol)))
        write(*, '(A,3ES10.2)') "# max difference, weight matrix (spectrum, mag, mass/lbol): ", d_spec, d_mag, d_scal
        call assert_true(d_spec <= SPEC_RTOL, "spectra (weight matrix)", total_tests, total_failures)
        call assert_true(d_mag <= MAG_TOL, "magnitudes (weight matrix)", total_tests, total_failures)
        call assert_true(d_scal <= 1.0e-12_wp, "mass, L_bol (weight matrix)", total_tests, total_failures)

        ! The same models twice: 24 >= 16 spectra take the many-spectra band kernels (and the
        ! cuBLAS product with -Dcublas=true), which the 12 above do not reach
        allocate(spec2(nspec, 2 * N_MOD), mags2(nbands, 2 * N_MOD))
        call compute_csp_batch(ctx, [psets, psets], mags2, status, spec=spec2)
        call assert_true(status == 0, "status 0 (24 models)", total_tests, total_failures)
        if (status /= 0) return
        call max_differences(spec2, mags2, reshape([ref_spec, ref_spec], [nspec, 2 * N_MOD]), &
                             reshape([ref_mags, ref_mags], [nbands, 2 * N_MOD]), d_spec, d_mag)
        write(*, '(A,2ES10.2)') "# max difference, 24 models (spectrum, mag): ", d_spec, d_mag
        call assert_true(d_spec <= SPEC_RTOL, "spectra (24 models)", total_tests, total_failures)
        call assert_true(d_mag <= MAG_TOL, "magnitudes (24 models)", total_tests, total_failures)
    end subroutine check_batch

    !> Largest relative spectrum difference (pixels above 1e-6 of each peak) and magnitude
    !> difference (bands within 30 mag of the brightest) over the columns.
    subroutine max_differences(spec, mags, ref_spec, ref_mags, d_spec, d_mag)
        real(WP), intent(in)  :: spec(:, :), mags(:, :), ref_spec(:, :), ref_mags(:, :)
        real(WP), intent(out) :: d_spec, d_mag
        real(WP) :: peak
        integer  :: m, i
        logical  :: bright(size(ref_mags, 1))

        d_spec = 0.0_wp
        d_mag = 0.0_wp
        do m = 1, size(ref_spec, 2)
            peak = maxval(ref_spec(:, m))
            do i = 1, size(ref_spec, 1)
                if (ref_spec(i, m) > 1.0e-6_wp * peak) &
                    d_spec = max(d_spec, abs(spec(i, m) / ref_spec(i, m) - 1.0_wp))
            end do
            ! Bands within 30 mag of the brightest (fainter ones carry numerically zero flux)
            bright = ref_mags(:, m) == ref_mags(:, m)
            where (bright) bright = ref_mags(:, m) < minval(ref_mags(:, m), mask=bright) + 30.0_wp
            d_mag = max(d_mag, maxval(abs(mags(:, m) - ref_mags(:, m)), mask=bright))
            if (any(bright .neqv. (mags(:, m) == mags(:, m) .and. bright))) d_mag = huge(1.0_wp)
        end do
    end subroutine max_differences

    !> Per-model metallicity: the bases of all metallicities resident, each model at its own
    !> logzsol, against compute_csp_scenario with that model's interpolated SSP as the basis
    !> (z = 0.5, IGM, nebular emission, AGN, dust emission).
    subroutine check_batch_z()
        real(WP), parameter :: logz(N_MOD) = [-1.6_wp, -1.2_wp, -0.9_wp, -0.55_wp, -0.3_wp, -0.1_wp, &
                                              0.0_wp, 0.1_wp, 0.18_wp, -0.75_wp, -1.4_wp, -0.2_wp]
        type(PARAMS) :: pset, psets(N_MOD)
        type(COMPSPOUT), allocatable :: results(:)
        real(WP), allocatable :: spec_z(:, :, :), mass_z(:, :), lbol_z(:, :)
        real(WP), allocatable :: spec(:, :), mags(:, :), ref_spec(:, :), ref_mags(:, :)
        real(WP), allocatable :: eml(:, :), ref_eml(:, :), indx(:, :), ref_indx(:, :)
        real(WP) :: mass(N_MOD), lbol(N_MOD), ref_mass(N_MOD), ref_lbol(N_MOD)
        real(WP) :: d_spec, d_mag, d_scal, d_line, d_indx, dz
        integer  :: nz, nt, nspec, nbands, z, m, status, zlo

        call print_group("Batched CSP: per-model metallicity (z = 0.5, IGM, nebular, AGN)")
        nz = ctx%state%nz
        nt = ctx%state%ntfull
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        allocate(spec_z(nspec, nt, nz), mass_z(nt, nz), lbol_z(nt, nz))
        do z = 1, nz
            call set_int('zmet', z)
            call fsps_context_prepare_pset(ctx)
            call fsps_context_get_pset(ctx, pset)
            call fsps_context_set_pset(ctx, pset)
            call generate_ssp_grid(ctx, pset, mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
        end do
        call set_int('zmet', 10)
        call set_int('dust_type', 2)
        call set_int('add_igm_absorption', 1)
        call set_int('add_neb_emission', 1)
        call set_int('add_agn_dust', 1)

        allocate(spec(nspec, N_MOD), mags(nbands, N_MOD), ref_spec(nspec, N_MOD), ref_mags(nbands, N_MOD))
        allocate(eml(NEMLINE, N_MOD), ref_eml(NEMLINE, N_MOD), indx(ctx%state%nindx, N_MOD), &
                 ref_indx(ctx%state%nindx, N_MOD))
        do m = 1, N_MOD
            call set_model(m, 0.5_wp)
            call set_float('logzsol', logz(m))
            call fsps_context_prepare_pset(ctx)
            call fsps_context_get_pset(ctx, psets(m))
            ! The interpolated SSP (compute_interpolated_ssp, simple Z interpolation)
            zlo = max(1, min(find_interval(log10(ctx%state%zlegend(1:nz) / ctx%state%zsol), logz(m)), nz - 1))
            dz = (logz(m) - log10(ctx%state%zlegend(zlo) / ctx%state%zsol)) / &
                 (log10(ctx%state%zlegend(zlo + 1) / ctx%state%zsol) - log10(ctx%state%zlegend(zlo) / ctx%state%zsol))
            ctx%state%ssp_basis_is_dirty = .true.
            call fsps_context_update_ssp_basis(ctx, &
                reshape((1.0_wp - dz) * spec_z(:, :, zlo) + dz * spec_z(:, :, zlo + 1), [nspec, nt, 1]), &
                reshape((1.0_wp - dz) * mass_z(:, zlo) + dz * mass_z(:, zlo + 1), [nt, 1]), &
                reshape((1.0_wp - dz) * lbol_z(:, zlo) + dz * lbol_z(:, zlo + 1), [nt, 1]), 1)
            call compute_csp_scenario(ctx, psets(m), 1, results, status)
            if (status /= 0) then
                write(*, *) 'ERROR: compute_csp_scenario returned status ', status
                stop 1
            end if
            ref_spec(:, m) = results(1)%spec(1:nspec)
            ref_mags(:, m) = results(1)%mags(1:nbands)
            ref_eml(:, m) = results(1)%emlines(1:NEMLINE)
            ref_indx(:, m) = results(1)%indx(1:ctx%state%nindx)
            ref_mass(m) = results(1)%mass_csp
            ref_lbol(m) = results(1)%lbol_csp
        end do

        ctx%state%ssp_basis_is_dirty = .true.
        call fsps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, nz)
        call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol, emlines=eml, indx=indx)
        call assert_true(status == 0, "status 0", total_tests, total_failures)
        if (status /= 0) return
        call max_differences(spec, mags, ref_spec, ref_mags, d_spec, d_mag)
        d_scal = max(maxval(abs(mass / ref_mass - 1.0_wp)), maxval(abs(lbol - ref_lbol)))
        d_line = maxval(abs(eml / ref_eml - 1.0_wp), mask=ref_eml > 1.0e-3_wp * maxval(ref_eml))
        d_indx = maxval(abs(indx - ref_indx))
        write(*, '(A,5ES10.2)') "# max difference (spectrum, mag, mass/lbol, lines, indices): ", &
            d_spec, d_mag, d_scal, d_line, d_indx
        call assert_true(d_spec <= SPEC_RTOL, "spectra", total_tests, total_failures)
        call assert_true(d_mag <= MAG_TOL, "magnitudes", total_tests, total_failures)
        call assert_true(d_scal <= 1.0e-12_wp, "mass, L_bol", total_tests, total_failures)
        call assert_true(d_line <= SPEC_RTOL, "emission lines", total_tests, total_failures)
        call assert_true(d_indx <= INDX_TOL, "spectral indices", total_tests, total_failures)
    end subroutine check_batch_z

    !> batch_spectra_device after a batch with spec: the spectra at the returned address (on the
    !> device in GPU builds) equal the ones copied back
    subroutine check_spectra_device(spec)
        real(WP), intent(in) :: spec(:, :)
        type(c_ptr) :: dptr
        integer :: ns, n, nb
        real(KP), allocatable, target :: buf(:, :)
        real(KP), pointer :: hp(:, :)
        logical :: ok
#if defined(_OPENACC) && !defined(ACC_MULTICORE)
        interface
            subroutine acc_d2h(dst, src, bytes) bind(C, name='acc_memcpy_from_device')
                import :: c_ptr, c_size_t
                type(c_ptr), value :: dst, src
                integer(c_size_t), value :: bytes
            end subroutine acc_d2h
        end interface
#elif defined(OMP_TARGET)
        integer :: st
#endif

        call batch_spectra_device(ctx, dptr, ns, n, nb)
        ok = c_associated(dptr) .and. ns == size(spec, 1) .and. n == size(spec, 2) .and. &
             nb == storage_size(1.0_KP) / 8
        if (ok) then
            allocate(buf(ns, n))
#if defined(_OPENACC) && !defined(ACC_MULTICORE)
            call acc_d2h(c_loc(buf), dptr, int(ns, c_size_t) * int(n, c_size_t) * int(nb, c_size_t))
#elif defined(OMP_TARGET)
            st = omp_target_memcpy(c_loc(buf), dptr, int(ns, c_size_t) * int(n, c_size_t) * int(nb, c_size_t), &
                                   0_c_size_t, 0_c_size_t, omp_get_initial_device(), omp_get_default_device())
            ok = st == 0
#else
            call c_f_pointer(dptr, hp, [ns, n])
            buf = hp
#endif
            ok = ok .and. all(real(buf, WP) == spec(1:ns, 1:n))
        end if
        call assert_true(ok, "spectra on the device (batch_spectra_device)", total_tests, total_failures)

        ! The same spectra on the host at kernel precision (batch_spectra_host)
        call batch_spectra_host(ctx, dptr, ns, n, nb)
        ok = c_associated(dptr) .and. ns == size(spec, 1) .and. n == size(spec, 2) .and. &
             nb == storage_size(1.0_KP) / 8
        if (ok) then
            call c_f_pointer(dptr, hp, [ns, n])
            ok = all(real(hp, WP) == spec(1:ns, 1:n))
        end if
        call assert_true(ok, "spectra on the host at kernel precision (batch_spectra_host)", total_tests, &
                         total_failures)
    end subroutine check_spectra_device

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer :: status

        call fsps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: status

        call fsps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_float

end program test_csp_batch
