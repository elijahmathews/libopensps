program test_csp_batch
    !> The batched CSP (compute_csp_batch), the one implementation of the CSP physics:
    !> - a batch of N models against N single calls (compute_csp_scenario, a batch of one), model
    !>   by model: spectra, magnitudes, spectral indices, mass, L_bol, dust mass and SFR, for SFH
    !>   types 0/1/4/5, birth-cloud and diffuse dust (dust types 0, 1, 2, 4), dust emission (Draine &
    !>   Li parameters varied per model), nebular emission (continuum and lines), AGN dust emission,
    !>   redshifts 0, 0.5 (with IGM absorption) and 1.2, per-model redshift and metallicity, every
    !>   smoothing mode; also the SFH weights given as a matrix, and 24 models (the many-spectra
    !>   kernels and cuBLAS products);
    !> - the IMF free per model (the template operator, ssp_op) against the basis batch with each
    !>   model's own SSP basis, with full physics;
    !> - identities: without dust, nebular emission, IGM and AGN a CSP is the SFH-weighted sum of
    !>   its SSP basis; the all-ages output divided by the mass formed equals the single-age call at
    !>   that age (in every mode, redshifted colours included); weights over ages and metallicities
    !>   are additive and index the right basis.
    !> Requires the data tree, like the regression tests.

    use opensps_precision, only: WP, KP
    use opensps_constants, only: NEMLINE
    use opensps_types, only: PARAMS, COMPSPOUT
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_csp, only: compute_csp_scenario, sfh_weights_core
    use opensps_csp_batch, only: compute_csp_batch, batch_spectra_device, batch_spectra_host
    use, intrinsic :: iso_c_binding, only: c_ptr, c_associated, c_loc, c_size_t, c_f_pointer
#if defined(OMP_TARGET)
    use omp_lib, only: omp_target_memcpy, omp_get_initial_device, omp_get_default_device
#endif
    use opensps_ssp, only: generate_ssp_grid
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_build, ssp_operator_free
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_move_to_device, &
                            opensps_context_set_pset, opensps_context_get_pset, opensps_context_prepare_pset, &
                            opensps_context_set_param_int, opensps_context_set_param_float, &
                            opensps_context_set_fast_mode
    use opensps_context_types, only: opensps_context_t
    use opensps_interpolation, only: find_interval
    use test_utils_mod, only: tap_finish, skip_without_data, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    integer, parameter :: N_MOD = 12
    integer :: total_tests = 0
    integer :: total_failures = 0
    ! Spectra and magnitudes: the batched path sums in a different order (and integrates the
    ! filters over each band's support), so they agree to the rounding of the kernel precision
    ! FP32 kernels: 100 eps(KP), as elsewhere; OpenACC multicore runs the FP32 reductions as
    ! plain per-core sums (~4e-6), the GPU as trees (~3e-7)
#if defined(OMP_TARGET) && defined(__flang__)
    ! FP64 on LLVM flang offload (AMD): the device log10 / pow differ from the host libm by ~1 ulp,
    ! and the SFH weights of 16 or more models (computed on the device) are differences of nearly
    ! equal integrals: 2.6e-10 relative in a pixel against the host weights of fewer models
    real(WP), parameter :: SPEC_RTOL = merge(1.0e-9_wp, 100.0_wp * real(epsilon(1.0_KP), WP), KP == WP)
#else
    real(WP), parameter :: SPEC_RTOL = merge(1.0e-10_wp, 100.0_wp * real(epsilon(1.0_KP), WP), KP == WP)
#endif
    real(WP), parameter :: MAG_TOL = merge(1.0e-9_wp, 1.0e-5_wp, KP == WP)
    ! Spectral indices (absolute, as tools/compare_builds)
    real(WP), parameter :: INDX_TOL = merge(1.0e-8_wp, 1.0e-4_wp, KP == WP)
    type(opensps_context_t), allocatable :: ctx
    real(WP) :: smooth_sigma = 0.0_wp   ! sigma_smooth of every model
    logical  :: zred_per_model = .false.  ! models at the redshifts ZLIST instead of one
    real(WP), parameter :: ZLIST(N_MOD) = [0.0_wp, 0.1_wp, 0.5_wp, 1.2_wp, 2.0_wp, 3.0_wp, &
                                           0.02_wp, 0.8_wp, 1.6_wp, 2.5_wp, 0.3_wp, 0.0_wp]

    call skip_without_data()
    call print_major_header("Batched CSP")

    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 10
    call opensps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call opensps_context_move_to_device(ctx)
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
    ! The IMF per model (template operator) against each model's own basis
    call check_imf_per_model()
    ! compute_csp_scenario: all-ages outputs against single-age calls, in each mode, on the
    ! one-metallicity basis
    call build_basis()
    call check_weighted_sums()
    call check_birth_cloud_split()
    call check_mass_formed()
    call check_all_ages(2, 0.0_wp, 0, 0, 0, "Calzetti")
    call check_all_ages(0, 0.5_wp, 1, 1, 1, "power law, z = 0.5, IGM, nebular, AGN")
    call set_int('smooth_velocity', 1)
    smooth_sigma = 150.0_wp
    call check_all_ages(4, 1.2_wp, 1, 0, 1, "Kriek & Conroy, z = 1.2, IGM, AGN, sigma_smooth = 150 km/s")
    ! The smoothing modes the batch runs on the host: the exact velocity convolution
    ! (smoothspec_fast = 0) and a constant width in wavelength (smooth_velocity = 0, 5 A)
    call set_int('smoothspec_fast', 0)
    call check_batch(2, 0.5_wp, 1, 1, 1, "Calzetti, z = 0.5, IGM, nebular, AGN, exact sigma_smooth = 150 km/s")
    call check_all_ages(2, 0.5_wp, 1, 1, 1, "Calzetti, z = 0.5, IGM, nebular, AGN, exact sigma_smooth = 150 km/s")
    call set_int('smoothspec_fast', 1)
    call set_int('smooth_velocity', 0)
    smooth_sigma = 5.0_wp
    call check_batch(2, 0.0_wp, 0, 0, 1, "Calzetti, AGN, sigma_smooth = 5 A")
    call check_all_ages(2, 0.0_wp, 0, 0, 1, "Calzetti, AGN, sigma_smooth = 5 A")
    call set_int('smooth_velocity', 1)
    smooth_sigma = 0.0_wp
    ! Redshifted colours: the magnitudes of each output at the redshift of its age
    call set_int('redshift_colors', 1)
    call check_all_ages(2, 0.5_wp, 1, 1, 1, "Calzetti, z = 0.5, IGM, nebular, AGN, redshifted colours")
    call set_int('redshift_colors', 0)
    ! Fast mode: the same (smoothed) spectrum, without magnitudes and indices
    call check_fast_mode()
    ! Weights over ages and metallicities do not fill the nebular kernels' weights: rejected
    call check_weights_z_nebular()
    ! Weights over ages and metallicities (several metallicities; tabular SFHs with Z per bin)
    call check_weights_z()

    call opensps_destroy(ctx)
    deallocate(ctx)
    call print_summary_line("Batched CSP", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    !> Fast mode skips the magnitudes and indices only: with velocity smoothing (sigma_smooth =
    !> 150 km/s) and dust, nebular emission and AGN, its spectrum equals the normal call's
    !> (smoothed: it differs from the unsmoothed one), at one age and at all ages.
    subroutine check_fast_mode()
        type(PARAMS) :: pset
        type(COMPSPOUT), allocatable :: normal(:), fast(:), raw(:)
        integer :: k, i, st1, st2, st3
        logical :: same, smoothed

        call print_group("Fast mode: the normal call's spectrum, smoothing included")
        call set_int('dust_type', 2)
        call set_int('add_neb_emission', 1)
        call set_int('add_agn_dust', 1)
        call set_int('smooth_velocity', 1)
        do k = 1, 2
            call set_model(1, 0.0_wp)
            if (k == 2) call set_float('tage', 0.0_wp)
            call set_float('sigma_smooth', 0.0_wp)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call compute_csp_scenario(ctx, pset, 1, raw, st1)
            call set_float('sigma_smooth', 150.0_wp)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call compute_csp_scenario(ctx, pset, 1, normal, st2)
            call opensps_context_set_fast_mode(ctx, .true.)
            call compute_csp_scenario(ctx, pset, 1, fast, st3)
            call opensps_context_set_fast_mode(ctx, .false.)
            same = st1 == 0 .and. st2 == 0 .and. st3 == 0 .and. size(fast) == size(normal)
            smoothed = same
            if (same) then
                do i = 1, size(normal)
                    same = same .and. all(fast(i)%spec == normal(i)%spec)
                end do
                smoothed = any(normal(size(normal))%spec /= raw(size(raw))%spec)
            end if
            call assert_true(same, merge("one age:  fast = normal spectrum", "all ages: fast = normal spectrum", &
                                         k == 1), total_tests, total_failures)
            call assert_true(smoothed, merge("one age:  smoothed", "all ages: smoothed", k == 1), &
                             total_tests, total_failures)
        end do
        call set_float('sigma_smooth', 0.0_wp)
        call set_int('add_neb_emission', 0)
        call set_int('add_agn_dust', 0)
    end subroutine check_fast_mode

    !> ssp_weights_z with nebular emission returns status 2, also on the one-metallicity basis
    !> (where nothing else would stop it and the nebular kernels would read unset weights).
    subroutine check_weights_z_nebular()
        type(PARAMS) :: pset
        real(WP), allocatable :: wz(:, :, :), mags(:, :)
        integer :: status

        call print_group("ssp_weights_z with nebular emission")
        call set_int('add_neb_emission', 1)
        call set_model(1, 0.0_wp)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        allocate(wz(ctx%state%ntfull, 1, 1), mags(ctx%state%nbands, 1))
        wz = 1.0e-3_wp
        call compute_csp_batch(ctx, [pset], mags, status, ssp_weights_z=wz)
        call assert_true(status == 2, "one resident basis: status 2", total_tests, total_failures)
        call set_int('add_neb_emission', 0)
    end subroutine check_weights_z_nebular

    !> SSP basis for the default (Chabrier, solar) parameters; the batch varies only CSP parameters.
    subroutine build_basis()
        type(PARAMS) :: pset
        real(WP), allocatable :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)

        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        allocate(mass_ssp(ctx%state%ntfull), lbol_ssp(ctx%state%ntfull), &
                 spec_ssp(ctx%state%nspec, ctx%state%ntfull))
        call opensps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
        call opensps_context_update_ssp_basis(ctx, reshape(spec_ssp, [size(spec_ssp, 1), size(spec_ssp, 2), 1]), &
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
        real(WP), allocatable :: eml(:, :), ref_eml(:, :), indx(:, :), ref_indx(:, :), ref_w(:, :), t1(:), t2(:)
        real(WP) :: d_spec, d_mag, d_scal, d_line, d_indx
        integer :: m, status, nspec, nbands

        call print_group("Batched CSP: " // label)
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        allocate(spec(nspec, N_MOD), mags(nbands, N_MOD), ref_spec(nspec, N_MOD), ref_mags(nbands, N_MOD))
        allocate(eml(NEMLINE, N_MOD), ref_eml(NEMLINE, N_MOD), indx(ctx%state%nindx, N_MOD), &
                 ref_indx(ctx%state%nindx, N_MOD), ref_w(ctx%state%ntfull, N_MOD), t1(ctx%state%ntfull), &
                 t2(ctx%state%ntfull))
        call set_int('dust_type', dust_type)
        call set_int('add_igm_absorption', igm)
        call set_int('add_neb_emission', neb)
        call set_int('add_agn_dust', agn)

        do m = 1, N_MOD
            call set_model(m, merge(ZLIST(m), zred, zred_per_model))
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, psets(m))
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
            call sfh_weights_core(ctx, psets(m), psets(m)%tage, 1, ref_w(:, m:m), t1, t2)
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





    !> The IMF free per model: a batch with the template operator (Kroupa slopes varied per model)
    !> against, model by model, the basis batch with the SSP basis generated for that model's IMF;
    !> z = 0.5, IGM, nebular emission, AGN, dust emission.
    subroutine check_imf_per_model()
        type(PARAMS) :: pset, psets(N_MOD)
        type(ssp_operator_t) :: op
        real(WP), allocatable :: spec(:, :), mags(:, :), ref_spec(:, :), ref_mags(:, :), eml(:, :), ref_eml(:, :)
        real(WP), allocatable :: s1(:, :), g1(:, :), e1(:, :)
        real(WP) :: mass(N_MOD), lbol(N_MOD), mdust(N_MOD), ref_mass(N_MOD), ref_lbol(N_MOD), ref_mdust(N_MOD)
        real(WP) :: d_spec, d_mag, d_scal, d_line, imf(3, N_MOD)
        integer  :: m, status, nspec, nbands

        call print_group("Batched CSP: IMF per model (template operator), Kroupa slopes varied")
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        allocate(spec(nspec, N_MOD), mags(nbands, N_MOD), ref_spec(nspec, N_MOD), ref_mags(nbands, N_MOD), &
                 eml(NEMLINE, N_MOD), ref_eml(NEMLINE, N_MOD), s1(nspec, 1), g1(nbands, 1), e1(NEMLINE, 1))
        call set_int('dust_type', 2)
        call set_int('add_igm_absorption', 1)
        call set_int('add_neb_emission', 1)
        call set_int('add_agn_dust', 1)
        call set_int('imf_type', 2)
        do m = 1, N_MOD
            imf(:, m) = [0.5_wp + 0.1_wp * m, 1.6_wp + 0.08_wp * m, 1.9_wp + 0.1_wp * mod(7 * m, N_MOD)]
        end do

        ! The operator (built with the default slopes; S does not depend on them)
        call set_model(1, 0.5_wp)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        call ssp_operator_build(ctx, pset, op)
        do m = 1, N_MOD
            call set_model(m, 0.5_wp)
            call set_float('imf1', imf(1, m)); call set_float('imf2', imf(2, m)); call set_float('imf3', imf(3, m))
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, psets(m))
        end do
        call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol, mdust=mdust, emlines=eml, &
                               ssp_op=op)
        call assert_true(status == 0, "status 0", total_tests, total_failures)

        ! Reference: each model with its own basis
        do m = 1, N_MOD
            call set_float('imf1', imf(1, m)); call set_float('imf2', imf(2, m)); call set_float('imf3', imf(3, m))
            call build_basis()
            call compute_csp_batch(ctx, psets(m:m), g1, status, spec=s1, mass=ref_mass(m:m), lbol=ref_lbol(m:m), &
                                   mdust=ref_mdust(m:m), emlines=e1)
            ref_spec(:, m) = s1(:, 1)
            ref_mags(:, m) = g1(:, 1)
            ref_eml(:, m) = e1(:, 1)
        end do
        call max_differences(spec, mags, ref_spec, ref_mags, d_spec, d_mag)
        d_scal = max(maxval(abs(mass / ref_mass - 1.0_wp)), maxval(abs(lbol - ref_lbol)))
        d_line = maxval(abs(eml / ref_eml - 1.0_wp), mask=ref_eml > 1.0e-3_wp * maxval(ref_eml))
        write(*, '(A,5ES10.2)') "# max difference (spectrum, mag, mass/lbol, mdust, lines): ", d_spec, d_mag, &
            d_scal, maxval(abs(mdust / ref_mdust - 1.0_wp)), d_line
        write(*, '(A,2F9.5)') "# the IMF matters: mass of models 1 and 12: ", ref_mass(1), ref_mass(N_MOD)
        call assert_true(d_spec <= SPEC_RTOL, "spectra", total_tests, total_failures)
        call assert_true(d_mag <= MAG_TOL, "magnitudes", total_tests, total_failures)
        call assert_true(d_scal <= 1.0e-12_wp, "mass, L_bol", total_tests, total_failures)
        call assert_true(maxval(abs(mdust / ref_mdust - 1.0_wp)) <= SPEC_RTOL, "dust mass", total_tests, total_failures)
        call assert_true(d_line <= SPEC_RTOL, "emission lines", total_tests, total_failures)

        call ssp_operator_free(op)
        call set_int('imf_type', 1)
        call set_float('imf1', 1.3_wp); call set_float('imf2', 2.3_wp); call set_float('imf3', 2.3_wp)
    end subroutine check_imf_per_model

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
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call opensps_context_set_pset(ctx, pset)
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
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, psets(m))
            ! The interpolated SSP (compute_interpolated_ssp, simple Z interpolation)
            zlo = max(1, min(find_interval(log10(ctx%state%zlegend(1:nz) / ctx%state%zsol), logz(m)), nz - 1))
            dz = (logz(m) - log10(ctx%state%zlegend(zlo) / ctx%state%zsol)) / &
                 (log10(ctx%state%zlegend(zlo + 1) / ctx%state%zsol) - log10(ctx%state%zlegend(zlo) / ctx%state%zsol))
            ctx%state%ssp_basis_is_dirty = .true.
            call opensps_context_update_ssp_basis(ctx, &
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
        call opensps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, nz)
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

    !> A single-age CSP without dust, nebular emission, IGM, AGN or smoothing is the SFH-weighted
    !> sum of its SSP basis: spec = sum_t w_t spec_ssp(t), mass = sum_t w_t mass_ssp(t),
    !> L_bol = sum_t w_t L_ssp(t), with the weights of sfh_weights_core (models 1-6: sfh 1, 4, 5,
    !> 0 and the sf_start case).
    subroutine check_weighted_sums()
        type(PARAMS) :: pset
        type(COMPSPOUT), allocatable :: results(:)
        real(WP), allocatable :: w(:, :), t1(:), t2(:), ref(:)
        real(WP) :: mass, lsum, d_spec, d_scal, peak
        integer  :: m, t, i, nt, nspec, status

        call print_group("compute_csp_scenario = SFH-weighted sum of the SSP basis (no dust, nebular, IGM, AGN)")
        nt = ctx%state%ntfull
        nspec = ctx%state%nspec
        allocate(w(nt, 1), t1(nt), t2(nt), ref(nspec))
        call set_int('dust_type', 0)
        call set_int('add_igm_absorption', 0)
        call set_int('add_neb_emission', 0)
        call set_int('add_agn_dust', 0)
        d_spec = 0.0_wp
        d_scal = 0.0_wp
        do m = 1, 6
            call set_model(m, 0.0_wp)
            call set_float('dust1', 0.0_wp)
            call set_float('dust2', 0.0_wp)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call compute_csp_scenario(ctx, pset, 1, results, status)
            call assert_true(status == 0, "status 0", total_tests, total_failures)
            if (status /= 0) return
            call sfh_weights_core(ctx, pset, pset%tage, 1, w, t1, t2)
            ref = 0.0_wp
            mass = 0.0_wp
            lsum = 0.0_wp
            do t = 1, nt
                if (.not. (w(t, 1) > 0.0_wp)) cycle
                ref = ref + w(t, 1) * real(ctx%state%ssp_basis_spec(1:nspec, t, 1), WP)
                mass = mass + w(t, 1) * ctx%state%ssp_basis_mass(t, 1)
                lsum = lsum + w(t, 1) * 10.0_wp**ctx%state%ssp_basis_lbol(t, 1)
            end do
            peak = maxval(ref)
            do i = 1, nspec
                if (ref(i) > 1.0e-6_wp * peak) d_spec = max(d_spec, abs(results(1)%spec(i) / ref(i) - 1.0_wp))
            end do
            d_scal = max(d_scal, abs(results(1)%mass_csp / mass - 1.0_wp), abs(results(1)%lbol_csp - log10(lsum)))
        end do
        write(*, '(A,2ES10.2)') "# max difference (spectrum, mass/lbol): ", d_spec, d_scal
        call assert_true(d_spec <= SPEC_RTOL, "spectra", total_tests, total_failures)
        call assert_true(d_scal <= 1.0e-12_wp, "mass, L_bol", total_tests, total_failures)
    end subroutine check_weighted_sums

    !> The two-component dust model built by hand from the SSP basis: power-law birth-cloud
    !> (dust1, dust1_index) and diffuse (dust2, dust_index) attenuation, normalized at 5500 A,
    !> spec = T_diffuse (sum_{t: log age <= dust_tesc} w_t T_birth S_t + sum_{t: older} w_t S_t)
    !> (Charlot & Fall 2000; legacy FSPS's young/old split at dust_tesc), models 1-6 (dust_tesc
    !> 7.0 and 7.5), no dust emission, nebular emission, IGM or AGN.
    subroutine check_birth_cloud_split()
        type(PARAMS) :: pset
        type(COMPSPOUT), allocatable :: results(:)
        real(WP), allocatable :: w(:, :), t1(:), t2(:), young(:), old(:), ref(:), lam(:)
        real(WP) :: d_spec, peak
        integer  :: m, t, i, nt, nspec, status, n_young, n_old

        call print_group("compute_csp_scenario = birth-cloud and diffuse dust built from the SSP basis")
        nt = ctx%state%ntfull
        nspec = ctx%state%nspec
        allocate(w(nt, 1), t1(nt), t2(nt), young(nspec), old(nspec), ref(nspec), lam(nspec))
        lam = ctx%state%spec_lambda(1:nspec)
        call set_int('dust_type', 0)
        call set_int('add_dust_emission', 0)
        call set_int('add_igm_absorption', 0)
        call set_int('add_neb_emission', 0)
        call set_int('add_agn_dust', 0)
        d_spec = 0.0_wp
        n_young = 0
        n_old = 0
        do m = 1, 6
            call set_model(m, 0.0_wp)
            call set_float('dust1', 1.0_wp)
            call set_float('dust2', 0.3_wp)
            call set_float('dust3', 0.0_wp)
            call set_float('frac_obrun', 0.0_wp)
            call set_float('frac_nodust', 0.0_wp)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call compute_csp_scenario(ctx, pset, 1, results, status)
            call assert_true(status == 0, "status 0", total_tests, total_failures)
            if (status /= 0) return
            call sfh_weights_core(ctx, pset, pset%tage, 1, w, t1, t2)
            young = 0.0_wp
            old = 0.0_wp
            do t = 1, nt
                if (.not. (w(t, 1) > 0.0_wp)) cycle
                if (ctx%state%time_full(t) <= pset%dust_tesc) then
                    young = young + w(t, 1) * real(ctx%state%ssp_basis_spec(1:nspec, t, 1), WP)
                    n_young = n_young + 1
                else
                    old = old + w(t, 1) * real(ctx%state%ssp_basis_spec(1:nspec, t, 1), WP)
                    n_old = n_old + 1
                end if
            end do
            ref = exp(-pset%dust2 * (lam / 5500.0_wp)**pset%dust_index) * &
                  (young * exp(-pset%dust1 * (lam / 5500.0_wp)**pset%dust1_index) + old)
            peak = maxval(ref)
            do i = 1, nspec
                if (ref(i) > 1.0e-6_wp * peak) d_spec = max(d_spec, abs(results(1)%spec(i) / ref(i) - 1.0_wp))
            end do
        end do
        call set_int('add_dust_emission', 1)
        write(*, '(A,ES10.2)') "# max difference: ", d_spec
        call assert_true(n_young > 0 .and. n_old > 0, "young and old ages both present", total_tests, total_failures)
        call assert_true(d_spec <= max(1.0e-12_wp, SPEC_RTOL), "spectra", total_tests, total_failures)
    end subroutine check_birth_cloud_split

    !> Mass formed against closed forms. (1) All ages, tau model without burst or constant part:
    !> output i is per unit mass formed over the full history (to the oldest isochrone age Tmax),
    !> so mformed = (1 - exp(-t_i / tau)) / (1 - exp(-Tmax / tau)), and its spectrum (no dust,
    !> nebular emission, IGM or AGN) is that fraction times the SFH-weighted sum of the basis at
    !> t_i (outputs with at least 1e-3 of the final mass). (2) A constant 1 Msun/yr tabular SFH
    !> from 0 to 6 Gyr: mformed = (t - 10**tiny_logt yr) Msun at tage = 5 Gyr and at the last
    !> table age (tage = -99).
    subroutine check_mass_formed()
        type(PARAMS) :: pset
        type(COMPSPOUT), allocatable :: results(:)
        real(WP), allocatable :: w(:, :), t1(:), t2(:), ref(:)
        real(WP), parameter :: TAU = 1.5_wp
        real(WP) :: tmax, tg, frac, d_mf, d_spec, peak
        integer  :: i, j, t, nt, nspec, status, k

        call print_group("Mass formed against closed forms (tau model at all ages, constant tabular SFR)")
        nt = ctx%state%ntfull
        nspec = ctx%state%nspec
        allocate(w(nt, 1), t1(nt), t2(nt), ref(nspec))
        call set_int('dust_type', 0)
        call set_int('add_igm_absorption', 0)
        call set_int('add_neb_emission', 0)
        call set_int('add_agn_dust', 0)
        call set_model(1, 0.0_wp)
        call set_float('dust1', 0.0_wp)
        call set_float('dust2', 0.0_wp)
        call set_float('tau', TAU)
        call set_float('const', 0.0_wp)
        call set_float('fburst', 0.0_wp)
        call set_float('sf_start', 0.0_wp)
        call set_float('sf_trunc', 0.0_wp)
        call set_float('tage', 0.0_wp)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        call compute_csp_scenario(ctx, pset, 1, results, status)
        call assert_true(status == 0 .and. size(results) == nt, "all ages: status 0, ntfull outputs", &
                         total_tests, total_failures)
        if (status /= 0) return
        tmax = 10.0_wp**(ctx%state%time_full(nt) - 9.0_wp)
        d_mf = 0.0_wp
        d_spec = 0.0_wp
        do i = 1, nt
            tg = 10.0_wp**(ctx%state%time_full(i) - 9.0_wp)
            frac = (1.0_wp - exp(-tg / TAU)) / (1.0_wp - exp(-tmax / TAU))
            if (frac < 1.0e-3_wp) cycle
            d_mf = max(d_mf, abs(results(i)%mformed / frac - 1.0_wp))
            call sfh_weights_core(ctx, pset, tg, 1, w, t1, t2)
            ref = 0.0_wp
            do t = 1, nt
                if (w(t, 1) > 0.0_wp) ref = ref + w(t, 1) * real(ctx%state%ssp_basis_spec(1:nspec, t, 1), WP)
            end do
            ref = frac * ref
            peak = maxval(ref)
            do j = 1, nspec
                if (ref(j) > 1.0e-6_wp * peak) d_spec = max(d_spec, abs(results(i)%spec(j) / ref(j) - 1.0_wp))
            end do
        end do
        write(*, '(A,2ES10.2)') "# all ages, max difference (mass formed, spectrum): ", d_mf, d_spec
        call assert_true(d_mf <= 1.0e-10_wp, "all ages: mass formed", total_tests, total_failures)
        call assert_true(d_spec <= max(1.0e-10_wp, SPEC_RTOL), "all ages: spectra", total_tests, total_failures)

        ctx%state%sfh_tab = 0.0_wp
        ctx%state%ntabsfh = 3
        ctx%state%sfh_tab(1, 1:3) = [0.0_wp, 3.0e9_wp, 6.0e9_wp]
        ctx%state%sfh_tab(2, 1:3) = 1.0_wp
        call set_int('sfh', 3)
        do k = 1, 2
            call set_float('tage', merge(5.0_wp, -99.0_wp, k == 1))
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call compute_csp_scenario(ctx, pset, 1, results, status)
            d_mf = huge(1.0_wp)
            if (status == 0) d_mf = abs(results(1)%mformed / (merge(5.0e9_wp, 6.0e9_wp, k == 1) - &
                                    10.0_wp**ctx%tiny_logt_val) - 1.0_wp)
            write(*, '(A,ES10.2)') "# tabular, difference of the mass formed: ", d_mf
            call assert_true(d_mf <= 1.0e-10_wp, merge("tabular, tage = 5:   mass formed", &
                                                        "tabular, tage = -99: mass formed", k == 1), &
                             total_tests, total_failures)
        end do
        call set_int('sfh', 1)
    end subroutine check_mass_formed

    !> All-ages mode (tage = 0) against single-age calls: output i, per unit mass formed over the
    !> full history, divided by the fraction formed by then (mformed), equals the call at that age
    !> (10**(time_full(i) - 9) Gyr, formed as compute_csp_scenario forms it), for models 1, 2, 3
    !> and 6 (sfh 1, 4, 5, 4 with sf_start) at three ages with at least 1e-3 of the final mass
    !> formed: every output is linear in the SFH weights (the AGN torus through L_bol too). The
    !> magnitudes differ by 2.5 log10(mformed), SFR by the factor mformed; indices not at all.
    subroutine check_all_ages(dust_type, zred, igm, neb, agn, label)
        integer, intent(in) :: dust_type, igm, neb, agn
        real(WP), intent(in) :: zred
        character(len=*), intent(in) :: label
        integer, parameter :: models(4) = [1, 2, 3, 6]
        real(WP), parameter :: LOGT(3) = [8.5_wp, 9.3_wp, 9.9_wp]
        type(PARAMS) :: pset
        type(COMPSPOUT), allocatable :: ra(:), rs(:)
        real(WP) :: d(5), f, peak, tg
        integer  :: im, k, i, j, nspec, nbands, st
        logical  :: ok

        call print_group("compute_csp_scenario, all ages = single ages: " // label)
        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        call set_int('dust_type', dust_type)
        call set_int('add_igm_absorption', igm)
        call set_int('add_neb_emission', neb)
        call set_int('add_agn_dust', agn)
        d = 0.0_wp
        ok = .true.
        do im = 1, size(models)
            call set_model(models(im), zred)
            call set_float('tage', 0.0_wp)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call compute_csp_scenario(ctx, pset, 1, ra, st)
            ok = ok .and. st == 0 .and. size(ra) == ctx%state%ntfull
            if (.not. ok) exit
            do k = 1, size(LOGT)
                i = minloc(abs(ctx%state%time_full(1:ctx%state%ntfull) - LOGT(k)), dim=1)
                f = ra(i)%mformed
                if (f < 1.0e-3_wp * maxval(ra(:)%mformed)) cycle
                tg = 10.0_wp**(ctx%state%time_full(i) - 9.0_wp)
                pset%tage = tg
                call compute_csp_scenario(ctx, pset, 1, rs, st)
                ok = ok .and. st == 0
                if (.not. ok) exit
                peak = maxval(rs(1)%spec(1:nspec))
                do j = 1, nspec
                    if (rs(1)%spec(j) > 1.0e-6_wp * peak) &
                        d(1) = max(d(1), abs(ra(i)%spec(j) / f / rs(1)%spec(j) - 1.0_wp))
                end do
                do j = 1, nbands
                    if (rs(1)%mags(j) == rs(1)%mags(j) .and. rs(1)%mags(j) < minval(rs(1)%mags, &
                        mask=rs(1)%mags == rs(1)%mags) + 30.0_wp) &
                        d(2) = max(d(2), abs(ra(i)%mags(j) + 2.5_wp * log10(f) - rs(1)%mags(j)))
                end do
                d(3) = max(d(3), abs(ra(i)%mass_csp / f / rs(1)%mass_csp - 1.0_wp), &
                           abs(ra(i)%lbol_csp - log10(f) - rs(1)%lbol_csp), &
                           abs(ra(i)%sfr - rs(1)%sfr * f) / max(abs(ra(i)%sfr), tiny(1.0_wp)), &
                           abs(ra(i)%age - rs(1)%age))
                if (maxval(rs(1)%emlines) > 0.0_wp) d(4) = max(d(4), maxval(abs(ra(i)%emlines / f / &
                    rs(1)%emlines - 1.0_wp), mask=rs(1)%emlines > 1.0e-3_wp * maxval(rs(1)%emlines)))
                d(5) = max(d(5), maxval(abs(ra(i)%indx - rs(1)%indx)))
            end do
        end do
        write(*, '(A,5ES10.2)') "# max difference (spectrum, mag, scalars, lines, indices): ", d
        call assert_true(ok, "status 0, ntfull outputs", total_tests, total_failures)
        call assert_true(d(1) <= SPEC_RTOL, "spectra", total_tests, total_failures)
        call assert_true(d(2) <= MAG_TOL, "magnitudes", total_tests, total_failures)
        call assert_true(d(3) <= max(1.0e-12_wp, SPEC_RTOL), "mass, L_bol, SFR, age", total_tests, total_failures)
        call assert_true(d(4) <= SPEC_RTOL, "emission lines", total_tests, total_failures)
        call assert_true(d(5) <= INDX_TOL, "spectral indices", total_tests, total_failures)
    end subroutine check_all_ages

    !> Weights over ages and metallicities (ssp_weights_z), the bases of all metallicities
    !> resident: the same SFH weights in metallicities z1 alone, z2 alone and both, with dust,
    !> dust emission, IGM and AGN (all linear in the weights): both = z1 + z2 (spectra, mass,
    !> L_bol); z1 alone = the plain weight matrix on the basis of z1 alone. Then
    !> compute_csp_scenario with nzin = nz: a tabular SFH with a metallicity per bin, at one age,
    !> at the end of the table (tage = -99) and at all ages.
    subroutine check_weights_z()
        type(PARAMS) :: pset, psets(1)
        type(COMPSPOUT), allocatable :: results(:)
        real(WP), allocatable :: spec_z(:, :, :), mass_z(:, :), lbol_z(:, :), wz(:, :, :), w(:, :), t1(:), t2(:)
        real(WP), allocatable :: s(:, :), mags(:, :)
        real(WP) :: mass(3), lbol(3), m1(1), l1(1), d_add, d_one, peak
        real(WP), parameter :: tab_age(5) = [0.0_wp, 2.0_wp, 6.0_wp, 10.0_wp, 13.0_wp]   ! Gyr
        real(WP), parameter :: tab_sfr(5) = [0.5_wp, 3.0_wp, 2.0_wp, 1.0_wp, 0.4_wp]     ! Msun/yr
        real(WP), parameter :: tab_met(5) = [0.0004_wp, 0.002_wp, 0.008_wp, 0.014_wp, 0.025_wp]
        integer, parameter :: Z1 = 4, Z2 = 9
        integer  :: nz, nt, nspec, z, k, i, st

        call print_group("Weights over ages and metallicities (ssp_weights_z); tabular SFH with Z per bin")
        nz = ctx%state%nz
        nt = ctx%state%ntfull
        nspec = ctx%state%nspec
        allocate(spec_z(nspec, nt, nz), mass_z(nt, nz), lbol_z(nt, nz), wz(nt, nz, 3), w(nt, 1), t1(nt), t2(nt), &
                 s(nspec, 3), mags(ctx%state%nbands, 3))
        do z = 1, nz
            call set_int('zmet', z)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call opensps_context_set_pset(ctx, pset)
            call generate_ssp_grid(ctx, pset, mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
        end do
        call set_int('zmet', 10)
        call opensps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, nz)
        call set_int('dust_type', 2)
        call set_int('add_igm_absorption', 1)
        call set_int('add_neb_emission', 0)
        call set_int('add_agn_dust', 1)
        call set_model(1, 0.5_wp)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        call sfh_weights_core(ctx, pset, pset%tage, 1, w, t1, t2)
        where (.not. (w > 0.0_wp)) w = 0.0_wp
        wz = 0.0_wp
        wz(:, Z1, 1) = w(:, 1)
        wz(:, Z2, 2) = w(:, 1)
        wz(:, Z1, 3) = w(:, 1)
        wz(:, Z2, 3) = w(:, 1)
        call compute_csp_batch(ctx, [pset, pset, pset], mags, st, spec=s, mass=mass, lbol=lbol, ssp_weights_z=wz)
        call assert_true(st == 0, "status 0 (ssp_weights_z)", total_tests, total_failures)
        if (st /= 0) return
        peak = maxval(s(:, 3))
        d_add = max(abs((mass(1) + mass(2)) / mass(3) - 1.0_wp), &
                    abs(log10(10.0_wp**lbol(1) + 10.0_wp**lbol(2)) - lbol(3)))
        do i = 1, nspec
            if (s(i, 3) > 1.0e-6_wp * peak) d_add = max(d_add, abs((s(i, 1) + s(i, 2)) / s(i, 3) - 1.0_wp))
        end do
        ! z1 alone against the one-metallicity basis of z1 with the plain weight matrix
        call opensps_context_update_ssp_basis(ctx, spec_z(:, :, Z1:Z1), mass_z(:, Z1:Z1), lbol_z(:, Z1:Z1), 1)
        psets(1) = pset
        call compute_csp_batch(ctx, psets, mags(:, 1:1), st, spec=s(:, 2:2), mass=m1, lbol=l1, ssp_weights=w)
        call assert_true(st == 0, "status 0 (one-metallicity basis)", total_tests, total_failures)
        d_one = max(abs(m1(1) / mass(1) - 1.0_wp), abs(l1(1) - lbol(1)))
        peak = maxval(s(:, 1))
        do i = 1, nspec
            if (s(i, 1) > 1.0e-6_wp * peak) d_one = max(d_one, abs(s(i, 2) / s(i, 1) - 1.0_wp))
        end do
        write(*, '(A,2ES10.2)') "# max difference (z1 + z2 vs both, z1 vs one-metallicity basis): ", d_add, d_one
        call assert_true(d_add <= max(1.0e-12_wp, SPEC_RTOL), "both metallicities = sum of each", &
                         total_tests, total_failures)
        call assert_true(d_one <= max(1.0e-12_wp, SPEC_RTOL), "one metallicity = the one-metallicity basis", &
                         total_tests, total_failures)

        ! A tabular SFH with a metallicity per bin through compute_csp_scenario (nzin = nz)
        call opensps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, nz)
        ctx%state%sfh_tab = 0.0_wp
        ctx%state%ntabsfh = size(tab_age)
        ctx%state%sfh_tab(1, 1:size(tab_age)) = tab_age * 1.0e9_wp
        ctx%state%sfh_tab(2, 1:size(tab_age)) = tab_sfr
        ctx%state%sfh_tab(3, 1:size(tab_age)) = tab_met
        call set_int('sfh', 3)
        do k = 1, 3
            call set_float('tage', merge(8.0_wp, merge(-99.0_wp, 0.0_wp, k == 2), k == 1))
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call compute_csp_scenario(ctx, pset, nz, results, st)
            call assert_true(st == 0 .and. size(results) == merge(nt, 1, k == 3) .and. &
                             all(results(:)%mass_csp == results(:)%mass_csp) .and. results(1)%mformed > 0.0_wp, &
                             trim(merge("tabular, Z per bin, tage = 8  ", "tabular, Z per bin, tage = -99", k == 1)) // &
                             trim(merge("            ", ", all ages  ", k /= 3)) // ": status 0, finite", &
                             total_tests, total_failures)
        end do
        call set_int('sfh', 1)
        call build_basis()
    end subroutine check_weights_z

    !> batch_spectra_device after a batch with spec: the spectra at the returned address (on the
    !> device in GPU builds) equal the ones copied back
    subroutine check_spectra_device(spec)
        real(WP), intent(in) :: spec(:, :)
        type(c_ptr) :: dptr
        integer :: ns, n, nb
        real(KP), allocatable, target :: buf(:, :)
        real(KP), pointer :: hp(:, :)
        logical :: ok
#if defined(OMP_TARGET)
        integer :: st
#endif

        call batch_spectra_device(ctx, dptr, ns, n, nb)
        ok = c_associated(dptr) .and. ns == size(spec, 1) .and. n == size(spec, 2) .and. &
             nb == storage_size(1.0_KP) / 8
        if (ok) then
            allocate(buf(ns, n))
#if defined(OMP_TARGET)
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

        call opensps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: status

        call opensps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_float

end program test_csp_batch
