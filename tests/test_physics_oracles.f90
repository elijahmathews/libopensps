program test_physics_oracles
    !> End-to-end checks of FSPS outputs against independent physical identities and published
    !> values, rather than against earlier output of the same code. Requires the FSPS data tree
    !> (OPENSPS_DATA_HOME / FSPS_DATA_HOME), like the regression tests.
    !>
    !>   1. Dust energy balance: with dust emission on, attenuation moves luminosity into the IR
    !>      without changing L_bol, for every dust_type.
    !>   2. Spectral vs isochrone L_bol: the frequency integral of an SSP spectrum matches the
    !>      isochrone L_bol to within the template mismatch of the stellar library (catches unit,
    !>      R^2 and template-indexing errors, which are typically factors of order unity or more).
    !>   3. Nebular H-beta vs ionizing photons: L(H-beta) / Q(H0) for a young SSP, with Q
    !>      integrated here from the spectrum without nebular emission, is the Case B value.
    !>   4. Photometric zero points: a flat f_nu = 3631 Jy source at 10 pc has M_AB = 0 in every
    !>      band, and the solar magnitudes match Willmer (2018).
    !>   5. Single-age vs all-ages CSP: tage on a time-grid node reproduces that node of the
    !>      tage = 0 (all ages) output.

    use fsps_precision, only: WP, KP
    use fsps_constants, only: NEMLINE, L_SOL, H_PLANCK, C_LIGHT
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
    use fsps_csp, only: compute_csp_scenario
    use fsps_ssp, only: generate_ssp_grid, fsps_ssp_set_per_star
    use fsps_photometry, only: compute_magnitudes
    use fsps_integration, only: integrate_trapezoid_array
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset, &
                            fsps_context_set_param_int, fsps_context_set_param_float
    use fsps_context_types, only: fsps_context_t
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true, &
                              assert_relative_error
    implicit none

    integer :: total_tests = 0
    integer :: total_failures = 0

    ! Relative rounding floor of the kernel precision KP (2.2e-14 for FP64, 1.2e-5 for FP32); tolerances on
    ! KP-computed quantities use max(tol, KP_RTOL) so the FP64 build keeps its original tolerance
    real(WP), parameter :: KP_RTOL = 100.0_wp * real(epsilon(1.0_KP), WP)

    call print_major_header("Physics Oracles (independent identities and published values)")

    call run_mist_miles_checks()
    call run_padova_basel_checks()
    call run_bpass_checks()

    call print_summary_line("Physics Oracles", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    ! ------------------------------------------------------------------------
    ! Drivers (one setup per library combination; setup dominates the run time)
    ! ------------------------------------------------------------------------

    subroutine run_mist_miles_checks()
        type(fsps_context_t) :: ctx

        call new_context(ctx, 'mist', 'miles', 10)
        call check_dust_energy_balance(ctx)
        call check_spectral_lbol(ctx, "MIST + MILES", 0.05_wp)
        call check_hbeta_case_b(ctx)
        call check_flat_ab_source(ctx)
        call check_solar_magnitudes(ctx)
        call check_single_vs_all_ages(ctx)
        call check_template_vs_per_star(ctx, "MIST + MILES")
        call fsps_destroy(ctx)
    end subroutine run_mist_miles_checks

    subroutine run_padova_basel_checks()
        type(fsps_context_t) :: ctx

        call new_context(ctx, 'pdva', 'basel', 20)
        call check_spectral_lbol(ctx, "Padova + BaSeL", 0.05_wp)
        call check_indices_defined(ctx, "Padova + BaSeL")
        call check_template_vs_per_star(ctx, "Padova + BaSeL")
        call fsps_destroy(ctx)
    end subroutine run_padova_basel_checks

    !> BPASS has no isochrones; its L_bol must still describe its spectra.
    subroutine run_bpass_checks()
        type(fsps_context_t) :: ctx

        call new_context(ctx, 'bpss', 'bpass', 10)
        call check_spectral_lbol(ctx, "BPASS", 1.0e-10_wp)
        call fsps_destroy(ctx)
    end subroutine run_bpass_checks

    ! ------------------------------------------------------------------------
    ! 1. Dust energy balance
    ! ------------------------------------------------------------------------

    !> Every dust_type with dust emission on must conserve L_bol; with it off, light is lost.
    subroutine check_dust_energy_balance(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        type(COMPSPOUT), allocatable :: results(:)
        real(WP) :: l_ref, l_dusty
        integer :: dust_type
        character(len=64) :: label

        call print_group("Dust energy balance: L_bol conserved by attenuation + emission")

        call reset_csp_params(ctx)
        call set_int(ctx, 'sfh', 1)
        call set_float(ctx, 'tau', 1.0_wp)
        call set_float(ctx, 'tage', 10.0_wp)
        call run_csp(ctx, results)
        l_ref = spectral_lbol(ctx, results(1)%spec)

        call set_float(ctx, 'dust1', 0.5_wp)
        call set_float(ctx, 'dust2', 0.3_wp)
        do dust_type = 0, 6
            call set_int(ctx, 'dust_type', dust_type)
            call run_csp(ctx, results)
            l_dusty = spectral_lbol(ctx, results(1)%spec)
            write(label, '(A,I0)') "L_bol conserved, dust_type = ", dust_type
            call assert_relative_error(l_ref, l_dusty, max(1.0e-8_wp, KP_RTOL), trim(label), total_tests, total_failures)
        end do

        ! Control: without dust emission the absorbed light must be missing
        call set_int(ctx, 'dust_type', 2)
        call set_int(ctx, 'add_dust_emission', 0)
        call run_csp(ctx, results)
        l_dusty = spectral_lbol(ctx, results(1)%spec)
        call assert_true(l_dusty < 0.95_wp * l_ref, "control: no dust emission loses L_bol", &
                         total_tests, total_failures)
        call set_int(ctx, 'add_dust_emission', 1)
        call reset_csp_params(ctx)
    end subroutine check_dust_energy_balance

    ! ------------------------------------------------------------------------
    ! 2. Spectral vs isochrone L_bol
    ! ------------------------------------------------------------------------

    !> The frequency integral of each SSP spectrum (no dust, no nebular emission) matches the
    !> SSP L_bol at every age with non-negligible light, to within rel_tol.
    subroutine check_spectral_lbol(ctx, label, rel_tol)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: label
        real(WP), intent(in) :: rel_tol
        real(WP), allocatable :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)
        real(WP) :: ratio, worst
        integer :: i, i_worst
        character(len=128) :: msg

        call print_group("Spectral vs isochrone L_bol: " // label)

        call reset_csp_params(ctx)
        call run_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)

        worst = 0.0_wp
        i_worst = 1
        do i = 1, size(lbol_ssp)
            if (lbol_ssp(i) < -8.0_wp) cycle
            ratio = spectral_lbol(ctx, spec_ssp(:, i)) / 10.0_wp**lbol_ssp(i)
            if (abs(ratio - 1.0_wp) > worst) then
                worst = abs(ratio - 1.0_wp)
                i_worst = i
            end if
        end do
        write(*, '(A,F8.5,A,F6.2)') "# max |L_spec / L_isoc - 1| = ", worst, " at log t = ", &
            ctx%state%time_full(i_worst)
        write(msg, '(A,ES8.1)') "all ages agree within ", rel_tol
        call assert_true(worst <= rel_tol, trim(msg), total_tests, total_failures)
    end subroutine check_spectral_lbol

    ! ------------------------------------------------------------------------
    ! 3. Nebular H-beta vs ionizing photons
    ! ------------------------------------------------------------------------

    !> For a 3 Myr SSP without dust, L(H-beta) / Q(H0) must be the Case B ratio. Q is integrated
    !> here, independently of fsps_gas, from the same SSP computed without nebular emission.
    !> Case B at 1e4 K and n_e = 100 cm^-3 gives 4.76e-13 erg (Osterbrock & Ferland 2006); the
    !> Cloudy grids span T_e ~ 7000-15000 K, so the ratio is allowed +-25%.
    subroutine check_hbeta_case_b(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        type(COMPSPOUT), allocatable :: results(:)
        real(WP), parameter :: LOG_AGE = 6.5_wp
        real(WP), parameter :: CASE_B_HBETA = 4.76e-13_wp
        real(WP), parameter :: NU_LYMAN_LIMIT = C_LIGHT / 911.75_wp
        integer :: i_hbeta, i_nu
        real(WP) :: q_h0, l_hbeta, ratio
        real(WP), allocatable :: photon_rate(:)

        call print_group("Nebular H-beta / Q(H0) equals Case B")

        i_hbeta = emission_line_index(ctx, 4861.0_wp)
        call assert_true(i_hbeta > 0, "H-beta found in the line list", total_tests, total_failures)
        if (i_hbeta <= 0) return

        call reset_csp_params(ctx)
        call set_int(ctx, 'sfh', 0)
        call set_float(ctx, 'tage', 10.0_wp**(LOG_AGE - 9.0_wp))

        ! Stellar spectrum alone: Q(H0) = Lsun * integral(L_nu / (h nu), nu > nu_912)
        call set_int(ctx, 'add_neb_emission', 0)
        call run_csp(ctx, results)
        photon_rate = 0.0_wp * results(1)%spec
        do i_nu = 1, size(photon_rate)
            if (ctx%state%spec_nu(i_nu) >= NU_LYMAN_LIMIT) &
                photon_rate(i_nu) = results(1)%spec(i_nu) / (H_PLANCK * ctx%state%spec_nu(i_nu))
        end do
        q_h0 = L_SOL * integrate_trapezoid_array(ctx%state%spec_nu(1:size(photon_rate)), photon_rate)

        call set_int(ctx, 'add_neb_emission', 1)
        call set_float(ctx, 'gas_logu', -2.0_wp)
        call set_float(ctx, 'gas_logz', 0.0_wp)
        call run_csp(ctx, results)
        l_hbeta = results(1)%emlines(i_hbeta) * L_SOL

        ratio = l_hbeta / q_h0
        write(*, '(A,ES10.3,A,ES10.3,A,ES10.3)') "# Q(H0) = ", q_h0, " s^-1, L(Hb) = ", l_hbeta, &
            " erg/s, ratio = ", ratio
        call assert_relative_error(CASE_B_HBETA, ratio, 0.25_wp, "L(H-beta) / Q(H0) within 25% of Case B", &
                                   total_tests, total_failures)
        call set_int(ctx, 'add_neb_emission', 0)
        call reset_csp_params(ctx)
    end subroutine check_hbeta_case_b

    ! ------------------------------------------------------------------------
    ! 4. Photometric zero points
    ! ------------------------------------------------------------------------

    !> f_nu = 3631 Jy at 10 pc has M_AB = 0 in every band, by definition of the AB system. The
    !> IRAC, PACS, SPIRE and IRAS bands are renormalized to the nu f_nu = const convention and
    !> MIPS to a beta = 2 blackbody (as in legacy FSPS), so they are excluded.
    subroutine check_flat_ab_source(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), parameter :: PC_CM = 3.0856775814913673e18_wp
        real(WP), parameter :: AB_FNU = 3631.0e-23_wp
        integer, parameter :: renormalized(17) = [53, 54, 55, 56, 90, 91, 92, 95, 96, 97, 98, 99, 100, &
                                                  101, 102, 103, 104]
        real(WP), allocatable :: spec(:), mags(:)
        logical, allocatable :: use_band(:)
        real(WP) :: lnu
        character(len=64) :: msg

        call print_group("AB zero point: flat 3631 Jy source at 10 pc")

        lnu = AB_FNU * 4.0_wp * acos(-1.0_wp) * (10.0_wp * PC_CM)**2 / L_SOL
        allocate(spec(ctx%state%nspec), mags(ctx%state%nbands))
        spec = lnu
        call compute_magnitudes(ctx, 0.0_wp, spec, mags)

        use_band = (mags == mags)
        use_band(pack(renormalized, renormalized <= size(mags))) = .false.
        write(*, '(A,ES10.3)') "# max |M_AB| = ", maxval(abs(mags), mask=use_band)
        ! FSPS uses the 48.60 zero point, i.e. 3630.8 Jy, so 3631 Jy is at -5e-5 mag
        write(msg, '(I0,A)') count(use_band), " bands give M_AB = 0 (to 1e-4 mag)"
        call assert_true(all(abs(mags) < 1.0e-4_wp .or. .not. use_band), trim(msg), total_tests, total_failures)
        call assert_true(count(use_band) > 100, "magnitudes computed for > 100 bands", &
                         total_tests, total_failures)
    end subroutine check_flat_ab_source

    !> Absolute AB magnitudes of the Sun from Willmer (2018, ApJS 236, 47), Table 3, compared
    !> with the solar magnitudes FSPS computes from its solar spectrum and filter curves.
    subroutine check_solar_magnitudes(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        integer, parameter :: n_check = 8
        integer, parameter :: band(n_check) = [1, 10, 11, 12, 14, 15, 16, 17]
        character(len=8), parameter :: names(n_check) = [character(len=8) :: &
            'V', '2MASS J', '2MASS H', '2MASS Ks', 'SDSS g', 'SDSS r', 'SDSS i', 'SDSS z']
        real(WP), parameter :: willmer_ab(n_check) = &
            [4.80_wp, 4.54_wp, 4.66_wp, 5.08_wp, 5.11_wp, 4.65_wp, 4.53_wp, 4.50_wp]
        ! Ks probes the solar spectrum where the STScI and Willmer reference spectra differ most
        real(WP), parameter :: tol(n_check) = [0.05_wp, 0.05_wp, 0.05_wp, 0.08_wp, 0.05_wp, 0.05_wp, &
                                               0.05_wp, 0.05_wp]
        integer :: k
        character(len=64) :: msg

        call print_group("Solar absolute AB magnitudes vs Willmer (2018)")

        do k = 1, n_check
            write(*, '(3A,F6.3,A,F6.3)') "# ", names(k), ": FSPS ", ctx%state%magsun(band(k)), &
                "  Willmer ", willmer_ab(k)
            write(msg, '(2A,F4.2,A)') trim(names(k)), " within ", tol(k), " mag"
            call assert_true(abs(ctx%state%magsun(band(k)) - willmer_ab(k)) < tol(k), trim(msg), &
                             total_tests, total_failures)
        end do
    end subroutine check_solar_magnitudes

    ! ------------------------------------------------------------------------
    ! 5. Single-age vs all-ages CSP
    ! ------------------------------------------------------------------------

    !> A tau model observed at a grid age reproduces the tage = 0 output at that grid node. The
    !> all-ages output is normalized to 1 Msun formed by the last age and reports the mass formed
    !> so far (mformed); a single age is normalized to 1 Msun formed by tage.
    subroutine check_single_vs_all_ages(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        type(COMPSPOUT), allocatable :: all_ages(:), single(:)
        real(WP), allocatable :: spec_all(:)
        real(WP) :: mass_all, lbol_all, t_node, mformed
        integer :: i_node, sfh_type
        character(len=96) :: msg

        call print_group("Single-age CSP equals the all-ages output at a grid node")

        do sfh_type = 1, 4, 3
            call reset_csp_params(ctx)
            call set_int(ctx, 'sfh', sfh_type)
            call set_float(ctx, 'tau', 2.0_wp)
            call set_float(ctx, 'dust2', 0.3_wp)
            call set_float(ctx, 'tage', 0.0_wp)
            call run_csp(ctx, all_ages)

            i_node = minloc(abs(all_ages(:)%age - 9.9_wp), dim=1)
            t_node = 10.0_wp**(all_ages(i_node)%age - 9.0_wp)
            mformed = all_ages(i_node)%mformed
            spec_all = all_ages(i_node)%spec(1:ctx%state%nspec) / mformed
            mass_all = all_ages(i_node)%mass_csp / mformed
            lbol_all = all_ages(i_node)%lbol_csp - log10(mformed)

            call set_float(ctx, 'tage', t_node)
            call run_csp(ctx, single)

            write(msg, '(A,I0,A,F7.4,A)') "sfh = ", sfh_type, ", tage = ", t_node, " Gyr: "
            call assert_relative_error(mass_all, single(1)%mass_csp, 1.0e-6_wp, trim(msg) // " mass", &
                                       total_tests, total_failures)
            call assert_relative_error(10.0_wp**lbol_all, 10.0_wp**single(1)%lbol_csp, 1.0e-6_wp, &
                                       trim(msg) // " L_bol", total_tests, total_failures)
            call assert_relative_error(spectral_lbol(ctx, spec_all), &
                                       spectral_lbol(ctx, single(1)%spec(1:ctx%state%nspec)), 1.0e-6_wp, &
                                       trim(msg) // " spectrum", total_tests, total_failures)
            write(*, '(A,ES10.2)') "# max |single - all-ages| / max(spectrum): ", &
                maxval(abs(single(1)%spec(1:ctx%state%nspec) - spec_all)) / maxval(spec_all)
            call assert_true(maxval(abs(single(1)%spec(1:ctx%state%nspec) - spec_all)) <= &
                             1.0e-6_wp * maxval(spec_all), trim(msg) // " spectrum element-wise", &
                             total_tests, total_failures)
        end do
        call reset_csp_params(ctx)
    end subroutine check_single_vs_all_ages

    !> The SSP as a sum over library templates (the CPU default) equals the sum over individual
    !> star spectra, with default parameters (which include AGB stars with dust shells) and with
    !> modified populations (blue stragglers, horizontal branch, stronger AGB dust).
    subroutine check_template_vs_per_star(ctx, label)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: label
        real(WP), allocatable :: mass_t(:), lbol_t(:), spec_t(:, :), mass_s(:), lbol_s(:), spec_s(:, :)
        integer :: pass
        logical :: ok
        character(len=40) :: what(2) = [character(len=40) :: "default parameters", &
                                        "sbss = 2, fbhb = 0.2, agb_dust = 2"]

        call print_group("SSP template sum equals per-star sum: " // label)
        call reset_csp_params(ctx)
        do pass = 1, 2
            if (pass == 2) then
                call set_float(ctx, 'sbss', 2.0_wp)
                call set_float(ctx, 'fbhb', 0.2_wp)
                call set_float(ctx, 'agb_dust', 2.0_wp)
            end if
            call fsps_ssp_set_per_star(.false.)
            call run_ssp(ctx, mass_t, lbol_t, spec_t)
            call fsps_ssp_set_per_star(.true.)
            call run_ssp(ctx, mass_s, lbol_s, spec_s)
            call fsps_ssp_set_per_star(.false.)
            ! FP64: every pixel to 1e-10. FP32 kernels: the two summation orders round differently,
            ! to ~1e-7 of the flux, so pixels more than 6 decades below an age's peak (near FP32
            ! underflow) are excluded
            if (KP == WP) then
                ok = all(abs(spec_t - spec_s) <= 1.0e-10_wp * max(abs(spec_s), 1.0e-30_wp * maxval(spec_s)))
            else
                ok = all(abs(spec_t - spec_s) <= KP_RTOL * abs(spec_s) .or. &
                         spec_s <= 1.0e-6_wp * spread(maxval(spec_s, dim=1), 1, size(spec_s, 1)))
            end if
            ! Mass and L_bol come from parallel reductions under OpenACC, which are not bitwise
            ! reproducible from run to run
            ok = ok .and. all(abs(mass_t - mass_s) <= 1.0e-12_wp * abs(mass_s)) .and. &
                 all(abs(lbol_t - lbol_s) <= 1.0e-12_wp * max(abs(lbol_s), 1.0_wp))
            write(*, '(A,3ES10.2)') "# max relative difference (spectrum, mass, L_bol): ", &
                maxval(abs(spec_t - spec_s) / max(abs(spec_s), 1.0e-30_wp * maxval(spec_s))), &
                maxval(abs(mass_t - mass_s) / max(abs(mass_s), tiny(1.0_wp))), maxval(abs(lbol_t - lbol_s))
            call assert_true(ok, trim(what(pass)), total_tests, total_failures)
        end do
        call set_float(ctx, 'sbss', 0.0_wp)
        call set_float(ctx, 'fbhb', 0.0_wp)
        call set_float(ctx, 'agb_dust', 1.0_wp)
    end subroutine check_template_vs_per_star

    !> Spectral indices of an ordinary old population are finite on a coarse (20 A) grid.
    subroutine check_indices_defined(ctx, label)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: label
        type(COMPSPOUT), allocatable :: results(:)

        call print_group("Spectral indices defined: " // label)

        call reset_csp_params(ctx)
        call set_int(ctx, 'sfh', 1)
        call set_float(ctx, 'tage', 10.0_wp)
        call run_csp(ctx, results)
        call assert_true(all(results(1)%indx == results(1)%indx), "no NaN indices", total_tests, total_failures)
        call reset_csp_params(ctx)
    end subroutine check_indices_defined

    ! ------------------------------------------------------------------------
    ! Helpers
    ! ------------------------------------------------------------------------

    subroutine new_context(ctx, isoc, spec, zmet)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: isoc, spec
        integer, intent(in) :: zmet

        call fsps_create(ctx)
        ctx%pset%zmet = zmet
        call fsps_setup(ctx, zmet, isoc_type_in=isoc, spec_type_in=spec, dust_type_in='DL07')
        call fsps_context_move_to_device(ctx)
        call set_int(ctx, 'zmet', zmet)
        call set_int(ctx, 'imf_type', 1)
    end subroutine new_context

    !> Restores the CSP parameters the checks change: Chabrier, SSP, no dust, no nebular.
    subroutine reset_csp_params(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        call set_int(ctx, 'sfh', 0)
        call set_float(ctx, 'tage', 0.0_wp)
        call set_float(ctx, 'tau', 1.0_wp)
        call set_float(ctx, 'dust1', 0.0_wp)
        call set_float(ctx, 'dust2', 0.0_wp)
        call set_int(ctx, 'dust_type', 0)
        call set_int(ctx, 'add_neb_emission', 0)
    end subroutine reset_csp_params

    subroutine run_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), allocatable, intent(out) :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)
        type(PARAMS) :: pset

        call fsps_context_prepare_pset(ctx)
        call fsps_context_get_pset(ctx, pset)
        allocate(mass_ssp(ctx%state%ntfull), lbol_ssp(ctx%state%ntfull), &
                 spec_ssp(ctx%state%nspec, ctx%state%ntfull))
        call fsps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
    end subroutine run_ssp

    !> SSP grid followed by the CSP on that basis (mirrors generate_opensps_fixtures).
    subroutine run_csp(ctx, results)
        type(fsps_context_t), intent(inout) :: ctx
        type(COMPSPOUT), allocatable, intent(inout) :: results(:)
        real(WP), allocatable :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)
        type(PARAMS) :: pset
        integer :: status

        call run_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
        call fsps_context_get_pset(ctx, pset)
        call fsps_context_update_ssp_basis(ctx, reshape(spec_ssp, [size(spec_ssp, 1), size(spec_ssp, 2), 1]), &
                                           reshape(mass_ssp, [size(mass_ssp), 1]), &
                                           reshape(lbol_ssp, [size(lbol_ssp), 1]), 1)
        if (allocated(results)) deallocate(results)
        call compute_csp_scenario(ctx, pset, 1, results, status)
        if (status /= 0) then
            write(*, *) 'ERROR: compute_csp_scenario returned status ', status
            stop 1
        end if
    end subroutine run_csp

    !> L_bol (Lsun) of a spectrum in Lsun/Hz.
    real(WP) function spectral_lbol(ctx, spec)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: spec(:)

        spectral_lbol = integrate_trapezoid_array(ctx%state%spec_nu(1:size(spec)), spec)
    end function spectral_lbol

    !> Index of the emission line closest to the given wavelength (Angstrom), or 0.
    integer function emission_line_index(ctx, lambda)
        use fsps_io, only: auxiliary_table_path
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: lambda
        emission_line_index = 0
        block
            real(WP) :: line_lambda
            integer :: unit_in, ios, i
            character(len=512) :: line
            ! The context's resolved data root, legacy or standard layout alike
            open(newunit=unit_in, file=auxiliary_table_path(ctx, 'emlines_info.dat'), status='old', &
                 action='read', iostat=ios)
            if (ios /= 0) return
            do i = 1, NEMLINE
                read(unit_in, '(A)', iostat=ios) line
                if (ios /= 0) exit
                read(line(1:index(line, ',') - 1), *, iostat=ios) line_lambda
                if (ios == 0 .and. abs(line_lambda - lambda) < 5.0_wp) then
                    emission_line_index = i
                    exit
                end if
            end do
            close(unit_in)
        end block
    end function emission_line_index

    subroutine set_int(ctx, key, value)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer :: status

        call fsps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_int

    subroutine set_float(ctx, key, value)
        type(fsps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: status

        call fsps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) then
            write(*, *) 'ERROR: cannot set ', key
            stop 1
        end if
    end subroutine set_float

end program test_physics_oracles
