module test_opensps_csp_mod
    use opensps_precision, only: WP, KP
    use opensps_constants, only: SAFE_FLOOR, NEMLINE, C_LIGHT, NTABMAX
    use opensps_types, only: params, sfhparams, compspout
    use opensps_context_types, only: opensps_context_t
    use opensps_context, only: opensps_context_update_ssp_basis
    use opensps_csp, only: sfh_weights_core, convert_sfhparams, compute_csp_scenario
    use opensps_interpolation, only: find_interval
    use opensps_smoothing, only: apply_smoothing
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_relative_error, assert_int_equals
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    real(WP), parameter :: EPS = 1.0e-6_wp
    real(WP), parameter :: REL_EPS = 1.0e-3_wp
    ! Relative rounding floor of the kernel precision KP (2.2e-14 for FP64, 1.2e-5 for FP32); tolerances on
    ! KP-computed quantities use max(tol, KP_RTOL) so the FP64 build keeps its original tolerance
    real(WP), parameter :: KP_RTOL = 100.0_wp * real(epsilon(1.0_KP), WP)

    public :: run_opensps_csp_tests, total_failures, total_tests

contains

    subroutine run_opensps_csp_tests()
        call print_minor_header("opensps_csp")
        ! (The CSP physics is tested on real contexts: test_csp_batch, test_physics_oracles, the
        ! regression references and tools/legacy_parity; here the SFH weights, smoothing and the
        ! driver's status codes)

        call test_convert_sfhparams_units()
        call test_single_ssp_dirac()
        call test_exponential_tau_ratio()
        call test_delayed_tau_peak()
        call test_burst_addition()
        call test_tabular_zmix_interpolation()
        call test_tabular_single_z()
        call test_simha_sfh_mixing()

        call test_smoothing_conserves_flux()

        call test_driver_status_codes()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_opensps_csp_tests

    ! ---------------------------------------------------------------------
    ! Helpers
    ! ---------------------------------------------------------------------
    subroutine fill_log_grid(arr, log_min, log_max)
        real(WP), dimension(:), intent(out) :: arr
        real(WP), intent(in) :: log_min, log_max
        integer :: i, n

        n = size(arr)
        do i = 1, n
            arr(i) = log_min + (log_max - log_min) * real(i - 1, WP) / real(n - 1, WP)
        end do
    end subroutine fill_log_grid

    integer function nearest_index(arr, value) result(idx)
        real(WP), dimension(:), intent(in) :: arr
        real(WP), intent(in) :: value

        idx = minloc(abs(arr - value), dim=1)
    end function nearest_index

    subroutine setup_basic_context(ctx, time_full, spec_lambda, nz, n_bands)
        type(opensps_context_t), allocatable, intent(out) :: ctx
        real(WP), dimension(:), intent(in) :: time_full
        real(WP), dimension(:), intent(in) :: spec_lambda
        integer, intent(in) :: nz, n_bands
        integer :: i

        allocate(ctx)

        allocate(ctx%state%time_full(size(time_full)))
        allocate(ctx%state%sfh_t_calc(NTABMAX))
        allocate(ctx%state%sfh_sfr_calc(NTABMAX))
        allocate(ctx%state%sfh_age_integrand(NTABMAX))
        ctx%state%time_full = time_full
        ctx%state%ntfull = size(time_full)
        ctx%state%nspec = size(spec_lambda)
        ctx%state%nz = nz
        ctx%state%nbands = n_bands
        ctx%state%nindx = 0
        ctx%state%check_sps_setup = 1

        ctx%interpolation_type_val = 1
        ctx%tiny_logt_val = 4.0_wp

        allocate(ctx%state%spec_lambda(size(spec_lambda)))
        allocate(ctx%state%spec_nu(size(spec_lambda)))
        ctx%state%spec_lambda = spec_lambda
        ctx%state%spec_nu = C_LIGHT / spec_lambda

        allocate(ctx%state%zlegend(nz))
        do i = 1, nz
            ctx%state%zlegend(i) = 0.01_wp + 0.01_wp * real(i - 1, WP)
        end do

        if (n_bands > 0) then
            allocate(ctx%state%bands(size(spec_lambda), n_bands))
            ctx%state%bands = 1.0_wp
        end if

        ctx%compute_vega_mags_val = 0
        ctx%redshift_colors_val = 0
        ctx%add_dust_emission_val = 0
        ctx%dust_type_val = 0
        ctx%add_agn_dust_val = 0
        ctx%nebemlineinspec_val = 0
        ctx%add_neb_continuum_val = 0
        ctx%add_neb_emission_val = 0
        ctx%add_igm_absorption_val = 0
        ctx%smooth_velocity_val = 1
        ctx%smoothspec_fast_val = 0

        ctx%state%nebem_line_pos = 5500.0_wp

        if (associated(ctx%state%bands)) then
        end if
    end subroutine setup_basic_context

    subroutine teardown_basic_context(ctx)
        type(opensps_context_t), allocatable, intent(inout) :: ctx

        if (.not. allocated(ctx)) return

        if (associated(ctx%state%time_full)) then
        end if
        if (associated(ctx%state%spec_lambda)) then
        end if
        if (associated(ctx%state%spec_nu)) then
        end if
        if (associated(ctx%state%bands)) then
        end if
        if (associated(ctx%state%zlegend)) then
        end if
        if (associated(ctx%state%neb_res_min)) then
        end if
        if (associated(ctx%state%gaussnebarr)) then
        end if
        if (allocated(ctx%state%ssp_temp_grid)) then
        end if

        if (associated(ctx%state%time_full)) deallocate(ctx%state%time_full)
        if (allocated(ctx%state%sfh_t_calc)) deallocate(ctx%state%sfh_t_calc)
        if (allocated(ctx%state%sfh_sfr_calc)) deallocate(ctx%state%sfh_sfr_calc)
        if (allocated(ctx%state%sfh_age_integrand)) deallocate(ctx%state%sfh_age_integrand)
        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%spec_nu)) deallocate(ctx%state%spec_nu)
        if (associated(ctx%state%bands)) deallocate(ctx%state%bands)
        if (associated(ctx%state%zlegend)) deallocate(ctx%state%zlegend)
        if (associated(ctx%state%indexdefined)) deallocate(ctx%state%indexdefined)
        if (associated(ctx%state%neb_res_min)) deallocate(ctx%state%neb_res_min)
        if (associated(ctx%state%gaussnebarr)) deallocate(ctx%state%gaussnebarr)

        deallocate(ctx)
    end subroutine teardown_basic_context




    ! ---------------------------------------------------------------------
    ! Group 1: SFH Weight Logic
    ! ---------------------------------------------------------------------
    subroutine test_convert_sfhparams_units()
        type(params) :: pset
        type(sfhparams) :: sfh
        real(WP) :: tage

        call print_group("SFH Params: Unit Conversion")

        pset%sfh = 1
        pset%tau = 2.0_wp
        pset%tburst = 3.0_wp
        pset%sf_trunc = 1.0_wp
        pset%sf_start = 0.5_wp
        pset%sf_slope = 0.1_wp
        tage = 5.0_wp

        call convert_sfhparams(pset, tage, sfh)

        call assert_float_equals(2.0e9_wp, sfh%tau, EPS, &
                                 "tau in years", total_tests, total_failures)
        call assert_float_equals((tage - pset%sf_start) * 1.0e9_wp, sfh%tage, EPS, &
                                 "tage offset conversion", total_tests, total_failures)
        call assert_float_equals((tage - pset%tburst) * 1.0e9_wp, sfh%tb, EPS, &
                                 "tburst lookback", total_tests, total_failures)
    end subroutine test_convert_sfhparams_units

    subroutine test_single_ssp_dirac()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:), weights(:,:,:)
        real(WP), allocatable :: spec_lambda(:)
        integer :: n, idx_max, idx_tage
        real(WP) :: log_tage

        call print_group("SFH Weights: Single SSP (Dirac)")

        n = 80
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(2))
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(weights(n, 1, 1))

        pset%sfh = 0
        log_tage = log10(5.0_wp * 1.0e9_wp)

        call sfh_weights(ctx, pset, 5.0_wp, 1, 1, weights)

        call assert_float_equals(1.0_wp, sum(weights(:,1,1)), EPS, &
                                 "SSP weights sum to 1", total_tests, total_failures)
        call assert_true(count(weights(:,1,1) > 1.0e-8_wp) <= 2, &
                         "SSP weights localized to one/two bins", total_tests, total_failures)

        idx_max = maxloc(weights(:,1,1), dim=1)
        idx_tage = find_interval(time_full, log_tage)
        call assert_true(idx_max == idx_tage .or. idx_max == idx_tage + 1, &
                         "SSP peak at target age bin", total_tests, total_failures)

        deallocate(time_full, weights, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_single_ssp_dirac

    subroutine test_exponential_tau_ratio()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:), weights(:,:,:)
        real(WP), allocatable :: spec_lambda(:)
        integer :: n, i1, i2
        real(WP) :: ratio

        call print_group("SFH Weights: Exponential Tau Ratio")

        n = 120
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(2))
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(weights(n, 1, 1))

        pset%sfh = 1
        pset%tau = 1.0_wp

        call sfh_weights(ctx, pset, 10.0_wp, 1, 1, weights)

        call assert_float_equals(1.0_wp, sum(weights(:,1,1)), EPS, &
                                 "Tau weights sum to 1", total_tests, total_failures)

        i1 = nearest_index(time_full, log10(1.0e9_wp))
        i2 = nearest_index(time_full, log10(2.0e9_wp))

        call assert_true(weights(i1,1,1) > SAFE_FLOOR .and. weights(i2,1,1) > SAFE_FLOOR, &
                         "Non-zero tau weights", total_tests, total_failures)

        ratio = weights(i1,1,1) / weights(i2,1,1)
        call assert_relative_error(0.5_wp * exp(-1.0_wp), ratio, 0.2_wp, &
                                   "Tau ratio ~ 0.5 * e^-1", total_tests, total_failures)

        deallocate(time_full, weights, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_exponential_tau_ratio

    subroutine test_delayed_tau_peak()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:), weights(:,:,:)
        real(WP), allocatable :: spec_lambda(:)
        integer :: n, i_peak

        call print_group("SFH Weights: Delayed Tau Peak")

        n = 100
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(2))
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(weights(n, 1, 1))

        pset%sfh = 4
        pset%tau = 2.0_wp

        call sfh_weights(ctx, pset, 10.0_wp, 1, 1, weights)

        i_peak = nearest_index(time_full, log10(8.0e9_wp))

        call assert_true(weights(i_peak,1,1) > weights(1,1,1), &
                         "Delayed tau: Peak > Start", total_tests, total_failures)
        call assert_true(weights(i_peak,1,1) > weights(n,1,1), &
                         "Delayed tau: Peak > End", total_tests, total_failures)

        deallocate(time_full, weights, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_delayed_tau_peak

    subroutine test_burst_addition()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:), weights_base(:,:,:), weights_burst(:,:,:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP) :: fburst, excess_mass
        integer :: n, i_burst

        call print_group("SFH Weights: Burst Addition")

        n = 100
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(2))
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(weights_base(n, 1, 1), weights_burst(n, 1, 1))

        pset%sfh = 1
        pset%tau = 1.0_wp
        pset%tburst = 9.0_wp
        fburst = 0.1_wp

        pset%fburst = 0.0_wp
        call sfh_weights(ctx, pset, 10.0_wp, 1, 1, weights_base)

        pset%fburst = fburst
        call sfh_weights(ctx, pset, 10.0_wp, 1, 1, weights_burst)

        call assert_float_equals(1.0_wp, sum(weights_burst(:,1,1)), EPS, &
                                 "Burst weights sum to 1", total_tests, total_failures)

        excess_mass = sum(weights_burst(:,1,1) - (1.0_wp - fburst) * weights_base(:,1,1))
        call assert_relative_error(fburst, excess_mass, 1.0e-3_wp, &
                                   "Burst excess mass ~ fburst", total_tests, total_failures)

        i_burst = nearest_index(time_full, log10(1.0e9_wp))
        call assert_true(weights_burst(i_burst,1,1) > weights_base(i_burst,1,1), &
                         "Burst creates localized spike", total_tests, total_failures)

        deallocate(time_full, weights_base, weights_burst, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_burst_addition

    subroutine test_tabular_zmix_interpolation()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:), weights(:,:,:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP) :: mass_z1, mass_z2, ratio, dz, zbin
        integer :: n

        call print_group("SFH Weights: Tabular Z-Mix Interpolation")

        n = 60
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 9.5_wp)

        allocate(spec_lambda(2))
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 2, 1)

        ctx%state%zlegend(1) = 0.01_wp
        ctx%state%zlegend(2) = 0.03_wp

        ctx%state%ntabsfh = 2
        ctx%state%sfh_tab(:, 1) = [0.0_wp, 1.0_wp, 0.019_wp]
        ctx%state%sfh_tab(:, 2) = [1.0e9_wp, 1.0_wp, 0.019_wp]

        allocate(weights(n, 2, 1))

        pset%sfh = 3
        call sfh_weights(ctx, pset, 1.0_wp, 2, 1, weights)

        mass_z1 = sum(weights(:,1,1))
        mass_z2 = sum(weights(:,2,1))
        ratio = mass_z2 / max(mass_z1 + mass_z2, SAFE_FLOOR)

        zbin = 0.019_wp
        dz = (log10(zbin) - log10(ctx%state%zlegend(1))) / &
             (log10(ctx%state%zlegend(2)) - log10(ctx%state%zlegend(1)))

        call assert_relative_error(dz, ratio, 2.0e-2_wp, &
                                   "Metallicity split matches log interpolation", total_tests, total_failures)

        deallocate(time_full, weights, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_tabular_zmix_interpolation

    subroutine test_tabular_single_z()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:), weights(:,:,:)
        real(WP), allocatable :: spec_lambda(:)
        integer :: n

        call print_group("SFH Weights: Tabular Single Z")

        n = 100
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(2))
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        ! Setup a simple constant SFR table to make the integral easy to verify
        ctx%state%ntabsfh = 3
        ! Forward time [years], SFR [M/yr], Z
        ctx%state%sfh_tab(1, 1:3) = [0.0_wp, 1.0e9_wp, 2.0e9_wp]
        ctx%state%sfh_tab(2, 1:3) = [2.0_wp, 2.0_wp, 2.0_wp]
        ctx%state%sfh_tab(3, 1:3) = [0.02_wp, 0.02_wp, 0.02_wp]

        allocate(weights(n, 1, 1))

        pset%sfh = 3

        ! Evaluate at tage = 1.5 Gyr.
        ! Because the SFR is exactly 2.0 M_sol/yr constantly, the total formed
        ! mass by 1.5 Gyr should be exactly 3.0e9 M_sol.
        call sfh_weights(ctx, pset, 1.5_wp, 1, 1, weights)

        call assert_relative_error(3.0e9_wp, sum(weights(:,1,1)), REL_EPS, &
                                   "Tabular single Z mass matches analytic integral", total_tests, total_failures)

        deallocate(time_full, weights, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_tabular_single_z

    subroutine test_simha_sfh_mixing()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:), weights_early(:,:,:), weights_late(:,:,:), weights_pure_tau(:,:,:)
        real(WP), allocatable :: spec_lambda(:)
        integer :: n

        call print_group("SFH Weights: Simha (sfh=5) Mixing")

        n = 100
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(2))
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(weights_early(n, 1, 1), weights_late(n, 1, 1), weights_pure_tau(n, 1, 1))

        pset%sfh = 5
        pset%tau = 2.0_wp
        pset%sf_start = 1.0_wp
        pset%sf_trunc = 5.0_wp
        pset%sf_slope = 0.5_wp

        ! Test 1: Before truncation (tage = 3.0 Gyr). Should be purely delayed tau.
        call sfh_weights(ctx, pset, 3.0_wp, 1, 1, weights_early)

        ! Test 2: After truncation (tage = 7.0 Gyr). Should be a mixed profile.
        call sfh_weights(ctx, pset, 7.0_wp, 1, 1, weights_late)

        ! Reference pure delayed tau for comparison
        pset%sfh = 4
        call sfh_weights(ctx, pset, 7.0_wp, 1, 1, weights_pure_tau)

        ! Assertions: Simha handles its own normalization to 1.0 M_sol formed.
        call assert_float_equals(1.0_wp, sum(weights_early(:,1,1)), EPS, &
                                 "Early Simha weights sum to 1", total_tests, total_failures)
        call assert_float_equals(1.0_wp, sum(weights_late(:,1,1)), EPS, &
                                 "Late Simha weights sum to 1", total_tests, total_failures)

        ! Ensure the linear mixing actually activated
        call assert_true(any(abs(weights_late(:,1,1) - weights_pure_tau(:,1,1)) > 1.0e-4_wp), &
                         "Late Simha differs from pure delayed tau", total_tests, total_failures)

        deallocate(time_full, weights_early, weights_late, weights_pure_tau, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_simha_sfh_mixing

    ! ---------------------------------------------------------------------
    ! Group 2: Integration & Young/Old Split
    ! ---------------------------------------------------------------------





    ! ---------------------------------------------------------------------
    ! Group 3: Physics & Post-Processing
    ! ---------------------------------------------------------------------


    subroutine test_smoothing_conserves_flux()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP), allocatable :: spec(:)
        real(WP) :: area_before, area_after, peak_before, peak_after
        integer :: n

        call print_group("Smoothing: Flux Conservation")

        n = 120
        allocate(time_full(3))
        time_full = [6.0_wp, 8.0_wp, 10.0_wp]

        allocate(spec_lambda(n))
        call fill_log_grid(spec_lambda, 3.6_wp, 3.85_wp)
        spec_lambda = 10.0_wp**spec_lambda

        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(spec(n))
        spec = 0.0_wp
        spec(n/2) = 1.0_wp

        pset%sigma_smooth = 200.0_wp
        pset%min_wave_smooth = minval(spec_lambda)
        pset%max_wave_smooth = maxval(spec_lambda)

        ctx%smooth_velocity_val = 1

        area_before = sum(spec)
        peak_before = maxval(spec)

        ! Call smoothing directly (operates on the host arrays)
        call apply_smoothing(ctx, spec_lambda, spec, &
                             pset%sigma_smooth, pset%min_wave_smooth, pset%max_wave_smooth)

        area_after = sum(spec)
        peak_after = maxval(spec)

        call assert_true(peak_after < peak_before, "Smoothing lowers peak", total_tests, total_failures)
        call assert_relative_error(area_before, area_after, 1.0e-2_wp, "Smoothing conserves flux", total_tests, total_failures)

        deallocate(time_full, spec_lambda, spec)
        call teardown_basic_context(ctx)
    end subroutine test_smoothing_conserves_flux

    ! ---------------------------------------------------------------------
    ! Group 4: Driver Logic & Output Modes
    ! ---------------------------------------------------------------------




    ! ---------------------------------------------------------------------
    ! Group 5: Conditional Branching & Edge Cases
    ! ---------------------------------------------------------------------
    subroutine test_driver_status_codes()
        type(opensps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        real(WP), allocatable :: time_full(:), spec_lambda(:)
        ! Add temporary variables to feed the basis updater
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        integer :: status

        call print_group("Driver: Error Status Codes")

        allocate(time_full(2), spec_lambda(2))
        time_full = [6.0_wp, 10.0_wp]
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 2, 1)

        pset%sfh = 0
        pset%tage = 1.0_wp

        ! 1. Test Status 1: Setup not checked
        ctx%state%check_sps_setup = 0
        call compute_csp_scenario(ctx, pset, 1, results, status)
        call assert_int_equals(1, status, "Status 1: SPS setup unchecked", total_tests, total_failures)
        ctx%state%check_sps_setup = 1 ! Restore for next checks

        ! 2. Test Status 3: SSP basis arrays unallocated
        ! (setup_basic_context does not allocate ssp_basis_* arrays)
        call compute_csp_scenario(ctx, pset, 1, results, status)
        call assert_int_equals(3, status, "Status 3: SSP basis unallocated", total_tests, total_failures)

        ! 3. Test Status 2: Nebular emission requested with multiple metallicities
        allocate(tspec_ssp(2, 2, 2), mass_ssp(2, 2), lbol_ssp(2, 2))
        tspec_ssp = 1.0_wp; mass_ssp = 1.0_wp; lbol_ssp = 0.0_wp
        call opensps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 2)

        ctx%add_neb_emission_val = 1

        ! Pass nzin=2 to trigger the error branch
        call compute_csp_scenario(ctx, pset, 2, results, status)
        call assert_int_equals(2, status, "Status 2: Nebular emission with nzin > 1", total_tests, total_failures)

        ! 4. Status 3: a basis of 2 metallicities for nzin = 1
        ctx%add_neb_emission_val = 0
        call compute_csp_scenario(ctx, pset, 1, results, status)
        call assert_int_equals(3, status, "Status 3: basis metallicities /= nzin", total_tests, total_failures)

        ! Cleanup
        deallocate(time_full, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
        if (allocated(results)) deallocate(results)
        call teardown_basic_context(ctx)
    end subroutine test_driver_status_codes

    !> SFH weights of output i_out (sfh_weights_core with its scratch arrays)
    subroutine sfh_weights(ctx, pset, tage, nzin, i_out, weights)
        type(opensps_context_t), intent(inout) :: ctx
        type(params), intent(in)               :: pset
        real(WP), intent(in)                   :: tage
        integer, intent(in)                    :: nzin, i_out
        real(WP), intent(inout), contiguous    :: weights(:,:,:)
        real(WP), allocatable :: t1(:), t2(:)

        allocate(t1(ctx%state%ntfull), t2(ctx%state%ntfull))
        call sfh_weights_core(ctx, pset, tage, nzin, weights(:, :, i_out), t1, t2)
    end subroutine sfh_weights

end module test_opensps_csp_mod
