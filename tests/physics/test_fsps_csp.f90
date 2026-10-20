module test_fsps_csp_mod
    use fsps_precision, only: WP, KP
    use fsps_constants, only: SAFE_FLOOR, NEMLINE, C_LIGHT, NTABMAX
    use fsps_types, only: params, sfhparams, compspout
    use fsps_context_types, only: fsps_context_t
    use fsps_context, only: fsps_context_prepare_csp_workspace, fsps_context_update_ssp_basis
    use fsps_csp, only: set_scenario_batch, compute_sfh_weights, convert_sfhparams, integrate_csp_step, apply_dust_physics, &
                        apply_post_processing, compute_csp_scenario
    use fsps_interpolation, only: find_interval
    use fsps_cosmology, only: compute_igm_transmission
    use fsps_smoothing, only: apply_smoothing
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

    public :: run_fsps_csp_tests, total_failures, total_tests

contains

    subroutine run_fsps_csp_tests()
        call print_minor_header("fsps_csp")
        ! These tests inspect the old single-call path's internal state (SFH weights, IGM
        ! transmission): run it whatever FSPS_SCENARIO_BATCH says
        call set_scenario_batch(.false.)

        call test_convert_sfhparams_units()
        call test_single_ssp_dirac()
        call test_exponential_tau_ratio()
        call test_delayed_tau_peak()
        call test_burst_addition()
        call test_tabular_zmix_interpolation()
        call test_tabular_single_z()
        call test_simha_sfh_mixing()

        call test_all_old_limit()
        call test_all_young_limit()
        call test_split_boundary()
        call test_mass_sum_consistency()
        call test_hotpath_equivalence()

        call test_dust_screen_logic()
        call test_igm_absorption_toggle()
        call test_smoothing_conserves_flux()

        call test_snapshot_vs_history()
        call test_mass_normalization_surviving()
        call test_nebular_precalculation()
        call test_fast_mode_skips_optional_outputs()

        call test_driver_status_codes()
        call test_driver_igm_precalc()
        call test_driver_tabular_tage_99()
        call test_tabular_mass_formed()

        call test_single_age_outputs_are_weighted_sums()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_csp_tests

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
        type(fsps_context_t), allocatable, intent(out) :: ctx
        real(WP), dimension(:), intent(in) :: time_full
        real(WP), dimension(:), intent(in) :: spec_lambda
        integer, intent(in) :: nz, n_bands
        integer :: i

        allocate(ctx)

        allocate(ctx%state%time_full(size(time_full)))
        allocate(ctx%state%sfh_w_tmp1(max(size(time_full), 3), size(time_full)))
        allocate(ctx%state%sfh_w_tmp2(max(size(time_full), 3), size(time_full)))
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
        ctx%compute_light_ages_val = 0
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

        !$acc enter data copyin(ctx)
        !$acc enter data copyin(ctx%state)
        !$acc enter data copyin(ctx%state%time_full, ctx%state%spec_lambda, ctx%state%spec_nu, ctx%state%zlegend)
        !$acc enter data attach(ctx%state%time_full)
        !$acc enter data attach(ctx%state%spec_lambda)
        !$acc enter data attach(ctx%state%spec_nu)
        !$acc enter data attach(ctx%state%zlegend)
        if (associated(ctx%state%bands)) then
            !$acc enter data copyin(ctx%state%bands)
            !$acc enter data attach(ctx%state%bands)
        end if
    end subroutine setup_basic_context

    subroutine teardown_basic_context(ctx)
        type(fsps_context_t), allocatable, intent(inout) :: ctx

        if (.not. allocated(ctx)) return

        if (associated(ctx%state%time_full)) then
            !$acc exit data delete(ctx%state%time_full)
        end if
        if (associated(ctx%state%spec_lambda)) then
            !$acc exit data delete(ctx%state%spec_lambda)
        end if
        if (associated(ctx%state%spec_nu)) then
            !$acc exit data delete(ctx%state%spec_nu)
        end if
        if (associated(ctx%state%bands)) then
            !$acc exit data delete(ctx%state%bands)
        end if
        if (associated(ctx%state%zlegend)) then
            !$acc exit data delete(ctx%state%zlegend)
        end if
        if (associated(ctx%state%neb_res_min)) then
            !$acc exit data delete(ctx%state%neb_res_min)
        end if
        if (associated(ctx%state%gaussnebarr)) then
            !$acc exit data delete(ctx%state%gaussnebarr)
        end if
        if (allocated(ctx%state%ssp_temp_grid)) then
            !$acc exit data delete(ctx%state%ssp_temp_grid)
        end if
        !$acc exit data delete(ctx%state)
        !$acc exit data delete(ctx)

        if (associated(ctx%state%time_full)) deallocate(ctx%state%time_full)
        if (allocated(ctx%state%sfh_w_tmp1)) deallocate(ctx%state%sfh_w_tmp1)
        if (allocated(ctx%state%sfh_w_tmp2)) deallocate(ctx%state%sfh_w_tmp2)
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

    subroutine map_csp_workspace_to_device(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (allocated(ctx%state%csp_weights)) then
            !$acc enter data copyin(ctx%state%csp_ssp_grid)
            !$acc enter data attach(ctx%state%csp_ssp_grid)
            !$acc enter data copyin(ctx%state%csp_emlin_grid)
            !$acc enter data attach(ctx%state%csp_emlin_grid)
            !$acc enter data copyin(ctx%state%csp_ssp_lum_linear)
            !$acc enter data attach(ctx%state%csp_ssp_lum_linear)
            !$acc enter data copyin(ctx%state%csp_igm_transmission)
            !$acc enter data attach(ctx%state%csp_igm_transmission)
            !$acc enter data copyin(ctx%state%csp_spec_final)
            !$acc enter data attach(ctx%state%csp_spec_final)
            !$acc enter data copyin(ctx%state%csp_emlin_final)
            !$acc enter data attach(ctx%state%csp_emlin_final)
            !$acc enter data copyin(ctx%state%csp_weights)
            !$acc enter data attach(ctx%state%csp_weights)
            !$acc enter data copyin(ctx%state%spec_young)
            !$acc enter data attach(ctx%state%spec_young)
            !$acc enter data copyin(ctx%state%spec_old)
            !$acc enter data attach(ctx%state%spec_old)
            !$acc enter data copyin(ctx%state%csp_emlin_young)
            !$acc enter data attach(ctx%state%csp_emlin_young)
            !$acc enter data copyin(ctx%state%csp_emlin_old)
            !$acc enter data attach(ctx%state%csp_emlin_old)

            if (allocated(ctx%state%out_csp_spec)) then
                !$acc enter data copyin(ctx%state%out_csp_spec)
                !$acc enter data attach(ctx%state%out_csp_spec)
                !$acc enter data copyin(ctx%state%out_csp_emlin)
                !$acc enter data attach(ctx%state%out_csp_emlin)
                !$acc enter data copyin(ctx%state%out_mass_csp)
                !$acc enter data attach(ctx%state%out_mass_csp)
                !$acc enter data copyin(ctx%state%out_lbol_csp)
                !$acc enter data attach(ctx%state%out_lbol_csp)
            end if
        end if

        ! Map Dust & Gas Workspaces
        if (allocated(ctx%state%dust_transmission_diffuse)) then
            !$acc enter data copyin(ctx%state%dust_transmission_diffuse)
            !$acc enter data attach(ctx%state%dust_transmission_diffuse)
        end if
        if (allocated(ctx%state%dust_frequencies)) then
            !$acc enter data copyin(ctx%state%dust_frequencies)
            !$acc enter data attach(ctx%state%dust_frequencies)
        end if
        if (allocated(ctx%state%dust_spec_total_work)) then
            !$acc enter data copyin(ctx%state%dust_spec_total_work)
            !$acc enter data attach(ctx%state%dust_spec_total_work)
        end if
        if (allocated(ctx%state%dust_emission_shape)) then
            !$acc enter data copyin(ctx%state%dust_emission_shape)
            !$acc enter data attach(ctx%state%dust_emission_shape)
        end if
        if (allocated(ctx%state%dust_emission_final)) then
            !$acc enter data copyin(ctx%state%dust_emission_final)
            !$acc enter data attach(ctx%state%dust_emission_final)
        end if
        if (allocated(ctx%state%gas_neb_cont_reduced)) then
            !$acc enter data copyin(ctx%state%gas_neb_cont_reduced)
            !$acc enter data attach(ctx%state%gas_neb_cont_reduced)
        end if
        if (allocated(ctx%state%gas_neb_line_reduced)) then
            !$acc enter data copyin(ctx%state%gas_neb_line_reduced)
            !$acc enter data attach(ctx%state%gas_neb_line_reduced)
        end if
        if (allocated(ctx%state%gas_current_step_cont)) then
            !$acc enter data copyin(ctx%state%gas_current_step_cont)
            !$acc enter data attach(ctx%state%gas_current_step_cont)
        end if
        if (allocated(ctx%state%gas_current_step_lines)) then
            !$acc enter data copyin(ctx%state%gas_current_step_lines)
            !$acc enter data attach(ctx%state%gas_current_step_lines)
        end if
        if (allocated(ctx%state%scalar_reductions)) then
            !$acc enter data copyin(ctx%state%scalar_reductions)
            !$acc enter data attach(ctx%state%scalar_reductions)
        end if
        if (allocated(ctx%state%sfh_t_calc)) then
            !$acc enter data copyin(ctx%state%sfh_t_calc)
            !$acc enter data attach(ctx%state%sfh_t_calc)
        end if
        if (allocated(ctx%state%sfh_sfr_calc)) then
            !$acc enter data copyin(ctx%state%sfh_sfr_calc)
            !$acc enter data attach(ctx%state%sfh_sfr_calc)
        end if
        if (allocated(ctx%state%sfh_age_integrand)) then
            !$acc enter data copyin(ctx%state%sfh_age_integrand)
            !$acc enter data attach(ctx%state%sfh_age_integrand)
        end if
        if (allocated(ctx%state%sfh_w_tmp1)) then
            !$acc enter data copyin(ctx%state%sfh_w_tmp1)
            !$acc enter data attach(ctx%state%sfh_w_tmp1)
        end if
        if (allocated(ctx%state%sfh_w_tmp2)) then
            !$acc enter data copyin(ctx%state%sfh_w_tmp2)
            !$acc enter data attach(ctx%state%sfh_w_tmp2)
        end if
    end subroutine map_csp_workspace_to_device

    subroutine unmap_csp_workspace_from_device(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        if (allocated(ctx%state%csp_weights)) then
            if (allocated(ctx%state%ssp_basis_spec)) then
                !$acc exit data delete(ctx%state%ssp_basis_spec)
            end if
            if (allocated(ctx%state%ssp_basis_mass)) then
                !$acc exit data delete(ctx%state%ssp_basis_mass)
            end if
            if (allocated(ctx%state%ssp_basis_lbol)) then
                !$acc exit data delete(ctx%state%ssp_basis_lbol)
            end if
            !$acc exit data delete(ctx%state%csp_weights)
            !$acc exit data delete(ctx%state%spec_young)
            !$acc exit data delete(ctx%state%spec_old)
            !$acc exit data delete(ctx%state%csp_emlin_young)
            !$acc exit data delete(ctx%state%csp_emlin_old)
            !$acc exit data delete(ctx%state%csp_ssp_grid)
            !$acc exit data delete(ctx%state%csp_emlin_grid)
            !$acc exit data delete(ctx%state%csp_ssp_lum_linear)
            !$acc exit data delete(ctx%state%csp_spec_final)
            !$acc exit data delete(ctx%state%csp_emlin_final)

            if (allocated(ctx%state%out_csp_spec)) then
                !$acc exit data delete(ctx%state%out_csp_spec)
                !$acc exit data delete(ctx%state%out_csp_emlin)
                !$acc exit data delete(ctx%state%out_mass_csp)
                !$acc exit data delete(ctx%state%out_lbol_csp)
            end if
        end if

        if (allocated(ctx%state%csp_igm_transmission)) then
            !$acc exit data delete(ctx%state%csp_igm_transmission)
        end if

        ! Unmap Dust & Gas Workspaces
        if (allocated(ctx%state%dust_transmission_diffuse)) then
            !$acc exit data delete(ctx%state%dust_transmission_diffuse)
        end if
        if (allocated(ctx%state%dust_frequencies)) then
            !$acc exit data delete(ctx%state%dust_frequencies)
        end if
        if (allocated(ctx%state%dust_spec_total_work)) then
            !$acc exit data delete(ctx%state%dust_spec_total_work)
        end if
        if (allocated(ctx%state%dust_emission_shape)) then
            !$acc exit data delete(ctx%state%dust_emission_shape)
        end if
        if (allocated(ctx%state%dust_emission_final)) then
            !$acc exit data delete(ctx%state%dust_emission_final)
        end if
        if (allocated(ctx%state%gas_neb_cont_reduced)) then
            !$acc exit data delete(ctx%state%gas_neb_cont_reduced)
        end if
        if (allocated(ctx%state%gas_neb_line_reduced)) then
            !$acc exit data delete(ctx%state%gas_neb_line_reduced)
        end if
        if (allocated(ctx%state%gas_current_step_cont)) then
            !$acc exit data delete(ctx%state%gas_current_step_cont)
        end if
        if (allocated(ctx%state%gas_current_step_lines)) then
            !$acc exit data delete(ctx%state%gas_current_step_lines)
        end if
        if (allocated(ctx%state%scalar_reductions)) then
            !$acc exit data delete(ctx%state%scalar_reductions)
        end if
        if (allocated(ctx%state%sfh_t_calc)) then
            !$acc exit data delete(ctx%state%sfh_t_calc)
        end if
        if (allocated(ctx%state%sfh_sfr_calc)) then
            !$acc exit data delete(ctx%state%sfh_sfr_calc)
        end if
        if (allocated(ctx%state%sfh_age_integrand)) then
            !$acc exit data delete(ctx%state%sfh_age_integrand)
        end if
        if (allocated(ctx%state%sfh_w_tmp1)) then
            !$acc exit data delete(ctx%state%sfh_w_tmp1)
        end if
        if (allocated(ctx%state%sfh_w_tmp2)) then
            !$acc exit data delete(ctx%state%sfh_w_tmp2)
        end if
    end subroutine unmap_csp_workspace_from_device

    subroutine setup_nebular_line_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        integer :: i

        ctx%add_neb_emission_val = 1
        ctx%add_neb_continuum_val = 0
        ctx%nebemlineinspec_val = 1
        ctx%setup_nebular_gaussians_val = 0
        ctx%smooth_velocity_val = 0

        if (.not. associated(ctx%state%neb_res_min)) then
            allocate(ctx%state%neb_res_min(NEMLINE))
        end if
        if (.not. associated(ctx%state%gaussnebarr)) then
            allocate(ctx%state%gaussnebarr(ctx%state%nspec, NEMLINE))
        end if

        ctx%state%neb_res_min = 1.0_wp
        ctx%state%nebem_line_pos = 6563.0_wp

        do i = 1, size(ctx%state%nebem_logz)
            ctx%state%nebem_logz(i) = -2.0_wp + 0.2_wp * real(i - 1, WP)
        end do
        do i = 1, size(ctx%state%nebem_logu)
            ctx%state%nebem_logu(i) = -4.0_wp + 0.5_wp * real(i - 1, WP)
        end do
        do i = 1, size(ctx%state%nebem_age)
            ctx%state%nebem_age(i) = 6.0_wp + 0.4_wp * real(i - 1, WP)
        end do

        ! Physical scale (Halpha: log L/Q ~ -45.4 in L_sun per photon/s), within FP32 range
        ctx%state%nebem_line = -99.0_wp
        ctx%state%nebem_line(1, :, :, :) = -45.0_wp

        if (associated(ctx%state%neb_res_min)) then
            !$acc enter data copyin(ctx%state%neb_res_min)
            !$acc enter data attach(ctx%state%neb_res_min)
        end if
        if (associated(ctx%state%gaussnebarr)) then
            !$acc enter data copyin(ctx%state%gaussnebarr)
            !$acc enter data attach(ctx%state%gaussnebarr)
        end if
        !$acc update device(ctx%add_neb_emission_val, ctx%add_neb_continuum_val, ctx%nebemlineinspec_val)
        !$acc update device(ctx%setup_nebular_gaussians_val, ctx%smooth_velocity_val)
        !$acc update device(ctx%state%nebem_line_pos, ctx%state%nebem_logz, ctx%state%nebem_logu)
        !$acc update device(ctx%state%nebem_age, ctx%state%nebem_line)
    end subroutine setup_nebular_line_context

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
        type(fsps_context_t), allocatable :: ctx
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

        call compute_sfh_weights(ctx, pset, 5.0_wp, 1, 1, weights)

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
        type(fsps_context_t), allocatable :: ctx
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

        call compute_sfh_weights(ctx, pset, 10.0_wp, 1, 1, weights)

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
        type(fsps_context_t), allocatable :: ctx
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

        call compute_sfh_weights(ctx, pset, 10.0_wp, 1, 1, weights)

        i_peak = nearest_index(time_full, log10(8.0e9_wp))

        call assert_true(weights(i_peak,1,1) > weights(1,1,1), &
                         "Delayed tau: Peak > Start", total_tests, total_failures)
        call assert_true(weights(i_peak,1,1) > weights(n,1,1), &
                         "Delayed tau: Peak > End", total_tests, total_failures)

        deallocate(time_full, weights, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_delayed_tau_peak

    subroutine test_burst_addition()
        type(fsps_context_t), allocatable :: ctx
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
        call compute_sfh_weights(ctx, pset, 10.0_wp, 1, 1, weights_base)

        pset%fburst = fburst
        call compute_sfh_weights(ctx, pset, 10.0_wp, 1, 1, weights_burst)

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
        type(fsps_context_t), allocatable :: ctx
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
        call compute_sfh_weights(ctx, pset, 1.0_wp, 2, 1, weights)

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
        type(fsps_context_t), allocatable :: ctx
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
        call compute_sfh_weights(ctx, pset, 1.5_wp, 1, 1, weights)

        call assert_relative_error(3.0e9_wp, sum(weights(:,1,1)), REL_EPS, &
                                   "Tabular single Z mass matches analytic integral", total_tests, total_failures)

        deallocate(time_full, weights, spec_lambda)
        call teardown_basic_context(ctx)
    end subroutine test_tabular_single_z

    subroutine test_simha_sfh_mixing()
        type(fsps_context_t), allocatable :: ctx
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
        call compute_sfh_weights(ctx, pset, 3.0_wp, 1, 1, weights_early)

        ! Test 2: After truncation (tage = 7.0 Gyr). Should be a mixed profile.
        call compute_sfh_weights(ctx, pset, 7.0_wp, 1, 1, weights_late)

        ! Reference pure delayed tau for comparison
        pset%sfh = 4
        call compute_sfh_weights(ctx, pset, 7.0_wp, 1, 1, weights_pure_tau)

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
    subroutine test_all_old_limit()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(KP), allocatable :: ssp_grid(:,:,:), emlin_grid(:,:,:)
        real(WP), allocatable :: mass_ssp(:,:), ssp_lum(:,:)
        integer :: n

        call print_group("CSP Integrator: All Old Limit")

        n = 40
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(1))
        spec_lambda = [5500.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(ssp_grid(1, n, 1))
        allocate(emlin_grid(NEMLINE, n, 1))
        allocate(mass_ssp(n, 1))
        allocate(ssp_lum(n, 1))

        ssp_grid = 1.0_wp
        emlin_grid = 0.0_wp
        mass_ssp = 1.0_wp
        ssp_lum = 1.0_wp

        call fsps_context_prepare_csp_workspace(ctx)
        call map_csp_workspace_to_device(ctx)

        pset%sfh = 0
        pset%dust_tesc = 3.0_wp

        !$acc data copyin(ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        ctx%state%sfh_t_calc(1) = 10.0_wp
        !$acc update device(ctx%state%sfh_t_calc(1:1))
        call integrate_csp_step(ctx, pset, 1, 1, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        !$acc wait(1)
        !$acc end data

        !$acc update host(ctx%state%spec_young, ctx%state%spec_old)
        call unmap_csp_workspace_from_device(ctx)

        call assert_true(all(abs(ctx%state%spec_young(:, 1)) <= 1.0e-8_wp), "Young component ~0", total_tests, total_failures)
        call assert_true(sum(ctx%state%spec_old(:, 1)) > 0.0_wp, "Old component carries flux", total_tests, total_failures)

        deallocate(time_full, spec_lambda, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        call teardown_basic_context(ctx)
    end subroutine test_all_old_limit

    subroutine test_all_young_limit()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(KP), allocatable :: ssp_grid(:,:,:), emlin_grid(:,:,:)
        real(WP), allocatable :: mass_ssp(:,:), ssp_lum(:,:)
        integer :: n

        call print_group("CSP Integrator: All Young Limit")

        n = 40
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(1))
        spec_lambda = [5500.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(ssp_grid(1, n, 1))
        allocate(emlin_grid(NEMLINE, n, 1))
        allocate(mass_ssp(n, 1))
        allocate(ssp_lum(n, 1))

        ssp_grid = 1.0_wp
        emlin_grid = 0.0_wp
        mass_ssp = 1.0_wp
        ssp_lum = 1.0_wp

        call fsps_context_prepare_csp_workspace(ctx)
        call map_csp_workspace_to_device(ctx)

        pset%sfh = 1
        pset%tau = 2.0_wp
        pset%dust_tesc = 10.15_wp

        !$acc data copyin(ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        ctx%state%sfh_t_calc(1) = 10.0_wp
        !$acc update device(ctx%state%sfh_t_calc(1:1))
        call integrate_csp_step(ctx, pset, 1, 1, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        !$acc wait(1)
        !$acc end data

        !$acc update host(ctx%state%spec_young, ctx%state%spec_old)
        call unmap_csp_workspace_from_device(ctx)

        call assert_true(all(abs(ctx%state%spec_old(:, 1)) <= 1.0e-8_wp), "Old component ~0", total_tests, total_failures)
        call assert_true(sum(ctx%state%spec_young(:, 1)) > 0.0_wp, "Young component carries flux", total_tests, total_failures)

        deallocate(time_full, spec_lambda, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        call teardown_basic_context(ctx)
    end subroutine test_all_young_limit

    subroutine test_split_boundary()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(KP), allocatable :: ssp_grid(:,:,:), emlin_grid(:,:,:)
        real(WP), allocatable :: mass_ssp(:,:), ssp_lum(:,:)
        real(WP) :: expected_young, expected_old
        integer :: n, k

        call print_group("CSP Integrator: Split Boundary")

        n = 50
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(1))
        spec_lambda = [5500.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(ssp_grid(1, n, 1))
        allocate(emlin_grid(NEMLINE, n, 1))
        allocate(mass_ssp(n, 1))
        allocate(ssp_lum(n, 1))

        ssp_grid = 1.0_wp
        emlin_grid = 0.0_wp
        mass_ssp = 1.0_wp
        ssp_lum = 1.0_wp

        call fsps_context_prepare_csp_workspace(ctx)
        call map_csp_workspace_to_device(ctx)

        pset%sfh = 1
        pset%tau = 2.0_wp
        pset%const = 1.0_wp
        pset%dust_tesc = 7.0_wp

        !$acc data copyin(ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        ctx%state%sfh_t_calc(1) = 10.0_wp
        !$acc update device(ctx%state%sfh_t_calc(1:1))
        call integrate_csp_step(ctx, pset, 1, 1, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        !$acc wait(1)
        !$acc end data

        !$acc update host(ctx%state%csp_weights, ctx%state%spec_young, ctx%state%spec_old)
        call unmap_csp_workspace_from_device(ctx)

        k = find_interval(time_full, 7.0_wp)
        expected_young = sum(ctx%state%csp_weights(1:k, 1, 1))
        expected_old = sum(ctx%state%csp_weights(k+1:n, 1, 1))

        call assert_relative_error(expected_young, real(ctx%state%spec_young(1, 1), WP), REL_EPS, &
                                   "Young sum matches weights", total_tests, total_failures)
        call assert_relative_error(expected_old, real(ctx%state%spec_old(1, 1), WP), REL_EPS, &
                                   "Old sum matches weights", total_tests, total_failures)

        deallocate(time_full, spec_lambda, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        call teardown_basic_context(ctx)
    end subroutine test_split_boundary

    subroutine test_mass_sum_consistency()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(KP), allocatable :: ssp_grid(:,:,:), emlin_grid(:,:,:)
        real(WP), allocatable :: mass_ssp(:,:), ssp_lum(:,:)
        integer :: n

        call print_group("CSP Integrator: Mass Sum Consistency")

        n = 60
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(1))
        spec_lambda = [5500.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(ssp_grid(1, n, 1))
        allocate(emlin_grid(NEMLINE, n, 1))
        allocate(mass_ssp(n, 1))
        allocate(ssp_lum(n, 1))

        ssp_grid = 1.0_wp
        emlin_grid = 0.0_wp
        mass_ssp = 1.0_wp
        ssp_lum = 1.0_wp

        call fsps_context_prepare_csp_workspace(ctx)
        call map_csp_workspace_to_device(ctx)

        pset%sfh = 1
        pset%tau = 1.0_wp
        pset%dust_tesc = 7.0_wp

        !$acc data copyin(ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        ctx%state%sfh_t_calc(1) = 10.0_wp
        !$acc update device(ctx%state%sfh_t_calc(1:1))
        call integrate_csp_step(ctx, pset, 1, 1, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        !$acc wait(1)
        !$acc end data

        !$acc update host(ctx%state%csp_weights, ctx%state%out_mass_csp(1:1))
        call unmap_csp_workspace_from_device(ctx)

        call assert_relative_error(sum(ctx%state%csp_weights(:,1,1)), ctx%state%out_mass_csp(1), REL_EPS, &
                                   "Mass matches sum of weights", total_tests, total_failures)
        call assert_relative_error(1.0_wp, ctx%state%out_mass_csp(1), 1.0e-3_wp, &
                                   "Mass ~1 for normalized weights", total_tests, total_failures)

        deallocate(time_full, spec_lambda, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        call teardown_basic_context(ctx)
    end subroutine test_mass_sum_consistency

    subroutine test_hotpath_equivalence()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(KP), allocatable :: ssp_grid(:,:,:), emlin_grid(:,:,:)
        real(WP), allocatable :: mass_ssp(:,:), ssp_lum(:,:)
        integer :: n, k, i, ispec, iem
        integer, parameter :: nz_mock = 3  ! Test multiple metallicities

        real(WP) :: mass_1, lbol_1
        real(WP), allocatable :: spec_young_1(:), spec_old_1(:), em_young_1(:), em_old_1(:)
        real(WP) :: max_err

        call print_group("CSP Integrator: Hotpath Equivalence (N=1 vs N>1)")

        n = 60
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(100))
        call fill_log_grid(spec_lambda, 3.0_wp, 4.0_wp)
        spec_lambda = 10.0_wp**spec_lambda

        ! Setup context with multiple metallicities
        call setup_basic_context(ctx, time_full, spec_lambda, nz_mock, 1)

        ! Allocate dummy inputs with nz_mock
        allocate(ssp_grid(size(spec_lambda), n, nz_mock))
        allocate(emlin_grid(NEMLINE, n, nz_mock))
        allocate(mass_ssp(n, nz_mock))
        allocate(ssp_lum(n, nz_mock))

        ! Fill the arrays with structured, non-uniform mock data
        do k = 1, nz_mock
            do i = 1, n
                mass_ssp(i, k) = real(i, WP) * 0.1_wp * real(k, WP)
                ssp_lum(i, k)  = real(i, WP) * 0.2_wp * real(k, WP)
                do ispec = 1, size(spec_lambda)
                    ssp_grid(ispec, i, k) = real(ispec, WP) + real(i, WP)*0.01_wp + real(k, WP)
                end do
                do iem = 1, NEMLINE
                    emlin_grid(iem, i, k) = real(iem, WP)*0.5_wp + real(i, WP)*0.01_wp + real(k, WP)
                end do
            end do
        end do

        call fsps_context_prepare_csp_workspace(ctx)
        call map_csp_workspace_to_device(ctx)

        pset%sfh = 1
        pset%tau = 2.0_wp
        pset%dust_tesc = 7.0_wp

        ! Mock weights so they aren't zero
        ctx%state%csp_weights = 0.5_wp
        !$acc update device(ctx%state%csp_weights)

        ! ==========================================================
        ! CALL 1: Run the `n_outputs == 1` hot path
        ! ==========================================================
        !$acc data copyin(ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        ctx%state%sfh_t_calc(1) = 10.0_wp
        !$acc update device(ctx%state%sfh_t_calc(1:1))

        call integrate_csp_step(ctx, pset, 1, 1, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        !$acc wait(1)

        ! Fetch results back to the host
        !$acc update host(ctx%state%out_mass_csp(1:1), ctx%state%out_lbol_csp(1:1))
        !$acc update host(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1))
        !$acc update host(ctx%state%csp_emlin_young(:, 1), ctx%state%csp_emlin_old(:, 1))

        mass_1 = ctx%state%out_mass_csp(1)
        lbol_1 = ctx%state%out_lbol_csp(1)

        allocate(spec_young_1(size(spec_lambda)))
        allocate(spec_old_1(size(spec_lambda)))
        allocate(em_young_1(NEMLINE))
        allocate(em_old_1(NEMLINE))

        spec_young_1 = ctx%state%spec_young(:, 1)
        spec_old_1   = ctx%state%spec_old(:, 1)
        em_young_1   = ctx%state%csp_emlin_young(:, 1)
        em_old_1     = ctx%state%csp_emlin_old(:, 1)

        ! ==========================================================
        ! CALL 2: Run the generic `n_outputs > 1` path
        ! ==========================================================
        ctx%state%out_mass_csp    = 0.0_wp
        ctx%state%out_lbol_csp    = 0.0_wp
        ctx%state%spec_young      = 0.0_wp
        ctx%state%spec_old        = 0.0_wp
        ctx%state%csp_emlin_young = 0.0_wp
        ctx%state%csp_emlin_old   = 0.0_wp
        !$acc update device(ctx%state%out_mass_csp, ctx%state%out_lbol_csp, ctx%state%spec_young)
        !$acc update device(ctx%state%spec_old, ctx%state%csp_emlin_young, ctx%state%csp_emlin_old)

        ! Give it a dummy second output to force `n_outputs = 2`
        ctx%state%sfh_t_calc(1) = 10.0_wp
        ctx%state%sfh_t_calc(2) = 5.0_wp
        !$acc update device(ctx%state%sfh_t_calc(1:2))

        ! Integrate with `n_outputs = 2`
        call integrate_csp_step(ctx, pset, 1, 2, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        !$acc wait(1)
        !$acc end data

        ! Fetch results back to the host
        !$acc update host(ctx%state%out_mass_csp(1:1), ctx%state%out_lbol_csp(1:1))
        !$acc update host(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1))
        !$acc update host(ctx%state%csp_emlin_young(:, 1), ctx%state%csp_emlin_old(:, 1))

        ! ==========================================================
        ! Strict Element-Wise Verifications
        ! ==========================================================
        call assert_float_equals(mass_1, ctx%state%out_mass_csp(1), REL_EPS, &
                                 "Scalars: Mass matches N>1 path", total_tests, total_failures)
        call assert_float_equals(lbol_1, ctx%state%out_lbol_csp(1), REL_EPS, &
                                 "Scalars: Lbol matches N>1 path", total_tests, total_failures)

        max_err = maxval(abs(spec_young_1 - ctx%state%spec_young(:, 1)))
        call assert_float_equals(0.0_wp, max_err, REL_EPS, &
                                 "Spectra: Young bin-by-bin match N>1", total_tests, total_failures)

        max_err = maxval(abs(spec_old_1 - ctx%state%spec_old(:, 1)))
        call assert_float_equals(0.0_wp, max_err, REL_EPS, &
                                 "Spectra: Old bin-by-bin match N>1", total_tests, total_failures)

        max_err = maxval(abs(em_young_1 - ctx%state%csp_emlin_young(:, 1)))
        call assert_float_equals(0.0_wp, max_err, REL_EPS, &
                                 "Emission: Young lines bin-by-bin match N>1", total_tests, total_failures)

        max_err = maxval(abs(em_old_1 - ctx%state%csp_emlin_old(:, 1)))
        call assert_float_equals(0.0_wp, max_err, REL_EPS, &
                                 "Emission: Old lines bin-by-bin match N>1", total_tests, total_failures)

        deallocate(spec_young_1, spec_old_1, em_young_1, em_old_1)
        deallocate(time_full, spec_lambda, ssp_grid, emlin_grid, mass_ssp, ssp_lum)
        call unmap_csp_workspace_from_device(ctx)
        call teardown_basic_context(ctx)
    end subroutine test_hotpath_equivalence

    ! ---------------------------------------------------------------------
    ! Group 3: Physics & Post-Processing
    ! ---------------------------------------------------------------------
    subroutine test_dust_screen_logic()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP), allocatable :: spec_total(:), emlin_total(:)
        real(WP) :: mdust

        call print_group("Dust: Birth Cloud vs Diffuse Screen")

        allocate(time_full(2))
        time_full = [6.0_wp, 7.0_wp]

        allocate(spec_lambda(3))
        spec_lambda = [5500.0_wp, 5500.0_wp, 5500.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        call fsps_context_prepare_csp_workspace(ctx)
        call map_csp_workspace_to_device(ctx)
        !$acc enter data create(pset)

        ctx%state%spec_young(:, 1) = 1.0_wp
        ctx%state%spec_old(:, 1) = 1.0_wp
        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp
        !$acc update device(ctx%state%spec_young(:, 1), ctx%state%spec_old(:, 1), &
        !$acc               ctx%state%csp_emlin_young(:, 1), ctx%state%csp_emlin_old(:, 1))

        allocate(spec_total(size(spec_lambda)))
        allocate(emlin_total(NEMLINE))

        pset%dust1 = 1.0_wp
        pset%dust2 = 0.0_wp
        pset%dust3 = 0.0_wp
        pset%dust1_index = -1.0_wp
        pset%frac_obrun = 0.0_wp
        pset%frac_nodust = 0.0_wp
        pset%uvb = 1.0_wp
        pset%wgp1 = 1
        pset%wgp2 = 1
        pset%wgp3 = 1

        !$acc update device(pset)
        call apply_dust_physics(ctx, pset, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%dust_spec_total_work(:, 1))

        call assert_true(ctx%state%dust_spec_total_work(1, 1) > 1.0_wp .and. ctx%state%dust_spec_total_work(1, 1) < 2.0_wp, &
                         "Young attenuated, old unattenuated", total_tests, total_failures)
        call assert_relative_error(exp(-1.0_wp), ctx%state%dust_spec_total_work(1, 1) - 1.0_wp, 1.0e-3_wp, &
                                   "Young component attenuated", total_tests, total_failures)

        deallocate(time_full, spec_lambda)
        call unmap_csp_workspace_from_device(ctx)
        !$acc exit data delete(pset)
        call teardown_basic_context(ctx)
    end subroutine test_dust_screen_logic

    subroutine test_igm_absorption_toggle()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout) :: result_no, result_yes
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP), allocatable :: spec_no(:), spec_yes(:), emlines(:)
        real(KP), allocatable :: igm(:)
        integer :: i_blue, i_red

        call print_group("IGM: Absorption Toggle")

        allocate(time_full(3))
        time_full = [6.0_wp, 8.0_wp, 10.0_wp]

        allocate(spec_lambda(4))
        spec_lambda = [900.0_wp, 1100.0_wp, 1300.0_wp, 1500.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        allocate(igm(size(spec_lambda)))
        allocate(result_no%spec(size(spec_lambda)))
        allocate(result_yes%spec(size(spec_lambda)))

        pset%sfh = 0
        pset%tage = 5.0_wp
        pset%zred = 3.0_wp
        pset%igm_factor = 1.0_wp

        call fsps_context_prepare_csp_workspace(ctx)
        call map_csp_workspace_to_device(ctx)
        !$acc enter data create(pset)

        ctx%state%dust_spec_total_work(:, 1) = 1.0_wp
        ctx%state%csp_emlin_young(:, 1) = 0.0_wp
        ctx%state%csp_emlin_old(:, 1) = 0.0_wp
        ctx%state%sfh_t_calc(1) = pset%tage
        !$acc update device(ctx%state%dust_spec_total_work(:, 1), ctx%state%csp_emlin_young(:, 1), &
        !$acc               ctx%state%csp_emlin_old(:, 1), ctx%state%sfh_t_calc(1:1))

        ctx%add_igm_absorption_val = 0
        !$acc update device(ctx%add_igm_absorption_val, pset)

        call apply_post_processing(ctx, pset, 1)
        !$acc wait(1)
        !$acc update host(ctx%state%out_csp_spec(:, 1))
        result_no%spec = ctx%state%out_csp_spec(:, 1)

        ctx%state%dust_spec_total_work(:, 1) = 1.0_wp
        ctx%add_igm_absorption_val = 1
        !$acc update device(ctx%state%dust_spec_total_work(:, 1), ctx%add_igm_absorption_val)

        !$acc data copyin(spec_lambda) copyout(igm)
        call compute_igm_transmission(spec_lambda, pset%zred, pset%igm_factor, igm)
        !$acc end data

        !$acc data copyin(igm)
        call apply_post_processing(ctx, pset, 1, igm)
        !$acc wait(1)
        !$acc end data
        !$acc update host(ctx%state%out_csp_spec(:, 1))
        result_yes%spec = ctx%state%out_csp_spec(:, 1)

        i_blue = 1
        i_red = 4

        call assert_true(result_yes%spec(i_blue) < result_no%spec(i_blue), &
                         "Blueward flux attenuated", total_tests, total_failures)
        call assert_relative_error(result_no%spec(i_red), result_yes%spec(i_red), 1.0e-6_wp, &
                                   "Redward flux unchanged", total_tests, total_failures)

        deallocate(time_full, spec_lambda, igm)
        !$acc exit data delete(pset)
        call unmap_csp_workspace_from_device(ctx)
        call teardown_basic_context(ctx)
    end subroutine test_igm_absorption_toggle

    subroutine test_smoothing_conserves_flux()
        type(fsps_context_t), allocatable :: ctx
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
    subroutine test_snapshot_vs_history()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        integer :: n, nspec

        call print_group("Driver: Snapshot vs History")

        n = 6
        nspec = 5
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(nspec))
        spec_lambda = [800.0_wp, 1200.0_wp, 3000.0_wp, 5500.0_wp, 9000.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        call fsps_context_prepare_csp_workspace(ctx)
        allocate(tspec_ssp(nspec, n, 1))
        allocate(mass_ssp(n, 1))
        allocate(lbol_ssp(n, 1))
        tspec_ssp = 1.0_wp
        mass_ssp = 1.0_wp
        lbol_ssp = 0.0_wp
        call fsps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 1)

        call map_csp_workspace_to_device(ctx)

        pset%sfh = 0
        pset%tage = 5.0_wp

        call compute_csp_scenario(ctx, pset, 1, results)
        call assert_int_equals(1, size(results), "Snapshot returns single output", total_tests, total_failures)
        call assert_float_equals(log10(5.0_wp * 1.0e9_wp), results(1)%age, 1.0e-5_wp, &
                                 "Snapshot age matches", total_tests, total_failures)

        deallocate(results)

        pset%tage = 0.0_wp
        call compute_csp_scenario(ctx, pset, 1, results)
        call assert_int_equals(n, size(results), "History returns ntfull outputs", total_tests, total_failures)
        call assert_float_equals(time_full(1), results(1)%age, 1.0e-5_wp, &
                                 "History age(1) matches grid", total_tests, total_failures)
        call assert_float_equals(time_full(n), results(n)%age, 1.0e-5_wp, &
                                 "History age(end) matches grid", total_tests, total_failures)

        deallocate(results)
        deallocate(time_full, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
        call unmap_csp_workspace_from_device(ctx)
        call teardown_basic_context(ctx)
    end subroutine test_snapshot_vs_history

    subroutine test_mass_normalization_surviving()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        integer :: n, nspec

        call print_group("Driver: Mass Normalization (Surviving)")

        n = 8
        nspec = 5
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.15_wp)

        allocate(spec_lambda(nspec))
        spec_lambda = [800.0_wp, 1200.0_wp, 3000.0_wp, 5500.0_wp, 9000.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        call fsps_context_prepare_csp_workspace(ctx)
        allocate(tspec_ssp(nspec, n, 1))
        allocate(mass_ssp(n, 1))
        allocate(lbol_ssp(n, 1))
        tspec_ssp = 1.0_wp
        mass_ssp = 1.0_wp
        lbol_ssp = 0.0_wp
        call fsps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 1)

        call map_csp_workspace_to_device(ctx)

        pset%sfh = 1
        pset%tau = 2.0_wp
        pset%tage = 13.0_wp

        call compute_csp_scenario(ctx, pset, 1, results)

        ! Every SSP in this basis retains all of its formed mass (mass_ssp = 1), and single-age
        ! weights are normalized to 1 Msun formed by tage, so the surviving mass is exactly
        ! sum(w * 1) = 1. (This previously asserted mass_csp < 1, which only held because
        ! mass_csp was wrongly rescaled by the formed-mass fraction mass_frac(tage) < 1.)
        call assert_relative_error(1.0_wp, results(1)%mass_csp, 1.0e-10_wp, &
                                   "Surviving mass = sum(w * mass_ssp) = 1", total_tests, total_failures)
        call assert_relative_error(1.0_wp, results(1)%mformed, 1.0e-6_wp, &
                                   "Formed mass normalized to 1", total_tests, total_failures)

        deallocate(results)
        deallocate(time_full, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
        call unmap_csp_workspace_from_device(ctx)
        call teardown_basic_context(ctx)
    end subroutine test_mass_normalization_surviving

    subroutine test_nebular_precalculation()
        type(fsps_context_t), allocatable :: ctx_no, ctx_yes
        type(params) :: pset
        type(compspout), allocatable :: res_no(:), res_yes(:)
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        integer :: n, nspec, line_idx

        call print_group("Driver: Nebular Pre-Calculation")

        n = 6
        nspec = 7
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(nspec))
        spec_lambda = [800.0_wp, 6000.0_wp, 6400.0_wp, 6563.0_wp, 6700.0_wp, 7000.0_wp, 9000.0_wp]

        call setup_basic_context(ctx_no, time_full, spec_lambda, 1, 1)
        call setup_basic_context(ctx_yes, time_full, spec_lambda, 1, 1)

        call setup_nebular_line_context(ctx_yes)

        allocate(tspec_ssp(nspec, n, 1))
        allocate(mass_ssp(n, 1))
        allocate(lbol_ssp(n, 1))
        tspec_ssp = 1.0e-13_wp   ! a physical SSP level (L_sun/Hz per M_sun), within FP32 range
        mass_ssp = 1.0_wp
        lbol_ssp = 0.0_wp

        call fsps_context_prepare_csp_workspace(ctx_no)
        call fsps_context_update_ssp_basis(ctx_no, tspec_ssp, mass_ssp, lbol_ssp, 1)
        call map_csp_workspace_to_device(ctx_no)
        call fsps_context_prepare_csp_workspace(ctx_yes)
        call fsps_context_update_ssp_basis(ctx_yes, tspec_ssp, mass_ssp, lbol_ssp, 1)
        call map_csp_workspace_to_device(ctx_yes)

        pset%sfh = 0
        pset%tage = 5.0_wp
        pset%gas_logz = -1.0_wp
        pset%gas_logu = -2.0_wp
        pset%sigma_smooth = 1.0_wp
        pset%frac_obrun = 0.0_wp
        ctx_yes%state%whlylim = 2

        call compute_csp_scenario(ctx_no, pset, 1, res_no)
        call compute_csp_scenario(ctx_yes, pset, 1, res_yes)

        line_idx = nearest_index(spec_lambda, 6563.0_wp)
        
        call assert_true(res_yes(1)%spec(line_idx) > res_no(1)%spec(line_idx), &
                         "Nebular lines increase flux", total_tests, total_failures)

        deallocate(res_no, res_yes)
        deallocate(time_full, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
        call unmap_csp_workspace_from_device(ctx_no)
        call unmap_csp_workspace_from_device(ctx_yes)
        call teardown_basic_context(ctx_no)
        call teardown_basic_context(ctx_yes)
    end subroutine test_nebular_precalculation

    subroutine test_fast_mode_skips_optional_outputs()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        real(WP), allocatable :: time_full(:)
        real(WP), allocatable :: spec_lambda(:)
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        integer :: n, nspec, center_idx

        call print_group("Driver: Fast Mode Output Gating")

        n = 6
        nspec = 9
        allocate(time_full(n))
        call fill_log_grid(time_full, 6.0_wp, 10.0_wp)

        allocate(spec_lambda(nspec))
        spec_lambda = [800.0_wp, 1200.0_wp, 2500.0_wp, 4000.0_wp, 5500.0_wp, 6500.0_wp, 7500.0_wp, 8500.0_wp, 9500.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 2)
        ctx%state%nindx = 2
        ctx%fast_mode = .true.

        call fsps_context_prepare_csp_workspace(ctx)
        allocate(tspec_ssp(nspec, n, 1))
        allocate(mass_ssp(n, 1))
        allocate(lbol_ssp(n, 1))
        tspec_ssp = SAFE_FLOOR
        center_idx = (nspec + 1)/2
        tspec_ssp(center_idx, :, 1) = 1.0_wp
        mass_ssp = 1.0_wp
        lbol_ssp = 0.0_wp

        call fsps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 1)
        call map_csp_workspace_to_device(ctx)

        pset%sfh = 0
        pset%tage = 5.0_wp
        pset%compute_mags = 1
        pset%compute_indices = 1
        pset%sigma_smooth = 300.0_wp
        pset%min_wave_smooth = minval(spec_lambda)
        pset%max_wave_smooth = maxval(spec_lambda)

        call compute_csp_scenario(ctx, pset, 1, results)

        call assert_int_equals(1, size(results), "Fast mode returns one snapshot", total_tests, total_failures)
        call assert_true(allocated(results(1)%spec), "Spectrum is generated", total_tests, total_failures)
        call assert_true(.not. allocated(results(1)%mags), "Magnitudes not allocated in fast mode", total_tests, total_failures)
        call assert_true(.not. allocated(results(1)%indx), "Indices not allocated in fast mode", total_tests, total_failures)
        call assert_true(results(1)%spec(center_idx) > 0.5_wp, &
                 "Core spectrum remains unsmoothed in fast mode", total_tests, total_failures)

        deallocate(results)
        deallocate(time_full, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
        call unmap_csp_workspace_from_device(ctx)
        call teardown_basic_context(ctx)
    end subroutine test_fast_mode_skips_optional_outputs

    ! ---------------------------------------------------------------------
    ! Group 5: Conditional Branching & Edge Cases
    ! ---------------------------------------------------------------------
    subroutine test_driver_status_codes()
        type(fsps_context_t), allocatable :: ctx
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
        call fsps_context_prepare_csp_workspace(ctx)

        ! Properly allocate and map the basis arrays to the device using the context API
        allocate(tspec_ssp(2, 2, 2), mass_ssp(2, 2), lbol_ssp(2, 2))
        tspec_ssp = 1.0_wp; mass_ssp = 1.0_wp; lbol_ssp = 0.0_wp
        call fsps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 2)
        call map_csp_workspace_to_device(ctx)

        ctx%add_neb_emission_val = 1

        ! Pass nzin=2 to trigger the error branch
        call compute_csp_scenario(ctx, pset, 2, results, status)
        call assert_int_equals(2, status, "Status 2: Nebular emission with nzin > 1", total_tests, total_failures)

        ! Cleanup
        deallocate(time_full, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
        if (allocated(results)) deallocate(results)
        call unmap_csp_workspace_from_device(ctx)
        call teardown_basic_context(ctx)
    end subroutine test_driver_status_codes

    subroutine test_driver_igm_precalc()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        real(WP), allocatable :: time_full(:), spec_lambda(:)
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)

        call print_group("Driver: IGM Pre-calculation Branch")

        allocate(time_full(2), spec_lambda(2))
        time_full = [6.0_wp, 10.0_wp]
        spec_lambda = [1000.0_wp, 1500.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        call fsps_context_prepare_csp_workspace(ctx)
        allocate(tspec_ssp(2, 2, 1), mass_ssp(2, 1), lbol_ssp(2, 1))
        tspec_ssp = 1.0_wp; mass_ssp = 1.0_wp; lbol_ssp = 0.0_wp
        call fsps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 1)
        call map_csp_workspace_to_device(ctx)

        pset%sfh = 0
        pset%tage = 5.0_wp

        ! Force the IGM precalc condition to evaluate as True
        pset%zred = 2.0_wp
        pset%igm_factor = 1.0_wp
        ctx%add_igm_absorption_val = 1

        ! Initialize to a dummy value (-1) to guarantee the conditional block executes and overwrites it
        ctx%state%csp_igm_transmission = -1.0_wp
        !$acc update device(ctx%state%csp_igm_transmission)

        call compute_csp_scenario(ctx, pset, 1, results)

        !$acc update host(ctx%state%csp_igm_transmission)
        call assert_true(all(ctx%state%csp_igm_transmission >= 0.0_wp), &
                         "IGM transmission array populated", total_tests, total_failures)

        deallocate(time_full, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
        if (allocated(results)) deallocate(results)
        call unmap_csp_workspace_from_device(ctx)
        call teardown_basic_context(ctx)
    end subroutine test_driver_igm_precalc

    subroutine test_driver_tabular_tage_99()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        real(WP), allocatable :: time_full(:), spec_lambda(:)
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        real(WP) :: max_tab_age

        call print_group("Driver: Tabular SFH with tage=-99")

        allocate(time_full(2), spec_lambda(2))
        time_full = [6.0_wp, 10.0_wp]
        spec_lambda = [5500.0_wp, 5600.0_wp]
        call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)

        call fsps_context_prepare_csp_workspace(ctx)
        allocate(tspec_ssp(2, 2, 1), mass_ssp(2, 1), lbol_ssp(2, 1))
        tspec_ssp = 1.0_wp; mass_ssp = 1.0_wp; lbol_ssp = 0.0_wp
        call fsps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 1)
        call map_csp_workspace_to_device(ctx)

        ! Trigger the tabular/array SFH pathway
        pset%sfh = 3
        ! Set the special "Evaluate at end of table" flag
        pset%tage = -99.0_wp

        ! Mock up a dummy tabular SFH
        ctx%state%ntabsfh = 2
        ctx%state%sfh_tab(1, 1:2) = [0.0_wp, 2.5e9_wp]
        ctx%state%sfh_tab(2, 1:2) = [1.0_wp, 1.0_wp]
        ctx%state%sfh_tab(3, 1:2) = [0.02_wp, 0.02_wp]

        max_tab_age = 2.5_wp ! (Gyr)

        call compute_csp_scenario(ctx, pset, 1, results)

        call assert_int_equals(1, size(results), "Only 1 output generated for tage=-99", total_tests, total_failures)
        call assert_float_equals(log10(max_tab_age * 1.0e9_wp), results(1)%age, 1.0e-5_wp, &
                                 "Output age evaluates exactly at max tabular age", total_tests, total_failures)

        deallocate(time_full, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
        if (allocated(results)) deallocate(results)
        call unmap_csp_workspace_from_device(ctx)
        call teardown_basic_context(ctx)
    end subroutine test_driver_tabular_tage_99

    !> Tabular SFHs are absolute (SFR in M_sun/yr, not normalized): the mass formed must be the
    !> integral of the SFR table up to tage over the lookback times the weights cover (>=
    !> 10**tiny_logt yr, as for every SFH), i.e. the sum of the SSP weights, consistent with the
    !> surviving mass (mass_csp = sum(w * mass_ssp)). It was reported as 1. Checked at tage > 0
    !> and with tage = -99 (evaluate at the end of the table), for sfh = 3.
    subroutine test_tabular_mass_formed()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        real(WP), allocatable :: time_full(:), spec_lambda(:), weights(:)
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        integer :: n, i, k
        real(WP), parameter :: tages(2) = [5.0_wp, -99.0_wp]
        real(WP), parameter :: expected(2) = [5.0e9_wp, 6.0e9_wp]  ! 1 M_sun/yr from t = 0
        character(len=96) :: label

        call print_group("Driver: Tabular SFH Mass Formed")

        n = 40
        allocate(time_full(n), weights(n), spec_lambda(2))
        call fill_log_grid(time_full, 6.0_wp, 10.15_wp)
        spec_lambda = [5500.0_wp, 5600.0_wp]
        allocate(tspec_ssp(2, n, 1), mass_ssp(n, 1), lbol_ssp(n, 1))
        tspec_ssp = 1.0_wp
        lbol_ssp = 0.0_wp
        do i = 1, n
            mass_ssp(i, 1) = 1.0_wp - 0.4_wp * real(i - 1, WP) / real(n - 1, WP)
        end do

        do k = 1, size(tages)
            call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)
            call fsps_context_prepare_csp_workspace(ctx)
            call fsps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 1)
            call map_csp_workspace_to_device(ctx)

            pset%sfh = 3
            pset%tage = tages(k)
            ctx%state%ntabsfh = 3
            ctx%state%sfh_tab(1, 1:3) = [0.0_wp, 3.0e9_wp, 6.0e9_wp]
            ctx%state%sfh_tab(2, 1:3) = [1.0_wp, 1.0_wp, 1.0_wp]
            ctx%state%sfh_tab(3, 1:3) = 0.0_wp

            call compute_csp_scenario(ctx, pset, 1, results)
            !$acc update host(ctx%state%csp_weights)
            weights = ctx%state%csp_weights(1:n, 1, 1)

            write(label, '(A,F6.1,A)') "mass formed = integral of the SFR table (tage = ", tages(k), ")"
            call assert_relative_error(expected(k) - 10.0_wp**ctx%tiny_logt_val, results(1)%mformed, 1.0e-10_wp, &
                                       trim(label), &
                                       total_tests, total_failures)
            write(label, '(A,F6.1,A)') "mass formed = sum of the SSP weights (tage = ", tages(k), ")"
            call assert_relative_error(sum(weights), results(1)%mformed, 1.0e-12_wp, trim(label), &
                                       total_tests, total_failures)
            write(label, '(A,F6.1,A)') "surviving mass = sum(w * mass_ssp) (tage = ", tages(k), ")"
            call assert_relative_error(sum(weights * mass_ssp(:, 1)), results(1)%mass_csp, 1.0e-10_wp, &
                                       trim(label), total_tests, total_failures)

            if (allocated(results)) deallocate(results)
            call unmap_csp_workspace_from_device(ctx)
            call teardown_basic_context(ctx)
        end do
        deallocate(time_full, weights, spec_lambda, tspec_ssp, mass_ssp, lbol_ssp)
    end subroutine test_tabular_mass_formed

    ! ---------------------------------------------------------------------
    ! Oracle: single-age CSP outputs are SFH-weighted sums of the SSP basis
    ! ---------------------------------------------------------------------

    !> With dust, nebular emission, IGM and smoothing disabled, a single-age CSP (tage > 0) must
    !> equal the SFH-weighted sum of its SSP basis for every user-facing output:
    !>     spec = sum_t w_t spec_ssp(t),  mass = sum_t w_t mass_ssp(t),  L = sum_t w_t L_ssp(t)
    !> with sum_t w_t = 1 (normalized to 1 Msun formed by tage). This checks results(:) rather than
    !> internal buffers, so any rescaling applied while packing results (e.g. by mass_frac, which
    !> only applies to the all-ages tage <= 0 mode) is caught. The default log-time interpolation
    !> is used so that the production SFH weight path is exercised.
    subroutine test_single_age_outputs_are_weighted_sums()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        real(WP), allocatable :: time_full(:), spec_lambda(:), weights(:)
        real(WP), allocatable :: tspec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        real(WP) :: expected
        integer :: n, nspec, i, k, l
        integer, parameter :: sfh_types(2) = [1, 4]
        real(WP), parameter :: taus(2) = [2.0_wp, 3.0_wp]
        character(len=96) :: label

        call print_group("Driver: Single-Age Outputs = Weighted SSP Sums (oracle)")

        n = 40
        nspec = 5
        allocate(time_full(n), weights(n))
        call fill_log_grid(time_full, 6.0_wp, 10.15_wp)
        allocate(spec_lambda(nspec))
        spec_lambda = [800.0_wp, 1200.0_wp, 3000.0_wp, 5500.0_wp, 9000.0_wp]

        ! Synthetic SSP basis whose spectrum, surviving mass and luminosity all vary with age
        allocate(tspec_ssp(nspec, n, 1), mass_ssp(n, 1), lbol_ssp(n, 1))
        do i = 1, n
            do l = 1, nspec
                tspec_ssp(l, i, 1) = (1.0_wp + 0.1_wp * real(l, WP)) * 10.0_wp**(-0.3_wp * (time_full(i) - 6.0_wp))
            end do
            mass_ssp(i, 1) = 1.0_wp - 0.4_wp * real(i - 1, WP) / real(n - 1, WP)
            lbol_ssp(i, 1) = -0.5_wp * (time_full(i) - 6.0_wp)
        end do

        do k = 1, size(sfh_types)
            call setup_basic_context(ctx, time_full, spec_lambda, 1, 1)
            ctx%interpolation_type_val = 0
            call fsps_context_prepare_csp_workspace(ctx)
            call fsps_context_update_ssp_basis(ctx, tspec_ssp, mass_ssp, lbol_ssp, 1)
            call map_csp_workspace_to_device(ctx)

            pset%sfh = sfh_types(k)
            pset%tau = taus(k)
            pset%tage = 5.0_wp

            call compute_csp_scenario(ctx, pset, 1, results)
            !$acc update host(ctx%state%csp_weights)
            weights = ctx%state%csp_weights(1:n, 1, 1)

            write(label, '(A,I0,A)') "SFH weights sum to 1 (sfh=", sfh_types(k), ")"
            call assert_relative_error(1.0_wp, sum(weights), 1.0e-10_wp, trim(label), total_tests, total_failures)

            write(label, '(A,I0,A)') "SFH weights non-negative (sfh=", sfh_types(k), ")"
            call assert_true(all(weights >= 0.0_wp), trim(label), total_tests, total_failures)

            write(label, '(A,I0,A)') "mass_csp = sum(w * mass_ssp) (sfh=", sfh_types(k), ")"
            call assert_relative_error(sum(weights * mass_ssp(:, 1)), results(1)%mass_csp, 1.0e-10_wp, &
                                       trim(label), total_tests, total_failures)

            write(label, '(A,I0,A)') "10**lbol_csp = sum(w * 10**lbol_ssp) (sfh=", sfh_types(k), ")"
            call assert_relative_error(sum(weights * 10.0_wp**lbol_ssp(:, 1)), 10.0_wp**results(1)%lbol_csp, &
                                       1.0e-10_wp, trim(label), total_tests, total_failures)

            do l = 1, nspec
                expected = sum(weights * tspec_ssp(l, :, 1))
                write(label, '(A,I0,A,I0,A)') "spec = sum(w * spec_ssp) (sfh=", sfh_types(k), ", lambda #", l, ")"
                call assert_relative_error(expected, results(1)%spec(l), max(1.0e-10_wp, KP_RTOL), &
                                           trim(label), total_tests, total_failures)
            end do

            deallocate(results)
            call unmap_csp_workspace_from_device(ctx)
            call teardown_basic_context(ctx)
        end do

        deallocate(time_full, spec_lambda, weights, tspec_ssp, mass_ssp, lbol_ssp)
    end subroutine test_single_age_outputs_are_weighted_sums

end module test_fsps_csp_mod
