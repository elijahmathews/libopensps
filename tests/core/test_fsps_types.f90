module test_fsps_types_mod
    use fsps_precision, only: WP
    use fsps_constants, only: NPZPHOT
    use fsps_types
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_int_equals, assert_true
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_types_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for fsps_types module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_fsps_types_tests()

        call print_minor_header("fsps_types")

        call test_params_initialization()
        call test_compspout_initialization()
        call test_sfhparams_initialization()
        call test_tlsf_initialization()
        call test_obsdat_initialization()
        call test_tpzphot_initialization()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_types_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: PARAMS INITIALIZATION
    ! ------------------------------------------------------------------------
    subroutine test_params_initialization()
        type(PARAMS) :: p
        
        call print_group("PARAMS Default Initialization")

        ! Test integer defaults
        call assert_int_equals(1, p%zmet, "zmet defaults to 1", total_tests, total_failures)
        call assert_int_equals(0, p%sfh, "sfh defaults to 0", total_tests, total_failures)
        call assert_int_equals(1, p%wgp1, "wgp1 defaults to 1", total_tests, total_failures)
        call assert_int_equals(-1, p%evtype, "evtype defaults to -1", total_tests, total_failures)
        call assert_int_equals(1, p%compute_mags, "compute_mags defaults to 1", total_tests, total_failures)
        call assert_int_equals(1, p%compute_indices, "compute_indices defaults to 1", total_tests, total_failures)

        ! Test critical float defaults (IMF, Dust, Gas, SFH)
        call assert_float_equals(1.0_wp, p%pagb, 1.0e-6_wp, "pagb defaults to 1.0", total_tests, total_failures)
        call assert_float_equals(11.0_wp, p%tburst, 1.0e-6_wp, "tburst defaults to 11.0", total_tests, total_failures)
        call assert_float_equals(1.3_wp, p%imf1, 1.0e-6_wp, "imf1 defaults to 1.3", total_tests, total_failures)
        call assert_float_equals(2.3_wp, p%imf2, 1.0e-6_wp, "imf2 defaults to 2.3", total_tests, total_failures)
        call assert_float_equals(2.3_wp, p%imf3, 1.0e-6_wp, "imf3 defaults to 2.3", total_tests, total_failures)
        call assert_float_equals(0.08_wp, p%vdmc, 1.0e-6_wp, "vdmc defaults to 0.08", total_tests, total_failures)
        call assert_float_equals(-99.0_wp, p%dust_clumps, 1.0e-6_wp, "dust_clumps defaults to -99.0", total_tests, total_failures)
        call assert_float_equals(7.0_wp, p%dust_tesc, 1.0e-6_wp, "dust_tesc defaults to 7.0", total_tests, total_failures)
        call assert_float_equals(3.1_wp, p%mwr, 1.0e-6_wp, "mwr defaults to 3.1", total_tests, total_failures)
        call assert_float_equals(0.5_wp, p%mdave, 1.0e-6_wp, "mdave defaults to 0.5", total_tests, total_failures)
        call assert_float_equals(-2.0_wp, p%gas_logu, 1.0e-6_wp, "gas_logu defaults to -2.0", total_tests, total_failures)
        call assert_float_equals(1.0_wp, p%frac_xrb, 1.0e-6_wp, "frac_xrb defaults to 1.0", total_tests, total_failures)

        ! Test strings
        call assert_true(trim(p%imf_filename) == '', "imf_filename is empty", total_tests, total_failures)
        call assert_true(trim(p%sfh_filename) == '', "sfh_filename is empty", total_tests, total_failures)

        ! Test allocatable arrays
        call assert_true(.not. allocated(p%mag_compute), "mag_compute is unallocated", total_tests, total_failures)
        call assert_true(.not. allocated(p%ssp_gen_age), "ssp_gen_age is unallocated", total_tests, total_failures)

    end subroutine test_params_initialization

    ! ------------------------------------------------------------------------
    ! TEST SUITE: COMPSPOUT INITIALIZATION
    ! ------------------------------------------------------------------------
    subroutine test_compspout_initialization()
        type(COMPSPOUT) :: out
        
        call print_group("COMPSPOUT Default Initialization")

        ! Test scalar defaults
        call assert_float_equals(0.0_wp, out%age, 1.0e-6_wp, "age defaults to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, out%mass_csp, 1.0e-6_wp, "mass_csp defaults to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, out%lbol_csp, 1.0e-6_wp, "lbol_csp defaults to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, out%sfr, 1.0e-6_wp, "sfr defaults to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, out%mdust, 1.0e-6_wp, "mdust defaults to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, out%mformed, 1.0e-6_wp, "mformed defaults to 0.0", total_tests, total_failures)

        ! Test allocatable arrays
        call assert_true(.not. allocated(out%mags), "mags array is unallocated", total_tests, total_failures)
        call assert_true(.not. allocated(out%spec), "spec array is unallocated", total_tests, total_failures)
        call assert_true(.not. allocated(out%indx), "indx array is unallocated", total_tests, total_failures)
        call assert_true(.not. allocated(out%emlines), "emlines array is unallocated", total_tests, total_failures)

    end subroutine test_compspout_initialization

    ! ------------------------------------------------------------------------
    ! TEST SUITE: SFHPARAMS INITIALIZATION
    ! ------------------------------------------------------------------------
    subroutine test_sfhparams_initialization()
        type(SFHPARAMS) :: sfh
        
        call print_group("SFHPARAMS Default Initialization")

        ! Test float defaults
        call assert_float_equals(1.0_wp, sfh%tau, 1.0e-6_wp, "tau defaults to 1.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, sfh%tage, 1.0e-6_wp, "tage defaults to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, sfh%tburst, 1.0e-6_wp, "tburst defaults to 0.0", total_tests, total_failures)

        ! Test integer defaults
        call assert_int_equals(0, sfh%type, "type defaults to 0", total_tests, total_failures)
        call assert_int_equals(0, sfh%use_simha_limits, "use_simha_limits defaults to 0", total_tests, total_failures)

    end subroutine test_sfhparams_initialization

    ! ------------------------------------------------------------------------
    ! TEST SUITE: MINOR TYPES INITIALIZATION
    ! ------------------------------------------------------------------------
    subroutine test_tlsf_initialization()
        type(TLSF) :: lsf_obj
        
        call print_group("TLSF Default Initialization")

        call assert_float_equals(0.0_wp, lsf_obj%minlam, 1.0e-6_wp, "minlam defaults to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, lsf_obj%maxlam, 1.0e-6_wp, "maxlam defaults to 0.0", total_tests, total_failures)
        call assert_true(.not. allocated(lsf_obj%lsf), "lsf array is unallocated", total_tests, total_failures)

    end subroutine test_tlsf_initialization

    subroutine test_obsdat_initialization()
        type(OBSDAT) :: obs
        
        call print_group("OBSDAT Default Initialization")

        call assert_float_equals(0.0_wp, obs%zred, 1.0e-6_wp, "zred defaults to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, obs%logsmass, 1.0e-6_wp, "logsmass defaults to 0.0", total_tests, total_failures)
        
        call assert_true(.not. allocated(obs%mags), "mags array is unallocated", total_tests, total_failures)
        call assert_true(.not. allocated(obs%magerr), "magerr array is unallocated", total_tests, total_failures)
        call assert_true(.not. allocated(obs%spec), "spec array is unallocated", total_tests, total_failures)
        call assert_true(.not. allocated(obs%specerr), "specerr array is unallocated", total_tests, total_failures)

    end subroutine test_obsdat_initialization

    subroutine test_tpzphot_initialization()
        type(TPZPHOT) :: tpz
        
        call print_group("TPZPHOT Default Initialization")

        call assert_int_equals(NPZPHOT, size(tpz%zz), "zz array is sized to NPZPHOT", total_tests, total_failures)
        call assert_int_equals(NPZPHOT, size(tpz%pz), "pz array is sized to NPZPHOT", total_tests, total_failures)
        
        call assert_float_equals(0.0_wp, sum(tpz%zz), 1.0e-6_wp, "zz array initialized to 0.0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, sum(tpz%pz), 1.0e-6_wp, "pz array initialized to 0.0", total_tests, total_failures)

    end subroutine test_tpzphot_initialization

end module test_fsps_types_mod
