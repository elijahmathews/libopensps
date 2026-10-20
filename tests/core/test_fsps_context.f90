module test_fsps_context_mod
    use fsps_precision, only: WP
    use fsps_types, only: PARAMS
    use fsps_context_types, only: fsps_context_t
    use fsps_context
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_int_equals, assert_true
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_context_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for fsps_context module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_fsps_context_tests()

        call print_minor_header("fsps_context")

        call test_context_create_destroy()
        call test_context_set_params_int()
        call test_context_set_params_float()
        call test_context_set_params_str()
        call test_context_pset_management()
        call test_context_workspace_allocation()
        call test_context_ssp_basis_update()
        call test_ssp_dirty_flag_covers_agb_dust_and_obrun()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_context_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: CREATE & DESTROY
    ! ------------------------------------------------------------------------
    subroutine test_context_create_destroy()
        type(fsps_context_t) :: ctx
        
        call print_group("Context Create & Destroy")

        call fsps_context_create(ctx)

        ! Check basic initializations
        call assert_true(.not. ctx%initialized, "Context initially not initialized", total_tests, total_failures)
        call assert_true(.not. ctx%fast_mode, "Fast mode initially false", total_tests, total_failures)
        
        ! Check some default physical constants
        call assert_float_equals(0.27_wp, ctx%om0_val, 1.0e-6_wp, "Default om0_val = 0.27", total_tests, total_failures)
        call assert_float_equals(72.0_wp, ctx%H0_val, 1.0e-6_wp, "Default H0_val = 72.0", total_tests, total_failures)
        call assert_float_equals(120.0_wp, ctx%imf_upper_limit_val, 1.0e-6_wp, "Default imf_upper_limit_val = 120.0", &
                                 total_tests, total_failures)

        ! Destroy context (should run without error)
        call fsps_context_destroy(ctx)
        call assert_true(.not. ctx%initialized, "Context de-initialized", total_tests, total_failures)

    end subroutine test_context_create_destroy

    ! ------------------------------------------------------------------------
    ! TEST SUITE: INTEGER PARAMETERS
    ! ------------------------------------------------------------------------
    subroutine test_context_set_params_int()
        type(fsps_context_t) :: ctx
        integer :: status
        
        call print_group("Integer Parameters (Set/Get)")
        call fsps_context_create(ctx)

        ! Valid non-sensitive parameter
        call fsps_context_set_param_int(ctx, 'sfh', 1, status)
        call assert_int_equals(0, status, "Set 'sfh' succeeds", total_tests, total_failures)
        call assert_int_equals(1, ctx%pset%sfh, "Value of 'sfh' updated", total_tests, total_failures)

        ! Valid SSP-sensitive parameter
        ctx%state%ssp_basis_is_dirty = .false.
        call fsps_context_set_param_int(ctx, 'imf_type', 1, status)
        call assert_int_equals(0, status, "Set 'imf_type' succeeds", total_tests, total_failures)
        call assert_int_equals(1, ctx%imf_type_val, "Value of 'imf_type' updated", total_tests, total_failures)
        call assert_true(ctx%state%ssp_basis_is_dirty, "SSP-sensitive int param flags dirty state", total_tests, total_failures)

        ! Fast mode toggle
        call fsps_context_set_param_int(ctx, 'fast_mode', 1, status)
        call assert_int_equals(0, status, "Set 'fast_mode' succeeds", total_tests, total_failures)
        call assert_true(ctx%fast_mode, "Fast mode activated", total_tests, total_failures)

        ! Unknown parameter
        call fsps_context_set_param_int(ctx, 'invalid_int_param', 42, status)
        call assert_int_equals(101, status, "Unknown int param returns FSPS_ERR_UNKNOWN_INT_PARAM (101)", &
                               total_tests, total_failures)

        call fsps_context_destroy(ctx)
    end subroutine test_context_set_params_int

    ! ------------------------------------------------------------------------
    ! TEST SUITE: FLOAT PARAMETERS
    ! ------------------------------------------------------------------------
    subroutine test_context_set_params_float()
        type(fsps_context_t) :: ctx
        integer :: status
        
        call print_group("Float Parameters (Set/Get)")
        call fsps_context_create(ctx)

        ! Valid non-sensitive parameter
        call fsps_context_set_param_float(ctx, 'om0', 0.3_wp, status)
        call assert_int_equals(0, status, "Set 'om0' succeeds", total_tests, total_failures)
        call assert_float_equals(0.3_wp, ctx%om0_val, 1.0e-6_wp, "Value of 'om0' updated", total_tests, total_failures)

        ! Valid SSP-sensitive parameter
        ctx%state%ssp_basis_is_dirty = .false.
        call fsps_context_set_param_float(ctx, 'imf_upper_limit', 100.0_wp, status)
        call assert_int_equals(0, status, "Set 'imf_upper_limit' succeeds", total_tests, total_failures)
        call assert_float_equals(100.0_wp, ctx%imf_upper_limit_val, 1.0e-6_wp, "Value of 'imf_upper_limit' updated", &
                                 total_tests, total_failures)
        call assert_true(ctx%state%ssp_basis_is_dirty, "SSP-sensitive float param flags dirty state", total_tests, total_failures)

        ! Unknown parameter
        call fsps_context_set_param_float(ctx, 'invalid_float_param', 3.14_wp, status)
        call assert_int_equals(102, status, "Unknown float param returns FSPS_ERR_UNKNOWN_FLOAT_PARAM (102)", &
                               total_tests, total_failures)

        call fsps_context_destroy(ctx)
    end subroutine test_context_set_params_float

    ! ------------------------------------------------------------------------
    ! TEST SUITE: STRING PARAMETERS
    ! ------------------------------------------------------------------------
    subroutine test_context_set_params_str()
        type(fsps_context_t) :: ctx
        integer :: status
        
        call print_group("String Parameters (Set/Get)")
        call fsps_context_create(ctx)

        ! Valid non-sensitive parameter
        call fsps_context_set_param_str(ctx, 'sfh_filename', 'test_sfh.dat', status)
        call assert_int_equals(0, status, "Set 'sfh_filename' succeeds", total_tests, total_failures)
        call assert_true(trim(ctx%pset%sfh_filename) == 'test_sfh.dat', "Value of 'sfh_filename' updated", &
                         total_tests, total_failures)

        ! Valid SSP-sensitive parameter
        ctx%state%ssp_basis_is_dirty = .false.
        call fsps_context_set_param_str(ctx, 'imf_filename', 'custom_imf.dat', status)
        call assert_int_equals(0, status, "Set 'imf_filename' succeeds", total_tests, total_failures)
        call assert_true(trim(ctx%pset%imf_filename) == 'custom_imf.dat', "Value of 'imf_filename' updated", &
                         total_tests, total_failures)
        call assert_true(ctx%state%ssp_basis_is_dirty, "SSP-sensitive str param flags dirty state", &
                         total_tests, total_failures)

        ! Unknown parameter
        call fsps_context_set_param_str(ctx, 'invalid_str_param', 'hello', status)
        call assert_int_equals(103, status, "Unknown str param returns FSPS_ERR_UNKNOWN_STRING_PARAM (103)", &
                               total_tests, total_failures)

        call fsps_context_destroy(ctx)
    end subroutine test_context_set_params_str

    ! ------------------------------------------------------------------------
    ! TEST SUITE: PSET MANAGEMENT
    ! ------------------------------------------------------------------------
    subroutine test_context_pset_management()
        type(fsps_context_t) :: ctx
        type(PARAMS) :: pset_in, pset_out
        
        call print_group("Parameter Set (PSET) Overrides")
        call fsps_context_create(ctx)

        ! Retrieve current pset
        call fsps_context_get_pset(ctx, pset_out)
        
        ! Modify pset with a non-SSP sensitive parameter
        pset_in = pset_out
        pset_in%sfh = 5
        
        ctx%state%ssp_basis_is_dirty = .false.
        call fsps_context_set_pset(ctx, pset_in)
        call assert_int_equals(5, ctx%pset%sfh, "PSET properly applied", total_tests, total_failures)
        call assert_true(.not. ctx%state%ssp_basis_is_dirty, "Non-sensitive PSET change doesn't dirty basis", &
                         total_tests, total_failures)

        ! Modify pset with an SSP sensitive parameter
        pset_in%zmet = 3
        call fsps_context_set_pset(ctx, pset_in)
        call assert_int_equals(3, ctx%pset%zmet, "PSET zmet properly applied", total_tests, total_failures)
        call assert_true(ctx%state%ssp_basis_is_dirty, "Sensitive PSET change dirties basis", total_tests, total_failures)

        call fsps_context_destroy(ctx)
    end subroutine test_context_pset_management

    ! ------------------------------------------------------------------------
    ! TEST SUITE: WORKSPACE ALLOCATIONS
    ! ------------------------------------------------------------------------
    subroutine test_context_workspace_allocation()
        type(fsps_context_t) :: ctx
        
        call print_group("CSP/PSET Workspace Allocation")
        call fsps_context_create(ctx)

        ! Set up mock sizing parameters inside context state
        ctx%state%nbands = 5
        ctx%state%nt = 10
        ctx%state%nspec = 20
        ctx%state%ntfull = 10
        ctx%state%nz = 3

        ! Prepare PSET (mag_compute, ssp_gen_age)
        call fsps_context_prepare_pset(ctx)
        call assert_true(allocated(ctx%pset%mag_compute), "mag_compute allocated", total_tests, total_failures)
        call assert_true(allocated(ctx%pset%ssp_gen_age), "ssp_gen_age allocated", total_tests, total_failures)
        call assert_int_equals(5, size(ctx%pset%mag_compute), "mag_compute sized to nbands", total_tests, total_failures)

        ! Prepare CSP Workspace
        call fsps_context_prepare_csp_workspace(ctx)
        call assert_true(allocated(ctx%state%csp_emlin_old), "csp_emlin_old allocated", total_tests, total_failures)
        call assert_true(allocated(ctx%state%csp_spec_final), "csp_spec_final allocated", total_tests, total_failures)
        call assert_true(allocated(ctx%state%csp_ssp_grid), "csp_ssp_grid allocated for nspec>0", total_tests, total_failures)

        call fsps_context_destroy(ctx)
    end subroutine test_context_workspace_allocation

    ! ------------------------------------------------------------------------
    ! TEST SUITE: SSP BASIS UPDATE
    ! ------------------------------------------------------------------------
    subroutine test_context_ssp_basis_update()
        type(fsps_context_t) :: ctx
        real(WP), allocatable :: spec_ssp(:,:,:), mass_ssp(:,:), lbol_ssp(:,:)
        integer :: nspec, nt, nzin
        
        call print_group("SSP Basis Updates")
        call fsps_context_create(ctx)

        nspec = 10
        nt = 5
        nzin = 2

        allocate(spec_ssp(nspec, nt, nzin))
        allocate(mass_ssp(nt, nzin))
        allocate(lbol_ssp(nt, nzin))

        spec_ssp = 1.0_wp
        mass_ssp = 2.0_wp
        lbol_ssp = 3.0_wp

        ctx%state%ssp_basis_is_dirty = .true.

        ! First call updates basis and allocates arrays internally
        call fsps_context_update_ssp_basis(ctx, spec_ssp, mass_ssp, lbol_ssp, nzin)
        
        call assert_true(allocated(ctx%state%ssp_basis_spec), "ssp_basis_spec allocated", total_tests, total_failures)
        call assert_true(.not. ctx%state%ssp_basis_is_dirty, "Dirty flag cleared after update", total_tests, total_failures)
        call assert_float_equals(2.0_wp, ctx%state%ssp_basis_mass(1,1), 1.0e-6_wp, "Basis mass data updated correctly", &
                                 total_tests, total_failures)

        ! Simulate state changing, mark dirty again to trigger re-update without re-alloc
        ctx%state%ssp_basis_is_dirty = .true.
        mass_ssp = 4.0_wp
        call fsps_context_update_ssp_basis(ctx, spec_ssp, mass_ssp, lbol_ssp, nzin)
        
        call assert_float_equals(4.0_wp, ctx%state%ssp_basis_mass(1,1), 1.0e-6_wp, "Basis mass data updated on dirty flag", &
                                 total_tests, total_failures)
        call assert_true(.not. ctx%state%ssp_basis_is_dirty, "Dirty flag cleared again", total_tests, total_failures)

        deallocate(spec_ssp, mass_ssp, lbol_ssp)
        ctx%initialized = .true. ! Prevents NVHPC OpenACC exit data crash on mock context
        call fsps_context_destroy(ctx)
    end subroutine test_context_ssp_basis_update

    !> Every parameter read during SSP generation must mark the cached SSP basis dirty, both through
    !> the string-keyed setters and through fsps_context_set_pset (pset_changes_ssp_basis):
    !> agb_dust and add_agb_dust_model enter get_stellar_spectrum (circumstellar AGB dust), and
    !> frac_obrun scales the ionizing flux when nebular emission is built into the SSPs. A CSP-only
    !> parameter (dust2) must not mark it dirty.
    subroutine test_ssp_dirty_flag_covers_agb_dust_and_obrun()
        type(fsps_context_t), allocatable :: ctx
        type(PARAMS) :: pset
        integer :: status, k
        character(len=16), parameter :: float_keys(2) = [character(len=16) :: 'agb_dust', 'frac_obrun']

        call print_group("SSP Dirty Flag: AGB Dust and frac_obrun")
        allocate(ctx)
        call fsps_context_create(ctx)

        do k = 1, size(float_keys)
            ctx%state%ssp_basis_is_dirty = .false.
            call fsps_context_set_param_float(ctx, trim(float_keys(k)), 0.5_wp, status)
            call assert_true(ctx%state%ssp_basis_is_dirty, "set_param_float('"//trim(float_keys(k))// &
                             "') flags the SSP basis dirty", total_tests, total_failures)
        end do

        ctx%state%ssp_basis_is_dirty = .false.
        call fsps_context_set_param_int(ctx, 'add_agb_dust_model', 0, status)
        call assert_true(ctx%state%ssp_basis_is_dirty, "set_param_int('add_agb_dust_model') flags the SSP basis dirty", &
                         total_tests, total_failures)

        ! Whole-pset updates
        call fsps_context_get_pset(ctx, pset)
        ctx%state%ssp_basis_is_dirty = .false.
        pset%agb_dust = pset%agb_dust + 1.0_wp
        call fsps_context_set_pset(ctx, pset)
        call assert_true(ctx%state%ssp_basis_is_dirty, "set_pset with new agb_dust flags dirty", total_tests, total_failures)

        ctx%state%ssp_basis_is_dirty = .false.
        pset%frac_obrun = pset%frac_obrun + 0.1_wp
        call fsps_context_set_pset(ctx, pset)
        call assert_true(ctx%state%ssp_basis_is_dirty, "set_pset with new frac_obrun flags dirty", total_tests, total_failures)

        ctx%state%ssp_basis_is_dirty = .false.
        pset%dust2 = pset%dust2 + 0.3_wp
        call fsps_context_set_pset(ctx, pset)
        call assert_true(.not. ctx%state%ssp_basis_is_dirty, "set_pset with only a new dust2 keeps the basis clean", &
                         total_tests, total_failures)

        call fsps_context_destroy(ctx)
        deallocate(ctx)
    end subroutine test_ssp_dirty_flag_covers_agb_dust_and_obrun

end module test_fsps_context_mod
