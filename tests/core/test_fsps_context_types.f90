module test_fsps_context_types_mod
    use fsps_precision, only: WP
    use fsps_context_types
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_int_equals, assert_true
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_context_types_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for fsps_context_types module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_fsps_context_types_tests()

        call print_minor_header("fsps_context_types")

        call test_empty_state_destroy()
        call test_allocatable_cleanup()
        call test_pointer_cleanup()
        call test_scalar_reset()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_context_types_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: EMPTY STATE DESTRUCTION
    ! ------------------------------------------------------------------------
    subroutine test_empty_state_destroy()
        type(fsps_context_state_t) :: state
        
        call print_group("Empty State Destruction")

        ! Calling destroy on a completely uninitialized state. 
        ! If any `allocated()` or `associated()` checks are missing in the 
        ! source code, this will trigger a segmentation fault.
        call fsps_context_state_destroy(state)
        
        call assert_true(.true., "Empty state destroyed without crashing", total_tests, total_failures)

    end subroutine test_empty_state_destroy

    ! ------------------------------------------------------------------------
    ! TEST SUITE: ALLOCATABLE CLEANUP
    ! ------------------------------------------------------------------------
    subroutine test_allocatable_cleanup()
        type(fsps_context_state_t) :: state
        
        call print_group("Allocatable Arrays Cleanup")

        ! Allocate a representative sample of arrays
        allocate(state%spec_ssp_zz(10, 10, 10))
        allocate(state%csp_weights(5, 5, 5))
        allocate(state%out_csp_spec(100, 2))
        allocate(state%gas_neb_line_reduced(10, 10))
        
        call assert_true(allocated(state%spec_ssp_zz), "spec_ssp_zz initially allocated", total_tests, total_failures)
        
        ! Destroy
        call fsps_context_state_destroy(state)
        
        ! Verify deallocation
        call assert_true(.not. allocated(state%spec_ssp_zz), "spec_ssp_zz safely deallocated", total_tests, total_failures)
        call assert_true(.not. allocated(state%csp_weights), "csp_weights safely deallocated", total_tests, total_failures)
        call assert_true(.not. allocated(state%out_csp_spec), "out_csp_spec safely deallocated", total_tests, total_failures)
        call assert_true(.not. allocated(state%gas_neb_line_reduced), "gas_neb_line_reduced safely deallocated", &
                         total_tests, total_failures)

    end subroutine test_allocatable_cleanup

    ! ------------------------------------------------------------------------
    ! TEST SUITE: POINTER CLEANUP
    ! ------------------------------------------------------------------------
    subroutine test_pointer_cleanup()
        type(fsps_context_state_t) :: state
        
        ! Dummy targets to associate pointers with
        real(WP), target :: dummy_1d(10)
        real(WP), target :: dummy_2d(10, 10)
        real(WP), target :: dummy_4d(2, 2, 2, 2)
        
        call print_group("Pointer Nullification")

        ! Associate a representative sample of pointers
        state%zlegend => dummy_1d
        state%indexdefined => dummy_2d
        state%nebem_cont => dummy_4d
        
        call assert_true(associated(state%zlegend), "zlegend initially associated", total_tests, total_failures)
        
        ! Destroy
        call fsps_context_state_destroy(state)
        
        ! Verify nullification
        call assert_true(.not. associated(state%zlegend), "zlegend safely nullified", total_tests, total_failures)
        call assert_true(.not. associated(state%indexdefined), "indexdefined safely nullified", total_tests, total_failures)
        call assert_true(.not. associated(state%nebem_cont), "nebem_cont safely nullified", total_tests, total_failures)

    end subroutine test_pointer_cleanup

    ! ------------------------------------------------------------------------
    ! TEST SUITE: SCALAR RESET
    ! ------------------------------------------------------------------------
    subroutine test_scalar_reset()
        type(fsps_context_state_t) :: state
        
        call print_group("Scalar State Reset")

        ! Mutate a variety of scalars to non-default values
        state%nt = 99
        state%nz = 5
        state%zsol = 0.02_wp
        state%str_dustem = 'CUSTOM'
        state%imf_lower_limit = 0.1_wp
        state%ssp_basis_is_dirty = .false.
        
        ! Destroy
        call fsps_context_state_destroy(state)
        
        ! Verify resets to default values matching the type declaration
        call assert_int_equals(0, state%nt, "nt reset to 0", total_tests, total_failures)
        call assert_int_equals(0, state%nz, "nz reset to 0", total_tests, total_failures)
        call assert_float_equals(0.0_wp, state%zsol, 1.0e-6_wp, "zsol reset to 0.0", total_tests, total_failures)
        call assert_true(trim(state%str_dustem) == 'DL07', "str_dustem reset to 'DL07'", total_tests, total_failures)
        call assert_float_equals(0.08_wp, state%imf_lower_limit, 1.0e-6_wp, "imf_lower_limit reset to 0.08", &
                                 total_tests, total_failures)
        call assert_true(state%ssp_basis_is_dirty, "ssp_basis_is_dirty reset to .true.", total_tests, total_failures)

    end subroutine test_scalar_reset

end module test_fsps_context_types_mod
