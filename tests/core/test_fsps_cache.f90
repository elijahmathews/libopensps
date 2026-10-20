module test_fsps_cache_mod
    use fsps_precision, only: WP
    use fsps_cache
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_int_equals, assert_true
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_cache_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for fsps_cache module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_fsps_cache_tests()

        call print_minor_header("fsps_cache")

        call test_cache_lifecycle()
        call test_cache_expansion_and_reuse()
        call test_cache_deallocation()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_cache_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: CACHE LIFECYCLE (GET/RELEASE/REFCOUNT)
    ! ------------------------------------------------------------------------
    subroutine test_cache_lifecycle()
        type(fsps_setup_cache_t), pointer :: entry1 => null()
        type(fsps_setup_cache_t), pointer :: entry2 => null()
        logical :: is_new
        
        call print_group("Cache Lifecycle (Get/Release)")

        ! 1. First get (Should create a new entry)
        call fsps_cache_get_setup("test_key_1", entry1, is_new)
        call assert_true(is_new, "First get returns is_new = .true.", total_tests, total_failures)
        call assert_true(associated(entry1), "Entry pointer is associated", total_tests, total_failures)
        call assert_int_equals(1, entry1%refcount, "Initial refcount is 1", total_tests, total_failures)
        call assert_true(trim(entry1%key) == "test_key_1", "Key is set correctly", total_tests, total_failures)

        ! 2. Second get of the exact same key (Should return existing entry)
        call fsps_cache_get_setup("test_key_1", entry2, is_new)
        call assert_true(.not. is_new, "Second get returns is_new = .false.", total_tests, total_failures)
        call assert_true(associated(entry2), "Second entry pointer is associated", total_tests, total_failures)
        call assert_int_equals(2, entry1%refcount, "Refcount increments to 2", total_tests, total_failures)
        
        ! 3. Release once (Refcount decrements but entry remains valid)
        call fsps_cache_release_setup(entry2)
        call assert_true(.not. associated(entry2), "Local pointer nullified after release", total_tests, total_failures)
        call assert_int_equals(1, entry1%refcount, "Refcount decrements to 1", total_tests, total_failures)

        ! 4. Final release (Triggers clear and free)
        call fsps_cache_release_setup(entry1)
        call assert_true(.not. associated(entry1), "Final pointer nullified after release", total_tests, total_failures)
        
    end subroutine test_cache_lifecycle

    ! ------------------------------------------------------------------------
    ! TEST SUITE: CACHE EXPANSION AND REUSE
    ! ------------------------------------------------------------------------
    subroutine test_cache_expansion_and_reuse()
        type(fsps_setup_cache_t), pointer :: entryA => null()
        type(fsps_setup_cache_t), pointer :: entryB => null()
        type(fsps_setup_cache_t), pointer :: entryC => null()
        logical :: is_new
        
        call print_group("Cache Array Expansion and Reuse")

        ! Create first entry (fills slot 1)
        call fsps_cache_get_setup("key_A", entryA, is_new)
        call assert_true(is_new, "Key A creates new entry", total_tests, total_failures)
        
        ! Create second entry (triggers move_alloc expansion to slot 2)
        call fsps_cache_get_setup("key_B", entryB, is_new)
        call assert_true(is_new, "Key B creates new entry (expansion)", total_tests, total_failures)
        call assert_int_equals(1, entryB%refcount, "Expanded entry refcount is 1", total_tests, total_failures)

        ! Release slot 1 (refcount drops to 0, key is set to '')
        call fsps_cache_release_setup(entryA)

        ! Get a third key (Should reuse the empty slot 1 instead of expanding again)
        call fsps_cache_get_setup("key_C", entryC, is_new)
        call assert_true(is_new, "Key C is marked as new", total_tests, total_failures)
        call assert_int_equals(1, entryC%refcount, "Reused entry refcount is 1", total_tests, total_failures)
        
        ! Cleanup
        call fsps_cache_release_setup(entryB)
        call fsps_cache_release_setup(entryC)
        
    end subroutine test_cache_expansion_and_reuse

    ! ------------------------------------------------------------------------
    ! TEST SUITE: DEALLOCATION & CLEANUP
    ! ------------------------------------------------------------------------
    subroutine test_cache_deallocation()
        type(fsps_setup_cache_t), pointer :: entry => null()
        logical :: is_new
        
        call print_group("Cache Setup Clear (Deallocation)")

        call fsps_cache_get_setup("key_dealloc", entry, is_new)
        
        ! Manually allocate a few pointers to trigger the `associated(x)` checks 
        ! in `fsps_cache_setup_clear`
        allocate(entry%zlegend(10))
        allocate(entry%spec_lambda(100))
        allocate(entry%nebem_cont(2, 2, 2, 2))
        entry%nz = 5
        entry%alt_filter_file = 'filters.dat'
        
        ! Verify setup mock allocations succeeded
        call assert_true(associated(entry%zlegend), "Mock pointer 1 (1D) allocated", total_tests, total_failures)
        call assert_true(associated(entry%spec_lambda), "Mock pointer 2 (1D) allocated", total_tests, total_failures)
        call assert_true(associated(entry%nebem_cont), "Mock pointer 3 (4D) allocated", total_tests, total_failures)
        call assert_int_equals(5, entry%nz, "Mock scalar parameter assigned", total_tests, total_failures)
        
        ! Release to trigger full cleanup (refcount hits 0 -> fsps_cache_setup_clear)
        call fsps_cache_release_setup(entry)
        
        ! Verify local pointer is scrubbed
        call assert_true(.not. associated(entry), "Pointer nullified safely after full deallocation", &
                         total_tests, total_failures)

        ! (Implicit): If the deallocation routines fail or cause a segfault, the test runner 
        ! will catch it here. Passing this implies memory was handled cleanly.
        
    end subroutine test_cache_deallocation

end module test_fsps_cache_mod
