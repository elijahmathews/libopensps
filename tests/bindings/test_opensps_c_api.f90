module test_opensps_c_api_mod
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_associated
    use opensps_c_api, only: opensps_context_create_handle, opensps_context_destroy_handle, ctx_pool
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, assert_true
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_opensps_c_api_tests, total_failures, total_tests

contains

    !> @brief Unit tests of the handle-based context pool of the C API (issue #77).
    !> @details OpenMP target mappings are keyed by host address: the device copy of a context is
    !>   found through the address of the context and of its allocatable components. A context must
    !>   therefore never move in host memory, in particular not when the pool grows to make room for
    !>   another context (a deep copy of the old contexts frees their components under the live
    !>   mappings; the first context's next batch then aborts on a GPU, and nothing at all happens
    !>   on a CPU build, so a GPU-less CI can only check the addresses). Needs no data.
    subroutine run_opensps_c_api_tests()
        call print_minor_header("opensps_c_api (context pool)")

        call test_pool_growth_keeps_contexts_in_place()
        call test_slot_reuse()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_opensps_c_api_tests

    subroutine test_pool_growth_keeps_contexts_in_place()
        integer(c_int) :: h(4), st
        type(c_ptr) :: ctx_addr(4), comp_addr(4)
        integer :: i, j
        logical :: ctx_same, comp_same, content_same

        call print_group("Growing the pool")
        ctx_same = .true.
        comp_same = .true.
        content_same = .true.

        do i = 1, 4
            call opensps_context_create_handle(h(i), st)
            call assert_true(st == 0 .and. h(i) == i, "Handle " // char(ichar('0') + i) // " created", &
                             total_tests, total_failures)
            ! Give the new context an allocatable component with recognizable content (a stand-in for
            ! the photometry grid and the batch output, which a batch allocates and maps)
            allocate (ctx_pool(h(i))%c%state%weight_ssp(3, 2))
            ctx_pool(h(i))%c%state%weight_ssp = real(i, kind(ctx_pool(h(i))%c%state%weight_ssp))
            ctx_addr(i) = c_loc(ctx_pool(h(i))%c)
            comp_addr(i) = c_loc(ctx_pool(h(i))%c%state%weight_ssp)

            ! Every context created so far (the pool grew by one slot above, for i > 1)
            do j = 1, i
                if (.not. c_associated(c_loc(ctx_pool(h(j))%c), ctx_addr(j))) ctx_same = .false.
                if (.not. allocated(ctx_pool(h(j))%c%state%weight_ssp)) then
                    comp_same = .false.
                else
                    if (.not. c_associated(c_loc(ctx_pool(h(j))%c%state%weight_ssp), comp_addr(j))) comp_same = .false.
                    if (any(ctx_pool(h(j))%c%state%weight_ssp /= real(j, kind(ctx_pool(h(j))%c%state%weight_ssp)))) &
                        content_same = .false.
                end if
            end do
        end do

        call assert_true(ctx_same, "No context moved in host memory when the pool grew", total_tests, total_failures)
        call assert_true(comp_same, "No allocatable component of a context moved when the pool grew", &
                         total_tests, total_failures)
        call assert_true(content_same, "The contents of the contexts survived the growth", total_tests, total_failures)

        do i = 1, 4
            call opensps_context_destroy_handle(h(i), st)
        end do
    end subroutine test_pool_growth_keeps_contexts_in_place

    subroutine test_slot_reuse()
        integer(c_int) :: a, b, c, st
        type(c_ptr) :: addr_b

        call print_group("Slot reuse")
        call opensps_context_create_handle(a, st)
        call opensps_context_create_handle(b, st)
        addr_b = c_loc(ctx_pool(b)%c)
        call opensps_context_destroy_handle(b, st)
        call opensps_context_create_handle(c, st)

        call assert_true(c == b, "A destroyed handle is reused before the pool grows", total_tests, total_failures)
        call assert_true(c_associated(c_loc(ctx_pool(c)%c), addr_b), "The reused slot keeps its context in place", &
                         total_tests, total_failures)

        call opensps_context_destroy_handle(c, st)
        call opensps_context_destroy_handle(a, st)
    end subroutine test_slot_reuse

end module test_opensps_c_api_mod
