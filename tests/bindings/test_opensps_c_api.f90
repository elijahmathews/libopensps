module test_opensps_c_api_mod
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_associated, c_char, c_null_char, c_double
    use opensps_c_api, only: opensps_context_create_handle, opensps_context_destroy_handle, ctx_pool, &
                             opensps_context_set_str_handle, opensps_context_set_int_handle, &
                             opensps_context_set_float_handle, opensps_context_get_paths_handle, &
                             opensps_context_get_dims_handle, opensps_context_get_nspec_handle, &
                             opensps_context_get_nindx_handle, opensps_context_get_zsol_handle
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
        call test_accessors_of_a_context()
        call test_setters_of_a_context()
        call test_accessors_reject_bad_handles()

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

    !> The handle-based accessors reach a context through the pool: a created context that has not
    !> been set up answers them (zero sizes, its own paths), and an unknown string parameter is
    !> refused. Needs no data.
    subroutine test_accessors_of_a_context()
        integer(c_int) :: h, st, n_spec, n_time, n_indx
        real(c_double) :: z_sol
        character(kind=c_char) :: sps(260), dat(260), out(260)

        call print_group("Accessors of a context")
        call opensps_context_create_handle(h, st)

        call opensps_context_set_str_handle(h, 'imf_filename'//c_null_char, 'imf.dat'//c_null_char, st)
        call assert_true(st == 0, "A string parameter is set through the pool", total_tests, total_failures)
        call opensps_context_set_str_handle(h, 'no_such_string'//c_null_char, 'x'//c_null_char, st)
        call assert_true(st /= 0, "An unknown string parameter is refused", total_tests, total_failures)

        call opensps_context_get_dims_handle(h, n_spec, n_time, st)
        call assert_true(st == 0 .and. n_spec == ctx_pool(h)%c%state%nspec .and. n_time == ctx_pool(h)%c%state%ntfull, &
                         "The dimensions are the context's own", total_tests, total_failures)
        call opensps_context_get_nspec_handle(h, n_spec, st)
        call assert_true(st == 0 .and. n_spec == ctx_pool(h)%c%state%nspec, "The number of pixels is the context's own", &
                         total_tests, total_failures)
        call opensps_context_get_nindx_handle(h, n_indx, st)
        call assert_true(st == 0 .and. n_indx == ctx_pool(h)%c%state%nindx, "The number of indices is the context's own", &
                         total_tests, total_failures)
        call opensps_context_get_zsol_handle(h, z_sol, st)
        call assert_true(st == 0 .and. z_sol == real(ctx_pool(h)%c%state%zsol, c_double), &
                         "The solar metallicity is the context's own", total_tests, total_failures)

        sps = 'x'
        call opensps_context_get_paths_handle(h, sps, 260_c_int, dat, 260_c_int, out, 260_c_int)
        call assert_true(any(sps == c_null_char), "The paths come back null-terminated", total_tests, total_failures)

        call opensps_context_destroy_handle(h, st)
    end subroutine test_accessors_of_a_context

    !> The int and float setters reach the context through the pool; the nebular parameters are
    !> SSP-sensitive only with nebular emission in them (add_neb_emission = 2), so both outcomes of
    !> that test are exercised, and an empty key is refused. Needs no data.
    subroutine test_setters_of_a_context()
        integer(c_int) :: h, st
        character(kind=c_char), parameter :: nul = c_null_char

        call print_group("Setters of a context")
        call opensps_context_create_handle(h, st)

        call opensps_context_set_int_handle(h, 'add_neb_emission'//nul, 0_c_int, st)
        call assert_true(st == 0, "An integer setting is set through the pool", total_tests, total_failures)
        call assert_true(ctx_pool(h)%c%add_neb_emission_val == 0, "... in this context's own settings", &
                         total_tests, total_failures)
        call opensps_context_set_int_handle(h, nul, 0_c_int, st)
        call assert_true(st /= 0, "An empty integer key is refused", total_tests, total_failures)

        ! Without nebular emission in the SSPs, a nebular parameter leaves them alone; another does not
        call opensps_context_set_float_handle(h, 'gas_logu'//nul, -2.0_c_double, st)
        call assert_true(st == 0, "A nebular float parameter is set (no emission in the SSPs)", total_tests, total_failures)
        call opensps_context_set_float_handle(h, 'imf1'//nul, 1.3_c_double, st)
        call assert_true(st == 0, "An IMF float parameter is set (it always changes the SSPs)", total_tests, total_failures)
        call opensps_context_set_int_handle(h, 'add_neb_emission'//nul, 2_c_int, st)
        call opensps_context_set_float_handle(h, 'gas_logu'//nul, -2.5_c_double, st)
        call assert_true(st == 0 .and. ctx_pool(h)%c%add_neb_emission_val == 2, &
                         "A nebular float parameter is set (emission in the SSPs)", total_tests, total_failures)
        call opensps_context_set_float_handle(h, nul, 0.0_c_double, st)
        call assert_true(st /= 0, "An empty float key is refused", total_tests, total_failures)
        call opensps_context_set_str_handle(h, nul, nul, st)
        call assert_true(st /= 0, "An empty string key is refused", total_tests, total_failures)

        call opensps_context_destroy_handle(h, st)
    end subroutine test_setters_of_a_context

    !> The same accessors on a handle that does not exist or was destroyed answer status 1 (the
    !> pool is consulted, and the answer does not depend on what the slot held before).
    subroutine test_accessors_reject_bad_handles()
        integer(c_int) :: h, st, n_spec, n_time

        call print_group("Accessors with a bad handle")
        call opensps_context_create_handle(h, st)
        call opensps_context_destroy_handle(h, st)

        call opensps_context_get_dims_handle(h, n_spec, n_time, st)
        call assert_true(st == 1, "A destroyed handle is rejected", total_tests, total_failures)
        call opensps_context_get_dims_handle(0_c_int, n_spec, n_time, st)
        call assert_true(st == 1, "Handle 0 is rejected", total_tests, total_failures)
        call opensps_context_get_dims_handle(h + 1000_c_int, n_spec, n_time, st)
        call assert_true(st == 1, "A handle beyond the pool is rejected", total_tests, total_failures)
    end subroutine test_accessors_reject_bad_handles

end module test_opensps_c_api_mod
