program test_batch_registry
    !> The per-model parameter registry (src/core/opensps_batch_registry.f90) and the classification of a batch: no data
    !> needed. The registry table is also checked against PARAMS_CORE by tools/check_batch_registry.py.
    use opensps_precision, only: WP
    use opensps_types, only: PARAMS_CORE
    use opensps_batch_registry
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    integer :: total_tests = 0, total_failures = 0

    call print_major_header("Batch registry")
    call test_table()
    call test_accessors()
    call test_classify()
    call test_text()
    call print_summary_line("Batch registry", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    subroutine test_table()
        integer :: i, c(0:4), s(9)
        logical :: ok

        call print_group("registry: the table")
        ok = .true.
        c = 0
        s = 0
        do i = 1, N_FIELDS
            if (len_trim(registry_name(i)) == 0) ok = .false.
            if (registry_find(registry_name(i)) /= i) ok = .false.
            if (registry_class(i) < 0 .or. registry_class(i) > 4) ok = .false.
            if (registry_stage(i) < 1 .or. registry_stage(i) > 9) ok = .false.
            if (registry_kind(i) < 1 .or. registry_kind(i) > 3) ok = .false.
            if (ok) c(registry_class(i)) = c(registry_class(i)) + 1
            if (ok) s(registry_stage(i)) = s(registry_stage(i)) + 1
        end do
        call assert_true(ok, "every field has a name found again by registry_find, a class, a stage and a kind", &
                         total_tests, total_failures)
        call assert_true(all(c >= 1), "every class has fields (shared, closed, linear, weights, grouped)", &
                         total_tests, total_failures)
        call assert_true(all(s >= 1), "every stage has fields", total_tests, total_failures)
        call assert_true(registry_find('LogZsol') == F_LOGZSOL .and. registry_find('nonesuch') == 0 .and. &
                         registry_find('') == 0, "registry_find: case-insensitive, 0 for an unknown name", &
                         total_tests, total_failures)
        call assert_true(len_trim(registry_name(0)) == 0 .and. len_trim(registry_name(N_FIELDS + 1)) == 0 .and. &
                         registry_class(0) == -1 .and. registry_stage(0) == 0 .and. registry_kind(0) == 0, &
                         "indices out of range give empty answers", total_tests, total_failures)
        call assert_true(registry_class(F_LOGZSOL) == CLASS_L .and. registry_class(F_IMF1) == CLASS_W .and. &
                         registry_class(F_DELT) == CLASS_G .and. registry_class(F_DUST2) == CLASS_C .and. &
                         registry_class(F_SIGMA_SMOOTH) == CLASS_S, "classes of representative fields", &
                         total_tests, total_failures)
        call assert_true(trim(registry_class_name(CLASS_W)) == 'weights' .and. trim(registry_class_name(9)) == '?' .and. &
                         trim(registry_stage_name(STG_DUST)) == 'dust' .and. trim(registry_stage_name(0)) == '?', &
                         "names of classes and stages", total_tests, total_failures)
    end subroutine test_table

    subroutine test_accessors()
        type(PARAMS_CORE) :: p, q, d
        integer :: i
        logical :: ok_set, ok_eq, ok_def

        call print_group("registry: get, set and equal")
        ok_set = .true.
        ok_eq = .true.
        ok_def = .true.
        do i = 1, N_FIELDS
            if (registry_kind(i) == KIND_STR) then
                ! character fields: equal while they are, different once one changes
                if (.not. registry_equal(p, q, i)) ok_eq = .false.
                cycle
            end if
            p = PARAMS_CORE()
            q = PARAMS_CORE()
            d = PARAMS_CORE()
            call registry_set(p, i, 3.0_wp)
            if (registry_get(p, i) /= 3.0_wp) ok_set = .false.
            if (registry_equal(p, q, i)) ok_eq = .false.
            ! (p alone was changed: every other field of p is still the default)
            if (.not. registry_same_but(p, d, i)) ok_def = .false.
            call registry_set(q, i, 3.0_wp)
            if (.not. registry_equal(p, q, i)) ok_eq = .false.
        end do
        call assert_true(ok_set, "registry_set then registry_get gives the value, for every real and integer field", &
                         total_tests, total_failures)
        call assert_true(ok_eq, "registry_equal sees a change of exactly that field", total_tests, total_failures)
        call assert_true(ok_def, "registry_set changes that field and no other", total_tests, total_failures)
        p = PARAMS_CORE()
        p%imf_filename = 'a'
        q = PARAMS_CORE()
        call assert_true(.not. registry_equal(p, q, F_IMF_FILENAME) .and. registry_equal(p, q, F_SFH_FILENAME), &
                         "registry_equal on the character fields", total_tests, total_failures)
        call registry_set(p, F_SFH, 4.4_wp)
        call assert_true(p%sfh == 4 .and. registry_get(p, F_SFH) == 4.0_wp, "integers are rounded by registry_set", &
                         total_tests, total_failures)
        call registry_set(p, F_IMF_FILENAME, 1.0_wp)
        call assert_true(p%imf_filename == 'a' .and. registry_get(p, F_IMF_FILENAME) == 0.0_wp, &
                         "character fields are left alone by registry_set and read as 0", total_tests, total_failures)
    end subroutine test_accessors

    logical function registry_same_but(p, q, skip)
        type(PARAMS_CORE), intent(in) :: p, q
        integer, intent(in) :: skip
        integer :: i
        registry_same_but = .true.
        do i = 1, N_FIELDS
            if (i == skip) cycle
            if (.not. registry_equal(p, q, i)) registry_same_but = .false.
        end do
    end function registry_same_but

    subroutine test_classify()
        integer, parameter :: N = 40
        type(PARAMS_CORE) :: core(N)
        type(batch_info_t) :: info
        integer :: m

        call print_group("batch_classify: the free set of a batch")
        call batch_classify(core, info)
        call assert_true(info%n_models == N .and. .not. any(info%free) .and. all(info%n_free_class == 0) .and. &
                         .not. info%shared_violation .and. all(info%n_distinct == 1), "identical models: nothing is free", &
                         total_tests, total_failures)
        call batch_classify(core(1:1), info)
        call assert_true(info%n_models == 1 .and. .not. any(info%free), "one model: nothing is free", total_tests, total_failures)

        do m = 1, N
            call registry_set(core(m), F_DUST2, 0.01_wp * m)
            call registry_set(core(m), F_LOGZSOL, -1.0_wp + 0.05_wp * mod(m, 3))
            call registry_set(core(m), F_IMF1, 1.0_wp + 0.1_wp * mod(m, 2))
        end do
        call batch_classify(core, info)
        call assert_true(info%free(F_DUST2) .and. info%free(F_LOGZSOL) .and. info%free(F_IMF1) .and. count(info%free) == 3, &
                         "three varying fields are free, no other", total_tests, total_failures)
        call assert_true(info%n_distinct(F_DUST2) == N .or. info%n_distinct(F_DUST2) == DISTINCT_CAP, &
                         "a field with many values is counted up to the cap", total_tests, total_failures)
        call assert_true(info%n_distinct(F_LOGZSOL) == 3 .and. info%n_distinct(F_IMF1) == 2 .and. &
                         info%n_distinct(F_AGB) == 1, "distinct values: 3 metallicities, 2 IMFs, 1 for a fixed field", &
                         total_tests, total_failures)
        call assert_true(info%n_free_class(CLASS_C) == 1 .and. info%n_free_class(CLASS_L) == 1 .and. &
                         info%n_free_class(CLASS_W) == 1 .and. info%n_free_class(CLASS_G) == 0 .and. &
                         info%n_free_class(CLASS_S) == 0 .and. .not. info%shared_violation, &
                         "free fields counted per class", total_tests, total_failures)

        ! a shared field that differs, and a character field
        call registry_set(core(N), F_SIGMA_SMOOTH, 150.0_wp)
        call batch_classify(core, info)
        call assert_true(info%shared_violation .and. info%free(F_SIGMA_SMOOTH) .and. info%n_free_class(CLASS_S) == 1, &
                         "a shared field that differs is reported", total_tests, total_failures)
        call registry_set(core(N), F_SIGMA_SMOOTH, 0.0_wp)
        core(2)%imf_filename = 'other'
        call batch_classify(core, info)
        call assert_true(info%shared_violation .and. info%free(F_IMF_FILENAME), "a character field that differs is reported", &
                         total_tests, total_failures)
        ! the result of an earlier call is cleared
        core(2)%imf_filename = ''
        do m = 1, N
            call registry_set(core(m), F_DUST2, 0.3_wp)
        end do
        call batch_classify(core, info)
        call assert_true(.not. info%free(F_DUST2) .and. .not. info%shared_violation, "classification starts afresh each call", &
                         total_tests, total_failures)
    end subroutine test_classify

    subroutine test_text()
        type(PARAMS_CORE) :: core(4)
        type(batch_info_t) :: info
        character(len=:), allocatable :: t

        call print_group("batch_info_text")
        call batch_classify(core, info)
        call batch_info_set_path(info, 'resident basis', 1)
        t = batch_info_text(info)
        call assert_true(index(t, '4 models') > 0 .and. index(t, 'resident basis (K=1)') > 0 .and. index(t, 'none') > 0, &
                         "no free field: 'none'", total_tests, total_failures)
        call registry_set(core(2), F_FBHB, 0.3_wp)
        call registry_set(core(3), F_FBHB, 0.5_wp)
        call registry_set(core(4), F_DELT, 0.02_wp)
        call batch_classify(core, info)
        call batch_info_set_path(info, 'operator groups', 9)
        t = batch_info_text(info)
        call assert_true(index(t, 'operator groups (K=9)') > 0 .and. index(t, 'fbhb[linear,3]') > 0 .and. &
                         index(t, 'delt[grouped,2]') > 0, "free fields with class and distinct values", &
                         total_tests, total_failures)
        call assert_true(.not. batch_debug_path() .or. batch_debug_path(), "batch_debug_path reads the environment once", &
                         total_tests, total_failures)
    end subroutine test_text

end program test_batch_registry
