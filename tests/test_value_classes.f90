program test_value_classes
    !> The grouping of models by the exact values of some of their parameters (opensps_value_classes): no data needed.
    use opensps_precision, only: WP
    use opensps_value_classes, only: classify_columns, count_distinct
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    integer :: total_tests = 0, total_failures = 0

    call print_major_header("Value classes")
    call test_basic()
    call test_idx_key()
    call test_cap_window()
    call print_summary_line("Value classes", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    subroutine test_basic()
        real(WP) :: v(2, 7)
        integer :: cls(7), rep(7), ncls
        logical :: ov

        call print_group("classify_columns: equal columns")
        ! columns: A B A C B A A  (A = (1, 2), B = (3, 4), C = (1, 4))
        v(:, 1) = [1.0_wp, 2.0_wp]; v(:, 2) = [3.0_wp, 4.0_wp]; v(:, 3) = [1.0_wp, 2.0_wp]; v(:, 4) = [1.0_wp, 4.0_wp]
        v(:, 5) = [3.0_wp, 4.0_wp]; v(:, 6) = [1.0_wp, 2.0_wp]; v(:, 7) = [1.0_wp, 2.0_wp]
        call classify_columns(v, 7, 0, cls, rep, ncls, ov)
        call assert_true(.not. ov .and. ncls == 3 .and. all(cls == [1, 2, 1, 3, 2, 1, 1]) .and. all(rep(1:3) == [1, 2, 4]), &
                         "classes of equal columns and their first members", total_tests, total_failures)
        call assert_true(count_distinct(v, 10) == 3 .and. count_distinct(v, 2) == 2, "count_distinct, and its cap", &
                         total_tests, total_failures)
        v(:, 4) = [1.0_wp, 2.0_wp + 1.0e-15_wp]
        call classify_columns(v, 7, 0, cls, rep, ncls, ov)
        call assert_true(ncls == 3 .and. cls(4) == 3, "equality is exact: a difference of 1e-15 is another value", &
                         total_tests, total_failures)
    end subroutine test_basic

    subroutine test_idx_key()
        real(WP) :: v(1, 6)
        integer :: cls(4), rep(4), ncls, idx(4), key(4)
        logical :: ov

        call print_group("classify_columns: a selection of columns and a key")
        v(1, :) = [5.0_wp, 1.0_wp, 5.0_wp, 2.0_wp, 1.0_wp, 5.0_wp]
        idx = [6, 5, 2, 1]               ! the values 5, 1, 1, 5
        call classify_columns(v, 4, 0, cls, rep, ncls, ov, idx=idx)
        call assert_true(ncls == 2 .and. all(cls == [1, 2, 2, 1]) .and. all(rep(1:2) == [1, 2]), &
                         "a selection: classes of the columns idx(j), representatives as positions in idx", &
                         total_tests, total_failures)
        key = [1, 1, 2, 1]               ! the third (a 1) has another key: a class of its own
        call classify_columns(v, 4, 0, cls, rep, ncls, ov, idx=idx, key=key)
        call assert_true(ncls == 3 .and. all(cls == [1, 2, 3, 1]), "the key must match as well", total_tests, total_failures)
    end subroutine test_idx_key

    subroutine test_cap_window()
        real(WP) :: v(1, 30)
        integer :: cls(30), rep(30), ncls, j
        logical :: ov

        call print_group("classify_columns: the cap and the window")
        do j = 1, 30
            v(1, j) = real(j, WP)
        end do
        call classify_columns(v, 8, 0, cls, rep, ncls, ov)
        call assert_true(ov .and. ncls == 8 .and. all(cls(1:8) == [(j, j = 1, 8)]), &
                         "more distinct columns than the cap: overflow, the classes found are valid", total_tests, total_failures)
        call classify_columns(v, 30, 0, cls, rep, ncls, ov)
        call assert_true(.not. ov .and. ncls == 30, "a cap that holds them all", total_tests, total_failures)
        ! a repeat of the first column after 20 others: outside a window of 16 it is a class of its own
        v(1, 1:20) = [(real(j, WP), j = 1, 20)]
        v(1, 21) = 1.0_wp
        call classify_columns(v(:, 1:21), 30, 16, cls(1:21), rep, ncls, ov)
        call assert_true(ncls == 21 .and. cls(21) == 21, "the window bounds the comparison: a late repeat is its own class", &
                         total_tests, total_failures)
        call classify_columns(v(:, 1:21), 30, 0, cls(1:21), rep, ncls, ov)
        call assert_true(ncls == 20 .and. cls(21) == 1, "without a window it is found", total_tests, total_failures)
        v(1, 21) = 5.0_wp                ! within 16 of class 5
        call classify_columns(v(:, 1:21), 30, 16, cls(1:21), rep, ncls, ov)
        call assert_true(ncls == 20 .and. cls(21) == 5, "inside the window it is found", total_tests, total_failures)
    end subroutine test_cap_window

end program test_value_classes
