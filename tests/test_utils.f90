module test_utils_mod
    !> Assertions and report formatting shared by the test programs.
    !>
    !> Two output modes:
    !>   - Interactive (default): coloured [PASS]/[FAIL] lines grouped under headers.
    !>   - TAP (FSPS_TEST_TAP=1, set by meson for these tests): one "ok N - module > group > label"
    !>     or "not ok N - ..." line per assertion, with headers and failure details as "#" comments,
    !>     so meson reports every assertion individually (console, testlog.json, testlog.junit.xml).
    !>     Programs call tap_finish() at the end to print the plan line.
    use fsps_precision, only: WP
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_class, ieee_negative_inf, ieee_class_type, operator(==)
    implicit none

    ! ANSI Color Codes
    character(len=4), parameter :: C_RESET    = char(27)//'[0m'
    character(len=4), parameter :: C_BOLD     = char(27)//'[1m'
    character(len=5), parameter :: C_RED      = char(27)//'[31m'
    character(len=5), parameter :: C_GREEN    = char(27)//'[32m'
    character(len=7), parameter :: C_RED_BOLD = char(27)//'[1;31m' ! Bold Red

    public :: assert_float_equals, assert_int_equals, assert_true
    public :: assert_is_nan, assert_is_neg_inf, assert_relative_error
    public :: print_major_header, print_minor_header, print_group, print_summary_line
    public :: tap_finish
    public :: C_RESET, C_BOLD, C_RED, C_GREEN, C_RED_BOLD

    logical, save, private :: mode_checked = .false.
    logical, save, private :: tap = .false.
    integer, save, private :: tap_count = 0
    character(len=128), save, private :: current_section = ''
    character(len=256), save, private :: current_group = ''

contains

    !> True when FSPS_TEST_TAP is set to a non-empty value other than 0.
    logical function tap_mode()
        character(len=8) :: buf
        integer :: stat

        if (.not. mode_checked) then
            call get_environment_variable('FSPS_TEST_TAP', buf, status=stat)
            tap = (stat == 0 .and. len_trim(buf) > 0 .and. trim(buf) /= '0')
            mode_checked = .true.
        end if
        tap_mode = tap
    end function tap_mode

    !> Prints the TAP plan line; call once at the end of a test program.
    subroutine tap_finish()
        if (tap_mode()) write(*, '(A,I0)') '1..', tap_count
    end subroutine tap_finish

    !> @brief Print a major section header in Bold
    subroutine print_major_header(text)
        character(len=*), intent(in) :: text

        if (tap_mode()) then
            write(*, '(2A)') '# ', trim(text)
            return
        end if
        print *
        print *, C_BOLD // "=========================================================" // C_RESET
        print *, C_BOLD // trim(text) // C_RESET
        print *, C_BOLD // "=========================================================" // C_RESET
    end subroutine

    !> @brief Print a minor section header in Bold (names the module in TAP test names)
    subroutine print_minor_header(text)
        character(len=*), intent(in) :: text

        current_section = text
        current_group = ''
        if (tap_mode()) then
            write(*, '(2A)') '# ', trim(text)
            return
        end if
        print *
        print *, C_BOLD // "---------------------------------------------------------" // C_RESET
        print *, C_BOLD // trim(text) // C_RESET
        print *, C_BOLD // "---------------------------------------------------------" // C_RESET
    end subroutine

    !> @brief Print a test group sub-header in Bold (names the group in TAP test names)
    subroutine print_group(text)
        character(len=*), intent(in) :: text

        current_group = text
        if (tap_mode()) then
            write(*, '(2A)') '# Group: ', trim(text)
            return
        end if
        print *, C_BOLD // ">> Group: " // trim(text) // C_RESET
    end subroutine

    !> @brief Print a formatted summary line (Label: Pass / Total)
    !> Handles the coloring logic: Green numbers if all passed, Bold Red if failures exist.
    subroutine print_summary_line(label, n_pass, n_total)
        character(len=*), intent(in) :: label
        integer, intent(in) :: n_pass, n_total
        character(len=20) :: s_pass, s_total ! Buffers for integer conversion

        ! Write integers to strings to allow concatenation
        write(s_pass, '(I0)') n_pass
        write(s_total, '(I0)') n_total

        if (tap_mode()) then
            write(*, '(5A)') '# ', trim(label), ': ', trim(s_pass) // '/', trim(s_total)
        else if (n_pass == n_total) then
            ! All passed: Green numbers
            print *, trim(label), ": ", C_GREEN // trim(s_pass) // C_RESET, "/", C_GREEN // trim(s_total) // C_RESET
        else
            ! Failures: Bold Red numbers
            print *, trim(label), ": ", C_RED_BOLD // trim(s_pass) // C_RESET, "/", C_RED_BOLD // trim(s_total) // C_RESET
        end if
    end subroutine

    !> Records one assertion: counts it, and prints its result in the active mode.
    subroutine report(passed, label, detail, n_tests, n_fails)
        logical, intent(in) :: passed
        character(len=*), intent(in) :: label, detail
        integer, intent(inout) :: n_tests, n_fails

        n_tests = n_tests + 1
        if (.not. passed) n_fails = n_fails + 1

        if (tap_mode()) then
            tap_count = tap_count + 1
            if (passed) then
                write(*, '(A,I0,2A)') 'ok ', tap_count, ' - ', tap_name(label)
            else
                write(*, '(A,I0,2A)') 'not ok ', tap_count, ' - ', tap_name(label)
                if (len_trim(detail) > 0) write(*, '(2A)') '#   ', trim(adjustl(detail))
            end if
        else if (passed) then
            print *, "  [" // C_GREEN // "PASS" // C_RESET // "] " // label
        else
            print *, "  [" // C_RED_BOLD // "FAIL" // C_RESET // "] " // label
            if (len_trim(detail) > 0) print *, "         " // trim(adjustl(detail))
        end if
    end subroutine report

    !> "section > group > label", with '#' (a TAP directive marker) replaced.
    function tap_name(label) result(name)
        character(len=*), intent(in) :: label
        character(len=:), allocatable :: name
        integer :: i

        name = trim(label)
        if (len_trim(current_group) > 0) name = trim(current_group) // ' > ' // name
        if (len_trim(current_section) > 0) name = trim(current_section) // ' > ' // name
        do i = 1, len(name)
            if (name(i:i) == '#') name(i:i) = 'N'
        end do
    end function tap_name

    subroutine assert_float_equals(expected, actual, tol, label, n_tests, n_fails)
        real(WP), intent(in) :: expected, actual, tol
        integer, intent(inout) :: n_tests, n_fails
        character(len=*), intent(in) :: label
        character(len=160) :: detail

        write(detail, *) "Expected: ", expected, " Got: ", actual, " Diff: ", abs(expected - actual)
        call report(abs(expected - actual) <= tol, label, detail, n_tests, n_fails)
    end subroutine assert_float_equals

    subroutine assert_int_equals(expected, actual, label, n_tests, n_fails)
        integer, intent(in) :: expected, actual
        integer, intent(inout) :: n_tests, n_fails
        character(len=*), intent(in) :: label
        character(len=160) :: detail

        write(detail, *) "Expected: ", expected, " Got: ", actual
        call report(expected == actual, label, detail, n_tests, n_fails)
    end subroutine assert_int_equals

    subroutine assert_true(condition, label, n_tests, n_fails)
        logical, intent(in) :: condition
        integer, intent(inout) :: n_tests, n_fails
        character(len=*), intent(in) :: label

        call report(condition, label, '', n_tests, n_fails)
    end subroutine assert_true

    subroutine assert_is_nan(val, label, n_tests, n_fails)
        real(WP), intent(in) :: val
        integer, intent(inout) :: n_tests, n_fails
        character(len=*), intent(in) :: label
        character(len=160) :: detail

        write(detail, *) "Expected NaN, Got: ", val
        call report(ieee_is_nan(val), label, detail, n_tests, n_fails)
    end subroutine assert_is_nan

    subroutine assert_is_neg_inf(val, label, n_tests, n_fails)
        real(WP), intent(in) :: val
        integer, intent(inout) :: n_tests, n_fails
        character(len=*), intent(in) :: label
        type(ieee_class_type) :: cl

        cl = ieee_class(val)
        call report(cl == ieee_negative_inf, label, "Expected -Inf", n_tests, n_fails)
    end subroutine assert_is_neg_inf

    subroutine assert_relative_error(expected, actual, rel_tol, label, n_tests, n_fails)
        real(WP), intent(in) :: expected, actual, rel_tol
        integer, intent(inout) :: n_tests, n_fails
        character(len=*), intent(in) :: label
        real(WP) :: rel_diff
        character(len=160) :: detail

        rel_diff = abs((expected - actual) / expected)
        write(detail, *) "Expected:", expected, " Got:", actual, " RelDiff:", rel_diff
        call report(rel_diff <= rel_tol, label, detail, n_tests, n_fails)
    end subroutine assert_relative_error

end module test_utils_mod
