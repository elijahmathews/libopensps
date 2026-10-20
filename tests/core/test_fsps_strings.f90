module test_fsps_strings_mod
    use fsps_precision, only: WP
    use fsps_strings
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, assert_true
    implicit none

    ! Global failure counter
    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_strings_tests, total_failures, total_tests

contains

    !> @brief
    !> Unit test suite for fsps_strings module.
    !> Designed for CI usage: exits with status 0 on success, 1 on failure.
    subroutine run_fsps_strings_tests()

        call print_minor_header("fsps_strings")

        call test_to_lower()
        call test_to_string_int()
        call test_to_string_real()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)

    end subroutine run_fsps_strings_tests

    ! ------------------------------------------------------------------------
    ! TEST SUITE: STRING CONVERSION (to_lower)
    ! ------------------------------------------------------------------------
    subroutine test_to_lower()
        call print_group("String Case Conversion")

        call assert_true(to_lower('MIST') == 'mist', "All uppercase to lowercase", total_tests, total_failures)
        call assert_true(to_lower('PaDoVa') == 'padova', "Mixed case to lowercase", total_tests, total_failures)
        call assert_true(to_lower('already_lower') == 'already_lower', "Already lowercase unchanged", &
                         total_tests, total_failures)
        call assert_true(to_lower('123_abc_DEF!') == '123_abc_def!', "Numbers and symbols unchanged", &
                         total_tests, total_failures)
        call assert_true(to_lower('') == '', "Empty string safe", total_tests, total_failures)
        
    end subroutine test_to_lower

    ! ------------------------------------------------------------------------
    ! TEST SUITE: INTEGER STRINGIFICATION (to_string_int)
    ! ------------------------------------------------------------------------
    subroutine test_to_string_int()
        call print_group("Integer to String")

        call assert_true(trim(to_string_int(42)) == '42', "Positive integer", total_tests, total_failures)
        call assert_true(trim(to_string_int(-123)) == '-123', "Negative integer", total_tests, total_failures)
        call assert_true(trim(to_string_int(0)) == '0', "Zero integer", total_tests, total_failures)
        
    end subroutine test_to_string_int

    ! ------------------------------------------------------------------------
    ! TEST SUITE: REAL STRINGIFICATION (to_string_real)
    ! ------------------------------------------------------------------------
    subroutine test_to_string_real()
        call print_group("Real to String")

        ! We use explicit format strings here to guarantee cross-compiler consistency, 
        ! since the default (G12.5) can render slightly differently depending on the compiler.
        
        call assert_true(trim(to_string_real(3.14_wp, '(F4.2)')) == '3.14', "Custom format (positive)", &
                         total_tests, total_failures)
        call assert_true(trim(to_string_real(-0.5_wp, '(F4.1)')) == '-0.5', "Custom format (negative)", &
                         total_tests, total_failures)
        call assert_true(trim(to_string_real(0.0_wp, '(F3.1)')) == '0.0', "Zero formatting", &
                         total_tests, total_failures)
        
    end subroutine test_to_string_real

end module test_fsps_strings_mod
