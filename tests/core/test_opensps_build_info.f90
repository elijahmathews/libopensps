module test_opensps_build_info_mod
    use, intrinsic :: iso_c_binding, only: c_char, c_null_char, c_ptr, c_associated, c_f_pointer
    use opensps_build_info, only: opensps_build_id, opensps_get_build_id
    use opensps_context_types, only: opensps_context_t
    use opensps_environment, only: opensps_print_env_info, opensps_print_optional
    use opensps_version, only: OPENSPS_VERSION_STRING
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, assert_true
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_opensps_build_info_tests, total_failures, total_tests

contains

    !> @brief Unit tests of the build identifier (the source revision a build was made from).
    !> @details The value itself depends on how the tests were built (a commit, a bare hash in a
    !>   shallow checkout, empty for a release or an archive), so the tests check the invariants
    !>   that hold for every build: a release reports nothing, the Fortran and C views agree, and
    !>   the banner can be printed.
    subroutine run_opensps_build_info_tests()
        call print_minor_header("opensps_build_info")

        call test_build_id_invariants()
        call test_c_view_matches_fortran()
        call test_banner()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_opensps_build_info_tests

    subroutine test_build_id_invariants()
        character(len=:), allocatable :: id
        integer :: i
        logical :: printable

        call print_group("Build identifier")
        id = opensps_build_id()

        ! The release tag is suppressed, so no build reports 'v<version>' as its identifier.
        call assert_true(id /= 'v'//OPENSPS_VERSION_STRING, "The release tag is reported as no build id", &
                         total_tests, total_failures)

        printable = .true.
        do i = 1, len(id)
            if (id(i:i) <= ' ' .or. id(i:i) > '~') printable = .false.
        end do
        call assert_true(printable, "Identifier is printable with no blanks or newlines", &
                         total_tests, total_failures)
    end subroutine test_build_id_invariants

    subroutine test_c_view_matches_fortran()
        character(len=:), allocatable :: id
        character(kind=c_char), pointer :: chars(:)
        type(c_ptr) :: p
        logical :: same
        integer :: i

        call print_group("C API view")
        id = opensps_build_id()
        p = opensps_get_build_id()
        call assert_true(c_associated(p), "opensps_get_build_id returns a pointer", total_tests, total_failures)
        if (.not. c_associated(p)) return

        call c_f_pointer(p, chars, [len(id) + 1])
        same = (chars(len(id) + 1) == c_null_char)
        do i = 1, len(id)
            if (chars(i) /= id(i:i)) same = .false.
        end do
        call assert_true(same, "Null-terminated and identical to the Fortran identifier", &
                         total_tests, total_failures)
    end subroutine test_c_view_matches_fortran

    subroutine test_banner()
        type(opensps_context_t) :: ctx
        character(len=200) :: line
        character(len=:), allocatable :: id
        integer :: u, ios
        logical :: has_version, has_build, build_ok

        call print_group("Environment banner")
        id = opensps_build_id()

        ! The banner goes to a scratch file (not the TAP stream) and is read back: the version
        ! line is there, and the Build line exactly when the build has an identifier.
        open (newunit=u, status='scratch', action='readwrite')
        call opensps_print_env_info(ctx, u)
        rewind (u)
        has_version = .false.
        has_build = .false.
        build_ok = .true.
        do
            read (u, '(A)', iostat=ios) line
            if (ios /= 0) exit
            if (index(line, 'OpenSPS Version: ' // OPENSPS_VERSION_STRING) > 0) has_version = .true.
            if (index(line, 'Build:') > 0) then
                has_build = .true.
                build_ok = (len(id) > 0 .and. index(line, id) > 0)
            end if
        end do
        close (u)
        call assert_true(has_version, "Banner shows the version", total_tests, total_failures)
        call assert_true(has_build .eqv. (len(id) > 0), "Banner shows a Build line exactly when there is a build id", &
                         total_tests, total_failures)
        call assert_true(build_ok, "The Build line carries the identifier", total_tests, total_failures)

        ! The build line is printed for a development build and omitted for a release (empty id)
        call check_optional_line('v0.2.0-37-g6de9eb6', ' Build: v0.2.0-37-g6de9eb6', &
                                 "A development build id prints a Build line")
        call check_optional_line('', '', "An empty build id (a release) prints nothing")
    end subroutine test_banner

    !> Runs opensps_print_optional into a scratch file and compares what it wrote
    subroutine check_optional_line(value, expected, label)
        character(len=*), intent(in) :: value, expected, label
        character(len=200) :: line
        integer :: u, ios
        logical :: ok

        open (newunit=u, status='scratch', action='readwrite')
        call opensps_print_optional(' Build: ', value, u)
        rewind (u)
        line = ''
        read (u, '(A)', iostat=ios) line
        if (len(expected) == 0) then
            ok = (ios /= 0)             ! end of file: no line was written
        else
            ok = (ios == 0 .and. trim(line) == expected)
        end if
        close (u)
        call assert_true(ok, label, total_tests, total_failures)
    end subroutine check_optional_line

end module test_opensps_build_info_mod
