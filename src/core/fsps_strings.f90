module fsps_strings
    !> @brief String manipulation utilities for FSPS.
    !>
    !> @details
    !> Provides helper functions for string handling that are not built into
    !> the Fortran standard library, such as case conversion.

    use fsps_precision, only: WP

    implicit none
    private

    public :: to_lower
    public :: to_string_int
    public :: to_string_real

    ! ---------------------------------------------------------------------
    ! Module constants
    ! ---------------------------------------------------------------------

contains

    !> @brief Converts a string to lowercase.
    !> @param[in] str The input string (e.g., "MIST").
    !> @return The lowercase version (e.g., "mist").
    pure function to_lower(str) result(res)
        character(len=*), intent(in) :: str
        character(len=len(str)) :: res
        integer :: i, code

        res = str
        do i = 1, len(str)
            code = iachar(str(i:i))
            ! ASCII 'A' is 65, 'Z' is 90. Add 32 to get lowercase.
            if (code >= 65 .and. code <= 90) then
                res(i:i) = achar(code + 32)
            end if
        end do
    end function to_lower

    !> @brief Converts an integer to a trimmed string.
    !> @details Useful for generating filenames or error messages without padding.
    pure function to_string_int(val) result(res)
        integer, intent(in) :: val
        character(len=32) :: res
        character(len=32) :: buffer

        write (buffer, '(I0)') val
        res = trim(buffer)
    end function to_string_int

    !> @brief Converts a real number to a string with specified precision.
    !> @details Uses standard output formatting.
    pure function to_string_real(val, fmt) result(res)
        real(WP), intent(in) :: val
        character(len=*), intent(in), optional :: fmt
        character(len=32) :: res
        character(len=32) :: buffer
        character(len=10) :: f

        if (present(fmt)) then
            f = fmt
        else
            f = '(G12.5)'
        end if

        write (buffer, f) val
        res = trim(adjustl(buffer))
    end function to_string_real

end module fsps_strings
