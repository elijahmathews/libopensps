!> \file opensps_trace.f90
!> \brief Fortran bindings for platform-neutral range tracing (ROCTX / NVTX).
module opensps_trace
    use, intrinsic :: iso_c_binding, only: c_char, c_null_char
    implicit none
    private

    public :: opensps_trace_push, opensps_trace_pop

    interface
        subroutine opensps_trace_range_push_c(name) bind(C, name="opensps_trace_range_push")
            import :: c_char
            character(kind=c_char), intent(in) :: name(*)
        end subroutine opensps_trace_range_push_c

        subroutine opensps_trace_range_pop_c() bind(C, name="opensps_trace_range_pop")
        end subroutine opensps_trace_range_pop_c
    end interface

contains

    subroutine opensps_trace_push(name)
        character(len=*), intent(in) :: name
#if defined(OPENSPS_ENABLE_TRACE)
        character(kind=c_char, len=len_trim(name) + 1) :: c_str
        c_str = trim(name) // c_null_char
        call opensps_trace_range_push_c(c_str)
#endif
    end subroutine opensps_trace_push

    subroutine opensps_trace_pop()
#if defined(OPENSPS_ENABLE_TRACE)
        call opensps_trace_range_pop_c()
#endif
    end subroutine opensps_trace_pop

end module opensps_trace
