!> Stand-ins for the few ieee_arithmetic entities the project uses (NaN, -Inf, is_nan, is_finite).
!> LLVM flang 22 rejects `use ieee_arithmetic` when it offloads to NVIDIA (the module declares REAL(10),
!> which the NVPTX target lacks); these need no compiler support and work in device code.
module opensps_ieee_compat
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
    implicit none
    private
    public :: ieee_value, ieee_is_nan, ieee_is_finite, ieee_quiet_nan, ieee_negative_inf, ieee_class, ieee_class_type, operator(==)
    type :: ieee_class_type
        integer :: v = 0
    end type
    type(ieee_class_type), parameter :: ieee_quiet_nan = ieee_class_type(1), ieee_negative_inf = ieee_class_type(2), ieee_other = ieee_class_type(0)
    interface operator(==)
        module procedure cls_eq
    end interface
    real(real64), parameter :: NAN64 = transfer(int(z'7FF8000000000000', int64), 0.0_real64)
    real(real64), parameter :: NINF64 = transfer(int(z'FFF0000000000000', int64), 0.0_real64)
    real(real32), parameter :: NAN32 = transfer(int(z'7FC00000', int32), 0.0_real32)
    real(real32), parameter :: NINF32 = transfer(int(z'FF800000', int32), 0.0_real32)
    interface ieee_value
        module procedure value64, value32
    end interface
    interface ieee_is_nan
        module procedure isnan64, isnan32
    end interface
    interface ieee_is_finite
        module procedure isfin64, isfin32
    end interface
contains
    elemental logical function cls_eq(a, b)
        type(ieee_class_type), intent(in) :: a, b
        cls_eq = a%v == b%v
    end function
    elemental function ieee_class(x) result(c)
        real(real64), intent(in) :: x
        type(ieee_class_type) :: c
        if (x /= x) then
            c = ieee_quiet_nan
        else if (x < -huge(x)) then
            c = ieee_negative_inf
        else
            c = ieee_other
        end if
    end function
    pure function value64(x, cls) result(v)
        !$omp declare target
        real(real64), intent(in) :: x
        type(ieee_class_type), intent(in) :: cls
        real(real64) :: v
        if (cls%v == 2) then
            v = NINF64
        else
            v = NAN64
        end if
    end function
    pure function value32(x, cls) result(v)
        !$omp declare target
        real(real32), intent(in) :: x
        type(ieee_class_type), intent(in) :: cls
        real(real32) :: v
        if (cls%v == 2) then
            v = NINF32
        else
            v = NAN32
        end if
    end function
    elemental logical function isnan64(x)
        !$omp declare target
        real(real64), intent(in) :: x
        isnan64 = x /= x
    end function
    elemental logical function isnan32(x)
        !$omp declare target
        real(real32), intent(in) :: x
        isnan32 = x /= x
    end function
    elemental logical function isfin64(x)
        !$omp declare target
        real(real64), intent(in) :: x
        isfin64 = x == x .and. abs(x) <= huge(x)
    end function
    elemental logical function isfin32(x)
        !$omp declare target
        real(real32), intent(in) :: x
        isfin32 = x == x .and. abs(x) <= huge(x)
    end function
end module opensps_ieee_compat
