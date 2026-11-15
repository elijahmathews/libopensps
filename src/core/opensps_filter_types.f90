module opensps_filter_types
    !> @brief
    !> What the context stores about its active filters (names, content hashes, diagnostics), kept
    !> apart from `opensps_filters` so that the cache and the context state can hold it
    !> (`opensps_filters` itself needs the integration module, which needs the context).
    use opensps_precision, only: WP
    implicit none
    private

    public :: filter_info_t, filter_info_clear
    public :: RESAMPLE_LINEAR, RESAMPLE_CONSERVING
    public :: GRID_POLICY_ERROR, GRID_POLICY_WARN, GRID_POLICY_ALLOW

    integer, parameter :: RESAMPLE_LINEAR = 0       !< legacy point interpolation
    integer, parameter :: RESAMPLE_CONSERVING = 1   !< hat-function projection, conserves integral T dlambda
    integer, parameter :: GRID_POLICY_ERROR = 0     !< a filter beyond the grid, or dead, is an error
    integer, parameter :: GRID_POLICY_WARN = 1      !< reported (flags, coverage), not an error
    integer, parameter :: GRID_POLICY_ALLOW = 2     !< reported only; the legacy list's behaviour

    !> What was done to each filter of the active set (one entry per filter)
    type :: filter_info_t
        integer :: resample = RESAMPLE_LINEAR
        integer :: grid_policy = GRID_POLICY_ALLOW
        character(len=64) :: grid_hash = ''
        character(len=64), allocatable :: name(:), hash(:)
        real(WP), allocatable :: coverage(:)
        integer, allocatable :: n_fwhm(:), flags(:), detector(:)
    end type filter_info_t

contains

    subroutine filter_info_clear(info)
        type(filter_info_t), intent(inout) :: info
        if (allocated(info%name)) deallocate (info%name, info%hash, info%coverage, info%n_fwhm, info%flags, &
                                              info%detector)
        info%grid_hash = ''
    end subroutine filter_info_clear

end module opensps_filter_types
