!> @file opensps_value_classes.f90
!> @brief Classes of equal columns: the grouping of models by the exact values of some of their parameters.
!>
!> A parameter that selects a precomputed structure (class G of opensps_batch_registry: the TP-AGB shifts, the smoothing
!> kernel) or whose derived quantity is expensive and shared (class W: the IMF weights) is served once per distinct value.
!> classify_columns finds the distinct columns of an array of parameter columns, with an optional integer key that must
!> match as well (the metallicity of a virtual model), a cap on the number of classes and a window of comparison:
!>  * cap: a batch with more distinct values than the structures it may hold gets overflow = .true. (the caller falls back
!>    or refuses; the classes found so far are valid);
!>  * window: only the last `window` classes are compared (0: all). For columns sorted by the key the equal ones are near,
!>    and the cost per column is bounded; a column not matched within the window starts a class of its own (the classes
!>    are then not minimal, but every member of a class is equal to its representative).
!> Equality is exact: a batch shares a value or it does not.
module opensps_value_classes
    use opensps_precision, only: WP
    implicit none
    private

    public :: classify_columns, count_distinct

contains

    !> The classes of the columns vals(:, idx(j)), j = 1 .. size(idx) (all columns in order if idx is absent), with the
    !> optional integer key(j) that must also match: cls(j) the class of column j (1 .. ncls), rep(c) the first column of
    !> class c (a position in idx), ncls the number of classes. At most cap classes (cap <= size(rep)); on overflow the
    !> classification stops, overflow is true and cls is valid up to the column before the one that did not fit.
    subroutine classify_columns(vals, cap, window, cls, rep, ncls, overflow, idx, key)
        real(WP), intent(in) :: vals(:,:)
        integer, intent(in) :: cap, window
        integer, intent(out) :: cls(:), rep(:), ncls
        logical, intent(out) :: overflow
        integer, intent(in), optional :: idx(:), key(:)
        integer :: j, n, r, c0, col, colr
        logical :: found

        n = size(cls)
        if (present(idx)) n = size(idx)
        ncls = 0
        overflow = .false.
        do j = 1, n
            col = j
            if (present(idx)) col = idx(j)
            found = .false.
            c0 = 1
            if (window > 0) c0 = max(1, ncls - window + 1)
            do r = c0, ncls
                colr = rep(r)
                if (present(idx)) colr = idx(rep(r))
                if (present(key)) then
                    if (key(j) /= key(rep(r))) cycle
                end if
                if (all(vals(:, col) == vals(:, colr))) then
                    cls(j) = r
                    found = .true.
                    exit
                end if
            end do
            if (found) cycle
            if (ncls == cap) then
                overflow = .true.
                return
            end if
            ncls = ncls + 1
            rep(ncls) = j
            cls(j) = ncls
        end do
    end subroutine classify_columns

    !> The number of distinct columns of vals(:, idx) (all columns if idx is absent), counted up to cap (a result of cap means
    !> at least cap)
    integer function count_distinct(vals, cap, idx) result(nd)
        real(WP), intent(in) :: vals(:,:)
        integer, intent(in) :: cap
        integer, intent(in), optional :: idx(:)
        integer, allocatable :: cls(:), rep(:)
        logical :: overflow
        integer :: n

        n = size(vals, 2)
        if (present(idx)) n = size(idx)
        allocate(cls(n), rep(max(cap, 1)))
        if (present(idx)) then
            call classify_columns(vals, cap, 0, cls, rep, nd, overflow, idx=idx)
        else
            call classify_columns(vals, cap, 0, cls, rep, nd, overflow)
        end if
    end function count_distinct

end module opensps_value_classes
