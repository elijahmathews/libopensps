module opensps_context_filters
    !> @brief
    !> Replaces the filters of a set-up context (doc/FILTERS_PLAN.md): the curves are cleaned,
    !> resampled onto the context's wavelength grid, normalized, calibrated, and the zero points
    !> (Vega and Sun in every band) and pivot wavelengths recomputed. The new arrays belong to the
    !> context; the shared setup cache, and every other context using it, is untouched.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use opensps_precision, only: WP
    use opensps_context_types, only: opensps_context_t, opensps_context_state_release_filters, batch_grid_release
    use opensps_context, only: opensps_context_move_to_device, opensps_context_remove_from_device
    use opensps_filter_types, only: filter_info_t, filter_info_clear
    use opensps_filters, only: filter_set_t, filter_set_clear, filters_build, filter_calibrate_band, filter_pivot, &
                               filters_zero_points, RESAMPLE_LINEAR, RESAMPLE_CONSERVING, GRID_POLICY_ERROR, &
                               GRID_POLICY_ALLOW, FILTER_OK, FILTER_ERR_COVERAGE, FILTER_ERR_DEAD, FILTER_ERR_HASH
    use opensps_io, only: read_filter_library, use_standard_library
    implicit none
    private

    public :: opensps_context_set_filters, opensps_context_set_filters_library
    public :: FILTERS_OK, FILTERS_ERR_NOT_SETUP, FILTERS_ERR_ARGUMENT, FILTERS_ERR_CURVE, FILTERS_ERR_COVERAGE, &
              FILTERS_ERR_DEAD, FILTERS_ERR_HASH, FILTERS_ERR_LIBRARY

    !> Status codes (the C API returns them as they are)
    integer, parameter :: FILTERS_OK = 0
    integer, parameter :: FILTERS_ERR_ARGUMENT = 3     !< empty set, bad mode/policy/reference band
    integer, parameter :: FILTERS_ERR_NOT_SETUP = 301
    integer, parameter :: FILTERS_ERR_CURVE = 320      !< a curve is unusable (message names the filter)
    integer, parameter :: FILTERS_ERR_COVERAGE = 321   !< a filter lies (partly) beyond the model wavelength grid
    integer, parameter :: FILTERS_ERR_DEAD = 322       !< a filter has no throughput on the grid
    integer, parameter :: FILTERS_ERR_HASH = 323       !< a library's stored hash does not match its curve
    integer, parameter :: FILTERS_ERR_LIBRARY = 324    !< no such filter library

contains

    !> @brief Installs `set` as the context's filters.
    !>
    !> @param[inout] ctx            a context that has been set up
    !> @param[in]    set            the filters (wavelengths in Angstrom)
    !> @param[in]    resample       RESAMPLE_CONSERVING (default) or RESAMPLE_LINEAR (the legacy point interpolation)
    !> @param[in]    grid_policy    GRID_POLICY_ERROR (default), _WARN or _ALLOW
    !> @param[in]    vega_ref_band  Vega system anchor (see `vega_ref_band` of the state): 0 (default) Vega = 0 in
    !>                              every band; k > 0 the legacy convention anchored on band k
    !> @param[in]    legacy_order   zero points from the uncalibrated bands, as for the legacy list (default .false.)
    !> @param[out]   status         FILTERS_OK or a FILTERS_ERR_* code
    !> @param[out]   message        what went wrong (the failing filter), blank on success
    subroutine opensps_context_set_filters(ctx, set, status, message, resample, grid_policy, vega_ref_band, &
                                           legacy_order)
        type(opensps_context_t), intent(inout) :: ctx
        type(filter_set_t), intent(in) :: set
        integer, intent(out) :: status
        character(len=*), intent(out) :: message
        integer, intent(in), optional :: resample, grid_policy, vega_ref_band
        logical, intent(in), optional :: legacy_order

        integer :: mode, policy, vref, st, bad, i, n, nspec
        logical :: legacy
        real(WP), allocatable :: bands(:, :), magvega(:), magsun(:), leff(:)
        type(filter_info_t) :: info
        logical :: was_on_device
        character(len=160) :: prefix, detail

        status = FILTERS_OK
        message = ''
        mode = RESAMPLE_CONSERVING
        policy = GRID_POLICY_ERROR
        vref = 0
        legacy = .false.
        if (present(resample)) mode = resample
        if (present(grid_policy)) policy = grid_policy
        if (present(vega_ref_band)) vref = vega_ref_band
        if (present(legacy_order)) legacy = legacy_order

        if (.not. ctx%initialized .or. .not. associated(ctx%state%spec_lambda) .or. &
            .not. associated(ctx%state%vega_spec) .or. .not. associated(ctx%state%sun_spec)) then
            status = FILTERS_ERR_NOT_SETUP
            message = 'set_filters: the context has not been set up'
            return
        end if
        n = set%n
        if (n < 1 .or. (mode /= RESAMPLE_LINEAR .and. mode /= RESAMPLE_CONSERVING) .or. policy < 0 .or. policy > 2 &
            .or. vref < 0 .or. vref > n) then
            status = FILTERS_ERR_ARGUMENT
            message = 'set_filters: empty set, or resample, grid policy or Vega reference band out of range'
            return
        end if
        nspec = size(ctx%state%spec_lambda)

        ! Everything is computed in new arrays first: a failure leaves the context as it was
        allocate (bands(nspec, n), magvega(n), magsun(n), leff(n))
        call filters_build(set, ctx%state%spec_lambda, mode, policy, .true., bands, info, st, bad)
        if (st /= FILTER_OK) then
            write (prefix, '(A,I0,A)') 'set_filters: filter ', bad, ' ('
            prefix = trim(prefix) // trim(label(set, bad)) // '): '
            select case (st)
            case (FILTER_ERR_COVERAGE)
                status = FILTERS_ERR_COVERAGE
                write (detail, '(A,F7.4,A)') 'only ', info%coverage(bad), &
                    ' of its response lies within the model wavelength range'
            case (FILTER_ERR_DEAD)
                status = FILTERS_ERR_DEAD
                detail = 'no throughput on the model wavelength grid'
            case (FILTER_ERR_HASH)
                status = FILTERS_ERR_HASH
                detail = 'the content hash differs from the one stored with the curve'
            case default
                status = FILTERS_ERR_CURVE
                write (detail, '(A,I0,A)') 'unusable curve (code ', st, ')'
            end select
            message = trim(prefix) // ' ' // trim(detail)
            return
        end if

        if (legacy) call filters_zero_points(ctx%state%spec_lambda, ctx%state%vega_spec, ctx%state%sun_spec, &
                                             bands, magvega, magsun)
        do i = 1, n
            if (set%has_cal(i)) call filter_calibrate_band(ctx%state%spec_lambda, bands(:, i), set%cal_lref(i), &
                                                           set%cal_expo(i))
        end do
        if (.not. legacy) call filters_zero_points(ctx%state%spec_lambda, ctx%state%vega_spec, ctx%state%sun_spec, &
                                                   bands, magvega, magsun)
        do i = 1, n
            leff(i) = filter_pivot(ctx%state%spec_lambda, bands(:, i))
        end do
        if (ctx%compute_vega_mags_val == 1) call sun_to_vega_system(magsun, magvega, vref)

        ! Install. The device copies of the old arrays go first, the batch's per-band tables with them.
        was_on_device = ctx%on_device
        if (was_on_device) call opensps_context_remove_from_device(ctx)
        call batch_grid_release(ctx%state%batch)
        if (allocated(ctx%state%last_maggies)) deallocate (ctx%state%last_maggies)   ! of the old bands
        call opensps_context_state_release_filters(ctx%state)
        allocate (ctx%state%bands(nspec, n), ctx%state%magvega(n), ctx%state%magsun(n), ctx%state%filter_leff(n))
        allocate (ctx%state%filter_info)
        ctx%state%bands = bands
        ctx%state%magvega = magvega
        ctx%state%magsun = magsun
        ctx%state%filter_leff = leff
        ctx%state%filter_info = info
        ctx%state%filters_owned = .true.
        ctx%state%nbands = n
        ctx%state%vega_ref_band = vref
        if (allocated(ctx%pset%mag_compute)) then
            deallocate (ctx%pset%mag_compute)
        end if
        allocate (ctx%pset%mag_compute(n))
        ctx%pset%mag_compute = 1
        if (was_on_device) call opensps_context_move_to_device(ctx)
        call filter_info_clear(info)
    end subroutine opensps_context_set_filters

    !> @brief Installs the filter library `filters/<name>` (any standard-format library of that kind) as the
    !> context's filters; detector types, calibration and stored hashes come from its manifest.
    subroutine opensps_context_set_filters_library(ctx, name, status, message, resample, grid_policy, vega_ref_band)
        type(opensps_context_t), intent(inout) :: ctx
        character(len=*), intent(in) :: name
        integer, intent(out) :: status
        character(len=*), intent(out) :: message
        integer, intent(in), optional :: resample, grid_policy, vega_ref_band

        type(filter_set_t) :: set

        status = FILTERS_OK
        message = ''
        if (.not. use_standard_library(ctx, 'filters', trim(name))) then
            status = FILTERS_ERR_LIBRARY
            message = 'set_filters_library: no standard-format filter library "' // trim(name) // '"'
            return
        end if
        call read_filter_library(ctx, trim(name), set)
        ! The library under the name of the default list keeps that list's zero-point order
        call opensps_context_set_filters(ctx, set, status, message, resample, grid_policy, vega_ref_band, &
                                         legacy_order=(trim(name) == 'allfilters'))
        call filter_set_clear(set)
    end subroutine opensps_context_set_filters_library

    function label(set, i) result(s)
        type(filter_set_t), intent(in) :: set
        integer, intent(in) :: i
        character(len=64) :: s
        s = set%name(i)
    end function label

    !> The Sun's magnitudes in the Vega system, as `convert_to_vega_system` does for the setup
    !> default (anchored on band `vref`; vref = 0: m_Vega = 0 in every band)
    subroutine sun_to_vega_system(magsun, magvega, vref)
        real(WP), intent(inout) :: magsun(:)
        real(WP), intent(in) :: magvega(:)
        integer, intent(in) :: vref
        integer :: i

        do i = 1, size(magsun)
            if (magsun(i) == 99.0_wp .or. magvega(i) == 99.0_wp) cycle
            if (vref > 0) then
                magsun(i) = (magsun(i) - magsun(vref)) - (magvega(i) - magvega(vref)) + magsun(vref)
            else
                magsun(i) = magsun(i) - magvega(i)
            end if
        end do
    end subroutine sun_to_vega_system

end module opensps_context_filters
