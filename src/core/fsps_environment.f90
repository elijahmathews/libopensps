module fsps_environment
    !> @brief
    !> Handles system environment interactions, path resolution, and resource cleanup
    !> for the FSPS library.
    !>
    !> @details
    !> This module replaces the legacy `sps_setup_utils`. It is responsible for:
    !> 1. Locating the FSPS data directory via environment variables or standard paths.
    !> 2. Setting up the output directory.
    !> 3. Tearing down the context and releasing cache resources.
    !> 4. Reporting runtime environment details (Version, OpenMP threads).

    use iso_fortran_env, only: error_unit, output_unit
    use fsps_context_types, only: fsps_context_t, fsps_context_state_destroy
    use fsps_cache, only: fsps_cache_release_setup
    use fsps_datalib, only: datalib_format_from_env, datalib_root_is_standard

    implicit none
    private

    public :: fsps_resolve_paths
    public :: fsps_cleanup
    public :: fsps_print_env_info

    ! ---------------------------------------------------------------------
    ! Module constants
    ! ---------------------------------------------------------------------

    !> @brief Hardcoded version string for the library
    character(len=*), parameter :: FSPS_VERSION = "3.2"

contains

    !> @brief Resolves the root paths for FSPS data and output.
    !>
    !> @details
    !> Searches for the FSPS data directory in the following order:
    !> 1. `OPENSPS_DATA_HOME` environment variable (preferred).
    !> 2. `SPS_HOME` environment variable (legacy FSPS).
    !> 3. `FSPS_DATA_HOME` environment variable.
    !> 4. `XDG_DATA_HOME/fsps` (Linux standard).
    !> 5. `~/.local/share/fsps` (User local).
    !> 6. `/usr/share/fsps` (System wide).
    !> 7. `/usr/local/share/fsps` (Local system wide).
    !>
    !> If found, populates `ctx%sps_home` and `ctx%data_home`.
    !> Also resolves `ctx%output_home` based on write permissions and standards.
    !>
    !> @param[inout] ctx The FSPS context to populate.
    subroutine fsps_resolve_paths(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        character(len=4096) :: env_buffer
        character(len=4096) :: candidate
        logical :: found
        integer :: stat_len, stat_status

        ctx%data_home = ''
        ctx%output_home = ''
        found = .false.

        ! ------------------------------------------------------------------------
        ! 1. SEARCH FOR DATA DIRECTORY
        ! ------------------------------------------------------------------------

        call get_environment_variable("OPENSPS_DATA_HOME", value=env_buffer, length=stat_len, status=stat_status)
        if (stat_status == 0 .and. stat_len > 0) then
            candidate = trim(env_buffer)
            if (check_data_exists(candidate)) then
                ctx%sps_home = trim(candidate)
                found = .true.
            end if
        end if

        ! Then the legacy variable SPS_HOME
        if (.not. found) call get_environment_variable("SPS_HOME", value=env_buffer, length=stat_len, &
                                                       status=stat_status)
        if (.not. found .and. stat_status == 0 .and. stat_len > 0) then
            candidate = trim(env_buffer)
            if (check_data_exists(candidate)) then
                ctx%sps_home = trim(candidate)
                found = .true.
            end if
        end if

        ! Check FSPS_DATA_HOME (Modern Variable)
        if (.not. found) then
            call get_environment_variable("FSPS_DATA_HOME", value=env_buffer, length=stat_len, status=stat_status)
            if (stat_status == 0 .and. stat_len > 0) then
                candidate = trim(env_buffer)
                if (check_data_exists(candidate)) then
                    ctx%sps_home = trim(candidate)
                    found = .true.
                end if
            end if
        end if

        ! Check XDG_DATA_HOME
        if (.not. found) then
            call get_environment_variable("XDG_DATA_HOME", value=env_buffer, length=stat_len, status=stat_status)
            if (stat_status == 0 .and. stat_len > 0) then
                candidate = trim(env_buffer)//'/fsps'
                if (check_data_exists(candidate)) then
                    ctx%sps_home = trim(candidate)
                    found = .true.
                end if
            end if
        end if

        ! Check HOME standard location
        if (.not. found) then
            call get_environment_variable("HOME", value=env_buffer, length=stat_len, status=stat_status)
            if (stat_status == 0 .and. stat_len > 0) then
                candidate = trim(env_buffer)//'/.local/share/fsps'
                if (check_data_exists(candidate)) then
                    ctx%sps_home = trim(candidate)
                    found = .true.
                end if
            end if
        end if

        ! Check System Paths
        if (.not. found) then
            candidate = '/usr/share/fsps'
            if (check_data_exists(candidate)) then
                ctx%sps_home = trim(candidate)
                found = .true.
            end if
        end if

        if (.not. found) then
            candidate = '/usr/local/share/fsps'
            if (check_data_exists(candidate)) then
                ctx%sps_home = trim(candidate)
                found = .true.
            end if
        end if

        ! ERROR: Not found
        if (.not. found) then
            write (error_unit, '(A)') '[FSPS_ENV] Error: FSPS data path not found.'
            write (error_unit, '(A)') '           Please set OPENSPS_DATA_HOME (or FSPS_DATA_HOME / SPS_HOME).'
            error stop 1
        end if

        ! Set Data Home
        ctx%data_home = trim(ctx%sps_home)//'/data'

        ! Library format: libraries with a manifest under <root>/libraries are read in the
        ! standard format unless FSPS_DATA_FORMAT=legacy (doc/DATA_FORMAT.md, Section 5)
        ctx%data_format = datalib_format_from_env()

        ! ------------------------------------------------------------------------
        ! 2. RESOLVE OUTPUT DIRECTORY
        ! ------------------------------------------------------------------------

        call get_environment_variable("FSPS_OUTPUT_HOME", value=env_buffer, length=stat_len, status=stat_status)

        if (stat_status == 0 .and. stat_len > 0) then
            ! Explicit override
            ctx%output_home = trim(env_buffer)
        else
            ! Logic: If installed in a system directory (e.g. /usr/), we cannot write there.
            ! Default to user home or current directory.
            if (index(ctx%sps_home, '/usr/') == 1) then
                call get_environment_variable("HOME", value=env_buffer, length=stat_len, status=stat_status)
                if (stat_status == 0 .and. stat_len > 0) then
                    ctx%output_home = trim(env_buffer)//'/.local/share/fsps'
                else
                    ctx%output_home = '.'
                end if
            else
                ! If local installation, output to root
                ctx%output_home = trim(ctx%sps_home)
            end if
        end if

    end subroutine fsps_resolve_paths

    !> @brief Prints a summary of the current FSPS runtime environment.
    !> @details Useful for debugging and verification during startup.
    subroutine fsps_print_env_info(ctx)
        type(fsps_context_t), intent(in) :: ctx

        write (output_unit, '(A)') '--------------------------------------------------'
        write (output_unit, '(A, A)') ' FSPS Version:  ', FSPS_VERSION
        write (output_unit, '(A, A)') ' Data Root:     ', trim(ctx%sps_home)
        write (output_unit, '(A, A)') ' Output Root:   ', trim(ctx%output_home)

        ! Note: OMP_GET_MAX_THREADS is intrinsic in many modern compilers,
        ! but strict F2008 standard compliance often requires the `omp_lib` module.
        ! We leave this simple print for now to avoid compilation complexity if OpenMP is off.
#ifdef _OPENMP
        write (output_unit, '(A)') ' OpenMP Status: Enabled'
#else
        write (output_unit, '(A)') ' OpenMP Status: Disabled (Serial)'
#endif
        write (output_unit, '(A)') '--------------------------------------------------'
    end subroutine fsps_print_env_info

    !> @brief Tears down the FSPS context and releases global cache resources.
    !>
    !> @details
    !> Disconnects the context from the shared cache and destroys any context-specific
    !> state. If this was the last context using the cache, the cache memory is freed.
    !>
    !> @param[inout] ctx The FSPS context to destroy.
    subroutine fsps_cleanup(ctx)
        type(fsps_context_t), intent(inout) :: ctx

        if (associated(ctx%setup_cache)) then
            call fsps_cache_release_setup(ctx%setup_cache)
            nullify (ctx%setup_cache)
        end if

        call fsps_context_state_destroy(ctx%state)

        ctx%initialized = .false.

    end subroutine fsps_cleanup

    ! ------------------------------------------------------------------------
    ! PRIVATE HELPERS
    ! ------------------------------------------------------------------------

    !> @brief Checks if the 'data' subdirectory exists within a given path.
    !> @param[in] path The root directory candidate.
    !> @return .true. if `path` is a standard data root (`opensps-data.json`, or the older `fsps-data.json`) or a legacy one
    !>         (`path/data/allfilters.dat` or `path/data/FILTER_LIST` exists).
    logical function check_data_exists(path)
        character(len=*), intent(in) :: path
        logical :: file_exists

        check_data_exists = datalib_root_is_standard(path)
        if (check_data_exists) return

        ! Check for critical data file to confirm valid directory
        inquire (file=trim(path)//'/data/allfilters.dat', exist=file_exists)

        if (.not. file_exists) then
            inquire (file=trim(path)//'/data/FILTER_LIST', exist=file_exists)
        end if

        check_data_exists = file_exists
    end function check_data_exists

end module fsps_environment
