program test_runner

    !> @brief
    !> Regression test runner for end-to-end FSPS numerical validation.
    !>
    !> @details
    !> This program validates FSPS outputs against a binary reference file that was
    !> produced from a known-good run. It executes both SSP and CSP workflows using
    !> the currently built code, then compares scalar and array outputs with a
    !> relative tolerance.
    !>
    !> The execution flow is:
    !> 1. Parse CLI options and environment-controlled tolerances.
    !> 2. Initialize and configure an FSPS context.
    !> 3. Read reference dimensions and payload from disk.
    !> 4. Regenerate SSP and CSP outputs from the same setup.
    !> 5. Compare all validated fields and report pass/fail.
    !>
    !> CLI usage:
    !> - `./test_runner [--isoc type] [--spec type] [--dust type] ref_file.bin`
    !>
    !> Environment controls:
    !> - `FSPS_TEST_RTOL`: relative tolerance override.
    !> - `FSPS_TEST_INDX_ATOL`: absolute tolerance floor for spectral indices (default 1e-30, i.e.
    !>   purely relative); set for kernel-precision (FP32) builds, where weak indices near zero
    !>   differ from the FP64 references by ~1e-5 in absolute terms.
    !> - `FSPS_TEST_VERBOSE`: enables detailed mismatch output.
    !> - `FSPS_TEST_MAXFAIL`: caps mismatch lines unless verbose mode is enabled.
    !>
    !> Exit status:
    !> - 0 on success (all comparisons within tolerance).
    !> - 1 on failure (setup/read errors or numerical mismatches).

    use, intrinsic :: ieee_arithmetic
    use fsps_precision, only: WP
    use fsps_constants, only: NEMLINE
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
    use fsps_csp, only: compute_csp_scenario
    use fsps_context_types, only: fsps_context_t
    use fsps_context, only: fsps_context_set_pset, fsps_context_move_to_device, &
                            fsps_context_remove_from_device, fsps_context_update_ssp_basis
    use fsps_ssp, only: generate_ssp_grid
    implicit none

    ! Exit codes
    integer, parameter :: EXIT_SUCCESS = 0
    integer, parameter :: EXIT_FAILURE = 1

    ! Default tolerance (can be overridden by env var FSPS_TEST_RTOL)
    real(WP), parameter :: DEFAULT_RTOL = 1.0e-5_wp

    ! Test arrays (allocatable)
    ! Reference data (read from disk)
    real(WP), allocatable, dimension(:, :) :: ref_spec_ssp
    real(WP), allocatable, dimension(:) :: ref_mass_ssp, ref_lbol_ssp
    type(COMPSPOUT), allocatable, dimension(:) :: ref_ocompsp

    ! New data (computed on the fly)
    real(WP), allocatable, dimension(:, :) :: new_spec_ssp_ctx
    real(WP), allocatable, dimension(:, :) :: new_spec_ssp_cmp
    real(WP), allocatable, dimension(:, :, :) :: new_spec_ssp3
    real(WP), allocatable, dimension(:, :) :: new_mass_ssp2, new_lbol_ssp2
    real(WP), allocatable, dimension(:) :: new_mass_ssp, new_lbol_ssp
    type(COMPSPOUT), allocatable, dimension(:) :: new_results

    ! Control variables
    type(fsps_context_t) :: ctx
    type(PARAMS) :: pset
    integer :: i, unit_in, status, arg_count
    character(len=1024) :: filename_in, env_buffer, arg_val
    character(len=20) :: isoc_arg, spec_arg, dust_arg
    logical :: isoc_set, spec_set, dust_set

    ! Dimensions read from file
    integer :: file_nspec, file_ntfull, file_nbands
    integer :: nspec_ctx, ntfull_ctx, nbands_ctx, nt_ctx, nindx_ctx

    ! Comparison stats
    real(WP) :: rtol
    real(WP) :: indx_atol = 1.0e-30_wp
    logical :: test_passed
    integer :: nfail
    logical :: verbose_output
    logical :: ref_exists
    integer :: max_fail_print, fail_printed
    logical :: fail_suppression_noted

    ! ------------------------------------------------------------------------
    ! PROGRAM INITIALIZATION
    ! ------------------------------------------------------------------------

    ! Configuration
    unit_in = 40

    call ensure_output_dir('OUTPUTS/')

    test_passed = .true.
    nfail = 0
    verbose_output = .false.
    max_fail_print = 20
    fail_printed = 0
    fail_suppression_noted = .false.
    isoc_set = .false.
    spec_set = .false.
    dust_set = .false.
    isoc_arg = 'mist'
    spec_arg = 'miles'
    dust_arg = 'DL07'
    filename_in = ''

    write(*, *) '========================================='
    write(*, *) 'FSPS TEST RUNNER'
    write(*, *) '========================================='

    ! ------------------------------------------------------------------------
    ! 1) INPUT ARGUMENTS AND ENVIRONMENT OPTIONS
    ! ------------------------------------------------------------------------

    ! Parse command line arguments
    arg_count = command_argument_count()
    i = 1
    do while (i <= arg_count)
        call get_command_argument(i, arg_val, status=status)
        if (status /= 0) exit

        if (trim(arg_val) == '--isoc') then
            i = i + 1
            if (i <= arg_count) then
                call get_command_argument(i, isoc_arg, status=status)
                isoc_set = .true.
            else
                write(*, *) 'ERROR: --isoc requires an argument'
                stop EXIT_FAILURE
            end if
        else if (trim(arg_val) == '--spec') then
            i = i + 1
            if (i <= arg_count) then
                call get_command_argument(i, spec_arg, status=status)
                spec_set = .true.
            else
                write(*, *) 'ERROR: --spec requires an argument'
                stop EXIT_FAILURE
            end if
        else if (trim(arg_val) == '--dust') then
            i = i + 1
            if (i <= arg_count) then
                call get_command_argument(i, dust_arg, status=status)
                dust_set = .true.
            else
                write(*, *) 'ERROR: --dust requires an argument'
                stop EXIT_FAILURE
            end if
        else
            ! Assume it's the filename if it doesn't start with --
            ! Or if we haven't found one yet.
            if (len_trim(filename_in) == 0) then
                filename_in = trim(arg_val)
            end if
        end if
        i = i + 1
    end do

    if (len_trim(filename_in) == 0) then
        write(*, *) 'ERROR: Must provide reference filename as argument.'
        write(*, *) 'Usage: ./test_runner [--isoc type] [--spec type] [--dust type] tests/data/sps_ref_XXX.bin'
        stop EXIT_FAILURE
    end if

    ! Without its reference (data release library references/sps, or tests/data) the test is
    ! skipped, not failed: exit status 77 is meson's SKIP
    inquire(file=trim(filename_in), exist=ref_exists)
    if (.not. ref_exists) then
        write(*, '(A)') 'SKIP: regression reference ' // trim(filename_in) // ' not found (data release library ' // &
            'references/sps; set OPENSPS_DATA_HOME or -Ddata_home)'
        stop 77
    end if

    ! Read tolerance from environment variable
    call get_environment_variable('FSPS_TEST_RTOL', value=env_buffer, status=status)
    if (status == 0) then
        read(env_buffer, *) rtol
        write(*, *) 'Using RTOL from environment: ', rtol
    else
        rtol = DEFAULT_RTOL
        write(*, *) 'Using default RTOL: ', rtol
    end if
    call get_environment_variable('FSPS_TEST_INDX_ATOL', value=env_buffer, status=status)
    if (status == 0) then
        read(env_buffer, *) indx_atol
        write(*, *) 'Using index ATOL from environment: ', indx_atol
    end if

    ! Optional verbose output
    call get_environment_variable('FSPS_TEST_VERBOSE', value=env_buffer, status=status)
    if (status == 0) then
        if (len_trim(env_buffer) > 0) then
            select case (env_buffer(1:1))
            case ('1', 't', 'T', 'y', 'Y')
                verbose_output = .true.
            end select
        end if
    end if

    ! Optional maximum printed failures
    call get_environment_variable('FSPS_TEST_MAXFAIL', value=env_buffer, status=status)
    if (status == 0) then
        read(env_buffer, *, iostat=status) max_fail_print
        if (status /= 0) max_fail_print = 20
    end if

    ! ------------------------------------------------------------------------
    ! 2) FSPS SETUP AND DIMENSION DISCOVERY
    ! ------------------------------------------------------------------------

    ! Initialize FSPS and check dimensions
    ! Note: We must initialize FSPS before allocating, but we must read the
    ! file header before we know if dimensions match.

    call fsps_create(ctx)
    ctx%imf_type_val = 1
    pset%zmet = 10

    write(*, *) 'Initializing FSPS...'
    ! Always provide defaults that match the reference generator unless overridden
    call fsps_setup(ctx, pset%zmet, isoc_type_in=trim(isoc_arg), spec_type_in=trim(spec_arg), &
                    dust_type_in=trim(dust_arg))

    write(*, *) 'Moving FSPS Context to Device...'
    call fsps_context_move_to_device(ctx)

    if (verbose_output) call dump_state('AFTER fsps_setup', ctx, pset)

    nspec_ctx = ctx%state%nspec
    ntfull_ctx = ctx%state%ntfull
    nbands_ctx = ctx%state%nbands
    nt_ctx = ctx%state%nt
    nindx_ctx = ctx%state%nindx

    ! Global dimensions are synchronized via fsps_context_apply_globals

    ! Allocate pset allocatable components
    if (allocated(pset%mag_compute)) deallocate(pset%mag_compute)
    allocate(pset%mag_compute(nbands_ctx))
    pset%mag_compute = 1

    if (allocated(pset%ssp_gen_age)) deallocate(pset%ssp_gen_age)
    allocate(pset%ssp_gen_age(nt_ctx))
    pset%ssp_gen_age = 1

    ! Open the reference file
    open(unit=unit_in, file=trim(filename_in), status='OLD', &
         form='UNFORMATTED', access='STREAM', iostat=status)
    if (status /= 0) then
        write(*, *) 'ERROR: Could not open file: ', trim(filename_in)
        stop EXIT_FAILURE
    end if

    ! Read Header
    read(unit_in) file_nspec
    read(unit_in) file_ntfull
    read(unit_in) file_nbands

    write(*, *) 'Reference Dimensions: nspec=', file_nspec, ' nt=', file_ntfull
    write(*, *) 'Compiled Dimensions:  nspec=', nspec_ctx, ' nt=', ntfull_ctx

    ! Strict dimension check
    if (file_nspec /= nspec_ctx .or. file_ntfull /= ntfull_ctx) then
        write(*, *) 'FATAL: Binary dimensions do not match compiled FSPS dimensions.'
        write(*, *) 'Ensure you are running the test with the same flags/arguments used to generate the data.'
        stop EXIT_FAILURE
    end if

    ! ------------------------------------------------------------------------
    ! 3) REFERENCE DATA LOAD
    ! ------------------------------------------------------------------------

    ! Allocate and read reference data
    allocate(ref_spec_ssp(nspec_ctx, ntfull_ctx))
    allocate(ref_mass_ssp(ntfull_ctx))
    allocate(ref_lbol_ssp(ntfull_ctx))
    allocate(ref_ocompsp(ntfull_ctx))

    allocate(new_spec_ssp_ctx(nspec_ctx, ntfull_ctx))
    allocate(new_spec_ssp_cmp(nspec_ctx, ntfull_ctx))
    allocate(new_mass_ssp(ntfull_ctx))
    allocate(new_lbol_ssp(ntfull_ctx))
    allocate(new_mass_ssp2(ntfull_ctx, 1))
    allocate(new_lbol_ssp2(ntfull_ctx, 1))
    allocate(new_spec_ssp3(nspec_ctx, ntfull_ctx, 1))
    ! new_results allocated by compute_csp_scenario

    write(*, *) 'Reading SSP reference data...'
    read(unit_in) ref_mass_ssp
    read(unit_in) ref_lbol_ssp
    read(unit_in) ref_spec_ssp

    write(*, *) 'Reading CSP reference data...'
    do i = 1, ntfull_ctx
        ! Allocate components of derived type before reading
        allocate(ref_ocompsp(i)%mags(nbands_ctx))
        allocate(ref_ocompsp(i)%spec(nspec_ctx))
        allocate(ref_ocompsp(i)%indx(nindx_ctx))
        allocate(ref_ocompsp(i)%emlines(NEMLINE))

        read(unit_in) ref_ocompsp(i)%age
        read(unit_in) ref_ocompsp(i)%mass_csp
        read(unit_in) ref_ocompsp(i)%lbol_csp
        read(unit_in) ref_ocompsp(i)%sfr
        read(unit_in) ref_ocompsp(i)%mdust
        read(unit_in) ref_ocompsp(i)%mformed
        read(unit_in) ref_ocompsp(i)%mags
        read(unit_in) ref_ocompsp(i)%spec
        read(unit_in) ref_ocompsp(i)%indx
        read(unit_in) ref_ocompsp(i)%emlines
    end do
    close(unit_in)

    ! ------------------------------------------------------------------------
    ! 4) REGENERATE CURRENT OUTPUTS (SSP + CSP)
    ! ------------------------------------------------------------------------

    ! Generate new data
    write(*, *) 'Generating new SSP data...'
    pset%sfh = 0
    pset%const = 0.0_wp
    pset%zred = 0.0_wp
    pset%dust1 = 0.0_wp
    pset%dust2 = 0.0_wp
    ctx%add_neb_emission_val = 1
    call fsps_context_set_pset(ctx, pset)
    if (verbose_output) call dump_state('BEFORE generate_ssp_grid (SSP)', ctx, pset)
    !$acc update device(ctx%state%mini_isoc, ctx%state%mact_isoc, ctx%state%logl_isoc, ctx%state%logt_isoc)
    !$acc update device(ctx%state%logg_isoc, ctx%state%phase_isoc, ctx%state%ffco_isoc, ctx%state%lmdot_isoc)
    !$acc update device(ctx%state%nmass_isoc, ctx%state%timestep_isoc)
    !$acc data copyin(pset) copy(new_mass_ssp, new_lbol_ssp, new_spec_ssp_ctx)
    !$acc update device(pset)
    ctx%add_neb_emission_val = 1
    !$acc update device(ctx%add_neb_emission_val)
    call generate_ssp_grid(ctx, pset, new_mass_ssp, new_lbol_ssp, new_spec_ssp_ctx)
    !$acc update self(new_mass_ssp, new_lbol_ssp, new_spec_ssp_ctx)
    !$acc end data
    if (verbose_output) call dump_ssp_summary('AFTER generate_ssp_grid (SSP)', ctx, new_mass_ssp, new_lbol_ssp)
    new_spec_ssp_cmp = new_spec_ssp_ctx

    write(*, *) 'Generating new CSP data...'
    pset%sfh = 1
    pset%tau = 2.0_wp
    pset%dust1 = 1.0_wp
    pset%dust2 = 0.3_wp
    call fsps_context_set_pset(ctx, pset)
    if (verbose_output) call dump_state('BEFORE generate_ssp_grid (CSP)', ctx, pset)
    !$acc update device(ctx%state%mini_isoc, ctx%state%mact_isoc, ctx%state%logl_isoc, ctx%state%logt_isoc)
    !$acc update device(ctx%state%logg_isoc, ctx%state%phase_isoc, ctx%state%ffco_isoc, ctx%state%lmdot_isoc)
    !$acc update device(ctx%state%nmass_isoc, ctx%state%timestep_isoc)
    !$acc data copyin(pset) copy(new_mass_ssp, new_lbol_ssp, new_spec_ssp_ctx)
    !$acc update device(pset)
    ctx%add_neb_emission_val = 1
    !$acc update device(ctx%add_neb_emission_val)
    call generate_ssp_grid(ctx, pset, new_mass_ssp, new_lbol_ssp, new_spec_ssp_ctx)
    !$acc update self(new_mass_ssp, new_lbol_ssp, new_spec_ssp_ctx)
    !$acc end data
    if (verbose_output) call dump_ssp_summary('AFTER generate_ssp_grid (CSP)', ctx, new_mass_ssp, new_lbol_ssp)

    new_mass_ssp2(:, 1) = new_mass_ssp
    new_lbol_ssp2(:, 1) = new_lbol_ssp
    ! Match COMPSP input layout: (nspec, ntfull, nzin)
    new_spec_ssp3(:, :, 1) = new_spec_ssp_ctx

    if (verbose_output) call dump_state('BEFORE compute_csp_scenario', ctx, pset)
    call fsps_context_update_ssp_basis(ctx, new_spec_ssp3, new_mass_ssp2, new_lbol_ssp2, 1)

    !$acc data copyin(pset) copy(new_spec_ssp_ctx)
    !$acc update device(pset)
    ctx%add_neb_emission_val = 1
    !$acc update device(ctx%add_neb_emission_val)
    call compute_csp_scenario(ctx, pset, 1, new_results)
    !$acc update self(new_spec_ssp_ctx)
    !$acc end data
    if (verbose_output) call dump_csp_summary('AFTER compute_csp_scenario', new_results)

    ! ------------------------------------------------------------------------
    ! 5) NUMERICAL COMPARISON AGAINST REFERENCE
    ! ------------------------------------------------------------------------

    ! Compare results
    write(*, *) 'Verifying results (RTOL = ', rtol, ')...'

    ! Helper internal subroutine to check arrays
    call check_array_2d('SSP Spectra', ref_spec_ssp, new_spec_ssp_cmp, nspec_ctx, ntfull_ctx)
    call check_array_1d('SSP Mass', ref_mass_ssp, new_mass_ssp, ntfull_ctx)
    call check_array_1d('SSP Lbol', ref_lbol_ssp, new_lbol_ssp, ntfull_ctx)

    ! Check CSP structure components manually
    do i = 1, ntfull_ctx
        ! Check Scalars
        call check_val('CSP Lbol', i, ref_ocompsp(i)%lbol_csp, new_results(i)%lbol_csp, atol=1.0e-12_wp)
        call check_val('CSP Mass', i, ref_ocompsp(i)%mass_csp, new_results(i)%mass_csp)
        call check_val('CSP SFR', i, ref_ocompsp(i)%sfr, new_results(i)%sfr)
        call check_val('CSP Dust Mass', i, ref_ocompsp(i)%mdust, new_results(i)%mdust)
        call check_val('CSP Mass Formed', i, ref_ocompsp(i)%mformed, new_results(i)%mformed)

        ! Check Arrays for EVERY time step
        call check_mags_1d('CSP Mags (flux)', ref_ocompsp(i)%mags, new_results(i)%mags, nbands_ctx)

        call check_array_1d('CSP Indx', ref_ocompsp(i)%indx, new_results(i)%indx, nindx_ctx, atol=indx_atol)
        call check_array_1d('CSP Emlines', ref_ocompsp(i)%emlines, new_results(i)%emlines, NEMLINE)

        ! Check Spectrum
        call check_array_1d('CSP Spec', ref_ocompsp(i)%spec, new_results(i)%spec, nspec_ctx)
    end do

    ! Report results
    write(*, *) '--------------------------------------------------'

    if (allocated(new_results)) deallocate(new_results)
    if (allocated(ref_ocompsp)) deallocate(ref_ocompsp)

    write(*, *) 'Removing FSPS Context from Device...'
    call fsps_context_remove_from_device(ctx)
    call fsps_destroy(ctx)

    write(*, *) 'Total failures:', nfail
    if (test_passed) then
        write(*, *) 'TEST RESULT: PASS'
        stop EXIT_SUCCESS
    else
        write(*, *) 'TEST RESULT: FAIL'
        stop EXIT_FAILURE
    end if

contains

    !> @brief
    !> Compare two 2D real arrays element-wise using relative tolerance.
    !>
    !> @details
    !> Tracks maximum absolute and relative differences, checks for NaNs, and
    !> increments global failure counters when any element exceeds the dynamic
    !> threshold `max(abs(ref) * rtol, 1.0e-30_wp)`.
    !>
    !> @param[in] label Label printed in diagnostics.
    !> @param[in] ref   Reference array.
    !> @param[in] new   Newly generated array.
    !> @param[in] d1    First dimension length.
    !> @param[in] d2    Second dimension length.
    subroutine check_array_2d(label, ref, new, d1, d2)
        character(*), intent(in) :: label
        integer, intent(in) :: d1, d2
        real(WP), dimension(d1, d2), intent(in) :: ref, new
        real(WP) :: delta, threshold, max_delta, max_rel, ref_val
        integer :: j, k, mj, mk

        max_delta = 0.0_wp
        max_rel = 0.0_wp
        mj = 1
        mk = 1

        do k = 1, d2
            do j = 1, d1
                if (ieee_is_nan(ref(j, k)) .or. ieee_is_nan(new(j, k))) then
                    write(*, *) 'FAIL: ', label, ' contains NaN at index (', j, ',', k, ')'
                    test_passed = .false.
                    return
                end if

                ref_val = ref(j, k)
                delta = abs(ref_val - new(j, k))
                if (abs(ref_val) > 0.0_wp) then
                    max_rel = max(max_rel, delta / abs(ref_val))
                end if
                if (delta > max_delta) then
                    max_delta = delta
                    mj = j
                    mk = k
                end if
                ! If ref is close to zero, use absolute tolerance, else relative
                threshold = max(abs(ref_val) * rtol, 1.0e-30_wp)
                if (delta > threshold) then
                    test_passed = .false.
                    nfail = nfail + 1
                    if (verbose_output .or. fail_printed < max_fail_print) then
                        write(*, *) 'FAIL: ', label, ' mismatch at index (', j, ',', k, ')'
                        write(*, *) '  Ref:', ref_val, ' New:', new(j, k), ' Diff:', delta
                        fail_printed = fail_printed + 1
                    else if (.not. fail_suppression_noted) then
                        write(*, *) 'NOTE: Further failure details suppressed (set FSPS_TEST_VERBOSE=1 to expand).'
                        fail_suppression_noted = .true.
                    end if
                end if
            end do
        end do
        write(*, *) 'SUMMARY: ', label, ' max abs diff=', max_delta, ' at (', mj, ',', mk, ')', &
                    ' max rel diff=', max_rel
    end subroutine check_array_2d

    !> @brief
    !> Compare two 1D real arrays element-wise using relative tolerance.
    !>
    !> @details
    !> Mirrors `check_array_2d` behavior for vector data and records index-wise
    !> diagnostics for the largest absolute and relative deviations.
    !>
    !> @param[in] label Label printed in diagnostics.
    !> @param[in] ref   Reference vector.
    !> @param[in] new   Newly generated vector.
    !> @param[in] d1    Vector length.
    subroutine check_array_1d(label, ref, new, d1, atol)
        character(*), intent(in) :: label
        integer, intent(in) :: d1
        real(WP), dimension(d1), intent(in) :: ref, new
        real(WP), intent(in), optional :: atol
        real(WP) :: delta, threshold, max_delta, max_rel, ref_val
        integer :: j, mj, mrj

        max_delta = 0.0_wp
        max_rel = 0.0_wp
        mj = 1
        mrj = 1

        do j = 1, d1
            if (ieee_is_nan(ref(j)) .or. ieee_is_nan(new(j))) then
                write(*, *) 'FAIL: ', label, ' contains NaN at index (', j, ')'
                test_passed = .false.
                return
            end if

            ref_val = ref(j)
            delta = abs(ref_val - new(j))
            if (abs(ref_val) > 0.0_wp) then
                if (delta / abs(ref_val) > max_rel) then
                    max_rel = delta / abs(ref_val)
                    mrj = j
                end if
            end if
            if (delta > max_delta) then
                max_delta = delta
                mj = j
            end if
            threshold = max(abs(ref_val) * rtol, 1.0e-30_wp)
            if (present(atol)) threshold = max(threshold, atol)
            if (delta > threshold) then
                test_passed = .false.
                nfail = nfail + 1
                if (verbose_output .or. fail_printed < max_fail_print) then
                    write(*, *) 'FAIL: ', label, ' mismatch at index (', j, ')'
                    write(*, *) '  Ref:', ref_val, ' New:', new(j), ' Diff:', delta
                    fail_printed = fail_printed + 1
                else if (.not. fail_suppression_noted) then
                    write(*, *) 'NOTE: Further failure details suppressed (set FSPS_TEST_VERBOSE=1 to expand).'
                    fail_suppression_noted = .true.
                end if
            end if
        end do
        write(*, *) 'SUMMARY: ', label, ' max abs diff=', max_delta, ' at (', mj, ')', &
                    ' max rel diff=', max_rel, ' at (', mrj, ')'
    end subroutine check_array_1d

    !> @brief
    !> Compare magnitude vectors in linear flux space.
    !>
    !> @details
    !> Magnitudes are transformed to relative linear flux prior to comparison so
    !> tolerances are evaluated on physically meaningful scale ratios. Exponents
    !> are clamped to avoid overflow/underflow in `exp`.
    !>
    !> @param[in] label   Label printed in diagnostics.
    !> @param[in] ref_mag Reference magnitudes.
    !> @param[in] new_mag Newly generated magnitudes.
    !> @param[in] d1      Vector length.
    subroutine check_mags_1d(label, ref_mag, new_mag, d1)
        character(*), intent(in) :: label
        integer, intent(in) :: d1
        real(WP), dimension(d1), intent(in) :: ref_mag, new_mag
        real(WP) :: delta, threshold, max_delta, max_rel, ref_val
        real(WP) :: flux_ref, flux_new, exp_ref, exp_new
        integer :: j, mj

        max_delta = 0.0_wp
        max_rel = 0.0_wp
        mj = 1

        do j = 1, d1
            if (ieee_is_nan(ref_mag(j)) .or. ieee_is_nan(new_mag(j))) then
                write(*, *) 'FAIL: ', label, ' contains NaN at index (', j, ')'
                test_passed = .false.
                return
            end if

            ! Convert magnitudes to linear flux units (relative scale)
            exp_ref = -0.4_wp * ref_mag(j) * log(10.0_wp)
            exp_new = -0.4_wp * new_mag(j) * log(10.0_wp)
            if (exp_ref < -700.0_wp) then
                flux_ref = 0.0_wp
            else if (exp_ref > 700.0_wp) then
                flux_ref = huge(1.0_wp)
            else
                flux_ref = exp(exp_ref)
            end if
            if (exp_new < -700.0_wp) then
                flux_new = 0.0_wp
            else if (exp_new > 700.0_wp) then
                flux_new = huge(1.0_wp)
            else
                flux_new = exp(exp_new)
            end if

            ref_val = flux_ref
            delta = abs(flux_ref - flux_new)
            if (abs(ref_val) > 0.0_wp) then
                max_rel = max(max_rel, delta / abs(ref_val))
            end if
            if (delta > max_delta) then
                max_delta = delta
                mj = j
            end if
            threshold = max(abs(ref_val) * rtol, 1.0e-30_wp)
            if (delta > threshold) then
                test_passed = .false.
                nfail = nfail + 1
                if (verbose_output .or. fail_printed < max_fail_print) then
                    write(*, *) 'FAIL: ', label, ' mismatch at index (', j, ')'
                    write(*, *) '  Ref mag:', ref_mag(j), ' New mag:', new_mag(j)
                    write(*, *) '  Ref flux:', flux_ref, ' New flux:', flux_new, ' Diff:', delta
                    fail_printed = fail_printed + 1
                else if (.not. fail_suppression_noted) then
                    write(*, *) 'NOTE: Further failure details suppressed (set FSPS_TEST_VERBOSE=1 to expand).'
                    fail_suppression_noted = .true.
                end if
            end if
        end do
        write(*, *) 'SUMMARY: ', label, ' max abs diff=', max_delta, ' at (', mj, ')', &
                    ' max rel diff=', max_rel
    end subroutine check_mags_1d

    !> @brief
    !> Compare a scalar value at a logical step index.
    !>
    !> @param[in] label Label printed in diagnostics.
    !> @param[in] idx   Logical step/time index for reporting.
    !> @param[in] r     Reference value.
    !> @param[in] n     Newly generated value.
    !> @param[in] atol  Optional absolute tolerance override.
    subroutine check_val(label, idx, r, n, atol)
        character(*), intent(in) :: label
        integer, intent(in) :: idx
        real(WP), intent(in) :: r, n
        real(WP), intent(in), optional :: atol

        real(WP) :: delta, threshold, local_atol

        if (ieee_is_nan(r) .or. ieee_is_nan(n)) then
            write(*, *) 'FAIL: ', label, ' contains NaN at step ', idx
            test_passed = .false.
            return
        end if

        local_atol = 1.0e-30_wp
        if (present(atol)) local_atol = atol

        delta = abs(r - n)
        threshold = max(abs(r) * rtol, local_atol)

        if (delta > threshold) then
            test_passed = .false.
            nfail = nfail + 1
            if (verbose_output .or. fail_printed < max_fail_print) then
                write(*, *) 'FAIL: ', label, ' mismatch at step ', idx
                write(*, *) '  Ref:', r, ' New:', n, ' Diff:', delta
                fail_printed = fail_printed + 1
            else if (.not. fail_suppression_noted) then
                write(*, *) 'NOTE: Further failure details suppressed (set FSPS_TEST_VERBOSE=1 to expand).'
                fail_suppression_noted = .true.
            end if
        end if
    end subroutine check_val

    !> @brief
    !> Emit diagnostic context and parameter state.
    !>
    !> @details
    !> Used only when verbose mode is enabled to help debug cross-build or
    !> cross-backend mismatches in regression runs.
    !>
    !> @param[in] label Human-readable checkpoint label.
    !> @param[in] ctx   FSPS context snapshot.
    !> @param[in] pset  Active parameter set snapshot.
    subroutine dump_state(label, ctx, pset)
        character(*), intent(in) :: label
        type(fsps_context_t), intent(in) :: ctx
        type(PARAMS), intent(in) :: pset

        write(*, *) '--- STATE:', trim(label)
        write(*, *) '  imf_type=', ctx%imf_type_val
        write(*, *) '  imf_lower_limit=', ctx%state%imf_lower_limit, ' imf_upper_limit=', ctx%state%imf_upper_limit
        write(*, *) '  dust_type=', ctx%dust_type_val, ' add_dust_emission=', ctx%add_dust_emission_val
        write(*, *) '  add_neb_emission=', ctx%add_neb_emission_val, ' nebemlineinspec=', ctx%nebemlineinspec_val
        write(*, *) '  interpolation_type=', ctx%interpolation_type_val, ' tiny_logt=', ctx%tiny_logt_val
        write(*, *) '  pset: sfh=', pset%sfh, ' tau=', pset%tau, ' const=', pset%const, ' fburst=', pset%fburst
        write(*, *) '  pset: sf_start=', pset%sf_start, ' sf_trunc=', pset%sf_trunc, ' tburst=', pset%tburst
        write(*, *) '  pset: dust1=', pset%dust1, ' dust2=', pset%dust2, ' zred=', pset%zred
        write(*, *) '  dims: ntfull=', ctx%state%ntfull, ' nspec=', ctx%state%nspec, ' nbands=', ctx%state%nbands
    end subroutine dump_state

    !> @brief
    !> Emit compact SSP summary diagnostics.
    !>
    !> @param[in] label    Human-readable checkpoint label.
    !> @param[in] ctx      FSPS context (used for time grid reporting).
    !> @param[in] mass_ssp SSP mass history.
    !> @param[in] lbol_ssp SSP bolometric luminosity history.
    subroutine dump_ssp_summary(label, ctx, mass_ssp, lbol_ssp)
        character(*), intent(in) :: label
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(:), intent(in) :: mass_ssp, lbol_ssp
        integer :: n

        n = size(mass_ssp)
        write(*, *) '--- SSP SUMMARY:', trim(label)
        write(*, *) '  mass_ssp(1)=', mass_ssp(1), ' mass_ssp(n)=', mass_ssp(n)
        write(*, *) '  lbol_ssp(1)=', lbol_ssp(1), ' lbol_ssp(n)=', lbol_ssp(n)
        write(*, *) '  time_full(1)=', ctx%state%time_full(1), ' time_full(n)=', ctx%state%time_full(n)
    end subroutine dump_ssp_summary

    !> @brief
    !> Emit compact CSP summary diagnostics.
    !>
    !> @param[in] label   Human-readable checkpoint label.
    !> @param[in] ocompsp CSP output series.
    subroutine dump_csp_summary(label, ocompsp)
        character(*), intent(in) :: label
        type(COMPSPOUT), dimension(:), intent(in) :: ocompsp
        integer :: n

        n = size(ocompsp)
        write(*, *) '--- CSP SUMMARY:', trim(label)
        write(*, *) '  age(1)=', ocompsp(1)%age, ' age(n)=', ocompsp(n)%age
        write(*, *) '  mass_csp(1)=', ocompsp(1)%mass_csp, ' mass_csp(n)=', ocompsp(n)%mass_csp
        write(*, *) '  lbol_csp(1)=', ocompsp(1)%lbol_csp, ' lbol_csp(n)=', ocompsp(n)%lbol_csp
    end subroutine dump_csp_summary

    !> @brief
    !> Ensure output directory exists before regression execution.
    !>
    !> @details
    !> Accepts either a directory path (`path/`) or a file path (`path/file`) and
    !> creates the directory component if it is missing.
    !>
    !> @param[in] path Directory path or file path with directory component.
    subroutine ensure_output_dir(path)
        character(*), intent(in) :: path
        integer :: last_slash, ierr
        logical :: exists
        character(:), allocatable :: dir_path

        ! If the path provided is the directory (ends in /)
        if (path(len_trim(path):len_trim(path)) == '/') then
            dir_path = path(1:len_trim(path) - 1)
        else
            ! Original logic for "path/to/file" strings
            last_slash = scan(path, '/', back=.true.)
            if (last_slash > 0) then
                dir_path = path(1:last_slash - 1)
            else
                return ! No directory part found
            end if
        end if

        inquire(file=dir_path, exist=exists)

        if (.not. exists) then
            write(*, *) 'Pre-test check: Creating missing directory: ', dir_path
            call execute_command_line('mkdir -p ' // dir_path, exitstat=ierr)
            if (ierr /= 0) then
                write(*, *) 'FATAL: Could not create output directory.'
                stop 1
            end if
        end if
    end subroutine ensure_output_dir

end program test_runner
