program test_tuning
    !> The tuning registry (src/core/opensps_tuning.f90): defaults, parsing, legal ranges, precedence of the
    !> environment and tuning_set, and the report. The environment is set by the test driver (meson.build):
    !> OPENSPS_TUNE_HOST_SFH_MAX=8 (valid), OPENSPS_TUNE_SSP_BATCH_CHUNK=lots (refused, default kept),
    !> OPENSPS_TUNE_PINNED_CHUNK_BYTES unset.
    use opensps_tuning
    use opensps_build_arch, only: BUILD_GPU_ARCH
    use test_utils_mod, only: tap_finish, print_major_header, print_group, print_summary_line, assert_true, &
                              assert_int_equals
    implicit none

    integer :: total_tests = 0, total_failures = 0
    integer :: v, i
    logical :: ok
    character(len=2000) :: buf
    character(len=16) :: scenario

    call get_command_argument(1, scenario)
    call print_major_header("tuning registry")
    if (trim(scenario) == 'badprofile') then
        call scenario_bad_profile()
        call finish()
    else if (trim(scenario) == 'noprofile') then
        call scenario_no_profile()
        call finish()
    end if

    call print_group("Parsing")
    call tuning_parse('2048', v, ok)
    call assert_true(ok .and. v == 2048, "plain integer", total_tests, total_failures)
    call tuning_parse(' 16M ', v, ok)
    call assert_true(ok .and. v == 16 * 1024 * 1024, "M suffix, blanks around", total_tests, total_failures)
    call tuning_parse('64k', v, ok)
    call assert_true(ok .and. v == 65536, "k suffix", total_tests, total_failures)
    call tuning_parse('-3', v, ok)
    call assert_true(ok .and. v == -3, "negative", total_tests, total_failures)
    call tuning_parse('12x', v, ok)
    call assert_true(.not. ok, "trailing junk refused", total_tests, total_failures)
    call tuning_parse('', v, ok)
    call assert_true(.not. ok, "empty refused", total_tests, total_failures)
    call tuning_parse('M', v, ok)
    call assert_true(.not. ok, "suffix alone refused", total_tests, total_failures)
    call tuning_parse('4G', v, ok)
    call assert_true(.not. ok, "overflow of a default integer refused", total_tests, total_failures)
    call tuning_parse('99999999999999', v, ok)
    call assert_true(.not. ok, "huge value refused", total_tests, total_failures)

    call print_group("Environment")
    call assert_int_equals(8, tuning_get(TUNE_P_HOST_SFH_MAX), "valid OPENSPS_TUNE_HOST_SFH_MAX is used", &
                           total_tests, total_failures)
    call assert_int_equals(2048, tuning_get(TUNE_P_SSP_BATCH_CHUNK), "invalid OPENSPS_TUNE_SSP_BATCH_CHUNK keeps the default", &
                           total_tests, total_failures)
    call assert_int_equals(16 * 1024 * 1024, tuning_get(TUNE_P_PINNED_CHUNK_BYTES), "unset parameter has its default", &
                           total_tests, total_failures)
    call assert_int_equals(0, tuning_get(0), "unknown id reads 0", total_tests, total_failures)

    call print_group("tuning_set")
    call tuning_set(TUNE_P_HOST_SFH_MAX, 0, ok)
    call assert_true(ok .and. tuning_get(TUNE_P_HOST_SFH_MAX) == 0, "set beats the environment (0 is legal)", &
                     total_tests, total_failures)
    call tuning_set(TUNE_P_SSP_BATCH_CHUNK, 0, ok)
    call assert_true(.not. ok .and. tuning_get(TUNE_P_SSP_BATCH_CHUNK) == 2048, "out-of-range set refused, value kept", &
                     total_tests, total_failures)
    call tuning_set(99, 1, ok)
    call assert_true(.not. ok, "unknown id refused", total_tests, total_failures)
    call assert_true(tuning_in_range(TUNE_P_PINNED_CHUNK_BYTES, 65536) .and. &
                     .not. tuning_in_range(TUNE_P_PINNED_CHUNK_BYTES, 65535), "range limits are inclusive", &
                     total_tests, total_failures)

    call print_group("Report and reset")
    open (newunit=i, status='scratch', action='readwrite')
    call tuning_report(i)
    rewind (i)
    buf = ''
    read (i, '(a)') buf
    call assert_true(index(buf, 'OpenSPS tuning') > 0, "report has its header", total_tests, total_failures)
    read (i, '(a)') buf
    call assert_true(index(buf, 'device profile') > 0, "report names the device profile", total_tests, total_failures)
    read (i, '(a)') buf
    call assert_true(index(buf, 'SSP_BATCH_CHUNK') > 0 .and. index(buf, 'default') > 0, "report lists a parameter", &
                     total_tests, total_failures)
    close (i)
    call tuning_reset()
    call assert_int_equals(8, tuning_get(TUNE_P_HOST_SFH_MAX), "reset re-reads the environment", &
                           total_tests, total_failures)
    call assert_true(len_trim(tuning_name(1)) > 0 .and. len_trim(tuning_name(0)) == 0 .and. tuning_count() == 3, &
                     "names and count", total_tests, total_failures)

    call print_group("Device profiles (OPENSPS_TUNE_PROFILE=cc80)")
    call tuning_reset()
    call assert_true(trim(tuning_profile()) == 'cc80', "the environment selects the profile", total_tests, total_failures)
    call assert_true(tuning_source(TUNE_P_HOST_SFH_MAX) == 1 .and. tuning_get(TUNE_P_HOST_SFH_MAX) == 8, &
                     "the environment beats the profile", total_tests, total_failures)
    call assert_true(tuning_source(TUNE_P_PINNED_CHUNK_BYTES) == 3 .and. &
                     tuning_get(TUNE_P_PINNED_CHUNK_BYTES) == 16 * 1024 * 1024, "a profile entry applies (source 3)", &
                     total_tests, total_failures)
    call assert_true(tuning_source(TUNE_P_SSP_BATCH_CHUNK) == 0, "a parameter with no entry has its default (source 0)", &
                     total_tests, total_failures)
    call tuning_select_profile('GFX1201', ok)
    call assert_true(ok .and. trim(tuning_profile()) == 'gfx1201' .and. tuning_get(TUNE_P_HOST_SFH_MAX) == 160 .and. &
                     tuning_source(TUNE_P_HOST_SFH_MAX) == 3, "selecting a profile by name (any case) applies it", &
                     total_tests, total_failures)
    call tuning_set(TUNE_P_HOST_SFH_MAX, 33, ok)
    call assert_true(ok .and. tuning_get(TUNE_P_HOST_SFH_MAX) == 33 .and. tuning_source(TUNE_P_HOST_SFH_MAX) == 2, &
                     "tuning_set beats the profile", total_tests, total_failures)
    call tuning_select_profile('nonesuch', ok)
    call assert_true(.not. ok .and. len_trim(tuning_profile()) == 0 .and. tuning_get(TUNE_P_HOST_SFH_MAX) == 16, &
                     "an unknown profile is refused and the defaults apply", total_tests, total_failures)
    call tuning_select_profile('', ok)
    call assert_true(ok .and. len_trim(tuning_profile()) == 0, "an empty name means no profile", total_tests, total_failures)
    call assert_true(tuning_profile_known('cc80') .and. tuning_profile_known('gfx1201') .and. &
                     .not. tuning_profile_known('cc99'), "profile names", total_tests, total_failures)

    call finish()

contains

    subroutine finish()
        call print_summary_line("tuning registry", total_tests - total_failures, total_tests)
        call tap_finish()
        if (total_failures > 0) stop 1
        stop 0
    end subroutine finish

    !> OPENSPS_TUNE_PROFILE=nonesuch: a message on stderr, no profile, the defaults
    subroutine scenario_bad_profile()
        call print_group("Unknown profile in the environment")
        call assert_true(len_trim(tuning_profile()) == 0, "no profile applies", total_tests, total_failures)
        call assert_true(tuning_get(TUNE_P_HOST_SFH_MAX) == 16 .and. tuning_source(TUNE_P_HOST_SFH_MAX) == 0, &
                         "the defaults apply", total_tests, total_failures)
    end subroutine scenario_bad_profile

    !> No OPENSPS_TUNE_* at all: the profile is the build's single offload target, none in a build without offload
    subroutine scenario_no_profile()
        call print_group("No environment")
        if (len_trim(BUILD_GPU_ARCH) == 0) then
            call assert_true(len_trim(tuning_profile()) == 0, "a build without offload has no profile", &
                             total_tests, total_failures)
        else if (index(BUILD_GPU_ARCH, ',') > 0) then
            call assert_true(len_trim(tuning_profile()) == 0, "a build for several targets has no profile", &
                             total_tests, total_failures)
        else
            call assert_true(len_trim(tuning_profile()) > 0, "a single-target build selects its profile", &
                             total_tests, total_failures)
        end if
        call assert_int_equals(2048, tuning_get(TUNE_P_SSP_BATCH_CHUNK), "defaults elsewhere", total_tests, total_failures)
    end subroutine scenario_no_profile

end program test_tuning
