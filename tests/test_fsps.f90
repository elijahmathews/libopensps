program test_fsps
    !> @brief
    !> Master unit test runner for all FSPS modules.
    !> Aggregates results from individual test modules.

    ! Core modules
    use test_fsps_cache_mod, only: run_fsps_cache_tests, &
                                   failures_cache => total_failures, &
                                   tests_cache => total_tests
    use test_fsps_context_mod, only: run_fsps_context_tests, &
                                     failures_context => total_failures, &
                                     tests_context => total_tests
    use test_fsps_context_types_mod, only: run_fsps_context_types_tests, &
                                           failures_context_types => total_failures, &
                                           tests_context_types => total_tests
    use test_fsps_strings_mod, only: run_fsps_strings_tests, &
                                     failures_strings => total_failures, &
                                     tests_strings => total_tests
    use test_fsps_types_mod, only: run_fsps_types_tests, &
                                   failures_types => total_failures, &
                                   tests_types => total_tests

    ! IMF modules
    use test_fsps_imf_mod, only: run_fsps_imf_tests, &
                                 failures_imf => total_failures, &
                                 tests_imf => total_tests
    ! Math modules
    use test_fsps_integration_mod, only: run_fsps_integration_tests, &
                                         failures_integration => total_failures, &
                                         tests_integration => total_tests
    use test_fsps_interpolation_mod, only: run_fsps_interpolation_tests, &
                                           failures_interpolation => total_failures, &
                                           tests_interpolation => total_tests
    use test_fsps_special_functions_mod, only: run_fsps_special_functions_tests, &
                                           failures_special_functions => total_failures, &
                                           tests_special_functions => total_tests
    use test_fsps_dust_mod, only: run_fsps_dust_tests, &
                                  failures_dust => total_failures, &
                                  tests_dust => total_tests
    use test_fsps_gas_mod, only: run_fsps_gas_tests, &
                                 failures_gas => total_failures, &
                                 tests_gas => total_tests
    use test_fsps_stellar_modifications_mod, only: run_fsps_stellar_modifications_tests, &
                                 failures_stellar_modifications => total_failures, &
                                 tests_stellar_modifications => total_tests
    use test_fsps_cosmology_mod, only: run_fsps_cosmology_tests, &
                                  failures_cosmology => total_failures, &
                                  tests_cosmology => total_tests
    use test_fsps_smoothing_mod, only: run_fsps_smoothing_tests, &
                                  failures_smoothing => total_failures, &
                                  tests_smoothing => total_tests
    use test_fsps_spectral_indices_mod, only: run_fsps_spectral_indices_tests, &
                                  failures_spectral_indices => total_failures, &
                                  tests_spectral_indices => total_tests
    use test_fsps_spectral_library_mod, only: run_fsps_spectral_library_tests, &
                                  failures_spectral_library => total_failures, &
                                  tests_spectral_library => total_tests
    use test_fsps_photometry_mod, only: run_fsps_photometry_tests, &
                                  failures_photometry => total_failures, &
                                  tests_photometry => total_tests
    use test_fsps_io_mod, only: run_fsps_io_tests, &
                                  failures_io => total_failures, &
                                  tests_io => total_tests
    use test_fsps_ssp_mod, only: run_fsps_ssp_tests, &
                                 failures_ssp => total_failures, &
                                 tests_ssp => total_tests
    use test_fsps_sfh_mod, only: run_fsps_sfh_tests, &
                                 failures_sfh => total_failures, &
                                 tests_sfh => total_tests
    use test_fsps_csp_mod, only: run_fsps_csp_tests, &
                                 failures_csp => total_failures, &
                                 tests_csp => total_tests
    ! Test utilities
    use test_utils_mod, only: print_summary_line, print_major_header, tap_finish
    implicit none

    integer :: grand_total_failures = 0
    integer :: grand_total_tests = 0
    ! Optional argument: run only the named module (e.g. "fsps_sfh"); meson registers one test per
    ! module this way so that they run in parallel and are reported separately
    character(len=64) :: only_module = ''

    call get_command_argument(1, only_module)

    call print_major_header("FSPS UNIT TEST SUITE")

    ! Run all the tests
    if (selected('fsps_cache')) call run_fsps_cache_tests()
    if (selected('fsps_context')) call run_fsps_context_tests()
    if (selected('fsps_context_types')) call run_fsps_context_types_tests()
    if (selected('fsps_strings')) call run_fsps_strings_tests()
    if (selected('fsps_types')) call run_fsps_types_tests()

    if (selected('fsps_io')) call run_fsps_io_tests()

    if (selected('fsps_integration')) call run_fsps_integration_tests()
    if (selected('fsps_interpolation')) call run_fsps_interpolation_tests()
    if (selected('fsps_special_functions')) call run_fsps_special_functions_tests()

    if (selected('fsps_cosmology')) call run_fsps_cosmology_tests()
    if (selected('fsps_csp')) call run_fsps_csp_tests()
    if (selected('fsps_dust')) call run_fsps_dust_tests()
    if (selected('fsps_gas')) call run_fsps_gas_tests()
    if (selected('fsps_imf')) call run_fsps_imf_tests()
    if (selected('fsps_sfh')) call run_fsps_sfh_tests()
    if (selected('fsps_ssp')) call run_fsps_ssp_tests()
    if (selected('fsps_stellar_modifications')) call run_fsps_stellar_modifications_tests()

    if (selected('fsps_photometry')) call run_fsps_photometry_tests()
    if (selected('fsps_smoothing')) call run_fsps_smoothing_tests()
    if (selected('fsps_spectral_indices')) call run_fsps_spectral_indices_tests()
    if (selected('fsps_spectral_library')) call run_fsps_spectral_library_tests()

    ! --- Summary ---
    call print_major_header("FSPS UNIT TEST FINAL REPORT")
    
    call print_summary_line( &
        "fsps_cache", &
        (tests_cache - failures_cache), &
        tests_cache &
    )
    call print_summary_line( &
        "fsps_context", &
        (tests_context - failures_context), &
        tests_context &
    )
    call print_summary_line( &
        "fsps_context_types", &
        (tests_context_types - failures_context_types), &
        tests_context_types &
    )
    call print_summary_line( &
        "fsps_strings", &
        (tests_strings - failures_strings), &
        tests_strings &
    )
    call print_summary_line( &
        "fsps_types", &
        (tests_types - failures_types), &
        tests_types &
    )

    call print_summary_line( &
        "fsps_io", &
        (tests_io - failures_io), &
        tests_io &
    )
    
    call print_summary_line( &
        "fsps_integration", &
        (tests_integration - failures_integration), &
        tests_integration &
    )
    call print_summary_line( &
        "fsps_interpolation", &
        (tests_interpolation - failures_interpolation), &
        tests_interpolation &
    )
    call print_summary_line( &
        "fsps_special_functions", &
        (tests_special_functions - failures_special_functions), &
        tests_special_functions &
    )
    
    call print_summary_line( &
        "fsps_cosmology", &
        (tests_cosmology - failures_cosmology), &
        tests_cosmology &
    )
    call print_summary_line( &
        "fsps_csp", &
        (tests_csp - failures_csp), &
        tests_csp &
    )
    call print_summary_line( &
        "fsps_dust", &
        (tests_dust - failures_dust), &
        tests_dust &
    )
    call print_summary_line( &
        "fsps_gas", &
        (tests_gas - failures_gas), &
        tests_gas &
    )
    call print_summary_line( &
        "fsps_imf", &
        (tests_imf - failures_imf), &
        tests_imf &
    )
    call print_summary_line( &
        "fsps_sfh", &
        (tests_sfh - failures_sfh), &
        tests_sfh &
    )
    call print_summary_line( &
        "fsps_ssp", &
        (tests_ssp - failures_ssp), &
        tests_ssp &
    )
    call print_summary_line( &
        "fsps_stellar_modifications", &
        (tests_stellar_modifications - failures_stellar_modifications), &
        tests_stellar_modifications &
    )

    call print_summary_line( &
        "fsps_photometry", &
        (tests_photometry - failures_photometry), &
        tests_photometry &
    )
    call print_summary_line( &
        "fsps_smoothing", &
        (tests_smoothing - failures_smoothing), &
        tests_smoothing &
    )
    call print_summary_line( &
        "fsps_spectral_indices", &
        (tests_spectral_indices - failures_spectral_indices), &
        tests_spectral_indices &
    )
    call print_summary_line( &
        "fsps_spectral_library", &
        (tests_spectral_library - failures_spectral_library), &
        tests_spectral_library &
    )
    
    grand_total_failures = failures_cache + &
                           failures_context + &
                           failures_context_types + &
                           failures_strings + &
                           failures_types + &
                           failures_io + &
                           failures_integration + &
                           failures_interpolation + &
                           failures_special_functions + &
                           failures_cosmology + &
                           failures_csp + &
                           failures_dust + &
                           failures_gas + &
                           failures_imf + &
                           failures_sfh + &
                           failures_ssp + &
                           failures_stellar_modifications + &
                           failures_photometry + &
                           failures_smoothing + &
                           failures_spectral_indices + &
                           failures_spectral_library
    
    grand_total_tests = tests_cache + &
                        tests_context + &
                        tests_context_types + &
                        tests_strings + &
                        tests_types + &
                        tests_io + &
                        tests_integration + &
                        tests_interpolation + &
                        tests_special_functions + &
                        tests_cosmology + &
                        tests_csp + &
                        tests_dust + &
                        tests_gas + &
                        tests_imf + &
                        tests_sfh + &
                        tests_ssp + &
                        tests_stellar_modifications + &
                        tests_photometry + &
                        tests_smoothing + &
                        tests_spectral_indices + &
                        tests_spectral_library

    print *
    call print_summary_line("Result", grand_total_tests - grand_total_failures, grand_total_tests)
    print *

    call tap_finish()

    if (grand_total_failures == 0) then
        stop 0
    else
        stop 1
    end if

contains

    logical function selected(module_name)
        character(len=*), intent(in) :: module_name

        selected = len_trim(only_module) == 0 .or. trim(only_module) == module_name
    end function selected

end program test_fsps