module test_fsps_io_mod
    use fsps_precision, only: WP
    use fsps_constants, only: SAFE_FLOOR, C_LIGHT, NM, NDIM_LOGT, NDIM_LOGG, NLAM_NEBCONT, NEBNZ, NEBNAGE, NEBNIP
    use fsps_context_types, only: fsps_context_t
    use fsps_types, only: params, compspout
    use fsps_io, only: read_isochrone_database, write_binary_spectral_lib, read_spectral_binary, read_bpass_data, &
                        load_zlegend_file, load_wavelength_grid, load_filter_definitions, load_nebular_grid, &
                        load_dust_emission_table, load_tabular_sfh, write_csp_output_files, write_isochrone_cmd, &
                        load_index_definitions, load_attenuation_curves, load_lsf_data, load_standard_sed
    use fsps_integration, only: integrate_trapezoid_array
    use fsps_datalib, only: datalib_t, datalib_open, datalib_axis_values, datalib_check_array, datalib_read_real
    use, intrinsic :: iso_fortran_env, only: int64
    use fsps_interpolation, only: interpolate_linear
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_int_equals, assert_true, assert_relative_error
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_io_tests, total_failures, total_tests

contains

    subroutine run_fsps_io_tests()
        call print_minor_header("fsps_io")

        call test_isochrone_standard()
        call test_isochrone_mist()
        call test_isochrone_parser_resilience()

        call test_spectral_binary_round_trip()
        call test_bpass_binary_loader()

        call test_mist_zlegend_parsing()
        call test_wavelength_grid_frequency()

        call test_filter_normalization()
        call test_filter_effective_wavelength()
        call test_filter_block_detection()

        call test_nebular_grid_interpolation()
        call test_dust_model_selection()

        call test_tabular_sfh_units()
        call test_tabular_sfh_clipping()

        call test_cmd_column_count()
        call test_output_mode_flags()

        call test_load_standard_sed()
        call test_load_lsf_data()
        call test_attenuation_curves()
        call test_index_definitions()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_io_tests

    ! ------------------------------------------------------------------------
    ! Group 1: Isochrone Database Parsing
    ! ------------------------------------------------------------------------
    subroutine test_isochrone_standard()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u

        allocate(ctx)

        call print_group("Isochrone Parser: Standard (Padova/BaSTI)")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/isochrones/Padova/Padova2007')

        call setup_iso_context(ctx, root, zlegend=0.0190_wp)

        file_path = trim(root) // '/data/isochrones/Padova/Padova2007/isoc_z0.0190.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(A)') '! header line'
        write(u, '(A)') '# track 1'
        write(u, '(8F10.4)') 9.0_wp, 1.0_wp, 0.9_wp, 0.1_wp, 3.70_wp, 4.0_wp, 0.0_wp, 1.0_wp
        write(u, '(8F10.4)') 9.0_wp, 2.0_wp, 1.8_wp, 0.2_wp, 3.72_wp, 4.1_wp, 0.1_wp, 2.0_wp
        write(u, '(8F10.4)') 9.0_wp, 3.0_wp, 2.7_wp, 0.3_wp, 3.74_wp, 4.2_wp, 0.2_wp, 3.0_wp
        close(u)

        call read_isochrone_database(ctx, 'pdva', 1)

        call assert_int_equals(3, ctx%state%nmass_isoc(1, 1), &
                               "nmass_isoc == 3", total_tests, total_failures)
        call assert_float_equals(1.0_wp, ctx%state%mini_isoc(1, 1, 1), 1.0e-6_wp, &
                                 "mini(1) matches", total_tests, total_failures)
        call assert_float_equals(2.0_wp, ctx%state%mini_isoc(1, 1, 2), 1.0e-6_wp, &
                                 "mini(2) matches", total_tests, total_failures)
        call assert_float_equals(3.0_wp, ctx%state%mini_isoc(1, 1, 3), 1.0e-6_wp, &
                                 "mini(3) matches", total_tests, total_failures)
        call assert_float_equals(-99.0_wp, ctx%state%lmdot_isoc(1, 1, 1), 1.0e-6_wp, &
                                 "lmdot default", total_tests, total_failures)

        call teardown_iso_context(ctx)
        deallocate(ctx)
    end subroutine test_isochrone_standard

    subroutine test_isochrone_mist()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u

        allocate(ctx)

        call print_group("Isochrone Parser: MIST 9-Column")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/isochrones/MIST')

        call setup_iso_context(ctx, root, zlegend=0.0190_wp)
        ctx%state%zsol = 0.0190_wp

        file_path = trim(root) // '/data/isochrones/MIST/isoc_zp0.00.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(A)') '! header'
        write(u, '(A)') '# track 1'
        write(u, '(9F10.4)') 9.0_wp, 1.1_wp, 1.0_wp, 0.1_wp, 3.70_wp, 4.0_wp, 0.0_wp, 1.0_wp, -5.0_wp
        write(u, '(9F10.4)') 9.0_wp, 1.2_wp, 1.1_wp, 0.2_wp, 3.72_wp, 4.1_wp, 0.1_wp, 2.0_wp, -4.5_wp
        write(u, '(9F10.4)') 9.0_wp, 1.3_wp, 1.2_wp, 0.3_wp, 3.74_wp, 4.2_wp, 0.2_wp, 3.0_wp, -4.0_wp
        close(u)

        call read_isochrone_database(ctx, 'mist', 1)

        call assert_float_equals(-5.0_wp, ctx%state%lmdot_isoc(1, 1, 1), 1.0e-6_wp, &
                                 "lmdot(1) read", total_tests, total_failures)
        call assert_float_equals(-4.5_wp, ctx%state%lmdot_isoc(1, 1, 2), 1.0e-6_wp, &
                                 "lmdot(2) read", total_tests, total_failures)
        call assert_float_equals(4.1_wp, ctx%state%logg_isoc(1, 1, 2), 1.0e-6_wp, &
                                 "logg maps to col 6", total_tests, total_failures)

        call teardown_iso_context(ctx)
        deallocate(ctx)
    end subroutine test_isochrone_mist

    subroutine test_isochrone_parser_resilience()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u

        allocate(ctx)

        call print_group("Isochrone Parser: Comments/Whitespace")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/isochrones/Padova/Padova2007')

        call setup_iso_context(ctx, root, zlegend=0.0190_wp)

        file_path = trim(root) // '/data/isochrones/Padova/Padova2007/isoc_z0.0190.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(A)') '   ! comment line before track'
        write(u, '(A)') '# track 1'
        write(u, '(A)') '   9.0  1.0  0.9  0.1  3.70  4.0  0.0  1.0'
        write(u, '(A)') ''
        write(u, '(A)') '   9.0  2.0  1.8  0.2  3.72  4.1  0.1  2.0'
        close(u)

        call read_isochrone_database(ctx, 'pdva', 1)

        call assert_int_equals(2, ctx%state%nmass_isoc(1, 1), "nmass ignores blanks", total_tests, total_failures)

        call teardown_iso_context(ctx)
        deallocate(ctx)
    end subroutine test_isochrone_parser_resilience

    ! ------------------------------------------------------------------------
    ! Group 2: Spectral Binary I/O
    ! ------------------------------------------------------------------------
    subroutine test_spectral_binary_round_trip()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec_out(:,:,:)
        character(len=256) :: root, file_ascii, file_bin
        integer :: u, i_g, i_t, i_s
        integer :: file_size, expected_size
        real(WP) :: expected

        allocate(ctx)

        call print_group("Spectral Binary Round Trip")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/spectra/MILES')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 4
        ctx%state%nzinit = 1
        allocate(ctx%state%zlegendinit(1))
        ctx%state%zlegendinit(1) = 0.0190_wp

        file_ascii = trim(root) // '/data/spectra/MILES/imiles_z0.0190.spectra'
        open(newunit=u, file=trim(file_ascii), status='replace', action='write')
        do i_g = 1, NDIM_LOGG
            do i_t = 1, NDIM_LOGT
                write(u, *) i_t, i_g
                write(u, *) (spec_value(i_s, i_t, i_g), i_s = 1, ctx%state%nspec)
            end do
        end do
        close(u)

        call write_binary_spectral_lib(ctx, 'miles', 1)

        allocate(spec_out(ctx%state%nspec, NDIM_LOGT, NDIM_LOGG))
        call read_spectral_binary(ctx, 'miles', 1, spec_out)

        expected = spec_value(2, 10, 3)
        call assert_float_equals(expected, spec_out(2, 10, 3), 1.0e-6_wp, &
                                 "Round-trip sample (2,10,3)", total_tests, total_failures)
        expected = spec_value(4, NDIM_LOGT, NDIM_LOGG)
        call assert_float_equals(expected, spec_out(4, NDIM_LOGT, NDIM_LOGG), 1.0e-6_wp, &
                                 "Round-trip sample (4,LT,LG)", total_tests, total_failures)

        file_bin = trim(root) // '/data/spectra/MILES/imiles_z0.0190.spectra.bin'

        if (allocated(spec_out)) deallocate(spec_out)
        if (associated(ctx%state%zlegendinit)) deallocate(ctx%state%zlegendinit)
        deallocate(ctx)
    end subroutine test_spectral_binary_round_trip

    subroutine test_bpass_binary_loader()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u, rec_len
        integer :: i, j, k
        real(WP), allocatable :: spec_write(:,:,:)

        allocate(ctx)

        call print_group("BPASS Binary Loader")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/isochrones/BPASS')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 4
        ctx%state%nt = 2
        ctx%state%nz = 2

        allocate(ctx%state%spec_lambda(ctx%state%nspec))
        allocate(ctx%state%spec_nu(ctx%state%nspec))
        allocate(ctx%state%time_full(ctx%state%nt))
        allocate(ctx%state%bpass_mass_ssp(ctx%state%nt, ctx%state%nz))
        allocate(ctx%state%bpass_spec_ssp(ctx%state%nspec, ctx%state%nt, ctx%state%nz))

        file_path = trim(root) // '/data/isochrones/BPASS/bpass.lambda'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        do i = 1, ctx%state%nspec
            write(u, '(F10.2)') 1000.0_wp * real(i, WP)
        end do
        close(u)

        file_path = trim(root) // '/data/isochrones/BPASS/bpass.mass'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        do i = 1, ctx%state%nt
            write(u, '(F10.2,2F10.2)') 1.0_wp * real(i, WP), 10.0_wp * real(i, WP), 20.0_wp * real(i, WP)
        end do
        close(u)

        allocate(spec_write(ctx%state%nspec, ctx%state%nt, ctx%state%nz))
        do k = 1, ctx%state%nz
            do j = 1, ctx%state%nt
                do i = 1, ctx%state%nspec
                    spec_write(i, j, k) = 1000.0_wp * real(k, WP) + 100.0_wp * real(j, WP) + real(i, WP)
                end do
            end do
        end do

        file_path = trim(root) // '/data/isochrones/BPASS/bpass_v2.2_salpeter100.ssp.bin'
        rec_len = ctx%state%nspec * ctx%state%nt * ctx%state%nz * 8
        open(newunit=u, file=trim(file_path), status='replace', access='direct', recl=rec_len, form='unformatted')
        write(u, rec=1) spec_write
        close(u)

        call read_bpass_data(ctx)

        call assert_float_equals(1000.0_wp, ctx%state%spec_lambda(1), 1.0e-6_wp, &
                                 "Lambda read", total_tests, total_failures)
        call assert_float_equals(C_LIGHT / 1000.0_wp, ctx%state%spec_nu(1), 1.0e-6_wp, &
                                 "Nu updated", total_tests, total_failures)
        call assert_float_equals(1101.0_wp, ctx%state%bpass_spec_ssp(1, 1, 1), 1.0e-6_wp, &
                                 "BPASS cube read", total_tests, total_failures)

        if (allocated(spec_write)) deallocate(spec_write)
        call teardown_bpass_context(ctx)
        deallocate(ctx)
    end subroutine test_bpass_binary_loader

    ! ------------------------------------------------------------------------
    ! Group 3: Metadata & Grid Initialization
    ! ------------------------------------------------------------------------
    subroutine test_mist_zlegend_parsing()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u
        real(WP) :: expected_m, expected_p

        allocate(ctx)

        call print_group("MIST Zlegend Parsing")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/isochrones/MIST')

        ctx%sps_home = trim(root)
        ctx%state%nz = 2
        ctx%state%zsol = 0.02_wp
        allocate(ctx%state%zlegend(ctx%state%nz))

        file_path = trim(root) // '/data/isochrones/MIST/zlegend.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(A)') 'm0.50'
        write(u, '(A)') 'p0.25'
        close(u)

        call load_zlegend_file(ctx, 'mist', .false.)

        expected_m = ctx%state%zsol * 10.0_wp**(-0.50_wp)
        expected_p = ctx%state%zsol * 10.0_wp**(0.25_wp)

        call assert_relative_error(expected_m, ctx%state%zlegend(1), 1.0e-6_wp, &
                                   "MIST m0.50", total_tests, total_failures)
        call assert_relative_error(expected_p, ctx%state%zlegend(2), 1.0e-6_wp, &
                                   "MIST p0.25", total_tests, total_failures)

        if (associated(ctx%state%zlegend)) deallocate(ctx%state%zlegend)
        deallocate(ctx)
    end subroutine test_mist_zlegend_parsing

    subroutine test_wavelength_grid_frequency()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u

        allocate(ctx)

        call print_group("Wavelength Grid Frequency Precalc")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/spectra/MILES')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 3
        allocate(ctx%state%spec_lambda(ctx%state%nspec))
        allocate(ctx%state%spec_nu(ctx%state%nspec))

        file_path = trim(root) // '/data/spectra/MILES/miles.lambda'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(F10.2)') 3000.0_wp
        write(u, '(F10.2)') 5000.0_wp
        write(u, '(F10.2)') 0.0_wp
        close(u)

        call load_wavelength_grid(ctx, 'miles')

        call assert_float_equals(3000.0_wp, ctx%state%spec_lambda(1), 1.0e-6_wp, "Lambda[1]", total_tests, total_failures)
        call assert_float_equals(C_LIGHT / 3000.0_wp, ctx%state%spec_nu(1), 1.0e-6_wp, "Nu[1]", total_tests, total_failures)
        call assert_float_equals(0.0_wp, ctx%state%spec_nu(3), 1.0e-6_wp, "Nu masked at zero", total_tests, total_failures)

        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%spec_nu)) deallocate(ctx%state%spec_nu)
        deallocate(ctx)
    end subroutine test_wavelength_grid_frequency

    ! ------------------------------------------------------------------------
    ! Group 4: Photometry & Calibration
    ! ------------------------------------------------------------------------
    subroutine test_filter_normalization()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u, i
        real(WP) :: norm_fac

        allocate(ctx)

        call print_group("Filter Normalization Integral")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 301
        ctx%state%nbands = 1
        allocate(ctx%state%spec_lambda(ctx%state%nspec))
        allocate(ctx%state%bands(ctx%state%nspec, ctx%state%nbands))
        allocate(ctx%state%filter_leff(ctx%state%nbands))

        ctx%state%bands = 0.0_wp

        do i = 1, ctx%state%nspec
            ctx%state%spec_lambda(i) = 3000.0_wp + 10.0_wp * real(i - 1, WP)
        end do

        file_path = trim(root) // '/data/test_filters.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(A)') '# test filter'
        write(u, '(2F10.2)') 3500.0_wp, 0.0_wp
        write(u, '(2F10.2)') 4000.0_wp, 1.0_wp
        write(u, '(2F10.2)') 5000.0_wp, 1.0_wp
        write(u, '(2F10.2)') 5500.0_wp, 0.0_wp
        close(u)

        call load_filter_definitions(ctx, 'test_filters.dat')

        norm_fac = integrate_trapezoid_array(ctx%state%spec_lambda, &
                                             ctx%state%bands(:, 1) / (ctx%state%spec_lambda + SAFE_FLOOR))
        call assert_float_equals(1.0_wp, norm_fac, 1.0e-3_wp, "Integral(T/lambda)=1", total_tests, total_failures)

        call teardown_filter_context(ctx)
        deallocate(ctx)
    end subroutine test_filter_normalization

    subroutine test_filter_effective_wavelength()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u, i
        real(WP) :: expected_leff, a, b

        allocate(ctx)

        call print_group("Filter Effective Wavelength")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 301
        ctx%state%nbands = 1
        allocate(ctx%state%spec_lambda(ctx%state%nspec))
        allocate(ctx%state%bands(ctx%state%nspec, ctx%state%nbands))
        allocate(ctx%state%filter_leff(ctx%state%nbands))

        ctx%state%bands = 0.0_wp

        do i = 1, ctx%state%nspec
            ctx%state%spec_lambda(i) = 3000.0_wp + 10.0_wp * real(i - 1, WP)
        end do

        file_path = trim(root) // '/data/test_filters.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(A)') '# test filter'
        write(u, '(2F10.2)') 3990.0_wp, 0.0_wp
        write(u, '(2F10.2)') 4000.0_wp, 1.0_wp
        write(u, '(2F10.2)') 5000.0_wp, 1.0_wp
        write(u, '(2F10.2)') 5010.0_wp, 0.0_wp
        close(u)

        call load_filter_definitions(ctx, 'test_filters.dat')

        a = 4000.0_wp
        b = 5000.0_wp
        expected_leff = sqrt(0.5_wp * (b * b - a * a) / log(b / a))
        call assert_relative_error(expected_leff, ctx%state%filter_leff(1), 5.0e-3_wp, &
                                   "Pivot lambda", total_tests, total_failures)

        call teardown_filter_context(ctx)
        deallocate(ctx)
    end subroutine test_filter_effective_wavelength

    subroutine test_filter_block_detection()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u

        allocate(ctx)

        call print_group("Filter Block Detection")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 6
        ctx%state%nbands = 2
        allocate(ctx%state%spec_lambda(ctx%state%nspec))
        allocate(ctx%state%bands(ctx%state%nspec, ctx%state%nbands))
        allocate(ctx%state%filter_leff(ctx%state%nbands))

        ! Initialize bands to zero
        ctx%state%bands = 0.0_wp

        ctx%state%spec_lambda = [300.0_wp, 600.0_wp, 900.0_wp, 1200.0_wp, 1800.0_wp, 2400.0_wp]

        ! Write a file WITH delimiters, matching standard format
        file_path = trim(root) // '/data/test_filters_explicit.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(A)') '# Block 1'
        write(u, '(2F10.2)') 1000.0_wp, 0.0_wp
        write(u, '(2F10.2)') 1500.0_wp, 1.0_wp
        write(u, '(2F10.2)') 2500.0_wp, 0.0_wp
        write(u, '(A)') '# Block 2'  ! <--- The fix: Explicit delimiter
        write(u, '(2F10.2)') 400.0_wp, 0.0_wp
        write(u, '(2F10.2)') 600.0_wp, 1.0_wp
        write(u, '(2F10.2)') 800.0_wp, 0.0_wp
        close(u)

        call load_filter_definitions(ctx, 'test_filters_explicit.dat')

        call assert_true(ctx%state%filter_leff(1) > 1200.0_wp, &
                         "Filter 1 leff in high range", total_tests, total_failures)
        call assert_true(ctx%state%filter_leff(2) < 1000.0_wp, &
                         "Filter 2 leff in low range", total_tests, total_failures)

        call teardown_filter_context(ctx)
        deallocate(ctx)
    end subroutine test_filter_block_detection

    ! ------------------------------------------------------------------------
    ! Group 5: Nebular & Dust Physics I/O
    ! ------------------------------------------------------------------------
    subroutine test_nebular_grid_interpolation()
        type(fsps_context_t), allocatable :: ctx
        character(len=512) :: root
        integer :: i
        type(datalib_t) :: lib
        integer(int64) :: off
        real(WP), allocatable :: raw_lam(:), raw_spec(:), raw_spec_log(:)
        real(WP), allocatable :: expected(:)

        allocate(ctx)

        call print_group("Nebular Grid Interpolation")

        ! The coarse nebular grid of the test library (standard format); no data/ needed
        call get_test_library_root(root)

        ctx%sps_home = trim(root)
        ctx%data_format = 'standard'
        ctx%state%nspec = 50
        allocate(ctx%state%spec_lambda(ctx%state%nspec))
        allocate(ctx%state%nebem_cont(ctx%state%nspec, NEBNZ, NEBNAGE, NEBNIP))

        ! Explicitly initialize fixed-size nebular arrays to prevent ifx
        ! from crashing on uninitialized memory if the file hits EOF early.
        ctx%state%nebem_line_pos = 0.0_wp
        ctx%state%nebem_line = 0.0_wp
        ctx%state%nebem_age = 0.0_wp
        ctx%state%nebem_logz = 0.0_wp
        ctx%state%nebem_logu = 0.0_wp

        do i = 1, ctx%state%nspec
            ctx%state%spec_lambda(i) = 1000.0_wp + 180.0_wp * real(i - 1, WP)
        end do

        call load_nebular_grid(ctx, 'mist', .false.)

        ! Expected: the first continuum of the library, log10(f + 1e-95), interpolated linearly
        call datalib_open(lib, root, 'nebular', 'mist_nodust')
        call datalib_axis_values(lib, 'wavelength', raw_lam)
        allocate(raw_spec(size(raw_lam)), raw_spec_log(size(raw_lam)), expected(ctx%state%nspec))
        call datalib_check_array(lib, 'continuum', [character(len=10) :: 'logZ', 'age', 'logU', 'wavelength'], &
                                 [NEBNZ, NEBNAGE, NEBNIP, size(raw_lam)], off)
        call datalib_read_real(lib, 'continuum', size(raw_spec, kind=int64), raw_spec, off)

        raw_spec_log = log10(raw_spec + 1.0e-95_wp)
        do i = 1, ctx%state%nspec
            expected(i) = interpolate_linear(raw_lam, raw_spec_log, ctx%state%spec_lambda(i))
        end do

        call assert_float_equals(expected(1), ctx%state%nebem_cont(1, 1, 1, 1), 1.0e-6_wp, &
                                 "Nebem interp [1]", total_tests, total_failures)
        call assert_float_equals(expected(25), ctx%state%nebem_cont(25, 1, 1, 1), 1.0e-6_wp, &
                                 "Nebem interp [25]", total_tests, total_failures)
        call assert_float_equals(expected(50), ctx%state%nebem_cont(50, 1, 1, 1), 1.0e-6_wp, &
                                 "Nebem interp [50]", total_tests, total_failures)

        deallocate(raw_lam, raw_spec, raw_spec_log, expected)
        call teardown_nebular_context(ctx)
        deallocate(ctx)
    end subroutine test_nebular_grid_interpolation

    subroutine test_dust_model_selection()
        type(fsps_context_t), allocatable :: ctx
        character(len=512) :: root
        integer :: i

        allocate(ctx)

        call print_group("Dust Model Selection (THEMIS vs DL07)")

        call get_test_library_root(root)

        ctx%sps_home = trim(root)
        ctx%data_format = 'standard'
        ctx%state%nspec = 50
        allocate(ctx%state%spec_lambda(ctx%state%nspec))
        ctx%state%spec_lambda = [(1000.0_wp + 200.0_wp * real(i - 1, WP), i = 1, ctx%state%nspec)]

        allocate(ctx%state%qpaharr(11))
        allocate(ctx%state%uminarr(37))
        allocate(ctx%state%dustem2_dustem(ctx%state%nspec, 11, 74))

        call load_dust_emission_table(ctx, 'THEMIS')

        call assert_int_equals(11, size(ctx%state%qpaharr), "THEMIS nqpah", total_tests, total_failures)
        call assert_true(size(ctx%state%uminarr) >= 35, "THEMIS numin >= 35", total_tests, total_failures)

        deallocate(ctx%state%qpaharr, ctx%state%uminarr, ctx%state%dustem2_dustem)

        allocate(ctx%state%qpaharr(7))
        allocate(ctx%state%uminarr(22))
        allocate(ctx%state%dustem2_dustem(ctx%state%nspec, 7, 44))

        call load_dust_emission_table(ctx, 'DL07')

        call assert_int_equals(7, size(ctx%state%qpaharr), "DL07 nqpah", total_tests, total_failures)
        call assert_int_equals(22, size(ctx%state%uminarr), "DL07 numin", total_tests, total_failures)

        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%qpaharr)) deallocate(ctx%state%qpaharr)
        if (associated(ctx%state%uminarr)) deallocate(ctx%state%uminarr)
        if (associated(ctx%state%dustem2_dustem)) deallocate(ctx%state%dustem2_dustem)
        deallocate(ctx)
    end subroutine test_dust_model_selection

    ! ------------------------------------------------------------------------
    ! Group 6: Runtime User Input (SFH)
    ! ------------------------------------------------------------------------
    subroutine test_tabular_sfh_units()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        character(len=256) :: root, file_path
        integer :: u

        allocate(ctx)

        call print_group("Tabular SFH Unit Conversion")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data')

        ctx%sps_home = trim(root)
        ctx%state%nz = 1

        file_path = trim(root) // '/data/sfh_units.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(3F10.2)') 1.0_wp, 2.0_wp, 0.02_wp
        close(u)

        pset%sfh = 2
        pset%sf_start = 0.0_wp
        ! Explicitly pad the string to avoid trailing garbage/truncation
        pset%sfh_filename = 'sfh_units.dat' // repeat(' ', 50 - len('sfh_units.dat'))

        call load_tabular_sfh(ctx, pset, 1)

        call assert_float_equals(1.0e9_wp, ctx%state%sfh_tab(1, 1), 1.0e-6_wp, &
                                 "Time converted to years", total_tests, total_failures)
        
        deallocate(ctx)
    end subroutine test_tabular_sfh_units

    subroutine test_tabular_sfh_clipping()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        character(len=256) :: root, file_path
        integer :: u

        allocate(ctx)

        call print_group("Tabular SFH SFR Clipping")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data')

        ctx%sps_home = trim(root)
        ctx%state%nz = 1

        file_path = trim(root) // '/data/sfh_clip.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(3F10.2)') 1.0_wp, -5.0_wp, 0.02_wp
        close(u)

        pset%sfh = 2
        pset%sf_start = 0.0_wp
        ! Explicitly pad the string
        pset%sfh_filename = 'sfh_clip.dat' // repeat(' ', 50 - len('sfh_clip.dat'))

        call load_tabular_sfh(ctx, pset, 1)

        call assert_float_equals(SAFE_FLOOR, ctx%state%sfh_tab(2, 1), 1.0e-12_wp, "SFR clipped", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_tabular_sfh_clipping

    ! ------------------------------------------------------------------------
    ! Group 7: Output Generation
    ! ------------------------------------------------------------------------
    subroutine test_cmd_column_count()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        character(len=256) :: root, file_path
        integer :: u, n_fields
        real(WP), allocatable :: time_grid(:), mini(:,:), mact(:,:), logl(:,:), logt(:,:), logg(:,:)
        real(WP), allocatable :: phase(:,:), ffco(:,:), lmdot(:,:), weights(:,:), mags(:,:,:)
        integer, allocatable :: nmass(:)

        allocate(ctx)

        call print_group("Isochrone CMD Column Count")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/OUTPUTS')

        ctx%output_home = trim(root)
        ctx%state%nbands = 5
        ctx%state%nz = 1
        allocate(ctx%state%zlegend(1))
        ctx%state%zlegend(1) = 0.0190_wp

        allocate(time_grid(1), nmass(1))
        allocate(mini(1, 2), mact(1, 2), logl(1, 2), logt(1, 2), logg(1, 2))
        allocate(phase(1, 2), ffco(1, 2), lmdot(1, 2), weights(1, 2))
        allocate(mags(1, 2, ctx%state%nbands))

        time_grid = 9.0_wp
        nmass = 2
        mini = 1.0_wp
        mact = 1.0_wp
        logl = 0.0_wp
        logt = 3.7_wp
        logg = 4.0_wp
        phase = 1.0_wp
        ffco = 0.0_wp
        lmdot = -5.0_wp
        weights = 1.0_wp
        mags = 20.0_wp

        pset%zmet = 1

        call write_isochrone_cmd(ctx, pset, 'test_cmd', time_grid, nmass, mini, mact, logl, logt, logg, &
                                 phase, ffco, lmdot, weights, mags)

        file_path = trim(root) // '/OUTPUTS/test_cmd.cmd'
        open(newunit=u, file=trim(file_path), status='old', action='read')
        read(u, '(A)') file_path
        read(u, '(A)') file_path
        close(u)

        n_fields = count_fields(trim(file_path))
        call assert_int_equals(11 + ctx%state%nbands, n_fields, "Data columns = 11 + nbands", total_tests, total_failures)

        deallocate(time_grid, nmass, mini, mact, logl, logt, logg, phase, ffco, lmdot, weights, mags)
        if (associated(ctx%state%zlegend)) deallocate(ctx%state%zlegend)
        deallocate(ctx)
    end subroutine test_cmd_column_count

    subroutine test_output_mode_flags()
        type(fsps_context_t), allocatable :: ctx
        type(params) :: pset
        type(compspout), allocatable :: results(:)
        character(len=256) :: root, mags_file, spec_file
        logical :: exists

        allocate(ctx)

        call print_group("CSP Output Mode Flags")

        call make_temp_root(root)
        call ensure_dir(trim(root) // '/OUTPUTS')

        ctx%output_home = trim(root)
        ctx%state%nz = 1
        ctx%state%nbands = 3
        allocate(ctx%state%zlegend(1))
        ctx%state%zlegend(1) = 0.0190_wp
        ctx%state%zsol = 0.0190_wp

        allocate(results(1))
        allocate(results(1)%mags(ctx%state%nbands))
        results(1)%mags = 20.0_wp
        results(1)%age = 1.0_wp
        results(1)%mass_csp = 1.0_wp
        results(1)%lbol_csp = 0.0_wp
        results(1)%sfr = 0.0_wp

        pset%sfh = 0
        pset%zmet = 1

        call write_csp_output_files(ctx, pset, results, 'test_out', 1)

        mags_file = trim(root) // '/OUTPUTS/test_out.mags'
        spec_file = trim(root) // '/OUTPUTS/test_out.spec'

        inquire(file=trim(mags_file), exist=exists)
        call assert_true(exists, "Mags file written", total_tests, total_failures)
        inquire(file=trim(spec_file), exist=exists)
        call assert_true(.not. exists, "Spec file not written", total_tests, total_failures)

        if (allocated(results)) deallocate(results) 
        
        if (associated(ctx%state%zlegend)) deallocate(ctx%state%zlegend)
        deallocate(ctx)
    end subroutine test_output_mode_flags

    ! ------------------------------------------------------------------------
    ! Group 8: Miscellaneous Tests
    ! ------------------------------------------------------------------------
    subroutine test_load_standard_sed()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u, i

        allocate(ctx)
        call print_group("Standard SED Loading")
        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/spectra')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 100
        ctx%state%nbands = 1
        allocate(ctx%state%spec_lambda(100), ctx%state%spec_nu(100))
        allocate(ctx%state%bands(100, 1), ctx%state%filter_leff(1))
        allocate(ctx%state%vega_spec(100), ctx%state%sun_spec(100))
        allocate(ctx%state%magvega(1), ctx%state%magsun(1))

        ! Setup: Lambda grid and a "perfect" filter
        do i=1, 100
            ctx%state%spec_lambda(i) = 3000.0_wp + 20.0_wp * real(i-1, WP)
            ctx%state%bands(i, 1) = 1.0_wp ! Flat response
        end do
        ctx%state%filter_leff(1) = 4000.0_wp

        ! Mock Vega (HAS Header)
        file_path = trim(root) // '/data/spectra/A0V_KURUCZ_92.SED'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, '(A)') '   # HEADER'
        do i=1, 1221
             ! Simple flat spectrum
            write(u, '(E16.8, 1X, E16.8)') 2000.0_wp + 10.0_wp*real(i,WP), 1.0e-12_wp 
        end do
        close(u)

        ! Mock Sun (NO Header) - FIX: Removed the header write here
        file_path = trim(root) // '/data/spectra/SUN_STScI.SED'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        ! write(u, '(A)') '   # HEADER'  <-- REMOVED
        do i=1, 1221
            write(u, '(E16.8, 1X, E16.8)') 2000.0_wp + 10.0_wp*real(i,WP), 1.0e-13_wp
        end do
        close(u)

        call load_standard_sed(ctx)

        ! Just verify it calculated *something* valid
        call assert_true(ctx%state%magvega(1) < 90.0_wp, "Vega Mag calculated", total_tests, total_failures)
        call assert_true(ctx%state%magsun(1) < 90.0_wp, "Sun Mag calculated", total_tests, total_failures)

        deallocate(ctx%state%spec_lambda, ctx%state%spec_nu, ctx%state%bands, ctx%state%filter_leff, &
                   ctx%state%vega_spec, ctx%state%sun_spec, ctx%state%magvega, ctx%state%magsun)
        if (allocated(ctx)) deallocate(ctx)
    end subroutine test_load_standard_sed

    subroutine test_load_lsf_data()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u, i

        allocate(ctx)
        call print_group("LSF Data Loading")
        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 50
        allocate(ctx%state%spec_lambda(50))
        allocate(ctx%state%lsfinfo%lsf(50))

        do i=1, 50
            ctx%state%spec_lambda(i) = 4000.0_wp + 20.0_wp * real(i-1, WP)
        end do

        file_path = trim(root) // '/data/lsf.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, *) 4000.0_wp, 10.0_wp
        write(u, *) 5000.0_wp, 20.0_wp
        close(u)

        call load_lsf_data(ctx)

        ! Check interpolation at index 25 (~4480A)
        ! Expected approx 14.8
        call assert_true(ctx%state%lsfinfo%lsf(25) > 10.0_wp .and. ctx%state%lsfinfo%lsf(25) < 20.0_wp, &
                         "LSF interpolated", total_tests, total_failures)

        deallocate(ctx%state%spec_lambda, ctx%state%lsfinfo%lsf)
        deallocate(ctx)
    end subroutine test_load_lsf_data

    subroutine test_attenuation_curves()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u, i, j

        allocate(ctx)
        call print_group("Attenuation Curves (WG00)")
        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data/dust')

        ctx%sps_home = trim(root)
        ctx%state%nspec = 10
        allocate(ctx%state%spec_lambda(10))
        ! (Lambda, 18 geom, 6 tau, 2 type)
        allocate(ctx%state%wgdust(10, 18, 6, 2)) 
        allocate(ctx%state%g03smcextn(10))

        ctx%state%spec_lambda = 5000.0_wp

        ! Mock alldirty_h.dat (Shell)
        file_path = trim(root) // '/data/dust/alldirty_h.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, *) 'header'
        write(u, *) 'header'
        write(u, *) 'header'
        do i=1, 18
            do j=1, 25
                ! Lam, Albedo, 6 Tau values
                write(u, *) 1000.0_wp + real(j,WP)*1000.0_wp, 0.5_wp, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            end do
        end do
        close(u)

        ! Mock alldirty_c.dat (Dusty) - same structure
        file_path = trim(root) // '/data/dust/alldirty_c.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        write(u, *) 'header'
        write(u, *) 'header'
        write(u, *) 'header'
        do i=1, 18
            do j=1, 25
                write(u, *) 1000.0_wp + real(j,WP)*1000.0_wp, 0.5_wp, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
            end do
        end do
        close(u)

        ! Mock Gordon03
        file_path = trim(root) // '/data/dust/Gordon03_table4.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        do i=1, 30
            write(u, *) 0.1_wp + 0.1_wp*real(i,WP), 0.0_wp, 1.0_wp
        end do
        close(u)

        call load_attenuation_curves(ctx)

        ! If we didn't crash, and values are non-zero, success.
        call assert_true(ctx%state%wgdust(1, 1, 1, 1) > 0.0_wp, "WG dust loaded", total_tests, total_failures)

        deallocate(ctx%state%spec_lambda, ctx%state%wgdust, ctx%state%g03smcextn)
        deallocate(ctx)
    end subroutine test_attenuation_curves

    subroutine test_index_definitions()
        type(fsps_context_t), allocatable :: ctx
        character(len=256) :: root, file_path
        integer :: u, i

        allocate(ctx)
        call print_group("Spectral Index Definitions")
        call make_temp_root(root)
        call ensure_dir(trim(root) // '/data')

        ctx%sps_home = trim(root)
        ctx%state%nindx = 30
        allocate(ctx%state%indexdefined(6, 30))

        file_path = trim(root) // '/data/allindices.dat'
        open(newunit=u, file=trim(file_path), status='replace', action='write')
        do i=1, 4
            write(u, *) 'header'
        end do
        do i=1, 30
            ! Simple band definitions
            write(u, *) 4000.0, 4010.0, 4020.0, 4030.0, 4040.0, 4050.0
        end do
        close(u)

        call load_index_definitions(ctx)

        ! Lick index 1 (Air) should be shifted to Vacuum (> 4000.0)
        call assert_true(ctx%state%indexdefined(1, 1) > 4000.0_wp, &
                         "Lick index air-to-vac conversion", total_tests, total_failures)
        
        ! Index 30 (Vacuum already) should remain identical
        call assert_float_equals(4000.0_wp, ctx%state%indexdefined(1, 30), 1.0e-6_wp, &
                                 "High index untouched", total_tests, total_failures)

        deallocate(ctx%state%indexdefined)
        deallocate(ctx)
    end subroutine test_index_definitions

    ! ------------------------------------------------------------------------
    ! Helpers
    ! ------------------------------------------------------------------------
    subroutine setup_iso_context(ctx, root, zlegend)
        type(fsps_context_t), intent(out) :: ctx
        character(len=*), intent(in) :: root
        real(WP), intent(in) :: zlegend

        ctx%sps_home = trim(root)
        ctx%state%nz = 1
        ctx%state%nt = 1
        ctx%state%zsol = 0.0190_wp

        allocate(ctx%state%mini_isoc(1, 1, NM))
        allocate(ctx%state%mact_isoc(1, 1, NM))
        allocate(ctx%state%logl_isoc(1, 1, NM))
        allocate(ctx%state%logt_isoc(1, 1, NM))
        allocate(ctx%state%logg_isoc(1, 1, NM))
        allocate(ctx%state%ffco_isoc(1, 1, NM))
        allocate(ctx%state%phase_isoc(1, 1, NM))
        allocate(ctx%state%lmdot_isoc(1, 1, NM))
        allocate(ctx%state%nmass_isoc(1, 1))
        allocate(ctx%state%timestep_isoc(1, 1))
        allocate(ctx%state%zlegend(1))

        ctx%state%mini_isoc = 0.0_wp
        ctx%state%mact_isoc = 0.0_wp
        ctx%state%logl_isoc = 0.0_wp
        ctx%state%logt_isoc = 0.0_wp
        ctx%state%logg_isoc = 0.0_wp
        ctx%state%ffco_isoc = 0.0_wp
        ctx%state%phase_isoc = 0.0_wp
        ctx%state%lmdot_isoc = 0.0_wp
        ctx%state%nmass_isoc = 0
        ctx%state%timestep_isoc = 0.0_wp
        ctx%state%zlegend(1) = zlegend
    end subroutine setup_iso_context

    subroutine teardown_iso_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        if (associated(ctx%state%mini_isoc)) deallocate(ctx%state%mini_isoc)
        if (associated(ctx%state%mact_isoc)) deallocate(ctx%state%mact_isoc)
        if (associated(ctx%state%logl_isoc)) deallocate(ctx%state%logl_isoc)
        if (associated(ctx%state%logt_isoc)) deallocate(ctx%state%logt_isoc)
        if (associated(ctx%state%logg_isoc)) deallocate(ctx%state%logg_isoc)
        if (associated(ctx%state%ffco_isoc)) deallocate(ctx%state%ffco_isoc)
        if (associated(ctx%state%phase_isoc)) deallocate(ctx%state%phase_isoc)
        if (associated(ctx%state%lmdot_isoc)) deallocate(ctx%state%lmdot_isoc)
        if (associated(ctx%state%nmass_isoc)) deallocate(ctx%state%nmass_isoc)
        if (associated(ctx%state%timestep_isoc)) deallocate(ctx%state%timestep_isoc)
        if (associated(ctx%state%zlegend)) deallocate(ctx%state%zlegend)
    end subroutine teardown_iso_context

    subroutine teardown_bpass_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%spec_nu)) deallocate(ctx%state%spec_nu)
        if (associated(ctx%state%time_full)) deallocate(ctx%state%time_full)
        if (associated(ctx%state%bpass_mass_ssp)) deallocate(ctx%state%bpass_mass_ssp)
        if (associated(ctx%state%bpass_spec_ssp)) deallocate(ctx%state%bpass_spec_ssp)
    end subroutine teardown_bpass_context

    subroutine teardown_filter_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%bands)) deallocate(ctx%state%bands)
        if (associated(ctx%state%filter_leff)) deallocate(ctx%state%filter_leff)
    end subroutine teardown_filter_context

    subroutine teardown_nebular_context(ctx)
        type(fsps_context_t), intent(inout) :: ctx
        if (associated(ctx%state%spec_lambda)) deallocate(ctx%state%spec_lambda)
        if (associated(ctx%state%nebem_cont)) deallocate(ctx%state%nebem_cont)
    end subroutine teardown_nebular_context

    subroutine ensure_dir(path)
        character(len=*), intent(in) :: path
        call execute_command_line('mkdir -p "' // trim(path) // '"')
    end subroutine ensure_dir

    subroutine make_temp_root(path)
        character(len=*), intent(out) :: path
        integer :: status, length
        character(len=256) :: value

        call get_environment_variable('TMPDIR', value, length=length, status=status)
        if (status == 0 .and. length > 0) then
            path = trim(value(1:length)) // '/fsps_io_tests'
        else
            path = '/tmp/fsps_io_tests'
        end if
        call ensure_dir(path)
    end subroutine make_temp_root

    !> The small standard-format test library: FSPS_TEST_LIBRARY (set by meson), else
    !> <source root>/tests/data/fsps-test-data.
    subroutine get_test_library_root(path)
        character(len=*), intent(out) :: path
        integer :: status, length
        character(len=512) :: value
        call get_environment_variable('FSPS_TEST_LIBRARY', value, length=length, status=status)
        if (status == 0 .and. length > 0) then
            path = trim(value(1:length))
        else
            call get_source_root(path)
            path = trim(path) // '/tests/data/fsps-test-data'
        end if
    end subroutine get_test_library_root

    subroutine get_source_root(path)
        character(len=*), intent(out) :: path
        integer :: status, length
        character(len=256) :: value

        call get_environment_variable('FSPS_HOME', value, length=length, status=status)
        if (status == 0 .and. length > 0) then
            path = trim(value(1:length))
            return
        end if

        call get_environment_variable('MESON_SOURCE_ROOT', value, length=length, status=status)
        if (status == 0 .and. length > 0) then
            path = trim(value(1:length))
            return
        end if

        call get_environment_variable('PWD', value, length=length, status=status)
        if (status == 0 .and. length > 0) then
            path = trim(value(1:length))
        else
            path = '.'
        end if
    end subroutine get_source_root

    real(WP) pure function spec_value(i_s, i_t, i_g) result(val)
        integer, intent(in) :: i_s, i_t, i_g
        val = 1.2345_wp + 0.1_wp * real(i_s - 1, WP) + 0.01_wp * real(i_t - 1, WP) + 0.001_wp * real(i_g - 1, WP)
    end function spec_value

    integer function count_fields(line) result(n_fields)
        character(len=*), intent(in) :: line
        integer :: i, len_line
        logical :: in_field

        n_fields = 0
        in_field = .false.
        len_line = len_trim(line)

        do i = 1, len_line
            if (line(i:i) /= ' ' .and. line(i:i) /= char(9)) then
                if (.not. in_field) then
                    n_fields = n_fields + 1
                    in_field = .true.
                end if
            else
                in_field = .false.
            end if
        end do
    end function count_fields

end module test_fsps_io_mod
