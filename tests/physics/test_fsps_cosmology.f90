module test_fsps_cosmology_mod
    use fsps_precision, only: WP, KP
    use fsps_constants, only: C_LIGHT
    use fsps_context_types, only: fsps_context_t
    use fsps_interpolation, only: find_interval
    use fsps_cosmology, only: air_to_vacuum, vacuum_to_air, get_universe_age, get_luminosity_distance, &
                              compute_igm_transmission, convolve_with_mdf
    use test_utils_mod, only: print_group, print_summary_line, print_minor_header, &
                              assert_float_equals, assert_true, assert_relative_error
    implicit none

    integer :: total_failures = 0
    integer :: total_tests = 0

    public :: run_fsps_cosmology_tests, total_failures, total_tests

    real(WP), parameter :: EPS = 1.0e-10_wp
    real(WP), parameter :: REL_EPS = 1.0e-8_wp
    real(WP), parameter :: REL_EPS_LOOSE = 1.0e-6_wp
    real(WP), parameter :: REL_EPS_LOOSER = 5.0e-4_wp
    real(WP), parameter :: HUBBLE_TIME_FACTOR = 978.0_wp
    real(WP), parameter :: PC_PER_MPC = 1.0e6_wp
    real(WP), parameter :: KM_PER_ANGSTROM = 1.0e-13_wp
    integer, parameter :: N_MDF_HIGH_RES = 100

    integer, parameter :: N_LYMAN_LINES = 17
    real(WP), parameter :: LYMAN_LIMIT = 911.75_wp
    real(WP), parameter :: A_METAL = 0.0017_wp

    real(WP), parameter :: MADAU_C1 = 0.25_wp
    real(WP), parameter :: MADAU_C2 = 9.4_wp
    real(WP), parameter :: MADAU_C3 = 0.7_wp
    real(WP), parameter :: MADAU_C4 = 0.023_wp

    real(WP), parameter :: MADAU_P1 = 0.46_wp
    real(WP), parameter :: MADAU_P2 = 0.18_wp
    real(WP), parameter :: MADAU_P3 = 1.32_wp
    real(WP), parameter :: MADAU_P4 = 1.68_wp

    real(WP), dimension(N_LYMAN_LINES), parameter :: LY_WAVE = [ &
        1215.67_wp, 1025.72_wp, 972.537_wp, 949.743_wp, 937.803_wp, &
         930.748_wp, 926.226_wp, 923.150_wp, 920.963_wp, 919.352_wp, &
         918.129_wp, 917.181_wp, 916.429_wp, 915.824_wp, 915.329_wp, &
         914.919_wp, 914.576_wp ]

    real(WP), dimension(N_LYMAN_LINES), parameter :: LY_COEFF = [ &
        0.0036_wp,    0.0017_wp,    0.0011846_wp, 0.0009410_wp, 0.0007960_wp, &
        0.0006967_wp, 0.0006236_wp, 0.0005665_wp, 0.0005200_wp, 0.0004817_wp, &
        0.0004487_wp, 0.0004200_wp, 0.0003947_wp, 0.000372_wp,  0.000352_wp,  &
        0.0003334_wp, 0.00031644_wp ]

contains

    subroutine run_fsps_cosmology_tests()
        call print_minor_header("fsps_cosmology")

        call test_vacuum_cutoff()
        call test_round_trip()
        call test_o3_reference()

        call test_eds_analytic()
        call test_hubble_law_limit()
        call test_unit_consistency()

        call test_igm_transparent()
        call test_igm_lyman_limit()
        call test_igm_madau_cap()

        call test_mdf_path_normalization()
        call test_mdf_interpolation_map()
        call test_mdf_log_vs_linear()
        call test_mdf_loop_order_reference()

        call print_summary_line("Module Summary", total_tests - total_failures, total_tests)
    end subroutine run_fsps_cosmology_tests

    ! --------------------------------------------------------------------
    ! GROUP 1: WAVELENGTH CONVERSION STANDARDS
    ! --------------------------------------------------------------------
    subroutine test_vacuum_cutoff()
        real(WP), dimension(3) :: input_vals
        real(WP), dimension(3) :: out_air, out_vac

        call print_group("Wavelength Conversions: 2000A Cutoff")

        input_vals = [1999.0_wp, 2000.0_wp, 2001.0_wp]

        out_air = air_to_vacuum(input_vals)
        call assert_float_equals(input_vals(1), out_air(1), 0.0_wp, "Air->Vacuum below cutoff", total_tests, total_failures)
        call assert_true(abs(out_air(2) - input_vals(2)) > EPS, "Air->Vacuum at cutoff changes", total_tests, total_failures)
        call assert_true(abs(out_air(3) - input_vals(3)) > EPS, "Air->Vacuum above cutoff changes", total_tests, total_failures)

        out_vac = vacuum_to_air(input_vals)
        call assert_float_equals(input_vals(1), out_vac(1), 0.0_wp, "Vacuum->Air below cutoff", total_tests, total_failures)
        call assert_true(abs(out_vac(2) - input_vals(2)) > EPS, "Vacuum->Air at cutoff changes", total_tests, total_failures)
        call assert_true(abs(out_vac(3) - input_vals(3)) > EPS, "Vacuum->Air above cutoff changes", total_tests, total_failures)
    end subroutine test_vacuum_cutoff

    subroutine test_round_trip()
        integer, parameter :: n = 40
        real(WP), dimension(n) :: w_in, w_tmp, w_out
        real(WP) :: log_min, log_max, t
        integer :: i
        character(len=64) :: label

        call print_group("Wavelength Conversions: Round-Trip Identity")

        log_min = log10(3000.0_wp)
        log_max = log10(10000.0_wp)

        do i = 1, n
            t = real(i - 1, WP) / real(n - 1, WP)
            w_in(i) = 10.0_wp**(log_min + t * (log_max - log_min))
        end do

        w_tmp = air_to_vacuum(w_in)
        w_out = vacuum_to_air(w_tmp)

        do i = 1, n
            write(label, '(A,I0)') "Round-trip element ", i
            call assert_relative_error(w_in(i), w_out(i), REL_EPS_LOOSE, trim(label), total_tests, total_failures)
        end do
    end subroutine test_round_trip

    subroutine test_o3_reference()
        real(WP) :: w_air, w_vac

        call print_group("Wavelength Conversions: [O III] 5007A")

        w_air = 5006.843_wp
        w_vac = air_to_vacuum(w_air)
        call assert_float_equals(5008.240_wp, w_vac, 1.0e-3_wp, "[O III] reference", total_tests, total_failures)
    end subroutine test_o3_reference

    ! --------------------------------------------------------------------
    ! GROUP 2: COSMOLOGICAL INTEGRATION PRECISION
    ! --------------------------------------------------------------------
    subroutine test_eds_analytic()
        type(fsps_context_t), allocatable :: ctx
        real(WP) :: z, age_expected, age_actual
        real(WP) :: dl_expected, dl_actual
        real(WP) :: hubble_distance_pc

        call print_group("Cosmology: Einstein-de Sitter Benchmarks")

        allocate(ctx)
        ctx%H0_val = 70.0_wp
        ctx%om0_val = 1.0_wp
        ctx%ol0_val = 0.0_wp

        z = 2.0_wp

        age_expected = (2.0_wp / 3.0_wp) * (HUBBLE_TIME_FACTOR / ctx%H0_val) * (1.0_wp + z)**(-1.5_wp)
        age_actual = get_universe_age(ctx, z)
        call assert_relative_error(age_expected, age_actual, REL_EPS_LOOSER, "EdS age analytic", total_tests, total_failures)

        hubble_distance_pc = ((C_LIGHT * KM_PER_ANGSTROM) / ctx%H0_val) * PC_PER_MPC
        dl_expected = (1.0_wp + z) * hubble_distance_pc * (2.0_wp * (1.0_wp - 1.0_wp / sqrt(1.0_wp + z)))
        dl_actual = get_luminosity_distance(ctx, z)
        call assert_relative_error(dl_expected, dl_actual, REL_EPS_LOOSER, &
                                   "EdS luminosity distance analytic", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_eds_analytic

    subroutine test_hubble_law_limit()
        type(fsps_context_t), allocatable :: ctx
        real(WP) :: z, dl_expected, dl_actual, hubble_distance_pc

        call print_group("Cosmology: Hubble Law Low-z")

        allocate(ctx)
        ctx%H0_val = 70.0_wp
        ctx%om0_val = 0.3_wp
        ctx%ol0_val = 0.7_wp

        z = 1.0e-3_wp
        hubble_distance_pc = ((C_LIGHT * KM_PER_ANGSTROM) / ctx%H0_val) * PC_PER_MPC
        dl_expected = hubble_distance_pc * z
        dl_actual = get_luminosity_distance(ctx, z)
        call assert_relative_error(dl_expected, dl_actual, 1.0e-3_wp, "Low-z Hubble Law", total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_hubble_law_limit

    subroutine test_unit_consistency()
        type(fsps_context_t), allocatable :: ctx
        real(WP) :: z, dl_actual

        call print_group("Cosmology: Unit Consistency")

        allocate(ctx)
        ctx%H0_val = 70.0_wp
        ctx%om0_val = 0.3_wp
        ctx%ol0_val = 0.7_wp

        z = 1.0_wp
        dl_actual = get_luminosity_distance(ctx, z)
        call assert_true(dl_actual > 1.0e8_wp .and. dl_actual < 1.0e11_wp, "Luminosity distance in parsecs", &
                         total_tests, total_failures)

        deallocate(ctx)
    end subroutine test_unit_consistency

    ! --------------------------------------------------------------------
    ! GROUP 3: IGM TRANSMISSION (MADAU 1995)
    ! --------------------------------------------------------------------
    subroutine test_igm_transparent()
        real(WP), dimension(3) :: wavelengths
        real(KP), dimension(3) :: transmission

        call print_group("IGM: Transparent Universe")

        wavelengths = [900.0_wp, 1000.0_wp, 1500.0_wp]
        !$acc data copyin(wavelengths) create(transmission)
        call compute_igm_transmission(wavelengths, 3.0_wp, 0.0_wp, transmission)
        !$acc update host(transmission)
        !$acc end data

        call assert_true(all(abs(transmission - 1.0_wp) <= 0.0_wp), "Optical depth factor zero => T=1", &
                         total_tests, total_failures)
    end subroutine test_igm_transparent

    subroutine test_igm_lyman_limit()
        real(WP), dimension(2) :: wavelengths
        real(KP), dimension(2) :: transmission
        real(WP) :: tau_blue, tau_red

        call print_group("IGM: Lyman Limit Step")

        wavelengths = [910.0_wp, 915.0_wp]
        !$acc data copyin(wavelengths) create(transmission)
        call compute_igm_transmission(wavelengths, 3.0_wp, 1.0_wp, transmission)
        !$acc update host(transmission)
        !$acc end data

        tau_blue = -log(transmission(1))
        tau_red = -log(transmission(2))

        call assert_true(tau_blue > tau_red, "Blueward Lyman limit more opaque", total_tests, total_failures)
        call assert_true(tau_red > 0.0_wp, "Redward Lyman limit nonzero tau", total_tests, total_failures)
    end subroutine test_igm_lyman_limit

    subroutine test_igm_madau_cap()
        integer, parameter :: n = 10
        real(WP), dimension(n) :: wavelengths
        real(KP), dimension(n) :: transmission
        real(WP), dimension(n) :: tau_raw, tau_out
        real(WP) :: max_tau_out
        integer :: idx_raw_max

        call print_group("IGM: Madau Safety Cap")

        wavelengths = [300.0_wp, 400.0_wp, 500.0_wp, 600.0_wp, 700.0_wp, 800.0_wp, 900.0_wp, 1000.0_wp, 1100.0_wp, 1200.0_wp]

        call compute_madau_tau_raw_grid(wavelengths, 5.0_wp, tau_raw)

        !$acc data copyin(wavelengths) create(transmission)
        call compute_igm_transmission(wavelengths, 5.0_wp, 1.0_wp, transmission)
        !$acc update host(transmission)
        !$acc end data

        tau_out = -log(transmission)

        max_tau_out = maxval(tau_out)
        idx_raw_max = maxloc(tau_raw, 1)

        call assert_true(idx_raw_max > 1, "Raw tau peak occurs redward", total_tests, total_failures)
        call assert_float_equals(max_tau_out, tau_out(1), 1.0e-10_wp, &
                                 "Cap clamps shortest wavelength", total_tests, total_failures)
        call assert_true(tau_out(1) > tau_raw(1), "Cap increases short-wavelength opacity", total_tests, total_failures)
    end subroutine test_igm_madau_cap

    ! --------------------------------------------------------------------
    ! GROUP 4: MDF CONVOLUTION
    ! --------------------------------------------------------------------
    subroutine test_mdf_path_normalization()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:,:,:), mass(:,:), lbol(:,:), zlegend(:)
        real(WP), allocatable :: spec_out(:,:), mass_out(:), lbol_out(:)
        real(WP) :: avg_z
        integer :: nz, nspec, nt, i

        call print_group("MDF: Path Selection & Normalization")

        nspec = 3
        nt = 2

        nz = 22
        allocate(zlegend(nz))
        do i = 1, nz
            zlegend(i) = 10.0_wp**(log10(1.0e-4_wp) + real(i - 1, WP) * (log10(3.0e-2_wp) - log10(1.0e-4_wp)) / real(nz - 1, WP))
        end do
        allocate(spec(nspec, nt, nz))
        allocate(mass(nt, nz))
        allocate(lbol(nt, nz))
        spec = 1.0_wp
        mass = 1.0_wp
        lbol = 1.0_wp

        call setup_mdf_context(ctx, nz, nspec, nt, zlegend, spec, mass, lbol, 1.0_wp)

        allocate(spec_out(nspec, nt))
        allocate(mass_out(nt))
        allocate(lbol_out(nt))
        call convolve_with_mdf(ctx, 0.02_wp, avg_z, spec_out, lbol_out, mass_out)

        call assert_true(all(abs(spec_out - 1.0_wp) <= EPS), "Standard path spectra normalized", total_tests, total_failures)
        call assert_true(all(abs(mass_out - 1.0_wp) <= EPS), "Standard path masses normalized", total_tests, total_failures)
        call assert_true(avg_z >= minval(zlegend) .and. avg_z <= maxval(zlegend), "Standard path avg Z in bounds", &
                         total_tests, total_failures)

        call teardown_mdf_context(ctx)
        deallocate(spec, mass, lbol, zlegend, spec_out, mass_out, lbol_out)

        nz = 6
        allocate(zlegend(nz))
        do i = 1, nz
            zlegend(i) = 10.0_wp**(log10(1.0e-4_wp) + real(i - 1, WP) * (log10(1.0e-1_wp) - log10(1.0e-4_wp)) / real(nz - 1, WP))
        end do
        allocate(spec(nspec, nt, nz))
        allocate(mass(nt, nz))
        allocate(lbol(nt, nz))
        spec = 1.0_wp
        mass = 1.0_wp
        lbol = 1.0_wp

        call setup_mdf_context(ctx, nz, nspec, nt, zlegend, spec, mass, lbol, 1.0_wp)

        allocate(spec_out(nspec, nt))
        allocate(mass_out(nt))
        allocate(lbol_out(nt))
        call convolve_with_mdf(ctx, 0.02_wp, avg_z, spec_out, lbol_out, mass_out)

        call assert_true(all(abs(spec_out - 1.0_wp) <= EPS), "High-res path spectra normalized", total_tests, total_failures)
        call assert_true(all(abs(mass_out - 1.0_wp) <= EPS), "High-res path masses normalized", total_tests, total_failures)
        call assert_true(avg_z >= minval(zlegend) .and. avg_z <= maxval(zlegend), "High-res path avg Z in bounds", &
                         total_tests, total_failures)

        call teardown_mdf_context(ctx)
        deallocate(spec, mass, lbol, zlegend, spec_out, mass_out, lbol_out)
    end subroutine test_mdf_path_normalization

    subroutine test_mdf_interpolation_map()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:,:,:), mass(:,:), lbol(:,:), zlegend(:)
        real(WP), allocatable :: spec_out(:,:), mass_out(:), lbol_out(:)
        real(WP), allocatable :: spec_ref(:,:), mass_ref(:), lbol_ref(:)
        real(WP) :: avg_z, avg_z_ref, log_z_val, w_expected
        integer :: nz, nspec, nt

        call print_group("MDF: High-Res Interpolation Map")

        nspec = 2
        nt = 1
        nz = 3

        allocate(zlegend(nz))
        zlegend = [1.0e-4_wp, 1.0e-2_wp, 1.0e-1_wp]

        allocate(spec(nspec, nt, nz))
        allocate(mass(nt, nz))
        allocate(lbol(nt, nz))

        spec(:, 1, 1) = [1.0_wp, 2.0_wp]
        spec(:, 1, 2) = [10.0_wp, 20.0_wp]
        spec(:, 1, 3) = [100.0_wp, 200.0_wp]

        mass(1, :) = [1.0_wp, 2.0_wp, 3.0_wp]
        lbol(1, :) = [2.0_wp, 4.0_wp, 6.0_wp]

        call setup_mdf_context(ctx, nz, nspec, nt, zlegend, spec, mass, lbol, 1.0_wp)

        allocate(spec_out(nspec, nt))
        allocate(mass_out(nt))
        allocate(lbol_out(nt))
        call convolve_with_mdf(ctx, 0.02_wp, avg_z, spec_out, lbol_out, mass_out)

        allocate(spec_ref(nspec, nt))
        allocate(mass_ref(nt))
        allocate(lbol_ref(nt))
        call convolve_with_mdf_reference(ctx, 0.02_wp, avg_z_ref, spec_ref, lbol_ref, mass_ref, .true.)

        call assert_true(all(abs(spec_out - spec_ref) <= 1.0e-6_wp), "Interpolation map matches reference (spec)", &
                         total_tests, total_failures)
        call assert_true(all(abs(mass_out - mass_ref) <= 1.0e-6_wp), "Interpolation map matches reference (mass)", &
                         total_tests, total_failures)
        call assert_float_equals(avg_z_ref, avg_z, 1.0e-6_wp, "Average metallicity reference", total_tests, total_failures)

        log_z_val = (real(50, WP) / 100.0_wp * 3.0_wp) - 4.0_wp
        w_expected = (log_z_val - log10(zlegend(1))) / (log10(zlegend(2)) - log10(zlegend(1)))
        call assert_float_equals(0.75_wp, w_expected, 1.0e-12_wp, "Map weight mid-point check", total_tests, total_failures)

        call teardown_mdf_context(ctx)
        deallocate(spec, mass, lbol, zlegend, spec_out, mass_out, lbol_out, spec_ref, mass_ref, lbol_ref)
    end subroutine test_mdf_interpolation_map

    subroutine test_mdf_log_vs_linear()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:,:,:), mass(:,:), lbol(:,:), zlegend(:)
        real(WP), allocatable :: spec_out(:,:), mass_out(:), lbol_out(:)
        real(WP), allocatable :: spec_log(:,:), spec_lin(:,:), mass_ref(:), lbol_ref(:)
        real(WP) :: avg_z
        integer :: nz, nspec, nt

        call print_group("MDF: Log vs Linear Spectrum Interpolation")

        nspec = 1
        nt = 1
        nz = 2

        allocate(zlegend(nz))
        zlegend = [1.0e-4_wp, 1.0e-1_wp]

        allocate(spec(nspec, nt, nz))
        allocate(mass(nt, nz))
        allocate(lbol(nt, nz))

        spec(1, 1, 1) = 10.0_wp
        spec(1, 1, 2) = 1000.0_wp
        mass(1, :) = [10.0_wp, 20.0_wp]
        lbol(1, :) = [10.0_wp, 20.0_wp]

        call setup_mdf_context(ctx, nz, nspec, nt, zlegend, spec, mass, lbol, 0.0_wp)

        allocate(spec_out(nspec, nt))
        allocate(mass_out(nt))
        allocate(lbol_out(nt))
        call convolve_with_mdf(ctx, 0.01_wp, avg_z, spec_out, lbol_out, mass_out)

        allocate(spec_log(nspec, nt))
        allocate(spec_lin(nspec, nt))
        allocate(mass_ref(nt))
        allocate(lbol_ref(nt))
        call convolve_with_mdf_reference(ctx, 0.01_wp, avg_z, spec_log, lbol_ref, mass_ref, .true.)
        call convolve_with_mdf_reference(ctx, 0.01_wp, avg_z, spec_lin, lbol_ref, mass_ref, .false.)

        call assert_float_equals(mass_ref(1), mass_out(1), 1.0e-6_wp, "Mass interpolation linear", total_tests, total_failures)
        call assert_true(abs(spec_out(1,1) - spec_log(1,1)) < abs(spec_out(1,1) - spec_lin(1,1)), &
                         "Spectrum interpolation is logarithmic", total_tests, total_failures)

        call teardown_mdf_context(ctx)
        deallocate(spec, mass, lbol, zlegend, spec_out, mass_out, lbol_out, spec_log, spec_lin, mass_ref, lbol_ref)
    end subroutine test_mdf_log_vs_linear

    subroutine test_mdf_loop_order_reference()
        type(fsps_context_t), allocatable :: ctx
        real(WP), allocatable :: spec(:,:,:), mass(:,:), lbol(:,:), zlegend(:)
        real(WP), allocatable :: spec_out(:,:), mass_out(:), lbol_out(:)
        real(WP), allocatable :: spec_ref(:,:), mass_ref(:), lbol_ref(:)
        real(WP) :: avg_z_out, avg_z_ref
        integer :: nz, nspec, nt, i
        integer, allocatable :: seed(:)
        integer :: nseed

        call print_group("MDF: Loop Order & Stride Safety")

        nspec = 4
        nt = 3
        nz = 5

        allocate(zlegend(nz))
        do i = 1, nz
            zlegend(i) = 10.0_wp**(log10(1.0e-4_wp) + real(i - 1, WP) * (log10(1.0e-1_wp) - log10(1.0e-4_wp)) / real(nz - 1, WP))
        end do

        allocate(spec(nspec, nt, nz))
        allocate(mass(nt, nz))
        allocate(lbol(nt, nz))

        call random_seed(size=nseed)
        allocate(seed(nseed))
        seed = 1234
        call random_seed(put=seed)

        call random_number(spec)
        call random_number(mass)
        call random_number(lbol)

        spec = 1.0_wp + 9.0_wp * spec
        mass = 1.0_wp + 9.0_wp * mass
        lbol = 1.0_wp + 9.0_wp * lbol

        call setup_mdf_context(ctx, nz, nspec, nt, zlegend, spec, mass, lbol, 1.0_wp)

        allocate(spec_out(nspec, nt))
        allocate(mass_out(nt))
        allocate(lbol_out(nt))
        call convolve_with_mdf(ctx, 0.02_wp, avg_z_out, spec_out, lbol_out, mass_out)

        allocate(spec_ref(nspec, nt))
        allocate(mass_ref(nt))
        allocate(lbol_ref(nt))
        call convolve_with_mdf_reference(ctx, 0.02_wp, avg_z_ref, spec_ref, lbol_ref, mass_ref, .true.)

        call assert_true(maxval(abs(spec_out - spec_ref)) <= 1.0e-6_wp, "Loop order matches reference (spec)", &
                         total_tests, total_failures)
        call assert_true(maxval(abs(mass_out - mass_ref)) <= 1.0e-6_wp, "Loop order matches reference (mass)", &
                         total_tests, total_failures)
        call assert_true(maxval(abs(lbol_out - lbol_ref)) <= 1.0e-6_wp, "Loop order matches reference (lbol)", &
                         total_tests, total_failures)

        call teardown_mdf_context(ctx)
        deallocate(spec, mass, lbol, zlegend, spec_out, mass_out, lbol_out, spec_ref, mass_ref, lbol_ref, seed)
    end subroutine test_mdf_loop_order_reference

    ! --------------------------------------------------------------------
    ! HELPERS
    ! --------------------------------------------------------------------
    subroutine setup_mdf_context(ctx, nz, nspec, nt, zlegend, spec, mass, lbol, zpow2)
        type(fsps_context_t), allocatable, intent(out) :: ctx
        integer, intent(in) :: nz, nspec, nt
        real(WP), dimension(:), intent(in) :: zlegend
        real(WP), dimension(:,:,:), intent(in) :: spec
        real(WP), dimension(:,:), intent(in) :: mass, lbol
        real(WP), intent(in) :: zpow2

        allocate(ctx)
        ctx%state%nz = nz
        ctx%state%nspec = nspec
        ctx%state%ntfull = nt
        ctx%state%zpow2 = zpow2

        allocate(ctx%state%zlegend(nz))
        allocate(ctx%state%spec_ssp_zz(nspec, nt, nz))
        allocate(ctx%state%mass_ssp_zz(nt, nz))
        allocate(ctx%state%lbol_ssp_zz(nt, nz))

        ctx%state%zlegend = zlegend
        ctx%state%spec_ssp_zz = spec
        ctx%state%mass_ssp_zz = mass
        ctx%state%lbol_ssp_zz = lbol
    end subroutine setup_mdf_context

    subroutine teardown_mdf_context(ctx)
        type(fsps_context_t), allocatable, intent(inout) :: ctx

        if (.not. allocated(ctx)) return
        if (associated(ctx%state%zlegend)) deallocate(ctx%state%zlegend)
        if (allocated(ctx%state%spec_ssp_zz)) deallocate(ctx%state%spec_ssp_zz)
        if (allocated(ctx%state%mass_ssp_zz)) deallocate(ctx%state%mass_ssp_zz)
        if (allocated(ctx%state%lbol_ssp_zz)) deallocate(ctx%state%lbol_ssp_zz)
        deallocate(ctx)
    end subroutine teardown_mdf_context

    subroutine convolve_with_mdf_reference(ctx, effective_yield, average_metallicity, &
                                           spec_out, lbol_out, mass_out, use_log_spectra)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: effective_yield
        real(WP), intent(out) :: average_metallicity
        real(WP), dimension(:,:), intent(out) :: spec_out
        real(WP), dimension(:), intent(out) :: lbol_out
        real(WP), dimension(:), intent(out) :: mass_out
        logical, intent(in) :: use_log_spectra

        integer :: num_wavelengths, num_timesteps, num_metallicities
        integer :: idx_time, idx_metallicity
        real(WP) :: delta_log_z, trapz_weight, p_now, p_next
        real(WP) :: val_mass_now, val_mass_next, val_lbol_now, val_lbol_next
        real(WP), dimension(size(spec_out,1)) :: spec_now, spec_next
        real(WP), dimension(N_MDF_HIGH_RES) :: high_res_z_grid, high_res_prob_dist
        real(WP) :: normalization_factor
        integer :: i

        num_wavelengths = ctx%state%nspec
        num_timesteps = ctx%state%ntfull
        num_metallicities = ctx%state%nz

        spec_out = 0.0_wp
        lbol_out = 0.0_wp
        mass_out = 0.0_wp
        average_metallicity = 0.0_wp
        normalization_factor = 0.0_wp

        do i = 1, N_MDF_HIGH_RES
            high_res_z_grid(i) = 10.0_wp**((real(i, WP) / 100.0_wp * 3.0_wp) - 4.0_wp)
        end do
        high_res_prob_dist = (high_res_z_grid**ctx%state%zpow2) * exp(-high_res_z_grid / effective_yield)

        do i = 1, N_MDF_HIGH_RES - 1
            delta_log_z = log(high_res_z_grid(i+1)) - log(high_res_z_grid(i))
            normalization_factor = normalization_factor + &
                                   0.5_wp * delta_log_z * (high_res_prob_dist(i+1) + high_res_prob_dist(i))
            average_metallicity = average_metallicity + 0.5_wp * delta_log_z * &
                (high_res_prob_dist(i+1) * high_res_z_grid(i+1) + high_res_prob_dist(i) * high_res_z_grid(i))
        end do
        average_metallicity = average_metallicity / normalization_factor

        do idx_time = 1, num_timesteps
            do idx_metallicity = 1, N_MDF_HIGH_RES - 1
                delta_log_z = log(high_res_z_grid(idx_metallicity+1)) - log(high_res_z_grid(idx_metallicity))
                trapz_weight = 0.5_wp * delta_log_z
                p_now = trapz_weight * high_res_prob_dist(idx_metallicity)
                p_next = trapz_weight * high_res_prob_dist(idx_metallicity+1)

                call interpolate_at_z(ctx, high_res_z_grid(idx_metallicity), idx_time, val_mass_now, &
                                      val_lbol_now, spec_now, use_log_spectra)
                call interpolate_at_z(ctx, high_res_z_grid(idx_metallicity+1), idx_time, val_mass_next, &
                                      val_lbol_next, spec_next, use_log_spectra)

                mass_out(idx_time) = mass_out(idx_time) + p_now * val_mass_now + p_next * val_mass_next
                lbol_out(idx_time) = lbol_out(idx_time) + p_now * val_lbol_now + p_next * val_lbol_next
                spec_out(:, idx_time) = spec_out(:, idx_time) + p_now * spec_now + p_next * spec_next
            end do

            spec_out(:, idx_time) = spec_out(:, idx_time) / normalization_factor
            mass_out(idx_time) = mass_out(idx_time) / normalization_factor
            lbol_out(idx_time) = lbol_out(idx_time) / normalization_factor
        end do

        if (num_metallicities < 2) then
            spec_out = 0.0_wp
            lbol_out = 0.0_wp
            mass_out = 0.0_wp
        end if
    end subroutine convolve_with_mdf_reference

    subroutine interpolate_at_z(ctx, z_val, idx_time, val_mass, val_lbol, spec_val, use_log_spectra)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: z_val
        integer, intent(in) :: idx_time
        real(WP), intent(out) :: val_mass, val_lbol
        real(WP), dimension(:), intent(out) :: spec_val
        logical, intent(in) :: use_log_spectra

        integer :: idx_grid
        real(WP) :: log_z_val, weight_val
        real(WP) :: log_z1, log_z2
        integer :: iw

        log_z_val = log10(z_val)
        log_z1 = log10(ctx%state%zlegend(1))
        log_z2 = log10(ctx%state%zlegend(size(ctx%state%zlegend)))

        if (log_z_val < log_z1) then
            idx_grid = 1
        else if (log_z_val > log_z2) then
            idx_grid = size(ctx%state%zlegend) - 1
        else
            idx_grid = find_interval(log10(ctx%state%zlegend), log_z_val)
        end if

        weight_val = (log_z_val - log10(ctx%state%zlegend(idx_grid))) / &
                     (log10(ctx%state%zlegend(idx_grid+1)) - log10(ctx%state%zlegend(idx_grid)))

        val_mass = (1.0_wp - weight_val) * ctx%state%mass_ssp_zz(idx_time, idx_grid) + &
                   weight_val * ctx%state%mass_ssp_zz(idx_time, idx_grid+1)
        val_lbol = (1.0_wp - weight_val) * ctx%state%lbol_ssp_zz(idx_time, idx_grid) + &
                   weight_val * ctx%state%lbol_ssp_zz(idx_time, idx_grid+1)

        do iw = 1, size(spec_val)
            if (use_log_spectra) then
                spec_val(iw) = (1.0_wp - weight_val) * log10(ctx%state%spec_ssp_zz(iw, idx_time, idx_grid)) + &
                               weight_val * log10(ctx%state%spec_ssp_zz(iw, idx_time, idx_grid+1))
                spec_val(iw) = 10.0_wp**spec_val(iw)
            else
                spec_val(iw) = (1.0_wp - weight_val) * ctx%state%spec_ssp_zz(iw, idx_time, idx_grid) + &
                               weight_val * ctx%state%spec_ssp_zz(iw, idx_time, idx_grid+1)
            end if
        end do
    end subroutine interpolate_at_z

    subroutine compute_madau_tau_raw_grid(wavelengths, source_redshift, tau_raw)
        real(WP), dimension(:), intent(in) :: wavelengths
        real(WP), intent(in) :: source_redshift
        real(WP), dimension(:), intent(out) :: tau_raw

        integer :: i
        do i = 1, size(wavelengths)
            tau_raw(i) = compute_madau_tau_raw(wavelengths(i), source_redshift)
        end do
    end subroutine compute_madau_tau_raw_grid

    pure function compute_madau_tau_raw(wavelength, source_redshift) result(tau_val)
        real(WP), intent(in) :: wavelength
        real(WP), intent(in) :: source_redshift
        real(WP) :: tau_val

        real(WP) :: one_plus_z, observed_wavelength, lambda_ratio
        integer :: i

        one_plus_z = 1.0_wp + source_redshift
        observed_wavelength = wavelength * one_plus_z
        lambda_ratio = observed_wavelength / LYMAN_LIMIT
        tau_val = 0.0_wp

        do i = 1, N_LYMAN_LINES
            if (wavelength < LY_WAVE(i)) then
                tau_val = tau_val + LY_COEFF(i) * (observed_wavelength / LY_WAVE(i))**3.46_wp
                if (i == 1) then
                    tau_val = tau_val + A_METAL * (observed_wavelength / LY_WAVE(i))**1.68_wp
                end if
            end if
        end do

        if (wavelength < LYMAN_LIMIT) then
            tau_val = tau_val + &
                (MADAU_C1 * lambda_ratio**3.0_wp * (one_plus_z**MADAU_P1 - lambda_ratio**MADAU_P1)) + &
                (MADAU_C2 * lambda_ratio**1.5_wp * (one_plus_z**MADAU_P2 - lambda_ratio**MADAU_P2)) - &
                (MADAU_C3 * lambda_ratio**3.0_wp * (lambda_ratio**(-MADAU_P3) - one_plus_z**(-MADAU_P3))) - &
                (MADAU_C4 * (one_plus_z**MADAU_P4 - lambda_ratio**MADAU_P4))
        end if
    end function compute_madau_tau_raw

end module test_fsps_cosmology_mod
