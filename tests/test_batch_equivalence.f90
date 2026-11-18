program test_batch_equivalence
    !> Any combination of free parameters gives the same numbers as each model alone (hot paths 3).
    !>
    !> The batched model has several paths chosen by which parameters differ among the models: uniform or per-model kernels
    !> for the redshift and the IGM, the resident basis or the operator with the group operators for the SSP, shared or
    !> per-model IMF weights. A batch of N models must give each model what a batch of that model alone gives, whichever
    !> path each of the two takes:
    !>  * the reference of model j is a batch of one on a resident basis built for the SSP parameters of that model (the
    !>    IMF, the isochrone factors, the TP-AGB shifts), exactly as the mode tests of test_csp_batch do;
    !>  * a batch with only closed-form parameters free (class C of opensps_batch_registry) runs the resident basis in both
    !>    (N against 1: the kernels chosen by the free set), one with an SSP-level parameter free runs the operator path
    !>    against the resident basis.
    !> Cases: every field that can be free alone (registry_range), seeded random subsets of 2-6 fields, and all fields
    !> together. The values of a field spread over its range across the models (a different order for each field); the grouped
    !> fields (delt, dell) take two values (a batch holds a few distinct pairs). Metallicity and [alpha/Fe] free per model need
    !> the bases of every metallicity and are covered by the mode tests of test_csp_batch.
    use opensps_precision, only: WP, KP
    use opensps_types, only: PARAMS
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_csp_batch, only: compute_csp_batch
    use opensps_ssp, only: generate_ssp_grid
    use opensps_ssp_batch, only: ssp_opset_t, ssp_opset_free, ssp_opset_build_mods
    use opensps_batch_registry, only: N_FIELDS, registry_name, registry_class, registry_range, registry_set, registry_get, &
                                      CLASS_S, F_LOGZSOL, F_AFE, F_FBHB, F_SBSS, F_AGB, F_PAGB, F_REDGB, F_DELT, F_DELL, &
                                      F_IMF1, F_IMF2, F_IMF3, F_VDMC, F_MDAVE, F_IMF_LOWER_LIMIT, F_IMF_UPPER_LIMIT
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_move_to_device, &
                            opensps_context_set_pset, opensps_context_get_pset, opensps_context_prepare_pset, &
                            opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use opensps_tuning, only: tuning_set, TUNE_P_SHARE_IMF_MIN_BATCH
    use test_utils_mod, only: tap_finish, skip_without_data, print_major_header, print_group, print_summary_line, assert_true
    implicit none

    integer, parameter :: NB = 5
    integer :: total_tests = 0
    integer :: total_failures = 0
    ! As test_csp_batch: the operator path against the resident basis, and a batch against a batch of one
    real(WP), parameter :: SPEC_RTOL = merge(1.0e-9_wp, 100.0_wp * real(epsilon(1.0_KP), WP), KP == WP)
    real(WP), parameter :: MAG_TOL = merge(1.0e-9_wp, 1.0e-5_wp, KP == WP)
    real(WP), parameter :: SCAL_TOL = merge(1.0e-9_wp, 1.0e-6_wp, KP == WP)
    type(opensps_context_t), allocatable :: ctx
    type(ssp_opset_t) :: op
    integer :: cases(N_FIELDS, 64), ncase(64), n_cases = 0
    integer(8) :: seed = 20261117_8
    logical :: tuning_ok
    integer :: c

    call skip_without_data()
    call print_major_header("Batch equivalence: any free set")
    call tuning_set(TUNE_P_SHARE_IMF_MIN_BATCH, 0, tuning_ok)
    if (.not. tuning_ok) error stop 'tuning_set failed'

    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 10
    call opensps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call opensps_context_move_to_device(ctx)
    ! A configuration with every stage on, so that every closed-form field acts: an exponentially declining SFH with dust
    ! (both components), dust emission, nebular emission, IGM absorption at a redshift, and an AGN torus
    call set_int('zmet', 10)
    call set_int('imf_type', 2)
    call set_int('sfh', 1)
    call set_float('tau', 1.5_wp)
    call set_float('tage', 6.0_wp)
    call set_float('dust1', 0.4_wp)
    call set_float('dust2', 0.3_wp)
    call set_int('add_dust_emission', 1)
    call set_int('add_neb_emission', 1)
    call set_int('add_igm_absorption', 1)
    call set_float('zred', 0.5_wp)
    call set_int('add_agn_dust', 1)
    call set_float('fagn', 0.05_wp)
    call set_int('compute_indices', 0)
    ! The velocity smoothing is on (sigma_smooth can be free), with the nebular lines out of the spectrum: their profile table is
    ! built once per batch from one sigma
    call set_int('smooth_velocity', 1)
    call set_int('smoothspec_fast', 1)
    call set_int('nebemlineinspec', 0)
    call build_basis()
    call base_operators()

    call make_cases()
    do c = 1, n_cases
        call run_case(cases(1:ncase(c), c))
    end do

    call ssp_opset_free(op)
    call opensps_destroy(ctx)
    deallocate(ctx)
    call print_summary_line("Batch equivalence", total_tests - total_failures, total_tests)
    call tap_finish()
    if (total_failures > 0) stop 1

contains

    !> The fields that can be free here: those with a range, not metallicity or [alpha/Fe] (see the program header)
    logical function can_vary(idx)
        integer, intent(in) :: idx
        real(WP) :: lo, hi

        call registry_range(idx, lo, hi)
        can_vary = hi > lo .and. idx /= F_LOGZSOL .and. idx /= F_AFE .and. registry_class(idx) /= CLASS_S
    end function can_vary

    !> The fields of the SSP stage that the free-modification operators and the IMF weights serve (the batch takes the operator
    !> path for them); the closed-form fields run on the resident basis
    logical function ssp_level(idx)
        integer, intent(in) :: idx

        select case (idx)
        case (F_FBHB, F_SBSS, F_AGB, F_PAGB, F_REDGB, F_DELT, F_DELL, F_IMF1, F_IMF2, F_IMF3, F_VDMC, F_MDAVE, &
             F_IMF_LOWER_LIMIT, F_IMF_UPPER_LIMIT)
            ssp_level = .true.
        case default
            ssp_level = .false.
        end select
    end function ssp_level

    !> A pseudo-random integer in 1..n (Park-Miller, 64-bit arithmetic: the same cases on every build)
    integer function next_int(n)
        integer, intent(in) :: n

        seed = mod(seed * 48271_8, 2147483647_8)
        next_int = int(mod(seed, int(n, 8))) + 1
    end function next_int

    subroutine make_cases()
        integer :: idx, k, j, sz, pick, tries
        integer :: pool(N_FIELDS), npool
        logical :: used(N_FIELDS)

        npool = 0
        do idx = 1, N_FIELDS
            if (can_vary(idx)) then
                npool = npool + 1
                pool(npool) = idx
            end if
        end do
        ! each alone
        do k = 1, npool
            n_cases = n_cases + 1
            ncase(n_cases) = 1
            cases(1, n_cases) = pool(k)
        end do
        ! seeded random subsets of 2-6 fields (at most as many cases as the table holds)
        do j = 1, 20
            sz = 2 + mod(j, 5)
            used = .false.
            n_cases = n_cases + 1
            ncase(n_cases) = 0
            tries = 0
            do while (ncase(n_cases) < sz .and. tries < 100)
                tries = tries + 1
                pick = pool(next_int(npool))
                if (used(pick)) cycle
                used(pick) = .true.
                ncase(n_cases) = ncase(n_cases) + 1
                cases(ncase(n_cases), n_cases) = pick
            end do
        end do
        ! all together (the grouped fields included, the IMF type shared)
        n_cases = n_cases + 1
        ncase(n_cases) = 0
        do k = 1, npool
            ncase(n_cases) = ncase(n_cases) + 1
            cases(ncase(n_cases), n_cases) = pool(k)
        end do
        if (n_cases > 64) error stop 'too many cases'
    end subroutine make_cases

    !> The batch of one case against a batch of each model alone
    subroutine run_case(fl)
        integer, intent(in) :: fl(:)
        type(PARAMS) :: base, psets(NB)
        real(WP), allocatable :: spec(:, :), mags(:, :), ref_spec(:, :), ref_mags(:, :), s1(:, :), g1(:, :)
        real(WP) :: mass(NB), lbol(NB), ref_mass(NB), ref_lbol(NB), lo, hi, u, d_spec, d_mag, d_scal, peak
        integer :: j, k, i, status, nspec, nbands
        logical :: ssp
        character(len=200) :: label
        character(len=32) :: nm

        nspec = ctx%state%nspec
        nbands = ctx%state%nbands
        allocate(spec(nspec, NB), mags(nbands, NB), ref_spec(nspec, NB), ref_mags(nbands, NB), s1(nspec, 1), g1(nbands, 1))
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, base)
        ! the label: the fields
        label = ''
        ssp = .false.
        do k = 1, size(fl)
            nm = registry_name(fl(k))
            if (len_trim(label) + len_trim(nm) + 2 < len(label)) label = trim(label) // ' ' // trim(nm)
            if (ssp_level(fl(k))) ssp = .true.
        end do
        call print_group("Batch equivalence, free:" // trim(label))

        do j = 1, NB
            psets(j) = base
            do k = 1, size(fl)
                call registry_range(fl(k), lo, hi)
                ! a different order of the models for each field: the fields are not correlated
                u = real(mod((j - 1) * (2 + mod(fl(k), 3)) + fl(k), NB), WP) / real(NB - 1, WP)
                if (fl(k) == F_DELT .or. fl(k) == F_DELL) u = merge(0.0_wp, 1.0_wp, mod(j, 2) == 0)
                call registry_set(psets(j)%params_core, fl(k), lo + (hi - lo) * u)
            end do
        end do

        ! the batch of N: the operator path when an SSP-level parameter is free, else the resident basis
        if (ssp) then
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol, ssp_ops=op)
        else
            call compute_csp_batch(ctx, psets, mags, status, spec=spec, mass=mass, lbol=lbol)
        end if
        call assert_true(status == 0, trim(label) // ": status 0", total_tests, total_failures)
        if (status /= 0) then
            deallocate(spec, mags, ref_spec, ref_mags, s1, g1)
            return
        end if

        ! each model alone: its SSP parameters on the context, a basis built for them, a batch of one
        do j = 1, NB
            if (ssp) then
                do k = 1, size(fl)
                    if (ssp_level(fl(k))) call set_field(fl(k), registry_get(psets(j)%params_core, fl(k)))
                end do
                call build_basis()
            end if
            call compute_csp_batch(ctx, psets(j:j), g1, status, spec=s1, mass=ref_mass(j:j), lbol=ref_lbol(j:j))
            if (status /= 0) then
                call assert_true(.false., trim(label) // ": the batch of one model alone: status 0", total_tests, total_failures)
                exit
            end if
            ref_spec(:, j) = s1(:, 1)
            ref_mags(:, j) = g1(:, 1)
        end do
        if (ssp) then
            do k = 1, size(fl)
                if (ssp_level(fl(k))) call set_field(fl(k), registry_get(base%params_core, fl(k)))
            end do
            call build_basis()
        end if
        if (status /= 0) then
            deallocate(spec, mags, ref_spec, ref_mags, s1, g1)
            return
        end if

        d_spec = 0.0_wp
        d_mag = 0.0_wp
        do j = 1, NB
            peak = maxval(ref_spec(:, j))
            do i = 1, nspec
                if (ref_spec(i, j) > 1.0e-6_wp * peak) d_spec = max(d_spec, abs(spec(i, j) / ref_spec(i, j) - 1.0_wp))
            end do
            do i = 1, nbands
                ! (bands within 30 mag of the brightest: the fainter ones carry numerically zero flux)
                if (ref_mags(i, j) == ref_mags(i, j) .and. ref_mags(i, j) < minval(ref_mags(:, j)) + 30.0_wp) &
                    d_mag = max(d_mag, abs(mags(i, j) - ref_mags(i, j)))
            end do
        end do
        d_scal = max(maxval(abs(mass / ref_mass - 1.0_wp)), maxval(abs(lbol - ref_lbol)))
        write(*, '(A,3ES10.2)') '#   max difference (spectrum, mag, mass/lbol): ', d_spec, d_mag, d_scal
        call assert_true(d_spec <= SPEC_RTOL, trim(label) // ": spectra the same as each model alone", total_tests, total_failures)
        call assert_true(d_mag <= MAG_TOL, trim(label) // ": magnitudes the same", total_tests, total_failures)
        call assert_true(d_scal <= SCAL_TOL, trim(label) // ": mass and L_bol the same", total_tests, total_failures)
        deallocate(spec, mags, ref_spec, ref_mags, s1, g1)
    end subroutine run_case

    !> The group operators of the context's SSP parameters (the free-modification set: the factors, the shifts and the IMF
    !> weights all vary through it)
    subroutine base_operators()
        type(PARAMS) :: pset

        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        call ssp_opset_build_mods(ctx, pset, .false., op)
    end subroutine base_operators

    !> The resident basis for the SSP parameters now on the context
    subroutine build_basis()
        type(PARAMS) :: pset
        real(WP), allocatable :: mass_ssp(:), lbol_ssp(:), spec_ssp(:, :)

        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        allocate(mass_ssp(ctx%state%ntfull), lbol_ssp(ctx%state%ntfull), spec_ssp(ctx%state%nspec, ctx%state%ntfull))
        call opensps_context_set_pset(ctx, pset)
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
        call opensps_context_update_ssp_basis(ctx, reshape(spec_ssp, [size(spec_ssp, 1), size(spec_ssp, 2), 1]), &
                                           reshape(mass_ssp, [size(mass_ssp), 1]), &
                                           reshape(lbol_ssp, [size(lbol_ssp), 1]), 1)
    end subroutine build_basis

    !> Sets a context parameter by registry index (real fields only: the SSP-level ones are)
    subroutine set_field(idx, v)
        integer, intent(in) :: idx
        real(WP), intent(in) :: v
        call set_float(trim(registry_name(idx)), v)
    end subroutine set_field

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        integer :: status

        call opensps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) then
            write(*, '(A,A)') 'FAIL: set_int ', key
            error stop 1
        end if
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        integer :: status

        call opensps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) then
            write(*, '(A,A)') 'FAIL: set_float ', key
            error stop 1
        end if
    end subroutine set_float

end program test_batch_equivalence
