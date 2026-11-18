program benchmark_csp_batch
    !> Per-model cost of the batched CSP (compute_csp_batch) against one compute_csp_scenario
    !> call per model, on the benchmark_phases setup (MIST + MILES + DL07, tau model with dust and
    !> dust emission). Tau, age and the dust parameters vary across the batch.
    !> Usage: benchmark_csp_batch [n_max [n_single [config [n_min]]]] (batch sizes 1, 100, 1000,
    !> 10^4 from n_min to n_max; n_min = 10^4 with n_single = 0 times one size, e.g. for profiling;
    !> OPENSPS_BENCH_SIZES=16,64,256 replaces the list of sizes, ascending, e.g. to find the crossover of a tuning parameter).
    !> config = "full": the real-world
    !> fitting configuration (dust1 = 0.5, dust2 = 0.3, DL07 dust emission, nebular emission,
    !> IGM at zred = 0.5, AGN with fagn = 0.1, spectral indices; tau, age, U_min and gas_logu vary);
    !> "full_smooth": the same with velocity smoothing, sigma_smooth = 150 km/s; "full_z" and
    !> "full_smooth_z": with the bases of all metallicities resident and logzsol varying per model;
    !> any config containing "_fit": a fitting population (zred, gas_logz, dust2, dust1, dust_index, fagn vary too);
    !> any config ending in "_imf": the IMF free per model (Kroupa, imf1-3 varied; the batch
    !> from the template operator, built once and timed); any config with "_grad": also the
    !> gradients (params_bar, with cotangents of the magnitudes, the spectra, mass and lbol); any config
    !> with "_cue": cue's nebular emission (nebular_model = 1, OPENSPS_CUE_DATA; gas_logn, gas_logno and
    !> gas_logco vary too), "_cue64" with its hidden layers in FP64 (cue_fp64 = 1; the default is the kernel precision).
    !> Matrix mode: OPENSPS_BENCH_FREE=<field>[,<field>...] (registry names, opensps_batch_registry; "-" for none, "list" to print
    !> the fields it can vary) makes exactly those fields free on top of the config's own variation (tau, tage, duste_umin and
    !> gas_logu in "full"), and derives the setup from their classes: logzsol loads every metallicity, a weights (IMF) or
    !> linear-mixture (isochrone factor) or grouped (delt, dell) field takes the operator path, the factors and shifts the
    !> group operators. OPENSPS_BENCH_DIST=uniform (default) | cluster (four tight clusters) | onecell (every model the same)
    !> sets how the free fields spread over their range. A combination the batch does not support prints
    !> "matrix_status <status>" instead of timings (tools/bench_matrix.py runs the matrix).
    !> Prints one line per metric: name, value: one compute_csp_scenario call per model, one
    !> all-ages call (tage = 0, every age of the grid) and, per batch size: magnitudes only; with spectra
    !> copied back to a host array (reused, as in a fitting loop); with spectra left on the device;
    !> with "_grad", the forward with spectra and the gradients (batch_N_grad_us_per_model).

    use opensps_precision, only: WP
    use opensps_types, only: PARAMS, COMPSPOUT
    use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
    use opensps_csp, only: compute_csp_scenario
    use opensps_csp_batch, only: compute_csp_batch, batch_spectra_device, batch_spectra_host, NBGRAD
    use, intrinsic :: iso_c_binding, only: c_ptr, c_associated
    use opensps_ssp, only: generate_ssp_grid
    use opensps_context, only: opensps_context_update_ssp_basis, opensps_context_move_to_device, &
                            opensps_context_set_pset, opensps_context_get_pset, opensps_context_prepare_pset, &
                            opensps_context_set_param_int, opensps_context_set_param_float
    use opensps_context_types, only: opensps_context_t
    use opensps_api, only: opensps_build_afe_basis
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_build
    use opensps_ssp_batch, only: ssp_opset_t, ssp_opset_build, ssp_opset_build_mods, ssp_batch_share_imf
    use opensps_batch_registry, only: registry_find, registry_set, registry_name, registry_class, N_FIELDS, CLASS_C, CLASS_L, &
                                      CLASS_W, CLASS_G, CLASS_S, F_LOGZSOL, F_AFE, F_FBHB, F_SBSS, F_AGB, F_PAGB, F_REDGB, &
                                      F_DELT, F_DELL, F_IMF1, F_IMF2, F_IMF3, F_VDMC, F_MDAVE
    use opensps_tuning, only: tuning_ct_count, tuning_ct_name, tuning_ct_value
    implicit none

    type(opensps_context_t), allocatable :: ctx
    type(PARAMS) :: pset
    type(PARAMS), allocatable :: psets(:)
    type(COMPSPOUT), allocatable :: results(:)
    real(WP), allocatable :: spec_ssp(:, :), mass_ssp(:), lbol_ssp(:), mags(:, :), spec(:, :)
    character(len=32) :: arg, config = 'basic'
    logical :: afe = .false., lib = .false., full, zvar = .false., imf = .false., grad = .false., fit = .false., cue = .false.
    logical :: mods = .false., imfs = .false., mods_build = .false., matrix = .false.
    character(len=8) :: noshare
    !> Matrix mode: the registry indices of the free fields, the distribution, the IMF type of the free IMF field
    integer :: free_idx(N_FIELDS) = 0, nfree = 0, matrix_imf_type = 2
    character(len=16) :: dist = 'uniform'
    real(WP), allocatable :: mbar(:, :), sbar(:, :), pbar(:, :), massv(:), lbolv(:), massb(:), lbolb(:)
    type(ssp_operator_t), allocatable :: ops(:)
    type(ssp_opset_t) :: opset
    real(WP), allocatable :: spec_z(:, :, :), mass_z(:, :), lbol_z(:, :)
    integer :: z
    character(len=64) :: bench_isoc = '', bench_spec = ''
    real(WP), parameter :: afe_grid(5) = [-0.2_wp, 0.0_wp, 0.2_wp, 0.4_wp, 0.6_wp]
    real(WP), allocatable :: indx(:, :)
    integer(8) :: t0, t1, rate
    integer :: n_max = 10000, n_single = 200, n_min = 1, status, nspec, ntfull, i, n, rep, n_rep, k
    integer, allocatable :: sizes(:)
    character(len=256) :: size_list
    integer :: size_len, size_stat, nsz, p0, p1
    real(WP) :: dt, total
    type(c_ptr) :: dptr
    integer :: k_nspec, k_n, k_bytes

    call get_command_argument(1, arg)
    if (len_trim(arg) > 0) read(arg, *) n_max
    call get_command_argument(2, arg)
    if (len_trim(arg) > 0) read(arg, *) n_single
    call get_command_argument(3, arg)
    if (len_trim(arg) > 0) config = arg
    call get_command_argument(4, arg)
    if (len_trim(arg) > 0) read(arg, *) n_min
    ! Batch sizes: 1, 100, 1000, 10^4, or the ascending comma-separated list of OPENSPS_BENCH_SIZES (e.g. to sweep
    ! the crossover of a tuning parameter: OPENSPS_BENCH_SIZES=16,64,256,1024)
    sizes = [1, 100, 1000, 10000]
    call get_environment_variable('OPENSPS_BENCH_SIZES', size_list, length=size_len, status=size_stat)
    if (size_stat == 0 .and. size_len > 0) then
        deallocate(sizes)
        allocate(sizes(0))
        p0 = 1
        do while (p0 <= size_len)
            p1 = index(size_list(p0:size_len), ',')
            p1 = merge(size_len, p0 + p1 - 2, p1 == 0)
            read (size_list(p0:p1), *) nsz
            sizes = [sizes, nsz]
            p0 = p1 + 2
        end do
    end if
    full =index(config, 'full') == 1
    afe = index(config, '_afe') > 0
    lib = index(config, '_lib') > 0     ! the [alpha/Fe] libraries at [alpha/Fe] = 0 only (the 2-slab path), for comparison
    zvar = index(config, '_z') > 0 .or. afe .or. lib
    imf = index(config, '_imf') > 0
    ! "_mods" (with "_imf"): the isochrone modification factors fbhb, sbss, agb, pagb, redgb free per model too
    ! (the group operators)
    mods = index(config, '_mods') > 0 .and. imf
    ! "_imfs" (also an "_imf" config): the operator path with every model on the same IMF (imf1-3 fixed), which the
    ! forward pass computes once per chunk; OPENSPS_BENCH_NOSHARE=1 turns that off (the IMF weights of every model, as before)
    imfs = index(config, '_imfs') > 0
    call get_environment_variable('OPENSPS_BENCH_NOSHARE', noshare)
    if (len_trim(noshare) > 0 .and. trim(noshare) /= '0') ssp_batch_share_imf = .false.
    grad = index(config, '_grad') > 0
    fit = index(config, '_fit') > 0
    cue = index(config, '_cue') > 0
    call parse_free()
    call system_clock(count_rate=rate)

    allocate(ctx)
    call opensps_create(ctx)
    ctx%pset%zmet = 10
    ! (zin = 0: the isochrones and libraries of every metallicity, for the configs that vary it;
    ! with zin = 10 the other metallicities would be empty)
    if (afe) then
        ! The [alpha/Fe] stack of the real libraries (the names from OPENSPS_BENCH_ISOC and _SPEC, default mist_afe and
        ! c3k_lr), built on the host and moved to the device with the context
        call get_environment_variable('OPENSPS_BENCH_ISOC', bench_isoc)
        call get_environment_variable('OPENSPS_BENCH_SPEC', bench_spec)
        if (len_trim(bench_isoc) == 0) bench_isoc = 'mist_afe'
        if (len_trim(bench_spec) == 0) bench_spec = 'c3k_lr'
        call opensps_setup(ctx, 0, isoc_type_in=trim(bench_isoc)//':afe=0.0', spec_type_in=trim(bench_spec)//':afe=0.0', &
                           dust_type_in='DL07')
        call set_int('imf_type', 1)
        call opensps_build_afe_basis(ctx, trim(bench_isoc), trim(bench_spec), afe_grid, status)
        if (status /= 0) stop 1
    else if (lib) then
        call get_environment_variable('OPENSPS_BENCH_ISOC', bench_isoc)
        call get_environment_variable('OPENSPS_BENCH_SPEC', bench_spec)
        if (len_trim(bench_isoc) == 0) bench_isoc = 'mist_afe'
        if (len_trim(bench_spec) == 0) bench_spec = 'c3k_lr'
        call opensps_setup(ctx, 0, isoc_type_in=trim(bench_isoc)//':afe=0.0', spec_type_in=trim(bench_spec)//':afe=0.0', &
                           dust_type_in='DL07')
    else
        call opensps_setup(ctx, merge(0, 10, zvar), isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    end if
    call opensps_context_move_to_device(ctx)
    call set_int('imf_type', merge(matrix_imf_type, 1, imf))
    call set_int('zmet', 10)
    call set_int('sfh', 1)
    call set_float('tau', 1.0_wp)
    call set_float('tage', 10.0_wp)
    call set_float('dust2', 0.3_wp)
    call set_int('compute_indices', merge(1, 0, full))
    if (full) then
        call set_float('dust1', 0.5_wp)
        call set_int('add_neb_emission', 1)
        call set_int('add_igm_absorption', 1)
        call set_float('zred', 0.5_wp)
        call set_int('add_agn_dust', 1)
        call set_float('fagn', 0.1_wp)
    end if
    if (cue) then
        call set_int('nebular_model', 1)
        call set_int('cue_fp64', merge(1, 0, index(config, '_cue64') > 0))
    end if
    if (index(config, 'full_smooth') == 1) then
        call set_int('smooth_velocity', 1)
        call set_float('sigma_smooth', 150.0_wp)
    end if
    call opensps_context_prepare_pset(ctx)
    call opensps_context_get_pset(ctx, pset)

    nspec = ctx%state%nspec
    ntfull = ctx%state%ntfull
    allocate(spec_ssp(nspec, ntfull), mass_ssp(ntfull), lbol_ssp(ntfull))
    call opensps_context_set_pset(ctx, pset)
    write(*, '(A,A)') 'config ', trim(config)
    write(*, '(A,L1)') 'share_imf ', ssp_batch_share_imf
    do i = 1, tuning_ct_count()      ! the compile-time tuning parameters this build has (provenance of a sweep)
        write(*, '(A,A,A,I0)') 'tuning_ct_', trim(tuning_ct_name(i)), ' ', tuning_ct_value(i)
    end do
    if (.not. afe) then
        call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
        call opensps_context_update_ssp_basis(ctx, reshape(spec_ssp, [nspec, ntfull, 1]), &
                                           reshape(mass_ssp, [ntfull, 1]), reshape(lbol_ssp, [ntfull, 1]), 1)
    end if
    if (zvar .and. .not. imf .and. .not. afe) then
        ! The bases of all metallicities (the single call keeps the zmet = 10 basis)
        allocate(spec_z(nspec, ntfull, ctx%state%nz), mass_z(ntfull, ctx%state%nz), lbol_z(ntfull, ctx%state%nz))
        do z = 1, ctx%state%nz
            call set_int('zmet', z)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call opensps_context_set_pset(ctx, pset)
            call generate_ssp_grid(ctx, pset, mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
        end do
        call set_int('zmet', 10)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
    end if

    if (mods .or. mods_build) then
        call system_clock(t0)
        call ssp_opset_build_mods(ctx, pset, zvar, opset)
        call system_clock(t1)
        write(*, '(A,F14.3)') 'ssp_operator_mods_build_s ', real(t1 - t0, WP) / rate
    else if (imf .and. zvar) then
        ! One operator per metallicity (the metallicity free per model: ssp_ops)
        call system_clock(t0)
        allocate(ops(ctx%state%nz))
        do z = 1, ctx%state%nz
            call set_int('zmet', z)
            call opensps_context_prepare_pset(ctx)
            call opensps_context_get_pset(ctx, pset)
            call opensps_context_set_pset(ctx, pset)
            call ssp_operator_build(ctx, pset, ops(z))
        end do
        call set_int('zmet', 10)
        call opensps_context_prepare_pset(ctx)
        call opensps_context_get_pset(ctx, pset)
        call opensps_context_set_pset(ctx, pset)
        call ssp_opset_build(ops, opset)
        call system_clock(t1)
        write(*, '(A,F14.3)') 'ssp_operators_build_s ', real(t1 - t0, WP) / rate
    else if (imf) then
        call system_clock(t0)
        allocate(ops(1))
        call ssp_operator_build(ctx, pset, ops(1))
        call ssp_opset_build(ops, opset)
        call system_clock(t1)
        write(*, '(A,F14.3)') 'ssp_operator_build_s ', real(t1 - t0, WP) / rate
    end if

    ! One compute_csp_scenario call per model (the existing path)
    total = 0.0_wp
    call compute_csp_scenario(ctx, pset, 1, results, status)
    call system_clock(t0)
    do i = 1, n_single
        call vary(pset, i)
        call compute_csp_scenario(ctx, pset, 1, results, status)
        if (status /= 0) stop 1
        total = total + results(1)%mags(1)
    end do
    call system_clock(t1)
    if (n_single > 0) write(*, '(A,F14.3)') 'single_call_us_per_model ', 1.0e6_wp * real(t1 - t0, WP) / rate / n_single

    ! One all-ages call (tage = 0: every age of the grid), n_single / 20 times
    if (n_single >= 20) then
        pset%tage = 0.0_wp
        call compute_csp_scenario(ctx, pset, 1, results, status)
        call system_clock(t0)
        do i = 1, n_single / 20
            call compute_csp_scenario(ctx, pset, 1, results, status)
            if (status /= 0) stop 1
            total = total + results(size(results))%mags(1)
        end do
        call system_clock(t1)
        write(*, '(A,F14.3)') 'all_ages_call_us ', 1.0e6_wp * real(t1 - t0, WP) / rate / (n_single / 20)
        call vary(pset, 1)
    end if

    ! Batched, magnitudes only and with spectra
    if (zvar .and. .not. imf .and. .not. afe) then
        ctx%state%ssp_basis_is_dirty = .true.
        call opensps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, ctx%state%nz)
    end if
    allocate(psets(n_max))
    do i = 1, n_max
        psets(i) = pset
        call vary(psets(i), i)
    end do
    allocate(mags(ctx%state%nbands, n_max))
    if (full) allocate(indx(ctx%state%nindx, n_max))
    do k = 1, size(sizes)
        n = sizes(k)
        if (n > n_max) exit
        if (n < n_min) cycle
        n_rep = max(3, min(200, 20000 / n))
        call batch(n, .false.)   ! warm-up
        if (status /= 0 .and. matrix) then
            write(*, '(A,I0)') 'matrix_status ', status
            call opensps_destroy(ctx)
            stop 0
        end if
        if (status /= 0) stop 2
        call system_clock(t0)
        do rep = 1, n_rep
            call batch(n, .false.)
        end do
        call system_clock(t1)
        dt = real(t1 - t0, WP) / rate / n_rep
        write(*, '(A,I0,A,F14.3)') 'batch_', n, '_mags_us_per_model ', 1.0e6_wp * dt / n
        total = total + mags(1, 1)

        allocate(spec(nspec, n))
        call batch(n, .true.)   ! warm-up: first touch of spec (a fitting loop reuses its array)
        call system_clock(t0)
        do rep = 1, n_rep
            call batch(n, .true.)
        end do
        call system_clock(t1)
        dt = real(t1 - t0, WP) / rate / n_rep
        write(*, '(A,I0,A,F14.3)') 'batch_', n, '_spec_us_per_model ', 1.0e6_wp * dt / n
        total = total + spec(1000, 1)
        deallocate(spec)

        ! Spectra left on the device (callers working there): the batch without spec, then the
        ! device address of its spectra
        call system_clock(t0)
        do rep = 1, n_rep
            call batch(n, .false.)
            call batch_spectra_device(ctx, dptr, k_nspec, k_n, k_bytes)
        end do
        call system_clock(t1)
        dt = real(t1 - t0, WP) / rate / n_rep
        if (.not. c_associated(dptr) .or. k_n /= n) stop 3
        write(*, '(A,I0,A,F14.3)') 'batch_', n, '_spec_on_device_us_per_model ', 1.0e6_wp * dt / n

        ! Spectra on the host at kernel precision (page-locked buffer, no conversion)
        call system_clock(t0)
        do rep = 1, n_rep
            call batch(n, .false.)
            call batch_spectra_host(ctx, dptr, k_nspec, k_n, k_bytes)
        end do
        call system_clock(t1)
        dt = real(t1 - t0, WP) / rate / n_rep
        if (.not. c_associated(dptr) .or. k_n /= n) stop 4
        write(*, '(A,I0,A,F14.3)') 'batch_', n, '_spec_host_kp_us_per_model ', 1.0e6_wp * dt / n

        ! The gradients: cotangents of every output (fixed, O(1) per band, the spectra's scaled by
        ! their peak), params_bar
        if (grad) then
            allocate(spec(nspec, n), mbar(ctx%state%nbands, n), sbar(nspec, n), pbar(NBGRAD, n), massv(n), lbolv(n), &
                     massb(n), lbolb(n))
            call batch(n, .true.)
            do i = 1, n
                do rep = 1, ctx%state%nbands
                    mbar(rep, i) = merge(sin(real(rep + 7 * i, WP)), 0.0_wp, mags(rep, i) == mags(rep, i))
                end do
                sbar(:, i) = cos(real([(rep, rep = 1, nspec)], WP) + i) / (maxval(abs(spec(:, i))) * nspec)
                massb(i) = 0.3_wp
                lbolb(i) = -0.2_wp
            end do
            call grad_batch(n)   ! warm-up
            if (status /= 0) stop 5
            call dump_gradients(n)
            call system_clock(t0)
            do rep = 1, n_rep
                call grad_batch(n)
            end do
            call system_clock(t1)
            dt = real(t1 - t0, WP) / rate / n_rep
            write(*, '(A,I0,A,F14.3)') 'batch_', n, '_grad_us_per_model ', 1.0e6_wp * dt / n
            total = total + pbar(1, 1)
            deallocate(spec, mbar, sbar, pbar, massv, lbolv, massb, lbolb)
        end if
    end do
    if (total /= total) write(*, *) total   ! keep the results live

    call opensps_destroy(ctx)
    deallocate(ctx)

contains

    !> The batch of the first n models (magnitudes; spectra too if with_spec; indices in "full").
    subroutine batch(n, with_spec)
        integer, intent(in) :: n
        logical, intent(in) :: with_spec
        if (imf .and. full .and. with_spec) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, spec=spec, indx=indx, ssp_ops=opset)
        else if (imf .and. full) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, indx=indx, ssp_ops=opset)
        else if (imf .and. with_spec) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, spec=spec, ssp_ops=opset)
        else if (imf) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, ssp_ops=opset)
        else if (full .and. with_spec) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, spec=spec, indx=indx)
        else if (full) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, indx=indx)
        else if (with_spec) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, spec=spec)
        else
            call compute_csp_batch(ctx, psets(1:n), mags, status)
        end if
    end subroutine batch

    !> OPENSPS_BENCH_DUMP=<file>: the gradients (params_bar, NBGRAD x n, FP64) of the last batch size timed, to compare two
    !> builds (tools/compare_builds/compare_grad_dump.py). Each size overwrites the file.
    subroutine dump_gradients(n)
        integer, intent(in) :: n
        character(len=512) :: fname
        integer :: flen, u
        call get_environment_variable('OPENSPS_BENCH_DUMP', fname, flen)
        if (flen <= 0) return
        open(newunit=u, file=fname(1:flen), form='unformatted', access='stream', status='replace')
        write(u) NBGRAD, n
        write(u) real(pbar(1:NBGRAD, 1:n), WP)
        close(u)
    end subroutine dump_gradients

    !> The batch of the first n models with its gradients (spectra, mass and lbol, and params_bar).
    subroutine grad_batch(n)
        integer, intent(in) :: n
        if (imf) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, spec=spec, mass=massv, lbol=lbolv, ssp_ops=opset, &
                                   mags_bar=mbar, spec_bar=sbar, params_bar=pbar, mass_bar=massb, lbol_bar=lbolb)
        else
            call compute_csp_batch(ctx, psets(1:n), mags, status, spec=spec, mass=massv, lbol=lbolv, &
                                   mags_bar=mbar, spec_bar=sbar, params_bar=pbar, mass_bar=massb, lbol_bar=lbolb)
        end if
    end subroutine grad_batch

    !> Model i: tau, age and dust vary over a fitting-like range ("full": the dust is fixed and
    !> gas_logu varies instead).
    subroutine vary(p, i)
        type(PARAMS), intent(inout) :: p
        integer, intent(in) :: i
        p%tau = 0.3_wp + 0.05_wp * mod(i, 97)
        p%tage = 1.0_wp + 0.1_wp * mod(i, 113)
        p%duste_umin = 1.0_wp + 0.2_wp * mod(i, 71)
        if (zvar) p%logzsol = -1.5_wp + 0.017_wp * mod(i, 97)
        if (afe) p%afe = -0.2_wp + 0.8_wp * real(mod(i, 89), WP) / 88.0_wp
        if (imf .and. .not. imfs) then
            p%imf1 = 0.9_wp + 0.01_wp * mod(i, 61)
            p%imf2 = 1.8_wp + 0.01_wp * mod(i, 83)
            p%imf3 = 2.0_wp + 0.01_wp * mod(i, 79)
        end if
        if (mods) then
            ! (MIST's 2nd and 12th metallicities do not take the horizontal-branch modification: legacy FSPS stops there)
            if (.not. zvar) p%fbhb = 0.05_wp + 0.01_wp * mod(i, 71)
            if (.not. zvar) p%sbss = 1.0_wp + 0.1_wp * mod(i, 53)
            p%agb = 0.5_wp + 0.02_wp * mod(i, 61)
            p%pagb = 0.5_wp + 0.03_wp * mod(i, 47)
            p%redgb = 0.7_wp + 0.01_wp * mod(i, 59)
        end if
        if (fit) then
            ! A fitting population: everything differs between models (redshift, gas, dust, AGN)
            p%zred = 0.1_wp + 0.0235_wp * mod(i, 103)
            p%gas_logz = -1.5_wp + 0.03_wp * mod(i, 67)
            p%dust2 = 0.05_wp + 0.01_wp * mod(i, 89)
            p%dust1 = 0.5_wp * p%dust2
            p%dust_index = -1.0_wp + 0.01_wp * mod(i, 53)
            p%fagn = 0.001_wp + 0.005_wp * mod(i, 37)
        end if
        if (cue) then
            p%gas_logn = 1.0_wp + 0.05_wp * mod(i, 59)
            p%gas_logno = -0.9_wp + 0.03_wp * mod(i, 53)
            p%gas_logco = -0.9_wp + 0.03_wp * mod(i, 47)
        end if
        if (full) then
            p%gas_logu = -3.5_wp + 0.05_wp * mod(i, 41)
        else
            p%dust2 = 0.05_wp + 0.01_wp * mod(i, 89)
            p%dust1 = 0.02_wp * mod(i, 53)
        end if
        if (nfree > 0) call vary_free(p, i)
    end subroutine vary

    !> Matrix mode (OPENSPS_BENCH_FREE): reads the free fields and the distribution, and derives the setup of the benchmark
    !> from their classes (see the program header). "list" prints the fields that can be varied and stops.
    subroutine parse_free()
        character(len=512) :: lst
        integer :: llen, lstat, p0, p1, idx, k
        logical :: wfree, mfree

        call get_environment_variable('OPENSPS_BENCH_DIST', dist)
        if (len_trim(dist) == 0) dist = 'uniform'
        call get_environment_variable('OPENSPS_BENCH_FREE', lst, length=llen, status=lstat)
        if (lstat /= 0 .or. llen <= 0) return
        if (lst(1:llen) == 'list') then
            do k = 1, N_FIELDS
                if (registry_class(k) /= CLASS_S) write(*, '(A,A,I0)') trim(registry_name(k)), ' ', registry_class(k)
            end do
            stop 0
        end if
        matrix = .true.
        imfs = .true.         ! the free IMF fields come from vary_free, not from the config's ramp
        wfree = .false.
        mfree = .false.
        if (lst(1:llen) /= '-') then
            p0 = 1
            do while (p0 <= llen)
                p1 = index(lst(p0:llen), ',')
                p1 = merge(llen, p0 + p1 - 2, p1 == 0)
                idx = registry_find(lst(p0:p1))
                if (idx == 0 .or. registry_class(idx) == CLASS_S) then
                    write(*, '(A,A)') 'matrix_status unknown_field ', lst(p0:p1)
                    stop 0
                end if
                nfree = nfree + 1
                free_idx(nfree) = idx
                select case (idx)
                case (F_LOGZSOL); zvar = .true.
                case (F_AFE); afe = .true.; zvar = .true.
                case (F_IMF1, F_IMF2, F_IMF3); wfree = .true.; matrix_imf_type = 2
                case (F_VDMC); wfree = .true.; matrix_imf_type = 3
                case (F_MDAVE); wfree = .true.; matrix_imf_type = 4
                case (F_FBHB, F_SBSS, F_AGB, F_PAGB, F_REDGB, F_DELT, F_DELL); mfree = .true.
                end select
                p0 = p1 + 2
            end do
        end if
        imf = imf .or. wfree .or. mfree
        mods_build = mfree
        write(*, '(A,A)') 'matrix_free ', lst(1:llen)
        write(*, '(A,A)') 'matrix_dist ', trim(dist)
    end subroutine parse_free

    !> The fraction in [0, 1] of model i for field idx: spread over the range ('uniform', decorrelated between fields), in
    !> four tight clusters ('cluster') or the same for every model ('onecell')
    real(WP) function spread(i, idx)
        integer, intent(in) :: i, idx
        real(WP) :: u
        real(WP), parameter :: centers(4) = [0.2_wp, 0.45_wp, 0.7_wp, 0.9_wp]

        u = real(mod(i * (7919 + 13 * idx), 10007), WP) / 10007.0_wp
        select case (trim(dist))
        case ('cluster')
            spread = centers(mod(i * 3 + idx, 4) + 1) + 0.01_wp * (u - 0.5_wp)
        case ('onecell')
            spread = 0.5_wp
        case default
            spread = u
        end select
    end function spread

    !> The range of the fields that can be free in the benchmark's configuration (a field outside it keeps its value)
    subroutine field_range(idx, lo, hi)
        integer, intent(in) :: idx
        real(WP), intent(out) :: lo, hi
        character(len=16) :: nm

        nm = registry_name(idx)
        lo = 0.0_wp
        hi = 0.0_wp
        select case (trim(nm))
        case ('tau'); lo = 0.3_wp; hi = 5.0_wp
        case ('tage'); lo = 1.0_wp; hi = 12.0_wp
        case ('const'); lo = 0.0_wp; hi = 0.4_wp
        case ('fburst'); lo = 0.0_wp; hi = 0.2_wp
        case ('tburst'); lo = 1.0_wp; hi = 3.0_wp
        case ('sf_start'); lo = 0.0_wp; hi = 1.0_wp
        case ('dust1'); lo = 0.0_wp; hi = 1.5_wp
        case ('dust2'); lo = 0.0_wp; hi = 1.0_wp
        case ('dust3'); lo = 0.0_wp; hi = 0.3_wp
        case ('dust_index'); lo = -1.0_wp; hi = 0.0_wp
        case ('dust1_index'); lo = -1.5_wp; hi = -0.5_wp
        case ('frac_nodust'); lo = 0.0_wp; hi = 0.3_wp
        case ('dust_tesc'); lo = 6.5_wp; hi = 7.5_wp
        case ('duste_gamma'); lo = 0.001_wp; hi = 0.2_wp
        case ('duste_umin'); lo = 0.5_wp; hi = 20.0_wp
        case ('duste_qpah'); lo = 0.5_wp; hi = 6.0_wp
        case ('gas_logu'); lo = -4.0_wp; hi = -1.0_wp
        case ('gas_logz'); lo = -1.5_wp; hi = 0.4_wp
        case ('frac_obrun'); lo = 0.0_wp; hi = 0.3_wp
        case ('fagn'); lo = 0.0_wp; hi = 0.5_wp
        case ('agn_tau'); lo = 5.0_wp; hi = 150.0_wp
        case ('zred'); lo = 0.1_wp; hi = 2.0_wp
        case ('igm_factor'); lo = 0.5_wp; hi = 1.5_wp
        case ('logzsol'); lo = -1.5_wp; hi = 0.05_wp
        case ('afe'); lo = -0.2_wp; hi = 0.6_wp
        case ('fbhb'); lo = 0.05_wp; hi = 0.75_wp
        case ('sbss'); lo = 1.0_wp; hi = 6.0_wp
        case ('agb'); lo = 0.5_wp; hi = 1.7_wp
        case ('pagb'); lo = 0.5_wp; hi = 1.9_wp
        case ('redgb'); lo = 0.7_wp; hi = 1.3_wp
        case ('delt'); lo = -0.1_wp; hi = 0.1_wp
        case ('dell'); lo = -0.1_wp; hi = 0.1_wp
        case ('imf1'); lo = 0.9_wp; hi = 1.5_wp
        case ('imf2'); lo = 1.8_wp; hi = 2.6_wp
        case ('imf3'); lo = 2.0_wp; hi = 2.8_wp
        case ('vdmc'); lo = 0.05_wp; hi = 0.15_wp
        case ('mdave'); lo = 0.3_wp; hi = 0.7_wp
        end select
    end subroutine field_range

    !> Model i: the free fields of matrix mode set over their ranges. The grouped fields (delt, dell) take one of four levels
    !> (the models are grouped by distinct pair: a continuous spread would exceed the number of pairs a batch may hold)
    subroutine vary_free(p, i)
        type(PARAMS), intent(inout) :: p
        integer, intent(in) :: i
        integer :: k, idx
        real(WP) :: lo, hi, u

        do k = 1, nfree
            idx = free_idx(k)
            call field_range(idx, lo, hi)
            if (hi == lo) cycle
            if (registry_class(idx) == CLASS_G) then
                u = (real(int(spread(i, 1) * 4.0_wp), WP) + 0.5_wp) / 4.0_wp
            else
                u = spread(i, idx)
            end if
            call registry_set(p%params_core, idx, lo + (hi - lo) * u)
        end do
    end subroutine vary_free

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        call opensps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) stop 1
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        call opensps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) stop 1
    end subroutine set_float

end program benchmark_csp_batch
