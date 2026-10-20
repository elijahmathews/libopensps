program benchmark_csp_batch
    !> Per-model cost of the batched CSP (compute_csp_batch) against one compute_csp_scenario
    !> call per model, on the benchmark_phases setup (MIST + MILES + DL07, tau model with dust and
    !> dust emission). Tau, age and the dust parameters vary across the batch.
    !> Usage: benchmark_csp_batch [n_max [n_single [config]]]. config = "full": the real-world
    !> fitting configuration (dust1 = 0.5, dust2 = 0.3, DL07 dust emission, nebular emission,
    !> IGM at zred = 0.5, AGN with fagn = 0.1, spectral indices; tau, age, U_min and gas_logu vary);
    !> "full_smooth": the same with velocity smoothing, sigma_smooth = 150 km/s; "full_z" and
    !> "full_smooth_z": with the bases of all metallicities resident and logzsol varying per model.
    !> Prints one line per metric: name, value. Per batch size: magnitudes only; with spectra
    !> copied back to a host array (reused, as in a fitting loop); with spectra left on the device.

    use fsps_precision, only: WP
    use fsps_types, only: PARAMS, COMPSPOUT
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
    use fsps_csp, only: compute_csp_scenario
    use fsps_csp_batch, only: compute_csp_batch, batch_spectra_device, batch_spectra_host
    use, intrinsic :: iso_c_binding, only: c_ptr, c_associated
    use fsps_ssp, only: generate_ssp_grid
    use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                            fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset, &
                            fsps_context_set_param_int, fsps_context_set_param_float
    use fsps_context_types, only: fsps_context_t
    implicit none

    type(fsps_context_t), allocatable :: ctx
    type(PARAMS) :: pset
    type(PARAMS), allocatable :: psets(:)
    type(COMPSPOUT), allocatable :: results(:)
    real(WP), allocatable :: spec_ssp(:, :), mass_ssp(:), lbol_ssp(:), mags(:, :), spec(:, :)
    character(len=32) :: arg, config = 'basic'
    logical :: full, zvar = .false.
    real(WP), allocatable :: spec_z(:, :, :), mass_z(:, :), lbol_z(:, :)
    integer :: z
    real(WP), allocatable :: indx(:, :)
    integer(8) :: t0, t1, rate
    integer :: n_max = 10000, n_single = 200, status, nspec, ntfull, i, n, rep, n_rep, k
    integer, parameter :: sizes(4) = [1, 100, 1000, 10000]
    real(WP) :: dt, total
    type(c_ptr) :: dptr
    integer :: k_nspec, k_n, k_bytes

    call get_command_argument(1, arg)
    if (len_trim(arg) > 0) read(arg, *) n_max
    call get_command_argument(2, arg)
    if (len_trim(arg) > 0) read(arg, *) n_single
    call get_command_argument(3, arg)
    if (len_trim(arg) > 0) config = arg
    full = index(config, 'full') == 1
    zvar = index(config, '_z') > 0
    call system_clock(count_rate=rate)

    allocate(ctx)
    call fsps_create(ctx)
    ctx%pset%zmet = 10
    call fsps_setup(ctx, 10, isoc_type_in='mist', spec_type_in='miles', dust_type_in='DL07')
    call fsps_context_move_to_device(ctx)
    call set_int('imf_type', 1)
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
    if (index(config, 'full_smooth') == 1) then
        call set_int('smooth_velocity', 1)
        call set_float('sigma_smooth', 150.0_wp)
    end if
    call fsps_context_prepare_pset(ctx)
    call fsps_context_get_pset(ctx, pset)

    nspec = ctx%state%nspec
    ntfull = ctx%state%ntfull
    allocate(spec_ssp(nspec, ntfull), mass_ssp(ntfull), lbol_ssp(ntfull))
    call fsps_context_set_pset(ctx, pset)
    write(*, '(A,A)') 'config ', trim(config)
    call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)
    call fsps_context_update_ssp_basis(ctx, reshape(spec_ssp, [nspec, ntfull, 1]), &
                                       reshape(mass_ssp, [ntfull, 1]), reshape(lbol_ssp, [ntfull, 1]), 1)
    if (zvar) then
        ! The bases of all metallicities (the single call keeps the zmet = 10 basis)
        allocate(spec_z(nspec, ntfull, ctx%state%nz), mass_z(ntfull, ctx%state%nz), lbol_z(ntfull, ctx%state%nz))
        do z = 1, ctx%state%nz
            call set_int('zmet', z)
            call fsps_context_prepare_pset(ctx)
            call fsps_context_get_pset(ctx, pset)
            call fsps_context_set_pset(ctx, pset)
            call generate_ssp_grid(ctx, pset, mass_z(:, z), lbol_z(:, z), spec_z(:, :, z))
        end do
        call set_int('zmet', 10)
        call fsps_context_prepare_pset(ctx)
        call fsps_context_get_pset(ctx, pset)
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
    write(*, '(A,F14.3)') 'single_call_us_per_model ', 1.0e6_wp * real(t1 - t0, WP) / rate / n_single

    ! Batched, magnitudes only and with spectra
    if (zvar) then
        ctx%state%ssp_basis_is_dirty = .true.
        call fsps_context_update_ssp_basis(ctx, spec_z, mass_z, lbol_z, ctx%state%nz)
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
        n_rep = max(3, min(200, 20000 / n))
        call batch(n, .false.)   ! warm-up
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
    end do
    if (total /= total) write(*, *) total   ! keep the results live

    call fsps_destroy(ctx)
    deallocate(ctx)

contains

    !> The batch of the first n models (magnitudes; spectra too if with_spec; indices in "full").
    subroutine batch(n, with_spec)
        integer, intent(in) :: n
        logical, intent(in) :: with_spec
        if (full .and. with_spec) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, spec=spec, indx=indx)
        else if (full) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, indx=indx)
        else if (with_spec) then
            call compute_csp_batch(ctx, psets(1:n), mags, status, spec=spec)
        else
            call compute_csp_batch(ctx, psets(1:n), mags, status)
        end if
    end subroutine batch

    !> Model i: tau, age and dust vary over a fitting-like range ("full": the dust is fixed and
    !> gas_logu varies instead).
    subroutine vary(p, i)
        type(PARAMS), intent(inout) :: p
        integer, intent(in) :: i
        p%tau = 0.3_wp + 0.05_wp * mod(i, 97)
        p%tage = 1.0_wp + 0.1_wp * mod(i, 113)
        p%duste_umin = 1.0_wp + 0.2_wp * mod(i, 71)
        if (zvar) p%logzsol = -1.5_wp + 0.017_wp * mod(i, 97)
        if (full) then
            p%gas_logu = -3.5_wp + 0.05_wp * mod(i, 41)
        else
            p%dust2 = 0.05_wp + 0.01_wp * mod(i, 89)
            p%dust1 = 0.02_wp * mod(i, 53)
        end if
    end subroutine vary

    subroutine set_int(key, value)
        character(len=*), intent(in) :: key
        integer, intent(in) :: value
        call fsps_context_set_param_int(ctx, key, value, status)
        if (status /= 0) stop 1
    end subroutine set_int

    subroutine set_float(key, value)
        character(len=*), intent(in) :: key
        real(WP), intent(in) :: value
        call fsps_context_set_param_float(ctx, key, value, status)
        if (status /= 0) stop 1
    end subroutine set_float

end program benchmark_csp_batch
