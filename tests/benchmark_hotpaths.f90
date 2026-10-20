program benchmark_hotpaths
    use iso_c_binding
    use fsps_api
    use fsps_types, only: COMPSPOUT
    use fsps_context_types, only: fsps_context_t
    use fsps_context, only: fsps_context_move_to_device, fsps_context_remove_from_device, fsps_context_set_fast_mode
    implicit none

    type(fsps_context_t) :: ctx
    type(COMPSPOUT), allocatable :: compsp(:)
    real(kind=c_double), allocatable :: mass_ssp(:), lbol_ssp(:), spec_ssp(:,:)
    real(kind=c_double), allocatable :: mass_ssp_2d(:,:), lbol_ssp_2d(:,:), spec_ssp_3d(:,:,:)
    
    real(kind=c_double) :: total_time
    integer(kind=c_long) :: count, rate, t1, t2
    integer :: i, n_iter_csp, n_iter_ssp
    integer :: status
    integer :: arg_count, arg_i
    integer :: n_spec, n_times, n_z, n_bands, n_indx
    integer :: csp_bypass_count
    real(c_double) :: val
    character(len=20) :: isoc_arg, spec_arg, dust_arg
    character(len=20) :: dust2_key, imf3_key
    character(len=20) :: mags_key, indx_key, tage_key
    character(len=255) :: arg_val

    ! Initialize keys
    dust2_key = "dust2"
    imf3_key  = "imf3"
    mags_key  = "compute_mags"
    indx_key  = "compute_indices"
    tage_key  = "tage"

    ! Defaults match test_runner-style usage
    isoc_arg = "mist"
    spec_arg = "miles"
    dust_arg = "DL07"

    ! Parse command-line arguments: --isoc, --spec, --dust
    arg_count = command_argument_count()
    arg_i = 1
    do while (arg_i <= arg_count)
        call get_command_argument(arg_i, arg_val, status=status)
        if (status /= 0) exit

        select case (trim(arg_val))
        case ("--isoc")
            arg_i = arg_i + 1
            if (arg_i <= arg_count) then
                call get_command_argument(arg_i, isoc_arg, status=status)
            else
                print *, "ERROR: --isoc requires an argument"
                stop 1
            end if
        case ("--spec")
            arg_i = arg_i + 1
            if (arg_i <= arg_count) then
                call get_command_argument(arg_i, spec_arg, status=status)
            else
                print *, "ERROR: --spec requires an argument"
                stop 1
            end if
        case ("--dust")
            arg_i = arg_i + 1
            if (arg_i <= arg_count) then
                call get_command_argument(arg_i, dust_arg, status=status)
            else
                print *, "ERROR: --dust requires an argument"
                stop 1
            end if
        case ("--help", "-h")
            print *, "Usage: benchmark_hotpaths [--isoc type] [--spec type] [--dust type]"
            print *, "Defaults: --isoc mist --spec miles --dust DL07"
            stop 0
        case default
            print *, "ERROR: Unknown argument: ", trim(arg_val)
            print *, "Usage: benchmark_hotpaths [--isoc type] [--spec type] [--dust type]"
            stop 1
        end select

        arg_i = arg_i + 1
    end do

    print *, "Creating FSPS Context..."
    call fsps_create(ctx)
    
    print *, "Setting up FSPS..."
    print *, "  isoc = ", trim(isoc_arg)
    print *, "  spec = ", trim(spec_arg)
    print *, "  dust = ", trim(dust_arg)
    call fsps_setup(ctx, 1, isoc_type_in=trim(isoc_arg), spec_type_in=trim(spec_arg), dust_type_in=trim(dust_arg))

    ! Get dimensions
    n_spec = ctx%state%nspec
    n_times = ctx%state%ntfull
    n_z = ctx%state%nz
    n_bands = ctx%state%nbands
    n_indx = ctx%state%nindx

    ! Allocate SSP buffers (1D/2D for single compute)
    allocate(mass_ssp(n_times))
    allocate(lbol_ssp(n_times))
    allocate(spec_ssp(n_spec, n_times))
    
    ! Allocate SSP buffers (2D/3D for CSP input)
    allocate(mass_ssp_2d(n_times, 1))
    allocate(lbol_ssp_2d(n_times, 1))
    allocate(spec_ssp_3d(n_spec, n_times, 1))
    
    ! Allocate CSP Output
    allocate(compsp(1))

    ! Disable heavy outputs
    call fsps_set_param_int(ctx, mags_key, 0, status)
    call fsps_set_param_int(ctx, indx_key, 0, status)

    ! Explicitly set tage > 0 to trigger the n_outputs == 1 hot path
    call fsps_set_param_float(ctx, tage_key, 13.7d0, status)
    if (status /= 0) then
        print *, "ERROR: fsps_set_param_float(tage) failed"
        stop 1
    end if

    ! Migrate the fully-populated context to the device
    call fsps_context_move_to_device(ctx)

    ! Enable Fast Mode to bypass host post-processing and syncs
    call fsps_context_set_fast_mode(ctx, .true.)

    ! Warmup
    print *, "Warming up..."
    call fsps_compute_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
    
    ! Fill 3D buffers
    mass_ssp_2d(:,1) = mass_ssp
    lbol_ssp_2d(:,1) = lbol_ssp
    spec_ssp_3d(:,:,1) = spec_ssp
    
    call fsps_compute_csp(ctx, 0, 1, "", mass_ssp_2d, lbol_ssp_2d, spec_ssp_3d, compsp)
    if (ctx%state%ssp_basis_is_dirty) then
        print *, "ERROR: SSP basis dirty flag expected false after initial CSP warmup"
        stop 1
    end if

    ! --- Benchmark 1: CSP Only ---
    print *, "Benchmarking CSP Only..."
    n_iter_csp = 1000
    csp_bypass_count = 0
    
    call system_clock(t1, rate)
    
    do i = 1, n_iter_csp
        val = 0.1d0 + dble(i)*0.01d0
        call fsps_set_param_float(ctx, dust2_key, val, status)
        if (status /= 0) then
            print *, "ERROR: fsps_set_param_float(dust2) failed with status=", status
            stop 1
        end if
        if (ctx%state%ssp_basis_is_dirty) then
            print *, "ERROR: dust2 update should not dirty SSP basis"
            stop 1
        end if
        
        call fsps_compute_csp(ctx, 0, 1, "", mass_ssp_2d, lbol_ssp_2d, spec_ssp_3d, compsp)
        if (ctx%state%ssp_basis_is_dirty) then
            print *, "ERROR: CSP-only iteration should keep SSP basis clean"
            stop 1
        end if
        csp_bypass_count = csp_bypass_count + 1
    end do
    
    call system_clock(t2)
    total_time = real(t2 - t1, kind=c_double) / real(rate, kind=c_double)
    print *, "CSP Only Avg Time (s): ", total_time / real(n_iter_csp)
    print *, "CSP-only iterations with SSP-basis bypass:", csp_bypass_count

    ! --- Benchmark 2: SSP + CSP ---
    print *, "Benchmarking SSP + CSP..."
    n_iter_ssp = 20 
    
    call system_clock(t1, rate)
    
    do i = 1, n_iter_ssp
        val = 1.3d0 + dble(i)*0.01d0
        call fsps_set_param_float(ctx, imf3_key, val, status)
        if (status /= 0) then
            print *, "ERROR: fsps_set_param_float(imf3) failed with status=", status
            stop 1
        end if
        if (.not. ctx%state%ssp_basis_is_dirty) then
            print *, "ERROR: imf3 update should dirty SSP basis"
            stop 1
        end if
        
        ! Recompute SSP
        call fsps_compute_ssp(ctx, mass_ssp, lbol_ssp, spec_ssp)
        
        ! Copy to 3D buffers
        mass_ssp_2d(:,1) = mass_ssp
        lbol_ssp_2d(:,1) = lbol_ssp
        spec_ssp_3d(:,:,1) = spec_ssp
        
        ! Compute CSP
        call fsps_compute_csp(ctx, 0, 1, "", mass_ssp_2d, lbol_ssp_2d, spec_ssp_3d, compsp)
        if (ctx%state%ssp_basis_is_dirty) then
            print *, "ERROR: SSP basis dirty flag should reset after CSP basis update"
            stop 1
        end if
    end do
    
    call system_clock(t2)
    total_time = real(t2 - t1, kind=c_double) / real(rate, kind=c_double)
    print *, "SSP + CSP Avg Time (s): ", total_time / real(n_iter_ssp)

    call fsps_context_remove_from_device(ctx)
    call fsps_destroy(ctx)

end program benchmark_hotpaths

