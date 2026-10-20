program benchmark_ssp_operator
    !> Costs of the batched SSP operator (fsps_ssp_operator) on one core, one metallicity:
    !> generate_ssp_grid; building S once; the per-model part (IMF weights of all isochrone
    !> points and W = S . w_IMF, alone and split); the full SSP grid from S (per-model part +
    !> the template product + post-processing, as generate_ssp_grid).
    !> Usage: benchmark_ssp_operator [isoc spec zmet]   (default: mist miles 10)

    use fsps_precision, only: WP
    use fsps_types, only: PARAMS
    use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
    use fsps_ssp, only: generate_ssp_grid, configure_imf_parameters
    use fsps_ssp_operator, only: ssp_operator_t, ssp_operator_build, ssp_operator_evaluate, &
                                 ssp_op_model_weights, ssp_op_imf_weights, ssp_op_spmv
    use fsps_imf, only: imf_pack_params, IMF_NPAR
    use fsps_context, only: fsps_context_get_pset, fsps_context_prepare_pset, fsps_context_set_param_int
    use fsps_context_types, only: fsps_context_t
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none

    type(fsps_context_t), allocatable :: ctx
    type(ssp_operator_t) :: op
    type(PARAMS) :: pset
    character(len=32) :: isoc = 'mist', spec = 'miles', arg
    integer :: zmet = 10, status, nt, ns, rep
    integer, parameter :: N_SSP = 3, N_W = 50
    real(WP), allocatable :: m(:), l(:), s(:,:), w(:), wrow(:)
    real(WP) :: p(IMF_NPAR), t_ssp, t_build, t_w, t_imf, t_spmv, t_eval
    integer(int64) :: c0, c1, rate

    if (command_argument_count() >= 3) then
        call get_command_argument(1, isoc)
        call get_command_argument(2, spec)
        call get_command_argument(3, arg)
        read(arg, *) zmet
    end if
    allocate(ctx)
    call fsps_create(ctx)
    ctx%pset%zmet = zmet
    call fsps_setup(ctx, zmet, isoc_type_in=trim(isoc), spec_type_in=trim(spec), dust_type_in='DL07')
    call fsps_context_set_param_int(ctx, 'zmet', zmet, status)
    call fsps_context_set_param_int(ctx, 'imf_type', 1, status)
    call fsps_context_prepare_pset(ctx)
    call fsps_context_get_pset(ctx, pset)
    nt = ctx%state%ntfull
    ns = ctx%state%nspec
    allocate(m(nt), l(nt), s(ns, nt))
    call system_clock(count_rate=rate)

    call generate_ssp_grid(ctx, pset, m, l, s)   ! warm-up
    call system_clock(c0)
    do rep = 1, N_SSP
        call generate_ssp_grid(ctx, pset, m, l, s)
    end do
    call system_clock(c1)
    t_ssp = real(c1 - c0, WP) / rate / N_SSP

    call system_clock(c0)
    do rep = 1, N_SSP
        call ssp_operator_build(ctx, pset, op)
    end do
    call system_clock(c1)
    t_build = real(c1 - c0, WP) / rate / N_SSP

    call configure_imf_parameters(ctx, pset)
    call imf_pack_params(ctx, p)
    allocate(w(op%n_pts), wrow(op%n_rows))
    call system_clock(c0)
    do rep = 1, N_W
        call ssp_op_model_weights(ctx%imf_type_val, p, op%n_pts, op%pt_mini, op%pt_m1, op%pt_m2, op%pt_first, &
                                  op%n_rows, op%nnz, op%row_ptr, op%col, op%val, w, wrow)
    end do
    call system_clock(c1)
    t_w = real(c1 - c0, WP) / rate / N_W
    call system_clock(c0)
    do rep = 1, N_W
        call ssp_op_imf_weights(ctx%imf_type_val, p, op%n_pts, op%pt_mini, op%pt_m1, op%pt_m2, op%pt_first, w)
    end do
    call system_clock(c1)
    t_imf = real(c1 - c0, WP) / rate / N_W
    call system_clock(c0)
    do rep = 1, N_W
        call ssp_op_spmv(op%n_rows, op%nnz, op%n_pts, op%row_ptr, op%col, op%val, w, wrow)
    end do
    call system_clock(c1)
    t_spmv = real(c1 - c0, WP) / rate / N_W

    call system_clock(c0)
    do rep = 1, N_SSP
        call ssp_operator_evaluate(ctx, pset, op, ctx%imf_type_val, p, m, l, s)
    end do
    call system_clock(c1)
    t_eval = real(c1 - c0, WP) / rate / N_SSP

    write(*, '(a, a, 1x, a, a, i0, a, i0, a, i0)') 'isochrones/library: ', trim(isoc), trim(spec), &
        ', zmet ', zmet, ', ages ', nt, ', pixels ', ns
    write(*, '(a, i0, a, i0, a, i0, a, i0, a, i0)') 'S: points ', op%n_pts, ', rows ', op%n_rows, ', entries ', &
        op%nnz, ', dust corners ', op%n_slots, ', direct stars ', op%n_direct
    write(*, '(a, f10.3, a)') 'generate_ssp_grid                    ', 1.0e3_wp * t_ssp, ' ms'
    write(*, '(a, f10.3, a)') 'build S (once)                       ', 1.0e3_wp * t_build, ' ms'
    write(*, '(a, f10.3, a)') 'per model: IMF weights + S . w       ', 1.0e3_wp * t_w, ' ms'
    write(*, '(a, f10.3, a)') '   of which IMF weights              ', 1.0e3_wp * t_imf, ' ms'
    write(*, '(a, f10.3, a)') '   of which S . w                    ', 1.0e3_wp * t_spmv, ' ms'
    write(*, '(a, f10.3, a)') 'SSP grid from S (incl. templates)    ', 1.0e3_wp * t_eval, ' ms'
    call fsps_destroy(ctx)
end program benchmark_ssp_operator
