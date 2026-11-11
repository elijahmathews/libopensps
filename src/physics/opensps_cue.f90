module opensps_cue
    !> @brief
    !> Nebular emission from cue (Li et al. 2024; arXiv:2405.04598): CLOUDY emulated by neural
    !> networks for an ionizing spectrum described by a 4-part power law, and free gas density,
    !> metallicity, N/O and C/O.
    !>
    !> @details
    !> cue has 17 networks of the "speculator" form (Alsing et al. 2020): one for the nebular
    !> continuum and one per group of emission lines (16 groups, by element and ionization
    !> potential, 138 lines). Each maps the same 12 inputs
    !>
    !>   (alpha_HeII, alpha_OII, alpha_HeI, alpha_HI, log F_OII/F_HeII, log F_HeI/F_OII,
    !>    log F_HI/F_HeI, log Q(U, n_H), n_H, log (O/H)/(O/H)_sun, log (N/O)/(N/O)_sun,
    !>    log (C/O)/(C/O)_sun)
    !>
    !> through three hidden layers of 256 units, y = (beta + (1 - beta) sigmoid(alpha a)) a with
    !> a = W x + b, to PCA coefficients, and the PCA basis gives log10 of the luminosities [erg/s
    !> for lines, erg/s/Hz for the continuum] of a nebula ionized by log Q(U, n_H) photons per
    !> second. The emission of a population with Q absorbed ionizing photons is that times
    !> Q / Q(U, n_H) (cue: 10**(log L - log Q(U, n_H) + log Q - log 3.839e33) in L_sun).
    !>
    !> This module holds the weights (read once from a converted library, tools/cue/convert_cue.py,
    !> found through the environment variable OPENSPS_CUE_DATA), the network evaluation for a batch
    !> of inputs, cue's fit of the 4-part power law to an ionizing spectrum
    !> (cue.utils.fit_4loglinear_ionparam), and the not-a-knot cubic spline cue uses to put the
    !> continuum on the output wavelength grid. The batched CSP (opensps_csp_batch) calls them
    !> with nebular_model = 1.
    !>
    !> Precision: the hidden layers run in the kernel precision (KP), as the weights were trained
    !> (FP32). The output layers, the PCA reconstruction (means ~50-100, output scales up to 20: in
    !> FP32 it rounds at ~1e-4 dex), the output scale and shift (log10 L ~ 40) and the
    !> normalization by Q are FP64, before the exponential (doc/MIXED_PRECISION.md, rule 3). The
    !> power-law fit is FP64.
    use, intrinsic :: iso_fortran_env, only: int64, error_unit
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_null_ptr
    use opensps_precision, only: WP, KP
    use opensps_datalib, only: datalib_t, datalib_open_dir, datalib_check_array, datalib_read_real, &
                               datalib_read_int, datalib_axis_values, datalib_axis_len, datalib_has_array
#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET)
    use opensps_accel_blas, only: accel_blas_init, accel_blas_gemm_strided_batched, accel_blas_dgemm_strided_batched, &
                                  ACCEL_BLAS_OP_N
#endif
    implicit none
    private

    integer, parameter, public :: CUE_NIN = 12      !< network inputs
    integer, parameter, public :: CUE_NHID = 256    !< units per hidden layer
    integer, parameter, public :: CUE_NGROUP = 16   !< line networks
    integer, parameter, public :: CUE_NNET = 17     !< line networks, then the continuum network
    integer, parameter, public :: CUE_NPCA_LINE = 16  !< PCA components of a line network (zero-padded)
    integer, parameter, public :: CUE_NION = 7      !< parameters of the ionizing spectrum's shape

    !> Segment edges of the power law [A]: He II, O II, He I and H I ionization (cue/constants.py)
    real(WP), parameter, public :: CUE_EDGE_HEII = 1.0e8_wp / 438908.8789_wp
    real(WP), parameter, public :: CUE_EDGE_OII = 1.0e8_wp / 283270.9_wp
    real(WP), parameter, public :: CUE_EDGE_HEI = 1.0e8_wp / 198310.66637_wp
    real(WP), parameter, public :: CUE_EDGE_HI = 911.6_wp
    !> cue's constants: Planck's constant [erg s] and c [A/s] of its fit, and the solar luminosity
    !> [erg/s] of its output normalization
    real(WP), parameter :: CUE_H = 6.626e-27_wp, CUE_C = 2.9979e18_wp
    real(WP), parameter, public :: CUE_LOG_LSUN = 33.584218112117405_wp   !< log10(3.839e33)
    !> log Q(U, n_H) = log U + log n_H + LOGQ_CONST (R_inner = 1e19 cm, c = 2.9979e10 cm/s)
    real(WP), parameter, public :: CUE_LOGQ_CONST = 49.57602700615259_wp

    !> Models per pass through the networks (bounds the scratch: 2 x 256 x 17 x CHUNK values)
    integer, parameter, public :: CUE_CHUNK = 4096
    !> Models per pass of the batch's cue stage (opensps_csp_batch; the tests lower it to run several
    !> passes with a remainder)
    integer, public, save :: cue_batch_chunk = CUE_CHUNK
    !> Models per pass through the networks' reverse pass (cue_forward_adjoint), which keeps the
    !> pre-activations of the three layers and a work array in FP64: 8 x 256 x 17 x CHUNK values
    integer, parameter, public :: CUE_BCHUNK = 1024
    !> Models per thread in the dense layers (register tile)
    integer, parameter :: TM = 4

    !> The networks and PCA bases of cue, as stored by tools/cue/convert_cue.py (Fortran order:
    !> the dimensions of the manifest reversed).
    type, public :: cue_weights_t
        logical :: loaded = .false.
        logical :: on_device = .false.
        integer :: nlines = 0      !< lines (138), in the networks' output order
        integer :: nlmax = 0       !< lines of the largest group
        integer :: ncw = 0         !< continuum wavelengths (1841)
        integer :: npc = 0         !< continuum PCA components (50)
        character(len=:), allocatable :: dir
        real(KP), allocatable :: w1(:,:,:), w2(:,:,:), w3(:,:,:)   !< (out, in, net)
        real(WP), allocatable :: w2d(:,:,:), w3d(:,:,:)             !< FP64 copies for a GPU BLAS (FP32 builds)
        real(KP), allocatable :: bias(:,:,:)                        !< (256, net, layer)
        real(KP), allocatable :: alpha(:,:,:), beta(:,:,:)          !< (256, net, layer)
        real(KP), allocatable :: in_shift(:,:), in_scale(:,:)       !< (12, net)
        ! The output layers and PCA bases are applied in FP64 (cue_forward): the PCA means are ~50-100
        ! and the output scales up to 20, so an FP32 reconstruction would round at ~1e-4 dex
        real(KP), allocatable :: lw4(:,:,:)                         !< (pca, 256, group)
        real(WP), allocatable :: lb4(:,:), lpsh(:,:), lpsc(:,:)     !< (pca, group)
        real(WP), allocatable :: lcmp(:,:,:)                        !< (line slot, pca, group)
        real(KP), allocatable :: lcmp_k(:,:,:)                      !< the lines' bases in KP (FP32 hidden layers)
        real(WP), allocatable :: lmean(:,:)                         !< (line slot, group)
        real(WP), allocatable :: lout_sh(:,:), lout_sc(:,:)         !< (line slot, group)
        integer, allocatable  :: npca(:), ncount(:), loff(:)        !< per group
        real(WP), allocatable :: line_lam(:)                        !< vacuum [A], output order
        real(KP), allocatable :: cw4(:,:)                           !< (pca, 256)
        real(WP), allocatable :: cb4(:), cpsh(:), cpsc(:)           !< (pca)
        real(WP), allocatable :: ccmp(:,:)                          !< (wavelength, pca)
        real(WP), allocatable :: cmean(:)
        real(WP), allocatable :: cout_sh(:), cout_sc(:), cont_lam(:)
    end type cue_weights_t

    !> cue's not-a-knot cubic spline from its continuum grid to an output grid, factored once:
    !> the second derivatives M solve a tridiagonal system (Thomas: cp, rden) per spectrum; output
    !> pixel j lies in knot interval k(j) at fraction t(j) (0 where the continuum is zero).
    type, public :: cue_spline_t
        integer :: n = 0, nout = 0
        logical :: on_device = .false.
        real(WP), allocatable :: xout(:)       !< the output grid it was built for
        real(WP), allocatable :: h(:)          !< knot spacings (n - 1)
        real(KP), allocatable :: cp(:), rden(:), aa(:)  !< Thomas factors of the reduced system
        real(KP), allocatable :: m0c(:), mnc(:) !< M_1 = m0c(1) M_2 + m0c(2) M_3; likewise at the end
        integer, allocatable  :: k(:)
        real(KP), allocatable :: t(:), hh6(:)
        !> The pixels of each knot interval (first and last; none: kst = nout + 1, ken = 0): the
        !> transpose gathers over them (cue_spline_adjoint)
        integer, allocatable  :: kst(:), ken(:)
    end type cue_spline_t

#if defined(OPENSPS_HIP) && defined(OMP_TARGET)
    !> The hidden layers' kernel (src/accel/hip): weights FP32 with FP32 or FP64 activations, or FP64
    interface
        subroutine hip_cue_dense_ss(k, nc, ld, w, b, al, be, x, y, stream) bind(C, name="opensps_hip_cue_dense_act_ss")
            import :: c_int, c_ptr
            integer(c_int), value :: k, nc, ld
            type(c_ptr), value :: w, b, al, be, x, y, stream
        end subroutine hip_cue_dense_ss
        subroutine hip_cue_dense_sd(k, nc, ld, w, b, al, be, x, y, stream) bind(C, name="opensps_hip_cue_dense_act_sd")
            import :: c_int, c_ptr
            integer(c_int), value :: k, nc, ld
            type(c_ptr), value :: w, b, al, be, x, y, stream
        end subroutine hip_cue_dense_sd
        subroutine hip_cue_dense_dd(k, nc, ld, w, b, al, be, x, y, stream) bind(C, name="opensps_hip_cue_dense_act_dd")
            import :: c_int, c_ptr
            integer(c_int), value :: k, nc, ld
            type(c_ptr), value :: w, b, al, be, x, y, stream
        end subroutine hip_cue_dense_dd
        !> The reverse pass: the activation (y; may be null) and the pre-activation (p) in FP64
        subroutine hip_cue_pre_sd(k, nc, ld, w, b, al, be, x, y, p, stream) bind(C, name="opensps_hip_cue_dense_pre_sd")
            import :: c_int, c_ptr
            integer(c_int), value :: k, nc, ld
            type(c_ptr), value :: w, b, al, be, x, y, p, stream
        end subroutine hip_cue_pre_sd
        subroutine hip_cue_pre_dd(k, nc, ld, w, b, al, be, x, y, p, stream) bind(C, name="opensps_hip_cue_dense_pre_dd")
            import :: c_int, c_ptr
            integer(c_int), value :: k, nc, ld
            type(c_ptr), value :: w, b, al, be, x, y, p, stream
        end subroutine hip_cue_pre_dd
        !> din = W^T dout of a 256 x 256 layer in FP64
        subroutine hip_cue_back_sd(nc, ld, w, d, o, stream) bind(C, name="opensps_hip_cue_dense_back_sd")
            import :: c_int, c_ptr
            integer(c_int), value :: nc, ld
            type(c_ptr), value :: w, d, o, stream
        end subroutine hip_cue_back_sd
        subroutine hip_cue_back_dd(nc, ld, w, d, o, stream) bind(C, name="opensps_hip_cue_dense_back_dd")
            import :: c_int, c_ptr
            integer(c_int), value :: nc, ld
            type(c_ptr), value :: w, d, o, stream
        end subroutine hip_cue_back_dd
    end interface
#endif

    type(cue_weights_t), target, save, public :: cue_w
    type(cue_spline_t), target, save, public :: cue_sp

    public :: cue_load, cue_load_dir, cue_unload, cue_data_dir, cue_device_put
    public :: cue_forward, cue_fit_ionizing, cue_ionizing_params, cue_inputs
    public :: cue_spline_build, cue_spline_apply, cue_spline_solve, cue_line_map
    public :: cue_log_ratio_power, cue_ion_bound, fit_segment
    public :: cue_forward_adjoint, cue_ionizing_params_adjoint, cue_spline_adjoint

contains

    ! ==========================================================================================
    ! Loading
    ! ==========================================================================================

    !> The library directory named by OPENSPS_CUE_DATA: the directory itself if it holds
    !> manifest.json, else <it>/libraries/nebular_emulator/cue. Empty if unset or not found.
    function cue_data_dir() result(dir)
        character(len=:), allocatable :: dir
        character(len=4096) :: env
        integer :: n, st
        logical :: ex

        dir = ''
        call get_environment_variable('OPENSPS_CUE_DATA', env, n, st)
        if (st /= 0 .or. n == 0) return
        inquire(file=trim(env) // '/manifest.json', exist=ex)
        if (ex) then
            dir = trim(env)
            return
        end if
        inquire(file=trim(env) // '/libraries/nebular_emulator/cue/manifest.json', exist=ex)
        if (ex) dir = trim(env) // '/libraries/nebular_emulator/cue'
    end function cue_data_dir

    !> Loads the weights named by OPENSPS_CUE_DATA into cue_w (once; later calls return at once).
    !> status: 0 loaded, 1 not found (variable unset or no library there).
    subroutine cue_load(status)
        integer, intent(out) :: status
        character(len=:), allocatable :: dir

        status = 0
        if (cue_w%loaded) return
        !$omp critical (opensps_cue_load)
        if (.not. cue_w%loaded) then
            dir = cue_data_dir()
            if (len(dir) == 0) then
                status = 1
            else
                call cue_load_dir(cue_w, dir)
            end if
        end if
        !$omp end critical (opensps_cue_load)
    end subroutine cue_load

    !> Reads a converted cue library from its directory into w (and, in GPU builds, onto the
    !> device). A malformed library stops with a message (as every data library).
    subroutine cue_load_dir(w, dir)
        type(cue_weights_t), intent(inout) :: w
        character(len=*), intent(in) :: dir

        type(datalib_t) :: lib
        real(WP), allocatable :: t3(:,:,:), t2(:,:), t1(:)
        integer :: g, l, nl

        call cue_unload(w)
        call datalib_open_dir(lib, dir, 'nebular_emulator', 'cue')
        w%dir = dir
        w%nlines = datalib_axis_len(lib, 'line')
        w%nlmax = datalib_axis_len(lib, 'line_slot')
        w%ncw = datalib_axis_len(lib, 'cont_wavelength')
        w%npc = datalib_axis_len(lib, 'pca_cont')
        if (datalib_axis_len(lib, 'net') /= CUE_NNET .or. datalib_axis_len(lib, 'input') /= CUE_NIN .or. &
            datalib_axis_len(lib, 'hidden') /= CUE_NHID .or. datalib_axis_len(lib, 'pca_line') /= CUE_NPCA_LINE) then
            write(error_unit, '(A)') '[OPENSPS_CUE] Error: ' // dir // ' is not a cue library of the expected architecture'
            error stop 1
        end if

        allocate(w%w1(CUE_NHID, CUE_NIN, CUE_NNET), w%w2(CUE_NHID, CUE_NHID, CUE_NNET), &
                 w%w3(CUE_NHID, CUE_NHID, CUE_NNET), w%bias(CUE_NHID, CUE_NNET, 3), &
                 w%alpha(CUE_NHID, CUE_NNET, 3), w%beta(CUE_NHID, CUE_NNET, 3))
        call read3(lib, 'w1', 'net', 'input', 'hidden', CUE_NNET, CUE_NIN, CUE_NHID, t3)
        w%w1 = real(t3, KP)
        call read3(lib, 'w2', 'net', 'hidden_in', 'hidden', CUE_NNET, CUE_NHID, CUE_NHID, t3)
        w%w2 = real(t3, KP)
        call read3(lib, 'w3', 'net', 'hidden_in', 'hidden', CUE_NNET, CUE_NHID, CUE_NHID, t3)
        w%w3 = real(t3, KP)
        call read2(lib, 'b1', 'net', 'hidden', CUE_NNET, CUE_NHID, t2)
        w%bias(:, :, 1) = real(t2, KP)
        call read2(lib, 'b2', 'net', 'hidden', CUE_NNET, CUE_NHID, t2)
        w%bias(:, :, 2) = real(t2, KP)
        call read2(lib, 'b3', 'net', 'hidden', CUE_NNET, CUE_NHID, t2)
        w%bias(:, :, 3) = real(t2, KP)
        call read3(lib, 'alpha', 'net', 'layer', 'hidden', CUE_NNET, 3, CUE_NHID, t3)
        do l = 1, 3
            w%alpha(:, :, l) = real(t3(:, l, :), KP)
        end do
        call read3(lib, 'beta', 'net', 'layer', 'hidden', CUE_NNET, 3, CUE_NHID, t3)
        do l = 1, 3
            w%beta(:, :, l) = real(t3(:, l, :), KP)
        end do
        call read2(lib, 'input_shift', 'net', 'input', CUE_NNET, CUE_NIN, t2)
        w%in_shift = real(t2, KP)
        call read2(lib, 'input_scale', 'net', 'input', CUE_NNET, CUE_NIN, t2)
        w%in_scale = real(t2, KP)

        allocate(w%npca(CUE_NGROUP), w%ncount(CUE_NGROUP), w%loff(CUE_NGROUP))
        call read_int1(lib, 'line_npca', 'line_group', CUE_NGROUP, w%npca)
        call read_int1(lib, 'line_count', 'line_group', CUE_NGROUP, w%ncount)
        nl = 0
        do g = 1, CUE_NGROUP
            w%loff(g) = nl
            nl = nl + w%ncount(g)
        end do
        if (nl /= w%nlines .or. any(w%npca > CUE_NPCA_LINE) .or. any(w%ncount > w%nlmax)) then
            write(error_unit, '(A)') '[OPENSPS_CUE] Error: inconsistent line groups in ' // dir
            error stop 1
        end if
        call read3(lib, 'line_w4', 'line_group', 'hidden_in', 'pca_line', CUE_NGROUP, CUE_NHID, CUE_NPCA_LINE, t3)
        w%lw4 = real(t3, KP)
        call read2(lib, 'line_b4', 'line_group', 'pca_line', CUE_NGROUP, CUE_NPCA_LINE, t2)
        w%lb4 = t2
        call read2(lib, 'line_pca_shift', 'line_group', 'pca_line', CUE_NGROUP, CUE_NPCA_LINE, t2)
        w%lpsh = t2
        call read2(lib, 'line_pca_scale', 'line_group', 'pca_line', CUE_NGROUP, CUE_NPCA_LINE, t2)
        w%lpsc = t2
        call read3(lib, 'line_components', 'line_group', 'pca_line', 'line_slot', CUE_NGROUP, CUE_NPCA_LINE, &
                   w%nlmax, t3)
        w%lcmp = t3
        call read2(lib, 'line_mean', 'line_group', 'line_slot', CUE_NGROUP, w%nlmax, t2)
        w%lmean = t2
        call read2(lib, 'line_out_shift', 'line_group', 'line_slot', CUE_NGROUP, w%nlmax, t2)
        w%lout_sh = t2
        w%lcmp_k = real(w%lcmp, KP)
        call read2(lib, 'line_out_scale', 'line_group', 'line_slot', CUE_NGROUP, w%nlmax, t2)
        w%lout_sc = t2
        allocate(w%line_lam(w%nlines))
        call datalib_axis_values(lib, 'line', w%line_lam)

        call read2(lib, 'cont_w4', 'hidden_in', 'pca_cont', CUE_NHID, w%npc, t2)
        w%cw4 = real(t2, KP)
        call read1(lib, 'cont_b4', 'pca_cont', w%npc, t1)
        w%cb4 = t1
        call read1(lib, 'cont_pca_shift', 'pca_cont', w%npc, t1)
        w%cpsh = t1
        call read1(lib, 'cont_pca_scale', 'pca_cont', w%npc, t1)
        w%cpsc = t1
        call read2(lib, 'cont_components', 'pca_cont', 'cont_wavelength', w%npc, w%ncw, t2)
        w%ccmp = t2
        call read1(lib, 'cont_mean', 'cont_wavelength', w%ncw, t1)
        w%cmean = t1
        call read1(lib, 'cont_out_shift', 'cont_wavelength', w%ncw, t1)
        w%cout_sh = t1
        call read1(lib, 'cont_out_scale', 'cont_wavelength', w%ncw, t1)
        w%cout_sc = t1
        allocate(w%cont_lam(w%ncw))
        call datalib_axis_values(lib, 'cont_wavelength', w%cont_lam)
        w%loaded = .true.
        call cue_device_put(w)
    end subroutine cue_load_dir

    !> Copies the weights to the device (GPU builds; nothing otherwise). cue_load_dir calls it;
    !> weights filled otherwise (the tests' synthetic networks) call it once filled.
    subroutine cue_device_put(w)
        type(cue_weights_t), intent(inout) :: w
        ! (one enter data for all components: LLVM's offload runtime maps the members of a derived
        ! type as one range of it, and a second, wider range over the same object is refused)
#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET)
        ! The FP64 hidden layers through the BLAS take FP64 weights (exact copies)
        if (.not. allocated(w%w2d)) then
            w%w2d = real(w%w2, WP)
            w%w3d = real(w%w3, WP)
        end if
        if (w%on_device) return
        !$omp target enter data map(to: w%w1, w%w2, w%w3, w%bias, w%alpha, w%beta, w%in_shift, w%in_scale, &
        !$omp&   w%lw4, w%lb4, w%lpsh, w%lpsc, w%lcmp, w%lmean, w%lout_sh, w%lout_sc, w%npca, w%ncount, w%loff, &
        !$omp&   w%cw4, w%cb4, w%cpsh, w%cpsc, w%ccmp, w%cmean, w%cout_sh, w%cout_sc, w%lcmp_k, w%w2d, w%w3d)
        w%on_device = .true.
#elif defined(OMP_TARGET)
        if (w%on_device) return
        !$omp target enter data map(to: w%w1, w%w2, w%w3, w%bias, w%alpha, w%beta, w%in_shift, w%in_scale, &
        !$omp&   w%lw4, w%lb4, w%lpsh, w%lpsc, w%lcmp, w%lmean, w%lout_sh, w%lout_sc, w%npca, w%ncount, w%loff, &
        !$omp&   w%cw4, w%cb4, w%cpsh, w%cpsc, w%ccmp, w%cmean, w%cout_sh, w%cout_sc, w%lcmp_k)
        w%on_device = .true.
#endif
    end subroutine cue_device_put

    !> Frees the weights (and their device copies).
    subroutine cue_unload(w)
        type(cue_weights_t), intent(inout) :: w
#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET)
        if (w%on_device) then
            !$omp target exit data map(delete: w%w1, w%w2, w%w3, w%bias, w%alpha, w%beta, w%in_shift, w%in_scale, &
            !$omp&   w%lw4, w%lb4, w%lpsh, w%lpsc, w%lcmp, w%lmean, w%lout_sh, w%lout_sc, w%npca, w%ncount, w%loff, &
            !$omp&   w%cw4, w%cb4, w%cpsh, w%cpsc, w%ccmp, w%cmean, w%cout_sh, w%cout_sc, w%lcmp_k, w%w2d, w%w3d)
        end if
#elif defined(OMP_TARGET)
        if (w%on_device) then
            !$omp target exit data map(delete: w%w1, w%w2, w%w3, w%bias, w%alpha, w%beta, w%in_shift, w%in_scale, &
            !$omp&   w%lw4, w%lb4, w%lpsh, w%lpsc, w%lcmp, w%lmean, w%lout_sh, w%lout_sc, w%npca, w%ncount, w%loff, &
            !$omp&   w%cw4, w%cb4, w%cpsh, w%cpsc, w%ccmp, w%cmean, w%cout_sh, w%cout_sc, w%lcmp_k)
        end if
#endif
        w%on_device = .false.
        w%loaded = .false.
        if (allocated(w%w1)) deallocate(w%w1, w%w2, w%w3, w%bias, w%alpha, w%beta, w%in_shift, w%in_scale)
        if (allocated(w%w2d)) deallocate(w%w2d, w%w3d)
        if (allocated(w%lw4)) deallocate(w%lw4, w%lb4, w%lpsh, w%lpsc, w%lcmp, w%lmean, w%lout_sh, w%lout_sc)
        if (allocated(w%npca)) deallocate(w%npca, w%ncount, w%loff)
        if (allocated(w%line_lam)) deallocate(w%line_lam)
        if (allocated(w%cw4)) deallocate(w%cw4, w%cb4, w%cpsh, w%cpsc, w%ccmp, w%cmean, w%cout_sh, w%cout_sc)
        if (allocated(w%lcmp_k)) deallocate(w%lcmp_k)
        if (allocated(w%cont_lam)) deallocate(w%cont_lam)
    end subroutine cue_unload

    subroutine read1(lib, name, a1, n1, out)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name, a1
        integer, intent(in) :: n1
        real(WP), allocatable, intent(out) :: out(:)
        integer(int64) :: off
        character(len=32) :: ax(1)
        ax(1) = a1
        call datalib_check_array(lib, name, ax, [n1], off)
        allocate(out(n1))
        call datalib_read_real(lib, name, int(n1, int64), out, off)
    end subroutine read1

    subroutine read2(lib, name, a1, a2, n1, n2, out)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name, a1, a2
        integer, intent(in) :: n1, n2
        real(WP), allocatable, intent(out) :: out(:,:)
        integer(int64) :: off
        character(len=32) :: ax(2)
        ax(1) = a1; ax(2) = a2
        call datalib_check_array(lib, name, ax, [n1, n2], off)
        allocate(out(n2, n1))
        call datalib_read_real(lib, name, int(n1, int64) * n2, out, off)
    end subroutine read2

    subroutine read3(lib, name, a1, a2, a3, n1, n2, n3, out)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name, a1, a2, a3
        integer, intent(in) :: n1, n2, n3
        real(WP), allocatable, intent(out) :: out(:,:,:)
        integer(int64) :: off
        character(len=32) :: ax(3)
        ax(1) = a1; ax(2) = a2; ax(3) = a3
        call datalib_check_array(lib, name, ax, [n1, n2, n3], off)
        allocate(out(n3, n2, n1))
        call datalib_read_real(lib, name, int(n1, int64) * n2 * n3, out, off)
    end subroutine read3

    subroutine read_int1(lib, name, a1, n1, out)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name, a1
        integer, intent(in) :: n1
        integer, intent(out) :: out(n1)
        integer(int64) :: off
        character(len=32) :: ax(1)
        ax(1) = a1
        call datalib_check_array(lib, name, ax, [n1], off)
        call datalib_read_int(lib, name, int(n1, int64), out, off)
    end subroutine read_int1

    ! ==========================================================================================
    ! Network inputs
    ! ==========================================================================================

    !> The 12 network inputs of n models: the 7 shape parameters ion(7, n) and the gas
    !> parameters gas(5, n) = (log U, log n_H, log Z_gas/Z_sun, log N/O, log C/O relative to solar);
    !> logq(n) = log Q(U, n_H), the photon rate the outputs are normalized to.
    subroutine cue_inputs(n, ion, gas, theta, logq)
        integer, intent(in)   :: n
        real(WP), intent(in)  :: ion(CUE_NION, n), gas(5, n)
        real(WP), intent(out) :: theta(CUE_NIN, n), logq(n)
        integer :: m, k

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(k)
#elif defined(_OPENMP)
        !$omp parallel do private(k)
#endif
        do m = 1, n
            do k = 1, CUE_NION
                theta(k, m) = ion(k, m)
            end do
            logq(m) = gas(1, m) + gas(2, m) + CUE_LOGQ_CONST
            theta(8, m) = logq(m)
            theta(9, m) = 10.0_wp**gas(2, m)
            theta(10, m) = gas(3, m)
            theta(11, m) = gas(4, m)
            theta(12, m) = gas(5, m)
        end do
    end subroutine cue_inputs

    ! ==========================================================================================
    ! The networks
    ! ==========================================================================================

    !> log10 of the line luminosities [erg/s] line_log(nlines, n), in the networks' output order,
    !> and of the continuum [erg/s/Hz] cont_log(ncw, n) on cue's grid, for the network inputs
    !> theta(12, n), as cue's predict_lines / predict_cont before their normalization.
    !> fp64: the hidden layers in FP64 also in FP32 builds (the weights, FP32 as trained, are
    !> converted exactly as read); else in the kernel precision (the same in FP64 builds).
    subroutine cue_forward(w, n, theta, line_log, cont_log, chunk, fp64)
        type(cue_weights_t), intent(in) :: w
        integer, intent(in)   :: n
        real(WP), intent(in)  :: theta(CUE_NIN, n)
        real(WP), intent(out) :: line_log(w%nlines, n), cont_log(w%ncw, n)
        integer, intent(in), optional :: chunk   !< models per pass (default CUE_CHUNK)
        logical, intent(in), optional :: fp64

        real(KP), allocatable :: ha(:,:,:), hb(:,:,:), x0(:,:,:)
        real(WP), allocatable :: hd(:,:,:), hd2(:,:,:), x0d(:,:,:), lpc(:,:,:), cpc(:,:)
        integer :: c0, nc, ch
        logical :: dbl

        ch = CUE_CHUNK
        if (present(chunk)) ch = max(1, chunk)
        ch = min(n, ch)
        if (ch < 1) return
        ! (in FP64 builds both branches compute the same thing; fp64 selects the FP64 one there too)
        dbl = .false.
        if (present(fp64)) dbl = fp64
        if (dbl) then
            allocate(hd(CUE_NHID, ch, CUE_NNET), lpc(CUE_NPCA_LINE, ch, CUE_NGROUP), cpc(w%npc, ch), &
                     hd2(CUE_NHID, ch, CUE_NNET), x0d(CUE_NIN, ch, CUE_NNET), ha(1, 1, 1), hb(1, 1, 1), x0(1, 1, 1))
        else
            ! (x0, free after the first layer, holds the lines' PCA coefficients)
            allocate(ha(CUE_NHID, ch, CUE_NNET), hb(CUE_NHID, ch, CUE_NNET), &
                     x0(max(CUE_NIN * CUE_NNET, CUE_NPCA_LINE * CUE_NGROUP), ch, 1), &
                     hd(CUE_NHID, ch, 1), lpc(1, 1, 1), cpc(w%npc, ch), hd2(1, 1, 1), x0d(1, 1, 1))
        end if
#ifdef OMP_TARGET
        !$omp target data map(alloc: ha, hb, x0, hd, hd2, x0d, lpc, cpc)
#endif
        do c0 = 0, n - 1, ch
            nc = min(ch, n - c0)
            if (dbl) then
                call net_inputs_d(nc, theta(1, c0 + 1), w%in_shift, w%in_scale, x0d, ch)
                call dense_act_d(CUE_NIN, nc, ch, w%w1, w%bias(1, 1, 1), w%alpha(1, 1, 1), w%beta(1, 1, 1), x0d, hd)
#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET) && !defined(OPENSPS_HIP)
                call dense_act_d_blas(nc, ch, w%w2d, w%bias(1, 1, 2), w%alpha(1, 1, 2), w%beta(1, 1, 2), hd, hd2)
                call dense_act_d_blas(nc, ch, w%w3d, w%bias(1, 1, 3), w%alpha(1, 1, 3), w%beta(1, 1, 3), hd2, hd)
#else
                call dense_act_d(CUE_NHID, nc, ch, w%w2, w%bias(1, 1, 2), w%alpha(1, 1, 2), w%beta(1, 1, 2), hd, hd2)
                call dense_act_d(CUE_NHID, nc, ch, w%w3, w%bias(1, 1, 3), w%alpha(1, 1, 3), w%beta(1, 1, 3), hd2, hd)
#endif
            else
                call net_inputs(nc, theta(1, c0 + 1), w%in_shift, w%in_scale, x0, ch)
                call dense_act(CUE_NIN, nc, ch, w%w1, w%bias(1, 1, 1), w%alpha(1, 1, 1), w%beta(1, 1, 1), x0, ha)
                call dense_act(CUE_NHID, nc, ch, w%w2, w%bias(1, 1, 2), w%alpha(1, 1, 2), w%beta(1, 1, 2), ha, hb)
                call dense_act(CUE_NHID, nc, ch, w%w3, w%bias(1, 1, 3), w%alpha(1, 1, 3), w%beta(1, 1, 3), hb, ha)
                ! The lines' output layers in KP (their error is the hidden layers'), the PCA means and
                ! the output affine in FP64; the continuum's output layer and PCA in FP64 (in KP they
                ! would add ~4e-5 dex): its network's last layer widened
                call line_outputs_k(w%nlines, w%nlmax, nc, ch, ha, w%lw4, w%lb4, w%lpsh, w%lpsc, w%lcmp_k, &
                                    w%lmean, w%lout_sh, w%lout_sc, w%npca, w%ncount, w%loff, x0, line_log(1, c0 + 1))
                call widen(nc, ch, ha(1, 1, CUE_NNET), hd)
                call cont_outputs(w%npc, w%ncw, nc, ch, hd, w%cw4, w%cb4, w%cpsh, w%cpsc, w%ccmp, w%cmean, &
                                  w%cout_sh, w%cout_sc, cpc, cont_log(1, c0 + 1))
                cycle
            end if
            call line_outputs(w%nlines, w%nlmax, nc, ch, hd, w%lw4, w%lb4, w%lpsh, w%lpsc, w%lcmp, w%lmean, &
                              w%lout_sh, w%lout_sc, w%npca, w%ncount, w%loff, lpc, line_log(1, c0 + 1))
            call cont_outputs(w%npc, w%ncw, nc, ch, hd(1, 1, CUE_NNET), w%cw4, w%cb4, w%cpsh, w%cpsc, w%ccmp, &
                              w%cmean, w%cout_sh, w%cout_sc, cpc, cont_log(1, c0 + 1))
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine cue_forward

    !> One network's last hidden layer in FP64, for its output layer.
    subroutine widen(nc, ch, h, hd)
        integer, intent(in)   :: nc, ch
        real(KP), intent(in)  :: h(CUE_NHID, ch)
        real(WP), intent(out) :: hd(CUE_NHID, ch)
        integer :: m, u
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, nc
            do u = 1, CUE_NHID
                hd(u, m) = real(h(u, m), WP)
            end do
        end do
    end subroutine widen

    !> net_inputs in FP64
    subroutine net_inputs_d(nc, theta, sh, sc, x0, ch)
        integer, intent(in)   :: nc, ch
        real(WP), intent(in)  :: theta(CUE_NIN, nc)
        real(KP), intent(in)  :: sh(CUE_NIN, CUE_NNET), sc(CUE_NIN, CUE_NNET)
        real(WP), intent(out) :: x0(CUE_NIN, ch, CUE_NNET)
        integer :: g, m, k

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do g = 1, CUE_NNET
            do m = 1, nc
                do k = 1, CUE_NIN
                    x0(k, m, g) = (theta(k, m) - real(sh(k, g), WP)) / real(sc(k, g), WP)
                end do
            end do
        end do
    end subroutine net_inputs_d

    !> dense_act with FP64 activations (the FP32 weights converted exactly)
    subroutine dense_act_d(nin, nc, ch, w, b, al, be, x, y)
        integer, intent(in)   :: nin, nc, ch
        real(KP), intent(in)  :: w(CUE_NHID, nin, CUE_NNET), b(CUE_NHID, CUE_NNET)
        real(KP), intent(in)  :: al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET)
        real(WP), intent(in)  :: x(nin, ch, CUE_NNET)
        real(WP), intent(out) :: y(CUE_NHID, ch, CUE_NNET)
        integer  :: g, mt, u, k, m0, ntile, m
        real(WP) :: a1, a2, a3, a4, wv, aa, bb

#if defined(OPENSPS_HIP) && defined(OMP_TARGET)
        call hip_dense_d(nin, nc, ch, w, b, al, be, x, y)
        return
#endif
        ntile = (nc + TM - 1) / TM
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(m0, a1, a2, a3, a4, wv, k, aa, m, bb)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(m0, a1, a2, a3, a4, wv, k, aa, m, bb)
#endif
        do g = 1, CUE_NNET
            do mt = 1, ntile
                do u = 1, CUE_NHID
                    m0 = (mt - 1) * TM
                    a1 = 0.0_wp; a2 = 0.0_wp; a3 = 0.0_wp; a4 = 0.0_wp
                    if (m0 + TM <= nc) then
                        do k = 1, nin
                            wv = real(w(u, k, g), WP)
                            a1 = a1 + wv * x(k, m0 + 1, g)
                            a2 = a2 + wv * x(k, m0 + 2, g)
                            a3 = a3 + wv * x(k, m0 + 3, g)
                            a4 = a4 + wv * x(k, m0 + 4, g)
                        end do
                    else
                        do k = 1, nin
                            wv = real(w(u, k, g), WP)
                            a1 = a1 + wv * x(k, m0 + 1, g)
                            if (m0 + 2 <= nc) a2 = a2 + wv * x(k, m0 + 2, g)
                            if (m0 + 3 <= nc) a3 = a3 + wv * x(k, m0 + 3, g)
                        end do
                    end if
                    bb = real(be(u, g), WP)
                    do m = 1, min(TM, nc - m0)
                        select case (m)
                        case (1); aa = a1
                        case (2); aa = a2
                        case (3); aa = a3
                        case default; aa = a4
                        end select
                        aa = aa + real(b(u, g), WP)
                        y(u, m0 + m, g) = (bb + (1.0_wp - bb) / (1.0_wp + exp(-real(al(u, g), WP) * aa))) * aa
                    end do
                end do
            end do
        end do
    end subroutine dense_act_d

    !> Normalized inputs of every network: x0(k, m, net) = (theta(k, m) - shift(k, net)) / scale(k, net)
    subroutine net_inputs(nc, theta, sh, sc, x0, ch)
        integer, intent(in)   :: nc, ch
        real(WP), intent(in)  :: theta(CUE_NIN, nc)
        real(KP), intent(in)  :: sh(CUE_NIN, CUE_NNET), sc(CUE_NIN, CUE_NNET)
        real(KP), intent(out) :: x0(CUE_NIN, ch, CUE_NNET)
        integer :: g, m, k

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do g = 1, CUE_NNET
            do m = 1, nc
                do k = 1, CUE_NIN
                    x0(k, m, g) = real((theta(k, m) - real(sh(k, g), WP)) / real(sc(k, g), WP), KP)
                end do
            end do
        end do
    end subroutine net_inputs

    !> One hidden layer of all 17 networks: y(u, m, g) = act(b(u, g) + sum_k w(u, k, g) x(k, m, g)),
    !> act(a) = (beta + (1 - beta) / (1 + exp(-alpha a))) a. Each thread computes TM models of one
    !> unit, so a weight is loaded once per TM products (the inputs are shared by the threads of a
    !> unit block). With a GPU BLAS the products are one strided batched GEMM over the networks.
    subroutine dense_act(nin, nc, ch, w, b, al, be, x, y)
        integer, intent(in)   :: nin, nc, ch
        real(KP), intent(in)  :: w(CUE_NHID, nin, CUE_NNET), b(CUE_NHID, CUE_NNET)
        real(KP), intent(in)  :: al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET)
        real(KP), intent(in)  :: x(nin, ch, CUE_NNET)
        real(KP), intent(out) :: y(CUE_NHID, ch, CUE_NNET)
        integer  :: g, mt, u, k, m0, ntile, m
        real(KP) :: a1, a2, a3, a4, wv, aa

#if defined(OPENSPS_HIP) && defined(OMP_TARGET)
        call hip_dense(nin, nc, ch, w, b, al, be, x, y)
        return
#endif
#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET)
        if (nin == CUE_NHID) then
            call gemm_layer(nin, ch, w, x, y)
            !$omp target teams distribute parallel do collapse(3) private(aa)
            do g = 1, CUE_NNET
                do m = 1, nc
                    do u = 1, CUE_NHID
                        aa = y(u, m, g) + b(u, g)
                        y(u, m, g) = (be(u, g) + (1.0_KP - be(u, g)) / (1.0_KP + exp(-al(u, g) * aa))) * aa
                    end do
                end do
            end do
            return
        end if
#endif
        ntile = (nc + TM - 1) / TM
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(m0, a1, a2, a3, a4, wv, k, aa, m)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(m0, a1, a2, a3, a4, wv, k, aa, m)
#endif
        do g = 1, CUE_NNET
            do mt = 1, ntile
                do u = 1, CUE_NHID
                    m0 = (mt - 1) * TM
                    a1 = 0.0_KP; a2 = 0.0_KP; a3 = 0.0_KP; a4 = 0.0_KP
                    if (m0 + TM <= nc) then
                        do k = 1, nin
                            wv = w(u, k, g)
                            a1 = a1 + wv * x(k, m0 + 1, g)
                            a2 = a2 + wv * x(k, m0 + 2, g)
                            a3 = a3 + wv * x(k, m0 + 3, g)
                            a4 = a4 + wv * x(k, m0 + 4, g)
                        end do
                    else
                        do k = 1, nin
                            wv = w(u, k, g)
                            a1 = a1 + wv * x(k, m0 + 1, g)
                            if (m0 + 2 <= nc) a2 = a2 + wv * x(k, m0 + 2, g)
                            if (m0 + 3 <= nc) a3 = a3 + wv * x(k, m0 + 3, g)
                        end do
                    end if
                    do m = 1, min(TM, nc - m0)
                        select case (m)
                        case (1); aa = a1
                        case (2); aa = a2
                        case (3); aa = a3
                        case default; aa = a4
                        end select
                        aa = aa + b(u, g)
                        y(u, m0 + m, g) = (be(u, g) + (1.0_KP - be(u, g)) / (1.0_KP + exp(-al(u, g) * aa))) * aa
                    end do
                end do
            end do
        end do
    end subroutine dense_act

#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET)
    !> y(:, :, g) = w(:, :, g) x(:, :, g) for the 17 networks: one strided batched GEMM.
    subroutine gemm_layer(nin, ch, w, x, y)
        use, intrinsic :: iso_c_binding, only: c_loc, c_long_long
        integer, intent(in) :: nin, ch
        real(KP), intent(in), target :: w(CUE_NHID, nin, CUE_NNET), x(nin, ch, CUE_NNET)
        real(KP), intent(inout), target :: y(CUE_NHID, ch, CUE_NNET)

        call accel_blas_init()
        !$omp target data use_device_addr(w, x, y)
        call accel_blas_gemm_strided_batched(ACCEL_BLAS_OP_N, ACCEL_BLAS_OP_N, CUE_NHID, ch, nin, 1.0_KP, &
                                             c_loc(w), CUE_NHID, int(CUE_NHID, c_long_long) * nin, &
                                             c_loc(x), nin, int(nin, c_long_long) * ch, &
                                             0.0_KP, c_loc(y), CUE_NHID, int(CUE_NHID, c_long_long) * ch, CUE_NNET)
        !$omp end target data
    end subroutine gemm_layer
#endif

#if defined(OPENSPS_ACCEL_BLAS) && defined(OMP_TARGET)
    !> A 256 -> 256 layer of all 17 networks in FP64 through the BLAS (one strided batched DGEMM),
    !> then the bias and the activation.
    subroutine dense_act_d_blas(nc, ch, wd, b, al, be, x, y)
        use, intrinsic :: iso_c_binding, only: c_loc, c_long_long
        integer, intent(in) :: nc, ch
        real(WP), intent(in), target :: wd(CUE_NHID, CUE_NHID, CUE_NNET), x(CUE_NHID, ch, CUE_NNET)
        real(KP), intent(in) :: b(CUE_NHID, CUE_NNET), al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET)
        real(WP), intent(inout), target :: y(CUE_NHID, ch, CUE_NNET)
        integer  :: g, m, u
        real(WP) :: aa, bb

        call accel_blas_init()
        !$omp target data use_device_addr(wd, x, y)
        call accel_blas_dgemm_strided_batched(ACCEL_BLAS_OP_N, ACCEL_BLAS_OP_N, CUE_NHID, ch, CUE_NHID, 1.0_wp, &
                                              c_loc(wd), CUE_NHID, int(CUE_NHID, c_long_long) * CUE_NHID, &
                                              c_loc(x), CUE_NHID, int(CUE_NHID, c_long_long) * ch, &
                                              0.0_wp, c_loc(y), CUE_NHID, int(CUE_NHID, c_long_long) * ch, CUE_NNET)
        !$omp end target data
        !$omp target teams distribute parallel do collapse(3) private(aa, bb)
        do g = 1, CUE_NNET
            do m = 1, nc
                do u = 1, CUE_NHID
                    aa = y(u, m, g) + real(b(u, g), WP)
                    bb = real(be(u, g), WP)
                    y(u, m, g) = (bb + (1.0_wp - bb) / (1.0_wp + exp(-real(al(u, g), WP) * aa))) * aa
                end do
            end do
        end do
    end subroutine dense_act_d_blas
#endif

#if defined(OPENSPS_HIP) && defined(OMP_TARGET)
    !> dense_act by the HIP kernel (KP weights and activations)
    subroutine hip_dense(nin, nc, ch, w, b, al, be, x, y)
        integer, intent(in) :: nin, nc, ch
        real(KP), intent(in), target :: w(CUE_NHID, nin, CUE_NNET), b(CUE_NHID, CUE_NNET)
        real(KP), intent(in), target :: al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET), x(nin, ch, CUE_NNET)
        real(KP), intent(inout), target :: y(CUE_NHID, ch, CUE_NNET)
        !$omp target data use_device_addr(w, b, al, be, x, y)
        if (kind(1.0_KP) == kind(1.0_WP)) then
            call hip_cue_dense_dd(nin, nc, ch, c_loc(w), c_loc(b), c_loc(al), c_loc(be), c_loc(x), c_loc(y), c_null_ptr)
        else
            call hip_cue_dense_ss(nin, nc, ch, c_loc(w), c_loc(b), c_loc(al), c_loc(be), c_loc(x), c_loc(y), c_null_ptr)
        end if
        !$omp end target data
    end subroutine hip_dense

    !> dense_act_d by the HIP kernel (KP weights, FP64 activations)
    subroutine hip_dense_d(nin, nc, ch, w, b, al, be, x, y)
        integer, intent(in) :: nin, nc, ch
        real(KP), intent(in), target :: w(CUE_NHID, nin, CUE_NNET), b(CUE_NHID, CUE_NNET)
        real(KP), intent(in), target :: al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET)
        real(WP), intent(in), target :: x(nin, ch, CUE_NNET)
        real(WP), intent(inout), target :: y(CUE_NHID, ch, CUE_NNET)
        !$omp target data use_device_addr(w, b, al, be, x, y)
        if (kind(1.0_KP) == kind(1.0_WP)) then
            call hip_cue_dense_dd(nin, nc, ch, c_loc(w), c_loc(b), c_loc(al), c_loc(be), c_loc(x), c_loc(y), c_null_ptr)
        else
            call hip_cue_dense_sd(nin, nc, ch, c_loc(w), c_loc(b), c_loc(al), c_loc(be), c_loc(x), c_loc(y), c_null_ptr)
        end if
        !$omp end target data
    end subroutine hip_dense_d

    !> The reverse pass's layer: the pre-activation a = b + W x, and y = act(a) (FP64; y not formed
    !> when with_act is false) by the HIP kernel
    subroutine hip_dense_pre(nin, nc, ch, w, b, al, be, x, a, y, with_act)
        integer, intent(in) :: nin, nc, ch
        real(KP), intent(in), target :: w(CUE_NHID, nin, CUE_NNET), b(CUE_NHID, CUE_NNET)
        real(KP), intent(in), target :: al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET)
        real(WP), intent(in), target :: x(nin, ch, CUE_NNET)
        real(WP), intent(inout), target :: a(CUE_NHID, ch, CUE_NNET), y(CUE_NHID, ch, CUE_NNET)
        logical, intent(in) :: with_act
        type(c_ptr) :: py
        !$omp target data use_device_addr(w, b, al, be, x, a, y)
        py = c_null_ptr
        if (with_act) py = c_loc(y)
        if (kind(1.0_KP) == kind(1.0_WP)) then
            call hip_cue_pre_dd(nin, nc, ch, c_loc(w), c_loc(b), c_loc(al), c_loc(be), c_loc(x), py, c_loc(a), c_null_ptr)
        else
            call hip_cue_pre_sd(nin, nc, ch, c_loc(w), c_loc(b), c_loc(al), c_loc(be), c_loc(x), py, c_loc(a), c_null_ptr)
        end if
        !$omp end target data
    end subroutine hip_dense_pre

    !> din = W^T dout (256 x 256 layers) by the HIP kernel
    subroutine hip_dense_back(nc, ch, w, dout, din)
        integer, intent(in) :: nc, ch
        real(KP), intent(in), target :: w(CUE_NHID, CUE_NHID, CUE_NNET)
        real(WP), intent(in), target :: dout(CUE_NHID, ch, CUE_NNET)
        real(WP), intent(inout), target :: din(CUE_NHID, ch, CUE_NNET)
        !$omp target data use_device_addr(w, dout, din)
        if (kind(1.0_KP) == kind(1.0_WP)) then
            call hip_cue_back_dd(nc, ch, c_loc(w), c_loc(dout), c_loc(din), c_null_ptr)
        else
            call hip_cue_back_sd(nc, ch, c_loc(w), c_loc(dout), c_loc(din), c_null_ptr)
        end if
        !$omp end target data
    end subroutine hip_dense_back
#endif

    !> The line networks' output layer and PCA bases: per group g, the PCA coefficients
    !> c = (W4 h + b4) scale + shift, then log L(line) = (c . components + mean) out_scale + out_shift.
    subroutine line_outputs(nlines, nlmax, nc, ch, h, lw4, lb4, lpsh, lpsc, lcmp, lmean, osh, osc, npca, &
                            ncount, loff, lpc, line_log)
        integer, intent(in)   :: nlines, nlmax, nc, ch
        real(WP), intent(in)  :: h(CUE_NHID, ch, CUE_NNET)
        real(KP), intent(in)  :: lw4(CUE_NPCA_LINE, CUE_NHID, CUE_NGROUP)
        real(WP), intent(in)  :: lb4(CUE_NPCA_LINE, CUE_NGROUP)
        real(WP), intent(in)  :: lpsh(CUE_NPCA_LINE, CUE_NGROUP), lpsc(CUE_NPCA_LINE, CUE_NGROUP)
        real(WP), intent(in)  :: lcmp(nlmax, CUE_NPCA_LINE, CUE_NGROUP), lmean(nlmax, CUE_NGROUP)
        real(WP), intent(in)  :: osh(nlmax, CUE_NGROUP), osc(nlmax, CUE_NGROUP)
        integer, intent(in)   :: npca(CUE_NGROUP), ncount(CUE_NGROUP), loff(CUE_NGROUP)
        real(WP), intent(out) :: lpc(CUE_NPCA_LINE, ch, CUE_NGROUP)
        real(WP), intent(inout) :: line_log(nlines, nc)
        integer  :: g, m, p, k, s
        real(WP) :: a, y

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(a, k)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(a, k)
#endif
        do g = 1, CUE_NGROUP
            do m = 1, nc
                do p = 1, CUE_NPCA_LINE
                    a = 0.0_wp
                    if (p <= npca(g)) then
                        do k = 1, CUE_NHID
                            a = a + real(lw4(p, k, g), WP) * h(k, m, g)
                        end do
                        a = (a + lb4(p, g)) * lpsc(p, g) + lpsh(p, g)
                    end if
                    lpc(p, m, g) = a
                end do
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(y, p)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(y, p)
#endif
        do g = 1, CUE_NGROUP
            do m = 1, nc
                do s = 1, nlmax
                    if (s <= ncount(g)) then
                        y = lmean(s, g)
                        do p = 1, npca(g)
                            y = y + lpc(p, m, g) * lcmp(s, p, g)
                        end do
                        line_log(loff(g) + s, m) = y * osc(s, g) + osh(s, g)
                    end if
                end do
            end do
        end do
    end subroutine line_outputs

    !> The continuum network's output layer and PCA basis: log L_nu on cue's grid.
    subroutine cont_outputs(npc, ncw, nc, ch, h, cw4, cb4, cpsh, cpsc, ccmp, cmean, osh, osc, cpc, cont_log)
        integer, intent(in)   :: npc, ncw, nc, ch
        real(WP), intent(in)  :: h(CUE_NHID, ch)   !< the continuum network's last hidden layer
        real(KP), intent(in)  :: cw4(npc, CUE_NHID)
        real(WP), intent(in)  :: cb4(npc), cpsh(npc), cpsc(npc), ccmp(ncw, npc), cmean(ncw)
        real(WP), intent(in)  :: osh(ncw), osc(ncw)
        real(WP), intent(out) :: cpc(npc, ch)
        real(WP), intent(inout) :: cont_log(ncw, nc)
        integer  :: m, p, k, i
        real(WP) :: a, y

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(a, k)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(a, k)
#endif
        do m = 1, nc
            do p = 1, npc
                a = 0.0_wp
                do k = 1, CUE_NHID
                    a = a + real(cw4(p, k), WP) * h(k, m)
                end do
                cpc(p, m) = (a + cb4(p)) * cpsc(p) + cpsh(p)
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(y, p)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(y, p)
#endif
        do m = 1, nc
            do i = 1, ncw
                y = cmean(i)
                do p = 1, npc
                    y = y + cpc(p, m) * ccmp(i, p)
                end do
                cont_log(i, m) = y * osc(i) + osh(i)
            end do
        end do
    end subroutine cont_outputs

    !> line_outputs with the hidden layer, the output layer and the PCA coefficients in KP (FP32
    !> hidden layers): only the PCA mean (~50-100), the output scale and shift are FP64.
    subroutine line_outputs_k(nlines, nlmax, nc, ch, h, lw4, lb4, lpsh, lpsc, lcmp, lmean, osh, osc, npca, &
                              ncount, loff, lpc, line_log)
        integer, intent(in)   :: nlines, nlmax, nc, ch
        real(KP), intent(in)  :: h(CUE_NHID, ch, CUE_NNET)
        real(KP), intent(in)  :: lw4(CUE_NPCA_LINE, CUE_NHID, CUE_NGROUP)
        real(WP), intent(in)  :: lb4(CUE_NPCA_LINE, CUE_NGROUP)
        real(WP), intent(in)  :: lpsh(CUE_NPCA_LINE, CUE_NGROUP), lpsc(CUE_NPCA_LINE, CUE_NGROUP)
        real(KP), intent(in)  :: lcmp(nlmax, CUE_NPCA_LINE, CUE_NGROUP)
        real(WP), intent(in)  :: lmean(nlmax, CUE_NGROUP)
        real(WP), intent(in)  :: osh(nlmax, CUE_NGROUP), osc(nlmax, CUE_NGROUP)
        integer, intent(in)   :: npca(CUE_NGROUP), ncount(CUE_NGROUP), loff(CUE_NGROUP)
        real(KP), intent(out) :: lpc(CUE_NPCA_LINE, ch, CUE_NGROUP)
        real(WP), intent(inout) :: line_log(nlines, nc)
        integer  :: g, m, p, k, s
        real(KP) :: a, y

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(a, k)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(a, k)
#endif
        do g = 1, CUE_NGROUP
            do m = 1, nc
                do p = 1, CUE_NPCA_LINE
                    a = 0.0_KP
                    if (p <= npca(g)) then
                        do k = 1, CUE_NHID
                            a = a + lw4(p, k, g) * h(k, m, g)
                        end do
                        a = (a + real(lb4(p, g), KP)) * real(lpsc(p, g), KP) + real(lpsh(p, g), KP)
                    end if
                    lpc(p, m, g) = a
                end do
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(y, p)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(y, p)
#endif
        do g = 1, CUE_NGROUP
            do m = 1, nc
                do s = 1, nlmax
                    if (s <= ncount(g)) then
                        y = 0.0_KP
                        do p = 1, npca(g)
                            y = y + lpc(p, m, g) * lcmp(s, p, g)
                        end do
                        line_log(loff(g) + s, m) = (lmean(s, g) + real(y, WP)) * osc(s, g) + osh(s, g)
                    end if
                end do
            end do
        end do
    end subroutine line_outputs_k


    ! ==========================================================================================
    ! The ionizing spectrum as a 4-part power law (cue.utils.fit_4loglinear_ionparam)
    ! ==========================================================================================

    !> The shape parameters ion(7, n) of n ionizing spectra spec(nw, n) [L_nu, any units] on the
    !> wavelength grid lam(nw) [A, increasing, reaching past 911.6 A]: one thread per spectrum.
    subroutine cue_ionizing_params(n, nw, lam, spec, ion)
        integer, intent(in)   :: n, nw
        real(WP), intent(in)  :: lam(nw)
        real(KP), intent(in)  :: spec(nw, n)
        real(WP), intent(out) :: ion(CUE_NION, n)
        real(WP) :: llam(nw)
        integer :: m

        llam = log10(lam)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do map(to: llam)
#elif defined(_OPENMP)
        !$omp parallel do
#endif
        do m = 1, n
            call cue_fit_ionizing(nw, lam, spec(1, m), ion(1, m), llam)
        end do
    end subroutine cue_ionizing_params

    !> cue's fit of one spectrum (cue.utils.fit_4loglinear_ionparam): in each segment (pixels in
    !> (edge_{s-1}, edge_s], the first from the grid's start) a power law F_nu = A lambda^alpha,
    !> minimizing 0.5 sum (log F - log A - alpha log lambda)^2 + 0.5 N (log Q - log Q_fit)^2 with
    !> Q = |trapz(F lambda / (h c), c / lambda)| and Q_fit = A (lambda_max^alpha -
    !> lambda_min^alpha) / (h alpha); then the log ratios of the segments' integrated
    !> luminosities over [1, 228, 353, 504, 911.6] A. All clipped to cue's training ranges.
    !> cue minimizes with SLSQP from an endpoint guess; here Newton's method on the same loss
    !> converges to its minimum to rounding. Differences from cue: a segment whose neighbour is
    !> also empty gets the ratio 0 (cue: NaN), and a one-pixel segment the slope 0.
    subroutine cue_fit_ionizing(nw, lam, spec, ion, llam)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)   :: nw
        real(WP), intent(in)  :: lam(nw)
        real(KP), intent(in)  :: spec(nw)
        real(WP), intent(out) :: ion(CUE_NION)
        real(WP), intent(in), optional :: llam(nw)   !< log10(lam), if known

        real(WP) :: edge(5), alpha(4), loga(4), logl(4), fnorm, floor_, ref
        logical  :: empty(4)
        integer  :: s, i0, i1, ilim, i, k

        edge(1) = 1.0_wp
        edge(2) = CUE_EDGE_HEII
        edge(3) = CUE_EDGE_OII
        edge(4) = CUE_EDGE_HEI
        edge(5) = CUE_EDGE_HI

        ! cue scales the spectrum by 1e-18 / F(first pixel beyond 911.6 A) and floors it at 1e-70
        ! of that scale; the floor matters only for empty pixels inside a fitted segment
        ilim = 0
        do i = 1, nw
            if (lam(i) <= CUE_EDGE_HI) ilim = i
        end do
        ref = 0.0_wp
        if (ilim < nw) ref = real(spec(ilim + 1), WP)
        if (ref <= 0.0_wp) then
            do i = 1, nw
                ref = max(ref, real(spec(i), WP))
            end do
        end if
        if (ref <= 0.0_wp) then
            do k = 1, CUE_NION
                ion(k) = cue_ion_bound(k, .false.)
            end do
            return
        end if
        fnorm = 1.0e-18_wp / ref
        floor_ = 1.0e-70_wp * fnorm

        i1 = 0
        do s = 1, 4
            i0 = i1 + 1
            i1 = 0
            do i = 1, nw
                if (lam(i) <= edge(s + 1)) i1 = i
            end do
            empty(s) = .true.
            do i = i0, i1
                if (spec(i) > 0.0_KP) empty(s) = .false.
            end do
            alpha(s) = 0.0_wp
            loga(s) = 0.0_wp
            if (.not. empty(s)) then
                if (present(llam)) then
                    call fit_segment(nw, lam, spec, i0, i1, fnorm, floor_, alpha(s), loga(s), llam)
                else
                    call fit_segment(nw, lam, spec, i0, i1, fnorm, floor_, alpha(s), loga(s))
                end if
                loga(s) = loga(s) - log10(fnorm)
            end if
        end do

        ! log of each segment's integrated luminosity (cue.utils.Ltotal over [1, edges]); the common
        ! constant log10(c L_sun) cancels in the ratios
        do s = 1, 4
            if (empty(s)) cycle
            logl(s) = loga(s) + cue_log_ratio_power(alpha(s) - 1.0_wp, edge(s), edge(s + 1))
        end do
        do s = 1, 4
            ion(s) = min(max(alpha(s), cue_ion_bound(s, .false.)), cue_ion_bound(s, .true.))
        end do
        do s = 1, 3
            if (empty(s) .and. empty(s + 1)) then
                ion(4 + s) = 0.0_wp
            else if (empty(s)) then
                ion(4 + s) = cue_ion_bound(4 + s, .true.)
            else if (empty(s + 1)) then
                ion(4 + s) = cue_ion_bound(4 + s, .false.)
            else
                ion(4 + s) = logl(s + 1) - logl(s)
            end if
            ion(4 + s) = min(max(ion(4 + s), cue_ion_bound(4 + s, .false.)), cue_ion_bound(4 + s, .true.))
        end do
    end subroutine cue_fit_ionizing

    !> Training range of shape parameter k (cue clips its fit to it): the upper end if hi, else the lower.
    pure real(WP) function cue_ion_bound(k, hi) result(v)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in) :: k
        logical, intent(in) :: hi
        select case (k)
        case (1); v = merge(42.0_wp, 1.0_wp, hi)
        case (2); v = merge(30.0_wp, -0.3_wp, hi)
        case (3); v = merge(14.0_wp, -1.0_wp, hi)
        case (4); v = merge(8.0_wp, -1.7_wp, hi)
        case (5); v = merge(10.1_wp, -1.0_wp, hi)
        case (6); v = merge(1.9_wp, -0.5_wp, hi)
        case default; v = merge(2.2_wp, -0.4_wp, hi)
        end select
    end function cue_ion_bound

    !> log10 |(b**beta - a**beta) / beta| (0 < a < b), without cancellation at small beta:
    !> beta log10 a + log10(ln(b / a) expm1(u) / u), u = beta ln(b / a).
    pure real(WP) function cue_log_ratio_power(beta, a, b) result(v)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: beta, a, b
        real(WP) :: lr, u

        lr = log(b / a)
        u = beta * lr
        v = beta * log10(a) + log10(lr * expm1_over_x(u))
    end function cue_log_ratio_power

    !> expm1(u) / u: a series at small |u|, the quotient elsewhere.
    pure real(WP) function expm1_over_x(u) result(e)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: u
        if (abs(u) < 1.0e-3_wp) then
            ! (truncation u**5 / 720 < 2e-18)
            e = 1.0_wp + u * (0.5_wp + u * (1.0_wp / 6.0_wp + u * (1.0_wp / 24.0_wp + u / 120.0_wp)))
        else
            ! (cancellation at most eps / |u| < 3e-13 relative)
            e = (exp(u) - 1.0_wp) / u
        end if
    end function expm1_over_x

    !> Newton's method on cue's loss for one segment (pixels i0..i1), in cue's normalized units.
    !> P(alpha) = log10((x1**alpha - x0**alpha) / alpha) and its derivatives are formed from
    !> u = alpha ln(x1 / x0) without cancellation.
    subroutine fit_segment(nw, lam, spec, i0, i1, fnorm, floor_, alpha, loga, llam)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)   :: nw, i0, i1
        real(WP), intent(in)  :: lam(nw), fnorm, floor_
        real(KP), intent(in)  :: spec(nw)
        real(WP), intent(out) :: alpha, loga
        real(WP), intent(in), optional :: llam(nw)

        real(WP) :: x0, x1, lr, sm(6), logq, q, f0, f1, xa, xb, fa, fb
        real(WP) :: p, dp, d2p, t, ga, gb, haa, hab, hbb, det, da, db, lcur, lnew, step, an, bn
        real(WP) :: pn, dpn, d2pn
        integer  :: i, it, ls

        x0 = lam(i0)
        x1 = lam(i1)
        ! sm: N, mean X, mean Y, and the sums about the means of (X - mean)^2, (X - mean)(Y - mean),
        ! (Y - mean)^2 (cue: X = log10 lambda, Y = log10 of the normalized spectrum); two passes, so
        ! the loss near its minimum (residuals ~1e-2 against log A ~ -20) is free of cancellation
        sm(1) = real(i1 - i0 + 1, WP)
        sm(2:6) = 0.0_wp
        do i = i0, i1
            if (present(llam)) then
                sm(2) = sm(2) + llam(i)
            else
                sm(2) = sm(2) + log10(lam(i))
            end if
            sm(3) = sm(3) + log10(max(real(spec(i), WP) * fnorm, floor_))
        end do
        sm(2) = sm(2) / sm(1)
        sm(3) = sm(3) / sm(1)
        do i = i0, i1
            if (present(llam)) then
                xa = llam(i) - sm(2)
            else
                xa = log10(lam(i)) - sm(2)
            end if
            f0 = log10(max(real(spec(i), WP) * fnorm, floor_)) - sm(3)
            sm(4) = sm(4) + xa * xa
            sm(5) = sm(5) + xa * f0
            sm(6) = sm(6) + f0 * f0
        end do
        ! Q of the segment: trapezoid in nu = c / lambda of F lambda / (h c)
        q = 0.0_wp
        do i = i0, i1 - 1
            xa = lam(i); xb = lam(i + 1)
            fa = max(real(spec(i), WP) * fnorm, floor_) * xa / (CUE_H * CUE_C)
            fb = max(real(spec(i + 1), WP) * fnorm, floor_) * xb / (CUE_H * CUE_C)
            q = q + 0.5_wp * (fa + fb) * (CUE_C / xa - CUE_C / xb)
        end do
        q = abs(q)

        if (i1 == i0 .or. q <= 0.0_wp) then
            ! One pixel (no slope; cue's guess divides by zero) or no photons: the mean only
            alpha = 0.0_wp
            loga = sm(3)
            return
        end if
        logq = log10(q)
        lr = log(x1 / x0)
        ! Start: the least-squares line of the log data (the minimum without the photon term; cue
        ! starts from the line through the end points; both reach the same minimum)
        if (sm(4) > 0.0_wp) then
            alpha = sm(5) / sm(4)
            loga = sm(3) - alpha * sm(2)
        else
            f0 = log10(max(real(spec(i0), WP) * fnorm, floor_))
            f1 = log10(max(real(spec(i1), WP) * fnorm, floor_))
            alpha = (f1 - f0) / log10(x1 / x0)
            loga = f1 - alpha * log10(x1)
        end if
        alpha = min(max(alpha, -40.0_wp), 100.0_wp)
        loga = min(max(loga, -200.0_wp), 100.0_wp)

        call seg_p(alpha, x0, lr, p, dp, d2p)
        lcur = seg_loss_p(alpha, loga, p, logq, sm)
        do it = 1, 100
            t = loga - log10(CUE_H) + p - logq
            ! Gradient and Hessian of L = 0.5 sum (a + alpha X - Y)^2 + 0.5 N t^2, with
            ! sum r = N rbar, rbar = a + alpha mean X - mean Y, and sum r X = alpha Sxx - Sxy + mean X N rbar
            f0 = loga + alpha * sm(2) - sm(3)
            ga = (alpha * sm(4) - sm(5) + sm(2) * sm(1) * f0) + sm(1) * t * dp
            gb = sm(1) * f0 + sm(1) * t
            haa = sm(4) + sm(1) * sm(2) * sm(2) + sm(1) * (dp * dp + t * d2p)
            hab = sm(1) * sm(2) + sm(1) * dp
            hbb = 2.0_wp * sm(1)
            det = haa * hbb - hab * hab
            if (haa > 0.0_wp .and. det > 0.0_wp) then
                da = -(hbb * ga - hab * gb) / det
                db = -(haa * gb - hab * ga) / det
            else
                ! Not convex here: a gradient step scaled by the diagonal
                da = -ga / max(abs(haa), sm(4) + sm(1))
                db = -gb / hbb
            end if
            step = 1.0_wp
            lnew = lcur
            an = alpha
            bn = loga
            pn = p; dpn = dp; d2pn = d2p
            do ls = 1, 60
                an = min(max(alpha + step * da, -40.0_wp), 100.0_wp)
                bn = min(max(loga + step * db, -200.0_wp), 100.0_wp)
                call seg_p(an, x0, lr, pn, dpn, d2pn)
                lnew = seg_loss_p(an, bn, pn, logq, sm)
                if (lnew <= lcur) exit
                ! A small Newton step in the quadratic region is taken without comparing losses: there
                ! the loss changes by less than its rounding, and refusing the step would stop the
                ! iteration (at a point that depends on the order of operations, i.e. on the build)
                if (ls == 1 .and. haa > 0.0_wp .and. det > 0.0_wp .and. &
                    abs(da) <= 1.0e-6_wp * max(1.0_wp, abs(alpha)) .and. &
                    abs(db) <= 1.0e-6_wp * max(1.0_wp, abs(loga))) exit
                step = 0.5_wp * step
            end do
            if (lnew > lcur .and. step < 1.0_wp) exit
            da = an - alpha
            db = bn - loga
            alpha = an
            loga = bn
            lcur = lnew
            p = pn; dp = dpn; d2p = d2pn
            if (abs(da) <= 1.0e-13_wp * max(1.0_wp, abs(alpha)) .and. &
                abs(db) <= 1.0e-13_wp * max(1.0_wp, abs(loga))) exit
        end do
    end subroutine fit_segment

    !> P(al) = log10((x1**al - x0**al) / al) (lr = ln(x1 / x0)) and its first two derivatives.
    subroutine seg_p(al, x0, lr, p, dp, d2p)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in)  :: al, x0, lr
        real(WP), intent(out) :: p, dp, d2p
        real(WP), parameter :: LN10 = log(10.0_wp)
        real(WP) :: u, phi, dphi, em, u2

        u = al * lr
        p = al * log10(x0) + log10(lr * expm1_over_x(u))
        ! d ln f / d al = ln x0 + lr phi(u), phi(u) = 1 / (1 - e^-u) - 1 / u
        if (abs(u) < 1.0e-2_wp) then
            u2 = u * u
            phi = 0.5_wp + u * (1.0_wp / 12.0_wp - u2 * (1.0_wp / 720.0_wp - u2 / 30240.0_wp))
            dphi = 1.0_wp / 12.0_wp - u2 * (1.0_wp / 240.0_wp - u2 / 6048.0_wp)
        else
            em = exp(-u)
            phi = 1.0_wp / (1.0_wp - em) - 1.0_wp / u
            dphi = -em / ((1.0_wp - em) * (1.0_wp - em)) + 1.0_wp / (u * u)
        end if
        dp = (log(x0) + lr * phi) / LN10
        d2p = lr * lr * dphi / LN10
    end subroutine seg_p

    !> cue's loss of a segment at (al, la), from the sums sm of fit_segment and P(al) = p.
    real(WP) function seg_loss_p(al, la, p, logq, sm) result(seg_loss)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: al, la, p, logq, sm(6)
        real(WP) :: tt, rss, rb
        tt = la - log10(CUE_H) + p - logq
        ! sum (la + al X - Y)^2 about the means: N rbar^2 + al^2 Sxx - 2 al Sxy + Syy
        rb = la + al * sm(2) - sm(3)
        rss = sm(1) * rb * rb + al * al * sm(4) - 2.0_wp * al * sm(5) + sm(6)
        seg_loss = 0.5_wp * rss + 0.5_wp * sm(1) * tt * tt
    end function seg_loss_p

    ! ==========================================================================================
    ! The continuum on the output grid (cue: scipy CubicSpline, not-a-knot, extrapolated)
    ! ==========================================================================================

    !> Factors the not-a-knot spline on the knots x(n) for the output grid xout(nout); output
    !> pixels at or below 911.6 A get no continuum (cue's cut). Rebuilt only when the grid changes.
    subroutine cue_spline_build(sp, n, x, nout, xout)
        type(cue_spline_t), intent(inout) :: sp
        integer, intent(in)  :: n, nout
        real(WP), intent(in) :: x(n), xout(nout)

        real(WP), allocatable :: a(:), b(:), c(:), cpw(:), rdw(:)
        real(WP) :: h0, h1, ha, hb, den
        integer  :: i, j, k, nr

        if (sp%n == n .and. sp%nout == nout) then
            if (all(sp%xout == xout)) return
        end if
        call spline_free(sp)
        sp%n = n
        sp%nout = nout
        allocate(sp%xout(nout), sp%h(n - 1))
        sp%xout = xout
        do i = 1, n - 1
            sp%h(i) = x(i + 1) - x(i)
        end do
        ! Unknowns M_2 .. M_{n-1} (nr of them); rows i = 2 .. n-1 of
        !   h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1} = 6 (d_i - d_{i-1}),
        ! with not-a-knot: M_1 = ((h1 + h2) M_2 - h1 M_3) / h2 and the mirror at the end
        nr = n - 2
        allocate(a(nr), b(nr), c(nr), cpw(nr), rdw(nr))
        do j = 1, nr
            i = j + 1
            a(j) = sp%h(i - 1)
            b(j) = 2.0_wp * (sp%h(i - 1) + sp%h(i))
            c(j) = sp%h(i)
        end do
        h0 = sp%h(1); h1 = sp%h(2)
        b(1) = b(1) + h0 * (h0 + h1) / h1
        c(1) = c(1) - h0 * h0 / h1
        ha = sp%h(n - 2); hb = sp%h(n - 1)
        b(nr) = b(nr) + hb * (ha + hb) / ha
        a(nr) = a(nr) - hb * hb / ha
        ! Thomas factors
        rdw(1) = 1.0_wp / b(1)
        cpw(1) = c(1) * rdw(1)
        do j = 2, nr
            den = b(j) - a(j) * cpw(j - 1)
            rdw(j) = 1.0_wp / den
            cpw(j) = c(j) * rdw(j)
        end do
        allocate(sp%cp(nr), sp%rden(nr), sp%aa(nr), sp%m0c(2), sp%mnc(2))
        sp%cp = real(cpw, KP)
        sp%rden = real(rdw, KP)
        sp%aa = real(a, KP)
        sp%m0c(1) = real((h0 + h1) / h1, KP)
        sp%m0c(2) = real(-h0 / h1, KP)
        sp%mnc(1) = real((ha + hb) / ha, KP)
        sp%mnc(2) = real(-hb / ha, KP)

        allocate(sp%k(nout), sp%t(nout), sp%hh6(nout), sp%kst(n - 1), sp%ken(n - 1))
        sp%kst = nout + 1
        sp%ken = 0
        k = 1
        do j = 1, nout
            sp%k(j) = 0
            sp%t(j) = 0.0_KP
            sp%hh6(j) = 0.0_KP
            if (xout(j) <= CUE_EDGE_HI) cycle
            do while (k < n - 1)
                if (xout(j) < x(k + 1)) exit
                k = k + 1
            end do
            ! (outside the knots: the end interval's polynomial, as scipy's extrapolation)
            sp%k(j) = k
            sp%t(j) = real((xout(j) - x(k)) / sp%h(k), KP)
            sp%hh6(j) = real(sp%h(k) * sp%h(k) / 6.0_wp, KP)
            sp%kst(k) = min(sp%kst(k), j)
            sp%ken(k) = max(sp%ken(k), j)
        end do
#ifdef OMP_TARGET
        !$omp target enter data map(to: sp%h, sp%cp, sp%rden, sp%aa, sp%m0c, sp%mnc, sp%k, sp%t, sp%hh6, sp%kst, sp%ken)
        sp%on_device = .true.
#endif
    end subroutine cue_spline_build

    subroutine spline_free(sp)
        type(cue_spline_t), intent(inout) :: sp
#ifdef OMP_TARGET
        if (sp%on_device) then
            !$omp target exit data map(delete: sp%h, sp%cp, sp%rden, sp%aa, sp%m0c, sp%mnc, sp%k, sp%t, sp%hh6, sp%kst, sp%ken)
        end if
#endif
        sp%on_device = .false.
        if (allocated(sp%xout)) deallocate(sp%xout, sp%h)
        if (allocated(sp%cp)) deallocate(sp%cp, sp%rden, sp%aa, sp%m0c, sp%mnc)
        if (allocated(sp%k)) deallocate(sp%k, sp%t, sp%hh6, sp%kst, sp%ken)
        sp%n = 0
        sp%nout = 0
    end subroutine spline_free

    !> Splines n spectra v(n_knots, n) onto the output grid, out(nout, n); mm (as v) is scratch for
    !> the second derivatives. cue_spline_solve, then the evaluation.
    subroutine cue_spline_apply(nk, nout, n, h, cp, rden, aa, m0c, mnc, k, t, hh6, v, mm, out)
        integer, intent(in)   :: nk, nout, n
        real(WP), intent(in)  :: h(nk - 1)
        real(KP), intent(in)  :: cp(nk - 2), rden(nk - 2), aa(nk - 2), m0c(2), mnc(2)
        integer, intent(in)   :: k(nout)
        real(KP), intent(in)  :: t(nout), hh6(nout)
        real(KP), intent(in)  :: v(nk, n)
        real(KP), intent(inout) :: mm(nk, n)
        real(KP), intent(out) :: out(nout, n)
        integer  :: m, j, kk
        real(KP) :: tt, u

        call cue_spline_solve(nk, n, h, cp, rden, aa, m0c, mnc, v, mm)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(kk, tt, u)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(kk, tt, u)
#endif
        do m = 1, n
            do j = 1, nout
                kk = k(j)
                if (kk == 0) then
                    out(j, m) = 0.0_KP
                else
                    tt = t(j)
                    u = 1.0_KP - tt
                    out(j, m) = u * v(kk, m) + tt * v(kk + 1, m) + &
                                hh6(j) * ((u * u * u - u) * mm(kk, m) + (tt * tt * tt - tt) * mm(kk + 1, m))
                end if
            end do
        end do
    end subroutine cue_spline_apply

    !> The spline's second derivatives mm(nk, n) of n spectra v(nk, n) (one thread per spectrum: the
    !> Thomas sweeps of the factored not-a-knot system; latency-bound by the recurrence, whatever the
    !> layout, so the knots are the leading dimension, as the evaluation reads them).
    subroutine cue_spline_solve(nk, n, h, cp, rden, aa, m0c, mnc, v, mm)
        integer, intent(in)   :: nk, n
        real(WP), intent(in)  :: h(nk - 1)
        real(KP), intent(in)  :: cp(nk - 2), rden(nk - 2), aa(nk - 2), m0c(2), mnc(2)
        real(KP), intent(in)  :: v(nk, n)
        real(KP), intent(inout) :: mm(nk, n)
        integer  :: m, j, i
        real(KP) :: r

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(j, i, r)
#elif defined(_OPENMP)
        !$omp parallel do private(j, i, r)
#endif
        do m = 1, n
            ! Forward sweep on rows j = 1..nk-2 (knot i = j + 1)
            do j = 1, nk - 2
                i = j + 1
                r = real(6.0_wp * (real(v(i + 1, m) - v(i, m), WP) / h(i) - &
                                   real(v(i, m) - v(i - 1, m), WP) / h(i - 1)), KP)
                if (j == 1) then
                    mm(i, m) = r * rden(1)
                else
                    mm(i, m) = (r - aa(j) * mm(i - 1, m)) * rden(j)
                end if
            end do
            ! Back substitution
            do j = nk - 3, 1, -1
                i = j + 1
                mm(i, m) = mm(i, m) - cp(j) * mm(i + 1, m)
            end do
            mm(1, m) = m0c(1) * mm(2, m) + m0c(2) * mm(3, m)
            mm(nk, m) = mnc(1) * mm(nk - 1, m) + mnc(2) * mm(nk - 2, m)
        end do
    end subroutine cue_spline_solve

    ! ==========================================================================================
    ! Lines
    ! ==========================================================================================

    !> For each of the context's lines (vacuum wavelengths lam(nl)) the cue line it receives
    !> (index in the networks' output order), or 0: the nearest cue line within a relative
    !> 1.5e-3 (cue's and FSPS's line lists quote some lines at slightly different wavelengths),
    !> each cue line to its nearest context line, the closer one winning a shared slot.
    subroutine cue_line_map(w, nl, lam, map)
        type(cue_weights_t), intent(in) :: w
        integer, intent(in)  :: nl
        real(WP), intent(in) :: lam(nl)
        integer, intent(out) :: map(nl)
        real(WP), parameter :: TOL = 1.5e-3_wp
        real(WP) :: best(nl), d
        integer  :: i, kk, ib

        map = 0
        best = huge(1.0_wp)
        do kk = 1, w%nlines
            ib = 0
            d = huge(1.0_wp)
            do i = 1, nl
                if (lam(i) <= 0.0_wp) cycle
                if (abs(lam(i) - w%line_lam(kk)) / lam(i) < d) then
                    d = abs(lam(i) - w%line_lam(kk)) / lam(i)
                    ib = i
                end if
            end do
            if (ib == 0 .or. d > TOL) cycle
            if (d < best(ib)) then
                best(ib) = d
                map(ib) = kk
            end if
        end do
    end subroutine cue_line_map

    ! ==========================================================================================
    ! Reverse pass (gradients)
    ! ==========================================================================================

    !> d act / d a of the hidden layers' activation act(a) = (beta + (1 - beta) sigma(alpha a)) a,
    !> sigma(x) = 1 / (1 + exp(-x)): beta + (1 - beta) sigma (1 + alpha a (1 - sigma)).
    pure real(WP) function act_deriv(a, al, be) result(d)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: a, al, be
        real(WP) :: s
        s = 1.0_wp / (1.0_wp + exp(-al * a))
        d = be + (1.0_wp - be) * s * (1.0_wp + al * a * (1.0_wp - s))
    end function act_deriv

    !> The reverse pass of cue_forward: the derivatives of the caller's loss with respect to the
    !> network inputs theta(12, n), given those with respect to the outputs, line_bar(nlines, n) (of
    !> the log10 line luminosities) and cont_bar(ncw, n) (of log10 of the continuum). The forward is
    !> recomputed in FP64 (the weights, FP32 as trained, converted exactly) with the pre-activations
    !> kept, so no activations are stored between the passes: the reverse pass is exact for the
    !> FP64 network (cue_fp64 = 1) and differs from the derivative of the FP32 hidden layers by their
    !> rounding (~1e-6 relative). Models in passes of CUE_BCHUNK (chunk).
    !> The weights' own derivatives are not formed: cue's weights are not parameters.
    subroutine cue_forward_adjoint(w, n, theta, line_bar, cont_bar, theta_bar, chunk)
        type(cue_weights_t), intent(in) :: w
        integer, intent(in)   :: n
        real(WP), intent(in)  :: theta(CUE_NIN, n), line_bar(w%nlines, n), cont_bar(w%ncw, n)
        real(WP), intent(out) :: theta_bar(CUE_NIN, n)
        integer, intent(in), optional :: chunk

        real(WP), allocatable :: x0(:,:,:), a1(:,:,:), a2(:,:,:), a3(:,:,:), yy(:,:,:), d1(:,:,:), d2(:,:,:), cb(:,:,:)
        integer :: c0, nc, ch, ncb

        ch = CUE_BCHUNK
        if (present(chunk)) ch = max(1, chunk)
        ch = min(n, ch)
        if (ch < 1) return
        ncb = max(CUE_NPCA_LINE, w%npc)
        allocate(x0(CUE_NIN, ch, CUE_NNET), a1(CUE_NHID, ch, CUE_NNET), a2(CUE_NHID, ch, CUE_NNET), &
                 a3(CUE_NHID, ch, CUE_NNET), yy(CUE_NHID, ch, CUE_NNET), d1(CUE_NHID, ch, CUE_NNET), &
                 d2(CUE_NHID, ch, CUE_NNET), cb(ncb, ch, CUE_NNET))
#ifdef OMP_TARGET
        !$omp target data map(alloc: x0, a1, a2, a3, yy, d1, d2, cb)
#endif
        do c0 = 0, n - 1, ch
            nc = min(ch, n - c0)
            ! Forward, keeping a1, a2, a3 (the pre-activations)
            call net_inputs_d(nc, theta(1, c0 + 1), w%in_shift, w%in_scale, x0, ch)
            call adj_layer(CUE_NIN, nc, ch, w%w1, w%bias(1, 1, 1), w%alpha(1, 1, 1), w%beta(1, 1, 1), x0, a1, yy, .true.)
            call adj_layer(CUE_NHID, nc, ch, w%w2, w%bias(1, 1, 2), w%alpha(1, 1, 2), w%beta(1, 1, 2), yy, a2, d1, .true.)
            call adj_layer(CUE_NHID, nc, ch, w%w3, w%bias(1, 1, 3), w%alpha(1, 1, 3), w%beta(1, 1, 3), d1, a3, d2, .false.)
            ! The output layers and PCA bases: the cotangents of the PCA coefficients, then of the
            ! last hidden layer's pre-activation (d1)
            call rev_head_coef(w%nlines, w%nlmax, w%ncw, w%npc, nc, ch, ncb, line_bar(1, c0 + 1), cont_bar(1, c0 + 1), &
                               w%lcmp, w%lout_sc, w%ccmp, w%cout_sc, w%npca, w%ncount, w%loff, cb)
            call rev_head_back(w%npc, nc, ch, ncb, cb, w%lw4, w%lpsc, w%cw4, w%cpsc, w%npca, d1)
            call rev_act_scale(nc, ch, w%alpha(1, 1, 3), w%beta(1, 1, 3), a3, d1)
            ! The hidden layers, backwards
            call adj_back(nc, ch, w%w3, d1, d2)
            call rev_act_scale(nc, ch, w%alpha(1, 1, 2), w%beta(1, 1, 2), a2, d2)
            call adj_back(nc, ch, w%w2, d2, d1)
            call rev_act_scale(nc, ch, w%alpha(1, 1, 1), w%beta(1, 1, 1), a1, d1)
            call rev_inputs(nc, ch, w%w1, w%in_scale, d1, theta_bar(1, c0 + 1))
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine cue_forward_adjoint

    !> One layer of the reverse pass's recompute: the pre-activation a = b + W x and, with with_act,
    !> y = act(a), in FP64 (one tiled kernel with HIP, loops otherwise)
    subroutine adj_layer(nin, nc, ch, w, b, al, be, x, a, y, with_act)
        integer, intent(in)   :: nin, nc, ch
        real(KP), intent(in)  :: w(CUE_NHID, nin, CUE_NNET), b(CUE_NHID, CUE_NNET)
        real(KP), intent(in)  :: al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET)
        real(WP), intent(in)  :: x(nin, ch, CUE_NNET)
        real(WP), intent(out) :: a(CUE_NHID, ch, CUE_NNET), y(CUE_NHID, ch, CUE_NNET)
        logical, intent(in)   :: with_act
#if defined(OPENSPS_HIP) && defined(OMP_TARGET)
        call hip_dense_pre(nin, nc, ch, w, b, al, be, x, a, y, with_act)
#else
        call rev_dense_pre(nin, nc, ch, w, b, x, a)
        if (with_act) call rev_act(nc, ch, al, be, a, y)
#endif
    end subroutine adj_layer

    !> din = W^T dout of a 256 x 256 layer, FP64 (HIP kernel or loops)
    subroutine adj_back(nc, ch, w, dout, din)
        integer, intent(in)   :: nc, ch
        real(KP), intent(in)  :: w(CUE_NHID, CUE_NHID, CUE_NNET)
        real(WP), intent(in)  :: dout(CUE_NHID, ch, CUE_NNET)
        real(WP), intent(out) :: din(CUE_NHID, ch, CUE_NNET)
#if defined(OPENSPS_HIP) && defined(OMP_TARGET)
        call hip_dense_back(nc, ch, w, dout, din)
#else
        call rev_dense_back(CUE_NHID, nc, ch, w, dout, din)
#endif
    end subroutine adj_back

    !> a(u, m, g) = b(u, g) + sum_k w(u, k, g) x(k, m, g), in FP64.
    subroutine rev_dense_pre(nin, nc, ch, w, b, x, a)
        integer, intent(in)   :: nin, nc, ch
        real(KP), intent(in)  :: w(CUE_NHID, nin, CUE_NNET), b(CUE_NHID, CUE_NNET)
        real(WP), intent(in)  :: x(nin, ch, CUE_NNET)
        real(WP), intent(out) :: a(CUE_NHID, ch, CUE_NNET)
        integer  :: g, m, u, k
        real(WP) :: s

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(s, k)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(s, k)
#endif
        do g = 1, CUE_NNET
            do m = 1, nc
                do u = 1, CUE_NHID
                    s = real(b(u, g), WP)
                    do k = 1, nin
                        s = s + real(w(u, k, g), WP) * x(k, m, g)
                    end do
                    a(u, m, g) = s
                end do
            end do
        end do
    end subroutine rev_dense_pre

    !> y = act(a) in FP64.
    subroutine rev_act(nc, ch, al, be, a, y)
        integer, intent(in)   :: nc, ch
        real(KP), intent(in)  :: al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET)
        real(WP), intent(in)  :: a(CUE_NHID, ch, CUE_NNET)
        real(WP), intent(out) :: y(CUE_NHID, ch, CUE_NNET)
        integer  :: g, m, u
        real(WP) :: bb

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(bb)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(bb)
#endif
        do g = 1, CUE_NNET
            do m = 1, nc
                do u = 1, CUE_NHID
                    bb = real(be(u, g), WP)
                    y(u, m, g) = (bb + (1.0_wp - bb) / (1.0_wp + exp(-real(al(u, g), WP) * a(u, m, g)))) * a(u, m, g)
                end do
            end do
        end do
    end subroutine rev_act

    !> d(u, m, g) *= act'(a(u, m, g)).
    subroutine rev_act_scale(nc, ch, al, be, a, d)
        integer, intent(in)     :: nc, ch
        real(KP), intent(in)    :: al(CUE_NHID, CUE_NNET), be(CUE_NHID, CUE_NNET)
        real(WP), intent(in)    :: a(CUE_NHID, ch, CUE_NNET)
        real(WP), intent(inout) :: d(CUE_NHID, ch, CUE_NNET)
        integer :: g, m, u

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do g = 1, CUE_NNET
            do m = 1, nc
                do u = 1, CUE_NHID
                    d(u, m, g) = d(u, m, g) * act_deriv(a(u, m, g), real(al(u, g), WP), real(be(u, g), WP))
                end do
            end do
        end do
    end subroutine rev_act_scale

    !> din(k, m, g) = sum_u w(u, k, g) dout(u, m, g): the transpose of a layer.
    subroutine rev_dense_back(nin, nc, ch, w, dout, din)
        integer, intent(in)   :: nin, nc, ch
        real(KP), intent(in)  :: w(CUE_NHID, nin, CUE_NNET)
        real(WP), intent(in)  :: dout(CUE_NHID, ch, CUE_NNET)
        real(WP), intent(out) :: din(nin, ch, CUE_NNET)
        integer  :: g, m, k, u
        real(WP) :: s

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(s, u)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(s, u)
#endif
        do g = 1, CUE_NNET
            do m = 1, nc
                do k = 1, nin
                    s = 0.0_wp
                    do u = 1, CUE_NHID
                        s = s + real(w(u, k, g), WP) * dout(u, m, g)
                    end do
                    din(k, m, g) = s
                end do
            end do
        end do
    end subroutine rev_dense_back

    !> The cotangents of the PCA coefficients of every network: the lines' (cb(p, m, g), g = 1 ..
    !> CUE_NGROUP: line_log(s) = (lmean + sum_p c_p lcmp(s, p)) osc + osh) and the continuum's
    !> (cb(p, m, CUE_NNET)).
    subroutine rev_head_coef(nlines, nlmax, ncw, npc, nc, ch, ncb, line_bar, cont_bar, lcmp, osc, ccmp, cosc, npca, &
                             ncount, loff, cb)
        integer, intent(in)   :: nlines, nlmax, ncw, npc, nc, ch, ncb
        real(WP), intent(in)  :: line_bar(nlines, nc), cont_bar(ncw, nc)
        real(WP), intent(in)  :: lcmp(nlmax, CUE_NPCA_LINE, CUE_NGROUP), osc(nlmax, CUE_NGROUP)
        real(WP), intent(in)  :: ccmp(ncw, npc), cosc(ncw)
        integer, intent(in)   :: npca(CUE_NGROUP), ncount(CUE_NGROUP), loff(CUE_NGROUP)
        real(WP), intent(out) :: cb(ncb, ch, CUE_NNET)
        integer  :: g, m, p, s, i
        real(WP) :: a

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(a, s)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(a, s)
#endif
        do g = 1, CUE_NGROUP
            do m = 1, nc
                do p = 1, ncb
                    a = 0.0_wp
                    if (p <= npca(g)) then
                        do s = 1, ncount(g)
                            a = a + line_bar(loff(g) + s, m) * osc(s, g) * lcmp(s, p, g)
                        end do
                    end if
                    cb(p, m, g) = a
                end do
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(a, i)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(a, i)
#endif
        do m = 1, nc
            do p = 1, ncb
                a = 0.0_wp
                if (p <= npc) then
                    do i = 1, ncw
                        a = a + cont_bar(i, m) * cosc(i) * ccmp(i, p)
                    end do
                end if
                cb(p, m, CUE_NNET) = a
            end do
        end do
    end subroutine rev_head_coef

    !> The cotangent of each network's last hidden layer h (units u) from those of its PCA
    !> coefficients: d(u, m, g) = sum_p cb(p, m, g) psc(p) w4(p, u).
    subroutine rev_head_back(npc, nc, ch, ncb, cb, lw4, lpsc, cw4, cpsc, npca, d)
        integer, intent(in)   :: npc, nc, ch, ncb
        real(WP), intent(in)  :: cb(ncb, ch, CUE_NNET)
        real(KP), intent(in)  :: lw4(CUE_NPCA_LINE, CUE_NHID, CUE_NGROUP), cw4(npc, CUE_NHID)
        real(WP), intent(in)  :: lpsc(CUE_NPCA_LINE, CUE_NGROUP), cpsc(npc)
        integer, intent(in)   :: npca(CUE_NGROUP)
        real(WP), intent(out) :: d(CUE_NHID, ch, CUE_NNET)
        integer  :: g, m, u, p
        real(WP) :: s

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(s, p)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(s, p)
#endif
        do g = 1, CUE_NNET
            do m = 1, nc
                do u = 1, CUE_NHID
                    s = 0.0_wp
                    if (g <= CUE_NGROUP) then
                        do p = 1, npca(g)
                            s = s + cb(p, m, g) * lpsc(p, g) * real(lw4(p, u, g), WP)
                        end do
                    else
                        do p = 1, npc
                            s = s + cb(p, m, g) * cpsc(p) * real(cw4(p, u), WP)
                        end do
                    end if
                    d(u, m, g) = s
                end do
            end do
        end do
    end subroutine rev_head_back

    !> theta_bar(k, m) = sum_g sum_u w1(u, k, g) d(u, m, g) / scale(k, g): through the first layer
    !> and the inputs' normalization (theta - shift) / scale.
    subroutine rev_inputs(nc, ch, w1, sc, d, theta_bar)
        integer, intent(in)   :: nc, ch
        real(KP), intent(in)  :: w1(CUE_NHID, CUE_NIN, CUE_NNET), sc(CUE_NIN, CUE_NNET)
        real(WP), intent(in)  :: d(CUE_NHID, ch, CUE_NNET)
        real(WP), intent(out) :: theta_bar(CUE_NIN, nc)
        integer  :: g, m, k, u
        real(WP) :: s, sg

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(s, sg, g, u)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(s, sg, g, u)
#endif
        do m = 1, nc
            do k = 1, CUE_NIN
                s = 0.0_wp
                do g = 1, CUE_NNET
                    sg = 0.0_wp
                    do u = 1, CUE_NHID
                        sg = sg + real(w1(u, k, g), WP) * d(u, m, g)
                    end do
                    s = s + sg / real(sc(k, g), WP)
                end do
                theta_bar(k, m) = s
            end do
        end do
    end subroutine rev_inputs

    ! ------------------------------------------------------------------------------------------
    ! The ionizing spectrum's fit
    ! ------------------------------------------------------------------------------------------

    !> The reverse pass of cue_ionizing_params: spec_bar(nw, n), the derivatives of the caller's
    !> loss with respect to the spectra, from ion_bar(7, n), those with respect to the shape
    !> parameters. One thread per spectrum (cue_fit_ionizing_adjoint).
    subroutine cue_ionizing_params_adjoint(n, nw, lam, spec, ion_bar, spec_bar)
        integer, intent(in)   :: n, nw
        real(WP), intent(in)  :: lam(nw), ion_bar(CUE_NION, n)
        real(KP), intent(in)  :: spec(nw, n)
        real(WP), intent(out) :: spec_bar(nw, n)
        real(WP) :: llam(nw)
        integer :: m

        llam = log10(lam)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do map(to: llam)
#elif defined(_OPENMP)
        !$omp parallel do
#endif
        do m = 1, n
            call cue_fit_ionizing_adjoint(nw, lam, spec(1, m), ion_bar(1, m), spec_bar(1, m), llam)
        end do
    end subroutine cue_ionizing_params_adjoint

    !> The reverse pass of cue_fit_ionizing for one spectrum. cue's shape parameters are the slopes
    !> alpha_s of the four segments' power laws A_s lambda^alpha_s, fitted by minimizing cue's loss
    !> l(alpha_s, log A_s; spec), and the log ratios of the segments' integrated luminosities,
    !> log L_s = log A_s + log10 |(b^(alpha_s - 1) - a^(alpha_s - 1)) / (alpha_s - 1)|, over the
    !> segment [a, b] (times the constant c L_sun, which cancels in the ratios), all clipped to cue's
    !> training ranges (where a parameter is clipped, its derivative is zero, as the forward's).
    !> The fit is a minimum, g(x*; spec) = dl / dx = 0, so by the implicit function theorem
    !> dx* / d spec_i = -H^-1 dg / d spec_i with the Hessian H of the loss at the minimum
    !> (fit_segment's Newton Hessian): the adjoint solves H lambda = -x_bar once per segment and
    !> contracts it with dg / d spec_i for every pixel.
    !> The normalization by the first pixel beyond 911.6 A has no derivative: it rescales every
    !> normalized pixel, the floored ones too (the floor is 1e-70 of the normalized spectrum), which
    !> shifts log A of the normalized fit and leaves alpha and the unnormalized log A unchanged. The
    !> floor of a pixel without flux has no derivative either, a segment at a bound of the
    !> Newton iteration (alpha_s in [-40, 100], log A_s in [-200, 100]) or without a positive
    !> definite Hessian (zero), and the segments the forward sets by convention (empty ones, the
    !> one-pixel segment's mean).
    !> Because the fit is invariant under a scaling of the spectrum, sum_i spec_i spec_bar_i = 0,
    !> unless a pixel has no flux: the floor does not scale with it.
    subroutine cue_fit_ionizing_adjoint(nw, lam, spec, ion_bar, spec_bar, llam)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)   :: nw
        real(WP), intent(in)  :: lam(nw), llam(nw)
        real(KP), intent(in)  :: spec(nw)
        real(WP), intent(in)  :: ion_bar(CUE_NION)
        real(WP), intent(out) :: spec_bar(nw)

        real(WP) :: edge(5), alpha(4), loga(4), logl(4), abar(4), bbar(4), lbar(4), fnorm, floor_, ref, lo, hi, rb
        real(WP) :: p, dp, d2p
        logical  :: empty(4)
        integer  :: s, i0(4), i1(4), i, ilim, k

        spec_bar(1:nw) = 0.0_wp
        edge(1) = 1.0_wp
        edge(2) = CUE_EDGE_HEII
        edge(3) = CUE_EDGE_OII
        edge(4) = CUE_EDGE_HEI
        edge(5) = CUE_EDGE_HI
        ilim = 0
        do i = 1, nw
            if (lam(i) <= CUE_EDGE_HI) ilim = i
        end do
        ref = 0.0_wp
        if (ilim < nw) ref = real(spec(ilim + 1), WP)
        if (ref <= 0.0_wp) then
            do i = 1, nw
                ref = max(ref, real(spec(i), WP))
            end do
        end if
        if (ref <= 0.0_wp) return   ! (the forward returns the bounds)
        fnorm = 1.0e-18_wp / ref
        floor_ = 1.0e-70_wp * fnorm

        ! The segments' fits again (the forward's: the same minimum)
        k = 0
        do s = 1, 4
            i0(s) = k + 1
            k = 0
            do i = 1, nw
                if (lam(i) <= edge(s + 1)) k = i
            end do
            i1(s) = k
            empty(s) = .true.
            do i = i0(s), i1(s)
                if (spec(i) > 0.0_KP) empty(s) = .false.
            end do
            alpha(s) = 0.0_wp
            loga(s) = 0.0_wp
            if (.not. empty(s)) then
                call fit_segment(nw, lam, spec, i0(s), i1(s), fnorm, floor_, alpha(s), loga(s), llam)
                loga(s) = loga(s) - log10(fnorm)
            end if
        end do
        do s = 1, 4
            if (empty(s)) cycle
            logl(s) = loga(s) + cue_log_ratio_power(alpha(s) - 1.0_wp, edge(s), edge(s + 1))
        end do

        ! Through the clipping and the ratios: the cotangents of alpha_s, log A_s (normalized units:
        ! the same derivative as of the unnormalized, which differs by a constant) and log L_s
        abar = 0.0_wp
        bbar = 0.0_wp
        lbar = 0.0_wp
        do s = 1, 4
            if (empty(s)) cycle
            if (alpha(s) > cue_ion_bound(s, .false.) .and. alpha(s) < cue_ion_bound(s, .true.)) abar(s) = ion_bar(s)
        end do
        do s = 1, 3
            if (empty(s) .or. empty(s + 1)) cycle
            rb = logl(s + 1) - logl(s)
            lo = cue_ion_bound(4 + s, .false.)
            hi = cue_ion_bound(4 + s, .true.)
            if (rb > lo .and. rb < hi) then
                lbar(s + 1) = lbar(s + 1) + ion_bar(4 + s)
                lbar(s) = lbar(s) - ion_bar(4 + s)
            end if
        end do
        do s = 1, 4
            if (empty(s)) cycle
            ! d log L_s = d log A_s + (d P / d beta) d alpha_s, beta = alpha_s - 1
            call seg_p(alpha(s) - 1.0_wp, edge(s), log(edge(s + 1) / edge(s)), p, dp, d2p)
            bbar(s) = lbar(s)
            abar(s) = abar(s) + lbar(s) * dp
        end do
        do s = 1, 4
            if (empty(s)) cycle
            if (abar(s) == 0.0_wp .and. bbar(s) == 0.0_wp) cycle
            call fit_segment_adjoint(nw, lam, spec, i0(s), i1(s), fnorm, floor_, alpha(s), loga(s) + log10(fnorm), &
                                     abar(s), bbar(s), llam, spec_bar)
        end do
    end subroutine cue_fit_ionizing_adjoint

    !> One segment's contribution to cue_fit_ionizing_adjoint: spec_bar(i0:i1) += abar d alpha / d
    !> spec_i + bbar d log A / d spec_i at the minimum (alpha, loga: log A in the normalized units).
    subroutine fit_segment_adjoint(nw, lam, spec, i0, i1, fnorm, floor_, alpha, loga, abar, bbar, llam, spec_bar)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)     :: nw, i0, i1
        real(WP), intent(in)    :: lam(nw), llam(nw), fnorm, floor_, alpha, loga, abar, bbar
        real(KP), intent(in)    :: spec(nw)
        real(WP), intent(inout) :: spec_bar(nw)

        real(WP), parameter :: LN10 = log(10.0_wp)
        real(WP) :: x0, x1, lr, sm(6), q, qraw, logq, p, dp, d2p, t, xa, xb, fa, fb
        real(WP) :: haa, hab, hbb, det, la, lb, dq, dt, ga, gb, yp, sgn
        integer  :: i

        x0 = lam(i0)
        x1 = lam(i1)
        sm(1) = real(i1 - i0 + 1, WP)
        sm(2:6) = 0.0_wp
        do i = i0, i1
            sm(2) = sm(2) + llam(i)
            sm(3) = sm(3) + log10(max(real(spec(i), WP) * fnorm, floor_))
        end do
        sm(2) = sm(2) / sm(1)
        sm(3) = sm(3) / sm(1)
        do i = i0, i1
            xa = llam(i) - sm(2)
            fa = log10(max(real(spec(i), WP) * fnorm, floor_)) - sm(3)
            sm(4) = sm(4) + xa * xa
            sm(5) = sm(5) + xa * fa
            sm(6) = sm(6) + fa * fa
        end do
        qraw = 0.0_wp
        do i = i0, i1 - 1
            xa = lam(i); xb = lam(i + 1)
            fa = max(real(spec(i), WP) * fnorm, floor_) * xa / (CUE_H * CUE_C)
            fb = max(real(spec(i + 1), WP) * fnorm, floor_) * xb / (CUE_H * CUE_C)
            qraw = qraw + 0.5_wp * (fa + fb) * (CUE_C / xa - CUE_C / xb)
        end do
        q = abs(qraw)

        if (i1 == i0 .or. q <= 0.0_wp) then
            ! The forward's convention: alpha = 0, log A = the mean of the normalized log spectrum
            do i = i0, i1
                if (real(spec(i), WP) * fnorm > floor_) &
                    spec_bar(i) = spec_bar(i) + bbar / (sm(1) * LN10 * real(spec(i), WP))
            end do
            return
        end if
        ! (a segment at a bound of the iteration: zero)
        if (alpha <= -40.0_wp .or. alpha >= 100.0_wp .or. loga <= -200.0_wp .or. loga >= 100.0_wp) return

        logq = log10(q)
        lr = log(x1 / x0)
        call seg_p(alpha, x0, lr, p, dp, d2p)
        t = loga - log10(CUE_H) + p - logq
        haa = sm(4) + sm(1) * sm(2) * sm(2) + sm(1) * (dp * dp + t * d2p)
        hab = sm(1) * sm(2) + sm(1) * dp
        hbb = 2.0_wp * sm(1)
        det = haa * hbb - hab * hab
        if (.not. (haa > 0.0_wp .and. det > 0.0_wp)) return
        ! lambda = -H^-1 (abar, bbar), in the order (alpha, log A)
        la = -(hbb * abar - hab * bbar) / det
        lb = -(haa * bbar - hab * abar) / det
        sgn = sign(1.0_wp, qraw)
        do i = i0, i1
            if (real(spec(i), WP) * fnorm <= floor_) cycle
            yp = 1.0_wp / (LN10 * real(spec(i), WP))   ! d log10(spec_i fnorm) / d spec_i
            ! d q / d spec_i: the pixel enters the trapezoids on both its sides
            dq = 0.0_wp
            if (i < i1) dq = dq + 0.5_wp * lam(i) / (CUE_H * CUE_C) * (CUE_C / lam(i) - CUE_C / lam(i + 1))
            if (i > i0) dq = dq + 0.5_wp * lam(i) / (CUE_H * CUE_C) * (CUE_C / lam(i - 1) - CUE_C / lam(i))
            dq = sgn * dq * fnorm
            dt = -dq / (LN10 * q)       ! d t / d spec_i, t = log A - log h + P - log q
            ! dg / d spec_i: g_A = sum r + N t, g_alpha = sum r X + N t P', r = A + alpha X - Y
            gb = -yp + sm(1) * dt
            ga = -llam(i) * yp + sm(1) * dp * dt
            spec_bar(i) = spec_bar(i) + la * ga + lb * gb
        end do
    end subroutine fit_segment_adjoint

    ! ------------------------------------------------------------------------------------------
    ! The spline
    ! ------------------------------------------------------------------------------------------

    !> The transpose of the spline of cue_spline_apply, for the young and old cotangents of the
    !> continuum at once (columns gy, go of gw, KP: the cotangents of the pixels the spline gives):
    !> uy(nk, n) = S^T gy, uo = S^T go, with S the (linear) map from the knot values v to the
    !> output pixels: the evaluation on the knot interval (gathered per knot over the pixels of its
    !> two intervals, kst, ken: one thread per model, column and knot), its second derivatives mm from
    !> the factored not-a-knot system (the transposed solve: U^T z = r, then L^T x = z, with the
    !> forward's own factors, per model and column), and the two end rows folded in.
    subroutine cue_spline_adjoint(nk, nout, n, h, cp, rden, aa, m0c, mnc, kst, ken, t, hh6, gy, go, uy, uo, mb)
        integer, intent(in)   :: nk, nout, n
        real(WP), intent(in)  :: h(nk - 1)
        real(KP), intent(in)  :: cp(nk - 2), rden(nk - 2), aa(nk - 2), m0c(2), mnc(2)
        integer, intent(in)   :: kst(nk - 1), ken(nk - 1)
        real(KP), intent(in)  :: t(nout), hh6(nout), gy(nout, n), go(nout, n)
        real(WP), intent(out) :: uy(nk, n), uo(nk, n)
        real(WP), intent(inout) :: mb(nk, n, 2)   !< scratch
        integer  :: m, c, i, j
        real(WP) :: sv, sm, tt, u, cj

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(3) private(sv, sm, tt, u, cj, j)
#elif defined(_OPENMP)
        !$omp parallel do collapse(3) private(sv, sm, tt, u, cj, j)
#endif
        do c = 1, 2
            do m = 1, n
                do i = 1, nk
                    sv = 0.0_wp
                    sm = 0.0_wp
                    ! (pixels in the interval i start at knot i: weights u = 1 - t)
                    if (i < nk) then
                        do j = kst(i), ken(i)
                            tt = real(t(j), WP)
                            u = 1.0_wp - tt
                            if (c == 1) then
                                cj = real(gy(j, m), WP)
                            else
                                cj = real(go(j, m), WP)
                            end if
                            sv = sv + u * cj
                            sm = sm + real(hh6(j), WP) * (u * u * u - u) * cj
                        end do
                    end if
                    ! (pixels in the interval i - 1 end at knot i: weights t)
                    if (i > 1) then
                        do j = kst(i - 1), ken(i - 1)
                            tt = real(t(j), WP)
                            if (c == 1) then
                                cj = real(gy(j, m), WP)
                            else
                                cj = real(go(j, m), WP)
                            end if
                            sv = sv + tt * cj
                            sm = sm + real(hh6(j), WP) * (tt * tt * tt - tt) * cj
                        end do
                    end if
                    if (c == 1) then
                        uy(i, m) = sv
                    else
                        uo(i, m) = sv
                    end if
                    mb(i, m, c) = sm
                end do
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do c = 1, 2
            do m = 1, n
                if (c == 1) then
                    call spline_adjoint_solve(nk, h, cp, rden, aa, m0c, mnc, uy(1, m), mb(1, m, c))
                else
                    call spline_adjoint_solve(nk, h, cp, rden, aa, m0c, mnc, uo(1, m), mb(1, m, c))
                end if
            end do
        end do
    end subroutine cue_spline_adjoint

    !> The second half of the spline's transpose for one spectrum: from the cotangent mb of the
    !> second derivatives (nk values; overwritten) to their contribution to the knot values' vb.
    subroutine spline_adjoint_solve(nk, h, cp, rden, aa, m0c, mnc, vb, mb)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)     :: nk
        real(WP), intent(in)    :: h(nk - 1)
        real(KP), intent(in)    :: cp(nk - 2), rden(nk - 2), aa(nk - 2), m0c(2), mnc(2)
        real(WP), intent(inout) :: vb(nk), mb(nk)
        integer  :: j, i
        real(WP) :: x

        ! The end rows (M_1, M_n from their neighbours)
        mb(2) = mb(2) + real(m0c(1), WP) * mb(1)
        mb(3) = mb(3) + real(m0c(2), WP) * mb(1)
        mb(nk - 1) = mb(nk - 1) + real(mnc(1), WP) * mb(nk)
        mb(nk - 2) = mb(nk - 2) + real(mnc(2), WP) * mb(nk)
        ! A^T x = mb(2:nk-1), A = L U: U^T z = r forward (z over mb), then L^T x = z backward
        do j = 2, nk - 2
            mb(j + 1) = mb(j + 1) - real(cp(j - 1), WP) * mb(j)
        end do
        mb(nk - 1) = mb(nk - 1) * real(rden(nk - 2), WP)
        do j = nk - 3, 1, -1
            mb(j + 1) = (mb(j + 1) - real(aa(j + 1), WP) * mb(j + 2)) * real(rden(j), WP)
        end do
        ! r_i = 6 ((v_{i+1} - v_i) / h_i - (v_i - v_{i-1}) / h_{i-1}), i = j + 1
        do j = 1, nk - 2
            i = j + 1
            x = mb(i)
            vb(i + 1) = vb(i + 1) + 6.0_wp * x / h(i)
            vb(i) = vb(i) - 6.0_wp * x * (1.0_wp / h(i) + 1.0_wp / h(i - 1))
            vb(i - 1) = vb(i - 1) + 6.0_wp * x / h(i - 1)
        end do
    end subroutine spline_adjoint_solve

end module opensps_cue
