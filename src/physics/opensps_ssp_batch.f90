module opensps_ssp_batch
    !> @brief The SSP stage of compute_csp_batch with the IMF free per model: the age sums of
    !> each model from the template operator (opensps_ssp_operator) instead of a resident SSP
    !> basis.
    !>
    !> @details
    !> With a basis, the young and old age sums of model m are basis . w_sfh(m) (basis = SSP
    !> spectra of all ages for the shared IMF: templates . B, B = S . w_IMF per age). With the IMF
    !> free per model the SFH is folded in first, so no SSP grid is built per model:
    !>
    !>   x(p, m) = w_IMF(p, m) w_sfh(age(p), m)               (each isochrone point p, FP64)
    !>   V(e, m) = sum_p S(e, p) x(p, m)                      (each effective template e, FP64)
    !>   young / old age sums = etpl . V_young / V_old         (kernel precision)
    !>
    !> split at the model's dust_tesc as the basis path splits the ages. Per age and model, the
    !> surviving mass (with the remnants), the linear L_bol and, for the nebular ages, the
    !> ionizing photon rate come from the same IMF weights through the operator's per-point
    !> coefficients; the nebular ages' EUV pixels are summed again with the escaping fraction
    !> (as add_nebular_batch does with a basis). Everything after the age sums is
    !> compute_csp_batch's.
    !>
    !> The models are processed in chunks (the IMF weights of a chunk, n_pts x chunk FP64, are
    !> the largest array). Every kernel runs on the device for an operator there (OpenMP target)
    !> and as OpenMP host loops otherwise, with the same arithmetic in the same order.

    use, intrinsic :: iso_c_binding, only: c_float
    use opensps_precision, only: WP, KP
    use opensps_tuning, only: TUNE_P_SSP_BATCH_CHUNK, TUNE_TEMPLATE_MT, TUNE_TEMPLATE_PT, TUNE_TEMPLATE_PIX_BLOCK, &
                              TUNE_TEMPLATE_KB, TUNE_GEMM_KB, TUNE_GEMM_NC_BATCHED, TUNE_IMF_GRAD_CHG, TUNE_HIP_GEMM_RM, &
                              TUNE_HIP_GEMM_RN, &
                              TUNE_HIP_GEMM_KC, TUNE_HIP_FOLD_JB, TUNE_HIP_WEIGHTS_PIECES, tuning_get
    use opensps_constants, only: SAFE_FLOOR
    use opensps_imf, only: imf_integral_p, imf_bin_weights_tab, imf_bin_logs, IMF_P_LOWER_LIMIT, IMF_P_UPPER_LIMIT, &
                           IMF_P_ALPHA, IMF_P_VDMC, IMF_P_MDAVE, IMF_P_LOWER_BOUND, IMF_NPAR, imf_integral_dp, &
                           integrate_powerlaw, powerlaw_dj_de
    use opensps_stellar_modifications, only: add_remnant_mass_p
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_to_device, ssp_operator_from_device, ssp_operator_build, &
                                    MOD_NGRP, MOD_GRP_TPAGB
    use opensps_types, only: params
    use opensps_context_types, only: opensps_context_t
    use opensps_trace, only: opensps_trace_push, opensps_trace_pop
    implicit none
    private

    public :: ssp_batch_age_sums_z, ssp_batch_age_sums_adjoint, NIMFG, dot_lanes
    public :: ssp_opset_t, ssp_opset_build, ssp_opset_free, ssp_opset_build_mods, ssp_batch_age_sums_mods, &
               ssp_batch_age_sums_adjoint_mods, ssp_mods_prepare_pairs, MAX_TP_PAIRS, MAX_TP_PAIRS_ALLZ, TP_FD_STEP

    !> The IMF parameters of the gradients (imf1, imf2, imf3, vdmc, mdave)
    integer, parameter :: NIMFG = 5
    !> The distinct nonzero (delt, dell) pairs of a context's batches the free-modification set keeps operators of
    !> The most distinct nonzero (delt, dell) pairs a free-modification set holds sets for (and a batch may use). Each pair
    !> keeps one set of the TP-AGB group (the operators of the set's metallicities: ~90 MB for one at the regression data's
    !> size, see MAX_TP_PAIRS_ALLZ for all) and the gradients build one more at a time for their central differences, so the
    !> cap bounds the memory at that of MAX_TP_PAIRS + 1 sets. (A cap of 32, with the four shifted sets of every pair kept,
    !> took 10 GB in the finite-difference tests, whose every step is a new pair.)
    integer, parameter :: MAX_TP_PAIRS = 8
    !> The same for a set whose operators cover every metallicity (ssp_opset_build_mods with all_z: the logzsol of each
    !> model free): a pair's set holds the operators of all of them, ~1 GB with MIST's 12 at the regression data's size
    integer, parameter :: MAX_TP_PAIRS_ALLZ = 4
    !> The step of the central differences of the TP-AGB group's outputs in delt and dell (dex). Not small: the spectra of the
    !> TP-AGB stars switch templates at thresholds, so the age sums are step functions of delt (measured, MIST, one model: steps
    !> of 1e-7 to 1e-4 in L, 50 to 150 of them per 2e-3 dex, i.e. every ~2e-5 dex), and a difference over a step below that
    !> returns one step (5.07, -7.56, -1.62 at 1e-5 where the slope averaged over the steps is 2.1, -2.9, -1.4). From 2e-4 dex
    !> up the central difference is the slope averaged over some tens of steps (2.12, 2.10, 2.15 at 2e-4, 5e-4, 1e-3), the
    !> response of the loss to delt on the scale a fit explores. With FP32 kernels the step is 1e-3, as the outputs' rounding
    !> (1e-7) is divided by it.
    real(WP), parameter :: TP_FD_STEP = merge(5.0e-4_wp, 1.0e-3_wp, KP == WP)

    !> The kernels run on the device in OpenMP target builds and as OpenMP host loops otherwise
#ifdef OMP_TARGET
    logical, parameter :: ON_DEV = .true.
#else
    logical, parameter :: ON_DEV = .false.
#endif

#if defined(OPENSPS_ACCEL_BLAS)
#ifdef KERNEL_SINGLE_PRECISION
    !> Column counts up to which the FP32 accelerated BLAS product uses the batched form (gemm_columns_batched),
    !> and its work array (pixels x NC_BATCHED x blocks, on the device once allocated)
    integer, parameter :: NC_BATCHED = TUNE_GEMM_NC_BATCHED
    real(KP), allocatable, save :: gtmp(:,:,:)
#endif
#endif

    !> The operators of all metallicities of a context (ssp_operator_build with pset%zmet = 1 ..
    !> nz), their point, piece, segment and template tables concatenated (offsets per
    !> metallicity) so that one kernel launch serves models of every metallicity. Built once
    !> (ssp_opset_build); the device copies are made on first use.
    type :: ssp_opset_t
        integer :: nz = 0, nt = 0, n_pts_max = 0, n_eff_max = 0, n_seg_max = 0, n_pc_max = 0
        type(ssp_operator_t), allocatable :: ops(:)
        !> Per metallicity: pieces, segments, templates; and the (0-based) offsets of its entries in the
        !> concatenated arrays: points, piece table (n_pc + 1 per metallicity), segment table
        !> (n_seg + 1), template entries, template table (n_eff + 1)
        integer, allocatable :: n_pc(:), n_seg(:), n_eff(:), pt_off(:), pc_off(:), seg_off(:), ent_off(:), eff_off(:)
        !> Per point (global index): first-of-age flag, age; piece table (global point indices), pieces
        !> of each age (local piece indices, nt + 1 per metallicity), points of each age (local, nt + 1
        !> per metallicity), segment table (global entry indices), segments of each template (local),
        !> template entries (local point indices)
        integer, allocatable :: pt_first(:), pt_age(:), pc_ptr(:), age_pc(:), pt_ptr(:), seg_ptr(:), seg_first(:), eff_pt(:)
        real(WP), allocatable :: pt_mini(:), pt_m1(:), pt_m2(:), pt_cm(:), pt_cl(:), pt_q(:), age_mmax(:), eff_val(:)
        !> Per point: ln m1 and ln(m2 / m1) (model-independent; imf_bin_weights_tab takes the power-law
        !> bins from them, one exponential per bin instead of a power)
        real(WP), allocatable :: pt_llo(:), pt_dl(:)
        !> FP32 table of the HIP weights kernel (forward only), 8 per point: ln m1 and ln(m2 / m1) each as a
        !> hi + lo pair of FP32 numbers (the argument of an exponential is then formed to ~1e-14, not to the
        !> FP32 epsilon times its size), the coefficients cm, cl and q * 2^-80 (photon rates exceed the
        !> FP32 range), one pad
        real(c_float), allocatable :: pt_f(:,:)
        !> The template entries' values eff_val as FP32 hi + lo pairs (the HIP fold's exact products), 2 per entry
        real(c_float), allocatable :: eff_f(:,:)
        !> The template entries by point (the reverse pass's gathers): point gp (global index) has the
        !> entries pe_k(pe_ptr(gp) .. pe_ptr(gp + 1) - 1) (global entry indices), of templates pe_e (local)
        integer, allocatable :: pe_ptr(:), pe_k(:), pe_e(:)
        logical :: on_device = .false.
        !> Free isochrone modifications (ssp_opset_build_mods): the set is a container of the operators of the modification
        !> groups, one set (the metallicities' operators) per group, and a batch is the sum over the groups of the
        !> coefficient of each model times the group's age sums (ssp_batch_age_sums_mods)
        logical :: free_mods = .false.
        type(ssp_opset_t), pointer :: grp(:) => null()
        !> The TP-AGB shifts delt and dell (the next groups): the sets of the TP-AGB group at the distinct nonzero pairs the
        !> batches so far used (ext(1:n_ext), pairs ext_t, ext_l; built when a batch needs one, ssp_mods_prepare_pairs); each is
        !> one more group of the batch with the coefficient of the TP-AGB group for the models at that pair. (The gradients'
        !> shifted sets, +-TP_FD_STEP in delt and in dell, are built one pair at a time in the reverse pass and freed after it:
        !> each is the size of a set of the TP-AGB group at every metallicity)
        !> n_ext: the highest slot of ext in use (a slot whose ext_ok is false is free: a pair a batch did not use is evicted
        !> when a new pair needs the room); max_ext: the pairs a batch may use
        integer :: n_ext = 0, max_ext = MAX_TP_PAIRS
        logical :: ext_ok(MAX_TP_PAIRS) = .false.
        type(ssp_opset_t), pointer :: ext(:) => null()
        real(WP) :: ext_t(MAX_TP_PAIRS) = 0.0_wp, ext_l(MAX_TP_PAIRS) = 0.0_wp
        !> What the groups were built from (the pairs' sets are built with the same)
        type(params) :: base_pset
        logical :: all_z = .false.
        !> The linear L_bol sums sl of each group at each age, model and metallicity pass of the last batch (the reverse pass)
        real(WP), allocatable :: slp(:,:,:,:)
    end type ssp_opset_t

contains

    !> The set of operators ops (one per metallicity, in the order of the context's metallicities;
    !> moved into set).
    subroutine ssp_opset_build(ops, set)
        type(ssp_operator_t), allocatable, intent(inout) :: ops(:)
        type(ssp_opset_t), intent(inout) :: set
        integer :: nz, iz, nt, t, i, np, npc, nsg, nen, nef, k

        call ssp_opset_free(set)
        nz = size(ops)
        nt = ops(1)%n_ages
        set%nz = nz
        set%nt = nt
        allocate(set%n_pc(nz), set%n_seg(nz), set%n_eff(nz), set%pt_off(nz), set%pc_off(nz), set%seg_off(nz), &
                 set%ent_off(nz), set%eff_off(nz))
        np = 0; npc = 0; nsg = 0; nen = 0; nef = 0
        do iz = 1, nz
            set%n_pc(iz) = ops(iz)%n_pc
            set%n_seg(iz) = ops(iz)%n_seg
            set%n_eff(iz) = ops(iz)%n_eff
            set%pt_off(iz) = np;  np = np + ops(iz)%n_pts
            set%pc_off(iz) = npc; npc = npc + ops(iz)%n_pc + 1
            set%seg_off(iz) = nsg; nsg = nsg + ops(iz)%n_seg + 1
            set%ent_off(iz) = nen; nen = nen + ops(iz)%eff_ptr(ops(iz)%n_eff + 1) - 1
            set%eff_off(iz) = nef; nef = nef + ops(iz)%n_eff + 1
            set%n_pts_max = max(set%n_pts_max, ops(iz)%n_pts)
            set%n_eff_max = max(set%n_eff_max, ops(iz)%n_eff)
            set%n_seg_max = max(set%n_seg_max, ops(iz)%n_seg)
            set%n_pc_max = max(set%n_pc_max, ops(iz)%n_pc)
        end do
        allocate(set%pt_first(max(np, 1)), set%pt_age(max(np, 1)), set%pt_mini(max(np, 1)), set%pt_m1(max(np, 1)), &
                 set%pt_m2(max(np, 1)), set%pt_cm(max(np, 1)), set%pt_cl(max(np, 1)), set%pt_q(max(np, 1)), &
                 set%pt_llo(max(np, 1)), set%pt_dl(max(np, 1)))
        allocate(set%pc_ptr(npc), set%seg_ptr(nsg), set%seg_first(nef), set%eff_pt(max(nen, 1)), set%eff_val(max(nen, 1)))
        allocate(set%age_pc(nz * (nt + 1)), set%pt_ptr(nz * (nt + 1)), set%age_mmax(nz * nt))
        do iz = 1, nz
            associate (o => ops(iz))
                i = set%pt_off(iz)
                set%pt_first(i + 1:i + o%n_pts) = o%pt_first
                set%pt_age(i + 1:i + o%n_pts) = o%pt_age
                set%pt_mini(i + 1:i + o%n_pts) = o%pt_mini
                set%pt_m1(i + 1:i + o%n_pts) = o%pt_m1
                set%pt_m2(i + 1:i + o%n_pts) = o%pt_m2
                set%pt_cm(i + 1:i + o%n_pts) = o%pt_cm
                set%pt_cl(i + 1:i + o%n_pts) = o%pt_cl
                set%pt_q(i + 1:i + o%n_pts) = o%pt_q
                set%pc_ptr(set%pc_off(iz) + 1:set%pc_off(iz) + o%n_pc + 1) = o%pc_ptr(1:o%n_pc + 1) + set%pt_off(iz)
                set%seg_ptr(set%seg_off(iz) + 1:set%seg_off(iz) + o%n_seg + 1) = o%seg_ptr(1:o%n_seg + 1) + set%ent_off(iz)
                set%seg_first(set%eff_off(iz) + 1:set%eff_off(iz) + o%n_eff + 1) = o%seg_first(1:o%n_eff + 1)
                k = o%eff_ptr(o%n_eff + 1) - 1
                set%eff_pt(set%ent_off(iz) + 1:set%ent_off(iz) + k) = o%eff_pt(1:k)
                set%eff_val(set%ent_off(iz) + 1:set%ent_off(iz) + k) = o%eff_val(1:k)
                set%age_pc((iz - 1) * (nt + 1) + 1:iz * (nt + 1)) = o%age_pc(1:nt + 1)
                set%pt_ptr((iz - 1) * (nt + 1) + 1:iz * (nt + 1)) = o%pt_ptr(1:nt + 1)
                set%age_mmax((iz - 1) * nt + 1:iz * nt) = o%age_mmax(1:nt)
            end associate
        end do
        call imf_bin_logs(set%pt_m1, set%pt_m2, set%pt_llo, set%pt_dl)
        allocate(set%eff_f(2, max(nen, 1)))
        set%eff_f = 0.0_c_float
        do i = 1, nen
            set%eff_f(1, i) = real(set%eff_val(i), c_float)
            set%eff_f(2, i) = real(set%eff_val(i) - real(set%eff_f(1, i), WP), c_float)
        end do
        allocate(set%pt_f(8, max(np, 1)))
        set%pt_f = 0.0_c_float
        do i = 1, np
            set%pt_f(1, i) = real(set%pt_llo(i), c_float)
            set%pt_f(2, i) = real(set%pt_llo(i) - real(set%pt_f(1, i), WP), c_float)
            set%pt_f(3, i) = real(set%pt_dl(i), c_float)
            set%pt_f(4, i) = real(set%pt_dl(i) - real(set%pt_f(3, i), WP), c_float)
            set%pt_f(5, i) = real(set%pt_cm(i), c_float)
            set%pt_f(6, i) = real(set%pt_cl(i), c_float)
            set%pt_f(7, i) = real(set%pt_q(i) * 2.0_wp**(-80), c_float)
        end do
        call entries_by_point(set, np, max(nen, 1))
        call move_alloc(ops, set%ops)
    end subroutine ssp_opset_build

    !> The operators of the modification groups of ctx at the context's current parameters (the non-factor ones: IMF type
    !> and limits are not used, evtype, masscut, agb_dust, the TP-AGB normalisation are the operators'), for the
    !> metallicity pset%zmet (all_z false) or every metallicity of the context (all_z true), as the container of a batch with
    !> the factors fbhb, sbss, agb, pagb and redgb free per model.
    subroutine ssp_opset_build_mods(ctx, pset, all_z, set)
        type(opensps_context_t), target, intent(inout) :: ctx
        type(params), intent(in) :: pset
        logical, intent(in) :: all_z
        type(ssp_opset_t), intent(inout) :: set
        type(ssp_operator_t), allocatable :: ops(:)
        type(params) :: p
        integer :: g, z, nzo

        call ssp_opset_free(set)
        nzo = 1
        if (all_z) nzo = ctx%state%nz
        allocate(set%grp(0:MOD_NGRP - 1))
        p = pset
        do g = 0, MOD_NGRP - 1
            allocate(ops(nzo))
            do z = 1, nzo
                if (all_z) p%zmet = z
                call ssp_operator_build(ctx, p, ops(z), group=g)
            end do
            call ssp_opset_build(ops, set%grp(g))
            if (allocated(ops)) deallocate(ops)
        end do
        set%free_mods = .true.
        set%nz = set%grp(0)%nz
        set%nt = set%grp(0)%nt
        set%base_pset = pset
        set%all_z = all_z
        allocate(set%ext(MAX_TP_PAIRS))
        set%n_ext = 0
        set%ext_ok = .false.
        set%max_ext = merge(MAX_TP_PAIRS_ALLZ, MAX_TP_PAIRS, all_z)
    end subroutine ssp_opset_build_mods

    !> The set of group g of a free-modification set: the groups of ssp_opset_build_mods (0 .. MOD_NGRP - 1), then the TP-AGB
    !> group at each pair of ext (MOD_NGRP + k - 1 for ext(k))
    function gset(set, g) result(sg)
        type(ssp_opset_t), intent(in), target :: set
        integer, intent(in) :: g
        type(ssp_opset_t), pointer :: sg
        if (g < MOD_NGRP) then
            sg => set%grp(g)
        else
            sg => set%ext(g - MOD_NGRP + 1)
        end if
    end function gset

    !> The operators of the TP-AGB group at the shifts (delt, dell), for the metallicities of the set (as ssp_opset_build_mods)
    subroutine ssp_opset_build_tp(ctx, base, delt, dell, out)
        type(opensps_context_t), target, intent(inout) :: ctx
        type(ssp_opset_t), intent(in) :: base
        real(WP), intent(in) :: delt, dell
        type(ssp_opset_t), intent(inout) :: out
        type(ssp_operator_t), allocatable :: ops(:)
        type(params) :: p
        integer :: z, nzo

        nzo = 1
        if (base%all_z) nzo = ctx%state%nz
        allocate(ops(nzo))
        p = base%base_pset
        do z = 1, nzo
            if (base%all_z) p%zmet = z
            call ssp_operator_build(ctx, p, ops(z), group=MOD_GRP_TPAGB, shift_t=delt, shift_l=dell)
        end do
        call ssp_opset_build(ops, out)
    end subroutine ssp_opset_build_tp

    !> The pair of each model (ipair: 0 for delt = dell = 0, else k of ext(k)), building the sets of the pairs not seen yet,
    !> in a free slot or one whose pair this batch does not use. status 2: a batch with more than set%max_ext distinct pairs.
    subroutine ssp_mods_prepare_pairs(ctx, set, delt, dell, ipair, status)
        type(opensps_context_t), target, intent(inout) :: ctx
        type(ssp_opset_t), intent(inout) :: set
        real(WP), intent(in) :: delt(:), dell(:)
        integer, intent(out) :: ipair(:), status
        integer :: m, k, ip, nnew, slot
        real(WP) :: nt_(MAX_TP_PAIRS), nl_(MAX_TP_PAIRS)
        integer :: where_(MAX_TP_PAIRS)
        logical :: inuse(MAX_TP_PAIRS)

        status = 0
        ipair = 0
        ! The distinct pairs of this batch
        nnew = 0
        do m = 1, size(delt)
            if (delt(m) == 0.0_wp .and. dell(m) == 0.0_wp) cycle
            do k = 1, nnew
                if (nt_(k) == delt(m) .and. nl_(k) == dell(m)) exit
            end do
            if (k > nnew) then
                if (nnew == set%max_ext) then
                    status = 2
                    return
                end if
                nnew = nnew + 1
                nt_(nnew) = delt(m)
                nl_(nnew) = dell(m)
            end if
        end do
        ! The slots of the pairs already built (those are kept), then the new pairs into free or unused slots
        inuse = .false.
        where_ = 0
        do k = 1, nnew
            do ip = 1, set%n_ext
                if (set%ext_ok(ip) .and. set%ext_t(ip) == nt_(k) .and. set%ext_l(ip) == nl_(k)) then
                    where_(k) = ip
                    inuse(ip) = .true.
                    exit
                end if
            end do
        end do
        do k = 1, nnew
            if (where_(k) > 0) cycle
            slot = 0
            do ip = 1, MAX_TP_PAIRS
                if (.not. set%ext_ok(ip)) then
                    slot = ip
                    exit
                end if
            end do
            if (slot == 0) then
                do ip = 1, MAX_TP_PAIRS
                    if (.not. inuse(ip)) then
                        slot = ip
                        exit
                    end if
                end do
                call ssp_opset_free(set%ext(slot))
                set%ext_ok(slot) = .false.
            end if
            call ssp_opset_build_tp(ctx, set, nt_(k), nl_(k), set%ext(slot))
            set%ext_t(slot) = nt_(k)
            set%ext_l(slot) = nl_(k)
            set%ext_ok(slot) = .true.
            set%n_ext = max(set%n_ext, slot)
            inuse(slot) = .true.
            where_(k) = slot
        end do
        do m = 1, size(delt)
            if (delt(m) == 0.0_wp .and. dell(m) == 0.0_wp) cycle
            do k = 1, nnew
                if (nt_(k) == delt(m) .and. nl_(k) == dell(m)) then
                    ipair(m) = where_(k)
                    exit
                end if
            end do
        end do
    end subroutine ssp_mods_prepare_pairs

    !> The template entries of each point (pe_ptr, pe_k, pe_e): counts, then a fill in template order.
    subroutine entries_by_point(set, np, nen)
        type(ssp_opset_t), intent(inout) :: set
        integer, intent(in) :: np, nen
        integer :: iz, e, s, sg, k, gp, eg
        integer, allocatable :: fill(:)

        allocate(set%pe_ptr(np + 1), set%pe_k(nen), set%pe_e(nen), fill(np))
        fill = 0
        do iz = 1, set%nz
            do e = 1, set%n_eff(iz)
                eg = set%eff_off(iz) + e
                do s = set%seg_first(eg), set%seg_first(eg + 1) - 1
                    sg = set%seg_off(iz) + s
                    do k = set%seg_ptr(sg), set%seg_ptr(sg + 1) - 1
                        gp = set%pt_off(iz) + set%eff_pt(k)
                        fill(gp) = fill(gp) + 1
                    end do
                end do
            end do
        end do
        set%pe_ptr(1) = 1
        do gp = 1, np
            set%pe_ptr(gp + 1) = set%pe_ptr(gp) + fill(gp)
        end do
        fill = 0
        do iz = 1, set%nz
            do e = 1, set%n_eff(iz)
                eg = set%eff_off(iz) + e
                do s = set%seg_first(eg), set%seg_first(eg + 1) - 1
                    sg = set%seg_off(iz) + s
                    do k = set%seg_ptr(sg), set%seg_ptr(sg + 1) - 1
                        gp = set%pt_off(iz) + set%eff_pt(k)
                        set%pe_k(set%pe_ptr(gp) + fill(gp)) = k
                        set%pe_e(set%pe_ptr(gp) + fill(gp)) = e
                        fill(gp) = fill(gp) + 1
                    end do
                end do
            end do
        end do
    end subroutine entries_by_point

    recursive subroutine ssp_opset_free(set)
        type(ssp_opset_t), intent(inout) :: set
        integer :: iz, ig, ip
        if (associated(set%grp)) then
            do ig = lbound(set%grp, 1), ubound(set%grp, 1)
                call ssp_opset_free(set%grp(ig))
            end do
            deallocate(set%grp)
            nullify(set%grp)
        end if
        if (allocated(set%slp)) deallocate(set%slp)
        if (associated(set%ext)) then
            do ig = 1, size(set%ext)
                call ssp_opset_free(set%ext(ig))
            end do
            deallocate(set%ext)
            nullify(set%ext)
        end if
        set%n_ext = 0
        set%ext_ok = .false.
        set%free_mods = .false.
#ifdef OMP_TARGET
        if (set%on_device) then
            !$omp target exit data map(delete: set%n_pc, set%n_seg, set%n_eff, set%pt_off, set%pc_off, set%seg_off, &
            !$omp     set%ent_off, set%eff_off, set%pt_first, set%pt_age, set%pc_ptr, set%age_pc, set%pt_ptr, set%seg_ptr, &
            !$omp     set%seg_first, set%eff_pt, set%pt_mini, set%pt_m1, set%pt_m2, set%pt_cm, set%pt_cl, set%pt_q, &
            !$omp     set%age_mmax, set%eff_val, set%pt_llo, set%pt_dl, set%pt_f, set%eff_f, set%pe_ptr, set%pe_k, set%pe_e)
            set%on_device = .false.
        end if
#endif
        if (allocated(set%ops)) then
            do iz = 1, size(set%ops)
                call ssp_operator_from_device(set%ops(iz))
            end do
            deallocate(set%ops)
        end if
        if (allocated(set%n_pc)) deallocate(set%n_pc, set%n_seg, set%n_eff, set%pt_off, set%pc_off, set%seg_off, &
                                            set%ent_off, set%eff_off)
        if (allocated(set%pt_first)) deallocate(set%pt_first, set%pt_age, set%pt_mini, set%pt_m1, set%pt_m2, set%pt_cm, &
                                                set%pt_cl, set%pt_q, set%pc_ptr, set%seg_ptr, set%seg_first, set%eff_pt, &
                                                set%eff_val, set%age_pc, set%pt_ptr, set%age_mmax, set%pt_llo, set%pt_dl, &
                                                set%pt_f, set%eff_f)
        if (allocated(set%pe_ptr)) deallocate(set%pe_ptr, set%pe_k, set%pe_e)
        set%nz = 0; set%nt = 0; set%n_pts_max = 0; set%n_eff_max = 0; set%n_seg_max = 0; set%n_pc_max = 0
    end subroutine ssp_opset_free

    !> Maps the set's tables and its operators to the device, once.
    subroutine ssp_opset_to_device(set)
        type(ssp_opset_t), intent(inout) :: set
        integer :: iz
#ifdef OMP_TARGET
        if (set%on_device) return
        !$omp target enter data map(to: set%n_pc, set%n_seg, set%n_eff, set%pt_off, set%pc_off, set%seg_off, &
        !$omp     set%ent_off, set%eff_off, set%pt_first, set%pt_age, set%pc_ptr, set%age_pc, set%pt_ptr, set%seg_ptr, &
        !$omp     set%seg_first, set%eff_pt, set%pt_mini, set%pt_m1, set%pt_m2, set%pt_cm, set%pt_cl, set%pt_q, &
        !$omp     set%age_mmax, set%eff_val, set%pt_llo, set%pt_dl, set%pt_f, set%eff_f, set%pe_ptr, set%pe_k, set%pe_e)
        do iz = 1, set%nz
            call ssp_operator_to_device(set%ops(iz))   ! (the template matrices)
        end do
        set%on_device = .true.
#endif
    end subroutine ssp_opset_to_device

    !> The age sums with the metallicity free per model: the SSP of model m is the blend
    !> (1 - dz(m)) SSP(zlo(m)) + dz(m) SSP(zhi(m)) of the SSPs of the two bracketing metallicities of
    !> the context (as compute_csp_batch interpolates a resident basis), each from its own
    !> operator (the set: one per metallicity). The blend is linear in the SFH-weighted template
    !> sums, so each model is processed twice, once per bracketing operator ("passes": lower and upper),
    !> with its SFH weights scaled by the blend coefficient, and the results are added. In a pass
    !> the models are sorted by metallicity (a permutation) and run in chunks; one launch of each kernel
    !> serves the models of every metallicity of the chunk (the tables of the set, offsets per
    !> metallicity), only the template products are per metallicity run. The mass and the ionizing
    !> photons are blended linearly and L_bol in its logarithm, as for a basis. One metallicity (nz = 1)
    !> is the one-pass special case, its products going straight into yo. Device (OpenMP target) and host
    !> (OpenMP loops) builds run the same code.
    recursive subroutine ssp_batch_age_sums_z(imf_type, remnants, n_mod, nt, nspec, set, zlo, zhi, dz, pm, wsfh, it_tesc, &
                                    nt_neb, whl, fesc, yo, smass, slum, q0, cg, slpass)
        logical, intent(in)  :: remnants
        integer, intent(in)  :: imf_type, n_mod, nt, nspec, nt_neb, whl
        type(ssp_opset_t), intent(inout) :: set
        integer, intent(in), contiguous  :: zlo(:), zhi(:), it_tesc(:)
        real(WP), intent(in), contiguous :: dz(:), pm(:,:), wsfh(:,:), fesc(:)
        real(KP), intent(inout), contiguous :: yo(:,:)
        real(WP), intent(inout), contiguous :: smass(:,:), slum(:,:), q0(:,:)
        !> The coefficients of the modification groups per model ((MOD_NGRP, n_mod); a set with free modifications,
        !> ssp_opset_build_mods, needs them: the batch is then the sum over the groups, ssp_batch_age_sums_mods)
        real(WP), intent(in), optional, contiguous :: cg(:,:)
        !> Out: the linear L_bol sum of this set's operators at each age, model and metallicity pass (nt, n_mod, 2)
        !> (the pass of zlo: 1, of zhi: 2; zero where the model has no weight in the pass), for the sum over groups
        real(WP), intent(inout), optional, contiguous :: slpass(:,:,:)

        integer :: CHUNK
        integer :: nc_max, neb_cols, nz, pass, m, k, t, ntq
        integer, allocatable :: perm1(:), perm2(:), viz1(:), viz2(:), ptr1(:), ptr2(:), cnt(:)
        real(WP), allocatable :: cz1(:), cz2(:), w(:,:), tot(:), vp(:,:,:), pps(:,:,:), slz_l(:,:), sqz_l(:,:), slp_d(:,:,:)
        real(KP), allocatable :: wk(:,:), v(:,:), veuv(:,:), yz(:,:)
        logical :: direct, fwd_w, want_sl

        if (set%free_mods) then
            if (.not. present(cg)) error stop 'ssp_batch_age_sums_z: a set with free modifications needs the group coefficients'
            call ssp_batch_age_sums_mods(imf_type, remnants, n_mod, nt, nspec, set, cg, zlo, zhi, dz, pm, wsfh, it_tesc, &
                                         nt_neb, whl, fesc, yo, smass, slum, q0)
            return
        end if
        want_sl = present(slpass)
        nz = set%nz
        ! One metallicity: the models in their own order, one pass, the products go straight into yo
        direct = nz == 1
        call ssp_opset_to_device(set)
        neb_cols = 0
        if (nt_neb > 0 .and. whl > 0) neb_cols = 1
        CHUNK = tuning_get(TUNE_P_SSP_BATCH_CHUNK)
        nc_max = min(CHUNK, n_mod)
        allocate(perm1(n_mod), perm2(n_mod), viz1(n_mod), viz2(n_mod), ptr1(nz + 1), ptr2(nz + 1), cnt(nz + 1), &
                 cz1(n_mod), cz2(n_mod))
        cz1 = 1.0_wp - dz(1:n_mod)
        cz2 = dz(1:n_mod)
        call group_models(n_mod, nz, zlo, cz1, perm1, ptr1, cnt)
        call group_models(n_mod, nz, zhi, cz2, perm2, ptr2, cnt)
        do k = 1, ptr1(nz + 1) - 1
            viz1(k) = zlo(perm1(k))
        end do
        do k = 1, ptr2(nz + 1) - 1
            viz2(k) = zhi(perm2(k))
        end do
        ntq = size(q0, 1)

        ! The FP64 point weights w serve the reverse pass; the forward reads only wk. The HIP weights kernel does
        ! not write w, so there it is a dummy array (the other paths use w as the scratch of imf_bin_weights_tab).
        fwd_w = .true.
#if defined(OPENSPS_HIP) && defined(OMP_TARGET)
        if (mod(imf_type, 10) == 0 .or. mod(imf_type, 10) == 2 .or. mod(imf_type, 10) == 4) fwd_w = .false.
#endif
        if (fwd_w) then
            allocate(w(nc_max, set%n_pts_max))
        else
            allocate(w(1, 1))
        end if
        allocate(wk(nc_max, set%n_pts_max), tot(nc_max), v(set%n_eff_max, 2 * nc_max))
        allocate(veuv(set%n_eff_max, max(1, neb_cols * 2 * nc_max)), vp(nc_max, 4, set%n_seg_max), &
                 pps(nc_max, 3, set%n_pc_max))
        if (want_sl) then
            allocate(slz_l(nt, nc_max), sqz_l(nt, nc_max))
        else
            allocate(slz_l(1, 1), sqz_l(1, 1))
        end if
        if (direct) then
            allocate(yz(1, 1))
        else
            allocate(yz(nspec, 2 * nc_max))
        end if
#ifdef OMP_TARGET
        !$omp target data map(alloc: w, wk, tot, v, veuv, vp, pps, yz, slz_l, sqz_l) map(to: perm1, perm2, viz1, viz2, cz1, cz2)
#endif
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do t = 1, nt
                smass(t, m) = 0.0_wp
                slum(t, m) = 0.0_wp
            end do
        end do
        if (nt_neb > 0) then
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, n_mod
                do t = 1, ntq
                    q0(t, m) = 0.0_wp
                end do
            end do
        end if
        if (want_sl) then
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, n_mod
                do t = 1, nt
                    slpass(t, m, 1) = 0.0_wp
                    slpass(t, m, 2) = 0.0_wp
                end do
            end do
        end if
        if (.not. direct) then
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, 2 * n_mod
                do k = 1, nspec
                    yo(k, m) = 0.0_KP
                end do
            end do
        end if
        do pass = 1, 2
            if (pass == 1) then
                call run_pass(ptr1(nz + 1) - 1, perm1, viz1, cz1, 1)
            else
                call run_pass(ptr2(nz + 1) - 1, perm2, viz2, cz2, 2)
            end if
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do t = 1, nt
                slum(t, m) = 10.0_wp**slum(t, m)
            end do
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif

    contains

        !> The np virtual models (model perm(j), metallicity viz(j), sorted by viz) of a pass, in chunks
        subroutine run_pass(np, perm, viz, cz, ipass)
            integer, intent(in) :: np, ipass
            integer, intent(in), contiguous :: perm(:), viz(:)
            real(WP), intent(in), contiguous :: cz(:)
            integer :: m0, nc, j, j0, jr, iz, ky, nrun

            do m0 = 1, np, CHUNK
                nc = min(CHUNK, np - m0 + 1)
                call opensps_trace_push("fwd_imf_pieces_multi")
                call imf_pieces_multi(imf_type, remnants, m0, nc, nt, set%n_pc_max, nt_neb, pm, set%n_pc, set%pt_off, &
                                      set%pc_off, set%pc_ptr, set%age_pc, set%pt_ptr, set%pt_mini, set%pt_m1, set%pt_m2, &
                                      set%pt_llo, set%pt_dl, set%pt_first, set%pt_cm, set%pt_cl, set%pt_q, set%age_mmax, &
                                      size(wk, 1), set%n_pts_max, w, wk, tot, pps, perm, viz, cz, smass, slum, q0, fwd_w, set%pt_f)
                call opensps_trace_pop()
                if (want_sl) then
                    call adj_age_sums_sl(m0, nc, nt, set%age_pc, viz, pps, slz_l, sqz_l)
                    call scatter_sl(m0, nc, nt, perm, slz_l, slpass, ipass)
                end if
                call opensps_trace_push("fwd_fold_multi")
                call fold_multi(m0, nc, set%n_eff_max, set%n_seg_max, nt_neb, neb_cols, wsfh, it_tesc, fesc, &
                                set%n_seg, set%n_eff, set%pt_off, set%seg_off, set%eff_off, set%ent_off, set%pt_age, &
                                set%eff_pt, set%eff_val, set%seg_ptr, set%seg_first, perm, viz, cz, wk, vp, v, veuv, set%eff_f)
                call opensps_trace_pop()
                ! The template products, per run of one metallicity in the chunk
                if (direct) then
                    call products(m0, nc, perm, viz, yo, m0 - 1, n_mod + m0 - 1)
                else
                    call products(m0, nc, perm, viz, yz, 0, nc)
                    call add_columns(m0, nc, nspec, n_mod, perm, yz, yo)
                end if
            end do
        end subroutine run_pass

        !> The template products of the runs of one metallicity in a chunk: young columns
        !> out(:, cy + j), old columns out(:, co + j), j = the chunk's virtual models
        subroutine products(m0, nc, perm, viz, out, cy, co)
            integer, intent(in) :: m0, nc, cy, co
            integer, intent(in), contiguous :: perm(:), viz(:)
            real(KP), intent(inout), contiguous :: out(:,:)
            integer :: j, j0, jr, iz, ky, nrun

            j0 = 1
            do while (j0 <= nc)
                iz = viz(m0 + j0 - 1)
                jr = j0
                do while (jr < nc)
                    if (viz(m0 + jr) /= iz) exit
                    jr = jr + 1
                end do
                nrun = jr - j0 + 1
                ky = 0
                do j = j0, jr
                    ky = max(ky, it_tesc(perm(m0 + j - 1)))
                end do
                ky = set%ops(iz)%k_age(max(1, min(nt, ky)))
                call template_columns(ON_DEV, nspec, nspec, size(v, 1), ky, nrun, set%ops(iz)%etpl, v, j0 - 1, out, &
                                      cy + j0 - 1)
                call template_columns(ON_DEV, nspec, nspec, size(v, 1), set%ops(iz)%n_eff, nrun, set%ops(iz)%etpl, v, &
                                      nc + j0 - 1, out, co + j0 - 1)
                if (neb_cols > 0) then
                    call template_columns(ON_DEV, whl, nspec, size(v, 1), ky, nrun, set%ops(iz)%etpl, veuv, j0 - 1, &
                                          out, cy + j0 - 1)
                    call template_columns(ON_DEV, whl, nspec, size(v, 1), set%ops(iz)%n_eff, nrun, set%ops(iz)%etpl, &
                                          veuv, nc + j0 - 1, out, co + j0 - 1)
                end if
                j0 = jr + 1
            end do
        end subroutine products

    end subroutine ssp_batch_age_sums_z

    !> The age sums of a batch with the isochrone modifications free per model (a set from ssp_opset_build_mods): the SSP
    !> of model m is the sum over the modification groups g of cg(g, m) times the SSP of the group's operators, so the
    !> young and old age sums, the surviving mass and the ionizing photon rate are the sums of the groups' (each from
    !> ssp_batch_age_sums_z, unchanged), weighted; the remnants (which do not depend on the factors) are added by the plain
    !> group alone. L_bol is the linear sum of the groups' sl at each metallicity, blended in its logarithm over the two
    !> bracketing metallicities as a basis does, so with the factors at their defaults it is that of a basis of the
    !> modified isochrones. The groups' sl are kept (set%slp) for the reverse pass. Groups no model uses are skipped.
    recursive subroutine ssp_batch_age_sums_mods(imf_type, remnants, n_mod, nt, nspec, set, cg, zlo, zhi, dz, pm, wsfh, it_tesc, &
                                       nt_neb, whl, fesc, yo, smass, slum, q0)
        logical, intent(in)  :: remnants
        integer, intent(in)  :: imf_type, n_mod, nt, nspec, nt_neb, whl
        type(ssp_opset_t), intent(inout) :: set
        real(WP), intent(in), contiguous :: cg(:,:)
        integer, intent(in), contiguous  :: zlo(:), zhi(:), it_tesc(:)
        real(WP), intent(in), contiguous :: dz(:), pm(:,:), wsfh(:,:), fesc(:)
        real(KP), intent(inout), contiguous :: yo(:,:)
        real(WP), intent(inout), contiguous :: smass(:,:), slum(:,:), q0(:,:)

        integer :: g, m, t, k, col, ntq, p, ng
        type(ssp_opset_t), pointer :: sg
        real(KP), allocatable :: yg(:,:)
        real(WP), allocatable :: sm_g(:,:), sl_g(:,:), q_g(:,:), slp(:,:,:,:)
        real(WP) :: lz, sx, czp
        real(KP) :: ck
        logical, allocatable :: used(:)

        ntq = size(q0, 1)
        ng = size(cg, 1)
        allocate(used(ng))
        do g = 1, ng
            used(g) = any(cg(g, 1:n_mod) /= 0.0_wp)
        end do
        allocate(yg(nspec, 2 * n_mod), sm_g(nt, n_mod), sl_g(nt, n_mod), q_g(ntq, n_mod), slp(nt, n_mod, 2, ng))
#ifdef OMP_TARGET
        !$omp target data map(alloc: yg, sm_g, sl_g, q_g) map(from: slp) map(to: cg)
#endif
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do col = 1, 2 * n_mod
            do k = 1, nspec
                yo(k, col) = 0.0_KP
            end do
        end do
        ! (a group's ssp_batch_age_sums_z leaves its q0 alone without nebular ages: it is read below)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do t = 1, ntq
                q_g(t, m) = 0.0_wp
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do t = 1, nt
                smass(t, m) = 0.0_wp
                slp(t, m, 1, :) = 0.0_wp
                slp(t, m, 2, :) = 0.0_wp
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do t = 1, ntq
                q0(t, m) = 0.0_wp
            end do
        end do
        do g = 0, ng - 1
            if (.not. used(g + 1)) cycle
            sg => gset(set, g)
            call ssp_batch_age_sums_z(imf_type, remnants .and. g == 0, n_mod, nt, nspec, sg, zlo, zhi, dz, pm, wsfh, &
                                      it_tesc, nt_neb, whl, fesc, yg, sm_g, sl_g, q_g, slpass=slp(:, :, :, g + 1))
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2) private(m, ck)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2) private(m, ck)
#endif
            do col = 1, 2 * n_mod
                do k = 1, nspec
                    m = col
                    if (col > n_mod) m = col - n_mod
                    ck = real(cg(g + 1, m), KP)
                    yo(k, col) = yo(k, col) + ck * yg(k, col)
                end do
            end do
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, n_mod
                do t = 1, nt
                    smass(t, m) = smass(t, m) + cg(g + 1, m) * sm_g(t, m)
                end do
            end do
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
            !$omp parallel do collapse(2)
#endif
            do m = 1, n_mod
                do t = 1, ntq
                    q0(t, m) = q0(t, m) + cg(g + 1, m) * q_g(t, m)
                end do
            end do
        end do
        ! L_bol: the groups' sums at each metallicity, then the logarithmic blend of the passes (as a basis' L_bol)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(sx, czp, lz, p, g)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(sx, czp, lz, p, g)
#endif
        do m = 1, n_mod
            do t = 1, nt
                sx = 0.0_wp
                do p = 1, 2
                    if (p == 1) then
                        czp = 1.0_wp - dz(m)
                    else
                        czp = dz(m)
                    end if
                    if (czp /= 0.0_wp) then
                        lz = 0.0_wp
                        do g = 1, ng
                            lz = lz + cg(g, m) * slp(t, m, p, g)
                        end do
                        sx = sx + czp * log10(max(lz, SAFE_FLOOR))
                    end if
                end do
                slum(t, m) = 10.0_wp**sx
            end do
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
        call move_alloc(slp, set%slp)
    end subroutine ssp_batch_age_sums_mods

    !> The reverse pass of ssp_batch_age_sums_z (the gradients of compute_csp_batch with ssp_ops).
    !> The age sums are T v (T: a metallicity's effective templates; v: the fold of the per-point IMF
    !> weights w_p with the SFH weights, v_e = sum over template e's segments and their ages t of
    !> cz wsfh(t) sum_p eff_val w_p), the EUV rows of the nebular ages with the escaping fraction f,
    !> and per age the L_bol (10^(sum_z cz log10 sl_z), sl = sum_p cl_p w_p) and ionizing photons
    !> (sum_z cz sq_z). From the cotangents of the young and old sums (gw), of L_bol (lsum_bar,
    !> through sum_t wsfh(t) L(t)) and of Q (qbar):
    !>   v_bar = T^T y_bar (rows above whl; the EUV rows apart),
    !>   wbar(t) = sum cz a_t (v_bar + f v_euv_bar) + lsum_bar L(t)   (the SFH weights),
    !>   w_p bar = eff_val cz wsfh(t) (v_bar + f v_euv_bar) + lsum_bar wsfh(t) L(t) cz cl_p / sl_z
    !>             + qbar(t) cz q_p,
    !>   imf_bar(i) = sum_p w_p bar d w_p / d theta_i (theta: imf1, imf2, imf3, vdmc, mdave; d w_p / d
    !>   theta by central differences of the closed-form bin integrals, imf_bin_weights_tab: step
    !>   1e-6, error ~1e-10 relative; in FP32 builds step 1e-3, ~1e-4),
    !>   fesc_bar, and zbar = d L / d dz (the two metallicity passes' difference, as the weights
    !>   blend them: linearly, L_bol in log).
    !> Chunks of CHG models (the point weights are chunk x points arrays); atomic adds where the
    !> fold's templates share ages and points.
    recursive subroutine ssp_batch_age_sums_adjoint(imf_type, remnants, n_mod, nt, nspec, set, zlo, zhi, dz, pm, wsfh, it_tesc, &
                                          nt_neb, whl, fesc, gw, slum, lsum_bar, qbar, wbar, imf_bar, fesc_bar, zbar, &
                                          smass, msum_bar, cg, cbar, lfac, ipair, dshift, ctx)
        logical, intent(in)  :: remnants
        integer, intent(in)  :: imf_type, n_mod, nt, nspec, nt_neb, whl
        type(ssp_opset_t), intent(inout) :: set
        integer, intent(in), contiguous  :: zlo(:), zhi(:), it_tesc(:)
        real(WP), intent(in), contiguous :: dz(:), pm(:,:), wsfh(:,:), fesc(:), slum(:,:), lsum_bar(:), qbar(:,:)
        real(KP), intent(in), contiguous :: gw(:,:)
        real(WP), intent(out), contiguous :: wbar(:,:), imf_bar(:,:), fesc_bar(:), zbar(:)
        !> The surviving mass per age and model (the forward's, with the remnants) and the mass output's
        !> cotangent (scaled as lsum_bar): mass = sum over ages of wsfh smass, smass = sum over the two
        !> metallicities of cz (stars + remnants), each a function of the IMF parameters
        real(WP), intent(in), contiguous :: smass(:,:), msum_bar(:)
        !> A set with free modifications (ssp_opset_build_mods): the groups' coefficients per model, and out the
        !> cotangent of each coefficient (MOD_NGRP, n_mod); the reverse pass is then ssp_batch_age_sums_adjoint_mods
        real(WP), intent(in), optional, contiguous :: cg(:,:)
        real(WP), intent(out), optional, contiguous :: cbar(:,:)
        !> L_bol's cotangent of the point weights supplied (nt, n_mod, 2: the passes of zlo and zhi): the factor of cl_p in
        !> wsfh(t) lfac(t, m, pass) instead of lsum_bar wsfh(t) L(t) cz / sl (the sum over groups of a batch with free
        !> modifications has another L(t) / sl); used by ssp_batch_age_sums_adjoint_mods
        real(WP), intent(in), optional, contiguous :: lfac(:,:,:)
        !> The TP-AGB shifts of a batch with free modifications: see ssp_batch_age_sums_adjoint_mods
        integer, intent(in), optional, contiguous :: ipair(:)
        real(WP), intent(out), optional, contiguous :: dshift(:,:)
        type(opensps_context_t), target, intent(inout), optional :: ctx

        integer, parameter :: CHG = TUNE_IMF_GRAD_CHG
        integer, parameter :: PROWS(NIMFG) = [IMF_P_ALPHA, IMF_P_ALPHA + 1, IMF_P_ALPHA + 2, IMF_P_VDMC, IMF_P_MDAVE]
        integer :: nz, m, t, j, np1, np2, nc_max, neb_cols, ntq
        integer, allocatable :: perm1(:), perm2(:), viz1(:), viz2(:), ptr(:), cnt(:), it1(:), it2(:)
        real(WP), allocatable :: cz1(:), cz2(:), zsg1(:), zsg2(:), fe1(:), fe2(:), ones(:), w(:,:), wm(:,:), tot(:), &
                                 pps(:,:,:), slz(:,:), sqz(:,:), wpb(:,:), vb(:,:,:), dsm(:,:), dsl(:,:), dsq(:,:), &
                                 pmp(:,:), part(:,:), dsp(:,:), sxa(:,:), sea(:,:), apt(:,:), amp(:,:,:), dtt(:,:), &
                                 remp(:,:), remm(:,:), agt(:,:,:), lfd(:,:,:)
        real(KP), allocatable :: wk(:,:)
        logical :: use_lfx

        use_lfx = present(lfac)
        if (use_lfx) then
            allocate(lfd(size(lfac, 1), size(lfac, 2), 2))
            lfd = lfac
        else
            allocate(lfd(1, 1, 2))
        end if
        if (set%free_mods) then
            if (.not. present(cg) .or. .not. present(cbar)) &
                error stop 'ssp_batch_age_sums_adjoint: a set with free modifications needs cg and cbar'
            ! (absent optional arguments are passed on as absent: without the shifts' arguments, no shift gradients)
            call ssp_batch_age_sums_adjoint_mods(imf_type, remnants, n_mod, nt, nspec, set, cg, zlo, zhi, dz, pm, wsfh, &
                                                 it_tesc, nt_neb, whl, fesc, gw, slum, lsum_bar, qbar, wbar, imf_bar, &
                                                 fesc_bar, zbar, smass, msum_bar, cbar, ipair, dshift, ctx)
            return
        end if
        nz = set%nz
        call ssp_opset_to_device(set)
        neb_cols = 0
        if (nt_neb > 0 .and. whl > 0) neb_cols = 1
        nc_max = min(CHG, n_mod)
        ntq = size(qbar, 1)
        allocate(perm1(n_mod), perm2(n_mod), viz1(n_mod), viz2(n_mod), ptr(nz + 1), cnt(nz + 1), &
                 cz1(n_mod), cz2(n_mod), zsg1(n_mod), zsg2(n_mod), ones(n_mod), &
                 it1(n_mod), it2(n_mod), fe1(n_mod), fe2(n_mod))
        allocate(w(nc_max, set%n_pts_max), wm(nc_max, set%n_pts_max), wk(nc_max, set%n_pts_max), tot(nc_max), &
                 pps(nc_max, 3, set%n_pc_max), slz(nt, nc_max), sqz(nt, nc_max), wpb(nc_max, set%n_pts_max), &
                 vb(set%n_eff_max, 4, nc_max), dsm(nt, n_mod), dsl(nt, n_mod), dsq(nt, n_mod), pmp(size(pm, 1), n_mod), &
                 part(set%n_pc_max, nc_max), dsp(nt, n_mod), sxa(nt, nc_max), sea(nt, nc_max), &
                 apt(NIMFG, nc_max), amp(NIMFG, nt, nc_max), dtt(NIMFG, nc_max), &
                 remp(nt, nc_max), remm(nt, nc_max), agt(NIMFG, nt, nc_max))
        ones = 1.0_wp
        ! Pass 1: zlo
        call group_models(n_mod, nz, zlo, ones, perm1, ptr, cnt)
        np1 = ptr(nz + 1) - 1
        do j = 1, np1
            viz1(j) = zlo(perm1(j))
            cz1(j) = 1.0_wp - dz(perm1(j))
            zsg1(j) = -1.0_wp
            it1(j) = it_tesc(perm1(j))
            if (neb_cols > 0) then
                fe1(j) = fesc(perm1(j))
            else
                fe1(j) = 1.0_wp
            end if
        end do
        ! Pass 2: zhi
        if (nz > 1) then
            call group_models(n_mod, nz, zhi, ones, perm2, ptr, cnt)
            np2 = ptr(nz + 1) - 1
            do j = 1, np2
                viz2(j) = zhi(perm2(j))
                cz2(j) = dz(perm2(j))
                zsg2(j) = 1.0_wp
                it2(j) = it_tesc(perm2(j))
                if (neb_cols > 0) then
                    fe2(j) = fesc(perm2(j))
                else
                    fe2(j) = 1.0_wp
                end if
            end do
        else
            np2 = 0
            perm2 = 0
            viz2 = 0
            cz2 = 0.0_wp
            zsg2 = 0.0_wp
            it2 = 0
            fe2 = 1.0_wp
        end if
#ifdef OMP_TARGET
        !$omp target data map(alloc: w, wm, wk, tot, pps, slz, sqz, wpb, vb, dsm, dsl, dsq, part, dsp, sxa, sea) &
        !$omp& map(alloc: apt, amp, dtt, remp, remm, agt, pmp) &
        !$omp& map(to: pm, ones, msum_bar, perm1, viz1, cz1, zsg1, it1, fe1, perm2, viz2, cz2, zsg2, it2, fe2) &
        !$omp& map(to: wsfh, slum, lsum_bar, qbar, lfd) &
        !$omp& map(from: wbar, imf_bar, fesc_bar, zbar)
#endif
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do
#elif defined(_OPENMP)
        !$omp parallel do
#endif
        do m = 1, n_mod
            do t = 1, nt
                ! (the SFH weights' L_bol and mass terms; the fold adds its part)
                wbar(t, m) = lsum_bar(m) * slum(t, m) + msum_bar(m) * smass(t, m)
            end do
            do j = 1, NIMFG
                imf_bar(j, m) = 0.0_wp
            end do
            fesc_bar(m) = 0.0_wp
            zbar(m) = 0.0_wp
        end do
        call run_pass_adj(np1, perm1, viz1, cz1, zsg1, it1, fe1, 1)
        if (nz > 1) call run_pass_adj(np2, perm2, viz2, cz2, zsg2, it2, fe2, 2)
#ifdef OMP_TARGET
        !$omp end target data
#endif

    contains

        subroutine run_pass_adj(np, perm, viz, cz, zsg, it, fe, ip)
            integer, intent(in) :: np, ip
            integer, intent(in), contiguous :: perm(:), viz(:), it(:)
            real(WP), intent(in), contiguous :: cz(:), zsg(:), fe(:)
            integer :: m0, nc, j0, jr, iz, ky, j, ig, isg, m, i
            real(WP) :: sgn

            do m0 = 1, np, CHG
                nc = min(CHG, np - m0 + 1)
                ! The point weights (and per age sl, sq of this metallicity; dsm: the surviving mass of this
                ! metallicity per age, for the mass output's dz)
                call adj_zero_cols(m0, nc, nt, perm, dsm)
                call opensps_trace_push("adj_imf_pieces_multi")
                call imf_pieces_multi(imf_type, remnants, m0, nc, nt, set%n_pc_max, nt_neb, pm, set%n_pc, set%pt_off, &
                                      set%pc_off, set%pc_ptr, set%age_pc, set%pt_ptr, set%pt_mini, set%pt_m1, set%pt_m2, &
                                      set%pt_llo, set%pt_dl, set%pt_first, set%pt_cm, set%pt_cl, set%pt_q, set%age_mmax, &
                                      size(w, 1), set%n_pts_max, w, wk, tot, pps, perm, viz, ones, dsm, dsl, dsq)
                call opensps_trace_pop()
                call opensps_trace_push("adj_age_sums_sl")
                call adj_age_sums_sl(m0, nc, nt, set%age_pc, viz, pps, slz, sqz)
                call opensps_trace_pop()
                call opensps_trace_push("adj_mass_dot")
                call adj_mass_dot(m0, nc, nt, perm, cz, zsg, wsfh, msum_bar, dsm, dsm, 0, 0, 1, pm, imf_bar, zbar)
                call opensps_trace_pop()
                call adj_zero(nc, set%n_pts_max, wpb)
                ! v_bar per run of one metallicity (the young product used the run's first ky templates)
                j0 = 1
                do while (j0 <= nc)
                    iz = viz(m0 + j0 - 1)
                    jr = j0
                    do while (jr < nc)
                        if (viz(m0 + jr) /= iz) exit
                        jr = jr + 1
                    end do
                    ky = 0
                    do j = j0, jr
                        ky = max(ky, it(m0 + j - 1))
                    end do
                    ky = set%ops(iz)%k_age(max(1, min(nt, ky)))
                    call opensps_trace_push("adj_template_bar")
                    call adj_template_bar(nspec, set%ops(iz)%n_eff, ky, neb_cols * whl, n_mod, m0, j0, jr, perm, &
                                          set%ops(iz)%etpl, set%ops(iz)%etpl_t, gw, set%n_eff_max, nc_max, vb)
                    call opensps_trace_pop()
                    j0 = jr + 1
                end do
                ! (w, wm: free until the IMF rows' differences below)
                call opensps_trace_push("adj_fold_gather")
                call adj_fold_gather(m0, nc, nt, nt_neb, neb_cols, set%n_pts_max, wsfh, it, fe, set%pt_off, &
                                     set%pt_ptr, set%pt_age, set%eff_val, set%pe_ptr, set%pe_k, set%pe_e, perm, viz, cz, &
                                     zsg, wk, vb, wbar, wpb, fesc_bar, zbar, w, wm, sxa, sea)
                call opensps_trace_pop()
                call opensps_trace_push("adj_lq")
                call adj_lq(m0, nc, nt, nt_neb, set%n_pc_max, set%n_pc, set%pt_off, set%pc_off, set%pc_ptr, set%pt_ptr, &
                            set%pt_age, set%pt_cl, set%pt_q, perm, viz, cz, zsg, wsfh, slum, lsum_bar, qbar, ntq, &
                            slz, sqz, wpb, zbar, lfd(:, :, ip), use_lfx)
                call opensps_trace_pop()
                ! The IMF parameters: Kroupa and Dave (power-law slopes and break) analytically
                ! (adj_imf_analytic); otherwise central differences of the point weights (only those the
                ! IMF type reads; the others' rows are zero)
                if (mod(imf_type, 10) == 2 .or. mod(imf_type, 10) == 4) then
                    call opensps_trace_push("adj_imf_analytic")
                    call adj_imf_analytic(imf_type, remnants, m0, nc, nt, set%n_pc_max, set%n_pts_max, set%n_pc, set%pt_off, &
                                          set%pc_off, set%pc_ptr, set%age_pc, set%pt_ptr, set%pt_mini, set%pt_m1, &
                                          set%pt_m2, set%pt_llo, set%pt_dl, set%pt_first, set%pt_cm, set%pt_age, set%age_mmax, &
                                          perm, viz, cz, wsfh, pm, &
                                          pmp, tot, size(wk, 1), wk, size(wpb, 1), wpb, msum_bar, apt, amp, agt, dtt, &
                                          remp, remm, imf_bar)
                    call opensps_trace_pop()
                    cycle
                end if
                do ig = 1, NIMFG
                    if (.not. imf_row_acts(imf_type, ig)) cycle
#if defined(OMP_TARGET)
                    !$omp target teams distribute parallel do
#elif defined(_OPENMP)
                    !$omp parallel do
#endif
                    do m = 1, n_mod
                        do i = 1, size(pm, 1)
                            pmp(i, m) = pm(i, m)
                        end do
                    end do
                    do isg = 1, 2
                        sgn = merge(1.0_wp, -1.0_wp, isg == 1)
                        call adj_perturb(n_mod, size(pm, 1), PROWS(ig), sgn, pm, pmp)
                        ! (the surviving mass at each side: dsp at +, dsm at -)
                        if (sgn > 0.0_wp) then
                            call adj_zero_cols(m0, nc, nt, perm, dsp)
                        else
                            call adj_zero_cols(m0, nc, nt, perm, dsm)
                        end if
                        if (sgn > 0.0_wp) then
                            call imf_pieces_multi(imf_type, remnants, m0, nc, nt, set%n_pc_max, nt_neb, pmp, set%n_pc, &
                                                  set%pt_off, set%pc_off, set%pc_ptr, set%age_pc, set%pt_ptr, set%pt_mini, &
                                                  set%pt_m1, set%pt_m2, set%pt_llo, set%pt_dl, set%pt_first, set%pt_cm, &
                                                  set%pt_cl, set%pt_q, set%age_mmax, size(w, 1), set%n_pts_max, w, wk, &
                                                  tot, pps, perm, viz, ones, dsp, dsl, dsq)
                        else
                            call imf_pieces_multi(imf_type, remnants, m0, nc, nt, set%n_pc_max, nt_neb, pmp, set%n_pc, &
                                                  set%pt_off, set%pc_off, set%pc_ptr, set%age_pc, set%pt_ptr, set%pt_mini, &
                                                  set%pt_m1, set%pt_m2, set%pt_llo, set%pt_dl, set%pt_first, set%pt_cm, &
                                                  set%pt_cl, set%pt_q, set%age_mmax, size(w, 1), set%n_pts_max, wm, wk, &
                                                  tot, pps, perm, viz, ones, dsm, dsl, dsq)
                        end if
                    end do
                    call adj_imf_dot(m0, nc, set%n_pc_max, set%n_pc, set%pt_off, set%pc_off, set%pc_ptr, perm, viz, &
                                     size(pm, 1), PROWS(ig), pm, w, wm, wpb, part, ig, imf_bar)
                    call adj_mass_dot(m0, nc, nt, perm, cz, zsg, wsfh, msum_bar, dsp, dsm, 1, PROWS(ig), ig, pm, &
                                      imf_bar, zbar)
                end do
            end do
        end subroutine run_pass_adj

    end subroutine ssp_batch_age_sums_adjoint

    !> The reverse pass of ssp_batch_age_sums_mods. The batch is sum_g cg(g, m) F_g with F_g the age sums, surviving
    !> mass, ionizing photon rate and linear L_bol sums of the group operators of the model (functions of the IMF, the
    !> SFH weights, the escaping fraction and the metallicity weight dz), and L_bol the logarithmic blend over the two
    !> metallicities of the sum over groups. Each group's outputs (SFH weights, IMF parameters, escaping fraction, dz) are
    !> those of the unchanged ssp_batch_age_sums_adjoint of its set, for the cotangents as given except L_bol's, weighted
    !> by the coefficient; the cotangent of each coefficient is the group's forward (recomputed) against the cotangents.
    !> L_bol's part is formed once from the groups' sl of the forward (set%slp), by d L / d sl(g, z) = L cz_z cg_g / L_z with
    !> L_z = sum_g cg_g sl(g, z): the SFH weights' lsum_bar L(t); dz's lsum_bar wsfh L ln 10 (log10 L_hi - log10 L_lo) (the
    !> passes the forward ran); and the point weights' lsum_bar L cz / L_z of the group's pass (lfac). The surviving
    !> mass's cotangent of the SFH weights is that of the total mass, given to the plain group's pass.
    recursive subroutine ssp_batch_age_sums_adjoint_mods(imf_type, remnants, n_mod, nt, nspec, set, cg, zlo, zhi, dz, pm, wsfh, &
                                               it_tesc, nt_neb, whl, fesc, gw, slum, lsum_bar, qbar, wbar, imf_bar, &
                                               fesc_bar, zbar, smass, msum_bar, cbar, ipair, dshift, ctx)
        logical, intent(in)  :: remnants
        integer, intent(in)  :: imf_type, n_mod, nt, nspec, nt_neb, whl
        type(ssp_opset_t), intent(inout) :: set
        real(WP), intent(in), contiguous :: cg(:,:)
        integer, intent(in), contiguous  :: zlo(:), zhi(:), it_tesc(:)
        real(WP), intent(in), contiguous :: dz(:), pm(:,:), wsfh(:,:), fesc(:), slum(:,:), lsum_bar(:), qbar(:,:)
        real(KP), intent(in), contiguous :: gw(:,:)
        real(WP), intent(out), contiguous :: wbar(:,:), imf_bar(:,:), fesc_bar(:), zbar(:), cbar(:,:)
        real(WP), intent(in), contiguous :: smass(:,:), msum_bar(:)
        !> With the TP-AGB shifts: the pair of each model (ssp_mods_prepare_pairs, with the shifted sets for the gradients) and out
        !> the cotangents of delt and dell (2, n_mod): the coefficient of the model's TP-AGB group times the central
        !> differences, in TP_FD_STEP, of the cotangent against the TP-AGB group's outputs at the shifted sets
        integer, intent(in), optional, contiguous :: ipair(:)
        real(WP), intent(out), optional, contiguous :: dshift(:,:)
        !> The context the shifted sets are built from (with ipair and dshift)
        type(opensps_context_t), target, intent(inout), optional :: ctx

        integer :: g, m, t, p, k, ntq, ni, ng, ip, j, row
        real(WP) :: t0, l0
        real(WP), allocatable :: slp_fd(:,:,:), cfd(:,:)
        type(ssp_opset_t) :: fds(1)
        type(ssp_opset_t), pointer :: sg
        real(KP), allocatable :: yg(:,:)
        real(WP), allocatable :: sm_g(:,:), sl_g(:,:), q_g(:,:), sm_zero(:,:), wb_g(:,:), ib_g(:,:), fe_g(:), zb_g(:), &
                                 lfac(:,:,:), slp_l(:,:,:,:), zl(:), cb(:), lzero(:)
        real(WP) :: lzp(2), czp(2), sx, ltot, acc, ac2
        logical, allocatable :: used(:)

        if (.not. allocated(set%slp)) error stop 'ssp_batch_age_sums_adjoint_mods: no forward pass of this batch'
        ! (lsum_bar is built on the device by the caller, whose host copy is current only when the lbol output has a cotangent: the
        ! host formulas below read it)
#ifdef OMP_TARGET
        !$omp target update from(lsum_bar)
#endif
        ntq = size(qbar, 1)
        ni = size(imf_bar, 1)
        ng = size(cg, 1)
        allocate(used(ng))
        do g = 1, ng
            used(g) = any(cg(g, 1:n_mod) /= 0.0_wp)
        end do
        allocate(slp_l, source=set%slp)
        allocate(lfac(nt, n_mod, 2), zl(n_mod), cb(n_mod), lzero(n_mod), sm_zero(nt, n_mod))
        allocate(yg(nspec, 2 * n_mod), sm_g(nt, n_mod), sl_g(nt, n_mod), q_g(ntq, n_mod))
        allocate(wb_g(nt, n_mod), ib_g(ni, n_mod), fe_g(n_mod), zb_g(n_mod))
        lzero = 0.0_wp
        sm_zero = 0.0_wp
        wbar = 0.0_wp
        imf_bar = 0.0_wp
        fesc_bar = 0.0_wp
        zbar = 0.0_wp
        cbar = 0.0_wp

        ! L_bol: the total L(t) of the forward, the point weights' factor, and the SFH weights' initial cotangent
        do m = 1, n_mod
            do t = 1, nt
                czp(1) = 1.0_wp - dz(m)
                czp(2) = dz(m)
                sx = 0.0_wp
                do p = 1, 2
                    lzp(p) = 0.0_wp
                    do g = 1, ng
                        lzp(p) = lzp(p) + cg(g, m) * slp_l(t, m, p, g)
                    end do
                    if (czp(p) /= 0.0_wp) sx = sx + czp(p) * log10(max(lzp(p), SAFE_FLOOR))
                end do
                ltot = 10.0_wp**sx
                do p = 1, 2
                    lfac(t, m, p) = 0.0_wp
                    if (czp(p) /= 0.0_wp .and. lzp(p) > SAFE_FLOOR) lfac(t, m, p) = lsum_bar(m) * ltot * czp(p) / lzp(p)
                end do
                wbar(t, m) = lsum_bar(m) * ltot
            end do
        end do

#ifdef OMP_TARGET
        !$omp target data map(alloc: yg, sm_g, sl_g, q_g) map(to: lfac, slp_l, cg, lsum_bar, msum_bar, qbar, sm_zero, lzero) &
        !$omp& map(from: zl, cb)
#endif
        ! (a group's ssp_batch_age_sums_z leaves its q0 alone without nebular ages: it is read below)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do m = 1, n_mod
            do t = 1, ntq
                q_g(t, m) = 0.0_wp
            end do
        end do
        ! d / d dz of L_bol's blend: the passes the forward ran, as adj_lq's (the log of the groups' sum, once)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(t, p, g, czp, lzp, sx, ltot, acc)
#elif defined(_OPENMP)
        !$omp parallel do private(t, p, g, czp, lzp, sx, ltot, acc)
#endif
        do m = 1, n_mod
            acc = 0.0_wp
            do t = 1, nt
                czp(1) = 1.0_wp - dz(m)
                czp(2) = dz(m)
                sx = 0.0_wp
                do p = 1, 2
                    lzp(p) = 0.0_wp
                    do g = 1, ng
                        lzp(p) = lzp(p) + cg(g, m) * slp_l(t, m, p, g)
                    end do
                    if (czp(p) /= 0.0_wp) sx = sx + czp(p) * log10(max(lzp(p), SAFE_FLOOR))
                end do
                ltot = 10.0_wp**sx
                do p = 1, 2
                    if (czp(p) /= 0.0_wp .and. lzp(p) > SAFE_FLOOR) &
                        acc = acc + merge(-1.0_wp, 1.0_wp, p == 1) * lsum_bar(m) * wsfh(m, t) * ltot * log(10.0_wp) * &
                                    log10(lzp(p))
                end do
            end do
            zl(m) = acc
        end do
        do g = 0, ng - 1
            if (.not. used(g + 1)) cycle
            sg => gset(set, g)
            if (g > 0) then
                ! The group's forward again, for the cotangent of its coefficient
                call ssp_batch_age_sums_z(imf_type, .false., n_mod, nt, nspec, sg, zlo, zhi, dz, pm, wsfh, it_tesc, &
                                          nt_neb, whl, fesc, yg, sm_g, sl_g, q_g)
                call group_cotangent(slp_l(:, :, :, g + 1))
                cbar(g + 1, 1:n_mod) = cb(1:n_mod)
            end if
            if (g == 0) then
                call ssp_batch_age_sums_adjoint(imf_type, remnants, n_mod, nt, nspec, sg, zlo, zhi, dz, pm, wsfh, &
                                                it_tesc, nt_neb, whl, fesc, gw, slum, lzero, qbar, wb_g, ib_g, fe_g, zb_g, &
                                                smass, msum_bar, lfac=lfac)
            else
                call ssp_batch_age_sums_adjoint(imf_type, .false., n_mod, nt, nspec, sg, zlo, zhi, dz, pm, wsfh, &
                                                it_tesc, nt_neb, whl, fesc, gw, slum, lzero, qbar, wb_g, ib_g, fe_g, zb_g, &
                                                sm_zero, msum_bar, lfac=lfac)
            end if
            do m = 1, n_mod
                wbar(:, m) = wbar(:, m) + cg(g + 1, m) * wb_g(:, m)
                imf_bar(:, m) = imf_bar(:, m) + cg(g + 1, m) * ib_g(:, m)
                fesc_bar(m) = fesc_bar(m) + cg(g + 1, m) * fe_g(m)
                zbar(m) = zbar(m) + cg(g + 1, m) * zb_g(m)
            end do
        end do
        ! The TP-AGB shifts: central differences of the cotangent against the TP-AGB group's outputs at the shifted sets
        if (present(dshift) .and. present(ipair) .and. present(ctx)) then
            dshift = 0.0_wp
            allocate(slp_fd(nt, n_mod, 2), cfd(4, n_mod))
#ifdef OMP_TARGET
            !$omp target data map(alloc: slp_fd)
#endif
            do ip = 0, set%n_ext
                if (.not. any(ipair(1:n_mod) == ip)) cycle
                t0 = 0.0_wp
                l0 = 0.0_wp
                if (ip > 0) then
                    t0 = set%ext_t(ip)
                    l0 = set%ext_l(ip)
                end if
                ! (the four shifted sets of this pair, one at a time: the memory of one set, then freed)
                do j = 1, 4
                    call ssp_opset_build_tp(ctx, set, t0 + merge(TP_FD_STEP, 0.0_wp, j == 1) - merge(TP_FD_STEP, 0.0_wp, j == 2), &
                                            l0 + merge(TP_FD_STEP, 0.0_wp, j == 3) - merge(TP_FD_STEP, 0.0_wp, j == 4), fds(1))
                    call ssp_batch_age_sums_z(imf_type, .false., n_mod, nt, nspec, fds(1), zlo, zhi, dz, pm, wsfh, &
                                              it_tesc, nt_neb, whl, fesc, yg, sm_g, sl_g, q_g, slpass=slp_fd)
                    call group_cotangent(slp_fd)
                    cfd(j, 1:n_mod) = cb(1:n_mod)
                    call ssp_opset_free(fds(1))
                end do
                row = 2
                if (ip > 0) row = MOD_NGRP + ip
                do m = 1, n_mod
                    if (ipair(m) /= ip) cycle
                    dshift(1, m) = cg(row, m) * (cfd(1, m) - cfd(2, m)) / (2.0_wp * TP_FD_STEP)
                    dshift(2, m) = cg(row, m) * (cfd(3, m) - cfd(4, m)) / (2.0_wp * TP_FD_STEP)
                end do
            end do
#ifdef OMP_TARGET
            !$omp end target data
#endif
        end if
#ifdef OMP_TARGET
        !$omp end target data
#endif
        zbar = zbar + zl

    contains

        !> cb(m) = the cotangents against the outputs of the group whose forward just ran (yg, sm_g, q_g and the sums sl of its
        !> passes, slp3): the spectra, the surviving mass, the ionizing photon rate and the L_bol sums
        subroutine group_cotangent(slp3)
            real(WP), intent(in), contiguous :: slp3(:,:,:)
            integer :: m, k, t, p
            real(WP) :: acc, ac2
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do private(k, t, p, acc, ac2)
#elif defined(_OPENMP)
            !$omp parallel do private(k, t, p, acc, ac2)
#endif
            do m = 1, n_mod
                acc = 0.0_wp
                do k = 1, nspec
                    acc = acc + real(gw(k, m), WP) * real(yg(k, m), WP) + &
                                real(gw(k, n_mod + m), WP) * real(yg(k, n_mod + m), WP)
                end do
                ac2 = 0.0_wp
                do t = 1, nt
                    ac2 = ac2 + msum_bar(m) * wsfh(m, t) * sm_g(t, m)
                    do p = 1, 2
                        ac2 = ac2 + wsfh(m, t) * lfac(t, m, p) * slp3(t, m, p)
                    end do
                end do
                do t = 1, ntq
                    ac2 = ac2 + qbar(t, m) * q_g(t, m)
                end do
                cb(m) = acc + ac2
            end do
#ifdef OMP_TARGET
            !$omp target update from(cb)
#endif
        end subroutine group_cotangent
    end subroutine ssp_batch_age_sums_adjoint_mods

    !> The IMF rows of Kroupa and Dave analytically: dw/d theta of each point weight w = N / M (N the
    !> bin's number integral, M the IMF's mass integral between its limits) is
    !> (dN/d theta - w dM/d theta) / M, with dN, dM from imf_integral_dp; then, per (piece, model),
    !> the sums of w_bar dw (the template, L_bol and Q cotangents of the points) and of the points'
    !> surviving masses times dw (the mass output's), per row; per model the rows. The remnants' part
    !> of the surviving mass (a scalar per age and model) by central differences, as before.
    subroutine adj_imf_analytic(imf_type, remnants, m0, nc, nt, n_pc_max, n_pts_max, n_pc, pt_off, pc_off, pc_ptr, age_pc, &
                                pt_ptr, pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, pt_first, pt_cm, pt_age, age_mmax, perm, viz, cz, &
                                wsfh, pm, pmp, tot, ldk, wk, ldw, wpb, msum_bar, apt, amp, agt, dtt, remp, remm, imf_bar)
        integer, intent(in) :: imf_type, m0, nc, nt, n_pc_max, n_pts_max, ldk, ldw
        logical, intent(in) :: remnants
        integer, intent(in), contiguous :: n_pc(:), pt_off(:), pc_off(:), pc_ptr(:), age_pc(:), pt_ptr(:), pt_first(:), &
                                           pt_age(:), perm(:), viz(:)
        real(WP), intent(in), contiguous :: pt_mini(:), pt_m1(:), pt_m2(:), pt_llo(:), pt_dl(:), pt_cm(:), age_mmax(:), &
                                            cz(:), wsfh(:,:), pm(:,:), tot(:), msum_bar(:)
        ! (explicit leading dimensions: a point's row is passed by its first element, as imf_pieces_multi)
        real(KP), intent(in) :: wk(ldk, *)
        real(WP), intent(in) :: wpb(ldw, *)
        real(WP), intent(inout), contiguous :: pmp(:,:), apt(:,:), amp(:,:,:), agt(:,:,:), dtt(:,:), remp(:,:), remm(:,:), &
            imf_bar(:,:)
        integer  :: j, q, m, iz, o, k0, n, ig, t, isg, i, npar, q1, q2
        real(WP) :: s, sm, sgn, h, p_apt(NIMFG), p_amp(NIMFG)

        npar = size(pm, 1)
        ! dM / d theta per model of the chunk
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(m, ig)
#elif defined(_OPENMP)
        !$omp parallel do private(m, ig)
#endif
        do j = 1, nc
            m = perm(m0 + j - 1)
            do ig = 1, NIMFG
                dtt(ig, j) = 0.0_wp
                if (imf_row_acts(imf_type, ig)) &
                    dtt(ig, j) = imf_integral_dp(imf_type, pm(:, m), pm(IMF_P_LOWER_LIMIT, m), &
                                                 pm(IMF_P_UPPER_LIMIT, m), .true., imf_row_param(ig))
            end do
        end do

#if defined(OPENSPS_HIP)
        call run_hip_adj_imf_analytic(m0, nc, nt, n_pts_max, size(wsfh, 1), imf_type, ldw, ldk, &
                                      perm, viz, cz, wsfh, pm, tot, dtt, msum_bar, &
                                      pt_off, pt_ptr, pt_age, pt_mini, pt_m1, pt_m2, &
                                      pt_llo, pt_dl, pt_first, pt_cm, wk, wpb, imf_bar)
#else
        ! Per (age, model): compute piece gradients and accumulate into agt (per age) and amp (per age)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(m, iz, o, q1, q2, q, k0, n, ig, p_apt, p_amp)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(m, iz, o, q1, q2, q, k0, n, ig, p_apt, p_amp)
#endif
        do t = 1, nt
            do j = 1, nc
                iz = viz(m0 + j - 1)
                m = perm(m0 + j - 1)
                o = pt_off(iz)
                q1 = age_pc((iz - 1) * (nt + 1) + t)
                q2 = age_pc((iz - 1) * (nt + 1) + t + 1) - 1
                do ig = 1, NIMFG
                    agt(ig, t, j) = 0.0_wp
                    amp(ig, t, j) = 0.0_wp
                end do
                do q = q1, q2
                    k0 = pc_ptr(pc_off(iz) + q)
                    n = pc_ptr(pc_off(iz) + q + 1) - k0
                    call imf_piece_grad(imf_type, pm(:, m), tot(j), dtt(:, j), n, pt_mini(k0:k0 + n - 1), &
                                        pt_m1(k0:k0 + n - 1), pt_m2(k0:k0 + n - 1), pt_llo(k0:k0 + n - 1), &
                                        pt_dl(k0:k0 + n - 1), pt_first(k0:k0 + n - 1), &
                                        pt_cm(k0:k0 + n - 1), wk(j, k0 - o), ldk, wpb(j, k0 - o), ldw, &
                                        p_apt, p_amp)
                    do ig = 1, NIMFG
                        if (imf_row_acts(imf_type, ig)) then
                            agt(ig, t, j) = agt(ig, t, j) + p_apt(ig)
                            if (msum_bar(m) /= 0.0_wp) amp(ig, t, j) = amp(ig, t, j) + p_amp(ig)
                        end if
                    end do
                end do
            end do
        end do
        ! Per model: the rows (the mass output's part over each age's pieces, weighted by the SFH)
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(m, iz, ig, t, s, sm)
#elif defined(_OPENMP)
        !$omp parallel do private(m, iz, ig, t, s, sm)
#endif
        do j = 1, nc
            m = perm(m0 + j - 1)
            iz = viz(m0 + j - 1)
            do ig = 1, NIMFG
                if (.not. imf_row_acts(imf_type, ig)) cycle
                s = 0.0_wp
                do t = 1, nt
                    s = s + agt(ig, t, j)
                end do
                if (msum_bar(m) /= 0.0_wp) then
                    sm = 0.0_wp
                    do t = 1, nt
                        sm = sm + wsfh(m, t) * amp(ig, t, j)
                    end do
                    s = s + msum_bar(m) * cz(m0 + j - 1) * sm
                end if
                imf_bar(ig, m) = imf_bar(ig, m) + s
            end do
        end do
#endif
        ! The remnants' mass (with the mass output's cotangent): central differences per (age, model)
        if (.not. remnants .or. .not. any(msum_bar /= 0.0_wp)) return
        do ig = 1, NIMFG
            if (.not. imf_row_acts(imf_type, ig)) cycle
            do isg = 1, 2
                sgn = merge(1.0_wp, -1.0_wp, isg == 1)
#if defined(OMP_TARGET)
                !$omp target teams distribute parallel do private(i, m)
#elif defined(_OPENMP)
                !$omp parallel do private(i, m)
#endif
                do j = 1, nc
                    m = perm(m0 + j - 1)
                    do i = 1, npar
                        pmp(i, m) = pm(i, m)
                    end do
                    pmp(imf_row_param(ig), m) = pm(imf_row_param(ig), m) + sgn * adj_step(pm(imf_row_param(ig), m))
                end do
#if defined(OMP_TARGET)
                !$omp target teams distribute parallel do collapse(2) private(m, iz, sm)
#elif defined(_OPENMP)
                !$omp parallel do collapse(2) private(m, iz, sm)
#endif
                do j = 1, nc
                    do t = 1, nt
                        m = perm(m0 + j - 1)
                        iz = viz(m0 + j - 1)
                        sm = 0.0_wp
                        if (pt_ptr((iz - 1) * (nt + 1) + t + 1) > pt_ptr((iz - 1) * (nt + 1) + t)) &
                            call add_remnant_mass_p(imf_type, pmp(:, m), sm, age_mmax((iz - 1) * nt + t))
                        if (isg == 1) then
                            remp(t, j) = sm
                        else
                            remm(t, j) = sm
                        end if
                    end do
                end do
            end do
#if defined(OMP_TARGET)
            !$omp target teams distribute parallel do private(m, t, sm, h)
#elif defined(_OPENMP)
            !$omp parallel do private(m, t, sm, h)
#endif
            do j = 1, nc
                m = perm(m0 + j - 1)
                if (msum_bar(m) == 0.0_wp) cycle
                h = adj_step(pm(imf_row_param(ig), m))
                sm = 0.0_wp
                do t = 1, nt
                    sm = sm + wsfh(m, t) * (remp(t, j) - remm(t, j))
                end do
                imf_bar(ig, m) = imf_bar(ig, m) + msum_bar(m) * cz(m0 + j - 1) * sm / (2.0_wp * h)
            end do
        end do
    end subroutine adj_imf_analytic

    !> One piece of adj_imf_analytic: over its points, ps(row) = sum of w_bar dw/d theta and
    !> pms(row) = sum of the surviving mass per star times dw/d theta, for the rows the IMF type reads.
    !> Bins inside one power-law segment take that segment's integral and slope derivative once for
    !> all rows (Kroupa 0.5**(a2 - a1), Dave mdave**(a2 - a1) continuity coefficients); bins across a
    !> break take imf_integral_dp per row.
    subroutine imf_piece_grad(imf_type, p, tt, dtt, n, mini, m1, m2, llo, dl, first, cm, wkp, ldk, wpbp, ldw, ps, pms)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)   :: imf_type, n, ldk, ldw
        real(WP), intent(in)  :: p(IMF_NPAR), tt, dtt(NIMFG)
        real(WP), intent(in)  :: mini(n), m1(n), m2(n), llo(n), dl(n), cm(n)
        integer, intent(in)   :: first(n)
        real(KP), intent(in)  :: wkp(ldk, n)
        real(WP), intent(in)  :: wpbp(ldw, n)
        real(WP), intent(out) :: ps(NIMFG), pms(NIMFG)
        integer  :: k, t, seg, ig
        real(WP) :: lo, hi, a1, a2, a3, md, c2, lc, jv, dj, dv(NIMFG), dw, wkk, inv_t, lnlo, lnr

        t = mod(imf_type, 10)
        do ig = 1, NIMFG
            ps(ig) = 0.0_wp
            pms(ig) = 0.0_wp
        end do
        if (.not. (tt > 0.0_wp)) return
        inv_t = 1.0_wp / tt
        a1 = p(IMF_P_ALPHA)
        a2 = p(IMF_P_ALPHA + 1)
        a3 = p(IMF_P_ALPHA + 2)
        md = p(IMF_P_MDAVE)
        if (t == 2) then
            c2 = 0.5_wp**(-a1 + a2)
            lc = log(0.5_wp)
        else
            c2 = md**(-a1 + a2)
            lc = log(md)
        end if
        do k = 1, n
            if (mini(k) < p(IMF_P_LOWER_LIMIT) .or. mini(k) > p(IMF_P_UPPER_LIMIT)) cycle
            if (first(k) == 1) then
                lo = p(IMF_P_LOWER_BOUND)
            else
                lo = m1(k)
            end if
            hi = m2(k)
            if (hi <= lo) cycle
            ! ln lo, ln(hi / lo): the point's (as imf_bin_weights_tab), or the first bin's own
            if (first(k) == 1) then
                lnlo = log(lo)
                lnr = log(hi / lo)
            else
                lnlo = llo(k)
                lnr = dl(k)
            end if
            do ig = 1, NIMFG
                dv(ig) = 0.0_wp
            end do
            seg = 0
            if (t == 2) then
                if (lo >= 0.08_wp .and. hi <= 0.5_wp) then
                    seg = 1
                else if (lo >= 0.5_wp .and. hi <= 1.0_wp) then
                    seg = 2
                else if (lo >= 1.0_wp) then
                    seg = 3
                end if
            else
                if (lo >= 0.08_wp .and. hi <= md) then
                    seg = 1
                else if (lo >= md) then
                    seg = 2
                end if
            end if
            if (seg == 1) then
                call powerlaw_j_dj(lnlo, lnr, 1.0_wp - a1, jv, dj)
                dv(1) = -dj
            else if (seg == 2 .and. t == 2) then
                call powerlaw_j_dj(lnlo, lnr, 1.0_wp - a2, jv, dj)
                dv(1) = -lc * c2 * jv
                dv(2) = lc * c2 * jv - c2 * dj
            else if (seg == 3) then
                call powerlaw_j_dj(lnlo, lnr, 1.0_wp - a3, jv, dj)
                dv(1) = -lc * c2 * jv
                dv(2) = lc * c2 * jv
                dv(3) = -c2 * dj
            else if (seg == 2) then
                call powerlaw_j_dj(lnlo, lnr, 1.0_wp - a2, jv, dj)
                dv(1) = -lc * c2 * jv
                dv(2) = lc * c2 * jv - c2 * dj
                dv(5) = c2 * (a2 - a1) / md * jv
            else
                dv(1) = imf_integral_dp(imf_type, p, lo, hi, .false., IMF_P_ALPHA)
                dv(2) = imf_integral_dp(imf_type, p, lo, hi, .false., IMF_P_ALPHA + 1)
                if (t == 2) then
                    dv(3) = imf_integral_dp(imf_type, p, lo, hi, .false., IMF_P_ALPHA + 2)
                else
                    dv(5) = imf_integral_dp(imf_type, p, lo, hi, .false., IMF_P_MDAVE)
                end if
            end if
            wkk = real(wkp(1, k), WP)
            do ig = 1, NIMFG
                dw = (dv(ig) - wkk * dtt(ig)) * inv_t
                ps(ig) = ps(ig) + wpbp(1, k) * dw
                pms(ig) = pms(ig) + cm(k) * dw
            end do
        end do
    end subroutine imf_piece_grad

    !> The parameter index (IMF_P_*) of IMF row ig: imf1, imf2, imf3, vdmc, mdave (a function: device
    !> code cannot index a parameter array on LLVM flang)
    pure integer function imf_row_param(ig)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in) :: ig
        select case (ig)
        case (1)
            imf_row_param = IMF_P_ALPHA
        case (2)
            imf_row_param = IMF_P_ALPHA + 1
        case (3)
            imf_row_param = IMF_P_ALPHA + 2
        case (4)
            imf_row_param = IMF_P_VDMC
        case default
            imf_row_param = IMF_P_MDAVE
        end select
    end function imf_row_param

    !> J = integral of m**(e - 1) from lo to hi = lo**e expm1(e L) / e and dJ/de, from ln lo and
    !> L = ln(hi / lo): lo**e once for both; g = expm1(e L) / e and g' = dg/de by series below
    !> |e L| = 1/2 (stopped when the terms no longer change the sums: narrow bins take a few) and in
    !> closed form above. dJ/de = lo**e (ln lo g + g').
    pure subroutine powerlaw_j_dj(lnlo, x, e, jv, dj)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in)  :: lnlo, x, e
        real(WP), intent(out) :: jv, dj
        real(WP) :: el, g, gp, term, u, ex, plo
        integer  :: n

        plo = exp(e * lnlo)
        el = e * x
        if (abs(el) < 0.5_wp) then
            g = x
            term = x
            u = 0.5_wp * x * x
            gp = u
            do n = 2, 24
                term = term * el / n
                if (n >= 3) u = u * el / n
                if (g + term == g .and. gp + (n - 1) * u == gp) exit
                g = g + term
                if (n >= 3) gp = gp + (n - 1) * u
            end do
        else
            ex = exp(el)
            g = (ex - 1.0_wp) / e
            gp = (x * ex - g) / e
        end if
        jv = plo * g
        dj = plo * (lnlo * g + gp)
    end subroutine powerlaw_j_dj

    !> Whether IMF row ig (imf1, imf2, imf3, vdmc, mdave) acts for imf_type (as imf_integral and
    !> imf_bin_weights_tab read them): Kroupa the three slopes, van Dokkum vdmc, Dave imf1, imf2 and
    !> mdave; Salpeter, Chabrier and the user IMF none of them.
    pure logical function imf_row_acts(imf_type, ig)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in) :: imf_type, ig
        select case (mod(imf_type, 10))
        case (2)
            imf_row_acts = ig <= 3
        case (3)
            imf_row_acts = ig == 4
        case (4)
            imf_row_acts = ig == 1 .or. ig == 2 .or. ig == 5
        case default
            imf_row_acts = .false.
        end select
    end function imf_row_acts

    !> The step of the IMF parameter's central difference: relative 1e-6 (the FP64 weights: error
    !> ~1e-10); in FP32 builds 1e-3 (the weights' FP32 rounding, ~1e-7, over the step: ~1e-4)
    pure real(WP) function adj_step(x)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: x
#if defined(KERNEL_SINGLE_PRECISION)
        adj_step = 1.0e-3_wp * max(abs(x), 1.0e-2_wp)
#else
        adj_step = 1.0e-6_wp * max(abs(x), 1.0e-2_wp)
#endif
    end function adj_step

    !> a(:, m) = 0 for the chunk's models m = perm(m0 .. m0 + nc - 1)
    subroutine adj_zero_cols(m0, nc, nt, perm, a)
        integer, intent(in) :: m0, nc, nt
        integer, intent(in), contiguous :: perm(:)
        real(WP), intent(inout) :: a(nt, *)
        integer :: j, t
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do j = 1, nc
            do t = 1, nt
                a(t, perm(m0 + j - 1)) = 0.0_wp
            end do
        end do
    end subroutine adj_zero_cols

    !> The mass output's cotangent through the surviving mass of the chunk's metallicity (sp, sm: per
    !> age and model). mode 0: d / d dz, zbar += zsg msum_bar sum_t wsfh sp; mode 1: the IMF row ig,
    !> imf_bar += msum_bar cz sum_t wsfh (sp - sm) / (2 step) (central differences, as the point weights)
    subroutine adj_mass_dot(m0, nc, nt, perm, cz, zsg, wsfh, msum_bar, sp, sm, mode, row, ig, pm, imf_bar, zbar)
        integer, intent(in) :: m0, nc, nt, mode, row, ig
        integer, intent(in), contiguous :: perm(:)
        real(WP), intent(in), contiguous :: cz(:), zsg(:), wsfh(:,:), msum_bar(:), sp(:,:), sm(:,:)
        real(WP), intent(in), contiguous :: pm(:,:)
        real(WP), intent(inout), contiguous :: imf_bar(:,:), zbar(:)
        integer :: j, t, m
        real(WP) :: s
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(m, t, s)
#elif defined(_OPENMP)
        !$omp parallel do private(m, t, s)
#endif
        do j = 1, nc
            m = perm(m0 + j - 1)
            if (msum_bar(m) == 0.0_wp) cycle
            s = 0.0_wp
            if (mode == 0) then
                do t = 1, nt
                    s = s + wsfh(m, t) * sp(t, m)
                end do
                zbar(m) = zbar(m) + zsg(m0 + j - 1) * msum_bar(m) * s
            else
                do t = 1, nt
                    s = s + wsfh(m, t) * (sp(t, m) - sm(t, m))
                end do
                imf_bar(ig, m) = imf_bar(ig, m) + msum_bar(m) * cz(m0 + j - 1) * s / (2.0_wp * adj_step(pm(row, m)))
            end if
        end do
    end subroutine adj_mass_dot

    !> pmp(row, :) = pm(row, :) +- the step
    subroutine adj_perturb(n_mod, npar, row, sgn, pm, pmp)
        integer, intent(in)     :: n_mod, npar, row
        real(WP), intent(in)    :: sgn, pm(npar, n_mod)
        real(WP), intent(inout) :: pmp(npar, n_mod)
        integer :: m
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do
#elif defined(_OPENMP)
        !$omp parallel do
#endif
        do m = 1, n_mod
            pmp(row, m) = pm(row, m) + sgn * adj_step(pm(row, m))
        end do
    end subroutine adj_perturb

    subroutine adj_zero(nc, npt, a)
        integer, intent(in) :: nc, npt
        real(WP), intent(inout) :: a(:,:)
        integer :: j, k
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2)
#endif
        do k = 1, npt
            do j = 1, nc
                a(j, k) = 0.0_wp
            end do
        end do
    end subroutine adj_zero

    !> slpass(t, perm(m0 + j - 1), ipass) = slz(t, j): the linear L_bol sums of the chunk's virtual models
    subroutine scatter_sl(m0, nc, nt, perm, slz, slpass, ipass)
        integer, intent(in) :: m0, nc, nt, ipass
        integer, intent(in), contiguous :: perm(:)
        real(WP), intent(in), contiguous :: slz(:,:)
        real(WP), intent(inout), contiguous :: slpass(:,:,:)
        integer :: j, t, m

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(m)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(m)
#endif
        do j = 1, nc
            do t = 1, nt
                m = perm(m0 + j - 1)
                slpass(t, m, ipass) = slz(t, j)
            end do
        end do
    end subroutine scatter_sl

    !> Per age and chunk model: sl and sq of this metallicity (the pieces' sums, as imf_pieces_multi)
    subroutine adj_age_sums_sl(m0, nc, nt, age_pc, viz, pps, slz, sqz)
        integer, intent(in) :: m0, nc, nt
        integer, intent(in), contiguous :: age_pc(:), viz(:)
        real(WP), intent(in), contiguous :: pps(:,:,:)
        real(WP), intent(out) :: slz(:,:), sqz(:,:)
        integer :: t, j, q, iz
        real(WP) :: sl, sq
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(q, sl, sq, iz)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(q, sl, sq, iz)
#endif
        do j = 1, nc
            do t = 1, nt
                iz = viz(m0 + j - 1)
                sl = 0.0_wp
                sq = 0.0_wp
                do q = age_pc((iz - 1) * (nt + 1) + t), age_pc((iz - 1) * (nt + 1) + t + 1) - 1
                    sl = sl + pps(j, 2, q)
                    sq = sq + pps(j, 3, q)
                end do
                slz(t, j) = sl
                sqz(t, j) = sq
            end do
        end do
    end subroutine adj_age_sums_sl

    !> v_bar(e, 1:4, j) = T^T of the young, old, young EUV and old EUV cotangents for the run j0..jr
    !> (rows above nwhl for 1, 2; rows 1..nwhl for 3, 4); the young product used templates <= ky.
    !> sum(x(1:n) * y(1:n)) on CPUs: eight partial sums advanced together (a loop the compiler
    !> vectorizes without reassociating: a single running sum cannot be), compensated in FP32 kernels,
    !> the lanes added in FP64. Host code only.
    function dot_lanes(n, x, y) result(s)
        integer, intent(in)  :: n
        real(KP), intent(in) :: x(n), y(n)
        real(WP) :: s
        integer, parameter :: NL = 8
        real(KP) :: a(NL), c(NL), t(NL), u(NL), ra, rc, rv, ru
        integer  :: i, l, n8

        ! (all lanes defined, and the remainder in scalars: with lane-array elements in the remainder
        ! loop, nvfortran 26.1 at -O3 returned NaN whenever n was not a multiple of NL)
        a = 0.0_KP
        c = 0.0_KP
        t = 0.0_KP
        u = 0.0_KP
        n8 = (n / NL) * NL
        if (KP == WP) then
            do i = 1, n8, NL
                a = a + x(i:i + NL - 1) * y(i:i + NL - 1)
            end do
            do i = n8 + 1, n
                a(1) = a(1) + x(i) * y(i)
            end do
        else
            do i = 1, n8, NL
                t = x(i:i + NL - 1) * y(i:i + NL - 1) - c
                u = a + t
                c = (u - a) - t
                a = u
            end do
            ra = a(1)
            rc = c(1)
            do i = n8 + 1, n
                rv = x(i) * y(i) - rc
                ru = ra + rv
                rc = (ru - ra) - rv
                ra = ru
            end do
            a(1) = ra
        end if
        s = 0.0_wp
        do l = 1, NL
            s = s + real(a(l), WP)
        end do
    end function dot_lanes

    subroutine adj_template_bar(nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, perm, etpl, etpl_t, gw, n_eff_max, nc_max, vb)
        integer, intent(in) :: nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, n_eff_max, nc_max
        integer, intent(in), contiguous :: perm(:)
        real(KP), intent(in) :: etpl(nspec, *), etpl_t(n_eff, *), gw(nspec, 2 * n_mod)
        real(WP), intent(inout) :: vb(n_eff_max, 4, nc_max)
        ! Register tile: each thread one template and 4 models (the template's pixels read once for all
        ! of them; models clamped at the end of the run, duplicates written twice with the same value).
        ! Scalar accumulators (an array tile spilled to scratch memory on the R9700), at kernel
        ! precision, compensated in FP32 kernels: s* over the non-EUV pixels, e* over the EUV pixels
        ! (i <= nwhl); *y young, *o old column
        integer  :: e, jj, mm, i
        real(KP) :: ev, term, tsum
        real(KP) :: sy, so, ey, eo
        real(KP) :: csy, cso, cey, ceo

#if !defined(OMP_TARGET)
        ! CPUs: per (template, model) four lane-vectorized dot products (dot_lanes); the template's
        ! column stays in cache across the run's models
        !$omp parallel do private(jj, mm)
        do e = 1, n_eff
            do jj = j0, jr
                mm = perm(m0 + jj - 1)
                if (e > ky) then
                    vb(e, 1, jj) = 0.0_wp
                    vb(e, 3, jj) = 0.0_wp
                else
                    vb(e, 1, jj) = dot_lanes(nspec - nwhl, etpl(nwhl + 1, e), gw(nwhl + 1, mm))
                    vb(e, 3, jj) = dot_lanes(nwhl, etpl(1, e), gw(1, mm))
                end if
                vb(e, 2, jj) = dot_lanes(nspec - nwhl, etpl(nwhl + 1, e), gw(nwhl + 1, n_mod + mm))
                vb(e, 4, jj) = dot_lanes(nwhl, etpl(1, e), gw(1, n_mod + mm))
            end do
        end do
#elif defined(OPENSPS_HIP)
        call run_hip_template_bar(nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, perm, etpl_t, gw, n_eff_max, nc_max, vb)
#else
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(i, mm, ev, term, tsum, sy, so, ey, eo, csy, cso, cey, ceo)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(i, mm, ev, term, tsum, sy, so, ey, eo, csy, cso, cey, ceo)
#endif
        do jj = j0, jr
            do e = 1, n_eff
                mm = perm(m0 + jj - 1)
                ey = 0.0_KP
                eo = 0.0_KP
                sy = 0.0_KP
                so = 0.0_KP
                cey = 0.0_KP
                ceo = 0.0_KP
                csy = 0.0_KP
                cso = 0.0_KP
                do i = 1, nwhl
                    ev = etpl_t(e, i)
                    if (KP == WP) then
                        ey = ey + ev * gw(i, mm)
                        eo = eo + ev * gw(i, n_mod + mm)
                    else
                        term = ev * gw(i, mm) - cey
                        tsum = ey + term
                        cey = (tsum - ey) - term
                        ey = tsum

                        term = ev * gw(i, n_mod + mm) - ceo
                        tsum = eo + term
                        ceo = (tsum - eo) - term
                        eo = tsum
                    end if
                end do
                do i = nwhl + 1, nspec
                    ev = etpl_t(e, i)
                    if (KP == WP) then
                        sy = sy + ev * gw(i, mm)
                        so = so + ev * gw(i, n_mod + mm)
                    else
                        term = ev * gw(i, mm) - csy
                        tsum = sy + term
                        csy = (tsum - sy) - term
                        sy = tsum

                        term = ev * gw(i, n_mod + mm) - cso
                        tsum = so + term
                        cso = (tsum - so) - term
                        so = tsum
                    end if
                end do
                if (e > ky) then
                    vb(e, 1, jj) = 0.0_wp
                    vb(e, 3, jj) = 0.0_wp
                else
                    vb(e, 1, jj) = real(sy, WP)
                    vb(e, 3, jj) = real(ey, WP)
                end if
                vb(e, 2, jj) = real(so, WP)
                vb(e, 4, jj) = real(eo, WP)
            end do
        end do
#endif
    end subroutine adj_template_bar

#if defined(OPENSPS_HIP)
    subroutine run_hip_template_bar(nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, perm, etpl_t, gw, n_eff_max, nc_max, vb)
        use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_null_ptr
        integer, intent(in) :: nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, n_eff_max, nc_max
        integer, intent(in), contiguous, target :: perm(:)
        real(KP), intent(in), target :: etpl_t(n_eff, *), gw(nspec, 2 * n_mod)
        real(WP), intent(inout), target :: vb(n_eff_max, 4, nc_max)

        interface
            subroutine opensps_hip_adj_template_bar_sp(nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, &
                                                       d_perm, d_etpl_t, d_gw, n_eff_max, nc_max, d_vb, stream) &
                bind(C, name="opensps_hip_adj_template_bar_sp")
                import :: c_int, c_ptr
                integer(c_int), value :: nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, n_eff_max, nc_max
                type(c_ptr), value :: d_perm, d_etpl_t, d_gw, d_vb, stream
            end subroutine opensps_hip_adj_template_bar_sp

            subroutine opensps_hip_adj_template_bar_dp(nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, &
                                                       d_perm, d_etpl_t, d_gw, n_eff_max, nc_max, d_vb, stream) &
                bind(C, name="opensps_hip_adj_template_bar_dp")
                import :: c_int, c_ptr
                integer(c_int), value :: nspec, n_eff, ky, nwhl, n_mod, m0, j0, jr, n_eff_max, nc_max
                type(c_ptr), value :: d_perm, d_etpl_t, d_gw, d_vb, stream
            end subroutine opensps_hip_adj_template_bar_dp

            subroutine opensps_hip_sync_stream(stream) bind(C, name="opensps_hip_sync_stream")
                import :: c_ptr
                type(c_ptr), value :: stream
            end subroutine opensps_hip_sync_stream
        end interface

#ifdef OMP_TARGET
        !$omp target data use_device_addr(perm, etpl_t, gw, vb)
#endif
#ifdef KERNEL_SINGLE_PRECISION
        call opensps_hip_adj_template_bar_sp(int(nspec, c_int), int(n_eff, c_int), int(ky, c_int), &
                                             int(nwhl, c_int), int(n_mod, c_int), int(m0, c_int), &
                                             int(j0, c_int), int(jr, c_int), &
                                             c_loc(perm(1)), c_loc(etpl_t(1, 1)), c_loc(gw(1, 1)), &
                                             int(n_eff_max, c_int), int(nc_max, c_int), c_loc(vb(1, 1, 1)), &
                                             c_null_ptr)
#else
        call opensps_hip_adj_template_bar_dp(int(nspec, c_int), int(n_eff, c_int), int(ky, c_int), &
                                             int(nwhl, c_int), int(n_mod, c_int), int(m0, c_int), &
                                             int(j0, c_int), int(jr, c_int), &
                                             c_loc(perm(1)), c_loc(etpl_t(1, 1)), c_loc(gw(1, 1)), &
                                             int(n_eff_max, c_int), int(nc_max, c_int), c_loc(vb(1, 1, 1)), &
                                             c_null_ptr)
#endif
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_hip_template_bar
#endif

    !> The fold in reverse: per template e and chunk model, its segments' ages t: a = sum eff_val w_p,
    !> G = v_bar(young or old) + (f if t <= nt_neb, else 1) v_euv_bar; wbar(t) += cz a G, w_p bar += eff_val cz
    !> wsfh(t) G, fesc_bar += cz a wsfh v_euv_bar, zbar += sign a wsfh G (atomic: templates share ages
    !> and points).
    subroutine adj_fold(m0, nc, n_eff_max, nt_neb, neb_cols, wsfh, it_tesc, fesc, n_eff, pt_off, seg_off, eff_off, &
                        pt_age, eff_pt, eff_val, seg_ptr, seg_first, perm, viz, cz, zsg, wk, vb, wbar, wpb, fesc_bar, zbar)
        integer, intent(in) :: m0, nc, n_eff_max, nt_neb, neb_cols
        real(WP), intent(in), contiguous :: wsfh(:,:), fesc(:), eff_val(:), cz(:), zsg(:), vb(:,:,:)
        real(KP), intent(in), contiguous :: wk(:,:)
        integer, intent(in), contiguous :: it_tesc(:), n_eff(:), pt_off(:), seg_off(:), eff_off(:), pt_age(:), eff_pt(:), &
                                           seg_ptr(:), seg_first(:), perm(:), viz(:)
        real(WP), intent(inout), contiguous :: wbar(:,:), wpb(:,:), fesc_bar(:), zbar(:)
        integer :: e, j, s, k, ks, t, tc, m, iz, eg, sg, kk
        real(WP) :: a, g, ge, x, c, f, xs

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(s, k, ks, t, tc, m, iz, eg, sg, kk, a, g, ge, x, c, &
        !$omp& f, xs)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(s, k, ks, t, tc, m, iz, eg, sg, kk, a, g, ge, x, c, f, xs)
#endif
        do j = 1, nc
            do e = 1, n_eff_max
                iz = viz(m0 + j - 1)
                if (e > n_eff(iz)) cycle
                m = perm(m0 + j - 1)
                c = cz(m0 + j - 1)
                f = fesc(m)
                eg = eff_off(iz) + e
                do s = seg_first(eg), seg_first(eg + 1) - 1
                    sg = seg_off(iz) + s
                    tc = 0
                    a = 0.0_wp
                    ks = seg_ptr(sg)
                    do k = seg_ptr(sg), seg_ptr(sg + 1)
                        if (k < seg_ptr(sg + 1)) then
                            t = pt_age(pt_off(iz) + eff_pt(k))
                        else
                            t = -1
                        end if
                        if (t /= tc) then
                            if (tc > 0) then
                                ! The cotangent of this age's coefficient
                                if (tc <= it_tesc(m)) then
                                    g = vb(e, 1, j)
                                    ge = vb(e, 3, j)
                                else
                                    g = vb(e, 2, j)
                                    ge = vb(e, 4, j)
                                end if
                                ! (the EUV rows take every age, the nebular ones with the escaping fraction)
                                if (neb_cols == 0) ge = 0.0_wp
                                if (tc <= nt_neb) then
                                    x = g + f * ge
                                else
                                    x = g + ge
                                end if
                                !$omp atomic update
                                wbar(tc, m) = wbar(tc, m) + c * a * x
                                !$omp atomic update
                                zbar(m) = zbar(m) + zsg(m0 + j - 1) * a * wsfh(m, tc) * x
                                if (ge /= 0.0_wp .and. tc <= nt_neb) then
                                    !$omp atomic update
                                    fesc_bar(m) = fesc_bar(m) + c * a * wsfh(m, tc) * ge
                                end if
                                xs = c * wsfh(m, tc) * x
                                if (xs /= 0.0_wp) then
                                    do kk = ks, k - 1
                                        !$omp atomic update
                                        wpb(j, eff_pt(kk)) = wpb(j, eff_pt(kk)) + eff_val(kk) * xs
                                    end do
                                end if
                            end if
                            tc = t
                            a = 0.0_wp
                            ks = k
                        end if
                        if (t > 0) a = a + eff_val(k) * real(wk(j, eff_pt(k)), WP)
                    end do
                end do
            end do
        end do
    end subroutine adj_fold

    !> adj_fold as gathers (no atomics: on GPUs the FP64 atomic updates of adj_fold, every template's
    !> thread adding into its model's w_bar, z_bar and fesc_bar, took ~4 s per call of 100 models on
    !> the R9700). With G(p) = sum over point p's template entries of val vb(e, young or old) and
    !> Ge(p) the same with the EUV columns, Gx = G + (fesc for the nebular ages, else 1) Ge:
    !> 1. per (point, model): wpb += cz wsfh(t_p) Gx, and the point weight times Gx and Ge (px, pq);
    !> 2. per (age, model): w_bar(t) += cz sum over the age's points of px (sxa, sea: the sums);
    !> 3. per model: z_bar += zsg sum_t wsfh(t) sxa(t), fesc_bar += cz sum_{t <= nt_neb} wsfh(t) sea(t).
    !> The same sums as adj_fold, reordered (FP64 rounding).
    subroutine adj_fold_gather(m0, nc, nt, nt_neb, neb_cols, n_pts_max, wsfh, it_tesc, fesc, pt_off, pt_ptr, pt_age, &
                               eff_val, pe_ptr, pe_k, pe_e, perm, viz, cz, zsg, wk, vb, wbar, wpb, fesc_bar, zbar, &
                               px, pq, sxa, sea)
        integer, intent(in) :: m0, nc, nt, nt_neb, neb_cols, n_pts_max
        real(WP), intent(in), contiguous :: wsfh(:,:), fesc(:), eff_val(:), cz(:), zsg(:), vb(:,:,:)
        real(KP), intent(in), contiguous :: wk(:,:)
        integer, intent(in), contiguous :: it_tesc(:), pt_off(:), pt_ptr(:), pt_age(:), pe_ptr(:), pe_k(:), pe_e(:), &
                                           perm(:), viz(:)
        real(WP), intent(inout), contiguous :: wbar(:,:), wpb(:,:), fesc_bar(:), zbar(:), px(:,:), pq(:,:), sxa(:,:), &
                                               sea(:,:)
        integer  :: p, j, q, k, e, t, m, iz, gp, np, c1
        real(WP) :: g, ge, gx, fac, c, sx, se

#if defined(OPENSPS_HIP)
        call run_hip_fold_gather(m0, nc, nt, nt_neb, neb_cols, n_pts_max, &
                                 size(vb, 1), size(vb, 3), size(wsfh, 1), &
                                 wsfh, it_tesc, fesc, pt_off, pt_ptr, eff_val, pe_ptr, pe_k, pe_e, &
                                 perm, viz, cz, zsg, wk, vb, wbar, wpb, fesc_bar, zbar)
#else
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(q, k, e, t, m, iz, gp, np, c1, g, ge, gx, fac, c)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(q, k, e, t, m, iz, gp, np, c1, g, ge, gx, fac, c)
#endif
        do p = 1, n_pts_max
            do j = 1, nc
                iz = viz(m0 + j - 1)
                np = pt_ptr((iz - 1) * (nt + 1) + nt + 1) - 1
                px(j, p) = 0.0_wp
                pq(j, p) = 0.0_wp
                if (p > np) cycle
                gp = pt_off(iz) + p
                t = pt_age(gp)
                if (t <= 0) cycle
                m = perm(m0 + j - 1)
                c1 = 2
                if (t <= it_tesc(m0 + j - 1)) c1 = 1
                g = 0.0_wp
                ge = 0.0_wp
                do q = pe_ptr(gp), pe_ptr(gp + 1) - 1
                    k = pe_k(q)
                    e = pe_e(q)
                    g = g + eff_val(k) * vb(e, c1, j)
                    if (neb_cols > 0) ge = ge + eff_val(k) * vb(e, c1 + 2, j)
                end do
                fac = 1.0_wp
                if (t <= nt_neb) fac = fesc(m0 + j - 1)
                gx = g + fac * ge
                c = cz(m0 + j - 1)
                wpb(j, p) = wpb(j, p) + c * wsfh(m, t) * gx
                px(j, p) = real(wk(j, p), WP) * gx
                pq(j, p) = real(wk(j, p), WP) * ge
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(p, m, iz, sx, se)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(p, m, iz, sx, se)
#endif
        do j = 1, nc
            do t = 1, nt
                iz = viz(m0 + j - 1)
                m = perm(m0 + j - 1)
                sx = 0.0_wp
                se = 0.0_wp
                do p = pt_ptr((iz - 1) * (nt + 1) + t), pt_ptr((iz - 1) * (nt + 1) + t + 1) - 1
                    sx = sx + px(j, p)
                    se = se + pq(j, p)
                end do
                wbar(t, m) = wbar(t, m) + cz(m0 + j - 1) * sx
                sxa(t, j) = sx
                sea(t, j) = se
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(t, m, sx, se)
#elif defined(_OPENMP)
        !$omp parallel do private(t, m, sx, se)
#endif
        do j = 1, nc
            m = perm(m0 + j - 1)
            sx = 0.0_wp
            se = 0.0_wp
            do t = 1, nt
                sx = sx + wsfh(m, t) * sxa(t, j)
                if (t <= nt_neb) se = se + wsfh(m, t) * sea(t, j)
            end do
            zbar(m) = zbar(m) + zsg(m0 + j - 1) * sx
            if (neb_cols > 0) fesc_bar(m) = fesc_bar(m) + cz(m0 + j - 1) * se
        end do
#endif
    end subroutine adj_fold_gather

#if defined(OPENSPS_HIP)
    subroutine run_hip_fold_gather(m0, nc, nt, nt_neb, neb_cols, n_pts_max, n_eff_max, nc_max, n_mod, &
                                   wsfh, it_tesc, fesc, pt_off, pt_ptr, eff_val, pe_ptr, pe_k, pe_e, &
                                   perm, viz, cz, zsg, wk, vb, wbar, wpb, fesc_bar, zbar)
        use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_null_ptr
        integer, intent(in) :: m0, nc, nt, nt_neb, neb_cols, n_pts_max, n_eff_max, nc_max, n_mod
        real(WP), intent(in), target, contiguous :: wsfh(n_mod, *), fesc(*), eff_val(*), cz(*), zsg(*), vb(n_eff_max, 4, *)
        real(KP), intent(in), target, contiguous :: wk(nc_max, *)
        integer, intent(in), target, contiguous :: it_tesc(*), pt_off(*), pt_ptr(*), pe_ptr(*), pe_k(*), pe_e(*), &
                                                   perm(*), viz(*)
        real(WP), intent(inout), target, contiguous :: wbar(nt, *), wpb(nc_max, *), fesc_bar(*), zbar(*)

        interface
            subroutine opensps_hip_adj_fold_gather_sp(m0, nc, nt, nt_neb, neb_cols, n_pts_max, n_eff_max, nc_max, n_mod, &
                                                      d_wsfh, d_it_tesc, d_fesc, d_pt_off, d_pt_ptr, d_eff_val, &
                                                      d_pe_ptr, d_pe_k, d_pe_e, d_perm, d_viz, d_cz, d_zsg, &
                                                      d_wk, d_vb, d_wbar, d_wpb, d_fesc_bar, d_zbar, stream) &
                bind(C, name="opensps_hip_adj_fold_gather_sp")
                import :: c_int, c_ptr
                integer(c_int), value :: m0, nc, nt, nt_neb, neb_cols, n_pts_max, n_eff_max, nc_max, n_mod
                type(c_ptr), value :: d_wsfh, d_it_tesc, d_fesc, d_pt_off, d_pt_ptr, d_eff_val
                type(c_ptr), value :: d_pe_ptr, d_pe_k, d_pe_e, d_perm, d_viz, d_cz, d_zsg
                type(c_ptr), value :: d_wk, d_vb, d_wbar, d_wpb, d_fesc_bar, d_zbar, stream
            end subroutine opensps_hip_adj_fold_gather_sp

            subroutine opensps_hip_adj_fold_gather_dp(m0, nc, nt, nt_neb, neb_cols, n_pts_max, n_eff_max, nc_max, n_mod, &
                                                      d_wsfh, d_it_tesc, d_fesc, d_pt_off, d_pt_ptr, d_eff_val, &
                                                      d_pe_ptr, d_pe_k, d_pe_e, d_perm, d_viz, d_cz, d_zsg, &
                                                      d_wk, d_vb, d_wbar, d_wpb, d_fesc_bar, d_zbar, stream) &
                bind(C, name="opensps_hip_adj_fold_gather_dp")
                import :: c_int, c_ptr
                integer(c_int), value :: m0, nc, nt, nt_neb, neb_cols, n_pts_max, n_eff_max, nc_max, n_mod
                type(c_ptr), value :: d_wsfh, d_it_tesc, d_fesc, d_pt_off, d_pt_ptr, d_eff_val
                type(c_ptr), value :: d_pe_ptr, d_pe_k, d_pe_e, d_perm, d_viz, d_cz, d_zsg
                type(c_ptr), value :: d_wk, d_vb, d_wbar, d_wpb, d_fesc_bar, d_zbar, stream
            end subroutine opensps_hip_adj_fold_gather_dp

            subroutine opensps_hip_sync_stream(stream) bind(C, name="opensps_hip_sync_stream")
                import :: c_ptr
                type(c_ptr), value :: stream
            end subroutine opensps_hip_sync_stream
        end interface

#ifdef OMP_TARGET
        !$omp target data use_device_addr(wsfh, it_tesc, fesc, pt_off, pt_ptr, eff_val, pe_ptr, pe_k, pe_e, &
        !$omp& perm, viz, cz, zsg, wk, vb, wbar, wpb, fesc_bar, zbar)
#endif
#ifdef KERNEL_SINGLE_PRECISION
        call opensps_hip_adj_fold_gather_sp(int(m0 - 1, c_int), int(nc, c_int), int(nt, c_int), &
                                            int(nt_neb, c_int), int(neb_cols, c_int), int(n_pts_max, c_int), &
                                            int(n_eff_max, c_int), int(nc_max, c_int), int(n_mod, c_int), &
                                            c_loc(wsfh(1, 1)), c_loc(it_tesc(1)), c_loc(fesc(1)), &
                                            c_loc(pt_off(1)), c_loc(pt_ptr(1)), c_loc(eff_val(1)), &
                                            c_loc(pe_ptr(1)), c_loc(pe_k(1)), c_loc(pe_e(1)), &
                                            c_loc(perm(1)), c_loc(viz(1)), c_loc(cz(1)), c_loc(zsg(1)), &
                                            c_loc(wk(1, 1)), c_loc(vb(1, 1, 1)), c_loc(wbar(1, 1)), &
                                            c_loc(wpb(1, 1)), c_loc(fesc_bar(1)), c_loc(zbar(1)), &
                                            c_null_ptr)
#else
        call opensps_hip_adj_fold_gather_dp(int(m0 - 1, c_int), int(nc, c_int), int(nt, c_int), &
                                            int(nt_neb, c_int), int(neb_cols, c_int), int(n_pts_max, c_int), &
                                            int(n_eff_max, c_int), int(nc_max, c_int), int(n_mod, c_int), &
                                            c_loc(wsfh(1, 1)), c_loc(it_tesc(1)), c_loc(fesc(1)), &
                                            c_loc(pt_off(1)), c_loc(pt_ptr(1)), c_loc(eff_val(1)), &
                                            c_loc(pe_ptr(1)), c_loc(pe_k(1)), c_loc(pe_e(1)), &
                                            c_loc(perm(1)), c_loc(viz(1)), c_loc(cz(1)), c_loc(zsg(1)), &
                                            c_loc(wk(1, 1)), c_loc(vb(1, 1, 1)), c_loc(wbar(1, 1)), &
                                            c_loc(wpb(1, 1)), c_loc(fesc_bar(1)), c_loc(zbar(1)), &
                                            c_null_ptr)
#endif
#ifdef OMP_TARGET
        !$omp end target data
#endif
        call opensps_hip_sync_stream(c_null_ptr)
    end subroutine run_hip_fold_gather

    subroutine run_hip_adj_imf_analytic(m0, nc, nt, n_pts_max, n_mod, imf_type, ldw, ldk, &
                                        perm, viz, cz, wsfh, pm, tot, dtt, msum_bar, &
                                        pt_off, pt_ptr, pt_age, pt_mini, pt_m1, pt_m2, &
                                        pt_llo, pt_dl, pt_first, pt_cm, wk, wpb, imf_bar)
        use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_null_ptr
        integer, intent(in) :: m0, nc, nt, n_pts_max, n_mod, imf_type, ldw, ldk
        integer, intent(in), target, contiguous :: perm(*), viz(*), pt_off(*), pt_ptr(*), pt_age(*), pt_first(*)
        real(WP), intent(in), target, contiguous :: cz(*), wsfh(n_mod, *), pm(312, *), tot(*), dtt(NIMFG, *), &
                                                   msum_bar(*), pt_mini(*), pt_m1(*), pt_m2(*), pt_llo(*), pt_dl(*), &
                                                   pt_cm(*)
        real(KP), intent(in), target, contiguous :: wk(ldk, *)
        real(WP), intent(in), target, contiguous :: wpb(ldw, *)
        real(WP), intent(inout), target, contiguous :: imf_bar(NIMFG, *)

        interface
            subroutine opensps_hip_adj_imf_analytic_sp(m0, nc, nt, n_pts_max, n_mod, imf_type, ldw, ldk, &
                                                       d_perm, d_viz, d_cz, d_wsfh, d_pm, d_tot, d_dtt, d_msum_bar, &
                                                       d_pt_off, d_pt_ptr, d_pt_age, d_pt_mini, d_pt_m1, d_pt_m2, &
                                                       d_pt_llo, d_pt_dl, d_pt_first, d_pt_cm, d_wk, d_wpb, d_imf_bar, &
                                                       stream) bind(C, name="opensps_hip_adj_imf_analytic_sp")
                import :: c_int, c_ptr
                integer(c_int), value :: m0, nc, nt, n_pts_max, n_mod, imf_type, ldw, ldk
                type(c_ptr), value :: d_perm, d_viz, d_cz, d_wsfh, d_pm, d_tot, d_dtt, d_msum_bar
                type(c_ptr), value :: d_pt_off, d_pt_ptr, d_pt_age, d_pt_mini, d_pt_m1, d_pt_m2
                type(c_ptr), value :: d_pt_llo, d_pt_dl, d_pt_first, d_pt_cm, d_wk, d_wpb, d_imf_bar, stream
            end subroutine opensps_hip_adj_imf_analytic_sp

            subroutine opensps_hip_adj_imf_analytic_dp(m0, nc, nt, n_pts_max, n_mod, imf_type, ldw, ldk, &
                                                       d_perm, d_viz, d_cz, d_wsfh, d_pm, d_tot, d_dtt, d_msum_bar, &
                                                       d_pt_off, d_pt_ptr, d_pt_age, d_pt_mini, d_pt_m1, d_pt_m2, &
                                                       d_pt_llo, d_pt_dl, d_pt_first, d_pt_cm, d_wk, d_wpb, d_imf_bar, &
                                                       stream) bind(C, name="opensps_hip_adj_imf_analytic_dp")
                import :: c_int, c_ptr
                integer(c_int), value :: m0, nc, nt, n_pts_max, n_mod, imf_type, ldw, ldk
                type(c_ptr), value :: d_perm, d_viz, d_cz, d_wsfh, d_pm, d_tot, d_dtt, d_msum_bar
                type(c_ptr), value :: d_pt_off, d_pt_ptr, d_pt_age, d_pt_mini, d_pt_m1, d_pt_m2
                type(c_ptr), value :: d_pt_llo, d_pt_dl, d_pt_first, d_pt_cm, d_wk, d_wpb, d_imf_bar, stream
            end subroutine opensps_hip_adj_imf_analytic_dp
        end interface

#ifdef OMP_TARGET
        !$omp target data use_device_addr(perm, viz, cz, wsfh, pm, tot, dtt, msum_bar, &
        !$omp& pt_off, pt_ptr, pt_age, pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, pt_first, pt_cm, &
        !$omp& wk, wpb, imf_bar)
#endif
#ifdef KERNEL_SINGLE_PRECISION
        call opensps_hip_adj_imf_analytic_sp(int(m0 - 1, c_int), int(nc, c_int), int(nt, c_int), &
                                             int(n_pts_max, c_int), int(n_mod, c_int), int(imf_type, c_int), &
                                             int(ldw, c_int), int(ldk, c_int), &
                                             c_loc(perm(1)), c_loc(viz(1)), c_loc(cz(1)), &
                                             c_loc(wsfh(1, 1)), c_loc(pm(1, 1)), c_loc(tot(1)), &
                                             c_loc(dtt(1, 1)), c_loc(msum_bar(1)), &
                                             c_loc(pt_off(1)), c_loc(pt_ptr(1)), c_loc(pt_age(1)), &
                                             c_loc(pt_mini(1)), c_loc(pt_m1(1)), c_loc(pt_m2(1)), &
                                             c_loc(pt_llo(1)), c_loc(pt_dl(1)), c_loc(pt_first(1)), &
                                             c_loc(pt_cm(1)), c_loc(wk(1, 1)), c_loc(wpb(1, 1)), &
                                             c_loc(imf_bar(1, 1)), c_null_ptr)
#else
        call opensps_hip_adj_imf_analytic_dp(int(m0 - 1, c_int), int(nc, c_int), int(nt, c_int), &
                                             int(n_pts_max, c_int), int(n_mod, c_int), int(imf_type, c_int), &
                                             int(ldw, c_int), int(ldk, c_int), &
                                             c_loc(perm(1)), c_loc(viz(1)), c_loc(cz(1)), &
                                             c_loc(wsfh(1, 1)), c_loc(pm(1, 1)), c_loc(tot(1)), &
                                             c_loc(dtt(1, 1)), c_loc(msum_bar(1)), &
                                             c_loc(pt_off(1)), c_loc(pt_ptr(1)), c_loc(pt_age(1)), &
                                             c_loc(pt_mini(1)), c_loc(pt_m1(1)), c_loc(pt_m2(1)), &
                                             c_loc(pt_llo(1)), c_loc(pt_dl(1)), c_loc(pt_first(1)), &
                                             c_loc(pt_cm(1)), c_loc(wk(1, 1)), c_loc(wpb(1, 1)), &
                                             c_loc(imf_bar(1, 1)), c_null_ptr)
#endif
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_hip_adj_imf_analytic

    subroutine run_hip_adj_lq(m0, nc, nt, nt_neb, ntq, n_pts_max, n_mod, ldw, &
                              perm, viz, cz, zsg, pt_off, pt_ptr, pt_age, pt_cl, pt_q, &
                              wsfh, slum, lsum_bar, qbar, slz, sqz, wpb, zbar)
        use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_null_ptr
        integer, intent(in) :: m0, nc, nt, nt_neb, ntq, n_pts_max, n_mod, ldw
        integer, intent(in), target, contiguous :: perm(*), viz(*), pt_off(*), pt_ptr(*), pt_age(*)
        real(WP), intent(in), target, contiguous :: cz(*), zsg(*), pt_cl(*), pt_q(*), &
                                                   wsfh(n_mod, *), slum(nt, *), lsum_bar(*), &
                                                   qbar(ntq, *), slz(nt, *), sqz(nt, *)
        real(WP), intent(inout), target, contiguous :: wpb(ldw, *), zbar(*)

        interface
            subroutine opensps_hip_adj_lq(m0, nc, nt, nt_neb, ntq, n_pts_max, n_mod, ldw, &
                                          d_perm, d_viz, d_cz, d_zsg, d_pt_off, d_pt_ptr, d_pt_age, &
                                          d_pt_cl, d_pt_q, d_wsfh, d_slum, d_lsum_bar, d_qbar, &
                                          d_slz, d_sqz, d_wpb, d_zbar, stream) bind(C, name="opensps_hip_adj_lq")
                import :: c_int, c_ptr
                integer(c_int), value :: m0, nc, nt, nt_neb, ntq, n_pts_max, n_mod, ldw
                type(c_ptr), value :: d_perm, d_viz, d_cz, d_zsg, d_pt_off, d_pt_ptr, d_pt_age
                type(c_ptr), value :: d_pt_cl, d_pt_q, d_wsfh, d_slum, d_lsum_bar, d_qbar
                type(c_ptr), value :: d_slz, d_sqz, d_wpb, d_zbar, stream
            end subroutine opensps_hip_adj_lq
        end interface

#ifdef OMP_TARGET
        !$omp target data use_device_addr(perm, viz, cz, zsg, pt_off, pt_ptr, pt_age, pt_cl, pt_q, &
        !$omp& wsfh, slum, lsum_bar, qbar, slz, sqz, wpb, zbar)
#endif
        call opensps_hip_adj_lq(int(m0 - 1, c_int), int(nc, c_int), int(nt, c_int), &
                                int(nt_neb, c_int), int(ntq, c_int), int(n_pts_max, c_int), &
                                int(n_mod, c_int), int(ldw, c_int), &
                                c_loc(perm(1)), c_loc(viz(1)), c_loc(cz(1)), c_loc(zsg(1)), &
                                c_loc(pt_off(1)), c_loc(pt_ptr(1)), c_loc(pt_age(1)), &
                                c_loc(pt_cl(1)), c_loc(pt_q(1)), &
                                c_loc(wsfh(1, 1)), c_loc(slum(1, 1)), c_loc(lsum_bar(1)), &
                                c_loc(qbar(1, 1)), c_loc(slz(1, 1)), c_loc(sqz(1, 1)), &
                                c_loc(wpb(1, 1)), c_loc(zbar(1)), c_null_ptr)
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_hip_adj_lq

    !> The HIP fold of the weights into the segments' sums vp (the first loop of fold_multi)
    subroutine run_hip_fold_multi(m0, nc, nt_neb, n_seg_max, wsfh, it_tesc, fesc, n_seg, pt_off, seg_off, pt_age, eff_pt, &
                                  seg_ptr, perm, viz, cz, w, vp, eff_f)
        use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_null_ptr
        integer, intent(in) :: m0, nc, nt_neb, n_seg_max
        real(WP), intent(in), target, contiguous :: wsfh(:,:), fesc(:), cz(:)
        integer, intent(in), target, contiguous :: it_tesc(:), n_seg(:), pt_off(:), seg_off(:), pt_age(:), eff_pt(:), &
                                                   seg_ptr(:), perm(:), viz(:)
        real(KP), intent(in), target, contiguous :: w(:,:)
        real(WP), intent(inout), target, contiguous :: vp(:,:,:)
        real(c_float), intent(in), target, contiguous :: eff_f(:,:)
        interface
            subroutine opensps_hip_fold_multi(m0, nc, nt_neb, ldw, n_mod_ld, ldv, nseg_max, jb, perm, viz, n_seg, pt_off, &
                seg_off, pt_age, eff_pt, seg_ptr, eff_f, wsfh, it_tesc, fesc, cz, w, vp, stream) &
                bind(C, name="opensps_hip_fold_multi")
                import :: c_int, c_ptr
                integer(c_int), value :: m0, nc, nt_neb, ldw, n_mod_ld, ldv, nseg_max, jb
                type(c_ptr), value :: perm, viz, n_seg, pt_off, seg_off, pt_age, eff_pt, seg_ptr, eff_f, wsfh, it_tesc, &
                                      fesc, cz, w, vp, stream
            end subroutine opensps_hip_fold_multi
        end interface
        ! fesc is a host array of the batch: mapped for the call
        !$omp target data map(to: fesc) use_device_addr(perm, viz, n_seg, pt_off, seg_off, pt_age, eff_pt, seg_ptr, eff_f, &
        !$omp& wsfh, it_tesc, cz, w, vp)
        !$omp target data use_device_addr(fesc)
        call opensps_hip_fold_multi(int(m0 - 1, c_int), int(nc, c_int), int(nt_neb, c_int), int(size(w, 1), c_int), &
                                    int(size(wsfh, 1), c_int), int(size(vp, 1), c_int), int(n_seg_max, c_int), &
                                    int(TUNE_HIP_FOLD_JB, c_int), &
                                    c_loc(perm(1)), c_loc(viz(1)), c_loc(n_seg(1)), c_loc(pt_off(1)), c_loc(seg_off(1)), &
                                    c_loc(pt_age(1)), c_loc(eff_pt(1)), c_loc(seg_ptr(1)), c_loc(eff_f(1, 1)), &
                                    c_loc(wsfh(1, 1)), c_loc(it_tesc(1)), c_loc(fesc(1)), c_loc(cz(1)), c_loc(w(1, 1)), &
                                    c_loc(vp(1, 1, 1)), c_null_ptr)
        !$omp end target data
        !$omp end target data
    end subroutine run_hip_fold_multi

    !> The HIP template products (FP32 kernels): false if the block size KB is not one the kernel is built for
    function run_hip_template_columns(np, k, nc, etpl, v, v0, yo, c0) result(done)
        use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_null_ptr
        integer, intent(in) :: np, k, nc, v0, c0
        real(KP), intent(in), target, contiguous :: etpl(:,:), v(:,:)
        real(KP), intent(inout), target, contiguous :: yo(:,:)
        logical :: done
        integer(c_int) :: rc
        interface
            function opensps_hip_template_columns(np, nc, k, rm, rn, kc, kb, lda, ldb, ldc, a, b, c, stream) result(rc) &
                bind(C, name="opensps_hip_template_columns")
                import :: c_int, c_ptr
                integer(c_int), value :: np, nc, k, rm, rn, kc, kb, lda, ldb, ldc
                type(c_ptr), value :: a, b, c, stream
                integer(c_int) :: rc
            end function opensps_hip_template_columns
        end interface
        !$omp target data use_device_addr(etpl, v, yo)
        rc = opensps_hip_template_columns(int(np, c_int), int(nc, c_int), int(k, c_int), int(TUNE_HIP_GEMM_RM, c_int), &
                                          int(TUNE_HIP_GEMM_RN, c_int), int(TUNE_HIP_GEMM_KC, c_int), int(TUNE_TEMPLATE_KB, &
                                          c_int), &
                                          int(size(etpl, 1), c_int), int(size(v, 1), c_int), int(size(yo, 1), c_int), &
                                          c_loc(etpl(1, 1)), c_loc(v(1, v0 + 1)), c_loc(yo(1, c0 + 1)), c_null_ptr)
        !$omp end target data
        done = rc == 0
    end function run_hip_template_columns

    subroutine run_hip_imf_pieces_weights(m0, nc, n_pc_max, n_pts_max, ldw, imf_type, &
                                          perm, viz, pm, tot, n_pc, pt_off, pc_off, pc_ptr, &
                                          pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, pt_first, &
                                          pt_cm, pt_cl, pt_q, w, wk, pps, store_w, pt_f)
        use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_loc, c_null_ptr
        integer, intent(in) :: m0, nc, n_pc_max, n_pts_max, ldw, imf_type
        !> False: the FP64 weights w are not written (a forward-only call: only wk, the weights the fold reads)
        logical, intent(in) :: store_w
        !> The set's FP32 table (needed when store_w is false: the FP32-arithmetic kernel)
        real(c_float), intent(in), target, contiguous, optional :: pt_f(:,:)
        integer, intent(in), target, contiguous :: perm(*), viz(*), n_pc(*), pt_off(*), pc_off(*), pc_ptr(*), pt_first(*)
        real(WP), intent(in), target, contiguous :: pm(312, *), tot(*), pt_mini(*), pt_m1(*), pt_m2(*), pt_llo(*), pt_dl(*), &
                                                    pt_cm(*), pt_cl(*), pt_q(*)
        real(WP), intent(inout), target, contiguous :: w(ldw, *), pps(ldw, 3, *)
        real(KP), intent(inout), target, contiguous :: wk(ldw, *)
        type(c_ptr) :: pw, pf

        interface
            subroutine opensps_hip_imf_pieces_weights_sp(m0, nc, n_pc_max, n_pts_max, ldw, imf_type, pieces, &
                perm, viz, pm, tot, n_pc, pt_off, pc_off, pc_ptr, pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, &
                pt_first, pt_cm, pt_cl, pt_q, w, wk, pps, ptf, stream) bind(C, name="opensps_hip_imf_pieces_weights_sp")
                import :: c_int, c_ptr
                integer(c_int), value :: m0, nc, n_pc_max, n_pts_max, ldw, imf_type, pieces
                type(c_ptr), value :: perm, viz, pm, tot, n_pc, pt_off, pc_off, pc_ptr, pt_mini, pt_m1, pt_m2, &
                                      pt_llo, pt_dl, pt_first, pt_cm, pt_cl, pt_q, w, wk, pps, ptf, stream
            end subroutine opensps_hip_imf_pieces_weights_sp
            subroutine opensps_hip_imf_pieces_weights_dp(m0, nc, n_pc_max, n_pts_max, ldw, imf_type, pieces, &
                perm, viz, pm, tot, n_pc, pt_off, pc_off, pc_ptr, pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, &
                pt_first, pt_cm, pt_cl, pt_q, w, wk, pps, stream) bind(C, name="opensps_hip_imf_pieces_weights_dp")
                import :: c_int, c_ptr
                integer(c_int), value :: m0, nc, n_pc_max, n_pts_max, ldw, imf_type, pieces
                type(c_ptr), value :: perm, viz, pm, tot, n_pc, pt_off, pc_off, pc_ptr, pt_mini, pt_m1, pt_m2, &
                                      pt_llo, pt_dl, pt_first, pt_cm, pt_cl, pt_q, w, wk, pps, stream
            end subroutine opensps_hip_imf_pieces_weights_dp
        end interface

#ifdef OMP_TARGET
        !$omp target data use_device_addr(perm, viz, pm, tot, n_pc, pt_off, pc_off, pc_ptr, &
        !$omp& pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, pt_first, pt_cm, pt_cl, pt_q, w, wk, pps)
#endif
        pw = c_null_ptr
        if (store_w) pw = c_loc(w(1, 1))
        pf = c_null_ptr
        if (present(pt_f)) then
#ifdef OMP_TARGET
            !$omp target data use_device_addr(pt_f)
#endif
            pf = c_loc(pt_f(1, 1))
#ifdef OMP_TARGET
            !$omp end target data
#endif
        end if
#ifdef KERNEL_SINGLE_PRECISION
        call opensps_hip_imf_pieces_weights_sp(int(m0 - 1, c_int), int(nc, c_int), int(n_pc_max, c_int), &
                                               int(n_pts_max, c_int), int(ldw, c_int), int(imf_type, c_int), &
                                               int(TUNE_HIP_WEIGHTS_PIECES, c_int), &
                                               c_loc(perm(1)), c_loc(viz(1)), c_loc(pm(1, 1)), c_loc(tot(1)), &
                                               c_loc(n_pc(1)), c_loc(pt_off(1)), c_loc(pc_off(1)), c_loc(pc_ptr(1)), &
                                               c_loc(pt_mini(1)), c_loc(pt_m1(1)), c_loc(pt_m2(1)), c_loc(pt_llo(1)), &
                                               c_loc(pt_dl(1)), c_loc(pt_first(1)), c_loc(pt_cm(1)), c_loc(pt_cl(1)), &
                                               c_loc(pt_q(1)), pw, c_loc(wk(1, 1)), c_loc(pps(1, 1, 1)), &
                                               pf, c_null_ptr)
#else
        call opensps_hip_imf_pieces_weights_dp(int(m0 - 1, c_int), int(nc, c_int), int(n_pc_max, c_int), &
                                               int(n_pts_max, c_int), int(ldw, c_int), int(imf_type, c_int), &
                                               int(TUNE_HIP_WEIGHTS_PIECES, c_int), &
                                               c_loc(perm(1)), c_loc(viz(1)), c_loc(pm(1, 1)), c_loc(tot(1)), &
                                               c_loc(n_pc(1)), c_loc(pt_off(1)), c_loc(pc_off(1)), c_loc(pc_ptr(1)), &
                                               c_loc(pt_mini(1)), c_loc(pt_m1(1)), c_loc(pt_m2(1)), c_loc(pt_llo(1)), &
                                               c_loc(pt_dl(1)), c_loc(pt_first(1)), c_loc(pt_cm(1)), c_loc(pt_cl(1)), &
                                               c_loc(pt_q(1)), pw, c_loc(wk(1, 1)), c_loc(pps(1, 1, 1)), &
                                               c_null_ptr)
#endif
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_hip_imf_pieces_weights
#endif

    !> L_bol and Q per point: w_p bar += lsum_bar wsfh(t) L(t) cz cl_p / sl_z + qbar(t) cz q_p; and their
    !> d / d dz (zbar: L in log, Q linearly).
    subroutine adj_lq(m0, nc, nt, nt_neb, n_pc_max, n_pc, pt_off, pc_off, pc_ptr, pt_ptr, pt_age, pt_cl, pt_q, perm, viz, cz, &
                      zsg, wsfh, slum, lsum_bar, qbar, ntq, slz, sqz, wpb, zbar, lfx, use_lfx)
        integer, intent(in) :: m0, nc, nt, nt_neb, n_pc_max, ntq
        integer, intent(in), contiguous :: n_pc(:), pt_off(:), pc_off(:), pc_ptr(:), pt_ptr(:), pt_age(:), perm(:), viz(:)
        real(WP), intent(in), contiguous :: pt_cl(:), pt_q(:), cz(:), zsg(:), wsfh(:,:), slum(:,:), lsum_bar(:), &
                                            qbar(:,:), slz(:,:), sqz(:,:)
        real(WP), intent(inout), contiguous :: wpb(:,:), zbar(:)
        !> use_lfx: the factor of cl_p is wsfh(m, t) lfx(t, m) (the L_bol of a batch with free modifications, see
        !> ssp_batch_age_sums_adjoint_mods) in place of lsum_bar wsfh L cz / sl; the zbar term then comes from lsum_bar
        !> alone (which that caller zeroes). The accelerated kernel does not take it
        real(WP), intent(in) :: lfx(:,:)
        logical, intent(in) :: use_lfx
        integer :: q, j, k, k0, n, iz, m, o, t
        real(WP) :: lf, qf, zs
#if defined(OPENSPS_HIP)
        if (.not. use_lfx) then
            call run_hip_adj_lq(m0, nc, nt, nt_neb, ntq, size(wpb, 2), size(wsfh, 1), size(wpb, 1), &
                                perm, viz, cz, zsg, pt_off, pt_ptr, pt_age, pt_cl, pt_q, &
                                wsfh, slum, lsum_bar, qbar, slz, sqz, wpb, zbar)
            return
        end if
#endif
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, k0, n, iz, m, o, t, lf, qf)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, k0, n, iz, m, o, t, lf, qf)
#endif
        do j = 1, nc
            do q = 1, n_pc_max
                iz = viz(m0 + j - 1)
                if (q > n_pc(iz)) cycle
                m = perm(m0 + j - 1)
                o = pt_off(iz)
                k0 = pc_ptr(pc_off(iz) + q)
                n = pc_ptr(pc_off(iz) + q + 1) - k0
                do k = k0, k0 + n - 1
                    t = pt_age(k)
                    lf = 0.0_wp
                    if (use_lfx) then
                        lf = wsfh(m, t) * lfx(t, m)
                    else if (slz(t, j) > SAFE_FLOOR) then
                        lf = lsum_bar(m) * wsfh(m, t) * slum(t, m) * cz(m0 + j - 1) / slz(t, j)
                    end if
                    qf = 0.0_wp
                    if (t <= nt_neb .and. t <= ntq) qf = qbar(t, m) * cz(m0 + j - 1)
                    wpb(j, k - o) = wpb(j, k - o) + lf * pt_cl(k) + qf * pt_q(k)
                end do
            end do
        end do
        ! d / d dz: L(t) = 10^(sum cz log10 sl), Q = sum cz sq
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(t, m, zs)
#elif defined(_OPENMP)
        !$omp parallel do private(t, m, zs)
#endif
        do j = 1, nc
            m = perm(m0 + j - 1)
            zs = 0.0_wp
            do t = 1, nt
                if (slz(t, j) > SAFE_FLOOR) zs = zs + lsum_bar(m) * wsfh(m, t) * slum(t, m) * log(10.0_wp) * &
                                                      log10(slz(t, j))
                if (t <= nt_neb .and. t <= ntq) zs = zs + qbar(t, m) * sqz(t, j)
            end do
            zbar(m) = zbar(m) + zsg(m0 + j - 1) * zs
        end do
    end subroutine adj_lq

    !> imf_bar(ig, m) += sum_p w_p bar (w+ - w-) / 2h (per piece, then per model)
    subroutine adj_imf_dot(m0, nc, n_pc_max, n_pc, pt_off, pc_off, pc_ptr, perm, viz, npar, row, pm, wplus, wm, wpb, part, &
                           ig, imf_bar)
        integer, intent(in) :: m0, nc, n_pc_max, npar, row, ig
        integer, intent(in), contiguous :: n_pc(:), pt_off(:), pc_off(:), pc_ptr(:), perm(:), viz(:)
        real(WP), intent(in) :: pm(npar, *)
        real(WP), intent(in), contiguous :: wplus(:,:), wm(:,:), wpb(:,:)
        real(WP), intent(inout), contiguous :: part(:,:), imf_bar(:,:)
        integer :: q, j, k, k0, n, iz, o, m
        real(WP) :: s
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, k0, n, iz, o, s)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, k0, n, iz, o, s)
#endif
        ! (pieces outer, models inner: consecutive threads read consecutive models of one point)
        do q = 1, n_pc_max
            do j = 1, nc
                iz = viz(m0 + j - 1)
                s = 0.0_wp
                if (q <= n_pc(iz)) then
                    o = pt_off(iz)
                    k0 = pc_ptr(pc_off(iz) + q)
                    n = pc_ptr(pc_off(iz) + q + 1) - k0
                    do k = k0 - o, k0 - o + n - 1
                        s = s + wpb(j, k) * (wplus(j, k) - wm(j, k))
                    end do
                end if
                part(q, j) = s
            end do
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(q, m, s)
#elif defined(_OPENMP)
        !$omp parallel do private(q, m, s)
#endif
        do j = 1, nc
            m = perm(m0 + j - 1)
            s = 0.0_wp
            do q = 1, n_pc_max
                s = s + part(q, j)
            end do
            imf_bar(ig, m) = imf_bar(ig, m) + s / (2.0_wp * adj_step(pm(row, m)))
        end do
    end subroutine adj_imf_dot

    !> imf_pieces_chunk for the virtual models m0 .. m0 + nc - 1 of a pass (model perm(v), metallicity
    !> viz(v)) with the tables of an ssp_opset_t: the thread (piece q, virtual model) takes the piece
    !> and the points of its metallicity's table (offsets pt_off, pc_off) and skips it if the
    !> metallicity has fewer pieces; the sums accumulate (cz(m)) into smass, slum (logarithm) and q0.
    subroutine imf_pieces_multi(imf_type, remnants, m0, nc, nt, n_pc_max, nt_neb, pm, n_pc, pt_off, pc_off, pc_ptr, &
                                age_pc, pt_ptr, pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, pt_first, pt_cm, pt_cl, pt_q, age_mmax, ldw, &
                                n_pts_max, w, wk, tot, pps, perm, viz, cz, smass, slum, q0, store_w, pt_f)
        logical, intent(in) :: remnants
        integer, intent(in) :: imf_type, m0, nc, nt, n_pc_max, nt_neb, ldw, n_pts_max
        real(WP), intent(in), contiguous :: pm(:,:), pt_mini(:), pt_m1(:), pt_m2(:), pt_llo(:), pt_dl(:), pt_cm(:), &
                                            pt_cl(:), pt_q(:), &
                                            age_mmax(:), cz(:)
        integer, intent(in), contiguous  :: n_pc(:), pt_off(:), pc_off(:), pc_ptr(:), age_pc(:), pt_ptr(:), pt_first(:), &
                                            perm(:), viz(:)
        real(WP), intent(inout) :: w(ldw, n_pts_max)
        real(KP), intent(inout) :: wk(ldw, n_pts_max)
        real(WP), intent(inout), contiguous :: tot(:), pps(:,:,:), smass(:,:), slum(:,:), q0(:,:)
        !> False: the forward-only call, which reads only wk; the HIP weights kernel then skips the FP64 array w
        !> (w may be a dummy 1 x 1 array). Default true: the reverse pass needs w.
        logical, intent(in), optional :: store_w
        !> The FP32 table of the set (ssp_opset_t%pt_f), read by the weights kernel of a forward-only call
        real(c_float), intent(in), optional, contiguous :: pt_f(:,:)
        integer  :: j, t, k, m, q, k0, n, iz, o, kl
        real(WP) :: sm, sl, sq
        logical  :: keep_w

        keep_w = .true.
        if (present(store_w)) keep_w = store_w
        if (.not. present(pt_f)) keep_w = .true.

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do 
#elif defined(_OPENMP)
        !$omp parallel do 
#endif
        do j = 1, nc
            tot(j) = imf_integral_p(imf_type, pm(:, perm(m0 + j - 1)), pm(IMF_P_LOWER_LIMIT, perm(m0 + j - 1)), &
                                    pm(IMF_P_UPPER_LIMIT, perm(m0 + j - 1)), .true.)
        end do
#if defined(OPENSPS_HIP) && defined(OMP_TARGET)
        if (mod(imf_type, 10) == 0 .or. mod(imf_type, 10) == 2 .or. mod(imf_type, 10) == 4) then
            if (keep_w) then
                call run_hip_imf_pieces_weights(m0, nc, n_pc_max, n_pts_max, ldw, imf_type, &
                                                perm, viz, pm, tot, n_pc, pt_off, pc_off, pc_ptr, &
                                                pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, pt_first, &
                                                pt_cm, pt_cl, pt_q, w, wk, pps, .true.)
            else
                call run_hip_imf_pieces_weights(m0, nc, n_pc_max, n_pts_max, ldw, imf_type, &
                                                perm, viz, pm, tot, n_pc, pt_off, pc_off, pc_ptr, &
                                                pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, pt_first, &
                                                pt_cm, pt_cl, pt_q, w, wk, pps, .false., pt_f)
            end if
        else
            !$omp target teams distribute parallel do collapse(2) private(k, m, k0, n, sm, sl, sq, iz, o, kl)
            do q = 1, n_pc_max
                do j = 1, nc
                    iz = viz(m0 + j - 1)
                    if (q > n_pc(iz)) cycle
                    m = perm(m0 + j - 1)
                    o = pt_off(iz)
                    k0 = pc_ptr(pc_off(iz) + q)
                    n = pc_ptr(pc_off(iz) + q + 1) - k0
                    sm = 0.0_wp
                    sl = 0.0_wp
                    sq = 0.0_wp
                    call imf_bin_weights_tab(imf_type, pm(:, m), tot(j), n, pt_mini(k0), &
                                         pt_m1(k0), pt_m2(k0), pt_llo(k0), &
                                         pt_dl(k0), pt_first(k0), &
                                         w(j, k0 - o), ldw)
                    do k = k0, k0 + n - 1
                        kl = k - o
                        sm = sm + pt_cm(k) * w(j, kl)
                        sl = sl + pt_cl(k) * w(j, kl)
                        sq = sq + pt_q(k) * w(j, kl)
                        wk(j, kl) = real(w(j, kl), KP)
                    end do
                    pps(j, 1, q) = sm
                    pps(j, 2, q) = sl
                    pps(j, 3, q) = sq
                end do
            end do
        end if
#elif defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, m, k0, n, sm, sl, sq, iz, o, kl)
        do q = 1, n_pc_max
            do j = 1, nc
                iz = viz(m0 + j - 1)
                if (q > n_pc(iz)) cycle
                m = perm(m0 + j - 1)
                o = pt_off(iz)
                k0 = pc_ptr(pc_off(iz) + q)
                n = pc_ptr(pc_off(iz) + q + 1) - k0
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                call imf_bin_weights_tab(imf_type, pm(:, m), tot(j), n, pt_mini(k0), &
                                     pt_m1(k0), pt_m2(k0), pt_llo(k0), &
                                     pt_dl(k0), pt_first(k0), &
                                     w(j, k0 - o), ldw)
                do k = k0, k0 + n - 1
                    kl = k - o
                    sm = sm + pt_cm(k) * w(j, kl)
                    sl = sl + pt_cl(k) * w(j, kl)
                    sq = sq + pt_q(k) * w(j, kl)
                    wk(j, kl) = real(w(j, kl), KP)
                end do
                pps(j, 1, q) = sm
                pps(j, 2, q) = sl
                pps(j, 3, q) = sq
            end do
        end do
#else
#if defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, m, k0, n, sm, sl, sq, iz, o, kl)
#endif
        do q = 1, n_pc_max
            do j = 1, nc
                iz = viz(m0 + j - 1)
                if (q > n_pc(iz)) cycle
                m = perm(m0 + j - 1)
                o = pt_off(iz)
                k0 = pc_ptr(pc_off(iz) + q)
                n = pc_ptr(pc_off(iz) + q + 1) - k0
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                call imf_bin_weights_tab(imf_type, pm(:, m), tot(j), n, pt_mini(k0:k0 + n - 1), &
                                     pt_m1(k0:k0 + n - 1), pt_m2(k0:k0 + n - 1), pt_llo(k0:k0 + n - 1), &
                                     pt_dl(k0:k0 + n - 1), pt_first(k0:k0 + n - 1), &
                                     w(j, k0 - o), ldw)
                do k = k0, k0 + n - 1
                    kl = k - o
                    sm = sm + pt_cm(k) * w(j, kl)
                    sl = sl + pt_cl(k) * w(j, kl)
                    sq = sq + pt_q(k) * w(j, kl)
                    wk(j, kl) = real(w(j, kl), KP)
                end do
                pps(j, 1, q) = sm
                pps(j, 2, q) = sl
                pps(j, 3, q) = sq
            end do
        end do
#endif
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(q, sm, sl, sq, iz, m)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(q, sm, sl, sq, iz, m)
#endif
        do t = 1, nt
            do j = 1, nc
                iz = viz(m0 + j - 1)
                m = perm(m0 + j - 1)
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                do q = age_pc((iz - 1) * (nt + 1) + t), age_pc((iz - 1) * (nt + 1) + t + 1) - 1
                    sm = sm + pps(j, 1, q)
                    sl = sl + pps(j, 2, q)
                    sq = sq + pps(j, 3, q)
                end do
                ! An empty isochrone has no mass (and no remnants), as ssp_operator_evaluate
                if (remnants .and. pt_ptr((iz - 1) * (nt + 1) + t + 1) > pt_ptr((iz - 1) * (nt + 1) + t)) &
                    call add_remnant_mass_p(imf_type, pm(:, m), sm, age_mmax((iz - 1) * nt + t))
                smass(t, m) = smass(t, m) + cz(m) * sm
                slum(t, m) = slum(t, m) + cz(m) * log10(max(sl, SAFE_FLOOR))
                if (t <= nt_neb) q0(t, m) = q0(t, m) + cz(m) * sq
            end do
        end do
    end subroutine imf_pieces_multi

    !> fold_segments with the tables of an ssp_opset_t (see imf_pieces_multi): the thread (segment s,
    !> virtual model) skips segments beyond its metallicity's; then (template e, virtual model).
    subroutine fold_multi(m0, nc, n_eff_max, n_seg_max, nt_neb, neb_cols, wsfh, it_tesc, fesc, n_seg, n_eff, pt_off, &
                          seg_off, eff_off, ent_off, pt_age, eff_pt, eff_val, seg_ptr, seg_first, perm, viz, cz, w, vp, v, &
                          veuv, eff_f)
        integer, intent(in) :: m0, nc, n_eff_max, n_seg_max, nt_neb, neb_cols
        real(WP), intent(in), contiguous :: wsfh(:,:), fesc(:), eff_val(:), cz(:)
        real(KP), intent(in), contiguous :: w(:,:)
        integer, intent(in), contiguous  :: it_tesc(:), n_seg(:), n_eff(:), pt_off(:), seg_off(:), eff_off(:), ent_off(:), &
                                            pt_age(:), eff_pt(:), seg_ptr(:), seg_first(:), perm(:), viz(:)
        real(WP), intent(inout), contiguous :: vp(:,:,:)
        real(KP), intent(inout), contiguous :: v(:,:), veuv(:,:)
        !> The entries' values as FP32 hi + lo pairs (ssp_opset_t%eff_f), read by the HIP fold
        real(c_float), intent(in), contiguous :: eff_f(:,:)
        integer  :: s, e, j, k, t, m, tesc, tc, iz, sg, eg
        real(WP) :: x, y, o, ye, oe, f, a, ae

#if defined(OPENSPS_HIP) && defined(OMP_TARGET) && defined(KERNEL_SINGLE_PRECISION)
        call run_hip_fold_multi(m0, nc, nt_neb, n_seg_max, wsfh, it_tesc, fesc, n_seg, pt_off, seg_off, pt_age, eff_pt, &
                                seg_ptr, perm, viz, cz, w, vp, eff_f)
#else

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, t, m, tesc, tc, x, y, o, ye, oe, f, a, ae, &
        !$omp     iz, sg) map(to: fesc)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, t, m, tesc, tc, x, y, o, ye, oe, f, a, ae, iz, sg)
#endif
        do s = 1, n_seg_max
            do j = 1, nc
                iz = viz(m0 + j - 1)
                if (s > n_seg(iz)) cycle
                m = perm(m0 + j - 1)
                sg = seg_off(iz) + s
                tesc = it_tesc(m)
                f = fesc(m)
                y = 0.0_wp; o = 0.0_wp; ye = 0.0_wp; oe = 0.0_wp
                tc = 0
                a = 0.0_wp
                do k = seg_ptr(sg), seg_ptr(sg + 1)   ! (one past the end: flushes the last age)
                    if (k < seg_ptr(sg + 1)) then
                        t = pt_age(pt_off(iz) + eff_pt(k))
                    else
                        t = -1
                    end if
                    if (t /= tc) then
                        if (tc > 0) then
                            x = a * wsfh(m, tc) * cz(m)
                            ae = x
                            if (tc <= nt_neb) ae = f * x
                            if (tc <= tesc) then
                                y = y + x
                                ye = ye + ae
                            else
                                o = o + x
                                oe = oe + ae
                            end if
                        end if
                        tc = t
                        a = 0.0_wp
                    end if
                    if (t > 0) a = a + eff_val(k) * real(w(j, eff_pt(k)), WP)
                end do
                vp(j, 1, s) = y
                vp(j, 2, s) = o
                vp(j, 3, s) = ye
                vp(j, 4, s) = oe
            end do
        end do
#endif
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(s, y, o, ye, oe, iz, eg)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(s, y, o, ye, oe, iz, eg)
#endif
        do e = 1, n_eff_max
            do j = 1, nc
                iz = viz(m0 + j - 1)
                if (e > n_eff(iz)) cycle
                eg = eff_off(iz) + e
                y = 0.0_wp; o = 0.0_wp; ye = 0.0_wp; oe = 0.0_wp
                do s = seg_first(eg), seg_first(eg + 1) - 1
                    y = y + vp(j, 1, s)
                    o = o + vp(j, 2, s)
                    ye = ye + vp(j, 3, s)
                    oe = oe + vp(j, 4, s)
                end do
                v(e, j) = real(y, KP)
                v(e, nc + j) = real(o, KP)
                if (neb_cols > 0) then
                    veuv(e, j) = real(ye, KP)
                    veuv(e, nc + j) = real(oe, KP)
                end if
            end do
        end do
    end subroutine fold_multi

    !> perm: the models m with a nonzero weight cz(m), grouped by key(m) (counting sort, models in
    !> order within a key); ptr(iz) .. ptr(iz + 1) - 1 the entries of key iz.
    subroutine group_models(n_mod, nz, key, cz, perm, ptr, cnt)
        integer, intent(in)  :: n_mod, nz, key(:)
        real(WP), intent(in) :: cz(:)
        integer, intent(out) :: perm(:), ptr(:), cnt(:)
        integer :: m
        cnt = 0
        do m = 1, n_mod
            if (cz(m) /= 0.0_wp) cnt(key(m)) = cnt(key(m)) + 1
        end do
        ptr(1) = 1
        do m = 1, nz
            ptr(m + 1) = ptr(m) + cnt(m)
        end do
        cnt(1:nz) = ptr(1:nz)
        do m = 1, n_mod
            if (cz(m) /= 0.0_wp) then
                perm(cnt(key(m))) = m
                cnt(key(m)) = cnt(key(m)) + 1
            end if
        end do
    end subroutine group_models

    !> yo(:, perm(m0 + j - 1)) += yz(:, j) and yo(:, n_mod + perm(m0 + j - 1)) += yz(:, nc + j): the
    !> young and old sums of a chunk added to the batch's.
    subroutine add_columns(m0, nc, nspec, n_mod, perm, yz, yo)
        integer, intent(in) :: m0, nc, nspec, n_mod
        integer, intent(in), contiguous :: perm(:)
        real(KP), intent(in), contiguous :: yz(:,:)
        real(KP), intent(inout), contiguous :: yo(:,:)
        integer :: j, i, m

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(m)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(m)
#endif
        do j = 1, nc
            do i = 1, nspec
                m = perm(m0 + j - 1)
                yo(i, m) = yo(i, m) + yz(i, j)
                yo(i, n_mod + m) = yo(i, n_mod + m) + yz(i, nc + j)
            end do
        end do
    end subroutine add_columns

    !> yo(1:np, c0 + j) = etpl(1:np, 1:k) . v(1:k, v0 + j), j = 1 .. nc: the age sums from the
    !> template weights, compensated (Kahan) with FP32 kernels.
    !> - cuBLAS builds (-Dcublas=true): GEMMs; with FP32 kernels over blocks of KB templates, each
    !>   block's product added to yo with compensation (a plain FP32 GEMM over ~2600 templates
    !>   loses ~1e-6).
    !> - Device: a register tile per thread, PT pixels (strided, so loads coalesce) x MT models,
    !>   the templates in order (the age_sums tile of compute_csp_batch; PT + MT loads per PT MT
    !>   multiply-adds instead of 2 per 1).
    !> - Host: per block of pixels, each template block read once for all the chunk's models.
    !> Each (pixel, column) sums the templates in order in the last two (the same results).
    subroutine template_columns(dev, np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
        logical, intent(in) :: dev
        integer, intent(in) :: np, nspec, n_eff, k, nc, v0, c0
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
        integer, parameter :: MT = TUNE_TEMPLATE_MT, PT = TUNE_TEMPLATE_PT, PIX_BLOCK = TUNE_TEMPLATE_PIX_BLOCK, &
                              KB = TUNE_TEMPLATE_KB
        integer  :: i, j, e, e0, mq, ip, npt, l, p, mm(MT), ii(PT), p0, p1, nb, q
        real(KP) :: bk(PT), wk(MT), acc(PT, MT), cmp(PT, MT), blk(PT, MT), term, tsum
        real(KP), allocatable :: a(:,:), c(:,:)

        if (np < 1 .or. nc < 1) return
        if (k < 1) then
            if (dev) then
#ifdef OMP_TARGET
                !$omp target teams distribute parallel do collapse(2)
#endif
                do j = 1, nc
                    do i = 1, np
                        yo(i, c0 + j) = 0.0_KP
                    end do
                end do
            else
                yo(1:np, c0 + 1:c0 + nc) = 0.0_KP
            end if
            return
        end if

        if (dev) then
            ! (A build with an accelerated BLAS keeps it for the template products: the BLAS is an explicit choice of the
            ! build, and cuBLAS' product beats the tiled kernel on an A100 (the HIP kernel took its place by mistake)
#if defined(OPENSPS_HIP) && defined(OMP_TARGET) && defined(KERNEL_SINGLE_PRECISION) && !defined(OPENSPS_ACCEL_BLAS)
            if (run_hip_template_columns(np, k, nc, etpl, v, v0, yo, c0)) return
#endif
#if defined(OPENSPS_ACCEL_BLAS)
            call gemm_columns(np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
            return
#endif
            npt = (np + PT - 1) / PT
#ifdef OMP_TARGET
            !$omp target teams loop collapse(2) private(mm, ii, bk, wk, acc, cmp, blk, e, e0, l, p, term, tsum)
#endif
            do mq = 1, (nc + MT - 1) / MT
                do ip = 1, npt
                    do l = 1, MT
                        ! Columns of this thread (clamped at the end; duplicates write the same value)
                        mm(l) = min((mq - 1) * MT + l, nc)
                    end do
                    do p = 1, PT
                        ii(p) = min(ip + (p - 1) * npt, np)
                    end do
                    acc = 0.0_KP
                    cmp = 0.0_KP
                    if (KP == WP) then
                        ! FP64 kernels: plain sums, skipping the zero weights (sparse in age)
                        do e = 1, k
                            do l = 1, MT
                                wk(l) = v(e, v0 + mm(l))
                            end do
                            do p = 1, PT
                                bk(p) = etpl(ii(p), e)
                            end do
                            do l = 1, MT
                                if (wk(l) == 0.0_KP) cycle
                                do p = 1, PT
                                    acc(p, l) = acc(p, l) + wk(l) * bk(p)
                                end do
                            end do
                        end do
                    else
                        ! FP32 kernels: blocks of KB templates summed with plain multiply-adds (all
                        ! terms >= 0: a block loses at most ~KB eps of itself), the blocks into the
                        ! total with compensation (rule 1 of doc/MIXED_PRECISION.md): ~2 instead of
                        ! ~5 operations per term. No zero-weight test here: its branches cost the gain
                        do e0 = 1, k, KB
                            blk = 0.0_KP
                            do e = e0, min(e0 + KB - 1, k)
                                do l = 1, MT
                                    wk(l) = v(e, v0 + mm(l))
                                end do
                                do p = 1, PT
                                    bk(p) = etpl(ii(p), e)
                                end do
                                do l = 1, MT
                                    do p = 1, PT
                                        blk(p, l) = blk(p, l) + wk(l) * bk(p)
                                    end do
                                end do
                            end do
                            do l = 1, MT
                                do p = 1, PT
                                    term = blk(p, l) - cmp(p, l)
                                    tsum = acc(p, l) + term
                                    cmp(p, l) = (tsum - acc(p, l)) - term
                                    acc(p, l) = tsum
                                end do
                            end do
                        end do
                    end if
                    do l = 1, MT
                        do p = 1, PT
                            yo(ii(p), c0 + mm(l)) = acc(p, l)
                        end do
                    end do
                end do
            end do
            return
        end if

        ! Host: blocks of pixels (threads), the templates in order, the columns, the pixels inner
        !$omp parallel do schedule(dynamic) private(p0, p1, nb, a, c, e, j, q, wk, term, tsum)
        do p0 = 1, np, PIX_BLOCK
            p1 = min(p0 + PIX_BLOCK - 1, np)
            nb = p1 - p0 + 1
            allocate(a(nb, nc), c(nb, nc))
            a = 0.0_KP
            c = 0.0_KP
            do e = 1, k
                do j = 1, nc
                    wk(1) = v(e, v0 + j)
                    if (wk(1) == 0.0_KP) cycle
                    if (KP == WP) then
                        do q = 1, nb
                            a(q, j) = a(q, j) + wk(1) * etpl(p0 + q - 1, e)
                        end do
                    else
                        do q = 1, nb
                            term = wk(1) * etpl(p0 + q - 1, e) - c(q, j)
                            tsum = a(q, j) + term
                            c(q, j) = (tsum - a(q, j)) - term
                            a(q, j) = tsum
                        end do
                    end if
                end do
            end do
            yo(p0:p1, c0 + 1:c0 + nc) = a
            deallocate(a, c)
        end do
    end subroutine template_columns

#if defined(OPENSPS_ACCEL_BLAS)
#ifdef KERNEL_SINGLE_PRECISION
    !> gemm_columns for few columns (FP32 kernels): the full blocks of KB templates in one strided
    !> batched GEMM into a work array, the remainder block in one more, then each (pixel, column)
    !> adds the blocks in order with compensation (the arithmetic of gemm_columns).
    subroutine gemm_columns_batched(np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
        use, intrinsic :: iso_c_binding, only: c_long_long
        integer, intent(in) :: np, nspec, n_eff, k, nc, v0, c0
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
        integer, parameter :: KB = TUNE_GEMM_KB
        integer :: nfull, rem, nb, i, j, b
        real(KP) :: term, tsum, acc, cmp

        nfull = k / KB
        rem = k - nfull * KB
        nb = nfull + merge(1, 0, rem > 0)
        if (.not. allocated(gtmp)) then
            allocate(gtmp(1, 1, 1))
        end if
        if (size(gtmp, 1) < np .or. size(gtmp, 2) < NC_BATCHED .or. size(gtmp, 3) < nb) then
            call free_gtmp()
            allocate(gtmp(max(np, size(gtmp, 1)), NC_BATCHED, max(nb, size(gtmp, 3))))
            !$omp target enter data map(alloc: gtmp)
        end if
        if (nfull > 0) then
            call run_batched(np, nc, KB, etpl, nspec, v, n_eff, v0, gtmp, size(gtmp, 1), &
                             int(size(gtmp, 1), c_long_long) * size(gtmp, 2), nfull)
        end if
        if (rem > 0) then
            call run_rem(np, nc, rem, etpl, nspec, nfull * KB, v, n_eff, v0, &
                         gtmp, size(gtmp, 1), nb)
        end if
        !$omp target teams distribute parallel do collapse(2) private(acc, cmp, term, tsum, b)
        do j = 1, nc
            do i = 1, np
                acc = 0.0_KP
                cmp = 0.0_KP
                do b = 1, nb
                    term = gtmp(i, j, b) - cmp
                    tsum = acc + term
                    cmp = (tsum - acc) - term
                    acc = tsum
                end do
                yo(i, c0 + j) = acc
            end do
        end do
    end subroutine gemm_columns_batched

    subroutine run_batched(np, nc, kb, etpl, nspec, v, n_eff, v0, gtmp, ld_gtmp, stride_gtmp, nfull)
        use, intrinsic :: iso_c_binding, only: c_long_long, c_loc
        use opensps_accel_blas, only: accel_blas_gemm_strided_batched, ACCEL_BLAS_OP_N
        integer, intent(in) :: np, nc, kb, nspec, n_eff, v0, ld_gtmp, nfull
        integer(c_long_long), intent(in) :: stride_gtmp
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: gtmp(:,:,:)
#ifdef OMP_TARGET
        !$omp target data use_device_addr(etpl, v, gtmp)
#endif
        call accel_blas_gemm_strided_batched(ACCEL_BLAS_OP_N, ACCEL_BLAS_OP_N, np, nc, kb, 1.0_KP, &
                                             c_loc(etpl), nspec, int(kb, c_long_long) * nspec, &
                                             c_loc(v(1, v0 + 1)), n_eff, int(kb, c_long_long), &
                                             0.0_KP, c_loc(gtmp), ld_gtmp, stride_gtmp, nfull)
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_batched

    subroutine run_rem(np, nc, rem, etpl, nspec, e_off, v, n_eff, v0, gtmp, ld_gtmp, b_off)
        use, intrinsic :: iso_c_binding, only: c_loc
        use opensps_accel_blas, only: accel_blas_gemm, ACCEL_BLAS_OP_N
        integer, intent(in) :: np, nc, rem, nspec, e_off, n_eff, v0, ld_gtmp, b_off
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: gtmp(:,:,:)
#ifdef OMP_TARGET
        !$omp target data use_device_addr(etpl, v, gtmp)
#endif
        call accel_blas_gemm(ACCEL_BLAS_OP_N, ACCEL_BLAS_OP_N, np, nc, rem, 1.0_KP, &
                             c_loc(etpl(1, e_off + 1)), nspec, c_loc(v(e_off + 1, v0 + 1)), n_eff, &
                             0.0_KP, c_loc(gtmp(1, 1, b_off)), ld_gtmp)
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_rem

    subroutine free_gtmp()
        if (allocated(gtmp)) then
            if (size(gtmp) > 1) then
                !$omp target exit data map(delete: gtmp)
            end if
            deallocate(gtmp)
        end if
    end subroutine free_gtmp
#endif

    !> template_columns with GPU BLAS (arrays on the device). FP64: one GEMM. FP32: a GEMM per
    !> block of KB templates into a work array, each added to yo with a compensation term.
    subroutine gemm_columns(np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
        use opensps_accel_blas, only: accel_blas_init
        integer, intent(in) :: np, nspec, n_eff, k, nc, v0, c0
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
        integer, parameter :: KB = TUNE_GEMM_KB
#ifdef KERNEL_SINGLE_PRECISION
        integer :: j0, nc1
#endif

        call accel_blas_init()
#ifdef KERNEL_SINGLE_PRECISION
        do j0 = 1, nc, NC_BATCHED
            nc1 = min(NC_BATCHED, nc - j0 + 1)
            call gemm_columns_batched(np, nspec, n_eff, k, nc1, etpl, v, v0 + j0 - 1, yo, c0 + j0 - 1)
        end do
#else
        call run_fp64(np, nc, k, etpl, nspec, v, n_eff, v0, yo, size(yo, 1), c0)
#endif
    end subroutine gemm_columns

#ifdef KERNEL_SINGLE_PRECISION
    subroutine run_block(np, nc, kb1, etpl, nspec, e0, v, n_eff, v0, tmp, ld_tmp)
        use, intrinsic :: iso_c_binding, only: c_loc
        use opensps_accel_blas, only: accel_blas_gemm, ACCEL_BLAS_OP_N
        integer, intent(in) :: np, nc, kb1, nspec, e0, n_eff, v0, ld_tmp
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: tmp(:,:)
#ifdef OMP_TARGET
        !$omp target data use_device_addr(etpl, v, tmp)
#endif
        call accel_blas_gemm(ACCEL_BLAS_OP_N, ACCEL_BLAS_OP_N, np, nc, kb1, 1.0_KP, &
                             c_loc(etpl(1, e0)), nspec, c_loc(v(e0, v0 + 1)), n_eff, &
                             0.0_KP, c_loc(tmp), ld_tmp)
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_block
#else
    subroutine run_fp64(np, nc, k, etpl, nspec, v, n_eff, v0, yo, ld_yo, c0)
        use, intrinsic :: iso_c_binding, only: c_loc
        use opensps_accel_blas, only: accel_blas_gemm, ACCEL_BLAS_OP_N
        integer, intent(in) :: np, nc, k, nspec, n_eff, v0, ld_yo, c0
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
#ifdef OMP_TARGET
        !$omp target data use_device_addr(etpl, v, yo)
#endif
        call accel_blas_gemm(ACCEL_BLAS_OP_N, ACCEL_BLAS_OP_N, np, nc, k, 1.0_KP, &
                             c_loc(etpl), nspec, c_loc(v(1, v0 + 1)), n_eff, &
                             0.0_KP, c_loc(yo(1, c0 + 1)), ld_yo)
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine run_fp64
#endif
#endif

end module opensps_ssp_batch
