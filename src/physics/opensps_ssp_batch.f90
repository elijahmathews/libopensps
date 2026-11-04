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

    use opensps_precision, only: WP, KP
    use opensps_constants, only: SAFE_FLOOR
    use opensps_imf, only: imf_integral_p, imf_bin_weights_tab, imf_bin_logs, IMF_P_LOWER_LIMIT, IMF_P_UPPER_LIMIT, &
                           IMF_P_ALPHA, IMF_P_VDMC, IMF_P_MDAVE, IMF_P_LOWER_BOUND, IMF_NPAR, imf_integral_dp, &
                           integrate_powerlaw, powerlaw_dj_de
    use opensps_stellar_modifications, only: add_remnant_mass_p
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_to_device, ssp_operator_from_device
    implicit none
    private

    public :: ssp_batch_age_sums_z, ssp_batch_age_sums_adjoint, NIMFG, dot_lanes
    public :: ssp_opset_t, ssp_opset_build, ssp_opset_free

    !> The IMF parameters of the gradients (imf1, imf2, imf3, vdmc, mdave)
    integer, parameter :: NIMFG = 5

    !> The kernels run on the device in OpenMP target builds and as OpenMP host loops otherwise
#ifdef OMP_TARGET
    logical, parameter :: ON_DEV = .true.
#else
    logical, parameter :: ON_DEV = .false.
#endif

#ifdef FSPS_CUBLAS
#ifdef KERNEL_SINGLE_PRECISION
    !> Column counts up to which the FP32 cuBLAS product uses the batched form (gemm_columns_batched),
    !> and its work array (pixels x NC_BATCHED x blocks, on the device once allocated)
    integer, parameter :: NC_BATCHED = 512
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
        !> The template entries by point (the reverse pass's gathers): point gp (global index) has the
        !> entries pe_k(pe_ptr(gp) .. pe_ptr(gp + 1) - 1) (global entry indices), of templates pe_e (local)
        integer, allocatable :: pe_ptr(:), pe_k(:), pe_e(:)
        logical :: on_device = .false.
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
        call entries_by_point(set, np, max(nen, 1))
        call move_alloc(ops, set%ops)
    end subroutine ssp_opset_build

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

    subroutine ssp_opset_free(set)
        type(ssp_opset_t), intent(inout) :: set
        integer :: iz
#ifdef OMP_TARGET
        if (set%on_device) then
            !$omp target exit data map(delete: set%n_pc, set%n_seg, set%n_eff, set%pt_off, set%pc_off, set%seg_off, &
            !$omp     set%ent_off, set%eff_off, set%pt_first, set%pt_age, set%pc_ptr, set%age_pc, set%pt_ptr, set%seg_ptr, &
            !$omp     set%seg_first, set%eff_pt, set%pt_mini, set%pt_m1, set%pt_m2, set%pt_cm, set%pt_cl, set%pt_q, &
            !$omp     set%age_mmax, set%eff_val, set%pt_llo, set%pt_dl, set%pe_ptr, set%pe_k, set%pe_e)
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
                                                set%eff_val, set%age_pc, set%pt_ptr, set%age_mmax, set%pt_llo, set%pt_dl)
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
        !$omp     set%age_mmax, set%eff_val, set%pt_llo, set%pt_dl, set%pe_ptr, set%pe_k, set%pe_e)
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
    subroutine ssp_batch_age_sums_z(imf_type, remnants, n_mod, nt, nspec, set, zlo, zhi, dz, pm, wsfh, it_tesc, &
                                    nt_neb, whl, fesc, yo, smass, slum, q0)
        logical, intent(in)  :: remnants
        integer, intent(in)  :: imf_type, n_mod, nt, nspec, nt_neb, whl
        type(ssp_opset_t), intent(inout) :: set
        integer, intent(in), contiguous  :: zlo(:), zhi(:), it_tesc(:)
        real(WP), intent(in), contiguous :: dz(:), pm(:,:), wsfh(:,:), fesc(:)
        real(KP), intent(inout), contiguous :: yo(:,:)
        real(WP), intent(inout), contiguous :: smass(:,:), slum(:,:), q0(:,:)

        integer, parameter :: CHUNK = 2048
        integer :: nc_max, neb_cols, nz, pass, m, k, t, ntq
        integer, allocatable :: perm1(:), perm2(:), viz1(:), viz2(:), ptr1(:), ptr2(:), cnt(:)
        real(WP), allocatable :: cz1(:), cz2(:), w(:,:), tot(:), vp(:,:,:), pps(:,:,:)
        real(KP), allocatable :: wk(:,:), v(:,:), veuv(:,:), yz(:,:)
        logical :: direct

        nz = set%nz
        ! One metallicity: the models in their own order, one pass, the products go straight into yo
        direct = nz == 1
        call ssp_opset_to_device(set)
        neb_cols = 0
        if (nt_neb > 0 .and. whl > 0) neb_cols = 1
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

        allocate(w(nc_max, set%n_pts_max), wk(nc_max, set%n_pts_max), tot(nc_max), v(set%n_eff_max, 2 * nc_max))
        allocate(veuv(set%n_eff_max, max(1, neb_cols * 2 * nc_max)), vp(nc_max, 4, set%n_seg_max), &
                 pps(nc_max, 3, set%n_pc_max))
        if (direct) then
            allocate(yz(1, 1))
        else
            allocate(yz(nspec, 2 * nc_max))
        end if
#ifdef OMP_TARGET
        !$omp target data map(alloc: w, wk, tot, v, veuv, vp, pps, yz) map(to: perm1, perm2, viz1, viz2, cz1, cz2)
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
                call run_pass(ptr1(nz + 1) - 1, perm1, viz1, cz1)
            else
                call run_pass(ptr2(nz + 1) - 1, perm2, viz2, cz2)
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
        subroutine run_pass(np, perm, viz, cz)
            integer, intent(in) :: np
            integer, intent(in), contiguous :: perm(:), viz(:)
            real(WP), intent(in), contiguous :: cz(:)
            integer :: m0, nc, j, j0, jr, iz, ky, nrun

            do m0 = 1, np, CHUNK
                nc = min(CHUNK, np - m0 + 1)
                call imf_pieces_multi(imf_type, remnants, m0, nc, nt, set%n_pc_max, nt_neb, pm, set%n_pc, set%pt_off, &
                                      set%pc_off, set%pc_ptr, set%age_pc, set%pt_ptr, set%pt_mini, set%pt_m1, set%pt_m2, &
                                      set%pt_llo, set%pt_dl, set%pt_first, set%pt_cm, set%pt_cl, set%pt_q, set%age_mmax, &
                                      size(w, 1), set%n_pts_max, w, wk, tot, pps, perm, viz, cz, smass, slum, q0)
                call fold_multi(m0, nc, set%n_eff_max, set%n_seg_max, nt_neb, neb_cols, wsfh, it_tesc, fesc, &
                                set%n_seg, set%n_eff, set%pt_off, set%seg_off, set%eff_off, set%ent_off, set%pt_age, &
                                set%eff_pt, set%eff_val, set%seg_ptr, set%seg_first, perm, viz, cz, wk, vp, v, veuv)
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
    !>   1e-6, error ~1e-10 relative; with -Dfp32_imf_weights step 1e-3, ~1e-4),
    !>   fesc_bar, and zbar = d L / d dz (the two metallicity passes' difference, as the weights
    !>   blend them: linearly, L_bol in log).
    !> Chunks of CHG models (the point weights are chunk x points arrays); atomic adds where the
    !> fold's templates share ages and points.
    subroutine ssp_batch_age_sums_adjoint(imf_type, remnants, n_mod, nt, nspec, set, zlo, zhi, dz, pm, wsfh, it_tesc, &
                                          nt_neb, whl, fesc, gw, slum, lsum_bar, qbar, wbar, imf_bar, fesc_bar, zbar, &
                                          smass, msum_bar)
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

        integer, parameter :: CHG = 256
        integer, parameter :: PROWS(NIMFG) = [IMF_P_ALPHA, IMF_P_ALPHA + 1, IMF_P_ALPHA + 2, IMF_P_VDMC, IMF_P_MDAVE]
        integer :: nz, pass, m, t, j, i, np, nc_max, neb_cols, ntq, m0, nc, j0, jr, iz, ky, ig, isg
        integer, allocatable :: perm(:), viz(:), ptr(:), cnt(:)
        real(WP), allocatable :: cz(:), ones(:), w(:,:), wm(:,:), tot(:), pps(:,:,:), slz(:,:), sqz(:,:), wpb(:,:), &
                                 vb(:,:,:), dsm(:,:), dsl(:,:), dsq(:,:), pmp(:,:), part(:,:), zsg(:), dsp(:,:), sxa(:,:), &
                                 sea(:,:), apt(:,:), amp(:,:,:), dtt(:,:), remp(:,:), remm(:,:), agt(:,:,:)
        real(KP), allocatable :: wk(:,:)
        real(WP) :: sgn

        nz = set%nz
        call ssp_opset_to_device(set)
        neb_cols = 0
        if (nt_neb > 0 .and. whl > 0) neb_cols = 1
        nc_max = min(CHG, n_mod)
        ntq = size(qbar, 1)
        allocate(perm(n_mod), viz(n_mod), ptr(nz + 1), cnt(nz + 1), cz(n_mod), ones(n_mod), zsg(n_mod))
        allocate(w(nc_max, set%n_pts_max), wm(nc_max, set%n_pts_max), wk(nc_max, set%n_pts_max), tot(nc_max), &
                 pps(nc_max, 3, set%n_pc_max), slz(nt, nc_max), sqz(nt, nc_max), wpb(nc_max, set%n_pts_max), &
                 vb(set%n_eff_max, 4, nc_max), dsm(nt, n_mod), dsl(nt, n_mod), dsq(nt, n_mod), pmp(size(pm, 1), n_mod), &
                 part(set%n_pc_max, nc_max), dsp(nt, n_mod), sxa(nt, nc_max), sea(nt, nc_max), &
                 apt(NIMFG, nc_max), amp(NIMFG, nt, nc_max), dtt(NIMFG, nc_max), &
                 remp(nt, nc_max), remm(nt, nc_max), agt(NIMFG, nt, nc_max))
        ones = 1.0_wp
#ifdef OMP_TARGET
        !$omp target data map(alloc: w, wm, wk, tot, pps, slz, sqz, wpb, vb, dsm, dsl, dsq, part, dsp, sxa, sea) &
        !$omp& map(alloc: apt, amp, dtt, remp, remm, agt) &
        !$omp& map(to: pm, ones, msum_bar) map(from: wbar, imf_bar, fesc_bar, zbar) map(alloc: perm, viz, cz, zsg, pmp)
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
            do i = 1, NIMFG
                imf_bar(i, m) = 0.0_wp
            end do
            fesc_bar(m) = 0.0_wp
            zbar(m) = 0.0_wp
        end do
        do pass = 1, 2
            ! Every model in both passes (with the metallicity free, d / d dz takes both, whatever cz)
            if (pass == 1) then
                cz = 1.0_wp - dz(1:n_mod)
                zsg = -1.0_wp
                call group_models(n_mod, nz, zlo, ones, perm, ptr, cnt)
            else
                if (nz == 1) exit
                cz = dz(1:n_mod)
                zsg = 1.0_wp
                call group_models(n_mod, nz, zhi, ones, perm, ptr, cnt)
            end if
            np = ptr(nz + 1) - 1
            do j = 1, np
                if (pass == 1) then
                    viz(j) = zlo(perm(j))
                else
                    viz(j) = zhi(perm(j))
                end if
            end do
            ! (cz, zsg by virtual model)
            do j = 1, np
                ones(j) = cz(perm(j))
            end do
            cz(1:np) = ones(1:np)
            zsg(1:np) = zsg(1)
            ones = 1.0_wp
#ifdef OMP_TARGET
            !$omp target update to(perm, viz, cz, zsg)
#endif
            do m0 = 1, np, CHG
                nc = min(CHG, np - m0 + 1)
                ! The point weights (and per age sl, sq of this metallicity; dsm: the surviving mass of this
                ! metallicity per age, for the mass output's dz)
                call adj_zero_cols(m0, nc, nt, perm, dsm)
                call imf_pieces_multi(imf_type, remnants, m0, nc, nt, set%n_pc_max, nt_neb, pm, set%n_pc, set%pt_off, &
                                      set%pc_off, set%pc_ptr, set%age_pc, set%pt_ptr, set%pt_mini, set%pt_m1, set%pt_m2, &
                                      set%pt_llo, set%pt_dl, set%pt_first, set%pt_cm, set%pt_cl, set%pt_q, set%age_mmax, &
                                      size(w, 1), set%n_pts_max, w, wk, tot, pps, perm, viz, ones, dsm, dsl, dsq)
                call adj_age_sums_sl(m0, nc, nt, set%age_pc, viz, pps, slz, sqz)
                call adj_mass_dot(m0, nc, nt, perm, cz, zsg, wsfh, msum_bar, dsm, dsm, 0, 0, 1, pm, imf_bar, zbar)
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
                        ky = max(ky, it_tesc(perm(m0 + j - 1)))
                    end do
                    ky = set%ops(iz)%k_age(max(1, min(nt, ky)))
                    call adj_template_bar(nspec, set%ops(iz)%n_eff, ky, neb_cols * whl, n_mod, m0, j0, jr, perm, &
                                          set%ops(iz)%etpl, set%ops(iz)%etpl_t, gw, set%n_eff_max, nc_max, vb)
                    j0 = jr + 1
                end do
                ! (w, wm: free until the IMF rows' differences below)
                call adj_fold_gather(m0, nc, nt, nt_neb, neb_cols, set%n_pts_max, wsfh, it_tesc, fesc, set%pt_off, &
                                     set%pt_ptr, set%pt_age, set%eff_val, set%pe_ptr, set%pe_k, set%pe_e, perm, viz, cz, &
                                     zsg, wk, vb, wbar, wpb, fesc_bar, zbar, w, wm, sxa, sea)
                call adj_lq(m0, nc, nt, nt_neb, set%n_pc_max, set%n_pc, set%pt_off, set%pc_off, set%pc_ptr, set%pt_age, &
                            set%pt_cl, set%pt_q, perm, viz, cz, zsg, wsfh, slum, lsum_bar, qbar, ntq, slz, sqz, wpb, zbar)
                ! The IMF parameters: Kroupa and Dave (power-law slopes and break) analytically
                ! (adj_imf_analytic); otherwise central differences of the point weights (only those the
                ! IMF type reads; the others' rows are zero)
                if (mod(imf_type, 10) == 2 .or. mod(imf_type, 10) == 4) then
                    call adj_imf_analytic(imf_type, remnants, m0, nc, nt, set%n_pc_max, set%n_pc, set%pt_off, &
                                          set%pc_off, set%pc_ptr, set%age_pc, set%pt_ptr, set%pt_mini, set%pt_m1, &
                                          set%pt_m2, set%pt_llo, set%pt_dl, set%pt_first, set%pt_cm, set%age_mmax, perm, &
                                          viz, cz, wsfh, pm, &
                                          pmp, tot, size(wk, 1), wk, size(wpb, 1), wpb, msum_bar, apt, amp, agt, dtt, &
                                          remp, remm, imf_bar)
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
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine ssp_batch_age_sums_adjoint

    !> The IMF rows of Kroupa and Dave analytically: dw/d theta of each point weight w = N / M (N the
    !> bin's number integral, M the IMF's mass integral between its limits) is
    !> (dN/d theta - w dM/d theta) / M, with dN, dM from imf_integral_dp; then, per (piece, model),
    !> the sums of w_bar dw (the template, L_bol and Q cotangents of the points) and of the points'
    !> surviving masses times dw (the mass output's), per row; per model the rows. The remnants' part
    !> of the surviving mass (a scalar per age and model) by central differences, as before.
    subroutine adj_imf_analytic(imf_type, remnants, m0, nc, nt, n_pc_max, n_pc, pt_off, pc_off, pc_ptr, age_pc, &
                                pt_ptr, pt_mini, pt_m1, pt_m2, pt_llo, pt_dl, pt_first, pt_cm, age_mmax, perm, viz, cz, &
                                wsfh, pm, pmp, tot, ldk, wk, ldw, wpb, msum_bar, apt, amp, agt, dtt, remp, remm, imf_bar)
        integer, intent(in) :: imf_type, m0, nc, nt, n_pc_max, ldk, ldw
        logical, intent(in) :: remnants
        integer, intent(in), contiguous :: n_pc(:), pt_off(:), pc_off(:), pc_ptr(:), age_pc(:), pt_ptr(:), pt_first(:), &
                                           perm(:), viz(:)
        real(WP), intent(in), contiguous :: pt_mini(:), pt_m1(:), pt_m2(:), pt_llo(:), pt_dl(:), pt_cm(:), age_mmax(:), &
                                            cz(:), wsfh(:,:), pm(:,:), tot(:), msum_bar(:)
        ! (explicit leading dimensions: a point's row is passed by its first element, as imf_pieces_multi)
        real(KP), intent(in) :: wk(ldk, *)
        real(WP), intent(in) :: wpb(ldw, *)
        real(WP), intent(inout), contiguous :: pmp(:,:), apt(:,:), amp(:,:,:), agt(:,:,:), dtt(:,:), remp(:,:), remm(:,:), imf_bar(:,:)
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
    !> ~1e-10); with -Dfp32_imf_weights 1e-3 (the weights' FP32 rounding, ~1e-7, over the step: ~1e-4)
    pure real(WP) function adj_step(x)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: x
#if defined(OPENSPS_FP32_IMF_WEIGHTS)
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
        integer  :: e, jb, nb, i, j1, j2, j3, j4, m1, m2, m3, m4, jj, mm
        real(KP) :: ev, term, tsum
        real(KP) :: sy1, sy2, sy3, sy4, so1, so2, so3, so4, ey1, ey2, ey3, ey4, eo1, eo2, eo3, eo4
        real(KP) :: csy1, csy2, csy3, csy4, cso1, cso2, cso3, cso4, cey1, cey2, cey3, cey4, ceo1, ceo2, ceo3, ceo4

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
#else
        nb = (jr - j0 + 4) / 4
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(i, j1, j2, j3, j4, m1, m2, m3, m4, ev, &
        !$omp& term, tsum, sy1, sy2, sy3, sy4, so1, so2, so3, so4, ey1, ey2, ey3, ey4, eo1, eo2, eo3, eo4, &
        !$omp& csy1, csy2, csy3, csy4, cso1, cso2, cso3, cso4, cey1, cey2, cey3, cey4, ceo1, ceo2, ceo3, ceo4)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(i, j1, j2, j3, j4, m1, m2, m3, m4, ev, term, tsum, sy1, sy2, sy3, &
        !$omp& sy4, so1, so2, so3, so4, ey1, ey2, ey3, ey4, eo1, eo2, eo3, eo4, csy1, csy2, csy3, csy4, cso1, &
        !$omp& cso2, cso3, cso4, cey1, cey2, cey3, cey4, ceo1, ceo2, ceo3, ceo4)
#endif
        do jb = 1, nb
            do e = 1, n_eff
                j1 = min(j0 + (jb - 1) * 4, jr)
                j2 = min(j0 + (jb - 1) * 4 + 1, jr)
                j3 = min(j0 + (jb - 1) * 4 + 2, jr)
                j4 = min(j0 + (jb - 1) * 4 + 3, jr)
                m1 = perm(m0 + j1 - 1)
                m2 = perm(m0 + j2 - 1)
                m3 = perm(m0 + j3 - 1)
                m4 = perm(m0 + j4 - 1)
                sy1 = 0.0_KP
                sy2 = 0.0_KP
                sy3 = 0.0_KP
                sy4 = 0.0_KP
                so1 = 0.0_KP
                so2 = 0.0_KP
                so3 = 0.0_KP
                so4 = 0.0_KP
                ey1 = 0.0_KP
                ey2 = 0.0_KP
                ey3 = 0.0_KP
                ey4 = 0.0_KP
                eo1 = 0.0_KP
                eo2 = 0.0_KP
                eo3 = 0.0_KP
                eo4 = 0.0_KP
                csy1 = 0.0_KP
                csy2 = 0.0_KP
                csy3 = 0.0_KP
                csy4 = 0.0_KP
                cso1 = 0.0_KP
                cso2 = 0.0_KP
                cso3 = 0.0_KP
                cso4 = 0.0_KP
                cey1 = 0.0_KP
                cey2 = 0.0_KP
                cey3 = 0.0_KP
                cey4 = 0.0_KP
                ceo1 = 0.0_KP
                ceo2 = 0.0_KP
                ceo3 = 0.0_KP
                ceo4 = 0.0_KP
                do i = 1, nwhl
                    ev = etpl_t(e, i)
                    if (KP == WP) then
                        ey1 = ey1 + ev * gw(i, m1)
                        eo1 = eo1 + ev * gw(i, n_mod + m1)
                        ey2 = ey2 + ev * gw(i, m2)
                        eo2 = eo2 + ev * gw(i, n_mod + m2)
                        ey3 = ey3 + ev * gw(i, m3)
                        eo3 = eo3 + ev * gw(i, n_mod + m3)
                        ey4 = ey4 + ev * gw(i, m4)
                        eo4 = eo4 + ev * gw(i, n_mod + m4)
                    else
                        term = ev * gw(i, m1) - cey1
                        tsum = ey1 + term
                        cey1 = (tsum - ey1) - term
                        ey1 = tsum
                        term = ev * gw(i, n_mod + m1) - ceo1
                        tsum = eo1 + term
                        ceo1 = (tsum - eo1) - term
                        eo1 = tsum
                        term = ev * gw(i, m2) - cey2
                        tsum = ey2 + term
                        cey2 = (tsum - ey2) - term
                        ey2 = tsum
                        term = ev * gw(i, n_mod + m2) - ceo2
                        tsum = eo2 + term
                        ceo2 = (tsum - eo2) - term
                        eo2 = tsum
                        term = ev * gw(i, m3) - cey3
                        tsum = ey3 + term
                        cey3 = (tsum - ey3) - term
                        ey3 = tsum
                        term = ev * gw(i, n_mod + m3) - ceo3
                        tsum = eo3 + term
                        ceo3 = (tsum - eo3) - term
                        eo3 = tsum
                        term = ev * gw(i, m4) - cey4
                        tsum = ey4 + term
                        cey4 = (tsum - ey4) - term
                        ey4 = tsum
                        term = ev * gw(i, n_mod + m4) - ceo4
                        tsum = eo4 + term
                        ceo4 = (tsum - eo4) - term
                        eo4 = tsum
                    end if
                end do
                do i = nwhl + 1, nspec
                    ev = etpl_t(e, i)
                    if (KP == WP) then
                        sy1 = sy1 + ev * gw(i, m1)
                        so1 = so1 + ev * gw(i, n_mod + m1)
                        sy2 = sy2 + ev * gw(i, m2)
                        so2 = so2 + ev * gw(i, n_mod + m2)
                        sy3 = sy3 + ev * gw(i, m3)
                        so3 = so3 + ev * gw(i, n_mod + m3)
                        sy4 = sy4 + ev * gw(i, m4)
                        so4 = so4 + ev * gw(i, n_mod + m4)
                    else
                        term = ev * gw(i, m1) - csy1
                        tsum = sy1 + term
                        csy1 = (tsum - sy1) - term
                        sy1 = tsum
                        term = ev * gw(i, n_mod + m1) - cso1
                        tsum = so1 + term
                        cso1 = (tsum - so1) - term
                        so1 = tsum
                        term = ev * gw(i, m2) - csy2
                        tsum = sy2 + term
                        csy2 = (tsum - sy2) - term
                        sy2 = tsum
                        term = ev * gw(i, n_mod + m2) - cso2
                        tsum = so2 + term
                        cso2 = (tsum - so2) - term
                        so2 = tsum
                        term = ev * gw(i, m3) - csy3
                        tsum = sy3 + term
                        csy3 = (tsum - sy3) - term
                        sy3 = tsum
                        term = ev * gw(i, n_mod + m3) - cso3
                        tsum = so3 + term
                        cso3 = (tsum - so3) - term
                        so3 = tsum
                        term = ev * gw(i, m4) - csy4
                        tsum = sy4 + term
                        csy4 = (tsum - sy4) - term
                        sy4 = tsum
                        term = ev * gw(i, n_mod + m4) - cso4
                        tsum = so4 + term
                        cso4 = (tsum - so4) - term
                        so4 = tsum
                    end if
                end do
                if (e > ky) then
                    vb(e, 1, j1) = 0.0_wp
                    vb(e, 3, j1) = 0.0_wp
                else
                    vb(e, 1, j1) = real(sy1, WP)
                    vb(e, 3, j1) = real(ey1, WP)
                end if
                vb(e, 2, j1) = real(so1, WP)
                vb(e, 4, j1) = real(eo1, WP)
                if (e > ky) then
                    vb(e, 1, j2) = 0.0_wp
                    vb(e, 3, j2) = 0.0_wp
                else
                    vb(e, 1, j2) = real(sy2, WP)
                    vb(e, 3, j2) = real(ey2, WP)
                end if
                vb(e, 2, j2) = real(so2, WP)
                vb(e, 4, j2) = real(eo2, WP)
                if (e > ky) then
                    vb(e, 1, j3) = 0.0_wp
                    vb(e, 3, j3) = 0.0_wp
                else
                    vb(e, 1, j3) = real(sy3, WP)
                    vb(e, 3, j3) = real(ey3, WP)
                end if
                vb(e, 2, j3) = real(so3, WP)
                vb(e, 4, j3) = real(eo3, WP)
                if (e > ky) then
                    vb(e, 1, j4) = 0.0_wp
                    vb(e, 3, j4) = 0.0_wp
                else
                    vb(e, 1, j4) = real(sy4, WP)
                    vb(e, 3, j4) = real(ey4, WP)
                end if
                vb(e, 2, j4) = real(so4, WP)
                vb(e, 4, j4) = real(eo4, WP)
            end do
        end do
#endif
    end subroutine adj_template_bar

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
                if (t <= it_tesc(m)) c1 = 1
                g = 0.0_wp
                ge = 0.0_wp
                do q = pe_ptr(gp), pe_ptr(gp + 1) - 1
                    k = pe_k(q)
                    e = pe_e(q)
                    g = g + eff_val(k) * vb(e, c1, j)
                    if (neb_cols > 0) ge = ge + eff_val(k) * vb(e, c1 + 2, j)
                end do
                fac = 1.0_wp
                if (t <= nt_neb) fac = fesc(m)
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
    end subroutine adj_fold_gather

    !> L_bol and Q per point: w_p bar += lsum_bar wsfh(t) L(t) cz cl_p / sl_z + qbar(t) cz q_p; and their
    !> d / d dz (zbar: L in log, Q linearly).
    subroutine adj_lq(m0, nc, nt, nt_neb, n_pc_max, n_pc, pt_off, pc_off, pc_ptr, pt_age, pt_cl, pt_q, perm, viz, cz, &
                      zsg, wsfh, slum, lsum_bar, qbar, ntq, slz, sqz, wpb, zbar)
        integer, intent(in) :: m0, nc, nt, nt_neb, n_pc_max, ntq
        integer, intent(in), contiguous :: n_pc(:), pt_off(:), pc_off(:), pc_ptr(:), pt_age(:), perm(:), viz(:)
        real(WP), intent(in), contiguous :: pt_cl(:), pt_q(:), cz(:), zsg(:), wsfh(:,:), slum(:,:), lsum_bar(:), &
                                            qbar(:,:), slz(:,:), sqz(:,:)
        real(WP), intent(inout), contiguous :: wpb(:,:), zbar(:)
        integer :: q, j, k, k0, n, iz, m, o, t
        real(WP) :: lf, qf, zs
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
                    if (slz(t, j) > SAFE_FLOOR) lf = lsum_bar(m) * wsfh(m, t) * slum(t, m) * cz(m0 + j - 1) / slz(t, j)
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
                                n_pts_max, w, wk, tot, pps, perm, viz, cz, smass, slum, q0)
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
        integer  :: j, t, k, m, q, k0, n, iz, o, kl
        real(WP) :: sm, sl, sq

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do 
#elif defined(_OPENMP)
        !$omp parallel do 
#endif
        do j = 1, nc
            tot(j) = imf_integral_p(imf_type, pm(:, perm(m0 + j - 1)), pm(IMF_P_LOWER_LIMIT, perm(m0 + j - 1)), &
                                    pm(IMF_P_UPPER_LIMIT, perm(m0 + j - 1)), .true.)
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, m, k0, n, sm, sl, sq, iz, o, kl)
#elif defined(_OPENMP)
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
                          veuv)
        integer, intent(in) :: m0, nc, n_eff_max, n_seg_max, nt_neb, neb_cols
        real(WP), intent(in), contiguous :: wsfh(:,:), fesc(:), eff_val(:), cz(:)
        real(KP), intent(in), contiguous :: w(:,:)
        integer, intent(in), contiguous  :: it_tesc(:), n_seg(:), n_eff(:), pt_off(:), seg_off(:), eff_off(:), ent_off(:), &
                                            pt_age(:), eff_pt(:), seg_ptr(:), seg_first(:), perm(:), viz(:)
        real(WP), intent(inout), contiguous :: vp(:,:,:)
        real(KP), intent(inout), contiguous :: v(:,:), veuv(:,:)
        integer  :: s, e, j, k, t, m, tesc, tc, iz, sg, eg
        real(WP) :: x, y, o, ye, oe, f, a, ae

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
        integer, parameter :: MT = 8, PT = 4, PIX_BLOCK = 256, KB = 4
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
#ifdef FSPS_CUBLAS
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

#ifdef FSPS_CUBLAS
#ifdef KERNEL_SINGLE_PRECISION
    !> gemm_columns for few columns (FP32 kernels): the full blocks of KB templates in one strided
    !> batched GEMM into a work array, the remainder block in one more, then each (pixel, column)
    !> adds the blocks in order with compensation (the arithmetic of gemm_columns).
    subroutine gemm_columns_batched(np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
        use opensps_batch_photometry, only: blas
        use cublas
        use, intrinsic :: iso_c_binding, only: c_long_long
        integer, intent(in) :: np, nspec, n_eff, k, nc, v0, c0
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
        integer, parameter :: KB = 256
        integer :: istat, nfull, rem, nb, i, j, b
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
            !$omp target data use_device_addr(etpl, v, gtmp)
            istat = cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, np, nc, KB, 1.0_KP, etpl, nspec, &
                                              int(KB, c_long_long) * nspec, v(1, v0 + 1), n_eff, int(KB, c_long_long), &
                                              0.0_KP, gtmp, size(gtmp, 1), int(size(gtmp, 1), c_long_long) * size(gtmp, 2), &
                                              nfull)
            !$omp end target data
            if (istat /= 0) error stop 'ssp batch: cuBLAS batched GEMM failed'
        end if
        if (rem > 0) then
            !$omp target data use_device_addr(etpl, v, gtmp)
            istat = cublasSgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_N, np, nc, rem, 1.0_KP, etpl(1, nfull * KB + 1), nspec, &
                                   v(nfull * KB + 1, v0 + 1), n_eff, 0.0_KP, gtmp(1, 1, nb), size(gtmp, 1))
            !$omp end target data
            if (istat /= 0) error stop 'ssp batch: cuBLAS GEMM failed'
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

    subroutine free_gtmp()
        if (allocated(gtmp)) then
            if (size(gtmp) > 1) then
                !$omp target exit data map(delete: gtmp)
            end if
            deallocate(gtmp)
        end if
    end subroutine free_gtmp
#endif

    !> template_columns with cuBLAS (arrays on the device). FP64: one GEMM. FP32: a GEMM per
    !> block of KB templates into a work array, each added to yo with a compensation term.
    subroutine gemm_columns(np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
        use opensps_batch_photometry, only: blas, blas_init
        use cublas
        integer, intent(in) :: np, nspec, n_eff, k, nc, v0, c0
        real(KP), intent(in), contiguous, target :: etpl(:,:), v(:,:)
        real(KP), intent(inout), contiguous, target :: yo(:,:)
        integer, parameter :: KB = 256
        integer :: istat, e0, kb1, i, j
        real(KP), allocatable :: tmp(:,:), cmp(:,:)
        real(KP) :: term, tsum

        call blas_init()
#ifdef KERNEL_SINGLE_PRECISION
        if (nc <= NC_BATCHED .and. k > KB) then
            ! Small column counts (the runs of one metallicity in a small batch): the blocks of KB
            ! templates as one strided batched GEMM and one compensated sum over the blocks, instead of
            ! a GEMM and an add per block (launch bound: ~20 launches per product)
            call gemm_columns_batched(np, nspec, n_eff, k, nc, etpl, v, v0, yo, c0)
            return
        end if
        allocate(tmp(np, nc), cmp(np, nc))
        !$omp target data map(alloc: tmp, cmp)
        !$omp target teams distribute parallel do collapse(2)
        do j = 1, nc
            do i = 1, np
                yo(i, c0 + j) = 0.0_KP
                cmp(i, j) = 0.0_KP
            end do
        end do
        do e0 = 1, k, KB
            kb1 = min(KB, k - e0 + 1)
            !$omp target data use_device_addr(etpl, v, tmp)
            istat = cublasSgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_N, np, nc, kb1, 1.0_KP, etpl(1, e0), nspec, &
                                   v(e0, v0 + 1), n_eff, 0.0_KP, tmp, np)
            !$omp end target data
            if (istat /= 0) error stop 'ssp batch: cuBLAS GEMM failed'
            !$omp target teams distribute parallel do collapse(2) private(term, tsum)
            do j = 1, nc
                do i = 1, np
                    term = tmp(i, j) - cmp(i, j)
                    tsum = yo(i, c0 + j) + term
                    cmp(i, j) = (tsum - yo(i, c0 + j)) - term
                    yo(i, c0 + j) = tsum
                end do
            end do
        end do
        !$omp end target data
#else
        !$omp target data use_device_addr(etpl, v, yo)
        istat = cublasDgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_N, np, nc, k, 1.0_KP, etpl, nspec, &
                               v(1, v0 + 1), n_eff, 0.0_KP, yo(1, c0 + 1), size(yo, 1))
        !$omp end target data
        if (istat /= 0) error stop 'ssp batch: cuBLAS GEMM failed'
#endif
    end subroutine gemm_columns
#endif

end module opensps_ssp_batch
