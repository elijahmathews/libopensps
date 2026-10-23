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
    use opensps_imf, only: imf_integral_p, imf_bin_weights, IMF_P_LOWER_LIMIT, IMF_P_UPPER_LIMIT
    use opensps_stellar_modifications, only: add_remnant_mass_p
    use opensps_ssp_operator, only: ssp_operator_t, ssp_operator_to_device, ssp_operator_from_device
    implicit none
    private

    public :: ssp_batch_age_sums, ssp_batch_age_sums_z
    public :: ssp_opset_t, ssp_opset_build, ssp_opset_free

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
        logical :: on_device = .false.
    end type ssp_opset_t

contains

    !> Young and old age sums yo (pixels x 2N: young 1:N, old N+1:2N), and per age and model the
    !> surviving mass smass and linear L_bol slum (ages x N) and the ionizing photons q0 (nebular
    !> ages x N), for the IMF parameters pm(:, m) of each model and its SFH weights wsfh (N x
    !> ages, FP64, zero where the batch zeroes them).
    !>
    !> The operator arrays are those of ssp_operator_t (batch form); dev: they and the work arrays
    !> are on the device (OpenMP target). it_tesc: last young age per model; nt_neb: number of
    !> nebular ages (0: none), fesc: their escaping EUV fraction per model, whl: EUV pixels.
    subroutine ssp_batch_age_sums(dev, imf_type, remnants, n_mod, nt, nspec, n_pts, n_eff, pm, wsfh, it_tesc, &
                                  pt_ptr, pt_mini, pt_m1, pt_m2, pt_first, pt_cm, pt_cl, pt_q, age_mmax, &
                                  pt_age, eff_ptr, eff_pt, eff_val, k_age, n_seg, seg_ptr, seg_first, n_pc, pc_ptr, age_pc, etpl, &
                                  nt_neb, whl, fesc, &
                                  yo, smass, slum, q0)
        logical, intent(in)  :: dev, remnants
        integer, intent(in)  :: imf_type, n_mod, nt, nspec, n_pts, n_eff, nt_neb, whl, n_seg, n_pc
        real(WP), intent(in), contiguous :: pm(:,:), wsfh(:,:), fesc(:)
        integer, intent(in), contiguous  :: it_tesc(:), pt_ptr(:), pt_first(:), k_age(:), seg_ptr(:), seg_first(:)
        integer, intent(in), contiguous  :: pc_ptr(:), age_pc(:)
        integer, intent(in), contiguous  :: pt_age(:), eff_ptr(:), eff_pt(:)
        real(WP), intent(in), contiguous :: pt_mini(:), pt_m1(:), pt_m2(:), pt_cm(:), pt_cl(:), pt_q(:), age_mmax(:)
        real(WP), intent(in), contiguous :: eff_val(:)
        real(KP), intent(in), contiguous :: etpl(:,:)
        real(KP), intent(inout), contiguous :: yo(:,:)
        real(WP), intent(inout), contiguous :: smass(:,:), slum(:,:), q0(:,:)

        !> Models per chunk: bounded by the IMF weights (n_pts x chunk FP64, ~1.7 GB at 2048 models
        !> of MIST)
        integer, parameter :: CHUNK = 2048
        real(WP), allocatable :: w(:,:), tot(:), vp(:,:,:), pps(:,:,:)
        real(KP), allocatable :: wk(:,:), v(:,:), veuv(:,:)
        integer :: m0, nc, ky, neb_cols, j
        logical :: useg
        integer, allocatable :: perm(:)
        real(WP), allocatable :: cz(:)

        neb_cols = 0
        if (nt_neb > 0 .and. whl > 0) neb_cols = 1
        allocate(w(min(CHUNK, n_mod), n_pts), tot(min(CHUNK, n_mod)), v(n_eff, 2 * min(CHUNK, n_mod)))
        allocate(veuv(n_eff, max(1, neb_cols * 2 * min(CHUNK, n_mod))))
        ! The IMF weights at kernel precision for the fold (half the memory traffic with FP32 kernels;
        ! the mass, L_bol and photon sums take the FP64 weights)
        allocate(wk(min(CHUNK, n_mod), n_pts))
        ! Device: the fold by segments of the template lists (4 x n_seg x chunk FP64 partial sums,
        ! ~260 MB at 2048 models of MIST); faster than one thread per template at every batch size
        ! tried on the A100 (N = 1: 15 -> 10 ms, 100: 128 -> 85 us/model, 10^4: 28.0 -> 26.7 us/model)
        useg = dev
        ! (The models in their own order, with unit weight: see ssp_batch_age_sums_z)
        allocate(perm(n_mod), cz(n_mod))
        do j = 1, n_mod
            perm(j) = j
        end do
        cz = 1.0_wp
        if (useg) then
            allocate(vp(min(CHUNK, n_mod), 4, n_seg), pps(min(CHUNK, n_mod), 3, n_pc))
        else
            allocate(vp(1, 1, 1), pps(1, 1, 1))
        end if
#ifdef OMP_TARGET
        !$omp target data if(dev) map(alloc: w, wk, tot, v, veuv, vp, pps) map(to: perm, cz)
#endif
        do m0 = 1, n_mod, CHUNK
            nc = min(CHUNK, n_mod - m0 + 1)
            if (useg) then
                call imf_pieces_chunk(imf_type, remnants, m0, nc, nt, n_pc, nt_neb, pm, pt_ptr, pc_ptr, age_pc, pt_mini, &
                                      pt_m1, pt_m2, pt_first, pt_cm, pt_cl, pt_q, age_mmax, size(w, 1), n_pts, w, wk, &
                                      tot, pps, perm, cz, .false., smass, slum, q0)
            else
                call imf_ages_chunk(dev, imf_type, remnants, m0, nc, nt, nt_neb, pm, pt_ptr, pt_mini, pt_m1, pt_m2, &
                                    pt_first, pt_cm, pt_cl, pt_q, age_mmax, size(w, 1), n_pts, w, wk, tot, smass, slum, q0)
            end if
            if (useg) then
                call fold_segments(m0, nc, n_eff, n_seg, nt_neb, neb_cols, wsfh, it_tesc, fesc, pt_age, eff_pt, eff_val, &
                                   seg_ptr, seg_first, perm, cz, wk, vp, v, veuv)
            else
                call fold_chunk(dev, m0, nc, n_eff, nt_neb, neb_cols, wsfh, it_tesc, fesc, pt_age, eff_ptr, eff_pt, &
                                eff_val, wk, v, veuv)
            end if
            ! Young sums: only the templates of the ages up to the chunk's last young age
            ky = 0
            do j = m0, m0 + nc - 1
                ky = max(ky, it_tesc(perm(j)))
            end do
            ky = k_age(max(1, min(nt, ky)))
            call template_columns(dev, nspec, nspec, n_eff, ky, nc, etpl, v, 0, yo, m0 - 1)
            call template_columns(dev, nspec, nspec, n_eff, n_eff, nc, etpl, v, nc, yo, n_mod + m0 - 1)
            if (neb_cols > 0) then
                ! The EUV pixels again, the nebular ages' templates times the escaping fraction
                call template_columns(dev, whl, nspec, n_eff, ky, nc, etpl, veuv, 0, yo, m0 - 1)
                call template_columns(dev, whl, nspec, n_eff, n_eff, nc, etpl, veuv, nc, yo, n_mod + m0 - 1)
            end if
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine ssp_batch_age_sums

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
                 set%pt_m2(max(np, 1)), set%pt_cm(max(np, 1)), set%pt_cl(max(np, 1)), set%pt_q(max(np, 1)))
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
        call move_alloc(ops, set%ops)
    end subroutine ssp_opset_build

    subroutine ssp_opset_free(set)
        type(ssp_opset_t), intent(inout) :: set
        integer :: iz
#ifdef OMP_TARGET
        if (set%on_device) then
            !$omp target exit data map(delete: set%n_pc, set%n_seg, set%n_eff, set%pt_off, set%pc_off, set%seg_off, &
            !$omp     set%ent_off, set%eff_off, set%pt_first, set%pt_age, set%pc_ptr, set%age_pc, set%pt_ptr, set%seg_ptr, &
            !$omp     set%seg_first, set%eff_pt, set%pt_mini, set%pt_m1, set%pt_m2, set%pt_cm, set%pt_cl, set%pt_q, &
            !$omp     set%age_mmax, set%eff_val)
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
                                                set%eff_val, set%age_pc, set%pt_ptr, set%age_mmax)
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
        !$omp     set%age_mmax, set%eff_val)
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
    !> photons are blended linearly and L_bol in its logarithm, as for a basis. Device builds only.
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
#ifdef OMP_TARGET
        logical, parameter :: dev = .true.
#else
        logical, parameter :: dev = .false.
#endif
        integer :: nc_max, neb_cols, nz, pass, m, k, t, ntq
        integer, allocatable :: perm1(:), perm2(:), viz1(:), viz2(:), ptr1(:), ptr2(:), cnt(:)
        real(WP), allocatable :: cz1(:), cz2(:), w(:,:), tot(:), vp(:,:,:), pps(:,:,:)
        real(KP), allocatable :: wk(:,:), v(:,:), veuv(:,:), yz(:,:)

        nz = set%nz
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
                 pps(nc_max, 3, set%n_pc_max), yz(nspec, 2 * nc_max))
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
                                      set%pt_first, set%pt_cm, set%pt_cl, set%pt_q, set%age_mmax, size(w, 1), &
                                      set%n_pts_max, w, wk, tot, pps, perm, viz, cz, smass, slum, q0)
                call fold_multi(m0, nc, set%n_eff_max, set%n_seg_max, nt_neb, neb_cols, wsfh, it_tesc, fesc, &
                                set%n_seg, set%n_eff, set%pt_off, set%seg_off, set%eff_off, set%ent_off, set%pt_age, &
                                set%eff_pt, set%eff_val, set%seg_ptr, set%seg_first, perm, viz, cz, wk, vp, v, veuv)
                ! The template products, per run of one metallicity in the chunk
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
                    call template_columns(dev, nspec, nspec, size(v, 1), ky, nrun, set%ops(iz)%etpl, v, j0 - 1, yz, j0 - 1)
                    call template_columns(dev, nspec, nspec, size(v, 1), set%ops(iz)%n_eff, nrun, set%ops(iz)%etpl, v, &
                                          nc + j0 - 1, yz, nc + j0 - 1)
                    if (neb_cols > 0) then
                        call template_columns(dev, whl, nspec, size(v, 1), ky, nrun, set%ops(iz)%etpl, veuv, j0 - 1, &
                                              yz, j0 - 1)
                        call template_columns(dev, whl, nspec, size(v, 1), set%ops(iz)%n_eff, nrun, set%ops(iz)%etpl, &
                                              veuv, nc + j0 - 1, yz, nc + j0 - 1)
                    end if
                    j0 = jr + 1
                end do
                call add_columns(m0, nc, nspec, n_mod, perm, yz, yo)
            end do
        end subroutine run_pass

    end subroutine ssp_batch_age_sums_z

    !> imf_pieces_chunk for the virtual models m0 .. m0 + nc - 1 of a pass (model perm(v), metallicity
    !> viz(v)) with the tables of an ssp_opset_t: the thread (piece q, virtual model) takes the piece
    !> and the points of its metallicity's table (offsets pt_off, pc_off) and skips it if the
    !> metallicity has fewer pieces; the sums accumulate (cz(m)) into smass, slum (logarithm) and q0.
    subroutine imf_pieces_multi(imf_type, remnants, m0, nc, nt, n_pc_max, nt_neb, pm, n_pc, pt_off, pc_off, pc_ptr, &
                                age_pc, pt_ptr, pt_mini, pt_m1, pt_m2, pt_first, pt_cm, pt_cl, pt_q, age_mmax, ldw, &
                                n_pts_max, w, wk, tot, pps, perm, viz, cz, smass, slum, q0)
        logical, intent(in) :: remnants
        integer, intent(in) :: imf_type, m0, nc, nt, n_pc_max, nt_neb, ldw, n_pts_max
        real(WP), intent(in), contiguous :: pm(:,:), pt_mini(:), pt_m1(:), pt_m2(:), pt_cm(:), pt_cl(:), pt_q(:), &
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
                call imf_bin_weights(imf_type, pm(:, m), tot(j), n, pt_mini(k0:k0 + n - 1), &
                                     pt_m1(k0:k0 + n - 1), pt_m2(k0:k0 + n - 1), pt_first(k0:k0 + n - 1), &
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

    !> IMF weights w(j, p) of models m = m0 .. m0+nc-1 (j = m - m0 + 1) at every isochrone point p
    !> (imf_bin_weights, as the SSP grid), and per age and model the surviving mass (with the
    !> remnants, as ssp_operator_evaluate), the linear L_bol and for the nebular ages the ionizing
    !> photon rate. One thread per (age, model): the age's points in order (each bin edge's power
    !> computed once), then their sums.
    subroutine imf_ages_chunk(dev, imf_type, remnants, m0, nc, nt, nt_neb, pm, pt_ptr, pt_mini, pt_m1, pt_m2, &
                              pt_first, pt_cm, pt_cl, pt_q, age_mmax, ldw, n_pts, w, wk, tot, smass, slum, q0)
        logical, intent(in) :: dev, remnants
        integer, intent(in) :: imf_type, m0, nc, nt, nt_neb
        real(WP), intent(in), contiguous :: pm(:,:), pt_mini(:), pt_m1(:), pt_m2(:), pt_cm(:), pt_cl(:), pt_q(:), &
                                            age_mmax(:)
        integer, intent(in), contiguous  :: pt_ptr(:), pt_first(:)
        integer, intent(in) :: ldw, n_pts
        real(WP), intent(inout) :: w(ldw, n_pts)
        real(KP), intent(inout) :: wk(ldw, n_pts)
        real(WP), intent(inout), contiguous :: tot(:), smass(:,:), slum(:,:), q0(:,:)
        integer  :: j, t, k, m, k0, n
        real(WP) :: sm, sl, sq

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do if(target: dev)
#elif defined(_OPENMP)
        !$omp parallel do
#endif
        do j = 1, nc
            tot(j) = imf_integral_p(imf_type, pm(:, m0 + j - 1), pm(IMF_P_LOWER_LIMIT, m0 + j - 1), &
                                    pm(IMF_P_UPPER_LIMIT, m0 + j - 1), .true.)
        end do
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, m, k0, n, sm, sl, sq) map(tofrom: q0) &
        !$omp if(target: dev)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, m, k0, n, sm, sl, sq)
#endif
        do t = 1, nt
            do j = 1, nc
                m = m0 + j - 1
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                k0 = pt_ptr(t)
                n = pt_ptr(t + 1) - k0
                if (n > 0) call imf_bin_weights(imf_type, pm(:, m), tot(j), n, pt_mini(k0:k0 + n - 1), &
                                                pt_m1(k0:k0 + n - 1), pt_m2(k0:k0 + n - 1), pt_first(k0:k0 + n - 1), &
                                                w(j, k0), ldw)
                do k = k0, k0 + n - 1
                    sm = sm + pt_cm(k) * w(j, k)
                    sl = sl + pt_cl(k) * w(j, k)
                    sq = sq + pt_q(k) * w(j, k)
                    wk(j, k) = real(w(j, k), KP)
                end do
                ! An empty isochrone has no mass (and no remnants), as ssp_operator_evaluate
                if (remnants .and. pt_ptr(t + 1) > pt_ptr(t)) call add_remnant_mass_p(imf_type, pm(:, m), sm, age_mmax(t))
                smass(t, m) = sm
                slum(t, m) = sl
                if (t <= nt_neb) q0(t, m) = sq
            end do
        end do
    end subroutine imf_ages_chunk

    !> imf_ages_chunk on the device with the points of each age cut into pieces (ssp_operator_t:
    !> pc_ptr, age_pc): one thread per (piece, model) makes the IMF weights of its points and their
    !> partial sums of mass, L_bol and photons, then one thread per (age, model) adds the pieces in
    !> order (and the remnants). The weights are the same as imf_ages_chunk's (a piece starts with
    !> both edges' powers computed, which are the values the previous bin would have handed over);
    !> the sums differ in FP64 rounding only.
    subroutine imf_pieces_chunk(imf_type, remnants, m0, nc, nt, n_pc, nt_neb, pm, pt_ptr, pc_ptr, age_pc, pt_mini, &
                                pt_m1, pt_m2, pt_first, pt_cm, pt_cl, pt_q, age_mmax, ldw, n_pts, w, wk, tot, pps, &
                                perm, cz, accumulate, smass, slum, q0)
        logical, intent(in) :: remnants, accumulate
        integer, intent(in) :: imf_type, m0, nc, nt, n_pc, nt_neb
        integer, intent(in), contiguous :: perm(:)
        real(WP), intent(in), contiguous :: cz(:)
        real(WP), intent(in), contiguous :: pm(:,:), pt_mini(:), pt_m1(:), pt_m2(:), pt_cm(:), pt_cl(:), pt_q(:), &
                                            age_mmax(:)
        integer, intent(in), contiguous  :: pt_ptr(:), pc_ptr(:), age_pc(:), pt_first(:)
        integer, intent(in) :: ldw, n_pts
        real(WP), intent(inout) :: w(ldw, n_pts)
        real(KP), intent(inout) :: wk(ldw, n_pts)
        real(WP), intent(inout), contiguous :: tot(:), pps(:,:,:), smass(:,:), slum(:,:), q0(:,:)
        integer  :: j, t, k, m, q, k0, n
        real(WP) :: sm, sl, sq

        !$omp target teams distribute parallel do
        do j = 1, nc
            tot(j) = imf_integral_p(imf_type, pm(:, perm(m0 + j - 1)), pm(IMF_P_LOWER_LIMIT, perm(m0 + j - 1)), &
                                    pm(IMF_P_UPPER_LIMIT, perm(m0 + j - 1)), .true.)
        end do
        !$omp target teams distribute parallel do collapse(2) private(k, m, k0, n, sm, sl, sq)
        do q = 1, n_pc
            do j = 1, nc
                m = perm(m0 + j - 1)
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                k0 = pc_ptr(q)
                n = pc_ptr(q + 1) - k0
                call imf_bin_weights(imf_type, pm(:, m), tot(j), n, pt_mini(k0:k0 + n - 1), &
                                     pt_m1(k0:k0 + n - 1), pt_m2(k0:k0 + n - 1), pt_first(k0:k0 + n - 1), &
                                     w(j, k0), ldw)
                do k = k0, k0 + n - 1
                    sm = sm + pt_cm(k) * w(j, k)
                    sl = sl + pt_cl(k) * w(j, k)
                    sq = sq + pt_q(k) * w(j, k)
                    wk(j, k) = real(w(j, k), KP)
                end do
                pps(j, 1, q) = sm
                pps(j, 2, q) = sl
                pps(j, 3, q) = sq
            end do
        end do
        !$omp target teams distribute parallel do collapse(2) private(q, sm, sl, sq)
        do t = 1, nt
            do j = 1, nc
                m = perm(m0 + j - 1)
                sm = 0.0_wp
                sl = 0.0_wp
                sq = 0.0_wp
                do q = age_pc(t), age_pc(t + 1) - 1
                    sm = sm + pps(j, 1, q)
                    sl = sl + pps(j, 2, q)
                    sq = sq + pps(j, 3, q)
                end do
                ! An empty isochrone has no mass (and no remnants), as ssp_operator_evaluate
                if (remnants .and. pt_ptr(t + 1) > pt_ptr(t)) call add_remnant_mass_p(imf_type, pm(:, m), sm, age_mmax(t))
                if (accumulate) then
                    ! A blend over the operators of several metallicities (ssp_batch_age_sums_z): the
                    ! mass and the photon rate linearly, L_bol in the logarithm (as the interpolated
                    ! SSP of compute_csp_batch with a basis); slum holds the logarithm until the end
                    smass(t, m) = smass(t, m) + cz(m) * sm
                    slum(t, m) = slum(t, m) + cz(m) * log10(max(sl, SAFE_FLOOR))
                    if (t <= nt_neb) q0(t, m) = q0(t, m) + cz(m) * sq
                else
                    smass(t, m) = sm
                    slum(t, m) = sl
                    if (t <= nt_neb) q0(t, m) = sq
                end if
            end do
        end do
    end subroutine imf_pieces_chunk

    !> Template weights of the chunk's models, the SFH folded in: v(e, j) young and v(e, nc + j)
    !> old sums over the points of template e of S(e, p) w(j, p) w_sfh(age(p), m); with nebular
    !> emission also veuv, the same with the nebular ages' terms times fesc(m). One thread per
    !> (template, model); the template's points are in age order: per age the sum of
    !> S(e, p) w(j, p) over its points, then times the model's SFH weight. FP64 sums, stored at
    !> kernel precision. (Memory-bound: w is read once per template entry.)
    subroutine fold_chunk(dev, m0, nc, n_eff, nt_neb, neb_cols, wsfh, it_tesc, fesc, pt_age, eff_ptr, eff_pt, &
                          eff_val, w, v, veuv)
        logical, intent(in) :: dev
        integer, intent(in) :: m0, nc, n_eff, nt_neb, neb_cols
        real(WP), intent(in), contiguous :: wsfh(:,:), fesc(:), eff_val(:)
        real(KP), intent(in), contiguous :: w(:,:)
        integer, intent(in), contiguous  :: it_tesc(:), pt_age(:), eff_ptr(:), eff_pt(:)
        real(KP), intent(inout), contiguous :: v(:,:), veuv(:,:)
        integer  :: e, j, k, t, m, tesc, tc
        real(WP) :: x, y, o, ye, oe, f, a, ae

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, t, m, tesc, tc, x, y, o, ye, oe, f, a, ae) &
        !$omp map(to: fesc) if(target: dev)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(k, t, m, tesc, tc, x, y, o, ye, oe, f, a, ae)
#endif
        do e = 1, n_eff
            do j = 1, nc
                m = m0 + j - 1
                tesc = it_tesc(m)
                f = fesc(m)
                y = 0.0_wp; o = 0.0_wp; ye = 0.0_wp; oe = 0.0_wp
                tc = 0
                a = 0.0_wp
                do k = eff_ptr(e), eff_ptr(e + 1)   ! (one past the end: flushes the last age)
                    if (k < eff_ptr(e + 1)) then
                        t = pt_age(eff_pt(k))
                    else
                        t = -1
                    end if
                    if (t /= tc) then
                        if (tc > 0) then
                            x = a * wsfh(m, tc)
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
                v(e, j) = real(y, KP)
                v(e, nc + j) = real(o, KP)
                if (neb_cols > 0) then
                    veuv(e, j) = real(ye, KP)
                    veuv(e, nc + j) = real(oe, KP)
                end if
            end do
        end do
    end subroutine fold_chunk

    !> fold_chunk with the lists of the templates cut into segments (ssp_operator_t: seg_ptr,
    !> seg_first), for small batches on the device: one thread per (segment, model) sums its ages
    !> (the segments end at age boundaries) into vp(model, 1:4, segment) = young, old, young and old
    !> with the escaping fraction; then one thread per (template, model) adds its segments in
    !> order. The same sums as fold_chunk with the ages grouped differently (FP64 rounding only).
    subroutine fold_segments(m0, nc, n_eff, n_seg, nt_neb, neb_cols, wsfh, it_tesc, fesc, pt_age, eff_pt, eff_val, &
                             seg_ptr, seg_first, perm, cz, w, vp, v, veuv)
        integer, intent(in) :: m0, nc, n_eff, n_seg, nt_neb, neb_cols
        integer, intent(in), contiguous :: perm(:)
        real(WP), intent(in), contiguous :: cz(:)
        real(WP), intent(in), contiguous :: wsfh(:,:), fesc(:), eff_val(:)
        real(KP), intent(in), contiguous :: w(:,:)
        integer, intent(in), contiguous  :: it_tesc(:), pt_age(:), eff_pt(:), seg_ptr(:), seg_first(:)
        real(WP), intent(inout), contiguous :: vp(:,:,:)
        real(KP), intent(inout), contiguous :: v(:,:), veuv(:,:)
        integer  :: s, e, j, k, t, m, tesc, tc
        real(WP) :: x, y, o, ye, oe, f, a, ae

#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(k, t, m, tesc, tc, x, y, o, ye, oe, f, a, ae) &
        !$omp map(to: fesc)
#endif
        do s = 1, n_seg
            do j = 1, nc
                m = perm(m0 + j - 1)
                tesc = it_tesc(m)
                f = fesc(m)
                y = 0.0_wp; o = 0.0_wp; ye = 0.0_wp; oe = 0.0_wp
                tc = 0
                a = 0.0_wp
                do k = seg_ptr(s), seg_ptr(s + 1)   ! (one past the end: flushes the last age)
                    if (k < seg_ptr(s + 1)) then
                        t = pt_age(eff_pt(k))
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
        !$omp target teams distribute parallel do collapse(2) private(s, y, o, ye, oe)
#endif
        do e = 1, n_eff
            do j = 1, nc
                y = 0.0_wp; o = 0.0_wp; ye = 0.0_wp; oe = 0.0_wp
                do s = seg_first(e), seg_first(e + 1) - 1
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
    end subroutine fold_segments

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
        integer, parameter :: MT = 8, PT = 4, PIX_BLOCK = 256
        integer  :: i, j, e, mq, ip, npt, l, p, mm(MT), ii(PT), p0, p1, nb, q
        real(KP) :: bk(PT), wk(MT), acc(PT, MT), cmp(PT, MT), term, tsum
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
            !$omp target teams loop collapse(2) private(mm, ii, bk, wk, acc, cmp, e, l, p, term, tsum)
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
                                if (KP == WP) then
                                    acc(p, l) = acc(p, l) + wk(l) * bk(p)
                                else
                                    term = wk(l) * bk(p) - cmp(p, l)
                                    tsum = acc(p, l) + term
                                    cmp(p, l) = (tsum - acc(p, l)) - term
                                    acc(p, l) = tsum
                                end if
                            end do
                        end do
                    end do
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
