module fsps_ssp_operator
    !> @brief The SSP as a sparse, IMF-independent operator applied to IMF weights (batched SSP,
    !> first step: one metallicity, on the CPU).
    !>
    !> @details
    !> **What it computes.** generate_ssp_grid sums, for every age t, the stars of the modified
    !> isochrone: each star j has a weight w_j and contributes w_j times at most four library
    !> templates (get_stellar_terms), w_j times the dust-shell AGB terms, or w_j times its own
    !> spectrum (stars that are not a template combination). Every isochrone modification is
    !> linear in the IMF weights w_IMF of the native isochrone points and otherwise independent
    !> of them:
    !>   - native point j:            w_j = f_j w_IMF(j)  (f_j: (1 - fbhb) on the HB, agb, pagb,
    !>                                                     redgb and TP-AGB normalisation factors)
    !>   - blue HB star i from j:     w_i = f_i w_IMF(j)
    !>   - blue straggler i:          w_i = g_i sum_{k on the HB} w_IMF(k)   (g_i = sbss / 20)
    !> so the template weights W of all ages are W = S . w_IMF, with S fixed once the isochrones,
    !> the spectral libraries and the non-IMF parameters are. This module builds S once
    !> (ssp_operator_build) and evaluates W, masses and L_bol for any IMF parameter set from it
    !> (ssp_op_model_weights), then the spectra with the same template sums as generate_ssp_grid
    !> (ssp_operator_evaluate).
    !>
    !> **Structure of S** (ssp_operator_t). Columns: the native isochrone points of all computed
    !> ages (pt_*; age t owns pt_ptr(t) .. pt_ptr(t+1) - 1), whose IMF weights are the closed-form
    !> bin integrals of compute_imf_weights. Rows (CSR: row_ptr, col, val), grouped by age
    !> (age_row_ptr), each with a kind and an index:
    !>   - ROW_MAIN:   main-library template (index (logg - 1) * nt_lib + teff, speclib order);
    !>   - ROW_SMALL:  WMBasic, post-AGB, WN, WC, carbon-star, O- and C-rich AGB templates
    !>                 (concatenated in that order, as the OpenACC deferred rows of fsps_ssp);
    !>   - ROW_DUSTY:  a template (main, carbon, O-rich, C-rich) of one dust-shell corner (slot):
    !>                 dust-shell AGB stars are bilinear in templates and the shell's transfer
    !>                 function, so their template weights are kept per corner of the dust grid
    !>                 (slot_key: C-rich flag, Teff index, tau index), with the corner weight in val;
    !>   - ROW_DIRECT: one star whose spectrum is not a template combination (direct_spec);
    !>   - ROW_MASS:   sum of w_j * current mass (surviving stellar mass, before remnants);
    !>   - ROW_LBOL:   sum of w_j * L_j (linear L_bol).
    !> Entry values are factor x template coefficient (x dust-corner weight). The remnant mass
    !> needs only the turn-off mass of each age (age_mmax) and the IMF (add_remnant_mass_p).
    !> The ionizing photon rate is computed downstream from the spectra, which are linear in W;
    !> nebular emission, X-ray binaries and smoothing are applied to the spectra afterwards
    !> (finish_ssp_grid), as in generate_ssp_grid.
    !>
    !> **What S depends on** (rebuild it when any of these change): the metallicity zmet and
    !> the isochrone / spectral-library / dust-shell data of the context; the isochrone
    !> modifications fbhb, sbss, delt, dell, agb, pagb, redgb and tpagb_norm_type; evtype and
    !> masscut (which stars are summed); agb_dust and add_agb_dust_model (the dust shells);
    !> use_wr_spectra, logt_wmb_hot and the other library-selection switches of
    !> get_stellar_terms; and the ages computed (ssp_gen_age). It does not depend on any IMF
    !> parameter: imf_type, imf1-3, vdmc, mdave, imf_lower_limit, imf_upper_limit, or a user IMF
    !> file (the bin edges of the points are the isochrone masses; the first point's lower edge,
    !> imf_lower_bound, and the mass limits are applied per IMF). The IMF limits only zero the
    !> weights of points outside them, so S keeps all points.
    !>
    !> **Per model** (ssp_op_model_weights; plain arrays, no allocation, no derived types, so it
    !> can become a batched device kernel): ~n_pts IMF bin integrals (imf_integral_p, the
    !> closed forms of fsps_imf, unchanged), one sparse product over nnz entries, and the
    !> remnant integrals per age. The template product (spectra = templates . W) is shared with
    !> generate_ssp_grid (add_main_library_templates, sum_small_templates, sum_dusty_slots).
    !>
    !> **Agreement with generate_ssp_grid**: the aggregation order differs (w_IMF x (f x coef)
    !> instead of (w_IMF x f) x coef, entries merged per point, dust corners ordered by first use
    !> over all points instead of over points with non-zero weight), so W agrees to rounding
    !> (FP64: ~1e-15 relative); the spectra, masses and L_bol to the same level, FP32 kernels to
    !> ~1e-7 (tests/test_ssp_operator.f90: all six IMF types, slopes, masses and limits changed,
    !> evaluated from one S; MIST and Padova, with and without isochrone modifications).
    !>
    !> **Measured costs** (tests/benchmark_ssp_operator; one core, gfortran -O3, FP64, Ryzen 7
    !> 7840U laptop; one metallicity, all ages):
    !>
    !>   | isochrones, library     | points | entries | generate_ssp_grid | build S | per model W | grid from S |
    !>   |-------------------------|--------|---------|-------------------|---------|-------------|-------------|
    !>   | MIST, MILES (107 ages)  | 105558 | 807037  | 113 ms            | 72 ms   | 4.7 ms      | 91 ms       |
    !>   | MIST, C3K (107 ages)    | 105558 | 819340  | 182 ms            | 45 ms   | 4.2 ms      | 159 ms      |
    !>   | Padova, BaSeL (94 ages) | 15605  | 93855   | 15 ms             | 5.1 ms  | 0.7 ms      | 13.5 ms     |
    !>
    !> "Per model W" (IMF weights of all points + S . w_IMF + the mass and L_bol rows) is ~4% of
    !> a full SSP grid; 85% of it is the IMF bin integrals (~38 ns each), the sparse product the
    !> rest. "Grid from S" adds the template product templates . W per age, which dominates and is
    !> what the next step removes: with the SFH folded first (W . sfh_weights) one product per
    !> model remains instead of one per age. MILES has 387 stars outside the template grids
    !> (direct spectra); C3K none.

    use fsps_precision, only: WP, KP
    use fsps_constants, only: NM, SAFE_FLOOR, TIME_RES_INCR
    use fsps_types, only: params
    use fsps_context_types, only: fsps_context_t, fsps_context_sync_device_settings
    use fsps_imf, only: imf_integral_p, IMF_NPAR, IMF_P_LOWER_LIMIT, IMF_P_UPPER_LIMIT, IMF_P_LOWER_BOUND
    use fsps_stellar_modifications, only: add_remnant_mass_p
    use fsps_ssp, only: isochrone_buffer_t, init_isochrone_buffer, free_isochrone_buffer, load_timestep_data, &
                        apply_isochrone_modifications, sum_small_templates, sum_dusty_slots, &
                        add_main_library_templates, finish_ssp_grid, add_column
    use fsps_spectral_library, only: get_stellar_spectrum, get_stellar_terms, SRC_MAIN_LIB, SRC_WMBASIC, &
                                     SRC_AGB_O, SRC_AGB_C, SRC_PAGB, SRC_WRN, SRC_WRC, SRC_AGB_CAR
    use fsps_dust, only: get_agb_dust_terms
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none
    private

    public :: ssp_operator_t, ssp_operator_build, ssp_operator_evaluate, ssp_operator_free
    public :: ssp_op_model_weights, ssp_op_imf_weights, ssp_op_spmv

    !> Row kinds (the order of the rows within an age)
    integer, parameter, public :: ROW_MAIN = 1, ROW_SMALL = 2, ROW_DUSTY = 3, ROW_DIRECT = 4, &
                                  ROW_MASS = 5, ROW_LBOL = 6

    real(WP), parameter :: LN10 = log(10.0_wp)
    !> Composite sort key of an entry: kind, row index, local point index (< 2**KEY_COL_BITS)
    integer, parameter :: KEY_COL_BITS = 12, KEY_IDX_BITS = 44

    type :: ssp_operator_t
        integer :: zmet = 0, n_ages = 0, nspec = 0
        integer :: nt_lib = 0, n_main = 0, n_small = 0, n_dtpl = 0
        integer :: n_pts = 0, n_rows = 0, nnz = 0, n_slots = 0, n_direct = 0
        !> Native isochrone points (columns): initial mass and IMF bin edges; for the first point
        !> of an age (pt_first = 1) the lower edge is imf_lower_bound, applied per IMF
        integer,  allocatable :: pt_ptr(:)
        real(WP), allocatable :: pt_mini(:), pt_m1(:), pt_m2(:)
        integer,  allocatable :: pt_first(:)
        !> Ages: computed (ssp_gen_age), number of stars after the modifications (0: empty
        !> isochrone), turn-off mass (largest native initial mass, for the remnants), rows
        logical,  allocatable :: age_on(:)
        integer,  allocatable :: age_nstars(:)
        real(WP), allocatable :: age_mmax(:)
        integer,  allocatable :: age_row_ptr(:), age_slot_ptr(:)
        !> Rows (CSR)
        integer,  allocatable :: row_ptr(:), row_kind(:), row_idx(:), col(:)
        real(WP), allocatable :: val(:)
        !> Dust-shell corners: flux_dagb indices (C-rich flag + 1, Teff, tau)
        integer,  allocatable :: slot_key(:,:)
        !> Spectra of the stars that are not template combinations
        real(KP), allocatable :: direct_spec(:,:)
        !> Library sizes (to unpack the small-library and dust-corner rows)
        integer :: n_wmb(2) = 0, n_pagb(2) = 0, n_wrn = 0, n_wrc = 0, n_car = 0, n_agbo = 0, n_agbc = 0
        integer :: ng_lib = 0
    end type ssp_operator_t

contains

    ! ------------------------------------------------------------------------
    ! BUILD (once per metallicity and non-IMF parameter set)
    ! ------------------------------------------------------------------------

    !> @brief Builds the IMF-independent operator S of metallicity pset%zmet.
    !> @details The context must be set up (fsps_setup). pset supplies the non-IMF parameters
    !> listed in the module notes; its IMF parameters are not used.
    subroutine ssp_operator_build(ctx, pset, op)
        type(fsps_context_t), target, intent(inout) :: ctx
        type(params), intent(in)              :: pset
        type(ssp_operator_t), intent(inout)   :: op

        type(isochrone_buffer_t) :: buf
        integer :: t, n_nat, n_all, n_after_hb, n_hb, j, k, i, nz, nspec, cap_pts
        logical :: hb_flag(NM)
        integer :: parent(NM)
        real(WP) :: probe(NM), tlog, lin_l
        ! Per-age entries (unsorted), slots and direct stars
        integer(int64), allocatable :: e_key(:)
        real(WP), allocatable :: e_val(:)
        integer :: ne
        integer, allocatable :: s_key(:,:)
        integer :: ns_age
        real(KP), allocatable :: d_spec(:,:), star(:)
        ! Global accumulators
        integer, allocatable :: g_row_ptr(:), g_kind(:), g_idx(:), g_col(:), g_slot_key(:,:)
        real(WP), allocatable :: g_val(:), g_mini(:), g_m1(:), g_m2(:)
        integer, allocatable :: g_first(:)
        ! Contributions (native point, factor) of the current star
        integer  :: c_k(NM), nc
        real(WP) :: c_f(NM)

        call ssp_operator_free(op)
        call fsps_context_sync_device_settings(ctx)
        if (trim(ctx%state%isoc_type) == 'bpss') then
            write(*,*) 'SSP_OPERATOR ERROR: BPASS SSPs are tabulated, there is no operator to build.'
            stop
        end if
        nz = pset%zmet
        nspec = ctx%state%nspec
        op%zmet = nz
        op%n_ages = ctx%state%nt
        op%nspec = nspec
        op%nt_lib = size(ctx%state%speclib, 3)
        op%ng_lib = size(ctx%state%speclib, 4)
        op%n_main = op%nt_lib * op%ng_lib
        op%n_wmb = [size(ctx%state%wmb_spec, 3), size(ctx%state%wmb_spec, 4)]
        op%n_pagb = [size(ctx%state%pagb_spec, 2), size(ctx%state%pagb_spec, 3)]
        op%n_wrn = size(ctx%state%wrn_spec, 2)
        op%n_wrc = size(ctx%state%wrc_spec, 2)
        op%n_car = size(ctx%state%agb_spec_car, 2)
        op%n_agbo = size(ctx%state%agb_spec_o, 2)
        op%n_agbc = size(ctx%state%agb_spec_c, 2)
        op%n_small = product(op%n_wmb) + product(op%n_pagb) + op%n_wrn + op%n_wrc + op%n_car + &
                     op%n_agbo + op%n_agbc
        op%n_dtpl = op%n_main + op%n_car + op%n_agbo + op%n_agbc

        allocate(op%pt_ptr(op%n_ages + 1), op%age_on(op%n_ages), op%age_nstars(op%n_ages), &
                 op%age_mmax(op%n_ages), op%age_row_ptr(op%n_ages + 1), op%age_slot_ptr(op%n_ages + 1))
        op%age_on = .false.; op%age_nstars = 0; op%age_mmax = -1.0_wp
        cap_pts = 4096
        allocate(g_mini(cap_pts), g_m1(cap_pts), g_m2(cap_pts), g_first(cap_pts))
        allocate(g_row_ptr(1025), g_kind(1024), g_idx(1024), g_col(16384), g_val(16384), g_slot_key(3, 64))
        allocate(e_key(16384), e_val(16384), s_key(3, 16), d_spec(nspec, 4), star(nspec))
        g_row_ptr(1) = 1
        op%n_pts = 0; op%n_rows = 0; op%nnz = 0; op%n_slots = 0; op%n_direct = 0

        call init_isochrone_buffer(buf)
        do t = 1, op%n_ages
            op%pt_ptr(t) = op%n_pts + 1
            op%age_row_ptr(t) = op%n_rows + 1
            op%age_slot_ptr(t) = op%n_slots + 1
            if (pset%ssp_gen_age(t) == 0) cycle
            op%age_on(t) = .true.
            tlog = ctx%state%timestep_isoc(nz, t)

            ! Native points and their IMF bin edges (as compute_imf_weights)
            call load_timestep_data(ctx, nz, t, buf)
            n_nat = buf%n_stars
            if (op%n_pts + n_nat > size(g_mini)) then
                cap_pts = max(op%n_pts + n_nat, 2 * size(g_mini))
                call grow_r1(g_mini, op%n_pts, cap_pts)
                call grow_r1(g_m1, op%n_pts, cap_pts)
                call grow_r1(g_m2, op%n_pts, cap_pts)
                call grow_i1(g_first, op%n_pts, cap_pts)
            end if
            do j = 1, n_nat
                k = op%n_pts + j
                g_mini(k) = buf%initial_mass(j)
                g_first(k) = merge(1, 0, j == 1)
                if (j == 1) then
                    g_m1(k) = 0.0_wp
                else
                    g_m1(k) = buf%initial_mass(j) - 0.5_wp * (buf%initial_mass(j) - buf%initial_mass(j-1))
                end if
                if (j == n_nat) then
                    g_m2(k) = buf%initial_mass(j)
                else
                    g_m2(k) = buf%initial_mass(j) + 0.5_wp * (buf%initial_mass(j+1) - buf%initial_mass(j))
                end if
            end do
            if (n_nat > 0) op%age_mmax(t) = maxval(buf%initial_mass(1:n_nat))

            ! Unit weights through the modifications: each star's factor(s)
            buf%weights(1:n_nat) = 1.0_wp
            hb_flag = .false.
            parent = 0
            call apply_isochrone_modifications(ctx, pset, tlog, buf, hb_flag, parent, n_after_hb)
            n_all = buf%n_stars
            op%age_nstars(t) = n_all
            probe(1:n_all) = buf%weights(1:n_all)
            n_hb = count(hb_flag(1:n_nat))

            ne = 0
            ns_age = 0
            do j = 1, n_all
                ! Contributions of the native points to star j's weight
                if (j <= n_nat) then
                    nc = 1; c_k(1) = j; c_f(1) = probe(j)
                else if (j <= n_after_hb) then
                    if (parent(j) < 1) then
                        write(*,*) 'SSP_OPERATOR ERROR: blue HB star without parent at age', t
                        stop
                    end if
                    nc = 1; c_k(1) = parent(j); c_f(1) = probe(j)
                else
                    ! Blue straggler: weight = g sum_{HB} w (probe: g n_hb)
                    nc = 0
                    if (n_hb > 0) then
                        do k = 1, n_nat
                            if (.not. hb_flag(k)) cycle
                            nc = nc + 1; c_k(nc) = k; c_f(nc) = probe(j) / real(n_hb, WP)
                        end do
                    end if
                end if
                if (nc == 0) cycle

                ! Mass and L_bol (all stars)
                lin_l = exp(buf%log_lum(j) * LN10)
                do i = 1, nc
                    call push(ROW_MASS, 1, c_k(i), c_f(i) * buf%current_mass(j))
                    call push(ROW_LBOL, 1, c_k(i), c_f(i) * lin_l)
                end do

                ! Spectrum (accumulate_spectrum's selection; zero-weight stars are skipped there)
                if (all(c_f(1:nc) == 0.0_wp)) cycle
                if (pset%evtype /= -1 .and. int(buf%phase(j)) /= pset%evtype) cycle
                if (buf%initial_mass(j) >= pset%masscut) cycle
                call star_terms(j)
            end do

            call flush_age(t)
        end do
        op%pt_ptr(op%n_ages + 1) = op%n_pts + 1
        op%age_row_ptr(op%n_ages + 1) = op%n_rows + 1
        op%age_slot_ptr(op%n_ages + 1) = op%n_slots + 1
        call free_isochrone_buffer(buf)

        allocate(op%pt_mini(op%n_pts), op%pt_m1(op%n_pts), op%pt_m2(op%n_pts), op%pt_first(op%n_pts))
        op%pt_mini = g_mini(1:op%n_pts); op%pt_m1 = g_m1(1:op%n_pts); op%pt_m2 = g_m2(1:op%n_pts)
        op%pt_first = g_first(1:op%n_pts)
        allocate(op%row_ptr(op%n_rows + 1), op%row_kind(op%n_rows), op%row_idx(op%n_rows), &
                 op%col(op%nnz), op%val(op%nnz), op%slot_key(3, max(op%n_slots, 1)), &
                 op%direct_spec(nspec, max(op%n_direct, 1)))
        op%row_ptr = g_row_ptr(1:op%n_rows + 1)
        op%row_kind = g_kind(1:op%n_rows); op%row_idx = g_idx(1:op%n_rows)
        op%col = g_col(1:op%nnz); op%val = g_val(1:op%nnz)
        op%slot_key = 0
        if (op%n_slots > 0) op%slot_key = g_slot_key(:, 1:op%n_slots)
        op%direct_spec = 0.0_KP
        if (op%n_direct > 0) op%direct_spec = d_spec(:, 1:op%n_direct)

    contains

        !> The template terms of star j (accumulate_by_templates, step 1) for its contributions.
        subroutine star_terms(j)
            integer, intent(in) :: j
            integer  :: src, nterm, i1(4), i2(4), rich, it, itau, d, n, slot, q, ii
            real(WP) :: coef(4), cd(4)
            logical  :: linear, dusty, active
            integer, parameter :: cdi(4) = [0, 1, 1, 0], cdj(4) = [0, 0, 1, 1]

            call get_stellar_terms(ctx, pset, buf%current_mass(j), buf%log_teff(j), &
                                   exp(buf%log_lum(j) * LN10), buf%log_g(j), buf%phase(j), &
                                   buf%co_ratio(j), buf%log_mdot(j), src, i1, i2, coef, nterm, linear, &
                                   allow_dust=associated(ctx%state%speclib_agb_smooth), dusty=dusty)
            if (linear .and. dusty) then
                call get_agb_dust_terms(ctx, pset%agb_dust, buf%current_mass(j), buf%log_teff(j), &
                                        buf%log_lum(j), buf%log_g(j), buf%co_ratio(j), buf%log_mdot(j), &
                                        rich, it, itau, cd, active)
                if (active .and. (src == SRC_MAIN_LIB .or. src == SRC_AGB_O .or. src == SRC_AGB_C .or. &
                                  src == SRC_AGB_CAR)) then
                    do d = 1, 4
                        if (cd(d) == 0.0_wp) cycle
                        slot = find_slot(rich + 1, it + cdi(d), itau + cdj(d))
                        do n = 1, nterm
                            select case (src)
                            case (SRC_MAIN_LIB); q = (i2(n) - 1) * op%nt_lib + i1(n)
                            case (SRC_AGB_CAR);  q = op%n_main + i1(n)
                            case (SRC_AGB_O);    q = op%n_main + op%n_car + i1(n)
                            case default;        q = op%n_main + op%n_car + op%n_agbo + i1(n)
                            end select
                            do ii = 1, nc
                                call push(ROW_DUSTY, (slot - 1) * op%n_dtpl + q, c_k(ii), c_f(ii) * coef(n) * cd(d))
                            end do
                        end do
                    end do
                    return
                else if (active) then
                    linear = .false.
                end if
            end if
            if (buf%log_lum(j) * LN10 < log(tiny(0.0_wp))) linear = .false.
            if (.not. linear) then
                call get_stellar_spectrum(ctx, pset, buf%current_mass(j), buf%log_teff(j), &
                                          exp(buf%log_lum(j) * LN10), buf%log_g(j), buf%phase(j), &
                                          buf%co_ratio(j), buf%log_mdot(j), star)
                op%n_direct = op%n_direct + 1
                if (op%n_direct > size(d_spec, 2)) call grow_k2(d_spec, op%n_direct - 1, 2 * size(d_spec, 2))
                d_spec(:, op%n_direct) = star
                do ii = 1, nc
                    call push(ROW_DIRECT, op%n_direct, c_k(ii), c_f(ii))
                end do
                return
            end if
            do n = 1, nterm
                select case (src)
                case (SRC_MAIN_LIB); q = (i2(n) - 1) * op%nt_lib + i1(n)
                case (SRC_WMBASIC);  q = (i2(n) - 1) * op%n_wmb(1) + i1(n)
                case (SRC_PAGB);     q = product(op%n_wmb) + (i2(n) - 1) * op%n_pagb(1) + i1(n)
                case (SRC_WRN);      q = product(op%n_wmb) + product(op%n_pagb) + i1(n)
                case (SRC_WRC);      q = product(op%n_wmb) + product(op%n_pagb) + op%n_wrn + i1(n)
                case (SRC_AGB_CAR);  q = product(op%n_wmb) + product(op%n_pagb) + op%n_wrn + op%n_wrc + i1(n)
                case (SRC_AGB_O);    q = product(op%n_wmb) + product(op%n_pagb) + op%n_wrn + op%n_wrc + &
                                         op%n_car + i1(n)
                case (SRC_AGB_C);    q = product(op%n_wmb) + product(op%n_pagb) + op%n_wrn + op%n_wrc + &
                                         op%n_car + op%n_agbo + i1(n)
                case default
                    cycle
                end select
                do ii = 1, nc
                    if (src == SRC_MAIN_LIB) then
                        call push(ROW_MAIN, q, c_k(ii), c_f(ii) * coef(n))
                    else
                        call push(ROW_SMALL, q, c_k(ii), c_f(ii) * coef(n))
                    end if
                end do
            end do
        end subroutine star_terms

        !> Local slot (this age) of a dust-grid corner, created on first use.
        integer function find_slot(r1, kt, ktau)
            integer, intent(in) :: r1, kt, ktau
            integer :: s
            do s = 1, ns_age
                if (s_key(1, s) == r1 .and. s_key(2, s) == kt .and. s_key(3, s) == ktau) then
                    find_slot = s
                    return
                end if
            end do
            if (ns_age == size(s_key, 2)) then
                call grow_i2(s_key, ns_age, 2 * ns_age)
            end if
            ns_age = ns_age + 1
            s_key(:, ns_age) = [r1, kt, ktau]
            find_slot = ns_age
        end function find_slot

        !> Appends entry (kind, row index, native point k of this age, value).
        subroutine push(kind, idx, k, v)
            integer, intent(in)  :: kind, idx, k
            real(WP), intent(in) :: v
            if (ne == size(e_key)) then
                call grow_i8(e_key, ne, 2 * ne)
                call grow_r1(e_val, ne, 2 * ne)
            end if
            ne = ne + 1
            e_key(ne) = ior(ishft(int(kind, int64), KEY_IDX_BITS + KEY_COL_BITS), &
                            ior(ishft(int(idx, int64), KEY_COL_BITS), int(k, int64)))
            e_val(ne) = v
        end subroutine push

        !> Sorts this age's entries by (row, point), merges duplicates and appends the rows.
        subroutine flush_age(t)
            integer, intent(in) :: t
            integer, allocatable :: perm(:)
            integer :: m, r0, kind, idx, k
            integer(int64) :: rkey, prev_rkey, prev_key
            integer(int64), parameter :: COL_MASK = 2_int64**KEY_COL_BITS - 1, IDX_MASK = 2_int64**KEY_IDX_BITS - 1

            ! Slots of this age become global
            do m = 1, ns_age
                op%n_slots = op%n_slots + 1
                if (op%n_slots > size(g_slot_key, 2)) call grow_i2(g_slot_key, op%n_slots - 1, 2 * size(g_slot_key, 2))
                g_slot_key(:, op%n_slots) = s_key(:, m)
            end do
            if (ne == 0) then
                op%n_pts = op%n_pts + n_nat
                return
            end if
            allocate(perm(ne))
            call sort_keys(e_key(1:ne), perm)
            prev_rkey = -1_int64
            prev_key = -1_int64
            r0 = op%age_slot_ptr(t) - 1
            do m = 1, ne
                rkey = ishft(e_key(perm(m)), -KEY_COL_BITS)
                if (e_key(perm(m)) == prev_key) then
                    g_val(op%nnz) = g_val(op%nnz) + e_val(perm(m))
                    cycle
                end if
                if (rkey /= prev_rkey) then
                    op%n_rows = op%n_rows + 1
                    if (op%n_rows + 1 > size(g_row_ptr)) then
                        call grow_i1(g_row_ptr, op%n_rows, 2 * size(g_row_ptr))
                        call grow_i1(g_kind, op%n_rows - 1, 2 * size(g_row_ptr))
                        call grow_i1(g_idx, op%n_rows - 1, 2 * size(g_row_ptr))
                    end if
                    kind = int(ishft(rkey, -KEY_IDX_BITS))
                    idx = int(iand(rkey, IDX_MASK))
                    if (kind == ROW_DUSTY) then
                        ! (local slot, template) -> (global slot, template)
                        idx = (r0 + (idx - 1) / op%n_dtpl) * op%n_dtpl + mod(idx - 1, op%n_dtpl) + 1
                    else if (kind == ROW_DIRECT) then
                        idx = idx   ! direct stars are numbered globally already
                    end if
                    g_kind(op%n_rows) = kind
                    g_idx(op%n_rows) = idx
                    g_row_ptr(op%n_rows) = op%nnz + 1
                    prev_rkey = rkey
                end if
                k = int(iand(e_key(perm(m)), COL_MASK))
                op%nnz = op%nnz + 1
                if (op%nnz > size(g_col)) then
                    call grow_i1(g_col, op%nnz - 1, 2 * size(g_col))
                    call grow_r1(g_val, op%nnz - 1, 2 * size(g_val))
                end if
                g_col(op%nnz) = op%n_pts + k
                g_val(op%nnz) = e_val(perm(m))
                g_row_ptr(op%n_rows + 1) = op%nnz + 1
                prev_key = e_key(perm(m))
            end do
            op%n_pts = op%n_pts + n_nat
        end subroutine flush_age

    end subroutine ssp_operator_build

    ! ------------------------------------------------------------------------
    ! Growing arrays (module procedures: with nvfortran 26.1 -acc, move_alloc on host-associated
    ! allocatables inside internal procedures corrupted the heap)
    ! ------------------------------------------------------------------------

    !> Grows a to size cap, keeping a(1:n_keep).
    subroutine grow_i1(a, n_keep, cap)
        integer, allocatable, intent(inout) :: a(:)
        integer, intent(in) :: n_keep, cap
        integer, allocatable :: b(:)
        allocate(b(cap))
        b(1:n_keep) = a(1:n_keep)
        call move_alloc(b, a)
    end subroutine grow_i1

    subroutine grow_i8(a, n_keep, cap)
        integer(int64), allocatable, intent(inout) :: a(:)
        integer, intent(in) :: n_keep, cap
        integer(int64), allocatable :: b(:)
        allocate(b(cap))
        b(1:n_keep) = a(1:n_keep)
        call move_alloc(b, a)
    end subroutine grow_i8

    subroutine grow_r1(a, n_keep, cap)
        real(WP), allocatable, intent(inout) :: a(:)
        integer, intent(in) :: n_keep, cap
        real(WP), allocatable :: b(:)
        allocate(b(cap))
        b(1:n_keep) = a(1:n_keep)
        call move_alloc(b, a)
    end subroutine grow_r1

    !> Grows the second dimension of a to cap, keeping a(:, 1:n_keep).
    subroutine grow_i2(a, n_keep, cap)
        integer, allocatable, intent(inout) :: a(:,:)
        integer, intent(in) :: n_keep, cap
        integer, allocatable :: b(:,:)
        allocate(b(size(a, 1), cap))
        b(:, 1:n_keep) = a(:, 1:n_keep)
        call move_alloc(b, a)
    end subroutine grow_i2

    subroutine grow_k2(a, n_keep, cap)
        real(KP), allocatable, intent(inout) :: a(:,:)
        integer, intent(in) :: n_keep, cap
        real(KP), allocatable :: b(:,:)
        allocate(b(size(a, 1), cap))
        b(:, 1:n_keep) = a(:, 1:n_keep)
        call move_alloc(b, a)
    end subroutine grow_k2

    !> Stable sort of keys (merge sort): perm such that keys(perm) is non-decreasing.
    subroutine sort_keys(keys, perm)
        integer(int64), intent(in) :: keys(:)
        integer, intent(out) :: perm(:)
        integer, allocatable :: tmp(:)
        integer :: n, width, lo, mid, hi, i, j, k
        n = size(keys)
        allocate(tmp(n))
        perm = [(i, i = 1, n)]
        width = 1
        do while (width < n)
            lo = 1
            do while (lo <= n)
                mid = min(lo + width - 1, n)
                hi = min(lo + 2 * width - 1, n)
                i = lo; j = mid + 1; k = lo
                do while (i <= mid .and. j <= hi)
                    if (keys(perm(j)) < keys(perm(i))) then
                        tmp(k) = perm(j); j = j + 1
                    else
                        tmp(k) = perm(i); i = i + 1
                    end if
                    k = k + 1
                end do
                do while (i <= mid)
                    tmp(k) = perm(i); i = i + 1; k = k + 1
                end do
                do while (j <= hi)
                    tmp(k) = perm(j); j = j + 1; k = k + 1
                end do
                lo = lo + 2 * width
            end do
            perm = tmp
            width = 2 * width
        end do
    end subroutine sort_keys

    subroutine ssp_operator_free(op)
        type(ssp_operator_t), intent(inout) :: op
        ! Component by component (no derived-type assignment: plain for every compiler)
        if (allocated(op%pt_ptr)) deallocate(op%pt_ptr)
        if (allocated(op%pt_mini)) deallocate(op%pt_mini)
        if (allocated(op%pt_m1)) deallocate(op%pt_m1)
        if (allocated(op%pt_m2)) deallocate(op%pt_m2)
        if (allocated(op%pt_first)) deallocate(op%pt_first)
        if (allocated(op%age_on)) deallocate(op%age_on)
        if (allocated(op%age_nstars)) deallocate(op%age_nstars)
        if (allocated(op%age_mmax)) deallocate(op%age_mmax)
        if (allocated(op%age_row_ptr)) deallocate(op%age_row_ptr)
        if (allocated(op%age_slot_ptr)) deallocate(op%age_slot_ptr)
        if (allocated(op%row_ptr)) deallocate(op%row_ptr)
        if (allocated(op%row_kind)) deallocate(op%row_kind)
        if (allocated(op%row_idx)) deallocate(op%row_idx)
        if (allocated(op%col)) deallocate(op%col)
        if (allocated(op%val)) deallocate(op%val)
        if (allocated(op%slot_key)) deallocate(op%slot_key)
        if (allocated(op%direct_spec)) deallocate(op%direct_spec)
        op%n_pts = 0; op%n_rows = 0; op%nnz = 0; op%n_slots = 0; op%n_direct = 0
    end subroutine ssp_operator_free

    ! ------------------------------------------------------------------------
    ! PER MODEL (plain arrays: the part that becomes a batched device kernel)
    ! ------------------------------------------------------------------------

    !> IMF weights of all native isochrone points (compute_imf_weights for every age): number of
    !> stars per bin per unit initial stellar mass. imf_type and p(IMF_NPAR) as in fsps_imf.
    subroutine ssp_op_imf_weights(imf_type, p, n_pts, pt_mini, pt_m1, pt_m2, pt_first, w)
        integer, intent(in)  :: imf_type, n_pts
        real(WP), intent(in) :: p(IMF_NPAR)
        real(WP), intent(in) :: pt_mini(n_pts), pt_m1(n_pts), pt_m2(n_pts)
        integer, intent(in)  :: pt_first(n_pts)
        real(WP), intent(out) :: w(n_pts)
        real(WP) :: m1, m2, total_mass, lo, hi
        integer :: k

        lo = p(IMF_P_LOWER_LIMIT)
        hi = p(IMF_P_UPPER_LIMIT)
        total_mass = imf_integral_p(imf_type, p, lo, hi, .true.)
        do k = 1, n_pts
            w(k) = 0.0_wp
            if (pt_mini(k) < lo .or. pt_mini(k) > hi) cycle
            if (pt_first(k) == 1) then
                m1 = p(IMF_P_LOWER_BOUND)
            else
                m1 = pt_m1(k)
            end if
            m2 = pt_m2(k)
            if (m2 <= m1) cycle      ! empty or inverted bin: no stars (compute_imf_weights)
            w(k) = imf_integral_p(imf_type, p, m1, m2, .false.)
            if (total_mass > 0.0_wp) w(k) = w(k) / total_mass
        end do
    end subroutine ssp_op_imf_weights

    !> wrow = S . w (one row per CSR row).
    subroutine ssp_op_spmv(n_rows, nnz, n_pts, row_ptr, col, val, w, wrow)
        integer, intent(in)  :: n_rows, nnz, n_pts
        integer, intent(in)  :: row_ptr(n_rows + 1), col(nnz)
        real(WP), intent(in) :: val(nnz), w(n_pts)
        real(WP), intent(out) :: wrow(n_rows)
        integer :: r, e
        real(WP) :: s
        do r = 1, n_rows
            s = 0.0_wp
            do e = row_ptr(r), row_ptr(r + 1) - 1
                s = s + val(e) * w(col(e))
            end do
            wrow(r) = s
        end do
    end subroutine ssp_op_spmv

    !> The per-model part: IMF weights w (work array, n_pts) and the row weights wrow = S . w
    !> (template weights, dust-corner weights, direct-star weights, mass and L_bol sums).
    subroutine ssp_op_model_weights(imf_type, p, n_pts, pt_mini, pt_m1, pt_m2, pt_first, &
                                    n_rows, nnz, row_ptr, col, val, w, wrow)
        integer, intent(in)  :: imf_type, n_pts, n_rows, nnz
        real(WP), intent(in) :: p(IMF_NPAR)
        real(WP), intent(in) :: pt_mini(n_pts), pt_m1(n_pts), pt_m2(n_pts)
        integer, intent(in)  :: pt_first(n_pts), row_ptr(n_rows + 1), col(nnz)
        real(WP), intent(in) :: val(nnz)
        real(WP), intent(out) :: w(n_pts), wrow(n_rows)
        call ssp_op_imf_weights(imf_type, p, n_pts, pt_mini, pt_m1, pt_m2, pt_first, w)
        call ssp_op_spmv(n_rows, nnz, n_pts, row_ptr, col, val, w, wrow)
    end subroutine ssp_op_model_weights

    ! ------------------------------------------------------------------------
    ! EVALUATE: generate_ssp_grid from the operator
    ! ------------------------------------------------------------------------

    !> @brief The SSP grid (as generate_ssp_grid) of the IMF (imf_type, p) from the operator.
    !> @details wrow_out, if present, returns the row weights S . w_IMF (size op%n_rows).
    subroutine ssp_operator_evaluate(ctx, pset, op, imf_type, p, mass_grid, lbol_grid, spec_grid, wrow_out)
        type(fsps_context_t), target, intent(inout) :: ctx
        type(params), intent(in)          :: pset
        type(ssp_operator_t), intent(in)  :: op
        integer, intent(in)               :: imf_type
        real(WP), intent(in)              :: p(IMF_NPAR)
        real(WP), intent(out), contiguous :: mass_grid(:), lbol_grid(:), spec_grid(:,:)
        real(WP), intent(out), optional   :: wrow_out(:)

        real(WP), allocatable :: w(:), wrow(:), w_main_all(:,:), w_small(:), dw_main(:,:,:), dw_o(:,:), &
                                 dw_c(:,:), dw_car(:,:)
        real(KP), allocatable :: acc(:), comp(:)
        integer, allocatable :: dkey(:,:)
        integer :: t, r, out_idx, nspec, nslot, s0, idx, sl, q, o1, o2, o3, o4, o5, o6
        real(WP) :: tot_mass, lin_l

        nspec = op%nspec
        mass_grid = 0.0_wp; lbol_grid = 0.0_wp; spec_grid = 0.0_wp
        allocate(w(op%n_pts), wrow(op%n_rows))
        call ssp_op_model_weights(imf_type, p, op%n_pts, op%pt_mini, op%pt_m1, op%pt_m2, op%pt_first, &
                                  op%n_rows, op%nnz, op%row_ptr, op%col, op%val, w, wrow)
        if (present(wrow_out)) wrow_out(1:op%n_rows) = wrow

        allocate(w_main_all(op%n_main, op%n_ages), w_small(op%n_small), acc(nspec), comp(nspec))
        w_main_all = 0.0_wp
        ! Offsets of the small libraries in w_small
        o1 = product(op%n_wmb); o2 = o1 + product(op%n_pagb); o3 = o2 + op%n_wrn; o4 = o3 + op%n_wrc
        o5 = o4 + op%n_car; o6 = o5 + op%n_agbo
        do t = 1, op%n_ages
            if (.not. op%age_on(t)) cycle
            out_idx = 1 + (t - 1) * TIME_RES_INCR

            ! Mass and L_bol (compute_integrated_properties)
            if (op%age_nstars(t) <= 0) then
                mass_grid(out_idx) = 0.0_wp
                lbol_grid(out_idx) = -99.0_wp
            else
                tot_mass = 0.0_wp; lin_l = 0.0_wp
                do r = op%age_row_ptr(t), op%age_row_ptr(t + 1) - 1
                    if (op%row_kind(r) == ROW_MASS) tot_mass = wrow(r)
                    if (op%row_kind(r) == ROW_LBOL) lin_l = wrow(r)
                end do
                if (ctx%add_stellar_remnants_val == 1) call add_remnant_mass_p(imf_type, p, tot_mass, op%age_mmax(t))
                mass_grid(out_idx) = tot_mass
                lbol_grid(out_idx) = log10(max(lin_l, SAFE_FLOOR))
            end if

            ! Spectrum of everything but the main library (accumulate_by_templates order:
            ! direct stars, small libraries, dust-shell corners)
            acc = 0.0_KP; comp = 0.0_KP
            w_small = 0.0_wp
            nslot = op%age_slot_ptr(t + 1) - op%age_slot_ptr(t)
            s0 = op%age_slot_ptr(t) - 1
            if (nslot > 0) then
                allocate(dkey(3, nslot), dw_main(op%nt_lib, op%ng_lib, nslot), dw_o(op%n_agbo, nslot), &
                         dw_c(op%n_agbc, nslot), dw_car(op%n_car, nslot))
                dw_main = 0.0_wp; dw_o = 0.0_wp; dw_c = 0.0_wp; dw_car = 0.0_wp
                do sl = 1, nslot
                    dkey(:, sl) = op%slot_key(:, s0 + sl)
                    dkey(1, sl) = dkey(1, sl) - 1    ! C-rich flag (sum_dusty_slots adds 1)
                end do
            end if
            do r = op%age_row_ptr(t), op%age_row_ptr(t + 1) - 1
                idx = op%row_idx(r)
                select case (op%row_kind(r))
                case (ROW_MAIN)
                    w_main_all(idx, t) = wrow(r)
                case (ROW_SMALL)
                    w_small(idx) = wrow(r)
                case (ROW_DIRECT)
                    call add_column(acc, comp, real(wrow(r), KP), op%direct_spec(:, idx))
                case (ROW_DUSTY)
                    sl = (idx - 1) / op%n_dtpl + 1 - s0
                    q = mod(idx - 1, op%n_dtpl) + 1
                    if (q <= op%n_main) then
                        dw_main(mod(q - 1, op%nt_lib) + 1, (q - 1) / op%nt_lib + 1, sl) = wrow(r)
                    else if (q <= op%n_main + op%n_car) then
                        dw_car(q - op%n_main, sl) = wrow(r)
                    else if (q <= op%n_main + op%n_car + op%n_agbo) then
                        dw_o(q - op%n_main - op%n_car, sl) = wrow(r)
                    else
                        dw_c(q - op%n_main - op%n_car - op%n_agbo, sl) = wrow(r)
                    end if
                end select
            end do
            call sum_small_templates(ctx, op%zmet, reshape(w_small(1:o1), op%n_wmb), &
                                     reshape(w_small(o1+1:o2), op%n_pagb), w_small(o2+1:o3), w_small(o3+1:o4), &
                                     w_small(o4+1:o5), w_small(o5+1:o6), w_small(o6+1:op%n_small), acc, comp)
            if (nslot > 0) then
                call sum_dusty_slots(ctx, op%zmet, nslot, dkey, dw_main, dw_o, dw_c, dw_car, acc, comp)
                deallocate(dkey, dw_main, dw_o, dw_c, dw_car)
            end if
            spec_grid(:, out_idx) = real(acc, WP)
        end do

        ! Main library of all ages (floors computed ages), then the post-processing
        call add_main_library_templates(ctx, pset, w_main_all, spec_grid)
        call finish_ssp_grid(ctx, pset, mass_grid, lbol_grid, spec_grid)
    end subroutine ssp_operator_evaluate

end module fsps_ssp_operator
