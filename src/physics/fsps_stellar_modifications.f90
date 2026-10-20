module fsps_stellar_modifications
    !> @brief Routines for modifying stellar populations and adding exotic components.
    !>
    !> @details
    !> This module consolidates routines that modify the standard isochrones or 
    !> spectra to account for specific stellar evolutionary phases and components 
    !> that require special handling, including:
    !> - Blue Straggler (BS) stars (formerly blue_stragglers.f90)
    !> - Giant Branch (GB) modifications (formerly gb_mod.f90)
    !> - Horizontal Branch (HB) morphology changes (formerly hb_mod.f90)
    !> - Stellar Remnants (WD, NS, BH) mass addition (formerly remnants_add.f90)
    !> - X-ray Binary (XRB) spectral contribution (formerly xrb_add.f90)

    use fsps_precision, only: WP
    use fsps_constants, only: NM, GRAVITY_L_M_T_COEFF, BHB_SBS_TIME
    use fsps_types, only: params
    use fsps_context_types, only: fsps_context_t
    use fsps_imf, only: get_imf_value, integrate_imf_interval_analytic, imf_integral_p, imf_pack_params, &
                        IMF_NPAR, IMF_P_LOWER_LIMIT, IMF_P_UPPER_LIMIT, IMF_P_MLIM_BH, IMF_P_MLIM_NS
    use fsps_interpolation, only: interpolate_linear, find_interval
    
    
    implicit none
    private

    ! ------------------------------------------------------------------------
    ! PUBLIC API
    ! ------------------------------------------------------------------------
    public :: apply_blue_stragglers
    public :: modify_giant_branch
    public :: modify_horizontal_branch
    public :: add_remnant_mass, add_remnant_mass_p
    public :: add_xray_binaries

    ! ------------------------------------------------------------------------
    ! CONSTANTS & PARAMETERS
    ! ------------------------------------------------------------------------
    
    ! Blue Straggler Constants
    integer, parameter :: N_BS_STARS = 20          !> Number of BS stars to add per isochrone
    real(WP), parameter :: BS_LUM_OFFSET = 0.2_wp  !> Luminosity offset (dex) for BS
    real(WP), parameter :: BS_LUM_EXTENT = 0.75_wp !> Extent (dex) of BS sequence
    real(WP), parameter :: ZAMS_LUM_LIMIT = 3.5_wp
    real(WP), parameter :: MSTO_TOLERANCE = 0.2_wp
    real(WP), parameter :: BS_PHASE_ID    = 7.0_wp

    ! Giant Branch Constants
    real(WP), parameter :: AGE_PADOVA_LOW = 8.0_wp
    real(WP), parameter :: AGE_PADOVA_HIGH = 9.1_wp
    
    ! Horizontal Branch Constants
    integer, parameter :: N_HB_SUBSTEPS = 10       !> Number of blue HB stars to add per HB star
    real(WP), parameter :: HB_BLUE_TEMP_MAX = 4.2_wp !> Max logT for distributed HB
    real(WP), parameter :: GRAD_THRESH_HB = -5.0e2_wp
    real(WP), parameter :: LUM_THRESH_MIN = 2.5_wp
    real(WP), parameter :: LUM_WIDTH_TOL = 0.1_wp

    ! Remnant Constants (Renzini & Ciotti 1993)
    real(WP), parameter :: MASS_NS_REMNANT = 1.4_wp
    real(WP), parameter :: WD_SLOPE = 0.077_wp
    real(WP), parameter :: WD_INTERCEPT = 0.48_wp

contains

    !> @brief
    !> Adds Blue Straggler (BS) stars to the isochrone.
    !>
    !> @details
    !> Extends the main sequence beyond the turn-off point to simulate Blue Stragglers.
    !> BS stars are just an extension of the main sequence, weighted relative to the 
    !> Horizontal Branch (HB).
    !>
    !> **Algorithm:**
    !> 1. Builds the ZAMS (Zero Age Main Sequence) reference with `build_zams_reference`.
    !> 2. Scans the current isochrone (`time_idx`) to find the Main Sequence Turn-Off (MSTO).
    !>    This is done by comparing the current star's Luminosity vs Temperature against 
    !>    the ZAMS relation. When they diverge (> 0.2 dex), we define that as the MSTO.
    !> 3. Adds `N_BS_STARS` (20) extending from the MSTO to higher luminosities.
    !>    - New properties are interpolated from the ZAMS relation.
    !>
    !> @param[inout] ctx          Simulation context.
    !> @param[in]    time_idx     Index of the current time step.
    !> @param[in]    metal_idx    Metallicity index
    !> @param[in]    s_bs         Specific frequency of Blue Stragglers relative to HB.
    !> @param[in]    hb_weight    Total weight of the Horizontal Branch population.
    !> @param[inout] n_mass       Number of mass points in the isochrone (updated).
    !> @param[inout] mass_ini     Array of initial masses (modified).
    !> @param[inout] mass_act     Array of actual masses (modified).
    !> @param[inout] log_l        Array of Log Luminosity (modified).
    !> @param[inout] log_t        Array of Log Temperature (modified).
    !> @param[inout] log_g        Array of Log Gravity (modified).
    !> @param[inout] phase        Array of evolutionary phases (modified).
    !> @param[inout] weights      Array of weights (modified).
    subroutine apply_blue_stragglers(ctx, time_idx, metal_idx, s_bs, hb_weight, n_mass, &
                                     mass_ini, mass_act, log_l, log_t, log_g, phase, &
                                     weights)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: time_idx, metal_idx
        real(WP), intent(in) :: s_bs
        real(WP), intent(in) :: hb_weight
        integer, dimension(:), intent(inout) :: n_mass
        
        real(WP), dimension(:,:), intent(inout), contiguous :: mass_ini, mass_act
        real(WP), dimension(:,:), intent(inout), contiguous :: log_l, log_t, log_g, phase
        real(WP), dimension(:), intent(inout), contiguous :: weights

        ! Local variables
        real(WP), dimension(:), allocatable :: zams_t_cache, zams_l_cache, zams_m_cache
        integer :: n_zams, idx_msto
        integer :: i, k, n_curr
        real(WP) :: bs_total_weight
        real(WP) :: lum_expected_on_zams, diff_from_zams
        real(WP) :: new_logl, new_logt, new_mass
        real(WP) :: inv_nbs

        ! 1. Safety Check (Legacy: IF (ctx%zin < -999999))
        if (ctx%zin < -9.0e5_wp) return

        bs_total_weight = s_bs * hb_weight
        n_curr = n_mass(time_idx)

        ! 2. Build the ZAMS reference from the isochrone set (not from the current age)
        call build_zams_reference(ctx, metal_idx, zams_t_cache, zams_l_cache, zams_m_cache, n_zams)
        if (n_zams < 2) return

        ! 3. Find MSTO
        idx_msto = 0
        diff_from_zams = 0.0_wp
        
        do while (diff_from_zams < MSTO_TOLERANCE .and. idx_msto < n_curr)
            idx_msto = idx_msto + 1
            
            lum_expected_on_zams = interpolate_linear(zams_t_cache, &
                                                      zams_l_cache, &
                                                      log_t(time_idx, idx_msto))
            
            if (lum_expected_on_zams /= lum_expected_on_zams) return
            
            diff_from_zams = abs(lum_expected_on_zams - log_l(time_idx, idx_msto))
        end do

        if (idx_msto < 2) return

        ! 4. Add Blue Straggler Stars
        if (n_curr + N_BS_STARS > NM) then
            write(*,*) '[FSPS-STELLAR] Error: Arrays full in apply_blue_stragglers.'
            stop
        end if

        idx_msto = idx_msto - 1
        inv_nbs = 1.0_wp / real(N_BS_STARS, WP)

        do k = 1, N_BS_STARS
            
            i = n_curr + k

            new_logl = log_l(time_idx, idx_msto) + BS_LUM_OFFSET + &
                       (BS_LUM_EXTENT * real(k,WP) * inv_nbs)

            log_l(time_idx, i) = new_logl

            ! FIX 3: Interpolate using the CACHED ZAMS arrays
            ! Previously you used log_l(1, ...), which was incorrect.
            new_mass = interpolate_linear(zams_l_cache, &
                                          zams_m_cache, &
                                          new_logl)
            
            if (new_mass /= new_mass) new_mass = mass_ini(time_idx, idx_msto)
            
            mass_ini(time_idx, i) = new_mass
            mass_act(time_idx, i) = new_mass 

            ! FIX 4: Interpolate Temperature from CACHED ZAMS arrays
            new_logt = interpolate_linear(zams_l_cache, &
                                          zams_t_cache, &
                                          new_logl)
            
            if (new_logt /= new_logt) new_logt = log_t(time_idx, idx_msto)
            
            log_t(time_idx, i) = new_logt

            log_g(time_idx, i) = log10(GRAVITY_L_M_T_COEFF * mass_act(time_idx, i)) - &
                                 log_l(time_idx, i) + 4.0_wp * log_t(time_idx, i)

            phase(time_idx, i) = BS_PHASE_ID
            weights(i)         = inv_nbs * bs_total_weight

        end do

        n_mass(time_idx) = n_mass(time_idx) + N_BS_STARS

        if (allocated(zams_t_cache)) deallocate(zams_t_cache)
        if (allocated(zams_l_cache)) deallocate(zams_l_cache)
        if (allocated(zams_m_cache)) deallocate(zams_m_cache)

    end subroutine apply_blue_stragglers

    !> @brief
    !> Builds the zero-age main sequence used to locate the turn-off and place blue stragglers.
    !>
    !> @details
    !> A star's ZAMS point is its position in the youngest isochrone in which it is on the
    !> main sequence (phase 0). Ages are scanned in ascending order; at each age, the MS points
    !> less massive than every MS star seen at younger ages are the stars that have just arrived
    !> on the MS. The result is sorted by initial mass.
    !>
    !> - Isochrone sets without a pre-main sequence (e.g. Padova) contribute every point at the
    !>   first age, which reproduces the legacy reference (the first isochrone).
    !> - Sets with a pre-main sequence (e.g. MIST, whose log t = 5 isochrone has only massive
    !>   stars on the MS) take their low-mass ZAMS from the ages at which those stars arrive.
    !>   Using the first isochrone there puts the turn-off among the M dwarfs (legacy FSPS) or
    !>   finds none at all.
    !>
    !> Points that break strict monotonicity in log L or log T (stars that evolved slightly
    !> within an age step) are dropped, since both serve as interpolation abscissae. As in
    !> legacy FSPS, the table ends at the first point with log L >= ZAMS_LUM_LIMIT.
    !>
    !> @param[in]  ctx        Simulation context (isochrone state).
    !> @param[in]  metal_idx  Metallicity index.
    !> @param[out] zams_t     ZAMS log Teff, ascending.
    !> @param[out] zams_l     ZAMS log L, ascending.
    !> @param[out] zams_m     ZAMS initial mass, ascending.
    !> @param[out] n_zams     Number of ZAMS points.
    subroutine build_zams_reference(ctx, metal_idx, zams_t, zams_l, zams_m, n_zams)
        type(fsps_context_t), intent(in) :: ctx
        integer, intent(in) :: metal_idx
        real(WP), dimension(:), allocatable, intent(out) :: zams_t, zams_l, zams_m
        integer, intent(out) :: n_zams

        real(WP), dimension(:), allocatable :: t_all, l_all, m_all
        real(WP) :: m_low, m_low_next, t_tmp, l_tmp, m_tmp
        integer :: i_age, j, k, n_all, n_new

        ! Pass 1: count stars arriving on the MS at each age
        n_all = 0
        m_low = huge(1.0_wp)
        do i_age = 1, size(ctx%state%nmass_isoc, 2)
            m_low_next = m_low
            do j = 1, ctx%state%nmass_isoc(metal_idx, i_age)
                if (nint(ctx%state%phase_isoc(metal_idx, i_age, j)) /= 0) cycle
                if (ctx%state%mini_isoc(metal_idx, i_age, j) < m_low) n_all = n_all + 1
                m_low_next = min(m_low_next, ctx%state%mini_isoc(metal_idx, i_age, j))
            end do
            m_low = m_low_next
        end do

        allocate(t_all(n_all), l_all(n_all), m_all(n_all))

        ! Pass 2: collect them
        n_new = 0
        m_low = huge(1.0_wp)
        do i_age = 1, size(ctx%state%nmass_isoc, 2)
            m_low_next = m_low
            do j = 1, ctx%state%nmass_isoc(metal_idx, i_age)
                if (nint(ctx%state%phase_isoc(metal_idx, i_age, j)) /= 0) cycle
                if (ctx%state%mini_isoc(metal_idx, i_age, j) < m_low) then
                    n_new = n_new + 1
                    t_all(n_new) = ctx%state%logt_isoc(metal_idx, i_age, j)
                    l_all(n_new) = ctx%state%logl_isoc(metal_idx, i_age, j)
                    m_all(n_new) = ctx%state%mini_isoc(metal_idx, i_age, j)
                end if
                m_low_next = min(m_low_next, ctx%state%mini_isoc(metal_idx, i_age, j))
            end do
            m_low = m_low_next
        end do

        ! Sort by initial mass (insertion sort; each age's block is already ordered)
        do j = 2, n_all
            t_tmp = t_all(j)
            l_tmp = l_all(j)
            m_tmp = m_all(j)
            k = j - 1
            do while (k >= 1)
                if (m_all(k) <= m_tmp) exit
                t_all(k + 1) = t_all(k)
                l_all(k + 1) = l_all(k)
                m_all(k + 1) = m_all(k)
                k = k - 1
            end do
            t_all(k + 1) = t_tmp
            l_all(k + 1) = l_tmp
            m_all(k + 1) = m_tmp
        end do

        ! Keep a strictly increasing sequence in log L and log T, up to ZAMS_LUM_LIMIT
        n_zams = 0
        do j = 1, n_all
            if (n_zams > 0) then
                if (l_all(j) <= l_all(n_zams) .or. t_all(j) <= t_all(n_zams)) cycle
            end if
            n_zams = n_zams + 1
            t_all(n_zams) = t_all(j)
            l_all(n_zams) = l_all(j)
            m_all(n_zams) = m_all(j)
            if (l_all(n_zams) >= ZAMS_LUM_LIMIT) exit
        end do

        zams_t = t_all(1:n_zams)
        zams_l = l_all(1:n_zams)
        zams_m = m_all(1:n_zams)

    end subroutine build_zams_reference

    !> @brief
    !> Modifies TP-AGB, HB, RGB, and post-AGB stars in the isochrone.
    !>
    !> @details
    !> This routine applies shifts to luminosity and temperature, and re-weights 
    !> specific evolutionary phases based on user parameters.
    !>
    !> **Phases Modified:**
    !> - **TP-AGB (Phase 5):**
    !>   - Applies specialized calibration corrections (Padova isochrones only).
    !>     Supports Conroy & Gunn (2010) or Villaume et al. (2014) normalizations.
    !>   - Applies user-defined `shift_logl` and `shift_logt`.
    !>   - Applies weight scaling `scale_agb`.
    !> - **Post-AGB (Phase 6):**
    !>   - Applies weight scaling `scale_pagb`.
    !> - **RGB/Red Clump/AGB (Phases 2-5):**
    !>   - Applies general giant branch weight scaling `scale_redgb`.
    !>   - *Note:* This overlaps with Phase 5; TP-AGB stars receive both scalings.
    !>
    !> @param[inout] ctx           Simulation context.
    !> @param[in]    time_idx      Current time step index.
    !> @param[in]    metal_idx     Current metallicity index.
    !> @param[in]    age_now       Current age of the population (log years).
    !> @param[in]    n_stars       Number of stars in the current isochrone.
    !> @param[in]    shift_logt    Shift in Log T (delt) for TP-AGB.
    !> @param[in]    shift_logl    Shift in Log L (dell) for TP-AGB.
    !> @param[in]    scale_pagb    Weight scaling factor for Post-AGB.
    !> @param[in]    scale_redgb   Weight scaling factor for RGB/Red Clump.
    !> @param[in]    scale_agb     Weight scaling factor for TP-AGB.
    !> @param[inout] log_l         Array of Log Luminosity (modified).
    !> @param[inout] log_t         Array of Log Temperature (modified).
    !> @param[in]    phase         Array of evolutionary phases.
    !> @param[inout] weights       Array of weights (modified).
    subroutine modify_giant_branch(ctx, time_idx, metal_idx, age_now, n_stars, &
                                   shift_logt, shift_logl, scale_pagb, &
                                   scale_redgb, scale_agb, &
                                   log_l, log_t, phase, weights)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: time_idx, metal_idx, n_stars
        real(WP), intent(in) :: age_now
        real(WP), intent(in) :: shift_logt, shift_logl
        real(WP), intent(in) :: scale_pagb, scale_redgb, scale_agb
        
        real(WP), dimension(:,:), intent(inout), contiguous :: log_l, log_t
        real(WP), dimension(:,:), intent(in), contiguous :: phase
        real(WP), dimension(:), intent(inout), contiguous :: weights

        ! Local variables
        integer :: i
        real(WP) :: current_phase
        real(WP) :: z_ratio_log
        real(WP) :: villaume_factor
        real(WP) :: combined_agb_scale
        real(WP) :: cg_shift_l, cg_shift_t_low_z, cg_shift_t_high_age
        logical :: apply_cg_norm
        
        logical :: is_padova
        integer :: norm_type

        ! Cache state checks
        is_padova = (trim(ctx%state%isoc_type) == 'pdva')
        norm_type = ctx%tpagb_norm_type_val

        ! Pre-calculate metallicity ratio for Normalization 1 (Conroy & Gunn)
        ! Ratio = Log10(Z / Zsol)
        z_ratio_log = 0.0_wp
        if (norm_type == 1 .and. is_padova) then
            if (ctx%state%zsol > 0.0_wp) then
                z_ratio_log = log10(ctx%state%zlegend(metal_idx) / ctx%state%zsol)
            end if
        end if

        ! Calculate the Villaume et al. (2014) scaling factor once
        villaume_factor = 1.0_wp
        if (is_padova .and. norm_type == 2) then
            villaume_factor = max(0.1_wp, 10.0_wp**(-1.0_wp + (age_now - 8.0_wp)/2.5_wp))
        end if

        ! Pre-calculate combined AGB weight scaling
        ! Phase 5 stars get Villaume * scale_agb * scale_redgb
        combined_agb_scale = villaume_factor * scale_agb
        if (abs(scale_redgb - 1.0_wp) > tiny(0.0_wp)) then
            combined_agb_scale = combined_agb_scale * scale_redgb
        end if

        ! Pre-calculate Conroy & Gunn (2010) shifts
        apply_cg_norm = (is_padova .and. norm_type == 1)
        cg_shift_l = 0.0_wp
        cg_shift_t_low_z = 0.0_wp
        cg_shift_t_high_age = 0.0_wp

        if (apply_cg_norm) then
            if (age_now > AGE_PADOVA_LOW .and. age_now < AGE_PADOVA_HIGH) then
                ! Intermediate Age
                cg_shift_l = -1.0_wp + (age_now - 8.0_wp) / 1.5_wp
                cg_shift_t_low_z = 0.10_wp
            else
                ! Outside Intermediate Age
                cg_shift_l = -max(min(0.4_wp, -z_ratio_log), 0.2_wp)
                cg_shift_t_high_age = 0.1_wp - min((age_now - AGE_PADOVA_HIGH) / 1.5_wp, 0.2_wp)
            end if
        end if

        ! --------------------------------------------------------------------
        ! MAIN LOOP
        ! --------------------------------------------------------------------
        do i = 1, n_stars
            
            current_phase = phase(time_idx, i)

            ! ----------------------------------------------------------------
            ! PHASE 5: TP-AGB MODIFICATIONS
            ! ----------------------------------------------------------------
            if (current_phase == 5.0_wp) then

                ! Apply pre-calculated combined weights
                weights(i) = weights(i) * combined_agb_scale

                ! Apply shifts (only if non-zero, to save adds/stores)
                if (abs(shift_logl) > 0.0_wp) log_l(time_idx, i) = log_l(time_idx, i) + shift_logl
                if (abs(shift_logt) > 0.0_wp) log_t(time_idx, i) = log_t(time_idx, i) + shift_logt

                ! Apply Padova-specific Normalizations
                if (apply_cg_norm) then
                    if (age_now > AGE_PADOVA_LOW .and. age_now < AGE_PADOVA_HIGH) then
                        log_l(time_idx, i) = log_l(time_idx, i) + cg_shift_l
                        if (z_ratio_log < -0.25_wp) then
                            log_t(time_idx, i) = log_t(time_idx, i) + cg_shift_t_low_z
                        end if
                    else
                        log_l(time_idx, i) = log_l(time_idx, i) + cg_shift_l
                        log_t(time_idx, i) = log_t(time_idx, i) + cg_shift_t_high_age
                    end if
                end if

            end if

            ! ----------------------------------------------------------------
            ! PHASE 6: POST-AGB MODIFICATIONS
            ! ----------------------------------------------------------------
            if (current_phase == 6.0_wp) then
                if (abs(scale_pagb - 1.0_wp) > tiny(0.0_wp)) then
                    weights(i) = weights(i) * scale_pagb
                end if
            end if

            ! ----------------------------------------------------------------
            ! GENERAL GIANT BRANCH SCALING (Phases 2, 3, 4, 5)
            ! ----------------------------------------------------------------
            if (current_phase >= 2.0_wp .and. current_phase <= 4.0_wp) then
                if (abs(scale_redgb - 1.0_wp) > tiny(0.0_wp)) then
                    weights(i) = weights(i) * scale_redgb
                end if
            end if

        end do

    end subroutine modify_giant_branch

    !> @brief
    !> Modifies the Horizontal Branch (HB) to include bluer stars.
    !>
    !> @details
    !> Redistributes a fraction `f_bhb` of Red Clump stars uniformly to higher 
    !> temperatures to simulate Blue HB stars (BHB).
    !>
    !> **Logic Breakdown:**
    !> - **Padova Isochrones:** The HB is not explicitly tagged. It is detected 
    !>   dynamically by looking for large negative luminosity gradients (`flip` logic).
    !>   For every HB point found, `N_HB_SUBSTEPS` (10) new blue stars are added.
    !> - **MIST/BaSTI Isochrones:** The HB is explicitly tagged as Phase 3.
    !>   For every HB point, 1 new blue star is added, and the original star is 
    !>   forced to the Red Clump temperature (`min_teff`).
    !>
    !> @param[inout] ctx          Simulation context.
    !> @param[in]    time_idx     Current time step index.
    !> @param[in]    f_bhb        Fraction of HB stars to redistribute to Blue.
    !> @param[in]    hb_time      Age of the population (used for turn-on check).
    !> @param[out]   hb_total_weight Total weight of HB stars found.
    !> @param[inout] n_mass       Number of mass points (updated).
    !> @param[inout] mass_ini     Initial mass array (modified).
    !> @param[inout] mass_act     Actual mass array (modified).
    !> @param[inout] log_l        Log Luminosity array (modified).
    !> @param[inout] log_t        Log Temperature array (modified).
    !> @param[inout] log_g        Log Gravity array (modified).
    !> @param[inout] phase        Phase array (modified).
    !> @param[inout] weights      Weights array (modified).
    subroutine modify_horizontal_branch(ctx, time_idx, f_bhb, hb_time, hb_total_weight, &
                                        n_mass, mass_ini, mass_act, log_l, log_t, &
                                        log_g, phase, weights, hb_flag, parent)
        type(fsps_context_t), intent(inout) :: ctx
        integer, intent(in) :: time_idx
        real(WP), intent(in) :: f_bhb, hb_time
        real(WP), intent(out) :: hb_total_weight
        integer, dimension(:), intent(inout) :: n_mass
        
        real(WP), dimension(:,:), intent(inout), contiguous :: mass_ini, mass_act
        real(WP), dimension(:,:), intent(inout), contiguous :: log_l, log_t, log_g, phase
        real(WP), dimension(:), intent(inout), contiguous :: weights
        !> Bookkeeping for the batched SSP operator (fsps_ssp_operator), no effect on the result:
        !> hb_flag(j) is set for the stars summed into hb_total_weight, parent(i) = j for every
        !> blue HB star i added from star j (entries of other stars are left unchanged)
        logical, dimension(:), intent(inout), optional :: hb_flag
        integer, dimension(:), intent(inout), optional :: parent

        ! Local variables
        integer :: i, j, k, n_curr
        integer :: n_blue_added
        real(WP) :: dl, dm, hb_lum_marker
        logical :: is_in_hb_region
        real(WP) :: inv_nbs
        
        ! MIST specific vars
        real(WP) :: min_teff_hb
        integer :: total_hb_stars
        integer :: mist_counter

        hb_total_weight = 0.0_wp
        n_curr = n_mass(time_idx)

        ! --------------------------------------------------------------------
        ! BRANCH 1: PADOVA ISOCHRONES
        ! --------------------------------------------------------------------
        ! Legacy method: Detect HB via large negative Luminosity/Mass gradient.
        if (trim(ctx%state%isoc_type) == 'pdva') then
            
            is_in_hb_region = .false.
            hb_lum_marker = -999.0_wp

            ! Iterate through the isochrone points
            do j = 2, n_curr
                
                ! Start of HB: d(LogL)/d(Mass) <= GRAD_THRESH_HB below a bright point. Tested
                ! without dividing, because the tables repeat the tip-of-RGB mass at the drop
                ! (dm = 0), which legacy MOD_HB evaluates as a -Inf gradient. As in legacy, the
                ! test is repeated on the HB (resetting the marker).
                dl = log_l(time_idx, j) - log_l(time_idx, j-1)
                dm = mass_ini(time_idx, j) - mass_ini(time_idx, j-1)
                if (dl < 0.0_wp .and. dm >= 0.0_wp .and. dl <= GRAD_THRESH_HB * dm .and. &
                    log_l(time_idx, j-1) > LUM_THRESH_MIN) then
                    
                    is_in_hb_region = .true.
                    hb_lum_marker = log_l(time_idx, j)
                end if

                ! Process HB stars
                if (is_in_hb_region) then
                    
                    ! Once the luminosity leaves the HB we are past it for good (legacy flip = -1);
                    ! later dips on the AGB are not a second HB.
                    if (abs(log_l(time_idx, j) - hb_lum_marker) >= LUM_WIDTH_TOL) exit

                    ! We are on the HB: Accumulate weight
                    hb_total_weight = hb_total_weight + weights(j)
                    if (present(hb_flag)) hb_flag(j) = .true.

                    ! Apply modification if requested and age is appropriate
                    if (f_bhb > 1.0e-3_wp .and. hb_time >= BHB_SBS_TIME) then
                        
                        ! Ensure we have space in arrays
                        if (n_mass(time_idx) + N_HB_SUBSTEPS > NM) then
                            write(*,*) '[FSPS-STELLAR] Error: Arrays full in modify_horizontal_branch (Padova).'
                            stop
                        end if

                        ! Add N_HB_SUBSTEPS (10) new stars
                        ! They inherit properties from the current HB star j
                        n_blue_added = N_HB_SUBSTEPS
                        inv_nbs = 1.0_wp / real(n_blue_added, WP)
                        
                        do k = 1, n_blue_added
                            i = n_mass(time_idx) + k
                            
                            mass_ini(time_idx, i) = mass_ini(time_idx, j)
                            mass_act(time_idx, i) = mass_act(time_idx, j)
                            log_l(time_idx, i)    = log_l(time_idx, j)
                            phase(time_idx, i)    = 8.0 ! Mark as modified
                            
                            ! Distribute Temperature uniformly to high T
                            ! Legacy: logt(t,j)+(4.2-logt(t,j))*i/REAL(nhb)
                            log_t(time_idx, i) = log_t(time_idx, j) + &
                                (HB_BLUE_TEMP_MAX - log_t(time_idx, j)) * real(k, WP)*inv_nbs

                            ! Recompute Gravity
                            log_g(time_idx, i) = log10(GRAVITY_L_M_T_COEFF * mass_act(time_idx, i)) - &
                                                 log_l(time_idx, i) + 4.0_wp * log_t(time_idx, i)

                            ! Distribute Weight
                            weights(i) = (f_bhb * weights(j)) * inv_nbs
                            if (present(parent)) parent(i) = j
                        end do

                        ! Reduce weight of the original red clump star
                        weights(j) = weights(j) * (1.0_wp - f_bhb)
                        
                        ! Update total count
                        n_mass(time_idx) = n_mass(time_idx) + n_blue_added
                    end if
                end if
            end do

        ! --------------------------------------------------------------------
        ! BRANCH 2: MIST / BaSTI ISOCHRONES
        ! --------------------------------------------------------------------
        ! Modern method: Explicit Phase 3 tags.
        else if (trim(ctx%state%isoc_type) == 'bsti' .or. &
                 trim(ctx%state%isoc_type) == 'mist') then

            ! 1. Pre-scan: Count HB stars and find the Red Clump (minimum Teff on HB)
            total_hb_stars = 0
            min_teff_hb = 1.0e6_wp ! Arbitrary high start

            do j = 1, n_curr
                if (phase(time_idx, j) == 3.0_wp) then
                    total_hb_stars = total_hb_stars + 1
                    
                    ! MIST specific filter: skip TRGB descenders using mass continuity check
                    ! Legacy: (mini(t,i+1)-mini(t,i)).GT.1E-6
                    if (j < n_curr) then
                         if (log_t(time_idx, j) < min_teff_hb .and. &
                            (mass_ini(time_idx, j+1) - mass_ini(time_idx, j) > 1.0e-6_wp)) then
                            min_teff_hb = log_t(time_idx, j)
                         end if
                    end if
                end if
            end do

            ! 2. Modification Loop
            mist_counter = 1
            inv_nbs = 1.0_wp / real(total_hb_stars, WP)
            
            do j = 1, n_curr
                
                ! Only modify Core Helium Burning stars (Phase 3)
                if (phase(time_idx, j) == 3.0_wp) then
                    
                    hb_total_weight = hb_total_weight + weights(j)
                    if (present(hb_flag)) hb_flag(j) = .true.

                    if (f_bhb > 1.0e-4_wp .and. hb_time >= BHB_SBS_TIME) then
                        
                         if (n_mass(time_idx) + 1 > NM) then
                            write(*,*) '[FSPS-STELLAR] Error: Arrays full in modify_horizontal_branch (MIST).'
                            stop
                        end if

                        ! Add ONE new blue star per HB star
                        n_mass(time_idx) = n_mass(time_idx) + 1
                        i = n_mass(time_idx) ! Index of new star

                        ! Copy base properties
                        mass_ini(time_idx, i) = mass_ini(time_idx, j)
                        mass_act(time_idx, i) = mass_act(time_idx, j)
                        log_l(time_idx, i)    = log_l(time_idx, j)
                        phase(time_idx, i)    = 8.0_wp ! Mark as modified
                        
                        ! Force original star to be strictly Red Clump (min_teff)
                        log_t(time_idx, j) = min_teff_hb

                        ! Set new star to Distributed Temperature
                        ! Legacy: logt + (4.5 - logt) * counter / total_hb
                        log_t(time_idx, i) = log_t(time_idx, j) + &
                            (4.5_wp - log_t(time_idx, j)) * real(mist_counter,WP) * inv_nbs
                        
                        mist_counter = mist_counter + 1

                        ! Recompute Gravity for new star
                        log_g(time_idx, i) = log10(GRAVITY_L_M_T_COEFF * mass_act(time_idx, i)) - &
                                             log_l(time_idx, i) + 4.0_wp * log_t(time_idx, i)

                        ! Swap Weights
                        weights(i) = f_bhb * weights(j)
                        weights(j) = weights(j) * (1.0_wp - f_bhb)
                        if (present(parent)) parent(i) = j
                        
                    end if
                end if
            end do

        end if

    end subroutine modify_horizontal_branch

    !> @brief
    !> Adds the mass of stellar remnants (WD, NS, BH) to the SSP mass budget.
    !>
    !> @details
    !> Integrates the IMF over the mass ranges that produce specific remnants,
    !> applying the initial-mass-to-remnant-mass relations from Renzini & Ciotti (1993).
    !>
    !> The mass ranges are handled dynamically:
    !> - **Black Holes:** Formed from stars with M > mlim_bh (typically 40 Msol). 
    !>   Remnant mass = 0.5 * Initial Mass.
    !> - **Neutron Stars:** Formed from stars with mlim_ns < M < mlim_bh. 
    !>   Remnant mass = 1.4 Msol.
    !> - **White Dwarfs:** Formed from stars with M < mlim_ns (typically 8.5 Msol). 
    !>   Remnant mass = 0.077 * M + 0.48.
    !>
    !> @param[inout] ctx             Simulation context.
    !> @param[inout] current_mass    Total mass of the SSP (updated).
    !> @param[in]    max_living_mass Maximum mass of stars still alive (M_max).
    subroutine add_remnant_mass(ctx, current_mass, max_living_mass)
        type(fsps_context_t), intent(inout) :: ctx
        real(WP), intent(inout) :: current_mass
        real(WP), intent(in) :: max_living_mass
        real(WP) :: p(IMF_NPAR)

        call imf_pack_params(ctx, p)
        call add_remnant_mass_p(ctx%imf_type_val, p, current_mass, max_living_mass)
    end subroutine add_remnant_mass

    !> add_remnant_mass with the IMF parameters packed in p(IMF_NPAR) (fsps_imf, imf_pack_params):
    !> no context, so usable for a batch of IMF parameter sets (fsps_ssp_operator).
    pure subroutine add_remnant_mass_p(imf_type, p, current_mass, max_living_mass)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in) :: imf_type
        real(WP), intent(in) :: p(IMF_NPAR)
        real(WP), intent(inout) :: current_mass
        real(WP), intent(in) :: max_living_mass

        real(WP) :: imf_norm
        real(WP) :: integration_min, integration_max
        real(WP) :: term_bh, term_ns, term_wd_const, term_wd_linear
        
        real(WP) :: limit_low, limit_high, limit_bh, limit_ns

        limit_low  = p(IMF_P_LOWER_LIMIT)
        limit_high = p(IMF_P_UPPER_LIMIT)
        limit_bh   = p(IMF_P_MLIM_BH)
        limit_ns   = p(IMF_P_MLIM_NS)
        ! 1. Normalize the weights (Mass-weighted IMF integral over the full range)
        !    This ensures we are working with the correct mass fractions.
        imf_norm = imf_integral_p(imf_type, p, limit_low, limit_high, mass_weighted=.true.)

        if (imf_norm <= 0.0_wp) then
            ! Guard against division by zero if IMF is invalid
            return 
        end if

        ! --------------------------------------------------------------------
        ! 2. Black Hole Remnants
        ! --------------------------------------------------------------------
        ! Range: [max(mlim_bh, max_living_mass), imf_upper_limit]
        ! Logic: BHs only form from stars that have died (Mass > max_living_mass).
        !        They must also be massive enough to be BH progenitors (Mass > mlim_bh).
        ! Formula: M_rem = 0.5 * M_initial
        
        integration_min = min(max(limit_bh, max_living_mass), limit_high)
        
        ! Only integrate if the range is valid (min < max)
        if (integration_min < limit_high) then
            term_bh = imf_integral_p(imf_type, p, integration_min, limit_high, mass_weighted=.true.)
            current_mass = current_mass + (0.5_wp * term_bh / imf_norm)
        end if

        ! --------------------------------------------------------------------
        ! 3. Neutron Star Remnants
        ! --------------------------------------------------------------------
        ! Range: [max(mlim_ns, max_living_mass), min(mlim_bh, imf_upper_limit)]
        ! Logic: NS form from dead stars (M > max_living) that are below the BH limit.
        ! Formula: M_rem = 1.4 (constant)
        
        ! We only check for NS if the current max living mass has dropped below the BH limit.
        if (max_living_mass <= limit_bh) then
            
            integration_min = min(max(limit_ns, max_living_mass), limit_high)
            integration_max = min(limit_bh, limit_high)

            if (integration_min < integration_max) then
                ! Integrate Number Density (wrapper_imf_number) because mass is constant (1.4)
                term_ns = imf_integral_p(imf_type, p, integration_min, integration_max, mass_weighted=.false.)
                current_mass = current_mass + (MASS_NS_REMNANT * term_ns / imf_norm)
            end if
        end if

        ! --------------------------------------------------------------------
        ! 4. White Dwarf Remnants
        ! --------------------------------------------------------------------
        ! Logic: WDs form from dead stars below the NS limit.
        
        if (max_living_mass <= limit_ns) then
            
            integration_min = min(max_living_mass, limit_high)
            integration_max = min(limit_ns, limit_high)

            if (integration_min < integration_max) then
                ! Term 1: Constant part (0.48 * Number of stars)
                term_wd_const = imf_integral_p(imf_type, p, integration_min, integration_max, mass_weighted=.false.)
                
                ! Term 2: Linear part (0.077 * Mass of stars)
                term_wd_linear = imf_integral_p(imf_type, p, integration_min, integration_max, mass_weighted=.true.)

                current_mass = current_mass + &
                               ((WD_INTERCEPT * term_wd_const) / imf_norm) + &
                               ((WD_SLOPE * term_wd_linear) / imf_norm)
            end if
        end if

    end subroutine add_remnant_mass_p

    !> @brief
    !> Adds X-ray Binary (XRB) emission to the output spectrum.
    !>
    !> @details
    !> Interpolates a pre-computed grid of XRB spectral templates (`spec_xrb`) 
    !> based on the SSP's metallicity and age, scales it by the user's fraction 
    !> `frac_xrb`, and adds it to the total spectrum.
    !>
    !> **Interpolation:**
    !> - **Metallicity:** Determined once based on the SSP's Z (`pset%zmet`).
    !> - **Age:** Determined at every time step based on `time_full(t)`.
    !> - Uses standard bilinear interpolation (Age, Z).
    !>
    !> @param[inout] ctx      Simulation context.
    !> @param[in]    pset     Parameter set containing XRB fraction.
    !> @param[in]    spec_in  Input spectrum (SSP only).
    !> @param[inout] spec_out Output spectrum (SSP + XRB).
    subroutine add_xray_binaries(ctx, pset, spec_in, spec_out)
        type(fsps_context_t), intent(inout) :: ctx
        type(params), intent(in) :: pset
        real(WP), dimension(:,:), intent(in), contiguous :: spec_in
        real(WP), dimension(:,:), intent(inout), contiguous :: spec_out

        ! Local variables
        integer :: t_idx
        integer :: idx_z, idx_age
        real(WP) :: val_z_log
        real(WP) :: weight_z, weight_age
        real(WP) :: w00, w10, w01, w11
        
        ! Pointers/Aliases for readability
        real(WP), dimension(:), pointer :: grid_z, grid_age, grid_time
        real(WP), dimension(:,:,:), pointer :: grid_spec
        integer :: n_z_grid, n_age_grid, n_time_steps

        ! Setup pointers to context data
        grid_z      => ctx%state%zmet_xrb
        grid_age    => ctx%state%ages_xrb
        grid_spec   => ctx%state%spec_xrb
        grid_time   => ctx%state%time_full
        
        n_z_grid    = ctx%state%nz_xrb
        n_age_grid  = ctx%state%nt_xrb
        n_time_steps = ctx%state%nt

        ! Initialize output with input (additive)
        spec_out = spec_in

        ! Quick exit if XRB fraction is effectively zero
        if (pset%frac_xrb <= tiny(0.0_wp)) return

        ! --------------------------------------------------------------------
        ! 1. METALLICITY INTERPOLATION (Constant for all time steps)
        ! --------------------------------------------------------------------
        ! Calculate Log10(Z/Zsol)
        val_z_log = log10(ctx%state%zlegend(pset%zmet) / ctx%state%zsol)

        ! Find interval in Z grid
        idx_z = find_interval(grid_z, val_z_log)
        ! Clamp index to [1, n-1] for safety
        idx_z = max(1, min(idx_z, n_z_grid - 1))

        ! Calculate weight (0.0 to 1.0)
        weight_z = (val_z_log - grid_z(idx_z)) / (grid_z(idx_z+1) - grid_z(idx_z))
        weight_z = max(0.0_wp, min(weight_z, 1.0_wp)) ! No extrapolation

        ! --------------------------------------------------------------------
        ! 2. TIME LOOP
        ! --------------------------------------------------------------------
        do t_idx = 1, n_time_steps

            ! Find interval in Age grid
            idx_age = find_interval(grid_age, grid_time(t_idx))
            ! Clamp index to [1, n-1]
            idx_age = max(1, min(idx_age, n_age_grid - 1))

            ! Calculate age weight
            weight_age = (grid_time(t_idx) - grid_age(idx_age)) / &
                         (grid_age(idx_age+1) - grid_age(idx_age))
            
            ! Check if we are inside the valid XRB age grid
            if (weight_age < 0.0_wp .or. weight_age > 1.0_wp) cycle

            ! Pre-compute scalars
            w00 = pset%frac_xrb * (1.0_wp - weight_age) * (1.0_wp - weight_z)
            w10 = pset%frac_xrb * (weight_age)          * (1.0_wp - weight_z)
            w01 = pset%frac_xrb * (1.0_wp - weight_age) * (weight_z)
            w11 = pset%frac_xrb * (weight_age)          * (weight_z)
            
            ! Array operation using pre-computed scalars
            ! This reduces 4 multiplications per wavelength point to just addition/fma
            spec_out(:, t_idx) = spec_out(:, t_idx) + &
                                 w00 * grid_spec(:, idx_age,   idx_z)   + &
                                 w10 * grid_spec(:, idx_age+1, idx_z)   + &
                                 w01 * grid_spec(:, idx_age,   idx_z+1) + &
                                 w11 * grid_spec(:, idx_age+1, idx_z+1)

        end do

    end subroutine add_xray_binaries

    ! ------------------------------------------------------------------------
    ! PRIVATE HELPER ROUTINES
    ! ------------------------------------------------------------------------

end module fsps_stellar_modifications