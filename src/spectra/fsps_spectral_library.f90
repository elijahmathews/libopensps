module fsps_spectral_library
    !> @file fsps_spectral_library.f90
    !> @brief Core module for generating individual stellar spectra from theoretical and empirical libraries.
    !>
    !> @details
    !> This module serves as the "Spectral Engine" of FSPS. It replaces the legacy `GETSPEC` subroutine
    !> with a modernized, modular, and vectorized implementation. Its primary responsibility is to
    !> calculate the Spectral Energy Distribution (SED) for a single star (or isochrone point)
    !> defined by its physical parameters (Mass, Temperature, Luminosity, Gravity) and evolutionary phase.
    !>
    !> **Key Functionality:**
    !> 1. **Physical Consistency:** Enforces consistency between Mass, Luminosity, Temperature, and Gravity
    !>    by recalculating surface gravity and radius derived from the Stefan-Boltzmann law and Newton's laws.
    !>    This ensures that modifications to stellar properties (e.g., during the AGB phase) are physically 
    !>    reflected in the spectrum.
    !> 2. **Library Dispatch:** Automatically selects the appropriate spectral library based on:
    !>    - **Phase:** Main Sequence, Red Giant, TP-AGB, Post-AGB, Wolf-Rayet.
    !>    - **Temperature:** Hot star extensions (WMBasic) vs Cool star libraries.
    !>    - **Composition:** Carbon-rich vs Oxygen-rich AGB stars.
    !> 3. **High-Performance Interpolation:** Utilizes vectorized, whole-array arithmetic for
    !>    bilinear (2D) and linear (1D) interpolations, ensuring optimal performance on modern CPU architectures
    !>    by maintaining memory locality.
    !> 4. **Robustness:** Handles grid edge cases (partial overlaps) and numerical underflows using
    !>    safe floors and nearest-neighbor snapping where data is missing, preventing integration failures.
    !>
    !> **Supported Libraries:**
    !> - **Main:** BaSeL (theoretical) or MILES (empirical).
    !> - **Hot Stars:** WMBasic (O-stars), Rauch (Post-AGB).
    !> - **Wolf-Rayet:** CMFGEN (Smith et al. 2002).
    !> - **TP-AGB:** Lancon & Mouhcine (O-rich), Aringer or Lancon & Wood (C-rich).
    !>
    !> **Output Units:**
    !> Returns Luminosity Density in units of \f$ L_{\odot} / \mathrm{Hz} \f$. The output is automatically scaled:
    !> - For Surface Flux grids: Scaled by \f$ 4\pi R^2 \f$.
    !> - For Normalized SEDs: Scaled by \f$ L_{\mathrm{bol}} \f$.

    use fsps_precision, only: WP, KP
    use fsps_context_types, only: fsps_context_t
    use fsps_types, only: params
    use fsps_constants, only: SAFE_FLOOR, PI, YEAR_TO_SECOND, C_LIGHT, G_NEWTON, GRAVITY_L_M_T_COEFF, &
                              M_SOL, L_SOL, NDIM_WR, NDIM_PAGB, N_AGB_C, N_AGB_O, N_AGB_CAR, CSTAR_ARINGER
    use fsps_interpolation, only: find_interval
    use fsps_dust, only: apply_agb_dust_screen, agb_circumstellar_tau

    implicit none
    private

    public :: get_stellar_spectrum, get_stellar_terms
    public :: SRC_MAIN_LIB, SRC_WMBASIC, SRC_AGB_O, SRC_AGB_C, SRC_PAGB, SRC_WRN, SRC_WRC, SRC_AGB_CAR

    ! ------------------------------------------------------------------------
    ! CONSTANTS
    ! ------------------------------------------------------------------------
    integer, parameter :: SRC_MAIN_LIB = 0
    integer, parameter :: SRC_WMBASIC  = 1
    integer, parameter :: SRC_AGB_O    = 2
    integer, parameter :: SRC_AGB_C    = 3
    integer, parameter :: SRC_WR       = 4
    integer, parameter :: SRC_PAGB     = 5
    ! Template libraries of get_stellar_terms (WR and C stars split by template set)
    integer, parameter :: SRC_WRN      = 6
    integer, parameter :: SRC_WRC      = 7
    integer, parameter :: SRC_AGB_CAR  = 8

    ! Library cuts
    real(WP), parameter :: LOGT_CUT_PAGB = 4.699_wp
    real(WP), parameter :: LOGT_CUT_AGB_COOL = 3.6_wp

    ! Floor for spectra at kernel precision (SAFE_FLOOR in the default build)
    real(KP), parameter :: KP_FLOOR = tiny(1.0_KP)

    ! WR-specific constants
    real(WP), parameter :: V_INF_WR = 3.0e8_wp
    real(WP), parameter :: KAPPA_ES = 0.346_wp
    real(WP), parameter :: MIX_FAC  = 0.4_wp

contains

    !> @brief Main dispatcher: Calculates the spectrum for a single star.
    !>
    !> @details
    !> This routine replaces the legacy `GETSPEC`. It orchestrates the process:
    !> 1. Calculates consistent physical parameters (Radius, Gravity).
    !> 2. Determines the appropriate spectral library based on Phase/Temp.
    !> 3. Dispatches to the specific interpolation routine.
    !> 4. Applies physical scaling (Surface Flux -> Luminosity Density).
    !> 5. Applies circumstellar AGB dust (if applicable).
    !>
    !> @param[in]    ctx    The FSPS context (contains large lookup tables).
    !> @param[in]    pset   User parameters (contains dust/IMF settings).
    !> @param[in]    mact   Actual mass (M_sol).
    !> @param[in]    logt   Log(Teff).
    !> @param[in]    lbol   Luminosity (L_sol) (Linear units).
    !> @param[in]    logg   Log(g) (Input from isochrone).
    !> @param[in]    phase  Evolutionary phase flag.
    !> @param[in]    ffco   Composition flag (AGB C/O ratio, etc).
    !> @param[in]    lmdot  Log(Mass Loss Rate).
    !> @param[in]    wght   IMF Weight (used for context, not calculation).
    !> @param[out]   spec   The resulting spectrum (L_sun/Hz).
    subroutine get_stellar_spectrum(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, spec)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(inout) :: ctx
        type(params),         intent(in)    :: pset
        real(WP),             intent(in)    :: mact, logt, lbol, logg
        real(WP),             intent(in)    :: phase, ffco, lmdot
        real(KP),             intent(out)   :: spec(:)

        integer  :: src, nterm, i1(4), i2(4)
        real(WP) :: coef(4)
        logical  :: linear, dusty, smooth_ir

        ! 1. The star as a combination of library templates (library choice, consistent gravity
        !    and radius, interpolation corners: get_stellar_terms)
        call get_stellar_terms(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, &
                               src, i1, i2, coef, nterm, linear, allow_dust=.true., dusty=dusty)

        ! 2. Evaluate it. AGB stars with a circumstellar dust shell (phases 4, 5) use the
        !    pre-smoothed templates beyond 3 um (see precompute_agb_smoothed_templates), as legacy
        !    FSPS smooths their spectra before applying the DUSTY transfer function
        if (linear) then
            smooth_ir = .false.
            if (dusty .and. associated(ctx%state%speclib_agb_smooth)) then
                smooth_ir = agb_circumstellar_tau(ctx, pset%agb_dust, mact, logt, log10(lbol), logg, ffco, lmdot) &
                            > SAFE_FLOOR
            end if
            call apply_stellar_terms(ctx, pset%zmet, src, i1, i2, coef, nterm, smooth_ir, spec)
        else
            spec = KP_FLOOR   ! no valid main-library corner
        end if

        ! 3. Circumstellar AGB dust (the legacy interface takes log10(lbol) and the isochrone logg)
        if (dusty) then
            call apply_agb_dust_screen(ctx, pset%agb_dust, spec, mact, logt, log10(lbol), logg, ffco, lmdot)
        end if

        ! 4. Zero-luminosity stars keep a strictly positive spectrum
        if (lbol <= tiny(0.0_wp)) then
            spec = max(spec, KP_FLOOR)
        end if

    end subroutine get_stellar_spectrum

    !> @brief Evaluates sum(coef(n) * template(n)) of library `src` (terms of get_stellar_terms),
    !> with the pre-smoothed templates beyond 3 um when `smooth_ir`.
    subroutine apply_stellar_terms(ctx, nz, src, i1, i2, coef, nterm, smooth_ir, spec)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        integer,  intent(in)  :: nz, src, nterm, i1(4), i2(4)
        real(WP), intent(in)  :: coef(4)
        logical,  intent(in)  :: smooth_ir
        real(KP), intent(out) :: spec(:)
        integer :: n, i0

        spec = 0.0_KP
        select case (src)
        case (SRC_MAIN_LIB)
            do n = 1, nterm
                spec = spec + real(coef(n), KP) * ctx%state%speclib(:, nz, i1(n), i2(n))
            end do
            if (smooth_ir) then
                i0 = size(spec) - size(ctx%state%speclib_agb_smooth, 1) + 1
                spec(i0:) = 0.0_KP
                do n = 1, nterm
                    spec(i0:) = spec(i0:) + real(coef(n), KP) * ctx%state%speclib_agb_smooth(:, nz, i1(n), i2(n))
                end do
            end if
        case (SRC_WMBASIC)
            do n = 1, nterm
                spec = spec + real(coef(n), KP) * ctx%state%wmb_spec(:, nz, i1(n), i2(n))
            end do
        case (SRC_PAGB)
            do n = 1, nterm
                spec = spec + real(coef(n), KP) * ctx%state%pagb_spec(:, i1(n), i2(n))
            end do
        case (SRC_WRN)
            do n = 1, nterm
                spec = spec + real(coef(n), KP) * ctx%state%wrn_spec(:, i1(n), nz)
            end do
        case (SRC_WRC)
            do n = 1, nterm
                spec = spec + real(coef(n), KP) * ctx%state%wrc_spec(:, i1(n), nz)
            end do
        case (SRC_AGB_CAR)
            do n = 1, nterm
                spec = spec + real(coef(n), KP) * ctx%state%agb_spec_car(:, i1(n))
            end do
            if (smooth_ir .and. associated(ctx%state%agb_spec_car_smooth)) then
                i0 = size(spec) - size(ctx%state%agb_spec_car_smooth, 1) + 1
                spec(i0:) = 0.0_KP
                do n = 1, nterm
                    spec(i0:) = spec(i0:) + real(coef(n), KP) * ctx%state%agb_spec_car_smooth(:, i1(n))
                end do
            end if
        case (SRC_AGB_O, SRC_AGB_C)
            ! Per unit wavelength templates: lambda^2 (the 1/c is in coef); the smoothed templates
            ! already include lambda^2 / c
            if (src == SRC_AGB_O) then
                do n = 1, nterm
                    spec = spec + real(coef(n), KP) * ctx%state%agb_spec_o(:, i1(n))
                end do
            else
                do n = 1, nterm
                    spec = spec + real(coef(n), KP) * ctx%state%agb_spec_c(:, i1(n))
                end do
            end if
            spec = real(ctx%state%spec_lambda(1:size(spec)), KP)**2 * spec
            if (smooth_ir) then
                if (src == SRC_AGB_O .and. associated(ctx%state%agb_spec_o_smooth)) then
                    i0 = size(spec) - size(ctx%state%agb_spec_o_smooth, 1) + 1
                    spec(i0:) = 0.0_KP
                    do n = 1, nterm
                        spec(i0:) = spec(i0:) + real(coef(n) * C_LIGHT, KP) * ctx%state%agb_spec_o_smooth(:, i1(n))
                    end do
                else if (src == SRC_AGB_C .and. associated(ctx%state%agb_spec_c_smooth)) then
                    i0 = size(spec) - size(ctx%state%agb_spec_c_smooth, 1) + 1
                    spec(i0:) = 0.0_KP
                    do n = 1, nterm
                        spec(i0:) = spec(i0:) + real(coef(n) * C_LIGHT, KP) * ctx%state%agb_spec_c_smooth(:, i1(n))
                    end do
                end if
            end if
        end select
    end subroutine apply_stellar_terms

    ! ------------------------------------------------------------------------
    ! PRIVATE HELPER PROCEDURES
    ! ------------------------------------------------------------------------

    !> @brief Find interpolation interval on a strided row in a 2D array.
    !>
    !> @details
    !> Specialized helper for lookups like `array2d(row, :)` where row slices
    !> are non-contiguous in memory (column-major layout). This avoids creating
    !> temporary contiguous copies on every call.
    pure function find_interval_row_2d(array2d, row, ncol, value) result(idx)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: array2d(:, :)
        integer,  intent(in) :: row, ncol
        real(WP), intent(in) :: value
        integer :: idx

        integer :: lower, upper, mid
        logical :: is_ascending

        if (ncol < 2) then
            idx = 1
            return
        end if

        is_ascending = (array2d(row, ncol) >= array2d(row, 1))

        lower = 0
        upper = ncol + 1

        do while (upper - lower > 1)
            mid = (upper + lower) / 2
            if (is_ascending .eqv. (value >= array2d(row, mid))) then
                lower = mid
            else
                upper = mid
            end if
        end do

        if (value == array2d(row, 1)) then
            idx = 1
        else if (value == array2d(row, ncol)) then
            idx = ncol - 1
        else
            idx = lower
        end if

    end function find_interval_row_2d

    !> @brief A star's spectrum as a linear combination of library templates.
    !>
    !> @details
    !> Same physics as get_stellar_spectrum, which it must mirror: the spectrum is
    !> sum(coef(n) * template(i1(n), i2(n))) of library `src`, where the template is
    !> speclib(:, zmet, i1, i2) (main), wmb_spec(:, zmet, i1, i2) (WMBasic), pagb_spec(:, i1, i2),
    !> wrn_spec / wrc_spec(:, i1, zmet), agb_spec_car(:, i1), and lambda^2 * agb_spec_o / agb_spec_c(:, i1)
    !> for the O- and C-rich AGB templates (the lambda^2 factor is applied by the caller).
    !> get_stellar_spectrum evaluates these terms, so the two cannot disagree. `linear` is false
    !> for AGB stars with a circumstellar dust shell (unless allow_dust) and when no main-library
    !> corner is valid (floor spectrum). Zero-luminosity stars are floored by get_stellar_spectrum.
    subroutine get_stellar_terms(ctx, pset, mact, logt, lbol, logg, phase, ffco, lmdot, &
                                 src, i1, i2, coef, nterm, linear, allow_dust, dusty)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        type(params),         intent(in) :: pset
        real(WP),             intent(in) :: mact, logt, lbol, logg, phase, ffco, lmdot
        integer,              intent(out) :: src, nterm
        integer,              intent(out) :: i1(4), i2(4)
        real(WP),             intent(out) :: coef(4)
        logical,              intent(out) :: linear
        !> With allow_dust, AGB stars with a dust shell also return their template terms and set
        !> `dusty`; the caller applies the shell (get_agb_dust_terms) and the smoothed IR templates
        logical, intent(in),  optional :: allow_dust
        logical, intent(out), optional :: dusty

        integer  :: i_phase, library_source, jlo, klo, nz, nt, ng, k_z, n_grid
        real(WP) :: logg_calc, gravity_cgs, r2_cm, scale_factor, logt_cut_wmb, t, u
        real(WP) :: mdot_cgs, r_phot, r_wr, twr
        logical  :: v1, v2, v3, v4

        src = SRC_MAIN_LIB; nterm = 0; i1 = 1; i2 = 1; coef = 0.0_wp; linear = .false.
        i_phase = int(phase)
        if (present(dusty)) dusty = .false.
        if ((i_phase == 4 .or. i_phase == 5) .and. ctx%add_agb_dust_model_val == 1 .and. &
            pset%agb_dust > SAFE_FLOOR) then
            if (.not. present(allow_dust)) return
            if (.not. allow_dust) return
            if (present(dusty)) dusty = .true.
        end if

        library_source = SRC_MAIN_LIB
        if (i_phase == 6 .and. logt >= LOGT_CUT_PAGB) then
            library_source = SRC_PAGB
        else if (i_phase == 9 .and. ctx%use_wr_spectra_val == 1) then
            library_source = SRC_WR
        else if (i_phase == 5 .and. logt < LOGT_CUT_AGB_COOL) then
            if (ffco <= 1.0_wp) then
                library_source = SRC_AGB_O
            else
                library_source = SRC_AGB_C
            end if
        else
            logt_cut_wmb = max(ctx%state%wmb_logt(1), ctx%logt_wmb_hot_val)
            if (i_phase == 0 .and. logt > logt_cut_wmb) library_source = SRC_WMBASIC
        end if

        select case (library_source)
        case (SRC_MAIN_LIB, SRC_WMBASIC, SRC_WR)
            ! Gravity consistent with M, L and Teff (modified stars: AGB, HB, ...), and radius
            if (lbol > tiny(0.0_wp)) then
                logg_calc = log10(GRAVITY_L_M_T_COEFF * mact / lbol) + (4.0_wp * logt)
            else
                logg_calc = logg
            end if
            gravity_cgs = 10.0_wp**logg_calc
            if (gravity_cgs > tiny(0.0_wp)) then
                r2_cm = (mact * M_SOL * G_NEWTON) / gravity_cgs
            else
                r2_cm = 0.0_wp
            end if
        case default
            r2_cm = 0.0_wp
            logg_calc = logg
        end select
        if (library_source == SRC_MAIN_LIB) then
            scale_factor = (16.0_wp * PI * PI * r2_cm) / L_SOL
        else
            scale_factor = lbol
        end if

        nz = pset%zmet
        linear = .true.
        select case (library_source)
        case (SRC_PAGB)
            src = SRC_PAGB
            if (ctx%state%zlegend(pset%zmet) / ctx%state%zsol > 0.5_wp) then
                k_z = 2
            else
                k_z = 1
            end if
            jlo = max(1, min(find_interval(ctx%state%pagb_logt, logt), NDIM_PAGB - 1))
            t = max(0.0_wp, min((logt - ctx%state%pagb_logt(jlo)) / &
                                (ctx%state%pagb_logt(jlo+1) - ctx%state%pagb_logt(jlo)), 1.0_wp))
            nterm = 2; i1(1:2) = [jlo, jlo + 1]; i2(1:2) = k_z
            coef(1:2) = [(1.0_wp - t) * scale_factor, t * scale_factor]
        case (SRC_WR)
            if (ctx%state%isoc_type == 'mist') then
                mdot_cgs = (10.0_wp**lmdot) * M_SOL / YEAR_TO_SECOND
                r_phot = sqrt(r2_cm)
                r_wr   = r_phot + (3.0_wp * KAPPA_ES * abs(mdot_cgs)) / (8.0_wp * PI * V_INF_WR)
                twr = (10.0_wp**logt) * sqrt(r_phot / r_wr)
                twr = log10((1.0_wp - MIX_FAC) * (10.0_wp**logt) + MIX_FAC * twr)
            else
                twr = logt
            end if
            if (ffco < 10.0_wp) then
                src = SRC_WRN
                jlo = max(1, min(find_interval(ctx%state%wrn_logt, twr), NDIM_WR - 1))
                t = max(0.0_wp, min((twr - ctx%state%wrn_logt(jlo)) / &
                                    (ctx%state%wrn_logt(jlo+1) - ctx%state%wrn_logt(jlo)), 1.0_wp))
            else
                src = SRC_WRC
                jlo = max(1, min(find_interval(ctx%state%wrc_logt, twr), NDIM_WR - 1))
                t = max(0.0_wp, min((twr - ctx%state%wrc_logt(jlo)) / &
                                    (ctx%state%wrc_logt(jlo+1) - ctx%state%wrc_logt(jlo)), 1.0_wp))
            end if
            nterm = 2; i1(1:2) = [jlo, jlo + 1]; i2(1:2) = nz
            coef(1:2) = [(1.0_wp - t) * scale_factor, t * scale_factor]
        case (SRC_AGB_O)
            src = SRC_AGB_O
            jlo = max(1, min(find_interval_row_2d(ctx%state%agb_logt_o, nz, N_AGB_O, logt), N_AGB_O - 1))
            t = max(0.0_wp, min((logt - ctx%state%agb_logt_o(nz, jlo)) / &
                                (ctx%state%agb_logt_o(nz, jlo+1) - ctx%state%agb_logt_o(nz, jlo)), 1.0_wp))
            nterm = 2; i1(1:2) = [jlo, jlo + 1]
            coef(1:2) = [(scale_factor / C_LIGHT) * (1.0_wp - t), (scale_factor / C_LIGHT) * t]
        case (SRC_AGB_C)
            if (CSTAR_ARINGER == 1) then
                src = SRC_AGB_CAR
                n_grid = N_AGB_CAR
                jlo = max(1, min(find_interval(ctx%state%agb_logt_car, logt), n_grid - 1))
                t = max(0.0_wp, min((logt - ctx%state%agb_logt_car(jlo)) / &
                                    (ctx%state%agb_logt_car(jlo+1) - ctx%state%agb_logt_car(jlo)), 1.0_wp))
                coef(1:2) = [(1.0_wp - t) * scale_factor, t * scale_factor]
            else
                src = SRC_AGB_C
                n_grid = N_AGB_C
                jlo = max(1, min(find_interval(ctx%state%agb_logt_c, logt), n_grid - 1))
                t = max(0.0_wp, min((logt - ctx%state%agb_logt_c(jlo)) / &
                                    (ctx%state%agb_logt_c(jlo+1) - ctx%state%agb_logt_c(jlo)), 1.0_wp))
                coef(1:2) = [(scale_factor / C_LIGHT) * (1.0_wp - t), (scale_factor / C_LIGHT) * t]
            end if
            nterm = 2; i1(1:2) = [jlo, jlo + 1]
        case (SRC_WMBASIC)
            src = SRC_WMBASIC
            nt = size(ctx%state%wmb_logt); ng = size(ctx%state%wmb_logg)
            jlo = max(1, min(find_interval(ctx%state%wmb_logt, logt), nt - 1))
            klo = max(1, min(find_interval(ctx%state%wmb_logg, logg_calc), ng - 1))
            t = max(0.0_wp, min((logt - ctx%state%wmb_logt(jlo)) / &
                                (ctx%state%wmb_logt(jlo+1) - ctx%state%wmb_logt(jlo)), 1.0_wp))
            u = max(0.0_wp, min((logg_calc - ctx%state%wmb_logg(klo)) / &
                                (ctx%state%wmb_logg(klo+1) - ctx%state%wmb_logg(klo)), 1.0_wp))
            nterm = 4
            i1 = [jlo, jlo + 1, jlo + 1, jlo]; i2 = [klo, klo, klo + 1, klo + 1]
            coef = [scale_factor * (1.0_wp - t) * (1.0_wp - u), scale_factor * t * (1.0_wp - u), &
                    scale_factor * t * u, scale_factor * (1.0_wp - t) * u]
        case default
            src = SRC_MAIN_LIB
            nt = size(ctx%state%speclib_logt); ng = size(ctx%state%speclib_logg)
            jlo = max(1, min(find_interval(ctx%state%speclib_logt, logt), nt - 1))
            klo = max(1, min(find_interval(ctx%state%speclib_logg, logg_calc), ng - 1))
            t = max(0.0_wp, min((logt - ctx%state%speclib_logt(jlo)) / &
                                (ctx%state%speclib_logt(jlo+1) - ctx%state%speclib_logt(jlo)), 1.0_wp))
            u = max(0.0_wp, min((logg_calc - ctx%state%speclib_logg(klo)) / &
                                (ctx%state%speclib_logg(klo+1) - ctx%state%speclib_logg(klo)), 1.0_wp))
            ! Corner validity and the legacy snapping order, as in get_main_lib_spectrum
            v1 = (ctx%state%speclib(ctx%state%whlam5000, nz, jlo,   klo)   > SAFE_FLOOR)
            v2 = (ctx%state%speclib(ctx%state%whlam5000, nz, jlo+1, klo)   > SAFE_FLOOR)
            v3 = (ctx%state%speclib(ctx%state%whlam5000, nz, jlo+1, klo+1) > SAFE_FLOOR)
            v4 = (ctx%state%speclib(ctx%state%whlam5000, nz, jlo,   klo+1) > SAFE_FLOOR)
            i1 = [jlo, jlo + 1, jlo + 1, jlo]; i2 = [klo, klo, klo + 1, klo + 1]
            if (v1 .and. v2 .and. v3 .and. v4) then
                nterm = 4
                coef = [scale_factor * (1.0_wp - t) * (1.0_wp - u), scale_factor * t * (1.0_wp - u), &
                        scale_factor * t * u, scale_factor * (1.0_wp - t) * u]
            else if (.not. (v1 .or. v2 .or. v3 .or. v4)) then
                linear = .false.   ! floor spectrum
            else
                nterm = 1
                if (v3) then
                    i1(1) = jlo + 1; i2(1) = klo + 1
                else if (v4) then
                    i1(1) = jlo; i2(1) = klo + 1
                else if (v2) then
                    i1(1) = jlo + 1; i2(1) = klo
                else
                    i1(1) = jlo; i2(1) = klo
                end if
                coef(1) = scale_factor
            end if
        end select
    end subroutine get_stellar_terms

end module fsps_spectral_library