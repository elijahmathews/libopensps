module fsps_imf
    !> @brief
    !> Handles Initial Mass Function (IMF) definitions and weighting.
    !>
    !> @details
    !> Contains routines to evaluate the IMF (dn/dM) for various standard
    !> parameterizations (Salpeter, Chabrier, Kroupa, etc.) and to compute
    !> the mass weights for SSP generation.
    
    use fsps_precision, only: WP
    use fsps_context_types, only: fsps_context_t
    implicit none

    private

    public :: compute_imf_weights
    public :: get_imf_value
    public :: integrate_imf_interval_analytic, imf_integral, imf_integral_p, imf_pack_params
    public :: IMF_NPAR, IMF_P_SALP, IMF_P_ALPHA, IMF_P_VDMC, IMF_P_MDAVE, IMF_P_LOWER_LIMIT, &
              IMF_P_UPPER_LIMIT, IMF_P_LOWER_BOUND, IMF_P_MLIM_BH, IMF_P_MLIM_NS, IMF_P_NUSER, IMF_P_USER
    public :: CHAB_MC, CHAB_SIGMA2, CHAB_IND

    ! ------------------------------------------------------------------------
    ! CONSTANTS
    ! ------------------------------------------------------------------------

    ! Chabrier 2003
    real(WP), parameter :: CHAB_MC     = 0.08_wp
    real(WP), parameter :: CHAB_SIGMA2 = 0.69_wp**2  ! Explicit squaring
    real(WP), parameter :: CHAB_IND    = 1.3_wp
    real(WP), parameter :: CHAB_LOG_MC = log10(CHAB_MC)
    real(WP), parameter :: CHAB_INV_2SIG2 = 1.0_wp / (2.0_wp * CHAB_SIGMA2)
    real(WP), parameter :: CHAB_HIGH_NORM = exp(-(CHAB_LOG_MC * CHAB_LOG_MC) * CHAB_INV_2SIG2)

    ! van Dokkum 2008
    real(WP), parameter :: VD_SIGMA2 = 0.69_wp**2
    real(WP), parameter :: VD_AH     = 0.0443_wp
    real(WP), parameter :: VD_IND    = 1.3_wp
    real(WP), parameter :: VD_AL     = 0.14_wp
    real(WP), parameter :: VD_NC     = 25.0_wp

    ! ------------------------------------------------------------------------
    ! PACKED IMF PARAMETERS
    ! ------------------------------------------------------------------------
    ! One IMF parameter set as a plain real vector p(IMF_NPAR) (the batched SSP operator keeps one
    ! per model; the IMF type is an integer passed alongside). Filled by imf_pack_params from
    ! the context after configure_imf_parameters.
    integer, parameter :: IMF_P_SALP = 1          !< Salpeter slope (ctx%state%salp_ind)
    integer, parameter :: IMF_P_ALPHA = 2         !< 2..4: Kroupa / Dave slopes (imf1, imf2, imf3)
    integer, parameter :: IMF_P_VDMC = 5          !< van Dokkum characteristic mass
    integer, parameter :: IMF_P_MDAVE = 6         !< Dave break mass
    integer, parameter :: IMF_P_LOWER_LIMIT = 7   !< imf_lower_limit (M_sun)
    integer, parameter :: IMF_P_UPPER_LIMIT = 8   !< imf_upper_limit (M_sun)
    integer, parameter :: IMF_P_LOWER_BOUND = 9   !< lower edge of the first isochrone mass bin
    integer, parameter :: IMF_P_MLIM_BH = 10      !< lowest initial mass forming a black hole
    integer, parameter :: IMF_P_MLIM_NS = 11      !< lowest initial mass forming a neutron star
    integer, parameter :: IMF_P_NUSER = 12        !< number of user IMF segments (as a real)
    integer, parameter :: IMF_P_USER = 13         !< 13..312: user segments (lower, upper, slope) x 100
    integer, parameter :: IMF_NPAR = IMF_P_USER + 3 * 100 - 1

contains

    !> Packs the context's IMF parameters (after configure_imf_parameters) into p(IMF_NPAR).
    subroutine imf_pack_params(ctx, p)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(out) :: p(IMF_NPAR)
        p = 0.0_wp
        p(IMF_P_SALP) = ctx%state%salp_ind
        p(IMF_P_ALPHA:IMF_P_ALPHA + 2) = ctx%state%imf_alpha
        p(IMF_P_VDMC) = ctx%state%imf_vdmc
        p(IMF_P_MDAVE) = ctx%state%imf_mdave
        p(IMF_P_LOWER_LIMIT) = ctx%state%imf_lower_limit
        p(IMF_P_UPPER_LIMIT) = ctx%state%imf_upper_limit
        p(IMF_P_LOWER_BOUND) = ctx%state%imf_lower_bound
        p(IMF_P_MLIM_BH) = ctx%state%mlim_bh
        p(IMF_P_MLIM_NS) = ctx%state%mlim_ns
        p(IMF_P_NUSER) = real(ctx%state%n_user_imf, WP)
        p(IMF_P_USER:IMF_NPAR) = reshape(ctx%state%imf_user_alpha, [3 * 100])
    end subroutine imf_pack_params

    !> imf_integral with the IMF parameters packed in p(IMF_NPAR) (imf_pack_params).
    pure function imf_integral_p(imf_type, p, m1, m2, mass_weighted) result(int_val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)  :: imf_type
        real(WP), intent(in) :: p(IMF_NPAR)
        real(WP), intent(in) :: m1, m2
        logical, intent(in)  :: mass_weighted
        real(WP) :: int_val
        int_val = imf_integral(imf_type, p(IMF_P_SALP), p(IMF_P_ALPHA:IMF_P_ALPHA + 2), p(IMF_P_VDMC), &
                               p(IMF_P_MDAVE), nint(p(IMF_P_NUSER)), p(IMF_P_USER:IMF_NPAR), m1, m2, mass_weighted)
    end function imf_integral_p

    !> @brief
    !> Computes the number of stars in each mass bin, normalized to 1 Msun total mass.
    !>
    !> @details
    !> 1. Calculates the number of stars in each mass bin i (integrating dn/dm).
    !> 2. Calculates the total mass of the system (integrating m*dn/dm).
    !> 3. Normalizes the weights so the total mass of the population is 1.0.
    !>
    !> @param[inout] ctx    Simulation context (contains IMF params).
    !> @param[in]    mini   Initial masses of the isochrone points.
    !> @param[out]   weights Output weights (Number of stars per unit total mass).
    !> @param[in]    nmass  Number of mass points.
    subroutine compute_imf_weights(ctx, mini, weights, nmass)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(:), intent(in) :: mini
        real(WP), dimension(:), intent(out) :: weights
        integer, intent(in) :: nmass

        real(WP) :: m1, m2, total_mass
        integer :: i

        ! Access limits from state
        real(WP) :: lower_limit, upper_limit, lower_bound

        lower_limit = ctx%state%imf_lower_limit
        upper_limit = ctx%state%imf_upper_limit
        lower_bound = ctx%state%imf_lower_bound
        weights = 0.0_wp

        ! 1. Calculate Number of Stars in each bin
        do i = 1, nmass
            
            ! Skip bins outside the requested IMF integration range
            if (mini(i) < lower_limit .or. mini(i) > upper_limit) cycle

            ! Define Bin Edges
            if (i == 1) then
                ! Special case for first bin (Geneva models)
                m1 = lower_bound
            else
                m1 = mini(i) - 0.5_wp * (mini(i) - mini(i-1))
            end if

            if (i == nmass) then
                m2 = mini(i)
            else
                m2 = mini(i) + 0.5_wp * (mini(i+1) - mini(i))
            end if

            ! Validate Monotonicity
            if (m2 < m1) then
                write(*, '("[FSPS-IMF] Warning: Non-monotonic mass bin at index ", I0, &
                         & " (m1=", ES10.3, ", m2=", ES10.3, ")")') i, m1, m2
                cycle
            end if
            
            ! Skip empty bins
            if (m2 == m1) cycle

            ! Integrate dn/dM over [m1, m2]
            ! Uses the wrapper_imf_count to integrate number density
            weights(i) = integrate_imf_interval_analytic(ctx, m1, m2, mass_weighted=.false.)
        end do

        ! 2. Calculate Total Mass (Normalization Factor)
        ! Integrate M * dn/dM over the full range [lower_limit, upper_limit]
        total_mass = integrate_imf_interval_analytic(ctx, lower_limit, upper_limit, mass_weighted=.true.)

        ! 3. Normalize
        if (total_mass > 0.0_wp) then
            weights = weights / total_mass
        end if

    end subroutine compute_imf_weights

    ! ------------------------------------------------------------------------
    ! INTEGRATION WRAPPERS
    ! These match the interface required by fsps_integration
    ! ------------------------------------------------------------------------

    !> @brief Analytic integral of the IMF over [m1, m2] (number, or mass when mass_weighted).
    !>
    !> @details
    !> Closed forms for every IMF type (mod(ctx%imf_type_val, 10)): 0 Salpeter, 1 Chabrier,
    !> 2 Kroupa, 3 van Dokkum, 4 Dave, 5 user-defined piecewise power law. The integrals are
    !> imf_integral (plain arguments, also used by the batched SSP operator, fsps_ssp_operator).
    pure function integrate_imf_interval_analytic(ctx, m1, m2, mass_weighted) result(int_val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: m1, m2
        logical, intent(in)  :: mass_weighted
        real(WP) :: int_val

        int_val = imf_integral(ctx%imf_type_val, ctx%state%salp_ind, ctx%state%imf_alpha, ctx%state%imf_vdmc, &
                               ctx%state%imf_mdave, ctx%state%n_user_imf, ctx%state%imf_user_alpha, &
                               m1, m2, mass_weighted)
    end function integrate_imf_interval_analytic

    !> @brief Analytic integral of the IMF over [m1, m2], with the IMF parameters as plain
    !> arguments (no context): usable from device code for a batch of IMF parameter sets.
    !>
    !> @details
    !> imf_type: as ctx%imf_type_val (its base type mod 10 is used); alpha: Kroupa / Dave slopes
    !> (ctx%state%imf_alpha); user_alpha(3, n_user_imf): user segments (lower mass, upper mass,
    !> slope), as ctx%state%imf_user_alpha. Unsupported types integrate to zero.
    pure function imf_integral(imf_type, salp_ind, alpha, vdmc, mdave, n_user_imf, user_alpha, &
                               m1, m2, mass_weighted) result(int_val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in)  :: imf_type, n_user_imf
        real(WP), intent(in) :: salp_ind, alpha(3), vdmc, mdave
        real(WP), intent(in) :: user_alpha(3, *)
        real(WP), intent(in) :: m1, m2
        logical, intent(in)  :: mass_weighted
        real(WP) :: int_val

        integer :: imf_type_base
        real(WP) :: lo, hi, a1, a2, a3, c2, vd_break
        real(WP) :: seg_lo, seg_hi, imfcu
        integer  :: n, n_user

        int_val = 0.0_wp
        lo = min(m1, m2)
        hi = max(m1, m2)
        if (hi <= lo) return

        imf_type_base = mod(imf_type, 10)

        select case (imf_type_base)
        case (0)
            int_val = integrate_powerlaw(lo, hi, salp_ind, mass_weighted, 1.0_wp)

        case (1)
            ! Chabrier (2003): lognormal below 1 M_sun (closed form with erf), power law above
            seg_lo = lo
            seg_hi = min(hi, 1.0_wp)
            if (seg_hi > seg_lo) int_val = int_val + integrate_lognormal(seg_lo, seg_hi, CHAB_LOG_MC, CHAB_SIGMA2, mass_weighted)
            seg_lo = max(lo, 1.0_wp)
            seg_hi = hi
            if (seg_hi > seg_lo) int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, CHAB_IND + 1.0_wp, &
                                                                       mass_weighted, CHAB_HIGH_NORM)

        case (3)
            ! van Dokkum (2008): scaled lognormal up to nc * mc, power law above (get_imf_value)
            vd_break = VD_NC * vdmc
            seg_lo = lo
            seg_hi = min(hi, vd_break)
            if (seg_hi > seg_lo) int_val = int_val + VD_AL * (0.5_wp * vd_break)**(-VD_IND) * &
                integrate_lognormal(seg_lo, seg_hi, log10(vdmc), VD_SIGMA2, mass_weighted)
            seg_lo = max(lo, vd_break)
            seg_hi = hi
            if (seg_hi > seg_lo) int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, VD_IND + 1.0_wp, &
                                                                       mass_weighted, VD_AH)

        case (2)
            a1 = alpha(1)
            a2 = alpha(2)
            a3 = alpha(3)
            c2 = 0.5_wp**(-a1 + a2)

            seg_lo = max(lo, 0.08_wp)
            seg_hi = min(hi, 0.5_wp)
            if (seg_hi > seg_lo) int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, a1, mass_weighted, 1.0_wp)

            seg_lo = max(lo, 0.5_wp)
            seg_hi = min(hi, 1.0_wp)
            if (seg_hi > seg_lo) int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, a2, mass_weighted, c2)

            seg_lo = max(lo, 1.0_wp)
            seg_hi = hi
            if (seg_hi > seg_lo) int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, a3, mass_weighted, c2)

        case (4)
            a1 = alpha(1)
            a2 = alpha(2)
            
            seg_lo = max(lo, 0.08_wp)
            seg_hi = min(hi, mdave)
            if (seg_hi > seg_lo) int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, a1, mass_weighted, 1.0_wp)

            seg_lo = max(lo, mdave)
            seg_hi = hi
            if (seg_hi > seg_lo) then
                int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, a2, mass_weighted, mdave**(-a1 + a2))
            end if

        case (5)
            n_user = n_user_imf
            imfcu = 1.0_wp

            ! First segment
            if (n_user >= 1) then
                seg_lo = max(lo, user_alpha(1,1))
                seg_hi = min(hi, user_alpha(2,1))
                if (seg_hi > seg_lo) then
                    int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, user_alpha(3,1), mass_weighted, 1.0_wp)
                end if
            end if

            do n = 2, n_user
                imfcu = imfcu * user_alpha(1,n)**(-user_alpha(3,n-1) + &
                        user_alpha(3,n))

                seg_lo = max(lo, user_alpha(1,n))
                seg_hi = min(hi, user_alpha(2,n))
                if (seg_hi > seg_lo) then
                    int_val = int_val + integrate_powerlaw(seg_lo, seg_hi, user_alpha(3,n), mass_weighted, imfcu)
                end if
            end do

        case default
            int_val = 0.0_wp
        end select
    end function imf_integral

    !> Integral of a lognormal IMF segment, exp(-(log10 m - log_mc)^2 / (2 sigma2)) / m, over
    !> [m1, m2], or of m times it when mass_weighted (Chabrier and van Dokkum below their break).
    !> With x = log10 m and k = ln 10:
    !>   number: k sigma sqrt(pi/2) [erf((x2 - a)/(sigma sqrt 2)) - erf((x1 - a)/(sigma sqrt 2))], a = log_mc
    !>   mass:   the same about b = a + k sigma^2, times exp(k a + k^2 sigma^2 / 2)
    pure function integrate_lognormal(m1, m2, log_mc, sigma2, mass_weighted) result(val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: m1, m2, log_mc, sigma2
        logical,  intent(in) :: mass_weighted
        real(WP) :: val
        real(WP), parameter :: K = log(10.0_wp)
        real(WP) :: sig, centre, scale
        sig = sqrt(sigma2)
        if (mass_weighted) then
            centre = log_mc + K * sigma2
            scale = exp(K * log_mc + 0.5_wp * K * K * sigma2)
        else
            centre = log_mc
            scale = 1.0_wp
        end if
        val = scale * K * sig * sqrt(0.5_wp * acos(-1.0_wp)) * &
              (erf((log10(m2) - centre) / (sig * sqrt(2.0_wp))) - erf((log10(m1) - centre) / (sig * sqrt(2.0_wp))))
    end function integrate_lognormal

    !> @brief Integral of coeff * m^(-alpha) or coeff * m^(1-alpha) over [lo, hi].
    pure function integrate_powerlaw(lo, hi, alpha, mass_weighted, coeff) result(val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: lo, hi, alpha, coeff
        logical, intent(in)  :: mass_weighted
        real(WP) :: val
        real(WP) :: denom

        if (hi <= lo) then
            val = 0.0_wp
            return
        end if

        if (mass_weighted) then
            denom = 2.0_wp - alpha
        else
            denom = 1.0_wp - alpha
        end if

        if (abs(denom) > 1.0e-12_wp) then
            if (mass_weighted) then
                val = coeff * (hi**(2.0_wp - alpha) - lo**(2.0_wp - alpha)) / denom
            else
                val = coeff * (hi**(1.0_wp - alpha) - lo**(1.0_wp - alpha)) / denom
            end if
        else
            ! log integral limit when exponent is -1
            val = coeff * log(hi / lo)
        end if
    end function integrate_powerlaw

    ! ------------------------------------------------------------------------
    ! CORE IMF LOGIC
    ! ------------------------------------------------------------------------

    !> @brief
    !> Evaluates the Initial Mass Function (IMF) at specific mass points.
    !>
    !> @details
    !> Calculates the unnormalized IMF value for the algorithm specified in `ctx%imf_type_val`.
    !> Supported IMF types (determined by `mod(ctx%imf_type_val, 10)`):
    !> - 0: Salpeter ([1955](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S))
    !> - 1: Chabrier ([2003](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C))
    !> - 2: Kroupa ([2001](https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K))
    !> - 3: van Dokkum ([2008](https://ui.adsabs.harvard.edu/abs/2008ApJ...674...29V))
    !> - 4: Dave ([2008](https://ui.adsabs.harvard.edu/abs/2008MNRAS.385..147D))
    !> - 5: User-defined (piecewise power law)
    !>
    !> @param[in] ctx            The simulation context containing IMF parameters.
    !> @param[in] mass           Array of stellar masses (in M_sun).
    !> @param[in] mass_weighted  Control flag for the output form:
    !>                           - .false.: Returns Number Density (dN/dM).
    !>                           - .true.:  Returns Mass Density (M * dN/dM approx dN/dlnM).
    !> @return    imf_val        The calculated IMF values.
    pure function get_imf_value(ctx, mass, mass_weighted) result(imf_val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: mass
        logical, intent(in) :: mass_weighted
        real(WP) :: imf_val
        
        integer :: imf_type_base
        real(WP) :: alpha1, alpha2, alpha3
        real(WP) :: log_m, vdmc, vd_break, vd_term, mdave
        real(WP) :: imfcu
        integer  :: n, n_user

        ! The original code used (type + 10) to signal mass-weighting.
        ! We strip that here to get the base algorithm type.
        imf_type_base = mod(ctx%imf_type_val, 10)

        select case (imf_type_base)
        case (0) 
            ! Salpeter (1955)
            imf_val = mass**(-ctx%state%salp_ind)
        case (1) 
            ! Chabrier (2003)
            if (mass < 1.0_wp) then
                log_m = log10(mass)
                imf_val = exp(-((log_m - CHAB_LOG_MC)**2) * CHAB_INV_2SIG2) / mass
            else
                imf_val = CHAB_HIGH_NORM * mass**(-(CHAB_IND + 1.0_wp))
            end if
        case (2) 
            ! Kroupa (2001)
            alpha1 = ctx%state%imf_alpha(1)
            alpha2 = ctx%state%imf_alpha(2)
            alpha3 = ctx%state%imf_alpha(3)

            if (mass >= 0.08_wp .and. mass < 0.5_wp) then
                imf_val = mass**(-alpha1)
            else if (mass >= 0.5_wp .and. mass < 1.0_wp) then
                imf_val = 0.5_wp**(-alpha1 + alpha2) * mass**(-alpha2)
            else if (mass >= 1.0_wp) then
                imf_val = 0.5_wp**(-alpha1 + alpha2) * mass**(-alpha3)
            else
                imf_val = 0.0_wp
            end if
        case (3) 
            ! van Dokkum (2008)
            vdmc = ctx%state%imf_vdmc
            vd_break = VD_NC * vdmc

            if (mass <= vd_break) then
                log_m = log10(mass)
                vd_term = ((log_m - log10(vdmc))**2) / (2.0_wp * VD_SIGMA2)
                imf_val = VD_AL * (0.5_wp * vd_break)**(-VD_IND) * exp(-vd_term)
            else
                imf_val = VD_AH * mass**(-VD_IND)
            end if

            ! Convert from dn/dlnM to dn/dM
            imf_val = imf_val / mass
        case (4) 
            ! Dave (2008)
            alpha1 = ctx%state%imf_alpha(1)
            alpha2 = ctx%state%imf_alpha(2)
            mdave  = ctx%state%imf_mdave

            if (mass >= 0.08_wp .and. mass < mdave) then
                imf_val = mass**(-alpha1)
            else if (mass >= mdave) then
                imf_val = mdave**(-alpha1 + alpha2) * mass**(-alpha2)
            else
                imf_val = 0.0_wp
            end if
        case (5) 
            ! User-defined
            n_user = ctx%state%n_user_imf
            imf_val = 0.0_wp

            if (mass >= ctx%state%imf_user_alpha(1,1) .and. &
                mass <  ctx%state%imf_user_alpha(2,1)) then
                imf_val = mass**(-ctx%state%imf_user_alpha(3,1))
            end if

            imfcu = 1.0_wp
            do n = 2, n_user
                if (mass >= ctx%state%imf_user_alpha(1,n) .and. &
                    mass <  ctx%state%imf_user_alpha(2,n)) then

                    imf_val = mass**(-ctx%state%imf_user_alpha(3,n)) * &
                              ctx%state%imf_user_alpha(1,n)**(-ctx%state%imf_user_alpha(3,n-1) + &
                              ctx%state%imf_user_alpha(3,n)) * imfcu
                end if

                imfcu = imfcu * ctx%state%imf_user_alpha(1,n)**(-ctx%state%imf_user_alpha(3,n-1) + &
                        ctx%state%imf_user_alpha(3,n))
            end do
        case default
            imf_val = 0.0_wp
        end select

        ! Apply Mass Weighting if requested
        if (mass_weighted) then
            imf_val = imf_val * mass
        end if

    end function get_imf_value

    ! ------------------------------------------------------------------------
    ! SPECIFIC IMF IMPLEMENTATIONS
    ! ------------------------------------------------------------------------

    pure subroutine imf_salpeter(ctx, m, val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: m
        real(WP), intent(out) :: val
        
        ! Power law: m^(-alpha)
        val = m**(-ctx%state%salp_ind)
    end subroutine imf_salpeter

    pure subroutine imf_chabrier(m, val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: m
        real(WP), intent(out) :: val
        real(WP) :: log_m, log_mc, term

        log_mc = log10(CHAB_MC)

        if (m < 1.0_wp) then
            ! Log-normal part
            log_m = log10(m)
            term = (log_m - log_mc)**2 / (2.0_wp * CHAB_SIGMA2)
            val = exp(-term) / m
        else
            ! Power law part
            term = log_mc**2 / (2.0_wp * CHAB_SIGMA2)
            val = exp(-term) * m**(-CHAB_IND) / m
        end if
    end subroutine imf_chabrier

    pure subroutine imf_kroupa(ctx, m, val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: m
        real(WP), intent(out) :: val
        real(WP), dimension(3) :: alpha

        alpha = ctx%state%imf_alpha

        if (m >= 0.08_wp .and. m < 0.5_wp) then
            val = m**(-alpha(1))
        else if (m >= 0.5_wp .and. m < 1.0_wp) then
            val = 0.5_wp**(-alpha(1) + alpha(2)) * m**(-alpha(2))
        else if (m >= 1.0_wp) then
            val = 0.5_wp**(-alpha(1) + alpha(2)) * m**(-alpha(3))
        else
            val = 0.0_wp
        end if
    end subroutine imf_kroupa

    pure subroutine imf_vandokkum(ctx, m, val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: m
        real(WP), intent(out) :: val
        real(WP) :: breakpoint, term, log_m, log_mc

        breakpoint = VD_NC * ctx%state%imf_vdmc
        log_mc = log10(ctx%state%imf_vdmc)

        if (m <= breakpoint) then
            ! Lognormal-ish
            log_m = log10(m)
            term = (log_m - log_mc)**2 / (2.0_wp * VD_SIGMA2)
            val = VD_AL * (0.5_wp * breakpoint)**(-VD_IND) * exp(-term)
        else
            ! Power law
            val = VD_AH * m**(-VD_IND)
        end if

        ! Convert from dn/dlnM to dn/dM
        val = val / m
    end subroutine imf_vandokkum

    pure subroutine imf_dave(ctx, m, val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: m
        real(WP), intent(out) :: val
        real(WP), dimension(3) :: alpha
        real(WP) :: mdave

        alpha = ctx%state%imf_alpha
        mdave = ctx%state%imf_mdave

        if (m >= 0.08_wp .and. m < mdave) then
            val = m**(-alpha(1))
        else if (m >= mdave) then
            val = mdave**(-alpha(1) + alpha(2)) * m**(-alpha(2))
        else
            val = 0.0_wp
        end if
    end subroutine imf_dave

    pure subroutine imf_user_defined(ctx, m, val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: m
        real(WP), intent(out) :: val
        integer :: n
        real(WP) :: imfcu
        integer :: n_user

        ! Alias for cleaner syntax
        n_user = ctx%state%n_user_imf

        val = 0.0_wp

        ! First segment
        if (m >= ctx%state%imf_user_alpha(1,1) .and. &
            m <  ctx%state%imf_user_alpha(2,1)) then
            val = m**(-ctx%state%imf_user_alpha(3,1))
        end if

        ! Subsequent segments
        imfcu = 1.0_wp
        do n = 2, n_user
            if (m >= ctx%state%imf_user_alpha(1,n) .and. &
                m <  ctx%state%imf_user_alpha(2,n)) then
                
                val = m**(-ctx%state%imf_user_alpha(3,n)) * &
                         ctx%state%imf_user_alpha(1,n)**(-ctx%state%imf_user_alpha(3,n-1) + &
                         ctx%state%imf_user_alpha(3,n)) * imfcu
            end if
            
            ! Update cumulative factor
            imfcu = imfcu * ctx%state%imf_user_alpha(1,n)**(-ctx%state%imf_user_alpha(3,n-1) + &
                    ctx%state%imf_user_alpha(3,n))
        end do
    end subroutine imf_user_defined

end module fsps_imf