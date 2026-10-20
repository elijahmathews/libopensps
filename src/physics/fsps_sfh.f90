module fsps_sfh
    !> @brief
    !> Core physics module for Star Formation History (SFH) integration and property calculation.
    !>
    !> @details
    !> This module consolidates all logic related to the time-evolution of star formation rates.
    !> It is responsible for transforming a continuous or tabular SFR(t) into discrete weights
    !> for the Simple Stellar Population (SSP) grid, as well as calculating derived physical
    !> properties like mass fractions, mean stellar ages, and specific SFRs.
    !>
    !> **Supported SFH Types:**
    !> * **0:** Simple Stellar Population (SSP) - Instantaneous burst at t=0.
    !> * **1:** $\tau$-Model (Exponential) - $\exp(-t/\tau)$.
    !> * **2:** Tabular SFH - Read from file.
    !> * **3:** Tabular SFH - Passed via array.
    !> * **4:** Delayed $\tau$-Model - $t \exp(-t/\tau)$.
    !> * **5:** Simha Model - Linear-Exponential cutoff.
    !>
    !> **Key Features & Optimizations:**
    !> * **Moment Method:** Uses an optimized integration scheme for SSP weighting that calculates
    !>   the 0th and 1st moments of the SFH over time segments, reducing expensive transcendental
    !>   function evaluations by ~50% compared to legacy node-based integration.
    !> * **Dual Pathways:** Dispatches to exact analytic solutions for parameterized SFHs (Types 1, 4, 5)
    !>   and robust numerical integration (Trapezoidal) for arbitrary tabular data (Types 2, 3).
    !> * **Type Safety:** Fully modernized, strictly typed (using `WP`), and encapsulated.

    use fsps_precision, only: WP
    use fsps_context_types, only: fsps_context_t
    use fsps_types, only: params, sfhparams, compspout
    use fsps_constants, only: SAFE_FLOOR, NTABMAX
    use fsps_special_functions, only: expi, gammainc
    use fsps_interpolation, only: find_interval
    use fsps_integration, only: integrate_trapezoid_array
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite

    implicit none
    private

    public :: compute_ssp_weights
    public :: get_sfh_properties_at_age
    public :: sfh_weights_core, convert_sfhparams
    public :: compute_sfh_statistics

    ! ------------------------------------------------------------------------
    ! CONSTANTS
    ! ------------------------------------------------------------------------
    !> @brief Natural logarithm of Euler's number (ln(10) inverse or similar usage)
    real(WP), parameter :: LOG_E = log10(exp(1.0_wp)) 
    
    !> @brief Minimum valid SFR to avoid underflow
    real(WP), parameter :: MIN_SFR = 1.0e-30_wp

    !> @brief Conversion from Gyr to Years
    real(WP), parameter :: GYR_TO_YR = 1.0e9_wp
    real(WP), parameter :: LOG_GYR_TO_YR = log10(GYR_TO_YR)

contains

    ! ========================================================================
    ! MAIN DRIVER ROUTINES
    ! ========================================================================

    !> @brief Calculates the weights for all SSPs to form the composite spectrum.
    !>
    !> @details
    !> Determines the contribution of each Simple Stellar Population (SSP) to the 
    !> total spectrum based on the Star Formation History (SFH).
    !>
    !> **Algorithm:**
    !> Uses the "Moment Method" optimization. Instead of integrating the weight function 
    !> for every SSP node individually (which duplicates work for overlapping intervals), 
    !> this routine iterates over the time segments defined by the SSP grid. 
    !>
    !> **Optimizations:**
    !> 1. **Loop Carry-Over:** For analytic SFHs, the indefinite integral at the upper 
    !>    bound of segment `j` is cached and reused as the lower bound for segment `j+1`.
    !> 2. **Vectorized Math:** Node values are computed once per step to determine `dt`,
    !>    avoiding redundant power/log calls.
    !>
    !> @param[in]    ctx      The FSPS context (contains time grids `time_full`).
    !> @param[in]    sfh      The SFH parameters structure.
    !> @param[in]    idx_min  Index of the youngest SSP to consider (optimization floor).
    !> @param[in]    idx_max  Index of the oldest SSP to consider (optimization ceiling).
    !> @param[inout] weights  The calculated weights for each SSP (size: `ntfull`).
    pure subroutine compute_ssp_weights(ctx, sfh, idx_min, idx_max, weights)
        type(fsps_context_t), intent(in) :: ctx
        type(SFHPARAMS), intent(in) :: sfh
        integer, intent(in) :: idx_min, idx_max
        ! Note: weights is intent(inout) rather than intent(out) to bypass
        ! NVHPC array descriptor reallocation bugs on the device.
        real(WP), dimension(:), intent(inout), contiguous :: weights

        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif

        integer :: j, nt
        integer :: j_start, j_end
        real(WP) :: inv_dt
        real(WP) :: m0, m1
        real(WP), dimension(2) :: limits
        real(WP) :: log_tb, burst_dt
        
        ! Optimization variables (carry-over)
        real(WP) :: m0_prev, m1_prev
        real(WP) :: m0_curr, m1_curr
        real(WP) :: node_lo, node_hi, lim_prev
        logical :: is_tabular
        
        ! Pre-calculated boundaries
        real(WP) :: log_t_start, log_t_end
        real(WP) :: t_start_lin, t_end_lin, limit_min_lin

        weights = 0.0_wp
        nt = ctx%state%ntfull

        ! --- CASE: SINGLE BURST (Type -1) ---
        if (sfh%type == -1) then
            if (sfh%tb < 0.0_wp) return
            log_tb = log10(max(sfh%tb, 10.0_wp**ctx%tiny_logt_val))

            if (log_tb <= ctx%state%time_full(1)) then
                weights(1) = 1.0_wp
                return
            end if

            j = find_interval(ctx%state%time_full, log_tb)
            j = min(max(j, 1), nt - 1)

            burst_dt = get_time_interval(ctx, ctx%state%time_full(j), ctx%state%time_full(j+1))
            
            weights(j)   = get_time_interval(ctx, log_tb, ctx%state%time_full(j+1)) / burst_dt
            weights(j+1) = get_time_interval(ctx, ctx%state%time_full(j), log_tb) / burst_dt
            return
        end if

        ! --- CASE: CONTINUOUS SFH (Analytic or Tabular) ---
        
        ! 1. Determine Valid Time Range for SFH
        limit_min_lin = 10.0_wp**ctx%tiny_logt_val
        if (sfh%use_simha_limits == 1) then
            t_start_lin = sfh%t0; t_end_lin = sfh%tq
        else
            t_start_lin = sfh%tq; t_end_lin = sfh%tage
        end if
        log_t_start = log10(max(t_start_lin, limit_min_lin))
        log_t_end   = log10(max(t_end_lin,   limit_min_lin))

        ! 2. Identify Active Segment Loop Bounds
        j_start = find_interval(ctx%state%time_full, log_t_start)
        j_end   = find_interval(ctx%state%time_full, log_t_end) + 1
        
        j_start = max(j_start, max(idx_min, 1))
        j_end   = min(j_end, min(idx_max, nt - 1))

        ! 3. Initialize Loop State (Carry-Over Optimization)
        is_tabular = (sfh%type == 2 .or. sfh%type == 3)
        
        ! The SSPs are interpolated linearly (in log or linear age) between the grid ages, so the
        ! weight of node j is the integral of the SFR times the hat kernel centred on time_full(j).
        ! The integration limits are clamped to the star-forming window [t_start, t_end], but the
        ! kernel's nodes are always the grid ages themselves: a window edge (tage, sf_start, sf_trunc,
        ! ...) inside a bin shortens the integral, not the bin.

        ! Pre-calculate the lower integration limit and its moments for the first segment
        limits(1) = min(max(ctx%state%time_full(j_start), log_t_start), log_t_end)
        lim_prev = limits(1)

        if (.not. is_tabular) then
            if (ctx%interpolation_type_val == 0) then
                call eval_indefinite_moments_log(limits(1), sfh, m0_prev, m1_prev)
            else
                call eval_indefinite_moments_lin(10.0_wp**limits(1), sfh, m0_prev, m1_prev)
            end if
        end if

        ! 4. Iterate over SEGMENTS [j, j+1]
        do j = j_start, j_end

            limits(2) = min(max(ctx%state%time_full(j+1), log_t_start), log_t_end)

            ! Skip segments outside the star-forming window (the moments are unchanged)
            if (limits(2) <= lim_prev) then
                lim_prev = limits(2)
                cycle
            end if

            ! Compute Moments M0 and M1 over the clamped limits
            if (is_tabular) then
                ! Tabular must integrate the specific interval every time
                limits(1) = lim_prev
                call compute_tabular_moments(ctx, limits, m0, m1)
            else
                ! Analytic: Use the Cached Carry-Over
                if (ctx%interpolation_type_val == 0) then
                    call eval_indefinite_moments_log(limits(2), sfh, m0_curr, m1_curr)
                else
                    call eval_indefinite_moments_lin(10.0_wp**limits(2), sfh, m0_curr, m1_curr)
                end if

                m0 = m0_curr - m0_prev
                m1 = m1_curr - m1_prev

                ! Update cache for next iteration
                m0_prev = m0_curr
                m1_prev = m1_curr
            end if

            ! Hat-kernel nodes: the grid ages of this bin
            if (ctx%interpolation_type_val == 0) then
                node_lo = ctx%state%time_full(j)
                node_hi = ctx%state%time_full(j+1)
            else
                node_lo = 10.0_wp**ctx%state%time_full(j)
                node_hi = 10.0_wp**ctx%state%time_full(j+1)
            end if
            inv_dt = 1.0_wp / (node_hi - node_lo)

            ! Distribute to Nodes
            weights(j)   = weights(j)   + (node_hi * m0 - m1) * inv_dt
            weights(j+1) = weights(j+1) + (m1 - node_lo * m0) * inv_dt

            lim_prev = limits(2)

        end do

        ! Handle Special Case: Index 0 (Youngest Edge)
        if (idx_min == 0) then
            limits(1) = clamp_integration_limits(ctx, ctx%tiny_logt_val, sfh)
            limits(2) = clamp_integration_limits(ctx, ctx%state%time_full(1), sfh)
            
            if (limits(1) < limits(2)) then
                ! We define the "0th" bin as the interval from tiny_logt to time_full(1).
                ! Since we don't have an SSP at tiny_logt, we assign ALL the mass 
                ! formed in this interval to the first available SSP (index 1).
                ! This preserves total mass conservation.
                
                call compute_segment_moments(ctx, limits, sfh, m0, m1)
                
                ! Add the total integrated SFR (M0) of this gap to the first node.
                weights(1) = weights(1) + m0
            end if
        end if

    end subroutine compute_ssp_weights

    !> @brief Computes mass fraction and normalized SFR at a specific cosmic time.
    !>
    !> @details
    !> Calculates two key physical properties of the stellar population at `age_gyr`:
    !> 1. **Mass Fraction:** The fraction of the total stellar mass (that will ever form 
    !>    by the limit of the isochrones) that has already formed by `age_gyr`.
    !> 2. **Normalized SFR:** The SFR at `age_gyr`, normalized such that the total 
    !>    integrated SFR from $t=0$ to $t=T_{max}$ is 1.0 $M_{\odot}$.
    !>
    !> Handles all supported SFH types:
    !> * **0:** SSP (Instantaneous burst).
    !> * **1:** Exponential Tau Model + Constant + Burst.
    !> * **4:** Delayed Tau Model + Constant + Burst.
    !> * **5:** Linear Cutoff (Simha) + Tau Model.
    !> * **2/3:** Tabular SFH (interpolated).
    !>
    !> @param[in]  ctx         FSPS context (contains time grids and tabular data).
    !> @param[in]  pset        User parameters (SFH settings).
    !> @param[in]  age_gyr     Age of the universe at this step (Gyr).
    !> @param[out] mass_frac   Fraction of formed mass relative to total possible mass.
    !> @param[out] sfr_norm    Normalized Star Formation Rate (1/yr).
    !> @param[out] frac_linear Fraction of mass formed in the linear phase (Type 5 only).
    pure subroutine get_sfh_properties_at_age(ctx, pset, age_gyr, mass_frac, sfr_norm, frac_linear)
        type(fsps_context_t), intent(in) :: ctx
        type(params), intent(in) :: pset
        real(WP), intent(in) :: age_gyr
        real(WP), intent(out) :: mass_frac, sfr_norm, frac_linear

        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif

        ! Local variables
        real(WP) :: t_max_gyr, t_prime_gyr, t_trunc_gyr, t_zero_sfr_gyr
        real(WP) :: t_hi_gyr
        real(WP) :: tau_gyr, time_ratio, exp_term
        real(WP) :: mass_tau, total_mass_tau, sfr_tau
        real(WP) :: mass_const, sfr_const
        real(WP) :: mfrac_burst
        real(WP) :: mass_linear, total_mass_linear, sfr_trunc
        real(WP) :: slope_m
        real(WP) :: norm_factor
        integer :: power_law_idx
        integer :: tab_idx

        ! Defaults
        mass_frac = 1.0_wp
        sfr_norm = 0.0_wp
        frac_linear = 0.0_wp

        ! --------------------------------------------------------------------
        ! TYPE 0: SSP (Simple Stellar Population)
        ! --------------------------------------------------------------------
        if (pset%sfh == 0) then
            ! All mass formed at t=0. SFR is formally infinite (delta function),
            ! but effectively zero for any t > 0 calculation here.
            return
        end if

        ! --------------------------------------------------------------------
        ! ANALYTIC SFHs (1=Tau, 4=Delayed, 5=Simha)
        ! --------------------------------------------------------------------
        if (any([1, 4, 5] == pset%sfh)) then
            
            ! 1. Time Definitions
            ! Tmax: Age of oldest isochrone in the library (Gyr)
            t_max_gyr = (10.0_wp**(ctx%state%time_full(ctx%state%ntfull) - LOG_GYR_TO_YR)) - pset%sf_start
            
            ! Tprime: Current age relative to start of SF (Gyr)
            t_prime_gyr = age_gyr - pset%sf_start
            
            ! Ttrunc: Truncation time (Gyr)
            if (pset%sf_trunc < SAFE_FLOOR .or. pset%sf_trunc < pset%sf_start) then
                t_trunc_gyr = t_max_gyr
            else
                t_trunc_gyr = pset%sf_trunc - pset%sf_start
            end if

            tau_gyr = max(pset%tau, SAFE_FLOOR)

            ! 2. Calculate Tau Component (Exponential / Delayed)
            ! ------------------------------------------------------------
            ! Power: 1 for Exp, 2 for Delayed (matches Gamma function arguments)
            power_law_idx = merge(1, 2, pset%sfh == 1)

            ! Total Mass (integrated to Tmax or Truncation)
            time_ratio = min(t_max_gyr, t_trunc_gyr) / tau_gyr
            total_mass_tau = tau_gyr * gammainc(power_law_idx, time_ratio)

            ! Current Mass & SFR
            if (t_prime_gyr < 0.0_wp) then
                mass_tau = 0.0_wp
                sfr_tau  = 0.0_wp
            else
                time_ratio = min(t_prime_gyr, t_trunc_gyr) / tau_gyr
                mass_tau = tau_gyr * gammainc(power_law_idx, time_ratio)
                
                ! SFR ~ (t/tau)^(p-1) * exp(-t/tau)
                exp_term = exp(-time_ratio)
                sfr_tau  = (time_ratio**(power_law_idx - 1)) * exp_term
            end if

            ! 3. Add Constant & Burst Components (Types 1 & 4 only)
            ! ------------------------------------------------------------
            if (pset%sfh == 1 .or. pset%sfh == 4) then
                
                ! -- Burst Component --
                ! Check if current time is past the burst time
                if (t_prime_gyr > (pset%tburst - pset%sf_start)) then
                    mfrac_burst = 1.0_wp
                else
                    mfrac_burst = 0.0_wp
                end if
                
                ! -- Constant Component --
                if (t_prime_gyr > 0.0_wp) then
                    sfr_const = 1.0_wp / min(t_max_gyr, t_trunc_gyr)
                    mass_const = max(min(t_prime_gyr, t_trunc_gyr), 0.0_wp) * sfr_const
                else
                    sfr_const = 0.0_wp
                    mass_const = 0.0_wp
                end if

                ! Combine components with weights
                ! Mass Fraction = Weighted sum of formed masses / Total masses
                norm_factor = (1.0_wp - pset%const - pset%fburst)
                
                mass_frac = (norm_factor * mass_tau / total_mass_tau) + &
                            (pset%const  * mass_const) + &
                            (pset%fburst * mfrac_burst)

                ! SFR Calculation
                if (t_prime_gyr > t_trunc_gyr) then
                    sfr_norm = 0.0_wp
                else
                    sfr_norm = (norm_factor * sfr_tau / total_mass_tau) + &
                               (pset%const * sfr_const)
                end if

            ! 4. Add Linear Component (Type 5 - Simha)
            ! ------------------------------------------------------------
            else if (pset%sfh == 5) then
                slope_m = -pset%sf_slope
                
                ! Find zero-crossing time for SFR
                if (slope_m > 0.0_wp) then
                    t_zero_sfr_gyr = t_trunc_gyr + (1.0_wp / slope_m)
                else
                    t_zero_sfr_gyr = t_max_gyr
                end if

                if (t_trunc_gyr <= 0.0_wp .or. t_trunc_gyr > t_max_gyr) then
                    ! No truncation effective
                    total_mass_linear = 0.0_wp
                    mass_linear = 0.0_wp
                    sfr_norm = sfr_tau / total_mass_tau
                    frac_linear = 0.0_wp
                else
                    ! Calculate Linear Mass Integral
                    t_hi_gyr = min(t_zero_sfr_gyr, t_max_gyr)
                    sfr_trunc = (t_trunc_gyr / pset%tau) * exp(-t_trunc_gyr / pset%tau)
                    
                    ! Integral of linear function: A*(dt) + 0.5*B*(dt^2)
                    ! Legacy formula is slightly algebraic rearranged:
                    total_mass_linear = sfr_trunc * ((t_hi_gyr - t_trunc_gyr) - &
                                        (slope_m / 2.0_wp) * (t_hi_gyr**2 + t_trunc_gyr**2) + &
                                        (slope_m * t_hi_gyr * t_trunc_gyr))

                    if (t_prime_gyr <= t_trunc_gyr) then
                        ! Before truncation starts
                        mass_linear = 0.0_wp
                        sfr_norm = sfr_tau / (total_mass_tau + total_mass_linear)
                    else
                        ! After truncation starts
                        t_hi_gyr = min(t_zero_sfr_gyr, t_prime_gyr)
                        
                        mass_linear = sfr_trunc * ((t_hi_gyr - t_trunc_gyr) - &
                                      (slope_m / 2.0_wp) * (t_hi_gyr**2 + t_trunc_gyr**2) + &
                                      (slope_m * t_hi_gyr * t_trunc_gyr))
                                      
                        sfr_norm = max(sfr_trunc * (1.0_wp - slope_m * (t_prime_gyr - t_trunc_gyr)), 0.0_wp) &
                                   / (total_mass_tau + total_mass_linear)
                    end if
                end if

                ! Combine Tau + Linear
                if (t_prime_gyr >= 0.0_wp) then
                    mass_frac = (mass_tau + mass_linear) / (total_mass_tau + total_mass_linear)
                    if ((mass_linear + mass_tau) > SAFE_FLOOR) then
                        frac_linear = mass_linear / (mass_linear + mass_tau)
                    end if
                else
                    mass_frac = 0.0_wp
                    frac_linear = 0.0_wp
                    sfr_norm = 0.0_wp
                end if

            end if

        ! --------------------------------------------------------------------
        ! TABULAR SFHs (2=File, 3=Array)
        ! --------------------------------------------------------------------
        else if (pset%sfh == 2 .or. pset%sfh == 3) then
            
            ! Find index in table such that tab(1, idx) <= age_gyr
            ! The table is stored in ctx%state%sfh_tab(1:3, :) -> 1=Time(Yr), 2=SFR
            ! Note: Table time is in YEARS, age_gyr is in GYR.
            
            tab_idx = find_interval_tab_time(ctx, ctx%state%ntabsfh, age_gyr * 1.0e9_wp)
            
            ! Clamp index to valid interpolation range [1, n-1]
            tab_idx = max(min(tab_idx, ctx%state%ntabsfh - 1), 1)

            ! Linear Interpolation of SFR
            slope_m = (ctx%state%sfh_tab(2, tab_idx+1) - ctx%state%sfh_tab(2, tab_idx)) / &
                      (ctx%state%sfh_tab(1, tab_idx+1) - ctx%state%sfh_tab(1, tab_idx))
            
            sfr_norm = ctx%state%sfh_tab(2, tab_idx) + &
                       slope_m * ((age_gyr * 1.0e9_wp) - ctx%state%sfh_tab(1, tab_idx))
            
            sfr_norm = max(sfr_norm, 0.0_wp)
        end if
        
        if (pset%sfh /= 2 .and. pset%sfh /= 3) then
            sfr_norm = sfr_norm / 1.0e9_wp
        end if

    end subroutine get_sfh_properties_at_age

    !> 

    !> @brief Computes derived SFH statistics (Mean Age, recent sSFRs).
    !>
    !> @details
    !> Calculates useful diagnostics for the stellar population:
    !> 1. **Mean Age:** Mass-weighted stellar age (Gyr).
    !> 2. **Recent sSFR:** Log10(Specific SFR) averaged over three recent lookback 
    !>    windows: 1 Myr, 10 Myr, and 100 Myr.
    !>
    !> **Implementation Pathways:**
    !> * **Analytic:** For Types 1 (Tau) and 4 (Delayed), uses exact analytic formulas 
    !>   involving exponentials and gammas.
    !> * **Numerical:** For Types 2 & 3 (Tabular), uses robust trapezoidal integration 
    !>   (`fsps_integration`) over the lookup table.
    !>
    !> @param[inout] ctx          FSPS context (required for Tabular SFH data).
    !> @param[in]    pset         User parameters.
    !> @param[in]    model        Output model structure (contains current age).
    !> @param[out]   ssfr_log_out Array of log10(sSFR) for [1Myr, 10Myr, 100Myr]. Units: log(yr^-1).
    !> @param[out]   mean_age     Mass-weighted average age (Gyr).
    subroutine compute_sfh_statistics(ctx, pset, model, ssfr_log_out, mean_age)
        type(fsps_context_t), intent(inout) :: ctx
        type(params), intent(in) :: pset
        type(compspout), intent(in) :: model
        real(WP), dimension(3), intent(out) :: ssfr_log_out
        real(WP), intent(out) :: mean_age

        ! Local vars for Analytic Path
        real(WP) :: age_universe_gyr, dt_sfr
        real(WP) :: tau
        real(WP) :: t_term
        real(WP) :: mass_formed_tau, numerator_age
        real(WP) :: ssfr_recent(3)
        real(WP) :: lookback_windows(3)
        real(WP) :: term1, term2, t1, t2
        integer :: i

        ! Local vars for Numerical Path
        real(WP) :: age_current_yr
        integer :: n_tab, idx_cut
        real(WP) :: slope_last, sfr_at_age
        real(WP) :: total_mass, mass_in_window
        real(WP) :: t_start_win_yr
        integer :: idx_win_start

        ! Constants
        ! Windows: 1 Myr, 10 Myr, 100 Myr (in Gyr)
        lookback_windows = [1.0e-3_wp, 1.0e-2_wp, 1.0e-1_wp]

        ! Current age of the universe in this model step (Gyr)
        ! model%age is log10(years).
        age_universe_gyr = (10.0_wp**model%age) / 1.0e9_wp
        dt_sfr = age_universe_gyr - pset%sf_start

        if (dt_sfr < 0.0_wp) then
            ssfr_log_out = -100.0_wp 
            mean_age = 0.0_wp
            return
        end if

        ! --------------------------------------------------------------------
        ! PATH 1: ANALYTIC (Type 1 & 4)
        ! --------------------------------------------------------------------
        if (pset%sfh == 1 .or. pset%sfh == 4) then
            
            tau = max(pset%tau, SAFE_FLOOR)
            t_term = dt_sfr / tau

            ! --- Mean Age (Mass-Weighted) ---
            if (pset%sfh == 1) then
                ! Exponential: exp(-t/tau)
                ! Derived mass-weighted age (relative to dt_sfr)
                numerator_age = tau * (1.0_wp - exp(-t_term) * (t_term + 1.0_wp))
                mass_formed_tau = 1.0_wp - exp(-t_term)
                
                mean_age = numerator_age / mass_formed_tau

                ! Recent Mass Formed (Unnormalized)
                do i = 1, 3
                    ssfr_recent(i) = exp(-(dt_sfr - lookback_windows(i))/tau) - exp(-t_term)
                    ssfr_recent(i) = ssfr_recent(i) / mass_formed_tau
                end do

            else ! Type 4
                ! Delayed: t * exp(-t/tau)
                numerator_age = (2.0_wp - exp(-t_term) * (t_term * (t_term + 2.0_wp) + 2.0_wp)) * tau
                mass_formed_tau = 1.0_wp - exp(-t_term) * (t_term + 1.0_wp)

                mean_age = numerator_age / mass_formed_tau

                do i = 1, 3
                   t2 = dt_sfr
                   t1 = max(dt_sfr - lookback_windows(i), 0.0_wp)
                   
                   term1 = exp(-(dt_sfr - lookback_windows(i))/tau) * ((dt_sfr - lookback_windows(i))/tau)
                   term2 = exp(-t_term) * t_term
                   
                   ssfr_recent(i) = (term1 - term2) / mass_formed_tau
                end do
            end if

            ! --- Add Constant Component ---
            ! Age of constant SFR part is just dt / 2
            mean_age = mean_age * (1.0_wp - pset%const) + &
                       (pset%const * dt_sfr / 2.0_wp)

            do i = 1, 3
                ssfr_recent(i) = ssfr_recent(i) * (1.0_wp - pset%const) + &
                                 (pset%const * lookback_windows(i) / dt_sfr)
            end do

            ! --- Convert Age to Lookback Time ---
            mean_age = dt_sfr - mean_age

            ! --- Add Burst ---
            if (age_universe_gyr > pset%tburst) then
                mean_age = (1.0_wp - pset%fburst) * mean_age + &
                           (pset%fburst * (age_universe_gyr - pset%tburst))
                
                do i = 1, 3
                    if ((dt_sfr - pset%tburst) <= lookback_windows(i)) then
                        ssfr_recent(i) = ssfr_recent(i) + pset%fburst
                    end if
                end do
            end if

            ! --- Final Normalization ---
            do i = 1, 3
                ssfr_log_out(i) = log10(max(ssfr_recent(i) / max(model%mass_csp, SAFE_FLOOR) / &
                                            (lookback_windows(i) * 1.0e9_wp), SAFE_FLOOR))
            end do

        ! --------------------------------------------------------------------
        ! PATH 2: NUMERICAL (Tabular / Types 2 & 3)
        ! --------------------------------------------------------------------
        else if (pset%sfh == 2 .or. pset%sfh == 3) then
            
            age_current_yr = age_universe_gyr * 1.0e9_wp
            n_tab = ctx%state%ntabsfh

            ! 1. Extract Valid History
            ! We need to integrate from t=0 to t=age_current_yr.
            ! The table might go beyond age_current_yr.
            
            ! Find index just below current age
            idx_cut = find_interval_tab_time(ctx, n_tab, age_current_yr)
            idx_cut = min(max(idx_cut, 1), n_tab - 1)

            ! Interpolate SFR at exactly age_current_yr
            slope_last = (ctx%state%sfh_tab(2, idx_cut+1) - ctx%state%sfh_tab(2, idx_cut)) / &
                         (ctx%state%sfh_tab(1, idx_cut+1) - ctx%state%sfh_tab(1, idx_cut))
            sfr_at_age = ctx%state%sfh_tab(2, idx_cut) + &
                         slope_last * (age_current_yr - ctx%state%sfh_tab(1, idx_cut))
            sfr_at_age = max(sfr_at_age, 0.0_wp)

            ! Fill arrays (converting strided table access to contiguous temp)
            do i = 1, idx_cut
                ctx%state%sfh_t_calc(i)   = ctx%state%sfh_tab(1, i)
                ctx%state%sfh_sfr_calc(i) = ctx%state%sfh_tab(2, i)
            end do
            ! Add the exact endpoint
            ctx%state%sfh_t_calc(idx_cut + 1)   = age_current_yr
            ctx%state%sfh_sfr_calc(idx_cut + 1) = sfr_at_age

            ! 2. Integrate Total Mass
            total_mass = integrate_trapezoid_array(ctx%state%sfh_t_calc(1:idx_cut+1), ctx%state%sfh_sfr_calc(1:idx_cut+1))
            
            if (total_mass <= SAFE_FLOOR) then
                mean_age = 0.0_wp
                ssfr_log_out = -100.0_wp
                return
            end if

            ! 3. Compute Mean Age
            ! Formula: Integral( (Age - t) * SFR(t) dt ) / TotalMass
            
            ! Vectorized calculation of integrand
            do i = 1, idx_cut + 1
                ctx%state%sfh_age_integrand(i) = (age_current_yr - ctx%state%sfh_t_calc(i)) * ctx%state%sfh_sfr_calc(i)
            end do
            
            mean_age = integrate_trapezoid_array(ctx%state%sfh_t_calc(1:idx_cut+1), ctx%state%sfh_age_integrand(1:idx_cut+1)) / &
                       total_mass
            mean_age = mean_age / 1.0e9_wp ! Convert yr -> Gyr

            ! 4. Compute sSFRs over windows
             do i = 1, 3
                 t_start_win_yr = age_current_yr - (lookback_windows(i) * 1.0e9_wp)
                 if (t_start_win_yr < 0.0_wp) t_start_win_yr = 0.0_wp

                 ! Find index in our local array
                 idx_win_start = find_interval(ctx%state%sfh_t_calc(1:idx_cut+1), t_start_win_yr)
                 idx_win_start = max(1, idx_win_start)

                 mass_in_window = integrate_trapezoid_array(ctx%state%sfh_t_calc(idx_win_start:idx_cut+1), &
                                                            ctx%state%sfh_sfr_calc(idx_win_start:idx_cut+1))

                 ! Normalization: sSFR = (Mass_Window / Total_Mass) / Window_Size
                 ssfr_log_out(i) = log10(max(mass_in_window / max(model%mass_csp, SAFE_FLOOR) / &
                                             (lookback_windows(i) * 1.0e9_wp), SAFE_FLOOR))
             end do
            
        else
            ! Unsupported type
            ssfr_log_out = -99.0_wp
            mean_age = 0.0_wp
        end if

    end subroutine compute_sfh_statistics

    ! ========================================================================
    ! PRIVATE MATH KERNELS (MOMENTS)
    ! ========================================================================

    !> @brief Calculates the 0th and 1st moments of the SFH over a specific time segment.
    !>
    !> @details
    !> Computes the definite integrals required for the Moment Method optimization:
    !> * $M_0 = \int_{t_1}^{t_2} \mathrm{SFR}(t) \, dt$
    !> * $M_1 = \int_{t_1}^{t_2} \mathrm{SFR}(t) \cdot x(t) \, dt$
    !>
    !> The variable $x(t)$ represents the interpolation coordinate:
    !> * If `ctx%interpolation_type_val == 0`: $x(t) = \log(t)$
    !> * If `ctx%interpolation_type_val == 1`: $x(t) = t$
    !>
    !> Dispatches to the appropriate indefinite integral kernel based on the context.
    !>
    !> @param[in]  ctx    FSPS context (defines interpolation type).
    !> @param[in]  limits Integration limits [Start, End] in log10(lookback years).
    !> @param[in]  sfh    SFH parameters.
    !> @param[out] m0     The 0th moment (Total SFR).
    !> @param[out] m1     The 1st moment (SFR weighted by time coordinate).
    pure subroutine compute_segment_moments(ctx, limits, sfh, m0, m1)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(2), intent(in) :: limits
        type(sfhparams), intent(in) :: sfh
        real(WP), intent(out) :: m0, m1

        real(WP) :: m0_up, m1_up, m0_lo, m1_lo
        real(WP) :: t_lin_up, t_lin_lo

        if (sfh%type == 2 .or. sfh%type == 3) then
            call compute_tabular_moments(ctx, limits, m0, m1)
            return
        end if

        if (ctx%interpolation_type_val == 0) then
            call eval_indefinite_moments_log(limits(2), sfh, m0_up, m1_up)
            call eval_indefinite_moments_log(limits(1), sfh, m0_lo, m1_lo)
        else
            t_lin_up = 10.0_wp**limits(2)
            t_lin_lo = 10.0_wp**limits(1)
            call eval_indefinite_moments_lin(t_lin_up, sfh, m0_up, m1_up)
            call eval_indefinite_moments_lin(t_lin_lo, sfh, m0_lo, m1_lo)
        end if

        m0 = m0_up - m0_lo
        m1 = m1_up - m1_lo
    end subroutine compute_segment_moments

    !> @brief Numerically integrates tabular SFH moments over a specific time segment.
    !>
    !> @details
    !> Performs trapezoidal integration of the tabular SFR data between the limits
    !> [t_start, t_end]. Handles the partial bins at the edges of the segment.
    !>
    !> @param[in]  ctx    FSPS context (contains sfh_tab).
    !> @param[in]  limits Integration limits [Start, End] in log10(lookback years).
    !> @param[out] m0     Integral of SFR(t) dt.
    !> @param[out] m1     Integral of SFR(t) * x(t) dt.
    pure subroutine compute_tabular_moments(ctx, limits, m0, m1)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), dimension(2), intent(in) :: limits
        real(WP), intent(out) :: m0, m1

        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif

        integer :: idx_start, idx_end, i
        real(WP) :: t_start_yr, t_end_yr
        real(WP) :: t_cur, t_next, dt_step
        real(WP) :: sfr_cur, sfr_next
        real(WP) :: x_cur, x_next
        real(WP) :: term0, term1
        real(WP) :: slope

        ! 1. Convert limits to linear years (Table is in Years)
        t_start_yr = 10.0_wp**limits(1)
        t_end_yr   = 10.0_wp**limits(2)

        ! 2. Find relevant table indices
        ! find_interval returns i such that tab(i) <= val < tab(i+1)
        idx_start = find_interval_tab_time(ctx, ctx%state%ntabsfh, t_start_yr)
        idx_end   = find_interval_tab_time(ctx, ctx%state%ntabsfh, t_end_yr)

        ! Clamp indices
        idx_start = max(1, min(idx_start, ctx%state%ntabsfh - 1))
        idx_end   = max(1, min(idx_end, ctx%state%ntabsfh - 1))

        m0 = 0.0_wp
        m1 = 0.0_wp

        ! 3. Integrate loop
        ! We iterate through the table bins that overlap with [t_start_yr, t_end_yr]
        do i = idx_start, idx_end
            
            ! Determine integration bounds for this specific table bin
            t_cur  = ctx%state%sfh_tab(1, i)
            t_next = ctx%state%sfh_tab(1, i+1)

            ! Clip to the requested segment limits
            t_cur  = max(t_cur, t_start_yr)
            t_next = min(t_next, t_end_yr)

            if (t_next <= t_cur) cycle

            ! Interpolate SFR at the clipped bounds
            ! Linear interpolation: SFR(t) = SFR_i + slope * (t - t_i)
            slope = (ctx%state%sfh_tab(2, i+1) - ctx%state%sfh_tab(2, i)) / &
                    (ctx%state%sfh_tab(1, i+1) - ctx%state%sfh_tab(1, i))

            sfr_cur  = ctx%state%sfh_tab(2, i) + slope * (t_cur - ctx%state%sfh_tab(1, i))
            sfr_next = ctx%state%sfh_tab(2, i) + slope * (t_next - ctx%state%sfh_tab(1, i))

            ! Determine coordinate X(t) values
            if (ctx%interpolation_type_val == 0) then
                ! Logarithmic: x = log10(t)
                x_cur  = log10(t_cur)
                x_next = log10(t_next)
            else
                ! Linear: x = t
                x_cur  = t_cur
                x_next = t_next
            end if

            ! Trapezoidal Integration
            dt_step = t_next - t_cur
            
            ! M0 = Integral(SFR) ~ 0.5 * (SFR1 + SFR2) * dt
            term0 = 0.5_wp * (sfr_cur + sfr_next) * dt_step
            m0 = m0 + term0

            ! M1 = Integral(SFR * x) ~ 0.5 * (SFR1*x1 + SFR2*x2) * dt
            term1 = 0.5_wp * (sfr_cur * x_cur + sfr_next * x_next) * dt_step
            m1 = m1 + term1
        end do

    end subroutine compute_tabular_moments

    !> @brief Evaluates indefinite moments assuming Logarithmic Time interpolation.
    !>
    !> @details
    !> Calculates analytic indefinite integrals:
    !> * $M_0 = \int \mathrm{SFR}(t) \, dt$
    !> * $M_1 = \int \mathrm{SFR}(t) \cdot \log(t) \, dt$
    !>
    !> Supports Constant, Exponential, Delayed Exponential, and Simha (Linear) SFHs.
    !>
    !> @param[in]  log_t  Integration variable $t$ (log10 years).
    !> @param[in]  sfh    SFH parameters.
    !> @param[out] m0     The evaluated 0th moment indefinite integral.
    !> @param[out] m1     The evaluated 1st moment indefinite integral.
    pure subroutine eval_indefinite_moments_log(log_t, sfh, m0, m1)
        type(sfhparams), intent(in) :: sfh
        real(WP), intent(in) :: log_t
        real(WP), intent(out) :: m0, m1

        real(WP) :: t_lin, t_tau, ei_val
        real(WP) :: term_const
        
        ! Type 5 specific vars (must be declared at top)
        real(WP) :: t_prime, slope_fac

        t_lin = 10.0_wp**log_t

        select case (sfh%type)
        case (0) ! Constant
            m0 = t_lin
            m1 = t_lin * (log_t - LOG_E)
        case (1) ! Exp
            if (sfh%tau <= SAFE_FLOOR) then; m0=0._wp; m1=0._wp; return; endif
            t_tau = max(t_lin / sfh%tau, SAFE_FLOOR)
            ei_val = expi(t_tau)
            if (.not. ieee_is_finite(ei_val)) ei_val = 0.0_wp
            
            m0 = exp(t_tau)
            m1 = log_t * m0 - LOG_E * ei_val
            
        case (4) ! Delayed
            if (sfh%tau <= SAFE_FLOOR) then; m0=0._wp; m1=0._wp; return; endif
            t_tau = max(t_lin / sfh%tau, SAFE_FLOOR)
            ei_val = expi(t_tau)
            if (.not. ieee_is_finite(ei_val)) ei_val = 0.0_wp
            
            m0 = -(t_lin - sfh%tage - sfh%tau) * exp(t_tau)
            
            ! M1 = int SFR(t) log10(t) dt with SFR = dm0/dt = (tage - t) e^(t/tau) / tau.
            ! Differentiating confirms dM1/dt = SFR * log10(t) only with this overall sign; the
            ! previous expression had the opposite sign, producing negative hat-function
            ! weights (and 45-100x wrong CSPs) for sfh = 4 and 5.
            term_const = sfh%tau * LOG_E
            m1 = -((t_lin - sfh%tage - sfh%tau) * log_t - term_const) * exp(t_tau) - &
                 (sfh%tage + sfh%tau) * LOG_E * ei_val

        case (5) ! Simha
            t_prime = max(0.0_wp, sfh%tage - sfh%sf_trunc)
            slope_fac = 1.0_wp - (sfh%sf_slope * t_prime)
            
            m0 = slope_fac * t_lin + (sfh%sf_slope * t_lin**2 / 2.0_wp)
            
            m1 = slope_fac * t_lin * (log_t - LOG_E) + &
                 (sfh%sf_slope * t_lin**2 / 2.0_wp) * (log_t - 0.5_wp * LOG_E)
        case default
            m0 = 0.0_wp; m1 = 0.0_wp
        end select
    end subroutine eval_indefinite_moments_log

    !> @brief Evaluates indefinite moments assuming Linear Time interpolation.
    !>
    !> @details
    !> Calculates analytic indefinite integrals:
    !> * $M_0 = \int \mathrm{SFR}(t) \, dt$
    !> * $M_1 = \int \mathrm{SFR}(t) \cdot t \, dt$
    !>
    !> Supports Constant, Exponential, Delayed Exponential, and Simha (Linear) SFHs.
    !>
    !> @param[in]  t      Integration variable $t$ (linear years).
    !> @param[in]  sfh    SFH parameters.
    !> @param[out] m0     The evaluated 0th moment indefinite integral.
    !> @param[out] m1     The evaluated 1st moment indefinite integral.
    pure subroutine eval_indefinite_moments_lin(t, sfh, m0, m1)
        type(sfhparams), intent(in) :: sfh
        real(WP), intent(in) :: t
        real(WP), intent(out) :: m0, m1

        real(WP) :: t_tau
        ! Type 5 specific vars (must be declared at top)
        real(WP) :: trunc_off, slope_f
        
        select case (sfh%type)
        case (0)
            m0 = t
            m1 = 0.5_wp * t**2
        case (1)
            if (sfh%tau <= SAFE_FLOOR) then; m0=0._wp; m1=0._wp; return; endif
            t_tau = t / sfh%tau
            m0 = exp(t_tau)
            m1 = (t - sfh%tau) * exp(t_tau)
        case (4)
            if (sfh%tau <= SAFE_FLOOR) then; m0=0._wp; m1=0._wp; return; endif
            t_tau = t / sfh%tau
            m0 = (sfh%tage - t + sfh%tau) * exp(t_tau)
            ! M1 = int SFR(t) t dt with SFR = dm0/dt = (tage - t) e^(t/tau) / tau, using
            ! int t e^(t/tau) dt = tau (t - tau) e^(t/tau) and
            ! int t^2 e^(t/tau) dt = tau (t^2 - 2 tau t + 2 tau^2) e^(t/tau).
            ! (The previous expression had the opposite overall sign.)
            m1 = ( sfh%tage*(t - sfh%tau) - (t**2 - 2.0_wp*t*sfh%tau + 2.0_wp*sfh%tau**2) ) * exp(t_tau)
            
        case (5)
            trunc_off = max(0.0_wp, sfh%tage - sfh%sf_trunc)
            slope_f = 1.0_wp - (sfh%sf_slope * trunc_off)
            
            m0 = slope_f * t + (sfh%sf_slope * 0.5_wp * t**2)
            m1 = slope_f * 0.5_wp * t**2 + (sfh%sf_slope * t**3 / 3.0_wp)
        case default
            m0 = 0.0_wp; m1 = 0.0_wp
        end select
    end subroutine eval_indefinite_moments_lin

    ! ========================================================================
    ! PRIVATE UTILITIES
    ! ========================================================================

    !> @brief Clips a proposed log-time integration limit to the valid SFH range.
    !>
    !> @details
    !> Ensures that integration limits do not exceed the physical start or end times 
    !> of the star formation history. 
    !> * **Simha SFH:** Integration bounds are `[sfh%t0, sfh%tq]`.
    !> * **Standard SFH:** Integration bounds are `[sfh%tq, sfh%tage]`.
    !>
    !> @note Marked `elemental` to allow efficient clamping of array segments.
    !>
    !> @param[in] ctx            The FSPS context (for minimum time safety floor).
    !> @param[in] log_t_limit    The proposed limit in log10(lookback years).
    !> @param[in] sfh            The SFH parameters structure.
    !> @return    clamped_log_t  The limit clipped to the valid range.
    elemental function clamp_integration_limits(ctx, log_t_limit, sfh) result(clamped_log_t)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: log_t_limit
        type(sfhparams), intent(in) :: sfh
        real(WP) :: clamped_log_t

        real(WP) :: t_start_linear, t_end_linear
        real(WP) :: log_t_start, log_t_end
        real(WP) :: limit_min_linear

        ! Calculate the minimum safe linear time to avoid log10(0)
        limit_min_linear = 10.0_wp**ctx%tiny_logt_val

        ! Determine the valid linear time window based on SFH type
        if (sfh%use_simha_limits == 1) then
            t_start_linear = sfh%t0
            t_end_linear   = sfh%tq
        else
            t_start_linear = sfh%tq
            t_end_linear   = sfh%tage
        end if

        ! Convert boundaries to log10, enforcing the safety floor
        ! Note: max() ensures we never take log10 of 0 or negative numbers
        log_t_start = log10(max(t_start_linear, limit_min_linear))
        log_t_end   = log10(max(t_end_linear,   limit_min_linear))

        ! Clamp the input `log_t_limit` to the valid window [log_t_start, log_t_end]
        clamped_log_t = min(max(log_t_limit, log_t_start), log_t_end)

    end function clamp_integration_limits

    !> @brief Calculates the time interval between two log-time points.
    !>
    !> @details
    !> Calculates the denominator ($\Delta x$) for weight averaging, respecting 
    !> the active interpolation scheme stored in `ctx`.
    !> * **Type 0 (Logarithmic):** Returns $\Delta \log t = \log t_2 - \log t_1$.
    !> * **Type 1 (Linear):** Returns $\Delta t = 10^{\log t_2} - 10^{\log t_1}$.
    !>
    !> @note Marked `elemental` to facilitate vectorization over time arrays.
    !>
    !> @param[in] ctx    The FSPS context containing interpolation settings.
    !> @param[in] logt1  Start time (log10 years).
    !> @param[in] logt2  End time (log10 years).
    !> @return    dt     The calculated time interval.
    elemental function get_time_interval(ctx, logt1, logt2) result(dt)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: logt1, logt2
        real(WP) :: dt

        ! 0 = Logarithmic interpolation
        if (ctx%interpolation_type_val == 0) then
            dt = logt2 - logt1
        else
            ! 1 = Linear interpolation (convert logt -> t)
            dt = (10.0_wp**logt2) - (10.0_wp**logt1)
        end if

    end function get_time_interval

    !> find_interval over the table times sfh_tab(1, 1:n), searched in place: the row is strided,
    !> and passing it to find_interval (contiguous dummy) makes a copy, a heap allocation per call
    !> in device code. Same result as find_interval(ctx%state%sfh_tab(1, 1:n), value).
    pure function find_interval_tab_time(ctx, n, value) result(idx)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        type(fsps_context_t), intent(in) :: ctx
        integer, intent(in)  :: n
        real(WP), intent(in) :: value
        integer :: idx, lower, upper, mid
        logical :: is_ascending

        if (n < 2) then
            idx = 1
            return
        end if
        is_ascending = (ctx%state%sfh_tab(1, n) >= ctx%state%sfh_tab(1, 1))
        lower = 0
        upper = n + 1
        do while (upper - lower > 1)
            mid = (upper + lower) / 2
            if (is_ascending .eqv. (value >= ctx%state%sfh_tab(1, mid))) then
                lower = mid
            else
                upper = mid
            end if
        end do
        if (value == ctx%state%sfh_tab(1, 1)) then
            idx = 1
        else if (value == ctx%state%sfh_tab(1, n)) then
            idx = n - 1
        else
            idx = lower
        end if
    end function find_interval_tab_time

    !> @brief compute_sfh_weights for one output, with caller-provided scratch columns (the batched
    !> CSP runs one per model on the device).
    subroutine sfh_weights_core(ctx, pset, tage, nzin, w, tmp1, tmp2)
        type(fsps_context_t), intent(inout) :: ctx
        type(params), intent(in)            :: pset
        real(WP), intent(in)                :: tage
        integer, intent(in)                 :: nzin
        real(WP), intent(inout), contiguous :: w(:,:)              ! (ages, nzin)
        real(WP), intent(inout), contiguous :: tmp1(:), tmp2(:)    ! scratch (ages)

        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif

        ! Local variables
        type(sfhparams) :: sfh
        real(WP) :: mass1, mass2
        real(WP) :: frac_linear, mass_frac, sfr, fburst_val
        real(WP) :: t1, t2, dt, zbin, dz
        integer :: i, k, imin, imax
        integer :: nt, j

        ! 1. Setup
        nt = ctx%state%ntfull
        do k = 1, nzin
            do i = 1, nt
                w(i, k) = 0.0_wp
            end do
        end do
        do i = 1, nt
            tmp1(i) = 0.0_wp
            tmp2(i) = 0.0_wp
        end do

        ! Initialize SFH struct with unit conversions (Gyr -> Yr)
        call convert_sfhparams(pset, tage, sfh)

        ! Optimization: Only calculate up to the age of the universe + buffer
        ! (Legacy logic: +2 steps to be safe against interpolation edge cases)
        imax = min(max(find_interval(ctx%state%time_full, log10(sfh%tage)) + 2, 1), nt)

        ! --------------------------------------------------------------------
        ! CASE: SSP (Type 0)
        ! --------------------------------------------------------------------
        if (pset%sfh == 0) then
            sfh%type = -1        ! -1 indicates Burst in compute_ssp_weights
            sfh%tb   = sfh%tage  ! Burst at 'now' (lookback time = tage)
            
            imin = max(imax - 2, 1) ! Optimization: SSP is local
            call compute_ssp_weights(ctx, sfh, imin, imax, w(:, 1))
            return
        end if

        ! --------------------------------------------------------------------
        ! CASE: Tau (1) or Delayed Tau (4)
        ! --------------------------------------------------------------------
        if (pset%sfh == 1 .or. pset%sfh == 4) then
            ! A. Main Exponential Component
            sfh%type = pset%sfh
            imin = 0
            call compute_ssp_weights(ctx, sfh, imin, imax, w(:, 1))

            ! Normalize to 1.0 mass
            mass1 = sum(w(1:imax, 1))
            if (mass1 < SAFE_FLOOR) mass1 = 1.0_wp
            w(:, 1) = w(:, 1) / mass1

            ! B. Add Constant and Burst (if requested)
            if (pset%const > 0.0_wp .or. pset%fburst > SAFE_FLOOR) then
                
                ! Constant Component
                sfh%type = 0 ! Constant
                call compute_ssp_weights(ctx, sfh, imin, imax, tmp1(:))
                mass1 = 0.0_wp
                do i = 1, imax
                    mass1 = mass1 + tmp1(i)
                end do
                if (mass1 < SAFE_FLOOR) mass1 = 1.0_wp

                ! Burst Component
                tmp2(:) = 0.0_wp
                fburst_val = 0.0_wp
                
                if (sfh%tb >= 0.0_wp) then
                    sfh%type = -1 ! Burst
                    call compute_ssp_weights(ctx, sfh, imin, imax, tmp2(:))
                    fburst_val = pset%fburst
                    
                    ! Extend imax to include burst if it happened earlier
                    imax = max(imax, min(max(find_interval(ctx%state%time_full, log10(sfh%tb)) + 2, 1), nt))
                end if

                ! Combine: (1 - C - B) * Tau + C * Const + B * Burst
                do i = 1, nt
                    w(i, 1) = (1.0_wp - pset%const - fburst_val) * w(i, 1) + &
                                           pset%const * (tmp1(i) / mass1) + &
                                           fburst_val * tmp2(i)
                end do
            end if
            return
        end if

        ! --------------------------------------------------------------------
        ! CASE: Simha (5)
        ! --------------------------------------------------------------------
        if (pset%sfh == 5) then
            ! A. Delayed Tau Portion
            sfh%type = 4
            imin = 0
            call compute_ssp_weights(ctx, sfh, imin, imax, tmp1(:))
            mass1 = 0.0_wp
            do i = 1, imax
                mass1 = mass1 + tmp1(i)
            end do

            ! B. Linear Cutoff Portion
            sfh%type = 5
            sfh%use_simha_limits = 1
            call compute_ssp_weights(ctx, sfh, imin, imax, tmp2(:))
            sfh%use_simha_limits = 0
            mass2 = 0.0_wp
            do i = 1, imax
                mass2 = mass2 + tmp2(i)
            end do

            ! Normalize
            if (mass1 < SAFE_FLOOR) mass1 = 1.0_wp
            if (mass2 < SAFE_FLOOR) mass2 = 1.0_wp

            ! C. Get Mixing Fraction
            call get_sfh_properties_at_age(ctx, pset, tage, mass_frac, sfr, frac_linear)

            ! Combine
            do i = 1, nt
                w(i, 1) = (tmp1(i) / mass1) * (1.0_wp - frac_linear) + &
                                       (tmp2(i) / mass2) * frac_linear
            end do
            return
        end if

        ! --------------------------------------------------------------------
        ! CASE: Tabular (2=File, 3=Array)
        ! --------------------------------------------------------------------
        if (pset%sfh == 2 .or. pset%sfh == 3) then
            ! We treat the table as a series of linear SFH segments.
            ! sfh_tab is in [Years, SFR, Z]
            
            ! Loop over table bins
            do j = 1, ctx%state%ntabsfh - 1
                
                ! Bin Edges in Lookback Time (Note: Table is in Forward Time)
                ! t1 = Older edge, t2 = Younger edge
                t1 = tage * 1.0e9_wp - ctx%state%sfh_tab(1, j+1)
                t2 = tage * 1.0e9_wp - ctx%state%sfh_tab(1, j)

                ! Skip bins entirely in the future
                if (t2 < 0.0_wp) cycle

                ! Average Z of bin
                zbin = (ctx%state%sfh_tab(3, j) + ctx%state%sfh_tab(3, j+1)) / 2.0_wp

                ! Calculate Slope for "Linear" Type 5 model
                ! Slope = - dSFR / dt / SFR_start (approx normalization for Simha model logic)
                ! Legacy code uses this specific definition compatible with Type 5 math.
                ! Note: Denominator is dt = (t2 - t1)
                dt = max(t2 - t1, SAFE_FLOOR)
                sfh%sf_slope = -(ctx%state%sfh_tab(2, j+1) - ctx%state%sfh_tab(2, j)) / &
                               dt / max(ctx%state%sfh_tab(2, j+1), SAFE_FLOOR)

                ! Set Integration Limits for this bin
                sfh%type = 5
                sfh%tq   = min(max(t1, 10.0_wp**ctx%tiny_logt_val), 10.0_wp**ctx%state%time_full(nt))
                sfh%tage = min(max(t2, 10.0_wp**ctx%tiny_logt_val), 10.0_wp**ctx%state%time_full(nt))
                sfh%sf_trunc = sfh%tage - sfh%tq ! Width of integration

                ! Calculate Mass formed in this bin (Analytic integral of linear segment)
                ! Mass = SFR_end * (1 + slope/2 * width) * width
                mass2 = ctx%state%sfh_tab(2, j+1) * &
                        (1.0_wp + sfh%sf_slope/2.0_wp * (sfh%tage + sfh%tq - 2.0_wp*t1)) * &
                        (sfh%tage - sfh%tq)

                ! Get Weights for this segment
                imin = min(max(find_interval(ctx%state%time_full, log10(max(t1, SAFE_FLOOR))) - 1, 0), nt)
                imax = min(max(find_interval(ctx%state%time_full, log10(max(t2, SAFE_FLOOR))) + 2, 0), nt)
                
                call compute_ssp_weights(ctx, sfh, imin, imax, tmp1(:))
                
                mass1 = 0.0_wp
                do i = 1, nt
                    mass1 = mass1 + tmp1(i)
                end do
                if (mass1 < SAFE_FLOOR) mass1 = 1.0_wp

                ! Distribute to Metallicities
                if (nzin > 1) then
                    ! Find Z interval
                    k = max(min(find_interval(ctx%state%zlegend, zbin), ctx%state%nz - 1), 1)
                    
                    ! Calc interpolation factor dz
                    dz = (log10(zbin) - log10(ctx%state%zlegend(k))) / &
                         (log10(ctx%state%zlegend(k+1)) - log10(ctx%state%zlegend(k)))
                    dz = max(min(dz, 1.0_wp), -1.0_wp) ! Clamp extrapolation
                    
                    ! Vectorized Add
                    do i = 1, nt
                        w(i, k)   = w(i, k) + &
                                                 (1.0_wp - dz) * tmp1(i) * (mass2 / mass1)
                        w(i, k+1) = w(i, k+1) + &
                                                 dz            * tmp1(i) * (mass2 / mass1)
                    end do
                else
                    ! Single Z
                    do i = 1, nt
                        w(i, 1) = w(i, 1) + tmp1(i) * (mass2 / mass1)
                    end do
                end if
            end do
        end if

    end subroutine sfh_weights_core

    !> @brief Converts user parameters (pset) into internal physical units (sfhparams).
    !>
    !> @details
    !> Performs necessary unit conversions (Gyr -> Years), calculates lookback times,
    !> and sets up derived parameters like truncation times and zero-crossings.
    !>
    !> @param[in]  pset Parameters from user.
    !> @param[in]  tage Age of the galaxy [Gyr].
    !> @param[out] sfh  Internal SFH structure.
    pure subroutine convert_sfhparams(pset, tage, sfh)
        type(params), intent(in)     :: pset
        real(WP), intent(in)         :: tage
        type(sfhparams), intent(out) :: sfh

        real(WP) :: start_time
        
        ! 1. Determine Start Time (Offset)
        if (any([1, 4, 5] == pset%sfh)) then
            start_time = pset%sf_start * 1.0e9_wp
        else
            start_time = 0.0_wp
        end if

        ! 2. Basic Conversions (Gyr -> Yr)
        sfh%tage     = tage * 1.0e9_wp - start_time
        sfh%tburst   = pset%tburst * 1.0e9_wp - start_time
        sfh%sf_trunc = pset%sf_trunc * 1.0e9_wp - start_time
        sfh%tau      = pset%tau * 1.0e9_wp
        
        ! Note sign flip: pset slope is (+ = increasing in forward time)
        ! sfh slope is (+ = increasing in lookback time)
        sfh%sf_slope = -pset%sf_slope / 1.0e9_wp

        ! 3. Derived Times (Lookback)
        sfh%tb = sfh%tage - sfh%tburst

        ! Truncation Time (Lookback)
        if (sfh%sf_trunc <= 0.0_wp .or. sfh%sf_trunc > sfh%tage) then
            sfh%tq = 0.0_wp
        else
            sfh%tq = sfh%tage - sfh%sf_trunc
        end if

        ! 4. Zero Crossing for Simha/Linear models
        ! t0 = time where SFR hits zero
        if (sfh%sf_slope > SAFE_FLOOR) then
            sfh%t0 = sfh%tq - (1.0_wp / sfh%sf_slope)
        else
            sfh%t0 = 0.0_wp
        end if

        ! Clamp t0 to be physical
        if (sfh%t0 > sfh%tq .or. sfh%t0 <= 0.0_wp) then
            sfh%t0 = 0.0_wp
        end if
        
        ! Initialize Simha flag
        sfh%use_simha_limits = 0

    end subroutine convert_sfhparams

end module fsps_sfh
