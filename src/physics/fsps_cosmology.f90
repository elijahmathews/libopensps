module fsps_cosmology
    !> @brief
    !> Unified module for cosmological calculations and spectral manipulations dependent on redshift or IGM physics.
    !>
    !> @details
    !> This module consolidates functionality related to:
    !> - Cosmological distance and age calculations (assuming a flat Lambda-CDM universe).
    !> - IGM transmission curves based on Madau (1995).
    !> - Vacuum <-> Air wavelength conversions based on Morton (1991).
    !> - Convolving stellar populations with Metallicity Distribution Functions (MDF).
    !>
    !> It replaces the legacy files: `cosmo_tuniv.f90`, `cosmo_lumdist.f90`, `igm_absorb.f90`, 
    !> `vacair_conv.f90`, and `cosmo_pz_convol.f90`.
    !>
    !> All calculations utilize the `fsps_context_t` for state access (e.g., cosmology parameters H0, Om0, Ol0)
    !> and adhere to `WP` precision.

    use fsps_precision, only: WP, KP
    use fsps_context_types, only: fsps_context_t
    use fsps_constants, only: C_LIGHT
    use fsps_interpolation, only: find_interval, interpolate_linear
    use fsps_integration, only: integrate_trapezoid_array

    implicit none
    private

    ! Public Interface
    public :: air_to_vacuum
    public :: vacuum_to_air
    public :: get_universe_age
    public :: get_luminosity_distance
    public :: compute_igm_transmission, igm_tau
    public :: convolve_with_mdf

    ! ------------------------------------------------------------------------
    ! CONSTANTS
    ! ------------------------------------------------------------------------
    
    ! --- Cosmology Integration Constants ---
    integer, parameter  :: N_COSMO_GRID       = 10000     !> Grid points for age/distance integration
    real(WP), parameter :: Z_BIG_BANG         = 1.0e4_wp  !> Proxy for infinity/Big Bang redshift
    real(WP), parameter :: HUBBLE_TIME_FACTOR = 978.0_wp  !> Conversion factor for 1/H0 to Gyr
    real(WP), parameter :: PC_PER_MPC         = 1.0e6_wp  !> Parsecs per Megaparsec
    real(WP), parameter :: KM_PER_ANGSTROM    = 1.0e-13_wp !> Angstroms to Kilometers

    ! --- Wavelength Conversion Constants (Morton 1991) ---
    real(WP), parameter :: VAC_AIR_CUTOFF     = 2000.0_wp !> No conversion below 2000 Angstroms
    
    ! Air -> Vacuum Coefficients
    real(WP), parameter :: MORTON_A0 = 6.4328e-5_wp
    real(WP), parameter :: MORTON_A1 = 2.94981e-2_wp
    real(WP), parameter :: MORTON_A2 = 146.0_wp
    real(WP), parameter :: MORTON_A3 = 2.5540e-4_wp
    real(WP), parameter :: MORTON_A4 = 41.0_wp
    
    ! Vacuum -> Air Coefficients
    real(WP), parameter :: MORTON_B0 = 2.735182e-4_wp
    real(WP), parameter :: MORTON_B1 = 131.4182_wp
    real(WP), parameter :: MORTON_B2 = 2.76249e8_wp

    ! --- IGM Absorption Constants (Madau 1995) ---
    integer, parameter  :: N_LYMAN_LINES = 17
    real(WP), parameter :: LYMAN_LIMIT   = 911.75_wp
    real(WP), parameter :: A_METAL       = 0.0017_wp
    
    ! Lyman Continuum Fitting Coefficients (Eq 16 approximation)
    real(WP), parameter :: MADAU_C1 = 0.25_wp
    real(WP), parameter :: MADAU_C2 = 9.4_wp
    real(WP), parameter :: MADAU_C3 = 0.7_wp
    real(WP), parameter :: MADAU_C4 = 0.023_wp
    
    real(WP), parameter :: MADAU_P1 = 0.46_wp
    real(WP), parameter :: MADAU_P2 = 0.18_wp
    real(WP), parameter :: MADAU_P3 = 1.32_wp
    real(WP), parameter :: MADAU_P4 = 1.68_wp

    ! Lyman Series Wavelengths (Angstroms)
    real(WP), dimension(N_LYMAN_LINES), parameter :: LY_WAVE = [ &
        1215.67_wp, 1025.72_wp, 972.537_wp, 949.743_wp, 937.803_wp, &
         930.748_wp, 926.226_wp, 923.150_wp, 920.963_wp, 919.352_wp, &
         918.129_wp, 917.181_wp, 916.429_wp, 915.824_wp, 915.329_wp, &
         914.919_wp, 914.576_wp ]

    ! Lyman Series Coefficients
    real(WP), dimension(N_LYMAN_LINES), parameter :: LY_COEFF = [ &
        0.0036_wp,    0.0017_wp,    0.0011846_wp, 0.0009410_wp, 0.0007960_wp, &
        0.0006967_wp, 0.0006236_wp, 0.0005665_wp, 0.0005200_wp, 0.0004817_wp, &
        0.0004487_wp, 0.0004200_wp, 0.0003947_wp, 0.000372_wp,  0.000352_wp,  &
        0.0003334_wp, 0.00031644_wp ]

    ! --- MDF Convolution Constants ---
    integer, parameter :: N_MDF_HIGH_RES = 100       !> Grid size for high-res log-Z interpolation

contains

    !> @brief
    !> Converts wavelengths from Air to Vacuum standard.
    !>
    !> @details
    !> Performs the conversion using the dispersion formula from Morton (1991, Ap.J. Suppl. 77, 119).
    !> This conversion is only applied to wavelengths greater than 2000 Å. For wavelengths
    !> below this threshold, the function acts as an identity pass-through, assuming vacuum conditions.
    !>
    !> The formula converts the input wavelength to wavenumber squared ($\sigma^2$) in inverse microns,
    !> calculates the refractive index scaling factor, and applies it.
    !>
    !> As an `elemental` function, this can be applied seamlessly to scalars or arrays of any rank.
    !>
    !> @param[in] wavelength_angstroms  Input wavelength in Angstroms (Air).
    !> @return                          Converted wavelength in Angstroms (Vacuum).
    elemental function air_to_vacuum(wavelength_angstroms) result(wavelength_vacuum)
        real(WP), intent(in) :: wavelength_angstroms
        real(WP) :: wavelength_vacuum
        real(WP) :: wavenumber_squared, refraction_factor

        ! No conversion for wavelengths < 2000 A
        if (wavelength_angstroms < VAC_AIR_CUTOFF) then
            wavelength_vacuum = wavelength_angstroms
            return
        end if

        ! Convert to wavenumber squared (sigma in inverse microns)
        ! sigma = 10^4 / lambda(A)
        wavenumber_squared = (1.0e4_wp / wavelength_angstroms)**2

        ! Compute conversion factor (Morton 1991)
        refraction_factor = 1.0_wp + MORTON_A0 + &
               (MORTON_A1 / (MORTON_A2 - wavenumber_squared)) + &
               (MORTON_A3 / (MORTON_A4 - wavenumber_squared))

        wavelength_vacuum = wavelength_angstroms * refraction_factor

    end function air_to_vacuum

    !> @brief
    !> Converts wavelengths from Vacuum to Air standard.
    !>
    !> @details
    !> Performs the inverse conversion using the formula from Morton (1991, Ap.J. Suppl. 77, 119).
    !> Like its counterpart, this conversion is strictly applied to wavelengths greater than 2000 Å.
    !> Wavelengths below this cutoff are returned unchanged.
    !>
    !> The implementation uses a polynomial expansion in terms of $1/\lambda^2$ to determine
    !> the refractive index scaling factor.
    !>
    !> As an `elemental` function, this can be applied seamlessly to scalars or arrays of any rank.
    !>
    !> @param[in] wavelength_angstroms  Input wavelength in Angstroms (Vacuum).
    !> @return                          Converted wavelength in Angstroms (Air).
    elemental function vacuum_to_air(wavelength_angstroms) result(wavelength_air)
        real(WP), intent(in) :: wavelength_angstroms
        real(WP) :: wavelength_air
        real(WP) :: refraction_factor

        ! No conversion for wavelengths < 2000 A
        if (wavelength_angstroms < VAC_AIR_CUTOFF) then
            wavelength_air = wavelength_angstroms
            return
        end if

        ! Compute conversion factor (Morton 1991)
        ! Formula uses wavelength in Angstroms directly
        refraction_factor = 1.0_wp + MORTON_B0 + &
               (MORTON_B1 / wavelength_angstroms**2) + &
               (MORTON_B2 / wavelength_angstroms**4)

        wavelength_air = wavelength_angstroms / refraction_factor

    end function vacuum_to_air

    !> @brief
    !> Computes the age of the Universe at a specific redshift.
    !>
    !> @details
    !> Calculates the lookback time from the Big Bang ($z \approx \infty$) to the specified redshift $z$.
    !> The calculation assumes a Flat $\Lambda$CDM cosmology ($\Omega_m + \Omega_\Lambda = 1$).
    !>
    !> The time $t(z)$ is computed via the integral:
    !> \f[ t(z) = \frac{1}{H_0} \int_{z}^{\infty} \frac{dz'}{(1+z') E(z')} \f]
    !> where $E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_\Lambda}$.
    !>
    !> **Algorithmic Details:**
    !> - Performs a change of variables to $u = \ln(1+z)$ to linearize the integration grid over cosmic history.
    !> - Uses a scalar accumulation loop (Trapezoidal rule) to compute the integral with $O(1)$ memory usage,
    !>   which is optimal for Automatic Differentiation (AD) and GPU register usage.
    !> - Integration runs from the target $z$ up to a proxy for the Big Bang ($z=10,000$).
    !>
    !> @param[in] ctx       The FSPS context structure containing cosmological parameters ($H_0, \Omega_m, \Omega_\Lambda$).
    !> @param[in] redshift  The target redshift.
    !> @return              Age of the universe in Gigayears (Gyr).
    pure function get_universe_age(ctx, redshift) result(age_gyr)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: redshift
        real(WP) :: age_gyr
        
        real(WP) :: hubble_time_gyr, log_one_plus_z_start, log_one_plus_z_end, d_log_z
        real(WP) :: integrand_now, integrand_prev, integral_sum, log_z_curr
        integer :: i

        hubble_time_gyr      = HUBBLE_TIME_FACTOR / ctx%H0_val
        log_one_plus_z_start = log(1.0_wp + redshift)
        log_one_plus_z_end   = log(Z_BIG_BANG) 
        d_log_z              = (log_one_plus_z_end - log_one_plus_z_start) / real(N_COSMO_GRID - 1, WP)

        ! Initial point (i=1)
        integrand_prev = 1.0_wp / sqrt(ctx%om0_val * exp(3.0_wp * log_one_plus_z_start) + ctx%ol0_val)
        integral_sum   = 0.0_wp

        ! Scalar accumulation loop
        do i = 2, N_COSMO_GRID
            log_z_curr    = log_one_plus_z_start + real(i - 1, WP) * d_log_z
            integrand_now = 1.0_wp / sqrt(ctx%om0_val * exp(3.0_wp * log_z_curr) + ctx%ol0_val)
            
            ! Trapezoid rule: 0.5 * dx * (f(a) + f(b))
            integral_sum   = integral_sum + 0.5_wp * d_log_z * (integrand_prev + integrand_now)
            integrand_prev = integrand_now
        end do

        age_gyr = integral_sum * hubble_time_gyr

    end function get_universe_age

    !> @brief
    !> Computes the Luminosity Distance ($D_L$) to a given redshift.
    !>
    !> @details
    !> Calculates the distance measure defined by $D_L = \sqrt{L / 4\pi F}$.
    !> For a Flat $\Lambda$CDM universe, this is related to the comoving transverse distance $D_M$ by:
    !> \f[ D_L = (1+z) D_M = (1+z) \frac{c}{H_0} \int_{0}^{z} \frac{dz'}{E(z')} \f]
    !>
    !> **Algorithmic Details:**
    !> - Integrates the dimensionless Hubble parameter $1/E(z)$ from $z'=0$ to $z'=z$.
    !> - Uses a linear grid with `N_COSMO_GRID` steps and vectorized scalar accumulation.
    !> - Explicitly handles units to return the result in Parsecs (pc), converting $c$ and $H_0$ appropriately.
    !>
    !> @param[in] ctx       The FSPS context structure containing cosmological parameters.
    !> @param[in] redshift  The target redshift.
    !> @return              Luminosity distance in Parsecs (pc).
    pure function get_luminosity_distance(ctx, redshift) result(luminosity_dist_pc)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: redshift
        real(WP) :: luminosity_dist_pc

        real(WP) :: hubble_distance_pc, delta_redshift, integral_sum, integrand_now, integrand_prev, z_curr
        integer :: i

        if (redshift <= 0.0_wp) then
            luminosity_dist_pc = 0.0_wp
            return
        end if

        hubble_distance_pc = ((C_LIGHT * KM_PER_ANGSTROM) / ctx%H0_val) * PC_PER_MPC
        delta_redshift     = redshift / real(N_COSMO_GRID - 1, WP)
        
        ! Initial point (z=0)
        ! E(0) = sqrt(om0 + ol0) = 1 (if flat)
        integrand_prev = 1.0_wp / sqrt(ctx%om0_val + ctx%ol0_val)
        integral_sum   = 0.0_wp
        
        ! Scalar accumulation loop
        do i = 2, N_COSMO_GRID
            z_curr        = real(i - 1, WP) * delta_redshift
            integrand_now = 1.0_wp / sqrt(ctx%om0_val * (1.0_wp + z_curr)**3 + ctx%ol0_val)
            
            integral_sum   = integral_sum + 0.5_wp * delta_redshift * (integrand_prev + integrand_now)
            integrand_prev = integrand_now
        end do

        luminosity_dist_pc = (1.0_wp + redshift) * hubble_distance_pc * integral_sum

    end function get_luminosity_distance

    !> @brief
    !> Calculates the Intergalactic Medium (IGM) transmission curve.
    !>
    !> @details
    !> Computes the fraction of flux transmitted through the IGM as a function of wavelength and redshift,
    !> based on the prescription by Madau (1995). The transmission is defined as $T = e^{-\tau_{\text{eff}}}$.
    !>
    !> The optical depth $\tau_{\text{eff}}$ includes contributions from:
    !> 1.  **Lyman Series Line Blanketing:** discrete absorption by Hydrogen clouds at $z < z_{\text{source}}$.
    !>     Sums contributions from the first 17 Lyman transition lines.
    !> 2.  **Lyman Continuum Absorption:** Photoelectric absorption for photons with $\lambda < 912$ Å.
    !>     Uses the approximate analytic fits (Eq. 16) from Madau (1995).
    !>
    !> @param[in]    wavelength_grid      Wavelength grid in Angstroms.
    !> @param[in]    source_redshift      Redshift of the source.
    !> @param[in]    optical_depth_factor Scaling factor for $\tau$.
    !> @param[out]   transmission         Transmission fraction array [0.0 - 1.0].
    subroutine compute_igm_transmission(wavelength_grid, source_redshift, optical_depth_factor, transmission)
        real(WP), dimension(:), intent(in) :: wavelength_grid
        real(WP), intent(in) :: source_redshift
        real(WP), intent(in) :: optical_depth_factor
        real(KP), dimension(:), intent(out) :: transmission

        real(WP) :: one_plus_z, observed_wavelength, lambda_ratio, tau_val
        real(WP) :: max_tau
        integer :: i, j, max_valid_idx, num_wavelengths

        one_plus_z      = 1.0_wp + source_redshift
        num_wavelengths = size(wavelength_grid)

        ! 1. Calculate Tau directly into transmission array (Single Pass)
        !$acc parallel loop present(wavelength_grid, transmission) private(observed_wavelength, lambda_ratio, tau_val, i)
        do j = 1, num_wavelengths
            observed_wavelength = wavelength_grid(j) * one_plus_z
            lambda_ratio        = observed_wavelength / LYMAN_LIMIT
            tau_val             = 0.0_wp

            ! A. Lyman Series Line Blanketing
            do i = 1, N_LYMAN_LINES
                if (wavelength_grid(j) < LY_WAVE(i)) then
                    tau_val = tau_val + LY_COEFF(i) * (observed_wavelength / LY_WAVE(i))**3.46_wp

                    ! Metal blanketing (only for Ly-alpha)
                    if (i == 1) then
                        tau_val = tau_val + A_METAL * (observed_wavelength / LY_WAVE(i))**1.68_wp
                    end if
                end if
            end do

            ! B. Lyman Continuum Absorption
            if (wavelength_grid(j) < LYMAN_LIMIT) then
                 tau_val = tau_val + &
                    (MADAU_C1 * lambda_ratio**3.0_wp * (one_plus_z**MADAU_P1 - lambda_ratio**MADAU_P1)) + &
                    (MADAU_C2 * lambda_ratio**1.5_wp * (one_plus_z**MADAU_P2 - lambda_ratio**MADAU_P2)) - &
                    (MADAU_C3 * lambda_ratio**3.0_wp * (lambda_ratio**(-MADAU_P3) - one_plus_z**(-MADAU_P3))) - &
                    (MADAU_C4 * (one_plus_z**MADAU_P4 - lambda_ratio**MADAU_P4))
            end if

            transmission(j) = tau_val
        end do

        ! 2. Safety Cap for Short Wavelengths
        ! Serial device execution to avoid PCIe bounce while tracking maxloc
        !$acc serial present(transmission)
        max_tau = 0.0_wp
        max_valid_idx = 1
        do j = 1, num_wavelengths
            if (transmission(j) > max_tau) then
                max_tau = transmission(j)
                max_valid_idx = j
            end if
        end do

        if (max_valid_idx > 1) then
            do j = 1, max_valid_idx
                transmission(j) = max_tau
            end do
        end if
        !$acc end serial

        ! 3. Convert to Transmission
        !$acc parallel loop present(transmission)
        do j = 1, num_wavelengths
            transmission(j) = exp(-transmission(j) * optical_depth_factor)
        end do

    end subroutine compute_igm_transmission

    !> @brief Madau (1995) IGM optical depth at rest wavelength `wavelength` for a source at
    !> redshift one_plus_z - 1 (the per-wavelength part of compute_igm_transmission, before the
    !> short-wavelength cap and the optical_depth_factor), for the batch's per-model IGM.
    !> compute_igm_transmission keeps its own copy of these lines: calling this function from
    !> its OpenACC kernel gave wrong optical depths in OpenACC multicore builds (nvfortran 25.1,
    !> IGM unit tests; GPU builds were fine), cause not found.
    function igm_tau(wavelength, one_plus_z) result(tau_val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: wavelength, one_plus_z
        real(WP) :: tau_val
        real(WP) :: observed_wavelength, lambda_ratio
        integer  :: i
        ! Local copies for OpenMP target builds: OpenMP device code (nvfortran 25.1) does not take
        ! the module's parameter arrays in a declare-target routine; OpenACC multicore (nvfortran
        ! 25.1) gets the local copies wrong (IGM unit tests fail), so other builds use the module's
#ifdef OMP_TARGET
        real(WP), parameter :: LYW(N_LYMAN_LINES) = LY_WAVE, LYC(N_LYMAN_LINES) = LY_COEFF
#else
#define LYW LY_WAVE
#define LYC LY_COEFF
#endif

        observed_wavelength = wavelength * one_plus_z
        lambda_ratio        = observed_wavelength / LYMAN_LIMIT
        tau_val             = 0.0_wp

        ! A. Lyman Series Line Blanketing
        do i = 1, N_LYMAN_LINES
            if (wavelength < LYW(i)) then
                tau_val = tau_val + LYC(i) * (observed_wavelength / LYW(i))**3.46_wp

                ! Metal blanketing (only for Ly-alpha)
                if (i == 1) then
                    tau_val = tau_val + A_METAL * (observed_wavelength / LYW(i))**1.68_wp
                end if
            end if
        end do

        ! B. Lyman Continuum Absorption
        if (wavelength < LYMAN_LIMIT) then
             tau_val = tau_val + &
                (MADAU_C1 * lambda_ratio**3.0_wp * (one_plus_z**MADAU_P1 - lambda_ratio**MADAU_P1)) + &
                (MADAU_C2 * lambda_ratio**1.5_wp * (one_plus_z**MADAU_P2 - lambda_ratio**MADAU_P2)) - &
                (MADAU_C3 * lambda_ratio**3.0_wp * (lambda_ratio**(-MADAU_P3) - one_plus_z**(-MADAU_P3))) - &
                (MADAU_C4 * (one_plus_z**MADAU_P4 - lambda_ratio**MADAU_P4))
        end if
    end function igm_tau

    !> @brief
    !> Convolves Single Stellar Populations (SSPs) with a Metallicity Distribution Function (MDF).
    !>
    !> @details
    !> This routine constructs a composite spectrum by weighting SSPs of different metallicities according
    !> to an analytic MDF $P(Z)$:
    !> \f[ P(Z) \propto Z^{\text{zpow}} \exp(-Z / \text{yield}) \f]
    !>
    !> **Dual-Path Strategy:**
    !> 1.  **Standard Path (nz=22):** If the context uses the standard 22-point metallicity grid, the routine
    !>     performs a direct weighted summation of the existing SSPs. Integration weights are pre-calculated
    !>     using the Trapezoidal rule in log-Z space. The accumulation loops are ordered to access memory linearly.
    !> 2.  **High-Res Path (Generic):** For non-standard grids, it generates a high-resolution (100-point)
    !>     logarithmic Z-grid. It pre-computes an "Interpolation Map" that links the high-res grid to the
    !>     native context grid indices. This allows for fast log-log interpolation of spectra without
    !>     performing repeated binary searches ($O(1)$ access vs $O(\log N)$).
    !>
    !> **Memory Optimization:**
    !> Uses `associate` and Fortran `block` constructs with automatic arrays to allocate temporary buffers
    !> on the stack rather than the heap. This eliminates allocation latency and heap fragmentation.
    !>
    !> @param[in]  ctx                  The FSPS context containing the base SSP spectral grids `spec_ssp_zz`.
    !> @param[in]  effective_yield      The effective yield parameter.
    !> @param[out] average_metallicity  The average metallicity ($\langle Z \rangle$).
    !> @param[out] convolved_spectrum   The convolved spectra [dim: nspec x nt].
    !> @param[out] convolved_lbol       The convolved bolometric luminosity [dim: nt].
    !> @param[out] convolved_mass       The convolved stellar mass [dim: nt].
    subroutine convolve_with_mdf(ctx, effective_yield, average_metallicity, convolved_spectrum, convolved_lbol, convolved_mass)
        type(fsps_context_t), intent(in) :: ctx
        real(WP), intent(in) :: effective_yield
        real(WP), intent(out) :: average_metallicity
        real(WP), dimension(:,:), intent(out), contiguous :: convolved_spectrum
        real(WP), dimension(:), intent(out), contiguous :: convolved_lbol
        real(WP), dimension(:), intent(out), contiguous :: convolved_mass

        ! Local Variables
        integer :: num_wavelengths, num_timesteps, num_metallicities
        integer :: idx_wavelength, idx_time, idx_metallicity, idx_grid, i
        integer :: idx_grid_next
        real(WP) :: power_law_index, normalization_factor, delta_log_z, weight_val, log_z_val
        real(WP) :: val_interpolated_now, val_interpolated_prev
        real(WP) :: w_next, p_now, p_next, trapz_weight
        real(WP) :: val_mass_now, val_mass_next, val_lbol_now, val_lbol_next
        
        ! High-res grid variables
        real(WP), dimension(N_MDF_HIGH_RES) :: high_res_z_grid, high_res_prob_dist
        integer, dimension(N_MDF_HIGH_RES) :: map_indices
        real(WP), dimension(N_MDF_HIGH_RES) :: map_weights
        
        ! Access state
        num_wavelengths   = ctx%state%nspec
        num_timesteps     = ctx%state%ntfull 
        num_metallicities = ctx%state%nz
        power_law_index   = ctx%state%zpow2
        
        associate(metallicity_grid => ctx%state%zlegend)

        ! Initialize outputs
        convolved_spectrum  = 0.0_wp
        convolved_lbol      = 0.0_wp
        convolved_mass      = 0.0_wp
        average_metallicity = 0.0_wp
        normalization_factor = 0.0_wp

        ! --------------------------------------------------------------------
        ! PATH 1: Standard Grid (nz=22) - Optimized Linear Access
        ! --------------------------------------------------------------------
        if (num_metallicities == 22) then
            block
                ! Automatic arrays (allocated on stack)
                real(WP), dimension(num_metallicities) :: mdf_probability, integration_weights

                mdf_probability = (metallicity_grid**power_law_index) * exp(-metallicity_grid / effective_yield)
                integration_weights = 0.0_wp
                
                do idx_metallicity = 1, num_metallicities - 1
                    delta_log_z = log(metallicity_grid(idx_metallicity+1)) - log(metallicity_grid(idx_metallicity))
                    integration_weights(idx_metallicity)   = integration_weights(idx_metallicity) + &
                                                             0.5_wp * delta_log_z * mdf_probability(idx_metallicity)
                    integration_weights(idx_metallicity+1) = integration_weights(idx_metallicity+1) + &
                                                             0.5_wp * delta_log_z * mdf_probability(idx_metallicity+1)
                end do

                normalization_factor = sum(integration_weights)
                average_metallicity  = sum(integration_weights * metallicity_grid) / normalization_factor

                ! Reordered loops for Linear Memory Access
                ! spec_ssp_zz is (nspec, nt, nz). Iterating nz outermost makes access somewhat stride-y 
                ! for the 3rd dimension, but allows us to reuse the weight scalar for an entire spectral sheet.
                
                do idx_metallicity = 1, num_metallicities
                    if (integration_weights(idx_metallicity) <= tiny(0.0_wp)) cycle
                    weight_val = integration_weights(idx_metallicity)
                    
                    convolved_lbol(1:num_timesteps) = convolved_lbol(1:num_timesteps) + &
                        ctx%state%lbol_ssp_zz(1:num_timesteps, idx_metallicity) * weight_val
                        
                    convolved_mass(1:num_timesteps) = convolved_mass(1:num_timesteps) + &
                        ctx%state%mass_ssp_zz(1:num_timesteps, idx_metallicity) * weight_val

                    do idx_time = 1, num_timesteps
                       convolved_spectrum(:, idx_time) = convolved_spectrum(:, idx_time) + &
                           ctx%state%spec_ssp_zz(:, idx_time, idx_metallicity) * weight_val
                    end do
                end do
                
                convolved_spectrum = convolved_spectrum / normalization_factor
                convolved_lbol     = convolved_lbol / normalization_factor
                convolved_mass     = convolved_mass / normalization_factor
            end block

        ! --------------------------------------------------------------------
        ! PATH 2: High-Resolution Grid - Pre-computed Weights
        ! --------------------------------------------------------------------
        else
            ! Generate High-Res Grid (Logarithmic)
            do i = 1, N_MDF_HIGH_RES
                high_res_z_grid(i) = 10.0_wp**( (real(i,WP)/100.0_wp * 3.0_wp) - 4.0_wp )
            end do

            ! Pre-compute Probability Distribution for this Z-distribution
            high_res_prob_dist = (high_res_z_grid**power_law_index) * exp(-high_res_z_grid / effective_yield)
            
            normalization_factor = 0.0_wp
            do i = 1, N_MDF_HIGH_RES - 1
                delta_log_z = log(high_res_z_grid(i+1)) - log(high_res_z_grid(i))
                normalization_factor = normalization_factor + 0.5_wp * delta_log_z * ( &
                    high_res_prob_dist(i+1) + high_res_prob_dist(i) &
                )
                average_metallicity  = average_metallicity  + 0.5_wp * delta_log_z * ( &
                    high_res_prob_dist(i+1)*high_res_z_grid(i+1) + high_res_prob_dist(i)*high_res_z_grid(i) &
                )
            end do
            average_metallicity = average_metallicity / normalization_factor

            ! Pre-compute Interpolation Map (Map High-Res Z -> Native Z Grid indices)
            do i = 1, N_MDF_HIGH_RES
                log_z_val = log10(high_res_z_grid(i))
                idx_grid  = find_interval(log10(metallicity_grid), log_z_val)
                
                map_indices(i) = idx_grid
                map_weights(i) = (log_z_val - log10(metallicity_grid(idx_grid))) / &
                                 (log10(metallicity_grid(idx_grid+1)) - log10(metallicity_grid(idx_grid)))
            end do

            ! Convolve Properties
            ! Optimization: Loop order flipped to Time -> Metallicity -> Wavelength (Implied Vector)
            ! This allows 'idx_wavelength' to be the innermost (contiguous) access.
            do idx_time = 1, num_timesteps
                do idx_metallicity = 1, N_MDF_HIGH_RES - 1
                    
                    ! Indices and weights for Z(i) and Z(i+1)
                    idx_grid    = map_indices(idx_metallicity)
                    weight_val  = map_weights(idx_metallicity)

                    idx_grid_next = map_indices(idx_metallicity+1)
                    w_next        = map_weights(idx_metallicity+1)

                    ! Trapezoidal integration setup
                    delta_log_z = log(high_res_z_grid(idx_metallicity+1)) - log(high_res_z_grid(idx_metallicity))
                    trapz_weight = 0.5_wp * delta_log_z
                    
                    p_now  = trapz_weight * high_res_prob_dist(idx_metallicity)
                    p_next = trapz_weight * high_res_prob_dist(idx_metallicity+1)

                    ! --- Update Lbol/Mass (Linear Interp on Z) ---
                    val_mass_now = (1.0_wp - weight_val) * ctx%state%mass_ssp_zz(idx_time, idx_grid) + &
                                   weight_val            * ctx%state%mass_ssp_zz(idx_time, idx_grid+1)
                    
                    val_mass_next = (1.0_wp - w_next) * ctx%state%mass_ssp_zz(idx_time, idx_grid_next) + &
                                    w_next            * ctx%state%mass_ssp_zz(idx_time, idx_grid_next+1)

                    convolved_mass(idx_time) = convolved_mass(idx_time) + &
                         (p_next * val_mass_next + p_now * val_mass_now)

                    val_lbol_now = (1.0_wp - weight_val) * ctx%state%lbol_ssp_zz(idx_time, idx_grid) + &
                                   weight_val            * ctx%state%lbol_ssp_zz(idx_time, idx_grid+1)
                    
                    val_lbol_next = (1.0_wp - w_next) * ctx%state%lbol_ssp_zz(idx_time, idx_grid_next) + &
                                    w_next            * ctx%state%lbol_ssp_zz(idx_time, idx_grid_next+1)

                    convolved_lbol(idx_time) = convolved_lbol(idx_time) + &
                         (p_next * val_lbol_next + p_now * val_lbol_now)

                    ! --- Update Spectrum (Log-Log Interp on Z) ---
                    ! Inner loop is over contiguous wavelength dimension (Column-Major)
                    do idx_wavelength = 1, num_wavelengths
                         
                         ! Reconstruct spectra at Z_hr(i)
                         val_interpolated_now = (1.0_wp - weight_val) * &
                                                log10(ctx%state%spec_ssp_zz(idx_wavelength, idx_time, idx_grid)) + &
                                                weight_val * &
                                                log10(ctx%state%spec_ssp_zz(idx_wavelength, idx_time, idx_grid+1))
                         val_interpolated_now = 10.0_wp**val_interpolated_now

                         ! Reconstruct spectra at Z_hr(i+1)
                         val_interpolated_prev = (1.0_wp - w_next) * &
                                                 log10(ctx%state%spec_ssp_zz(idx_wavelength, idx_time, idx_grid_next)) + &
                                                 w_next * &
                                                 log10(ctx%state%spec_ssp_zz(idx_wavelength, idx_time, idx_grid_next+1))
                         val_interpolated_prev = 10.0_wp**val_interpolated_prev
                         
                         convolved_spectrum(idx_wavelength, idx_time) = convolved_spectrum(idx_wavelength, idx_time) + &
                            (p_next * val_interpolated_prev + p_now * val_interpolated_now)
                    end do
                end do
                
                convolved_spectrum(:, idx_time) = convolved_spectrum(:, idx_time) / normalization_factor
                convolved_mass(idx_time) = convolved_mass(idx_time) / normalization_factor
                convolved_lbol(idx_time) = convolved_lbol(idx_time) / normalization_factor
            end do

        end if

        end associate

    end subroutine convolve_with_mdf

end module fsps_cosmology