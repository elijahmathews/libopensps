module fsps_constants
    !> @brief
    !> Defines global constants and array dimensions for FSPS.
    !>
    !> @details
    !> This module consolidates all physical constants, array limits, and configuration
    !> parameters used throughout the FSPS codebase.

    use, intrinsic :: iso_fortran_env, only: real64
    use fsps_precision, only: WP

    implicit none
    public

    ! ------------------------------------------------------------------------
    ! Precision & configuration
    ! ------------------------------------------------------------------------

    !> @brief Verbosity flags.
    integer, parameter :: VERBOSE_MINIMAL = 0
    integer, parameter :: VERBOSE_DEBUG = 1

    !> @brief Controls the level of output to screen.
    integer, parameter :: VERBOSE = VERBOSE_MINIMAL

    ! ------------------------------------------------------------------------
    ! Physical constants (fundamental)
    ! ------------------------------------------------------------------------
    ! Defined first so they can be used to calculate derived constants below.

    !> @brief π (machine precision).
    real(WP), parameter :: PI = acos(-1.0_wp)

    !> @brief Planck constant [erg s] (2022 CODATA; exact).
    real(WP), parameter :: H_PLANCK = 6.62607015e-27_wp

    !> @brief Boltzmann constant [erg K⁻¹] (2022 CODATA; exact).
    real(WP), parameter :: K_BOLTZMANN = 1.380649e-16_wp

    !> @brief Speed of light in vacuum [Å s⁻¹] (2022 CODATA; exact).
    real(WP), parameter :: C_LIGHT = 2.99792458e18_wp

    !> @brief Newtonian constant of gravitation [cm³ g⁻¹ s⁻²] (2022 CODATA; estimated).
    real(WP), parameter :: G_NEWTON = 6.6743e-8_wp

    !> @brief Nominal solar radius [cm] (IAU 2015 Resolution B3; exact).
    real(WP), parameter :: R_SOL = 6.957e10_wp

    !> @brief Nominal solar luminosity [erg s⁻¹] (IAU 2015 Resolution B3; exact).
    real(WP), parameter :: L_SOL = 3.828e33_wp

    !> @brief Nominal solar mass parameter [cm³ s⁻²] (IAU 2015 Resolution B3; exact).
    real(WP), parameter :: M_SOL_PAR = 1.3271244e26_wp

    !> @brief Astronomical unit [cm] (IAU 2012 Resolution B2; exact).
    real(WP), parameter :: A_U = 1.495978707e13_wp

    ! ------------------------------------------------------------------------
    ! Physical constants (derived)
    ! ------------------------------------------------------------------------

    !> @brief Parsec [cm].
    real(WP), parameter :: PARSEC = (648000.0_wp/PI)*A_U

    !> @brief Solar mass [g].
    real(WP), parameter :: M_SOL = M_SOL_PAR/G_NEWTON

    ! Products of these overflow single precision (K_BOLTZMANN**4 ~ 4e-63, (10 pc)**2 ~ 1e39),
    ! so they are evaluated in double precision and converted once.

    !> @brief Stefan-Boltzmann constant [erg cm⁻² s⁻¹ K⁻⁴].
    real(WP), parameter :: SIGMA_SB = real((2.0_real64*acos(-1.0_real64)**5*1.380649e-16_real64**4)/ &
                           (15.0_real64*(2.99792458e18_real64*1.0e-8_real64)**2*6.62607015e-27_real64**3), WP)

    !> @brief Coefficient to calculate surface gravity (g) from L, M, and Teff.
    !> Derivation: From g = GM/R² and L = 4πR²σT⁴, we substitute R² to get:
    !>             g = (4π * G * σ_sb) * (M * T⁴ / L)
    !> This constant contains the (4π * G * σ_sb) term, adjusted for solar units.
    real(WP), parameter :: GRAVITY_L_M_T_COEFF = real(4.0_real64*acos(-1.0_real64)* &
        ((2.0_real64*acos(-1.0_real64)**5*1.380649e-16_real64**4)/ &
         (15.0_real64*(2.99792458e18_real64*1.0e-8_real64)**2*6.62607015e-27_real64**3))* &
        1.3271244e26_real64/3.828e33_real64, WP)

    !> @brief Reference log-flux for absolute bolometric magnitude [log10(erg s⁻¹ cm⁻²)].
    !> Represents the log10 flux of 1.0 solar luminosity at a distance of 10 parsecs.
    !> Used to convert theoretical luminosity to absolute magnitude (M_bol).
    real(WP), parameter :: ABS_MAG_ZEROPOINT_LOG = real(log10(3.828e33_real64/(4.0_real64*acos(-1.0_real64)* &
        (10.0_real64*(648000.0_real64/acos(-1.0_real64))*1.495978707e13_real64)**2)), WP)

    !> @brief Julian year [s].
    real(WP), parameter :: YEAR_TO_SECOND = 3.15576e7_wp

    ! ------------------------------------------------------------------------
    ! FSPS specific parameters
    ! ------------------------------------------------------------------------

    !> @brief Turn-on time for BHB and SBS phases, time is in log(yrs).
    real(WP), parameter :: BHB_SBS_TIME = 9.5_wp

    !> @brief The factor by which we increase the time array.
    integer, parameter :: TIME_RES_INCR = 1

    !> @brief Carbon star library selection.
    integer, parameter :: CSTAR_ARINGER = 1

    !> @brief BaSeL library normalization: 'wlbc' = Teff-color relations.
    character(4), parameter :: BASEL_STR = 'wlbc'

    ! ------------------------------------------------------------------------
    ! Array dimensions
    ! ------------------------------------------------------------------------

    !> @brief Wavelength dimension of each library.
    integer, parameter :: NSPEC_MILES = 5994
    integer, parameter :: NSPEC_BASEL = 1963
    integer, parameter :: NSPEC_C3K = 11149
    integer, parameter :: NSPEC_BPASS = 15000

    !> @brief Number of metallicity points in the isochrone grid.
    integer, parameter :: NZ_MIST = 12
    integer, parameter :: NZ_PADOVA = 22
    integer, parameter :: NZ_PARSEC = 15
    integer, parameter :: NZ_BASTI = 10
    integer, parameter :: NZ_GENEVA = 5
    integer, parameter :: NZ_BPASS = 12

    !> @brief Number of age points in the isochrone grid.
    integer, parameter :: NT_MIST = 107
    integer, parameter :: NT_PADOVA = 94
    integer, parameter :: NT_PARSEC = 93
    integer, parameter :: NT_BASTI = 94
    integer, parameter :: NT_GENEVA = 51
    integer, parameter :: NT_BPASS = 43

    !> @brief Dimensions for XRB spectral library.
    integer, parameter :: NT_XRB = 10
    integer, parameter :: NZ_XRB = 11
    integer, parameter :: NSPEC_XRB = 15000

    !> @brief Max dimension of array for each isochrone.
    integer, parameter :: NM = 2000

    !> @brief Max number of lines to read in.
    integer, parameter :: NLINES = 1000000

    !> @brief Max number of lines in tabulated SFH, LSF.
    integer, parameter :: NTABMAX = 20000

    !> @brief Dimensions of BaSeL library (log T and log g).
    integer, parameter :: NDIM_LOGT = 68
    integer, parameter :: NDIM_LOGG = 19

    !> @brief Number of O-rich, C-rich AGB spectra, and Aringer C-rich spectra.
    integer, parameter :: N_AGB_O = 9
    integer, parameter :: N_AGB_C = 5
    integer, parameter :: N_AGB_CAR = 9

    !> @brief Number of post-AGB spectra.
    integer, parameter :: NDIM_PAGB = 14

    !> @brief Number of WR spectra.
    integer, parameter :: NDIM_WR = 12

    !> @brief Dimensions of WMBasic grid.
    integer, parameter :: NDIM_WMB_LOGT = 11
    integer, parameter :: NDIM_WMB_LOGG = 3

    !> @brief Parameters for circumstellar dust models (tau and Teff points).
    integer, parameter :: NTAU_DAGB = 50
    integer, parameter :: NTEFF_DAGB = 6

    !> @brief Number of emission lines and continuum emission points.
    integer, parameter :: NEMLINE = 166
    integer, parameter :: NLAM_NEBCONT = 1963
    integer, parameter :: NEBNZ = 11

    !> @brief Number of metallicity, age, and ionization parameter points (nebular).
    integer, parameter :: NEBNAGE = 10
    integer, parameter :: NEBNIP = 7
    integer, parameter :: NAGNDUST = 9

    !> @brief Number of spectral points in the input library (AGN).
    integer, parameter :: NAGNDUST_SPEC = 125

    !> @brief Structure for using P(z) in chi2 (legacy/unused).
    integer, parameter :: NPZPHOT = 200

    ! ------------------------------------------------------------------------
    ! Numerical limits
    ! ------------------------------------------------------------------------

    real(WP), parameter :: SAFE_FLOOR = tiny(1.0_wp)

end module fsps_constants
