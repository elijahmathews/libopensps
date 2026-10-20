module fsps_types
    !> @brief
    !> Core data structures used across FSPS.
    !>
    !> @details
    !> Defines parameter sets and output containers used throughout the library.

    use fsps_precision, only: WP
    use fsps_constants, only: NPZPHOT

    implicit none
    save
    private

    public :: PARAMS
    public :: COMPSPOUT
    public :: SFHPARAMS
    public :: TLSF
    public :: OBSDAT
    public :: TPZPHOT

    ! ---------------------------------------------------------------------
    ! Derived types
    ! ---------------------------------------------------------------------

    !> @brief Parameter set used to generate a model.
    type PARAMS
        real(WP) :: pagb = 1.0_wp, dell = 0.0_wp, delt = 0.0_wp, fbhb = 0.0_wp, sbss = 0.0_wp, tau = 1.0_wp, &
                    const = 0.0_wp, tage = 0.0_wp, fburst = 0.0_wp, tburst = 11.0_wp, dust1 = 0.0_wp, dust2 = 0.0_wp, &
                    logzsol = 0.0_wp, zred = 0.0_wp, pmetals = 0.02_wp, imf1 = 1.3_wp, imf2 = 2.3_wp, imf3 = 2.3_wp, &
                    vdmc = 0.08_wp, dust_clumps = -99.0_wp, frac_nodust = 0.0_wp, dust_index = -0.7_wp, &
                    dust_tesc = 7.0_wp, frac_obrun = 0.0_wp, uvb = 1.0_wp, mwr = 3.1_wp, redgb = 1.0_wp, agb = 1.0_wp, &
                    dust1_index = -1.0_wp, mdave = 0.5_wp, sf_start = 0.0_wp, sf_trunc = 0.0_wp, sf_slope = 0.0_wp, &
                    duste_gamma = 0.01_wp, duste_umin = 1.0_wp, duste_qpah = 3.5_wp, fcstar = 1.0_wp, &
                    masscut = 150.0_wp, sigma_smooth = 0.0_wp, agb_dust = 1.0_wp, min_wave_smooth = 1.0e3_wp, &
                    max_wave_smooth = 1.0e4_wp, gas_logu = -2.0_wp, gas_logz = 0.0_wp, igm_factor = 1.0_wp, &
                    fagn = 0.0_wp, agn_tau = 10.0_wp, frac_xrb = 1.0_wp, dust3 = 0.0_wp
        integer :: zmet = 1, sfh = 0, wgp1 = 1, wgp2 = 1, wgp3 = 1, evtype = -1
        integer :: compute_mags = 1, compute_indices = 1
        integer, allocatable :: mag_compute(:)
        integer, allocatable :: ssp_gen_age(:)
        character(50) :: imf_filename = '', sfh_filename = ''
    end type PARAMS

    !> @brief Output container for CSP results.
    type COMPSPOUT
        real(WP) :: age = 0.0_wp, mass_csp = 0.0_wp, lbol_csp = 0.0_wp, sfr = 0.0_wp, mdust = 0.0_wp, mformed = 0.0_wp
        real(WP), allocatable :: mags(:)
        real(WP), allocatable :: spec(:)
        real(WP), allocatable :: indx(:)
        real(WP), allocatable :: emlines(:)
    end type COMPSPOUT

    !> @brief SFH parameters converted to intrinsic units.
    type SFHPARAMS
        real(WP) :: tau = 1.0_wp, tage = 0.0_wp, tburst = 0.0_wp, sf_trunc = 0.0_wp, sf_slope = 0.0_wp, &
                    tq = 0.0_wp, t0 = 0.0_wp, tb = 0.0_wp
        integer :: type = 0, use_simha_limits = 0
    end type SFHPARAMS

    !> @brief Line-spread function container.
    type TLSF
        real(WP), allocatable :: lsf(:)
        real(WP) :: minlam = 0.0_wp, maxlam = 0.0_wp
    end type TLSF

    !> @brief Observational data container.
    type OBSDAT
        real(WP) :: zred = 0.0_wp, logsmass = 0.0_wp
        real(WP), allocatable :: mags(:), magerr(:)
        real(WP), allocatable :: spec(:), specerr(:)
    end type OBSDAT

    !> @brief Photometric redshift grid container.
    type TPZPHOT
        real(WP), dimension(NPZPHOT) :: zz = 0.0_wp, pz = 0.0_wp
    end type TPZPHOT

end module fsps_types
