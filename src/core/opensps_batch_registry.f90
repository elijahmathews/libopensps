!> @file opensps_batch_registry.f90
!> @brief The per-model parameters of a batch (the members of PARAMS_CORE), the class of each, and the free set of a batch.
!>
!> A batch of models (compute_csp_batch) may differ in any of these. The class of a parameter says how it enters the
!> model and so which mechanism serves it and what its cost grows with (doc/HOT_PATHS.md):
!>  * C, closed form: read by uniform per-model kernels (SFH, dust, dust emission, gas, AGN, redshift): no cost of its own;
!>  * L, linear mixture: the coefficient of a fixed set of sources (metallicity, [alpha/Fe], the isochrone factors):
!>    the batch is a sum over sources;
!>  * W, weights: per-point weights of the SSP, computed once per distinct value (the IMF parameters);
!>  * G, grouped discrete: one precomputed structure per distinct value, gathered per model (the TP-AGB shifts);
!>  * S, shared: must be equal across the batch (status 2 otherwise) or are not per-model quantities.
!> The table below is generated and kept in step with PARAMS_CORE by tools/check_batch_registry.py (a unit test): a member
!> of PARAMS_CORE that is not registered fails it.
!>
!> batch_classify reports which fields differ in a batch, how many distinct values the differing ones take, and whether a
!> shared field differs. It is a diagnostic: compute_csp_batch runs it only when its optional argument info is present or
!> OPENSPS_DEBUG_PATH is set, so the default path pays nothing.
module opensps_batch_registry
    use, intrinsic :: iso_fortran_env, only: error_unit
    use opensps_precision, only: WP
    use opensps_types, only: PARAMS_CORE
    implicit none
    private

    public :: registry_name, registry_class, registry_stage, registry_kind, registry_find, registry_get, registry_set
    public :: registry_equal, registry_range, registry_class_name, registry_stage_name, batch_classify, batch_info_set_path
    public :: batch_info_text, batch_info_print, batch_debug_path

    integer, parameter, public :: CLASS_S = 0, CLASS_C = 1, CLASS_L = 2, CLASS_W = 3, CLASS_G = 4
    integer, parameter, public :: STG_SFH = 1, STG_SSP = 2, STG_DUST = 3, STG_DUSTEM = 4, STG_NEBULAR = 5, STG_AGN = 6, &
                                  STG_OBS = 7, STG_SMOOTH = 8, STG_OUTPUT = 9
    integer, parameter, public :: KIND_REAL = 1, KIND_INT = 2, KIND_STR = 3
    !> Distinct values are counted up to this many (a field with more is reported as at least this)
    integer, parameter, public :: DISTINCT_CAP = 16

    integer, parameter, public :: N_FIELDS = 63
    integer, parameter, public :: F_TAU = 1
    integer, parameter, public :: F_CONST = 2
    integer, parameter, public :: F_TAGE = 3
    integer, parameter, public :: F_FBURST = 4
    integer, parameter, public :: F_TBURST = 5
    integer, parameter, public :: F_SF_START = 6
    integer, parameter, public :: F_SF_TRUNC = 7
    integer, parameter, public :: F_SF_SLOPE = 8
    integer, parameter, public :: F_SFH = 9
    integer, parameter, public :: F_SFH_FILENAME = 10
    integer, parameter, public :: F_LOGZSOL = 11
    integer, parameter, public :: F_AFE = 12
    integer, parameter, public :: F_FBHB = 13
    integer, parameter, public :: F_SBSS = 14
    integer, parameter, public :: F_AGB = 15
    integer, parameter, public :: F_PAGB = 16
    integer, parameter, public :: F_REDGB = 17
    integer, parameter, public :: F_DELT = 18
    integer, parameter, public :: F_DELL = 19
    integer, parameter, public :: F_IMF1 = 20
    integer, parameter, public :: F_IMF2 = 21
    integer, parameter, public :: F_IMF3 = 22
    integer, parameter, public :: F_VDMC = 23
    integer, parameter, public :: F_MDAVE = 24
    integer, parameter, public :: F_MASSCUT = 25
    integer, parameter, public :: F_AGB_DUST = 26
    integer, parameter, public :: F_EVTYPE = 27
    integer, parameter, public :: F_ZMET = 28
    integer, parameter, public :: F_PMETALS = 29
    integer, parameter, public :: F_IMF_FILENAME = 30
    integer, parameter, public :: F_WGP1 = 31
    integer, parameter, public :: F_WGP2 = 32
    integer, parameter, public :: F_WGP3 = 33
    integer, parameter, public :: F_FCSTAR = 34
    integer, parameter, public :: F_FRAC_XRB = 35
    integer, parameter, public :: F_DUST1 = 36
    integer, parameter, public :: F_DUST2 = 37
    integer, parameter, public :: F_DUST3 = 38
    integer, parameter, public :: F_DUST_INDEX = 39
    integer, parameter, public :: F_DUST1_INDEX = 40
    integer, parameter, public :: F_DUST_CLUMPS = 41
    integer, parameter, public :: F_FRAC_NODUST = 42
    integer, parameter, public :: F_DUST_TESC = 43
    integer, parameter, public :: F_UVB = 44
    integer, parameter, public :: F_MWR = 45
    integer, parameter, public :: F_DUSTE_GAMMA = 46
    integer, parameter, public :: F_DUSTE_UMIN = 47
    integer, parameter, public :: F_DUSTE_QPAH = 48
    integer, parameter, public :: F_GAS_LOGU = 49
    integer, parameter, public :: F_GAS_LOGZ = 50
    integer, parameter, public :: F_GAS_LOGN = 51
    integer, parameter, public :: F_GAS_LOGNO = 52
    integer, parameter, public :: F_GAS_LOGCO = 53
    integer, parameter, public :: F_FRAC_OBRUN = 54
    integer, parameter, public :: F_FAGN = 55
    integer, parameter, public :: F_AGN_TAU = 56
    integer, parameter, public :: F_ZRED = 57
    integer, parameter, public :: F_IGM_FACTOR = 58
    integer, parameter, public :: F_SIGMA_SMOOTH = 59
    integer, parameter, public :: F_MIN_WAVE_SMOOTH = 60
    integer, parameter, public :: F_MAX_WAVE_SMOOTH = 61
    integer, parameter, public :: F_COMPUTE_MAGS = 62
    integer, parameter, public :: F_COMPUTE_INDICES = 63

    character(len=16), parameter :: field_names(N_FIELDS) = [character(len=16) :: &
        'tau', 'const', 'tage', 'fburst', 'tburst', &
        'sf_start', 'sf_trunc', 'sf_slope', 'sfh', 'sfh_filename', &
        'logzsol', 'afe', 'fbhb', 'sbss', 'agb', &
        'pagb', 'redgb', 'delt', 'dell', 'imf1', &
        'imf2', 'imf3', 'vdmc', 'mdave', 'masscut', &
        'agb_dust', 'evtype', 'zmet', 'pmetals', 'imf_filename', &
        'wgp1', 'wgp2', 'wgp3', 'fcstar', 'frac_xrb', &
        'dust1', 'dust2', 'dust3', 'dust_index', 'dust1_index', &
        'dust_clumps', 'frac_nodust', 'dust_tesc', 'uvb', 'mwr', &
        'duste_gamma', 'duste_umin', 'duste_qpah', 'gas_logu', 'gas_logz', &
        'gas_logn', 'gas_logno', 'gas_logco', 'frac_obrun', 'fagn', &
        'agn_tau', 'zred', 'igm_factor', 'sigma_smooth', 'min_wave_smooth', &
        'max_wave_smooth', 'compute_mags', 'compute_indices']
    integer, parameter :: field_class(N_FIELDS) = [ &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, &
        3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, &
        0, 0, 0]
    integer, parameter :: field_stage(N_FIELDS) = [ &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, &
        3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, &
        8, 9, 9]
    integer, parameter :: field_kind(N_FIELDS) = [ &
        1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
        1, 1, 1, 1, 1, 1, 2, 2, 1, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
        1, 2, 2]

    character(len=8), parameter :: class_names(0:4) = [character(len=8) :: 'shared', 'closed', 'linear', 'weights', 'grouped']
    character(len=8), parameter :: stage_names(9) = [character(len=8) :: 'sfh', 'ssp', 'dust', 'dust-em', 'nebular', 'agn', &
                                                                          'observer', 'smooth', 'output']

    !> What a batch is: the fields that differ, how often, and (set by compute_csp_batch) the SSP source of the path
    type, public :: batch_info_t
        integer :: n_models = 0
        !> The field differs among the models
        logical :: free(N_FIELDS) = .false.
        !> Distinct values among the models (up to DISTINCT_CAP); 1 where the field does not differ
        integer :: n_distinct(N_FIELDS) = 1
        !> Number of free fields of each class (CLASS_S .. CLASS_G)
        integer :: n_free_class(0:4) = 0
        !> A shared field (class S) differs: the batch cannot be served in one call
        logical :: shared_violation = .false.
        !> The SSP source of the path ('resident basis', 'resident bases, Z', 'operator', 'operator groups', ...) and its
        !> number of sources per model K (1, 2 or 4 slabs of a resident basis; the groups of the modification operators)
        character(len=40) :: ssp_source = ''
        integer :: k_sources = 0
    end type batch_info_t

contains

    function registry_name(i) result(s)
        integer, intent(in) :: i
        character(len=16) :: s
        s = ''
        if (i >= 1 .and. i <= N_FIELDS) s = field_names(i)
    end function registry_name

    integer function registry_class(i) result(c)
        integer, intent(in) :: i
        c = -1
        if (i >= 1 .and. i <= N_FIELDS) c = field_class(i)
    end function registry_class

    integer function registry_stage(i) result(c)
        integer, intent(in) :: i
        c = 0
        if (i >= 1 .and. i <= N_FIELDS) c = field_stage(i)
    end function registry_stage

    integer function registry_kind(i) result(c)
        integer, intent(in) :: i
        c = 0
        if (i >= 1 .and. i <= N_FIELDS) c = field_kind(i)
    end function registry_kind

    function registry_class_name(c) result(s)
        integer, intent(in) :: c
        character(len=8) :: s
        s = '?'
        if (c >= 0 .and. c <= 4) s = class_names(c)
    end function registry_class_name

    function registry_stage_name(c) result(s)
        integer, intent(in) :: c
        character(len=8) :: s
        s = '?'
        if (c >= 1 .and. c <= 9) s = stage_names(c)
    end function registry_stage_name

    !> The index of a field by name (case-insensitive), 0 if it is not registered
    integer function registry_find(name) result(idx)
        character(len=*), intent(in) :: name
        character(len=16) :: key
        integer :: i, k

        idx = 0
        key = adjustl(name)
        do k = 1, len_trim(key)
            if (key(k:k) >= 'A' .and. key(k:k) <= 'Z') key(k:k) = achar(iachar(key(k:k)) + 32)
        end do
        do i = 1, N_FIELDS
            if (trim(field_names(i)) == trim(key)) then
                idx = i
                return
            end if
        end do
    end function registry_find

    !> The value of field i of p as a real (integers converted; character fields give 0)
    function registry_get(p, i) result(v)
        type(PARAMS_CORE), intent(in) :: p
        integer, intent(in) :: i
        real(WP) :: v

        v = 0.0_wp
        select case (i)
        case (F_TAU); v = p%tau
        case (F_CONST); v = p%const
        case (F_TAGE); v = p%tage
        case (F_FBURST); v = p%fburst
        case (F_TBURST); v = p%tburst
        case (F_SF_START); v = p%sf_start
        case (F_SF_TRUNC); v = p%sf_trunc
        case (F_SF_SLOPE); v = p%sf_slope
        case (F_SFH); v = real(p%sfh, WP)
        case (F_LOGZSOL); v = p%logzsol
        case (F_AFE); v = p%afe
        case (F_FBHB); v = p%fbhb
        case (F_SBSS); v = p%sbss
        case (F_AGB); v = p%agb
        case (F_PAGB); v = p%pagb
        case (F_REDGB); v = p%redgb
        case (F_DELT); v = p%delt
        case (F_DELL); v = p%dell
        case (F_IMF1); v = p%imf1
        case (F_IMF2); v = p%imf2
        case (F_IMF3); v = p%imf3
        case (F_VDMC); v = p%vdmc
        case (F_MDAVE); v = p%mdave
        case (F_MASSCUT); v = p%masscut
        case (F_AGB_DUST); v = p%agb_dust
        case (F_EVTYPE); v = real(p%evtype, WP)
        case (F_ZMET); v = real(p%zmet, WP)
        case (F_PMETALS); v = p%pmetals
        case (F_WGP1); v = real(p%wgp1, WP)
        case (F_WGP2); v = real(p%wgp2, WP)
        case (F_WGP3); v = real(p%wgp3, WP)
        case (F_FCSTAR); v = p%fcstar
        case (F_FRAC_XRB); v = p%frac_xrb
        case (F_DUST1); v = p%dust1
        case (F_DUST2); v = p%dust2
        case (F_DUST3); v = p%dust3
        case (F_DUST_INDEX); v = p%dust_index
        case (F_DUST1_INDEX); v = p%dust1_index
        case (F_DUST_CLUMPS); v = p%dust_clumps
        case (F_FRAC_NODUST); v = p%frac_nodust
        case (F_DUST_TESC); v = p%dust_tesc
        case (F_UVB); v = p%uvb
        case (F_MWR); v = p%mwr
        case (F_DUSTE_GAMMA); v = p%duste_gamma
        case (F_DUSTE_UMIN); v = p%duste_umin
        case (F_DUSTE_QPAH); v = p%duste_qpah
        case (F_GAS_LOGU); v = p%gas_logu
        case (F_GAS_LOGZ); v = p%gas_logz
        case (F_GAS_LOGN); v = p%gas_logn
        case (F_GAS_LOGNO); v = p%gas_logno
        case (F_GAS_LOGCO); v = p%gas_logco
        case (F_FRAC_OBRUN); v = p%frac_obrun
        case (F_FAGN); v = p%fagn
        case (F_AGN_TAU); v = p%agn_tau
        case (F_ZRED); v = p%zred
        case (F_IGM_FACTOR); v = p%igm_factor
        case (F_SIGMA_SMOOTH); v = p%sigma_smooth
        case (F_MIN_WAVE_SMOOTH); v = p%min_wave_smooth
        case (F_MAX_WAVE_SMOOTH); v = p%max_wave_smooth
        case (F_COMPUTE_MAGS); v = real(p%compute_mags, WP)
        case (F_COMPUTE_INDICES); v = real(p%compute_indices, WP)
        end select
    end function registry_get

    !> Sets field i of p (integers rounded; character fields are left as they are)
    subroutine registry_set(p, i, v)
        type(PARAMS_CORE), intent(inout) :: p
        integer, intent(in) :: i
        real(WP), intent(in) :: v

        select case (i)
        case (F_TAU); p%tau = v
        case (F_CONST); p%const = v
        case (F_TAGE); p%tage = v
        case (F_FBURST); p%fburst = v
        case (F_TBURST); p%tburst = v
        case (F_SF_START); p%sf_start = v
        case (F_SF_TRUNC); p%sf_trunc = v
        case (F_SF_SLOPE); p%sf_slope = v
        case (F_SFH); p%sfh = nint(v)
        case (F_LOGZSOL); p%logzsol = v
        case (F_AFE); p%afe = v
        case (F_FBHB); p%fbhb = v
        case (F_SBSS); p%sbss = v
        case (F_AGB); p%agb = v
        case (F_PAGB); p%pagb = v
        case (F_REDGB); p%redgb = v
        case (F_DELT); p%delt = v
        case (F_DELL); p%dell = v
        case (F_IMF1); p%imf1 = v
        case (F_IMF2); p%imf2 = v
        case (F_IMF3); p%imf3 = v
        case (F_VDMC); p%vdmc = v
        case (F_MDAVE); p%mdave = v
        case (F_MASSCUT); p%masscut = v
        case (F_AGB_DUST); p%agb_dust = v
        case (F_EVTYPE); p%evtype = nint(v)
        case (F_ZMET); p%zmet = nint(v)
        case (F_PMETALS); p%pmetals = v
        case (F_WGP1); p%wgp1 = nint(v)
        case (F_WGP2); p%wgp2 = nint(v)
        case (F_WGP3); p%wgp3 = nint(v)
        case (F_FCSTAR); p%fcstar = v
        case (F_FRAC_XRB); p%frac_xrb = v
        case (F_DUST1); p%dust1 = v
        case (F_DUST2); p%dust2 = v
        case (F_DUST3); p%dust3 = v
        case (F_DUST_INDEX); p%dust_index = v
        case (F_DUST1_INDEX); p%dust1_index = v
        case (F_DUST_CLUMPS); p%dust_clumps = v
        case (F_FRAC_NODUST); p%frac_nodust = v
        case (F_DUST_TESC); p%dust_tesc = v
        case (F_UVB); p%uvb = v
        case (F_MWR); p%mwr = v
        case (F_DUSTE_GAMMA); p%duste_gamma = v
        case (F_DUSTE_UMIN); p%duste_umin = v
        case (F_DUSTE_QPAH); p%duste_qpah = v
        case (F_GAS_LOGU); p%gas_logu = v
        case (F_GAS_LOGZ); p%gas_logz = v
        case (F_GAS_LOGN); p%gas_logn = v
        case (F_GAS_LOGNO); p%gas_logno = v
        case (F_GAS_LOGCO); p%gas_logco = v
        case (F_FRAC_OBRUN); p%frac_obrun = v
        case (F_FAGN); p%fagn = v
        case (F_AGN_TAU); p%agn_tau = v
        case (F_ZRED); p%zred = v
        case (F_IGM_FACTOR); p%igm_factor = v
        case (F_SIGMA_SMOOTH); p%sigma_smooth = v
        case (F_MIN_WAVE_SMOOTH); p%min_wave_smooth = v
        case (F_MAX_WAVE_SMOOTH); p%max_wave_smooth = v
        case (F_COMPUTE_MAGS); p%compute_mags = nint(v)
        case (F_COMPUTE_INDICES); p%compute_indices = nint(v)
        end select
    end subroutine registry_set

    !> Whether field i has the same value in a and b (exactly: a batch shares a value or it does not)
    logical function registry_equal(a, b, i) result(same)
        type(PARAMS_CORE), intent(in) :: a, b
        integer, intent(in) :: i

        select case (i)
        case (F_SFH_FILENAME); same = a%sfh_filename == b%sfh_filename
        case (F_IMF_FILENAME); same = a%imf_filename == b%imf_filename
        case default
            same = registry_get(a, i) == registry_get(b, i)
        end select
    end function registry_equal

    !> The range a field that can be free is sampled over by tests and benchmarks (roughly a fit's prior, inside what the
    !> batch supports). lo = hi = 0 for a field that is not varied: the shared ones, the file names, the switches, and the
    !> SFH parameters of forms the base configurations do not use
    subroutine registry_range(i, lo, hi)
        integer, intent(in) :: i
        real(WP), intent(out) :: lo, hi

        lo = 0.0_wp
        hi = 0.0_wp
        select case (i)
        case (F_TAU); lo = 0.3_wp; hi = 5.0_wp
        case (F_TAGE); lo = 1.0_wp; hi = 12.0_wp
        case (F_CONST); lo = 0.0_wp; hi = 0.4_wp
        case (F_FBURST); lo = 0.0_wp; hi = 0.2_wp
        case (F_TBURST); lo = 1.0_wp; hi = 3.0_wp
        case (F_SF_START); lo = 0.0_wp; hi = 1.0_wp
        case (F_DUST1); lo = 0.0_wp; hi = 1.5_wp
        case (F_DUST2); lo = 0.0_wp; hi = 1.0_wp
        case (F_DUST3); lo = 0.0_wp; hi = 0.3_wp
        case (F_DUST_INDEX); lo = -1.0_wp; hi = 0.0_wp
        case (F_DUST1_INDEX); lo = -1.5_wp; hi = -0.5_wp
        case (F_FRAC_NODUST); lo = 0.0_wp; hi = 0.3_wp
        case (F_DUST_TESC); lo = 6.5_wp; hi = 7.5_wp
        case (F_DUSTE_GAMMA); lo = 0.001_wp; hi = 0.2_wp
        case (F_DUSTE_UMIN); lo = 0.5_wp; hi = 20.0_wp
        case (F_DUSTE_QPAH); lo = 0.5_wp; hi = 6.0_wp
        case (F_GAS_LOGU); lo = -4.0_wp; hi = -1.0_wp
        case (F_GAS_LOGZ); lo = -1.5_wp; hi = 0.4_wp
        case (F_FRAC_OBRUN); lo = 0.0_wp; hi = 0.3_wp
        case (F_FAGN); lo = 0.0_wp; hi = 0.5_wp
        case (F_AGN_TAU); lo = 5.0_wp; hi = 150.0_wp
        case (F_ZRED); lo = 0.1_wp; hi = 2.0_wp
        case (F_IGM_FACTOR); lo = 0.5_wp; hi = 1.5_wp
        case (F_LOGZSOL); lo = -1.5_wp; hi = 0.05_wp
        case (F_AFE); lo = -0.2_wp; hi = 0.6_wp
        case (F_FBHB); lo = 0.05_wp; hi = 0.75_wp
        case (F_SBSS); lo = 1.0_wp; hi = 6.0_wp
        case (F_AGB); lo = 0.5_wp; hi = 1.7_wp
        case (F_PAGB); lo = 0.5_wp; hi = 1.9_wp
        case (F_REDGB); lo = 0.7_wp; hi = 1.3_wp
        case (F_DELT); lo = -0.1_wp; hi = 0.1_wp
        case (F_DELL); lo = -0.1_wp; hi = 0.1_wp
        case (F_IMF1); lo = 0.9_wp; hi = 1.5_wp
        case (F_IMF2); lo = 1.8_wp; hi = 2.6_wp
        case (F_IMF3); lo = 2.0_wp; hi = 2.8_wp
        case (F_VDMC); lo = 0.05_wp; hi = 0.15_wp
        case (F_MDAVE); lo = 0.3_wp; hi = 0.7_wp
        end select
    end subroutine registry_range

    !> Classifies a batch: which fields differ among the models, how many distinct values they take (counted up to
    !> DISTINCT_CAP against the first DISTINCT_CAP distinct ones), and whether a shared field differs. The path fields
    !> (ssp_source, k_sources) are left for the caller.
    subroutine batch_classify(core, info)
        type(PARAMS_CORE), intent(in) :: core(:)
        type(batch_info_t), intent(inout) :: info
        integer :: i, m, r, nd, n
        integer :: reps(DISTINCT_CAP)
        logical :: found

        n = size(core)
        info%n_models = n
        info%free = .false.
        info%n_distinct = 1
        info%n_free_class = 0
        info%shared_violation = .false.
        if (n < 2) return
        do i = 1, N_FIELDS
            ! Is the field free? (the first model that differs from model 1 settles it)
            do m = 2, n
                if (.not. registry_equal(core(1), core(m), i)) then
                    info%free(i) = .true.
                    exit
                end if
            end do
            if (.not. info%free(i)) cycle
            info%n_free_class(field_class(i)) = info%n_free_class(field_class(i)) + 1
            if (field_class(i) == CLASS_S) info%shared_violation = .true.
            ! Distinct values: against the representatives found so far
            nd = 1
            reps(1) = 1
            do m = 2, n
                found = .false.
                do r = 1, nd
                    if (registry_equal(core(m), core(reps(r)), i)) then
                        found = .true.
                        exit
                    end if
                end do
                if (.not. found) then
                    if (nd == DISTINCT_CAP) exit
                    nd = nd + 1
                    reps(nd) = m
                end if
            end do
            info%n_distinct(i) = nd
        end do
    end subroutine batch_classify

    !> The path of a batch: the SSP source and its number of sources per model
    subroutine batch_info_set_path(info, ssp_source, k_sources)
        type(batch_info_t), intent(inout) :: info
        character(len=*), intent(in) :: ssp_source
        integer, intent(in) :: k_sources

        info%ssp_source = ssp_source
        info%k_sources = k_sources
    end subroutine batch_info_set_path

    !> One line of text: the models, the SSP source, and the free fields with their class and distinct values
    function batch_info_text(info) result(s)
        type(batch_info_t), intent(in) :: info
        character(len=:), allocatable :: s
        character(len=24) :: num
        integer :: i

        write(num, '(I0)') info%n_models
        s = 'opensps batch: ' // trim(num) // ' models, ssp ' // trim(info%ssp_source)
        write(num, '(I0)') info%k_sources
        s = s // ' (K=' // trim(num) // '), free:'
        do i = 1, N_FIELDS
            if (.not. info%free(i)) cycle
            write(num, '(I0)') info%n_distinct(i)
            s = s // ' ' // trim(field_names(i)) // '[' // trim(registry_class_name(field_class(i))) // ',' // trim(num)
            if (info%n_distinct(i) >= DISTINCT_CAP) s = s // '+'
            s = s // ']'
        end do
        if (info%shared_violation) s = s // ' (a shared field differs)'
        if (count(info%free) == 0) s = s // ' none'
    end function batch_info_text

    subroutine batch_info_print(info)
        type(batch_info_t), intent(in) :: info
        write(error_unit, '(A)') batch_info_text(info)
    end subroutine batch_info_print

    !> OPENSPS_DEBUG_PATH set (to anything but 0): compute_csp_batch prints the path of every batch to standard error.
    !> Read once.
    logical function batch_debug_path() result(on)
        logical, save :: read_env = .false., value = .false.
        character(len=8) :: v
        integer :: st

        if (.not. read_env) then
            call get_environment_variable('OPENSPS_DEBUG_PATH', v, status=st)
            value = st == 0 .and. len_trim(v) > 0 .and. trim(v) /= '0'
            read_env = .true.
        end if
        on = value
    end function batch_debug_path

end module opensps_batch_registry
