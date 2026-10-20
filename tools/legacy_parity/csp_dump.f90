PROGRAM CSP_DUMP
  ! Legacy FSPS driver for the parity matrix: computes the SSP grid and the CSP for one case and
  ! writes them in the OPSF format of tests/generate_opensps_fixtures.f90 (same array names), so
  ! that legacy and fork outputs can be compared case by case.
  !
  ! Usage: csp_dump --out case.opsf [--csp-only] key=value ...
  ! Keys are the parameter names of tests/data/param_cases.psv; each is applied to the legacy
  ! PARAMS structure or to the sps_vars global of the same name. The library selection is fixed
  ! at compile time (-DMIST=1 -DMILES=1 etc.).

  USE sps_vars
  USE sps_utils
  IMPLICIT NONE

  INTEGER, PARAMETER :: MAXARG = 64
  REAL(SP), DIMENSION(nspec,ntfull) :: spec_ssp
  REAL(SP), DIMENSION(nspec,ntfull,1) :: spec_ssp3
  REAL(SP), DIMENSION(ntfull) :: mass_ssp, lbol_ssp
  REAL(SP), DIMENSION(ntfull,1) :: mass_ssp2, lbol_ssp2
  TYPE(PARAMS) :: pset
  TYPE(COMPSPOUT), DIMENSION(ntfull) :: ocompsp
  CHARACTER(256) :: arg, outfile = 'case.opsf'
  CHARACTER(100) :: dummy_out = ''
  CHARACTER(256), DIMENSION(MAXARG) :: overrides
  INTEGER :: i, k, n_over = 0, n_out, u
  LOGICAL :: csp_only = .FALSE.

  ! ------------------------------------------------------------------
  ! Arguments
  ! ------------------------------------------------------------------
  i = 1
  DO WHILE (i <= COMMAND_ARGUMENT_COUNT())
     CALL GET_COMMAND_ARGUMENT(i, arg)
     IF (TRIM(arg) == '--out') THEN
        i = i + 1
        CALL GET_COMMAND_ARGUMENT(i, outfile)
     ELSE IF (TRIM(arg) == '--csp-only') THEN
        csp_only = .TRUE.
     ELSE IF (INDEX(arg, '=') > 0) THEN
        n_over = n_over + 1
        overrides(n_over) = arg
     ELSE
        WRITE(*,*) 'csp_dump: unknown argument ', TRIM(arg)
        STOP 1
     END IF
     i = i + 1
  END DO

  ! Same starting point as generate_opensps_fixtures: solar metallicity index 10 unless overridden
  pset%zmet = 10
  DO k = 1, n_over
     CALL apply(TRIM(overrides(k)))
  END DO

  CALL SPS_SETUP(pset%zmet)
  CALL SSP_GEN(pset, mass_ssp, lbol_ssp, spec_ssp)

  spec_ssp3(:,:,1) = spec_ssp
  mass_ssp2(:,1) = mass_ssp
  lbol_ssp2(:,1) = lbol_ssp
  CALL COMPSP(0, 1, dummy_out, mass_ssp2, lbol_ssp2, spec_ssp3, pset, ocompsp)

  IF (pset%tage > 0.0) THEN
     n_out = 1
  ELSE
     n_out = ntfull
  END IF

  ! Legacy COMPSP fills the indices only when writing its .indx file (write_compsp = 4);
  ! compute them from each output spectrum with the same routine instead
  DO k = 1, n_out
     CALL GETINDX(spec_lambda, ocompsp(k)%spec, ocompsp(k)%indx)
  END DO

  ! ------------------------------------------------------------------
  ! Output (OPSF: 'OPSF', int32 version, then name/dims/float64 records)
  ! ------------------------------------------------------------------
  OPEN(NEWUNIT=u, FILE=TRIM(outfile), STATUS='REPLACE', FORM='UNFORMATTED', ACCESS='STREAM')
  WRITE(u) 'OPSF'
  WRITE(u) INT(1, 4)
  CALL put('wavelengths', REAL(spec_lambda, 8), [nspec])
  CALL put('time_full', REAL(time_full, 8), [ntfull])
  CALL put('meta/zmet', [REAL(pset%zmet, 8)], [1])
  CALL put('ssp/mass', REAL(mass_ssp, 8), [ntfull])
  CALL put('ssp/lbol', REAL(lbol_ssp, 8), [ntfull])
  IF (.NOT. csp_only) CALL put('ssp/spec', REAL(RESHAPE(spec_ssp, [nspec*ntfull]), 8), [nspec, ntfull])
  CALL put('csp/age', [(REAL(ocompsp(k)%age, 8), k = 1, n_out)], [n_out])
  CALL put('csp/mass', [(REAL(ocompsp(k)%mass_csp, 8), k = 1, n_out)], [n_out])
  CALL put('csp/lbol', [(REAL(ocompsp(k)%lbol_csp, 8), k = 1, n_out)], [n_out])
  CALL put('csp/sfr', [(REAL(ocompsp(k)%sfr, 8), k = 1, n_out)], [n_out])
  CALL put('csp/mdust', [(REAL(ocompsp(k)%mdust, 8), k = 1, n_out)], [n_out])
  CALL put('csp/mformed', [(REAL(ocompsp(k)%mformed, 8), k = 1, n_out)], [n_out])
  CALL put('csp/spec', [(REAL(ocompsp(k)%spec, 8), k = 1, n_out)], [nspec, n_out])
  CALL put('csp/mags', [(REAL(ocompsp(k)%mags, 8), k = 1, n_out)], [nbands, n_out])
  CALL put('csp/indx', [(REAL(ocompsp(k)%indx, 8), k = 1, n_out)], [nindx, n_out])
  CALL put('csp/emlines', [(REAL(ocompsp(k)%emlines, 8), k = 1, n_out)], [nemline, n_out])
  CLOSE(u)

CONTAINS

  SUBROUTINE put(name, values, dims)
    CHARACTER(*), INTENT(in) :: name
    REAL(8), INTENT(in) :: values(:)
    INTEGER, INTENT(in) :: dims(:)
    WRITE(u) INT(LEN_TRIM(name), 4), TRIM(name)
    WRITE(u) INT(SIZE(dims), 4), INT(dims, 4)
    WRITE(u) values
  END SUBROUTINE put

  !> Applies one key=value to the PARAMS structure or to the sps_vars global of the same name.
  SUBROUTINE apply(kv)
    CHARACTER(*), INTENT(in) :: kv
    CHARACTER(64) :: key
    CHARACTER(128) :: val
    REAL(SP) :: r
    INTEGER :: n, eq

    eq = INDEX(kv, '=')
    key = kv(1:eq-1)
    val = kv(eq+1:)
    READ(val, *) r
    n = NINT(r)

    SELECT CASE (TRIM(key))
    ! sps_vars globals
    CASE ('imf_type');             imf_type = n
    CASE ('add_agb_dust_model');   add_agb_dust_model = n
    CASE ('add_agn_dust');         add_agn_dust = n
    CASE ('add_dust_emission');    add_dust_emission = n
    CASE ('add_igm_absorption');   add_igm_absorption = n
    CASE ('add_neb_emission');     add_neb_emission = n
    CASE ('add_stellar_remnants'); add_stellar_remnants = n
    CASE ('smooth_velocity');      smooth_velocity = n
    CASE ('use_wr_spectra');       use_wr_spectra = n
    CASE ('dust_type');            dust_type = n
    ! PARAMS integers
    CASE ('zmet');                 pset%zmet = n
    CASE ('sfh');                  pset%sfh = n
    ! PARAMS reals
    CASE ('agb');          pset%agb = r
    CASE ('agb_dust');     pset%agb_dust = r
    CASE ('agn_tau');      pset%agn_tau = r
    CASE ('const');        pset%const = r
    CASE ('dell');         pset%dell = r
    CASE ('delt');         pset%delt = r
    CASE ('dust1');        pset%dust1 = r
    CASE ('dust1_index');  pset%dust1_index = r
    CASE ('dust2');        pset%dust2 = r
    CASE ('dust3');        pset%dust3 = r
    CASE ('duste_gamma');  pset%duste_gamma = r
    CASE ('duste_qpah');   pset%duste_qpah = r
    CASE ('duste_umin');   pset%duste_umin = r
    CASE ('dust_index');   pset%dust_index = r
    CASE ('dust_tesc');    pset%dust_tesc = r
    CASE ('fagn');         pset%fagn = r
    CASE ('fbhb');         pset%fbhb = r
    CASE ('fburst');       pset%fburst = r
    CASE ('frac_nodust');  pset%frac_nodust = r
    CASE ('frac_obrun');   pset%frac_obrun = r
    CASE ('gas_logu');     pset%gas_logu = r
    CASE ('gas_logz');     pset%gas_logz = r
    CASE ('mdave');        pset%mdave = r
    CASE ('mwr');          pset%mwr = r
    CASE ('pagb');         pset%pagb = r
    CASE ('redgb');        pset%redgb = r
    CASE ('sbss');         pset%sbss = r
    CASE ('sf_slope');     pset%sf_slope = r
    CASE ('sf_start');     pset%sf_start = r
    CASE ('sf_trunc');     pset%sf_trunc = r
    CASE ('sigma_smooth'); pset%sigma_smooth = r
    CASE ('tage');         pset%tage = r
    CASE ('tau');          pset%tau = r
    CASE ('tburst');       pset%tburst = r
    CASE ('uvb');          pset%uvb = r
    CASE ('vdmc');         pset%vdmc = r
    CASE ('zred');         pset%zred = r
    CASE DEFAULT
       WRITE(*,*) 'csp_dump: unknown parameter ', TRIM(key)
       STOP 1
    END SELECT
  END SUBROUTINE apply

END PROGRAM CSP_DUMP
