! Legacy FSPS (v4, built with -DAFE_FLAG=1: MIST v2.5 + C3K_LR) SSP dump at the five [alpha/Fe]
! nodes of one metallicity index, all phases or a single evolutionary phase.
!
!   legacy_dump ZMET [EVTYPE [ADD_AGB_DUST]]
!
! Writes ssp_zZZ_evE.bin (stream: ntfull, nspec as int32, then spec(nspec,ntfull,nafe) as float64).
! The isochrone mass and L_bol are not written: they match this fork to 2e-6 (IMF integrals).
PROGRAM LEGACY_DUMP
  USE sps_vars
  USE sps_utils
  IMPLICIT NONE
  REAL(SP), DIMENSION(nspec,ntfull) :: spec_ssp
  REAL(SP), DIMENSION(nspec,ntfull,nafe) :: spec_all
  REAL(SP), DIMENSION(ntfull) :: mass_ssp, lbol_ssp
  CHARACTER(100) :: fn, arg
  TYPE(PARAMS) :: pset
  INTEGER :: ia, zmet, ev
  CALL GET_COMMAND_ARGUMENT(1, arg); READ(arg,*) zmet
  ev = -1
  IF (COMMAND_ARGUMENT_COUNT().GE.2) THEN; CALL GET_COMMAND_ARGUMENT(2, arg); READ(arg,*) ev; ENDIF
  IF (COMMAND_ARGUMENT_COUNT().GE.3) THEN; CALL GET_COMMAND_ARGUMENT(3, arg); READ(arg,*) add_agb_dust_model; ENDIF
  imf_type = 1
  pset%zmet = zmet
  pset%evtype = ev
  CALL SPS_SETUP(pset%zmet)
  DO ia = 1, nafe
     pset%afeindx = ia
     CALL SSP_GEN(pset, mass_ssp, lbol_ssp, spec_ssp)
     spec_all(:,:,ia) = spec_ssp
  END DO
  WRITE(fn,'(A,I2.2,A,I0,A)') 'legacy_z', zmet, '_ev', ev, '.bin'
  OPEN(12, FILE=TRIM(fn), FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE')
  WRITE(12) INT(ntfull), INT(nspec)
  WRITE(12) spec_all
  CLOSE(12)
END PROGRAM LEGACY_DUMP
