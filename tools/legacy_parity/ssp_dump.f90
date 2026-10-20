PROGRAM SSP_DUMP
  ! Legacy FSPS driver for the parity comparison: dumps a solar-metallicity SSP grid.
  ! Usage: ssp_dump [imf_type [add_agb_dust_model]]   (defaults: 1 = Chabrier, legacy default)
  ! Writes legacy_ssp.bin: int32 ntfull, nspec, sp; then float64 time_full, spec_lambda,
  ! mass_ssp, lbol_ssp and spec_ssp(nspec, ntfull).
  USE sps_vars
  USE sps_utils
  IMPLICIT NONE
  REAL(SP), DIMENSION(nspec,ntfull) :: spec_ssp
  REAL(SP), DIMENSION(ntfull) :: mass_ssp, lbol_ssp
  TYPE(PARAMS) :: pset
  CHARACTER(256) :: arg
  imf_type = 1
  pset%zmet = 10
  CALL GETARG(1, arg)
  IF (LEN_TRIM(arg) > 0) READ(arg,*) imf_type
  CALL GETARG(2, arg)
  IF (LEN_TRIM(arg) > 0) READ(arg,*) add_agb_dust_model
  CALL SPS_SETUP(pset%zmet)
  pset%sfh = 0
  CALL SSP_GEN(pset, mass_ssp, lbol_ssp, spec_ssp)
  OPEN(10, FILE='legacy_ssp.bin', ACCESS='stream', FORM='unformatted', STATUS='replace')
  WRITE(10) ntfull, nspec, sp
  WRITE(10) REAL(time_full,8), REAL(spec_lambda,8), REAL(mass_ssp,8), REAL(lbol_ssp,8)
  WRITE(10) REAL(spec_ssp,8)
  CLOSE(10)
END PROGRAM SSP_DUMP
