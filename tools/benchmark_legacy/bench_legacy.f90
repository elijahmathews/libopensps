PROGRAM BENCH_LEGACY
  ! Timing benchmark for legacy FSPS, matched step for step by tests/benchmark_phases.f90 in the
  ! fork (same libraries, parameters and work per call). Usage: bench_legacy [n_ssp [n_csp]]
  ! Prints one line per metric: name, seconds.
  USE sps_vars
  USE sps_utils
  IMPLICIT NONE
  REAL(SP), DIMENSION(nspec,ntfull) :: spec_ssp
  REAL(SP), DIMENSION(nspec,ntfull,1) :: spec_ssp3
  REAL(SP), DIMENSION(ntfull) :: mass_ssp, lbol_ssp
  REAL(SP), DIMENSION(ntfull,1) :: mass_ssp2, lbol_ssp2
  TYPE(PARAMS) :: pset
  TYPE(COMPSPOUT), DIMENSION(ntfull) :: ocompsp
  CHARACTER(100) :: outfile = ''
  CHARACTER(32) :: arg
  INTEGER(8) :: t0, t1, rate
  INTEGER :: i, n_ssp = 5, n_csp = 200
  REAL(8) :: t_setup, t_first, total

  CALL GET_COMMAND_ARGUMENT(1, arg); IF (LEN_TRIM(arg) > 0) READ(arg,*) n_ssp
  CALL GET_COMMAND_ARGUMENT(2, arg); IF (LEN_TRIM(arg) > 0) READ(arg,*) n_csp
  CALL SYSTEM_CLOCK(t0, rate)

  ! Common parameters: Chabrier, solar MIST, tau model observed at 10 Gyr, dust + dust emission
  imf_type = 1
  pset%zmet = 10
  pset%sfh = 1
  pset%tau = 1.0
  pset%tage = 10.0
  pset%dust2 = 0.3

  ! 1. Time to first spectrum: setup + SSP grid + one CSP
  CALL SPS_SETUP(pset%zmet)
  CALL SYSTEM_CLOCK(t1); t_setup = REAL(t1 - t0, 8) / rate
  CALL SSP_GEN(pset, mass_ssp, lbol_ssp, spec_ssp)
  CALL basis()
  CALL COMPSP(0, 1, outfile, mass_ssp2, lbol_ssp2, spec_ssp3, pset, ocompsp)
  CALL SYSTEM_CLOCK(t1); t_first = REAL(t1 - t0, 8) / rate
  WRITE(*,'(A,F12.6)') 'setup_s ', t_setup
  WRITE(*,'(A,F12.6)') 'first_spectrum_s ', t_first

  ! 2. Per SSP call (full SSP grid regeneration)
  CALL SYSTEM_CLOCK(t0)
  DO i = 1, n_ssp
     CALL SSP_GEN(pset, mass_ssp, lbol_ssp, spec_ssp)
  END DO
  CALL SYSTEM_CLOCK(t1)
  WRITE(*,'(A,F12.6)') 'ssp_call_s ', REAL(t1 - t0, 8) / rate / n_ssp
  CALL basis()

  ! 3. Per CSP call (SSP basis cached; tau and dust vary)
  total = 0.0
  CALL SYSTEM_CLOCK(t0)
  DO i = 1, n_csp
     pset%tau = 0.5 + 0.01 * MOD(i, 100)
     pset%dust2 = 0.1 + 0.002 * MOD(i, 100)
     CALL COMPSP(0, 1, outfile, mass_ssp2, lbol_ssp2, spec_ssp3, pset, ocompsp)
     total = total + ocompsp(1)%spec(1000)
  END DO
  CALL SYSTEM_CLOCK(t1)
  WRITE(*,'(A,F12.6)') 'csp_call_s ', REAL(t1 - t0, 8) / rate / n_csp

  ! 4. Nebular emission: per SSP call and per CSP call (gas parameters vary)
  add_neb_emission = 1
  CALL SYSTEM_CLOCK(t0)
  DO i = 1, n_ssp
     CALL SSP_GEN(pset, mass_ssp, lbol_ssp, spec_ssp)
  END DO
  CALL SYSTEM_CLOCK(t1)
  WRITE(*,'(A,F12.6)') 'ssp_call_nebular_s ', REAL(t1 - t0, 8) / rate / n_ssp
  CALL basis()
  CALL SYSTEM_CLOCK(t0)
  DO i = 1, n_csp
     pset%tau = 0.5 + 0.01 * MOD(i, 100)
     pset%gas_logu = -3.0 + 0.01 * MOD(i, 100)
     CALL COMPSP(0, 1, outfile, mass_ssp2, lbol_ssp2, spec_ssp3, pset, ocompsp)
     total = total + ocompsp(1)%spec(1000)
  END DO
  CALL SYSTEM_CLOCK(t1)
  WRITE(*,'(A,F12.6)') 'csp_call_nebular_s ', REAL(t1 - t0, 8) / rate / n_csp
  IF (total < 0.0) WRITE(*,*) total   ! keep the results live

CONTAINS

  SUBROUTINE basis()
    spec_ssp3(:,:,1) = spec_ssp
    mass_ssp2(:,1) = mass_ssp
    lbol_ssp2(:,1) = lbol_ssp
  END SUBROUTINE basis

END PROGRAM BENCH_LEGACY
