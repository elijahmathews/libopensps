PROGRAM GENERATE_TEST_DATA

  ! Generates reference data for FSPS regression testing.
  ! Uses allocatable arrays to support multiple compile-time configurations.

   USE fsps_precision, ONLY: WP
   USE fsps_constants, ONLY: NEMLINE
   USE fsps_types, ONLY: PARAMS, COMPSPOUT
   use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
   USE fsps_csp, ONLY: compute_csp_scenario
   use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                           fsps_context_set_pset
   USE fsps_context_types, ONLY: fsps_context_t
   USE fsps_ssp, ONLY: generate_ssp_grid
  IMPLICIT NONE

  ! Variables for SSP generation (allocatable)
   REAL(WP), ALLOCATABLE, DIMENSION(:,:) :: spec_ssp
   REAL(WP), ALLOCATABLE, DIMENSION(:,:,:) :: spec_ssp3
   REAL(WP), ALLOCATABLE, DIMENSION(:)   :: mass_ssp, lbol_ssp
   REAL(WP), ALLOCATABLE, DIMENSION(:,:) :: mass_ssp2, lbol_ssp2
  
  ! Variables for CSP generation (allocatable)
   TYPE(COMPSPOUT), ALLOCATABLE, DIMENSION(:) :: results
  
   ! Control variables
   TYPE(fsps_context_t) :: ctx
  TYPE(PARAMS) :: pset
   INTEGER :: i, unit_out, status, arg_count
   INTEGER :: nspec_ctx, ntfull_ctx, nbands_ctx, nt_ctx, nindx_ctx
  CHARACTER(LEN=50) :: filename_out
  CHARACTER(LEN=255) :: arg_val
   CHARACTER(LEN=20) :: isoc_arg, spec_arg, dust_arg
  LOGICAL :: isoc_set, spec_set, dust_set

  ! Exit codes
  INTEGER, PARAMETER :: EXIT_FAILURE = 1

  ! Setup and initialization
  
  ! Name of the reference output file
  filename_out = 'sps_test_output.bin'
  unit_out = 40

   isoc_set = .FALSE.
   spec_set = .FALSE.
   dust_set = .FALSE.
   isoc_arg = 'mist'
   spec_arg = 'miles'
   dust_arg = 'DL07'

  WRITE(*,*) '--------------------------------------------------'
  WRITE(*,*) 'FSPS REFERENCE DATA GENERATOR'

  ! Parse command line arguments
  arg_count = COMMAND_ARGUMENT_COUNT()
  i = 1
  DO WHILE (i <= arg_count)
     CALL GET_COMMAND_ARGUMENT(i, arg_val, STATUS=status)
     IF (status /= 0) EXIT
     
     IF (TRIM(arg_val) == '--isoc') THEN
        i = i + 1
        IF (i <= arg_count) THEN
           CALL GET_COMMAND_ARGUMENT(i, isoc_arg, STATUS=status)
           isoc_set = .TRUE.
        ELSE
           WRITE(*,*) 'ERROR: --isoc requires an argument'
           STOP EXIT_FAILURE
        END IF
     ELSE IF (TRIM(arg_val) == '--spec') THEN
        i = i + 1
        IF (i <= arg_count) THEN
           CALL GET_COMMAND_ARGUMENT(i, spec_arg, STATUS=status)
           spec_set = .TRUE.
        ELSE
           WRITE(*,*) 'ERROR: --spec requires an argument'
           STOP EXIT_FAILURE
        END IF
     ELSE IF (TRIM(arg_val) == '--dust') THEN
        i = i + 1
        IF (i <= arg_count) THEN
           CALL GET_COMMAND_ARGUMENT(i, dust_arg, STATUS=status)
           dust_set = .TRUE.
        ELSE
           WRITE(*,*) 'ERROR: --dust requires an argument'
           STOP EXIT_FAILURE
        END IF
     END IF
     i = i + 1
  END DO
  
   ! Normalize empty args (treat as defaults)
   IF (LEN_TRIM(isoc_arg) == 0) isoc_arg = 'mist'
   IF (LEN_TRIM(spec_arg) == 0) spec_arg = 'miles'
   IF (LEN_TRIM(dust_arg) == 0) dust_arg = 'DL07'

   ! Initialize FSPS parameters (MIST/MILES defaults)
  ! We call this FIRST so we can be sure parameters like ntfull/nspec are set
  ! before we allocate (though they are static in the current codebase).
   call fsps_create(ctx)
   ctx%imf_type_val = 1          
  pset%zmet = 10        
  
  ! Use defaults unless overridden by arguments
  call fsps_setup(ctx, pset%zmet, isoc_type_in=TRIM(isoc_arg), &
     spec_type_in=TRIM(spec_arg), dust_type_in=TRIM(dust_arg))

  ! Allocate the persistent CSP/SSP workspaces (mirrors test_runner.f90)
  call fsps_context_move_to_device(ctx)

  ! Cache context dimensions for local allocations
  nspec_ctx = ctx%state%nspec
  ntfull_ctx = ctx%state%ntfull
  nbands_ctx = ctx%state%nbands
  nt_ctx = ctx%state%nt
  nindx_ctx = ctx%state%nindx

  IF (nspec_ctx <= 0 .OR. ntfull_ctx <= 0) THEN
   WRITE(*,*) 'ERROR: fsps_setup did not initialize dimensions.'
     WRITE(*,*) 'nspec=', nspec_ctx, ' ntfull=', ntfull_ctx
     STOP EXIT_FAILURE
  END IF

  ! Memory allocation
  WRITE(*,*) 'Allocating memory (nspec:', nspec_ctx, ' ntfull:', ntfull_ctx, ')...'
  ALLOCATE(spec_ssp(nspec_ctx,ntfull_ctx))
  ALLOCATE(mass_ssp(ntfull_ctx))
  ALLOCATE(lbol_ssp(ntfull_ctx))
   ! results is allocated by compute_csp_scenario
   ALLOCATE(spec_ssp3(nspec_ctx, ntfull_ctx, 1))
   ALLOCATE(mass_ssp2(ntfull_ctx, 1))
   ALLOCATE(lbol_ssp2(ntfull_ctx, 1))

  ! Write header
  WRITE(*,*) 'Writing to file: ', TRIM(filename_out)
  OPEN(UNIT=unit_out, FILE=TRIM(filename_out), STATUS='REPLACE', &
       FORM='UNFORMATTED', ACCESS='STREAM')

  WRITE(*,*) 'Writing Header...'
   WRITE(unit_out) nspec_ctx
   WRITE(unit_out) ntfull_ctx
   WRITE(unit_out) nbands_ctx

  ! Test Case 1: Simple SSP (Solar, Chabrier)
  WRITE(*,*) 'Running Test Case 1: SSP (Solar, Chabrier)...'

  ! Define SSP Parameters
  pset%sfh   = 0     ! SSP
  pset%const = 0.0
  pset%zred  = 0.0
  pset%dust1 = 0.0
  pset%dust2 = 0.0
   ctx%add_neb_emission_val = 1
  
  ! Allocate pset allocatable components
  IF (ALLOCATED(pset%mag_compute)) DEALLOCATE(pset%mag_compute)
   ALLOCATE(pset%mag_compute(nbands_ctx))
  pset%mag_compute = 1
  
  IF (ALLOCATED(pset%ssp_gen_age)) DEALLOCATE(pset%ssp_gen_age)
   ALLOCATE(pset%ssp_gen_age(nt_ctx))
  pset%ssp_gen_age = 1

  ! Compute SSP
   call fsps_context_set_pset(ctx, pset)
   CALL generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)

  ! Write SSP Data
  WRITE(*,*) 'Saving SSP results...'
  WRITE(unit_out) mass_ssp
  WRITE(unit_out) lbol_ssp
  WRITE(unit_out) spec_ssp

  ! Test Case 2: CSP (Tau Model, Dusty)
  WRITE(*,*) 'Running Test Case 2: CSP (Tau=2.0, Dust=1.0)...'

  pset%sfh   = 1     ! Tau model
  pset%tau   = 2.0   
  pset%dust1 = 1.0   
  pset%dust2 = 0.3
  
  ! Re-run generate_ssp_grid
   call fsps_context_set_pset(ctx, pset)
   CALL generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)

  ! Compute CSP
   mass_ssp2(:,1) = mass_ssp
   lbol_ssp2(:,1) = lbol_ssp
   spec_ssp3(:,:,1) = spec_ssp
   call fsps_context_update_ssp_basis(ctx, spec_ssp3, mass_ssp2, lbol_ssp2, 1)
   CALL compute_csp_scenario(ctx, pset, 1, results)

  ! Write CSP Data
  WRITE(*,*) 'Saving CSP results...'
   DO i = 1, ntfull_ctx
     WRITE(unit_out) results(i)%age
     WRITE(unit_out) results(i)%mass_csp
     WRITE(unit_out) results(i)%lbol_csp
     WRITE(unit_out) results(i)%sfr
     WRITE(unit_out) results(i)%mdust
     WRITE(unit_out) results(i)%mformed
     WRITE(unit_out) results(i)%mags      
     WRITE(unit_out) results(i)%spec      
     WRITE(unit_out) results(i)%indx      
     WRITE(unit_out) results(i)%emlines   
  END DO

  ! Cleanup
  CLOSE(unit_out)
   DEALLOCATE(spec_ssp, mass_ssp, lbol_ssp)
   IF (ALLOCATED(results)) DEALLOCATE(results)
  IF (ALLOCATED(pset%mag_compute)) DEALLOCATE(pset%mag_compute)
  IF (ALLOCATED(pset%ssp_gen_age)) DEALLOCATE(pset%ssp_gen_age)
  
   call fsps_destroy(ctx)

  WRITE(*,*) 'Complete. Data saved.'

END PROGRAM GENERATE_TEST_DATA
