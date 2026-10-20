PROGRAM TEST_CACHE
  USE fsps_context_types, ONLY: fsps_context_t
  use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
  IMPLICIT NONE

  TYPE(fsps_context_t) :: ctx1, ctx2, ctx3
  LOGICAL :: same_cache, diff_cache

  WRITE(*,*) '========================================='
  WRITE(*,*) 'FSPS CACHE REUSE TEST'
  WRITE(*,*) '========================================='

  call fsps_create(ctx1)
  call fsps_create(ctx2)
  call fsps_create(ctx3)

  call fsps_setup(ctx1, -1, 'mist', 'miles', 'DL07')
  call fsps_setup(ctx2, -1, 'mist', 'miles', 'DL07')
  call fsps_setup(ctx3, -1, 'mist', 'basel', 'DL07')

  same_cache = ASSOCIATED(ctx1%setup_cache, ctx2%setup_cache)
  diff_cache = .NOT.ASSOCIATED(ctx1%setup_cache, ctx3%setup_cache)

  IF (.NOT.(same_cache .AND. diff_cache)) THEN
     WRITE(*,*) 'Cache reuse test FAIL'
    call fsps_destroy(ctx1)
    call fsps_destroy(ctx2)
    call fsps_destroy(ctx3)
     STOP 1
  ENDIF

  call fsps_destroy(ctx1)
  call fsps_destroy(ctx2)
  call fsps_destroy(ctx3)

  WRITE(*,*) 'Cache reuse test PASS'
END PROGRAM TEST_CACHE
