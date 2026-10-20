PROGRAM TEST_CACHE
  USE opensps_context_types, ONLY: opensps_context_t
  use opensps_api, only: opensps_create, opensps_setup, opensps_destroy
  IMPLICIT NONE

  TYPE(opensps_context_t) :: ctx1, ctx2, ctx3
  LOGICAL :: same_cache, diff_cache

  WRITE(*,*) '========================================='
  WRITE(*,*) 'FSPS CACHE REUSE TEST'
  WRITE(*,*) '========================================='

  call opensps_create(ctx1)
  call opensps_create(ctx2)
  call opensps_create(ctx3)

  call opensps_setup(ctx1, -1, 'mist', 'miles', 'DL07')
  call opensps_setup(ctx2, -1, 'mist', 'miles', 'DL07')
  call opensps_setup(ctx3, -1, 'mist', 'basel', 'DL07')

  same_cache = ASSOCIATED(ctx1%setup_cache, ctx2%setup_cache)
  diff_cache = .NOT.ASSOCIATED(ctx1%setup_cache, ctx3%setup_cache)

  IF (.NOT.(same_cache .AND. diff_cache)) THEN
     WRITE(*,*) 'Cache reuse test FAIL'
    call opensps_destroy(ctx1)
    call opensps_destroy(ctx2)
    call opensps_destroy(ctx3)
     STOP 1
  ENDIF

  call opensps_destroy(ctx1)
  call opensps_destroy(ctx2)
  call opensps_destroy(ctx3)

  WRITE(*,*) 'Cache reuse test PASS'
END PROGRAM TEST_CACHE
