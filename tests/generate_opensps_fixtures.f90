PROGRAM GENERATE_OPENSPS_FIXTURES

  ! Generates golden reference fixtures for the libopensps parity suite.
  !
  ! Usage:
  !   generate_opensps_fixtures --isoc mist --spec miles [--dust DL07] --out case.opsf key=value ...
  !
  ! Each key=value pair is applied through fsps_context_set_param_int/float (the same
  ! keys accepted by the C API), both before fsps_setup (setup-time flags) and after it.
  !
  ! Output format ("OPSF" container, little-endian stream):
  !   char(4)  magic = 'OPSF'
  !   int32    version = 1
  !   repeated until EOF:
  !     int32 name_len, char(name_len) name
  !     int32 ndim, int32 dims(ndim)          (Fortran order: first dimension fastest)
  !     real64 data(product(dims))

   use iso_fortran_env, only: int32, real64
   use fsps_precision, only: WP
   use fsps_constants, only: NEMLINE
   use fsps_types, only: PARAMS, COMPSPOUT
   use fsps_api, only: fsps_create, fsps_setup, fsps_destroy
   use fsps_csp, only: compute_csp_scenario
   use fsps_csp_batch, only: compute_csp_batch
   use fsps_ssp, only: generate_ssp_grid, configure_imf_parameters
   use fsps_imf, only: compute_imf_weights
   use fsps_context, only: fsps_context_update_ssp_basis, fsps_context_move_to_device, &
                           fsps_context_set_pset, fsps_context_get_pset, fsps_context_prepare_pset, &
                           fsps_context_set_param_int, fsps_context_set_param_float
   use fsps_context_types, only: fsps_context_t
   implicit none

   integer, parameter :: MAX_OVERRIDES = 64
   integer, parameter :: EXIT_FAILURE = 1

   type(fsps_context_t) :: ctx
   type(PARAMS) :: pset
   type(COMPSPOUT), allocatable :: results(:)

   character(len=256) :: isoc_arg = 'mist', spec_arg = 'miles', dust_arg = 'DL07'
   character(len=1024) :: out_arg = 'case.opsf', arg_val
   character(len=256) :: overrides(MAX_OVERRIDES)
   integer :: n_overrides = 0
   logical :: csp_only = .false.
   !> --batch: single-age CSPs (tage > 0) through compute_csp_batch, on N_BATCH copies of the
   !> model (enough for the many-spectra kernels and the cuBLAS products); the first is written
   logical :: use_batch = .false.
   integer, parameter :: N_BATCH = 32
   type(PARAMS), allocatable :: psets_b(:)
   real(WP), allocatable :: mags_b(:,:), spec_b(:,:), indx_b(:,:), eml_b(:,:)
   real(WP) :: mass_b(N_BATCH), lbol_b(N_BATCH), mdust_b(N_BATCH), sfr_b(N_BATCH)

   real(WP), allocatable :: spec_ssp(:,:), mass_ssp(:), lbol_ssp(:)
   real(WP), allocatable :: spec_ssp3(:,:,:), mass_ssp2(:,:), lbol_ssp2(:,:)
   real(WP), allocatable :: mini(:), weights(:), tmp2(:,:)
   integer :: i, j, k, status, nspec, ntfull, nt, nbands, nindx, n_out, unit_out, zmet, nm
   integer :: t_sel(3)
   character(len=64) :: label

   ! ------------------------------------------------------------------
   ! 1. Parse command line
   ! ------------------------------------------------------------------
   i = 1
   do while (i <= command_argument_count())
      call get_command_argument(i, arg_val)
      select case (trim(arg_val))
      case ('--isoc')
         i = i + 1; call get_command_argument(i, isoc_arg)
      case ('--spec')
         i = i + 1; call get_command_argument(i, spec_arg)
      case ('--dust')
         i = i + 1; call get_command_argument(i, dust_arg)
      case ('--out')
         i = i + 1; call get_command_argument(i, out_arg)
      case ('--csp-only')
         csp_only = .true.
      case ('--batch')
         use_batch = .true.
      case default
         if (index(arg_val, '=') == 0) then
            write(*,*) 'ERROR: unrecognised argument ', trim(arg_val)
            stop EXIT_FAILURE
         end if
         if (n_overrides >= MAX_OVERRIDES) then
            write(*,*) 'ERROR: too many key=value overrides'
            stop EXIT_FAILURE
         end if
         n_overrides = n_overrides + 1
         overrides(n_overrides) = trim(arg_val)
      end select
      i = i + 1
   end do

   ! ------------------------------------------------------------------
   ! 2. Context creation and setup (mirrors tests/test_runner.f90)
   ! ------------------------------------------------------------------
   call fsps_create(ctx)
   ctx%pset%zmet = 10
   call apply_overrides()

   call fsps_setup(ctx, ctx%pset%zmet, isoc_type_in=trim(isoc_arg), &
                   spec_type_in=trim(spec_arg), dust_type_in=trim(dust_arg))
   call fsps_context_move_to_device(ctx)

   ! Re-apply after setup so that values reset during setup are restored
   call apply_overrides()
   call fsps_context_prepare_pset(ctx)
   call fsps_context_get_pset(ctx, pset)

   nspec  = ctx%state%nspec
   ntfull = ctx%state%ntfull
   nt     = ctx%state%nt
   nbands = ctx%state%nbands
   nindx  = ctx%state%nindx
   zmet   = pset%zmet

   open(newunit=unit_out, file=trim(out_arg), status='replace', form='unformatted', access='stream')
   write(unit_out) 'OPSF'
   write(unit_out) int(1, int32)

   call put('wavelengths', real(ctx%state%spec_lambda(1:nspec), real64), [nspec])
   call put('time_full', real(ctx%state%time_full(1:ntfull), real64), [ntfull])
   call put('meta/zmet', [real(zmet, real64)], [1])

   ! ------------------------------------------------------------------
   ! 3. IMF weights on the native isochrone mass grid at three ages
   ! ------------------------------------------------------------------
   call configure_imf_parameters(ctx, pset)
   t_sel = [1, max(1, nt / 2), nt]
   do j = 1, 3
      nm = ctx%state%nmass_isoc(zmet, t_sel(j))
      if (nm <= 0) cycle
      allocate(mini(nm), weights(nm))
      mini = ctx%state%mini_isoc(zmet, t_sel(j), 1:nm)
      call compute_imf_weights(ctx, mini, weights, nm)
      write(label, '(A,I0)') 'imf/t', t_sel(j)
      call put(trim(label)//'/mini', real(mini, real64), [nm])
      call put(trim(label)//'/weights', real(weights, real64), [nm])
      deallocate(mini, weights)
   end do

   ! ------------------------------------------------------------------
   ! 4. SSP grid
   ! ------------------------------------------------------------------
   allocate(spec_ssp(nspec, ntfull), mass_ssp(ntfull), lbol_ssp(ntfull))
   call fsps_context_set_pset(ctx, pset)
   call generate_ssp_grid(ctx, pset, mass_ssp, lbol_ssp, spec_ssp)

   call put('ssp/mass', real(mass_ssp, real64), [ntfull])
   call put('ssp/lbol', real(lbol_ssp, real64), [ntfull])
   if (.not. csp_only) then
      call put('ssp/spec', real(reshape(spec_ssp, [nspec * ntfull]), real64), [nspec, ntfull])
   end if

   ! ------------------------------------------------------------------
   ! 5. CSP built on the SSP basis above
   ! ------------------------------------------------------------------
   allocate(spec_ssp3(nspec, ntfull, 1), mass_ssp2(ntfull, 1), lbol_ssp2(ntfull, 1))
   spec_ssp3(:, :, 1) = spec_ssp
   mass_ssp2(:, 1) = mass_ssp
   lbol_ssp2(:, 1) = lbol_ssp
   call fsps_context_update_ssp_basis(ctx, spec_ssp3, mass_ssp2, lbol_ssp2, 1)
   call compute_csp_scenario(ctx, pset, 1, results, status)
   if (status /= 0) then
      write(*,*) 'ERROR: compute_csp_scenario returned status ', status
      stop EXIT_FAILURE
   end if

   if (use_batch .and. pset%tage > 0.0_wp) then
      allocate(psets_b(N_BATCH), mags_b(nbands, N_BATCH), spec_b(nspec, N_BATCH), indx_b(nindx, N_BATCH), &
               eml_b(NEMLINE, N_BATCH))
      psets_b = pset
      call compute_csp_batch(ctx, psets_b, mags_b, status, spec=spec_b, mass=mass_b, lbol=lbol_b, mdust=mdust_b, &
                             sfr=sfr_b, emlines=eml_b, indx=indx_b)
      if (status == 0) then
         results(1)%spec(1:nspec) = spec_b(:, 1)
         results(1)%mags(1:nbands) = mags_b(:, 1)
         if (pset%compute_indices == 1) results(1)%indx(1:nindx) = indx_b(:, 1)
         results(1)%emlines(1:NEMLINE) = eml_b(:, 1)
         results(1)%mass_csp = mass_b(1)
         results(1)%lbol_csp = lbol_b(1)
         results(1)%mdust = mdust_b(1)
         results(1)%sfr = sfr_b(1)
         write(*,*) 'batch: used'
      else
         write(*,*) 'batch: not supported for this case, status ', status
      end if
   end if

   n_out = size(results)
   call put('csp/age', real([(results(k)%age, k = 1, n_out)], real64), [n_out])
   call put('csp/mass', real([(results(k)%mass_csp, k = 1, n_out)], real64), [n_out])
   call put('csp/lbol', real([(results(k)%lbol_csp, k = 1, n_out)], real64), [n_out])
   call put('csp/sfr', real([(results(k)%sfr, k = 1, n_out)], real64), [n_out])
   call put('csp/mdust', real([(results(k)%mdust, k = 1, n_out)], real64), [n_out])
   call put('csp/mformed', real([(results(k)%mformed, k = 1, n_out)], real64), [n_out])

   allocate(tmp2(nspec, n_out))
   do k = 1, n_out
      tmp2(:, k) = results(k)%spec(1:nspec)
   end do
   call put('csp/spec', real(reshape(tmp2, [nspec * n_out]), real64), [nspec, n_out])
   deallocate(tmp2)

   allocate(tmp2(nbands, n_out))
   do k = 1, n_out
      tmp2(:, k) = results(k)%mags(1:nbands)
   end do
   call put('csp/mags', real(reshape(tmp2, [nbands * n_out]), real64), [nbands, n_out])
   deallocate(tmp2)

   allocate(tmp2(nindx, n_out))
   do k = 1, n_out
      tmp2(:, k) = results(k)%indx(1:nindx)
   end do
   call put('csp/indx', real(reshape(tmp2, [nindx * n_out]), real64), [nindx, n_out])
   deallocate(tmp2)

   allocate(tmp2(NEMLINE, n_out))
   do k = 1, n_out
      tmp2(:, k) = results(k)%emlines(1:NEMLINE)
   end do
   call put('csp/emlines', real(reshape(tmp2, [NEMLINE * n_out]), real64), [NEMLINE, n_out])
   deallocate(tmp2)

   ! SFH weights over SSP ages (first metallicity slot) for each output age
   if (allocated(ctx%state%csp_weights)) then
      allocate(tmp2(ntfull, n_out))
      do k = 1, n_out
         tmp2(:, k) = ctx%state%csp_weights(1:ntfull, 1, k)
      end do
      call put('csp/sfh_weights', real(reshape(tmp2, [ntfull * n_out]), real64), [ntfull, n_out])
      deallocate(tmp2)
   end if

   close(unit_out)
   call fsps_destroy(ctx)
   write(*,*) 'Wrote ', trim(out_arg)

contains

   !> Writes one named array record to the output stream.
   subroutine put(name, values, dims)
      character(len=*), intent(in) :: name
      real(real64), intent(in) :: values(:)
      integer, intent(in) :: dims(:)
      write(unit_out) int(len_trim(name), int32), trim(name)
      write(unit_out) int(size(dims), int32), int(dims, int32)
      write(unit_out) values
   end subroutine put

   !> Applies every key=value override, trying the integer setter before the float one.
   subroutine apply_overrides()
      integer :: n, eq, ival, ios, st
      real(WP) :: rval
      character(len=64) :: key
      character(len=192) :: valstr

      do n = 1, n_overrides
         eq = index(overrides(n), '=')
         key = overrides(n)(1:eq - 1)
         valstr = overrides(n)(eq + 1:)
         st = 1
         ! Only treat the value as an integer when it contains no decimal point or exponent
         if (scan(valstr, '.eE') == 0) then
            read(valstr, *, iostat=ios) ival
            if (ios == 0) call fsps_context_set_param_int(ctx, trim(key), ival, st)
         end if
         if (st /= 0) then
            read(valstr, *, iostat=ios) rval
            if (ios /= 0) then
               write(*,*) 'ERROR: cannot parse value for ', trim(key)
               stop EXIT_FAILURE
            end if
            call fsps_context_set_param_float(ctx, trim(key), rval, st)
         end if
         if (st /= 0) then
            write(*,*) 'ERROR: unknown parameter ', trim(key)
            stop EXIT_FAILURE
         end if
      end do
   end subroutine apply_overrides

END PROGRAM GENERATE_OPENSPS_FIXTURES
