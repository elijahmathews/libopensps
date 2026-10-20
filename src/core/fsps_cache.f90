module fsps_cache
    !> @brief
    !> Shared setup cache for FSPS contexts.
    !>
    !> @details
    !> Manages shared, read-only arrays (spectral libraries, filters, etc.) that
    !> can be reused across contexts to reduce I/O and allocation costs.

    use fsps_precision, only: WP, KP
    use fsps_constants, only: ndim_logt, ndim_logg, ndim_wmb_logt, ndim_wmb_logg, &
                              n_agb_o, n_agb_c, n_agb_car, ndim_pagb, ndim_wr, ntau_dagb, nteff_dagb, &
                              nemline, nebnz, nebnage, nebnip, nagndust, nm

    implicit none
    private

    public :: fsps_setup_cache_t
    public :: fsps_setup_tables_t
    public :: fsps_setup_cache_ptr_t
    public :: fsps_cache_get_setup
    public :: fsps_cache_release_setup
    public :: fsps_setup_lock
    public :: fsps_setup_unlock

    ! Process-wide recursive lock serializing setup and release of shared cache entries
    ! (implemented with a pthread mutex in fsps_setup_lock.c)
    interface
        subroutine fsps_setup_lock() bind(C, name='fsps_setup_lock')
        end subroutine fsps_setup_lock
        subroutine fsps_setup_unlock() bind(C, name='fsps_setup_unlock')
        end subroutine fsps_setup_unlock
    end interface

    ! ---------------------------------------------------------------------
    ! Module constants
    ! ---------------------------------------------------------------------

    !> @brief Small grid axes and tables the setup loaders write into fixed-size (per-context)
    !> members of the context state alongside the shared arrays.
    !>
    !> @details
    !> They are recorded here by the context that populates a cache entry and copied into every
    !> later context that hits the entry, so a cache hit needs no file I/O.
    type :: fsps_setup_tables_t
        ! No default initialization: save_setup_tables writes every component, and initializing
        ! the 2 MB type makes gfortran build a stack temporary on allocation
        real(WP) :: speclib_logt(ndim_logt)
        real(WP) :: speclib_logg(ndim_logg)
        real(WP) :: wmb_logt(ndim_wmb_logt)
        real(WP) :: wmb_logg(ndim_wmb_logg)
        real(WP) :: agb_logt_car(n_agb_car)
        real(WP) :: pagb_logt(ndim_pagb)
        real(WP) :: wrn_logt(ndim_wr)
        real(WP) :: wrc_logt(ndim_wr)
        real(WP) :: tau1_dagb(2, ntau_dagb)
        real(WP) :: teff_dagb(2, nteff_dagb)
        real(WP) :: nebem_line_pos(nemline)
        real(WP) :: nebem_line(nemline, nebnz, nebnage, nebnip)
        real(WP) :: xnebem_line(nemline, nebnz, nebnage, nebnip)
        real(WP) :: nebem_logz(nebnz)
        real(WP) :: nebem_age(nebnage)
        real(WP) :: nebem_logu(nebnip)
        real(WP) :: agndust_tau(nagndust)
    end type fsps_setup_tables_t

    type :: fsps_setup_cache_t
        character(len=512) :: key = ''
        integer :: refcount = 0
        integer :: nz = 0
        integer :: nt = 0
        integer :: nspec = 0
        integer :: nzinit = 0
        integer :: nbands = 0
        integer :: nindx = 0
        integer :: ntfull = 0
        integer :: nspec_xrb = 0
        integer :: nt_xrb = 0
        integer :: nz_xrb = 0
        character(len=30) :: alt_filter_file = ''
        character(len=64) :: isoc_type = ''
        character(len=64) :: spec_type = ''
        character(len=64) :: dust_type = ''
        integer :: setup_nebular_gaussians = 0
        integer :: smooth_velocity = 0
        integer :: add_neb_emission = 0
        integer :: add_neb_continuum = 0
        integer :: add_dust_emission = 0
        integer :: add_agn_dust = 0
        integer :: add_xrb_emission = 0
        integer :: add_agb_dust_model = 0
        integer :: use_wr_spectra = 0

        !> Per-context tables recorded by the populating context (see fsps_setup_tables_t).
        !> Allocatable so that an entry stays small: a 2 MB component made gfortran build the
        !> default-initialized entry on the stack, overflowing callers that hold a context there.
        type(fsps_setup_tables_t), allocatable :: tables

        real(WP), pointer :: indexdefined(:, :) => null()
        real(WP), pointer :: wgdust(:, :, :, :) => null()
        real(WP), pointer :: g03smcextn(:) => null()
        real(WP), pointer :: bands(:, :) => null()
        real(WP), pointer :: magsun(:) => null()
        real(WP), pointer :: magvega(:) => null()
        real(WP), pointer :: filter_leff(:) => null()
        real(WP), pointer :: vega_spec(:) => null()
        real(WP), pointer :: sun_spec(:) => null()
        real(WP), pointer :: spec_lambda(:) => null()
        real(WP), pointer :: spec_nu(:) => null()
        real(WP), pointer :: spec_res(:) => null()
        real(kind(1.0)), pointer :: speclib(:, :, :, :) => null()
        real(kind(1.0)), pointer :: wmb_spec(:, :, :, :) => null()
        real(KP), pointer :: agb_spec_o(:, :) => null()
        real(WP), pointer :: agb_logt_o(:, :) => null()
        real(KP), pointer :: agb_spec_c(:, :) => null()
        real(WP), pointer :: agb_logt_c(:) => null()
        real(KP), pointer :: agb_spec_car(:, :) => null()
        !> Templates used for dusty AGB stars, smoothed as legacy ADD_AGB_DUST smooths each such star
        !> (10^4 km/s, lambda >= 3 um); only the smoothed segment lambda >= AGB_SMOOTH_LAMBDA_MIN is
        !> stored. O-rich and Lancon-Wood C-rich templates include their lambda^2/c factor.
        real(KP), pointer :: speclib_agb_smooth(:, :, :, :) => null()
        real(KP), pointer :: agb_spec_o_smooth(:, :) => null()
        real(KP), pointer :: agb_spec_c_smooth(:, :) => null()
        real(KP), pointer :: agb_spec_car_smooth(:, :) => null()
        real(KP), pointer :: pagb_spec(:, :, :) => null()
        real(KP), pointer :: wrn_spec(:, :, :) => null()
        real(KP), pointer :: wrc_spec(:, :, :) => null()
        real(WP), pointer :: qpaharr(:) => null()
        real(WP), pointer :: uminarr(:) => null()
        real(WP), pointer :: lambda_dustem(:) => null()
        real(WP), pointer :: dustem_dustem(:, :) => null()
        real(WP), pointer :: dustem2_dustem(:, :, :) => null()
        real(KP), pointer :: flux_dagb(:, :, :, :) => null()
        real(WP), pointer :: nebem_cont(:, :, :, :) => null()
        real(WP), pointer :: xnebem_cont(:, :, :, :) => null()
        real(WP), pointer :: neb_res_min(:) => null()
        real(KP), pointer :: gaussnebarr(:, :) => null()
        real(WP), pointer :: agndust_spec(:, :) => null()
        real(WP), pointer :: mact_isoc(:, :, :) => null()
        real(WP), pointer :: logl_isoc(:, :, :) => null()
        real(WP), pointer :: logt_isoc(:, :, :) => null()
        real(WP), pointer :: logg_isoc(:, :, :) => null()
        real(WP), pointer :: ffco_isoc(:, :, :) => null()
        real(WP), pointer :: phase_isoc(:, :, :) => null()
        real(WP), pointer :: mini_isoc(:, :, :) => null()
        real(WP), pointer :: lmdot_isoc(:, :, :) => null()
        integer, pointer :: nmass_isoc(:, :) => null()
        real(WP), pointer :: timestep_isoc(:, :) => null()
        real(WP), pointer :: zlegend(:) => null()
        real(WP), pointer :: zlegendinit(:) => null()
        real(WP), pointer :: bpass_spec_ssp(:, :, :) => null()
        real(WP), pointer :: bpass_mass_ssp(:, :) => null()
        real(WP), pointer :: lam_xrb(:) => null()
        real(WP), pointer :: spec_xrb(:, :, :) => null()
        real(WP), pointer :: ages_xrb(:) => null()
        real(WP), pointer :: zmet_xrb(:) => null()
        real(WP), pointer :: time_full(:) => null()
    end type fsps_setup_cache_t

    type :: fsps_setup_cache_ptr_t
        type(fsps_setup_cache_t), pointer :: p => null()
    end type fsps_setup_cache_ptr_t

    type(fsps_setup_cache_ptr_t), allocatable :: setup_cache(:)

contains

    !> @brief Acquire or create a shared setup cache entry.
    !> @param[in] key Cache key string.
    !> @param[out] entry Cache entry pointer.
    !> @param[out] is_new True if a new cache entry was created.
    subroutine fsps_cache_get_setup(key, entry, is_new)
        character(len=*), intent(in) :: key
        type(fsps_setup_cache_t), pointer :: entry
        logical, intent(out) :: is_new

        integer :: i, empty_slot
        type(fsps_setup_cache_ptr_t), allocatable :: tmp(:)

        is_new = .false.
        empty_slot = 0
        nullify (entry)

        if (allocated(setup_cache)) then
            do i = 1, size(setup_cache)
                if (.not. associated(setup_cache(i)%p)) cycle
                if (len_trim(setup_cache(i)%p%key) == 0) then
                    if (empty_slot == 0) empty_slot = i
                else if (trim(setup_cache(i)%p%key) == trim(key)) then
                    setup_cache(i)%p%refcount = setup_cache(i)%p%refcount + 1
                    entry => setup_cache(i)%p
                    return
                end if
            end do
        end if

        is_new = .true.
        if (allocated(setup_cache)) then
            if (empty_slot > 0) then
                setup_cache(empty_slot)%p%key = trim(key)
                setup_cache(empty_slot)%p%refcount = 1
                entry => setup_cache(empty_slot)%p
                return
            end if
            allocate (tmp(size(setup_cache) + 1))
            tmp(1:size(setup_cache)) = setup_cache
            call move_alloc(tmp, setup_cache)
        else
            allocate (setup_cache(1))
        end if

        ! Explicitly allocate the cache object on the heap
        allocate(setup_cache(size(setup_cache))%p)
        setup_cache(size(setup_cache))%p%key = trim(key)
        setup_cache(size(setup_cache))%p%refcount = 1
        entry => setup_cache(size(setup_cache))%p
    end subroutine fsps_cache_get_setup

    !> @brief Release a setup cache entry and free it when refcount hits zero.
    !> @param[inout] entry Cache entry pointer.
    subroutine fsps_cache_release_setup(entry)
        type(fsps_setup_cache_t), pointer :: entry

        if (.not. associated(entry)) return

        ! The refcount and the shared arrays are touched by every context's setup and release
        call fsps_setup_lock()
        entry%refcount = entry%refcount - 1
        if (entry%refcount <= 0) then
            call fsps_cache_setup_clear(entry)
        end if
        call fsps_setup_unlock()

        nullify (entry)
    end subroutine fsps_cache_release_setup

    !> @brief Clear and deallocate all arrays in a cache entry.
    !> @param[inout] entry Cache entry to clear.
    subroutine fsps_cache_setup_clear(entry)
        type(fsps_setup_cache_t), intent(inout) :: entry

        if (associated(entry%indexdefined)) deallocate (entry%indexdefined)
        if (associated(entry%wgdust)) deallocate (entry%wgdust)
        if (associated(entry%g03smcextn)) deallocate (entry%g03smcextn)
        if (associated(entry%bands)) deallocate (entry%bands)
        if (associated(entry%magsun)) deallocate (entry%magsun)
        if (associated(entry%magvega)) deallocate (entry%magvega)
        if (associated(entry%filter_leff)) deallocate (entry%filter_leff)
        if (associated(entry%vega_spec)) deallocate (entry%vega_spec)
        if (associated(entry%sun_spec)) deallocate (entry%sun_spec)
        if (associated(entry%spec_lambda)) deallocate (entry%spec_lambda)
        if (associated(entry%spec_nu)) deallocate (entry%spec_nu)
        if (associated(entry%spec_res)) deallocate (entry%spec_res)
        if (associated(entry%speclib)) deallocate (entry%speclib)
        if (associated(entry%wmb_spec)) deallocate (entry%wmb_spec)
        if (associated(entry%agb_spec_o)) deallocate (entry%agb_spec_o)
        if (associated(entry%agb_logt_o)) deallocate (entry%agb_logt_o)
        if (associated(entry%agb_spec_c)) deallocate (entry%agb_spec_c)
        if (associated(entry%agb_logt_c)) deallocate (entry%agb_logt_c)
        if (associated(entry%agb_spec_car)) deallocate (entry%agb_spec_car)
        if (associated(entry%speclib_agb_smooth)) deallocate (entry%speclib_agb_smooth)
        if (associated(entry%agb_spec_o_smooth)) deallocate (entry%agb_spec_o_smooth)
        if (associated(entry%agb_spec_c_smooth)) deallocate (entry%agb_spec_c_smooth)
        if (associated(entry%agb_spec_car_smooth)) deallocate (entry%agb_spec_car_smooth)
        if (associated(entry%pagb_spec)) deallocate (entry%pagb_spec)
        if (associated(entry%wrn_spec)) deallocate (entry%wrn_spec)
        if (associated(entry%wrc_spec)) deallocate (entry%wrc_spec)
        if (associated(entry%qpaharr)) deallocate (entry%qpaharr)
        if (associated(entry%uminarr)) deallocate (entry%uminarr)
        if (associated(entry%lambda_dustem)) deallocate (entry%lambda_dustem)
        if (associated(entry%dustem_dustem)) deallocate (entry%dustem_dustem)
        if (associated(entry%dustem2_dustem)) deallocate (entry%dustem2_dustem)
        if (associated(entry%flux_dagb)) deallocate (entry%flux_dagb)
        if (associated(entry%nebem_cont)) deallocate (entry%nebem_cont)
        if (associated(entry%xnebem_cont)) deallocate (entry%xnebem_cont)
        if (associated(entry%neb_res_min)) deallocate (entry%neb_res_min)
        if (associated(entry%gaussnebarr)) deallocate (entry%gaussnebarr)
        if (associated(entry%agndust_spec)) deallocate (entry%agndust_spec)
        if (associated(entry%mact_isoc)) deallocate (entry%mact_isoc)
        if (associated(entry%logl_isoc)) deallocate (entry%logl_isoc)
        if (associated(entry%logt_isoc)) deallocate (entry%logt_isoc)
        if (associated(entry%logg_isoc)) deallocate (entry%logg_isoc)
        if (associated(entry%ffco_isoc)) deallocate (entry%ffco_isoc)
        if (associated(entry%phase_isoc)) deallocate (entry%phase_isoc)
        if (associated(entry%mini_isoc)) deallocate (entry%mini_isoc)
        if (associated(entry%lmdot_isoc)) deallocate (entry%lmdot_isoc)
        if (associated(entry%nmass_isoc)) deallocate (entry%nmass_isoc)
        if (associated(entry%timestep_isoc)) deallocate (entry%timestep_isoc)
        if (associated(entry%zlegend)) deallocate (entry%zlegend)
        if (associated(entry%zlegendinit)) deallocate (entry%zlegendinit)
        if (associated(entry%bpass_spec_ssp)) deallocate (entry%bpass_spec_ssp)
        if (associated(entry%bpass_mass_ssp)) deallocate (entry%bpass_mass_ssp)
        if (associated(entry%lam_xrb)) deallocate (entry%lam_xrb)
        if (associated(entry%spec_xrb)) deallocate (entry%spec_xrb)
        if (associated(entry%ages_xrb)) deallocate (entry%ages_xrb)
        if (associated(entry%zmet_xrb)) deallocate (entry%zmet_xrb)
        if (associated(entry%time_full)) deallocate (entry%time_full)

        if (allocated(entry%tables)) deallocate (entry%tables)
        entry%key = ''
        entry%refcount = 0
        entry%nz = 0
        entry%nt = 0
        entry%nspec = 0
        entry%nzinit = 0
        entry%nbands = 0
        entry%nindx = 0
        entry%ntfull = 0
        entry%nspec_xrb = 0
        entry%nt_xrb = 0
        entry%nz_xrb = 0
    end subroutine fsps_cache_setup_clear

end module fsps_cache
