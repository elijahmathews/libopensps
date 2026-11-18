! The adjoint of the age sums and of L_bol for a batch whose models blend K slabs of the resident basis (K = AS_K = 1, 2 or 4, as
! age_sums_slabs.inc), from the descriptor slab(K, N) of the models; the weights of the blend are those of the model's position
! (1 - d, d) in metallicity and (1 - a, a) in [alpha/Fe] (K = 4: their products, slab j = 1 + ix + 2 ia, ix, ia = 0, 1).
!
! At every age k and model m: w_bar(k, m) = sum over pixels of the model's SSP times the young cotangent (k <= dust_tesc) or the old
! one, plus lsum_bar L_bol(k) (nebular ages: the EUV rows, the pixels up to whlylim, carry the escaping fraction fesc, gneb(4) =
! d L / d fesc, and nwb, the nebular continuum's and lines' part, is added). The blend is linear in the slabs, so each slab's dots with the
! cotangents are formed once (S_j over the non-EUV pixels, E_j over the EUV ones: compensated at kernel precision, on CPUs with
! dot_lanes) and everything else is a combination of them in FP64: w_bar with the weights, d L / d dz (gneb(5)) and d L / d da
! (gneb(6)) with the derivatives of the weights, the L_bol blend and its derivatives likewise. One body for K = 1, 2, 4.
! The work of one (model, age) is its own routine (AS_ONE), which the device kernel calls: its small arrays are local to it, which
! needs neither a BLOCK construct (nvfortran does not accept one in a parallel region) nor private arrays (LLVM flang maps them).
!   AS_NAME   the name of the routine
!   AS_ONE    the name of the routine of one (model, age)
!   AS_K      the number of slabs

    !> Model m at age k: its w_bar (wb), the EUV part of the escaping-fraction derivative (eo), and the derivatives with respect to dz
    !> (zo) and da (ao) through this age
    subroutine AS_ONE(m, k, n_mod, nspec, nt, nzb, basis, basis_t, blbol, it_tesc, slab, dz, da, &
                      lum_linear, lsum_bar, gw, ntn, whlylim, fesc, w2, nwb, wb, eo, zo, ao)
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, parameter :: KS = AS_K
        integer, intent(in)   :: m, k, n_mod, nspec, nt, nzb, it_tesc(n_mod), slab(KS, n_mod), ntn, whlylim
        real(KP), intent(in)  :: basis(nspec, nt, nzb), basis_t(nt, nspec, nzb), gw(nspec, 2 * n_mod), w2(nt, 2 * n_mod)
        real(WP), intent(in)  :: blbol(nt, nzb), dz(n_mod), da(n_mod), lum_linear(nt), lsum_bar(n_mod), fesc(n_mod)
        real(WP), intent(in)  :: nwb(max(1, ntn), n_mod)
        real(WP), intent(out) :: wb, eo, zo, ao

        integer  :: i, jj, ix, ia, col, nlim, sl(KS)
        real(KP) :: fk
        real(WP) :: s(KS), e(KS), wt(KS), dzw(KS), daw(KS), wz, wa, fkw, swt, ewt, dzs, dze, das, dae, lk, dlz, dla
#if defined(OMP_TARGET)
        real(KP) :: sk(KS), ck(KS), ek(KS), ce(KS), term, tsum
#endif
        col = n_mod + m
        if (k <= it_tesc(m)) col = m
        ! The EUV pixels of a nebular age: summed apart, then weighted by fesc
        nlim = 0
        if (k <= ntn) nlim = whlylim
        fk = 1.0_KP
        if (k <= ntn) fk = real(fesc(m), KP)
        fkw = real(fk, WP)
        ! The weights of the blend and their derivatives with respect to dz and da
        do jj = 1, KS
            sl(jj) = slab(jj, m)
            ix = mod(jj - 1, 2)
            ia = (jj - 1) / 2
            wz = 1.0_wp
            wa = 1.0_wp
            if (KS >= 2) wz = merge(dz(m), 1.0_wp - dz(m), ix == 1)
            if (KS == 4) wa = merge(da(m), 1.0_wp - da(m), ia == 1)
            wt(jj) = wz * wa
            dzw(jj) = merge(1.0_wp, -1.0_wp, ix == 1) * wa
            daw(jj) = wz * merge(1.0_wp, -1.0_wp, ia == 1)
        end do
#if !defined(OMP_TARGET)
        ! CPUs: per slab two lane-vectorized dot products (dot_lanes): the non-EUV and the EUV pixels
        do jj = 1, KS
            s(jj) = dot_lanes(nspec - nlim, basis(nlim + 1, k, sl(jj)), gw(nlim + 1, col))
            e(jj) = dot_lanes(nlim, basis(1, k, sl(jj)), gw(1, col))
        end do
#else
        ! One thread per (model, age), serial over the pixels (the basis transposed: ages fastest, so that the threads of a
        ! warp read together), the sums compensated at kernel precision
        do jj = 1, KS
            sk(jj) = 0.0_KP; ck(jj) = 0.0_KP; ek(jj) = 0.0_KP; ce(jj) = 0.0_KP
        end do
        do i = 1, nlim
            do jj = 1, KS
                term = basis_t(k, i, sl(jj)) * gw(i, col) - ce(jj)
                tsum = ek(jj) + term
                ce(jj) = (tsum - ek(jj)) - term
                ek(jj) = tsum
            end do
        end do
        do i = nlim + 1, nspec
            do jj = 1, KS
                term = basis_t(k, i, sl(jj)) * gw(i, col) - ck(jj)
                tsum = sk(jj) + term
                ck(jj) = (tsum - sk(jj)) - term
                sk(jj) = tsum
            end do
        end do
        do jj = 1, KS
            s(jj) = real(sk(jj), WP)
            e(jj) = real(ek(jj), WP)
        end do
#endif
        ! The SSP of the model: its dots with the cotangents, and its L_bol
        swt = wt(1) * s(1)
        ewt = wt(1) * e(1)
        do jj = 2, KS
            swt = swt + wt(jj) * s(jj)
            ewt = ewt + wt(jj) * e(jj)
        end do
        if (KS == 1) then
            lk = lum_linear(k)
        else
            dlz = wt(1) * blbol(k, sl(1))
            do jj = 2, KS
                dlz = dlz + wt(jj) * blbol(k, sl(jj))
            end do
            lk = 10.0_wp**dlz
        end if
        eo = ewt * real(w2(k, col), WP)
        wb = swt + fkw * ewt + lsum_bar(m) * lk
        if (k <= ntn) wb = wb + nwb(k, m)
        ! d L / d dz and d L / d da through age k: the dots and the L_bol blend with the derivatives of the weights
        zo = 0.0_wp
        ao = 0.0_wp
        if (KS >= 2) then
            dzs = dzw(1) * s(1)
            dze = dzw(1) * e(1)
            dlz = dzw(1) * blbol(k, sl(1))
            do jj = 2, KS
                dzs = dzs + dzw(jj) * s(jj)
                dze = dze + dzw(jj) * e(jj)
                dlz = dlz + dzw(jj) * blbol(k, sl(jj))
            end do
            zo = real(w2(k, col), WP) * ((dzs + fkw * dze) + lsum_bar(m) * lk * log(10.0_wp) * dlz)
        end if
        if (KS == 4) then
            das = daw(1) * s(1)
            dae = daw(1) * e(1)
            dla = daw(1) * blbol(k, sl(1))
            do jj = 2, KS
                das = das + daw(jj) * s(jj)
                dae = dae + daw(jj) * e(jj)
                dla = dla + daw(jj) * blbol(k, sl(jj))
            end do
            ao = real(w2(k, col), WP) * ((das + fkw * dae) + lsum_bar(m) * lk * log(10.0_wp) * dla)
        end if
    end subroutine AS_ONE

    subroutine AS_NAME(n_mod, nspec, nt, nzb, basis, basis_t, blbol, it_tesc, slab, dz, da, lum_linear, &
                       lsum_bar, gw, w_bar, ntn, whlylim, fesc, w2, nwb, gneb)
        integer, parameter :: KS = AS_K
        integer, intent(in)     :: n_mod, nspec, nt, nzb, it_tesc(n_mod), slab(KS, n_mod), ntn, whlylim
        real(KP), intent(in)    :: basis(nspec, nt, nzb), basis_t(nt, nspec, nzb), gw(nspec, 2 * n_mod), w2(nt, 2 * n_mod)
        real(WP), intent(in)    :: blbol(nt, nzb), dz(n_mod), da(n_mod), lum_linear(nt), lsum_bar(n_mod), fesc(n_mod)
        real(WP), intent(in)    :: nwb(max(1, ntn), n_mod)
        real(WP), intent(out)   :: w_bar(nt, n_mod)
        real(WP), intent(inout) :: gneb(6, n_mod)   ! (4): set here; (5), (6): added to

        integer  :: m, k
        real(WP) :: wb, eo, zo, ao, sf, sd, sa
        real(WP), allocatable :: euv(:,:), zd(:,:), ad(:,:)

        allocate(euv(max(1, ntn), n_mod), zd(nt, n_mod), ad(nt, n_mod))
#ifdef OMP_TARGET
        !$omp target data map(alloc: euv, zd, ad)
#endif
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do collapse(2) private(wb, eo, zo, ao)
#elif defined(_OPENMP)
        !$omp parallel do collapse(2) private(wb, eo, zo, ao)
#endif
        do m = 1, n_mod
            do k = 1, nt
                call AS_ONE(m, k, n_mod, nspec, nt, nzb, basis, basis_t, blbol, it_tesc, slab, dz, da, &
                            lum_linear, lsum_bar, gw, ntn, whlylim, fesc, w2, nwb, wb, eo, zo, ao)
                w_bar(k, m) = wb
                if (k <= ntn) euv(k, m) = eo
                zd(k, m) = zo
                ad(k, m) = ao
            end do
        end do
        ! d L / d fesc, d L / d dz and d L / d da per model
#if defined(OMP_TARGET)
        !$omp target teams distribute parallel do private(k, sf, sd, sa)
#elif defined(_OPENMP)
        !$omp parallel do private(k, sf, sd, sa)
#endif
        do m = 1, n_mod
            sf = 0.0_wp
            do k = 1, ntn
                sf = sf + euv(k, m)
            end do
            gneb(4, m) = sf
            if (KS >= 2) then
                sd = 0.0_wp
                do k = 1, nt
                    sd = sd + zd(k, m)
                end do
                gneb(5, m) = gneb(5, m) + sd
            end if
            if (KS == 4) then
                sa = 0.0_wp
                do k = 1, nt
                    sa = sa + ad(k, m)
                end do
                gneb(6, m) = gneb(6, m) + sa
            end if
        end do
#ifdef OMP_TARGET
        !$omp end target data
#endif
    end subroutine AS_NAME
