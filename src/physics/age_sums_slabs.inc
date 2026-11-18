! The age sums of a batch whose models blend K slabs of the resident basis (K = AS_K, fixed at compile time so that the
! register tiles of the device kernels are sized for it), from the descriptor of the models: model m's SSP is
! sum_j wt(j, m) basis(:, :, slab(j, m)) at every age, and its young and old spectra (ages up to it_tesc(m), and after) are
! the sums over the ages of the weight times that. Included by opensps_csp_batch for K = 2 (the two metallicity slabs)
! and K = 4 (those of two [alpha/Fe] planes); the blend is summed in the order of the descriptor.
!   AS_NAME      the name of the routine
!   AS_K         the number of slabs
!   AS_FLANG_MT, AS_FLANG_PT   the register tile of the LLVM flang / AMD kernel (tuning parameters of opensps_tuning)
!   AS_NVF_MT    the models per thread of the nvfortran kernel
    subroutine AS_NAME(nspec, nt, nsl, n_mod, basis, w2flat, it_tesc, slab, wt, yo)
        integer, parameter :: KS = AS_K
        integer, intent(in)   :: nspec, nt, nsl, n_mod, it_tesc(n_mod), slab(KS, n_mod)
        real(KP), intent(in)  :: basis(nspec, nt, nsl), w2flat(nt * 2 * n_mod), wt(KS, n_mod)
        real(KP), intent(out) :: yo(nspec, 2 * n_mod)
#if defined(OMP_TARGET) && defined(__flang__)
        ! LLVM flang / AMD. The models are visited sorted by their first slab (their metallicity cell), so that a tile of MT
        ! consecutive models usually shares its slabs; each thread does PT pixels (strided, so that the loads are coalesced)
        ! of the tile, which shares the per-model weight loads among the pixels. The per-model offsets are formed once,
        ! outside the age loop (inside it, address arithmetic and loads were ~97% of the instructions). The tile shape
        ! matters (registers spill): see opensps_tuning for the measurements.
        integer, parameter :: MT = AS_FLANG_MT, PT = AS_FLANG_PT
        integer  :: m, mq, ip, npt, key, ncell
        integer, allocatable :: zord(:), starts(:)

        ncell = maxval(slab(1, 1:n_mod))
        allocate(zord(n_mod), starts(ncell + 2))
        starts = 0
        do m = 1, n_mod
            starts(slab(1, m) + 1) = starts(slab(1, m) + 1) + 1
        end do
        starts(1) = 1
        do key = 2, ncell + 1
            starts(key) = starts(key) + starts(key - 1)
        end do
        do m = 1, n_mod
            zord(starts(slab(1, m))) = m
            starts(slab(1, m)) = starts(slab(1, m)) + 1
        end do

        npt = (nspec + PT - 1) / PT
        ! (Work arrays in a block of the loop body: flang maps `private` arrays, a copy each way per launch)
        !$omp target teams distribute parallel do collapse(2) map(to: zord)
        do mq = 1, (n_mod + MT - 1) / MT
            do ip = 1, npt
              block
                integer  :: i, k, l, p, j, mm(MT), ii(PT), it(MT), oy(MT), oo(MT), sl(KS, MT)
                real(KP) :: b(PT, KS), bk, wy, y(PT, MT), o(PT, MT), c(KS, MT)
                logical  :: same
                do l = 1, MT
                    mm(l) = zord(min((mq - 1) * MT + l, n_mod))
                    do j = 1, KS
                        c(j, l) = wt(j, mm(l))
                        sl(j, l) = slab(j, mm(l))
                    end do
                    it(l) = it_tesc(mm(l))
                    oy(l) = (mm(l) - 1) * nt          ! offsets of the model's young and old weight columns
                    oo(l) = (n_mod + mm(l) - 1) * nt
                end do
                do p = 1, PT
                    ii(p) = min(ip + (p - 1) * npt, nspec)
                end do
                y = 0.0_KP
                o = 0.0_KP
                same = .true.
                do l = 2, MT
                    do j = 1, KS
                        if (sl(j, l) /= sl(j, 1)) same = .false.
                    end do
                end do
                if (same) then
                    do k = 1, nt
                        do j = 1, KS
                            do p = 1, PT
                                b(p, j) = basis(ii(p), k, sl(j, 1))
                            end do
                        end do
                        do l = 1, MT
                            if (k <= it(l)) then
                                wy = w2flat(oy(l) + k)
                                do p = 1, PT
                                    bk = c(1, l) * b(p, 1)
                                    do j = 2, KS
                                        bk = bk + c(j, l) * b(p, j)
                                    end do
                                    y(p, l) = y(p, l) + wy * bk
                                end do
                            else
                                wy = w2flat(oo(l) + k)
                                do p = 1, PT
                                    bk = c(1, l) * b(p, 1)
                                    do j = 2, KS
                                        bk = bk + c(j, l) * b(p, j)
                                    end do
                                    o(p, l) = o(p, l) + wy * bk
                                end do
                            end if
                        end do
                    end do
                else
                    do k = 1, nt
                        do l = 1, MT
                            if (k <= it(l)) then
                                wy = w2flat(oy(l) + k)
                                do p = 1, PT
                                    bk = c(1, l) * basis(ii(p), k, sl(1, l))
                                    do j = 2, KS
                                        bk = bk + c(j, l) * basis(ii(p), k, sl(j, l))
                                    end do
                                    y(p, l) = y(p, l) + wy * bk
                                end do
                            else
                                wy = w2flat(oo(l) + k)
                                do p = 1, PT
                                    bk = c(1, l) * basis(ii(p), k, sl(1, l))
                                    do j = 2, KS
                                        bk = bk + c(j, l) * basis(ii(p), k, sl(j, l))
                                    end do
                                    o(p, l) = o(p, l) + wy * bk
                                end do
                            end if
                        end do
                    end do
                end if
                do l = 1, MT
                    do p = 1, PT
                        yo(ii(p), mm(l)) = y(p, l)
                        yo(ii(p), n_mod + mm(l)) = o(p, l)
                    end do
                end do
              end block
            end do
        end do
#elif defined(OMP_TARGET)
        ! nvfortran: MT models per thread, one pixel per thread
        integer, parameter :: MT = AS_NVF_MT
        integer  :: mq, i, k, l, j, mm(MT), sl(KS, MT)
        real(KP) :: bk, y(MT), o(MT), c(KS, MT)

        !$omp target teams distribute parallel do collapse(2) private(k, l, j, bk, mm, y, o, c, sl)
        do mq = 1, (n_mod + MT - 1) / MT
            do i = 1, nspec
                do l = 1, MT
                    mm(l) = min((mq - 1) * MT + l, n_mod)
                    do j = 1, KS
                        c(j, l) = wt(j, mm(l))
                        sl(j, l) = slab(j, mm(l))
                    end do
                    y(l) = 0.0_KP
                    o(l) = 0.0_KP
                end do
                do k = 1, nt
                    do l = 1, MT
                        bk = c(1, l) * basis(i, k, sl(1, l))
                        do j = 2, KS
                            bk = bk + c(j, l) * basis(i, k, sl(j, l))
                        end do
                        if (k <= it_tesc(mm(l))) then
                            y(l) = y(l) + w2flat(k + (mm(l) - 1) * nt) * bk
                        else
                            o(l) = o(l) + w2flat(k + (n_mod + mm(l) - 1) * nt) * bk
                        end if
                    end do
                end do
                do l = 1, MT
                    yo(i, mm(l)) = y(l)
                    yo(i, n_mod + mm(l)) = o(l)
                end do
            end do
        end do
#else
        call age_sums_host(KS, nspec, nt, nsl, n_mod, basis, w2flat, it_tesc, slab, wt, yo)
#endif
    end subroutine AS_NAME
