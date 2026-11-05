!> \file opensps_accel_blas.f90
!> \brief Platform-independent GPU accelerated BLAS interfaces (cuBLAS, rocBLAS, hipBLAS).
!>
!> Provides a unified abstraction layer across GPU vendors for GEMM operations
!> executed within OpenMP target regions. Handles library initialization, stream
!> binding, synchronization, and GEMM kernel dispatch.
module opensps_accel_blas
    use opensps_precision, only: WP, KP
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_associated, c_int, c_long_long
#if defined(OPENSPS_CUBLAS)
    use cublas
    use omp_lib, only: omp_get_default_device
    use, intrinsic :: iso_c_binding, only: c_intptr_t
#endif
    implicit none
    private

    public :: accel_blas_init, accel_blas_sync, accel_blas_destroy
    public :: accel_blas_gemm, accel_blas_gemm_strided_batched
    public :: accel_blas_ready
    public :: ACCEL_BLAS_OP_N, ACCEL_BLAS_OP_T

    integer, parameter :: ACCEL_BLAS_OP_N = 0
    integer, parameter :: ACCEL_BLAS_OP_T = 1

#if defined(OPENSPS_CUBLAS)
    type(cublasHandle), save :: blas_handle
    ! Raw cuBLAS C entry points taking device c_ptrs (nvfortran's cublas module generics want arrays)
    interface
        integer(c_int) function c_sgemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) bind(C, name="cublasSgemm_v2")
            import :: c_int, c_ptr, c_intptr_t
            type(c_ptr), value :: h
            integer(c_int), value :: ta, tb, m, n, k, lda, ldb, ldc
            real(kind=4), intent(in) :: alpha, beta
            type(c_ptr), value :: a, b, c
        end function
        integer(c_int) function c_dgemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) bind(C, name="cublasDgemm_v2")
            import :: c_int, c_ptr, c_intptr_t
            type(c_ptr), value :: h
            integer(c_int), value :: ta, tb, m, n, k, lda, ldb, ldc
            real(kind=8), intent(in) :: alpha, beta
            type(c_ptr), value :: a, b, c
        end function
        integer(c_int) function c_sgemm_sb(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, nb) &
                bind(C, name="cublasSgemmStridedBatched")
            import :: c_int, c_ptr, c_intptr_t, c_long_long
            type(c_ptr), value :: h
            integer(c_int), value :: ta, tb, m, n, k, lda, ldb, ldc, nb
            integer(c_long_long), value :: sa, sb, sc
            real(kind=4), intent(in) :: alpha, beta
            type(c_ptr), value :: a, b, c
        end function
        integer(c_int) function c_dgemm_sb(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, nb) &
                bind(C, name="cublasDgemmStridedBatched")
            import :: c_int, c_ptr, c_intptr_t, c_long_long
            type(c_ptr), value :: h
            integer(c_int), value :: ta, tb, m, n, k, lda, ldb, ldc, nb
            integer(c_long_long), value :: sa, sb, sc
            real(kind=8), intent(in) :: alpha, beta
            type(c_ptr), value :: a, b, c
        end function
    end interface
    logical, save :: blas_ready = .false.

    interface
        type(c_ptr) function ompx_get_cuda_stream(device, nowait) bind(C, name="ompx_get_cuda_stream")
            import :: c_ptr, c_int
            integer(c_int), value :: device, nowait
        end function ompx_get_cuda_stream
    end interface

#elif defined(OPENSPS_ROCBLAS)
    type(c_ptr), save :: blas_handle = c_null_ptr
    type(c_ptr), save :: blas_stream = c_null_ptr
    logical, save :: blas_ready = .false.

    integer(c_int), parameter :: ROCBLAS_OP_N = 111_c_int
    integer(c_int), parameter :: ROCBLAS_OP_T = 112_c_int

    interface
        function rocblas_create_handle(handle) bind(c, name="rocblas_create_handle")
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int) :: rocblas_create_handle
        end function rocblas_create_handle

        function rocblas_destroy_handle(handle) bind(c, name="rocblas_destroy_handle")
            import :: c_ptr, c_int
            type(c_ptr), value :: handle
            integer(c_int) :: rocblas_destroy_handle
        end function rocblas_destroy_handle

        function rocblas_set_stream(handle, stream) bind(c, name="rocblas_set_stream")
            import :: c_ptr, c_int
            type(c_ptr), value :: handle, stream
            integer(c_int) :: rocblas_set_stream
        end function rocblas_set_stream

        function rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) &
            bind(c, name="rocblas_sgemm")
            import :: c_ptr, c_int, KP
            type(c_ptr), value :: handle
            integer(c_int), value :: transA, transB, m, n, k, lda, ldb, ldc
            real(KP) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int) :: rocblas_sgemm
        end function rocblas_sgemm

        function rocblas_dgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) &
            bind(c, name="rocblas_dgemm")
            import :: c_ptr, c_int, WP
            type(c_ptr), value :: handle
            integer(c_int), value :: transA, transB, m, n, k, lda, ldb, ldc
            real(WP) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int) :: rocblas_dgemm
        end function rocblas_dgemm

        function rocblas_sgemm_strided_batched(handle, transA, transB, m, n, k, alpha, &
                                               A, lda, stride_a, B, ldb, stride_b, beta, &
                                               C, ldc, stride_c, batch_count) &
            bind(c, name="rocblas_sgemm_strided_batched")
            import :: c_ptr, c_int, c_long_long, KP
            type(c_ptr), value :: handle
            integer(c_int), value :: transA, transB, m, n, k, lda, ldb, ldc, batch_count
            integer(c_long_long), value :: stride_a, stride_b, stride_c
            real(KP) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int) :: rocblas_sgemm_strided_batched
        end function rocblas_sgemm_strided_batched

        function rocblas_dgemm_strided_batched(handle, transA, transB, m, n, k, alpha, &
                                               A, lda, stride_a, B, ldb, stride_b, beta, &
                                               C, ldc, stride_c, batch_count) &
            bind(c, name="rocblas_dgemm_strided_batched")
            import :: c_ptr, c_int, c_long_long, WP
            type(c_ptr), value :: handle
            integer(c_int), value :: transA, transB, m, n, k, lda, ldb, ldc, batch_count
            integer(c_long_long), value :: stride_a, stride_b, stride_c
            real(WP) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int) :: rocblas_dgemm_strided_batched
        end function rocblas_dgemm_strided_batched

        function hipStreamCreate(stream) bind(c, name="hipStreamCreate")
            import :: c_ptr, c_int
            type(c_ptr) :: stream
            integer(c_int) :: hipStreamCreate
        end function hipStreamCreate

        function hipStreamDestroy(stream) bind(c, name="hipStreamDestroy")
            import :: c_ptr, c_int
            type(c_ptr), value :: stream
            integer(c_int) :: hipStreamDestroy
        end function hipStreamDestroy

        function hipStreamSynchronize(stream) bind(c, name="hipStreamSynchronize")
            import :: c_ptr, c_int
            type(c_ptr), value :: stream
            integer(c_int) :: hipStreamSynchronize
        end function hipStreamSynchronize
    end interface

#elif defined(OPENSPS_HIPBLAS)
    type(c_ptr), save :: blas_handle = c_null_ptr
    type(c_ptr), save :: blas_stream = c_null_ptr
    logical, save :: blas_ready = .false.

    integer(c_int), parameter :: HIPBLAS_OP_N = 111_c_int
    integer(c_int), parameter :: HIPBLAS_OP_T = 112_c_int

    interface
        function hipblasCreate(handle) bind(c, name="hipblasCreate")
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int) :: hipblasCreate
        end function hipblasCreate

        function hipblasDestroy(handle) bind(c, name="hipblasDestroy")
            import :: c_ptr, c_int
            type(c_ptr), value :: handle
            integer(c_int) :: hipblasDestroy
        end function hipblasDestroy

        function hipblasSetStream(handle, stream) bind(c, name="hipblasSetStream")
            import :: c_ptr, c_int
            type(c_ptr), value :: handle, stream
            integer(c_int) :: hipblasSetStream
        end function hipblasSetStream

        function hipblasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) &
            bind(c, name="hipblasSgemm")
            import :: c_ptr, c_int, KP
            type(c_ptr), value :: handle
            integer(c_int), value :: transA, transB, m, n, k, lda, ldb, ldc
            real(KP) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int) :: hipblasSgemm
        end function hipblasSgemm

        function hipblasDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) &
            bind(c, name="hipblasDgemm")
            import :: c_ptr, c_int, WP
            type(c_ptr), value :: handle
            integer(c_int), value :: transA, transB, m, n, k, lda, ldb, ldc
            real(WP) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int) :: hipblasDgemm
        end function hipblasDgemm

        function hipblasSgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                                            A, lda, stride_a, B, ldb, stride_b, beta, &
                                            C, ldc, stride_c, batch_count) &
            bind(c, name="hipblasSgemmStridedBatched")
            import :: c_ptr, c_int, c_long_long, KP
            type(c_ptr), value :: handle
            integer(c_int), value :: transA, transB, m, n, k, lda, ldb, ldc, batch_count
            integer(c_long_long), value :: stride_a, stride_b, stride_c
            real(KP) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int) :: hipblasSgemmStridedBatched
        end function hipblasSgemmStridedBatched

        function hipblasDgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                                            A, lda, stride_a, B, ldb, stride_b, beta, &
                                            C, ldc, stride_c, batch_count) &
            bind(c, name="hipblasDgemmStridedBatched")
            import :: c_ptr, c_int, c_long_long, WP
            type(c_ptr), value :: handle
            integer(c_int), value :: transA, transB, m, n, k, lda, ldb, ldc, batch_count
            integer(c_long_long), value :: stride_a, stride_b, stride_c
            real(WP) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int) :: hipblasDgemmStridedBatched
        end function hipblasDgemmStridedBatched

        function hipStreamCreate(stream) bind(c, name="hipStreamCreate")
            import :: c_ptr, c_int
            type(c_ptr) :: stream
            integer(c_int) :: hipStreamCreate
        end function hipStreamCreate

        function hipStreamDestroy(stream) bind(c, name="hipStreamDestroy")
            import :: c_ptr, c_int
            type(c_ptr), value :: stream
            integer(c_int) :: hipStreamDestroy
        end function hipStreamDestroy

        function hipStreamSynchronize(stream) bind(c, name="hipStreamSynchronize")
            import :: c_ptr, c_int
            type(c_ptr), value :: stream
            integer(c_int) :: hipStreamSynchronize
        end function hipStreamSynchronize
    end interface
#endif

contains

    pure function accel_blas_ready() result(res)
        logical :: res
#if defined(OPENSPS_ACCEL_BLAS)
        res = blas_ready
#else
        res = .false.
#endif
    end function accel_blas_ready

    !> Initialize the accelerated BLAS library context.
    subroutine accel_blas_init()
#if defined(OPENSPS_CUBLAS)
        integer :: istat
        character(len=16) :: mode
        if (blas_ready) return
        istat = cublasCreate(blas_handle)
#ifdef OMP_TARGET
        istat = cublasSetStream(blas_handle, int(transfer(ompx_get_cuda_stream(int(omp_get_default_device(), c_int), 0_c_int), &
                                                          0_c_intptr_t), 8))
#endif
        call get_environment_variable('OPENSPS_CUBLAS_MATH', mode)
        select case (trim(mode))
        case ('tf32')
            istat = cublasSetMathMode(blas_handle, CUBLAS_TF32_TENSOR_OP_MATH)
        case ('pedantic')
            istat = cublasSetMathMode(blas_handle, CUBLAS_PEDANTIC_MATH)
        end select
        blas_ready = .true.
#elif defined(OPENSPS_ROCBLAS)
        integer(c_int) :: istat
        if (blas_ready) return
        istat = hipStreamCreate(blas_stream)
        if (istat /= 0) error stop 'accel_blas: hipStreamCreate failed'
        istat = rocblas_create_handle(blas_handle)
        if (istat /= 0) error stop 'accel_blas: rocblas_create_handle failed'
        istat = rocblas_set_stream(blas_handle, blas_stream)
        if (istat /= 0) error stop 'accel_blas: rocblas_set_stream failed'
        blas_ready = .true.
#elif defined(OPENSPS_HIPBLAS)
        integer(c_int) :: istat
        if (blas_ready) return
        istat = hipStreamCreate(blas_stream)
        if (istat /= 0) error stop 'accel_blas: hipStreamCreate failed'
        istat = hipblasCreate(blas_handle)
        if (istat /= 0) error stop 'accel_blas: hipblasCreate failed'
        istat = hipblasSetStream(blas_handle, blas_stream)
        if (istat /= 0) error stop 'accel_blas: hipblasSetStream failed'
        blas_ready = .true.
#endif
    end subroutine accel_blas_init

    !> Synchronize asynchronous streams if needed (e.g. on AMD ROCm when using a dedicated stream).
    subroutine accel_blas_sync()
#if defined(OPENSPS_ROCBLAS) || defined(OPENSPS_HIPBLAS)
        integer(c_int) :: istat
        if (blas_ready .and. c_associated(blas_stream)) then
            istat = hipStreamSynchronize(blas_stream)
            if (istat /= 0) error stop 'accel_blas: hipStreamSynchronize failed'
        end if
#endif
    end subroutine accel_blas_sync

    !> Destroy accelerated BLAS handle and streams.
    subroutine accel_blas_destroy()
#if defined(OPENSPS_CUBLAS)
        integer :: istat
        if (blas_ready) then
            istat = cublasDestroy(blas_handle)
            blas_ready = .false.
        end if
#elif defined(OPENSPS_ROCBLAS)
        integer(c_int) :: istat
        if (blas_ready) then
            istat = rocblas_destroy_handle(blas_handle)
            istat = hipStreamDestroy(blas_stream)
            blas_handle = c_null_ptr
            blas_stream = c_null_ptr
            blas_ready = .false.
        end if
#elif defined(OPENSPS_HIPBLAS)
        integer(c_int) :: istat
        if (blas_ready) then
            istat = hipblasDestroy(blas_handle)
            istat = hipStreamDestroy(blas_stream)
            blas_handle = c_null_ptr
            blas_stream = c_null_ptr
            blas_ready = .false.
        end if
#endif
    end subroutine accel_blas_destroy

    !> Dispatch general matrix-matrix product: C = alpha * op(A) * op(B) + beta * C
    !> Operates on device pointers inside an OpenMP target data region.
    subroutine accel_blas_gemm(transa_in, transb_in, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, ldc)
        integer, intent(in) :: transa_in, transb_in, m, n, k, lda, ldb, ldc
        real(KP), intent(in) :: alpha, beta
        type(c_ptr), intent(in), value :: dev_a, dev_b, dev_c
#if defined(OPENSPS_CUBLAS)
        integer(c_int) :: istat, ta, tb
        ta = merge(CUBLAS_OP_T, CUBLAS_OP_N, transa_in == ACCEL_BLAS_OP_T)
        tb = merge(CUBLAS_OP_T, CUBLAS_OP_N, transb_in == ACCEL_BLAS_OP_T)
#ifdef KERNEL_SINGLE_PRECISION
        istat = c_sgemm(blas_handle%handle, ta, tb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, ldc)
#else
        istat = c_dgemm(blas_handle%handle, ta, tb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, ldc)
#endif
        if (istat /= 0) error stop 'accel_blas: cuBLAS GEMM failed'
#elif defined(OPENSPS_ROCBLAS)
        integer(c_int) :: istat, ta, tb
        ta = merge(ROCBLAS_OP_T, ROCBLAS_OP_N, transa_in == ACCEL_BLAS_OP_T)
        tb = merge(ROCBLAS_OP_T, ROCBLAS_OP_N, transb_in == ACCEL_BLAS_OP_T)
#ifdef KERNEL_SINGLE_PRECISION
        istat = rocblas_sgemm(blas_handle, ta, tb, int(m, c_int), int(n, c_int), int(k, c_int), &
                              alpha, dev_a, int(lda, c_int), dev_b, int(ldb, c_int), beta, dev_c, int(ldc, c_int))
#else
        istat = rocblas_dgemm(blas_handle, ta, tb, int(m, c_int), int(n, c_int), int(k, c_int), &
                              alpha, dev_a, int(lda, c_int), dev_b, int(ldb, c_int), beta, dev_c, int(ldc, c_int))
#endif
        if (istat /= 0) error stop 'accel_blas: rocBLAS GEMM failed'
        call accel_blas_sync()
#elif defined(OPENSPS_HIPBLAS)
        integer(c_int) :: istat, ta, tb
        ta = merge(HIPBLAS_OP_T, HIPBLAS_OP_N, transa_in == ACCEL_BLAS_OP_T)
        tb = merge(HIPBLAS_OP_T, HIPBLAS_OP_N, transb_in == ACCEL_BLAS_OP_T)
#ifdef KERNEL_SINGLE_PRECISION
        istat = hipblasSgemm(blas_handle, ta, tb, int(m, c_int), int(n, c_int), int(k, c_int), &
                             alpha, dev_a, int(lda, c_int), dev_b, int(ldb, c_int), beta, dev_c, int(ldc, c_int))
#else
        istat = hipblasDgemm(blas_handle, ta, tb, int(m, c_int), int(n, c_int), int(k, c_int), &
                             alpha, dev_a, int(lda, c_int), dev_b, int(ldb, c_int), beta, dev_c, int(ldc, c_int))
#endif
        if (istat /= 0) error stop 'accel_blas: hipBLAS GEMM failed'
        call accel_blas_sync()
#endif
    end subroutine accel_blas_gemm

    !> Dispatch strided batched GEMM: C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
    subroutine accel_blas_gemm_strided_batched(transa_in, transb_in, m, n, k, alpha, &
                                               dev_a, lda, stride_a, dev_b, ldb, stride_b, &
                                               beta, dev_c, ldc, stride_c, batch_count)
        integer, intent(in) :: transa_in, transb_in, m, n, k, lda, ldb, ldc, batch_count
        integer(c_long_long), intent(in) :: stride_a, stride_b, stride_c
        real(KP), intent(in) :: alpha, beta
        type(c_ptr), intent(in), value :: dev_a, dev_b, dev_c
#if defined(OPENSPS_CUBLAS)
        integer(c_int) :: istat, ta, tb
        ta = merge(CUBLAS_OP_T, CUBLAS_OP_N, transa_in == ACCEL_BLAS_OP_T)
        tb = merge(CUBLAS_OP_T, CUBLAS_OP_N, transb_in == ACCEL_BLAS_OP_T)
#ifdef KERNEL_SINGLE_PRECISION
        istat = c_sgemm_sb(blas_handle%handle, ta, tb, m, n, k, alpha, &
                                          dev_a, lda, stride_a, dev_b, ldb, stride_b, &
                                          beta, dev_c, ldc, stride_c, batch_count)
#else
        istat = c_dgemm_sb(blas_handle%handle, ta, tb, m, n, k, alpha, &
                                          dev_a, lda, stride_a, dev_b, ldb, stride_b, &
                                          beta, dev_c, ldc, stride_c, batch_count)
#endif
        if (istat /= 0) error stop 'accel_blas: cuBLAS batched GEMM failed'
#elif defined(OPENSPS_ROCBLAS)
        integer(c_int) :: istat, ta, tb
        ta = merge(ROCBLAS_OP_T, ROCBLAS_OP_N, transa_in == ACCEL_BLAS_OP_T)
        tb = merge(ROCBLAS_OP_T, ROCBLAS_OP_N, transb_in == ACCEL_BLAS_OP_T)
#ifdef KERNEL_SINGLE_PRECISION
        istat = rocblas_sgemm_strided_batched(blas_handle, ta, tb, int(m, c_int), int(n, c_int), int(k, c_int), &
                                              alpha, dev_a, int(lda, c_int), stride_a, &
                                              dev_b, int(ldb, c_int), stride_b, beta, &
                                              dev_c, int(ldc, c_int), stride_c, int(batch_count, c_int))
#else
        istat = rocblas_dgemm_strided_batched(blas_handle, ta, tb, int(m, c_int), int(n, c_int), int(k, c_int), &
                                              alpha, dev_a, int(lda, c_int), stride_a, &
                                              dev_b, int(ldb, c_int), stride_b, beta, &
                                              dev_c, int(ldc, c_int), stride_c, int(batch_count, c_int))
#endif
        if (istat /= 0) error stop 'accel_blas: rocBLAS batched GEMM failed'
        call accel_blas_sync()
#elif defined(OPENSPS_HIPBLAS)
        integer(c_int) :: istat, ta, tb
        ta = merge(HIPBLAS_OP_T, HIPBLAS_OP_N, transa_in == ACCEL_BLAS_OP_T)
        tb = merge(HIPBLAS_OP_T, HIPBLAS_OP_N, transb_in == ACCEL_BLAS_OP_T)
#ifdef KERNEL_SINGLE_PRECISION
        istat = hipblasSgemmStridedBatched(blas_handle, ta, tb, int(m, c_int), int(n, c_int), int(k, c_int), &
                                           alpha, dev_a, int(lda, c_int), stride_a, &
                                           dev_b, int(ldb, c_int), stride_b, beta, &
                                           dev_c, int(ldc, c_int), stride_c, int(batch_count, c_int))
#else
        istat = hipblasDgemmStridedBatched(blas_handle, ta, tb, int(m, c_int), int(n, c_int), int(k, c_int), &
                                           alpha, dev_a, int(lda, c_int), stride_a, &
                                           dev_b, int(ldb, c_int), stride_b, beta, &
                                           dev_c, int(ldc, c_int), stride_c, int(batch_count, c_int))
#endif
        if (istat /= 0) error stop 'accel_blas: hipBLAS batched GEMM failed'
        call accel_blas_sync()
#endif
    end subroutine accel_blas_gemm_strided_batched

end module opensps_accel_blas
