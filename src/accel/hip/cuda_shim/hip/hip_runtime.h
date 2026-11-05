// Minimal HIP-on-CUDA shim: lets nvcc compile opensps_hip_kernels.hip for NVIDIA GPUs.
// Covers exactly the HIP API the kernels use.
#pragma once
#include <cuda_runtime.h>
typedef cudaStream_t hipStream_t;
#define hipStreamSynchronize cudaStreamSynchronize
#define hipDeviceSynchronize cudaDeviceSynchronize
#define __shfl_down(v, o, w) __shfl_down_sync(0xffffffffu, (v), (o), (w))
