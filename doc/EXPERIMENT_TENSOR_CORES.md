# Experiment: TF32 / BF16 tensor cores in the batched CSP (Step 1 only: stopped at the gate)

Branch `exp/tensor-core-precision` (from `origin/fix/issue-chooser-security`, b05523a). Experiment only:
no default changes, no PR. 2026-11-15, NVIDIA A100-PCIE-40GB, nvhpc 25.1 (nvfortran, CUDA 12), cc80.

## Result

The GEMM-shaped steps are **6-9 % of the batch time** at every size (gate: 25 %), so the work stopped
after Step 1. Even an infinitely fast GEMM would give at most 1.06-1.09x end to end.
Steps 2 (precision emulation) and 3 (hardware timing) were **not run**.

## Setup and commands

    source /etc/profile.d/modules.sh; module load nvhpc/25.1; unset SPS_HOME; export FC=nvfortran CC=nvc
    export OPENSPS_DATA_HOME=<data-v4 root> OMP_TARGET_OFFLOAD=MANDATORY
    meson setup buildtc --buildtype=release -Ddata_home=$OPENSPS_DATA_HOME -Daccel=omp_gpu -Dgpu_arch=cc80 \
          -Dkernel_precision=single -Daccel_blas=cublas && meson compile -C buildtc
    OPENSPS_BENCH_SIZES=1000,10000,100000 buildtc/benchmark_csp_batch 100000 0 full          # timings
    OPENSPS_BENCH_SIZES=$N nsys profile -t cuda buildtc/benchmark_csp_batch $N 0 full $N        # N = 1000, 10000, 100000
    nsys sqlite export; kernels split into calls at the SFH-weights kernel; "mags-only" = the warm-up and timed
    calls of the magnitudes-only, spectra-on-device and spectra-on-host variants (the with-spectra-copy variant is excluded).

Config `full` (MIST + MILES + DL07, dust, nebular, IGM, AGN, indices). Benchmark (us/model): mags only
10.8 / 5.5 / 5.6, with spectra copied back 14.8 / 12.2 / 13.4 at N = 1e3 / 1e4 / 1e5.

## Where the time goes (magnitudes only, fraction of the batch's wall span; GPU kernel busy is 86-89 % of it)

| step | N=1e3 | N=1e4 | N=1e5 |
|---|---|---|---|
| nebular kernels (continuum + lines, 3 kernels) | 39.4 % | 48.8 % | 49.2 % |
| dust attenuation (`attenuate_batch`) | 6.4 % | 8.4 % | 8.4 % |
| AGN / line profiles / dust emission / indices / IGM / other | 16.3 % | 19.6 % | 19.5 % |
| mass/lbol weights + observed-frame photometry (not GEMMs) | 3.2 % | 2.1 % | 2.3 % |
| SFH weights (`compute_csp_batch` L912; not a GEMM) | 12.5 % | 1.9 % | 0.9 % |
| **age sum GEMM** (SSP basis x SFH weights) | 2.6 % | 3.2 % | 3.2 % |
| **band-flux GEMM** (band table x spectrum) | 5.5 % | 4.5 % | 2.6 % |
| **GEMMs together** | **8.0 %** | **7.8 %** | **5.8 %** |

Generous upper bound, also counting the SFH-weights kernel, the mass/lbol weight kernel and the observed-frame
photometry kernel as "GEMM-shaped": 23.8 % at N=1e3 (launch-bound, dominated by the weights kernel), 11.8 % at
1e4, 9.0 % at 1e5. The gate holds at every size on both definitions (the generous one is just under 25 % at N=1e3).

## GEMM shapes and achieved speed (FP32 cuBLAS `sgemm`, peak 19.49 TFLOPS)

| GEMM | shape (m x n x k) | N=1e3 | N=1e4 | N=1e5 |
|---|---|---|---|---|
| age sum, NN | 5994 pixels x 2N x 107 ages | 16.4 TFLOPS (84 %) | 17.1 (88 %) | 17.7 (91 %) |
| band flux, TN, 24 chunks of 256 pixels | 159 bands x N x 256 | 5.6 (29 %) | 9.1 (47 %) | 15.4 (79 %) |

(Shapes read from a temporary trace in `accel_blas_gemm`, not committed; times are nsys medians.) The age-sum GEMM is
already at 84-91 % of the FP32 peak; the band-flux GEMM is small and launch-limited below N=1e5.

## Against the predictions

- GEMMs 3-6x faster: not measured (Step 3 not run). Upper limit from hardware: A100 TF32 is 8x the FP32 peak
  (156 vs 19.5 TFLOPS); GEMM times here are 0.2-15 ms per call, so at best they shrink to a few percent of the batch.
- End to end 1.1-1.3x: **not reachable.** Amdahl with the GEMM fraction f = 0.06-0.09: 4x faster GEMMs give
  1.05-1.07x, infinitely fast 1.06-1.09x. Reaching 1.1x would need f >= 0.09 and an infinite speed-up.
- The batch is dominated by the nebular kernels (~49 % at N >= 1e4), then attenuation (8 %): speed work belongs there.

## Not run

Precision emulation (Step 2): TF32/BF16/FP16 rounding of weights, basis and band table against FP64 and FP32
builds. Hardware timing and the experimental flag (Step 3). Other configs (`full_smooth`, `_z`, `_imf`, `_fit`):
`_imf` adds template-sum GEMMs (`gemm_columns`, strided batched), which would raise f; not profiled. Other devices.
Fixed clocks were not set. Profile is from one run per size.
