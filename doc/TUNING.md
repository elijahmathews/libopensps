# Hardware tuning parameters

Every number in the batched kernels and drivers whose best value depends on the machine lives in one
module, `src/core/opensps_tuning.f90`. The kernels read it; nothing else holds such a number.

**Rules for contributors and for coding agents**

1. A chunk size, block size, tile or team size, batch-size crossover or summation chunk is never a
   literal in a kernel or a driver. It is a named parameter of the registry (add it there, with a
   comment that says what it is and what it changes).
2. A value that is right for one device and not for the others is a **profile entry** in the registry
   (below), not a new default and not a new `#ifdef` or `if (flang)` in the source. The default is
   for hardware nobody has measured; change it only with measurements on more than one device.
3. A profile entry needs a measurement: an M-number in "Measured device data" below, with the machine,
   toolchain, command, data and date. No measurement, no entry. Never fill the table from memory
   or from what a datasheet suggests.
4. `tools/check_tuning_literals.py` (test `unit_tuning_literals`) fails when a name of the list in
   the script is defined as a literal outside the registry. Add the names of your new parameters to
   it.
5. Compile-time parameters that change a floating-point sum (the summation parameters below) change
   rounding: run the unit and regression suites in FP64 and FP32 (and the GPU comparison,
   `tools/compare_builds`) before changing one.

## The two kinds of parameter

| Kind | Where | Changes the result? | Overridable |
|---|---|---|---|
| Compile-time (`TUNE_*`) | named constants | the summation parameters change the rounding of FP sums, so they are gated by the FP32 (1e-5) and FP64 (1e-10) tests of the build | not at run time: they fix array extents, trip counts and the order of sums inside device kernels; **at build time**, with `-Dtuning_overrides` (below) |
| Run-time (`TUNE_P_*`) | `tuning_get(id)` | no beyond the gates: chunk sizes, and the batch size below which the SFH weights are formed on the host | yes |

Compile-time parameters (what each one is and what it changes: the comments in the module):
`TUNE_SEG_LEN`, `TUNE_PT_PIECE` (SSP operator; A100-tuned guesses, #46), `TUNE_PIX_BLOCK`,
`TUNE_BAND_BLOCK`, `TUNE_GEMM_MIN_COLS`, `TUNE_GEMM_KB`, `TUNE_GEMM_NC_BATCHED`, `TUNE_NCH`,
`TUNE_NCHG`, `TUNE_MBLK`, `TUNE_SMALL_N`, `TUNE_CSP_PT`, `TUNE_BAND_PIX_PER_THREAD`,
`TUNE_IMF_GRAD_CHG`, `TUNE_SSP_STAR_BATCH`, the register tiles of the age sums
(`TUNE_AGE_SUMS_*`) and of the template sums (`TUNE_TEMPLATE_*`).

The native HIP kernels (`src/accel/hip`) take their tile and block sizes from the registry too: the Fortran
caller passes the `TUNE_HIP_*` values to the wrapper, which instantiates the kernel for that shape (templates)
or launches it with that block size. A kernel is built only for the shapes listed here; a value outside the list
makes the wrapper return "not supported" and the caller runs its portable kernel (or, for the block sizes,
clamps to the legal range), so a wrong override is slow, never wrong. `tools/check_tuning_literals.py` refuses a
`#define` of these sizes in the HIP sources and a `TUNE_HIP_*` parameter that no source reads.

| Parameter | Default | Supported | What it is |
|---|---|---|---|
| `HIP_GEMM_RM` | 8 | 4, 8 | rows per thread of the shared-memory products (template products, free-Z age sums, band fluxes): 16 RM pixels per block |
| `HIP_GEMM_RN` | 4 | 4, 8 | columns (models) per thread: 16 RN per block (and RN x 16 models per tile of the age sums) |
| `HIP_GEMM_KC` | 16 | 16 | templates, ages or pixels staged through shared memory per step |
| `TEMPLATE_KB` | 4 | 4, 8 (HIP) | terms per plain block of the compensated sum (also read by the Fortran kernels): changes the rounding |
| `HIP_FOLD_JB` | 128 | 32 to 128 | models per block of the fold of the point weights |
| `HIP_SMOOTH_PB` / `HIP_SMOOTH_KCH` | 64 / 256 | 64 with 128, 256, 512; 32 with 128, 256 | output pixels per block / window pixels staged per step of the velocity smoothing |
| `HIP_SMOOTH_KB` | 4 | any >= 1 | taps per plain block of the smoothing sum: changes the rounding |
| `HIP_ATT_THREADS` / `HIP_OBS_THREADS` | 256 / 128 | 32 to 256 / 32 to 128 | threads per block of the dust attenuation / the observed-frame interpolation |
| `HIP_WEIGHTS_PIECES` | 4 | 1 to 4 | pieces per block (with 32 models) of the IMF point weights |
| `HIP_BAND_SMALL_N` | 40 | any | batches of fewer models use the one-workgroup-per-(model, band) band kernel |

These defaults were measured on one device (M4, R9700); by rule 2 they stay defaults until a second device is
measured. The legacy kernels of the HIP file keep literal block sizes (the nebular continuum, the cue dense layers
`CUE_BU`/`CUE_BM`/`CUE_BK`, `BF_SMALL_THREADS`, the fixed 128 / 256 threads of the older kernels): they predate the
rule and are the next ones to move.

### Overriding a compile-time parameter at build time (sweeps and tuners)

`-Dtuning_overrides=NAME=VALUE[,NAME=VALUE...]` builds with other values of compile-time parameters, with the
registry's names without the `TUNE_` prefix:

```sh
meson setup b-seg128 -Daccel=omp_gpu -Dgpu_arch=gfx1201 -Dtuning_overrides=SEG_LEN=128,PT_PIECE=16
```

This is how a compile-time parameter is swept (one build per value) and how a tuner will build its variants; it is
not for releases or packages. Meson refuses a name the registry does not define (a typo would otherwise change
nothing and look like a result), a repeated name, and a value that is not a non-negative integer
(`tools/tuning/overrides.py`, tested by `unit_tuning_overrides`). Whether a value is *legal* for its parameter (a
tile that fits the registers, a block that divides the trip count) is not checked: that is what the sweep and the
tests of both precisions decide, so run `unit` and `regression` in FP64 and FP32 for every variant before using a
number. Every build records what it was compiled with: `tuning_report` lists the compile-time parameters with
`default` or `override`, and `benchmark_csp_batch` prints `tuning_ct_<NAME> <value>` for each at the start, so a
sweep's results carry the values they were measured with. `tools/check_tuning_literals.py` (`unit_tuning_literals`)
fails when a compile-time constant of the registry lacks its override pair or its table row.

Run-time parameters:

| Name | Default | Range | Meaning |
|---|---|---|---|
| `SSP_BATCH_CHUNK` | 2048 | 1 to 1e6 | models per chunk of the batched SSP |
| `HOST_SFH_MAX` | 16 | 0 to 1e6 | batches smaller than this form the SFH weights on the host; 0 sends every batch to the device |
| `PINNED_CHUNK_BYTES` | 16 MiB | 64 KiB to 1 GiB | bytes per chunk of the pinned host-device copies (`OPENSPS_PINNED` offload builds) |
| `SHARE_IMF_MIN_BATCH` | 1024 | 0 to 1e6 | batches of at least this many models, some of which share their IMF parameters, compute the IMF weights once per distinct IMF and copy them (M6); 0 shares in every batch |

## How a run-time value is resolved

In this order, a later layer overriding an earlier one:

1. the **default** (the table above);
2. the **device profile**: the entries for the profile in effect. The profile is that of the build's
   `-Dgpu_arch` when it names one target (`cc80`, `gfx1201`; `gfx90a:xnack+` counts as `gfx90a`);
   `OPENSPS_TUNE_PROFILE=<name>` selects one by hand, and is the only way for a build for several
   targets (`cc80,cc90`) to get a profile, since the code does not query the device. An unknown
   name is refused with one message on stderr and no profile applies. A build without offload has
   none;
3. the environment variable **`OPENSPS_TUNE_<NAME>`** (an integer; a `K`, `M` or `G` suffix means
   1024, 1024^2, 1024^3), read once at the first use. A value that is not an integer in the legal
   range is refused with one message on stderr and the layer below stays;
4. `tuning_set(id, value, ok)` in Fortran (tests, programs that tune themselves).

`tuning_report(unit)` prints the profile in effect and every parameter with its value, where it came
from (`default`, `profile`, `env`, `set`) and its range.

## Adding a profile entry

1. Measure it, on the device, with the build the entry is for: sweep the parameter over a range
   that brackets the best value (`benchmark_csp_batch` takes `OPENSPS_BENCH_SIZES=16,64,256,...` for
   batch-size sweeps and the `OPENSPS_TUNE_*` variables set the parameter), at least three
   repeats, interleaved, with `OMP_TARGET_OFFLOAD=MANDATORY`. Check the results do not change
   beyond the gates.
2. Add the measurement to "Measured device data" below (next M-number: machine, toolchain, command,
   data, date, who).
3. Add the triple to the profile table of the registry (`prof_arch`, `prof_id`, `prof_val`,
   `prof_ref`), with the M-number.
4. Run `unit_tuning` and the suites; say in the pull request which device you measured on.

An entry that equals the default is still worth adding: it records that the value was measured on
that device and keeps it when the default moves.

## Measured device data

The only device data in the registry is what is listed here. There is no table of hardware
specifications (compute units, caches, clocks) behind the defaults; if a later step adds one (for a
formula default from the device's properties), each figure needs a citation to the vendor's
documentation or to the output of the vendor's tool on the device, in this section.

**M1. `cc80` (NVIDIA A100), `HOST_SFH_MAX = 16`.** Below this batch size the SFH weights are formed
on the host: on the device each model's SFH integral is one serial thread (~0.3 ms per model on the
A100, the same on the P100), on the host ~20 us per model. Measured in the OpenACC to OpenMP-target
work on the Penn State ICDS A100 node (nvhpc 25.1). The raw data was not kept in the repository;
the value and its reason are recorded in the comment of `HOST_SFH_MAX` as it stood in
`src/physics/opensps_csp_batch.f90` (commit a98d8c3, "perf(gpu): OpenMP target at OpenACC's speed
(#21)", already present in the import commit 561e8e8) and the mechanism in `doc/GPU_RESULTS.md`
("Batches below the host threshold ..."). The crossover depends on the host core and the GPU's
single-thread FP64 speed. This entry should be re-measured with the sweep above when the A100 is
next available.

**M2. `cc80` (NVIDIA A100), `PINNED_CHUNK_BYTES = 16 MiB`.** ~16 MB column chunks of the pinned copy
measured best on the A100 node, PCIe 3 host: large enough to reach the DMA rate (~13.5 GB/s), small
enough to overlap most of the conversion. Recorded in `doc/GPU_RESULTS.md`, "Spectra copy-back"
(`gpu/a100-copyback`), and in the comment of `CHUNK_BYTES` it replaces (commit a98d8c3). Raw
sweep not kept.

**M3. `gfx1201` (AMD Radeon AI PRO R9700), `HOST_SFH_MAX = 160`.** Measured 2026-11-14 on the
desktop of the maintainer: R9700 (gfx1201, 64 CUs, 32 GB), Ryzen 9 9950X host, Fedora 44, LLVM flang
22.1.8, ROCm 7.1.1 (Fedora), FP32 kernels, libopensps at the commit of this change
(`feat/tuning-profiles`). Command: `OPENSPS_BENCH_SIZES=8,16,32,64,128,256,512,1024
OPENSPS_TUNE_HOST_SFH_MAX=<0 or 1000000> OMP_TARGET_OFFLOAD=MANDATORY
benchmark_csp_batch 1024 0 full`, three runs of each setting, median. `0` puts every batch on the
device, `1000000` every batch on the host. Microseconds per model, magnitudes only:

| batch | device | host | host / device |
|---|---|---|---|
| 8 | 498.2 | 363.9 | 0.73 |
| 16 | 187.4 | 139.6 | 0.74 |
| 32 | 102.5 | 81.1 | 0.79 |
| 64 | 58.1 | 48.0 | 0.83 |
| 128 | 34.4 | 30.7 | 0.89 |
| 256 | 20.7 | 24.4 | 1.18 |
| 512 | 13.2 | 23.3 | 1.77 |
| 1024 | 11.2 | 20.8 | 1.85 |

The host path is faster below the crossover, which interpolates (log scale) to about 170 models;
160 is set slightly below it, so that the host path is taken only where it is measured faster. The
spectra column gives the same shape (host faster up to 128, 1.31x slower at 256). The value is
for this GPU with this host CPU; a slower host core moves it down. Not covered: other AMD parts.
Also confirmed on this machine (not an entry, equal to the default): `SSP_BATCH_CHUNK` at batch
10^4, `full_imf`: 256 gives 54.0 us per model, 2048 gives 48.7, 16384 gives 58.4.


**M4. `gfx1201` (AMD Radeon AI PRO R9700), the HIP kernels' tile and block sizes.** Measured 2026-11-15 on the
desktop of the maintainer (Ryzen 9 9950X, Fedora 44, the ROCm 7.2.4 container `tools/containers`, amdflang,
hipcc, FP32 kernels `-Dkernel_precision=single -Daccel=omp_gpu -Dgpu_arch=gfx1201 -Daccel_hip=true`), branch
`perf/forward-bottlenecks`, `rocprofv3 --kernel-trace`, `benchmark_csp_batch 10000 0 full_smooth_z_imf_fit`,
average kernel time per launch (the profiler's noise is ~5%; the clock ramps over the first calls):

| kernel | variant | ms per launch |
|---|---|---|
| template products (`template_columns_kernel`) | 128 x 64 tile (RM 8, RN 4) | 1.12 |
| | 128 x 128 tile (RM 8, RN 8) | 1.78 |
| fold (`fold_multi_kernel`), models per block | 128 | 8.9 (8.9, 9.0) |
| | 64 | 12.1 (11.7, 12.5) |
| | 32 | 23.0 (22.6, 23.5) |
| weights, expm1 as a degree-6 series for |y| < 1/8 | library expm1f / series | 6.78 / 6.57 (not adopted) |

The other sizes (smoothing tile, `HIP_ATT_THREADS`, `HIP_OBS_THREADS`, `HIP_WEIGHTS_PIECES`,
`HIP_BAND_SMALL_N`) are the values the kernels were written with and were not swept; their entries record that,
not a measurement. The raw traces are not kept in the repository.


**M5. `cc80` (NVIDIA A100-PCIE-40GB), the HIP kernels at their defaults.** Measured 2026-11-16 (the A100 agent, Penn State
ICDS; nvhpc 25.1, nvcc 12.6, data-v4, `OMP_TARGET_OFFLOAD=MANDATORY`, `-Daccel_hip=true`, FP32 + cuBLAS), head 28dccbe against its
parent c70ef1b (the OpenMP kernels), microseconds per model, three runs each (agreement ~2%), `benchmark_csp_batch`,
config `full_smooth_z_imf`:

| | N = 1000 | N = 10000 |
|---|---|---|
| forward, parent | 41.4 - 41.9 | 30.7 |
| forward, 28dccbe | 32.8 - 33.6 | 23.2 |
| gradient, parent | 339 | 299 |
| gradient, 28dccbe | 207 - 209 | 184.5 |

(Gradient: two runs each.) With the template products left to cuBLAS (the HIP kernel preempted it before 28dccbe: forward
54 - 55 at N = 1000). This is a check that the R9700 defaults (M4) do not hurt on a second device, not a better value: no
profile entry is added (rule 3: an entry needs a sweep of the parameter on that device).

**M6. `gfx1201` (AMD Radeon AI PRO R9700), `SHARE_IMF_MIN_BATCH` (no profile entry: the default 1024 is from this sweep).**
Measured 2026-11-17 on the desktop of the maintainer (Ryzen 9 9950X, the ROCm 7.2.4 container, FP32 kernels,
`-Daccel=omp_gpu -Dgpu_arch=gfx1201 -Daccel_hip=true`, `OMP_TARGET_OFFLOAD=MANDATORY`), branch
`feat/iso-mods-pr4-shared-imf`, `benchmark_csp_batch 10000 0 <config> 500` with `OPENSPS_BENCH_SIZES=500,1000,2048,4096,10000`
after a warm-up run (the clocks ramp), `OPENSPS_TUNE_SHARE_IMF_MIN_BATCH=0` (every batch shares), sharing off with
`OPENSPS_BENCH_NOSHARE=1`. The configs have all models on one Kroupa IMF through the operator path (`full_imfs`) and
with the five isochrone factors free per model, nine group operators (`full_imfs_mods`). Microseconds per model,
median of three runs (on / off; run-to-run spread ~5%, and up to 15% at N = 500):

| config | N = 500 | 1000 | 2048 | 4096 | 10000 |
|---|---|---|---|---|---|
| `full_imfs` | 47.0 / 44.1 | 29.0 / 30.0 | 26.1 / 24.4 | 22.4 / 22.9 | 20.7 / 23.1 |
| `full_imfs_mods` | 65.5 / 63.7 | 39.5 / 43.7 | 32.5 / 37.4 | 32.3 / 34.2 | 32.4 / 33.9 |

From N = 1000 the sharing is faster in every row but one noisy `full_imfs` size (N = 2048: 26.1 against 24.4, its
three runs 22.0 - 26.1 on and 24.2 - 28.5 off), by 3 - 13%; at N = 500 the two are within the noise. 1024 is the smallest
size tested with a consistent gain, not a fitted crossover. The saving is ~2 us per model (the weights of the one distinct
IMF are a single column of a latency-bound kernel); an earlier version that allocated its scratch at the width of a chunk
(2048 models) instead of the number of distinct IMFs had a fixed cost of ~0.8 ms per call of the age sums (nine calls
with free factors), which made it slower than no sharing up to N = 2048 (1.15x at 500, 1.04x at 2048) and 5% faster at
10^4: sizing the scratch by the distinct count removed the cost. On the host (FP64, 8 threads, that earlier version,
N = 1000, `full_imfs`): 1100 against 1114 us per model, 1%: the weights are not where the CPU's time goes. Not covered: other
devices; the A100 waits for the whole stack of #45.

## Where this is going

Planned, in this order, each on top of the registry without touching the kernels again:

1. A C API to read and set the run-time parameters (a new entry point: a minor version).
2. A sweep tool that measures the parameters on the machine and writes a profile; the sweeps of #46
   for `TUNE_SEG_LEN` and `TUNE_PT_PIECE` (which need the gates run per value).
3. A device query (HIP or CUDA runtime) so that a build for several architectures picks its
   profile, and formula defaults from the device's properties (each cited, see above).
4. Moving the HIP launch parameters (`opensps_hip_kernels.hip`: tile sizes, `__launch_bounds__`)
   and the compile-time tile parameters under run-time variants selected from the profile.
