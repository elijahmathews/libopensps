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
| Compile-time (`TUNE_*`) | named constants | the summation parameters change the rounding of FP sums, so they are gated by the FP32 (1e-5) and FP64 (1e-10) tests of the build | no: they fix array extents, trip counts and the order of sums inside device kernels |
| Run-time (`TUNE_P_*`) | `tuning_get(id)` | no beyond the gates: chunk sizes, and the batch size below which the SFH weights are formed on the host | yes |

Compile-time parameters (what each one is and what it changes: the comments in the module):
`TUNE_SEG_LEN`, `TUNE_PT_PIECE` (SSP operator; A100-tuned guesses, #46), `TUNE_PIX_BLOCK`,
`TUNE_BAND_BLOCK`, `TUNE_GEMM_MIN_COLS`, `TUNE_GEMM_KB`, `TUNE_GEMM_NC_BATCHED`, `TUNE_NCH`,
`TUNE_NCHG`, `TUNE_MBLK`, `TUNE_SMALL_N`, `TUNE_CSP_PT`, `TUNE_BAND_PIX_PER_THREAD`,
`TUNE_IMF_GRAD_CHG`, `TUNE_SSP_STAR_BATCH`, the register tiles of the age sums
(`TUNE_AGE_SUMS_*`) and of the template sums (`TUNE_TEMPLATE_*`).

Run-time parameters:

| Name | Default | Range | Meaning |
|---|---|---|---|
| `SSP_BATCH_CHUNK` | 2048 | 1 to 1e6 | models per chunk of the batched SSP |
| `HOST_SFH_MAX` | 16 | 0 to 1e6 | batches smaller than this form the SFH weights on the host; 0 sends every batch to the device |
| `PINNED_CHUNK_BYTES` | 16 MiB | 64 KiB to 1 GiB | bytes per chunk of the pinned host-device copies (`OPENSPS_PINNED` offload builds) |

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

## Where this is going

Planned, in this order, each on top of the registry without touching the kernels again:

1. A C API to read and set the run-time parameters (a new entry point: a minor version).
2. A sweep tool that measures the parameters on the machine and writes a profile; the sweeps of #46
   for `TUNE_SEG_LEN` and `TUNE_PT_PIECE` (which need the gates run per value).
3. A device query (HIP or CUDA runtime) so that a build for several architectures picks its
   profile, and formula defaults from the device's properties (each cited, see above).
4. Moving the HIP launch parameters (`opensps_hip_kernels.hip`: tile sizes, `__launch_bounds__`)
   and the compile-time tile parameters under run-time variants selected from the profile.
