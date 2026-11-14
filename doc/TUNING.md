# Hardware tuning parameters

Every size in the batched kernels whose best value depends on the machine lives in one module,
`src/core/opensps_tuning.f90`. The kernels read it; nothing else holds such a number. A new tuning
constant goes there, with the machine it was measured on and what it changes.

## Two kinds of parameter

| Kind | Where | Changes the result? | Overridable |
|---|---|---|---|
| Compile-time defaults (`TUNE_*`) | named constants | the summation parameters change the rounding of FP sums, so they are gated by the FP32 (1e-5) and FP64 (1e-10) tests of the build | no: they fix array extents, trip counts and the order of sums inside device kernels |
| Run-time parameters (`TUNE_P_*`) | `tuning_get(id)` | no beyond the gates: chunk sizes, and the batch size below which the SFH weights are formed on the host | yes |

Compile-time constants: `TUNE_SEG_LEN`, `TUNE_PT_PIECE` (SSP operator; A100-tuned guesses, #46),
`TUNE_PIX_BLOCK`, `TUNE_BAND_BLOCK`, `TUNE_GEMM_MIN_COLS`, `TUNE_NCH`, `TUNE_NCHG`, `TUNE_MBLK`,
`TUNE_SMALL_N`, `TUNE_CSP_PT`.

Run-time parameters:

| Name | Default | Range | Meaning |
|---|---|---|---|
| `SSP_BATCH_CHUNK` | 2048 | 1 to 1e6 | models per chunk of the batched SSP |
| `HOST_SFH_MAX` | 16 | 0 to 1e6 | batches smaller than this form the SFH weights on the host (A100: ~0.3 ms per model on the device, ~20 us on the host); 0 sends every batch to the device |
| `PINNED_CHUNK_BYTES` | 16 MiB | 64 KiB to 1 GiB | bytes per chunk of the pinned host-device copies (`OPENSPS_PINNED` offload builds) |

## Overriding a run-time parameter

Set `OPENSPS_TUNE_<NAME>` (an integer; a `K`, `M` or `G` suffix means 1024, 1024^2, 1024^3), for
example `OPENSPS_TUNE_HOST_SFH_MAX=64`. The environment is read once, at the first use. A value that
is not an integer in the legal range is refused with one message on stderr and the default is kept.
Fortran programs may call `tuning_set(id, value, ok)` (it wins over the environment) and
`tuning_reset()`; `tuning_report(unit)` prints every parameter with its value, source and range.

## Where this is going

Planned, in this order, each on top of the registry without touching the kernels again:

1. Device profiles: a table of defaults per architecture (`cc80`, `gfx1201`, ...) chosen from the
   build's `gpu_arch`, so that `HOST_SFH_MAX` and the chunk sizes differ by device.
2. A C API to read and set the run-time parameters (a new entry point: a minor version).
3. A sweep tool that measures the parameters on the machine and writes a profile; the sweeps of
   #46 for `TUNE_SEG_LEN` and `TUNE_PT_PIECE` (which need the gates run per value).
4. Moving the HIP launch parameters (`opensps_hip_kernels.hip`: tile sizes, `__launch_bounds__`)
   under the same registry.
