# GPU results (P100 and A100 sessions)

Running log of the GPU sessions (plan: `doc/GPU_PLAN.md`). Newest status at the top; dated
entries below. The A100 session (2026-10-16) is first; the P100 session's status and entries follow.

## Questions for Elijah

(Decisions that are yours; the agent took the conservative option and kept working.) All five
questions of the A100 session were answered on 2026-10-16:

1. Legacy C API `compute_interpolated_ssp` calls: fix them. Done (argument order and keywords;
   the clashing `fsps_compute_ssp` / `fsps_compute_csp` imports renamed). The legacy layer
   (`FSPS_ENABLE_LEGACY`, not built by meson) still does not compile: it refers to globals that
   no longer exist (`om0`, `ol0`, `h0`, `tiny_logt`, ..., `ntabmax`) and has a shape error at
   `fsps_c_api.f90` ~1900; left as an open problem.
2. Fitting parameters: logmass, logzsol, dust2, dust_ratio, dust_index, the non-parametric SFH
   (logsfr_ratio_i via the SSP-weight matrix), fagn, agn_tau, gas_logz, gas_logu, duste_umin,
   duste_gamma, duste_qpah, zred. `c_batch` now varies exactly these (dust1 = dust_ratio * dust2
   and logmass formed by the caller).
3. `compute_csp_batch` moves the context to the device itself when needed: done.
4. TF32 stays off by default (it is only reachable through `FSPS_CUBLAS_MATH=tf32`).
5. Order of the late requests: accepted.

Session 2 (2026-10-19), new questions (conservative choice taken, work continued):

6. The batch now keeps its spectra in a context-owned buffer between calls (on the device,
   host copy page-locked): needed for device-resident output and to avoid re-allocating and
   re-pinning per call, but it holds nspec x N kernel-precision values (240 MB FP32 / 480 MB
   FP64 at N = 10000) until `fsps_context_remove_from_device` / destroy. Kept resident; should
   there be an explicit release call (or a size cap) for callers that run one huge batch?
7. Page-locking that buffer (`cuMemHostRegister`) is on by default in NVIDIA GPU builds;
   `FSPS_NO_PIN=1` turns it off. Fine as a default on shared nodes?
9. One code path, step 1: OpenACC multicore keeps the old single-call path (its FP32 per-core
   sums put the batch 2e-6 from the all-ages output, over the oracle's 1e-6). Alternatives:
   loosen that oracle to 100 eps(KP) in FP32 builds (the AGENTS.md rule for FP32 tolerances),
   or leave multicore on the old path until OpenMP host threads replace it. Kept the latter.
8. Item 5 (batched SSP generation with IMF parameters per model) was not started: see
   "Next" in `doc/GPU_PLAN.md` for the first step. Keep it next in line?

## One code path, step 1 (2026-10-19, branch `refactor/scenario-as-batch`)

Roadmap priority 2 (`doc/ROADMAP.md`): `compute_csp_scenario` as a batch of one, following
the design note below. Branch from `gpu/a100-call-overhead2` (#17, not yet merged; #14 is),
PR against `develop`. Node `p-gc-3025` (A100-PCIE-40GB), nvhpc 25.1, gcc 13.4.

**What it does.** With `FSPS_SCENARIO_BATCH=1`, a single call with `tage > 0` runs as
`compute_csp_batch` on one model and is packed into `results(1)` exactly as before (spectrum,
lines, magnitudes or -99, indices or -99, mass, L_bol, dust mass, SFR, `mformed = 1`, age).
It stays on the old path for several metallicities (`nzin > 1`), redshift colours,
light-weighted ages, fast mode, and anything the batch rejects (status /= 0: smoothing other
than the device velocity kernel, invalid dust settings). The old path is the default until
the new one passes everywhere; `set_scenario_batch(.false.)` selects it in code (used by the
tests that inspect its internals or use it as the batch's reference).
- Structure: `sfh_weights_core` / `convert_sfhparams` moved from `fsps_csp` to `fsps_sfh`
  (re-exported), and the batch of one is an external procedure
  (`src/physics/fsps_scenario_batch.f90`): `fsps_context` uses `fsps_csp`, and
  `fsps_csp_batch` uses `fsps_context`, so `fsps_csp` cannot use the batch module directly.
  (Lesson: ninja stops without an error message on a Fortran module cycle, and the stale
  library then made the old path look like the new one; confirm the path with
  `NVCOMPILER_ACC_NOTIFY=1`, which lists the batch's kernels.)
- No new kernels and no OpenACC changes (OpenACC freeze respected).

**Correctness.**
- Unit + regression suites with `FSPS_SCENARIO_BATCH=1`: 33/33 in all nine builds (OpenACC
  FP32 / FP64, cuBLAS FP32 / FP64, OpenMP target FP32 / FP64, OpenACC multicore FP32, gfortran
  FP64 / FP32); `test_csp_batch` 166/166. With the old path 33/33 as before.
- OpenACC multicore stays on the old path: through the batch, the single-age vs all-ages oracle
  (`test_physics_oracles`, element-wise within 1e-6 of the peak) differed by 1.4-2.1e-6 there,
  because multicore runs the FP32 reductions as plain per-core sums; the GPU and CPU FP32 builds
  give 4e-8 with either path (the oracle now prints this difference). Conservative choice, see
  Questions 9.
- Case matrix (35 cases), gfortran FP64, new path vs old path in the same build: identical in
  most cases; largest differences 1.2e-12 per pixel (velocity smoothing: the batch applies the
  same sparse operator on the columns), 2.4e-15 (nebular), 7e-15 mag; PASS at `--rtol 1e-10`.
  `csp_tau_allages` (tage <= 0) takes the old path and is identical.

- Case matrix through the new path on the GPU, against the nvfortran CPU build (old path):
  OpenACC FP64 1.6e-11 per pixel, 1.3e-11 mag, 8.5e-12 indices (`--rtol 1e-10`, PASS; the same
  numbers with the old path and as in session 1); OpenACC FP32 4.2e-7 per pixel, 5.1e-7 mag,
  1.1e-5 indices (PASS). (Against the gfortran CPU build the FP64 GPU differs by up to 2e-9 with
  either path: compare with the nvfortran CPU build, as the case-matrix notes say.)

**Single-call time** (`benchmark_csp_batch 0 300 <config>`, us per call; GPU runs alternated
old / new twice):

| | old path | batch of one |
|---|---|---|
| A100, OpenACC FP32, basic | 572-619 | **513-515** (-14%) |
| A100, OpenACC FP32, full physics + smoothing | 1374-1409 | **1100-1151** (-19%) |
| CPU (gfortran FP64, one core), basic | 3585 | 4820 (+34%) |
| CPU (gfortran FP64, one core), full physics + smoothing | 12016 | 8875 (-26%) |

The CPU basic case is slower through the batch (to look at before the batch path becomes the
default: the batch's host loops are written for many models, e.g. the age sums' 4 x 8 tiles).

**Next steps:** all-ages mode (`tage <= 0`) as a batch of `ntfull` columns with a weight
matrix; then redshift colours, light ages, MDF sums; make the batch path the default once it
passes in all builds and the CPU basic case is no slower; then remove the old path.

## Status (A100 session 2, 2026-10-19)

If restarted: read this section, then `git log` of `gpu/a100-call-overhead2` (stack tip).

Node `p-gc-3003` (A100-PCIE-40GB, addressing mode None, 2x Xeon Gold 6248R; nvhpc 25.1, gcc
13.4). All branches start from `gpu/a100-final` (PR #10), PRs open, none merged:

| Branch | On | PR | Content |
|---|---|---|---|
| `fix/asan-csp-batch-leak` | `gpu/a100-final` | #11 | nightly ASan leak in `test_csp_batch` |
| `fix/omp-maps-llvm` | `fix/omp-device-no-malloc` | #13 | AMD 780M fault: audit + `FSPS_OMP_CHECK_PRE` (status there) |
| `fix/omp-flang-component-maps` | `fix/omp-flang-map-lifetimes` | #16 | OpenMP maps by address (flang root causes) |
| `gpu/a100-copyback` | `fix/asan-csp-batch-leak` | #14 | spectra copy-back, device / host outputs |
| `gpu/a100-call-overhead2` | `gpu/a100-copyback` | #17 | per-call overhead (batch N = 1, single call) |

Builds of this session: worktrees `../fsps-wt` (`bgpu32`, `bgpu`, `bgpucb32`, `bgpucb`,
`bomp32`, `bomp`, `bmc32`, `bcpu`, `bcpu32`) and `../fsps-wt3` (`bgpu32`, `bgpu32pg`: gprof);
main tree `build_a100_nightly_asan` (gfortran ASan + UBSan, nightly configuration).
Local runs need `SPS_HOME` unset (it overrides `FSPS_DATA_HOME`).

**1. Nightly AddressSanitizer** (PR #11): `test_csp_batch` never deallocated its allocatable
context after `fsps_destroy` (LeakSanitizer: 2,556,536 bytes in 3 allocations, from `MAIN__`
and `fsps_context_set_pset`). Fixed with `deallocate(ctx)` (also two benchmarks). The nightly
job reproduced locally on the PR #10 tip (gfortran 13.4, `-Denable_stress_testing=true
-Db_sanitize=address,undefined`): leak without the fix; with it unit + regression 33/33
(`--timeout-multiplier 3`) and the stress suite (all phases, `FSPS_STRESS_ITERS=10`,
`OMP_NUM_THREADS=4`) clean.

**Interrupt: OpenMP offload fault on the AMD 780M** (PR #13, details in that branch's
`doc/GPU_RESULTS.md`). Not reproducible on the A100 (no HMM; memcheck, `mem:separate` and
`-Minfo=mp` clean; a removed map crashes nvfortran too). LLVM flang could not offload to NVIDIA
here (AOMP 22 has no NVPTX backend; the LLVM release has no offload runtime). A static audit,
including flang 22's gfx1103 device IR and map tables (AOMP built the project), found no
missing map. New `FSPS_OMP_CHECK_PRE=1` checks presence and attachment of every array the
batch kernels use and stops naming the culprit; to run on the 780M.

**2. Spectra copy-back** (`gpu/a100-copyback`). The batch's spectra live in a context-owned
buffer (`ctx%state%batch_out`): allocated on the device only when a batch outgrows it (x1.5),
its host copy page-locked through the CUDA driver (`src/core/fsps_pinned.c`, resolved with
`dlopen` at run time; `FSPS_NO_PIN=1` disables). Findings:
- registering the buffer alone gained nothing: the OpenACC runtime stages `update self`
  through its own pinned buffers even for registered memory (~4.6 GB/s effective). The copy now
  goes straight into the buffer with `cuMemcpyDtoHAsync` on the OpenACC queue's stream
  (~13.5 GB/s), in ~16 MB column chunks on two queues, the host converting chunk k to WP while
  chunk k+1 is in flight (falls back to `update self` when not pinned);
- nvfortran 25.1: `update self(sp(:, c1:c2)) async(q)` with `c2` a reassigned local issued an
  empty section (`size=5994x0` in the runtime trace); the chunk copy is a separate routine with
  the bounds as arguments;
- new outputs for callers working on the device or in single precision:
  `batch_spectra_device` / `fsps_context_batch_spec_device` (device address, no copy) and
  `batch_spectra_host` / `fsps_context_batch_spec_host` (the page-locked host buffer at kernel
  precision, one DMA, no conversion). Tested in `test_csp_batch` (values through the returned
  addresses equal the copied-back spectra, every group) and `test_c_batch`.
- `benchmark_csp_batch` now touches `spec` once before timing (a fitting loop reuses its
  array; before, the first of 3 repetitions at N = 10000 paid 480 MB of page faults) and times
  the device and host-kernel-precision variants.

  N = 10000, us per model (A100; base = PR #10 code with the same warmed `spec`):

  | | base FP32 | now FP32 | base FP64 | now FP64 |
  |---|---|---|---|---|
  | magnitudes only, basic | 2.87 | 2.97 | 4.10 | 4.10 |
  | + spectra copied back (WP), basic | 17.9 | **9.6** | 27.2 | **11.8** |
  | + spectra at kernel precision on the host, basic | | **4.7** | | **7.7** |
  | + spectra left on the device, basic | | 2.95 | | 4.07 |
  | magnitudes only, full physics + smoothing | 7.10 | 6.89 | 8.49 | 8.61 |
  | + spectra copied back (WP), full physics + smoothing | 21.5 | **13.4** | 32.7 | **16.2** |

  Copy-back cost (spectra minus magnitudes-only), basic: FP32 15.1 -> 6.6 us (-56%), FP64
  23.1 -> 7.7 us (-67%); at kernel precision 1.7 / 3.6 us (the DMA). What remains of the WP
  path in FP32 is the host conversion to double (single thread, ~3-5 us per model).

**3. Per-call overhead** (`gpu/a100-call-overhead2`). Nsight Systems at N = 1 (full physics):
per batch call ~67 uploads, ~13 downloads, ~54 stream synchronizations, 22 launches; launches
cost 3.8 us, but each small pageable transfer ~8 us plus a ~10 us synchronization. The data
region alone uploaded 28 small arrays and 24 array descriptors. Changes:
- per-model inputs (15 arrays) and outputs (6) are sections of one array per type (`ipk`,
  `rpk`, `lpk`, `rout`), the nebular setup's 13 host-filled arrays likewise (`pack_nebular`),
  and the nebular data region is skipped without nebular emission: ~67 -> 29 uploads, ~13 -> 7
  downloads per call;
- the SFH-weight kernel is split, so given or host-computed weights (N < 16) do not pay the
  device call's six section-descriptor transfers;
- the indices use the resident wavelengths in OpenACC builds (no 48 kB upload per call); the
  photometry uploads redshift and distance in one array and reads the Vega magnitudes from the
  resident copy;
- `compute_csp_scenario`: gprof (CPU sampling is blocked on the node, `perf_event_paranoid` 2;
  Nsight Compute too) showed ~0.65 ms of host time per call in `apply_nebular_emission`,
  interpolating the CLOUDY grids at (gas_logz, gas_logu) for every pixel and nebular age
  (~60,000 interpolations) and uploading the 0.5 MB result; that loop now runs on the device,
  where the grids are. What is left of the host time per call is spread thin (one-time setup
  dominates the profile).

  us per call (A100, FP32 OpenACC; N = 1 is one batch call of one model):

  | | basic, before | basic, now | full physics + smoothing, before | now |
  |---|---|---|---|---|
  | batch, N = 1 | 606 | **427** | 1067 | **828** |
  | one `compute_csp_scenario` call | 678 | 625 | 2169 | **1393** |
  | batch, N = 100, per model | 17.7 | 16.4 | 26.1 | 25.0 |

  ("Before" is the copy-back branch; the basic single call is within run-to-run noise, 580-680
  us. Final stack-tip numbers at N = 10000 are in the copy-back table above: full physics with
  spectra copied back 21.5 -> 13.4 us per model FP32, 32.7 -> 16.2 FP64.)

  Not done: the scratch arrays created per call (9 descriptor uploads) could also live in the
  context; the single call still makes ~40 small transfers; and the async kernel queue of the
  first session (crashed with a race) was not retried.

**4. Design note: the single call as a batch of one** (below; no code).

**5. Batched SSP generation with IMF parameters per model**: not started (the time went to the
AMD interrupt). First step in `doc/GPU_PLAN.md`, "Next".

**Validation** (stack tip, all nine builds: OpenACC FP32 / FP64, cuBLAS FP32 / FP64, OpenMP
target FP32 / FP64 with `OMP_TARGET_OFFLOAD=MANDATORY`, OpenACC multicore FP32, gfortran FP64 /
FP32): unit + regression 33/33 and `test_csp_batch` 166/166 in all nine (multicore after the
fix below). Case matrix (35 cases) against the nvfortran CPU build: OpenACC FP32 worst 4.2e-7
per pixel, 5.1e-7 mag, 3.0e-6 indices; OpenACC FP64 1.6e-11 per pixel, 1.3e-11 mag (`--rtol
1e-10`): PASS, the same as in session 1. Found by the validation: the nebular grid
reduction's device loop segfaulted under OpenACC multicore (reading the grids through ctx,
like `compute_igm_transmission`'s loop in session 1); multicore keeps the host loop.

Follow-up for the OpenMP mapping work (`fix/omp-flang-component-maps`, see that branch): the
batch output buffer (`batch_out_reserve` / `batch_out_release`) still names `o%spec` in OpenMP
enter / exit data; when both lines meet it must use `omp_data_alloc` / `omp_data_delete`.

### Design note: `compute_csp_scenario` as a batch of one (no code yet)

Goal: one implementation of the CSP physics. `compute_csp_scenario(ctx, pset, nzin, results,
status)` becomes a thin wrapper that builds a batch from `pset`, calls `compute_csp_batch`
and packs the outputs into `results(:)` (`COMPSPOUT`). Today the two paths are separate
implementations of the same physics. They agree to 6.7e-16 (FP64, same SFH weights) and are
tested against each other (`test_csp_batch`), but every feature is written twice and the
single call carries its own per-call costs.

**Mapping**
- `tage > 0` (one output): a batch of one, `psets = [pset]`. The outputs map directly:
  `spec`, `mags`, `mass`, `lbol`, `mdust`, `sfr`, `emlines`, `indx`; `mformed = 1`;
  `age = log10(tage) + 9`.
- `tage <= 0` (all ages, `n_outputs = ntfull`): a batch of `ntfull` models that share `pset`,
  with an SSP-weight matrix. Column i holds the SFH weights at age i, normalised per unit mass
  formed over the full history (as `integrate_csp_step` does). After the batch, scale `mass`,
  `lbol` and the spectrum by the fraction formed by then (`mass_frac`), as
  `apply_post_processing` does. `sfr` per output comes from the host
  (`get_sfh_properties_at_age`), since the batch returns NaN with given weights.
- `tage = -99` with tabular SFHs (`sfh` 2, 3): one model at `max(sfh_tab age)`. This needs
  the batch to accept `tage = -99` with `sfh = 2/3`; today it computes weights only for
  `tage > 0` or a given matrix. Simplest is to give it the matrix from the host.
- The batch's host SFH path below 16 models already reproduces the single call's weights
  exactly, so N = 1 keeps today's FP64 agreement.

**What blocks it** (each is a `status = 2` path of the batch today, or a semantic difference):
1. `redshift_colors = 1`: magnitudes at the redshift of each output's age, with the spectrum
   and IGM at `pset%zred`. The batch's per-model `zred` also moves IGM and the observed frame,
   so it would need a "magnitudes-only redshift" per column (`zred_col` without IGM).
2. `compute_light_ages = 1`: light-weighted ages replace magnitudes and indices. Not in the
   batch; it needs a second age-weighted sum (one more column per model in the age sums).
3. Several metallicities summed (`nzin > 1`: the MDF / `pmetals` modes of `zcontinuous`
   2 and 3). The batch interpolates between two bases per model (`zcontinuous = 1`) but does
   not sum all `nz` bases with MDF weights. Possible as a per-model weight over metallicities
   in `age_sums_z` (nz instead of 2 terms). Nebular emission with `nzin > 1` is rejected by
   both paths today.
4. Invalid dust settings (`uvb < 0`, `wgp1/wgp2 < 1`): the batch returns status 2, while
   the single call's attenuation routine silently returns early for them. The wrapper has to
   keep the old behaviour, or both should report it. Witt & Gordon dust itself is supported
   on the device (`compute_attenuation_curve_point`, `dust_type = 3`).
5. Smoothing other than fast velocity smoothing: LSF smoothing (`smooth_lsf`), the exact
   path (`smoothspec_fast = 0`), and a `sigma_smooth` that differs per model (irrelevant for a
   batch of one). The single call falls back to the host per output for these.
6. `fast_mode` (skip magnitudes and indices): easy, it just means the optional outputs are
   absent.
7. Outputs per age in all-ages mode (107 columns) are cheap for the batch. The single call's
   per-output host loop for magnitudes (when `dev_mags` is false) disappears, which removes
   blockers 1 and 5 only if the batch covers them.
8. Numerical identity of the regression references: the scenario is the reference of the
   regression and parameter-matrix fixtures. The batch sums ages in a tiled order (identical
   to the scenario only up to FP64 rounding, 1e-16..1e-11) and adds nebular emission after
   the age sums instead of to each SSP. The FP64 CPU fixtures would move by ~1e-11 and need a
   regenerated reference or a looser `--rtol` on the scenario cases.
9. Builds without a device: the batch runs on the host in CPU builds. Its N = 1 host cost
   has not been measured against the single call's (3.2 ms basic, one core). The OpenMP
   target build would gain the single call on the device, which it does not offload today.
10. Interfaces: `results(:)` reallocation semantics (callers reuse `results`), the legacy C
    layer, and the Python bindings expect `COMPSPOUT`. The wrapper keeps them.

**Order** (each step keeps both paths and checks one against the other):
(a) the wrapper for `tage > 0` with none of 1-6 active; fall back to the old path otherwise;
(b) all-ages mode via the weight matrix; (c) `redshift_colors`, light ages, the MDF sum; (d)
LSF / exact smoothing on the device; (e) remove the old path.
Gains: one physics implementation, and the single call at the batch's N = 1 cost (1.1 ms
full physics after the packing work, vs 1.2-1.4 ms for the single call after this session's
nebular change, 2.3 ms before).


## Summary: P100 and A100

Nodes: P100-PCIE-12GB with 2x Xeon E5-2680 v4 (Broadwell), nvhpc 25.1; A100-PCIE-40GB with 2x
Xeon Gold 6248R (Cascade Lake), nvhpc 25.1 (the newest installed; CUDA 12.6 runtime on a CUDA
13.0 driver). The P100 numbers are from the P100 session's code (branch `gpu/p100-pset-resident`
plus the merged `develop` items it had); the A100 numbers from this session's branch tip
(`gpu/a100-tf32`), which also includes the A100 session's kernel and feature work, so the
columns compare hardware and software together. CPU references run one core of each node.

**Validation.** All configurations pass the unit + regression suites (33 tests) and the case
matrix against an nvfortran CPU build on the A100: OpenACC FP64 / FP32, cuBLAS FP64 / FP32,
OpenMP target FP64 / FP32 (first device run; `OMP_TARGET_OFFLOAD=MANDATORY`), nvfortran and
gfortran CPU FP64 / FP32. FP64 GPU = CPU to 1.6e-11 per pixel; FP32 GPU vs FP64 CPU 4.2e-7 per
pixel, 5e-7 mag.

**`benchmark_phases 5 200`** (MIST + MILES + DL07; seconds per SSP grid, ms per CSP call):

| | P100 FP64 | P100 FP32 | A100 FP64 | A100 FP32 | A100 OpenMP tgt FP64 / FP32 | CPU core, P100 node | CPU core, A100 node |
|---|---|---|---|---|---|---|---|
| SSP grid (s) | 0.270 | 0.270 | 0.072 | 0.069 | 0.135 / 0.115 | 0.45 | 0.145 |
| CSP call (ms) | 0.67 | 0.67 | 0.69 | 0.59 | 3.2 / 6.8 | 4.8 | 3.2 |
| CSP call, nebular (ms) | 2.7 | 2.4 | 1.75 | 1.59 | 13.0 / 16.6 | 46 | 13.0 |

(SSP grid: the A100 gain over the P100 is mostly the exact IMF integrals merged into `develop`
since, which took the Romberg host time out, and the faster host core; the single CSP call is
host- and launch-bound on both GPUs. The OpenMP-target build offloads only the template
product and the batch's two products, so its CSP call runs on the host.)

**Batched CSP, `benchmark_csp_batch 10000 200`** (us per model, magnitudes only; basic
configuration: tau model, dust1/dust2, DL07 dust emission; P100 FP64 rows are from the first
version of the batch, before the kernel work):

| N | P100 FP32 OpenACC | P100 FP32 cuBLAS | A100 FP32 OpenACC | A100 FP32 cuBLAS | A100 FP64 OpenACC | A100 FP64 cuBLAS |
|---|---|---|---|---|---|---|
| one `compute_csp_scenario` call | 660 | 660 | 634 | 568 | 648 | 586 |
| 1 | 1688 | 1076 | 1128 | 1006 | 1230 | 983 |
| 100 | 31 | 24 | 19.0 | 17.0 | 20.6 | 16.5 |
| 1000 | 16 | 9.4 | 4.1 | 3.6 | 4.5 | 3.8 |
| 10000 | 14.6 | 8.1 | **2.9** | **2.2** | 4.1 | 2.3 |
| 10000, spectra copied back | | 31.4 | 19.4 | 18.7 | 29.1 | 28.0 |

**Full physics** (`benchmark_csp_batch 10000 200 full_smooth`: dust1 0.5, dust2 0.3, DL07,
nebular emission, IGM at z = 0.5, AGN fagn 0.1, spectral indices, sigma_smooth 150 km/s; A100
only, the P100 batch did not support these):

| N | A100 FP32 OpenACC | A100 FP32 cuBLAS | A100 FP64 OpenACC | A100 FP64 cuBLAS |
|---|---|---|---|---|
| one `compute_csp_scenario` call | 2179 | 2315 | 2268 | 2241 |
| 1 | 1538 | 1521 | 1505 | 1416 |
| 100 | 27.5 | 25.4 | 29.1 | 25.1 |
| 1000 | 8.4 | 8.0 | 9.2 | 8.6 |
| 10000 | **7.2** | **6.5** | 8.7 | 7.1 |
| 10000, spectra copied back | 23.9 | 23.1 | 34.6 | 33.0 |

**OpenMP target offload** (end of the session, branch `gpu/a100-omp-batch`): the whole batch on
the device with the same loops as OpenACC; N = 10000, us per model, OpenMP / OpenACC: FP32
basic 3.6 / 2.9, FP32 full physics 7.8 / 7.0, FP64 basic 4.7 / 4.2, FP64 full physics 9.5 / 8.9
(0.3-1 ms more per call at small N). Before: 833 / 3788 us (host fallback).

**Against the forecasts** (A100): SSP grid ~0.1 s -> 0.07 s; single CSP call with full physics
~2.5 ms -> 2.2-2.3 ms; batched CSP per model ~8.5 us FP32 / ~13 us FP64 -> 7.2 / 8.7 us
(OpenACC; 6.5 / 7.1 with cuBLAS); the plan's ~4 us (FP32, basic configuration) -> 2.9 us.
Batched SSP+CSP per model was not measured (no batched SSP generation yet). FP64 is close to
FP32 on the A100 (FP64 at half the FP32 rate, and much of the batch is memory- and
transcendental-bound in FP64 anyway); that will not carry over to consumer GPUs.

**What remains** (also in `doc/GPU_PLAN.md`, "Next"): (1) done later in the session: the whole
batch now runs with OpenMP target offload (3.6 us per model FP32 basic, 7.8 full physics, vs
2.9 / 7.0 with OpenACC; Status, branch `gpu/a100-omp-batch`); next there, the single CSP call and
the SSP stages, and a run on an AMD or Intel GPU; (2) per-call overhead, ~1 ms at N = 1 and the
single call (launches, small pageable transfers, host set-up); (3) spectra copy-back (pageable
D2H at ~2.4 GB/s on this node: 16 us per model at N = 10000; pinned or device-resident output);
(4) on GPUs with slow FP64, the FP64 exponentials of the nebular continuum and attenuation.
TF32 is not usable for the age sums at this project's tolerances (net 3e-4 mag) and gains only
3-10%; FP64 tensor cores change nothing, numerically or in speed, at these product sizes.

## Status (A100 session, 2026-10-16)

If restarted: read this section, then `git log` of the newest `gpu/a100-*` branch.

**Branches** (each stacked on the previous one, from `develop` 9dc0c19; merge in this order):
`gpu/a100-cublas-threshold` -> `gpu/a100-batch-nebular` -> `gpu/a100-batch-smoothing` ->
`gpu/a100-batch-neb-fast` -> `gpu/a100-batch-metallicity` -> `gpu/a100-fp32-kernels` ->
`gpu/a100-batch-pipeline` -> `gpu/a100-tf32` -> `gpu/a100-call-overhead` -> `gpu/a100-omp-batch`
-> `gpu/a100-omp-portability` -> `gpu/a100-final` (tip). Every branch was validated in all ten builds before its commit, except
`gpu/a100-call-overhead` (dev builds only; its successor `gpu/a100-omp-batch` was validated in
full).

**Setup**
- Node `p-gc-3009`, NVIDIA A100-PCIE-40GB, driver 580.178.04 (CUDA 13.0 driver API), 2x Xeon
  Gold 6248R (Cascade Lake, 3.0 GHz), 12 cores allocated. Slurm job 56476035, 9 h.
- Compiler: `module load nvhpc/25.1` (nvfortran/nvc 25.1-0, CUDA 12.6). `module avail nvhpc`
  lists 21.9, 23.1 and 25.1; the driver could run HPC SDK 26.x (CUDA 13), but 25.1 is the newest
  installed and supports cc80. CPU references: nvfortran 25.1 without accel (`build_a100_nvcpu`,
  `build_a100_nvcpu32`) and gfortran 13.4 (`module load gcc/13.4.0`; `build_a100_cpu`,
  `build_a100_cpu32`).
- Builds (all `--buildtype=release`, GPU ones `-Dgpu_arch=cc80`): `build_a100_gpu` / `gpu32`
  (OpenACC FP64 / FP32), `build_a100_gpucb` / `gpu32cb` (`-Dcublas=true`), `build_a100_omp` /
  `omp32` (`-Daccel=omp_gpu`, run with `OMP_TARGET_OFFLOAD=MANDATORY`). The P100 session's
  `build_*` directories target cc60 and are not used.

**Done**
- Step 0: the P100 session's uncommitted change (cuBLAS band product only for >= 16 spectra)
  moved onto branch `gpu/a100-cublas-threshold` from `develop` (9dc0c19). (A stale worktree of
  `develop` from an earlier session, whose directory no longer existed, was pruned.)
  `test_csp_batch` now also runs a 24-model batch: the 12-model batch never reached the
  many-spectra band kernels (OpenACC `band_fluxes` switches at 16 spectra too) nor, with the
  threshold, the cuBLAS product.
- Item 1 (all configurations validated on the A100, branch `gpu/a100-cublas-threshold`):
  unit + regression 32/32 in all ten builds (OpenMP ones with `OMP_TARGET_OFFLOAD=MANDATORY`:
  the first device run of the OpenMP target code); `test_csp_batch` 32/32 everywhere. Case
  matrix (35 cases) against `build_a100_nvcpu`, worst case:

  | Build | spec (912 A-2.5 um) | spec (all) | mag | indx | lines | result |
  |---|---|---|---|---|---|---|
  | OpenACC FP64 (`--rtol 1e-10`) | 1.6e-11 | 3.4e-11 | 1.3e-11 | 8.5e-12 | 3.3e-16 | PASS |
  | cuBLAS FP64 (`--rtol 1e-10`) | 1.6e-11 | 3.4e-11 | 1.3e-11 | 8.5e-12 | 3.3e-16 | PASS |
  | OpenMP target FP64 (`--rtol 1e-10`) | 2.2e-16 | 7.1e-15 | 1.4e-14 | 2.4e-16 | 0 | PASS |
  | OpenACC FP32 | 4.2e-7 | 4.2e-7 | 5.1e-7 | 3.0e-6 | 3.5e-7 | PASS |
  | cuBLAS FP32 | 4.2e-7 | 4.2e-7 | 5.1e-7 | 3.0e-6 | 3.5e-7 | PASS |
  | OpenMP target FP32 | 4.2e-7 | 4.4e-7 | 1.7e-7 | 3.4e-6 | 3.5e-7 | PASS |

  (OpenMP FP64 is near bit-identical to the CPU because in that mode only the template product
  and the batch kernels are offloaded; the OpenACC FP64 builds run the per-star SSP kernel and
  device dust and nebular kernels.)

- Nebular emission, AGN dust emission, spectral indices and emission-line luminosities in
  `compute_csp_batch` (branch `gpu/a100-batch-nebular`, stacked on `gpu/a100-cublas-threshold`):
  per model, as `compute_csp_scenario` does to each young SSP (EUV escape, continuum and lines
  from the CLOUDY grid in gas_logz / gas_logu, FP64 log space), summed over ages; line
  luminosities attenuated as in `fsps_dust` (birth cloud, diffuse dust at the line wavelength),
  their absorbed part added to the energy balance when lines are not in the spectrum; AGN
  after the IGM. `test_csp_batch` 80/80 in all ten builds (new groups: nebular; IGM + nebular
  + AGN; AGN; lines not in the spectrum). Worst differences against `compute_csp_scenario`,
  FP32 GPU: spectra 5.4e-7, magnitudes 5.3e-7, indices 4.3e-6, lines 0 (identical); FP64 GPU:
  spectra 8.6e-11, magnitudes 4.1e-12, indices 6.8e-11, lines 3.3e-11. Suites 32/32 everywhere.
  Also fixed: `compute_spectral_indices_columns` crashed ("partially present") when the output
  array had more columns than spectra.
- First full real-world benchmark (`benchmark_csp_batch 10000 200 full`: dust1 0.5, dust2 0.3,
  DL07, nebular, IGM at z = 0.5, fagn 0.1, indices; us per model, magnitudes + indices):

  | N | OpenACC FP32 | cuBLAS FP32 | OpenACC FP64 | cuBLAS FP64 |
  |---|---|---|---|---|
  | one `compute_csp_scenario` call | 2230 | 2221 | 2123 | 2339 |
  | 1 | 1423 | 1332 | 1387 | 1327 |
  | 100 | 33.6 | 30.9 | 35.2 | 30.7 |
  | 10000 | 14.4 | 13.0 | 16.2 | 14.0 |
  | 10000, spectra copied back | 31.0 | 29.6 | 42.7 | 40.0 |

- Velocity smoothing (branch `gpu/a100-batch-smoothing`, stacked): `build_velocity_kernel`'s
  sparse operator (the default `smoothspec_fast = 1` path of `apply_smoothing`) cached with the
  redshift grid data and applied on the device to all columns (`smooth_columns`): in
  `compute_csp_batch` (sigma_smooth shared by the batch, after AGN, before magnitudes and
  indices) and in `compute_csp_scenario`, where it replaces the per-output host loop (so
  magnitudes stay on the device with smoothing). Case `csp_smooth`, GPU vs nvfortran CPU (host
  smoothing): FP64 1.1e-15 per pixel, 1.4e-14 mag; FP32 1.7e-7, 2.5e-7 mag. `test_csp_batch`
  (new group: z = 0.5, IGM, nebular, AGN, sigma_smooth = 150 km/s) 90/90, suites 32/32, case
  matrix PASS, all builds.

- Nebular continuum speed-up (branch `gpu/a100-batch-neb-fast`, stacked): the kernel did one
  FP64 `10**x` per (pixel, model, age), 47 nebular ages. Now per (pixel, model) and CLOUDY age
  interval: the grid's (Z, U) interpolation written out (a device call with the 4-d grid per
  interval cost 53 -> 41 ms), the upper endpoint reused as the next lower one, and 10**lg as a
  geometric progression in age (the log-age grid is uniform, so wa steps by a constant inside an
  interval): two FP64 exponentials per interval, one multiply-add per age against precomputed
  C = SFH weight * Q, interval sums in FP64. Full configuration, FP32, N = 10000: kernel
  81 -> 21 ms; 14.4 -> 9.1 us per model. Accuracy unchanged (FP32 spectra vs
  `compute_csp_scenario` 6.5e-7, FP64 1.0e-11). Tests 90/90, suites 32/32, all builds. Nsight
  Compute confirmed unavailable (`ERR_NVGPUCTRPERM`); kernels were tuned from Nsight Systems
  timings and experiments.

- Per-model metallicity (branch `gpu/a100-batch-metallicity`, stacked): with the SSP bases of
  all metallicities resident (`fsps_context_update_ssp_basis(..., nz)`), each model is
  interpolated in log Z at its logzsol between the bracketing bases (`compute_interpolated_ssp`'s
  simple Z interpolation: linear in spectra, mass and log L_bol, not clamped): age sums with
  per-model bases (`age_sums_z`, portable kernel, also used in cuBLAS builds), mass and L_bol
  from the interpolated SSPs, and nebular emission from them (Q interpolated, being linear in
  the spectrum; EUV rows from the interpolated basis). New test group against
  `compute_csp_scenario` on each model's interpolated SSP (z = 0.5, IGM, nebular, AGN): FP32
  spectra 4.5e-7, mags 2.6e-7, lines 2.6e-7, indices 2.3e-6; FP64 1e-11. 96/96 and suites 32/32
  in all builds. Cost: none measurable (full configuration, FP32, N = 10000: 8.3 us per model
  with 12 resident bases vs 8.5 with one).
- Full configuration with smoothing (`benchmark_csp_batch 10000 50 full_smooth[_z]`, FP32
  OpenACC, A100): 8.9 us per model (8.6 with per-model metallicity); N = 100: 29 us; N = 1:
  1.7 ms; one `compute_csp_scenario` call 2.4 ms.

- FP32 kernels (branch `gpu/a100-fp32-kernels`, stacked), full configuration with smoothing,
  FP32 OpenACC, N = 10000: magnitudes assembled on the device (`magnitudes_from_fluxes`; the
  host loop was ~20 ms per call); age sums register-tiled (4 pixels x 8 models per thread, one
  running sum per (pixel, model) restarted at it_tesc: same order, identical results), 10 -> 4 ms;
  the AGN kernel reuses the diffuse transmission of the attenuation pass, 4.7 -> 0.6 ms.
  8.9 -> 7.1 us per model; GPU kernels 57 ms per call of ~71. Tests 96/96, suites 32/32.

- Fitting-pipeline batch (branch `gpu/a100-batch-pipeline`, stacked):
  - `ssp_weights(ntfull, N)`: a precomputed SSP-weight matrix instead of SFH parameters
    (non-parametric SFHs; tage not used, SFR NaN). Tested in every batch group with the
    scenario's own weights fed back: FP32 spectra <= 6.5e-7, FP64 <= 4.7e-13, mass/L_bol 1e-16.
  - Per-model zred and igm_factor: IGM per model on the device (`igm_tau`, factored out of
    `compute_igm_transmission`, which now calls it: same results), observed-frame resampling per
    (pixel, model) on the device, distance term per model, shared filters. New test group
    (z = 0 ... 3, IGM, nebular, AGN): FP32 spectra 6.5e-7, mags 2.5e-7; FP64 1e-11.
  - C API `fsps_context_compute_csp_batch(handle, n, n_params, names, values, ssp_weights, mags,
    spec, mass, lbol, mdust, sfr, emlines, indx, status)` (`include/fsps.h`): one array of n
    doubles per named parameter on top of the context's parameters; builds the SSP basis on
    demand (all metallicities if logzsol is among the names); SSP parameters give status 2,
    unknown names status 3 (message via `fsps_get_last_error`). New C test `c_batch`
    (regression suite): 6 models with tau, tage, dust2, gas_logu, zred, logzsol varied through
    the arrays = the same models set one at a time (identical), plus the error statuses.
  - nvfortran 25.1 bug found: with `-acc=gpu`, a disassociated pointer passed to an optional
    assumed-shape dummy can be reported as present (reproduced in a 20-line program: absent
    without `-acc`, present with it, for the fourth of four optional arguments). The C entry
    point branches explicitly instead.
  - Bug found (AddressSanitizer, gfortran `-Db_sanitize=address` build of the C test): the batch
    sized its SFH scratch from the context's CSP workspace, which only exists after a
    `compute_csp_scenario` call or `move_to_device`; a context used only through the batch (the
    C path in CPU builds) overflowed the heap. Now sized from the ages. Both batch tests are
    clean under ASan.
  - Validation: suites 33/33 (with `c_batch`), batch test 146/146 in all ten builds; case matrix
    PASS; nvfortran CPU outputs unchanged (`--rtol 1e-14`) by the IGM refactoring.
  - Also: the parametric batch's FP64 difference to `compute_csp_scenario` (8.6e-11, open problem
    below) is the device-computed SFH weights: with the scenario's weights as a matrix the
    batch agrees to 6.7e-16.

- Item 3, TF32 and FP64 tensor cores (branch `gpu/a100-tf32`, stacked): cuBLAS math-mode switch
  `FSPS_CUBLAS_MATH` (`tf32`: TF32 tensor cores for the FP32 products; `pedantic`: no tensor
  cores; unset: cuBLAS default, i.e. FP32 units for SGEMM and FP64 tensor cores for DGEMM on
  the A100), and `generate_opensps_fixtures --batch`, which recomputes every single-age CSP case
  (25 of the 35) through `compute_csp_batch` on 32 copies of the model (enough for the cuBLAS
  products) so the case matrix measures the net error of the whole batch pipeline
  (`FIXTURE_ARGS=--batch tools/compare_builds/run_cases.sh ...`). Net error against the FP64
  nvfortran CPU matrix (scenario path), worst over all cases:

  | Batch, cuBLAS | spec (912 A-2.5 um) | spec (all) | mag | indx | lines | result |
  |---|---|---|---|---|---|---|
  | FP32, default math | 4.2e-7 | 4.2e-7 | 3.3e-7 | 1.1e-5 | 3.5e-7 | PASS |
  | FP32, TF32 | 6.8e-4 | 6.8e-4 | 3.1e-4 | 6.4e-3 | 3.5e-7 | FAIL (all 25 batch cases) |
  | FP64, default (FP64 tensor cores) | 1.6e-11 | 1.2e-10 | 1.3e-11 | 2.3e-11 | 1.2e-12 | PASS (`--rtol 1e-10`) |
  | FP64, pedantic (no tensor cores) | 1.6e-11 | 1.2e-10 | 1.3e-11 | 2.3e-11 | 1.2e-12 | PASS |

  TF32 rounds both inputs to 10-bit mantissas (relative 2^-11 ~ 4.9e-4); in the age sums a few
  ages dominate each pixel, so the input rounding passes into the spectra almost unchanged
  (4.8e-4), then into magnitudes (2-3e-4) and indices. The band product alone tolerates it (the
  all-ages case, where only the magnitudes use TF32: 3.1e-7 mag). TF32 is therefore not usable
  for the age sums under this project's tolerances (FP32 ~1e-6, 1e-5 mag); 3e-4 mag is below
  typical photometric errors, but it would have to be an explicit, documented choice.
  FP64 tensor cores (DMMA) are IEEE FP64 FMAs: identical results to pedantic.
  With `FSPS_CUBLAS_MATH=tf32` the FP32 cuBLAS build fails `test_csp_batch` (66/146 pass); FP64
  is unaffected; default and pedantic pass 146/146.
  Speed (`benchmark_csp_batch 10000 200`, us per model at N = 10000, magnitudes only):

  | cuBLAS build | default | tf32 | pedantic |
  |---|---|---|---|
  | FP32, basic | 2.2 | 1.9 | 2.2 |
  | FP32, full physics + smoothing | 6.5 | 6.3 | 6.4 |
  | FP64, basic | 2.3 | 2.3 | 2.3 |
  | FP64, full physics + smoothing | 7.1 | 7.1 | 7.1 |

  The products are a small part of a call after the kernel work (cuBLAS SGEMM age sums ~1 ms of
  ~22 ms at N = 10000, basic), so TF32 gains 3-10% for 1000x the error, and FP64 tensor cores
  give no measurable gain.

- Item 4: final benchmarks of every build (`benchmark outputs in the session scratch directory`, numbers in the Summary above) and the
  write-up (Summary section; `doc/GPU_PLAN.md`, "Next: after the A100 session").

- Per-call overhead, started (branch `gpu/a100-call-overhead`, paused for the OpenMP port): a
  batch of N = 1 spends ~0.6 ms in kernels of ~1.9 ms per call (full physics); the SFH-weight
  kernel alone was 314 us (one serial device thread). Below 16 models the SFH weights are now
  computed on the host (as the single call does): 1720 -> 1500 us per call at N = 1, and
  small FP64 batches agree with `compute_csp_scenario` to 6.7e-16 (8.6e-11 before). Validated
  in the dev builds only (batch test 146/146 FP64 and FP32); next there: async kernel queue
  (~20 synchronous launches per call), fewer small transfers.

- OpenMP target offload of the whole batch (branch `gpu/a100-omp-batch`, stacked; "Next" item 1
  of `doc/GPU_PLAN.md`):
  - Device residency: an OpenMP twin of every OpenACC enter / exit / update directive of the
    context (`fsps_context_move_to_device`, `remove_from_device`, `update_ssp_basis`, the
    settings sync, the batch caches), `!$omp declare target` beside each `!$acc routine seq`
    (63 device routines), and `target data` regions beside the OpenACC data regions. nvfortran
    attaches the pointer components of `ctx%state` correctly (checked in a small program first).
  - Kernels: the OpenMP directive on the same loop as the OpenACC one (36 kernels in
    `fsps_csp_batch.f90`, `fsps_batch_photometry.f90` and the indices); the two separate OpenMP
    copies of the age sums and band fluxes are gone. Gang + inner vector-reduction loops map to
    `target teams distribute` + inner `parallel do reduction`.
  - Fixed on the way: (1) the band-flux kernel as `teams distribute` over (band, block) with an
    inner `parallel do` ran 8x slower than OpenACC (56 vs 6.7 ms); one combined
    `target teams distribute parallel do collapse(3)` matches OpenACC. (2) nvfortran 25.1 bug:
    in OpenMP device code, passing an assumed-shape dummy of the enclosing host routine to a
    device function gives an illegal address (reproduced in a small program; local arrays and
    their sections work): the index kernel takes a local copy of the wavelengths. (3) The
    OpenMP mirrors in `update_ssp_basis` and `prepare_line_gaussians` mapped arrays of contexts
    never moved to the device, which `remove_from_device` never unmapped: stale present-table
    entries made `unit_csp` fail intermittently; they are now `if(ctx%on_device)`, and
    `move_to_device` maps an existing basis. (4) `igm_tau` takes local copies of the Lyman
    parameter arrays (OpenMP device code rejects the module's).
  - Validation: suites 33/33 and batch test 146/146 in all ten builds (OpenMP with
    `OMP_TARGET_OFFLOAD=MANDATORY`), case matrix PASS; and the case matrix through the batch
    (`FIXTURE_ARGS=--batch`, 25 cases) against FP64 CPU: OpenMP FP32 4.2e-7 per pixel, 3.0e-7
    mag, 1.2e-5 indices; OpenMP FP64 1.0e-11 per pixel (OpenACC: 4.2e-7 / 1.5e-7 / 1.1e-5 and
    1.6e-11).
  - Speed (`benchmark_csp_batch 10000 200`, us per model, magnitudes only; OpenMP / OpenACC,
    same code; the host fallback before this branch: 833 / 3788 at N = 10000, FP32):

    | N | FP32 basic | FP32 full physics | FP64 basic | FP64 full physics |
    |---|---|---|---|---|
    | 1 | 1125 / 827 | 1671 / 1311 | 1151 / 904 | 1680 / 1308 |
    | 100 | 30.0 / 19.0 | 37.9 / 27.7 | 37.0 / 20.9 | 45.5 / 29.8 |
    | 1000 | 5.2 / 4.1 | 9.4 / 8.5 | 6.4 / 4.6 | 11.2 / 9.3 |
    | 10000 | 3.6 / 2.9 | 7.8 / 7.0 | 4.7 / 4.2 | 9.5 / 8.9 |

    At large N the OpenMP build is within 10-25% of OpenACC; at small N the difference is in
    the low-parallelism kernels (Nsight Systems, one model, full physics: kernels 0.54 vs 0.33 ms
    of ~1.6 ms per call; e.g. age sums 96 vs 41 us, band fluxes 56 vs 13 us), where nvfortran's
    default OpenMP team and thread sizes differ from the OpenACC mapping. Tuning note:
    `num_teams` / `thread_limit` per kernel is a per-GPU choice and was not set for the A100. The single CSP call
    (`compute_csp_scenario`) is not ported and runs on the host in the OpenMP build (7 ms FP32).

- Per-call overhead, second attempt (not committed): the OpenACC batch kernels on one async
  queue (one wait before the photometry instead of ~20 host synchronizations) crashed the batch
  test with a corrupted stack (host segfault); reverted. It is a real race: the same build
  passes 146/146 with `NVCOMPILER_ACC_SYNCHRONOUS=1`. `-Minfo=accel` shows no implicit copies
  in the kernels (only the batch data region's clauses), so the cause is still open. Expected
  gain ~0.15 ms per call.

- OpenMP portability fixes from the AMD Radeon 780M test (branch `gpu/a100-omp-portability`,
  stacked on `gpu/a100-omp-batch`):
  - Cherry-picked from `test/780m-omp-batch`: f27d2f8 (`-Daccel=omp_gpu` flags for LLVM flang)
    and ce8f02f (device code without `log_gamma`, without Fortran-runtime copy-in/out in
    `get_agb_dust_terms`, contiguous index dummies).
  - The OpenMP mapping conflict LLVM rejected ("explicit extension not allowed", 936 bytes, 16
    bytes apart): 936 bytes is exactly the batch's `nebular_setup_t` with flang descriptors (13
    rank-1 and 4 rank-2 allocatable components plus 24 bytes of scalars), which was mapped whole
    and by component, and referred to through a dummy inside kernels. Now the kernels take its
    arrays as dummies (`nebular_kernels`, `attenuate_lines`), and it is mapped through its
    components only, in both programming models. The smoothing kernel likewise takes the
    operator's arrays instead of reading them through the batch cache (`smooth_kernel`).
    Kernels now refer only to `ctx` (mapped whole once, components attached at
    `move_to_device`) and elements of the mapped `psets` array. Not reproducible here (no LLVM
    offload toolchain on the cluster); to be confirmed on the 780M.
  - `c_batch` on the OpenMP build here: PASSED in ~20 s. On the 780M its timeout may be the
    SSP bases of all 12 metallicities it builds (mostly host work in the OpenMP build) rather
    than a hang; worth timing with a larger test timeout.
  - Rules for device code from now on: `doc/GPU_PLAN.md`, "Rules learned so far".

  - Validation: suites 33/33 and batch test 146/146 in all ten builds, case matrix PASS; the
    nvfortran CPU outputs match the pre-cherry-pick ones to `--rtol 1e-14`.

- Final items (branch `gpu/a100-final`, stacked on `gpu/a100-omp-portability`): the legacy
  `compute_interpolated_ssp` calls fixed; `compute_csp_batch` moves the context to the device
  when needed; `c_batch` varies exactly the fitting parameters (logzsol, dust2, dust1 =
  dust_ratio * dust2, dust_index, fagn, agn_tau, gas_logz, gas_logu, duste_umin, duste_gamma,
  duste_qpah, zred) with a non-parametric SFH as an SSP-weight matrix: identical to setting the
  models one at a time.
  Validation: suites 33/33, batch test 146/146 in all ten builds, case matrix PASS.

- CI fixes after PR #10's first run (4 of 15 jobs failed): (1) gfortran `-fopenmp` (CPU
  threading, ThreadSanitizer job) compiled the OpenMP *target* directives and rejected maps of
  derived types with allocatable components: every target / declare-target directive and the
  inner `parallel do` of the kernels are now under `#ifdef OMP_TARGET`; (2) OpenACC multicore
  (both precisions) failed the IGM tests: `compute_igm_transmission` is back to its own loop
  (calling `igm_tau` from it gave wrong optical depths under multicore; `igm_tau` stays for the
  batch's per-model IGM, which passes there); (3) LLVM flang: `test_c_batch` links libm. Checked
  with targeted runs before the job ended (unit suites, batch test 146/146, `c_batch`) on
  OpenACC multicore FP32, OpenACC GPU FP32 / FP64, OpenMP target FP32, gfortran, gfortran
  `-fopenmp`; not the full ten-build validation (no time left on the job). CI on the PR is the
  check.

**In progress / queue**
- PR #10 (https://github.com/elijahmathews/fsps/pull/10) from `gpu/a100-final` (the tip of the
  stack) to `develop`, opened for CI; not merged. Session ended here.

**Open problems**
- AMD Radeon 780M (LLVM flang 22 + ROCm 7.1.1, `gpu/a100-omp-portability` tip): builds, 31/33
  pass with `OMP_TARGET_OFFLOAD=MANDATORY`, but `csp_batch` and `c_batch` abort (after 9 s /
  22 s), and direct reruns hang the GPU (amdgpu resets; it is the display GPU, so testing is
  paused). Still open after the mapping fix of that branch; to debug from a text console (next:
  `LIBOMPTARGET_INFO=-1` / `LIBOMPTARGET_DEBUG` to find the failing construct, then a reduced
  batch, e.g. one model without nebular emission).
- Legacy C API (`FSPS_ENABLE_LEGACY`): does not compile (see Questions, 1).
- Nsight Compute cannot run: `RmProfilingAdminOnly: 1` on this node and `ncu` fails with
  `ERR_NVGPUCTRPERM` (GPU performance counters for admins only). Kernels are tuned from Nsight
  Systems timings instead.
- OpenACC FP64 GPU builds: `test_csp_batch` spectra differ from `compute_csp_scenario` by up to
  8.6e-11 (tolerance 1e-10) and mass / L_bol / SFR by 7.7e-13, against <= 6e-16 and 0 in the
  CPU and OpenMP builds. The batch computes SFH weights on the device, the single-output
  scenario on the host (since the P100 session). Confirmed: given the scenario's weights as a
  matrix, the batch agrees to 6.7e-16; the 8.6e-11 is device vs host arithmetic in
  `sfh_weights_core` (libm / FMA contraction), within the tolerance but with little margin.

## P100 session status (final)

**Done**
- Environment checks: GPU visible (`nvidia-smi`: Tesla P100-PCIE-12GB, driver 580.178.04);
  `git push --dry-run origin HEAD` succeeds (HTTPS through the `gh` credential helper).
- Build fix: Fortran executables now link with the Fortran compiler (nvc 25.1 does not add
  the Fortran runtime). Branch `gpu/p100-baseline`.
- Builds: `build_cpu` (gfortran 13.4, FP64), `build_gpu` (nvfortran 25.1, cc60, FP64),
  `build_gpu32` (FP32 kernels).
- Device data-residency fixes (context lifecycle, AGN kernel, one unit test): all builds pass
  31/31 (`build_cpu`, `build_gpu`, `build_gpu32`).
- Case matrix (tools/compare_builds): FP64 GPU = nvfortran CPU to 1.6e-11 per pixel; FP32 GPU
  vs FP64 CPU 3.3e-7 per pixel, 2e-7 mag (same as the laptop FP32 numbers). PASS.

- Goal 1 baseline benchmarks and Nsight Systems profile (below): GPU SSP grid is 3-5x slower
  than one CPU core (per-star kernel = 95% of GPU time); CSP call 2.8 ms (FP64) vs 4.0 ms CPU.

- Goal 2 (branch `gpu/p100-ssp-templates`, stacked on `gpu/p100-baseline`): the SSP template
  sum runs in OpenACC builds; the main-library product is a device kernel (3.2 ms per grid).
  SSP grid 2.48 s -> 0.38 s (FP64 GPU), 1.33 s -> 0.39 s (FP32 GPU); nvfortran CPU 0.45 s.
  What remains is host work (weight aggregation, dust-shell AGB corners, small libraries,
  IMF weights). Fixed an FP32 subnormal problem in the AGB template sums (nvfortran only).
  All builds 31/31; case matrix PASS.

- Goal 3 (branch `gpu/p100-batched-csp`, stacked on `gpu/p100-ssp-templates`): batched CSP
  entry point `compute_csp_batch` + `tests/test_csp_batch` + `tests/benchmark_csp_batch`.
  GPU FP32 at N = 10000: 14.6 us per model (OpenACC), 8.1 us (cuBLAS, `-Dcublas=true`),
  vs 5.9 ms for one `compute_csp_scenario` call (1.8 ms FP64). Also fixed on this branch:
  stale device copies of changed settings (dust_type, IGM, ... ignored on the GPU) and
  per-thread device heap allocations (crash at N = 10000). All builds 32/32; case matrix PASS.

- Continuation items 2-3 (branches `gpu/p100-csp-allages`, `gpu/p100-device-mags`, stacked):
  all-ages CSP sum collapsed over outputs (8x); magnitudes of all outputs computed on the
  device (`fsps_batch_photometry`, shared with the batch); single-output SFH weights on the
  host. P100 `compute_csp_scenario`: single call 1.93 -> 0.66 ms (FP64), 5.8 -> 0.66 ms (FP32);
  all-ages call (107 outputs) 94 -> 6.0 ms (FP64), 500 -> 5.0 ms (FP32). All builds 32/32.

- Nebular emission in the CSP call (branch `gpu/p100-device-mags`): no longer re-sends the
  SSP grid per young age; all young ages in a few kernels. P100 CSP call with nebular
  emission 33 -> 2.8 ms (FP64), 22 -> 2.4 ms (FP32). All GPU builds 32/32; case matrix PASS.

- Continuation item 1 (branch `gpu/p100-ssp-small-libs`): the other libraries' templates and
  the dust-shell AGB corners are summed on the device with the main library; per-age buffer
  uploads removed. P100 SSP grid 0.38 -> 0.27 s (FP64), 0.33 -> 0.27 s (FP32); what remains is
  host work, half of it IMF integrals (see open questions). Fixed the template-vs-per-star
  oracle, which compared the template path with itself in nvfortran builds.

- Continuation item 4 (branch `gpu/p100-omp-target`): OpenMP target offload versions of the
  template product, the batched age sums and the band fluxes (`-Daccel=omp_gpu`). Compiles;
  passes 32/32 with `OMP_TARGET_OFFLOAD=DISABLED` (host fallback). Not runnable on the P100
  (see open problems).

- Item 3 remainder (branch `gpu/p100-pset-resident`): `pset` mapped once per CSP call
  (0.668 -> 0.639 ms, FP64); spectral indices of all outputs computed on the device
  (all-ages call with indices 26 -> 6.4 ms); fixed a 25% gfortran FP64 SSP slowdown introduced
  with the FP32 AGB rescaling (now FP32-only; gfortran FP64 output bit-identical to `develop`).
- Session ended before item 5 (final write-up) and before restricting cuBLAS to >= 16 spectra
  (single-spectrum magnitudes with cuBLAS lose accuracy, 5.2e-6, within tolerance, no speed gain).

**Verified on the laptop (2026-10-16, branch tip)**
- gfortran FP64 and FP32: unit + regression 32/32; FP64 outputs bit-identical to `develop` on
  all 36 matrix cases; CPU CSP call with nebular emission 17.9 -> 5.8 ms (Ryzen 7840U).
- nvfortran 26.1 OpenACC multicore FP64 32/32; FP32 failed `csp_batch` at 3-4.7e-6 against a
  2e-6 tolerance (multicore FP32 reductions are plain per-core sums); tolerance now 100 eps(KP).
- `-Daccel=gpu` (cc80,cc90, SDK 26.1) compiles in both precisions.

**In progress**
- Next session (A100): see "Next" in doc/GPU_PLAN.md and the open problems below.

**Open problems**
- *OpenMP offload cannot run on this node.* nvfortran's OpenMP GPU offload needs compute
  capability >= 7.0 ("OpenMP GPU Offload is available only on systems with NVIDIA GPUs with
  compute capability >= cc70"); the P100 is cc60. No offload-capable gfortran/LLVM is installed
  (gcc 12.2-13.4 have no OFFLOAD_TARGET_NAMES; Intel LLVM targets Intel GPUs). `-Daccel=omp_gpu`
  is built for cc70,cc80 and validated with host fallback only; the device run with
  `OMP_TARGET_OFFLOAD=MANDATORY` (which aborts here, as it should) is for an A100 node.
- *Open question (physics/parity, needs a decision):* half of the remaining SSP-grid time on the
  GPU is host IMF integrals (Romberg: IMF weights ~127 ms, remnant masses ~44 ms of ~270 ms).
  Branch `perf/imf-analytic` (exact erf integrals for Chabrier) is marked "needs a parity
  decision"; not merged or used here (conservative choice).
- Branches are stacked (each later one contains the earlier ones' commits), because the
  device-residency fixes on `gpu/p100-baseline` are needed for any GPU run; merge them in order
  (baseline, ssp-templates, ...), or rebase after the first is merged.
- gfortran 13.4 and nvfortran 25.1 CPU builds differ by up to 8.5e-10 per pixel (4e-10 mag)
  for the power-law IMFs (Salpeter, Kroupa, Dave); Chabrier agrees to 2e-13. Not a GPU issue
  (the GPU matches nvfortran CPU to 2e-13 on those cases), but an FP64 GPU build compared
  against a *gfortran* CPU build fails `--rtol 1e-10` there; compare FP64 GPU builds against
  an nvfortran CPU build (`build_nvcpu`), or use `--rtol 1e-9`.

## Entries

### 2026-10-15: node and toolchain

- Node `p-gc-3150` (ICDS), 2x Xeon E5-2680 v4 (5 cores allocated), 251 GB RAM.
- GPU: Tesla P100-PCIE-12GB, driver 580.178.04 (CUDA 13.0 driver API).
- Compilers: `module load nvhpc/25.1` (nvfortran/nvc 25.1-0, Nsight Systems 2024.7.1);
  CPU reference built with `module load gcc/13.4.0` (the system gfortran is 8.5).
  meson 1.10.1, numpy 1.26.4.
- Build fix: meson linked the test executables with `nvc` (libfsps has one C source) and
  nvc 25.1 does not link `libnvf`, so every Fortran program failed with
  `undefined reference to pgf90io_*`. Added `link_language : 'fortran'` to every Fortran
  executable (the library already had it). CPU builds are unaffected.
- CPU (`build_cpu`, gfortran 13.4): `meson test --suite unit --suite regression`, 31/31 OK, 83 s.

### 2026-10-15: device data-residency bugs (GPU FP64 suite 25/31 -> 31/31)

First run of `meson test -C build_gpu --suite unit --suite regression` on the P100:
6 failures, all host/device mapping bugs that OpenACC multicore (CI) cannot see:

1. `fsps_destroy` never removed a context from the device. Its freed host arrays kept their
   present-table entries; the next context's arrays, allocated at overlapping host addresses,
   then failed with "variable in data clause is partially present"
   (`ctx%state%gaussnebarr`, `fsps_context_move_to_device`). Broke param_regression_* and
   physics_oracles. Fix: `ctx%on_device` flag; destroy, setup (which repoints and reallocates
   state arrays) and `sps_takedown` call `fsps_context_remove_from_device` first; a second
   `move_to_device` refreshes instead of stacking reference counts.
2. `fsps_context_remove_from_device` on a context never mapped, as the first OpenACC operation
   in the process, segfaults inside the 25.1 runtime (`__pgi_uacc_select_devid`). Hit by the
   photometry unit tests' teardown-before-setup. Fix: it returns when `.not. ctx%on_device`.
3. `apply_agn_dust_emission` had `present(settings)` for the caller's host `pset`. Fix:
   `copyin(settings)`; the kernel also now waits for queue 1 (`wait(1)`), since the preceding
   CSP post-processing kernel writing the spectrum runs `async(1)` and the AGN kernel ran on
   the default queue (a race on a GPU, harmless on multicore).
4. Unit test `test_remnants_ignore_appended_stragglers` passed a host-only isochrone buffer to
   `compute_integrated_properties` (device reductions). Fix: map it, as the neighbouring
   remnant test already does.

Results after the fixes (meson test, unit + regression):

| Build | Result | Wall time (4 jobs) |
|---|---|---|
| build_cpu (gfortran 13.4, FP64) | 31/31 | 83 s (5 jobs) |
| build_gpu (nvfortran 25.1, cc60, FP64) | 31/31 | ~170 s |
| build_gpu32 (FP32 kernels) | 31/31 | |

### 2026-10-15: case matrix, GPU vs CPU (tools/compare_builds)

```
tools/compare_builds/run_cases.sh build_cpu   out/cpu      # gfortran 13.4, FP64
tools/compare_builds/run_cases.sh build_nvcpu out/nvcpu    # nvfortran 25.1, no accel, FP64
tools/compare_builds/run_cases.sh build_gpu   out/gpu
tools/compare_builds/run_cases.sh build_gpu32 out/gpu32
python3 tools/compare_builds/compare_builds.py out/gpu out/nvcpu --rtol 1e-10   # PASS
python3 tools/compare_builds/compare_builds.py out/gpu32 out/cpu                # PASS
```

Worst case over the 36 cases:

| Comparison | dlogM | dlogL | spec (912 A-2.5 um) | spec (all) | mag | indx | lines |
|---|---|---|---|---|---|---|---|
| FP64 GPU vs nvfortran CPU | 5.4e-13 | 7e-12 | 1.6e-11 | 1.2e-10 | 1.3e-11 | 2.3e-11 | 1.2e-12 |
| FP32 GPU vs gfortran CPU (FP64) | 1.2e-12 | 1.8e-11 | 3.3e-7 | 3.4e-7 | 2.0e-7 | 4.1e-6 | 2.3e-7 |

Note the FP64 GPU path computes SSPs per star (the OpenACC kernel) while the CPU builds use the
template sum; they agree to 2e-13 per pixel on the SSP cases.

FP64 GPU vs gfortran CPU fails `--rtol 1e-10` on four SSP cases (ssp_salpeter 5.1e-10,
ssp_kroupa 4.7e-10, ssp_dave 4.7e-10, ssp_kroupa_zlow 8.5e-10). Isolated with two more nvfortran
builds on the same commit (`build_nvcpu`: no accel; `build_mc`: `-Daccel=multicore`): both
show the same 5e-10 against gfortran, and the GPU matches both to 2e-13. It is a
gfortran-vs-nvfortran difference in the power-law IMF weights (Chabrier agrees to 2e-13),
not a device effect.

### 2026-10-15: baseline benchmarks (goal 1)

`build_<x>/benchmark_phases 5 200` (MIST + MILES + DL07, Chabrier; seconds; one CPU core of the
Xeon E5-2680 v4 for the CPU builds; GPU builds use the P100 with the host orchestration on the
same core):

| Metric | gfortran CPU FP64 | nvfortran CPU FP64 | GPU FP64 | GPU FP32 kernels |
|---|---|---|---|---|
| setup | 9.47 | 6.41 | 7.17 | 7.55 |
| first spectrum | 9.96 | 6.86 | 9.71 | 8.94 |
| SSP grid (one call) | 0.477 | 0.424 | 2.475 | 1.332 |
| CSP call | 0.00404 | 0.00481 | 0.00283 | 0.00594 |
| SSP grid, nebular | 0.477 | 0.415 | 2.477 | 1.333 |
| CSP call, nebular | 0.0327 | 0.0461 | 0.0380 | 0.0293 |

(Laptop Ryzen 7840U for reference: SSP 0.147 s, CSP 1.7 ms, CSP+nebular 16 ms; this Broadwell
core is ~3x slower.) Setup is file I/O on GPFS and varies run to run.

Nsight Systems, `nsys profile -t cuda,openacc build_gpu32/benchmark_phases 2 50`
(2+1 SSP grids, 101 CSP calls, 50 with nebular emission), GPU time:

| Kernel / operation | Total | Per call | Share |
|---|---|---|---|
| `accumulate_spectrum` line 711: per-star spectra (one thread per star, seq over pixels) | 5.55 s | 10.4 ms x 535 ages | 94.8% of kernel time |
| `accumulate_spectrum` line 753: weighted sum over stars | 54 ms | 0.10 ms | 0.9% |
| `integrate_csp_step` line 367: CSP age sum, nt x nz per output | 85 ms | 0.84 ms per CSP call | 1.5% |
| nebular kernels (`add_lines_to_spectrum` etc.) | ~100 ms | ~45 us x 2350 | 1.7% |
| dust attenuation + emission | ~11 ms | ~0.1 ms per CSP | 0.2% |
| H2D copies | 0.66 s, 11098 copies | median 1.5 us, max 43 ms | |
| D2H copies | 10 ms, 5303 copies | | |

`cuStreamSynchronize` accounts for 6.2 s of host API time (the host waits on every kernel).
Conclusions: (1) SSP time is entirely the per-star kernel: each thread loops over the full
spectrum of one star, with ~1000-2000 threads per age, so the P100 is latency bound; this is
goal 2 (template GEMM). (2) The CSP call is ~1 ms of GPU work per call plus host orchestration
and many small transfers; the age sum (line 367) is the largest kernel.

### 2026-10-16: goal 2, SSP template sum on the GPU (branch `gpu/p100-ssp-templates`)

**Change.** `generate_ssp_grid` now uses the template path in OpenACC builds as well: per age,
`accumulate_by_templates` aggregates the main-library weights on the host in FP64
(`w_main_all`, templates x ages) and sums the other libraries on the host; after the age loop
`add_main_library_templates` runs on the device: one thread per (pixel, age) sums the used
templates of the resident `speclib(:, zmet, :, :)` in the host order, compensated (Kahan) at
FP32 kernel precision, plain at FP64. Results come back in a private array and are added on
the host, so callers that map `spec_grid` themselves (test_runner does) stay consistent; the
routine also refreshes caller-mapped outputs at exit (`update device ... if_present`). The
per-star OpenACC kernel stays as the reference (`ssp_per_star = .true.`, used by
`physics_oracles`).

**Bug found on the way (nvfortran FP32, CPU and GPU).** The first FP32 GPU run failed
`param_regression_ssp` (ssp_plain: 104 far-IR pixels off by up to 1%). The same error, to the
last digit, appeared in an nvfortran FP32 *CPU* build, and not in gfortran FP32: the template
path had never been compiled by nvfortran in a non-OpenACC build (CI's nvfortran job is
OpenACC multicore, which used the per-star kernel). Dumping the per-age remainder showed the
O/C-rich AGB term: the template path folds the IMF weight in before lambda^2, so at 1-2.7 mm
(lambda^2 ~ 7e14 A^2) the sum is ~1e-37, at the bottom of the FP32 range. gfortran keeps
subnormals; nvfortran's runtime and GPUs flush them. (Ruled out first: optimization level,
FMA, vectorization, and the `-Mflushz/-Mnoflushz` main-program flag, which did not change the
result.) Fix: those sums carry an exact factor 2^64, removed after lambda^2 (bit-identical in
FP64). This is rule 5 of `doc/MIXED_PRECISION.md` (keep values in range) for an intermediate.

**Validation.** Unit + regression 31/31 in all four builds (gfortran FP64 and FP32, GPU FP64 and
FP32). Case matrix, worst over 36 cases:

| Comparison | spec (912 A-2.5 um) | spec (all) | mag | indx | lines |
|---|---|---|---|---|---|
| FP64 GPU vs nvfortran CPU (rtol 1e-10) | 1.6e-11 | 1.2e-10 | 1.3e-11 | 2.3e-11 | 1.2e-12 |
| FP32 GPU vs gfortran CPU FP64 (defaults) | 3.1e-7 | 4.1e-7 | 2.0e-7 | 4.1e-6 | 2.3e-7 |

**Speed** (`benchmark_phases 5 200`, seconds per SSP grid):

| Build | before | after |
|---|---|---|
| GPU FP64 | 2.475 | 0.383 |
| GPU FP32 | 1.332 | 0.387 |
| nvfortran CPU FP64 (reference) | | 0.446 |

Nsight: `add_main_library_templates` kernel 3.16 ms per grid (FP32, Kahan, one thread per
pixel x age, 1292 templates x 107 ages x 5994 pixels). The SSP call is now host-bound. gprof
of the FP32 GPU build (`-pg`, 7 grids): per grid ~0.2 s in `accumulate_by_templates`
(aggregation, the per-age small-library sums `add_column`, and above all the dust-shell AGB
corners, where each corner slot does several full-spectrum passes per age) and ~0.09 s in the
IMF weights (`get_imf_value` / Romberg `refine_trapezoid`). Next for SSP: move the per-age
small-library and dust-corner sums to the device as products over all ages (continuation item
1); the IMF weights need the analytic integrals (branch `perf/imf-analytic`) to leave the host.
A batch of grids with different IMF parameters would currently be host-bound at ~0.38 s per
grid, so it was not timed separately.

### 2026-10-16: goal 3, batched CSP (branch `gpu/p100-batched-csp`)

**New entry point** `compute_csp_batch(ctx, psets, mags, status [, spec, mass, lbol, mdust, sfr])`
in `src/physics/fsps_csp_batch.f90` (`compute_csp_scenario` unchanged). N single-age models
(`tage > 0`) against the resident SSP basis (one metallicity), sharing zred / igm_factor;
SFH, dust attenuation (dust1/2/3, indices, birth-cloud age, frac_obrun, frac_nodust, curve
parameters) and dust emission (U_min, q_PAH, gamma) vary per model. Status 2 for nebular
emission, AGN, smoothing, redshifted colors, light ages, tage <= 0 (decided to keep v1 to the
stages that are matrix products or element-wise; these are listed as open items). Stages:

1. SFH weights per model on the device in FP64 (`sfh_weights_core`, one thread per model;
   factored out of `compute_sfh_weights` with caller-provided scratch), split into young/old
   at each model's `dust_tesc`, cast once to KP; mass and L_bol from them (FP64).
2. Age sums: one product basis(pixels x ages) . [W_young W_old](ages x 2N).
3. Attenuation + absorbed light density: element-wise over (pixel, model), as in `fsps_dust`.
4. Dust emission: per-model reductions (gang per model), Draine & Li template interpolated per
   model on the device from the resident grid, added with the energy-balance normalization.
5. Magnitudes: observed-frame interpolation, then per-(band, model) filter integrals over each
   band's support (Kahan at FP32), or with `-Dcublas=true` a split-K product (256-pixel blocks,
   blocks summed in FP64).

**Validation.** `tests/test_csp_batch.f90` (regression suite, all builds): 12 models x 4
configurations (Calzetti z = 0; power law z = 0.5 with IGM; Cardelli z = 0; Kriek & Conroy
z = 1.2; SFH types 0, 1, 4, 5, bursts, constant, sf_start/trunc/slope, dust emission on)
against `compute_csp_scenario` model by model:

| Build | spectrum (rel.) | magnitudes | mass, L_bol, SFR | dust mass |
|---|---|---|---|---|
| gfortran CPU FP64 | 0 (bit-identical) | 1.4e-14 | 0 | 0 |
| GPU FP32, OpenACC | 2.6e-7 | 2.7e-7 | 1.1e-16 | 1.1e-7 |
| GPU FP32, cuBLAS | 6.8e-7 | 4.4e-7 | 1.1e-16 | 1.1e-7 |

All builds pass unit + regression (32 tests, including the new one); case matrix unchanged
(FP64 GPU = nvfortran CPU to 1.6e-11; FP32 GPU vs FP64 CPU 3.1e-7 / 2e-7 mag).

**Bugs found while building it** (all fixed on this branch):
- *Stale settings on the device.* Kernels read scalar settings (`dust_type`,
  `add_igm_absorption`, ...) from the device copy of `ctx`, written only by `move_to_device`.
  A setting changed afterwards had no effect on the GPU: the test's IGM case came out without
  IGM absorption, and all dust types used the first one. This affects `compute_csp_scenario`
  too (any caller that changes a setting after moving the context). Fix: the setters mark the
  context dirty and the compute entry points refresh the device copy
  (`fsps_context_sync_device_settings`).
- *Device heap exhaustion.* The batch crashed at N = 10000 (illegal address; fine at 1000;
  `compute-sanitizer` pointed at local memory in `compute_ssp_weights`; a larger stack did not
  help, a 256 MB device heap did). Device code allocated temporaries on the 8 MB device heap
  in every thread: the context's state arrays are pointers, so passing them to `contiguous`
  dummies (`find_interval`, ...) needed a runtime contiguity copy, and the tabular SFH code
  passed the strided row `sfh_tab(1, :)`. Fix: state pointer arrays are declared `contiguous`
  (they always point to whole allocated arrays) and the table search runs in place. A scan of
  the PTX of every module (`-gpu=keep`, grep `malloc`) now finds no device allocations.
- An out-of-bounds read in the new observed-frame kernel at z = 0 (found with a
  `-fcheck=all` gfortran build).

**Speed** (`build_x/benchmark_csp_batch 10000 200`; microseconds per model; tau, age, dust1/2
and U_min vary; `mags` = magnitudes only, `spec` = also copying the spectra back):

| N | GPU FP32 OpenACC | GPU FP32 cuBLAS | GPU FP64 OpenACC | GPU FP64 cuBLAS |
|---|---|---|---|---|
| one `compute_csp_scenario` call | 5837 | 5754 | 1832 | 1916 |
| 1 | 5859 | 7932 | 6827 | 10267 |
| 100 | 89 | 119 | 110 | 119 |
| 1000 | 51 | 22 | 64 | 22 |
| 10000 (mags) | 41.9 | 10.9 | 57.6 | 12.5 |
| 10000 (spec) | 62.2 | 31.4 | 86.9 | 40.8 |

(The FP32 single call is slower than FP64 because its band integrals run on the host,
compensated and serial per band; `doc/MIXED_PRECISION.md` notes the same on the CPU.)
Against the forecast of ~4 us per model (A100, FP32, large N), the P100 reaches 10.9 us with
cuBLAS. Where the rest goes, N = 10000, OpenACC FP32 before the last kernel changes (timers in
the driver): host stage (indices, SFR) and transfers small; device stages: age sums +
attenuation + dust emission ~128 ms; band fluxes ~237 ms; host magnitude assembly 22 ms;
spectra copy-back ~200 ms (240 MB pageable D2H).

**Kernel work after the first version** (OpenACC FP32 unless noted; us per model, mags only):

| Step | N = 1 | N = 100 | N = 1000 | N = 10000 |
|---|---|---|---|---|
| first version (host SFH weights) | 5753 | 99 | 63 | 57.5 |
| SFH weights on the device, no device heap allocations | 5859 | 89 | 51 | 41.9 |
| band fluxes per (band, model block), transposed observed spectrum | 6190 | 79 | 30 | 21.4 |
| age sums: one pass (young or old per age), 4 models per thread | 6665 | 79 | 25 | 17.4 |
| keep transmission + dust template from the attenuation pass | 6070 | 73 | 23 | 15.4 |
| cache grid-only data per redshift (on the device) | 1688 | 31 | 16 | 14.6 |
| same, cuBLAS FP32 | 1076 | 24 | 9.4 | **8.1** |

Profile of the OpenACC FP32 build at N = 10000 before the last two steps (per call): age sums
81 ms (before the one-pass change), band fluxes 36 ms, attenuation 26 ms, dust reductions
18 ms, dust emission 13 ms, SFH weights 3.3 ms, observed frame 4 ms. With spectra copied back
(`spec`), add ~20 us per model at N = 10000: 240 MB through pageable host memory.

### 2026-10-16: continuation items 2 and 3, compute_csp_scenario on the device

Timing harness: `compute_csp_scenario` with tage = 0 (107 outputs, tau model, dust), and
`benchmark_phases` for single calls; ms per call on the P100 unless noted.

| Change | single call FP64 / FP32 | all-ages FP64 / FP32 |
|---|---|---|
| before (branch `gpu/p100-batched-csp`) | 1.93 / 5.8 | 86 / 500 |
| item 2: multi-output age sums, one thread per (output, pixel) | 1.93 / 5.8 | 89 / 499 (kernel 4.6 -> 0.57 ms) |
| item 3: magnitudes of all outputs on the device | 1.93 / 2.0 | 6.1 / 5.4 |
| single output: SFH weights on the host; band reductions for < 16 spectra | **0.66 / 0.66** | **6.0 / 5.0** |
| nvfortran CPU, one Broadwell core (reference) | 4.8 | 278 |

- Item 2: the multi-output spectra and emission-line sums in `integrate_csp_step` ran one gang
  per output with the ages sequential and a loop-carried accumulator (`-Minfo`: serialized).
  They now use the single-output kernel collapsed over (output, pixel); same summation order,
  same results. (The batched GEMM form of the plan is what `compute_csp_batch` does; here the
  weights differ per output only in the age range, and the collapsed kernel is 0.57 ms.)
- Item 3: the per-output host loop computed magnitudes on the host after a device round trip
  of each spectrum (FP32: compensated serial band sums, 4.6 ms per output). Now, without
  smoothing, redshifted colors or light-weighted ages, `compute_magnitudes_columns` computes all
  outputs' magnitudes from `out_csp_spec` on the device (OpenACC builds only; CPU results
  unchanged), and indices take each spectrum directly. The batch grid cache had to be released
  after the call for contexts mapped by hand (unit tests), which never reach
  `remove_from_device`.
- Single output: `compute_sfh_weights` ran on one device thread (0.65 ms); the host takes
  ~20 us. Band sums for one spectrum ran one thread per band (0.65 ms); a gang reduction per
  (band, spectrum) is used below 16 spectra (pairwise tree, as the dust integrals).
- Profile of the remaining single call (FP32, 0.66 ms): device kernels ~0.3 ms (dust
  attenuation and emission ~0.1 ms, age sum 50 us, post-processing 35 us, ...), the rest
  launches, small transfers and host packing. With nebular emission: 22-33 ms per call, which
  is the next item (47 kernel launches per call and host work).

### 2026-10-16: nebular emission in the CSP call

`benchmark_phases` "CSP call, nebular" (gas_logu varies per call): 33 ms (FP64) / 22 ms (FP32)
on the GPU, against 16 ms on the laptop CPU. Two causes:

1. `process_ionizing_radiation` ran `!$acc update device(spec_in)` for every young age, and
   `spec_in` is the whole SSP grid: ~50 x 5 MB host-to-device per call (`apply_nebular_emission`
   did it once more on entry). Callers either map it (the resident basis) or rely on `pcopyin`,
   so the updates were redundant. Removed: 33 -> 7.5 ms.
2. Per young age, ~6 kernel launches and a synchronous copy-back of Q. `nebular_all_ages` (OpenACC
   only; the CPU loop is unchanged) does EUV escape, Q per age (gang reduction), continuum, line
   luminosities and line profiles each in one kernel over (pixel or line, age), with Q kept on
   the device: 7.5 -> 2.8 ms (FP64), 2.4 ms (FP32).

Case matrix after this: FP64 GPU vs nvfortran CPU, csp_nebular 7.9e-14 per pixel, lines 2e-16;
FP32 GPU vs FP64 CPU, csp_nebular 3.1e-7 per pixel, 3.5e-7 mag, lines 2.3e-7 (all cases PASS;
worst magnitude 4.6e-7, csp_ssp_offgrid, from the single-spectrum band reduction).

Current `benchmark_phases 5 200` on the P100 (s): SSP grid 0.38 / 0.33, CSP call 0.00066 /
0.00066, SSP grid with nebular 0.40 / 0.33, CSP with nebular 0.0028 / 0.0024 (FP64 / FP32).

### 2026-10-16: continuation item 1, the rest of the SSP template sum on the device

Timers in `generate_ssp_grid` (FP32 GPU, per grid, before): load 9 ms, isochrone physics (IMF
weights) 127 ms, buffer upload + integrated properties 56 ms, template aggregation + other
libraries + dust corners 133 ms (of which: aggregation 56, other libraries ~10, dust corners
75), main-library kernel 3-5 ms; total 330 ms.

1. Buffer uploads: the isochrone buffer is only needed on the device by the per-star kernel;
   with the template path the 9 per-age uploads and the device reductions of
   `compute_integrated_properties` are skipped (`if(ssp_per_star)` clauses). The remaining
   44 ms there is `add_remnant_mass` (an IMF integral, host).
2. Other libraries (WMBasic, post-AGB, WN/WC, carbon stars, O/C-rich AGB): in OpenACC builds
   their template weights are deferred as extra rows of the all-ages weight matrix and summed
   in the main-library kernel (O/C-rich apart, times lambda^2, with the exact 2^64 factor).
3. Dust-shell AGB corners: each age's corners are recorded (transfer-function indices and the
   corner's template weights, CSR-style) and the device kernel rebuilds each corner's template
   sum (smoothed beyond 3 um; O/C-rich below it times lambda^2) times its transfer function,
   compensated at FP32 as on the host. 75 ms -> ~0 on the host.

CPU builds keep the per-age host sums (unchanged results). SSP grid, P100: FP64 0.38 -> 0.27 s,
FP32 0.33 -> 0.27 s (nvfortran CPU: 0.45 s; gfortran CPU 0.48 s, one Broadwell core).

**Test gap found:** `physics_oracles` (template sum = per-star sum) set `fsps_ssp::ssp_per_star`
by assignment from the test program. With nvfortran, a program that assigns a module variable
of a shared library changes its own copy (the library keeps its own): both runs used the
template path, the check printed 0.00 and passed vacuously in every nvfortran build
(including CI's multicore job, where it compared per-star with per-star). New setter
`fsps_ssp_set_per_star`; the oracle now compares the two paths on the GPU: FP64 8.9e-12 per
pixel (tolerance 1e-10), FP32 within KP_RTOL above 1e-6 of the peak (the printed maximum,
1.8e5, is from pixels near FP32 underflow, ~1e-30 of the grid maximum, which the GPU flushes
to zero).

**nvfortran quirk:** assigning `dusty%slot_ptr(1:n)` (a component of an optional
`intent(in)` derived-type dummy) to an unallocated local allocatable left it unset (garbage
indices, illegal address in the kernel); explicit `allocate` fixed it.

Current `benchmark_phases 5 200` (s), P100, FP64 / FP32: SSP 0.270 / 0.270, CSP 0.00067 /
0.00067, SSP + nebular 0.269 / 0.278, CSP + nebular 0.0027 / 0.0024. Case matrix: FP64 GPU vs
nvfortran CPU 1.6e-11 per pixel, 1.3e-11 mag; FP32 GPU vs FP64 CPU 3.8e-7, 4.6e-7 mag. PASS.

### 2026-10-16: continuation item 4, OpenMP target offload

New build mode `-Daccel=omp_gpu` (nvfortran `-mp=gpu`, `-DOMP_TARGET`, OpenACC off): OpenMP
`target teams distribute parallel do` versions of the main-library template product
(`main_templates_omp` in `fsps_ssp.f90`), the batched age sums and the band fluxes (same
arithmetic as the OpenACC kernels; arrays mapped per call with `map(to:/from:)`, no device
residency yet). Everything else runs on the host in this mode.

```
FC=nvfortran CC=nvc meson setup build_omp32 --buildtype=release -Daccel=omp_gpu \
    -Dgpu_arch=cc70,cc80 -Dkernel_precision=single
OMP_TARGET_OFFLOAD=MANDATORY build_omp32/test_csp_batch   # aborts on the P100 (cc60)
OMP_TARGET_OFFLOAD=DISABLED meson test -C build_omp32 --suite unit --suite regression  # 32/32
```

With `-Dgpu_arch=cc60` nvfortran 25.1 refuses the target regions: OpenMP GPU offload needs
cc70+. So on this node the OpenMP path is validated numerically through host fallback only
(csp_batch: spectra 5.9e-7, magnitudes 1.5e-7 against `compute_csp_scenario`); the device run
is pending an A100 (V100 or newer).
