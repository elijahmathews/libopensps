# GPU plan: first GPU session (A100 or P100)

Runbook for the first session on an NVIDIA GPU node (A100, or P100: see below). The goal is to measure, then build the two
kernels that matter, with every step validated against the CPU build. Work on a branch
(`gpu/a100-<topic>`), commit and push as you go (`git push origin <branch>`), and do not merge.

## Where things stand

- **CPU (one Ryzen 7 7840U core, MIST + MILES, FP64):** SSP grid 0.147 s, CSP call 1.7 ms, CSP
  with nebular emission 16 ms (legacy FSPS: 10.6 s, 3.5 ms, 31 ms). `tests/benchmark_phases.f90`.
- **SSP generation** is a sum over library templates (`accumulate_by_templates` in
  `src/physics/fsps_ssp.f90`): per age, template weights are aggregated on the CPU in FP64 (cheap
  scalar work, ~40% of SSP time today), then each used template is added once (~40%). This is
  a matrix product: spectra(pixels x ages) = templates(pixels x templates) . W(templates x ages).
  Dust-shell AGB stars add one such product per dust-grid corner, multiplied by its transfer
  function. The OpenACC build still uses the older per-star kernel (`ssp_per_star`); the
  physics oracle `check_template_vs_per_star` asserts the two agree.
- **FP32 kernels** (`-Dkernel_precision=single`) reproduce FP64 to ~3e-7 per pixel and 2e-7 mag;
  rules and evidence in `doc/MIXED_PRECISION.md`. Use them on the GPU from the start.
- **Existing GPU code** is OpenACC (nvfortran, `-Daccel=gpu`, targets from `-Dgpu_arch`, default `cc80,cc90`): device
  residency in `fsps_context.f90`, kernels in SSP (per star), dust, gas, CSP and indices, with
  host orchestration and `update host/device` between stages. It has only been run on CPU cores
  (OpenACC multicore) in CI; it has not been validated on a GPU recently. The long-term
  direction is OpenMP target offload (portable to AMD and Intel GPUs; `doc/CI.md`).

## Known before starting (laptop compile of `-Daccel=gpu`, HPC SDK 26.1, both precisions)

- Both precisions compile for the GPU without errors. HPC SDK 26.1 rejects `cc70` (Volta),
  which the build used to hard-code; `-Dgpu_arch=cc80` is enough on an A100 (use `cc70` only
  with an SDK older than 26 on V100s).
- `-Minfo=accel` reports, to look at on the device:
  - the CSP age sum serializes when several output ages are computed (`tage = 0`): "loop carried
    reuse of `spec_young` / `csp_emlin_young`" in `integrate_csp_step`; the single-age path
    used in fitting is parallel; the batched GEMM form replaces both;
  - the Romberg integrator calls a function through a procedure argument, which device code
    cannot do; it runs on the host (IMF weights) today, but keeps IMF weights off the GPU
    (the exact Chabrier integrals on branch `perf/imf-analytic` would remove it);
  - `pset` / `settings` are copied in implicitly on each kernel launch (small per-call cost).
- nvfortran needs `CC=nvc` (the C sources get the OpenACC flags). If it fails with "Error in
  path /usr/lib/gcc/...", its `localrc` is stale after a GCC upgrade: rerun
  `makelocalrc -x <bindir>` (or point `NVLOCALRC` at a regenerated copy).

## On a P100 node (ICDS cluster)

The first session may run on a P100 (Pascal, `cc60`) instead of an A100. HPC SDK 26.x ships
only CUDA 13 (`cc75` and newer), so load `nvhpc/25.1` (CUDA 12, still supports Pascal):

```
module load nvhpc/25.1
FC=nvfortran CC=nvc meson setup build_gpu --buildtype=release -Daccel=gpu -Dgpu_arch=cc60
```

(If `meson`/`ninja` are missing: `python3 -m pip install --user meson ninja`.) CI tests
nvfortran 26.1 only, so an older-compiler front-end issue may appear at the first build; fix
it in place. A P100 answers the correctness and where-the-time-goes questions as well as an
A100; it cannot measure A100 absolute speed, TF32 or FP64 tensor-core GEMMs, and Nsight
Compute does not support Pascal (Nsight Systems and `compute-sanitizer` do). Its FP64 runs at
half the FP32 rate, so FP32 gains there (~2x) understate those on consumer GPUs.

## Session goals, in order

### 1. Baseline the existing OpenACC build on the A100

```
module load nvhpc            # or whatever provides nvfortran / nvc on the cluster
nvfortran --version; nvidia-smi
FC=nvfortran CC=nvc meson setup build_gpu --buildtype=release -Daccel=gpu -Dgpu_arch=cc80
FC=nvfortran CC=nvc meson setup build_gpu32 --buildtype=release -Daccel=gpu -Dgpu_arch=cc80 -Dkernel_precision=single
meson compile -C build_gpu && meson compile -C build_gpu32
export SPS_HOME=$PWD FSPS_DATA_HOME=$PWD
meson test -C build_gpu   --suite unit --suite regression
meson test -C build_gpu32 --suite unit --suite regression
build_gpu/benchmark_phases 5 200; build_gpu32/benchmark_phases 5 200
nsys profile -o gpu_base build_gpu32/benchmark_phases 2 50    # if Nsight Systems is available
```

Also build the plain CPU reference on the node (`meson setup build_cpu --buildtype=release`)
for comparisons on the same machine. Record: test results, the six benchmark lines for each
build, and where GPU time goes (kernel vs transfer vs host). If tests fail on the device but
pass under OpenACC multicore, that is a device data-residency bug (missing `update` /
`present`); fix it before anything else.

### 2. SSP template sum on the GPU

Make `accumulate_by_templates` (or a batched variant over all ages) run on the device:

- Keep the weight aggregation on the CPU (FP64) at first: it already produces the main-library
  W(templates x ages) for the whole grid (`w_main_all` in `generate_ssp_grid`), and
  `add_main_library_templates` is the single CPU routine that does spectra += templates . W.
  Replace that routine's body with the device GEMM. The remaining per-age work (small
  libraries, dust-shell corners, ~20% of SSP time) can follow the same pattern: one product
  per dust corner over all ages, then times that corner's transfer function.
- Device: templates resident (main library slice `speclib(:, zmet, :, :)` is FP32 already,
  ~31 MB for MILES), one GEMM per SSP grid. Prototype with cuBLAS SGEMM (through `iso_c_binding`
  or nvfortran's `cublas` module) to find the ceiling, then an OpenMP target (`!$omp target teams
  distribute parallel do`) or OpenACC kernel for portability.
- Accuracy: plain FP32 accumulation over ~1000 templates loses ~1e-5. Use split-K / blocked
  accumulation (or cuBLAS, which accumulates in FP32 tiles) and check against the CPU.
- Gate: `physics_oracles` and `param_regression_ssp` pass; FP32 GPU vs FP64 CPU within ~1e-6
  per pixel above 1e-6 of each age's peak (`tools/legacy_parity` OPSF outputs and the
  comparison approach in `doc/MIXED_PRECISION.md`).
- Then time a full SSP grid and, if possible, a batch of grids with different IMF parameters.

### 3. Batched CSP

The fitting workload: N parameter sets against one resident SSP basis. Add a batched entry point
(new routine; keep `compute_csp_scenario` unchanged) and a benchmark that times per-model cost
for N = 1, 100, 10 000:

- age sum: basis(pixels x ages) . weights(ages x N), a GEMM (SFH weights computed on the CPU
  in FP64 per model, or on the device later);
- attenuation and dust emission: element-wise plus per-model reductions (absorbed luminosity
  integrated directly, as in `fsps_dust.f90`);
- magnitudes: bands(bands x pixels) . spectra(pixels x N), another GEMM.

Forecast to compare against: ~4 us per model (FP32) on the A100 at large N.

## Validation tools

- **GPU against CPU, all 36 cases, one command each:**
  ```
  tools/compare_builds/run_cases.sh build_cpu out_cpu       # reference (FP64 CPU)
  tools/compare_builds/run_cases.sh build_gpu32 out_gpu32   # build under test
  python3 tools/compare_builds/compare_builds.py out_gpu32 out_cpu   # exits non-zero on failure
  ```
  Defaults fit FP32 kernels against FP64 (laptop FP32-vs-FP64 worst case: 3.3e-7 per pixel,
  2e-7 mag); an FP64 GPU build against the CPU should pass with `--rtol 1e-10`.

- `meson test --suite unit --suite regression` (31 tests), in both precisions.
- `tools/legacy_parity/run_matrix.sh LEGACY_SRC LEGACY_SPS_HOME OUT` with `BUILD=build_gpu32`
  (needs a legacy FSPS source and data tree on the node; optional).
- `build_*/generate_opensps_fixtures --isoc mist --spec miles --dust DL07 --out x.opsf key=value ...`
  writes OPSF files; compare GPU against CPU outputs case by case (`tests/data/param_cases.psv`
  has the 36 cases; `tools/legacy_parity/compare_legacy.py` has `read_opsf`).

## Rules learned so far

- Never `-fast` / `-ffast-math`: they delete the compensated sums.
- FP32 parallel reductions are not bitwise reproducible run to run (~1e-7); tests comparing two
  runs need KP-aware tolerances.
- Keep logarithms that are exponentiated later in FP64; integrate absorbed light directly;
  precompute grid-only weights in FP64 (`doc/MIXED_PRECISION.md`).
- `!$acc routine seq` is required on every routine called inside device code (a missing one
  silently broke the AGB dust screen in CI once).
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for OpenMP target runs so a silent host fallback fails.
- Device code must build with toolchains other than nvfortran (LLVM flang + ROCm on AMD, Intel):
  - no libm calls without a device fallback (AMD's GPU libm has no `lgamma`; `log_gamma` of an
    integer became a sum of logs);
  - no array arguments that need a Fortran-runtime copy-in/out (no GPU Fortran runtime on AMD):
    pass whole arrays or columns to `contiguous` dummies, copy strided rows to local fixed-size
    arrays first;
  - consistent maps: a derived-type object is mapped either whole or through its components,
    never both, and kernels do not refer to a derived type mapped only through components
    (pass its arrays as dummies; `nebular_kernels`, `smooth_kernel`). LLVM's offload runtime
    rejects overlapping maps that nvfortran tolerates;
  - (nvfortran 25.1, OpenMP) do not pass an assumed-shape dummy of the host routine to a device
    function inside a target region; use a local copy.

## OpenACC and OpenMP

OpenACC is frozen: no new OpenACC-only features, and new kernels get OpenMP directives first.
The plan and criteria for retiring it are in `doc/ROADMAP.md` (priority 2).

## Recording results

Write measurements and findings to `doc/GPU_RESULTS.md` (date, node, GPU, driver, nvfortran
version, commands, numbers, profiles summarised), commit with the code, and push the branch.

## Next

What the next sessions work on, and in which order, is in `doc/ROADMAP.md`. Read
`doc/GPU_RESULTS.md` ("Status") for the state the last session left.
