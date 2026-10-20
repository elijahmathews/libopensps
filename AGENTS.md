# Agent guide

Instructions for AI coding agents working on libopensps (and a quick reference for people).
Read this first; the linked documents hold the detail. Keep this file short and current:
when a rule here changes, change it here.

## The project

A modernized fork of FSPS (Flexible Stellar Population Synthesis; Conroy et al.), in Fortran
2008+ (or C/C++/etc. where applicable), built with Meson. Goals: physics that is correct and
traceable to legacy FSPS or to independent identities, and fast forward models (FP32 kernels,
GPU offload, batched SSPs and CSPs for SED fitting).

- `src/core` (contexts, types, precision), `src/physics` (IMF, isochrones, SSP, CSP, dust,
  nebular, AGN, batched CSP), `src/spectra` (libraries, filters, indices, batched photometry),
  `src/math`, `src/io`, `src/bindings` (C API; `include/`).
- Two real kinds (`src/core/opensps_precision.f90`): `WP` is always FP64 (setup, isochrones, IMF
  and SFH weights, the public API); `KP` is the kernel precision, FP64 by default and FP32
  with `-Dkernel_precision=single`.
- Assume the user is an astronomer and an SPS expert: explain software matters plainly; justify
  every physics decision with a reason that would satisfy a referee.

## Build and test

Data (stellar libraries, isochrones, nebular grids, dust, filters, regression references) is
not in the repository: it comes from versioned releases of `elijahmathews/opensps-data`.

```
python3 tools/fetch_data.py ~/opensps-data --release <release pinned by the repository>
export OPENSPS_DATA_HOME=~/opensps-data
meson setup build --buildtype=release                              # CPU, FP64
meson setup build32 --buildtype=release -Dkernel_precision=single  # FP32 kernels
meson compile -C build
meson test -C build --suite unit                       # needs no data
meson test -C build --suite unit --suite regression    # needs OPENSPS_DATA_HOME
```

`-Ddata_home=<root>` points the test suites at another data root. Without data, the
regression tests that need it are skipped with a message, not failed: say so when reporting.

GPU builds (details and per-machine notes: `doc/GPU_PLAN.md`, results: `doc/GPU_RESULTS.md`;
what comes next: `doc/ROADMAP.md`):

- OpenACC: `FC=nvfortran CC=nvc meson setup b -Daccel=gpu -Dgpu_arch=cc80` (add
  `-Dcublas=true` for cuBLAS batch products); `-Daccel=multicore` runs the same code on CPU
  cores. OpenACC is frozen: new kernels get OpenMP directives first.
- OpenMP target: `-Daccel=omp_gpu` with nvfortran or LLVM flang. Run with
  `OMP_TARGET_OFFLOAD=MANDATORY` so a silent host fallback fails.

Other tools: `tests/benchmark_phases`, `tests/benchmark_csp_batch` (timings);
`tools/compare_builds/` (one build against another over the case matrix, e.g. GPU vs CPU);
`tools/legacy_parity/` (against legacy FSPS on identical data); `tools/opensps_data/` (data
format, conversion, packing; format and workflows in `doc/DATA_FORMAT.md`).

## Correctness gates

- The unit and regression suites pass in FP64 **and** FP32 builds before a pull request.
- Regression references change only for an intended, explained change: regenerate them with
  the documented commands, bump the references version and pin
  (`doc/DATA_FORMAT.md`, "regenerating the references"), and state the size of the change
  and its cause in the pull request. Publishing the new references is the user's step.
- Physics changes are checked against legacy FSPS (`tools/legacy_parity/run_matrix.sh`); every
  deliberate difference is listed with its cause in `tools/legacy_parity/README.md`.
- GPU builds match the CPU build: FP64 to ~1e-10 per pixel, FP32 to ~5e-7
  (`tools/compare_builds`).
- Data format changes keep reading the previous format (a documented transition alias) and
  keep the parity suite bit-identical.
- Never claim a test passed without having run it; report failures with their output, and
  skipped tests as skipped.

## Hard rules

- Agents must follow the same rules as humans for contributions to the codebase. Refer to
  `CONTRIBUTING.md` for contribution rules.
- Never `-ffast-math`, nvfortran `-fast`, or any flag that reassociates floating point: it
  deletes the compensated sums.
- FP32 kernels follow `doc/MIXED_PRECISION.md` ("Rules for FP32 kernels"): compensated or
  pairwise long sums; no differences of nearly equal integrals; no FP32 logarithms that are
  exponentiated later; grid-only factors formed in FP64; physical units; floors of
  `tiny(1.0_KP)`. Test tolerances in FP32 builds use `KP` (e.g. 100 eps of `KP`).
- Device code (OpenACC and OpenMP target):
  - `!$acc routine seq` on every routine called from device code;
  - no libm calls (e.g. `log_gamma`) without a device-safe fallback;
  - `contiguous` dummies (no runtime copy-in/out);
  - no array constructors or other temporaries (LLVM flang mallocs them on the device);
  - map a derived type either whole or by component, never both;
  - map everything a kernel touches explicitly: NVIDIA GPUs may tolerate a missing map,
    AMD GPUs fault.
  More in `doc/GPU_PLAN.md` ("Rules learned so far").
- Keep `compute_csp_scenario` and `compute_csp_batch` physically identical; a physics change
  goes into both and the batch test (`test_csp_batch`) checks them against each other.

## Git workflow

- Work on a topic branch from `main`; commit small, verified steps and push often:
  `git push origin <branch>` (GitHub). Never push to other remotes.
- Where possible, PRs should be small, isolated changes to the codebase and be merged in
  as a squashed commit (thus making the individual commit messages and numbers of commits
  somewhat irrelevant on the PRs themselves). This keeps the development history on `main`
  linear with each commit having green CI and good behavior. Large, structural changes
  should likely be handled on a larger development branch which is also provided protections.
- Agents should commit unsigned (signing may need a hardware key):
  `git -c commit.gpgsign=false commit`. The user is the author; end the message with a
  `Co-Authored-By:` trailer naming the agent. Humans are asked to commit signed if
  possible, but not a hard requirement.
- Open pull requests against `main` (CI runs only there; see above rules for PRs); do not
  merge them, the user does. Stacked branches: say the merge order in the PR, or set up as
  a GitHub stack.
- Commit messages: follow the format and style mandated in `CONTRIBUTING.md` (Conventional
  Commits). Because pull requests are squash-merged, the **pull request title and description
  become the commit on `main`**: give the title a Conventional Commits form and the
  description the body that commit needs (what changed, why, numbers, breaking changes),
  ending with the `Co-Authored-By:` trailer.

## Machines

- Penn State ICDS cluster: `module load nvhpc/25.1` (CUDA 12; P100 is `cc60`, A100 `cc80`).
  HPC SDK 26 dropped Pascal and Volta.
- User `elijahmathews`'s laptop has an AMD Radeon 780M iGPU (gfx1103) that also drives the
  display. A GPU fault or hang there resets the GPU and kills the desktop session. Do not
  run GPU offload tests on it without the user's go-ahead; debug with host runs
  (`OMP_TARGET_OFFLOAD=DISABLED`) and by reading the device IR first. Unset
  `HSA_OVERRIDE_GFX_VERSION` for gfx1103 builds.
