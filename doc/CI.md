# Continuous integration

Three workflows, split by how often a check needs to run:

| Workflow | When | What |
|---|---|---|
| `ci.yml` | Every push to `develop`/`master`, every pull request (skipped for documentation-only changes) | Six compiler/hardware builds running the unit and regression suites; AddressSanitizer and ThreadSanitizer jobs running focused stress phases |
| `nightly.yml` | Daily, and on demand | All stress phases at more iterations (ASan, TSan, nvfortran OpenACC); newest gfortran on Linux and macOS; optional GPU job on a self-hosted runner |
| `packaging.yml` | Changes to packaging or build files, pushes to `master`, weekly, on demand | Fedora RPM and Ubuntu DEB, each running the test suite with the distribution's compiler |

Pull requests that supersede a running build cancel it. Every job has a timeout.

## What each per-change job covers

| Job | Compiler | Hardware | Notes |
|---|---|---|---|
| gfortran / Linux x86-64 / coverage | gfortran (Ubuntu 24.04) | x86-64 | Only job with the full data tree (C3K); uploads coverage |
| gfortran / Linux arm64 | gfortran | arm64 (Ampere) | Linux on ARM |
| gfortran / macOS arm64 | gfortran 14 (Homebrew) | Apple silicon | macOS toolchain and linker |
| LLVM flang / Linux x86-64 | flang 20 | x86-64 | LLVM front end |
| Intel ifx / Linux x86-64 | ifx 2026.1 | x86-64 | Intel LLVM-based compiler |
| nvfortran / OpenACC multicore | nvfortran 26.1, `-Daccel=multicore` | x86-64 | OpenACC directive paths on host cores |
| AddressSanitizer + UBSan | gfortran | x86-64 | Unit tests; stress phases A (leaks) and C (reallocation) |
| ThreadSanitizer | gfortran + OpenMP threads | x86-64 | Unit tests; stress phases B (concurrent compute) and D (concurrent setup/release) |

Every non-sanitizer job runs the complete unit and regression suites, including the physics
oracles and all 34 cases of the parameterized regression. The regression references were
generated with gfortran on x86-64 and are compared at a relative tolerance of 1e-5, so the other
compilers and the arm64 machines are checked against them, not only against themselves.

## Test reporting

The test programs built on `tests/test_utils.f90` print one TAP line per assertion when
`FSPS_TEST_TAP=1` (meson sets it), and coloured `[PASS]`/`[FAIL]` lines when run by hand. Meson
parses the TAP, so:

- each of the 21 unit-test modules is its own meson test (`unit_sfh`, `unit_dust`, ...), run in
  parallel and runnable alone: `meson test -C build unit_sfh`;
- `meson test` shows the assertion count per test and names failing assertions;
- `build/meson-logs/testlog.junit.xml` has every assertion as a test case, which CI publishes
  (job summary and pull-request annotations, via `mikepenz/action-junit-report`).

A test program built on `test_utils` must call `tap_finish()` before exiting, and should print
diagnostic lines starting with `#` so that they are TAP comments.

## Design decisions

- **Compilers are installed, not pulled as containers.** `fortran-lang/setup-fortran` installs
  ifx and nvfortran from the vendors' package repositories. The previous Intel and NVIDIA
  container images took 6-7 minutes each to pull, and the NVIDIA job then spent 11 minutes in
  `dnf update`.
- **Sparse checkout.** The 600 MB C3K spectral library is checked out only by the coverage job.
  Elsewhere `meson.build` sees that `data/spectra/C3K` is missing, does not register the C3K
  regression test, and the parameterized regression skips its C3K case (`--skip-spec c3k`).
  Checkouts use a blobless partial clone so the skipped files are never downloaded.
- **Sharded regression.** The parameterized regression is registered as three tests (`ssp_`,
  `csp_`, `lib_` cases), which meson runs in parallel with the other regression tests.
- **One job per sanitizer, focused on what it finds.** ThreadSanitizer slows code by 5-15x and the
  stress test reloads all data in every iteration, so running every phase under it exceeded the
  40-minute limit on every run. TSan now runs the two concurrent phases; ASan the leak and
  reallocation phases. Nightly runs every phase under both, with longer timeouts.
- **No separate CPU-OpenMP regression job.** OpenMP is used for host threading only in the stress
  harness (covered by the TSan job), and the library's GPU direction is OpenMP target offload,
  which needs its own axis (below), not a rerun of the CPU suites.
- **Packaging runs when it can change.** RPM and DEB builds do not depend on ordinary source edits,
  which `ci.yml` already builds and tests.

## Hardware and compilers not covered by hosted runners

GitHub's hosted runners have no GPUs. The nightly workflow has a `gpu-openacc` job that runs on a
self-hosted runner with the labels `self-hosted`, `linux`, `gpu-nvidia` and the NVIDIA HPC SDK
installed. It is skipped unless the repository variable `GPU_RUNNER` is `true`. To enable it:

1. Register the machine as a runner (repository Settings, Actions, Runners) with the extra label
   `gpu-nvidia`, and install the NVIDIA HPC SDK so that `nvfortran` and `nvc` are on its `PATH`.
2. Set the repository variable `GPU_RUNNER=true` (Settings, Secrets and variables, Actions).

A self-hosted runner on a public repository can be asked to run code from pull requests; keep the
GPU job on `schedule`/`workflow_dispatch` only (as now), and restrict the runner to this
repository.

GitHub also offers paid GPU runners (NVIDIA T4) for organization accounts; the same job works
there by changing `runs-on`.

## Adding OpenMP target offload

The planned move from OpenACC to OpenMP target offload adds a backend axis. Most of it can be
tested without a GPU: compile with each vendor's offload flags, and run the suites with
`OMP_TARGET_OFFLOAD=DISABLED`, which executes the target regions on the host. That checks that
the directives compile and that the host fallback gives the reference results, on every compiler.

| Compiler | Offload flags (indicative) | Hosted-runner check | Device check |
|---|---|---|---|
| gfortran | `-fopenmp -foffload=nvptx-none` (needs the `gcc-offload-nvptx` package) | Host fallback | NVIDIA GPU |
| ifx | `-fiopenmp -fopenmp-targets=spir64` | Host fallback, or Intel's CPU OpenCL device | Intel GPU |
| nvfortran | `-mp=gpu -gpu=...` | Host fallback | NVIDIA GPU |
| AMD flang (`amdflang`, ROCm) | `-fopenmp --offload-arch=gfx...` | Compile and host fallback in a ROCm container | AMD GPU |

Suggested steps when the OpenMP backend lands:

1. Add a meson option (for example `-Doffload=openmp`) next to `accel`, setting the per-compiler
   flags above.
2. In `ci.yml`, add host-fallback rows for gfortran, ifx and nvfortran with
   `OMP_TARGET_OFFLOAD=DISABLED`, and keep the OpenACC row until OpenACC is retired.
3. In `nightly.yml`, add the AMD flang row (ROCm container; larger download, so not per-change)
   and device rows on self-hosted GPU runners (`gpu-nvidia`, and `gpu-amd` if available), run with
   `OMP_TARGET_OFFLOAD=MANDATORY` so that a silent fallback to the host fails the job.

## Running the same checks locally

```
meson setup build --buildtype=release && meson compile -C build
meson test -C build --suite unit --suite regression

# ThreadSanitizer stress phases, as in CI
meson setup build_tsan -Denable_stress_testing=true -Db_sanitize=thread -Denable_openmp=true
meson compile -C build_tsan
TSAN_OPTIONS="suppressions=$PWD/tests/tsan.supp halt_on_error=1" \
  FSPS_STRESS_PHASES=BD FSPS_STRESS_ITERS=4 meson test -C build_tsan --suite stress
```

Workflows are checked with [actionlint](https://github.com/rhysd/actionlint)
(`.github/actionlint.yaml` declares the self-hosted runner label).
