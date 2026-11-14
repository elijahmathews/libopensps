<p align="center">
  <img src="assets/logo.svg" alt="OpenSPS logo" width="180px" height="180px" />
</p>

<h1 align="center">OpenSPS</h1>

<p align="center">
  <strong>A High-Performance, Hardware-Accelerated Stellar Population Synthesis Library</strong>
</p>

<p align="center">
  <a href="https://github.com/elijahmathews/libopensps/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/elijahmathews/libopensps/ci.yml?label=CI&logo=github&logoColor=white" alt="CI Status"></a>
  <a href="https://codecov.io/gh/elijahmathews/libopensps"><img src="https://img.shields.io/codecov/c/github/elijahmathews/libopensps?label=Coverage&logo=codecov&logoColor=white" alt="Coverage"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NDAgNjQwIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIHY3LjMuMSBieSBAZm9udGF3ZXNvbWUgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbSBMaWNlbnNlIC0gaHR0cHM6Ly9mb250YXdlc29tZS5jb20vbGljZW5zZS9mcmVlIENvcHlyaWdodCAyMDI2IEZvbnRpY29ucywgSW5jLi0tPjxwYXRoIGZpbGw9IiNmZmZmZmYiIGQ9Ik0zODQgOTZMNTEyIDk2QzUyOS43IDk2IDU0NCAxMTAuMyA1NDQgMTI4QzU0NCAxNDUuNyA1MjkuNyAxNjAgNTEyIDE2MEwzOTguNCAxNjBDMzkzLjIgMTg1LjggMzc1LjUgMjA3LjEgMzUyIDIxNy4zTDM1MiA1MTJMNTEyIDUxMkM1MjkuNyA1MTIgNTQ0IDUyNi4zIDU0NCA1NDRDNTQ0IDU2MS43IDUyOS43IDU3NiA1MTIgNTc2TDEyOCA1NzZDMTEwLjMgNTc2IDk2IDU2MS43IDk2IDU0NEM5NiA1MjYuMyAxMTAuMyA1MTIgMTI4IDUxMkwyODggNTEyTDI4OCAyMTcuM0MyNjQuNSAyMDcgMjQ2LjggMTg1LjcgMjQxLjYgMTYwTDEyOCAxNjBDMTEwLjMgMTYwIDk2IDE0NS43IDk2IDEyOEM5NiAxMTAuMyAxMTAuMyA5NiAxMjggOTZMMjU2IDk2QzI3MC42IDc2LjYgMjkzLjggNjQgMzIwIDY0QzM0Ni4yIDY0IDM2OS40IDc2LjYgMzg0IDk2ek00MzkuNiAzODRMNTg0LjQgMzg0TDUxMiAyNTkuOEw0MzkuNiAzODR6TTUxMiA0ODBDNDQ5LjEgNDgwIDM5Ni44IDQ0NiAzODYgNDAxLjFDMzgzLjQgMzkwLjEgMzg3IDM3OC44IDM5Mi43IDM2OUw0ODcuOSAyMDUuOEM0OTIuOSAxOTcuMiA1MDIuMSAxOTIgNTEyIDE5MkM1MjEuOSAxOTIgNTMxLjEgMTk3LjMgNTM2LjEgMjA1LjhMNjMxLjMgMzY5QzYzNyAzNzguOCA2NDAuNiAzOTAuMSA2MzggNDAxLjFDNjI3LjIgNDQ1LjkgNTc0LjkgNDgwIDUxMiA0ODB6TTEyNi44IDI1OS44TDU0LjQgMzg0TDE5OS4zIDM4NEwxMjYuOCAyNTkuOHpNLjkgNDAxLjFDLTEuNyAzOTAuMSAxLjkgMzc4LjggNy42IDM2OUwxMDIuOCAyMDUuOEMxMDcuOCAxOTcuMiAxMTcgMTkyIDEyNi45IDE5MkMxMzYuOCAxOTIgMTQ2IDE5Ny4zIDE1MSAyMDUuOEwyNDYuMiAzNjlDMjUxLjkgMzc4LjggMjU1LjUgMzkwLjEgMjUyLjkgNDAxLjFDMjQyLjEgNDQ1LjkgMTg5LjggNDgwIDEyNi45IDQ4MEM2NCA0ODAgMTEuNyA0NDYgLjkgNDAxLjF6Ii8+PC9zdmc+Cg==" alt="License"></a>
  <a href="meson.build"><img src="https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Felijahmathews%2Flibopensps%2Fmain%2Fmeson.build&search=version%20%3A%20%27(%5B0-9A-Za-z.%2B-%5D%2B)%27&replace=%241&label=Version&logo=git&logoColor=white" alt="Version"></a>
</p>

<p align="center">
  <a href="#building"><img src="https://img.shields.io/badge/AMD-OpenMP%20Offload%20%7C%20ROCm-ED1C24?logo=amd&logoColor=white" alt="AMD GPU support"></a>
  <a href="#building"><img src="https://img.shields.io/badge/NVIDIA-OpenMP%20Offload%20%7C%20CUDA-76B900?logo=nvidia&logoColor=white" alt="NVIDIA GPU support"></a>
</p>

<p align="center">
  <a href="https://fortran-lang.org"><img src="https://img.shields.io/badge/Fortran-2008%2B-734f96?logo=fortran&logoColor=white" alt="Fortran 2008+"></a>
  <a href="https://github.com/elijahmathews/opensps-python"><img src="https://img.shields.io/badge/Python%20Bindings-opensps--python-3776AB?logo=python&logoColor=white" alt="Python bindings: opensps-python"></a>
</p>

> [!WARNING]
> **OpenSPS is very newly developed software and is probably a bit fragile.** Interfaces may
> change without notice, there are certainly bugs we have not found yet, and nothing here has
> had the years of use that FSPS has. Validate results against your own checks (and against
> legacy FSPS) before relying on them in published work. Buyer beware! If something breaks, please
> [open an issue](https://github.com/elijahmathews/libopensps/issues).

**Contributions are very welcome!** Bug reports, physics checks, new platforms, documentation and
code are all appreciated. See [Contributing](#contributing).

## What is OpenSPS?

OpenSPS is a modernized version of [FSPS](https://github.com/cconroy20/fsps) (Flexible Stellar
Population Synthesis; Conroy, Gunn & White 2009; Conroy & Gunn 2010), written in Fortran 2008+ with
a stable C API. It computes simple and composite stellar populations (SSPs and CSPs): spectra,
photometry, spectral indices, emission lines, stellar masses and bolometric luminosities. The aim is
physics that stays traceable to legacy FSPS, with a code base built for fitting large numbers of
models.

The project is called OpenSPS; this repository and its shared library are named `libopensps`
(`libopensps.so`, header `opensps.h`, C prefix `opensps_`).

What it adds to FSPS:

- **Batched composite populations.** `compute_csp_batch` evaluates thousands of models in one call
  (SFH parameters or a matrix of SSP weights; dust, dust emission, nebular emission, AGN,
  metallicity, redshift, IGM and smoothing per model).
- **GPU offload.** OpenMP target offload runs the SSP template product and the batched CSP on AMD
  (LLVM flang, ROCm) and NVIDIA (nvfortran, CUDA) GPUs, with optional native HIP kernels and
  vendor BLAS (rocBLAS, hipBLAS or cuBLAS) for the matrix products.
- **Analytic gradients.** Reverse-mode (adjoint) derivatives of the batched outputs with respect to
  37 physical parameters, non-parametric SFH weights and enrichment histories, with no
  finite-difference noise.
- **Free IMF parameters per model.** SSP operators decouple the isochrones from the IMF integrals,
  so IMF slopes, characteristic masses and bounds can vary from galaxy to galaxy within a batch.
- **Optional FP32 kernels.** `-Dkernel_precision=single` runs the per-pixel kernels in single
  precision (setup, isochrones and the other weights stay in double). It is much faster on GPUs with slow
  FP64, and the regression suite checks it against the FP64 references. See
  [doc/MIXED_PRECISION.md](doc/MIXED_PRECISION.md) for the accuracy evidence.
- **[α/Fe] as an SSP axis** (FSPS v4's MIST v2.5 and C3K-LR libraries, optional archives of
  `data-v4`) and **cue nebular emission** (`nebular_model = 1`), both with analytic gradients.
- **Thread-safe contexts.** Independent `Context` objects replace FSPS's global state, so several
  setups can live in one process.
- **A stable C API** (`include/opensps.h`) that Python, C++, Julia, Rust and others can bind to.

On the GPU, the batched forward model costs microseconds to tens of microseconds per model, and the
GPU builds are checked against the CPU build over the case matrix in both precisions.
Configurations and measurements are in [doc/GPU_RESULTS.md](doc/GPU_RESULTS.md).

## Quick start

CPU build with gfortran, using the shared data library (GPU builds are covered in
[Building](#building)):

```sh
git clone https://github.com/elijahmathews/libopensps.git
cd libopensps

# Stellar libraries, isochrones, nebular grids, dust, filters (not part of this repository)
python3 tools/fetch_data.py ~/opensps-data --release data-v3
export OPENSPS_DATA_HOME=~/opensps-data

meson setup build --buildtype=release
meson compile -C build
meson test -C build --suite unit --suite regression
```

## Requirements

- A Fortran 2008 compiler (see the table below) and a C compiler. The code builds with
  `-std=f2008` (checked in CI, gfortran) and CI uses gfortran 13 or newer, LLVM flang 20, Intel ifx
  2026.1 and nvfortran 26.1; older compilers are not tested.
- [Meson](https://mesonbuild.com) 1.1 or newer and [Ninja](https://ninja-build.org). LLVM flang needs
  a recent Meson (1.11 has been used); older releases pass classic-flang flags to it. Install with
  `python3 -m pip install --user meson ninja` if your system's version is too old.
- Python 3 (standard library only) for `tools/fetch_data.py`.
- The data. It is published as versioned releases of
  [opensps-data](https://github.com/elijahmathews/opensps-data) and is not in this repository.
  `tools/fetch_data.py` downloads and checksums a release; point `OPENSPS_DATA_HOME` at it. (At run
  time the library also searches `SPS_HOME`, `FSPS_DATA_HOME`, `$XDG_DATA_HOME/fsps`,
  `~/.local/share/fsps` and the system `share/fsps` directories, in that order.) The format is
  described in [doc/DATA_FORMAT.md](doc/DATA_FORMAT.md).

## Building

Supported platforms and compilers, in order of how much they are used and tested:

| Target | Compiler | How it is tested |
|---|---|---|
| **AMD GPU** (`gfx1201`, Radeon AI PRO R9700) | LLVM flang 22 with ROCm | Developed and run on real hardware |
| **NVIDIA GPU** (`cc80`, A100) | NVIDIA HPC SDK `nvfortran` | Run on real hardware |
| Linux x86-64 CPU | gfortran | CI, plus local runs |
| Linux arm64 CPU | gfortran | CI only |
| macOS arm64 CPU | gfortran | CI only |
| Linux x86-64 CPU | Intel ifx, LLVM flang, nvfortran | CI only |

The two GPU targets come first because that is where development and testing happen. The last four
rows are secondary targets (they are on the [roadmap](doc/ROADMAP.md)) that are only exercised by
CI, so expect rougher edges there. Windows is not currently tested. An LLVM flang offload build
that runs on the host (no GPU, or `OMP_TARGET_OFFLOAD=DISABLED`) is not supported: use a CPU build
for host runs ([CHANGELOG.md](CHANGELOG.md), "Known limitations").

### The GPU build flags

Three options control GPU builds, and **we recommend using all three** for GPU execution:

| Option | Values | What it does |
|---|---|---|
| `-Daccel=omp_gpu` | `none` (default), `omp_gpu` | Offloads the SSP template product and the batched CSP with OpenMP target directives. |
| `-Daccel_blas=` | `none` (default), `cublas`, `rocblas`, `hipblas` | Uses the vendor BLAS for the GPU matrix products. `cublas` with nvfortran; `rocblas` or `hipblas` with LLVM flang. |
| `-Daccel_hip=true` | `false` (default), `true` | Builds the native HIP kernels for the bottleneck phases: `hipcc` on AMD ROCm, `nvcc` through a small HIP-on-CUDA shim on NVIDIA. Needs `hipcc` (AMD) or `nvcc` (NVIDIA) at build time. |

`-Dgpu_arch=` selects the device target: an LLVM offload architecture (`gfx1201`, `gfx942`) for
flang, or the `nvfortran -gpu=` value (`cc80` for the A100) for NVIDIA. At run
time, set `OMP_TARGET_OFFLOAD=MANDATORY` so that a missing or unusable GPU stops the program
instead of silently running on the host.

### AMD GPUs (LLVM flang and ROCm)

On Fedora the packages are `rocminfo rocm-hip-devel rocm-runtime-devel rocm-omp-devel
rocm-device-libs flang flang-runtime libomp-devel clang gcc-gfortran meson ninja-build` plus
`llvm`, `clang-tools-extra` and `lld` for the offload linker. The full recipe, with the reasons for
each flag, is in [doc/GPU_PLAN.md](doc/GPU_PLAN.md) ("On the R9700 desktop").

```sh
# Device-library path as your ROCm install lays it out; the second variable is a Meson list.
D=--rocm-device-lib-path=/usr/lib64/rocm/llvm/lib/clang/20/lib/amdgcn/bitcode
LA="['$D','-Xoffload-linker','--library-path=/usr/lib64/amdgcn-amd-amdhsa']"

FC=flang CC=clang meson setup build --buildtype=release \
    -Daccel=omp_gpu -Dgpu_arch=gfx1201 \
    -Daccel_hip=true -Daccel_blas=rocblas \
    -Dfortran_args=$D -Dc_args=$D \
    -Dfortran_link_args="$LA" -Dc_link_args="$LA"
meson compile -C build
OMP_TARGET_OFFLOAD=MANDATORY meson test -C build --suite unit --suite regression --num-processes 1
```

If several AMD GPUs are visible (for example an integrated GPU as well), hide the others with
`ROCR_VISIBLE_DEVICES=GPU-<uuid>` (the UUID is in `rocminfo`). Containers with pinned ROCm and flang
toolchains, which also work on clusters through Apptainer, are in
[tools/containers](tools/containers/README.md); inside the ROCm image the build needs none of the
device-library flags.

### NVIDIA GPUs (nvfortran and CUDA)

Install the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) (the release workflow builds
with 26.1) and make sure `nvfortran`, `nvc` and `nvcc` are on your `PATH`.

```sh
FC=nvfortran CC=nvc meson setup build --buildtype=release \
    -Daccel=omp_gpu -Dgpu_arch=cc80 \
    -Daccel_hip=true -Daccel_blas=cublas \
    -Dcpu_baseline=x86-64-v3
meson compile -C build
OMP_TARGET_OFFLOAD=MANDATORY meson test -C build --suite unit --suite regression --num-processes 1
```

### CPU

```sh
meson setup build --buildtype=release                                # FP64 kernels
meson setup build32 --buildtype=release -Dkernel_precision=single    # FP32 kernels
meson compile -C build
```

Useful options:

- `-Denable_openmp=true` runs the library's OpenMP loops on the host cores.
- `-Dcpu_baseline=generic|x86-64-v3|native` sets the instruction-set level (default `generic`).
- `-Dkernel_precision=single` builds FP32 kernels (see [doc/MIXED_PRECISION.md](doc/MIXED_PRECISION.md)).
- `-Dtests=false` skips the test and benchmark programs (use this when building as a subproject).
- `-Dtrace=true` adds ROCTX/NVTX markers for GPU profiling.

Run both the unit and the regression suites in an FP64 **and** an FP32 build before sending a pull
request; the unit suite needs no data, and the regression suite needs `OPENSPS_DATA_HOME`.

### Prebuilt libraries

Tagged releases attach tarballs (CPU, NVIDIA `cc80`, AMD `gfx1201`) built by
[`release.yml`](.github/workflows/release.yml), plus RPM and DEB packages and a source archive.
The prebuilt libraries need a recent Linux (glibc 2.34 or newer for the NVIDIA build) and, for the
GPU builds, the matching vendor runtime and driver.

### Installing and linking

```sh
meson install -C build        # honours --prefix and DESTDIR
```

installs `libopensps`, `opensps.h` and a pkg-config file (`pkg-config --cflags --libs opensps`).
Projects that build the library as a Meson subproject can use `dependency('opensps')` and usually
pass `-Dtests=false`.

## Using OpenSPS

### From Python

[`elijahmathews/opensps-python`](https://github.com/elijahmathews/opensps-python) is the companion
Python package and the easiest way to use the library. It provides:

- `opensps.CSPBatch`: the batched forward model and its analytic gradients, on CPU or GPU;
- `opensps.StellarPopulation`: a near drop-in replacement for `python-fsps`;
- `opensps.torch` and `opensps.jax`: differentiable PyTorch and JAX interfaces for fitting,
  simulation-based inference and gradient-based samplers.

It loads a `libopensps` you have built or installed; set `OPENSPS_LIB_DIR` or `OPENSPS_LIB_PATH` to
choose which build (for example, your GPU build). Nothing is on PyPI yet, so install it from GitHub
(see that repository's README for current instructions).

```python
import numpy as np
import opensps

ctx = opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07")
res = ctx.batch({"dust2": np.linspace(0.1, 1.0, 1000), "tage": 8.0}).compute()
print(res.mags.shape)   # (1000, nbands)
```

### From C

`include/opensps.h` is the whole interface, and [doc/C_API.md](doc/C_API.md) documents every entry
point, status code and array layout (arrays are column-major). A minimal batched example:

```c
#include <stdio.h>
#include <stdlib.h>
#include <opensps.h>

#define CHECK(s) do { if ((s) != 0) { fprintf(stderr, "opensps error %d at line %d\n", (s), __LINE__); return 1; } } while (0)

int main(void) {
    int h, status;

    opensps_context_create(&h, &status);                             CHECK(status);
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);  CHECK(status);

    /* Parameters shared by every model in the batch. */
    opensps_context_set_float(h, "tau", 2.0, &status);               CHECK(status);

    int nbands, nspec;
    opensps_context_get_nbands(h, &nbands, &status);                 CHECK(status);
    opensps_context_get_nspec(h, &nspec, &status);                   CHECK(status);

    /* Four models that differ in tage and dust2. Outputs are column-major (quantity x n). */
    enum { N = 4 };
    double tage[N]  = {1.0, 3.0, 6.0, 10.0};
    double dust2[N] = {0.1, 0.3, 0.5, 0.8};
    const char *names[2]    = {"tage", "dust2"};
    const double *values[2] = {tage, dust2};

    double *mags = malloc(sizeof(double) * nbands * N);
    double *mass = malloc(sizeof(double) * N);
    opensps_context_compute_csp_batch(h, N, 2, names, values, NULL,
                                      mags, NULL, mass, NULL, NULL, NULL, NULL, NULL, &status);
    CHECK(status);

    for (int i = 0; i < N; i++)
        printf("tage = %5.1f Gyr  dust2 = %.1f  surviving mass = %.4f  mag[0] = %.3f\n",
               tage[i], dust2[i], mass[i], mags[0 + nbands * i]);

    free(mags); free(mass);
    opensps_context_destroy(h, &status);
    return status;
}
```

Compile and run it against an installed or in-tree build (set `OPENSPS_DATA_HOME` first):

```sh
cc example.c -Iinclude -Ibuild -Lbuild -lopensps -o example
LD_LIBRARY_PATH=build ./example
```

Pass `NULL` for any output you do not need; the batch skips what is not requested. The batch has
gradient (`opensps_context_compute_csp_batch_grad`) and component (`..._batch_parts`) variants, and a
single-model path (`opensps_context_compute_csp`). GPU-resident spectra can be kept on the device
(`opensps_context_batch_spec_device`) for pipelines that go on to use them on the GPU.

### From Fortran

The library is a set of Fortran modules under `src/`, but the C API is the supported public
interface. The Fortran module interfaces are internal and may change.

## Testing and validation

```sh
meson test -C build --suite unit                      # no data needed
meson test -C build --suite unit --suite regression   # needs OPENSPS_DATA_HOME
```

The regression suite compares against stored reference outputs from the data release. Other tools:

- `tools/compare_builds`: one build against another over the case matrix (for example the GPU build
  against the same code on the host);
- `tools/legacy_parity`: comparisons with legacy FSPS on identical data;
- `tests/benchmark_phases` and `tests/benchmark_csp_batch`: timings.

[doc/CI.md](doc/CI.md) describes what CI runs.

## Documentation

| | |
|---|---|
| [doc/C_API.md](doc/C_API.md) | The C API reference |
| [doc/DATA_FORMAT.md](doc/DATA_FORMAT.md) | The data format, fetching and packing |
| [doc/GPU_PLAN.md](doc/GPU_PLAN.md), [doc/GPU_RESULTS.md](doc/GPU_RESULTS.md) | GPU builds, rules learned and measurements |
| [doc/MIXED_PRECISION.md](doc/MIXED_PRECISION.md) | The FP32 kernels and their accuracy |
| [doc/ROADMAP.md](doc/ROADMAP.md) | What comes next |
| [doc/RELEASING.md](doc/RELEASING.md) | How a release is made (for maintainers) |
| [doc/MANUAL.pdf](doc/MANUAL.pdf) | The legacy FSPS manual (parameters and physics) |
| [AGENTS.md](AGENTS.md) | Rules for contributors and coding agents |

## Contributing

Contributions are very welcome, whether you are an SPS expert, a GPU programmer or someone who
found a confusing sentence in the documentation. Good places to start are bug reports (with a
minimal reproducer if you can), physics cross-checks against legacy FSPS, testing on hardware and
compilers we cannot reach (especially the secondary targets above), and documentation.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) first: physics must stay correct and traceable, and
the unit and regression suites must pass in FP64 and FP32 builds. Be kind to each other
([Code of Conduct](CODE_OF_CONDUCT.md)); report security problems as described in
[SECURITY.md](SECURITY.md).

## Citation

If you use OpenSPS in your research, please cite it (see [CITATION.cff](CITATION.cff)) **and**
the FSPS papers: [Conroy, Gunn & White (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...699..486C)
and [Conroy & Gunn (2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...712..833C). Cite the data
release you used as well (`data_release.bib` in the data directory).

## License and credits

OpenSPS is released under the [MIT license](LICENSE.md), which retains the FSPS copyright
notice. It is derived from FSPS by Charlie Conroy and contributors; [AUTHORS](AUTHORS) lists
everyone whose work it builds on. Much of the modernized code was written with AI coding agents
(Claude, by Anthropic) under the direction and review of the maintainer.
