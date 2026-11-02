# Containers

Pinned toolchains for builds, comparisons and profiling outside the host's own packages.

| Directory | Image | For |
|---|---|---|
| `rocm-7.2.4/` | ROCm 7.2.4 on Ubuntu 24.04 (`rocm/dev-ubuntu-24.04:7.2.4-complete`, ~22 GB) | `amdflang` (LLVM flang 22, ROCm's build) and `rocprofv3`: per-kernel GPU profiles on AMD GPUs, and a second ROCm version to compare against the host's |
| `flang-22-fc44/` | Fedora 44 with LLVM flang 22.1.8 (~1.5 GB) | the R9700 desktop's exact compiler elsewhere, e.g. on a cluster through Apptainer; its OpenMP offload runtime has both the AMD and the CUDA (nvptx) plugins |

Build (rootless podman):

```
podman build -t localhost/opensps-rocm:7.2.4 tools/containers/rocm-7.2.4
podman build -t localhost/opensps-flang:22.1.8-fc44 tools/containers/flang-22-fc44
```

Run with the GPU, the source (read-only), the data and a build directory (`run.sh`):

```
ROCR_VISIBLE_DEVICES=GPU-<uuid> BUILD=$PWD/cbuild tools/containers/run.sh localhost/opensps-rocm:7.2.4 \
  bash -c 'FC=amdflang CC=amdclang meson setup /build/b /src --buildtype=release -Daccel=omp_gpu \
           -Dgpu_arch=gfx1201 && meson compile -C /build/b && meson test -C /build/b --suite unit'
```

In the ROCm image the builds need no device-library flags (ROCm's standard layout). Profile with
`rocprofv3 --kernel-trace --stats -d /build/prof -o run --output-format csv -- <benchmark>`
(`benchmark_csp_batch N 0 CONFIG N` times one batch size).

On a cluster without root (Apptainer): `podman save -o img.tar <image>`, copy, then
`apptainer build img.sif docker-archive://img.tar` and `apptainer exec --nv img.sif ...` (NVIDIA:
bind a CUDA toolkit for `ptxas`/`nvlink`, `--bind $CUDA_HOME`, and pass `--cuda-path`).

Notes:

- Ubuntu 24.04's Meson (1.3.2) does not know LLVM flang (it passes classic-flang flags): the
  ROCm image installs Meson 1.11.2 with pip. `rocprofv3` needs `libdw1`.
- `rocprofv3 --pmc` (hardware counters) aborts if the machine has a GPU it has no counters for
  (e.g. an APU's iGPU, gfx1036); kernel traces work. Passing only one GPU's `/dev/dri` nodes does
  not help: the ROCm runtime then fails to initialize.
- The images are pinned (base image tags, compiler versions); update them deliberately and note
  the versions in results.
