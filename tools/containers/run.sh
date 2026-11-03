#!/bin/bash
# Runs a command in a libopensps container with the GPU, the source tree (read-only), the data
# (read-only) and a build directory mounted, as /src, /data and /build (rootless podman).
#
# Usage: tools/containers/run.sh IMAGE COMMAND...
# Environment: SRC (source tree, default this repository), DATA (default OPENSPS_DATA_HOME),
#              BUILD (writable build directory, default ./container-build),
#              ROCR_VISIBLE_DEVICES (passed through: select the GPU on machines with several).
# Example: tools/containers/run.sh localhost/opensps-rocm:7.2.4 \
#              bash -c 'FC=amdflang CC=amdclang meson setup /build/b /src -Daccel=omp_gpu -Dgpu_arch=gfx1201'
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${1:?usage: $0 IMAGE COMMAND...}"; shift
SRC="$(realpath "${SRC:-$HERE/../..}")"
DATA="$(realpath "${DATA:-${OPENSPS_DATA_HOME:?set DATA or OPENSPS_DATA_HOME}}")"
BUILD="$(realpath -m "${BUILD:-container-build}")"
mkdir -p "$BUILD"
# All of /dev/dri: with only one GPU's nodes the ROCm runtime fails when the kernel's topology
# lists several GPUs; select the GPU with ROCR_VISIBLE_DEVICES instead
exec podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
    --security-opt label=disable --ulimit core=0 \
    ${ROCR_VISIBLE_DEVICES:+-e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"} \
    -e OMP_TARGET_OFFLOAD="${OMP_TARGET_OFFLOAD:-MANDATORY}" \
    -e OPENSPS_DATA_HOME=/data -e FSPS_DATA_HOME=/data -e SPS_HOME=/data \
    -v "$SRC:/src:ro" -v "$DATA:/data:ro" -v "$BUILD:/build" -w /build "$IMAGE" "$@"
