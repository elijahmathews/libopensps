#!/bin/bash
# Runs every case of a case table through one build's generate_opensps_fixtures, writing
# OUTDIR/<case>.opsf (compare two builds with compare_builds.py).
#
# Usage: tools/compare_builds/run_cases.sh BUILD_DIR OUTDIR [CASES.psv]
# Environment: JOBS (parallel cases, default 4); FIXTURE_ARGS (extra generate_opensps_fixtures
# options, e.g. --batch); DATA_HOME (data root, default the repository; with FSPS_DATA_FORMAT=standard
# and a converted root this compares the two data layouts with one build).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$(realpath "${1:?usage: $0 BUILD_DIR OUTDIR [CASES.psv]}")"
OUT="$(realpath -m "${2:?usage: $0 BUILD_DIR OUTDIR [CASES.psv]}")"
CASES="$(realpath "${3:-$ROOT/tests/data/param_cases.psv}")"
mkdir -p "$OUT"
run_case() {
    local name isoc spec dust flags overrides extra=()
    IFS='|' read -r name isoc spec dust flags overrides <<< "$1"
    [[ "$flags" == *csp* ]] && extra+=(--csp-only)
    # shellcheck disable=SC2206
    extra+=(${FIXTURE_ARGS:-})
    # shellcheck disable=SC2086
    (cd "$OUT" && OPENSPS_DATA_HOME="$DATA_HOME" SPS_HOME="$DATA_HOME" FSPS_DATA_HOME="$DATA_HOME" "$BUILD/generate_opensps_fixtures" \
        --isoc "$isoc" --spec "$spec" --dust "$dust" "${extra[@]}" --out "$name.opsf" $overrides \
        > "$name.log" 2>&1) || { echo "failed: $name (see $OUT/$name.log)" >&2; return 1; }
}
export -f run_case
DATA_HOME="${DATA_HOME:-$ROOT}"
export ROOT BUILD OUT FIXTURE_ARGS DATA_HOME
grep -v '^#' "$CASES" | grep . | tr '\n' '\0' | xargs -0 -P "${JOBS:-4}" -I{} bash -c 'run_case "$1"' _ {}
