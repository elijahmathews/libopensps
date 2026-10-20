#!/bin/bash
# Runs every case of tests/data/param_cases.psv through legacy FSPS and this fork, on identical
# data, and compares them case by case with compare_matrix.py (see README.md, "Parity matrix").
#
# Usage: tools/legacy_parity/run_matrix.sh LEGACY_SRC LEGACY_SPS_HOME [OUTDIR]
#   LEGACY_SRC       clean legacy FSPS src/ directory (e.g. git -C fsps archive HEAD src | tar -x)
#   LEGACY_SPS_HOME  legacy-layout data tree whose files match this fork's data/ tree
#   OUTDIR           scratch directory (default: ./legacy_parity_matrix)
# Environment: JOBS (parallel cases, default 4), BUILD (fork build directory, default build),
#              CASES (case table, default tests/data/param_cases.psv)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEGACY_SRC="$(realpath "${1:?usage: $0 LEGACY_SRC LEGACY_SPS_HOME [OUTDIR]}")"
LEGACY_HOME="$(realpath "${2:?usage: $0 LEGACY_SRC LEGACY_SPS_HOME [OUTDIR]}")"
OUT="$(realpath -m "${3:-legacy_parity_matrix}")"
JOBS="${JOBS:-4}"
BUILD="$(realpath -m "$ROOT/${BUILD:-build}")"
CASES="$(realpath "${CASES:-$ROOT/tests/data/param_cases.psv}")"

# Legacy builds, one per library combination used by the cases (isoc|spec|dust -> flags)
declare -A FLAGS=(
    [mist_miles_DL07]=""
    [mist_miles_themis]="-DDL07=0 -DTHEMIS=1"
    [pdva_basel_DL07]="-DMIST=0 -DPADOVA=1 -DMILES=0 -DBASEL=1"
    [mist_c3k_DL07]="-DMILES=0 -DC3K=1"
    [bpss_bpass_DL07]="-DMIST=0 -DBPASS=1"
)

mkdir -p "$OUT/legacy" "$OUT/fork"

# 1. Data: the legacy tree, with the corrected nebular line tables (upstream fc7db5a) that the
#    fork ships; legacy's own tables carry wrong metallicity and age labels
rm -rf "$OUT/sps_home"
mkdir -p "$OUT/sps_home/nebular"
for d in "$LEGACY_HOME"/*; do
    [ "$(basename "$d")" = nebular ] || ln -s "$d" "$OUT/sps_home/"
done
ln -s "$LEGACY_HOME"/nebular/* "$OUT/sps_home/nebular/"
for k in ND_mist ND_pdva ND_prsc WD_mist WD_pdva WD_prsc; do
    rm "$OUT/sps_home/nebular/ZAU_$k.lines"
    cp "$ROOT/data/nebular/ZAU_$k.lines" "$OUT/sps_home/nebular/"
done

# 2. Legacy builds with the fork's constants and the upstream AGN fix (patches/)
build_legacy() {
    local name=$1 flags=$2 dir="$OUT/build_$1"
    rm -rf "$dir"
    cp -r "$LEGACY_SRC" "$dir"
    sed -i -f "$HERE/patches/constants.sed" "$dir/sps_vars.f90"
    sed -i -f "$HERE/patches/agn_dust.sed" "$dir/agn_dust.f90"
    grep -q '4.0479831348E10' "$dir/sps_vars.f90" || { echo "constants patch did not apply" >&2; return 1; }
    grep -q 'pset%dust2 \* attn_curve' "$dir/agn_dust.f90" || { echo "AGN patch did not apply" >&2; return 1; }
    cp "$HERE/csp_dump.f90" "$dir/"
    (
        cd "$dir"
        rm -f ./*.o ./*.mod
        F90FLAGS="$flags" make -s simple > build.log 2>&1
        objs=$(ls ./*.o | grep -vE '/(simple|lesssimple|autosps|spec_bin)\.o$')
        # shellcheck disable=SC2086
        gfortran -O3 -cpp $flags csp_dump.f90 $objs -o csp_dump >> build.log 2>&1
    ) || { echo "legacy build $name failed (see $dir/build.log)" >&2; return 1; }
}
pids=()
for name in "${!FLAGS[@]}"; do
    build_legacy "$name" "${FLAGS[$name]}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# 3. Fork
meson compile -C "$BUILD" generate_opensps_fixtures > /dev/null

# 4. Cases
run_case() {
    local name isoc spec dust flags overrides extra=()
    IFS='|' read -r name isoc spec dust flags overrides <<< "$1"
    [[ "$flags" == *csp* ]] && extra+=(--csp-only)
    local legacy="$OUT/build_${isoc}_${spec%%_*}_${dust}"   # c3k_afe+0.0 -> c3k
    [ -x "$legacy/csp_dump" ] || { echo "no legacy build for $isoc|$spec|$dust ($name)" >&2; return 1; }
    # shellcheck disable=SC2086
    (cd "$legacy" && SPS_HOME="$OUT/sps_home" ./csp_dump --out "$OUT/legacy/$name.opsf" \
        "${extra[@]}" $overrides > "$OUT/legacy/$name.log" 2>&1) || { echo "legacy failed: $name" >&2; return 1; }
    # shellcheck disable=SC2086
    (cd "$OUT/fork" && OPENSPS_DATA_HOME="$ROOT" SPS_HOME="$ROOT" FSPS_DATA_HOME="$ROOT" "$BUILD/generate_opensps_fixtures" \
        --isoc "$isoc" --spec "$spec" --dust "$dust" "${extra[@]}" --out "$name.opsf" $overrides \
        > "$name.log" 2>&1) || { echo "fork failed: $name" >&2; return 1; }
}
export -f run_case
export OUT ROOT BUILD
status=0
grep -v '^#' "$CASES" | grep . | tr '\n' '\0' | xargs -0 -P "$JOBS" -I{} bash -c 'run_case "$1"' _ {} || status=1

# 5. Compare (failed cases are reported as missing)
python3 "$HERE/compare_matrix.py" "$OUT" "$CASES" | tee "$OUT/report.txt" || status=1
exit $status
