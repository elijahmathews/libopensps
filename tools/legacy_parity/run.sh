#!/bin/bash
# Builds legacy FSPS with the exact surface-gravity constant, dumps a Chabrier solar MIST + MILES
# SSP grid from it and from this fork (AGB dust off), and compares them (see README.md).
#
# Usage: tools/legacy_parity/run.sh LEGACY_SRC LEGACY_SPS_HOME [OUTDIR]
#   LEGACY_SRC       legacy FSPS src/ directory (built MIST + MILES, the defaults in sps_vars.f90)
#   LEGACY_SPS_HOME  legacy-layout data tree whose files match this fork's data/ tree
#   OUTDIR           scratch directory (default: ./legacy_parity_out)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEGACY_SRC="$(realpath "${1:?usage: $0 LEGACY_SRC LEGACY_SPS_HOME [OUTDIR]}")"
LEGACY_HOME="$(realpath "${2:?usage: $0 LEGACY_SRC LEGACY_SPS_HOME [OUTDIR]}")"
OUT="$(realpath -m "${3:-legacy_parity_out}")"
IMF_TYPE=1
AGB_DUST=0

# 1. Legacy build with gsig4pi = 1/(4 pi sigma) exactly (legacy rounds it to 1/4.13e10, 1.77% off)
mkdir -p "$OUT"
rm -rf "$OUT/src"
cp -r "$LEGACY_SRC" "$OUT/src"
sed -i 's|gsig4pi = 1/4.13E10|gsig4pi = 1/40584229580.29736_sp|' "$OUT/src/sps_vars.f90"
grep -q '40584229580' "$OUT/src/sps_vars.f90" || { echo "gsig4pi patch did not apply" >&2; exit 1; }
cp "$HERE/ssp_dump.f90" "$OUT/src/"
(
    cd "$OUT/src"
    rm -f ./*.o ./*.mod
    make -s simple > build.log 2>&1
    objs=$(ls ./*.o | grep -vE '/(simple|lesssimple|autosps|spec_bin)\.o$')
    gfortran -O3 -cpp ssp_dump.f90 $objs -o ssp_dump >> build.log 2>&1
)
(cd "$OUT" && SPS_HOME="$LEGACY_HOME" ./src/ssp_dump "$IMF_TYPE" "$AGB_DUST" > legacy.log 2>&1)

# 2. Fork
meson compile -C "$ROOT/${BUILD:-build}" generate_opensps_fixtures > /dev/null
(cd "$OUT" && OPENSPS_DATA_HOME="$ROOT" SPS_HOME="$ROOT" FSPS_DATA_HOME="$ROOT" "$ROOT/${BUILD:-build}/generate_opensps_fixtures" \
    --isoc mist --spec miles --dust DL07 --out fork_ssp.opsf \
    zmet=10 imf_type="$IMF_TYPE" add_agb_dust_model="$AGB_DUST" > fork.log 2>&1)

# 3. Compare
python3 "$HERE/compare_legacy.py" "$OUT/fork_ssp.opsf" "$OUT/legacy_ssp.bin"
