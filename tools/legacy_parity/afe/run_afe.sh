#!/bin/bash
# [alpha/Fe] parity: legacy FSPS v4 (AFE_FLAG build: MIST v2.5 + C3K_LR) against this fork at the five
# [alpha/Fe] nodes of one metallicity (see ../README.md, "[alpha/Fe] against legacy FSPS v4").
#
# Usage: tools/legacy_parity/afe/run_afe.sh LEGACY_V4_SRC LEGACY_V4_SPS_HOME AFE_DATA_ROOT [OUTDIR]
#   LEGACY_V4_SRC       src/ of cconroy20/fsps master (a `git archive` export; read only)
#   LEGACY_V4_SPS_HOME  the same export's root (SPECTRA/, ISOCHRONES/, ...)
#   AFE_DATA_ROOT       data root holding the converted mist_afe and c3k_lr libraries
#                       (tools/opensps_data/convert_legacy.py --upstream-mist/--upstream-c3k)
# Environment: ZMET (metallicity index, default 11 = solar), EVTYPE (-1 = all phases, or 0-9),
#              AGB_DUST (default 1), BUILD (fork build directory, default build; relative to the repository root unless absolute)
#
# The legacy copy is patched (in OUTDIR, never in the export) with the fork's constants
# (../patches/constants.sed), and with the [Fe/H] label fix: legacy reads the C3K labels with
# F4.2, which truncates "-2.25" to -2.2 (see the README), so legacy's metallicity grid is
# wrong for every other node unless this is patched. Takes ~2 min.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$(realpath "${1:?usage: $0 LEGACY_V4_SRC LEGACY_V4_SPS_HOME AFE_DATA_ROOT [OUTDIR]}")"
HOME_V4="$(realpath "${2:?}")"
DATA="$(realpath "${3:?}")"
OUT="$(realpath -m "${4:-afe_parity_out}")"
ZMET="${ZMET:-11}"; EV="${EVTYPE:--1}"; DUST="${AGB_DUST:-1}"
BLD="${BUILD:-build}"; case "$BLD" in /*) ;; *) BLD="$ROOT/$BLD" ;; esac

mkdir -p "$OUT"; rm -rf "$OUT/src"; cp -r "$SRC" "$OUT/src"
sed -i -f "$HERE/../patches/constants.sed" "$OUT/src/sps_vars.f90"
sed -i "s/READ(zlegend_str2(z),'(F4.2)')/READ(zlegend_str2(z),'(F5.2)')/" "$OUT/src/sps_setup.f90"
grep -q "F5.2" "$OUT/src/sps_setup.f90" || { echo "F4.2 patch did not apply" >&2; exit 1; }
cp "$HERE/legacy_dump.f90" "$OUT/src/"
(
    cd "$OUT/src"; rm -f ./*.o ./*.mod
    FL="-O2 -cpp -DAFE_FLAG=1"
    gfortran $FL -c sps_vars.f90 && gfortran $FL -c sps_utils.f90
    objs=$(sed -n '/^COMMON/,/agn_dust.o/p' Makefile | sed 's/COMMON =//;s/\\//' | tr -d '\n')
    for o in $objs; do [ -f "$o" ] || gfortran $FL -c "${o%.o}.f90"; done
    gfortran $FL legacy_dump.f90 -o legacy_dump $objs
) > "$OUT/legacy_build.log" 2>&1
(cd "$OUT" && SPS_HOME="$HOME_V4" ./src/legacy_dump "$ZMET" "$EV" "$DUST" > legacy.log 2>&1)

# The fork's side: a driver linked against the fork's library
meson compile -C "$BLD" opensps > /dev/null
MODDIR="$(dirname "$(find "$BLD" -name opensps_types.mod | head -1)")"
gfortran -I"$MODDIR" "$HERE/opensps_dump.f90" -L"$BLD" -lopensps -Wl,-rpath,"$BLD" -o "$OUT/opensps_dump"
(cd "$OUT" && OPENSPS_DATA_HOME="$DATA" ./opensps_dump "$ZMET" "$EV" "$DUST" > fork.log 2>&1)

python3 "$HERE/compare_afe.py" "$OUT/legacy_z$(printf %02d "$ZMET")_ev${EV}.bin" "$OUT/opensps_z$(printf %02d "$ZMET")_ev${EV}.bin"
