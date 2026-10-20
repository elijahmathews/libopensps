#!/bin/bash
# Generates the libopensps parity fixture matrix (OPSF containers) into OUTDIR.
# Usage: tests/generate_opensps_fixtures.sh OUTDIR
#
# Cases are read from tests/data/param_cases.psv, one per line:
#   name|isoc|spec|dust|flags|key=value overrides
# flags: "csp" omits the SSP spectral grid (it duplicates the case's SSP-family base).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
OUTDIR="$(realpath -m "${1:?usage: $0 OUTDIR}")"
GENERATOR="$PROJECT_ROOT/build/generate_opensps_fixtures"

export SPS_HOME="$PROJECT_ROOT"
export OPENSPS_DATA_HOME="$PROJECT_ROOT"
export FSPS_DATA_HOME="$PROJECT_ROOT"

meson compile -C "$PROJECT_ROOT/build" generate_opensps_fixtures > /dev/null
mkdir -p "$OUTDIR"

# The case matrix is shared with the parameterized regression test (test_param_regression)
CASES="$SCRIPT_DIR/data/param_cases.psv"
cases=()
while IFS= read -r line; do
  [[ -z "${line// }" || "$line" == \#* ]] || cases+=("$line")
done < "$CASES"

for line in "${cases[@]}"; do
  IFS='|' read -r name isoc spec dust flags overrides <<< "$line"
  extra=()
  [[ "$flags" == *csp* ]] && extra+=(--csp-only)
  echo "== $name"
  # shellcheck disable=SC2086
  (cd "$OUTDIR" && "$GENERATOR" --isoc "$isoc" --spec "$spec" --dust "$dust" "${extra[@]}" \
      --out "$name.opsf" $overrides > "$name.log" 2>&1) || { echo "FAILED: $name (see $OUTDIR/$name.log)"; continue; }
  printf '%s|%s|%s|%s|%s|%s\n' "$name" "$isoc" "$spec" "$dust" "$flags" "$overrides" >> "$OUTDIR/MANIFEST.tmp"
done

mv "$OUTDIR/MANIFEST.tmp" "$OUTDIR/MANIFEST.psv"
echo "Fixtures written to $OUTDIR"
