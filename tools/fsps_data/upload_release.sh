#!/bin/bash
# Publishes a data release made by pack_data.py as a GitHub release of elijahmathews/opensps-data.
# Not run by any tool or CI: review the release directory first, then run it by hand.
#
# Usage: tools/fsps_data/upload_release.sh RELEASE_DIR TAG [--draft]
#
# RELEASE_DIR is pack_data.py's OUTDIR; UPLOAD.txt there lists the files new in this release (its
# archives, the release manifest TAG.json and DATA_README.md); archives reused from earlier
# releases stay where they are and are not uploaded again. Needs gh, authenticated with write
# access to the data repository (create it once: gh repo create elijahmathews/opensps-data --public).
#
# Equivalent commands:
#   gh release create TAG --repo elijahmathews/opensps-data --title "Data release TAG" \
#       --notes-file RELEASE_DIR/DATA_README.md [--draft]
#   gh release upload TAG --repo elijahmathews/opensps-data RELEASE_DIR/<each file in UPLOAD.txt>
# Afterwards: tools/fetch_data.py /tmp/check --release TAG   (downloads and verifies everything)
set -euo pipefail
REPO="elijahmathews/opensps-data"
DIR="$(realpath "${1:?usage: $0 RELEASE_DIR TAG [--draft]}")"
TAG="${2:?usage: $0 RELEASE_DIR TAG [--draft]}"
DRAFT="${3:-}"

[[ -f "$DIR/UPLOAD.txt" && -f "$DIR/$TAG.json" ]] || { echo "$DIR is not the release directory of $TAG" >&2; exit 1; }
mapfile -t FILES < <(sed "s|^|$DIR/|" "$DIR/UPLOAD.txt")
for f in "${FILES[@]}"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }; done

echo "Uploading ${#FILES[@]} files to $REPO release $TAG:"
printf '  %s\n' "${FILES[@]##*/}"
read -r -p "Proceed? [y/N] " ok
[[ "$ok" == y || "$ok" == Y ]] || exit 1

gh release create "$TAG" --repo "$REPO" --title "Data release $TAG" --notes-file "$DIR/DATA_README.md" $DRAFT
gh release upload "$TAG" --repo "$REPO" "${FILES[@]}"
echo "Done. Check: tools/fetch_data.py /tmp/opensps-data-check --release $TAG --cache /tmp/opensps-data-check-cache"
