#!/usr/bin/env python3
"""Check that every file that repeats the project version agrees with meson.build.

meson.build (`project(..., version : 'X.Y.Z')`) is the single source of truth. The C header, the
Fortran version module, the tarball names and the release workflow all derive from it. The
files that cannot derive it (they are read by other tools as plain text) are checked here:

    CITATION.cff           version: "X.Y.Z"   (if present)
    packaging/opensps.spec Version: X.Y.Z
    debian/changelog       opensps (X.Y.Z-N) ...   (newest entry)

Usage: tools/check_versions.py [TAG]   TAG, e.g. v0.1.0, must equal the version as well.
Standard library only. Exit status 1 on any mismatch.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def first(pattern, path, what):
    text = (ROOT / path).read_text()
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        sys.exit(f"{path}: could not find {what}")
    return m.group(1)


def main(argv):
    truth = first(r"^\s*version\s*:\s*'([^']+)'", "meson.build", "the project version")
    found = {
        "CITATION.cff": first(r'^version:\s*"?([^"\s]+)"?\s*$', "CITATION.cff", "an uncommented version:"),
        "packaging/opensps.spec": first(r"^Version:\s*(\S+)", "packaging/opensps.spec", "Version:"),
        "debian/changelog": first(r"\A\S+ \(([^)]+?)-[^-)]+\)", "debian/changelog", "the newest entry"),
    }
    if len(argv) > 1:
        found[f"tag {argv[1]}"] = argv[1].removeprefix("v")
    bad = {k: v for k, v in found.items() if v != truth}
    for k, v in found.items():
        print(f"{'OK  ' if k not in bad else 'FAIL'} {k}: {v}")
    print(f"meson.build: {truth}")
    if bad:
        sys.exit("version mismatch: " + ", ".join(bad))


main(sys.argv)
