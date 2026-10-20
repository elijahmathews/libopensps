#!/usr/bin/env python3
"""Regression test: the references found are the ones this tree expects (tests/data/references.json).

Usage: check_references.py PIN SPS_DIR PARAM_DIR

For each references library the pin lists, compares the SHA-256 of every file in the resolved
directory (from the data root, or the in-tree fallback) with the pinned value, and on a
mismatch says which version is expected and how to get it. Exits 77 (skipped) when no
references are found, as the regression tests that use them do.
"""
import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    pin_path, sps_dir, param_dir = sys.argv[1:4]
    pin = json.load(open(pin_path))
    found, bad = 0, []
    for name, d in (("references/sps", sps_dir), ("references/param", param_dir)):
        entry = pin[name]
        if not os.path.isdir(d):
            print(f"{name}: not found ({d})")
            continue
        found += 1
        for fname, want in sorted(entry["files"].items()):
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                bad.append(f"{name}: {fname} missing in {d}")
            elif sha256(p) != want:
                bad.append(f"{name}: {fname} in {d} differs from the pinned version v{entry['version']}")
        if not any(b.startswith(name) for b in bad):
            print(f"{name}: v{entry['version']}, {len(entry['files'])} files match the pin ({d})")
    if found == 0:
        print("SKIP: no regression references found (data release libraries references/sps and references/param, "
              "or tests/data); set OPENSPS_DATA_HOME or -Ddata_home to a data root that has them")
        return 77
    if bad:
        print("Regression references do not match tests/data/references.json:\n  " + "\n  ".join(bad))
        print("Use the data release that provides the pinned versions (tools/fetch_data.py), or, after an intended "
              "physics change, regenerate them and run tools/fsps_data/make_references.py (doc/DATA_FORMAT.md, "
              "Section 7).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
