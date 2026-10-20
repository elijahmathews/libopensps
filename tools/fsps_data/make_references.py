#!/usr/bin/env python3
"""Build the regression-reference libraries and the pin file of the repository.

Usage:
    make_references.py OUT_ROOT [--sps-dir DIR] [--param-dir DIR]
                       [--sps-version N] [--param-version N] [--pin tests/data/references.json]

Copies, byte for byte, the references of tests/test_runner.f90 (``sps_ref_*.bin``, written by
tests/generate_test_data.sh; default --sps-dir tests/data) into ``references/sps`` and those of
tests/test_param_regression.f90 (``<case>.opsf``, written with --generate; default --param-dir
tests/data/param_ref) into ``references/param`` under OUT_ROOT/libraries, and writes the pin file
(name, version and SHA-256 of every file) that the regression suite checks the references
against. Give a library a new version whenever its files change (doc/DATA_FORMAT.md, Section 7);
then update OUT_ROOT's root index (fsps_data.write_root_index) and pack a release with
pack_data.py --previous.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsps_data import LibraryWriter  # noqa: E402
from provenance import provenance  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPS_FILES = ["sps_ref_MILES-1_MIST-1.bin", "sps_ref_MILES-0_MIST-0_BASEL-1_PADOVA-1.bin",
             "sps_ref_MILES-1_MIST-1_THEMIS-1_DL07-0.bin", "sps_ref_MILES-0_C3K-1_MIST-1.bin",
             "sps_ref_MIST-0_BPASS-1.bin"]


def build(out, name, title, files, version):
    lib = LibraryWriter(out, "references", name, title=title, provenance=provenance("references", name),
                        version=version)
    for f in files:
        lib.file(os.path.basename(f), f)
    lib.write()
    return {"version": version, "files": {d["file"]: d["sha256"] for d in lib.opaque.values()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_root")
    ap.add_argument("--sps-dir", default=os.path.join(REPO, "tests", "data"))
    ap.add_argument("--param-dir", default=os.path.join(REPO, "tests", "data", "param_ref"))
    ap.add_argument("--sps-version", type=int, default=1)
    ap.add_argument("--param-version", type=int, default=1)
    ap.add_argument("--pin", default=os.path.join(REPO, "tests", "data", "references.json"))
    a = ap.parse_args(argv)

    sps = [os.path.join(a.sps_dir, f) for f in SPS_FILES]
    missing = [f for f in sps if not os.path.exists(f)]
    if missing:
        sys.exit("missing references: " + ", ".join(missing))
    param = sorted(glob.glob(os.path.join(a.param_dir, "*.opsf")))
    if not param:
        sys.exit(f"no .opsf references in {a.param_dir}")
    pin = {"comment": "Regression references this tree expects (libraries references/sps and references/param "
                      "of the data release); written by tools/fsps_data/make_references.py",
           "references/sps": build(a.out_root, "sps", "Regression references of test_runner", sps, a.sps_version),
           "references/param": build(a.out_root, "param", "Regression references of test_param_regression",
                                     param, a.param_version)}
    with open(a.pin, "w") as f:
        json.dump(pin, f, indent=1)
        f.write("\n")
    print(f"references/sps v{a.sps_version} ({len(sps)} files), references/param v{a.param_version} "
          f"({len(param)} files) in {a.out_root}; pin written to {a.pin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
