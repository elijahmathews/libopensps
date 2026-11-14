#!/usr/bin/env python3
"""Fail when a hardware tuning parameter is defined as a literal outside src/core/opensps_tuning.f90.

The names below are the tuning parameters that used to be literals in the kernels (doc/TUNING.md). A
definition such as `integer, parameter :: NCH = 64` anywhere else in src/ would put a device-dependent
number back in the source. Define it from the registry instead (`integer, parameter :: NCH = TUNE_NCH`).
Add a name here when you add a parameter to the registry that used to be, or could be mistaken for, a local
constant. Usage: check_tuning_literals.py [source root]
"""
import pathlib
import re
import sys

NAMES = ("SEG_LEN", "PT_PIECE", "PIX_BLOCK", "BAND_BLOCK", "GEMM_MIN_COLS", "NCH", "NCHG", "MBLK", "SMALL_N",
         "PT", "MT", "KB", "BATCH_SIZE", "NC_BATCHED", "CHG", "CH", "HOST_SFH_MAX", "CHUNK", "CHUNK_BYTES", "NTILE", "TILE_E", "TILE_M")
# NAME = <number or arithmetic of numbers> inside a declaration (or `, NAME = 1` in a list of them)
PAT = re.compile(r"(?:::|,)\s*(%s)\s*=\s*[0-9][0-9_*\s+\-/]*(?=,|$|!)" % "|".join(NAMES), re.I)


def main(root):
    bad = []
    for path in sorted(pathlib.Path(root, "src").rglob("*.f90")):
        if path.name == "opensps_tuning.f90":
            continue
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            code = line.split("!", 1)[0]
            if "parameter" in code.lower() and PAT.search(code):
                bad.append("%s:%d: %s" % (path, n, line.strip()))
    for b in bad:
        print(b)
    if bad:
        print("tuning parameters must come from src/core/opensps_tuning.f90 (doc/TUNING.md)")
        return 1
    print("OK: no tuning parameter is a literal outside the registry")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
