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


def check_registry(root):
    """Every compile-time constant of the registry has its override pair and its row in the table."""
    text = pathlib.Path(root, "src/core/opensps_tuning.f90").read_text()
    declared = set(re.findall(r"integer, parameter, public :: TUNE_(?!P_)([A-Z0-9_]+) =", text))
    ifdefs = set(re.findall(r"^#ifdef TUNE_([A-Z0-9_]+)_OVERRIDE", text, re.M))
    ct_names = re.search(r"ct_names\(N_CT\) = \[character\(len=24\) :: &(.*?)\]", text, re.S)
    table = set(re.findall(r"'([A-Z0-9_]+)'", ct_names.group(1))) if ct_names else set()
    values = set(re.findall(r"TUNE_([A-Z0-9_]+)", re.search(r"ct_values\(N_CT\) = \[(.*?)\]", text, re.S).group(1)))
    over = set(re.findall(r"CT_OVR_([A-Z0-9_]+)", re.search(r"ct_over\(N_CT\) = \[(.*?)\]", text, re.S).group(1)))
    problems = []
    for what, got in (("an #ifdef TUNE_<NAME>_OVERRIDE pair", ifdefs), ("a row in ct_names", table),
                      ("a row in ct_values", values), ("a row in ct_over", over)):
        for name in sorted(declared - got):
            problems.append("TUNE_%s lacks %s" % (name, what))
        for name in sorted(got - declared):
            problems.append("%s is listed for %s but is not a compile-time constant" % (name, what))
    return problems


def main(root):
    bad = check_registry(root)
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
        print("tuning parameters must come from src/core/opensps_tuning.f90 (doc/TUNING.md); a compile-time constant "
              "needs its override pair and table rows there")
        return 1
    print("OK: no tuning parameter is a literal outside the registry; every compile-time constant is overridable")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
