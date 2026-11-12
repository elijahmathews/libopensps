#!/usr/bin/env python3
"""Meson wrapper: the [alpha/Fe] stack tests on the five-node variants derived from the small test library.

Usage: run_afe_derived_test.py EXE TEST_ROOT WORK_DIR [tap]

Derives the libraries `mist5` and `stellar_spectra/miles5` (tools/opensps_data/make_afe_variants.py: five
[alpha/Fe] nodes, each different) next to the test library, then runs EXE (tests/test_afe_library.f90 or
tests/test_c_afe.c) with OPENSPS_AFE_DATA at that root and OPENSPS_AFE_ISOC / OPENSPS_AFE_SPEC naming them:
the same checks as on the real libraries, with no external data. Without NumPy (needed to derive them) the
test is skipped: exit status 77, or with `tap` (EXE speaks TAP, where meson counts 77 as an error) a TAP
SKIP line and status 0.
"""
import importlib.util
import os
import subprocess
import sys

exe, test_root, work = sys.argv[1:4]
tap = len(sys.argv) > 4 and sys.argv[4] == "tap"
here = os.path.dirname(os.path.abspath(__file__))
if importlib.util.find_spec("numpy") is None:
    print("1..0 # SKIP NumPy not found (needed to derive the [alpha/Fe] variants)", flush=True)
    sys.exit(0 if tap else 77)
root = os.path.join(os.path.abspath(work), "opensps-test-data-afe5-" + os.path.basename(exe))
subprocess.run([sys.executable, os.path.join(here, "..", "tools", "opensps_data", "make_afe_variants.py"),
                test_root, root], check=True)
env = dict(os.environ, OPENSPS_AFE_DATA=root, OPENSPS_DATA_HOME=root, OPENSPS_AFE_ISOC="mist5", OPENSPS_AFE_SPEC="miles5")
sys.exit(subprocess.run([exe], env=env).returncode)
