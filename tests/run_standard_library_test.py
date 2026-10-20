#!/usr/bin/env python3
"""Meson wrapper: derive the [alpha/Fe] variants, then run test_standard_library.

Usage: run_standard_library_test.py EXE TEST_ROOT WORK_DIR

Without NumPy (needed to build the variants) only the end-to-end part runs.
"""
import importlib.util
import os
import subprocess
import sys

exe, test_root, work = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))
args = [exe, test_root]
if importlib.util.find_spec("numpy") is not None:
    afe_root = os.path.join(os.path.abspath(work), "fsps-test-data-afe")
    subprocess.run([sys.executable, os.path.join(here, "..", "tools", "fsps_data", "make_afe_variants.py"),
                    test_root, afe_root], check=True)
    args.append(afe_root)
else:
    print("# NumPy not found: [alpha/Fe] selection tests skipped", flush=True)
sys.exit(subprocess.run(args).returncode)
