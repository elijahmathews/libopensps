#!/usr/bin/env python3
"""Meson wrapper of tests/test_c_filters_malformed.c: writes a good filter library with
tools/opensps_data/filters_tool.py, then copies of it whose manifest metadata is malformed in one way each, and
checks that the good one loads and that each bad one stops the reader with its own message on stderr.

Usage: run_filters_malformed_test.py EXE TEST_ROOT WORK_DIR
Without NumPy (needed by the tool) the test is skipped (exit status 77).
"""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys

exe, test_root, work = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))
if importlib.util.find_spec("numpy") is None:
    print("SKIP: NumPy not found (needed by tools/opensps_data/filters_tool.py)")
    sys.exit(77)
sys.path.insert(0, os.path.join(here, "..", "tools", "opensps_data"))
import filters_tool  # noqa: E402

root = os.path.join(os.path.abspath(work), "opensps-test-data-filters-malformed")
if os.path.exists(root):
    shutil.rmtree(root)
shutil.copytree(test_root, root)
os.makedirs(work, exist_ok=True)

VOTABLE = """<?xml version="1.0"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE name="SVO Filter Profile Service">
<TABLE name="t"><PARAM name="filterID" value="Test/Test.%(n)s"/><PARAM name="DetectorType" value="%(d)d"/>
<FIELD name="Wavelength" unit="Angstrom"/><FIELD name="Transmission" unit=""/>
<DATA><TABLEDATA>%(rows)s</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>
"""
specs = [("P", 1, [4500.0, 4800.0, 5200.0, 5500.0], [0.2, 1.0, 1.0, 0.2]),
         ("E", 0, [6000.0, 6500.0, 7000.0, 7500.0, 8000.0], [0.0, 0.5, 1.0, 0.5, 0.0])]
files = []
for n, d, lam, tr in specs:
    path = os.path.join(work, f"filters_malformed_{n}.xml")
    rows = "".join(f"<TR><TD>{x!r}</TD><TD>{y!r}</TD></TR>" for x, y in zip(lam, tr))
    with open(path, "w") as f:
        f.write(VOTABLE % dict(n=n, d=d, rows=rows))
    files.append(path)
curves = [filters_tool.curve_from_file(f, 0, "angstrom", {"Test/Test.E": (7000.0, -1.0)}) for f in files]
filters_tool.pack(root, "good", curves, "test filters")
libdir = os.path.join(root, "libraries", "filters")


def variant(name, edit):
    shutil.copytree(os.path.join(libdir, "good"), os.path.join(libdir, name))
    mpath = os.path.join(libdir, name, "manifest.json")
    with open(mpath) as f:
        m = json.load(f)
    m["name"] = name
    edit(m["metadata"])
    with open(mpath, "w") as f:
        json.dump(m, f)


def set_field(i, key, value):
    def edit(meta):
        meta["filters"][i][key] = value
    return edit


def drop_entry(meta):
    meta["filters"] = meta["filters"][:1]


def bad_exponent(meta):
    meta["filters"][1]["calibration"] = {"lambda_ref": 7000.0, "exponent": "steep"}


cases = [
    # (library, message expected on stderr)
    ("bad_detector", set_field(0, "detector", "bolometer"), 'detector must be "photon" or "energy"'),
    ("bad_lambda_ref", set_field(1, "calibration", {"lambda_ref": -5.0, "exponent": -1.0}), "bad calibration of filter 2"),
    ("bad_exponent", bad_exponent, "bad calibration of filter 2"),
    ("short_list", drop_entry, 'metadata "filters" must have one entry per filter'),
]
for name, edit, _ in cases:
    variant(name, edit)

env = dict(os.environ, OPENSPS_DATA_HOME=root, FSPS_DATA_HOME=root, SPS_HOME="", OMP_NUM_THREADS="2")
failures = 0

r = subprocess.run([exe, "good"], env=env, capture_output=True, text=True)
if r.returncode != 0 or "status 0" not in r.stdout:
    print(f"FAIL: the good library: exit {r.returncode}, {r.stdout!r}, {r.stderr!r}")
    failures += 1
for name, _, message in cases:
    r = subprocess.run([exe, name], env=env, capture_output=True, text=True)
    if r.returncode == 0 or message not in r.stderr or f"filters/{name}" not in r.stderr:
        print(f"FAIL: {name}: exit {r.returncode}, expected {message!r} in stderr, got {r.stderr!r}")
        failures += 1
    if "status" in r.stdout:
        print(f"FAIL: {name}: the reader returned a status instead of stopping: {r.stdout!r}")
        failures += 1
print("all malformed-library checks passed" if failures == 0 else f"{failures} failure(s)")
sys.exit(1 if failures else 0)
