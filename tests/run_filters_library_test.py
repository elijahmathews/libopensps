#!/usr/bin/env python3
"""Meson wrapper of tests/test_c_filters.c: derives a data root next to the small test library with two
filter libraries written by tools/opensps_data/filters_tool.py (a photon-counting and an energy-counting
filter from VOTables, one calibrated) and a copy with a wrong content hash, and runs the C test with their
names and the hashes the Python reference implementation computed.

Usage: run_filters_library_test.py EXE TEST_ROOT WORK_DIR
Without NumPy (needed by the tool) the test is skipped (exit status 77).
"""
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

root = os.path.join(os.path.abspath(work), "opensps-test-data-filters")
if os.path.exists(root):
    shutil.rmtree(root)
shutil.copytree(test_root, root)

VOTABLE = """<?xml version="1.0"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE name="SVO Filter Profile Service">
<TABLE name="t"><PARAM name="filterID" value="Test/Test.%(n)s"/><PARAM name="DetectorType" value="%(d)d"/>
<FIELD name="Wavelength" unit="Angstrom"/><FIELD name="Transmission" unit=""/>
<DATA><TABLEDATA>%(rows)s</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>
"""
specs = [("P", 1, [4500.0, 4800.0, 5200.0, 5500.0], [0.2, 1.0, 1.0, 0.2]),                      # photon counter
         ("E", 0, [6000.0, 6500.0, 7000.0, 7500.0, 8000.0], [0.0, 0.5, 1.0, 0.5, 0.0])]         # energy counter
files = []
for n, d, lam, tr in specs:
    path = os.path.join(work, f"filters_test_{n}.xml")
    rows = "".join(f"<TR><TD>{x!r}</TD><TD>{y!r}</TD></TR>" for x, y in zip(lam, tr))
    with open(path, "w") as f:
        f.write(VOTABLE % dict(n=n, d=d, rows=rows))
    files.append(path)
curves = [filters_tool.curve_from_file(f, 0, "angstrom", {"Test/Test.E": (7000.0, -1.0)}) for f in files]
man = filters_tool.pack(root, "svo_test", curves, "test filters")
hashes = [e["hash"] for e in man["metadata"]["filters"]]
# The same library with a hash that does not match its curve
shutil.copytree(os.path.join(root, "libraries", "filters", "svo_test"),
                os.path.join(root, "libraries", "filters", "svo_test_bad"))
mpath = os.path.join(root, "libraries", "filters", "svo_test_bad", "manifest.json")
with open(mpath) as f:
    m = json.load(f)
m["name"] = "svo_test_bad"
m["metadata"]["filters"][0]["hash"] = "0" * 64
with open(mpath, "w") as f:
    json.dump(m, f)

env = dict(os.environ, OPENSPS_DATA_HOME=root, FSPS_DATA_HOME=root, SPS_HOME="",
           OPENSPS_FILTERS_LIB="svo_test", OPENSPS_FILTERS_LIB_BAD="svo_test_bad",
           OPENSPS_FILTERS_HASH0=hashes[0], OPENSPS_FILTERS_HASH1=hashes[1], OMP_NUM_THREADS="2")
sys.exit(subprocess.run([exe], env=env).returncode)
