#!/usr/bin/env python3
"""Derive [alpha/Fe] test variants of a standard root's MIST isochrones and MILES spectra.

Usage: make_afe_variants.py SOURCE_ROOT OUT_ROOT

OUT_ROOT links every library of SOURCE_ROOT and adds, for the [alpha/Fe] tests:
  isochrones/mist1, stellar_spectra/miles1  an afe axis with the single value 0 (same data):
                                            must load exactly like the axis-free libraries;
  isochrones/mist2, stellar_spectra/miles2  afe = [0.4, 0.0] (default 0), the 0.4 block first:
                                            0.0 is the original data, 0.4 has log T + 0.01 and
                                            the flux doubled, so a selection is visible.
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsps_data import Library, LibraryWriter, root_index_path  # noqa: E402

COLS = ["mini", "mact", "logl", "logt", "logg", "composition", "phase", "log_mdot"]


def copy_axes(w, lib):
    for ax, d in lib.manifest["axes"].items():
        if "labels" in d:
            w.axis(ax, d["units"], labels=d["labels"])
        elif "file" in d:
            w.axis(ax, d["units"], data=lib.axis_values(ax))
        else:
            w.axis(ax, d["units"], values=lib.axis_values(ax))


def isochrones(src, out, name, afe):
    lib = Library.open(src, "isochrones", "mist")
    w = LibraryWriter(out, "isochrones", name, title="afe test variant")
    w.metadata.update(lib.metadata)
    w.axis("afe", "[alpha/Fe]", values=afe, default=0.0)
    copy_axes(w, lib)
    n = len(afe)
    npts, logage = lib.array("npoints"), lib.array("log_age")
    w.array("npoints", np.stack([npts] * n), axes=["afe", "Z", "age_index"], dtype="<i4", units="count")
    w.array("log_age", np.stack([logage] * n), axes=["afe", "Z", "age_index"], units="log10(yr)")
    for c in COLS:
        a = lib.array(c)
        blocks = [a + 0.01 if (c == "logt" and v != 0.0) else a for v in afe]
        w.array(c, np.concatenate(blocks), units=lib.manifest["arrays"][c]["units"])
    w.write()


def spectra(src, out, name, afe):
    lib = Library.open(src, "stellar_spectra", "miles")
    w = LibraryWriter(out, "stellar_spectra", name, title="afe test variant")
    w.metadata.update(lib.metadata)
    w.axis("afe", "[alpha/Fe]", values=afe, default=0.0)
    copy_axes(w, lib)
    f = lib.array("flux")
    w.array("flux", np.stack([f * np.float32(2.0) if v != 0.0 else f for v in afe]),
            axes=["afe", "Z", "logg", "logT", "wavelength"], dtype="<f4", units="H_nu")
    w.array("resolution", lib.array("resolution"), axes=["wavelength"], units="as source")
    w.write()


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, out = (os.path.abspath(p) for p in argv)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    idx = root_index_path(src)
    shutil.copy2(idx, os.path.join(out, os.path.basename(idx)))
    os.symlink(os.path.join(src, "data"), os.path.join(out, "data"))
    for kind in os.listdir(os.path.join(src, "libraries")):
        os.makedirs(os.path.join(out, "libraries", kind))
        for name in os.listdir(os.path.join(src, "libraries", kind)):
            os.symlink(os.path.join(src, "libraries", kind, name), os.path.join(out, "libraries", kind, name))
    isochrones(src, out, "mist1", [0.0])
    isochrones(src, out, "mist2", [0.4, 0.0])
    spectra(src, out, "miles1", [0.0])
    spectra(src, out, "miles2", [0.4, 0.0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
