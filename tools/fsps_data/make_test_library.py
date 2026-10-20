#!/usr/bin/env python3
"""Build the small test library (tests/data/fsps-test-data) from a converted standard root.

Usage: make_test_library.py STANDARD_ROOT OUT_ROOT

Coarse subsets of what a default setup (MIST + MILES + DL07, nebular emission without dust in
the ionized gas) reads: two metallicities, every 5th isochrone age and every 4th point, every
25th wavelength of the spectral library, and coarser raw grids for the auxiliary spectra,
stored as FP32 where the full library is FP64. The physics is that of the full libraries at
low resolution: good for checking that a setup and a CSP run end to end, not for science. The
regression and legacy-parity suites use the full data.
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsps_data import Library, LibraryWriter, write_root_index  # noqa: E402

TITLE_NOTE = "coarse test subset of the full library; not for science"

# library -> {axis: selection (slice or index list)}; arrays are sliced along their named axes
SPEC = {
    ("stellar_spectra", "miles"): {"Z": [1, 3], "wavelength": slice(None, None, 25)},
    ("nebular", "mist_nodust"): {"wavelength": slice(None, None, 10)},
    ("dust_emission", "DL07"): {"wavelength": slice(None, None, 8)},
    ("dust_emission", "THEMIS"): {"wavelength": slice(None, None, 8)},
    ("agn_torus", "clumpy"): {},
    ("attenuation", "dirty"): {},
    ("attenuation", "smc_bar"): {},
    ("reference_spectra", "vega"): {},
    ("reference_spectra", "sun"): {},
    ("hot_star_spectra", "wmbasic"): {"wavelength": slice(None, None, 25)},
    ("hot_star_spectra", "cmfgen_wn"): {"wavelength": slice(None, None, 10)},
    ("hot_star_spectra", "cmfgen_wc"): {"wavelength": slice(None, None, 10)},
    ("hot_star_spectra", "post_agb"): {"wavelength": slice(None, None, 40)},
    ("agb_spectra", "orich"): {"wavelength": slice(None, None, 25)},
    ("agb_spectra", "crich"): {"wavelength": slice(None, None, 25)},
    ("agb_spectra", "crich_aringer"): {"wavelength": slice(None, None, 40)},
    ("dusty_agb", "orich"): {"wavelength": slice(None, None, 10)},
    ("dusty_agb", "crich"): {"wavelength": slice(None, None, 10)},
    ("xrb", "xsp"): {"wavelength": slice(None, None, 50)},
}
ISOC_Z = [5, 9]            # MIST [Fe/H] = -1.0 and 0.0
ISOC_AGES = slice(0, None, 5)
ISOC_POINTS = 4
FILTER_POINTS = 40
SMALL_TABLES = ["allindices.dat", "lsf.dat", "imf.dat", "sfh.dat", "emlines_info.dat", "FILTER_LIST"]


def narrow(a):
    return a.astype("<f4") if a.dtype == np.float64 else a


def subset_generic(src, out, kind, name, sel):
    lib = Library.open(src, kind, name)
    m = lib.manifest
    prov = dict(m.get("provenance", {}), subset_of=f"{kind}/{name} v{m.get('version', 1)}",
                subset={k: str(v) for k, v in sel.items()})
    w = LibraryWriter(out, kind, name, title=(m.get("title", "") + " - " + TITLE_NOTE), provenance=prov)
    w.metadata.update(lib.metadata)
    for ax, desc in m["axes"].items():
        extra = {k: v for k, v in desc.items() if k not in ("units", "values", "labels", "file", "dtype", "shape",
                                                            "sha256", "description")}
        if "labels" in desc:
            w.axis(ax, desc["units"], labels=list(np.array(desc["labels"])[sel.get(ax, slice(None))]), **extra)
        else:
            v = lib.axis_values(ax)[sel.get(ax, slice(None))]
            if "file" in desc:
                w.axis(ax, desc["units"], data=v, **extra)
            else:
                w.axis(ax, desc["units"], values=v, **extra)
    for arr, desc in m["arrays"].items():
        a = lib.array(arr)
        for d, ax in enumerate(desc.get("axes", [])):
            if ax in sel:
                idx = [slice(None)] * a.ndim
                idx[d] = sel[ax]
                a = a[tuple(idx)]
        w.array(arr, narrow(np.ascontiguousarray(a)), axes=desc.get("axes"), units=desc.get("units", ""))
    w.write()


def subset_isochrones(src, out):
    lib = Library.open(src, "isochrones", "mist")
    npts = lib.array("npoints")
    logage = lib.array("log_age")
    cols = {c: lib.array(c) for c in ["mini", "mact", "logl", "logt", "logg", "composition", "phase", "log_mdot"]}
    starts = np.concatenate([[0], np.cumsum(npts.ravel())]).reshape(-1)
    nt = npts.shape[1]
    ages = list(range(nt))[ISOC_AGES]
    new_n = np.zeros((len(ISOC_Z), len(ages)), dtype=np.int32)
    new_cols = {c: [] for c in cols}
    for i, iz in enumerate(ISOC_Z):
        for j, it in enumerate(ages):
            s0 = starts[iz * nt + it]
            n = npts[iz, it]
            take = np.arange(s0, s0 + n)[::ISOC_POINTS]
            if n > 0 and take[-1] != s0 + n - 1:
                take = np.append(take, s0 + n - 1)   # keep the last point (end of the isochrone)
            new_n[i, j] = len(take)
            for c in cols:
                new_cols[c].append(cols[c][take])
    w = LibraryWriter(out, "isochrones", "mist", title="MIST isochrones - " + TITLE_NOTE,
                      provenance=dict(lib.manifest.get("provenance", {}), subset_of="isochrones/mist"))
    w.metadata.update(lib.metadata)
    w.axis("Z", "mass fraction", values=lib.axis_values("Z")[ISOC_Z])
    w.axis("age_index", "index", values=list(range(1, len(ages) + 1)))
    w.array("npoints", new_n, axes=["Z", "age_index"], dtype="<i4", units="count")
    w.array("log_age", logage[ISOC_Z][:, ages], axes=["Z", "age_index"], units="log10(yr)")
    for c in cols:
        w.array(c, np.concatenate(new_cols[c]), units=lib.manifest["arrays"][c]["units"])
    w.write()


def subset_filters(src, out):
    lib = Library.open(src, "filters", "allfilters")
    npts = lib.array("npoints")
    lam, tr = lib.array("wavelength"), lib.array("transmission")
    starts = np.concatenate([[0], np.cumsum(npts)])
    keep_l, keep_t, new_n = [], [], []
    for i in range(len(npts)):
        idx = np.arange(starts[i], starts[i + 1])
        if len(idx) > FILTER_POINTS:
            idx = idx[np.linspace(0, len(idx) - 1, FILTER_POINTS).round().astype(int)]
        keep_l.append(lam[idx])
        keep_t.append(tr[idx])
        new_n.append(len(idx))
    w = LibraryWriter(out, "filters", "allfilters", title="Default filter set - " + TITLE_NOTE,
                      provenance=dict(lib.manifest.get("provenance", {}), subset_of="filters/allfilters"))
    w.metadata.update(lib.metadata)
    w.axis("filter", "", labels=lib.axis_values("filter"))
    w.array("npoints", np.array(new_n, dtype=np.int32), axes=["filter"], dtype="<i4", units="count")
    w.array("wavelength", np.concatenate(keep_l), units="angstrom")
    w.array("transmission", np.concatenate(keep_t), units="relative throughput")
    w.write()


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, out = argv
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "data"))
    for (kind, name), sel in SPEC.items():
        subset_generic(src, out, kind, name, sel)
        print(f"  {kind}/{name}")
    subset_isochrones(src, out)
    subset_filters(src, out)
    for fn in SMALL_TABLES:
        shutil.copy2(os.path.join(src, "data", fn), os.path.join(out, "data", fn))
    write_root_index(out, "test", "Coarse test subsets of the default libraries (not for science)")
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f"test library: {total / 1e6:.1f} MB in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
