#!/usr/bin/env python3
"""Convert the legacy FSPS data layout (``$SPS_HOME/data``) into the standard format.

Usage:
    convert_legacy.py LEGACY_ROOT OUT_ROOT [--only KIND[/NAME] ...] [--release R] [--list]

LEGACY_ROOT contains ``data/``; OUT_ROOT receives ``opensps-data.json``, ``libraries/`` and a
copy of the small text tables in ``data/``. Every reader below mirrors the Fortran reader named
in its docstring value for value (list-directed text is parsed with the same correctly rounded
conversion), so the Fortran loader reproduces the legacy arrays bit for bit; the binary
libraries are copied byte for byte. See doc/DATA_FORMAT.md.
"""

import argparse
import decimal
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsps_data import LibraryWriter, ListReader, fortran_float, split_tokens, write_root_index  # noqa: E402
from provenance import provenance  # noqa: E402

# Sizes and grids that the legacy code holds as constants (src/core/fsps_constants.f90,
# src/io/fsps_io.f90, src/core/fsps_initialization.f90)
NDIM_LOGT, NDIM_LOGG = 68, 19
NM = 2000
ISOCHRONES = {
    # name: (directory under data/isochrones, nt, nz, solar metallicity, title)
    "mist": ("MIST", 107, 12, 0.0142, "MIST isochrones"),
    "pdva": ("Padova/Padova2007", 94, 22, 0.019, "Padova 2007 isochrones"),
    "prsc": ("PARSEC", 93, 15, 0.01524, "PARSEC isochrones"),
    "bsti": ("BaSTI", 94, 10, 0.020, "BaSTI isochrones"),
    "gnva": ("Geneva", 51, 5, 0.020, "Geneva isochrones"),
}
SPECTRA = {
    # name: (directory, file stem for the binaries, lambda/res stem, nspec, nzinit, solar Z, title)
    "miles": ("MILES", "imiles", "miles", 5994, 5, 0.019, "MILES empirical library merged with BaSeL"),
    "basel": ("BaSeL3.1", "basel_wlbc", "basel", 1963, 6, 0.020, "BaSeL 3.1 (WLBC) theoretical library"),
    "c3k_afe+0.0": ("C3K", "c3k_afe+0.0", "c3k_afe+0.0", 11149, 11, 0.0134,
                    "C3K theoretical library, [alpha/Fe] = +0.0"),
}
NEB = dict(nlam=1963, nz=11, nage=10, nu=7, nline=166)
DUSTEM = {
    # name: (nlam, numin, nqpah, qpah grid, umin grid) as in fsps_io.f90
    "DL07": (1001, 22, 7, [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58],
             [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0,
              2.5, 3.0, 4.0, 5.0, 7.0, 8.0, 12.0, 15.0, 20.0, 25.0]),
    "THEMIS": (576, 37, 11, [(100.0 / 2.2) * q for q in
                             [0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40]],
               [0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
                0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
                7.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 80.0]),
}
# Filters renormalized after the photon-counting normalization (apply_legacy_filter_norm):
# 1-based filter index -> (reference wavelength in A, exponent of (lambda/lambda_ref))
FILTER_CALIB = {}
for _i, _lam in zip([53, 54, 55, 56, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                    [3.550, 4.493, 5.731, 7.872, 70.0, 100.0, 160.0, 250.0, 350.0, 500.0,
                     12.0, 25.0, 60.0, 100.0]):
    FILTER_CALIB[_i] = (1.0e4 * _lam, -1.0)
for _i, _lam in zip([90, 91, 92], [23.68, 71.42, 155.9]):
    FILTER_CALIB[_i] = (1.0e4 * _lam, -2.0)
NBANDS = 159
N_SED = 1221
XRB_AGES_MYR = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 12.6, 16.0, 20.0]
XRB_FEH = [-1.3, -1.0, -0.8, -0.7, -0.5, -0.4, -0.3, -0.2, 0.0, 0.2, 0.3]
XRB_FEH_STR = ["-1.30", "-1.00", "-0.80", "-0.70", "-0.50", "-0.40", "-0.30", "-0.20", "+0.00", "+0.20", "+0.30"]
SMALL_TABLES = ["allindices.dat", "lsf.dat", "imf.dat", "sfh.dat", "emlines_info.dat", "FILTER_LIST",
                "magsun.dat", "filter_lambda_eff.dat", "ev_phases.tex", "README.md"]


def f64(x):
    return "%6.4f" % x


def one_per_record(path, n):
    """``do i = 1, n: read(u, *) x(i)``: the first value of each of n records."""
    r = ListReader(path)
    return np.array([r.read(1)[0] for _ in range(n)])


class Converter:
    def __init__(self, legacy_root, out_root):
        self.data = os.path.join(legacy_root, "data")
        self.out = out_root
        self.done = []

    def p(self, *parts):
        return os.path.join(self.data, *parts)

    def writer(self, kind, name, title, source, notes=""):
        prov = {"converted_from": "legacy FSPS layout: data/" + source}
        prov.update(provenance(kind, name))
        lib = LibraryWriter(self.out, kind, name, title=title, notes=notes, provenance=prov)
        return lib

    def finish(self, lib):
        lib.write()
        self.done.append(f"{lib.kind}/{lib.name}")
        print(f"  {lib.kind}/{lib.name}", flush=True)

    # ------------------------------------------------------------------ stellar spectra
    def stellar_spectra(self, name):
        """read_spectral_binary, load_wavelength_grid, load_spectral_resolution, load_zlegend_file."""
        d, stem, lstem, nspec, nzinit, zsol, title = SPECTRA[name]
        zleg = one_per_record(self.p("spectra", d, "zlegend.dat"), nzinit)
        lib = self.writer("stellar_spectra", name, title, f"spectra/{d}")
        lib.metadata["solar_metallicity"] = zsol
        lib.axis("Z", "mass fraction", values=zleg)
        lib.axis("logT", "log10(K)", values=one_per_record(self.p("spectra", "BaSeL3.1", "basel_logt.dat"), NDIM_LOGT))
        lib.axis("logg", "log10(cm s^-2)", values=one_per_record(self.p("spectra", "BaSeL3.1", "basel_logg.dat"), NDIM_LOGG))
        lib.axis("wavelength", "angstrom", data=one_per_record(self.p("spectra", d, lstem + ".lambda"), nspec),
                 medium="vacuum")
        flux = np.empty((nzinit, NDIM_LOGG, NDIM_LOGT, nspec), dtype="<f4")
        for iz, z in enumerate(zleg):
            path = self.p("spectra", d, f"{stem}_z{f64(z)}.spectra.bin")
            raw = np.fromfile(path, dtype="<f4")  # direct-access record, no markers, x86 byte order
            if raw.size != nspec * NDIM_LOGT * NDIM_LOGG:
                raise ValueError(f"{path}: {raw.size} values")
            flux[iz] = raw.reshape(NDIM_LOGG, NDIM_LOGT, nspec)
        lib.array("flux", flux, axes=["Z", "logg", "logT", "wavelength"], dtype="<f4",
                  units="erg s^-1 cm^-2 Hz^-1 (Eddington flux H_nu; L_nu = 16 pi^2 R^2 H_nu)")
        lib.array("resolution", one_per_record(self.p("spectra", d, lstem + ".res"), nspec), axes=["wavelength"],
                  units="angstrom (FWHM) or km/s, as the legacy .res file")
        self.finish(lib)
        # Byte-for-byte check against the legacy binaries
        back = np.fromfile(os.path.join(lib.dir, "flux.bin"), dtype="<u1")
        legacy = b"".join(open(self.p("spectra", d, f"{stem}_z{f64(z)}.spectra.bin"), "rb").read() for z in zleg)
        if back.tobytes() != legacy:
            raise RuntimeError(f"{name}: flux.bin differs from the legacy binaries")

    # ------------------------------------------------------------------ isochrones
    @staticmethod
    def parse_isochrone_file(path, nt, is_mist):
        """read_isochrone_database: tracks split by '#' or non-numeric lines."""
        ncol = 9 if is_mist else 8
        tracks = []      # list of (log age, rows)
        current = None
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s[0] == "#":
                    current = None
                    continue
                toks = split_tokens(s)
                try:
                    if len(toks) < ncol:
                        raise ValueError
                    vals = [fortran_float(t) for t in toks[:ncol]]
                except ValueError:
                    current = None
                    continue
                if current is None:
                    current = []
                    tracks.append((vals[0], current))
                    if len(tracks) > nt:
                        raise ValueError(f"{path}: more than {nt} isochrones")
                current.append(vals[1:])
                if len(current) > NM:
                    raise ValueError(f"{path}: more than {NM} points in an isochrone")
        if len(tracks) != nt:
            print(f"  warning: {path}: {len(tracks)} isochrones, expected {nt} (as the legacy reader warns)")
        return tracks

    def isochrones(self, name):
        d, nt, nz, zsol, title = ISOCHRONES[name]
        is_mist = name == "mist"
        lib = self.writer("isochrones", name, title, f"isochrones/{d}")
        lib.metadata["solar_metallicity"] = zsol
        r = ListReader(self.p("isochrones", d, "zlegend.dat"))
        zleg, fnames = [], []
        for _ in range(nz):
            if is_mist:
                s = r.lines[r.pos][:5]
                r.pos += 1
                v = fortran_float(s[1:5])
                # load_zlegend_file: 10**(-v) * zsol for 'm', 10**v * zsol otherwise
                zleg.append((10.0 ** (-v) if s[0] == "m" else 10.0 ** v) * zsol)
                fnames.append(f"isoc_z{s}.dat")
            else:
                z = r.read(1)[0]
                zleg.append(z)
                fnames.append(f"isoc_z{f64(z)}.dat")
        lib.axis("Z", "mass fraction", values=zleg)
        lib.axis("age_index", "index", values=list(range(1, nt + 1)),
                 description="isochrone number; the age of each is in log_age")
        npts = np.zeros((nz, nt), dtype=np.int32)
        logage = np.zeros((nz, nt))
        cols = []
        for iz, fn in enumerate(fnames):
            tracks = self.parse_isochrone_file(self.p("isochrones", d, fn), nt, is_mist)
            for it, (age, rows) in enumerate(tracks):
                npts[iz, it] = len(rows)
                logage[iz, it] = age
                cols.extend(rows)
        cols = np.array(cols, dtype=np.float64)
        lib.array("npoints", npts, axes=["Z", "age_index"], dtype="<i4", units="count")
        lib.array("log_age", logage, axes=["Z", "age_index"], units="log10(yr)")
        names = [("mini", "M_sun (initial mass)"), ("mact", "M_sun (current mass)"), ("logl", "log10(L/L_sun)"),
                 ("logt", "log10(Teff/K)"), ("logg", "log10(g / cm s^-2)"),
                 ("composition", "C/O ratio or surface abundance flag, as the legacy column"),
                 ("phase", "evolutionary phase code")]
        if is_mist:
            names.append(("log_mdot", "log10(Mdot / M_sun yr^-1)"))
        for k, (cname, units) in enumerate(names):
            lib.array(cname, cols[:, k], units=units,
                      description="one value per point; points ordered by Z, then isochrone, then mass")
        self.finish(lib)

    def ssp_bpass(self):
        """read_bpass_data."""
        nspec, nt, nz = 15000, 43, 12
        d = "isochrones/BPASS"
        lib = self.writer("ssp", "bpss", "BPASS v2.2 simple stellar populations (Salpeter, 100 M_sun)", d)
        lib.metadata["solar_metallicity"] = 0.020
        lib.axis("Z", "mass fraction", values=one_per_record(self.p(d, "zlegend.dat"), nz))
        r = ListReader(self.p(d, "bpass.mass"))
        mass = np.zeros((nz, nt))
        ages = np.zeros(nt)
        for i in range(nt):
            v = r.read(1 + nz)
            ages[i] = v[0]
            mass[:, i] = v[1:]
        lib.axis("log_age", "log10(yr)", values=ages)
        lib.axis("wavelength", "angstrom", data=one_per_record(self.p(d, "bpass.lambda"), nspec), medium="vacuum")
        spec = np.fromfile(self.p(d, "bpass_v2.2_salpeter100.ssp.bin"), dtype="<f8")
        if spec.size != nspec * nt * nz:
            raise ValueError("BPASS binary size")
        lib.array("spectra", spec.reshape(nz, nt, nspec), axes=["Z", "log_age", "wavelength"],
                  units="L_sun Hz^-1 per M_sun formed")
        lib.array("stellar_mass", mass, axes=["Z", "log_age"], units="M_sun per M_sun formed")
        self.finish(lib)
        back = open(os.path.join(lib.dir, "spectra.bin"), "rb").read()
        if back != open(self.p(d, "bpass_v2.2_salpeter100.ssp.bin"), "rb").read():
            raise RuntimeError("bpss: spectra.bin differs from the legacy binary")

    # ------------------------------------------------------------------ nebular
    def nebular(self, isoc, dust, xray=False):
        """load_nebular_grid (and the X-ray-irradiated variant)."""
        tag = ("WX_" if xray else "") + ("WD" if dust else "ND")
        stem = f"ZAU_{tag}_{isoc}"
        name = isoc + ("_xray" if xray else "") + ("_dust" if dust else "_nodust")
        n = NEB
        lib = self.writer("nebular", name,
                          f"Nebular continuum and lines for the {isoc} isochrones, "
                          f"{'with' if dust else 'without'} dust in the ionized gas"
                          + (", X-ray binaries included in the ionizing spectrum" if xray else ""),
                          f"nebular/{stem}.{{cont,lines}}")
        rc = ListReader(self.p("nebular", stem + ".cont"))
        rc.skip()
        lam = rc.read(n["nlam"])
        cont = np.zeros((n["nz"], n["nage"], n["nu"], n["nlam"]))
        labels = np.zeros((n["nz"], n["nage"], n["nu"], 3))
        for i in range(n["nz"]):
            for j in range(n["nage"]):
                for k in range(n["nu"]):
                    labels[i, j, k] = rc.read(3)
                    cont[i, j, k] = rc.read(n["nlam"])
        rl = ListReader(self.p("nebular", stem + ".lines"))
        rl.skip()
        linepos = rl.read(n["nline"])
        lines = np.zeros((n["nz"], n["nage"], n["nu"], n["nline"]))
        for i in range(n["nz"]):
            for j in range(n["nage"]):
                for k in range(n["nu"]):
                    lab = rl.read(3)
                    if not np.array_equal(lab, labels[i, j, k]):
                        raise ValueError(f"{stem}: line labels differ from the continuum labels at {i, j, k}")
                    lines[i, j, k] = rl.read(n["nline"])
        # The axis values are the labels; the legacy reader keeps the last one read for each index
        logz = labels[:, -1, -1, 0]
        age = labels[-1, :, -1, 1]
        logu = labels[-1, -1, :, 2]
        for ax, vals in ((0, logz), (1, age), (2, logu)):
            ref = vals.reshape([-1 if a == ax else 1 for a in range(3)])
            if not np.all(np.broadcast_to(ref, labels.shape[:3]) == labels[..., ax]):
                raise ValueError(f"{stem}: grid labels are not a regular grid")
        lib.axis("logZ", "log10(Z/Z_sun)", values=logz)
        lib.axis("age", "yr", values=age)
        lib.axis("logU", "log10(ionization parameter)", values=logu)
        lib.axis("wavelength", "angstrom", data=lam, medium="vacuum")
        lib.axis("line_wavelength", "angstrom", data=linepos, medium="vacuum",
                 description="line identifications in data/emlines_info.dat")
        lib.array("continuum", cont, axes=["logZ", "age", "logU", "wavelength"],
                  units="L_sun Hz^-1 per unit ionizing photon rate (s^-1)")
        lib.array("lines", lines, axes=["logZ", "age", "logU", "line_wavelength"],
                  units="L_sun per unit ionizing photon rate (s^-1)")
        self.finish(lib)

    # ------------------------------------------------------------------ dust
    def dust_emission(self, name):
        """load_dust_emission_table."""
        nlam, numin, nqpah, qpah, umin = DUSTEM[name]
        lib = self.writer("dust_emission", name, f"{name} dust emission templates (Milky Way grain mix, R_V = 3.1)",
                          f"dust/dustem/{name}_MW3.1_*.dat")
        lib.axis("qpah", "percent (PAH mass fraction)", values=qpah)
        lib.axis("umin", "U_min (MW interstellar radiation field units)", values=umin)
        lib.axis("component", "", labels=["umin", "umin_to_umax"],
                 description="delta-function radiation field at U_min, and the power-law U_min..U_max mixture")
        spec = np.zeros((nqpah, numin, 2, nlam))
        lam = None
        for k in range(nqpah):
            fn = f"{name}_MW3.1_100.dat" if k == 10 else f"{name}_MW3.1_{k}0.dat"
            r = ListReader(self.p("dust", "dustem", fn))
            r.skip(2)
            raw = np.array([r.read(1 + 2 * numin) for _ in range(nlam)])
            lam_k = raw[:, 0] * 1.0e4  # microns -> A, the same multiplication as the legacy reader
            if lam is None:
                lam = lam_k
            elif not np.array_equal(lam, lam_k):
                raise ValueError(f"{fn}: wavelength grid differs between q_PAH files")
            spec[k] = raw[:, 1:].T.reshape(numin, 2, nlam)
        lib.axis("wavelength", "angstrom", data=lam)
        lib.array("spectra", spec, axes=["qpah", "umin", "component", "wavelength"],
                  units="Jy cm^2 sr^-1 per H nucleon (L_nu)")
        self.finish(lib)

    def agn_torus(self):
        """load_agn_dust_models."""
        fn = "Nenkova08_y010_torusg_n10_q2.0.dat"
        r = ListReader(self.p("dust", fn))
        r.skip(3)
        tau = r.read(9)
        raw = np.array([r.read(10) for _ in range(125)])
        lib = self.writer("agn_torus", "clumpy", "Clumpy AGN torus models (Y = 10, sigma = 10, N0 = 10, q = 2)",
                          "dust/" + fn)
        lib.axis("tau", "optical depth of a cloud (V band)", values=tau)
        lib.axis("wavelength", "angstrom", values=raw[:, 0])
        lib.array("spectra", raw[:, 1:].T, axes=["tau", "wavelength"], units="f_nu, normalized to unity")
        self.finish(lib)

    def attenuation(self):
        """load_attenuation_curves."""
        nlam, nblk = 25, 18
        curves = np.zeros((2, nblk, 6, nlam))
        tauv = np.zeros((2, nblk))
        lam = None
        for s, fn in enumerate(["alldirty_h.dat", "alldirty_c.dat"]):
            r = ListReader(self.p("dust", fn))
            r.skip(3)
            for i in range(nblk):
                rows = np.array([r.read(8) for _ in range(nlam)])
                if lam is None:
                    lam = rows[:, 0].copy()
                elif not np.array_equal(lam, rows[:, 0]):
                    raise ValueError(f"{fn}: wavelength grid differs between blocks")
                if not np.all(rows[:, 1] == rows[0, 1]):
                    raise ValueError(f"{fn}: tau_V varies within a block")
                tauv[s, i] = rows[0, 1]
                curves[s, i] = rows[:, 2:].T
        if not np.array_equal(tauv[0], tauv[1]):
            raise ValueError("tau_V grids differ between the homogeneous and clumpy files")
        lib = self.writer("attenuation", "dirty", "Radiative-transfer attenuation curves (dusty galactic environments)",
                          "dust/alldirty_{h,c}.dat")
        lib.axis("structure", "", labels=["homogeneous", "clumpy"])
        lib.axis("tau_v", "V-band optical depth", values=tauv[0])
        lib.axis("model", "", labels=["MW_dusty", "MW_shell", "MW_cloudy", "SMC_dusty", "SMC_shell", "SMC_cloudy"],
                 description="grain mix and geometry, the columns of the legacy files")
        lib.axis("wavelength", "angstrom", values=lam)
        lib.array("attenuation", curves, axes=["structure", "tau_v", "model", "wavelength"],
                  units="attenuation optical depth")
        self.finish(lib)

        fn = "Gordon03_table4.dat"
        r = ListReader(self.p("dust", fn))
        rows = np.array([r.read(3) for _ in range(30)])[::-1]  # the legacy reader stores them reversed
        lib = self.writer("attenuation", "smc_bar", "SMC bar extinction curve", "dust/" + fn)
        lib.axis("wavelength", "angstrom", values=rows[:, 0] * 1.0e4)
        lib.array("extinction", rows[:, 2], axes=["wavelength"], units="A(lambda)/A(V)")
        self.finish(lib)

    # ------------------------------------------------------------------ filters, reference spectra
    def filters(self):
        """load_filter_definitions (raw points; duplicates and negative values are handled on load)."""
        path = self.p("allfilters.dat")
        with open(path) as f:
            lines = f.read().splitlines()
        pos = 1  # header
        lam, trans, npts, descr = [], [], [], []
        for i_filt in range(NBANDS):
            n = 0
            d = ""
            while True:
                if pos >= len(lines):
                    if n > 0 and i_filt == NBANDS - 1:
                        break
                    raise ValueError(f"{path}: end of file in filter {i_filt + 1}")
                s = lines[pos].strip()
                if not s:
                    pos += 1
                    continue
                if s[0] == "#":
                    if n > 0:
                        break
                    d = s[1:].strip()
                    pos += 1
                    continue
                toks = split_tokens(s)
                lam.append(fortran_float(toks[0]))
                trans.append(fortran_float(toks[1]))
                n += 1
                pos += 1
            npts.append(n)
            descr.append(d)
        names = {}
        with open(self.p("FILTER_LIST")) as f:
            for line in f:
                t = line.split(None, 2)
                if len(t) >= 2 and t[0].isdigit():
                    names[int(t[0])] = (t[1], t[2].strip() if len(t) > 2 else "")
        lib = self.writer("filters", "allfilters", "Default filter set", "allfilters.dat, FILTER_LIST")
        lib.axis("filter", "", labels=[names.get(i + 1, (f"filter{i + 1}", ""))[0] for i in range(NBANDS)])
        lib.array("npoints", np.array(npts, dtype=np.int32), axes=["filter"], dtype="<i4", units="count")
        lib.array("wavelength", np.array(lam), units="angstrom",
                  description="concatenated filter curves, npoints[i] points for filter i")
        lib.array("transmission", np.array(trans), units="relative throughput (photon counting)")
        info = []
        for i in range(NBANDS):
            e = {"name": names.get(i + 1, ("", ""))[0], "description": names.get(i + 1, ("", descr[i]))[1] or descr[i]}
            if i + 1 in FILTER_CALIB:
                lam0, expo = FILTER_CALIB[i + 1]
                e["calibration"] = {"lambda_ref": lam0, "exponent": expo}
            info.append(e)
        lib.metadata["filters"] = info
        lib.metadata["calibration_note"] = (
            "Each curve is normalized to integral T/lambda dlambda = 1 on the model grid; a filter with "
            "'calibration' is then divided by integral (lambda/lambda_ref)^exponent T/lambda dlambda, "
            "the convention of IR instruments calibrated on a nu f_nu = const (exponent -1) or "
            "beta = 2 blackbody (exponent -2) source.")
        self.finish(lib)

    def reference_spectra(self):
        """load_standard_sed."""
        for name, fn, skip, fd, title in (
                ("vega", "A0V_KURUCZ_92.SED", 1, "f_lambda", "Vega (A0 V model atmosphere)"),
                ("sun", "SUN_STScI.SED", 0, "f_nu", "The Sun")):
            r = ListReader(self.p("spectra", fn))
            r.skip(skip)
            rows = np.array([r.read(2) for _ in range(N_SED)])
            lib = self.writer("reference_spectra", name, title, "spectra/" + fn)
            lib.metadata["flux_density"] = fd
            lib.axis("wavelength", "angstrom", data=rows[:, 0])
            lib.array("flux", rows[:, 1], axes=["wavelength"],
                      units="erg s^-1 cm^-2 A^-1 (arbitrary scale)" if fd == "f_lambda"
                      else "erg s^-1 cm^-2 Hz^-1 at 10 pc")
            self.finish(lib)

    # ------------------------------------------------------------------ hot stars, AGB
    def hot_stars(self):
        d = "spectra/Hot_spectra"
        # WMBasic (load_wmbasic_spectra)
        teff = one_per_record(self.p(d, "WMBASIC.teff"), 11)
        zleg = one_per_record(self.p(d, "WMBASIC_zlegend.dat"), 12)
        spec = np.zeros((12, 3, 11, 5508))
        lam = None
        for iz, z in enumerate(zleg):
            r = ListReader(self.p(d, f"WMBASIC_z{f64(z)}.spec"))
            raw = np.array([r.read(34) for _ in range(5508)])
            if lam is None:
                lam = raw[:, 0].copy()
            elif not np.array_equal(lam, raw[:, 0]):
                raise ValueError("WMBasic wavelength grids differ")
            spec[iz] = raw[:, 1:].T.reshape(3, 11, 5508)
        lib = self.writer("hot_star_spectra", "wmbasic", "WMBasic O-star spectra", d + "/WMBASIC_*")
        lib.axis("Z", "mass fraction", values=zleg)
        lib.axis("logg", "log10(cm s^-2)", values=[3.5, 4.0, 4.5])
        lib.axis("logT", "log10(K)", values=teff)
        lib.axis("wavelength", "angstrom", data=lam)
        lib.array("flux", spec, axes=["Z", "logg", "logT", "wavelength"], units="L_nu / L_bol (Hz^-1)")
        self.finish(lib)
        # CMFGEN Wolf-Rayet (load_wr_spectra)
        for kind in ("WN", "WC"):
            r = ListReader(self.p(d, f"CMFGEN_{kind}.teff"))
            logt = r.read(12)
            r = ListReader(self.p(d, f"CMFGEN_{kind}_Zall.spec"))
            lamwr = r.read(1963)
            spec = np.zeros((5, 12, 1963))
            zmet = np.zeros((5, 12))
            for j in range(5):
                for i in range(12):
                    lab = r.read(2)
                    zmet[j, i] = lab[1]
                    spec[j, i] = r.read(1963)
            if not np.all(zmet == zmet[:, -1:]):
                raise ValueError(f"CMFGEN_{kind}: Z label varies within a block")
            lib = self.writer("hot_star_spectra", f"cmfgen_{kind.lower()}", f"CMFGEN Wolf-Rayet spectra ({kind})",
                              f"{d}/CMFGEN_{kind}*")
            lib.axis("Z", "mass fraction", values=zmet[:, -1])
            lib.axis("logT", "log10(K)", values=logt)
            lib.axis("wavelength", "angstrom", data=lamwr)
            lib.array("flux", spec, axes=["Z", "logT", "wavelength"], units="L_nu / L_bol (Hz^-1)")
            self.finish(lib)
        # Post-AGB (load_post_agb_spectra)
        teff = one_per_record(self.p(d, "ipagb.teff"), 14)
        spec = np.zeros((2, 14, 9281))
        lam = None
        for j, fn in enumerate(["ipagb_halo.spec", "ipagb_solar.spec"]):
            r = ListReader(self.p(d, fn))
            raw = np.array([r.read(15) for _ in range(9281)])
            if lam is None:
                lam = raw[:, 0].copy()
            elif not np.array_equal(lam, raw[:, 0]):
                raise ValueError("post-AGB wavelength grids differ")
            spec[j] = raw[:, 1:].T
        lib = self.writer("hot_star_spectra", "post_agb", "Hydrogen-deficient post-AGB spectra", d + "/ipagb*")
        lib.axis("composition", "", labels=["halo", "solar"])
        lib.axis("teff", "K", values=teff)
        lib.axis("wavelength", "angstrom", data=lam)
        lib.array("flux", spec, axes=["composition", "teff", "wavelength"], units="L_nu / L_bol (Hz^-1)")
        self.finish(lib)

    def agb(self):
        """load_agb_spectra."""
        d = "spectra/AGB_spectra"
        r = ListReader(self.p(d, "Orich.teff"))
        r.skip()
        logz = r.read(23)[1:]
        teff_o = np.array([r.read(23)[1:] for _ in range(9)])        # [type, logZ]
        r = ListReader(self.p(d, "Crich.teff"))
        r.skip()
        teff_c = np.array([r.read(2)[1] for _ in range(5)])
        teff_car = ListReader(self.p(d, "Crich_Aringer.teff")).read(9)
        for name, fn, ntype, nlam, title in (("orich", "Orich", 9, 6146, "Empirical O-rich TP-AGB spectra"),
                                             ("crich", "Crich", 5, 6146, "Empirical C-rich TP-AGB spectra"),
                                             ("crich_aringer", "Crich_Aringer", 9, 9032,
                                              "Theoretical C-rich TP-AGB spectra")):
            r = ListReader(self.p(d, fn + ".spec"))
            raw = np.array([r.read(1 + ntype) for _ in range(nlam)])
            lib = self.writer("agb_spectra", name, title, f"{d}/{fn}.{{teff,spec}}")
            lib.axis("type", "spectral type index", values=list(range(1, ntype + 1)))
            lib.axis("wavelength", "angstrom", data=raw[:, 0])
            if name == "orich":
                lib.axis("logZ", "log10(Z/Z_sun)", values=logz)
                lib.array("teff", teff_o, axes=["type", "logZ"], units="K")
            else:
                lib.array("teff", teff_c if name == "crich" else teff_car, axes=["type"], units="K")
            units = ("F_lambda normalized to unit integral over angstrom" if name != "crich_aringer"
                     else "L_nu / L_bol (Hz^-1)")
            lib.array("flux", raw[:, 1:].T, axes=["type", "wavelength"], units=units)
            self.finish(lib)

    def dusty_agb(self):
        """load_dusty_agb_spectra."""
        for name in ("orich", "crich"):
            fn = f"{name.capitalize()}_dusty.spec"
            r = ListReader(self.p("dust", "dusty", fn))
            nlam = r.read_int(1)[0]
            lam = r.read(nlam)
            teff = np.zeros((6, 50))
            tau = np.zeros((6, 50))
            flux = np.zeros((6, 50, nlam))
            for i in range(6):
                for j in range(50):
                    teff[i, j], tau[i, j] = r.read(2)
                    flux[i, j] = r.read(nlam)
            if not (np.all(teff == teff[:, -1:]) and np.all(tau == tau[-1:, :])):
                raise ValueError(f"{fn}: (Teff, tau) labels are not a regular grid")
            lib = self.writer("dusty_agb", name, f"Circumstellar dust shells of {name[0].upper()}-rich AGB stars",
                              "dust/dusty/" + fn)
            lib.axis("teff", "K", values=teff[:, -1])
            lib.axis("tau", "optical depth at 1 micron", values=tau[-1, :])
            lib.axis("wavelength", "angstrom", data=lam)
            lib.array("transfer", flux, axes=["teff", "tau", "wavelength"],
                      units="ratio of emergent to input spectrum")
            self.finish(lib)

    def xrb(self):
        """load_xrb_spectra."""
        nspec = 15000
        lam = one_per_record(self.p("spectra", "xrb", "xsp.lambda"), nspec)
        spec = np.zeros((11, 10, nspec))
        for j, s in enumerate(XRB_FEH_STR):
            r = ListReader(self.p("spectra", "xrb", f"xsp_feh{s}.spec"))
            for i in range(10):
                spec[j, i] = r.read(nspec)
        lib = self.writer("xrb", "xsp", "X-ray binary spectra of simple populations", "spectra/xrb")
        lib.axis("feh", "[Fe/H]", values=XRB_FEH)
        # log10(age / yr), correctly rounded as the compile-time constant of the legacy code
        logage = [float(decimal.Decimal(repr(a)).log10()) + 6.0 for a in XRB_AGES_MYR]
        lib.axis("log_age", "log10(yr)", values=logage)
        lib.axis("wavelength", "angstrom", data=lam)
        lib.array("flux", spec, axes=["feh", "log_age", "wavelength"],
                  units="L_sun Hz^-1 per M_sun (scaled by L_sun on load)")
        self.finish(lib)

    # ------------------------------------------------------------------ upstream [alpha/Fe] layouts
    @staticmethod
    def afe_from_iso_tag(tag):
        """MIST isochrone tag 'm2', 'p0', 'p4' -> -0.2, 0.0, +0.4."""
        v = int(tag[1:]) / 10.0
        return -v if tag[0] == "m" else v

    def upstream_mist(self, src, name="mist", zsol=0.0185, nt=107):
        """Alpha-enhanced MIST isochrones of the upstream layout.

        ``src/zlegend.dat`` holds lines '(A4,F6.2)': a file tag ('m250') and [Fe/H]; the files
        are ``isoc_feh_<tag>_afe_<m2|p0|p2|p4|p6>_vvcrit0.4_full.dat`` (MIST format, 9 columns).
        Output: axes afe (default 0) and feh; an ``available`` mask when not every [alpha/Fe]
        exists at every [Fe/H].
        """
        import glob
        tags, feh = [], []
        with open(os.path.join(src, "zlegend.dat")) as f:
            for line in f:
                if line.strip():
                    tags.append(line[:4])
                    feh.append(fortran_float(line[4:10].strip()))
        afe_tags = sorted({os.path.basename(p).split("_afe_")[1].split("_")[0]
                           for p in glob.glob(os.path.join(src, "isoc_feh_*_afe_*_vvcrit0.4_full.dat"))},
                          key=self.afe_from_iso_tag)
        afe = [self.afe_from_iso_tag(t) for t in afe_tags]
        nfeh, nafe = len(tags), len(afe_tags)
        avail = np.zeros((nafe, nfeh), dtype=np.int32)
        npts = np.zeros((nafe, nfeh, nt), dtype=np.int32)
        logage = np.zeros((nafe, nfeh, nt))
        cols = []
        for ia, at in enumerate(afe_tags):
            for iz, zt in enumerate(tags):
                path = os.path.join(src, f"isoc_feh_{zt}_afe_{at}_vvcrit0.4_full.dat")
                if not os.path.exists(path):
                    continue
                avail[ia, iz] = 1
                for it, (age, rows) in enumerate(self.parse_isochrone_file(path, nt, True)):
                    npts[ia, iz, it] = len(rows)
                    logage[ia, iz, it] = age
                    cols.extend(rows)
        cols = np.array(cols, dtype=np.float64).reshape(-1, 8)
        lib = LibraryWriter(self.out, "isochrones", name, title="MIST isochrones with [alpha/Fe] variations",
                            provenance=dict(provenance("isochrones", name), converted_from="upstream FSPS layout: " + src))
        lib.metadata["solar_metallicity"] = zsol
        lib.axis("afe", "[alpha/Fe] (dex)", values=afe, default=0.0)
        lib.axis("feh", "[Fe/H] (dex); Z = solar_metallicity * 10^[Fe/H]", values=feh)
        lib.axis("age_index", "index", values=list(range(1, nt + 1)))
        lib.array("npoints", npts, axes=["afe", "feh", "age_index"], dtype="<i4", units="count")
        lib.array("log_age", logage, axes=["afe", "feh", "age_index"], units="log10(yr)")
        if not avail.all():
            lib.array("available", avail, axes=["afe", "feh"], dtype="<i4", units="1 if tabulated, 0 if absent")
        for k, cname in enumerate(["mini", "mact", "logl", "logt", "logg", "composition", "phase", "log_mdot"]):
            lib.array(cname, cols[:, k], units="as the legacy isochrone column",
                      description="one value per point; ordered by afe, feh, isochrone, mass")
        self.finish(lib)
        return lib

    def upstream_c3k(self, src, name=None, zsol=0.0185):
        """Per-[alpha/Fe] C3K spectra of the upstream layout (``SPECTRA/C3K/<name>/``).

        ``<name>.lambda``, ``<name>.res``, ``<name>_zlegend.dat`` ([Fe/H] strings such as -2.50),
        ``logt.dat`` (log T in the first column), ``logg.dat``, and one FP32 binary per
        ([Fe/H], [alpha/Fe]): ``<name>_feh<feh>_afe<afe>.spec.bin`` = (nspec, nlogt, nlogg).
        """
        import glob
        name = name or os.path.basename(os.path.normpath(src))
        lam = np.array([fortran_float(split_tokens(l)[0]) for l in open(os.path.join(src, name + ".lambda"))
                        if l.strip()])
        res = np.array([fortran_float(split_tokens(l)[0]) for l in open(os.path.join(src, name + ".res"))
                        if l.strip()])
        ztags = [l.strip() for l in open(os.path.join(src, name + "_zlegend.dat")) if l.strip()]
        logt = np.array([fortran_float(split_tokens(l)[0]) for l in open(os.path.join(src, "logt.dat")) if l.strip()])
        logg = np.array([fortran_float(split_tokens(l)[0]) for l in open(os.path.join(src, "logg.dat")) if l.strip()])
        afe_tags = sorted({os.path.basename(p)[:-len(".spec.bin")].split("_afe")[1]
                           for p in glob.glob(os.path.join(src, f"{name}_feh*_afe*.spec.bin"))}, key=float)
        nspec, nt, ng = len(lam), len(logt), len(logg)
        flux = np.full((len(afe_tags), len(ztags), ng, nt, nspec), np.nan, dtype="<f4")
        avail = np.zeros((len(afe_tags), len(ztags)), dtype=np.int32)
        for ia, at in enumerate(afe_tags):
            for iz, zt in enumerate(ztags):
                path = os.path.join(src, f"{name}_feh{zt}_afe{at}.spec.bin")
                if not os.path.exists(path):
                    continue
                raw = np.fromfile(path, dtype="<f4")
                if raw.size != nspec * nt * ng:
                    raise ValueError(f"{path}: {raw.size} values, expected {nspec * nt * ng}")
                flux[ia, iz] = raw.reshape(ng, nt, nspec)
                avail[ia, iz] = 1
        lib = LibraryWriter(self.out, "stellar_spectra", name, title=f"C3K spectra ({name}) with [alpha/Fe] variations",
                            provenance=dict(provenance("stellar_spectra", name), converted_from="upstream FSPS layout: " + src))
        lib.metadata["solar_metallicity"] = zsol
        lib.axis("afe", "[alpha/Fe] (dex)", values=[float(t) for t in afe_tags], default=0.0)
        lib.axis("feh", "[Fe/H] (dex); Z = solar_metallicity * 10^[Fe/H]", values=[float(t) for t in ztags])
        lib.axis("logT", "log10(K)", values=logt)
        lib.axis("logg", "log10(cm s^-2)", values=logg)
        lib.axis("wavelength", "angstrom", data=lam, medium="vacuum")
        lib.array("flux", flux, axes=["afe", "feh", "logg", "logT", "wavelength"], dtype="<f4",
                  units="erg s^-1 cm^-2 Hz^-1 (Eddington flux H_nu)")
        lib.array("resolution", res, axes=["wavelength"], units="as the legacy .res file")
        if not avail.all():
            lib.array("available", avail, axes=["afe", "feh"], dtype="<i4", units="1 if tabulated, 0 if absent")
        self.finish(lib)
        return lib

    def small_tables(self):
        os.makedirs(os.path.join(self.out, "data"), exist_ok=True)
        for fn in SMALL_TABLES:
            src = self.p(fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.out, "data", fn))

    # ------------------------------------------------------------------
    def jobs(self):
        j = []
        for n in SPECTRA:
            j.append((f"stellar_spectra/{n}", lambda n=n: self.stellar_spectra(n)))
        for n in ISOCHRONES:
            j.append((f"isochrones/{n}", lambda n=n: self.isochrones(n)))
        j.append(("ssp/bpss", self.ssp_bpass))
        for iso in ("mist", "pdva", "prsc", "bpss"):
            for dust in (False, True):
                j.append((f"nebular/{iso}_{'dust' if dust else 'nodust'}",
                          lambda iso=iso, dust=dust: self.nebular(iso, dust)))
        for dust in (False, True):
            j.append((f"nebular/bpss_xray_{'dust' if dust else 'nodust'}",
                      lambda dust=dust: self.nebular("bpss", dust, xray=True)))
        for n in DUSTEM:
            j.append((f"dust_emission/{n}", lambda n=n: self.dust_emission(n)))
        j += [("agn_torus/clumpy", self.agn_torus), ("attenuation", self.attenuation),
              ("filters/allfilters", self.filters), ("reference_spectra", self.reference_spectra),
              ("hot_star_spectra", self.hot_stars), ("agb_spectra", self.agb), ("dusty_agb", self.dusty_agb),
              ("xrb/xsp", self.xrb)]
        return j


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("legacy_root")
    ap.add_argument("out_root")
    ap.add_argument("--only", nargs="*", default=None, help="kind or kind/name prefixes to convert")
    ap.add_argument("--release", default="unreleased")
    ap.add_argument("--list", action="store_true", help="list the conversion jobs and exit")
    ap.add_argument("--upstream-mist", metavar="DIR",
                    help="also convert upstream alpha-enhanced MIST isochrones (ISOCHRONES/MIST) as isochrones/mist_afe")
    ap.add_argument("--upstream-c3k", metavar="DIR", action="append", default=[],
                    help="also convert an upstream per-[alpha/Fe] C3K directory (SPECTRA/C3K/<name>)")
    a = ap.parse_args(argv)
    conv = Converter(a.legacy_root, a.out_root)
    if a.upstream_mist or a.upstream_c3k:
        os.makedirs(a.out_root, exist_ok=True)
        if a.upstream_mist:
            conv.upstream_mist(a.upstream_mist, name="mist_afe")
        for d in a.upstream_c3k:
            conv.upstream_c3k(d)
        if a.only is None:
            a.only = ["(none)"]
    jobs = conv.jobs()
    if a.list:
        for name, _ in jobs:
            print(name)
        return 0
    os.makedirs(a.out_root, exist_ok=True)
    for name, fn in jobs:
        if a.only and not any(name.startswith(o) or o.startswith(name) for o in a.only):
            continue
        fn()
    conv.small_tables()
    write_root_index(a.out_root, a.release, "Libraries converted from the legacy FSPS layout")
    print(f"converted {len(conv.done)} libraries into {a.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
