#!/usr/bin/env python3
"""Convert the released weights of cue (Li et al. 2024, https://github.com/yi-jia-li/cue) into an
opensps-data library that libopensps reads (``nebular_model = 1``).

cue emulates CLOUDY nebular emission (128 + 10 lines and the nebular continuum) with 17 small
neural networks ("speculator" architecture, Alsing et al. 2020): one for the continuum and one per
group of lines (by element and ionization potential). Each takes the same 12 inputs, has three
hidden layers of 256 units, and predicts PCA coefficients; the PCA basis gives log10 of the
luminosities. This tool reads the networks and PCA bases from cue's data directory and writes
them, unchanged (FP32, as trained), as arrays of one library:

    <out>/opensps-data.json
    <out>/libraries/nebular_emulator/cue/manifest.json
    <out>/libraries/nebular_emulator/cue/*.bin

Point libopensps at it with ``export OPENSPS_CUE_DATA=<out>``. The weights are not distributed
with libopensps; this conversion is the user's step (cue is MIT licensed).

Only NumPy is needed: cue's pickles name classes from TensorFlow, scikit-learn and cue itself,
which are replaced here by stand-ins that keep their stored attributes (no code of those
packages runs). With ``--reference`` the tool also stores reference outputs for the tests of
libopensps: the networks evaluated in FP64 by the same algebra as cue's NumPy path
(``Speculator.log_spectrum_``, ``IncrementalPCA.inverse_transform``), and, when SciPy and dill are
importable, cue's own ionizing-spectrum fit (``cue.utils.fit_4loglinear_ionparam``) on synthetic
spectra.

Usage:
    python3 tools/cue/convert_cue.py <cue checkout or its src/cue/data> <out> [--reference]
"""

import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys


import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "opensps_data"))
from opensps_data import ROOT_INDEX, FORMAT, FORMAT_VERSION, LibraryWriter  # noqa: E402

KIND = "nebular_emulator"
NAME = "cue"

# Line groups in cue's order (cue/utils.py: nn_name, nn_ion)
GROUPS = ["H1", "He1", "He2", "C1", "C2C3", "C4", "N", "O1", "O2", "O3",
          "ionE_1", "ionE_2", "S4", "Ar4", "Ne3", "Ne4"]
GROUP_IONS = [["H  1"], ["He 1"], ["He 2"], ["C  1"], ["C  2", "C  3"], ["C  4"],
              ["N  1", "N  2", "N  3"], ["O  1"], ["O  2"], ["O  3"],
              ["Mg 2", "Fe 2", "Si 2", "Al 2", "P  2", "S  2", "Cl 2", "Ar 2"],
              ["Al 3", "Si 3", "S  3", "Cl 3", "Ar 3", "Ne 2"],
              ["S  4"], ["Ar 4"], ["Ne 3"], ["Ne 4"]]
N_IN, N_HIDDEN, N_LAYERS = 12, 256, 3
NPCA_LINE_PAD = 16            # line groups have 2-13 PCA components; padded with zeros
CONT_FIRST = 122              # cue's continuum: FSPSlam.dat[122:] (915 A - 1e8 A)

# Training ranges of the 12 inputs (cue paper, Table 1; logQ from log U and n_H)
INPUT_NAMES = ["ionspec_index1", "ionspec_index2", "ionspec_index3", "ionspec_index4",
               "ionspec_logLratio1", "ionspec_logLratio2", "ionspec_logLratio3",
               "gas_logq", "gas_nh", "gas_logz", "gas_logno", "gas_logco"]


# ---------------------------------------------------------------------------------------------
# Reading cue's pickles without TensorFlow, scikit-learn or cue
# ---------------------------------------------------------------------------------------------

class _Stub:
    """Stand-in for a pickled class instance: keeps its state as attributes."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _ListStub(list):
    """Stand-in for TensorFlow's ListWrapper (a list)."""

    def __setstate__(self, state):
        self._state = state


def _create_array(f, args, state, npdict=None):
    """dill's _create_array: rebuild an ndarray from its reduce tuple."""
    a = f(*args)
    a.__setstate__(state)
    if npdict:
        a.__dict__.update(npdict)
    return a


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy.core"):
            alt = module.replace("numpy.core", "numpy._core")
            try:
                return super().find_class(alt, name)
            except (ImportError, AttributeError):
                pass
        if module == "dill._dill" and name == "_create_array":
            return _create_array
        if name == "ListWrapper":
            return _ListStub
        if module.startswith("cue") or module.startswith("sklearn") or module.startswith("tensorflow"):
            return _Stub
        return super().find_class(module, name)


def _load(path):
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def _speculator(path):
    """The attributes Speculator.save writes, in order (cue/nn.py)."""
    (W, b, alphas, betas, pshift, pscale, pca_shift, pca_scale, lss_shift, lss_scale, ptm,
     n_par, n_wav, wav, n_pcas, n_hidden, n_layers, arch) = _load(path)
    W = [np.asarray(w) for w in W]
    b = [np.asarray(x) for x in b]
    net = dict(W=W, b=b, alpha=[np.asarray(a) for a in alphas], beta=[np.asarray(x) for x in betas],
               pshift=np.asarray(pshift), pscale=np.asarray(pscale),
               pca_shift=np.asarray(pca_shift), pca_scale=np.asarray(pca_scale),
               lss_shift=np.asarray(lss_shift), lss_scale=np.asarray(lss_scale),
               ptm=np.asarray(ptm), n_pcas=int(n_pcas))
    if len(W) != N_LAYERS + 1 or W[0].shape != (N_IN, N_HIDDEN) or \
            any(w.shape != (N_HIDDEN, N_HIDDEN) for w in W[1:N_LAYERS]) or W[-1].shape[0] != N_HIDDEN:
        raise SystemExit(f"{path}: unexpected architecture {[w.shape for w in W]}")
    return net


def _pca(path):
    """The IncrementalPCA of a cue SpectrumPCA pickle (inverse_transform: X @ components_ + mean_)."""
    p = _load(path).PCA
    if getattr(p, "whiten", False):
        raise SystemExit(f"{path}: whitened PCA is not supported")
    return np.asarray(p.components_), np.asarray(p.mean_)


# ---------------------------------------------------------------------------------------------
# Line list (cue/utils.py)
# ---------------------------------------------------------------------------------------------

def air_to_vacuum(lam):
    """Vacuum wavelength [A] of an air wavelength (Morton 2000 / IAU; as in the SDSS and FSPS
    tools). cue quotes air wavelengths above 2000 A (CLOUDY's convention)."""
    lam = np.asarray(lam, dtype=np.float64)
    s2 = (1.0e4 / lam) ** 2
    n = 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)
    return np.where(lam > 2000.0, lam * n, lam)


def line_list(data):
    names = np.load(os.path.join(data, "lineList_replaceblnd_name.npy"))
    lam = np.load(os.path.join(data, "lineList_wav.npy"))
    order = np.argsort(lam)
    sorted_names, sorted_lam = names[order], lam[order]
    ele = np.array([str(s)[:4].rstrip() for s in sorted_names])
    sel = [np.where(np.isin(ele, np.atleast_1d(ions)))[0] for ions in GROUP_IONS]
    return sorted_names, sorted_lam, sel


# ---------------------------------------------------------------------------------------------
# Forward pass, as cue's NumPy path (for the reference outputs)
# ---------------------------------------------------------------------------------------------

def forward_log(net, comp, mean, theta):
    """log10 luminosities: Speculator.log_spectrum_, then PCA.inverse_transform, then the
    output scale and shift (cue/emulator.py predict_lines / predict_cont). FP64 arithmetic on
    the FP32 weights."""
    x = (theta - net["pshift"].astype(np.float64)) / net["pscale"].astype(np.float64)
    for i in range(N_LAYERS):
        a = x @ net["W"][i].astype(np.float64) + net["b"][i].astype(np.float64)
        al = net["alpha"][i].astype(np.float64)
        be = net["beta"][i].astype(np.float64)
        x = (be + (1.0 - be) / (1.0 + np.exp(-al * a))) * a
    pc = x @ net["W"][-1].astype(np.float64) + net["b"][-1].astype(np.float64)
    pc = pc * net["pca_scale"].astype(np.float64) + net["pca_shift"].astype(np.float64)
    y = pc @ comp.astype(np.float64) + mean.astype(np.float64)
    return y * net["lss_scale"].astype(np.float64) + net["lss_shift"].astype(np.float64)


def _logq(logu, lognh):
    """cue.utils.logQ: log10 Q of the inputs from log U and log n_H (R_inner = 1e19 cm)."""
    return logu + np.log10(4.0 * np.pi) + 2.0 * np.log10(1.0e19) + lognh + np.log10(2.9979e10)


def reference_thetas(n, seed=20261110):
    """Inputs inside cue's training box (Table 1), as the networks take them (logQ, n_H)."""
    rng = np.random.default_rng(seed)
    lo = np.array([1.0, -0.3, -1.0, -1.7, -1.0, -0.5, -0.4, -4.0, 1.0, -2.2, -1.0, -1.0])
    hi = np.array([42.0, 30.0, 14.0, 8.0, 10.1, 1.9, 2.2, -1.0, 4.0, 0.5, np.log10(5.4), np.log10(5.4)])
    u = lo + (hi - lo) * rng.random((n, 12))
    th = u.copy()
    th[:, 7] = _logq(u[:, 7], u[:, 8])
    th[:, 8] = 10.0 ** u[:, 8]
    return th


def _cue_utils(cue_src):
    """cue.utils.fit_4loglinear_ionparam without importing cue's package (which needs
    TensorFlow): a stand-in package `cue` whose modules load from cue's source directory."""
    try:
        import dill  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return None
    # The package's spec (so importlib.resources finds its data), without running its __init__
    spec = importlib.util.spec_from_file_location("cue", os.path.join(cue_src, "__init__.py"),
                                                  submodule_search_locations=[cue_src])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["cue"] = pkg
    # NumPy 2 removed the names cue's utils use; the replacements compute the same
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    try:
        from cue import utils  # noqa: E402
    except Exception as e:  # pragma: no cover - depends on the environment
        print(f"[convert_cue] cue.utils not importable ({e}); no fit reference", file=sys.stderr)
        return None
    return utils


def fit_minimum(utils, wave, spec):
    """cue.utils.fit_4loglinear_ionparam, with cue's own loss and gradient minimized to
    convergence (SLSQP, then BFGS from its result at gtol 1e-12): cue's SLSQP stops early on some
    segments (loss above the minimum by up to ~15%), so the minimum, not SLSQP's stopping point,
    is the reference. Returns the 7 parameters clipped as cue clips them."""
    from scipy.optimize import minimize
    edges = [utils.HeII_edge, utils.OII_edge, utils.HeI_edge, 911.6]
    ind = np.array([max(np.where(wave <= e)[0]) for e in edges]) + 1
    ind = np.insert(ind, 0, 0)
    norm = 1e-18 / spec[ind[-1]]
    ns = np.clip(spec * norm, 1e-70 * norm, np.inf)
    coeff = np.zeros((4, 2))
    for i in range(4):
        sl = slice(ind[i], ind[i + 1])
        if not np.any(spec[sl] > 0):
            coeff[i] = [0, -np.inf]
            continue
        x, y = wave[sl], ns[sl]
        q = np.abs(np.trapezoid(y * x / (utils.h * utils.c), x=utils.c / x))
        args = (x[0], x[-1], np.log(x[0]), np.log(x[-1]), np.log10(x), np.log10(y), np.log10(q), len(x))
        a0 = np.log10(y[-1] / y[0]) / np.log10(x[-1] / x[0])
        b0 = np.log10(y[-1]) - a0 * np.log10(x[-1])
        r = minimize(utils.objective_func_loglinear_analytical, [a0, b0], jac=utils.gradient_func_loglinear_analytical,
                     args=args, bounds=[(-40, 100), (-200, 100)], method="SLSQP")
        r = minimize(utils.objective_func_loglinear_analytical, r.x, jac=utils.gradient_func_loglinear_analytical,
                     args=args, method="BFGS", options={"gtol": 1e-12})
        coeff[i] = r.x
        coeff[i, 1] -= np.log10(norm)
    lr = np.diff(np.squeeze(utils.Ltotal(param=coeff.reshape(1, 4, 2))))
    lo = [1, -0.3, -1, -1.7, -1, -0.5, -0.4]
    hi = [42, 30, 14, 8, 10.1, 1.9, 2.2]
    return np.clip(np.concatenate([coeff[:, 0], lr]), lo, hi)


def synthetic_ionizing_spectra(wave, n, seed=7):
    """Spectra resembling young stellar populations below 912 A (blackbodies with breaks at the
    He II, O II and He I edges, a few with no flux in the bluest segment), in L_sun/Hz."""
    rng = np.random.default_rng(seed)
    h, c, k = 6.626e-27, 2.9979e18, 1.380649e-16
    out = np.zeros((n, wave.size))
    for i in range(n):
        T = rng.uniform(3.0e4, 1.2e5)
        nu = c / wave
        bb = nu ** 3 / np.expm1(np.minimum(h * nu / (k * T), 700.0))
        for edge, drop in ((227.838, rng.uniform(0.0, 3.0)), (353.0, rng.uniform(0.0, 0.5)),
                           (504.26, rng.uniform(0.0, 1.0))):
            bb = np.where(wave < edge, bb * 10.0 ** (-drop), bb)
        bb *= 1.0 + 0.05 * rng.standard_normal(wave.size)
        if i % 5 == 4:
            bb[wave < 227.838] = 0.0
        out[i] = np.maximum(bb, 0.0) / bb[wave < 911.6].max() * 1.0e-12
    return out


def git_describe(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cue", help="cue checkout (or its src/cue/data directory)")
    ap.add_argument("out", help="output data root (set OPENSPS_CUE_DATA to it)")
    ap.add_argument("--reference", action="store_true",
                    help="also store reference outputs for the libopensps tests")
    ap.add_argument("--n-reference", type=int, default=64)
    args = ap.parse_args(argv)

    data = args.cue
    for cand in (os.path.join(data, "src", "cue", "data"), os.path.join(data, "data"), data):
        if os.path.isfile(os.path.join(cand, "speculator_cont_new.pkl")):
            data = cand
            break
    else:
        raise SystemExit(f"no cue data directory (speculator_cont_new.pkl) under {args.cue}")
    cue_src = os.path.dirname(data)
    repo = os.path.dirname(os.path.dirname(cue_src))

    # Networks and PCA bases
    cont = _speculator(os.path.join(data, "speculator_cont_new.pkl"))
    cont_comp, cont_mean = _pca(os.path.join(data, "pca_cont_new.pkl"))
    lines, lcomp, lmean = [], [], []
    for g in GROUPS:
        lines.append(_speculator(os.path.join(data, f"speculator_line_new_{g}.pkl")))
        c, m = _pca(os.path.join(data, f"pca_line_new_{g}.pkl"))
        lcomp.append(c)
        lmean.append(m)
    nets = lines + [cont]                       # the hidden layers of all 17, continuum last

    # Consistency checks: the PCA basis the network was trained against
    for net, comp, g in zip(nets, lcomp + [cont_comp], GROUPS + ["cont"]):
        if comp.shape != net["ptm"].shape or not np.allclose(comp, net["ptm"], rtol=0, atol=1e-6):
            raise SystemExit(f"{g}: network PCA matrix differs from the PCA basis")
        if comp.shape[0] != net["W"][-1].shape[1] or comp.shape[0] != net["n_pcas"]:
            raise SystemExit(f"{g}: PCA size {comp.shape[0]} vs network output {net['W'][-1].shape[1]}")

    names, sorted_lam, sel = line_list(data)
    nlines = np.array([len(s) for s in sel], dtype=np.int32)
    for g, s, comp in zip(GROUPS, sel, lcomp):
        if comp.shape[1] != len(s):
            raise SystemExit(f"{g}: {comp.shape[1]} PCA outputs for {len(s)} lines")
    order = np.concatenate(sel)                 # line k of the output (group order) -> sorted index
    lam_air = sorted_lam[order]
    lam_vac = air_to_vacuum(lam_air)
    nmax = int(nlines.max())
    ng = len(GROUPS)

    cont_lam = np.genfromtxt(os.path.join(data, "FSPSlam.dat"))[CONT_FIRST:]
    if not np.all(np.diff(cont_lam) > 0) or cont_lam.size != cont_comp.shape[1]:
        raise SystemExit("unexpected continuum wavelength grid")

    # Stacked arrays (C order; libopensps reads them with the dimensions reversed)
    w1 = np.stack([n["W"][0] for n in nets])                   # (17, 12, 256)
    b1 = np.stack([n["b"][0] for n in nets])                   # (17, 256)
    w2 = np.stack([n["W"][1] for n in nets])                   # (17, 256, 256)
    b2 = np.stack([n["b"][1] for n in nets])
    w3 = np.stack([n["W"][2] for n in nets])
    b3 = np.stack([n["b"][2] for n in nets])
    alpha = np.stack([np.stack(n["alpha"]) for n in nets])     # (17, 3, 256)
    beta = np.stack([np.stack(n["beta"]) for n in nets])
    in_shift = np.stack([n["pshift"] for n in nets])           # (17, 12)
    in_scale = np.stack([n["pscale"] for n in nets])

    # (each padded array keeps its source's precision: the networks are FP32, scikit-learn's PCA
    # bases FP64; LibraryWriter refuses to narrow)
    def dt(name):
        return np.result_type(*[n[name] if not isinstance(n[name], list) else n[name][-1] for n in lines])
    lw4 = np.zeros((ng, N_HIDDEN, NPCA_LINE_PAD), np.result_type(*[n["W"][-1] for n in lines]))
    lb4 = np.zeros((ng, NPCA_LINE_PAD), np.result_type(*[n["b"][-1] for n in lines]))
    lpsh = np.zeros((ng, NPCA_LINE_PAD), dt("pca_shift"))
    lpsc = np.zeros((ng, NPCA_LINE_PAD), dt("pca_scale"))
    lcmp = np.zeros((ng, NPCA_LINE_PAD, nmax), np.result_type(*lcomp))
    lmn = np.zeros((ng, nmax), np.result_type(*lmean))
    llsh = np.zeros((ng, nmax), dt("lss_shift"))
    llsc = np.zeros((ng, nmax), dt("lss_scale"))
    npca = np.zeros(ng, np.int32)
    for g, (net, comp, mean) in enumerate(zip(lines, lcomp, lmean)):
        k, nl = comp.shape
        if k > NPCA_LINE_PAD:
            raise SystemExit(f"{GROUPS[g]}: {k} PCA components > {NPCA_LINE_PAD}")
        npca[g] = k
        lw4[g, :, :k] = net["W"][-1]
        lb4[g, :k] = net["b"][-1]
        lpsh[g, :k] = net["pca_shift"]
        lpsc[g, :k] = net["pca_scale"]
        lcmp[g, :k, :nl] = comp
        lmn[g, :nl] = mean
        llsh[g, :nl] = net["lss_shift"]
        llsc[g, :nl] = net["lss_scale"]

    src_commit = git_describe(repo)
    prov = {"converted_from": f"cue (https://github.com/yi-jia-li/cue) commit {src_commit}",
            "converter": "tools/cue/convert_cue.py",
            "license": "MIT (cue: Copyright (c) 2025 Yijia Li & contributors)",
            "reference": "Li et al. 2025, ApJ 986, 9 (arXiv:2405.04598)"}
    lib = LibraryWriter(args.out, KIND, NAME, title="cue nebular emission emulator (Li et al. 2024)",
                        notes="Speculator networks and PCA bases of cue, unchanged (FP32 as trained). "
                              "Inputs: the 12 of INPUT_NAMES (logQ from log U and n_H with R_inner = 1e19 cm, "
                              "n_H linear). Outputs: log10 L [erg/s] of the lines and log10 L_nu [erg/s/Hz] "
                              "of the continuum for that Q.",
                        provenance=prov)
    md = lib.metadata
    md.update({"n_inputs": N_IN, "n_hidden": N_HIDDEN, "n_layers": N_LAYERS, "n_line_groups": ng,
               "n_lines": int(nlines.sum()), "n_line_max": nmax, "n_pca_line_pad": NPCA_LINE_PAD,
               "n_pca_cont": int(cont_comp.shape[0]), "n_cont_wave": int(cont_lam.size),
               "lsun_cgs": 3.839e33, "r_inner_cm": 1.0e19, "c_cgs": 2.9979e10,
               "cont_lyman_cut_angstrom": 911.6, "input_names": INPUT_NAMES,
               "edges_angstrom": [1.0, 1e8 / 438908.8789, 1e8 / 283270.9, 1e8 / 198310.66637, 911.6],
               "activation": "speculator: (beta + (1 - beta) sigmoid(alpha x)) x",
               "cue_commit": src_commit})
    lib.axis("net", "", labels=GROUPS + ["cont"], description="network: 16 line groups, then the continuum")
    lib.axis("line_group", "", labels=GROUPS)
    lib.axis("input", "", labels=INPUT_NAMES)
    lib.axis("hidden", "", values=list(range(N_HIDDEN)))
    lib.axis("hidden_in", "", values=list(range(N_HIDDEN)))
    lib.axis("layer", "", values=[1, 2, 3])
    lib.axis("pca_line", "", values=list(range(NPCA_LINE_PAD)))
    lib.axis("line_slot", "", values=list(range(nmax)))
    lib.axis("pca_cont", "", values=list(range(int(cont_comp.shape[0]))))
    lib.axis("line", "angstrom", data=lam_vac, medium="vacuum",
             labels_air=[str(s) for s in names[order]],
             description="the 138 lines in output order (by group); vacuum wavelengths "
                         "(cue quotes air above 2000 A; converted, Morton 2000)")
    lib.axis("cont_wavelength", "angstrom", data=cont_lam.astype(np.float64),
             description="cue's continuum grid (FSPS's BaSeL grid from index 122)")

    lib.array("w1", w1, axes=["net", "input", "hidden"], description="layer 1 (in, out) per network")
    lib.array("b1", b1, axes=["net", "hidden"])
    lib.array("w2", w2, axes=["net", "hidden_in", "hidden"])
    lib.array("b2", b2, axes=["net", "hidden"])
    lib.array("w3", w3, axes=["net", "hidden_in", "hidden"])
    lib.array("b3", b3, axes=["net", "hidden"])
    lib.array("alpha", alpha, axes=["net", "layer", "hidden"])
    lib.array("beta", beta, axes=["net", "layer", "hidden"])
    lib.array("input_shift", in_shift, axes=["net", "input"])
    lib.array("input_scale", in_scale, axes=["net", "input"])
    lib.array("line_w4", lw4, axes=["line_group", "hidden_in", "pca_line"], description="zero-padded")
    lib.array("line_b4", lb4, axes=["line_group", "pca_line"])
    lib.array("line_pca_shift", lpsh, axes=["line_group", "pca_line"])
    lib.array("line_pca_scale", lpsc, axes=["line_group", "pca_line"])
    lib.array("line_components", lcmp, axes=["line_group", "pca_line", "line_slot"])
    lib.array("line_mean", lmn, axes=["line_group", "line_slot"])
    lib.array("line_out_shift", llsh, axes=["line_group", "line_slot"])
    lib.array("line_out_scale", llsc, axes=["line_group", "line_slot"])
    lib.array("line_npca", npca, axes=["line_group"])
    lib.array("line_count", nlines, axes=["line_group"])
    lib.array("cont_w4", cont["W"][-1], axes=["hidden_in", "pca_cont"])
    lib.array("cont_b4", cont["b"][-1], axes=["pca_cont"])
    lib.array("cont_pca_shift", cont["pca_shift"], axes=["pca_cont"])
    lib.array("cont_pca_scale", cont["pca_scale"], axes=["pca_cont"])
    lib.array("cont_components", cont_comp, axes=["pca_cont", "cont_wavelength"])
    lib.array("cont_mean", cont_mean, axes=["cont_wavelength"])
    lib.array("cont_out_shift", cont["lss_shift"], axes=["cont_wavelength"])
    lib.array("cont_out_scale", cont["lss_scale"], axes=["cont_wavelength"])

    if args.reference:
        nref = args.n_reference
        th = reference_thetas(nref)
        lref = np.concatenate([forward_log(n, c, m, th) for n, c, m in zip(lines, lcomp, lmean)], axis=1)
        cref = forward_log(cont, cont_comp, cont_mean, th)
        lib.axis("reference", "", values=list(range(nref)))
        lib.array("reference_theta", th, axes=["reference", "input"], dtype="<f8",
                  description="network inputs (logQ, n_H linear) inside the training box")
        lib.array("reference_line_log", lref, axes=["reference", "line"], dtype="<f8",
                  description="log10 L [erg/s] by cue's NumPy algebra in FP64")
        lib.array("reference_cont_log", cref, axes=["reference", "cont_wavelength"], dtype="<f8")
        utils = _cue_utils(cue_src)
        if utils is not None:
            wave = np.geomspace(90.0, 1200.0, 400)
            spec = synthetic_ionizing_spectra(wave, 24)
            keys = ["ionspec_index1", "ionspec_index2", "ionspec_index3", "ionspec_index4",
                    "ionspec_logLratio1", "ionspec_logLratio2", "ionspec_logLratio3"]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slsqp = np.array([[utils.fit_4loglinear_ionparam(wave, s)[k] for k in keys] for s in spec])
                pars = np.array([fit_minimum(utils, wave, s) for s in spec])
            lib.axis("fit_wavelength", "angstrom", data=wave)
            lib.axis("fit_case", "", values=list(range(spec.shape[0])))
            lib.axis("ion_param", "", labels=keys)
            lib.array("reference_fit_spec", spec, axes=["fit_case", "fit_wavelength"], dtype="<f8",
                      units="L_sun/Hz", description="synthetic ionizing spectra")
            lib.array("reference_fit_params", pars, axes=["fit_case", "ion_param"], dtype="<f8",
                      description="cue's fit (fit_4loglinear_ionparam) with its loss minimized to convergence")
            lib.array("reference_fit_params_slsqp", slsqp, axes=["fit_case", "ion_param"], dtype="<f8",
                      description="cue.utils.fit_4loglinear_ionparam as released (SLSQP, which stops early "
                                  "on some segments)")
    lib.write()

    idx = os.path.join(args.out, ROOT_INDEX)
    with open(idx, "w") as f:
        json.dump({"format": FORMAT, "format_version": FORMAT_VERSION, "release": f"cue-{src_commit[:12]}",
                   "title": "cue weights converted for libopensps",
                   "libraries": [{"kind": KIND, "name": NAME, "path": f"libraries/{KIND}/{NAME}"}]},
                  f, indent=1)
        f.write("\n")
    print(f"wrote {lib.dir}: 17 networks, {int(nlines.sum())} lines, {cont_lam.size} continuum "
          f"wavelengths{' + references' if args.reference else ''}")
    print(f"export OPENSPS_CUE_DATA={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
