"""Compare the legacy and fork outputs of the parity matrix (run_matrix.sh), case by case.

Usage: python3 compare_matrix.py OUTDIR [CASES.psv]

OUTDIR holds legacy/<case>.opsf and fork/<case>.opsf. Prints one row per case and exits
non-zero if any tolerance is exceeded. Default tolerances are about twice the residuals of
the cases that hit no intentional difference; cases that do hit one get a wider tolerance,
named with the difference in REASONS (see README.md, "Intentional differences").
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_legacy import read_opsf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Columns: max |d log M| and |d log L_bol| (dex); max per-pixel |fork/legacy - 1| over 912 A -
# 2.5 um and above 5 um; max |d mag| for bands with lambda_eff below and above 2.5 um; max
# |d index|; max relative emission-line difference (lines above 1e-3 of the strongest).
# mag_uvnir is set by the filter clamp (legacy normalizes before clamping the negative
# transmission its edge extrapolation produces: LSST u 0.0008 mag) and mag_ir by the same for
# Bessell M (0.004 mag); both apply to every case.
COLUMNS = ["dlogM", "dlogL", "spec_uvnir", "spec_ir", "mag_uvnir", "mag_ir", "indx", "lines"]
DEFAULT_TOL = dict(dlogM=2e-6, dlogL=2e-6, spec_uvnir=5e-6, spec_ir=8e-2,
                   mag_uvnir=3e-3, mag_ir=1e-2, indx=1e-4, lines=3e-6)

# Wider tolerances for cases that exercise an intentional difference (None: not checked)
REASONS = {
    "sbss_hb": "blue stragglers and horizontal branch placed from the native isochrone",
    "smoothing": "fork velocity-smoothing kernel (exact at MILES grid-spacing jumps)",
    "calzetti": "Calzetti R_V = 4.05 exactly (legacy 4.0455)",
    "reddy": "Reddy curve constant blueward of exactly 1500 A (legacy: of the pixel below)",
    "ccm": "CCM UV/optical junction matched at exactly x = 3.3 (legacy: at the nearest pixel)",
    "self_abs": "analytic dust self-absorption (legacy's iteration loses up to 0.25% L_bol)",
    "bpass_lbol": "BPASS L_bol from the spectrum integral (legacy reports log L = 0)",
}
OVERRIDES = {
    "ssp_mods": ("sbss_hb", dict(dlogM=None, dlogL=None, spec_uvnir=None, mag_uvnir=None, mag_ir=None,
                                    indx=None)),
    "csp_smooth": ("smoothing", dict(spec_uvnir=3e-2, indx=2e-3)),
    "csp_dust2": ("calzetti", dict(spec_uvnir=1e-3)),
    "csp_dust4": ("calzetti", dict(spec_uvnir=1e-3)),
    "csp_duste_off": ("calzetti", dict(spec_uvnir=1e-3)),
    "csp_duste_qpah": ("calzetti", dict(spec_uvnir=1e-3)),
    "lib_themis": ("calzetti", dict(spec_uvnir=1e-3)),
    "csp_dust1": ("ccm", dict(spec_uvnir=5e-5)),
    "csp_dust6": ("reddy", dict(spec_uvnir=5e-3)),
    "csp_nebular": ("self_abs", dict(spec_uvnir=2e-5)),
    "csp_dust0": ("self_abs", dict(spec_uvnir=2e-4, mag_ir=3e-2)),
    "csp_dust0_full": ("self_abs", dict(spec_uvnir=6e-4, mag_ir=7e-2)),
    "lib_padova_basel": ("self_abs", dict(spec_uvnir=2e-4, mag_ir=3e-2)),
    "lib_padova_basel_offgrid": ("self_abs", dict(spec_uvnir=2e-4, mag_ir=3e-2)),
    "lib_mist_c3k": ("self_abs", dict(spec_uvnir=2e-4, mag_ir=4e-2)),
    "lib_bpass": ("bpass_lbol", dict(dlogL=None, spec_uvnir=2e-4, mag_ir=3e-2)),
}
SPLIT_UM = 2.5


def pivot_wavelengths(path):
    """Pivot wavelength of each filter in allfilters.dat (filter_lambda_eff.dat lacks the newer ones)."""
    curves, cur = [], None
    for line in open(path):
        if line.lstrip().startswith("#"):
            cur = []
            curves.append(cur)
        elif line.strip():
            cur.append([float(v) for v in line.split()[:2]])
    piv = []
    for c in curves:
        lam, t = np.array(c).T
        piv.append(math.sqrt(np.trapezoid(t * lam, lam) / np.trapezoid(t / lam, lam)))
    return np.array(piv)


def max_ratio_dev(fork, legacy, mask):
    good = mask & (legacy > 0)
    return float(np.max(np.abs(fork[good] / legacy[good] - 1))) if good.any() else 0.0


def compare(L, F, band_ir):
    w = L["wavelengths"]
    uvnir = (w > 912) & (w < SPLIT_UM * 1e4)
    ir = w > 5e4
    ages = np.atleast_1d(F["csp/age"])
    sel = np.arange(ages.size)
    fs, ls = np.atleast_2d(F["csp/spec"]), np.atleast_2d(L["csp/spec"])
    dmag = np.abs(np.atleast_2d(F["csp/mags"])[sel] - np.atleast_2d(L["csp/mags"])[sel])
    fe, le = np.atleast_2d(F["csp/emlines"])[sel], np.atleast_2d(L["csp/emlines"])[sel]
    strong = (le > 1e-3 * le.max()) & (le > 1e-60)
    return dict(
        dlogM=float(np.max(np.abs(np.log10(F["csp/mass"]) - np.log10(L["csp/mass"])))),
        dlogL=float(np.max(np.abs(F["csp/lbol"] - L["csp/lbol"]))),
        spec_uvnir=max(max_ratio_dev(fs[i], ls[i], uvnir) for i in sel),
        spec_ir=max(max_ratio_dev(fs[i], ls[i], ir) for i in sel),
        mag_uvnir=float(np.nanmax(dmag[:, ~band_ir])),
        mag_ir=float(np.nanmax(dmag[:, band_ir])),
        indx=float(np.nanmax(np.abs(np.atleast_2d(F["csp/indx"])[sel] - np.atleast_2d(L["csp/indx"])[sel]))),
        lines=float(np.max(np.abs(fe[strong] / le[strong] - 1))) if strong.any() else 0.0,
    )


def main():
    out = sys.argv[1]
    cases_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "tests/data/param_cases.psv")
    cases = [l.split("|")[0] for l in open(cases_file) if l.strip() and not l.startswith("#")]
    lam_eff = pivot_wavelengths(os.path.join(ROOT, "data/allfilters.dat"))
    band_ir = lam_eff >= SPLIT_UM * 1e4

    print(f"{'case':26s} " + " ".join(f"{c:>10s}" for c in COLUMNS) + "  note")
    failures = []
    for case in cases:
        paths = [os.path.join(out, side, case + ".opsf") for side in ("legacy", "fork")]
        if not all(os.path.exists(p) for p in paths):
            print(f"{case:26s} missing output")
            failures.append((case, "missing output"))
            continue
        res = compare(*(read_opsf(p) for p in paths), band_ir)
        reason, extra = OVERRIDES.get(case, (None, {}))
        tol = {**DEFAULT_TOL, **extra}
        cells = []
        for c in COLUMNS:
            bad = tol[c] is not None and not (res[c] <= tol[c])
            cells.append(f"{res[c]:9.2e}{'!' if bad else ' '}")
            if bad:
                failures.append((case, f"{c} = {res[c]:.3g} > {tol[c]:.3g}"))
        print(f"{case:26s} " + " ".join(cells) + (f"  {reason}" if reason else ""))

    print()
    for key, text in REASONS.items():
        print(f"  {key:11s} {text}")
    if failures:
        print(f"\nFAIL: {len(failures)} tolerance(s) exceeded")
        for case, msg in failures:
            print(f"  {case}: {msg}")
        sys.exit(1)
    print("\nPASS: all cases within tolerance")


if __name__ == "__main__":
    main()
