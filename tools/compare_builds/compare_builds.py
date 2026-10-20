"""Compare two builds' outputs case by case (e.g. GPU vs CPU, FP32 kernels vs FP64).

Usage: python3 compare_builds.py TEST_DIR REFERENCE_DIR [CASES.psv] [--rtol R] [--mag-tol M]

Both directories come from run_cases.sh. Spectra are compared per pixel where the reference is
above 1e-6 of each spectrum's peak (FP32 rounding dominates below), magnitudes only for bands
within 30 mag of the brightest (fainter ones carry numerically zero flux), spectral indices in
absolute terms, emission lines above 1e-3 of the strongest. Exits non-zero if a case exceeds
the tolerances; the defaults suit FP32 kernels against FP64 (measured worst: 3.3e-7 per pixel,
2e-7 mag, 5e-6 indices; doc/MIXED_PRECISION.md).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "legacy_parity"))
from compare_legacy import read_opsf  # noqa: E402


def compare(A, B):
    w = B["wavelengths"]
    uv = (w > 912) & (w < 2.5e4)
    a, b = np.atleast_2d(A["csp/spec"]), np.atleast_2d(B["csp/spec"])
    big = b > 1e-6 * b.max(axis=1, keepdims=True)
    r = np.abs(a / np.where(b > 0, b, 1) - 1) * big
    ma, mb = np.atleast_2d(A["csp/mags"]), np.atleast_2d(B["csp/mags"])
    bright = mb < np.nanmin(mb, axis=1, keepdims=True) + 30
    le, fe = np.atleast_2d(B["csp/emlines"]), np.atleast_2d(A["csp/emlines"])
    strong = le > 1e-3 * le.max() if le.max() > 0 else np.zeros_like(le, bool)
    return dict(
        dlogM=float(np.max(np.abs(np.log10(A["csp/mass"]) - np.log10(B["csp/mass"])))),
        dlogL=float(np.max(np.abs(A["csp/lbol"] - B["csp/lbol"]))),
        spec=float(r[:, uv].max()), spec_all=float(r.max()),
        mag=float(np.nanmax(np.where(bright, np.abs(ma - mb), 0))),
        indx=float(np.nanmax(np.abs(np.atleast_2d(A["csp/indx"]) - np.atleast_2d(B["csp/indx"])))),
        lines=float(np.max(np.abs(fe[strong] / le[strong] - 1))) if strong.any() else 0.0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("test"); p.add_argument("reference")
    p.add_argument("cases", nargs="?", default=os.path.join(ROOT, "tests/data/param_cases.psv"))
    p.add_argument("--rtol", type=float, default=2e-6, help="per-pixel spectrum, lines, mass/L_bol (dex)")
    p.add_argument("--mag-tol", type=float, default=1e-5)
    p.add_argument("--indx-tol", type=float, default=1e-4)
    args = p.parse_args()
    cases = [l.split("|")[0] for l in open(args.cases) if l.strip() and not l.startswith("#")]
    cols = ["dlogM", "dlogL", "spec", "spec_all", "mag", "indx", "lines"]
    tol = dict(dlogM=args.rtol, dlogL=args.rtol, spec=args.rtol, spec_all=10 * args.rtol,
               mag=args.mag_tol, indx=args.indx_tol, lines=args.rtol)
    print(f"{'case':26s} " + " ".join(f"{c:>9s}" for c in cols))
    bad = []
    for c in cases:
        paths = [os.path.join(d, c + ".opsf") for d in (args.test, args.reference)]
        if not all(os.path.exists(x) for x in paths):
            print(f"{c:26s} missing"); bad.append((c, "missing")); continue
        res = compare(*(read_opsf(x) for x in paths))
        cells = []
        for k in cols:
            over = res[k] > tol[k]
            cells.append(f"{res[k]:8.1e}{'!' if over else ' '}")
            if over:
                bad.append((c, f"{k} = {res[k]:.2e} > {tol[k]:.1e}"))
        print(f"{c:26s} " + " ".join(cells))
    if bad:
        print(f"\nFAIL: {len(bad)}"); [print(f"  {c}: {m}") for c, m in bad]; sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
