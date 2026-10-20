"""Compare a legacy FSPS SSP dump (ssp_dump) with the fork's SSPs (an OPSF file).

Usage: python3 compare_legacy.py FORK.opsf LEGACY_SSP.bin

Both sides must be run on the same data tree, with legacy built with the exact gsig4pi
(see README.md). Exits non-zero if any tolerance is exceeded.
"""
import struct
import sys

import numpy as np

# Tolerances, about twice the residuals measured for MIST + MILES, Chabrier, solar, with
# add_agb_dust_model = 0 (README.md). Populations younger than log t = 7 (Wolf-Rayet and hot
# stars) differ more, for reasons not yet identified, and get their own tier.
TOL_LOG_MASS = 1e-6          # dex, all ages (measured 1.2e-8)
TOL_LOG_LBOL = 1e-6          # dex, all ages (measured 5.6e-8)
LOG_AGE_SPLIT = 7.0
# (median |fork/legacy - 1| over 0.37-2.5 um, per-pixel |r - 1| above 912 A, bolometric |r - 1|)
TOL_OLD = dict(median=5e-4, pixel=2e-3, bol=3e-4)      # measured 2.6e-4, 9.8e-4 (p99), 1.0e-4
TOL_YOUNG = dict(median=1e-2, pixel=3e-2, bol=5e-3)    # measured 4.9e-3, 2.1e-2 (max), 2.9e-3
PIXEL_PERCENTILE_OLD = 99


def read_opsf(path):
    """Reads an OPSF container into {name: ndarray} (last Fortran dimension first)."""
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"OPSF":
            raise ValueError(f"{path}: not an OPSF file")
        f.read(4)
        while True:
            head = f.read(4)
            if not head:
                break
            name = f.read(struct.unpack("<i", head)[0]).decode()
            ndim = struct.unpack("<i", f.read(4))[0]
            dims = struct.unpack(f"<{ndim}i", f.read(4 * ndim))
            data = np.frombuffer(f.read(8 * int(np.prod(dims))), "<f8")
            out[name] = data.reshape(dims[::-1])
    return out


def read_legacy(path):
    """Reads the stream written by ssp_dump.f90."""
    raw = open(path, "rb").read()
    nt, ns, _ = np.frombuffer(raw[:12], np.int32)
    a = np.frombuffer(raw[12:], np.float64)
    t, lam = a[:nt], a[nt:nt + ns]
    mass, lbol = a[nt + ns:2 * nt + ns], a[2 * nt + ns:3 * nt + ns]
    spec = a[3 * nt + ns:].reshape(nt, ns)
    return t, lam, mass, lbol, spec


def main():
    fork = read_opsf(sys.argv[1])
    t, lam, mass, lbol, spec = read_legacy(sys.argv[2])
    if not (np.allclose(lam, fork["wavelengths"]) and np.allclose(t, fork["time_full"])):
        sys.exit("wavelength or age grids differ: were both run on the same data tree?")
    fspec, fmass, flbol = fork["ssp/spec"], fork["ssp/mass"], fork["ssp/lbol"]

    opt = (lam > 3700) & (lam < 2.5e4)
    ion = lam > 912
    failures = []

    def check(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            failures.append(msg)

    dmass = np.abs(np.log10(fmass) - np.log10(mass)).max()
    dlbol = np.abs(flbol - lbol).max()
    check(dmass <= TOL_LOG_MASS, f"max |d log M| = {dmass:.1e} dex (tol {TOL_LOG_MASS:.0e})")
    check(dlbol <= TOL_LOG_LBOL, f"max |d log L_bol| = {dlbol:.1e} dex (tol {TOL_LOG_LBOL:.0e})")

    worst = {tier: dict(median=0.0, pixel=0.0, bol=0.0) for tier in ("old", "young")}
    for i in range(len(t)):
        good = spec[i] > 0
        if not good.any():
            continue
        r = np.full_like(spec[i], np.nan)
        r[good] = fspec[i][good] / spec[i][good]
        dev = np.abs(r - 1)
        tier = "old" if t[i] >= LOG_AGE_SPLIT else "young"
        pixel = np.nanpercentile(dev[ion], PIXEL_PERCENTILE_OLD) if tier == "old" else np.nanmax(dev[ion])
        w = worst[tier]
        w["median"] = max(w["median"], abs(np.nanmedian(r[opt]) - 1))
        w["pixel"] = max(w["pixel"], pixel)
        w["bol"] = max(w["bol"], abs(np.trapezoid(fspec[i], lam) / np.trapezoid(spec[i], lam) - 1))

    for tier, tol, label in (("old", TOL_OLD, f"log t >= {LOG_AGE_SPLIT}"), ("young", TOL_YOUNG, f"log t < {LOG_AGE_SPLIT}")):
        w = worst[tier]
        pix = f"p{PIXEL_PERCENTILE_OLD}" if tier == "old" else "max"
        check(w["median"] <= tol["median"],
              f"{label}: optical/NIR median |r-1| = {w['median']:.1e} (tol {tol['median']:.0e})")
        check(w["pixel"] <= tol["pixel"],
              f"{label}: per-pixel {pix} |r-1| = {w['pixel']:.1e} (tol {tol['pixel']:.0e})")
        check(w["bol"] <= tol["bol"],
              f"{label}: bolometric |r-1| = {w['bol']:.1e} (tol {tol['bol']:.0e})")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
