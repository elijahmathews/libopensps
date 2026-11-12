#!/usr/bin/env python3
"""Compares legacy_zZZ_evE.bin with opensps_zZZ_evE.bin (see run_afe.sh).

Prints, per [alpha/Fe] node, the largest |difference| / (peak of that age's spectrum) over
100 A - 100 um, and the median and 99.9th percentile of the per-pixel relative difference
where the flux is above 1e-6 of the peak between 912 A and 100 um.
"""
import sys
import numpy as np

AFE = (-0.2, 0.0, 0.2, 0.4, 0.6)


def read(fn):
    b = open(fn, 'rb').read()
    nt, ns = np.frombuffer(b, np.int32, 2, 0)
    a = np.frombuffer(b, np.float64, ns * nt * 5, 8).reshape((ns, nt, 5), order='F')
    return a.transpose(1, 2, 0)          # (age, afe, wavelength)


def main(legacy, opensps, lam_file=None):
    L = read(legacy)
    O = read(opensps)
    ns = L.shape[2]
    lam = np.loadtxt(lam_file) if lam_file else np.arange(ns, dtype=float)
    for ia in range(5):
        l, o = L[:, ia, :], O[:, ia, :]
        pk = l.max(axis=1, keepdims=True)
        d = np.abs(o - l)
        sel = (lam > 100) & (lam < 1e6) if lam_file else np.ones(ns, bool)
        worst = (d / pk)[:, sel].max()
        m = (l > 1e-6 * pk) & ((lam > 912) & (lam < 1e6))[None, :] if lam_file else (l > 1e-6 * pk)
        rel = d[m] / l[m]
        print(f"[a/Fe]={AFE[ia]:+.1f}: max |d|/peak {worst:.1e}; per-pixel median {np.median(rel):.1e}, "
              f"p99.9 {np.percentile(rel, 99.9):.1e}, max {rel.max():.1e}")


if __name__ == '__main__':
    main(*sys.argv[1:4])
