"""Compare gradient dumps of benchmark_csp_batch (OPENSPS_BENCH_DUMP): params_bar per parameter.

  compare_grad_dump.py REFERENCE TEST              two files: per parameter the worst difference relative to the largest
                                                   |reference| of that parameter over the models (the scale of the gradient),
                                                   and the overall worst
  compare_grad_dump.py DIR_A DIR_B [--tol 0]       every .grad file of the same name in the two directories (what
                                                   tools/bench_matrix.py --dump-grad writes): exit status 1 on a difference
                                                   beyond --tol (largest |difference| relative to the scale of its parameter;
                                                   0, the default, requires identical bits, which a refactor must give on the
                                                   same build and device), a missing file or a different shape
"""
import argparse
import os
import sys

import numpy as np


def load(f):
    d = open(f, "rb").read()
    nb, n = np.frombuffer(d[:8], dtype=np.int32)
    return np.frombuffer(d[8:], dtype=np.float64).reshape(n, nb).T


def stats(r, t):
    scale = np.abs(r).max(axis=1)
    err = np.abs(t - r).max(axis=1) / np.where(scale > 0, scale, 1.0)
    rel = np.abs(t - r) / np.maximum(np.abs(r), 1e-3 * np.where(scale > 0, scale, 1.0)[:, None])
    live = scale > 0
    print("parameters with a gradient: %d of %d" % (live.sum(), len(scale)))
    print("worst |dg| / max|g| over parameters: %.2e (median over parameters %.2e)" % (err.max(), np.median(err[live])))
    print("worst elementwise (floor 1e-3 of the scale): %.2e" % rel[live].max())


def compare_files(a, b, tol):
    """True if the two dumps agree within tol (bit for bit for 0); prints what differs."""
    r, t = load(a), load(b)
    if r.shape != t.shape:
        print("%s: shape %s against %s" % (os.path.basename(a), r.shape, t.shape))
        return False
    if tol == 0.0:
        both_nan = np.isnan(r) & np.isnan(t)
        bad = ~((r == t) | both_nan)
        if bad.any():
            print("%s: %d of %d values differ bit for bit (parameters %s)" %
                  (os.path.basename(a), bad.sum(), bad.size, sorted(set(np.nonzero(bad)[0].tolist()))[:12]))
            return False
        return True
    scale = np.abs(r).max(axis=1)
    err = np.abs(t - r).max(axis=1) / np.where(scale > 0, scale, 1.0)
    if err.max() > tol:
        print("%s: largest difference %.2e of the scale of its parameter (parameter %d) > %g" %
              (os.path.basename(a), err.max(), int(err.argmax()), tol))
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.0)
    args = ap.parse_args()
    if os.path.isdir(args.a) and os.path.isdir(args.b):
        names = sorted(f for f in os.listdir(args.a) if f.endswith(".grad"))
        names_b = sorted(f for f in os.listdir(args.b) if f.endswith(".grad"))
        ok = True
        for f in sorted(set(names) | set(names_b)):
            if f not in names or f not in names_b:
                print("%s: only in %s" % (f, "A" if f in names else "B"))
                ok = False
            elif not compare_files(os.path.join(args.a, f), os.path.join(args.b, f), args.tol):
                ok = False
        bad = 0 if ok else 1
        print("compared %d file(s)%s" % (len(set(names) & set(names_b)),
                                         ", all agree" + (" bit for bit" if args.tol == 0.0 else " to %g" % args.tol)
                                         if ok else ", some differ or are missing"))
        sys.exit(bad)
    stats(load(args.a), load(args.b))


main()
