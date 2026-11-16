"""Compare two gradient dumps of benchmark_csp_batch (OPENSPS_BENCH_DUMP): params_bar per parameter.

Usage: python3 compare_grad_dump.py REFERENCE TEST. Prints, per parameter, the worst difference relative to the
largest |reference| of that parameter over the models (the scale of the gradient), and the overall worst.
"""
import sys

import numpy as np


def load(f):
    d = open(f, "rb").read()
    nb, n = np.frombuffer(d[:8], dtype=np.int32)
    return np.frombuffer(d[8:], dtype=np.float64).reshape(n, nb).T


r = load(sys.argv[1])
t = load(sys.argv[2])
scale = np.abs(r).max(axis=1)
err = np.abs(t - r).max(axis=1) / np.where(scale > 0, scale, 1.0)
rel = np.abs(t - r) / np.maximum(np.abs(r), 1e-3 * np.where(scale > 0, scale, 1.0)[:, None])
live = scale > 0
print("parameters with a gradient: %d of %d" % (live.sum(), len(scale)))
print("worst |dg| / max|g| over parameters: %.2e (median over parameters %.2e)" % (err.max(), np.median(err[live])))
print("worst elementwise (floor 1e-3 of the scale): %.2e" % rel[live].max())
