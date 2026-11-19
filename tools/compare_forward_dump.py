#!/usr/bin/env python3
"""Compares forward dumps of the batched model (benchmark_csp_batch with OPENSPS_BENCH_DUMP_FWD=<file>): magnitudes and spectra.

  compare_forward_dump.py A B [--tol 0]          two files
  compare_forward_dump.py DIR_A DIR_B [--tol 0]  every file of the same name in the two directories

--tol is the largest allowed relative difference (against the larger of the two values, floored at the smallest normal
number); 0, the default, requires identical bits, which is what a refactor must give on the same build and device.
tools/bench_matrix.py --dump DIR writes one file per free set, distribution and size. Exit status 1 on any difference
beyond the tolerance, a missing file or a different shape.
"""
import argparse
import os
import struct
import sys

try:
    import numpy as np
except ImportError:      # the comparison is exact without it; the tolerance needs it
    np = None


def read(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    nb, n, ns = struct.unpack('<3i', data[:12])
    body = data[12:]
    cnt_m, cnt_s = nb * n, ns * n
    if len(body) != 8 * (cnt_m + cnt_s):
        raise ValueError('%s: size does not match its header (%d x %d, %d)' % (path, nb, n, ns))
    return (nb, n, ns), body[:8 * cnt_m], body[8 * cnt_m:]


def compare(a, b, tol):
    ra, rb = read(a), read(b)
    if ra[0] != rb[0]:
        return 'shape %s against %s' % (ra[0], rb[0])
    problems = []
    for label, xa, xb in (('magnitudes', ra[1], rb[1]), ('spectra', ra[2], rb[2])):
        if xa == xb:
            continue
        if tol <= 0.0 or np is None:
            diff = sum(1 for i in range(0, len(xa), 8) if xa[i:i + 8] != xb[i:i + 8])
            problems.append('%s: %d values differ bit for bit' % (label, diff))
            continue
        va = np.frombuffer(xa, dtype='<f8')
        vb = np.frombuffer(xb, dtype='<f8')
        scale = np.maximum(np.maximum(np.abs(va), np.abs(vb)), np.finfo(float).tiny)
        worst = float(np.max(np.abs(va - vb) / scale))
        if worst > tol:
            problems.append('%s: largest relative difference %.3g > %.3g' % (label, worst, tol))
    return '; '.join(problems)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--tol', type=float, default=0.0)
    args = ap.parse_args()
    pairs = []
    if os.path.isdir(args.a) and os.path.isdir(args.b):
        names = sorted(set(os.listdir(args.a)) | set(os.listdir(args.b)))
        for nm in names:
            pairs.append((nm, os.path.join(args.a, nm), os.path.join(args.b, nm)))
    else:
        pairs.append((os.path.basename(args.a), args.a, args.b))
    bad = 0
    for nm, pa, pb in pairs:
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print('%s: missing in one of the two' % nm)
            bad += 1
            continue
        msg = compare(pa, pb, args.tol)
        if msg:
            print('%s: %s' % (nm, msg))
            bad += 1
    print('compared %d file(s), %d differ%s' % (len(pairs), bad, '' if args.tol > 0 else ' (bit for bit)'))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
