#!/usr/bin/env python3
"""The free-parameter matrix benchmark: per-model cost of a batch as a function of which parameters are free.

Runs tests/benchmark_csp_batch in matrix mode (OPENSPS_BENCH_FREE, OPENSPS_BENCH_DIST; see the header of the program) for

  * the baseline (nothing free beyond the config's own tau, tage, duste_umin, gas_logu),
  * every field of a list alone,
  * pairs of the SSP-level fields,
  * all of a list together,

for each distribution and batch size, and writes a TSV (distribution, free set, size, magnitudes-only and with-spectra
microseconds per model, status). The report compares:

  * a field alone against the baseline (a ratio above --alone-limit is flagged);
  * a pair against the sum of its parts, cost(A+B) / (cost(A) + cost(B) - cost(base)), the super-additivity (above
    --pair-limit is flagged: the combination costs more than its members explain, a cliff);
  * a distribution against the uniform one for the same free set (above --dist-limit is flagged: the cost depends on how
    the values are spread).

Usage:
  bench_matrix.py EXE [--sizes 1000,10000] [--dists uniform,cluster,onecell] [--fields a,b,...] [--pairs a+b,c+d]
                  [--all a,b,c,...] [--reps 2] [--tsv out.tsv] [--report] [--config full] [--env NAME=VALUE ...]
  bench_matrix.py --report-only out.tsv

Exit status 0 always (a flagged cost is a finding, not a failure); a tripwire is a later step.
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

# fields that act in the benchmark's configuration ("full", an exponentially declining SFH) and are not already free there
ALONE = ['dust1', 'dust2', 'dust3', 'dust_index', 'dust1_index', 'frac_nodust', 'dust_tesc', 'duste_gamma', 'duste_qpah',
         'gas_logz', 'frac_obrun', 'fagn', 'agn_tau', 'zred', 'igm_factor',
         'logzsol', 'afe', 'fbhb', 'sbss', 'agb', 'pagb', 'redgb', 'delt', 'dell', 'imf1', 'imf2', 'imf3', 'vdmc', 'mdave',
         'sigma_smooth', 'imf_lower_limit', 'imf_upper_limit']
PAIRS = ['logzsol+imf1', 'logzsol+fbhb', 'imf1+fbhb', 'imf1+delt', 'fbhb+delt', 'logzsol+delt', 'dust2+imf1', 'dust2+logzsol',
         'zred+logzsol', 'agb+fbhb', 'afe+logzsol']
ALL_SET = ['dust1', 'dust2', 'dust_index', 'duste_qpah', 'gas_logz', 'fagn', 'zred', 'igm_factor', 'logzsol', 'imf1', 'imf2',
           'imf3', 'agb', 'pagb', 'redgb']
LINE = re.compile(r'^batch_(\d+)_(mags|spec)_us_per_model\s+([0-9.eE+-]+)')


def run(exe, free, dist, sizes, config, env_extra, n_single=0, dump=None, dump_grad=None):
    env = dict(os.environ)
    env['OPENSPS_BENCH_FREE'] = free if free else '-'
    env['OPENSPS_BENCH_DIST'] = dist
    env['OPENSPS_BENCH_SIZES'] = sizes
    env.update(env_extra)
    if dump:
        env['OPENSPS_BENCH_DUMP_FWD'] = dump
    if dump_grad:
        env['OPENSPS_BENCH_DUMP'] = dump_grad
    nmax = max(int(x) for x in sizes.split(','))
    nmin = min(int(x) for x in sizes.split(','))
    p = subprocess.run([exe, str(nmax), str(n_single), config, str(nmin)], env=env, capture_output=True, text=True)
    out = {'status': 0, 'mags': {}, 'spec': {}}
    for line in p.stdout.splitlines():
        m = LINE.match(line)
        if m:
            out[m.group(2)][int(m.group(1))] = float(m.group(3))
        elif line.startswith('matrix_status'):
            tok = line.split()
            out['status'] = tok[1]
    if p.returncode != 0 and out['status'] == 0:
        out['status'] = 'exit%d' % p.returncode
    return out


def median_runs(exe, free, dist, sizes, config, env_extra, reps, dump=None, dump_grad=None):
    runs = [run(exe, free, dist, sizes, config, env_extra, dump=dump, dump_grad=dump_grad) for _ in range(reps)]
    bad = [r for r in runs if r['status'] != 0]
    if bad:
        return {'status': bad[0]['status'], 'mags': {}, 'spec': {}}
    res = {'status': 0, 'mags': {}, 'spec': {}}
    for kind in ('mags', 'spec'):
        for n in runs[0][kind]:
            res[kind][n] = statistics.median(r[kind][n] for r in runs if n in r[kind])
    return res


def report(rows, alone_limit, pair_limit, dist_limit):
    cost = {}
    status = {}
    for d, free, n, mags, spec, st in rows:
        cost[(d, free, int(n))] = float(mags) if mags != '' else None
        status[(d, free, int(n))] = st
    dists = sorted(set(k[0] for k in cost))
    sizes = sorted(set(k[2] for k in cost))
    frees = sorted(set(k[1] for k in cost), key=lambda f: (f.count('+'), f))
    print('\nmagnitudes-only microseconds per model; ratio to the baseline; * = above the limit')
    for d in dists:
        for n in sizes:
            base = cost.get((d, '-', n))
            print('\n== distribution %s, N = %d (baseline %s)' % (d, n, '%.2f' % base if base else 'n/a'))
            for f in frees:
                if f == '-' or '+' in f or f == 'ALL':
                    continue
                c = cost.get((d, f, n))
                st = status.get((d, f, n), '0')
                if c is None:
                    print('  %-14s status %s' % (f, st))
                else:
                    r = c / base if base else float('nan')
                    print('  %-14s %8.2f   x%.2f %s' % (f, c, r, '*' if r > alone_limit else ''))
            print('  -- pairs: cost(A+B) / (cost(A) + cost(B) - base)')
            for f in frees:
                if '+' not in f:
                    continue
                a, b = f.split('+')
                ca, cb, cab = cost.get((d, a, n)), cost.get((d, b, n)), cost.get((d, f, n))
                if cab is None or ca is None or cb is None or not base:
                    print('  %-22s status %s' % (f, status.get((d, f, n), '?')))
                    continue
                expect = ca + cb - base
                r = cab / expect if expect > 0 else float('nan')
                print('  %-22s %8.2f   expected %.2f  x%.2f %s' % (f, cab, expect, r, '*' if r > pair_limit else ''))
            c = cost.get((d, 'ALL', n))
            if c is not None and base:
                print('  all together: %.2f (x%.2f of the baseline)' % (c, c / base))
    if 'uniform' in dists:
        print('\n== dependence on the distribution (cost / cost with the uniform spread, divided by the same ratio of the '
              'baseline, which no distribution touches: it removes a drift of the clocks between the runs)')
        for d in dists:
            if d == 'uniform':
                continue
            for n in sizes:
                b_d, b_u = cost.get((d, '-', n)), cost.get(('uniform', '-', n))
                corr = (b_d / b_u) if (b_d and b_u) else 1.0
                print('  %-9s N=%-6d baseline ratio x%.2f (the noise floor of this comparison)' % (d, n, corr))
                for f in frees:
                    if f == '-':
                        continue
                    c, u = cost.get((d, f, n)), cost.get((('uniform'), f, n))
                    if c is None or u is None:
                        continue
                    r = c / u / corr
                    if r > dist_limit or r < 1.0 / dist_limit:
                        print('  %-9s N=%-6d %-22s x%.2f *' % (d, n, f, r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exe', nargs='?')
    ap.add_argument('--sizes', default='1000,10000')
    ap.add_argument('--dists', default='uniform')
    ap.add_argument('--fields', default=None, help='comma list (default: the standard set); "none" for none')
    ap.add_argument('--pairs', default=None, help='a+b,c+d (default: the standard pairs); "none" for none')
    ap.add_argument('--all', default=None, help='comma list for the all-together run; "none" to skip')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--config', default='full')
    ap.add_argument('--tsv', default=None)
    ap.add_argument('--dump', default=None, help='directory for the forward dumps (one file per distribution and free set,\n                    the largest size), to compare two commits with compare_forward_dump.py')
    ap.add_argument('--dump-grad', default=None, help='directory for the gradient dumps (params_bar of the largest size, one file\n                    per distribution and free set; the benchmark then also runs the gradients), to compare two commits with\n                    tools/compare_builds/compare_grad_dump.py')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--resume', action='store_true', help='skip the (distribution, free set) rows already in --tsv and append the rest')
    ap.add_argument('--report-only', default=None)
    ap.add_argument('--alone-limit', type=float, default=1.5)
    ap.add_argument('--pair-limit', type=float, default=1.15)
    ap.add_argument('--dist-limit', type=float, default=1.15)
    ap.add_argument('--env', nargs='*', default=[], help='NAME=VALUE passed to the benchmark')
    a = ap.parse_args()

    if a.report_only:
        with open(a.report_only) as fh:
            rows = [tuple(r) for r in csv.reader(fh, delimiter='\t')][1:]
        report(rows, a.alone_limit, a.pair_limit, a.dist_limit)
        return
    if not a.exe:
        ap.error('the benchmark executable is required')
    env_extra = dict(x.split('=', 1) for x in a.env)
    fields = ALONE if a.fields is None else ([] if a.fields == 'none' else a.fields.split(','))
    pairs = PAIRS if a.pairs is None else ([] if a.pairs == 'none' else a.pairs.split(','))
    allset = ALL_SET if a.all is None else ([] if a.all == 'none' else a.all.split(','))
    sizes = [int(x) for x in a.sizes.split(',')]
    rows = []
    done = set()
    header = ['dist', 'free', 'n', 'mags_us_per_model', 'spec_us_per_model', 'status']
    if a.tsv and a.resume and os.path.exists(a.tsv):
        with open(a.tsv) as fh:
            rows = [tuple(r) for r in csv.reader(fh, delimiter='\t')][1:]
        done = set((r[0], r[1]) for r in rows)
    if a.tsv and not (a.resume and os.path.exists(a.tsv)):
        with open(a.tsv, 'w', newline='') as fh:
            csv.writer(fh, delimiter='\t').writerow(header)
    sets = [('-', '')] + [(f, f) for f in fields] + [(p, p.replace('+', ',')) for p in pairs]
    if allset:
        sets.append(('ALL', ','.join(allset)))
    # the distributions of one free set are measured one after the other (set outer): a drift of the device's clocks over the
    # run then does not become a difference between distributions
    for label, free in sets:
        for d in a.dists.split(','):
            if (d, label) in done:
                continue
            dump = None
            if a.dump:
                os.makedirs(a.dump, exist_ok=True)
                dump = os.path.join(a.dump, '%s__%s.bin' % (d, label.replace('+', '_')))
            dump_grad = None
            if a.dump_grad:
                os.makedirs(a.dump_grad, exist_ok=True)
                dump_grad = os.path.join(a.dump_grad, '%s__%s.grad' % (d, label.replace('+', '_')))
            res = median_runs(a.exe, free, d, a.sizes, a.config + ('_grad' if a.dump_grad and '_grad' not in a.config else ''),
                              env_extra, a.reps, dump, dump_grad)
            new_rows = []
            for n in sizes:
                m = res['mags'].get(n)
                s_ = res['spec'].get(n)
                new_rows.append((d, label, str(n), '' if m is None else '%.3f' % m, '' if s_ is None else '%.3f' % s_,
                                 str(res['status'])))
            rows += new_rows
            if a.tsv:
                with open(a.tsv, 'a', newline='') as fh:
                    csv.writer(fh, delimiter='\t').writerows(new_rows)
            print('%-9s %-24s %s' % (d, label, ' '.join('%s' % ('%.2f' % res['mags'][n] if n in res['mags'] else res['status'])
                                                       for n in sizes)), flush=True)
    if a.report or not a.tsv:
        report(rows, a.alone_limit, a.pair_limit, a.dist_limit)


if __name__ == '__main__':
    main()
