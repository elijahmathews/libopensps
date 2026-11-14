#!/usr/bin/env python3
"""Sweep a tuning parameter: run a benchmark under several settings, interleaved, and report medians.

A *series* is a label, an executable and optional environment settings:

    label=EXE[@NAME=VALUE[,NAME=VALUE...]]

so that a run-time parameter is swept with the registry's environment variables
(`OPENSPS_TUNE_<NAME>`, doc/TUNING.md) on one executable, and a compile-time parameter with one
executable per value (each built with that value). The repeats of all series are interleaved, and
the order rotates from one repeat to the next, so that thermal drift and clock ramps do not favour
a series. Every series runs once first and that run is discarded (page faults, JIT, clock ramp).

    sweep.py --config full --sizes 16,64,256 --repeats 5 \\
        --series default=build/benchmark_csp_batch \\
        --series chunk256=build/benchmark_csp_batch@OPENSPS_TUNE_SSP_BATCH_CHUNK=256 \\
        --series chunk8k=build/benchmark_csp_batch@OPENSPS_TUNE_SSP_BATCH_CHUNK=8192 \\
        --out results.json

Output: per metric, the median of each series (microseconds per model for benchmark_csp_batch), its
spread (min to max of the repeats), the ratio to the first series, and which series is fastest.
A difference smaller than the spread is noise, not a result. The JSON written with --out carries
the raw repeats and the machine and toolchain, which a profile entry in doc/TUNING.md must cite.
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time


def parse_series(spec):
    label, _, rest = spec.partition("=")
    if not label or not rest:
        raise SystemExit(f"bad --series '{spec}': expected label=EXE[@NAME=VALUE,...]")
    exe, _, envs = rest.partition("@")
    env = {}
    for item in filter(None, envs.split(",")):
        k, _, v = item.partition("=")
        env[k] = v
    return {"label": label, "exe": os.path.abspath(exe), "env": env}


def run_once(series, args, base_env, timeout):
    env = dict(base_env)
    env.update(series["env"])
    cmd = [series["exe"]] + args
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{series['label']}: exit {p.returncode}\n{p.stdout[-400:]}\n{p.stderr[-400:]}")
    out = {}
    for line in p.stdout.splitlines():
        f = line.split()
        if len(f) == 2:
            try:
                out[f[0]] = float(f[1])
            except ValueError:
                pass
    return out


def machine_info():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "host": platform.node(),
        "kernel": platform.release(),
        "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"),
        "cpus": os.cpu_count(),
        "git": sh("git rev-parse --short HEAD"),
        "flang": sh("flang --version 2>/dev/null | head -1"),
        "gfortran": sh("gfortran --version 2>/dev/null | head -1"),
        "gpu": sh("rocminfo 2>/dev/null | grep -m3 -E 'Marketing Name|gfx' | tr '\\n' ' '"),
    }
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", required=True, help="label=EXE[@NAME=VALUE,...] (the first is the reference)")
    ap.add_argument("--config", default="full", help="benchmark_csp_batch config (default: full)")
    ap.add_argument("--sizes", default="16,64,256,1024", help="batch sizes (OPENSPS_BENCH_SIZES), ascending")
    ap.add_argument("--n-single", type=int, default=0, help="single-call count passed to the benchmark (0: skip)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--metrics", default=r"batch_\d+_(mags|spec)_us_per_model", help="regexp of the metrics to report")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per run")
    ap.add_argument("--env", action="append", default=[], help="NAME=VALUE for every run (e.g. OMP_TARGET_OFFLOAD=MANDATORY)")
    ap.add_argument("--out", help="write the raw results and the machine to this JSON file")
    a = ap.parse_args()

    series = [parse_series(s) for s in a.series]
    base_env = dict(os.environ)
    base_env["OPENSPS_BENCH_SIZES"] = a.sizes
    for e in a.env:
        k, _, v = e.partition("=")
        base_env[k] = v
    nmax = int(a.sizes.split(",")[-1])
    args = [str(nmax), str(a.n_single), a.config]
    pat = re.compile(a.metrics)

    for s in series:      # warm-up, discarded
        run_once(s, args, base_env, a.timeout)
    runs = {s["label"]: [] for s in series}
    k = len(series)
    for r in range(a.repeats):
        for j in range(k):
            s = series[(j + r) % k]            # the order rotates
            runs[s["label"]].append(run_once(s, args, base_env, a.timeout))
            print(f"  repeat {r + 1}/{a.repeats} {s['label']}", file=sys.stderr, flush=True)

    metrics = sorted({m for rs in runs.values() for run in rs for m in run if pat.fullmatch(m)},
                     key=lambda m: (m.split("_")[2], int(m.split("_")[1])) if m.startswith("batch_") else (m, 0))
    ref = series[0]["label"]
    print(f"config={a.config} sizes={a.sizes} repeats={a.repeats}; median [min-max] in us per model; ratio to '{ref}'")
    for m in metrics:
        print(f"\n{m}")
        med = {s["label"]: statistics.median(run[m] for run in runs[s["label"]] if m in run) for s in series}
        best = min(med, key=med.get)
        for s in series:
            vals = [run[m] for run in runs[s["label"]] if m in run]
            flag = "  <- fastest" if s["label"] == best and len(series) > 1 else ""
            print(f"  {s['label']:<24s} {med[s['label']]:10.2f} [{min(vals):9.2f}-{max(vals):9.2f}]  x{med[s['label']] / med[ref]:5.3f}{flag}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"machine": machine_info(), "args": vars(a), "series": series, "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
