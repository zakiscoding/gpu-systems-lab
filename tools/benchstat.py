"""Repeat a benchmark, record GPU state, and report mean, spread and a 95% confidence interval.

    python3 tools/benchstat.py run --warmup 2 --repeats 10 --out results/rigor/fused.json -- cuda/build/fused
    python3 tools/benchstat.py table results/rigor/*.json        # markdown summary

`run` executes the command `warmup` times (output discarded), then `repeats` times. Each time it
records the GPU's SM clock, memory clock, temperature and power before and after, and parses every
result row the program prints. The cuda/ programs print `name  ms  metric  max_err  ok`, and the
parser takes those columns. For each row it reports n, mean, SD, coefficient of variation, a 95%
CI for the mean (Student's t, n-1 degrees of freedom), and min/max, for both ms and the metric.

Why: a single timing hides run-to-run variance, boost-clock and thermal drift, and the first-run
penalty (JIT, cache, clock ramp). Reporting mean ± CI with the clocks next to it says how much of a
difference between two numbers is real.
"""

import argparse
import glob
import json
import math
import re
import statistics
import subprocess
import sys
import time

# two-sided 95% Student-t critical values, df = 1..30 (beyond 30, 1.96 is within 2%)
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
ROW = re.compile(r"^\s*(?P<name>\S.*?\S)\s+(?P<ms>\d+\.\d+)\s+(?P<metric>\d+(?:\.\d+)?)\s+(?P<err>\S+)\s+(?P<ok>yes|no|-)\s*$")


def t95(df):
    return T95[df - 1] if 1 <= df <= len(T95) else 1.96


def describe(xs):
    """n, mean, sd, cv %, 95% CI half-width and bounds, min, max for a list of numbers."""
    n = len(xs)
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    half = t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4), "cv_pct": round(100 * sd / mean, 2) if mean else None,
            "ci95_half": round(half, 4), "ci95": [round(mean - half, 4), round(mean + half, 4)],
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def parse_rows(text):
    """{row name: (ms, metric)} from a cuda/ program's output table."""
    out = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            out[m["name"].strip()] = (float(m["ms"]), float(m["metric"]))
    return out


def gpu_state():
    q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        sm, mem, temp, power, util = (x.strip() for x in out)
        return {"sm_mhz": int(sm), "mem_mhz": int(mem), "temp_c": int(temp), "power_w": float(power), "util_pct": int(util)}
    except Exception:  # noqa: BLE001 - state is best effort
        return {}


def run(args):
    for _ in range(args.warmup):
        subprocess.run(args.cmd, capture_output=True, text=True)
    runs = []
    for i in range(args.repeats):
        before = gpu_state()
        t0 = time.time()
        p = subprocess.run(args.cmd, capture_output=True, text=True)
        rows = parse_rows(p.stdout)
        runs.append({"i": i, "rc": p.returncode, "wall_s": round(time.time() - t0, 2), "gpu_before": before,
                     "gpu_after": gpu_state(), "rows": rows})
        if p.returncode != 0:
            print(p.stdout[-1500:], p.stderr[-1500:], file=sys.stderr)
    names = list(runs[0]["rows"])
    summary = {}
    for name in names:
        ms = [r["rows"][name][0] for r in runs if name in r["rows"]]
        metric = [r["rows"][name][1] for r in runs if name in r["rows"]]
        summary[name] = {"ms": describe(ms), "metric": describe(metric)}
    clocks = [r["gpu_before"].get("sm_mhz") for r in runs] + [r["gpu_after"].get("sm_mhz") for r in runs]
    clocks = [c for c in clocks if c]
    rec = {"cmd": args.cmd, "warmup": args.warmup, "repeats": args.repeats,
           "all_rc_zero": all(r["rc"] == 0 for r in runs), "summary": summary,
           "sm_clock_mhz_range": [min(clocks), max(clocks)] if clocks else None,
           "temp_c_range": [min(r["gpu_after"].get("temp_c", 0) for r in runs), max(r["gpu_after"].get("temp_c", 0) for r in runs)],
           "runs": runs}
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    for name, s in summary.items():
        m = s["metric"]
        print(f"{name:<30} metric {m['mean']:10.2f} ± {m['ci95_half']:.2f} (95% CI, n={m['n']}, CV {m['cv_pct']}%)")


def table(files):
    print("| program | row | metric mean ± 95% CI | CV | ms mean | n | SM clock (MHz) |")
    print("|---|---|---:|---:|---:|---:|---|")
    for f in sorted(files):
        rec = json.load(open(f))
        prog = rec["cmd"][-1].split("/")[-1] if len(rec["cmd"]) == 1 else rec["cmd"][0].split("/")[-1]
        clk = rec.get("sm_clock_mhz_range") or ["?", "?"]
        for name, s in rec["summary"].items():
            m, ms = s["metric"], s["ms"]
            print(f"| {prog} | {name} | {m['mean']:,.1f} ± {m['ci95_half']:,.1f} | {m['cv_pct']}% | {ms['mean']:.3f} | {m['n']} | {clk[0]}–{clk[1]} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--repeats", type=int, default=10)
    r.add_argument("--out", required=True)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.mode == "run":
        a.cmd = [c for c in a.cmd if c != "--"]
        run(a)
    else:
        table([f for pat in a.files for f in glob.glob(pat)])


if __name__ == "__main__":
    main()
