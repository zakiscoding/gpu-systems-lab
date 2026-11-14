"""Nightly GPU performance regression check, submitted through GopherGrid.

Run once a night by a systemd timer on the GPU PC (regress/gg-nightly.timer), not by a GitHub
runner: a self-hosted runner on a public repo would let any pull request run code on this PC.

1. Submit a GopherGrid `benchmark` job (suite `matmul`: gpu-systems-lab's smoke test, 8192^2, every
   precision) to whichever scheduler answers. It needs 1 GPU and 900 MB, so it runs next to the live
   vllm-chat without stopping it.
2. Wait for it, then append one record per run to a JSON-lines history (TFLOPS per precision, the
   job's GPU state, attempt, wall time).
3. For each precision, compare with the median of the previous 7 successful runs. More than 5%
   below that baseline counts as a regression.
4. Write Prometheus gauges for node-exporter's textfile collector: TFLOPS, baseline, the regression
   flag, last-success time. Grafana ("GPU nightly regression") charts them, and the
   `GpuNightlyRegression` alert fires on the flag.

    python3 regress/nightly.py                         # what the timer runs
    python3 regress/nightly.py --label contention-test # tag a run (kept in the history)
"""

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

SCHEDULERS = os.environ.get("GG_SCHEDULERS", "http://127.0.0.1:8091,http://100.91.113.98:8090").split(",")
HISTORY = Path(os.environ.get("GG_NIGHTLY_HISTORY", "/var/lib/gg-nightly/history.jsonl"))
PROM_FILE = Path(os.environ.get("GG_NIGHTLY_PROM", "/var/lib/node_exporter/textfile/gg_nightly.prom"))
WINDOW = 7          # baseline = median of this many previous successful runs
THRESHOLD = 0.05    # flag a drop of more than 5%
MIN_BASELINE = 3    # don't judge until there are at least this many previous runs


# ---------------------------------------------------------------- pure logic (unit-tested)

def tflops_by_precision(job):
    """{"float16": 63.6, ...} from a finished matmul benchmark job."""
    rows = ((job.get("result") or {}).get("summary") or {}).get("matmul") or []
    return {r["precision"]: r["tflops"] for r in rows if "precision" in r and "tflops" in r}


def evaluate(history, current):
    """Per precision: baseline (median of the last WINDOW successful runs before this one), the
    ratio current/baseline, and whether that is a regression. Runs labelled as tests are not used
    as baseline. Returns {precision: {"value", "baseline", "ratio", "regression", "n_baseline"}}."""
    prior = [h for h in history if h.get("ok") and not h.get("label")][-WINDOW:]
    out = {}
    for prec, value in current.items():
        vals = [h["tflops"][prec] for h in prior if prec in h.get("tflops", {})]
        if len(vals) < MIN_BASELINE:
            out[prec] = {"value": value, "baseline": None, "ratio": None, "regression": False, "n_baseline": len(vals)}
            continue
        base = statistics.median(vals)
        ratio = value / base
        out[prec] = {"value": value, "baseline": base, "ratio": round(ratio, 4),
                     "regression": ratio < 1 - THRESHOLD, "n_baseline": len(vals)}
    return out


def prometheus_text(verdict, ok, ts, label=""):
    """Textfile-collector exposition for one run."""
    lines = ["# HELP gg_nightly_tflops Nightly GopherGrid matmul benchmark, TFLOPS by precision.",
             "# TYPE gg_nightly_tflops gauge",
             "# HELP gg_nightly_baseline_tflops Median of the previous 7 successful nightly runs.",
             "# TYPE gg_nightly_baseline_tflops gauge",
             "# HELP gg_nightly_regression 1 if this run is more than 5% below its baseline.",
             "# TYPE gg_nightly_regression gauge"]
    lab = f',label="{label}"' if label else ""
    for prec, v in sorted(verdict.items()):
        lines.append(f'gg_nightly_tflops{{precision="{prec}"{lab}}} {v["value"]}')
        if v["baseline"] is not None:
            lines.append(f'gg_nightly_baseline_tflops{{precision="{prec}"{lab}}} {v["baseline"]}')
        lines.append(f'gg_nightly_regression{{precision="{prec}"{lab}}} {1 if v["regression"] else 0}')
    lines += ["# HELP gg_nightly_job_ok 1 if the last nightly job succeeded.", "# TYPE gg_nightly_job_ok gauge",
              f"gg_nightly_job_ok {1 if ok else 0}",
              "# HELP gg_nightly_last_run_timestamp_seconds Unix time of the last nightly run.",
              "# TYPE gg_nightly_last_run_timestamp_seconds gauge", f"gg_nightly_last_run_timestamp_seconds {ts:.0f}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- I/O

def api(base, method, path, body=None):
    req = urllib.request.Request(base + path, method=method, headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def submit_and_wait(timeout_s=1200):
    spec = {"type": "benchmark", "payload": {"suite": "matmul", "size": 8192, "iters": 50, "label": "nightly"},
            "timeout_ms": 900_000, "max_attempts": 2,
            "resources": {"cpu_millis": 2000, "memory_mb": 4096, "gpus": 1, "gpu_memory_mb": 900}}
    last = None
    for base in SCHEDULERS:          # any scheduler accepts submissions (shared Postgres)
        try:
            job = api(base, "POST", "/v1/jobs", spec)
            break
        except OSError as e:
            last = e
    else:
        raise RuntimeError(f"no scheduler answered: {last}")
    t0 = time.time()
    while job["status"] in ("queued", "running") and time.time() - t0 < timeout_s:
        time.sleep(10)
        job = api(base, "GET", f"/v1/jobs/{job['id']}")
    job["_scheduler"] = base
    job["_wall_s"] = round(time.time() - t0, 1)
    return job


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)       # node-exporter must never read a half-written file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="tag a test run; tagged runs never become baseline")
    args = ap.parse_args()
    history = [json.loads(l) for l in HISTORY.read_text().splitlines() if l.strip()] if HISTORY.exists() else []
    ts = time.time()
    try:
        job = submit_and_wait()
        ok = job["status"] == "succeeded"
    except Exception as e:  # noqa: BLE001 - a failed night is a data point too
        job, ok = {"status": "error", "last_error": str(e)[:300]}, False
    tflops = tflops_by_precision(job) if ok else {}
    verdict = evaluate(history, tflops)
    rec = {"ts": ts, "ok": ok, "label": args.label, "job_id": job.get("id"), "status": job.get("status"),
           "attempt": job.get("attempt"), "scheduler": job.get("_scheduler"), "wall_s": job.get("_wall_s"),
           "tflops": tflops, "verdict": verdict, "error": job.get("last_error")}
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    write_atomic(PROM_FILE, prometheus_text(verdict, ok, ts, args.label))
    flagged = [p for p, v in verdict.items() if v["regression"]]
    print(json.dumps({"ok": ok, "tflops": tflops, "regressions": flagged,
                      "ratios": {p: v["ratio"] for p, v in verdict.items()}}))


if __name__ == "__main__":
    main()
