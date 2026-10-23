"""Compare local and Modal benchmark results. Local only: makes no Modal calls.

Reads results/*.json (smoke_<host>_*.json from local runs, modal_<gpu>_*.json
from modal/smoke.py, modal_ddp_*.json from modal/ddp_nccl.py) and prints:
  - matmul TFLOPS per precision, H2D bandwidth, cold start, estimated $/run
  - a DDP/NCCL summary if any modal_ddp_*.json exist

Usage: python modal/compare.py [--results results] [--latest]
"""

import argparse
import json
from pathlib import Path

# APPROXIMATE Modal GPU prices in USD per GPU-hour. Verify at https://modal.com/pricing
# before trusting any $ figure below; CPU/RAM charges (small) are not included.
PRICE_PER_GPU_HOUR_APPROX = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100": 2.10,       # 40 GB
    "A100-80GB": 2.50,
    "H100": 3.95,
}

PRECISIONS = ("bfloat16", "float16", "tf32", "float32")
SHORT = {"bfloat16": "BF16", "float16": "FP16", "tf32": "TF32", "float32": "FP32"}


def load(results_dir: Path) -> tuple[list[tuple[Path, dict]], list[tuple[Path, dict]]]:
    """Read every results/*.json and split into (smoke records, DDP records),
    each a list of (path, record) sorted by filename. Unreadable files are
    skipped with a message; other JSON (e.g. FSDP runs) is ignored."""
    smoke, ddp = [], []
    for path in sorted(results_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {path.name}: {e}")
            continue
        if path.name.startswith("modal_ddp_"):
            ddp.append((path, rec))
        elif "matmul" in rec and (path.name.startswith("smoke_") or path.name.startswith("modal_")):
            smoke.append((path, rec))
    return smoke, ddp


def est_cost(modal_meta: dict) -> float | None:
    """Approximate USD for one run: laptop wall seconds x list price x GPU count.
    None if the GPU type has no price or the record has no wall time."""
    price = PRICE_PER_GPU_HOUR_APPROX.get(modal_meta.get("gpu_request", "").split(":")[0])
    if price is None or modal_meta.get("wall_seconds") is None:
        return None
    # Wall time from the laptop is an upper bound on billed container time.
    return modal_meta["wall_seconds"] / 3600 * price * modal_meta.get("gpu_count", 1)


def keep_latest(rows: list[tuple[Path, dict]], key) -> list[tuple[Path, dict]]:
    """Keep one row per key(path, record), the last one seen (newest, since rows
    are sorted by timestamped filename)."""
    latest = {}
    for path, rec in rows:  # sorted by filename, which ends in a timestamp
        latest[key(path, rec)] = (path, rec)
    return list(latest.values())


def fmt(v, spec: str = "") -> str:
    """format(v, spec), or "-" for a missing value."""
    return "-" if v is None else format(v, spec)


def print_smoke(rows: list[tuple[Path, dict]]) -> None:
    """Print the matmul table (TFLOPS per precision, H2D GiB/s, cold start and
    wall seconds, ~USD per run), then each Modal run's BF16 speed relative to
    the first local run in `rows`."""
    header = (f"{'source':<14}{'GPU':<30}" + "".join(f"{SHORT[p]:>7}" for p in PRECISIONS)
              + f"{'H2D':>7}{'cold s':>8}{'wall s':>8}{'~$/run':>9}")
    print("matmul TFLOPS (8192^2 unless noted), H2D pinned GiB/s")
    print(header)
    print("-" * len(header))
    baseline = None
    for path, rec in rows:
        tf = {m["precision"]: m["tflops"] for m in rec["matmul"]}
        meta = rec.get("modal")
        source = f"modal:{meta['gpu_request']}" if meta else rec["environment"]["host"][:13]
        if meta is None and baseline is None:
            baseline = tf
        cost = est_cost(meta) if meta else None
        sizes = {m["size"] for m in rec["matmul"]}
        gpu = rec["environment"]["gpu"].removeprefix("NVIDIA ")
        if sizes != {8192}:
            gpu += f" @{'/'.join(map(str, sorted(sizes)))}"
        print(f"{source:<14}{gpu[:29]:<30}"
              + "".join(f"{fmt(tf.get(p), '.1f'):>7}" for p in PRECISIONS)
              + f"{fmt(rec.get('transfer', {}).get('h2d_pinned_gib_s'), '.1f'):>7}"
              + f"{fmt(meta and meta.get('cold_start_seconds'), '.1f'):>8}"
              + f"{fmt(meta and meta.get('wall_seconds'), '.1f'):>8}"
              + f"{('$' + format(cost, '.3f')) if cost is not None else 'local':>9}")

    if baseline:
        print("\nBF16 vs first local baseline:")
        for path, rec in rows:
            if rec.get("modal"):
                bf = next((m["tflops"] for m in rec["matmul"] if m["precision"] == "bfloat16"), None)
                if bf and baseline.get("bfloat16"):
                    print(f"  {rec['modal']['gpu_request']:<10} {bf / baseline['bfloat16']:.2f}x")


def print_ddp(rows: list[tuple[Path, dict]]) -> None:
    """Print one row per DDP/NCCL run: peak all-reduce bus bandwidth (GB/s),
    1-GPU vs N-GPU samples/s, speedup, scaling efficiency and ~USD per run."""
    print("\nDDP / NCCL (weak scaling, fixed per-GPU batch)")
    header = (f"{'request':<12}{'GPU':<24}{'NCCL':>8}{'peak busbw':>12}{'1-GPU sps':>11}"
              f"{'N-GPU sps':>11}{'speedup':>9}{'eff':>6}{'~$/run':>9}")
    print(header)
    print("-" * len(header))
    for path, rec in rows:
        meta = rec.get("modal", {})
        peak = max((r["busbw_gb_s"] for r in rec.get("allreduce", [])), default=None)
        cost = est_cost(meta)
        print(f"{meta.get('gpu_request', '?'):<12}{rec['gpu'].removeprefix('NVIDIA ')[:23]:<24}"
              f"{rec.get('nccl', '?'):>8}{fmt(peak, '.1f'):>12}"
              f"{rec['train_1gpu']['samples_per_s']:>11.0f}{rec['train_ngpu']['samples_per_s']:>11.0f}"
              f"{rec['speedup']:>9.2f}{rec['scaling_efficiency']:>6.0%}"
              f"{('$' + format(cost, '.3f')) if cost is not None else '-':>9}")


def main() -> None:
    """CLI: load results/, optionally keep only the newest run per host/GPU,
    and print both tables."""
    default_results = Path(__file__).resolve().parent.parent / "results"
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=default_results)
    p.add_argument("--latest", action="store_true", help="only the newest run per host / GPU request")
    args = p.parse_args()

    smoke, ddp = load(args.results)
    if not smoke and not ddp:
        raise SystemExit(f"no result JSON files in {args.results}")

    if args.latest:
        smoke = keep_latest(smoke, lambda p, r: (r.get("modal") or {}).get("gpu_request") or r["environment"]["host"])
        ddp = keep_latest(ddp, lambda p, r: r.get("modal", {}).get("gpu_request"))

    # local runs first, then Modal runs by BF16 TFLOPS
    smoke.sort(key=lambda pr: (pr[1].get("modal") is not None,
                               -next((m["tflops"] for m in pr[1]["matmul"] if m["precision"] == "bfloat16"), 0)))
    if smoke:
        print_smoke(smoke)
    if ddp:
        print_ddp(ddp)
    print("\n$ figures are APPROXIMATE (laptop wall time x GPU list price, excludes CPU/RAM);"
          " verify at https://modal.com/pricing")


if __name__ == "__main__":
    main()
