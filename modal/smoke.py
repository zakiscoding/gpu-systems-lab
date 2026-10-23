"""Roadmap step 4: run benchmarks/smoke_test.py unchanged on a Modal GPU.

The repo's benchmarks/ dir is mounted into the container and smoke_test.py is
executed as a subprocess, so the cloud run measures exactly what the local
RTX 5070 run measured. The JSON record comes back to the laptop with a `modal`
block added (requested GPU, cold start, wall time) and is saved to
results/modal_<gpu>_<ts>.json.

Usage (from the repo root, see modal/README.md):
    modal run modal/smoke.py                # default L4
    modal run modal/smoke.py --gpu H100
    MODAL_LAB_GPU=A10G modal run modal/smoke.py
"""

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/lab"

ALLOWED_GPUS = ("T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100")
DEFAULT_GPU = os.environ.get("MODAL_LAB_GPU", "L4")

# Match the local baseline: torch 2.11.0 built against CUDA 12.8.
# uv searches extra indexes first, so torch resolves from the cu128 wheel index
# and numpy from PyPI.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.11.0",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(REPO / "benchmarks", f"{REMOTE_REPO}/benchmarks", ignore=["__pycache__", "*.pyc"])
)

app = modal.App("gpu-lab-smoke", image=image)


@app.function(gpu=DEFAULT_GPU, timeout=15 * 60)
def run_smoke(size: int = 8192, iters: int = 50):
    """Generator: first yield marks 'container is up and running our code'
    (used for cold-start timing on the laptop, no clock-skew issues), the
    second yield is the smoke_test.py JSON record."""
    yield {"event": "started", "container_host": socket.gethostname()}

    out_dir = Path("/tmp/results")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, f"{REMOTE_REPO}/benchmarks/smoke_test.py",
         "--size", str(size), "--iters", str(iters), "--out", str(out_dir)],
        cwd=REMOTE_REPO,
    )
    bench_s = time.perf_counter() - t0

    files = sorted(out_dir.glob("smoke_*.json"))
    if not files:
        raise RuntimeError(f"smoke_test.py wrote no JSON (exit code {proc.returncode})")
    record = json.loads(files[-1].read_text())
    record["modal_remote"] = {"exit_code": proc.returncode, "bench_seconds": round(bench_s, 2)}
    yield {"event": "result", "record": record}


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, size: int = 8192, iters: int = 50):
    if gpu not in ALLOWED_GPUS:
        raise SystemExit(f"--gpu must be one of {ALLOWED_GPUS}, got {gpu!r}")

    fn = run_smoke.with_options(gpu=gpu)
    t_call = time.perf_counter()
    cold_start_s = None
    record = None
    for msg in fn.remote_gen(size=size, iters=iters):
        if msg["event"] == "started":
            cold_start_s = time.perf_counter() - t_call
            print(f"[modal] container up on {gpu} after {cold_start_s:.1f}s")
        elif msg["event"] == "result":
            record = msg["record"]
    wall_s = time.perf_counter() - t_call

    if record is None:
        raise SystemExit("no result returned from Modal")

    record["modal"] = {
        "gpu_request": gpu,
        "gpu_count": 1,
        "cold_start_seconds": round(cold_start_s, 2) if cold_start_s is not None else None,
        "wall_seconds": round(wall_s, 2),
        "modal_client": modal.__version__,
    }

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    path = out / f"modal_{gpu}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))

    bf16 = next((m["tflops"] for m in record["matmul"] if m["precision"] == "bfloat16"), None)
    print(f"[modal] {record['environment']['gpu']}: BF16 {bf16} TFLOPS, "
          f"H2D {record['transfer']['h2d_pinned_gib_s']} GiB/s, "
          f"cold start {record['modal']['cold_start_seconds']}s, wall {wall_s:.1f}s")
    print(f"[modal] wrote {path}")

    if record["modal_remote"]["exit_code"] != 0:
        raise SystemExit("smoke_test.py reported failing correctness checks (see log above)")
