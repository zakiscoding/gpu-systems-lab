"""Roadmap step 5 scaffold: NCCL collectives + DDP scaling on a multi-GPU Modal container.

Inside one container with N GPUs this runs modal/ddp_train.py twice via torchrun:
  1. --nproc-per-node=1 (CUDA_VISIBLE_DEVICES=0) -> single-GPU training baseline
  2. --nproc-per-node=N -> all_reduce bandwidth sweep + DDP training
and reports weak-scaling efficiency = throughput_N / (N * throughput_1).

Saved to results/modal_ddp_<gpu>x<N>_<ts>.json on the laptop.

Usage (from the repo root, see modal/README.md):
    modal run modal/ddp_nccl.py                    # default L4:2
    modal run modal/ddp_nccl.py --gpu A100:4
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REMOTE = "/root/lab"

ALLOWED_TYPES = ("T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100")
DEFAULT_GPU = os.environ.get("MODAL_LAB_DDP_GPU", "L4:2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.11.0",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_file(HERE / "ddp_train.py", f"{REMOTE}/ddp_train.py")
)

app = modal.App("gpu-lab-ddp-nccl", image=image)


def _torchrun(nproc: int, out: str, extra: list[str], env: dict) -> dict:
    """Run ddp_train.py under torchrun with `nproc` local ranks (blocking; raises
    CalledProcessError on failure) and return the JSON record rank 0 wrote to `out`."""
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc-per-node={nproc}", f"{REMOTE}/ddp_train.py", "--out", out, *extra]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=REMOTE)
    return json.loads(Path(out).read_text())


@app.function(gpu=DEFAULT_GPU, timeout=30 * 60)
def run_ddp(train_args: list[str]):
    """Generator run in the container: yield "started" (for cold-start timing), run
    the 1-GPU baseline and the N-GPU run back to back, then yield the combined
    record with speedup = sps_N / sps_1 and scaling_efficiency = speedup / N."""
    yield {"event": "started"}

    import torch  # only needed in the container

    n = torch.cuda.device_count()
    env = {**os.environ, "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN")}
    subprocess.run(["nvidia-smi", "topo", "-m"])  # show NVLink vs PCIe between GPUs

    t0 = time.perf_counter()
    # baseline: one rank pinned to GPU 0; no all_reduce sweep (nothing to reduce across)
    single = _torchrun(1, "/tmp/ddp_1.json", ["--skip-allreduce", *train_args],
                       {**env, "CUDA_VISIBLE_DEVICES": "0"})
    multi = _torchrun(n, f"/tmp/ddp_{n}.json", train_args, env)
    bench_s = time.perf_counter() - t0

    s1 = single["train"]["samples_per_s"]
    sn = multi["train"]["samples_per_s"]
    yield {"event": "result", "record": {
        "gpu": multi["gpu"],
        "gpu_count": n,
        "torch": multi["torch"],
        "nccl": multi["nccl"],
        "allreduce": multi.get("allreduce", []),
        "train_1gpu": single["train"],
        "train_ngpu": multi["train"],
        "speedup": round(sn / s1, 3),
        "scaling_efficiency": round(sn / (n * s1), 3),
        "bench_seconds": round(bench_s, 2),
    }}


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, steps: int = 100, batch: int = 256, max_mib: int = 1024):
    """Local entry point: validate --gpu (TYPE:N, N >= 2), run run_ddp on Modal
    (paid, 30 min timeout), add timing metadata and write
    results/modal_ddp_<type>x<N>_<ts>.json."""
    gpu_type, _, count = gpu.partition(":")
    if gpu_type not in ALLOWED_TYPES or not count.isdigit() or int(count) < 2:
        raise SystemExit(f"--gpu must look like TYPE:N with N>=2 and TYPE in {ALLOWED_TYPES}, got {gpu!r}")

    train_args = ["--steps", str(steps), "--batch", str(batch), "--max-mib", str(max_mib)]
    t_call = time.perf_counter()
    cold_start_s = None
    record = None
    for msg in run_ddp.with_options(gpu=gpu).remote_gen(train_args):
        if msg["event"] == "started":
            cold_start_s = time.perf_counter() - t_call
            print(f"[modal] container up on {gpu} after {cold_start_s:.1f}s")
        else:
            record = msg["record"]
    wall_s = time.perf_counter() - t_call
    if record is None:
        raise SystemExit("no result returned from Modal")

    record["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
    record["modal"] = {
        "gpu_request": gpu,
        "gpu_count": int(count),
        "cold_start_seconds": round(cold_start_s, 2) if cold_start_s is not None else None,
        "wall_seconds": round(wall_s, 2),
        "modal_client": modal.__version__,
    }

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    path = out / f"modal_ddp_{gpu_type}x{count}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))

    peak = max((r["busbw_gb_s"] for r in record["allreduce"]), default=None)
    print(f"[modal] {record['gpu_count']}x {record['gpu']}: peak all_reduce busbw {peak} GB/s, "
          f"DDP speedup {record['speedup']}x, efficiency {record['scaling_efficiency']:.0%}, "
          f"wall {wall_s:.1f}s")
    print(f"[modal] wrote {path}")
