"""Roadmap step 1: local PyTorch CUDA smoke test and matmul benchmark.

Checks the GPU works end to end, then measures matmul throughput across
precisions. Writes a JSON record to results/ so runs can be compared later
(including against Modal in step 4).

Usage: python benchmarks/smoke_test.py [--size 8192] [--iters 50]
"""

import argparse
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def environment() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "vram_gib": round(props.total_memory / 2**30, 2),
        "arch_list": torch.cuda.get_arch_list(),
    }


def correctness_checks() -> dict:
    """Small ops compared against CPU so a broken kernel fails loudly."""
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    cpu = a @ b
    gpu = (a.cuda() @ b.cuda()).cpu()
    max_err = (cpu - gpu).abs().max().item()

    x = torch.arange(1_000_000, device="cuda", dtype=torch.float64)
    reduction_ok = x.sum().item() == 999_999 * 1_000_000 / 2

    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = torch.nn.functional.relu(torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda"))
    autocast_ok = y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all())

    return {
        # TF32 may be enabled for fp32 matmul, so allow a loose tolerance
        "matmul_vs_cpu_max_abs_err": max_err,
        "matmul_ok": max_err < 1e-1,
        "reduction_ok": reduction_ok,
        "autocast_bf16_ok": autocast_ok,
    }


def bench_matmul(size: int, dtype: torch.dtype, iters: int, tf32: bool = False) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)

    for _ in range(5):  # warmup: cuBLAS heuristics, clocks ramp up
        a @ b
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        a @ b
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    tflops = 2 * size**3 / (ms / 1000) / 1e12
    label = "tf32" if tf32 else str(dtype).removeprefix("torch.")
    del a, b
    return {
        "precision": label,
        "size": size,
        "ms_per_iter": round(ms, 3),
        "tflops": round(tflops, 2),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }


def bench_h2d(mib: int = 1024) -> dict:
    """Host-to-device bandwidth with pinned memory (sensitive to PCIe link)."""
    src = torch.empty(mib * 2**20, dtype=torch.uint8).pin_memory()
    src.cuda(non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        src.cuda(non_blocking=True)
    torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / 5
    return {"h2d_pinned_gib_s": round(mib / 1024 / secs, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA not available (torch {torch.__version__}, build CUDA {torch.version.cuda})")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment(),
        "checks": correctness_checks(),
        "matmul": [
            bench_matmul(args.size, torch.float32, args.iters),
            bench_matmul(args.size, torch.float32, args.iters, tf32=True),
            bench_matmul(args.size, torch.float16, args.iters),
            bench_matmul(args.size, torch.bfloat16, args.iters),
        ],
        "transfer": bench_h2d(),
    }

    env = record["environment"]
    print(f"{env['gpu']} (sm_{env['compute_capability'].replace('.', '')}, {env['vram_gib']} GiB) "
          f"| torch {env['torch']} | CUDA {env['cuda_build']}")
    for name, ok in record["checks"].items():
        if name.endswith("_ok"):
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {'precision':<10}{'ms/iter':>10}{'TFLOPS':>10}{'peak MiB':>10}   ({args.size}x{args.size})")
    for r in record["matmul"]:
        print(f"  {r['precision']:<10}{r['ms_per_iter']:>10}{r['tflops']:>10}{r['peak_vram_mib']:>10}")
    print(f"\n  H2D pinned: {record['transfer']['h2d_pinned_gib_s']} GiB/s")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"smoke_{env['host']}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {path}")

    if not all(v for k, v in record["checks"].items() if k.endswith("_ok")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
