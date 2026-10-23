"""Targets the router can choose between, with the numbers it decides on.

Speeds are BF16 matmul TFLOPS. Where this repo measured a GPU (results/*.json
from benchmarks/smoke_test.py) the measured value replaces the estimate at
load time; everything else is marked `estimated`. Prices are APPROXIMATE list
prices per GPU-hour and must be checked at https://modal.com/pricing before
trusting a cost decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Target:
    """One place a job can run: the local card or a Modal GPU type, with the
    per-GPU numbers route.evaluate() scores it on."""

    name: str  # "local" or a Modal GPU type ("H100", "L4", ...)
    vram_gb: float  # per GPU
    bf16_tflops: float  # per GPU
    usd_per_gpu_hour: float  # 0 for the local card
    queue_s: float  # typical wait before the job starts
    max_gpus: int
    nvlink: bool  # fast GPU-to-GPU link for multi-GPU jobs
    measured: bool = False  # bf16_tflops came from a results/ JSON
    note: str = ""

    @property
    def local(self) -> bool:
        """True for the home RTX 5070 (free, but only usable when online and idle)."""
        return self.name == "local"


# queue_s: H100 started in ~4 s in every run here; L4:2 once waited 43 min.
DEFAULT_TARGETS = (
    Target("local", 12.0, 67.7, 0.0, 0.0, 1, False, note="RTX 5070 on zak-gpu-node"),
    Target("T4", 16.0, 65.0, 0.59, 30.0, 8, False, note="no native BF16; FP16 figure"),
    Target("L4", 24.0, 121.0, 0.80, 60.0, 8, False, note="2x L4: no P2P, 6.65 GB/s all-reduce"),
    Target("A10G", 24.0, 125.0, 1.10, 30.0, 4, False),
    Target("L40S", 48.0, 362.0, 1.95, 30.0, 8, False),
    Target("A100-80GB", 80.0, 312.0, 2.50, 20.0, 8, True),
    Target("H100", 80.0, 989.0, 3.95, 5.0, 8, True, note="2x H100: 329 GB/s NCCL all-reduce"),
)


def _measured_tflops(results_dir: Path) -> dict[str, float]:
    """Best BF16 TFLOPS per target from smoke-test JSONs (local + modal_*).

    Modal records are keyed by the GPU type they requested ("H100:2" -> "H100");
    a record without a `modal` block counts only if it ran on the RTX 5070.
    """
    best: dict[str, float] = {}
    for f in results_dir.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
            bf16 = next(m["tflops"] for m in rec["matmul"] if m["precision"] == "bfloat16")
        except (ValueError, KeyError, StopIteration, TypeError):
            continue  # DDP records and partial files have no matmul table
        modal = rec.get("modal")
        if modal:
            key = modal.get("gpu_request", "").split(":")[0]
        elif "RTX 5070" in rec.get("environment", {}).get("gpu", ""):
            key = "local"
        else:
            continue
        best[key] = max(best.get(key, 0.0), bf16)
    return best


def load_targets(results_dir: Path | None = None) -> list[Target]:
    """DEFAULT_TARGETS with measured TFLOPS (from results/ by default) swapped in
    and flagged `measured=True`; targets never benchmarked keep their estimate."""
    measured = _measured_tflops(results_dir or REPO / "results")
    return [
        replace(t, bf16_tflops=measured[t.name], measured=True) if t.name in measured else t
        for t in DEFAULT_TARGETS
    ]
