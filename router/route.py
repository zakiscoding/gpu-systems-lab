"""Pick where a GPU job should run: the local RTX 5070 or a Modal GPU.

Rules, in order:
  1. A target must fit the job: per-GPU VRAM (with headroom), GPU count, and
     for multi-GPU jobs a fast GPU-to-GPU link (2x L4 trained 3x *slower* than
     1 GPU in this repo, 2x H100 over NVLink ran 1.57x faster).
  2. It must meet the deadline (queue + estimated runtime) and the cost cap.
  3. Among feasible targets: the local card wins when it is online and idle
     (it's free); otherwise the cheapest, then the fastest.

Runtime is estimated from the job's measured runtime on the RTX 5070, scaled
by BF16 throughput (a compute-bound assumption; memory-bound jobs scale less).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import Target
from .probe import LocalState

VRAM_HEADROOM = 0.9  # leave 10% for the CUDA context, allocator, fragmentation


@dataclass(frozen=True)
class Job:
    name: str
    vram_gb: float  # per GPU
    local_minutes: float  # runtime on the RTX 5070 (or equivalent work)
    gpus: int = 1
    max_usd: float = float("inf")
    deadline_min: float = float("inf")
    command: str = ""  # what to run, relative to the repo root


@dataclass
class Option:
    target: Target
    runtime_min: float
    total_min: float  # queue + runtime
    cost_usd: float
    feasible: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class Decision:
    job: Job
    choice: Option | None
    options: list[Option]
    local: LocalState


def evaluate(job: Job, target: Target, local: LocalState, local_tflops: float) -> Option:
    reasons = []
    runtime = job.local_minutes * local_tflops / target.bf16_tflops
    if job.gpus > 1:
        # data-parallel efficiency measured here: 2x H100 (NVLink) 78%, 2x L4 (PCIe) 17%
        eff = 0.78 if target.nvlink else 0.17
        runtime = runtime / (job.gpus * eff)
    total = target.queue_s / 60 + runtime
    cost = target.usd_per_gpu_hour * job.gpus * runtime / 60  # Modal bills running time, not the queue

    vram = target.vram_gb
    if target.local:
        if not local.online:
            reasons.append(f"offline ({local.error or 'unreachable'})")
        else:
            vram = min(vram, local.vram_free_gb)  # the desktop and other jobs hold some
            if local.busy:
                reasons.append(f"busy ({local.util_pct:.0f}% util, {local.compute_procs} CUDA procs)")
    if job.vram_gb > vram * VRAM_HEADROOM:
        reasons.append(f"needs {job.vram_gb:g} GB, has {vram:.1f} GB")
    if job.gpus > target.max_gpus:
        reasons.append(f"needs {job.gpus} GPUs, max {target.max_gpus}")
    if job.gpus > 1 and not target.nvlink:
        reasons.append("no NVLink for multi-GPU")
    if total > job.deadline_min:
        reasons.append(f"{total:.1f} min > deadline {job.deadline_min:g} min")
    if cost > job.max_usd:
        reasons.append(f"${cost:.2f} > cap ${job.max_usd:g}")
    return Option(target, runtime, total, cost, not reasons, reasons)


def decide(job: Job, targets: list[Target], local: LocalState) -> Decision:
    local_tflops = next(t.bf16_tflops for t in targets if t.local)
    options = [evaluate(job, t, local, local_tflops) for t in targets]
    feasible = [o for o in options if o.feasible]
    # free local first, then cheapest, then fastest to finish
    feasible.sort(key=lambda o: (not o.target.local, round(o.cost_usd, 4), o.total_min))
    return Decision(job, feasible[0] if feasible else None, options, local)
