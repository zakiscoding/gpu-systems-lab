"""Sharded training on a multi-GPU Modal container: DDP vs FSDP2 memory/throughput, a model
that only fits when sharded, and DCP checkpoint/resume correctness.

All runs execute modal/fsdp_gpt.py through torchrun inside ONE container (default H100:2):

  compare : ~1.3B GPT, same config under DDP, FSDP, FSDP + activation checkpointing
  big     : ~5.6B GPT. DDP needs 16 B/param = 84 GiB of state per GPU (> 80 GB H100) -> OOM;
            FSDP over 2 GPUs needs 42 GiB per GPU -> trains
  resume  : ~0.2B GPT under FSDP. Uninterrupted N steps vs (N/2 steps + DCP save) then
            (resume from DCP to N). Losses at matching steps are compared.
  resume_det: the same with --deterministic (math SDPA, deterministic cuBLAS) -> bit-exact

Saved to results/modal_fsdp_<gpu>x<N>_<ts>.json on the laptop.

Usage (from the repo root):
    modal run modal/fsdp_train.py                              # all suites on H100:2
    modal run modal/fsdp_train.py --suites compare,resume
    modal run modal/fsdp_train.py --tiny                       # seconds-long pipeline check
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
DEFAULT_GPU = os.environ.get("MODAL_LAB_FSDP_GPU", "H100:2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.11.0",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_file(HERE / "fsdp_gpt.py", f"{REMOTE}/fsdp_gpt.py")
)

app = modal.App("gpu-lab-fsdp", image=image)

# GPT shapes. params ~= 12*L*d^2 + (2*vocab + seq)*d
MID = ["--d-model", "2048", "--n-layers", "24", "--n-heads", "16", "--batch", "4", "--seq", "1024"]   # ~1.34B
BIG = ["--d-model", "3072", "--n-layers", "48", "--n-heads", "24", "--batch", "2", "--seq", "1024"]   # ~5.64B
SMALL = ["--d-model", "1024", "--n-layers", "12", "--n-heads", "16", "--batch", "8", "--seq", "1024"]  # ~0.22B
TINY = ["--d-model", "256", "--n-layers", "2", "--n-heads", "4", "--batch", "2", "--seq", "256", "--vocab", "4096"]


def plan(suites: list[str], tiny: bool, steps: int, resume_steps: int) -> list[tuple[str, list[str]]]:
    mid, big, small = (TINY, TINY, TINY) if tiny else (MID, BIG, SMALL)
    s = ["--steps", str(steps)]
    runs = []
    if "compare" in suites:
        runs += [("ddp_mid", ["--mode", "ddp", *mid, *s]),
                 ("fsdp_mid", ["--mode", "fsdp", *mid, *s]),
                 ("fsdp_mid_ac", ["--mode", "fsdp", "--ac", *mid, *s])]
    if "big" in suites:
        runs += [("ddp_big", ["--mode", "ddp", *big, *s]),
                 ("fsdp_big", ["--mode", "fsdp", *big, *s])]
    half = resume_steps // 2
    for prefix, det in (("resume", []), ("resume_det", ["--deterministic"])):
        if prefix in suites:
            ck = f"/tmp/ckpt_{prefix}"
            base = ["--mode", "fsdp", *small, *det]
            runs += [(f"{prefix}_full", [*base, "--steps", str(resume_steps)]),
                     (f"{prefix}_part1", [*base, "--steps", str(half), "--ckpt-dir", ck, "--ckpt-final"]),
                     (f"{prefix}_part2", [*base, "--steps", str(resume_steps), "--ckpt-dir", ck, "--resume"])]
    return runs


def resume_check(full: dict, part1: dict, part2: dict) -> dict:
    half = part2["resumed_from"]["step"]
    common = sorted(int(k) for k in part2["losses"] if k in full["losses"])
    diffs = [abs(full["losses"][str(k)] - part2["losses"][str(k)]) for k in common]
    pre = [abs(full["losses"][k] - part1["losses"][k]) for k in part1["losses"]]
    return {
        "resumed_from_step": half,
        "compared_steps": [common[0], common[-1]] if common else None,
        "max_abs_diff_before_ckpt": max(pre) if pre else None,
        "max_abs_diff_after_resume": max(diffs) if diffs else None,
        "bit_exact_steps": sum(d == 0.0 for d in diffs),
        "n_compared": len(diffs),
        "save_s": part1["ckpt_saves"][-1]["save_s"] if part1.get("ckpt_saves") else None,
        "load_s": part2["resumed_from"]["load_s"],
        "ckpt_gib": round(part1.get("ckpt_bytes", 0) / 2**30, 3),
        "loss_table": [{"step": k, "uninterrupted": full["losses"][str(k)], "resumed": part2["losses"][str(k)]}
                       for k in common],
    }


@app.function(gpu=DEFAULT_GPU, timeout=30 * 60)
def run_suite(runs: list[tuple[str, list[str]]], run_timeout: int):
    yield {"event": "started"}
    import torch

    n = torch.cuda.device_count()
    env = {**os.environ, "NCCL_DEBUG": "WARN", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    out = {}
    for name, args in runs:
        path = f"/tmp/{name}.json"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={n}",
               f"{REMOTE}/fsdp_gpt.py", "--out", path, *args]
        print(f"\n=== {name}\n$ {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        try:
            rc = subprocess.run(cmd, env=env, cwd=REMOTE, timeout=run_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        wall = round(time.perf_counter() - t0, 2)
        rec = json.loads(Path(path).read_text()) if Path(path).exists() else {"status": "failed"}
        rec.update(returncode=rc, wall_s=wall)
        out[name] = rec
        yield {"event": "run", "name": name, "status": rec.get("status"), "wall_s": wall}
    yield {"event": "result", "runs": out, "gpu_count": n}


def _row(name, r):
    s = r.get("state_per_rank", {})
    return (f"  {name:<14}{r.get('status', '?'):>8}{r.get('params_billions', 0):>9}"
            f"{s.get('total_gib', '-'):>12}{r.get('peak_alloc_gib', '-'):>12}{r.get('peak_reserved_gib', '-'):>12}"
            f"{r.get('step_ms', '-'):>10}{r.get('tokens_per_s', '-'):>12}{r.get('mfu', '-'):>8}")


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, suites: str = "compare,big,resume,resume_det", tiny: bool = False,
         steps: int = 15, resume_steps: int = 30, run_timeout: int = 300):
    gpu_type, _, count = gpu.partition(":")
    if not count.isdigit() or int(count) < 2:
        raise SystemExit(f"--gpu must look like TYPE:N with N>=2, got {gpu!r}")
    runs = plan(suites.split(","), tiny, steps, resume_steps)

    t_call = time.perf_counter()
    cold = None
    result = None
    for msg in run_suite.with_options(gpu=gpu).remote_gen(runs, run_timeout):
        if msg["event"] == "started":
            cold = time.perf_counter() - t_call
            print(f"[modal] container up on {gpu} after {cold:.1f}s")
        elif msg["event"] == "run":
            print(f"[modal] {msg['name']}: {msg['status']} in {msg['wall_s']}s")
        else:
            result = msg
    wall = time.perf_counter() - t_call
    if result is None:
        raise SystemExit("no result returned from Modal")

    rr = result["runs"]
    record = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
              "tiny": tiny, "runs": rr,
              "modal": {"gpu_request": gpu, "cold_start_seconds": round(cold, 2) if cold else None,
                        "wall_seconds": round(wall, 2), "modal_client": modal.__version__}}
    for prefix in ("resume", "resume_det"):
        names = [f"{prefix}_{x}" for x in ("full", "part1", "part2")]
        if all(k in rr and rr[k].get("status") == "ok" for k in names):
            record[f"{prefix}_check"] = resume_check(*(rr[k] for k in names))

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    path = out / f"modal_fsdp_{gpu_type}x{count}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))

    print(f"\n  {'run':<14}{'status':>8}{'params B':>9}{'state GiB':>12}{'peak alloc':>12}{'peak resv':>12}"
          f"{'step ms':>10}{'tok/s':>12}{'MFU':>8}")
    for name, r in rr.items():
        print(_row(name, r))
    for prefix in ("resume", "resume_det"):
        if f"{prefix}_check" in record:
            rc = record[f"{prefix}_check"]
            print(f"\n  {prefix}: from step {rc['resumed_from_step']}, max |loss diff| before ckpt (run-to-run) "
                  f"{rc['max_abs_diff_before_ckpt']:.3g}, after resume {rc['max_abs_diff_after_resume']:.3g} "
                  f"({rc['bit_exact_steps']}/{rc['n_compared']} bit-exact), "
                  f"ckpt {rc['ckpt_gib']} GiB, save {rc['save_s']}s, load {rc['load_s']}s")
    print(f"[modal] wall {wall:.1f}s, wrote {path}")
