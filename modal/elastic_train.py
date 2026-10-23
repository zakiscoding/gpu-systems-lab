"""Fault tolerance on a multi-GPU Modal container: torchrun --max-restarts with a deliberately
killed rank, restart from the latest DCP checkpoint, and the cost of the failure.

Inside one container (default H100:2), modal/fsdp_gpt.py (FSDP2) runs via
    torchrun --standalone --nproc-per-node=N --max-restarts=3 ...
  baseline  : no failure, no checkpoints (reference loss curve and step time)
  kill_k<K> : checkpoint every K steps; rank 1 SIGKILLs itself right after step --kill-at-step
              on the first attempt. torchrun tears down the group, restarts all workers, they
              load latest.json and train to the end.

Every rank appends timestamped events (process start, init, ckpt load/save, step, kill) to a
JSON-lines file; the timeline below is reconstructed from it:
  detect+respawn = new process start - kill          (torchrun agent)
  startup        = init done - new process start     (import torch, NCCL init, build FSDP model)
  ckpt load      = DCP load time
  recomputed     = kill step - checkpoint step        (work thrown away)
  time lost      = time the restarted job passes the kill step again - kill time

Saved to results/modal_elastic_<gpu>x<N>_<ts>.json on the laptop.

Usage (from the repo root):
    modal run modal/elastic_train.py                         # K = 10 and 25, kill after step 48 of 60
    modal run modal/elastic_train.py --intervals 5,20 --kill-at 37 --steps 50
    modal run modal/elastic_train.py --tiny
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
DEFAULT_GPU = os.environ.get("MODAL_LAB_ELASTIC_GPU", "H100:2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.11.0",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_file(HERE / "fsdp_gpt.py", f"{REMOTE}/fsdp_gpt.py")
)

app = modal.App("gpu-lab-elastic", image=image)

MODEL = ["--mode", "fsdp", "--d-model", "1024", "--n-layers", "24", "--n-heads", "16",
         "--batch", "8", "--seq", "1024", "--deterministic"]  # ~0.37B params; bit-exact replay after restart
TINY = ["--mode", "fsdp", "--d-model", "256", "--n-layers", "2", "--n-heads", "4",
        "--batch", "2", "--seq", "256", "--vocab", "4096", "--deterministic"]


def analyze(events: list[dict], baseline: dict | None, steps: int) -> dict:
    """Reconstruct the failure timeline of one kill run from its event log.

    `events` are the JSON lines every rank wrote (attempt 0 = before the kill,
    >= 1 = after torchrun's restart). Times are wall-clock seconds; the formulas
    are in the module docstring. The checkpoint resumed from is the last rank-0
    save at or before the kill step. recompute_s_est = recomputed steps x median
    step time (steps <= 3 excluded as warmup). With `baseline` it also reports
    extra wall time and the largest per-step loss difference after resuming
    (0.0 with --deterministic).
    """
    r0 = [e for e in events if e["rank"] == 0]
    kill = next((e for e in events if e["ev"] == "kill"), None)
    if kill is None:
        return {"error": "no kill event"}
    t_kill, k_step = kill["t"], kill["step"]
    saves0 = [e for e in r0 if e["ev"] == "ckpt_save" and e["attempt"] == 0]
    ck_step = max((e["step"] for e in saves0 if e["step"] <= k_step), default=0)
    a1 = [e for e in events if e["attempt"] >= 1]
    starts = [e["t_proc_start"] for e in a1 if e["ev"] == "start"]
    init1 = next((e["t"] for e in r0 if e["ev"] == "init" and e["attempt"] >= 1), None)
    load1 = next((e for e in r0 if e["ev"] == "ckpt_load" and e["attempt"] >= 1), None)
    steps1 = [e for e in r0 if e["ev"] == "step" and e["attempt"] >= 1]
    first1 = steps1[0] if steps1 else None
    reach = next((e["t"] for e in steps1 if e["step"] == k_step), None)
    end = max(e["t"] for e in events if e["ev"] == "step")
    begin = min(e["t"] for e in r0 if e["ev"] == "step" and e["attempt"] == 0)
    step_dts = sorted(e["dt"] for e in r0 if e["ev"] == "step" and e["attempt"] == 0 and e["step"] > 3)
    med_dt = step_dts[len(step_dts) // 2] if step_dts else None

    out = {
        "kill_step": k_step,
        "ckpt_step": ck_step,
        "recomputed_steps": k_step - ck_step,
        "restarts": max(e["attempt"] for e in events),
        "detect_respawn_s": round(min(starts) - t_kill, 2) if starts else None,
        "startup_s": round(init1 - min(starts), 2) if init1 and starts else None,
        "ckpt_load_s": round(load1["secs"], 2) if load1 else 0.0,
        "first_step_after_restart_s": round(first1["t"] - t_kill, 2) if first1 else None,
        "time_lost_s": round(reach - t_kill, 2) if reach else None,
        "recompute_s_est": round((k_step - ck_step) * med_dt, 2) if med_dt else None,
        "step_ms": round(med_dt * 1e3, 1) if med_dt else None,
        "ckpt_saves": len(saves0) + len([e for e in r0 if e["ev"] == "ckpt_save" and e["attempt"] >= 1]),
        "ckpt_save_s_mean": round(sum(e["secs"] for e in saves0) / len(saves0), 2) if saves0 else None,
        "train_wall_s": round(end - begin, 2),
        "finished": any(e["ev"] == "done" for e in a1),
    }
    if baseline:
        out["baseline_train_wall_s"] = baseline["train_wall_s"]
        out["extra_wall_s"] = round(out["train_wall_s"] - baseline["train_wall_s"], 2)
        bl = baseline["losses"]
        after = {e["step"]: e["loss"] for e in steps1}
        diffs = [abs(after[s] - bl[s]) for s in after if s in bl]
        out["max_abs_loss_diff_vs_baseline"] = max(diffs) if diffs else None
        out["final_loss"] = after.get(steps)
        out["baseline_final_loss"] = bl.get(steps)
    return out


@app.function(gpu=DEFAULT_GPU, timeout=30 * 60)
def run_elastic(model_args: list[str], steps: int, kill_at: int, intervals: list[int],
                max_restarts: int, run_timeout: int):
    """Generator run in the container: the baseline, then one kill run per
    checkpoint interval in `intervals`, each a separate torchrun launch with
    --max-restarts. Yields "started", one "run" event per launch, then "result"
    with every run's analysis and raw events. A launch that exceeds
    `run_timeout` seconds is recorded with returncode "timeout"."""
    yield {"event": "started"}
    import torch

    n = torch.cuda.device_count()
    env = {**os.environ, "NCCL_DEBUG": "WARN", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    results = {}
    baseline = None
    for name, k in [("baseline", 0)] + [(f"kill_k{k}", k) for k in intervals]:
        evf, ck = Path(f"/tmp/{name}.events.jsonl"), f"/tmp/ckpt_{name}"
        extra = ["--steps", str(steps), "--events", str(evf)]
        # k = checkpoint interval in steps; 0 = baseline (no checkpoints, no kill)
        if k:
            extra += ["--ckpt-dir", ck, "--ckpt-every", str(k), "--resume", "--kill-at-step", str(kill_at)]
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={n}",
               # poll workers every 0.1 s (default 5 s) so detect+respawn measures restart cost, not polling
               f"--max-restarts={max_restarts}", "--monitor-interval=0.1",
               f"{REMOTE}/fsdp_gpt.py", *model_args, *extra]
        print(f"\n=== {name}\n$ {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        try:
            rc = subprocess.run(cmd, env=env, cwd=REMOTE, timeout=run_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        wall = round(time.perf_counter() - t0, 2)
        events = [json.loads(l) for l in evf.read_text().splitlines()] if evf.exists() else []
        r0steps = [e for e in events if e["rank"] == 0 and e["ev"] == "step"]
        if name == "baseline":
            baseline = {
                "returncode": rc, "torchrun_wall_s": wall,
                "train_wall_s": round(r0steps[-1]["t"] - r0steps[0]["t"], 2) if r0steps else None,
                "losses": {e["step"]: e["loss"] for e in r0steps},
            }
            res = {k2: v for k2, v in baseline.items() if k2 != "losses"}
            res["final_loss"] = baseline["losses"].get(steps)
        else:
            res = {"returncode": rc, "torchrun_wall_s": wall, "ckpt_every": k,
                   **analyze(events, baseline, steps)}
        res["events"] = events
        results[name] = res
        yield {"event": "run", "name": name, "returncode": rc, "wall_s": wall}
    yield {"event": "result", "runs": results, "gpu_count": n,
           "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__)}  # plain str: no torch on the laptop


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, steps: int = 60, kill_at: int = 48, intervals: str = "10,25",
         max_restarts: int = 3, tiny: bool = False, run_timeout: int = 400):
    """Local entry point: validate --gpu (TYPE:N, N >= 2), run the experiment on
    Modal (paid, 30 min timeout), write results/modal_elastic_<type>x<N>_<ts>.json
    and print one row of timeline numbers per checkpoint interval."""
    gpu_type, _, count = gpu.partition(":")
    if not count.isdigit() or int(count) < 2:
        raise SystemExit(f"--gpu must look like TYPE:N with N>=2, got {gpu!r}")
    ks = [int(k) for k in intervals.split(",") if k]

    t_call = time.perf_counter()
    cold = None
    result = None
    for msg in run_elastic.with_options(gpu=gpu).remote_gen(
            TINY if tiny else MODEL, steps, kill_at, ks, max_restarts, run_timeout):
        if msg["event"] == "started":
            cold = time.perf_counter() - t_call
            print(f"[modal] container up on {gpu} after {cold:.1f}s")
        elif msg["event"] == "run":
            print(f"[modal] {msg['name']}: rc={msg['returncode']} in {msg['wall_s']}s")
        else:
            result = msg
    wall = time.perf_counter() - t_call
    if result is None:
        raise SystemExit("no result returned from Modal")

    record = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
              "gpu": result["gpu"], "gpu_count": result["gpu_count"], "torch": result["torch"],
              "tiny": tiny, "steps": steps, "kill_at": kill_at, "max_restarts": max_restarts,
              "model_args": TINY if tiny else MODEL, "runs": result["runs"],
              "modal": {"gpu_request": gpu, "cold_start_seconds": round(cold, 2) if cold else None,
                        "wall_seconds": round(wall, 2), "modal_client": modal.__version__}}
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    path = out / f"modal_elastic_{gpu_type}x{count}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))

    b = result["runs"]["baseline"]
    print(f"\n  baseline: train wall {b['train_wall_s']}s, final loss {b['final_loss']}")
    cols = ["ckpt_every", "ckpt_step", "recomputed_steps", "detect_respawn_s", "startup_s", "ckpt_load_s",
            "time_lost_s", "ckpt_save_s_mean", "ckpt_saves", "extra_wall_s", "max_abs_loss_diff_vs_baseline"]
    print("  " + " | ".join(cols))
    for name, r in result["runs"].items():
        if name != "baseline":
            print("  " + " | ".join(str(r.get(c)) for c in cols))
    print(f"[modal] wall {wall:.1f}s, wrote {path}")
