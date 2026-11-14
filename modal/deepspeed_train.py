"""DeepSpeed ZeRO 1/2/3 next to DDP and FSDP2 on a 2x H100 Modal container.

Same GPT model, data, optimizer and timing as modal/fsdp_train.py. In ONE container:

  mid : ~1.34B GPT (batch 4 x 1024 per GPU): DDP, FSDP2, ZeRO-1, ZeRO-2, ZeRO-3
  big : ~5.64B GPT (batch 2 x 1024 per GPU): FSDP2, ZeRO-1, ZeRO-2, ZeRO-3
        (DDP needs 16 B/param = 84 GiB per GPU and OOMs, measured in fsdp_train.py)

DDP/FSDP2 rows run modal/fsdp_gpt.py, ZeRO rows run modal/deepspeed_gpt.py, both via torchrun.
Saved to results/modal_deepspeed_<gpu>x<N>_<ts>.json on the laptop.

Usage (from the repo root):
    modal run modal/deepspeed_train.py                 # mid + big, H100:2, ~12 min (~$2)
    modal run modal/deepspeed_train.py --suites mid
    modal run modal/deepspeed_train.py --tiny          # pipeline check
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
DEFAULT_GPU = os.environ.get("MODAL_LAB_DS_GPU", "H100:2")
DEEPSPEED_VERSION = "0.20.0"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.11.0", "numpy", "setuptools", "wheel", "ninja", "packaging",
                    extra_index_url="https://download.pytorch.org/whl/cu128")
    # second step: DeepSpeed's setup.py imports torch. DS_BUILD_OPS=0: no CUDA ops prebuilt or JIT'd
    .env({"DS_BUILD_OPS": "0"})
    .uv_pip_install(f"deepspeed=={DEEPSPEED_VERSION}", extra_options="--no-build-isolation")
    .add_local_file(HERE / "fsdp_gpt.py", f"{REMOTE}/fsdp_gpt.py")
    .add_local_file(HERE / "deepspeed_gpt.py", f"{REMOTE}/deepspeed_gpt.py")
)

app = modal.App("gpu-lab-deepspeed", image=image)

MID = ["--d-model", "2048", "--n-layers", "24", "--n-heads", "16", "--batch", "4", "--seq", "1024"]   # ~1.34B
BIG = ["--d-model", "3072", "--n-layers", "48", "--n-heads", "24", "--batch", "2", "--seq", "1024"]   # ~5.64B
TINY = ["--d-model", "256", "--n-layers", "2", "--n-heads", "4", "--batch", "2", "--seq", "256", "--vocab", "4096"]


def plan(suites: list[str], tiny: bool, steps: int, phase_timing: bool = False,
         zero_only: bool = False) -> list[tuple[str, str, list[str]]]:
    """(run name, worker script, worker args) in run order. --phase-timing: ZeRO runs time
    forward/backward/step separately (extra syncs). --zero-only: skip the DDP/FSDP rows."""
    extra = ["--phase-timing"] if phase_timing else []
    s = ["--steps", str(steps)]
    runs = []
    for suite, shape, with_ddp in (("mid", MID, True), ("big", BIG, False)):
        if suite not in suites:
            continue
        shape = TINY if tiny else shape
        if with_ddp and not zero_only:
            runs.append((f"ddp_{suite}", "fsdp_gpt.py", ["--mode", "ddp", *shape, *s]))
        if not zero_only:
            runs.append((f"fsdp_{suite}", "fsdp_gpt.py", ["--mode", "fsdp", *shape, *s]))
        for st in (1, 2, 3):
            runs.append((f"zero{st}_{suite}", "deepspeed_gpt.py", ["--stage", str(st), *shape, *s, *extra]))
    return runs


@app.function(gpu=DEFAULT_GPU, timeout=40 * 60)
def run_suite(runs: list[tuple[str, str, list[str]]], run_timeout: int):
    """One torchrun launch per run on all GPUs in the container; yields progress then the records."""
    yield {"event": "started"}
    import torch

    n = torch.cuda.device_count()
    env = {**os.environ, "NCCL_DEBUG": "WARN", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "PYTHONPATH": REMOTE}
    out = {}
    for name, script, args in runs:
        path = f"/tmp/{name}.json"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={n}",
               f"{REMOTE}/{script}", "--out", path, *args]
        print(f"\n=== {name}\n$ {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        try:
            rc = subprocess.run(cmd, env=env, cwd=REMOTE, timeout=run_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        wall = round(time.perf_counter() - t0, 2)
        rec = json.loads(Path(path).read_text()) if Path(path).exists() else {"status": "failed"}
        rec.update(returncode=rc, wall_s=wall)
        rec.pop("ckpt_saves", None)
        out[name] = rec
        yield {"event": "run", "name": name, "status": rec.get("status"), "wall_s": wall}
    yield {"event": "result", "runs": out, "gpu_count": n,
           "versions": subprocess.run([sys.executable, "-c", "import torch,deepspeed;"
                                       "print(torch.__version__, deepspeed.__version__, torch.cuda.nccl.version())"],
                                      capture_output=True, text=True).stdout.strip()}


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, suites: str = "mid,big", tiny: bool = False, steps: int = 15,
         run_timeout: int = 420, phase_timing: bool = False, zero_only: bool = False):
    """Run the plan on Modal (paid), write results/modal_deepspeed_<type>x<N>_<ts>.json, print a table."""
    gpu_type, _, count = gpu.partition(":")
    runs = plan(suites.split(","), tiny, steps, phase_timing, zero_only)
    t_call, cold, result = time.perf_counter(), None, None
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
    record = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "tiny": tiny,
              "versions": result["versions"], "runs": result["runs"],
              "modal": {"gpu_request": gpu, "cold_start_seconds": round(cold, 2) if cold else None,
                        "wall_seconds": round(wall, 2), "modal_client": modal.__version__}}
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    tag = "_phases" if phase_timing else ""
    path = out / f"modal_deepspeed{tag}_{gpu_type}x{count}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\n  {'run':<12}{'status':>8}{'params B':>10}{'peak alloc':>12}{'step ms':>10}{'tok/s':>10}{'MFU':>8}")
    for name, r in result["runs"].items():
        print(f"  {name:<12}{r.get('status', '?'):>8}{r.get('params_billions', '-'):>10}{r.get('peak_alloc_gib', '-'):>12}"
              f"{r.get('step_ms', '-'):>10}{r.get('tokens_per_s', '-'):>10}{r.get('mfu', '-'):>8}")
    for name, r in result["runs"].items():
        if r.get("phase_ms_median"):
            print(f"  {name:<12} phases ms: {r['phase_ms_median']}")
    print(f"versions: {result['versions']}\n[modal] wall {wall:.1f}s, wrote {path}")
