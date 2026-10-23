"""Run an arbitrary repo command on a Modal GPU (the router's cloud executor).

Same image as modal/smoke.py (torch 2.11.0+cu128) with benchmarks/ mounted,
so any job that runs in the local .venv runs here unchanged. `python` at the
start of the command is replaced with the container's interpreter.

Usage (normally called by `python -m router run`):
    modal run modal/run_job.py --gpu H100 --command "python benchmarks/smoke_test.py --size 4096"
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/lab"
# the decorator's GPU; main() overrides it per call with .with_options(gpu=...)
DEFAULT_GPU = os.environ.get("MODAL_LAB_JOB_GPU", "L4")

# identical to modal/smoke.py's image, so its build cache is reused
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.11.0", "numpy", extra_index_url="https://download.pytorch.org/whl/cu128")
    .add_local_dir(REPO / "benchmarks", f"{REMOTE_REPO}/benchmarks", ignore=["__pycache__", "*.pyc"])
)

app = modal.App("gpu-lab-job", image=image)


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60)
def run_command(command: str) -> dict:
    """Run `command` from the mounted repo inside the container and return
    {exit, seconds, output}. Output is stdout+stderr, truncated to the last 20,000
    characters to keep the return payload small. Billed for the whole call (1 h max)."""
    argv = shlex.split(command)
    if argv and argv[0] == "python":
        argv[0] = sys.executable
    t0 = time.perf_counter()
    p = subprocess.run(argv, cwd=REMOTE_REPO, capture_output=True, text=True)
    return {"exit": p.returncode, "seconds": round(time.perf_counter() - t0, 2),
            "output": (p.stdout + p.stderr)[-20000:]}


@app.local_entrypoint()
def main(command: str, gpu: str = DEFAULT_GPU):
    """Local entry point: launch `command` on one Modal `gpu` (e.g. "H100", "L4:2"),
    print its output and timings, and exit with the command's exit code. Costs money."""
    t0 = time.perf_counter()
    r = run_command.with_options(gpu=gpu).remote(command)
    print(r["output"])
    print(f"[modal] {gpu}: exit {r['exit']}, command {r['seconds']}s, wall {time.perf_counter() - t0:.1f}s")
    if r["exit"] != 0:
        raise SystemExit(r["exit"])
