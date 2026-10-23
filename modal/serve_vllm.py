"""Inference serving on Modal: the serving/ vLLM benchmark on 1x H100.

Server and client run in one container: serving/bench_sweep.py --launch starts
`vllm serve` on 127.0.0.1 inside the sandbox, waits for /health, runs the same
`vllm bench serve` concurrency sweep as the RTX 5070 run, and stops the server.
Summaries plus raw per-concurrency JSON come back to the laptop and are saved
to results/serving/modal_<label>_*.json.

HF weights are cached in a Modal Volume (gpu-lab-hf-cache) so a rerun does not
download them again.

Usage (from the repo root, see serving/README.md):
    modal run modal/serve_vllm.py                                   # Qwen2.5-1.5B
    modal run modal/serve_vllm.py --models Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct
    modal run modal/serve_vllm.py --concurrency 1,4,16,64,256
"""

import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/lab"

# Keep in sync with the version installed by serving/setup_node.sh on the node.
VLLM_VERSION = "0.32.0"
GPU = "H100"

hf_cache = modal.Volume.from_name("gpu-lab-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    # debian_slim has no CUDA toolkit, and FlashInfer's top-k/top-p sampler is
    # JIT-compiled with nvcc on first use ("Could not find nvcc"). Use vLLM's
    # PyTorch sampler instead; the node has nvcc and uses FlashInfer's.
    .env({"HF_HOME": "/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_file(REPO / "serving" / "bench_sweep.py", f"{REMOTE}/serving/bench_sweep.py")
)

app = modal.App("gpu-lab-serve-vllm", image=image)


@app.function(gpu=GPU, timeout=40 * 60, volumes={"/hf": hf_cache})
def bench(models: list, concurrency: str, input_len: int, output_len: int,
          gpu_mem_util: float, max_model_len: int):
    yield {"event": "started", "container_host": socket.gethostname()}
    for model in models:
        label = "h100_" + model.split("/")[-1].lower().replace(".", "")
        out = Path("/tmp/results") / label
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, f"{REMOTE}/serving/bench_sweep.py", "--launch",
             "--model", model, "--label", label, "--out-dir", str(out),
             "--concurrency", concurrency, "--input-len", str(input_len),
             "--output-len", str(output_len), "--gpu-mem-util", str(gpu_mem_util),
             "--max-model-len", str(max_model_len)],
            cwd=REMOTE)
        hf_cache.commit()
        files = {p.name: p.read_text() for p in out.iterdir() if p.suffix in (".json", ".md", ".log")}
        yield {"event": "result", "model": model, "label": label, "exit_code": proc.returncode,
               "seconds": round(time.perf_counter() - t0, 1), "files": files}


@app.local_entrypoint()
def main(models: str = "Qwen/Qwen2.5-1.5B-Instruct", concurrency: str = "1,4,16,64,256",
         input_len: int = 512, output_len: int = 256, gpu_mem_util: float = 0.90,
         max_model_len: int = 4096):
    out = REPO / "results" / "serving"
    out.mkdir(parents=True, exist_ok=True)
    t_call = time.perf_counter()
    for msg in bench.remote_gen(models.split(","), concurrency, input_len, output_len,
                                gpu_mem_util, max_model_len):
        if msg["event"] == "started":
            print(f"[modal] container up on {GPU} after {time.perf_counter() - t_call:.1f}s")
            continue
        for name, text in msg["files"].items():
            (out / f"modal_{name}").write_text(text, encoding="utf-8")
        summaries = [n for n in msg["files"] if "_summary_" in n and n.endswith(".md")]
        for n in summaries:
            print(msg["files"][n])
        print(f"[modal] {msg['model']}: exit {msg['exit_code']}, {msg['seconds']}s in container")
    wall = time.perf_counter() - t_call
    print(f"[modal] wall {wall:.0f}s, ~${wall / 3600 * 3.95:.2f} at ~$3.95/h H100 (approx)")
    print(f"[modal] wrote results to {out}")
