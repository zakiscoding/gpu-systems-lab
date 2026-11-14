"""Concurrency sweep against a vLLM OpenAI-compatible server.

Runs `vllm bench serve` once per concurrency level with fixed random-token
prompts (same input/output length every request, EOS ignored), so the only
variable is how many requests are in flight. Each raw result JSON is kept and
one summary JSON + Markdown table is written next to them.

With --launch the script also starts `vllm serve` itself, waits for /health,
parses the KV-cache numbers from the server log, and stops the server at the
end. That is how modal/serve_vllm.py runs it inside a single container.

Usage (GPU node, from the repo root):
    ~/venvs/vllm/bin/python serving/bench_sweep.py --launch --label rtx5070
    # or against an already running server (serving/launch_server.sh):
    ~/venvs/vllm/bin/python serving/bench_sweep.py --label rtx5070
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Natural-language prompts for --dataset custom (serving/make_prompts.py writes them).
# Random-token prompts suit throughput sweeps but make speculative decoding look
# useless: neither n-grams nor a draft model can predict a continuation of noise.
CUSTOM_PROMPTS = REPO / "serving" / "prompts" / "chat_prompts.jsonl"

# Server flags used for every benchmark run (with --launch). Start an external
# server the same way: bash serving/launch_server.sh --no-enable-prefix-caching --generation-config vllm
#  * no prefix caching: random prompts from a fixed seed repeat across runs and
#    would otherwise hit the cache (we saw 30-40% hit rates), skipping prefill.
#  * --generation-config vllm: ignore the model's generation_config.json
#    (Qwen sets top_k/top_p/repetition_penalty), so the benchmark measures the
#    engine rather than the sampler (its top-p Triton kernel JITs mid-run).
BENCH_SERVE_ARGS = ["--no-enable-prefix-caching", "--generation-config", "vllm"]

# Lines vLLM prints at startup that describe KV-cache capacity. Phrasing has
# shifted across versions, so match loosely and keep the raw lines too.
KV_PATTERNS = {
    "kv_cache_memory_gib": re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    "max_concurrency_at_max_len": re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x"),
    "model_weights_gib": re.compile(r"Model loading took\s*([\d.]+)\s*Gi?B"),
}


def vllm_cli():
    """Path to the `vllm` executable: the one next to this interpreter (the venv's)
    if present, else whatever is on PATH."""
    exe = Path(sys.executable).with_name("vllm")
    return str(exe) if exe.exists() else (shutil.which("vllm") or "vllm")


def gpu_info():
    """GPU name, used/total memory (MiB) and driver version from nvidia-smi, or
    {"error": ...} if it can't be read. Never raises."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        name, used, total, drv = [x.strip() for x in out.split(",")]
        return {"gpu": name, "mem_used_mib": int(used), "mem_total_mib": int(total), "driver": drv}
    except Exception as e:  # nvidia-smi can be missing/odd inside sandboxes
        return {"error": repr(e)}


def wait_health(base_url, timeout_s, proc=None):
    """Poll <base_url>/health every 2 s until it returns 200; return seconds waited.

    Raises TimeoutError after `timeout_s`, or RuntimeError as soon as `proc`
    (the server we launched, if any) has exited, instead of waiting it out.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server not healthy after {timeout_s}s")


def parse_kv(log_text):
    """Pull KV-cache capacity numbers out of a vLLM server log: KV cache memory
    (GiB), KV cache size (tokens), max concurrency at max length, and model
    weights (GiB), plus up to 8 matching raw lines for when a pattern misses."""
    kv = {}
    for key, pat in KV_PATTERNS.items():
        m = pat.search(log_text)
        if not m:
            continue
        if key == "max_concurrency_at_max_len":
            kv["max_concurrency_tokens_per_request"] = int(m.group(1).replace(",", ""))
            kv[key] = float(m.group(2))
        elif key == "kv_cache_tokens":
            kv[key] = int(m.group(1).replace(",", ""))
        else:
            kv[key] = float(m.group(1))
    kv["raw_lines"] = [ln.strip() for ln in log_text.splitlines()
                       if re.search(r"KV cache|Maximum concurrency|Model loading took", ln)][:8]
    return kv


def ensure_pandas():
    """vLLM's custom-dataset loader reads JSONL with pandas, which the vllm/vllm-openai
    image doesn't ship ("Please install vllm[bench]"). Install it into this container
    before the first --dataset custom run."""
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("[bench] installing pandas for the custom dataset", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pandas"], check=True)


def dataset_args(args, output_len):
    """`vllm bench serve` flags for the prompt source: fixed-length random tokens,
    or the repo's natural-language prompts with a fixed output length."""
    if args.dataset == "custom":
        return ["--dataset-name", "custom", "--dataset-path", str(CUSTOM_PROMPTS),
                "--custom-output-len", str(output_len)]
    return ["--dataset-name", "random", "--random-input-len", str(args.input_len),
            "--random-output-len", str(output_len)]


def spec_counters(base_url):
    """Speculative-decoding counters from the server's Prometheus /metrics:
    {"draft": tokens proposed, "accepted": tokens accepted}, summed over labels.
    Empty when speculation is off (the metrics don't exist)."""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for key, metric in (("draft", "vllm:spec_decode_num_draft_tokens_total"),
                        ("accepted", "vllm:spec_decode_num_accepted_tokens_total")):
        vals = [float(ln.rsplit(" ", 1)[1]) for ln in text.splitlines()
                if ln.startswith(metric) and not ln.startswith("#")]
        if vals:
            out[key] = sum(vals)
    return out


def run_one(args, conc, out_dir, stamp):
    """Run one `vllm bench serve` at `conc` requests in flight and return its summary row.

    Sends max(--min-prompts, --prompts-per-conc * conc) prompts at an unlimited
    request rate, so the concurrency cap is the only limiter. The raw result
    JSON is saved in `out_dir`; the row keeps throughput (req/s, tok/s), TTFT /
    TPOT / ITL / end-to-end latency p50/p99 (ms) and GPU memory used (MiB).
    Raises RuntimeError if the benchmark fails.
    """
    n = max(args.min_prompts, args.prompts_per_conc * conc)
    fname = f"{args.label}_c{conc}_{stamp}.json"
    cmd = [
        vllm_cli(), "bench", "serve",
        "--backend", "openai",
        "--base-url", args.base_url,
        "--endpoint", "/v1/completions",
        "--model", args.model,
        *dataset_args(args, args.output_len),
        "--ignore-eos",
        "--num-prompts", str(n),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
        "--seed", str(conc),  # different prompts per level
        "--save-result", "--result-dir", str(out_dir), "--result-filename", fname,
    ]
    print(f"[bench] concurrency={conc} prompts={n}", flush=True)
    t0 = time.time()
    spec_before = spec_counters(args.base_url)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
        raise RuntimeError(f"vllm bench serve failed at concurrency {conc}")
    raw = json.loads((out_dir / fname).read_text())
    g = gpu_info()
    spec_after = spec_counters(args.base_url)

    def p(name, pct):
        """Percentile `pct` of a latency metric from the raw result (ms), or None."""
        return raw.get(f"p{pct}_{name}_ms")

    row = {
        "concurrency": conc,
        "num_prompts": n,
        "completed": raw.get("completed"),
        "duration_s": round(raw.get("duration", 0), 2),
        "request_throughput": round(raw["request_throughput"], 3),
        "output_tok_s": round(raw["output_throughput"], 1),
        "total_tok_s": round(raw.get("total_token_throughput", 0), 1),
        "ttft_p50_ms": round(p("ttft", 50), 1),
        "ttft_p95_ms": round(p("ttft", 95), 1),
        "ttft_p99_ms": round(p("ttft", 99), 1),
        "tpot_p50_ms": round(p("tpot", 50), 2),
        "tpot_p95_ms": round(p("tpot", 95), 2),
        "tpot_p99_ms": round(p("tpot", 99), 2),
        "itl_p50_ms": round(p("itl", 50), 2),
        "itl_p95_ms": round(p("itl", 95), 2),
        "itl_p99_ms": round(p("itl", 99), 2),
        "e2el_p50_ms": round(p("e2el", 50), 1),
        "e2el_p95_ms": round(p("e2el", 95), 1),
        "e2el_p99_ms": round(p("e2el", 99), 1),
        "gpu_mem_used_mib": g.get("mem_used_mib"),
        "wall_s": round(time.time() - t0, 1),
        "raw_file": fname,
    }
    if spec_after.get("draft"):
        drafted = spec_after["draft"] - spec_before.get("draft", 0)
        accepted = spec_after.get("accepted", 0) - spec_before.get("accepted", 0)
        row["spec_draft_tokens"] = int(drafted)
        row["spec_accepted_tokens"] = int(accepted)
        row["spec_acceptance_rate"] = round(accepted / drafted, 3) if drafted else None
    print(f"[bench]   {row['output_tok_s']} out tok/s, {row['request_throughput']} req/s, "
          f"TTFT p50/p99 {row['ttft_p50_ms']}/{row['ttft_p99_ms']} ms, "
          f"TPOT p50/p99 {row['tpot_p50_ms']}/{row['tpot_p99_ms']} ms", flush=True)
    return row


def markdown(summary):
    """Render the sweep summary as a Markdown heading, a config line and one table
    row per concurrency level (latencies in ms)."""
    lines = [
        f"### {summary['label']}: {summary['model']} on {summary['gpu'].get('gpu')}",
        "",
        f"vLLM {summary['vllm_version']}, input {summary['input_len']} / output "
        f"{summary['output_len']} tokens, KV cache {summary['kv_cache'].get('kv_cache_tokens')} tokens "
        f"({summary['kv_cache'].get('kv_cache_memory_gib')} GiB)",
        "",
        "| conc | req/s | out tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p50 | E2E p95 | spec accept |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['concurrency']} | {r['request_throughput']} | {r['output_tok_s']} | "
            f"{r['ttft_p50_ms']} | {r.get('ttft_p95_ms')} | {r['tpot_p50_ms']} | {r.get('tpot_p95_ms')} | "
            f"{r['e2el_p50_ms']} | {r.get('e2el_p95_ms')} | {r.get('spec_acceptance_rate', '-')} |")
    return "\n".join(lines) + "\n"


def main():
    """Run the sweep and write <label>_summary_<ts>.json / .md to --out-dir.

    With --launch: start `vllm serve` in its own process group (so the whole
    tree can be signalled), log to <label>_server_<ts>.log, wait up to 20 min
    for /health, and always stop it at the end (SIGTERM, SIGKILL after 60 s).
    Without it: require an already-healthy server at --base-url within 60 s.
    An untimed warm-up run at the highest concurrency precedes the measured
    levels. Returns the summary dict.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--prompts-per-conc", type=int, default=4)
    ap.add_argument("--min-prompts", type=int, default=16)
    ap.add_argument("--label", default="local")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "serving"))
    ap.add_argument("--server-log", default=None, help="vllm serve log to parse KV-cache size from")
    ap.add_argument("--launch", action="store_true", help="start vllm serve here and stop it afterwards")
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--extra-serve-args", default="", help="extra args for vllm serve (space separated)")
    ap.add_argument("--dataset", choices=["random", "custom"], default="random",
                    help="random: fixed --input-len random tokens; custom: serving/prompts/chat_prompts.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "custom":
        ensure_pandas()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_before = gpu_info()

    server = None
    startup_s = None
    log_path = Path(args.server_log) if args.server_log else None
    if args.launch:
        log_path = out_dir / f"{args.label}_server_{stamp}.log"
        # serve on the port named in --base-url so the benchmark client finds it
        port = args.base_url.rsplit(":", 1)[-1].strip("/")
        cmd = [vllm_cli(), "serve", args.model, "--host", "127.0.0.1", "--port", port,
               "--gpu-memory-utilization", str(args.gpu_mem_util),
               "--max-model-len", str(args.max_model_len)] + BENCH_SERVE_ARGS + args.extra_serve_args.split()
        print("[serve]", " ".join(cmd), flush=True)
        logf = open(log_path, "w")
        env = dict(os.environ)
        # Running the venv's python without `activate` leaves venv/bin (ninja,
        # used by FlashInfer's JIT) off PATH.
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
        # Same sampler on every machine: debian_slim on Modal has no nvcc for
        # FlashInfer's JIT sampler, so both sides use vLLM's PyTorch sampler.
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        server = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  start_new_session=True, env=env)
        try:
            startup_s = wait_health(args.base_url, 1200, server)
        except Exception:
            logf.flush()
            print(log_path.read_text()[-6000:])
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
            raise
        print(f"[serve] healthy after {startup_s:.0f}s", flush=True)
    else:
        wait_health(args.base_url, 60)

    try:
        kv = parse_kv(log_path.read_text()) if log_path and log_path.exists() else {}
        gpu_loaded = gpu_info()
        # Warm-up at the largest batch so first-use costs (lazy JIT, allocator
        # growth) land here instead of in the measured runs.
        top = max(int(c) for c in args.concurrency.split(","))
        subprocess.run([vllm_cli(), "bench", "serve", "--backend", "openai", "--base-url", args.base_url,
                        "--endpoint", "/v1/completions", "--model", args.model,
                        *dataset_args(args, 32), "--ignore-eos", "--num-prompts", str(top), "--max-concurrency", str(top),
                        "--seed", "9999"],
                       capture_output=True, text=True)
        rows = [run_one(args, int(c), out_dir, stamp) for c in args.concurrency.split(",")]
    finally:
        if server is not None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)

    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception:
        vllm_version = "unknown"
    summary = {
        "label": args.label,
        "model": args.model,
        "vllm_version": vllm_version,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "gpu_mem_util": args.gpu_mem_util if args.launch else None,
        "max_model_len": args.max_model_len if args.launch else None,
        "server_startup_s": round(startup_s, 1) if startup_s else None,
        "gpu": gpu_before,
        "gpu_mem_used_mib_before": gpu_before.get("mem_used_mib"),
        "gpu_mem_used_mib_loaded": gpu_loaded.get("mem_used_mib"),
        "serve_args": BENCH_SERVE_ARGS if args.launch else None,
        "extra_serve_args": args.extra_serve_args.split() if args.launch else None,
        "dataset": args.dataset,
        "kv_cache": kv,
        "rows": rows,
        "timestamp": stamp,
    }
    spath = out_dir / f"{args.label}_summary_{stamp}.json"
    spath.write_text(json.dumps(summary, indent=2))
    md = markdown(summary)
    (out_dir / f"{args.label}_summary_{stamp}.md").write_text(md)
    print(md)
    print(f"[bench] wrote {spath}")
    return summary


if __name__ == "__main__":
    main()
