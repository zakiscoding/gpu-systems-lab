"""Same-client, same-prompt comparison of OpenAI-compatible LLM servers.

Every target is driven by the same client, `vllm bench serve` (run from the
pinned vllm/vllm-openai image, so the client never changes between targets),
with the same random-token prompts: fixed input/output lengths, EOS ignored,
the same seed per concurrency level. Only the server behind --base-url
changes, so differences in throughput and latency come from the server.

Usage (GPU PC, WSL Ubuntu, from the repo root, one server up at a time):
    python3 serving/triton/bench_compare.py --label vllm --base-url http://127.0.0.1:8000 \
        --model qwen2.5-1.5b-instruct
    python3 serving/triton/bench_compare.py --label triton-trtllm --base-url http://127.0.0.1:9000 \
        --model ensemble

Writes raw per-level JSON plus <label>_summary_<ts>.json/.md to --out-dir.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLIENT_IMAGE = "vllm/vllm-openai:v0.32.0"
TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"
PERCENTILES = (50, 95, 99)
# vllm 0.32's client always sends "repetition_penalty": 1.0 (a no-op). Triton's
# OpenAI frontend rejects unknown fields with 422, so every target is benchmarked
# with the same patched copy of the client module that leaves that one line out.
CLIENT_MODULE = "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/lib/endpoint_request_func.py"


def patched_client(out_dir):
    """Write the client module minus the repetition_penalty line; return its path."""
    dest = out_dir / "endpoint_request_func.py"
    src = subprocess.run(["docker", "run", "--rm", "--entrypoint", "cat", CLIENT_IMAGE, CLIENT_MODULE],
                         capture_output=True, text=True, check=True).stdout
    kept = [ln for ln in src.splitlines(keepends=True) if '"repetition_penalty": 1.0,' not in ln]
    if len(kept) != len(src.splitlines(keepends=True)) - 1:
        raise RuntimeError("expected exactly one repetition_penalty line in the client")
    dest.write_text("".join(kept))
    return dest


def gpu_info():
    """GPU name, memory used/total (MiB), driver, power (W), or {"error": ...}."""
    try:
        # systemd units and WSL often lack nvidia-smi on PATH; WSL keeps it in /usr/lib/wsl/lib.
        smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.used,memory.total,driver_version,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        name, used, total, drv, power, util = [x.strip() for x in out.split(",")]
        return {"gpu": name, "mem_used_mib": int(used), "mem_total_mib": int(total), "driver": drv,
                "power_w": float(power), "util_pct": int(util)}
    except Exception as e:
        return {"error": repr(e)}


def wait_ready(base_url, timeout_s=60):
    """Wait until GET <base_url>/v1/models answers 200."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models", headers=auth_headers())
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return json.loads(r.read())
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"{base_url} not ready after {timeout_s}s")


def auth_headers():
    key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def bench_cmd(args, conc, n, fname, out_dir, seed, output_len):
    """`docker run ... vllm bench serve` against the target. The HF cache is
    mounted read-only for the tokenizer; results land in out_dir."""
    hf = Path.home() / ".cache" / "huggingface"
    env = ["-e", "HF_HUB_OFFLINE=1"]
    if os.environ.get("OPENAI_API_KEY"):
        env += ["-e", "OPENAI_API_KEY"]  # value passed through the environment, never on argv
    return [
        "docker", "run", "--rm", "--network", "host", *env,
        "-v", f"{hf}:/root/.cache/huggingface:ro",
        "-v", f"{out_dir.resolve()}:/out",
        "-v", f"{(out_dir / 'endpoint_request_func.py').resolve()}:{CLIENT_MODULE}:ro",
        "--entrypoint", "vllm", CLIENT_IMAGE,
        "bench", "serve",
        "--backend", "openai",
        "--base-url", args.base_url,
        "--endpoint", "/v1/completions",
        "--model", args.model,
        "--tokenizer", TOKENIZER,
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(output_len),
        "--ignore-eos",
        "--num-prompts", str(n),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", ",".join(str(p) for p in PERCENTILES),
        "--seed", str(seed),
        "--save-result", "--result-dir", "/out", "--result-filename", fname,
    ]


def summarize(raw, conc, n, wall_s, fname, g):
    """One table row from a raw `vllm bench serve` result."""
    def p(metric, pct):
        v = raw.get(f"p{pct}_{metric}_ms")
        return None if v is None else round(v, 2)

    row = {
        "concurrency": conc,
        "num_prompts": n,
        "completed": raw.get("completed"),
        "failed": n - (raw.get("completed") or 0),
        "duration_s": round(raw.get("duration", 0), 2),
        "request_throughput": round(raw["request_throughput"], 3),
        "output_tok_s": round(raw["output_throughput"], 1),
        "total_tok_s": round(raw.get("total_token_throughput", 0), 1),
        "gpu_mem_used_mib": g.get("mem_used_mib"),
        "gpu_power_w": g.get("power_w"),
        "wall_s": round(wall_s, 1),
        "raw_file": fname,
    }
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for pct in PERCENTILES:
            row[f"{metric}_p{pct}_ms"] = p(metric, pct)
    return row


def run_level(args, conc, out_dir, stamp):
    n = max(args.min_prompts, args.prompts_per_conc * conc)
    fname = f"{args.label}_c{conc}_{stamp}.json"
    print(f"[bench] {args.label} concurrency={conc} prompts={n}", flush=True)
    t0 = time.time()
    proc = subprocess.run(bench_cmd(args, conc, n, fname, out_dir, conc, args.output_len),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
        raise RuntimeError(f"bench failed at concurrency {conc}")
    g = gpu_info()  # sampled right after the level, while the server still holds its memory
    row = summarize(json.loads((out_dir / fname).read_text()), conc, n, time.time() - t0, fname, g)
    if row["failed"]:
        raise RuntimeError(f"{row['failed']} of {n} requests failed at concurrency {conc}; not recording")
    print(f"[bench]   {row['output_tok_s']} out tok/s | TTFT p50/p95 {row['ttft_p50_ms']}/{row['ttft_p95_ms']} ms"
          f" | E2E p50/p95 {row['e2el_p50_ms']}/{row['e2el_p95_ms']} ms | failed {row['failed']}", flush=True)
    return row


def markdown(s):
    lines = [
        f"### {s['label']}: {s['model']} on {s['gpu'].get('gpu')}",
        "",
        f"input {s['input_len']} / output {s['output_len']} tokens, client `{CLIENT_IMAGE}` `vllm bench serve`",
        "",
        "| conc | req/s | out tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p50 | E2E p95 | failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["rows"]:
        lines.append(
            f"| {r['concurrency']} | {r['request_throughput']} | {r['output_tok_s']} | {r['ttft_p50_ms']} | "
            f"{r['ttft_p95_ms']} | {r['tpot_p50_ms']} | {r['tpot_p95_ms']} | {r['e2el_p50_ms']} | "
            f"{r['e2el_p95_ms']} | {r['failed']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="model name the server expects in requests")
    ap.add_argument("--concurrency", default="1,4,16,64,256")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--prompts-per-conc", type=int, default=4)
    ap.add_argument("--min-prompts", type=int, default=16)
    ap.add_argument("--notes", default="", help="free-text server config, stored in the summary")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "triton"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patched_client(out_dir)
    models = wait_ready(args.base_url)
    gpu_before = gpu_info()

    # Untimed warm-up at the top level, so lazy init and allocator growth don't land in a measured run.
    top = max(int(c) for c in args.concurrency.split(","))
    subprocess.run(bench_cmd(args, top, top, f"{args.label}_warmup_{stamp}.json", out_dir, 9999, 32),
                   capture_output=True, text=True)

    rows = [run_level(args, int(c), out_dir, stamp) for c in args.concurrency.split(",")]
    summary = {
        "label": args.label,
        "model": args.model,
        "served_models": [m.get("id") for m in models.get("data", [])],
        "base_url": args.base_url,
        "notes": args.notes,
        "client_image": CLIENT_IMAGE,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "gpu": gpu_before,
        "rows": rows,
        "timestamp": stamp,
    }
    spath = out_dir / f"{args.label}_summary_{stamp}.json"
    spath.write_text(json.dumps(summary, indent=2))
    md = markdown(summary)
    (out_dir / f"{args.label}_summary_{stamp}.md").write_text(md)
    print(md)
    print(f"[bench] wrote {spath}")


if __name__ == "__main__":
    main()
