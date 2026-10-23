# Inference serving: vLLM on the RTX 5070 vs 1x H100

[homelab](https://github.com/zakiscoding/homelab) (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../README.md) / `serving/` · [← repo README](../README.md) · [Results](../docs/RESULTS.md) · [Code guide](../docs/CODE-GUIDE.md)

An OpenAI-compatible LLM server ([vLLM](https://github.com/vllm-project/vllm)) on the home GPU node,
benchmarked with a concurrency sweep, then the same model and the same benchmark on a Modal H100.
The question: how far does batching push throughput before latency or KV-cache memory gives out, and
what does a token cost on each machine?

| File | What it does |
|---|---|
| [setup_node.sh](setup_node.sh) | Creates `~/venvs/vllm` on the node (plain venv + uv, no sudo). It is separate from the repo `.venv` because vLLM pins its own torch. |
| [launch_server.sh](launch_server.sh) | `vllm serve` on `127.0.0.1:8000` (refuses `0.0.0.0`). Optional `VLLM_API_KEY`. |
| [bench_sweep.py](bench_sweep.py) | Runs `vllm bench serve` at concurrency 1, 4, 16, 64, 256. Writes raw JSON, a summary JSON and a Markdown table to `results/serving/` (gitignored). With `--launch` it also starts and stops the server and parses the KV-cache size from its log. |
| [../modal/serve_vllm.py](../modal/serve_vllm.py) | Runs the same `bench_sweep.py --launch` inside one Modal H100 container, with server and client together, and copies the results back. |
| [compose.yml](compose.yml) + [make-env.sh](make-env.sh) | The **always-on chat backend** for the homelab web UI: `vllm/vllm-openai:v0.32.0` in Docker on the tailnet, with an API key. |

## Homelab chat backend (always on)

The homelab web UI's `/chat` and RAG answers come from this server whenever the GPU PC runs Ubuntu.
The iMac API ([homelab-services](https://github.com/zakiscoding/homelab-services) `api/llm.py`,
`LLM_BACKEND=auto`) uses vLLM when it answers, and falls back to Ollama when the PC is booted into
Windows.

```bash
# on the node, once
cd ~/gpu-systems-lab/serving && ./make-env.sh && docker compose up -d
```

| Setting | Value | Why |
|---|---|---|
| Image | `vllm/vllm-openai:v0.32.0` (pinned) | same vLLM as the benchmarks, runs on sm_120 |
| Address | `100.121.51.35:8000/v1` | tailnet only; the LAN IP refuses connections (tested) |
| Auth | bearer `VLLM_API_KEY` from `serving/.env` (mode 600, gitignored) | requests without the key get 401 (tested) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, served as `qwen2.5-1.5b-instruct` | fits beside other GPU work |
| GPU memory | `--gpu-memory-utilization 0.45` (~5.4 GB) | leaves room for the desktop and k3s jobs. k3s can't see this usage, so `docker compose stop` before big GPU jobs |
| Restart | `unless-stopped` + healthcheck on `/health` | comes back after a reboot |

Tested from the laptop through the homelab API code: the UI's default `llama3.2:latest` maps to the
served model, a one-sentence answer takes 0.9 s, and with vLLM unreachable `auto` switches to Ollama.

## Setup and commands

```bash
# on the GPU node (ssh gpu-ts), from ~/gpu-systems-lab
bash serving/setup_node.sh                 # installs vllm 0.32.0 + torch 2.13.0+cu132, ~8 GB

# interactive server (localhost only; reach it from the laptop with ssh -N -L 8000:127.0.0.1:8000 gpu-ts)
bash serving/launch_server.sh
curl -s 127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"GPUs are","max_tokens":32}'

# benchmark: starts its own server, sweeps, stops it (~4 min)
~/venvs/vllm/bin/python serving/bench_sweep.py --launch --label rtx5070_qwen25-15b-instruct
```

```bash
# on the laptop, from the repo root (PYTHONUTF8/PYTHONIOENCODING=utf-8 on Windows)
modal run modal/serve_vllm.py                                            # 1.5B on H100
modal run modal/serve_vllm.py --models Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct
```

**Benchmark settings** (identical on both GPUs):

- `vllm bench serve` with the `random` dataset: 512 input and 256 output tokens per request, `--ignore-eos`.
- Request rate `inf`, capped by `--max-concurrency`. That means a closed loop: C requests are always in flight.
- `max(16, 4*C)` prompts per level, with a different seed per level.
- `--max-model-len 4096`. `--gpu-memory-utilization` is 0.80 on the 5070 (it also drives the desktop and is shared) and 0.90 on the H100.
- Server flags `--no-enable-prefix-caching --generation-config vllm`, plus `VLLM_USE_FLASHINFER_SAMPLER=0`. The reasons are under [Pitfalls](#pitfalls-hit-along-the-way).
- One warm-up pass at the top concurrency before measuring.

## vLLM on Blackwell (sm_120)

**It worked out of the box.** On 2026-10-23, `uv pip install vllm --torch-backend=auto` resolved
**vLLM 0.32.0 with torch 2.13.0+cu132**. That torch build includes `sm_120` in its arch list, and
driver 595.99.02 supports CUDA 13.2. No nightly wheel or source build was needed. The same 0.32.0
wheel runs on Modal's H100 (sm_90).

The first start on the 5070 spent 376 s just downloading and loading the weights. A start with the
weights cached but no compile cache reached `/health` in 218 s, including torch.compile and CUDA
graph capture. Warm restarts take **38 s** because the compile cache is in `~/.cache/vllm`. On
Modal, startup to `/health` took 106 s for 1.5B and 148 s for 7B. There is no compile cache in a
fresh container, and the HF weights come from a Modal Volume after the first download.

## Results

GPU state before the final 5070 run was idle (671 MiB used, 0% utilization) with no other compute
processes. Other agents were installing software on the node earlier (load average ~18), but not
during this run. Raw files: `results/serving/rtx5070_*`, `results/serving/modal_h100_*`.

### Qwen2.5-1.5B-Instruct: RTX 5070 vs H100

Latencies are in ms. TPOT is time per output token after the first; ITL is the gap between streamed tokens.

| conc | 5070 out tok/s | H100 out tok/s | H100/5070 | 5070 req/s | H100 req/s | 5070 TTFT p50 / p99 | H100 TTFT p50 / p99 | 5070 TPOT p50 / p99 | H100 TPOT p50 / p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 172 | 434 | 2.5x | 0.67 | 1.69 | 41.5 / 43.0 | 16.6 / 18.6 | 5.66 / 5.68 | 2.25 / 2.25 |
| 4 | 612 | 1,647 | 2.7x | 2.39 | 6.43 | 82.8 / 115 | 33.2 / 44.6 | 6.21 / 6.44 | 2.30 / 2.30 |
| 16 | 1,987 | 5,842 | 2.9x | 7.76 | 22.8 | 269 / 414 | 72.0 / 90.5 | 7.00 / 7.89 | 2.45 / 2.55 |
| 64 | 4,430 | **16,843** | 3.8x | 17.3 | 65.8 | 387 / 1,557 | 179 / 280 | 13.1 / 14.1 | 3.09 / 3.39 |
| 256 | **4,791** | 11,989 | 2.5x | 18.7 | 46.8 | 557 / **7,423** | 555 / 974 | **52.5** / 53.7 | 19.2 / 20.0 |

ITL p50 / p99 (ms). 5070: 5.67/6.27, 6.14/7.08, 6.60/8.94, 8.58/99.9, 22.1/144. H100: 2.25/2.95,
2.30/2.62, 2.45/3.20, 5.76/27.8, 17.6/43.7.

### Memory and KV cache

| | RTX 5070 (12 GB) | H100 80GB, 1.5B | H100 80GB, 7B |
|---|---:|---:|---:|
| Weights (bf16) | 2.98 GiB | 2.98 GiB | 14.29 GiB |
| KV cache memory | 5.09 GiB | 66.32 GiB | 54.14 GiB |
| KV cache size | **190,624 tokens** | 2,483,488 tokens | 1,013,824 tokens |
| 4096-token requests that fit at once | 46.5 | 606 | 247.5 |
| GPU memory used while serving (nvidia-smi, includes desktop on the 5070) | 9.8 GiB | 71.9 GiB | 72.1 GiB |
| Max tokens per scheduler step (vLLM default) | 2,048 | 8,192 | 8,192 |
| Peak KV usage in the c=256 sweep (server log) | **94.6%** | 5.5% | 18.8% |

### Qwen2.5-7B-Instruct on H100 only

At bf16, 15.2 GB of weights does not fit in the 5070's 12 GB.

| conc | out tok/s | req/s | TTFT p50 / p99 | TPOT p50 / p99 | ITL p50 / p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 162 | 0.63 | 20.2 / 21.9 | 6.12 / 6.12 | 6.11 / 6.72 |
| 4 | 625 | 2.44 | 47.8 / 58.1 | 6.22 / 6.28 | 6.17 / 7.01 |
| 16 | 2,237 | 8.74 | 155 / 193 | 6.53 / 6.99 | 6.47 / 7.33 |
| 64 | 6,269 | 24.5 | 504 / 756 | 8.27 / 9.79 | 7.38 / 16.5 |
| 256 | **8,436** | 33.0 | 1,675 / 3,078 | 23.8 / 28.6 | 18.9 / 179 |

The 7B on an H100 decodes one stream at about the same speed as the 1.5B on the 5070 (162 vs 172
tok/s, TPOT 6.1 vs 5.7 ms). It also has 5x the KV capacity.

### Cost per token (approximate)

The H100 price is Modal's ~$3.95/h list price, and it is approximate. The 5070 is already owned, so
its marginal cost is electricity only. That figure is **an assumption, not a measurement**: full
250 W board power at $0.15/kWh, or about $0.04/h. Board power was not logged during the runs.

| Setup | out tok/s | Output tokens per $ | $ per 1M output tokens |
|---|---:|---:|---:|
| H100, 1.5B, c=1 | 434 | 0.40 M | ~$2.53 |
| H100, 1.5B, c=64 (best) | 16,843 | 15.4 M | **~$0.065** |
| H100, 7B, c=1 | 162 | 0.15 M | ~$6.77 |
| H100, 7B, c=256 (best) | 8,436 | 7.7 M | **~$0.13** |
| RTX 5070, 1.5B, c=1 (electricity only) | 172 | ~16 M | ~$0.06 |
| RTX 5070, 1.5B, c=256 (electricity only) | 4,791 | ~460 M | ~$0.002 |

**Spend on this experiment:** about 991 s of H100 time across three Modal runs, which is about
**$1.09**. That includes a first run that failed after 114 s because FlashInfer could not find nvcc.

## Conclusions

1. **Batching is almost free throughput, up to a point.** On the 5070, going from 1 to 64 concurrent
   requests raises output throughput **25.7x** (172 → 4,430 tok/s). Per-token latency only rises
   2.3x (TPOT 5.7 → 13.1 ms). Decode is memory-bound: each step reads all the weights once, whether
   it serves 1 sequence or 64. The H100 holds a flat TPOT of ~2.3 ms all the way to c=16. At c=64 it
   reaches **38.8x** its single-stream rate at 3.1 ms.
2. **Past saturation, extra concurrency only adds queueing.** On the 5070, going from c=64 to c=256
   adds 8% throughput (4,430 → 4,791) but costs 4x TPOT (13 → 52 ms). TTFT p99 rises from 1.6 s to
   **7.4 s**, and median end-to-end latency from 3.7 s to 14.0 s. TTFT climbs earlier than TPOT
   because new prompts wait behind the 2,048-token per-step prefill budget. At c=16 the 5070's TTFT
   p50 is already 269 ms, against 72 ms on the H100 with its 8,192-token budget.
3. **KV-cache capacity is the ceiling on the 12 GB card.** 256 concurrent requests × 768 tokens is
   196,608 tokens, more than the 190,624-token cache. KV usage peaked at 94.6%, and the server log
   shows up to 31 requests waiting while the cache was 65–80% full. Halving the per-request context
   or quantizing the KV cache would raise the concurrency ceiling. The H100 has 13x more KV tokens
   for the same model. Its KV usage peaked at 5.5% for 1.5B and 18.8% for 7B.
4. **The H100 is 2.5x faster at low concurrency and up to 3.8x faster at its best batch.** The
   single-stream gap (2.5x) is close to the memory-bandwidth gap (3.35 TB/s vs 672 GB/s is 5x),
   minus the fixed per-step overhead. The 10x matmul gap in [docs/RESULTS.md](../docs/RESULTS.md)
   only shows up once batches are large enough to be compute-bound.
5. **The H100 1.5B run lost throughput from c=64 to c=256 (16.8k → 12.0k tok/s), and the cause is
   not confirmed.** TPOT went from 3.1 to 19 ms while KV usage peaked at 5.5%, so neither GPU
   memory nor KV is the limit. The likely cause is CPU: the API server, detokenization of 256
   streams and the benchmark client all share one gVisor container, and no `cpu=` was requested. The
   next step is to rerun with `@app.function(cpu=8)`, which was not done to stay within budget.
6. **Cost:** a busy H100 serves the 1.5B model at about **$0.07 per 1M output tokens**. At c=1 it
   costs **$2.53 per 1M**, because single-stream decode wastes most of the GPU. Serving is only cheap
   when it is batched. The 5070 is far cheaper per token on marginal cost (electricity), but it
   delivers 3.5x less peak throughput and runs out of KV memory at about 250 concurrent 768-token
   requests. A 7B model at bf16 does not fit on it at all.

## Pitfalls hit along the way

- **FlashInfer sampler JIT.** FlashInfer compiles its top-k/top-p sampling kernel with nvcc on first
  use. On Modal's `debian_slim` image this failed ("Could not find nvcc"). On the node it failed with
  "No such file: 'ninja'" because `~/venvs/vllm/bin` was not on `PATH` when the venv's python was
  called directly. `bench_sweep.py` now prepends the venv bin to `PATH` and sets
  `VLLM_USE_FLASHINFER_SAMPLER=0`, so both GPUs use the same sampler.
- **Prefix-cache hits in a "random" benchmark.** With a fixed seed, each run regenerated the same
  first prompts as the previous run, and vLLM's prefix cache reported 30–40% hit rates. That skips
  prefill and flatters TTFT. The fix was `--no-enable-prefix-caching` plus a different seed per level.
- **The model's generation config.** Qwen ships `top_k=20, top_p=0.8, repetition_penalty=1.1`. That
  triggered a Triton top-p kernel JIT in the middle of the run, which put 1.6–3.3 s TTFT p99 spikes
  into the H100 c=16/c=64 results. `--generation-config vllm` uses vLLM's neutral defaults instead,
  and the warm-up now runs at the top concurrency.
- **KV size depends on free memory at startup.** An earlier attempt on the 5070 reported only
  2.57 GiB / 96,304 tokens of KV cache at the same 0.80 utilization. The final run reported
  5.09 GiB / 190,624 tokens. The cause was not isolated. On a shared card, check `nvidia-smi` and the
  `GPU KV cache size` log line before trusting a run.
