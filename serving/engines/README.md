# vLLM vs SGLang vs TensorRT-LLM on the RTX 5070, with FlashAttention vs FlashInfer

[gpu-systems-lab](../../README.md) / `serving/engines/` · [Triton + TensorRT-LLM setup](../triton/README.md) · [Results](../../docs/RESULTS.md)

This builds on the [Triton + TensorRT-LLM comparison](../triton/README.md): same model, same client, same
prompts. It adds **SGLang** as a third engine and compares **attention kernels** inside the two engines that
let you choose. It also answers the question a capacity plan actually asks: *how much throughput can you
get while still meeting a latency target?*

All five configurations ran back to back in **one GPU window** on 2026-11-14 (17:09–17:56 UTC), with the live
`vllm-chat` stopped:

| label | engine | attention |
|---|---|---|
| `vllm-fa` | vLLM 0.32.0 | FlashAttention (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`) |
| `vllm-fi` | vLLM 0.32.0 | FlashInfer |
| `sglang-fi` | SGLang 0.5.23 | FlashInfer (`--attention-backend flashinfer`) |
| `sglang-triton` | SGLang 0.5.23 | Triton kernels (`--attention-backend triton`) |
| `trtllm` | Triton Inference Server 26.07 + TensorRT-LLM 1.2.1 | the compiled engine's fused attention (from [build_engine.sh](../triton/build_engine.sh)) |

**Same for every row:**
- **Model:** Qwen2.5-1.5B-Instruct, bf16.
- **Memory:** 80% of the card for the vLLM and SGLang servers. TensorRT-LLM uses the engine built in PR #8 (95% of what's free after loading it).
- **Context:** 4096.
- **Caching:** prefix/radix caching off.
- **Client:** the pinned `vllm bench serve` client from [bench_compare.py](../triton/bench_compare.py), random prompts of 512 tokens in and 256 out, `ignore_eos`.
- **Load:** concurrency 1, 4, 16, 32, 64, 128 and 256, with `max(16, 4×C)` prompts each.
- **Warm-up:** one untimed run at concurrency 256 before measuring.

**0 failed requests at every level of every engine.** Raw per-level JSON and summaries are in
[results/engines/](../../results/engines/).

| File | What it does |
|---|---|
| [run_window.sh](run_window.sh) | Stops `vllm-chat`, runs the five configurations one at a time, then restores and verifies `vllm-chat` from an `EXIT` trap. Verification covers the model-cache mount, local health, the tailnet port, the gateway and a chat completion. `DRY_RUN=1` starts every engine with a 0.5B model at 15% of the card next to the live chat, to prove the commands work before taking a window |
| [goodput.py](goodput.py) | Best output tok/s among the measured levels that meet a TTFT p95 / TPOT p95 target (4 tests in [test_goodput.py](test_goodput.py)) |

## Throughput (output tokens/s)

| concurrency | vLLM FA | vLLM FlashInfer | SGLang FlashInfer | SGLang Triton | TensorRT-LLM |
|---:|---:|---:|---:|---:|---:|
| 1 | 160 | 166 | 172 | **174** | 146 |
| 4 | 550 | 578 | 590 | **594** | 503 |
| 16 | **1,889** | 1,869 | 832 | 768 | 1,609 |
| 32 | 3,022 | **3,048** | 1,516 | 1,382 | 2,518 |
| 64 | **4,260** | 4,235 | 2,588 | 2,422 | 3,579 |
| 128 | 5,030 | **5,046** | 4,117 | 3,883 | 4,457 |
| 256 | 4,670 | 4,729 | 5,118 | **5,243** | 4,584 |

## Latency (ms): time to first token and time per output token

| | TTFT p50 at 1 / 16 / 64 / 256 | TPOT p50 at 1 / 16 / 64 / 256 |
|---|---|---|
| vLLM FA | 35 / 257 / 398 / 685 | 6.1 / 7.5 / 13.4 / 52.7 |
| vLLM FlashInfer | 33 / 268 / 401 / 694 | 5.9 / 7.3 / 13.5 / 52.6 |
| SGLang FlashInfer | 34 / 291 / 831 / 2,999 | 5.7 / **18.1** / 21.6 / **38.4** |
| SGLang Triton | 33 / 282 / 839 / 2,917 | **5.6** / 19.7 / 23.2 / 37.5 |
| TensorRT-LLM | 34 / 379 / 958 / 1,136 | 6.7 / 8.4 / 13.8 / 46.6 |

## Throughput under a latency target

Best measured level that meets both limits (`goodput.py`):

| | peak tok/s | **best tok/s with TTFT p95 ≤ 1 s and TPOT p95 ≤ 30 ms** | best with TTFT p95 ≤ 500 ms and TPOT p95 ≤ 20 ms |
|---|---:|---:|---:|
| vLLM FA | 5,030 (128) | **3,022** (32) | **1,889** (16) |
| vLLM FlashInfer | 5,046 (128) | **3,048** (32) | 578 (4) |
| SGLang FlashInfer | 5,118 (256) | 1,516 (32) | 590 (4) |
| SGLang Triton | **5,243** (256) | 1,382 (32) | 594 (4) |
| TensorRT-LLM | 4,584 (256) | 2,518 (32) | 503 (4) |

## What it shows

1. **The peak number picks the wrong engine for chat.** SGLang has the highest raw throughput (5,243
   tok/s at 256 in flight), but its first token then takes 3 s at p50. Under a 1 s / 30 ms latency
   target, **vLLM delivers 3,000+ tok/s, about 2x SGLang and 1.2x TensorRT-LLM**. With a tighter
   500 ms / 20 ms target, only vLLM with FlashAttention still qualifies at 16 concurrent requests. Its
   FlashInfer twin's TTFT p95 at 16 was 578 ms. That's within run-to-run noise of the limit, so the
   500 ms column is a cliff edge, not a ranking.
2. **SGLang's dip at 16–128 requests is a configuration effect, and its log shows why.** With 80% of
   the card given to weights and KV cache, only 0.76 GB was left, so SGLang captured decode CUDA
   graphs for **batch sizes 1, 2, 4 and 8 only** (`Capture target decode CUDA graph end. bs=[1, 2, 4, 8]`).
   Above 8 running requests every decode step ran eagerly (`cuda graph: False` in its batch log),
   paying Python and kernel-launch overhead per step: TPOT 18–23 ms vs vLLM's 7–13 ms. At 256 in flight
   the batch is big enough to amortize that, so SGLang catches up and wins. Not tested, because it needs
   another chat-down window: a lower `--mem-fraction-static` or an explicit `--cuda-graph-max-bs`, which
   trades some KV cache for graphs at every batch size. This is the most likely place SGLang would gain.
3. **FlashAttention vs FlashInfer barely matters at this size.** Inside vLLM the two backends are
   within 1–5% at every level, and FlashInfer is slightly ahead at low concurrency. Inside SGLang,
   FlashInfer beats the Triton attention kernels by 6–11% at 16–128, and Triton wins at 1–4 and 256.
   With 512-token prompts and a 1.5B model, attention is a small part of each step; the batching
   engine around it matters more.
4. **TensorRT-LLM reproduces day to day.** Its numbers match the previous day's run in
   [serving/triton](../triton/README.md) within 2% (146 vs 144 tok/s at 1, 4,584 vs 4,556 at 256).
   At 256 its TTFT p50 is 1.1 s, between vLLM (0.7 s) and SGLang (3.0 s), but its p95 is the worst
   (7.9 s). On this card it never beats vLLM below 256.
5. **Memory:** all five sat at 11.1–11.8 GB of the 12 GB card during the runs, including about
   1.4 GB that Windows holds.

## Running it

Takes the chat down for about 45 minutes, so only in an agreed window:

```bash
DRY_RUN=1 serving/engines/run_window.sh     # next to the live chat; proves every start command
serving/engines/run_window.sh               # the real window; restores + verifies vllm-chat on exit
python serving/engines/goodput.py "results/engines/*_summary_*.json"
```

**A note on the first attempt:** the first window failed in two ways. A shell bug started no engine,
and the restore ran under systemd with `HOME` unset. The chat's compose file mounts
`${HOME}/.cache/huggingface`, so compose recreated `vllm-chat` with an empty model cache and it
crash-looped for about 14 minutes. The script now sets `HOME`, checks the mount after the restore, and
must pass a dry run before a window.
