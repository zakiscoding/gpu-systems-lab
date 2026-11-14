# NVIDIA serving stack: Triton + TensorRT-LLM vs vLLM on the RTX 5070

[homelab](https://github.com/zakiscoding/homelab) (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../../README.md) / `serving/triton/` · [← vLLM serving](../README.md) · [Results](../../docs/RESULTS.md)

The same model (Qwen2.5-1.5B-Instruct, bf16), the same prompts and the same benchmark client, served two
ways on one 12 GB RTX 5070:

- **vLLM 0.32.0** (`vllm/vllm-openai:v0.32.0`), the homelab's live chat backend.
- **Triton Inference Server with the TensorRT-LLM backend** (`nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3`:
  TensorRT-LLM 1.2.1, TensorRT 10.14), a compiled bf16 engine behind the in-flight batcher, served through
  Triton's OpenAI-compatible frontend.

**Result:** with the KV cache fixed at the same 171,360 tokens, Triton + TensorRT-LLM gives **10% more
throughput at 256 concurrent requests** (4,556 vs 4,135 output tok/s) and 16% lower time per output token
(46.7 vs 55.4 ms). At 1–64 concurrent requests, the range a chat UI actually sees, Triton is **11–17% slower**
(vLLM 12–21% faster).
TensorRT-LLM also needs about 2 GB more GPU memory for the same cache. Given that memory back as KV cache,
vLLM's default configuration ties Triton at 256 (4,585 vs 4,556 tok/s).

| File | What it does |
|---|---|
| [build_engine.sh](build_engine.sh) | Inside the Triton image: HF checkpoint → TensorRT-LLM checkpoint → engine (`trtllm-build`), then fills the `inflight_batcher_llm` model repository (preprocessing → tensorrt_llm → postprocessing ensemble) |
| [serve_triton.sh](serve_triton.sh) | Starts Triton + the OpenAI frontend on `127.0.0.1:9000` and waits until `/v1/models` answers |
| [bench_compare.py](bench_compare.py) | Runs the concurrency sweep against any OpenAI-compatible server with the pinned `vllm bench serve` client. Writes raw JSON and a summary. It refuses to record a level where any request failed |
| [measured/](measured/) | The five summary JSONs behind the tables below (raw per-level files stayed on the node) |

## Method

Everything ran in WSL2 Ubuntu on the GPU PC's Windows boot (driver 616.56), one server at a time, with the
live chat vLLM stopped for the window (about 55 minutes). Windows itself holds about 1.4 GB of the card.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, bf16 weights on both servers |
| Client | `vllm bench serve` from `vllm/vllm-openai:v0.32.0`, run in a container on the host network, for every target |
| Load | `random` dataset, 512 input / 256 output tokens, `ignore_eos`, request rate `inf` capped by `--max-concurrency` (closed loop) |
| Levels | concurrency 1, 4, 16, 64, 256; `max(16, 4×C)` prompts per level; seed = C, so both servers get identical prompts |
| Warm-up | one untimed run at the top level before measuring |
| Context | max sequence length 4096 on both; vLLM `--max-model-len 4096`, engine `--max_seq_len 4096` |
| Batching | vLLM's continuous batching vs TensorRT-LLM `inflight_fused_batching`, max batch 256, 8192 tokens per step, chunked prefill on both |
| Caching | prefix caching off in vLLM, `enable_kv_cache_reuse:False` in TensorRT-LLM (the random prompts repeat across runs) |
| KV cache | **171,360 tokens on both** (4.58 GiB). TensorRT-LLM got this at `kv_cache_free_gpu_mem_fraction 0.95` of what was free after loading the engine; vLLM was pinned to the same size with `--kv-cache-memory 4913233920` |
| Sampler | vLLM `--generation-config vllm` (ignores Qwen's top-k/top-p defaults), greedy on both |

The vLLM client always sends `"repetition_penalty": 1.0`, a no-op that Triton's frontend rejects with HTTP 422
(the first Triton attempt failed every request). `bench_compare.py` mounts a copy of the client module without
that one line, and every run in the tables used it.

## Results

Output tokens/s and latency. Triton is the mean of two runs; vLLM is the KV-matched run.

| Concurrent requests | vLLM tok/s | Triton + TRT-LLM tok/s | Triton vs vLLM | TTFT p50 vLLM / Triton | TPOT p50 vLLM / Triton | E2E p50 vLLM / Triton | E2E p95 vLLM / Triton |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 161 | 144 | −11% | 34 / 33 ms | 6.1 / 6.8 ms | 1.59 / 1.78 s | 1.61 / 1.80 s |
| 4 | 580 | 505 | −13% | 101 / 99 ms | 6.5 / 7.6 ms | 1.76 / 2.02 s | 1.78 / 2.06 s |
| 16 | 1,888 | 1,566 | −17% | 261 / 346 ms | 7.4 / 8.6 ms | 2.16 / 2.52 s | 2.27 / 2.58 s |
| 64 | 4,160 | 3,637 | −13% | 315 / 860 ms | 14.0 / 14.2 ms | 3.89 / 4.46 s | 4.96 / 4.69 s |
| 256 | 4,135 | **4,556** | **+10%** | 1,551 / 1,138 ms | 55.4 / **46.7** ms | 16.0 / **13.2** s | 19.5 / 20.1 s |

GPU memory in use during the runs (whole card, including Windows): **9.5 GB** for vLLM, **11.5 GB** for Triton.

**Repeatability.** Triton's two runs agree within 3% at every level (4,565 and 4,548 tok/s at 256). vLLM did
not repeat as well at 256: three runs gave 4,094 (first run, original client), 4,585 (default 207,632-token KV
cache) and 4,135 (KV-matched). At 1–64 its runs agree within 7%. So the low-concurrency gap is solid, while
the 256-level edge is real against the matched cache but within vLLM's own spread against its default one.

| vLLM run | KV cache | tok/s at 1 / 16 / 64 / 256 |
|---|---:|---|
| default (`--gpu-memory-utilization 0.80`) | 207,632 tokens | 162 / 1,892 / 4,193 / 4,585 |
| KV-matched (`--kv-cache-memory`) | 171,360 tokens | 161 / 1,888 / 4,160 / 4,135 |
| first run, unpatched client | 207,632 tokens | 155 / 1,868 / 3,938 / 4,094 |

## What it means

- **TensorRT-LLM's decode step is faster under full load.** At 256 in flight its TPOT is 16% lower. That is the
  compiled engine plus its fused attention kernels: when the batch is large, kernel quality is what counts.
- **The Python frontend costs at low load.** At 1–64 requests TensorRT-LLM's TPOT is 1–17% higher than vLLM's,
  but TTFT at 64 is 2.7x higher (860 vs 315 ms). Each request goes through Triton's OpenAI frontend and a
  three-model ensemble (Python tokenizer → engine → Python detokenizer), and vLLM's CUDA graphs keep its small
  batches cheap. The native gRPC endpoint with client-side tokenization, or the C++ `tensorrt_llm_bls` path,
  would remove some of that. This benchmark measures the OpenAI-compatible path both servers expose.
- **Memory matters as much as speed on a 12 GB card.** The engine and its runtime need about 2 GB more than vLLM
  for the same cache. Spent as KV cache, that memory lets vLLM hold more sequences, which is why its default
  setup matches Triton at 256.
- **For the homelab's chat traffic (a few requests at a time), vLLM stays the backend.** TensorRT-LLM would pay
  off for a batch or offline workload that keeps hundreds of requests in flight.
- **Build cost:** checkpoint conversion took 8 s (166 s wall including container start and imports), the
  engine build 20 s, and the engine is 3.4 GB on disk. The image is 42 GB against vLLM's 33 GB.

## Reproduce

```bash
# GPU PC, WSL Ubuntu. Stop anything else on the GPU first (docker compose stop vllm in serving/).
docker pull nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3
mkdir -p ~/nvidia-serving/work
docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
  -v ~/nvidia-serving/work:/work -v "$PWD/serving/triton:/scripts:ro" \
  --entrypoint bash nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3 /scripts/build_engine.sh
bash serving/triton/serve_triton.sh
python3 serving/triton/bench_compare.py --label triton-trtllm --base-url http://127.0.0.1:9000 --model ensemble
docker rm -f triton-trtllm
```

For vLLM, start `vllm/vllm-openai:v0.32.0` on another port with the flags in the method table, then run
`bench_compare.py --label vllm --base-url http://127.0.0.1:8001 --model qwen2.5-1.5b-instruct`.

## Limits and what wasn't tried

- **NVIDIA NIM** wasn't used. NIM containers need an NGC API key tied to the user's NVIDIA account, and I
  didn't confirm a NIM profile exists for a 12 GB GeForce card. Triton + TensorRT-LLM, used here directly, is
  the serving stack NIM's LLM containers package.
- **One GPU, one model size.** No tensor parallelism, NVLink or MIG: the 5070 has one GPU and no MIG.
- **WSL2 on Windows**, not the bare-metal Ubuntu boot. vLLM ran about 5% slower here than in the earlier
  Ubuntu-boot runs in [RESULTS](../../docs/RESULTS.md) (4,193 vs 4,430 tok/s at 64).
- **bf16 only.** FP8 and INT4 engines (TensorRT-LLM's main advantage on Blackwell) are a follow-up, next to the
  repo's vLLM quantization runs.
- Engine defaults otherwise: no CUDA graphs in the TensorRT-LLM executor and no speculative decoding.
