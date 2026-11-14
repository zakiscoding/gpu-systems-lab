# JAX/XLA port of the GPT training benchmark

[gpu-systems-lab](../README.md) / `jax-xla/` · [← repo README](../README.md) · [Results](../docs/RESULTS.md)

The repo's PyTorch GPT benchmark ([modal/fsdp_gpt.py](../modal/fsdp_gpt.py)), ported to JAX + optax
and compared on the same RTX 5070. The port is checked against the PyTorch model, not just timed:
it computes the same function, the frameworks train on identical batches, and their losses agree.

| File | What it is |
|---|---|
| [jax_gpt.py](jax_gpt.py) | The port: same architecture, GPT-2 init, AdamW(0.9, 0.95), bf16 compute on fp32 masters. The whole train step (forward, backward, AdamW) is one `jax.jit` XLA program. `--attn xla\|cudnn`, `--data-parallel` (shards the batch over all local devices with `jax.sharding`, for TPU hosts) |
| [torch_gpt_bench.py](torch_gpt_bench.py) | PyTorch side: the unchanged `fsdp_gpt.GPT` on one GPU, eager or `--compile`, optional `--fused-adam` |
| [data.py](data.py) | The synthetic batches, shared by both, so they see identical tokens |
| [test_parity.py](test_parity.py) | Copies a PyTorch model's weights into JAX and checks logits, loss and gradients match (CPU, fp32) |
| [tpu_colab.ipynb](tpu_colab.ipynb) | Runs `jax_gpt.py` on a free Colab/Kaggle TPU |

## Results: RTX 5070, 136M-parameter GPT

Measured 2026-11-14 in WSL2 on the GPU PC (driver 616.56): torch 2.11.0+cu128, JAX/jaxlib 0.12.0.1
(CUDA 12 plugin), optax 0.2.9. The model is d 768, 12 layers, 12 heads, seq 1024, vocab 32768. The
`vllm-chat` server stayed up on the same GPU (idle, ~5.4 GB). Each run is 60 steps with the median
step after 10 warmup steps; batch-4 rows are 3 separate process runs. Raw JSON:
[results/jax/](../results/jax/).

**Batch 4 × 1024 tokens:**

| | step (ms), 3 runs | mean ± sd | tokens/s | model TFLOPS | peak memory |
|---|---|---:|---:|---:|---:|
| PyTorch eager | 84.64 / 84.49 / 84.45 | 84.53 ± 0.08 | 48.4k | 39.5 | 4.79 GiB |
| torch.compile | 81.71 / 81.71 / 82.31 | 81.91 ± 0.28 | 50.1k | 40.9 | 4.79 GiB |
| torch.compile + fused AdamW | 73.70 / 73.42 / 73.42 | 73.51 ± 0.13 | 55.6k | 45.4 | 4.79 GiB |
| **JAX, cuDNN flash attention** | 68.60 / 68.87 / 69.22 | **68.90 ± 0.25** | **59.7k** | **48.8** | **3.98 GiB** |
| JAX, XLA attention | OOM (it asked for one 5.75 GiB buffer) | | | | |

**Batch 2** (single runs; here the XLA-attention variant fits):

| | step (ms) | tokens/s | peak memory |
|---|---:|---:|---:|
| PyTorch eager | 53.45 | 38.3k | 3.29 GiB |
| torch.compile | 52.20 | 39.2k | 3.28 GiB |
| JAX, XLA attention | 53.67 | 38.2k | 4.59 GiB |
| **JAX, cuDNN flash attention** | **39.06** | **52.4k** | **2.89 GiB** |

Model TFLOPS = 6 × params × tokens / step time (attention FLOPs not counted). The 5070's dense BF16
peak measured in this repo is 67.7 TFLOPS, so JAX + cuDNN reaches 72% of it.

Every run trains the same way. Loss goes from about 10.5 to 6.25 in 60 steps at batch 4: PyTorch
6.258, JAX 6.253. The two start from different random inits, since PyTorch's and JAX's RNGs
differ.

## What the numbers say

1. **The attention kernel is the biggest single factor.** Without a fused attention kernel, XLA
   materializes the 12 × 1024 × 1024 score matrices for every layer. That cost 40% more memory than
   PyTorch at batch 2 and ran out of memory at batch 4. With `implementation="cudnn"` (cuDNN's
   flash attention, the counterpart of PyTorch's SDPA flash kernel), JAX is the fastest and
   smallest of all.
2. **Whole-step compilation helps, and so does fusing the optimizer.** JAX's jit compiles forward,
   backward and the AdamW update into one XLA program. `torch.compile` here only compiles the
   model's forward/backward (3% faster than eager). Swapping in `AdamW(fused=True)` cut another
   8.4 ms per step (10%). That's how much the unfused optimizer was costing at 136M parameters.
   Even then, JAX is 6% faster than the best PyTorch configuration.
3. **Compile time is the price.** JAX took 17 s to compile, torch.compile 15–31 s on its first run
   and 1.4–3.6 s once its cache was warm. Eager PyTorch starts in under a second.
4. **The run-to-run noise is small.** The spread across 3 runs is 0.1–0.4% of the mean, much
   smaller than any difference above.

## TPU

[tpu_colab.ipynb](tpu_colab.ipynb) runs the same script on a free TPU: one core at the same
shapes, then data-parallel over every core on the host with `jax.sharding`. It needs a Google
sign-in, so it runs in the user's own Colab/Kaggle session. Results go here once it has run.

## Run it

```bash
# GPU (WSL2 / Linux), from the repo root; JAX must not preallocate when vLLM shares the card
export XLA_PYTHON_CLIENT_PREALLOCATE=false
C="--d-model 768 --n-layers 12 --n-heads 12 --seq 1024 --batch 4 --vocab 32768 --steps 60 --warmup 10"
python jax-xla/jax_gpt.py $C --attn cudnn --out jax.json
python jax-xla/torch_gpt_bench.py $C --compile --fused-adam --out torch.json
JAX_PLATFORMS=cpu python -m pytest jax-xla/test_parity.py     # 3 passed
```

Setup on the WSL node: `pip install "jax[cuda12]" optax` in a venv (jax 0.12.0.1). `torch.compile`
also needs `python3-dev`: Triton compiles a small C launcher against `Python.h`, and without it
every compiled run fails.
