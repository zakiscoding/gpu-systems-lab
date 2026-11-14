# DeepSpeed ZeRO next to FSDP2 on 2x H100

[gpu-systems-lab](../README.md) / `docs/` · [← repo README](../README.md) · [Results](RESULTS.md) · [FSDP write-up](../distributed/README.md)

The same GPT decoder, synthetic data, AdamW settings and timing rules as the FSDP2 runs in
[distributed/](../distributed/README.md), trained with **DeepSpeed ZeRO stages 1, 2 and 3**. All
rows of each table ran in **one container, back to back**, so the hardware and software are
identical.

Measured 2026-11-14 on Modal `H100:2` (H100 80GB HBM3, NVLink): torch 2.11.0+cu128, NCCL 2.28.9,
DeepSpeed 0.20.0, bf16, 15 steps, median step time after 3 warmup steps. Raw JSON:
[results/modal_deepspeed_H100x2_20261113_231702.json](../results/modal_deepspeed_H100x2_20261113_231702.json)
(main run) and
[results/modal_deepspeed_phases_H100x2_20261113_232311.json](../results/modal_deepspeed_phases_H100x2_20261113_232311.json)
(per-phase timing).

Code: [modal/deepspeed_gpt.py](../modal/deepspeed_gpt.py) (the torchrun worker; it imports the model
and data from `fsdp_gpt.py` unchanged) and [modal/deepspeed_train.py](../modal/deepspeed_train.py)
(the Modal app).

## Results

**1.34B parameters** (d 2048, 24 layers, batch 4 × 1024 tokens per GPU):

| per GPU | DDP | FSDP2 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---:|---:|---:|---:|---:|
| peak memory (GiB) | 31.0 | **15.6** | 18.4 | 18.4 | 20.7 |
| step time (ms) | 125.0 | **108.5** | 114.2 | 113.7 | 180.3 |
| tokens/s (both GPUs) | 65.5k | **75.5k** | 71.7k | 72.1k | 45.4k |
| MFU | 26.7% | **30.8%** | 29.2% | 29.4% | 18.5% |

**5.64B parameters** (d 3072, 48 layers, batch 2 × 1024 per GPU). DDP needs 16 B/param, which is
84 GiB per GPU, so it OOMs, as measured before:

| per GPU | FSDP2 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---:|---:|---:|---:|
| peak memory (GiB) | **52.7** | 63.3 | 63.3 | 54.8 |
| step time (ms) | **288.3** | 875.6 | 959.7 | 508.1 |
| tokens/s | **14.2k** | 4.7k | 4.3k | 8.1k |
| MFU | **24.3%** | 8.0% | 7.3% | 13.8% |

All runs train: the loss falls from about 10.8 to 6.7 (1.34B) and from 11.0 to 7.5 (5.64B) in 15
steps, the same trajectory for every method.

### Where the time goes (per-phase timing, extra syncs, separate run)

| 5.64B, ms | forward | backward | optimizer step |
|---|---:|---:|---:|
| ZeRO-1 | 35 | 244 | **945** |
| ZeRO-2 | 35 | 253 | **936** |
| ZeRO-3 | **214** | 261 | 145 |

| 1.34B, ms | forward | backward | optimizer step |
|---|---:|---:|---:|
| ZeRO-1 | 18 | 56 | 46 |
| ZeRO-3 | **92** | 110 | 42 |

## What the numbers say

1. **At 1.34B, ZeRO-1/2 are within 5% of FSDP2.** FSDP2 is still a bit faster and uses 15% less
   memory. ZeRO-1 and ZeRO-2 measured exactly the same peak (18.36 GiB). At this size the gradient
   buffers and DeepSpeed's 500M-element communication buckets matter more than the extra gradient
   sharding that stage 2 adds.
2. **ZeRO-3 pays for all-gathers in the forward pass.** Like FSDP, it shards the parameters, but its
   forward took 92 ms at 1.34B (vs 18 ms for ZeRO-1) and 214 ms at 5.64B. With these settings the
   per-layer all-gathers are not hidden behind compute. FSDP2's per-block prefetch does better on
   the same model: 288 ms for a whole 5.64B step, against ZeRO-3's 508 ms.
3. **At 5.64B, ZeRO-1/2 are bottlenecked by the optimizer step, not communication.** About 940 ms
   of the 1.2 s step (with timing syncs) is `engine.step()`. The setup uses PyTorch's `AdamW` as a
   client optimizer, the same one FSDP2 uses, so no CUDA op had to be compiled. DeepSpeed then
   steps it over one flat 2.8B-element fp32 partition per GPU, and at 63 GiB peak there's little
   memory headroom. Not tested here: DeepSpeed's `FusedAdam` (needs a CUDA toolkit to JIT-compile)
   and smaller buckets. Both are the obvious next experiment, and the ZeRO-1/2 rows shouldn't be
   read as DeepSpeed's best.
4. **Everything fits that DDP couldn't.** All three ZeRO stages and FSDP2 train the 5.64B model on
   two 80 GB GPUs. ZeRO-3 and FSDP2 both shard everything and end up within 2 GiB of each other
   (54.8 vs 52.7 GiB peak).

## Not done

- **Megatron-LM.** Tensor and pipeline parallelism need Megatron-Core plus Transformer Engine
  built against the CUDA toolkit, and a model rewritten into Megatron's layer classes. With only
  2 GPUs, tensor parallel over NVLink would be the interesting case. That's a separate,
  larger piece of work, not attempted here.
- **ZeRO-Offload / ZeRO-Infinity** (CPU/NVMe offload). Modal's gVisor sandbox blocks host memory
  pinning (`cuMemHostRegister`), as found in the earlier MPI work, so offload would be slow or
  fail there.

## Run it

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
modal run modal/deepspeed_train.py                                    # DDP/FSDP/ZeRO 1-3, both sizes, ~3 min
modal run modal/deepspeed_train.py --zero-only --phase-timing         # per-phase breakdown
modal run modal/deepspeed_train.py --tiny --suites mid --steps 6      # pipeline check
```

**Cost:** the main run used 183 s of wall time on 2x H100 and the phase run 178 s, so about
**$0.80** for both at $3.95/GPU-hour, plus a small tiny run. Modal's queue for `H100:2` once took
16 minutes to give a container; that wait isn't billed.
