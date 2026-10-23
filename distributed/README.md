# Sharded training (FSDP2) and fault tolerance

This section trains a GPT-style decoder on 2x H100 (NVLink) on Modal. It covers three things:

1. DDP vs FSDP: per-GPU memory and step time, plus a 5.6B model that does not fit with DDP but trains with FSDP.
2. Sharded checkpoints with `torch.distributed.checkpoint` (DCP), and a check that resuming reproduces the same loss curve.
3. Fault tolerance with `torchrun --max-restarts`. One rank is SIGKILLed mid-run, the group restarts from the latest checkpoint, and the cost of the failure is measured.

Everything below was measured on 2026-10-23 with torch 2.11.0+cu128 and NCCL 2.28.9 on `H100:2`. Raw JSON is in
`results/modal_fsdp_H100x2_20261023_131833.json` and `results/modal_elastic_H100x2_20261023_132756.json`
(gitignored).

## Files

| File | What it does |
|---|---|
| [`modal/fsdp_gpt.py`](../modal/fsdp_gpt.py) | The torchrun worker. GPT decoder (pre-LN, SDPA causal attention, 4x GELU MLP), synthetic tokens, `--mode ddp\|fsdp`, `--ac`, DCP save/resume, `--kill-at-step`, JSON-lines event log. Also runs on CPU with gloo (`--device cpu`) |
| [`modal/fsdp_train.py`](../modal/fsdp_train.py) | Modal app. Runs the suites `compare`, `big`, `resume` and `resume_det` via torchrun in one multi-GPU container and writes `results/modal_fsdp_*.json` |
| [`modal/elastic_train.py`](../modal/elastic_train.py) | Modal app. Runs `torchrun --max-restarts=3` with a baseline and kill runs at two checkpoint intervals, rebuilds the failure timeline from the events, and writes `results/modal_elastic_*.json` |

Implementation choices:
- FSDP uses FSDP2 (`torch.distributed.fsdp.fully_shard`). Each transformer block is sharded, then the root module. Mixed precision is `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`. The model is built on the `meta` device, then `to_empty` and init run, so no rank ever materializes the full model.
- DDP uses fp32 params with `torch.autocast(bf16)` and `gradient_as_bucket_view=True`. With that flag, grads alias the all-reduce buckets and do not need a second copy.
- Both modes use AdamW with fp32 master weights. Optimizer state is 16 bytes per parameter: 4 for params, 4 for grads and 8 for the two Adam moments.
- Each batch is a pure function of (seed, step, rank). Every sequence repeats a random 32-token motif drawn from 512 token ids, so the loss falls quickly. A resumed run sees exactly the batches an uninterrupted run would have seen.
- Checkpoints go to the container's local disk (`/tmp`). `latest.json` is published by rank 0 only after every shard is written and a barrier passes, so a crash during a save keeps the previous checkpoint.

## Run

From the repo root, on the Windows laptop:

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore::DeprecationWarning
PY=/c/Users/cashh/.venvs/gpu-lab/Scripts/python.exe

$PY -m modal run modal/fsdp_train.py                      # all suites, H100:2, ~2.7 min wall
$PY -m modal run modal/fsdp_train.py --suites compare     # just DDP vs FSDP
$PY -m modal run modal/fsdp_train.py --tiny               # small shapes, pipeline check
$PY -m modal run modal/elastic_train.py                   # kill after step 48/60, K = 10 and 25, ~2 min wall
$PY -m modal run modal/elastic_train.py --intervals 5,20 --kill-at 37 --steps 50
```

Logic test without a GPU (gloo on CPU; this is how the code was checked on `gpu-ts` before any Modal run):

```bash
torchrun --standalone --nproc-per-node=2 modal/fsdp_gpt.py --device cpu --mode fsdp \
  --d-model 64 --n-layers 2 --n-heads 4 --seq 64 --vocab 256 --batch 2 --steps 8 --ckpt-dir ck --ckpt-final
torchrun --standalone --nproc-per-node=2 --max-restarts=2 modal/fsdp_gpt.py --device cpu \
  --d-model 64 --n-layers 2 --n-heads 4 --seq 64 --vocab 256 --batch 2 --steps 16 \
  --ckpt-dir ck2 --ckpt-every 5 --resume --kill-at-step 12 --events ev.jsonl
```

## 1. DDP vs FSDP on 2x H100

The model has vocab 32768 and sequence length 1024. Times are the median step after 3 warmup steps (15 steps total), taking the slower of the two ranks.
MFU is 6·N·tokens / (time · 989 TFLOPS dense BF16), with attention FLOPs excluded.

| Run | Params | Batch/GPU | Model state / GPU | Peak allocated / GPU | Step | Tokens/s (2 GPUs) | MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| DDP | 1.34B | 4×1024 | 20.03 GiB | 31.0 GiB | 125.8 ms | 65.1k | 26.6% |
| FSDP2 | 1.34B | 4×1024 | 10.02 GiB | 15.6 GiB | 113.6 ms | 72.1k | 29.4% |
| FSDP2 + activation ckpt | 1.34B | 4×1024 | 10.02 GiB | 12.8 GiB | 145.4 ms | 56.3k | 23.0% |
| **DDP** | **5.64B** | 2×1024 | needs 84.05 GiB | **OOM** at 77.45 GiB (GPU has 79.17) | — | — | — |
| **FSDP2** | **5.64B** | 2×1024 | **42.03 GiB** | **52.7 GiB** | 293.4 ms | 14.0k | 23.9% |

An earlier run (`results/modal_fsdp_H100x2_20261023_130600.json`) gave the same memory numbers to within 0.01 GiB and step times within 3%:
DDP 125.4, FSDP 112.1, FSDP+AC 142.6 and FSDP 5.6B 298.2 ms.

The 5.6B model genuinely does not fit with DDP. No memory cap was applied: `--mem-frac` exists but was not used. DDP ran out of memory in its first step, with 77.45 GiB allocated on an 80 GB H100.

### Memory math (per GPU, N = 2 GPUs, P = parameters)

| | DDP | FSDP (full shard) | 1.34B: DDP / FSDP | 5.64B: DDP / FSDP |
|---|---|---|---:|---:|
| fp32 params | 4P | 4P/N | 5.01 / 2.50 GiB | 21.0 / 10.5 GiB |
| fp32 grads | 4P | 4P/N | 5.01 / 2.50 GiB | 21.0 / 10.5 GiB |
| AdamW m, v (fp32) | 8P | 8P/N | 10.02 / 5.01 GiB | 42.0 / 21.0 GiB |
| **Total state** | **16P** | **16P/N** | **20.03 / 10.02 GiB** | **84.05 / 42.03 GiB** |
| Transient | bf16 autocast copies of the weights (2P) + activations | one gathered bf16 block at a time (plus prefetch) + activations | | |

The "Model state" column in the results table is measured by summing the local tensor sizes of params, grads and optimizer state. It matches 16P and 16P/2 exactly.
For the 1.34B model, the gap between state and peak is 11.0 GiB for DDP and 5.6 GiB for FSDP. Both include activations and the fp32 logits for 4096 tokens. The DDP gap also includes the 2.5 GiB of bf16 weight copies that autocast caches during forward. Activation checkpointing cuts FSDP peak by another 2.8 GiB and costs 28% more step time, because the forward is recomputed in backward.

With FSDP, per-GPU state scales as 16P/N. On N GPUs the largest trainable model grows roughly N-fold, until activations and the one-layer gather buffer dominate. DDP is capped at about (80 GB − activations) / 16 B ≈ 4.5B params on an 80 GB card, whatever N is.

### Throughput cost of sharding

On NVLink, the cost of sharding was not visible: FSDP2 was 10% faster than DDP on the 1.34B model. Per step, each rank moves about the same volume either way:
- DDP: an fp32 gradient all-reduce, 2(N−1)/N · 4P ≈ 5.4 GB.
- FSDP: a bf16 all-gather in forward plus another in backward (2 × 2P/N ≈ 2.7 GB in total) and an fp32 reduce-scatter (4P/N ≈ 2.7 GB).

At 329 GB/s busbw (from [modal/ddp_nccl.py](../modal/ddp_nccl.py)), that is ~16 ms of traffic per step, and both modes overlap it with compute. The precision paths also differ:
- DDP with autocast recasts every fp32 weight to bf16 on every forward.
- FSDP2 casts the local shard once, then gathers bf16, which is half the bytes.

That likely explains why FSDP came out ahead, so treat this comparison as "DDP+autocast vs FSDP2 mixed precision", not as pure communication cost. On PCIe or across nodes the picture changes. The same repo measured 6.65 GB/s on 2x L4, where 5.4 GB per step would take ~0.8 s.

## 2. Checkpoint and resume (DCP, FSDP2, 0.22B model, 2 GPUs)

Each test has three runs:
1. An uninterrupted 30-step run.
2. A 15-step run that ends with a DCP save of the sharded model and optimizer (`get_state_dict` → `dcp.save`).
3. A fresh process group that loads the checkpoint (`dcp.load` → `set_state_dict`) and trains steps 16 to 30.

| | Default kernels (flash SDPA) | `--deterministic` (math SDPA, `CUBLAS_WORKSPACE_CONFIG`) |
|---|---:|---:|
| Max abs. loss diff, steps 1–15 (two identical runs, no resume) | 4.6e-5 | **0** |
| Max abs. loss diff, steps 16–30 (resumed vs uninterrupted) | 7.6e-5 | **0** |
| Bit-exact steps after resume | 0 / 15 | **15 / 15** |
| Loss at step 16 / 30 (uninterrupted) | 6.5059 / 6.3482 | 6.5060 / 6.3482 |
| Checkpoint size | 2.45 GiB | 2.45 GiB |
| Save / load time | 1.93 s / 1.46 s | 2.03 s / 1.61 s |
| Step time | 50.1 ms | 113.8 ms |

Resume is exact. With default kernels, the resumed curve differs from the uninterrupted one by the same ~1e-4 that two identical uninterrupted runs already differ by. That noise comes from flash-attention backward, which accumulates dQ with atomics. With deterministic kernels both differences are exactly 0, at every step after the resume.
The deterministic mode costs 2.3x step time here because of the math attention path, so it is for verification, not for production.
The checkpoint is 12 B/param (fp32 params plus two Adam moments; grads are not saved) = 0.22B × 12 B = 2.45 GiB. It was written at about 1.25 GiB/s to the container disk.

## 3. Fault tolerance: torchrun restart after a SIGKILL (0.37B model, FSDP2, 2 GPUs)

The test runs `torchrun --standalone --nproc-per-node=2 --max-restarts=3 --monitor-interval=0.1` for 60 steps with `--deterministic`, so losses can be compared bit for bit.
On the first attempt, rank 1 calls `os.kill(getpid(), SIGKILL)` right after step 48. torchrun detects the dead worker, SIGTERMs rank 0 (stuck in an NCCL collective) and starts both workers again. The new workers load `latest.json` and train to step 60.

| | Baseline (no ckpt, no failure) | K = 10 | K = 25 |
|---|---:|---:|---:|
| Checkpoint step used for restart | — | 40 | 25 |
| Recomputed steps (48 − ckpt) | — | 8 | 23 |
| Kill → new worker process starts (torchrun detect + respawn) | — | 1.17 s | 1.01 s |
| Worker startup (import torch, NCCL init, build FSDP model) | — | 4.60 s | 4.40 s |
| DCP load | — | 2.44 s | 2.99 s |
| Recompute (steps × 215 ms) | — | 1.72 s | 4.95 s |
| **Time lost to the failure** (kill → back at step 48) | — | **10.95 s** | **14.36 s** |
| Checkpoint saves × mean save time | — | 6 × 2.39 s = 14.3 s | 2 × 3.35 s = 6.7 s |
| Training wall time (step 1 → 60) | 12.75 s | 37.76 s | 33.43 s |
| Extra wall time vs baseline | — | +25.0 s | +20.7 s |
| Final loss / max diff vs baseline after restart | 6.333405 | 6.333405 / **0** | 6.333405 / **0** |

The job finished after exactly one restart in both runs. Every post-restart loss is bit-identical to the run without a failure.
Most of the time lost is a fixed restart cost of about 8–9 s (respawn, startup and load), and that cost is independent of K. The recompute term grows with K: on average K/2 steps per failure.

### Checkpoint interval trade-off

Let δ be the cost of one save, K the interval in steps, t the step time and M the mean time between failures. Then:
- checkpoint overhead per step = δ / K
- expected loss per failure = R + (K/2)·t, where R ≈ 8–9 s is the fixed restart cost

Minimizing the total gives Young's approximation, interval ≈ √(2δM) of wall time. With the measured δ ≈ 2.4 s:

| MTBF M | Optimal interval √(2δM) | In steps at 215 ms |
|---|---:|---:|
| 10 min | 54 s | ~250 |
| 1 h | 131 s | ~610 |
| 24 h | 11 min | ~3,000 |

In this experiment (one forced failure every 13 s of training), K = 10 spent more on saves (14.3 s) than it saved in recompute (3.2 s). The synchronous `dcp.save` stalls training for about 11 steps' worth of time.
Two levers lower the cost:
- `dcp.async_save`, which moves the write off the critical path.
- Saving less often to local disk and occasionally to durable storage.

## Findings and caveats

- **Stale NCCL bootstrap after a restart (found and fixed).** The first elastic run on 2x H100 failed on every restart attempt. The restarted rank 1 spent 64 s retrying `socketPollConnect ... Connection refused` to rank 0's NCCL bootstrap port.
  - torchrun keeps its agent TCPStore across restarts, and workers connect to it without a per-attempt prefix. This torch version's `_create_c10d_store` comments that "we create a new TCPStore for every retry", but the evidence suggests the restarted rank 1 read the dead attempt's NCCL unique id before the new rank 0 overwrote it.
  - The fix is in `fsdp_gpt.py`: `init_process_group(store=PrefixStore(f"fsdp_gpt/attempt_{TORCHELASTIC_RESTART_COUNT}/", ...))`. After that, both kill runs restarted cleanly. The CPU/gloo restart test passed even without the fix, probably because gloo ranks initialize fast enough to win the race.
- Checkpoints live on the container disk. That survives a process crash, which is what torchrun restarts handle, but not the loss of the container or node. For node failure, write to a `modal.Volume` or shared filesystem and restart with multi-node rendezvous (`--rdzv-backend=c10d`). Neither was tested here.
- `nvidia-smi topo -m` does not work in the gVisor sandbox, so the 2x H100 NVLink figure comes from the earlier NCCL bandwidth test, not from topology output.
- The loss falls quickly (10.6 → 6.3 in 30 steps) only because the synthetic data uses 512 token ids. These runs measure systems behaviour, not model quality.

## Modal cost

There were five 2x H100 runs, with about 13.6 min of container time in total:
- two full `fsdp_train.py` runs (2.0 and 2.7 min)
- one elastic run that hung on the stale-NCCL bug (~5 min, stopped manually)
- one elastic run whose results were lost to a client-side deserialization bug (the payload contained `torch.__version__`, now `str()`-ed; ~1.8 min)
- the final elastic run (2.1 min)

At about $7.9/h for two H100s, that is roughly **$1.8**.
