"""The fsdp_gpt.py GPT decoder trained with DeepSpeed ZeRO stage 1, 2 or 3 (torchrun worker).

Same model, same synthetic data (fsdp_gpt.batch_for), same AdamW hyperparameters and the same
timing/MFU rules as modal/fsdp_gpt.py, so a ZeRO run lines up row for row with the DDP/FSDP2
runs. Launched by modal/deepspeed_train.py, or locally on CPU for a logic check is NOT
supported (DeepSpeed ZeRO needs CUDA).

Memory model per GPU with bf16 + AdamW (world = W):
  ZeRO-1: bf16 params 2 + bf16 grads 2 + (fp32 master 4 + Adam 8) / W     -> 10 B/param at W=2
  ZeRO-2: bf16 params 2 + (bf16 grads 2 + fp32 master 4 + Adam 8) / W     ->  9 B/param
  ZeRO-3: (2 + 2 + 4 + 8) / W                                             ->  8 B/param
  (DDP, for reference: 16 B/param; FSDP2 here: 16 / W = 8 B/param)

The optimizer is torch.optim.AdamW handed to deepspeed.initialize as a client optimizer
(the FSDP runs use the same one), so no DeepSpeed CUDA op has to be JIT-compiled.
"""

import time

T_PROC_START = time.time()

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fsdp_gpt import GPT, H100_BF16_DENSE_TFLOPS, batch_for, init_weights


def ds_config(stage: int, micro_batch: int, world: int) -> dict:
    """DeepSpeed config: bf16, ZeRO `stage`, no grad accumulation, no clipping (the FSDP
    runs don't clip either). Bucket sizes are DeepSpeed's defaults made explicit."""
    zero = {"stage": stage, "overlap_comm": True, "contiguous_gradients": True,
            "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8}
    if stage == 3:
        zero.update(stage3_prefetch_bucket_size=5e8, stage3_param_persistence_threshold=1e5,
                    stage3_max_live_parameters=1e9, stage3_gather_16bit_weights_on_model_save=False)
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": 1,
        "train_batch_size": micro_batch * world,
        "bf16": {"enabled": True},
        "zero_optimization": zero,
        "zero_allow_untested_optimizer": True,
        "gradient_clipping": 0.0,
        "wall_clock_breakdown": False,
        "steps_per_print": 10**9,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--data-vocab", type=int, default=512)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=24)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--ac", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--phase-timing", action="store_true",
                   help="cuda-synchronize between forward / backward / step and time each (adds syncs)")
    args = p.parse_args()

    import deepspeed

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    record = {"mode": f"zero{args.stage}", "world_size": world, "gpu": torch.cuda.get_device_name(local_rank),
              "torch": torch.__version__, "deepspeed": deepspeed.__version__,
              "config": {k: v for k, v in vars(args).items() if k != "out"},
              "gpu_total_gib": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2)}
    cfg = ds_config(args.stage, args.batch, world)
    losses, step_times, status, phases = {}, [], "ok", []
    try:
        torch.manual_seed(args.seed)
        # ZeRO-3: zero.Init partitions each parameter as it is created, so no rank ever holds the
        # whole model (the analogue of FSDP2's meta-device build). Stages 1/2 replicate params anyway.
        ctx = deepspeed.zero.Init(config_dict_or_path=cfg, dtype=torch.bfloat16) if args.stage == 3 else torch.device(device)
        with ctx:
            model = GPT(args.vocab, args.seq, args.d_model, args.n_layers, args.n_heads, args.ac)
        if args.stage == 3:
            # init each partitioned param in place: gather it, initialize on rank 0's RNG, re-partition
            for name, prm in model.named_parameters():
                with deepspeed.zero.GatheredParameters([prm], modifier_rank=0):
                    if rank == 0:
                        _init_one(name, prm, args.n_layers)
        else:
            init_weights(model, args.n_layers)
        n_params = sum(getattr(prm, "ds_numel", prm.numel()) for prm in model.parameters())
        record["params_billions"] = round(n_params / 1e9, 4)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=cfg)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        record["mem_after_init_gib"] = round(torch.cuda.memory_allocated(device) / 2**30, 2)

        engine.train()
        for step in range(1, args.steps + 1):
            x, y = batch_for(step, rank, args, device)
            t0 = time.perf_counter()
            logits = engine(x)
            loss = F.cross_entropy(logits.float().view(-1, args.vocab), y.reshape(-1))
            if args.phase_timing:
                torch.cuda.synchronize(); t1 = time.perf_counter()
            engine.backward(loss)
            if args.phase_timing:
                torch.cuda.synchronize(); t2 = time.perf_counter()
            engine.step()
            if args.phase_timing:
                torch.cuda.synchronize(); t3 = time.perf_counter()
                if step > args.warmup:
                    phases.append((t1 - t0, t2 - t1, t3 - t2))
            avg = loss.detach().clone()
            dist.all_reduce(avg)
            lv = avg.item() / world
            dt = time.perf_counter() - t0
            losses[step] = lv
            if step > args.warmup:
                step_times.append(dt)
            if rank == 0:
                print(f"  [zero{args.stage}] step {step:4d}  loss {lv:.4f}  {dt * 1e3:8.1f} ms", flush=True)
    except torch.OutOfMemoryError as e:
        status = "oom"
        record["error"] = str(e).split("\n")[0][:300]
    record["status"] = status
    record["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    record["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 2)

    if status == "oom":
        if rank == 0 and args.out:
            Path(args.out).write_text(json.dumps(record, indent=2))
            print(f"  [zero{args.stage}] OOM: {record['error']}", flush=True)
        os._exit(0)

    t = torch.tensor([statistics.median(step_times)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    step_s = t.item()
    tokens = args.batch * args.seq * world
    record["step_ms"] = round(step_s * 1e3, 2)
    record["tokens_per_s"] = round(tokens / step_s)
    record["mfu"] = round(6 * n_params * tokens / step_s / (world * H100_BF16_DENSE_TFLOPS * 1e12), 4)
    m = torch.tensor([record["peak_alloc_gib"], record["peak_reserved_gib"]], device=device)
    dist.all_reduce(m, op=dist.ReduceOp.MAX)
    record["peak_alloc_gib"], record["peak_reserved_gib"] = [round(v, 2) for v in m.tolist()]
    record["losses"] = {str(k): v for k, v in losses.items()}
    if phases:
        record["phase_ms_median"] = {k: round(statistics.median(p[i] for p in phases) * 1e3, 2)
                                     for i, k in enumerate(("forward", "backward", "optimizer_step"))}
    if rank == 0:
        if args.out:
            Path(args.out).write_text(json.dumps(record, indent=2))
        print(f"  [zero{args.stage}] {record['params_billions']}B x{world}: peak alloc {record['peak_alloc_gib']} GiB, "
              f"step {record['step_ms']} ms, {record['tokens_per_s']} tok/s, MFU {record['mfu']}", flush=True)
    dist.destroy_process_group()


@torch.no_grad()
def _init_one(name: str, p: torch.Tensor, n_layers: int) -> None:
    """init_weights() for a single (gathered) parameter, same rules."""
    import math
    if p.dim() == 1:
        p.fill_(0.0 if name.endswith("bias") else 1.0)
    else:
        std = 0.02 / math.sqrt(2 * n_layers) if name.endswith(("proj.weight", "fc2.weight")) else 0.02
        torch.nn.init.normal_(p, mean=0.0, std=std)


if __name__ == "__main__":
    main()
