"""Roadmap step 5: NCCL all_reduce bandwidth test + tiny DDP training loop.

Launched by torchrun (see modal/ddp_nccl.py), e.g.
    torchrun --nproc-per-node=2 modal/ddp_train.py --out /tmp/ddp.json

1. all_reduce sweep 1 MiB .. 1 GiB (fp32), reporting algbw and busbw the same
   way nccl-tests does: algbw = bytes / time, busbw = algbw * 2(n-1)/n.
2. DDP training of a synthetic-data MLP with a fixed per-GPU batch (weak
   scaling), reporting mean step time and samples/s.

Rank 0 writes a JSON record to --out.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def allreduce_sweep(world: int, device: torch.device, warmup: int, iters: int, max_mib: int) -> list[dict]:
    """Time fp32 all_reduce at 1 MiB .. max_mib (powers of 4) across all ranks.

    Each size gets `warmup` untimed calls, then `iters` timed ones; the mean per
    call is MAX-reduced across ranks. Returns one row per size with time (us),
    algbw and busbw (GB/s, 1e9 bytes, as nccl-tests reports them).
    """
    results = []
    for mib in [m for m in (1, 4, 16, 64, 256, 1024) if m <= max_mib]:
        # fp32 = 4 bytes per element
        numel = mib * 2**20 // 4
        buf = torch.ones(numel, dtype=torch.float32, device=device)
        for _ in range(warmup):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        secs = (time.perf_counter() - t0) / iters

        # Slowest rank defines the collective time.
        t = torch.tensor([secs], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        secs = t.item()

        nbytes = numel * 4
        algbw = nbytes / secs / 1e9
        # ring all_reduce moves 2(n-1)/n of the buffer per rank, so busbw is comparable
        # across GPU counts and to the link's raw bandwidth
        busbw = algbw * 2 * (world - 1) / world
        results.append({
            "size_mib": mib,
            "time_us": round(secs * 1e6, 1),
            "algbw_gb_s": round(algbw, 2),
            "busbw_gb_s": round(busbw, 2),
        })
        del buf
    return results


def make_mlp(width: int, depth: int) -> nn.Module:
    """`depth` Linear(width, width)+GELU blocks and a 10-class head: a cheap,
    compute-heavy model whose gradients are big enough to exercise all_reduce."""
    layers = [nn.Linear(width, width), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.GELU()]
    layers.append(nn.Linear(width, 10))
    return nn.Sequential(*layers)


def train(device: torch.device, local_rank: int, world: int, args) -> dict:
    """Time a DDP training loop on one fixed synthetic batch per rank (weak scaling).

    Each rank seeds its own batch (1234 + rank) so ranks see different data, the
    model init is identical (seed 0), and the forward runs under BF16 autocast.
    Step time is the mean over `steps` after warmup, MAX-reduced across ranks;
    samples/s counts the global batch. With world == 1 the model is not wrapped.
    """
    torch.manual_seed(0)
    model = make_mlp(args.width, args.depth).to(device)
    ddp = DDP(model, device_ids=[local_rank]) if world > 1 else model
    opt = torch.optim.SGD(ddp.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    g = torch.Generator(device=device).manual_seed(1234 + dist.get_rank())
    x = torch.randn(args.batch, args.width, device=device, generator=g)
    y = torch.randint(0, 10, (args.batch,), device=device, generator=g)

    def step():
        """One optimizer step; DDP all-reduces gradients during backward()."""
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(ddp(x), y)
        loss.backward()
        opt.step()
        return loss

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - t0) / args.steps

    t = torch.tensor([step_s], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    step_s = t.item()

    params = sum(p.numel() for p in model.parameters())
    return {
        "params_millions": round(params / 1e6, 2),
        "per_gpu_batch": args.batch,
        "global_batch": args.batch * world,
        "steps": args.steps,
        "step_ms": round(step_s * 1000, 3),
        "samples_per_s": round(args.batch * world / step_s, 1),
        "final_loss": round(loss.item(), 4),
    }


def main() -> None:
    """torchrun worker: init NCCL on LOCAL_RANK's GPU, run the all_reduce sweep
    (multi-GPU only, unless --skip-allreduce) and the training benchmark, then
    rank 0 writes the JSON record to --out and prints a summary."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--skip-allreduce", action="store_true")
    p.add_argument("--max-mib", type=int, default=1024)
    p.add_argument("--ar-iters", type=int, default=20)
    p.add_argument("--width", type=int, default=4096)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=10)
    args = p.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    rank = dist.get_rank()

    record = {
        "world_size": world,
        "gpu": torch.cuda.get_device_name(local_rank),
        "torch": torch.__version__,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
    }
    if world > 1 and not args.skip_allreduce:
        record["allreduce"] = allreduce_sweep(world, device, warmup=5, iters=args.ar_iters, max_mib=args.max_mib)
    record["train"] = train(device, local_rank, world, args)

    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)
        if "allreduce" in record:
            print(f"\n  all_reduce fp32, {world} GPUs ({record['gpu']}, NCCL {record['nccl']})")
            print(f"  {'size MiB':>10}{'time us':>12}{'algbw GB/s':>12}{'busbw GB/s':>12}")
            for r in record["allreduce"]:
                print(f"  {r['size_mib']:>10}{r['time_us']:>12}{r['algbw_gb_s']:>12}{r['busbw_gb_s']:>12}")
        tr = record["train"]
        print(f"\n  DDP x{world}: {tr['params_millions']}M params, batch {tr['per_gpu_batch']}/GPU, "
              f"{tr['step_ms']} ms/step, {tr['samples_per_s']} samples/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
