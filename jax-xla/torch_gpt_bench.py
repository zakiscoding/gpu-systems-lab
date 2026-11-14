"""PyTorch side of the JAX comparison: the unchanged modal/fsdp_gpt.py GPT on one GPU.

Same shapes, data (data.py), init and AdamW(0.9, 0.95, wd 0) as jax/jax_gpt.py. fp32 master
params with bf16 autocast (fsdp_gpt's DDP path), fp32 cross-entropy. --compile wraps the model
in torch.compile (TorchInductor + Triton kernels); --fused-adam uses AdamW(fused=True).

    python jax-xla/torch_gpt_bench.py --steps 50 --out results/torch_eager.json
    python jax-xla/torch_gpt_bench.py --compile --steps 50 --out results/torch_compile.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "modal"))
sys.path.insert(0, HERE)
from data import batch_for  # noqa: E402
from fsdp_gpt import GPT, init_weights  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--data-vocab", type=int, default=512)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    model = GPT(args.vocab, args.seq, args.d_model, args.n_layers, args.n_heads, ac=False).to(dev)
    init_weights(model, args.n_layers)
    n_params = sum(x.numel() for x in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
                            fused=args.fused_adam)
    fwd = torch.compile(model) if args.compile else model
    name = "torch.compile" if args.compile else "torch eager"

    def step_fn(x, y):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(x)
        loss = F.cross_entropy(logits.float().view(-1, args.vocab), y.reshape(-1))
        loss.backward()
        opt.step()
        return loss

    losses, times, first_s = {}, [], None
    torch.cuda.reset_peak_memory_stats(dev)
    for step in range(1, args.steps + 1):
        x, y = (torch.from_numpy(a).long().to(dev) for a in batch_for(step, args))
        t0 = time.perf_counter()
        lv = step_fn(x, y).item()  # .item() synchronizes
        dt = time.perf_counter() - t0
        if step == 1:
            first_s = dt  # includes torch.compile's compile time
        losses[step] = lv
        if step > args.warmup:
            times.append(dt)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"  [{name}] step {step:4d}  loss {lv:.4f}  {dt * 1e3:8.2f} ms", flush=True)

    step_s = statistics.median(times)
    tokens = args.batch * args.seq
    rec = {"framework": name, "torch": torch.__version__,
           "device": torch.cuda.get_device_name(dev), "params_millions": round(n_params / 1e6, 2),
           "config": vars(args), "first_step_s": round(first_s, 2), "step_ms": round(step_s * 1e3, 2),
           "step_ms_p10_p90": [round(float(np.percentile(times, 10)) * 1e3, 2),
                               round(float(np.percentile(times, 90)) * 1e3, 2)],
           "tokens_per_s": round(tokens / step_s), "train_tflops": round(6 * n_params * tokens / step_s / 1e12, 2),
           "peak_mem_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
           "loss_first_last": [losses[1], losses[args.steps]], "losses": losses}
    print(f"  [{name}] {rec['params_millions']}M params: {rec['step_ms']} ms/step, "
          f"{rec['tokens_per_s']} tok/s, first step {rec['first_step_s']} s, peak {rec['peak_mem_gib']} GiB", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
