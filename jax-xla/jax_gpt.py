"""The modal/fsdp_gpt.py GPT decoder, ported to JAX + optax, as a single-host training benchmark.

Same architecture as the PyTorch model (pre-LayerNorm blocks, causal SDPA attention, 4x GELU MLP,
no linear biases, learned positions, untied LM head), the same GPT-2 style init, the same
synthetic "repeated 32-token motif" data and the same AdamW(0.9, 0.95, wd 0) update. Mixed
precision matches the PyTorch side too: fp32 master params, bf16 matmuls, fp32 loss.

The whole train step (forward, backward, AdamW) is one jax.jit-compiled XLA program. With
--data-parallel the batch is sharded over every local device with jax.sharding (one TPU host has
4-8 cores; params are replicated and XLA inserts the gradient all-reduce itself).

    python jax-xla/jax_gpt.py --steps 50 --out results/jax_gpt.json          # GPU or TPU or CPU
    python jax-xla/jax_gpt.py --data-parallel --batch 32                       # all local devices

Prints and writes: median step time (after --warmup), tokens/s, compile time, peak device memory.
"""

import argparse
import json
import math
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------------------------------------------------------------- model

def init_params(key, vocab, seq, d, layers):
    """GPT-2 style init mirroring fsdp_gpt.init_weights: N(0, 0.02), residual output projections
    (proj, fc2) scaled by 1/sqrt(2*layers), LayerNorm weight 1 / bias 0. Linear weights are stored
    (in, out), i.e. transposed relative to torch.nn.Linear."""
    keys = iter(jax.random.split(key, 4 + 4 * layers))
    n = lambda shape, std=0.02: std * jax.random.normal(next(keys), shape, jnp.float32)
    resid = 0.02 / math.sqrt(2 * layers)
    blocks = []
    for _ in range(layers):
        blocks.append({
            "ln1_w": jnp.ones(d), "ln1_b": jnp.zeros(d),
            "qkv": n((d, 3 * d)), "proj": n((d, d), resid),
            "ln2_w": jnp.ones(d), "ln2_b": jnp.zeros(d),
            "fc1": n((d, 4 * d)), "fc2": n((4 * d, d), resid),
        })
    return {"tok": n((vocab, d)), "pos": n((seq, d)), "blocks": blocks,
            "lnf_w": jnp.ones(d), "lnf_b": jnp.zeros(d), "head": n((d, vocab))}


def layer_norm(x, w, b, eps=1e-5):
    """torch.nn.LayerNorm semantics (biased variance, eps 1e-5), computed in fp32."""
    xf = x.astype(jnp.float32)
    mu = xf.mean(-1, keepdims=True)
    var = ((xf - mu) ** 2).mean(-1, keepdims=True)
    return ((xf - mu) * jax.lax.rsqrt(var + eps) * w + b).astype(x.dtype)


def forward(params, idx, heads, dtype=jnp.bfloat16, attn=None):
    """(batch, seq) int32 -> (batch, seq, vocab) logits in `dtype`. Params are cast to `dtype`
    for the matmuls (fp32 masters stay untouched), like FSDP's MixedPrecisionPolicy."""
    p = jax.tree.map(lambda a: a.astype(dtype), params)
    B, T = idx.shape
    x = p["tok"][idx] + p["pos"][:T]
    for blk in p["blocks"]:
        h = layer_norm(x, blk["ln1_w"], blk["ln1_b"])
        q, k, v = jnp.split(h @ blk["qkv"], 3, axis=-1)
        D = q.shape[-1] // heads
        q, k, v = (t.reshape(B, T, heads, D) for t in (q, k, v))           # BTNH layout
        # attn=None: XLA's own lowering (materializes the T x T scores); "cudnn": cuDNN's fused
        # flash attention, the counterpart of PyTorch's SDPA flash kernel
        a = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation=attn)
        x = x + a.reshape(B, T, -1) @ blk["proj"]
        h = layer_norm(x, blk["ln2_w"], blk["ln2_b"])
        x = x + jax.nn.gelu(h @ blk["fc1"], approximate=False) @ blk["fc2"]
    x = layer_norm(x, p["lnf_w"], p["lnf_b"])
    return x @ p["head"]


def loss_fn(params, x, y, heads, dtype, attn=None):
    logits = forward(params, x, heads, dtype, attn).astype(jnp.float32)    # fp32 softmax / CE
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


# ---------------------------------------------------------------- data

from data import batch_for  # noqa: E402  (shared with torch_gpt_bench.py: identical batches)


# ---------------------------------------------------------------- main

def peak_bytes():
    """Peak bytes in use on device 0 if the backend reports it (GPU/TPU), else None."""
    try:
        st = jax.devices()[0].memory_stats() or {}
        return st.get("peak_bytes_in_use")
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--data-vocab", type=int, default=512)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--batch", type=int, default=4, help="global batch (split over devices with --data-parallel)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp32", action="store_true", help="compute in fp32 instead of bf16")
    p.add_argument("--data-parallel", action="store_true", help="shard the batch over all local devices")
    p.add_argument("--attn", choices=["xla", "cudnn"], default="xla",
                   help="attention kernel: XLA lowering (any backend) or cuDNN flash (NVIDIA GPUs)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dtype = jnp.float32 if args.fp32 else jnp.bfloat16
    devs = jax.devices()
    params = init_params(jax.random.PRNGKey(args.seed), args.vocab, args.seq, args.d_model, args.n_layers)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    opt = optax.adamw(args.lr, b1=0.9, b2=0.95, weight_decay=0.0)
    opt_state = opt.init(params)

    data_sharding = None
    if args.data_parallel and len(devs) > 1:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        mesh = Mesh(np.array(devs), ("data",))
        data_sharding = NamedSharding(mesh, P("data"))
        repl = NamedSharding(mesh, P())
        params = jax.device_put(params, repl)
        opt_state = jax.device_put(opt_state, repl)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, args.n_heads, dtype,
                                                  None if args.attn == "xla" else args.attn)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    def put(a):
        return jax.device_put(a, data_sharding) if data_sharding is not None else jnp.asarray(a)

    x, y = batch_for(1, args)
    t0 = time.perf_counter()
    compiled = train_step.lower(params, opt_state, put(x), put(y)).compile()
    compile_s = time.perf_counter() - t0

    losses, times = {}, []
    for step in range(1, args.steps + 1):
        x, y = batch_for(step, args)
        xb, yb = put(x), put(y)
        t0 = time.perf_counter()
        params, opt_state, loss = compiled(params, opt_state, xb, yb)
        lv = float(loss)                                   # blocks until the step is done
        dt = time.perf_counter() - t0
        losses[step] = lv
        if step > args.warmup:
            times.append(dt)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"  [jax] step {step:4d}  loss {lv:.4f}  {dt * 1e3:8.2f} ms", flush=True)

    step_s = statistics.median(times)
    tokens = args.batch * args.seq
    rec = {"framework": f"jax ({args.attn} attention)", "jax": jax.__version__, "backend": jax.default_backend(),
           "devices": [d.device_kind for d in devs], "n_devices_used": len(devs) if data_sharding is not None else 1,
           "params_millions": round(n_params / 1e6, 2), "config": vars(args),
           "compile_s": round(compile_s, 2), "step_ms": round(step_s * 1e3, 2),
           "step_ms_p10_p90": [round(np.percentile(times, 10) * 1e3, 2), round(np.percentile(times, 90) * 1e3, 2)],
           "tokens_per_s": round(tokens / step_s), "train_tflops": round(6 * n_params * tokens / step_s / 1e12, 2),
           "peak_mem_gib": round(peak_bytes() / 2**30, 2) if peak_bytes() else None,
           "loss_first_last": [losses[1], losses[args.steps]], "losses": losses}
    print(f"  [jax] {rec['params_millions']}M params on {rec['n_devices_used']}x {devs[0].device_kind}: "
          f"{rec['step_ms']} ms/step, {rec['tokens_per_s']} tok/s, compile {rec['compile_s']} s, "
          f"peak {rec['peak_mem_gib']} GiB", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
