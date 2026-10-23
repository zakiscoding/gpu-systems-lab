"""GPT-style decoder trained on synthetic tokens with DDP or FSDP2, plus DCP checkpoints
and a deliberate-crash hook for torchrun fault-tolerance tests.

Launched by torchrun inside a Modal container (see modal/fsdp_train.py and
modal/elastic_train.py), or locally on CPU with gloo for logic tests:

    torchrun --standalone --nproc-per-node=2 modal/fsdp_gpt.py --device cpu \
        --mode fsdp --d-model 64 --n-layers 2 --n-heads 4 --seq 32 --vocab 256 --steps 4

Memory model (both modes keep fp32 master weights; compute runs in bf16):
  DDP : every rank holds params(4B) + grads(4B) + AdamW exp_avg/exp_avg_sq(8B) = 16 B/param
  FSDP: each rank holds 16 B/param / world, plus one layer's gathered bf16 params at a time

Data is a pure function of (seed, step, rank), so a resumed run sees exactly the batches an
uninterrupted run would have seen, which is what makes loss trajectories comparable.

Rank 0 writes a JSON record to --out. With --events, every rank appends JSON lines
(start / init / ckpt_load / step / ckpt_save / kill) to a shared file for restart analysis.
"""

import time

T_PROC_START = time.time()  # before the slow `import torch`, to measure restart latency honestly

import argparse
import json
import math
import os
import signal
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# MFU denominator, per GPU; MFU figures are only meaningful on H100
H100_BF16_DENSE_TFLOPS = 989.0
# set by --deterministic; read by Block.forward to pick the math SDPA kernel
DETERMINISTIC = False


# ---------------------------------------------------------------- model

class Block(nn.Module):
    """Pre-LayerNorm transformer block: causal multi-head self-attention, then a
    4x-wide GELU MLP, each added back to the residual stream. Linears have no bias."""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        """(batch, seq, d_model) -> same shape. Under --deterministic attention runs
        on the math SDPA kernel so the backward pass is reproducible."""
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        if DETERMINISTIC:  # flash/mem-efficient backward accumulate dQ with atomics
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    """Decoder-only transformer: token + learned position embeddings, `layers`
    Blocks, final LayerNorm and an untied LM head. With `ac` (activation
    checkpointing) each block's activations are recomputed in backward instead
    of stored: less memory, roughly one extra forward of compute."""

    def __init__(self, vocab: int, seq: int, d: int, layers: int, heads: int, ac: bool):
        super().__init__()
        self.ac = ac
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        """(batch, seq) token ids -> (batch, seq, vocab) logits."""
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            if self.ac and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return self.head(self.ln_f(x))


@torch.no_grad()
def init_weights(model: nn.Module, n_layers: int):
    """GPT-2 style init, applied after to_empty() (the model is built on the meta
    device, so its tensors start uninitialized): weights ~ N(0, 0.02), with the
    residual output projections (proj, fc2) scaled by 1/sqrt(2 * n_layers) so the
    residual stream doesn't grow with depth; LayerNorm weight 1, bias 0."""
    for name, p in model.named_parameters():
        if p.dim() == 1:  # LayerNorm
            p.fill_(0.0 if name.endswith("bias") else 1.0)
        else:
            std = 0.02 / math.sqrt(2 * n_layers) if name.endswith(("proj.weight", "fc2.weight")) else 0.02
            nn.init.normal_(p, mean=0.0, std=std)


# ---------------------------------------------------------------- data

def batch_for(step: int, rank: int, args, device):
    """Deterministic in (seed, step, rank): each sequence repeats a random 32-token motif drawn
    from the first --data-vocab ids, so the loss falls quickly (unigram, then copying) instead of
    sitting at ln(vocab) as it would on iid noise."""
    g = torch.Generator().manual_seed(args.seed * 1_000_003 + step * 1009 + rank)
    motif = torch.randint(0, min(args.data_vocab, args.vocab), (args.batch, 32), generator=g)
    toks = motif.repeat(1, (args.seq + 1) // 32 + 1)[:, : args.seq + 1]
    toks = toks.to(device, non_blocking=True)
    return toks[:, :-1], toks[:, 1:]


# ---------------------------------------------------------------- helpers

def local_numel(t: torch.Tensor) -> int:
    """Element count of this rank's local shard (FSDP2 DTensor) or of a plain tensor."""
    return t.to_local().numel() if hasattr(t, "to_local") else t.numel()


def local_bytes(t: torch.Tensor) -> int:
    """Bytes of this rank's local shard (FSDP2 DTensor) or of a plain tensor."""
    lt = t.to_local() if hasattr(t, "to_local") else t
    return lt.numel() * lt.element_size()


def state_bytes(model, opt) -> dict:
    """Training state held on this rank, in GiB: params, grads and optimizer
    state (AdamW exp_avg/exp_avg_sq; 0-dim `step` counters skipped). Under FSDP
    these are local shards, which is where the 1/world saving shows up."""
    p = sum(local_bytes(x) for x in model.parameters())
    g = sum(local_bytes(x.grad) for x in model.parameters() if x.grad is not None)
    o = sum(local_bytes(v) for st in opt.state.values() for v in st.values()
            if torch.is_tensor(v) and v.dim() > 0)
    gib = 2**30
    return {"params_gib": round(p / gib, 3), "grads_gib": round(g / gib, 3),
            "optim_gib": round(o / gib, 3), "total_gib": round((p + g + o) / gib, 3)}


class Events:
    """Append-only JSON-lines event log shared by every rank (a no-op without
    --events). `attempt` is torchrun's restart count, so events from before
    and after a restart can be told apart."""

    def __init__(self, path: str | None, rank: int):
        self.path, self.rank = path, rank
        self.attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))

    def emit(self, ev: str, **kw):
        """Append one event stamped with time.time() (wall clock, comparable across
        processes). fsync'd so the line survives a SIGKILL right after it."""
        if not self.path:
            return
        line = json.dumps({"ev": ev, "t": time.time(), "rank": self.rank, "attempt": self.attempt, **kw})
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def latest_ckpt(ckpt_dir: str | None):
    """The {"step", "path"} pointer in <ckpt_dir>/latest.json, or None if there
    is no checkpoint (or no --ckpt-dir)."""
    if not ckpt_dir:
        return None
    ptr = Path(ckpt_dir) / "latest.json"
    return json.loads(ptr.read_text()) if ptr.exists() else None


def save_ckpt(model, opt, step: int, ckpt_dir: str, rank: int) -> float:
    """Save model + optimizer as a sharded DCP checkpoint in <ckpt_dir>/step_<NNNNNN>
    (each rank writes its own shards), then have rank 0 atomically repoint
    latest.json at it. Collective: every rank must call it. Returns seconds."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    t0 = time.perf_counter()
    path = Path(ckpt_dir) / f"step_{step:06d}"
    msd, osd = get_state_dict(model, opt)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(path))
    dist.barrier()
    if rank == 0:  # publish only after every shard is on disk: a crash mid-save keeps the old pointer
        tmp = Path(ckpt_dir) / "latest.json.tmp"
        tmp.write_text(json.dumps({"step": step, "path": str(path)}))
        os.replace(tmp, Path(ckpt_dir) / "latest.json")
    dist.barrier()
    return time.perf_counter() - t0


def load_ckpt(model, opt, path: str) -> float:
    """Load a DCP checkpoint into the model and optimizer in place, each rank
    reading the shards it owns. Collective: every rank must call it. Returns seconds."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    t0 = time.perf_counter()
    msd, osd = get_state_dict(model, opt)  # also materializes lazy AdamW state so shapes match
    sd = {"model": msd, "optim": osd}
    dcp.load(sd, checkpoint_id=path)
    set_state_dict(model, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
    dist.barrier()
    return time.perf_counter() - t0


def du_bytes(path: str) -> int:
    """Total size in bytes of the files under `path` (checkpoint size on disk)."""
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


# ---------------------------------------------------------------- main

def build(args, device, world):
    """Create the model on the meta device, wrap it for --mode, then allocate and
    initialize the real weights on `device`. Returns (model, AdamW, param count).

    FSDP: fully_shard each block and then the root (embeddings, ln_f, head), so
    params are all-gathered one block at a time; with bf16 the gathered params
    are bf16 and gradients are reduced in fp32. DDP (world > 1): full replica
    per rank; bf16 comes from autocast in the training loop instead.
    """
    torch.manual_seed(args.seed)
    with torch.device("meta"):
        model = GPT(args.vocab, args.seq, args.d_model, args.n_layers, args.n_heads, args.ac)
    n_params = sum(p.numel() for p in model.parameters())

    if args.mode == "fsdp":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mesh = init_device_mesh(device.type, (world,))
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if args.bf16 else MixedPrecisionPolicy()
        for b in model.blocks:
            fully_shard(b, mesh=mesh, mp_policy=mp)
        fully_shard(model, mesh=mesh, mp_policy=mp)  # root: embeddings, ln_f, head
        model.to_empty(device=device)
        init_weights(model, args.n_layers)
        wrapped = model
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model.to_empty(device=device)
        init_weights(model, args.n_layers)
        # gradient_as_bucket_view: grads alias the all-reduce buckets, so DDP costs 16 B/param, not 20
        wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                      gradient_as_bucket_view=True) if world > 1 else model

    opt = torch.optim.AdamW(wrapped.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    return wrapped, opt, n_params


def main() -> None:
    """torchrun worker: set up the process group, build, optionally resume, train,
    optionally checkpoint and crash, then write the JSON record (rank 0).

    Steps are numbered globally (1..--steps): a resumed run starts after the
    checkpoint's step and, because batch_for() depends only on (seed, step,
    rank), sees the same batches as an uninterrupted run. Per-step timing
    excludes the first --warmup steps; the reported step time is the median,
    MAX-reduced across ranks. MFU = 6 * params * tokens / step time over the
    H100 dense BF16 peak (attention FLOPs ignored).

    Fault-injection: with --kill-at-step, rank --kill-rank SIGKILLs itself right
    after that step on attempt 0 only, so torchrun restarts the group and the
    next attempt (with --resume) runs to the end.

    A CUDA OOM during build or training is caught: rank 0 writes status "oom"
    and every rank exits via os._exit(0) without touching collectives, so a
    suite records the OOM instead of hanging.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--mode", choices=["ddp", "fsdp"], default="fsdp")
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--data-vocab", type=int, default=512, help="synthetic tokens use ids [0, data_vocab)")
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=24)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--batch", type=int, default=4, help="sequences per GPU per step")
    p.add_argument("--ac", action="store_true", help="activation checkpointing per block")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic kernels (math SDPA, cuBLAS workspace config) for bit-exact replays")
    p.add_argument("--steps", type=int, default=20, help="train until this global step")
    p.add_argument("--warmup", type=int, default=3, help="first N steps excluded from timing")
    p.add_argument("--mem-frac", type=float, default=None,
                   help="torch.cuda.set_per_process_memory_fraction cap (simulate a smaller GPU)")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--ckpt-final", action="store_true", help="save a checkpoint after the last step")
    p.add_argument("--resume", action="store_true", help="load latest.json from --ckpt-dir if present")
    p.add_argument("--kill-at-step", type=int, default=0,
                   help="SIGKILL --kill-rank right after this step, first torchrun attempt only")
    p.add_argument("--kill-rank", type=int, default=1)
    p.add_argument("--events", default=None, help="append JSON-lines events here (all ranks)")
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.deterministic:
        global DETERMINISTIC
        DETERMINISTIC = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before the first cuBLAS handle
        torch.use_deterministic_algorithms(True, warn_only=True)
    # Namespace every key by torchrun attempt. torchrun keeps its TCPStore across restarts, and
    # without this a restarted rank 1 read the dead attempt's NCCL unique id (rank 0's old
    # bootstrap port) before the new rank 0 overwrote it -> "connection refused" on 2x H100.
    attempt = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    base_store, _, _ = next(dist.rendezvous("env://"))
    store = dist.PrefixStore(f"fsdp_gpt/attempt_{attempt}/", base_store)
    pg = dict(store=store, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device, **pg)
        if args.mem_frac:
            torch.cuda.set_per_process_memory_fraction(args.mem_frac, device)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", **pg)
    rank, world = dist.get_rank(), dist.get_world_size()
    ev = Events(args.events, rank)
    ev.emit("start", t_proc_start=T_PROC_START)
    on_cuda = device.type == "cuda"

    record = {
        "mode": args.mode, "world_size": world, "attempt": ev.attempt,
        "gpu": torch.cuda.get_device_name(local_rank) if on_cuda else "cpu",
        "torch": torch.__version__,
        "config": {k: v for k, v in vars(args).items() if k not in ("out", "events")},
    }
    if on_cuda:
        record["gpu_total_gib"] = round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2)
        if args.mem_frac:
            record["mem_cap_gib"] = round(record["gpu_total_gib"] * args.mem_frac, 2)

    losses: dict[int, float] = {}
    step_times: list[float] = []
    ckpt_saves: list[dict] = []
    start_step = 0
    status = "ok"
    try:
        model, opt, n_params = build(args, device, world)
        record["params_billions"] = round(n_params / 1e9, 4)
        if on_cuda:
            torch.cuda.synchronize()
        ev.emit("init")

        ck = latest_ckpt(args.ckpt_dir) if args.resume else None
        if ck:
            secs = load_ckpt(model, opt, ck["path"])
            start_step = ck["step"]
            record["resumed_from"] = {"step": start_step, "load_s": round(secs, 3)}
            ev.emit("ckpt_load", step=start_step, secs=secs)
        if args.ckpt_dir:
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

        if on_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for step in range(start_step + 1, args.steps + 1):
            x, y = batch_for(step, rank, args, device)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            # FSDP already computes in bf16 via MixedPrecisionPolicy; DDP gets bf16 from autocast
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=args.bf16 and args.mode == "ddp"):
                logits = model(x)
            # softmax / cross-entropy over the vocab in fp32 for numerical stability
            loss = F.cross_entropy(logits.float().view(-1, args.vocab), y.reshape(-1))
            loss.backward()
            opt.step()
            avg = loss.detach().clone()
            dist.all_reduce(avg)  # .item() below also synchronizes the GPU for timing
            # all_reduce sums, so divide by world for the mean loss across ranks
            lv = avg.item() / world
            dt = time.perf_counter() - t0
            losses[step] = lv
            if step > start_step + args.warmup:
                step_times.append(dt)
            ev.emit("step", step=step, loss=lv, dt=dt)
            if rank == 0:
                print(f"  [{args.mode} r{ev.attempt}] step {step:4d}  loss {lv:.4f}  {dt * 1e3:8.1f} ms", flush=True)

            # fault injection: attempt 0 only, so the restarted job runs through
            if args.kill_at_step and step == args.kill_at_step and ev.attempt == 0 and rank == args.kill_rank:
                ev.emit("kill", step=step)
                os.kill(os.getpid(), signal.SIGKILL)  # no cleanup, like a crashed host process

            if args.ckpt_dir and ((args.ckpt_every and step % args.ckpt_every == 0)
                                  or (args.ckpt_final and step == args.steps)):
                secs = save_ckpt(model, opt, step, args.ckpt_dir, rank)
                ckpt_saves.append({"step": step, "save_s": round(secs, 3)})
                ev.emit("ckpt_save", step=step, secs=secs)
                if rank == 0:
                    print(f"  [ckpt] step {step} saved in {secs:.2f}s", flush=True)

        if args.ckpt_dir and ckpt_saves and rank == 0:
            record["ckpt_bytes"] = du_bytes(Path(args.ckpt_dir) / f"step_{ckpt_saves[-1]['step']:06d}")
        record["state_per_rank"] = state_bytes(model, opt)
    except torch.OutOfMemoryError as e:
        status = "oom"
        record["error"] = str(e).split("\n")[0][:300]
    record["status"] = status

    if on_cuda:
        record["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        record["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 2)

    if status == "oom":
        # Do not touch collectives: the peer may be stuck in one. Report and leave.
        if rank == 0 and args.out:
            Path(args.out).write_text(json.dumps(record, indent=2))
            print(f"  [{args.mode}] OOM: {record['error']}", flush=True)
        os._exit(0)

    # median step time per rank, then the slowest rank's (it sets the pace)
    if step_times:
        t = torch.tensor([statistics.median(step_times)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        step_s = t.item()
        tokens = args.batch * args.seq * world
        record["step_ms"] = round(step_s * 1e3, 2)
        record["tokens_per_s"] = round(tokens / step_s)
        if on_cuda:
            # 6*N*tokens (matmul FLOPs, attention excluded), against H100 dense BF16 peak
            record["mfu"] = round(6 * n_params * tokens / step_s / (world * H100_BF16_DENSE_TFLOPS * 1e12), 4)
    if on_cuda:
        m = torch.tensor([record["peak_alloc_gib"], record["peak_reserved_gib"]], device=device)
        dist.all_reduce(m, op=dist.ReduceOp.MAX)
        record["peak_alloc_gib"], record["peak_reserved_gib"] = [round(v, 2) for v in m.tolist()]
    record["losses"] = {str(k): v for k, v in losses.items()}
    record["ckpt_saves"] = ckpt_saves
    record["t_proc_start"] = T_PROC_START

    if rank == 0:
        if args.out:
            Path(args.out).write_text(json.dumps(record, indent=2))
        s = record["state_per_rank"]
        print(f"  [{args.mode}] {record['params_billions']}B params x{world}: state/rank {s['total_gib']} GiB, "
              f"peak alloc {record.get('peak_alloc_gib')} GiB, step {record.get('step_ms')} ms, "
              f"{record.get('tokens_per_s')} tok/s, MFU {record.get('mfu')}", flush=True)
    ev.emit("done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
