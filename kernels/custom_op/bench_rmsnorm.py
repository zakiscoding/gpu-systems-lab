"""labops::rmsnorm (C++/CUDA custom op) vs TorchInductor's generated kernel vs eager PyTorch.

For each shape: median of repeated triton.testing.do_bench runs, effective bandwidth (one read of
x, one write of y), and max error against an fp32 reference. Then a torch.profiler pass that lists
the CUDA kernels each version actually launches, and Inductor's generated kernel source is saved
for the write-up.

    python kernels/custom_op/bench_rmsnorm.py
    nsys profile -o out/rmsnorm python kernels/custom_op/bench_rmsnorm.py --nsys
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import rmsnorm_op  # noqa: E402

SHAPES = [(4096, 4096), (16384, 4096), (8192, 8192), (32768, 1024)]


def bench(fn, reps):
    runs = sorted(triton.testing.do_bench(fn, warmup=50, rep=300) for _ in range(reps))
    return runs[len(runs) // 2], runs[0], runs[-1]


def kernels_launched(fn, iters=5):
    """{kernel name: (calls, total GPU us)} from torch.profiler for `iters` calls of fn."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" and e.count:
            out[e.key[:90]] = (e.count, round(e.device_time_total, 1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--nsys", action="store_true")
    p.add_argument("--sweep-vpt", action="store_true", help="time v2 at 1/2/4/8 vectors per thread (env LABOPS_RMSNORM_VPT, one process each)")
    p.add_argument("--out", default=os.path.join(REPO, "results", "kernels", "rmsnorm_custom_op.json"))
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    if args.sweep_vpt:
        x = torch.randn(16384, 4096, device="cuda", dtype=dtype)
        w = torch.randn(4096, device="cuda", dtype=dtype)
        med, lo, hi = bench(lambda: rmsnorm_op.rmsnorm(x, w), args.reps)
        print(json.dumps({"vpt": os.environ.get("LABOPS_RMSNORM_VPT", "default"), "ms": round(med, 4),
                          "gbs": round(2 * x.numel() * 2 / (med * 1e6), 1)}))
        return

    code_dir = os.path.join(REPO, "results", "kernels", "inductor_rmsnorm")
    os.makedirs(code_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(code_dir, "cache"))
    inductor = torch.compile(rmsnorm_op.rmsnorm_reference, dynamic=False)
    custom_in_compile = torch.compile(lambda x, w: rmsnorm_op.rmsnorm(x, w), fullgraph=True, dynamic=False)

    rows = []
    for (r, c) in SHAPES:
        x = torch.randn(r, c, device="cuda", dtype=dtype)
        w = torch.randn(c, device="cuda", dtype=dtype)
        ref = rmsnorm_op.rmsnorm_reference(x.float(), w.float())
        impls = {
            "eager PyTorch": lambda: rmsnorm_op.rmsnorm_reference(x, w),
            "torch.compile (Inductor kernel)": lambda: inductor(x, w),
            "custom op v1 (scalar loads)": lambda: torch.ops.labops.rmsnorm_naive(x, w, 1e-6),
            "custom op v2 (16-B vector loads)": lambda: rmsnorm_op.rmsnorm(x, w),
            "custom op inside torch.compile": lambda: custom_in_compile(x, w),
        }
        if args.nsys:
            for fn in impls.values():
                for _ in range(20):
                    fn()
            continue
        nbytes = 2 * x.numel() * x.element_size()
        for name, fn in impls.items():
            err = (fn().float() - ref).abs().max().item()
            med, lo, hi = bench(fn, args.reps)
            rows.append({"shape": [r, c], "impl": name, "ms_median": round(med, 4), "ms_min": round(lo, 4),
                         "ms_max": round(hi, 4), "gbs": round(nbytes / (med * 1e6), 1), "max_abs_err": err})
            print(f"  {r:>6}x{c:<5} {name:<36} {med:7.4f} ms [{lo:.4f}-{hi:.4f}] {nbytes / (med * 1e6):7.1f} GB/s  err {err:.2e}")
    torch.cuda.synchronize()
    if args.nsys:
        return

    x = torch.randn(16384, 4096, device="cuda", dtype=dtype)
    w = torch.randn(4096, device="cuda", dtype=dtype)
    prof = {"eager PyTorch": kernels_launched(lambda: rmsnorm_op.rmsnorm_reference(x, w)),
            "torch.compile (Inductor kernel)": kernels_launched(lambda: inductor(x, w)),
            "custom op v1 (scalar loads)": kernels_launched(lambda: torch.ops.labops.rmsnorm_naive(x, w, 1e-6)),
            "custom op v2 (16-B vector loads)": kernels_launched(lambda: rmsnorm_op.rmsnorm(x, w))}
    for name, ks in prof.items():
        print(f"  [{name}] kernels per 5 calls:")
        for k, (n, us) in ks.items():
            print(f"      {n:3d} x {us / n:8.1f} us  {k}")

    # keep Inductor's generated Triton source for the write-up
    gen = sorted(glob.glob(os.path.join(code_dir, "cache", "**", "*.py"), recursive=True), key=os.path.getmtime)
    kernel_srcs = [g for g in gen if "triton_" in open(g, errors="ignore").read() and "rsqrt" in open(g, errors="ignore").read()]
    if kernel_srcs:
        with open(kernel_srcs[-1], errors="ignore") as f, open(os.path.join(code_dir, "inductor_rmsnorm_kernel.py"), "w") as g:
            g.write(f.read())

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    rec = {"dtype": args.dtype, "torch": torch.__version__, "triton": triton.__version__, "gpu": smi,
           "method": f"triton.testing.do_bench x{args.reps}, median/min/max ms",
           "results": rows, "profiler_kernels_16384x4096": prof}
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print("  wrote", args.out)


if __name__ == "__main__":
    main()
