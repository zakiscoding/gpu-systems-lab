"""Fused bias + GELU(tanh) written in the Triton language, against the repo's hand-written CUDA
kernel (cuda/fused.cu, compiled here as a torch extension from the *same* source) and
torch.compile (TorchInductor generates its own Triton kernel), plus eager PyTorch (two kernels).

Same problem as cuda/fused.cu: z = gelu_tanh(x + bias), x is rows x cols fp32 row-major, bias has
`cols` entries. It is memory-bound: the ideal is one read of x and one write of z, 2 * n * 4 bytes.

    python kernels/triton/bias_gelu.py                       # 16384 x 4096, writes results JSON
    python kernels/triton/bias_gelu.py --rows 4096 --cols 1024
    nsys profile -o out/bias_gelu python kernels/triton/bias_gelu.py --nsys   # short run for traces

Correctness: every implementation is checked against a float64 reference (max relative error).
"""

import argparse
import json
import os
import platform
import subprocess
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
K_GELU: tl.constexpr = tl.constexpr(0.7978845608028654)  # sqrt(2/pi); Triton kernels only read constexpr globals


# ---------------------------------------------------------------- Triton

@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (512, 1024, 2048, 4096) for w in (4, 8)],
    key=["n_cols"],
)
@triton.jit
def bias_gelu_kernel(x_ptr, b_ptr, z_ptr, n_elem, n_cols, BLOCK: tl.constexpr):
    """One program per BLOCK contiguous elements of the flattened tensor. Loads are coalesced and
    vectorized by the compiler; bias[i % cols] is a tiny, cache-resident gather."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs % n_cols, mask=mask)
    v = x + b
    z = 0.5 * v * (1.0 + libdevice.tanh(K_GELU * (v + 0.044715 * v * v * v)))
    tl.store(z_ptr + offs, z, mask=mask)


def bias_gelu_triton(x, b, out=None):
    out = torch.empty_like(x) if out is None else out
    n = x.numel()
    bias_gelu_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, b, out, n, x.shape[-1])
    return out


# ---------------------------------------------------------------- CUDA (cuda/fused.cu's kernel)

CUDA_SRC = r"""
#include <torch/extension.h>
// Copied verbatim from cuda/fused.cu: gelu_tanh + bias_gelu_fused (grid-stride, 256 threads,
// SMs x 8 blocks, as launched there).
__host__ __device__ inline float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}
__global__ void bias_gelu_fused(const float* __restrict__ x, const float* __restrict__ bias,
                                float* __restrict__ z, size_t n, int cols) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        z[i] = gelu_tanh(x[i] + bias[i % cols]);
}
torch::Tensor bias_gelu_cuda(torch::Tensor x, torch::Tensor b, torch::Tensor z) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
    bias_gelu_fused<<<sms * 8, 256>>>(x.data_ptr<float>(), b.data_ptr<float>(), z.data_ptr<float>(),
                                      (size_t)x.numel(), (int)x.size(-1));
    return z;
}
"""


def load_cuda():
    from torch.utils.cpp_extension import load_inline
    return load_inline("bias_gelu_cuda_ext", cpp_sources="torch::Tensor bias_gelu_cuda(torch::Tensor x, torch::Tensor b, torch::Tensor z);",
                       cuda_sources=CUDA_SRC, functions=["bias_gelu_cuda"],
                       extra_cuda_cflags=["-O3"], verbose=False)


# ---------------------------------------------------------------- harness

def eager(x, b):
    return F.gelu(x + b, approximate="tanh")


def reference(x, b):
    v = x.double() + b.double()
    return (0.5 * v * (1.0 + torch.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))).float()


def max_err(a, ref):
    """max |a - ref| / max(|ref|, 1): relative for large values, absolute near zero (GELU has a
    zero crossing, where a pure relative error is meaningless)."""
    return ((a.double() - ref.double()).abs() / ref.double().abs().clamp_min(1.0)).max().item()


def bench(fn, reps=7):
    """triton.testing.do_bench (CUDA-graph-free, L2 flushed between calls) `reps` times;
    returns (median ms, min ms, max ms) over the repeats."""
    runs = [triton.testing.do_bench(fn, warmup=50, rep=300) for _ in range(reps)]
    runs.sort()
    return runs[len(runs) // 2], runs[0], runs[-1]


def gpu_state():
    q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used"
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--nsys", action="store_true", help="20 launches of each, no timing loop (for nsys traces)")
    p.add_argument("--out", default=os.path.join(REPO, "results", "kernels", "bias_gelu.json"))
    args = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    x = torch.randn(args.rows, args.cols, device=dev)
    b = torch.randn(args.cols, device=dev)
    z = torch.empty_like(x)
    ref = reference(x, b)
    n_bytes = 2 * x.numel() * 4

    t0 = time.perf_counter()
    cuda_ext = load_cuda()
    build_s = time.perf_counter() - t0
    compiled = torch.compile(eager, fullgraph=True)

    impls = {
        "eager (add + gelu, 2 kernels)": lambda: eager(x, b),
        "torch.compile (Inductor Triton)": lambda: compiled(x, b),
        "Triton-lang (this file)": lambda: bias_gelu_triton(x, b, z),
        "CUDA (cuda/fused.cu kernel)": lambda: cuda_ext.bias_gelu_cuda(x, b, z),
    }
    if args.nsys:
        for name, fn in impls.items():
            for _ in range(20):
                fn()
        torch.cuda.synchronize()
        return

    rows = {}
    for name, fn in impls.items():
        out = fn()
        torch.cuda.synchronize()
        err = max_err(out, ref)
        med, lo, hi = bench(fn, args.reps)
        moved = 4 * x.numel() * 4 if name.startswith("eager") else n_bytes  # eager writes+rereads x+b
        rows[name] = {"ms_median": round(med, 4), "ms_min": round(lo, 4), "ms_max": round(hi, 4),
                      "effective_gbs": round(n_bytes / (med * 1e6), 1),
                      "actual_traffic_gbs": round(moved / (med * 1e6), 1), "max_err": err}
        print(f"  {name:<34} {med:7.4f} ms  [{lo:.4f}-{hi:.4f}]  {n_bytes / (med * 1e6):6.1f} GB/s eff.  err {err:.1e}")
    best = bias_gelu_kernel.best_config
    rec = {"problem": {"rows": args.rows, "cols": args.cols, "dtype": "float32",
                       "ideal_bytes": n_bytes},
           "device": torch.cuda.get_device_name(dev), "torch": torch.__version__, "triton": triton.__version__,
           "cuda": torch.version.cuda, "host": platform.node(), "gpu_state_after": gpu_state(),
           "cuda_ext_build_s": round(build_s, 1), "triton_best_config": str(best),
           "method": f"triton.testing.do_bench x{args.reps} (warmup 50 ms, rep 300 ms, L2 flush), median/min/max",
           "results": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  triton best config: {best}\n  wrote {args.out}")


if __name__ == "__main__":
    main()
