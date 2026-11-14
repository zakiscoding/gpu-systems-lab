# A C++/CUDA PyTorch custom operator that works with torch.compile

[gpu-systems-lab](../../README.md) / `kernels/custom_op/` · [Results](../../docs/RESULTS.md)

`labops::rmsnorm` is a fused RMSNorm (`y = x · rsqrt(mean(x²) + eps) · w`, fp32 accumulation, for
fp32/fp16/bf16) written in CUDA and registered as a real PyTorch dispatcher op. It's compared
with the kernel TorchInductor generates for the same math.

| File | What it is |
|---|---|
| [rmsnorm.cu](rmsnorm.cu) | Two kernels. **v1**: one block per row, scalar loads, warp-shuffle + shared-memory reduction. **v2**: the same, with 16-byte vector loads and stores, and the row kept in registers between the two passes. `TORCH_LIBRARY` defines `labops::rmsnorm` (v2, falls back to v1 for unaligned or odd sizes) and `labops::rmsnorm_naive` (v1) |
| [rmsnorm_op.py](rmsnorm_op.py) | Builds it with `torch.utils.cpp_extension.load`. `torch.library.register_fake` adds the meta kernel and `register_autograd` the backward (plain PyTorch ops, so Inductor can fuse them). This is what makes it work under torch.compile |
| [test_rmsnorm_op.py](test_rmsnorm_op.py) | 16 tests: 3 dtypes × 4 shapes against the reference, an unaligned input, gradients, **`torch.library.opcheck`** (schema, fake tensor, autograd and AOT-dispatch checks), and `torch.compile(fullgraph=True)` around it |
| [bench_rmsnorm.py](bench_rmsnorm.py) | Timings, a vectors-per-thread sweep, torch.profiler kernel lists, and a dump of Inductor's generated kernel to [results/kernels/inductor_rmsnorm/](../../results/kernels/inductor_rmsnorm/) |

## Results: RTX 5070, bf16

Measured 2026-11-14 in the [kernel lab container](../Dockerfile) (CUDA 12.8, torch 2.11.0, Triton
3.6.0) on WSL2, with the idle `vllm-chat` on the card. Each row is the median of 7
`triton.testing.do_bench` runs, min–max in brackets. GB/s counts one read of `x` and one write of
`y`. Raw JSON: [results/kernels/rmsnorm_custom_op.json](../../results/kernels/rmsnorm_custom_op.json).

| rows × hidden | eager | **Inductor** | custom v1 (scalar) | **custom v2 (vector)** | v2 inside torch.compile |
|---|---:|---:|---:|---:|---:|
| 4096 × 4096 | 1.156 ms | **0.124** (541 GB/s) | 0.231 (290) | **0.124** (541) | 0.126 (535) |
| 16384 × 4096 | 4.788 | **0.482** (558) | 0.866 (310) | **0.478** (562) | 0.479 (560) |
| 8192 × 8192 | 4.722 | **0.482** (557) | 0.776 (346) | **0.483** (556) | 0.483 (556) |
| 32768 × 1024 | 2.350 | **0.242** (554) | 0.362 (370) | **0.241** (558) | 0.242 (554) |

All agree with an fp32 reference to bf16 precision (max abs error 3.1e-2, the same for eager).
The min–max spread over the 7 runs is under 4% of the median in every cell.

**Vectors per thread for v2** (16384 × 4096, one 16-byte vector = 8 bf16):

| vectors per thread | 1 | **2** | 4 | 8 |
|---|---:|---:|---:|---:|
| GB/s | 493 | **563** | 560 | 556 |

## What the numbers say

1. **The custom op matches Inductor, but only after the second version.** v1 had the obvious
   design and got 290–370 GB/s, 54–67% of Inductor. v2 changed two things: 16-byte vector
   accesses (8 bf16 per load instead of 1) and keeping the row in registers so `x` is read once.
   That brought it to 541–562 GB/s, level with Inductor at every shape and 83% of the 672 GB/s
   DRAM peak.
2. **Two vectors per thread beats one by 14%.** With one 16-byte vector per thread, a 4096-wide row
   needs 512 threads per block, which limits the number of rows (blocks) resident per SM. Two per
   thread halves the block size, doubles the rows in flight, and gives each thread two independent
   loads to overlap. More than 2 didn't help.
3. **A first launch config failed outright.** 1024 threads × 8 cached `uint4` per thread needs
   more registers than an SM has: `cudaErrorLaunchOutOfResources` at hidden size 8192.
   `__launch_bounds__(512)` and a 512-thread cap fixed it. The test suite didn't catch it because
   no test shape reached that path; the benchmark did.
4. **Inductor generates essentially the same kernel.** Its output for 32768 × 1024 is a
   `persistent_reduction` with the whole 1024-wide row in one program
   (`R0_BLOCK = 1024`), fp32 accumulation, `libdevice.rsqrt`, and an `evict_last` hint on the
   weight vector so it stays in L2. Both versions launch exactly one kernel per call
   (torch.profiler: `triton_per/red_fused__to_copy_add_mean_mul_pow_rsqrt_0` vs
   `rmsnorm_vec_kernel`). Eager makes 9 kernel launches (7 different kernels) for the same math and is 10x slower.
5. **Wrapping the op in torch.compile costs nothing.** Because of the schema and the fake kernel,
   Dynamo traces it as one opaque node with `fullgraph=True`, no graph break, and the same speed as
   calling it directly.

So for a standard op like RMSNorm, Inductor's generated kernel is already near the hardware
limit. A custom CUDA op is worth writing when you need something Inductor can't express or
schedule well, and getting it to parity took vector memory access and a tuned launch
configuration.

## Profiling

torch.profiler (CUPTI) gives the per-kernel launches and GPU times quoted above. Nsight Systems
and Nsight Compute ran into the same WSL2 limits described in
[../triton/README.md](../triton/README.md#profiling): Compute is blocked by `ERR_NVGPUCTRPERM`
until GPU performance counters are enabled for all users on the Windows host.

## Run it

```bash
docker build -t gsl/kernel-lab:cu128 kernels/
docker run --rm --gpus all -v $PWD:/lab gsl/kernel-lab:cu128 python -m pytest kernels/custom_op/test_rmsnorm_op.py
docker run --rm --gpus all -v $PWD:/lab gsl/kernel-lab:cu128 python kernels/custom_op/bench_rmsnorm.py
for v in 1 2 4 8; do docker run --rm --gpus all -e LABOPS_RMSNORM_VPT=$v -v $PWD:/lab gsl/kernel-lab:cu128 \
  python kernels/custom_op/bench_rmsnorm.py --sweep-vpt; done
```
