# A fused kernel in the Triton language vs hand-written CUDA vs torch.compile

[gpu-systems-lab](../../README.md) / `kernels/triton/` · [CUDA kernels](../../cuda/README.md) · [Results](../../docs/RESULTS.md)

[bias_gelu.py](bias_gelu.py) implements the repo's fused bias + GELU(tanh), the op in
[cuda/fused.cu](../../cuda/fused.cu), as a Triton kernel and compares it with:
- the CUDA kernel from `fused.cu`, compiled from the same source as a torch extension and launched
  the same way;
- `torch.compile` of `F.gelu(x + b, approximate="tanh")`, for which TorchInductor writes its own
  Triton kernel;
- eager PyTorch, which runs two kernels and writes the intermediate out to DRAM.

The op is memory-bound. The ideal is one read of `x` and one write of `z`: 2 × 64M × 4 B = 512 MiB
for the 16384 × 4096 fp32 problem. The Triton kernel is 15 lines: one program per `BLOCK`
contiguous elements, masked loads, `libdevice.tanh`, and an autotuned `BLOCK` × `num_warps`.

## Results: RTX 5070, 16384 × 4096 fp32

Measured 2026-11-14 in the [kernel lab container](../Dockerfile) (CUDA 12.8, torch 2.11.0, Triton
3.6.0) on WSL2, with the idle `vllm-chat` resident on the card. Each row is the median of 7
`triton.testing.do_bench` runs (50 ms warmup, 300 ms of reps, L2 flushed between calls); the
brackets give min–max over the 7. Error is max |z − ref| / max(|ref|, 1) against a float64
reference. Raw JSON: [results/kernels/bias_gelu.json](../../results/kernels/bias_gelu.json).

| implementation | ms | effective GB/s | % of 672 GB/s DRAM peak | error |
|---|---:|---:|---:|---:|
| eager (add, then gelu: 2 kernels) | 1.924 [1.912–1.941] | 279.0 | 42% | 2.4e-7 |
| torch.compile (Inductor's Triton) | 0.962 [0.959–0.965] | 558.2 | 83% | 2.4e-7 |
| **Triton-lang (this kernel)** | **0.960** [0.959–0.963] | **559.0** | **83%** | 2.4e-7 |
| CUDA (cuda/fused.cu's kernel) | 1.094 [1.089–1.103] | 490.9 | 73% | 2.4e-7 |

The standalone `cuda/build/fused` binary, timed with its own cudaEvent loop, measured 1.064 ms
(504.7 GB/s) for the same kernel. That matches the extension's number within 3%.

## What the numbers say

1. **Fusion is the 2x.** Eager runs two memory-bound kernels and round-trips the intermediate
   through DRAM. Every fused version moves half the bytes and takes half the time, as
   `fused.cu`'s own measurement found (1.97x).
2. **The Triton kernel ties Inductor and beats the hand-written CUDA by 14%.** The CUDA kernel
   uses a grid-stride loop with one `float` per thread per iteration (4-byte loads). Triton
   assigns each program a contiguous `BLOCK` (autotuned: 1024 elements, 8 warps) and compiles the
   masked loads into 16-byte vector loads. The same lesson came up in the RMSNorm custom op
   ([../custom_op/](../custom_op/README.md)): there, switching to 16-byte loads took a CUDA kernel
   from 310 to 562 GB/s. Inductor's generated kernel and the hand-written Triton
   one, the two land within 0.2%.
3. **83% of peak is about the ceiling for a read+write stream.** The repo's best pure-read kernel
   (warp-shuffle reduction) reached 95%. Writes cost more, and the copy-style kernels in
   `cuda/README.md` land around 500–630 GB/s.

## Profiling

- **Kernel count:** eager runs `add` and `gelu` as two kernels; Inductor and the hand-written
  versions are one kernel each.
- **Nsight Systems** (`nsys profile --trace=cuda`, reports in
  [results/kernels/nsight/](../../results/kernels/nsight/)): records the CUDA API timeline, but
  under WSL2 it recorded no GPU kernel activity ("does not contain CUDA kernel data").
- **Nsight Compute** fails with `ERR_NVGPUCTRPERM`. The Windows driver only gives administrators
  access to the GPU performance counters by default. Turning on "Allow access to the GPU
  performance counters to all users" in the NVIDIA Control Panel (Developer settings) enables it.
  The DRAM-throughput and memory-transaction metrics will be added here once that's on.

## Run it

```bash
docker build -t gsl/kernel-lab:cu128 kernels/          # CUDA 12.8 devel + torch 2.11 + Nsight CLIs
docker run --rm --gpus all -v $PWD:/lab gsl/kernel-lab:cu128 python kernels/triton/bias_gelu.py
docker run --rm --gpus all -v $PWD:/lab gsl/kernel-lab:cu128 \
  nsys profile --trace=cuda -o results/kernels/nsight/bias_gelu python kernels/triton/bias_gelu.py --nsys
```
