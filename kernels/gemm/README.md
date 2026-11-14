# GEMM, step by step, on the RTX 5070

[gpu-systems-lab](../../README.md) / `kernels/gemm/` · [CUDA kernels](../../cuda/README.md) · [Roofline](../../docs/ROOFLINE.md)

[gemm.cu](gemm.cu) builds a square GEMM (`C = A·B`, row-major, N = 4096) one optimization at a
time. Each kernel is checked against cuBLAS on the same inputs and benchmarked next to cuBLAS and
CUTLASS. The repo's earlier [cuda/matmul.cu](../../cuda/matmul.cu) stopped at shared-memory tiling
(13% of cuBLAS). This continues up to the tensor cores.

## Results

Measured 2026-11-14 in the [kernel lab container](../Dockerfile) (CUDA 12.8, `-O3 -arch=sm_120`,
CUTLASS v4.8.1 headers) on WSL2, next to the idle `vllm-chat`. Each kernel is timed as the median
of 15 runs after a warm-up, and the whole program was run 3 times; the table shows mean ± SD over
those 3. SM clock during the runs was 2.78–2.84 GHz at 50–64 °C and 172–182 W. Raw output, plus
GPU state before and after each run: [results/gemm/](../../results/gemm/).

| step | what changed | ms | TFLOPS | % of cuBLAS | speedup vs previous |
|---|---|---:|---:|---:|---:|
| 1 naive | one thread per C element, `threadIdx.x` walks **rows** | 273.5 | 0.50 ± 0.00 | 2.2% | |
| 2 coalesced | `threadIdx.x` walks **columns** | 68.7 | 2.00 ± 0.00 | 8.9% | **4.0x** |
| 3 shared-memory tiling | 32×32 tiles of A and B in shared memory | 59.5 | 2.31 ± 0.00 | 10.3% | 1.16x |
| 4 1-D register tiling | 64×64 block tile, 8 outputs per thread | 17.8 | 7.74 ± 0.01 | 34.4% | **3.35x** |
| 5 2-D register tiling | 128×128 block tile, 8×8 outputs per thread | 19.3 | 7.14 ± 0.01 | 31.7% | 0.92x |
| 6 vectorized | float4 loads, A transposed in shared memory | 7.48 | 18.38 ± 0.10 | **81.7%** | **2.58x** |
| cuBLAS SGEMM (pedantic FP32) | | 6.11 | 22.49 ± 0.04 | 100% | |
| CUTLASS SIMT FP32 (default tile) | | 6.11 | 22.50 ± 0.02 | 100.0% | |
| **7 WMMA tensor cores** (FP16 in, FP32 acc) | 4 warps, 128×128 block, 16×16×16 fragments | 3.17 | 43.43 ± 0.18 | 65.3% of FP16 | 2.36x over step 6 |
| cuBLAS GEMM FP16 → FP32 | | 2.07 | 66.54 ± 0.06 | 100% | |
| CUTLASS TensorOp FP16 (128×128×32, 3 stages) | | 2.06 | 66.58 ± 0.21 | 100.0% | |

**Correctness:** every FP32 kernel matched cuBLAS exactly at N = 4096 (max |diff| = 0), and to
1.9e-6 relative at N = 1024. They all accumulate over k in the same order with FMA, so they agree
bit for bit with each other. Agreement with cuBLAS depends on whether cuBLAS's kernel for that
size also reduces k in order. At N = 1024 its result differed slightly, consistent with a
split-K kernel. WMMA and CUTLASS TensorOp also matched cuBLAS FP16 bit for bit.

An earlier version of the check reported 0 error by accident: `std::max` silently drops NaN. It
now returns infinity on any NaN or Inf.

## Why each step helps (or doesn't)

The explanations come from the access patterns and the arithmetic. **Nsight Compute was not
available:** the Windows host restricts GPU performance counters (`ERR_NVGPUCTRPERM`) and the
setting was left off. So the bank-conflict and occupancy statements below are inferred from the
code, not measured.

1. **Naive → coalesced (4.0x): global memory transactions.** With `threadIdx.x` on rows, the 32
   threads of a warp read A at addresses N floats apart, so one load instruction touches 32
   different 32-byte sectors. They also write C 32 rows apart. Putting `threadIdx.x` on columns
   makes B and C accesses 128 contiguous bytes per warp, and A becomes one broadcast address.
   Same FLOPs, about 1/32 of the memory transactions.
2. **Shared-memory tiling (1.16x): reuse, but still too many shared loads per FMA.** Each A and B
   value is now read from DRAM once per 32×32 tile, not once per output. But each FMA still needs
   2 shared-memory loads, so the kernel is limited by shared-memory bandwidth. That's why the gain
   is small (and why `cuda/matmul.cu` saw TILE=32 *slower* than TILE=16).
3. **1-D register tiling (3.35x): FMAs per shared load.** Each thread keeps 8 accumulators and
   reuses one `Bs` value for 8 FMAs: 9 shared loads per 8 FMAs instead of 16.
4. **2-D register tiling (0.92x, a regression):** in theory 16 shared loads feed 64 FMAs. But the
   `Bs` reads are `Bs[k*128 + tcol*8 + j]`: adjacent threads are 8 floats apart, so a warp hits
   only 4 of the 32 banks, an 8-way bank conflict. The A tile is stored row-major with an 8-wide
   K, which conflicts too. The 4x better FMA:load ratio is eaten by serialized shared accesses.
5. **Vectorized + transposed A (2.58x): fixing the shared-memory access.** A is stored
   transposed (`As[k][m]`), so a thread's 8 A values are contiguous. Both operands are read as
   `float4`, one 128-bit instruction per 4 values. That cuts the shared-load instruction count by
   4x and turns the strided pattern into wide loads the hardware serves in fewer wavefronts.
   Global loads are float4 too. This one change takes the kernel from 32% to **82% of cuBLAS**.
   The remaining gap is what cuBLAS/CUTLASS add: double-buffered tiles loaded asynchronously
   (`cp.async`), swizzled shared layouts with no conflicts at all, and tuned tile shapes.
6. **WMMA tensor cores (2.36x over the best FP32 kernel):** the `mma` pipe does a 16×16×16 FP16
   product per warp instruction. The simple version reaches 65% of cuBLAS FP16. It has no
   double-buffering, so the tensor cores sit idle while each new 128×32 tile loads. CUTLASS with
   a 3-stage `cp.async` pipeline and the same 128×128 tile reaches cuBLAS's 66.6 TFLOPS, so the
   pipeline is the missing piece.

**CUTLASS on sm_120:** the default `Sm80` TensorOp configuration (128×256×64 tiles) fails at
launch with `Error Internal`, because it asks for more shared memory per block than the RTX 5070
allows. An explicit 128×128×32, 3-stage configuration fixes it and matches cuBLAS.

## Run it

```bash
git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass /opt/cutlass
docker run --rm --gpus all -v $PWD/kernels/gemm:/g -v /opt/cutlass:/cutlass -w /g gsl/kernel-lab:cu128 \
  bash -c "make CUTLASS=/cutlass && ./build/gemm 4096 15"
```
