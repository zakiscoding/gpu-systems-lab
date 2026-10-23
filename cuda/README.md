# CUDA and LLVM foundations (roadmap step 2)

Small, heavily commented CUDA C++ programs. Each one checks correctness against
the CPU (or cuBLAS), times kernels with `cudaEvent`, prints a compact table, and
exits nonzero on a mismatch.

| File | What it teaches |
|---|---|
| `common.cuh` | `CUDA_CHECK` macro, `cudaEvent` timer with warmup, deterministic RNG, error metric, table printing |
| `vector_add.cu` | Thread indexing, grid-stride loops, `float4` vector loads, effective bandwidth vs. a D2D `cudaMemcpy` |
| `matmul.cu` | Naive vs. shared-memory tiled SGEMM (TILE 16 and 32), `__syncthreads`, arithmetic intensity, the gap to cuBLAS |
| `reduction.cu` | One-atomic-per-element vs. shared-memory tree vs. warp `__shfl_down_sync`, compared with `cub::DeviceReduce` |
| `fused.cu` | Kernel fusion: bias + GELU as two kernels vs. one, which halves DRAM traffic for a memory-bound op |

## Build and run

Run these on the GPU node. They need CUDA Toolkit 13.x at `/usr/local/cuda`, and
clang-21 for `make llvm`.

```bash
cd ~/gpu-systems-lab/cuda
make            # build/vector_add, build/matmul, build/reduction, build/fused
make run        # run all four; stops at the first FAIL
make ptx        # nvcc PTX for every program -> build/ptx/*.ptx
make llvm       # clang-21 device-side IR (-O0 and -O3) + clang PTX -> build/llvm/
make clean
```

Each binary takes an optional size, for example `build/matmul 1024`,
`build/vector_add 1000000` or `build/fused 4096 1024`. To override the toolchain,
pass `make NVCC=... CUDA_PATH=... ARCH=sm_120`. Inspect SASS with
`cuobjdump -sass build/matmul`.

## Results (RTX 5070, measured)

Node setup: RTX 5070 (sm_120, 48 SMs, 12 GB GDDR7, 672 GB/s theoretical), driver
595.99.02, CUDA 13.2 (nvcc V13.2.86), Ubuntu 24.04, built with `-O3 -lineinfo`.
Each number is the mean of 20 timed iterations after warmup (3 for the atomic
reduction, 5 for naive matmul, 10 for tiled matmul).

### Memory-bound kernels (GB/s = bytes the kernel must move / time)

| Program | Kernel | Size | Time (ms) | GB/s |
|---|---|---|---:|---:|
| vector_add | grid-stride, scalar | 64M floats | 1.415 | 569 |
| vector_add | grid-stride, `float4` | 64M floats | 1.366 | 590 |
| vector_add | `cudaMemcpy` D2D (reference) | 256 MiB | 0.928 | 578 |
| reduction | atomic, one per element | 32M floats | 43.87 | 3.1 |
| reduction | shared-memory tree | 32M floats | 0.272 | 494 |
| reduction | warp shuffle + `float4` | 32M floats | 0.213 | 630 |
| reduction | `cub::DeviceReduce::Sum` | 32M floats | 0.212 | 634 |
| fused | unfused add_bias + gelu (1024 MiB moved) | 16384 x 4096 | 2.092 | 513 |
| fused | fused bias_gelu (512 MiB moved) | 16384 x 4096 | 1.087 | 494 |

Fusion speedup is **1.93x**, which matches the 2x cut in DRAM traffic.

### Matmul (FP32, N = 4096, 2N^3 FLOPs)

| Kernel | Time (ms) | GFLOPS | % of cuBLAS |
|---|---:|---:|---:|
| cuBLAS `sgemm` (pedantic FP32, no TF32) | 6.54 | 21,017 | 100% |
| naive (16x16 block) | 70.05 | 1,962 | 9% |
| tiled TILE=16 | 51.59 | 2,664 | 13% |
| tiled TILE=32 | 59.75 | 2,300 | 11% |

All programs printed PASS. Error metrics:

- **vector_add:** exact (error 0).
- **reduction:** error is at most 3e-9 relative to sum(|x|).
- **fused:** max relative error is 1.2e-7 against a CPU double reference.
- **matmul:** cuBLAS matched a CPU double reference on 512 sampled entries with a
  relative error of 3.2e-5.

At N=4096 the naive and tiled kernels were bit-identical to cuBLAS. All three add
over k in the same order with FMA. At N=1024 they differ by about 3e-5.
Before each kernel runs, the output buffer is filled with NaN, so a kernel that
writes nothing cannot pass.

### Takeaways

- **Simple kernels already reach 85-90% of the DRAM peak.** vector_add, the
  warp-shuffle reduction and the fused kernel all land around 570-630 GB/s, the
  same as `cudaMemcpy`. Once a kernel is memory-bound, removing bytes (fusion)
  is the main lever left.
- **A single atomic address serializes everything.** The naive reduction runs
  about 200x slower than the tree version, because every one of the 32M
  `atomicAdd`s goes to one L2 location.
- **Warp shuffles beat shared memory.** The warp-shuffle reduction is 1.27x
  faster than the shared-memory tree and matches CUB.
- **Tiling alone is not enough.** TILE=16 gives only 1.36x over naive, because
  the naive kernel already gets heavy reuse from L1/L2.
- **TILE=32 is slower than TILE=16.** A 32x32 tile means 1024 threads per
  block. The SM holds at most 1536 threads, so only one such block fits per SM
  (67% occupancy), and each `__syncthreads` stalls the whole SM.
- **Closing the 8x gap to cuBLAS takes more work.** It needs register tiling
  (several outputs per thread), vectorized shared-memory loads and double
  buffering. Those are the next steps after this one.

## LLVM / clang notes

`make llvm` builds all four files with `clang-21 -x cuda --cuda-device-only
--cuda-gpu-arch=sm_120`. clang 21.1.8 **does accept `sm_120`**: the IR carries
`"target-cpu"="sm_120"` and the PTX says `.target sm_120`. It warns that CUDA 13.2
is newer than its "latest partially supported version 12.8".

clang's CUDA wrapper does not work with CUDA 13 headers as shipped. The Makefile
fixes this with flags only, and does not modify the toolkit:

1. `__clang_cuda_runtime_wrapper.h` includes `texture_fetch_functions.h`, which
   CUDA 13 removed. The Makefile generates an empty stand-in in
   `build/clang-compat/` and adds that directory to the include path.
2. `crt/math_functions.hpp` uses `_NV_RSQRT_SPECIFIER`, but on clang's include
   path nothing defines it. The fix is `-D_NV_RSQRT_SPECIFIER=`.
3. CUB, Thrust and libcu++ moved to `include/cccl` in CUDA 13. nvcc adds that
   directory automatically and clang does not, so the Makefile passes
   `-isystem $(CUDA_PATH)/include/cccl`.

What the IR and PTX show:

- **`vector_add.O0.ll` vs `vector_add.O3.ll`:** -O0 has 20 `alloca`s (every
  local lives in stack memory). After SROA/mem2reg at -O3 there are none, and
  the file is half the length. Thread IDs show up as
  `@llvm.nvvm.read.ptx.sreg.*` intrinsics.
- **Vector loads:** both nvcc (PTX ISA 9.2) and clang/LLVM NVPTX (PTX ISA 8.7)
  turn `float4` into `ld.global.nc.v4.b32`. The `.nc` (read-only cache path)
  comes from `const __restrict__`.
- **Different unrolling:** in matmul, nvcc's PTX contains 159 `fma.rn.f32` and
  clang's contains 29. The inner loops are unrolled differently. A good next
  experiment is to diff the two and compare their timings.
