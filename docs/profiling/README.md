# Profiling evidence (roadmap step 6)

[homelab](https://github.com/zakiscoding/homelab) (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../../README.md) / `docs/profiling/` · [← repo README](../../README.md) · [Results](../RESULTS.md) · [Code guide](../CODE-GUIDE.md)

Nsight Systems profiles of the `cuda/` kernels on the RTX 5070 node. Nsight
Compute is installed but blocked by the driver's performance-counter permission
(see [below](#nsight-compute-blocked-by-gpu-counter-permission)), so the
hardware-counter (SOL / occupancy) numbers are not here yet.

| File | Contents |
|---|---|
| `nsys_matmul.txt` | program output under nsys + kernel, grid/block and memop summaries for `build/matmul` (N = 4096) |
| `nsys_fused.txt` | same, for `build/fused` (16384 x 4096) |
| `nsys_reduction.txt` | same, for `build/reduction` (32M floats) |
| `ncu_permission_error.txt` | the `ERR_NVGPUCTRPERM` failure from `ncu` and the driver setting that causes it |

Setup: RTX 5070 (sm_120, 48 SMs, 672 GB/s theoretical DRAM), driver 595.99.02,
CUDA 13.2, Ubuntu 24.04, binaries built by `make` with `-O3 -lineinfo`.
Nsight Systems 2025.6.3, Nsight Compute 2026.1.1. Profiled 2026-10-23.

The `.nsys-rep` files are not committed. Rebuild them with the commands at the
end.

## How to read these numbers

- Kernel times are the **Avg (ns)** column of `cuda_gpu_kern_sum`, i.e. the GPU
  execution time recorded by the CUDA trace, including warmup launches.
- **Time (%)** in the nsys tables depends on how many times each kernel was
  launched (the atomic reduction runs 5 times, the others 22), so compare the
  per-launch averages, not the percentages.
- GB/s below = bytes the kernel must move to/from DRAM (inputs read once, output
  written once) / nsys average time. That is effective bandwidth, not a
  counter measurement.
- Rows marked *derived* are arithmetic on the source code and the measured
  times. They are not hardware counters.

## 1. Matmul: naive vs tiled vs cuBLAS (FP32, N = 4096)

| Kernel | Grid x Block | Launches | Avg time (ms) | GFLOPS | vs cuBLAS |
|---|---|---:|---:|---:|---:|
| cuBLAS (`cutlass_80_simt_sgemm_128x128_8x4`) | 256x4 x 128 | 24 | 6.019 | 22,834 | 1.00x |
| `matmul_naive` | 256x256 x 16x16 | 6 | 67.140 | 2,047 | 11.2x slower |
| `matmul_tiled<16>` | 256x256 x 16x16 | 12 | 47.908 | 2,869 | 7.96x slower |
| `matmul_tiled<32>` | 128x128 x 32x32 | 12 | 57.603 | 2,386 | 9.57x slower |

GFLOPS = 2N^3 / avg time = 137.4 GFLOP per call. Run-to-run spread is tiny
(tiled16 std dev 5 us on 48 ms).

What the trace shows:

- **cuBLAS uses a SIMT kernel, not tensor cores.** The program asks for pedantic
  FP32, and nsys shows the kernel is `cutlass_80_simt_sgemm_128x128_8x4`. The
  8x lead is therefore plain CUDA-core FMA, not TF32.
- **cuBLAS gives each thread 128 outputs.** Its launch is 1024 blocks of 128
  threads, one block per 128x128 tile of C (4096^2 / 128^2 = 1024), so 128
  outputs per thread. Our kernels compute one output per thread
  (256x256 blocks x 256 threads = N^2 threads).
- **Why tiled16 is still ~8x off** *(derived)*: in `matmul_tiled<16>` every
  FMA needs two shared-memory loads (`As[ty][k]`, `Bs[k][tx]`), so the inner
  loop is limited by load/store throughput, not FMA throughput. A thread that owns an
  8x16 register tile loads 8 + 16 = 24 values per k-step and does 128 FMAs
  (5.3 FMAs per load instead of 0.5). Both kernels request a similar volume
  from global memory: about 34.4 GB per call for tiled16 (717 GB/s requested)
  and about 4.3 GB for a 128x128 block tile (714 GB/s at cuBLAS speed). The gap
  comes from on-chip reuse in registers, not from DRAM traffic.
- **The naive kernel depends on caches** *(derived)*: each thread reads 2N
  floats, about 550 GB requested per call, which is 8.2 TB/s at 67 ms. That is
  12x the DRAM peak, so most loads hit L1/L2. This is why tiling only buys 1.40x.
- **TILE=32 is slower than TILE=16** (57.6 vs 47.9 ms). 32x32 = 1024-thread
  blocks, and an SM holds 1536 threads, so only one block fits per SM. Each
  `__syncthreads` then stalls the whole SM. ncu's Occupancy section would
  confirm this. It is blocked for now.
- Memory copies are not a factor. The 64 MiB H2D copies take 4.4 ms each
  (about 15 GB/s, pageable host memory), but they happen once, outside the
  timed loop.

## 2. Bias + GELU: unfused vs fused (16384 x 4096 FP32, 256 MiB per tensor)

| Kernel | Grid x Block | Launches | Avg time (ms) | DRAM bytes | GB/s | % of 672 GB/s |
|---|---|---:|---:|---:|---:|---:|
| `add_bias` | 384 x 256 | 46 | 1.041 | 536.9 MB | 516 | 77% |
| `gelu` | 384 x 256 | 46 | 1.025 | 536.9 MB | 524 | 78% |
| unfused total (`add_bias` + `gelu`) | | | 2.066 | 1073.7 MB | 520 | 77% |
| `bias_gelu_fused` | 384 x 256 | 23 | 1.068 | 536.9 MB | 503 | 75% |

**Speedup from fusion: 2.066 / 1.068 = 1.93x.**

- All three kernels run at the same ~500-520 GB/s, so they are memory-bound.
  The GELU math (`tanhf`, several FMAs) costs almost nothing. The fused kernel does
  the bias add and the GELU in one pass and is only 4% slower than
  `add_bias` alone.
- Fusion removes one write and one read of the 256 MiB intermediate, which
  halves DRAM traffic. Time halves with it, as expected for a bandwidth-bound op.
- 75-78% of theoretical is below the ~95% that the reduction reaches (next
  section). These kernels both read and write, while the reduction only reads.
  Mixed read/write traffic loses some DRAM efficiency to bus turnaround.
  vector_add and `cudaMemcpy` D2D land in the same ~570-590 GB/s range. That
  explanation is a hypothesis: ncu's MemoryWorkloadAnalysis section would
  confirm it.

## 3. Reduction (32M floats = 134.2 MB read)

| Kernel | Grid x Block | Launches | Avg time (us) | GB/s | % of 672 GB/s |
|---|---|---:|---:|---:|---:|
| `reduce_atomic_naive` | 131072 x 256 | 5 | 40,307 | 3.3 | 0.5% |
| `reduce_smem_tree` | 384 x 256 | 22 | 263.7 | 509 | 76% |
| `reduce_warp_shuffle` | 384 x 256 | 22 | 209.4 | 641 | 95% |
| CUB `DeviceReduceKernel` + `SingleTileKernel` | 720 x 512, then 1 x 512 | 22 | 209.2 + 1.0 | 638 | 95% |

- **The warp-shuffle kernel reaches 95% of theoretical DRAM bandwidth** and
  ties CUB (209.4 us vs 210.3 us for CUB's two kernels combined). That leaves no
  headroom: at this point the reduction costs the same as reading the input once.
- The atomic version accounts for 93% of all GPU time in the trace, even
  though it ran only 5 times against 22 for each of the others. That is
  ~150x slower per launch than the tree (40.3 ms vs 0.264 ms). 32M `atomicAdd`s to a single address
  serialize in L2.
- CUB uses two kernels. The second one (`DeviceReduceSingleTileKernel`, 1 us)
  combines per-block partials. Our kernels do that step with one atomic per
  block instead. The trace shows the cost difference is negligible.

## Nsight Compute: blocked by GPU counter permission

Every `ncu` run fails before collecting any metrics:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

The driver is in its default "admin only" mode (`/proc/driver/nvidia/params`
shows `RmProfilingAdminOnly: 1`). This profiling run did not change any system
settings. To enable it (you need to do this yourself, with sudo), pick one:

1. **Allow all users (persistent):**
   ```bash
   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
   sudo update-initramfs -u
   sudo reboot
   grep RmProfilingAdminOnly /proc/driver/nvidia/params   # expect: 0
   ```
2. **One-off:** run ncu as root with the full path, e.g.
   `sudo /usr/local/cuda/bin/ncu ...` (sudo drops `/usr/local/cuda/bin` from PATH).

After that, these commands produce the planned reports (`-c 1` profiles the first
matching launch; `matmul_tiled` matches TILE=16 because it launches first):

```bash
cd ~/gpu-systems-lab/cuda
S="--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy"
ncu $S -k regex:matmul_naive    -c 1 build/matmul    > ncu_matmul_naive.txt
ncu $S -k regex:matmul_tiled    -c 1 build/matmul    > ncu_matmul_tiled16.txt
ncu $S -k regex:'^add_bias'     -c 1 build/fused     > ncu_add_bias.txt
ncu $S -k regex:'^gelu'         -c 1 build/fused     > ncu_gelu.txt
ncu $S -k regex:bias_gelu_fused -c 1 build/fused     > ncu_bias_gelu_fused.txt
ncu $S -k regex:reduce_warp_shuffle -c 1 build/reduction > ncu_reduce_warp_shuffle.txt
```

What to expect: the matmul kernels should show high SM/LSU throughput and low
DRAM %. The fused, unfused and reduction kernels should show DRAM throughput
near the top of SOL. The `matmul_tiled<32>` Occupancy section should show one
block per SM.

(nsys also warns that CPU sampling is unavailable, because `perf_event_paranoid`
is 4. That has no effect on these runs, which only trace CUDA (`-t cuda`).)

## Reproduce

On the GPU node (Nsight Systems is at `/usr/local/bin/nsys`, or under
`/opt/nvidia/nsight-systems/`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
make -C ~/gpu-systems-lab/cuda
mkdir -p ~/prof && cd ~/prof
for p in matmul fused reduction; do
  nsys profile -t cuda --stats=false -f true -o nsys_$p ~/gpu-systems-lab/cuda/build/$p
  nsys stats --force-export=true -f column \
    --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    nsys_$p.nsys-rep > nsys_$p.txt
done
```

`nsys stats --help-reports` lists the report names available in your nsys
version. To view the timeline, open `nsys_*.nsys-rep` in `nsys-ui`.
