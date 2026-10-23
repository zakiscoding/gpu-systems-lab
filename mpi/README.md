# MPI experiments (roadmap step 5, "MPI second")

[homelab](https://github.com/zakiscoding/homelab) (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../README.md) / `mpi/` · [← repo README](../README.md) · [Results](../docs/RESULTS.md) · [Code guide](../docs/CODE-GUIDE.md)

NCCL handles collectives for PyTorch DDP (see `modal/`). These experiments cover what
MPI is for: explicit point-to-point messaging, process launch, and the neighbour
halo-exchange pattern behind most HPC codes.

| Program | What it teaches |
|---|---|
| `pingpong.cu` | Latency and bandwidth between 2 ranks: host buffers vs GPU data staged through host |
| `allreduce.cu` | `MPI_Allreduce` sweep reported as algbw/busbw, the same convention as nccl-tests and `modal/ddp_train.py` |
| `jacobi.cu` | 2D heat diffusion split into row strips; each step runs a stencil kernel, then swaps halo rows with `MPI_Sendrecv`. Checked bit-for-bit against a single-process run |

Every program detects CUDA-aware MPI at runtime (`MPIX_Query_cuda_support`) and passes device
pointers straight to MPI when it can, otherwise stages through pinned host memory.

## Build and run (on the GPU node)

```bash
sudo apt install openmpi-bin libopenmpi-dev   # Open MPI 4.1.6 on Ubuntu 24.04
make -C mpi            # nvcc -ccbin=mpicxx, sm_120
make -C mpi run        # pingpong (2 ranks), allreduce host + GPU, jacobi (NP=4)
make -C mpi run-mps    # same under CUDA MPS
make -C mpi scaling    # jacobi at 1/2/4/8 ranks
```

Ubuntu's Open MPI is built **without** CUDA support (`mpi_built_with_cuda_support: false`), so
all GPU results below use the staged path.

## Results

RTX 5070, Open MPI 4.1.6, CUDA 13.2, one node, all ranks sharing the one GPU.

### Ping-pong (2 ranks)

| Size | host latency | host GB/s | staged | staged + MPS |
|---|---:|---:|---:|---:|
| 8 B | 0.17 µs | 0.05 | 59.3 µs | 8.2 µs |
| 64 B | 0.22 µs | 0.30 | 58.9 µs | 7.9 µs |
| 4 KiB | 1.14 µs | 3.6 | 58.1 µs | 9.5 µs |
| 256 KiB | 15.0 µs | 17.5 | 40.4 µs | 45.4 µs |
| 2 MiB | 99.0 µs | 21.2 | 261.8 µs (8.0 GB/s) | 305.2 µs (6.9 GB/s) |

Host-to-host over shared memory peaks around 21 GB/s. Staging GPU data costs two PCIe
copies per hop. Without MPS, small messages pay ~58 µs of GPU context switching.

### All-reduce (4 ranks, busbw GB/s)

| Size | host | GPU staged | GPU staged + MPS |
|---|---:|---:|---:|
| 1 KiB | 1.39 (1.1 µs) | 0.00 (1232 µs) | 0.12 (13 µs) |
| 64 KiB | 5.88 | 2.27 | 2.32 |
| 256 KiB | 9.30 | 3.36 | 3.35 |
| 4 MiB | 6.72 | 3.24 | 3.31 |
| 256 MiB | 2.20 | 1.66 | 1.84 |

Shared-memory all-reduce on one host peaks around 9 GB/s busbw at 256 KiB and falls off once
buffers leave cache. Compare the NCCL sweep from `modal/ddp_nccl.py`: NCCL keeps data on the
GPUs and uses NVLink/PCIe peer-to-peer, so it should be one to two orders of magnitude faster.

### Jacobi halo exchange (4096², 500 steps)

| Ranks | wall | compute / halo | Gcell/s | matches serial |
|---|---:|---|---:|---|
| 1 | 0.126 s | 96% / 4% | 66.6 | yes |
| 2 | 0.235 s | 60% / 40% | 35.7 | yes |
| 4 | 0.400 s | 32% / 68% | 20.9 | yes |
| 8 | 0.703 s | 33% / 67% | 11.9 | yes |
| 4, MPS | 0.132 s | 57% / 43% | 63.7 | yes |

### One rank per GPU: 2x H100 on Modal

`modal/mpi_multigpu.py` builds these programs in a 2-GPU container (CUDA 12.8, the same Ubuntu
Open MPI 4.1.6) and binds each rank to its own GPU (`bind_local_gpu()`). Jacobi at 8192², 500 steps:

| Ranks / GPUs | wall | compute / halo | Gcell/s | matches serial |
|---|---:|---|---:|---|
| 1 / 1 | 0.143 s | 94% / 6% | 234.6 | yes |
| 2 / 2 | 0.104 s | 67% / 33% | 322.7 | yes |

**1.38x on 2 GPUs.** It's a real speedup at last, unlike the shared-GPU runs above, but below 2x:
compute halves (0.134 s to 0.070 s), while the halo exchange goes through host memory because this
Open MPI isn't CUDA-aware. Inside Modal's gVisor sandbox even host ping-pong peaks at 5.7 GB/s
(21 GB/s on the local node), and staged GPU all-reduce at 1.9 GB/s busbw. CUDA-aware MPI or NCCL
(GPU to GPU over NVLink) is what would close the gap to 2x.

#### CUDA-aware Open MPI (`modal run modal/mpi_multigpu.py --gpu H100:2 --cuda-aware`)

The same Open MPI 4.1.6 built from source `--with-cuda`. Device pointers go straight to MPI, and
the programs switch to the direct path automatically.

| Test (2x H100) | stock (staged via host) | CUDA-aware (direct) |
|---|---:|---:|
| ping-pong 8 B latency | 15.2 µs | 9.7 µs |
| ping-pong 16 MiB | 4.2 GB/s | **324 GB/s** |
| all-reduce 256 MiB, busbw | 1.51 GB/s | 0.69 GB/s |
| Jacobi 2 ranks, halo time | 0.034 s | 0.072 s |
| Jacobi 2 ranks vs 1 rank | 1.38x | 0.95x |

- **Point-to-point is a huge win.** Large GPU-to-GPU messages hit 324 GB/s, the same NVLink speed
  NCCL reached (329 GB/s): the `smcuda` BTL moves the data with CUDA IPC, and IPC works inside
  Modal's gVisor sandbox.
- **Collectives and small messages got worse.** Open MPI 4.1's reductions on GPU buffers still copy
  to the host, and every run warns `cuMemHostRegister ... failed`: the sandbox won't let CUDA pin
  MPI's shared-memory segment, so those copies use slow pageable memory. Jacobi's 32 KiB halo rows
  take that path, and its 2-GPU run lost its speedup.
- **Lesson:** "CUDA-aware" isn't one switch. Big point-to-point transfers get NVLink speed, but
  collectives need NCCL (or UCC/UCX-based MPI) to stay on the GPU. In a sandbox that blocks host
  pinning, hand-written staging with your own pinned buffers can beat the library.

## Takeaways

- **One GPU does not strong-scale.** Every rank shares the same SMs, so splitting the grid adds
  halo traffic without adding compute. These runs show the decomposition is *correct* (bit-identical
  at every rank count). Real speedup needs one GPU per rank: Modal multi-GPU, or several nodes.
- **MPS matters whenever ranks share a GPU.** It cut the 4-rank Jacobi run from 0.401 s to
  0.132 s (3.0x) and small staged messages from ~58 µs to ~8 µs by removing context switches.
- **CUDA-aware MPI is the next step.** Building Open MPI with UCX + CUDA would replace staging
  with direct device transfers. The code already handles it; only the library changes.
