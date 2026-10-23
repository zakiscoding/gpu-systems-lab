# GPU Systems Lab

CUDA, NCCL, MPI, and LLVM experiments across a local RTX 5070 and Modal GPUs.

**→ [Results, architecture and runbooks](docs/RESULTS.md)** · [job router](router/README.md) · [profiling](docs/profiling/README.md)

## Goal
Build practical GPU-systems skills by progressing from a verified single-GPU baseline to reproducible containers and distributed multi-GPU workloads.

## Infrastructure

| Resource | Role |
|---|---|
| RTX 5070 desktop | Native Ubuntu development and single-GPU CUDA/PyTorch experiments |
| Modal | On-demand multi-GPU runs and NCCL/DDP experiments |
| iMac | App, database, metrics, and control-plane services |
| Laptop | Documentation, Git, orchestration, and SSH testing |

## Roadmap

### 0. Safety and identity baseline
- Confirm Ubuntu runs from the external 750 GB drive.
- Keep the internal Windows NVMe untouched and unmounted in Ubuntu.
- Verify `nvidia-smi`, GPU visibility, disk mounts, hostname, and Tailscale identity before remote control-plane work.

### 1. Local RTX 5070 baseline
- Create an isolated Python environment.
- Install a PyTorch CUDA wheel compatible with the installed NVIDIA driver.
- Run tensor, matmul, and mixed-precision benchmarks.
- Record GPU model, driver, CUDA runtime, timing, VRAM use, and throughput.

### 2. CUDA and LLVM foundations
- Implement small CUDA C++ kernels: vector add, tiled matrix multiplication, reductions, and fused operations.
- Profile with Nsight Systems/Compute when useful.
- Use LLVM/Clang tooling to inspect compilation, IR, and optimization choices.

### 3. Reproducible GPU containers
- Add Docker plus NVIDIA Container Toolkit after native workloads pass.
- Package a benchmark image and pin Python/package versions.
- Document build, run, and validation commands.

### 4. Modal parity
- Run the same benchmark and training workload on Modal.
- Compare local versus cloud performance, startup time, GPU memory, and estimated cost.

### 5. Distributed training: NCCL first, MPI second
- Use PyTorch `torchrun` + DDP on Modal for multi-GPU training.
- Validate NCCL collectives and scaling with 2+ GPUs.
- Add MPI only for experiments where its process-launch and communication model is the point; it is not required for normal PyTorch DDP.

### 6. Portfolio deliverables
- Publish benchmark results, architecture notes, profiling screenshots, and reproducible runbooks.
- Build a small job-router/control-plane demo that chooses local RTX versus Modal based on VRAM, queue time, and cost.

## Principles
- Start locally; use Modal only when multi-GPU capacity is genuinely needed.
- Do not install the full CUDA Toolkit unless a C++/custom-kernel task requires it.
- Keep local hardware, cloud configuration, secrets, and generated artifacts out of Git.
- Make every experiment reproducible and measurable.

## Status
- [x] Ubuntu 24.04.5 installed on external 750 GB drive
- [x] RTX 5070 recognized by NVIDIA driver and `nvidia-smi`
- [x] Local PyTorch CUDA smoke test (torch 2.11.0+cu128; BF16 67.7 / FP16 64.1 / TF32 33.8 / FP32 23.7 TFLOPS at 8192², H2D 44 GiB/s)
- [x] System deps: CUDA 13.2 toolkit, LLVM 21, Docker + NVIDIA Container Toolkit (`sudo bash scripts/install_system_deps.sh`)
- [x] CUDA kernels + PTX/LLVM IR inspection (`cuda/`: vector add, tiled matmul vs cuBLAS, reductions, fused bias+GELU)
- [x] MPI on one node (`mpi/`: ping-pong, all-reduce, Jacobi halo exchange; MPS gives 3x on shared-GPU runs)
- [x] MPI one rank per GPU on Modal (`modal/mpi_multigpu.py`: 2x H100, Jacobi 1.38x faster, halo staged via host)
- [x] CUDA-aware Open MPI on 2x H100 (`--cuda-aware`: 324 GB/s point-to-point, but collectives slower in the sandbox)
- [x] Multi-GPU NCCL run (2x H100: all-reduce 329 GB/s busbw; DDP 1.02x at batch 256, 1.57x at batch 4096)
- [x] MPI across two machines (`mpi/multinode/`: RTX 5070 + M1 iMac over Tailscale; GPU and CPU results bit-exact, network-bound)
- [x] Containerized benchmark (`docker/`: pinned cu128 image, `run.sh validate`; matches native within 1%)
- [x] Modal parity run (H100: BF16 677.8 TFLOPS, 10x the RTX 5070; cold start 3.7 s)
- [x] Portfolio write-up ([docs/RESULTS.md](docs/RESULTS.md)) and Nsight Systems profiles ([docs/profiling/](docs/profiling/))
- [x] Job router (`router/`: local RTX vs Modal by VRAM, deadline, cost, NVLink; 8 unit tests)
- [x] Inference serving (`serving/`: vLLM 0.32, Qwen2.5-1.5B 4.4k tok/s on RTX 5070 vs 16.8k on H100 at concurrency 64)
- [x] Sharded training (`distributed/`: FSDP2 halves per-GPU state; 5.6B model OOMs under DDP, trains under FSDP on 2x H100)
- [x] Fault tolerance (torchrun restarts + DCP checkpoints: killed rank recovered in 11-14 s, bit-identical final loss)
- [x] Observability (`observability/`: dcgm-exporter + Prometheus + Grafana on the node, tailnet-only)
- [ ] Scheduling: k3s + NVIDIA device plugin (needs k3s installed on the node)
- [ ] Nsight Compute counters (needs `NVreg_RestrictProfilingToAdminUsers=0` on the node)
