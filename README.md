# GPU Systems Lab

CUDA, NCCL, MPI, and LLVM experiments across a local RTX 5070 and Modal GPUs.

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
- [x] Modal account connected; parity and NCCL scripts ready in `modal/` (not yet run)
- [ ] Containerized benchmark
- [ ] Modal parity run
- [ ] Multi-GPU NCCL/DDP run
