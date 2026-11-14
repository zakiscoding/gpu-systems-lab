# Kernel lab image for the RTX 5070 under WSL2 Docker (--gpus all): CUDA 12.8 toolkit (nvcc, for
# torch C++/CUDA extensions), Nsight Systems + Nsight Compute CLIs, torch 2.11 (bundles Triton 3.6).
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3-venv python3-dev git ninja-build cuda-nsight-systems-12-8 cuda-nsight-compute-12-8 \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir torch==2.11.0 numpy pytest \
      --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
ENV PATH=/venv/bin:/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
WORKDIR /lab
