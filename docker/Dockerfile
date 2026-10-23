# Roadmap step 3: reproducible benchmark image for the RTX 5070 (sm_120).
#
# The PyTorch cu128 wheel bundles its own CUDA user-space libraries (cuBLAS,
# cuDNN, NCCL, ...), so the small CUDA *base* image is enough: it only needs to
# carry the NVIDIA container env vars. The driver (libcuda) is injected at run
# time by the NVIDIA Container Toolkit from the host (595.99.02, CUDA <= 13.2).
#
# Build from the repo root:  docker build -f docker/Dockerfile -t gpu-systems-lab/bench:torch2.11.0-cu128 .
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ARG UID=1000
ARG GID=1000
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 24.04 ships Python 3.12.3 -- the same interpreter as the native venv.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Pinned packages, frozen from the native venv. Torch comes from the cu128
# index; everything else falls through to PyPI.
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN python3 -m venv /opt/venv \
 && pip install \
      --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple \
      -r /tmp/requirements.lock.txt \
 && rm /tmp/requirements.lock.txt

# Fail the build on a wrong wheel. No GPU is visible during `docker build`, so
# torch.cuda.get_arch_list() is empty here; read the compiled arch flags instead.
RUN python -c "import torch; v = torch.__version__; a = torch._C._cuda_getArchFlags(); print(v, a); assert v == '2.11.0+cu128', v; assert 'sm_120' in a, a"

# Non-root user matching the host user so files written to the /results bind
# mount are owned by zak, not root. The base image's 'ubuntu' user holds uid 1000.
RUN (userdel -r ubuntu 2>/dev/null || true) \
 && groupadd -g "${GID}" bench \
 && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash bench \
 && mkdir -p /results \
 && chown bench:bench /results

WORKDIR /app
COPY --chown=bench:bench benchmarks/ benchmarks/
USER bench

VOLUME ["/results"]
CMD ["python", "benchmarks/smoke_test.py", "--out", "/results"]
