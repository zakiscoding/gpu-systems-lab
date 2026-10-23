# Reproducible GPU container (roadmap step 3)

**[homelab](https://github.com/zakiscoding/homelab)** (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../README.md) / `docker/` · [← repo README](../README.md) · [Results](../docs/RESULTS.md) · [Code guide](../docs/CODE-GUIDE.md)

Packages `benchmarks/smoke_test.py` into a pinned image so the RTX 5070 baseline
can be rerun anywhere with Docker + NVIDIA Container Toolkit (and later compared
against Modal in step 4).

## What is pinned

| Layer | Pin |
|---|---|
| Base image | `nvidia/cuda:12.8.1-base-ubuntu24.04` (`sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2`) |
| Python | 3.12.3 (Ubuntu 24.04 `python3`, same as the native venv) |
| PyTorch | `torch==2.11.0+cu128` from `download.pytorch.org/whl/cu128` (includes sm_120) |
| Everything else | `docker/requirements.lock.txt`, frozen from the native venv (`numpy==2.5.4`, `triton==3.6.0`, `nvidia-*-cu12`, ...) |
| Driver | not in the image: injected from the host by the NVIDIA Container Toolkit (595.99.02, supports CUDA <= 13.2) |

The base image is the small CUDA `base` flavor, not `runtime`: the PyTorch wheel
already bundles cuBLAS/cuDNN/NCCL as `nvidia-*` pip packages, so a `runtime`
image would ship those libraries twice. The build fails if torch is not
`2.11.0+cu128` or its arch list lacks `sm_120`.

The container runs as a non-root `bench` user whose UID/GID are set at build
time to the invoking host user, so result files in `results/` are owned by you.

## Commands

Run on the GPU node from the repo root (`~/gpu-systems-lab`):

```bash
bash docker/run.sh build      # docker build -f docker/Dockerfile --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t gpu-systems-lab/bench:torch2.11.0-cu128 .
bash docker/run.sh check      # docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
bash docker/run.sh run        # docker run --rm --gpus all --ipc=host -v $PWD/results:/results gpu-systems-lab/bench:torch2.11.0-cu128
bash docker/run.sh validate   # check + run + assert all *_ok checks, torch version, sm_120
bash docker/run.sh size       # image size
bash docker/run.sh shell      # interactive shell inside the image
```

Extra arguments after `run`/`validate` go to `smoke_test.py`, e.g.
`bash docker/run.sh run --size 4096 --iters 20`. Override the image tag with
`IMAGE=...` and the output directory with `RESULTS=...`.

Plain Docker equivalent of `run`:

```bash
docker run --rm --gpus all -v "$PWD/results:/results" gpu-systems-lab/bench:torch2.11.0-cu128
```

## Results

Measured 2026-10-23 on the GPU node (Ubuntu 24.04, RTX 5070 sm_120, driver
595.99.02, Docker 29.1.3 with the containerd image store, NVIDIA Container
Toolkit). 8192x8192 matmul, 50 iters.

### Validation

- `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`: sees the RTX 5070, driver 595.99.02, CUDA 13.2.
- `bash docker/run.sh validate`: all checks PASS (`matmul_ok`, `reduction_ok`, `autocast_bf16_ok`), torch `2.11.0+cu128`, compute capability 12.0. Takes about 26 s end to end.
- Inside the container it runs as `uid=1000(bench)` with Python 3.12.3, and result JSON in `results/` is owned by `zak`.

### Image size

| Measure | Size |
|---|---|
| `docker image ls` (containerd store: unpacked + compressed content) | 11.6 GB |
| Image content (`docker image inspect .Size`, roughly what a pull transfers) | 4.15 GB |
| `/opt/venv` inside the image (torch + `nvidia-*` CUDA wheels) | 6.7 GB (same as the native `.venv`) |
| Base image `nvidia/cuda:12.8.1-base-ubuntu24.04` | ~0.1 GB |

Almost all of it is the PyTorch wheel and its bundled CUDA libraries. Using a
`runtime` or `devel` CUDA base would add several more GB with no benefit.
A cold build takes about 15 min, mostly downloading the 820 MB torch wheel and
the ~3 GB of `nvidia-*` wheels.

### Container vs native venv

Three interleaved runs (native, docker, native, docker, ...) back to back,
so both sides see the same clocks and temperature:

| Metric | Native baseline (top-level README) | Native runs 1/2/3 | Container runs 1/2/3 | Container best vs native best |
|---|---|---|---|---|
| BF16 TFLOPS | 67.7 | 67.71 / 67.49 / 67.50 | 63.66 / 67.50 / 67.49 | -0.3% |
| FP16 TFLOPS | 64.1 | 63.89 / 64.80 / 64.20 | 62.70 / 64.05 / 64.31 | -0.8% |
| TF32 TFLOPS | 33.8 | 32.49 / 33.76 / 33.67 | 33.76 / 33.68 / 32.05 | 0.0% |
| FP32 TFLOPS | 23.7 | 23.20 / 23.51 / 23.41 | 23.54 / 23.44 / 23.36 | +0.1% |
| H2D pinned GiB/s | 44.2 | 17.07 / 43.91 / 42.12 | 28.54 / 44.11 / 17.53 | +0.5% |

Takeaways:

- **There's no measurable container overhead for GPU compute.** The container
  runs the same cuBLAS kernels through the same driver, so matmul throughput
  matches native within run-to-run noise (about 1%).
- The first `validate` run, on a cold GPU, gave FP16/BF16 59.5 TFLOPS. Later warm
  runs matched native, so expect some variance and compare best-of-N or
  median-of-N, not a single run.
- **H2D bandwidth is noisy on both sides** (17 to 44 GiB/s). The benchmark only
  times 5 copies of 1 GiB, so one stall skews the result. The peak (~44 GiB/s)
  is the same in and out of Docker.
- `--ipc=host` is set for future DataLoader work and is not needed for this benchmark.
- Result files from the container are named `smoke_<host>-docker_<UTC time>.json`
  (the container clock is UTC), so they can be told apart from native runs.
