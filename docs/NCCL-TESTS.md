# nccl-tests on 2x H100: measured vs what NVLink should give

[gpu-systems-lab](../README.md) / `docs/` · [← repo README](../README.md) · [Results](RESULTS.md) · [Code guide](CODE-GUIDE.md)

The repo's earlier NCCL number (329 GB/s all-reduce) came from a PyTorch `dist.all_reduce` loop
([modal/ddp_train.py](../modal/ddp_train.py)). This run uses NVIDIA's own benchmark,
[nccl-tests](https://github.com/NVIDIA/nccl-tests), across all six standard collectives, and
compares the results with the hardware limit.

Measured 2026-11-14 on Modal `H100:2`: two **H100 80GB HBM3 (SXM)**, driver 610.57.04, NCCL
2.25.1 (from the `nvidia/cuda:12.8.1-devel` image), nccl-tests v2.21.1 built for sm_90. One
process drives both GPUs (`-g 2`), 8 B to 8 GiB in powers of two, 5 warmup + 20 timed
iterations, with correctness checking on (`-c 1`). Every size of every test had **0 wrong
results**.

Code: [modal/nccl_tests.py](../modal/nccl_tests.py) (Modal app),
[modal/nccl_parse.py](../modal/nccl_parse.py) (log parser, unit-tested in
[modal/test_nccl_parse.py](../modal/test_nccl_parse.py)). Raw logs and parsed JSON:
[results/nccl_tests/H100x2_20261113_222226/](../results/nccl_tests/H100x2_20261113_222226/).

## Results

Bus bandwidth (`busbw`, GB/s), best of out-of-place and in-place:

| collective | 1 MiB | 16 MiB | 256 MiB | 1 GiB | **8 GiB** | % of 478 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| all_reduce | 69.0 | 206.5 | 298.7 | 324.7 | **344.4** | 72% |
| all_gather | 42.2 | 157.4 | 257.8 | 294.8 | **311.7** | 65% |
| reduce_scatter | 39.7 | 151.3 | 239.4 | 274.7 | **293.4** | 61% |
| broadcast | 85.2 | 271.2 | 361.8 | 370.6 | **373.4** | 78% |
| alltoall | 39.5 | 156.9 | 286.5 | 352.4 | **364.1** | 76% |
| sendrecv | 62.0 | 201.9 | 302.2 | 359.2 | **367.0** | 77% |
| all_reduce, `NCCL_P2P_DISABLE=1` | 12.4 | 23.5 | 24.9 | 25.7 | **26.6** | 6% |

The smallest all_reduce (8 bytes) took **9.8 µs**, which is the latency floor for one collective.

## What the hardware should reach

- `nvidia-smi nvlink -s` shows **18 NVLink links per GPU at 26.56 GB/s each**, so
  18 × 26.56 = **478 GB/s per direction**. NVIDIA's marketing figure, 900 GB/s, counts both
  directions.
- On an HGX H100 board each GPU's 18 links go to NVSwitches, so any pair of GPUs can in principle
  use the full 478 GB/s.
- `busbw` is the number to compare with that limit. nccl-tests scales the user-visible `algbw`
  by the traffic each rank must move. With 2 ranks, all_reduce `busbw` equals `algbw` (factor
  2(n-1)/n = 1), and all_gather/reduce_scatter `busbw` is half their `algbw`.
- NCCL's own topology model (`NCCL_DEBUG=INFO`, saved in the all_reduce log) picked **12 ring
  channels at 30 GB/s each = 360 GB/s** over `P2P/direct` (NVLink). The measured 344 GB/s
  all-reduce is **96% of what NCCL itself planned for**, and 72% of the raw link rate.

## What the numbers say

1. **Large messages reach 72–78% of the raw NVLink rate.** Pure data movement (broadcast,
   sendrecv, alltoall) gets 364–373 GB/s. Reductions are lower because the GPU also has to add
   the data: all-reduce 344, reduce_scatter 293.
2. **Message size matters more than anything else.** At 1 MiB, all-reduce reaches only 69 GB/s
   (20% of its peak). It needs about 256 MiB to get within 15% of peak. This is why DDP and FSDP
   bucket gradients into tens or hundreds of MB before reducing them.
3. **Without NVLink peer-to-peer, the same all-reduce is 13x slower** (26.6 vs 344 GB/s). With
   `NCCL_P2P_DISABLE=1`, traffic goes through host shared memory, which is roughly what a PCIe
   box without P2P would get. The repo's earlier 2x L4 PCIe run measured 6.65 GB/s.
4. **nccl-tests and PyTorch agree.** The earlier `torch.distributed` loop measured 329 GB/s at
   1 GiB on a different day and node. nccl-tests measured 325 at 1 GiB here.

## What isn't covered (and why)

- **No InfiniBand or RDMA.** Modal runs every job in one container on one host, so these are
  intra-node NVLink numbers only. Multi-node NCCL over InfiniBand/RoCE (GPUDirect RDMA, SHARP,
  `NET/IB`) needs a cluster with a fabric. The home lab has none: the RTX 5070 PC and the iMac
  connect over 1 GbE/Wi-Fi and Tailscale, and the repo's cross-machine MPI run used TCP.
- **No NVLink SHARP (NVLS).** NCCL only uses the NVSwitch's in-switch reduction for larger
  groups. With 2 GPUs it chose plain rings.
- **The topology matrix is missing.** `nvidia-smi topo -m` fails inside Modal's gVisor
  sandbox, and PCI bus IDs read as `[Unknown Error]`. NVLink status still works.

## Run it

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
modal run modal/nccl_tests.py            # H100:2, 8 B .. 8 GiB, ~2 min wall, ~$0.30
modal run modal/nccl_tests.py --quick    # up to 64 MiB
python -m pytest modal/test_nccl_parse.py
```

The first run builds the image (CUDA devel image + nccl-tests compile, CPU only, a few minutes).
This run's container time was 113 s on 2x H100 at $3.95/GPU-hour, about **$0.25**.
