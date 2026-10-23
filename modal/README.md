# Modal runs (roadmap steps 4 and 5)

| File | What it does | Runs on |
|---|---|---|
| `smoke.py` | Step 4: runs `benchmarks/smoke_test.py` unchanged on one Modal GPU, saves `results/modal_<gpu>_<ts>.json` | Modal |
| `ddp_nccl.py` | Step 5: on one N-GPU container, runs `ddp_train.py` via `torchrun` (1 GPU baseline, then N GPUs) | Modal |
| `ddp_train.py` | NCCL `all_reduce` sweep 1 MiB to 1 GiB (algbw/busbw) + synthetic-data MLP DDP loop | inside the container |
| `compare.py` | Prints local vs Modal table (TFLOPS, H2D, cold start, approx $/run). No Modal calls | laptop |

The image pins `torch==2.11.0` from the cu128 wheel index to match the local RTX 5070 baseline
(BF16 67.7 / FP16 64.1 / TF32 33.8 / FP32 23.7 TFLOPS at 8192², H2D 44.2 GiB/s).

## One-time setup (Windows laptop)

```powershell
$venv = "C:\Users\cashh\.venvs\gpu-lab\Scripts"
& "$venv\modal.exe" setup          # opens a browser, writes a token to ~/.modal.toml
& "$venv\modal.exe" profile current
```

Set a spending limit in the Modal dashboard (Settings > Usage/Billing) before the first run.

## Run

Run from the repo root so `results/` lands next to `benchmarks/`:

```powershell
cd C:\Users\cashh\OneDrive\Desktop\gpu-systems-lab
$modal = "C:\Users\cashh\.venvs\gpu-lab\Scripts\modal.exe"

# Step 4: single GPU smoke test (default L4)
& $modal run modal/smoke.py
& $modal run modal/smoke.py --gpu A10G     # also: T4, L4, L40S, A100, A100-80GB, H100
# or: $env:MODAL_LAB_GPU = "L40S"; & $modal run modal/smoke.py

# Step 5: NCCL + DDP (default L4:2)
& $modal run modal/ddp_nccl.py
& $modal run modal/ddp_nccl.py --gpu A100:2 --steps 200 --batch 512 --max-mib 1024
```

The first run builds the image (downloads the ~3 GB torch wheel on Modal's builders, a few
minutes, CPU-only). Later runs reuse the cached image. Use `--size 4096 --iters 20` on `smoke.py`
for a quicker, cheaper check.

## Compare

Local runs write to `results/` on the GPU node, so copy them to the laptop first:

```powershell
mkdir results -Force
scp "gpu-ts:~/gpu-systems-lab/results/*.json" results/
C:\Users\cashh\.venvs\gpu-lab\Scripts\python.exe modal/compare.py            # all runs
C:\Users\cashh\.venvs\gpu-lab\Scripts\python.exe modal/compare.py --latest   # newest per GPU
```

`compare.py` picks up `smoke_<host>_*.json` (local), `modal_<gpu>_*.json` (step 4) and
`modal_ddp_*.json` (step 5). `results/` is gitignored, so summarize numbers into docs before committing.

## Cost cautions

Prices in `compare.py` are **approximate list prices per GPU-hour; verify at
https://modal.com/pricing**. Billing is per second of container time, plus small CPU/RAM charges.

| Run | Expected wall time (warm image) | Approx cost |
|---|---|---|
| `smoke.py` on L4 / A10G | 1 to 2 min | about $0.02 to $0.04 |
| `smoke.py` on L40S / A100 | about 1 min | about $0.03 to $0.05 |
| `smoke.py` on H100 | about 1 min | about $0.07 |
| `ddp_nccl.py` on L4:2 | 3 to 5 min | about $0.08 to $0.15 |
| `ddp_nccl.py` on A100:2 / H100:2 | 2 to 4 min | about $0.15 to $0.55 |

- Every run is `modal run` (ephemeral app): the container stops when the run ends. Do not `modal deploy`
  these; nothing here needs to stay up.
- Ctrl+C stops a run. Check `modal app list` afterwards to confirm nothing is still running.
- Function timeouts cap a runaway job: 15 min for `smoke.py`, 30 min for `ddp_nccl.py`.
- Larger GPU counts (`H100:8`) multiply cost by N; start with `:2`.
- Cold start (call to first output) is in each JSON under `modal.cold_start_seconds`; `wall_seconds`
  is used as the upper-bound billed time in the $ estimate.

## Notes

- L4/A10G/L40S multi-GPU containers usually talk over PCIe, A100/H100 over NVLink. `ddp_nccl.py` prints
  `nvidia-smi topo -m` so the busbw numbers can be read against the interconnect.
- `NCCL_DEBUG` defaults to `WARN`; set it to `INFO` inside `ddp_nccl.py` env if a collective hangs.
- Scaling efficiency is weak scaling (fixed per-GPU batch): `throughput_N / (N * throughput_1)`.
