# Job router: local RTX 5070 or Modal

**[homelab](https://github.com/zakiscoding/homelab)** (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../README.md) / `router/` · [← repo README](../README.md) · [Results](../docs/RESULTS.md) · [Code guide](../docs/CODE-GUIDE.md)

A small control plane that decides where a GPU job runs, explains why, and runs it.

```bash
python -m router targets                     # catalog: measured vs estimated speeds, prices, queue
python -m router route --job finetune-7b     # decide + explain, runs nothing
python -m router run --job smoke             # decide, then execute (local over SSH, or Modal)
python -m router run --vram-gb 40 --local-minutes 90 --max-usd 3 --command "python train.py" --yes
python -m unittest router.test_route         # routing rules, no SSH or Modal needed
```

## Homelab job queue integration

`router/homelab_worker.py` connects the router to the homelab control plane
([homelab-services](https://github.com/zakiscoding/homelab-services) on the iMac, with Postgres as
the source of truth). Submit a job with a `route` spec. The router worker is the only one that
claims it (`/internal/worker/jobs/claim?routed=true`; the Windows QLoRA worker skips routed jobs),
decides, runs it locally over SSH or on Modal, and records the decision as the job's experiment.

```bash
# submit (any tailnet machine, user API key)
curl -X POST http://100.91.113.98:8000/jobs -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"route": {"command": "python benchmarks/smoke_test.py --size 4096", "vram_gb": 1, "local_minutes": 0.5, "max_usd": 0.50}}'

# worker on the laptop (controller): config in router/worker.local.env (gitignored)
python -m router.homelab_worker            # poll every 10 s
python -m router.homelab_worker --once     # one claim (tests)
```

Tested end to end. Rows from the iMac's Postgres (`jobs` joined to `experiments`):

| Job | Chosen | Runtime | Est. $ | Local busy | Note |
|---|---|---:|---:|---|---|
| 80911ec2 | L40S (Modal) | 18 s | 0.003 | true | bug: the always-on vLLM server counted as a busy GPU |
| defdd23f | **local** | 21 s | 0.00 | false | after the fix: vLLM is "resident", its 5 GB comes out of free VRAM |

- **Resident servers.** Processes matching `ROUTER_RESIDENT_PROCS` (default `VLLM::EngineCore`)
  count against free VRAM but don't make the card busy. "Busy" means another CUDA process, or peak
  utilization ≥ 20% over three samples.
- **Spend guard.** Jobs without `max_usd` get `MAX_USD_PER_JOB` (default $1). Modal launches need
  no `--yes` here, because the cap is the confirmation.
- **Lease renewal.** Train leases are 15 min. While a routed job runs, a background thread renews
  the lease every 5 min (`POST /internal/worker/jobs/{id}/renew`), so long jobs aren't requeued by
  another worker's stale-lease sweep.

## How it decides

1. **Live probe** of `zak-gpu-node` over SSH (`nvidia-smi`): online? free VRAM? utilization? any CUDA
   process? The node dual-boots into Windows, so "offline" is a normal answer and only takes the
   local card out of the running.
2. **Fit:** per-GPU VRAM with 10% headroom (on the local card, *free* VRAM, since the desktop holds
   ~1.2 GB), GPU count, and **NVLink for multi-GPU jobs**. This repo measured 2x L4 training at
   0.34x one GPU, and 2x H100 at 1.57x.
3. **Runtime estimate:** the job's RTX 5070 minutes scaled by BF16 TFLOPS. Measured speeds from
   `results/*.json` override the catalog estimates automatically. Multi-GPU uses the measured
   data-parallel efficiency (78% NVLink).
4. **Constraints:** deadline (queue + runtime) and cost cap. Modal bills running time, not queue
   time.
5. **Choice:** the local card if it is feasible and idle (free), else the cheapest feasible Modal
   GPU, then the fastest. Paid launches need `--yes`.

## Example decisions (live probe, local idle)

| Job | Constraint | Chosen | Why |
|---|---|---|---|
| `smoke` | 1 GB, 0.5 min | **local** | fits, idle, free |
| `finetune-7b` | 22 GB, cap $5 | **L40S** ~$1.09, 34 min | local has 10.7 GB free; 24 GB cards fail the 10% headroom |
| `eval-sweep` | 8 GB, 4 h locally, deadline 60 min | **L40S** ~$1.46, 45 min | local, T4, L4, A10G all miss the deadline |
| `ddp-2gpu` | 2 GPUs | **H100 x2** ~$0.51, 4 min | only NVLink targets qualify; H100's speed makes it cheaper than A100 |

## Files

| File | Role |
|---|---|
| `catalog.py` | targets: VRAM, BF16 TFLOPS (measured overrides estimates), $/GPU-h, queue, NVLink |
| `probe.py` | SSH probe of the local node |
| `route.py` | feasibility checks, runtime/cost estimates, choice |
| `jobs.py` | named example jobs |
| `__main__.py` | CLI; executes locally via SSH or on Modal via `modal/run_job.py` |
| `homelab_worker.py` | claims routed jobs from the iMac API, renews the lease, decides, runs, records the decision |
| `test_route.py` | 10 unit tests (no SSH or Modal) |

## Limits

- Prices and unmeasured TFLOPS are estimates (marked `*`); verify at modal.com/pricing.
- Runtime scaling assumes compute-bound work. Memory-bound jobs scale with bandwidth instead, so
  they gain less from an H100.
- Queue times are typical values from this repo's runs, not live. 2x L4 once waited 43 minutes.
- Local execution assumes the repo is synced to `~/gpu-systems-lab` on the node with its `.venv`.
