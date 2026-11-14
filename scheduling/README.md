# GPU scheduling: k3s + NVIDIA device plugin on the RTX 5070

[homelab](https://github.com/zakiscoding/homelab) (docs hub) · [homelab-services](https://github.com/zakiscoding/homelab-services) · [homelab-web](https://github.com/zakiscoding/homelab-web) · [homelab-gpu](https://github.com/zakiscoding/homelab-gpu) · **gpu-systems-lab**

[gpu-systems-lab](../README.md) / `scheduling/` · [← repo README](../README.md) · [Results](../docs/RESULTS.md) · [Code guide](../docs/CODE-GUIDE.md)

A single-node Kubernetes cluster (k3s) on `zak-gpu-node` that schedules GPU work as Jobs. The NVIDIA
device plugin advertises the card as the resource `nvidia.com/gpu`, and the scheduler queues any pod
that asks for more GPUs than are free. That gives a queue for free; time-slicing changes how many
pods can share the card.

Everything below was run on 2026-10-23 and is real output (raw logs in `demo/results/`).

## What is in the cluster

| Object | Namespace | Notes |
|---|---|---|
| k3s v1.36.5+k3s1, node `zak-gpu-node` | - | Installed by the user (systemd). `--disable traefik --disable servicelb` |
| RuntimeClass `nvidia` (handler `nvidia`) | cluster | k3s creates this automatically when it finds `nvidia-container-runtime`. `manifests/01-runtimeclass.yaml` is for other clusters; it was not applied here |
| DaemonSet `nvidia-device-plugin` | kube-system | `nvcr.io/nvidia/k8s-device-plugin:v0.19.3`, `runtimeClassName: nvidia`, NVML discovery |
| ConfigMap `nvidia-device-plugin-config` | kube-system | Plugin config. Currently **exclusive** (`nvidia.com/gpu: 1`) |
| Namespace `gpu-jobs` | - | Where GPU Jobs run |
| ConfigMap `gpu-bench` | gpu-jobs | `bench.py`, a bf16 8192x8192 matmul loop that prints TFLOPS as JSON |

Images (pinned, already pulled into k3s containerd, so later pods start in about 2 s):

- `nvcr.io/nvidia/k8s-device-plugin:v0.19.3`
- `nvcr.io/nvidia/cuda:12.8.1-base-ubuntu24.04`: 101 MB, pulled in 17 s
- `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`: 4.2 GB, pulled in 5 m 34 s plus about 3 min to unpack on the HDD (first Job: 8 m 42 s from submit to done)

## Files

| File | Purpose |
|---|---|
| `manifests/00-namespace.yaml` | `gpu-jobs` namespace |
| `manifests/01-runtimeclass.yaml` | RuntimeClass `nvidia`. Not needed on k3s |
| `manifests/10-device-plugin-config-exclusive.yaml` | Plugin config: 1 GPU = 1 schedulable unit |
| `manifests/11-device-plugin-config-timesliced.yaml` | Plugin config: `sharing.timeSlicing` with 4 replicas |
| `manifests/20-device-plugin.yaml` | Device plugin DaemonSet |
| `manifests/30-nvidia-smi-pod.yaml` | Smoke-test Pod: `nvidia-smi` with `limits: nvidia.com/gpu: 1` |
| `manifests/40-bench-configmap.yaml` | `bench.py`. Env vars: `N`, `DURATION` (fixed time) or `ITERS` (fixed work) |
| `manifests/41-matmul-job.yaml` | Job template: 15 s bf16 matmul benchmark on 1 GPU |
| `demo/queue-jobs.yaml` | 3 Jobs, 1 GPU each, 45 s of matmuls each (fixed time) |
| `demo/queue-jobs-fixedwork.yaml` | 3 Jobs, 1 GPU each, 2800 matmuls each (fixed work, about 45 s alone) |
| `demo/run_queue_demo.sh` | Submits a job file, records a timestamped pod watch, events and a per-pod timeline |
| `demo/timeline.py` | Per-pod created, scheduled, start and end times, plus TFLOPS from the logs |
| `demo/results/out-*/` | Raw output of the four demo runs (`summary.txt`, `watch.txt`, `events.txt`) |
| `gpu-operator/` | NVIDIA GPU Operator values, time-slicing ConfigMap, migration runbook. Prepared, not applied |
| [`../operator/`](../operator/README.md) | `GpuJob` CRD + controller: the declarative version of `submit.py` |
| `submit.py` | Job spec to Kubernetes Job (GPU limit, NVIDIA runtime), then waits and prints state changes, logs and timings. Stdlib plus `kubectl` |

## Commands (on the node, as zak)

```bash
export KUBECONFIG=~/.kube/config
cd ~/gpu-systems-lab/scheduling

# install (exclusive mode)
kubectl apply -f manifests/00-namespace.yaml -f manifests/10-device-plugin-config-exclusive.yaml \
              -f manifests/20-device-plugin.yaml -f manifests/40-bench-configmap.yaml
kubectl get node zak-gpu-node -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'   # -> 1

# smoke tests
kubectl apply -f manifests/30-nvidia-smi-pod.yaml && kubectl -n gpu-jobs logs nvidia-smi
kubectl apply -f manifests/41-matmul-job.yaml && kubectl -n gpu-jobs logs -f job/matmul-bf16

# submit anything
python3 submit.py --name mm --env DURATION=20 --configmap gpu-bench:/bench -- python /bench/bench.py
python3 submit.py --name smi --image nvcr.io/nvidia/cuda:12.8.1-base-ubuntu24.04 -- nvidia-smi
python3 submit.py --name x --dry-run -- python train.py        # print the Job JSON only

# demos
bash demo/run_queue_demo.sh exclusive                              # 3 x 45 s
bash demo/run_queue_demo.sh exclusive-fixed demo/queue-jobs-fixedwork.yaml

# switch to time-slicing (4 replicas) and back
kubectl apply -f manifests/11-device-plugin-config-timesliced.yaml
kubectl -n kube-system rollout restart ds/nvidia-device-plugin      # the plugin reads its config at start
kubectl apply -f manifests/10-device-plugin-config-exclusive.yaml
kubectl -n kube-system rollout restart ds/nvidia-device-plugin
```

## 1. GPU in a pod

`nvidia-smi` Pod (`runtimeClassName: nvidia`, `limits: nvidia.com/gpu: 1`):

```
| NVIDIA-SMI 595.99.02              Driver Version: 595.99.02      CUDA Version: 13.2     |
|   0  NVIDIA GeForce RTX 5070        Off |   00000000:01:00.0 Off |                  N/A |
|  0%   33C    P8              1W /  250W |     802MiB /  12227MiB |      0%      Default |
```

`matmul-bf16` Job (PyTorch 2.7.1+cu128 in the container, sm_120):

```
host=matmul-bf16-jv74v gpu=NVIDIA GeForce RTX 5070 cap=(12, 0) torch=2.7.1+cu128 cuda=12.8
{"n": 8192, "iters": 940, "seconds": 15.26, "tflops_avg": 67.7, "tflops_best_window": 68.2, "mem_alloc_gb": 0.28}
```

**67.7 bf16 TFLOPS in the pod, the same as the native 67.7 in `docs/RESULTS.md`.** The container,
the NVIDIA runtime and the device plugin cost nothing in throughput.

## 2. Queue demo: exclusive mode (`nvidia.com/gpu: 1`)

Three Jobs submitted at the same moment (14:53:07), each asking for 1 GPU and running 45 s of matmuls.

Pod states (from `demo/results/out-exclusive/watch.txt`, condensed):

```
14:53:07 queue-1-cvs8h   0/1   Pending             0     0s
14:53:07 queue-2-cp7k7   0/1   Pending             0     0s
14:53:07 queue-3-8l68z   0/1   Pending             0     0s
14:53:09 queue-1-cvs8h   1/1   Running             0     2s
14:53:57 queue-1-cvs8h   0/1   Completed           0     50s
14:54:01 queue-2-cp7k7   1/1   Running             0     54s
14:54:48 queue-2-cp7k7   0/1   Completed           0     101s
14:54:52 queue-3-8l68z   1/1   Running             0     105s
14:55:39 queue-3-8l68z   0/1   Completed           0     2m32s
```

Scheduler events (`events.txt`):

```
18:53:07Z queue-1-cvs8h Scheduled         Successfully assigned gpu-jobs/queue-1-cvs8h to zak-gpu-node
18:53:07Z queue-2-cp7k7 FailedScheduling  0/1 nodes are available: 1 Insufficient nvidia.com/gpu. ...
18:53:07Z queue-3-8l68z FailedScheduling  0/1 nodes are available: 1 Insufficient nvidia.com/gpu. ...
18:53:58Z queue-2-cp7k7 Scheduled         Successfully assigned gpu-jobs/queue-2-cp7k7 to zak-gpu-node
18:53:58Z queue-3-8l68z FailedScheduling  0/1 nodes are available: 1 Insufficient nvidia.com/gpu. ...
18:54:50Z queue-3-8l68z Scheduled         Successfully assigned gpu-jobs/queue-3-8l68z to zak-gpu-node
```

Timeline (seconds after submit):

| pod | scheduled | start | end | run | TFLOPS |
|---|---|---|---|---|---|
| queue-1 | 0 | 2 | 49 | 47 | 67.5 |
| queue-2 | 51 | 54 | 100 | 46 | 67.5 |
| queue-3 | 103 | 105 | 151 | 46 | 67.5 |

Makespan 151 s. The jobs ran strictly one at a time, each at full speed. The hand-off gap
between one job finishing and the next one starting was about 5 s: pod completion, then the
scheduler retry, then container start. Kubernetes does not promise FIFO order (it retries
unschedulable pods with backoff), but with three equal pods it came out FIFO here.

`submit.py` shows the same thing when a second job lands on a busy GPU:

```
[15:03:48] Pending (Unschedulable: 0/1 nodes are available: 1 Insufficient nvidia.com/gpu. ...)
[15:04:15] Pending (ContainerCreating)
[15:04:17] Running
[15:04:26] Succeeded
...
queue+pull wait 29s | run 7s | total 36s | exit 0
```

## 3. Time-slicing (`replicas: 4`)

After applying `11-device-plugin-config-timesliced.yaml` and restarting the plugin, the plugin
logged `"timeSlicing": {"replicas": 4}` and the node's allocatable `nvidia.com/gpu` became **4**.
The same three Jobs were all scheduled at 18:56:09, with no `FailedScheduling` events, and ran
side by side on the one card.

The 45 s fixed-time jobs are not a fair comparison, because each one simply does a third of the
work. So both modes were also run with **fixed work** (2800 matmuls per job):

| Mode | allocatable | Per-job TFLOPS | Aggregate TFLOPS | Per-job run time | Job completion times (s) | Mean completion | Makespan |
|---|---|---|---|---|---|---|---|
| Exclusive, fixed work | 1 | 67.5 | 67.5 | 47 s | 49 / 101 / 153 | 101 s | **153 s** |
| Time-sliced x4, fixed work | 4 | 21.4 | 64.2 | 145 s | 149 / 149 / 149 | 149 s | **149 s** |
| Exclusive, 45 s each | 1 | 67.5 | 67.5 | 46-47 s | 49 / 100 / 151 | 100 s | 151 s |
| Time-sliced x4, 45 s each | 4 | 21.7-21.8 | 65.3 | 47 s | 50 / 51 / 51 | 51 s | 51 s (but only 1/3 of the work) |

During the time-sliced run the host `nvidia-smi` showed three CUDA processes at 578 MiB each,
100% utilization, 2403 MiB used and 199 W.

What this shows:

- Time-slicing does **not** add throughput. The card was already saturated by one job. The
  aggregate went from 67.5 to 64.2 TFLOPS, about 5% lost to context switching between the
  three CUDA contexts.
- Makespan was about the same (149 vs 153 s). Time-slicing only won the roughly 5 s hand-off gaps.
- Mean completion time got **worse** (149 vs 101 s). Under exclusive FIFO the first job finishes in
  49 s. Under time-slicing every job finishes at the end. For equal, compute-bound jobs, exclusive
  queueing is strictly better for latency.
- Time-slicing pays off when jobs do **not** saturate the GPU: small inference servers, notebooks,
  bursty dev or CI pods, data-loader-bound training. In those cases sharing fills idle SMs and pods
  stop queueing behind each other.

Trade-offs:

- No memory isolation. Every replica sees all 12 GB, and the `nvidia.com/gpu: 1` request does not
  cap memory, so one pod can OOM the others. Here three 0.6 GB jobs fit easily, but three 5 GB
  vLLM pods would not.
- No fault isolation. A crash or XID error that resets the GPU kills every pod sharing it.
- Contention is invisible to the scheduler. It sees 4 "GPUs" and reports nothing about the slowdown.
  A job's runtime depends on its neighbours (here, 3x slower).
- No priorities or weights. The driver's round-robin gives every context an equal slice.
- `failRequestsGreaterThanOne: true` is set, because asking for 2 replicas of the same card would
  give no extra compute.
- Alternatives: MPS (`sharing.mps`) gives real concurrent kernel execution and per-client memory
  limits. MIG is hardware partitioning with isolation, but the RTX 5070 does not support it.

**Restored:** `10-device-plugin-config-exclusive.yaml` was applied and the plugin restarted.
Allocatable is back to `nvidia.com/gpu: 1`, and the exclusive fixed-work run above was taken after
the restore.

## GpuJob operator and the GPU Operator plan (new, not yet on this node)

Two follow-ups build on this directory. Neither has run on `zak-gpu-node` yet; the node is
booted into Windows.

- **[`operator/`](../operator/README.md): a `GpuJob` custom resource and a Go controller**
  (controller-runtime) that turns each GpuJob into the same kind of Job `submit.py` builds:
  `runtimeClassName: nvidia`, an `nvidia.com/gpu` limit (so it passes the admission policy
  above), owner references, and a status that says *why* a job is pending
  (`queued: 0/1 nodes are available: 1 Insufficient nvidia.com/gpu`). Tested with envtest
  (16 test functions, 31 cases with subtests, all passing) and run on a local kind cluster with
  a **fake** `nvidia.com/gpu: 1` and a stub `nvidia` RuntimeClass (handler `runc`): two jobs, one
  queued behind the other, with this directory's admission policy applied. Real-GPU run pending.
- **[`gpu-operator/`](gpu-operator/README.md): replacing the static device plugin with the
  NVIDIA GPU Operator** (chart v26.7.2, host driver and toolkit, time-slicing to 4 replicas),
  with a migration and rollback runbook. **Prepared, not yet applied.** Validated offline only:
  the values render against the real chart, every key exists in it, kubeconform passes. One
  finding from preparing it: the GPU Operator's pods use the `nvidia` runtime class without a
  GPU limit, so the admission policy's binding must exempt the `gpu-operator` namespace as
  well, or the migration is blocked by the policy.

## How `router/` could add a `k3s` target (not implemented)

Today `router` treats the local card as "free if idle". When a CUDA process is running, the probe
reports busy and the router falls back to paid Modal. With k3s, a busy local card becomes a queue
instead of a dead end:

1. **Catalog:** add `Target("k3s", 12.0, 67.7, 0.0, queue_min=<estimate>, ...)`, or turn `local`
   into `k3s` once every local job goes through the cluster. The measured 67.7 TFLOPS in a pod means
   the existing speed numbers carry over unchanged.
2. **Probe:** next to `nvidia-smi`, run `kubectl get pods -A --field-selector=status.phase!=Succeeded -o json`
   over SSH and sum the remaining run time of running and Pending GPU pods (each job carries its
   `local_minutes` estimate as an annotation). That gives a real queue estimate for the deadline check.
3. **Decision:** the local card is feasible if `queue + runtime <= deadline`, even when it is busy.
   It stays free, so it wins over Modal whenever the deadline allows. Modal remains the overflow
   path for deadline-bound or large-VRAM jobs.
4. **Execute:** instead of `ssh gpu-ts "cd ~/gpu-systems-lab && <cmd>"`, run
   `ssh gpu-ts python3 ~/gpu-systems-lab/scheduling/submit.py --name <job> --image <img> -- <cmd>`.
   `submit.py` already prints the Pending reason, logs, and queue and run durations for the
   router to report. Running jobs as pods also gives cleanup (`ttlSecondsAfterFinished`), timeouts
   (`activeDeadlineSeconds`) and an audit trail of who ran what.

## Cleanup

```bash
export KUBECONFIG=~/.kube/config; cd ~/gpu-systems-lab/scheduling
kubectl -n gpu-jobs delete job --all                     # finished / queued jobs
kubectl delete -f manifests/20-device-plugin.yaml -f manifests/10-device-plugin-config-exclusive.yaml \
               -f manifests/40-bench-configmap.yaml -f manifests/00-namespace.yaml
# RuntimeClass nvidia belongs to k3s; leave it.
```

Full k3s removal (the user runs this; it needs root and deletes the cluster, its containerd images
including the 4.2 GB PyTorch image, and the systemd unit):

```bash
sudo /usr/local/bin/k3s-uninstall.sh
```

## Hardening items

- The k3s API server (`*:6443`) and kubelet (`*:10250`) listen on **all interfaces**, not only the
  Tailscale IP 100.121.51.35 (`ss -ltn`). Both require TLS client-certificate auth, but they are
  still exposed on the LAN. **Fix:** [`scripts/harden_node.sh`](../scripts/harden_node.sh), run by
  the owner, enables ufw and blocks 22/6443/10250 on the LAN interfaces only (Tailscale, Docker and
  pod networking untouched). Controller-manager and scheduler (10257 and 10259) are already on
  127.0.0.1.
- The device plugin runs with `allowPrivilegeEscalation: false` and all capabilities dropped. It
  needs only the hostPath `/var/lib/kubelet/device-plugins`.
- **The GPU limit is accounting, not enforcement** (tested). A pod with
  `runtimeClassName: nvidia` and *no* `nvidia.com/gpu` limit still got the card: `nvidia-smi -L`
  listed the RTX 5070, and `torch.cuda.is_available()` returned `True` in the PyTorch image. The
  CUDA images set `NVIDIA_VISIBLE_DEVICES=all` and the runtime (`mode = "auto"`, default
  `/etc/nvidia-container-runtime/config.toml`) honours it, so such a pod bypasses the queue
  entirely. A pod *without* the runtime class got nothing (`/dev/nvidia*` missing, no
  `nvidia-smi`). Today the queue only holds if every GPU pod goes through `submit.py` or the
  manifests here. For enforcement, either set `accept-nvidia-visible-devices-envvar-when-unprivileged = false`
  plus `accept-nvidia-visible-devices-as-volume-mounts = true` in the runtime config, with the
  plugin's `deviceListStrategy: volume-mounts` (a host change for the user), or add an admission
  policy (ValidatingAdmissionPolicy) that rejects `runtimeClassName: nvidia` pods without a GPU limit.
  **Fixed with the admission policy:** `manifests/50-require-gpu-limit-policy.yaml` (built-in CEL
  policy, no webhook; kube-system exempt for the device plugin). Tested with server-side dry runs:
  an nvidia-runtime pod without a limit is **denied**; pods with a GPU limit, and pods without the
  runtime class, are allowed; the device plugin still reports `nvidia.com/gpu: 1`.
- The kubeconfig is mode 600 and holds cluster-admin credentials.
