# GPU Operator with time-slicing (prepared, not yet applied)

[gpu-systems-lab](../../README.md) / [`scheduling/`](../README.md) / `gpu-operator/` · [GpuJob operator](../../operator/README.md)

> **Status: prepared, not yet applied. It needs the Ubuntu boot of the GPU PC.** Nothing in
> this directory has run on `zak-gpu-node` yet, so there are no measurements here. What was
> checked offline (2026-11-13, on the laptop): the values render against the real chart, every
> values key exists in the chart, and kubeconform passes (output at the end).

Today the k3s node runs the plain NVIDIA device plugin as a static DaemonSet
(`../manifests/20-device-plugin.yaml`), switched between exclusive and time-sliced by hand-applying
a ConfigMap and restarting it. This directory replaces that with the
[NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator), Helm chart **v26.7.2**, which
manages the device plugin, GPU feature discovery (node labels) and validation as one
`ClusterPolicy`, with the card time-sliced into 4 `nvidia.com/gpu` units.

## Files

| File | Purpose |
|---|---|
| `values.yaml` | Chart values. Driver and toolkit come from the host, CDI off, MIG manager off, dcgm-exporter off, device plugin reads `time-slicing-config` |
| `values-toolkit-k3s.yaml` | Optional overlay: the operator manages the container toolkit, pointed at k3s's containerd config and socket |
| `time-slicing-config.yaml` | ConfigMap `time-slicing-config` (namespace `gpu-operator`), key `any`: 4 replicas, `renameByDefault: false`, `failRequestsGreaterThanOne: true` |
| `namespace.yaml` | `gpu-operator` namespace, created first so the ConfigMap can go in before the chart |
| `gpu-limit-policy-binding.yaml` | The admission policy binding with `gpu-operator` exempt next to `kube-system` (see below) |
| `validate.sh` | The offline checks: chart key check, `helm template`, kubeconform. Docker only |

## Choices

**Driver: host.** `driver.enabled: false`. Driver 595.99.02 is already installed and used by
vLLM, Docker and k3s. The operator's driver container would try to load its own kernel module.

**Toolkit: host (default) or operator (overlay).** The host already has nvidia-container-toolkit,
and k3s found `nvidia-container-runtime` at start, wrote the `nvidia` runtime into its generated
containerd config and created RuntimeClass `nvidia`. That already works (the matmul pod ran at
67.7 TFLOPS). With `toolkit.enabled: true` the operator's toolkit container would also edit k3s's
containerd config, so the file would have two owners, and k3s regenerates that file on every
restart. So the default keeps the host toolkit. `values-toolkit-k3s.yaml` is the documented
alternative for a node without a host toolkit:

```yaml
toolkit:
  enabled: true
  env:
    - {name: CONTAINERD_CONFIG,         value: /var/lib/rancher/k3s/agent/etc/containerd/config.toml}
    - {name: CONTAINERD_SOCKET,         value: /run/k3s/containerd/containerd.sock}
    - {name: CONTAINERD_RUNTIME_CLASS,  value: nvidia}
    - {name: CONTAINERD_SET_AS_DEFAULT, value: "false"}
```

**CDI off** with the host toolkit. The chart default (`cdi.enabled: true`) makes the device plugin
hand out devices as CDI names, which needs CDI specs that the operator's toolkit generates. With the
host toolkit, the plugin keeps injecting `NVIDIA_VISIBLE_DEVICES` the way the plain plugin does now.

**dcgm-exporter off.** `observability/` already runs dcgm-exporter in Docker on `127.0.0.1:9400`,
scraped by the node's Prometheus and the iMac. A second exporter would start a second embedded
`nv-hostengine` on the same GPU and add nothing. Moving the exporter into the cluster would mean
switching both Prometheus scrape configs to the in-cluster Service; that is a separate change.
Note that with time-slicing, DCGM still reports one physical GPU: per-pod attribution of SM
time is not available.

**MIG manager off.** The GeForce RTX 5070 has no MIG (and no NVLink).

## Runbook (on the node, as zak, `KUBECONFIG=~/.kube/config`)

Before: allocatable `nvidia.com/gpu` is `1` (exclusive) and the plain plugin runs in kube-system.

```bash
cd ~/gpu-systems-lab/scheduling

# 0. Nothing GPU-bound running in the cluster (Jobs, GpuJobs). vLLM in Docker is not affected.
kubectl get pods -A -o wide | grep -v Completed
kubectl get node zak-gpu-node -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'   # 1

# 1. Exempt the operator's namespace from the GPU-limit admission policy FIRST.
#    Its operands run with runtimeClassName nvidia and no GPU limit, like the plain plugin.
kubectl apply -f gpu-operator/gpu-limit-policy-binding.yaml

# 2. Remove the plain device plugin and its config (the operator brings its own).
kubectl delete -f manifests/20-device-plugin.yaml
kubectl -n kube-system delete configmap nvidia-device-plugin-config
kubectl get node zak-gpu-node -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'   # 0 after the kubelet notices

# 3. Install the operator with the time-slicing config.
kubectl apply -f gpu-operator/namespace.yaml -f gpu-operator/time-slicing-config.yaml
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
helm install gpu-operator nvidia/gpu-operator --version v26.7.2 \
  -n gpu-operator -f gpu-operator/values.yaml --wait --timeout 10m

# 4. Verify.
kubectl -n gpu-operator get pods                           # operator, NFD, device plugin, GFD, validator
kubectl get clusterpolicy cluster-policy -o jsonpath='{.status.state}{"\n"}'              # ready
kubectl get node zak-gpu-node -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'   # 4
kubectl get node zak-gpu-node --show-labels | tr , '\n' | grep -E 'nvidia.com/gpu\.(product|replicas|sharing-strategy)'
#   expect: product=NVIDIA-GeForce-RTX-5070-SHARED (GFD appends -SHARED), replicas=4, sharing-strategy=time-slicing
kubectl apply -f manifests/30-nvidia-smi-pod.yaml && kubectl -n gpu-jobs logs nvidia-smi
bash demo/run_queue_demo.sh timesliced-op demo/queue-jobs-fixedwork.yaml   # compare with the 2026-10-23 numbers
```

The time-sliced run with the plain plugin measured 21.4 TFLOPS per job and 64.2 aggregate for
3 jobs (`../README.md`, section 3). The operator uses the same plugin (v0.20.1 instead of v0.19.3)
with the same sharing config, so the numbers should be close, but that is a prediction, not a result.

### Switch between time-sliced and exclusive (operator way)

The device plugin picks a named config from the ConfigMap. Add an `exclusive` key next to `any`
(`version: v1` / `flags: {migStrategy: none}`), then per node:

```bash
kubectl label node zak-gpu-node nvidia.com/device-plugin.config=exclusive --overwrite   # 1 unit
kubectl label node zak-gpu-node nvidia.com/device-plugin.config-                        # back to default "any" (4)
```

The plugin's config-manager sidecar picks up the label change; no manual rollout restart.

### Rollback to the plain device plugin

```bash
helm uninstall gpu-operator -n gpu-operator
kubectl delete -f gpu-operator/time-slicing-config.yaml -f gpu-operator/namespace.yaml
# The chart leaves its CRDs (operator.cleanupCRD is false); remove them only if not reinstalling:
#   kubectl get crd -o name | grep -E 'nvidia\.com$' | xargs kubectl delete
#   (v26.7.2 ships clusterpolicies, gpuclusters, nvidiadrivers, computedomains, computedomaincliques)
kubectl apply -f manifests/10-device-plugin-config-exclusive.yaml -f manifests/20-device-plugin.yaml
kubectl apply -f manifests/50-require-gpu-limit-policy.yaml       # restores the kube-system-only binding
kubectl get node zak-gpu-node -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'   # 1
```

NFD labels (`feature.node.kubernetes.io/*`) stay on the node after the uninstall. They are
harmless; `kubectl label node zak-gpu-node <key>-` removes them.

## How the GpuJob operator fits

The [GpuJob operator](../../operator/README.md) does not care which plugin advertises
`nvidia.com/gpu`. It creates Jobs with `runtimeClassName: nvidia` and an `nvidia.com/gpu` limit,
and the scheduler does the rest. After the migration:

- `spec.gpus: 1` means **one time slice, not one card**. Four GpuJobs run at once instead of
  queueing. For compute-bound jobs that is slower on average (mean completion 149 s vs 101 s
  in the measured fixed-work run); for small or bursty jobs it removes the queue.
- `spec.gpus: 2` or more is **rejected by the device plugin** (`failRequestsGreaterThanOne: true`):
  two slices of the same card give no extra compute. The pod fails at admission on the node
  with an `UnexpectedAdmissionError`, and the GpuJob shows `Failed`. The CRD allows up to 8 for
  multi-GPU nodes, so this is a runtime rejection, not a validation error.
- **No memory isolation.** Every slice sees all 12 GB. `spec.memory` is the container's RAM, not
  VRAM; nothing caps VRAM per pod, so one job can OOM its neighbours. A 5 GB vLLM pod and two
  4 GB training jobs on one card is the kind of mix that breaks.
- **No fault isolation.** An XID error or GPU reset kills every pod sharing the card.
- **No MIG on the RTX 5070**, so hardware partitioning is not an option. MPS
  (`sharing.mps` in the same ConfigMap) is the step up: concurrent kernels plus per-client
  memory limits. Not tried here.
- The `Scheduled=False` / "queued: ... Insufficient nvidia.com/gpu" status the operator shows
  appears only once 4 slices are in use.

## Offline validation (run 2026-11-13 on the laptop)

```
$ bash scheduling/gpu-operator/validate.sh
== helm: pull chart v26.7.2, render
rendered: 28 objects
== keys: every values key exists in the chart
  values.yaml: ok
  values-toolkit-k3s.yaml: ok
  ClusterPolicy.driver.enabled = False
  ClusterPolicy.toolkit.enabled = False
  ClusterPolicy.cdi.enabled = False
  ClusterPolicy.migManager.enabled = False
  ClusterPolicy.dcgmExporter.enabled = False
  ClusterPolicy.devicePlugin.config.name = 'time-slicing-config'
  ClusterPolicy.devicePlugin.config.default = 'any'
  ClusterPolicy.operator.runtimeClass = 'nvidia'
== kubeconform
Summary: 69 resources found in 14 files - Valid: 67, Invalid: 0, Errors: 0, Skipped: 2
```

Helm 3.19.0 (`alpine/helm`), kubeconform v0.7.0 against Kubernetes 1.33.0 schemas (the newest
in its schema catalog; the node runs 1.36.5). The 2 skipped objects are the operator's own
custom resources (`ClusterPolicy`), for which kubeconform has no schema. kubeconform covers
`scheduling/manifests/` too.

What this does **not** prove: that the operands start on this node, that the toolkit-less setup
validates, or the `nvidia.com/gpu: 4` capacity. Those need the runbook on the real node.
