#!/usr/bin/env bash
# The kind demo itself. Runs INSIDE the tools container (see run-demo.sh), with the
# Docker socket mounted, operator/ at /src and scheduling/ at /scheduling.
# Every command is echoed with a timestamp so the output doubles as the transcript.
set -euo pipefail
CLUSTER=${CLUSTER:-gpujob-op}
NODE=$CLUSTER-control-plane
NS=gpu-jobs
cd /src

ts() { date -u +%H:%M:%S; }
run() { echo; echo "[$(ts)] \$ $*"; "$@" 2>&1; }
note() { echo; echo "[$(ts)] # $*"; }

cleanup() {
  [ -n "${WATCH_PID:-}" ] && kill "$WATCH_PID" 2>/dev/null || true
  if [ "${KEEP_CLUSTER:-0}" != 1 ]; then
    note "deleting the kind cluster $CLUSTER"
    kind delete cluster --name "$CLUSTER" 2>&1 || true
  fi
}
trap cleanup EXIT

note "1. kind cluster (one node, like the k3s box); no GPU on this machine"
run kind create cluster --name "$CLUSTER" --wait 120s
# Reach the API server over the kind Docker network from this container.
docker network connect kind "$(hostname)" 2>/dev/null || true
kind get kubeconfig --internal --name "$CLUSTER" > /tmp/kubeconfig
export KUBECONFIG=/tmp/kubeconfig
run kubectl version
run kubectl get nodes -o wide

note "2. fake GPU: advertise nvidia.com/gpu: 1 on the node (status patch; no device plugin)"
run kubectl patch node "$NODE" --subresource=status --type=json \
  -p '[{"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"1"}]'
until [ "$(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')" = 1 ]; do sleep 1; done
run kubectl get node "$NODE" -o jsonpath='capacity={.status.capacity.nvidia\.com/gpu} allocatable={.status.allocatable.nvidia\.com/gpu}{"\n"}'

note "3. stand-in RuntimeClass nvidia (handler runc), the repo's namespace and admission policy"
run kubectl apply -f hack/kind/runtimeclass-runc.yaml
run kubectl apply -f /scheduling/manifests/00-namespace.yaml -f /scheduling/manifests/50-require-gpu-limit-policy.yaml
sleep 3
note "policy check: an nvidia-runtime pod without a GPU limit must be denied"
run kubectl apply --dry-run=server -f hack/kind/no-limit-pod.yaml || true

note "4. install the operator (image loaded into the node, not pulled)"
run kind load docker-image gpujob-operator:dev --name "$CLUSTER"
run kubectl apply -k config/default
run kubectl -n gpujob-system rollout status deploy/gpujob-operator --timeout=120s
run kubectl get crd gpujobs.gpu.zakiscoding.dev

note "5. two GpuJobs at once, one fake GPU: demo-b must queue behind demo-a"
kubectl -n $NS get gpujobs -w --output-watch-events 2>&1 | while IFS= read -r line; do echo "[$(ts)] watch | $line"; done &
WATCH_PID=$!
sleep 1
run kubectl apply -f hack/kind/demo-jobs.yaml
sleep 8
run kubectl -n $NS get gpujobs -o wide
run kubectl -n $NS get jobs,pods -o wide
run kubectl -n $NS get gpujob demo-b -o jsonpath='{range .status.conditions[*]}{.type}={.status} {.reason}: {.message}{"\n"}{end}'
run kubectl -n $NS wait gpujob/demo-a --for=jsonpath='{.status.phase}'=Succeeded --timeout=180s
run kubectl -n $NS wait gpujob/demo-b --for=jsonpath='{.status.phase}'=Running --timeout=60s
run kubectl -n $NS wait gpujob/demo-b --for=jsonpath='{.status.phase}'=Succeeded --timeout=180s
run kubectl -n $NS get gpujobs -o wide
run kubectl -n $NS logs job/demo-a
run kubectl -n $NS get events --field-selector involvedObject.kind=Pod --sort-by=.lastTimestamp \
  -o custom-columns=TIME:.lastTimestamp,POD:.involvedObject.name,REASON:.reason,MESSAGE:.message

note "6. a failing job: exit 3, backoffLimit 0"
run kubectl apply -f hack/kind/demo-fail.yaml
run kubectl -n $NS wait gpujob/demo-fail --for=jsonpath='{.status.phase}'=Failed --timeout=180s
run kubectl -n $NS get gpujob demo-fail -o wide

note "operator log so far: one 'created Job' line per GpuJob"
kubectl -n gpujob-system logs deploy/gpujob-operator 2>&1 | grep -E '"created Job"|starting manager' | cut -c1-220 || true

note "7. idempotency on a live cluster: restart the operator, still exactly one Job per GpuJob"
run kubectl -n gpujob-system rollout restart deploy/gpujob-operator
run kubectl -n gpujob-system rollout status deploy/gpujob-operator --timeout=120s
sleep 10
run kubectl -n $NS get jobs -L gpu.zakiscoding.dev/gpujob
note "the new operator pod re-reconciled every GpuJob and created nothing:"
kubectl -n gpujob-system logs deploy/gpujob-operator 2>&1 | grep -E 'acquired|Starting workers|"created Job"' | cut -c1-220 || true
echo "created-Job lines after restart: $(kubectl -n gpujob-system logs deploy/gpujob-operator 2>&1 | grep -c '"created Job"' || true)"

note "8. cascade delete: deleting a GpuJob removes its Job and pod (owner reference + GC)"
run kubectl -n $NS delete gpujob demo-a --wait=true
sleep 5
run kubectl -n $NS get jobs,pods -l gpu.zakiscoding.dev/gpujob=demo-a

note "9. spec is immutable"
run kubectl -n $NS patch gpujob demo-b --type=merge -p '{"spec":{"gpus":2}}' || true

kill "$WATCH_PID" 2>/dev/null || true; WATCH_PID=
