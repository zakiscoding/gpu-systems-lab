#!/usr/bin/env bash
# Submit 3 one-GPU Jobs at once and record how the scheduler runs them.
# Usage: bash run_queue_demo.sh <label> [jobs.yaml]      (label names the output directory)
#   queue-jobs.yaml            3 jobs x 45 s of matmuls (fixed time)  -> label demo=gpu-queue
#   queue-jobs-fixedwork.yaml  3 jobs x 2800 matmuls (fixed work)     -> label demo=gpu-queue-fixed
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/config}
cd "$(dirname "$0")"
label=${1:-run}
file=${2:-queue-jobs.yaml}
sel=$(grep -m1 -o "demo: [a-z-]*" "$file" | sed "s/: /=/")
out=out-$label
mkdir -p "$out"

kubectl -n gpu-jobs delete job -l "$sel" --ignore-not-found --wait=true >/dev/null
echo "allocatable nvidia.com/gpu: $(kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}')" | tee "$out/summary.txt"

# timestamped pod watch
kubectl -n gpu-jobs get pods -l "$sel" -w --no-headers 2>/dev/null \
  | while read -r line; do echo "$(date +%T) $line"; done > "$out/watch.txt" &
watcher=$!

date +%T | sed 's/^/submitted at /' | tee -a "$out/summary.txt"
kubectl apply -f "$file"
kubectl -n gpu-jobs wait --for=condition=complete job -l "$sel" --timeout=900s
date +%T | sed 's/^/all complete at /' | tee -a "$out/summary.txt"
sleep 2; kill $watcher 2>/dev/null || true

pods=$(kubectl -n gpu-jobs get pods -l "$sel" -o jsonpath='{.items[*].metadata.name}' | tr ' ' '|')
kubectl -n gpu-jobs get events --sort-by=.metadata.creationTimestamp \
  -o custom-columns=TIME:.metadata.creationTimestamp,COUNT:.count,OBJECT:.involvedObject.name,REASON:.reason,MESSAGE:.message \
  | grep -E "^TIME|$pods" > "$out/events.txt" || true
python3 timeline.py "$sel" gpu-jobs | tee -a "$out/summary.txt"
