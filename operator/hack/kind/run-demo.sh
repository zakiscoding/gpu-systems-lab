#!/usr/bin/env bash
# Build the operator image and the tools image, then run demo.sh against a fresh kind
# cluster named gpujob-op (deleted at the end). Output: hack/kind/transcript.txt.
#   bash operator/hack/kind/run-demo.sh            (Docker Desktop, Linux containers)
#   KEEP_CLUSTER=1 bash operator/hack/kind/run-demo.sh   to leave the cluster up
set -euo pipefail
cd "$(dirname "$0")/../.."                       # operator/
op=$(pwd -W 2>/dev/null || pwd)
sched=$(cd ../scheduling && (pwd -W 2>/dev/null || pwd))
export MSYS_NO_PATHCONV=1

docker build -q -t gpujob-operator:dev .
docker build -q -t gpujob-op-tools:dev -f hack/kind/tools.Dockerfile hack/kind
docker run --rm --name gpujob-op-tools \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$op":/src -v "$sched":/scheduling:ro \
  -e CLUSTER=gpujob-op -e KEEP_CLUSTER="${KEEP_CLUSTER:-0}" \
  gpujob-op-tools:dev bash /src/hack/kind/demo.sh | tee hack/kind/transcript.txt
