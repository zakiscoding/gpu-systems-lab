#!/usr/bin/env bash
# Run a command inside golang:1.26 with operator/ bind-mounted at /src, e.g.
#   ./hack/in-docker.sh make generate
#   ./hack/in-docker.sh make test
# Module and build caches live in named Docker volumes so reruns are fast.
set -euo pipefail
cd "$(dirname "$0")/.."
src=$(pwd -W 2>/dev/null || pwd)   # Git Bash on Windows: docker wants the C:/... form
export MSYS_NO_PATHCONV=1
if [ $# -eq 0 ]; then set -- make all; fi
exec docker run --rm \
  -v "$src":/src -w /src \
  -v gpujob-op-gomod:/go/pkg/mod -v gpujob-op-gocache:/root/.cache \
  -e ENVTEST_DIR=/root/.cache/envtest \
  golang:1.26 "$@"
