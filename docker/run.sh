#!/usr/bin/env bash
# Roadmap step 3: build / run / validate the containerized benchmark.
# Run on the GPU node from anywhere inside the repo:
#   bash docker/run.sh build      # build the pinned image
#   bash docker/run.sh check      # driver + toolkit sanity check (nvidia-smi in a CUDA base image)
#   bash docker/run.sh run [args] # run benchmarks/smoke_test.py, JSON -> results/
#   bash docker/run.sh validate   # check + run + assert the checks passed and torch/GPU are as expected
#   bash docker/run.sh shell      # interactive shell in the image
#   bash docker/run.sh size       # print image size
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-gpu-systems-lab/bench:torch2.11.0-cu128}"
CHECK_IMAGE="nvidia/cuda:12.8.1-base-ubuntu24.04"
RESULTS="${RESULTS:-$REPO_ROOT/results}"

cmd="${1:-validate}"
shift || true

run_bench() {
  mkdir -p "$RESULTS"
  # --hostname tags the result file (smoke_<host>-docker_*.json) instead of a random container id
  docker run --rm --gpus all --ipc=host \
    --hostname "$(hostname)-docker" \
    -v "$RESULTS:/results" \
    "$IMAGE" python benchmarks/smoke_test.py --out /results "$@"
}

case "$cmd" in
  build)
    docker build -f "$REPO_ROOT/docker/Dockerfile" \
      --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
      -t "$IMAGE" "$REPO_ROOT"
    ;;
  check)
    docker run --rm --gpus all "$CHECK_IMAGE" nvidia-smi
    ;;
  run)
    run_bench "$@"
    ;;
  validate)
    docker run --rm --gpus all "$CHECK_IMAGE" nvidia-smi -L
    before="$(ls -1 "$RESULTS"/smoke_*.json 2>/dev/null | wc -l || true)"
    run_bench "$@"
    latest="$(ls -1t "$RESULTS"/smoke_*.json | head -1)"
    after="$(ls -1 "$RESULTS"/smoke_*.json | wc -l)"
    [[ "$after" -gt "$before" ]] || { echo "FAIL: no new result JSON written" >&2; exit 1; }
    docker run --rm -i -v "$RESULTS:/results:ro" "$IMAGE" python - "/results/$(basename "$latest")" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
env, checks = r["environment"], r["checks"]
problems = [k for k, v in checks.items() if k.endswith("_ok") and not v]
if env["torch"] != "2.11.0+cu128":
    problems.append(f"torch {env['torch']}")
if env["compute_capability"] != "12.0":
    problems.append(f"compute capability {env['compute_capability']}")
tf = {m["precision"]: m["tflops"] for m in r["matmul"]}
print(f"validate: {env['gpu']} | torch {env['torch']} | " +
      " ".join(f"{k}={v}" for k, v in tf.items()) +
      f" | H2D {r['transfer']['h2d_pinned_gib_s']} GiB/s")
if problems:
    sys.exit("FAIL: " + ", ".join(problems))
print(f"PASS ({sys.argv[1]})")
EOF
    ;;
  shell)
    docker run --rm -it --gpus all --ipc=host -v "$RESULTS:/results" "$IMAGE" bash
    ;;
  size)
    docker image ls "$IMAGE" --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
    ;;
  *)
    echo "usage: $0 {build|check|run|validate|shell|size} [smoke_test args]" >&2
    exit 2
    ;;
esac
