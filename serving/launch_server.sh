#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server on the GPU node.
#
# Binds to 127.0.0.1 by default: reach it from the laptop with an SSH tunnel
#   ssh -N -L 8000:127.0.0.1:8000 gpu-ts
# or set HOST to the node's tailnet IP (tailscale ip -4) to serve on the
# tailnet only. Never 0.0.0.0: the API has no auth unless VLLM_API_KEY is set.
#
# Usage:
#   bash serving/launch_server.sh
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct PORT=8000 GPU_MEM_UTIL=0.80 bash serving/launch_server.sh
#   bash serving/launch_server.sh --no-enable-prefix-caching --generation-config vllm   # as benchmarked
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# The 5070 drives a desktop (Xorg, browser: ~1.2 GB) and is shared with other
# jobs, so leave headroom instead of vLLM's 0.90 default.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

if [[ "$HOST" == "0.0.0.0" ]]; then
    echo "refusing HOST=0.0.0.0; use 127.0.0.1 or the tailnet IP" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
API_KEY_ARGS=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then API_KEY_ARGS=(--api-key "$VLLM_API_KEY"); fi

exec vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${API_KEY_ARGS[@]}" "$@"
