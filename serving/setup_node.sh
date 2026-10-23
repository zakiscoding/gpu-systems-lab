#!/usr/bin/env bash
# Create a dedicated venv for vLLM on the GPU node.
#
# vLLM pins its own torch build, so it gets its own venv (~/venvs/vllm by
# default) and never touches the repo's .venv (torch 2.11.0+cu128).
#
# Usage:
#   bash serving/setup_node.sh                 # latest vLLM release from PyPI
#   VLLM_SPEC="vllm==X.Y.Z" bash serving/setup_node.sh
#   VLLM_NIGHTLY=1 bash serving/setup_node.sh  # nightly wheels (fallback for new GPUs)
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/venvs/vllm}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"

# Plain venv + uv inside it: nothing is installed system-wide, no sudo.
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip uv

if [[ "${VLLM_NIGHTLY:-0}" == "1" ]]; then
    uv pip install -U vllm --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/nightly
else
    uv pip install "$VLLM_SPEC" --torch-backend=auto
fi

python - <<'PY'
import torch, vllm
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), "sm_%d%d" % torch.cuda.get_device_capability(0))
print("arch list", torch.cuda.get_arch_list())
PY
