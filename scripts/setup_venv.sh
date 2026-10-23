#!/usr/bin/env bash
# Roadmap step 1: isolated Python env with a CUDA build of PyTorch.
# RTX 5070 is Blackwell (sm_120) and needs a cu128-or-newer wheel.
# Usage (on the GPU node, from the repo root): bash scripts/setup_venv.sh [cu128|cu130]
set -euo pipefail

CUDA_TAG="${1:-cu128}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -m pip install -r requirements.txt

python - <<'EOF'
import torch
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)} | arch list: {torch.cuda.get_arch_list()}")
EOF
