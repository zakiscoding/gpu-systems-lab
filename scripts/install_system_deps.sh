#!/usr/bin/env bash
# One-time root setup for roadmap steps 2-3. Run it yourself on the GPU node:
#   sudo bash scripts/install_system_deps.sh
#
# Installs:
#   - CUDA Toolkit 13.2 (nvcc, Nsight Systems/Compute) - toolkit only, the driver is left alone
#   - LLVM/Clang 21 (clang, llvm-dis, opt, lld) from apt.llvm.org
#   - Docker Engine (Ubuntu docker.io) + NVIDIA Container Toolkit
#
# Pinned to 13.2 because driver 595.99.02 reports CUDA 13.2; a newer toolkit
# would compile code the installed driver cannot run.
set -euo pipefail

CUDA_PKG="cuda-toolkit-13-2"
LLVM_VER=21
TARGET_USER="${SUDO_USER:-zak}"

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo: sudo bash $0" >&2
  exit 1
fi

step() { printf '\n==> %s\n' "$1"; }

step "NVIDIA CUDA apt repo"
KEYRING_DEB="/home/${TARGET_USER}/downloads/cuda-keyring_1.1-1_all.deb"
if [[ ! -f "$KEYRING_DEB" ]]; then
  KEYRING_DEB="$(mktemp --suffix=.deb)"
  wget -q -O "$KEYRING_DEB" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
fi
dpkg -i "$KEYRING_DEB"

step "LLVM ${LLVM_VER} apt repo"
install -d -m 0755 /etc/apt/keyrings
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor --yes -o /etc/apt/keyrings/llvm.gpg
echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VER} main" \
  > /etc/apt/sources.list.d/llvm-${LLVM_VER}.list

apt-get update

step "CUDA toolkit (${CUDA_PKG}) - no driver packages"
apt-get install -y --no-install-recommends "$CUDA_PKG"

step "LLVM/Clang ${LLVM_VER}"
apt-get install -y "clang-${LLVM_VER}" "llvm-${LLVM_VER}" "lld-${LLVM_VER}" "clang-tools-${LLVM_VER}"

step "Docker + NVIDIA Container Toolkit"
apt-get install -y docker.io docker-compose-v2 nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker
# lets TARGET_USER run docker without sudo (docker group is root-equivalent)
usermod -aG docker "$TARGET_USER"

step "PATH for CUDA"
cat > /etc/profile.d/cuda.sh <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
EOF

step "verify"
/usr/local/cuda/bin/nvcc --version | tail -2
"clang-${LLVM_VER}" --version | head -1
docker --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo
echo "Done. Log out and back in (or reconnect ssh) so the docker group and CUDA PATH apply."
