#!/usr/bin/env bash
# Roadmap step 0: read-only safety and identity baseline for the GPU node.
# Usage (from laptop): ssh gpu-ts 'bash -s' < scripts/check_env.sh
set -u

section() { printf '\n== %s\n' "$1"; }

section "host"
hostname
uname -r
. /etc/os-release && echo "$PRETTY_NAME"

section "tailscale"
tailscale ip -4 2>/dev/null || echo "tailscale CLI not available"

section "disks"
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL
echo
echo "root filesystem: $(findmnt -n -o SOURCE /)"
if findmnt -rn -o SOURCE | grep -qi nvme; then
  echo "WARNING: an NVMe device is mounted (internal Windows drive should stay unmounted):"
  findmnt -rn -o SOURCE,TARGET | grep -i nvme
else
  echo "OK: no NVMe partitions mounted"
fi

section "gpu"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,pcie.link.gen.current --format=csv
nvidia-smi | sed -n '1,4p'

section "cuda toolkit"
command -v nvcc >/dev/null && nvcc --version | tail -2 || echo "nvcc not installed (fine until custom kernels)"

section "python"
python3 --version
if python3 -m venv --help >/dev/null 2>&1 && python3 -c 'import ensurepip' 2>/dev/null; then
  echo "venv + ensurepip available"
else
  echo "MISSING: python3-venv (sudo apt install python3.12-venv)"
fi

section "tools"
git --version 2>/dev/null || echo "git missing"
docker --version 2>/dev/null || echo "docker not installed (step 3)"

section "resources"
free -h | sed -n '1,2p'
echo "cpus: $(nproc)"
df -h / | tail -1
