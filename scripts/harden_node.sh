#!/usr/bin/env bash
# One-time hardening for zak-gpu-node, run by the owner (not by automation):
#   sudo bash scripts/harden_node.sh
#
# 1. Firewall: block SSH (22), the k3s API (6443) and the kubelet (10250) on
#    the LAN interfaces. Everything else stays allowed, so Tailscale (incl.
#    Tailscale SSH), Docker, and k3s pod networking are unaffected; the
#    Grafana/Prometheus ports are already bound to the tailnet IP only.
# 2. Nsight Compute: let non-root users read GPU performance counters
#    (takes effect after the next reboot).
# 3. Remove the passwordless-sudo rule added during setup.
#
# Undo: `ufw disable`; `rm /etc/modprobe.d/nvidia-profiling.conf && update-initramfs -u`.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "run with sudo: sudo bash $0" >&2; exit 1; }

LAN_IFACES="${LAN_IFACES:-wlp5s0 enp4s0 enx94105ab3333c}"  # Wi-Fi + both Ethernet ports
BLOCKED_PORTS="22,6443,10250"

step() { printf '\n==> %s\n' "$1"; }

step "Firewall: default allow, block ${BLOCKED_PORTS}/tcp on: ${LAN_IFACES}"
ufw default allow incoming
ufw default allow outgoing
ufw default allow routed
for iface in $LAN_IFACES; do
  ufw deny in on "$iface" to any port "$BLOCKED_PORTS" proto tcp
done
ufw --force enable
ufw status verbose

step "Nsight Compute: allow profiling counters for non-root users (after reboot)"
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf
update-initramfs -u
echo "now: $(grep RmProfilingAdminOnly /proc/driver/nvidia/params)  (becomes 0 after reboot)"

step "Remove passwordless sudo for zak"
rm -f /etc/sudoers.d/zak-nopasswd '/etc/sudoers.d/zak-nopasswd~'
visudo -c
ls /etc/sudoers.d/

echo
echo "Done. Reboot when convenient to apply the profiling setting:  sudo reboot"
