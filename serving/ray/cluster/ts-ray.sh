#!/usr/bin/env bash
# Join the tailnet (ephemeral auth key from /run/secrets/ts_authkey, in-memory state, so the node
# disappears when the container stops), then start Ray on the tailnet address.
#   ROLE=head                      -> ray start --head
#   ROLE=worker HEAD_IP=100.x.y.z  -> ray start --address=$HEAD_IP:6379
set -euo pipefail
S=/tmp/tailscaled.sock
tailscaled --tun=tailscale0 --state=mem: --socket=$S >/tmp/tailscaled.log 2>&1 &
for _ in $(seq 30); do [ -S $S ] && break; sleep 1; done
tailscale --socket=$S up --authkey="file:/run/secrets/ts_authkey" --hostname="${TS_HOSTNAME:?}" --accept-dns=false --timeout=60s
IP=$(tailscale --socket=$S ip -4)
echo "tailnet ip: $IP"
COMMON=(--node-ip-address="$IP" --num-cpus="${NUM_CPUS:-4}" --disable-usage-stats)
if [ "${ROLE:-worker}" = head ]; then
  ray start --head --port=6379 --dashboard-host=0.0.0.0 "${COMMON[@]}"
else
  ray start --address="${HEAD_IP:?}:6379" "${COMMON[@]}"
fi
# stay in the foreground; leave the tailnet on stop
trap 'ray stop --force; tailscale --socket=$S logout; exit 0' TERM INT
sleep infinity & wait
