#!/usr/bin/env bash
# Bring the GPU PC's Windows-boot stack (WSL2 Ubuntu) back up and check it end to end.
# Run from the laptop (controller); it reaches the PC over the tailnet.
#
#   scripts/gpu-pc-up.sh            start WSL, start anything that's down, then check
#   scripts/gpu-pc-up.sh --check    only check, change nothing
#
# Safe to run any time: it never recreates or restarts a running container
# (compose up --no-recreate), it only starts what is stopped or missing.
#
# Most of this happens on its own after `wsl --shutdown` or a reboot: systemd starts
# Docker and Slurm, every container has restart=unless-stopped, Windows Tailscale keeps
# the `tailscale serve` ports, and the "WSL Keep Alive" task restarts WSL within 5 min.
# This script just does it now and tells you whether it worked.
set -uo pipefail

SSH_HOST="${GPU_PC_SSH:-zak-ts}"
TS_IP="${GPU_PC_TS_IP:-100.88.71.33}"
GATEWAY="${GATEWAY_URL:-https://vllm-gateway.vllm-gateway.workers.dev}"
VLLM_WAIT="${VLLM_WAIT:-420}"   # seconds to wait for vLLM to load the model
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

fail=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
quiet_ssh() { ssh -o ConnectTimeout=10 "$SSH_HOST" "$@" 2>&1 | tr -d '\r\0' \
  | grep -v -i 'post-quantum\|store now\|openssh.com/pq\|may need to be upgraded'; }

if (( ! CHECK_ONLY )); then
  echo "== Starting WSL on the GPU PC"
  # Kick the keep-alive task so WSL stays up after this SSH session ends.
  quiet_ssh 'schtasks /run /tn "WSL Keep Alive"' >/dev/null

  quiet_ssh "wsl -d Ubuntu -u root -- bash -s" <<EOF
set -u
for i in \$(seq 1 60); do systemctl is-active --quiet docker && break; sleep 2; done
systemctl is-active --quiet docker && echo "  docker active" || { echo "  docker did not start"; exit 1; }

for u in munge slurmctld slurmd; do
  systemctl is-active --quiet \$u || systemctl start \$u
  echo "  \$u: \$(systemctl is-active \$u)"
done
echo "  gg-nightly.timer: \$(systemctl is-active gg-nightly.timer)"

# up DIR <compose -f args> [-- service...]
up() {
  local d=\$1; shift; local f=()
  while [ \$# -gt 0 ] && [ "\$1" != -- ]; do f+=("\$1"); shift; done
  [ "\${1:-}" = -- ] && shift
  (cd "\$d" && docker compose "\${f[@]}" up -d --no-recreate "\$@" 2>&1 | grep -v -i 'running\|up-to-date' | sed 's/^/  /')
}
up /root/gpu-systems-lab/serving       -f compose.yml -f compose.wsl.yml
up /root/gpu-systems-lab/cloudflare    -f compose.yml -f compose.wsl.yml
# Grafana is left out on purpose: the iMac's Grafana shows these metrics.
up /root/gpu-systems-lab/observability -f docker-compose.yml -f compose.wsl.yml -- node-exporter dcgm-exporter prometheus
up /root/gophergrid                    -f gpu-pc.yml

echo "  waiting for vllm-chat to be healthy (model load takes a few minutes after a cold start)"
for i in \$(seq 1 \$(( $VLLM_WAIT / 5 ))); do
  s=\$(docker inspect -f '{{.State.Health.Status}}' vllm-chat 2>/dev/null)
  [ "\$s" = healthy ] && break; sleep 5
done
echo "  vllm-chat: \$(docker inspect -f '{{.State.Health.Status}}' vllm-chat 2>/dev/null)"
EOF
fi

echo "== Containers"
# The PC's SSH shell is cmd.exe, so anything with | or quotes goes through bash on stdin.
containers=$(quiet_ssh "wsl -d Ubuntu -u root -- bash -s" <<'EOF'
for c in vllm-chat cloudflared obs-prometheus obs-dcgm-exporter obs-node-exporter \
         gophergrid-scheduler-2-1 gophergrid-gpu-worker-1 gophergrid-cpu-worker-1; do
  printf '%s|%s\n' "$c" "$(docker ps --filter "name=^$c\$" --format '{{.Status}}')"
done
for u in docker munge slurmctld slurmd gg-nightly.timer; do printf '%s|%s\n' "$u" "$(systemctl is-active $u)"; done
EOF
)
# On demand: vLLM runs when you work on inference; the rest only for the nightly benchmark
# (regress/gpu-stack.sh). Stopped is normal in --check; full mode starts them all.
ON_DEMAND=" vllm-chat gophergrid-scheduler-2-1 gophergrid-gpu-worker-1 gophergrid-cpu-worker-1 obs-dcgm-exporter "
while IFS='|' read -r n s; do
  [[ -z "$n" ]] && continue
  if (( CHECK_ONLY )) && [[ -z "$s" && "$ON_DEMAND" == *" $n "* ]]; then
    ok "$(printf '%-26s %s' "$n" "off (on demand, nightly only)")"; continue
  fi
  case "$s" in
    Up*|active) ok "$(printf '%-26s %s' "$n" "$s")" ;;
    *)          bad "$(printf '%-26s %s' "$n" "${s:-not running}")" ;;
  esac
done <<< "$containers"
[[ -z "$containers" ]] && bad "could not reach WSL over SSH ($SSH_HOST)"

echo "== Over the tailnet ($TS_IP)"
code() { curl -s -o /dev/null -m 8 -w '%{http_code}' "$@"; }
c=$(code "http://$TS_IP:8000/health")
if [[ $c == 000 ]] && (( CHECK_ONLY )); then
  ok "vLLM off (on demand; the gateway uses its Workers AI fallback)"
else
  [[ $c == 200 ]] && ok "vLLM /health 200" || bad "vLLM /health $c"
  c=$(code "http://$TS_IP:8000/v1/models"); [[ $c == 401 ]] && ok "vLLM /v1/models 401 without a key (auth on)" || bad "vLLM /v1/models $c, expected 401"
fi
c=$(code "http://$TS_IP:9090/-/ready");    [[ $c == 200 ]] && ok "Prometheus ready" || bad "Prometheus $c"
for p in 8091 9102 9103; do
  c=$(code "http://$TS_IP:$p/metrics")
  if [[ $c == 200 ]]; then ok "GopherGrid :$p metrics"
  elif (( CHECK_ONLY )); then ok "GopherGrid :$p off (on demand, nightly only)"
  else bad "GopherGrid :$p $c"; fi
done

echo "== Public"
c=$(code "$GATEWAY/health"); [[ $c == 200 ]] && ok "LLM gateway /health 200" || bad "LLM gateway /health $c"

echo
if (( fail )); then echo "Something is down; see FAIL lines above."; exit 1; fi
echo "GPU PC stack is up."
