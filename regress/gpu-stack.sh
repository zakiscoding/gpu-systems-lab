#!/usr/bin/env bash
# Start or stop the GPU PC's on-demand containers around the nightly benchmark.
# Called by the gg-nightly.service drop-in (gg-nightly.on-demand.conf); safe to run by hand.
#
#   gpu-stack.sh up     start them, wait for the GPU worker; remembers if they were already up
#   gpu-stack.sh down   stop them again, unless they were already running before `up`
#                       (you were working on the PC) or GG_KEEP_UP=1
#
# Prometheus and node-exporter are not touched: the nightly result reaches the iMac
# through them, and the regression alerts are evaluated there.
set -uo pipefail

CONTAINERS=(gophergrid-scheduler-2-1 gophergrid-gpu-worker-1 gophergrid-cpu-worker-1 obs-dcgm-exporter)
STATE=/run/gg-nightly/was-up
log() { echo "gpu-stack: $*"; }

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

case "${1:-}" in
  up)
    mkdir -p "$(dirname "$STATE")"; rm -f "$STATE"
    if running gophergrid-gpu-worker-1; then
      touch "$STATE"; log "already up, leaving it running afterwards"
    fi
    docker start "${CONTAINERS[@]}" >/dev/null
    # The worker's metrics port answers once it has started and registered.
    for _ in $(seq 1 30); do
      curl -sf -o /dev/null -m 3 http://127.0.0.1:9102/metrics && break; sleep 2
    done
    curl -sf -o /dev/null -m 3 http://127.0.0.1:9102/metrics || { log "GPU worker did not come up"; exit 1; }
    sleep 10   # one heartbeat, so the scheduler sees the worker before the job arrives
    log "up: ${CONTAINERS[*]}"
    ;;
  down)
    if [ -e "$STATE" ] || [ "${GG_KEEP_UP:-0}" = 1 ]; then
      log "was up before the run, leaving it running"; exit 0
    fi
    docker stop "${CONTAINERS[@]}" >/dev/null
    log "stopped: ${CONTAINERS[*]}"
    ;;
  *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac
