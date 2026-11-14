#!/usr/bin/env bash
# Drive the LB lab on the GPU node (WSL): bring it up, measure overhead, then two failover runs
# (HAProxy L7 retries on, then off). In each run, r1 is SIGKILLed 20 s in and started again at
# 40 s. Results go to ./out. Leaves vllm-chat alone; `docker compose down` at the end.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
DC="docker compose -p lb-lab -f compose.yml"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a out/timeline.log; }

$DC up -d --wait
log "up: r1, r2, haproxy, client"
until $DC exec -T client python -c "import httpx" 2>/dev/null; do sleep 2; done

log "overhead: start (first requests right after startup)"
$DC exec -T client python /lab/loadtest.py overhead | tee out/overhead-run.log || log "overhead: FAILED (kept going)"

for mode in retries noretries; do
  if [ "$mode" = noretries ]; then
    sed -e '/^    retry-on/d' -e 's/^    retries 2/    retries 0/' haproxy.cfg > out/haproxy-noretries.cfg
    HAPROXY_CFG=./out/haproxy-noretries.cfg $DC up -d --no-deps --force-recreate haproxy
    sleep 3
  fi
  log "failover-$mode: load start"
  $DC exec -T client python /lab/loadtest.py failover "failover-$mode" 100 &
  load=$!
  sleep 20; log "failover-$mode: KILL r1"; docker kill lb-lab-r1-1 >/dev/null
  sleep 20; log "failover-$mode: START r1"; docker start lb-lab-r1-1 >/dev/null
  until [ "$(docker inspect -f '{{.State.Health.Status}}' lb-lab-r1-1)" = healthy ]; do sleep 1; done
  log "failover-$mode: r1 healthy again"
  wait $load
  log "failover-$mode: load done"
done
$DC down
log "down"
