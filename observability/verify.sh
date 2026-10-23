#!/usr/bin/env bash
# End-to-end check of the observability stack. Run on the GPU node from observability/.
#   ./verify.sh          health + current values
#   ./verify.sh --load   also runs benchmarks/smoke_test.py and reports peak values during it
set -euo pipefail
BIND_IP=$(grep -s '^BIND_IP=' .env | cut -d= -f2); BIND_IP=${BIND_IP:-127.0.0.1}
P="http://$BIND_IP:9090"; G="http://$BIND_IP:3000"

q() {  # q <promql> -> "value" (first series) or "no data"
  curl -s --get "$P/api/v1/query" --data-urlencode "query=$1" |
    python3 -c 'import sys,json; r=json.load(sys.stdin)["data"]["result"]; print(r[0]["value"][1] if r else "no data")'
}

echo "== containers"; docker compose ps --format '{{.Name}}  {{.Status}}  {{.Ports}}'
echo "== exporter: $(curl -s 127.0.0.1:9400/metrics | grep -c '^DCGM_FI') DCGM series"
echo "== scrape targets"
curl -s "$P/api/v1/targets" | python3 -c 'import sys,json
for t in json.load(sys.stdin)["data"]["activeTargets"]: print(" ", t["labels"]["job"], t["health"])'
echo "== grafana: $(curl -s "$G/api/health" | python3 -c 'import sys,json; print(json.load(sys.stdin)["database"])')," \
     "dashboard: $(curl -s "$G/api/dashboards/uid/gpu-node" | python3 -c 'import sys,json; print(json.load(sys.stdin)["dashboard"]["title"])')"

declare -A Q=(
  [gpu_util_pct]='DCGM_FI_DEV_GPU_UTIL'
  [sm_active]='DCGM_FI_PROF_SM_ACTIVE'
  [tensor_active]='DCGM_FI_PROF_PIPE_TENSOR_ACTIVE'
  [mem_used_mib]='DCGM_FI_DEV_FB_USED'
  [mem_total_mib]='DCGM_FI_DEV_FB_TOTAL'
  [power_w]='DCGM_FI_DEV_POWER_USAGE'
  [temp_c]='DCGM_FI_DEV_GPU_TEMP'
  [sm_clock_mhz]='DCGM_FI_DEV_SM_CLOCK'
  [pcie_rx_Bps]='DCGM_FI_PROF_PCIE_RX_BYTES'
  [pcie_tx_Bps]='DCGM_FI_PROF_PCIE_TX_BYTES'
  [cpu_busy_pct]='100*(1-avg(rate(node_cpu_seconds_total{mode="idle"}[30s])))'
  [ram_used_gib]='(node_memory_MemTotal_bytes-node_memory_MemAvailable_bytes)/2^30'
)
KEYS=(gpu_util_pct sm_active tensor_active mem_used_mib mem_total_mib power_w temp_c sm_clock_mhz pcie_rx_Bps pcie_tx_Bps cpu_busy_pct ram_used_gib)

echo "== current values"
for k in "${KEYS[@]}"; do printf '  %-14s %s\n' "$k" "$(q "${Q[$k]}")"; done

if [[ "${1:-}" == "--load" ]]; then
  REPO=$(cd .. && pwd); start=$(date +%s)
  echo "== running smoke_test.py (load)"
  (cd "$REPO" && ./.venv/bin/python benchmarks/smoke_test.py >/tmp/obs-smoke.log 2>&1)
  sleep 12   # let the last samples get scraped
  win=$(( $(date +%s) - start ))s
  echo "== peak over the ${win} load window (max_over_time)"
  for k in gpu_util_pct sm_active tensor_active mem_used_mib power_w temp_c sm_clock_mhz pcie_rx_Bps pcie_tx_Bps; do
    expr=${Q[$k]}
    expr="max_over_time(${expr}[${win}])"
    printf '  %-14s %s\n' "$k" "$(q "$expr")"
  done
fi

if [[ "${1:-}" == "--load" ]]; then
  echo "== timeline (query_range, 5s step): t  gpu_util%  sm_active  power_W  temp_C  pcie_rx_MB/s"
  end=$(date +%s); beg=$(( end - ${win%s} - 10 ))
  for m in DCGM_FI_DEV_GPU_UTIL DCGM_FI_PROF_SM_ACTIVE DCGM_FI_DEV_POWER_USAGE DCGM_FI_DEV_GPU_TEMP DCGM_FI_PROF_PCIE_RX_BYTES; do
    curl -s --get "$P/api/v1/query_range" --data-urlencode "query=$m" \
      --data-urlencode "start=$beg" --data-urlencode "end=$end" --data-urlencode "step=5" > "/tmp/obs-$m.json"
  done
  python3 - "$beg" <<'PY'
import json, sys
ms = ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_PROF_PCIE_RX_BYTES"]
s = {m: dict(json.load(open(f"/tmp/obs-{m}.json"))["data"]["result"][0]["values"]) for m in ms}
beg = int(sys.argv[1])
for t in sorted(s[ms[0]]):
    v = [float(s[m].get(t, "nan")) for m in ms]
    print(f"  +{int(t)-beg:3d}s  {v[0]:5.0f}  {v[1]:6.2f}  {v[2]:6.1f}  {v[3]:4.0f}  {v[4]/1e6:9.1f}")
PY
fi
