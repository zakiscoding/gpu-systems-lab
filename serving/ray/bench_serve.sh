#!/usr/bin/env bash
# Ray Serve LLM vs plain vLLM, same image (rayproject/ray-llm: Ray 2.60.1 + vLLM 0.29.0), same
# model (Qwen2.5-0.5B-Instruct), same engine settings (12-15% of the RTX 5070, next to the live
# vllm-chat), same load (`vllm bench serve`, random 256 in / 128 out, concurrency 1/16/64).
# Runs inside the container; writes JSON per (engine, concurrency) to /out.
set -euo pipefail
MODEL=Qwen/Qwen2.5-0.5B-Instruct
UTIL=${GPU_UTIL:-0.15}
CONCS=${CONCS:-"1 16 64"}
NPROMPTS=${NPROMPTS:-400}
export HF_HUB_OFFLINE=1
mkdir -p /out

wait_ready() {  # $1 = base url; ready = a real completion succeeds (not just /health)
  for _ in $(seq 1 300); do
    curl -sf "$1/v1/completions" -H 'content-type: application/json' \
      -d '{"model":"qwen0.5b","prompt":"hi","max_tokens":4}' >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

bench() {  # $1 = engine name, $2 = port
  # warm-up at the largest batch first (cold kernels otherwise land in the first measurement)
  vllm bench serve --backend openai --base-url "http://127.0.0.1:$2" --model "$MODEL" --served-model-name qwen0.5b \
    --dataset-name random --random-input-len 256 --random-output-len 128 --num-prompts 128 \
    --max-concurrency 64 --ignore-eos >/dev/null 2>&1 || true
  for c in $CONCS; do
    for rep in 1 2 3; do
      vllm bench serve --backend openai --base-url "http://127.0.0.1:$2" --model "$MODEL" --served-model-name qwen0.5b \
        --dataset-name random --random-input-len 256 --random-output-len 128 --num-prompts $NPROMPTS \
        --max-concurrency "$c" --ignore-eos --seed $rep --save-result --result-dir /out \
        --result-filename "$1_c${c}_r${rep}.json" 2>&1 | grep -E "Output token throughput|Mean TTFT|Mean TPOT" | tr -s ' ' | tr '\n' ' '
      echo " <- $1 c=$c rep=$rep"
    done
  done
}

echo "== plain vLLM $(python -c 'import vllm;print(vllm.__version__)')"
vllm serve "$MODEL" --served-model-name qwen0.5b --port 8001 --gpu-memory-utilization "$UTIL" \
  --max-model-len 2048 --max-num-seqs 64 --no-enable-prefix-caching > /out/vllm_server.log 2>&1 &
VPID=$!
wait_ready http://127.0.0.1:8001
bench vllm 8001
kill $VPID; wait $VPID 2>/dev/null || true
sleep 5

echo "== Ray Serve LLM $(python -c 'import ray;print(ray.__version__)')"
cd /lab
GPU_UTIL=$UTIL serve run serve_config.yaml > /out/ray_server.log 2>&1 &
RPID=$!
wait_ready http://127.0.0.1:8002
bench rayserve 8002
kill $RPID; wait $RPID 2>/dev/null || true
ray stop --force >/dev/null 2>&1 || true
echo "== done"
