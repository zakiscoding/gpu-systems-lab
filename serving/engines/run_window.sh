#!/usr/bin/env bash
# One GPU window on the GPU PC (WSL): stop the live vllm-chat, run the same sweep against five
# server configurations one at a time, then restore vllm-chat and verify it. The restore runs from
# an EXIT trap, so it also happens if anything below fails or the script is interrupted.
#
#   vllm-fa        vLLM 0.32.0, FlashAttention backend
#   vllm-fi        vLLM 0.32.0, FlashInfer backend
#   sglang-fi      SGLang 0.5.23, FlashInfer attention
#   sglang-triton  SGLang 0.5.23, Triton attention kernels
#   trtllm         Triton Inference Server + TensorRT-LLM 1.2.1 (engine from serving/triton/build_engine.sh)
#
# Same model (Qwen2.5-1.5B-Instruct bf16), same 80% memory budget, prefix/radix caching off, same client
# (serving/triton/bench_compare.py: pinned `vllm bench serve`, 512 in / 256 out, ignore_eos).
set -uo pipefail
# systemd units start without HOME, and the chat's compose file mounts ${HOME}/.cache/huggingface:
# an unset HOME made the first restore recreate vllm-chat with an empty model cache (crash loop).
export HOME=/root
REPO=${REPO:-/opt/engines-lab}
OUT=$REPO/results/engines
CHAT_DIR=/root/gpu-systems-lab/serving
MODEL=Qwen/Qwen2.5-1.5B-Instruct
NAME=qwen2.5-1.5b-instruct
UTIL=0.80
CONC=${CONC:-1,4,16,32,64,128,256}
TARGETS=${TARGETS:-"vllm-fa vllm-fi sglang-fi sglang-triton trtllm"}
# DRY_RUN=1: leave vllm-chat alone; start each engine with the 0.5B model at 15% of the GPU next to
# it and run two small levels, to prove every start command and the client path work.
DRY_RUN=${DRY_RUN:-0}
if [ "$DRY_RUN" = 1 ]; then
  MODEL=Qwen/Qwen2.5-0.5B-Instruct; UTIL=0.15; CONC=1,4; OUT=$OUT/dryrun
  TARGETS=${DRY_TARGETS:-"vllm-fa vllm-fi sglang-fi sglang-triton"}
fi
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/window.log"; }

restore() {
  docker rm -f eng-server triton-trtllm >/dev/null 2>&1
  if [ "$DRY_RUN" = 1 ]; then log "dry run done: vllm-chat untouched"; return; fi
  log "restore: starting vllm-chat"
  (cd "$CHAT_DIR" && docker compose -f compose.yml -f compose.wsl.yml up -d vllm >/dev/null 2>&1)
  for _ in $(seq 120); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' vllm-chat 2>/dev/null)" = healthy ] && break; sleep 5
  done
  log "restore: vllm-chat $(docker inspect -f '{{.State.Health.Status}}' vllm-chat 2>/dev/null)"
  log "verify: local /health $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)"
  log "verify: tailnet :8000 /v1/models without key $(curl -s -o /dev/null -w '%{http_code}' http://100.88.71.33:8000/v1/models) (401 expected)"
  log "verify: gateway /health $(curl -s -o /dev/null -w '%{http_code}' https://vllm-gateway.vllm-gateway.workers.dev/health)"
  log "WINDOW END"
}
trap restore EXIT

wait_up() {  # $1 = base url, $2 = timeout s
  for _ in $(seq $(( $2 / 3 ))); do
    curl -sf "$1/v1/models" >/dev/null 2>&1 && return 0
    docker ps -q -f name=eng-server -f name=triton-trtllm | grep -q . || return 1
    sleep 3
  done
  return 1
}

start() {  # $1 = target; prints base url
  docker rm -f eng-server >/dev/null 2>&1
  local hf=/root/.cache/huggingface
  local common=(--gpus all --ipc host --shm-size 4g -e HF_HUB_OFFLINE=1 -v "$hf:/root/.cache/huggingface")
  case $1 in
    vllm-fa|vllm-fi)
      local be=FLASH_ATTN; [ "$1" = vllm-fi ] && be=FLASHINFER
      docker run -d --name eng-server "${common[@]}" -e VLLM_ATTENTION_BACKEND=$be -p 127.0.0.1:8010:8000 \
        vllm/vllm-openai:v0.32.0 --model $MODEL --served-model-name $NAME --gpu-memory-utilization $UTIL \
        --max-model-len 4096 --max-num-seqs 256 --no-enable-prefix-caching --generation-config vllm >/dev/null
      echo http://127.0.0.1:8010 ;;
    sglang-fi|sglang-triton)
      local be=flashinfer; [ "$1" = sglang-triton ] && be=triton
      docker run -d --name eng-server "${common[@]}" -p 127.0.0.1:30000:30000 lmsysorg/sglang:v0.5.23-runtime \
        python3 -m sglang.launch_server --model-path $MODEL --served-model-name $NAME --host 0.0.0.0 --port 30000 \
        --mem-fraction-static $UTIL --context-length 4096 --max-running-requests 256 \
        --attention-backend $be --disable-radix-cache >/dev/null
      echo http://127.0.0.1:30000 ;;
    trtllm)
      WORK=/root/nvidia-serving/work bash "$REPO/serving/triton/serve_triton.sh" >/dev/null 2>&1
      echo http://127.0.0.1:9000 ;;
  esac
}

if [ "$DRY_RUN" != 1 ]; then
  log "WINDOW START: stopping vllm-chat (user-approved in the items-6-21 session)"
  docker stop vllm-chat >/dev/null
fi
for t in $TARGETS; do
  log "$t: starting"
  url=$(start $t)
  if ! wait_up "$url" 600; then
    log "$t: FAILED to start"; docker logs --tail 40 eng-server > "$OUT/${t}_startup_failure.log" 2>&1; continue
  fi
  log "$t: up at $url"
  model=$NAME; [ $t = trtllm ] && model=ensemble
  # one untimed warm-up at the top level, so cold kernels don't land in the measurement
  python3 "$REPO/serving/triton/bench_compare.py" --label "$t-warmup" --base-url "$url" --model $model \
    --concurrency 256 --out-dir "$OUT/warmup" >/dev/null 2>&1
  python3 "$REPO/serving/triton/bench_compare.py" --label "$t" --base-url "$url" --model $model \
    --concurrency "$CONC" --out-dir "$OUT" --notes "see run_window.sh: $t" 2>&1 | grep "\[bench\]   " | tee -a "$OUT/window.log"
  docker logs eng-server > "$OUT/${t}_server.log" 2>&1 || docker logs triton-trtllm > "$OUT/${t}_server.log" 2>&1
  docker rm -f eng-server triton-trtllm >/dev/null 2>&1
  log "$t: done"
done
