#!/usr/bin/env bash
# Starts Triton Inference Server (TensorRT-LLM backend) with Triton's
# OpenAI-compatible frontend on 127.0.0.1:9000, using the model repository
# that build_engine.sh wrote to ~/nvidia-serving/work/repo.
#
#   bash serving/triton/serve_triton.sh        # start (detached container "triton-trtllm")
#   docker rm -f triton-trtllm                 # stop
#
# Localhost only, like the vLLM bench server. The frontend serves the
# "ensemble" model (preprocessing -> tensorrt_llm -> postprocessing).
set -euo pipefail
IMAGE=nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3
WORK=${WORK:-$HOME/nvidia-serving/work}
HF_DIR=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/ | head -1)
HF_DIR=/root/.cache/huggingface/hub/${HF_DIR#*/hub/}

docker rm -f triton-trtllm 2>/dev/null || true
docker run -d --name triton-trtllm --gpus all --ipc host --shm-size 2g \
  -p 127.0.0.1:9000:9000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$WORK:/work" \
  -w /opt/tritonserver/python/openai/openai_frontend \
  --entrypoint python3 "$IMAGE" \
  main.py --model-repository /work/repo --backend tensorrtllm \
    --tokenizer "$HF_DIR" --host 0.0.0.0 --openai-port 9000
# --host 0.0.0.0 is inside the container only; the published port binds 127.0.0.1.

until curl -sf 127.0.0.1:9000/v1/models >/dev/null; do
  if ! docker ps -q -f name=triton-trtllm | grep -q .; then docker logs --tail 50 triton-trtllm; exit 1; fi
  sleep 3
done
curl -s 127.0.0.1:9000/v1/models
