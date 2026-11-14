#!/usr/bin/env bash
# Builds a TensorRT-LLM engine for Qwen2.5-1.5B-Instruct and a Triton model
# repository (inflight_batcher_llm) around it. Runs INSIDE the Triton image:
#
#   docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
#     -v ~/nvidia-serving/work:/work -v "$PWD/serving/triton:/scripts:ro" \
#     nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3 bash /scripts/build_engine.sh
#
# Output: /work/ckpt (TRT-LLM checkpoint), /work/engine (rank0.engine), /work/repo (Triton models).
set -euo pipefail

HF_DIR=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/ | head -1)
DTYPE=${DTYPE:-bfloat16}          # same weights precision as the vLLM run (bf16)
MAX_BATCH=${MAX_BATCH:-256}       # highest concurrency in the sweep
MAX_SEQ=${MAX_SEQ:-4096}          # vLLM ran with --max-model-len 4096
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-8192}  # tokens per engine step (vLLM's default max_num_batched_tokens is 8192 here)
# KV-cache capacity in tokens. Defaults to what vLLM 0.32 allocated at
# --gpu-memory-utilization 0.80 on the same card, so both engines get the same cache.
KV_TOKENS=${KV_TOKENS:-207632}

echo "[build] model $HF_DIR"
t0=$(date +%s)
if [ "${SKIP_ENGINE:-0}" != 1 ]; then
python3 /app/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir "$HF_DIR" --output_dir /work/ckpt --dtype "$DTYPE"
t1=$(date +%s)

trtllm-build --checkpoint_dir /work/ckpt --output_dir /work/engine \
  --gemm_plugin auto \
  --max_batch_size "$MAX_BATCH" --max_seq_len "$MAX_SEQ" --max_num_tokens "$MAX_NUM_TOKENS" \
  --kv_cache_type paged --use_paged_context_fmha enable \
  --remove_input_padding enable
t2=$(date +%s)
echo "[build] convert $((t1 - t0)) s, engine build $((t2 - t1)) s"
du -sh /work/engine
fi

rm -rf /work/repo
cp -r /app/all_models/inflight_batcher_llm /work/repo
FILL="python3 /app/tools/fill_template.py -i"
$FILL /work/repo/preprocessing/config.pbtxt \
  "tokenizer_dir:${HF_DIR},triton_max_batch_size:${MAX_BATCH},preprocessing_instance_count:1"
$FILL /work/repo/postprocessing/config.pbtxt \
  "tokenizer_dir:${HF_DIR},triton_max_batch_size:${MAX_BATCH},postprocessing_instance_count:1"
$FILL /work/repo/tensorrt_llm/config.pbtxt \
  "triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH},decoupled_mode:True,max_beam_width:1,engine_dir:/work/engine,max_tokens_in_paged_kv_cache:${KV_TOKENS},kv_cache_free_gpu_mem_fraction:0.95,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,prompt_embedding_table_data_type:TYPE_FP16,enable_chunked_context:True,logits_datatype:TYPE_FP32"
$FILL /work/repo/ensemble/config.pbtxt \
  "triton_max_batch_size:${MAX_BATCH},logits_datatype:TYPE_FP32"
$FILL /work/repo/tensorrt_llm_bls/config.pbtxt \
  "triton_max_batch_size:${MAX_BATCH},decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,prompt_embedding_table_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"
echo "[build] model repository ready in /work/repo"
