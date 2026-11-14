"""Ray Serve LLM app: the same model and engine settings as the plain `vllm serve` baseline in
bench_serve.sh, behind Ray Serve's OpenAI-compatible router (one replica, no autoscaling), so the
comparison isolates what Ray Serve adds on top of the vLLM engine.

    serve run serve_llm:app        # inside rayproject/ray-llm (Ray 2.60.1, vLLM 0.29.0)
"""

import os

from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

llm_config = LLMConfig(
    model_loading_config={"model_id": "qwen0.5b", "model_source": MODEL},
    deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
    engine_kwargs={
        "gpu_memory_utilization": float(os.environ.get("GPU_UTIL", "0.15")),
        "max_model_len": 2048,
        "max_num_seqs": 64,
        "enable_prefix_caching": False,
    },
    accelerator_type=None,
)

app = build_openai_app({"llm_configs": [llm_config]})
