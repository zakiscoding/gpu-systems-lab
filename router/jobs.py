"""Named example jobs. Numbers for `smoke` come from this repo's runs; the
others are realistic shapes for showing the router's choices."""

from .route import Job

JOBS = {
    # benchmarks/smoke_test.py: ~1 GB peak VRAM, ~30 s on the RTX 5070
    "smoke": Job("smoke", vram_gb=1, local_minutes=0.5,
                 command="python benchmarks/smoke_test.py"),
    # LoRA fine-tune of a 7B model in bf16: weights alone are ~14 GB
    "finetune-7b": Job("finetune-7b", vram_gb=22, local_minutes=180, max_usd=5,
                       command="python training/train.py --model 7b"),
    # a big eval sweep that has to finish within the hour
    "eval-sweep": Job("eval-sweep", vram_gb=8, local_minutes=240, deadline_min=60, max_usd=3,
                      command="python eval/sweep.py"),
    # 2-GPU data-parallel training (needs NVLink to beat one GPU)
    "ddp-2gpu": Job("ddp-2gpu", vram_gb=10, local_minutes=60, gpus=2, max_usd=5,
                    command="python modal/ddp_train.py"),
}
