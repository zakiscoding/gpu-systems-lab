#!/bin/bash
# FSDP2 GPT (modal/fsdp_gpt.py, unchanged) on the RTX 5070 under Slurm: one rank, bf16.
# ~37M params so it fits next to the always-on vLLM (which holds ~5.4 GB) and desktop apps.
#SBATCH --job-name=fsdp-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5070:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/opt/slurm-lab/out/%x-%j.log
set -euo pipefail
LAB=/opt/slurm-lab
echo "job $SLURM_JOB_ID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES gres=$SLURM_JOB_GPUS"
srun "$LAB/venv/bin/python" -m torch.distributed.run --standalone --nproc-per-node=1 \
  "$LAB/lab/fsdp_gpt.py" --mode fsdp --d-model 512 --n-layers 8 --n-heads 8 --seq 512 --batch 8 \
  --vocab 8192 --steps ${STEPS:-60} --out "$LAB/out/$SLURM_JOB_NAME-$SLURM_JOB_ID.json"
