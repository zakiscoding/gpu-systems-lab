#!/bin/bash
# NCCL all_reduce sweep + DDP loop (modal/ddp_train.py, unchanged) on the one local GPU.
# With a single rank this only proves NCCL initializes and runs under Slurm on the node;
# the 2-GPU bandwidth numbers are from Modal (docs/NCCL-TESTS.md).
#SBATCH --job-name=nccl-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5070:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=/opt/slurm-lab/out/%x-%j.log
set -euo pipefail
LAB=/opt/slurm-lab
echo "job $SLURM_JOB_ID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
srun "$LAB/venv/bin/python" -m torch.distributed.run --standalone --nproc-per-node=1 \
  "$LAB/lab/ddp_train.py" --max-mib 256 --width 2048 --depth 4 --batch 256 --steps 60 \
  --out "$LAB/out/$SLURM_JOB_NAME-$SLURM_JOB_ID.json"
