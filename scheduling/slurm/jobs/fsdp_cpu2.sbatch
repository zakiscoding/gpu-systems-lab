#!/bin/bash
# The same FSDP2 code path with 2 ranks on CPU (gloo), to exercise real sharding
# (the node has one GPU). Asks for no GPU, so it runs while a GPU job holds the card.
#SBATCH --job-name=fsdp-cpu2
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/opt/slurm-lab/out/%x-%j.log
set -euo pipefail
LAB=/opt/slurm-lab
export OMP_NUM_THREADS=4
srun "$LAB/venv/bin/python" -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$LAB/lab/fsdp_gpt.py" --device cpu --mode fsdp --d-model 128 --n-layers 4 --n-heads 4 --seq 128 \
  --batch 4 --vocab 2048 --no-bf16 --steps 20 --out "$LAB/out/$SLURM_JOB_NAME-$SLURM_JOB_ID.json"
