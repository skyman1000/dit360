#!/bin/bash
#SBATCH --job-name=dit360-stitched1092
#SBATCH --partition=debug
#SBATCH --nodelist=GPU3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360/DiT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/DiT360/outputs/slurm/stitched1092_%j.log
#SBATCH --error=/data-nfs/gpu1-2/u13529658780/DIT360/DiT360/outputs/slurm/stitched1092_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}; start $(date --iso-8601=seconds)"
# Use the existing dit360 environment without relying on interactive conda activation.
srun --unbuffered /data-nfs/gpu1-2/u13529658780/DIT360/DiT360/.venv/bin/python -u inference_benchmark.py \
  --prompts ../benchmark_assets/mp3d_stitched1092/prompts.jsonl \
  --output outputs/mp3d_stitched1092_perid_g3_s28 \
  --seed-mode per-id \
  --seed 0
echo "Completed $(date --iso-8601=seconds)"
