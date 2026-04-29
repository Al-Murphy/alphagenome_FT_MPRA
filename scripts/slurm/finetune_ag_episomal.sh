#!/bin/bash
# Finetune AlphaGenome on Gosai episomal MPRA — array of 9 jobs (3 cells × 3 seeds).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-8 \
#       /grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA/scripts/slurm/finetune_ag_episomal.sh
#
#SBATCH --job-name=ag_epi
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5

REPO=/grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA
cd "$REPO"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ---- venv (reuse the AG venv from ALBench-S2F) ----
source /grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/bin/activate

# ---- Array index → (cell, seed) ----
CELLS=(K562 HepG2 SKNSH)
SEEDS=(42 1042 2042)
CELL=${CELLS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

# ---- Gosai data lives under per-cell ALBench-S2F dirs (one TSV per cell) ----
DATA_PATH=/grid/wsbs/home_norepl/christen/ALBench-S2F/data/$(echo "$CELL" | tr '[:upper:]' '[:lower:]')

# ---- AG checkpoint shared across cells ----
AG_CKPT=/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1

# ---- Run name & output ----
RUN_NAME="episomal-${CELL}-seed${SEED}"
CKPT_DIR="$REPO/results/models/checkpoints/episomal"

echo "=== AG episomal | cell=$CELL | seed=$SEED — $(date) ==="
python scripts/finetune_episomal_mpra.py \
    --config "configs/episomal_${CELL}.json" \
    --cell_type "$CELL" \
    --data_path "$DATA_PATH" \
    --base_checkpoint_path "$AG_CKPT" \
    --checkpoint_dir "$CKPT_DIR" \
    --wandb_name "$RUN_NAME" \
    --no_wandb \
    --seed "$SEED"
