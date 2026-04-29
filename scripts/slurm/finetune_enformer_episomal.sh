#!/bin/bash
# Finetune Enformer (conv-only) on Gosai episomal MPRA — 9 jobs (3 cells × 3 seeds).
# One run produces both Stage 1 (Probing) and Stage 2 (Fine-tuned) best ckpts.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-8 \
#       /grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA/scripts/slurm/finetune_enformer_episomal.sh
#
#SBATCH --job-name=enf_epi
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5

REPO=/grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA
cd "$REPO"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ---- venv ----
source /grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/bin/activate

# ---- Array index → (cell, seed) ----
CELLS=(K562 HepG2 SKNSH)
SEEDS=(42 1042 2042)
CELL=${CELLS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

DATA_PATH=/grid/wsbs/home_norepl/christen/ALBench-S2F/data/$(echo "$CELL" | tr '[:upper:]' '[:lower:]')

RUN_NAME="enformer-episomal-${CELL}-seed${SEED}"
CKPT_DIR="$REPO/results/models/checkpoints/episomal_enformer"

echo "=== Enformer episomal | cell=$CELL | seed=$SEED — $(date) ==="
python scripts/finetune_enformer_episomal_mpra.py \
    --config "configs/episomal_${CELL}.json" \
    --cell_type "$CELL" \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "$CKPT_DIR" \
    --wandb_name "$RUN_NAME" \
    --no_wandb \
    --second_stage_lr 1e-4 \
    --second_stage_epochs 30 \
    --seed "$SEED"
