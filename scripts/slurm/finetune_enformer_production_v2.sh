#!/bin/bash
#SBATCH --job-name=enf_prod2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
REPO=/grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA
cd "$REPO"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Force temp dir onto /grid so Lightion ckpt atomic-save (rename from
# /tmp/ to /grid/...) does not hit the cross-device EXDEV error.
export TMPDIR="$REPO/.tmp_lightning"
mkdir -p "$TMPDIR"
source /grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/bin/activate

CELLS=(K562 HepG2 SKNSH)
SEEDS=(42 1042 2042)
CELL=${CELLS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
DATA=/grid/wsbs/home_norepl/christen/ALBench-S2F/data/$(echo "$CELL" | tr "[:upper:]" "[:lower:]")
RUN_NAME="enformer-prod2-$CELL-seed$SEED"
CKPT_DIR="$REPO/results/models/checkpoints/episomal_enformer_production_v2"

echo "=== $RUN_NAME — $(date) ==="
echo "TMPDIR=$TMPDIR"

python scripts/finetune_enformer_episomal_mpra.py \
    --cell_type "$CELL" \
    --data_path "$DATA" \
    --checkpoint_dir "$CKPT_DIR" \
    --wandb_name "$RUN_NAME" \
    --no_wandb \
    --seed "$SEED" \
    --learning_rate 1e-3 \
    --weight_decay 1e-6 \
    --second_stage_lr 2e-4 \
    --second_stage_epochs 15 \
    --max_shift 10 \
    --optimizer adam \
    --early_stopping_patience 3 \
    --batch_size 128
