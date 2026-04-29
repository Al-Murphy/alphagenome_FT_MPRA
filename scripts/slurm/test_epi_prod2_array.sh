#!/bin/bash
#SBATCH --job-name=test_epi_p2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#
# Evaluate the 9 production_v2 Enformer ckpts on all 3 episomal test sets
# (in-distribution / SNV-effects / OOD-designed) per cell × seed × stage.
# Output predictions are tagged with "prod2" in the run_name to keep them
# distinct from any older eval runs in results/episomal_predictions/.

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5

REPO=/grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA
cd "$REPO"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source /grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/bin/activate

# Same array layout as finetune_enformer_production_v2.sh (3 cells × 3 seeds).
CELLS=(K562 HepG2 SKNSH)
SEEDS=(42 1042 2042)
CELL=${CELLS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
DATA=/grid/wsbs/home_norepl/christen/ALBench-S2F/data/$(echo "$CELL" | tr '[:upper:]' '[:lower:]')

CKPT_BASE="$REPO/results/models/checkpoints/episomal_enformer_production_v2/$CELL/enformer-prod2-$CELL-seed$SEED"
S1_DIR="$CKPT_BASE"
S2_DIR="$CKPT_BASE/stage2"
OUT="$REPO/results/episomal_predictions"
mkdir -p "$OUT"

# Pick the lowest-val_loss ckpt by parsing val_loss=X.YYYY out of the filename,
# skipping truncated/zero-byte files (Lightning sometimes saves a 0B ckpt next
# to the real one when atomic-save races with disk-quota errors; the real one
# is then written as ``...-v1.ckpt`` with the same val_loss).
pick_best_ckpt() {
    local dir=$1 prefix=$2 minsize=104857600   # 100 MB
    for f in "$dir"/${prefix}*-val_loss=*.ckpt; do
        [ -f "$f" ] || continue
        sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
        if [ "$sz" -ge "$minsize" ]; then
            vl=$(echo "$f" | awk -F'val_loss=' '{print $2}' | sed 's/[^0-9.].*$//')
            printf "%s\t%s\n" "$vl" "$f"
        fi
    done | sort -k1n | head -1 | cut -f2
}

S1_CKPT=$(pick_best_ckpt "$S1_DIR" "best-epoch=" || true)
S2_CKPT=$(pick_best_ckpt "$S2_DIR" "best-stage2-epoch=" || true)

echo "=== prod2 eval: $CELL seed=$SEED ==="
echo "  S1: ${S1_CKPT:-<missing>}"
echo "  S2: ${S2_CKPT:-<missing>}"
echo "  data: $DATA"
echo "  out:  $OUT"

if [ -z "$S1_CKPT" ] || [ ! -f "$S1_CKPT" ]; then
    echo "Stage 1 checkpoint not found for $CELL seed=$SEED — skipping."
else
    echo '=== Stage 1 (probing) ==='
    python scripts/test_episomal_mpra.py \
        --model_type enformer_probing \
        --checkpoint_path "$S1_CKPT" \
        --cell_type "$CELL" \
        --data_path "$DATA" \
        --output_dir "$OUT" \
        --run_name "enformer-prod2-$CELL-seed$SEED-stage1"
fi

if [ -z "$S2_CKPT" ] || [ ! -f "$S2_CKPT" ]; then
    echo "Stage 2 checkpoint not found for $CELL seed=$SEED — skipping."
else
    echo '=== Stage 2 (finetuned) ==='
    python scripts/test_episomal_mpra.py \
        --model_type enformer_finetuned \
        --checkpoint_path "$S2_CKPT" \
        --cell_type "$CELL" \
        --data_path "$DATA" \
        --output_dir "$OUT" \
        --run_name "enformer-prod2-$CELL-seed$SEED-stage2"
fi

echo "=== Done $CELL seed=$SEED — $(date) ==="
