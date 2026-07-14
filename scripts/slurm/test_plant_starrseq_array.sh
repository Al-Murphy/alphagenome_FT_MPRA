#!/bin/bash
# Re-score every saved AlphaGenome plant STARR-seq checkpoint on the held-out test
# split — 12 cells (2 tissues x 3 modes x {finetune, probe}).
#
# Independent check on results/plant_starrseq/reference/alphagenome_*.json, which
# were produced by the upstream autotune codebase rather than by this repo.
#
# Submit:
#   sbatch --array=0-11 scripts/slurm/test_plant_starrseq_array.sh
#
#SBATCH --job-name=ag_plant_test
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

set -euo pipefail

# alphagenome_research pulls variant-scoring calibration from gs://alphagenome/... at
# model-create time. TensorFlow's libcurl looks for CA certs at the Debian path, which
# does not exist on these RHEL8 nodes -> "SSL CA cert (path? access rights?)".
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

REPO=/grid/koo/home/amurphy/projects/alphagenome_FT_MPRA
WEIGHTS=/grid/koo/home/shared/models/alphagenome_encoder/jax
# base AlphaGenome all_folds weights (already in the HF cache — avoids a download per task)
BASE_CKPT=/grid/koo/home/amurphy/.cache/huggingface/hub/models--google--alphagenome-all-folds/snapshots

cd "$REPO"

# 12 cells: tissue mode method
CELLS=(
  "leaf combined finetune"   "leaf combined probe"
  "leaf enhancer finetune"   "leaf enhancer probe"
  "leaf promoter_only finetune" "leaf promoter_only probe"
  "proto combined finetune"  "proto combined probe"
  "proto enhancer finetune"  "proto enhancer probe"
  "proto promoter_only finetune" "proto promoter_only probe"
)

read -r TISSUE MODE METHOD <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"

# orbax layout: <run>/stage1 (ridge probe) and <run>/stage2 (fine-tuned)
PROBE_FLAG=""
STAGE=stage2
if [[ "$METHOD" == "probe" ]]; then
  PROBE_FLAG="--probe"
  STAGE=stage1
fi

# resolve the single snapshot dir under the HF cache
BASE=$(find "$BASE_CKPT" -mindepth 1 -maxdepth 1 -type d | head -1)

echo "=== task $SLURM_ARRAY_TASK_ID: $TISSUE $MODE $METHOD ==="
echo "base checkpoint: $BASE"

.venv/bin/python scripts/test_ft_model_plant_starrseq.py \
  --tissue "$TISSUE" \
  --mode "$MODE" \
  $PROBE_FLAG \
  --checkpoint_dir "$WEIGHTS/plant-starrseq-${TISSUE}-${MODE}/${STAGE}" \
  --base_checkpoint_path "$BASE" \
  --output_dir ./results/plant_starrseq/retest \
  --save_predictions
