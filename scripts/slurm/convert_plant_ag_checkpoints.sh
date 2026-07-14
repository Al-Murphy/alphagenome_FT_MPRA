#!/bin/bash
# Convert the 6 plant STARR-seq AlphaGenome checkpoints from autotune's flat pickle
# layout into the shared orbax layout used by the other jax/ entries.
#
# ONE CELL PER TASK — this must be an array, not a loop. Writing several orbax stores
# from a single process leaves every store after the first unreadable (TensorStore
# throws "error while reading array index" on restore), so each conversion gets its own
# process.
#
# Submit:
#   sbatch --array=0-5 scripts/slurm/convert_plant_ag_checkpoints.sh
#
#SBATCH --job-name=ag_plant_convert
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G

set -euo pipefail

# alphagenome_research reads variant-scoring calibration from gs://; TF's libcurl looks
# for CA certs at the Debian path, which does not exist on these RHEL8 nodes.
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

REPO=/grid/koo/home/amurphy/projects/alphagenome_FT_MPRA
SHARED=/grid/koo/home/shared/models/alphagenome_encoder/jax
BASE_CKPT=/grid/koo/home/amurphy/.cache/huggingface/hub/models--google--alphagenome-all-folds/snapshots

cd "$REPO"
BASE=$(find "$BASE_CKPT" -mindepth 1 -maxdepth 1 -type d | head -1)
echo "base checkpoint: $BASE"

CELLS=(
  "leaf combined" "leaf enhancer" "leaf promoter_only"
  "proto combined" "proto enhancer" "proto promoter_only"
)
read -r TISSUE MODE <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID: $TISSUE $MODE ==="

.venv/bin/python scripts/convert_plant_ag_checkpoints_to_orbax.py \
  --src "$SHARED" \
  --dest "$SHARED" \
  --base_checkpoint_path "$BASE" \
  --tissue "$TISSUE" \
  --mode "$MODE"
