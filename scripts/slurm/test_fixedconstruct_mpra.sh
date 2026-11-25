#!/bin/bash
#SBATCH --job-name=test_fixedcon
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u

# Full-dataset test evaluation of a corrected-construct retrain.
#
# The per-epoch test_pearson printed during training is BATCH-AVERAGED and
# therefore attenuated (see docs/randinit_control.md and the selection-metric
# note in the random-init write-up), so it must not be compared against the
# published full-dataset numbers. This runs the proper full-split evaluation.
#
# No --construct_suffix is passed on purpose: these checkpoints RECORD their
# construct, so this also verifies that test_ft_model_mpra.py reads it back from
# the checkpoint rather than falling back to the legacy tail.
#
#   sbatch --export=ALL,CELL_TYPE=HepG2 scripts/slurm/test_fixedconstruct_mpra.sh

REPO=/grid/koo/home/amurphy/projects/alphagenome_FT_MPRA
cd "$REPO"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

CELL_TYPE="${CELL_TYPE:?set CELL_TYPE=HepG2|K562|WTC11}"
RUN="mpra-${CELL_TYPE}-optimal-fixedconstruct"
CKROOT="$REPO/results/models/checkpoints/${CELL_TYPE}/${RUN}"
BASE=$(ls -d /grid/koo/home/amurphy/.cache/huggingface/hub/models--google--alphagenome-all-folds/snapshots/*/ | head -1)
# Deliberately NOT results/test_predictions/ — that directory already holds stale
# root files from the random-init collision; keep these separate.
OUTDIR="$REPO/results/test_predictions/fixedconstruct/${CELL_TYPE}"

mkdir -p logs "$OUTDIR"
echo "cell type : $CELL_TYPE"
echo "run       : $RUN"

for STAGE in stage1 stage2; do
  [ -d "$CKROOT/$STAGE" ] || { echo "SKIP $STAGE (not present)"; continue; }
  echo
  echo "=== ${CELL_TYPE} / ${STAGE} ==="
  .venv/bin/python scripts/test_ft_model_mpra.py \
    --checkpoint_dir "$CKROOT/$STAGE" \
    --cell_type "$CELL_TYPE" \
    --base_checkpoint_path "$BASE" \
    --output_dir "$OUTDIR"
done

echo
echo "=== summary ==="
for STAGE in stage1 stage2; do
  F="$OUTDIR/${STAGE}_${CELL_TYPE}_test_metrics.csv"
  [ -f "$F" ] && printf "%-7s %s\n" "$STAGE" "$(tail -1 "$F")"
done
