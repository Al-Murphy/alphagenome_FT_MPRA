#!/bin/bash
#SBATCH --job-name=cagi5_fixedcon
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

# CAGI5 zero-shot evaluation of the corrected-construct retrains.
#
# CAGI5 is affected by the construct bug only THROUGH THE WEIGHTS: these scripts
# score a 281 bp *genomic* window around each variant and never build the
# reporter construct. So this is pure inference against the new checkpoints.
#
# Only HepG2 and K562 are scored - CAGI5 has no WTC11 elements
# (HepG2: F9, LDLR, SORT1 / K562: GP1BB, HBB, HBG1, PKLR).
#
# Eight runs: {HepG2,K562} x {stage1 probing, stage2 fine-tuned} x {plain, aug}.
#
# OUTPUT DIR IS MANDATORY HERE. test_cagi5_zero_shot_mpra.py names its files
# after the checkpoint DIRECTORY, which is "stage1"/"stage2" for every model, so
# writing to the default directory would overwrite the published CSVs - the same
# collision class that corrupted the benchmark tables in Sep 2026.
#
#   sbatch scripts/slurm/cagi5_fixedconstruct_mpra.sh

REPO=/grid/koo/home/amurphy/projects/alphagenome_FT_MPRA
cd "$REPO"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

BASE=$(ls -d /grid/koo/home/amurphy/.cache/huggingface/hub/models--google--alphagenome-all-folds/snapshots/*/ | head -1)
OUTDIR="$REPO/results/cagi5_evaluations/fixedconstruct"
mkdir -p logs "$OUTDIR"

for CELL in HepG2 K562; do
  CK="$REPO/results/models/checkpoints/${CELL}/mpra-${CELL}-optimal-fixedconstruct"
  for STAGE in stage1 stage2; do
    [ -d "$CK/$STAGE" ] || { echo "SKIP ${CELL}/${STAGE} (not present)"; continue; }
    for AUG in plain aug; do
      echo
      echo "=== ${CELL} / ${STAGE} / ${AUG} ==="
      EXTRA=()
      # The published augmented runs were unseeded; seed these so they can be
      # reproduced exactly. Shift sampling is a small source of run-to-run noise.
      [ "$AUG" = "aug" ] && EXTRA=(--use_position_shift --max_shift 20 \
                                   --n_shift_samples 3 --use_reverse_complement \
                                   --random_seed 0)
      .venv/bin/python scripts/test_cagi5_zero_shot_mpra.py \
        --checkpoint_dir "$CK/$STAGE" \
        --cell_type "$CELL" \
        --base_checkpoint_path "$BASE" \
        --output_dir "$OUTDIR" \
        "${EXTRA[@]}"
    done
  done
done

echo
echo "=== summary ==="
for F in "$OUTDIR"/*summary.csv; do
  [ -f "$F" ] || continue
  printf "%-62s %s\n" "$(basename "$F" _summary.csv)" \
    "$(.venv/bin/python -c "import pandas as pd,sys;r=pd.read_csv(sys.argv[1]).iloc[0];print(f\"pearson={r['pearson_all']:.4f} spearman={r['spearman_all']:.4f}\")" "$F")"
done
