#!/bin/bash
#SBATCH --job-name=construct_audit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u

# Suffix-sensitivity audit for the lentiMPRA reporter construct.
#
# The 281 bp model input is the 230 bp assayed element (left adapter + 200 bp CRE +
# right adapter, already in the `seq` column) plus a 51 bp CONSTANT tail. That tail
# was built from the Agarwal et al. 2025 Table S13 reverse primer `minP-enh-r`
# copied without reverse-complementing it, so it encodes the right adapter a second
# time (reverse-complemented) followed by 21 nt of the minimal promoter backwards,
# then a fixed placeholder barcode. Read on the reporter strand the tail should be
# the minimal promoter in sense orientation followed by the spacer.
#
# Because the tail is CONSTANT across every training example it carries no
# information that can discriminate between elements, so a model may simply ignore
# it. This job measures whether that is so, WITHOUT retraining: it scores one
# already-released checkpoint under three different tails.
#
#   legacy  - what the released models were trained with (reproduces the paper number)
#   fixed   - the corrected reporter-strand sequence
#   scram   - a seeded shuffle of `legacy`, same base composition, biologically meaningless
#
# Reading:
#   legacy ~= fixed ~= scram  -> the encoder ignores the constant tail; the bug does
#                               not move results, only the construct description.
#   legacy >> fixed, scram    -> the encoder is keyed to the exact tail it saw in
#                               training; affected models must be retrained.

REPO=/grid/koo/home/amurphy/projects/alphagenome_FT_MPRA
cd "$REPO"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# alphagenome_research fetches variant-scoring calibration from gs://alphagenome/...
# at model-create time; libcurl looks for CA certs at the Debian path on these RHEL8
# nodes. Same workaround as test_plant_starrseq_array.sh.
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

CELL_TYPE="${CELL_TYPE:-K562}"
WEIGHTS=/grid/koo/home/shared/models/alphagenome_encoder/jax
CKPT="$WEIGHTS/mpra-${CELL_TYPE}-optimal/stage2"
BASE=$(ls -d /grid/koo/home/amurphy/.cache/huggingface/hub/models--google--alphagenome-all-folds/snapshots/*/ | head -1)
OUTDIR="$REPO/results/construct_audit"

LEGACY="TCCATTATATACCCTCTAGTGTCGGTTCACGCAATGAGAGACTGAGGCCAC"
FIXED="CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTAC"
SCRAM="TGTCTGTTCTCTCAGAAGGTGGAACGAGCTGAAACCCCCGTTAATCACCTA"

mkdir -p logs "$OUTDIR"
echo "cell type : $CELL_TYPE"
echo "checkpoint: $CKPT"
echo "base      : $BASE"

# ARMS defaults to the three-way comparison; override e.g.
#   sbatch --export=ALL,ARMS=fallback scripts/slurm/construct_audit_mpra.sh
# The `fallback` arm passes NO --construct_suffix, so it exercises the
# resolution path used in normal evaluation: read the construct pinned in the
# checkpoint, and for a checkpoint that predates pinning fall back to the legacy
# tail. On a pre-correction checkpoint it must reproduce the `legacy` number.
ARMS="${ARMS:-legacy fixed scram}"

for ARM in $ARMS; do
  echo
  echo "=== arm: $ARM ==="
  case "$ARM" in
    legacy) SUFFIX="$LEGACY" ;;
    fixed)  SUFFIX="$FIXED" ;;
    scram)  SUFFIX="$SCRAM" ;;
    fallback)
      .venv/bin/python scripts/test_ft_model_mpra.py \
        --checkpoint_dir "$CKPT" \
        --cell_type "$CELL_TYPE" \
        --base_checkpoint_path "$BASE" \
        --output_dir "$OUTDIR"
      continue
      ;;
    *) echo "unknown arm: $ARM" >&2; exit 1 ;;
  esac
  .venv/bin/python scripts/test_ft_model_mpra.py \
    --checkpoint_dir "$CKPT" \
    --cell_type "$CELL_TYPE" \
    --base_checkpoint_path "$BASE" \
    --construct_suffix "$SUFFIX" \
    --suffix_tag "$ARM" \
    --output_dir "$OUTDIR"
done

echo
echo "=== summary ==="
for ARM in legacy fixed scram; do
  F="$OUTDIR/stage2_${CELL_TYPE}_test_${ARM}_metrics.csv"
  [ -f "$F" ] && echo "$ARM: $(cat "$F")"
done
