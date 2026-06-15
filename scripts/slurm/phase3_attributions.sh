#!/bin/bash
#SBATCH --job-name=phase3_attr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u
REPO=/grid/koo/home/amurphy/projects/alphagenome_FT_MPRA
cd "$REPO"
mkdir -p logs results/modisco
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/analysis/phase3_attributions.py \
  --n_seqs "${N_SEQS:-3000}" --batch_size "${BATCH_SIZE:-128}" --method "${METHOD:-gradient_corrected}"
