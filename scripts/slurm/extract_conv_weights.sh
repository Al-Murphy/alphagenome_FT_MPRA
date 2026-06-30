#!/bin/bash
#SBATCH --job-name=conv_weights
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u
REPO=/grid/koo/home/amurphy/projects/alphagenome_FT_MPRA
cd "$REPO"; mkdir -p logs results/filter_qdist
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/analysis/extract_conv_weights.py
