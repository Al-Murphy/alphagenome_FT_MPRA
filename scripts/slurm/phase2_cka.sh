#!/bin/bash
#SBATCH --job-name=phase2_cka
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u

REPO=/grid/koo/home/amurphy/projects/alphagenome_FT_MPRA
cd "$REPO"
mkdir -p logs results/cka
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# MODE defaults to introspect; override:  sbatch --export=ALL,MODE=cka scripts/slurm/phase2_cka.sh
MODE="${MODE:-introspect}"
echo "=== Phase 2 CKA: mode=$MODE ==="
.venv/bin/python scripts/analysis/phase2_encoder_cka.py --mode "$MODE" --n_seqs "${N_SEQS:-1536}" --batch_size "${BATCH_SIZE:-64}"
