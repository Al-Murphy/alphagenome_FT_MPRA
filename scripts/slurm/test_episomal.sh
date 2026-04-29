#!/bin/bash
#SBATCH --job-name=test_epi
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5

REPO=/grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA
cd "$REPO"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source /grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/bin/activate

DATA_PATH=/grid/wsbs/home_norepl/christen/ALBench-S2F/data/k562
S1_CKPT="$REPO/results/models/checkpoints/episomal_enformer/K562/enformer-episomal-K562-seed42/best-epoch=06-val_loss=0.5602.ckpt"
S2_CKPT="$REPO/results/models/checkpoints/episomal_enformer/K562/enformer-episomal-K562-seed42/stage2/best-stage2-epoch=07-val_loss=0.5274.ckpt"
OUTDIR="$REPO/results/episomal_predictions"
mkdir -p "$OUTDIR"

echo '=== Stage 1 (Probing) eval ==='
python scripts/test_episomal_mpra.py     --model_type enformer_probing     --checkpoint_path "$S1_CKPT"     --cell_type K562     --data_path "$DATA_PATH"     --output_dir "$OUTDIR"     --run_name enformer-K562-seed42-stage1

echo
echo '=== Stage 2 (Fine-tuned) eval ==='
python scripts/test_episomal_mpra.py     --model_type enformer_finetuned     --checkpoint_path "$S2_CKPT"     --cell_type K562     --data_path "$DATA_PATH"     --output_dir "$OUTDIR"     --run_name enformer-K562-seed42-stage2
