#!/bin/bash
#SBATCH --job-name=test_epi
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=00:30:00
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

# Array index → (cell, seed)
declare -A CELLS=([0]=K562 [1]=K562 [2]=HepG2 [3]=HepG2)
declare -A SEEDS=([0]=1042 [1]=2042 [2]=42 [3]=1042)
CELL=${CELLS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DATA=/grid/wsbs/home_norepl/christen/ALBench-S2F/data/$(echo "$CELL" | tr '[:upper:]' '[:lower:]')

S1_DIR="$REPO/results/models/checkpoints/episomal_enformer/$CELL/enformer-episomal-$CELL-seed$SEED"
S2_DIR="$S1_DIR/stage2"
S1_CKPT=$(ls "$S1_DIR"/best-epoch=*.ckpt 2>/dev/null | head -1)
S2_CKPT=$(ls "$S2_DIR"/best-stage2-epoch=*.ckpt 2>/dev/null | head -1)
OUT="$REPO/results/episomal_predictions"
mkdir -p "$OUT"

echo '=== Stage 1 ==='
python scripts/test_episomal_mpra.py --model_type enformer_probing --checkpoint_path "$S1_CKPT" --cell_type $CELL --data_path $DATA --output_dir "$OUT" --run_name "enformer-$CELL-seed$SEED-stage1"
echo '=== Stage 2 ==='
python scripts/test_episomal_mpra.py --model_type enformer_finetuned --checkpoint_path "$S2_CKPT" --cell_type $CELL --data_path $DATA --output_dir "$OUT" --run_name "enformer-$CELL-seed$SEED-stage2"
