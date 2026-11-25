#!/bin/bash
#SBATCH --job-name=ft_AG_HepG2_fixedcon
#SBATCH --output=out/ft_AG_HepG2_fixedcon_%j.out
#SBATCH --error=out/ft_AG_HepG2_fixedcon_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=koolab
#SBATCH --partition=kooq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amurphy@cshl.edu
#SBATCH --export=ALL

# LentiMPRA HepG2 retrained on the CORRECTED reporter construct.
#
# The previous runs appended the last 36 nt of the Agarwal et al. 2025 Table S13
# reverse primer minP-enh-r without reverse-complementing it, which repeated the
# right adapter and ran the minimal promoter backwards. This run uses the
# reporter-strand sequence. Hyperparameters are identical to
# configs/mpra_HepG2.json; only the construct and the run name differ, so the
# comparison against the released model is controlled.
#
# Output: results/models/checkpoints/HepG2/mpra-HepG2-optimal
# The released mpra-HepG2-optimal checkpoints are left untouched.
#
# On kooq/koolab (30-day MaxWall) rather than bio_ai: the original runs took
# 30-45 h, uncomfortably close to the 48 h cap on bio_ai.

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Running CORRECTED-construct configuration for HepG2"

cd /grid/koo/home/amurphy/projects/alphagenome_FT_MPRA

source .venv/bin/activate

# alphagenome_research fetches variant-scoring calibration from gs://alphagenome/...
# at model-create time; libcurl looks for CA certs at the Debian path on RHEL8.
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

srun python scripts/finetune_mpra.py --config configs/mpra_HepG2.json
