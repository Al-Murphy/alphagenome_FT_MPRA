#!/bin/bash
# Script to run FIMO on attribution-derived sequences
#
# FIMO (Find Individual Motif Occurrences) scans a motif database against sequences
# to find statistically significant motif matches.
#
# Prerequisites:
# 1. Install MEME suite: https://meme-suite.org/meme/doc/download.html
#    Or use conda: conda install -c bioconda meme
#
# 2. Download a motif database (MEME format):
#    - JASPAR: https://jaspar.genereg.net/downloads/
#    - HOCOMOCO: http://hocomoco11.autosome.ru/
#    - CIS-BP: http://cisbp.ccbr.utoronto.ca/
#    - Or use the JASPAR 2024 CORE non-redundant vertebrates:
#      wget https://jaspar.genereg.net/api/v1/database/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.zip
#
# Usage:
#   bash scripts/run_fimo.sh \
#       --sequences <sequences.fa> \
#       --motif_db <motif_database.meme> \
#       --output_dir <output_directory> \
#       [--pvalue_threshold 1e-4]

set -e

# Default values
P_VALUE_THRESHOLD="1e-4"
OUTPUT_DIR=""
SEQUENCES=""
MOTIF_DB=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sequences)
            SEQUENCES="$2"
            shift 2
            ;;
        --motif_db)
            MOTIF_DB="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pvalue_threshold)
            P_VALUE_THRESHOLD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --sequences <sequences.fa> --motif_db <motif_database.meme> --output_dir <output_dir> [--pvalue_threshold 1e-4]"
            echo ""
            echo "Arguments:"
            echo "  --sequences          FASTA file with sequences to scan"
            echo "  --motif_db          MEME format motif database"
            echo "  --output_dir        Output directory for FIMO results"
            echo "  --pvalue_threshold   P-value threshold (default: 1e-4)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ -z "$SEQUENCES" ]]; then
    echo "Error: --sequences is required"
    exit 1
fi

if [[ -z "$MOTIF_DB" ]]; then
    echo "Error: --motif_db is required"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output_dir is required"
    exit 1
fi

# Check if files exist
if [[ ! -f "$SEQUENCES" ]]; then
    echo "Error: Sequences file not found: $SEQUENCES"
    exit 1
fi

if [[ ! -f "$MOTIF_DB" ]]; then
    echo "Error: Motif database not found: $MOTIF_DB"
    exit 1
fi

# Check if FIMO is installed
if ! command -v fimo &> /dev/null; then
    echo "Error: FIMO not found. Please install MEME suite:"
    echo "  conda install -c bioconda meme"
    echo "  or visit: https://meme-suite.org/meme/doc/download.html"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Running FIMO"
echo "=========================================="
echo "Sequences: $SEQUENCES"
echo "Motif database: $MOTIF_DB"
echo "Output directory: $OUTPUT_DIR"
echo "P-value threshold: $P_VALUE_THRESHOLD"
echo ""

# Run FIMO
fimo \
    --oc "$OUTPUT_DIR" \
    --thresh "$P_VALUE_THRESHOLD" \
    "$MOTIF_DB" \
    "$SEQUENCES"

echo ""
echo "=========================================="
echo "FIMO complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Main output file: $OUTPUT_DIR/fimo.txt"
echo "  Contains all motif matches with p-values"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/fimo.txt | head -20"
echo ""
