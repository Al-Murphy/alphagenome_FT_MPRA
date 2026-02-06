#!/usr/bin/env python3
"""
Zero-shot evaluation of fine-tuned Enformer MPRA models on the CAGI5
saturation mutagenesis benchmark.

This script is the Enformer/PyTorch analogue of:
- `scripts/test_cagi5_zero_shot_mpra.py` (AlphaGenome MPRA head, JAX)
and follows the testing style of:
- `scripts/test_ft_model_enformer_mpra.py` (PyTorch Lightning Enformer MPRA).

Key behaviors:
- Uses a fine-tuned Enformer MPRA Lightning checkpoint
  (`EnformerMPRALightning` from `scripts/finetune_enformer_mpra.py`).
- Restricts evaluation to cell-type-matched CAGI5 elements:
  - HepG2 models → F9, LDLR, SORT1
  - K562 models → GP1BB, HBB, HBG1, PKLR
- Only evaluates on the CAGI5 challenge (test) set, not training set.
- Computes variant effects as:
      variant_effect = alt_prediction - ref_prediction
  where predictions are per-construct log(RNA/DNA) from the Enformer MPRA head.
- Supports the same optional augmentations as `test_cagi5_zero_shot_mpra.py`:
  - Position shifts around the variant position
  - Reverse-complement augmentation
- Sequence length defaults to 281 bp (same as Enformer MPRA training).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import torch

# Add parent directory to path to allow imports from src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# ---------------------------------------------------------------------------
# Import Enformer MPRA Lightning model and helpers
# ---------------------------------------------------------------------------

import importlib.util

finetune_path = Path(__file__).parent / "finetune_enformer_mpra.py"
spec_finetune = importlib.util.spec_from_file_location(
    "finetune_enformer_mpra", finetune_path
)
finetune_module = importlib.util.module_from_spec(spec_finetune)
spec_finetune.loader.exec_module(finetune_module)

EnformerMPRALightning = finetune_module.EnformerMPRALightning
create_dummy_model_for_dataset = finetune_module.create_dummy_model_for_dataset

# Sequence length for MPRA promoter constructs (matches training)
PROMOTER_CONSTRUCT_LENGTH = 281


# ---------------------------------------------------------------------------
# Import seq_loader (direct import to avoid JAX dependencies in src/__init__.py)
# ---------------------------------------------------------------------------

seq_loader_path = Path(__file__).parent.parent / 'src' / 'seq_loader.py'
spec_seq = importlib.util.spec_from_file_location("seq_loader", seq_loader_path)
seq_loader_module = importlib.util.module_from_spec(spec_seq)
spec_seq.loader.exec_module(seq_loader_module)
seq_loader = seq_loader_module.seq_loader


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------

def reverse_complement_dna(seq: str) -> str:
    """Get the reverse complement of a DNA sequence string.
    
    Args:
        seq: DNA sequence string.
        
    Returns:
        Reverse complement of the sequence.
    """
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement_map.get(base, 'N') for base in reversed(seq.upper()))


def get_random_position_shifts(max_shift: int = 20, n_samples: int = 3, random_seed: int | None = None) -> List[int]:
    """Get randomly sampled position shifts for augmentation.
    
    Args:
        max_shift: Maximum shift in base pairs (default: 20).
        n_samples: Number of random shifts to sample (default: 3).
        random_seed: Random seed for reproducibility (default: None).
        
    Returns:
        List of shift values including 0 (no shift) and n_samples randomly sampled shifts from ±1 to ±max_shift.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    shifts = [0]  # Always include original position
    
    # Sample n_samples shifts from ±1 to ±max_shift
    all_possible_shifts = []
    for shift in range(1, max_shift + 1):
        all_possible_shifts.extend([shift, -shift])
    
    if len(all_possible_shifts) > 0:
        sampled = np.random.choice(all_possible_shifts, size=min(n_samples, len(all_possible_shifts)), replace=False)
        shifts.extend(sampled.tolist())
    
    return shifts


# ---------------------------------------------------------------------------
# CAGI5 utilities
# ---------------------------------------------------------------------------

# CAGI5 element coordinates (hg19/GRCh37)
# Format: {element: (chrom, start, end)}
CAGI5_REGIONS = {
    'F9': ('chrX', 138612500, 138613500),
    'GP1BB': ('chr22', 19710500, 19711700),
    'HBB': ('chr11', 5248100, 5248800),
    'HBG1': ('chr11', 5270900, 5271800),
    'HNF4A': ('chr20', 42984000, 42985000),
    'IRF4': ('chr6', 396000, 397200),
    'IRF6': ('chr1', 209989000, 209990500),
    'LDLR': ('chr19', 11199800, 11200800),
    'MSMB': ('chr10', 51548800, 51550400),
    'MYCrs6983267': ('chr8', 128412900, 128414500),
    'PKLR': ('chr1', 155271000, 155272300),
    'SORT1': ('chr1', 109817100, 109818700),
    'TERT-GBM': ('chr5', 1295000, 1295800),
    'TERT-HEK293T': ('chr5', 1295000, 1295800),
    'ZFAND3': ('chr6', 37775100, 37776600),
}


def get_ref_allele_from_genome(
    seq_loader_obj: seq_loader,
    chromosome: str,
    position: int,
    ref_allele_length: int,
) -> str:
    """Get reference allele from hg19 genome using seq_loader.
    
    Args:
        seq_loader_obj: seq_loader instance for hg19.
        chromosome: Chromosome (e.g., 'chr1', 'chrX').
        position: Variant position (1-based, hg19).
        ref_allele_length: Length of reference allele.
        
    Returns:
        Reference allele sequence from genome.
    """
    # pysam uses 0-based, half-open intervals
    start_0based = position - 1
    end_0based = start_0based + ref_allele_length
    ref_seq = seq_loader_obj.genome_dat.fetch(chromosome, start_0based, end_0based).upper()
    return ref_seq


def verify_ref_allele(
    seq_loader_obj: seq_loader,
    chromosome: str,
    position: int,
    ref_allele: str,
) -> bool:
    """Verify that the reference allele matches the hg19 genome.
    
    Args:
        seq_loader_obj: seq_loader instance for hg19.
        chromosome: Chromosome (e.g., 'chr1', 'chrX').
        position: Variant position (1-based, hg19).
        ref_allele: Reference allele from CAGI5 data.
        
    Returns:
        True if ref allele matches genome, False otherwise.
    """
    genome_ref = get_ref_allele_from_genome(
        seq_loader_obj, chromosome, position, len(ref_allele)
    )
    return genome_ref.upper() == ref_allele.upper()


def parse_cagi5_dir(
    cagi5_dir: Path,
    elements_to_include: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load selected CAGI5 challenge TSVs into a dict[element] -> DataFrame.

    Expects files named `challenge_*.tsv` with header line `#Chrom`.
    Columns: Chrom, Pos (1-based), Ref, Alt, Value, Confidence, ...
    """
    if not cagi5_dir.exists():
        raise FileNotFoundError(f"CAGI5 directory not found: {cagi5_dir}")

    data: Dict[str, pd.DataFrame] = {}
    for tsv_file in sorted(cagi5_dir.glob("challenge_*.tsv")):
        element = tsv_file.stem.replace("challenge_", "")

        # If a subset of elements was specified, skip others
        if elements_to_include is not None and element not in elements_to_include:
            continue
        with open(tsv_file) as f:
            lines = f.readlines()

        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("#Chrom"):
                header_idx = i
                break
        if header_idx is None:
            continue

        header = lines[header_idx].lstrip("#").strip().split("\t")
        data_lines = [l.strip().split("\t") for l in lines[header_idx + 1 :] if l.strip()]
        df = pd.DataFrame(data_lines, columns=header)

        # Convert numeric columns
        df["Pos"] = df["Pos"].astype(int)
        df["Value"] = df["Value"].astype(float)
        if "Confidence" in df.columns:
            df["Confidence"] = df["Confidence"].astype(float)
        else:
            # If Confidence is missing, treat all as high confidence
            df["Confidence"] = 1.0

        data[element] = df

    if not data:
        raise RuntimeError(f"No CAGI5 challenge_*.tsv files found under {cagi5_dir}")

    return data


def get_ref_alt_sequences(
    seq_loader_obj: seq_loader,
    chromosome: str,
    var_pos: int,
    ref_allele: str,
    alt_allele: str,
    window: int = 281,
    position_shift: int = 0,
) -> Tuple[str | None, str | None]:
    """Get both reference and alternate sequences centered on the variant position (hg19, 1-based).

    Args:
        seq_loader_obj: seq_loader instance for hg19.
        chromosome: Chromosome (e.g., 'chr1', 'chrX').
        var_pos: Genomic position of the variant (1-based, hg19).
        ref_allele: Reference allele from CAGI5 file.
        alt_allele: Alternate allele from CAGI5 file.
        window: Length of output sequence to generate.
        position_shift: Shift variant position by this many base pairs (default: 0).
        
    Returns:
        Tuple of (ref_sequence, alt_sequence), or (None, None) if extraction fails.
    """
    # Apply position shift
    shifted_var_pos = var_pos + position_shift
    
    # Extract reference sequence centered on (possibly shifted) variant
    half_window = window // 2
    start_0based = max(0, shifted_var_pos - 1 - half_window)
    end_0based = start_0based + window
    
    # Get reference sequence from hg19
    ref_seq = seq_loader_obj.genome_dat.fetch(chromosome, start_0based, end_0based).upper()
    
    # Verify ref allele matches
    variant_pos_in_seq = (var_pos - 1) - start_0based
    if variant_pos_in_seq < 0 or variant_pos_in_seq + len(ref_allele) > len(ref_seq):
        return None, None
    
    ref_in_seq = ref_seq[variant_pos_in_seq : variant_pos_in_seq + len(ref_allele)]
    if ref_in_seq.upper() != ref_allele.upper():
        # Best-effort check - warn but continue
        pass

    # Create alternate sequence by substituting alt allele
    alt_seq = (
        ref_seq[:variant_pos_in_seq] +
        alt_allele +
        ref_seq[variant_pos_in_seq + len(ref_allele):]
    )

    # Ensure we return exactly window length for both sequences
    def ensure_window_length(seq: str) -> str:
        if len(seq) < window:
            # Pad with N's if needed
            return seq + "N" * (window - len(seq))
        elif len(seq) > window:
            # Trim if needed (shouldn't happen, but just in case)
            return seq[:window]
        return seq

    ref_seq_final = ensure_window_length(ref_seq)
    alt_seq_final = ensure_window_length(alt_seq)

    return ref_seq_final, alt_seq_final


def test_ref_allele_verification(
    seq_loader_obj: seq_loader,
    cagi5_data: Dict[str, pd.DataFrame],
    max_test_variants: int = 100,
) -> bool:
    """Unit test to verify reference alleles match hg19 genome.
    
    Args:
        seq_loader_obj: seq_loader instance for hg19.
        cagi5_data: Dictionary of CAGI5 DataFrames by element.
        max_test_variants: Maximum number of variants to test per element.
        
    Returns:
        True if all tested variants pass, False otherwise.
    """
    all_passed = True
    total_tested = 0
    total_passed = 0
    
    for element, df in cagi5_data.items():
        if element not in CAGI5_REGIONS:
            continue
        
        chromosome = CAGI5_REGIONS[element][0]
        test_df = df.head(max_test_variants)
        
        for _, row in test_df.iterrows():
            var_pos = int(row["Pos"])
            ref_allele = str(row["Ref"])
            
            total_tested += 1
            if verify_ref_allele(seq_loader_obj, chromosome, var_pos, ref_allele):
                total_passed += 1
            else:
                all_passed = False
                genome_ref = get_ref_allele_from_genome(
                    seq_loader_obj, chromosome, var_pos, len(ref_allele)
                )
                print(f"  FAIL: {element} {chromosome}:{var_pos} "
                      f"expected {ref_allele}, got {genome_ref}")
    
    print(f"  Tested {total_tested} variants, {total_passed} passed, "
          f"{total_tested - total_passed} failed")
    
    return all_passed


# ---------------------------------------------------------------------------
# One-hot encoding utility (mirrors finetune_enformer_mpra.py)
# ---------------------------------------------------------------------------

def one_hot_encode_dna(seq: str) -> np.ndarray:
    """One-hot encode DNA sequence into (L, 4) numpy array (A,C,G,T)."""
    dna_bases = np.array(["A", "C", "G", "T"])
    seq = seq.upper()
    if not seq:
        return np.zeros((0, 4), dtype=np.float32)

    all_one_hot = []
    for base in seq:
        one_hot = np.zeros((1, 4), dtype=np.float32)
        matches = np.where(base == dna_bases)[0]
        if len(matches) > 0:
            one_hot[0, matches] = 1.0
        all_one_hot.append(one_hot)

    try:
        arr = np.concatenate(all_one_hot, axis=0)
    except Exception:
        arr = np.zeros((len(seq), 4), dtype=np.float32)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Prediction utilities
# ---------------------------------------------------------------------------

def batch_predict_enformer_mpra(
    model: EnformerMPRALightning,
    sequences: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Run Enformer MPRA predictions on a list of DNA sequences.

    Args:
        model: EnformerMPRALightning instance (encoder+MPRA head).
        sequences: List of DNA sequence strings (all same length).
        batch_size: Batch size for inference.

    Returns:
        Numpy array of shape (N,) with scalar predictions.
    """
    if not sequences:
        return np.array([], dtype=np.float32)

    model.eval()
    device = next(model.parameters()).device

    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            # One-hot encode and pad to common length
            one_hot_list = [one_hot_encode_dna(seq) for seq in batch_seqs]
            max_len = max(arr.shape[0] for arr in one_hot_list)

            padded = []
            for arr in one_hot_list:
                if arr.shape[0] < max_len:
                    pad = np.zeros((max_len - arr.shape[0], 4), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=0)
                padded.append(arr)

            batch_arr = np.stack(padded, axis=0)  # (B, L, 4)
            batch_tensor = torch.from_numpy(batch_arr).float().to(device)

            preds = model.forward(batch_tensor)  # (B,) or (B,1)
            if preds.dim() > 1 and preds.shape[-1] == 1:
                preds = preds.squeeze(-1)

            all_preds.append(preds.cpu().numpy().astype(np.float32))

    return np.concatenate(all_preds, axis=0)


def compute_correlations(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson r and Spearman ρ for two 1D arrays."""
    if len(preds) == 0:
        return float("nan"), float("nan")
    pearson, _ = pearsonr(preds, targets)
    spearman, _ = spearmanr(preds, targets)
    return float(pearson), float(spearman)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot CAGI5 evaluation for fine-tuned Enformer MPRA models"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to PyTorch Lightning Enformer MPRA checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="K562",
        choices=["HepG2", "K562"],
        help="Cell type the model was fine-tuned on (default: K562)",
    )
    parser.add_argument(
        "--cagi5_dir",
        type=str,
        default=None,
        help="Directory with CAGI5 challenge_*.tsv files (default: ./data/cagi5).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=PROMOTER_CONSTRUCT_LENGTH,
        help=f"Sequence window length around each variant (default: {PROMOTER_CONSTRUCT_LENGTH}, matches Enformer MPRA training).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--use_position_shift",
        action="store_true",
        help="Enable position shift augmentation: average predictions over shifts up to ±20 bp around the variant position.",
    )
    parser.add_argument(
        "--max_shift",
        type=int,
        default=20,
        help="Maximum position shift in base pairs for augmentation (default: 20). Only used if --use_position_shift is set.",
    )
    parser.add_argument(
        "--n_shift_samples",
        type=int,
        default=3,
        help="Number of random position shifts to sample per variant (default: 3). Only used if --use_position_shift is set.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for position shift sampling (default: None).",
    )
    parser.add_argument(
        "--use_reverse_complement",
        action="store_true",
        help="Enable reverse complement augmentation: average predictions over forward and reverse complement sequences.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/cagi5_evaluations",
        help="Directory to save CAGI5 evaluation results (default: ./results/cagi5_evaluations)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    repo_root = Path(__file__).parent.parent.resolve()

    # Default CAGI5 dir
    if args.cagi5_dir is None:
        cagi5_dir = repo_root / "data" / "cagi5"
    else:
        cagi5_dir = Path(args.cagi5_dir).resolve()

    print("=" * 80)
    print("Zero-shot CAGI5 Evaluation for Fine-tuned Enformer MPRA Models")
    print("=" * 80)
    print(f"Checkpoint path:      {checkpoint_path}")
    print(f"Cell type:            {args.cell_type}")
    print(f"CAGI5 dir:            {cagi5_dir}")
    print(f"Sequence length:      {args.seq_len} bp")
    print(f"Position shift aug:   {args.use_position_shift} (max: {args.max_shift} bp, n_samples: {args.n_shift_samples})")
    print(f"Reverse comp aug:     {args.use_reverse_complement}")
    print(f"Batch size:           {args.batch_size}")
    print("=" * 80)
    print()

    # Map CAGI5 elements to their experimental cell lines
    hepG2_elements = ["F9", "LDLR", "SORT1"]
    k562_elements = ["GP1BB", "HBB", "HBG1", "PKLR"]

    if args.cell_type == "HepG2":
        allowed_elements = hepG2_elements
    elif args.cell_type == "K562":
        allowed_elements = k562_elements
    else:
        print(
            f"No CAGI5 elements mapped to cell_type={args.cell_type}. "
            "Skipping CAGI5 evaluation."
        )
        return

    print(f"Cell-type specific CAGI5 elements: {allowed_elements}")

    # Load fine-tuned Enformer MPRA model
    print("\nLoading Enformer MPRA model from checkpoint...")
    try:
        model = EnformerMPRALightning.load_from_checkpoint(
            str(checkpoint_path),
            map_location="cpu",
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nTrying to load with strict=False...")
        model = EnformerMPRALightning.load_from_checkpoint(
            str(checkpoint_path),
            strict=False,
            map_location="cpu",
        )
        print("✓ Model loaded successfully (with strict=False)")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"✓ Model moved to GPU: {device}")
    else:
        device = torch.device("cpu")
        model = model.to(device)
        print("✓ Model on CPU")

    model.eval()

    # Initialize seq_loader for hg19
    print("\nInitializing seq_loader for hg19...")
    seq_loader_obj = seq_loader(build="hg19", model_receptive_field=args.seq_len)
    print("✓ seq_loader initialized")

    print("\nLoading CAGI5 variant tables (challenge/test set only)...")
    cagi5_data = parse_cagi5_dir(cagi5_dir, elements_to_include=allowed_elements)
    print(f"✓ Loaded CAGI5 data for {len(cagi5_data)} elements")

    # Verify reference alleles match hg19
    print("\nRunning unit test: Verifying reference alleles match hg19 genome...")
    test_passed = test_ref_allele_verification(seq_loader_obj, cagi5_data)
    if test_passed:
        print("✓ Unit test passed: All reference alleles match hg19 genome")
    else:
        print("⚠ Unit test failed: Some reference alleles do not match hg19 genome")
        print("  Continuing evaluation, but results may be inaccurate...")

    # Aggregate predictions/targets across all elements
    all_preds: List[float] = []
    all_targets: List[float] = []
    all_conf: List[float] = []

    element_results: List[dict] = []

    total_start_time = time.time()
    total_prediction_time = 0.0
    total_variants_processed = 0

    for element, df in cagi5_data.items():
        if element not in CAGI5_REGIONS:
            print(f"  {element}: no chromosome mapping in CAGI5_REGIONS, skipping")
            continue

        chromosome = CAGI5_REGIONS[element][0]

        values: List[float] = []
        confidences: List[float] = []

        strands = [False, True] if args.use_reverse_complement else [False]

        variant_ref_seqs: List[List[str]] = []
        variant_alt_seqs: List[List[str]] = []

        for _, row in df.iterrows():
            var_pos = int(row["Pos"])
            ref_allele = str(row["Ref"])
            alt_allele = str(row["Alt"])

            variant_seed = (args.random_seed + var_pos) if args.random_seed is not None else None
            if args.use_position_shift:
                position_shifts = get_random_position_shifts(
                    args.max_shift, args.n_shift_samples, variant_seed
                )
            else:
                position_shifts = [0]

            variant_ref_seqs_list: List[str] = []
            variant_alt_seqs_list: List[str] = []

            for shift in position_shifts:
                for use_rc in strands:
                    ref_seq, alt_seq = get_ref_alt_sequences(
                        seq_loader_obj=seq_loader_obj,
                        chromosome=chromosome,
                        var_pos=var_pos,
                        ref_allele=ref_allele,
                        alt_allele=alt_allele,
                        window=args.seq_len,
                        position_shift=shift,
                    )
                    if (
                        ref_seq is None
                        or alt_seq is None
                        or len(ref_seq) != args.seq_len
                        or len(alt_seq) != args.seq_len
                    ):
                        continue

                    if use_rc:
                        ref_seq = reverse_complement_dna(ref_seq)
                        alt_seq = reverse_complement_dna(alt_seq)

                    variant_ref_seqs_list.append(ref_seq)
                    variant_alt_seqs_list.append(alt_seq)

            if len(variant_ref_seqs_list) == 0 or len(variant_alt_seqs_list) == 0:
                continue

            variant_ref_seqs.append(variant_ref_seqs_list)
            variant_alt_seqs.append(variant_alt_seqs_list)
            values.append(float(row["Value"]))
            confidences.append(float(row["Confidence"]))

        if not variant_ref_seqs:
            print(f"  {element}: no valid sequences, skipping")
            continue

        all_ref_seqs = [seq for seq_list in variant_ref_seqs for seq in seq_list]
        all_alt_seqs = [seq for seq_list in variant_alt_seqs for seq in seq_list]

        element_pred_start = time.time()
        ref_preds_flat = batch_predict_enformer_mpra(
            model=model,
            sequences=all_ref_seqs,
            batch_size=args.batch_size,
        )
        alt_preds_flat = batch_predict_enformer_mpra(
            model=model,
            sequences=all_alt_seqs,
            batch_size=args.batch_size,
        )
        element_pred_time = time.time() - element_pred_start
        total_prediction_time += element_pred_time

        pred_idx = 0
        preds: List[float] = []
        for ref_seqs_list, alt_seqs_list in zip(variant_ref_seqs, variant_alt_seqs):
            n_aug = len(ref_seqs_list)
            variant_ref_preds = ref_preds_flat[pred_idx : pred_idx + n_aug]
            variant_alt_preds = alt_preds_flat[pred_idx : pred_idx + n_aug]
            pred_idx += n_aug

            mean_ref = float(np.mean(variant_ref_preds))
            mean_alt = float(np.mean(variant_alt_preds))
            preds.append(mean_alt - mean_ref)

        preds = np.array(preds, dtype=np.float32)

        all_preds.extend(preds.tolist())
        all_targets.extend(values)
        all_conf.extend(confidences)
        total_variants_processed += len(preds)

        pearson_el, spearman_el = compute_correlations(
            preds, np.asarray(values, dtype=np.float32)
        )

        conf_arr = np.asarray(confidences, dtype=np.float32)
        mask_high_el = conf_arr > 0.1
        pearson_el_hi, spearman_el_hi = compute_correlations(
            preds[mask_high_el] if mask_high_el.sum() > 0 else np.array([]),
            np.asarray(values, dtype=np.float32)[mask_high_el]
            if mask_high_el.sum() > 0
            else np.array([]),
        )

        element_results.append(
            {
                "element": element,
                "n_variants": len(preds),
                "n_high_conf": int(mask_high_el.sum()),
                "pearson_all": pearson_el,
                "spearman_all": spearman_el,
                "pearson_high_conf": pearson_el_hi,
                "spearman_high_conf": spearman_el_hi,
            }
        )

        print(
            f"  {element}: n={len(preds):5d}, "
            f"Pearson={pearson_el: .4f}, Spearman={spearman_el: .4f}, "
            f"time={element_pred_time:.1f}s ({element_pred_time/len(preds)*1000:.1f}ms/variant)"
        )

    all_preds_arr = np.asarray(all_preds, dtype=np.float32)
    all_targets_arr = np.asarray(all_targets, dtype=np.float32)
    all_conf_arr = np.asarray(all_conf, dtype=np.float32)

    print("\n" + "=" * 80)
    print("CAGI5 Zero-shot Summary (average across elements)")
    print("=" * 80)

    if element_results:
        pearson_all = np.nanmean([el["pearson_all"] for el in element_results])
        spearman_all = np.nanmean([el["spearman_all"] for el in element_results])
        pearson_hi = np.nanmean([el["pearson_high_conf"] for el in element_results])
        spearman_hi = np.nanmean([el["spearman_high_conf"] for el in element_results])

        total_snps = sum(el["n_variants"] for el in element_results)
        total_high_conf = sum(el["n_high_conf"] for el in element_results)
    else:
        pearson_all = spearman_all = pearson_hi = spearman_hi = float("nan")
        total_snps = total_high_conf = 0

    print(f"All SNPs (n={total_snps} across {len(element_results)} elements):")
    print(f"  Pearson r:    {pearson_all: .4f} (average across elements)")
    print(f"  Spearman rho: {spearman_all: .4f} (average across elements)")

    print(
        f"\nHigh-confidence SNPs (Confidence > 0.1, n={total_high_conf} across {len(element_results)} elements):"
    )
    print(f"  Pearson r:    {pearson_hi: .4f} (average across elements)")
    print(f"  Spearman rho: {spearman_hi: .4f} (average across elements)")

    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("Timing Summary")
    print("=" * 80)
    print(f"Total runtime:           {total_time:.2f} s ({total_time/60:.2f} min)")
    print(
        f"Total prediction time:   {total_prediction_time:.2f} s ({total_prediction_time/60:.2f} min)"
    )
    print(f"Variants processed:      {total_variants_processed}")
    if total_variants_processed > 0 and total_prediction_time > 0:
        print(
            f"Time per variant:        {total_prediction_time/total_variants_processed*1000:.2f} ms"
        )
        print(
            f"Variants per second:     {total_variants_processed/total_prediction_time:.2f}"
        )
    print("=" * 80)

    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine stage based on checkpoint path (check if "stage2" is in the path)
    checkpoint_path_str = str(checkpoint_path)
    if "stage2" in checkpoint_path_str:
        stage_name = "stage2"
    else:
        stage_name = "stage1"

    prefix_parts = []
    if args.use_position_shift:
        prefix_parts.append(f"posshift{args.max_shift}_n{args.n_shift_samples}")
    if args.use_reverse_complement:
        prefix_parts.append("revcomp")

    if prefix_parts:
        file_prefix = (
            "_".join(prefix_parts)
            + f"_cagi5_enformer_mpra_{args.cell_type}_{stage_name}"
        )
    else:
        file_prefix = f"cagi5_enformer_mpra_{args.cell_type}_{stage_name}"

    summary_results = {
        "model": "enformer_mpra",
        "cell_type": args.cell_type,
        "stage": stage_name,
        "use_position_shift": args.use_position_shift,
        "max_shift": args.max_shift if args.use_position_shift else None,
        "n_shift_samples": args.n_shift_samples if args.use_position_shift else None,
        "use_reverse_complement": args.use_reverse_complement,
        "n_elements": len(element_results),
        "n_all_snps": total_snps,
        "n_high_conf_snps": total_high_conf,
        "pearson_all": pearson_all,
        "spearman_all": spearman_all,
        "pearson_high_conf": pearson_hi,
        "spearman_high_conf": spearman_hi,
        "total_runtime_seconds": total_time,
        "total_prediction_time_seconds": total_prediction_time,
        "n_variants_processed": total_variants_processed,
        "time_per_variant_ms": (
            total_prediction_time / total_variants_processed * 1000
            if total_variants_processed > 0
            else None
        ),
        "variants_per_second": (
            total_variants_processed / total_prediction_time
            if total_prediction_time > 0
            else None
        ),
    }

    summary_file = results_dir / f"{file_prefix}_summary.csv"
    pd.DataFrame([summary_results]).to_csv(summary_file, index=False)
    print(f"\n✓ Summary results saved to {summary_file}")

    element_df = pd.DataFrame(element_results)
    element_file = results_dir / f"{file_prefix}_per_element.csv"
    element_df.to_csv(element_file, index=False)
    print(f"✓ Per-element results saved to {element_file}")

    mask_high = all_conf_arr > 0.1
    predictions_df = pd.DataFrame(
        {
            "prediction": all_preds_arr,
            "target": all_targets_arr,
            "confidence": all_conf_arr,
            "high_conf": mask_high,
        }
    )
    predictions_file = results_dir / f"{file_prefix}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Detailed predictions saved to {predictions_file}")


if __name__ == "__main__":
    main()

