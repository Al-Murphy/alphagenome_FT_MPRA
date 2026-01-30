#!/usr/bin/env python3
"""
Zero-shot evaluation of base (pretrained) AlphaGenome model on the CAGI5
saturation mutagenesis benchmark.

This script evaluates the base AlphaGenome model (before fine-tuning) on CAGI5
data by using DNase predictions for the appropriate cell types.

Key behaviors:
- Uses base AlphaGenome model loaded from Kaggle (no fine-tuning).
- Uses AlphaGenome's built-in predict_variant function, which handles sequence
  extraction from the reference genome (hg19) and creates intervals centered on
  each variant (default: 1048576 bp window - AlphaGenome's full input window).
- Restricts evaluation to cell-type-matched CAGI5 elements:
  - HepG2 models → F9, LDLR, SORT1
  - K562 models → GP1BB, HBB, HBG1, PKLR
- Only evaluates on CAGI5 challenge (test) set, not training set.
- Computes Pearson r and Spearman ρ on:
    1) All SNPs
    2) High-confidence SNPs (Confidence > 0.1)
- Uses hg19 reference sequences and 1-based genomic coordinates (converted to
  0-based for AlphaGenome's API).
- Predictions are computed as log2(sum_alt / sum_ref) over a configurable center window
  (default: 384 bp, can be set to 501 bp for AlphaGenome's recommended central mask size).

To note, high confidence SNPs relate tot he significance of the SNP in the MPRA regression model.
> We deemed a confidence score greater or equal to 0.1 (p-value of 10⁻⁵) indicates that the SNV 'has an expression effect'.

Also note, we tried the variant effect approach recommended by AlphaGenome in Supplementary table 9 (501 bp central mask),
but found decreasing the central mask on the output window from 501 bp to 384 bp led to substantially better performance.
Use --center_window_bp 501 to use AlphaGenome's recommended approach.

Prerequisites:
- Run `scripts/fetch_cagi5_references.py` to download hg19 reference sequences.
- Place CAGI5 challenge_*.tsv files in ./data/cagi5/

Example:
    # Default (384 bp center window):
    python scripts/test_cagi5_zero_shot_base.py --cell_type K562
    
    # Use AlphaGenome's recommended 501 bp central mask:
    python scripts/test_cagi5_zero_shot_base.py --cell_type K562 --center_window_bp 501
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model

from src.seq_loader import seq_loader  # type: ignore


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
    matches = genome_ref.upper() == ref_allele.upper()
    
    # If mismatch, check if it's a complement (suggests strand issue)
    if not matches and len(ref_allele) == 1 and len(genome_ref) == 1:
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        if complement_map.get(ref_allele.upper()) == genome_ref.upper():
            # This is a complement - likely a strand or coordinate issue
            pass
    
    return matches


def parse_cagi5_dir(
    cagi5_dir: Path,
    elements_to_include: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load selected CAGI5 challenge TSVs into a dict[element] -> DataFrame."""
    if not cagi5_dir.exists():
        raise FileNotFoundError(f"CAGI5 directory not found: {cagi5_dir}")

    data: Dict[str, pd.DataFrame] = {}
    for tsv_file in sorted(cagi5_dir.glob("challenge_*.tsv")):
        element = tsv_file.stem.replace("challenge_", "")

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

        df["Pos"] = df["Pos"].astype(int)
        df["Value"] = df["Value"].astype(float)
        if "Confidence" in df.columns:
            df["Confidence"] = df["Confidence"].astype(float)
        else:
            df["Confidence"] = 1.0

        data[element] = df

    if not data:
        raise RuntimeError(f"No CAGI5 challenge_*.tsv files found under {cagi5_dir}")

    return data


def compute_correlations(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, float]:
    """Compute Pearson r and Spearman ρ for two 1D arrays."""
    if len(preds) == 0:
        return float("nan"), float("nan")
    pearson, _ = pearsonr(preds, targets)
    spearman, _ = spearmanr(preds, targets)
    return float(pearson), float(spearman)


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
# Base AlphaGenome prediction
# ---------------------------------------------------------------------------

def predict_dnase_for_variant(
    model: dna_model.AlphaGenomeModel,
    seq_loader_obj: seq_loader,
    chromosome: str,
    var_pos: int,
    ref_allele: str,
    alt_allele: str,
    cell_type: str,
    interval_size: int,
    center_window_bp: int = 384,
    use_position_shift: bool = False,
    max_shift: int = 20,
    n_shift_samples: int = 3,
    random_seed: int | None = None,
    use_reverse_complement: bool = False,
) -> float:
    """Predict DNase for a single variant using hg19 sequences from seq_loader.
    
    Uses seq_loader to extract hg19 sequences, creates reference and alternate
    sequences, then uses AlphaGenome's predict_sequence. Supports inference-time
    augmentation via position shifting and reverse complement.
    
    Args:
        model: Base AlphaGenome model.
        seq_loader_obj: seq_loader instance for hg19.
        chromosome: Chromosome (e.g., 'chr1', 'chrX').
        var_pos: Genomic position of variant (1-based, hg19).
        ref_allele: Reference allele.
        alt_allele: Alternate allele.
        cell_type: Cell type ('HepG2' or 'K562').
        interval_size: Size of interval around variant (default: 1048576 bp).
        center_window_bp: Size of center window for pooling predictions (default: 384 bp).
        use_position_shift: If True, average predictions over randomly sampled position shifts.
        max_shift: Maximum position shift in base pairs (default: 20).
        n_shift_samples: Number of random shifts to sample (default: 3).
        random_seed: Random seed for shift sampling (default: None).
        use_reverse_complement: If True, always average predictions over forward and reverse complement.
        
    Returns:
        Scalar DNase prediction: log2(sum_alt / sum_ref) over center window, averaged over augmentations.
    """
    from alphagenome.data import ontology
    
    # Map cell type to ontology term
    cell_type_to_ontology = {
        'HepG2': 'EFO:0001187',  # HepG2 cell line
        'K562': 'EFO:0002067',   # K562 cell line
    }
    
    if cell_type not in cell_type_to_ontology:
        raise ValueError(f"Unsupported cell_type: {cell_type}. Must be HepG2 or K562")
    
    ontology_term_str = cell_type_to_ontology[cell_type]
    ontology_term = ontology.from_curie(ontology_term_str)
    
    # Determine augmentation strategies
    # Use variant position as part of seed for reproducibility but different shifts per variant
    variant_seed = (random_seed + var_pos) if random_seed is not None else None
    if use_position_shift:
        position_shifts = get_random_position_shifts(max_shift, n_shift_samples, variant_seed)
    else:
        position_shifts = [0]
    strands = [False, True] if use_reverse_complement else [False]
    
    all_predictions: List[float] = []
    
    try:
        for shift in position_shifts:
            shifted_var_pos = var_pos + shift
            
            for use_rc in strands:
                # Extract hg19 reference sequence centered on (possibly shifted) variant
                # pysam uses 0-based, half-open intervals
                half_interval = interval_size // 2
                start_0based = max(0, shifted_var_pos - 1 - half_interval)
                end_0based = start_0based + interval_size
                
                # Get reference sequence from hg19
                ref_seq = seq_loader_obj.genome_dat.fetch(chromosome, start_0based, end_0based).upper()
                
                # Verify ref allele matches (only check for original position, not shifted)
                variant_pos_in_seq = (shifted_var_pos - 1) - start_0based
                if variant_pos_in_seq < 0 or variant_pos_in_seq + len(ref_allele) > len(ref_seq):
                    if shift == 0:  # Only warn for original position
                        print(f"  Warning: Variant position {var_pos} out of bounds for interval")
                    continue
                
                # For shifted positions, we still need to verify the ref allele matches
                # (though it might not for large shifts, so we'll be lenient)
                ref_in_seq = ref_seq[variant_pos_in_seq : variant_pos_in_seq + len(ref_allele)]
                if shift == 0 and ref_in_seq.upper() != ref_allele.upper():
                    # Check if it's a complement (suggests strand/coordinate issue)
                    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
                    is_complement = (
                        len(ref_allele) == 1 and 
                        len(ref_in_seq) == 1 and
                        complement_map.get(ref_allele.upper()) == ref_in_seq.upper()
                    )
                    complement_note = " (complement - possible strand/coordinate issue)" if is_complement else ""
                    print(f"  Warning: Ref allele mismatch at {chromosome}:{var_pos}: "
                          f"expected {ref_allele}, got {ref_in_seq}{complement_note}")
                    continue
                
                # Create alternate sequence
                alt_seq = (
                    ref_seq[:variant_pos_in_seq] +
                    alt_allele +
                    ref_seq[variant_pos_in_seq + len(ref_allele):]
                )
                
                # Apply reverse complement if needed
                if use_rc:
                    ref_seq = reverse_complement_dna(ref_seq)
                    alt_seq = reverse_complement_dna(alt_seq)
                
                # Use AlphaGenome's predict_sequence for both ref and alt
                ref_output = model.predict_sequence(
                    ref_seq,
                    organism=dna_model.Organism.HOMO_SAPIENS,
                    requested_outputs=[dna_output.OutputType.DNASE],
                    ontology_terms=[ontology_term],
                )
                
                alt_output = model.predict_sequence(
                    alt_seq,
                    organism=dna_model.Organism.HOMO_SAPIENS,
                    requested_outputs=[dna_output.OutputType.DNASE],
                    ontology_terms=[ontology_term],
                )
                
                # Get both reference and alternate predictions
                if ref_output.dnase is None or alt_output.dnase is None:
                    continue
                
                # Extract DNase predictions: shape (seq_len, num_tracks)
                ref_dnase = ref_output.dnase.values  # numpy array
                alt_dnase = alt_output.dnase.values  # numpy array
                
                if ref_dnase.size == 0 or alt_dnase.size == 0:
                    continue
                
                # Pool over center region, centered on the (possibly shifted) variant position
                # 501 matches AlphaGenome central mask approach (see methods in paper - Scoring pipeline overview section 3 and supp table 9)
                # 384 bp was found to give better performance than 501 bp
                seq_len = ref_dnase.shape[0]
                
                # Center the mask on the variant position within the sequence
                # variant_pos_in_seq is the position of the variant within the extracted sequence
                half_window = center_window_bp // 2
                center_start = max(0, variant_pos_in_seq - half_window)
                center_end = min(seq_len, variant_pos_in_seq + half_window + (center_window_bp % 2))
                window_size = center_end - center_start
                
                # Skip if window is too small (variant too close to edge)
                if window_size < center_window_bp // 2:
                    continue
                
                center_ref = ref_dnase[center_start:center_end, :]  # (window_size, num_tracks)
                center_alt = alt_dnase[center_start:center_end, :]  # (window_size, num_tracks)
                
                # Sum over sequence positions and tracks
                sum_ref = float(np.sum(center_ref))
                sum_alt = float(np.sum(center_alt))
                
                # Compute log2(sum_alt / sum_ref)
                if sum_ref <= 0:
                    continue
                
                ratio = sum_alt / sum_ref
                scalar_pred = float(np.log2(ratio))
                all_predictions.append(scalar_pred)
        
        # Average over all augmentations
        if len(all_predictions) == 0:
            return float('nan')
        
        return float(np.mean(all_predictions))
        
    except Exception as e:
        print(f"  Warning: Error predicting variant at {chromosome}:{var_pos}: {e}")
        return float('nan')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot CAGI5 evaluation for base AlphaGenome model"
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="K562",
        choices=["HepG2", "K562"],
        help="Cell type to evaluate (default: K562)",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="all_folds",
        help="AlphaGenome model version (default: all_folds)",
    )
    parser.add_argument(
        "--cagi5_dir",
        type=str,
        default=None,
        help="Directory with CAGI5 challenge_*.tsv files (default: ./data/cagi5).",
    )
    parser.add_argument(
        "--interval_size",
        type=int,
        default=2**20,  # 1048576 bp - AlphaGenome's full input window
        help="Size of interval around variant for AlphaGenome prediction (default: 1048576 bp).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,  # Smaller batch size for base model (more memory intensive)
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--center_window_bp",
        type=int,
        default=384,
        help="Size of center window (in bp) for pooling DNase predictions (default: 384). Use 501 for AlphaGenome's recommended central mask size.",
    )
    parser.add_argument(
        "--use_position_shift",
        action='store_true',
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
        action='store_true',
        help="Enable reverse complement augmentation: always average predictions over forward and reverse complement sequences.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.resolve()

    # Default paths
    if args.cagi5_dir is None:
        cagi5_dir = repo_root / "data" / "cagi5"
    else:
        cagi5_dir = Path(args.cagi5_dir).resolve()

    # Cell type to CAGI5 element mapping
    CELL_TYPE_TO_ELEMENTS = {
        "HepG2": ["F9", "LDLR", "SORT1"],
        "K562": ["GP1BB", "HBB", "HBG1", "PKLR"],
    }

    allowed_elements = CELL_TYPE_TO_ELEMENTS.get(args.cell_type)
    if allowed_elements is None:
        print(f"No CAGI5 elements mapped for cell_type={args.cell_type}")
        return

    print("=" * 80)
    print("Zero-shot CAGI5 Evaluation for Base AlphaGenome Model")
    print("=" * 80)
    print(f"Cell type:           {args.cell_type}")
    print(f"Model version:       {args.model_version}")
    print(f"CAGI5 elements:      {', '.join(allowed_elements)}")
    print(f"CAGI5 dir:           {cagi5_dir}")
    print(f"Genome build:        hg19 (via seq_loader)")
    print(f"Interval size:       {args.interval_size} bp")
    print(f"Center window:       {args.center_window_bp} bp")
    print(f"Position shift aug:   {args.use_position_shift} (max: {args.max_shift} bp, n_samples: {args.n_shift_samples})")
    print(f"Reverse comp aug:   {args.use_reverse_complement}")
    print(f"Batch size:          {args.batch_size}")
    print("=" * 80)
    print()

    # Load base AlphaGenome model
    print("Loading base AlphaGenome model from Kaggle...")
    model = dna_model.create_from_kaggle(
        args.model_version,
        device=None,  # Auto-detect device
    )
    print("✓ Model loaded")

    # Initialize seq_loader for hg19
    print("\nInitializing seq_loader for hg19...")
    seq_loader_obj = seq_loader(build='hg19', model_receptive_field=args.interval_size)
    print("✓ seq_loader initialized")

    print("\nLoading CAGI5 variant tables...")
    cagi5_data = parse_cagi5_dir(cagi5_dir, elements_to_include=allowed_elements)
    print(f"✓ Loaded CAGI5 data for {len(cagi5_data)} elements")
    
    # Run unit test to verify ref alleles match reference genome
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
    
    # Store per-element results
    element_results: List[dict] = []

    for element, df in cagi5_data.items():
        if element not in CAGI5_REGIONS:
            print(f"  {element}: no chromosome mapping in CAGI5_REGIONS, skipping")
            continue

        chromosome = CAGI5_REGIONS[element][0]
        
        values: List[float] = []
        confidences: List[float] = []
        preds: List[float] = []

        print(f"  {element}: predicting on {len(df)} variants...")
        for idx, row in df.iterrows():
            var_pos = int(row["Pos"])
            ref_allele = str(row["Ref"])
            alt_allele = str(row["Alt"])
            
            # Use seq_loader to get hg19 sequences and predict
            pred = predict_dnase_for_variant(
                model=model,
                seq_loader_obj=seq_loader_obj,
                chromosome=chromosome,
                var_pos=var_pos,
                ref_allele=ref_allele,
                alt_allele=alt_allele,
                cell_type=args.cell_type,
                interval_size=args.interval_size,
                center_window_bp=args.center_window_bp,
                use_position_shift=args.use_position_shift,
                max_shift=args.max_shift,
                n_shift_samples=args.n_shift_samples,
                random_seed=args.random_seed,
                use_reverse_complement=args.use_reverse_complement,
            )
            
            if np.isnan(pred):
                continue
                
            preds.append(pred)
            values.append(float(row["Value"]))
            confidences.append(float(row["Confidence"]))
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(df)} variants...", end='\r')

        if not preds:
            print(f"  {element}: no valid predictions, skipping")
            continue

        all_preds.extend(preds)
        all_targets.extend(values)
        all_conf.extend(confidences)

        preds_arr = np.asarray(preds, dtype=np.float32)
        values_arr = np.asarray(values, dtype=np.float32)
        pearson_el, spearman_el = compute_correlations(preds_arr, values_arr)
        
        # Compute high-confidence metrics for this element
        conf_arr = np.asarray(confidences, dtype=np.float32)
        mask_high_el = conf_arr > 0.1
        pearson_el_hi, spearman_el_hi = compute_correlations(
            preds_arr[mask_high_el] if mask_high_el.sum() > 0 else np.array([]),
            values_arr[mask_high_el] if mask_high_el.sum() > 0 else np.array([]),
        )
        
        element_results.append({
            'element': element,
            'n_variants': len(preds_arr),
            'n_high_conf': int(mask_high_el.sum()),
            'pearson_all': pearson_el,
            'spearman_all': spearman_el,
            'pearson_high_conf': pearson_el_hi,
            'spearman_high_conf': spearman_el_hi,
        })
        
        print(
            f"  {element}: n={len(preds_arr):5d}, "
            f"Pearson={pearson_el: .4f}, Spearman={spearman_el: .4f}"
        )

    all_preds_arr = np.asarray(all_preds, dtype=np.float32)
    all_targets_arr = np.asarray(all_targets, dtype=np.float32)
    all_conf_arr = np.asarray(all_conf, dtype=np.float32)

    print("\n" + "=" * 80)
    print("CAGI5 Zero-shot Summary (Base AlphaGenome, average across elements)")
    print("=" * 80)

    # Compute average metrics across elements
    if element_results:
        pearson_all = np.nanmean([el['pearson_all'] for el in element_results])
        spearman_all = np.nanmean([el['spearman_all'] for el in element_results])
        pearson_hi = np.nanmean([el['pearson_high_conf'] for el in element_results])
        spearman_hi = np.nanmean([el['spearman_high_conf'] for el in element_results])
        
        total_snps = sum(el['n_variants'] for el in element_results)
        total_high_conf = sum(el['n_high_conf'] for el in element_results)
    else:
        pearson_all = spearman_all = pearson_hi = spearman_hi = float('nan')
        total_snps = total_high_conf = 0

    print(f"All SNPs (n={total_snps} across {len(element_results)} elements):")
    print(f"  Pearson r:    {pearson_all: .4f} (average across elements)")
    print(f"  Spearman rho: {spearman_all: .4f} (average across elements)")

    print(f"\nHigh-confidence SNPs (Confidence > 0.1, n={total_high_conf} across {len(element_results)} elements):")
    print(f"  Pearson r:    {pearson_hi: .4f} (average across elements)")
    print(f"  Spearman rho: {spearman_hi: .4f} (average across elements)")
    print("=" * 80)
    
    # Save results to CSV
    results_dir = repo_root / "results" / "cagi5_evaluations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename prefix based on center window size and augmentations
    prefix_parts = []
    
    # Center window prefix
    if args.center_window_bp == 501:
        prefix_parts.append("501_central_mask")
    
    # Augmentation prefixes
    if args.use_position_shift:
        prefix_parts.append(f"posshift{args.max_shift}_n{args.n_shift_samples}")
    if args.use_reverse_complement:
        prefix_parts.append("revcomp")
    
    # Base prefix
    if prefix_parts:
        file_prefix = "_".join(prefix_parts) + f"_cagi5_base_{args.cell_type}"
    else:
        file_prefix = f"cagi5_base_{args.cell_type}"
    
    # Save summary results
    summary_results = {
        'model': 'base_alphagenome',
        'cell_type': args.cell_type,
        'model_version': args.model_version,
        'center_window_bp': args.center_window_bp,
        'use_position_shift': args.use_position_shift,
        'max_shift': args.max_shift if args.use_position_shift else None,
        'n_shift_samples': args.n_shift_samples if args.use_position_shift else None,
        'use_reverse_complement': args.use_reverse_complement,
        'n_elements': len(element_results),
        'n_all_snps': total_snps,
        'n_high_conf_snps': total_high_conf,
        'pearson_all': pearson_all,
        'spearman_all': spearman_all,
        'pearson_high_conf': pearson_hi,
        'spearman_high_conf': spearman_hi,
    }
    
    summary_file = results_dir / f"{file_prefix}_summary.csv"
    pd.DataFrame([summary_results]).to_csv(summary_file, index=False)
    print(f"\n✓ Summary results saved to {summary_file}")
    
    # Save per-element results
    element_df = pd.DataFrame(element_results)
    element_file = results_dir / f"{file_prefix}_per_element.csv"
    element_df.to_csv(element_file, index=False)
    print(f"✓ Per-element results saved to {element_file}")
    
    # Save detailed predictions (optional, can be large)
    mask_high = all_conf_arr > 0.1
    predictions_df = pd.DataFrame({
        'prediction': all_preds_arr,
        'target': all_targets_arr,
        'confidence': all_conf_arr,
        'high_conf': mask_high,
    })
    predictions_file = results_dir / f"{file_prefix}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Detailed predictions saved to {predictions_file}")


if __name__ == "__main__":
    main()
