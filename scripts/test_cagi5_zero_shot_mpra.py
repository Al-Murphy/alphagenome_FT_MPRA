#!/usr/bin/env python3
"""
Zero-shot evaluation of fine-tuned AlphaGenome MPRA models on the CAGI5
saturation mutagenesis benchmark.

Key behaviors:
- Uses the same finetuned MPRA head (`mpra_head`) and loading logic as
  `scripts/test_ft_model_mpra.py`.
- Restricts evaluation to cell-type-matched CAGI5 elements:
  - HepG2 models → F9, LDLR, SORT1
  - K562 models → GP1BB, HBB, HBG1, PKLR
- Only evaluates on CAGI5 challenge (test) set, not training set.
- Computes variant effects as the difference between alt and ref predictions:
  variant_effect = alt_prediction - ref_prediction. 
  Note this is becuase we are using the MPRA head, which is already in log scale,
  it predicts log(RNA/DNA). So would not make sense to use the genome-wide model 
  approach of log(sum(alt)/sum(ref)). The MPRA values already log-transformed &
  residualized / normalized relative to background
- Computes Pearson r and Spearman ρ on:
    1) All SNPs
    2) High-confidence SNPs (Confidence > 0.1)
- Uses hg19 reference sequences and 1-based genomic coordinates.
- Sequence length matches training: 281 bp (full promoter construct).

To note, high confidence SNPs relate tot he significance of the SNP in the MPRA regression model.
> We deemed a confidence score greater or equal to 0.1 (p-value of 10⁻⁵) indicates that the SNV ‘has an expression effect’.

Prerequisites:
- Run `scripts/fetch_cagi5_references.py` to download hg19 reference sequences.
- Place CAGI5 challenge_*.tsv files in ./data/cagi5/

Example:
    # First, fetch reference sequences (one-time setup):
    python scripts/fetch_cagi5_references.py
    
    # Then evaluate:
    python scripts/test_cagi5_zero_shot_mpra.py \\
        --checkpoint_dir ./results/models/checkpoints/mpra_encoder_head_K562 \\
        --cell_type K562
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model

from alphagenome_ft import (
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)
from src import EncoderMPRAHead  # type: ignore

from src.seq_loader import seq_loader  # type: ignore


PROMOTER_CONSTRUCT_LENGTH = 281  # same init_seq_len as finetune_mpra.py


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


def batch_predict_mpra(
    model,
    sequences: List[str],
    head_name: str = "mpra_head",
    batch_size: int = 64,
) -> np.ndarray:
    """Run MPRA head predictions on a list of DNA sequences."""
    if not sequences:
        return np.array([], dtype=np.float32)

    # Infer pooling config from head metadata
    head_config = getattr(model, "_head_configs", {}).get(head_name, None)
    if head_config is not None:
        metadata = getattr(head_config, "metadata", {}) or {}
        pooling_type = metadata.get("pooling_type", "sum")
    else:
        metadata = {}
        pooling_type = "sum"

    all_preds: List[np.ndarray] = []
    device = model._device_context._device  # type: ignore[attr-defined]

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]

        # One-hot encode using AlphaGenome's encoder
        one_hot_list = [model._one_hot_encoder.encode(seq) for seq in batch_seqs]  # type: ignore[attr-defined]
        max_len = max(arr.shape[0] for arr in one_hot_list)

        # Pad to same length
        padded = []
        for arr in one_hot_list:
            if arr.shape[0] < max_len:
                pad = np.zeros((max_len - arr.shape[0], 4), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            padded.append(arr)

        batch_seq = jnp.array(np.stack(padded, axis=0))
        organism_index = jnp.zeros((batch_seq.shape[0],), dtype=jnp.int32)

        with model._device_context:  # type: ignore[attr-defined]
            predictions = model._predict(  # type: ignore[attr-defined]
                model._params,
                model._state,
                batch_seq,
                organism_index,
                negative_strand_mask=jnp.zeros(batch_seq.shape[0], dtype=bool),
                strand_reindexing=jax.device_put(
                    model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,  # type: ignore[attr-defined]
                    device,
                ),
            )

        head_predictions = predictions[head_name]
        head_np = np.array(head_predictions)
        seq_len = head_np.shape[1]

        # Pool like in test_ft_model_mpra.get_predictions_for_saving
        if pooling_type == "flatten":
            pooled = head_np.squeeze(1)
        elif pooling_type == "center":
            center_idx = seq_len // 2
            pooled = head_np[:, center_idx, :]
        else:
            center_bp = metadata.get("center_bp", 256) if metadata else 256
            center_window_positions = max(1, center_bp // 128)
            window_size = min(center_window_positions, seq_len)
            center_start = max((seq_len - window_size) // 2, 0)
            center_end = center_start + window_size
            center_preds = head_np[:, center_start:center_end, :]

            if pooling_type == "mean":
                pooled = center_preds.mean(axis=1)
            elif pooling_type == "max":
                pooled = center_preds.max(axis=1)
            else:
                pooled = center_preds.sum(axis=1)

        if pooled.shape[-1] == 1:
            pooled = pooled[:, 0]

        all_preds.append(pooled.astype(np.float32))

    return np.concatenate(all_preds, axis=0)


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
# Model loading (adapted from scripts/test_ft_model_mpra.py)
# ---------------------------------------------------------------------------

def load_finetuned_mpra_model(
    checkpoint_dir: Path,
    base_checkpoint_path: str | None = None,
) -> object:
    """Load fine-tuned AlphaGenome MPRA model (encoder head) from checkpoint_dir.

    This mirrors the logic in `scripts/test_ft_model_mpra.py` so that:
    - The MPRA head metadata (center_bp, pooling_type, etc.) matches training.
    - Both Stage 1 (heads-only) and Stage 2 (full-model) checkpoints work.
    """
    import orbax.checkpoint as ocp

    checkpoint_dir = checkpoint_dir.resolve()
    config_path = checkpoint_dir / "config.json"

    # Default head metadata (overridden by config if present)
    head_metadata = {
        "center_bp": 256,
        "pooling_type": "sum",
    }
    save_full_model = True

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            head_cfg = (
                cfg.get("head_configs", {})
                .get("mpra_head", {})
                .get("metadata", {})
            )
            head_metadata.update(head_cfg)
            save_full_model = cfg.get("save_full_model", False)
        except Exception:
            pass

    # Register MPRA head with (possibly) config-derived metadata
    register_custom_head(
        "mpra_head",
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name="mpra_head",
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata=head_metadata,
        ),
    )

    # Create model with encoder-output head on short promoter constructs
    init_seq_len = PROMOTER_CONSTRUCT_LENGTH
    model = create_model_with_custom_heads(
        "all_folds",
        custom_heads=["mpra_head"],
        checkpoint_path=base_checkpoint_path,
        use_encoder_output=True,
        init_seq_len=init_seq_len,
    )

    # Load checkpoint parameters using Orbax
    checkpoint_path = checkpoint_dir / "checkpoint"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpointer = ocp.StandardCheckpointer()
    loaded_params, loaded_state = checkpointer.restore(checkpoint_path)

    if save_full_model:
        # Full-model checkpoint: deep-merge params/state into model
        def deep_merge_params(model_params, checkpoint_params):
            import copy

            merged = copy.deepcopy(model_params)
            if isinstance(checkpoint_params, dict) and isinstance(model_params, dict):
                for key, checkpoint_value in checkpoint_params.items():
                    if key in model_params:
                        if isinstance(model_params[key], dict) and isinstance(
                            checkpoint_value, dict
                        ):
                            merged[key] = deep_merge_params(
                                model_params[key], checkpoint_value
                            )
                        else:
                            merged[key] = checkpoint_value
                    else:
                        merged[key] = checkpoint_value
            else:
                merged = checkpoint_params
            return merged

        model._params = deep_merge_params(model._params, loaded_params)  # type: ignore[attr-defined]
        model._state = deep_merge_params(model._state, loaded_state)  # type: ignore[attr-defined]
    else:
        # Heads-only checkpoint: merge head params into base model
        def merge_head_params(model_params, loaded_head_params):
            import copy

            merged = copy.deepcopy(model_params)

            if isinstance(loaded_head_params, dict):
                head_keys = {
                    k: v
                    for k, v in loaded_head_params.items()
                    if isinstance(k, str) and k.startswith("head/")
                }
                if head_keys:
                    for key, value in head_keys.items():
                        merged[key] = value

            if isinstance(loaded_head_params, dict) and "alphagenome/head" in loaded_head_params:
                if "alphagenome/head" not in merged:
                    merged["alphagenome/head"] = {}
                for head_name, head_params in loaded_head_params["alphagenome/head"].items():
                    merged["alphagenome/head"][head_name] = head_params

            if isinstance(loaded_head_params, dict) and "alphagenome" in loaded_head_params:
                if isinstance(loaded_head_params["alphagenome"], dict):
                    if "head" in loaded_head_params["alphagenome"]:
                        if "alphagenome" not in merged or not isinstance(
                            merged.get("alphagenome"), dict
                        ):
                            merged["alphagenome"] = {}
                        if "head" not in merged["alphagenome"]:
                            merged["alphagenome"]["head"] = {}
                        for head_name, head_params in loaded_head_params["alphagenome"]["head"].items():
                            merged["alphagenome"]["head"][head_name] = head_params

            return merged

        model._params = merge_head_params(model._params, loaded_params)  # type: ignore[attr-defined]
        model._state = merge_head_params(model._state, loaded_state)  # type: ignore[attr-defined]

    # Ensure parameters/state on correct device
    device = model._device_context._device  # type: ignore[attr-defined]
    model._params = jax.device_put(model._params, device)  # type: ignore[attr-defined]
    model._state = jax.device_put(model._state, device)  # type: ignore[attr-defined]

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot CAGI5 evaluation for fine-tuned AlphaGenome MPRA models"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to fine-tuned MPRA checkpoint directory",
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="K562",
        choices=["HepG2", "K562"],
        help="Cell type the model was fine-tuned on (default: K562)",
    )
    parser.add_argument(
        "--base_checkpoint_path",
        type=str,
        default=None,
        help="Optional local AlphaGenome checkpoint directory (bypasses Kaggle).",
    )
    parser.add_argument(
        "--cagi5_dir",
        type=str,
        default=None,
        help=(
            "Directory with CAGI5 challenge_*.tsv files "
            "(default: ./data/cagi5)."
        ),
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=PROMOTER_CONSTRUCT_LENGTH,
        help=f"Sequence window length around each variant (default: {PROMOTER_CONSTRUCT_LENGTH}, matches training).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
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

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    repo_root = Path(__file__).parent.parent.resolve()

    # Default paths to local data directory if not provided explicitly
    if args.cagi5_dir is None:
        cagi5_dir = repo_root / "data" / "cagi5"
    else:
        cagi5_dir = Path(args.cagi5_dir).resolve()

    print("=" * 80)
    print("Zero-shot CAGI5 Evaluation for Fine-tuned AlphaGenome MPRA Models")
    print("=" * 80)
    print(f"Checkpoint dir:      {checkpoint_dir}")
    print(f"Cell type:           {args.cell_type}")
    print(f"CAGI5 dir:           {cagi5_dir}")
    print(f"Genome build:        hg19 (via seq_loader)")
    print(f"Sequence length:     {args.seq_len} bp")
    print(f"Position shift aug:   {args.use_position_shift} (max: {args.max_shift} bp, n_samples: {args.n_shift_samples})")
    print(f"Reverse comp aug:   {args.use_reverse_complement}")
    print(f"Batch size:          {args.batch_size}")
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
        # No known CAGI5 sets for this cell type
        print(
            f"No CAGI5 elements mapped to cell_type={args.cell_type}. "
            "Skipping CAGI5 evaluation."
        )
        return

    print(f"Cell-type specific CAGI5 elements: {allowed_elements}")

    # Load model
    print("Loading fine-tuned MPRA model...")
    model = load_finetuned_mpra_model(
        checkpoint_dir=checkpoint_dir,
        base_checkpoint_path=args.base_checkpoint_path,
    )
    print("✓ Model loaded")

    # Initialize seq_loader for hg19
    print("\nInitializing seq_loader for hg19...")
    seq_loader_obj = seq_loader(build='hg19', model_receptive_field=args.seq_len)
    print("✓ seq_loader initialized")

    print("\nLoading CAGI5 variant tables (challenge/test set only)...")
    # Only load challenge_* files (held-out CAGI5 test set) and restrict
    # to elements whose experimental cell line matches the fine-tuned model.
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
        
        # Determine augmentation strategies
        strands = [False, True] if args.use_reverse_complement else [False]
        
        # Collect all sequences for batching
        variant_ref_seqs: List[List[str]] = []
        variant_alt_seqs: List[List[str]] = []

        for _, row in df.iterrows():
            var_pos = int(row["Pos"])
            ref_allele = str(row["Ref"])
            alt_allele = str(row["Alt"])
            
            # Get random position shifts for this variant
            # Use variant position as part of seed for reproducibility but different shifts per variant
            variant_seed = (args.random_seed + var_pos) if args.random_seed is not None else None
            if args.use_position_shift:
                position_shifts = get_random_position_shifts(args.max_shift, args.n_shift_samples, variant_seed)
            else:
                position_shifts = [0]
            
            # Collect sequences for all augmentations
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
                    if ref_seq is None or alt_seq is None or len(ref_seq) != args.seq_len or len(alt_seq) != args.seq_len:
                        continue
                    
                    # Apply reverse complement if needed
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

        # Flatten sequences for batch prediction
        all_ref_seqs = [seq for seq_list in variant_ref_seqs for seq in seq_list]
        all_alt_seqs = [seq for seq_list in variant_alt_seqs for seq in seq_list]
        
        # Predict on all sequences in batches
        ref_preds_flat = batch_predict_mpra(
            model=model,
            sequences=all_ref_seqs,
            head_name="mpra_head",
            batch_size=args.batch_size,
        )
        
        alt_preds_flat = batch_predict_mpra(
            model=model,
            sequences=all_alt_seqs,
            head_name="mpra_head",
            batch_size=args.batch_size,
        )
        
        # Reshape predictions back to per-variant, per-augmentation
        pred_idx = 0
        preds: List[float] = []
        for ref_seqs_list, alt_seqs_list in zip(variant_ref_seqs, variant_alt_seqs):
            n_aug = len(ref_seqs_list)
            variant_ref_preds = ref_preds_flat[pred_idx:pred_idx + n_aug]
            variant_alt_preds = alt_preds_flat[pred_idx:pred_idx + n_aug]
            pred_idx += n_aug
            
            # Average predictions over augmentations and calculate variant effect
            mean_ref = float(np.mean(variant_ref_preds))
            mean_alt = float(np.mean(variant_alt_preds))
            preds.append(mean_alt - mean_ref)
        
        preds = np.array(preds)

        all_preds.extend(preds.tolist())
        all_targets.extend(values)
        all_conf.extend(confidences)

        pearson_el, spearman_el = compute_correlations(preds, np.asarray(values, dtype=np.float32))
        
        # Compute high-confidence metrics for this element
        conf_arr = np.asarray(confidences, dtype=np.float32)
        mask_high_el = conf_arr > 0.1
        pearson_el_hi, spearman_el_hi = compute_correlations(
            preds[mask_high_el] if mask_high_el.sum() > 0 else np.array([]),
            np.asarray(values, dtype=np.float32)[mask_high_el] if mask_high_el.sum() > 0 else np.array([]),
        )
        
        element_results.append({
            'element': element,
            'n_variants': len(preds),
            'n_high_conf': int(mask_high_el.sum()),
            'pearson_all': pearson_el,
            'spearman_all': spearman_el,
            'pearson_high_conf': pearson_el_hi,
            'spearman_high_conf': spearman_el_hi,
        })
        
        print(
            f"  {element}: n={len(preds):5d}, "
            f"Pearson={pearson_el: .4f}, Spearman={spearman_el: .4f}"
        )

    all_preds_arr = np.asarray(all_preds, dtype=np.float32)
    all_targets_arr = np.asarray(all_targets, dtype=np.float32)
    all_conf_arr = np.asarray(all_conf, dtype=np.float32)

    print("\n" + "=" * 80)
    print("CAGI5 Zero-shot Summary (average across elements)")
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
    
    # Extract checkpoint name for filename
    checkpoint_name = Path(args.checkpoint_dir).name
    
    # Determine filename prefix based on augmentations
    prefix_parts = []
    
    # Augmentation prefixes
    if args.use_position_shift:
        prefix_parts.append(f"posshift{args.max_shift}_n{args.n_shift_samples}")
    if args.use_reverse_complement:
        prefix_parts.append("revcomp")
    
    # Base prefix
    if prefix_parts:
        file_prefix = "_".join(prefix_parts) + f"_cagi5_finetuned_{args.cell_type}_{checkpoint_name}"
    else:
        file_prefix = f"cagi5_finetuned_{args.cell_type}_{checkpoint_name}"
    
    # Save summary results
    summary_results = {
        'model': 'finetuned_mpra',
        'cell_type': args.cell_type,
        'checkpoint': checkpoint_name,
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

