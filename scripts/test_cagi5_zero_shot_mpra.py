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
- Computes Pearson r and Spearman ρ on:
    1) All SNPs
    2) High-confidence SNPs (Confidence > 0.1)
- Uses hg19 reference sequences and 1-based genomic coordinates.
- Sequence length matches training: 281 bp (full promoter construct).

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


PROMOTER_CONSTRUCT_LENGTH = 281  # same init_seq_len as finetune_mpra.py


# ---------------------------------------------------------------------------
# CAGI5 utilities (adapted from RankProject/scripts/evaluate_cagi5.py)
# ---------------------------------------------------------------------------

def load_cagi5_references(references_path: Path) -> Dict[str, dict]:
    """Load CAGI5 reference sequences (hg19) from JSON file."""
    if not references_path.exists():
        raise FileNotFoundError(
            f"Reference file not found: {references_path}\n"
            f"Run: python scripts/fetch_cagi5_references.py"
        )
    with open(references_path) as f:
        refs = json.load(f)
    return refs


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


def get_variant_sequence(
    ref_seq: str,
    ref_start: int,
    var_pos: int,
    ref_allele: str,
    alt_allele: str,
    window: int = 230,
) -> str | None:
    """Create variant sequence centered on the variant position (hg19, 1-based).

    Args:
        ref_seq: Full reference sequence for the element (hg19).
        ref_start: Genomic start position of ref_seq (1-based).
        var_pos: Genomic position of the variant (1-based).
        ref_allele: Reference allele from CAGI5 file.
        alt_allele: Alternate allele from CAGI5 file.
        window: Length of output sequence to generate.
    """
    # Convert to 0-based index within ref_seq
    idx = var_pos - ref_start
    if idx < 0 or idx >= len(ref_seq):
        return None

    # Best-effort reference allele check (do not hard-fail if mismatch)
    ref_in_seq = ref_seq[idx : idx + len(ref_allele)]
    if ref_in_seq.upper() != ref_allele.upper():
        # Keep going; CAGI5 files can contain edge-case indels.
        pass

    # Create variant sequence by substituting alt allele
    var_seq = ref_seq[:idx] + alt_allele + ref_seq[idx + len(ref_allele) :]

    # Extract window centered on variant
    center = idx + len(alt_allele) // 2
    half_window = window // 2
    start = center - half_window
    end = start + window

    if start < 0:
        pad_left = -start
        seq = "N" * pad_left + var_seq[: window - pad_left]
    elif end > len(var_seq):
        pad_right = end - len(var_seq)
        seq = var_seq[start:] + "N" * pad_right
    else:
        seq = var_seq[start:end]

    return seq[:window]


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
        "--references",
        type=str,
        default=None,
        help=(
            "Path to cagi5_references.json (hg19). "
            "Default: ./data/cagi5/cagi5_references.json."
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

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    repo_root = Path(__file__).parent.parent.resolve()

    # Default paths to local data directory if not provided explicitly
    if args.cagi5_dir is None:
        cagi5_dir = repo_root / "data" / "cagi5"
    else:
        cagi5_dir = Path(args.cagi5_dir).resolve()

    if args.references is None:
        references_path = repo_root / "data" / "cagi5" / "cagi5_references.json"
    else:
        references_path = Path(args.references).resolve()

    print("=" * 80)
    print("Zero-shot CAGI5 Evaluation for Fine-tuned AlphaGenome MPRA Models")
    print("=" * 80)
    print(f"Checkpoint dir:      {checkpoint_dir}")
    print(f"Cell type:           {args.cell_type}")
    print(f"CAGI5 dir:           {cagi5_dir}")
    print(f"References (hg19):   {references_path}")
    print(f"Sequence length:     {args.seq_len} bp")
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

    # Load references & CAGI5 data
    print("\nLoading CAGI5 reference sequences (hg19)...")
    references = load_cagi5_references(references_path)
    print(f"✓ Loaded {len(references)} reference elements")

    print("\nLoading CAGI5 variant tables (challenge/test set only)...")
    # Only load challenge_* files (held-out CAGI5 test set) and restrict
    # to elements whose experimental cell line matches the fine-tuned model.
    cagi5_data = parse_cagi5_dir(cagi5_dir, elements_to_include=allowed_elements)
    print(f"✓ Loaded CAGI5 data for {len(cagi5_data)} elements")

    # Aggregate predictions/targets across all elements
    all_preds: List[float] = []
    all_targets: List[float] = []
    all_conf: List[float] = []

    for element, df in cagi5_data.items():
        if element not in references:
            print(f"  {element}: no reference sequence in JSON, skipping")
            continue

        ref_info = references[element]
        ref_seq = ref_info["sequence"]
        ref_start = ref_info["start"]  # 1-based hg19 genomic start

        sequences: List[str] = []
        values: List[float] = []
        confidences: List[float] = []

        for _, row in df.iterrows():
            seq = get_variant_sequence(
                ref_seq=ref_seq,
                ref_start=ref_start,
                var_pos=int(row["Pos"]),
                ref_allele=str(row["Ref"]),
                alt_allele=str(row["Alt"]),
                window=args.seq_len,
            )
            if seq is None or len(seq) != args.seq_len:
                continue
            sequences.append(seq)
            values.append(float(row["Value"]))
            confidences.append(float(row["Confidence"]))

        if not sequences:
            print(f"  {element}: no valid sequences, skipping")
            continue

        preds = batch_predict_mpra(
            model=model,
            sequences=sequences,
            head_name="mpra_head",
            batch_size=args.batch_size,
        )

        all_preds.extend(preds.tolist())
        all_targets.extend(values)
        all_conf.extend(confidences)

        pearson_el, spearman_el = compute_correlations(preds, np.asarray(values, dtype=np.float32))
        print(
            f"  {element}: n={len(preds):5d}, "
            f"Pearson={pearson_el: .4f}, Spearman={spearman_el: .4f}"
        )

    all_preds_arr = np.asarray(all_preds, dtype=np.float32)
    all_targets_arr = np.asarray(all_targets, dtype=np.float32)
    all_conf_arr = np.asarray(all_conf, dtype=np.float32)

    print("\n" + "=" * 80)
    print("CAGI5 Zero-shot Summary (across all included elements)")
    print("=" * 80)

    # All SNPs
    pearson_all, spearman_all = compute_correlations(all_preds_arr, all_targets_arr)
    print(f"All SNPs (n={len(all_preds_arr)}):")
    print(f"  Pearson r:    {pearson_all: .4f}")
    print(f"  Spearman rho: {spearman_all: .4f}")

    # High-confidence SNPs (Confidence > 0.1)
    mask_high = all_conf_arr > 0.1
    pearson_hi, spearman_hi = compute_correlations(
        all_preds_arr[mask_high],
        all_targets_arr[mask_high],
    )
    print(f"\nHigh-confidence SNPs (Confidence > 0.1, n={int(mask_high.sum())}):")
    print(f"  Pearson r:    {pearson_hi: .4f}")
    print(f"  Spearman rho: {spearman_hi: .4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

