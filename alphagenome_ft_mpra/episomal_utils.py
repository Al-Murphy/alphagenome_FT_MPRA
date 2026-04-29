"""
Shared helpers for the episomal MPRA (Gosai et al. 2024) dataset.

Pure-pandas/numpy: no JAX, no PyTorch. Both ``data.py`` (the JAX dataset
class ``EpisomalMPRADataset``) and ``enf_utils.py`` (the PyTorch
``EpisomalMPRADatasetPyTorch``) import from here so the two paths cannot
drift on data loading, padding, or one-hot encoding.

This is a dataset-specific counterpart to ``seq_loader.py``: kept small and
import-light so the PyTorch-only Enformer training path stays JAX-free.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

# Chromosome-based split definitions (Gosai paper conventions).
TEST_CHROMOSOMES = {"chr7", "chr13"}
VAL_CHROMOSOMES = {"chr19", "chr21", "chrX"}

# Per-cell label column in the supplementary TSV.
CELL_TYPE_LABEL_COLUMNS = {
    "K562": "K562_log2FC",
    "HepG2": "HepG2_log2FC",
    "SKNSH": "SKNSH_log2FC",
}
VALID_CELL_TYPES = list(CELL_TYPE_LABEL_COLUMNS.keys())
VALID_SPLITS = ["train", "val", "test"]

# Filename of the main supplementary table (Nature Genetics).
DATA_FILENAME = "DATA-Table_S2__MPRA_dataset.txt"

# All Gosai sequences are 200 bp.
SEQUENCE_LENGTH = 200


# ── Public helpers ───────────────────────────────────────────────────────────


def _parse_chromosome(id_str: str) -> Optional[str]:
    """Extract chromosome from Gosai ID format ``chr:pos:ref:alt:type:wc``."""
    parts = str(id_str).split(":")
    if len(parts) >= 1:
        chrom = parts[0]
        if chrom.startswith("chr"):
            return chrom
    return None


def _one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence to a ``(seq_len, 4)`` float32 array.

    Encoding: A=0, C=1, G=2, T=3. ``N`` (and any other character) maps to
    ``[0.25, 0.25, 0.25, 0.25]``.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = seq.upper()
    ohe = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            ohe[i, mapping[base]] = 1.0
        else:
            ohe[i, :] = 0.25
    return ohe


def _reverse_complement_ohe(ohe: np.ndarray) -> np.ndarray:
    """Reverse complement a one-hot encoded sequence.

    Reverses the sequence axis and swaps A↔T, C↔G channels.
    """
    return ohe[::-1, [3, 2, 1, 0]].copy()


def pad_n_bases(seq: str, n: int) -> str:
    """Add ``n`` ``'N'`` flanking bases around ``seq``.

    For odd ``n`` the extra base goes on the right. This is the single
    source of truth for padding so training datasets and the inference
    pipeline (``test_episomal_mpra.py``) cannot drift to give different
    sequence lengths for the same ``pad_n_bases`` setting.
    """
    if n <= 0:
        return seq
    left = n // 2
    right = n - left
    return "N" * left + seq + "N" * right


def _load_gosai_data(
    data_path: str,
    cell_type: str,
    split: str,
) -> pd.DataFrame:
    """Load and filter Gosai episomal MPRA data for a cell type and split.

    Args:
        data_path: Directory containing ``DATA-Table_S2__MPRA_dataset.txt``.
        cell_type: One of ``K562``, ``HepG2``, ``SKNSH``.
        split: One of ``train``, ``val``, ``test``.

    Returns:
        DataFrame with columns ``sequence``, ``label``, ``chromosome``, ``id``.
    """
    assert cell_type in VALID_CELL_TYPES, f"cell_type must be one of {VALID_CELL_TYPES}"
    assert split in VALID_SPLITS, f"split must be one of {VALID_SPLITS}"

    filepath = os.path.join(data_path, DATA_FILENAME)
    assert os.path.exists(filepath), f"Data file not found: {filepath}"

    label_col = CELL_TYPE_LABEL_COLUMNS[cell_type]

    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    # Resolve chromosome: prefer a dedicated 'chr' column (Gosai schema), fall
    # back to parsing the leading token of the IDs column for archives that
    # ship only the IDs.
    if "chr" in df.columns:
        df["chromosome"] = df["chr"].astype(str).apply(
            lambda c: c if c.startswith("chr") else f"chr{c}"
        )
    elif "IDs" in df.columns:
        df["chromosome"] = df["IDs"].apply(_parse_chromosome)
    else:
        raise ValueError(
            "Gosai TSV must have either a 'chr' column or parseable 'IDs'"
        )

    df = df.dropna(subset=["chromosome", label_col, "sequence"])
    df = df[df["sequence"].str.len() >= 198]

    if split == "test":
        df = df[df["chromosome"].isin(TEST_CHROMOSOMES)]
    elif split == "val":
        df = df[df["chromosome"].isin(VAL_CHROMOSOMES)]
    else:  # train
        exclude = TEST_CHROMOSOMES | VAL_CHROMOSOMES
        df = df[~df["chromosome"].isin(exclude)]

    return pd.DataFrame({
        "sequence": df["sequence"].values,
        "label": df[label_col].values.astype(np.float32),
        "chromosome": df["chromosome"].values,
        "id": df["IDs"].values if "IDs" in df.columns else range(len(df)),
    }).reset_index(drop=True)


def get_episomal_test_sets(
    data_path: str,
    cell_type: str = "K562",
) -> dict:
    """Return all episomal MPRA test sets available under ``data_path``.

    Always returns a ``reference`` entry (the chr7+chr13 genomic test split).
    Optionally returns ``designed`` and ``snv`` entries when the matching
    files exist under ``data_path/test_sets/``.

    Output schema:
      * ``reference``  : ``{"sequences": [...], "labels": ndarray}``
      * ``designed``   : ``{"sequences": [...], "labels": ndarray}``
      * ``snv``        : ``{"ref_sequences": [...], "alt_sequences": [...],
                           "true_delta": ndarray}``
    """
    test_sets: dict = {}

    # 1. Genomic reference (chr7/13 test sequences).
    ref_data = _load_gosai_data(data_path, cell_type, "test")
    if len(ref_data) > 0:
        test_sets["reference"] = {
            "sequences": ref_data["sequence"].tolist(),
            "labels": ref_data["label"].values,
        }

    # 2. High-activity designed sequences (OOD). Per-cell file preferred,
    # K562 file as a fallback for older drops that only shipped the K562 set.
    cell_lower = cell_type.lower()
    candidate_paths = [
        os.path.join(data_path, "test_sets", f"test_ood_designed_{cell_lower}.tsv"),
        os.path.join(data_path, "test_sets", "test_ood_designed_k562.tsv"),
    ]
    designed_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if designed_path:
        designed_df = pd.read_csv(designed_path, sep="\t")
        label_col = CELL_TYPE_LABEL_COLUMNS.get(cell_type, "K562_log2FC")
        if label_col in designed_df.columns:
            test_sets["designed"] = {
                "sequences": designed_df["sequence"].str[:SEQUENCE_LENGTH].tolist(),
                "labels": designed_df[label_col].values.astype(np.float32),
            }

    # 3. SNV pairs.
    snv_path = os.path.join(data_path, "test_sets", "test_snv_pairs_hashfrag.tsv")
    if os.path.exists(snv_path):
        snv_df = pd.read_csv(snv_path, sep="\t")
        label_col = CELL_TYPE_LABEL_COLUMNS.get(cell_type, "K562_log2FC")
        ref_col = f"{label_col.replace('_log2FC', '')}_log2FC_ref"
        alt_col = f"{label_col.replace('_log2FC', '')}_log2FC_alt"

        if ref_col not in snv_df.columns:  # legacy column naming fallback
            ref_col = "K562_log2FC_ref"
            alt_col = "K562_log2FC_alt"

        if ref_col in snv_df.columns and alt_col in snv_df.columns:
            test_sets["snv"] = {
                "ref_sequences": snv_df["sequence_ref"].str[:SEQUENCE_LENGTH].tolist(),
                "alt_sequences": snv_df["sequence_alt"].str[:SEQUENCE_LENGTH].tolist(),
                "true_delta": (snv_df[alt_col] - snv_df[ref_col]).values.astype(np.float32),
            }

    return test_sets


def standardize_to_sequence_length(seq: str) -> str:
    """Center-crop / center-pad a sequence so it ends up exactly
    ``SEQUENCE_LENGTH`` bp. Keeps train and inference paths aligned."""
    if len(seq) < SEQUENCE_LENGTH:
        diff = SEQUENCE_LENGTH - len(seq)
        return "N" * (diff // 2) + seq + "N" * (diff - diff // 2)
    if len(seq) > SEQUENCE_LENGTH:
        start = (len(seq) - SEQUENCE_LENGTH) // 2
        return seq[start:start + SEQUENCE_LENGTH]
    return seq
