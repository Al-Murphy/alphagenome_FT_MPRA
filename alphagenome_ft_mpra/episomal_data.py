"""
Dataset classes for episomal MPRA (Gosai et al. 2024) finetuning.

The Gosai dataset contains ~800K 200bp sequences across 3 cell types (K562,
HepG2, SK-N-SH); splits are chromosome-based (test=chr7+chr13,
val=chr19+chr21+chrX, train=remainder). EpisomalMPRADataset is JAX-based for
AlphaGenome; EpisomalMPRADatasetPyTorch is the numpy version for
Enformer/LegNet/DREAM-RNN.
"""

import os
import numpy as np
import pandas as pd
from typing import Any

# Chromosome-based split definitions
TEST_CHROMOSOMES = {"chr7", "chr13"}
VAL_CHROMOSOMES = {"chr19", "chr21", "chrX"}

CELL_TYPE_LABEL_COLUMNS = {
    "K562": "K562_log2FC",
    "HepG2": "HepG2_log2FC",
    "SKNSH": "SKNSH_log2FC",
}

VALID_CELL_TYPES = list(CELL_TYPE_LABEL_COLUMNS.keys())
VALID_SPLITS = ["train", "val", "test"]

# Data file name
DATA_FILENAME = "DATA-Table_S2__MPRA_dataset.txt"

# Sequence length
SEQUENCE_LENGTH = 200


def _parse_chromosome(id_str: str) -> str | None:
    """Extract chromosome from Gosai ID format (chr:pos:ref:alt:type:wc)."""
    parts = str(id_str).split(":")
    if len(parts) >= 1:
        chrom = parts[0]
        if chrom.startswith("chr"):
            return chrom
    return None


def _one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence to (seq_len, 4) array.

    Encoding: A=0, C=1, G=2, T=3. N maps to [0.25, 0.25, 0.25, 0.25].
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = seq.upper()
    ohe = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            ohe[i, mapping[base]] = 1.0
        else:
            ohe[i, :] = 0.25  # N or ambiguous
    return ohe


def _reverse_complement_ohe(ohe: np.ndarray) -> np.ndarray:
    """Reverse complement a one-hot encoded sequence.

    Reverses sequence order and swaps A↔T, C↔G.
    """
    # Reverse along sequence axis, then swap channels: A(0)↔T(3), C(1)↔G(2)
    return ohe[::-1, [3, 2, 1, 0]].copy()


def _load_gosai_data(
    data_path: str,
    cell_type: str,
    split: str,
) -> pd.DataFrame:
    """Load and filter Gosai episomal MPRA data for a specific cell type and split.

    Args:
        data_path: Directory containing DATA-Table_S2__MPRA_dataset.txt
        cell_type: One of K562, HepG2, SKNSH
        split: One of train, val, test

    Returns:
        DataFrame with columns: sequence, label, chromosome
    """
    assert cell_type in VALID_CELL_TYPES, f"cell_type must be one of {VALID_CELL_TYPES}"
    assert split in VALID_SPLITS, f"split must be one of {VALID_SPLITS}"

    filepath = os.path.join(data_path, DATA_FILENAME)
    assert os.path.exists(filepath), f"Data file not found: {filepath}"

    label_col = CELL_TYPE_LABEL_COLUMNS[cell_type]

    # Load data
    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    # Resolve chromosome: prefer a dedicated 'chr' column (Gosai schema), and
    # fall back to parsing the leading token of the IDs column for archives
    # that ship only the IDs.
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

    # Filter: must have valid chromosome and non-null label
    df = df.dropna(subset=["chromosome", label_col, "sequence"])

    # Filter by sequence length (must be >= 198bp)
    df = df[df["sequence"].str.len() >= 198]

    # Chromosome-based split
    if split == "test":
        df = df[df["chromosome"].isin(TEST_CHROMOSOMES)]
    elif split == "val":
        df = df[df["chromosome"].isin(VAL_CHROMOSOMES)]
    else:  # train
        exclude = TEST_CHROMOSOMES | VAL_CHROMOSOMES
        df = df[~df["chromosome"].isin(exclude)]

    # Build output DataFrame
    result = pd.DataFrame({
        "sequence": df["sequence"].values,
        "label": df[label_col].values.astype(np.float32),
        "chromosome": df["chromosome"].values,
        "id": df["IDs"].values if "IDs" in df.columns else range(len(df)),
    })
    result = result.reset_index(drop=True)

    return result


class EpisomalMPRADatasetPyTorch:
    """PyTorch/numpy dataset for episomal MPRA (Gosai et al. 2024).

    Compatible with Enformer, LegNet, DREAM-RNN, and Malinois training.
    Returns numpy arrays that PyTorch DataLoaders can convert to tensors.

    Usage:
        dataset = EpisomalMPRADatasetPyTorch(
            path_to_data='./data/gosai_episomal',
            cell_type='K562',
            split='train',
            reverse_complement=True,
        )
        sample = dataset[0]  # {'seq': ndarray(200, 4), 'y': float}
    """

    def __init__(
        self,
        model=None,  # Optional, for API compatibility with LentiMPRADatasetPyTorch
        path_to_data: str = "./data/gosai_episomal",
        cell_type: str = "K562",
        split: str = "train",
        reverse_complement: bool = False,
        reverse_complement_likelihood: float = 0.5,
        random_shift: bool = False,
        random_shift_likelihood: float = 0.5,
        max_shift: int = 10,
        pad_n_bases: int = 0,
        subset_frac: float = 1.0,
        seed: int = 42,
    ):
        assert cell_type in VALID_CELL_TYPES, f"cell_type must be one of {VALID_CELL_TYPES}"
        assert split in VALID_SPLITS, f"split must be one of {VALID_SPLITS}"

        self.cell_type = cell_type
        self.split = split
        self.reverse_complement = reverse_complement
        self.reverse_complement_likelihood = reverse_complement_likelihood
        self.random_shift = random_shift
        self.random_shift_likelihood = random_shift_likelihood
        self.max_shift = max_shift
        self.pad_n_bases = pad_n_bases
        self.rng = np.random.RandomState(seed)

        # Load data
        self.data = _load_gosai_data(path_to_data, cell_type, split)

        if subset_frac < 1.0:
            n = int(len(self.data) * subset_frac)
            self.data = self.data.sample(n=n, random_state=self.rng).reset_index(drop=True)

        print(f"Loaded {len(self.data)} episomal MPRA samples for {cell_type} {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = str(row["sequence"])
        label = float(row["label"])

        # Standardize to SEQUENCE_LENGTH
        if len(seq) < SEQUENCE_LENGTH:
            pad = SEQUENCE_LENGTH - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > SEQUENCE_LENGTH:
            start = (len(seq) - SEQUENCE_LENGTH) // 2
            seq = seq[start:start + SEQUENCE_LENGTH]

        # Optional N-padding (asymmetric for odd totals so the full pad_n_bases
        # is added, matching the call signature in test_episomal_mpra.py).
        if self.pad_n_bases > 0:
            left = self.pad_n_bases // 2
            right = self.pad_n_bases - left
            seq = "N" * left + seq + "N" * right

        # One-hot encode
        ohe = _one_hot_encode(seq)

        # Random shift augmentation
        if self.random_shift and self.rng.random() < self.random_shift_likelihood:
            shift = self.rng.randint(-self.max_shift, self.max_shift + 1)
            if shift != 0:
                ohe = np.roll(ohe, shift, axis=0)
                if shift > 0:
                    ohe[:shift, :] = 0.25  # Fill with N
                else:
                    ohe[shift:, :] = 0.25

        # Reverse complement augmentation
        if self.reverse_complement and self.rng.random() < self.reverse_complement_likelihood:
            ohe = _reverse_complement_ohe(ohe)

        return {"seq": ohe, "y": label}


class EpisomalMPRADataset:
    """JAX dataset for episomal MPRA (Gosai et al. 2024).

    Compatible with AlphaGenome training via alphagenome_ft.
    Returns JAX arrays matching the LentiMPRADataset interface.

    Usage:
        dataset = EpisomalMPRADataset(
            model=ag_model,
            path_to_data='./data/gosai_episomal',
            cell_type='K562',
            split='train',
        )
        sample = dataset[0]  # {'seq': jax array, 'y': scalar, 'organism_index': jax array}
    """

    def __init__(
        self,
        model: Any,  # AlphaGenome model with _one_hot_encoder
        path_to_data: str = "./data/gosai_episomal",
        cell_type: str = "K562",
        split: str = "train",
        random_shift: bool = False,
        random_shift_likelihood: float = 0.5,
        max_shift: int = 10,
        reverse_complement: bool = False,
        reverse_complement_likelihood: float = 0.5,
        pad_n_bases: int = 0,
        subset_frac: float = 1.0,
        rng_key=None,
        use_cached_embeddings: bool = False,
        cache_file: str | None = None,
    ):
        import jax
        import jax.numpy as jnp
        from alphagenome_research.model import dna_model

        assert cell_type in VALID_CELL_TYPES
        assert split in VALID_SPLITS

        self.model = model
        self.cell_type = cell_type
        self.split = split
        self.organism = dna_model.Organism.HOMO_SAPIENS
        self.pad_n_bases = pad_n_bases

        if rng_key is None:
            self.rng_key = jax.random.PRNGKey(42)
        else:
            self.rng_key = rng_key

        # Augmentations
        self.reverse_complement = reverse_complement
        self.reverse_complement_likelihood = reverse_complement_likelihood
        self.random_shift = random_shift
        self.random_shift_likelihood = random_shift_likelihood
        self.max_shift = max_shift

        # Load data
        self.data = _load_gosai_data(path_to_data, cell_type, split)

        if subset_frac < 1.0:
            n = int(len(self.data) * subset_frac)
            self.data = self.data.sample(n=n).reset_index(drop=True)

        print(f"Loaded {len(self.data)} episomal MPRA samples for {cell_type} {split}")

        # Cached embeddings
        self.use_cached_embeddings = use_cached_embeddings
        if use_cached_embeddings:
            if random_shift or reverse_complement:
                print("Warning: Augmentations disabled when using cached embeddings")
                self.random_shift = False
                self.reverse_complement = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import jax
        import jax.numpy as jnp

        row = self.data.iloc[idx]
        seq = str(row["sequence"])
        label = float(row["label"])

        # Standardize to SEQUENCE_LENGTH
        if len(seq) < SEQUENCE_LENGTH:
            pad = SEQUENCE_LENGTH - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > SEQUENCE_LENGTH:
            start = (len(seq) - SEQUENCE_LENGTH) // 2
            seq = seq[start:start + SEQUENCE_LENGTH]

        # Optional N-padding (asymmetric for odd totals so the full pad_n_bases
        # is added, matching the call signature in test_episomal_mpra.py).
        if self.pad_n_bases > 0:
            left = self.pad_n_bases // 2
            right = self.pad_n_bases - left
            seq = "N" * left + seq + "N" * right

        # Random shift
        if self.random_shift:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            if jax.random.uniform(subkey) < self.random_shift_likelihood:
                self.rng_key, subkey2 = jax.random.split(self.rng_key)
                shift = int(jax.random.randint(subkey2, (), -self.max_shift, self.max_shift + 1))
                if shift > 0:
                    seq = "N" * shift + seq[:-shift]
                elif shift < 0:
                    seq = seq[-shift:] + "N" * (-shift)

        # Reverse complement
        if self.reverse_complement:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            if jax.random.uniform(subkey) < self.reverse_complement_likelihood:
                comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
                seq = "".join(comp.get(b, "N") for b in reversed(seq.upper()))

        # Encode using AG one-hot encoder (matches LentiMPRADataset call signature).
        ohe = self.model._one_hot_encoder.encode(seq)

        return {
            "seq": ohe,
            "y": label,
            "organism_index": jnp.array([0]),
        }


# ── Test set helpers ──────────────────────────────────────────────────


def get_episomal_test_sets(
    data_path: str,
    cell_type: str = "K562",
) -> dict:
    """Get all episomal MPRA test sets for evaluation.

    Returns:
        Dict mapping test set name to (sequences, labels) tuple.
        For SNV: returns (ref_seqs, alt_seqs, ref_labels, alt_labels, true_delta).
    """
    test_sets = {}

    # 1. Genomic reference (chr7/13 test sequences)
    ref_data = _load_gosai_data(data_path, cell_type, "test")
    if len(ref_data) > 0:
        test_sets["reference"] = {
            "sequences": ref_data["sequence"].tolist(),
            "labels": ref_data["label"].values,
        }

    # 2. High-activity designed sequences (OOD)
    # These are loaded from a separate test set file if available
    designed_path = os.path.join(data_path, "test_sets", "test_ood_designed_k562.tsv")
    if os.path.exists(designed_path):
        designed_df = pd.read_csv(designed_path, sep="\t")
        label_col = CELL_TYPE_LABEL_COLUMNS.get(cell_type, "K562_log2FC")
        if label_col in designed_df.columns:
            test_sets["designed"] = {
                "sequences": designed_df["sequence"].str[:SEQUENCE_LENGTH].tolist(),
                "labels": designed_df[label_col].values.astype(np.float32),
            }

    # 3. SNV pairs
    snv_path = os.path.join(data_path, "test_sets", "test_snv_pairs_hashfrag.tsv")
    if os.path.exists(snv_path):
        snv_df = pd.read_csv(snv_path, sep="\t")
        label_col = CELL_TYPE_LABEL_COLUMNS.get(cell_type, "K562_log2FC")
        ref_col = f"{label_col.replace('_log2FC', '')}_log2FC_ref"
        alt_col = f"{label_col.replace('_log2FC', '')}_log2FC_alt"

        # Flexible column naming
        if ref_col not in snv_df.columns:
            ref_col = "K562_log2FC_ref"
            alt_col = "K562_log2FC_alt"

        if ref_col in snv_df.columns and alt_col in snv_df.columns:
            test_sets["snv"] = {
                "ref_sequences": snv_df["sequence_ref"].str[:SEQUENCE_LENGTH].tolist(),
                "alt_sequences": snv_df["sequence_alt"].str[:SEQUENCE_LENGTH].tolist(),
                "true_delta": (snv_df[alt_col] - snv_df[ref_col]).values.astype(np.float32),
            }

    return test_sets
