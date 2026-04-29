"""Tests for the episomal MPRA (Gosai 2024) dataset module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from alphagenome_ft_mpra.enf_utils import EpisomalMPRADatasetPyTorch
from alphagenome_ft_mpra.episomal_utils import (
    DATA_FILENAME,
    SEQUENCE_LENGTH,
    TEST_CHROMOSOMES,
    VAL_CHROMOSOMES,
    _load_gosai_data,
    _one_hot_encode,
    _parse_chromosome,
    _reverse_complement_ohe,
    get_episomal_test_sets,
    pad_n_bases,
)


def _write_synthetic_gosai(tmp: str, n_per_chrom: int = 10, seed: int = 0) -> str:
    """Write a synthetic Gosai-format TSV under ``tmp`` and return its path."""
    rng = np.random.default_rng(seed)
    chroms = ["chr1", "chr2", "chr3", "chr19", "chr21", "chrX", "chr7", "chr13"]
    rows = []
    for i in range(n_per_chrom * len(chroms)):
        chrom = chroms[i % len(chroms)]
        seq = "".join(rng.choice(["A", "C", "G", "T"], size=SEQUENCE_LENGTH))
        rows.append({
            "IDs": f"{chrom}:{1000 + i}:A:T:ref:wc",
            "sequence": seq,
            "K562_log2FC": float(rng.normal()),
            "HepG2_log2FC": float(rng.normal()),
            "SKNSH_log2FC": float(rng.normal()),
        })
    path = os.path.join(tmp, DATA_FILENAME)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_one_hot_encode_acgtn():
    ohe = _one_hot_encode("ACGTN")
    assert ohe.shape == (5, 4)
    assert ohe.dtype == np.float32
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
         [0.25, 0.25, 0.25, 0.25]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(ohe, expected)


def test_reverse_complement_ohe():
    ohe = _one_hot_encode("ACGTN")
    rc = _reverse_complement_ohe(ohe)
    assert rc.shape == ohe.shape
    # ACGTN reverse complement: NACGT
    expected = _one_hot_encode("NACGT")
    np.testing.assert_allclose(rc, expected)


def test_parse_chromosome():
    assert _parse_chromosome("chr7:12345:A:T:ref:wc") == "chr7"
    assert _parse_chromosome("chr13:5:G:C:ref:1") == "chr13"
    assert _parse_chromosome("garbage") is None
    assert _parse_chromosome("chrX:1:N:N:ref:wc") == "chrX"


@pytest.mark.parametrize("cell_type", ["K562", "HepG2", "SKNSH"])
def test_chromosome_split_sizes(cell_type):
    """Each split (train/val/test) holds the expected chromosomes."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_gosai(tmp, n_per_chrom=10)
        train = _load_gosai_data(tmp, cell_type, "train")
        val = _load_gosai_data(tmp, cell_type, "val")
        test = _load_gosai_data(tmp, cell_type, "test")

        # 8 distinct chroms, 10 rows each: train=3 chr × 10, val=3 chr × 10, test=2 chr × 10
        assert len(train) == 30
        assert len(val) == 30
        assert len(test) == 20

        # Splits must respect the chromosome partition
        assert set(test["chromosome"].unique()) <= TEST_CHROMOSOMES
        assert set(val["chromosome"].unique()) <= VAL_CHROMOSOMES
        assert not (set(train["chromosome"].unique()) & TEST_CHROMOSOMES)
        assert not (set(train["chromosome"].unique()) & VAL_CHROMOSOMES)


def test_pad_n_bases_helper_returns_full_total():
    """The shared pad_n_bases() helper must add exactly n bases for any n."""
    seq = "A" * 50
    for n in (0, 1, 2, 80, 81, 82):
        out = pad_n_bases(seq, n)
        assert len(out) == 50 + n, f"pad_n_bases(.., {n}) → len {len(out)}"
        # Padding is N's only
        if n > 0:
            assert out[: n // 2] == "N" * (n // 2)
            assert out[len(out) - (n - n // 2):] == "N" * (n - n // 2)
        # Original sequence is intact in the middle
        left = n // 2
        assert out[left:left + len(seq)] == seq


def test_reverse_complement_is_involution():
    """RC(RC(x)) == x — guards the channel-swap and reversal logic."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), size=200))
    ohe = _one_hot_encode(seq)
    np.testing.assert_allclose(_reverse_complement_ohe(_reverse_complement_ohe(ohe)), ohe)


def test_splits_are_disjoint_by_chromosome():
    """Train / val / test chromosomes must not overlap."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_gosai(tmp, n_per_chrom=10)
        chr_train = set(_load_gosai_data(tmp, "K562", "train")["chromosome"].unique())
        chr_val = set(_load_gosai_data(tmp, "K562", "val")["chromosome"].unique())
        chr_test = set(_load_gosai_data(tmp, "K562", "test")["chromosome"].unique())
        assert chr_train & chr_val == set()
        assert chr_train & chr_test == set()
        assert chr_val & chr_test == set()


def test_same_sequences_across_cells():
    """All 3 cell types must see the same sequences in each split — only the
    label column differs. Catches regressions where cell-specific filtering
    (e.g. dropping rows with null cell labels) would diverge across cells."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_gosai(tmp, n_per_chrom=10)
        for split in ("train", "val", "test"):
            k = set(_load_gosai_data(tmp, "K562", split)["sequence"])
            h = set(_load_gosai_data(tmp, "HepG2", split)["sequence"])
            s = set(_load_gosai_data(tmp, "SKNSH", split)["sequence"])
            assert k == h == s, f"{split} differs across cells"


def test_pad_n_bases_full_total_for_odd_padding():
    """Padding by an odd amount (e.g. 81 → 200bp + 81bp = 281bp) must add
    the full pad_n_bases, asymmetrically split. Earlier code split as
    `p = pad // 2` on each side, which dropped 1 base for odd pads."""
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        rows = [{
            "IDs": "chr1:1:A:T:ref:wc",
            "sequence": "".join(rng.choice(list("ACGT"), size=SEQUENCE_LENGTH)),
            "K562_log2FC": 0.0, "HepG2_log2FC": 0.0, "SKNSH_log2FC": 0.0,
        }]
        pd.DataFrame(rows).to_csv(
            os.path.join(tmp, DATA_FILENAME), sep="\t", index=False,
        )

        for pad in (0, 80, 81, 82):
            ds = EpisomalMPRADatasetPyTorch(
                path_to_data=tmp,
                cell_type="K562",
                split="train",
                pad_n_bases=pad,
            )
            sample = ds[0]
            assert sample["seq"].shape == (SEQUENCE_LENGTH + pad, 4), \
                f"pad={pad}: expected length {SEQUENCE_LENGTH + pad}, got {sample['seq'].shape[0]}"


def test_chr_column_takes_precedence_over_ids():
    """Real Gosai TSV ships a dedicated 'chr' column with bare digits and IDs
    in 'chr:pos:ref:alt:wc' form (5th token is the alt allele, not an
    allele-type tag). _load_gosai_data must prefer the 'chr' column and
    prepend 'chr' as needed, rather than parsing IDs."""
    with tempfile.TemporaryDirectory() as tmp:
        # Bare-digit chr values (matches real Gosai TSV); IDs without 'chr' prefix
        rows = [
            {"IDs": "7:100:G:T:wC", "chr": "7", "sequence": "A" * 200,
             "K562_log2FC": 0.5, "HepG2_log2FC": 0.0, "SKNSH_log2FC": 0.0},
            {"IDs": "13:200:G:T:wC", "chr": "13", "sequence": "C" * 200,
             "K562_log2FC": 0.6, "HepG2_log2FC": 0.0, "SKNSH_log2FC": 0.0},
            {"IDs": "1:300:G:T:wC", "chr": "1", "sequence": "G" * 200,
             "K562_log2FC": 0.7, "HepG2_log2FC": 0.0, "SKNSH_log2FC": 0.0},
            {"IDs": "X:400:G:T:wC", "chr": "X", "sequence": "T" * 200,
             "K562_log2FC": 0.8, "HepG2_log2FC": 0.0, "SKNSH_log2FC": 0.0},
        ]
        pd.DataFrame(rows).to_csv(
            os.path.join(tmp, DATA_FILENAME), sep="\t", index=False
        )

        train = _load_gosai_data(tmp, "K562", "train")
        val = _load_gosai_data(tmp, "K562", "val")
        test = _load_gosai_data(tmp, "K562", "test")

        # chr 7 + chr 13 → test; chr X → val; chr 1 → train
        assert set(test["chromosome"].unique()) == {"chr7", "chr13"}
        assert set(val["chromosome"].unique()) == {"chrX"}
        assert set(train["chromosome"].unique()) == {"chr1"}
        # Chromosomes were inferred from the 'chr' column, NOT by parsing IDs
        # (which lack the 'chr' prefix) — so no rows would have survived if
        # the IDs path were taken instead.
        assert len(train) == 1
        assert len(val) == 1
        assert len(test) == 2


def test_pytorch_dataset_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_gosai(tmp, n_per_chrom=10)
        ds = EpisomalMPRADatasetPyTorch(
            path_to_data=tmp,
            cell_type="K562",
            split="train",
            reverse_complement=True,
            random_shift=True,
            max_shift=10,
            seed=0,
        )
        assert len(ds) == 30

        sample = ds[0]
        assert sample["seq"].shape == (SEQUENCE_LENGTH, 4)
        assert sample["seq"].dtype == np.float32
        # One-hot row sums must equal 1 (or 4 × 0.25 for N) at every position
        np.testing.assert_allclose(sample["seq"].sum(axis=1), 1.0)
        assert isinstance(sample["y"], float)


def test_get_episomal_test_sets_reference_only():
    """Without designed/SNV files, only the genomic reference set is returned."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_gosai(tmp, n_per_chrom=10)
        sets = get_episomal_test_sets(tmp, "K562")
        assert "reference" in sets
        assert sets["reference"]["labels"].shape == (20,)
        assert len(sets["reference"]["sequences"]) == 20
        assert "designed" not in sets
        assert "snv" not in sets


def test_get_episomal_test_sets_with_snv():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_gosai(tmp, n_per_chrom=10)
        snv_dir = os.path.join(tmp, "test_sets")
        os.makedirs(snv_dir)
        rng = np.random.default_rng(1)
        snv_rows = []
        for _ in range(15):
            ref = "".join(rng.choice(["A", "C", "G", "T"], size=SEQUENCE_LENGTH))
            alt = ref[:100] + ("A" if ref[100] != "A" else "C") + ref[101:]
            snv_rows.append({
                "sequence_ref": ref,
                "sequence_alt": alt,
                "K562_log2FC_ref": float(rng.normal()),
                "K562_log2FC_alt": float(rng.normal()),
            })
        pd.DataFrame(snv_rows).to_csv(
            os.path.join(snv_dir, "test_snv_pairs_hashfrag.tsv"),
            sep="\t",
            index=False,
        )

        sets = get_episomal_test_sets(tmp, "K562")
        assert "snv" in sets
        assert len(sets["snv"]["ref_sequences"]) == 15
        assert len(sets["snv"]["alt_sequences"]) == 15
        assert sets["snv"]["true_delta"].shape == (15,)
