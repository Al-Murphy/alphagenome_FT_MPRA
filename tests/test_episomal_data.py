"""Tests for the episomal MPRA (Gosai 2024) dataset module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from alphagenome_ft_mpra.episomal_data import (
    DATA_FILENAME,
    SEQUENCE_LENGTH,
    TEST_CHROMOSOMES,
    VAL_CHROMOSOMES,
    EpisomalMPRADatasetPyTorch,
    _load_gosai_data,
    _one_hot_encode,
    _parse_chromosome,
    _reverse_complement_ohe,
    get_episomal_test_sets,
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
