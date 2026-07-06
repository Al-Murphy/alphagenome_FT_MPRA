"""Tests for the plant STARR-seq (Jores 2021) dataset module.

Network-free: all tests build synthetic ``jores21_*`` TSVs in a temp dir and
exercise the pure-pandas/numpy helpers in ``plant_starrseq_utils``.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from alphagenome_ft_mpra.plant_starrseq_utils import (
    ENHANCER_153,
    PROMOTER_LENGTH,
    SEQUENCE_LENGTH,
    UTR_MAP,
    _load_plant_starrseq_data,
    _one_hot_encode,
    _reverse_complement_ohe,
    build_construct,
    build_sequence_for_mode,
    circular_shift_str,
    pad_or_trim_str,
    reverse_complement_str,
)

_ENH_TAG = {True: "35SEnh", False: "noEnh"}


def _rand_seq(rng, n):
    return "".join(rng.choice(list("ACGT"), size=n))


def _write_synthetic(tmp: str, n_train: int = 40, n_test: int = 10, seed: int = 0) -> None:
    """Write the 8 jores21 TSVs with disjoint train/test gene sets per tissue."""
    rng = np.random.default_rng(seed)
    for tissue in ("leaf", "proto"):
        for enh in (True, False):
            for part, n, offset in (("train", n_train, 0), ("test", n_test, 10_000)):
                rows = []
                for i in range(n):
                    sp = ["At", "Zm", "Sb"][i % 3]
                    rows.append({
                        "gene": f"{sp}_{part}_{offset + i}",
                        "sp": sp,
                        "type": "promoter",
                        "sequence": _rand_seq(rng, PROMOTER_LENGTH),
                        "enrichment": float(rng.normal()),
                    })
                path = os.path.join(tmp, f"jores21_{tissue}_{_ENH_TAG[enh]}_{part}.tsv")
                pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


# ── One-hot / string helpers ─────────────────────────────────────────────────


def test_one_hot_encode_acgtn():
    ohe = _one_hot_encode("ACGTN")
    assert ohe.shape == (5, 4)
    assert ohe.dtype == np.float32
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
         [0.25, 0.25, 0.25, 0.25]], dtype=np.float32,
    )
    np.testing.assert_allclose(ohe, expected)


def test_reverse_complement_ohe_involution():
    rng = np.random.default_rng(0)
    ohe = _one_hot_encode(_rand_seq(rng, 200))
    np.testing.assert_allclose(_reverse_complement_ohe(_reverse_complement_ohe(ohe)), ohe)


def test_reverse_complement_str():
    assert reverse_complement_str("ACGT") == "ACGT"
    assert reverse_complement_str("AAAA") == "TTTT"
    assert reverse_complement_str("ACGTN") == "NACGT"


def test_pad_or_trim_str():
    assert len(pad_or_trim_str("ACGT", 10)) == 10
    assert pad_or_trim_str("ACGT", 10) == "ACGT" + "N" * 6
    assert pad_or_trim_str("ACGTACGT", 4) == "ACGT"


def test_circular_shift_str():
    assert circular_shift_str("ABCDE", 0) == "ABCDE"
    assert circular_shift_str("ABCDE", 2) == "DEABC"


# ── Construct assembly ───────────────────────────────────────────────────────


def test_build_construct_uses_real_enhancer():
    rng = np.random.default_rng(0)
    promoter = "A" * PROMOTER_LENGTH
    with_enh = build_construct(promoter, "Zm", True, rng)
    # the real construct starts with the 153 bp CaMV 35S enhancer
    assert with_enh.startswith(ENHANCER_153)
    assert promoter in with_enh
    assert UTR_MAP["Zm"] in with_enh


def test_build_construct_random_upstream_without_enhancer():
    rng = np.random.default_rng(0)
    promoter = "A" * PROMOTER_LENGTH
    no_enh = build_construct(promoter, "Zm", False, rng)
    # without the enhancer the 153 bp upstream filler is random (not the CaMV enhancer)
    assert not no_enh.startswith(ENHANCER_153)
    assert len(no_enh[:len(ENHANCER_153)]) == len(ENHANCER_153)


def test_utr_by_species():
    # At and Zm share the maize ZmUTR (68 bp); Sb uses SbUTR (102 bp)
    assert UTR_MAP["At"] == UTR_MAP["Zm"]
    assert len(UTR_MAP["At"]) == 68
    assert len(UTR_MAP["Sb"]) == 102


@pytest.mark.parametrize("mode,expected_len", [
    ("promoter_only", PROMOTER_LENGTH),
    ("enhancer", SEQUENCE_LENGTH),
    ("combined", SEQUENCE_LENGTH),
])
def test_build_sequence_for_mode_lengths(mode, expected_len):
    row = pd.Series({"sequence": "A" * PROMOTER_LENGTH, "sp": "Zm", "use_enh": mode != "combined"})
    rng = np.random.default_rng(0)
    seq = build_sequence_for_mode(row, mode, rng)
    assert len(seq) == expected_len


# ── Split loader ─────────────────────────────────────────────────────────────


def test_load_splits_shapes_and_columns():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic(tmp, n_train=40, n_test=10)
        train = _load_plant_starrseq_data(tmp, "leaf", "enhancer", "train")
        val = _load_plant_starrseq_data(tmp, "leaf", "enhancer", "val")
        test = _load_plant_starrseq_data(tmp, "leaf", "enhancer", "test")

        for df in (train, val, test):
            assert set(df.columns) == {"sequence", "enrichment", "sp", "use_enh"}
        # val_frac=0.1 of 40 = 4 val, 36 train; 10 test
        assert len(val) == 4
        assert len(train) == 36
        assert len(test) == 10
        # enhancer mode always flags use_enh True
        assert train["use_enh"].all()


def test_train_val_are_disjoint_and_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic(tmp, n_train=40, n_test=10)
        train = _load_plant_starrseq_data(tmp, "leaf", "enhancer", "train")
        val = _load_plant_starrseq_data(tmp, "leaf", "enhancer", "val")
        # deterministic seed-42 permutation split -> no leakage between train and val
        assert set(train["sequence"]) & set(val["sequence"]) == set()
        # re-loading gives the same split
        val2 = _load_plant_starrseq_data(tmp, "leaf", "enhancer", "val")
        assert list(val["sequence"]) == list(val2["sequence"])


def test_combined_concatenates_enhancer_and_noenh():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic(tmp, n_train=40, n_test=10)
        test = _load_plant_starrseq_data(tmp, "leaf", "combined", "test")
        # combined test = 35SEnh test (10) + noEnh test (10)
        assert len(test) == 20
        assert test["use_enh"].sum() == 10
        assert (~test["use_enh"]).sum() == 10


def test_promoter_only_and_enhancer_share_the_same_table():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic(tmp, n_train=40, n_test=10)
        prom = _load_plant_starrseq_data(tmp, "leaf", "promoter_only", "test")
        enh = _load_plant_starrseq_data(tmp, "leaf", "enhancer", "test")
        # both read the 35SEnh table -> identical raw promoters
        assert list(prom["sequence"]) == list(enh["sequence"])


# ── Torch dataset (skipped if torch is absent) ───────────────────────────────


def test_torch_dataset_round_trip():
    pytest.importorskip("torch")
    from alphagenome_ft_mpra.plant_torch import (
        PlantStarrSeqTorchDataset, make_onehot_collate_fn,
    )

    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic(tmp, n_train=40, n_test=10)
        ds = PlantStarrSeqTorchDataset(tmp, "leaf", "combined", "train",
                                       random_shift=True, reverse_complement=True)
        assert len(ds) == 72  # (40 enh + 40 noenh) train minus 10% val = 72

        seq, target = ds[0]
        assert isinstance(seq, str)
        assert isinstance(target, float)

        collate = make_onehot_collate_fn(SEQUENCE_LENGTH)
        x, y = collate([ds[0], ds[1], ds[2]])
        assert x.shape == (3, SEQUENCE_LENGTH, 4)
        # one-hot rows sum to 1 (ACGT) or 4*0.25 (N)
        import numpy as _np
        _np.testing.assert_allclose(x.sum(axis=2).numpy(), 1.0, atol=1e-6)
