"""Tests for scripts/zeroshot_base_AG_mpra_predict.py (dummy model, no weights)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model, one_hot_encoder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "zeroshot_base_AG_mpra_predict.py"
spec = importlib.util.spec_from_file_location("zeroshot_script", SCRIPT)
zs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zs)

HUMAN = dna_model.Organism.HOMO_SAPIENS
DNASE = dna_output.OutputType.DNASE
CAGE = dna_output.OutputType.CAGE
L = 2048
# DNASE: tracks 0,1 real (K562, HepG2), track 2 padding. Value of real track t at
# every position = (t + 1) * (number of non-N bases at that position).
CURIES = {DNASE: ["EFO:0002067", "EFO:0001187", ""], CAGE: ["EFO:0002067", ""]}
PADDING = {DNASE: [False, False, True], CAGE: [False, True]}


class _Meta:
    strand_reindexing = jnp.array([], dtype=jnp.int32)

    def __init__(self):
        self.padding = {k: np.array(v) for k, v in PADDING.items()}

    def get(self, output_type):
        if output_type not in CURIES:
            return None
        return pd.DataFrame({"ontology_curie": CURIES[output_type]})


class _Ctx:
    _device = jax.devices()[0]


class _DummyModel:
    """Mimics the real model: padding tracks are NaN in the raw `_predict` output."""

    def __init__(self):
        self._one_hot_encoder = one_hot_encoder.DNAOneHotEncoder()
        self._device_context = _Ctx()
        self._params, self._state = {}, {}
        self._metadata = {HUMAN: _Meta(), dna_model.Organism.MUS_MUSCULUS: _Meta()}
        self.calls = []  # (one_hot batch, negative_strand_mask)

    def _predict(self, params, state, seqs, org_idx, *, requested_outputs,
                 negative_strand_mask, strand_reindexing):
        arr = np.asarray(seqs)
        self.calls.append((arr, np.asarray(negative_strand_mask), np.asarray(org_idx)))
        base = arr.sum(-1)  # (B, S)
        out = {}
        for ot in requested_outputs:
            pads = PADDING[ot]
            tracks = [np.full_like(base, np.nan) if p else base * (t + 1)
                      for t, p in enumerate(pads)]
            out[ot] = jnp.asarray(np.stack(tracks, -1), dtype=jnp.float32)
        return out


def _batch(model, seqs, *, ontology_terms=None, outputs=(DNASE,), **kw):
    return zs.predict_batch(
        model, seqs, window_bps=kw.pop("window_bps", [None] * len(seqs)),
        organism=HUMAN, output_types=outputs, ontology_terms=ontology_terms,
        pooling=kw.pop("pooling", "mean"), **kw)


def test_default_no_curie_is_not_nan_and_drops_padding_tracks():
    seqs = ["A" * L, "N" * (L // 2) + "C" * (L // 2)]
    res = _batch(_DummyModel(), seqs)
    vals = [r["DNASE"] for r in res]
    assert not any(np.isnan(vals))
    # mean over the two real tracks (1 and 2) -> 1.5 * mean(base)
    assert vals == pytest.approx([1.5, 0.75])


def test_ontology_filter_matches_track_masks():
    from alphagenome.data import ontology
    res = _batch(_DummyModel(), ["A" * L], ontology_terms=[ontology.from_curie("EFO:0001187")])
    assert res[0]["DNASE"] == pytest.approx(2.0)  # only track 1


def test_center_window_uses_given_window_not_stripped_length():
    seq = "N" * 1000 + "A" * 48 + "N" * 1000
    res = _batch(_DummyModel(), [seq], pooling="center_window", window_bps=[48])
    assert res[0]["DNASE"] == pytest.approx(1.5 * 48)


def test_reverse_complement_runs_negative_strand_pass_and_averages():
    m = _DummyModel()
    seq = "A" * (L - 4) + "CCCC"
    res = _batch(m, [seq], use_reverse_complement=True, pooling="sum")
    assert len(m.calls) == 2
    assert not m.calls[0][1].any() and m.calls[1][1].all()
    # second pass received the reverse complement
    rc_expected = np.asarray(m._one_hot_encoder.encode("GGGG" + "T" * (L - 4)))
    assert np.array_equal(m.calls[1][0][0], rc_expected)
    assert res[0]["DNASE"] == pytest.approx(1.5 * L)


def test_organism_index_uses_converter():
    m = _DummyModel()
    zs.predict_batch(m, ["A" * L], window_bps=[None], organism=dna_model.Organism.MUS_MUSCULUS,
                     output_types=[DNASE], ontology_terms=None, pooling="mean")
    assert m.calls[0][2].tolist() == [1]


def test_unequal_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        _batch(_DummyModel(), ["A" * L, "A" * (L * 2)])


@pytest.mark.parametrize("bad", [3000, 100, -2048])
def test_validate_standardize_length_rejects(bad):
    with pytest.raises(ValueError):
        zs.validate_standardize_length(bad)


@pytest.mark.parametrize("ok", [0, 2048, 4096])
def test_validate_standardize_length_accepts(ok):
    zs.validate_standardize_length(ok)


def test_assemble_constructs_honours_template():
    tpl = "{promoter}{left_adapter}{insert}{right_adapter}{barcode}"
    (c,), (n,) = zs.assemble_constructs(
        ["acgt"], template=tpl, left_adapter="LL", right_adapter="RR",
        promoter="PP", barcode="BB", standardize_length_bp=0)
    assert c == "PPLLACGTRRBB" and n == 12
    (d,), _ = zs.assemble_constructs(
        ["ACGT"], template=zs_default_template(), left_adapter="LL", right_adapter="RR",
        promoter="PP", barcode="BB", standardize_length_bp=0)
    assert d == "LLACGTRRPPBB"


def zs_default_template():
    return "{left_adapter}{insert}{right_adapter}{promoter}{barcode}"


def _run_main(monkeypatch, tmp_path, model, extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tsv = tmp_path / "in.tsv"
    pd.DataFrame({"sequence": ["ACGT" * 50, "TTGCA" * 40], "score": [1.0, 2.0]}).to_csv(
        tsv, sep="\t", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(zs, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(sys, "argv", ["x", "--input", str(tsv), "--output_dir", str(out),
                                      "--standardize_length", "2048", "--batch_size", "2", *extra])
    zs.main()
    return pd.read_csv(out / "predictions.csv"), pd.read_csv(out / "metrics_summary.csv")


def test_main_two_rows_default_config_not_all_nan(monkeypatch, tmp_path):
    preds, metrics = _run_main(monkeypatch, tmp_path, _DummyModel(),
                               ["--output_types", "DNASE", "CAGE"])
    assert len(preds) == 2
    for col in ("DNASE", "CAGE"):
        assert preds[col].notna().all()
    assert (metrics["n_valid"] == 2).all()


def test_main_passes_construct_template(monkeypatch, tmp_path):
    m = _DummyModel()
    _run_main(monkeypatch, tmp_path, m, [
        "--output_types", "DNASE", "--promoter", "GGGGGGGG",
        "--construct_template", "{promoter}{insert}"])
    first = np.asarray(m.calls[0][0])[0]
    decoded = "".join("ACGT"[i] if row.any() else "N" for row in first for i in [row.argmax()])
    stripped = decoded.strip("N")
    assert stripped.startswith("GGGGGGGG") and stripped.endswith("ACGT" * 50)


def test_main_reverse_complement_flag_is_live(monkeypatch, tmp_path):
    m = _DummyModel()
    _run_main(monkeypatch, tmp_path, m, ["--output_types", "DNASE", "--reverse_complement_avg"])
    assert any(call[1].all() for call in m.calls)


def test_main_rejects_bad_standardize_length(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, _DummyModel(), ["--standardize_length", "3000"])


def test_bfloat16_outputs_are_pooled_in_float32():
    class _BF16(_DummyModel):
        def _predict(self, *a, **k):
            out = super()._predict(*a, **k)
            return {ot: v.astype(jnp.bfloat16) for ot, v in out.items()}

    res = _batch(_BF16(), ["A" * L], pooling="sum")
    assert res[0]["DNASE"] == pytest.approx(1.5 * L, rel=1e-3)


# ---- position-shift test-time augmentation --------------------------------


def test_shift_sequence_moves_content_and_keeps_length():
    s = "NNNNACGTNNNN"
    assert zs.shift_sequence(s, 2) == "NNNNNNACGTNN"
    assert zs.shift_sequence(s, -3) == "NACGTNNNNNNN"
    assert zs.shift_sequence(s, 0) == s


def test_shift_sequence_refuses_to_drop_real_bases():
    with pytest.raises(ValueError, match="push real sequence"):
        zs.shift_sequence("ACGTNN", 3)


def test_random_shifts_reproducible_and_do_not_touch_global_rng():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = zs.get_random_position_shifts(20, 3, 7)
    b = zs.get_random_position_shifts(20, 3, 7)
    c = zs.get_random_position_shifts(20, 3, 8)
    assert a == b and a != c
    assert a[0] == 0 and len(a) == 4 and len(set(a)) == 4
    assert all(1 <= abs(x) <= 20 for x in a[1:])
    assert np.array_equal(np.random.get_state()[1], before)


def test_pool_track_center_override():
    v = np.zeros((100, 1))
    v[70:80] = 1.0
    assert zs.pool_track(v, pooling="center_window", center_window_bp=10, center=75) == 10
    assert zs.pool_track(v, pooling="center_window", center_window_bp=10) == 0


def test_pooling_window_follows_shifted_insert():
    insert = 48
    seq = "N" * ((L - insert) // 2) + "A" * insert + "N" * ((L - insert) // 2)
    kw = dict(pooling="center_window", window_bps=[insert])
    base = _batch(_DummyModel(), [seq], **kw)[0]["DNASE"]
    shifted = _batch(_DummyModel(), [seq], shifts=[[17]], **kw)[0]["DNASE"]
    assert base == pytest.approx(1.5 * insert)
    # window pinned at L//2 would give a partial overlap; following gives all of it
    assert shifted == pytest.approx(1.5 * insert)


def test_shifts_cross_with_strands_and_average():
    m = _DummyModel()
    seq = "N" * 1000 + "A" * 48 + "N" * 1000
    _batch(m, [seq, seq], shifts=[[0, 5, -5], [0, 3, -3]], use_reverse_complement=True,
           pooling="center_window", window_bps=[48, 48])
    assert len(m.calls) == 6  # 3 shifts x 2 strands
    assert [bool(c[1].all()) for c in m.calls] == [False, True] * 3


def test_ragged_shifts_rejected():
    with pytest.raises(ValueError, match="same length"):
        _batch(_DummyModel(), ["A" * L, "A" * L], shifts=[[0, 1], [0]])


def test_main_position_shift_is_seeded_and_batch_size_independent(monkeypatch, tmp_path):
    common = ["--output_types", "DNASE", "--use_position_shift", "--max_shift", "20",
              "--random_seed", "5", "--center_window_bp", "0"]
    a, _ = _run_main(monkeypatch, tmp_path / "a", _DummyModel(), common)
    b, _ = _run_main(monkeypatch, tmp_path / "b", _DummyModel(), common)
    c, _ = _run_main(monkeypatch, tmp_path / "c", _DummyModel(), common + ["--batch_size", "1"])
    np.testing.assert_allclose(a["DNASE"], b["DNASE"])
    np.testing.assert_allclose(a["DNASE"], c["DNASE"])


def test_main_rejects_max_shift_larger_than_padding(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, _DummyModel(),
                  ["--use_position_shift", "--max_shift", "5000"])
