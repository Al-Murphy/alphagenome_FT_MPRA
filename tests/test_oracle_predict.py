"""Tests for MPRAOracle predict APIs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from alphagenome_research.model import dna_model, one_hot_encoder

from alphagenome_ft_mpra.oracle import MPRAOracle


class _DummyDeviceContext:
    def __init__(self):
        self._device = jax.devices()[0]

    def __enter__(self):
        return self._device

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyMetadata:
    def __init__(self):
        self.strand_reindexing = jnp.array([], dtype=jnp.int32)


class _DummyModel:
    def __init__(self):
        self._one_hot_encoder = one_hot_encoder.DNAOneHotEncoder()
        self._device_context = _DummyDeviceContext()
        self._params = {}
        self._state = {}
        self._metadata = {dna_model.Organism.HOMO_SAPIENS: _DummyMetadata()}
        self.last_sequence_shape: tuple[int, ...] | None = None

    def _predict(
        self,
        params,
        state,
        sequences,
        organism_indices,
        *,
        negative_strand_mask,
        strand_reindexing,
    ):
        del params, state, organism_indices, negative_strand_mask, strand_reindexing
        arr = np.asarray(sequences)
        self.last_sequence_shape = arr.shape
        # One scalar per position so pooling behavior is easy to validate.
        per_position = arr.sum(axis=-1, keepdims=True)
        return {"mpra_head": jnp.asarray(per_position, dtype=jnp.float32)}


def _make_oracle(
    model: _DummyModel,
    *,
    pooling_type: str = "sum",
    center_bp: int = 10000,
    left_adapter: str | None = "A",
    right_adapter: str | None = "C",
    promoter: str | None = "G",
    barcode: str | None = "T",
) -> MPRAOracle:
    return MPRAOracle(
        model,
        head_name="mpra_head",
        pooling_type=pooling_type,
        center_bp=center_bp,
        left_adapter=left_adapter,
        right_adapter=right_adapter,
        promoter=promoter,
        barcode=barcode,
    )


@pytest.fixture
def oracle_and_model():
    model = _DummyModel()
    oracle = _make_oracle(model)
    return oracle, model


def test_predict_accepts_2d_and_3d_onehot_arrays(oracle_and_model):
    oracle, model = oracle_and_model
    seq2 = np.asarray(model._one_hot_encoder.encode("AC"), dtype=np.float32)

    pred_2d = oracle.predict(seq2, mode="full")
    pred_3d = oracle.predict(np.stack([seq2, seq2], axis=0), mode="full")

    np.testing.assert_allclose(pred_2d, np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(pred_3d, np.asarray([2.0, 2.0], dtype=np.float32))


def test_predict_sequence_matches_predict_onehot_across_modes(oracle_and_model):
    oracle, model = oracle_and_model
    seqs = ["AC", "GT"]
    payload_onehot = np.stack([
        np.asarray(model._one_hot_encoder.encode(s), dtype=np.float32) for s in seqs
    ], axis=0)

    for mode in ("core", "flanked", "full"):
        from_seq = oracle.predict_sequence(seqs, mode=mode, batch_size=1)
        from_onehot = oracle.predict(payload_onehot, mode=mode, batch_size=1)
        np.testing.assert_allclose(from_seq, from_onehot)


def test_mode_changes_construct_length(oracle_and_model):
    oracle, model = oracle_and_model
    payload = np.asarray(model._one_hot_encoder.encode("AC"), dtype=np.float32)

    pred_full = oracle.predict(payload, mode="full")
    pred_flanked = oracle.predict(payload, mode="flanked")
    pred_core = oracle.predict(payload, mode="core")

    # Payload length is 2. flanked adds promoter + barcode (2 bp total), core adds 4 bp.
    np.testing.assert_allclose(pred_full, np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(pred_flanked, np.asarray([4.0], dtype=np.float32))
    np.testing.assert_allclose(pred_core, np.asarray([6.0], dtype=np.float32))


def test_none_construct_pieces_are_skipped():
    model = _DummyModel()
    oracle = _make_oracle(
        model,
        left_adapter=None,
        right_adapter=None,
        promoter=None,
        barcode=None,
    )
    payload = np.asarray(model._one_hot_encoder.encode("AC"), dtype=np.float32)

    pred_full = oracle.predict(payload, mode="full")
    pred_flanked = oracle.predict(payload, mode="flanked")
    pred_core = oracle.predict(payload, mode="core")

    np.testing.assert_allclose(pred_full, np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(pred_flanked, np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(pred_core, np.asarray([2.0], dtype=np.float32))


def test_validation_errors(oracle_and_model):
    oracle, _ = oracle_and_model

    with pytest.raises(ValueError, match="Invalid mode"):
        oracle.predict(np.zeros((3, 4), dtype=np.float32), mode="bad")
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        oracle.predict(np.zeros((3, 4), dtype=np.float32), batch_size=0)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        oracle.predict(np.zeros((3,), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        oracle.predict(np.zeros((3, 5), dtype=np.float32))
    with pytest.raises(TypeError, match="NumPy or JAX array"):
        oracle.predict("ACGT")
    with pytest.raises(TypeError, match="NumPy or JAX array"):
        oracle.predict([np.zeros((3, 4), dtype=np.float32)])
    with pytest.raises(TypeError, match="to be str"):
        oracle.predict_sequence([123])  # type: ignore[list-item]


def test_pooling_regression_modes():
    model = _DummyModel()
    preds = np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32)

    oracle_sum = _make_oracle(model, pooling_type="sum", center_bp=10000)
    oracle_mean = _make_oracle(model, pooling_type="mean", center_bp=10000)
    oracle_max = _make_oracle(model, pooling_type="max", center_bp=10000)
    oracle_center = _make_oracle(model, pooling_type="center")
    oracle_flatten = _make_oracle(model, pooling_type="flatten")

    np.testing.assert_allclose(oracle_sum._pool_predictions(preds), np.asarray([6.0]))
    np.testing.assert_allclose(oracle_mean._pool_predictions(preds), np.asarray([2.0]))
    np.testing.assert_allclose(oracle_max._pool_predictions(preds), np.asarray([3.0]))
    np.testing.assert_allclose(oracle_center._pool_predictions(preds), np.asarray([2.0]))

    flatten_preds = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        oracle_flatten._pool_predictions(flatten_preds),
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )
