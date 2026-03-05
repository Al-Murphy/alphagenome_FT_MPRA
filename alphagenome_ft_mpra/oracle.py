"""Oracle API for MPRA inference with fine-tuned AlphaGenome checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import HeadConfig, HeadType, load_checkpoint, register_custom_head

from .mpra_heads import EncoderMPRAHead

LEFT_ADAPTER = "AGGACCGGATCAACT"
RIGHT_ADAPTER= "CATTGCGTGAACCGA"
DEFAULT_PROMOTER = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
DEFAULT_BARCODE = "AGAGACTGAGGCCAC"


class MPRAOracle:
    """Lightweight predictor around a fine-tuned MPRA head."""

    def __init__(
        self,
        model,
        *,
        head_name: str = "mpra_head",
        pooling_type: str = "sum",
        center_bp: int = 256,
        left_adapter: str | None = LEFT_ADAPTER,
        right_adapter: str | None = RIGHT_ADAPTER,
        promoter: str | None = DEFAULT_PROMOTER,
        barcode: str | None = DEFAULT_BARCODE,
    ):
        self.model = model
        self.head_name = head_name
        self.pooling_type = pooling_type
        self.center_bp = center_bp
        self.left_adapter = left_adapter
        self.right_adapter = right_adapter
        self.promoter = promoter
        self.barcode = barcode

    def _assemble_sequence(self, seq: str, mode: str) -> str:
        mode = mode.lower()
        clean_seq = seq.strip().upper()

        if mode == "core":
            parts = [
                self.left_adapter,
                clean_seq,
                self.right_adapter,
                self.promoter,
                self.barcode,
            ]
        elif mode == "flanked":
            parts = [clean_seq, self.promoter, self.barcode]
        elif mode == "full":
            parts = [clean_seq]
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: core, flanked, full")

        return "".join(part for part in parts if part is not None)

    def _encode_batch(self, seqs: list[str]) -> jnp.ndarray:
        encoded = [jnp.array(self.model._one_hot_encoder.encode(s)) for s in seqs]
        max_len = max(arr.shape[0] for arr in encoded)

        padded = []
        for arr in encoded:
            if arr.shape[0] < max_len:
                pad = jnp.zeros((max_len - arr.shape[0], 4), dtype=arr.dtype)
                arr = jnp.concatenate([arr, pad], axis=0)
            padded.append(arr)

        return jnp.stack(padded, axis=0)

    def _pool_predictions(self, head_predictions: np.ndarray) -> np.ndarray:
        seq_len = head_predictions.shape[1]

        if self.pooling_type == "flatten":
            pooled = head_predictions.squeeze(1)
        elif self.pooling_type == "center":
            center_idx = seq_len // 2
            pooled = head_predictions[:, center_idx, :]
        else:
            center_window_positions = max(1, self.center_bp // 128)
            window_size = min(center_window_positions, seq_len)
            center_start = max((seq_len - window_size) // 2, 0)
            center_end = center_start + window_size
            center_preds = head_predictions[:, center_start:center_end, :]

            if self.pooling_type == "mean":
                pooled = center_preds.mean(axis=1)
            elif self.pooling_type == "max":
                pooled = center_preds.max(axis=1)
            else:
                pooled = center_preds.sum(axis=1)

        if pooled.ndim == 2 and pooled.shape[-1] == 1:
            pooled = pooled[:, 0]

        return np.asarray(pooled)

    def _predict_batch(self, seqs: list[str]) -> np.ndarray:
        batch_seq = self._encode_batch(seqs)
        batch_org = jnp.zeros((len(seqs),), dtype=jnp.int32)

        with self.model._device_context:
            predictions = self.model._predict(
                self.model._params,
                self.model._state,
                batch_seq,
                batch_org,
                negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
                strand_reindexing=jax.device_put(
                    self.model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                    self.model._device_context._device,
                ),
            )

        head_predictions = np.array(predictions[self.head_name])
        return self._pool_predictions(head_predictions)

    def predict(
        self,
        seq: str | Iterable[str],
        *,
        mode: str = "core",
        batch_size: int = 64,
    ) -> np.ndarray:
        """Predict MPRA activity.

        Args:
            seq: Single sequence or iterable of sequences.
            mode: Sequence format mode: "core", "flanked", or "full".
            batch_size: Number of sequences per inference batch.
        """
        if isinstance(seq, str):
            seqs = [seq]
        else:
            seqs = list(seq)

        if not seqs:
            return np.array([])

        assembled = [self._assemble_sequence(s, mode=mode) for s in seqs]

        all_preds = []
        for i in range(0, len(assembled), batch_size):
            batch = assembled[i : i + batch_size]
            all_preds.append(self._predict_batch(batch))

        return np.concatenate(all_preds, axis=0)


def load_oracle(
    checkpoint_dir: str | Path,
    *,
    base_checkpoint_path: str | Path | None = None,
    base_model_version: str = "all_folds",
    device=None,
    left_adapter: str | None = None,
    right_adapter: str | None = None,
    promoter: str | None = DEFAULT_PROMOTER,
    barcode: str | None = DEFAULT_BARCODE,
) -> MPRAOracle:
    """Load a fine-tuned MPRA oracle from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.json"

    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    custom_heads = config.get("custom_heads", []) or []
    head_name = "mpra_head"
    if custom_heads:
        head_name = custom_heads[0]

    head_cfg = config.get("head_configs", {}).get(head_name, {})
    head_metadata = head_cfg.get("metadata", {}) or {}
    num_tracks = int(head_cfg.get("num_tracks", 1))

    register_custom_head(
        head_name,
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name=head_name,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=num_tracks,
            metadata=head_metadata,
        ),
    )

    # Keep init_seq_len behavior aligned with existing MPRA scripts.
    load_kwargs = dict(base_model_version=base_model_version)
    if base_checkpoint_path is not None:
        load_kwargs["base_checkpoint_path"] = str(base_checkpoint_path)
    if device is not None:
        load_kwargs["device"] = device

    try:
        model = load_checkpoint(str(checkpoint_dir), **load_kwargs)
    except TypeError:
        model = load_checkpoint(str(checkpoint_dir))

    return MPRAOracle(
        model,
        head_name=head_name,
        pooling_type=head_metadata.get("pooling_type", "sum"),
        center_bp=int(head_metadata.get("center_bp", 256)),
        left_adapter=left_adapter,
        right_adapter=right_adapter,
        promoter=promoter,
        barcode=barcode,
    )
