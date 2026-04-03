"""Oracle API for MPRA inference with fine-tuned AlphaGenome checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import (
    CustomAlphaGenomeModel,
    HeadConfig,
    HeadType,
    load_checkpoint,
    register_custom_head,
)

from .mpra_heads import EncoderMPRAHead

DEFAULT_LEFT_ADAPTER = "AGGACCGGATCAACT"
DEFAULT_RIGHT_ADAPTER= "CATTGCGTGAACCGA"
DEFAULT_PROMOTER = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
DEFAULT_BARCODE = "AGAGACTGAGGCCAC"

_PREDICT_REQUESTED_OUTPUTS: tuple = tuple(dna_output.OutputType)


class MPRAOracle:
    """Lightweight predictor around a fine-tuned MPRA head."""

    def __init__(
        self,
        model: CustomAlphaGenomeModel,
        *,
        head_name: str = "mpra_head",
        pooling_type: str = "sum",
        center_bp: int = 256,
        left_adapter: str | None = DEFAULT_LEFT_ADAPTER,
        right_adapter: str | None = DEFAULT_RIGHT_ADAPTER,
        promoter: str | None = DEFAULT_PROMOTER,
        barcode: str | None = DEFAULT_BARCODE,
    ):
        self.model: CustomAlphaGenomeModel = model
        self.head_name = head_name
        self.pooling_type = pooling_type
        self.center_bp = center_bp
        self.left_adapter = left_adapter
        self.right_adapter = right_adapter
        self.promoter = promoter
        self.barcode = barcode

    @staticmethod
    def _validate_mode(mode: str) -> str:
        norm_mode = mode.lower()
        if norm_mode not in {"core", "flanked", "full"}:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: core, flanked, full"
            )
        return norm_mode

    def _encode_sequence(self, seq: str) -> np.ndarray:
        return np.asarray(
            self.model._one_hot_encoder.encode(seq.strip().upper()),
            dtype=np.float32,
        )

    def _normalize_onehot_inputs(
        self,
        onehot: np.ndarray | jnp.ndarray,
    ) -> list[np.ndarray]:
        if not isinstance(onehot, (np.ndarray, jnp.ndarray)):
            raise TypeError(
                "predict() expects a NumPy or JAX array. "
                "Use predict_sequence() for strings."
            )

        arr = np.asarray(onehot, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[-1] != 4:
                raise ValueError(f"Expected onehot shape (S, 4); got {arr.shape}.")
            return [arr]
        if arr.ndim == 3:
            if arr.shape[-1] != 4:
                raise ValueError(f"Expected onehot shape (B, S, 4); got {arr.shape}.")
            return [arr[i] for i in range(arr.shape[0])]
        raise ValueError(f"Expected onehot rank 2 or 3 array; got rank {arr.ndim}.")

    def _normalize_sequence_inputs(self, seq: str | Iterable[str]) -> list[str]:
        if isinstance(seq, str):
            return [seq.strip().upper()]

        seqs = list(seq)
        normalized: list[str] = []
        for i, item in enumerate(seqs):
            if not isinstance(item, str):
                raise TypeError(
                    f"Expected sequence element {i} to be str; got {type(item).__name__}."
                )
            normalized.append(item.strip().upper())
        return normalized

    def _construct_parts_for_mode(self, mode: str) -> tuple[np.ndarray | None, ...]:
        left = (
            self._encode_sequence(self.left_adapter)
            if self.left_adapter is not None
            else None
        )
        right = (
            self._encode_sequence(self.right_adapter)
            if self.right_adapter is not None
            else None
        )
        promoter = (
            self._encode_sequence(self.promoter)
            if self.promoter is not None
            else None
        )
        barcode = (
            self._encode_sequence(self.barcode)
            if self.barcode is not None
            else None
        )

        if mode == "core":
            return left, right, promoter, barcode
        if mode == "flanked":
            return None, None, promoter, barcode
        return None, None, None, None

    def _apply_mode_to_onehot(self, payloads: list[np.ndarray], mode: str) -> list[np.ndarray]:
        left, right, promoter, barcode = self._construct_parts_for_mode(mode)
        constructed: list[np.ndarray] = []
        for payload in payloads:
            parts: list[np.ndarray] = []
            if left is not None:
                parts.append(left)
            parts.append(payload)
            if right is not None:
                parts.append(right)
            if promoter is not None:
                parts.append(promoter)
            if barcode is not None:
                parts.append(barcode)
            constructed.append(np.concatenate(parts, axis=0).astype(np.float32, copy=False))
        return constructed

    @staticmethod
    def _pad_onehot_batch(onehot: list[np.ndarray]) -> jnp.ndarray:
        max_len = max(arr.shape[0] for arr in onehot)
        padded: list[np.ndarray] = []

        for arr in onehot:
            if arr.shape[0] < max_len:
                pad = np.zeros((max_len - arr.shape[0], 4), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            padded.append(arr.astype(np.float32, copy=False))

        return jnp.asarray(np.stack(padded, axis=0), dtype=jnp.float32)

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

    def _predict_onehot_batch(self, onehot_batch: list[np.ndarray]) -> np.ndarray:
        batch_seq = self._pad_onehot_batch(onehot_batch)
        batch_org = jnp.zeros((batch_seq.shape[0],), dtype=jnp.int32)

        with self.model._device_context:
            predictions = self.model._predict(
                self.model._params,
                self.model._state,
                batch_seq,
                batch_org,
                requested_outputs=_PREDICT_REQUESTED_OUTPUTS,
                negative_strand_mask=jnp.zeros(batch_seq.shape[0], dtype=bool),
                strand_reindexing=jax.device_put(
                    self.model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                    self.model._device_context._device,
                ),
            )

        head_predictions = np.array(predictions[self.head_name])
        return self._pool_predictions(head_predictions)

    def _predict_payloads(
        self,
        payloads: list[np.ndarray],
        *,
        mode: str,
        batch_size: int,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {batch_size}.")
        if not payloads:
            return np.array([])

        norm_mode = self._validate_mode(mode)
        assembled = self._apply_mode_to_onehot(payloads, norm_mode)

        all_preds = []
        for i in range(0, len(assembled), batch_size):
            batch = assembled[i : i + batch_size]
            all_preds.append(self._predict_onehot_batch(batch))
        return np.concatenate(all_preds, axis=0)

    def predict(
        self,
        onehot: np.ndarray | jnp.ndarray,
        *,
        mode: str = "core",
        batch_size: int = 64,
    ) -> np.ndarray:
        """Predict MPRA activity from one-hot encoded sequences.

        Args:
            onehot: Onehot payload(s), shape (S, 4) or (B, S, 4).
                Channel order must be A, C, G, T.
            mode: Construct mode:
                - "core": left_adapter + payload + right_adapter + promoter + barcode
                - "flanked": payload + promoter + barcode
                - "full": payload only
            batch_size: Number of sequences per inference batch.
        """
        payloads = self._normalize_onehot_inputs(onehot)
        return self._predict_payloads(payloads, mode=mode, batch_size=batch_size)

    def predict_sequences(
        self,
        sequences: str | Iterable[str],
        *,
        mode: str = "core",
        batch_size: int = 64,
    ) -> np.ndarray:
        """Predict MPRA activity from raw DNA sequence strings.

        Args:
            seq: Single sequence or iterable of payload sequences.
            mode: Construct mode:
                - "core": left_adapter + payload + right_adapter + promoter + barcode
                - "flanked": payload + promoter + barcode
                - "full": payload only
            batch_size: Number of sequences per inference batch.
        """
        payloads = self._normalize_sequence_inputs(sequences)
        onehot_payloads = [self._encode_sequence(s) for s in payloads]
        return self._predict_payloads(onehot_payloads, mode=mode, batch_size=batch_size)


def _leaves_with_paths(tree, prefix: str = "") -> list[tuple[str, Any]]:
    """Recursively collect (path, leaf) for all array leaves in a pytree."""
    out: list[tuple[str, any]] = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            p = f"{prefix}/{k}" if prefix else str(k)
            if hasattr(v, "shape") and hasattr(v, "size"):
                out.append((p, v))
            else:
                out.extend(_leaves_with_paths(v, p))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            p = f"{prefix}/{i}"
            if hasattr(v, "shape") and hasattr(v, "size"):
                out.append((p, v))
            else:
                out.extend(_leaves_with_paths(v, p))
    return out


def _print_encoder_head_param_count(model, head_name: str) -> None:
    """Print encoder + head parameter counts for a minimal oracle model.

    Handles both nested (alphagenome/sequence_encoder, head/mpra_head) and
    flat key layouts (e.g. head/mpra_head/~predict/... at top level).
    """
    params = getattr(model, "_params", None)
    if params is None:
        return

    n_encoder = 0
    n_head = 0
    # Match encoder: "alphagenome/sequence_encoder/..." or flat "alphagenome/sequence_encoder/..."
    # Match head: "head/mpra_head/..." or "alphagenome/head/mpra_head/..."
    encoder_marker = "sequence_encoder"
    head_marker = f"head/{head_name}"

    for path, leaf in _leaves_with_paths(params):
        try:
            n = int(leaf.size)
        except AttributeError:
            continue
        # Encoder: path contains sequence_encoder and not transformer/decoder
        if encoder_marker in path and "transformer" not in path and "decoder" not in path:
            n_encoder += n
        # Head: path contains head/mpra_head
        elif head_marker in path:
            n_head += n

    if n_encoder > 0 or n_head > 0:
        print(
            f"  Encoder parameters:   {n_encoder:,}\n"
            f"  Head parameters:      {n_head:,}\n"
            f"  Encoder+Head total:   {n_encoder + n_head:,}"
        )


def load_oracle(
    checkpoint_dir: str | Path,
    *,
    base_checkpoint_path: str | Path | None = None,
    base_model_version: str = "all_folds",
    device=None,
    left_adapter: str | None = DEFAULT_LEFT_ADAPTER,
    right_adapter: str | None = DEFAULT_RIGHT_ADAPTER,
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

    # Also show the effective encoder+head size, which is the part actually used
    # by the minimal oracle (encoder-only forward + MPRA head).
    _print_encoder_head_param_count(model, head_name)

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
