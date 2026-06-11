"""Minibatch CKA (Centered Kernel Alignment) core — framework-agnostic (numpy).

Lifted from clgenomics/utils/analysis.py (Kornblith et al., ICML 2019;
unbiased HSIC estimator). The clgenomics version is tied to PyTorch forward
hooks; AlphaGenome here is JAX/Haiku, so we keep ONLY the math and accept
plain numpy feature matrices that the caller extracts however it likes.

CKA over minibatches accumulates three HSIC terms per layer across batches and
combines at the end:  CKA = mean HSIC(K,L) / sqrt(mean HSIC(K,K) * mean HSIC(L,L)).

Each layer's features are flattened to (N, D) per batch (N = batch size).
Linear CKA uses Gram matrices K = X X^T, L = Y Y^T.
"""
from __future__ import annotations

import numpy as np


def _unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimate (Song et al. 2012; as in Nguyen et al. 2021 Eq 3).

    K, L are (N, N) Gram matrices with diagonals already zeroed.
    """
    n = K.shape[0]
    if n <= 3:
        raise ValueError(f"HSIC needs N>3, got N={n}")
    ones = np.ones((n, 1), dtype=np.float64)
    K = K.astype(np.float64)
    L = L.astype(np.float64)
    tr = np.trace(K @ L)
    term2 = (ones.T @ K @ ones @ ones.T @ L @ ones) / ((n - 1) * (n - 2))
    term3 = (ones.T @ K @ L @ ones) * 2 / (n - 2)
    return float((tr + term2.item() - term3.item()) / (n * (n - 3)))


def _gram_linear(feats: np.ndarray) -> np.ndarray:
    """Linear Gram matrix with zeroed diagonal (for unbiased HSIC)."""
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)
    # Guard against degenerate (constant) features that make HSIC NaN.
    if feats.std() < 1e-9:
        feats = feats + np.random.default_rng(0).standard_normal(feats.shape) * 1e-9
    g = feats @ feats.T
    np.fill_diagonal(g, 0.0)
    return g


class CKAAccumulator:
    """Accumulate per-layer HSIC terms across minibatches, then finalize CKA.

    Usage:
        acc = CKAAccumulator(layer_names)
        for batch features dict_a, dict_b:
            acc.update(dict_a, dict_b)     # {layer_name: (N, ...) array}
        result = acc.finalize()           # {layer_name: cka_float}
    """

    def __init__(self, layer_names: list[str]):
        self.layer_names = list(layer_names)
        # [n_layers, 3] -> columns: HSIC(K,K), HSIC(K,L), HSIC(L,L)
        self._acc = np.zeros((len(self.layer_names), 3), dtype=np.float64)
        self._n_batches = 0

    def update(self, feats_a: dict, feats_b: dict) -> None:
        for i, name in enumerate(self.layer_names):
            if name not in feats_a or name not in feats_b:
                continue
            K = _gram_linear(feats_a[name])
            L = _gram_linear(feats_b[name])
            self._acc[i, 0] += _unbiased_hsic(K, K)
            self._acc[i, 1] += _unbiased_hsic(K, L)
            self._acc[i, 2] += _unbiased_hsic(L, L)
        self._n_batches += 1

    def finalize(self) -> dict[str, float]:
        if self._n_batches == 0:
            raise RuntimeError("no batches accumulated")
        acc = self._acc / self._n_batches
        denom = np.sqrt(acc[:, 0] * acc[:, 2]) + 1e-12
        cka = acc[:, 1] / denom
        return {name: float(cka[i]) for i, name in enumerate(self.layer_names)}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Single-shot unbiased linear CKA between two (N, D) feature matrices."""
    K = _gram_linear(x)
    L = _gram_linear(y)
    hkl = _unbiased_hsic(K, L)
    hkk = _unbiased_hsic(K, K)
    hll = _unbiased_hsic(L, L)
    return float(hkl / (np.sqrt(hkk * hll) + 1e-12))
