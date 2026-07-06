"""
PyTorch building blocks for the plant STARR-seq (Jores 2021) runner scripts.

Torch-guarded (imported lazily by the PlantCAD2 / Jores CNN runners), analogous
to ``enf_utils.py``. Holds the shared torch Dataset + collate builders, the
attention-pool / mean-pool MPRA heads, and the from-scratch Jores CNN. All
construct assembly + splits are delegated to ``plant_starrseq_utils`` so the
PyTorch and JAX paths cannot drift.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .plant_starrseq_utils import (
    PROMOTER_LENGTH,
    SEQUENCE_LENGTH,
    _load_plant_starrseq_data,
    _one_hot_encode,
    build_sequence_for_mode,
)


class PlantStarrSeqTorchDataset(Dataset):
    """Returns ``(dna_string, target_float)``; encoding happens in the collate fn."""

    def __init__(
        self,
        data_path: str,
        tissue: str,
        mode: str,
        split: str = "train",
        random_shift: bool = False,
        reverse_complement: bool = False,
        max_shift: int = 25,
        val_frac: float = 0.1,
        seed: int = 42,
    ):
        self.mode = mode
        self.random_shift = random_shift
        self.reverse_complement = reverse_complement
        self.max_shift = max_shift
        self._rng = np.random.default_rng(seed)
        self.data = _load_plant_starrseq_data(
            data_path, tissue, mode, split, val_frac=val_frac, seed=seed,
        )
        print(f"Loaded {len(self.data)} plant STARR-seq samples for {tissue} {mode} {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = build_sequence_for_mode(
            row, self.mode, self._rng,
            random_shift=self.random_shift, max_shift=self.max_shift,
            reverse_complement=self.reverse_complement,
        )
        return seq, float(row["enrichment"])


def make_onehot_collate_fn(seq_length: int = SEQUENCE_LENGTH) -> Callable:
    """(string, float) -> ((B, L, 4) float tensor, (B,) target tensor)."""

    def collate(batch):
        strings, targets = zip(*batch)
        arrs = []
        for s in strings:
            oh = _one_hot_encode(s)
            if oh.shape[0] < seq_length:
                pad = np.zeros((seq_length - oh.shape[0], 4), dtype=np.float32)
                oh = np.concatenate([oh, pad], axis=0)
            else:
                oh = oh[:seq_length]
            arrs.append(oh)
        return (torch.from_numpy(np.stack(arrs)),
                torch.tensor(targets, dtype=torch.float32))

    return collate


def make_tokenizer_collate_fn(tokenizer, pad_to_multiple_of: int | None = None) -> Callable:
    """(string, float) -> (input_ids long tensor, (B,) target tensor) for HF tokenizers."""

    def collate(batch):
        strings, targets = zip(*batch)
        kwargs = dict(add_special_tokens=False, padding=True, return_tensors="pt")
        if pad_to_multiple_of is not None:
            kwargs["pad_to_multiple_of"] = pad_to_multiple_of
        tokens = tokenizer(list(strings), **kwargs)
        return tokens["input_ids"], torch.tensor(targets, dtype=torch.float32)

    return collate


def create_plant_dataloaders(
    data_path: str,
    tissue: str,
    mode: str,
    batch_size: int,
    collate_fn: Callable,
    random_shift: bool = False,
    reverse_complement: bool = False,
    max_shift: int = 25,
    num_workers: int = 2,
    seed: int = 42,
):
    """(train, val, test) torch DataLoaders. Augmentation applies to train only."""
    train_ds = PlantStarrSeqTorchDataset(
        data_path, tissue, mode, "train", random_shift=random_shift,
        reverse_complement=reverse_complement, max_shift=max_shift, seed=seed,
    )
    val_ds = PlantStarrSeqTorchDataset(data_path, tissue, mode, "val", seed=seed)
    test_ds = PlantStarrSeqTorchDataset(data_path, tissue, mode, "test", seed=seed)

    kw = dict(num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
    )


# ── Heads ────────────────────────────────────────────────────────────────────


class MPRAHead(nn.Module):
    """Attention-pool MLP head over token features (B, T, C) -> scalar.

    Matches the head used by the NTv3-post / PlantCAD2 finetune runs: LayerNorm,
    multi-head softmax attention pooling over the token axis, then a residual MLP.
    """

    def __init__(self, encoder_dim: int, hidden_size: int = 1024, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(encoder_dim)
        self.attn_score = nn.Linear(encoder_dim, n_heads)
        self.fc1 = nn.Linear(n_heads * encoder_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, C)
        x = self.norm(x)
        scores = self.attn_score(x)                    # (B, T, H)
        weights = torch.softmax(scores, dim=1)         # softmax over T
        pooled = torch.einsum("bth,btc->bhc", weights, x)  # (B, H, C)
        pooled = pooled.reshape(pooled.shape[0], -1)   # (B, H*C)
        h = F.gelu(self.fc1(pooled))
        h = self.dropout(h)
        h = h + F.gelu(self.fc2(h))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


class MeanPoolHead(nn.Module):
    """Mean-pool over tokens then MLP (alternative to attention pooling)."""

    def __init__(self, encoder_dim: int, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(encoder_dim)
        self.fc1 = nn.Linear(encoder_dim, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x).mean(dim=1)
        h = self.dropout(F.gelu(self.fc1(x)))
        return self.out(h).squeeze(-1)


# ── Jores CNN (from scratch) ─────────────────────────────────────────────────


class BiConv1D(nn.Module):
    """RC-aware first conv layer: apply the same kernel to the forward and the
    reverse-complemented weights, sum the ReLU'd activations (Jores et al. 2021)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_layers: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_normal_(self.weight)
        self.extra = nn.ModuleList([
            nn.Conv1d(out_ch, out_ch, kernel_size, padding="same") for _ in range(n_layers - 1)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, C, L)
        pad = self.kernel_size // 2
        fwd = F.conv1d(x, self.weight, self.bias, padding=pad)
        rc = F.conv1d(x, torch.flip(self.weight, dims=[1, 2]), self.bias, padding=pad)
        h = F.relu(fwd) + F.relu(rc)
        h = self.dropout(h)
        for conv in self.extra:
            h = self.dropout(F.relu(conv(h)))
        return h


class JoresCNN(nn.Module):
    """End-to-end plant-promoter CNN trained from scratch (motif warm-start optional)."""

    def __init__(self, seq_len: int, dropout: float = 0.15):
        super().__init__()
        self.biconv = BiConv1D(4, 128, kernel_size=13, n_layers=2, dropout=dropout)
        self.conv = nn.Conv1d(128, 128, kernel_size=13, padding="same")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * 128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):  # x: (B, L, 4)
        x = x.transpose(1, 2)                 # (B, 4, L)
        h = self.biconv(x)
        h = self.dropout(F.relu(self.conv(h)))
        h = h.flatten(1)
        h = F.relu(self.bn(self.fc(h)))
        return self.out(h).squeeze(-1)


def seq_len_for_mode(mode: str) -> int:
    return PROMOTER_LENGTH if mode == "promoter_only" else SEQUENCE_LENGTH


# ── Ridge probe helpers (shared by all cache-once probes) ────────────────────


def ridge_fit(X, y, lam):
    """Closed-form ridge on centered features/targets; returns (w, xb, yb)."""
    xb = X.mean(0)
    yb = float(y.mean())
    Xc = X - xb
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1], dtype=np.float64)
    w = np.linalg.solve(A, Xc.T @ (y - yb))
    return w, xb, yb


def ridge_predict(X, w, xb, yb):
    return (X - xb) @ w + yb


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a).flatten(), np.asarray(b).flatten())[0, 1])


def select_ridge(Xtr, ytr, Xva, yva, lambdas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)):
    """Fit ridge for each lambda, pick the best on val Pearson. Returns (val_r, lam, w, xb, yb)."""
    best = None
    for lam in lambdas:
        w, xb, yb = ridge_fit(Xtr.astype(np.float64), ytr.astype(np.float64), lam)
        vr = pearson(ridge_predict(Xva, w, xb, yb), yva)
        if best is None or vr > best[0]:
            best = (vr, lam, w, xb, yb)
    return best
