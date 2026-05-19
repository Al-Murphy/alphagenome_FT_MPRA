"""Enformer FT proper-eval — FAST version (bs=16, bf16).

Same panels as eval_enformer_proper_panels.py: 32k Ref + 32k Alt + 22k Designed.
Increases batch_size from 4 → 16 (4x speedup) and uses bf16 autocast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, str(REPO))

from experiments.train_foundation_stage2 import _forward_enformer, _predict_test_sequences

CELL_UC = {"k562": "K562", "hepg2": "HepG2", "sknsh": "SKNSH"}


def _safe(a, b):
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return float(pearsonr(a[m], b[m])[0]) if m.sum() >= 3 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["k562", "hepg2", "sknsh"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    from enformer_pytorch import Enformer
    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = {}

    encoder = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    enc_path = Path(args.ckpt).parent / "best_encoder.pt"
    if enc_path.exists():
        es = torch.load(enc_path, map_location="cpu", weights_only=False)
        if isinstance(es, dict) and "model_state_dict" in es:
            es = es["model_state_dict"]
        encoder.load_state_dict(es, strict=False)
        print(f"Encoder loaded from {enc_path}")
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    from experiments.train_foundation_cached import MLPHead
    head = MLPHead(3072, hidden_dim=512, dropout=0.1)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["model_state_dict"])
    head.to(device).eval()

    def pred(seqs):
        return _predict_test_sequences(
            encoder, head, _forward_enformer, seqs, device,
            batch_size=args.batch_size,
            amp_dtype=torch.bfloat16,
            use_amp=True,
        )

    metrics = {"cell": args.cell, "batch_size": args.batch_size}
    lcol = f"{CELL_UC[args.cell]}_log2FC"

    # Full chr 7+13 (Ref + Alt combined, ~65k)
    print(f"[1/2] Full chr 7+13 (Ref + Alt) bs={args.batch_size}...")
    import time
    t0 = time.time()
    df = pd.read_csv(REPO / "data/k562/test_sets/test_chr7_13_all.tsv", sep="\t")
    seqs = df["sequence"].tolist()
    labels = df[lcol].to_numpy(dtype=np.float32)
    preds = pred(seqs)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({len(seqs)/dt:.0f} seqs/s)")

    ref_mask = df["allele"].values == "R"
    alt_mask = df["allele"].values == "A"
    metrics["Reference"] = {"pearson_r": _safe(preds[ref_mask], labels[ref_mask]),
                            "n": int(ref_mask.sum())}
    metrics["Alt"] = {"pearson_r": _safe(preds[alt_mask], labels[alt_mask]),
                      "n": int(alt_mask.sum())}
    np.savez_compressed(out_dir / "predictions_full_chr.npz",
                        preds=preds, labels=labels, allele=df["allele"].values)
    print(f"  Reference: {metrics['Reference']['pearson_r']:.4f}  "
          f"Alt: {metrics['Alt']['pearson_r']:.4f}")

    # Designed/OOD
    print("[2/2] Designed/OOD...")
    t0 = time.time()
    ood_path = REPO / f"data/k562/test_sets/test_ood_designed_{args.cell}.tsv"
    if not ood_path.exists():
        ood_path = REPO / f"data/{args.cell}/test_sets/test_ood_designed_{args.cell}.tsv"
    if ood_path.exists():
        ood = pd.read_csv(ood_path, sep="\t")
        ood_col = lcol if lcol in ood.columns else "K562_log2FC"
        ood_seqs = ood["sequence"].tolist()
        ood_labels = ood[ood_col].to_numpy(dtype=np.float32)
        ood_preds = pred(ood_seqs)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s ({len(ood_seqs)/dt:.0f} seqs/s)")
        metrics["Designed"] = {"pearson_r": _safe(ood_preds, ood_labels),
                                "n": int(len(ood))}
        np.savez_compressed(out_dir / "predictions_designed.npz",
                            preds=ood_preds, labels=ood_labels)
        print(f"  Designed: {metrics['Designed']['pearson_r']:.4f}")
    else:
        print(f"  Designed file not found: {ood_path}")
        metrics["Designed"] = {"pearson_r": float("nan"), "n": 0}

    (out_dir / "proper_eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
