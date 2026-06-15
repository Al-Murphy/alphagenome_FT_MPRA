#!/usr/bin/env python3
"""Styled Phase 1 figure: per-filter TOMTOM match rate vs JASPAR DBs.

The key 'shared-subpart' evidence: the fraction of informative first-layer
filters that match a known TF is ~species-independent — fine-tuning on fly does
not raise the insect-DB match rate above pretrained or human. Reads the TOMTOM
outputs + phase1_extract_summary.json written by phase1_filter_motifs.py.
Repo house style; CPU only.

Usage:
  python scripts/analysis/plot_phase1_tomtom_rate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
FILT = REPO / "results/filters"

MODELS = [  # (key, label, color) — gray = pre-finetuning baseline
    ("pretrained", "Pre-finetuning", "#9E9E9E"),
    ("fly_ft",     "Fly (S2)",       "#A65141"),
    ("human_ft",   "Human (HepG2)",  "#394165"),
]
DBS = [("vertebrate", "Vertebrate JASPAR"), ("fly", "Insect JASPAR")]


def setup_plot_style():
    sns.set(font_scale=1.2)
    sns.set_style("white")


def save_plots(fig, out_path: Path, dpi=1200, formats=("pdf", "png")):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        f = out_path.with_suffix(f".{fmt}")
        fig.savefig(f, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"  saved {f}")


def _filters_with_hit(tsv: Path) -> int:
    if not tsv.exists():
        return 0
    qs = set()
    for ln in tsv.read_text().splitlines():
        if not ln or ln.startswith("#") or ln.startswith("Query_ID"):
            continue
        q = ln.split("\t")[0]
        if q:
            qs.add(q)
    return len(qs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(FILT / "phase1_tomtom_rate"))
    args = ap.parse_args()

    n_motifs = {m: json.loads((FILT / "phase1_extract_summary.json").read_text())[m]["n_motifs"]
                for m, _, _ in MODELS}

    # rate[model][db] = filters-with-hit / informative filters (%)
    rate = {}
    for m, _, _ in MODELS:
        rate[m] = {}
        for db, _ in DBS:
            hits = _filters_with_hit(FILT / f"tomtom_{m}_vs_{db}" / "tomtom.tsv")
            rate[m][db] = 100.0 * hits / max(1, n_motifs[m])

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = np.arange(len(DBS))
    w = 0.26
    for i, (m, label, color) in enumerate(MODELS):
        vals = [rate[m][db] for db, _ in DBS]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=color, edgecolor="black",
                      linewidth=1, alpha=0.9, label=f"{label}  (n={n_motifs[m]})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in DBS], fontsize=13)
    ax.set_ylabel("Informative filters matching a known TF (%)", fontsize=13)
    ax.set_ylim(0, 70)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=11, title="First-layer filters from")
    ax.get_legend().get_title().set_fontsize(10.5)

    fig.suptitle("First-layer filters match known TFs at a species-independent rate",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.925,
             "TOMTOM of stem (1 bp) filter PWMs vs JASPAR · % of informative filters with ≥1 hit (q<0.1)",
             ha="center", va="top", fontsize=9.5, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_plots(fig, Path(args.out))
    for m, _, _ in MODELS:
        print(f"  {m}: " + ", ".join(f"{db}={rate[m][db]:.1f}%" for db, _ in DBS))


if __name__ == "__main__":
    main()
