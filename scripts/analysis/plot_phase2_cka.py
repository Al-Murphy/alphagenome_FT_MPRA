#!/usr/bin/env python3
"""Styled plot of the Phase 2 encoder CKA depth gradient.

Reads results/cka/phase2_encoder_cka.json (written by phase2_encoder_cka.py)
and renders the per-tap CKA-vs-pretrained curves in the repo's house style
(seaborn white theme, top/right spines removed, frameless legend, the paper
palette, png+pdf at dpi 1200). CPU only — no model load needed.

Usage:
  python scripts/analysis/plot_phase2_cka.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]

# 7 encoder taps, shallow -> deep, with human-readable Stem/Block labels + bp.
TAPS = ["bin_size_1", "bin_size_2", "bin_size_4", "bin_size_8",
        "bin_size_16", "bin_size_32", "bin_size_64"]
TAP_LABELS = ["Stem\n1 bp", "Block 1\n2 bp", "Block 2\n4 bp", "Block 3\n8 bp",
              "Block 4\n16 bp", "Block 5\n32 bp", "Block 6\n64 bp"]

# Colors drawn from the repo palette (paper lentiMPRA/STARR panels).
SERIES = {
    "fly_ft":   {"label": "Fly (S2)",      "color": "#A65141", "marker": "o"},   # terracotta
    "human_ft": {"label": "Human (HepG2)", "color": "#394165", "marker": "s"},   # navy
}


def setup_plot_style():
    sns.set(font_scale=1.2)
    sns.set_style("white")


def save_plots(fig, out_path: Path, dpi=1200, formats=("pdf", "png")):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        f = out_path.with_suffix(f".{fmt}")
        fig.savefig(f, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"  saved {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(REPO / "results/cka/phase2_encoder_cka.json"))
    ap.add_argument("--out", default=str(REPO / "results/cka/phase2_encoder_cka"))
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    cka = data["cka_vs_pretrained"]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = list(range(len(TAPS)))

    # reference: CKA = 1 means representation unchanged by fine-tuning
    ax.axhline(1.0, color="#B0B0B0", lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(len(TAPS) - 1, 1.005, "unchanged", color="#8A8A8A",
            fontsize=10, ha="right", va="bottom")

    for name, sty in SERIES.items():
        if name not in cka:
            continue
        y = [cka[name][t] for t in TAPS]
        ax.plot(x, y, sty["marker"] + "-", color=sty["color"], lw=2.2,
                markersize=8, markeredgecolor="white", markeredgewidth=0.8,
                label=sty["label"], zorder=3)

    # annotate the story: conserved stem vs reorganized deep output (fly)
    if "fly_ft" in cka:
        ax.annotate(f"{cka['fly_ft'][TAPS[0]]:.2f}", (0, cka["fly_ft"][TAPS[0]]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9, color="#A65141")
        ax.annotate(f"{cka['fly_ft'][TAPS[-1]]:.2f}", (len(TAPS) - 1, cka["fly_ft"][TAPS[-1]]),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=9, color="#A65141")

    ax.set_xticks(x)
    ax.set_xticklabels(TAP_LABELS, fontsize=11)
    ax.set_xlabel("Encoder stage  (shallow → deep)", fontsize=14)
    ax.set_ylabel("CKA vs pre-finetuning encoder", fontsize=14)
    ax.set_ylim(0.40, 1.03)
    ax.set_title("Encoder representational drift under fine-tuning", fontsize=15, fontweight="bold")
    # lighter subtitle giving the assays/probe
    ax.text(0.5, 1.005, "Drosophila DeepSTARR vs human LentiMPRA fine-tunes · probed on DeepSTARR test sequences",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="#666666")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=12, title="Fine-tuned on")
    ax.get_legend().get_title().set_fontsize(11)

    fig.tight_layout()
    save_plots(fig, Path(args.out))


if __name__ == "__main__":
    main()
