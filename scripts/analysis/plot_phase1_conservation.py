#!/usr/bin/env python3
"""Styled Phase 1 figure: per-filter conservation of first-layer conv filters.

Reads the Hungarian-matching outputs written by phase1_filter_motifs.py
(results/filters/conservation_pretrained_vs_{fly_ft,human_ft}.npz) and plots
the distribution of each filter's best-match correlation to its pretrained
counterpart — fly (S2) vs human (HepG2). Low = the first-layer detector was
remodeled by fine-tuning; ~1 = conserved. Repo house style; CPU only.

Usage:
  python scripts/analysis/plot_phase1_conservation.py
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

SERIES = {
    "fly_ft":   {"label": "Fly (S2)",      "color": "#A65141"},   # terracotta
    "human_ft": {"label": "Human (HepG2)", "color": "#394165"},   # navy
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
    ap.add_argument("--out", default=str(FILT / "phase1_filter_conservation"))
    args = ap.parse_args()

    # informative first-layer filter counts (TOMTOM query set) per model
    n_inf = {}
    sumpath = FILT / "phase1_extract_summary.json"
    if sumpath.exists():
        s = json.loads(sumpath.read_text())
        n_inf = {m: s[m]["n_motifs"] for m in ("pretrained", "fly_ft", "human_ft") if m in s}

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bins = np.linspace(0.0, 1.0, 41)

    summary = []
    for name, sty in SERIES.items():
        npz = FILT / f"conservation_pretrained_vs_{name}.npz"
        if not npz.exists():
            print(f"missing {npz}; run phase1 --mode extract first")
            continue
        mc = np.load(npz)["matched_corr"]
        mean = float(mc.mean())
        frac_hi = float((mc > 0.9).mean())
        summary.append((name, mean, frac_hi))
        nlab = f"{n_inf[name]} filters · " if name in n_inf else ""
        ax.hist(mc, bins=bins, histtype="stepfilled", alpha=0.5,
                color=sty["color"], edgecolor=sty["color"], linewidth=1.4,
                label=f"{sty['label']}  ({nlab}mean {mean:.2f}, {frac_hi*100:.0f}% > 0.9)",
                zorder=2)
        ax.axvline(mean, color=sty["color"], ls=(0, (4, 3)), lw=1.6, zorder=3)

    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("Per-filter best-match correlation vs pre-finetuning", fontsize=14)
    ax.set_ylabel("Number of first-layer filters", fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5, title="Fine-tuned on")
    ax.get_legend().get_title().set_fontsize(11)

    fig.suptitle("First-layer conv filters: conserved or remodeled?",
                 fontsize=15, fontweight="bold", y=0.99)
    sub = "Hungarian-matched stem (1 bp) filters · probed on DeepSTARR test sequences"
    if {"pretrained", "fly_ft", "human_ft"} <= set(n_inf):
        sub += (f"\nInformative filters: pretrained {n_inf['pretrained']} → "
                f"fly {n_inf['fly_ft']} / human {n_inf['human_ft']} "
                f"(fly recruits ~{n_inf['fly_ft'] - n_inf['pretrained']} more)")
    fig.text(0.5, 0.93, sub, ha="center", va="top", fontsize=9.5, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_plots(fig, Path(args.out))
    for name, mean, frac in summary:
        print(f"  {name}: mean matched corr={mean:.3f}, frac>0.9={frac:.3f}")


if __name__ == "__main__":
    main()
