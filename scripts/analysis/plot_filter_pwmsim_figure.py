#!/usr/bin/env python3
"""Single figure combining filter PWM-similarity conservation with top full-attribution motifs.

The main panel reproduces the per-filter conservation histogram for pretrained vs
fine-tuned encoders. A compact lower row adds a small motif strip for the top
full-attribution motifs discovered for the fine-tuned fly model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
FILT = REPO / "results/filters"
OUTDIR = REPO / "results/plots"

SERIES = {
    "fly_ft": {"label": "Fly (S2)", "color": "#A65141"},
    "human_ft": {"label": "Human (HepG2)", "color": "#394165"},
}


def setup_plot_style() -> None:
    sns.set(font_scale=1.1)
    sns.set_style("white")


def save_plots(fig, out_path: Path, dpi: int = 1200, formats=("pdf", "png")) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        f = out_path.with_suffix(f".{fmt}")
        fig.savefig(f, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"  saved {f}")


def parse_meme(path: Path):
    motifs = []
    name = None
    rows = []
    collecting = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("MOTIF"):
            if name is not None and rows:
                motifs.append((name, np.array(rows, dtype=float)))
            name = s.split()[1]
            rows = []
            collecting = False
        elif s.startswith("letter-probability"):
            collecting = True
            rows = []
        elif collecting:
            parts = s.split()
            if len(parts) == 4:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    collecting = False
            else:
                collecting = False
    if name is not None and rows:
        motifs.append((name, np.array(rows, dtype=float)))
    return motifs


def draw_logo(ax, ppm, title: str | None = None) -> None:
    ppm = np.asarray(ppm, dtype=float)
    if ppm.ndim != 2 or ppm.shape[1] != 4:
        ax.text(0.5, 0.5, "missing motif", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    x = np.arange(ppm.shape[0])
    colors = {"A": "#4C78A8", "C": "#F58518", "G": "#54A24B", "T": "#E45756"}
    bottom = np.zeros(ppm.shape[0])
    for i, base in enumerate("ACGT"):
        vals = ppm[:, i]
        ax.bar(x, vals, bottom=bottom, width=0.8, color=colors[base], edgecolor="none", alpha=0.9)
        bottom += vals
    ax.set_xlim(-0.5, ppm.shape[0] - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUTDIR / "filter_pwmsim_figure"))
    args = ap.parse_args()
    setup_plot_style()

    n_inf = {}
    sumpath = FILT / "phase1_extract_summary.json"
    if sumpath.exists():
        s = json.loads(sumpath.read_text())
        n_inf = {m: s[m]["n_motifs"] for m in ("pretrained", "fly_ft", "human_ft") if m in s}

    fig = plt.figure(figsize=(10.5, 7.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.65], hspace=0.32, wspace=0.22)
    ax_hist = fig.add_subplot(gs[0, :])
    ax_dev = fig.add_subplot(gs[1, 0])
    ax_hk = fig.add_subplot(gs[1, 1])

    bins = np.linspace(0.0, 1.0, 41)
    summary = []
    for name, sty in SERIES.items():
        npz = FILT / f"conservation_pretrained_vs_{name}.npz"
        if not npz.exists():
            print(f"missing {npz}; run phase1 extraction first")
            continue
        mc = np.load(npz)["matched_corr"]
        mean = float(mc.mean())
        frac_hi = float((mc > 0.9).mean())
        summary.append((name, mean, frac_hi))
        nlab = f"{n_inf[name]} filters · " if name in n_inf else ""
        ax_hist.hist(
            mc,
            bins=bins,
            histtype="stepfilled",
            alpha=0.45,
            color=sty["color"],
            edgecolor=sty["color"],
            linewidth=1.3,
            label=f"{sty['label']}  ({nlab}mean {mean:.2f}, {frac_hi * 100:.0f}% > 0.9)",
            zorder=2,
        )
        ax_hist.axvline(mean, color=sty["color"], ls=(0, (4, 3)), lw=1.5, zorder=3)

    ax_hist.set_xlim(0.2, 1.02)
    ax_hist.set_xlabel("Per-filter best-match correlation vs pre-finetuning", fontsize=12)
    ax_hist.set_ylabel("Number of first-layer filters", fontsize=12)
    ax_hist.set_title("First-layer conv filters: conserved or remodeled?", fontsize=14, fontweight="bold")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.legend(loc="upper left", frameon=False, fontsize=9.5)

    # Compact lower row: top motifs from full attribution analysis for the fine-tuned fly model.
    motif_paths = [
        ("Developmental", REPO / "results/modisco/stage2_dev_motifs.meme"),
        ("Housekeeping", REPO / "results/modisco/stage2_hk_motifs.meme"),
    ]
    for ax, (label, path) in zip((ax_dev, ax_hk), motif_paths):
        ax.set_title(f"Top full-attribution motifs: {label}", fontsize=10.5, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if path.exists():
            motifs = parse_meme(path)
            if motifs:
                name, ppm = motifs[0]
                draw_logo(ax, ppm, title=name)
            else:
                ax.text(0.5, 0.5, f"No motifs in {path.name}", transform=ax.transAxes, ha="center", va="center")
        else:
            ax.text(0.5, 0.5, f"Missing {path.name}", transform=ax.transAxes, ha="center", va="center")

    fig.suptitle("Filter PWM similarity and full-attribution motif recovery", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_plots(fig, Path(args.out))
    for name, mean, frac in summary:
        print(f"  {name}: mean matched corr={mean:.3f}, frac>0.9={frac:.3f}")


if __name__ == "__main__":
    main()
