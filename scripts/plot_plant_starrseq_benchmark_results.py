#!/usr/bin/env python3
"""
Plant STARR-seq (Jores 2021) benchmark bar plot.

Held-out test Pearson r on the combined data mode, by tissue (Leaf, Proto), for
five models — Jores CNN, PlantCaduceus (l32), PlantCAD2, NTv3-post, AlphaGenome —
comparing stage-1 probing (frozen backbone) vs stage-2 fine-tuning. The Jores CNN
is a from-scratch baseline (fine-tuned only, no probing).

Style mirrors ``scripts/plot_benchmark_results.py`` (seaborn white, grouped bars,
black bar edges, rotated bold value labels, dashed y-grid, palette ``pal``). Data
values are hardcoded here (same convention as ``plot_benchmark_results.py``).

Output: results/plant_starrseq/plots/plant_starrseq_benchmark.png (PNG only).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_plot_style():
    sns.set(font_scale=1.2)
    sns.set_style("white")
    # same palette as scripts/plot_benchmark_results.py / plot_cagi5_results.py
    pal = ["#A65141", "#E7CDC2", "#80A0C7", "#394165", "#B1934A",
           "#DCA258", "#100F14", "#8B9DAF", "#EEDA9D", "#E8DCCF"]
    return pal


def load_plant_data():
    """Combined-mode held-out test Pearson r. Probing = stage-1 (frozen backbone,
    trained head); Fine-tuned = stage-2 (unfrozen)."""
    data = [
        # Jores CNN — from-scratch baseline (fine-tuned only)
        {"model": "Jores CNN", "regime": "Fine-tuned", "tissue": "Leaf",  "pearson_r": 0.806},
        {"model": "Jores CNN", "regime": "Fine-tuned", "tissue": "Proto", "pearson_r": 0.785},
        # PlantCaduceus (l32)
        {"model": "PlantCaduceus", "regime": "Probing",    "tissue": "Leaf",  "pearson_r": 0.7969},
        {"model": "PlantCaduceus", "regime": "Fine-tuned", "tissue": "Leaf",  "pearson_r": 0.8902},
        {"model": "PlantCaduceus", "regime": "Probing",    "tissue": "Proto", "pearson_r": 0.8133},
        {"model": "PlantCaduceus", "regime": "Fine-tuned", "tissue": "Proto", "pearson_r": 0.8846},
        # PlantCAD2
        {"model": "PlantCAD2", "regime": "Probing",    "tissue": "Leaf",  "pearson_r": 0.7745},
        {"model": "PlantCAD2", "regime": "Fine-tuned", "tissue": "Leaf",  "pearson_r": 0.8927},
        {"model": "PlantCAD2", "regime": "Probing",    "tissue": "Proto", "pearson_r": 0.7735},
        {"model": "PlantCAD2", "regime": "Fine-tuned", "tissue": "Proto", "pearson_r": 0.8481},
        # NTv3-post
        {"model": "NTv3-post", "regime": "Probing",    "tissue": "Leaf",  "pearson_r": 0.8420},
        {"model": "NTv3-post", "regime": "Fine-tuned", "tissue": "Leaf",  "pearson_r": 0.8834},
        {"model": "NTv3-post", "regime": "Probing",    "tissue": "Proto", "pearson_r": 0.8449},
        {"model": "NTv3-post", "regime": "Fine-tuned", "tissue": "Proto", "pearson_r": 0.8746},
        # AlphaGenome
        {"model": "AlphaGenome", "regime": "Probing",    "tissue": "Leaf",  "pearson_r": 0.8294},
        {"model": "AlphaGenome", "regime": "Fine-tuned", "tissue": "Leaf",  "pearson_r": 0.8899},
        {"model": "AlphaGenome", "regime": "Probing",    "tissue": "Proto", "pearson_r": 0.8372},
        {"model": "AlphaGenome", "regime": "Fine-tuned", "tissue": "Proto", "pearson_r": 0.8795},
    ]
    df = pd.DataFrame(data)
    df["model_label"] = df.apply(lambda r: f"{r['model']} ({r['regime']})", axis=1)
    return df


# left-to-right order: Jores CNN -> PlantCaduceus -> PlantCAD2 -> NTv3-post -> AlphaGenome
MODEL_ORDER = [
    "Jores CNN (Fine-tuned)",
    "PlantCaduceus (Probing)", "PlantCaduceus (Fine-tuned)",
    "PlantCAD2 (Probing)", "PlantCAD2 (Fine-tuned)",
    "NTv3-post (Probing)", "NTv3-post (Fine-tuned)",
    "AlphaGenome (Probing)", "AlphaGenome (Fine-tuned)",
]


def model_colors(pal):
    # Jores CNN = #E8DCCF (pal[9]); each other model gets a light(probing)/dark(fine-tuned) pair.
    return {
        "Jores CNN (Fine-tuned)": pal[9],           # #E8DCCF
        "PlantCaduceus (Probing)": pal[8], "PlantCaduceus (Fine-tuned)": pal[5],
        "PlantCAD2 (Probing)": pal[7], "PlantCAD2 (Fine-tuned)": pal[4],
        "NTv3-post (Probing)": pal[1], "NTv3-post (Fine-tuned)": pal[0],
        "AlphaGenome (Probing)": pal[2], "AlphaGenome (Fine-tuned)": pal[3],
    }


def plot_plant_benchmark(dat, pal, figsize=(13, 6.5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = model_colors(pal)
    tissue_order = ["Leaf", "Proto"]

    x_pos = np.arange(len(tissue_order))
    width = 0.092  # 9 bars per group — narrow enough that groups never overlap

    for i, model in enumerate(MODEL_ORDER):
        model_data = dat[dat["model_label"] == model]
        if len(model_data) == 0:
            continue
        values = []
        for tissue in tissue_order:
            td = model_data[model_data["tissue"] == tissue]
            values.append(td["pearson_r"].values[0] if len(td) > 0 else 0)

        offset = (i - len(MODEL_ORDER) / 2) * width + width / 2
        bars = ax.bar(x_pos + offset, values, width, label=model,
                      color=colors[model], alpha=0.9, edgecolor="black", linewidth=1)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0025,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=bar.get_facecolor(), rotation=90)

    ax.set_xlabel("Tissue", fontsize=12)
    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title("Plant STARR-seq (Jores 2021) — combined mode, held-out test", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tissue_order)
    ax.set_ylim([0.5, 1])
    ax.grid(axis="y", alpha=0.5, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9, ncol=3)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--output_dir", type=str, default="results/plant_starrseq/plots")
    parser.add_argument("--output_name", type=str, default="plant_starrseq_benchmark")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    pal = setup_plot_style()
    fig = plot_plant_benchmark(load_plant_data(), pal)

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.output_name}.png"
    fig.savefig(png, format="png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
