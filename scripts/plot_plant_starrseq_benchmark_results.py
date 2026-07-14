#!/usr/bin/env python3
"""
Plant STARR-seq (Jores 2021) benchmark bar plots.

Held-out test Pearson r for five models — Jores CNN, PlantCaduceus (l32),
PlantCAD2, NTv3-post, AlphaGenome — comparing stage-1 probing (frozen backbone)
vs stage-2 fine-tuning. The Jores CNN is a from-scratch baseline (fine-tuned
only, no probing).

Three figures:
  * plant_starrseq_benchmark.png                 — combined mode, all 5 models.
  * plant_starrseq_benchmark_no_plantcaduceus.png — combined mode, 4 models.
  * plant_starrseq_benchmark_modes.png           — 6 panels (tissue x mode),
                                                    PlantCaduceus excluded.

Values are read from results/plant_starrseq/reference/*.json (test_pearson), so
the figures always track the committed metrics.

NOTE — two caveats about PlantCaduceus, both of which are why it gets its own
figure variants rather than being shown alongside the others unqualified:

1. It only ever ran the **combined** mode. There are no enhancer / promoter-only
   runs for it, so it is dropped from the per-mode figure entirely rather than
   sitting empty in 4 of the 6 panels.

2. Its "probing" number is **not the same quantity** as everyone else's. For
   AlphaGenome, NTv3 and PlantCAD2, "probing" means a closed-form ridge on
   mean-pooled frozen encoder features (`probe_head.npz`). For PlantCaduceus it is
   a *trained attention-pool MLP head* on a frozen backbone (a stage-1 `best.pt`) —
   a much more expressive head. Its probing bar (0.7969 leaf / 0.8133 proto, the
   highest in the figure) therefore overstates its frozen-backbone quality relative
   to the other three, and the two are not directly comparable.

Style mirrors ``scripts/plot_benchmark_results.py`` (seaborn white, grouped bars,
black bar edges, rotated bold value labels, dashed y-grid).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "results" / "plant_starrseq" / "reference"

MODELS = ["jores", "plantcaduceus", "plantcad2", "ntv3", "alphagenome"]
MODEL_LABEL = {
    "jores": "Jores CNN",
    "plantcaduceus": "PlantCaduceus",
    "plantcad2": "PlantCAD2",
    "ntv3": "NTv3-post",
    "alphagenome": "AlphaGenome",
}
TISSUES = ["leaf", "proto"]
TISSUE_LABEL = {"leaf": "Leaf", "proto": "Proto"}
MODES = ["combined", "enhancer", "promoter_only"]
MODE_LABEL = {"combined": "Combined", "enhancer": "Enhancer", "promoter_only": "Promoter-only"}
METHODS = [("probe", "Probing"), ("finetune", "Fine-tuned")]

# One light (probing) / dark (fine-tuned) pair per model, drawn from the repo
# palette so Enformer/AlphaGenome bars match the lentiMPRA and episomal panels.
# One pair (sage) sits outside the base palette: the only free pair left in `pal`
# was a blue-gray that reads too close to AlphaGenome's steel blue.
MODEL_COLORS = {
    "Jores CNN (Fine-tuned)":      "#D6D1C7",   # grey-cream baseline
    "PlantCaduceus (Probing)":     "#B5C4A5",   # light sage
    "PlantCaduceus (Fine-tuned)":  "#5A6B4A",   # forest
    "PlantCAD2 (Probing)":         "#EEDA9D",   # light gold
    "PlantCAD2 (Fine-tuned)":      "#DCA258",   # orange
    "NTv3-post (Probing)":         "#EFC4C0",   # pink salmon
    "NTv3-post (Fine-tuned)":      "#A65141",   # terracotta
    "AlphaGenome (Probing)":       "#80A0C7",   # steel blue
    "AlphaGenome (Fine-tuned)":    "#394165",   # navy
}

# Value labels are drawn in the bar's own colour; on the light bars that is
# unreadable, so those get a darkened stand-in.
TEXT_COLORS = {
    "#D6D1C7": "#7E7869",
    "#EEDA9D": "#8F7A3E",
    "#B5C4A5": "#5A6B4A",
    "#EFC4C0": "#A65141",
}

MODEL_ORDER = [
    "Jores CNN (Fine-tuned)",
    "PlantCaduceus (Probing)", "PlantCaduceus (Fine-tuned)",
    "PlantCAD2 (Probing)", "PlantCAD2 (Fine-tuned)",
    "NTv3-post (Probing)", "NTv3-post (Fine-tuned)",
    "AlphaGenome (Probing)", "AlphaGenome (Fine-tuned)",
]


def setup_plot_style():
    sns.set(font_scale=1.2)
    sns.set_style("white")


def load_plant_data():
    """Held-out test Pearson r for every model x tissue x mode x method that has a
    committed reference metric. PlantCaduceus only ran the combined mode, and the
    Jores CNN is fine-tuned only, so those cells are simply absent."""
    rows = []
    for model in MODELS:
        for tissue in TISSUES:
            for mode in MODES:
                for method, regime in METHODS:
                    path = REFERENCE_DIR / f"{model}_{tissue}_{mode}_{method}.json"
                    if not path.exists():
                        continue
                    metrics = json.loads(path.read_text())
                    rows.append({
                        "model": MODEL_LABEL[model],
                        "regime": regime,
                        "tissue": TISSUE_LABEL[tissue],
                        "mode": mode,
                        "pearson_r": metrics["test_pearson"],
                    })
    df = pd.DataFrame(rows)
    df["model_label"] = df["model"] + " (" + df["regime"] + ")"
    return df


def _label_bars(ax, bars, values, color, fontsize=9):
    text_color = TEXT_COLORS.get(color, color)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0025,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=fontsize,
                    fontweight="bold", color=text_color, rotation=90)


def plot_plant_benchmark(dat, figsize=(13, 6.5), drop_models=(), compact=False):
    """Combined mode only, grouped by tissue — the headline figure.

    ``compact`` narrows everything (fonts, labels, legend) so the figure holds up when
    dropped into a multi-panel layout at subplot width rather than shown full-bleed.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    label_fs = 6 if compact else 9
    legend_fs = 7 if compact else 9
    legend_ncol = 3
    axis_fs = 8 if compact else 12
    title_fs = 9 if compact else 14
    dat = dat[(dat["mode"] == "combined") & (~dat["model"].isin(drop_models))]
    tissue_order = ["Leaf", "Proto"]

    order = [m for m in MODEL_ORDER
             if not any(m.startswith(d + " (") for d in drop_models)]

    x_pos = np.arange(len(tissue_order))
    width = 0.092 * 9 / max(len(order), 1)  # keep total group width constant

    for i, model in enumerate(order):
        model_data = dat[dat["model_label"] == model]
        if len(model_data) == 0:
            continue
        values = []
        for tissue in tissue_order:
            td = model_data[model_data["tissue"] == tissue]
            values.append(td["pearson_r"].values[0] if len(td) > 0 else 0)

        offset = (i - len(order) / 2) * width + width / 2
        color = MODEL_COLORS[model]
        bars = ax.bar(x_pos + offset, values, width, label=model,
                      color=color, alpha=0.9, edgecolor="black",
                      linewidth=0.6 if compact else 1)
        _label_bars(ax, bars, values, color, fontsize=label_fs)

    ax.set_xlabel("Tissue", fontsize=axis_fs)
    ax.set_ylabel("Pearson Correlation", fontsize=axis_fs)
    title = ("Plant STARR-seq — combined mode" if compact
             else "Plant STARR-seq (Jores 2021) — combined mode, held-out test")
    # in compact mode the legend sits between the title and the axes, so pad the title
    # clear of it (7 entries over 3 columns = 3 legend rows)
    ax.set_title(title, fontsize=title_fs, pad=42 if compact else 6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if compact:
        # seaborn's default spine/tick weight is too heavy once the figure is scaled
        # down to subplot width
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tissue_order, fontsize=axis_fs)
    ax.tick_params(axis="y", labelsize=axis_fs)
    ax.set_ylim([0.5, 1])
    ax.grid(axis="y", alpha=0.5, linestyle="--")
    if compact:
        # at subplot width an in-axes legend collides with the value labels, so it sits
        # above the axes instead (the title is padded clear of it)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False,
                  fontsize=legend_fs, ncol=legend_ncol,
                  columnspacing=0.8, handlelength=1.2, handletextpad=0.4)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False,
                  fontsize=legend_fs, ncol=legend_ncol,
                  columnspacing=0.8, handlelength=1.2, handletextpad=0.4)

    plt.tight_layout()
    return fig


def plot_plant_benchmark_modes(dat, figsize=(18, 9), drop_models=()):
    """Six panels: tissue (rows) x mode (cols). Within a panel, x is the model and
    each model shows its probing bar next to its fine-tuned bar."""
    fig, axes = plt.subplots(len(TISSUES), len(MODES), figsize=figsize, sharey=True)
    model_names = [MODEL_LABEL[m] for m in MODELS
                   if MODEL_LABEL[m] not in drop_models]
    width = 0.38

    for row, tissue in enumerate(TISSUES):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            panel = dat[(dat["tissue"] == TISSUE_LABEL[tissue]) & (dat["mode"] == mode)]

            for i, model in enumerate(model_names):
                md = panel[panel["model"] == model]
                regimes = [r for _, r in METHODS if r in set(md["regime"])]
                # Jores CNN has no probing run, so its single bar is centred.
                offsets = {"Probing": -width / 2, "Fine-tuned": width / 2}
                if regimes == ["Fine-tuned"]:
                    offsets = {"Fine-tuned": 0.0}

                for regime in regimes:
                    val = md[md["regime"] == regime]["pearson_r"].values[0]
                    label = f"{model} ({regime})"
                    color = MODEL_COLORS[label]
                    bars = ax.bar(i + offsets[regime], [val], width,
                                  color=color, alpha=0.9, edgecolor="black", linewidth=1)
                    _label_bars(ax, bars, [val], color, fontsize=8)

            ax.set_title(f"{TISSUE_LABEL[tissue]} — {MODE_LABEL[mode]}", fontsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks(np.arange(len(model_names)))
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
            ax.set_ylim([0.5, 1])
            ax.grid(axis="y", alpha=0.5, linestyle="--")
            if col == 0:
                ax.set_ylabel("Pearson Correlation", fontsize=12)

    # One shared legend: light = probing, dark = fine-tuned, per model.
    legend_order = [m for m in MODEL_ORDER
                    if not any(m.startswith(d + " (") for d in drop_models)]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m], edgecolor="black",
                             linewidth=1, alpha=0.9, label=m) for m in legend_order]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Plant STARR-seq (Jores 2021) — held-out test, by tissue and data mode",
                 fontsize=15)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--output_dir", type=str, default="results/plant_starrseq/plots")
    parser.add_argument("--output_name", type=str, default="plant_starrseq_benchmark")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    setup_plot_style()
    dat = load_plant_data()

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # PlantCaduceus only ever ran the combined mode, so it is dropped from the
    # per-mode figure entirely rather than leaving it absent from 4 of 6 panels. The
    # combined figure is rendered both with and without it (see NOTE in the docstring:
    # its "probing" bar is not a ridge probe like the others).
    figures = [
        ("", plot_plant_benchmark(dat)),
        # narrow: this variant is meant to sit as a subplot in a multi-panel figure
        ("_no_plantcaduceus", plot_plant_benchmark(
            dat, figsize=(5, 4.5), drop_models=("PlantCaduceus",), compact=True)),
        ("_modes", plot_plant_benchmark_modes(dat, drop_models=("PlantCaduceus",))),
    ]
    for suffix, fig in figures:
        png = out_dir / f"{args.output_name}{suffix}.png"
        fig.savefig(png, format="png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
