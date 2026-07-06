"""Plot the plant STARR-seq (Jores 2021) benchmark as grouped bar charts.

Renders the full model x tissue x mode grid (full-finetune + linear-probe test
Pearson r) from the committed reference metrics (or live results) into a clear,
non-overlapping grouped bar figure: rows = {Full finetune, Linear probe},
columns = {leaf, proto}, x-axis = data mode, one bar per model.

USAGE:
    python scripts/plot_plant_starrseq_benchmark_results.py
    python scripts/plot_plant_starrseq_benchmark_results.py --from_live --output_dir results/plant_starrseq/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "results" / "plant_starrseq" / "reference"
LIVE_DIR = REPO_ROOT / "results" / "plant_starrseq"

MODELS = ["ntv3", "alphagenome", "plantcad2", "jores"]
MODEL_LABEL = {
    "ntv3": "NTv3-post",
    "alphagenome": "AlphaGenome",
    "plantcad2": "PlantCAD2",
    "jores": "Jores CNN",
}
MODEL_COLOR = {
    "ntv3": "#4C72B0",
    "alphagenome": "#DD8452",
    "plantcad2": "#55A868",
    "jores": "#C44E52",
}
TISSUES = ["leaf", "proto"]
TISSUE_LABEL = {"leaf": "Leaf", "proto": "Protoplast"}
MODES = ["promoter_only", "enhancer", "combined"]
MODE_LABEL = {"promoter_only": "promoter\nonly", "enhancer": "enhancer", "combined": "combined"}
METHODS = [("finetune", "Full finetune"), ("probe", "Linear probe")]


def _read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _value(model, tissue, mode, method, prefer_live):
    live = LIVE_DIR / model / tissue / mode / method / "metrics.json"
    ref = REFERENCE_DIR / f"{model}_{tissue}_{mode}_{method}.json"
    rec = (_read(live) or _read(ref)) if prefer_live else (_read(ref) or _read(live))
    return None if rec is None else rec.get("test_pearson")


def _panel(ax, method, tissue, prefer_live):
    """Draw one grouped bar panel (models grouped within each data mode)."""
    models = [m for m in MODELS if not (method == "probe" and m == "jores")]
    n = len(models)
    group_w = 0.8
    bar_w = group_w / n
    x = np.arange(len(MODES))

    for i, model in enumerate(models):
        vals = [_value(model, tissue, mode, method, prefer_live) for mode in MODES]
        heights = [v if isinstance(v, (int, float)) else 0.0 for v in vals]
        # center the n bars within each group
        offset = (i - (n - 1) / 2) * bar_w
        bars = ax.bar(x + offset, heights, bar_w * 0.92,
                      color=MODEL_COLOR[model], label=MODEL_LABEL[model],
                      edgecolor="white", linewidth=0.5, zorder=3)
        for rect, v in zip(bars, vals):
            if isinstance(v, (int, float)):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABEL[m] for m in MODES], fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.04)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--from_live", action="store_true",
                        help="Prefer live results/<model>/... over committed reference.")
    parser.add_argument("--output_dir", type=str, default="results/plant_starrseq/plots")
    parser.add_argument("--output_name", type=str, default="plant_starrseq_benchmark")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=True)
    for r, (method, method_label) in enumerate(METHODS):
        for c, tissue in enumerate(TISSUES):
            ax = axes[r][c]
            _panel(ax, method, tissue, args.from_live)
            if r == 0:
                ax.set_title(TISSUE_LABEL[tissue], fontsize=13, fontweight="bold", pad=10)
            if c == 0:
                ax.set_ylabel(f"{method_label}\n\ntest Pearson r", fontsize=11)

    # single shared legend below the figure so it never overlaps the bars
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m]) for m in MODELS]
    labels = [MODEL_LABEL[m] for m in MODELS]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Plant STARR-seq (Jores 2021) benchmark — held-out test Pearson r",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.output_name}.png"
    pdf = out_dir / f"{args.output_name}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
