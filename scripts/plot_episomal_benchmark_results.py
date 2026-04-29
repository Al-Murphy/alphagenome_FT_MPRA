#!/usr/bin/env python3
"""Generate bar plots comparing models on the Gosai episomal MPRA benchmark.

Two layouts (selected with ``--style``):

* ``cell_averaged`` (default) — one panel, three test-set groups, one bar per
  model with sample-size-weighted average across cells. Mirrors the lentiMPRA
  / STARR-seq panels in the paper.
* ``per_cell`` — one panel per test set, one group per cell type, models
  side-by-side. Useful when the per-cell breakdown matters.

Inputs (any one of the following):
  --results_csv PATH   Pre-aggregated CSV with columns
                        [model, regime, cell_type, test_set, pearson_r, seed,
                         (n_samples)]
  --metrics_dir DIR    Directory of *_metrics.json files written by
                        test_episomal_mpra.py (one JSON per checkpoint).
  --bar_final_root DIR Optional ALBench-S2F-style ``bar_final/`` tree for the
                        baseline + AG models — combined with ``--metrics_dir``
                        for the Enformer numbers.
  Defaults: built-in placeholder values so the script runs end-to-end.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Color palette shared with the lentiMPRA / STARR-seq panels in the paper, so
# Enformer/AlphaGenome bars look the same across all three benchmarks.
# Malinois reuses the DeepSTARR baseline color (#E8DCCF) at Alan's request.
MODEL_COLORS = {
    "MPRALegNet":               "#E8DCCF",   # baseline cream (lentiMPRA)
    "DREAM-RNN":                "#8B9DAF",   # blue-gray (STARR-seq baseline)
    "Malinois":                 "#E8DCCF",   # same as DeepSTARR baseline
    "Enf. MPRA (Probing)":      "#E7CDC2",   # light salmon
    "Enf. MPRA (Fine-tuned)":   "#A65141",   # terracotta
    "AG MPRA (Probing)":        "#80A0C7",   # steel blue
    "AG MPRA (Fine-tuned)":     "#394165",   # navy
}

# Darker text annotations on light-colored bars so the value labels stay legible.
TEXT_COLORS = {
    "#E8DCCF": "#8B7D6B",
    "#E7CDC2": "#8B6B61",
    "#8B9DAF": "#5A6A7A",
}

# Per-cell layout uses all 7 models. The cell-averaged layout drops MPRALegNet
# because it shares a color with Malinois (per Alan's color-collision rule).
MODEL_ORDER = [
    "MPRALegNet",
    "DREAM-RNN",
    "Malinois",
    "Enf. MPRA (Probing)",
    "Enf. MPRA (Fine-tuned)",
    "AG MPRA (Probing)",
    "AG MPRA (Fine-tuned)",
]
MODEL_ORDER_CELL_AVG = [
    "DREAM-RNN",
    "Malinois",
    "Enf. MPRA (Probing)",
    "Enf. MPRA (Fine-tuned)",
    "AG MPRA (Probing)",
    "AG MPRA (Fine-tuned)",
]

CELL_ORDER = ["K562", "HepG2", "SKNSH"]
CELL_LABELS = {"K562": "K562", "HepG2": "HepG2", "SKNSH": "SK-N-SH"}
TEST_SET_ORDER = ["reference", "designed", "snv"]
TEST_SET_LABELS = {
    "reference": "Genomic Reference",
    "designed": "High-Activity Designed",
    "snv": "SNV Effects (Δ Pearson)",
}


def setup_plot_style():
    sns.set(font_scale=1.2)
    sns.set_style("white")


def _model_label(model_type: str) -> str:
    mapping = {
        "legnet": "MPRALegNet",
        "dream_rnn": "DREAM-RNN",
        "malinois": "Malinois",
        "enformer_probing": "Enf. MPRA (Probing)",
        "enformer_finetuned": "Enf. MPRA (Fine-tuned)",
        "ag_probing": "AG MPRA (Probing)",
        "ag_finetuned": "AG MPRA (Fine-tuned)",
    }
    return mapping.get(model_type, model_type)


def load_results_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "cell_type", "test_set", "pearson_r"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results_csv missing columns: {missing}")
    return df


def load_results_from_metrics_dir(path: Path) -> pd.DataFrame:
    """Load *_metrics.json files written by test_episomal_mpra.py."""
    rows = []
    for jf in sorted(path.glob("*_metrics.json")):
        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError:
            print(f"  Skipping unreadable JSON: {jf}")
            continue
        model_label = _model_label(data.get("model_type", ""))
        cell_type = data.get("cell_type", "K562")
        # Extract seed from filename if present (e.g. ..._seed42_..._metrics.json)
        seed_match = re.search(r"seed(\d+)", jf.stem)
        seed = int(seed_match.group(1)) if seed_match else 0
        for ts, m in data.get("test_sets", {}).items():
            rows.append({
                "model": model_label,
                "cell_type": cell_type,
                "test_set": ts,
                "pearson_r": m.get("pearson", float("nan")),
                "n_samples": int(m.get("n_samples", 0)) or 0,
                "seed": seed,
                "source_file": str(jf.name),
            })
    return pd.DataFrame(rows)


# ── Aggregator for ALBench-S2F-style bar_final/ result.json files ───────────
#
# Each baseline model in the bar_final layout writes:
#   bar_final/{cell}/{model_subdir}/seed_*/.../result.json
# with two flavours of test_metrics:
#   • flat:        {"in_dist": {...}, "ood": {...}, "snv_abs": {...}, "snv_delta": {...}}
#   • multitask:   {"k562": {...flat...}, "hepg2": {...}, "sknsh": {...}}     (e.g. Malinois)
# The flat case applies to LegNet, DREAM-RNN, AG S1/S2, etc.
# The multitask case nests metrics under the cell key.

# subdir name → display label used in the bar plot.
# (Malinois has multiple training variants. The plain 'malinois' subdir is
# the one that trained successfully on all three cells with positive metrics
# — both 'malinois_paper' and 'malinois_chr_split' produced degenerate /
# negative HepG2 + SKNSH numbers in their result.json files and should be
# avoided.)
DEFAULT_BAR_FINAL_MODELS = {
    "legnet": "MPRALegNet",
    "dream_rnn": "DREAM-RNN",
    "malinois": "Malinois",
    "ag_s1_pred": "AG MPRA (Probing)",
    "ag_s2_real_labels": "AG MPRA (Fine-tuned)",
}


def _flat_metrics_to_rows(model: str, cell: str, seed, tm: dict) -> list[dict]:
    """Map the four expected metric keys onto the plot's three test sets."""
    out = []
    sources = {
        "reference": tm.get("in_dist") or tm.get("in_distribution") or {},
        "designed":  tm.get("ood") or {},
        "snv":       tm.get("snv_delta") or {},
    }
    for ts, m in sources.items():
        val = m.get("pearson_r")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        out.append({"model": model, "cell_type": cell, "test_set": ts,
                    "pearson_r": float(val),
                    "n_samples": int(m.get("n", 0)) or 0,
                    "seed": seed})
    return out


def load_results_from_bar_final(
    root: Path,
    model_subdirs: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate ALBench-S2F bar_final/ result.json files into the plot's schema.

    Handles both the flat test_metrics layout (LegNet, DREAM-RNN, AG, …) and
    the multitask nested-by-cell layout (Malinois) automatically.
    """
    model_subdirs = model_subdirs or DEFAULT_BAR_FINAL_MODELS
    rows = []
    cell_aliases = {"k562": "K562", "hepg2": "HepG2", "sknsh": "SKNSH"}
    for cell_dir, cell_pretty in cell_aliases.items():
        for subdir, label in model_subdirs.items():
            for jf in sorted(
                (root / cell_dir / subdir).rglob("result.json")
            ):
                try:
                    data = json.loads(jf.read_text())
                except json.JSONDecodeError:
                    continue
                seed_m = re.search(r"seed[_]?(\d+)", str(jf))
                seed = int(seed_m.group(1)) if seed_m else data.get("seed", 0)
                tm = data.get("test_metrics", {}) or {}
                # Multitask: nested under cell-name keys.
                if any(k in tm for k in ("k562", "hepg2", "sknsh")):
                    nested = tm.get(cell_dir, {})
                    rows.extend(_flat_metrics_to_rows(label, cell_pretty, seed, nested))
                else:
                    rows.extend(_flat_metrics_to_rows(label, cell_pretty, seed, tm))
    return pd.DataFrame(rows)


def placeholder_data() -> pd.DataFrame:
    """Reasonable placeholders so the script renders before real results land."""
    rows = []
    np.random.seed(0)
    for model in MODEL_ORDER:
        for cell in CELL_ORDER:
            for ts in TEST_SET_ORDER:
                base = {
                    "MPRALegNet": 0.74,
                    "DREAM-RNN": 0.76,
                    "Malinois": 0.78,
                    "Enf. MPRA (Probing)": 0.80,
                    "Enf. MPRA (Fine-tuned)": 0.83,
                    "AG MPRA (Probing)": 0.84,
                    "AG MPRA (Fine-tuned)": 0.87,
                }[model]
                if ts == "designed":
                    base -= 0.10
                elif ts == "snv":
                    base -= 0.40
                rows.append({"model": model, "cell_type": cell, "test_set": ts,
                             "pearson_r": base + np.random.uniform(-0.01, 0.01),
                             "seed": 0})
    return pd.DataFrame(rows)


def plot_panel(ax, df: pd.DataFrame, test_set: str, ylim: tuple[float, float]):
    sub = df[df["test_set"] == test_set]
    width = 0.12
    x_pos = np.arange(len(CELL_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        model_data = sub[sub["model"] == model]
        if model_data.empty:
            continue
        means, stds = [], []
        for cell in CELL_ORDER:
            cell_df = model_data[model_data["cell_type"] == cell]
            if cell_df.empty:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(float(cell_df["pearson_r"].mean()))
                stds.append(float(cell_df["pearson_r"].std(ddof=0))
                            if len(cell_df) > 1 else 0)
        offset = (i - len(MODEL_ORDER) / 2) * width + width / 2
        bars = ax.bar(x_pos + offset, means, width,
                      yerr=stds, label=model,
                      color=MODEL_COLORS[model], alpha=0.9,
                      edgecolor="black", linewidth=1,
                      error_kw=dict(ecolor="black", capsize=2, lw=0.8))
        for bar, val in zip(bars, means):
            if not np.isnan(val) and val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                        color=bar.get_facecolor(), rotation=90)

    ax.set_xlabel("Cell Type", fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title(TEST_SET_LABELS[test_set], fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([CELL_LABELS[c] for c in CELL_ORDER])
    ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.4, linestyle="--")


def plot_episomal_benchmark(df: pd.DataFrame, figsize=(18, 5)):
    setup_plot_style()
    fig, axes = plt.subplots(1, len(TEST_SET_ORDER), figsize=figsize)

    panel_ylim = {
        "reference": (0.5, 1.0),
        "designed": (0.0, 0.9),
        "snv": (0.0, 0.7),
    }

    for ax, ts in zip(axes, TEST_SET_ORDER):
        plot_panel(ax, df, ts, panel_ylim.get(ts, (0.0, 1.0)))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=4, fontsize=10)
    fig.suptitle("Gosai Episomal MPRA Benchmark", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# ── Cell-averaged single-panel plot (lentiMPRA / STARR-seq style) ────────
#
# Bar height  = sample-size-weighted average of per-(cell, seed) Pearson r
#               (weight = n_samples for that cell's test set).
# Error bar   = sample-size-weighted standard deviation across the same set.
#
# Drops MPRALegNet by default because it shares a color with Malinois in the
# repo's palette.

CELL_AVG_GROUP_KEYS = ["reference", "snv", "designed"]
CELL_AVG_GROUP_LABELS = [
    "Genomic Reference\nSequences",
    "SNV Effects",
    "High-Activity\nDesigned Sequences",
]


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Sample-size-weighted mean + Cochran-style reliability-weighted SD."""
    if len(values) == 0:
        return float("nan"), float("nan")
    w = weights.astype(float)
    if w.sum() == 0:
        w = np.ones_like(w)
    wm = float((values * w).sum() / w.sum())
    if len(values) > 1:
        denom = w.sum() - (w ** 2).sum() / w.sum()
        wsd = float(np.sqrt((w * (values - wm) ** 2).sum() / denom)) if denom > 0 else 0.0
    else:
        wsd = 0.0
    return wm, wsd


def plot_episomal_cell_averaged(df: pd.DataFrame, figsize=(9, 6),
                                 ylim=(0.0, 1.0),
                                 model_order=None) -> plt.Figure:
    """Single-panel cell-averaged bar plot (matches the lentiMPRA / STARR-seq
    panel style: 3 test-set groups × 6 model bars per group)."""
    setup_plot_style()
    if model_order is None:
        model_order = MODEL_ORDER_CELL_AVG
    model_order = [m for m in model_order if m in set(df["model"])]
    n_models = len(model_order)
    n_groups = len(CELL_AVG_GROUP_KEYS)
    width = 0.13
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=figsize)
    has_n_samples = "n_samples" in df.columns
    for i, model in enumerate(model_order):
        color = MODEL_COLORS.get(model, "#888888")
        means, stds = [], []
        for ts in CELL_AVG_GROUP_KEYS:
            sub = df[(df["model"] == model) & (df["test_set"] == ts)]
            if sub.empty:
                means.append(0.0); stds.append(0.0)
                continue
            vals = sub["pearson_r"].to_numpy(dtype=float)
            wts = (sub["n_samples"].to_numpy(dtype=float)
                   if has_n_samples else np.ones(len(vals)))
            if (wts == 0).all():
                wts = np.ones_like(wts)
            wm, wsd = _weighted_stats(vals, wts)
            means.append(wm if wm == wm else 0.0)
            stds.append(wsd if wsd == wsd else 0.0)

        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(
            x + offset, means, width,
            yerr=[s if s > 0 else 0 for s in stds],
            capsize=3, label=model, color=color,
            edgecolor="black", linewidth=1, alpha=0.9,
        )
        txt_color = TEXT_COLORS.get(color, color)
        for rect, val, err in zip(bars, means, stds):
            if val > 0:
                cx = rect.get_x() + rect.get_width() / 2.0
                ax.annotate(f"{val:.3f}", xy=(cx, val + err),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=9, fontweight="bold", rotation=90,
                            color=txt_color)

    ax.set_xticks(x)
    ax.set_xticklabels(CELL_AVG_GROUP_LABELS)
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(ylim)
    ax.set_title("Episomal MPRA", fontsize=14)
    ax.yaxis.grid(alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def save_plots(fig, out_path: Path, dpi=1200, formats=("pdf", "png")):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        if fmt == "pdf":
            f = out_path.with_suffix(".pdf")
            fig.savefig(f, format="pdf", bbox_inches="tight")
        elif fmt == "png":
            f = out_path.with_suffix(".png")
            fig.savefig(f, format="png", dpi=dpi, bbox_inches="tight")
        else:
            f = out_path.with_suffix(f".{fmt}")
            fig.savefig(f, format=fmt, dpi=dpi, bbox_inches="tight")
        saved.append(f)
        print(f"✓ Saved {f}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Bar plot for Gosai episomal MPRA benchmark "
                    "(6 models × 3 test sets × 3 cell types)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_csv", type=str, default=None,
                        help="Pre-aggregated CSV with columns "
                             "[model, cell_type, test_set, pearson_r, (seed)]")
    parser.add_argument("--bar_final_root", type=str, default=None,
                        help="Optional path to an ALBench-S2F-style 'bar_final/' "
                             "directory whose result.json files will be aggregated "
                             "for the baseline + AG models. Combine with "
                             "--metrics_dir to pull Enformer FT_MPRA results from "
                             "a separate location.")
    parser.add_argument("--metrics_dir", type=str, default=None,
                        help="Directory of *_metrics.json files from "
                             "test_episomal_mpra.py")
    parser.add_argument("--output_dir", type=str,
                        default="results/comparison_tables/plots/episomal")
    parser.add_argument("--output_name", type=str,
                        default="episomal_benchmark")
    parser.add_argument("--style", type=str, default="cell_averaged",
                        choices=["cell_averaged", "per_cell"],
                        help="Plot layout. 'cell_averaged' = single panel, "
                             "3 test-set groups, sample-size-weighted average "
                             "across cells (matches lentiMPRA / STARR-seq plots). "
                             "'per_cell' = one panel per test set, cells on x-axis.")
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--formats", type=str, nargs="+",
                        default=["pdf", "png"], choices=["pdf", "png", "svg"])
    parser.add_argument("--figsize", type=float, nargs=2,
                        default=None, metavar=("WIDTH", "HEIGHT"),
                        help="Override figure size. Default: (9, 6) for "
                             "cell_averaged, (18, 5) for per_cell.")
    args = parser.parse_args()

    frames = []
    if args.results_csv:
        frames.append(load_results_from_csv(Path(args.results_csv)))
    if args.bar_final_root:
        frames.append(load_results_from_bar_final(Path(args.bar_final_root)))
    if args.metrics_dir:
        frames.append(load_results_from_metrics_dir(Path(args.metrics_dir)))

    if frames:
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        if df.empty:
            print("WARNING: All input sources were empty — using placeholder data.")
            df = placeholder_data()
    else:
        print("INFO: No --results_csv / --bar_final_root / --metrics_dir given "
              "— using placeholder data.")
        df = placeholder_data()

    print(f"Loaded {len(df)} rows across {df['model'].nunique()} models, "
          f"{df['cell_type'].nunique()} cell types, "
          f"{df['test_set'].nunique()} test sets.")

    if args.style == "cell_averaged":
        figsize = tuple(args.figsize) if args.figsize else (9, 6)
        fig = plot_episomal_cell_averaged(df, figsize=figsize)
    else:
        figsize = tuple(args.figsize) if args.figsize else (18, 5)
        fig = plot_episomal_benchmark(df, figsize=figsize)
    out_path = Path(args.output_dir) / args.output_name
    save_plots(fig, out_path, dpi=args.dpi, formats=tuple(args.formats))
    plt.close(fig)


if __name__ == "__main__":
    main()
