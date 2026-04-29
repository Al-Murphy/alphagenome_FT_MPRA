#!/usr/bin/env python3
"""Generate bar plots comparing models on the Gosai episomal MPRA benchmark.

Layout: one panel per test set (Reference, Designed, SNV) showing 6 models × 3
cell types (K562, HepG2, SK-N-SH). Pearson correlation on the y-axis. Errorbars
across seeds when multiple seeds are available.

Inputs (any one of the following):
  --results_csv PATH   Pre-aggregated CSV with columns
                        [model, regime, cell_type, test_set, pearson_r, seed]
  --metrics_dir DIR    Directory of *_metrics.json files written by
                        test_episomal_mpra.py (one JSON per checkpoint).
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


# Shared palette across the repo (matches plot_benchmark_results.py)
PALETTE = ["#A65141", "#E7CDC2", "#80A0C7", "#394165", "#B1934A", "#DCA258",
           "#100F14", "#8B9DAF", "#EEDA9D", "#E8DCCF"]


# 6-model schema for the episomal benchmark.
MODEL_ORDER = [
    "MPRALegNet",
    "DREAM-RNN",
    "Malinois",
    "Enf. MPRA (Probing)",
    "Enf. MPRA (Fine-tuned)",
    "AG MPRA (Probing)",
    "AG MPRA (Fine-tuned)",
]

# Color assignments mirror the lentiMPRA plot so the paper has a consistent legend.
MODEL_COLORS = {
    "MPRALegNet":               PALETTE[9],   # #E8DCCF
    "DREAM-RNN":                PALETTE[7],   # #8B9DAF
    "Malinois":                 PALETTE[1],   # #E7CDC2 (light salmon — distinct from Enf. probing)
    "Enf. MPRA (Probing)":      PALETTE[5],   # #DCA258 (gold) so it stays separable from Malinois
    "Enf. MPRA (Fine-tuned)":   PALETTE[0],   # #A65141
    "AG MPRA (Probing)":        PALETTE[2],   # #80A0C7
    "AG MPRA (Fine-tuned)":     PALETTE[3],   # #394165
}

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

# subdir name → display label used in the bar plot
DEFAULT_BAR_FINAL_MODELS = {
    "legnet": "MPRALegNet",
    "dream_rnn": "DREAM-RNN",
    "malinois_paper": "Malinois",
    "ag_s1_pred": "AG MPRA (Probing)",
    "ag_s2_real_labels": "AG MPRA (Fine-tuned)",
}


def _flat_metrics_to_rows(model: str, cell: str, seed, tm: dict) -> list[dict]:
    """Map the four expected metric keys onto the plot's three test sets."""
    out = []
    in_dist = (tm.get("in_dist") or tm.get("in_distribution") or {}).get("pearson_r")
    ood = (tm.get("ood") or {}).get("pearson_r")
    snv_delta = (tm.get("snv_delta") or {}).get("pearson_r")
    for ts, val in (("reference", in_dist), ("designed", ood), ("snv", snv_delta)):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        out.append({"model": model, "cell_type": cell, "test_set": ts,
                    "pearson_r": float(val), "seed": seed})
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
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--formats", type=str, nargs="+",
                        default=["pdf", "png"], choices=["pdf", "png", "svg"])
    parser.add_argument("--figsize", type=float, nargs=2,
                        default=[18, 5], metavar=("WIDTH", "HEIGHT"))
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

    fig = plot_episomal_benchmark(df, figsize=tuple(args.figsize))
    out_path = Path(args.output_dir) / args.output_name
    save_plots(fig, out_path, dpi=args.dpi, formats=tuple(args.formats))
    plt.close(fig)


if __name__ == "__main__":
    main()
