"""Compare models on the AlphaGenome fold-0 held-out test subset.

The fold-0 subset is the set of LentiMPRA test-fold CREs (rev==0, fold==10) whose
hg38 position lies inside AlphaGenome's fold-0 held-out TEST regions — the same
subset the fold-0 AG models are evaluated on.

All baseline prediction CSVs (MPRALegNet, Enformer, AlphaGenome) are written in the
canonical test-fold order (rev==0 -> fold==10 -> reset_index), so this script
subsets them positionally with a length-N boolean mask. Row alignment is asserted
by checking each file's ``actual`` column against the dataset ``mean_value`` order,
so a misaligned/stale file fails loudly instead of producing a wrong number.

Outputs:
  results/test_predictions_fold0/<cell>_fold0_subset.tsv   (seq_id, mean_value, in_fold0_subset)
  results/comparison_tables/mpra_comparison_fold0_subset_detailed.csv
  results/comparison_tables/mpra_comparison_fold0_subset_summary.csv  (Pearson r table)

Usage:
  python scripts/subset_compare_fold0.py
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome_ft_mpra import LentiMPRADataset

CELLS = ["HepG2", "K562", "WTC11"]


def metrics(pred: np.ndarray, act: np.ndarray) -> dict:
    pred = np.asarray(pred, float).ravel()
    act = np.asarray(act, float).ravel()
    mse = float(np.mean((pred - act) ** 2))
    pearson = float(np.corrcoef(pred, act)[0, 1])
    ss_res = float(np.sum((act - pred) ** 2))
    ss_tot = float(np.sum((act - np.mean(act)) ** 2))
    r2 = float(1 - ss_res / ss_tot)
    return {"mse": mse, "pearson": pearson, "r2": r2, "n_samples": len(pred)}


def subset_existing(
    csv_path: Path, mask: np.ndarray, y: np.ndarray, model: str, cell: str,
    plot_model: str, plot_regime: str,
) -> dict | None:
    """Subset a full-test-fold prediction CSV by ``mask`` and recompute metrics."""
    if csv_path is None or not Path(csv_path).exists():
        print(f"  [skip] {model} {cell}: no prediction file")
        return None
    df = pd.read_csv(csv_path)
    if len(df) != len(mask):
        raise ValueError(
            f"{model} {cell}: {csv_path} has {len(df)} rows but canonical test fold has "
            f"{len(mask)} — cannot positionally subset."
        )
    if not np.allclose(df["actual"].to_numpy(float), y, atol=1e-3):
        raise ValueError(
            f"{model} {cell}: 'actual' column does not match dataset mean_value order in "
            f"{csv_path} — row alignment broken; refusing to subset."
        )
    sub = df[mask]
    m = metrics(sub["prediction"], sub["actual"])
    m.update({"model": model, "cell_type": cell, "source": str(csv_path),
              "plot_model": plot_model, "plot_regime": plot_regime})
    print(f"  {model:24s} {cell}: n={m['n_samples']}  Pearson r={m['pearson']:.4f}")
    return m


def resolve_enformer(results_dir: Path, cell: str, finetuned: bool) -> Path | None:
    files = sorted(glob.glob(str(results_dir / "test_predictions" / "enformer" / f"*{cell}_test_predictions.csv")))
    for f in files:
        name = os.path.basename(f)
        if finetuned and "stage2" in name:
            return Path(f)
        if not finetuned and "stage2" not in name:
            return Path(f)
    return None


def resolve_mpralegnet(results_dir: Path, cell: str) -> Path | None:
    files = sorted(glob.glob(str(results_dir / "mpralegnet_predictions" / f"*{cell}_test_predictions.csv")))
    return Path(files[0]) if files else None


def read_ag_fold0_metrics(
    results_dir: Path, cell: str, stage: str, model: str, plot_regime: str, expected_n: int,
) -> dict | None:
    """Read an AG fold-0 model's metrics (already computed on the subset by the eval job).

    The eval job evaluates each stage's checkpoint directly with the fold-0 filter, so the
    saved metrics are already on the subset. stage1 = probing, stage2 = fine-tuned.
    """
    f = results_dir / "test_predictions_fold0" / f"{stage}_{cell}_test_metrics.csv"
    if not f.exists():
        print(f"  [pending] {model} {cell}: {f.name} not found yet (eval job not finished)")
        return None
    m = pd.read_csv(f).iloc[0].to_dict()
    if int(m.get("n_samples", -1)) != expected_n:
        print(f"  [warn] {model} {cell}: n_samples={m.get('n_samples')} != subset size {expected_n}")
    m.update({"model": model, "cell_type": cell, "source": str(f),
              "plot_model": "AG MPRA", "plot_regime": plot_regime})
    print(f"  {model:24s} {cell}: n={int(m['n_samples'])}  Pearson r={float(m['pearson']):.4f}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--filter_version", default="fold_0")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    (results_dir / "test_predictions_fold0").mkdir(parents=True, exist_ok=True)
    (results_dir / "comparison_tables").mkdir(parents=True, exist_ok=True)

    rows = []
    for cell in CELLS:
        print(f"\n=== {cell} ===")
        full = LentiMPRADataset(model=None, cell_type=cell, split="test", ag_test_filter_version=None)
        sub = LentiMPRADataset(model=None, cell_type=cell, split="test", ag_test_filter_version=args.filter_version)
        y = full.data["mean_value"].to_numpy(float)
        ids = full.data["seq_id"].to_numpy()
        sub_ids = set(sub.data["seq_id"])
        mask = np.array([i in sub_ids for i in ids])
        assert mask.sum() == len(sub), f"{cell}: mask {mask.sum()} != subset {len(sub)}"

        # Export the canonical subset definition (full order + membership flag).
        pd.DataFrame({"seq_id": ids, "mean_value": y, "in_fold0_subset": mask}).to_csv(
            results_dir / "test_predictions_fold0" / f"{cell}_fold0_subset.tsv", sep="\t", index=False
        )
        print(f"  subset: {mask.sum()}/{len(mask)} CREs in AG {args.filter_version} held-out TEST regions")

        n = int(mask.sum())
        for m in [
            subset_existing(resolve_mpralegnet(results_dir, cell), mask, y, "MPRALegNet", cell, "MPRALegNet", ""),
            subset_existing(resolve_enformer(results_dir, cell, False), mask, y, "Enformer probing", cell, "Enf. MPRA", "Probing"),
            subset_existing(resolve_enformer(results_dir, cell, True), mask, y, "Enformer fine-tuned", cell, "Enf. MPRA", "Fine-tuned"),
            read_ag_fold0_metrics(results_dir, cell, "stage1", "AG fold-0 probing", "Probing", n),
            read_ag_fold0_metrics(results_dir, cell, "stage2", "AG fold-0 fine-tuned", "Fine-tuned", n),
        ]:
            if m:
                rows.append(m)

    detailed = pd.DataFrame(rows)
    detailed[["model", "cell_type", "n_samples", "pearson", "r2", "mse", "source"]].to_csv(
        results_dir / "comparison_tables" / "mpra_comparison_fold0_subset_detailed.csv", index=False
    )
    summary = (
        detailed.pivot(index="model", columns="cell_type", values="pearson").round(3).reindex(columns=CELLS)
    )
    summary.to_csv(results_dir / "comparison_tables" / "mpra_comparison_fold0_subset_summary.csv")

    # Plot-ready CSV consumable by plot_benchmark_results.py --lentimpra_csv.
    # Select the plot-label source columns first, THEN rename, to avoid a duplicate
    # 'model' column (detailed already has its own descriptive 'model').
    plot_df = detailed[["plot_model", "plot_regime", "cell_type", "pearson"]].rename(
        columns={"plot_model": "model", "plot_regime": "regime", "pearson": "pearson_r"}
    )
    plot_df["regime"] = plot_df["regime"].fillna("")
    plot_csv = results_dir / "comparison_tables" / "mpra_comparison_fold0_subset_plot.csv"
    plot_df.to_csv(plot_csv, index=False)

    print("\n=== fold-0 subset Pearson r ===")
    print(summary.to_string())
    print(f"\nWrote comparison tables -> results/comparison_tables/mpra_comparison_fold0_subset_*.csv")
    print(f"Plot-ready table        -> {plot_csv}")


if __name__ == "__main__":
    main()
