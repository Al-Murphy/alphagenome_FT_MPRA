#!/usr/bin/env python3
"""Collate benchmark results from all cell types into summary tables (test and validation).

Reads per-cell CSVs:
  - Test:       results/benchmark_results_{HepG2,K562,WTC11}.csv
  - Validation: results/benchmark_val_results_{HepG2,K562,WTC11}.csv

Usage:
    python scripts/collate_benchmark_results.py
    python scripts/collate_benchmark_results.py --results_dir results/ --output results/benchmark_summary.csv
"""

import argparse
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class SplitConfig:
    """One result split (e.g. test vs validation) with matching CSV column names."""

    file_stem: str  # benchmark_results or benchmark_val_results
    pearson_col: str
    loss_col: str
    epoch_col: str
    output_suffix: str  # "" for test -> *_stage1.csv; "_val" -> *_val_stage1.csv
    label: str  # for console banners


TEST_SPLIT = SplitConfig(
    file_stem='benchmark_results',
    pearson_col='best_test_pearson',
    loss_col='best_test_loss',
    epoch_col='best_test_epoch',
    output_suffix='',
    label='TEST',
)

VAL_SPLIT = SplitConfig(
    file_stem='benchmark_val_results',
    pearson_col='best_val_pearson',
    loss_col='best_val_loss',
    epoch_col='best_val_epoch',
    output_suffix='_val',
    label='VALIDATION',
)


def extract_hyperparameter(run_name: str) -> Tuple[str, str]:
    """Extract the hyperparameter being tested from run name and training mode.
    
    Examples:
        'benchmark-HepG2-baseline-default' -> ('baseline-default', 'stage1')
        'benchmark-K562-nl-512-256' -> ('nl-512-256', 'stage1')
        'benchmark-WTC11-do-0.3' -> ('do-0.3', 'stage1')
        'benchmark-HepG2-s2-baseline-es' -> ('s2-baseline-es', 'stage2')
        'benchmark-K562-s2-s1ep20' -> ('s2-s1ep20', 'stage2')
    
    Returns:
        Tuple of (hyperparameter_name, training_mode)
        where training_mode is either 'stage1' or 'stage2'
    """
    parts = run_name.split('-')
    # Skip 'benchmark' and cell type (e.g., 'HepG2')
    if len(parts) >= 3:
        hyperparam = '-'.join(parts[2:])
        # Check if this is a two-stage training run (starts with 's2-')
        if hyperparam.startswith('s2-'):
            return hyperparam, 'stage2'
        else:
            return hyperparam, 'stage1'
    return run_name, 'stage1'


def load_and_process_results(
    results_dir: Path, split: SplitConfig, *, required: bool
) -> pd.DataFrame:
    """Load per-cell CSVs for one split, dedupe, and add hyperparameter columns."""
    all_results = []

    for cell_type in ['HepG2', 'K562', 'WTC11']:
        results_file = results_dir / f'{split.file_stem}_{cell_type}.csv'
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping")
            continue

        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} results from {results_file}")
        all_results.append(df)

    if not all_results:
        if required:
            raise FileNotFoundError(
                f"No {split.file_stem}_*.csv files found under {results_dir}"
            )
        print(f"No {split.label} benchmark files found; skipping {split.label} summaries.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Same run_name can appear multiple times: training appends to CSV on reruns, or merged WandB exports.
    # Keep the last row per run_name (append order = most recent job last).
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=['run_name'], keep='last')
    n_dropped = n_before - len(combined)
    if n_dropped:
        print(f"[{split.label}] Dropped {n_dropped} duplicate run_name row(s); kept last occurrence (most recent).")

    # Extract hyperparameter name and training mode
    combined[['hyperparameter', 'training_mode']] = combined['run_name'].apply(
        lambda x: pd.Series(extract_hyperparameter(x))
    )

    # If two different run_names map to the same hyperparameter (unusual), keep the better Pearson.
    combined = combined.sort_values(split.pearson_col, ascending=False)
    combined = combined.drop_duplicates(subset=['cell_type', 'hyperparameter', 'training_mode'], keep='first')

    return combined


def create_summary_table(df: pd.DataFrame, pearson_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create pivot tables showing performance across cell types for stage 1 and stage 2."""
    stage1_df = df[df['training_mode'] == 'stage1'].copy()
    stage2_df = df[df['training_mode'] == 'stage2'].copy()

    def create_pivot(data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return pd.DataFrame()

        pivot = data.pivot_table(
            index='hyperparameter',
            columns='cell_type',
            values=pearson_col,
            aggfunc='first',
        )

        col_order = ['HepG2', 'K562', 'WTC11']
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        pivot['Average'] = pivot.mean(axis=1)
        pivot['Rank'] = pivot['Average'].rank(ascending=False).astype(int)
        pivot = pivot.sort_values('Average', ascending=False)

        for col in pivot.columns:
            if col != 'Rank':
                pivot[col] = pivot[col].round(4)

        return pivot

    stage1_pivot = create_pivot(stage1_df)
    stage2_pivot = create_pivot(stage2_df)

    return stage1_pivot, stage2_pivot


def create_detailed_table(df: pd.DataFrame, split: SplitConfig) -> pd.DataFrame:
    """Create detailed table with loss and Pearson for this split."""
    detailed = df[
        [
            'hyperparameter',
            'cell_type',
            'training_mode',
            split.loss_col,
            split.pearson_col,
            split.epoch_col,
        ]
    ].copy()
    detailed = detailed.sort_values(['training_mode', 'hyperparameter', 'cell_type'])
    detailed[split.loss_col] = detailed[split.loss_col].round(4)
    detailed[split.pearson_col] = detailed[split.pearson_col].round(4)
    return detailed


def print_summary(
    split: SplitConfig,
    stage1_pivot: pd.DataFrame,
    stage2_pivot: pd.DataFrame,
):
    """Print formatted summary to console."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS SUMMARY — {split.label} ({split.pearson_col})")
    print("=" * 80)

    if len(stage1_pivot) > 0:
        print("\n" + "=" * 80)
        print(f"STAGE 1 ONLY — {split.label} (Head Training with Cached Embeddings)")
        print("=" * 80)
        print("\n📊 PERFORMANCE BY HYPERPARAMETER (Pearson Correlation)")
        print("-" * 70)
        print(stage1_pivot.to_string())

        print(f"\n🏆 BEST HYPERPARAMETERS BY CELL TYPE (Stage 1, {split.label}):")
        print("-" * 50)
        for col in ['HepG2', 'K562', 'WTC11']:
            if col in stage1_pivot.columns:
                best_hp = stage1_pivot[col].idxmax()
                best_val = stage1_pivot.loc[best_hp, col]
                print(f"  {col}: {best_hp} (r={best_val:.4f})")

        best_overall = stage1_pivot['Average'].idxmax()
        best_avg = stage1_pivot.loc[best_overall, 'Average']
        print(f"\n🥇 OVERALL BEST (Stage 1, {split.label}): {best_overall} (avg r={best_avg:.4f})")

        if 'baseline-default' in stage1_pivot.index:
            baseline_avg = stage1_pivot.loc['baseline-default', 'Average']
            improvement = best_avg - baseline_avg
            print(f"   vs. baseline: +{improvement:.4f} ({improvement/baseline_avg*100:.1f}% improvement)")

    if len(stage2_pivot) > 0:
        print("\n\n" + "=" * 80)
        print(f"STAGE 1+2 — {split.label} (Two-Stage Training: Head → Head+Encoder)")
        print("=" * 80)
        print("\n📊 PERFORMANCE BY STAGE 1 STOPPING POINT (Pearson Correlation)")
        print("-" * 70)
        print(stage2_pivot.to_string())

        print(f"\n🏆 BEST STAGE 1 STOPPING POINT BY CELL TYPE ({split.label}):")
        print("-" * 50)
        for col in ['HepG2', 'K562', 'WTC11']:
            if col in stage2_pivot.columns:
                best_hp = stage2_pivot[col].idxmax()
                best_val = stage2_pivot.loc[best_hp, col]
                print(f"  {col}: {best_hp} (r={best_val:.4f})")

        best_overall = stage2_pivot['Average'].idxmax()
        best_avg = stage2_pivot.loc[best_overall, 'Average']
        print(f"\n🥇 OVERALL BEST (Stage 1+2, {split.label}): {best_overall} (avg r={best_avg:.4f})")

        if 's2-baseline-es' in stage2_pivot.index:
            baseline_avg = stage2_pivot.loc['s2-baseline-es', 'Average']
            improvement = best_avg - baseline_avg
            print(f"   vs. two-stage baseline: +{improvement:.4f} ({improvement/baseline_avg*100:.1f}% improvement)")

    if len(stage1_pivot) > 0 and len(stage2_pivot) > 0:
        print("\n\n" + "=" * 80)
        print(f"STAGE 1 vs STAGE 1+2 COMPARISON — {split.label}")
        print("=" * 80)
        best_s1_avg = stage1_pivot['Average'].max()
        best_s2_avg = stage2_pivot['Average'].max()
        best_s1_hp = stage1_pivot['Average'].idxmax()
        best_s2_hp = stage2_pivot['Average'].idxmax()

        print(f"Best Stage 1:     {best_s1_hp} (avg r={best_s1_avg:.4f})")
        print(f"Best Stage 1+2:   {best_s2_hp} (avg r={best_s2_avg:.4f})")

        if best_s2_avg > best_s1_avg:
            improvement = best_s2_avg - best_s1_avg
            print(f"\n✨ Stage 1+2 improves by: +{improvement:.4f} ({improvement/best_s1_avg*100:.1f}%)")
        else:
            decline = best_s1_avg - best_s2_avg
            print(f"\n⚠️  Stage 1 still better by: +{decline:.4f} ({decline/best_s2_avg*100:.1f}%)")


def write_split_outputs(
    split: SplitConfig,
    output_prefix: Path,
    stage1_pivot: pd.DataFrame,
    stage2_pivot: pd.DataFrame,
    detailed: pd.DataFrame,
) -> None:
    """Write stage1 / stage2 / detailed CSVs for one split."""
    suf = split.output_suffix

    if len(stage1_pivot) > 0:
        stage1_file = Path(str(output_prefix) + f'{suf}_stage1.csv')
        stage1_pivot.to_csv(stage1_file)
        print(f"\n✓ [{split.label}] Stage 1 summary saved to: {stage1_file}")

    if len(stage2_pivot) > 0:
        stage2_file = Path(str(output_prefix) + f'{suf}_stage2.csv')
        stage2_pivot.to_csv(stage2_file)
        print(f"✓ [{split.label}] Stage 2 summary saved to: {stage2_file}")

    if len(detailed) > 0:
        detailed_file = Path(str(output_prefix) + f'{suf}_detailed.csv')
        detailed.to_csv(detailed_file, index=False)
        print(f"✓ [{split.label}] Detailed results saved to: {detailed_file}")


def process_split(results_dir: Path, output_prefix: Path, split: SplitConfig, *, required: bool) -> None:
    """Load one split, build tables, save CSVs, print summary."""
    print(f"\n{'=' * 80}\nLoading {split.label} benchmark results ({split.file_stem}_*.csv)...\n{'=' * 80}")
    df = load_and_process_results(results_dir, split, required=required)
    if len(df) == 0:
        return

    print(f"[{split.label}] Total unique results: {len(df)}")
    print(f"  Stage 1 results: {len(df[df['training_mode'] == 'stage1'])}")
    print(f"  Stage 2 results: {len(df[df['training_mode'] == 'stage2'])}")

    stage1_pivot, stage2_pivot = create_summary_table(df, split.pearson_col)
    detailed = create_detailed_table(df, split)

    write_split_outputs(split, output_prefix, stage1_pivot, stage2_pivot, detailed)
    print_summary(split, stage1_pivot, stage2_pivot)


def main():
    parser = argparse.ArgumentParser(description='Collate benchmark results (test and validation)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing benchmark_results_*.csv and benchmark_val_results_*.csv')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path prefix (default: results/benchmark_summary)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_prefix = Path(args.output) if args.output else results_dir / 'benchmark_summary'

    process_split(results_dir, output_prefix, TEST_SPLIT, required=True)
    process_split(results_dir, output_prefix, VAL_SPLIT, required=False)


if __name__ == '__main__':
    main()
