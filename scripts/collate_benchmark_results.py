#!/usr/bin/env python3
"""Collate benchmark results from all cell types into a summary table.

Usage:
    python scripts/collate_benchmark_results.py
    python scripts/collate_benchmark_results.py --results_dir results/ --output results/benchmark_summary.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple


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


def load_and_process_results(results_dir: Path) -> pd.DataFrame:
    """Load all benchmark result CSVs and combine them."""
    all_results = []
    
    for cell_type in ['HepG2', 'K562', 'WTC11']:
        results_file = results_dir / f'benchmark_results_{cell_type}.csv'
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping")
            continue
        
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} results from {results_file}")
        all_results.append(df)
    
    if not all_results:
        raise FileNotFoundError("No benchmark results found!")
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Extract hyperparameter name and training mode
    combined[['hyperparameter', 'training_mode']] = combined['run_name'].apply(
        lambda x: pd.Series(extract_hyperparameter(x))
    )
    
    # Remove duplicates: keep the best result (highest Pearson) for each cell_type + hyperparameter + training_mode
    combined = combined.sort_values('best_test_pearson', ascending=False)
    combined = combined.drop_duplicates(subset=['cell_type', 'hyperparameter', 'training_mode'], keep='first')
    
    return combined


def create_summary_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create pivot tables showing performance across cell types for stage 1 and stage 2.
    
    Returns:
        Tuple of (stage1_pivot, stage2_pivot)
    """
    # Split into stage 1 and stage 2 results
    stage1_df = df[df['training_mode'] == 'stage1'].copy()
    stage2_df = df[df['training_mode'] == 'stage2'].copy()
    
    def create_pivot(data: pd.DataFrame) -> pd.DataFrame:
        """Helper to create pivot table for a dataset."""
        if len(data) == 0:
            return pd.DataFrame()
        
        # Pivot: rows = hyperparameter, columns = cell type, values = best_test_pearson
        pivot = data.pivot_table(
            index='hyperparameter',
            columns='cell_type',
            values='best_test_pearson',
            aggfunc='first'
        )
        
        # Reorder columns
        col_order = ['HepG2', 'K562', 'WTC11']
        pivot = pivot[[c for c in col_order if c in pivot.columns]]
        
        # Add average column
        pivot['Average'] = pivot.mean(axis=1)
        
        # Add rank column (1 = best)
        pivot['Rank'] = pivot['Average'].rank(ascending=False).astype(int)
        
        # Sort by average (descending)
        pivot = pivot.sort_values('Average', ascending=False)
        
        # Round values for display
        for col in pivot.columns:
            if col != 'Rank':
                pivot[col] = pivot[col].round(4)
        
        return pivot
    
    stage1_pivot = create_pivot(stage1_df)
    stage2_pivot = create_pivot(stage2_df)
    
    return stage1_pivot, stage2_pivot


def create_detailed_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create detailed table with both loss and Pearson."""
    # Select key columns (include training_mode)
    detailed = df[['hyperparameter', 'cell_type', 'training_mode', 'best_test_loss', 'best_test_pearson', 'best_test_epoch']].copy()
    detailed = detailed.sort_values(['training_mode', 'hyperparameter', 'cell_type'])
    
    # Round values
    detailed['best_test_loss'] = detailed['best_test_loss'].round(4)
    detailed['best_test_pearson'] = detailed['best_test_pearson'].round(4)
    
    return detailed


def print_summary(stage1_pivot: pd.DataFrame, stage2_pivot: pd.DataFrame, detailed: pd.DataFrame):
    """Print formatted summary to console."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Stage 1 results
    if len(stage1_pivot) > 0:
        print("\n" + "="*80)
        print("STAGE 1 ONLY (Head Training with Cached Embeddings)")
        print("="*80)
        print("\n📊 PERFORMANCE BY HYPERPARAMETER (Pearson Correlation)")
        print("-"*70)
        print(stage1_pivot.to_string())
        
        # Highlight best for each cell type
        print("\n🏆 BEST HYPERPARAMETERS BY CELL TYPE (Stage 1):")
        print("-"*50)
        for col in ['HepG2', 'K562', 'WTC11']:
            if col in stage1_pivot.columns:
                best_hp = stage1_pivot[col].idxmax()
                best_val = stage1_pivot.loc[best_hp, col]
                print(f"  {col}: {best_hp} (r={best_val:.4f})")
        
        # Overall winner
        best_overall = stage1_pivot['Average'].idxmax()
        best_avg = stage1_pivot.loc[best_overall, 'Average']
        print(f"\n🥇 OVERALL BEST (Stage 1): {best_overall} (avg r={best_avg:.4f})")
        
        # Compare to baseline
        if 'baseline-default' in stage1_pivot.index:
            baseline_avg = stage1_pivot.loc['baseline-default', 'Average']
            improvement = best_avg - baseline_avg
            print(f"   vs. baseline: +{improvement:.4f} ({improvement/baseline_avg*100:.1f}% improvement)")
    
    # Stage 2 results
    if len(stage2_pivot) > 0:
        print("\n\n" + "="*80)
        print("STAGE 1+2 (Two-Stage Training: Head → Head+Encoder)")
        print("="*80)
        print("\n📊 PERFORMANCE BY STAGE 1 STOPPING POINT (Pearson Correlation)")
        print("-"*70)
        print(stage2_pivot.to_string())
        
        # Highlight best for each cell type
        print("\n🏆 BEST STAGE 1 STOPPING POINT BY CELL TYPE:")
        print("-"*50)
        for col in ['HepG2', 'K562', 'WTC11']:
            if col in stage2_pivot.columns:
                best_hp = stage2_pivot[col].idxmax()
                best_val = stage2_pivot.loc[best_hp, col]
                print(f"  {col}: {best_hp} (r={best_val:.4f})")
        
        # Overall winner
        best_overall = stage2_pivot['Average'].idxmax()
        best_avg = stage2_pivot.loc[best_overall, 'Average']
        print(f"\n🥇 OVERALL BEST (Stage 1+2): {best_overall} (avg r={best_avg:.4f})")
        
        # Compare to two-stage baseline
        if 's2-baseline-es' in stage2_pivot.index:
            baseline_avg = stage2_pivot.loc['s2-baseline-es', 'Average']
            improvement = best_avg - baseline_avg
            print(f"   vs. two-stage baseline: +{improvement:.4f} ({improvement/baseline_avg*100:.1f}% improvement)")
    
    # Compare best stage 1 vs best stage 2
    if len(stage1_pivot) > 0 and len(stage2_pivot) > 0:
        print("\n\n" + "="*80)
        print("STAGE 1 vs STAGE 1+2 COMPARISON")
        print("="*80)
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


def main():
    parser = argparse.ArgumentParser(description='Collate benchmark results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing benchmark_results_*.csv files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path prefix (default: results/benchmark_summary)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_prefix = Path(args.output) if args.output else results_dir / 'benchmark_summary'
    
    # Load and process
    print("Loading benchmark results...")
    df = load_and_process_results(results_dir)
    print(f"Total unique results: {len(df)}")
    print(f"  Stage 1 results: {len(df[df['training_mode'] == 'stage1'])}")
    print(f"  Stage 2 results: {len(df[df['training_mode'] == 'stage2'])}")
    
    # Create summaries
    stage1_pivot, stage2_pivot = create_summary_table(df)
    detailed = create_detailed_table(df)
    
    # Save stage 1 summary
    if len(stage1_pivot) > 0:
        stage1_file = Path(str(output_prefix) + '_stage1.csv')
        stage1_pivot.to_csv(stage1_file)
        print(f"\n✓ Stage 1 summary saved to: {stage1_file}")
    
    # Save stage 2 summary
    if len(stage2_pivot) > 0:
        stage2_file = Path(str(output_prefix) + '_stage2.csv')
        stage2_pivot.to_csv(stage2_file)
        print(f"✓ Stage 2 summary saved to: {stage2_file}")
    
    # Save detailed results
    detailed_file = Path(str(output_prefix) + '_detailed.csv')
    detailed.to_csv(detailed_file, index=False)
    print(f"✓ Detailed results saved to: {detailed_file}")
    
    # Print summary
    print_summary(stage1_pivot, stage2_pivot, detailed)


if __name__ == '__main__':
    main()
