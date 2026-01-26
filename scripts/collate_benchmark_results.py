#!/usr/bin/env python3
"""Collate benchmark results from all cell types into a summary table.

Usage:
    python scripts/collate_benchmark_results.py
    python scripts/collate_benchmark_results.py --results_dir results/ --output results/benchmark_summary.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def extract_hyperparameter(run_name: str) -> str:
    """Extract the hyperparameter being tested from run name.
    
    Examples:
        'benchmark-HepG2-baseline-default' -> 'baseline-default'
        'benchmark-K562-nl-512-256' -> 'nl-512-256'
        'benchmark-WTC11-do-0.3' -> 'do-0.3'
    """
    parts = run_name.split('-')
    # Skip 'benchmark' and cell type (e.g., 'HepG2')
    if len(parts) >= 3:
        return '-'.join(parts[2:])
    return run_name


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
    
    # Extract hyperparameter name
    combined['hyperparameter'] = combined['run_name'].apply(extract_hyperparameter)
    
    # Remove duplicates: keep the best result (highest Pearson) for each cell_type + hyperparameter
    combined = combined.sort_values('best_test_pearson', ascending=False)
    combined = combined.drop_duplicates(subset=['cell_type', 'hyperparameter'], keep='first')
    
    return combined


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table showing performance across cell types."""
    # Pivot: rows = hyperparameter, columns = cell type, values = best_test_pearson
    pivot = df.pivot_table(
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


def create_detailed_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create detailed table with both loss and Pearson."""
    # Select key columns
    detailed = df[['hyperparameter', 'cell_type', 'best_test_loss', 'best_test_pearson', 'best_test_epoch']].copy()
    detailed = detailed.sort_values(['hyperparameter', 'cell_type'])
    
    # Round values
    detailed['best_test_loss'] = detailed['best_test_loss'].round(4)
    detailed['best_test_pearson'] = detailed['best_test_pearson'].round(4)
    
    return detailed


def print_summary(pivot: pd.DataFrame, detailed: pd.DataFrame):
    """Print formatted summary to console."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    print("\n📊 PERFORMANCE BY HYPERPARAMETER (Pearson Correlation)")
    print("-"*70)
    print(pivot.to_string())
    
    # Highlight best for each cell type
    print("\n🏆 BEST HYPERPARAMETERS BY CELL TYPE:")
    print("-"*50)
    for col in ['HepG2', 'K562', 'WTC11']:
        if col in pivot.columns:
            best_hp = pivot[col].idxmax()
            best_val = pivot.loc[best_hp, col]
            print(f"  {col}: {best_hp} (r={best_val:.4f})")
    
    # Overall winner
    best_overall = pivot['Average'].idxmax()
    best_avg = pivot.loc[best_overall, 'Average']
    print(f"\n🥇 OVERALL BEST: {best_overall} (avg r={best_avg:.4f})")
    
    # Compare to baseline
    if 'baseline-default' in pivot.index:
        baseline_avg = pivot.loc['baseline-default', 'Average']
        improvement = best_avg - baseline_avg
        print(f"   vs. baseline: +{improvement:.4f} ({improvement/baseline_avg*100:.1f}% improvement)")


def main():
    parser = argparse.ArgumentParser(description='Collate benchmark results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing benchmark_results_*.csv files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (default: results/benchmark_summary.csv)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_file = Path(args.output) if args.output else results_dir / 'benchmark_summary.csv'
    
    # Load and process
    print("Loading benchmark results...")
    df = load_and_process_results(results_dir)
    print(f"Total unique results: {len(df)}")
    
    # Create summaries
    pivot = create_summary_table(df)
    detailed = create_detailed_table(df)
    
    # Save
    pivot.to_csv(output_file)
    print(f"\n✓ Summary saved to: {output_file}")
    
    detailed_file = output_file.parent / 'benchmark_detailed.csv'
    detailed.to_csv(detailed_file, index=False)
    print(f"✓ Detailed results saved to: {detailed_file}")
    
    # Print summary
    print_summary(pivot, detailed)


if __name__ == '__main__':
    main()
