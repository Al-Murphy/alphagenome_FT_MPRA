#!/usr/bin/env python3
"""Regenerate benchmark results CSVs from WandB run history.

This script queries WandB for all benchmark runs and extracts their test metrics
to recreate the benchmark_results_{cell_type}.csv files.

Usage:
    python scripts/regenerate_benchmark_results.py
    python scripts/regenerate_benchmark_results.py --project alphagenome-mpra --output_dir results/
"""

import argparse
import csv
from pathlib import Path
import wandb


def extract_cell_type_from_run_name(run_name: str) -> str:
    """Extract cell type from run name.
    
    Examples:
        'benchmark-HepG2-baseline-default' -> 'HepG2'
        'benchmark-K562-nl-512-256' -> 'K562'
        'benchmark-WTC11-do-0.3' -> 'WTC11'
        'benchmark-HepG2-s2-baseline-es' -> 'HepG2'
    """
    parts = run_name.split('-')
    if len(parts) >= 2:
        return parts[1]
    return 'unknown'


def extract_training_mode_from_run_name(run_name: str) -> str:
    """Extract training mode from run name.
    
    Examples:
        'benchmark-HepG2-baseline-default' -> 'stage1'
        'benchmark-K562-nl-512-256' -> 'stage1'
        'benchmark-HepG2-s2-baseline-es' -> 'stage2'
        'benchmark-K562-s2-s1ep20' -> 'stage2'
    
    Returns:
        'stage1' or 'stage2'
    """
    parts = run_name.split('-')
    # Check if run name contains 's2-' pattern (stage 2)
    if len(parts) >= 3 and parts[2] == 's2':
        return 'stage2'
    return 'stage1'


def get_final_test_metrics(run):
    """Extract final test metrics from a WandB run.
    
    Returns dict with test_loss, test_pearson, and best_test_epoch.
    """
    # Get run summary (contains final metrics)
    summary = run.summary
    
    # Try to get test metrics from summary
    test_loss = summary.get('test_loss', None)
    test_pearson = summary.get('test_pearson', None)
    
    # If not in summary, get from history (last logged values)
    if test_loss is None or test_pearson is None:
        history = run.history(keys=['test_loss', 'test_pearson', 'epoch'])
        if len(history) > 0:
            # Get last row with test metrics
            test_rows = history.dropna(subset=['test_loss', 'test_pearson'])
            if len(test_rows) > 0:
                last_test = test_rows.iloc[-1]
                test_loss = last_test.get('test_loss', None)
                test_pearson = last_test.get('test_pearson', None)
    
    # Try to find best test metrics from history
    best_test_loss = test_loss
    best_test_pearson = test_pearson
    best_test_epoch = summary.get('epoch', 0)
    
    # Scan history for best test Pearson
    try:
        history = run.history(keys=['test_pearson', 'test_loss', 'epoch'])
        if len(history) > 0:
            test_rows = history.dropna(subset=['test_pearson'])
            if len(test_rows) > 0:
                # Find row with best (highest) test Pearson
                best_idx = test_rows['test_pearson'].idxmax()
                best_row = test_rows.loc[best_idx]
                best_test_pearson = best_row['test_pearson']
                best_test_loss = best_row.get('test_loss', test_loss)
                best_test_epoch = int(best_row.get('epoch', 0))
    except Exception as e:
        print(f"  Warning: Could not scan history for best metrics: {e}")
    
    return {
        'best_test_loss': best_test_loss,
        'best_test_pearson': best_test_pearson,
        'best_test_epoch': best_test_epoch,
        'final_test_loss': test_loss,
        'final_test_pearson': test_pearson,
    }


def regenerate_results(project: str, output_dir: Path, run_prefix: str = 'benchmark'):
    """Query WandB and regenerate benchmark results CSVs."""
    api = wandb.Api()
    
    print(f"Querying WandB project: {project}")
    print(f"Looking for runs starting with: {run_prefix}")
    
    # Get all runs from project
    runs = api.runs(project)
    
    # Filter for benchmark runs
    benchmark_runs = [r for r in runs if r.name and r.name.startswith(run_prefix)]
    print(f"Found {len(benchmark_runs)} benchmark runs")
    
    if len(benchmark_runs) == 0:
        print("No benchmark runs found! Check your project name and run prefix.")
        return
    
    # Group runs by cell type
    results_by_cell_type = {
        'HepG2': [],
        'K562': [],
        'WTC11': [],
    }
    
    print("\nExtracting metrics from runs...")
    for run in benchmark_runs:
        run_name = run.name
        cell_type = extract_cell_type_from_run_name(run_name)
        training_mode = extract_training_mode_from_run_name(run_name)
        
        if cell_type not in results_by_cell_type:
            print(f"  Warning: Unknown cell type '{cell_type}' for run {run_name}")
            continue
        
        print(f"  Processing: {run_name} ({run.state}) [{training_mode}]")
        
        # Skip failed/crashed runs
        if run.state not in ['finished', 'running']:
            print(f"    Skipping (state: {run.state})")
            continue
        
        # Extract metrics
        metrics = get_final_test_metrics(run)
        
        # Check if we got valid metrics
        if metrics['best_test_pearson'] is None:
            print(f"    Warning: No test metrics found for {run_name}")
            continue
        
        # Add to results
        results_by_cell_type[cell_type].append({
            'run_name': run_name,
            'cell_type': cell_type,
            'training_mode': training_mode,
            'best_test_loss': metrics['best_test_loss'],
            'best_test_pearson': metrics['best_test_pearson'],
            'best_test_epoch': metrics['best_test_epoch'],
            'final_test_loss': metrics['final_test_loss'],
            'final_test_pearson': metrics['final_test_pearson'],
        })
        
        print(f"    ✓ Best test Pearson: {metrics['best_test_pearson']:.4f} (epoch {metrics['best_test_epoch']})")
    
    # Write CSV files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nWriting CSV files...")
    for cell_type, results in results_by_cell_type.items():
        if len(results) == 0:
            print(f"  Warning: No results for {cell_type}")
            continue
        
        output_file = output_dir / f'benchmark_results_{cell_type}.csv'
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['run_name', 'cell_type', 'training_mode', 'best_test_loss', 'best_test_pearson', 
                         'best_test_epoch', 'final_test_loss', 'final_test_pearson']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"  ✓ {output_file}: {len(results)} results")
    
    print(f"\n✓ Regenerated benchmark results in {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review the regenerated CSV files")
    print(f"  2. Run: python scripts/collate_benchmark_results.py --results_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate benchmark results CSVs from WandB'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='alphagenome-mpra',
        help='WandB project name (default: alphagenome-mpra)'
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        help='WandB entity/username (default: your default entity)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for CSV files (default: results/)'
    )
    parser.add_argument(
        '--run_prefix',
        type=str,
        default='benchmark',
        help='Prefix for benchmark run names (default: benchmark)'
    )
    args = parser.parse_args()
    
    # Construct full project path
    if args.entity:
        project = f"{args.entity}/{args.project}"
    else:
        project = args.project
    
    output_dir = Path(args.output_dir)
    
    try:
        regenerate_results(project, output_dir, args.run_prefix)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're logged into WandB: wandb login")
        print("  2. Check your project name with: wandb project ls")
        print("  3. Verify entity/username if needed: --entity YOUR_USERNAME")
        raise


if __name__ == '__main__':
    main()
