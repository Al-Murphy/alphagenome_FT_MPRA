"""
Create a comparison table of MPRA model performance across cell lines.

Compiles results from:
- AlphaGenome Stage 1 (frozen encoder, head only)
- AlphaGenome Stage 2 (full fine-tuning)
- Enformer Stage 1 (frozen encoder, head only)
- Enformer Stage 2 (full fine-tuning)
- MPRALegNet (baseline)

Across cell lines: HepG2, K562, WTC11

USAGE:
    python scripts/create_mpra_comparison_table.py --output_dir ./results/comparison_tables
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional


def load_metrics_file(file_path: Path) -> Optional[Dict[str, float]]:
    """Load metrics from a CSV file.
    
    Args:
        file_path: Path to metrics CSV file
        
    Returns:
        Dictionary with metrics or None if file doesn't exist
    """
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        if len(df) == 0:
            return None
        # Return first row as dict
        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None


def find_alphagenome_metrics(results_dir: Path, cell_type: str, stage: str) -> Optional[Dict[str, float]]:
    """Find AlphaGenome metrics file.
    
    Args:
        results_dir: Base results directory
        cell_type: Cell type (HepG2, K562, WTC11)
        stage: Stage (stage1 or stage2)
        
    Returns:
        Metrics dictionary or None
    """
    # Try cell_type subdirectory first
    file_path = results_dir / "test_predictions" / cell_type / f"{stage}_{cell_type}_test_metrics.csv"
    if file_path.exists():
        print(f"Found AlphaGenome metrics in {file_path}")
        return load_metrics_file(file_path)
    
    # Fall back to root test_predictions directory
    file_path = results_dir / "test_predictions" / f"{stage}_{cell_type}_test_metrics.csv"
    if file_path.exists():
        print(f"Found AlphaGenome metrics in {file_path}")
        return load_metrics_file(file_path)
    
    return None


def find_enformer_metrics(results_dir: Path, cell_type: str, stage: str) -> Optional[Dict[str, float]]:
    """Find Enformer metrics file.
    
    Args:
        results_dir: Base results directory
        cell_type: Cell type (HepG2, K562, WTC11)
        stage: Stage (stage1 or stage2)
        
    Returns:
        Metrics dictionary or None
    """
    enformer_dir = results_dir / "test_predictions" / "enformer"
    
    if not enformer_dir.exists():
        return None
    
    # Find all metrics files for this cell type
    pattern = f"*{cell_type}_test_metrics.csv"
    matching_files = list(enformer_dir.glob(pattern))
    
    if not matching_files:
        return None
    
    # For stage1, look for files without "stage2" in the name
    # For stage2, look for files with "stage2" in the name
    for file_path in matching_files:
        filename = file_path.name
        if stage == "stage1" and "stage2" not in filename:
            return load_metrics_file(file_path)
        elif stage == "stage2" and "stage2" in filename:
            return load_metrics_file(file_path)
    
    # If no exact match, return the first one found (with warning)
    if matching_files:
        print(f"Warning: Using {matching_files[0].name} for Enformer {stage} {cell_type}")
        return load_metrics_file(matching_files[0])
    
    return None


def find_mpralegnet_metrics(results_dir: Path, cell_type: str) -> Optional[Dict[str, float]]:
    """Find MPRALegNet metrics file.
    
    Args:
        results_dir: Base results directory
        cell_type: Cell type (HepG2, K562, WTC11)
        
    Returns:
        Metrics dictionary or None
    """
    legnet_dir = results_dir / "mpralegnet_predictions"
    
    if not legnet_dir.exists():
        return None
    
    # MPRALegNet files follow pattern: legnet_*_{cell_type}_test_metrics.csv
    pattern = f"*{cell_type}_test_metrics.csv"
    matching_files = list(legnet_dir.glob(pattern))
    
    if not matching_files:
        return None
    
    # Return the first match (should be unique per cell type)
    return load_metrics_file(matching_files[0])


def create_comparison_table(results_dir: Path) -> pd.DataFrame:
    """Create comparison table of all models across cell lines.
    
    Args:
        results_dir: Base results directory
        
    Returns:
        DataFrame with models as rows and cell types as columns
    """
    cell_types = ["HepG2", "K562", "WTC11"]
    models = [
        ("MPRALegNet", "mpralegnet", None),
        ("Enformer probing", "enformer", "stage1"),
        ("Enformer fine-tuned", "enformer", "stage2"),
        ("AlphaGenome probing", "alphagenome", "stage1"),
        ("AlphaGenome fine-tuned", "alphagenome", "stage2"),
    ]
    
    # Initialize table with NaN
    table_data = {}
    
    for model_name, model_type, stage in models:
        row_data = {}
        
        for cell_type in cell_types:
            if model_type == "alphagenome":
                metrics = find_alphagenome_metrics(results_dir, cell_type, stage)
            elif model_type == "enformer":
                metrics = find_enformer_metrics(results_dir, cell_type, stage)
            elif model_type == "mpralegnet":
                metrics = find_mpralegnet_metrics(results_dir, cell_type)
            else:
                metrics = None
            
            if metrics is not None:
                # Format as "Pearson r" only, always with 3 decimal places
                pearson = metrics.get("pearson", np.nan)
                
                if not np.isnan(pearson):
                    # Format to exactly 3 decimal places (e.g., 0.820 -> "0.820", not "0.82")
                    row_data[cell_type] = f"{pearson:.3f}"
                else:
                    row_data[cell_type] = "N/A"
            else:
                row_data[cell_type] = "N/A"
        
        table_data[model_name] = row_data
    
    # Create DataFrame
    df = pd.DataFrame(table_data).T
    df.columns = cell_types
    df.index.name = "Model"
    
    # Ensure all numeric values are formatted as strings with exactly 3 decimal places
    # This prevents pandas from converting them back to floats and losing trailing zeros
    for col in df.columns:
        def format_value(x):
            if x == "N/A" or pd.isna(x) or x == "":
                return str(x)
            try:
                # Convert to float and format to exactly 3 decimal places
                return f"{float(x):.3f}"
            except (ValueError, TypeError):
                return str(x)
        
        df[col] = df[col].apply(format_value)
    
    return df


def create_detailed_table(results_dir: Path) -> pd.DataFrame:
    """Create detailed comparison table with separate metrics.
    
    Args:
        results_dir: Base results directory
        
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    cell_types = ["HepG2", "K562", "WTC11"]
    models = [
        ("MPRALegNet", "mpralegnet", None),
        ("Enformer probing", "enformer", "stage1"),
        ("Enformer fine-tuned", "enformer", "stage2"),
        ("AlphaGenome probing", "alphagenome", "stage1"),
        ("AlphaGenome fine-tuned", "alphagenome", "stage2"),
    ]
    
    # Initialize detailed data
    detailed_data = []
    
    for model_name, model_type, stage in models:
        for cell_type in cell_types:
            if model_type == "alphagenome":
                metrics = find_alphagenome_metrics(results_dir, cell_type, stage)
            elif model_type == "enformer":
                metrics = find_enformer_metrics(results_dir, cell_type, stage)
            elif model_type == "mpralegnet":
                metrics = find_mpralegnet_metrics(results_dir, cell_type)
            else:
                metrics = None
            
            if metrics is not None:
                # Format numeric values to 3 decimal places
                pearson = metrics.get("pearson", np.nan)
                r2 = metrics.get("r2", np.nan)
                mse = metrics.get("mse", np.nan)
                n_samples = metrics.get("n_samples", np.nan)
                
                detailed_data.append({
                    "Model": model_name,
                    "Cell Type": cell_type,
                    "Pearson r": f"{pearson:.3f}" if not np.isnan(pearson) else "N/A",
                    "R²": f"{r2:.3f}" if not np.isnan(r2) else "N/A",
                    "MSE": f"{mse:.3f}" if not np.isnan(mse) else "N/A",
                    "N samples": int(n_samples) if not np.isnan(n_samples) else "N/A",
                })
            else:
                detailed_data.append({
                    "Model": model_name,
                    "Cell Type": cell_type,
                    "Pearson r": "N/A",
                    "R²": "N/A",
                    "MSE": "N/A",
                    "N samples": "N/A",
                })
    
    df = pd.DataFrame(detailed_data)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Create comparison table of MPRA model performance'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Base results directory (default: ./results)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/comparison_tables',
        help='Output directory for tables (default: ./results/comparison_tables)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'latex', 'markdown', 'all'],
        default='all',
        help='Output format (default: all)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Creating MPRA Model Comparison Table")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create summary table (Pearson r only)
    print("Creating summary table...")
    summary_table = create_comparison_table(results_dir)
    print("\nSummary Table (Pearson r):")
    print(summary_table)
    print()
    
    # Create detailed table
    print("Creating detailed table...")
    detailed_table = create_detailed_table(results_dir)
    print("\nDetailed Table:")
    print(detailed_table)
    print()
    
    # Save tables
    if args.format in ['csv', 'all']:
        summary_file = output_dir / "mpra_comparison_summary.csv"
        detailed_file = output_dir / "mpra_comparison_detailed.csv"
        
        summary_table.to_csv(summary_file)
        print(f"✓ Summary table saved to {summary_file}")
        
        detailed_table.to_csv(detailed_file, index=False)
        print(f"✓ Detailed table saved to {detailed_file}")
    
    if args.format in ['latex', 'all']:
        summary_file = output_dir / "mpra_comparison_summary.tex"
        detailed_file = output_dir / "mpra_comparison_detailed.tex"
        
        # Summary table values are already formatted as strings with 3 decimal places
        summary_table.to_latex(summary_file)
        print(f"✓ Summary LaTeX table saved to {summary_file}")
        
        # Detailed table values are already formatted as strings with 3 decimal places
        detailed_table.to_latex(detailed_file, index=False)
        print(f"✓ Detailed LaTeX table saved to {detailed_file}")
    
    if args.format in ['markdown', 'all']:
        summary_file = output_dir / "mpra_comparison_summary.md"
        detailed_file = output_dir / "mpra_comparison_detailed.md"
        
        # Write markdown manually to preserve 3 decimal place formatting
        def write_markdown_table(df, file_path):
            """Write DataFrame to markdown format preserving string formatting."""
            with open(file_path, 'w') as f:
                # Write header
                headers = [df.index.name or ''] + list(df.columns)
                header_line = '| ' + ' | '.join(str(h) for h in headers) + ' |\n'
                f.write(header_line)
                
                # Write separator (first column left-aligned, rest right-aligned)
                sep_parts = []
                for i, h in enumerate(headers):
                    width = len(str(h))
                    if i == 0:
                        sep_parts.append(':' + '-' * (width - 1))
                    else:
                        sep_parts.append('-' * (width - 1) + ':')
                f.write('|' + '|'.join(sep_parts) + '|\n')
                
                # Write rows
                for idx in df.index:
                    row = [str(idx)] + [str(df.loc[idx, col]) for col in df.columns]
                    f.write('| ' + ' | '.join(row) + ' |\n')
        
        write_markdown_table(summary_table, summary_file)
        print(f"✓ Summary Markdown table saved to {summary_file}")
        
        # For detailed table, use to_markdown but ensure values are strings
        with open(detailed_file, 'w') as f:
            f.write(detailed_table.to_markdown(index=False))
        print(f"✓ Detailed Markdown table saved to {detailed_file}")
    
    print("\n" + "=" * 80)
    print("✓ Comparison tables created successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
