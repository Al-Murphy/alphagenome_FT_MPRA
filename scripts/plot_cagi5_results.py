#!/usr/bin/env python3
"""
Generate violin plots comparing AG vs AG MPRA models on CAGI5 benchmark.

This script creates two plots:
1. Comparison by model (AG vs AG MPRA)
2. Comparison by model and cell line (AG/AG MPRA × K562/HepG2)

Outputs are saved as both PDF and PNG formats.

To note, high confidence SNPs relate tot he significance of the SNP in the MPRA regression model.
> We deemed a confidence score greater or equal to 0.1 (p-value of 10⁻⁵) indicates that the SNV ‘has an expression effect’.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style():
    """Set up the plotting style and color palette."""
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")
    
    # Color palette
    pal = ["#8B9DAF", "#A65141", "#1d6cb1", "#394165", "#80A0C7", "#E7CDC2",
           "#B1934A", "#DCA258", "#EEDA9D", "#E8DCCF"]
    
    return pal


def load_cagi5_data(results_dir):
    """Load CAGI5 per-element results from CSV files."""
    pth = os.path.join(results_dir, "cagi5_*_per_element.csv")
    pths = glob.glob(pth)
    
    if not pths:
        raise FileNotFoundError(f"No CAGI5 per-element CSV files found in {results_dir}")
    
    dat = []
    for pth in pths:
        dat_i = pd.read_csv(pth)
        dat_i["model"] = "AG MPRA" if os.path.basename(pth).split("_")[1] == "finetuned" else "AG"
        dat_i["cell_type"] = os.path.basename(pth).split("_")[2]
        dat.append(dat_i)
    
    dat = pd.concat(dat, ignore_index=True)
    return dat


def prepare_data(dat):
    """Reshape data to long format for plotting."""
    # Reshape data to long format
    dat_long = pd.melt(
        dat,
        id_vars=['element', 'model', 'cell_type', 'n_variants', 'n_high_conf'],
        value_vars=['pearson_all', 'pearson_high_conf'],
        var_name='snp_type',
        value_name='pearson_r'
    )
    
    # Map to readable labels
    dat_long['snp_type'] = dat_long['snp_type'].map({
        'pearson_all': 'All SNPs',
        'pearson_high_conf': 'High Confidence SNPs'
    })
    
    # Get N numbers for each group
    dat_long['n'] = dat_long.apply(
        lambda row: row['n_variants'] if row['snp_type'] == 'All SNPs' else row['n_high_conf'],
        axis=1
    )
    
    return dat_long


def plot_by_model(dat_long, pal, figsize=(10, 5)):
    """Create violin plot comparing AG vs AG MPRA (aggregated across cell types)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    snp_types = ['All SNPs', 'High Confidence SNPs']
    model_order = ['AG', 'AG MPRA']
    
    for idx, snp_type in enumerate(snp_types):
        ax = axes[idx]
        data_subset = dat_long[dat_long['snp_type'] == snp_type]
        
        # Create violin plot with explicit order
        sns.violinplot(
            data=data_subset,
            x='model',
            y='pearson_r',
            ax=ax,
            order=model_order,
            palette={'AG': pal[3], 'AG MPRA': pal[4]},
            inner=None,
            alpha=0.8
        )
        
        # Get actual x positions from the plot
        xtick_positions = ax.get_xticks()
        model_to_xpos = {label.get_text(): pos for pos, label in zip(xtick_positions, ax.get_xticklabels())}
        
        # Calculate statistics for title (only AG MPRA)
        ag_mpra_data = data_subset[data_subset['model'] == 'AG MPRA']
        n_total = ag_mpra_data['n'].sum()
        n_elements = len(ag_mpra_data)
        title_parts = [f'{snp_type}', f'N={n_total} SNPs, {n_elements} regulatory elements']
        
        # Add individual regulatory element dots using actual x positions
        for model in model_order:
            model_data = data_subset[data_subset['model'] == model]
            x_pos = model_to_xpos[model]
            # Add jitter
            jitter = np.random.normal(0, 0.035, len(model_data))
            ax.scatter(
                x_pos + jitter,
                model_data['pearson_r'],
                s=40,
                alpha=0.9,
                edgecolors='black',
                linewidth=1,
                zorder=10,
                color=pal[4] if model == 'AG MPRA' else pal[3]
            )
            
            # Add mean Pearson r as a horizontal line segment
            mean_pearson = model_data['pearson_r'].mean()
            line_color = pal[4] if model == 'AG MPRA' else pal[3]
            ax.plot(
                [x_pos - 0.25, x_pos + 0.25],
                [mean_pearson, mean_pearson],
                color=line_color,
                linestyle='--',
                linewidth=2.5,
                alpha=0.9,
                zorder=5
            )
            # Add mean value as text
            ax.text(
                x_pos,
                mean_pearson - 0.35,
                f'μ={mean_pearson:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color=line_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=line_color, linewidth=1.5)
            )
        
        ax.set_title('\n'.join(title_parts))
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel('Pearson Correlation')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_by_model_and_cell(dat_long, pal, figsize=(16, 6)):
    """Create violin plot comparing AG vs AG MPRA split by cell line."""
    # Create a combined label for model and cell type
    dat_long = dat_long.copy()
    dat_long['model_cell'] = dat_long['model'] + ' (' + dat_long['cell_type'] + ')'
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    snp_types = ['All SNPs', 'High Confidence SNPs']
    
    # Define order: AG (K562), AG MPRA (K562), AG (HepG2), AG MPRA (HepG2)
    model_cell_order = ['AG (K562)', 'AG MPRA (K562)', 'AG (HepG2)', 'AG MPRA (HepG2)']
    
    for idx, snp_type in enumerate(snp_types):
        ax = axes[idx]
        data_subset = dat_long[dat_long['snp_type'] == snp_type]
        
        # Create violin plot with explicit order
        sns.violinplot(
            data=data_subset,
            x='model_cell',
            y='pearson_r',
            ax=ax,
            order=model_cell_order,
            palette={
                'AG (K562)': pal[3],
                'AG MPRA (K562)': pal[4],
                'AG (HepG2)': pal[3],
                'AG MPRA (HepG2)': pal[4]
            },
            inner=None,
            alpha=0.8
        )
        
        # Get actual x positions from the plot
        xtick_positions = ax.get_xticks()
        model_cell_to_xpos = {label.get_text(): pos for pos, label in zip(xtick_positions, ax.get_xticklabels())}
        
        # Calculate statistics for title (only AG MPRA, aggregated across cell types)
        ag_mpra_data = data_subset[data_subset['model'] == 'AG MPRA']
        n_total = ag_mpra_data['n'].sum()
        n_elements = len(ag_mpra_data)
        title_parts = [f'{snp_type}', f'N={n_total} SNPs, {n_elements} regulatory elements']
        
        # Add individual regulatory element dots using actual x positions
        for model_cell in model_cell_order:
            model_cell_data = data_subset[data_subset['model_cell'] == model_cell]
            if len(model_cell_data) == 0:
                continue
            x_pos = model_cell_to_xpos[model_cell]
            # Add jitter
            jitter = np.random.normal(0, 0.035, len(model_cell_data))
            ax.scatter(
                x_pos + jitter,
                model_cell_data['pearson_r'],
                s=40,
                alpha=0.9,
                edgecolors='black',
                linewidth=1,
                zorder=10,
                color=pal[4] if 'AG MPRA' in model_cell else pal[3]
            )
            
            # Add mean Pearson r as a horizontal line segment
            mean_pearson = model_cell_data['pearson_r'].mean()
            line_color = pal[4] if 'AG MPRA' in model_cell else pal[3]
            ax.plot(
                [x_pos - 0.25, x_pos + 0.25],
                [mean_pearson, mean_pearson],
                color=line_color,
                linestyle='--',
                linewidth=2.5,
                alpha=0.9,
                zorder=5
            )
            # Add mean value as text
            ax.text(
                x_pos,
                mean_pearson - 0.35,
                f'μ={mean_pearson:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color=line_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=line_color, linewidth=1.5)
            )
        
        ax.set_title('\n'.join(title_parts))
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel('Pearson Correlation')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def save_plots(fig, output_path, dpi=1200, formats=['pdf', 'png']):
    """Save figure in multiple formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for fmt in formats:
        if fmt == 'pdf':
            filepath = output_path.with_suffix('.pdf')
            fig.savefig(filepath, format='pdf', bbox_inches='tight')
        elif fmt == 'png':
            filepath = output_path.with_suffix('.png')
            fig.savefig(filepath, format='png', dpi=dpi, bbox_inches='tight')
        else:
            filepath = output_path.with_suffix(f'.{fmt}')
            fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight')
        
        saved_files.append(filepath)
        print(f"✓ Saved {filepath}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate violin plots comparing AG vs AG MPRA models on CAGI5 benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/cagi5_evaluations',
        help='Directory containing CAGI5 per-element CSV files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/cagi5_evaluations/plots',
        help='Directory to save output plots'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=1200,
        help='DPI for PNG output'
    )
    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=['pdf', 'png'],
        choices=['pdf', 'png', 'svg'],
        help='Output formats'
    )
    parser.add_argument(
        '--figsize_model',
        type=float,
        nargs=2,
        default=[10, 5],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size for model comparison plot (width, height)'
    )
    parser.add_argument(
        '--figsize_cell',
        type=float,
        nargs=2,
        default=[16, 6],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size for model+cell comparison plot (width, height)'
    )
    parser.add_argument(
        '--skip_model_plot',
        action='store_true',
        help='Skip the model-only comparison plot'
    )
    parser.add_argument(
        '--skip_cell_plot',
        action='store_true',
        help='Skip the model+cell comparison plot'
    )
    
    args = parser.parse_args()
    
    # Setup
    pal = setup_plot_style()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CAGI5 Results Plotting")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}")
    print(f"Formats: {', '.join(args.formats)}")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading CAGI5 data...")
    dat = load_cagi5_data(results_dir)
    print(f"✓ Loaded data for {len(dat)} elements")
    
    # Prepare data
    print("Preparing data...")
    dat_long = prepare_data(dat)
    print(f"✓ Prepared {len(dat_long)} data points")
    print()
    
    # Generate plots
    if not args.skip_model_plot:
        print("Generating model comparison plot...")
        fig_model = plot_by_model(dat_long, pal, figsize=tuple(args.figsize_model))
        save_plots(
            fig_model,
            output_dir / 'cagi5_model_comparison',
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_model)
        print()
    
    if not args.skip_cell_plot:
        print("Generating model+cell comparison plot...")
        fig_cell = plot_by_model_and_cell(dat_long, pal, figsize=tuple(args.figsize_cell))
        save_plots(
            fig_cell,
            output_dir / 'cagi5_model_cell_comparison',
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_cell)
        print()
    
    print("=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
