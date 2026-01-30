#!/usr/bin/env python3
"""
Generate violin plots comparing AG vs AG MPRA models on CAGI5 benchmark.

This script creates multiple plots:
1. Comparison by model (AG vs AG MPRA) - original version
2. Comparison by model (AG (501bp) vs AG vs AG MPRA) - with 501bp version
3. Comparison by model and cell line (AG/AG MPRA × K562/HepG2) - original version
4. Comparison by model and cell line (AG (501bp)/AG/AG MPRA × K562/HepG2) - with 501bp version

Outputs are saved as both PDF and PNG formats.

To note, high confidence SNPs relate tot he significance of the SNP in the MPRA regression model.
> We deemed a confidence score greater or equal to 0.1 (p-value of 10⁻⁵) indicates that the SNV 'has an expression effect'.
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
    # Load regular files
    pth = os.path.join(results_dir, "cagi5_*_per_element.csv")
    pths = glob.glob(pth)
    
    # Also load 501 central mask files
    pth_501 = os.path.join(results_dir, "501_central_mask_cagi5_*_per_element.csv")
    pths_501 = glob.glob(pth_501)
    
    all_pths = pths + pths_501
    
    if not all_pths:
        raise FileNotFoundError(f"No CAGI5 per-element CSV files found in {results_dir}")
    
    dat = []
    for pth in all_pths:
        dat_i = pd.read_csv(pth)
        basename = os.path.basename(pth)
        
        # Check if this is a 501 central mask file
        is_501 = basename.startswith("501_central_mask_")
        
        if is_501:
            # Remove the 501_central_mask_ prefix for parsing
            basename_clean = basename.replace("501_central_mask_", "")
            parts = basename_clean.split("_")
            if parts[1] == "base":
                dat_i["model"] = "AG (501bp)"
                dat_i["cell_type"] = parts[2]
            else:
                # Shouldn't happen, but handle gracefully
                dat_i["model"] = "AG MPRA"
                dat_i["cell_type"] = parts[2] if len(parts) > 2 else "Unknown"
        else:
            # Regular file parsing
            parts = basename.split("_")
            dat_i["model"] = "AG MPRA" if parts[1] == "finetuned" else "AG"
            dat_i["cell_type"] = parts[2]
        
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


def plot_by_model(dat_long, pal, include_501bp=True, figsize=(12, 5)):
    """Create violin plot comparing AG vs AG (501bp) vs AG MPRA (aggregated across cell types).
    
    Args:
        dat_long: Long-format data for plotting
        pal: Color palette
        include_501bp: If True, include AG (501bp) model; if False, only AG and AG MPRA
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    snp_types = ['All SNPs', 'High Confidence SNPs']
    
    if include_501bp:
        model_order = ['AG (501bp)', 'AG', 'AG MPRA']
    else:
        model_order = ['AG', 'AG MPRA']
    
    # Color mapping for models
    model_colors = {
        'AG': pal[3],
        'AG (501bp)': pal[0],  # Different shade for 501bp version
        'AG MPRA': pal[4]
    }
    
    for idx, snp_type in enumerate(snp_types):
        ax = axes[idx]
        data_subset = dat_long[dat_long['snp_type'] == snp_type]
        
        # Filter to only include models that exist in the data
        available_models = [m for m in model_order if m in data_subset['model'].values]
        
        # Create violin plot with explicit order
        sns.violinplot(
            data=data_subset,
            x='model',
            y='pearson_r',
            ax=ax,
            order=available_models,
            palette={m: model_colors[m] for m in available_models},
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
        for model in available_models:
            model_data = data_subset[data_subset['model'] == model]
            if len(model_data) == 0:
                continue
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
                color=model_colors[model]
            )
            
            # Add mean Pearson r as a horizontal line segment
            mean_pearson = model_data['pearson_r'].mean()
            line_color = model_colors[model]
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
                mean_pearson - 0.4,
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


def plot_by_model_and_cell(dat_long, pal, include_501bp=True, figsize=(20, 6)):
    """Create violin plot comparing AG vs AG (501bp) vs AG MPRA split by cell line.
    
    Args:
        dat_long: Long-format data for plotting
        pal: Color palette
        include_501bp: If True, include AG (501bp) model; if False, only AG and AG MPRA
        figsize: Figure size tuple
    """
    # Create a combined label for model and cell type
    dat_long = dat_long.copy()
    dat_long['model_cell'] = dat_long['model'] + ' (' + dat_long['cell_type'] + ')'
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    snp_types = ['All SNPs', 'High Confidence SNPs']
    
    if include_501bp:
        # Define order: AG (501bp) (K562), AG (K562), AG MPRA (K562), AG (501bp) (HepG2), AG (HepG2), AG MPRA (HepG2)
        model_cell_order = [
            'AG (501bp) (K562)',
            'AG (K562)', 
            'AG MPRA (K562)', 
            'AG (501bp) (HepG2)',
            'AG (HepG2)', 
            'AG MPRA (HepG2)'
        ]
    else:
        # Define order: AG (K562), AG MPRA (K562), AG (HepG2), AG MPRA (HepG2)
        model_cell_order = [
            'AG (K562)', 
            'AG MPRA (K562)', 
            'AG (HepG2)', 
            'AG MPRA (HepG2)'
        ]
    
    # Color mapping for model-cell combinations
    model_cell_colors = {
        'AG (K562)': pal[3],
        'AG (501bp) (K562)': pal[0],
        'AG MPRA (K562)': pal[4],
        'AG (HepG2)': pal[3],
        'AG (501bp) (HepG2)': pal[0],
        'AG MPRA (HepG2)': pal[4]
    }
    
    for idx, snp_type in enumerate(snp_types):
        ax = axes[idx]
        data_subset = dat_long[dat_long['snp_type'] == snp_type]
        
        # Filter to only include model-cell combinations that exist in the data
        available_model_cells = [mc for mc in model_cell_order if mc in data_subset['model_cell'].values]
        
        # Create violin plot with explicit order
        sns.violinplot(
            data=data_subset,
            x='model_cell',
            y='pearson_r',
            ax=ax,
            order=available_model_cells,
            palette={mc: model_cell_colors[mc] for mc in available_model_cells},
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
        for model_cell in available_model_cells:
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
                color=model_cell_colors[model_cell]
            )
            
            # Add mean Pearson r as a horizontal line segment
            mean_pearson = model_cell_data['pearson_r'].mean()
            line_color = model_cell_colors[model_cell]
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
        help='Figure size for model comparison plot without 501bp (width, height). With 501bp uses 12x5.'
    )
    parser.add_argument(
        '--figsize_cell',
        type=float,
        nargs=2,
        default=[16, 6],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size for model+cell comparison plot without 501bp (width, height). With 501bp uses 20x6.'
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
    
    # Generate plots - both with and without 501bp version
    if not args.skip_model_plot:
        # Version without 501bp (original)
        print("Generating model comparison plot (without 501bp)...")
        fig_model = plot_by_model(dat_long, pal, include_501bp=False, figsize=tuple(args.figsize_model))
        save_plots(
            fig_model,
            output_dir / 'cagi5_model_comparison',
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_model)
        print()
        
        # Version with 501bp
        print("Generating model comparison plot (with 501bp)...")
        fig_model_501 = plot_by_model(dat_long, pal, include_501bp=True, figsize=(12, 5))
        save_plots(
            fig_model_501,
            output_dir / 'cagi5_model_comparison_with_501bp',
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_model_501)
        print()
    
    if not args.skip_cell_plot:
        # Version without 501bp (original)
        print("Generating model+cell comparison plot (without 501bp)...")
        fig_cell = plot_by_model_and_cell(dat_long, pal, include_501bp=False, figsize=tuple(args.figsize_cell))
        save_plots(
            fig_cell,
            output_dir / 'cagi5_model_cell_comparison',
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_cell)
        print()
        
        # Version with 501bp
        print("Generating model+cell comparison plot (with 501bp)...")
        fig_cell_501 = plot_by_model_and_cell(dat_long, pal, include_501bp=True, figsize=(20, 6))
        save_plots(
            fig_cell_501,
            output_dir / 'cagi5_model_cell_comparison_with_501bp',
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_cell_501)
        print()
    
    print("=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
