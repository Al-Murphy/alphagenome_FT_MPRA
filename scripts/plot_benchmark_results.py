#!/usr/bin/env python3
"""
Generate bar plots comparing models on lentiMPRA and STARR-seq benchmarks.

This script creates two plots:
1. lentiMPRA benchmark comparison across cell types (HepG2, K562, WTC11)
2. STARR-seq benchmark comparison across promoter types (Developmental, House-keeping)

Outputs are saved as both PDF and PNG formats.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style():
    """Set up the plotting style and color palette."""
    sns.set(font_scale=1.2)
    sns.set_style("white")
    
    # Color palette (matching plot_cagi5_results.py)
    pal = ["#A65141", "#E7CDC2", "#80A0C7", "#394165","#B1934A", "#DCA258", "#100F14", "#8B9DAF", "#EEDA9D", "#E8DCCF"]
    
    return pal


def load_lentimpra_data():
    """Load lentiMPRA benchmark data.
    
    Returns:
        DataFrame with columns: model, regime, cell_type, pearson_r
    """
    data = [
        # MPRALegNet
        {'model': 'MPRALegNet', 'regime': '', 'cell_type': 'HepG2', 'pearson_r': 0.781},
        {'model': 'MPRALegNet', 'regime': '', 'cell_type': 'K562', 'pearson_r': 0.81},
        {'model': 'MPRALegNet', 'regime': '', 'cell_type': 'WTC11', 'pearson_r': 0.727},
        # Enformer Probing
        {'model': 'Enf. MPRA', 'regime': 'Probing', 'cell_type': 'HepG2', 'pearson_r': 0.82},
        {'model': 'Enf. MPRA', 'regime': 'Probing', 'cell_type': 'K562', 'pearson_r': 0.82},
        {'model': 'Enf. MPRA', 'regime': 'Probing', 'cell_type': 'WTC11', 'pearson_r': 0.8},
        # Enformer Fine-tuned
        {'model': 'Enf. MPRA', 'regime': 'Fine-tuned', 'cell_type': 'HepG2', 'pearson_r': 0.83},
        {'model': 'Enf. MPRA', 'regime': 'Fine-tuned', 'cell_type': 'K562', 'pearson_r': 0.835},
        {'model': 'Enf. MPRA', 'regime': 'Fine-tuned', 'cell_type': 'WTC11', 'pearson_r': 0.804},
        # AlphaGenome Probing
        {'model': 'AG MPRA', 'regime': 'Probing', 'cell_type': 'HepG2', 'pearson_r': 0.882},
        {'model': 'AG MPRA', 'regime': 'Probing', 'cell_type': 'K562', 'pearson_r': 0.873},
        {'model': 'AG MPRA', 'regime': 'Probing', 'cell_type': 'WTC11', 'pearson_r': 0.821},
        # AlphaGenome Fine-tuned
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'cell_type': 'HepG2', 'pearson_r': 0.885},
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'cell_type': 'K562', 'pearson_r': 0.877},
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'cell_type': 'WTC11', 'pearson_r': 0.822},
    ]
    
    df = pd.DataFrame(data)
    
    # Create model_label column for plotting
    df['model_label'] = df.apply(
        lambda row: row['model'] if row['regime'] == '' else f"{row['model']} ({row['regime']})",
        axis=1
    )
    
    return df


def load_starrseq_data():
    """Load STARR-seq benchmark data.
    
    Returns:
        DataFrame with columns: model, regime, promoter_type, pearson_r
    """
    data = [
        # DeepSTARR
        {'model': 'DeepSTARR', 'regime': '', 'promoter_type': 'Developmental', 'pearson_r': 0.656},
        {'model': 'DeepSTARR', 'regime': '', 'promoter_type': 'House-keeping', 'pearson_r': 0.736},
        # Dream-RNN
        {'model': 'Dream-RNN', 'regime': '', 'promoter_type': 'Developmental', 'pearson_r': 0.665},
        {'model': 'Dream-RNN', 'regime': '', 'promoter_type': 'House-keeping', 'pearson_r': 0.746},
        # Enformer Probing
        {'model': 'Enf. MPRA', 'regime': 'Probing', 'promoter_type': 'Developmental', 'pearson_r': 0.579},
        {'model': 'Enf. MPRA', 'regime': 'Probing', 'promoter_type': 'House-keeping', 'pearson_r': 0.568},
        # Enformer Fine-tuned
        {'model': 'Enf. MPRA', 'regime': 'Fine-tuned', 'promoter_type': 'Developmental', 'pearson_r': 0.678},
        {'model': 'Enf. MPRA', 'regime': 'Fine-tuned', 'promoter_type': 'House-keeping', 'pearson_r': 0.739},
        # AlphaGenome Probing
        {'model': 'AG MPRA', 'regime': 'Probing', 'promoter_type': 'Developmental', 'pearson_r': 0.741},
        {'model': 'AG MPRA', 'regime': 'Probing', 'promoter_type': 'House-keeping', 'pearson_r': 0.797},
        # AlphaGenome Fine-tuned
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'promoter_type': 'Developmental', 'pearson_r': 0.741},
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'promoter_type': 'House-keeping', 'pearson_r': 0.8},
    ]
    
    df = pd.DataFrame(data)
    
    # Create model_label column for plotting
    df['model_label'] = df.apply(
        lambda row: row['model'] if row['regime'] == '' else f"{row['model']} ({row['regime']})",
        axis=1
    )
    
    return df


def plot_lentimpra_benchmark(dat, pal, figsize=(8, 5)):
    """Create bar plot for lentiMPRA benchmark.
    
    Args:
        dat: DataFrame with lentiMPRA data
        pal: Color palette
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Define model order
    model_order = [
        'MPRALegNet',
        'Enf. MPRA (Probing)',
        'Enf. MPRA (Fine-tuned)',
        'AG MPRA (Probing)',
        'AG MPRA (Fine-tuned)'
    ]
    
    # Define cell type order
    cell_order = ['HepG2', 'K562', 'WTC11']
    
    # Color mapping for models
    model_colors = {
        'MPRALegNet': pal[9],
        'Enf. MPRA (Probing)': pal[1],
        'Enf. MPRA (Fine-tuned)': pal[0],
        'AG MPRA (Probing)': pal[2],
        'AG MPRA (Fine-tuned)': pal[3],
    }
    
    # Create grouped bar plot
    x_pos = np.arange(len(cell_order))
    width = 0.15  # Width of bars
    
    for i, model in enumerate(model_order):
        model_data = dat[dat['model_label'] == model]
        if len(model_data) == 0:
            continue
        
        # Get values in cell_order
        values = []
        for cell in cell_order:
            cell_data = model_data[model_data['cell_type'] == cell]
            if len(cell_data) > 0:
                values.append(cell_data['pearson_r'].values[0])
            else:
                values.append(0)
        
        # Calculate bar positions
        offset = (i - len(model_order) / 2) * width + width / 2
        bars = ax.bar(
            x_pos + offset,
            values,
            width,
            label=model,
            color=model_colors[model],
            alpha=0.9,
            edgecolor='black',
            linewidth=1
        )
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                bar_color = bar.get_facecolor()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.001,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    color=bar_color,
                    rotation=90
                )
    
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('lentiMPRA', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cell_order)
    ax.set_ylim([0.5, 1])
    ax.grid(axis='y', alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', frameon=False, fontsize=10, ncol=2)
    
    plt.tight_layout()
    return fig


def plot_starrseq_benchmark(dat, pal, figsize=(8, 5)):
    """Create bar plot for STARR-seq benchmark.
    
    Args:
        dat: DataFrame with STARR-seq data
        pal: Color palette
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Define model order
    model_order = [
        'DeepSTARR',
        'Dream-RNN',
        'Enf. MPRA (Probing)',
        'Enf. MPRA (Fine-tuned)',
        'AG MPRA (Probing)',
        'AG MPRA (Fine-tuned)'
    ]
    
    # Define promoter type order
    promoter_order = ['Developmental', 'House-keeping']
    
    # Color mapping for models
    model_colors = {
        'DeepSTARR': pal[9],
        'Dream-RNN': pal[7],
        'Enf. MPRA (Probing)': pal[1],
        'Enf. MPRA (Fine-tuned)': pal[0],
        'AG MPRA (Probing)': pal[2],
        'AG MPRA (Fine-tuned)': pal[3],
    }
    
    # Create grouped bar plot
    x_pos = np.arange(len(promoter_order))
    width = 0.13  # Width of bars
    
    for i, model in enumerate(model_order):
        model_data = dat[dat['model_label'] == model]
        if len(model_data) == 0:
            continue
        
        # Get values in promoter_order
        values = []
        for promoter in promoter_order:
            promoter_data = model_data[model_data['promoter_type'] == promoter]
            if len(promoter_data) > 0:
                values.append(promoter_data['pearson_r'].values[0])
            else:
                values.append(0)
        
        # Calculate bar positions
        offset = (i - len(model_order) / 2) * width + width / 2
        bars = ax.bar(
            x_pos + offset,
            values,
            width,
            label=model,
            color=model_colors[model],
            alpha=0.9,
            edgecolor='black',
            linewidth=1
        )
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                bar_color = bar.get_facecolor()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.001,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    color=bar_color,
                    rotation=90
                )
    
    ax.set_xlabel('Promoter Type', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('STARR-seq', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(promoter_order)
    ax.set_ylim([0.5, 1])
    ax.grid(axis='y', alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    
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
        description="Generate bar plots comparing models on lentiMPRA and STARR-seq benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/comparison_tables/plots',
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
        '--figsize_lentimpra',
        type=float,
        nargs=2,
        default=[7, 6],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size for lentiMPRA plot (width, height)'
    )
    parser.add_argument(
        '--figsize_starrseq',
        type=float,
        nargs=2,
        default=[7, 6],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size for STARR-seq plot (width, height)'
    )
    parser.add_argument(
        '--skip_lentimpra',
        action='store_true',
        help='Skip the lentiMPRA plot'
    )
    parser.add_argument(
        '--skip_starrseq',
        action='store_true',
        help='Skip the STARR-seq plot'
    )
    
    args = parser.parse_args()
    
    # Setup
    pal = setup_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Benchmark Results Plotting")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}")
    print(f"Formats: {', '.join(args.formats)}")
    print("=" * 80)
    print()
    
    # Generate lentiMPRA plot
    if not args.skip_lentimpra:
        print("Generating lentiMPRA benchmark plot...")
        lentimpra_dat = load_lentimpra_data()
        fig_lentimpra = plot_lentimpra_benchmark(
            lentimpra_dat,
            pal,
            figsize=tuple(args.figsize_lentimpra)
        )
        save_plots(
            fig_lentimpra,
            output_dir / 'lentimpra_benchmark',
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_lentimpra)
        print()
    
    # Generate STARR-seq plot
    if not args.skip_starrseq:
        print("Generating STARR-seq benchmark plot...")
        starrseq_dat = load_starrseq_data()
        fig_starrseq = plot_starrseq_benchmark(
            starrseq_dat,
            pal,
            figsize=tuple(args.figsize_starrseq)
        )
        save_plots(
            fig_starrseq,
            output_dir / 'starrseq_benchmark',
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_starrseq)
        print()
    
    print("=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
