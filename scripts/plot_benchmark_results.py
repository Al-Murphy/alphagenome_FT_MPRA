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


def load_lentimpra_data(csv_path=None):
    """Load lentiMPRA benchmark data.

    Args:
        csv_path: Optional path to a CSV with columns model, regime, cell_type,
            pearson_r (e.g. results/comparison_tables/mpra_comparison_fold0_subset_plot.csv
            produced by scripts/subset_compare_fold0.py). If None, uses the hardcoded
            full-test numbers below.

    Returns:
        DataFrame with columns: model, regime, cell_type, pearson_r, model_label
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        df["regime"] = df["regime"].fillna("").astype(str).replace("nan", "")
        df["model_label"] = df.apply(
            lambda row: row["model"] if row["regime"] == "" else f"{row['model']} ({row['regime']})",
            axis=1,
        )
        return df

    data = [
        # MPRALegNet
        {'model': 'MPRALegNet', 'regime': '', 'cell_type': 'HepG2', 'pearson_r':  0.781},
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
        {'model': 'AG MPRA', 'regime': 'Probing', 'cell_type': 'HepG2', 'pearson_r': 0.864},
        {'model': 'AG MPRA', 'regime': 'Probing', 'cell_type': 'K562', 'pearson_r': 0.853},
        {'model': 'AG MPRA', 'regime': 'Probing', 'cell_type': 'WTC11', 'pearson_r': 0.829},
        # AlphaGenome Fine-tuned
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'cell_type': 'HepG2', 'pearson_r': 0.887},
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'cell_type': 'K562', 'pearson_r': 0.879},
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'cell_type': 'WTC11', 'pearson_r': 0.839},
        # AlphaGenome random initialisation
        {'model': 'AG MPRA', 'regime': 'Random Init', 'cell_type': 'HepG2', 'pearson_r': 0.661},
        {'model': 'AG MPRA', 'regime': 'Random Init', 'cell_type': 'K562', 'pearson_r': 0.697},
        {'model': 'AG MPRA', 'regime': 'Random Init', 'cell_type': 'WTC11', 'pearson_r': 0.626},
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
        {'model': 'AG MPRA', 'regime': 'Probing', 'promoter_type': 'Developmental', 'pearson_r': 0.620},
        {'model': 'AG MPRA', 'regime': 'Probing', 'promoter_type': 'House-keeping', 'pearson_r': 0.631},
        # AlphaGenome Fine-tuned
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'promoter_type': 'Developmental', 'pearson_r': 0.739},
        {'model': 'AG MPRA', 'regime': 'Fine-tuned', 'promoter_type': 'House-keeping', 'pearson_r': 0.8},
    ]
    
    df = pd.DataFrame(data)
    
    # Create model_label column for plotting
    df['model_label'] = df.apply(
        lambda row: row['model'] if row['regime'] == '' else f"{row['model']} ({row['regime']})",
        axis=1
    )
    
    return df


def plot_lentimpra_benchmark(dat, pal, figsize=(8, 5), include_random_init=False, title='lentiMPRA'):
    """Create bar plot for lentiMPRA benchmark.

    Args:
        dat: DataFrame with lentiMPRA data
        pal: Color palette
        figsize: Figure size tuple
        title: Plot title (e.g. 'lentiMPRA (AG fold-0 held-out subset)').
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
    
    #filter random init
    if not include_random_init:
        dat = dat[dat['regime'] != 'Random Init']
    else:
        model_order.append('AG MPRA (Random Init)')
        model_colors['AG MPRA (Random Init)'] = pal[4]
            
    
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
                    height + 0.0025,
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
    ax.set_title(title, fontsize=14)
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cell_order)
    ax.set_ylim([0.5, 1])
    ax.grid(axis='y', alpha=0.5, linestyle='--')
    # Slightly raise legend position
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=10, ncol=2)
    
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
                    height + 0.0025,
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
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(promoter_order)
    ax.set_ylim([0.5, 1])
    ax.grid(axis='y', alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=10, ncol=1)
    
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
        '--lentimpra_csv',
        type=str,
        default=None,
        help='Optional CSV (model, regime, cell_type, pearson_r) to source lentiMPRA '
             'numbers from instead of the hardcoded full-test values. Use '
             'results/comparison_tables/mpra_comparison_fold0_subset_plot.csv for the '
             'AG fold-0 held-out-subset benchmark.'
    )
    parser.add_argument(
        '--lentimpra_title',
        type=str,
        default='lentiMPRA',
        help='Title for the lentiMPRA plot.'
    )
    parser.add_argument(
        '--lentimpra_name',
        type=str,
        default='lentimpra_benchmark',
        help='Output basename for the lentiMPRA plot (no extension). Use a distinct '
             'name (e.g. lentimpra_benchmark_fold0) when plotting a --lentimpra_csv so '
             'it does not overwrite the default full-test benchmark plot.'
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
        lentimpra_dat = load_lentimpra_data(csv_path=args.lentimpra_csv)
        fig_lentimpra = plot_lentimpra_benchmark(
            lentimpra_dat,
            pal,
            figsize=tuple(args.figsize_lentimpra),
            title=args.lentimpra_title
        )
        save_plots(
            fig_lentimpra,
            output_dir / args.lentimpra_name,
            dpi=args.dpi,
            formats=args.formats
        )
        plt.close(fig_lentimpra)
        print()
        # Plot random-initialisation variant only for the hardcoded full-test data
        # (the --lentimpra_csv fold-0 table has no Random Init series).
        if args.lentimpra_csv is None:
            fig_random = plot_lentimpra_benchmark(
                lentimpra_dat,
                pal,
                figsize=tuple(args.figsize_lentimpra),
                include_random_init=True
            )
            save_plots(
                fig_random,
                output_dir / f'{args.lentimpra_name}_random_init',
                dpi=args.dpi,
                formats=args.formats
            )
            plt.close(fig_random)
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
