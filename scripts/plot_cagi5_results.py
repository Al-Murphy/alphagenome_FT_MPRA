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
import re
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


def load_cagi5_data(results_dir, include_augmented=True):
    """Load CAGI5 per-element results from CSV files.
    
    Args:
        results_dir: Directory containing CSV files
        include_augmented: If True, also load files with augmentation prefixes
        
    Returns:
        DataFrame with all loaded data
    """
    # Load regular files
    pth = os.path.join(results_dir, "cagi5_*_per_element.csv")
    pths = glob.glob(pth)
    
    # Also load 501 central mask files
    pth_501 = os.path.join(results_dir, "501_central_mask_cagi5_*_per_element.csv")
    pths_501 = glob.glob(pth_501)
    
    all_pths = pths + pths_501
    
    # Load augmented files if requested
    if include_augmented:
        # Pattern for augmented files (e.g., posshift20_n3_revcomp_cagi5_base_K562_per_element.csv)
        pth_aug = os.path.join(results_dir, "*_cagi5_*_per_element.csv")
        pths_aug = glob.glob(pth_aug)
        # Filter to only include augmented files (exclude already loaded ones)
        pths_aug = [p for p in pths_aug if p not in all_pths and (
            "posshift" in os.path.basename(p) or 
            "revcomp" in os.path.basename(p)
        )]
        all_pths.extend(pths_aug)
    
    if not all_pths:
        raise FileNotFoundError(f"No CAGI5 per-element CSV files found in {results_dir}")
    
    dat = []
    for pth in all_pths:
        dat_i = pd.read_csv(pth)
        basename = os.path.basename(pth)
        
        # Parse filename to extract model type, cell type, and augmentation info
        # Remove _per_element.csv suffix
        name_base = basename.replace("_per_element.csv", "")
        
        # Check for augmentation prefixes
        has_posshift = "posshift" in name_base
        has_revcomp = "revcomp" in name_base
        is_501 = "501_central_mask" in name_base
        
        # Extract augmentation info
        aug_suffix = ""
        if has_posshift:
            # Extract posshift info (e.g., posshift20_n3)
            posshift_match = re.search(r'posshift(\d+)_n(\d+)', name_base)
            if posshift_match:
                max_shift, n_samples = posshift_match.groups()
                aug_suffix += f" (shift±{max_shift},n={n_samples})"
        if has_revcomp:
            aug_suffix += " (revcomp)"
        
        # Remove augmentation prefixes for parsing
        name_clean = name_base
        if is_501:
            name_clean = name_clean.replace("501_central_mask_", "")
        if has_posshift:
            # Remove posshift pattern
            name_clean = re.sub(r'posshift\d+_n\d+_', '', name_clean)
        if has_revcomp:
            name_clean = name_clean.replace("revcomp_", "")
        
        # Parse model and cell type
        parts = name_clean.split("_")
        if parts[1] == "base":
            model_name = "AG"
            if is_501:
                model_name = "AG (501bp)"
            dat_i["model"] = model_name + aug_suffix
            dat_i["cell_type"] = parts[2]
        elif parts[1] == "finetuned":
            dat_i["model"] = "AG MPRA" + aug_suffix
            dat_i["cell_type"] = parts[2]
        else:
            # Fallback
            dat_i["model"] = "Unknown"
            dat_i["cell_type"] = parts[2] if len(parts) > 2 else "Unknown"
        
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


def plot_augmentation_comparison(non_aug_dat_long, aug_dat_long, pal, figsize=(14, 5)):
    """Create plot comparing non-augmented vs augmented models.
    
    Args:
        non_aug_dat_long: Long-format data for non-augmented models
        aug_dat_long: Long-format data for augmented models
        pal: Color palette
        figsize: Figure size tuple
    """
    # Combine data and add augmentation flag
    non_aug_dat_long = non_aug_dat_long.copy()
    non_aug_dat_long['augmented'] = 'No Augmentation'
    
    aug_dat_long = aug_dat_long.copy()
    aug_dat_long['augmented'] = 'With Augmentation'
    
    # Extract base model name (remove augmentation suffix)
    # Note: 501bp models are already excluded from non_aug_dat, so we don't need to handle them here
    def get_base_model(model_name):
        model_str = str(model_name)
        # Remove augmentation suffixes
        if 'shift' in model_str or 'revcomp' in model_str:
            # Remove everything after first augmentation marker
            # e.g., "AG (shift±20,n=3) (revcomp)" -> "AG"
            # e.g., "AG (501bp) (shift±20,n=3)" -> "AG" (501bp models in aug_dat should be converted to base)
            parts = model_str.split(' (')
            if len(parts) > 1:
                # Check if first part after split is augmentation or 501bp
                first_part = parts[0]
                if 'shift' in parts[1] or 'revcomp' in parts[1]:
                    # First part is the base
                    return first_part
                elif '501bp' in parts[1]:
                    # Has 501bp with augmentation - convert to base "AG" for comparison
                    if len(parts) > 2 and ('shift' in parts[2] or 'revcomp' in parts[2]):
                        return first_part  # Return just "AG" for comparison
                    else:
                        # This shouldn't happen since we filtered out non-augmented 501bp, but handle it
                        return first_part  # "AG (501bp)" -> "AG"
        # For non-augmented models (should only be "AG" or "AG MPRA" at this point)
        # If somehow a 501bp model got through, convert it
        if '501bp' in model_str:
            return model_str.split(' (')[0]  # "AG (501bp)" -> "AG"
        return model_str
    
    non_aug_dat_long['base_model'] = non_aug_dat_long['model'].apply(get_base_model)
    aug_dat_long['base_model'] = aug_dat_long['model'].apply(get_base_model)
    
    # Combine
    combined_dat = pd.concat([non_aug_dat_long, aug_dat_long], ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    snp_types = ['All SNPs', 'High Confidence SNPs']
    
    for idx, snp_type in enumerate(snp_types):
        ax = axes[idx]
        data_subset = combined_dat[combined_dat['snp_type'] == snp_type]
        
        # Create grouped violin plot
        sns.violinplot(
            data=data_subset,
            x='base_model',
            y='pearson_r',
            hue='augmented',
            ax=ax,
            palette={'No Augmentation': pal[3], 'With Augmentation': pal[4]},
            inner=None,
            alpha=0.8
        )
        
        # Get x positions
        xtick_positions = ax.get_xticks()
        base_models = [label.get_text() for label in ax.get_xticklabels()]
        
        # Add mean lines and text for each group
        for i, base_model in enumerate(base_models):
            x_pos = xtick_positions[i]
            for aug_type, offset in [('No Augmentation', -0.2), ('With Augmentation', 0.2)]:
                group_data = data_subset[
                    (data_subset['base_model'] == base_model) & 
                    (data_subset['augmented'] == aug_type)
                ]
                if len(group_data) > 0:
                    mean_val = group_data['pearson_r'].mean()
                    color = pal[3] if aug_type == 'No Augmentation' else pal[4]
                    ax.plot(
                        [x_pos + offset - 0.15, x_pos + offset + 0.15],
                        [mean_val, mean_val],
                        color=color,
                        linestyle='--',
                        linewidth=2,
                        alpha=0.9,
                        zorder=5
                    )
                    ax.text(
                        x_pos + offset,
                        mean_val - 0.4,
                        f'μ={mean_val:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1)
                    )
        
        # Calculate stats for title
        ag_mpra_data = data_subset[data_subset['base_model'] == 'AG MPRA']
        n_total = ag_mpra_data['n'].sum() if len(ag_mpra_data) > 0 else 0
        n_elements = len(ag_mpra_data['element'].unique()) if len(ag_mpra_data) > 0 else 0
        title_parts = [f'{snp_type}', f'N={n_total} SNPs, {n_elements} regulatory elements']
        
        ax.set_title('\n'.join(title_parts))
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Pearson Correlation')
        ax.set_ylim([0, 1])
        if idx == 1:
            ax.legend(title='', loc='upper right', bbox_to_anchor=(1.60, 1), frameon=False)
        else:
            #remove legend all together
            ax.legend().remove()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_augmentation_comparison_by_cell(non_aug_dat_long, aug_dat_long, pal, figsize=(22, 6)):
    """Create plot comparing non-augmented vs augmented models split by cell line.
    
    Args:
        non_aug_dat_long: Long-format data for non-augmented models
        aug_dat_long: Long-format data for augmented models
        pal: Color palette
        figsize: Figure size tuple
    """
    # Combine data and add augmentation flag
    non_aug_dat_long = non_aug_dat_long.copy()
    non_aug_dat_long['augmented'] = 'No Augmentation'
    
    aug_dat_long = aug_dat_long.copy()
    aug_dat_long['augmented'] = 'With Augmentation'
    
    # Extract base model name (remove augmentation suffix and 501bp)
    def get_base_model(model_name):
        model_str = str(model_name)
        # Remove augmentation suffixes
        if 'shift' in model_str or 'revcomp' in model_str:
            # Remove everything after first augmentation marker
            parts = model_str.split(' (')
            if len(parts) > 1:
                first_part = parts[0]
                if 'shift' in parts[1] or 'revcomp' in parts[1]:
                    return first_part
                elif '501bp' in parts[1]:
                    if len(parts) > 2 and ('shift' in parts[2] or 'revcomp' in parts[2]):
                        return first_part
                    else:
                        return model_str
        # Remove 501bp for comparison if present
        if '501bp' in model_str:
            return model_str.split(' (')[0]
        return model_str
    
    non_aug_dat_long['base_model'] = non_aug_dat_long['model'].apply(get_base_model)
    aug_dat_long['base_model'] = aug_dat_long['model'].apply(get_base_model)
    
    # Create model_cell labels
    non_aug_dat_long['model_cell'] = non_aug_dat_long['base_model'] + ' (' + non_aug_dat_long['cell_type'] + ')'
    aug_dat_long['model_cell'] = aug_dat_long['base_model'] + ' (' + aug_dat_long['cell_type'] + ')'
    
    # Combine
    combined_dat = pd.concat([non_aug_dat_long, aug_dat_long], ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    snp_types = ['All SNPs', 'High Confidence SNPs']
    
    # Get unique model_cell combinations and create order
    model_cell_order = []
    for model in ['AG', 'AG MPRA']:
        for cell in ['K562', 'HepG2']:
            model_cell_order.append(f'{model} ({cell})')
    
    for idx, snp_type in enumerate(snp_types):
        ax = axes[idx]
        data_subset = combined_dat[combined_dat['snp_type'] == snp_type]
        
        # Create grouped violin plot
        sns.violinplot(
            data=data_subset,
            x='model_cell',
            y='pearson_r',
            hue='augmented',
            ax=ax,
            order=model_cell_order,
            palette={'No Augmentation': pal[3], 'With Augmentation': pal[4]},
            inner=None,
            alpha=0.8
        )
        
        # Get x positions
        xtick_positions = ax.get_xticks()
        
        # Add mean lines and text for each group
        for i, model_cell in enumerate(model_cell_order):
            x_pos = xtick_positions[i]
            for aug_type, offset in [('No Augmentation', -0.2), ('With Augmentation', 0.2)]:
                group_data = data_subset[
                    (data_subset['model_cell'] == model_cell) & 
                    (data_subset['augmented'] == aug_type)
                ]
                if len(group_data) > 0:
                    mean_val = group_data['pearson_r'].mean()
                    color = pal[3] if aug_type == 'No Augmentation' else pal[4]
                    ax.plot(
                        [x_pos + offset - 0.15, x_pos + offset + 0.15],
                        [mean_val, mean_val],
                        color=color,
                        linestyle='--',
                        linewidth=2,
                        alpha=0.9,
                        zorder=5
                    )
                    ax.text(
                        x_pos + offset,
                        mean_val - 0.4,
                        f'μ={mean_val:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1)
                    )
        
        # Calculate stats for title
        ag_mpra_data = data_subset[data_subset['base_model'] == 'AG MPRA']
        n_total = ag_mpra_data['n'].sum() if len(ag_mpra_data) > 0 else 0
        n_elements = len(ag_mpra_data['element'].unique()) if len(ag_mpra_data) > 0 else 0
        title_parts = [f'{snp_type}', f'N={n_total} SNPs, {n_elements} regulatory elements']
        
        ax.set_title('\n'.join(title_parts))
        ax.set_xlabel('Model (Cell Type)', fontsize=12)
        ax.set_ylabel('Pearson Correlation')
        ax.set_ylim([0, 1])
        #only plot legend for second plot
        if idx == 1:
            ax.legend(title='', loc='upper right', bbox_to_anchor=(.98, 1.22), frameon=False)
        else:
            #remove legend all together
            ax.legend().remove()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
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
    parser.add_argument(
        '--skip_aug_plots',
        action='store_true',
        help='Skip the augmentation comparison plots'
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
    dat = load_cagi5_data(results_dir, include_augmented=True)
    print(f"✓ Loaded data for {len(dat)} elements")
    
    # Check if we have augmented data
    has_augmented = any("shift" in str(m) or "revcomp" in str(m) for m in dat["model"].unique())
    if has_augmented:
        print(f"  Found augmented results: {[m for m in dat['model'].unique() if 'shift' in str(m) or 'revcomp' in str(m)]}")
    
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
    
    # Generate augmentation comparison plots if data is available
    if not args.skip_aug_plots and has_augmented:
        print("Generating augmentation comparison plots...")
        
        # Filter to only augmented models for comparison
        # Exclude 501bp models from non-augmented data to match other plots
        aug_dat = dat[dat["model"].str.contains("shift|revcomp", na=False, regex=True)].copy()
        # Only include non-augmented models that are NOT 501bp versions (to match other plots)
        non_aug_dat = dat[
            ~dat["model"].str.contains("shift|revcomp", na=False, regex=True) &
            ~dat["model"].str.contains("501bp", na=False, regex=True)
        ].copy()
        
        if len(aug_dat) > 0 and len(non_aug_dat) > 0:
            # Create comparison: base model vs augmented versions
            aug_dat_long = prepare_data(aug_dat)
            non_aug_dat_long = prepare_data(non_aug_dat)
            
            # Plot augmentation comparison by model
            print("  Generating augmentation comparison plot (by model)...")
            fig_aug_model = plot_augmentation_comparison(
                non_aug_dat_long, aug_dat_long, pal, figsize=(14, 5)
            )
            save_plots(
                fig_aug_model,
                output_dir / 'cagi5_augmentation_comparison',
                dpi=args.dpi,
                formats=args.formats
            )
            plt.close(fig_aug_model)
            print()
            
            # Plot augmentation comparison by model and cell
            print("  Generating augmentation comparison plot (by model and cell)...")
            fig_aug_cell = plot_augmentation_comparison_by_cell(
                non_aug_dat_long, aug_dat_long, pal, figsize=(22, 6)
            )
            save_plots(
                fig_aug_cell,
                output_dir / 'cagi5_augmentation_comparison_by_cell',
                dpi=args.dpi,
                formats=args.formats
            )
            plt.close(fig_aug_cell)
            print()
        else:
            print("  Skipping augmentation plots: insufficient data")
            print()
    
    print("=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
