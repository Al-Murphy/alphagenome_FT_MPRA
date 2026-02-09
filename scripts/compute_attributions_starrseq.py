#!/usr/bin/env python3
"""
Compute attribution maps and sequence logos for DeepSTARR sequences.

This script computes attributions using various methods (DeepSHAP, gradients, etc.)
and generates visualization plots for specified sequences from the DeepSTARR test dataset.

DeepSTARR predicts two types of enhancer activity:
- Developmental enhancer activity (Dev_log2_enrichment) - output_index=0
- Housekeeping enhancer activity (Hk_log2_enrichment) - output_index=1

USAGE:
    # Compute attributions for a specific sequence (developmental task)
    python scripts/compute_attributions_starrseq.py \
        --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
        --sequence_id 0 \
        --output_task dev

    # Compute attributions for top 10 sequences by developmental activity
    python scripts/compute_attributions_starrseq.py \
        --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
        --top_n 10 \
        --output_task dev \
        --attribution_method deepshap

    # Compute attributions for housekeeping task
    python scripts/compute_attributions_starrseq.py \
        --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
        --top_n 5 \
        --output_task hk \
        --attribution_method gradient_x_input

    # Test motif in dinucleotide-shuffled backgrounds (developmental task)
    python scripts/compute_attributions_starrseq.py \
        --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
        --top_n 20 \
        --motif CAAAG \
        --shuffle_background \
        --output_task dev \
        --attribution_method deepshap

    # Export sequences for FIMO motif discovery
    python scripts/compute_attributions_starrseq.py \
        --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
        --top_n 20 \
        --output_task dev \
        --attribution_method deepshap \
        --export_for_fimo
"""

import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
import traceback
import random

from alphagenome_ft import load_checkpoint, register_custom_head, HeadConfig, HeadType
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from src import DeepSTARRHead, DeepSTARRDataset, STARRSeqDataLoader

# Import utility functions from the MPRA attribution script
# Both scripts are in the same directory, so we can import directly
import sys
from pathlib import Path

# Add scripts directory to path to allow importing compute_attributions
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import from compute_attributions module
from compute_attributions import (
    parse_meme_file,
    pfm_to_consensus,
    pwm_to_consensus,
    parse_jaspar_pfm,
    sample_motif_from_pfm,
    dinucleotide_shuffle,
    insert_motif,
    decode_one_hot,
    one_hot_encode,
    compute_attributions,
    extract_seqlets,
    export_sequences_for_fimo,
    generate_plots,
)


def find_top_sequences(dataset, n=10, output_task='dev'):
    """Find top N sequences by activity for specified task.
    
    Args:
        dataset: DeepSTARRDataset instance
        n: Number of top sequences to return
        output_task: 'dev' for developmental, 'hk' for housekeeping
    """
    print(f"Finding top {n} sequences by {output_task} activity...")
    activities = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        y = sample['y']  # Shape: (2,) - [dev, hk]
        if output_task == 'dev':
            activity = float(y[0])
        elif output_task == 'hk':
            activity = float(y[1])
        else:
            raise ValueError(f"Unknown output_task: {output_task}. Must be 'dev' or 'hk'")
        activities.append((idx, activity))
    
    # Sort by activity (descending)
    activities.sort(key=lambda x: x[1], reverse=True)
    top_n = activities[:n]
    
    print(f"✓ Found top {n} sequences:")
    for rank, (idx, activity) in enumerate(top_n, 1):
        print(f"  {rank}. idx={idx}, {output_task} activity={activity:.6f}")
    print()
    
    return top_n


def sample_random_sequences(dataset, n=10, random_state=None):
    """Randomly sample N sequences from dataset."""
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    print(f"Randomly sampling {n} sequences...")
    total_sequences = len(dataset)
    if n > total_sequences:
        print(f"  Warning: Requested {n} sequences but only {total_sequences} available. Using all sequences.")
        n = total_sequences
    
    # Randomly sample indices
    indices = np.random.choice(total_sequences, size=n, replace=False)
    # Get activities for sampled sequences
    sampled = []
    for idx in indices:
        sample = dataset[int(idx)]
        y = sample['y']  # Shape: (2,) - [dev, hk]
        sampled.append((int(idx), float(y[0]), float(y[1])))  # (idx, dev_activity, hk_activity)
    
    print(f"✓ Sampled {len(sampled)} sequences:")
    for rank, (idx, dev_act, hk_act) in enumerate(sampled, 1):
        print(f"  {rank}. idx={idx}, dev_activity={dev_act:.6f}, hk_activity={hk_act:.6f}")
    print()
    
    return sampled


def process_sequence(model, dataset, sequence_idx, output_base_dir, method='deepshap', head_name='deepstarr_head',
                     output_task='dev', motif=None, motif_position=None, motif_center=False, 
                     shuffle_background=False, n_shuffles=20, return_attributions=False, **kwargs):
    """Process a single sequence: compute attributions and generate plots.
    
    Args:
        model: Trained model
        dataset: Test dataset
        sequence_idx: Index of sequence in dataset
        output_base_dir: Base output directory
        method: Attribution method
        head_name: Name of custom head
        output_task: 'dev' for developmental, 'hk' for housekeeping
        motif: Optional motif sequence to insert
        motif_position: Position to insert motif
        motif_center: If True, insert motif at center position
        shuffle_background: If True, shuffle sequence before inserting motif
        return_attributions: If True, return attributions instead of generating plots
        **kwargs: Additional arguments for attribution computation
    
    Returns:
        If return_attributions=True: tuple (attributions, sequence_str, motif_position)
        Otherwise: list of generated plot paths
    """
    print(f"\n{'='*80}")
    print(f"Processing sequence {sequence_idx} (output_task={output_task})")
    if motif:
        print(f"  Motif: {motif}")
        if shuffle_background:
            print(f"  Background: Dinucleotide-shuffled")
    print(f"{'='*80}")
    
    # Get sequence from dataset
    sample = dataset[sequence_idx]
    sequence = sample['seq']  # Shape: (length, 4) from dataset
    organism_index = sample['organism_index']  # Shape: () or (1,)
    y = sample['y']  # Shape: (2,) - [dev, hk]
    
    # Get activity for the specified task
    if output_task == 'dev':
        activity = float(y[0])
        output_index = 0
    elif output_task == 'hk':
        activity = float(y[1])
        output_index = 1
    else:
        raise ValueError(f"Unknown output_task: {output_task}. Must be 'dev' or 'hk'")
    
    # Ensure sequence is a JAX array and add batch dimension if missing
    sequence = jnp.array(sequence)
    if sequence.ndim == 2:
        sequence = sequence[None, :, :]  # (length, 4) -> (1, length, 4)
    
    # Ensure organism_index is a JAX array with batch dimension
    organism_index = jnp.array(organism_index)
    if organism_index.ndim == 0:
        organism_index = organism_index[None]  # () -> (1,)
    
    # Decode sequence string
    sequence_str = decode_one_hot(sequence)
    original_sequence_str = sequence_str
    
    # DeepSTARR sequences have adapters: upstream_adapter + CRE + downstream_adapter
    # Extract the CRE part (middle 249bp for DeepSTARR)
    # Adapters are: upstream="TCCCTACACGACGCTCTTCCGATCT" (25bp), downstream="AGATCGGAAGAGCACACGTCTGAACT" (26bp)
    upstream_adapter = "TCCCTACACGACGCTCTTCCGATCT"
    downstream_adapter = "AGATCGGAAGAGCACACGTCTGAACT"
    
    # Find CRE region (between adapters)
    cre_start = len(upstream_adapter)
    cre_end = len(sequence_str) - len(downstream_adapter)
    cre_seq = sequence_str[cre_start:cre_end]
    
    # Handle motif insertion
    motif_insertion_pos = None
    if motif:
        # Insert motif into CRE region
        test_seq_str = cre_seq
        
        # Handle multiple shuffles if shuffle_background is True
        if shuffle_background and motif:
            # Shuffle background multiple times and average attributions
            print(f"  Shuffling background {n_shuffles} times and averaging attributions...")
            all_shuffle_attributions = []
            all_shuffle_sequences = []
            all_motif_positions = []
            
            base_random_state = kwargs.get('random_state', 42)
            
            for shuffle_idx in range(n_shuffles):
                # Use different random state for each shuffle
                shuffle_random_state = base_random_state + shuffle_idx if base_random_state is not None else None
                
                # Shuffle the CRE sequence
                shuffled_cre = dinucleotide_shuffle(test_seq_str, random_state=shuffle_random_state)
                
                # Insert motif into shuffled CRE
                try:
                    shuffled_cre_with_motif, shuffle_motif_pos = insert_motif(
                        shuffled_cre, motif, position=motif_position, center=motif_center, 
                        random_state=shuffle_random_state
                    )
                    
                    # Reconstruct full sequence with adapters
                    shuffle_sequence_str = upstream_adapter + shuffled_cre_with_motif + downstream_adapter
                    shuffle_sequence = one_hot_encode(shuffle_sequence_str)
                    
                    # Compute attributions for this shuffle (without mean correction for now)
                    shuffle_attributions_raw = compute_attributions(
                        model, shuffle_sequence, organism_index, method=method, head_name=head_name, 
                        apply_mean_correction=False, output_index=output_index, **kwargs
                    )
                    
                    all_shuffle_attributions.append(shuffle_attributions_raw)
                    all_shuffle_sequences.append(shuffle_sequence_str)
                    all_motif_positions.append(shuffle_motif_pos + cre_start)  # Adjust for adapter offset
                    
                except Exception as e:
                    print(f"    ✗ Error in shuffle {shuffle_idx+1}/{n_shuffles}: {e}")
                    continue
            
            if not all_shuffle_attributions:
                print(f"  ✗ No successful shuffles, cannot compute averaged attributions")
                if return_attributions:
                    return None, None, None
                return []
            
            # Average attributions across all shuffles (raw, before mean correction)
            stacked_attributions = jnp.stack(all_shuffle_attributions)  # (n_shuffles, batch, seq_len, 4)
            attributions_raw = jnp.mean(stacked_attributions, axis=0)  # (batch, seq_len, 4)
            
            # Apply mean correction AFTER averaging across shuffles
            if method == 'deepshap' or method == 'gradient_x_input':
                attributions = attributions_raw - jnp.mean(attributions_raw, axis=-1, keepdims=True)
            else:
                attributions = attributions_raw
            
            # Use first shuffle's sequence and motif position for display
            sequence_str = all_shuffle_sequences[0]
            motif_insertion_pos = all_motif_positions[0]
            sequence = one_hot_encode(sequence_str)
            
            print(f"  ✓ Averaged attributions over {len(all_shuffle_attributions)} shuffles")
            
        elif shuffle_background:
            # Single shuffle (no motif case - shouldn't happen but handle gracefully)
            shuffled_cre = dinucleotide_shuffle(test_seq_str, random_state=kwargs.get('random_state'))
            print(f"  ✓ CRE sequence shuffled (dinucleotide-preserving)")
            sequence_str = upstream_adapter + shuffled_cre + downstream_adapter
            sequence = one_hot_encode(sequence_str)
        
        # Insert motif into CRE (if not already done in shuffle loop)
        if motif and not (shuffle_background and motif):
            try:
                cre_with_motif, cre_motif_pos = insert_motif(
                    test_seq_str, motif, position=motif_position, center=motif_center, 
                    random_state=kwargs.get('random_state')
                )
                print(f"  ✓ Motif inserted at position {cre_motif_pos} in CRE sequence")
                
                # Reconstruct full sequence with adapters
                sequence_str = upstream_adapter + cre_with_motif + downstream_adapter
                motif_insertion_pos = cre_motif_pos + cre_start  # Adjust for adapter offset
                
                # Re-encode sequence with motif
                sequence = one_hot_encode(sequence_str)
            except Exception as e:
                print(f"  ✗ Error inserting motif: {e}")
                if return_attributions:
                    return None, None, None
                return []
    
    print(f"  Sequence length: {len(sequence_str)}bp")
    print(f"  CRE region: {cre_start}-{cre_end}bp")
    print(f"  Sequence (first 50bp): {sequence_str[:50]}...")
    if motif and motif_insertion_pos is not None:
        # Highlight motif in sequence preview
        motif_end = min(motif_insertion_pos + len(motif), len(sequence_str))
        print(f"  Motif region ({motif_insertion_pos}-{motif_end}): {sequence_str[motif_insertion_pos:motif_end]}")
    print(f"  {output_task} activity: {activity:.6f}")
    
    # Create output directory (only if output_base_dir is provided and not returning attributions)
    if return_attributions:
        # Don't create output directory when just collecting attributions for averaging
        output_dir = None
    elif output_base_dir is not None:
        if motif:
            output_dir = Path(output_base_dir) / f'seq{sequence_idx}_{output_task}_motif_{motif}'
            if shuffle_background:
                output_dir = Path(output_base_dir) / f'seq{sequence_idx}_{output_task}_motif_{motif}_shuffled'
        else:
            output_dir = Path(output_base_dir) / f'seq{sequence_idx}_{output_task}'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {output_dir}")
    else:
        output_dir = None
    
    # Compute attributions (if not already computed in shuffle loop)
    if not (shuffle_background and motif):
        print(f"\n  Computing {method} attributions for {output_task} task (output_index={output_index})...")
        try:
            # Get raw attributions first (before mean correction)
            attributions_raw = compute_attributions(
                model, sequence, organism_index, method=method, head_name=head_name, 
                apply_mean_correction=False, output_index=output_index, **kwargs
            )
            # Apply mean correction if needed
            if method == 'deepshap' or method == 'gradient_x_input':
                attributions = attributions_raw - jnp.mean(attributions_raw, axis=-1, keepdims=True)
            else:
                attributions = attributions_raw
            
            print(f"  ✓ Attributions computed: shape {attributions.shape}")
            print(f"    Stats: min={jnp.min(attributions):.6f}, max={jnp.max(attributions):.6f}, mean={jnp.mean(attributions):.6f}")
            
            # If motif was inserted, highlight motif region in attributions
            if motif and motif_insertion_pos is not None:
                motif_attributions_corrected = attributions[0, motif_insertion_pos:motif_insertion_pos+len(motif), :]
                motif_attributions_raw = attributions_raw[0, motif_insertion_pos:motif_insertion_pos+len(motif), :]
                
                # Sum of corrected (will be ~0 due to mean correction)
                motif_attrib_sum_corrected = jnp.sum(motif_attributions_corrected)
                # Sum of raw (actual signal)
                motif_attrib_sum_raw = jnp.sum(motif_attributions_raw)
                # Sum of absolute values (magnitude)
                motif_attrib_sum_abs = jnp.sum(jnp.abs(motif_attributions_corrected))
                
                print(f"    Motif region (raw, before mean correction): sum={motif_attrib_sum_raw:.6f}")
                print(f"    Motif region (corrected, after mean correction): sum={motif_attrib_sum_corrected:.6f} (expected ~0)")
                print(f"    Motif region (corrected, absolute): sum={motif_attrib_sum_abs:.6f}")
        except Exception as e:
            print(f"  ✗ Error computing attributions: {e}")
            traceback.print_exc()
            if return_attributions:
                return None, None, None
            return []
    else:
        # Attributions already computed and averaged in shuffle loop
        print(f"\n  ✓ Attributions (averaged over {n_shuffles} shuffles): shape {attributions.shape}")
        print(f"    Stats: min={jnp.min(attributions):.6f}, max={jnp.max(attributions):.6f}, mean={jnp.mean(attributions):.6f}")
        
        # If motif was inserted, highlight motif region in attributions
        if motif and motif_insertion_pos is not None:
            motif_attributions_corrected = attributions[0, motif_insertion_pos:motif_insertion_pos+len(motif), :]
            motif_attributions_raw = attributions_raw[0, motif_insertion_pos:motif_insertion_pos+len(motif), :]
            
            # Sum of corrected (will be ~0 due to mean correction)
            motif_attrib_sum_corrected = jnp.sum(motif_attributions_corrected)
            # Sum of raw (actual signal)
            motif_attrib_sum_raw = jnp.sum(motif_attributions_raw)
            # Sum of absolute values (magnitude)
            motif_attrib_sum_abs = jnp.sum(jnp.abs(motif_attributions_corrected))
            
            print(f"    Motif region (raw, before mean correction): sum={motif_attrib_sum_raw:.6f}")
            print(f"    Motif region (corrected, after mean correction): sum={motif_attrib_sum_corrected:.6f} (expected ~0)")
            print(f"    Motif region (corrected, absolute): sum={motif_attrib_sum_abs:.6f}")
    
    # Return attributions if requested (for averaging)
    if return_attributions:
        return attributions, sequence_str, motif_insertion_pos
    
    # Generate plots (only if output_dir is provided)
    if output_dir is not None:
        print(f"\n  Generating plots...")
        plots = generate_plots(model, sequence, attributions, sequence_str, output_dir, method=method, head_name=head_name)
        
        # Save motif information if applicable
        if motif and motif_insertion_pos is not None:
            motif_info_path = output_dir / 'motif_info.txt'
            with open(motif_info_path, 'w') as f:
                f.write(f"Motif: {motif}\n")
                f.write(f"Insertion position: {motif_insertion_pos}\n")
                f.write(f"CRE insertion position: {motif_insertion_pos - cre_start}\n")
                f.write(f"Original sequence: {original_sequence_str}\n")
                f.write(f"Modified sequence: {sequence_str}\n")
                f.write(f"Shuffled background: {shuffle_background}\n")
                f.write(f"Output task: {output_task}\n")
                f.write(f"Activity: {activity:.6f}\n")
            print(f"  ✓ Motif info saved: {motif_info_path.name}")
        
        print(f"\n  ✓ Completed sequence {sequence_idx}")
        print(f"    Plots saved to: {output_dir}")
    else:
        # Just collecting attributions, no plots
        plots = []
        print(f"\n  ✓ Collected attributions for sequence {sequence_idx}")
    
    return plots


def main():
    parser = argparse.ArgumentParser(
        description='Compute attribution maps and sequence logos for DeepSTARR sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--output_task',
        type=str,
        required=True,
        choices=['dev', 'hk'],
        help='Output task: "dev" for developmental enhancer activity, "hk" for housekeeping enhancer activity'
    )
    
    # Sequence selection (mutually exclusive)
    seq_group = parser.add_mutually_exclusive_group(required=True)
    seq_group.add_argument(
        '--sequence_id',
        type=int,
        help='Specific test sequence ID to analyze'
    )
    seq_group.add_argument(
        '--top_n',
        type=int,
        help='Number of top sequences by activity to analyze (e.g., 10 for top 10)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/deepstarr',
        help='Path to DeepSTARR data directory (default: ./data/deepstarr)'
    )
    parser.add_argument(
        '--attribution_method',
        type=str,
        default='deepshap',
        choices=['deepshap', 'gradient', 'gradient_x_input', 'ism'],
        help='Attribution method to use (default: deepshap). '
             'Options: deepshap, gradient, gradient_x_input, ism'
    )
    parser.add_argument(
        '--head_name',
        type=str,
        default='deepstarr_head',
        help='Name of the custom head to use (default: deepstarr_head)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/plots',
        help='Base output directory for plots (default: results/plots)'
    )
    parser.add_argument(
        '--base_checkpoint_path',
        type=str,
        default=None,
        help='Optional path to base AlphaGenome checkpoint (if not using Kaggle)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu', 'cuda'],
        help='Device to use (default: auto - tries GPU first, falls back to CPU)'
    )
    parser.add_argument(
        '--n_references',
        type=int,
        default=20,
        help='Number of reference sequences for DeepSHAP (default: 20)'
    )
    parser.add_argument(
        '--reference_type',
        type=str,
        default='shuffle',
        choices=['shuffle', 'uniform', 'gc_match'],
        help='Type of reference sequences for DeepSHAP (default: shuffle)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--export_for_fimo',
        action='store_true',
        help='Export sequences and high-attribution regions (seqlets) as FASTA files for FIMO motif discovery'
    )
    parser.add_argument(
        '--seqlet_threshold',
        type=float,
        default=90.0,
        help='Percentile threshold for extracting high-attribution seqlets (default: 90.0)'
    )
    parser.add_argument(
        '--seqlet_min_length',
        type=int,
        default=5,
        help='Minimum seqlet length in bp (default: 5)'
    )
    parser.add_argument(
        '--seqlet_max_length',
        type=int,
        default=50,
        help='Maximum seqlet length in bp (default: 50)'
    )
    
    # Motif analysis arguments
    motif_group = parser.add_mutually_exclusive_group()
    motif_group.add_argument(
        '--motif',
        type=str,
        default=None,
        help='Motif sequence to insert (e.g., "AGGTCA" for HNF4A). If provided, motif will be inserted into sequences.'
    )
    motif_group.add_argument(
        '--motif_file',
        type=str,
        default=None,
        help='Path to motif file (PFM or MEME format from JASPAR). If provided, motif will be extracted and inserted.'
    )
    parser.add_argument(
        '--motif_consensus_method',
        type=str,
        default='max',
        choices=['max', 'sample'],
        help='Method to convert PFM/PWM to consensus sequence: "max" uses most frequent/probable base, "sample" samples from distribution (default: max)'
    )
    parser.add_argument(
        '--motif_position',
        type=int,
        default=None,
        help='Position to insert motif (default: None = random position)'
    )
    parser.add_argument(
        '--shuffle_background',
        action='store_true',
        help='If set, shuffle sequence background (dinucleotide-preserving) before inserting motif'
    )
    parser.add_argument(
        '--n_shuffles',
        type=int,
        default=20,
        help='Number of background shuffles to average over when shuffle_background=True (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Parse motif file if provided (same logic as MPRA script)
    motif_sequence = args.motif
    motif_name = None
    if args.motif_file:
        motif_file_path = Path(args.motif_file)
        if not motif_file_path.exists():
            print(f"Error: Motif file not found: {motif_file_path}")
            sys.exit(1)
        
        print(f"\n{'='*80}")
        print("Parsing motif file...")
        print("="*80)
        
        try:
            if motif_file_path.suffix.lower() == '.pfm' or 'pfm' in motif_file_path.name.lower():
                motif_id, motif_name, pfm_matrix, consensus_seq = parse_jaspar_pfm(motif_file_path)
                print(f"  File format: PFM (JASPAR)")
                print(f"  Motif ID: {motif_id}")
                print(f"  Motif name: {motif_name}")
                print(f"  Motif length: {pfm_matrix.shape[1]}bp")
                if args.motif_consensus_method == 'max':
                    motif_sequence = consensus_seq
                else:
                    motif_sequence = pfm_to_consensus(pfm_matrix, method=args.motif_consensus_method, random_state=args.random_state)
                print(f"  Consensus sequence ({args.motif_consensus_method}): {motif_sequence}")
            elif motif_file_path.suffix.lower() == '.meme' or 'meme' in motif_file_path.name.lower():
                motif_name, pwm_matrix = parse_meme_file(motif_file_path)
                print(f"  File format: MEME")
                print(f"  Motif name: {motif_name}")
                print(f"  Motif length: {pwm_matrix.shape[0]}bp")
                motif_sequence = pwm_to_consensus(pwm_matrix, method=args.motif_consensus_method, random_state=args.random_state)
                print(f"  Consensus sequence ({args.motif_consensus_method}): {motif_sequence}")
            else:
                # Try to auto-detect format
                try:
                    motif_id, motif_name, pfm_matrix, consensus_seq = parse_jaspar_pfm(motif_file_path)
                    print(f"  Auto-detected format: PFM (JASPAR)")
                    print(f"  Motif ID: {motif_id}")
                    print(f"  Motif name: {motif_name}")
                    print(f"  Motif length: {pfm_matrix.shape[1]}bp")
                    if args.motif_consensus_method == 'max':
                        motif_sequence = consensus_seq
                    else:
                        motif_sequence = pfm_to_consensus(pfm_matrix, method=args.motif_consensus_method, random_state=args.random_state)
                    print(f"  Consensus sequence ({args.motif_consensus_method}): {motif_sequence}")
                except:
                    motif_name, pwm_matrix = parse_meme_file(motif_file_path)
                    print(f"  Auto-detected format: MEME")
                    print(f"  Motif name: {motif_name}")
                    print(f"  Motif length: {pwm_matrix.shape[0]}bp")
                    motif_sequence = pwm_to_consensus(pwm_matrix, method=args.motif_consensus_method, random_state=args.random_state)
                    print(f"  Consensus sequence ({args.motif_consensus_method}): {motif_sequence}")
        except Exception as e:
            print(f"  ✗ Error parsing motif file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("✓ Motif file parsed successfully\n")
    
    # Determine device
    if args.device == 'auto':
        try:
            device = jax.devices('gpu')[0]
            print(f"Using device: {device}")
        except (IndexError, RuntimeError):
            try:
                device = jax.devices('cuda')[0]
                print(f"Using device: {device}")
            except (IndexError, RuntimeError):
                device = jax.devices('cpu')[0]
                print(f"Using device: {device} (GPU not available)")
    elif args.device in ['gpu', 'cuda']:
        try:
            device = jax.devices('gpu')[0] if args.device == 'gpu' else jax.devices('cuda')[0]
        except (IndexError, RuntimeError):
            print(f"Warning: {args.device} not available, falling back to CPU")
            device = jax.devices('cpu')[0]
    else:
        device = jax.devices('cpu')[0]
    
    # Load checkpoint config to get head metadata
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    head_metadata = {
        'center_bp': 256,
        'pooling_type': 'flatten',
        'nl_size': 512,
        'do': 0.5,
        'activation': 'relu',
    }
    config_path = checkpoint_dir / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            head_cfg = (cfg.get('head_configs', {})
                           .get('deepstarr_head', {})
                           .get('metadata', {}))
            head_metadata.update(head_cfg)
            print(f"Loaded head metadata from checkpoint: {head_metadata}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint config: {e}")
    
    # Register custom head (required before loading checkpoint)
    print("\n" + "="*80)
    print("Registering custom DeepSTARR head...")
    print("="*80)
    register_custom_head(
        'deepstarr_head',
        DeepSTARRHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='deepstarr_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=2,  # Two outputs: developmental and housekeeping
            metadata=head_metadata
        )
    )
    print("✓ Head registered\n")
    
    # Load trained model using load_checkpoint (now handles minimal models correctly)
    print("="*80)
    print("Loading trained model...")
    print("="*80)
    
    # For DeepSTARR, infer init_seq_len from checkpoint or use default
    # DeepSTARR sequences are 300bp (249bp CRE + 25bp upstream adapter + 26bp downstream adapter)
    init_seq_len = None
    if head_metadata.get('pooling_type') == 'flatten':
        # For flatten pooling, use center_bp as init_seq_len (matches training)
        init_seq_len = head_metadata.get('center_bp', 300)
        print(f"Using init_seq_len={init_seq_len}bp from center_bp (flatten pooling)")
    else:
        # For other pooling types, use default DeepSTARR sequence length
        init_seq_len = 300
        print(f"Using init_seq_len={init_seq_len}bp (default DeepSTARR sequence length)")
    
    model = load_checkpoint(
        str(checkpoint_dir),
        base_model_version='all_folds',
        base_checkpoint_path=args.base_checkpoint_path,
        device=device,
        init_seq_len=init_seq_len,
    )
    print("✓ Model loaded successfully")
    print(f"  Custom heads: {model._custom_heads}")
    print()
    
    # Load test dataset
    print("="*80)
    print("Loading test dataset...")
    print("="*80)
    test_dataset = DeepSTARRDataset(
        model=model,
        path_to_data=args.data_path,
        split='test',
        organism=dna_model.Organism.HOMO_SAPIENS,
        random_shift=False,
        reverse_complement=False
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples\n")
    
    # Determine which sequences to process
    if args.sequence_id is not None:
        if args.sequence_id < 0 or args.sequence_id >= len(test_dataset):
            print(f"Error: sequence_id {args.sequence_id} is out of range (0-{len(test_dataset)-1})")
            sys.exit(1)
        sequences_to_process = [(args.sequence_id, None, None)]  # (idx, dev_activity, hk_activity)
    else:
        # If motif is provided, use random sampling instead of top by activity
        if motif_sequence:
            sequences_to_process = sample_random_sequences(
                test_dataset, n=args.top_n, random_state=args.random_state
            )
        else:
            # Find top N sequences by activity for the specified task
            top_sequences = find_top_sequences(test_dataset, n=args.top_n, output_task=args.output_task)
            sequences_to_process = [(idx, None, None) for idx, _ in top_sequences]
    
    # Create base output directory
    output_base_dir = Path(args.output_dir) / f'attribution_deepstarr_{args.output_task}_{args.attribution_method}'
    if motif_sequence:
        motif_label = motif_name if motif_name else motif_sequence
        motif_label = motif_label.replace(' ', '_').replace('/', '_')
        output_base_dir = output_base_dir / f'motif_{motif_label}'
        if args.shuffle_background:
            output_base_dir = output_base_dir / 'shuffled_background'
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base output directory: {output_base_dir}\n")
    
    # Print motif analysis info if applicable
    if motif_sequence:
        print("="*80)
        print("MOTIF ANALYSIS MODE")
        print("="*80)
        if motif_name:
            print(f"Motif name: {motif_name}")
        print(f"Motif sequence: {motif_sequence}")
        print(f"Motif length: {len(motif_sequence)}bp")
        if args.motif_file:
            print(f"Source: {args.motif_file}")
            print(f"Consensus method: {args.motif_consensus_method}")
        print(f"Output task: {args.output_task}")
        if args.shuffle_background:
            print(f"Background: Dinucleotide-shuffled")
            print(f"  Number of shuffles per sequence: {args.n_shuffles}")
        else:
            print(f"Background: Original test sequences")
        print()
    
    # Determine if we should average attributions (when using top_n with motif)
    average_attributions = (args.top_n is not None and motif_sequence is not None)
    use_center_position = average_attributions and args.motif_position is None
    
    # Process each sequence
    all_plots = []
    all_attributions = []
    all_sequences = []
    all_motif_positions = []
    sequences_for_fimo = []
    
    for seq_data in sequences_to_process:
        if len(seq_data) == 3:
            seq_idx, dev_act, hk_act = seq_data
        else:
            seq_idx = seq_data[0]
            dev_act = hk_act = None
        
        if average_attributions:
            # Collect attributions for averaging
            result = process_sequence(
                model=model,
                dataset=test_dataset,
                sequence_idx=seq_idx,
                output_base_dir=None,
                method=args.attribution_method,
                head_name=args.head_name,
                output_task=args.output_task,
                motif=motif_sequence,
                motif_position=args.motif_position,
                motif_center=use_center_position,
                shuffle_background=args.shuffle_background,
                n_shuffles=args.n_shuffles,
                n_references=args.n_references,
                reference_type=args.reference_type,
                random_state=args.random_state,
                return_attributions=True,
            )
            if result[0] is not None:
                attributions, seq_str, motif_pos = result
                all_attributions.append(attributions)
                all_sequences.append(seq_str)
                all_motif_positions.append(motif_pos)
                
                # Store for FIMO export if requested
                if args.export_for_fimo:
                    sample = test_dataset[seq_idx]
                    y = sample['y']
                    sequences_for_fimo.append({
                        'sequence_idx': seq_idx,
                        'sequence_str': seq_str,
                        'attributions': attributions,
                        'activity': float(y[0] if args.output_task == 'dev' else y[1])
                    })
        else:
            # Generate individual plots
            plots = process_sequence(
                model=model,
                dataset=test_dataset,
                sequence_idx=seq_idx,
                output_base_dir=output_base_dir,
                method=args.attribution_method,
                head_name=args.head_name,
                output_task=args.output_task,
                motif=motif_sequence,
                motif_position=args.motif_position,
                motif_center=use_center_position,
                shuffle_background=args.shuffle_background,
                n_shuffles=args.n_shuffles,
                n_references=args.n_references,
                reference_type=args.reference_type,
                random_state=args.random_state,
            )
            all_plots.extend(plots)
            
            # Store for FIMO export if requested
            if args.export_for_fimo:
                sample = test_dataset[seq_idx]
                seq = sample['seq']
                seq_jax = jnp.array(seq)
                if seq_jax.ndim == 2:
                    seq_jax = seq_jax[None, :, :]
                seq_str = decode_one_hot(seq_jax)
                
                org_idx = jnp.array(sample['organism_index'])
                if org_idx.ndim == 0:
                    org_idx = org_idx[None]
                
                output_index = 0 if args.output_task == 'dev' else 1
                attr = compute_attributions(
                    model, seq_jax, org_idx, method=args.attribution_method, 
                    head_name=args.head_name,
                    n_references=args.n_references,
                    reference_type=args.reference_type,
                    random_state=args.random_state,
                    output_index=output_index,
                )
                
                y = sample['y']
                sequences_for_fimo.append({
                    'sequence_idx': seq_idx,
                    'sequence_str': seq_str,
                    'attributions': attr,
                    'activity': float(y[0] if args.output_task == 'dev' else y[1])
                })
    
    # Average attributions if requested
    if average_attributions and all_attributions:
        print("\n" + "="*80)
        print("AVERAGING ATTRIBUTIONS")
        print("="*80)
        print(f"Averaging attributions over {len(all_attributions)} sequences")
        
        # Stack and average
        stacked_attributions = jnp.stack(all_attributions)
        averaged_attributions = jnp.mean(stacked_attributions, axis=0)
        
        print(f"✓ Averaged attributions: shape {averaged_attributions.shape}")
        print(f"  Stats: min={jnp.min(averaged_attributions):.6f}, max={jnp.max(averaged_attributions):.6f}, mean={jnp.mean(averaged_attributions):.6f}")
        
        # Get representative sequence
        if all_sequences:
            representative_seq = all_sequences[0]
            motif_pos = all_motif_positions[0] if all_motif_positions else None
            
            if motif_pos is not None:
                motif_attributions = averaged_attributions[0, motif_pos:motif_pos+len(motif_sequence), :]
                motif_attrib_sum = jnp.sum(motif_attributions)
                motif_attrib_sum_abs = jnp.sum(jnp.abs(motif_attributions))
                print(f"  Motif region (pos {motif_pos}-{motif_pos+len(motif_sequence)}) attribution sum: {motif_attrib_sum:.6f} (corrected, expected ~0)")
                print(f"  Motif region (pos {motif_pos}-{motif_pos+len(motif_sequence)}) attribution sum (absolute): {motif_attrib_sum_abs:.6f}")
        
        # Create output directory for averaged results
        avg_output_dir = output_base_dir / 'averaged'
        avg_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Generating averaged plots...")
        print(f"  Output directory: {avg_output_dir}")
        
        # Generate averaged plots
        representative_sequence_onehot = one_hot_encode(representative_seq)
        plots = generate_plots(
            model, 
            representative_sequence_onehot,
            averaged_attributions,
            representative_seq,
            avg_output_dir, 
            method=args.attribution_method, 
            head_name=args.head_name
        )
        all_plots.extend(plots)
        
        # Save averaged attribution info
        info_path = avg_output_dir / 'averaged_info.txt'
        with open(info_path, 'w') as f:
            f.write(f"Averaged over {len(all_attributions)} sequences\n")
            f.write(f"Output task: {args.output_task}\n")
            f.write(f"Motif: {motif_sequence}\n")
            if motif_name:
                f.write(f"Motif name: {motif_name}\n")
            if motif_pos is not None:
                f.write(f"Motif position: {motif_pos}\n")
            f.write(f"Shuffled background: {args.shuffle_background}\n")
            f.write(f"Sequence indices: {[seq_data[0] for seq_data in sequences_to_process]}\n")
        print(f"  ✓ Averaged info saved: {info_path.name}")
        
        # Store averaged data for FIMO export if requested
        if args.export_for_fimo:
            sequences_for_fimo.append({
                'sequence_idx': 'averaged',
                'sequence_str': representative_seq,
                'attributions': averaged_attributions,
                'activity': 'averaged'
            })
    
    # Export sequences for FIMO if requested
    if args.export_for_fimo and sequences_for_fimo:
        print("\n" + "="*80)
        print("EXPORTING SEQUENCES FOR FIMO")
        print("="*80)
        fimo_output_dir = output_base_dir / 'fimo_input'
        full_seqs_path, seqlets_path, scores_path = export_sequences_for_fimo(
            sequences_for_fimo,
            fimo_output_dir,
            method=args.attribution_method,
            threshold_percentile=args.seqlet_threshold,
            min_length=args.seqlet_min_length,
            max_length=args.seqlet_max_length,
        )
        print(f"\n✓ FIMO export complete")
        print(f"  Full sequences: {full_seqs_path}")
        print(f"  High-attribution seqlets: {seqlets_path}")
        print(f"  Attribution scores: {scores_path}")
        print()
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Processed {len(sequences_to_process)} sequence(s)")
    print(f"Output task: {args.output_task}")
    print(f"Attribution method: {args.attribution_method}")
    if motif_sequence:
        if motif_name:
            print(f"Motif analysis: {motif_name} ({motif_sequence})")
        else:
            print(f"Motif analysis: {motif_sequence}")
        if args.shuffle_background:
            print(f"  Background: Shuffled")
        else:
            print(f"  Background: Original")
    print(f"Generated {len(all_plots)} plot(s)")
    print(f"All plots saved to: {output_base_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
