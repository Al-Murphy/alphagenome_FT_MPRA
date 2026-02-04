#!/usr/bin/env python3
"""
Compute attribution maps and sequence logos for MPRA sequences.

This script computes attributions using various methods (DeepSHAP, gradients, etc.)
and generates visualization plots for specified sequences from the test dataset.

USAGE:
    # Compute attributions for a specific sequence
    python scripts/compute_attributions.py \
        --checkpoint_dir ./results/models/checkpoints/HepG2/mpra-HepG2-optimal/stage2 \
        --cell_type HepG2 \
        --sequence_id 10350

    # Compute attributions for top 10 sequences by activity
    python scripts/compute_attributions.py \
        --checkpoint_dir ./results/models/checkpoints/HepG2/mpra-HepG2-optimal/stage2 \
        --cell_type HepG2 \
        --top_n 10 \
        --attribution_method deepshap

    # Use gradient-based attribution instead
    python scripts/compute_attributions.py \
        --checkpoint_dir ./results/models/checkpoints/HepG2/mpra-HepG2-optimal/stage2 \
        --cell_type HepG2 \
        --top_n 5 \
        --attribution_method gradient_x_input

    # Test HNF4A/G motif in dinucleotide-shuffled backgrounds
    python scripts/compute_attributions.py \
        --checkpoint_dir ./results/models/checkpoints/HepG2/mpra-HepG2-optimal/stage2 \
        --cell_type HepG2 \
        --top_n 10 \
        --motif CAAAG \
        --shuffle_background \
        --attribution_method deepshap

    # Test motif from JASPAR PFM file
    python scripts/compute_attributions.py \
        --checkpoint_dir ./results/models/checkpoints/HepG2/mpra-HepG2-optimal/stage2 \
        --cell_type HepG2 \
        --top_n 10 \
        --motif_file ./data/motifs/MA0114.1.pfm \
        --shuffle_background \
        --attribution_method deepshap

    # Test motif from JASPAR MEME file with sampling
    python scripts/compute_attributions.py \
        --checkpoint_dir ./results/models/checkpoints/HepG2/mpra-HepG2-optimal/stage2 \
        --cell_type HepG2 \
        --sequence_id 10350 \
        --motif_file ./data/motifs/MA0114.1.meme \
        --motif_consensus_method sample \
        --motif_position 100 \
        --shuffle_background

    # Export sequences for FIMO motif discovery
    python scripts/compute_attributions.py \
        --checkpoint_dir ./results/models/checkpoints/HepG2/mpra-HepG2-optimal/stage2 \
        --cell_type HepG2 \
        --top_n 20 \
        --attribution_method deepshap \
        --export_for_fimo
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
import traceback
import random

from alphagenome_ft import load_checkpoint, register_custom_head, HeadConfig, HeadType
from alphagenome.models import dna_output
from src import EncoderMPRAHead, LentiMPRADataset




def parse_meme_file(meme_path):
    """
    Parse a MEME format motif file.
    
    Returns:
        tuple: (motif_name, pwm_matrix) where pwm_matrix is (length, 4) array
               with columns [A, C, G, T]
    """
    with open(meme_path, 'r') as f:
        content = f.read()
    
    # Find MOTIF section
    motif_section = None
    lines = content.split('\n')
    in_motif = False
    motif_name = None
    pwm_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('MOTIF'):
            # Extract motif name (e.g., "MOTIF MA0114.1 HNF4A" -> "HNF4A")
            parts = line.split()
            if len(parts) >= 3:
                motif_name = ' '.join(parts[2:])  # Everything after "MOTIF MA0114.1"
            elif len(parts) == 2:
                motif_name = parts[1]
            else:
                motif_name = 'Unknown'
            in_motif = True
        elif in_motif and line.startswith('letter-probability matrix'):
            # Next lines are the PWM
            continue
        elif in_motif and line and not line.startswith('URL'):
            # Parse probability line (4 values: A, C, G, T)
            try:
                values = [float(x) for x in line.split()]
                if len(values) == 4:
                    pwm_lines.append(values)
            except ValueError:
                # Not a probability line, might be end of motif
                if pwm_lines:
                    break
        elif in_motif and line.startswith('URL'):
            # End of motif
            break
    
    if not pwm_lines:
        raise ValueError(f"No probability matrix found in MEME file: {meme_path}")
    
    pwm_matrix = np.array(pwm_lines)
    
    # Validate: should have 4 columns (A, C, G, T)
    if pwm_matrix.shape[1] != 4:
        raise ValueError(f"PWM should have 4 columns (A, C, G, T), found {pwm_matrix.shape[1]}")
    
    return motif_name, pwm_matrix


def pfm_to_consensus(pfm_matrix, method='max', random_state=None):
    """
    Convert PFM (Position Frequency Matrix) to consensus sequence.
    
    Args:
        pfm_matrix: (4, length) array with rows [A, T, C, G] (JASPAR format)
        method: 'max' for most frequent base, 'sample' for sampling from frequencies
        random_state: Random seed for reproducibility
    
    Returns:
        Consensus sequence string
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    base_order = ['A', 'T', 'C', 'G']  # JASPAR order
    motif_length = pfm_matrix.shape[1]
    consensus = []
    
    for pos in range(motif_length):
        freqs = pfm_matrix[:, pos]  # [A, T, C, G] frequencies
        
        if method == 'max':
            # Use most frequent base
            base_idx = np.argmax(freqs)
            consensus.append(base_order[base_idx])
        elif method == 'sample':
            # Sample from frequencies (normalize to probabilities)
            probs = freqs / (freqs.sum() + 1e-10)
            base_idx = np.random.choice(4, p=probs)
            consensus.append(base_order[base_idx])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return ''.join(consensus)


def pwm_to_consensus(pwm_matrix, method='max', random_state=None):
    """
    Convert PWM (Position Weight Matrix) to consensus sequence.
    
    Args:
        pwm_matrix: (length, 4) array with columns [A, C, G, T] (probabilities)
        method: 'max' for highest probability base, 'sample' for sampling from probabilities
        random_state: Random seed for reproducibility
    
    Returns:
        Consensus sequence string
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    base_order = ['A', 'C', 'G', 'T']  # MEME format order
    consensus = []
    
    for pos in range(pwm_matrix.shape[0]):
        probs = pwm_matrix[pos, :]  # [A, C, G, T] probabilities
        
        if method == 'max':
            # Use highest probability base
            base_idx = np.argmax(probs)
            consensus.append(base_order[base_idx])
        elif method == 'sample':
            # Sample from probabilities
            base_idx = np.random.choice(4, p=probs)
            consensus.append(base_order[base_idx])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return ''.join(consensus)


def decode_one_hot(one_hot_seq):
    """Convert one-hot encoded sequence back to DNA string."""
    base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    seq_str = ''
    # one_hot_seq shape: (batch, length, 4) or (length, 4)
    if one_hot_seq.ndim == 3:
        one_hot_seq = one_hot_seq[0]  # Take first batch element
    for pos in range(one_hot_seq.shape[0]):
        base_idx = jnp.argmax(one_hot_seq[pos])
        seq_str += base_map[int(base_idx)]
    return seq_str


def one_hot_encode(sequence: str) -> jnp.ndarray:
    """Convert DNA sequence string to one-hot encoding."""
    base_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    seq_len = len(sequence)
    one_hot = jnp.zeros((1, seq_len, 4), dtype=jnp.float32)
    
    for i, base in enumerate(sequence.upper()):
        if base in base_map:
            one_hot = one_hot.at[0, i, base_map[base]].set(1.0)
    
    return one_hot


def parse_jaspar_pfm(pfm_file):
    """
    Parse a JASPAR Position Frequency Matrix (PFM) file.
    
    JASPAR PFM format:
    >MA0114.1 HNF4A
     28.00   2.00  12.00   5.00   3.00  59.00  53.00  56.00   4.00   6.00   3.00   4.00  42.00
      7.00   2.00   4.00  23.00  51.00   1.00   2.00   1.00   4.00   2.00  22.00  49.00   7.00
     27.00  56.00  35.00  20.00   4.00   3.00  10.00   8.00  58.00  33.00  11.00   5.00  10.00
      5.00   7.00  16.00  19.00   9.00   4.00   2.00   2.00   1.00  26.00  31.00   9.00   8.00
    
    Rows are: A, T, C, G (in that order)
    
    Args:
        pfm_file: Path to PFM file or file-like object
    
    Returns:
        tuple: (motif_id, motif_name, pfm_matrix, consensus_sequence)
        - motif_id: JASPAR ID (e.g., "MA0114.1")
        - motif_name: Motif name (e.g., "HNF4A")
        - pfm_matrix: numpy array of shape (4, motif_length) with frequencies
        - consensus_sequence: Consensus sequence string (most frequent base at each position)
    """
    if isinstance(pfm_file, (str, Path)):
        with open(pfm_file, 'r') as f:
            lines = f.readlines()
    else:
        lines = pfm_file.readlines()
    
    # Parse header
    header_line = lines[0].strip()
    if not header_line.startswith('>'):
        raise ValueError(f"PFM file must start with '>' header line. Got: {header_line}")
    
    # Extract motif ID and name
    header_parts = header_line[1:].strip().split(None, 1)  # Split on whitespace, max 1 split
    motif_id = header_parts[0] if len(header_parts) > 0 else "UNKNOWN"
    motif_name = header_parts[1] if len(header_parts) > 1 else motif_id
    
    # Parse frequency matrix (4 rows: A, T, C, G)
    matrix_rows = []
    base_order = ['A', 'T', 'C', 'G']
    
    for i, line in enumerate(lines[1:5], 1):  # Next 4 lines
        if i > 4:
            break
        row = [float(x) for x in line.strip().split()]
        matrix_rows.append(row)
    
    if len(matrix_rows) != 4:
        raise ValueError(f"PFM file must have 4 rows (A, T, C, G). Found {len(matrix_rows)} rows")
    
    # Convert to numpy array: shape (4, motif_length)
    pfm_matrix = np.array(matrix_rows, dtype=np.float32)
    motif_length = pfm_matrix.shape[1]
    
    # Generate consensus sequence (most frequent base at each position)
    consensus_sequence = ''
    for pos in range(motif_length):
        # Get frequencies for this position: [A, T, C, G]
        pos_freqs = pfm_matrix[:, pos]
        # Find base with highest frequency
        max_base_idx = np.argmax(pos_freqs)
        consensus_sequence += base_order[max_base_idx]
    
    return motif_id, motif_name, pfm_matrix, consensus_sequence


def sample_motif_from_pfm(pfm_matrix, random_state=None):
    """
    Sample a sequence from a Position Frequency Matrix.
    
    Args:
        pfm_matrix: numpy array of shape (4, motif_length) with frequencies
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled sequence string
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    base_order = ['A', 'T', 'C', 'G']
    motif_length = pfm_matrix.shape[1]
    sampled_sequence = ''
    
    for pos in range(motif_length):
        # Get frequencies for this position: [A, T, C, G]
        pos_freqs = pfm_matrix[:, pos]
        # Normalize to probabilities
        pos_probs = pos_freqs / (pos_freqs.sum() + 1e-10)
        # Sample base according to probabilities
        sampled_base_idx = np.random.choice(4, p=pos_probs)
        sampled_sequence += base_order[sampled_base_idx]
    
    return sampled_sequence


def dinucleotide_shuffle(sequence_str, random_state=None):
    """
    Perform dinucleotide-preserving shuffle of a DNA sequence.
    
    This preserves dinucleotide frequencies while randomizing the sequence.
    Uses the algorithm from Altschul & Erickson (1985).
    
    Args:
        sequence_str: DNA sequence as string
        random_state: Random seed for reproducibility
    
    Returns:
        Shuffled sequence string
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Convert to list for easier manipulation
    seq_list = list(sequence_str.upper())
    
    # Perform multiple random swaps of adjacent dinucleotides
    # This preserves dinucleotide frequencies approximately
    n_swaps = len(seq_list) // 2
    for _ in range(n_swaps):
        # Pick a random position (avoid last position)
        if len(seq_list) > 1:
            i = random.randint(0, len(seq_list) - 2)
            # Swap adjacent bases
            seq_list[i], seq_list[i+1] = seq_list[i+1], seq_list[i]
    
    return ''.join(seq_list)


def insert_motif(sequence_str, motif, position=None, center=False, random_state=None):
    """
    Insert a motif into a sequence at a specified position.
    
    Args:
        sequence_str: DNA sequence as string
        motif: Motif sequence as string (e.g., "AGGTCA" for HNF4A)
        position: Position to insert motif (None = random or center, int = specific position)
        center: If True, insert at center position (overrides position if not None)
        random_state: Random seed for reproducibility
    
    Returns:
        Sequence string with motif inserted, and the insertion position
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    seq_list = list(sequence_str)
    motif_len = len(motif)
    seq_len = len(seq_list)
    
    # Determine insertion position
    if center:
        # Insert at center
        position = (seq_len - motif_len) // 2
    elif position is None:
        # Random position that allows motif to fit
        max_pos = seq_len - motif_len
        if max_pos < 0:
            raise ValueError(f"Sequence too short ({seq_len}) to insert motif of length {motif_len}")
        position = random.randint(0, max_pos)
    else:
        # Validate position
        if position < 0 or position + motif_len > seq_len:
            raise ValueError(f"Position {position} out of range for motif of length {motif_len}")
    
    # Insert motif by replacing bases at position
    for i, base in enumerate(motif.upper()):
        seq_list[position + i] = base
    
    return ''.join(seq_list), position


def find_top_sequences(dataset, n=10):
    """Find top N sequences by activity."""
    print(f"Finding top {n} sequences by activity...")
    activities = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        activities.append((idx, float(sample['y'])))
    
    # Sort by activity (descending)
    activities.sort(key=lambda x: x[1], reverse=True)
    top_n = activities[:n]
    
    print(f"✓ Found top {n} sequences:")
    for rank, (idx, activity) in enumerate(top_n, 1):
        print(f"  {rank}. idx={idx}, activity={activity:.6f}")
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
    sampled = [(int(idx), float(dataset[idx]['y'])) for idx in indices]
    
    print(f"✓ Sampled {len(sampled)} sequences:")
    for rank, (idx, activity) in enumerate(sampled, 1):
        print(f"  {rank}. idx={idx}, activity={activity:.6f}")
    print()
    
    return sampled


def compute_attributions(model, sequence, organism_index, method='deepshap', head_name='mpra_head', **kwargs):
    """Compute attributions using the specified method."""
    if method == 'deepshap':
        attributions = model.compute_deepshap_attributions(
            sequence=sequence,
            organism_index=organism_index,
            head_name=head_name,
            n_references=kwargs.get('n_references', 20),
            reference_type=kwargs.get('reference_type', 'shuffle'),
            random_state=kwargs.get('random_state', 42),
            output_index=kwargs.get('output_index', None),
        )
    elif method == 'gradient':
        attributions = model.compute_input_gradients(
            sequence=sequence,
            organism_index=organism_index,
            head_name=head_name,
            gradients_x_input=False,
            output_index=kwargs.get('output_index', None),
        )
    elif method == 'gradient_x_input':
        attributions = model.compute_input_gradients(
            sequence=sequence,
            organism_index=organism_index,
            head_name=head_name,
            gradients_x_input=True,
            output_index=kwargs.get('output_index', None),
        )
    else:
        raise ValueError(f"Unknown attribution method: {method}. Must be one of: deepshap, gradient, gradient_x_input")
    
    return attributions


def extract_seqlets(attributions, sequence_str, threshold_percentile=90, min_length=5, max_length=50):
    """
    Extract high-attribution regions (seqlets) from a sequence.
    
    Args:
        attributions: Attribution scores, shape (batch, seq_len, 4) or (seq_len, 4)
        sequence_str: DNA sequence string
        threshold_percentile: Percentile threshold for high-attribution positions (default: 90)
        min_length: Minimum seqlet length (default: 5)
        max_length: Maximum seqlet length (default: 50)
    
    Returns:
        List of (start, end, seqlet_sequence, attribution_sum) tuples
    """
    # Convert to numpy and handle batch dimension
    attr_np = np.asarray(attributions)
    if attr_np.ndim == 3:
        attr_np = attr_np[0]  # Take first batch element
    
    # Sum attributions across bases for each position
    position_scores = np.sum(attr_np, axis=1)  # (seq_len,)
    
    # Calculate threshold
    threshold = np.percentile(position_scores, threshold_percentile)
    
    # Find positions above threshold
    high_attr_positions = position_scores >= threshold
    
    # Extract contiguous regions (seqlets)
    seqlets = []
    in_seqlet = False
    seqlet_start = 0
    
    for i, is_high in enumerate(high_attr_positions):
        if is_high and not in_seqlet:
            # Start of new seqlet
            seqlet_start = i
            in_seqlet = True
        elif not is_high and in_seqlet:
            # End of seqlet
            seqlet_end = i
            seqlet_length = seqlet_end - seqlet_start
            if min_length <= seqlet_length <= max_length:
                seqlet_seq = sequence_str[seqlet_start:seqlet_end]
                attr_sum = np.sum(position_scores[seqlet_start:seqlet_end])
                seqlets.append((seqlet_start, seqlet_end, seqlet_seq, float(attr_sum)))
            in_seqlet = False
    
    # Handle seqlet that extends to end of sequence
    if in_seqlet:
        seqlet_end = len(sequence_str)
        seqlet_length = seqlet_end - seqlet_start
        if min_length <= seqlet_length <= max_length:
            seqlet_seq = sequence_str[seqlet_start:seqlet_end]
            attr_sum = np.sum(position_scores[seqlet_start:seqlet_end])
            seqlets.append((seqlet_start, seqlet_end, seqlet_seq, float(attr_sum)))
    
    return seqlets


def export_sequences_for_fimo(sequences_data, output_dir, method='deepshap', 
                               threshold_percentile=90.0, min_length=5, max_length=50):
    """
    Export sequences and high-attribution regions (seqlets) as FASTA files for FIMO analysis.
    
    Args:
        sequences_data: List of dicts with keys: 'sequence_idx', 'sequence_str', 'attributions', 'activity'
        output_dir: Output directory for FASTA files
        method: Attribution method name
        threshold_percentile: Percentile threshold for extracting seqlets (default: 90.0)
        min_length: Minimum seqlet length (default: 5)
        max_length: Maximum seqlet length (default: 50)
    
    Returns:
        Tuple of (full_sequences_fasta_path, seqlets_fasta_path, scores_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export 1: Full sequences with high attribution
    full_seqs_path = output_dir / f'sequences_for_fimo_{method}.fa'
    with open(full_seqs_path, 'w') as f:
        for data in sequences_data:
            seq_idx = data['sequence_idx']
            seq_str = data['sequence_str']
            activity = data.get('activity', 'N/A')
            f.write(f">seq{seq_idx}_activity_{activity}\n")
            f.write(f"{seq_str}\n")
    
    # Export 2: High-attribution regions (seqlets)
    seqlets_path = output_dir / f'seqlets_for_fimo_{method}.fa'
    all_seqlets = []
    with open(seqlets_path, 'w') as f:
        for data in sequences_data:
            seq_idx = data['sequence_idx']
            seq_str = data['sequence_str']
            attributions = data['attributions']
            
            # Extract seqlets
            seqlets = extract_seqlets(
                attributions, seq_str, 
                threshold_percentile=threshold_percentile,
                min_length=min_length,
                max_length=max_length
            )
            for seqlet_idx, (start, end, seqlet_seq, attr_sum) in enumerate(seqlets):
                all_seqlets.append({
                    'seq_idx': seq_idx,
                    'seqlet_idx': seqlet_idx,
                    'start': start,
                    'end': end,
                    'sequence': seqlet_seq,
                    'attribution_sum': attr_sum
                })
                f.write(f">seq{seq_idx}_seqlet{seqlet_idx}_pos{start}-{end}_attr{attr_sum:.2f}\n")
                f.write(f"{seqlet_seq}\n")
    
    # Export 3: Attribution scores as BED-like format for reference
    scores_path = output_dir / f'attribution_scores_{method}.tsv'
    with open(scores_path, 'w') as f:
        f.write("sequence_idx\tposition\tbase\tattribution_score\n")
        for data in sequences_data:
            seq_idx = data['sequence_idx']
            seq_str = data['sequence_str']
            attributions = data['attributions']
            
            attr_np = np.asarray(attributions)
            if attr_np.ndim == 3:
                attr_np = attr_np[0]
            
            for pos in range(len(seq_str)):
                base = seq_str[pos]
                base_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}.get(base, -1)
                if base_idx >= 0:
                    attr_score = float(attr_np[pos, base_idx])
                    f.write(f"{seq_idx}\t{pos}\t{base}\t{attr_score:.6f}\n")
    
    print(f"  ✓ Exported {len(sequences_data)} sequences to: {full_seqs_path.name}")
    print(f"  ✓ Exported {len(all_seqlets)} seqlets to: {seqlets_path.name}")
    print(f"  ✓ Exported attribution scores to: {scores_path.name}")
    
    return full_seqs_path, seqlets_path, scores_path


def generate_plots(model, sequence, attributions, sequence_str, output_dir, method='deepshap', head_name='mpra_head'):
    """Generate all plots for a sequence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_generated = []
    
    # 1. Attribution map
    try:
        save_path = output_dir / f'attribution_map_{method}.png'
        model.plot_attribution_map(
            sequence=sequence,
            gradients=attributions,
            sequence_str=sequence_str,
            save_path=str(save_path)
        )
        plots_generated.append(save_path)
        print(f"  ✓ Attribution map saved: {save_path.name}")
    except Exception as e:
        print(f"  ✗ Error plotting attribution map: {e}")
        traceback.print_exc()
    
    # 2. Sequence logo
    # For DeepSHAP: masked to sequence with raw attribution values (standard)
    # For gradient×input: raw attribution values (unnormalized, standard practice)
    # For gradient: information content (shows relative importance)
    try:
        if method == 'deepshap':
            save_path = output_dir / 'sequence_logo_deepshap.png'
            model.plot_sequence_logo(
                sequence=sequence,
                gradients=attributions,
                save_path=str(save_path),
                logo_type='information',  # Will be overridden by mask_to_sequence=True
                mask_to_sequence=True  # Standard for DeepSHAP: raw values, masked to sequence
            )
        elif method == 'gradient_x_input':
            # For gradient×input, use raw values (unnormalized) - standard practice
            # This shows the actual gradient×input attribution values
            save_path = output_dir / 'sequence_logo_gradient_x_input.png'
            model.plot_sequence_logo(
                sequence=sequence,
                gradients=attributions,
                save_path=str(save_path),
                logo_type='weight',  # Raw values, no normalization
                mask_to_sequence=False  # Show all bases
            )
        else:  # gradient
            save_path = output_dir / f'sequence_logo_{method}_information.png'
            model.plot_sequence_logo(
                sequence=sequence,
                gradients=attributions,
                save_path=str(save_path),
                logo_type='information',  # Information content (bits)
                mask_to_sequence=False
            )
        plots_generated.append(save_path)
        print(f"  ✓ Sequence logo saved: {save_path.name}")
    except Exception as e:
        print(f"  ✗ Error plotting sequence logo: {e}")
        traceback.print_exc()
    
    return plots_generated


def process_sequence(model, dataset, sequence_idx, output_base_dir, method='deepshap', head_name='mpra_head', 
                     motif=None, motif_position=None, motif_center=False, shuffle_background=False, 
                     return_attributions=False, promoter_seq=None, rand_barcode=None, **kwargs):
    """Process a single sequence: compute attributions and generate plots.
    
    Args:
        model: Trained model
        dataset: Test dataset
        sequence_idx: Index of sequence in dataset
        output_base_dir: Base output directory
        method: Attribution method
        head_name: Name of custom head
        motif: Optional motif sequence to insert (e.g., "AGGTCA")
        motif_position: Position to insert motif (None = random or center, int = specific)
        motif_center: If True, insert motif at center position
        shuffle_background: If True, shuffle sequence before inserting motif
        return_attributions: If True, return attributions instead of generating plots
        **kwargs: Additional arguments for attribution computation
    
    Returns:
        If return_attributions=True: tuple (attributions, sequence_str, motif_position)
        Otherwise: list of generated plot paths
    """
    print(f"\n{'='*80}")
    print(f"Processing sequence {sequence_idx}")
    if motif:
        print(f"  Motif: {motif}")
        if shuffle_background:
            print(f"  Background: Dinucleotide-shuffled")
    print(f"{'='*80}")
    
    # Get sequence from dataset
    sample = dataset[sequence_idx]
    sequence = sample['seq']  # Shape: (length, 4) from dataset
    organism_index = sample['organism_index']  # Shape: () or (1,)
    activity = float(sample['y'])
    
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
    
    # Handle motif insertion
    motif_insertion_pos = None
    if motif:
        # Extract test sequence, promoter, and barcode if available
        # Sequence structure: test_seq + promoter_seq + rand_barcode (possibly with padding)
        test_seq_str = sequence_str
        promoter_str = ""
        barcode_str = ""
        suffix_str = ""
        
        if promoter_seq:
            # Find promoter position in sequence
            promoter_start = sequence_str.find(promoter_seq)
            if promoter_start >= 0:
                # Split sequence: test_seq | promoter | barcode/suffix
                test_seq_str = sequence_str[:promoter_start]
                remaining = sequence_str[promoter_start:]
                
                # Extract promoter
                if len(remaining) >= len(promoter_seq):
                    promoter_str = remaining[:len(promoter_seq)]
                    suffix_str = remaining[len(promoter_seq):]
                    
                    # Try to extract barcode if it exists
                    if rand_barcode and rand_barcode in suffix_str:
                        barcode_start = suffix_str.find(rand_barcode)
                        barcode_str = suffix_str[barcode_start:barcode_start + len(rand_barcode)]
                        # Keep any padding between promoter and barcode
                        suffix_str = suffix_str[:barcode_start] + suffix_str[barcode_start + len(rand_barcode):]
                else:
                    # Promoter not fully present, keep as is
                    promoter_str = remaining
            else:
                # Promoter not found, might be shuffled or different format
                # In this case, treat entire sequence as test sequence
                print(f"  Warning: Promoter sequence not found in sequence, treating entire sequence as test sequence")
        else:
            # No promoter info available, treat entire sequence as test sequence
            pass
        
        # Shuffle only the test sequence part (not promoter/barcode)
        if shuffle_background:
            test_seq_str = dinucleotide_shuffle(test_seq_str, random_state=kwargs.get('random_state'))
            print(f"  ✓ Test sequence shuffled (dinucleotide-preserving, promoter/barcode preserved)")
        
        # Insert motif into test sequence
        try:
            test_seq_str, motif_insertion_pos = insert_motif(
                test_seq_str, motif, position=motif_position, center=motif_center, random_state=kwargs.get('random_state')
            )
            print(f"  ✓ Motif inserted at position {motif_insertion_pos} in test sequence")
            
            # Reconstruct full sequence: test_seq + promoter + suffix (which includes barcode if present)
            sequence_str = test_seq_str + promoter_str + suffix_str
            
            # Re-encode sequence with motif
            sequence = one_hot_encode(sequence_str)
        except Exception as e:
            print(f"  ✗ Error inserting motif: {e}")
            if return_attributions:
                return None, None, None
            return []
    
    print(f"  Sequence length: {len(sequence_str)}bp")
    print(f"  Sequence (first 50bp): {sequence_str[:50]}...")
    if motif and motif_insertion_pos is not None:
        # Highlight motif in sequence preview
        motif_end = min(motif_insertion_pos + len(motif), len(sequence_str))
        print(f"  Motif region ({motif_insertion_pos}-{motif_end}): {sequence_str[motif_insertion_pos:motif_end]}")
    print(f"  Original activity: {activity:.6f}")
    
    # Create output directory (only if output_base_dir is provided and not returning attributions)
    if return_attributions:
        # Don't create output directory when just collecting attributions for averaging
        output_dir = None
    elif output_base_dir is not None:
        if motif:
            output_dir = Path(output_base_dir) / f'seq{sequence_idx}_motif_{motif}'
            if shuffle_background:
                output_dir = Path(output_base_dir) / f'seq{sequence_idx}_motif_{motif}_shuffled'
        else:
            output_dir = Path(output_base_dir) / f'seq{sequence_idx}'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {output_dir}")
    else:
        output_dir = None
    
    # Compute attributions
    print(f"\n  Computing {method} attributions...")
    try:
        attributions = compute_attributions(
            model, sequence, organism_index, method=method, head_name=head_name, **kwargs
        )
        print(f"  ✓ Attributions computed: shape {attributions.shape}")
        print(f"    Stats: min={jnp.min(attributions):.6f}, max={jnp.max(attributions):.6f}, mean={jnp.mean(attributions):.6f}")
        
        # If motif was inserted, highlight motif region in attributions
        if motif and motif_insertion_pos is not None:
            motif_attributions = attributions[0, motif_insertion_pos:motif_insertion_pos+len(motif), :]
            motif_attrib_sum = jnp.sum(motif_attributions)
            print(f"    Motif region attribution sum: {motif_attrib_sum:.6f}")
    except Exception as e:
        print(f"  ✗ Error computing attributions: {e}")
        traceback.print_exc()
        if return_attributions:
            return None, None, None
        return []
    
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
                f.write(f"Original sequence: {original_sequence_str}\n")
                f.write(f"Modified sequence: {sequence_str}\n")
                f.write(f"Shuffled background: {shuffle_background}\n")
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
        description='Compute attribution maps and sequence logos for MPRA sequences',
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
        '--cell_type',
        type=str,
        required=True,
        choices=['HepG2', 'K562', 'WTC11'],
        help='Cell type (HepG2, K562, or WTC11)'
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
        '--attribution_method',
        type=str,
        default='deepshap',
        choices=['deepshap', 'gradient', 'gradient_x_input'],
        help='Attribution method to use (default: deepshap)'
    )
    parser.add_argument(
        '--head_name',
        type=str,
        default='mpra_head',
        help='Name of the custom head to use (default: mpra_head)'
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
        '--output_index',
        type=int,
        default=None,
        help='Index of output track for multi-track heads (default: None - averages all tracks)'
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
    
    args = parser.parse_args()
    
    # Parse motif file if provided
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
                    motif_sequence = consensus_seq  # Use pre-computed consensus
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
                # Try to auto-detect format (try PFM first, then MEME)
                try:
                    motif_id, motif_name, pfm_matrix, consensus_seq = parse_jaspar_pfm(motif_file_path)
                    print(f"  Auto-detected format: PFM (JASPAR)")
                    print(f"  Motif ID: {motif_id}")
                    print(f"  Motif name: {motif_name}")
                    print(f"  Motif length: {pfm_matrix.shape[1]}bp")
                    if args.motif_consensus_method == 'max':
                        motif_sequence = consensus_seq  # Use pre-computed consensus
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
    
    # Register custom head (required before loading checkpoint)
    print("\n" + "="*80)
    print("Registering custom head...")
    print("="*80)
    register_custom_head(
        'mpra_head',
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='mpra_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata={}
        )
    )
    print("✓ Head registered\n")
    
    # Load trained model
    print("="*80)
    print("Loading trained model...")
    print("="*80)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    model = load_checkpoint(
        str(checkpoint_dir),
        base_model_version='all_folds',
        base_checkpoint_path=args.base_checkpoint_path,
        device=device
    )
    print("✓ Model loaded")
    print(f"  Custom heads: {model._custom_heads}")
    print()
    
    # Load test dataset
    print("="*80)
    print("Loading test dataset...")
    print("="*80)
    test_dataset = LentiMPRADataset(
        model=model,
        cell_type=args.cell_type,
        split='test',
        random_shift=False,
        reverse_complement=False,
        pad_n_bases=0
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples\n")
    
    # Get promoter and barcode sequences from dataset for motif handling
    promoter_seq = test_dataset.promoter_seq if hasattr(test_dataset, 'promoter_seq') else None
    rand_barcode = test_dataset.rand_barcode if hasattr(test_dataset, 'rand_barcode') else None
    
    # Determine which sequences to process
    if args.sequence_id is not None:
        if args.sequence_id < 0 or args.sequence_id >= len(test_dataset):
            print(f"Error: sequence_id {args.sequence_id} is out of range (0-{len(test_dataset)-1})")
            sys.exit(1)
        sequences_to_process = [(args.sequence_id, None)]  # (idx, activity)
    else:
        # If motif is provided, use random sampling instead of top by activity
        if motif_sequence:
            sequences_to_process = sample_random_sequences(
                test_dataset, n=args.top_n, random_state=args.random_state
            )
        else:
            # Find top N sequences by activity
            top_sequences = find_top_sequences(test_dataset, n=args.top_n)
            sequences_to_process = top_sequences
    
    # Create base output directory
    output_base_dir = Path(args.output_dir) / f'attribution_{args.cell_type}_{args.attribution_method}'
    if motif_sequence:
        # Use motif name if available, otherwise use sequence
        motif_label = motif_name if motif_name else motif_sequence
        # Sanitize for filesystem (remove spaces, special chars)
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
        
        # Determine insertion position strategy
        if args.top_n is not None:
            # When using top_n, default to center for alignment
            if args.motif_position is not None:
                print(f"Insertion position: {args.motif_position} (user-specified)")
            else:
                print(f"Insertion position: Center (for alignment across sequences)")
                if motif_sequence:
                    print(f"  → Using random sampling (not top by activity)")
                    print(f"  → Attributions will be averaged over {args.top_n} randomly sampled sequences")
                else:
                    print(f"  → Attributions will be averaged over top {args.top_n} sequences")
        elif args.motif_position is not None:
            print(f"Insertion position: {args.motif_position}")
        else:
            print(f"Insertion position: Random")
        
        if args.shuffle_background:
            print(f"Background: Dinucleotide-shuffled (preserves dinucleotide frequencies)")
        else:
            print(f"Background: Original test sequences")
        print()
    
    # Determine if we should average attributions (when using top_n with motif)
    average_attributions = (args.top_n is not None and motif_sequence is not None)
    # Use center position when averaging (to align motifs across sequences)
    use_center_position = average_attributions and args.motif_position is None
    
    # Process each sequence
    all_plots = []
    all_attributions = []
    all_sequences = []
    all_motif_positions = []
    sequences_for_fimo = []  # Store data for FIMO export
    
    for seq_idx, activity in sequences_to_process:
        if average_attributions:
            # Collect attributions for averaging (don't create individual output directories or plots)
            result = process_sequence(
                model=model,
                dataset=test_dataset,
                sequence_idx=seq_idx,
                output_base_dir=None,  # Don't create individual directories when averaging
                method=args.attribution_method,
                head_name=args.head_name,
                motif=motif_sequence,
                motif_position=args.motif_position,
                motif_center=use_center_position,
                shuffle_background=args.shuffle_background,
                promoter_seq=promoter_seq,
                rand_barcode=rand_barcode,
                n_references=args.n_references,
                reference_type=args.reference_type,
                random_state=args.random_state,
                output_index=args.output_index,
                return_attributions=True,  # This prevents directory creation and plot generation
            )
            if result[0] is not None:
                attributions, seq_str, motif_pos = result
                all_attributions.append(attributions)
                all_sequences.append(seq_str)
                all_motif_positions.append(motif_pos)
                
                # Store for FIMO export if requested
                if args.export_for_fimo:
                    sequences_for_fimo.append({
                        'sequence_idx': seq_idx,
                        'sequence_str': seq_str,
                        'attributions': attributions,
                        'activity': activity
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
                motif=motif_sequence,
                motif_position=args.motif_position,
                motif_center=use_center_position,
                shuffle_background=args.shuffle_background,
                promoter_seq=promoter_seq,
                rand_barcode=rand_barcode,
                n_references=args.n_references,
                reference_type=args.reference_type,
                random_state=args.random_state,
                output_index=args.output_index,
            )
            all_plots.extend(plots)
            
            # Store for FIMO export if requested (when not averaging)
            if args.export_for_fimo:
                # Get sequence and attributions from the processed sequence
                # We need to recompute to get the full sequence (with promoter/barcode)
                sample = test_dataset[seq_idx]
                seq = sample['seq']
                seq_jax = jnp.array(seq)
                if seq_jax.ndim == 2:
                    seq_jax = seq_jax[None, :, :]
                seq_str = decode_one_hot(seq_jax)
                
                # Compute attributions for this sequence
                org_idx = jnp.array(sample['organism_index'])
                if org_idx.ndim == 0:
                    org_idx = org_idx[None]
                
                attr = compute_attributions(
                    model, seq_jax, org_idx, method=args.attribution_method, 
                    head_name=args.head_name,
                    n_references=args.n_references,
                    reference_type=args.reference_type,
                    random_state=args.random_state,
                    output_index=args.output_index,
                )
                
                sequences_for_fimo.append({
                    'sequence_idx': seq_idx,
                    'sequence_str': seq_str,
                    'attributions': attr,
                    'activity': activity
                })
    
    # Average attributions if requested
    if average_attributions and all_attributions:
        print("\n" + "="*80)
        print("AVERAGING ATTRIBUTIONS")
        print("="*80)
        print(f"Averaging attributions over {len(all_attributions)} sequences")
        
        # Stack and average
        stacked_attributions = jnp.stack(all_attributions)  # (n_seqs, batch, seq_len, 4)
        averaged_attributions = jnp.mean(stacked_attributions, axis=0)  # (batch, seq_len, 4)
        
        print(f"✓ Averaged attributions: shape {averaged_attributions.shape}")
        print(f"  Stats: min={jnp.min(averaged_attributions):.6f}, max={jnp.max(averaged_attributions):.6f}, mean={jnp.mean(averaged_attributions):.6f}")
        
        # Get representative sequence (first one, or consensus if all same length)
        if all_sequences:
            # Use first sequence as representative (all should have same length and motif position)
            representative_seq = all_sequences[0]
            motif_pos = all_motif_positions[0] if all_motif_positions else None
            
            # Highlight motif region in averaged attributions
            if motif_pos is not None:
                motif_attributions = averaged_attributions[0, motif_pos:motif_pos+len(motif_sequence), :]
                motif_attrib_sum = jnp.sum(motif_attributions)
                print(f"  Motif region (pos {motif_pos}-{motif_pos+len(motif_sequence)}) attribution sum: {motif_attrib_sum:.6f}")
        
        # Create output directory for averaged results
        avg_output_dir = output_base_dir / 'averaged'
        avg_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Generating averaged plots...")
        print(f"  Output directory: {avg_output_dir}")
        
        # Generate averaged plots
        # Need to create a one-hot encoded sequence for the plot function
        representative_sequence_onehot = one_hot_encode(representative_seq)
        plots = generate_plots(
            model, 
            representative_sequence_onehot,  # One-hot encoded sequence
            averaged_attributions,  # Averaged attributions
            representative_seq,  # Sequence string for display
            avg_output_dir, 
            method=args.attribution_method, 
            head_name=args.head_name
        )
        all_plots.extend(plots)
        
        # Save averaged attribution info
        info_path = avg_output_dir / 'averaged_info.txt'
        with open(info_path, 'w') as f:
            f.write(f"Averaged over {len(all_attributions)} sequences\n")
            f.write(f"Motif: {motif_sequence}\n")
            if motif_name:
                f.write(f"Motif name: {motif_name}\n")
            if motif_pos is not None:
                f.write(f"Motif position: {motif_pos}\n")
            f.write(f"Shuffled background: {args.shuffle_background}\n")
            f.write(f"Sequence indices: {[idx for idx, _ in sequences_to_process]}\n")
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
        print(f"\n  To run FIMO:")
        print(f"    fimo --oc {fimo_output_dir}/fimo_output --thresh 1e-4 <motif_database.meme> {seqlets_path}")
        print(f"  Or on full sequences:")
        print(f"    fimo --oc {fimo_output_dir}/fimo_output --thresh 1e-4 <motif_database.meme> {full_seqs_path}")
        print()
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Processed {len(sequences_to_process)} sequence(s)")
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
