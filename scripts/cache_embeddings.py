"""
Pre-compute and cache encoder embeddings for MPRA sequences.

This script runs the AlphaGenome encoder on all sequences in the dataset
and saves the embeddings to a cache file. This speeds up training by
avoiding repeated forward passes through the encoder.

USAGE:
    python scripts/cache_embeddings.py --cell_type HepG2
    python scripts/cache_embeddings.py --cell_type K562 --cache_dir ./cache
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_ft import (
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)
from src import EncoderMPRAHead, LentiMPRADataset


def compute_sequence_hash(sequence: str) -> str:
    """Compute SHA256 hash of sequence for cache key."""
    return hashlib.sha256(sequence.encode()).hexdigest()


def extract_encoder_output(model, sequences, organism_indices):
    """Extract encoder output from model for given sequences.
    
    Args:
        model: CustomAlphaGenomeModel with use_encoder_output=True
        sequences: One-hot encoded sequences (batch, seq_len, 4)
        organism_indices: Organism indices (batch,)
        
    Returns:
        Encoder output array (batch, seq_len//128, D)
    """
    import haiku as hk
    from alphagenome_research.model import model as model_lib
    import jmp
    
    policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
    
    @hk.transform_with_state
    def encoder_forward(dna_sequence, organism_index):
        """Forward pass that extracts only encoder output."""
        with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
            with hk.mixed_precision.push_policy(model_lib.SequenceEncoder, policy):
                with hk.name_scope('alphagenome'):
                    trunk, intermediates = model_lib.SequenceEncoder()(dna_sequence)
                    encoder_output = trunk
                    return encoder_output
    
    # Apply the forward function
    with model._device_context:
        encoder_output, _ = encoder_forward.apply(
            model._params,
            model._state,
            None,  # rng
            sequences,
            organism_indices
        )
        
        return encoder_output


def cache_embeddings_for_dataset(
    model,
    dataset: LentiMPRADataset,
    cache_file: Path,
    batch_size: int = 32,
):
    """Pre-compute and cache encoder embeddings for all sequences in dataset.
    
    Args:
        model: CustomAlphaGenomeModel with use_encoder_output=True
        dataset: LentiMPRADataset (should have augmentations disabled)
        cache_file: Path to save cache file
        batch_size: Batch size for processing
    """
    print(f"Computing embeddings for {len(dataset)} sequences...")
    
    # Create cache dictionary: {sequence_hash: encoder_output}
    cache = {}
    sequence_to_hash = {}  # For debugging: {sequence: hash}
    
    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        # Collect batch data
        batch_sequences = []
        batch_hashes = []
        batch_labels = []
        batch_org_indices = []
        
        for idx in range(start_idx, end_idx):
            # Get raw sequence (before augmentations)
            sequence = dataset.data.iloc[idx]['seq'] + dataset.promoter_seq + dataset.rand_barcode
            if dataset.pad_n_bases > 0:
                padding_amount = dataset.pad_n_bases // 2
                sequence = 'N' * padding_amount + sequence + 'N' * padding_amount
            
            # Compute hash
            seq_hash = compute_sequence_hash(sequence)
            batch_hashes.append(seq_hash)
            sequence_to_hash[sequence] = seq_hash
            
            # Convert to one-hot
            sequence_onehot = model._one_hot_encoder.encode(sequence)
            sequence_onehot = jnp.array(sequence_onehot)
            batch_sequences.append(sequence_onehot)
            
            # Get label and organism index
            batch_labels.append(dataset.data.iloc[idx]['mean_value'])
            org_idx = jnp.array([0]) if dataset.organism == dna_model.Organism.HOMO_SAPIENS else jnp.array([1])
            batch_org_indices.append(org_idx)
        
        # Stack sequences (handle variable length)
        seqs = batch_sequences
        max_len = max(s.shape[0] for s in seqs)
        padded_seqs = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                padding = jnp.zeros((max_len - seq.shape[0], 4))
                seq = jnp.concatenate([seq, padding], axis=0)
            padded_seqs.append(seq)
        
        batch_seq = jnp.stack(padded_seqs, axis=0)  # (batch, seq_len, 4)
        batch_org = jnp.stack(batch_org_indices, axis=0).squeeze(-1)  # (batch,)
        
        # Extract encoder output
        encoder_outputs = extract_encoder_output(model, batch_seq, batch_org)
        
        # Convert to numpy and store in cache
        encoder_outputs_np = np.array(encoder_outputs)
        
        for i, seq_hash in enumerate(batch_hashes):
            # Store encoder output (remove padding if needed)
            # Note: encoder output shape is (batch, seq_len//128, D)
            # We need to handle variable sequence lengths
            cache[seq_hash] = encoder_outputs_np[i]
    
    # Save cache to file
    print(f"Saving cache to {cache_file}...")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle (more efficient for numpy arrays)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    
    # Also save metadata
    metadata = {
        'num_sequences': len(cache),
        'cell_type': dataset.cell_type,
        'split': dataset.split,
        'folds': dataset.chosen_fold,
        'promoter_seq': dataset.promoter_seq,
        'rand_barcode': dataset.rand_barcode,
        'pad_n_bases': dataset.pad_n_bases,
    }
    
    metadata_file = cache_file.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Cached {len(cache)} embeddings")
    print(f"  Cache file: {cache_file}")
    print(f"  Metadata file: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute and cache encoder embeddings for MPRA sequences'
    )
    parser.add_argument(
        '--cell_type',
        type=str,
        default='HepG2',
        choices=['HepG2', 'K562', 'WTC11'],
        help='Cell type for MPRA data'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to cache'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./.cache/embeddings',
        help='Directory to save cache files'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--path_to_data',
        type=str,
        default='./data/legnet_lentimpra',
        help='Path to MPRA data directory'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Caching Encoder Embeddings")
    print("=" * 80)
    print(f"Cell type: {args.cell_type}")
    print(f"Split: {args.split}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    print()
    
    # Create model (required for one-hot encoding and forward pass)
    print("Creating model...")
    register_custom_head(
        'mpra_head',
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='mpra_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata={'center_bp': 256, 'pooling_type': 'sum'}
        )
    )
    
    model = create_model_with_custom_heads(
        'all_folds',
        custom_heads=['mpra_head'],
        use_encoder_output=True
    )
    print("✓ Model created")
    
    # Create dataset (with augmentations disabled)
    print(f"\nLoading dataset (cell_type={args.cell_type}, split={args.split})...")
    dataset = LentiMPRADataset(
        model=model,
        path_to_data=args.path_to_data,
        cell_type=args.cell_type,
        split=args.split,
        random_shift=False,
        reverse_complement=False,
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Determine cache file path
    cache_dir = Path(args.cache_dir)
    cache_file = cache_dir / f"{args.cell_type}_{args.split}_embeddings.pkl"
    
    # Compute and cache embeddings
    print(f"\nComputing embeddings...")
    cache_embeddings_for_dataset(
        model,
        dataset,
        cache_file,
        batch_size=args.batch_size,
    )
    
    print("\n" + "=" * 80)
    print("Caching Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
