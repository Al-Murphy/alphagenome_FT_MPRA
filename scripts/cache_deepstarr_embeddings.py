"""
Cache DeepSTARR encoder embeddings for fast training.

This script pre-computes encoder outputs for all DeepSTARR sequences
and saves them to disk. Training with cached embeddings is 10-100x faster.

Usage:
    python scripts/cache_deepstarr_embeddings.py --split train
    python scripts/cache_deepstarr_embeddings.py --split val  
    python scripts/cache_deepstarr_embeddings.py --split test
"""

import argparse
import pickle
import hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
from tqdm import tqdm

from alphagenome_ft import create_model_with_custom_heads
from alphagenome_research.model import dna_model
from src import DeepSTARRDataset


def compute_sequence_hash(sequence: str) -> str:
    """Compute SHA256 hash of sequence for cache key."""
    return hashlib.sha256(sequence.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Cache DeepSTARR embeddings')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--data_path', type=str, default='./data/deepstarr')
    parser.add_argument('--output_dir', type=str, default='./.cache/embeddings')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Caching DeepSTARR {args.split} embeddings")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"deepstarr_{args.split}_embeddings.pkl"
    
    # Create model (no custom heads needed, just encoder)
    print("\nLoading AlphaGenome model...")
    model = create_model_with_custom_heads(
        'all_folds',
        custom_heads=[],  # No custom heads needed
        use_encoder_output=True
    )
    print("✓ Model loaded")
    
    # Load dataset (no augmentations)
    print(f"\nLoading {args.split} dataset...")
    dataset = DeepSTARRDataset(
        model=model,
        path_to_data=args.data_path,
        split=args.split,
        organism=dna_model.Organism.HOMO_SAPIENS,
        random_shift=False,
        reverse_complement=False,
        use_cached_embeddings=False,
    )
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Cache embeddings
    print(f"\nComputing encoder embeddings...")
    embeddings_cache = {}
    
    # Process in batches for efficiency
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=len(dataset), desc="Encoding") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset))
            
            # Get batch
            batch_seqs = []
            batch_orgs = []
            batch_hashes = []
            
            for idx in range(start_idx, end_idx):
                sample = dataset[idx]
                batch_seqs.append(sample['seq'])
                batch_orgs.append(sample['organism_index'])
                
                # Compute hash for this sequence
                seq_str = dataset.data.iloc[idx]['seq']
                seq_hash = compute_sequence_hash(seq_str)
                batch_hashes.append(seq_hash)
            
            # Stack into batch
            batch_seqs = jnp.stack(batch_seqs, axis=0)
            batch_orgs = jnp.stack(batch_orgs, axis=0)
            
            # Forward pass through encoder
            with model._device_context:
                _ = model._predict(
                    model._params,
                    model._state,
                    batch_seqs,
                    batch_orgs,
                    negative_strand_mask=jnp.zeros(len(batch_seqs), dtype=bool),
                    strand_reindexing=jax.device_put(
                        model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                        model._device_context._device
                    ),
                )
            
            # Get encoder outputs
            encoder_outputs = model._last_encoder_output  # (batch, seq_len//128, D)
            
            # Store in cache
            for i, seq_hash in enumerate(batch_hashes):
                embeddings_cache[seq_hash] = encoder_outputs[i]
            
            pbar.update(end_idx - start_idx)
            
            # Print first batch info
            if batch_idx == 0:
                print(f"\n  Embedding shape: {encoder_outputs[0].shape}")
                print(f"  Encoder positions: {encoder_outputs[0].shape[0]}")
                print(f"  Feature dimension: {encoder_outputs[0].shape[1]}")
    
    # Save cache
    print(f"\nSaving embeddings to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_cache, f)
    
    print(f"✓ Saved {len(embeddings_cache)} embeddings")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"\nTo use cached embeddings in training:")
    print(f"  python scripts/finetune_starrseq.py \\")
    print(f"      --use_cached_embeddings \\")
    print(f"      --cache_file {output_file}")


if __name__ == '__main__':
    main()
