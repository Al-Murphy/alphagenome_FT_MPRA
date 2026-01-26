"""
Data classes for AlphaGenome MPRA finetuning.
"""

import os
import pickle
import hashlib
import pandas as pd
import jax
import jax.numpy as jnp
from typing import Any
from pathlib import Path
from alphagenome_research.model import dna_model
from Bio import SeqIO


class LentiMPRADataset:
    def __init__(
        self,
        model: Any,  # Accepts dna_model.AlphaGenomeModel or CustomAlphaGenomeModel
        path_to_data: str = "./data/legnet_lentimpra",
        cell_type: str = "HepG2",
        split: str = "train",
        test_fold: int = [10],
        val_fold: int = [1],
        train_fold: int = [2, 3, 4, 5, 6, 7, 8, 9],
        promoter_seq: str = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG",
        rand_barcode: str = "AGAGACTGAGGCCAC",
        organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
        random_shift: bool = False,
        random_shift_likelihood: float = 0.5,
        max_shift: int = 15, 
        reverse_complement: bool = False,
        reverse_complement_likelihood: float = 0.5,
        rng_key: jax.Array | None = None,
        subset_frac: float = 1.0,
        pad_n_bases: int = 0,
        use_cached_embeddings: bool = False,
        cache_file: str | None = None,
    ):
        assert split in ["train", "val", "test"], f"split must be one of train, val, test"
        assert cell_type in ["HepG2", "K562", "WTC11"], f"cell_type must be one of HepG2, K562, WTC11"
        assert os.path.exists(path_to_data), f"path_to_data must exist"
        
        self.path_to_data = path_to_data
        self.cell_type = cell_type
        self.split = split
        # get test, val, train folds
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.train_fold = train_fold
        self.chosen_fold = train_fold if split == "train" else val_fold if split == "val" else test_fold
        self.promoter_seq = promoter_seq
        self.rand_barcode = rand_barcode
        # model used for ohe
        self.model = model
        self.organism = organism
        # rev comp
        assert reverse_complement_likelihood >= 0 and reverse_complement_likelihood <= 1, "reverse_complement_likelihood must be between 0 and 1"
        self.reverse_complement = reverse_complement
        self.reverse_complement_likelihood = reverse_complement_likelihood
        self.random_shift = random_shift
        self.random_shift_likelihood = random_shift_likelihood
        self.max_shift = max_shift
        assert random_shift_likelihood >= 0 and random_shift_likelihood <= 1, "random_shift_likelihood must be between 0 and 1"
        assert max_shift >= 0, "max_shift must be greater than 0"
        assert max_shift <= 15, "max_shift must be less than or equal to 15 - i.e. the adaptor length on the sequence"
        
        assert subset_frac >= 0 and subset_frac <= 1, "subset_frac must be between 0 and 1"
        self.subset_frac = subset_frac
        
        assert pad_n_bases >= 0, "pad_n_bases must be greater than or equal to 0"
        self.pad_n_bases = pad_n_bases
        
        # Initialize PRNG key for JAX random number generation
        if rng_key is None:
            self.rng_key = jax.random.PRNGKey(42)
        else:
            self.rng_key = rng_key
        
        # load the data and process FIRST (needed for cache loading)
        self.data = pd.read_csv(os.path.join(path_to_data, f"{cell_type}.tsv"), sep="\t")
        self.data = self.data[self.data["rev"] == 0]  # https://github.com/autosome-ru/human_legnet/issues/1
        # filt by fold
        self.data = self.data[self.data["fold"].isin(self.chosen_fold)]
        self.data = self.data.reset_index(drop=True)
        if self.subset_frac < 1.0:
            self.data = self.data.sample(frac=self.subset_frac)
        print(f"Loaded {len(self.data)} samples for {split} split")
        
        # Cached embeddings support (must come AFTER data loading)
        self.use_cached_embeddings = use_cached_embeddings
        self._embedding_cache = None
        if use_cached_embeddings:
            if cache_file is None:
                raise ValueError("cache_file must be provided when use_cached_embeddings=True")
            self._load_embedding_cache(cache_file)
            # When using cached embeddings, disable augmentations
            if random_shift or reverse_complement:
                print("Warning: Augmentations disabled when using cached embeddings")
                self.random_shift = False
                self.reverse_complement = False
            else:
                self.random_shift = random_shift
                self.reverse_complement = reverse_complement
        else:
            self.random_shift = random_shift
            self.reverse_complement = reverse_complement
    
    def _compute_sequence_hash(self, sequence: str) -> str:
        """Compute SHA256 hash of sequence for cache key."""
        return hashlib.sha256(sequence.encode()).hexdigest()
    
    def _load_embedding_cache(self, cache_file: str):
        """Load embedding cache and create index-based lookup for fast access.
        
        This pre-computes the sequence→hash mapping ONCE at load time,
        so __getitem__ becomes a simple index lookup (no hashing per access).
        """
        cache_path = Path(cache_file)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        print(f"Loading embedding cache from {cache_file}...")
        with open(cache_path, 'rb') as f:
            hash_to_embedding = pickle.load(f)
        
        print(f"✓ Loaded {len(hash_to_embedding)} cached embeddings")
        
        # Pre-compute index→embedding mapping for FAST access
        # This does all the expensive string ops and hashing ONCE
        print("Building index-based lookup (one-time cost)...")
        self._embeddings_by_idx = []
        self._labels_by_idx = []
        
        missing_count = 0
        for idx in range(len(self.data)):
            # Build sequence string (do this once, not per-access)
            sequence = self.data.iloc[idx]['seq'] + self.promoter_seq + self.rand_barcode
            if self.pad_n_bases > 0:
                padding_amount = self.pad_n_bases // 2
                sequence = 'N' * padding_amount + sequence + 'N' * padding_amount
            
            # Compute hash once
            seq_hash = self._compute_sequence_hash(sequence)
            
            if seq_hash not in hash_to_embedding:
                missing_count += 1
                if missing_count <= 3:
                    print(f"  Warning: Hash {seq_hash[:16]}... not in cache for idx {idx}")
                continue
            
            # Store embedding and label by index
            self._embeddings_by_idx.append(hash_to_embedding[seq_hash])
            self._labels_by_idx.append(self.data.iloc[idx]['mean_value'])
        
        if missing_count > 0:
            print(f"  Warning: {missing_count} sequences not found in cache")
        
        # Update data length to match what we actually have
        self._cached_len = len(self._embeddings_by_idx)
        print(f"✓ Built fast index lookup for {self._cached_len} samples")
        
    def __len__(self):
        # Return total number of samples
        # For cached embeddings, use the pre-computed count
        if self.use_cached_embeddings and hasattr(self, '_cached_len'):
            return self._cached_len
        return len(self.data)
    
    def _reverse_complement_onehot(self, seq_onehot: jnp.ndarray, force: bool = False) -> tuple[jnp.ndarray, jax.Array]:
        """
        Reverse complement a one-hot encoded DNA sequence.
        
        Uses alphagenome_research approach:
        1. Reverse along sequence dimension
        2. Swap channels: A↔T (0↔3), C↔G (1↔2)
        
        This matches the approach used in alphagenome_research/model/augmentation.py
        for reverse complementing predictions, adapted for one-hot sequences.
        
        Args:
            seq_onehot: One-hot encoded sequence (seq_len, 4) or (batch, seq_len, 4)
            force: If True, always reverse complement regardless of random check
            
        Returns:
            Tuple of (reverse_complemented_sequence, updated_rng_key)
        """
        # Generate random number for augmentation (if not forced)
        if not force:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            rand_num = jax.random.uniform(subkey, shape=(1,))[0]
            if rand_num > self.reverse_complement_likelihood:
                return seq_onehot, self.rng_key
        
        # Reverse complement: reverse sequence and swap channels
        # Channel mapping: A(0)↔T(3), C(1)↔G(2) -> [3, 2, 1, 0]
        # This matches the strand_reindexing used in alphagenome_research:
        # Matches the approach in `alphagenome_research/model/augmentation.py` lines 68-70
        strand_reindexing = jnp.array([3, 2, 1, 0])
        
        # Reverse along sequence dimension (axis 0 for 2D, axis 1 for 3D)
        if seq_onehot.ndim == 2:
            # Shape: (seq_len, 4)
            rev_seq = seq_onehot[::-1, :]
            rev_comp = rev_seq[:, strand_reindexing]
        else:
            # Shape: (batch, seq_len, 4)
            rev_seq = seq_onehot[:, ::-1, :]
            rev_comp = rev_seq[:, :, strand_reindexing]
        
        return rev_comp, self.rng_key
    
    def _random_shift_onehot(self, seq_onehot: jnp.ndarray, force: bool = False) -> tuple[jnp.ndarray, jax.Array]:
        """
        Randomly shift the sequence by a random number of bases.
        """
        # Check if we should apply the shift (random check)
        if not force:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            rand_num = jax.random.uniform(subkey, shape=(1,))[0]
            if rand_num > self.random_shift_likelihood:
                return seq_onehot, self.rng_key
        
        # Get random shift amount (split key again for new randomness)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        shift_amount = jax.random.randint(subkey, shape=(), minval=-self.max_shift, maxval=self.max_shift + 1)
        
        # Shift the sequence
        seq_onehot = jnp.roll(seq_onehot, shift_amount, axis=0)
        return seq_onehot, self.rng_key
    
    def __getitem__(self, idx):
        organism_index = jnp.array([0]) if self.organism == dna_model.Organism.HOMO_SAPIENS else jnp.array([1])
        
        if self.use_cached_embeddings:
            # FAST path: Simple index lookup (no string ops, no hashing!)
            # All expensive operations were done once at load time
            encoder_output = jnp.array(self._embeddings_by_idx[idx])
            label = self._labels_by_idx[idx]
            
            return {
                "encoder_output": encoder_output,
                "y": label,
                "organism_index": organism_index
            }
        else:
            # Non-cached path: do full sequence processing
            sequence = self.data.iloc[idx]['seq'] + self.promoter_seq + self.rand_barcode
            if self.pad_n_bases > 0:
                padding_amount = self.pad_n_bases // 2
                sequence = 'N' * padding_amount + sequence + 'N' * padding_amount
            label = self.data.iloc[idx]['mean_value']
            # Normal mode: return one-hot encoded sequence
            sequence_onehot = self.model._one_hot_encoder.encode(sequence)
            sequence_onehot = jnp.array(sequence_onehot)  # Convert to JAX array
            
            if self.reverse_complement:
                # Get random number to check if we should reverse complement
                sequence_onehot, self.rng_key = self._reverse_complement_onehot(sequence_onehot)
                # Don't need to touch y since it's a scalar
            if self.random_shift:
                sequence_onehot, self.rng_key = self._random_shift_onehot(sequence_onehot)
            
            return {
                "seq": sequence_onehot,
                "y": label,
                "organism_index": organism_index
            }


class MPRADataLoader:
    """Simple DataLoader for JAX/Haiku training.
    
    Batches samples from a dataset and stacks them into JAX arrays.
    """
    
    def __init__(
        self,
        dataset: LentiMPRADataset,
        batch_size: int = 32,
        shuffle: bool = False,
        rng_key: jax.Array | None = None,
    ):
        """
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: If True, shuffle dataset before each epoch
            rng_key: Random key for shuffling (required if shuffle=True)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_key = rng_key
        
        if shuffle and rng_key is None:
            self.rng_key = jax.random.PRNGKey(42)
    
    def __iter__(self):
        """Create iterator over batches."""
        # Get indices
        indices = jnp.arange(len(self.dataset))
        
        # Shuffle if requested
        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            indices = jax.random.permutation(subkey, indices)
        
        # Batch the indices
        num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            batch_indices = indices[start_idx:end_idx].tolist()
            
            # Get samples
            batch_samples = [self.dataset[int(idx)] for idx in batch_indices]
            
            # Stack into batches
            batch = self._stack_batch(batch_samples)
            yield batch
    
    def _stack_batch(self, samples: list[dict]) -> dict:
        """Stack list of samples into batched arrays.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Batched dictionary with stacked arrays
        """
        # Check if using cached embeddings
        if "encoder_output" in samples[0]:
            # Stack cached encoder embeddings
            encoder_outputs = [s["encoder_output"] for s in samples]
            # Handle variable length by padding or using max length
            max_len = max(e.shape[0] for e in encoder_outputs)
            max_feat = encoder_outputs[0].shape[1]  # Feature dimension
            
            padded_embeddings = []
            for emb in encoder_outputs:
                if emb.shape[0] < max_len:
                    padding = jnp.zeros((max_len - emb.shape[0], max_feat))
                    emb = jnp.concatenate([emb, padding], axis=0)
                padded_embeddings.append(emb)
            
            batch_encoder_output = jnp.stack(padded_embeddings, axis=0)  # (batch, seq_len//128, D)
            
            # Stack labels
            batch_y = jnp.array([s["y"] for s in samples])  # (batch,)
            
            # Stack organism indices
            batch_org = jnp.stack([s["organism_index"] for s in samples], axis=0)  # (batch, 1)
            batch_org = batch_org.squeeze(-1) if batch_org.shape[-1] == 1 else batch_org  # (batch,)
            
            return {
                "encoder_output": batch_encoder_output,
                "y": batch_y,
                "organism_index": batch_org,
            }
        else:
            # Normal mode: stack sequences
            seqs = [s["seq"] for s in samples]
            max_len = max(s.shape[0] for s in seqs)
            
            # Pad sequences to same length
            padded_seqs = []
            for seq in seqs:
                if seq.shape[0] < max_len:
                    padding = jnp.zeros((max_len - seq.shape[0], 4))
                    seq = jnp.concatenate([seq, padding], axis=0)
                padded_seqs.append(seq)
            
            batch_seq = jnp.stack(padded_seqs, axis=0)  # (batch, seq_len, 4)
            
            # Stack labels
            batch_y = jnp.array([s["y"] for s in samples])  # (batch,)
            
            # Stack organism indices
            batch_org = jnp.stack([s["organism_index"] for s in samples], axis=0)  # (batch, 1)
            batch_org = batch_org.squeeze(-1) if batch_org.shape[-1] == 1 else batch_org  # (batch,)
            
            return {
                "seq": batch_seq,
                "y": batch_y,
                "organism_index": batch_org,
            }
    
    def __len__(self):
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DeepSTARRDataset:
    """Dataset for DeepSTARR enhancer activity prediction.
    
    DeepSTARR predicts two outputs for each sequence:
    - Developmental enhancer activity (Dev_log2_enrichment)
    - Housekeeping enhancer activity (Hk_log2_enrichment)
    
    Data format:
    - Sequences in FASTA files (249bp CREs)
    - Activities in TSV files with columns: Dev_log2_enrichment, Hk_log2_enrichment
    """
    
    def __init__(
        self,
        model: Any,
        path_to_data: str = "./data/deepstarr",
        split: str = "train",
        organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
        random_shift: bool = False,
        random_shift_likelihood: float = 0.5,
        max_shift: int = 20,
        reverse_complement: bool = False,
        reverse_complement_likelihood: float = 0.5,
        rng_key: jax.Array | None = None,
        subset_frac: float = 1.0,
        use_cached_embeddings: bool = False,
        cache_file: str | None = None,
    ):
        """
        Args:
            model: AlphaGenome model (for one-hot encoding)
            path_to_data: Path to deepstarr data directory
            split: One of 'train', 'val', or 'test'
            organism: Organism (should be DROSOPHILA_MELANOGASTER for DeepSTARR)
            random_shift: Apply random shifts to sequences (augmentation)
            random_shift_likelihood: Probability of applying random shift
            max_shift: Maximum shift amount in base pairs
            reverse_complement: Apply reverse complement augmentation
            reverse_complement_likelihood: Probability of applying reverse complement
            rng_key: Random key for JAX
            subset_frac: Fraction of dataset to use (for debugging)
            use_cached_embeddings: Use pre-computed encoder embeddings
            cache_file: Path to cached embeddings file
        """
        assert split in ["train", "val", "test"], f"split must be one of train, val, test"
        assert os.path.exists(path_to_data), f"path_to_data must exist: {path_to_data}"
        
        self.path_to_data = path_to_data
        self.split = split
        self.model = model
        self.organism = organism
        
        # Augmentation settings
        assert reverse_complement_likelihood >= 0 and reverse_complement_likelihood <= 1
        assert random_shift_likelihood >= 0 and random_shift_likelihood <= 1
        assert max_shift >= 0
        self.reverse_complement = reverse_complement
        self.reverse_complement_likelihood = reverse_complement_likelihood
        self.random_shift = random_shift
        self.random_shift_likelihood = random_shift_likelihood
        self.max_shift = max_shift
        
        assert subset_frac > 0 and subset_frac <= 1, "subset_frac must be between 0 and 1"
        self.subset_frac = subset_frac
        
        # Initialize PRNG key
        if rng_key is None:
            self.rng_key = jax.random.PRNGKey(42)
        else:
            self.rng_key = rng_key
        
        # Load data FIRST (needed for cache loading)
        self._load_data()
        
        # Cached embeddings support (must come AFTER data loading)
        self.use_cached_embeddings = use_cached_embeddings
        self._embedding_cache = None
        if use_cached_embeddings:
            if cache_file is None:
                raise ValueError("cache_file must be provided when use_cached_embeddings=True")
            self._load_embedding_cache(cache_file)
            # Disable augmentations when using cached embeddings
            if random_shift or reverse_complement:
                print("Warning: Augmentations disabled when using cached embeddings")
                self.random_shift = False
                self.reverse_complement = False
    
    def _load_data(self):
        """Load sequences and activities from FASTA and TSV files."""
        # Map split names to file names
        split_map = {
            'train': 'Train',
            'val': 'Val',
            'test': 'Test',
        }
        file_suffix = split_map[self.split]
        
        # Load sequences from FASTA
        fasta_file = os.path.join(self.path_to_data, f"Sequences_{file_suffix}.fa")
        sequences = []
        seq_ids = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
            seq_ids.append(record.id)
        
        # Load activities from TSV
        activity_file = os.path.join(self.path_to_data, f"Sequences_activity_{file_suffix}.txt")
        activities_df = pd.read_csv(activity_file, sep="\t")
        
        # Verify counts match (account for header in activity file)
        assert len(sequences) == len(activities_df), \
            f"Sequence count ({len(sequences)}) doesn't match activity count ({len(activities_df)})"
        
        # Create DataFrame with sequences and activities
        self.data = pd.DataFrame({
            'seq_id': seq_ids,
            'seq': sequences,
            'dev_activity': activities_df['Dev_log2_enrichment'].values,
            'hk_activity': activities_df['Hk_log2_enrichment'].values,
        })
        
        # Apply subset if requested
        if self.subset_frac < 1.0:
            self.data = self.data.sample(frac=self.subset_frac, random_state=42)
            self.data = self.data.reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} DeepSTARR sequences for {self.split} split")
    
    def _compute_sequence_hash(self, sequence: str) -> str:
        """Compute SHA256 hash of sequence for cache key."""
        return hashlib.sha256(sequence.encode()).hexdigest()
    
    def _load_embedding_cache(self, cache_file: str):
        """Load embedding cache and create index-based lookup for fast access."""
        cache_path = Path(cache_file)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        print(f"Loading embedding cache from {cache_file}...")
        with open(cache_path, 'rb') as f:
            hash_to_embedding = pickle.load(f)
        
        print(f"✓ Loaded {len(hash_to_embedding)} cached embeddings")
        
        # Pre-compute index→embedding mapping for FAST access
        print("Building index-based lookup (one-time cost)...")
        self._embeddings_by_idx = []
        self._labels_by_idx = []
        
        missing_count = 0
        for idx in range(len(self.data)):
            sequence = self.data.iloc[idx]['seq']
            seq_hash = self._compute_sequence_hash(sequence)
            
            if seq_hash not in hash_to_embedding:
                missing_count += 1
                if missing_count <= 3:
                    print(f"  Warning: Hash {seq_hash[:16]}... not in cache for idx {idx}")
                continue
            
            # Store embedding and labels by index
            dev_activity = self.data.iloc[idx]['dev_activity']
            hk_activity = self.data.iloc[idx]['hk_activity']
            self._embeddings_by_idx.append(hash_to_embedding[seq_hash])
            self._labels_by_idx.append((dev_activity, hk_activity))
        
        if missing_count > 0:
            print(f"  Warning: {missing_count} sequences not found in cache")
        
        self._cached_len = len(self._embeddings_by_idx)
        print(f"✓ Built fast index lookup for {self._cached_len} samples")
    
    def __len__(self):
        if self.use_cached_embeddings and hasattr(self, '_cached_len'):
            return self._cached_len
        return len(self.data)
    
    def _reverse_complement_onehot(self, seq_onehot: jnp.ndarray, force: bool = False) -> tuple[jnp.ndarray, jax.Array]:
        """Reverse complement a one-hot encoded DNA sequence."""
        if not force:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            rand_num = jax.random.uniform(subkey, shape=(1,))[0]
            if rand_num > self.reverse_complement_likelihood:
                return seq_onehot, self.rng_key
        
        # Reverse complement: reverse sequence and swap channels
        # A(0)↔T(3), C(1)↔G(2) -> [3, 2, 1, 0]
        strand_reindexing = jnp.array([3, 2, 1, 0])
        
        if seq_onehot.ndim == 2:
            rev_seq = seq_onehot[::-1, :]
            rev_comp = rev_seq[:, strand_reindexing]
        else:
            rev_seq = seq_onehot[:, ::-1, :]
            rev_comp = rev_seq[:, :, strand_reindexing]
        
        return rev_comp, self.rng_key
    
    def _random_shift_onehot(self, seq_onehot: jnp.ndarray, force: bool = False) -> tuple[jnp.ndarray, jax.Array]:
        """Randomly shift the sequence by a random number of bases."""
        if not force:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            rand_num = jax.random.uniform(subkey, shape=(1,))[0]
            if rand_num > self.random_shift_likelihood:
                return seq_onehot, self.rng_key
        
        self.rng_key, subkey = jax.random.split(self.rng_key)
        shift_amount = jax.random.randint(subkey, shape=(), minval=-self.max_shift, maxval=self.max_shift + 1)
        
        seq_onehot = jnp.roll(seq_onehot, shift_amount, axis=0)
        return seq_onehot, self.rng_key
    
    def __getitem__(self, idx):
        organism_index = jnp.array([0]) if self.organism == dna_model.Organism.HOMO_SAPIENS else jnp.array([1])
        
        if self.use_cached_embeddings:
            # FAST path: Simple index lookup
            encoder_output = jnp.array(self._embeddings_by_idx[idx])
            dev_activity, hk_activity = self._labels_by_idx[idx]
            labels = jnp.array([dev_activity, hk_activity])
            
            return {
                "encoder_output": encoder_output,
                "y": labels,
                "organism_index": organism_index
            }
        else:
            # Non-cached path: do full sequence processing
            sequence = self.data.iloc[idx]['seq']
            dev_activity = self.data.iloc[idx]['dev_activity']
            hk_activity = self.data.iloc[idx]['hk_activity']
            labels = jnp.array([dev_activity, hk_activity])
            
            # One-hot encode
            sequence_onehot = self.model._one_hot_encoder.encode(sequence)
            sequence_onehot = jnp.array(sequence_onehot)
            
            # Apply augmentations
            if self.reverse_complement:
                sequence_onehot, self.rng_key = self._reverse_complement_onehot(sequence_onehot)
            if self.random_shift:
                sequence_onehot, self.rng_key = self._random_shift_onehot(sequence_onehot)
            
            return {
                "seq": sequence_onehot,
                "y": labels,
                "organism_index": organism_index
            }


class STARRSeqDataLoader:
    """DataLoader for DeepSTARR dataset (works with DeepSTARRDataset).
    
    Similar to MPRADataLoader but handles 2-output labels.
    """
    
    def __init__(
        self,
        dataset: DeepSTARRDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        rng_key: jax.Array | None = None,
    ):
        """
        Args:
            dataset: DeepSTARRDataset to load from
            batch_size: Number of samples per batch
            shuffle: If True, shuffle dataset before each epoch
            rng_key: Random key for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_key = rng_key
        
        if shuffle and rng_key is None:
            self.rng_key = jax.random.PRNGKey(42)
    
    def __iter__(self):
        """Create iterator over batches."""
        indices = jnp.arange(len(self.dataset))
        
        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            indices = jax.random.permutation(subkey, indices)
        
        num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            batch_indices = indices[start_idx:end_idx].tolist()
            
            batch_samples = [self.dataset[int(idx)] for idx in batch_indices]
            batch = self._stack_batch(batch_samples)
            yield batch
    
    def _stack_batch(self, samples: list[dict]) -> dict:
        """Stack list of samples into batched arrays."""
        if "encoder_output" in samples[0]:
            # Cached embeddings mode
            encoder_outputs = [s["encoder_output"] for s in samples]
            max_len = max(e.shape[0] for e in encoder_outputs)
            max_feat = encoder_outputs[0].shape[1]
            
            padded_embeddings = []
            for emb in encoder_outputs:
                if emb.shape[0] < max_len:
                    padding = jnp.zeros((max_len - emb.shape[0], max_feat))
                    emb = jnp.concatenate([emb, padding], axis=0)
                padded_embeddings.append(emb)
            
            batch_encoder_output = jnp.stack(padded_embeddings, axis=0)
            batch_y = jnp.stack([s["y"] for s in samples], axis=0)  # (batch, 2)
            batch_org = jnp.stack([s["organism_index"] for s in samples], axis=0)
            batch_org = batch_org.squeeze(-1) if batch_org.shape[-1] == 1 else batch_org
            
            return {
                "encoder_output": batch_encoder_output,
                "y": batch_y,
                "organism_index": batch_org,
            }
        else:
            # Normal mode: stack sequences
            seqs = [s["seq"] for s in samples]
            max_len = max(s.shape[0] for s in seqs)
            
            padded_seqs = []
            for seq in seqs:
                if seq.shape[0] < max_len:
                    padding = jnp.zeros((max_len - seq.shape[0], 4))
                    seq = jnp.concatenate([seq, padding], axis=0)
                padded_seqs.append(seq)
            
            batch_seq = jnp.stack(padded_seqs, axis=0)
            batch_y = jnp.stack([s["y"] for s in samples], axis=0)  # (batch, 2)
            batch_org = jnp.stack([s["organism_index"] for s in samples], axis=0)
            batch_org = batch_org.squeeze(-1) if batch_org.shape[-1] == 1 else batch_org
            
            return {
                "seq": batch_seq,
                "y": batch_y,
                "organism_index": batch_org,
            }
    
    def __len__(self):
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size