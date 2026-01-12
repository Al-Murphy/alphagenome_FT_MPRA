"""
Data classes for AlphaGenome MPRA finetuning.
"""

import os
import pandas as pd
import jax
import jax.numpy as jnp
from typing import Any
from alphagenome_research.model import dna_model


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
        # Initialize PRNG key for JAX random number generation
        if rng_key is None:
            self.rng_key = jax.random.PRNGKey(42)
        else:
            self.rng_key = rng_key
        
        # load the data and process
        self.data = pd.read_csv(os.path.join(path_to_data, f"{cell_type}.tsv"), sep="\t")
        self.data = self.data[self.data["rev"] == 0]  # https://github.com/autosome-ru/human_legnet/issues/1
        # filt by fold
        self.data = self.data[self.data["fold"].isin(self.chosen_fold)]
        self.data = self.data.reset_index(drop=True)
        if self.subset_frac < 1.0:
            self.data = self.data.sample(frac=self.subset_frac)
        print(f"Loaded {len(self.data)} samples for {split} split")
        
    def __len__(self):
        # Return total number of samples
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
        # Get the sequence and label for the given index
        sequence = self.data.iloc[idx]['seq'] + self.promoter_seq + self.rand_barcode
        label = self.data.iloc[idx]['mean_value']
        
        # Convert sequence to one-hot encoding
        sequence_onehot = self.model._one_hot_encoder.encode(sequence)
        sequence_onehot = jnp.array(sequence_onehot)  # Convert to JAX array
        
        organism_index = jnp.array([0]) if self.organism == dna_model.Organism.HOMO_SAPIENS else jnp.array([1])
        
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
        # Stack sequences (handle variable length by padding or using max length)
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