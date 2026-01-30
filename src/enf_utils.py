"""
Utils for fine-tuning the Enformer model for MPRA activity prediction.

Usage:

import torch
from enformer_pytorch import from_pretrained

PROMOTER_CONSTRUCT_LENGTH = 281

enformer = from_pretrained('EleutherAI/enformer-official-rough',use_tf_gamma = False)

model = EncoderMPRAHead(
    enformer = enformer,
    num_tracks = 1
)
"""


import torch
from torch import nn
from copy import deepcopy
from contextlib import contextmanager
from typing import Optional
from enformer_pytorch.modeling_enformer import Enformer
import os
import pandas as pd
import numpy as np

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

# better sequential
def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# controlling freezing of layers

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_batchnorms_(model):
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]

    for bn in bns:
        bn.eval()
        bn.track_running_stats = False
        set_module_requires_grad_(bn, False)

def freeze_all_but_layernorms_(model):
    for m in model.modules():
        set_module_requires_grad_(m, isinstance(m, nn.LayerNorm))

def freeze_all_but_last_n_layers_(enformer, n):
    assert isinstance(enformer, Enformer)
    freeze_all_layers_(enformer)

    transformer_blocks = enformer.transformer

    for module in transformer_blocks[-n:]:
        set_module_requires_grad_(module, True)


# Updated get_enformer_embeddings to bypass _trunk when transformer is Identity
def get_enformer_embeddings(
    model,
    seq,
    freeze = False,
    train_layernorms_only = False,
    train_last_n_layers_only = None,
    enformer_kwargs: dict = {}
):
    freeze_batchnorms_(model)

    if train_layernorms_only:
        assert not freeze, 'you set the intent to train the layernorms of the enformer, yet also indicated you wanted to freeze the entire model'
        freeze_all_but_layernorms_(model)

    if exists(train_last_n_layers_only):
        assert not freeze, 'you set the intent to train last N layers of enformer, but also indicated you wanted to freeze the entire network'
        freeze_all_but_last_n_layers_(model, train_last_n_layers_only)

    enformer_context = null_context() if not freeze else torch.no_grad()

    with enformer_context:
        # Check if transformer is Identity (meaning we want to stop at conv_tower)
        # This bypasses the _trunk which includes TargetLengthCrop that enforces minimum length
        if isinstance(model.transformer, nn.Identity):
            # Directly call stem + conv_tower to avoid length restrictions
            # Input format: (B, L, 4) -> (B, 4, L) for conv layers
            from einops import rearrange
            x = rearrange(seq, "b n d -> b d n")
            x = model.stem(x)
            x = model.conv_tower(x)
            # Output format: (B, D, L) -> (B, L, D) for consistency
            embeddings = rearrange(x, "b d n -> b n d")
        else:
            # Use normal path through _trunk
            embeddings = model(seq, return_only_embeddings = True, **enformer_kwargs)

        if freeze:
            embeddings = embeddings.detach()

    return embeddings

class EncoderMPRAHead(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        num_tracks,
        transformer_embed_fn: nn.Module = nn.Identity(),
        output_activation: Optional[nn.Module] = nn.Identity(), #nn.Softplus(),
        auto_set_target_length = True
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        ENCODER_DIM = 1536

        self.enformer = enformer

        self.auto_set_target_length = auto_set_target_length

        #use pytorch enformer style of setting layers to identity beywond where we want
        self.enformer = deepcopy(enformer)
        #keeping stem and conv tower as is....
        self.enformer.transformer = nn.Identity()
        self.enformer.crop_final = nn.Identity()
        self.enformer.final_pointwise = nn.Identity()

        self.to_tracks = Sequential(
            #need to flatten first
            nn.Flatten(),
            nn.LayerNorm(ENCODER_DIM*3), #this 3 is for the 3 128bp bins
            nn.Linear(ENCODER_DIM*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_tracks),
            output_activation
        )

    def forward(
        self,
        seq,
        *,
        target = None,
        freeze_enformer = False,
        finetune_enformer_ln_only = False,
        finetune_last_n_layers_only = None
    ):
        enformer_kwargs = dict()

        if exists(target) and self.auto_set_target_length:
            enformer_kwargs = dict(target_length = target.shape[-2])

        
        embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)
        
        preds = self.to_tracks(embeddings)

        if not exists(target):
            return preds

        return torch.nn.functional.mse_loss(preds, target)


class LentiMPRADatasetPyTorch:
    """PyTorch-only version of LentiMPRADataset (no JAX dependencies).
    
    This is a simplified version that uses numpy arrays and Python's random
    module instead of JAX, making it compatible with PyTorch workflows.
    """
    
    def __init__(
        self,
        model,  # Model with _one_hot_encoder attribute
        path_to_data: str = "./data/legnet_lentimpra",
        cell_type: str = "HepG2",
        split: str = "train",
        test_fold: list = [10],
        val_fold: list = [1],
        train_fold: list = [2, 3, 4, 5, 6, 7, 8, 9],
        promoter_seq: str = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG",
        rand_barcode: str = "AGAGACTGAGGCCAC",
        random_shift: bool = False,
        random_shift_likelihood: float = 0.5,
        max_shift: int = 15,
        reverse_complement: bool = False,
        reverse_complement_likelihood: float = 0.5,
        subset_frac: float = 1.0,
        pad_n_bases: int = 0,
    ):
        assert split in ["train", "val", "test"], f"split must be one of train, val, test"
        assert cell_type in ["HepG2", "K562", "WTC11"], f"cell_type must be one of HepG2, K562, WTC11"
        assert os.path.exists(path_to_data), f"path_to_data must exist"
        
        self.path_to_data = path_to_data
        self.cell_type = cell_type
        self.split = split
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.train_fold = train_fold
        self.chosen_fold = train_fold if split == "train" else val_fold if split == "val" else test_fold
        self.promoter_seq = promoter_seq
        self.rand_barcode = rand_barcode
        self.model = model
        
        assert reverse_complement_likelihood >= 0 and reverse_complement_likelihood <= 1
        self.reverse_complement = reverse_complement
        self.reverse_complement_likelihood = reverse_complement_likelihood
        self.random_shift = random_shift
        self.random_shift_likelihood = random_shift_likelihood
        self.max_shift = max_shift
        assert random_shift_likelihood >= 0 and random_shift_likelihood <= 1
        assert max_shift >= 0 and max_shift <= 15
        
        assert subset_frac >= 0 and subset_frac <= 1
        self.subset_frac = subset_frac
        assert pad_n_bases >= 0
        self.pad_n_bases = pad_n_bases
        
        # Initialize random state (using numpy for reproducibility)
        self.rng = np.random.RandomState(42)
        
        # Load data
        self.data = pd.read_csv(os.path.join(path_to_data, f"{cell_type}.tsv"), sep="\t")
        self.data = self.data[self.data["rev"] == 0]  # Filter reverse sequences
        self.data = self.data[self.data["fold"].isin(self.chosen_fold)]
        self.data = self.data.reset_index(drop=True)
        if self.subset_frac < 1.0:
            self.data = self.data.sample(frac=self.subset_frac, random_state=42)
        print(f"Loaded {len(self.data)} samples for {split} split")
    
    def __len__(self):
        return len(self.data)
    
    def _reverse_complement_onehot(self, seq_onehot: np.ndarray, force: bool = False) -> np.ndarray:
        """Reverse complement a one-hot encoded DNA sequence using numpy."""
        if not force:
            if self.rng.random() > self.reverse_complement_likelihood:
                return seq_onehot
        
        # Reverse complement: reverse sequence and swap channels
        # Channel mapping: A(0)↔T(3), C(1)↔G(2) -> [3, 2, 1, 0]
        strand_reindexing = np.array([3, 2, 1, 0])
        
        if seq_onehot.ndim == 2:
            # Shape: (seq_len, 4)
            rev_seq = seq_onehot[::-1, :]
            rev_comp = rev_seq[:, strand_reindexing]
        else:
            # Shape: (batch, seq_len, 4)
            rev_seq = seq_onehot[:, ::-1, :]
            rev_comp = rev_seq[:, :, strand_reindexing]
        
        return rev_comp
    
    def _random_shift_onehot(self, seq_onehot: np.ndarray, force: bool = False) -> np.ndarray:
        """Randomly shift the sequence by a random number of bases using numpy."""
        if not force:
            if self.rng.random() > self.random_shift_likelihood:
                return seq_onehot
        
        shift_amount = self.rng.randint(-self.max_shift, self.max_shift + 1)
        seq_onehot = np.roll(seq_onehot, shift_amount, axis=0)
        return seq_onehot
    
    def __getitem__(self, idx):
        # Build sequence
        sequence = self.data.iloc[idx]['seq'] + self.promoter_seq + self.rand_barcode
        if self.pad_n_bases > 0:
            padding_amount = self.pad_n_bases // 2
            sequence = 'N' * padding_amount + sequence + 'N' * padding_amount
        
        label = self.data.iloc[idx]['mean_value']
        
        # One-hot encode
        sequence_onehot = self.model._one_hot_encoder.encode(sequence)
        sequence_onehot = np.array(sequence_onehot, dtype=np.float32)
        
        # Apply augmentations
        if self.reverse_complement:
            sequence_onehot = self._reverse_complement_onehot(sequence_onehot)
        if self.random_shift:
            sequence_onehot = self._random_shift_onehot(sequence_onehot)
        
        return {
            "seq": sequence_onehot,
            "y": label,
        }