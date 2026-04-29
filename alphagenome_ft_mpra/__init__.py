"""
Source utilities for AlphaGenome finetuning.

This package provides utility classes and functions for finetuning AlphaGenome models on MPRA data.
"""

__version__ = "0.1.2"

from .mpra_heads import MPRAHead, EncoderMPRAHead, DeepSTARRHead
from .data import (
    LentiMPRADataset,
    MPRADataLoader,
    DeepSTARRDataset,
    STARRSeqDataLoader,
    EpisomalMPRADataset,
)
from .episomal_utils import get_episomal_test_sets
from .oracle import MPRAOracle, load_oracle
from .training import train, validate

# EpisomalMPRADatasetPyTorch lives in enf_utils.py alongside the other PyTorch
# datasets so the Enformer training path stays JAX-free. We re-export it here
# for convenience, but only if torch is available — keeping import-light for
# JAX-only AG users.
try:
    from .enf_utils import EpisomalMPRADatasetPyTorch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

__all__ = [
    '__version__',
    'MPRAHead',
    'EncoderMPRAHead',
    'DeepSTARRHead',
    'LentiMPRADataset',
    'MPRADataLoader',
    'DeepSTARRDataset',
    'STARRSeqDataLoader',
    'EpisomalMPRADataset',
    'get_episomal_test_sets',
    'MPRAOracle',
    'load_oracle',
    'train',
    'validate',
]
if _HAS_TORCH:
    __all__.append('EpisomalMPRADatasetPyTorch')
