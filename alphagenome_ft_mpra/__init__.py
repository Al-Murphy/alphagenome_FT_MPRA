"""
Source utilities for AlphaGenome finetuning.

This package provides utility classes and functions for finetuning AlphaGenome models on MPRA data.
"""

__version__ = "0.1.2"

from .mpra_heads import MPRAHead, EncoderMPRAHead, DeepSTARRHead
from .data import LentiMPRADataset, MPRADataLoader, DeepSTARRDataset, STARRSeqDataLoader
from .oracle import MPRAOracle, load_oracle
from .training import train, validate

__all__ = [
    '__version__',
    'MPRAHead',
    'EncoderMPRAHead',
    'DeepSTARRHead',
    'LentiMPRADataset',
    'MPRADataLoader',
    'DeepSTARRDataset',
    'STARRSeqDataLoader',
    'MPRAOracle',
    'load_oracle',
    'train',
    'validate',
]
