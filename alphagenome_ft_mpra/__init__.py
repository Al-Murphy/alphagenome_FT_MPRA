"""
Source utilities for AlphaGenome finetuning.

This package provides utility classes and functions for finetuning AlphaGenome models on MPRA data.
"""

__version__ = "0.1.2"

# The AlphaGenome path pulls the JAX / Haiku / alphagenome_research stack. The
# plant STARR-seq runners for the other models (PlantCAD2, Jores CNN, NTv3) run
# in their own environments that do NOT have that stack, but still need to import
# the JAX-free helpers below (plant_starrseq_utils / plant_torch). So the
# JAX-dependent exports are optional — same pattern as the torch guard.
try:
    from .mpra_heads import MPRAHead, EncoderMPRAHead, DeepSTARRHead, PlantMPRAHead
    from .data import (
        LentiMPRADataset,
        MPRADataLoader,
        DeepSTARRDataset,
        STARRSeqDataLoader,
        EpisomalMPRADataset,
        PlantStarrSeqDataset,
    )
    from .episomal_utils import get_episomal_test_sets
    from .oracle import MPRAOracle, load_oracle
    from .training import train, validate
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

# Always available — pure pandas/numpy, no JAX/torch.
from .plant_starrseq_utils import (
    build_plant_starrseq_dataset,
    data_is_present as plant_starrseq_data_is_present,
)

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
    'build_plant_starrseq_dataset',
    'plant_starrseq_data_is_present',
]
if _HAS_JAX:
    __all__ += [
        'MPRAHead',
        'EncoderMPRAHead',
        'DeepSTARRHead',
        'PlantMPRAHead',
        'LentiMPRADataset',
        'MPRADataLoader',
        'DeepSTARRDataset',
        'STARRSeqDataLoader',
        'EpisomalMPRADataset',
        'PlantStarrSeqDataset',
        'get_episomal_test_sets',
        'MPRAOracle',
        'load_oracle',
        'train',
        'validate',
    ]
if _HAS_TORCH:
    __all__.append('EpisomalMPRADatasetPyTorch')
