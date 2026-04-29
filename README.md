# Adapting AlphaGenome to MPRA data

![Modular Generalist seq2func models](assets/images/modular_generalists.png)

## Generalist Seq2Func Models for perturbation data Finetuning

This repository demonstrates finetuning **generalist seq2func models** (here AlphaGenome & Enformer) on MPRA (Massively Parallel Reporter Assay) and STARR-seq data. The modular approach shown here can be applied to **any generalist seq2func model** that provides sequence embeddings, making it a flexible framework for regulatory sequence prediction tasks.

The goal is to think of any pretrained generalist model as **modular components** that can be used separately for their _cis_-regulatory logic. Here we finetuned the generalists' encoders to predict reporter activity from genomic sequences, applying it to lentiMPRA and DeepSTARR datasets and evaluating performance zero-shot on CAGI5 data.

This approach leverages the rich sequence representations learned by large-scale generalist models while adapting them to specific regulatory tasks through task-specific prediction heads.

## Installation

### Prerequisites

1. **For AlphaGenome models**: Install AlphaGenome Research:
```bash
pip install git+https://github.com/google-deepmind/alphagenome_research.git
```

2. **For Enformer models**: Install Enformer PyTorch:
```bash
pip install enformer-pytorch
```

3. Install this project:
```bash
git clone https://github.com/Al-Murphy/alphagenome_FT_MPRA.git
cd alphagenome_FT_MPRA
pip install -e .
```

This will automatically install the `alphagenome-ft` package as a dependency (for AlphaGenome models) and other required packages.

## Contents

- **Overview**
  - [Architecture](#architecture)
  - [Quick Start](#quick-start)
- **Code Guides**
  - [Source utilities (`alphagenome_ft_mpra/`)](alphagenome_ft_mpra/README.md)
  - [Training / evaluation scripts (`scripts/`)](scripts/README.md)
  - [Config files (`configs/`)](configs/README.md)


## Architecture

### Modular Encoder Approach

This project demonstrates a **modular approach** to using generalist seq2func models:

```
DNA Sequence (B, S, 4)
    ↓
Generalist Model Backbone (frozen)
    ├── AlphaGenome: Encoder → Transformer → Decoder
    ├── Enformer: Convolutional blocks → Transformer → Output heads
    └── Other seq2func models...
    ↓
Sequence Embeddings (extracted from backbone)
    ├── High-resolution embeddings (1bp)
    ├── Low-resolution embeddings (128bp)
    └── Architecture-specific features
    ↓
Custom Task-Specific Heads (trainable)
    ├── MPRAHead: Reporter activity prediction
    ├── DeepSTARRHead: Enhancer activity prediction
    └── YOUR_CUSTOM_HEAD ← Add here
```

### Supported Models

1. **AlphaGenome**:
   - Multi-resolution embeddings (1bp, 128bp, pairwise)
   - Uses [`alphagenome-ft`](https://github.com/genomicsxai/alphagenome_ft) for finetuning utilities
   - See that repository for documentation on custom heads, parameter freezing, and model wrapping

2. **Enformer**:
   - Encoder-level embeddings at 128bp resolution
   - PyTorch implementation with custom heads
   - See `alphagenome_ft_mpra/enf_utils.py` for Enformer-specific utilities

3. **Other Generalist Models**:
   - The modular approach can be extended to any seq2func model
   - Key requirement: ability to extract sequence embeddings from the backbone
   - Custom heads can be implemented following the same pattern

### Fine-tuning Strategy

1. **Backbone**: Held fixed during training via **Optax optimizer masking** in `alphagenome-ft` (not only `stop_gradient` on parameters). Call `freeze_except_head(...)` before training so the default training loop uses a heads-only optimizer.
2. **Embeddings**: Extract multi-resolution sequence representations
3. **Heads**: Train task-specific prediction layers on top of frozen embeddings

This approach allows leveraging rich pretrained representations while efficiently adapting to new tasks.


## Quick Start

### AlphaGenome Example

```python
import jax
import jax.numpy as jnp
from alphagenome_research.model import dna_model
from alphagenome.models import dna_output
from alphagenome_ft import (
    CustomHead,
    HeadConfig,
    HeadType,
    register_custom_head,
    wrap_pretrained_model,
    add_custom_heads_to_model,
)
from alphagenome_ft_mpra.mpra_heads import MPRAHead

# 1. Register custom MPRA head
register_custom_head(
    'mpra_head',
    MPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='mpra_head',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
        metadata={'center_bp': 128, 'pooling_type': 'flatten', 'embedding_mode': '1bp'}
    )
)

# 2. Load pretrained model and add MPRA head
base_model = dna_model.create_from_kaggle('all_folds')
model = wrap_pretrained_model(base_model)
model = add_custom_heads_to_model(model, custom_heads=['mpra_head'])

# 3. Mark head-only finetuning (backbone fixed during training)
model.freeze_except_head('mpra_head')  # sets optimizer hint; see alphagenome-ft docs

# 4. Train on your MPRA data (training uses alphagenome_ft.optimizer_utils masking)
# See scripts/finetune_mpra.py for complete training example
```

### Enformer Example

```python
import torch
from enformer_pytorch import from_pretrained
from alphagenome_ft_mpra.enf_utils import EncoderMPRAHead

# 1. Load pretrained Enformer
enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=False)

# 2. Create model with custom MPRA head
model = EncoderMPRAHead(
    enformer=enformer,
    num_tracks=1,
    center_bp=256,
    pooling_type='sum'
)

# 3. Freeze Enformer backbone (only train head)
model.freeze_backbone()

# 4. Train on your MPRA data
# See scripts/finetune_enformer_mpra.py for complete training example
```

### MPRA Oracle API (pretrained checkpoints)
You can also use the pretrained model as an oracle. Currently, `MPRAOracle` is only supported.

```python
from alphagenome_ft_mpra import load_oracle

oracle = load_oracle(
    "/path/to/checkpoint_dir",
    # Optional construct pieces (set to None to skip)
    left_adapter=None,
    right_adapter=None,
    promoter="TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG",
    barcode="AGAGACTGAGGCCAC",
)

# mode="core": add left/right adapters + promoter + barcode (if provided)
# mode="flanked": add promoter + barcode (if provided)
# mode="full": no sequence additions

# Usage 1) onehot in shape: (S, 4) or (B, S, 4)
scores = oracle.predict(onehot, mode="core")

# Usage 2) string convenience wrapper
scores = oracle.predict_sequences(["ACGT..."], mode="core")
```


### Using Configuration Files

For both AlphaGenome and Enformer, you can use pre-configured hyperparameters:

```bash
# AlphaGenome with LentiMPRA
python scripts/finetune_mpra.py --config configs/mpra_HepG2.json

# Enformer with LentiMPRA
python scripts/finetune_enformer_mpra.py --config configs/mpra_HepG2.json

# AlphaGenome with DeepSTARR
python scripts/finetune_starrseq.py --config configs/starrseq.json

# Enformer with DeepSTARR
python scripts/finetune_enformer_starrseq.py --config configs/starrseq.json

# AlphaGenome with episomal MPRA (Gosai 2024)
python scripts/finetune_episomal_mpra.py --config configs/episomal_K562.json

# Enformer with episomal MPRA (Gosai 2024)
python scripts/finetune_enformer_episomal_mpra.py --config configs/episomal_K562.json
```

## Datasets

This repo expects datasets to live under `data/` (gitignored). None of the data
ships in the repository — fetch each dataset yourself from the cited source and
drop it at the indicated path.

| Dataset | Path | Source |
|---|---|---|
| LentiMPRA (Agarwal et al. 2025) | `data/legnet_lentimpra/` | per-cell TSV files (`HepG2.tsv`, `K562.tsv`, `WTC11.tsv`) |
| DeepSTARR (de Almeida et al. 2022) | `data/deepstarr/` | DeepSTARR genomic-context data |
| CAGI5 saturation mutagenesis | `data/cagi5/` | use `scripts/fetch_cagi5_references.py` to fetch reference sequences |
| **Episomal MPRA (Gosai et al. 2024)** | **`data/gosai_episomal/`** | **`DATA-Table_S2__MPRA_dataset.txt`** (main TSV; supplementary table from the Nature Genetics paper). Use `scripts/fetch_episomal_data.py` to download from the Tewhey-lab public bucket linked in the BODA2 README. Optional test sets at `data/gosai_episomal/test_sets/test_ood_designed_k562.tsv` and `data/gosai_episomal/test_sets/test_snv_pairs_hashfrag.tsv` enable the high-activity-designed and SNV-effects evaluations. |

## Project Structure

```
alphagenome_FT_MPRA/
├── alphagenome_ft_mpra/      # Source code
│   ├── mpra_heads.py         # Custom prediction heads (MPRAHead, EncoderMPRAHead, DeepSTARRHead)
│   ├── enf_utils.py          # Enformer-specific utilities and heads
│   ├── data.py               # JAX dataset classes (LentiMPRADataset, DeepSTARRDataset, EpisomalMPRADataset)
│   ├── episomal_utils.py     # Shared helpers for the Gosai 2024 episomal MPRA (chr splits, padding, OHE)
│   ├── seq_loader.py         # Sequence loading utilities
│   ├── training.py           # Training utilities and helpers
│   ├── oracle.py             # MPRA oracle loading + predict(onehot, mode=...) and predict_sequence(...)
│   └── __init__.py
├── scripts/                  # Executable training and evaluation scripts
│   ├── finetune_mpra.py      # Finetune AlphaGenome on LentiMPRA
│   ├── finetune_enformer_mpra.py  # Finetune Enformer on LentiMPRA
│   ├── finetune_starrseq.py  # Finetune AlphaGenome on DeepSTARR
│   ├── finetune_enformer_starrseq.py  # Finetune Enformer on DeepSTARR
│   ├── fetch_episomal_data.py     # Download the Gosai 2024 MPRA table into data/gosai_episomal/
│   ├── finetune_episomal_mpra.py  # Finetune AlphaGenome on episomal MPRA (Gosai 2024)
│   ├── finetune_enformer_episomal_mpra.py  # Finetune Enformer on episomal MPRA
│   ├── test_episomal_mpra.py  # Evaluate episomal MPRA models across 3 test sets
│   ├── plot_episomal_benchmark_results.py  # Bar plots for the episomal benchmark
│   ├── test_ft_model_*.py    # Evaluation scripts for finetuned models
│   ├── test_cagi5_zero_shot_*.py  # Zero-shot evaluation on CAGI5 benchmark
│   ├── compute_attributions_lentimpra.py  # Attribution analysis (DeepSHAP, gradients)
│   ├── compute_attributions_starrseq.py  # Attribution analysis (DeepSHAP, gradients)
│   ├── cache_embeddings.py   # Pre-compute embeddings for faster training
│   ├── create_mpra_comparison_table.py  # Generate performance comparison tables
│   └── README.md             # Script documentation
├── configs/                  # Hyperparameter configuration files
│   ├── mpra_HepG2.json       # Optimal config for HepG2 cell line
│   ├── mpra_K562.json        # Optimal config for K562 cell line
│   ├── mpra_WTC11.json       # Optimal config for WTC11 cell line
│   ├── starrseq.json         # Optimal config for DeepSTARR dataset
│   └── README.md             # Config file documentation
├── data/                     # Datasets
│   ├── legnet_lentimpra/     # LentiMPRA training data
│   ├── deepstarr/            # DeepSTARR dataset
│   ├── cagi5/                # CAGI5 benchmark data
│   └── motifs/               # Motif analysis data
├── results/                  # Training outputs and evaluations
│   ├── models/               # Saved model checkpoints
│   ├── benchmark_*.csv       # Benchmark results
│   ├── plots/                # Generated plots and figures
│   └── mpralegnet_predictions/  # LegNet baseline predictions
├── assets/                    # Images and figures
│   └── images/
│       └── modular_generalists.png
├── test.ipynb                # Example notebook
├── main.py                   # Entry point
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## Key Features

- **Model-Agnostic Design**: Works with any generalist seq2func model (AlphaGenome, Enformer, etc.)
- **Modular Architecture**: Separate frozen backbones from trainable task-specific heads
- **Multiple Datasets**: Support for LentiMPRA (multiple cell lines) and DeepSTARR
- **Flexible Embedding Access**: Use different resolution embeddings (1bp, 128bp, encoder-only)
- **Two-Stage Training**: Optional cached-embedding training for faster iteration
- **Comprehensive Evaluation**: Zero-shot benchmarks, attribution analysis, and comparison tables
- **Production-Ready Configs**: Pre-optimized hyperparameters for each dataset/cell line

## Extending to Other Models

To add support for another generalist seq2func model:

1. **Extract Embeddings**: Implement a function to extract sequence embeddings from your model
2. **Create Custom Head**: Implement a head class (see `alphagenome_ft_mpra/mpra_heads.py` for examples)
3. **Wrap Model**: Create a wrapper that freezes the backbone and exposes embeddings
4. **Add Training Script**: Follow the pattern in `scripts/finetune_*.py`

The key principle is: **freeze the generalist backbone, train only the task-specific head**.

## License

This project extends AlphaGenome and uses Enformer. Please refer to the original licenses:
- AlphaGenome: See [AlphaGenome Research license](https://github.com/google-deepmind/alphagenome_research)
- Enformer: See [Enformer license](https://github.com/deepmind/deepmind-research/tree/master/enformer)
