# AlphaGenome MPRA Finetuning

This repository demonstrates finetuning AlphaGenome on MPRA (Massively Parallel Reporter Assay) data using the [`alphagenome-ft`](https://github.com/YOUR_USERNAME/alphagenome_ft) package.

The goal is to finetune AlphaGenome to predict reporter activity from genomic sequences, with downstream applications to lentiMPRA data.

## Installation

### Prerequisites

1. Install AlphaGenome Research:
```bash
pip install git+https://github.com/google-deepmind/alphagenome_research.git
```

2. Install this project:
```bash
git clone https://github.com/YOUR_USERNAME/alphagenome_FT_MPRA.git
cd alphagenome_FT_MPRA
pip install -e .
```

This will automatically install the `alphagenome-ft` package as a dependency.

## Architecture

AlphaGenome Model Architecture:

DNA Sequence (B, S, 4)
    ↓
SequenceEncoder (convolutional downsampling)
    ↓
TransformerTower (9 transformer blocks with pairwise attention)
    ↓
SequenceDecoder (convolutional upsampling)
    ↓
Embeddings:
  - embeddings_1bp: (B, S, 1536) - High resolution
  - embeddings_128bp: (B, S//128, 3072) - Low resolution  
  - embeddings_pair: (B, S//2048, S//2048, 128) - Pairwise
    ↓
Heads (task-specific predictions):
  - ATAC, DNASE, RNA_SEQ, etc.
  - YOUR_CUSTOM_HEAD ← Add here
  
1. **Backbone**: Encoder + Transformer + Decoder 
2. **Embeddings**: Multi-resolution representations 
3. **Heads**: Task-specific prediction layers

**Fine-tuning: Custom Heads**: Define and register your own prediction heads for task-specific finetuning.

This project uses the [`alphagenome-ft`](https://github.com/Al-Murphy/alphagenome_ft) package for finetuning utilities. See that repository for full documentation on:
- Creating custom prediction heads
- Parameter freezing and management
- Model wrapping and configuration


## Quick Start



```python
import jax
import jax.numpy as jnp
from alphagenome_research.model import dna_model
from alphagenome_ft import (
    CustomHead,
    HeadConfig,
    HeadType,
    register_custom_head,
    wrap_pretrained_model,
    add_custom_heads_to_model,
)
from src.ft_utils import MPRAHead

# 1. Register custom MPRA head (already defined in src/ft_utils.py)
register_custom_head(
    'mpra_head',
    MPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='mpra_head',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
        metadata={'center_window_size': 128, 'pooling_type': 'mean'}
    )
)

# 2. Load pretrained model and add MPRA head
base_model = dna_model.create_from_kaggle('all_folds')
model = wrap_pretrained_model(base_model)
model = add_custom_heads_to_model(model, custom_heads=['mpra_head'])

# 3. Freeze backbone for finetuning
model.freeze_backbone()

# 4. Train on your MPRA data
# ... your training loop here ...
```

## Project Structure

```
alphagenome_FT_MPRA/
├── src/
│   ├── ft_utils.py          # Custom MPRAHead definition
│   └── __init__.py
├── data/
│   └── legnet_lentimpra/    # MPRA training data
├── test.ipynb               # Example notebook
├── README.md
└── pyproject.toml
```

## License

This project extends AlphaGenome. Please refer to the original AlphaGenome license for usage terms.
