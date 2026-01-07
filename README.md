# AlphaGenome Finetuning - Applied to MPRA data

This repository provides utilities for finetuning the AlphaGenome model with custom prediction heads and parameter freezing capabilities, **without modifying the original `alphagenome_research` codebase**.

Downstream applications will be fine-tuning on lentiMPRA data.

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

The `alphagenome_ft` code is included in this repository. Make sure you have [`alphagenome_research`](https://github.com/google-deepmind/alphagenome_research/) installed to use it.

OPTION A: create_model_with_custom_heads()
  * Creates model with custom heads ONLY
  * Use for finetuning on your specific task
  * Replaces standard heads with your custom ones
  * Keeps pretrained backbone

OPTION B: wrap_pretrained_model()
  * Wraps existing pretrained model
  * Keeps all standard heads
  * Adds parameter management methods
  * Can also add custom heads to other heads with add_custom_heads_to_model()

Both options provide:
  • freeze_parameters(paths, prefixes)
  • unfreeze_parameters(paths, prefixes)
  • freeze_backbone()
  • freeze_all_heads(except_heads)
  • freeze_except_head(head_name)
  • get_parameter_paths()
  • get_head_parameter_paths()
  • get_backbone_parameter_paths()
  • count_parameters()


## Quick Start

### Option 1: Wrap Existing Model (Parameter Management Only)

Use this when you want to add parameter management to the standard pretrained model:

```python
import jax
from alphagenome_research.model import dna_model
from alphagenome_ft import wrap_pretrained_model

# Load pretrained model
base_model = dna_model.create_from_kaggle('all_folds') #if using CPU - device=jax.devices('cpu')[0]

# Wrap to add finetuning methods
model = wrap_pretrained_model(base_model)

# Now you can use parameter management methods
model.freeze_backbone()  # Freeze encoder/transformer/decoder - just not heads
model.get_parameter_paths()  # List all parameters
print(f"Total parameters: {model.count_parameters():,}")

# Now you can also add a new head if you like
from alphagenome_ft import (
    CustomHead,
    register_custom_head,
    HeadType
)
import haiku as hk
from alphagenome.models import dna_output

class MyMPRAHead(CustomHead):
    def predict(self, embeddings, organism_index, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        return {'predictions': hk.Linear(self._num_tracks)(x)}
    
    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        mse = jnp.mean((predictions['predictions'] - targets) ** 2)
        return {'loss': mse}

register_custom_head(
    'my_mpra',
    MyMPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='my_mpra',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

model = add_custom_heads_to_model(model, custom_heads=['my_mpra'])

```

### Option 2: Create Model with Custom Head

Use this when you want to finetune on a specific task with your own prediction head:

```python
import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome.models import dna_output
from alphagenome_ft import (
    CustomHead,
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)

# 1. Define your custom head
class MyMPRAHead(CustomHead):
    def predict(self, embeddings, organism_index, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        return {'predictions': hk.Linear(self._num_tracks)(x)}
    
    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        mse = jnp.mean((predictions['predictions'] - targets) ** 2)
        return {'loss': mse}

# 2. Register it
register_custom_head(
    'my_mpra',
    MyMPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='my_mpra',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
    )
)

# 3. Create model with custom head
model = create_model_with_custom_heads(
    'all_folds',
    custom_heads=['my_mpra'],
) #if using CPU - device=jax.devices('cpu')[0]

# 4. Freeze everything except your custom head
model.freeze_except_head('my_mpra')

#Train
```

## License

This project extends AlphaGenome. Please refer to the original AlphaGenome license for usage terms.
