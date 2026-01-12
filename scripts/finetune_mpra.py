"""
Finetune AlphaGenome with MPRA head on LentiMPRA dataset from 
[Agarwal et al., 2025](https://www.nature.com/articles/s41586-024-08430-9)
"""



import jax
import jax.numpy as jnp
from alphagenome.models import dna_output
from alphagenome.data import genome
# Import the finetuning extensions
from alphagenome_ft import (
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)
from alphagenome_research.model import dna_model
from src import EncoderMPRAHead, LentiMPRADataset, MPRADataLoader, train


print("Registering custom MPRA head...")
register_custom_head(
    'mpra_head',
    EncoderMPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='mpra_head',
        output_type=dna_output.OutputType.RNA_SEQ,  # Reuse RNA_SEQ output type
        num_tracks=1,  # Predict single value per position
    )
)

model_with_custom = create_model_with_custom_heads(
    'all_folds',
    custom_heads=['mpra_head'],
    use_encoder_output=True  # Enable encoder output for EncoderMPRAHead
)

#freeze everything except the mpra head
model_with_custom.freeze_except_head('mpra_head')


#datasets        
train_dataset = LentiMPRADataset(model=model_with_custom, cell_type='HepG2', split='train', random_shift=True, reverse_complement=True, subset_frac=0.01)
val_dataset = LentiMPRADataset(model=model_with_custom, cell_type='HepG2', split='val', random_shift=False, reverse_complement=False, subset_frac=0.01)
#loaders
train_loader = MPRADataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = MPRADataLoader(val_dataset, batch_size=32, shuffle=False)

# training
# Standard training
history = train(model_with_custom, train_loader, val_loader, num_epochs=3, learning_rate=1e-4,use_wandb=True,wandb_project="alphagenome-mpra",wandb_name="mpra-head-encoder")
print("history",history)
#
# # Training with gradient accumulation (reduces memory usage)
# # Processes batches in 4 mini-batches, using 1/4 the GPU memory
history = train(
    model_with_custom, train_loader, val_loader,
    num_epochs=3, learning_rate=1e-4,
    gradient_accumulation_steps=16,
    use_wandb=True,
    wandb_project="alphagenome-mpra",
    wandb_name="mpra-head-encoder-gradient-accumulation-16",
)
print("history",history)