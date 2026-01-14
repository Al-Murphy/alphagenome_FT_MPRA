"""
Finetune AlphaGenome with MPRA head on LentiMPRA dataset from 
[Agarwal et al., 2025](https://www.nature.com/articles/s41586-024-08430-9)

USAGE EXAMPLES:

1. Basic training (default parameters):
   python scripts/finetune_mpra.py

2. Custom hyperparameters:
   python scripts/finetune_mpra.py \
       --num_epochs 20 \
       --learning_rate 1e-4 \
       --batch_size 64

3. Train without Weights & Biases logging:
   python scripts/finetune_mpra.py --no_wandb

4. Train on different cell type:
   python scripts/finetune_mpra.py --cell_type K562

5. Train with gradient accumulation (reduces memory):
   python scripts/finetune_mpra.py \
       --batch_size 32 \
       --gradient_accumulation_steps 4

6. Custom checkpoint directory and early stopping:
   python scripts/finetune_mpra.py \
       --checkpoint_dir ./my_checkpoints/mpra_model \
       --early_stopping_patience 10

7. Full model training (unfreeze backbone):
   python scripts/finetune_mpra.py \
       --no_freeze_backbone \
       --save_full_model \
       --learning_rate 1e-5
"""

import argparse
import jax
import jax.numpy as jnp
from alphagenome.models import dna_output
from alphagenome.data import genome
from pathlib import Path
# Import the finetuning extensions
from alphagenome_ft import (
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)
from src import EncoderMPRAHead, LentiMPRADataset, MPRADataLoader, train


def main():
    parser = argparse.ArgumentParser(
        description='Finetune AlphaGenome with MPRA head on LentiMPRA dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument(
        '--cell_type',
        type=str,
        default='HepG2',
        help='Cell type for MPRA data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training and validation'
    )
    parser.add_argument(
        '--random_shift',
        action='store_true',
        default=True,
        help='Apply random shifts to training data (augmentation)'
    )
    parser.add_argument(
        '--reverse_complement',
        action='store_true',
        default=True,
        help='Apply reverse complement augmentation to training data'
    )
    
    # Model parameters
    parser.add_argument(
        '--center_bp',
        type=int,
        default=256,
        help='Number of base pairs to pool from center (256=2 positions, 384=3 positions)'
    )
    parser.add_argument(
        '--pooling_type',
        type=str,
        default='sum',
        choices=['mean', 'sum', 'max'],
        help='Pooling type for MPRA head'
    )
    parser.add_argument(
        '--no_freeze_backbone',
        action='store_true',
        help='Do not freeze backbone (train full model, not just head)'
    )
    
    # Training parameters
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps (reduces memory usage)'
    )
    
    # Checkpointing and early stopping
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./results/models/checkpoints/',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--save_full_model',
        action='store_true',
        help='Save full model including backbone (otherwise only saves head)'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5,
        help='Number of epochs without improvement before early stopping'
    )
    
    # Weights & Biases logging
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='alphagenome-mpra',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_name',
        type=str,
        default='mpra-head-encoder',
        help='Weights & Biases run name'
    )
    
    args = parser.parse_args()
    
    # Construct full checkpoint path from base dir, cell type, and run name
    from pathlib import Path
    checkpoint_path = Path(args.checkpoint_dir) / args.cell_type / args.wandb_name
    
    # Print configuration
    print("=" * 80)
    print("AlphaGenome MPRA Fine-tuning")
    print("=" * 80)
    print(f"Cell type:                  {args.cell_type}")
    print(f"Batch size:                 {args.batch_size}")
    print(f"Number of epochs:           {args.num_epochs}")
    print(f"Learning rate:              {args.learning_rate}")
    print(f"Center bp:                  {args.center_bp}")
    print(f"Pooling type:               {args.pooling_type}")
    print(f"Freeze backbone:            {not args.no_freeze_backbone}")
    print(f"Gradient accumulation:      {args.gradient_accumulation_steps}")
    print(f"Early stopping patience:    {args.early_stopping_patience}")
    print(f"Checkpoint path:            {checkpoint_path}")
    print(f"Save full model:            {args.save_full_model}")
    print(f"Use W&B:                    {not args.no_wandb}")
    if not args.no_wandb:
        print(f"W&B project:                {args.wandb_project}")
        print(f"W&B run name:               {args.wandb_name}")
    print("=" * 80)
    print()
    
    # Register custom MPRA head
    print("Registering custom MPRA head...")
    register_custom_head(
        'mpra_head',
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='mpra_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata={
                'center_bp': args.center_bp,
                'pooling_type': args.pooling_type
            }
        )
    )
    print("✓ Custom head registered")
    
    # Create model
    print("\nCreating model with custom heads...")
    model_with_custom = create_model_with_custom_heads(
        'all_folds',
        custom_heads=['mpra_head'],
        use_encoder_output=True
    )
    print("✓ Model created")
    
    # Freeze backbone if requested
    if not args.no_freeze_backbone:
        print("\nFreezing backbone (training head only)...")
        model_with_custom.freeze_except_head('mpra_head')
        print("✓ Backbone frozen")
    else:
        print("\nTraining full model (backbone + head)...")
    
    # Create datasets
    print(f"\nLoading datasets (cell_type={args.cell_type})...")
    train_dataset = LentiMPRADataset(
        model=model_with_custom,
        cell_type=args.cell_type,
        split='train',
        random_shift=args.random_shift,
        reverse_complement=args.reverse_complement
    )
    val_dataset = LentiMPRADataset(
        model=model_with_custom,
        cell_type=args.cell_type,
        split='val',
        random_shift=False,
        reverse_complement=False
    )
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset:   {len(val_dataset)} samples")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = MPRADataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = MPRADataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    
    # Train model
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    history = train(
        model_with_custom,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=str(checkpoint_path),
        save_full_model=args.save_full_model,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final train loss:    {history['train_loss'][-1]:.6f}")
    print(f"Final train pearson: {history['train_pearson'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss:      {history['val_loss'][-1]:.6f}")
        print(f"Final val pearson:   {history['val_pearson'][-1]:.4f}")
        print(f"Best val loss:       {min(history['val_loss']):.6f}")
        best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        print(f"Best epoch:          {best_epoch}")
    print("=" * 80)
    
    return history


if __name__ == '__main__':
    main()
