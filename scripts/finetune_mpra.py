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

8. Evaluate validation/test multiple times per epoch:
   python scripts/finetune_mpra.py \
       --val_eval_frequency 3 \
       --test_eval_frequency 2

9. Two-stage training (unfreeze encoder in Stage 2):
   python scripts/finetune_mpra.py \
       --num_epochs 50 \
       --learning_rate 1e-3 \
       --second_stage_lr 1e-5
   Note: When two-stage training is enabled, Stage 1 automatically saves the full model
   (not just the head) to enable Stage 2 to properly load and unfreeze the encoder.

10. Resume from Stage 2 (skip Stage 1, requires Stage 1 checkpoint):
   python scripts/finetune_mpra.py \
       --num_epochs 50 \
       --second_stage_lr 1e-5 \
       --resume_from_stage2

11. Use cached embeddings for faster training (Stage 1 only, no augmentations):
   # First, pre-compute embeddings:
   python scripts/cache_embeddings.py --cell_type HepG2 --split train
   python scripts/cache_embeddings.py --cell_type HepG2 --split val
   python scripts/cache_embeddings.py --cell_type HepG2 --split test
   
   # Then train with cached embeddings:
   python scripts/finetune_mpra.py \
       --cell_type HepG2 \
       --use_cached_embeddings \
       --cache_file ./.cache/embeddings/HepG2_train_embeddings.pkl \
       --num_epochs 100 \
       --learning_rate 1e-3
   Note: Cached embeddings disable augmentations and two-stage training.
   This is ideal for hyperparameter sweeps where you want fast iteration.
"""

import argparse
import json
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


PROMOTER_CONSTRUCT_LENGTH = 281


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def apply_config_to_args(args, config: dict):
    """Apply config values to args namespace.
    
    Note: Command-line arguments will override config values since they're parsed after.
    This function applies config as defaults that can be overridden.
    """
    # Apply top-level config
    if 'cell_type' in config:
        args.cell_type = config['cell_type']
    
    # Apply data config
    if 'data' in config:
        data_config = config['data']
        args.batch_size = data_config.get('batch_size', args.batch_size)
        args.random_shift = data_config.get('random_shift', args.random_shift)
        args.random_shift_likelihood = data_config.get('random_shift_likelihood', args.random_shift_likelihood)
        args.reverse_complement = data_config.get('reverse_complement', args.reverse_complement)
        args.pad_n_bases = data_config.get('pad_n_bases', args.pad_n_bases)
    
    # Apply model config
    if 'model' in config:
        model_config = config['model']
        args.center_bp = model_config.get('center_bp', args.center_bp)
        args.pooling_type = model_config.get('pooling_type', args.pooling_type)
        args.nl_size = model_config.get('nl_size', args.nl_size)
        args.do = model_config.get('do', args.do)
        args.activation = model_config.get('activation', args.activation)
    
    # Apply training config
    if 'training' in config:
        train_config = config['training']
        args.num_epochs = train_config.get('num_epochs', args.num_epochs)
        args.learning_rate = train_config.get('learning_rate', args.learning_rate)
        args.optimizer = train_config.get('optimizer', args.optimizer)
        args.weight_decay = train_config.get('weight_decay', args.weight_decay)
        args.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', args.gradient_accumulation_steps)
        args.gradient_clip = train_config.get('gradient_clip', args.gradient_clip)
        args.lr_scheduler = train_config.get('lr_scheduler', args.lr_scheduler)
        args.no_val_split = train_config.get('no_val_split', args.no_val_split)
        args.early_stopping_patience = train_config.get('early_stopping_patience', args.early_stopping_patience)
        args.val_eval_frequency = train_config.get('val_eval_frequency', args.val_eval_frequency)
        args.test_eval_frequency = train_config.get('test_eval_frequency', args.test_eval_frequency)
    
    # Apply two-stage config
    if 'two_stage' in config:
        two_stage_config = config['two_stage']
        if two_stage_config.get('enabled', False):
            args.second_stage_lr = two_stage_config.get('second_stage_lr', args.second_stage_lr)
            args.second_stage_epochs = two_stage_config.get('second_stage_epochs', args.second_stage_epochs)
    
    # Apply cached embeddings config
    if 'cached_embeddings' in config:
        cache_config = config['cached_embeddings']
        args.use_cached_embeddings = cache_config.get('use_cached_embeddings', args.use_cached_embeddings)
        args.cache_file = cache_config.get('cache_file', args.cache_file)
    
    # Apply checkpointing config
    if 'checkpointing' in config:
        checkpoint_config = config['checkpointing']
        args.checkpoint_dir = checkpoint_config.get('checkpoint_dir', args.checkpoint_dir)
        args.save_full_model = checkpoint_config.get('save_full_model', args.save_full_model)
    
    # Apply wandb config
    if 'wandb' in config:
        wandb_config = config['wandb']
        args.no_wandb = not wandb_config.get('enabled', True)
        args.wandb_project = wandb_config.get('project', args.wandb_project)
        args.wandb_name = wandb_config.get('wandb_name', args.wandb_name)
    
    # Apply base checkpoint path
    if 'base_checkpoint_path' in config:
        args.base_checkpoint_path = config.get('base_checkpoint_path', args.base_checkpoint_path)

def main():
    parser = argparse.ArgumentParser(
        description='Finetune AlphaGenome with MPRA head on LentiMPRA dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file (load first, then other args can override)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file with hyperparameters. '
             'Command-line arguments override config file values.'
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
        '--random_shift_likelihood',
        type=float,
        default=0.5,
        help='Likelihood of applying random shifts to training data'
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
        choices=['mean', 'sum', 'max', 'center', 'flatten'],
        help='Pooling type for MPRA head: sum/mean/max (pool center window), center (single position), flatten (all positions)'
    )
    parser.add_argument(
        '--pad_n_bases',
        type=int,
        default=0,
        help='Number of bases to pad on each side of the sequence'
    )
    parser.add_argument(
        '--no_freeze_backbone',
        action='store_true',
        help='Do not freeze backbone (train full model, not just head)'
    )
    parser.add_argument(
        '--nl_size',
        type=str,
        default='1024',
        help='Hidden layer sizes: single int (e.g., "1024") or comma-separated list (e.g., "512,256") for multiple layers'
    )
    parser.add_argument(
        '--do',
        type=float,
        default=None,
        help='Dropout rate for the MLP - None means no dropout'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        choices=['relu', 'gelu'],
        help='Activation function: relu or gelu'
    )
    parser.add_argument(
        '--base_checkpoint_path',
        type=str,
        default=None,
        help='Optional local AlphaGenome checkpoint directory. '
             'If provided, the base model will be loaded from this path '
             'instead of using Kaggle.'
    )
    
    # Cached embeddings (for faster training)
    parser.add_argument(
        '--use_cached_embeddings',
        action='store_true',
        help='Use pre-computed cached encoder embeddings (requires cache file). '
             'When enabled, augmentations are automatically disabled.'
    )
    parser.add_argument(
        '--cache_file',
        type=str,
        default=None,
        help='Path to cached embeddings file (required if use_cached_embeddings=True). '
             'Should be created using scripts/cache_embeddings.py'
    )
    
    # Training parameters
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
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
    parser.add_argument(
        '--gradient_clip',
        type=float,
        default=None,
        help='Gradient clipping value. If set, gradients are clipped to this maximum norm (e.g., 1.0 or 5.0). Default: None (no clipping)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'adamw'],
        help='Optimizer type: adam or adamw'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=None,
        help='Weight decay (L2 regularization) for Adam(W) optimizer (e.g., 1e-6)'
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default=None,
        choices=['plateau', 'cosine'],
        help='Learning rate scheduler: plateau (reduce on plateau) or cosine (cosine annealing)'
    )
    parser.add_argument(
        '--no_val_split',
        action='store_true',
        help='Put all data in training set (no validation split). Use for cosine scheduler without validation.'
    )
    parser.add_argument(
        '--save_test_results',
        type=str,
        default=None,
        help='Path to CSV file to save test set performance results for benchmarking'
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
        help='Save full model including all backbone layers (encoder + transformer + decoder + head). '
             'By default, saves minimal model (encoder + head). Use --no-save_minimal_model to save head only.'
    )
    parser.add_argument(
        '--save_minimal_model',
        action='store_true',
        help='Save minimal model (encoder + custom head only, skips transformer/decoder). '
             'This is the default behavior. Use --save_full_model to save full model, or '
             'use --no-save_minimal_model to save head only (no alphagenome layers).'
    )
    parser.add_argument(
        '--no-save_minimal_model',
        dest='save_minimal_model',
        action='store_false',
        help='Disable minimal model saving (saves head only, no alphagenome layers). '
             'Use with --save_full_model to save full model instead.'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5,
        help='Number of epochs without improvement before early stopping'
    )
    parser.add_argument(
        '--val_eval_frequency',
        type=int,
        default=4,
        help='Number of times to evaluate on validation set per epoch (default: 1, at end of epoch)'
    )
    parser.add_argument(
        '--test_eval_frequency',
        type=int,
        default=4,
        help='Number of times to evaluate on test set per epoch (default: 1, at end of epoch)'
    )
    parser.add_argument(
        '--second_stage_lr',
        type=float,
        default=None,
        help='Learning rate for Stage 2 (unfreeze encoder). If provided, enables two-stage training: '
             'Stage 1 trains head only, then loads best checkpoint, unfreezes encoder, and continues '
             'training for second_stage_epochs with this learning rate.'
    )
    parser.add_argument(
        '--second_stage_epochs',
        type=int,
        default=50,
        help='Number of epochs for Stage 2 training (default: 50). Only used when second_stage_lr is provided.'
    )
    parser.add_argument(
        '--resume_from_stage2',
        action='store_true',
        help='Skip Stage 1 training and resume directly from Stage 2. Requires checkpoint_dir to contain '
             'a Stage 1 checkpoint. second_stage_lr must be provided.'
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
    
    # Two-pass parsing: first get config path, then apply config as defaults
    # This allows config to provide defaults that command-line args can override
    temp_args, _ = parser.parse_known_args()
    
    # Load config if provided
    config = None
    if temp_args.config:
        print(f"Loading config from: {temp_args.config}")
        config = load_config(temp_args.config)
        print("✓ Config loaded")
    
    # Apply config values as new defaults for parser
    if config:
        # Update parser defaults with config values
        if 'cell_type' in config:
            parser.set_defaults(cell_type=config['cell_type'])
        
        if 'data' in config:
            data_config = config['data']
            parser.set_defaults(
                batch_size=data_config.get('batch_size', 32),
                random_shift=data_config.get('random_shift', True),
                random_shift_likelihood=data_config.get('random_shift_likelihood', 0.5),
                reverse_complement=data_config.get('reverse_complement', True),
                pad_n_bases=data_config.get('pad_n_bases', 0),
            )
        
        if 'model' in config:
            model_config = config['model']
            parser.set_defaults(
                center_bp=model_config.get('center_bp', 256),
                pooling_type=model_config.get('pooling_type', 'sum'),
                nl_size=model_config.get('nl_size', '1024'),
                do=model_config.get('do', None),
                activation=model_config.get('activation', 'relu'),
            )
        
        if 'training' in config:
            train_config = config['training']
            parser.set_defaults(
                num_epochs=train_config.get('num_epochs', 100),
                learning_rate=train_config.get('learning_rate', 1e-3),
                optimizer=train_config.get('optimizer', 'adam'),
                weight_decay=train_config.get('weight_decay', None),
                gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
                gradient_clip=train_config.get('gradient_clip', None),
                lr_scheduler=train_config.get('lr_scheduler', None),
                no_val_split=train_config.get('no_val_split', False),
                early_stopping_patience=train_config.get('early_stopping_patience', 5),
                val_eval_frequency=train_config.get('val_eval_frequency', 4),
                test_eval_frequency=train_config.get('test_eval_frequency', 4),
            )
        
        if 'two_stage' in config:
            two_stage_config = config['two_stage']
            if two_stage_config.get('enabled', False):
                parser.set_defaults(
                    second_stage_lr=two_stage_config.get('second_stage_lr', None),
                    second_stage_epochs=two_stage_config.get('second_stage_epochs', 50),
                )
        
        if 'cached_embeddings' in config:
            cache_config = config['cached_embeddings']
            parser.set_defaults(
                use_cached_embeddings=cache_config.get('use_cached_embeddings', False),
                cache_file=cache_config.get('cache_file', None),
            )
        
        if 'checkpointing' in config:
            checkpoint_config = config['checkpointing']
            parser.set_defaults(
                checkpoint_dir=checkpoint_config.get('checkpoint_dir', './results/models/checkpoints/'),
                save_full_model=checkpoint_config.get('save_full_model', False),
            )
        
        if 'wandb' in config:
            wandb_config = config['wandb']
            parser.set_defaults(
                no_wandb=not wandb_config.get('enabled', True),
                wandb_project=wandb_config.get('project', 'alphagenome-mpra'),
                wandb_name=wandb_config.get('wandb_name', 'mpra-head-encoder'),
            )
        
        if 'base_checkpoint_path' in config:
            parser.set_defaults(base_checkpoint_path=config.get('base_checkpoint_path', None))
    
    # Now parse with updated defaults (command-line args will override config)
    import sys
    args = parser.parse_args()
    
    # Set default save mode: minimal model (encoder + head) unless explicitly overridden
    # Check if flags were explicitly set by checking sys.argv
    save_full_model_set = '--save_full_model' in sys.argv
    no_save_minimal_set = '--no-save_minimal_model' in sys.argv
    
    if save_full_model_set:
        # User explicitly wants full model
        args.save_minimal_model = False
    elif no_save_minimal_set:
        # User explicitly wants head only
        args.save_minimal_model = False
    else:
        # Default to minimal model if neither flag was explicitly set
        args.save_minimal_model = True
    
    # Construct full checkpoint path from base dir, cell type, and run name
    from pathlib import Path
    checkpoint_path = (Path(args.checkpoint_dir) / args.cell_type / args.wandb_name).resolve()
    
    # Print configuration
    print("=" * 80)
    print("AlphaGenome MPRA Fine-tuning")
    print("=" * 80)
    print(f"Cell type:                  {args.cell_type}")
    print(f"Batch size:                 {args.batch_size}")
    print(f"Number of epochs:           {args.num_epochs}")
    print(f"Learning rate:              {args.learning_rate}")
    print(f"Optimizer:                  {args.optimizer}")
    print(f"Weight decay:               {args.weight_decay if args.weight_decay else 'None'}")
    print(f"Center bp:                  {args.center_bp}")
    print(f"Pooling type:               {args.pooling_type}")
    print(f"Freeze backbone:            {not args.no_freeze_backbone}")
    print(f"Gradient accumulation:      {args.gradient_accumulation_steps}")
    print(f"Gradient clipping:          {args.gradient_clip if args.gradient_clip else 'None'}")
    print(f"Early stopping patience:    {args.early_stopping_patience}")
    print(f"Val eval frequency:         {args.val_eval_frequency} per epoch")
    print(f"Test eval frequency:        {args.test_eval_frequency} per epoch")
    if args.second_stage_lr:
        print(f"Two-stage training:         Enabled")
        if args.resume_from_stage2:
            print(f"Resume from Stage 2:        Yes (skipping Stage 1)")
        else:
            print(f"Stage 1 epochs:             {args.num_epochs}")
        print(f"Stage 2 epochs:             {args.second_stage_epochs}")
        print(f"Stage 2 learning rate:      {args.second_stage_lr}")
    else:
        print(f"Two-stage training:         Disabled")
    print(f"Checkpoint path:            {checkpoint_path}")
    if args.save_minimal_model:
        print(f"Save model:                 Minimal (encoder + heads only)")
    elif args.save_full_model:
        print(f"Save model:                 Full model")
    else:
        print(f"Save model:                 Head only")
    print(f"Use W&B:                    {not args.no_wandb}")
    if args.base_checkpoint_path is not None:
        print(f"Base AlphaGenome checkpoint:{args.base_checkpoint_path}")
    if not args.no_wandb:
        print(f"W&B project:                {args.wandb_project}")
        print(f"W&B run name:               {args.wandb_name}")
    print("=" * 80)
    print()
    
    # Parse nl_size (can be single int or comma-separated list)
    try:
        if ',' in args.nl_size:
            nl_size = [int(x.strip()) for x in args.nl_size.split(',')]
        else:
            nl_size = int(args.nl_size)
    except ValueError:
        raise ValueError(f"Invalid nl_size format: {args.nl_size}. Use single int (e.g., '1024') or comma-separated list (e.g., '512,256')")
    
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
                'pooling_type': args.pooling_type,
                'nl_size': nl_size,
                'do': args.do,
                'activation': args.activation,
            }
        )
    )
    print("✓ Custom head registered")
    
    # For flatten pooling with cached embeddings, we need to know the embedding length
    # BEFORE initializing the model (to set correct weight matrix sizes)
    init_seq_len = None
    if args.use_cached_embeddings and args.pooling_type == 'flatten':
        import pickle
        print(f"\nLoading cache header to get embedding dimensions for flatten pooling...")
        with open(args.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        # Get max embedding length from cache
        sample_embedding = next(iter(cache_data.values()))
        max_emb_len = sample_embedding.shape[0]  # encoder positions
        # Convert to sequence length (encoder has 128bp resolution)
        init_seq_len = max_emb_len * 128
        print(f"  Embedding length: {max_emb_len} positions → init_seq_len={init_seq_len} bp")
        del cache_data  # Free memory
        
    if not args.use_cached_embeddings:
        init_seq_len = PROMOTER_CONSTRUCT_LENGTH #full promoter construct length
        if args.pad_n_bases > 0:
            init_seq_len += args.pad_n_bases
        print(f"\nUsing sequence length: {init_seq_len} bp")
    
    # Create model
    print("\nCreating model with custom heads...")
    model_with_custom = create_model_with_custom_heads(
        'all_folds',
        custom_heads=['mpra_head'],
        checkpoint_path=args.base_checkpoint_path,
        use_encoder_output=True,
        init_seq_len=init_seq_len
    )
    print("✓ Model created")
    
    # Freeze backbone if requested
    if not args.no_freeze_backbone:
        print("\nFreezing backbone (training head only)...")
        model_with_custom.freeze_except_head('mpra_head')
        print("✓ Backbone frozen")
    else:
        print("\nTraining full model (backbone + head)...")
    
    # Save mode already determined above after parsing
    
    # Validate cached embeddings setup
    if args.use_cached_embeddings:
        if args.cache_file is None:
            raise ValueError("--cache_file must be provided when --use_cached_embeddings is enabled")
        if args.second_stage_lr is not None:
            raise ValueError("Cached embeddings cannot be used with two-stage training (--second_stage_lr). "
                           "Stage 2 requires training the encoder, which needs sequences, not cached embeddings.")
        print(f"\nUsing cached embeddings from: {args.cache_file}")
        print("  Note: Augmentations are automatically disabled when using cached embeddings")
        # When using cached embeddings, default to saving full model for easier downstream use
        if not args.save_full_model and not args.save_minimal_model:
            print("  Note: Setting save_full_model=True for cached embeddings mode (recommended for downstream use)")
            args.save_full_model = True
    
    # Create datasets
    print(f"\nLoading datasets (cell_type={args.cell_type})...")
    train_dataset = LentiMPRADataset(
        model=model_with_custom,
        cell_type=args.cell_type,
        split='train',
        random_shift=args.random_shift if not args.use_cached_embeddings else False,
        random_shift_likelihood=args.random_shift_likelihood,
        reverse_complement=args.reverse_complement if not args.use_cached_embeddings else False,
        use_cached_embeddings=args.use_cached_embeddings,
        cache_file=args.cache_file if args.use_cached_embeddings else None,
        pad_n_bases=args.pad_n_bases
    )
    # Determine cache files for val and test splits
    val_cache_file = None
    test_cache_file = None
    if args.use_cached_embeddings and args.cache_file:
        from pathlib import Path
        cache_path = Path(args.cache_file)
        # Assume cache file naming: {cell_type}_{split}_embeddings.pkl
        val_cache_file = str(cache_path.parent / f"{args.cell_type}_val_embeddings.pkl")
        test_cache_file = str(cache_path.parent / f"{args.cell_type}_test_embeddings.pkl")
    
    # Handle no validation split case (merge train and val)
    if args.no_val_split:
        print("  Note: Merging train and val splits (no_val_split=True)")
        val_dataset = None
    else:
        val_dataset = LentiMPRADataset(
            model=model_with_custom,
            cell_type=args.cell_type,
            split='val',
            random_shift=False,
            reverse_complement=False,
            use_cached_embeddings=args.use_cached_embeddings,
            cache_file=val_cache_file,
            pad_n_bases=args.pad_n_bases
        )
    
    test_dataset = LentiMPRADataset(
        model=model_with_custom,
        cell_type=args.cell_type,
        split='test',
        random_shift=False,
        reverse_complement=False,
        use_cached_embeddings=args.use_cached_embeddings,
        cache_file=test_cache_file,
        pad_n_bases=args.pad_n_bases
    )
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"✓ Val dataset:   {len(val_dataset)} samples")
    else:
        print(f"✓ Val dataset:   None (no_val_split=True)")
    print(f"✓ Test dataset:  {len(test_dataset)} samples")
    
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
    ) if val_dataset else None
    test_loader = MPRADataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print(f"✓ Train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"✓ Val batches:   {len(val_loader)}")
    else:
        print(f"✓ Val batches:   None (no validation split)")
    print(f"✓ Test batches:  {len(test_loader)}")
    
    # Train model
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    # Prepare wandb config with hyperparameters
    wandb_config_dict = {
        'cell_type': args.cell_type,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'pooling_type': args.pooling_type,
        'nl_size': args.nl_size,
        'activation': args.activation,
        'dropout': args.do,
        'lr_scheduler': args.lr_scheduler,
        'no_val_split': args.no_val_split,
    }
    
    history = train(
        model_with_custom,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=str(checkpoint_path),
        save_full_model=args.save_full_model,
        save_minimal_model=args.save_minimal_model,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip=args.gradient_clip,
        val_eval_frequency=args.val_eval_frequency,
        test_eval_frequency=args.test_eval_frequency,
        second_stage_lr=args.second_stage_lr,
        second_stage_epochs=args.second_stage_epochs,
        resume_from_stage2=args.resume_from_stage2,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_config=wandb_config_dict,
        use_cached_embeddings=args.use_cached_embeddings,
        lr_scheduler=args.lr_scheduler,
        save_test_results=args.save_test_results,
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
    if history['test_loss']:
        print(f"Final test loss:      {history['test_loss'][-1]:.6f}")
        print(f"Final test pearson:   {history['test_pearson'][-1]:.4f}")
        print(f"Best test loss:       {min(history['test_loss']):.6f}")
    print("=" * 80)
    
    return history


if __name__ == '__main__':
    main()
