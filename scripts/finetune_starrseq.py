"""
Finetune AlphaGenome with DeepSTARR head on DeepSTARR dataset from
[de Almeida et al., 2022](https://www.nature.com/articles/s41588-022-01048-5)

Compare performance against DeepSTARR (Pearson R Developmental enhancers - 0.656, PCC Housekeeping enhancers - 0.736)
and Dream-RNN (Pearson R Developmental enhancers - 0.665, PCC Housekeeping enhancers - 0.746)
DeepSTARR values from downlaoding weights and running test set. 
DREAM-RNN values are from [DEGU](https://www.nature.com/articles/s44387-025-00053-3) Supp Fig.2a-b (mean of ensemble)

DeepSTARR predicts two types of enhancer activity:
- Developmental enhancer activity (Dev_log2_enrichment)
- Housekeeping enhancer activity (Hk_log2_enrichment)

USAGE EXAMPLES:

1. Basic training (default parameters):
   python scripts/finetune_starrseq.py

2. Custom hyperparameters:
   python scripts/finetune_starrseq.py \
       --num_epochs 20 \
       --learning_rate 1e-4 \
       --batch_size 64

3. Train without Weights & Biases logging:
   python scripts/finetune_starrseq.py --no_wandb

4. Train with gradient accumulation (reduces memory):
   python scripts/finetune_starrseq.py \
       --batch_size 32 \
       --gradient_accumulation_steps 4

5. Custom checkpoint directory and early stopping:
   python scripts/finetune_starrseq.py \
       --checkpoint_dir ./my_checkpoints/deepstarr_model \
       --early_stopping_patience 10

6. Full model training (unfreeze backbone):
   python scripts/finetune_starrseq.py \
       --no_freeze_backbone \
       --save_full_model \
       --learning_rate 1e-5

7. Evaluate validation/test multiple times per epoch:
   python scripts/finetune_starrseq.py \
       --val_eval_frequency 3 \
       --test_eval_frequency 2

8. Two-stage training (unfreeze encoder in Stage 2):
   python scripts/finetune_starrseq.py \
       --num_epochs 50 \
       --learning_rate 1e-3 \
       --second_stage_lr 1e-5

9. Resume from Stage 2 (skip Stage 1, requires Stage 1 checkpoint):
   python scripts/finetune_starrseq.py \
       --num_epochs 50 \
       --second_stage_lr 1e-5 \
       --resume_from_stage2

10. Use cached embeddings for faster training (Stage 1 only, no augmentations):
    # First, pre-compute embeddings:
    python scripts/cache_embeddings.py --dataset deepstarr --split train
    python scripts/cache_embeddings.py --dataset deepstarr --split val
    python scripts/cache_embeddings.py --dataset deepstarr --split test
    
    # Then train with cached embeddings:
    python scripts/finetune_starrseq.py \
        --use_cached_embeddings \
        --cache_file ./.cache/embeddings/deepstarr_train_embeddings.pkl \
        --num_epochs 100 \
        --learning_rate 1e-3

11. Use a local base AlphaGenome checkpoint instead of Kaggle:
    python scripts/finetune_starrseq.py \
        --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
        --base_checkpoint_path /home/USERNAME/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1/
"""

import argparse
import json
import jax
import jax.numpy as jnp
from alphagenome.models import dna_output
from alphagenome.data import genome
from alphagenome_research.model import dna_model
from pathlib import Path
# Import the finetuning extensions
from alphagenome_ft import (
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)
from src import DeepSTARRHead, DeepSTARRDataset, STARRSeqDataLoader, train


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Finetune AlphaGenome with DeepSTARR head on DeepSTARR dataset',
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
        '--data_path',
        type=str,
        default='./data/deepstarr',
        help='Path to DeepSTARR data directory'
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
        help='Apply random shifts to training data (augmentation) - We have the StarrSeq adaptors making this possible'
    )
    parser.add_argument(
        '--random_shift_likelihood',
        type=float,
        default=0.5,
        help='Likelihood of applying random shifts to training data'
    )
    parser.add_argument(
        '--max_shift',
        type=int,
        default=25,
        help='Maximum shift amount in base pairs'
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
        help='Number of base pairs to pool from center (256=2 positions)'
    )
    parser.add_argument(
        '--pooling_type',
        type=str,
        default='flatten',
        choices=['mean', 'sum', 'max', 'center', 'flatten'],
        help='Pooling type for head: sum/mean/max (pool center window), center (single position), flatten (all positions)'
    )
    parser.add_argument(
        '--no_freeze_backbone',
        action='store_true',
        help='Do not freeze backbone (train full model, not just head)'
    )
    parser.add_argument(
        '--nl_size',
        type=str,
        default='512,512',
        help='Hidden layer sizes: single int (e.g., "1024") or comma-separated list (e.g., "512,256") for multiple layers'
    )
    parser.add_argument(
        '--do',
        type=float,
        default=0.5,
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
        default=1,
        help='Number of epochs without improvement before early stopping'
    )
    parser.add_argument(
        '--val_eval_frequency',
        type=int,
        default=20,
        help='Number of times to evaluate on validation set per epoch (default: 20)'
    )
    parser.add_argument(
        '--test_eval_frequency',
        type=int,
        default=20,
        help='Number of times to evaluate on test set per epoch (default: 20)'
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
        default='alphagenome-deepstarr',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_name',
        type=str,
        default='deepstarr-head-encoder',
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
        if 'data' in config:
            data_config = config['data']
            parser.set_defaults(
                data_path=data_config.get('data_path', './data/deepstarr'),
                batch_size=data_config.get('batch_size', 32),
                random_shift=data_config.get('random_shift', True),
                random_shift_likelihood=data_config.get('random_shift_likelihood', 0.5),
                max_shift=data_config.get('max_shift', 25),
                reverse_complement=data_config.get('reverse_complement', True),
            )
        
        if 'model' in config:
            model_config = config['model']
            parser.set_defaults(
                center_bp=model_config.get('center_bp', 256),
                pooling_type=model_config.get('pooling_type', 'flatten'),
                nl_size=model_config.get('nl_size', '512,512'),
                do=model_config.get('do', 0.5),
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
                early_stopping_patience=train_config.get('early_stopping_patience', 1),
                val_eval_frequency=train_config.get('val_eval_frequency', 20),
                test_eval_frequency=train_config.get('test_eval_frequency', 20),
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
                wandb_project=wandb_config.get('project', 'alphagenome-deepstarr'),
                wandb_name=wandb_config.get('wandb_name', 'deepstarr-head-encoder'),
            )
        
        if 'base_checkpoint_path' in config:
            parser.set_defaults(base_checkpoint_path=config.get('base_checkpoint_path', None))
    
    # Now parse with updated defaults (command-line args will override config)
    args = parser.parse_args()
    
    # Construct full checkpoint path from base dir and run name
    from pathlib import Path
    checkpoint_path = (Path(args.checkpoint_dir) / "deepstarr" / args.wandb_name).resolve()
    
    # Print configuration
    print("=" * 80)
    print("AlphaGenome DeepSTARR Fine-tuning")
    print("=" * 80)
    print(f"Data path:                  {args.data_path}")
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
    
    # Register custom DeepSTARR head (2 outputs: developmental and housekeeping)
    print("Registering custom DeepSTARR head...")
    register_custom_head(
        'deepstarr_head',
        DeepSTARRHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='deepstarr_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=2,  # Two outputs: developmental and housekeeping
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
    # For non-cached mode with flatten, use the actual sequence length (249bp sequence +51bp adaptors = 300bp for DeepSTARR)
    init_seq_len = None
    if args.pooling_type == 'flatten':
        if args.use_cached_embeddings:
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
        else:
            # For non-cached mode, use the actual sequence length (249bp for DeepSTARR)
            init_seq_len = 249+51
            print(f"\nUsing flatten pooling with sequence length: {init_seq_len} bp")
    
    # Create model
    print("\nCreating model with custom heads...")
    model_with_custom = create_model_with_custom_heads(
        'all_folds',
        custom_heads=['deepstarr_head'],
        checkpoint_path=args.base_checkpoint_path,
        use_encoder_output=True,
        init_seq_len=init_seq_len
    )
    print("✓ Model created")
    
    # Freeze backbone if requested
    if not args.no_freeze_backbone:
        print("\nFreezing backbone (training head only)...")
        model_with_custom.freeze_except_head('deepstarr_head')
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
        if not args.save_full_model and not args.save_minimal_model:
            print("  Note: Setting save_full_model=True for cached embeddings mode (recommended for downstream use)")
            args.save_full_model = True
    
    # Create datasets
    print(f"\nLoading DeepSTARR datasets...")
    train_dataset = DeepSTARRDataset(
        model=model_with_custom,
        path_to_data=args.data_path,
        split='train',
        organism=dna_model.Organism.HOMO_SAPIENS, # DROSOPHILA_MELANOGASTER for DeepSTARR so we use HOMO_SAPIENS for now, could test mus musculus later
        random_shift=args.random_shift if not args.use_cached_embeddings else False,
        random_shift_likelihood=args.random_shift_likelihood,
        max_shift=args.max_shift,
        reverse_complement=args.reverse_complement if not args.use_cached_embeddings else False,
        use_cached_embeddings=args.use_cached_embeddings,
        cache_file=args.cache_file if args.use_cached_embeddings else None,
    )
    
    # Determine cache files for val and test splits
    val_cache_file = None
    test_cache_file = None
    if args.use_cached_embeddings and args.cache_file:
        from pathlib import Path
        cache_path = Path(args.cache_file)
        # Assume cache file naming: deepstarr_{split}_embeddings.pkl
        val_cache_file = str(cache_path.parent / "deepstarr_val_embeddings.pkl")
        test_cache_file = str(cache_path.parent / "deepstarr_test_embeddings.pkl")
    
    # Handle no validation split case (merge train and val)
    if args.no_val_split:
        print("  Note: Merging train and val splits (no_val_split=True)")
        val_dataset = None
    else:
        val_dataset = DeepSTARRDataset(
            model=model_with_custom,
            path_to_data=args.data_path,
            split='val',
            organism=dna_model.Organism.HOMO_SAPIENS, #dna_model.Organism.DROSOPHILA_MELANOGASTER,
            random_shift=False,
            reverse_complement=False,
            use_cached_embeddings=args.use_cached_embeddings,
            cache_file=val_cache_file,
        )
    
    test_dataset = DeepSTARRDataset(
        model=model_with_custom,
        path_to_data=args.data_path,
        split='test',
        organism=dna_model.Organism.HOMO_SAPIENS, #dna_model.Organism.DROSOPHILA_MELANOGASTER,
        random_shift=False,
        reverse_complement=False,
        use_cached_embeddings=args.use_cached_embeddings,
        cache_file=test_cache_file,
    )
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"✓ Val dataset:   {len(val_dataset)} samples")
    else:
        print(f"✓ Val dataset:   None (no_val_split=True)")
    print(f"✓ Test dataset:   {len(test_dataset)} samples")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = STARRSeqDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = STARRSeqDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    ) if val_dataset else None
    test_loader = STARRSeqDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader) if val_loader else 'None'}")
    print(f"✓ Test batches:   {len(test_loader)}")
    
    # Train model
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    # Prepare wandb config with hyperparameters
    wandb_config_dict = {
        'dataset': 'deepstarr',
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
        head_name='deepstarr_head',
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
