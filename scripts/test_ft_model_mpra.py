"""
Test fine-tuned AlphaGenome MPRA model on test data.

Loads a fine-tuned model checkpoint and evaluates it on the test split of the
LentiMPRA dataset from [Agarwal et al., 2025](https://www.nature.com/articles/s41586-024-08430-9)

USAGE EXAMPLES:

1. Basic usage - test model and print metrics:
   python scripts/test_ft_model_mpra.py --checkpoint_dir ./results/models/checkpoints/mpra_encoder_head

2. Save predictions to custom directory:
   python scripts/test_ft_model_mpra.py \
       --checkpoint_dir ./results/models/checkpoints/mpra_encoder_head \
       --output_dir ./results/test_predictions

3. Test on different cell type with custom batch size:
   python scripts/test_ft_model_mpra.py \
       --checkpoint_dir ./results/models/checkpoints/mpra_encoder_head \
       --cell_type K562 \
       --batch_size 64

4. Use in Python script or notebook:
   from test_ft_model_mpra import get_predictions, compute_metrics
   from alphagenome_ft import load_checkpoint
   from src import LentiMPRADataset, MPRADataLoader
   
   # Load model
   model = load_checkpoint('./results/models/checkpoints/mpra_encoder_head')
   
   # Create test dataset
   test_dataset = LentiMPRADataset(model, cell_type='HepG2', split='test')
   test_loader = MPRADataLoader(test_dataset, batch_size=32, shuffle=False)
   
   # Get predictions
   predictions, actuals = get_predictions(model, test_loader)
   
   # Compute metrics
   metrics = compute_metrics(predictions, actuals)
   print(metrics)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model

# Import the finetuning extensions
from alphagenome_ft import (
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)
from src import EncoderMPRAHead, LentiMPRADataset, MPRADataLoader


def get_predictions_for_saving(
    model,
    dataloader: MPRADataLoader,
    head_name: str = 'mpra_head',
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and actuals for saving to CSV.
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for test batches
        head_name: Name of the custom head
        
    Returns:
        Tuple of (predictions, actuals) as numpy arrays
    """
    all_predictions = []
    all_actuals = []
    
    # Get pooling type from head config
    head_config = getattr(model, "_head_configs", {}).get(head_name, None)
    if head_config is not None:
        metadata = getattr(head_config, "metadata", {}) or {}
        pooling_type = metadata.get("pooling_type", "sum")
    else:
        pooling_type = "sum"
    
    for batch in dataloader:
        with model._device_context:
            predictions = model._predict(
                model._params,
                model._state,
                batch['seq'],
                batch['organism_index'],
                negative_strand_mask=jnp.zeros(len(batch['seq']), dtype=bool),
                strand_reindexing=jax.device_put(
                    model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                    model._device_context._device
                ),
            )
        
        head_predictions = predictions[head_name]
        head_np = np.array(head_predictions)
        seq_len = head_np.shape[1]
        
        # Pool exactly like head's loss() method
        if pooling_type == "flatten":
            pooled = head_np.squeeze(1)
        elif pooling_type == "center":
            center_idx = seq_len // 2
            pooled = head_np[:, center_idx, :]
        else:
            # For mean/max/sum: use center window based on center_bp
            # Get center_bp from metadata (default 256bp = 2 encoder positions at 128bp resolution)
            center_bp = metadata.get("center_bp", 256) if metadata else 256
            center_window_positions = max(1, center_bp // 128)
            window_size = min(center_window_positions, seq_len)
            center_start = max((seq_len - window_size) // 2, 0)
            center_end = center_start + window_size
            center_preds = head_np[:, center_start:center_end, :]
            
            if pooling_type == "mean":
                pooled = center_preds.mean(axis=1)
            elif pooling_type == "max":
                pooled = center_preds.max(axis=1)
            else:
                pooled = center_preds.sum(axis=1)
        
        if pooled.shape[-1] == 1:
            pooled = pooled[:, 0]
        
        all_predictions.append(pooled)
        all_actuals.append(np.array(batch['y']))
    
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    return predictions, actuals


def compute_metrics_using_head_loss(
    model,
    dataloader: MPRADataLoader,
    head_name: str = 'mpra_head',
) -> dict:
    """Compute metrics using the head's loss function for pooling, then compute on all data.
    
    This ensures pooling matches training exactly, but computes Pearson correctly
    on all data at once (not per-batch averaged).
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for test batches
        head_name: Name of the custom head
        
    Returns:
        Dictionary with metrics
    """
    # Get the loss function from the model to ensure pooling matches
    loss_fn = model.create_loss_fn_for_head(head_name)
    
    all_predictions = []
    all_actuals = []
    
    print(f"Running inference on {len(dataloader)} batches...")
    
    for i, batch in enumerate(dataloader):
        # Get predictions (no gradients)
        with model._device_context:
            predictions = model._predict(
                model._params,
                model._state,
                batch['seq'],
                batch['organism_index'],
                negative_strand_mask=jnp.zeros(len(batch['seq']), dtype=bool),
                strand_reindexing=jax.device_put(
                    model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                    model._device_context._device
                ),
            )
        
        # Get predictions for our head
        head_predictions = predictions[head_name]
        
        # Use head's loss function to get pooled predictions (ensures pooling matches training)
        # We'll extract the pooled predictions from the loss computation
        loss_batch = {'targets': batch['y']}
        loss_dict = loss_fn(head_predictions, loss_batch)
        
        # Extract pooled predictions by replicating the head's pooling logic
        # This ensures we get the exact same pooled values as the loss function
        head_config = getattr(model, "_head_configs", {}).get(head_name, None)
        if head_config is not None:
            metadata = getattr(head_config, "metadata", {}) or {}
            pooling_type = metadata.get("pooling_type", "sum")
            center_bp = metadata.get("center_bp", 256)
        else:
            pooling_type = "sum"
            center_bp = 256
        
        head_np = np.array(head_predictions)
        seq_len = head_np.shape[1]
        
        # Pool exactly like head's loss() method (with center window for mean/max/sum)
        if pooling_type == "flatten":
            pooled = head_np.squeeze(1)
        elif pooling_type == "center":
            center_idx = seq_len // 2
            pooled = head_np[:, center_idx, :]
        else:
            # For mean/max/sum: use center window based on center_bp
            center_window_positions = max(1, center_bp // 128)
            window_size = min(center_window_positions, seq_len)
            center_start = max((seq_len - window_size) // 2, 0)
            center_end = center_start + window_size
            center_preds = head_np[:, center_start:center_end, :]
            
            if pooling_type == "mean":
                pooled = center_preds.mean(axis=1)
            elif pooling_type == "max":
                pooled = center_preds.max(axis=1)
            else:
                pooled = center_preds.sum(axis=1)
        
        if pooled.shape[-1] == 1:
            pooled = pooled[:, 0]
        
        all_predictions.append(pooled)
        all_actuals.append(np.array(batch['y']))
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
            print(f"  Processed {i + 1}/{len(dataloader)} batches")
    
    # Concatenate all batches
    predictions_all = np.concatenate(all_predictions, axis=0)
    actuals_all = np.concatenate(all_actuals, axis=0)
    
    # Compute metrics on ALL data at once (correct way)
    mse = np.mean((predictions_all.flatten() - actuals_all.flatten()) ** 2)
    pearson = np.corrcoef(predictions_all.flatten(), actuals_all.flatten())[0, 1]
    ss_res = np.sum((actuals_all.flatten() - predictions_all.flatten()) ** 2)
    ss_tot = np.sum((actuals_all.flatten() - np.mean(actuals_all.flatten())) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': float(mse),
        'pearson': float(pearson),  # Computed on all data (correct)
        'r2': float(r2),
        'n_samples': len(predictions_all),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned AlphaGenome MPRA model on test data'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--cell_type',
        type=str,
        default='HepG2',
        help='Cell type to test on (default: HepG2)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/test_predictions',
        help='Directory to save predictions (default: ./results/test_predictions)'
    )
    
    args = parser.parse_args()
    
    # Resolve checkpoint directory to an absolute path for Orbax
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    
    print("=" * 80)
    print("Testing Fine-tuned AlphaGenome MPRA Model")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Cell type: {args.cell_type}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Try to load head configuration from checkpoint (if available) so that
    # the test-time head definition exactly matches the training-time one.
    head_metadata = {
        'center_bp': 256,
        'pooling_type': 'sum',
    }
    # By default, assume full-model checkpoint if config is missing
    save_full_model = True
    config_path = checkpoint_dir / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            head_cfg = (cfg.get('head_configs', {})
                           .get('mpra_head', {})
                           .get('metadata', {}))
            # Merge, giving precedence to values from the checkpoint config.
            head_metadata.update(head_cfg)
            save_full_model = cfg.get('save_full_model', False)
        except Exception:
            # Fall back to defaults if anything goes wrong reading config.
            pass

    # Register custom MPRA head using (possibly) checkpoint-derived metadata.
    print("Registering custom MPRA head...")
    register_custom_head(
        'mpra_head',
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='mpra_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata=head_metadata,
        ),
    )
    
    # Match the training-time model creation for BOTH Stage 1 (heads-only) and
    # Stage 2 (full-model) checkpoints: encoder-output head on short MPRA
    # promoter constructs, with the same init_seq_len used in finetune_mpra.py.
    PROMOTER_CONSTRUCT_LENGTH = 281
    init_seq_len = PROMOTER_CONSTRUCT_LENGTH
    print(f"\nCreating MPRA model with encoder output (init_seq_len={init_seq_len} bp)...")
    model = create_model_with_custom_heads(
        'all_folds',
        custom_heads=['mpra_head'],
        use_encoder_output=True,
        init_seq_len=init_seq_len,
    )
    
    # Load checkpoint parameters directly with Orbax into this model
    import orbax.checkpoint as ocp
    checkpoint_path = checkpoint_dir / 'checkpoint'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpointer = ocp.StandardCheckpointer()
    loaded_params, loaded_state = checkpointer.restore(checkpoint_path)

    if save_full_model:
        # Full-model checkpoint (Stage 2): merge checkpoint params into model structure
        # to ensure Haiku transform compatibility
        print("Detected full-model checkpoint (save_full_model=True). "
              "Merging checkpoint parameters into model structure...")
        
        def deep_merge_params(model_params, checkpoint_params):
            """Recursively merge checkpoint parameters into model parameter structure.
            
            This ensures that the Haiku transform structure matches, while replacing
            all parameter values from the checkpoint.
            """
            import copy
            merged = copy.deepcopy(model_params)
            
            if isinstance(checkpoint_params, dict) and isinstance(model_params, dict):
                for key, checkpoint_value in checkpoint_params.items():
                    if key in model_params:
                        # Key exists in both - recurse if both are dicts
                        if isinstance(model_params[key], dict) and isinstance(checkpoint_value, dict):
                            merged[key] = deep_merge_params(model_params[key], checkpoint_value)
                        else:
                            # Replace with checkpoint value
                            merged[key] = checkpoint_value
                    else:
                        # New key from checkpoint - add it
                        merged[key] = checkpoint_value
            else:
                # Not both dicts - use checkpoint value
                merged = checkpoint_params
            
            return merged
        
        model._params = deep_merge_params(model._params, loaded_params)
        model._state = deep_merge_params(model._state, loaded_state)
        
        # Ensure parameters and state are on the correct device
        device = model._device_context._device
        model._params = jax.device_put(model._params, device)
        model._state = jax.device_put(model._state, device)
    else:
        # Heads-only checkpoint (Stage 1): merge loaded head params into the
        # existing model while keeping the pretrained backbone parameters.
        def merge_head_params(model_params, loaded_head_params):
            """Merge loaded head parameters into model parameters.

            This is adapted from alphagenome_ft.custom_model.load_checkpoint and
            alphagenome_FT_MPRA.src.training._train_stage to support both flat
            and nested Haiku parameter tree structures.
            """
            import copy
            merged = copy.deepcopy(model_params)

            # Structure 1: Flat keys like 'head/{head_name}/...' (use_encoder_output=True mode)
            if isinstance(loaded_head_params, dict):
                head_keys = {
                    k: v
                    for k, v in loaded_head_params.items()
                    if isinstance(k, str) and k.startswith('head/')
                }
                if head_keys:
                    for key, value in head_keys.items():
                        merged[key] = value

            # Structure 2: 'alphagenome/head' (encoder-only mode, nested)
            if isinstance(loaded_head_params, dict) and 'alphagenome/head' in loaded_head_params:
                if 'alphagenome/head' not in merged:
                    merged['alphagenome/head'] = {}
                for head_name, head_params in loaded_head_params['alphagenome/head'].items():
                    merged['alphagenome/head'][head_name] = head_params

            # Structure 3: 'alphagenome' -> 'head' (standard mode, nested)
            if isinstance(loaded_head_params, dict) and 'alphagenome' in loaded_head_params:
                if isinstance(loaded_head_params['alphagenome'], dict):
                    if 'head' in loaded_head_params['alphagenome']:
                        if 'alphagenome' not in merged or not isinstance(merged.get('alphagenome'), dict):
                            merged['alphagenome'] = {}
                        if 'head' not in merged['alphagenome']:
                            merged['alphagenome']['head'] = {}
                        for head_name, head_params in loaded_head_params['alphagenome']['head'].items():
                            merged['alphagenome']['head'][head_name] = head_params

            return merged

        model._params = merge_head_params(model._params, loaded_params)
        model._state = merge_head_params(model._state, loaded_state)
        print("Detected heads-only checkpoint (save_full_model=False). "
              "Merged head parameters into base model.")

    # Ensure parameters and state are on the correct device
    device = model._device_context._device
    model._params = jax.device_put(model._params, device)
    model._state = jax.device_put(model._state, device)

    print("✓ Model loaded successfully")
    
    # Create test dataset
    print(f"\nLoading test dataset (cell_type={args.cell_type})...")
    test_dataset = LentiMPRADataset(
        model=model,
        cell_type=args.cell_type,
        split='test',
        random_shift=False,
        reverse_complement=False
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # Create test dataloader
    test_loader = MPRADataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print(f"✓ Test dataloader created: {len(test_loader)} batches")
    
    # Compute metrics using head's loss function (matches training/wandb)
    print("\nComputing metrics using head's loss function (matching training)...")
    metrics = compute_metrics_using_head_loss(model, test_loader, head_name='mpra_head')
    
    # Also get predictions for saving to CSV
    print("\nCollecting predictions for saving...")
    predictions, actuals = get_predictions_for_saving(model, test_loader, head_name='mpra_head')
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Number of samples: {metrics['n_samples']}")
    print(f"MSE:              {metrics['mse']:.6f}")
    print(f"Pearson r:        {metrics['pearson']:.4f}")
    print(f"R²:               {metrics['r2']:.4f}")
    print("=" * 80)
    
    # Save predictions as CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from checkpoint dir name
    checkpoint_name = Path(args.checkpoint_dir).name
    predictions_file = output_dir / f"{checkpoint_name}_{args.cell_type}_test_predictions.csv"
    metrics_file = output_dir / f"{checkpoint_name}_{args.cell_type}_test_metrics.csv"
    
    print(f"\nSaving predictions to {predictions_file}...")
    
    # Create DataFrame with predictions and actuals
    df_predictions = pd.DataFrame({
        'sample_id': range(len(predictions)),
        'prediction': predictions.flatten(),
        'actual': actuals.flatten(),
        'error': (predictions.flatten() - actuals.flatten()),
        'squared_error': (predictions.flatten() - actuals.flatten()) ** 2,
    })
    df_predictions.to_csv(predictions_file, index=False)
    print(f"✓ Predictions saved to {predictions_file}")
    
    # Save metrics as separate CSV
    print(f"Saving metrics to {metrics_file}...")
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(metrics_file, index=False)
    print(f"✓ Metrics saved to {metrics_file}")
    
    print("\n✓ Testing complete!")
    
    return predictions, actuals, metrics


if __name__ == '__main__':
    main()