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
    load_checkpoint,
)
from src import EncoderMPRAHead, LentiMPRADataset, MPRADataLoader


def get_predictions(
    model,
    dataloader: MPRADataLoader,
    head_name: str = 'mpra_head',
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and actuals from model on test data.
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for test batches
        head_name: Name of the custom head
        
    Returns:
        Tuple of (predictions, actuals) as numpy arrays
    """
    all_predictions = []
    all_actuals = []
    
    # Infer pooling settings from the head config to match training.
    # IMPORTANT: We must exactly mirror EncoderMPRAHead.loss(), which:
    # - For 'flatten': squeezes the sequence dimension (batch, 1, num_tracks) -> (batch, num_tracks)
    # - For 'center': takes a single center position
    # - For 'mean'/'max'/'sum': pools over the ENTIRE sequence dimension
    head_config = getattr(model, "_head_configs", {}).get(head_name, None)
    if head_config is not None:
        metadata = getattr(head_config, "metadata", {}) or {}
        pooling_type = metadata.get("pooling_type", "sum")
    else:
        # Fallback to defaults used in training script
        pooling_type = "sum"
    
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
        head_predictions = predictions[head_name]  # (B, L_enc, num_tracks)
        
        # Convert to numpy for pooling
        head_np = np.array(head_predictions)
        seq_len = head_np.shape[1]
        
        # Pool EXACTLY like EncoderMPRAHead.loss()
        if pooling_type == "flatten":
            # In flatten mode, predict() returns (B, 1, num_tracks)
            pooled = head_np.squeeze(1)  # (B, num_tracks)
        elif pooling_type == "center":
            # Take only the center position
            center_idx = seq_len // 2
            pooled = head_np[:, center_idx, :]  # (B, num_tracks)
        else:
            # For mean/max/sum, pool over the ENTIRE sequence dimension
            if pooling_type == "mean":
                pooled = head_np.mean(axis=1)  # (B, num_tracks)
            elif pooling_type == "max":
                pooled = head_np.max(axis=1)
            else:  # "sum" or fallback
                pooled = head_np.sum(axis=1)
        
        # Collapse num_tracks if it's 1
        if pooled.shape[-1] == 1:
            pooled = pooled[:, 0]
        
        all_predictions.append(pooled)
        all_actuals.append(np.array(batch['y']))
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
            print(f"  Processed {i + 1}/{len(dataloader)} batches")
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    return predictions, actuals


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        Dictionary with metrics
    """
    # MSE
    mse = np.mean((predictions - actuals) ** 2)
    
    # Pearson correlation
    pearson = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]
    
    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': float(mse),
        'pearson': float(pearson),
        'r2': float(r2),
        'n_samples': len(predictions),
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
    
    if save_full_model:
        # Stage 2 (and any full-model) checkpoints: let alphagenome_ft reconstruct the
        # exact model + transform that was used during training to avoid Haiku
        # parameter/topology mismatches.
        print("\nDetected full-model checkpoint (save_full_model=True).")
        print("Loading full model with alphagenome_ft.load_checkpoint()...")
        model = load_checkpoint(
            checkpoint_dir,
            base_model_version='all_folds',
        )
        print("✓ Full model loaded successfully")
    else:
        # Stage 1 (heads-only) checkpoints: match the training-time model creation:
        # encoder-output head on short MPRA promoter constructs.
        PROMOTER_CONSTRUCT_LENGTH = 281
        init_seq_len = PROMOTER_CONSTRUCT_LENGTH
        print(f"\nDetected heads-only checkpoint. Creating MPRA model with "
              f"encoder output (init_seq_len={init_seq_len} bp)...")
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

        # Heads-only checkpoint: merge loaded head params into the existing model
        # while keeping the pretrained backbone parameters from the base model.
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

        # Ensure parameters and state are on the correct device
        device = model._device_context._device
        model._params = jax.device_put(model._params, device)
        model._state = jax.device_put(model._state, device)

        print("✓ Heads-only checkpoint merged successfully")
    
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
    
    # Get predictions
    print("\nRunning inference...")
    predictions, actuals = get_predictions(model, test_loader, head_name='mpra_head')
    print(f"✓ Inference complete: {len(predictions)} predictions")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, actuals)
    
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