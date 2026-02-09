"""
Test fine-tuned AlphaGenome DeepSTARR model on test data.

Loads a fine-tuned model checkpoint and evaluates it on the test split of the
DeepSTARR dataset from [de Almeida et al., 2022](https://www.nature.com/articles/s41588-022-01048-5)

DeepSTARR predicts two types of enhancer activity:
- Developmental enhancer activity (Dev_log2_enrichment)
- Housekeeping enhancer activity (Hk_log2_enrichment)

USAGE EXAMPLES:

1. Basic usage - test model and print metrics:
   python scripts/test_ft_model_starrseq.py --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder

2. Save predictions to custom directory:
   python scripts/test_ft_model_starrseq.py \
       --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
       --output_dir ./results/test_predictions/starrseq

3. Test with custom batch size:
   python scripts/test_ft_model_starrseq.py \
       --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
       --batch_size 64

4. Use a local base AlphaGenome checkpoint instead of Kaggle:
   python scripts/test_ft_model_starrseq.py \
       --checkpoint_dir ./results/models/checkpoints/deepstarr/deepstarr-head-encoder \
       --base_checkpoint_path /home/USERNAME/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1/

5. Use in Python script or notebook:
   from test_ft_model_starrseq import get_predictions, compute_metrics
   from alphagenome_ft import load_checkpoint
   from src import DeepSTARRDataset, STARRSeqDataLoader
   
   # Load model
   model = load_checkpoint('./results/models/checkpoints/deepstarr/deepstarr-head-encoder')
   
   # Create test dataset
   test_dataset = DeepSTARRDataset(model, split='test')
   test_loader = STARRSeqDataLoader(test_dataset, batch_size=32, shuffle=False)
   
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
    load_checkpoint,
)
from src import DeepSTARRHead, DeepSTARRDataset, STARRSeqDataLoader


def get_predictions_for_saving(
    model,
    dataloader: STARRSeqDataLoader,
    head_name: str = 'deepstarr_head',
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and actuals for saving to CSV.
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for test batches
        head_name: Name of the custom head
        
    Returns:
        Tuple of (predictions, actuals) as numpy arrays
        - predictions: shape (N, 2) with [dev, hk] predictions
        - actuals: shape (N, 2) with [dev, hk] actual values
    """
    all_predictions = []
    all_actuals = []
    
    # Get pooling type and center_bp from head config
    head_config = getattr(model, "_head_configs", {}).get(head_name, None)
    if head_config is not None:
        metadata = getattr(head_config, "metadata", {}) or {}
        pooling_type = metadata.get("pooling_type", "flatten")
        center_bp = metadata.get("center_bp", 256)
    else:
        pooling_type = "flatten"
        center_bp = 256
    
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
        
        # Pool exactly like head's loss() method (with center window for mean/max/sum)
        if pooling_type == "flatten":
            pooled = head_np.squeeze(1)
        elif pooling_type == "center":
            seq_len = head_np.shape[1]
            center_idx = seq_len // 2
            pooled = head_np[:, center_idx, :]
        else:
            # For mean/max/sum: use center window based on center_bp
            seq_len = head_np.shape[1]
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
        
        all_predictions.append(pooled)
        all_actuals.append(np.array(batch['y']))
    
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    return predictions, actuals


def compute_metrics_using_head_loss(
    model,
    dataloader: STARRSeqDataLoader,
    head_name: str = 'deepstarr_head',
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
        
        # Use head's loss function to ensure we're using the same pooling logic
        # We'll extract the pooled predictions by replicating the head's pooling
        loss_batch = {'targets': batch['y']}
        loss_dict = loss_fn(head_predictions, loss_batch)
        
        # Extract pooled predictions by replicating the head's pooling logic
        head_config = getattr(model, "_head_configs", {}).get(head_name, None)
        if head_config is not None:
            metadata = getattr(head_config, "metadata", {}) or {}
            pooling_type = metadata.get("pooling_type", "flatten")
            center_bp = metadata.get("center_bp", 256)
        else:
            pooling_type = "flatten"
            center_bp = 256
        
        head_np = np.array(head_predictions)
        
        # Pool exactly like head's loss() method (with center window for mean/max/sum)
        if pooling_type == "flatten":
            pooled = head_np.squeeze(1)
        elif pooling_type == "center":
            seq_len = head_np.shape[1]
            center_idx = seq_len // 2
            pooled = head_np[:, center_idx, :]
        else:
            # For mean/max/sum: use center window based on center_bp
            seq_len = head_np.shape[1]
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
        
        all_predictions.append(pooled)
        all_actuals.append(np.array(batch['y']))
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(f"  Processed {i + 1}/{len(dataloader)} batches")
    
    # Concatenate all batches
    predictions_all = np.concatenate(all_predictions, axis=0)  # (N, 2)
    actuals_all = np.concatenate(all_actuals, axis=0)  # (N, 2)
    
    # Compute metrics on ALL data at once (correct way)
    # Overall metrics
    mse = np.mean((predictions_all - actuals_all) ** 2)
    pearson = np.corrcoef(predictions_all.flatten(), actuals_all.flatten())[0, 1]
    ss_res = np.sum((actuals_all - predictions_all) ** 2)
    ss_tot = np.sum((actuals_all - np.mean(actuals_all)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Per-task metrics
    dev_pred = predictions_all[:, 0]
    dev_actual = actuals_all[:, 0]
    dev_mse = np.mean((dev_pred - dev_actual) ** 2)
    dev_pearson = np.corrcoef(dev_pred, dev_actual)[0, 1]
    dev_ss_res = np.sum((dev_actual - dev_pred) ** 2)
    dev_ss_tot = np.sum((dev_actual - np.mean(dev_actual)) ** 2)
    dev_r2 = 1 - (dev_ss_res / dev_ss_tot)
    
    hk_pred = predictions_all[:, 1]
    hk_actual = actuals_all[:, 1]
    hk_mse = np.mean((hk_pred - hk_actual) ** 2)
    hk_pearson = np.corrcoef(hk_pred, hk_actual)[0, 1]
    hk_ss_res = np.sum((hk_actual - hk_pred) ** 2)
    hk_ss_tot = np.sum((hk_actual - np.mean(hk_actual)) ** 2)
    hk_r2 = 1 - (hk_ss_res / hk_ss_tot)
    
    return {
        'mse': float(mse),
        'pearson': float(pearson),  # Computed on all data (correct)
        'r2': float(r2),
        'n_samples': len(predictions_all),
        'dev_mse': float(dev_mse),
        'dev_pearson': float(dev_pearson),  # Computed on all data (correct)
        'dev_r2': float(dev_r2),
        'hk_mse': float(hk_mse),
        'hk_pearson': float(hk_pearson),  # Computed on all data (correct)
        'hk_r2': float(hk_r2),
    }


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute evaluation metrics for DeepSTARR (two outputs).
    
    Args:
        predictions: Predicted values, shape (N, 2) with [dev, hk]
        actuals: Actual values, shape (N, 2) with [dev, hk]
        
    Returns:
        Dictionary with overall and per-task metrics
    """
    # Ensure 2D shape
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if actuals.ndim == 1:
        actuals = actuals[:, None]
    
    # Overall MSE (across both tasks)
    mse = np.mean((predictions - actuals) ** 2)
    
    # Overall Pearson correlation
    pearson = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]
    
    # Per-task metrics
    # Task 0: Developmental enhancers
    dev_pred = predictions[:, 0]
    dev_actual = actuals[:, 0]
    dev_mse = np.mean((dev_pred - dev_actual) ** 2)
    dev_pearson = np.corrcoef(dev_pred, dev_actual)[0, 1]
    dev_ss_res = np.sum((dev_actual - dev_pred) ** 2)
    dev_ss_tot = np.sum((dev_actual - np.mean(dev_actual)) ** 2)
    dev_r2 = 1 - (dev_ss_res / dev_ss_tot)
    
    # Task 1: Housekeeping enhancers
    hk_pred = predictions[:, 1]
    hk_actual = actuals[:, 1]
    hk_mse = np.mean((hk_pred - hk_actual) ** 2)
    hk_pearson = np.corrcoef(hk_pred, hk_actual)[0, 1]
    hk_ss_res = np.sum((hk_actual - hk_pred) ** 2)
    hk_ss_tot = np.sum((hk_actual - np.mean(hk_actual)) ** 2)
    hk_r2 = 1 - (hk_ss_res / hk_ss_tot)
    
    # Overall R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        # Overall metrics
        'mse': float(mse),
        'pearson': float(pearson),
        'r2': float(r2),
        'n_samples': len(predictions),
        # Developmental enhancer metrics
        'dev_mse': float(dev_mse),
        'dev_pearson': float(dev_pearson),
        'dev_r2': float(dev_r2),
        # Housekeeping enhancer metrics
        'hk_mse': float(hk_mse),
        'hk_pearson': float(hk_pearson),
        'hk_r2': float(hk_r2),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned AlphaGenome DeepSTARR model on test data'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/deepstarr',
        help='Path to DeepSTARR data directory (default: ./data/deepstarr)'
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
        default='./results/test_predictions/starrseq',
        help='Directory to save predictions (default: ./results/test_predictions)'
    )
    parser.add_argument(
        '--base_checkpoint_path',
        type=str,
        default=None,
        help='Optional local AlphaGenome checkpoint directory. '
             'If provided, the base model will be loaded from this path '
             'instead of using Kaggle.'
    )
    
    args = parser.parse_args()
    
    # Resolve checkpoint directory to an absolute path for Orbax
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    base_checkpoint_path = args.base_checkpoint_path
    
    print("=" * 80)
    print("Testing Fine-tuned AlphaGenome DeepSTARR Model")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    if base_checkpoint_path is not None:
        print(f"Base AlphaGenome checkpoint: {base_checkpoint_path}")
    print()
    
    # Try to load head configuration from checkpoint (if available) so that
    # the test-time head definition exactly matches the training-time one.
    head_metadata = {
        'center_bp': 256,
        'pooling_type': 'flatten',
    }
    config_path = checkpoint_dir / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            head_cfg = (cfg.get('head_configs', {})
                           .get('deepstarr_head', {})
                           .get('metadata', {}))
            # Merge, giving precedence to values from the checkpoint config.
            head_metadata.update(head_cfg)
            print(f"Loaded head config from checkpoint: {head_metadata}")
        except Exception as e:
            print(f"Warning: Could not load head config from checkpoint: {e}")
            # Fall back to defaults if anything goes wrong reading config.
            pass

    # Register custom DeepSTARR head using (possibly) checkpoint-derived metadata.
    print("\nRegistering custom DeepSTARR head...")
    register_custom_head(
        'deepstarr_head',
        DeepSTARRHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name='deepstarr_head',
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=2,  # Two outputs: developmental and housekeeping
            metadata=head_metadata,
        ),
    )
    
    # For flatten pooling, we need to determine the correct init_seq_len from
    # the checkpoint's metadata. This ensures the model head is initialized
    # with the same input size as during training.
    init_seq_len = None  # Default: let model decide
    
    if head_metadata.get('pooling_type') == 'flatten':
        # For flatten pooling, use center_bp from metadata as init_seq_len
        # This matches how the training script initializes the model
        init_seq_len = head_metadata.get('center_bp', 256)
        print(f"Flatten pooling detected, using init_seq_len={init_seq_len}bp from center_bp")
    
    # Load trained model using load_checkpoint (now handles minimal models correctly)
    print("\nLoading trained model...")
    model = load_checkpoint(
        str(checkpoint_dir),
        base_model_version='all_folds',
        base_checkpoint_path=base_checkpoint_path,
        device=None,  # Will use default device
        init_seq_len=init_seq_len,
    )
    print("✓ Model loaded successfully")
    
    # Create test dataset
    print(f"\nLoading test dataset from {args.data_path}...")
    test_dataset = DeepSTARRDataset(
        model=model,
        path_to_data=args.data_path,
        split='test',
        organism=dna_model.Organism.HOMO_SAPIENS, #dna_model.Organism.DROSOPHILA_MELANOGASTER,
        random_shift=False,
        reverse_complement=False
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # Create test dataloader
    test_loader = STARRSeqDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print(f"✓ Test dataloader created: {len(test_loader)} batches")
    
    # Compute metrics using head's loss function (matches training/wandb)
    print("\nComputing metrics using head's loss function (matching training)...")
    metrics = compute_metrics_using_head_loss(model, test_loader, head_name='deepstarr_head')
    
    # Also get predictions for saving to CSV
    print("\nCollecting predictions for saving...")
    predictions, actuals = get_predictions_for_saving(model, test_loader, head_name='deepstarr_head')
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Number of samples:        {metrics['n_samples']}")
    print()
    print("Overall Metrics:")
    print(f"  MSE:                    {metrics['mse']:.6f}")
    print(f"  Pearson r:              {metrics['pearson']:.4f}")
    print(f"  R²:                     {metrics['r2']:.4f}")
    print()
    print("Developmental Enhancers:")
    print(f"  MSE:                    {metrics['dev_mse']:.6f}")
    print(f"  Pearson r:              {metrics['dev_pearson']:.4f}")
    print(f"  R²:                     {metrics['dev_r2']:.4f}")
    print()
    print("Housekeeping Enhancers:")
    print(f"  MSE:                    {metrics['hk_mse']:.6f}")
    print(f"  Pearson r:              {metrics['hk_pearson']:.4f}")
    print(f"  R²:                     {metrics['hk_r2']:.4f}")
    print("=" * 80)
    
    # Reference: DeepSTARR paper performance
    print("\nReference DeepSTARR model:")
    print("  Dev Pearson r: 0.656")
    print("  Hk Pearson r:  0.736")
    print("Reference Dream-RNN model:")
    print("  Dev Pearson r: 0.665")
    print("  Hk Pearson r:  0.746")
    print("=" * 80)
    
    # Save predictions as CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from checkpoint dir name
    checkpoint_name = Path(args.checkpoint_dir).name
    predictions_file = output_dir / f"{checkpoint_name}_deepstarr_test_predictions.csv"
    metrics_file = output_dir / f"{checkpoint_name}_deepstarr_test_metrics.csv"
    
    print(f"\nSaving predictions to {predictions_file}...")
    
    # Create DataFrame with predictions and actuals
    df_predictions = pd.DataFrame({
        'sample_id': range(len(predictions)),
        'dev_prediction': predictions[:, 0],
        'dev_actual': actuals[:, 0],
        'dev_error': predictions[:, 0] - actuals[:, 0],
        'hk_prediction': predictions[:, 1],
        'hk_actual': actuals[:, 1],
        'hk_error': predictions[:, 1] - actuals[:, 1],
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
