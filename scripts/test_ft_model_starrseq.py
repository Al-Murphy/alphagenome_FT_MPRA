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

4. Use in Python script or notebook:
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
    create_model_with_custom_heads,
)
from src import DeepSTARRHead, DeepSTARRDataset, STARRSeqDataLoader


def get_predictions(
    model,
    dataloader: STARRSeqDataLoader,
    head_name: str = 'deepstarr_head',
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and actuals from model on test data.
    
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
    
    # Infer pooling settings from the head config to match training
    head_config = getattr(model, "_head_configs", {}).get(head_name, None)
    if head_config is not None:
        metadata = getattr(head_config, "metadata", {}) or {}
        pooling_type = metadata.get("pooling_type", "flatten")
    else:
        # Fallback to defaults used in training script
        pooling_type = "flatten"
    
    print(f"Running inference on {len(dataloader)} batches...")
    print(f"  Pooling type: {pooling_type}")
    
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
        head_predictions = predictions[head_name]  # Shape depends on pooling_type
        
        # Convert to numpy
        head_np = np.array(head_predictions)
        
        # Debug output for first batch
        if i == 0:
            print(f"\n  DEBUG - First batch:")
            print(f"    Input seq shape: {batch['seq'].shape}")
            print(f"    Head predictions shape: {head_np.shape}")
            print(f"    Head predictions dtype: {head_np.dtype}")
            print(f"    Head predictions sample value: {head_np[0]}")
            targets_np = np.array(batch['y'])
            print(f"    Targets shape: {targets_np.shape}")
            print(f"    Targets dtype: {targets_np.dtype}")
            print(f"    Targets sample value: {targets_np[0]}")
            # Check if predictions are numeric
            try:
                pred_min = float(head_np.min())
                pred_max = float(head_np.max())
                pred_mean = float(head_np.mean())
                print(f"    Head predictions stats: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}")
            except (TypeError, ValueError) as e:
                print(f"    Head predictions stats ERROR: {e}")
            try:
                tgt_min = float(targets_np.min())
                tgt_max = float(targets_np.max())
                tgt_mean = float(targets_np.mean())
                print(f"    Targets stats: min={tgt_min:.4f}, max={tgt_max:.4f}, mean={tgt_mean:.4f}")
            except (TypeError, ValueError) as e:
                print(f"    Targets stats ERROR: {e}")
        
        # Pool to get (B, 2) predictions - must match DeepSTARRHead.loss() logic!
        if pooling_type == "flatten":
            # For flatten, predict() returns (B, 1, 2) - already pooled, just squeeze
            pooled = head_np.squeeze(1)  # (B, 2)
        elif pooling_type == "center":
            # Take center position only
            seq_len = head_np.shape[1]
            center_idx = seq_len // 2
            pooled = head_np[:, center_idx, :]  # (B, 2)
        else:
            # For mean/max/sum, pool over the sequence dimension
            seq_len = head_np.shape[1]
            if pooling_type == "mean":
                pooled = head_np.mean(axis=1)  # (B, 2)
            elif pooling_type == "max":
                pooled = head_np.max(axis=1)
            else:  # "sum" or fallback
                pooled = head_np.sum(axis=1)
        
        all_predictions.append(pooled)
        all_actuals.append(np.array(batch['y']))
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(f"  Processed {i + 1}/{len(dataloader)} batches")
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 2)
    actuals = np.concatenate(all_actuals, axis=0)  # (N, 2)
    
    return predictions, actuals


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
    
    args = parser.parse_args()
    
    # Resolve checkpoint directory to an absolute path for Orbax
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    
    print("=" * 80)
    print("Testing Fine-tuned AlphaGenome DeepSTARR Model")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Try to load head configuration from checkpoint (if available) so that
    # the test-time head definition exactly matches the training-time one.
    head_metadata = {
        'center_bp': 249,
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

    # Load checkpoint early to inspect weight shapes (needed for flatten pooling)
    import orbax.checkpoint as ocp
    checkpoint_path = checkpoint_dir / 'checkpoint'
    if not checkpoint_path.exists():
        stage1_path = checkpoint_dir / 'stage1' / 'checkpoint'
        if stage1_path.exists():
            checkpoint_path = stage1_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {stage1_path}")
    
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpointer = ocp.StandardCheckpointer()
    loaded_params, loaded_state = checkpointer.restore(checkpoint_path)
    
    # For flatten pooling, we need to determine the correct init_seq_len from
    # the checkpoint's weight shapes. This ensures the model head is initialized
    # with the same input size as during training.
    init_seq_len = None  # Default: let model decide
    
    if head_metadata.get('pooling_type') == 'flatten':
        # For flatten pooling, use center_bp from metadata as init_seq_len
        # This matches how the training script initializes the model
        init_seq_len = head_metadata.get('center_bp', 249)
        print(f"Flatten pooling detected, using init_seq_len={init_seq_len}bp from center_bp")

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
    
    # Create DeepSTARR model that uses encoder output only (no transformer/decoder),
    # matching how the model was trained for short enhancer sequences.
    print("\nCreating DeepSTARR model with encoder output...")
    model = create_model_with_custom_heads(
        'all_folds',
        custom_heads=['deepstarr_head'],
        use_encoder_output=True,
        init_seq_len=init_seq_len,
    )
    
    # Checkpoint was already loaded above for flatten pooling detection
    # loaded_params and loaded_state are already available

    # Determine whether this is a full-model checkpoint or heads-only checkpoint.
    config_path = checkpoint_dir / 'config.json'
    if not config_path.exists():
        config_path = checkpoint_path.parent / 'config.json'
    
    save_full_model = True  # default to full model if no config is found
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        save_full_model = cfg.get('save_full_model', False)

    if save_full_model:
        # Full model checkpoint: replace entire parameter/state trees
        model._params = loaded_params
        model._state = loaded_state
    else:
        # Heads-only checkpoint: merge loaded head params into the existing model
        def merge_head_params(model_params, loaded_head_params):
            """Merge loaded head parameters into model parameters."""
            import copy
            merged = copy.deepcopy(model_params)

            # Structure 1: Flat keys like 'head/{head_name}/...'
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
        
        # Debug: show loaded head parameter shapes
        print("\nDEBUG - Loaded head parameter keys:")
        head_keys = [k for k in loaded_params.keys() if 'deepstarr' in k.lower() or k.startswith('head/')]
        for key in head_keys[:10]:
            val = loaded_params[key]
            if hasattr(val, 'shape'):
                print(f"  {key}: {val.shape}")
            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    if hasattr(subval, 'shape'):
                        print(f"  {key}/{subkey}: {subval.shape}")

    print("✓ Model loaded successfully")
    
    # Debug: find ALL deepstarr_head related parameters in the model
    print("\nDEBUG - Searching for deepstarr_head parameters in model:")
    def find_deepstarr_keys(d, prefix=''):
        results = []
        if isinstance(d, dict):
            for k, v in d.items():
                full_key = f"{prefix}/{k}" if prefix else k
                if 'deepstarr' in str(k).lower():
                    results.append((full_key, type(v).__name__, getattr(v, 'shape', None)))
                if isinstance(v, dict):
                    results.extend(find_deepstarr_keys(v, full_key))
        return results
    
    deepstarr_keys = find_deepstarr_keys(model._params)
    if deepstarr_keys:
        for key, type_name, shape in deepstarr_keys:
            if shape:
                print(f"  {key}: {shape}")
            else:
                print(f"  {key}: ({type_name})")
    else:
        print("  WARNING: No deepstarr_head parameters found!")
    
    # Also check if there are MULTIPLE paths with deepstarr (this would indicate a problem)
    flat_keys = [k for k in model._params.keys() if 'deepstarr' in str(k).lower()]
    nested_keys = find_deepstarr_keys(model._params)
    print(f"\n  Flat deepstarr keys: {len(flat_keys)}")
    print(f"  Nested deepstarr keys: {len(nested_keys)}")
    
    # Show first few flat keys to help identify structure
    print(f"  Flat key examples: {flat_keys[:3]}")
    
    # Compare a specific parameter value from checkpoint vs model to verify they match
    if 'head/deepstarr_head/~predict/output' in loaded_params:
        chkpt_output_b = loaded_params['head/deepstarr_head/~predict/output'].get('b')
        if chkpt_output_b is not None:
            print(f"\n  Checkpoint output/b: {np.array(chkpt_output_b)}")
    if 'head/deepstarr_head/~predict/output' in model._params:
        model_output_b = model._params['head/deepstarr_head/~predict/output'].get('b')
        if model_output_b is not None:
            print(f"  Model output/b: {np.array(model_output_b)}")
    
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
    
    # Get predictions
    print("\nRunning inference...")
    predictions, actuals = get_predictions(model, test_loader, head_name='deepstarr_head')
    print(f"✓ Inference complete: {len(predictions)} predictions")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, actuals)
    
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
    print("\nReference (DeepSTARR paper):")
    print("  Dev Pearson r: 0.68")
    print("  Hk Pearson r:  0.74")
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
