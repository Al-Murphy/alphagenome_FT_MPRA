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
        
        # Convert to numpy and collect
        all_predictions.append(np.array(head_predictions))
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
    
    print("=" * 80)
    print("Testing Fine-tuned AlphaGenome MPRA Model")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Cell type: {args.cell_type}")
    print(f"Batch size: {args.batch_size}")
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
                'center_bp': 256,
                'pooling_type': 'sum'
            }
        )
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint_dir}...")
    model = load_checkpoint(
        args.checkpoint_dir,
        base_model_version='all_folds',
    )
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

