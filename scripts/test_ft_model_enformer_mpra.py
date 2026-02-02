"""
Test fine-tuned Enformer MPRA model on test data.

Loads a fine-tuned PyTorch Lightning checkpoint and evaluates it on the test split of the
LentiMPRA dataset from [Agarwal et al., 2025](https://www.nature.com/articles/s41586-024-08430-9)

USAGE EXAMPLES:

1. Basic usage - test model and print metrics:
   python scripts/test_ft_model_enformer_mpra.py --checkpoint_path ./results/models/checkpoints/enformer/HepG2/enformer-HepG2/best-epoch=10-val_loss=0.1234.ckpt

2. Save predictions to custom directory:
   python scripts/test_ft_model_enformer_mpra.py \
       --checkpoint_path ./results/models/checkpoints/enformer/HepG2/enformer-HepG2/best-epoch=10-val_loss=0.1234.ckpt \
       --output_dir ./results/test_predictions/enformer

3. Test on different cell type with custom batch size:
   python scripts/test_ft_model_enformer_mpra.py \
       --checkpoint_path ./results/models/checkpoints/enformer/K562/enformer-K562/best-epoch=10-val_loss=0.1234.ckpt \
       --cell_type K562 \
       --batch_size 64

4. Test Stage 2 checkpoint:
   python scripts/test_ft_model_enformer_mpra.py \
       --checkpoint_path ./results/models/checkpoints/enformer/HepG2/enformer-HepG2/stage2/best-stage2-epoch=05-val_loss=0.1234.ckpt
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional

# Import Enformer utilities
import importlib.util
enf_utils_path = Path(__file__).parent.parent / 'src' / 'enf_utils.py'
spec = importlib.util.spec_from_file_location("enf_utils", enf_utils_path)
enf_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enf_utils)
EncoderMPRAHead = enf_utils.EncoderMPRAHead
LentiMPRADatasetPyTorch = enf_utils.LentiMPRADatasetPyTorch

# Import from finetune_enformer_mpra.py using dynamic import
finetune_path = Path(__file__).parent / 'finetune_enformer_mpra.py'
spec_finetune = importlib.util.spec_from_file_location("finetune_enformer_mpra", finetune_path)
finetune_module = importlib.util.module_from_spec(spec_finetune)
spec_finetune.loader.exec_module(finetune_module)
PyTorchLentiMPRADataset = finetune_module.PyTorchLentiMPRADataset
LentiMPRADataModule = finetune_module.LentiMPRADataModule
EnformerMPRALightning = finetune_module.EnformerMPRALightning
create_dummy_model_for_dataset = finetune_module.create_dummy_model_for_dataset


def get_predictions_for_saving(
    model: EnformerMPRALightning,
    dataloader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and actuals for saving to CSV.
    
    Args:
        model: EnformerMPRALightning instance
        dataloader: PyTorch DataLoader for test batches
        
    Returns:
        Tuple of (predictions, actuals) as numpy arrays
    """
    model.eval()
    all_predictions = []
    all_actuals = []
    
    device = next(model.parameters()).device
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (seq, labels) in enumerate(dataloader):
            # Move to device
            seq = seq.to(device)
            labels = labels.to(device)
            
            # Get predictions
            preds = model.forward(seq)
            
            # Ensure scalar predictions
            if preds.dim() > 1:
                preds = preds.squeeze(-1) if preds.shape[-1] == 1 else preds.squeeze()
            
            all_predictions.append(preds.cpu().numpy())
            all_actuals.append(labels.cpu().numpy())
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Processed {batch_idx + 1}/{total_batches} batches")
    
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    return predictions, actuals


def compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> dict:
    """Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        Dictionary with metrics
    """
    # Flatten arrays
    pred_flat = predictions.flatten()
    actual_flat = actuals.flatten()
    
    # Compute metrics
    mse = np.mean((pred_flat - actual_flat) ** 2)
    pearson = np.corrcoef(pred_flat, actual_flat)[0, 1]
    ss_res = np.sum((actual_flat - pred_flat) ** 2)
    ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mse': float(mse),
        'pearson': float(pearson),
        'r2': float(r2),
        'n_samples': len(pred_flat),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned Enformer MPRA model on test data'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to PyTorch Lightning checkpoint file (.ckpt)'
    )
    parser.add_argument(
        '--cell_type',
        type=str,
        default='HepG2',
        choices=['HepG2', 'K562', 'WTC11'],
        help='Cell type to test on (default: HepG2)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/legnet_lentimpra',
        help='Path to LentiMPRA data directory (default: ./data/legnet_lentimpra)'
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
        default='./results/test_predictions/enformer',
        help='Directory to save predictions (default: ./results/test_predictions/enformer)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loader workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print("=" * 80)
    print("Testing Fine-tuned Enformer MPRA Model")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Cell type: {args.cell_type}")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint...")
    try:
        # Try loading with map_location to handle device mismatches
        model = EnformerMPRALightning.load_from_checkpoint(
            str(checkpoint_path),
            map_location='cpu'  # Load to CPU first, then move to GPU if available
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nTrying to load with strict=False...")
        # If loading fails, try with strict=False (in case of architecture changes)
        model = EnformerMPRALightning.load_from_checkpoint(
            str(checkpoint_path),
            strict=False,
            map_location='cpu'
        )
        print("✓ Model loaded successfully (with strict=False)")
    
    # Move model to appropriate device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        print(f"✓ Model moved to GPU: {device}")
    else:
        device = torch.device('cpu')
        model = model.to(device)
        print(f"✓ Model on CPU")
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy model for dataset initialization
    dummy_model = create_dummy_model_for_dataset()
    
    # Create data module
    print(f"\nSetting up test dataset for {args.cell_type}...")
    data_module = LentiMPRADataModule(
        dummy_model=dummy_model,
        data_path=args.data_path,
        cell_type=args.cell_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_shift=False,  # No augmentation during testing
        reverse_complement=False,  # No augmentation during testing
        pad_n_bases=0,
    )
    data_module.setup('test')
    print(f"✓ Test dataset loaded: {len(data_module.test_dataset)} samples")
    
    # Create test dataloader
    test_loader = data_module.test_dataloader()
    print(f"✓ Test dataloader created: {len(test_loader)} batches")
    
    # Get predictions
    print("\nRunning inference on test set...")
    predictions, actuals = get_predictions_for_saving(model, test_loader)
    
    # Compute metrics
    print("Computing metrics...")
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
    
    # Create filename from checkpoint path
    checkpoint_name = checkpoint_path.stem  # Get filename without extension
    # Clean up checkpoint name (remove special characters)
    checkpoint_name = checkpoint_name.replace('=', '-').replace(' ', '_')
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
