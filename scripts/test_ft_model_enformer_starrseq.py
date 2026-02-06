"""
Test fine-tuned Enformer DeepSTARR model on DeepSTARR test data.

This script mirrors the style of:
- `test_ft_model_enformer_mpra.py` (PyTorch Lightning checkpoint testing)
and provides metrics similar to:
- `test_ft_model_starrseq.py` (per-task DeepSTARR metrics).

USAGE EXAMPLES:

1. Basic usage - test model and print metrics:
   python scripts/test_ft_model_enformer_starrseq.py \
       --checkpoint_path ./results/models/checkpoints/enformer/deepstarr/enformer-deepstarr/best-epoch=10-val_loss=0.1234.ckpt

2. Save predictions to custom directory:
   python scripts/test_ft_model_enformer_starrseq.py \
       --checkpoint_path ./results/models/checkpoints/enformer/deepstarr/enformer-deepstarr/best-epoch=10-val_loss=0.1234.ckpt \
       --output_dir ./results/test_predictions/enformer_deepstarr
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Import Enformer DeepSTARR utilities from finetuning script
import importlib.util

finetune_path = Path(__file__).parent / "finetune_enformer_starrseq.py"
spec_finetune = importlib.util.spec_from_file_location(
    "finetune_enformer_starrseq", finetune_path
)
finetune_module = importlib.util.module_from_spec(spec_finetune)
spec_finetune.loader.exec_module(finetune_module)

PyTorchDeepSTARRDataset = finetune_module.PyTorchDeepSTARRDataset
DeepSTARRDataModule = finetune_module.DeepSTARRDataModule
EnformerDeepSTARRLightning = finetune_module.EnformerDeepSTARRLightning
create_dummy_model_for_dataset = finetune_module.create_dummy_model_for_dataset


def get_predictions_for_saving(
    model: EnformerDeepSTARRLightning,
    dataloader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and actuals for saving to CSV.

    Args:
        model: EnformerDeepSTARRLightning instance
        dataloader: PyTorch DataLoader for test batches

    Returns:
        Tuple of (predictions, actuals) as numpy arrays
        - predictions: shape (N, 2) with [dev, hk] predictions
        - actuals: shape (N, 2) with [dev, hk] actual values
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

            # Get predictions: (batch, 2)
            preds = model.forward(seq)

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
    dev_r2 = 1 - (dev_ss_res / dev_ss_tot) if dev_ss_tot > 0 else 0.0

    # Task 1: Housekeeping enhancers
    hk_pred = predictions[:, 1]
    hk_actual = actuals[:, 1]
    hk_mse = np.mean((hk_pred - hk_actual) ** 2)
    hk_pearson = np.corrcoef(hk_pred, hk_actual)[0, 1]
    hk_ss_res = np.sum((hk_actual - hk_pred) ** 2)
    hk_ss_tot = np.sum((hk_actual - np.mean(hk_actual)) ** 2)
    hk_r2 = 1 - (hk_ss_res / hk_ss_tot) if hk_ss_tot > 0 else 0.0

    # Overall R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        # Overall metrics
        "mse": float(mse),
        "pearson": float(pearson),
        "r2": float(r2),
        "n_samples": int(len(predictions)),
        # Developmental enhancer metrics
        "dev_mse": float(dev_mse),
        "dev_pearson": float(dev_pearson),
        "dev_r2": float(dev_r2),
        # Housekeeping enhancer metrics
        "hk_mse": float(hk_mse),
        "hk_pearson": float(hk_pearson),
        "hk_r2": float(hk_r2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned Enformer DeepSTARR model on DeepSTARR test data"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to PyTorch Lightning checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/deepstarr",
        help="Path to DeepSTARR data directory (default: ./data/deepstarr)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/test_predictions/enformer_deepstarr",
        help="Directory to save predictions (default: ./results/test_predictions/enformer_deepstarr)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print("=" * 80)
    print("Testing Fine-tuned Enformer DeepSTARR Model")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Load model from checkpoint
    print("Loading model from checkpoint...")
    try:
        # Try loading with map_location to handle device mismatches
        model = EnformerDeepSTARRLightning.load_from_checkpoint(
            str(checkpoint_path),
            map_location="cpu",  # Load to CPU first, then move to GPU if available
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nTrying to load with strict=False...")
        # If loading fails, try with strict=False (in case of architecture changes)
        model = EnformerDeepSTARRLightning.load_from_checkpoint(
            str(checkpoint_path),
            strict=False,
            map_location="cpu",
        )
        print("✓ Model loaded successfully (with strict=False)")

    # Move model to appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"✓ Model moved to GPU: {device}")
    else:
        device = torch.device("cpu")
        model = model.to(device)
        print("✓ Model on CPU")

    # Set model to eval mode
    model.eval()

    # Create dummy model for dataset initialization
    dummy_model = create_dummy_model_for_dataset()

    # Create data module
    print(f"\nSetting up DeepSTARR test dataset...")
    data_module = DeepSTARRDataModule(
        dummy_model=dummy_model,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_shift=False,  # No augmentation during testing
        random_shift_likelihood=0.0,
        max_shift=0,
        reverse_complement=False,  # No augmentation during testing
    )
    data_module.setup("test")
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

    # Reference: DeepSTARR paper performance (as in test_ft_model_starrseq.py)
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

    # Create filename from checkpoint path
    checkpoint_name = checkpoint_path.stem  # Get filename without extension
    # Clean up checkpoint name (remove special characters)
    checkpoint_name = checkpoint_name.replace("=", "-").replace(" ", "_")
    predictions_file = output_dir / f"{checkpoint_name}_deepstarr_test_predictions.csv"
    metrics_file = output_dir / f"{checkpoint_name}_deepstarr_test_metrics.csv"

    print(f"\nSaving predictions to {predictions_file}...")

    # Create DataFrame with predictions and actuals
    df_predictions = pd.DataFrame(
        {
            "sample_id": range(len(predictions)),
            "dev_prediction": predictions[:, 0],
            "dev_actual": actuals[:, 0],
            "dev_error": predictions[:, 0] - actuals[:, 0],
            "hk_prediction": predictions[:, 1],
            "hk_actual": actuals[:, 1],
            "hk_error": predictions[:, 1] - actuals[:, 1],
        }
    )
    df_predictions.to_csv(predictions_file, index=False)
    print(f"✓ Predictions saved to {predictions_file}")

    # Save metrics as separate CSV
    print(f"Saving metrics to {metrics_file}...")
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(metrics_file, index=False)
    print(f"✓ Metrics saved to {metrics_file}")

    print("\n✓ Testing complete!")

    return predictions, actuals, metrics


if __name__ == "__main__":
    main()

