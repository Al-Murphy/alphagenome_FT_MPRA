"""
Test LegNet MPRA model on test data.

Loads a trained LegNet model checkpoint and evaluates it on the test split of the
LentiMPRA dataset from [Agarwal et al., 2025](https://www.nature.com/articles/s41586-024-08430-9)

USAGE EXAMPLES:

1. Basic usage - test model and print metrics:
   python scripts/test_mpralegnet.py \
       --config ./data/legnet_lentimpra/hepg2_config.json \
       --model ./data/legnet_lentimpra/hepg2_best_model_test10_val1.ckpt

2. Test on different cell type:
   python scripts/test_mpralegnet.py \
       --config ./data/legnet_lentimpra/*_config.json \
       --model ./data/legnet_lentimpra/*_best_model_test10_val1.ckpt \
       --cell_type K562

3. Use GPU and save to custom directory:
   python scripts/test_mpralegnet.py \
       --config ./data/legnet_lentimpra/*_config.json \
       --model ./data/legnet_lentimpra/*_best_model_test10_val1.ckpt \
       --device 0 \
       --output_dir ./results/legnet_predictions
"""
import os
import pandas as pd
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True #type: ignore
    torch.backends.cudnn.benchmark = False #type: ignore

CODES = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

INV_CODES = {value: key for key, value in CODES.items()}

COMPL = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G',
    'N': 'N'
}

def n2id(n):
    return CODES[n.upper()]

def id2n(i):
    return INV_CODES[i]

def n2compl(n):
    return COMPL[n.upper()]

def parameter_count(model):
    pars = 0  
    for _, p  in model.named_parameters():    
        pars += torch.prod(torch.tensor(p.shape))
    return pars

def revcomp(seq):
    return "".join((n2compl(x) for x in reversed(seq)))

def get_rev(df):
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    revdf['rev'] = 1
    return revdf

def add_rev(df):
    df = df.copy()
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    df['rev'] = 0
    revdf['rev'] = 1
    df = pd.concat([df, revdf]).reset_index(drop=True)
    return df

class Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()
    def forward(self, seq):
        if isinstance(seq, torch.FloatTensor):
            return seq
        seq = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq))
        code = F.one_hot(code, num_classes=5) # 5th class is N
        
        code[code[:, 4] == 1] = 0.25 # encode Ns with .25
        code = code[:, :4].float() 
        return code.transpose(0, 1)

class MPRALegNetDataset:
    def __init__(
        self,
        path_to_data: str = "./data/legnet_lentimpra",
        cell_type: str = "HepG2",
        split: str = "train",
        test_fold: int = [10],
        val_fold: int = [1],
        train_fold: int = [2, 3, 4, 5, 6, 7, 8, 9],
        reverse_complement: bool = False,
        reverse_complement_likelihood: float = 0.5,
        subset_frac: float = 1.0,
    ):
        assert split in ["train", "val", "test"], f"split must be one of train, val, test"
        assert cell_type in ["HepG2", "K562", "WTC11"], f"cell_type must be one of HepG2, K562, WTC11"
        assert os.path.exists(path_to_data), f"path_to_data must exist"
        
        self.path_to_data = path_to_data
        self.cell_type = cell_type
        self.split = split
        # get test, val, train folds
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.train_fold = train_fold
        self.chosen_fold = train_fold if split == "train" else val_fold if split == "val" else test_fold
        # rev comp
        assert reverse_complement_likelihood >= 0 and reverse_complement_likelihood <= 1, "reverse_complement_likelihood must be between 0 and 1"
        self.reverse_complement = reverse_complement
        self.reverse_complement_likelihood = reverse_complement_likelihood
        assert subset_frac >= 0 and subset_frac <= 1, "subset_frac must be between 0 and 1"
        self.subset_frac = subset_frac
        
        # load the data and process
        self.data = pd.read_csv(os.path.join(path_to_data, f"{cell_type}.tsv"), sep="\t")
        self.data = self.data[self.data["rev"] == 0]  # https://github.com/autosome-ru/human_legnet/issues/1
        # filt by fold
        self.data = self.data[self.data["fold"].isin(self.chosen_fold)]
        self.data = self.data.reset_index(drop=True)
        if self.subset_frac < 1.0:
            self.data = self.data.sample(frac=self.subset_frac)
        print(f"Loaded {len(self.data)} samples for {split} split")
        #in case multithreading don't initiate class
        self.seq2tensor = None
        
    def __len__(self):
        # Return total number of samples
        return len(self.data)
    
    def _reverse_complement_onehot(self, seq_onehot: np.ndarray, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Reverse complement a one-hot encoded DNA sequence.
        
        1. Reverse along sequence dimension
        2. Swap channels: A↔T (0↔3), C↔G (1↔2)
        
        This matches the approach used in alphagenome_research/model/augmentation.py
        for reverse complementing predictions, adapted for one-hot sequences.
        
        Args:
            seq_onehot: One-hot encoded sequence (seq_len, 4) or (batch, seq_len, 4)
            force: If True, always reverse complement regardless of random check
            
        Returns:
            Tuple of (reverse_complemented_sequence, updated_rng_key)
        """
        # Generate random number for augmentation (if not forced)
        if not force:
            rand_num = np.random.uniform(0, 1)
            if rand_num > self.reverse_complement_likelihood:
                return seq_onehot
        
        # Reverse complement: reverse sequence and swap channels
        # Channel mapping: A(0)↔T(3), C(1)↔G(2) -> [3, 2, 1, 0]
        # This matches the strand_reindexing used in alphagenome_research:
        # Matches the approach in `alphagenome_research/model/augmentation.py` lines 68-70
        strand_reindexing = np.array([3, 2, 1, 0])
        
        # Reverse along sequence dimension (axis 0 for 2D, axis 1 for 3D)
        if seq_onehot.ndim == 2:
            # Shape: (4, seq_len)
            rev_seq = seq_onehot[:, ::-1]
            rev_comp = rev_seq[strand_reindexing, :]
        else:
            # Shape: (batch, 4, seq_len)
            rev_seq = seq_onehot[:, :, ::-1]
            rev_comp = rev_seq[:, strand_reindexing, :]
        
        return rev_comp

    
    def __getitem__(self, idx):
        # Get the sequence and label for the given index
        sequence = self.data.iloc[idx]['seq']
        label = self.data.iloc[idx]['mean_value']
        
        # Convert sequence to one-hot encoding
        if self.seq2tensor is None:
            self.seq2tensor = Seq2Tensor()
        sequence_onehot = self.seq2tensor(sequence)

        if self.reverse_complement:
            # Get random number to check if we should reverse complement
            sequence_onehot = self._reverse_complement_onehot(sequence_onehot)
        
        return {
            "seq": sequence_onehot,
            "y": label
        }
        
import sys
from pathlib import Path

# Add parent directory to path to import human_legnet
parent_dir = Path(__file__).resolve().parent.parent.parent
human_legnet_dir = parent_dir / 'human_legnet'

# Add both parent and human_legnet to path
# (human_legnet needs to be in path for its internal relative imports)
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(human_legnet_dir))

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from human_legnet.training_config import TrainingConfig
from human_legnet.trainer import LitModel


def get_predictions_and_actuals(
    model: LitModel,
    train_cfg: TrainingConfig,
    dataset: MPRALegNetDataset,
    device: str = 'cpu',
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and actuals from LegNet model on dataset using simple forward pass.
    
    Args:
        model: LitModel instance
        train_cfg: Training configuration
        dataset: MPRALegNetDataset instance
        device: Device to run on ('cpu' or 'cuda')
        batch_size: Batch size for DataLoader
        
    Returns:
        Tuple of (predictions, actuals) as numpy arrays
    """
    # Set model to eval mode
    model.eval()
    
    # Move model to device
    if device != 'cpu':
        model = model.to(device)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers if hasattr(train_cfg, 'num_workers') else 0,
    )
    
    all_predictions = []
    all_actuals = []
    
    print(f"Running inference on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Get sequence and actuals from batch
            seq = batch['seq']  # (batch, 4, seq_len)
            actuals = batch['y']  # (batch,)
            
            # Add reverse channel if needed
            if train_cfg.use_reverse_channel:
                # Add reverse channel (all zeros for forward pass)
                batch_size, _, seq_len = seq.shape
                rev_channel = torch.zeros((batch_size, 1, seq_len), dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, rev_channel], dim=1)  # (batch, 5, seq_len)
            
            # Move to device
            if device != 'cpu':
                seq = seq.to(device)
            
            # Forward pass
            predictions = model.model(seq)  # Direct model forward pass
            
            # Convert to numpy and collect
            all_predictions.append(predictions.cpu().numpy())
            all_actuals.append(actuals.numpy())
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(f"  Processed {i + 1}/{len(dataloader)} batches")
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # Flatten predictions if needed
    if predictions.ndim > 1:
        if predictions.shape[1] == 1:
            predictions = predictions.flatten()
        else:
            # If multi-dimensional, take mean or first dimension as needed
            predictions = predictions.squeeze()
    
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Actuals shape: {actuals.shape}")
    
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
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mse': float(mse),
        'pearson': float(pearson),
        'r2': float(r2),
        'n_samples': len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test LegNet MPRA model on test data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='./data/legnet_lentimpra/*_config.json', #replace star with cell type
        help='Path to model config JSON file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./data/legnet_lentimpra/*_best_model_test10_val1.ckpt', #replace star with cell type
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--cell_type',
        type=str,
        default='HepG2',
        choices=['HepG2', 'K562', 'WTC11'],
        help='Cell type for MPRA data'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Data split to evaluate on'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use: "cpu" or GPU index (e.g., "0", "1"). If not provided, will use GPU if available.'
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
        default='./results/mpralegnet_predictions',
        help='Directory to save predictions (default: ./results/mpralegnet_predictions)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/legnet_lentimpra',
        help='Path to LentiMPRA data directory'
    )
    
    args = parser.parse_args()
    
    # Replace * in config and model paths with cell_type if present
    if '*' in args.config:
        args.config = args.config.replace('*', args.cell_type.lower())
    if '*' in args.model:
        args.model = args.model.replace('*', args.cell_type.lower())
    
    # Auto-detect device if not provided
    if args.device is None:
        if torch.cuda.is_available():
            device_str = 'cuda'
            print("GPU detected, using GPU")
        else:
            device_str = 'cpu'
            print("No GPU detected, using CPU")
    elif args.device == 'cpu':
        device_str = 'cpu'
    else:
        # GPU specified by index
        if torch.cuda.is_available():
            device_str = f'cuda:{args.device}'
        else:
            print(f"Warning: GPU {args.device} requested but not available, falling back to CPU")
            device_str = 'cpu'
    
    # Print configuration
    print("=" * 80)
    print("LegNet MPRA Model Testing")
    print("=" * 80)
    print(f"Config:           {args.config}")
    print(f"Model:            {args.model}")
    print(f"Cell type:         {args.cell_type}")
    print(f"Split:             {args.split}")
    print(f"Data path:         {args.data_path}")
    print(f"Device:            {device_str}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Output directory:  {args.output_dir}")
    print("=" * 80)
    print()
    
    # Load training config
    print("Loading training configuration...")
    train_cfg = TrainingConfig.from_json(args.config)
    print("✓ Config loaded")
    
    # Load model
    print("Loading model checkpoint...")
    # PyTorch 2.6+ requires safe globals for checkpoint loading
    # Use context manager to allow getattr for loading checkpoints saved with older PyTorch
    try:
        # Try using safe_globals context manager (PyTorch 2.6+)
        with torch.serialization.safe_globals([getattr]):
            model = LitModel.load_from_checkpoint(
                args.model,
                tr_cfg=train_cfg
            )
    except (AttributeError, TypeError):
        # If safe_globals context manager doesn't exist, try add_safe_globals
        try:
            torch.serialization.add_safe_globals([getattr])
            model = LitModel.load_from_checkpoint(
                args.model,
                tr_cfg=train_cfg
            )
        except (AttributeError, TypeError):
            # Fallback: temporarily monkey-patch torch.load to use weights_only=False
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            try:
                model = LitModel.load_from_checkpoint(
                    args.model,
                    tr_cfg=train_cfg
                )
            finally:
                torch.load = original_load
    except Exception as e:
        if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
            # Last resort: monkey-patch torch.load
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            try:
                model = LitModel.load_from_checkpoint(
                    args.model,
                    tr_cfg=train_cfg
                )
            finally:
                torch.load = original_load
        else:
            raise
    
    print("✓ Model loaded")
    
    # Create dataset
    print(f"\nLoading {args.split} dataset (cell_type={args.cell_type})...")
    dataset = MPRALegNetDataset(
        path_to_data=args.data_path,
        cell_type=args.cell_type,
        split=args.split,
        reverse_complement=False  # Don't augment test data
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Get predictions and actuals using simple forward pass
    predictions, actuals = get_predictions_and_actuals(
        model, train_cfg, dataset, device=device_str, batch_size=args.batch_size
    )
    
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
    
    # Create filename from model name and cell type
    model_name = Path(args.model).stem
    predictions_file = output_dir / f"legnet_{model_name}_{args.cell_type}_{args.split}_predictions.csv"
    metrics_file = output_dir / f"legnet_{model_name}_{args.cell_type}_{args.split}_metrics.csv"
    
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
