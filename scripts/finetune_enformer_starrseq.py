"""
Finetune Enformer with DeepSTARR head on DeepSTARR dataset using PyTorch Lightning.

This script uses the same optimal hyperparameters from starrseq.json config
and the same dataset class (DeepSTARRDatasetPyTorch) for fair comparison.

USAGE:
    python scripts/finetune_enformer_starrseq.py --config configs/starrseq.json
"""

import argparse
import json
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningDataModule
import numpy as np
from typing import Optional

# Import Enformer utilities (direct import to avoid JAX dependencies in src/__init__.py)
enf_utils_path = Path(__file__).parent.parent / 'src' / 'enf_utils.py'
spec = importlib.util.spec_from_file_location("enf_utils", enf_utils_path)
enf_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enf_utils)
EncoderDeepSTARRHead = enf_utils.EncoderDeepSTARRHead
DeepSTARRDatasetPyTorch = enf_utils.DeepSTARRDatasetPyTorch

from enformer_pytorch import from_pretrained


class PyTorchDeepSTARRDataset(torch.utils.data.Dataset):
    """PyTorch wrapper for DeepSTARRDatasetPyTorch that converts numpy arrays to PyTorch tensors."""
    
    def __init__(self, dataset: DeepSTARRDatasetPyTorch):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Convert numpy array to PyTorch tensor
        # seq is one-hot encoded: (length, 4)
        seq = torch.from_numpy(sample['seq']).float()
        # labels are (2,) array: [dev_activity, hk_activity]
        labels = torch.from_numpy(sample['y']).float()
        
        return seq, labels


class DeepSTARRDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for DeepSTARR dataset."""
    
    def __init__(
        self,
        dummy_model,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        random_shift: bool = True,
        random_shift_likelihood: float = 0.5,
        max_shift: int = 25,
        reverse_complement: bool = True,
    ):
        super().__init__()
        self.dummy_model = dummy_model
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_shift = random_shift
        self.random_shift_likelihood = random_shift_likelihood
        self.max_shift = max_shift
        self.reverse_complement = reverse_complement
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Create datasets for train/val/test."""
        if stage == 'fit' or stage is None:
            self.train_dataset = PyTorchDeepSTARRDataset(
                DeepSTARRDatasetPyTorch(
                    model=self.dummy_model,
                    path_to_data=self.data_path,
                    split='train',
                    random_shift=self.random_shift,
                    random_shift_likelihood=self.random_shift_likelihood,
                    max_shift=self.max_shift,
                    reverse_complement=self.reverse_complement,
                )
            )
            self.val_dataset = PyTorchDeepSTARRDataset(
                DeepSTARRDatasetPyTorch(
                    model=self.dummy_model,
                    path_to_data=self.data_path,
                    split='val',
                    random_shift=False,
                    reverse_complement=False,
                )
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = PyTorchDeepSTARRDataset(
                DeepSTARRDatasetPyTorch(
                    model=self.dummy_model,
                    path_to_data=self.data_path,
                    split='test',
                    random_shift=False,
                    reverse_complement=False,
                )
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    @staticmethod
    def _collate_fn(batch):
        """Collate function to pad sequences to same length."""
        sequences, labels = zip(*batch)
        labels = torch.stack(labels)  # (batch, 2)
        
        # Find max length
        max_len = max(seq.shape[0] for seq in sequences)
        seq_dim = sequences[0].shape[1]  # Should be 4 for one-hot
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if seq.shape[0] < max_len:
                pad = torch.zeros(max_len - seq.shape[0], seq_dim)
                padded = torch.cat([seq, pad], dim=0)
            else:
                padded = seq
            padded_sequences.append(padded)
        
        # Stack into batch: (batch, length, 4)
        seq_batch = torch.stack(padded_sequences, dim=0)
        
        return seq_batch, labels


class EnformerDeepSTARRLightning(pl.LightningModule):
    """PyTorch Lightning module for Enformer DeepSTARR fine-tuning."""
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        optimizer: str = 'adam',
        gradient_clip: Optional[float] = None,
        lr_scheduler: Optional[str] = None,
        second_stage_lr: Optional[float] = None,
        second_stage_epochs: int = 50,
        num_epochs: int = 100,
        nl_size: int = 512,
        do: float = 0.5,
        activation: str = 'relu',
        pooling_type: str = 'flatten',
        center_bp: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Parse nl_size (can be int or comma-separated string)
        if isinstance(nl_size, str):
            nl_size = [int(x.strip()) for x in nl_size.split(',')]
        elif isinstance(nl_size, int):
            nl_size = [nl_size]
        
        # Load pretrained Enformer
        print("Loading pretrained Enformer...")
        enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=False)
        
        # Create DeepSTARR head (matches enf_utils.py design)
        self.model = EncoderDeepSTARRHead(
            enformer=enformer,
            num_tracks=2,  # Developmental and housekeeping
            nl_size=nl_size,
            do=do,
            activation=activation,
            pooling_type=pooling_type,
            center_bp=center_bp,
        )
        
        # Store training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.gradient_clip = gradient_clip
        self.lr_scheduler = lr_scheduler
        self.second_stage_lr = second_stage_lr
        self.second_stage_epochs = second_stage_epochs
        self.num_epochs = num_epochs
        
        # Track stage
        self.current_stage = 1
        
        # Freeze Enformer encoder initially (Stage 1)
        self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze Enformer encoder (Stage 1: train head only)."""
        for param in self.model.enformer.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen (Stage 1)")
    
    def _unfreeze_encoder(self):
        """Unfreeze Enformer encoder (Stage 2: train full model)."""
        for param in self.model.enformer.parameters():
            param.requires_grad = True
        print("✓ Encoder unfrozen (Stage 2)")
    
    def forward(self, seq_batch):
        """Forward pass for batched sequences.
        
        Args:
            seq_batch: (batch, length, 4) tensor of one-hot sequences
            
        Returns:
            (batch, 2) tensor: [dev_activity, hk_activity]
        """
        freeze = (self.current_stage == 1)
        preds = self.model(seq_batch, freeze_enformer=freeze)
        return preds
    
    def training_step(self, batch, batch_idx):
        seq, y = batch  # y is (batch, 2)
        y_pred = self.forward(seq)  # (batch, 2)
        loss = F.mse_loss(y_pred, y)
        
        with torch.no_grad():
            # Compute Pearson correlation for each output
            dev_pearson = self._pearson_corr(y_pred[:, 0], y[:, 0])
            hk_pearson = self._pearson_corr(y_pred[:, 1], y[:, 1])
            mean_pearson = (dev_pearson + hk_pearson) / 2
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_pearson', mean_pearson, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_dev_pearson', dev_pearson, on_step=False, on_epoch=True)
        self.log('train_hk_pearson', hk_pearson, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        seq, y = batch
        y_pred = self.forward(seq)
        loss = F.mse_loss(y_pred, y)
        
        with torch.no_grad():
            dev_pearson = self._pearson_corr(y_pred[:, 0], y[:, 0])
            hk_pearson = self._pearson_corr(y_pred[:, 1], y[:, 1])
            mean_pearson = (dev_pearson + hk_pearson) / 2
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pearson', mean_pearson, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dev_pearson', dev_pearson, on_step=False, on_epoch=True)
        self.log('val_hk_pearson', hk_pearson, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        seq, y = batch
        y_pred = self.forward(seq)
        loss = F.mse_loss(y_pred, y)
        
        with torch.no_grad():
            dev_pearson = self._pearson_corr(y_pred[:, 0], y[:, 0])
            hk_pearson = self._pearson_corr(y_pred[:, 1], y[:, 1])
            mean_pearson = (dev_pearson + hk_pearson) / 2
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pearson', mean_pearson, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_dev_pearson', dev_pearson, on_step=False, on_epoch=True, logger=True)
        self.log('test_hk_pearson', hk_pearson, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def _pearson_corr(self, x, y):
        """Compute Pearson correlation coefficient."""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
        return numerator / (denominator + 1e-8)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Select optimizer
        # Note: PyTorch optimizers automatically ignore parameters with requires_grad=False
        # For flatten pooling, to_tracks will be created on first forward pass
        if self.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay if self.weight_decay else 0.0
            )
        else:  # adam
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay if self.weight_decay else 0.0
            )
        
        # Configure learning rate scheduler
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
        elif self.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                }
            }
        else:
            return optimizer


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def one_hot_encode_dna(seq: str, force: bool = False):
    """
    One-hot encode DNA sequence (from clg/src/dataset_utils.py with modifications).
    
    Args:
        seq: DNA sequence string
        force: If True, allow non-ACGT characters
        
    Returns:
        One-hot encoded numpy array of shape (length, 4)
    """
    dna_bases = np.array(['A', 'C', 'G', 'T'])
    seq = seq.upper()
    
    # Handle empty sequences
    if not seq:
        return np.zeros((0, len(dna_bases)), dtype=np.float32)
    
    all_one_hot = []
    for seq_i in seq:
        if not force:
            if seq_i not in dna_bases:
                # If invalid base and not forcing, replace with N (encoded as all zeros)
                one_hot = np.zeros((1, len(dna_bases)), dtype=np.float32)
            else:
                one_hot = np.zeros((1, len(dna_bases)), dtype=np.float32)
                one_hot[0, np.where(seq_i == dna_bases)] = 1
        else:
            # Force mode - just use zeros for non-ACGT bases
            one_hot = np.zeros((1, len(dna_bases)), dtype=np.float32)
            matches = np.where(seq_i == dna_bases)[0]
            if len(matches) > 0:
                one_hot[0, matches] = 1
        
        all_one_hot.append(one_hot)
    
    # Check if we have valid data to concatenate
    if not all_one_hot:
        return np.zeros((0, len(dna_bases)), dtype=np.float32)
    
    try:
        all_one_hot = np.concatenate(all_one_hot, axis=0)
    except Exception as e:
        # If concatenation fails, create a safe fallback
        print(f"Error in one_hot_encode_dna concatenation: {e}. Creating fallback encoding.")
        return np.zeros((len(seq), len(dna_bases)), dtype=np.float32)
    
    # Return as (length, 4) format
    return all_one_hot.astype(np.float32)


def create_dummy_model_for_dataset():
    """Create a dummy model for dataset initialization (needed for DeepSTARRDatasetPyTorch).
    
    Uses one_hot_encode_dna function (based on Tangermeme approach with modifications)
    instead of AlphaGenome's encoder.
    """
    class CustomOneHotEncoder:
        """Wrapper around one_hot_encode_dna to match AlphaGenome interface."""
        def encode(self, sequence: str):
            """One-hot encode a DNA sequence.
            
            Args:
                sequence: DNA sequence string
                
            Returns:
                numpy array of shape (length, 4) with one-hot encoding
            """
            return one_hot_encode_dna(sequence, force=True)
    
    class DummyModel:
        def __init__(self):
            self._one_hot_encoder = CustomOneHotEncoder()
    
    return DummyModel()


def main():
    parser = argparse.ArgumentParser(
        description='Finetune Enformer with DeepSTARR head on DeepSTARR dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    parser.add_argument('--data_path', type=str, default='./data/deepstarr')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['plateau', 'cosine'])
    parser.add_argument('--second_stage_lr', type=float, default=1e-5)
    parser.add_argument('--second_stage_epochs', type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='./results/models/checkpoints/enformer/')
    parser.add_argument('--wandb_project', type=str, default='enformer-deepstarr')
    parser.add_argument('--wandb_name', type=str, default="enformer-deepstarr-optimal-do05")
    parser.add_argument('--no_wandb', action='store_true')
    
    # Model parameters
    parser.add_argument('--nl_size', type=str, default='512,512', help='Hidden layer sizes: single int or comma-separated list')
    parser.add_argument('--do', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'])
    parser.add_argument('--pooling_type', type=str, default='flatten', choices=['flatten', 'mean', 'sum', 'max', 'center'])
    parser.add_argument('--center_bp', type=int, default=256, help='Center bp for non-flatten pooling')
    
    # Data augmentation parameters
    parser.add_argument('--random_shift', action='store_true', default=True)
    parser.add_argument('--random_shift_likelihood', type=float, default=0.5)
    parser.add_argument('--max_shift', type=int, default=25)
    parser.add_argument('--reverse_complement', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
        
        if 'data' in config:
            data_config = config['data']
            args.data_path = data_config.get('data_path', args.data_path)
            args.batch_size = data_config.get('batch_size', args.batch_size)
            args.random_shift = data_config.get('random_shift', args.random_shift)
            args.random_shift_likelihood = data_config.get('random_shift_likelihood', args.random_shift_likelihood)
            args.max_shift = data_config.get('max_shift', args.max_shift)
            args.reverse_complement = data_config.get('reverse_complement', args.reverse_complement)
        
        if 'model' in config:
            model_config = config['model']
            args.center_bp = model_config.get('center_bp', args.center_bp)
            args.pooling_type = model_config.get('pooling_type', args.pooling_type)
            args.nl_size = str(model_config.get('nl_size', args.nl_size))
            args.do = model_config.get('do', args.do)
            args.activation = model_config.get('activation', args.activation)
        
        if 'training' in config:
            train_config = config['training']
            args.num_epochs = train_config.get('num_epochs', args.num_epochs)
            args.learning_rate = train_config.get('learning_rate', args.learning_rate)
            args.optimizer = train_config.get('optimizer', args.optimizer)
            args.weight_decay = train_config.get('weight_decay', args.weight_decay)
            args.gradient_clip = train_config.get('gradient_clip', args.gradient_clip)
            args.lr_scheduler = train_config.get('lr_scheduler', args.lr_scheduler)
            args.early_stopping_patience = train_config.get('early_stopping_patience', args.early_stopping_patience)
        
        if 'two_stage' in config:
            two_stage_config = config['two_stage']
            if two_stage_config.get('enabled', False):
                args.second_stage_lr = two_stage_config.get('second_stage_lr', args.second_stage_lr)
                args.second_stage_epochs = two_stage_config.get('second_stage_epochs', args.second_stage_epochs)
        
        if 'wandb' in config:
            wandb_config = config['wandb']
            if not args.no_wandb:
                args.wandb_project = wandb_config.get('project', args.wandb_project)
                if args.wandb_name is None:
                    args.wandb_name = wandb_config.get('wandb_name', 'enformer-deepstarr')
        print("✓ Config loaded")
    
    if args.wandb_name is None:
        args.wandb_name = 'enformer-deepstarr'
    
    # Create dummy model for dataset
    dummy_model = create_dummy_model_for_dataset()
    
    # Create data module
    print(f"\nSetting up DeepSTARR data module...")
    data_module = DeepSTARRDataModule(
        dummy_model=dummy_model,
        data_path=args.data_path,
        batch_size=args.batch_size,
        random_shift=args.random_shift,
        random_shift_likelihood=args.random_shift_likelihood,
        max_shift=args.max_shift,
        reverse_complement=args.reverse_complement,
    )
    data_module.setup()
    print(f"✓ Train: {len(data_module.train_dataset)} samples")
    print(f"✓ Val:   {len(data_module.val_dataset)} samples")
    print(f"✓ Test:  {len(data_module.test_dataset)} samples")
    
    # Create model
    print("\nCreating Enformer DeepSTARR model...")
    model = EnformerDeepSTARRLightning(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay if args.weight_decay else 0.0,
        optimizer=args.optimizer,
        gradient_clip=args.gradient_clip,
        lr_scheduler=args.lr_scheduler,
        second_stage_lr=args.second_stage_lr,
        second_stage_epochs=args.second_stage_epochs,
        num_epochs=args.num_epochs,
        nl_size=args.nl_size,
        do=args.do,
        activation=args.activation,
        pooling_type=args.pooling_type,
        center_bp=args.center_bp,
    )
    print("✓ Model created")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Model checkpointing
    checkpoint_dir = Path(args.checkpoint_dir) / "deepstarr" / args.wandb_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Setup logger
    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'optimizer': args.optimizer,
                'weight_decay': args.weight_decay,
                'learning_rate': args.learning_rate,
                'lr_scheduler': args.lr_scheduler,
                'two_stage': args.second_stage_lr is not None,
                'second_stage_lr': args.second_stage_lr,
                'nl_size': args.nl_size,
                'do': args.do,
                'activation': args.activation,
                'pooling_type': args.pooling_type,
            }
        )
    
    # Train Stage 1
    print("\n" + "="*80)
    print("Starting Stage 1 Training (Frozen Encoder)")
    print("="*80)
    
    trainer_stage1 = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        log_every_n_steps=10,
    )
    
    trainer_stage1.fit(model, data_module)
    
    # Stage 2 training (if enabled)
    if args.second_stage_lr:
        print("\n" + "="*80)
        print("TRANSITIONING TO STAGE 2")
        print("="*80)
        
        # Load best Stage 1 checkpoint
        if checkpoint_callback.best_model_path:
            print(f"Loading best Stage 1 checkpoint: {checkpoint_callback.best_model_path}")
            model = EnformerDeepSTARRLightning.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                learning_rate=args.second_stage_lr,
                second_stage_lr=args.second_stage_lr,
                second_stage_epochs=args.second_stage_epochs,
                num_epochs=args.num_epochs,
                nl_size=args.nl_size,
                do=args.do,
                activation=args.activation,
                pooling_type=args.pooling_type,
                center_bp=args.center_bp,
            )
            model.current_stage = 2
            model._unfreeze_encoder()
        
        # Create new checkpoint callback for Stage 2
        checkpoint_dir_stage2 = checkpoint_dir / "stage2"
        checkpoint_dir_stage2.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback_stage2 = ModelCheckpoint(
            dirpath=str(checkpoint_dir_stage2),
            filename='best-stage2-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=True,
        )
        
        callbacks_stage2 = [
            EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience, mode='min'),
            checkpoint_callback_stage2,
            LearningRateMonitor(logging_interval='epoch'),
        ]
        
        trainer_stage2 = pl.Trainer(
            max_epochs=args.second_stage_epochs,
            callbacks=callbacks_stage2,
            logger=logger,
            gradient_clip_val=args.gradient_clip,
            accelerator='gpu',
            devices=1,
            precision='16-mixed',
            log_every_n_steps=10,
        )
        
        print("Starting Stage 2 Training (Unfrozen Encoder)")
        trainer_stage2.fit(model, data_module)
        trainer = trainer_stage2
    else:
        trainer = trainer_stage1
    
    # Test
    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80)
    
    # Run test - metrics should be logged automatically via self.log() in test_step
    test_results = trainer.test(model, data_module)
    
    # Explicitly log test metrics to WandB if logger is available
    if logger is not None and test_results and len(test_results) > 0:
        test_metrics = test_results[0]
        logger.log_metrics(test_metrics, step=trainer.global_step)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    if args.second_stage_lr and checkpoint_callback_stage2.best_model_path:
        print(f"Best Stage 2 checkpoint: {checkpoint_callback_stage2.best_model_path}")
    else:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
