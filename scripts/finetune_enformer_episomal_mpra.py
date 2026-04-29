"""Finetune Enformer with MPRA head on episomal MPRA (Gosai et al. 2024).

Mirrors `finetune_enformer_mpra.py` but swaps in the episomal dataset class and
chr-based splits. Uses the conv-only Enformer encoder defined in
`alphagenome_ft_mpra.enf_utils.EncoderMPRAHead` (transformer/crop_final/
final_pointwise replaced with Identity), so each ~30min training run on H100 is
much faster than full-stack Enformer fine-tuning.

The existing Enformer head expects an input that produces 3 conv-tower bins
(matches the 281bp lentiMPRA construct). We pad each 200bp episomal sequence to
281bp (pad_n_bases=81 default) so that the same head is reused without
modification.

USAGE:
    python scripts/finetune_enformer_episomal_mpra.py \\
        --cell_type K562 --config configs/episomal_K562.json
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# Load enf_utils and episomal_data as modules (mirrors finetune_enformer_mpra.py).
# Direct-loading sidesteps `alphagenome_ft_mpra/__init__.py`, which would otherwise
# import JAX — keeping the Enformer training path PyTorch-only.
def _load_module(name: str, relpath: str):
    path = Path(__file__).parent.parent / "alphagenome_ft_mpra" / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_enf_utils = _load_module("enf_utils", "enf_utils.py")
EncoderMPRAHead = _enf_utils.EncoderMPRAHead

_episomal_data = _load_module("episomal_data", "episomal_data.py")
EpisomalMPRADatasetPyTorch = _episomal_data.EpisomalMPRADatasetPyTorch

from enformer_pytorch import from_pretrained  # noqa: E402


# Pad 200bp Gosai sequence to match the 281bp lentiMPRA construct length so the
# Enformer conv tower yields 3 bins (matching EncoderMPRAHead's LayerNorm dim).
EPISOMAL_PAD_TO_LENTI_CONSTRUCT = 81


class _PyTorchEpisomalDataset(torch.utils.data.Dataset):
    """Wraps EpisomalMPRADatasetPyTorch to produce torch tensors."""

    def __init__(self, dataset: EpisomalMPRADatasetPyTorch):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        seq = torch.from_numpy(sample["seq"]).float()
        label = torch.tensor(float(sample["y"]), dtype=torch.float32)
        return seq, label


class EpisomalMPRADataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        cell_type: str,
        batch_size: int = 64,
        num_workers: int = 4,
        random_shift: bool = True,
        random_shift_likelihood: float = 0.5,
        max_shift: int = 10,
        reverse_complement: bool = True,
        pad_n_bases: int = EPISOMAL_PAD_TO_LENTI_CONSTRUCT,
    ):
        super().__init__()
        self.data_path = data_path
        self.cell_type = cell_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_shift = random_shift
        self.random_shift_likelihood = random_shift_likelihood
        self.max_shift = max_shift
        self.reverse_complement = reverse_complement
        self.pad_n_bases = pad_n_bases

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        common = dict(
            path_to_data=self.data_path,
            cell_type=self.cell_type,
            pad_n_bases=self.pad_n_bases,
        )
        if stage in (None, "fit"):
            self.train_dataset = _PyTorchEpisomalDataset(
                EpisomalMPRADatasetPyTorch(
                    split="train",
                    random_shift=self.random_shift,
                    random_shift_likelihood=self.random_shift_likelihood,
                    max_shift=self.max_shift,
                    reverse_complement=self.reverse_complement,
                    **common,
                )
            )
            self.val_dataset = _PyTorchEpisomalDataset(
                EpisomalMPRADatasetPyTorch(
                    split="val",
                    random_shift=False,
                    reverse_complement=False,
                    **common,
                )
            )
        if stage in (None, "test"):
            self.test_dataset = _PyTorchEpisomalDataset(
                EpisomalMPRADatasetPyTorch(
                    split="test",
                    random_shift=False,
                    reverse_complement=False,
                    **common,
                )
            )

    @staticmethod
    def _collate_fn(batch):
        sequences, labels = zip(*batch)
        labels = torch.stack(labels)
        max_len = max(seq.shape[0] for seq in sequences)
        seq_dim = sequences[0].shape[1]
        padded = []
        for seq in sequences:
            if seq.shape[0] < max_len:
                pad = torch.zeros(max_len - seq.shape[0], seq_dim)
                padded.append(torch.cat([seq, pad], dim=0))
            else:
                padded.append(seq)
        return torch.stack(padded, dim=0), labels

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


class EnformerEpisomalLightning(pl.LightningModule):
    """Same logic as `EnformerMPRALightning`, kept local to avoid circular imports."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        optimizer: str = "adam",
        gradient_clip: Optional[float] = None,
        lr_scheduler: Optional[str] = None,
        second_stage_lr: Optional[float] = None,
        second_stage_epochs: int = 50,
        num_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        print("Loading pretrained Enformer (conv-only)...")
        enformer = from_pretrained("EleutherAI/enformer-official-rough", use_tf_gamma=False)
        self.model = EncoderMPRAHead(enformer=enformer, num_tracks=1)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.gradient_clip = gradient_clip
        self.lr_scheduler = lr_scheduler
        self.second_stage_lr = second_stage_lr
        self.second_stage_epochs = second_stage_epochs
        self.num_epochs = num_epochs

        self.current_stage = 1
        self._freeze_encoder()

    def _freeze_encoder(self):
        for p in self.model.enformer.parameters():
            p.requires_grad = False
        print("✓ Encoder frozen (Stage 1)")

    def _unfreeze_encoder(self):
        for p in self.model.enformer.parameters():
            p.requires_grad = True
        print("✓ Encoder unfrozen (Stage 2)")

    def forward(self, seq):
        freeze = self.current_stage == 1
        preds = self.model(seq, freeze_enformer=freeze)
        if preds.dim() > 1 and preds.shape[-1] == 1:
            return preds.squeeze(-1)
        return preds

    @staticmethod
    def _pearson(x, y):
        xc = x - x.mean()
        yc = y - y.mean()
        return (xc * yc).sum() / (
            torch.sqrt((xc**2).sum() * (yc**2).sum()) + 1e-8
        )

    def training_step(self, batch, _):
        seq, y = batch
        y_pred = self.forward(seq)
        loss = F.mse_loss(y_pred, y)
        with torch.no_grad():
            self.log("train_pearson", self._pearson(y_pred, y),
                     on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        seq, y = batch
        y_pred = self.forward(seq)
        loss = F.mse_loss(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pearson", self._pearson(y_pred, y),
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        seq, y = batch
        y_pred = self.forward(seq)
        loss = F.mse_loss(y_pred, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_pearson", self._pearson(y_pred, y),
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_cls = torch.optim.AdamW if self.optimizer_name.lower() == "adamw" else torch.optim.Adam
        optimizer = opt_cls(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay or 0.0,
        )
        if self.lr_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=self.learning_rate * 0.01
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        if self.lr_scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": sched, "monitor": "val_loss",
                                     "interval": "epoch"}}
        return optimizer


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Finetune Enformer with MPRA head on Gosai episomal MPRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cell_type", type=str, default="K562",
                        choices=["K562", "HepG2", "SKNSH"])
    parser.add_argument("--data_path", type=str, default="./data/gosai_episomal")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gradient_clip", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        choices=["plateau", "cosine"])
    parser.add_argument("--second_stage_lr", type=float, default=None)
    parser.add_argument("--second_stage_epochs", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--max_shift", type=int, default=10)
    parser.add_argument("--pad_n_bases", type=int,
                        default=EPISOMAL_PAD_TO_LENTI_CONSTRUCT)
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./results/models/checkpoints/episomal_enformer/")
    parser.add_argument("--wandb_project", type=str, default="enformer-episomal-mpra")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        if "cell_type" in cfg:
            args.cell_type = cfg["cell_type"]
        if "data_path" in cfg:
            args.data_path = cfg["data_path"]
        if "data" in cfg:
            d = cfg["data"]
            args.batch_size = d.get("batch_size", args.batch_size)
            args.max_shift = d.get("max_shift", args.max_shift)
        if "training" in cfg:
            t = cfg["training"]
            args.num_epochs = t.get("num_epochs", args.num_epochs)
            args.learning_rate = t.get("learning_rate", args.learning_rate)
            args.optimizer = t.get("optimizer", args.optimizer)
            args.weight_decay = t.get("weight_decay", args.weight_decay)
            args.gradient_clip = t.get("gradient_clip", args.gradient_clip)
            args.lr_scheduler = t.get("lr_scheduler", args.lr_scheduler)
            args.early_stopping_patience = t.get(
                "early_stopping_patience", args.early_stopping_patience)
        if "two_stage" in cfg and cfg["two_stage"].get("enabled", False):
            ts = cfg["two_stage"]
            args.second_stage_lr = ts.get("second_stage_lr", args.second_stage_lr)
            args.second_stage_epochs = ts.get(
                "second_stage_epochs", args.second_stage_epochs)
        if "wandb" in cfg and not args.no_wandb:
            wb = cfg["wandb"]
            args.wandb_project = wb.get("project", args.wandb_project)
            if args.wandb_name is None:
                args.wandb_name = wb.get("wandb_name", None)

    if args.wandb_name is None:
        args.wandb_name = f"enformer-episomal-{args.cell_type}-seed{args.seed}"

    pl.seed_everything(args.seed, workers=True)

    print("=" * 80)
    print("Enformer Episomal MPRA Fine-tuning (Gosai 2024)")
    print("=" * 80)
    print(f"Cell type: {args.cell_type} | batch_size: {args.batch_size}")
    print(f"epochs: {args.num_epochs} | lr: {args.learning_rate} | wd: {args.weight_decay}")
    print(f"pad_n_bases: {args.pad_n_bases} (200bp → {200 + args.pad_n_bases}bp)")
    if args.second_stage_lr:
        print(f"Two-stage: stage1={args.num_epochs}ep, "
              f"stage2={args.second_stage_epochs}ep @ {args.second_stage_lr}")
    print("=" * 80)

    # Data
    print(f"\nSetting up Gosai episomal data module for {args.cell_type}...")
    dm = EpisomalMPRADataModule(
        data_path=args.data_path,
        cell_type=args.cell_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_shift=True,
        random_shift_likelihood=0.5,
        max_shift=args.max_shift,
        reverse_complement=True,
        pad_n_bases=args.pad_n_bases,
    )
    dm.setup()
    print(f"✓ Train: {len(dm.train_dataset)} | "
          f"Val: {len(dm.val_dataset)} | Test: {len(dm.test_dataset)}")

    # Model
    print("\nCreating Enformer-MPRA model (conv-only)...")
    model = EnformerEpisomalLightning(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay or 0.0,
        optimizer=args.optimizer,
        gradient_clip=args.gradient_clip,
        lr_scheduler=args.lr_scheduler,
        second_stage_lr=args.second_stage_lr,
        second_stage_epochs=args.second_stage_epochs,
        num_epochs=args.num_epochs,
    )

    # Stage 1 callbacks
    ckpt_dir = Path(args.checkpoint_dir) / args.cell_type / args.wandb_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss",
                      patience=args.early_stopping_patience, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        ckpt_cb,
    ]

    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "cell_type": args.cell_type,
                "data_source": "gosai_episomal",
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "learning_rate": args.learning_rate,
                "lr_scheduler": args.lr_scheduler,
                "two_stage": args.second_stage_lr is not None,
                "second_stage_lr": args.second_stage_lr,
                "seed": args.seed,
            },
        )

    print("\n" + "=" * 80 + "\nStage 1 Training (Frozen Encoder)\n" + "=" * 80)
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip,
        accelerator="gpu",
        devices=1,
        precision=16,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)

    # Stage 2
    if args.second_stage_lr:
        print("\n" + "=" * 80 + "\nStage 2 Training (Unfrozen Encoder)\n" + "=" * 80)
        if ckpt_cb.best_model_path:
            print(f"Loading best Stage 1 checkpoint: {ckpt_cb.best_model_path}")
            model = EnformerEpisomalLightning.load_from_checkpoint(
                ckpt_cb.best_model_path,
                learning_rate=args.second_stage_lr,
                second_stage_lr=args.second_stage_lr,
                second_stage_epochs=args.second_stage_epochs,
                num_epochs=args.num_epochs,
            )
            model.current_stage = 2
            model._unfreeze_encoder()

        ckpt_dir2 = ckpt_dir / "stage2"
        ckpt_dir2.mkdir(parents=True, exist_ok=True)
        ckpt_cb2 = ModelCheckpoint(
            dirpath=str(ckpt_dir2),
            filename="best-stage2-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        callbacks2 = [
            EarlyStopping(monitor="val_loss",
                          patience=args.early_stopping_patience, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
            ckpt_cb2,
        ]
        trainer = pl.Trainer(
            max_epochs=args.second_stage_epochs,
            callbacks=callbacks2,
            logger=logger,
            gradient_clip_val=args.gradient_clip,
            accelerator="gpu",
            devices=1,
            precision=16,
            log_every_n_steps=10,
        )
        trainer.fit(model, dm)

    # Test
    print("\n" + "=" * 80 + "\nEvaluating on Test Set\n" + "=" * 80)
    test_results = trainer.test(model, dm)
    if logger is not None and test_results:
        logger.log_metrics(test_results[0], step=trainer.global_step)

    print("\n✓ Training Complete!")


if __name__ == "__main__":
    main()
