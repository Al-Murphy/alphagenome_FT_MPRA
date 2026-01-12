# Source Utilities for AlphaGenome Finetuning

This package provides utility classes and functions for finetuning AlphaGenome models on MPRA data.

## Components

* **`mpra_heads.py`** - Custom head implementations (`MPRAHead`, `EncoderMPRAHead`)
* **`data.py`** - Dataset and DataLoader classes (`LentiMPRADataset`, `MPRADataLoader`)
* **`training.py`** - Training loop functions (`train`, `validate`, `train_epoch`)