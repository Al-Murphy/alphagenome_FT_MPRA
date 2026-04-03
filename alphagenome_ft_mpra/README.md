# Source

Core utilities and classes for AlphaGenome MPRA finetuning.

* **`mpra_heads.py`** - MPRA head implementations (MPRAHead, EncoderMPRAHead, DeepSTARRHead) with flexible architecture access.
* **`data.py`** - Dataset and DataLoader classes (LentiMPRADataset, MPRADataLoader, DeepSTARRDataset, STARRSeqDataLoader).
* **`training.py`** - Training loop functions (train, validate). Stage 1 uses `alphagenome_ft.optimizer_utils.create_optimizer` with heads-only masking when `freeze_except_head` was called on the model (see `alphagenome-ft` README / `docs/finetune_head_only.md`).
* **`enf_utils.py`** - Utilities for fine-tuning Enformer model for MPRA activity prediction.
* **`seq_loader.py`** - Sequence loader for extracting DNA sequences from reference genomes (hg19/hg38).
* **`__init__.py`** - Package initialization and exports.
