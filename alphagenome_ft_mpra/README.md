# Source

Core utilities and classes for AlphaGenome MPRA finetuning.

* **`mpra_heads.py`** - MPRA head implementations (MPRAHead, EncoderMPRAHead, DeepSTARRHead) with flexible architecture access.
* **`data.py`** - Dataset and DataLoader classes (LentiMPRADataset, MPRADataLoader, DeepSTARRDataset, STARRSeqDataLoader, EpisomalMPRADataset, PlantStarrSeqDataset).
* **`plant_starrseq_utils.py`** - Shared helpers for the plant STARR-seq (Jores 2021) dataset: construct assembly, the three data modes, split-aware loader, and the dataset builder (rebuilds the Jores21 enrichment tables from the paper's GitHub data). Pure pandas/numpy — no JAX/torch.
* **`plant_torch.py`** - PyTorch building blocks for the plant STARR-seq runner scripts (torch dataset + collates, attention/mean-pool MPRA heads, the from-scratch Jores CNN, ridge-probe helpers). Torch-guarded like `enf_utils.py`.
* **`training.py`** - Training loop functions (train, validate). Stage 1 uses `alphagenome_ft.optimizer_utils.create_optimizer` with heads-only masking when `freeze_except_head` was called on the model (see `alphagenome-ft` README / `docs/finetune_head_only.md`).
* **`enf_utils.py`** - Utilities for fine-tuning Enformer model for MPRA activity prediction.
* **`seq_loader.py`** - Sequence loader for extracting DNA sequences from reference genomes (hg19/hg38).
* **`__init__.py`** - Package initialization and exports.
