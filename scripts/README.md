# Scripts

Executable scripts for training, evaluation, benchmarking, and data processing.

* **`test_cagi5_zero_shot_base.py`** - Zero-shot evaluation of base AlphaGenome model on CAGI5 saturation mutagenesis benchmark.
* **`test_cagi5_zero_shot_mpra.py`** - Zero-shot evaluation of fine-tuned MPRA models on CAGI5 benchmark.
* **`finetune_mpra.py`** - Finetune AlphaGenome with MPRA head on LentiMPRA dataset.
* **`finetune_enformer_mpra.py`** - Finetune Enformer model with MPRA head on LentiMPRA dataset using PyTorch Lightning.
* **`finetune_starrseq.py`** - Finetune AlphaGenome with DeepSTARR head on DeepSTARR dataset.
* **`test_ft_model_mpra.py`** - Test fine-tuned MPRA model on test data.
* **`test_ft_model_enformer_mpra.py`** - Test fine-tuned Enformer MPRA model on test data.
* **`test_ft_model_starrseq.py`** - Test fine-tuned DeepSTARR model on test data.
* **`test_mpralegnet.py`** - Test LegNet MPRA model on test data.
* **`compute_attributions.py`** - Compute attribution maps and sequence logos for MPRA sequences using DeepSHAP, gradients, or gradient×input methods. Supports single sequences or top N sequences by activity.
* **`create_mpra_comparison_table.py`** - Create comparison table of MPRA model performance across models and cell lines.
* **`plot_cagi5_results.py`** - Generate violin plots comparing models on CAGI5 benchmark.
* **`fetch_cagi5_references.py`** - Fetch reference sequences for CAGI5 elements from UCSC genome browser.
* **`cache_embeddings.py`** - Pre-compute and cache MPRA encoder embeddings for faster training.
* **`cache_deepstarr_embeddings.py`** - Pre-compute and cache DeepSTARR encoder embeddings for faster training.
* **`generate_benchmark_jobs.py`** - Generate SLURM job scripts for hyperparameter sweeps.
* **`collate_benchmark_results.py`** - Collate benchmark results from all cell types into summary table.
* **`regenerate_benchmark_results.py`** - Regenerate benchmark results CSVs from WandB run history.
