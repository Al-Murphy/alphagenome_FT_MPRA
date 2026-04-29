# Scripts

Executable scripts for training, evaluation, benchmarking, and data processing.

The lists below are organized in the **typical order you would run things** for each analysis.

## 1. LentiMPRA finetuning workflow (AlphaGenome & Enformer)

- **Training**
  - **`finetune_mpra.py`** – Finetune AlphaGenome with MPRA head on the LentiMPRA dataset.
  - **`finetune_enformer_mpra.py`** – Finetune Enformer with MPRA head on LentiMPRA (PyTorch Lightning).

- **Evaluation**
  - **`test_ft_model_mpra.py`** – Evaluate fine-tuned AlphaGenome MPRA models on held‑out test data.
  - **`test_ft_model_enformer_mpra.py`** – Evaluate fine-tuned Enformer MPRA models on held‑out test data.

- **Attributions / interpretation**
  - **`compute_attributions_lentimpra.py`** – Compute attribution maps and sequence logos for LentiMPRA using DeepSHAP, gradients, or gradient×input. Supports single sequences or top‑N sequences by activity.

## 2. Episomal MPRA finetuning workflow (Gosai et al. 2024)

The Gosai dataset contains ~800K 200bp sequences across 3 cell types (K562,
HepG2, SK-N-SH) measured by an episomal MPRA. Splits are chromosome-based:
test = chr7 + chr13, val = chr19 + chr21 + chrX, train = remainder. Three test
sets are evaluated: **Genomic Reference** (chr7/chr13), **High-Activity
Designed** (OOD), and **SNV Effects** (Δ Pearson on alt − ref pairs).

Place the Gosai data at `./data/gosai_episomal/` with these files:
  - `DATA-Table_S2__MPRA_dataset.txt` – main TSV (sequences + per-cell log2FC)
  - `test_sets/test_ood_designed_k562.tsv` *(optional, designed test set)*
  - `test_sets/test_snv_pairs_hashfrag.tsv` *(optional, SNV pair test set)*

- **Training**
  - **`finetune_episomal_mpra.py`** – Finetune AlphaGenome with the MPRA head
    on Gosai episomal data. Mirrors `finetune_mpra.py`; reads
    `configs/episomal_<cell>.json`. Supports two-stage training
    (`--second_stage_lr 1e-5`).
  - **`finetune_enformer_episomal_mpra.py`** – Finetune Enformer with the
    conv-only MPRA head on Gosai episomal data. Pads 200bp sequences to 281bp
    (`--pad_n_bases 81` default) so the encoder yields 3 conv-tower bins —
    matching the existing `EncoderMPRAHead`.

- **Evaluation**
  - **`test_episomal_mpra.py`** – Unified evaluation across the 3 episomal
    test sets. `--model_type` selects the backend
    (`ag_probing`, `ag_finetuned`, `enformer_probing`, `enformer_finetuned`).
    Writes `<run>_<cell>_metrics.json` plus per-test-set predictions CSVs.

- **Plotting**
  - **`plot_episomal_benchmark_results.py`** – Generate the 3-panel bar plot
    (Reference / Designed / SNV) comparing 6 models × 3 cell types. Reads
    metrics produced by `test_episomal_mpra.py`
    (`--metrics_dir`) or a pre-aggregated CSV (`--results_csv`).

## 3. DeepSTARR finetuning workflow (AlphaGenome & Enformer)

- **Training**
  - **`finetune_starrseq.py`** – Finetune AlphaGenome with DeepSTARR head on the DeepSTARR dataset.
  - **`finetune_enformer_starrseq.py`** – Finetune Enformer with DeepSTARR head on the DeepSTARR dataset.

- **Evaluation**
  - **`test_ft_model_starrseq.py`** – Evaluate fine-tuned AlphaGenome DeepSTARR models.
  - **`test_ft_model_enformer_starrseq.py`** – Evaluate fine-tuned Enformer DeepSTARR models.

- **Attributions / interpretation**
  - **`compute_attributions_starrseq.py`** – Compute attribution maps and sequence logos for STARR‑seq using DeepSHAP, gradients, or gradient×input. Supports single sequences or top‑N sequences by activity.

## 4. CAGI5 saturation mutagenesis & zero‑shot analyses

- **Zero‑shot and fine‑tuned CAGI5 evaluations**
  - **`test_cagi5_zero_shot_base.py`** – Zero‑shot evaluation of the **base AlphaGenome model** on the CAGI5 saturation mutagenesis benchmark.
  - **`test_cagi5_zero_shot_mpra.py`** – CAGI5 evaluation of **fine‑tuned AlphaGenome MPRA models**.
  - **`test_cagi5_zero_shot_enformer_mpra.py`** – CAGI5 evaluation of **fine‑tuned Enformer MPRA models**.

- **Comparison tables & plots**
  - **`create_mpra_comparison_table.py`** – Create comparison tables of MPRA model performance across models and cell lines.
  - **`plot_cagi5_results.py`** – Generate violin plots comparing models on the CAGI5 benchmark.
  - **`plot_benchmark_results.py`** – Generate bar plots comparing models on LentiMPRA and STARR‑seq benchmarks.

## 5. Embedding caching, benchmarking, and job management

- **Caching for faster training**
  - **`cache_embeddings.py`** – Pre‑compute and cache AlphaGenome MPRA encoder embeddings.
  - **`cache_deepstarr_embeddings.py`** – Pre‑compute and cache DeepSTARR encoder embeddings.

- **Benchmark sweep utilities**
  - **`generate_benchmark_jobs.py`** – Generate SLURM job scripts for hyperparameter sweeps.
  - **`collate_benchmark_results.py`** – Collate benchmark results from all cell types into a summary table.
  - **`regenerate_benchmark_results.py`** – Regenerate benchmark results CSVs from Weights & Biases (WandB) run history.

## 6. Baselines and helper scripts

- **`test_mpralegnet.py`** – Evaluate the LegNet MPRA baseline model on LentiMPRA test data.
- **`fetch_cagi5_references.py`** – Fetch reference sequences for CAGI5 elements from UCSC Genome Browser.
