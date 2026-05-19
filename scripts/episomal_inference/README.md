# Episomal MPRA — inference & aggregation pipeline

End-to-end pipeline to reproduce the six-model bar plot
(`assets/episomal_bar_results_may19.csv` →
`scripts/plot_episomal_benchmark_results.py`) on the **standardized** Gosai
episomal MPRA test sets:

| Panel | Test set | n | Source file |
|---|---|---:|---|
| Genomic Reference | chr 7+13, allele=R | 32,831 | `test_chr7_13_all.tsv` |
| Genomic Alternate | chr 7+13, allele=A | 32,860 | `test_chr7_13_all.tsv` |
| High-Activity Designed | OOD designed | 22,962 | `test_ood_designed_{cell}.tsv` |
| SNV Effects (Δ Pearson) | mono-allelic 30k subset | 29,636 | `test_snv_pairs.tsv` filtered to `Alt_*`-free IDs |

## Step 1 — Train models

Use the canonical training scripts:

```bash
# AG MPRA (Probing): produced as Stage 1 of `finetune_episomal_mpra.py`
# AG MPRA (Fine-tuned, joint): finetune_episomal_mpra.py with --joint --multitask
# Enf MPRA (Probing & Fine-tuned UF=all): finetune_enformer_episomal_mpra.py
#   with --second_stage_lr 5e-5 --gradient_clip 0.5
# DREAM-RNN, Malinois: trained in ALBench-S2F (https://github.com/trchristensen-99/ALBench-S2F)
```

The relevant SLURM scripts are under `scripts/slurm/`.

## Step 2 — Standardized inference

Each script writes per-cell predictions to `outputs/proper_eval/{model}/{cell}/seed{N}/...`.

| Script | Model | Notes |
|---|---|---|
| `eval_enformer_proper_panels_sharded.py` | Enf FT (UF=all) + Enf Probing | sharded (`--shard-idx`, `--total-shards`) so each job processes a slice of test seqs; chained back together by `aggregate_sharded.py` |
| `eval_enformer_proper_panels_fast.py` | Enf single-shard variant | one job per (cell, seed), `bs=16`, bf16 — slower than sharded but simpler |
| `eval_enformer_snv_shard.py` | Enf 45k SNV-pair only | needed for Enf Probing where the proper-panels script doesn't cover SNV |
| `eval_dream_rnn_members.py` | DREAM-RNN | saves all 3 ensemble members' predictions per (cell, seed); aggregator picks the **best non-broken** single member by 32k Ref Pearson |
| `eval_dream_rnn_all_panels.py` | DREAM-RNN ensemble | (deprecated — kept for comparison; bar plot uses single non-broken member instead) |
| `eval_ag_joint_full_chr.py` | AG MPRA (Fine-tuned, joint) | JAX Orbax-checkpoint inference on the full 66k chr 7+13 |

Malinois (PyTorch BassetBranched) is evaluated with the existing
`scripts/infer_pytorch_chr7_13.py` from ALBench-S2F.

AG MPRA (Probing) and AG MPRA (Fine-tuned, separate) save their full-chr
predictions during training (`test_predictions.npz` next to each
`result.json`) — no separate inference job needed.

## Step 3 — Aggregate to CSV

```bash
# Walks all saved predictions, computes Pearson r on every panel for every
# (model, cell, seed), and writes outputs/all_panels_aggregated.csv with
# columns: model, cell, seed, panel, pearson_r.
python scripts/episomal_inference/aggregate_all_panels.py

# Aggregates sharded Enf FT/Probing + DREAM-RNN per-member into a single
# CSV with the panels expected by the bar-plot script
# (test_set ∈ {reference, snv_abs, snv, designed}).
python scripts/episomal_inference/aggregate_sharded.py
```

The bar-plot CSV ends up at `assets/episomal_bar_results_may19.csv`.

## Step 4 — Bar plot

```bash
python scripts/plot_episomal_benchmark_results.py \
  --results_csv assets/episomal_bar_results_may19.csv \
  --style cell_averaged \
  --output_dir results --output_name episomal_bar_may19
```

## Notes on panel choices

- **30k mono SNV Δ** (not 45k): the full 45k SNV-pair set includes
  ~15k multi-allelic context-expansion rows (744 variants × multiple
  haplotypes). These artificially inflate Δ Pearson by ~0.30. The mono
  subset (IDs without an `Alt_` tag) is the canonical SNV pair set.
- **DREAM-RNN single non-broken member**: the trained DREAM-RNN
  checkpoint is a 3-member ensemble (per the original recipe). For a
  fair comparison against single-model archs, we pick the member with
  the highest 32k Ref Pearson per (cell, seed). This drops the rare
  collapsed seeds (e.g. `sknsh/seed_0/member_0` → Pearson −0.002)
  without ensembling.
- **No hashfrag-filtered subsets**: every panel above uses the full
  chr 7+13 or full 45k SNV-pair test sets (or the 30k mono subset for
  Δ). The earlier `*_hashfrag.tsv` files are not part of this pipeline.
