---
license: other
license_name: alphagenome
license_link: https://deepmind.google.com/science/alphagenome/model-terms
library_name: alphagenome-ft-mpra
tags:
  - biology
  - genomics
  - dna
  - mpra
  - starr-seq
  - alphagenome
  - regulatory-genomics
---

# AlphaGenome Encoder — fine-tuned MPRA / STARR-seq checkpoints

Fine-tuned **AlphaGenome encoder** checkpoints for massively parallel reporter assays.
The AlphaGenome transformer is bypassed: a regression head is trained on the raw encoder
output (128 bp resolution), which is both far cheaper and — on these short-sequence
reporter tasks — more accurate than using the full model.

Four benchmarks, in both JAX (Haiku) and PyTorch where available:

- **lentiMPRA** (Agarwal et al.) — K562, HepG2, WTC11
- **lentiMPRA** (Gosai et al.) — K562, HepG2, SKNSH
- **Drosophila STARR-seq** (DeepSTARR; de Almeida et al.) — developmental + housekeeping
- **Plant STARR-seq** (Jores et al. 2021) — tobacco leaf and maize protoplast, 3 data modes
- **Plant multi-condition MPRA** (Jores et al. 2026) — 5 conditions (cold/dark/light/warm/maize)

Code: [Al-Murphy/alphagenome_FT_MPRA](https://github.com/Al-Murphy/alphagenome_FT_MPRA)

---

## ⚠️ Licence

These are **fine-tuned derivatives of AlphaGenome**. The model parameters, their outputs,
and any derivatives thereof remain subject to Google DeepMind's
[AlphaGenome Model Terms](https://deepmind.google.com/science/alphagenome/model-terms),
**including the restriction to non-commercial use**. The base parameters were created by
Google DeepMind and are the property of Google LLC.

Only the fine-tuning *code* is Apache-2.0. We are not relicensing the weights.

Loading also requires the base AlphaGenome weights
([`google/alphagenome-all-folds`](https://huggingface.co/google/alphagenome-all-folds)),
which are **access-gated** — accept the terms there and `huggingface-cli login` first.

## Usage

```bash
pip install git+https://github.com/Al-Murphy/alphagenome_FT_MPRA
```

```python
from alphagenome_ft_mpra.hub import list_pretrained, load_pretrained

list_pretrained()

model = load_pretrained('plant-starrseq-leaf-combined')   # JAX, fine-tuned
model = load_pretrained('mpra_K562')                      # PyTorch
preds = model.predict_sequences(['ACGT...'], construct_mode='promoter_barcode')
```

`load_pretrained` reads each checkpoint's `config.json` to build the right head at the
right width — see the repo's [docs/model_weights.md](https://github.com/Al-Murphy/alphagenome_FT_MPRA/blob/main/docs/model_weights.md)
for the manual path and the gotchas.

## Contents

`stage1` = frozen encoder (head only trained); `stage2` = encoder fine-tuned.

### `torch/` — test Pearson r

| Checkpoint | Task | frozen | fine-tuned |
|---|---|---|---|
| `mpra_K562` | lentiMPRA K562 | 0.8528 | **0.8806** |
| `mpra_HepG2` | lentiMPRA HepG2 | 0.8681 | **0.8880** |
| `mpra_WTC11` | lentiMPRA WTC11 | 0.8324 | **0.8411** |
| `starrseq_drosophila` | DeepSTARR (dev + hk) | 0.6184 | **0.7468** |
| `jores_multicondition` | Jores 2026 plant MPRA (5 cond.) | — | **0.841** |

Drosophila is the mean of the two tasks (fine-tuned: dev 0.7193, hk 0.7744).
`jores_multicondition` is the mean across 5 conditions (cold 0.822, dark 0.887,
light 0.873, warm 0.853, maize 0.770). Its head is `MPRAHead(num_outputs=5)`
(`head_type="mpra"`), so it loads with the **standard** `alphagenome-encoder-ft` like
the other torch checkpoints — output columns `[cold, dark, light, warm, maize]`.

### `jax/` — plant STARR-seq (Jores 2021), test Pearson r

Every value below was re-verified by loading the released checkpoint and re-running
inference.

| Tissue | Mode | probe (stage1) | fine-tuned (stage2) |
|---|---|---|---|
| leaf | combined | 0.7821 | **0.8899** |
| leaf | enhancer | 0.7660 | **0.8749** |
| leaf | promoter_only | 0.6876 | **0.7802** |
| proto | combined | 0.7884 | **0.8795** |
| proto | enhancer | 0.6870 | **0.8036** |
| proto | promoter_only | 0.7015 | **0.7683** |

### `jax/` — Agarwal et al. lentiMPRA (`mpra-<cell>-optimal`), test Pearson r

These carry the **corrected** reporter construct (see Notes). Retrained 2026-11-24.

| Checkpoint | Task | probe (stage1) | fine-tuned (stage2) |
|---|---|---|---|
| `mpra-HepG2-optimal` | lentiMPRA HepG2 | 0.8643 | **0.8885** |
| `mpra-K562-optimal` | lentiMPRA K562 | 0.8555 | **0.8801** |
| `mpra-WTC11-optimal` | lentiMPRA WTC11 | 0.8299 | **0.8403** |

Plus `jax/deepstarr-optimal` (see the paper for its metrics).

### `jax/` — Gosai et al. lentiMPRA (`Gosai-<cell>-optimal`)

Cell-averaged Pearson r (from the paper's 4-panel figure):

| Panel | Probing (S1) | Fine-tuned (S2) |
|---|---|---|
| Genomic Reference | 0.877 | 0.910 |
| High-Activity Designed | 0.709 | 0.776 |
| SNV Effects (Ref−Alt) | 0.364 | 0.400 |

Reproduced here: loading `Gosai-K562-optimal/stage2` and scoring the held-out chr7+13
K562 set gives Pearson 0.9195.

## Notes

- **Plant `stage1` is a ridge probe, not a model.** It carries no encoder weights — the
  features come from the *unmodified pretrained* encoder. Use `load_plant_probe()`.
- **Plant head width varies per cell** (4096 / 2048 / 1024). It's recorded in each
  checkpoint's `config.json`; don't assume a default.
- **Inputs are reporter constructs**, not bare genomic sequence — each checkpoint expects
  the construct it was trained on (the 230 bp element plus a 51 bp constant tail for
  lentiMPRA, library adapters for DeepSTARR, or 35S enhancer + core promoter + 5′ UTR +
  barcode for plant). See the repo docs.
- **lentiMPRA: two tails exist, they are not interchangeable, and the JAX and PyTorch
  checkpoints currently use different ones.** The original tail was taken from the
  Agarwal et al. 2025 Table S13 reverse primer `minP-enh-r` copied without
  reverse-complementing it, so it repeats the right adapter and runs the minimal
  promoter backwards. The corrected tail is the minimal promoter in sense orientation
  followed by the spacer, and it is the library default (`LENTIMPRA_SUFFIX`).

  **Every lentiMPRA checkpoint published here now uses the corrected tail** — the JAX ones
  retrained 2026-11-24, the PyTorch ones 2026-11-25. Each records its own construct
  (`config.json` for JAX, the `construct` field for PyTorch) and the loaders read it back,
  so you do not have to pass it by hand.

  Getting this wrong is not a rounding error: scoring a checkpoint with the tail it was
  *not* trained on costs ~0.02 Pearson (K562: 0.8793 → 0.8601), about what a
  composition-matched *scramble* costs (0.8555) — the encoder keys on the exact tail it saw
  in training. Retraining recovers the same performance, within 0.003 for JAX and 0.006 for
  PyTorch, so the two tails are equivalent in quality; they are simply not interchangeable.

  The pre-correction weights are kept on the lab's shared storage as
  `*_legacyconstruct`, and are deliberately **not** published here. Anything reproducing the
  controls, the Enformer baselines or the hyperparameter sweep of the accompanying write-up
  needs those, not these.
- Plant test constructs use a random barcode per row, so Pearson reproduces to ~±0.0001;
  quote plant numbers to 3 decimals.

## Citation

A paper is in preparation. Until it is available, please cite the blog post:

> Murphy, A., Durán, A., & Koo, P. K. (2026). *Adapting AlphaGenome to MPRA data.* Genomics x AI Blog, 20 February 2026. https://genomicsxai.github.io/blogs/2026-002/. https://doi.org/10.5281/zenodo.20272900

```bibtex
@article{Murphy2026,
  author  = {Murphy, Alan and Dur{\'a}n, Alejandra and Koo, Peter K.},
  title   = {Adapting AlphaGenome to MPRA data},
  journal = {Genomics x AI Blog},
  year    = {2026},
  month   = {February},
  day     = {20},
  url     = {https://genomicsxai.github.io/blogs/2026-002/},
  doi     = {10.5281/zenodo.20272900}
}
```

Please also cite [AlphaGenome](https://deepmind.google.com/science/alphagenome) (Google
DeepMind) and the underlying datasets (Agarwal et al.; Gosai et al.; de Almeida et al.;
Jores et al. 2021).
