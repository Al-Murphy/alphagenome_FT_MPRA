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
- **Drosophila STARR-seq** (DeepSTARR; de Almeida et al.) — developmental + housekeeping
- **Plant STARR-seq** (Jores et al. 2021) — tobacco leaf and maize protoplast, 3 data modes

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
| `mpra_K562` | lentiMPRA K562 | 0.8580 | **0.8785** |
| `mpra_HepG2` | lentiMPRA HepG2 | 0.8688 | **0.8876** |
| `mpra_WTC11` | lentiMPRA WTC11 | 0.8278 | **0.8344** |
| `starrseq_drosophila` | DeepSTARR (dev + hk) | 0.6184 | **0.7468** |

Drosophila is the mean of the two tasks (fine-tuned: dev 0.7193, hk 0.7744).

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

Plus `jax/{mpra-K562,mpra-HepG2,mpra-WTC11,deepstarr}-optimal` (see the paper for
their metrics).

## Notes

- **Plant `stage1` is a ridge probe, not a model.** It carries no encoder weights — the
  features come from the *unmodified pretrained* encoder. Use `load_plant_probe()`.
- **Plant head width varies per cell** (4096 / 2048 / 1024). It's recorded in each
  checkpoint's `config.json`; don't assume a default.
- **Inputs are reporter constructs**, not bare genomic sequence — each checkpoint expects
  the construct it was trained on (promoter+barcode, library adapters, or 35S
  enhancer + core promoter + 5′ UTR + barcode). See the repo docs.
- Plant test constructs use a random barcode per row, so Pearson reproduces to ~±0.0001;
  quote plant numbers to 3 decimals.

## Citation

If you use these weights, please cite our paper (in preparation) and the underlying
datasets (Agarwal et al.; de Almeida et al.; Jores et al. 2021), as well as
AlphaGenome (Google DeepMind).
