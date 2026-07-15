# Released model weights

Fine-tuned AlphaGenome **encoder** checkpoints (the transformer is bypassed; a
regression head sits on the raw encoder output) for four MPRA / STARR-seq benchmarks.

**Hub:** [`Al-Murphy/alphagenome-encoder-ft`](https://huggingface.co/Al-Murphy/alphagenome-encoder-ft)

---

## ⚠️ Licence — read before use

These checkpoints are **fine-tuned derivatives of AlphaGenome**. Under the
[AlphaGenome Model Terms](https://deepmind.google.com/science/alphagenome/model-terms),
a model fine-tuned from AlphaGenome is a *Derivative* and remains subject to the same
terms — **including the restriction to non-commercial use**. The base parameters were
created by Google DeepMind and are the property of Google LLC.

The fine-tuning **code** in this repository is Apache-2.0. The **weights** are not ours
to relicense.

Loading a checkpoint also requires the base AlphaGenome weights
([`google/alphagenome-all-folds`](https://huggingface.co/google/alphagenome-all-folds)),
which are **access-gated**: accept the terms on that page and `huggingface-cli login`
before first use, or pass `base_checkpoint_path=` pointing at a local copy.

---

## Install

```bash
pip install -e .                      # this repo (pulls alphagenome, alphagenome_ft, ...)
pip install huggingface_hub
```

For the PyTorch checkpoints you additionally need
[`alphagenome-encoder-ft`](https://github.com/MasayukiNagai/alphagenome-encoder-ft).
The JAX and PyTorch stacks do **not** coexist — use separate environments.

## Load

```python
from alphagenome_ft_mpra.hub import list_pretrained, load_pretrained

list_pretrained()

# JAX (Haiku) — the fine-tuned (stage-2) model
model = load_pretrained('plant-starrseq-leaf-combined')

# PyTorch (the framework is inferred from the checkpoint name)
model = load_pretrained('mpra_K562')
preds = model.predict_sequences(['ACGT...'], construct_mode='promoter_barcode')
```

`load_pretrained` reads each checkpoint's own `config.json` to decide which head class
to build and how wide it is, so you don't have to. That matters: the plant heads are
4096 / 2048 / 1024 wide **depending on the cell**, and they use unnamed Haiku modules,
so they need `PlantMPRAHead` rather than `EncoderMPRAHead` (see below).

Already have the weights on a shared filesystem? Skip the download:

```python
model = load_pretrained('plant-starrseq-leaf-combined',
                        local_dir='/grid/koo/home/shared/models/alphagenome_encoder/jax/plant-starrseq-leaf-combined')
```

---

## What's in the release

### PyTorch (`torch/`)

`stage1` = `frozen_encoder.pt` (encoder frozen, head trained);
`stage2` = `finetuned_encoder.pt` (encoder unfrozen). Test Pearson r:

| Checkpoint | Task | frozen | fine-tuned |
|---|---|---|---|
| `mpra_K562` | lentiMPRA K562 | 0.8580 | **0.8785** |
| `mpra_HepG2` | lentiMPRA HepG2 | 0.8688 | **0.8876** |
| `mpra_WTC11` | lentiMPRA WTC11 | 0.8278 | **0.8344** |
| `starrseq_drosophila` | DeepSTARR (dev + hk) | 0.6184 | **0.7468** |

Drosophila `test_pearson` is the mean of the two tasks (fine-tuned: dev 0.7193, hk 0.7744).

### JAX (`jax/`)

Layout `<run>/stage1/` and `<run>/stage2/`, each `{config.json, checkpoint/}` (orbax) —
the format `alphagenome_ft.load_checkpoint()` expects.

| Checkpoint | Task | Input |
|---|---|---|
| `mpra-K562-optimal` | lentiMPRA K562 | 281 bp |
| `mpra-HepG2-optimal` | lentiMPRA HepG2 | 281 bp |
| `mpra-WTC11-optimal` | lentiMPRA WTC11 | 281 bp |
| `deepstarr-optimal` | DeepSTARR (dev + hk) | 256 bp |
| `plant-starrseq-{leaf,proto}-combined` | Jores 2021 plant STARR-seq | 437 bp |
| `plant-starrseq-{leaf,proto}-enhancer` | " (35S enhancer construct) | 437 bp |
| `plant-starrseq-{leaf,proto}-promoter_only` | " (core promoter) | 170 bp |

Plant test Pearson r (every value re-verified by loading the released checkpoint and
re-running inference):

| Tissue | Mode | probe (stage1) | fine-tuned (stage2) | head width |
|---|---|---|---|---|
| leaf | combined | 0.7821 | **0.8899** | 4096 |
| leaf | enhancer | 0.7660 | **0.8749** | 1024 |
| leaf | promoter_only | 0.6876 | **0.7802** | 1024 |
| proto | combined | 0.7884 | **0.8795** | 2048 |
| proto | enhancer | 0.6870 | **0.8036** | 1024 |
| proto | promoter_only | 0.7015 | **0.7683** | 1024 |

> The four `*-optimal` runs carry no `metrics.json`; their numbers are in the paper.

---

## Gotchas

**Plant `stage1` is a ridge probe, not a model.** It has no encoder weights of its own —
the features come from the *unmodified pretrained* AlphaGenome encoder. Load it with
`load_plant_probe()` and predict with `(X - xb) @ w + yb`, where
`X = encoder_output.mean(axis=1)`.

**The plant head is not `EncoderMPRAHead`.** The plant checkpoints were produced by the
upstream `autotune` codebase, whose Haiku modules are unnamed — their parameters are
keyed `head/mpra_head/~predict/{layer_norm, linear, linear_1}`, where `EncoderMPRAHead`
uses `norm`/`hidden_0`/`output`. Haiku restores by module path, so `EncoderMPRAHead`
cannot load them. Use `PlantMPRAHead` (what `load_pretrained` does).

**Head width varies per plant cell** (4096 / 2048 / 1024, table above), and
`load_checkpoint()` builds the head from the *registered* `HeadConfig`, **not** from the
checkpoint's `config.json`. So the metadata must be registered before calling it or the
restore dies on a shape mismatch. Again, `load_pretrained` handles this.

**Test-set jitter (plant).** Constructs draw a random 12 bp barcode (and a random 153 bp
filler on `noEnh` rows) at data-build time, so the test set is not byte-identical between
builds. Pearson reproduces to ~±0.0001 — quote plant numbers to 3 decimals.

**Input constructs matter.** Each checkpoint expects the construct it was trained on
(promoter + barcode for lentiMPRA; library adapters for DeepSTARR; 35S enhancer + core
promoter + 5' UTR + barcode for plant). The torch checkpoints bundle a `ConstructSpec` —
use `predict_sequences(..., construct_mode=...)` and it assembles them for you. See
[plant_starrseq_caveats.md](plant_starrseq_caveats.md) for the plant construct.

## Reproducing

```bash
python scripts/test_ft_model_plant_starrseq.py --tissue leaf --mode combined \
    --checkpoint_dir <weights>/plant-starrseq-leaf-combined/stage2
```
