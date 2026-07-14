# Plant STARR-seq (Jores 2021) — caveats for the write-up

Findings from verifying the plant benchmark against the saved weights. The headline is
that **the published numbers are correct** — but three things about *how* they were
produced need stating in the paper, and one comparison in the figure is not
apples-to-apples.

## 1. The numbers are verified

All 12 AlphaGenome cells were re-scored from the saved checkpoints (rebuild the model,
load the weights, re-run inference on the held-out test split). Every cell reproduces
the committed value to **≤ 0.0001**:

| tissue | mode | method | reference | re-scored |
|---|---|---|---|---|
| leaf | combined | finetune | 0.8899 | 0.8899 |
| leaf | combined | probe | 0.7821 | 0.7821 |
| leaf | enhancer | finetune | 0.8749 | 0.8749 |
| leaf | enhancer | probe | 0.7660 | 0.7659 |
| leaf | promoter_only | finetune | 0.7802 | 0.7802 |
| leaf | promoter_only | probe | 0.6876 | 0.6876 |
| proto | combined | finetune | 0.8795 | 0.8795 |
| proto | combined | probe | 0.7884 | 0.7884 |
| proto | enhancer | finetune | 0.8036 | 0.8036 |
| proto | enhancer | probe | 0.6870 | 0.6870 |
| proto | promoter_only | finetune | 0.7683 | 0.7683 |
| proto | promoter_only | probe | 0.7015 | 0.7014 |

The residual ~0.0001 is expected: test constructs draw a random 12 bp barcode (and a
random 153 bp upstream filler on `noEnh` rows) at data-build time, so the test set is not
byte-identical between builds. **Quote plant numbers to 3 decimals, not 4.**

Reproduce with `scripts/test_ft_model_plant_starrseq.py`.

## 2. ⚠️ "Probing" means two different things in the figure

This is the one that affects a claim we actually make.

| Model | What its "probe" bar is |
|---|---|
| AlphaGenome | closed-form ridge on mean-pooled frozen encoder features |
| NTv3-post | closed-form ridge on mean-pooled frozen encoder features |
| PlantCAD2 | closed-form ridge on mean-pooled frozen encoder features |
| **PlantCaduceus** | **a *trained* attention-pool MLP head on a frozen backbone** |

PlantCaduceus's probing number is a stage-1 `best.pt` — a multi-head attention pooling
layer plus a 2-layer MLP, trained with early stopping. The other three are a single
closed-form ridge solve. That is a far more expressive head, and PlantCaduceus
accordingly posts the **highest probing score in the figure** (0.7969 leaf / 0.8133
proto).

**Do not read that as "PlantCaduceus has the best frozen representations."** It had the
best *head*. The comparison is not controlled.

Options for the write-up: either (a) describe PlantCaduceus's probing bar explicitly as
a trained linear-probe head rather than a ridge, or (b) use
`plant_starrseq_benchmark_no_plantcaduceus.png`, which drops it entirely. A ridge probe
for PlantCaduceus was never run; it would be cheap (~1 GPU-hour, no training) if we want
the controlled comparison, and would be expected to come in **below** 0.7969.

## 3. PlantCaduceus only ran the `combined` mode

There are no enhancer or promoter-only runs for PlantCaduceus — not in the reference
metrics, not as checkpoints, not in the upstream `results.tsv`. It is therefore excluded
from `plant_starrseq_benchmark_modes.png` (the 6-panel tissue × mode figure) rather than
being left absent from 4 of its 6 panels.

## 4. NTv3's numbers were measured with dropout active at test time

Upstream's NTv3 head holds an `nnx.Dropout(0.1)`. `nnx.Dropout` defaults to
`deterministic=False`, and the upstream code never calls `.eval()` nor passes
`deterministic=True` — its final test loop is a bare `preds = head(enc_out)`. So the
published NTv3 figures were computed with 10% of the head's activations randomly zeroed
on each test batch.

Re-scoring with dropout off moves every cell **up**, but only slightly:

| tissue | mode | published (dropout on) | dropout off | delta |
|---|---|---|---|---|
| leaf | combined | 0.8822 | 0.8831 | +0.0009 |
| leaf | enhancer | 0.8717 | 0.8719 | +0.0002 |
| leaf | promoter_only | 0.7888 | 0.7890 | +0.0002 |
| proto | combined | 0.8756 | 0.8768 | +0.0012 |
| proto | enhancer | 0.8035 | 0.8042 | +0.0007 |
| proto | promoter_only | 0.7748 | 0.7748 | +0.0000 |

Largest correction is **+0.0012** — below the 3-decimal precision we should be quoting,
and it changes no ranking. **The published NTv3 numbers stand.** The only real
consequence is that upstream's eval is stochastic rather than reproducible; this repo's
NTv3 runner sets `deterministic=True` and is unaffected.

## 5. The checkpoints did not come from this repo

All plant checkpoints were produced by the upstream `autotune` codebase
(`/grid/koo/home/duran/autotune`), not by this repo's runners. That mattered in three
ways, all now fixed here:

- The AlphaGenome runner recorded `history["test_pearson"][-1]` (the *last* epoch) while
  saving `best.pt` (the *best-val* epoch) — so a fresh run reported a metric describing a
  different model than the one it saved. Now reports the best-val epoch's test score.
- The PlantCAD2 runner's `_embed()` undid the RCPS reverse-complement half by flipping
  only the sequence axis; upstream flipped sequence **and** channel axes. The old code
  fed channel-permuted features to the head, silently. Fixed.
- The NTv3 runner only implemented stage 1 (frozen backbone) while the reference numbers
  come from a stage-2 (unfrozen) run, so it could not reproduce them even in principle.
  Stage 2 is now implemented.

Weights are archived at
`/grid/koo/home/shared/models/alphagenome_encoder/jax/plant-starrseq-<tissue>-<mode>/`
(`stage1/` = probe, `stage2/` = fine-tuned), in the standard `alphagenome_ft` orbax
layout, and every one has been re-verified by loading it and re-running inference.

**The AlphaGenome head width is not constant across cells** — it is 4096 for
leaf/combined, 2048 for proto/combined and 1024 for the other four (the sweep tuned it
per mode). The committed configs all claimed `nl_size: 1024` and so did not describe the
published runs; worse, with one config per *tissue* covering three modes of differing
width, a single `model_params.nl_size` could not describe them even in principle.

Fixed: the AlphaGenome configs now carry a `mode_overrides` block giving the true width
per mode, and the runner applies it. Each value was checked against the `nl_size` stored
in the corresponding checkpoint's `stage2/config.json` — all six match.

If the write-up quotes a head size anywhere, it must be per-cell, not a single number.

(For the record: `center_bp: 256` in those configs is **inert**, not a bug — with
`pooling_type: flatten` the head flattens every encoder position and never consults it.)
