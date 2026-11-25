# Manuscript update scope — corrected lentiMPRA construct

What changes in the write-up now that `mpra-{HepG2,K562,WTC11}-optimal` have been
retrained on the corrected reporter tail, and what deliberately does not.
Background and evidence: [`lentimpra_construct_audit.md`](lentimpra_construct_audit.md).

**The one-line summary for the paper: nothing concludes differently.** Every number
that moves, moves by ≤0.0023 Pearson. The changes below are about accuracy of the
record, not about results.

---

## 1. Numbers that change

Full-split test Pearson on fold 10 (n = 19,670 K562 / 4,622 WTC11 / HepG2 as before):

| | published | corrected | Δ |
|---|---|---|---|
| HepG2 probing | 0.864 | 0.864 | 0.0000 |
| HepG2 fine-tuned | 0.887 | **0.889** | +0.0011 |
| K562 probing | 0.853 | **0.855** | +0.0023 |
| K562 fine-tuned | 0.879 | **0.880** | +0.0008 |
| WTC11 probing | 0.829 | **0.830** | +0.0005 |
| WTC11 fine-tuned | 0.839 | **0.840** | +0.0009 |

Five of the six move at three decimal places.

---

## 2. Figures and tables, with the script that regenerates each

| Manuscript item | What it shows | Script | Data source | Status |
|---|---|---|---|---|
| **Main lentiMPRA benchmark figure** (AG vs Enformer vs MPRALegNet, 3 cell types × probing/fine-tuned) | 6 AG bars move | `scripts/plot_benchmark_results.py` | ⚠️ **hard-coded in the script**, not read from CSVs | ✅ **values updated and figure regenerated** → `results/comparison_tables/plots/lentimpra_benchmark.{pdf,png}` |
| **lentiMPRA benchmark LaTeX table** | same 6 numbers | hand-maintained | — | ✅ **updated** → `results/comparison_tables/all_benchmark_tables.md` |
| **CAGI5 figures** (model comparison, per-cell, augmentation) | AG probing/fine-tuned rows | `scripts/plot_cagi5_results.py` | `results/cagi5_evaluations/*.csv` | ✅ **regenerated** from the corrected CSVs |
| **lentiMPRA attribution panel** (`results/plots/attribution_K562_gradient_x_input`) | attributions from the K562 fine-tuned model | `scripts/compute_attributions.py` | `mpra-K562-optimal/stage2` | ⬜ **regenerate** — both the weights *and* the 51 bp tail it attributes over have changed |
| Random-init control (Fig. S3a) | — | `plot_benchmark_results.py --lentimpra_name lentimpra_benchmark_random_init` | legacy | ✅ unchanged by design |
| Fold-0 leakage control (Fig. S3b) | — | `plot_benchmark_results.py --lentimpra_csv` | legacy | ✅ unchanged by design |
| Hyperparameter sweep (Table S4) | — | — | legacy (embedding caches also legacy-baked) | ✅ unchanged by design |
| Model efficiency (Table S5) | inference speed | `docs/model_efficiency_table.md` | CAGI5 timing columns | ⚠️ **do not regenerate — see §4** |
| DeepSTARR / Gosai / plant / Jores figures | — | — | disjoint flanks | ✅ unaffected |

### Commands

```bash
# lentiMPRA attribution panel - the one regeneration still outstanding.
# Check the element list matches what the published figure used.
.venv/bin/python scripts/compute_attributions.py \
    --checkpoint_dir ./results/models/checkpoints/K562/mpra-K562-optimal/stage2 \
    --cell_type K562 ...
```

---

## 3. Text changes

1. **Methods, "lentiMPRA (Agarwal et al. 2025)"** — the quoted 36 bp promoter and
   15 bp barcode strings are wrong. Replace with the reporter-strand description.
   Drop or qualify "matching the assayed sequence". Drop-in paragraph is in the
   audit doc.
2. **Methods** — add the sentence stating that the controls (S3a, S3b), the
   Enformer baselines and the Table S4 sweep were run on the pre-correction
   construct, that each is internally consistent, and that retraining changes
   held-out performance by **less than 0.003 in Pearson r**. Drop-in text in the
   audit doc; the bracket is now filled.
3. **Table S4 caption** — footnote the construct.
4. **Fig. 4a / 4d caption** — the AG models use the corrected construct and the
   Enformer baselines the legacy one. At ≤0.003 this needs a caption note, not a
   rerun (see §4).
5. **Data/code availability** — the Hugging Face weights have been replaced
   (`Al-Murphy/alphagenome-encoder-ft`, JAX `mpra-<cell>-optimal`). If the
   manuscript cites a specific revision, update it.

---

## 4. Two traps — things that look like they need updating but must not be

### Table S5 inference speed: keep the published numbers

Table S5 derives ms/variant from the `posshift20_n3_revcomp` CAGI5 summary CSVs,
which have just been replaced. The new runs report **7.2–7.3 ms/variant against the
published 9.5–10.0**, which would inflate the speed-up from 526×/552× to ~700×.

That difference is **node variation, not the model.** Retraining changes weight
*values*, not architecture, parameter count or FLOPs, so inference cost is
unchanged by construction. The new runs simply landed on a faster/less contended
GPU — the same job took 26 min wall-clock this time versus ~4 min for the
published run, i.e. the machine was behaving differently in both directions.

Worse, regenerating only the AlphaGenome rows would compare AG timed on one node
against Enformer and MPRALegNet timed on another, which biases the headline
"~500× faster" claim in our own favour. **Keep Table S5 exactly as published.** If
it ever is re-measured, re-time every row in one session on one node.

### The Enformer baseline does not need rerunning

Enformer *is* affected by the same bug — `enf_utils.py:216-217` hard-codes the
identical legacy tail, with no `construct_suffix` parameter and no config path —
but it was trained and evaluated on that tail throughout, so its numbers are
internally consistent. Since the AG retrains came in at parity, the Fig. 4a/4d
comparison is not meaningfully un-matched, and the ~60 GPU-h rerun is not
justified. A caption note covers it.

MPRALegNet is not affected at all: it reads the bare 230 bp `seq` column and never
builds a tail.

---

## 5. Released artifacts

| Artifact | State |
|---|---|
| HF `jax/mpra-<cell>-optimal` | ✅ replaced with corrected weights, mirrored exactly, model card updated |
| HF `torch/mpra_<cell>` | ⚠️ **still legacy** — PyTorch retraining is pending (K562 done upstream, HepG2/WTC11 not). The card now states the split explicitly. |
| Shared `jax/mpra-<cell>-optimal` | ✅ corrected; pre-correction kept as `*-legacyconstruct` (not published) |
| Shared `jax/README.md` | ✅ updated |
| Shared `torch/README.md` | ⚠️ **still documents the legacy tail as if correct.** Owned by another user (`kachu`), not group-writable — needs its owner to update it. |

---

## 6. Checklist

- [x] Retrain all three cell types on the corrected construct
- [x] Score full-split test; confirm parity
- [x] Re-run CAGI5 against the new checkpoints (both plain and augmented arms)
- [x] Update the hard-coded figure values and regenerate the lentiMPRA figure
- [x] Update the lentiMPRA LaTeX benchmark table
- [x] Swap checkpoint directories; publish to Hugging Face; update the model card
- [x] Update the shared `jax/README.md`
- [x] Re-run `plot_cagi5_results.py`
- [ ] Regenerate the K562 attribution panel
- [ ] Apply the Methods / caption edits to the manuscript
- [ ] Get `torch/README.md` updated by its owner
- [ ] Retrain the PyTorch HepG2 and WTC11 checkpoints, then refresh `torch/` on HF
