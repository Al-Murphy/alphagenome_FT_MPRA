# lentiMPRA reporter construct: orientation bug, scope and fix

Raised by Masayuki Nagai as
[alphagenome-encoder-ft#15](https://github.com/MasayukiNagai/alphagenome-encoder-ft/pull/15);
validated independently here on 2026-11-23.

## The bug

The 281 bp lentiMPRA model input is the 230 bp assayed element (left adapter +
200 bp CRE + right adapter — the `seq` column of `data/legnet_lentimpra/*.tsv`)
plus a 51 bp constant tail. That tail was built from the last 36 nt of the
reverse primer `minP-enh-r` (Agarwal et al. 2025, Supplementary Table S13),
copied verbatim, plus a fixed 15 bp placeholder barcode.

Table S13 writes reverse primers 5'→3' on the strand **opposite** the reporter,
so they must be reverse-complemented before they describe the construct. They
were not.

```
RC("TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG")
  = "CATTGCGTGAACCGA" + "CACTAGAGGGTATATAATGGA"
     └─ right adapter ┘   └─ first 21 nt of the minP ┘
```

So the input carried the right adapter **twice** — once forward at the end of
`seq`, once reverse-complemented — then 21 of the ~35 nt of minimal promoter
running backwards, then a fixed barcode that in reality is a random N15 sitting
87 nt downstream of the element, outside the window entirely.

It survived review because the legacy string contains `TATATA`, which reads like
a TATA box at a glance. It is the reverse complement of one.

## Evidence the corrected orientation is right

Three independent checks, all reproduced in `tests/test_lentimpra_construct.py`:

1. **The primer anneals to the adapter.** `RC(minP-enh-r)` begins with
   `RIGHT_ADAPTER`, which is exactly why copying its tail duplicated the adapter.
2. **The two primers join cleanly.** `RC(minP-enh-r)` and
   `RC(pLSmP-bc-primer-r)` overlap by 15 nt and join into 123 nt of
   reporter-strand sequence; the first 102 nt run to the end of the barcode.
3. **It lands on the EGFP start codon.** Continued past the barcode the sequence
   reads `ACCGGT` (AgeI) + `CGCCACC` (Kozak) + `ATG` — the start of EGFP in the
   pLS reporter. The legacy sequence produces nothing of the sort.

Corrected tail (`LENTIMPRA_SUFFIX`, the first 51 nt of `LENTIMPRA_DOWNSTREAM`):
the minimal promoter in sense orientation, then the spacer. 41 of the 51 bases
change, i.e. 14.6% of the model input. The first 230 bp are untouched.

## Scope

Only lentiMPRA. Every other short-sequence dataset uses its own, disjoint
flanks — verified string by string:

| Dataset | Affected | Flanks |
|---|---|---|
| **lentiMPRA** | **yes** | the construct above |
| DeepSTARR | no | Illumina TruSeq adapters (25 bp / 26 bp) |
| Gosai episomal | no | none — `N` padding only |
| Plant STARR-seq | no | 35S enhancer + UTR + random 12 bp barcode |
| Jores multi-condition | no | Jores 15 bp adapters |

Released models affected: `jax/mpra-{HepG2,K562,WTC11}-optimal` and
`torch/mpra_{HepG2,K562,WTC11}`. The other 13 released models are clean.

CAGI5 is affected only **through the checkpoints**: the CAGI5 scripts score a
281 bp *genomic* window around each variant and never build the construct.

## The mismatch penalty: you cannot hot-swap the fix

Released K562 model (`jax/mpra-K562-optimal/stage2`), same checkpoint, scored
under three different tails (`scripts/slurm/construct_audit_mpra.sh`, job
4065781, n = 19,670):

| Tail | Test Pearson | vs legacy |
|---|---|---|
| legacy (what it was trained on) | **0.8793** | — (paper reports 0.879 ✓) |
| corrected | 0.8601 | −0.019 |
| scramble (same composition, meaningless) | 0.8555 | −0.024 |

`corrected ≈ scramble`, both ~0.02 below legacy. The encoder is keyed to the
*exact* constant tail it saw in training; substituting the biologically correct
sequence costs about the same as substituting nonsense. The corrected tail beats
the scramble by only 0.005.

Two consequences:

- **The published numbers are not wrong.** The released models were trained and
  evaluated on the same (legacy) construct, so they are internally consistent.
  What is wrong is the Methods description, and the claim that the input matches
  the assayed sequence.
- **Patching the constants without retraining would silently drop every released
  model by ~0.02.** Hence the per-checkpoint pinning below.

Nagai's retrained A/B in the PyTorch package (K562, 3 seeds) puts retrained
legacy and retrained corrected at parity (.8771 vs .8777 test), so the corrected
models are expected to land at the published numbers to ~3 decimals.

## The fix

- `alphagenome_ft_mpra/data.py` gains `LENTIMPRA_DOWNSTREAM` (102 bp, through
  the N15 barcode), `LENTIMPRA_SUFFIX` (the 51 bp that fill the input) and
  `LENTIMPRA_SUFFIX_LEGACY` (kept so pre-correction checkpoints stay
  reproducible — do not tidy it away).
- `LentiMPRADataset` takes `construct_suffix`, defaulting to the corrected
  sequence. The historical `promoter_seq`/`rand_barcode` pair still pins a
  construct explicitly.
- `finetune_mpra.py` resolves the construct once and **writes it into the
  checkpoint** (`head_configs.mpra_head.metadata.construct_suffix`).
- `test_ft_model_mpra.py` reads the construct back from the checkpoint. For
  checkpoints that predate pinning it falls back to the **legacy** suffix (what
  they were trained on) and says so loudly. It never silently applies the new
  default to an old checkpoint.

## What is being retrained, and what is not

Retrained on the corrected construct (jobs 4065831–3, kooq/koolab):

- `mpra-{HepG2,K562,WTC11}-optimal` — `configs/mpra_{cell}.json`, which now carry
  `"construct_suffix": null` (the corrected tail). Hyperparameters are unchanged
  from the published runs; set `"construct_suffix": "legacy"` to reproduce those.

  The retrains were launched from temporary `*_fixedconstruct` copies of the
  configs and job scripts, since folded back into the canonical files. So on disk
  the corrected checkpoints currently sit under
  `results/models/checkpoints/<cell>/mpra-<cell>-optimal-fixedconstruct/`, while
  `mpra-<cell>-optimal/` still holds the released legacy weights. The scoring
  scripts default to the canonical name and take `RUN=` / `RUN_TEMPLATE=` to
  point at either — see the open item at the end about swapping them.

### Retrain results (full-split test Pearson)

Scored with `scripts/slurm/test_mpra.sh`, which passes no
`--construct_suffix`, so the construct is read back from the checkpoint — this
also validates the pinning round trip. Do **not** compare against the
`test_pearson` printed per epoch during training: that is batch-averaged and
attenuated.

| cell | stage | legacy (published) | corrected | Δ |
|---|---|---|---|---|
| HepG2 | probing | 0.8643 | 0.8643 | 0.0000 |
| HepG2 | fine-tuned | 0.8874 | 0.8885 | +0.0011 |
| K562 | probing | 0.8532 | 0.8555 | +0.0023 |
| K562 | fine-tuned | 0.8793 | 0.8801 | +0.0008 |
| WTC11 | probing | 0.8294 | 0.8299 | +0.0005 |
| WTC11 | fine-tuned | 0.8394 | 0.8403 | +0.0009 |

**All three cell types are at parity.** Every Δ is positive and ≤ +0.0023,
matching the direction of the PyTorch A/B. The largest is K562 probing (+0.0023);
the three fine-tuned arms — the numbers the paper reports — move by +0.0011,
+0.0008 and +0.0009. All are an order of magnitude below the ~0.02 train/test
mismatch penalty and inside the per-split SE of r (≈0.0033 K562, ≈0.0050 HepG2,
≈0.0091 WTC11), so none is distinguishable from noise. Probing barely moves,
which is what the mechanism predicts: with the encoder frozen, the constant tail
contributes a fixed offset that the head's bias absorbs, so only the fine-tuned
arm has much room to move.

K562 is worth one note: the published run hit the 48 h wall at stage-2 epoch
46/50, whereas this one early-stopped at epoch 41 (best epoch 35-37) on kooq
without a wall. So the retrain is the better-trained of the two and still lands
within 0.001 — the construct is not doing anything either way.

This also confirms that the ~0.02 drop measured earlier was purely a train/test
mismatch, not a deficiency of the corrected construct: trained on it, the model
scores the same as before.

**Deliberately not retrained** (decision 2026-11-23): the fold-0 leakage control
(Fig. S3b), the random-init control (Fig. S3a), the Enformer lentiMPRA
baselines, and the 63-run hyperparameter sweep (Table S4, whose embedding caches
are also baked with the legacy construct). Each of these was trained *and*
evaluated on the legacy construct, so each is internally consistent and its
published numbers stand. The Methods must state this.

This rested on the retrained models landing at parity with the published
numbers. **Resolved 2026-11-24: they do.** 0.8885 / 0.8801 / 0.8403 against the
published 0.8874 / 0.8793 / 0.8394. The controls, the Enformer baselines and the
Table S4 sweep therefore still sit on the same scale as the main models, and the
decision not to rerun them stands.

### Open point: the Enformer baseline — resolved

Fig. 4a and Fig. 4d compare the AlphaGenome encoder against an Enformer encoder
run through the *same* recipe. If the AG numbers had moved to the corrected
construct while Enformer stayed on legacy, that comparison would no longer be
input-matched. **Since the AG retrains came in at parity (≤0.0023), the effect is
numerically negligible and a Methods/caption note covers it — the three Enformer
runs (~60 GPU-h) do not need rerunning.** Enformer *is* affected by the same bug
(`enf_utils.py:216-217` hardcodes the identical legacy tail, with no
`construct_suffix` parameter and no config path), but it was trained and
evaluated on that tail throughout, so its published numbers are internally
consistent. MPRALegNet is not affected at all: it reads the bare 230 bp `seq`
column and never builds the construct.

### CAGI5 re-scored against the retrained checkpoints

CAGI5 is affected only *through the weights* — the CAGI5 scripts score a 281 bp
**genomic** window around each variant and never build the reporter construct —
so this is pure inference (`scripts/slurm/cagi5_mpra.sh`, job 4075333). Only
HepG2 and K562 have CAGI5 elements (HepG2: F9, LDLR, SORT1; K562: GP1BB, HBB,
HBG1, PKLR); there is no WTC11 set. Both the plain and the test-time-augmented
(`posshift20_n3` + revcomp) arms were re-run, 8 runs in all.

Pooled across all SNPs:

| run | legacy r | corrected r | Δr | legacy ρ | corrected ρ | Δρ |
|---|---|---|---|---|---|---|
| probing HepG2 | 0.6776 | 0.6845 | +0.0069 | 0.5711 | 0.5504 | −0.0207 |
| probing K562 | 0.6905 | 0.6925 | +0.0020 | 0.5319 | 0.5392 | +0.0072 |
| fine-tuned HepG2 | 0.6850 | 0.6855 | +0.0005 | 0.5638 | 0.5529 | −0.0109 |
| fine-tuned K562 | 0.6316 | 0.6486 | +0.0170 | 0.4715 | 0.4903 | +0.0187 |
| probing HepG2, aug | 0.6825 | 0.6788 | −0.0037 | 0.5848 | 0.5626 | −0.0222 |
| probing K562, aug | 0.6963 | 0.7086 | +0.0123 | 0.5380 | 0.5588 | +0.0208 |
| fine-tuned HepG2, aug | 0.6900 | 0.6887 | −0.0012 | 0.5827 | 0.5663 | −0.0164 |
| fine-tuned K562, aug | 0.6573 | 0.6651 | +0.0078 | 0.5021 | 0.5121 | +0.0101 |

Movement here is larger than on the in-domain lentiMPRA test split (up to
±0.02 rather than ±0.002), which is expected: this is zero-shot transfer to a
different assay on 3-4 elements, so each number rests on far less data. There is
no systematic degradation. The pattern is cell-type-split rather than
construct-driven — **K562 improves on every one of its eight statistics, HepG2's
Spearman falls in all four of its arms** while its Pearson barely moves. Nothing
here is outside what element-level resampling on this few elements would produce,
and the conclusions the paper draws from CAGI5 are unchanged.

One caveat on the augmented rows: the new runs were given `--random_seed 0`
whereas the published ones were unseeded, so part of each augmented Δ is
shift-sampling noise rather than a weight difference.

### Where the result files live

`results/` is gitignored, so these are not under version control. The corrected
numbers have been promoted to the canonical paths that the table and figure
scripts read:

- `results/test_predictions/<cell>/` — corrected; legacy copies kept in
  `results/test_predictions/legacy_construct/<cell>/`.
- `results/cagi5_evaluations/` — corrected for the `finetuned`/`probing` rows;
  the 24 legacy AlphaGenome CSVs kept in
  `results/cagi5_evaluations/legacy_construct/`. The `base`, `enformer_mpra` and
  `mpralegnet` files were not touched — those models stay on the legacy
  construct by design.

So `create_mpra_comparison_table.py` and `plot_cagi5_results.py` now pick up the
corrected AlphaGenome numbers with no flag changes. To go back, swap the two
directories again; nothing was deleted.

### Open item: swap the on-disk checkpoint directories

The corrected checkpoints are still parked under
`results/models/checkpoints/<cell>/mpra-<cell>-optimal-fixedconstruct/` and the
released legacy weights still occupy `mpra-<cell>-optimal/`. Checkpoints are not
in git (`results/` is gitignored), so a rename here has no undo through version
control — hence it has not been done automatically. Before the Hugging Face
re-upload, decide one of:

- rename `mpra-<cell>-optimal` to `mpra-<cell>-optimal-legacyconstruct` and
  promote the corrected one into the canonical name (what the scripts now expect
  by default), or
- keep both names and always pass `RUN=` / `RUN_TEMPLATE=` explicitly.

Whichever is chosen, the legacy weights must be kept: they are what the
released models, the controls, the Enformer baselines and the Table S4 sweep
were all evaluated against.

## Still open

The manuscript text, the figures and the tables were updated on 2026-11-25, and the
internal planning note that tracked that work has been removed. What it listed and is
still outstanding:

- **The K562 lentiMPRA attribution panel** (`results/plots/attribution_K562_gradient_x_input`)
  has not been regenerated. It dates from Feb 2026, so it was produced with the legacy
  weights *and* the legacy 51 bp tail — both the model and the sequence it attributes
  over have since changed. Regenerate with
  `scripts/compute_attributions.py --checkpoint_dir results/models/checkpoints/K562/mpra-K562-optimal/stage2`,
  matching the element list the published panel used.
- **`/grid/koo/home/shared/models/alphagenome_encoder/torch/README.md`** still documents
  the legacy tail as if it were correct. The file is owned by another user and is not
  group-writable, so its owner has to make the change. `jax/README.md` is updated.
- **The PyTorch checkpoints are still on the legacy construct.** Retraining is under way
  in `alphagenome-encoder-ft` (branch `lentimpra-match-jax`, fold splits, configs
  mirroring the JAX ones); `torch/` on Hugging Face needs refreshing once they land. The
  model card already states which framework uses which tail.
- **Table S5 must not be regenerated from the new CAGI5 CSVs** — see the note below.

### Table S5 inference speed: keep the published numbers

Table S5 derives ms/variant from the `posshift20_n3_revcomp` CAGI5 summary CSVs, which
were replaced when the retrains were scored. The new runs report **7.2-7.3 ms/variant
against the published 9.5-10.0**, which would inflate the speed-up from 526x/552x to
~700x.

That difference is **node variation, not the model**. Retraining changes weight values,
not architecture, parameter count or FLOPs, so inference cost is unchanged by
construction; the new runs simply landed on a less contended GPU (26 min wall-clock
versus ~4 min for the published run, i.e. the machine differed in both directions).
Regenerating only the AlphaGenome rows would also compare AG timed on one node against
Enformer and MPRALegNet timed on another, biasing the "~500x faster" claim in our own
favour. If it is ever re-measured, re-time every row in one session on one node.

## Manuscript changes required

Regardless of what the retrains show:

- **Methods, "lentiMPRA (Agarwal et al. 2025)"** — the quoted 36 bp promoter and
  15 bp barcode strings are wrong and must be replaced with the reporter-strand
  description: 230 bp element + the minimal promoter in sense orientation + the
  spacer, 281 bp total, with the random barcode lying downstream *outside* the
  window. Drop or qualify "matching the assayed sequence".
- **Methods** — state that the controls (S3a, S3b), the Enformer baselines and
  the Table S4 sweep were run on the pre-correction construct, and why that is
  sound (each is internally consistent; corrected and legacy are at parity after
  retraining).
- **Table S4 caption** — footnote the construct.
- **HF model card** (`assets/hf_model_card.md`) and
  `/grid/koo/home/shared/models/alphagenome_encoder/*/README.md` — the lentiMPRA
  entries describe a "promoter+barcode" construct; update, and state which
  construct each checkpoint carries. *(Done for the repo copies; the shared-folder
  READMEs and the HF upload still need pushing.)*

### Drop-in replacement for the Methods paragraph

Replacing the last sentence of **Methods → "lentiMPRA (Agarwal et al. 2025)"**:

> Each 230-bp candidate element (the 200-bp element with its 15-bp library
> adaptors) was followed by the 51 bp that the library's cloning PCRs place
> immediately downstream of it on the reporter strand — the minimal promoter in
> sense orientation (CACTAGAGGGTATATAATGGAAGCTCGACTTCCAG) followed by the start
> of the spacer (CTTGGCAATCCGGTAC) — to form the 281-bp construct presented to
> the model. This downstream sequence was obtained by reverse-complementing the
> reverse primers minP-enh-r and pLSmP-bc-primer-r of Supplementary Table S13 of
> Agarwal et al. 2025, which are written on the strand opposite the reporter,
> and joining them on their 15-nt overlap. The random 15-nt barcode begins 87 nt
> downstream of the element and therefore falls outside the 281-bp window.

And a new sentence, in Methods or in the captions of Fig. S3a/S3b and Table S4:

> The random-initialisation control (Supplementary Fig. S3a), the fold-0 leakage
> control (Supplementary Fig. S3b), the Enformer encoder baselines and the
> hyperparameter sweep (Supplementary Table S4) were run with an earlier version
> of this downstream sequence, in which the Supplementary Table S13 reverse
> primer was used without reverse-complementing it. Each of those analyses was
> trained and evaluated with the same construct throughout and is therefore
> internally consistent; retraining on the corrected construct changes held-out
> performance by less than 0.003 in Pearson r (largest observed difference across
> the three cell lines, in both the probing and the fine-tuned setting).

That bracket is now filled from the retrain table above: all three cell types are
scored, the largest |Δ| anywhere in the table is +0.0023 (K562 probing), and the
three fine-tuned arms differ by at most +0.0011. "Less than 0.003" is the honest
bound; if you prefer to quote only the fine-tuned arms it is "less than 0.002".
