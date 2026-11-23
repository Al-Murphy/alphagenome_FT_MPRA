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
102 nt downstream of the element, outside the window entirely.

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

- `mpra-{HepG2,K562,WTC11}-optimal-fixedconstruct` — configs
  `configs/mpra_{cell}_fixedconstruct.json`, hyperparameters identical to the
  originals, only the construct and run name differ.

**Deliberately not retrained** (decision 2026-11-23): the fold-0 leakage control
(Fig. S3b), the random-init control (Fig. S3a), the Enformer lentiMPRA
baselines, and the 63-run hyperparameter sweep (Table S4, whose embedding caches
are also baked with the legacy construct). Each of these was trained *and*
evaluated on the legacy construct, so each is internally consistent and its
published numbers stand. The Methods must state this.

This rests on the retrained models landing at parity with the published numbers.
**If the retrains do not reproduce ≈0.887 / 0.879 / 0.839, revisit** — a real
shift would mean the controls no longer sit on the same scale as the main
models, and they would have to be rerun too.

### Open point: the Enformer baseline

Fig. 4a and Fig. 4d compare the AlphaGenome encoder against an Enformer encoder
run through the *same* recipe. If the AG numbers move to the corrected construct
and Enformer stays on legacy, that comparison is no longer input-matched. At
parity the numerical effect is negligible and a caption note covers it; if the
retrains shift, the three Enformer runs (~60 GPU-h) should follow.

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
> and joining them on their 15-nt overlap. The random 15-nt barcode lies 102 nt
> downstream of the element and therefore falls outside the 281-bp window.

And a new sentence, in Methods or in the captions of Fig. S3a/S3b and Table S4:

> The random-initialisation control (Supplementary Fig. S3a), the fold-0 leakage
> control (Supplementary Fig. S3b), the Enformer encoder baselines and the
> hyperparameter sweep (Supplementary Table S4) were run with an earlier version
> of this downstream sequence, in which the Supplementary Table S13 reverse
> primer was used without reverse-complementing it. Each of those analyses was
> trained and evaluated with the same construct throughout and is therefore
> internally consistent; retraining on the corrected construct changes held-out
> performance by less than **[FILL: measured Δ from jobs 4065831–3]**.

The bracketed number must come from the retrains before this sentence is used.
