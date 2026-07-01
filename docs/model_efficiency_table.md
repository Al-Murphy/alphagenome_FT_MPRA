# Model size and inference-speed comparison

Parameter counts are measured directly from the model checkpoints; inference speed is
taken from the CAGI5 zero-shot variant-effect evaluation, run under a **matched setting**
(`posshift20_n3_revcomp`: 3 position shifts × 2 strands = 6 forward passes per variant)
on a single GPU over the CAGI5 lentiMPRA regulatory elements (2,748 HepG2 / K562 variants).
Speed is the mean of the HepG2 and K562 runs; "speed-up" is relative to running the full
AlphaGenome stack end-to-end.

Both AlphaGenome and Enformer are used as **local convolutional feature extractors**: only
the conv encoder is run (the transformer is bypassed) with a lightweight task head, either
frozen (probing) or fine-tuned. This is what makes the approach compact and fast on short
inputs — ~500× faster than the full AlphaGenome model.

| Model | Conv encoder params | Head params | Total (used) | ms / variant | Speed-up vs full AlphaGenome |
|---|---:|---:|---:|---:|---:|
| Full AlphaGenome (end-to-end) | — | — | 451 M | 5,246 | 1× |
| AlphaGenome encoder + head — probing | 90 M (frozen) | 4.7 M | 94.7 M | 10.0 | **526×** |
| AlphaGenome encoder + head — fine-tuned | 90 M | 4.7 M | 94.7 M | 9.5 | **552×** |
| Enformer conv encoder + head — fine-tuned | 51 M | 2.6 M | 54 M | 11.7 | 448× |
| MPRALegNet (trained from scratch) | — | — | 1.33 M | 1.4 | 3,862× |

Exact measured values: full AlphaGenome (all_folds) 450,553,221 params, of which the
convolutional encoder is 89,987,584 (~90 M); Enformer conv encoder (stem 1,230,337 +
conv_tower 49,674,252) = 50,904,589 (~51 M) + MPRA head (`to_tracks`) 2,632,193 (~2.6 M);
MPRALegNet 1,330,548; AlphaGenome MPRA head ~4.7 M (`configs/mpra_HepG2.json`).
**Note:** the Enformer-MPRA model sets its transformer, crop and final-pointwise layers to
`Identity` and reads embeddings straight from the conv tower
(`alphagenome_ft_mpra/enf_utils.py`), so only ~51 M runs; the checkpoint still stores the
unused transformer (~174 M), final pointwise (~5 M) and original 5,313/1,643-track output
heads (~21 M) as dead weight (full Enformer would be ~250 M). Per-cell-type ms/variant —
full AlphaGenome: HepG2 5,010 / K562 5,481; AG encoder+head probing: 10.7 / 9.3;
fine-tuned: 10.2 / 8.8; Enformer conv encoder+head (fine-tuned): 12.8 / 10.7;
MPRALegNet: 1.31 / 1.40.

Sources: `results/cagi5_evaluations/posshift20_n3_revcomp_cagi5_*_summary.csv`
(columns `time_per_variant_ms`, `variants_per_second`, `total_runtime_seconds`);
checkpoints under `results/models/checkpoints/`, `data/legnet_lentimpra/`, and the
kagglehub AlphaGenome `all_folds` checkpoint.

---

## LaTeX

```latex
\begin{table}[h!]
    \centering
    \caption{Model size and inference speed. Parameter counts are measured from the
    model checkpoints; inference speed is the per-variant wall-clock from the CAGI5
    zero-shot variant-effect evaluation on a single GPU, under a matched setting
    (3 position shifts $\times$ 2 strands per variant) over the CAGI5 lentiMPRA
    elements, averaged over the HepG2 and K562 runs. Both AlphaGenome and Enformer are
    used as local convolutional feature extractors (only the conv encoder is run; the
    transformer is bypassed) with a lightweight head; speed-up is relative to running the
    full AlphaGenome stack end-to-end \cite{avsec2026advancing}. Reusing the pretrained
    AlphaGenome encoder is $\sim$500$\times$ faster than the full model on these short
    inputs.}
    \label{tab:efficiency}
    \begin{tabular}{lrrrrr}
        \toprule
        Model & Conv encoder & Head & Total & ms/variant & Speed-up \\
        \midrule
        Full AlphaGenome (end-to-end)              & --       & --     & 451\,M  & 5246 & 1$\times$ \\
        AlphaGenome encoder + head (probing)        & 90\,M    & 4.7\,M & 94.7\,M & 10.0 & 526$\times$ \\
        AlphaGenome encoder + head (fine-tuned)     & 90\,M    & 4.7\,M & 94.7\,M & 9.5  & 552$\times$ \\
        Enformer \cite{avsec2021effective} conv encoder + head (fine-tuned) & 51\,M & 2.6\,M & 54\,M & 11.7 & 448$\times$ \\
        MPRALegNet (from scratch)                   & --       & --     & 1.33\,M & 1.4  & 3862$\times$ \\
        \bottomrule
    \end{tabular}
\end{table}
```
