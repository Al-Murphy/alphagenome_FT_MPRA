# Model size and inference-speed comparison (encoder-only vs full models)

Parameter counts are measured directly from the model checkpoints; inference speed is
taken from the CAGI5 zero-shot variant-effect evaluation, run under a **matched setting**
(`posshift20_n3_revcomp`: 3 position shifts × 2 strands = 6 forward passes per variant)
on a single GPU over the CAGI5 lentiMPRA regulatory elements (2,748 HepG2 / K562 variants).
Speed is the mean of the HepG2 and K562 runs; "speed-up" is relative to the full
AlphaGenome stack. Encoder-only AlphaGenome (probing or fine-tuning) reuses the ~90 M
pretrained convolutional encoder while running only its local layers, giving ~500× faster
inference than the full model on these short inputs.

| Model | Total params | Trainable params | ms / variant | variants / s | Speed-up vs full AlphaGenome |
|---|---:|---:|---:|---:|---:|
| Full AlphaGenome (encoder + transformer + decoders) | 451 M | — | 5,246 | 0.19 | 1× |
| AlphaGenome encoder + head — probing | 94.7 M | 4.7 M | 10.0 | 100 | **526×** |
| AlphaGenome encoder + head — fine-tuned | 94.7 M | 94.7 M | 9.5 | 105 | **552×** |
| Enformer + head | 305 M | (head) | 11.7 | 85 | 448× |
| MPRALegNet (trained from scratch) | 1.33 M | 1.33 M | 1.4 | 736 | 3,862× |

Exact measured values: full AlphaGenome 450,553,221 params (encoder 89,987,584);
Enformer checkpoint 304,787,784; MPRALegNet 1,330,548; AlphaGenome MPRA head ~4.7 M
(README, `configs/mpra_HepG2.json`). Per-cell-type ms/variant — full AlphaGenome:
HepG2 5,010 / K562 5,481; AG encoder+head probing: 10.7 / 9.3; fine-tuned: 10.2 / 8.8;
Enformer+head: 12.8 / 10.7; MPRALegNet: 1.31 / 1.40.

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
    elements. Speed is averaged over the HepG2 and K562 runs and the speed-up is
    relative to the full AlphaGenome stack \cite{avsec2026advancing}. Using the
    pretrained AlphaGenome encoder as a standalone feature extractor (probing or
    fine-tuning) is $\sim$500$\times$ faster than the full model on these short inputs.}
    \label{tab:efficiency}
    \begin{tabular}{lrrrr}
        \toprule
        Model & Total params & Trainable & ms/variant & Speed-up \\
        \midrule
        Full AlphaGenome                       & 451\,M  & --      & 5246 & 1$\times$ \\
        AlphaGenome encoder + head (probing)    & 94.7\,M & 4.7\,M  & 10.0 & 526$\times$ \\
        AlphaGenome encoder + head (fine-tuned) & 94.7\,M & 94.7\,M & 9.5  & 552$\times$ \\
        Enformer + head \cite{avsec2021effective} & 305\,M & (head) & 11.7 & 448$\times$ \\
        MPRALegNet (from scratch)               & 1.33\,M & 1.33\,M & 1.4  & 3862$\times$ \\
        \bottomrule
    \end{tabular}
\end{table}
```
