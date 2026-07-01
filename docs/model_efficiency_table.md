# Model size and inference-speed comparison

Parameter counts are measured directly from the model checkpoints; inference speed is
taken from the CAGI5 zero-shot variant-effect evaluation, run under a **matched setting**
(`posshift20_n3_revcomp`: 3 position shifts × 2 strands = 6 forward passes per variant)
on a single GPU over the CAGI5 lentiMPRA regulatory elements (2,748 HepG2 / K562 variants).
Speed is the mean of the HepG2 and K562 runs; "speed-up" is relative to running the full
AlphaGenome stack end-to-end.

Both AlphaGenome and Enformer are used the same way — the pretrained backbone as a
feature extractor with a lightweight task head, either frozen (probing) or updated
(fine-tuning). Reusing the ~90 M AlphaGenome convolutional encoder this way gives ~500×
faster inference than the full AlphaGenome model on these short inputs.

| Model | Backbone params | Head params | Total | ms / variant | Speed-up vs full AlphaGenome |
|---|---:|---:|---:|---:|---:|
| Full AlphaGenome (end-to-end) | — | — | 451 M | 5,246 | 1× |
| AlphaGenome encoder + head — probing | 90 M (frozen) | 4.7 M | 94.7 M | 10.0 | **526×** |
| AlphaGenome encoder + head — fine-tuned | 90 M | 4.7 M | 94.7 M | 9.5 | **552×** |
| Enformer encoder + head — fine-tuned | 281 M | 2.6 M | 283 M | 11.7 | 448× |
| MPRALegNet (trained from scratch) | — | — | 1.33 M | 1.4 | 3,862× |

Exact measured values: full AlphaGenome (all_folds) 450,553,221 params, of which the
convolutional encoder is 89,987,584 (~90 M); Enformer trunk 280,779,803 (~281 M) + MPRA
head (`to_tracks`) 2,632,193 (~2.6 M); MPRALegNet 1,330,548; AlphaGenome MPRA head ~4.7 M
(`configs/mpra_HepG2.json`). **Note:** the Enformer checkpoint additionally retains its
original 5,313-track human and 1,643-track mouse output heads (~21 M) that are unused for
the MPRA task; these are excluded from the functional count above (the full checkpoint is
304,787,784). Per-cell-type ms/variant — full AlphaGenome: HepG2 5,010 / K562 5,481;
AG encoder+head probing: 10.7 / 9.3; fine-tuned: 10.2 / 8.8; Enformer encoder+head
(fine-tuned): 12.8 / 10.7; MPRALegNet: 1.31 / 1.40.

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
    used as a pretrained backbone plus a lightweight head (probing or fine-tuning);
    speed-up is relative to running the full AlphaGenome stack end-to-end
    \cite{avsec2026advancing}. The Enformer trunk is reported without its unused original
    human/mouse output heads. Reusing the pretrained AlphaGenome encoder is
    $\sim$500$\times$ faster than the full model on these short inputs.}
    \label{tab:efficiency}
    \begin{tabular}{lrrrrr}
        \toprule
        Model & Backbone & Head & Total & ms/variant & Speed-up \\
        \midrule
        Full AlphaGenome (end-to-end)              & --       & --     & 451\,M  & 5246 & 1$\times$ \\
        AlphaGenome encoder + head (probing)        & 90\,M    & 4.7\,M & 94.7\,M & 10.0 & 526$\times$ \\
        AlphaGenome encoder + head (fine-tuned)     & 90\,M    & 4.7\,M & 94.7\,M & 9.5  & 552$\times$ \\
        Enformer \cite{avsec2021effective} encoder + head (fine-tuned) & 281\,M & 2.6\,M & 283\,M & 11.7 & 448$\times$ \\
        MPRALegNet (from scratch)                   & --       & --     & 1.33\,M & 1.4  & 3862$\times$ \\
        \bottomrule
    \end{tabular}
\end{table}
```
