#!/usr/bin/env python3
"""Phase 3 figure: new whole motifs from fine-tuning, split by partial motif conservation.

Left: the whole-motif TFs that fine-tuning (stage2) newly adds over probing
(stage1), per task, stacked into those whose SUBPART the frozen (pretrained)
encoder already detects as a first-layer filter ("conserved partial motif") vs those
with no frozen partial motif ("novel partial motif, learned de novo"). Developmental new
motifs are mostly conserved-partial motif; housekeeping mostly novel-partial motif.
Right: example logos of each mode — frozen-encoder partial motif (or ✗ if none) next
to the fine-tuned TF-MoDISco whole motif.

CPU only. Run after phase3_modisco.py + phase1 extract/tomtom.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import logomaker

REPO = Path(__file__).resolve().parents[2]
C_CONS = "#80A0C7"   # conserved partial motif (shared / present in frozen encoder)
C_NOVEL = "#A65141"  # novel partial motif (learned de novo)

# (TF, mode, frozen-filter (meme,name) or None, whole motif (meme,name), q_whole)
EXAMPLES = [
    ("AP-1 (Jra)", "conserved",
     ("results/filters/pretrained_filters.meme", "pretrained_f502"),
     ("results/modisco/stage2_dev_motifs.meme", "stage2_dev_p_pattern_2"), "1.4e-4"),
    ("SREBP", "conserved",
     ("results/filters/pretrained_filters.meme", "pretrained_f316"),
     ("results/modisco/stage2_dev_motifs.meme", "stage2_dev_p_pattern_8"), "4.7e-3"),
    ("DRE (Dref)", "novel", None,
     ("results/modisco/stage2_hk_motifs.meme", "stage2_hk_p_pattern_0"), "1.4e-5"),
    ("M1BP", "novel", None,
     ("results/modisco/stage2_hk_motifs.meme", "stage2_hk_p_pattern_1"), "4.6e-9"),
]


def _targets(tsv):
    s = set()
    if not Path(tsv).exists():
        return s
    for ln in Path(tsv).read_text().splitlines():
        if ln.startswith(("Query_ID", "#")) or not ln.strip():
            continue
        c = ln.split("\t")
        if len(c) >= 2 and c[1]:
            s.add(c[1])
    return s


def _split_counts(db="fly"):
    frozen = _targets(REPO / f"results/filters/tomtom_pretrained_vs_{db}/tomtom.tsv")
    out = {}
    for task in ("dev", "hk"):
        s1 = _targets(REPO / f"results/modisco/tomtom_stage1_{task}_vs_{db}/tomtom.tsv")
        s2 = _targets(REPO / f"results/modisco/tomtom_stage2_{task}_vs_{db}/tomtom.tsv")
        new = s2 - s1
        out[task] = (len(new & frozen), len(new - frozen))   # (conserved, novel)
    return out


def parse_meme_all(path):
    motifs, name, collecting, rows = {}, None, False, []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if s.startswith("MOTIF"):
            if name and rows:
                motifs[name] = np.array(rows, float)
            name, rows, collecting = s.split()[1], [], False
        elif s.startswith("letter-probability"):
            collecting, rows = True, []
        elif collecting:
            parts = s.split()
            if len(parts) == 4:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    collecting = False
            else:
                collecting = False
    if name and rows:
        motifs[name] = np.array(rows, float)
    return motifs


def _trim_ic(ppm, frac=0.22, pad=1, floor=0.1):
    """Crop to the motif core: keep the span between the first and last position
    whose IC is >= frac * max-IC (relative threshold isolates the core and ignores
    scattered low-IC flank positions that an absolute threshold would keep)."""
    ppm = np.clip(ppm, 1e-9, 1)
    ic = (ppm * np.log2(ppm / 0.25)).sum(1)
    thr = max(frac * ic.max(), floor)
    keep = np.where(ic >= thr)[0]
    if keep.size == 0:
        return ppm
    s, e = max(keep[0] - pad, 0), min(keep[-1] + 1 + pad, ppm.shape[0])
    return ppm[s:e]


def _logo(ax, ppm, frac=0.22):
    ppm = _trim_ic(ppm, frac=frac)
    df = pd.DataFrame(ppm, columns=["A", "C", "G", "T"])
    ic = logomaker.transform_matrix(df, from_type="probability", to_type="information")
    logomaker.Logo(ic, ax=ax, color_scheme="classic", show_spines=False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)


def _empty(ax, txt="no frozen\npartial motif (✗)"):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color="#B04030", style="italic")


def main():
    sns.set(font_scale=1.1); sns.set_style("white")
    split = _split_counts("fly")

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = gridspec.GridSpec(4, 3, width_ratios=[1.35, 1, 1.3], hspace=1.0, wspace=0.2,
                           left=0.07, right=0.985, top=0.78, bottom=0.07)

    # --- Panel A: grouped bars of NEW whole-motif TFs, conserved vs novel ---
    axb = fig.add_subplot(gs[:, 0])
    tasks = ["dev", "hk"]; x = np.arange(len(tasks)); w = 0.36
    cons = [split[t][0] for t in tasks]; nov = [split[t][1] for t in tasks]
    axb.bar(x - w/2, cons, w, color=C_CONS, edgecolor="black", lw=1, label="Conserved partial motif\n(in frozen encoder)")
    axb.bar(x + w/2, nov, w, color=C_NOVEL, edgecolor="black", lw=1, label="Novel partial motif\n(learned de novo)")
    ymax = max(max(cons), max(nov))
    for i in range(len(tasks)):
        tot = cons[i] + nov[i]
        for val, xpos in ((cons[i], x[i]-w/2), (nov[i], x[i]+w/2)):
            pct = 100*val/tot if tot else 0
            axb.text(xpos, val + ymax*0.02, f"{val} ({pct:.0f}%)", ha="center", va="bottom",
                     fontsize=8.8, fontweight="bold")
        axb.text(x[i], max(cons[i], nov[i]) + ymax*0.13, f"total: {tot}", ha="center", va="bottom",
                 fontsize=9.5, color="#333", fontweight="bold")
    axb.set_xticks(x); axb.set_xticklabels(["Developmental", "Housekeeping"], fontsize=12)
    axb.set_ylabel("New whole-motif TFs added by fine-tuning", fontsize=12)
    axb.set_ylim(0, ymax*1.36)
    axb.legend(loc="upper right", frameon=False, fontsize=8.6)
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)

    # --- Right: example logos, frozen partial motif -> fine-tuned whole motif ---
    cache = {}
    def load(m):
        return cache.setdefault(m, parse_meme_all(REPO / m))
    for i, (tf, mode, frozen, (p3m, p3n), q3) in enumerate(EXAMPLES):
        col = C_CONS if mode == "conserved" else C_NOVEL
        ax_s = fig.add_subplot(gs[i, 1]); ax_w = fig.add_subplot(gs[i, 2])
        if frozen is not None and frozen[1] in load(frozen[0]):
            _logo(ax_s, load(frozen[0])[frozen[1]], frac=0.18)
            ax_s.text(0.5, -0.2, "frozen filter", transform=ax_s.transAxes, ha="center", va="top", fontsize=7.5, color="#666")
        else:
            _empty(ax_s)
        p3 = load(p3m)
        if p3n in p3:
            _logo(ax_w, p3[p3n], frac=0.25)
            ax_w.text(0.5, -0.2, f"q={q3}", transform=ax_w.transAxes, ha="center", va="top", fontsize=7.5, color="#666")
        ax_s.set_title(tf, fontsize=10, fontweight="bold", color=col, pad=4)

    # panel-A title raised inline with the logo-column headers
    fig.text(0.235, 0.84, "New whole motifs:\nconserved vs novel partial motif", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color="#333")
    # column headers (mode is conveyed by the colored TF titles + frozen/✗ annotations)
    fig.text(0.555, 0.84, "Frozen-encoder partial motif", ha="center", fontsize=10.5, fontweight="bold", color="#333")
    fig.text(0.86, 0.84, "Fine-tuned whole motif", ha="center", fontsize=10.5, fontweight="bold", color="#333")
    
    fig.suptitle("Fine-tuning composes conserved partial motifs (developmental) and learns novel ones (housekeeping)",
                 fontsize=13.5, fontweight="bold", y=0.965)
    out = REPO / "results/modisco/phase3_motifs"
    for fmt in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=1200 if fmt == "pdf" else 300, bbox_inches="tight")
        print(f"saved {out.with_suffix('.'+fmt)}  | split(fly): {split}")


if __name__ == "__main__":
    main()
