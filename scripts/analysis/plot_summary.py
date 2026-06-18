#!/usr/bin/env python3
"""Single multi-panel summary of the Drosophila motif-transfer analysis (Phases 0-3).

A: Phase 0  — probing vs fine-tuning performance (DeepSTARR dev/hk).
B: Phase 1  — first-layer partial-motif conservation (fly vs same-species human).
C: Phase 2  — encoder representational drift by depth (CKA vs frozen encoder).
D: Phase 3  — new whole motifs split by conserved vs novel partial motif.
E: Phase 3  — example whole motifs fine-tuning builds (conserved dev | novel hk).

CPU only. Regenerates from saved phase outputs.
"""
from __future__ import annotations

import csv
import json
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
GRAY, NAVY, TERRA, STEEL = "#9E9E9E", "#394165", "#A65141", "#80A0C7"
TAPS = ["bin_size_1", "bin_size_2", "bin_size_4", "bin_size_8", "bin_size_16", "bin_size_32", "bin_size_64", "bin_size_128"]
TAP_BP = ["1", "2", "4", "8", "16", "32", "64", "128"]

# whole-motif examples for panel E: (label, color, meme, motif, q, note)
MOTIFS = [
    ("AP-1 (Jra)", STEEL, "stage2_dev_p_pattern_2", "1.4e-4", "frozen partial motif"),
    ("SREBP", STEEL, "stage2_dev_p_pattern_8", "4.7e-3", "frozen partial motif"),
    ("DRE (Dref)", TERRA, "stage2_hk_p_pattern_0", "1.4e-5", "no frozen partial motif"),
    ("M1BP", TERRA, "stage2_hk_p_pattern_1", "4.6e-9", "no frozen partial motif"),
]
MOTIF_MEME = {"dev": REPO / "results/modisco/stage2_dev_motifs.meme",
              "hk": REPO / "results/modisco/stage2_hk_motifs.meme"}


# ---------- shared helpers ----------
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
        out[task] = (len(new & frozen), len(new - frozen))
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


def _trim_ic(ppm, frac=0.25, pad=1, floor=0.1):
    ppm = np.clip(ppm, 1e-9, 1)
    ic = (ppm * np.log2(ppm / 0.25)).sum(1)
    thr = max(frac * ic.max(), floor)
    keep = np.where(ic >= thr)[0]
    if keep.size == 0:
        return ppm
    return ppm[max(keep[0]-pad, 0):min(keep[-1]+1+pad, ppm.shape[0])]


def _logo(ax, ppm):
    df = pd.DataFrame(_trim_ic(ppm), columns=["A", "C", "G", "T"])
    ic = logomaker.transform_matrix(df, from_type="probability", to_type="information")
    logomaker.Logo(ic, ax=ax, color_scheme="classic", show_spines=False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)


def _panel_letter(ax, L):
    # place in FIGURE coords so the offset above the panel top is constant
    # (axes-relative offsets drift for the half-height panels)
    pos = ax.get_position()
    ax.figure.text(pos.x0 - 0.028, pos.y1 + 0.013, L, fontsize=15, fontweight="bold", va="bottom", ha="left")


# ---------- panels ----------
def panel_gap(ax):
    def read(p):
        return next(csv.DictReader(open(p)))
    s1 = read(REPO / "results/test_predictions/starrseq/stage1_deepstarr_test_metrics.csv")
    s2 = read(REPO / "results/test_predictions/starrseq/stage2_deepstarr_test_metrics.csv")
    tasks = ["dev", "hk"]; x = np.arange(2); w = 0.36
    pr = [float(s1[f"{t}_pearson"]) for t in tasks]; ft = [float(s2[f"{t}_pearson"]) for t in tasks]
    ax.bar(x-w/2, pr, w, color=GRAY, edgecolor="black", lw=1, label="Probing (frozen)")
    ax.bar(x+w/2, ft, w, color=NAVY, edgecolor="black", lw=1, label="Fine-tuned")
    for xi, a, b in zip(x, pr, ft):
        ax.text(xi-w/2, a+0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi+w/2, b+0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Dev", "Hk"]); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Test Pearson's r")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.0), borderaxespad=0.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Probing vs fine-tuning", fontsize=11, fontweight="bold")


def _filters_with_hit(tsv):
    qs = set()
    if not Path(tsv).exists():
        return 0
    for ln in Path(tsv).read_text().splitlines():
        if ln.startswith(("Query_ID", "#")) or not ln.strip():
            continue
        q = ln.split("\t")[0]
        if q:
            qs.add(q)
    return len(qs)


def panel_tomtom_rate(ax):
    summary = json.loads((REPO / "results/filters/phase1_extract_summary.json").read_text())
    models = [("pretrained", "Frozen", GRAY), ("fly_ft", "Fly (S2)", TERRA), ("human_ft", "Human", NAVY)]
    dbs = [("vertebrate", "Vertebrate"), ("fly", "Insect")]
    x = np.arange(len(dbs)); w = 0.26
    for i, (m, lab, color) in enumerate(models):
        n = summary[m]["n_motifs"]
        vals = [100 * _filters_with_hit(REPO / f"results/filters/tomtom_{m}_vs_{db}/tomtom.tsv") / n
                for db, _ in dbs]
        bars = ax.bar(x + (i-1)*w, vals, w, color=color, edgecolor="black", lw=0.8, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=color)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in dbs]); ax.set_ylim(0, 70)
    ax.set_ylabel("Filters matching a known TF (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Shared partial-motif vocabulary", fontsize=11, fontweight="bold")


def panel_conservation(ax):
    bins = np.linspace(0.2, 1.0, 33)
    for name, color, lab in [("fly_ft", TERRA, "Fly (S2)"), ("human_ft", NAVY, "Human (HepG2)")]:
        mc = np.load(REPO / f"results/filters/conservation_pretrained_vs_{name}.npz")["matched_corr"]
        ax.hist(mc, bins=bins, histtype="stepfilled", alpha=0.5, color=color, lw=1.3,
                edgecolor=color, label=f"{lab} (μ={mc.mean():.2f})")
        ax.axvline(mc.mean(), color=color, ls=(0, (4, 3)), lw=1.4)
    ax.set_xlim(0.2, 1.02); ax.set_xlabel("Per-filter match vs frozen encoder")
    ax.set_ylabel("First-layer filters"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Partial-motif conservation", fontsize=11, fontweight="bold")


def panel_cka(ax):
    cka = json.loads((REPO / "results/cka/phase2_encoder_cka.json").read_text())["cka_vs_pretrained"]
    x = np.arange(len(TAPS))
    ax.plot(x, [cka["fly_ft"][t] for t in TAPS], "o-", color=TERRA, lw=2, label="Fly (S2)")
    ax.plot(x, [cka["human_ft"][t] for t in TAPS], "s--", color=NAVY, lw=2, label="Human (HepG2)")
    ax.axhline(1.0, color="#bbb", lw=1, ls=(0, (3, 3)))
    ax.set_xticks(x); ax.set_xticklabels(TAP_BP); ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("encoder depth (bp/position)"); ax.set_ylabel("CKA vs frozen encoder")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Encoder drift by depth", fontsize=11, fontweight="bold")


def panel_split(ax):
    sp = _split_counts("fly"); tasks = ["dev", "hk"]; x = np.arange(2); w = 0.36
    cons = [sp[t][0] for t in tasks]; nov = [sp[t][1] for t in tasks]
    ax.bar(x-w/2, cons, w, color=STEEL, edgecolor="black", lw=1, label="Conserved partial motif")
    ax.bar(x+w/2, nov, w, color=TERRA, edgecolor="black", lw=1, label="Novel partial motif")
    ymax = max(max(cons), max(nov))
    for i in range(2):
        tot = cons[i]+nov[i]
        for v, xi in ((cons[i], x[i]-w/2), (nov[i], x[i]+w/2)):
            ax.text(xi, v+ymax*0.02, f"{v} ({100*v/tot:.0f}%)", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.text(x[i], max(cons[i], nov[i])+ymax*0.13, f"total {tot}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(["Dev", "Hk"]); ax.set_ylim(0, ymax*1.4)
    ax.set_ylabel("New whole-motif TFs"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Conserved vs novel partial motif", fontsize=11, fontweight="bold")


def panel_motifs(axes):
    cache = {k: parse_meme_all(v) for k, v in MOTIF_MEME.items()}
    for ax, (lab, color, mname, q, note) in zip(axes, MOTIFS):
        task = "dev" if "dev" in mname else "hk"
        ppm = cache[task].get(mname)
        if ppm is not None:
            _logo(ax, ppm)
        ax.set_title(lab, fontsize=10.5, fontweight="bold", color=color, pad=12)
        ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=color)
        ax.text(0.5, -0.18, f"q={q}", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#666")


def main():
    sns.set(font_scale=1.0); sns.set_style("white")
    fig = plt.figure(figsize=(16.5, 9))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.42], hspace=0.55,
                              left=0.055, right=0.985, top=0.86, bottom=0.06)
    top = outer[0].subgridspec(2, 4, wspace=0.42, hspace=0.7)
    bot = outer[1].subgridspec(1, 4, wspace=0.22)

    axA = fig.add_subplot(top[0, 0])   # P0 gap (half height)
    axB = fig.add_subplot(top[1, 0])   # P1 conservation histogram (half height)
    axC = fig.add_subplot(top[:, 1])   # P1 shared vocabulary
    axD = fig.add_subplot(top[:, 2])   # P2 CKA depth
    axE = fig.add_subplot(top[:, 3])   # P3 split
    panel_gap(axA); panel_conservation(axB); panel_tomtom_rate(axC); panel_cka(axD); panel_split(axE)
    for ax, L in zip((axA, axB, axC, axD, axE), "abcde"):
        _panel_letter(ax, L)

    axF = [fig.add_subplot(bot[i]) for i in range(4)]
    panel_motifs(axF)
    _panel_letter(axF[0], "f")
    # short strip label placed in the inter-row gap (centered, clear of x-labels)
    fig.text(0.5, 0.33, "Example whole motifs built by fine-tuning   "
             "(blue = developmental / conserved · red = housekeeping / de novo)",
             fontsize=10, fontweight="bold", color="#333", ha="center")

    fig.suptitle("AlphaGenome fine-tuned on Drosophila DeepSTARR: conserved partial motifs recomposed (developmental), "
                 "novel ones learned (housekeeping)", fontsize=13.5, fontweight="bold", y=0.975)
    out = REPO / "results/summary_figure"
    for fmt in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=1200 if fmt == "pdf" else 220, bbox_inches="tight")
        print(f"saved {out.with_suffix('.'+fmt)}")


if __name__ == "__main__":
    main()
