#!/usr/bin/env python3
"""Single figure for the cross-species motif analysis (AlphaGenome on DeepSTARR).

a  Encoder CKA vs the frozen encoder across depth (fly vs same-species human).
b  Paired first-conv filter change: each first-conv kernel (15x4) in the fine-tuned
   encoder vs the SAME filter (slot) in the frozen encoder, best-offset Pearson
   similarity of the RAW WEIGHTS (all 768 filters). Plotted as change = 1 - similarity
   on a log axis (raw weights move little, so the legible comparison is how much).
c  Per-filter scatter of the same change, fly vs human.
d  Top TF-MoDISco motifs the fine-tuned model uses (by seqlet support), developmental
   (top row) and housekeeping (bottom row), labelled by their best JASPAR-insect match.

CPU only. Needs results/cka/phase2_encoder_cka.json, results/filter_qdist/*_conv1_w.npy,
results/modisco/stage2_*_{motifs.meme,modisco.h5,tomtom...}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker

REPO = Path(__file__).resolve().parents[2]
QDIR = REPO / "results/filter_qdist"
MOD = REPO / "results/modisco"
FLYDB = REPO / "data/motifs_insects/jaspar_insects_combined.meme"
GRAY, NAVY, TERRA, STEEL = "#9E9E9E", "#394165", "#A65141", "#80A0C7"
DEVC, HKC = "#2E6E4E", "#A65141"          # developmental (green) / housekeeping (brown) colour key
TAPS = ["bin_size_1", "bin_size_2", "bin_size_4", "bin_size_8",
        "bin_size_16", "bin_size_32", "bin_size_64", "bin_size_128"]
TAP_BP = ["1", "2", "4", "8", "16", "32", "64", "128"]
MAXOFF = 4
N_MOTIF = 6      # max top motifs per task in panel d
QMAX = 1e-3      # only label motifs whose closest insect-TF match is at least this confident
# tomtom best-hit is noisy; fix calls that are biologically wrong and drop redundant ones.
LABEL_OVERRIDE = {"stage2_dev_p_pattern_0": "Dref"}  # TATCGAT is the DRE element, not pnr (GATA)
EXCLUDE = {"hr3"}  # AGTGTGACC = same Ohler-1 element already shown as M1BP


# ---------- parsing / metrics ----------
def parse_meme(path):
    out, name, collecting, rows = {}, None, False, []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if s.startswith("MOTIF"):
            if name is not None and rows:
                out[name] = np.array(rows, float)
            name, rows, collecting = s.split()[1], [], False
        elif s.startswith("letter-probability"):
            collecting, rows = True, []
        elif collecting:
            p = s.split()
            if len(p) == 4:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    collecting = False
            else:
                collecting = False
    if name is not None and rows:
        out[name] = np.array(rows, float)
    return out


def _id2name():
    m = {}
    for ln in FLYDB.read_text().splitlines():
        if ln.startswith("MOTIF"):
            p = ln.split()
            if len(p) >= 3:
                m[p[1]] = p[2]
    return m


def _matches(stage, task, id2name):
    """{query motif -> {tf_name: best q}} from the {stage} tomtom vs insect DB."""
    tsv = MOD / f"tomtom_{stage}_{task}_vs_fly" / "tomtom.tsv"
    out = {}
    if tsv.exists():
        for ln in tsv.read_text().splitlines():
            if ln.startswith(("Query_ID", "#")) or not ln.strip():
                continue
            c = ln.split("\t")
            if len(c) < 6 or not c[1]:
                continue
            try:
                q = float(c[5])
            except ValueError:
                continue
            tf = id2name.get(c[1], c[1])
            d = out.setdefault(c[0], {})
            if tf not in d or q < d[tf]:
                d[tf] = q
    return out


def _pattern_for(stage, task, tf, id2name, qmax=QMAX):
    """Highest-support motif in ({stage},{task}) whose best insect match is `tf`
    (confidently, q<qmax). Returns (ppm, q) or None if this stage did not assemble it."""
    if tf is None:
        return None
    meme = parse_meme(MOD / f"{stage}_{task}_motifs.meme")
    matches = _matches(stage, task, id2name)
    best, best_ns = None, -1
    with h5py.File(MOD / f"{stage}_{task}_modisco.h5", "r") as f:
        for grp, sgn in [("pos_patterns", "p"), ("neg_patterns", "n")]:
            if grp not in f:
                continue
            for pn in f[grp]:
                mname = f"{stage}_{task}_{sgn}_{pn}"
                mm = matches.get(mname)
                if mname not in meme or not mm:
                    continue
                if min(mm, key=mm.get) != tf or mm[tf] >= qmax:   # tf must be THE best, confident match
                    continue
                ns = int(np.array(f[grp][pn]["seqlets"]["n_seqlets"])[0])
                if ns > best_ns:
                    best_ns, best = ns, (meme[mname], mm[tf])
    return best


def _filter_sim(Wp, Wq, maxoff=MAXOFF):
    W, best = Wp.shape[0], np.nan
    for d in range(-maxoff, maxoff + 1):
        a, b = (Wp[d:], Wq[:W - d]) if d >= 0 else (Wp[:W + d], Wq[-d:])
        if a.shape[0] < 5:
            continue
        av, bv = a.ravel(), b.ravel()
        if av.std() < 1e-9 or bv.std() < 1e-9:
            continue
        r = float(np.corrcoef(av, bv)[0, 1])
        best = r if np.isnan(best) else max(best, r)
    return best


def _weight_sims():
    Wf = np.load(QDIR / "pretrained_conv1_w.npy")
    out = {}
    for name in ("fly_ft", "human_ft"):
        Wt = np.load(QDIR / f"{name}_conv1_w.npy")
        s = {f: _filter_sim(Wf[:, :, f], Wt[:, :, f]) for f in range(Wf.shape[2])}
        out[name] = {f: v for f, v in s.items() if not np.isnan(v)}
    return out


def _trim_ic(ppm, frac=0.25, pad=1, floor=0.1):
    ppm = np.clip(ppm, 1e-9, 1)
    ic = (ppm * np.log2(ppm / 0.25)).sum(1)
    thr = max(frac * ic.max(), floor)
    keep = np.where(ic >= thr)[0]
    if keep.size == 0:
        return ppm
    return ppm[max(keep[0] - pad, 0):min(keep[-1] + 1 + pad, ppm.shape[0])]


def _revcomp(ppm):
    return ppm[::-1, ::-1]                          # reverse positions + complement (ACGT->TGCA)


def _ol_corr(A, B):
    """Best gapless-overlap Pearson between two PPMs (background-subtracted, IC-trimmed)."""
    A, B = _trim_ic(A) - 0.25, _trim_ic(B) - 0.25
    la, lb, best = len(A), len(B), -2.0
    for d in range(-(la - 4), (lb - 4) + 1):
        ia, ib = max(0, -d), max(0, d)
        ov = min(la - ia, lb - ib)
        if ov < 4:
            continue
        a, b = A[ia:ia + ov].ravel(), B[ib:ib + ov].ravel()
        if a.std() > 1e-9 and b.std() > 1e-9:
            best = max(best, float(np.corrcoef(a, b)[0, 1]))
    return best


def _orient(P, Q):
    """Return P or its reverse complement, whichever aligns better to Q (same-strand display)."""
    return _revcomp(P) if _ol_corr(_revcomp(P), Q) > _ol_corr(P, Q) else P


# ---------- panels ----------
def _cka(ax):
    cka = json.loads((REPO / "results/cka/phase2_encoder_cka.json").read_text())["cka_vs_pretrained"]
    x = np.arange(len(TAPS))
    ax.plot(x, [cka["fly_ft"][t] for t in TAPS], "o-", color=TERRA, lw=2, label="Fly (S2)")
    ax.plot(x, [cka["human_ft"][t] for t in TAPS], "s--", color=NAVY, lw=2, label="Human (HepG2)")
    ax.axhline(1.0, color="#bbb", lw=1, ls=(0, (3, 3)))
    ax.set_xticks(x); ax.set_xticklabels(TAP_BP)
    ax.set_xlabel("encoder depth (bp/position)"); ax.set_ylabel("CKA vs frozen encoder")
    ax.set_ylim(0.4, 1.05); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Encoder drift by depth", fontsize=12, fontweight="bold")


def _dist(ax, sims):
    nfilt = 0
    for name, color, lab in [("fly_ft", TERRA, "Fly (S2)"), ("human_ft", NAVY, "Human (HepG2)")]:
        v = np.clip(1.0 - np.array(list(sims[name].values())), 1e-4, None)
        nfilt = v.size
        x = np.sort(v); y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, drawstyle="steps-post", color=color, lw=2,
                label=f"{lab}\n(median {np.median(v):.3f})")
        ax.axvline(np.median(v), color=color, ls=(0, (4, 3)), lw=1.3)
    ax.set_xscale("log")
    ax.set_xlabel(r"per-filter weight change  (1 $-$ similarity to frozen)")
    ax.set_ylabel("cumulative fraction of filters")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelspacing=0.9)
    ax.figure.canvas.draw()                        # realise legend extent, then tuck n= just below it
    inv = ax.transAxes.inverted()
    bb = leg.get_window_extent().transformed(inv)
    lx = leg.get_lines()[0].get_window_extent().transformed(inv).x0   # align with legend line handles
    ax.text(lx, bb.y0 - 0.02, f"n = {nfilt} filters", transform=ax.transAxes,
            va="top", fontsize=8.5, color="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("First-conv filter change", fontsize=12, fontweight="bold")


def _scatter(ax, sims):
    common = sorted(set(sims["fly_ft"]) & set(sims["human_ft"]))
    xh = np.clip(1.0 - np.array([sims["human_ft"][f] for f in common]), 1e-4, None)
    yf = np.clip(1.0 - np.array([sims["fly_ft"][f] for f in common]), 1e-4, None)
    lim = [8e-5, 1.0]
    ax.plot(lim, lim, color="#999", lw=1, ls=(0, (4, 3)))
    ax.scatter(xh, yf, s=8, alpha=0.35, color="#555", edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    above = float(np.mean(yf > xh))
    ax.text(0.04, 0.96, f"{above*100:.0f}% above diagonal\n(fly changed more)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#A0392B")
    ax.text(0.04, 0.80, f"n = {xh.size} filters", transform=ax.transAxes,
            va="top", fontsize=8.5, color="#555")
    ax.set_xlabel(r"Human (HepG2) weight change  (1 $-$ sim)")
    ax.set_ylabel(r"Fly (S2) weight change  (1 $-$ sim)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Per-filter: fly vs human change", fontsize=12, fontweight="bold")


# panel d columns: (display name, task, probing-TF, fine-tuned-TF). None = not assembled at that stage.
CONSERVED = [("GATA", "dev", "srp", "GATAe"), ("AP-1", "dev", "Jra", "Jra"),
             ("STAT", "dev", "Stat92E", "Stat92E"), ("E-box", "hk", "crp", "crp")]
DE_NOVO = [("DRE", "hk", None, "Dref"), ("M1BP", "hk", None, "M1BP")]


def _logo_cell(ax, res):
    """Draw one motif logo (res = (ppm, q)) or a 'not detected' placeholder (res = None)."""
    if res is None:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, 0.5, "not\ndetected", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, style="italic", color="#c2c2c2")
        return
    ppm, q = res
    df = pd.DataFrame(_trim_ic(ppm), columns=["A", "C", "G", "T"])
    ic = logomaker.transform_matrix(df, from_type="probability", to_type="information")
    logomaker.Logo(ic, ax=ax, color_scheme="classic", show_spines=False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)
    ax.text(0.5, -0.08, f"$q$={q:.0e}", transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color="#999")


def _motifs(fig, top, bot):
    """Panel d: same whole motifs before (probing = frozen encoder) vs after fine-tuning,
    split into those the pretrained encoder already assembled (conserved) vs learned de novo."""
    id2name = _id2name()
    gsc = fig.add_gridspec(2, len(CONSERVED), left=0.10, right=0.66, top=top, bottom=bot, hspace=0.55, wspace=0.28)
    gsd = fig.add_gridspec(2, len(DE_NOVO), left=0.735, right=0.98, top=top, bottom=bot, hspace=0.55, wspace=0.28)

    def block(gs, cols, rowlab):
        for ci, (name, task, ptf, ftf) in enumerate(cols):
            pr = _pattern_for("stage1", task, ptf, id2name)
            ft = _pattern_for("stage2", task, ftf, id2name)
            if pr and ft:                          # show both on the same strand
                pr = (_orient(pr[0], ft[0]), pr[1])
            for ri, res in enumerate([pr, ft]):
                ax = fig.add_subplot(gs[ri, ci])
                _logo_cell(ax, res)
                if ri == 0:
                    ax.set_title(name, fontsize=9.5, fontweight="bold", pad=4,
                                 color=DEVC if task == "dev" else HKC)
                if ci == 0 and rowlab:
                    ax.set_ylabel(["Probing\n(frozen)", "Fine-tuned"][ri], fontsize=9,
                                  fontweight="bold", color="#555", rotation=90, labelpad=8)
    block(gsc, CONSERVED, True); block(gsd, DE_NOVO, False)

    hy = top + 0.028
    fig.text((0.10 + 0.66) / 2, hy, "Already present in pretrained encoder (conserved)",
             ha="center", fontsize=10.5, fontweight="bold", color="#333")
    fig.text((0.735 + 0.98) / 2, hy, "Learned de novo",
             ha="center", fontsize=10.5, fontweight="bold", color="#333")


ATTR = [("DRE", "hk", "TATCGATA"), ("M1BP", "hk", "AGTGTGACC")]   # panel e de novo examples


def _attr_data(task, motif, half=13):
    """Highest FT-vs-probing contrast occurrence of `motif`; returns per-base attribution
    (onehot*hyp) windows for probing and fine-tuned on the SAME enhancer + motif span."""
    s1 = np.load(MOD / f"stage1_{task}.npz"); s2 = np.load(MOD / f"stage2_{task}.npz")
    oh, h1, h2 = s1["onehot"], s1["hyp"], s2["hyp"]
    c1, c2 = (oh * h1).sum(-1), (oh * h2).sum(-1)
    ml = len(motif); rc = motif.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    seqs = ["".join("ACGT"[b] for b in x) for x in oh.argmax(-1)]
    best, bestd = None, -1e9
    for i, sq in enumerate(seqs):
        for m in ({motif} if motif == rc else {motif, rc}):
            j = sq.find(m)
            if j >= 0 and (c2[i, j:j + ml].sum() - c1[i, j:j + ml].sum()) > bestd:
                bestd, best = c2[i, j:j + ml].sum() - c1[i, j:j + ml].sum(), (i, j, m)
    i, j, m = best
    c = j + ml // 2
    lo, hi = max(0, c - half), min(oh.shape[1], c + half + 1)
    prw, ftw = (oh * h1)[i, lo:hi], (oh * h2)[i, lo:hi]
    span = (j - lo, j - lo + ml)
    if m == rc:                                    # display on the motif's + strand
        prw, ftw = prw[::-1, ::-1], ftw[::-1, ::-1]
        W = prw.shape[0]; span = (W - span[1], W - span[0])
    return prw, ftw, span


def _attr_logo(ax, contrib, ylim, span):
    ax.axvspan(span[0] - 0.5, span[1] - 0.5, color="#F2D24E", alpha=0.28, zorder=0)
    logomaker.Logo(pd.DataFrame(contrib, columns=["A", "C", "G", "T"]), ax=ax,
                   color_scheme="classic", flip_below=False, show_spines=False)
    ax.axhline(0, color="#bbb", lw=0.6, zorder=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(*ylim)


def _attr_panel(fig, top, bot):
    gs = fig.add_gridspec(2, len(ATTR), left=0.11, right=0.90, top=top, bottom=bot, hspace=0.28, wspace=0.16)
    for ci, (name, task, motif) in enumerate(ATTR):
        pr, ft, span = _attr_data(task, motif)
        mag = max(np.abs(pr).max(), np.abs(ft).max())
        ylim = (-0.1 * mag, 1.05 * mag)
        for ri, contrib in enumerate([pr, ft]):
            ax = fig.add_subplot(gs[ri, ci])
            _attr_logo(ax, contrib, ylim, span)
            if ri == 0:
                ax.set_title(f"{name}  (housekeeping)", fontsize=9.5, fontweight="bold", color=HKC, pad=4)
            if ci == 0:
                ax.set_ylabel(["Probing\n(frozen)", "Fine-tuned"][ri], fontsize=9,
                              fontweight="bold", color="#555", rotation=90, labelpad=8)


def main():
    sns.set(font_scale=1.0); sns.set_style("white")
    sims = _weight_sims()
    fig = plt.figure(figsize=(16, 13))
    gs_top = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.94, bottom=0.71, wspace=0.30)
    axA = fig.add_subplot(gs_top[0, 0]); axB = fig.add_subplot(gs_top[0, 1]); axC = fig.add_subplot(gs_top[0, 2])
    _cka(axA); _dist(axB, sims); _scatter(axC, sims)
    _motifs(fig, top=0.59, bot=0.43)
    _attr_panel(fig, top=0.35, bot=0.22)
    for ax, L in zip((axA, axB, axC), "abc"):
        ax.text(-0.13, 1.04, L, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    # panel d header + dev/hk colour key
    fig.text(0.028, 0.655, "d", fontsize=15, fontweight="bold")
    fig.text(0.50, 0.655, "Whole motifs before (probing) vs after fine-tuning",
             ha="center", fontsize=11, fontweight="bold", color="#333")
    fig.text(0.405, 0.637, "developmental", ha="right", fontsize=8.5, fontweight="bold", color=DEVC)
    fig.text(0.42, 0.637, "|", ha="center", fontsize=8.5, color="#bbb")
    fig.text(0.435, 0.637, "housekeeping", ha="left", fontsize=8.5, fontweight="bold", color=HKC)
    # panel e header
    fig.text(0.028, 0.39, "e", fontsize=15, fontweight="bold")
    fig.text(0.50, 0.39, "Per-base attribution on enhancers containing a de novo motif",
             ha="center", fontsize=11, fontweight="bold", color="#333")
    fig.suptitle("Cross-species fine-tuning: encoder drift, first-conv filter change and learned motifs",
                 fontsize=14.5, fontweight="bold", y=0.985)
    for fmt in ("png", "pdf"):
        f = REPO / f"results/filter_qdist/filter_motif_figure.{fmt}"
        fig.savefig(f, dpi=1200 if fmt == "pdf" else 220, bbox_inches="tight")
        print(f"saved {f}")
    for name in ("fly_ft", "human_ft"):
        v = 1 - np.array(list(sims[name].values()))
        print(f"  {name}: median weight CHANGE={np.median(v):.4f}  n={v.size}")


if __name__ == "__main__":
    main()
