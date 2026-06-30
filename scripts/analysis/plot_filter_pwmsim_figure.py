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


def _matches(task, id2name):
    """{query motif -> {tf_name: best q}} from the stage2 tomtom vs insect DB."""
    tsv = MOD / f"tomtom_stage2_{task}_vs_fly" / "tomtom.tsv"
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


def _top_motifs(task, n, id2name):
    """Top-n motifs by seqlet support, keeping only confident, sensible TF matches.

    Drops patterns whose best insect match is weak (q >= QMAX); applies LABEL_OVERRIDE
    for tomtom calls that are biologically wrong (e.g. the TATCGAT element is DRE/Dref,
    not the GATA factor pnr). For an overridden label the displayed q is that TF's own
    q for the pattern. Returns [(ppm, tf_label, q), ...]."""
    meme = parse_meme(MOD / f"stage2_{task}_motifs.meme")
    matches = _matches(task, id2name)
    rows = []
    with h5py.File(MOD / f"stage2_{task}_modisco.h5", "r") as f:
        for grp, sgn in [("pos_patterns", "p"), ("neg_patterns", "n")]:
            if grp not in f:
                continue
            for pn in f[grp]:
                ns = int(np.array(f[grp][pn]["seqlets"]["n_seqlets"])[0])
                mname = f"stage2_{task}_{sgn}_{pn}"
                mm = matches.get(mname)
                if mname not in meme or not mm:
                    continue
                best_q = min(mm.values())
                lab = LABEL_OVERRIDE.get(mname, min(mm, key=mm.get))
                if lab in EXCLUDE or best_q >= QMAX:   # confidence judged on the pattern's best match
                    continue
                rows.append((ns, meme[mname], lab, mm.get(lab, best_q)))
    rows.sort(key=lambda r: r[0], reverse=True)
    seen, out = set(), []                         # one entry per TF (highest-support pattern)
    for ns, ppm, lab, q in rows:
        if lab in seen:
            continue
        seen.add(lab)
        out.append((ppm, lab, q, ns))
        if len(out) >= n:
            break
    return out


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


def _logo(ax, ppm, label, q):
    df = pd.DataFrame(_trim_ic(ppm), columns=["A", "C", "G", "T"])
    ic = logomaker.transform_matrix(df, from_type="probability", to_type="information")
    logomaker.Logo(ic, ax=ax, color_scheme="classic", show_spines=False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)
    ax.set_title(label, fontsize=8.5, fontweight="bold", color="#333", pad=2)
    ax.text(0.5, -0.06, f"$q$={q:.0e}", transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color="#888")


def _motifs(fig, top, bot):
    """Panel d split into column groups: Shared (dev+hk) | Developmental | Housekeeping."""
    id2name = _id2name()
    dev = _top_motifs("dev", 8, id2name)
    hk = _top_motifs("hk", 8, id2name)
    devm = {l: (p, q, n) for p, l, q, n in dev}
    hkm = {l: (p, q, n) for p, l, q, n in hk}
    shared_labs = sorted((set(devm) & set(hkm)),
                         key=lambda l: max(devm[l][2], hkm[l][2]), reverse=True)[:2]

    def pick(l):                                   # shared logo from whichever task has more seqlets
        a, b = devm.get(l), hkm.get(l)
        return a if a[2] >= b[2] else b
    shared = [(pick(l)[0], l, pick(l)[1]) for l in shared_labs]
    dev_only = [(p, l, q) for p, l, q, _n in dev if l not in shared_labs][:4]
    hk_only = [(p, l, q) for p, l, q, _n in hk if l not in shared_labs][:2]

    gss = fig.add_gridspec(2, 1, left=0.07, right=0.255, top=top, bottom=bot, hspace=0.85)
    gsd = fig.add_gridspec(2, 2, left=0.315, right=0.735, top=top, bottom=bot, hspace=0.85, wspace=0.3)
    gsh = fig.add_gridspec(2, 1, left=0.795, right=0.98, top=top, bottom=bot, hspace=0.85)

    def place(gs, items, ncol):
        for i in range(gs.nrows * ncol):
            r, c = divmod(i, ncol)
            ax = fig.add_subplot(gs[r, c])
            if i < len(items):
                _logo(ax, *items[i])
            else:
                ax.axis("off")
    place(gss, shared, 1); place(gsd, dev_only, 2); place(gsh, hk_only, 1)

    hy = top + 0.025
    fig.text(0.1625, hy, "Shared (dev + hk)", ha="center", fontsize=10.5, fontweight="bold", color="#5A5A5A")
    fig.text(0.525, hy, "Developmental", ha="center", fontsize=10.5, fontweight="bold", color="#2E6E4E")
    fig.text(0.8875, hy, "Housekeeping", ha="center", fontsize=10.5, fontweight="bold", color="#8A5A2B")


def main():
    sns.set(font_scale=1.0); sns.set_style("white")
    sims = _weight_sims()
    fig = plt.figure(figsize=(16, 9))
    gs_top = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.90, bottom=0.56, wspace=0.30)
    axA = fig.add_subplot(gs_top[0, 0]); axB = fig.add_subplot(gs_top[0, 1]); axC = fig.add_subplot(gs_top[0, 2])
    _cka(axA); _dist(axB, sims); _scatter(axC, sims)
    _motifs(fig, top=0.40, bot=0.05)
    for ax, L in zip((axA, axB, axC), "abc"):
        ax.text(-0.13, 1.06, L, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    fig.text(0.06, 0.475, "d", fontsize=15, fontweight="bold")
    fig.text(0.55, 0.475, "Sequence motifs the fine-tuned model relies on "
             "(TF-MoDISco on attributions; labelled by closest insect TF, $q$)",
             ha="center", fontsize=11, fontweight="bold", color="#333")
    fig.suptitle("Cross-species fine-tuning: encoder drift, first-conv filter change and learned motifs",
                 fontsize=14.5, fontweight="bold", y=0.975)
    for fmt in ("png", "pdf"):
        f = REPO / f"results/filter_qdist/filter_motif_figure.{fmt}"
        fig.savefig(f, dpi=1200 if fmt == "pdf" else 220, bbox_inches="tight")
        print(f"saved {f}")
    for name in ("fly_ft", "human_ft"):
        v = 1 - np.array(list(sims[name].values()))
        print(f"  {name}: median weight CHANGE={np.median(v):.4f}  n={v.size}")


if __name__ == "__main__":
    main()
