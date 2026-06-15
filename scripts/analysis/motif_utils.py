"""Motif/filter utilities for Phase 1 (first-layer conv filter interpretation).

Pure numpy/scipy — framework-agnostic, CPU-testable. Implements the Koo-lab
style activation-based filter visualization: collect the subsequences that
maximally activate each convolutional filter, build a PFM/PPM, score its
information content, write MEME for TOMTOM, and quantify how conserved a
model's filters are vs another model's (Hungarian matching on activation
correlation). Also builds a combined MEME database from per-motif files.

Alphabet order is ACGT throughout (matches the repo's one-hot + JASPAR memes).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHABET = "ACGT"
MEME_TOMTOM = Path.home() / "local/meme/bin/tomtom"


def run_tomtom(query_meme, db_meme, outdir, thresh=0.1, min_overlap=3, meme_bin=MEME_TOMTOM):
    """Run MEME-suite TOMTOM (query motifs vs a MEME database). Returns the out dir.

    Parses tomtom.tsv and returns (n_query_with_hit, n_total_matches).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(meme_bin), "-no-ssc", "-oc", str(outdir), "-thresh", str(thresh),
           "-min-overlap", str(min_overlap), str(query_meme), str(db_meme)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tsv = outdir / "tomtom.tsv"
    qs, n = set(), 0
    if tsv.exists():
        for ln in tsv.read_text().splitlines():
            if not ln or ln.startswith("#") or ln.startswith("Query_ID"):
                continue
            p = ln.split("\t")
            if len(p) >= 2 and p[1]:
                qs.add(p[0]); n += 1
    return outdir, len(qs), n


# --------------------------------------------------------------------------- #
# Activation -> PFM
# --------------------------------------------------------------------------- #
def filter_pfm(seqs_onehot, acts, filt, window, threshold_frac=0.5, min_sites=10,
               max_sites=None):
    """Build a PFM for one filter from its max-activating subsequences.

    Args:
        seqs_onehot: (N, L, 4) one-hot sequences (the same passed through the encoder).
        acts:        (N, L, F) first-layer activations (full resolution, F filters).
        filt:        filter index into F.
        window:      subsequence width to extract (~ filter receptive field).
        threshold_frac: keep positions where act >= threshold_frac * global-max-for-this-filter.
        min_sites:   return None if fewer activating sites than this.
        max_sites:   optionally cap number of sites (takes the highest-activation ones).

    Returns:
        dict {pfm (window,4) counts, n_sites, max_act} or None.
    """
    a = acts[:, :, filt]                      # (N, L)
    amax = float(a.max())
    if amax <= 0:
        return None
    thr = threshold_frac * amax
    half = window // 2
    N, L = a.shape
    ns, ls = np.where(a >= thr)               # activating (seq, pos) pairs
    # keep windows fully inside the sequence
    keep = (ls - half >= 0) & (ls - half + window <= L)
    ns, ls = ns[keep], ls[keep]
    if max_sites is not None and ns.size > max_sites:
        order = np.argsort(a[ns, ls])[::-1][:max_sites]
        ns, ls = ns[order], ls[order]
    if ns.size < min_sites:
        return None
    pfm = np.zeros((window, 4), np.float64)
    for n, p in zip(ns, ls):
        pfm += seqs_onehot[n, p - half:p - half + window, :]
    return {"pfm": pfm, "n_sites": int(ns.size), "max_act": amax}


def pfm_to_ppm(pfm, pseudocount=0.0):
    """Normalize a count matrix (W,4) to probabilities row-wise."""
    pfm = np.asarray(pfm, np.float64) + pseudocount
    rs = pfm.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    return pfm / rs


def information_content(ppm, background=(0.25, 0.25, 0.25, 0.25)):
    """Total information content (bits) of a PPM (W,4), small-sample uncorrected."""
    ppm = np.clip(np.asarray(ppm, np.float64), 1e-9, 1.0)
    bg = np.asarray(background, np.float64)
    ic_per_pos = (ppm * np.log2(ppm / bg)).sum(1)
    return float(ic_per_pos.sum())


def trim_to_ic(ppm, min_ic=0.2, background=(0.25, 0.25, 0.25, 0.25)):
    """Trim low-information flanks of a PPM (returns trimmed ppm + (start,end))."""
    ppm = np.asarray(ppm, np.float64)
    bg = np.asarray(background, np.float64)
    per_pos = (np.clip(ppm, 1e-9, 1) * np.log2(np.clip(ppm, 1e-9, 1) / bg)).sum(1)
    keep = np.where(per_pos >= min_ic)[0]
    if keep.size == 0:
        return ppm, (0, ppm.shape[0])
    s, e = int(keep[0]), int(keep[-1]) + 1
    return ppm[s:e], (s, e)


# --------------------------------------------------------------------------- #
# MEME I/O
# --------------------------------------------------------------------------- #
def write_meme(motifs, path, background=(0.25, 0.25, 0.25, 0.25)):
    """Write motifs to a MEME v4 file.

    motifs: list of dicts {name, ppm (W,4), nsites (optional)}.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a, c, g, t = background
    with open(path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write(f"A {a:.5f} C {c:.5f} G {g:.5f} T {t:.5f}\n\n")
        for m in motifs:
            ppm = np.asarray(m["ppm"], np.float64)
            w = ppm.shape[0]
            nsites = int(m.get("nsites", 20))
            f.write(f"MOTIF {m['name']}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {w} nsites= {nsites} E= 0\n")
            for row in ppm:
                f.write(" " + " ".join(f"{x:.6f}" for x in row) + "\n")
            f.write("\n")
    return path


def build_combined_meme(motif_dir, out_path, background=(0.25, 0.25, 0.25, 0.25)):
    """Concatenate per-motif MEME files (e.g. data/motifs/MA*.meme) into one DB."""
    motif_dir = Path(motif_dir)
    files = sorted(motif_dir.glob("*.meme"))
    blocks = []
    for fp in files:
        lines = fp.read_text().splitlines()
        # keep from the first 'MOTIF' line to end of that file's matrix block
        try:
            start = next(i for i, ln in enumerate(lines) if ln.startswith("MOTIF"))
        except StopIteration:
            continue
        blocks.append("\n".join(lines[start:]).strip())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a, c, g, t = background
    with open(out_path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write(f"A {a:.5f} C {c:.5f} G {g:.5f} T {t:.5f}\n\n")
        for b in blocks:
            f.write(b + "\n\n")
    return out_path, len(blocks)


# --------------------------------------------------------------------------- #
# Cross-model filter conservation (Hungarian matching on activation correlation)
# --------------------------------------------------------------------------- #
def filter_activation_summary(acts, mode="max"):
    """Reduce (N, L, F) activations to a per-sequence per-filter matrix (N, F)."""
    if mode == "max":
        return acts.max(1)
    if mode == "mean":
        return acts.mean(1)
    raise ValueError(mode)


def _corr_matrix(A, B):
    """Pearson corr between every column of A (N,F) and every column of B (N,F) -> (F,F)."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return A.T @ B  # (F, F)


def hungarian_filter_match(acts_a, acts_b, reduce_mode="max"):
    """Match model A's filters to model B's by activation correlation (optimal assignment).

    acts_a, acts_b: (N, L, F) activations on the SAME N sequences.
    Returns dict with the assignment, the matched correlation per filter, and the
    diagonal (same-index) correlation for reference.
    """
    SA = filter_activation_summary(acts_a, reduce_mode)  # (N,F)
    SB = filter_activation_summary(acts_b, reduce_mode)
    C = _corr_matrix(SA, SB)                              # (F,F)
    row, col = linear_sum_assignment(-C)                 # maximize total correlation
    matched_corr = C[row, col]
    diag_corr = np.diag(C)
    return {
        "assignment": np.stack([row, col], 1),           # (F,2) a_idx -> b_idx
        "matched_corr": matched_corr,                    # best-assignment corr per A filter
        "diag_corr": diag_corr,                          # same-index corr (filters didn't move)
        "mean_matched_corr": float(matched_corr.mean()),
        "mean_diag_corr": float(diag_corr.mean()),
    }
