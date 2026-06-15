#!/usr/bin/env python3
"""Phase 3 (CPU stage) — TF-MoDISco motif discovery + TOMTOM on attribution maps.

Takes the per-condition attribution arrays written by phase3_attributions.py
(results/modisco/{cond}_{task}.npz with keys onehot, hyp), runs TF-MoDISco-lite
to discover de-novo *whole* motifs, extracts each motif's PPM, and TOMTOMs them
against the fly (insect) and vertebrate JASPAR DBs. The decisive question: which
whole motifs does the fine-tuned (stage2) model discover that probing (stage1)
does not — and are the new ones fly-specific (housekeeping DRE/Ohler, etc.)?

No jax — runs on a login/CPU node. Run phase3_attributions.py (GPU) first.

Usage:
  python scripts/analysis/phase3_modisco.py --cond stage2 --task dev
  python scripts/analysis/phase3_modisco.py --all          # every npz in results/modisco
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent))
import motif_utils as mu  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODISCO_DIR = REPO / "results/modisco"
FLY_DB = REPO / "data/motifs_insects/jaspar_insects_combined.meme"
VERT_DB = REPO / "results/filters/jaspar_vertebrates.meme"


def run_modisco(onehot, hyp, out_h5, max_seqlets=20000, window=15, verbose=True):
    """Run TF-MoDISco-lite; save patterns to out_h5. onehot/hyp: (N, L, 4)."""
    import modiscolite
    pos, neg = modiscolite.tfmodisco.TFMoDISco(
        one_hot=onehot.astype("float32"),
        hypothetical_contribs=hyp.astype("float32"),
        sliding_window_size=window,
        flank_size=5,
        max_seqlets_per_metacluster=max_seqlets,
        target_seqlet_fdr=0.05,
        verbose=verbose,
    )
    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    modiscolite.io.save_hdf5(str(out_h5), pos, neg, window_size=window)
    return out_h5


def _patterns_to_meme(h5_path, meme_path, name_prefix):
    """Extract each MoDISco pattern's PPM (the 'sequence' dataset) -> MEME file."""
    motifs = []
    with h5py.File(h5_path, "r") as f:
        for grp in ("pos_patterns", "neg_patterns"):
            if grp not in f:
                continue
            sign = "p" if grp == "pos_patterns" else "n"
            for pname in f[grp]:
                ppm = np.asarray(f[grp][pname]["sequence"])  # (W, 4), rows sum ~1
                nseqlets = f[grp][pname]["seqlets"]["n_seqlets"][0] if "seqlets" in f[grp][pname] else 0
                motifs.append({"name": f"{name_prefix}_{sign}_{pname}",
                               "ppm": ppm, "nsites": int(nseqlets)})
    mu.write_meme(motifs, meme_path)
    return meme_path, len(motifs)


def process(cond, task, max_seqlets, window):
    npz = MODISCO_DIR / f"{cond}_{task}.npz"
    if not npz.exists():
        print(f"  missing {npz}; run phase3_attributions.py first")
        return None
    d = np.load(npz)
    onehot, hyp = d["onehot"], d["hyp"]
    print(f"\n=== {cond} / {task}: {onehot.shape[0]} seqs (L={onehot.shape[1]}) ===")
    h5 = MODISCO_DIR / f"{cond}_{task}_modisco.h5"
    run_modisco(onehot, hyp, h5, max_seqlets=max_seqlets, window=window)
    meme, n = _patterns_to_meme(h5, MODISCO_DIR / f"{cond}_{task}_motifs.meme", f"{cond}_{task}")
    print(f"  discovered {n} motifs -> {meme.name}")
    res = {"cond": cond, "task": task, "n_motifs": n}
    for db_name, db in [("fly", FLY_DB), ("vertebrate", VERT_DB)]:
        if not Path(db).exists():
            print(f"  (skip {db_name}: {db} missing)")
            continue
        outdir = MODISCO_DIR / f"tomtom_{cond}_{task}_vs_{db_name}"
        _, nq, nm = mu.run_tomtom(meme, db, outdir)
        print(f"  TOMTOM vs {db_name}: {nq}/{n} motifs matched ({nm} total hits)")
        res[f"{db_name}_matched"] = nq
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", default="stage2", help="condition tag (e.g. stage1/stage2)")
    ap.add_argument("--task", default="dev", choices=["dev", "hk"])
    ap.add_argument("--all", action="store_true", help="process every *_*.npz in results/modisco")
    ap.add_argument("--max_seqlets", type=int, default=20000)
    ap.add_argument("--window", type=int, default=15)
    args = ap.parse_args()

    if args.all:
        combos = sorted({tuple(p.stem.split("_")[:2]) for p in MODISCO_DIR.glob("*_*.npz")
                         if not p.stem.endswith("modisco")})
        for cond, task in combos:
            process(cond, task, args.max_seqlets, args.window)
    else:
        process(args.cond, args.task, args.max_seqlets, args.window)


if __name__ == "__main__":
    main()
