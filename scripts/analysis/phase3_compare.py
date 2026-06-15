#!/usr/bin/env python3
"""Phase 3 comparison — which whole motifs does fine-tuning (stage2) add over probing (stage1)?

Reads the TOMTOM outputs written by phase3_modisco.py and reports, per task
(dev/hk) and per database (fly/insect, vertebrate), the set of TFs matched by
the fine-tuned model's discovered motifs that the probing model's motifs do NOT
match — i.e. the whole motifs that encoder fine-tuning newly enables. The
housekeeping (hk) fly-specific set is the decisive test (DRE/Ohler etc.).

CPU only. Run after phase3_modisco.py --all.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODISCO = REPO / "results/modisco"
DBS = {
    "fly": REPO / "data/motifs_insects/jaspar_insects_combined.meme",
    "vertebrate": REPO / "results/filters/jaspar_vertebrates.meme",
}
TASKS = ["dev", "hk"]
CONDS = ["stage1", "stage2"]


def _id2name(meme_path):
    m = {}
    if Path(meme_path).exists():
        for ln in Path(meme_path).read_text().splitlines():
            if ln.startswith("MOTIF"):
                p = ln.split()
                if len(p) >= 2:
                    m[p[1]] = p[2] if len(p) > 2 else p[1]
    return m


def _matched_targets(tsv):
    """Set of distinct target motif IDs matched (q<thresh already applied by tomtom)."""
    ts = set()
    if not Path(tsv).exists():
        return ts
    for ln in Path(tsv).read_text().splitlines():
        if not ln or ln.startswith("#") or ln.startswith("Query_ID"):
            continue
        p = ln.split("\t")
        if len(p) >= 2 and p[1]:
            ts.add(p[1])
    return ts


def _n_motifs(cond, task):
    meme = MODISCO / f"{cond}_{task}_motifs.meme"
    if not meme.exists():
        return 0
    return sum(1 for ln in meme.read_text().splitlines() if ln.startswith("MOTIF"))


def main():
    summary = {}
    for db_name, db in DBS.items():
        id2name = _id2name(db)
        print(f"\n{'='*72}\nDATABASE: {db_name}\n{'='*72}")
        for task in TASKS:
            t = {c: _matched_targets(MODISCO / f"tomtom_{c}_{task}_vs_{db_name}" / "tomtom.tsv")
                 for c in CONDS}
            n = {c: _n_motifs(c, task) for c in CONDS}
            new_in_s2 = sorted(t["stage2"] - t["stage1"])
            lost = sorted(t["stage1"] - t["stage2"])
            names_new = sorted({id2name.get(x, x) for x in new_in_s2})
            print(f"\n--- task={task} ---")
            print(f"  motifs discovered: stage1={n['stage1']}  stage2={n['stage2']}")
            print(f"  distinct {db_name} TFs matched: stage1={len(t['stage1'])}  stage2={len(t['stage2'])}")
            print(f"  TFs NEW in stage2 (fine-tuning adds): {len(names_new)}")
            if names_new:
                print("    " + ", ".join(names_new))
            summary[f"{db_name}_{task}"] = {
                "n_motifs_stage1": n["stage1"], "n_motifs_stage2": n["stage2"],
                "tfs_stage1": len(t["stage1"]), "tfs_stage2": len(t["stage2"]),
                "new_in_stage2": names_new, "lost_in_stage2": sorted({id2name.get(x, x) for x in lost}),
            }
    out = MODISCO / "phase3_comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
