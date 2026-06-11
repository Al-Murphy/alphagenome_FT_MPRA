#!/usr/bin/env python3
"""Phase 1 — first-layer conv filter interpretation (encoder motif subparts).

Tests hypothesis predictions P1/P3 at the level of individual conv filters:
do the encoder's early filters (motif "subparts") stay CONSERVED under
cross-species fine-tuning, and what known TFs do they match?

Two stages:
  extract  (GPU): run the shared Drosophila DeepSTARR probe seqs through each
                  encoder, capture first-layer activations (bin_size_1: 768
                  filters, ~19bp RF), build per-filter PFMs via max-activating
                  subsequences, write MEME per model, and Hungarian-match each
                  fine-tuned model's filters to the pretrained ones (activation
                  correlation) -> a per-filter conservation score.
  tomtom   (CPU, login node OK): TOMTOM each model's filter MEME against a
                  vertebrate JASPAR DB (built from data/motifs) and, if provided,
                  a Drosophila/insect DB (--fly_db) to find genuinely fly motifs.

Usage:
  python scripts/analysis/phase1_filter_motifs.py --mode extract --n_seqs 512
  python scripts/analysis/phase1_filter_motifs.py --mode tomtom \
      --fly_db /path/to/fly_motifs.meme
Outputs under results/filters/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from encoder_common import (  # noqa: E402
    REPO, get_device, load_encoders, init_encoder_transform, EncoderRunner,
    encoder_differs, get_test_seqs,
)
import motif_utils as mu  # noqa: E402

FIRST_TAP = "bin_size_1"
OUT = REPO / "results/filters"
MEME_BIN = Path.home() / "local/meme/bin/tomtom"


# --------------------------------------------------------------------------- #
def _collect_activations(runner, seqs, batch_size):
    """Run seqs through one encoder, return first-layer activations (N, L, F)."""
    chunks = []
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = jnp.asarray(seqs[b0:b0 + batch_size])
        chunks.append(runner.taps(batch, which=[FIRST_TAP])[FIRST_TAP])
    return np.concatenate(chunks, 0)


def _filters_to_meme(seqs, acts, model_name, window, threshold_frac, min_sites, max_sites):
    """Build per-filter PFMs -> MEME + per-filter stats (IC, n_sites)."""
    F = acts.shape[2]
    motifs, stats = [], []
    for f in range(F):
        res = mu.filter_pfm(seqs, acts, f, window=window, threshold_frac=threshold_frac,
                            min_sites=min_sites, max_sites=max_sites)
        if res is None:
            stats.append({"filter": f, "n_sites": 0, "ic": 0.0})
            continue
        ppm = mu.pfm_to_ppm(res["pfm"], pseudocount=1.0)
        ic = mu.information_content(ppm)
        tppm, (s, e) = mu.trim_to_ic(ppm, min_ic=0.2)
        if tppm.shape[0] >= 3:
            motifs.append({"name": f"{model_name}_f{f}", "ppm": tppm, "nsites": res["n_sites"]})
        stats.append({"filter": f, "n_sites": res["n_sites"], "ic": ic,
                      "trim": [s, e], "max_act": res["max_act"]})
    meme_path = mu.write_meme(motifs, OUT / f"{model_name}_filters.meme")
    return meme_path, stats, len(motifs)


def extract(args):
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"JAX device: {device}\n")
    base, enc = load_encoders(device, include_human=not args.no_human)

    probe = get_test_seqs(base, n_seqs=8)
    init_params, init_state = init_encoder_transform(probe)
    runners = {name: EncoderRunner(name, p, s, init_params, init_state)
               for name, (p, s) in enc.items()}

    # differ-guard (same as Phase 2)
    pre_p = enc["pretrained"][0]
    for name in [n for n in enc if n != "pretrained"]:
        nd, _, md = encoder_differs(pre_p, enc[name][0])
        assert nd > 0, f"{name} encoder identical to pretrained"
        print(f"[differ-guard] {name}: changed={nd} max|Δ|={md:.4g}")

    seqs = get_test_seqs(base, n_seqs=args.n_seqs)
    print(f"\nProbe: {seqs.shape[0]} DeepSTARR seqs (L={seqs.shape[1]}); "
          f"first tap {FIRST_TAP}, window={args.window}")

    acts = {}
    summary = {}
    for name, runner in runners.items():
        print(f"  activations: {name} ...")
        a = _collect_activations(runner, seqs, args.batch_size)
        acts[name] = a
        meme_path, stats, n_motifs = _filters_to_meme(
            seqs, a, name, args.window, args.threshold_frac, args.min_sites, args.max_sites)
        n_active = sum(s["n_sites"] > 0 for s in stats)
        print(f"    -> {n_motifs} informative filters ({n_active}/{a.shape[2]} active) -> {meme_path.name}")
        np.savez_compressed(OUT / f"{name}_filter_stats.npz",
                            filters=np.array([s["filter"] for s in stats]),
                            n_sites=np.array([s["n_sites"] for s in stats]),
                            ic=np.array([s["ic"] for s in stats]))
        summary[name] = {"n_motifs": n_motifs, "n_active": int(n_active),
                         "meme": str(meme_path)}

    # ---- Hungarian filter conservation: pretrained vs each fine-tune ----
    cons = {}
    for name in [n for n in enc if n != "pretrained"]:
        m = mu.hungarian_filter_match(acts["pretrained"], acts[name], reduce_mode="max")
        np.savez_compressed(OUT / f"conservation_pretrained_vs_{name}.npz",
                            assignment=m["assignment"], matched_corr=m["matched_corr"],
                            diag_corr=m["diag_corr"])
        cons[name] = {"mean_matched_corr": m["mean_matched_corr"],
                      "mean_diag_corr": m["mean_diag_corr"],
                      "frac_highly_conserved": float((m["matched_corr"] > 0.9).mean())}
        print(f"\n[filter conservation] pretrained vs {name}: "
              f"mean best-match corr={m['mean_matched_corr']:.3f}, "
              f"same-index corr={m['mean_diag_corr']:.3f}, "
              f"frac>0.9={cons[name]['frac_highly_conserved']:.3f}")

    summary["conservation"] = cons
    with open(OUT / "phase1_extract_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {OUT/'phase1_extract_summary.json'}")
    if "fly_ft" in cons and "human_ft" in cons:
        fly_c = cons["fly_ft"]["mean_matched_corr"]
        hum_c = cons["human_ft"]["mean_matched_corr"]
        print(f"\nFilter conservation: fly={fly_c:.3f} vs human={hum_c:.3f} — "
              f"{'fly LESS conserved (species-driven)' if fly_c < hum_c else 'similar/fly more conserved'}")


def _run_tomtom(query_meme, db_meme, outdir, thresh=0.1):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(MEME_BIN), "-no-ssc", "-oc", str(outdir), "-thresh", str(thresh),
           "-min-overlap", "3", str(query_meme), str(db_meme)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    tsv = outdir / "tomtom.tsv"
    if tsv.exists():
        rows = [ln for ln in tsv.read_text().splitlines() if ln and not ln.startswith("#")]
        n_hits = max(0, len(rows) - 1)
        queries = {r.split("\t")[0] for r in rows[1:] if "\t" in r}
        print(f"    {n_hits} matches across {len(queries)} filters")
    return outdir


def tomtom(args):
    OUT.mkdir(parents=True, exist_ok=True)
    if not MEME_BIN.exists():
        sys.exit(f"tomtom not found at {MEME_BIN}")
    # vertebrate DB from the repo's per-motif JASPAR files
    vert_db = OUT / "jaspar_vertebrates.meme"
    if not vert_db.exists():
        _, n = mu.build_combined_meme(REPO / "data/motifs", vert_db)
        print(f"Built vertebrate DB: {n} motifs -> {vert_db}")
    dbs = {"vertebrate": vert_db}
    if args.fly_db:
        fly = Path(args.fly_db)
        if fly.exists():
            dbs["fly"] = fly
        else:
            print(f"WARNING: --fly_db {fly} not found; skipping fly TOMTOM.")
    else:
        print("NOTE: no --fly_db given. Download JASPAR insects or FlyFactorSurvey "
              "(MEME format) to test for genuinely Drosophila motifs.")

    models = ["pretrained", "fly_ft", "human_ft"]
    for model in models:
        qm = OUT / f"{model}_filters.meme"
        if not qm.exists():
            print(f"skip {model}: {qm} missing (run --mode extract first)")
            continue
        for db_name, db in dbs.items():
            print(f"TOMTOM {model} vs {db_name}:")
            _run_tomtom(qm, db, OUT / f"tomtom_{model}_vs_{db_name}", thresh=args.thresh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["extract", "tomtom"], required=True)
    ap.add_argument("--n_seqs", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window", type=int, default=19, help="subsequence width (~ first-layer RF)")
    ap.add_argument("--threshold_frac", type=float, default=0.5)
    ap.add_argument("--min_sites", type=int, default=10)
    ap.add_argument("--max_sites", type=int, default=3000)
    ap.add_argument("--no_human", action="store_true")
    ap.add_argument("--fly_db", default=None, help="MEME file of Drosophila/insect motifs (tomtom mode)")
    ap.add_argument("--thresh", type=float, default=0.1, help="tomtom q-value threshold")
    args = ap.parse_args()
    print(f"=== Phase 1 filters: mode={args.mode} ===")
    (extract if args.mode == "extract" else tomtom)(args)


if __name__ == "__main__":
    main()
