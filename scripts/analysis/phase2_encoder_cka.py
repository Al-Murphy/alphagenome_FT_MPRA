#!/usr/bin/env python3
"""Phase 2 — layer-wise CKA depth gradient of the AlphaGenome encoder.

Tests hypothesis prediction P1: early conv layers are CONSERVED under
cross-species fine-tuning (high CKA vs pretrained), and representational drift
GROWS WITH DEPTH. Same-species human-MPRA fine-tune is the control: if
Drosophila (DeepSTARR) fine-tuning drifts the encoder MORE than same-species
human fine-tuning, the extra drift is species-driven new-motif learning.

All encoders are probed on the SAME Drosophila DeepSTARR test sequences:
  - pretrained encoder (base AlphaGenome 'all_folds')   [== probing encoder]
  - fly  fine-tuned encoder (DeepSTARR stage2)
  - human fine-tuned encoder (LentiMPRA HepG2 stage2)   [same-species control]
CKA is measured per encoder tap (7 SequenceEncoder intermediates,
bin_size_{1..64}: stem -> 6 DownResBlocks, 1bp -> 128bp).

RUN ON A GPU NODE. Stage it:
  python scripts/analysis/phase2_encoder_cka.py --mode introspect      # validate plumbing
  python scripts/analysis/phase2_encoder_cka.py --mode cka --n_seqs 1536 --batch_size 64
Outputs (cka): results/cka/phase2_encoder_cka.json + depth-gradient plot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from encoder_common import (  # noqa: E402
    REPO, TAPS, TAP_LABELS, get_device, load_encoders, init_encoder_transform,
    EncoderRunner, encoder_differs, get_test_seqs,
)
from cka_core import CKAAccumulator  # noqa: E402


def _build_runners(enc, init_params, init_state):
    return {name: EncoderRunner(name, p, s, init_params, init_state)
            for name, (p, s) in enc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["introspect", "cka"], default="introspect")
    ap.add_argument("--n_seqs", type=int, default=1536)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--no_human", action="store_true", help="skip the human-MPRA control")
    ap.add_argument("--out", default=str(REPO / "results/cka/phase2_encoder_cka.json"))
    args = ap.parse_args()

    print(f"=== Phase 2 CKA: mode={args.mode} ===")
    device = get_device()
    print(f"JAX device: {device}\n")

    base, enc = load_encoders(device, include_human=not args.no_human)

    # Init standalone transform to learn its param/state key structure.
    probe = get_test_seqs(base, n_seqs=8)
    init_params, init_state = init_encoder_transform(probe)
    print(f"\nStandalone encoder transform: {len(init_params)} param modules, "
          f"{len(init_state)} state modules.")

    runners = _build_runners(enc, init_params, init_state)

    # ---- differ-guard: fine-tuned encoders MUST differ from pretrained ----
    pre_p, _ = enc["pretrained"]
    for name in [n for n in enc if n != "pretrained"]:
        nd, ns, md = encoder_differs(pre_p, enc[name][0])
        status = "OK" if nd > 0 else "!!! TRIVIAL CKA RISK"
        print(f"[differ-guard] {name}: changed={nd} identical={ns} max|Δ|={md:.5g}  {status}")
        assert nd > 0, f"{name} encoder identical to pretrained — CKA would be trivially 1.0"

    if args.mode == "introspect":
        print("\n--- INTROSPECT: per-tap activation shapes on 2 sequences ---")
        taps = runners["pretrained"].taps(jnp.asarray(probe[:2]))
        for t, lbl in zip(TAPS, TAP_LABELS):
            shp = taps[t].shape if t in taps else "MISSING"
            print(f"  {t:14s} {lbl:18s} shape={shp}")
        print("\nIntrospect OK. If shapes look right and differ-guard passed, run --mode cka.")
        return

    # ---- CKA mode ----
    seqs = get_test_seqs(base, n_seqs=args.n_seqs)
    print(f"\nProbe set: {seqs.shape[0]} Drosophila DeepSTARR test seqs (L={seqs.shape[1]}), "
          f"batch_size={args.batch_size}")
    comparisons = [n for n in enc if n != "pretrained"]
    accs = {name: CKAAccumulator(TAPS) for name in comparisons}
    n = seqs.shape[0]
    for b0 in range(0, n, args.batch_size):
        batch = jnp.asarray(seqs[b0:b0 + args.batch_size])
        if batch.shape[0] <= 3:
            continue  # unbiased HSIC needs N>3
        f_pre = runners["pretrained"].taps(batch)
        for name in comparisons:
            accs[name].update(f_pre, runners[name].taps(batch))
        print(f"  batch {b0}/{n}", end="\r")

    cka = {name: accs[name].finalize() for name in comparisons}
    result = {
        "probe": "drosophila_deepstarr_test", "n_seqs": int(n),
        "batch_size": int(args.batch_size), "taps": TAPS, "tap_labels": TAP_LABELS,
        "cka_vs_pretrained": cka,
        "drift_vs_pretrained": {name: {t: 1.0 - cka[name][t] for t in TAPS} for name in comparisons},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n\n=== CKA vs pretrained (1.0 = unchanged; lower = more drift) ===")
    hdr = f"{'tap':18s}" + "".join(f"{n:>12s}" for n in comparisons)
    print(hdr)
    for t, lbl in zip(TAPS, TAP_LABELS):
        print(f"{lbl:18s}" + "".join(f"{cka[n][t]:12.4f}" for n in comparisons))
    if "fly_ft" in cka and "human_ft" in cka:
        print("\nfly drifts MORE than human (supports species-driven new motifs) at:")
        for t, lbl in zip(TAPS, TAP_LABELS):
            if (1 - cka["fly_ft"][t]) > (1 - cka["human_ft"][t]):
                print(f"  {lbl}")
    print(f"\nSaved -> {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(TAPS))
        plt.figure(figsize=(8, 4.5))
        markers = {"fly_ft": "o-", "human_ft": "s--"}
        labels = {"fly_ft": "fly FT (DeepSTARR)", "human_ft": "human FT (LentiMPRA, control)"}
        for name in comparisons:
            plt.plot(x, [cka[name][t] for t in TAPS], markers.get(name, "o-"),
                     label=labels.get(name, name))
        plt.xticks(x, TAP_LABELS, rotation=30, ha="right")
        plt.ylabel("CKA vs pretrained encoder")
        plt.xlabel("encoder depth (shallow -> deep)")
        plt.title("Encoder representational drift under fine-tuning")
        plt.legend(); plt.tight_layout()
        out_png = Path(args.out).with_suffix(".png")
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
