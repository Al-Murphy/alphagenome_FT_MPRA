#!/usr/bin/env python3
"""Phase 3 (GPU stage) — attribution maps for TF-MoDISco motif discovery.

For each condition (probing stage1, fine-tuned stage2) and task (dev, hk),
loads the FULL DeepSTARR model (encoder + head; init_seq_len=300 so the flatten
head matches the stored 4608-dim weights) and computes per-base RAW gradients of
the task output w.r.t. the input one-hot — i.e. the *hypothetical contributions*
TF-MoDISco expects (one_hot * hyp = actual contribs). Saves
results/modisco/{cond}_{task}.npz (keys: onehot, hyp) for phase3_modisco.py.

Sequences = the top-`n_seqs` DeepSTARR test enhancers by that task's activity
(strongest motif signal). RUN ON A GPU NODE.

Usage:
  python scripts/analysis/phase3_attributions.py --n_seqs 3000 --batch_size 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from encoder_common import (  # noqa: E402  (reuse device + head registration + probe loader)
    REPO, BASE_CKPT, get_device, register_heads,
)
from alphagenome_ft import load_checkpoint  # noqa: E402
from alphagenome_research.model import dna_model  # noqa: E402
from alphagenome_ft_mpra import DeepSTARRDataset  # noqa: E402

MODISCO_DIR = REPO / "results/modisco"
CKPTS = {
    "stage1": REPO / "results/models/checkpoints/deepstarr/deepstarr-optimal/stage1",
    "stage2": REPO / "results/models/checkpoints/deepstarr/deepstarr-optimal/stage2",
}
TASK_IDX = {"dev": 0, "hk": 1}
HEAD = "deepstarr_head"


# All-None settings: skip remote fasta/gtf/calibration loads (attribution doesn't
# need them) — in particular the eager gs:// calibration_scores.pb fetch that fails
# on compute nodes without GCS/SSL. Keep BOTH organisms so the organism-embedding
# shape still matches the pretrained (human+mouse) weights.
_ORG_SETTINGS = {
    dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(),
    dna_model.Organism.MUS_MUSCULUS: dna_model.OrganismSettings(),
}


def _load_full_model(ckpt, device):
    # init_seq_len=300 -> 3 encoder positions -> flatten head 4608 (matches stored)
    return load_checkpoint(str(ckpt), base_model_version="all_folds",
                           base_checkpoint_path=str(BASE_CKPT), device=device,
                           init_seq_len=300, organism_settings=_ORG_SETTINGS)


def _head_pred(model, seq, org, head, ti):
    """Predictions for output track `ti` via the model's wrapped _predict.

    Works around the alphagenome_ft bug where `_custom_forward_fn` is left unset
    when a custom forward exists (so compute_input_gradients/deepshap mis-route).
    For a custom-head-only model, requested_outputs and the strand args are unused
    (the standard-prediction branch of wrapped_predict is skipped), so dummy values
    are safe. Returns (B, P) — the per-position predictions for track ti.
    """
    B = seq.shape[0]
    preds = model._predict(
        model._params, model._state, seq, org,
        requested_outputs=None,
        negative_strand_mask=jnp.zeros((B,), dtype=jnp.bool_),
        strand_reindexing=jnp.array([], dtype=jnp.int32),
        rng=None,
    )
    if hasattr(preds, "_custom"):
        out = preds._custom.get(head)
    elif hasattr(preds, "get"):
        out = preds.get(head)
    else:
        out = preds[head]
    return jnp.asarray(out)[..., ti]          # (B, P)


def _head_scalar(model, seq, org, head, ti):
    """Per-sequence scalar prediction for track ti (B,)."""
    out = _head_pred(model, jnp.asarray(seq), jnp.asarray(org), head, ti)
    if out.ndim == 2:
        out = jnp.mean(out, axis=1)           # pool positions (flatten head: P=1)
    return np.asarray(out, np.float32)


def _make_grad_fn(model, head, ti):
    """JIT-compiled gradient of the summed track-ti output w.r.t. the input (B,L,4)."""
    def _loss(seq, org):
        return jnp.sum(_head_pred(model, seq, org, head, ti))
    return jax.jit(jax.grad(_loss))


def _ism_hypothetical(model, seq1, org1, head, ti, chunk=512):
    """Faithful in-silico-mutagenesis hypothetical contributions (L,4): the change
    in predicted activity when each position is set to each base. No Koo correction
    needed (mutants stay on the simplex). 4 batched forwards of L mutants per seq.

    NOTE: implemented but not yet validated on GPU — verify before relying on it.
    """
    L = seq1.shape[1]
    base = np.asarray(seq1[0], np.float32)            # (L,4)
    wt = _head_scalar(model, seq1, org1.reshape(1), head, ti)[0]
    hyp = np.zeros((L, 4), np.float32)
    di = np.arange(L)
    for a in range(4):
        M = np.broadcast_to(base, (L, L, 4)).copy()   # mutant l sets position l -> base a
        M[di, di, :] = 0.0
        M[di, di, a] = 1.0
        preds = np.empty(L, np.float32)
        for c0 in range(0, L, chunk):
            mb = M[c0:c0 + chunk]
            ob = np.full((mb.shape[0],), int(np.asarray(org1).reshape(-1)[0]))
            preds[c0:c0 + mb.shape[0]] = _head_scalar(model, mb, ob, head, ti)
        hyp[:, a] = preds - wt
    return hyp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_seqs", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--conds", nargs="+", default=["stage1", "stage2"])
    ap.add_argument("--tasks", nargs="+", default=["dev", "hk"])
    ap.add_argument("--method", default="gradient_corrected",
                    choices=["gradient_corrected", "gradient_raw", "ism"],
                    help="MoDISco hypothetical contribs. 'gradient_corrected' (default, recommended): "
                         "raw gradient with the per-position across-base mean-subtraction of "
                         "Majdandzic, Rajesh & Koo 2023 (Genome Biol) that removes the off-simplex "
                         "one-hot artifact. 'gradient_raw': uncorrected. 'ism': faithful in-silico "
                         "mutagenesis hypothetical (most accurate, no correction needed, but ~4·L× "
                         "cost — use small --n_seqs). NOTE: the repo's compute_deepshap_attributions "
                         "is NOT true DeepSHAP (orig_grad - mean(ref_grad), no DeepLIFT rescale) — "
                         "deliberately not used.")
    args = ap.parse_args()

    MODISCO_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"JAX device: {device}\n")
    register_heads()

    for cond in args.conds:
        print(f"\n===== loading {cond} ({CKPTS[cond].name}) =====")
        model = _load_full_model(CKPTS[cond], device)
        # dataset (needs a model for one-hot encoding); load once per condition
        ds = DeepSTARRDataset(model=model, path_to_data=str(REPO / "data/deepstarr"),
                              split="test", organism=dna_model.Organism.HOMO_SAPIENS,
                              random_shift=False, reverse_complement=False)
        ys = np.stack([np.asarray(ds[i]["y"]) for i in range(len(ds))])  # (N,2)

        for task in args.tasks:
            ti = TASK_IDX[task]
            order = np.argsort(ys[:, ti])[::-1][:args.n_seqs]   # top by this task's activity
            seqs = np.stack([np.asarray(ds[int(i)]["seq"], np.float32) for i in order])
            orgs = np.stack([np.asarray(ds[int(i)]["organism_index"]) for i in order]).reshape(-1)
            print(f"  {cond}/{task}: {seqs.shape[0]} top enhancers (L={seqs.shape[1]}), "
                  f"method={args.method}")

            hyp = np.empty_like(seqs)
            if args.method == "ism":
                for i in range(seqs.shape[0]):
                    hyp[i] = _ism_hypothetical(model, seqs[i:i + 1], orgs[i:i + 1], HEAD, ti)
                    print(f"    ism seq {i}/{seqs.shape[0]}", end="\r")
            else:
                grad_fn = _make_grad_fn(model, HEAD, ti)   # raw gradient = MoDISco hypothetical
                for b0 in range(0, seqs.shape[0], args.batch_size):
                    sb = jnp.asarray(seqs[b0:b0 + args.batch_size])
                    ob = jnp.asarray(orgs[b0:b0 + args.batch_size])
                    g = np.asarray(grad_fn(sb, ob), np.float32)
                    if args.method == "gradient_corrected":
                        # Koo 2023 correction: subtract per-position mean across the 4 bases
                        # (project onto simplex tangent; removes off-simplex one-hot artifact).
                        g = g - g.mean(axis=-1, keepdims=True)
                    hyp[b0:b0 + sb.shape[0]] = g
                    print(f"    batch {b0}/{seqs.shape[0]}", end="\r")

            out = MODISCO_DIR / f"{cond}_{task}.npz"
            np.savez_compressed(out, onehot=seqs, hyp=hyp)
            print(f"\n  saved {out}")

    print("\nDone. Next (CPU): python scripts/analysis/phase3_modisco.py --all")


if __name__ == "__main__":
    main()
