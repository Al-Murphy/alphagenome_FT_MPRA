"""Shared machinery for AlphaGenome encoder analyses (Phase 1 filters, Phase 2 CKA).

Single source of truth for the parts that touch the live JAX/Haiku model:
model loading, the standalone SequenceEncoder transform that exposes all
intermediate taps, param-key reconciliation, the differ-guard, and the shared
Drosophila DeepSTARR probe set. Import these instead of duplicating.

We deliberately do NOT use alphagenome_ft.load_checkpoint: it rebuilds and
shape-validates the task head, whose flatten dim depends on the model's
training sequence length (DeepSTARR ~300bp -> 3 encoder positions -> 4608;
LentiMPRA differs). We only need ENCODER weights, which are sequence-length
independent. So we restore the saved pytree with strict=False /
restore_type=np.ndarray (validated on CPU): encoder leaves come back exactly,
the head comes back at its stored shape and is discarded. A template model
(built with the matching head NAME) supplies the restore-target structure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import orbax.checkpoint as ocp

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model import model as model_lib
from alphagenome_ft import (
    create_model_with_heads,
    register_custom_head,
    HeadConfig,
    HeadType,
)
from alphagenome_ft_mpra import DeepSTARRHead, DeepSTARRDataset, EncoderMPRAHead

REPO = Path(__file__).resolve().parents[2]
BASE_CKPT = Path.home() / ".cache/kagglehub/models/google/alphagenome/jax/all_folds/1"

# 8 encoder taps, shallow -> deep: stem (1bp) + 6 DownResBlocks (2-64bp) + final OUTPUT.
# bin_size_N = the tap whose positions are N bp wide. bin_size_128 is the post-final-pool
# encoder OUTPUT (the `out` returned by SequenceEncoder = ExtendedEmbeddings.encoder_output,
# the tensor the task head reads); it is not in the intermediates dict, so taps() injects it.
TAPS = ["bin_size_1", "bin_size_2", "bin_size_4", "bin_size_8",
        "bin_size_16", "bin_size_32", "bin_size_64", "bin_size_128"]
TAP_LABELS = ["Stem (1 bp)", "Block 1 (2 bp)", "Block 2 (4 bp)", "Block 3 (8 bp)",
              "Block 4 (16 bp)", "Block 5 (32 bp)", "Block 6 (64 bp)", "Output (128 bp)"]

# Default fine-tuned checkpoints (stage2 = encoder unfrozen) + their head names.
FLY_CKPT = REPO / "results/models/checkpoints/deepstarr/deepstarr-optimal/stage2"
HUMAN_CKPT = REPO / "results/models/checkpoints/HepG2/mpra-HepG2-optimal/stage2"
FLY_HEAD = "deepstarr_head"
HUMAN_HEAD = "mpra_head"


@hk.transform_with_state
def encoder_fn(dna_sequence):
    """Standalone encoder forward exposing ALL intermediate taps.

    Replicates the exact name_scope ('alphagenome') used by alphagenome_ft's
    encoder-only forward so Haiku param keys line up with the loaded model.
    """
    with hk.name_scope("alphagenome"):
        out, intermediates = model_lib.SequenceEncoder()(dna_sequence)
    return out, intermediates


def get_device():
    try:
        d = jax.devices("gpu")[0]
    except (IndexError, RuntimeError):
        d = jax.devices("cpu")[0]
        print("WARNING: no GPU visible; running on CPU will be slow.")
    return d


def register_heads():
    """Register both heads so templates / structure match the saved checkpoints."""
    register_custom_head(
        "deepstarr_head", DeepSTARRHead,
        HeadConfig(type=HeadType.GENOME_TRACKS, name="deepstarr_head",
                   output_type=dna_output.OutputType.RNA_SEQ, num_tracks=2,
                   metadata={"center_bp": 256, "pooling_type": "flatten",
                             "nl_size": 2048, "do": 0.5, "activation": "relu"}),
    )
    register_custom_head(
        "mpra_head", EncoderMPRAHead,
        HeadConfig(type=HeadType.GENOME_TRACKS, name="mpra_head",
                   output_type=dna_output.OutputType.RNA_SEQ, num_tracks=1,
                   metadata={"center_bp": 256, "pooling_type": "flatten",
                             "nl_size": 1024, "do": 0.1, "activation": "relu"}),
    )


def params_state(model):
    """Pull (_params, _state) off a loaded model, tolerant to attr naming."""
    params = getattr(model, "_params", None)
    state = getattr(model, "_state", None)
    if params is None:
        raise AttributeError(
            f"{type(model).__name__} has no _params; "
            f"attrs: {[a for a in dir(model) if not a.startswith('__')]}")
    return params, (state if state is not None else {})


def _filter_encoder(tree):
    """Keep only Haiku module entries belonging to the sequence encoder."""
    if not isinstance(tree, dict):
        return {}
    return {k: v for k, v in tree.items() if "sequence_encoder" in str(k)}


# All-None settings: skip remote fasta/gtf/calibration loads (we never need them) — in
# particular the eager gs:// calibration_scores.pb fetch that fails on compute nodes without
# GCS/SSL access. Keep BOTH organisms so the organism-embedding shape matches the pretrained
# (human+mouse) weights.
_ORG_SETTINGS = {
    dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(),
    dna_model.Organism.MUS_MUSCULUS: dna_model.OrganismSettings(),
}


def _make_template(head_name, device):
    """Full model with the given head NAME — used only for restore-target structure."""
    return create_model_with_heads(
        "all_folds", heads=[head_name], use_encoder_output=True,
        checkpoint_path=str(BASE_CKPT), init_seq_len=256, device=device,
        organism_settings=_ORG_SETTINGS,
    )


def load_encoder_params_state(checkpoint_dir, template_model):
    """Restore ONLY encoder (params, state) from a stage2 checkpoint as numpy.

    Uses the template's minimal slice-tree structure as the restore target, with
    strict=False so the head's shape mismatch is tolerated (head discarded).
    """
    target = template_model._checkpoint_slice_trees(False, True)  # (params, state)
    path = (Path(checkpoint_dir) / "checkpoint").resolve()
    restore_args = jax.tree_util.tree_map(
        lambda _x: ocp.ArrayRestoreArgs(restore_type=np.ndarray, strict=False), target)
    out = ocp.PyTreeCheckpointer().restore(
        str(path), ocp.args.PyTreeRestore(item=target, restore_args=restore_args))
    if isinstance(out, (tuple, list)) and len(out) == 2:
        params, state = out
    else:
        params, state = out, {}
    return _filter_encoder(params), _filter_encoder(state or {})


def _encoder_subtree(transform_params, enc_params):
    """Pull the keys this transform needs from an encoder param/state dict."""
    want = set(transform_params.keys())
    have = set(enc_params.keys())
    missing = sorted(want - have)
    if missing:
        print("\n!!! Param-key mismatch between standalone encoder and checkpoint.")
        print("    transform wants (sample):", sorted(want)[:6])
        print("    checkpoint has  (sample):", sorted(have)[:12])
        print("    missing                 :", missing[:12])
        raise KeyError(f"{len(missing)} encoder keys not found in checkpoint tree")
    return {k: enc_params[k] for k in transform_params}


def _encoder_state_subtree(transform_state, enc_state):
    if not transform_state:
        return {}
    have = set(enc_state.keys()) if enc_state else set()
    sub = {k: enc_state[k] for k in transform_state if k in have}
    for k in set(transform_state) - have:  # fall back to init for unsaved state
        sub[k] = transform_state[k]
    return sub


class EncoderRunner:
    """Applies encoder_fn with one model's encoder weights, returns taps."""

    def __init__(self, name, enc_params, enc_state, init_params, init_state):
        self.name = name
        self.params = _encoder_subtree(init_params, enc_params)
        self.state = _encoder_state_subtree(init_state, enc_state)
        self._rng = jax.random.PRNGKey(0)

    def taps(self, seq_batch, which=None):
        """seq_batch: (B, L, 4) -> {tap: np.ndarray (B, ...)} for `which` taps (default all)."""
        which = which or TAPS
        (out, inter), _ = encoder_fn.apply(self.params, self.state, self._rng, seq_batch)
        feats = dict(inter)
        feats["bin_size_128"] = out   # final post-pool encoder output (128 bp), what the head reads
        return {t: np.asarray(feats[t], dtype=np.float32) for t in which if t in feats}


def init_encoder_transform(probe_seq):
    """Init the standalone transform to discover its param/state key structure."""
    dummy = jnp.asarray(probe_seq[:2])
    return encoder_fn.init(jax.random.PRNGKey(0), dummy)


def load_encoders(device, include_human=True):
    """Returns (base_model, {name: (enc_params, enc_state)}).

    base_model is the pretrained model (also used to build the DeepSTARR probe
    set). 'pretrained' encoder == base's own encoder weights (== probing encoder).
    """
    register_heads()
    print("Loading pretrained base AlphaGenome (all_folds)...")
    base = _make_template(FLY_HEAD, device)  # deepstarr_head template == fly structure
    bp, bs = params_state(base)
    enc = {"pretrained": (_filter_encoder(bp), _filter_encoder(bs))}

    print("Restoring fly fine-tuned encoder (DeepSTARR stage2)...")
    enc["fly_ft"] = load_encoder_params_state(FLY_CKPT, base)

    if include_human:
        print("Restoring human fine-tuned encoder (LentiMPRA HepG2 stage2, control)...")
        human_tpl = _make_template(HUMAN_HEAD, device)
        enc["human_ft"] = load_encoder_params_state(HUMAN_CKPT, human_tpl)
    return base, enc


def encoder_differs(enc_a, enc_b):
    """Differ-guard over encoder param dicts: (#changed, #identical, max|Δ|)."""
    n_diff = n_same = 0
    max_d = 0.0
    for k in enc_a:
        if k not in enc_b:
            continue
        a_mod, b_mod = enc_a[k], enc_b[k]
        if not isinstance(a_mod, dict):
            continue
        for pk in a_mod:
            if pk not in b_mod:
                continue
            a = np.asarray(a_mod[pk], np.float32)
            b = np.asarray(b_mod[pk], np.float32)
            if a.shape != b.shape:
                continue
            d = float(np.abs(a - b).max())
            max_d = max(max_d, d)
            n_diff += int(d > 1e-7)
            n_same += int(d <= 1e-7)
    return n_diff, n_same, max_d


def get_test_seqs(model, n_seqs, seed=0):
    """One-hot Drosophila DeepSTARR test sequences — the shared probe set."""
    ds = DeepSTARRDataset(
        model=model, path_to_data=str(REPO / "data/deepstarr"), split="test",
        organism=dna_model.Organism.HOMO_SAPIENS, random_shift=False, reverse_complement=False,
    )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_seqs, len(ds)), replace=False)
    seqs = np.stack([np.asarray(ds[int(i)]["seq"], np.float32) for i in idx])
    return seqs  # (n_seqs, L, 4)
