"""Test a fine-tuned / probed AlphaGenome model on plant STARR-seq (Jores 2021).

Eval-only counterpart to ``scripts/finetune_plant_starrseq.py``. Loads a saved
checkpoint and scores it on the held-out **test** split; it never trains. This is
the independent check on the numbers in ``results/plant_starrseq/reference/``.

IMPORTANT — the checkpoints were NOT produced by this repo. They come from the
upstream ``autotune`` codebase (``/grid/koo/home/duran/autotune``), which uses a
different head implementation, so this script reproduces *autotune's* head rather
than ``EncoderMPRAHead``:

  * autotune's head leaves its Haiku modules unnamed, so the saved params are keyed
    ``head/mpra_head/~predict/{layer_norm,linear,linear_1}``. ``EncoderMPRAHead``
    names them ``norm``/``hidden_0``, so its params would not load by name.
  * autotune used ``HIDDEN_SIZE=4096``; the repo configs say ``nl_size=1024``.
    Hidden size is therefore inferred from the checkpoint, not from the config.

The checkpoint is a plain Python **pickle** (despite the ``.pt`` name — it is not a
torch archive) holding ``{params, val_pearson, test_pearson, config}``, where
``params`` is the Haiku param tree as numpy arrays.

USAGE:
    # score a fine-tuned (stage-2) checkpoint
    python scripts/test_ft_model_plant_starrseq.py --tissue leaf --mode combined \\
        --checkpoint_dir /grid/koo/home/shared/models/alphagenome_encoder/jax/plant_starrseq_leaf_combined

    # score a linear probe (ridge head .npz on frozen encoder features)
    python scripts/test_ft_model_plant_starrseq.py --tissue leaf --mode combined --probe \\
        --checkpoint_dir /grid/koo/home/shared/models/alphagenome_encoder/jax/plant_starrseq_leaf_combined
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphagenome.models import dna_output
from alphagenome_ft import (
    CustomHead,
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
    load_checkpoint,
)
from alphagenome_ft_mpra import PlantStarrSeqDataset, MPRADataLoader
from alphagenome_ft_mpra.plant_starrseq_utils import (
    PROMOTER_LENGTH,
    SEQUENCE_LENGTH,
    ridge_predict,
    write_run_metrics,
)

ENCODER_DIM = 1536  # AlphaGenome encoder output channels
PROBE_HEAD = "plant_probe_pool"
FINETUNE_HEAD = "mpra_head"

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "results" / "plant_starrseq" / "reference"

DEFAULT_HIDDEN_SIZE = 4096  # autotune's HIDDEN_SIZE for the plant runs


class PlantMPRAHead(CustomHead):
    """LayerNorm -> flatten -> Linear(H) -> ReLU -> [dropout] -> Linear(1) -> scalar.

    The architecture of autotune's ``HepG2MPRAHead``
    (``lib/backbones/alphagenome_jax.py``). The Haiku modules are deliberately left
    unnamed so the auto-generated names (``layer_norm``, ``linear``, ``linear_1``)
    match the keys in the saved param tree — ``EncoderMPRAHead`` names them
    ``norm``/``hidden_0`` and therefore cannot load these weights.

    Width comes from ``metadata['nl_size']`` so the head is self-describing: a
    checkpoint's ``config.json`` carries it, and ``load_checkpoint`` can rebuild the
    head without out-of-band knowledge. (Upstream read it from a module global.)

    Dropout is skipped whenever there is no RNG stack, which is the case for
    ``model._predict`` (rng=None) — i.e. always off here, as at eval upstream.
    """

    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(name=name, num_tracks=num_tracks, output_type=output_type,
                         num_organisms=num_organisms, metadata=metadata)
        self._hidden_size = (metadata or {}).get("nl_size", DEFAULT_HIDDEN_SIZE)

    def predict(self, embeddings, organism_index, **kwargs):
        x = embeddings.encoder_output  # (B, T, ENCODER_DIM)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = x.reshape(x.shape[0], -1)  # (B, T * ENCODER_DIM)
        x = jax.nn.relu(hk.Linear(self._hidden_size)(x))
        x = hk.Linear(1)(x)
        return x.squeeze(-1)  # (B,)

    def loss(self, predictions, batch):
        targets = batch["targets"]
        mse = jnp.mean((predictions - targets) ** 2)
        return {"loss": mse, "mse": mse}


class ProbePoolHead(CustomHead):
    """Mean-pooled frozen encoder output (B, ENCODER_DIM) — the probe's features."""

    def predict(self, embeddings, organism_index, **kwargs):
        return embeddings.encoder_output.mean(axis=1)

    def loss(self, predictions, batch):
        return {"loss": jnp.mean(predictions ** 2)}


def load_ag_checkpoint(path):
    """The '.pt' files are pickles of a Haiku param tree, not torch archives."""
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_hidden_size(params):
    """Hidden width from the saved head, since the repo configs disagree with it."""
    w = params[f"head/{FINETUNE_HEAD}/~predict/linear"]["w"]
    return int(w.shape[1])


def build_model(head_name, head_class, seq_len, base_checkpoint_path, num_tracks=1,
                metadata=None):
    register_custom_head(
        head_name, head_class,
        HeadConfig(
            type=HeadType.GENOME_TRACKS, name=head_name,
            output_type=dna_output.OutputType.RNA_SEQ, num_tracks=num_tracks,
            metadata=metadata or {},
        ),
    )
    return create_model_with_custom_heads(
        "all_folds", custom_heads=[head_name],
        checkpoint_path=base_checkpoint_path,
        use_encoder_output=True, init_seq_len=seq_len,
    )


def make_test_loader(model, args):
    """Held-out test split, augmentation off (as upstream at eval time)."""
    test_ds = PlantStarrSeqDataset(
        model=model, path_to_data=args.data_path, tissue=args.tissue, mode=args.mode,
        split="test", random_shift=False, reverse_complement=False, seed=args.seed,
    )
    return MPRADataLoader(test_ds, batch_size=args.batch_size, shuffle=False)


def predict_head(model, params, loader, head_name):
    """Replicates autotune's final test loop (finetune.py, 'scipy pearsonr on test set')."""
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model

    organism_enum = ag_dna_model.Organism.HOMO_SAPIENS
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = jax.device_put(
        model._metadata[organism_enum].strand_reindexing,
        model._device_context._device,
    )
    state = model._state

    preds, tgts = [], []
    for batch in loader:
        seqs = batch["seq"]
        B, L = seqs.shape[0], seqs.shape[1]
        out = model._predict(
            params, state, seqs,
            jnp.full((B,), organism_index_value, dtype=jnp.int32),
            requested_outputs=None,
            negative_strand_mask=jnp.zeros((B, L), dtype=jnp.bool_),
            strand_reindexing=strand_reindexing,
        )
        preds.append(np.asarray(out[head_name], dtype=np.float32))
        tgts.append(np.asarray(batch["y"], dtype=np.float32))
    return np.concatenate(preds), np.concatenate(tgts)


def test_finetune(args, seq_len):
    """Score the saved best-val stage-2 checkpoint on the test split.

    Handles both layouts: the orbax checkpoint written by ``save_checkpoint`` (a
    ``checkpoint/`` dir beside a ``config.json``), and autotune's original flat pickle.
    """
    ckpt_dir = Path(args.checkpoint_dir)
    recorded = None

    if (ckpt_dir / "checkpoint").exists():
        # Orbax layout. load_checkpoint rebuilds the head from the *registered* HeadConfig,
        # NOT from the checkpoint's config.json — so the head metadata must be registered
        # from that file first. Head width varies per cell (4096 / 2048 / 1024); registering
        # without it silently falls back to DEFAULT_HIDDEN_SIZE and the restore then dies on
        # a shape mismatch.
        print(f"Loading orbax checkpoint: {ckpt_dir}")
        cfg = json.loads((ckpt_dir / "config.json").read_text())
        head_meta = cfg["head_configs"][FINETUNE_HEAD].get("metadata") or {}
        print(f"Head metadata from config.json: {head_meta}")
        register_custom_head(
            FINETUNE_HEAD, PlantMPRAHead,
            HeadConfig(type=HeadType.GENOME_TRACKS, name=FINETUNE_HEAD,
                       output_type=dna_output.OutputType.RNA_SEQ, num_tracks=1,
                       metadata=head_meta),
        )
        model = load_checkpoint(
            str(ckpt_dir),
            base_checkpoint_path=args.base_checkpoint_path,
            init_seq_len=seq_len,
        )
        params = model._params
        metrics_file = ckpt_dir / "metrics.json"
        if metrics_file.exists():
            recorded = json.loads(metrics_file.read_text()).get("test_pearson")
    else:
        ckpt_path = ckpt_dir / args.finetune_file
        print(f"Loading pickled checkpoint: {ckpt_path}")
        ckpt = load_ag_checkpoint(ckpt_path)
        hidden_size = infer_hidden_size(ckpt["params"])
        print(f"Head hidden size (from checkpoint): {hidden_size}")
        model = build_model(FINETUNE_HEAD, PlantMPRAHead, seq_len,
                            args.base_checkpoint_path,
                            metadata={"nl_size": hidden_size})
        params = jax.device_put(ckpt["params"], model._device_context._device)
        recorded = ckpt.get("test_pearson")

    print(f"Checkpoint's own recorded test_pearson: {recorded}")

    loader = make_test_loader(model, args)
    t0 = time.time()
    preds, tgts = predict_head(model, params, loader, FINETUNE_HEAD)
    print(f"Scored {len(tgts)} test sequences in {time.time() - t0:.1f}s")
    return preds, tgts, recorded


def test_probe(args, seq_len):
    """Re-extract frozen encoder features and apply the saved ridge head."""
    model = build_model(PROBE_HEAD, ProbePoolHead, seq_len, args.base_checkpoint_path,
                        num_tracks=ENCODER_DIM)

    ckpt_dir = Path(args.checkpoint_dir)
    if (ckpt_dir / "checkpoint").exists():
        import orbax.checkpoint as ocp
        print(f"Loading orbax ridge store: {ckpt_dir / 'checkpoint'}")
        z = ocp.StandardCheckpointer().restore(str(ckpt_dir / "checkpoint"))
    else:
        probe_path = ckpt_dir / args.probe_file
        print(f"Loading probe ridge head: {probe_path}")
        z = np.load(probe_path)
    w, xb, yb = np.asarray(z["w"]), np.asarray(z["xb"]), float(z["yb"])

    loader = make_test_loader(model, args)
    t0 = time.time()
    Xte, yte = predict_head(model, model._params, loader, PROBE_HEAD)
    preds = ridge_predict(Xte, w, xb, yb)
    print(f"Extracted + scored {len(yte)} test sequences in {time.time() - t0:.1f}s "
          f"(lambda={float(z['lam']):g})")
    return preds, yte, None


def reference_value(tissue, mode, method):
    ref = REFERENCE_DIR / f"alphagenome_{tissue}_{mode}_{method}.json"
    if not ref.exists():
        return None
    return json.loads(ref.read_text()).get("test_pearson")


def build_parser():
    p = argparse.ArgumentParser(
        description="Score a saved AlphaGenome plant STARR-seq checkpoint on the test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tissue", type=str, default="leaf", choices=["leaf", "proto"])
    p.add_argument("--mode", type=str, default="combined",
                   choices=["promoter_only", "enhancer", "combined"])
    p.add_argument("--probe", action="store_true",
                   help="Score the ridge probe instead of the fine-tuned head.")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Dir holding the finetune pickle and/or probe_head.npz.")
    p.add_argument("--finetune_file", type=str, default="finetuned_encoder.pkl")
    p.add_argument("--probe_file", type=str, default="probe_head.npz")
    p.add_argument("--base_checkpoint_path", type=str, default=None,
                   help="Local base AlphaGenome all_folds dir (else HF/Kaggle download).")
    p.add_argument("--data_path", type=str, default="./data/jores_plant_starrseq")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--output_dir", type=str, default="./results/plant_starrseq/retest")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    seq_len = PROMOTER_LENGTH if args.mode == "promoter_only" else SEQUENCE_LENGTH
    method = "probe" if args.probe else "finetune"

    print("=" * 80)
    print(f"AlphaGenome plant STARR-seq (Jores 2021) — TEST ({method.upper()})")
    print("=" * 80)
    print(f"Tissue / mode:   {args.tissue} / {args.mode}")
    print(f"Sequence length: {seq_len} bp")
    print(f"Checkpoint dir:  {args.checkpoint_dir}")
    print(f"Device:          {jax.devices()[0]}")
    print("=" * 80)

    if args.probe:
        preds, tgts, ckpt_metric = test_probe(args, seq_len)
    else:
        preds, tgts, ckpt_metric = test_finetune(args, seq_len)

    # np.corrcoef, not scipy.stats.pearsonr: scipy's array-API shim probes the empty
    # `torch` stub in this venv and dies on `torch.Tensor`. Numerically identical.
    test_pearson = float(np.corrcoef(preds.flatten(), tgts.flatten())[0, 1])
    test_mse = float(np.mean((preds.flatten() - tgts.flatten()) ** 2))
    reference = reference_value(args.tissue, args.mode, method)

    out_dir = Path(args.output_dir) / "alphagenome" / args.tissue / args.mode / method
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metrics(
        str(out_dir), "alphagenome", args.tissue, args.mode, method,
        "stage2" if method == "finetune" else "probe",
        test_pearson, test_mse=test_mse, checkpoint=str(args.checkpoint_dir),
    )
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["reference_test_pearson"] = reference
    metrics["checkpoint_recorded_test_pearson"] = (
        float(ckpt_metric) if ckpt_metric is not None else None
    )
    metrics["eval_protocol"] = "best-val checkpoint scored on held-out test"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    if args.save_predictions:
        np.savez(out_dir / "test_predictions.npz", predictions=preds, actuals=tgts)

    print()
    print(f"test_pearson (this script):   {test_pearson:.4f}")
    if reference is not None:
        print(f"test_pearson (reference json): {reference:.4f}   "
              f"delta {test_pearson - reference:+.4f}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    sys.exit(main())
