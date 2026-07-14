"""Finetune / probe AlphaGenome with an MPRA head on plant STARR-seq (Jores 2021).

The Jores21 plant promoter STARR-seq assay measures core-promoter activity in two
systems (leaf, proto) across three data modes:

    promoter_only  — raw 170 bp core promoter
    enhancer       — 437 bp construct WITH the CaMV 35S enhancer
    combined       — 437 bp constructs, +/- enhancer rows

This is the AlphaGenome path (native to this repo's env). ``--probe`` runs a
cache-once linear probe (mean-pooled frozen encoder features + closed-form ridge)
instead of head finetuning. Both write a normalized ``metrics.json`` that
``reproduce_plant_starrseq_table.py`` consumes.

USAGE:
    # two-stage finetune (frozen head -> unfreeze encoder)
    python scripts/finetune_plant_starrseq.py \\
        --config configs/plant_starrseq_alphagenome_leaf.json --mode combined

    # linear probe (frozen backbone, ridge head)
    python scripts/finetune_plant_starrseq.py \\
        --config configs/plant_starrseq_alphagenome_leaf.json --mode combined --probe
"""

import argparse
import json
import sys
import time
from pathlib import Path

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
)
from alphagenome_ft_mpra import (
    EncoderMPRAHead,
    PlantStarrSeqDataset,
    MPRADataLoader,
    train,
)
from alphagenome_ft_mpra.plant_starrseq_utils import (
    PROMOTER_LENGTH,
    SEQUENCE_LENGTH,
    write_run_metrics,
)

ENCODER_DIM = 1536  # AlphaGenome encoder output channels
PROBE_HEAD = "plant_probe_pool"
FINETUNE_HEAD = "mpra_head"


def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r") as f:
        return json.load(f)


# no-param head: returns the mean-pooled frozen encoder output (B, ENCODER_DIM).
# used only by --probe to extract features in one pass per split.
class ProbePoolHead(CustomHead):
    def predict(self, embeddings, organism_index, **kwargs):
        return embeddings.encoder_output.mean(axis=1)

    def loss(self, predictions, batch):
        return {"loss": jnp.mean(predictions ** 2)}


def _ridge_fit(X, y, lam):
    xb = X.mean(0)
    yb = float(y.mean())
    Xc = X - xb
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1], dtype=np.float64)
    w = np.linalg.solve(A, Xc.T @ (y - yb))
    return w, xb, yb


def _ridge_predict(X, w, xb, yb):
    return (X - xb) @ w + yb


def _pearson(a, b):
    return float(np.corrcoef(np.asarray(a).flatten(), np.asarray(b).flatten())[0, 1])


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Finetune/probe AlphaGenome on plant STARR-seq (Jores 2021)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config (CLI args override config values).")

    # Data
    parser.add_argument("--tissue", type=str, default="leaf", choices=["leaf", "proto"])
    parser.add_argument("--mode", type=str, default="combined",
                        choices=["promoter_only", "enhancer", "combined"])
    parser.add_argument("--data_path", type=str, default="./data/jores_plant_starrseq")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--random_shift", action="store_true", default=True)
    parser.add_argument("--random_shift_likelihood", type=float, default=0.5)
    parser.add_argument("--max_shift", type=int, default=15)
    parser.add_argument("--reverse_complement", action="store_true", default=True)

    # Model / head
    parser.add_argument("--center_bp", type=int, default=256)
    parser.add_argument("--pooling_type", type=str, default="flatten",
                        choices=["mean", "sum", "max", "center", "flatten"])
    parser.add_argument("--nl_size", type=str, default="1024")
    parser.add_argument("--do", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--base_checkpoint_path", type=str, default=None)

    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default=None, choices=["plateau", "cosine"])
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--val_eval_frequency", type=int, default=4)
    parser.add_argument("--test_eval_frequency", type=int, default=4)

    # Two-stage
    parser.add_argument("--second_stage_lr", type=float, default=None)
    parser.add_argument("--second_stage_epochs", type=int, default=50)

    # Probe
    parser.add_argument("--probe", action="store_true",
                        help="Run a cache-once linear probe instead of finetuning.")

    # Output / misc
    parser.add_argument("--checkpoint_dir", type=str, default="./results/plant_starrseq/alphagenome")
    parser.add_argument("--results_dir", type=str, default="./results/plant_starrseq")
    parser.add_argument("--no_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="plant-starrseq-alphagenome")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _apply_mode_overrides(parser, config, mode):
    """Apply the ``mode_overrides`` block for the selected mode.

    The sweep tuned the head width per data mode (leaf/combined 4096, proto/combined
    2048, the rest 1024), but there is only one config per tissue — so the per-mode
    values live here rather than in ``model_params``. Without this the runner would
    build a 1024-wide head for every mode and could not reproduce the published runs.
    """
    overrides = (config.get("mode_overrides") or {}).get(mode)
    if not overrides:
        return
    for section, values in overrides.items():
        if section == "model_params":
            parser.set_defaults(**values)
        else:
            raise ValueError(f"Unsupported mode_overrides section: {section!r}")
    print(f"Applied mode_overrides[{mode}]: {overrides}")


def _apply_config_defaults(parser, config):
    if "tissue" in config:
        parser.set_defaults(tissue=config["tissue"])
    if "mode" in config:
        parser.set_defaults(mode=config["mode"])
    d = config.get("data", {})
    parser.set_defaults(
        data_path=d.get("data_path", "./data/jores_plant_starrseq"),
        batch_size=d.get("batch_size", 128),
        random_shift=d.get("random_shift", True),
        random_shift_likelihood=d.get("random_shift_likelihood", 0.5),
        max_shift=d.get("max_shift", 15),
        reverse_complement=d.get("reverse_complement", True),
    )
    m = config.get("model_params", {})
    parser.set_defaults(
        center_bp=m.get("center_bp", 256),
        pooling_type=m.get("pooling_type", "flatten"),
        nl_size=m.get("nl_size", "1024"),
        do=m.get("do", 0.1),
        activation=m.get("activation", "relu"),
    )
    t = config.get("training", {})
    parser.set_defaults(
        num_epochs=t.get("num_epochs", 100),
        learning_rate=t.get("learning_rate", 3e-4),
        optimizer=t.get("optimizer", "adam"),
        weight_decay=t.get("weight_decay", 1e-6),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        gradient_clip=t.get("gradient_clip", 1.0),
        lr_scheduler=t.get("lr_scheduler", None),
        early_stopping_patience=t.get("early_stopping_patience", 5),
        val_eval_frequency=t.get("val_eval_frequency", 4),
        test_eval_frequency=t.get("test_eval_frequency", 4),
    )
    ts = config.get("two_stage", {})
    if ts.get("enabled", False):
        parser.set_defaults(
            second_stage_lr=ts.get("second_stage_lr", None),
            second_stage_epochs=ts.get("second_stage_epochs", 50),
        )
    if "base_checkpoint_path" in config:
        parser.set_defaults(base_checkpoint_path=config["base_checkpoint_path"])


def _make_datasets(model, args, seq_len):
    train_ds = PlantStarrSeqDataset(
        model=model, path_to_data=args.data_path, tissue=args.tissue, mode=args.mode,
        split="train", random_shift=args.random_shift,
        random_shift_likelihood=args.random_shift_likelihood, max_shift=args.max_shift,
        reverse_complement=args.reverse_complement, seed=args.seed,
    )
    val_ds = PlantStarrSeqDataset(
        model=model, path_to_data=args.data_path, tissue=args.tissue, mode=args.mode,
        split="val", random_shift=False, reverse_complement=False, seed=args.seed,
    )
    test_ds = PlantStarrSeqDataset(
        model=model, path_to_data=args.data_path, tissue=args.tissue, mode=args.mode,
        split="test", random_shift=False, reverse_complement=False, seed=args.seed,
    )
    return train_ds, val_ds, test_ds


def run_finetune(args, seq_len):
    if "," in args.nl_size:
        nl_size = [int(x.strip()) for x in args.nl_size.split(",")]
    else:
        nl_size = int(args.nl_size)

    register_custom_head(
        FINETUNE_HEAD,
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS, name=FINETUNE_HEAD,
            output_type=dna_output.OutputType.RNA_SEQ, num_tracks=1,
            metadata={
                "center_bp": args.center_bp, "pooling_type": args.pooling_type,
                "nl_size": nl_size, "do": args.do, "activation": args.activation,
            },
        ),
    )

    model = create_model_with_custom_heads(
        "all_folds", custom_heads=[FINETUNE_HEAD],
        checkpoint_path=args.base_checkpoint_path,
        use_encoder_output=True, init_seq_len=seq_len,
    )
    model.freeze_except_head(FINETUNE_HEAD)

    train_ds, val_ds, test_ds = _make_datasets(model, args, seq_len)
    train_loader = MPRADataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = MPRADataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = MPRADataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    ckpt_dir = Path(args.checkpoint_dir) / args.tissue / args.mode

    history = train(
        model, train_loader, val_loader, test_loader,
        num_epochs=args.num_epochs, learning_rate=args.learning_rate,
        checkpoint_dir=str(ckpt_dir), save_minimal_model=True,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip=args.gradient_clip,
        val_eval_frequency=args.val_eval_frequency,
        test_eval_frequency=args.test_eval_frequency,
        second_stage_lr=args.second_stage_lr, second_stage_epochs=args.second_stage_epochs,
        use_wandb=not args.no_wandb, wandb_project=args.wandb_project,
        lr_scheduler=args.lr_scheduler,
    )

    # Report the metrics of the checkpoint that was actually saved. ``train`` writes
    # best.pt on the lowest val_loss, so the reported test score must come from that
    # same epoch — not from the last one (which is a different, typically overfit
    # model) and not from the best test epoch (which would be selecting on test).
    val_pearson = test_pearson = None
    val_losses = history.get("val_loss") or []
    if val_losses:
        best_idx = int(np.argmin(val_losses))
        val_pearsons = history.get("val_pearson") or []
        test_pearsons = history.get("test_pearson") or []
        if best_idx < len(val_pearsons):
            val_pearson = val_pearsons[best_idx]
        if best_idx < len(test_pearsons):
            test_pearson = test_pearsons[best_idx]
        else:
            # val and test were evaluated on different cadences, so the best-val epoch
            # has no matching test eval — refuse to report a mismatched number.
            print(f"WARNING: no test eval at the best-val epoch ({best_idx}); "
                  f"val_eval_frequency and test_eval_frequency must match for the "
                  f"reported test_pearson to describe the saved checkpoint.")
    stage = "stage2" if args.second_stage_lr else "stage1"

    out_dir = Path(args.results_dir) / "alphagenome" / args.tissue / args.mode / "finetune"
    write_run_metrics(
        str(out_dir), "alphagenome", args.tissue, args.mode, "finetune", stage,
        test_pearson, val_pearson=val_pearson, checkpoint=str(ckpt_dir),
    )
    print(f"\nFinetune done. tissue={args.tissue} mode={args.mode} "
          f"val_pearson={val_pearson} test_pearson={test_pearson}")
    return test_pearson


def run_probe(args, seq_len):
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model

    register_custom_head(
        PROBE_HEAD, ProbePoolHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS, name=PROBE_HEAD,
            output_type=dna_output.OutputType.RNA_SEQ, num_tracks=ENCODER_DIM,
        ),
    )
    model = create_model_with_custom_heads(
        "all_folds", custom_heads=[PROBE_HEAD],
        checkpoint_path=args.base_checkpoint_path,
        use_encoder_output=True, init_seq_len=seq_len,
    )

    organism_enum = ag_dna_model.Organism.HOMO_SAPIENS
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = jax.device_put(
        model._metadata[organism_enum].strand_reindexing, model._device_context._device,
    )
    params, state = model._params, model._state

    def extract(loader):
        feats, tgts = [], []
        for batch in loader:
            seqs = batch["seq"]
            B, L = seqs.shape[0], seqs.shape[1]
            preds = model._predict(
                params, state, seqs,
                jnp.full((B,), organism_index_value, dtype=jnp.int32),
                requested_outputs=None,
                negative_strand_mask=jnp.zeros((B, L), dtype=jnp.bool_),
                strand_reindexing=strand_reindexing,
            )
            feats.append(np.asarray(preds[PROBE_HEAD], dtype=np.float32))
            tgts.append(np.asarray(batch["y"], dtype=np.float32))
        return np.concatenate(feats), np.concatenate(tgts)

    train_ds, val_ds, test_ds = _make_datasets(model, args, seq_len)
    # probe features are deterministic — no augmentation
    train_ds.random_shift = train_ds.reverse_complement = False
    loaders = [MPRADataLoader(ds, batch_size=args.batch_size, shuffle=False)
               for ds in (train_ds, val_ds, test_ds)]

    t0 = time.time()
    Xtr, ytr = extract(loaders[0])
    Xva, yva = extract(loaders[1])
    Xte, yte = extract(loaders[2])

    best = None
    for lam in [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        w, xb, yb = _ridge_fit(Xtr.astype(np.float64), ytr.astype(np.float64), lam)
        vr = _pearson(_ridge_predict(Xva, w, xb, yb), yva)
        if best is None or vr > best[0]:
            best = (vr, lam, w, xb, yb)
    val_pearson, lam, w, xb, yb = best

    pred_te = _ridge_predict(Xte, w, xb, yb)
    test_pearson = _pearson(pred_te, yte)
    test_mse = float(np.mean((pred_te - yte) ** 2))

    out_dir = Path(args.results_dir) / "alphagenome" / args.tissue / args.mode / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "probe_head.npz", w=w, xb=xb, yb=yb, lam=lam)
    write_run_metrics(
        str(out_dir), "alphagenome", args.tissue, args.mode, "probe", "probe",
        test_pearson, val_pearson=val_pearson, test_mse=test_mse,
        checkpoint=str(out_dir / "probe_head.npz"),
    )
    print(f"\nProbe done. tissue={args.tissue} mode={args.mode} best_lambda={lam} "
          f"val_pearson={val_pearson:.4f} test_pearson={test_pearson:.4f} "
          f"extract_seconds={time.time() - t0:.1f}")
    return test_pearson


def main():
    parser = _build_parser()
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        print(f"Loading config from: {temp_args.config}")
        config = load_config(temp_args.config)
        _apply_config_defaults(parser, config)
        # re-parse so `mode` reflects CLI > config > argparse default, then apply the
        # per-mode head overrides on top
        mode_args, _ = parser.parse_known_args()
        _apply_mode_overrides(parser, config, mode_args.mode)
    args = parser.parse_args()

    seq_len = PROMOTER_LENGTH if args.mode == "promoter_only" else SEQUENCE_LENGTH

    print("=" * 80)
    print(f"AlphaGenome plant STARR-seq (Jores 2021) — "
          f"{'PROBE' if args.probe else 'FINETUNE'}")
    print("=" * 80)
    print(f"Tissue / mode:   {args.tissue} / {args.mode}")
    print(f"Sequence length: {seq_len} bp")
    print(f"Data path:       {args.data_path}")
    print("=" * 80)

    if args.probe:
        run_probe(args, seq_len)
    else:
        run_finetune(args, seq_len)


if __name__ == "__main__":
    sys.exit(main())
