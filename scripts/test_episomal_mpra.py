"""Test fine-tuned MPRA models on the 3 episomal MPRA test sets.

Test sets evaluated (when available under <data_path>/test_sets/):
  1. reference  — chr7/chr13 genomic test sequences with measured log2FC
  2. designed   — high-activity designed sequences (OOD)
  3. snv        — SNV pairs; metric is Pearson(predicted_delta, observed_delta)

Currently supported model types:
  - ag_probing, ag_finetuned        : AlphaGenome with MPRA head
  - enformer_probing, enformer_finetuned : conv-only Enformer with MPRA head

Outputs:
  - <output_dir>/<run_name>_<cell>_metrics.json   (all test sets)
  - <output_dir>/<run_name>_<cell>_<set>_predictions.csv (per test set)
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome_ft_mpra.episomal_utils import (
    SEQUENCE_LENGTH,
    _load_gosai_data,
    _one_hot_encode,
    pad_n_bases as _pad_n_bases,
    get_episomal_test_sets,
)


# ── Metric helpers ────────────────────────────────────────────────────


def _pearson(y_pred, y_true):
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    if len(y_pred) < 2:
        return float("nan")
    return float(np.corrcoef(y_pred, y_true)[0, 1])


def _metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    mse = float(np.mean((y_pred - y_true) ** 2))
    pearson = _pearson(y_pred, y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "n_samples": int(len(y_pred)),
        "mse": mse,
        "pearson": pearson,
        "r2": r2,
    }


# ── AlphaGenome inference ─────────────────────────────────────────────


def _ag_predict_batch(model, sequences, head_name="mpra_head"):
    """Predict scalar log2FC for a list of DNA strings using a trained AG model."""
    import jax
    import jax.numpy as jnp
    from alphagenome.models import dna_output
    from alphagenome_research.model import dna_model

    organism = dna_model.Organism.HOMO_SAPIENS
    requested = tuple(dna_output.OutputType)

    head_config = getattr(model, "_head_configs", {}).get(head_name, None)
    metadata = getattr(head_config, "metadata", {}) if head_config else {}
    pooling_type = metadata.get("pooling_type", "sum")
    center_bp = metadata.get("center_bp", 256)

    preds = []
    for s in sequences:
        if len(s) < SEQUENCE_LENGTH:
            pad = SEQUENCE_LENGTH - len(s)
            s = "N" * (pad // 2) + s + "N" * (pad - pad // 2)
        elif len(s) > SEQUENCE_LENGTH:
            start = (len(s) - SEQUENCE_LENGTH) // 2
            s = s[start:start + SEQUENCE_LENGTH]
        ohe = model._one_hot_encoder.encode(s, organism)
        seq_batch = jnp.expand_dims(ohe, 0)
        with model._device_context:
            out = model._predict(
                model._params,
                model._state,
                seq_batch,
                jnp.array([[0]]),
                requested_outputs=requested,
                negative_strand_mask=jnp.zeros(1, dtype=bool),
                strand_reindexing=jax.device_put(
                    model._metadata[organism].strand_reindexing,
                    model._device_context._device,
                ),
            )
        head_out = np.array(out[head_name])
        seq_len = head_out.shape[1]
        if pooling_type == "flatten":
            pooled = head_out.squeeze(1)
        elif pooling_type == "center":
            pooled = head_out[:, seq_len // 2, :]
        else:
            window = max(1, center_bp // 128)
            window = min(window, seq_len)
            cs = max((seq_len - window) // 2, 0)
            ce = cs + window
            slice_ = head_out[:, cs:ce, :]
            if pooling_type == "mean":
                pooled = slice_.mean(axis=1)
            elif pooling_type == "max":
                pooled = slice_.max(axis=1)
            else:  # sum
                pooled = slice_.sum(axis=1)
        if pooled.shape[-1] == 1:
            pooled = pooled[:, 0]
        preds.append(float(pooled[0]))
    return np.asarray(preds, dtype=np.float32)


def _load_ag_model(checkpoint_dir, base_checkpoint_path=None, init_seq_len=200):
    from alphagenome.models import dna_output  # noqa
    from alphagenome_ft import (
        HeadConfig,
        HeadType,
        register_custom_head,
        load_checkpoint,
    )
    from alphagenome_ft_mpra import EncoderMPRAHead

    ckpt_dir = Path(checkpoint_dir).resolve()
    if not (ckpt_dir / "config.json").exists():
        for sub in ("stage2", "stage1"):
            cand = ckpt_dir / sub
            if (cand / "config.json").exists():
                ckpt_dir = cand
                break
    head_metadata = {"center_bp": 256, "pooling_type": "flatten"}
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            head_cfg = (
                cfg.get("head_configs", {}).get("mpra_head", {}).get("metadata", {})
            )
            head_metadata.update(head_cfg)
        except Exception:
            pass

    register_custom_head(
        "mpra_head",
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name="mpra_head",
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata=head_metadata,
        ),
    )

    return load_checkpoint(
        str(ckpt_dir),
        base_model_version="all_folds",
        base_checkpoint_path=base_checkpoint_path,
        device=None,
        init_seq_len=init_seq_len,
    )


# ── Enformer inference ────────────────────────────────────────────────


def _load_enformer_model(checkpoint_path):
    """Load the Lightning checkpoint trained by finetune_enformer_episomal_mpra.py."""
    finetune_path = (
        Path(__file__).parent / "finetune_enformer_episomal_mpra.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finetune_enformer_episomal_mpra", finetune_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    EnformerEpisomalLightning = module.EnformerEpisomalLightning
    pad = module.EPISOMAL_PAD_TO_LENTI_CONSTRUCT

    model = EnformerEpisomalLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model, pad


def _enformer_predict_batch(model, sequences, pad_n_bases=81, batch_size=32, device="cuda"):
    import torch

    model = model.to(device)
    preds = []
    target_len = SEQUENCE_LENGTH + pad_n_bases  # must match training input length

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            ohes = []
            for s in batch:
                # Center-crop or center-pad to SEQUENCE_LENGTH.
                if len(s) < SEQUENCE_LENGTH:
                    s = _pad_n_bases(s, SEQUENCE_LENGTH - len(s))
                elif len(s) > SEQUENCE_LENGTH:
                    start = (len(s) - SEQUENCE_LENGTH) // 2
                    s = s[start:start + SEQUENCE_LENGTH]
                # Then add the train-time pad_n_bases via the shared helper
                # so train and inference cannot drift.
                s = _pad_n_bases(s, pad_n_bases)
                # Defensive truncate/pad to target_len in case upstream lied
                # about input length.
                if len(s) > target_len:
                    s = s[:target_len]
                elif len(s) < target_len:
                    s = s + "N" * (target_len - len(s))
                ohes.append(_one_hot_encode(s))
            seq_batch = torch.tensor(np.stack(ohes, axis=0), dtype=torch.float32,
                                     device=device)
            out = model.forward(seq_batch)
            preds.append(out.cpu().numpy().flatten())
    return np.concatenate(preds)


# ── Test set runners ──────────────────────────────────────────────────


def _run_predict(predict_fn, test_sets):
    """Run predict_fn on each test set; return {test_set: {metrics, predictions}}."""
    results = {}
    for name, payload in test_sets.items():
        if name == "snv":
            ref_preds = predict_fn(payload["ref_sequences"])
            alt_preds = predict_fn(payload["alt_sequences"])
            pred_delta = alt_preds - ref_preds
            metrics = _metrics(pred_delta, payload["true_delta"])
            results[name] = {
                "metrics": metrics,
                "predictions": {
                    "ref_pred": ref_preds.tolist(),
                    "alt_pred": alt_preds.tolist(),
                    "pred_delta": pred_delta.tolist(),
                    "true_delta": payload["true_delta"].tolist(),
                },
            }
        else:
            preds = predict_fn(payload["sequences"])
            metrics = _metrics(preds, payload["labels"])
            results[name] = {
                "metrics": metrics,
                "predictions": {
                    "prediction": preds.tolist(),
                    "actual": np.asarray(payload["labels"]).tolist(),
                },
            }
        print(f"  [{name}] N={metrics['n_samples']} "
              f"Pearson={metrics['pearson']:.4f} MSE={metrics['mse']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate episomal MPRA models on reference / designed / SNV test sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_type", type=str, required=True,
                        choices=[
                            "ag_probing", "ag_finetuned",
                            "enformer_probing", "enformer_finetuned",
                        ])
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="AG: directory; Enformer: .ckpt file")
    parser.add_argument("--cell_type", type=str, default="K562",
                        choices=["K562", "HepG2", "SKNSH"])
    parser.add_argument("--data_path", type=str, default="./data/gosai_episomal")
    parser.add_argument("--base_checkpoint_path", type=str, default=None,
                        help="Local AG base checkpoint (AG models only)")
    parser.add_argument("--output_dir", type=str,
                        default="./results/episomal_predictions")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override; defaults to checkpoint dirname")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--init_seq_len", type=int, default=200)
    args = parser.parse_args()

    run_name = args.run_name or Path(args.checkpoint_path).name

    print("=" * 80)
    print(f"Testing {args.model_type} on episomal MPRA ({args.cell_type})")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Data path:  {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Build test sets dict (reference always available; designed/snv if files exist)
    test_sets = get_episomal_test_sets(args.data_path, args.cell_type)
    if not test_sets:
        # Fallback: at least include the reference split, even if helper found nothing
        ref = _load_gosai_data(args.data_path, args.cell_type, "test")
        test_sets["reference"] = {
            "sequences": ref["sequence"].tolist(),
            "labels": ref["label"].values,
        }
    print(f"Found test sets: {list(test_sets.keys())}")

    # Build predict_fn for the chosen backend
    if args.model_type.startswith("ag_"):
        print("\nLoading AlphaGenome model...")
        model = _load_ag_model(
            args.checkpoint_path,
            base_checkpoint_path=args.base_checkpoint_path,
            init_seq_len=args.init_seq_len,
        )
        predict_fn = lambda seqs: _ag_predict_batch(model, list(seqs))  # noqa: E731
    else:
        print("\nLoading Enformer Lightning checkpoint...")
        model, pad = _load_enformer_model(args.checkpoint_path)
        device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"
        predict_fn = lambda seqs: _enformer_predict_batch(  # noqa: E731
            model, list(seqs), pad_n_bases=pad,
            batch_size=args.batch_size, device=device,
        )

    print("\nRunning evaluation on test sets...")
    results = _run_predict(predict_fn, test_sets)

    # Save metrics JSON
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / f"{run_name}_{args.cell_type}_metrics.json"
    metrics_summary = {
        "model_type": args.model_type,
        "cell_type": args.cell_type,
        "checkpoint_path": str(args.checkpoint_path),
        "test_sets": {k: v["metrics"] for k, v in results.items()},
    }
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))
    print(f"\n✓ Saved metrics to {metrics_path}")

    # Save predictions CSV per test set
    for name, payload in results.items():
        pred_path = out / f"{run_name}_{args.cell_type}_{name}_predictions.csv"
        pd.DataFrame(payload["predictions"]).to_csv(pred_path, index=False)
        print(f"✓ Saved {name} predictions to {pred_path}")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for name, payload in results.items():
        m = payload["metrics"]
        print(f"  {name:10s} N={m['n_samples']:6d}  "
              f"Pearson={m['pearson']:.4f}  MSE={m['mse']:.4f}  R²={m['r2']:.4f}")


if __name__ == "__main__":
    main()
