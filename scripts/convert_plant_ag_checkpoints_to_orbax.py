"""Convert the plant STARR-seq AlphaGenome checkpoints to the shared orbax layout.

The checkpoints as produced by ``autotune`` are single Python pickles of a Haiku param
tree (misleadingly named ``.pt``). Every other AlphaGenome checkpoint in
``/grid/koo/home/shared/models/alphagenome_encoder/jax/`` uses the layout that
``alphagenome_ft``'s ``model.save_checkpoint()`` emits:

    <run>/stage1/{config.json, checkpoint/}      # orbax store
    <run>/stage2/{config.json, checkpoint/}

This script rebuilds the model with the plant head, injects the pickled params, and
re-saves through ``save_checkpoint`` so the result is a first-class checkpoint that
``alphagenome_ft.load_checkpoint()`` can open — same as deepstarr-optimal et al.

The stage-1 probe is a closed-form ridge on frozen encoder features, not a trained
model, so it has no AlphaGenome params of its own. It is written as an orbax store of
the ridge arrays (w, xb, yb, lam) plus a config.json that says so.

USAGE:
    python scripts/convert_plant_ag_checkpoints_to_orbax.py \\
        --src /grid/koo/home/shared/models/alphagenome_encoder/jax \\
        --dest /grid/koo/home/shared/models/alphagenome_encoder/jax \\
        --base_checkpoint_path <local all_folds dir>
"""

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ft_model_plant_starrseq import (  # noqa: E402
    FINETUNE_HEAD,
    PlantMPRAHead,
    build_model,
    infer_hidden_size,
)
from alphagenome_ft_mpra.plant_starrseq_utils import (  # noqa: E402
    PROMOTER_LENGTH,
    SEQUENCE_LENGTH,
)

TISSUES = ["leaf", "proto"]
MODES = ["combined", "enhancer", "promoter_only"]


def convert_stage2(src_dir, dest_dir, tissue, mode, base_ckpt):
    """Rebuild the model, inject the pickled params, re-save via save_checkpoint."""
    pkl = src_dir / f"plant_starrseq_{tissue}_{mode}" / "finetuned_encoder.pkl"
    with open(pkl, "rb") as f:
        ckpt = pickle.load(f)

    seq_len = PROMOTER_LENGTH if mode == "promoter_only" else SEQUENCE_LENGTH
    hidden_size = infer_hidden_size(ckpt["params"])

    model = build_model(FINETUNE_HEAD, PlantMPRAHead, seq_len, base_ckpt,
                        metadata={"nl_size": hidden_size, "pooling_type": "flatten",
                                  "activation": "relu", "do": 0.1})

    head_key = f"head/{FINETUNE_HEAD}/~predict/linear"
    print(f"  pickle head width : {ckpt['params'][head_key]['w'].shape}")
    print(f"  freshly-built head: {model._params[head_key]['w'].shape}")

    # swap in the fine-tuned params (the pickle carries backbone + head)
    model._params = jax.device_put(ckpt["params"], model._device_context._device)
    print(f"  after assignment  : {model._params[head_key]['w'].shape}")

    # The saved checkpoint must contain the FINE-TUNED head, not the freshly-initialised
    # one. Assert rather than trust: a silent mismatch here writes a useless 1.8GB file.
    saved_w = model._params[head_key]["w"]
    expected_w = ckpt["params"][head_key]["w"]
    assert saved_w.shape == expected_w.shape, (
        f"head shape {saved_w.shape} != checkpoint {expected_w.shape}")
    assert np.allclose(np.asarray(saved_w), np.asarray(expected_w)), \
        "model params are not the checkpoint's params — assignment did not take"

    out = dest_dir / f"plant-starrseq-{tissue}-{mode}" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(out), save_minimal_model=True)

    # keep the provenance the pickle carried
    meta = out / "metrics.json"
    meta.write_text(json.dumps({
        "val_pearson": float(ckpt["val_pearson"]),
        "test_pearson": float(ckpt["test_pearson"]),
        "source_config": ckpt.get("config"),
        "source_pickle": str(pkl),
    }, indent=2) + "\n")
    print(f"  stage2 -> {out}  (val {ckpt['val_pearson']:.4f} / test {ckpt['test_pearson']:.4f})")
    return hidden_size


def convert_stage1(src_dir, dest_dir, tissue, mode):
    """Write the ridge probe as an orbax store + a config.json describing it."""
    npz = src_dir / f"plant_starrseq_{tissue}_{mode}" / "probe_head.npz"
    z = np.load(npz)
    ridge = {k: np.asarray(z[k]) for k in ("w", "xb", "yb", "lam")}

    out = dest_dir / f"plant-starrseq-{tissue}-{mode}" / "stage1"
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = out / "checkpoint"
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(ckpt_path), ridge)
    checkpointer.wait_until_finished()

    (out / "config.json").write_text(json.dumps({
        "custom_heads": ["plant_probe_ridge"],
        "head_configs": {
            "plant_probe_ridge": {
                "source": "ridge",
                "note": ("Closed-form ridge on mean-pooled frozen AlphaGenome encoder "
                         "features. Not an alphagenome_ft model checkpoint — the encoder "
                         "is the unmodified pretrained all_folds backbone."),
                "encoder_dim": int(ridge["w"].shape[0]),
                "lambda": float(ridge["lam"]),
                "predict": "(X - xb) @ w + yb, X = encoder_output.mean(axis=1)",
            }
        },
        "save_full_model": False,
        "save_minimal_model": False,
        "use_encoder_output": True,
    }, indent=2) + "\n")
    print(f"  stage1 -> {out}  (ridge, lambda={float(ridge['lam']):g})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--src", type=str, required=True,
                   help="Dir holding the flat plant_starrseq_<tissue>_<mode>/ entries.")
    p.add_argument("--dest", type=str, required=True)
    p.add_argument("--base_checkpoint_path", type=str, default=None)
    p.add_argument("--tissue", type=str, default=None, choices=TISSUES)
    p.add_argument("--mode", type=str, default=None, choices=MODES)
    args = p.parse_args()

    src, dest = Path(args.src), Path(args.dest)
    tissues = [args.tissue] if args.tissue else TISSUES
    modes = [args.mode] if args.mode else MODES

    for tissue in tissues:
        for mode in modes:
            print(f"=== {tissue} / {mode} ===")
            convert_stage1(src, dest, tissue, mode)
            convert_stage2(src, dest, tissue, mode, args.base_checkpoint_path)


if __name__ == "__main__":
    sys.exit(main())
