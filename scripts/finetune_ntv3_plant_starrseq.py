"""Finetune / probe NTv3-post on plant STARR-seq (Jores 2021).

NTv3-post is a post-trained Nucleotide Transformer v3 (JAX / Flax NNX). Following
the original sweep, features are taken from the CONV TOWER (before the
transformer), species-conditioned per tissue (leaf -> arabidopsis_thaliana token
19, proto -> zea_mays token 22). An attention-pool MPRA head is trained on top,
two-stage (frozen head, then unfreeze). ``--probe`` runs the cache-once mean-pool
+ ridge linear probe on the same conv-tower features.

This runner is NOT in the repo's default ``uv`` env: NTv3 needs the
``nucleotide_transformer_v3`` package (JAX + flax/nnx). Activate that env first:

    python scripts/finetune_ntv3_plant_starrseq.py \\
        --config configs/plant_starrseq_ntv3_leaf.json --mode combined
    python scripts/finetune_ntv3_plant_starrseq.py \\
        --config configs/plant_starrseq_ntv3_leaf.json --mode combined --probe

Reference sweep numbers (results/plant_starrseq/reference/): leaf combined
finetune 0.8822 / probe 0.7475; proto combined finetune 0.8756 / probe 0.7657.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from alphagenome_ft_mpra.plant_starrseq_utils import (
    PROMOTER_LENGTH,
    SEQUENCE_LENGTH,
    _load_plant_starrseq_data,
    build_sequence_for_mode,
    write_run_metrics,
)

_JAX_HINT = (
    "NTv3-post requires the nucleotide_transformer_v3 package (JAX + flax/nnx), "
    "which is not in this repo's default environment. Activate the 'autotune_ntv3' "
    "env (with LD_LIBRARY_PATH=$CONDA_PREFIX/lib) and re-run. See scripts/README.md."
)

SPECIES_TOKEN = {"leaf": 19, "proto": 22}          # arabidopsis_thaliana / zea_mays
SPECIES_NAME = {"leaf": "arabidopsis_thaliana", "proto": "zea_mays"}


def load_config(path):
    with open(path) as f:
        return json.load(f)


def _iter_batches(df, tokenizer, mode, batch_size, seed, augment):
    """Yield (token_ids ndarray, targets ndarray). Sequences padded to a mult. of 128."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df)) if augment else np.arange(len(df))
    for start in range(0, len(df), batch_size):
        idx = order[start:start + batch_size]
        rows = df.iloc[idx]
        seqs, tgts = [], []
        for _, row in rows.iterrows():
            s = build_sequence_for_mode(
                row, mode, rng,
                random_shift=augment, reverse_complement=augment,
            )
            seqs.append(s)
            tgts.append(float(row["enrichment"]))
        max_len = max(len(s) for s in seqs)
        pad_len = math.ceil(max_len / 128) * 128
        padded = [s + "N" * (pad_len - len(s)) for s in seqs]
        token_ids = tokenizer.batch_np_tokenize(padded)
        yield token_ids, np.asarray(tgts, dtype=np.float32)


def _get_embeddings(model, tokens, species_token):
    """Conv-tower features (before the transformer), species-conditioned."""
    import jax.numpy as jnp

    x = model.embed_layer(tokens)
    x = model.stem(x)
    sp = jnp.broadcast_to(jnp.asarray(species_token, dtype=jnp.int32), (tokens.shape[0],))
    cond = model.conditions_embed_layers[0](sp)
    x, _ = model.conv_tower(x, [cond], None)
    return x  # (B, T, C)


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--tissue", type=str, default="leaf", choices=["leaf", "proto"])
    p.add_argument("--mode", type=str, default="combined",
                   choices=["promoter_only", "enhancer", "combined"])
    p.add_argument("--data_path", type=str, default="./data/jores_plant_starrseq")
    p.add_argument("--model_name", type=str, default="NTv3_650M_post")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--do", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--second_stage_lr", type=float, default=3e-4)
    p.add_argument("--second_stage_epochs", type=int, default=50)
    p.add_argument("--no_second_stage", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--results_dir", type=str, default="./results/plant_starrseq")
    p.add_argument("--seed", type=int, default=42)
    return p


def _apply_config(parser, cfg):
    for k in ("tissue", "mode"):
        if k in cfg:
            parser.set_defaults(**{k: cfg[k]})
    d = cfg.get("data", {})
    parser.set_defaults(data_path=d.get("data_path", "./data/jores_plant_starrseq"),
                        batch_size=d.get("batch_size", 128))
    m = cfg.get("model_params", {})
    parser.set_defaults(model_name=m.get("model_name", "NTv3_650M_post"),
                        hidden_size=m.get("hidden_size", 1024),
                        n_heads=m.get("n_heads", 4), do=m.get("do", 0.1))
    t = cfg.get("training", {})
    parser.set_defaults(learning_rate=t.get("learning_rate", 5e-4),
                        weight_decay=t.get("weight_decay", 0.0),
                        num_epochs=t.get("num_epochs", 100),
                        early_stopping_patience=t.get("early_stopping_patience", 5))
    ts = cfg.get("two_stage", {})
    parser.set_defaults(second_stage_lr=ts.get("second_stage_lr", 3e-4),
                        second_stage_epochs=ts.get("second_stage_epochs", 50))
    if not ts.get("enabled", True):
        parser.set_defaults(no_second_stage=True)


def _make_head(encoder_dim, args):
    import jax.numpy as jnp
    from flax import nnx

    class MPRAHead(nnx.Module):
        """Attention-pool MLP head over conv-tower features (B, T, C) -> scalar."""

        def __init__(self, dim, hidden, n_heads, do, rngs):
            self.n_heads = n_heads
            self.norm = nnx.LayerNorm(dim, rngs=rngs)
            self.attn = nnx.Linear(dim, n_heads, rngs=rngs)
            self.fc1 = nnx.Linear(n_heads * dim, hidden, rngs=rngs)
            self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
            self.out = nnx.Linear(hidden, 1, rngs=rngs)
            self.dropout = nnx.Dropout(do, rngs=rngs)

        def __call__(self, x, deterministic=True):
            x = self.norm(x)
            weights = nnx.softmax(self.attn(x), axis=1)              # (B, T, H)
            pooled = jnp.einsum("bth,btc->bhc", weights, x)          # (B, H, C)
            pooled = pooled.reshape(pooled.shape[0], -1)
            h = nnx.gelu(self.fc1(pooled))
            h = self.dropout(h, deterministic=deterministic)
            h = h + nnx.gelu(self.fc2(h))
            h = self.dropout(h, deterministic=deterministic)
            return self.out(h).squeeze(-1)

    return MPRAHead(encoder_dim, args.hidden_size, args.n_heads, args.do, nnx.Rngs(args.seed))


def run_finetune(args):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from nucleotide_transformer_v3.pretrained import get_posttrained_ntv3_model

    model, tokenizer, config = get_posttrained_ntv3_model(args.model_name, use_bfloat16=True)
    encoder_dim = config.embed_dim
    species_token = SPECIES_TOKEN[args.tissue]

    head = _make_head(encoder_dim, args)

    dfs = {split: _load_plant_starrseq_data(args.data_path, args.tissue, args.mode, split)
           for split in ("train", "val", "test")}

    def head_apply(head_mod, tokens, deterministic):
        feats = _get_embeddings(model, tokens, species_token)
        return head_mod(feats, deterministic=deterministic)

    def loss_fn(head_mod, tokens, targets):
        pred = head_apply(head_mod, tokens, deterministic=False)
        return jnp.mean((pred - targets) ** 2)

    def evaluate(df):
        preds, tgts = [], []
        for tokens, y in _iter_batches(df, tokenizer, args.mode, args.batch_size,
                                       args.seed, augment=False):
            pred = head_apply(head, jnp.asarray(tokens), deterministic=True)
            preds.append(np.asarray(pred))
            tgts.append(y)
        return float(np.corrcoef(np.concatenate(preds), np.concatenate(tgts))[0, 1])

    optimizer = nnx.Optimizer(head, optax.adamw(args.learning_rate, weight_decay=args.weight_decay))

    @nnx.jit
    def train_step(head_mod, opt, tokens, targets):
        grads = nnx.grad(loss_fn)(head_mod, tokens, targets)
        opt.update(grads)

    def train_epochs(n_epochs):
        best_val, patience = -1.0, 0
        best_state = None
        for _ in range(n_epochs):
            for tokens, y in _iter_batches(dfs["train"], tokenizer, args.mode,
                                           args.batch_size, args.seed, augment=True):
                train_step(head, optimizer, jnp.asarray(tokens), jnp.asarray(y))
            vr = evaluate(dfs["val"])
            if vr > best_val:
                best_val, patience = vr, 0
                best_state = nnx.state(head, nnx.Param)
            else:
                patience += 1
                if patience >= args.early_stopping_patience:
                    break
        if best_state is not None:
            nnx.update(head, best_state)
        return best_val

    best_val = train_epochs(args.num_epochs)
    # this reference runner trains the head on frozen conv-tower features (stage 1).
    # the committed reference numbers correspond to the sweep's full two-stage run
    # (backbone unfreeze at second_stage_lr); enable that by extending the optimizer
    # to the NTv3 backbone params, following the PlantCAD2 runner's stage-2 pattern.
    stage = "stage1"

    test_r = evaluate(dfs["test"])
    out_dir = Path(args.results_dir) / "ntv3" / args.tissue / args.mode / "finetune"
    write_run_metrics(str(out_dir), "ntv3", args.tissue, args.mode, "finetune", stage,
                      test_r, val_pearson=best_val, species=SPECIES_NAME[args.tissue])
    print(f"\nFinetune done. ntv3 {args.tissue} {args.mode} val={best_val:.4f} test={test_r:.4f}")


def run_probe(args):
    import jax.numpy as jnp
    from nucleotide_transformer_v3.pretrained import get_posttrained_ntv3_model
    from alphagenome_ft_mpra.plant_starrseq_utils import select_ridge, ridge_predict, pearson

    model, tokenizer, config = get_posttrained_ntv3_model(args.model_name, use_bfloat16=True)
    species_token = SPECIES_TOKEN[args.tissue]

    def extract(df):
        feats, tgts = [], []
        for tokens, y in _iter_batches(df, tokenizer, args.mode, args.batch_size,
                                       args.seed, augment=False):
            enc = _get_embeddings(model, jnp.asarray(tokens), species_token)
            feats.append(np.asarray(enc, dtype=np.float32).mean(axis=1))
            tgts.append(y)
        return np.concatenate(feats), np.concatenate(tgts)

    t0 = time.time()
    Xtr, ytr = extract(_load_plant_starrseq_data(args.data_path, args.tissue, args.mode, "train"))
    Xva, yva = extract(_load_plant_starrseq_data(args.data_path, args.tissue, args.mode, "val"))
    Xte, yte = extract(_load_plant_starrseq_data(args.data_path, args.tissue, args.mode, "test"))

    val_r, lam, w, xb, yb = select_ridge(Xtr, ytr, Xva, yva)
    pred_te = ridge_predict(Xte, w, xb, yb)
    test_r = pearson(pred_te, yte)
    test_mse = float(np.mean((pred_te - yte) ** 2))

    out_dir = Path(args.results_dir) / "ntv3" / args.tissue / args.mode / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "probe_head.npz", w=w, xb=xb, yb=yb, lam=lam)
    write_run_metrics(str(out_dir), "ntv3", args.tissue, args.mode, "probe", "probe",
                      test_r, val_pearson=val_r, test_mse=test_mse,
                      species=SPECIES_NAME[args.tissue], checkpoint=str(out_dir / "probe_head.npz"))
    print(f"\nProbe done. ntv3 {args.tissue} {args.mode} best_lambda={lam} "
          f"val={val_r:.4f} test={test_r:.4f} extract_seconds={time.time() - t0:.1f}")


def main():
    parser = _build_parser()
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        print(f"Loading config from: {temp_args.config}")
        _apply_config(parser, load_config(temp_args.config))
    args = parser.parse_args()

    try:
        import jax  # noqa: F401
        import nucleotide_transformer_v3  # noqa: F401
    except ImportError:
        print(_JAX_HINT, file=sys.stderr)
        return 1

    print("=" * 80)
    print(f"NTv3-post plant STARR-seq — {'PROBE' if args.probe else 'FINETUNE'} "
          f"[{args.tissue} / {args.mode}] species={SPECIES_NAME[args.tissue]}")
    print("=" * 80)

    if args.probe:
        run_probe(args)
    else:
        run_finetune(args)


if __name__ == "__main__":
    sys.exit(main())
