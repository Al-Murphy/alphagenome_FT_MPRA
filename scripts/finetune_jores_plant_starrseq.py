"""Train the Jores CNN from scratch on plant STARR-seq (Jores 2021).

The Jores et al. 2021 CNN is a from-scratch baseline (no pretrained backbone), so
it is single-stage and has no linear-probe variant. It takes one-hot 437 bp
constructs (170 bp for ``promoter_only``) and predicts the enrichment scalar.

Requires a PyTorch environment (not tied to the repo's default ``uv`` env, but any
recent torch works — there is no mamba/JAX dependency here).

    python scripts/finetune_jores_plant_starrseq.py \\
        --config configs/plant_starrseq_jores_leaf.json --mode combined

Reference sweep numbers (results/plant_starrseq/reference/): leaf combined 0.8632,
proto combined 0.8651.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from alphagenome_ft_mpra.plant_starrseq_utils import write_run_metrics

_TORCH_HINT = "The Jores CNN runner requires PyTorch. Install torch and re-run."


def load_config(path):
    with open(path) as f:
        return json.load(f)


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--tissue", type=str, default="leaf", choices=["leaf", "proto"])
    p.add_argument("--mode", type=str, default="combined",
                   choices=["promoter_only", "enhancer", "combined"])
    p.add_argument("--data_path", type=str, default="./data/jores_plant_starrseq")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_shift", type=int, default=15)
    p.add_argument("--do", type=float, default=0.15)
    p.add_argument("--learning_rate", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--probe", action="store_true",
                   help="No-op: the Jores CNN is from-scratch and has no probe.")
    p.add_argument("--results_dir", type=str, default="./results/plant_starrseq")
    p.add_argument("--seed", type=int, default=42)
    return p


def _apply_config(parser, cfg):
    for k in ("tissue", "mode"):
        if k in cfg:
            parser.set_defaults(**{k: cfg[k]})
    d = cfg.get("data", {})
    parser.set_defaults(data_path=d.get("data_path", "./data/jores_plant_starrseq"),
                        batch_size=d.get("batch_size", 128), max_shift=d.get("max_shift", 15))
    m = cfg.get("model_params", {})
    parser.set_defaults(do=m.get("do", 0.15))
    t = cfg.get("training", {})
    parser.set_defaults(learning_rate=t.get("learning_rate", 2e-3),
                        weight_decay=t.get("weight_decay", 0.0),
                        num_epochs=t.get("num_epochs", 200),
                        early_stopping_patience=t.get("early_stopping_patience", 5))


def run(args):
    import torch
    from torch.optim import Adam
    from alphagenome_ft_mpra.plant_torch import (
        JoresCNN, create_plant_dataloaders, make_onehot_collate_fn,
        pearson, seq_len_for_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = seq_len_for_mode(args.mode)
    collate = make_onehot_collate_fn(seq_len)

    train_loader, val_loader, test_loader = create_plant_dataloaders(
        args.data_path, args.tissue, args.mode, args.batch_size, collate,
        random_shift=True, reverse_complement=True, max_shift=args.max_shift,
        num_workers=2, seed=args.seed,
    )

    model = JoresCNN(seq_len, dropout=args.do).to(device)
    opt = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def evaluate(loader):
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for x, y in loader:
                preds.append(model(x.to(device)).cpu().numpy())
                tgts.append(y.numpy())
        return pearson(np.concatenate(preds), np.concatenate(tgts))

    best_val, best_state, patience = -1.0, None, 0
    for _ in range(args.num_epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        vr = evaluate(val_loader)
        if vr > best_val:
            best_val, patience = vr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    test_r = evaluate(test_loader)

    out_dir = Path(args.results_dir) / "jores" / args.tissue / args.mode / "finetune"
    write_run_metrics(str(out_dir), "jores", args.tissue, args.mode, "finetune", "single",
                      test_r, val_pearson=best_val)
    print(f"\nTraining done. jores {args.tissue} {args.mode} val={best_val:.4f} test={test_r:.4f}")


def main():
    parser = _build_parser()
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        print(f"Loading config from: {temp_args.config}")
        _apply_config(parser, load_config(temp_args.config))
    args = parser.parse_args()

    if args.probe:
        print("The Jores CNN is trained from scratch and has no linear-probe variant; "
              "nothing to do for --probe.", file=sys.stderr)
        return 0

    try:
        import torch  # noqa: F401
    except ImportError:
        print(_TORCH_HINT, file=sys.stderr)
        return 1

    print("=" * 80)
    print(f"Jores CNN (from scratch) plant STARR-seq [{args.tissue} / {args.mode}]")
    print("=" * 80)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
