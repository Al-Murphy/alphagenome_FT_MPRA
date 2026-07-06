"""Finetune / probe PlantCAD2 on plant STARR-seq (Jores 2021).

PlantCAD2 (a Caduceus/Mamba2 plant genome LM) is loaded frozen; an attention-pool
MPRA head is trained on top (two-stage: frozen head, then unfreeze the backbone at
a low LR). ``--probe`` runs the cache-once mean-pool + ridge linear probe.

This runner is NOT part of the repo's default ``uv`` environment: PlantCAD2 needs
the torch + mamba-ssm (causal-conv1d) stack. Install that separately, then run:

    python scripts/finetune_plantcad2_plant_starrseq.py \\
        --config configs/plant_starrseq_plantcad2_leaf.json --mode combined
    python scripts/finetune_plantcad2_plant_starrseq.py \\
        --config configs/plant_starrseq_plantcad2_leaf.json --mode combined --probe

Numbers from the reference sweep (see results/plant_starrseq/reference/): leaf
combined finetune 0.8927 / probe 0.7038; proto combined finetune 0.8481.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from alphagenome_ft_mpra.plant_starrseq_utils import write_run_metrics

_TORCH_HINT = (
    "PlantCAD2 requires the torch + transformers + mamba-ssm stack, which is not "
    "in this repo's default environment. Create the 'plantcad2' env (torch 2.6 + "
    "causal-conv1d + mamba-ssm) and re-run. See scripts/README.md."
)

HF_ORG = "kuleshov-group"
ENCODER_DIM = 1024


def load_config(path):
    with open(path) as f:
        return json.load(f)


def _load_backbone(model_name, weights_dir, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        f"{HF_ORG}/{model_name}", trust_remote_code=True, cache_dir=weights_dir,
    )
    model = AutoModel.from_pretrained(
        f"{HF_ORG}/{model_name}", trust_remote_code=True, cache_dir=weights_dir,
    )
    model.eval().to(device)
    return model, tokenizer


def _embed(model, input_ids):
    """RC-invariant PlantCAD2 features: 0.5 * (fwd + flip(rc)) from the 2048-ch RCPS output."""
    import torch

    out = model(input_ids=input_ids)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    d = hidden.shape[-1] // 2
    fwd = hidden[..., :d]
    rc = torch.flip(hidden[..., d:], dims=[1])
    return 0.5 * (fwd + rc)  # (B, L, 1024)


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--tissue", type=str, default="leaf", choices=["leaf", "proto"])
    p.add_argument("--mode", type=str, default="combined",
                   choices=["promoter_only", "enhancer", "combined"])
    p.add_argument("--data_path", type=str, default="./data/jores_plant_starrseq")
    p.add_argument("--model_name", type=str, default="PlantCAD2-Medium-l48-d1024")
    p.add_argument("--weights_dir", type=str, default="./data/plantcad2_weights")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_shift", type=int, default=25)
    p.add_argument("--head_pooling", type=str, default="attention", choices=["attention", "mean"])
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--do", type=float, default=0.2)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--second_stage_lr", type=float, default=1e-5)
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
                        batch_size=d.get("batch_size", 32), max_shift=d.get("max_shift", 25))
    m = cfg.get("model_params", {})
    parser.set_defaults(model_name=m.get("model_name", "PlantCAD2-Medium-l48-d1024"),
                        weights_dir=m.get("weights_dir", "./data/plantcad2_weights"),
                        head_pooling=m.get("head_pooling", "attention"),
                        hidden_size=m.get("hidden_size", 1024),
                        n_heads=m.get("n_heads", 4), do=m.get("do", 0.2))
    t = cfg.get("training", {})
    parser.set_defaults(learning_rate=t.get("learning_rate", 5e-4),
                        weight_decay=t.get("weight_decay", 0.0),
                        num_epochs=t.get("num_epochs", 100),
                        early_stopping_patience=t.get("early_stopping_patience", 5))
    ts = cfg.get("two_stage", {})
    parser.set_defaults(second_stage_lr=ts.get("second_stage_lr", 1e-5),
                        second_stage_epochs=ts.get("second_stage_epochs", 50))
    if not ts.get("enabled", True):
        parser.set_defaults(no_second_stage=True)


def _make_loaders(args, tokenizer):
    from alphagenome_ft_mpra.plant_torch import (
        create_plant_dataloaders, make_tokenizer_collate_fn,
    )
    collate = make_tokenizer_collate_fn(tokenizer)
    return create_plant_dataloaders(
        args.data_path, args.tissue, args.mode, args.batch_size, collate,
        random_shift=True, reverse_complement=True, max_shift=args.max_shift,
        num_workers=2, seed=args.seed,
    )


def run_finetune(args):
    import torch
    from torch.optim import AdamW
    from alphagenome_ft_mpra.plant_torch import MPRAHead, MeanPoolHead

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, tokenizer = _load_backbone(args.model_name, args.weights_dir, device)
    for prm in backbone.parameters():
        prm.requires_grad = False

    head = (MPRAHead(ENCODER_DIM, args.hidden_size, args.n_heads, args.do)
            if args.head_pooling == "attention"
            else MeanPoolHead(ENCODER_DIM, args.hidden_size, args.do)).to(device)

    train_loader, val_loader, test_loader = _make_loaders(args, tokenizer)

    def evaluate(loader):
        head.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for input_ids, y in loader:
                emb = _embed(backbone, input_ids.to(device))
                preds.append(head(emb).cpu().numpy())
                tgts.append(y.numpy())
        from alphagenome_ft_mpra.plant_torch import pearson
        return pearson(np.concatenate(preds), np.concatenate(tgts))

    def train_epochs(opt, n_epochs, train_backbone):
        best_val, best_state, patience = -1.0, None, 0
        for _ in range(n_epochs):
            head.train()
            backbone.train() if train_backbone else backbone.eval()
            for input_ids, y in train_loader:
                opt.zero_grad()
                ctx = torch.enable_grad() if train_backbone else torch.no_grad()
                with ctx:
                    emb = _embed(backbone, input_ids.to(device))
                if not train_backbone:
                    emb = emb.detach()
                pred = head(emb)
                loss = torch.nn.functional.mse_loss(pred, y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(head.parameters()) + (list(backbone.parameters()) if train_backbone else []),
                    1.0)
                opt.step()
            vr = evaluate(val_loader)
            if vr > best_val:
                best_val, patience = vr, 0
                best_state = ({k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                              {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()} if train_backbone else None)
            else:
                patience += 1
                if patience >= args.early_stopping_patience:
                    break
        return best_val, best_state

    # stage 1: head only
    opt1 = AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val, best_state = train_epochs(opt1, args.num_epochs, train_backbone=False)
    if best_state:
        head.load_state_dict(best_state[0])

    stage = "stage1"
    if not args.no_second_stage:
        for prm in backbone.parameters():
            prm.requires_grad = True
        opt2 = AdamW(list(head.parameters()) + list(backbone.parameters()),
                     lr=args.second_stage_lr, weight_decay=args.weight_decay)
        val2, state2 = train_epochs(opt2, args.second_stage_epochs, train_backbone=True)
        if val2 >= best_val and state2:
            head.load_state_dict(state2[0])
            backbone.load_state_dict(state2[1])
            best_val = val2
        stage = "stage2"

    test_r = evaluate(test_loader)
    out_dir = Path(args.results_dir) / "plantcad2" / args.tissue / args.mode / "finetune"
    write_run_metrics(str(out_dir), "plantcad2", args.tissue, args.mode, "finetune", stage,
                      test_r, val_pearson=best_val)
    print(f"\nFinetune done. plantcad2 {args.tissue} {args.mode} "
          f"val={best_val:.4f} test={test_r:.4f}")


def run_probe(args):
    import torch
    from alphagenome_ft_mpra.plant_torch import select_ridge, ridge_predict, pearson

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, tokenizer = _load_backbone(args.model_name, args.weights_dir, device)
    for prm in backbone.parameters():
        prm.requires_grad = False

    # deterministic features: no augmentation
    args_no_aug = argparse.Namespace(**vars(args))
    from alphagenome_ft_mpra.plant_torch import (
        create_plant_dataloaders, make_tokenizer_collate_fn,
    )
    collate = make_tokenizer_collate_fn(tokenizer)
    train_loader, val_loader, test_loader = create_plant_dataloaders(
        args.data_path, args.tissue, args.mode, args.batch_size, collate,
        random_shift=False, reverse_complement=False, num_workers=2, seed=args.seed,
    )

    @torch.no_grad()
    def extract(loader):
        feats, tgts = [], []
        for input_ids, y in loader:
            emb = _embed(backbone, input_ids.to(device))
            feats.append(emb.float().mean(dim=1).cpu().numpy())
            tgts.append(y.numpy())
        return np.concatenate(feats), np.concatenate(tgts)

    t0 = time.time()
    Xtr, ytr = extract(train_loader)
    Xva, yva = extract(val_loader)
    Xte, yte = extract(test_loader)

    val_r, lam, w, xb, yb = select_ridge(Xtr, ytr, Xva, yva)
    pred_te = ridge_predict(Xte, w, xb, yb)
    test_r = pearson(pred_te, yte)
    test_mse = float(np.mean((pred_te - yte) ** 2))

    out_dir = Path(args.results_dir) / "plantcad2" / args.tissue / args.mode / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "probe_head.npz", w=w, xb=xb, yb=yb, lam=lam)
    write_run_metrics(str(out_dir), "plantcad2", args.tissue, args.mode, "probe", "probe",
                      test_r, val_pearson=val_r, test_mse=test_mse,
                      checkpoint=str(out_dir / "probe_head.npz"))
    print(f"\nProbe done. plantcad2 {args.tissue} {args.mode} best_lambda={lam} "
          f"val={val_r:.4f} test={test_r:.4f} extract_seconds={time.time() - t0:.1f}")


def main():
    parser = _build_parser()
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        print(f"Loading config from: {temp_args.config}")
        _apply_config(parser, load_config(temp_args.config))
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print(_TORCH_HINT, file=sys.stderr)
        return 1

    print("=" * 80)
    print(f"PlantCAD2 plant STARR-seq — {'PROBE' if args.probe else 'FINETUNE'} "
          f"[{args.tissue} / {args.mode}]")
    print("=" * 80)

    if args.probe:
        run_probe(args)
    else:
        run_finetune(args)


if __name__ == "__main__":
    sys.exit(main())
