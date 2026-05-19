"""Sharded Enformer inference: process only a subset of test seqs per job, save predictions npz.
The aggregator combines shards by sequence index and computes Pearson r.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

REPO = Path('/grid/wsbs/home_norepl/christen/ALBench-S2F')
sys.path.insert(0, str(REPO))
from experiments.train_foundation_stage2 import _forward_enformer, _predict_test_sequences

CELL_UC = {'k562':'K562','hepg2':'HepG2','sknsh':'SKNSH'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=['k562','hepg2','sknsh'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--shard-idx', type=int, required=True)
    ap.add_argument('--total-shards', type=int, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    from enformer_pytorch import Enformer
    if not hasattr(Enformer, 'all_tied_weights_keys'):
        Enformer.all_tied_weights_keys = {}
    encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
    enc_path = Path(args.ckpt).parent / 'best_encoder.pt'
    if enc_path.exists():
        es = torch.load(enc_path, map_location='cpu', weights_only=False)
        if isinstance(es, dict) and 'model_state_dict' in es:
            es = es['model_state_dict']
        encoder.load_state_dict(es, strict=False)
        print(f'Encoder loaded from {enc_path}')
    else:
        print(f'Using pretrained encoder (no best_encoder.pt — probing mode)')
    encoder.eval().to(device)
    for p in encoder.parameters(): p.requires_grad = False

    from experiments.train_foundation_cached import MLPHead
    head = MLPHead(3072, hidden_dim=512, dropout=0.1)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    head.load_state_dict(ckpt['model_state_dict'])
    head.to(device).eval()

    def pred(seqs):
        return _predict_test_sequences(encoder, head, _forward_enformer, seqs, device,
                                        batch_size=args.batch_size, amp_dtype=torch.bfloat16, use_amp=True)

    def shard_indices(n_total):
        per = (n_total + args.total_shards - 1) // args.total_shards
        start = args.shard_idx * per
        end = min(start + per, n_total)
        return start, end

    # Full chr 7+13 — sharded
    df = pd.read_csv(REPO / 'data/k562/test_sets/test_chr7_13_all.tsv', sep='\t')
    s, e = shard_indices(len(df))
    print(f'[1/2] Full chr 7+13 shard {args.shard_idx}/{args.total_shards}: rows [{s}:{e}] (n={e-s})')
    sub_df = df.iloc[s:e]
    sub_preds = pred(sub_df['sequence'].tolist())
    np.savez_compressed(out_dir / f'shard{args.shard_idx}_full_chr.npz',
                        preds=sub_preds, start=s, end=e)
    print(f'  Saved {out_dir}/shard{args.shard_idx}_full_chr.npz')

    # Designed/OOD — sharded
    ood_path = REPO / f'data/{args.cell}/test_sets/test_ood_designed_{args.cell}.tsv'
    if ood_path.exists():
        ood = pd.read_csv(ood_path, sep='\t')
        s, e = shard_indices(len(ood))
        print(f'[2/2] Designed shard {args.shard_idx}/{args.total_shards}: rows [{s}:{e}] (n={e-s})')
        sub_ood = ood.iloc[s:e]
        sub_preds = pred(sub_ood['sequence'].tolist())
        np.savez_compressed(out_dir / f'shard{args.shard_idx}_designed.npz',
                            preds=sub_preds, start=s, end=e)
        print(f'  Saved {out_dir}/shard{args.shard_idx}_designed.npz')


if __name__ == '__main__':
    main()
