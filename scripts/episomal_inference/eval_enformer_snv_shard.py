"""Enformer 45k SNV-pair (ref + alt) inference — sharded."""
import argparse, sys, numpy as np, pandas as pd, torch
from pathlib import Path
REPO = Path('/grid/wsbs/home_norepl/christen/ALBench-S2F')
sys.path.insert(0, str(REPO))
from experiments.train_foundation_stage2 import _forward_enformer, _predict_test_sequences

ap = argparse.ArgumentParser()
ap.add_argument('--cell', required=True, choices=['k562','hepg2','sknsh'])
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--shard-idx', type=int, required=True)
ap.add_argument('--total-shards', type=int, required=True)
args = ap.parse_args()
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda')

from enformer_pytorch import Enformer
if not hasattr(Enformer, 'all_tied_weights_keys'): Enformer.all_tied_weights_keys = {}
encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
enc_path = Path(args.ckpt).parent / 'best_encoder.pt'
if enc_path.exists():
    es = torch.load(enc_path, map_location='cpu', weights_only=False)
    if isinstance(es, dict) and 'model_state_dict' in es: es = es['model_state_dict']
    encoder.load_state_dict(es, strict=False)
encoder.eval().to(device)
for p in encoder.parameters(): p.requires_grad = False
from experiments.train_foundation_cached import MLPHead
head = MLPHead(3072, hidden_dim=512, dropout=0.1)
ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
head.load_state_dict(ckpt['model_state_dict'])
head.to(device).eval()

def pred(seqs):
    return _predict_test_sequences(encoder, head, _forward_enformer, seqs, device,
                                    batch_size=16, amp_dtype=torch.bfloat16, use_amp=True)

snv = pd.read_csv(REPO / 'data/k562/test_sets/test_snv_pairs.tsv', sep='\t')
per = (len(snv) + args.total_shards - 1) // args.total_shards
s = args.shard_idx * per; e = min(s + per, len(snv))
print(f'SNV shard {args.shard_idx}/{args.total_shards}: rows [{s}:{e}]')
sub = snv.iloc[s:e]
ref_pred = pred(sub['sequence_ref'].tolist())
alt_pred = pred(sub['sequence_alt'].tolist())
np.savez_compressed(out_dir / f'shard{args.shard_idx}_snv.npz',
                    ref_pred=ref_pred, alt_pred=alt_pred, start=s, end=e)
print(f'Saved')
