"""Run inference for EACH ensemble member separately + save predictions per member.
Aggregator can then pick the best non-degenerate member per (cell, seed).
"""
import argparse, sys, numpy as np, pandas as pd, torch
from pathlib import Path
REPO = Path('/grid/wsbs/home_norepl/christen/ALBench-S2F')
sys.path.insert(0, str(REPO))
from data.utils import one_hot_encode
from models.dream_rnn import create_dream_rnn

def encode(seqs):
    out = []
    for s in seqs:
        s = s.upper()
        if len(s) < 200: pad=200-len(s); s='N'*(pad//2)+s+'N'*(pad-pad//2)
        elif len(s) > 200: start=(len(s)-200)//2; s=s[start:start+200]
        oh = one_hot_encode(s, add_singleton_channel=False)
        rc = np.zeros((1, oh.shape[1]), dtype=np.float32)
        out.append(np.concatenate([oh, rc], axis=0))
    return np.stack(out)

def predict(model, seqs, device, batch_size=512):
    xt = torch.from_numpy(encode(seqs)).float()
    preds = []
    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            b = xt[i:i+batch_size].to(device)
            preds.append(model.predict(b, use_reverse_complement=True).cpu().float().numpy().reshape(-1))
    return np.concatenate(preds)

ap = argparse.ArgumentParser()
ap.add_argument('--cell', required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out-dir', required=True)
args = ap.parse_args()
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda')

ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
msds = ck['model_state_dicts']
print(f'Loaded {len(msds)} ensemble members')

df = pd.read_csv(REPO / 'data/k562/test_sets/test_chr7_13_all.tsv', sep='\t')
snv = pd.read_csv(REPO / 'data/k562/test_sets/test_snv_pairs.tsv', sep='\t')
ood = pd.read_csv(REPO / f'data/{args.cell}/test_sets/test_ood_designed_{args.cell}.tsv', sep='\t')

for i, sd in enumerate(msds):
    m = create_dream_rnn(input_channels=5, sequence_length=200, task_mode='k562',
                          hidden_dim=320, cnn_filters=160, dropout_cnn=0.1, dropout_lstm=0.1)
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    print(f'== Member {i} ==')
    in_dist_pred = predict(m, df['sequence'].tolist(), device)
    snv_ref_pred = predict(m, snv['sequence_ref'].tolist(), device)
    snv_alt_pred = predict(m, snv['sequence_alt'].tolist(), device)
    ood_pred = predict(m, ood['sequence'].tolist(), device)
    np.savez_compressed(out_dir / f'member{i}_predictions.npz',
        in_dist_pred=in_dist_pred, snv_ref_pred=snv_ref_pred,
        snv_alt_pred=snv_alt_pred, ood_pred=ood_pred)
    del m; torch.cuda.empty_cache()
print('Done')
