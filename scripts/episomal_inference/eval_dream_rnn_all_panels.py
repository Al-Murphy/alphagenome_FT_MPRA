"""DREAM-RNN inference on all panels — handles ensemble ckpts (avg across N models)."""
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

REPO = Path('/grid/wsbs/home_norepl/christen/ALBench-S2F')
sys.path.insert(0, str(REPO))
from data.utils import one_hot_encode
from models.dream_rnn import create_dream_rnn

CELL_UC = {'k562':'K562','hepg2':'HepG2','sknsh':'SKNSH'}


def encode(seqs):
    out = []
    for s in seqs:
        s = s.upper()
        if len(s) < 200:
            pad = 200 - len(s); s = 'N'*(pad//2) + s + 'N'*(pad-pad//2)
        elif len(s) > 200:
            start = (len(s)-200)//2; s = s[start:start+200]
        oh = one_hot_encode(s, add_singleton_channel=False)  # (4, 200)
        rc = np.zeros((1, oh.shape[1]), dtype=np.float32)
        out.append(np.concatenate([oh, rc], axis=0))
    return np.stack(out)


def predict_ensemble(models, seqs, device, batch_size=512):
    arr = encode(seqs)
    xt = torch.from_numpy(arr).float()
    member_preds = []
    for m in models:
        preds = []
        with torch.no_grad():
            for i in range(0, len(xt), batch_size):
                b = xt[i:i+batch_size].to(device)
                p = m.predict(b, use_reverse_complement=True)
                preds.append(p.cpu().float().numpy().reshape(-1))
        member_preds.append(np.concatenate(preds))
    return np.mean(member_preds, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=['k562','hepg2','sknsh'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    msds = ck['model_state_dicts'] if isinstance(ck, dict) and 'model_state_dicts' in ck else [ck.get('model_state_dict', ck)]
    print(f'Loaded ensemble of {len(msds)} DREAM-RNN models')

    models = []
    for sd in msds:
        m = create_dream_rnn(input_channels=5, sequence_length=200, task_mode='k562',
                              hidden_dim=320, cnn_filters=160, dropout_cnn=0.1, dropout_lstm=0.1)
        m.load_state_dict(sd, strict=True)
        m.to(device).eval()
        models.append(m)

    print('[1/3] full chr 7+13...')
    df = pd.read_csv(REPO / 'data/k562/test_sets/test_chr7_13_all.tsv', sep='\t')
    in_dist_pred = predict_ensemble(models, df['sequence'].tolist(), device)

    print('[2/3] SNV pairs 45k...')
    snv = pd.read_csv(REPO / 'data/k562/test_sets/test_snv_pairs.tsv', sep='\t')
    snv_ref_pred = predict_ensemble(models, snv['sequence_ref'].tolist(), device)
    snv_alt_pred = predict_ensemble(models, snv['sequence_alt'].tolist(), device)

    print('[3/3] OOD designed...')
    ood = pd.read_csv(REPO / f'data/{args.cell}/test_sets/test_ood_designed_{args.cell}.tsv', sep='\t')
    ood_pred = predict_ensemble(models, ood['sequence'].tolist(), device)

    np.savez_compressed(out_dir / 'test_predictions.npz',
        in_dist_pred=in_dist_pred, snv_ref_pred=snv_ref_pred,
        snv_alt_pred=snv_alt_pred, ood_pred=ood_pred)
    print(f'Saved {out_dir}/test_predictions.npz')


if __name__ == '__main__':
    main()
