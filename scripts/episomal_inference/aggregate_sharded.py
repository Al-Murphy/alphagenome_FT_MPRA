"""Aggregate sharded Enformer + per-member DREAM-RNN predictions on standardized 32k panels."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
REPO = Path('/grid/wsbs/home_norepl/christen/ALBench-S2F')
CELL_UC = {'k562':'K562','hepg2':'HepG2','sknsh':'SKNSH'}
chr_all = pd.read_csv(REPO / 'data/k562/test_sets/test_chr7_13_all.tsv', sep='\t')
ref_mask = chr_all['allele'].values == 'R'
alt_mask = chr_all['allele'].values == 'A'
snv_pairs = pd.read_csv(REPO / 'data/k562/test_sets/test_snv_pairs.tsv', sep='\t')
snv_mono_mask = (
    ~snv_pairs['IDs_ref'].astype(str).str.contains('Alt_', na=False)
    & ~snv_pairs['IDs_alt'].astype(str).str.contains('Alt_', na=False)
).to_numpy()
ood = {c: pd.read_csv(REPO / f'data/{c}/test_sets/test_ood_designed_{c}.tsv', sep='\t')
       for c in ('k562','hepg2','sknsh')}
ood_col = {'k562':'K562_log2FC','hepg2':'HepG2_log2FC','sknsh':'SKNSH_log2FC'}

def safe_r(p, t):
    p=np.asarray(p,dtype=np.float64); t=np.asarray(t,dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    return float(pearsonr(p[m], t[m])[0]) if m.sum() > 3 else float('nan')

def chr_labels(cell): return chr_all[f'{CELL_UC[cell]}_log2FC'].to_numpy(dtype=np.float32)

def reassemble_shards(seed_dir, prefix, expected_total, total_shards=4):
    pieces = []
    for i in range(total_shards):
        f = seed_dir / f'shard{i}_{prefix}.npz'
        if not f.exists(): return None
        d = np.load(f, allow_pickle=True)
        pieces.append((int(d['start']), int(d['end']), d['preds']))
    pieces.sort()
    full = np.full(expected_total, np.nan, dtype=np.float32)
    for s,e,p in pieces:
        full[s:e] = p
    return full

rows = []

# Enformer FT + Probing — sharded
for model_dir, model_name in [('enf_ft_ufall_v2', 'Enf. MPRA (Fine-tuned)'),
                                ('enf_probing', 'Enf. MPRA (Probing)')]:
    base = REPO / f'outputs/proper_eval/{model_dir}'
    if not base.exists(): continue
    for cell_dir in sorted(base.iterdir()):
        cell = cell_dir.name
        for sd in sorted(cell_dir.glob('seed*')):
            seed = int(sd.name.replace('seed',''))
            chr_pred = reassemble_shards(sd, 'full_chr', len(chr_all))
            ood_pred = reassemble_shards(sd, 'designed', len(ood[cell]))
            if chr_pred is None: continue
            labels = chr_labels(cell)
            r_ref32 = safe_r(chr_pred[ref_mask], labels[ref_mask])
            r_alt32 = safe_r(chr_pred[alt_mask], labels[alt_mask])
            r_ood = safe_r(ood_pred, ood[cell][ood_col[cell]].to_numpy(dtype=np.float32)) if ood_pred is not None else float('nan')
            for panel, val in [('reference_32k', r_ref32), ('alt_32k', r_alt32), ('designed_ood', r_ood)]:
                rows.append({'model':model_name,'cell':cell,'seed':seed,'panel':panel,'pearson_r':val})

# DREAM-RNN per-member: pick best member per (cell, seed) based on reference_32k
def get_snv_labels(cell):
    if cell == 'k562':
        return (snv_pairs['K562_log2FC_ref'].to_numpy(dtype=np.float32),
                snv_pairs['K562_log2FC_alt'].to_numpy(dtype=np.float32))
    col = f'{CELL_UC[cell]}_log2FC'
    rmap = chr_all[['IDs', col]].rename(columns={'IDs':'IDs_ref', col:'lref'})
    amap = chr_all[['IDs', col]].rename(columns={'IDs':'IDs_alt', col:'lalt'})
    merged = snv_pairs[['IDs_ref','IDs_alt']].merge(rmap, on='IDs_ref', how='left').merge(amap, on='IDs_alt', how='left')
    return merged['lref'].to_numpy(dtype=np.float32), merged['lalt'].to_numpy(dtype=np.float32)

drnn_base = REPO / 'outputs/all_panels_eval/dream_rnn'
for cell_dir in sorted(drnn_base.iterdir()):
    cell = cell_dir.name
    for sd in sorted(cell_dir.glob('seed*')):
        seed = int(sd.name.replace('seed',''))
        members = sorted(sd.glob('member*_predictions.npz'))
        if not members: continue
        labels = chr_labels(cell)
        # Evaluate each member, pick best by reference_32k Pearson
        best_r, best_m_pred = -np.inf, None
        for m in members:
            d = np.load(m, allow_pickle=True)
            r = safe_r(d['in_dist_pred'][ref_mask], labels[ref_mask])
            if np.isfinite(r) and r > best_r:
                best_r = r; best_m_pred = d
        if best_m_pred is None: continue
        r_ref = safe_r(best_m_pred['in_dist_pred'][ref_mask], labels[ref_mask])
        r_alt = safe_r(best_m_pred['in_dist_pred'][alt_mask], labels[alt_mask])
        ood_lbl = ood[cell][ood_col[cell]].to_numpy(dtype=np.float32)
        r_ood = safe_r(best_m_pred['ood_pred'], ood_lbl)
        sref_lbl, salt_lbl = get_snv_labels(cell)
        d_pred = best_m_pred['snv_alt_pred'] - best_m_pred['snv_ref_pred']
        d_true = salt_lbl - sref_lbl
        m_mask = snv_mono_mask
        r_d_mono = safe_r(d_pred[m_mask], d_true[m_mask])
        for panel, val in [('reference_32k', r_ref), ('alt_32k', r_alt),
                            ('designed_ood', r_ood), ('delta_30k_mono', r_d_mono)]:
            rows.append({'model':'DREAM-RNN','cell':cell,'seed':seed,'panel':panel,'pearson_r':val})

import sys
df = pd.DataFrame(rows)
df.to_csv(REPO / 'outputs/aggregated_32k_panels.csv', index=False)
print(f'Wrote {len(df)} rows')
cm = df.groupby(['model','cell','panel'])['pearson_r'].mean().reset_index()
piv = cm.pivot_table(index='model', columns='panel', values='pearson_r')
print(piv.round(4).to_string())
