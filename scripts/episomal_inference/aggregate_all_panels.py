"""Aggregate all 6 panel types for every model with saved predictions.

Panels computed (per cell × seed where available):
  - reference_32k    : 32k unique ref alleles  (chr 7+13 allele=R)
  - alt_32k          : 32k unique alt alleles  (chr 7+13 allele=A)
  - reference_45k    : 45k SNV-pair ref allele (test_snv_pairs.tsv ref)
  - alt_45k          : 45k SNV-pair alt allele (test_snv_pairs.tsv alt)
  - delta_45k        : 45k SNV-pair Δ          (alt − ref)
  - designed_ood     : 22k OOD designed
  - snv_filt2962     : 2962 chr-filtered hashfrag SNV pair Δ

Output: prints rows in (model, cell, seed, panel, pearson_r, n) → also writes CSV.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

REPO = Path('/grid/wsbs/home_norepl/christen/ALBench-S2F')
CELL_UC = {'k562': 'K562', 'hepg2': 'HepG2', 'sknsh': 'SKNSH'}

# Load test sets once
chr_all = pd.read_csv(REPO / 'data/k562/test_sets/test_chr7_13_all.tsv', sep='\t')
ref_mask = chr_all['allele'].values == 'R'
alt_mask = chr_all['allele'].values == 'A'

snv_pairs_k562 = pd.read_csv(REPO / 'data/k562/test_sets/test_snv_pairs.tsv', sep='\t')
hashfrag_pairs = pd.read_csv(REPO / 'data/k562/test_sets/test_snv_pairs_hashfrag.tsv', sep='\t')

# Mono-allelic mask: drop rows where either ref or alt ID has an Alt_ tag
# (those rows are multi-allelic context expansions that artificially inflate Pearson).
# Per COMPREHENSIVE_STATUS: 45k full inflates delta by ~0.20 over the 30k mono subset.
snv_mono_mask = (
    ~snv_pairs_k562['IDs_ref'].astype(str).str.contains('Alt_', na=False)
    & ~snv_pairs_k562['IDs_alt'].astype(str).str.contains('Alt_', na=False)
).to_numpy()
print(f'SNV mono-allelic subset: {snv_mono_mask.sum()} / {len(snv_pairs_k562)} pairs')

ood = {c: pd.read_csv(REPO / f'data/{c}/test_sets/test_ood_designed_{c}.tsv', sep='\t')
       for c in ('k562', 'hepg2', 'sknsh')}
ood_col = {'k562': 'K562_log2FC', 'hepg2': 'HepG2_log2FC', 'sknsh': 'SKNSH_log2FC'}


def safe_r(p, t):
    p = np.asarray(p, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    return float(pearsonr(p[m], t[m])[0]) if m.sum() > 3 else float('nan')


# Get SNV-pair labels for each cell. The K562 file has K562_log2FC_ref/alt cols.
# For HepG2/SKNSH we need to merge chr_all to retrieve those cells' labels.
def get_snv_labels(cell):
    """Return (ref_label, alt_label) arrays for the 45k SNV pairs in cell-specific labels."""
    if cell == 'k562':
        ref_lab = snv_pairs_k562['K562_log2FC_ref'].to_numpy(dtype=np.float32)
        alt_lab = snv_pairs_k562['K562_log2FC_alt'].to_numpy(dtype=np.float32)
        return ref_lab, alt_lab
    col = f'{CELL_UC[cell]}_log2FC'
    ref_map = chr_all[['IDs', col]].rename(columns={'IDs': 'IDs_ref', col: 'lref'})
    alt_map = chr_all[['IDs', col]].rename(columns={'IDs': 'IDs_alt', col: 'lalt'})
    merged = snv_pairs_k562[['IDs_ref', 'IDs_alt']].merge(ref_map, on='IDs_ref', how='left').merge(alt_map, on='IDs_alt', how='left')
    return merged['lref'].to_numpy(dtype=np.float32), merged['lalt'].to_numpy(dtype=np.float32)


def chr_labels(cell):
    col = f'{CELL_UC[cell]}_log2FC'
    return chr_all[col].to_numpy(dtype=np.float32)


def compute_panels_from_full_chr(preds_full, preds_snv_ref, preds_snv_alt, preds_ood, cell):
    """Generic panel computation from saved AG-style predictions.

    preds_full : (66712,) preds aligned to chr_all rows
    preds_snv_ref/_alt : (45543,) preds aligned to snv_pairs rows
    preds_ood : (22k+,) preds aligned to ood[cell]
    """
    labels = chr_labels(cell)
    out = {}
    out['reference_32k'] = safe_r(preds_full[ref_mask], labels[ref_mask])
    out['alt_32k'] = safe_r(preds_full[alt_mask], labels[alt_mask])
    snv_ref_lbl, snv_alt_lbl = get_snv_labels(cell)
    # 45k full SNV-pair panels (inflated by multi-allelic context expansion)
    out['reference_45k'] = safe_r(preds_snv_ref, snv_ref_lbl)
    out['alt_45k'] = safe_r(preds_snv_alt, snv_alt_lbl)
    out['delta_45k'] = safe_r(preds_snv_alt - preds_snv_ref, snv_alt_lbl - snv_ref_lbl)
    # Mono-allelic 30k subset (no Alt_ tag) — preferred for headline numbers
    m = snv_mono_mask
    out['reference_30k_mono'] = safe_r(preds_snv_ref[m], snv_ref_lbl[m])
    out['alt_30k_mono'] = safe_r(preds_snv_alt[m], snv_alt_lbl[m])
    out['delta_30k_mono'] = safe_r((preds_snv_alt - preds_snv_ref)[m], (snv_alt_lbl - snv_ref_lbl)[m])
    if preds_ood is not None:
        ood_lbl = ood[cell][ood_col[cell]].to_numpy(dtype=np.float32)
        out['designed_ood'] = safe_r(preds_ood, ood_lbl)
    return out


# ── Walk models ─────────────────────────────────────────────────────────────
rows = []

# AG S1 best-HP (3 seeds × 3 cells; has full-chr predictions)
print('AG S1 best HP...')
for cell in ('k562', 'hepg2', 'sknsh'):
    for seed in (42, 1042, 2042):
        npz = REPO / f'outputs/chr_split_v2/{cell}/ag_s1_best_hp/seed_{seed}/genomic/n750000/hp0/seed{seed}/test_predictions.npz'
        if not npz.exists(): continue
        d = np.load(npz, allow_pickle=True)
        ood_preds = d['ood_pred'] if 'ood_pred' in d.keys() else None
        panels = compute_panels_from_full_chr(d['in_dist_pred'], d['snv_ref_pred'], d['snv_alt_pred'], ood_preds, cell)
        for k, v in panels.items():
            rows.append({'model': 'AG MPRA (Probing)', 'cell': cell, 'seed': seed, 'panel': k, 'pearson_r': v})

# AG S2 separate (best HP K562 + best S1 HepG2/SKNSH)
print('AG S2 separate...')
for cell in ('k562', 'hepg2', 'sknsh'):
    base_dir = 'ag_s2_best_hp' if cell == 'k562' else 'ag_s2_from_best_s1'
    for seed in (42, 1042, 2042):
        npz = REPO / f'outputs/chr_split_v2/{cell}/{base_dir}/seed_{seed}/genomic/n600000/hp0/seed{seed}/test_predictions.npz'
        if not npz.exists(): continue
        d = np.load(npz, allow_pickle=True)
        ood_preds = d['ood_pred'] if 'ood_pred' in d.keys() else None
        panels = compute_panels_from_full_chr(d['in_dist_pred'], d['snv_ref_pred'], d['snv_alt_pred'], ood_preds, cell)
        for k, v in panels.items():
            rows.append({'model': 'AG MPRA (Fine-tuned-separate)', 'cell': cell, 'seed': seed, 'panel': k, 'pearson_r': v})

# Malinois pub_cosine (3 seeds × 3 cells) — from proper_eval with rich npz
print('Malinois pub_cosine...')
for cell in ('k562', 'hepg2', 'sknsh'):
    for seed in (42, 1042, 2042):
        npz = REPO / f'outputs/proper_eval/malinois_pub_cosine/{cell}/seed{seed}/test_predictions_proper.npz'
        if not npz.exists(): continue
        d = np.load(npz, allow_pickle=True)
        # 32k panels from 'ref_pred'/'alt_pred' (already split)
        labels = chr_labels(cell)
        # ref_pred is on 32831 rows = ref-only subset
        # Need to align — Malinois proper_eval saved ref_pred/ref_true separately
        ref_panel = safe_r(d['ref_pred'], d['ref_true'])
        alt_panel = safe_r(d['alt_pred'], d['alt_true'])
        snv_ref_lbl, snv_alt_lbl = get_snv_labels(cell)
        snv_ref_45k = safe_r(d['snv_ref_full_pred'], snv_ref_lbl)
        snv_alt_45k = safe_r(d['snv_alt_full_pred'], snv_alt_lbl)
        delta_45k = safe_r(d['snv_alt_full_pred'] - d['snv_ref_full_pred'], snv_alt_lbl - snv_ref_lbl)
        # Mono-allelic 30k subset
        m = snv_mono_mask
        snv_ref_mono = safe_r(d['snv_ref_full_pred'][m], snv_ref_lbl[m])
        snv_alt_mono = safe_r(d['snv_alt_full_pred'][m], snv_alt_lbl[m])
        delta_mono = safe_r((d['snv_alt_full_pred'] - d['snv_ref_full_pred'])[m], (snv_alt_lbl - snv_ref_lbl)[m])
        ood_pearson = safe_r(d['ood_pred'], d['ood_true'])
        panels = {'reference_32k': ref_panel, 'alt_32k': alt_panel,
                  'reference_45k': snv_ref_45k, 'alt_45k': snv_alt_45k,
                  'delta_45k': delta_45k,
                  'reference_30k_mono': snv_ref_mono, 'alt_30k_mono': snv_alt_mono,
                  'delta_30k_mono': delta_mono,
                  'designed_ood': ood_pearson}
        for k, v in panels.items():
            rows.append({'model': 'Malinois', 'cell': cell, 'seed': seed, 'panel': k, 'pearson_r': v})

# AG S2 joint multi-head (3 seeds × 3 cells; has 31k in_dist + 45k SNV per cell + per-cell OOD)
print('AG S2 joint...')
for seed in (42, 1042, 2042):
    npz = REPO / f'outputs/chr_split_v2/joint_multitask/ag_s2/seed_{seed}/test_predictions.npz'
    if not npz.exists(): continue
    d = np.load(npz, allow_pickle=True)
    for cell in ('k562', 'hepg2', 'sknsh'):
        # in_dist is hashfrag-filtered (31435) so we cannot compute 32k Ref/Alt from it
        snv_ref_lbl, snv_alt_lbl = get_snv_labels(cell)
        snv_ref_45k = safe_r(d[f'snv_ref_pred_{cell}'], snv_ref_lbl)
        snv_alt_45k = safe_r(d[f'snv_alt_pred_{cell}'], snv_alt_lbl)
        delta_45k = safe_r(d[f'snv_alt_pred_{cell}'] - d[f'snv_ref_pred_{cell}'], snv_alt_lbl - snv_ref_lbl)
        m = snv_mono_mask
        snv_ref_mono = safe_r(d[f'snv_ref_pred_{cell}'][m], snv_ref_lbl[m])
        snv_alt_mono = safe_r(d[f'snv_alt_pred_{cell}'][m], snv_alt_lbl[m])
        delta_mono = safe_r((d[f'snv_alt_pred_{cell}'] - d[f'snv_ref_pred_{cell}'])[m], (snv_alt_lbl - snv_ref_lbl)[m])
        ood_lbl = ood[cell][ood_col[cell]].to_numpy(dtype=np.float32)
        ood_pearson = safe_r(d[f'ood_pred_{cell}'], ood_lbl)
        panels = {'reference_45k': snv_ref_45k, 'alt_45k': snv_alt_45k,
                  'delta_45k': delta_45k,
                  'reference_30k_mono': snv_ref_mono, 'alt_30k_mono': snv_alt_mono,
                  'delta_30k_mono': delta_mono,
                  'designed_ood': ood_pearson}
        for k, v in panels.items():
            rows.append({'model': 'AG MPRA (Fine-tuned-joint)', 'cell': cell, 'seed': seed, 'panel': k, 'pearson_r': v})

# Summary table
import sys
df = pd.DataFrame(rows)
out_csv = REPO / 'outputs/all_panels_aggregated.csv'
df.to_csv(out_csv, index=False)
print(f'\nWrote {out_csv} ({len(df)} rows)')

print('\n=== Cell-averaged Pearson r per (model, panel) ===')
agg = df.groupby(['model', 'panel'])['pearson_r'].agg(['mean', 'std', 'count']).reset_index()
pivot = agg.pivot(index='model', columns='panel', values='mean')
print(pivot.round(4))
