"""AG S2 joint multitask: inference on full chr 7+13 + 45k SNV + 22k OOD.

Loads the Orbax checkpoint, runs RC-averaged prediction per cell head, saves npz.
Mirrors the training script's predict_step but on the full test set instead of hashfrag-filtered.
"""
import argparse, sys, time, numpy as np, pandas as pd
from pathlib import Path

REPO = Path('/grid/wsbs/home_norepl/christen/ALBench-S2F')
sys.path.insert(0, str(REPO))
import jax, jax.numpy as jnp
from experiments.exp1_1_scaling_multitask import _build_encoder, CELL_LINES, JOINT_S2_CONFIG
from alphagenome_ft import create_model_with_heads
from models.alphagenome_heads import register_s2f_head

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--ckpt-dir', required=True, help='Path to best_model/checkpoint directory')
ap.add_argument('--out-dir', required=True)
args = ap.parse_args()
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

# Register 3 heads (must match training)
head_names = {cl: f's2f_joint_s2_{cl}_{args.seed}' for cl in CELL_LINES}
for cl in CELL_LINES:
    register_s2f_head(head_name=head_names[cl], arch='boda-flatten-512-512', task_mode='k562',
                      num_tracks=1, dropout_rate=JOINT_S2_CONFIG['dropout'])

print('Building model...')
weights_path = '/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1'
model = create_model_with_heads('all_folds', heads=[head_names[cl] for cl in CELL_LINES],
                                 checkpoint_path=weights_path, use_encoder_output=True, detach_backbone=False)

print(f'Loading params from {args.ckpt_dir}...')
import orbax.checkpoint as ocp
checkpointer = ocp.StandardCheckpointer()
loaded = checkpointer.restore(Path(args.ckpt_dir).resolve())
# Loaded params have same structure as model._params
best_params = loaded

requested_outputs = [head_names[cl] for cl in CELL_LINES]

@jax.jit
def predict_step(params, seqs):
    preds_dict = model._predict(
        params, model._state, seqs,
        jnp.zeros(len(seqs), dtype=jnp.int32),
        requested_outputs=requested_outputs,
        negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
        strand_reindexing=None)
    out = {}
    for cl in CELL_LINES:
        p = preds_dict[head_names[cl]]
        out[cl] = jnp.squeeze(p, axis=-1) if p.ndim > 1 else p
    return out

def _rc_onehot(x):
    rc = x[:, ::-1, :]
    rc = rc.at[:, :, :4].set(rc[:, :, [3, 2, 1, 0]])
    return rc

_encode_one = _build_encoder()

def predict_seqs(seqs, batch_size=32):
    onehot = np.stack([_encode_one(s) for s in seqs])
    out = {cl: [] for cl in CELL_LINES}
    for i in range(0, len(onehot), batch_size):
        b = jnp.array(onehot[i:i+batch_size])
        p_fwd = predict_step(best_params, b)
        p_rc = predict_step(best_params, _rc_onehot(b))
        for cl in CELL_LINES:
            avg = (np.array(p_fwd[cl]) + np.array(p_rc[cl])) / 2.0
            out[cl].append(avg.reshape(-1))
    return {cl: np.concatenate(out[cl]) for cl in CELL_LINES}

# Test sets
print('[1/3] Full chr 7+13 (66k)...')
t0 = time.time()
chr_all = pd.read_csv(REPO / 'data/k562/test_sets/test_chr7_13_all.tsv', sep='\t')
chr_preds = predict_seqs(chr_all['sequence'].tolist())
print(f'  done in {time.time()-t0:.1f}s')

print('[2/3] 45k SNV pairs...')
t0 = time.time()
snv = pd.read_csv(REPO / 'data/k562/test_sets/test_snv_pairs.tsv', sep='\t')
snv_ref_preds = predict_seqs(snv['sequence_ref'].tolist())
snv_alt_preds = predict_seqs(snv['sequence_alt'].tolist())
print(f'  done in {time.time()-t0:.1f}s')

print('[3/3] 22k OOD per cell...')
ood_preds = {}
for cl in CELL_LINES:
    t0 = time.time()
    ood = pd.read_csv(REPO / f'data/{cl}/test_sets/test_ood_designed_{cl}.tsv', sep='\t')
    p_dict = predict_seqs(ood['sequence'].tolist())
    ood_preds[cl] = p_dict[cl]  # use only the matching cell's prediction
    print(f'  {cl} done in {time.time()-t0:.1f}s')

# Save
to_save = {}
for cl in CELL_LINES:
    to_save[f'in_dist_pred_{cl}'] = chr_preds[cl]
    to_save[f'snv_ref_pred_{cl}'] = snv_ref_preds[cl]
    to_save[f'snv_alt_pred_{cl}'] = snv_alt_preds[cl]
    to_save[f'ood_pred_{cl}'] = ood_preds[cl]
to_save['in_dist_sequences'] = np.array(chr_all['sequence'].tolist(), dtype=object)
np.savez_compressed(out_dir / 'test_predictions_full_chr.npz', **to_save)
print(f'Saved {out_dir}/test_predictions_full_chr.npz')
