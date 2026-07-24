"""Download and load the released AlphaGenome-encoder checkpoints from Hugging Face.

    from alphagenome_ft_mpra.hub import load_pretrained, list_pretrained

    list_pretrained()                                  # what's available
    model = load_pretrained('plant-starrseq-leaf-combined')   # JAX
    model = load_pretrained('mpra_K562')                      # PyTorch (inferred)

Everything the loader needs (which head class, how wide it is, which stage) is read
from the checkpoint's own ``config.json``, so callers do not have to know that the
plant heads are 4096/2048/1024 wide depending on the cell, or that they use unnamed
Haiku modules and therefore need :class:`~alphagenome_ft_mpra.PlantMPRAHead` rather
than :class:`~alphagenome_ft_mpra.EncoderMPRAHead`.

LICENCE. These are fine-tuned derivatives of AlphaGenome. The model parameters, their
outputs, and any derivatives thereof remain subject to Google DeepMind's AlphaGenome
Model Terms (https://deepmind.google.com/science/alphagenome/model-terms) — including
the restriction to non-commercial use. The base parameters are the property of Google
LLC. Only the fine-tuning code in this repository is Apache-2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

HF_REPO_ID = 'Al-Murphy/alphagenome-encoder-ft'

# name -> (framework, path within the HF repo, description)
# 'stage1' is the frozen-encoder result, 'stage2' the fine-tuned one.
CHECKPOINTS: dict[str, dict] = {
    # --- JAX (Haiku) -------------------------------------------------------
    'mpra-K562-optimal': {
        'framework': 'jax', 'path': 'jax/mpra-K562-optimal',
        'task': 'lentiMPRA K562 activity (scalar)', 'seq_len': 281,
    },
    'mpra-HepG2-optimal': {
        'framework': 'jax', 'path': 'jax/mpra-HepG2-optimal',
        'task': 'lentiMPRA HepG2 activity (scalar)', 'seq_len': 281,
    },
    'mpra-WTC11-optimal': {
        'framework': 'jax', 'path': 'jax/mpra-WTC11-optimal',
        'task': 'lentiMPRA WTC11 activity (scalar)', 'seq_len': 281,
    },
    'deepstarr-optimal': {
        'framework': 'jax', 'path': 'jax/deepstarr-optimal',
        'task': 'Drosophila STARR-seq (dev + hk enhancer activity)', 'seq_len': 256,
    },
    # Plant STARR-seq (Jores 2021). 437 bp for enhancer/combined, 170 bp promoter-only.
    **{
        f'plant-starrseq-{tissue}-{mode}': {
            'framework': 'jax', 'path': f'jax/plant-starrseq-{tissue}-{mode}',
            'task': f'Plant STARR-seq {tissue} / {mode} (log2 enrichment)',
            'seq_len': 170 if mode == 'promoter_only' else 437,
        }
        for tissue in ('leaf', 'proto')
        for mode in ('combined', 'enhancer', 'promoter_only')
    },
    # Gosai et al. lentiMPRA, boda-flatten-512-512 head (600 bp one-hot input).
    **{
        f'Gosai-{cell}-optimal': {
            'framework': 'jax', 'path': f'jax/Gosai-{cell}-optimal',
            'task': f'Gosai lentiMPRA {cell} activity (scalar)', 'seq_len': 600,
        }
        for cell in ('K562', 'HepG2', 'SKNSH')
    },
    # --- PyTorch -----------------------------------------------------------
    'mpra_K562': {
        'framework': 'torch', 'path': 'torch/mpra_K562',
        'task': 'lentiMPRA K562 activity (scalar)', 'seq_len': 281,
    },
    'mpra_HepG2': {
        'framework': 'torch', 'path': 'torch/mpra_HepG2',
        'task': 'lentiMPRA HepG2 activity (scalar)', 'seq_len': 281,
    },
    'mpra_WTC11': {
        'framework': 'torch', 'path': 'torch/mpra_WTC11',
        'task': 'lentiMPRA WTC11 activity (scalar)', 'seq_len': 281,
    },
    'starrseq_drosophila': {
        'framework': 'torch', 'path': 'torch/starrseq_drosophila',
        'task': 'Drosophila STARR-seq (dev + hk enhancer activity)', 'seq_len': 256,
    },
}


def list_pretrained() -> None:
    """Print the available checkpoints."""
    for fw in ('jax', 'torch'):
        print(f'{fw}:')
        for name, spec in CHECKPOINTS.items():
            if spec['framework'] == fw:
                print(f"  {name:36} {spec['task']}")


def _resolve(name: str) -> dict:
    if name not in CHECKPOINTS:
        raise KeyError(
            f'Unknown checkpoint {name!r}. Available: {sorted(CHECKPOINTS)}'
        )
    return CHECKPOINTS[name]


def download(name: str, *, repo_id: str = HF_REPO_ID, revision: str | None = None) -> Path:
    """Fetch just this checkpoint's files from the Hub. Returns the local directory."""
    from huggingface_hub import snapshot_download

    spec = _resolve(name)
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"{spec['path']}/**"],
    )
    return Path(local) / spec['path']


def load_pretrained(
    name: str,
    *,
    stage: str = 'stage2',
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
    base_checkpoint_path: str | None = None,
    local_dir: str | Path | None = None,
):
    """Load a released checkpoint, ready for inference.

    Args:
      name: key from :data:`CHECKPOINTS` (see :func:`list_pretrained`).
      stage: JAX only. ``'stage2'`` (fine-tuned, the default and usually what you want)
        or ``'stage1'`` (frozen encoder). For the plant models ``stage1`` is a ridge
        probe, not a model — use :func:`load_plant_probe` for those.
      base_checkpoint_path: local copy of the base AlphaGenome ``all_folds`` weights.
        If omitted these are fetched from ``google/alphagenome-all-folds``, which is
        access-gated: accept the terms and ``huggingface-cli login`` first.
      local_dir: load from this directory instead of the Hub (e.g. a shared filesystem).

    Returns:
      JAX: a ``CustomAlphaGenomeModel``. Call it via ``model._predict(...)``, or see
      ``scripts/test_ft_model_plant_starrseq.py`` for a worked prediction loop.
      Torch: an ``EncoderMPRAModel`` — ``model(x)`` or ``model.predict_sequences([...])``.
    """
    spec = _resolve(name)
    root = Path(local_dir) if local_dir else download(name, repo_id=repo_id, revision=revision)

    if spec['framework'] == 'torch':
        from alphagenome_encoder_ft import EncoderMPRAModel

        fname = 'finetuned_encoder.pt' if stage == 'stage2' else 'frozen_encoder.pt'
        model = EncoderMPRAModel.from_checkpoint(str(root / fname))
        model.eval()
        return model

    # --- JAX ---------------------------------------------------------------
    from alphagenome.models import dna_output
    from alphagenome_ft import (
        HeadConfig, HeadType, register_custom_head, load_checkpoint,
    )
    from .mpra_heads import EncoderMPRAHead, DeepSTARRHead, PlantMPRAHead, GosaiMPRAHead

    ckpt_dir = root / stage
    cfg = json.loads((ckpt_dir / 'config.json').read_text())
    head_name = cfg['custom_heads'][0]
    head_cfg = cfg['head_configs'][head_name]

    if head_cfg.get('source') == 'ridge':
        raise ValueError(
            f"{name}/{stage} is a ridge probe, not a model — use load_plant_probe()."
        )

    # Pick the head class the checkpoint was actually written with. Each family used a
    # different head, and they are NOT interchangeable (different module names and, for
    # Gosai, a different LayerNorm op) — see the head docstrings.
    if name.startswith('plant-starrseq-'):
        head_cls = PlantMPRAHead              # autotune, unnamed Haiku modules
    elif name.startswith('Gosai-'):
        head_cls = GosaiMPRAHead              # ALBench-S2F boda-flatten-512-512
    elif head_cfg.get('num_tracks', 1) == 2:
        head_cls = DeepSTARRHead
    else:
        head_cls = EncoderMPRAHead

    # load_checkpoint builds the head from the REGISTERED HeadConfig, not from the
    # checkpoint's config.json — so the metadata (notably nl_size, which varies per
    # plant cell) has to be registered here or the restore fails on a shape mismatch.
    register_custom_head(
        head_name,
        head_cls,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name=head_name,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=head_cfg.get('num_tracks', 1),
            metadata=head_cfg.get('metadata') or {},
        ),
    )
    return load_checkpoint(
        str(ckpt_dir),
        base_checkpoint_path=base_checkpoint_path,
        init_seq_len=spec['seq_len'],
    )


def load_plant_probe(
    name: str,
    *,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
    local_dir: str | Path | None = None,
) -> dict:
    """Load a plant ``stage1`` linear probe (a closed-form ridge, not a model).

    Returns ``{'w', 'xb', 'yb', 'lam'}``. Predict with::

        X = encoder_output.mean(axis=1)      # frozen PRETRAINED AlphaGenome encoder
        y = (X - probe['xb']) @ probe['w'] + probe['yb']

    Note the features come from the *unmodified* pretrained encoder — the probe has no
    encoder weights of its own.
    """
    import numpy as np
    import orbax.checkpoint as ocp

    if not name.startswith('plant-starrseq-'):
        raise ValueError('load_plant_probe is only for the plant-starrseq-* checkpoints.')

    root = Path(local_dir) if local_dir else download(name, repo_id=repo_id, revision=revision)
    store = ocp.StandardCheckpointer().restore(str(root / 'stage1' / 'checkpoint'))
    return {k: np.asarray(v) for k, v in store.items()}
