"""Publish the fine-tuned AlphaGenome-encoder weights to the Hugging Face Hub.

Uploads /grid/koo/home/shared/models/alphagenome_encoder/{torch,jax} as one repo:

    <repo>/torch/<run>/{frozen_encoder.pt, finetuned_encoder.pt}
    <repo>/jax/<run>/stage{1,2}/{config.json, checkpoint/}
    <repo>/README.md          <- assets/hf_model_card.md

LICENCE. These are AlphaGenome Derivatives and remain subject to DeepMind's AlphaGenome
Model Terms (non-commercial). The model card states this; do not strip it.

Dry run first (prints what would be uploaded, touches nothing):

    python scripts/upload_weights_to_hf.py --dry_run

Then, once you are happy:

    huggingface-cli login
    python scripts/upload_weights_to_hf.py --repo_id Al-Murphy/alphagenome-encoder-ft
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED = Path('/grid/koo/home/shared/models/alphagenome_encoder')
CARD = REPO_ROOT / 'assets' / 'hf_model_card.md'

# Superseded by mpra_K562; no reason to ship it publicly.
EXCLUDE = {'torch/mpra_K562_old'}


def collect(shared: Path):
    """Everything we intend to publish, as (local_path, path_in_repo)."""
    items = []
    for framework in ('torch', 'jax'):
        for run in sorted((shared / framework).iterdir()):
            if not run.is_dir():
                continue
            rel = f'{framework}/{run.name}'
            if rel in EXCLUDE:
                print(f'  SKIP  {rel} (excluded)')
                continue
            items.append((run, rel))
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    p.add_argument('--repo_id', type=str, default='Al-Murphy/alphagenome-encoder-ft')
    p.add_argument('--shared', type=str, default=str(SHARED))
    p.add_argument('--private', action='store_true',
                   help='Create the repo private (flip to public in the UI when ready).')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    shared = Path(args.shared)
    if not CARD.exists():
        sys.exit(f'Model card not found: {CARD}')

    items = collect(shared)
    total = sum(f.stat().st_size for run, _ in items for f in run.rglob('*') if f.is_file())

    print(f'\nrepo_id : {args.repo_id}')
    print(f'card    : {CARD}')
    print(f'payload : {len(items)} runs, {total / 1e9:.1f} GB\n')
    for run, rel in items:
        size = sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
        print(f'  {rel:44} {size / 1e6:8.0f} MB')

    if args.dry_run:
        print('\n--dry_run: nothing uploaded.')
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', private=args.private, exist_ok=True)
    print(f'\nrepo ready: https://huggingface.co/{args.repo_id}')

    # card first, so the licence notice is live before any weights land
    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo='README.md',
        repo_id=args.repo_id,
        repo_type='model',
        commit_message='Add model card (AlphaGenome Derivative; non-commercial terms)',
    )
    print('  uploaded README.md')

    for run, rel in items:
        api.upload_folder(
            folder_path=str(run),
            path_in_repo=rel,
            repo_id=args.repo_id,
            repo_type='model',
            commit_message=f'Add {rel}',
        )
        print(f'  uploaded {rel}')

    print(f'\nDone: https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    sys.exit(main())
