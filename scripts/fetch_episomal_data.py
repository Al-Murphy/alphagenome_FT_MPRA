#!/usr/bin/env python3
"""
Fetch the Gosai et al. 2024 episomal MPRA dataset.

Downloads ``Table_S2__MPRA_dataset.txt`` from the Tewhey lab public bucket
(linked from the BODA2 repo README) and saves it under
``data/gosai_episomal/DATA-Table_S2__MPRA_dataset.txt`` so that
``finetune_episomal_mpra.py`` and ``finetune_enformer_episomal_mpra.py`` can
load it via the default ``--data_path ./data/gosai_episomal`` argument.

Per the BODA2 maintainers (README, Feb 16 2024), the canonical training-data
table is updated; an earlier copy posted on bioRxiv contained a known issue.

Usage:
    python scripts/fetch_episomal_data.py
    python scripts/fetch_episomal_data.py --url <override-url>
    python scripts/fetch_episomal_data.py --output data/gosai_episomal
"""

import argparse
import sys
from pathlib import Path

import requests

# Canonical source — see boda2/README.md (Tewhey lab public-data bucket).
DEFAULT_URL = (
    "https://storage.googleapis.com/tewhey-public-data/CODA_resources/"
    "Table_S2__MPRA_dataset.txt"
)
# Filename used by the loader (alphagenome_ft_mpra/episomal_utils.DATA_FILENAME).
TARGET_FILENAME = "DATA-Table_S2__MPRA_dataset.txt"


def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream ``url`` into ``dest`` with a simple progress indicator."""
    print(f"Fetching {url}")
    print(f"  -> {dest}")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Source URL for the MPRA dataset table. Default: {DEFAULT_URL}",
    )
    parser.add_argument(
        "--output",
        default="data/gosai_episomal",
        help="Output directory (default: data/gosai_episomal)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() \
        else Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / TARGET_FILENAME
    if dest.exists() and not args.force:
        size_mb = dest.stat().st_size / 1e6
        print(f"Already present: {dest} ({size_mb:.1f} MB) — skipping. "
              f"Pass --force to re-download.")
        return 0

    try:
        download(args.url, dest)
    except requests.RequestException as e:
        print(f"Download failed: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return 1

    print(f"Done. Saved {dest.stat().st_size / 1e6:.1f} MB to {dest}.")
    print()
    print("Optional follow-on test sets (high-activity-designed, SNV-effects)")
    print(f"belong under {out_dir / 'test_sets'}/ — see the project README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
