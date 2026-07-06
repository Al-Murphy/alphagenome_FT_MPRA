#!/usr/bin/env python3
"""
Fetch + build the Jores et al. 2021 plant promoter STARR-seq dataset.

Unlike the other datasets (which ship a single supplementary table), the Jores21
activity table is rebuilt from the paper's public GitHub barcode counts by
replicating its enrichment pipeline. This script downloads those raw inputs and
writes the 8 built tables

    data/jores_plant_starrseq/jores21_{leaf,proto}_{35SEnh,noEnh}_{train,test}.tsv

so ``finetune_plant_starrseq.py`` (and the NTv3 / PlantCAD2 / Jores runners) can
load them via the default ``--data_path ./data/jores_plant_starrseq``.

Source: github.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters
(a pre-processed h5sd copy is also on Zenodo, records/7140083; the GitHub build is
the canonical path here).

Usage:
    python scripts/fetch_plant_starrseq_data.py
    python scripts/fetch_plant_starrseq_data.py --output data/jores_plant_starrseq
    python scripts/fetch_plant_starrseq_data.py --force
"""

import argparse
import sys
from pathlib import Path

from alphagenome_ft_mpra.plant_starrseq_utils import (
    build_plant_starrseq_dataset,
    data_is_present,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--output",
        default="data/jores_plant_starrseq",
        help="Output directory for the built TSVs (default: data/jores_plant_starrseq)",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Where to cache raw GitHub downloads (default: <output>/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the 8 TSVs already exist.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (
        Path(args.output).resolve()
        if Path(args.output).is_absolute()
        else (repo_root / args.output).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if data_is_present(str(out_dir)) and not args.force:
        print(f"Already present: 8 Jores21 tables under {out_dir} — skipping. "
              f"Pass --force to rebuild.")
        return 0

    print(f"Building Jores21 plant STARR-seq dataset into {out_dir}")
    print("(downloading raw barcode counts + annotations, then computing enrichment)")
    try:
        build_plant_starrseq_dataset(str(out_dir), cache_dir=args.cache_dir)
    except Exception as e:  # noqa: BLE001 — surface any download/build failure clearly
        print(f"Build failed: {e}", file=sys.stderr)
        return 1

    print(f"Done. Built 8 tables under {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
