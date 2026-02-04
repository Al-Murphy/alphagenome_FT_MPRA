#!/usr/bin/env python
"""
Run FIMO over a directory of MEME motif files and summarize significant hits.

Typical usage (from project root):

    python scripts/run_fimo_batch.py \
        --motif_dir data/motifs \
        --seqlets_fasta results/plots/attribution_HepG2_deepshap/fimo_input/seqlets_for_fimo_deepshap.fa \
        --output_root results/plots/attribution_HepG2_deepshap/fimo_input/fimo_all_motifs \
        --fimo_p_thresh 1e-4 \
        --qvalue_thresh 0.05

This will:
  1) Run FIMO once per `*.meme` file in `motif_dir`, placing each result in:
         <output_root>/fimo_<motif_basename>/
  2) Parse all `fimo.txt` files and write:
         <output_root>/fimo_significant_hits.tsv
         <output_root>/fimo_motif_counts.tsv

`fimo_significant_hits.tsv` contains one row per significant site (q-value <= qvalue_thresh),
sorted by q-value. `fimo_motif_counts.tsv` summarizes the number of significant sites per motif.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from glob import glob
from pathlib import Path
from shutil import which
from typing import List, Dict, Any


def run_cmd(cmd: List[str]) -> None:
    """Run a shell command and raise if it fails."""
    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Command failed with exit code", result.returncode, file=sys.stderr)
        print("STDOUT:\n", result.stdout, file=sys.stderr)
        print("STDERR:\n", result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run_fimo_for_motif(
    motif_file: Path,
    seqlets_fasta: Path,
    output_root: Path,
    fimo_p_thresh: float,
) -> Path:
    """Run FIMO for a single MEME motif file."""
    motif_name = motif_file.stem  # e.g., MA0114.1 from MA0114.1.meme
    outdir = output_root / f"fimo_{motif_name}"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "fimo",
        "--oc",
        str(outdir),
        "--thresh",
        str(fimo_p_thresh),
        str(motif_file),
        str(seqlets_fasta),
    ]
    run_cmd(cmd)
    return outdir


def parse_fimo_results(
    output_root: Path,
    qvalue_thresh: float,
) -> None:
    """
    Parse all fimo.txt files under output_root and write:
      - fimo_significant_hits.tsv
      - fimo_motif_counts.tsv
    """
    all_hits: List[Dict[str, Any]] = []

    # FIMO output directories are expected to be `fimo_<motif_name>`
    for fimo_dir in sorted(output_root.glob("fimo_*")):
        fimo_txt = fimo_dir / "fimo.txt"
        if not fimo_txt.exists():
            # FIMO may skip writing if there are no hits; skip silently.
            continue

        with fimo_txt.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    qval = float(row.get("q-value", "nan"))
                except ValueError:
                    continue
                if qval <= qvalue_thresh:
                    row["_fimo_dir"] = str(fimo_dir)
                    row["_motif_file"] = str(fimo_dir.name.replace("fimo_", ""))  # motif name
                    all_hits.append(row)

    if not all_hits:
        print(f"No hits found with q-value <= {qvalue_thresh}. Nothing to write.")
        return

    # Sort by q-value ascending
    all_hits.sort(key=lambda r: float(r.get("q-value", "inf")))

    # Determine output paths
    sig_hits_path = output_root / "fimo_significant_hits.tsv"
    counts_path = output_root / "fimo_motif_counts.tsv"

    # Write significant hits
    fieldnames = [
        "_motif_file",  # derived from directory name (e.g., MA0114.1)
        "motif_id",
        "motif_alt_id",
        "sequence_name",
        "start",
        "stop",
        "strand",
        "score",
        "p-value",
        "q-value",
        "matched_sequence",
        "_fimo_dir",
    ]

    with sig_hits_path.open("w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in all_hits:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Summarize counts per motif_id (or motif_file)
    counts: Dict[str, int] = {}
    for row in all_hits:
        key = row.get("motif_id") or row.get("_motif_file")
        counts[key] = counts.get(key, 0) + 1

    with counts_path.open("w", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["motif_id", "significant_site_count"])
        for motif_id, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            writer.writerow([motif_id, count])

    print(f"Wrote significant hits to: {sig_hits_path}")
    print(f"Wrote motif counts to:     {counts_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FIMO for all MEME motif files in a directory and summarize significant hits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--motif_dir",
        required=True,
        help="Directory containing MEME motif files (e.g., JASPAR *.meme).",
    )
    parser.add_argument(
        "--seqlets_fasta",
        required=True,
        help="FASTA file with seqlets (e.g., seqlets_for_fimo_deepshap.fa).",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root output directory where FIMO outputs and summaries will be written.",
    )
    parser.add_argument(
        "--fimo_p_thresh",
        type=float,
        default=1e-4,
        help="P-value threshold passed to FIMO (--thresh).",
    )
    parser.add_argument(
        "--qvalue_thresh",
        type=float,
        default=0.05,
        help="q-value threshold for defining significant hits when summarizing.",
    )

    args = parser.parse_args()

    # Basic validation
    motif_dir = Path(args.motif_dir)
    seqlets_fasta = Path(args.seqlets_fasta)
    output_root = Path(args.output_root)

    if which("fimo") is None:
        print(
            "Error: 'fimo' command not found. Make sure the MEME suite is installed and 'fimo' is on your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not motif_dir.is_dir():
        print(f"Error: motif_dir does not exist or is not a directory: {motif_dir}", file=sys.stderr)
        sys.exit(1)

    if not seqlets_fasta.is_file():
        print(f"Error: seqlets_fasta not found: {seqlets_fasta}", file=sys.stderr)
        sys.exit(1)

    motif_files = sorted(glob(str(motif_dir / "*.meme")))
    if not motif_files:
        print(f"No .meme files found in motif_dir: {motif_dir}", file=sys.stderr)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Running FIMO for motifs")
    print("=" * 80)
    print(f"Motif directory:   {motif_dir}")
    print(f"Seqlets FASTA:     {seqlets_fasta}")
    print(f"Output root:       {output_root}")
    print(f"FIMO p-threshold:  {args.fimo_p_thresh}")
    print(f"q-value threshold: {args.qvalue_thresh}")
    print("")

    # Run FIMO for each motif
    for motif_path_str in motif_files:
        motif_file = Path(motif_path_str)
        print("-" * 80)
        print(f"Motif: {motif_file.name}")
        try:
            run_fimo_for_motif(
                motif_file=motif_file,
                seqlets_fasta=seqlets_fasta,
                output_root=output_root,
                fimo_p_thresh=args.fimo_p_thresh,
            )
        except RuntimeError as e:
            print(f"Skipping {motif_file.name} due to error: {e}", file=sys.stderr)
            continue

    print("\n" + "=" * 80)
    print("Parsing FIMO outputs and summarizing significant hits")
    print("=" * 80)
    parse_fimo_results(output_root=output_root, qvalue_thresh=args.qvalue_thresh)


if __name__ == "__main__":
    main()

