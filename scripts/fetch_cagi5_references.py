#!/usr/bin/env python3
"""
Fetch reference sequences for CAGI5 elements from UCSC genome browser (hg19).

Saves sequences to data/cagi5/cagi5_references.json

Usage:
    python scripts/fetch_cagi5_references.py
"""

import json
import requests
import time
from pathlib import Path

# CAGI5 element coordinates (hg19/GRCh37)
# Format: {element: (chrom, start, end)}
# We fetch a window around the variants to ensure we have enough context
CAGI5_REGIONS = {
    'F9': ('chrX', 138612500, 138613500),
    'GP1BB': ('chr22', 19710500, 19711700),
    'HBB': ('chr11', 5248100, 5248800),
    'HBG1': ('chr11', 5270900, 5271800),
    'HNF4A': ('chr20', 42984000, 42985000),
    'IRF4': ('chr6', 396000, 397200),
    'IRF6': ('chr1', 209989000, 209990500),
    'LDLR': ('chr19', 11199800, 11200800),
    'MSMB': ('chr10', 51548800, 51550400),
    'MYCrs6983267': ('chr8', 128412900, 128414500),
    'PKLR': ('chr1', 155271000, 155272300),
    'SORT1': ('chr1', 109817100, 109818700),
    'TERT-GBM': ('chr5', 1295000, 1295800),
    'TERT-HEK293T': ('chr5', 1295000, 1295800),
    'ZFAND3': ('chr6', 37775100, 37776600),
}


def fetch_sequence_ucsc(chrom: str, start: int, end: int, genome: str = 'hg19') -> str:
    """Fetch sequence from UCSC DAS server."""
    url = f"https://genome.ucsc.edu/cgi-bin/das/{genome}/dna?segment={chrom}:{start},{end}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Parse XML response
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)

    # Find DNA sequence
    for dna in root.iter('DNA'):
        seq = dna.text.strip().replace('\n', '').replace(' ', '').upper()
        return seq

    raise ValueError(f"Could not parse sequence from UCSC response")


def fetch_sequence_ensembl(chrom: str, start: int, end: int) -> str:
    """Fetch sequence from Ensembl REST API (GRCh37)."""
    # Remove 'chr' prefix for Ensembl
    chrom_ensembl = chrom.replace('chr', '')

    url = f"https://grch37.rest.ensembl.org/sequence/region/human/{chrom_ensembl}:{start}..{end}:1"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data['seq'].upper()


def main():
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / "data" / "cagi5"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cagi5_references.json"

    references = {}

    print("Fetching CAGI5 reference sequences from UCSC (hg19)...")
    print("=" * 60)

    for element, (chrom, start, end) in CAGI5_REGIONS.items():
        print(f"Fetching {element} ({chrom}:{start}-{end})...", end=" ", flush=True)

        try:
            seq = fetch_sequence_ucsc(chrom, start, end)
            references[element] = {
                'chrom': chrom,
                'start': start,
                'end': end,
                'sequence': seq,
                'length': len(seq)
            }
            print(f"OK ({len(seq)} bp)")
        except Exception as e:
            print(f"UCSC failed, trying Ensembl...", end=" ", flush=True)
            try:
                seq = fetch_sequence_ensembl(chrom, start, end)
                references[element] = {
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'sequence': seq,
                    'length': len(seq)
                }
                print(f"OK ({len(seq)} bp)")
            except Exception as e2:
                print(f"FAILED: {e2}")

        # Rate limiting
        time.sleep(0.5)

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(references, f, indent=2)

    print("=" * 60)
    print(f"Saved {len(references)} reference sequences to {output_file}")

    # Summary
    print("\nSummary:")
    for elem, data in references.items():
        print(f"  {elem}: {data['length']} bp")


if __name__ == "__main__":
    main()
