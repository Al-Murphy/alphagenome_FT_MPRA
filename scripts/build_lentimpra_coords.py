"""Build the LentiMPRA hg38 coordinate table used for the AG fold genomic test filter.

Parses Table S3 ("large-scale lib design") from Agarwal et al. 2025 — one sheet per
cell type (HepG2 / K562 / WTC11) — and writes a tidy TSV mapping each element
``name`` (matching the ``seq_id`` column of data/legnet_lentimpra/<cell>.tsv,
including the ``_Reversed:`` / ``_R`` orientation entries) to its hg38 coordinates.

The three sheets differ in header row, columns, naming and chromosome format, so the
header row is detected per sheet and the chromosome is normalised to ``chr#``.

Source: https://zenodo.org/records/10558183
        "Table S3 - large-scale lib design.xlsx"

Usage:
    python scripts/build_lentimpra_coords.py \
        --xlsx data/legnet_lentimpra/design_tables/TableS3_large_scale_lib_design.xlsx \
        --out  data/legnet_lentimpra/lentimpra_hg38_coords.tsv
"""

import argparse
import os
import urllib.request

import openpyxl
import pandas as pd

ZENODO_URL = (
    "https://zenodo.org/records/10558183/files/"
    "Table%20S3%20-%20large-scale%20lib%20design.xlsx?download=1"
)
SHEET_FOR = {
    "HepG2": "HepG2 large-scale",
    "K562": "K562 large-scale",
    "WTC11": "WTC11 large-scale",
}


def _find_header_row(path: str, sheet: str) -> int:
    """Return the 0-based row index whose first cell is the literal 'name'."""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=12)):
        if row[0].value == "name":
            wb.close()
            return i
    wb.close()
    raise RuntimeError(f"Could not find 'name' header in sheet '{sheet}'")


def _norm_chr(value) -> str:
    s = str(value).strip()
    return s if s.lower().startswith("chr") else f"chr{s}"


def build(xlsx: str, out: str) -> None:
    if not os.path.exists(xlsx):
        os.makedirs(os.path.dirname(xlsx), exist_ok=True)
        print(f"Downloading Table S3 from Zenodo -> {xlsx}")
        urllib.request.urlretrieve(ZENODO_URL, xlsx)

    frames = []
    for cell_type, sheet in SHEET_FOR.items():
        header_row = _find_header_row(xlsx, sheet)
        d = pd.read_excel(xlsx, sheet_name=sheet, header=header_row)
        seq_col = [c for c in d.columns if str(c).startswith("230nt")][0]
        d = d.rename(columns={seq_col: "seq230"})
        d = d[["name", "category", "chr.hg38", "start.hg38", "stop.hg38", "str.hg38", "seq230"]].copy()
        d["chr.hg38"] = d["chr.hg38"].map(_norm_chr)
        d["start.hg38"] = pd.to_numeric(d["start.hg38"], errors="coerce").astype("Int64")
        d["stop.hg38"] = pd.to_numeric(d["stop.hg38"], errors="coerce").astype("Int64")
        d.insert(0, "cell_type", cell_type)
        frames.append(d)
        print(f"{cell_type}: header_row={header_row}, rows={len(d)}, unique names={d['name'].nunique()}")

    coords = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    coords.to_csv(out, sep="\t", index=False)
    print(f"Wrote {len(coords)} rows -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--xlsx",
        default="data/legnet_lentimpra/design_tables/TableS3_large_scale_lib_design.xlsx",
        help="Path to Table S3 xlsx (downloaded from Zenodo if missing).",
    )
    p.add_argument(
        "--out",
        default="data/legnet_lentimpra/lentimpra_hg38_coords.tsv",
        help="Output TSV path.",
    )
    args = p.parse_args()
    build(args.xlsx, args.out)


if __name__ == "__main__":
    main()
