"""
Shared helpers for the plant STARR-seq (Jores et al. 2021) promoter dataset.

Pure-pandas/numpy: no JAX, no PyTorch. The JAX dataset class
``PlantStarrSeqDataset`` (in ``data.py``) and the PyTorch runner scripts
(``scripts/finetune_{ntv3,plantcad2,jores}_plant_starrseq.py``) all import from
here so no two paths can drift on construct assembly, splits, or one-hot
encoding. This is the plant counterpart to ``episomal_utils.py``.

Jores et al. 2021 measured plant promoter activity by STARR-seq in two systems
(tobacco leaf and maize protoplast), across three species (Arabidopsis / maize /
sorghum) with and without the CaMV 35S enhancer. This module rebuilds the
per-promoter enrichment table from the paper's public GitHub data and exposes it
in three data modes:

  * ``promoter_only`` — the raw 170 bp core promoter, no reporter construct.
  * ``enhancer``      — the full 437 bp construct WITH the 35S enhancer.
  * ``combined``      — 437 bp constructs, both enhancer and no-enhancer rows.

Paper: Jores, T. et al. "Synthetic promoter designs enabled by a comprehensive
analysis of plant core promoters." Nature Plants 7, 842-855 (2021).
Data:  github.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters
"""

import gzip  # noqa: F401 — kept for parity with the paper's gzip count files
import math
import os
import urllib.request
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

# 437 bp = full construct (Sorghum worst case: 153 enh + 170 promoter + 102 UTR + 12 barcode).
SEQUENCE_LENGTH = 437
# 170 bp = the raw core promoter, used for the promoter_only mode.
PROMOTER_LENGTH = 170

VALID_TISSUES = ["leaf", "proto"]
VALID_MODES = ["promoter_only", "enhancer", "combined"]
VALID_SPLITS = ["train", "val", "test"]

# Full CaMV 35S enhancer (-199..-47 relative to the 35S TSS, 153 bp; CaMV genome V00141.1).
ENHANCER_153 = (
    "AGATCTCTCTGCCGACAGTGGTCCCAAAGATGGACCCCCACCCACGAGGAGCATCGTGGAAAAAGAAGAC"
    "GTTCCAACCACGTCTTCAAAGCAAGTGGATTGATGTGACATCTCCACTGACGTAAGGGATGACGCACAAT"
    "CCCACTATCCTTC"
)

# Full-length species-specific 5' UTRs (Jores et al. 2021, Supplementary Table 8).
# Arabidopsis and maize both use the maize histone H3.2 UTR (ZmUTR); sorghum uses SbUTR.
UTR_MAP = {
    "At": "CCCGTCCGAACTCCGAACCCCAGAACAGAGCAAAGCCTCCTCGGCCTCCCTGTCCCCAGCCTTCCCCG",  # ZmUTR, 68 bp
    "Zm": "CCCGTCCGAACTCCGAACCCCAGAACAGAGCAAAGCCTCCTCGGCCTCCCTGTCCCCAGCCTTCCCCG",  # ZmUTR, 68 bp
    "Sb": "TACCACCCTCGTCTCGCTCCAATTCCCCACCGCAAATCCAGAGCCTTCCATTTCAAACACTTCGGAGCAACATCTCCCTTCTCCCCAGCCCAATCACCCGCC",  # SbUTR, 102 bp
}

# Per-tissue TSV naming written by the builder.
_ENH_TAG = {True: "35SEnh", False: "noEnh"}


# ── One-hot / string helpers ─────────────────────────────────────────────────

_NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence to a ``(len(seq), 4)`` float32 array.

    Encoding: A=0, C=1, G=2, T=3. ``N`` (and any other character) maps to
    ``[0.25, 0.25, 0.25, 0.25]``, matching ``episomal_utils._one_hot_encode`` so
    both datasets encode identically.
    """
    seq = seq.upper()
    ohe = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = _NUCLEOTIDE_MAP.get(base)
        if idx is not None:
            ohe[i, idx] = 1.0
        else:
            ohe[i, :] = 0.25
    return ohe


def _reverse_complement_ohe(ohe: np.ndarray) -> np.ndarray:
    """Reverse complement a one-hot sequence (reverse axis 0, swap A<->T, C<->G)."""
    return ohe[::-1, [3, 2, 1, 0]].copy()


def reverse_complement_str(seq: str) -> str:
    """Reverse complement a DNA string (used for string-level augmentation)."""
    return seq.translate(_COMPLEMENT)[::-1]


def circular_shift_str(seq: str, shift: int) -> str:
    """Circularly roll a string by ``shift`` (positive = rightward)."""
    if shift == 0:
        return seq
    return seq[-shift:] + seq[:-shift]


def pad_or_trim_str(seq: str, length: int) -> str:
    """Right-pad with ``N`` or right-trim ``seq`` to exactly ``length`` bases."""
    if len(seq) < length:
        return seq + "N" * (length - len(seq))
    return seq[:length]


def build_construct(
    promoter: str, sp: str, use_enhancer: bool, rng: np.random.Generator
) -> str:
    """Assemble the reporter construct exactly as measured in the STARR-seq assay.

    Layout: ``[upstream 153 bp] + [170 bp promoter] + [species UTR] + [12 bp barcode]``.
    The upstream 153 bp is the real CaMV 35S enhancer (``ENHANCER_153``) when
    ``use_enhancer`` else a random 153 bp filler, matching the +/- enhancer
    conditions in the paper. The barcode is random (it carries no information and
    is later trimmed / padded to ``SEQUENCE_LENGTH``).
    """
    if use_enhancer:
        upstream = ENHANCER_153
    else:
        upstream = "".join(rng.choice(list("ACGT"), size=len(ENHANCER_153)))

    utr = UTR_MAP[sp]
    barcode = "".join(rng.choice(list("ACGT"), size=12))

    return upstream + promoter + utr + barcode


# ── Split-aware loader ───────────────────────────────────────────────────────


def _load_plant_starrseq_data(
    data_path: str,
    tissue: str,
    mode: str,
    split: str,
    val_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Load one (tissue, mode, split) of the built Jores21 tables.

    Returns a DataFrame with columns ``sequence`` (raw 170 bp promoter),
    ``enrichment`` (float32 target), ``sp`` (``At``/``Zm``/``Sb``), and
    ``use_enh`` (whether the row's construct carries the 35S enhancer).

    Split logic (matches the original sweep):
      * test  — the held-out CNN test genes, from ``jores21_<tissue>_*_test.tsv``.
      * train/val — a deterministic ``val_frac`` permutation split (seed 42) of
        ``jores21_<tissue>_*_train.tsv``.
    For ``promoter_only`` and ``enhancer`` only the ``35SEnh`` table is used
    (same promoters, enhancer construct); ``combined`` concatenates the
    ``35SEnh`` and ``noEnh`` tables with a per-row ``use_enh`` flag.
    """
    assert tissue in VALID_TISSUES, f"tissue must be one of {VALID_TISSUES}"
    assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
    assert split in VALID_SPLITS, f"split must be one of {VALID_SPLITS}"

    def _read(enh_tag: str, part: str) -> pd.DataFrame:
        path = os.path.join(data_path, f"jores21_{tissue}_{enh_tag}_{part}.tsv")
        assert os.path.exists(path), (
            f"Plant STARR-seq table not found: {path}. "
            f"Run scripts/fetch_plant_starrseq_data.py first."
        )
        return pd.read_csv(path, sep="\t")

    if mode in ("promoter_only", "enhancer"):
        if split in ("train", "val"):
            df = _read("35SEnh", "train")
            idx = np.random.default_rng(seed).permutation(len(df))
            n_val = int(len(df) * val_frac)
            df = df.iloc[idx[:n_val]] if split == "val" else df.iloc[idx[n_val:]]
        else:
            df = _read("35SEnh", "test")
        df = df.copy()
        df["use_enh"] = True
    else:  # combined
        part = "train" if split in ("train", "val") else "test"
        df_enh = _read("35SEnh", part).copy()
        df_noenh = _read("noEnh", part).copy()
        df_enh["use_enh"] = True
        df_noenh["use_enh"] = False
        df = pd.concat([df_enh, df_noenh], ignore_index=True)
        if split in ("train", "val"):
            idx = np.random.default_rng(seed).permutation(len(df))
            n_val = int(len(df) * val_frac)
            df = df.iloc[idx[:n_val]] if split == "val" else df.iloc[idx[n_val:]]

    return pd.DataFrame({
        "sequence": df["sequence"].astype(str).values,
        "enrichment": df["enrichment"].values.astype(np.float32),
        "sp": df["sp"].values,
        "use_enh": df["use_enh"].values,
    }).reset_index(drop=True)


def build_sequence_for_mode(
    row: pd.Series,
    mode: str,
    rng: np.random.Generator,
    random_shift: bool = False,
    shift_likelihood: float = 0.5,
    max_shift: int = 25,
    reverse_complement: bool = False,
    reverse_complement_likelihood: float = 0.5,
) -> str:
    """Turn one loaded row into the final model-input DNA string for ``mode``.

    ``promoter_only`` returns the 170 bp promoter (optionally RC'd);
    ``enhancer``/``combined`` build the 437 bp construct with optional circular
    shift then RC. This is the single source of truth for both the JAX dataset
    and the PyTorch runners.
    """
    seq = str(row["sequence"])

    if mode == "promoter_only":
        seq = pad_or_trim_str(seq, PROMOTER_LENGTH)
        if reverse_complement and rng.random() < reverse_complement_likelihood:
            seq = reverse_complement_str(seq)
        return seq

    seq = build_construct(seq, row["sp"], bool(row["use_enh"]), rng)

    if random_shift and rng.random() < shift_likelihood:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        seq = circular_shift_str(seq, shift)

    seq = pad_or_trim_str(seq, SEQUENCE_LENGTH)

    if reverse_complement and rng.random() < reverse_complement_likelihood:
        seq = reverse_complement_str(seq)

    return seq


# ═════════════════════════════════════════════════════════════════════════════
# Dataset builder — rebuilds the Jores21 enrichment tables from the paper's
# public GitHub data (replicates the R enrichment pipeline). Called by
# scripts/fetch_plant_starrseq_data.py.
# ═════════════════════════════════════════════════════════════════════════════

_REPO_BASE = (
    "https://raw.githubusercontent.com/tobjores/"
    "Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/main"
)
_ZENODO_URL = "https://zenodo.org/records/7140083/files/jores21.zip?download=1"

_SPECIES = ["At", "Zm", "Sb"]
_TISSUES = ["leaf", "proto"]
_ENHANCER_CONDS = [True, False]
_REPS = [1, 2]
_READ_COUNT_CUTOFF = 5
_SP_PREFIXES = {"AT": "At", "Zm": "Zm", "ENSRNA": "Sb", "SORBI_": "Sb"}


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return
    urllib.request.urlretrieve(url, dest)


def download_raw_data(cache_dir: str) -> None:
    """Download all raw inputs (subassembly, barcode counts, annotations, CNN splits)."""
    sub_dir = os.path.join(cache_dir, "subassembly")
    _download(
        f"{_REPO_BASE}/data/subassembly/subassembly_pPSup_plPRO_variant.tsv.gz",
        os.path.join(sub_dir, "subassembly_pPSup_plPRO_variant.tsv.gz"),
    )
    _download(
        f"{_REPO_BASE}/data/subassembly/controls_ZmUTR.tsv",
        os.path.join(sub_dir, "controls_ZmUTR.tsv"),
    )
    _download(
        f"{_REPO_BASE}/data/subassembly/controls_SbUTR.tsv",
        os.path.join(sub_dir, "controls_SbUTR.tsv"),
    )

    annot_dir = os.path.join(cache_dir, "annotation")
    for species_full in ["Arabidopsis", "Maize", "Sorghum"]:
        _download(
            f"{_REPO_BASE}/data/promoter_annotation/{species_full}_all_promoters_unique.tsv",
            os.path.join(annot_dir, f"{species_full}_all_promoters_unique.tsv"),
        )

    for tissue in _TISSUES:
        for sp in _SPECIES:
            for rep in _REPS:
                rep_dir = os.path.join(cache_dir, "barcode_counts", tissue, f"{sp}_Rep{rep}")
                for enh in ["35SEnh", "noEnh"]:
                    roman = "I" if rep == 1 else "II"
                    base = f"barcodes_pPSup_{sp}PRO_{enh}_{roman}"

                    fname = f"{base}_dark.count.gz"
                    _download(
                        f"{_REPO_BASE}/data/barcode_counts/{tissue}/{sp}_Rep{rep}/{fname}",
                        os.path.join(rep_dir, fname),
                    )

                    # proto At_Rep2 reuses Rep1 input, so its own input file does not exist
                    if not (tissue == "proto" and sp == "At" and rep == 2):
                        fname = f"{base}_inp.count.gz"
                        _download(
                            f"{_REPO_BASE}/data/barcode_counts/{tissue}/{sp}_Rep{rep}/{fname}",
                            os.path.join(rep_dir, fname),
                        )

                    if tissue == "leaf":
                        fname = f"{base}_light.count.gz"
                        _download(
                            f"{_REPO_BASE}/data/barcode_counts/{tissue}/{sp}_Rep{rep}/{fname}",
                            os.path.join(rep_dir, fname),
                        )

    cnn_dir = os.path.join(cache_dir, "cnn_splits")
    for tissue in _TISSUES:
        _download(
            f"{_REPO_BASE}/CNN/CNN_test_{tissue}.tsv",
            os.path.join(cnn_dir, f"CNN_test_{tissue}.tsv"),
        )


def _gene_to_sp(gene: str) -> str:
    for prefix, sp in _SP_PREFIXES.items():
        if gene.startswith(prefix):
            return sp
    return "unknown"


def _load_annotation(cache_dir: str) -> pd.DataFrame:
    annot_dir = os.path.join(cache_dir, "annotation")
    sp_map = {"Arabidopsis": "At", "Maize": "Zm", "Sorghum": "Sb"}

    frames = []
    for species_full, sp in sp_map.items():
        df = pd.read_csv(
            os.path.join(annot_dir, f"{species_full}_all_promoters_unique.tsv"), sep="\t"
        )
        df["sp"] = sp
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _load_controls(cache_dir: str) -> pd.DataFrame:
    sub_dir = os.path.join(cache_dir, "subassembly")

    zm_ctrl = pd.read_csv(os.path.join(sub_dir, "controls_ZmUTR.tsv"), sep="\t")
    zm_ctrl["utr_group"] = "ZmUTR"

    sb_ctrl = pd.read_csv(os.path.join(sub_dir, "controls_SbUTR.tsv"), sep="\t")
    sb_ctrl["utr_group"] = "SbUTR"

    return pd.concat([zm_ctrl, sb_ctrl], ignore_index=True)


def _load_count_file(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", header=None, names=["count", "barcode"], compression="gzip",
    )


def filter_subassembly(cache_dir: str) -> pd.DataFrame:
    """Resolve barcode->gene assignments and flag full-length WT promoters.

    Returns columns: barcode, gene, start, stop, variant, FL, sp, type.
    """
    sub_path = os.path.join(
        cache_dir, "subassembly", "subassembly_pPSup_plPRO_variant.tsv.gz"
    )
    raw = pd.read_csv(sub_path, sep="\t", compression="gzip")

    controls = _load_controls(cache_dir)
    annotation = _load_annotation(cache_dir)

    raw["sp"] = raw["gene"].apply(_gene_to_sp)

    ctrl_barcodes = set(controls["barcode"])
    ctrl_map = controls.set_index("barcode")[["promoter", "enhancer"]].to_dict("index")

    def _recode_control(row):
        if row["barcode"] in ctrl_barcodes:
            info = ctrl_map[row["barcode"]]
            pro = "withPRO" if info["promoter"] == "35S" else "noPRO"
            enh = "withENH" if info["enhancer"] == "35S" else "noENH"
            return f"control-{pro}-{enh}"
        return row["gene"]

    raw["gene"] = raw.apply(_recode_control, axis=1)

    # resolve ambiguous barcodes: keep the winning gene only if it dominates 10:1
    bc_gene_counts = raw.groupby("barcode")["gene"].nunique()
    ambiguous_bcs = set(bc_gene_counts[bc_gene_counts > 1].index)

    resolved = []
    for bc, group in raw[raw["barcode"].isin(ambiguous_bcs)].groupby("barcode"):
        counts = group.groupby("gene")["assembly.count"].sum().sort_values(ascending=False)
        if len(counts) >= 2 and counts.iloc[0] >= 10 * counts.iloc[1]:
            resolved.append(group[group["gene"] == counts.index[0]])

    unambiguous = raw[~raw["barcode"].isin(ambiguous_bcs)]
    raw = pd.concat([unambiguous] + resolved, ignore_index=True) if resolved else unambiguous

    raw["FL"] = (raw["start"] == 1) & (raw["stop"] == 170)

    annot_slim = annotation[["gene", "type"]].drop_duplicates()
    raw = raw.merge(annot_slim, on="gene", how="left")
    raw.loc[raw["gene"].str.startswith("control"), "type"] = "control"

    return raw[["barcode", "gene", "start", "stop", "variant", "FL", "sp", "type"]]


def compute_enrichment(cache_dir: str, subassembly: pd.DataFrame) -> pd.DataFrame:
    """Compute per-gene log2 enrichment for every (tissue, species, enhancer) condition.

    Returns columns: sys (tissue), sp, enhancer (bool), gene, type, enrichment.
    """
    all_results = []

    for tissue in _TISSUES:
        for sp in _SPECIES:
            sub_sp = subassembly[subassembly["sp"] == sp].copy()
            controls_mask = subassembly["gene"].str.startswith("control")
            sub_with_ctrl = pd.concat([
                sub_sp[~sub_sp["gene"].str.startswith("control")],
                subassembly[controls_mask],
            ]).drop_duplicates(subset=["barcode"])

            for enh in _ENHANCER_CONDS:
                enh_str = _ENH_TAG[enh]
                rep_enrichments = []

                for rep in _REPS:
                    roman = "I" if rep == 1 else "II"
                    rep_dir = os.path.join(cache_dir, "barcode_counts", tissue, f"{sp}_Rep{rep}")
                    base = f"barcodes_pPSup_{sp}PRO_{enh_str}_{roman}"

                    # proto At_Rep2 uses Rep1 input
                    if tissue == "proto" and sp == "At" and rep == 2:
                        inp_dir = os.path.join(cache_dir, "barcode_counts", tissue, f"{sp}_Rep1")
                        inp_base = f"barcodes_pPSup_{sp}PRO_{enh_str}_I"
                        inp_path = os.path.join(inp_dir, f"{inp_base}_inp.count.gz")
                    else:
                        inp_path = os.path.join(rep_dir, f"{base}_inp.count.gz")

                    dark_path = os.path.join(rep_dir, f"{base}_dark.count.gz")

                    if not os.path.exists(dark_path) or not os.path.exists(inp_path):
                        continue

                    inp = _load_count_file(inp_path)
                    dark = _load_count_file(dark_path)

                    merged = dark.merge(inp, on="barcode", suffixes=("_out", "_inp"))
                    merged = merged.merge(
                        sub_with_ctrl[["barcode", "gene", "variant", "FL", "type"]], on="barcode"
                    )

                    merged = merged[
                        (merged["count_inp"] >= _READ_COUNT_CUTOFF)
                        & (merged["count_out"] >= _READ_COUNT_CUTOFF)
                    ]
                    if len(merged) == 0:
                        continue

                    total_out = merged["count_out"].sum()
                    total_inp = merged["count_inp"].sum()
                    merged["enrichment"] = np.log2(
                        (merged["count_out"] / total_out) / (merged["count_inp"] / total_inp)
                    )

                    rep_enrichments.append(merged.assign(rep=rep))

                if not rep_enrichments:
                    continue

                combined = pd.concat(rep_enrichments, ignore_index=True)

                wt_mask = (
                    combined["gene"].str.startswith("control")
                    | ((combined["variant"] == "WT") & combined["FL"])
                ) & (combined["gene"] != "35Spr")
                combined = combined[wt_mask]

                # per-replicate normalization: subtract the no-enhancer control median
                ctrl_gene = "control-withPRO-noENH"
                for rep_val in combined["rep"].unique():
                    rep_mask = combined["rep"] == rep_val
                    ctrl_mask = rep_mask & (combined["gene"] == ctrl_gene)
                    if ctrl_mask.sum() > 0:
                        ctrl_median = combined.loc[ctrl_mask, "enrichment"].median()
                        combined.loc[rep_mask, "enrichment"] -= ctrl_median

                combined = combined[~combined["gene"].str.startswith("control")]

                agg_by_bc = (
                    combined.groupby(["gene", "type", "rep"])["enrichment"].median().reset_index()
                )
                agg_by_rep = (
                    agg_by_bc.groupby(["gene", "type"])["enrichment"].mean().reset_index()
                )
                agg_by_rep["sys"] = tissue
                agg_by_rep["sp"] = sp
                agg_by_rep["enhancer"] = enh

                all_results.append(agg_by_rep)

    return pd.concat(all_results, ignore_index=True)


def _load_test_genes(cache_dir: str, tissue: str) -> set:
    path = os.path.join(cache_dir, "cnn_splits", f"CNN_test_{tissue}.tsv")
    df = pd.read_csv(path, sep="\t")
    return set(df["gene"])


def export_datasets(
    cache_dir: str, output_dir: str, enrichment: pd.DataFrame, annotation: pd.DataFrame
) -> None:
    """Join enrichment to promoter sequences and write the 8 per-tissue TSVs."""
    os.makedirs(output_dir, exist_ok=True)

    seq_map = annotation[["gene", "sequence"]].drop_duplicates().set_index("gene")["sequence"]

    for tissue in _TISSUES:
        test_genes = _load_test_genes(cache_dir, tissue)

        for enh in _ENHANCER_CONDS:
            enh_str = _ENH_TAG[enh]
            subset = enrichment[
                (enrichment["sys"] == tissue) & (enrichment["enhancer"] == enh)
            ].copy()
            subset["sequence"] = subset["gene"].map(seq_map)
            subset = subset.dropna(subset=["sequence"])

            train = subset[~subset["gene"].isin(test_genes)]
            test = subset[subset["gene"].isin(test_genes)]

            cols = ["gene", "sp", "type", "sequence", "enrichment"]
            train[cols].to_csv(
                os.path.join(output_dir, f"jores21_{tissue}_{enh_str}_train.tsv"),
                sep="\t", index=False,
            )
            test[cols].to_csv(
                os.path.join(output_dir, f"jores21_{tissue}_{enh_str}_test.tsv"),
                sep="\t", index=False,
            )
            print(
                f"{tissue} {enh_str}: train={len(train):,}, test={len(test):,}, "
                f"enrichment=[{subset['enrichment'].min():.3f}, {subset['enrichment'].max():.3f}]"
            )


def build_plant_starrseq_dataset(output_dir: str, cache_dir: Optional[str] = None) -> None:
    """Build the full Jores21 plant STARR-seq dataset from the paper's raw data.

    Downloads the raw GitHub inputs into ``cache_dir`` (default:
    ``<output_dir>/raw``), rebuilds the enrichment table, and writes the 8
    ``jores21_<tissue>_<enh>_<split>.tsv`` files into ``output_dir``.
    """
    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "raw")

    download_raw_data(cache_dir)

    subassembly = filter_subassembly(cache_dir)
    enrichment = compute_enrichment(cache_dir, subassembly)
    annotation = _load_annotation(cache_dir)

    export_datasets(cache_dir, output_dir, enrichment, annotation)


def download_zenodo(dest_dir: str) -> None:
    """Alternate path: download the pre-processed jores21.zip (h5sd) from Zenodo.

    Not used by the default build pipeline; provided for parity with the paper's
    Zenodo record (records/7140083). ``build_plant_starrseq_dataset`` is the
    canonical source of the TSVs this repo trains on.
    """
    import h5py  # noqa: F401 — surface a clear error if the h5 stack is missing

    zip_path = os.path.join(dest_dir, "jores21.zip")
    os.makedirs(dest_dir, exist_ok=True)
    urllib.request.urlretrieve(_ZENODO_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)


def write_run_metrics(
    out_dir: str,
    model: str,
    tissue: str,
    mode: str,
    method: str,
    stage: str,
    test_pearson: float,
    val_pearson: Optional[float] = None,
    test_mse: Optional[float] = None,
    species: Optional[str] = None,
    checkpoint: str = "",
) -> str:
    """Write a normalized ``metrics.json`` for one run.

    Same schema as the committed ``results/plant_starrseq/reference/*.json`` so
    ``reproduce_plant_starrseq_table.py`` can read live-run outputs and reference
    numbers interchangeably. ``method`` is ``finetune`` or ``probe``.
    """
    import json

    os.makedirs(out_dir, exist_ok=True)
    rec = {
        "model": model,
        "tissue": tissue,
        "mode": mode,
        "method": method,
        "stage": stage,
        "test_pearson": None if test_pearson is None else round(float(test_pearson), 4),
        "val_pearson": None if val_pearson is None else round(float(val_pearson), 4),
        "test_mse": None if test_mse is None else float(test_mse),
        "species": species,
        "source_checkpoint": checkpoint,
        "source_commit": "",
    }
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return path


# ── Ridge probe helpers (pure numpy; shared by all cache-once probes) ─────────
# Kept torch-free here so the JAX-only NTv3 probe can use them without pulling in
# torch. plant_torch re-exports these for the PyTorch runners.


def ridge_fit(X, y, lam):
    """Closed-form ridge on centered features/targets; returns (w, xb, yb)."""
    xb = X.mean(0)
    yb = float(y.mean())
    Xc = X - xb
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1], dtype=np.float64)
    w = np.linalg.solve(A, Xc.T @ (y - yb))
    return w, xb, yb


def ridge_predict(X, w, xb, yb):
    return (X - xb) @ w + yb


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a).flatten(), np.asarray(b).flatten())[0, 1])


def select_ridge(Xtr, ytr, Xva, yva, lambdas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)):
    """Fit ridge for each lambda, pick the best on val Pearson. Returns (val_r, lam, w, xb, yb)."""
    best = None
    for lam in lambdas:
        w, xb, yb = ridge_fit(Xtr.astype(np.float64), ytr.astype(np.float64), lam)
        vr = pearson(ridge_predict(Xva, w, xb, yb), yva)
        if best is None or vr > best[0]:
            best = (vr, lam, w, xb, yb)
    return best


def data_is_present(data_path: str) -> bool:
    """True if all 8 built TSVs exist under ``data_path``."""
    for tissue in _TISSUES:
        for enh in _ENHANCER_CONDS:
            for part in ("train", "test"):
                p = os.path.join(data_path, f"jores21_{tissue}_{_ENH_TAG[enh]}_{part}.tsv")
                if not os.path.exists(p):
                    return False
    return True
