#!/usr/bin/env python3
"""
Zero-shot evaluation of the BASE (pretrained, non-fine-tuned) AlphaGenome model
on an MPRA-style dataset of short reporter sequences.

WHAT THIS SCRIPT DOES
----------------------
1. Loads the pretrained AlphaGenome model (Kaggle or Hugging Face weights,
   your choice), with NO custom heads and NO fine-tuning.
2. Reads your MPRA test set (a CSV/TSV with a sequence column and a score
   column).
3. Optionally assembles a full reporter construct (insert + promoter +
   barcode / adapters) if you know your assay's vector design -- off by
   default, in which case your sequence is used exactly as given.
4. Runs `model.predict_sequence(...)` or `model._predict(...)` on each construct and extracts one or
   more genomic-track outputs (CAGE / RNA_SEQ / DNASE / ATAC / PROCAP) for the
   cell type(s) / ontology term(s) you specify.
5. Pools each track's per-base-pair signal into a single scalar per sequence
   (sum / mean / max / center-window / whole-sequence), exactly analogous to
   the pooling used by the fine-tuned `MPRAHead`/`MPRAOracle` classes in
   `alphagenome_ft_mpra`.
6. Correlates the pooled zero-shot prediction against your experimental score
   (Pearson r, Spearman rho) for every (output_type, ontology) combination you
   asked for, so you can see which readout is the best zero-shot proxy for
   your assay.
7. Saves per-sequence predictions + a ranked summary table, and (optionally)
   a scatter plot of the best-performing combination.


USAGE EXAMPLES
--------------
# 1. See what cell types / ontology terms exist for a given output type
#    (useful before you pick --ontology_curie values):
    python zeroshot_base_AG_mpra_predict.py --list_ontologies --output_types CAGE

# 2. Basic zero-shot scan across multiple readouts, no cell-type restriction
#    (averages over ALL tracks of each output type -- a generic, pan-context
#    proxy; good first pass when you don't know which cell type matters):
    python zeroshot_base_AG_mpra_predict.py \\
        --input my_mpra_test_set.tsv \\
        --sequence_col sequence --score_col score \\
        --output_types CAGE RNA_SEQ DNASE

# 3. Restrict to a specific cell type (e.g. K562 CAGE + DNase), and pool
#    over the center 384bp of the prediction track (AlphaGenome's typical
#    scoring window, see test_cagi5_zero_shot_base.py):
    python zeroshot_base_AG_mpra_predict.py \\
        --input my_mpra_test_set.tsv \\
        --output_types CAGE DNASE \\
        --ontology_curie EFO:0002067 \\
        --pooling center_window --center_window_bp 384

# 4. If, for example, your 600bp inserts need a fixed downstream minimal-promoter +
#    barcode appended to mimic your real reporter vector before scoring
#    (like LentiMPRA's `promoter_seq` + `rand_barcode` in
#    alphagenome_ft_mpra/data.py):
    python zeroshot_base_AG_mpra_predict.py \\
        --input my_mpra_test_set.tsv \\
        --promoter TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG \\
        --barcode AGAGACTGAGGCCAC \\
        --output_types RNA_SEQ

# 5. Quick smoke-test on the first 20 rows only, on CPU:
    python zeroshot_base_AG_mpra_predict.py --input my_mpra_test_set.tsv \\
        --max_sequences 20 --device cpu

# 6. A STARR-Seq example:
    python zeroshot_base_AG_mpra_predict.py \\
    --input my_mpra_test_set.tsv \\
            --promoter TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG \\
            --barcode AGAGACTGAGGCCAC \\
            --left_adapter CCCGTCCGAACTCCGAACCCCAGAACAGAGCAAAGCCTCCTCGGCCTCCCTGTCCCCAGCCTTCCCCG \\
            --output_types RNA_SEQ \\
            --construct_template "{promoter}{left_adapter}{insert}{right_adapter}{barcode}"

NOTE ON SEQUENCE LENGTH
------------------------
By default this script center-pads every construct with 'N' up to
MIN_SAFE_SEQUENCE_LENGTH (4,096 bp) before calling `predict_sequence`. 'N' one-hot encodes
to the zero vector (no information), and pooling defaults to
`--pooling center_window` with an automatic per-sequence window matching
each row's own original (pre-padding) length -- so the padded flanks are
never included in the pooled score.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
import traceback
import string

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import jax
import jax.numpy as jnp

# --- AlphaGenome imports -----------------------------------------------
# NOTE: `alphagenome` (the base API/typing/client package) and
# `alphagenome_research` (this model repo) only. No `alphagenome_ft`.
from alphagenome.data import ontology
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

from alphagenome_ft_mpra.model_utils import load_model

# MIN_SAFE_SEQUENCE_LENGTH = 4096

SUPPORTED_OUTPUT_TYPES = {
    dna_output.OutputType.CAGE,
    dna_output.OutputType.RNA_SEQ,
    dna_output.OutputType.DNASE,
    dna_output.OutputType.ATAC,
    dna_output.OutputType.PROCAP,
}


# ---------------------------------------------------------------------------
# Sequence construction helpers
# ---------------------------------------------------------------------------


def reverse_complement(seq: str) -> str:
    """Reverse complements a DNA sequence string (N-safe)."""
    comp = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N",
        "a": "t",
        "t": "a",
        "c": "g",
        "g": "c",
        "n": "n",
    }
    return "".join(comp.get(b, "N") for b in reversed(seq))


def standardize_length(seq: str, target_length: int) -> str:
    """Center-pads (with 'N') or center-crops `seq` to exactly `target_length`.

    Mirrors `alphagenome_ft_mpra/episomal_utils.py:standardize_to_sequence_length`
    so that behavior here matches the rest of the codebase. Padding with 'N'
    is safe: AlphaGenome's one-hot encoder maps any non-ACGT character to the
    zero vector, i.e. "no sequence information here".
    """
    n = len(seq)
    if n == target_length:
        return seq
    if n < target_length:
        diff = target_length - n
        left = diff // 2
        right = diff - left
        return ("N" * left) + seq + ("N" * right)
    # n > target_length: center crop.
    start = (n - target_length) // 2
    return seq[start : start + target_length]


def build_construct(
    insert: str,
    *,
    left_adapter: str | None,
    right_adapter: str | None,
    promoter: str | None,
    barcode: str | None,
    template: str = "{left_adapter}{insert}{right_adapter}{promoter}{barcode}",
) -> str:
    """Assembles a full reporter construct around a variable insert.

    With all of left_adapter/right_adapter/promoter/barcode set to None
    (the default), this is the identity function -- the insert is scored
    exactly as provided. Set these if you know your MPRA vector's fixed
    flanking elements and want to present AlphaGenome with a more realistic
    construct.
    """

    values = {
        "insert": insert,
        "left_adapter": left_adapter or "",
        "right_adapter": right_adapter or "",
        "promoter": promoter or "",
        "barcode": barcode or "",
    }

    field_names = {name for _, name, _, _ in string.Formatter().parse(template) if name}

    unknown = field_names - set(values)
    if unknown:
        raise ValueError(
            f"Unknown placeholder(s) {sorted(unknown)} in --construct_template."
            f" Valid placeholders: {sorted(values)}."
        )
    if "insert" not in field_names:
        raise ValueError(
            "--construct_template must include '{insert}' -- the variable"
            " sequence being scored can't be silently omitted."
        )

    return template.format(**values)


# ---------------------------------------------------------------------------
# Ontology inspection
# ---------------------------------------------------------------------------


def list_ontologies(
    output_types: Iterable[dna_output.OutputType],
    organism: dna_model.Organism,
) -> None:
    metadata = metadata_lib.load(organism)
    for output_type in output_types:
        df = metadata.get(output_type)
        if df is None:
            print(f"\n[{output_type.name}] not available for {organism.name}.")
            continue
        print(f"\n[{output_type.name}] {len(df)} tracks total.")
        if "ontology_curie" not in df:
            print("  (tissue/ontology agnostic output type)")
            continue
        sub_cols = [c for c in ("ontology_curie", "name", "biosample_name") if c in df]
        uniq = df[sub_cols].drop_duplicates(subset=["ontology_curie"])
        with pd.option_context("display.max_rows", 200, "display.width", 160):
            print(uniq.sort_values("ontology_curie").to_string(index=False))


def validate_ontology_curies(
    curies: list[str],
    model: dna_model.AlphaGenomeModel,
    output_types: list[dna_output.OutputType],
    organism: dna_model.Organism,
) -> None:
    """Validates the ontology curies. A CURIE only needs to match ONE output type
    to be valid (e.g. a CAGE-specific curie alongside an RNA_SEQ-specific
    one in the same --ontology_curie list is a legitimate, intentional use)."""
    metadata = model._metadata[organism]  # pylint: disable=protected-access
    valid_curies = set()
    for output_type in output_types:
        df = metadata.get(output_type)
        if df is not None and "ontology_curie" in df:
            valid_curies.update(df["ontology_curie"])

    bad_curies = [c for c in curies if c not in valid_curies]
    if bad_curies:
        raise ValueError(
            f"--ontology_curie value(s) {bad_curies} match zero tracks for"
            f" output types {[t.name for t in output_types]} and organism"
            f" {organism.name}. Use --list_ontologies to see valid CURIEs for"
            " these output types before retrying."
        )


# ---------------------------------------------------------------------------
# Prediction + pooling
# ---------------------------------------------------------------------------


def pool_track(
    values: np.ndarray,
    *,
    pooling: str,
    center_window_bp: int | None,
) -> float:
    """Pools a (seq_len, n_tracks) prediction array to a single scalar.

    `values` is averaged across tracks first (e.g. multiple CAGE assays
    matching the same ontology term), then pooled across sequence positions.

    NOTE ON STRAND: for strand-specific output types (CAGE, PROCAP, RNA_SEQ),
    a single cell type/ontology term can match BOTH a '+' strand track and a
    '-' strand track -- e.g. K562 CAGE resolves to two tracks, one per
    strand. This function's track-averaging step does not distinguish them:
    sense and antisense signal get averaged together into one number. This
    can mask strongly directional promoter activity (e.g. high '+', near-
    zero '-' averaging to a misleadingly moderate value indistinguishable
    from a weak bidirectional promoter). If strand matters for your
    interpretation, inspect --list_ontologies' track metadata for the
    output types you're using to check how many tracks match per ontology
    term, and consider whether averaging them is appropriate for your case.

    `center_window_bp=None` with `pooling='center_window'` means "use the
    whole `values` array as the window" -- callers that pad sequences (see
    MIN_SAFE_SEQUENCE_LENGTH) should pass the *original*, pre-padding insert
    length here so the window covers just the real sequence and not the
    flanking padding.
    """
    if values.size == 0:
        return float("nan")

    # Average across tracks (e.g. replicate assays for the same cell type).
    per_position = values.mean(axis=-1)  # (seq_len,)
    seq_len = per_position.shape[0]

    if pooling == "sum":
        return float(np.sum(per_position))
    if pooling == "mean":
        return float(np.mean(per_position))
    if pooling == "max":
        return float(np.max(per_position))
    if pooling == "center":
        return float(per_position[seq_len // 2])
    if pooling == "center_window":
        window = center_window_bp if center_window_bp is not None else seq_len
        half = window // 2
        start = max(0, seq_len // 2 - half)
        end = min(seq_len, seq_len // 2 + half + (window % 2))
        return float(np.sum(per_position[start:end]))
    raise ValueError(f"Unrecognized pooling {pooling!r}")


def predict_one_sequence(
    model: dna_model.AlphaGenomeModel,
    sequence: str,
    *,
    organism: dna_model.Organism,
    output_types: list[dna_output.OutputType],
    ontology_terms: list[ontology.OntologyTerm] | None,
    use_reverse_complement: bool,
    pooling: str,
    center_window_bp: int | None,
) -> dict[str, float]:
    """Runs the base model on one sequence, returns {output_type_name: scalar}."""
    strands = [sequence]
    if use_reverse_complement:
        strands.append(reverse_complement(sequence))

    per_output_values: dict[str, list[float]] = {ot.name: [] for ot in output_types}

    for strand_seq in strands:
        output = model.predict_sequence(
            strand_seq,
            organism=organism,
            requested_outputs=output_types,
            ontology_terms=ontology_terms,
        )
        for output_type in output_types:
            track_output = getattr(output, output_type.name.lower())
            if track_output is None:
                continue
            pooled = pool_track(
                np.asarray(track_output.values),
                pooling=pooling,
                center_window_bp=center_window_bp,
            )
            per_output_values[output_type.name].append(pooled)

    return {
        name: float(np.mean(vals)) if vals else float("nan")
        for name, vals in per_output_values.items()
    }


def predict_batch(
    model: dna_model.AlphaGenomeModel,
    sequences: list[str],
    *,
    organism: dna_model.Organism,
    output_types: Iterable[dna_output.OutputType],
    ontology_terms: list[ontology.OntologyTerm] | None,
    pooling: str,
    center_window_bp: int | None,
) -> list[dict[str, float]]:
    """Runs the base model on a batch of sequences. (NOTE: Strands are currently ignored)"""
    lengths = {len(s) for s in sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"predict_batch requires every sequence to be the same length"
            f" (got lengths {lengths}) -- pad all sequences to the same"
            " --pad_length before calling this."
        )

    output_types = tuple(set(output_types))
    window_bps = [
        center_window_bp if center_window_bp is not None else len(seq.strip("N"))
        for seq in sequences
    ]

    one_hot_batch = np.stack(
        [np.asarray(model._one_hot_encoder.encode(seq)) for seq in sequences], axis=0
    )

    organism_idx = 0 if organism == dna_model.Organism.HOMO_SAPIENS else 1
    organism_index = np.full((len(sequences),), organism_idx, dtype=np.int32)

    batched_output = model._predict(
        model._params,
        model._state,
        one_hot_batch,
        organism_index,
        requested_outputs=output_types,
        negative_strand_mask=jnp.zeros(one_hot_batch.shape[0], dtype=bool),
        strand_reindexing=jax.device_put(
            model._metadata[organism].strand_reindexing, model._device_context._device
        ),
    )

    results = []

    for idx, window_bp in enumerate(window_bps):
        per_seq_result = {}
        for output_type in output_types:
            track_output = batched_output.get(output_type)
            if track_output is None:
                per_seq_result[output_type.name] = float("nan")
                continue
            values_idx = np.asarray(track_output)[idx]
            mask = ontology_mask(model, organism, output_type, ontology_terms)
            if mask is not None:
                values_idx = values_idx[..., mask]
            pooled = pool_track(
                values_idx,
                pooling=pooling,
                center_window_bp=window_bp,
            )
            per_seq_result[output_type.name] = pooled
        results.append(per_seq_result)
    return results


def ontology_mask(model, organism, output_type, ontology_terms):
    if ontology_terms is None:
        return None
    df = model._metadata[organism].get(output_type)
    if df is None or "ontology_curie" not in df:
        return None
    requested = {o.ontology_curie for o in ontology_terms}
    return np.array([c in requested for c in df["ontology_curie"]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CSV/TSV with sequence + score columns."
        " Not required with --list_ontologies.",
    )
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--score_col", type=str, default="score")

    parser.add_argument(
        "--source",
        type=str,
        default="kaggle",
        choices=["kaggle", "huggingface", "local"],
        help="'local' loads a checkpoint directory you"
        " already downloaded with NO"
        " Kaggle/Hugging Face login or network access"
        " -- use this on HPC compute nodes.",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="all_folds",
        help="e.g. 'all_folds' or 'fold_0'. Ignored when --source local.",
    )
    parser.add_argument(
        "--local_checkpoint_dir",
        type=str,
        default=None,
        help="Path to a local AlphaGenome checkpoint"
        " directory. Required when --source local.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "gpu", "tpu"],
        help="Force a device; default auto-detects GPU/TPU"
        " and falls back to CPU with a warning.",
    )

    parser.add_argument(
        "--output_types",
        type=str,
        nargs="+",
        default=["CAGE", "RNA_SEQ", "DNASE"],
        help="One or more of: "
        + ", ".join(t.name for t in SUPPORTED_OUTPUT_TYPES)
        + ". Note: CAGE/PROCAP/RNA_SEQ are strand-specific"
        " -- sense and antisense tracks are averaged"
        " together with no strand separation; see"
        " pool_track docstring.",
    )
    parser.add_argument(
        "--ontology_curie",
        type=str,
        nargs="+",
        default=None,
        help="e.g. EFO:0002067 (K562), EFO:0001187 (HepG2)."
        " If omitted, all tracks of each output type are"
        " averaged together (a generic, cell-type-agnostic"
        " proxy). Use --list_ontologies to browse options.",
    )
    parser.add_argument(
        "--list_ontologies",
        action="store_true",
        help="Print available ontology terms for --output_types and exit.",
    )

    parser.add_argument(
        "--pooling",
        type=str,
        default="center_window",
        choices=["sum", "mean", "max", "center", "center_window"],
    )
    parser.add_argument(
        "--center_window_bp",
        type=int,
        default=501,
        help="Used only when --pooling center_window. Default"
        " (None): auto -- use each sequence's own"
        " original (pre-padding, pre-adapter) length, so"
        " padding added by --standardize_length is"
        " excluded from the pooled score. Set explicitly"
        " to pool a fixed window instead (e.g. 384, the"
        " window used by test_cagi5_zero_shot_base.py).",
    )
    parser.add_argument(
        "--input_length", type=int, default=4096, help="The input length for AG model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="For making batched predictions"
    )

    parser.add_argument(
        "--standardize_length",
        type=int,
        default=4096,
        help="Center-pads (with 'N') or center-crops every"
        " construct to this many bp before scoring."
        f" Default: 4096 --"
        " AlphaGenome's pairwise attention machinery"
        " needs a minimum input length to run at all;"
        " feeding short MPRA inserts (a few hundred bp)"
        " directly raises a JAX shape-broadcast error."
        " Set to 0 to disable padding entirely (only"
        " safe if your sequences are already long"
        " enough on their own). 'N' padding one-hot"
        " encodes to the zero vector, i.e. no sequence"
        " information, so it does not bias the"
        " prediction -- just pool with"
        " --pooling center_window (the default) so the"
        " padded flanks are excluded from the score.",
    )
    parser.add_argument("--left_adapter", type=str, default=None)
    parser.add_argument("--right_adapter", type=str, default=None)
    parser.add_argument(
        "--promoter",
        type=str,
        default=None,
        help="Fixed minimal-promoter sequence to append, if"
        " your assay places one downstream of the insert"
        " (see LentiMPRA construct in data.py).",
    )
    parser.add_argument("--barcode", type=str, default=None)
    parser.add_argument(
        "--construct_template",
        type=str,
        default="{left_adapter}{insert}{right_adapter}{promoter}{barcode}",
        help="Format-string controlling MPRA construct layout, using named"
        " placeholders {insert} (required), {left_adapter},"
        " {right_adapter}, {promoter}, {barcode}. Default matches"
        " lentiMPRA (fixed promoter appended after the insert). For"
        " STARR-seq (insert cloned downstream of the promoter), use e.g."
        " '{promoter}{insert}{barcode}'.",
    )

    parser.add_argument(
        "--reverse_complement_avg",
        action="store_true",
        help="Average forward-strand and reverse-complement predictions per sequence.",
    )

    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Only score the first N rows (smoke testing).",
    )
    parser.add_argument("--output_dir", type=str, default="./results/zero_shot_mpra")
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Save a scatter plot for the best-performing"
        " (output_type, metric) combination.",
    )

    parser.add_argument("--log_transform", action="store_true", default=True)
    parser.add_argument(
        "--no_log_transform",
        dest="log_transform",
        action="store_false",
        help="Disable log-transforming the pooled prediction before"
        " computing Pearson r. On by default.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Printing all arguments
    # -------------------------------------------------------------------

    print("=" * 72)
    print("Arguments:")
    max_len = max(len(k) for k in vars(args))
    for arg, value in sorted(vars(args).items()):
        print(f"  --{arg:<{max_len}} : {value}")
    print("=" * 72)

    if args.source == "local" and not args.local_checkpoint_dir:
        parser.error("--local_checkpoint_dir is required when --source local")

    organism = dna_model.Organism.HOMO_SAPIENS
    output_types = [dna_output.OutputType[name.upper()] for name in args.output_types]
    bad_types = [ot for ot in output_types if ot not in SUPPORTED_OUTPUT_TYPES]
    if bad_types:
        parser.error(
            f"--output_types {[t.name for t in bad_types]} not supported: "
            "pool_track assumes 1bp-resolution tracks. Supported types: "
            f"{[t.name for t in SUPPORTED_OUTPUT_TYPES]}."
        )

    if args.list_ontologies:
        list_ontologies(output_types, organism)
        return

    ontology_terms = (
        [ontology.from_curie(c) for c in args.ontology_curie]
        if args.ontology_curie
        else None
    )

    model = load_model(
        args.source,
        args.model_version,
        args.device,
        local_checkpoint_dir=args.local_checkpoint_dir,
    )

    if args.ontology_curie:
        validate_ontology_curies(args.ontology_curie, model, output_types, organism)

    if args.list_ontologies:
        list_ontologies(model, output_types, organism)
        return

    if args.input is None:
        parser.error("--input is required unless --list_ontologies is set.")

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    input_path = Path(args.input)
    sep = "\t" if input_path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(input_path, sep=sep)
    for col in (args.sequence_col, args.score_col):
        if col not in df.columns:
            raise ValueError(
                f"Column {col!r} not found in {args.input}. Available columns:"
                f" {list(df.columns)}"
            )
    if args.max_sequences is not None:
        df = df.head(args.max_sequences).reset_index(drop=True)

    print(f"Loaded {len(df)} sequences from {args.input}")
    lengths = df[args.sequence_col].str.len()
    print(
        f"Sequence length: min={lengths.min()} max={lengths.max()} "
        f"mean={lengths.mean():.1f}"
    )

    # -------------------------------------------------------------------
    # Assemble constructs
    # -------------------------------------------------------------------
    constructs = []
    pre_padding_lengths = []
    for seq in df[args.sequence_col].astype(str):
        seq = seq.strip().upper()
        construct = build_construct(
            seq,
            left_adapter=args.left_adapter,
            right_adapter=args.right_adapter,
            promoter=args.promoter,
            barcode=args.barcode,
        )
        pre_padding_lengths.append(len(construct))
        if args.standardize_length:
            construct = standardize_length(construct, args.standardize_length)
        constructs.append(construct)
    df = df.copy()
    df["_construct"] = constructs
    df["_pre_padding_length"] = pre_padding_lengths
    df["_construct_length"] = [len(c) for c in constructs]

    if (
        not args.standardize_length
        and (df["_pre_padding_length"] < args.input_length).any()
    ):
        print(
            f"WARNING: --standardize_length is disabled and some constructs are"
            f" shorter than {args.input_length}bp. AlphaGenome's pairwise"
            " attention machinery requires a minimum input length; short raw"
            " sequences WILL raise a JAX shape-broadcast error in"
            " predict_sequence. Re-enable padding (the default) unless you are"
            " sure every sequence is already long enough.",
            file=sys.stderr,
        )
    elif df["_construct_length"].nunique() > 1:
        print(
            f"WARNING: constructs have {df['_construct_length'].nunique()}"
            " distinct lengths. JAX will recompile for each new shape, which"
            " is slow -- consider --standardize_length.",
            file=sys.stderr,
        )

    # -------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------
    center_window_desc = (
        f"={args.center_window_bp}bp"
        if args.center_window_bp is not None
        else "=auto (each sequence's own original length)"
    )
    print(
        f"\nRunning zero-shot AlphaGenome predictions for output types:"
        f" {[t.name for t in output_types]}"
        f" | ontology: {args.ontology_curie or 'ALL (averaged)'}"
        f" | pooling: {args.pooling}"
        f"{center_window_desc if args.pooling == 'center_window' else ''}"
    )

    records = []
    t0 = time.time()
    constructs = df["_construct"].tolist()
    n_batches = (len(constructs) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(n_batches), desc="Predicting"):
        start = batch_idx * args.batch_size
        batch = constructs[start : start + args.batch_size]
        try:
            batch_preds = predict_batch(
                model,
                batch,
                organism=organism,
                output_types=output_types,
                ontology_terms=ontology_terms,
                pooling=args.pooling,
                center_window_bp=args.center_window_bp,
            )
        except Exception as e:
            print(
                f"\n  WARNING: prediction failed for batch {batch_idx}: {traceback.print_exc()}"
            )
            batch_preds = []
            row_center_window_bp = (
                args.center_window_bp
                if args.center_window_bp is not None
                else int(df["_pre_padding_length"].iloc[i])
            )
            for i, construct in tqdm(
                enumerate(df["_construct"]), desc="Predicting", total=len(df)
            ):
                try:
                    preds = predict_one_sequence(
                        model,
                        construct,
                        organism=organism,
                        output_types=output_types,
                        ontology_terms=ontology_terms,
                        use_reverse_complement=args.reverse_complement_avg,
                        pooling=args.pooling,
                        center_window_bp=row_center_window_bp,
                    )
                except Exception as e:  # pylint: disable=broad-except
                    print(
                        f"\n  WARNING: prediction failed for row {i}: {traceback.print_exc()}"
                    )
                    preds = {ot.name: float("nan") for ot in output_types}
            batch_preds.append(preds)
        records.extend(batch_preds)

    print()
    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s ({total_time / len(df) * 1000:.1f} ms/sequence)")

    pred_df = pd.DataFrame(records)
    out_df = df.drop(columns=["_construct"]).join(pred_df)

    # -------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------
    score = out_df[args.score_col].astype(float).values
    metric_rows = []
    for output_type in output_types:
        pred = out_df[output_type.name].values
        valid = ~(np.isnan(pred) | np.isnan(score))
        n_valid = int(valid.sum())
        if n_valid < 3:
            pearson = spearman = float("nan")
        else:
            pred_for_corr = np.log1p(pred) if args.log_transform else pred
            pearson, _ = pearsonr(pred_for_corr[valid], score[valid])
            spearman, _ = spearmanr(pred_for_corr[valid], score[valid])
        metric_rows.append(
            {
                "output_type": output_type.name,
                "ontology": (
                    ",".join(args.ontology_curie) if args.ontology_curie else "ALL"
                ),
                "pooling": args.pooling,
                "n_valid": n_valid,
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
        )
    metrics_df = pd.DataFrame(metric_rows).sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False
    )

    print("\n" + "=" * 72)
    print("Zero-shot AlphaGenome vs. experimental MPRA score")
    print("=" * 72)
    print(metrics_df.to_string(index=False))

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics_summary.csv"
    out_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved per-sequence predictions to {pred_path}")
    print(f"Saved metrics summary to {metrics_path}")

    if args.save_plot and len(metrics_df) > 0:
        best = metrics_df.iloc[0]
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 5))
            x = out_df[best["output_type"]].values
            y = score
            ax.scatter(x, y, s=10, alpha=0.5)
            ax.set_xlabel(f"Predicted {best['output_type']} (zero-shot, pooled)")
            ax.set_ylabel(args.score_col)
            ax.set_title(
                f"Pearson r={best['pearson_r']:.3f}, "
                f"Spearman rho={best['spearman_rho']:.3f}"
            )
            fig.tight_layout()
            plot_path = out_dir / f"best_{best['output_type']}_scatter.png"
            fig.savefig(plot_path, dpi=150)
            print(f"Saved scatter plot to {plot_path}")
        except ImportError:
            print("matplotlib not installed; skipping --save_plot.", file=sys.stderr)


if __name__ == "__main__":
    main()
