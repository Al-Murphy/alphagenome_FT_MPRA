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
4. Runs the model in batches (`model._predict`, with the same track
   filtering as `predict_sequence`) and extracts one or
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
#    over the center 501bp of the prediction track (AlphaGenome's scoring
#    window; the default). 384bp is the window used by
#    test_cagi5_zero_shot_base.py:
    python zeroshot_base_AG_mpra_predict.py \\
        --input my_mpra_test_set.tsv \\
        --output_types CAGE DNASE \\
        --ontology_curie EFO:0002067 \\
        --pooling center_window --center_window_bp 501

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
--standardize_length (4,096 bp; must be a multiple of 2048, 0 disables).
'N' one-hot encodes to the zero vector (no information). Pooling defaults to
`--pooling center_window` over the center --center_window_bp (501) bp; pass
`--center_window_bp 0` to instead use a per-sequence window equal to each
row's own pre-padding construct length, so padded flanks are never pooled.

NOTE ON MODEL VERSION (data leakage)
-------------------------------------
`all_folds` (the distilled model) was trained on the whole reference genome,
so for human/mouse genomic sequences (e.g. lentiMPRA elements) a good
correlation can be memorisation. Use `--model_version fold_0` (or another
fold) for a leakage-free number on genomic sequences; use `all_folds` for
out-of-distribution inputs (synthetic libraries, other species, variants).
Reporting both is recommended.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
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


def shift_sequence(seq: str, shift: int) -> str:
    """Shifts `seq` right by `shift` bp (left if negative), keeping its length.

    The vacated end is filled with 'N'. Raises if the shift would push real
    (non-'N') bases off the end, i.e. the construct must have at least
    |shift| bp of 'N' padding on the side it moves towards.
    """
    if shift == 0:
        return seq
    n = len(seq)
    if abs(shift) >= n:
        raise ValueError(f"|shift|={abs(shift)} >= sequence length {n}.")
    if shift > 0:
        dropped, kept = seq[n - shift :], seq[: n - shift]
        out = "N" * shift + kept
    else:
        dropped, kept = seq[:-shift], seq[-shift:]
        out = kept + "N" * (-shift)
    if dropped.strip("N"):
        raise ValueError(
            f"Shift of {shift} bp would push real sequence off the input;"
            " use a larger --standardize_length or a smaller --max_shift."
        )
    return out


def get_random_position_shifts(
    max_shift: int = 20,
    n_samples: int = 3,
    random_seed: int | None = None,
) -> list[int]:
    """Position shifts for test-time augmentation (mirrors the CAGI5 script).

    Returns [0] plus `n_samples` distinct shifts drawn from +-1..+-max_shift.
    Uses a local Generator, so the global NumPy RNG state is untouched and a
    given `random_seed` always reproduces the same shifts.
    """
    rng = np.random.default_rng(random_seed)
    candidates = [s for k in range(1, max_shift + 1) for s in (k, -k)]
    sampled = rng.choice(
        candidates, size=min(n_samples, len(candidates)), replace=False
    )
    return [0] + [int(x) for x in sampled]


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
    center: int | None = None,
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
    --standardize_length) should pass the *original*, pre-padding insert
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
    # `center` is the position the window/point is centred on. It defaults to
    # the middle of the array, but with position-shift augmentation the insert
    # moves, so callers pass the shifted centre to keep the window on the insert.
    mid = seq_len // 2 if center is None else int(np.clip(center, 0, seq_len - 1))
    if pooling == "center":
        return float(per_position[mid])
    if pooling == "center_window":
        window = center_window_bp if center_window_bp is not None else seq_len
        half = window // 2
        start = max(0, mid - half)
        end = min(seq_len, mid + half + (window % 2))
        return float(np.sum(per_position[start:end]))
    raise ValueError(f"Unrecognized pooling {pooling!r}")


def _run_model(
    model: dna_model.AlphaGenomeModel,
    sequences: list[str],
    *,
    organism: dna_model.Organism,
    output_types: tuple[dna_output.OutputType, ...],
    negative_strand: bool,
) -> dict[dna_output.OutputType, np.ndarray]:
    """Runs one batched forward pass; returns raw (B, S, n_tracks) arrays.

    The raw arrays still contain the 'Padding' tracks (NaN) -- callers must
    apply the track masks from `metadata_lib.create_track_masks`.

    With `negative_strand=True` the sequences are expected to already be
    reverse-complemented; `_predict` then flips the output back to the forward
    orientation and swaps the +/- strand tracks (via `strand_reindexing`), so
    forward and reverse-complement outputs are position- and strand-aligned.
    """
    one_hot_batch = np.stack(
        [np.asarray(model._one_hot_encoder.encode(seq)) for seq in sequences],
        axis=0,
    )
    organism_index = np.full(
        (len(sequences),), dna_model.convert_to_organism_index(organism), dtype=np.int32
    )
    batched_output = model._predict(
        model._params,
        model._state,
        one_hot_batch,
        organism_index,
        requested_outputs=output_types,
        negative_strand_mask=jnp.full(
            (one_hot_batch.shape[0],), negative_strand, dtype=bool
        ),
        strand_reindexing=jax.device_put(
            model._metadata[organism].strand_reindexing,
            model._device_context._device,
        ),
    )
    return {
        # Model outputs may be bfloat16; pool in float32 (matches predict_sequence,
        # which upcasts) to avoid accumulation error over thousands of positions.
        ot: np.asarray(batched_output[ot], dtype=np.float32)
        for ot in output_types
        if batched_output.get(ot) is not None
    }


def predict_batch(
    model: dna_model.AlphaGenomeModel,
    sequences: list[str],
    *,
    window_bps: list[int | None],
    organism: dna_model.Organism,
    output_types: Iterable[dna_output.OutputType],
    ontology_terms: list[ontology.OntologyTerm] | None,
    pooling: str,
    use_reverse_complement: bool = False,
    shifts: list[list[int]] | None = None,
) -> list[dict[str, float]]:
    """Runs the base model on a batch of equal-length sequences.

    Track selection goes through `metadata_lib.create_track_masks` -- the same
    call `predict_sequence` makes -- so 'Padding' (NaN) tracks are always
    dropped and the result is numerically identical to the per-sequence API.

    `window_bps[i]` is the center-window size for sequence i (`None` = whole
    array); it is only used when `pooling == 'center_window'`. With
    `use_reverse_complement` the forward and reverse-complement pooled values
    are averaged.

    `shifts[i]` lists the position shifts (bp) applied to sequence i; every
    sequence must have the same number of shifts. Pass k holds each sequence's
    k-th shift. Each shift is crossed with the strands and everything is
    averaged. The pooling window is centred on the *shifted* insert
    (`len // 2 + shift`), not on the middle of the track. Default: no shift.
    """
    lengths = {len(s) for s in sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"predict_batch requires every sequence to be the same length"
            f" (got lengths {lengths}) -- pad all sequences to the same"
            " length (--standardize_length) before calling this."
        )
    if len(window_bps) != len(sequences):
        raise ValueError(
            f"window_bps has {len(window_bps)} entries for {len(sequences)} sequences."
        )

    output_types = tuple(sorted(set(output_types), key=lambda t: t.name))
    track_masks = metadata_lib.create_track_masks(
        model._metadata[organism],
        requested_outputs=output_types,
        requested_ontologies=ontology_terms,
    )

    if shifts is None:
        shifts = [[0] for _ in sequences]
    if len(shifts) != len(sequences) or len({len(x) for x in shifts}) != 1:
        raise ValueError(
            "shifts must hold one list per sequence, all of the same length."
        )
    seq_len = len(sequences[0])

    # Each pass: (sequences fed to the model, negative strand?, per-seq centre).
    passes = []
    for k in range(len(shifts[0])):
        shifted = [shift_sequence(s, sh[k]) for s, sh in zip(sequences, shifts)]
        centers = [seq_len // 2 + sh[k] for sh in shifts]
        passes.append((shifted, False, centers))
        if use_reverse_complement:
            passes.append(([reverse_complement(s) for s in shifted], True, centers))

    pooled: dict[str, list[list[float]]] = {
        ot.name: [[] for _ in sequences] for ot in output_types
    }
    for strand_seqs, negative, centers in passes:
        raw = _run_model(
            model,
            strand_seqs,
            organism=organism,
            output_types=output_types,
            negative_strand=negative,
        )
        for output_type, values in raw.items():
            mask = track_masks.get(output_type)
            if mask is None:
                continue
            values = values[..., mask]
            for idx, window_bp in enumerate(window_bps):
                pooled[output_type.name][idx].append(
                    pool_track(
                        values[idx],
                        pooling=pooling,
                        center_window_bp=window_bp,
                        center=centers[idx],
                    )
                )

    return [
        {
            name: float(np.mean(per_seq[idx])) if per_seq[idx] else float("nan")
            for name, per_seq in pooled.items()
        }
        for idx in range(len(sequences))
    ]


def validate_standardize_length(length: int) -> None:
    """AlphaGenome needs the input length to be a multiple of 2048 bp
    (0 disables padding/cropping)."""
    if length < 0 or length % 2048 != 0:
        raise ValueError(
            f"--standardize_length must be 0 (disabled) or a positive multiple"
            f" of 2048 (e.g. 2048, 4096, 16384); got {length}."
        )


def assemble_constructs(
    inserts: Iterable[str],
    *,
    template: str,
    left_adapter: str | None,
    right_adapter: str | None,
    promoter: str | None,
    barcode: str | None,
    standardize_length_bp: int,
) -> tuple[list[str], list[int]]:
    """Builds the constructs and returns (padded constructs, pre-padding lengths)."""
    constructs, pre_padding_lengths = [], []
    for seq in inserts:
        construct = build_construct(
            seq.strip().upper(),
            left_adapter=left_adapter,
            right_adapter=right_adapter,
            promoter=promoter,
            barcode=barcode,
            template=template,
        )
        pre_padding_lengths.append(len(construct))
        if standardize_length_bp:
            construct = standardize_length(construct, standardize_length_bp)
        constructs.append(construct)
    return constructs, pre_padding_lengths


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
        help="e.g. 'all_folds' or 'fold_0'. Ignored when --source local."
        " Use fold_0 (leakage-free) for human/mouse genomic sequences and"
        " all_folds for out-of-distribution inputs; see module docstring.",
    )
    parser.add_argument(
        "--local_checkpoint_dir",
        type=str,
        default=None,
        help="Path to a local AlphaGenome checkpoint"
        " directory. Required when --source local.",
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="human",
        choices=["human", "mouse"],
        help="Organism whose tracks/metadata are used.",
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
        help="Used only when --pooling center_window. Default 501"
        " (AlphaGenome's scoring window). Use 0 for auto: each"
        " sequence's own pre-padding construct length, so padding"
        " added by --standardize_length is excluded. 384 is the window"
        " used by test_cagi5_zero_shot_base.py.",
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
        " Default: 4096. Must be a multiple of 2048 (the model's"
        " input-length requirement); short MPRA inserts"
        " (a few hundred bp) cannot be fed directly."
        " Set to 0 to disable padding entirely (only valid if"
        " every construct already has the same length and that"
        " length is a multiple of 2048). 'N' padding one-hot"
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
        help="Fixed minimal-promoter sequence placed where"
        " --construct_template puts it (default: after the insert,"
        " as in the LentiMPRA construct in data.py).",
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
        help="Average forward-strand and reverse-complement predictions per"
        " sequence (the reverse-complement output is flipped back and"
        " strand-swapped so both are aligned). Doubles runtime.",
    )

    parser.add_argument(
        "--use_position_shift",
        action="store_true",
        help="Test-time augmentation: also score the construct shifted by"
        " random +-1..--max_shift bp (plus the unshifted one), crossed with"
        " strands if --reverse_complement_avg, and average. The pooling"
        " window follows the shifted insert. Needs N padding of at least"
        " --max_shift bp on each side (see --standardize_length).",
    )
    parser.add_argument("--max_shift", type=int, default=20)
    parser.add_argument(
        "--n_shift_samples",
        type=int,
        default=3,
        help="Number of random shifts per sequence (in addition to shift 0).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed for shift sampling. Row i uses seed+i, so results are"
        " reproducible and independent of --batch_size.",
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

    organism = {
        "human": dna_model.Organism.HOMO_SAPIENS,
        "mouse": dna_model.Organism.MUS_MUSCULUS,
    }[args.organism]
    try:
        validate_standardize_length(args.standardize_length)
    except ValueError as e:
        parser.error(str(e))
    if args.center_window_bp < 0:
        parser.error("--center_window_bp must be >= 0 (0 = auto).")
    if args.use_position_shift and (args.max_shift < 1 or args.n_shift_samples < 1):
        parser.error("--max_shift and --n_shift_samples must be >= 1.")
    if args.use_position_shift and not args.standardize_length:
        parser.error("--use_position_shift requires --standardize_length > 0.")
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
    constructs, pre_padding_lengths = assemble_constructs(
        df[args.sequence_col].astype(str),
        template=args.construct_template,
        left_adapter=args.left_adapter,
        right_adapter=args.right_adapter,
        promoter=args.promoter,
        barcode=args.barcode,
        standardize_length_bp=args.standardize_length,
    )
    df = df.copy()
    df["_construct"] = constructs
    df["_pre_padding_length"] = pre_padding_lengths
    df["_construct_length"] = [len(c) for c in constructs]

    if args.use_position_shift:
        room = (df["_construct_length"] - df["_pre_padding_length"]).min() // 2
        if args.max_shift > room:
            parser.error(
                f"--max_shift {args.max_shift} exceeds the N padding available"
                f" on each side ({room} bp) for the longest construct; raise"
                " --standardize_length or lower --max_shift."
            )

    if not args.standardize_length:
        lens = df["_construct_length"]
        if lens.nunique() > 1 or (lens % 2048 != 0).any():
            parser.error(
                "--standardize_length 0 requires all constructs to have the"
                " same length, a multiple of 2048; got lengths"
                f" {sorted(lens.unique())[:5]}."
            )

    # -------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------
    center_window_desc = (
        f"={args.center_window_bp}bp"
        if args.center_window_bp
        else "=auto (each sequence's own pre-padding length)"
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
        window_bps = [
            args.center_window_bp or pre_padding_lengths[i]
            for i in range(start, start + len(batch))
        ]
        if args.use_position_shift:
            shifts = [
                get_random_position_shifts(
                    args.max_shift, args.n_shift_samples, args.random_seed + i
                )
                for i in range(start, start + len(batch))
            ]
        else:
            shifts = None
        batch_preds = predict_batch(
            model,
            batch,
            window_bps=window_bps,
            organism=organism,
            output_types=output_types,
            ontology_terms=ontology_terms,
            pooling=args.pooling,
            use_reverse_complement=args.reverse_complement_avg,
            shifts=shifts,
        )
        records.extend(batch_preds)

    print()
    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s ({total_time / len(df) * 1000:.1f} ms/sequence)")

    assert len(records) == len(df), (len(records), len(df))
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
