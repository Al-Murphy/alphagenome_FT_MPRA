"""The lentiMPRA reporter construct must be read on the reporter strand.

Supplementary Table S13 of Agarwal et al. 2025 writes the library's reverse
primers 5'->3' on the strand OPPOSITE the reporter, so they have to be
reverse-complemented before they describe the construct presented to the model.
Copying a reverse primer's tail verbatim (what the pipeline did before this was
corrected) repeats the right adapter and runs the minimal promoter backwards.

These tests re-derive the corrected sequence from the primers so the constants
cannot drift, and pin the legacy suffix so pre-correction checkpoints stay
reproducible.
"""

import pytest

from alphagenome_ft_mpra.data import (
    LENTIMPRA_DOWNSTREAM,
    LENTIMPRA_SUFFIX,
    LENTIMPRA_SUFFIX_LEGACY,
)

# The two reverse primers of Supplementary Table S13 that build everything
# downstream of the insert, verbatim and in the orientation the table prints.
_MINP_ENH_R = "ACAGTACCGGATTGCCAAGCTGGAAGTCGAGCTTCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
_BC_PRIMER_R = (
    "CTCACCATGGTGGCGACCGGTNNNNNNNNNNNNNNNCTTAGCTTTCGCTTAGCGATGTGTTCACTTTGCA"
    "CAGTACCGGATTGC"
)

RIGHT_ADAPTER = "CATTGCGTGAACCGA"
INPUT_LENGTH = 281
ELEMENT_LENGTH = 230  # left adapter + 200 bp CRE + right adapter, i.e. the `seq` column


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def test_downstream_is_the_primers_read_on_the_reporter_strand():
    first = _reverse_complement(_MINP_ENH_R)
    # minP-enh-r anneals to the right adapter, which is why copying its tail
    # verbatim duplicated the adapter.
    assert first.startswith(RIGHT_ADAPTER)

    second = _reverse_complement(_BC_PRIMER_R)
    overlap = next(k for k in range(len(first), 14, -1) if first.endswith(second[:k]))
    derived = (first + second[overlap:])[len(RIGHT_ADAPTER):]

    assert derived.startswith(LENTIMPRA_DOWNSTREAM)
    assert len(LENTIMPRA_DOWNSTREAM) == 102
    assert LENTIMPRA_DOWNSTREAM.endswith("N" * 15)
    assert "N" not in LENTIMPRA_DOWNSTREAM[:-15]


def test_downstream_runs_into_the_egfp_start_codon():
    """Orientation check independent of the primers: past the barcode the
    reporter strand must read AgeI + Kozak + ATG."""
    first = _reverse_complement(_MINP_ENH_R)
    second = _reverse_complement(_BC_PRIMER_R)
    overlap = next(k for k in range(len(first), 14, -1) if first.endswith(second[:k]))
    derived = (first + second[overlap:])[len(RIGHT_ADAPTER):]

    after_barcode = derived[len(LENTIMPRA_DOWNSTREAM):]
    assert after_barcode.startswith("ACCGGT")       # AgeI
    assert after_barcode.startswith("ACCGGTCGCCACCATG")  # ... Kozak + EGFP ATG


def test_suffix_fills_the_input_and_keeps_the_barcode_outside_it():
    assert LENTIMPRA_SUFFIX == LENTIMPRA_DOWNSTREAM[:INPUT_LENGTH - ELEMENT_LENGTH]
    assert len(LENTIMPRA_SUFFIX) == 51
    # The random barcode must never reach the model as literal N bases.
    assert "N" not in LENTIMPRA_SUFFIX


def test_legacy_suffix_is_the_uncorrected_reverse_primer_tail():
    """Pin the pre-correction constant: checkpoints trained with it must stay
    reproducible, so it may not be 'tidied' away."""
    assert len(LENTIMPRA_SUFFIX_LEGACY) == 51
    # It is the reverse primer's last 36 nt copied without reverse-complementing,
    # which is why reverse-complementing it hands back the right adapter.
    promoter_part = LENTIMPRA_SUFFIX_LEGACY[:36]
    assert _MINP_ENH_R.endswith(promoter_part)
    assert _reverse_complement(promoter_part).startswith(RIGHT_ADAPTER)


def test_legacy_and_corrected_suffixes_are_not_interchangeable():
    """Guards the ~0.02 Pearson mismatch penalty measured on the released
    K562 model (0.8793 legacy -> 0.8601 corrected)."""
    assert LENTIMPRA_SUFFIX != LENTIMPRA_SUFFIX_LEGACY
    differing = sum(a != b for a, b in zip(LENTIMPRA_SUFFIX, LENTIMPRA_SUFFIX_LEGACY))
    assert differing == 41


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, LENTIMPRA_SUFFIX),
        ({"construct_suffix": "ACGT"}, "ACGT"),
        ({"promoter_seq": "AAA", "rand_barcode": "TTT"}, "AAATTT"),
    ],
)
def test_dataset_resolves_the_construct(kwargs, expected, tmp_path):
    """The dataset defaults to the corrected construct, and the historical
    promoter_seq/rand_barcode pair still pins a construct explicitly."""
    from alphagenome_ft_mpra import LentiMPRADataset

    ds = LentiMPRADataset(model=None, cell_type="K562", split="test", **kwargs)
    assert ds.construct_suffix == expected
    # Downstream readers concatenate these two in order.
    assert ds.promoter_seq + ds.rand_barcode == expected
