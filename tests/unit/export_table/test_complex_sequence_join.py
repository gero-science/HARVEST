"""Protein-complex sequences must be joined with ';'.

Multi-protein complexes used to concatenate sequences with an empty separator,
producing a single string that downstream export could not split. Each member
chain must stay addressable via `;`-separated fields.
"""
import json

import pytest

import export_table


pytestmark = pytest.mark.regression


def test_complex_sequences_joined_with_semicolon(tmp_path):
    """3-protein complex must produce a `;`-joined sequence string."""
    payload = {
        "complexes": [
            {
                "complex_name": "IL-12 receptor signaling",
                "organism": "human",
                "species": "Homo sapiens",
                "proteins": [
                    {
                        "gene": "TYK2",
                        "sequence": "MPRGSEKKAPVEAY",
                        "accession": "P29597",
                        "uniprot_id": "TYK2_HUMAN",
                    },
                    {
                        "gene": "JAK2",
                        "sequence": "MGMACLTMTEMEGT",
                        "accession": "O60674",
                        "uniprot_id": "JAK2_HUMAN",
                    },
                    {
                        "gene": "STAT4",
                        "sequence": "MSQWNQVQQLEIKF",
                        "accession": "Q14765",
                        "uniprot_id": "STAT4_HUMAN",
                    },
                ],
            }
        ]
    }
    (tmp_path / "protein_complexes.json").write_text(json.dumps(payload))

    result = export_table.load_protein_complexes(str(tmp_path))
    rec = result["IL-12 receptor signaling"]["human"]

    expected_sequence = "MPRGSEKKAPVEAY;MGMACLTMTEMEGT;MSQWNQVQQLEIKF"
    assert rec["sequence"] == expected_sequence, (
        "Complex sequences must be joined with ';'. If they get concatenated "
        "without a separator, downstream consumers can no longer split them "
        "back into per-protein chains."
    )
    assert rec["gene"] == "TYK2;JAK2;STAT4"
    assert rec["accession"] == "P29597;O60674;Q14765"
    assert rec["uniprot_id"] == "TYK2_HUMAN;JAK2_HUMAN;STAT4_HUMAN"


def test_single_protein_complex_no_trailing_separator(tmp_path):
    """A complex with a single protein must NOT have a trailing ';'."""
    payload = {
        "complexes": [
            {
                "complex_name": "X",
                "organism": "human",
                "proteins": [{"gene": "G", "sequence": "AAA", "accession": "P1",
                              "uniprot_id": "G_HUMAN"}],
            }
        ]
    }
    (tmp_path / "protein_complexes.json").write_text(json.dumps(payload))
    rec = export_table.load_protein_complexes(str(tmp_path))["X"]["human"]
    assert rec["sequence"] == "AAA"
    assert rec["gene"] == "G"


def test_proteins_with_missing_sequence_are_dropped_from_join(tmp_path):
    """Members without a `sequence` field must be skipped, not produce empty
    chunks like ';;'."""
    payload = {
        "complexes": [
            {
                "complex_name": "X",
                "organism": "human",
                "proteins": [
                    {"gene": "A", "sequence": "AAA"},
                    {"gene": "B"},
                    {"gene": "C", "sequence": "CCC"},
                ],
            }
        ]
    }
    (tmp_path / "protein_complexes.json").write_text(json.dumps(payload))
    rec = export_table.load_protein_complexes(str(tmp_path))["X"]["human"]
    assert rec["sequence"] == "AAA;CCC"
    assert ";;" not in rec["sequence"]
