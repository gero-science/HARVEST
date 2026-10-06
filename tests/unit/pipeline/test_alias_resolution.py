import asyncio
from types import SimpleNamespace

import pipeline.alias_resolution as alias_resolution


def test_empty_agent2_usage_shape_matches_legacy_accounting():
    assert alias_resolution.empty_agent2_usage() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "request_count": 0,
        "token_per_request": [],
        "max_tokens_per_request": 0,
        "completion_tokens_per_request": [],
        "max_completion_tokens_per_request": 0,
        "pyopsin_filtered_count": 0,
    }


def test_bypass_agent2_alias_resolution_accepts_extractor_enriched_rows():
    measures = [
        {
            "compound": "Example 1",
            "molecule_name": "Example 1",
            "molecule_smiles": "CCO",
            "agent2_resolution_source": "verified_smiles",
        }
    ]

    resolved, unresolved, processing_time, usage = alias_resolution.bypass_agent2_alias_resolution(measures)

    assert unresolved == []
    assert processing_time == 0.0
    assert usage == alias_resolution.empty_agent2_usage()
    assert resolved == [{
        "compound": "Example 1",
        "molecule_name": "Example 1",
        "molecule_smiles": "CCO",
        "agent2_resolution_source": alias_resolution.ACCEPTED_EXTRACTOR_SOURCE,
        "original_alias": "Example 1",
    }]
    assert measures[0]["agent2_resolution_source"] == "verified_smiles"


def test_bypass_agent2_alias_resolution_keeps_rows_without_smiles_unresolved():
    measures = [{"compound": "Example 2", "molecule_name": "Example 2"}]

    resolved, unresolved, processing_time, usage = alias_resolution.bypass_agent2_alias_resolution(measures)

    assert resolved == []
    assert processing_time == 0.0
    assert usage == alias_resolution.empty_agent2_usage()
    assert unresolved == [{
        "compound": "Example 2",
        "molecule_name": "Example 2",
        "agent2_resolution_source": "unresolved",
        "unresolved_reason": alias_resolution.UNRESOLVED_MISSING_SMILES_REASON,
    }]


def test_resolve_aliases_for_patent_tags_extractor_rows(monkeypatch):
    """The only path: rows with SMILES are accepted, the rest are unresolved."""
    resolved, unresolved, processing_time, usage = asyncio.run(alias_resolution.resolve_aliases_for_patent(
        measures_list=[
            {"compound": "Example 1", "molecule_smiles": "CCO"},
            {"compound": "Example 2"},
        ],
    ))

    assert [row["compound"] for row in resolved] == ["Example 1"]
    assert resolved[0]["agent2_resolution_source"] == alias_resolution.ACCEPTED_EXTRACTOR_SOURCE
    assert [row["compound"] for row in unresolved] == ["Example 2"]
    assert unresolved[0]["unresolved_reason"] == alias_resolution.UNRESOLVED_MISSING_SMILES_REASON
    assert processing_time == 0.0
    assert usage == alias_resolution.empty_agent2_usage()
