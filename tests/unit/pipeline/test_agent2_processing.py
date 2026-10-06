from pipeline.accounting import create_global_stats
from pipeline.agent2_processing import (
    ACCEPTED_EXTRACTOR_SOURCE,
    UNRESOLVED_MISSING_SMILES_REASON,
    bypass_agent2_alias_resolution,
    build_agent2_debug_results,
    calculate_molecule_resolution_stats,
    empty_agent2_usage,
    update_agent2_data_counts,
    update_agent2_usage_stats,
)


def test_calculate_molecule_resolution_stats_keeps_current_source_rules():
    resolved = [
        {"molecule_smiles": "CCO", "agent2_resolution_source": "verified_both"},
        {"molecule_smiles": "CCN", "agent2_resolution_source": "verified_name"},
        {"molecule_smiles": "", "agent2_resolution_source": "llm_unverified"},
        {"agent2_resolution_source": "unresolved"},
    ]

    stats = calculate_molecule_resolution_stats(resolved)

    assert stats == {
        "molecules_with_smiles": 2,
        "verified_molecules": 2,
        "unverified_molecules": 1,
    }


def test_agent2_global_accounting_matches_worker_contract():
    global_stats = create_global_stats()
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.25,
        "request_count": 1,
        "max_tokens_per_request": 15,
        "max_completion_tokens_per_request": 5,
        "pyopsin_filtered_count": 2,
    }
    resolved = [{"molecule_smiles": "CCO", "agent2_resolution_source": "verified_smiles"}]
    unresolved = [{"compound": "Example 2"}]
    molecule_stats = calculate_molecule_resolution_stats(resolved)

    update_agent2_usage_stats(global_stats, usage)
    update_agent2_data_counts(global_stats, resolved, unresolved, molecule_stats)

    assert global_stats["agent2"]["total_tokens"] == 15
    assert global_stats["agent2"]["pyopsin_filtered_count"] == 2
    assert global_stats["data_counts"]["patents_processed"] == 1
    assert global_stats["data_counts"]["aliases_resolved"] == 1
    assert global_stats["data_counts"]["aliases_unresolved"] == 1
    assert global_stats["data_counts"]["molecules_resolved_with_smiles"] == 1
    assert global_stats["data_counts"]["molecules_resolved_verified"] == 1


def test_build_agent2_debug_results_preserves_resolution_source_counts():
    resolved = [
        {"agent2_resolution_source": "verified_both"},
        {"agent2_resolution_source": "verified_smiles"},
        {"agent2_resolution_source": "verified_smiles"},
        {"agent2_resolution_source": "llm_unverified"},
    ]
    molecule_stats = calculate_molecule_resolution_stats(resolved)

    debug_results = build_agent2_debug_results(
        "USUNITTESTA1",
        resolved,
        unresolved=[],
        processing_time=1.5,
        molecule_stats=molecule_stats,
    )

    assert debug_results["resolution_sources"] == {
        "verified_both": 1,
        "verified_smiles": 2,
        "verified_name": 0,
        "llm_unverified": 1,
    }
    assert debug_results["processing_time"] == 1.5


def test_bypass_agent2_alias_resolution_accepts_extractor_enriched_rows():
    measures = [
        {
            "compound": "Example 1",
            "molecule_name": "Example 1",
            "molecule_smiles": "CCO",
            "agent2_resolution_source": "verified_smiles",
        }
    ]

    resolved, unresolved, processing_time, usage = bypass_agent2_alias_resolution(measures)

    assert unresolved == []
    assert processing_time == 0.0
    assert usage == empty_agent2_usage()
    assert resolved == [{
        "compound": "Example 1",
        "molecule_name": "Example 1",
        "molecule_smiles": "CCO",
        "agent2_resolution_source": ACCEPTED_EXTRACTOR_SOURCE,
        "original_alias": "Example 1",
    }]
    assert measures[0]["agent2_resolution_source"] == "verified_smiles"


def test_bypass_agent2_alias_resolution_keeps_rows_without_smiles_unresolved():
    measures = [{"compound": "Example 2", "molecule_name": "Example 2"}]

    resolved, unresolved, processing_time, usage = bypass_agent2_alias_resolution(measures)

    assert resolved == []
    assert processing_time == 0.0
    assert usage == empty_agent2_usage()
    assert unresolved == [{
        "compound": "Example 2",
        "molecule_name": "Example 2",
        "agent2_resolution_source": "unresolved",
        "unresolved_reason": UNRESOLVED_MISSING_SMILES_REASON,
    }]


def test_bypass_agent2_alias_resolution_fills_original_alias_stably():
    resolved, _, _, _ = bypass_agent2_alias_resolution([
        {"compound": "Example 3", "molecule_smiles": "CCN"},
        {
            "compound": "Example 4",
            "molecule_name": "Preferred Alias",
            "original_alias": "Existing Alias",
            "molecule_smiles": "CCC",
        },
    ])

    assert resolved[0]["original_alias"] == "Example 3"
    assert resolved[1]["original_alias"] == "Existing Alias"
