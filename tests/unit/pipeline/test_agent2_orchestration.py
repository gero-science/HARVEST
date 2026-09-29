import asyncio
from types import SimpleNamespace

import pytest

from pipeline.accounting import create_global_stats, empty_agent1_usage
from pipeline.alias_resolution import empty_agent2_usage
import pipeline.agent2_orchestration as orchestration


def agent2_usage(**overrides):
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.25,
        "request_count": 1,
        "token_per_request": [15],
        "max_tokens_per_request": 15,
        "completion_tokens_per_request": [5],
        "max_completion_tokens_per_request": 5,
        "pyopsin_filtered_count": 2,
    }
    usage.update(overrides)
    return usage


def install_common_success_mocks(monkeypatch, calls, resolved, unresolved):
    usage = agent2_usage()

    async def fake_resolve_aliases_for_patent(**kwargs):
        calls.append(("resolve", kwargs))
        return resolved, unresolved, 1.5, usage

    async def fake_save_patent_result_artifacts(*args, **kwargs):
        calls.append(("save_results", args, kwargs))

    async def fake_save_debug_data(*args, **kwargs):
        calls.append(("save_debug", args, kwargs))

    async def fake_save_patent_statistics(**kwargs):
        calls.append(("save_stats", kwargs))

    monkeypatch.setattr(orchestration, "resolve_aliases_for_patent", fake_resolve_aliases_for_patent)
    monkeypatch.setattr(orchestration, "save_patent_result_artifacts", fake_save_patent_result_artifacts)
    monkeypatch.setattr(orchestration, "save_debug_data", fake_save_debug_data)
    monkeypatch.setattr(orchestration, "save_patent_statistics", fake_save_patent_statistics)

    return usage


def run_process_patent_agent2(
    tmp_path,
    global_stats,
    debug_mode=False,
    measures_list=None,
    document=None,
):
    if document is None:
        document = SimpleNamespace(chemistry_nodes=[SimpleNamespace(molecule_smiles="CCO")])
    return asyncio.run(
        orchestration.process_patent_agent2(
            patent_id="USUNITTESTA1",
            measures_list=measures_list if measures_list is not None else [{"compound": "Example 1"}],
            patent_text="patent text",
            document=document,
            output_dir=tmp_path,
            processed_patents_file=tmp_path / "processed_patents.txt",
            global_stats=global_stats,
            detailed_error_log=tmp_path / "debug_error_log.jsonl",
            agent1_usage_for_patent=empty_agent1_usage(),
            debug_mode=debug_mode,
        )
    )


def test_process_patent_agent2_success_updates_stats_and_saves_outputs(tmp_path, monkeypatch):
    resolved = [{"molecule_smiles": "CCO", "agent2_resolution_source": "verified_smiles"}]
    unresolved = [{"compound": "Example 2"}]
    global_stats = create_global_stats()
    calls = []
    usage = install_common_success_mocks(monkeypatch, calls, resolved, unresolved)

    result = run_process_patent_agent2(tmp_path, global_stats)

    assert result == ("USUNITTESTA1", 1, 1, usage)
    assert global_stats["agent2"]["total_tokens"] == 15
    assert global_stats["agent2"]["pyopsin_filtered_count"] == 2
    assert global_stats["data_counts"]["patents_processed"] == 1
    assert global_stats["data_counts"]["aliases_resolved"] == 1
    assert global_stats["data_counts"]["aliases_unresolved"] == 1
    assert global_stats["data_counts"]["molecules_resolved_with_smiles"] == 1
    assert global_stats["data_counts"]["molecules_resolved_verified"] == 1

    save_results = [call for call in calls if call[0] == "save_results"][0]
    assert save_results[1][1:3] == (resolved, unresolved)

    save_stats = [call for call in calls if call[0] == "save_stats"][0][1]
    assert save_stats["resolved_count"] == 1
    assert save_stats["unresolved_count"] == 1
    assert save_stats["molecules_with_smiles"] == 1
    assert save_stats["verified_molecules"] == 1
    assert not [call for call in calls if call[0] == "save_debug"]


def test_process_patent_agent2_forwards_document_cdx_data(tmp_path, monkeypatch):
    global_stats = create_global_stats()
    calls = []
    install_common_success_mocks(monkeypatch, calls, resolved=[{"molecule_smiles": "CCO"}], unresolved=[])

    cdx_data = {"00013": {"smiles": "CCO", "inchikey": "KEY"}}
    document = SimpleNamespace(
        chemistry_nodes=[SimpleNamespace(molecule_smiles="CCO")],
        cdx_data=cdx_data,
    )

    run_process_patent_agent2(tmp_path, global_stats, document=document)

    save_results = [call for call in calls if call[0] == "save_results"][0]
    assert save_results[2]["cdx_data"] == cdx_data


def test_process_patent_agent2_without_cdx_data_passes_none(tmp_path, monkeypatch):
    global_stats = create_global_stats()
    calls = []
    install_common_success_mocks(monkeypatch, calls, resolved=[{"molecule_smiles": "CCO"}], unresolved=[])

    run_process_patent_agent2(tmp_path, global_stats)

    save_results = [call for call in calls if call[0] == "save_results"][0]
    assert save_results[2]["cdx_data"] is None


def test_process_patent_agent2_preserves_empty_unresolved_list(tmp_path, monkeypatch):
    resolved = [{"compound": "Example 1", "agent2_resolution_source": "llm_unverified"}]
    unresolved = []
    global_stats = create_global_stats()
    calls = []
    install_common_success_mocks(monkeypatch, calls, resolved, unresolved)

    result = run_process_patent_agent2(tmp_path, global_stats)

    assert result[1:3] == (1, 0)
    save_results = [call for call in calls if call[0] == "save_results"][0]
    assert save_results[1][1:3] == (resolved, [])


def test_process_patent_agent2_debug_mode_controls_debug_artifact_save(tmp_path, monkeypatch):
    resolved = [{"molecule_smiles": "CCO", "agent2_resolution_source": "verified_both"}]
    global_stats = create_global_stats()
    calls = []
    install_common_success_mocks(monkeypatch, calls, resolved, unresolved=[])

    run_process_patent_agent2(tmp_path, global_stats, debug_mode=True)

    assert [call[0] for call in calls].count("save_debug") == 1


def test_process_patent_agent2_error_records_detail_and_reraises(tmp_path, monkeypatch):
    global_stats = create_global_stats()
    calls = []

    async def fake_resolve_aliases_for_patent(**kwargs):
        raise RuntimeError("alias failure")

    def fake_append_error_detail(error_log_file, error_detail):
        calls.append(("append_error", error_log_file, error_detail))

    monkeypatch.setattr(orchestration, "resolve_aliases_for_patent", fake_resolve_aliases_for_patent)
    monkeypatch.setattr(orchestration, "append_error_detail", fake_append_error_detail)

    with pytest.raises(RuntimeError, match="alias failure"):
        run_process_patent_agent2(tmp_path, global_stats)

    assert len(calls) == 1
    assert calls[0][2]["agent"] == "agent2"
    assert calls[0][2]["patent_id"] == "USUNITTESTA1"
    assert calls[0][2]["error_class"] == "RuntimeError"
    assert global_stats["data_counts"]["patents_processed"] == 0


def test_process_patent_agent2_tags_extractor_rows_and_reports_unresolved(tmp_path, monkeypatch):
    global_stats = create_global_stats()
    calls = []

    async def fake_save_patent_result_artifacts(*args, **kwargs):
        calls.append(("save_results", args, kwargs))

    async def fake_save_patent_statistics(**kwargs):
        calls.append(("save_stats", kwargs))

    monkeypatch.setattr(orchestration, "save_patent_result_artifacts", fake_save_patent_result_artifacts)
    monkeypatch.setattr(orchestration, "save_patent_statistics", fake_save_patent_statistics)

    result = run_process_patent_agent2(
        tmp_path,
        global_stats,
        measures_list=[
            {"compound": "Example 1", "molecule_name": "Example 1", "molecule_smiles": "CCO"},
            {"compound": "Example 2"},
        ],
    )

    assert result[0:3] == ("USUNITTESTA1", 1, 1)
    assert result[3] == empty_agent2_usage()
    assert global_stats["agent2"]["request_count"] == 0
    assert global_stats["data_counts"]["patents_processed"] == 1
    assert global_stats["data_counts"]["aliases_resolved"] == 1
    assert global_stats["data_counts"]["aliases_unresolved"] == 1

    save_results = [call for call in calls if call[0] == "save_results"][0]
    resolved, unresolved = save_results[1][1:3]
    assert resolved[0]["agent2_resolution_source"] == "skipped_enriched_by_agent1"
    assert resolved[0]["original_alias"] == "Example 1"
    assert unresolved[0]["unresolved_reason"] == "missing_molecule_smiles"
