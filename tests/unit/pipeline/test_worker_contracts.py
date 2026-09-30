import asyncio
import json
from types import SimpleNamespace

import pytest

import pipeline.workers as workers
from pipeline.accounting import empty_agent1_usage


@pytest.fixture(autouse=True)
def run_to_thread_inline(monkeypatch):
    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(workers.asyncio, "to_thread", inline_to_thread)


def test_save_patent_results_writes_only_non_empty_result_files_and_processed_marker(tmp_path):
    patent_id = "USUNITTESTA1"
    processed_file = tmp_path / "processed_patents.txt"
    resolved = [
        {
            "compound": "Example 1",
            "molecule_smiles": "CCO",
        }
    ]

    asyncio.run(
        workers.save_patent_results(
            patent_id=patent_id,
            resolved=resolved,
            unresolved=[],
            output_dir=tmp_path,
            processed_patents_file=processed_file,
        )
    )

    patent_dir = tmp_path / patent_id
    resolved_path = patent_dir / f"{patent_id}_resolved.json"
    assert resolved_path.exists()
    assert not (patent_dir / f"{patent_id}_unresolved.json").exists()
    assert processed_file.read_text(encoding="utf-8") == f"{patent_id}\n"
    assert json.loads(resolved_path.read_text(encoding="utf-8"))[0]["compound"] == "Example 1"


def test_save_patent_results_empty_lists_are_valid_no_resolved_state(tmp_path):
    patent_id = "USUNITTESTA1"
    processed_file = tmp_path / "processed_patents.txt"

    asyncio.run(
        workers.save_patent_results(
            patent_id=patent_id,
            resolved=[],
            unresolved=[],
            output_dir=tmp_path,
            processed_patents_file=processed_file,
        )
    )

    patent_dir = tmp_path / patent_id
    assert patent_dir.exists()
    assert not (patent_dir / f"{patent_id}_resolved.json").exists()
    assert not (patent_dir / f"{patent_id}_unresolved.json").exists()
    assert processed_file.read_text(encoding="utf-8") == f"{patent_id}\n"


def _run_aggregator(tmp_path, monkeypatch, chunks):
    """Drive the aggregator over `chunks`, stubbing out everything downstream."""
    async def fake_process_patent_agent2(*args):
        return args[0], 1, 0, {}

    async def noop_async(*args, **kwargs):
        return None

    monkeypatch.setattr(workers, "process_patent_agent2", fake_process_patent_agent2)
    monkeypatch.setattr(workers, "_save_error_logs", noop_async)
    monkeypatch.setattr(workers, "_print_final_statistics", lambda *a, **k: None)

    import pipeline.statistics as statistics

    monkeypatch.setattr(statistics, "save_statistics_to_files", noop_async)

    queue = asyncio.Queue()
    for chunk in chunks:
        queue.put_nowait(chunk)
    queue.put_nowait(None)

    asyncio.run(
        workers.worker_agent2_aggregator(
            queue, output_dir=tmp_path, start_time=0.0, debug_mode=False
        )
    )
    return tmp_path / "processed_patents.txt"


def _chunk(patent_id, extracted_data, total_chunks=1, error_detail=None):
    return (
        patent_id,
        extracted_data,
        total_chunks,
        "patent text",
        empty_agent1_usage(),
        SimpleNamespace(chemistry_nodes=[]),
        {},
        error_detail,
        "patent.zip",
    )


def test_patent_whose_every_chunk_failed_is_not_marked_processed(tmp_path, monkeypatch):
    """A transient Agent 1 outage must not permanently retire a patent.

    An errored chunk still counts as received, so the patent looks complete
    with an empty measures list. Marking it processed would make --resume skip
    it forever and silently drop it from the corpus.
    """
    error = {"error_type": "api_error", "detail": "boom"}
    processed_file = _run_aggregator(tmp_path, monkeypatch, [
        _chunk("USFAILEDA1", [], total_chunks=2, error_detail=error),
        _chunk("USFAILEDA1", [], total_chunks=2, error_detail=error),
    ])

    assert not processed_file.exists() or "USFAILEDA1" not in processed_file.read_text()


def test_patent_with_no_data_and_no_errors_is_marked_processed(tmp_path, monkeypatch):
    """The genuinely-empty case still has to be retired, or runs never finish."""
    processed_file = _run_aggregator(tmp_path, monkeypatch, [
        _chunk("USEMPTYA1", []),
    ])

    assert processed_file.read_text(encoding="utf-8") == "USEMPTYA1\n"


def test_worker_agent2_aggregator_does_not_create_an_agent2_llm(tmp_path, monkeypatch):
    calls = []

    def fail_from_config(*args, **kwargs):
        raise AssertionError("The aggregator must not create an Agent2 LLM")

    async def fake_process_patent_agent2(*args):
        calls.append(("process", args))
        return args[0], 1, 0, {}

    async def noop_async(*args, **kwargs):
        calls.append(("noop", args, kwargs))

    def noop_sync(*args, **kwargs):
        calls.append(("print_stats", args, kwargs))

    async def fake_save_statistics_to_files(*args, **kwargs):
        calls.append(("save_statistics", args, kwargs))

    monkeypatch.setattr(workers.LLM, "from_config", fail_from_config)
    monkeypatch.setattr(workers, "process_patent_agent2", fake_process_patent_agent2)
    monkeypatch.setattr(workers, "_save_error_logs", noop_async)
    monkeypatch.setattr(workers, "_print_final_statistics", noop_sync)

    import pipeline.statistics as statistics

    monkeypatch.setattr(statistics, "save_statistics_to_files", fake_save_statistics_to_files)

    queue = asyncio.Queue()
    queue.put_nowait((
        "USUNITTESTA1",
        [{"compound": "Example 1", "molecule_smiles": "CCO"}],
        1,
        "patent text",
        empty_agent1_usage(),
        SimpleNamespace(chemistry_nodes=[]),
        {},
        None,
        "patent.zip",
    ))
    queue.put_nowait(None)

    result = asyncio.run(
        workers.worker_agent2_aggregator(
            queue,
            output_dir=tmp_path,
            start_time=0.0,
            debug_mode=False,
        )
    )

    assert result == []
    process_call = [call for call in calls if call[0] == "process"][0]
    assert process_call[1][-1] is False
