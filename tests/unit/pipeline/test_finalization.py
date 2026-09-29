import asyncio
import json

import pipeline.finalization as finalization


def test_save_error_logs_noops_when_debug_disabled(tmp_path):
    error_log = tmp_path / "debug_error_log.jsonl"
    error_log.write_text('{"patent_id": "US1"}\n', encoding="utf-8")

    asyncio.run(finalization.save_error_logs(tmp_path, error_log, error_count=1, debug_mode=False))

    assert not (tmp_path / "debug_problematic_patents.json").exists()


def test_save_error_logs_writes_problematic_patents_summary(tmp_path):
    error_log = tmp_path / "debug_error_log.jsonl"
    error_log.write_text(
        json.dumps({
            "timestamp": 1.0,
            "agent": "agent1",
            "patent_id": "USUNITTESTA1",
            "error_type": "parse_error",
            "error_class": "ValueError",
            "error_message": "x" * 220,
        }) + "\n",
        encoding="utf-8",
    )

    asyncio.run(finalization.save_error_logs(tmp_path, error_log, error_count=1, debug_mode=True))

    summary = json.loads((tmp_path / "debug_problematic_patents.json").read_text(encoding="utf-8"))
    assert summary[0]["patent_id"] == "USUNITTESTA1"
    assert summary[0]["error_count"] == 1
    assert summary[0]["error_types"] == ["parse_error"]
    assert summary[0]["errors"][0]["error_message"].endswith("...")
