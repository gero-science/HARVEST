"""Tests for per-patent result artifact writing."""

import asyncio
import json

from pipeline.artifacts import save_patent_result_artifacts


PATENT_ID = "US20240001234A1"


def save(tmp_path, **kwargs):
    payload = {
        "patent_id": PATENT_ID,
        "resolved": [{"compound": "Example 1"}],
        "unresolved": [],
        "output_dir": tmp_path,
        "processed_patents_file": tmp_path / "processed_patents.txt",
    }
    payload.update(kwargs)

    asyncio.run(
        save_patent_result_artifacts(
            payload["patent_id"],
            payload["resolved"],
            payload["unresolved"],
            payload["output_dir"],
            payload["processed_patents_file"],
            cdx_data=payload.get("cdx_data"),
        )
    )
    return tmp_path / PATENT_ID


def test_cdx_results_written_in_expected_format(tmp_path):
    cdx_data = {
        "00013": {"smiles": "CCO", "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"},
        "13": {"smiles": "CCO", "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"},
        "00014": {"smiles": None, "inchikey": None},
    }

    patent_dir = save(tmp_path, cdx_data=cdx_data)

    payload = json.loads((patent_dir / "cdx_results.json").read_text(encoding="utf-8"))
    assert payload["patent_id"] == PATENT_ID
    assert payload["compounds"] == cdx_data
    assert payload["processed_at"]

    assert (patent_dir / f"{PATENT_ID}_resolved.json").exists()


def test_cdx_results_not_written_without_cdx_data(tmp_path):
    patent_dir = save(tmp_path, cdx_data=None)
    assert not (patent_dir / "cdx_results.json").exists()

    patent_dir = save(tmp_path, cdx_data={})
    assert not (patent_dir / "cdx_results.json").exists()
