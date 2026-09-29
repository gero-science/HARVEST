"""Tests for the export_table.py entry point: re-exports and CLI arguments."""

import sys

import export_table


def test_export_table_reexports_public_names():
    for name in [
        "PARQUET_SCHEMA",
        "OUTPUT_COLUMNS",
        "run_bindingdb_processing",
        "process_patent_worker",
        "binding_to_parquet_row",
        "load_single_proteins",
        "load_protein_complexes",
        "load_cdx_results",
    ]:
        assert hasattr(export_table, name), name


def test_cli_defaults_to_local_only(monkeypatch, capsys):
    calls = []

    def fake_run_bindingdb_processing(*args, **kwargs):
        calls.append((args, kwargs))
        return True, "ok", 0.0

    monkeypatch.setattr(export_table, "run_bindingdb_processing", fake_run_bindingdb_processing)
    monkeypatch.setattr(sys, "argv", ["export_table.py", "input", "out.parquet"])

    export_table.main()

    assert calls[0][1] == {"use_opsin": False, "structure_source": "cdx"}
    assert "ok" in capsys.readouterr().out


def test_cli_enables_opsin_only_when_flag_is_present(monkeypatch):
    calls = []

    def fake_run_bindingdb_processing(*args, **kwargs):
        calls.append((args, kwargs))
        return True, "ok", 0.0

    monkeypatch.setattr(export_table, "run_bindingdb_processing", fake_run_bindingdb_processing)
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_table.py", "input", "out.parquet", "--use-opsin"],
    )

    export_table.main()

    assert calls[0][1] == {"use_opsin": True, "structure_source": "cdx"}
