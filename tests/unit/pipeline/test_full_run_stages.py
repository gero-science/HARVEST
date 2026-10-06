"""Tests for full pipeline stage orchestration."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline import full_run
from pipeline.full_run import (
    RUN_SUMMARY_FILENAME,
    StageError,
    _resolve_verify_source_dir,
    run_full_pipeline,
    validate_stage_inputs,
)
from pipeline.run_config import STAGES, PipelineRunConfig, VerifySettings


PATENT_ID = "US20240001234A1"


@pytest.fixture
def base_config(tmp_path):
    return PipelineRunConfig(input_path=str(tmp_path / "patents.zip"), output_dir=str(tmp_path / "results"))


@pytest.fixture
def stub_stages(monkeypatch):
    """Replace every stage with a recorder; returns the call log."""
    calls: list[str] = []

    def make_handler(name: str):
        async def handler(config):
            calls.append(name)
            return {"stage": name}

        return handler

    monkeypatch.setattr(
        full_run,
        "STAGE_HANDLERS",
        {name: make_handler(name) for name in STAGES},
    )
    return calls


def make_resolved_patent(output_dir):
    patent_dir = output_dir / PATENT_ID
    patent_dir.mkdir(parents=True, exist_ok=True)
    (patent_dir / f"{PATENT_ID}_resolved.json").write_text("[]", encoding="utf-8")


def read_summary(config: PipelineRunConfig) -> dict:
    return json.loads((Path(config.output_dir) / RUN_SUMMARY_FILENAME).read_text(encoding="utf-8"))


def test_stages_run_in_requested_order(base_config, stub_stages):
    exit_code = asyncio.run(run_full_pipeline(base_config, STAGES))

    assert exit_code == 0
    assert stub_stages == list(STAGES)


def test_only_selected_stages_run(base_config, stub_stages, tmp_path):
    make_resolved_patent(tmp_path / "results")

    exit_code = asyncio.run(run_full_pipeline(base_config, ("proteins", "export")))

    assert exit_code == 0
    assert stub_stages == ["proteins", "export"]


def test_failed_stage_stops_the_run(base_config, monkeypatch):
    calls: list[str] = []

    async def ok(config):
        calls.append("extract")
        return {}

    async def boom(config):
        calls.append("proteins")
        raise StageError("protein data missing")

    async def never(config):  # pragma: no cover - must not run
        calls.append("export")
        return {}

    monkeypatch.setattr(
        full_run,
        "STAGE_HANDLERS",
        {"extract": ok, "proteins": boom, "export": never, "postprocess": never},
    )

    exit_code = asyncio.run(run_full_pipeline(base_config, ("extract", "proteins", "export")))

    assert exit_code == 1
    assert calls == ["extract", "proteins"]

    summary = read_summary(base_config)
    statuses = {entry["stage"]: entry["status"] for entry in summary["stages"]}
    assert statuses == {"extract": "ok", "proteins": "failed", "export": "skipped"}


def test_continue_on_error_runs_later_stages(base_config, monkeypatch):
    calls: list[str] = []

    async def boom(config):
        calls.append("extract")
        raise RuntimeError("extraction crashed")

    async def ok(config):
        calls.append("export")
        return {}

    monkeypatch.setattr(
        full_run,
        "STAGE_HANDLERS",
        {"extract": boom, "proteins": ok, "export": ok, "postprocess": ok},
    )
    config = replace(base_config, common=replace(base_config.common, continue_on_error=True))

    exit_code = asyncio.run(run_full_pipeline(config, ("extract", "export")))

    assert exit_code == 1
    assert calls == ["extract", "export"]


def test_run_summary_records_stages_and_config(base_config, stub_stages):
    asyncio.run(run_full_pipeline(base_config, ("extract",)))

    summary = read_summary(base_config)

    assert summary["stages_requested"] == ["extract"]
    assert summary["stages"][0]["status"] == "ok"
    assert summary["stages"][0]["artifacts"] == {"stage": "extract"}
    assert summary["config"]["output_dir"] == base_config.output_dir
    assert "total_seconds" in summary


def test_validate_stage_inputs_requires_extraction_artifacts(tmp_path):
    config = PipelineRunConfig(output_dir=str(tmp_path / "results"))

    with pytest.raises(StageError, match="No \\*_resolved.json found"):
        validate_stage_inputs(config, ("export",))

    make_resolved_patent(tmp_path / "results")
    validate_stage_inputs(config, ("export",))


def test_validate_stage_inputs_skips_check_when_extract_runs(tmp_path):
    config = PipelineRunConfig(output_dir=str(tmp_path / "results"))

    validate_stage_inputs(config, ("extract", "export"))


def test_validate_stage_inputs_requires_parquet_for_postprocess(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    config = PipelineRunConfig(output_dir=str(output_dir))

    with pytest.raises(StageError, match="Post-processing input Parquet not found"):
        validate_stage_inputs(config, ("postprocess",))

    (output_dir / "main_res.parquet").write_text("", encoding="utf-8")
    validate_stage_inputs(config, ("postprocess",))


def test_missing_extraction_artifacts_fail_before_any_stage(base_config, stub_stages):
    with pytest.raises(StageError):
        asyncio.run(run_full_pipeline(base_config, ("export",)))

    assert stub_stages == []


def test_extraction_counts_come_from_pipeline_statistics(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "pipeline_statistics.json").write_text(
        json.dumps({
            "data_counts": {"patents_processed": 0, "bioactivity_data_found": 0},
            "error_analysis": {"total_agent1_errors": 2, "total_agent2_errors": 0},
        }),
        encoding="utf-8",
    )

    assert full_run.read_extraction_counts(output_dir) == {
        "patents_processed": 0,
        "bioactivity_data_found": 0,
        "agent1_errors": 2,
        "agent2_errors": 0,
    }


def test_extraction_counts_tolerate_missing_statistics(tmp_path):
    assert full_run.read_extraction_counts(tmp_path) == {}


def test_proteins_stage_passes_recommended_defaults(tmp_path, monkeypatch):
    import protein_postprocessor

    recorded = {}

    class FakePostprocessor:
        def __init__(self, **kwargs):
            recorded["init"] = kwargs

        async def process_results_dir(self, results_dir, **kwargs):
            recorded["call"] = {"results_dir": results_dir, **kwargs}

        async def close(self):
            recorded["closed"] = True

    monkeypatch.setattr(protein_postprocessor, "ProteinPostprocessor", FakePostprocessor)

    protein_data = tmp_path / "protein_data"
    protein_data.mkdir()
    (protein_data / "uniprot_sprot.fasta").write_text(">sp|P0|TEST\nMKV\n")
    config = PipelineRunConfig(
        output_dir=str(tmp_path / "results"),
        proteins=replace(PipelineRunConfig().proteins, protein_data_path=str(protein_data)),
    )

    asyncio.run(full_run._stage_proteins(config))

    assert recorded["call"]["num_workers"] == 5
    # Full runs now match the `python -m protein_postprocessor` defaults.
    assert recorded["call"]["use_stage1_context"] is True
    assert recorded["call"]["reprocess_all"] is True
    assert recorded["call"]["force"] is False
    assert recorded["closed"] is True


def test_proteins_stage_requires_protein_data(tmp_path):
    config = PipelineRunConfig(
        output_dir=str(tmp_path / "results"),
        proteins=replace(PipelineRunConfig().proteins, protein_data_path=str(tmp_path / "absent")),
    )

    with pytest.raises(StageError, match="Protein data directory not found"):
        asyncio.run(full_run._stage_proteins(config))


def test_proteins_stage_requires_the_fasta_not_just_the_directory(tmp_path):
    """An empty protein_data dir used to pass, then fail later inside the resolver."""
    data_dir = tmp_path / "protein_data"
    data_dir.mkdir()
    config = PipelineRunConfig(
        output_dir=str(tmp_path / "results"),
        proteins=replace(PipelineRunConfig().proteins, protein_data_path=str(data_dir)),
    )

    with pytest.raises(StageError, match="UniProt FASTA not found"):
        asyncio.run(full_run._stage_proteins(config))


def test_proteins_stage_reads_the_fasta_from_protein_data_path(tmp_path, monkeypatch):
    """--protein-data-path used to be decorative: validated, then ignored."""
    data_dir = tmp_path / "elsewhere"
    data_dir.mkdir()
    (data_dir / "uniprot_sprot.fasta").write_text(">sp|P0|TEST\nMKV\n")

    captured = {}

    class FakePostprocessor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def process_results_dir(self, *a, **k):
            pass

        async def close(self):
            pass

    import protein_postprocessor

    monkeypatch.setattr(protein_postprocessor, "ProteinPostprocessor", FakePostprocessor)
    config = PipelineRunConfig(
        output_dir=str(tmp_path / "results"),
        proteins=replace(PipelineRunConfig().proteins, protein_data_path=str(data_dir)),
    )

    asyncio.run(full_run._stage_proteins(config))

    assert captured["fasta_path"] == data_dir / "uniprot_sprot.fasta"


def test_export_stage_forwards_settings_and_reports_failure(tmp_path, monkeypatch):
    import export_table

    recorded = {}

    def fake_export(**kwargs):
        recorded.update(kwargs)
        return True, "42 rows", 1.5

    monkeypatch.setattr(export_table, "run_bindingdb_processing", fake_export)

    output_dir = tmp_path / "results"
    config = PipelineRunConfig(
        output_dir=str(output_dir),
        export=replace(
            PipelineRunConfig().export,
            workers=3,
            batch_size=7,
            use_opsin=True,
            patent_dict=str(tmp_path / "absent.json"),
        ),
    )

    artifacts = asyncio.run(full_run._stage_export(config))

    assert artifacts == {"parquet": str(output_dir / "main_res.parquet")}
    assert recorded["workers"] == 3
    assert recorded["batch_size"] == 7
    assert recorded["use_opsin"] is True
    # A missing mapping dictionary is a warning, not a failure.
    assert recorded["patent_dict_file"] is None

    monkeypatch.setattr(
        export_table, "run_bindingdb_processing", lambda **kwargs: (False, "no patents", 0.2)
    )

    with pytest.raises(StageError, match="BindingDB export failed"):
        asyncio.run(full_run._stage_export(config))


def test_postprocess_stage_builds_final_postprocessing_config(tmp_path, monkeypatch):
    from final_postprocessing import run as fp_run

    recorded = {}

    def fake_run(input_path, output_path, config):
        recorded["input"] = Path(input_path)
        recorded["output"] = Path(output_path)
        recorded["config"] = config
        return output_path

    monkeypatch.setattr(fp_run, "run_final_postprocessing", fake_run)

    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "main_res.parquet").write_text("", encoding="utf-8")
    config = PipelineRunConfig(
        output_dir=str(output_dir),
        postprocess=replace(
            PipelineRunConfig().postprocess,
            skip_add_clean_best=True,
            triplet_count=0,
            workers=2,
        ),
    )

    artifacts = asyncio.run(full_run._stage_postprocess(config))

    assert artifacts == {"parquet": str(output_dir / "main_res_clean.parquet")}
    assert recorded["input"] == output_dir / "main_res.parquet"
    assert recorded["output"] == output_dir / "main_res_clean.parquet"
    assert recorded["config"].skip_add_clean_best is True
    assert recorded["config"].triplet_count == 0
    assert recorded["config"].workers == 2


def test_postprocess_stage_reports_missing_input(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    config = PipelineRunConfig(output_dir=str(output_dir))

    with pytest.raises(StageError, match="Post-processing input Parquet not found"):
        asyncio.run(full_run._stage_postprocess(config))


def test_collect_input_paths_from_directory(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "b.zip").write_text("", encoding="utf-8")
    (corpus / "a.zip").write_text("", encoding="utf-8")
    (corpus / "notes.txt").write_text("", encoding="utf-8")

    config = PipelineRunConfig(input_path=str(corpus), output_dir=str(tmp_path / "results"))
    paths = asyncio.run(full_run.collect_input_paths(config))

    assert [p.rsplit("/", 1)[-1] for p in paths] == ["a.zip", "b.zip"]


def test_collect_input_paths_rejects_non_zip_file(tmp_path):
    archive = tmp_path / "patents.tar"
    archive.write_text("", encoding="utf-8")
    config = PipelineRunConfig(input_path=str(archive), output_dir=str(tmp_path / "results"))

    with pytest.raises(StageError, match="must have a .zip extension"):
        asyncio.run(full_run.collect_input_paths(config))


def test_collect_input_paths_rejects_missing_path(tmp_path):
    config = PipelineRunConfig(input_path=str(tmp_path / "absent.zip"), output_dir=str(tmp_path / "results"))

    with pytest.raises(StageError, match="Input path not found"):
        asyncio.run(full_run.collect_input_paths(config))


# -- Verify source_dir resolution (issue 1) ----------------------------------


def test_resolve_verify_source_dir_from_explicit_setting(tmp_path):
    d = tmp_path / "zips"
    d.mkdir()
    config = PipelineRunConfig(verify=VerifySettings(source_dir=str(d)))
    assert _resolve_verify_source_dir(config) == str(d)


def test_resolve_verify_source_dir_from_directory_input_path(tmp_path):
    d = tmp_path / "patents"
    d.mkdir()
    config = PipelineRunConfig(input_path=str(d))
    assert _resolve_verify_source_dir(config) == str(d)


def test_resolve_verify_source_dir_from_zip_input_path(tmp_path):
    f = tmp_path / "patent.zip"
    f.touch()
    config = PipelineRunConfig(input_path=str(f))
    assert _resolve_verify_source_dir(config) == str(tmp_path)


def test_resolve_verify_source_dir_from_input_list(tmp_path):
    zip_dir = tmp_path / "data"
    zip_dir.mkdir()
    (zip_dir / "US123.zip").touch()

    list_file = tmp_path / "patents.txt"
    list_file.write_text(f"# comment\n{zip_dir / 'US123.zip'}\n")

    config = PipelineRunConfig(input_list=str(list_file))
    assert _resolve_verify_source_dir(config) == str(zip_dir)


def test_resolve_verify_source_dir_returns_none_when_nothing_set():
    assert _resolve_verify_source_dir(PipelineRunConfig()) is None


def test_validate_stage_inputs_fails_early_when_verify_unresolvable(tmp_path):
    config = PipelineRunConfig(output_dir=str(tmp_path))
    with pytest.raises(StageError, match="patent ZIP directory"):
        validate_stage_inputs(config, ("extract", "verify"))


def test_validate_stage_inputs_fails_early_when_verify_source_is_file(tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.touch()
    config = PipelineRunConfig(
        output_dir=str(tmp_path),
        verify=VerifySettings(source_dir=str(f)),
    )
    with pytest.raises(StageError, match="must be a directory"):
        validate_stage_inputs(config, ("extract", "verify"))


def test_validate_stage_inputs_skips_verify_check_when_not_in_stages(tmp_path):
    config = PipelineRunConfig(output_dir=str(tmp_path))
    validate_stage_inputs(config, ("extract",))
