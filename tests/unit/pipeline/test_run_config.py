"""Tests for the YAML run config and stage resolution."""

from argparse import Namespace
from pathlib import Path

import pytest

from pipeline.run_config import (
    DEFAULT_STAGES,
    STAGES,
    PipelineRunConfig,
    PostprocessSettings,
    apply_cli_overrides,
    config_to_dict,
    load_run_config,
    resolve_stages,
    validate_run_config,
)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def cli_args(**overrides) -> Namespace:
    defaults = {
        "input_path": None,
        "input_list": None,
        "limit": None,
        "output_dir": None,
        "log_level": None,
        "debug": False,
        "resume": False,
        "continue_on_error": False,
        "protein_workers": None,
        "protein_data_path": None,
        "export_workers": None,
        "postprocess_workers": None,
        "bdb_json": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_load_config_reads_sections(tmp_path):
    path = write_config(
        tmp_path,
        """
input:
  path: data/hallu_100
  limit: 5
output:
  dir: results/run1
stages: [extract, export]
common:
  log_level: DEBUG
  debug: true
proteins:
  workers: 12
export:
  use_opsin: true
  output_file: custom.parquet
postprocess:
  skip_add_clean_best: true
""",
    )

    config = load_run_config(path)

    assert config.input_path == "data/hallu_100"
    assert config.limit == 5
    assert config.output_dir == "results/run1"
    assert config.stages == ("extract", "export")
    assert config.common.log_level == "DEBUG"
    assert config.common.debug is True
    assert config.proteins.workers == 12
    assert config.export.use_opsin is True
    assert config.postprocess.skip_add_clean_best is True


def test_load_config_keeps_defaults_for_missing_sections(tmp_path):
    path = write_config(tmp_path, "output:\n  dir: results/run1\n")

    config = load_run_config(path)

    assert config.stages is None
    assert config.proteins.workers == 5
    assert config.proteins.stage1_context is True
    assert config.proteins.reprocess_all is True
    assert config.export.workers == 6
    assert config.postprocess.output_file == "main_res_clean.parquet"


def test_load_config_rejects_unknown_top_level_key(tmp_path):
    path = write_config(tmp_path, "outputs:\n  dir: results/run1\n")

    with pytest.raises(ValueError, match="Unknown top-level key"):
        load_run_config(path)


def test_load_config_rejects_unknown_section_key(tmp_path):
    path = write_config(tmp_path, "export:\n  worker: 6\n")

    with pytest.raises(ValueError, match="Unknown key\\(s\\) in config section 'export'"):
        load_run_config(path)


def test_load_config_rejects_unknown_stage(tmp_path):
    path = write_config(tmp_path, "stages: [extract, magic]\n")

    with pytest.raises(ValueError, match="Unknown stage 'magic'"):
        load_run_config(path)


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run_config(tmp_path / "absent.yaml")


def test_stages_are_normalized_to_canonical_order(tmp_path):
    path = write_config(tmp_path, "stages: [postprocess, extract, extract]\n")

    assert load_run_config(path).stages == ("extract", "postprocess")


def test_cli_overrides_win_over_config(tmp_path):
    path = write_config(
        tmp_path,
        """
input:
  path: config.zip
  limit: 5
output:
  dir: results/from_config
proteins:
  workers: 5
export:
  workers: 6
""",
    )
    config = load_run_config(path)

    merged = apply_cli_overrides(
        config,
        cli_args(
            input_path="cli.zip",
            limit=2,
            output_dir="results/from_cli",
            protein_workers=20,
            export_workers=3,
            postprocess_workers=4,
            bdb_json="data/other.json",
            log_level="WARNING",
        ),
    )

    assert merged.input_path == "cli.zip"
    assert merged.limit == 2
    assert merged.output_dir == "results/from_cli"
    assert merged.proteins.workers == 20
    assert merged.export.workers == 3
    assert merged.postprocess.workers == 4
    assert merged.export.patent_dict == "data/other.json"
    assert merged.common.log_level == "WARNING"


def test_cli_flags_only_switch_settings_on(tmp_path):
    path = write_config(tmp_path, "common:\n  debug: true\n  resume: true\n")
    config = load_run_config(path)

    merged = apply_cli_overrides(config, cli_args())

    assert merged.common.debug is True
    assert merged.common.resume is True

    plain = apply_cli_overrides(PipelineRunConfig(), cli_args(debug=True, continue_on_error=True))

    assert plain.common.debug is True
    assert plain.common.continue_on_error is True
    assert plain.common.resume is False


def test_unknown_extract_key_is_rejected(tmp_path):
    """A stale `alias_to_name:` must error, not be silently ignored."""
    path = tmp_path / "cfg.yaml"
    path.write_text("extract:\n  alias_to_name: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="alias_to_name"):
        load_run_config(path)


def test_resolve_stages_precedence():
    config = PipelineRunConfig(stages=("extract", "proteins"))

    assert resolve_stages(config, stages_arg="export,extract") == ("extract", "export")
    assert resolve_stages(config, from_stage="export") == ("export", "postprocess")
    assert resolve_stages(config) == ("extract", "proteins")
    assert resolve_stages(PipelineRunConfig()) == DEFAULT_STAGES
    assert resolve_stages(PipelineRunConfig(), stages_arg="all") == STAGES


def test_resolve_stages_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown stage 'agent9'"):
        resolve_stages(PipelineRunConfig(), stages_arg="agent9")

    with pytest.raises(ValueError, match="Unknown stage 'agent9'"):
        resolve_stages(PipelineRunConfig(), from_stage="agent9")


def test_validate_requires_output_dir():
    with pytest.raises(ValueError, match="Output directory is not set"):
        validate_run_config(PipelineRunConfig(input_path="a.zip"), ("extract",))


def test_validate_requires_input_only_for_extract():
    config = PipelineRunConfig(output_dir="results/run1")

    validate_run_config(config, ("export",))

    with pytest.raises(ValueError, match="Stage 'extract' needs input"):
        validate_run_config(config, ("extract",))


def test_validate_rejects_conflicting_inputs():
    config = PipelineRunConfig(output_dir="results/run1", input_path="a.zip", input_list="list.txt")

    with pytest.raises(ValueError, match="both an input path and an input list"):
        validate_run_config(config, ("extract",))


def test_validate_rejects_bad_log_level():
    config = PipelineRunConfig(output_dir="results/run1")
    config = apply_cli_overrides(config, cli_args(log_level="LOUD"))

    with pytest.raises(ValueError, match="Invalid log level"):
        validate_run_config(config, ("export",))


def test_artifact_paths_are_relative_to_output_dir():
    config = PipelineRunConfig(output_dir="results/run1")

    assert config.export_output_path == Path("results/run1/main_res.parquet")
    assert config.postprocess_input_path == Path("results/run1/main_res.parquet")
    assert config.postprocess_output_path == Path("results/run1/main_res_clean.parquet")

    absolute = PipelineRunConfig(
        output_dir="results/run1",
        postprocess=PostprocessSettings(input_file="/tmp/in.parquet"),
    )

    assert absolute.postprocess_input_path == Path("/tmp/in.parquet")


def test_config_to_dict_is_json_friendly():
    payload = config_to_dict(PipelineRunConfig(output_dir="results/run1", stages=("extract",)))

    assert payload["output_dir"] == "results/run1"
    assert payload["stages"] == ["extract"]
    assert payload["proteins"]["workers"] == 5


def test_example_config_is_valid():
    example = Path(__file__).resolve().parents[3] / "configs" / "pipeline.example.yaml"

    config = load_run_config(example)
    stages = resolve_stages(config)

    assert stages == STAGES
    validate_run_config(config, stages)
