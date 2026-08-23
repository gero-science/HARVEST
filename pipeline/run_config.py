"""Declarative configuration of a full pipeline run.

A run is a sequence of stages (see :data:`STAGES`) with per-stage settings.
Settings come from an optional YAML file and can be overridden from the CLI.
Secrets and LLM settings stay in ``.env`` / :mod:`config` and never appear here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Optional

STAGES: tuple[str, ...] = ("extract", "proteins", "verify", "export", "postprocess")

# Stages run when none are requested: everything except ``postprocess``, which
# is opt-in via ``--stages all``.
DEFAULT_STAGES: tuple[str, ...] = ("extract", "proteins", "verify", "export")

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class CommonSettings:
    log_level: str = "INFO"
    debug: bool = False
    resume: bool = False
    continue_on_error: bool = False


@dataclass(frozen=True)
class ExtractSettings:
    """No settings today. Kept so an ``extract:`` block in a config file is still
    validated, and a stale key such as the removed ``alias_to_name`` raises a
    clear error instead of being silently ignored."""


@dataclass(frozen=True)
class ProteinSettings:
    workers: int = 5
    stage1_context: bool = True
    reprocess_all: bool = True
    force: bool = False
    singles_only: bool = False
    complexes_only: bool = False
    protein_data_path: str = "data/protein_data"


@dataclass(frozen=True)
class VerifySettings:
    """Hallucination annotation: write ``_hallu.json`` sidecars."""
    # Directory of patent ZIP files.  Defaults to ``input.path`` at runtime.
    source_dir: Optional[str] = None
    workers: int = 16
    force: bool = False
    force_hallu: bool = False


@dataclass(frozen=True)
class ExportSettings:
    workers: int = 6
    batch_size: int = 200
    patent_dict: Optional[str] = "curated_data/patent_mapping.csv"
    use_opsin: bool = False
    output_file: str = "main_res.parquet"
    force: bool = False
    cache_dir: Optional[str] = None


@dataclass(frozen=True)
class PostprocessSettings:
    # None means "use the export stage output".
    input_file: Optional[str] = None
    output_file: str = "main_res_clean.parquet"
    workers: Optional[int] = None
    chunk_size: int = 50_000
    enrichment_only: bool = False
    skip_add_clean_best: bool = False
    skip_phenotypic: bool = False
    skip_fragments: bool = False
    skip_duplicates: bool = False
    heavy_atoms: int = 15
    min_count: int = 5
    triplet_count: int = 5
    cache_path: Optional[str] = None
    no_cache: bool = False


@dataclass(frozen=True)
class PipelineRunConfig:
    input_path: Optional[str] = None
    input_list: Optional[str] = None
    limit: Optional[int] = None
    output_dir: Optional[str] = None
    # None means "not specified"; resolve_stages() decides the final list.
    stages: Optional[tuple[str, ...]] = None
    common: CommonSettings = field(default_factory=CommonSettings)
    extract: ExtractSettings = field(default_factory=ExtractSettings)
    proteins: ProteinSettings = field(default_factory=ProteinSettings)
    verify: VerifySettings = field(default_factory=VerifySettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    postprocess: PostprocessSettings = field(default_factory=PostprocessSettings)

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a stage artifact path relative to ``output_dir``."""
        path = Path(value)
        if path.is_absolute() or self.output_dir is None:
            return path
        return Path(self.output_dir) / path

    @property
    def export_output_path(self) -> Path:
        return self.resolve_path(self.export.output_file)

    @property
    def postprocess_input_path(self) -> Path:
        if self.postprocess.input_file:
            return self.resolve_path(self.postprocess.input_file)
        return self.export_output_path

    @property
    def postprocess_output_path(self) -> Path:
        return self.resolve_path(self.postprocess.output_file)


_TOP_LEVEL_KEYS = (
    "input",
    "output",
    "stages",
    "common",
    "extract",
    "proteins",
    "verify",
    "export",
    "postprocess",
)


def _build_section(section_type: type, data: Any, section_name: str):
    if data is None:
        return section_type()
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping, got {type(data).__name__}")

    known = {f.name for f in fields(section_type)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in config section '{section_name}': {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(known))}"
        )
    return section_type(**data)


def _normalize_stage_list(value: Any, source: str) -> tuple[str, ...]:
    """Normalize a stage list into canonical order, validating names."""
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value]
    else:
        raise ValueError(f"{source} must be a list or comma-separated string, got {type(value).__name__}")

    items = [item for item in items if item]
    if not items:
        raise ValueError(f"{source} is empty")

    if len(items) == 1 and items[0].lower() == "all":
        return STAGES

    requested = set()
    for item in items:
        name = item.lower()
        if name not in STAGES:
            raise ValueError(
                f"Unknown stage '{item}' in {source}. Allowed: {', '.join(STAGES)} (or 'all')"
            )
        requested.add(name)

    return tuple(stage for stage in STAGES if stage in requested)


def load_run_config(config_path: str | Path) -> PipelineRunConfig:
    """Load a YAML run config. Unknown keys are errors, not silent typos."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyYAML is required to read pipeline config files: pip install -r requirements.txt"
        ) from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline config must be a mapping at top level: {path}")

    unknown = sorted(set(raw) - set(_TOP_LEVEL_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown top-level key(s) in {path}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(_TOP_LEVEL_KEYS)}"
        )

    input_section = raw.get("input") or {}
    if not isinstance(input_section, dict):
        raise ValueError("Config section 'input' must be a mapping")
    unknown_input = sorted(set(input_section) - {"path", "list", "limit"})
    if unknown_input:
        raise ValueError(
            f"Unknown key(s) in config section 'input': {', '.join(unknown_input)}. "
            "Allowed: limit, list, path"
        )

    output_section = raw.get("output") or {}
    if not isinstance(output_section, dict):
        raise ValueError("Config section 'output' must be a mapping")
    unknown_output = sorted(set(output_section) - {"dir"})
    if unknown_output:
        raise ValueError(
            f"Unknown key(s) in config section 'output': {', '.join(unknown_output)}. Allowed: dir"
        )

    stages = raw.get("stages")
    stages_tuple = _normalize_stage_list(stages, f"'stages' in {path}") if stages is not None else None

    return PipelineRunConfig(
        input_path=input_section.get("path"),
        input_list=input_section.get("list"),
        limit=input_section.get("limit"),
        output_dir=output_section.get("dir"),
        stages=stages_tuple,
        common=_build_section(CommonSettings, raw.get("common"), "common"),
        extract=_build_section(ExtractSettings, raw.get("extract"), "extract"),
        proteins=_build_section(ProteinSettings, raw.get("proteins"), "proteins"),
        verify=_build_section(VerifySettings, raw.get("verify"), "verify"),
        export=_build_section(ExportSettings, raw.get("export"), "export"),
        postprocess=_build_section(PostprocessSettings, raw.get("postprocess"), "postprocess"),
    )


def _replace_section(section, **updates):
    filtered = {key: value for key, value in updates.items() if value is not None}
    return replace(section, **filtered) if filtered else section


def apply_cli_overrides(config: PipelineRunConfig, args: Any) -> PipelineRunConfig:
    """Apply explicitly passed CLI values on top of the config file.

    Value arguments override when they are not None. Boolean store_true flags can
    only switch a setting on; use the config file to switch something off.
    """
    def value(name: str):
        return getattr(args, name, None)

    def flag(name: str) -> Optional[bool]:
        return True if getattr(args, name, False) else None

    top_updates = {
        "input_path": value("input_path"),
        "input_list": value("input_list"),
        "limit": value("limit"),
        "output_dir": value("output_dir"),
    }

    common = _replace_section(
        config.common,
        log_level=value("log_level"),
        debug=flag("debug"),
        resume=flag("resume"),
        continue_on_error=flag("continue_on_error"),
    )

    extract = config.extract

    verify = _replace_section(
        config.verify,
        source_dir=value("verify_source_dir"),
        workers=value("verify_workers"),
    )

    proteins = _replace_section(
        config.proteins,
        workers=value("protein_workers"),
        protein_data_path=value("protein_data_path"),
    )

    export = _replace_section(
        config.export,
        workers=value("export_workers"),
        patent_dict=value("bdb_json"),
    )

    postprocess = _replace_section(
        config.postprocess,
        workers=value("postprocess_workers"),
    )

    return replace(
        config,
        **{key: val for key, val in top_updates.items() if val is not None},
        common=common,
        extract=extract,
        proteins=proteins,
        verify=verify,
        export=export,
        postprocess=postprocess,
    )


def resolve_stages(
    config: PipelineRunConfig,
    *,
    stages_arg: Any = None,
    from_stage: Optional[str] = None,
) -> tuple[str, ...]:
    """Pick the stage list to run.

    Precedence: ``--stages`` > ``--from-stage`` > config ``stages`` >
    :data:`DEFAULT_STAGES`. ``--from-stage X`` runs X and every later stage,
    including ``postprocess``.
    """
    if stages_arg:
        return _normalize_stage_list(stages_arg, "--stages")

    if from_stage:
        name = str(from_stage).strip().lower()
        if name not in STAGES:
            raise ValueError(
                f"Unknown stage '{from_stage}' in --from-stage. Allowed: {', '.join(STAGES)}"
            )
        return STAGES[STAGES.index(name):]

    if config.stages:
        return config.stages

    return DEFAULT_STAGES


def validate_run_config(config: PipelineRunConfig, stages: tuple[str, ...]) -> None:
    """Fail fast on configurations that cannot produce the requested stages."""
    if not stages:
        raise ValueError("No pipeline stages selected")

    if config.common.log_level.upper() not in LOG_LEVELS:
        raise ValueError(
            f"Invalid log level '{config.common.log_level}'. Allowed: {', '.join(LOG_LEVELS)}"
        )

    if not config.output_dir:
        raise ValueError("Output directory is not set: use --output-dir or 'output.dir' in the config")

    if "extract" in stages:
        if config.input_path and config.input_list:
            raise ValueError("Cannot use both an input path and an input list")
        if not config.input_path and not config.input_list:
            raise ValueError(
                "Stage 'extract' needs input: use --input-path/--input-list or 'input' in the config"
            )

    if config.proteins.singles_only and config.proteins.complexes_only:
        raise ValueError("Cannot combine proteins.singles_only with proteins.complexes_only")


def describe_config(config: PipelineRunConfig, stages: tuple[str, ...]) -> str:
    """Human-readable summary of a run, used in logs and the run summary file."""
    lines = [
        f"Stages: {' -> '.join(stages)}",
        f"Output dir: {config.output_dir}",
    ]
    if "extract" in stages:
        source = config.input_path or config.input_list
        lines.append(f"Input: {source} (limit={config.limit})")
    if "proteins" in stages:
        lines.append(
            f"Proteins: workers={config.proteins.workers}, "
            f"stage1_context={config.proteins.stage1_context}, "
            f"reprocess_all={config.proteins.reprocess_all}, force={config.proteins.force}"
        )
    if "verify" in stages:
        src = config.verify.source_dir or config.input_path or "(input.path)"
        lines.append(
            f"Verify: source_dir={src}, workers={config.verify.workers}, "
            f"force={config.verify.force}"
        )
    if "export" in stages:
        lines.append(
            f"Export: workers={config.export.workers}, batch_size={config.export.batch_size}, "
            f"use_opsin={config.export.use_opsin} -> {config.export_output_path}"
        )
    if "postprocess" in stages:
        lines.append(
            f"Postprocess: {config.postprocess_input_path} -> {config.postprocess_output_path}"
        )
    return "\n".join(lines)


def config_to_dict(value: Any) -> Any:
    """Serialize a config (or a section) into plain JSON-friendly data."""
    if is_dataclass(value):
        return {f.name: config_to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [config_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
