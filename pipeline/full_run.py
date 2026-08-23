"""Full pipeline orchestration: extract -> proteins -> verify -> export -> postprocess.

Each stage is an independent step over the same output directory, so a run can
start from any stage as long as the artifacts of the previous ones are present.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .run_config import (
    PipelineRunConfig,
    config_to_dict,
    describe_config,
    validate_run_config,
)
from .utils import (
    async_read_file_list,
    save_git_commit_hash,
    save_incomplete_patents,
    setup_logging,
)

RUN_SUMMARY_FILENAME = "pipeline_run_summary.json"


class StageError(RuntimeError):
    """A pipeline stage could not run or failed."""


def has_resolved_patents(output_dir: str | Path) -> bool:
    """True if the results directory already holds per-patent extraction output."""
    root = Path(output_dir)
    if not root.is_dir():
        return False
    return any(root.glob("*/*_resolved.json"))


def validate_stage_inputs(config: PipelineRunConfig, stages: tuple[str, ...]) -> None:
    """Check artifacts of stages that are not part of this run."""
    needs_extraction = "proteins" in stages or "verify" in stages or "export" in stages
    if needs_extraction and "extract" not in stages:
        if not has_resolved_patents(config.output_dir):
            raise StageError(
                f"No *_resolved.json found in {config.output_dir}: "
                "run the 'extract' stage first or point --output-dir at an existing run"
            )

    if "postprocess" in stages and "export" not in stages:
        parquet = config.postprocess_input_path
        if not parquet.exists():
            raise StageError(
                f"Post-processing input Parquet not found: {parquet}. "
                "Run the 'export' stage first or set postprocess.input_file"
            )


async def collect_input_paths(config: PipelineRunConfig) -> list[str]:
    """Build the list of patent ZIP archives for the extraction stage."""
    if config.input_path:
        path = Path(config.input_path)
        if path.is_file():
            if path.suffix.lower() != ".zip":
                raise StageError(f"Input file must have a .zip extension: {path}")
            return [str(path)]
        if path.is_dir():
            paths = sorted(str(p.resolve()) for p in path.glob("*.zip"))
            if not paths:
                raise StageError(f"No ZIP files found in directory: {path}")
            return paths
        raise StageError(f"Input path not found: {path}")

    try:
        paths = await async_read_file_list(config.input_list)
    except FileNotFoundError as exc:
        raise StageError(str(exc)) from exc
    if not paths:
        raise StageError(f"No valid ZIP files found in the list: {config.input_list}")
    return paths


async def _stage_extract(config: PipelineRunConfig) -> dict[str, Any]:
    from .core import run_pipeline_async

    input_paths = await collect_input_paths(config)
    logging.info(f"Extraction input files: {len(input_paths)}")

    try:
        incomplete_patents = await run_pipeline_async(
            input_paths,
            str(config.output_dir),
            config.limit,
            config.common.resume,
            config.common.debug,
        )
    except KeyboardInterrupt:
        save_incomplete_patents(str(config.output_dir), [{
            'patent_id': 'unknown',
            'zip_file_path': '',
            'reason': 'Pipeline interrupted by user (KeyboardInterrupt)',
            'stage': 'pipeline_interrupted',
        }])
        raise
    except Exception as exc:
        save_incomplete_patents(str(config.output_dir), [{
            'patent_id': 'unknown',
            'zip_file_path': '',
            'reason': f'Critical pipeline error: {exc}',
            'stage': 'pipeline_critical_error',
        }])
        raise

    if incomplete_patents:
        save_incomplete_patents(str(config.output_dir), incomplete_patents)
        logging.warning(f"Found {len(incomplete_patents)} incomplete patents")

    return {
        "input_files": len(input_paths),
        "incomplete_patents": len(incomplete_patents or []),
        **read_extraction_counts(config.output_dir),
    }


def read_extraction_counts(output_dir: str | Path) -> dict[str, Any]:
    """Pull headline counters out of `pipeline_statistics.json` for the run summary.

    Per-patent extraction failures do not fail the stage, so without these
    counters a run where every patent failed still looks successful.
    """
    stats_path = Path(output_dir) / "pipeline_statistics.json"
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    counts = stats.get("data_counts") or {}
    errors = stats.get("error_analysis") or {}
    return {
        "patents_processed": counts.get("patents_processed"),
        "bioactivity_data_found": counts.get("bioactivity_data_found"),
        "agent1_errors": errors.get("total_agent1_errors"),
        "agent2_errors": errors.get("total_agent2_errors"),
    }


async def _stage_proteins(config: PipelineRunConfig) -> dict[str, Any]:
    from protein_postprocessor import ProteinPostprocessor
    from protein_postprocessor.config_postprocess import DEFAULT_FASTA_FILENAME

    settings = config.proteins
    protein_data_path = Path(settings.protein_data_path)
    if not protein_data_path.exists():
        raise StageError(
            f"Protein data directory not found: {protein_data_path}. "
            f"Run: python scripts/download_protein_data.py"
        )

    # Check the FASTA itself, not just the directory: an empty data/protein_data/
    # used to pass here and fail much later inside FastaGeneResolver, after the
    # extract stage had already spent LLM budget. This is the exact path handed
    # to the resolver below, so the check cannot pass and then read elsewhere.
    fasta_path = protein_data_path / DEFAULT_FASTA_FILENAME
    if not fasta_path.exists():
        raise StageError(
            f"UniProt FASTA not found: {fasta_path}. "
            f"Run: python scripts/download_protein_data.py"
        )

    postprocessor = ProteinPostprocessor(
        debug_mode=config.common.debug,
        debug_output_dir=str(Path(config.output_dir) / "protein_debug"),
        zip_test_dir=str(config.output_dir),
        fasta_path=fasta_path,
    )
    try:
        await postprocessor.process_results_dir(
            str(config.output_dir),
            num_workers=settings.workers,
            use_stage1_context=settings.stage1_context,
            reprocess_all=settings.reprocess_all,
            force=settings.force,
            singles_only=settings.singles_only,
            complexes_only=settings.complexes_only,
        )
    finally:
        await postprocessor.close()

    return {"results_dir": str(config.output_dir)}


async def _stage_verify(config: PipelineRunConfig) -> dict[str, Any]:
    from verify_llm_res import annotate_all

    settings = config.verify
    source_dir = settings.source_dir or config.input_path
    if not source_dir:
        raise StageError(
            "The 'verify' stage needs the patent ZIP directory. "
            "Set verify.source_dir in the config, or use --verify-source-dir, "
            "or let it default from input.path."
        )
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise StageError(f"Verify source directory not found: {source_path}")

    summary = await asyncio.to_thread(
        annotate_all,
        config.output_dir,
        source_dir,
        workers=settings.workers,
        force=settings.force,
        force_hallu=settings.force_hallu,
    )

    if summary["failed"] == summary["total"] and summary["total"] > 0:
        raise StageError(
            f"All {summary['total']} patents failed hallucination annotation"
        )

    return summary


async def _stage_export(config: PipelineRunConfig) -> dict[str, Any]:
    from export_table import run_bindingdb_processing

    settings = config.export
    output_file = config.export_output_path
    patent_dict = settings.patent_dict
    if patent_dict and not Path(patent_dict).exists():
        logging.warning(f"Patent mapping dictionary not found, continuing without it: {patent_dict}")
        patent_dict = None

    cache_dir = settings.cache_dir
    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent.parent / "cache")

    success, message, elapsed = await asyncio.to_thread(
        run_bindingdb_processing,
        input_dir=str(config.output_dir),
        output_file=str(output_file),
        workers=settings.workers,
        cache_dir=cache_dir,
        patent_dict_file=patent_dict,
        batch_size=settings.batch_size,
        force=settings.force,
        use_opsin=settings.use_opsin,
    )

    if not success:
        raise StageError(f"BindingDB export failed after {elapsed:.1f}s: {message}")

    logging.info(f"BindingDB export finished in {elapsed:.1f}s: {message}")
    return {"parquet": str(output_file)}


async def _stage_postprocess(config: PipelineRunConfig) -> dict[str, Any]:
    from final_postprocessing.run import FinalPostprocessingConfig, run_final_postprocessing

    settings = config.postprocess
    input_path = config.postprocess_input_path
    if not input_path.exists():
        raise StageError(f"Post-processing input Parquet not found: {input_path}")

    output_path = config.postprocess_output_path
    fp_config = FinalPostprocessingConfig(
        workers=settings.workers,
        chunk_size=settings.chunk_size,
        enrichment_only=settings.enrichment_only,
        skip_add_clean_best=settings.skip_add_clean_best,
        skip_phenotypic=settings.skip_phenotypic,
        skip_fragments=settings.skip_fragments,
        skip_duplicates=settings.skip_duplicates,
        heavy_atoms=settings.heavy_atoms,
        min_count=settings.min_count,
        triplet_count=settings.triplet_count,
        cache_path=Path(settings.cache_path) if settings.cache_path else None,
        no_cache=settings.no_cache,
    )

    await asyncio.to_thread(run_final_postprocessing, input_path, output_path, fp_config)
    return {"parquet": str(output_path)}


StageHandler = Callable[[PipelineRunConfig], Awaitable[dict[str, Any]]]

STAGE_HANDLERS: dict[str, StageHandler] = {
    "extract": _stage_extract,
    "proteins": _stage_proteins,
    "verify": _stage_verify,
    "export": _stage_export,
    "postprocess": _stage_postprocess,
}


def _write_run_summary(
    config: PipelineRunConfig,
    stages: tuple[str, ...],
    stage_results: list[dict[str, Any]],
    total_seconds: float,
) -> Optional[Path]:
    summary_path = Path(config.output_dir) / RUN_SUMMARY_FILENAME
    payload = {
        "stages_requested": list(stages),
        "total_seconds": round(total_seconds, 3),
        "stages": stage_results,
        "config": config_to_dict(config),
    }
    try:
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logging.error(f"Could not write run summary {summary_path}: {exc}")
        return None
    return summary_path


async def run_full_pipeline(config: PipelineRunConfig, stages: tuple[str, ...]) -> int:
    """Run the selected stages in canonical order. Returns a process exit code."""
    validate_run_config(config, stages)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.common.log_level, str(output_dir))

    validate_stage_inputs(config, stages)

    if config.common.debug:
        save_git_commit_hash(str(output_dir))

    logging.info("Pipeline run configuration:\n%s", describe_config(config, stages))

    stage_results: list[dict[str, Any]] = []
    exit_code = 0
    run_start = time.time()

    try:
        for index, stage in enumerate(stages, start=1):
            logging.info(f"===== Stage {index}/{len(stages)}: {stage} =====")
            started = time.time()
            try:
                artifacts = await STAGE_HANDLERS[stage](config)
            except Exception as exc:
                duration = time.time() - started
                logging.error(f"Stage '{stage}' failed after {duration:.1f}s: {exc}")
                stage_results.append({
                    "stage": stage,
                    "status": "failed",
                    "seconds": round(duration, 3),
                    "error": str(exc),
                })
                exit_code = 1
                if not config.common.continue_on_error:
                    remaining = stages[index:]
                    for skipped in remaining:
                        stage_results.append({"stage": skipped, "status": "skipped"})
                    if remaining:
                        logging.error(f"Skipping remaining stages: {', '.join(remaining)}")
                    break
            else:
                duration = time.time() - started
                logging.info(f"Stage '{stage}' finished in {duration:.1f}s")
                stage_results.append({
                    "stage": stage,
                    "status": "ok",
                    "seconds": round(duration, 3),
                    "artifacts": artifacts or {},
                })
    finally:
        summary_path = _write_run_summary(config, stages, stage_results, time.time() - run_start)
        if summary_path:
            logging.info(f"Run summary: {summary_path}")

    if exit_code == 0:
        logging.info("Pipeline finished.")
    else:
        logging.error("Pipeline finished with errors.")
    return exit_code
