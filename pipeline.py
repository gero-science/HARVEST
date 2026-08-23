#!/usr/bin/env python3
"""
CLI entry point for the patent bioactivity pipeline.

A run is a sequence of stages over one output directory:

1. ``extract``     - Agent 1 extraction
2. ``proteins``    - ``protein_postprocessor`` (single_proteins / protein_complexes)
3. ``verify``      - hallucination annotation (writes ``_hallu.json`` sidecars)
4. ``export``      - BindingDB Parquet export (``main_res.parquet``)
5. ``postprocess`` - ``final_postprocessing`` cleanup chain (``main_res_clean.parquet``)

Stages are selected with ``--stages`` / ``--from-stage``, or in a YAML config
passed via ``--config`` (see ``configs/pipeline.example.yaml``). CLI arguments
override config values. LLM credentials stay in ``.env``.
"""

import argparse
import asyncio
import sys

from pipeline.full_run import StageError, run_full_pipeline
from pipeline.run_config import (
    STAGES,
    PipelineRunConfig,
    apply_cli_overrides,
    load_run_config,
    resolve_stages,
)


def str_to_bool(value):
    """Parse string boolean values correctly."""
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patent bioactivity pipeline: extraction, protein resolution, Parquet export and cleanup.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Usage examples:

1. Extraction only:
   python pipeline.py --input-path patents.zip --output-dir results --stages extract

2. Everything end to end, including final_postprocessing:
   python pipeline.py --input-path data/hallu_100 --output-dir results/run1 --stages all

3. Catch up on an existing results directory (no LLM extraction):
   python pipeline.py --output-dir results/run1 --from-stage export

4. Run from a config file:
   python pipeline.py --config configs/pipeline.example.yaml
"""
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-path", type=str, help="Path to a single ZIP archive or a directory of archives.")
    group.add_argument("--input-list", type=str, help="Path to a text file listing ZIP archives (one file per line).")

    parser.add_argument("--output-dir", type=str, help="Directory for saving results (required unless set in --config).")
    parser.add_argument("--config", type=str, help="YAML config with stages and per-stage settings.")
    parser.add_argument(
        "--stages",
        type=str,
        help=f"Comma-separated stages to run, or 'all'. Available: {', '.join(STAGES)}.",
    )
    parser.add_argument(
        "--from-stage",
        type=str,
        help="Run this stage and every later one (includes postprocess).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of patents to process (applies across all input files).")
    parser.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.add_argument("--resume", action="store_true", help="Resume mode: shows additional information about continuing a previous session. Already processed patents are always skipped.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: saves intermediate data (tables and chemical compounds) in patent folders.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running later stages after a stage fails (exit code stays non-zero).")
    parser.add_argument("--protein-data-path", type=str, default=None, help="Path to the protein data directory (checked before the proteins stage).")
    parser.add_argument("--protein-workers", type=int, default=None, help="Parallel patents in the proteins stage.")
    parser.add_argument("--verify-source-dir", type=str, default=None, help="Patent ZIP directory for the verify stage (defaults to --input-path).")
    parser.add_argument("--verify-workers", type=int, default=None, help="Workers for the hallucination annotation stage.")
    parser.add_argument("--export-workers", type=int, default=None, help="Workers for the BindingDB export stage.")
    parser.add_argument("--postprocess-workers", type=int, default=None, help="Workers for add_clean_best in the postprocess stage.")
    parser.add_argument("--patent-dict", "--bdb-json", dest="bdb_json", type=str, default=None, help="Patent application→publication mapping (CSV or JSONL). Default: curated_data/patent_mapping.csv.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_run_config(args.config) if args.config else PipelineRunConfig()
        config = apply_cli_overrides(config, args)
        stages = resolve_stages(
            config,
            stages_arg=args.stages,
            from_stage=args.from_stage,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    try:
        return asyncio.run(run_full_pipeline(config, stages))
    except (StageError, ValueError) as exc:
        print(f"Pipeline configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Pipeline interrupted by user (Ctrl+C)", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
