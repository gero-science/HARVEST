"""CLI entry point: ``python -m verify_llm_res``.

Annotate per-patent LLM results with hallucination flags.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .annotate import annotate_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m verify_llm_res",
        description=(
            "Run the hallucination detector against patent XML sources and write "
            "a _hallu.json sidecar next to each _resolved.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory with per-patent LLM results (each in its own subdirectory).",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory of patent ZIP files named {patent_number}.zip.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel workers (default: 16).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing _hallu.json files.",
    )
    parser.add_argument(
        "--force-hallu",
        action="store_true",
        help="Re-run only patents whose existing _hallu.json has non-empty flags.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    summary = annotate_all(
        args.results_dir,
        args.source_dir,
        workers=args.workers,
        force=args.force,
        force_hallu=args.force_hallu,
    )
    return 1 if summary["failed"] == summary["total"] and summary["total"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
