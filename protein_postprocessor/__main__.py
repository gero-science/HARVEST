#!/usr/bin/env python3
"""
Protein Postprocessor - parallel per-patent protein processing.

Reads *_resolved.json from patent folders and processes them through the LLM in parallel.
Saves results to:
- single_proteins.json - individual proteins
- protein_complexes.json - protein complexes

Usage:
    python -m protein_postprocessor results/next_17000
    python -m protein_postprocessor results/next_17000 --workers 100
    python -m protein_postprocessor results/next_17000 --singles-only
    python -m protein_postprocessor results/next_17000 --complexes-only
    python -m protein_postprocessor results/next_17000 --force
"""

import argparse
import asyncio
import logging
from pathlib import Path

from .protein_postprocessor import ProteinPostprocessor
from pipeline.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Protein Postprocessor - per-patent protein processing."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Pipeline results directory (contains patent folders)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--singles-only",
        action="store_true",
        help="Process individual proteins only (no complexes)"
    )
    parser.add_argument(
        "--complexes-only",
        action="store_true",
        help="Process complexes only (no individual proteins)"
    )
    parser.add_argument(
        "--with-stage1-context",
        action="store_true",
        dest="with_stage1_context",
        default=True,
        help="Pass Stage 1 context to the LLM (default: enabled)"
    )
    parser.add_argument(
        "--no-stage1-context",
        action="store_false",
        dest="with_stage1_context",
        help="Do not pass Stage 1 context to the LLM"
    )
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        dest="reprocess_all",
        default=True,
        help="Process all proteins, ignoring sequence in _resolved.json (default: enabled)"
    )
    parser.add_argument(
        "--no-reprocess-all",
        action="store_false",
        dest="reprocess_all",
        help="Process only proteins without sequence in _resolved.json"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess patents even if single_proteins.json already exists"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Directory does not exist: {results_dir}")
        return 1
    
    # Configure logging with file output
    setup_logging(args.log_level, str(results_dir), "protein_postprocessor.log")
    
    # Create postprocessor
    postprocessor = ProteinPostprocessor(
        debug_mode=args.debug,
        debug_output_dir=str(results_dir / "protein_debug"),
        zip_test_dir=str(results_dir)
    )
    
    async def run():
        try:
            await postprocessor.process_results_dir(
                str(results_dir),
                num_workers=args.workers,
                use_stage1_context=args.with_stage1_context,
                reprocess_all=args.reprocess_all,
                force=args.force,
                singles_only=args.singles_only,
                complexes_only=args.complexes_only
            )
        finally:
            await postprocessor.close()
    
    asyncio.run(run())
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
