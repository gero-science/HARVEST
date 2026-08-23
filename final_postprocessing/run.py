"""One-command runner for optional Parquet final post-processing.

Default chain (excludes ``add_final_structure``):

1. ``add_clean_best`` — year, clean_smiles, clean_inchi_key
2. ``filter_phenotypic_assays`` — drop phenotypic / off-target assays
3. ``filter_fragments`` — fragment / frequent-triplet cleanup (needs clean_smiles)
4. ``filter_duplicates`` — bad activity quadruplets

Hallucination filtering is handled earlier in the pipeline: the ``verify``
stage writes ``_hallu.json`` sidecars, and the ``export`` stage reads them
to drop flagged rows at build time.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .add_clean_best import default_workers, run_add_clean_best

PACKAGE_DIR = Path(__file__).resolve().parent

STEPS = (
    "add_clean_best",
    "filter_phenotypic_assays",
    "filter_fragments",
    "filter_duplicates",
)


@dataclass(frozen=True)
class FinalPostprocessingConfig:
    workers: int | None = None
    chunk_size: int = 50_000
    enrichment_only: bool = False
    skip_add_clean_best: bool = False
    skip_phenotypic: bool = False
    skip_fragments: bool = False
    skip_duplicates: bool = False
    heavy_atoms: int = 15
    min_count: int = 5
    triplet_count: int = 5
    cache_path: Path | None = None
    no_cache: bool = False


def _run_script(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(PACKAGE_DIR / script), *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _parquet_has_column(path: Path, column: str) -> bool:
    import pyarrow.parquet as pq

    return column in pq.ParquetFile(path).schema_arrow.names


def _enabled_steps(
    config: FinalPostprocessingConfig,
    *,
    input_has_clean_smiles: bool = True,
) -> list[str]:
    if config.enrichment_only and config.skip_add_clean_best:
        raise ValueError("Cannot combine --enrichment-only with --skip-add-clean-best")

    steps: list[str] = []
    if not config.skip_add_clean_best:
        steps.append("add_clean_best")
    if config.enrichment_only:
        if not steps:
            raise ValueError("--enrichment-only requires add_clean_best")
        return steps

    if not config.skip_phenotypic:
        steps.append("filter_phenotypic_assays")

    skip_fragments = config.skip_fragments
    if (
        not skip_fragments
        and config.skip_add_clean_best
        and not input_has_clean_smiles
    ):
        # filter_fragments reads clean_smiles; cannot run without enrichment
        skip_fragments = True
    if not skip_fragments:
        steps.append("filter_fragments")

    if not config.skip_duplicates:
        steps.append("filter_duplicates")
    if not steps:
        raise ValueError("No post-processing steps enabled")
    return steps


def run_final_postprocessing(
    input_path: str | Path,
    output_path: str | Path,
    config: FinalPostprocessingConfig | None = None,
) -> Path:
    """Run the optional final post-processing chain and write ``output_path``."""
    config = config or FinalPostprocessingConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workers = config.workers if config.workers is not None else default_workers()
    has_clean = _parquet_has_column(input_path, "clean_smiles")
    auto_skip_fragments = (
        config.skip_add_clean_best
        and not config.skip_fragments
        and not has_clean
    )
    steps = _enabled_steps(config, input_has_clean_smiles=has_clean)
    print("=" * 72)
    print("Final Parquet post-processing")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Steps:  {' -> '.join(steps)}")
    if auto_skip_fragments:
        print(
            "Note: skipping filter_fragments (no clean_smiles column and "
            "add_clean_best disabled)"
        )
    # Resolve the cache default here rather than letting add_clean_best derive
    # it from its own output: that output is a temporary step file whenever
    # add_clean_best is not the last step, so the cache would be built inside
    # the TemporaryDirectory below and deleted with it. Every run would then
    # re-standardize the whole corpus while reporting a cache path that never
    # gets written.
    cache_path = config.cache_path
    if not config.no_cache and cache_path is None:
        cache_path = output_path.parent / "clean_smiles_cache.sqlite"

    if "add_clean_best" in steps:
        print(f"Workers: {workers}")
        if config.no_cache:
            print("Cache:  disabled")
        else:
            print(f"Cache:  {cache_path}")
    print("=" * 72)

    with tempfile.TemporaryDirectory(prefix="final_postprocessing_") as tmp:
        tmp_dir = Path(tmp)
        current = input_path

        for index, step in enumerate(steps):
            is_last = index == len(steps) - 1
            next_path = output_path if is_last else tmp_dir / f"step_{index:02d}_{step}.parquet"

            print(f"\n--- Step {index + 1}/{len(steps)}: {step} ---")
            if step == "add_clean_best":
                run_add_clean_best(
                    current,
                    next_path,
                    workers=workers,
                    chunk_size=config.chunk_size,
                    cache_path=cache_path,
                    no_cache=config.no_cache,
                )
            elif step == "filter_phenotypic_assays":
                _run_script(
                    "filter_phenotypic_assays.py",
                    [str(current), str(next_path)],
                )
            elif step == "filter_fragments":
                _run_script(
                    "filter_fragments.py",
                    [
                        str(current),
                        str(next_path),
                        "--heavy-atoms",
                        str(config.heavy_atoms),
                        "--min-count",
                        str(config.min_count),
                        "--triplet-count",
                        str(config.triplet_count),
                    ],
                )
            elif step == "filter_duplicates":
                _run_script(
                    "filter_duplicates.py",
                    ["--input", str(current), "--output", str(next_path)],
                )
            else:  # pragma: no cover
                raise ValueError(f"Unknown step: {step}")

            current = next_path

        if current.resolve() != output_path.resolve():
            shutil.copy2(current, output_path)

    print("\n" + "=" * 72)
    print(f"Done. Wrote {output_path}")
    print("=" * 72)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m final_postprocessing",
        description=(
            "Optional post-BindingDB Parquet cleanup: add year/clean_smiles/clean_inchi_key "
            "columns and apply cleaning filters. Does not run add_final_structure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m final_postprocessing results/run/main_res.parquet results/run/main_res_clean.parquet
  python -m final_postprocessing in.parquet out.parquet --enrichment-only --workers 8
  python -m final_postprocessing in.parquet out.parquet --skip-add-clean-best
  python -m final_postprocessing in.parquet out.parquet --skip-duplicates --triplet-count 0
        """.strip(),
    )
    parser.add_argument("input_file", type=Path, help="Input BindingDB-style parquet")
    parser.add_argument("output_file", type=Path, help="Output parquet path")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Workers for add_clean_best (default: cpu_count-2 = {default_workers()})",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Chunk size for add_clean_best")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="SQLite cache for clean_smiles (default: <output_dir>/clean_smiles_cache.sqlite)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk cache for clean_smiles",
    )
    parser.add_argument(
        "--enrichment-only",
        action="store_true",
        help="Only add year/clean_smiles/clean_inchi_key; skip filters",
    )
    parser.add_argument(
        "--skip-add-clean-best",
        action="store_true",
        help=(
            "Skip year/clean_smiles enrichment; run filters only. "
            "If input has no clean_smiles, filter_fragments is skipped automatically."
        ),
    )
    parser.add_argument("--skip-phenotypic", action="store_true", help="Skip phenotypic assay filter")
    parser.add_argument("--skip-fragments", action="store_true", help="Skip fragment/triplet filter")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip duplicate quadruplet filter")
    parser.add_argument("--heavy-atoms", type=int, default=15, help="Fragment filter heavy-atom threshold")
    parser.add_argument("--min-count", type=int, default=5, help="Fragment filter min SMILES count")
    parser.add_argument(
        "--triplet-count",
        type=int,
        default=5,
        help="Fragment filter triplet threshold (0 disables)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = FinalPostprocessingConfig(
        workers=args.workers,
        chunk_size=args.chunk_size,
        enrichment_only=args.enrichment_only,
        skip_add_clean_best=args.skip_add_clean_best,
        skip_phenotypic=args.skip_phenotypic,
        skip_fragments=args.skip_fragments,
        skip_duplicates=args.skip_duplicates,
        heavy_atoms=args.heavy_atoms,
        min_count=args.min_count,
        triplet_count=args.triplet_count,
        cache_path=args.cache_path,
        no_cache=args.no_cache,
    )
    run_final_postprocessing(args.input_file, args.output_file, config)
    return 0
