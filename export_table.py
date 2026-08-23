#!/usr/bin/env python3
"""
BindingDB Pipeline - assemble the final Parquet table from patents.

Reads data from patent directories:
- _resolved.json - links between compounds and proteins
- single_proteins.json - protein information
- cdx_results.json - CDX parsing results (written during extraction)

Normalizes and enriches data, outputs a Parquet table in BindingDB format.

Usage:
    python export_table.py results/swap81 output.parquet --workers 8
"""

import argparse

from bindingdb_export.artifacts import (
    CDX_RESULTS_FILENAME,
    get_patent_directories,
    load_cdx_results,
    load_patent_number_dict,
    load_stage1_organisms,
)
from bindingdb_export.batches import merge_parquet_batches
from bindingdb_export.cdx import apply_cdx_results_to_bindings, extract_chem_num_from_chemical_id
from bindingdb_export.enrichment import process_single_patent
from bindingdb_export.organisms import (
    classify_species_source,
    is_nonspecific_organism,
    normalize_organism,
)
from bindingdb_export.patent_worker import process_patent_worker
from bindingdb_export.proteins import load_protein_complexes, load_single_proteins
from bindingdb_export.processing import run_bindingdb_processing
from bindingdb_export.rows import binding_to_parquet_row
from bindingdb_export.schema import OUTPUT_COLUMNS, PARQUET_SCHEMA

__all__ = [
    "CDX_RESULTS_FILENAME",
    "OUTPUT_COLUMNS",
    "PARQUET_SCHEMA",
    "apply_cdx_results_to_bindings",
    "binding_to_parquet_row",
    "classify_species_source",
    "extract_chem_num_from_chemical_id",
    "get_patent_directories",
    "is_nonspecific_organism",
    "load_cdx_results",
    "load_patent_number_dict",
    "load_protein_complexes",
    "load_single_proteins",
    "load_stage1_organisms",
    "merge_parquet_batches",
    "normalize_organism",
    "process_patent_worker",
    "process_single_patent",
    "run_bindingdb_processing",
]


def main():
    parser = argparse.ArgumentParser(
        description="Build a Parquet table from patents in BindingDB format"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory with patent results (results/swap81)",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output Parquet file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caches",
    )
    parser.add_argument(
        "--patent-dict",
        type=str,
        default=None,
        help="Patent application→publication mapping (CSV or JSONL). Default: curated_data/patent_mapping.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of patents per batch (default: 200)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Ignore progress and start from scratch",
    )
    parser.add_argument(
        "--use-opsin",
        action="store_true",
        help="Enable PyOpsin for IUPAC→SMILES conversion (uses a lot of memory)",
    )
    parser.add_argument(
        "--structure-source",
        choices=["cdx", "mol", "none"],
        default="cdx",
        help="Source for structural-drawing SMILES: cdx (default), mol, or none",
    )

    args = parser.parse_args()

    success, message, elapsed_time = run_bindingdb_processing(
        args.input_dir,
        args.output_file,
        args.workers,
        args.cache_dir,
        args.patent_dict,
        args.batch_size,
        args.force,
        use_opsin=args.use_opsin,
        structure_source=args.structure_source,
    )

    if success:
        print(f"✓ {message}")
        print(f"Time: {elapsed_time:.2f} seconds")
    else:
        print(f"✗ {message}")
        print(f"Time: {elapsed_time:.2f} seconds")
        exit(1)


if __name__ == "__main__":
    main()
