#!/usr/bin/env python3
"""
Script to filter phenotypic/off-target assay records from parquet files.
Uses DuckDB for fast processing.

Usage:
    python filter_phenotypic_assays.py input.parquet output.parquet [--dry-run]
"""

import argparse
import duckdb


# Filter condition - records matching these patterns will be REMOVED
FILTER_CONDITION = """
    -- Core keywords (phenotypic assays)
    assay ILIKE '%proliferat%' OR
    assay ILIKE '%antiproliferat%' OR
    assay ILIKE '%cytotoxic%' OR
    assay ILIKE '%viability%' OR
    assay ILIKE '%growth%' OR
    assay ILIKE '%death%' OR
    assay ILIKE '%killing%' OR
    assay ILIKE '%cell activity%' OR
    
    -- Patch voltage clamp electrophysiology (per word)
    assay ILIKE '%patch%' OR
    assay ILIKE '%voltage%' OR
    assay ILIKE '%clamp%' OR
    assay ILIKE '%electrophysiology%' OR
    
    -- HEK and Transfection (overexpression systems)
    assay ILIKE '%HEK%' OR
    assay ILIKE '%transfection%' OR
    
    -- Additional assay methods
    assay ILIKE '%endocytosis%' OR
    assay ILIKE '%MTT%' OR
    assay ILIKE '%ATPlite%' OR
    assay ILIKE '%alamar%blue%' OR
    assay ILIKE '%alamarblue%' OR
    assay ILIKE '%BrdU%' OR
    assay ILIKE '%CCK-8%' OR
    assay ILIKE '%CCK8%' OR
    assay ILIKE '%celltiter%' OR
    assay ILIKE '%cell titer%' OR
    assay ILIKE '%MTS%' OR
    assay ILIKE '%REMA%' OR
    assay ILIKE '%WST%'
"""


def main():
    parser = argparse.ArgumentParser(description='Filter phenotypic assays from parquet files')
    parser.add_argument('input_file', help='Input parquet file')
    parser.add_argument('output_file', help='Output parquet file')
    parser.add_argument('--dry-run', action='store_true', help='Show statistics only; do not save')
    args = parser.parse_args()

    con = duckdb.connect()

    # Statistics
    total = con.execute(f"SELECT COUNT(*) FROM '{args.input_file}'").fetchone()[0]
    print(f'Total records: {total:,}')

    to_remove = con.execute(f"""
        SELECT COUNT(*) FROM '{args.input_file}'
        WHERE {FILTER_CONDITION}
    """).fetchone()[0]
    print(f'Will be removed: {to_remove:,}')
    print(f'Will remain: {total - to_remove:,}')

    # Examples of records to remove
    print('\n--- Example assays to remove (top 30): ---')
    result = con.execute(f"""
        SELECT assay, COUNT(*) as cnt FROM '{args.input_file}'
        WHERE {FILTER_CONDITION}
        GROUP BY assay
        ORDER BY cnt DESC
        LIMIT 30
    """).fetchall()
    for r in result:
        print(f'{r[1]:>6} | {r[0][:80] if r[0] else None}')

    if args.dry_run:
        print('\n[DRY RUN] File not saved.')
        return

    # Filter and save
    print('\nFiltering and saving...')
    con.execute(f"""
        COPY (
            SELECT * FROM '{args.input_file}'
            WHERE NOT ({FILTER_CONDITION})
        ) TO '{args.output_file}' (FORMAT PARQUET)
    """)

    # Verify result
    result_count = con.execute(f"SELECT COUNT(*) FROM '{args.output_file}'").fetchone()[0]
    print(f'Records in new file: {result_count:,}')

    # Verification
    check = con.execute(f"""
        SELECT COUNT(*) FROM '{args.output_file}'
        WHERE {FILTER_CONDITION}
    """).fetchone()[0]
    print(f'Records with forbidden assays: {check}')

    print(f'\nDone! File saved: {args.output_file}')


if __name__ == '__main__':
    main()
