"""
Filter duplicate quadruplets (patent, assay_id, alias, activity_type) from a parquet file.

Strategy:
1. Remove ALL records from quadruplets with count >= 4
2. If a patent has 10+ such bad quadruplets, remove the entire patent

Uses DuckDB to avoid loading the full table into pandas.
"""
from __future__ import annotations

import argparse

import duckdb

QUAD_THRESHOLD = 4
PATENT_THRESHOLD = 10


def log(msg: str) -> None:
    print(msg, flush=True)


def main(input_path: str, output_path: str) -> None:
    con = duckdb.connect()

    log("Loading metadata via DuckDB...")
    total_rows, n_patents = con.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT patent_number)
        FROM '{input_path}'
        """
    ).fetchone()
    log(f"  Input: {total_rows:,} rows, {n_patents:,} patents")

    # Activity type + quadruplet counts + nuke patents in one SQL pipeline.
    # Priority matches pandas np.where chain: IC50 > Ki > Kd > EC50 > None
    log("Computing bad quadruplets...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE annotated AS
        SELECT
            *,
            CASE
                WHEN try_cast("IC50 (nM)" AS DOUBLE) IS NOT NULL THEN 'IC50'
                WHEN try_cast("Ki (nM)" AS DOUBLE) IS NOT NULL THEN 'Ki'
                WHEN try_cast("Kd (nM)" AS DOUBLE) IS NOT NULL THEN 'Kd'
                WHEN try_cast("EC50 (nM)" AS DOUBLE) IS NOT NULL THEN 'EC50'
                ELSE 'None'
            END AS _act_type
        FROM '{input_path}'
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE bad_quads AS
        SELECT
            patent_number,
            assay_id,
            original_alias,
            _act_type,
            COUNT(*) AS _cnt
        FROM annotated
        GROUP BY patent_number, assay_id, original_alias, _act_type
        HAVING COUNT(*) >= {QUAD_THRESHOLD}
        """
    )

    n_bad_quads, n_bad_quad_records = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(_cnt), 0) FROM bad_quads"
    ).fetchone()
    log(f"  Bad quads (count >= {QUAD_THRESHOLD}): {n_bad_quads:,}, records: {n_bad_quad_records:,}")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE nuke_patents AS
        SELECT patent_number
        FROM bad_quads
        GROUP BY patent_number
        HAVING COUNT(*) >= {PATENT_THRESHOLD}
        """
    )
    n_nuke = con.execute("SELECT COUNT(*) FROM nuke_patents").fetchone()[0]
    log(f"  Patents with {PATENT_THRESHOLD}+ bad quads (nuke): {n_nuke:,}")

    bad_quad_records = con.execute(
        """
        SELECT COUNT(*)
        FROM annotated a
        INNER JOIN bad_quads b
          ON a.patent_number IS NOT DISTINCT FROM b.patent_number
         AND a.assay_id IS NOT DISTINCT FROM b.assay_id
         AND a.original_alias IS NOT DISTINCT FROM b.original_alias
         AND a._act_type = b._act_type
        """
    ).fetchone()[0]

    nuke_records = con.execute(
        """
        SELECT COUNT(*)
        FROM annotated a
        WHERE a.patent_number IN (SELECT patent_number FROM nuke_patents)
        """
    ).fetchone()[0]

    # Union remove count
    n_removed = con.execute(
        """
        SELECT COUNT(*)
        FROM annotated a
        WHERE a.patent_number IN (SELECT patent_number FROM nuke_patents)
           OR EXISTS (
                SELECT 1 FROM bad_quads b
                WHERE a.patent_number IS NOT DISTINCT FROM b.patent_number
                  AND a.assay_id IS NOT DISTINCT FROM b.assay_id
                  AND a.original_alias IS NOT DISTINCT FROM b.original_alias
                  AND a._act_type = b._act_type
           )
        """
    ).fetchone()[0]

    log(f"  Bad quad records: {bad_quad_records:,}")
    log(f"  Nuke patent records: {nuke_records:,}")
    pct = (100.0 * n_removed / total_rows) if total_rows else 0.0
    log(f"  Total to remove (union): {n_removed:,} ({pct:.2f}%)")

    log(f"\nSaving to {output_path}...")
    # Drop helper column; keep original schema columns only
    con.execute(
        f"""
        COPY (
            SELECT * EXCLUDE (_act_type)
            FROM annotated a
            WHERE a.patent_number NOT IN (SELECT patent_number FROM nuke_patents)
              AND NOT EXISTS (
                    SELECT 1 FROM bad_quads b
                    WHERE a.patent_number IS NOT DISTINCT FROM b.patent_number
                      AND a.assay_id IS NOT DISTINCT FROM b.assay_id
                      AND a.original_alias IS NOT DISTINCT FROM b.original_alias
                      AND a._act_type = b._act_type
              )
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    out_rows, out_patents = con.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT patent_number)
        FROM '{output_path}'
        """
    ).fetchone()

    log("\nResult:")
    log(f"  Output: {out_rows:,} rows, {out_patents:,} patents")
    log(f"  Removed: {n_removed:,} records ({pct:.2f}%)")
    log(f"  Removed patents entirely: {n_nuke:,}")
    log("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Input parquet file.")
    parser.add_argument("--output", required=True, help="Output parquet file.")
    args = parser.parse_args()
    main(args.input, args.output)
