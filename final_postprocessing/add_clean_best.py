#!/usr/bin/env python3
"""
Add year, clean_smiles, and clean_inchi_key columns to a HARVEST parquet.

The input parquet already contains the best SMILES in 'Ligand SMILES'
(source selection is done at export time).  This step:

1. Extracts year from patent_number.
2. Standardises 'Ligand SMILES' → clean_smiles (tautomer-canonical, uncharged,
   largest fragment) with an optional SQLite cache.
3. Computes InChI Keys from clean_smiles → clean_inchi_key.

Usage:
    python final_postprocessing/add_clean_best.py input.parquet output.parquet [--workers N]
    python -m final_postprocessing input.parquet output.parquet
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def extract_year_from_patent(patent_number: str) -> str | None:
    """Extract the year from a patent number (first 4 digits after US)."""
    if not patent_number or pd.isna(patent_number):
        return None
    match = re.search(r"US(\d{4})", str(patent_number))
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
#  SMILES standardisation
# ---------------------------------------------------------------------------

def standardize_smiles_worker(smiles):
    """Worker function for standardizing a single SMILES."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if not smiles or pd.isna(smiles):
        return None

    try:
        smiles = smiles.replace("[N]", "[NH]").replace("[O]", "[OH]")
        smiles = smiles.replace("[C-]#[N+]", "N#C")
        smiles = smiles.replace("[2C]", "C")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger()
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        try:
            te = rdMolStandardize.TautomerEnumerator()
            taut_mol = te.Canonicalize(uncharged_parent_clean_mol)
        except RuntimeError:
            taut_mol = None

        if taut_mol is None:
            return None

        standardized_smiles = Chem.MolToSmiles(taut_mol)

        mol_check = Chem.MolFromSmiles(standardized_smiles)
        if mol_check is None:
            return None

        return standardized_smiles
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  SQLite cache helpers
# ---------------------------------------------------------------------------

def _open_cache(cache_path: Path | None) -> sqlite3.Connection | None:
    if cache_path is None:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS clean_smiles ("
        "smiles TEXT PRIMARY KEY, "
        "clean TEXT"
        ")"
    )
    conn.commit()
    return conn


def _cache_lookup(conn: sqlite3.Connection, smiles_list: list[str]) -> dict[str, str | None]:
    """Return {smiles: clean_or_None} for keys present in the cache."""
    found: dict[str, str | None] = {}
    batch_size = 900
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cur = conn.execute(
            f"SELECT smiles, clean FROM clean_smiles WHERE smiles IN ({placeholders})",
            batch,
        )
        for smiles, clean in cur.fetchall():
            found[smiles] = clean
    return found


def _cache_store(conn: sqlite3.Connection, mapping: dict[str, str | None]) -> None:
    if not mapping:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO clean_smiles (smiles, clean) VALUES (?, ?)",
        [(s, c) for s, c in mapping.items()],
    )
    conn.commit()


def _standardize_unique_smiles(
    unique_smiles: list[str],
    *,
    workers: int,
    cache_conn: sqlite3.Connection | None,
    standardize_batch_size: int = 500,
) -> dict[str, str | None]:
    """Standardize unique SMILES with cache hits + process pool for misses."""
    result_map: dict[str, str | None] = {}
    if not unique_smiles:
        return result_map

    missing = list(unique_smiles)
    cache_hits = 0
    if cache_conn is not None:
        cached = _cache_lookup(cache_conn, unique_smiles)
        cache_hits = len(cached)
        result_map.update(cached)
        missing = [s for s in unique_smiles if s not in cached]

    print(
        f"  Unique SMILES: {len(unique_smiles):,} | "
        f"cache hits: {cache_hits:,} | to standardize: {len(missing):,}"
    )

    if not missing:
        return result_map

    new_results: dict[str, str | None] = {}
    t0 = time.perf_counter()
    pbar = tqdm(total=len(missing), desc="Standardizing SMILES")
    if workers <= 1:
        for smiles in missing:
            new_results[smiles] = standardize_smiles_worker(smiles)
            pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for i in range(0, len(missing), standardize_batch_size):
                batch = missing[i : i + standardize_batch_size]
                futures = {
                    executor.submit(standardize_smiles_worker, s): s for s in batch
                }
                for future in as_completed(futures):
                    smiles = futures[future]
                    try:
                        new_results[smiles] = future.result()
                    except Exception:
                        new_results[smiles] = None
                    pbar.update(1)
    pbar.close()

    elapsed = time.perf_counter() - t0
    print(f"  Standardized {len(missing):,} SMILES in {elapsed:.1f}s ({workers} workers)")

    result_map.update(new_results)
    if cache_conn is not None:
        _cache_store(cache_conn, new_results)
        print(f"  Cache updated with {len(new_results):,} entries")

    return result_map


# ---------------------------------------------------------------------------
#  InChI Key computation
# ---------------------------------------------------------------------------

def _compute_inchi_key(smiles: str) -> str | None:
    """Compute InChI Key from a SMILES string."""
    from rdkit import Chem

    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Parquet I/O helpers
# ---------------------------------------------------------------------------

def _build_output_schema(input_schema: pa.Schema) -> pa.Schema:
    new_fields = [
        pa.field("year", pa.string()),
        pa.field("clean_smiles", pa.string()),
        pa.field("clean_inchi_key", pa.string()),
    ]
    existing = set(input_schema.names)
    output_fields = list(input_schema)
    for field in new_fields:
        if field.name not in existing:
            output_fields.append(field)
    return pa.schema(output_fields)


def _write_enriched_chunk(
    result_df: pd.DataFrame,
    output_schema: pa.Schema,
    writer: pq.ParquetWriter | None,
    output_path: Path,
) -> pq.ParquetWriter:
    for col in output_schema.names:
        if col not in result_df.columns:
            result_df[col] = None
    result_df = result_df[[col for col in output_schema.names]]
    result_table = pa.Table.from_pandas(
        result_df, schema=output_schema, preserve_index=False
    )
    if writer is None:
        writer = pq.ParquetWriter(output_path, output_schema)
    writer.write_table(result_table)
    return writer


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def run_add_clean_best(
    input_path: str | Path,
    output_path: str | Path,
    *,
    workers: int | None = None,
    chunk_size: int = 50_000,
    cache_path: str | Path | None = None,
    no_cache: bool = False,
) -> Path:
    """Add year / clean_smiles / clean_inchi_key to a parquet file.

    Standardises 'Ligand SMILES' directly (source selection is done at
    export time, so the column already contains the best SMILES).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    n_workers = workers if workers is not None else default_workers()
    resolved_cache: Path | None = None
    if not no_cache:
        if cache_path is not None:
            resolved_cache = Path(cache_path)
        else:
            resolved_cache = output_path.parent / "clean_smiles_cache.sqlite"

    print("=" * 60)
    print("Adding columns to parquet file")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Workers: {n_workers}, chunk size: {chunk_size}")
    print(f"Cache: {resolved_cache if resolved_cache else '(disabled)'}")
    print("=" * 60)

    pf_input = pq.ParquetFile(input_path)
    total_rows = pf_input.metadata.num_rows
    input_schema = pf_input.schema_arrow
    output_schema = _build_output_schema(input_schema)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    print(f"\nInput file: {total_rows:,} rows ({total_chunks} chunks)")

    # --- Collect unique Ligand SMILES ---
    print("\nScanning for unique SMILES...")
    unique_smiles: set[str] = set()
    t0 = time.perf_counter()
    for batch in tqdm(pf_input.iter_batches(batch_size=chunk_size, columns=["Ligand SMILES"]),
                      total=total_chunks, desc="Scanning"):
        col = batch.to_pandas()["Ligand SMILES"].dropna()
        col = col[col != ""]
        unique_smiles.update(col.astype(str).tolist())
    print(f"  Found {len(unique_smiles):,} unique SMILES in {time.perf_counter() - t0:.1f}s")

    # --- Standardize unique SMILES ---
    print("\nStandardizing unique SMILES...")
    cache_conn = _open_cache(resolved_cache)
    try:
        result_map = _standardize_unique_smiles(
            sorted(unique_smiles),
            workers=n_workers,
            cache_conn=cache_conn,
        )
    finally:
        if cache_conn is not None:
            cache_conn.close()

    # --- Compute clean_inchi_key for all unique clean_smiles ---
    unique_clean = {v for v in result_map.values() if v}
    print(f"\nComputing InChI Keys for {len(unique_clean):,} unique clean_smiles...")
    t0_ik = time.perf_counter()
    inchi_key_map: dict[str, str | None] = {}
    if unique_clean:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            clean_list = sorted(unique_clean)
            futures = {executor.submit(_compute_inchi_key, s): s for s in clean_list}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="InChI Keys"):
                smi = futures[fut]
                try:
                    inchi_key_map[smi] = fut.result()
                except Exception:
                    inchi_key_map[smi] = None
    ok = sum(1 for v in inchi_key_map.values() if v)
    print(f"  Computed {ok:,}/{len(unique_clean):,} InChI Keys in {time.perf_counter() - t0_ik:.1f}s")

    # --- Write output: add year + clean_smiles + clean_inchi_key ---
    print("\nWriting enriched parquet...")
    writer: pq.ParquetWriter | None = None
    total_processed = 0

    pf_input = pq.ParquetFile(input_path)
    for batch in tqdm(pf_input.iter_batches(batch_size=chunk_size),
                      total=total_chunks, desc="Writing chunks"):
        df = batch.to_pandas()

        # Year
        df["year"] = df["patent_number"].apply(extract_year_from_patent)

        # clean_smiles + clean_inchi_key from Ligand SMILES
        raw = df["Ligand SMILES"].fillna("")
        valid_mask = raw != ""
        df["clean_smiles"] = None
        df["clean_inchi_key"] = None
        if valid_mask.any():
            smiles_series = raw[valid_mask].astype(str)
            clean_vals = [result_map.get(s) for s in smiles_series]
            df.loc[valid_mask, "clean_smiles"] = clean_vals
            df.loc[valid_mask, "clean_inchi_key"] = [
                inchi_key_map.get(c) if c else None for c in clean_vals
            ]

        writer = _write_enriched_chunk(df, output_schema, writer, output_path)
        total_processed += len(df)

    if writer is not None:
        writer.close()

    print("\nVerifying result...")
    pf_result = pq.ParquetFile(output_path)
    print(f"Output file: {output_path}")
    print(f"Total rows: {pf_result.metadata.num_rows:,}")
    print(f"Row groups: {pf_result.metadata.num_row_groups}")
    print(f"Columns: {len(pf_result.schema_arrow.names)}")
    print("\nNew columns:")
    for col in ["year", "clean_smiles", "clean_inchi_key"]:
        print(f"  - {col}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Add year, clean_smiles, and clean_inchi_key columns"
    )
    parser.add_argument("input_file", help="Input parquet file")
    parser.add_argument("output_file", help="Output parquet file")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of workers (default: cpu_count-2 = {default_workers()})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50000, help="Chunk size (default: 50000)"
    )
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
    args = parser.parse_args()
    run_add_clean_best(
        args.input_file,
        args.output_file,
        workers=args.workers,
        chunk_size=args.chunk_size,
        cache_path=args.cache_path,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
