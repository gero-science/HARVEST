#!/usr/bin/env python3
"""
Add final structure columns to a HARVEST parquet table.

The script keeps the clean_smiles/clean_inchi_key columns untouched and
writes three production columns:

  final_smiles
  final_inchikey
  structure_choice_reason

final_smiles is the selected standardized SMILES. final_inchikey is always
computed from final_smiles.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

try:
    from final_postprocessing.add_clean_best import standardize_smiles_worker
except ImportError:  # pragma: no cover - allows running as a script by path
    from add_clean_best import standardize_smiles_worker

RDLogger.DisableLog("rdApp.*")


OPSIN_SMILES_COL = "Ligand SMILES"
CDX_SMILES_COL = "smiles_cdx"
FINAL_COLUMNS = ["final_smiles", "final_inchikey", "structure_choice_reason"]


@dataclass(frozen=True)
class FinalStructureSelection:
    final_smiles: str | None
    final_inchikey: str | None
    structure_choice_reason: str


def _nonempty(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip() != ""


def _clean_value(value: Any) -> str | None:
    return str(value).strip() if _nonempty(value) else None


def inchikey_from_smiles(smiles: str | None) -> str | None:
    """Return RDKit InChIKey for SMILES, or None if parsing/conversion fails."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def first_inchikey_block(inchikey: str | None) -> str:
    if not inchikey:
        return ""
    return str(inchikey).split("-")[0]


def mol_desc(smiles: str | None) -> dict[str, Any] | None:
    """Return the RDKit features used to classify OPSIN/CDX conflicts."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        frag_heavy_atoms = sorted((frag.GetNumHeavyAtoms() for frag in frags), reverse=True)
        return {
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "fragment_count": len(frags),
            "fragment_heavy_atoms": tuple(frag_heavy_atoms[:8]),
        }
    except Exception:
        return None


def classify_structure_pair(
    opsin_smiles: str | None,
    cdx_smiles: str | None,
    *,
    desc_cache: dict[str, dict[str, Any] | None] | None = None,
) -> str:
    """
    Classify the raw OPSIN/CDX structural disagreement.

    These classes mirror validation/cdx_vs_opsin_analysis.ipynb and the
    BDB-referee analysis used to choose the default source.
    """
    desc_cache = desc_cache if desc_cache is not None else {}

    def get_desc(smiles: str | None) -> dict[str, Any] | None:
        if not smiles:
            return None
        if smiles not in desc_cache:
            desc_cache[smiles] = mol_desc(smiles)
        return desc_cache[smiles]

    opsin_desc = get_desc(opsin_smiles)
    cdx_desc = get_desc(cdx_smiles)
    if not opsin_desc or not cdx_desc:
        return "unclassified"

    delta = opsin_desc["heavy_atoms"] - cdx_desc["heavy_atoms"]
    abs_delta = abs(delta)

    if opsin_desc["fragment_count"] > 1 and cdx_desc["fragment_count"] == 1:
        return "opsin_multifragment_cdx_single"
    if cdx_desc["fragment_count"] > 1 and opsin_desc["fragment_count"] == 1:
        return "cdx_multifragment_opsin_single"
    if abs_delta == 0:
        return "same_heavy_atoms_different_connectivity"
    if abs_delta <= 2:
        return "small_size_delta_1_2_atoms"
    if opsin_desc["heavy_atoms"] > cdx_desc["heavy_atoms"]:
        return "opsin_larger"
    return "cdx_larger"


def reason_for_conflict_class(conflict_class: str) -> tuple[str, str]:
    """
    Return (preferred_source, reason) for a classified OPSIN/CDX conflict.

    The defaults follow the BDB-referee decisive statistics:
    OPSIN is preferred for every class except same-size graph conflicts, where
    CDX was supported more often.
    """
    if conflict_class == "opsin_larger":
        return "opsin", "prefer_opsin_opsin_larger"
    if conflict_class == "cdx_multifragment_opsin_single":
        return "opsin", "prefer_opsin_cdx_multifragment"
    if conflict_class == "cdx_larger":
        return "opsin", "prefer_opsin_cdx_larger"
    if conflict_class == "opsin_multifragment_cdx_single":
        return "opsin", "prefer_opsin_opsin_multifragment"
    if conflict_class == "small_size_delta_1_2_atoms":
        return "opsin", "prefer_opsin_small_delta"
    if conflict_class == "same_heavy_atoms_different_connectivity":
        return "cdx", "prefer_cdx_same_size_graph"
    return "opsin", "prefer_opsin_unclassified"


def select_final_structure(
    opsin_smiles: Any,
    cdx_smiles: Any,
    *,
    opsin_clean_smiles: str | None = None,
    cdx_clean_smiles: str | None = None,
    opsin_clean_inchikey: str | None = None,
    cdx_clean_inchikey: str | None = None,
    desc_cache: dict[str, dict[str, Any] | None] | None = None,
) -> FinalStructureSelection:
    """Select final clean structure for one row."""
    opsin_raw = _clean_value(opsin_smiles)
    cdx_raw = _clean_value(cdx_smiles)

    if opsin_clean_smiles is None and opsin_raw:
        opsin_clean_smiles = standardize_smiles_worker(opsin_raw)
    if cdx_clean_smiles is None and cdx_raw:
        cdx_clean_smiles = standardize_smiles_worker(cdx_raw)

    opsin_final_candidate = opsin_clean_smiles or opsin_raw
    cdx_final_candidate = cdx_clean_smiles or cdx_raw

    if opsin_clean_inchikey is None:
        opsin_clean_inchikey = inchikey_from_smiles(opsin_clean_smiles)
    if cdx_clean_inchikey is None:
        cdx_clean_inchikey = inchikey_from_smiles(cdx_clean_smiles)

    if opsin_raw and not cdx_raw:
        final_smiles = opsin_final_candidate
        return FinalStructureSelection(
            final_smiles=final_smiles,
            final_inchikey=inchikey_from_smiles(final_smiles),
            structure_choice_reason="only_opsin",
        )

    if cdx_raw and not opsin_raw:
        final_smiles = cdx_final_candidate
        return FinalStructureSelection(
            final_smiles=final_smiles,
            final_inchikey=inchikey_from_smiles(final_smiles),
            structure_choice_reason="only_cdx",
        )

    if not opsin_raw and not cdx_raw:
        return FinalStructureSelection(None, None, "prefer_opsin_unclassified")

    opsin_block = first_inchikey_block(opsin_clean_inchikey)
    cdx_block = first_inchikey_block(cdx_clean_inchikey)
    if opsin_block and cdx_block and opsin_block == cdx_block:
        final_smiles = opsin_final_candidate
        return FinalStructureSelection(
            final_smiles=final_smiles,
            final_inchikey=inchikey_from_smiles(final_smiles),
            structure_choice_reason="sources_match",
        )

    conflict_class = classify_structure_pair(opsin_raw, cdx_raw, desc_cache=desc_cache)
    source, reason = reason_for_conflict_class(conflict_class)
    final_smiles = opsin_final_candidate if source == "opsin" else cdx_final_candidate
    return FinalStructureSelection(
        final_smiles=final_smiles,
        final_inchikey=inchikey_from_smiles(final_smiles),
        structure_choice_reason=reason,
    )


def _build_clean_maps(smiles_values: list[str]) -> tuple[dict[str, str | None], dict[str, str | None]]:
    unique_smiles = sorted(set(smiles_values))
    clean_map = {smiles: standardize_smiles_worker(smiles) for smiles in unique_smiles}
    unique_clean = sorted({clean for clean in clean_map.values() if clean})
    inchikey_map = {smiles: inchikey_from_smiles(smiles) for smiles in unique_clean}
    return clean_map, inchikey_map


def process_chunk(chunk_df: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """Add final structure columns to one DataFrame chunk."""
    df = chunk_df.copy()
    for col in [OPSIN_SMILES_COL, CDX_SMILES_COL]:
        if col not in df.columns:
            df[col] = None

    smiles_values: list[str] = []
    for col in [OPSIN_SMILES_COL, CDX_SMILES_COL]:
        smiles_values.extend(_clean_value(value) for value in df[col].tolist() if _nonempty(value))

    clean_map, inchikey_map = _build_clean_maps(smiles_values)
    desc_cache: dict[str, dict[str, Any] | None] = {}

    final_smiles: list[str | None] = []
    final_inchikey: list[str | None] = []
    reasons: list[str] = []

    for row in df[[OPSIN_SMILES_COL, CDX_SMILES_COL]].itertuples(index=False, name=None):
        opsin_raw = _clean_value(row[0])
        cdx_raw = _clean_value(row[1])
        opsin_clean = clean_map.get(opsin_raw) if opsin_raw else None
        cdx_clean = clean_map.get(cdx_raw) if cdx_raw else None
        selection = select_final_structure(
            opsin_raw,
            cdx_raw,
            opsin_clean_smiles=opsin_clean,
            cdx_clean_smiles=cdx_clean,
            opsin_clean_inchikey=inchikey_map.get(opsin_clean),
            cdx_clean_inchikey=inchikey_map.get(cdx_clean),
            desc_cache=desc_cache,
        )
        final_smiles.append(selection.final_smiles)
        final_inchikey.append(selection.final_inchikey)
        reasons.append(selection.structure_choice_reason)

    df["final_smiles"] = final_smiles
    df["final_inchikey"] = final_inchikey
    df["structure_choice_reason"] = reasons
    return df, Counter(reasons)


def _process_chunk_with_index(args: tuple[int, pd.DataFrame]) -> tuple[int, pd.DataFrame, Counter]:
    chunk_idx, chunk_df = args
    processed, reason_counts = process_chunk(chunk_df)
    return chunk_idx, processed, reason_counts


def _output_schema(input_schema: pa.Schema) -> pa.Schema:
    existing = set(FINAL_COLUMNS)
    fields = [field for field in input_schema if field.name not in existing]
    fields.extend(pa.field(name, pa.string()) for name in FINAL_COLUMNS)
    return pa.schema(fields)


def _write_chunk(writer: pq.ParquetWriter | None, output_path: Path, schema: pa.Schema, df: pd.DataFrame) -> pq.ParquetWriter:
    for col in schema.names:
        if col not in df.columns:
            df[col] = None
    df = df[schema.names]
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, schema)
    writer.write_table(table)
    return writer


def add_final_structure_columns(
    input_path: Path,
    output_path: Path,
    *,
    workers: int = 1,
    chunk_size: int = 50_000,
) -> Counter:
    """Read input parquet chunk-by-chunk and write output parquet with final columns."""
    parquet_file = pq.ParquetFile(input_path)
    schema = _output_schema(parquet_file.schema_arrow)
    total_rows = parquet_file.metadata.num_rows
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_counts: Counter = Counter()

    if workers <= 1:
        try:
            for batch in tqdm(
                parquet_file.iter_batches(batch_size=chunk_size),
                total=total_chunks,
                desc="Adding final structure",
            ):
                df, counts = process_chunk(batch.to_pandas())
                total_counts.update(counts)
                writer = _write_chunk(writer, output_path, schema, df)
        finally:
            if writer is not None:
                writer.close()
        return total_counts

    pending: dict[int, tuple[pd.DataFrame, Counter]] = {}
    futures = set()
    next_to_write = 0

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pbar = tqdm(total=total_chunks, desc="Adding final structure")
            for chunk_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
                future = executor.submit(_process_chunk_with_index, (chunk_idx, batch.to_pandas()))
                futures.add(future)

                if len(futures) >= workers * 2:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for completed in done:
                        idx, df, counts = completed.result()
                        pending[idx] = (df, counts)
                    while next_to_write in pending:
                        df, counts = pending.pop(next_to_write)
                        total_counts.update(counts)
                        writer = _write_chunk(writer, output_path, schema, df)
                        next_to_write += 1
                        pbar.update(1)

            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for completed in done:
                    idx, df, counts = completed.result()
                    pending[idx] = (df, counts)
                while next_to_write in pending:
                    df, counts = pending.pop(next_to_write)
                    total_counts.update(counts)
                    writer = _write_chunk(writer, output_path, schema, df)
                    next_to_write += 1
                    pbar.update(1)
            pbar.close()
    finally:
        if writer is not None:
            writer.close()

    return total_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add final_smiles/final_inchikey/structure_choice_reason columns to a parquet file."
    )
    parser.add_argument("input_file", type=Path, help="Input parquet file")
    parser.add_argument("output_file", type=Path, help="Output parquet file")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Rows per chunk")
    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(args.input_file)

    print("=" * 72)
    print("Adding final structure columns")
    print(f"Input:      {args.input_file}")
    print(f"Output:     {args.output_file}")
    print(f"Workers:    {args.workers}")
    print(f"Chunk size: {args.chunk_size:,}")
    print("=" * 72)

    counts = add_final_structure_columns(
        args.input_file,
        args.output_file,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    print("\nstructure_choice_reason counts:")
    total = sum(counts.values())
    for reason, count in counts.most_common():
        pct = 100 * count / total if total else 0
        print(f"  {reason:36s} {count:12,} {pct:6.2f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
