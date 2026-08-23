"""Parquet batch helpers for BindingDB export."""

import os
from typing import Any, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .rows import binding_to_parquet_row
from .schema import OUTPUT_COLUMNS, PARQUET_SCHEMA


def merge_parquet_batches(batches_dir: str, output_file: str) -> int:
    """
    Merge Parquet batches into one file.

    Returns:
        Number of rows in the final file
    """
    batch_files = sorted([
        os.path.join(batches_dir, f)
        for f in os.listdir(batches_dir)
        if f.endswith(".parquet")
    ])

    if not batch_files:
        return 0

    tables = [pq.read_table(f) for f in batch_files]
    merged = pa.concat_tables(tables)
    pq.write_table(merged, output_file)

    return merged.num_rows


def write_binding_batch(
    batch_bindings: list[Dict[str, Any]],
    batch_file: str,
    patent_dict: Dict[str, int],
) -> tuple[int, int]:
    if batch_bindings:
        rows = []
        for binding in batch_bindings:
            row = binding_to_parquet_row(binding, patent_dict)
            if row:
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
            for field in PARQUET_SCHEMA:
                col = field.name
                if col in df.columns and field.type == pa.string():
                    df[col] = df[col].fillna("").astype(str)
            if "normalized_pub_number" in df.columns:
                df["normalized_pub_number"] = pd.to_numeric(
                    df["normalized_pub_number"], errors="coerce"
                ).fillna(0).astype("int64")

            table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
            pq.write_table(table, batch_file)

            batch_written = len(rows)
            batch_skipped = len(batch_bindings) - len(rows)
        else:
            batch_written = 0
            batch_skipped = len(batch_bindings)
            pq.write_table(
                pa.Table.from_pydict({col: [] for col in OUTPUT_COLUMNS}, schema=PARQUET_SCHEMA),
                batch_file,
            )
    else:
        batch_written = 0
        batch_skipped = 0
        pq.write_table(
            pa.Table.from_pydict({col: [] for col in OUTPUT_COLUMNS}, schema=PARQUET_SCHEMA),
            batch_file,
        )

    return batch_written, batch_skipped
