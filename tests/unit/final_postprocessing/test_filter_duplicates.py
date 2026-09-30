"""Tests for DuckDB-based duplicate quadruplet filter."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from final_postprocessing.filter_duplicates import main as filter_duplicates_main


def _write_quad_fixture(path: Path) -> None:
    # One patent with a bad quad (4 identical rows) and one clean patent
    patents = ["US20090176883A1"] * 4 + ["US20100000001A1"] * 2
    aliases = ["Example 1"] * 4 + ["Example 2", "Example 3"]
    table = pa.table(
        {
            "patent_number": patents,
            "assay_id": ["A1"] * 6,
            "original_alias": aliases,
            "IC50 (nM)": ["10"] * 6,
            "Ki (nM)": [None] * 6,
            "Kd (nM)": [None] * 6,
            "EC50 (nM)": [None] * 6,
            "assay": ["binding"] * 6,
        }
    )
    pq.write_table(table, path)


def test_filter_duplicates_removes_bad_quads(tmp_path: Path):
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_quad_fixture(inp)

    filter_duplicates_main(str(inp), str(out))

    df = pq.read_table(out).to_pandas()
    # Bad quad (4 rows) removed; 2 clean rows remain
    assert len(df) == 2
    assert set(df["patent_number"]) == {"US20100000001A1"}
    assert "_act_type" not in df.columns
