"""Tests for the optional final_postprocessing one-command runner."""

from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from final_postprocessing.add_clean_best import run_add_clean_best, standardize_smiles_worker
from final_postprocessing.run import (
    FinalPostprocessingConfig,
    build_arg_parser,
    run_final_postprocessing,
)


def _tiny_bindingdb_parquet(path: Path, smiles: list[str] | None = None) -> None:
    smiles = smiles or ["CCO", "CCCO"]
    n = len(smiles)
    table = pa.table(
        {
            "Ligand SMILES": smiles,
            "Ligand InChI Key": [
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "BDERNNFJNOPAEC-UHFFFAOYSA-N",
            ][:n]
            + ["X" * 27] * max(0, n - 2),
            "smiles_source": ["py2opsin"] * n,
            "smiles_cdx": [None] * n,
            "inchi_key_cdx": [None] * n,
            "patent_number": ["US20090176883A1"] * n,
            "assay_id": ["ASSAY_001"] * n,
            "original_alias": [f"Example {i}" for i in range(1, n + 1)],
            "assay": ["binding assay"] * n,
            "IC50 (nM)": [str(10 * i) for i in range(1, n + 1)],
            "Ki (nM)": [None] * n,
            "Kd (nM)": [None] * n,
            "EC50 (nM)": [None] * n,
            "compound": [f"Example {i}" for i in range(1, n + 1)],
            "protein_target_name": ["TRPA1"] * n,
        }
    )
    pq.write_table(table, path)


def test_build_arg_parser_help():
    parser = build_arg_parser()
    help_text = parser.format_help()
    assert "enrichment-only" in help_text
    assert "add_final_structure" in help_text
    assert "cache-path" in help_text
    assert "no-cache" in help_text
    assert "skip-add-clean-best" in help_text


def test_skip_add_clean_best_runs_filters_without_fragments(tmp_path: Path):
    """Raw BindingDB parquet has no clean_smiles → fragments auto-skipped."""
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _tiny_bindingdb_parquet(inp)

    run_final_postprocessing(
        inp,
        out,
        FinalPostprocessingConfig(skip_add_clean_best=True, workers=1),
    )

    df = pq.read_table(out).to_pandas()
    assert len(df) == 2
    assert "clean_smiles" not in df.columns
    assert "year" not in df.columns


def test_enrichment_only_writes_clean_columns(tmp_path: Path):
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _tiny_bindingdb_parquet(inp)

    run_final_postprocessing(
        inp,
        out,
        FinalPostprocessingConfig(
            enrichment_only=True, workers=1, chunk_size=10, no_cache=True
        ),
    )

    df = pq.read_table(out).to_pandas()
    assert len(df) == 2
    for col in ("year", "clean_smiles", "clean_inchi_key"):
        assert col in df.columns
        assert df[col].notna().all()
    assert set(df["year"].astype(str)) == {"2009"}


def test_clean_smiles_cache_hit_skips_restanardize(tmp_path: Path):
    """Second run with the same cache must not call standardize for cached SMILES."""
    inp = tmp_path / "in.parquet"
    out1 = tmp_path / "out1.parquet"
    out2 = tmp_path / "out2.parquet"
    cache = tmp_path / "clean_smiles_cache.sqlite"
    # Duplicate SMILES across rows so unique set is smaller than row count
    _tiny_bindingdb_parquet(inp, smiles=["CCO", "CCO", "CCCO"])

    run_add_clean_best(
        inp, out1, workers=1, chunk_size=10, cache_path=cache, no_cache=False
    )
    assert cache.exists()

    call_count = {"n": 0}
    real = standardize_smiles_worker

    def counting_worker(smiles):
        call_count["n"] += 1
        return real(smiles)

    with mock.patch(
        "final_postprocessing.add_clean_best.standardize_smiles_worker",
        side_effect=counting_worker,
    ):
        run_add_clean_best(
            inp, out2, workers=1, chunk_size=10, cache_path=cache, no_cache=False
        )

    assert call_count["n"] == 0
    df1 = pq.read_table(out1).to_pandas()
    df2 = pq.read_table(out2).to_pandas()
    assert df1["clean_smiles"].tolist() == df2["clean_smiles"].tolist()
