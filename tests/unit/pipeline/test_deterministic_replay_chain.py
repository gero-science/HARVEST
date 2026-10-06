"""Deterministic extract→export chain on synthetic patent folders."""

import asyncio
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bindingdb_export.processing import run_bindingdb_processing
from llm.extractor import BioactivityExtractor
from llm.stage_formats import StageFormats
from patent_processor.data_structures import ChemistryNode, PatentDocument
from pipeline.alias_resolution import ACCEPTED_EXTRACTOR_SOURCE, resolve_aliases_for_patent


def _make_agent() -> BioactivityExtractor:
    agent = BioactivityExtractor.__new__(BioactivityExtractor)
    agent.logger = logging.getLogger("tests.pipeline.replay")
    return agent


def _make_document(*chemistry_nodes: ChemistryNode) -> PatentDocument:
    return PatentDocument(
        file_path="memory.zip",
        patent_id="USUNITTESTA1",
        chemistry_nodes=list(chemistry_nodes),
    )


def _assay(assay_id: str = "A1") -> dict:
    return {
        "assay_id": assay_id,
        "protein_target_name": "EGFR",
        "is_complex": "N",
        "protein_modification": "wild type",
        "assay_description": "enzyme inhibition assay",
        "reasoning": "p0001",
        "assay": "biochemical",
        "organism": "human",
        "extreme_conditions": "N",
    }


def _bioactivity(compound: str = "Example 1", assay_id: str = "A1") -> dict:
    return {
        "compound": compound,
        "assay_id": assay_id,
        "binding_metric": "IC50",
        "value": "12",
        "unit": "nM",
        "reasoning": "p0002",
    }


def _compound(alias: str = "Example 1", chemical_id: str = "CHEM-US-00001") -> dict:
    return {
        "compound": alias,
        "compound_IUPAC_identifier": "ethanol",
        "reasoning": "p0003",
        "chemical_id": chemical_id,
    }


def test_stage_format_headers_are_stable():
    assert StageFormats.STAGE1_HEADER.split("\t")[0] == "assay_id"
    assert "binding_metric" in StageFormats.STAGE2_HEADER
    assert "chemical_id" in StageFormats.STAGE3_HEADER


def test_merge_enrich_bypass_matches_export_contract():
    """Replay merge → SMILES enrich → Agent 2 bypass (former stage-to-parquet path)."""
    node = ChemistryNode(
        chemistry_id="CHEM-US-00001",
        chem_num="00001",
        smiles="CCO",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    )
    document = _make_document(node)
    agent = _make_agent()

    merged = agent._merge_with_assay_id(
        [_assay()],
        [_bioactivity()],
        [_compound()],
        document,
    )
    valid_rows = [row for row in merged if agent._validate_item(row)]
    assert len(valid_rows) == 1

    for row in valid_rows:
        row["molecule_name"] = row.get("compound")

    agent._enrich_with_smiles(valid_rows, document)
    assert valid_rows[0]["molecule_smiles"]  # canonical SMILES from RDKit (e.g. C(C)O for ethanol)

    resolved, unresolved, _, usage = asyncio.run(
        resolve_aliases_for_patent(valid_rows),
    )
    assert unresolved == []
    assert usage["request_count"] == 0
    assert len(resolved) == 1
    assert resolved[0]["agent2_resolution_source"] == ACCEPTED_EXTRACTOR_SOURCE
    assert resolved[0]["original_alias"] == "Example 1"


def test_synthetic_patent_directory_exports_parquet(tmp_path):
    """End-to-end BindingDB export on a minimal in-repo patent folder."""
    patent_id = "USUNITTESTA1"
    input_root = tmp_path / "inputs"
    patent_dir = input_root / patent_id
    patent_dir.mkdir(parents=True)

    resolved = [{
        "compound": "Example 1",
        "protein_target_name": "EGFR",
        "is_complex": "N",
        "binding_metric": "IC50",
        "value": "12",
        "unit": "nM",
        "molecule_smiles": "CCO",
        "molecule_inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "smiles_source": "table_chemistry_tag",
        "agent2_resolution_source": ACCEPTED_EXTRACTOR_SOURCE,
        "organism": "human",
        "assay": "biochemical",
    }]
    (patent_dir / f"{patent_id}_resolved.json").write_text(
        json.dumps(resolved, ensure_ascii=False),
        encoding="utf-8",
    )
    (patent_dir / "single_proteins.json").write_text(
        json.dumps({
            "proteins": [{
                "protein_name": "EGFR",
                "organism": "human",
                "sequence": "MTEYKLVVVGAGGVGKSALTI",
                "gene": "EGFR",
                "uniprot_id": "P00533",
                "accession": "P00533",
            }],
        }),
        encoding="utf-8",
    )
    (patent_dir / "protein_complexes.json").write_text('{"complexes": []}', encoding="utf-8")
    (patent_dir / "cdx_results.json").write_text(
        json.dumps({
            "patent_id": patent_id,
            "compounds": {
                "00001": {
                    "smiles": "CCO",
                    "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                },
            },
        }),
        encoding="utf-8",
    )

    mapping = Path("curated_data/patent_mapping.csv")
    if not mapping.exists():
        pytest.skip("curated_data/patent_mapping.csv not available")

    output_file = tmp_path / "main_res.parquet"
    success, message, _elapsed = run_bindingdb_processing(
        input_dir=str(input_root),
        output_file=str(output_file),
        workers=1,
        cache_dir=str(tmp_path / "cache"),
        patent_dict_file=str(mapping),
        batch_size=1,
        force=True,
        use_opsin=False,
    )

    assert success is True
    assert output_file.exists()
    table = pq.read_table(output_file)
    assert table.num_rows >= 1
    df = table.to_pandas()
    assert patent_id in set(df["patent_number"])
    assert (df["IC50 (nM)"].astype(str).str.strip() != "").any()
