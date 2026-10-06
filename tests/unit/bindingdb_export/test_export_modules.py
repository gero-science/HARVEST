"""Unit tests for bindingdb_export modules."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from enrich_data.ligand_info_extractor import LigandInfoExtractor
from bindingdb_export import artifacts, batches, enrichment, organisms, processing, proteins, rows
from bindingdb_export.schema import OUTPUT_COLUMNS


def _parquet_ready_row(**overrides):
    row = {
        "Ligand SMILES": "CCO",
        "Ligand InChI Key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "smiles_source": "py2opsin",
        "Sequence": "MTEYK",
        "binding_metric": "IC50",
        "value": 12.0,
        "patent_number": "USUNITTESTA1",
        "protein_target_name": "EGFR",
        "compound": "Example 1",
    }
    row.update(overrides)
    return row


class _DictCache(dict):
    cache_path = "unit-test-cache"

    def close(self):
        pass


class _FakeLigandExtractor:
    def __init__(self):
        self.smiles_requests = []
        self.extract_requests = []
        self.mw_requests = []

    def get_inchi_key_by_smiles(self, smiles):
        self.smiles_requests.append(smiles)
        return {
            "CCO": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "NCC": "NCC-INCHIKEY",
        }.get(smiles)

    def extract(self, molecule_name):
        self.extract_requests.append(molecule_name)
        return "EXTRACTED-INCHIKEY", "EXTRACTED-SMILES"

    def get_molecular_weight(self, smiles):
        self.mw_requests.append(smiles)
        return 500.0


def test_load_resolved_bindings_reads_legacy_and_resolved_json(tmp_path):
    patent_dir = tmp_path / "USUNITTESTA1"
    nested = patent_dir / "nested"
    nested.mkdir(parents=True)

    (patent_dir / "USUNITTESTA1_resolved.json").write_text(
        json.dumps([{"compound": "A"}]),
        encoding="utf-8",
    )
    (nested / "03_final_output.json").write_text(
        json.dumps([{"compound": "B"}]),
        encoding="utf-8",
    )
    (patent_dir / "bad_resolved.json").write_text("{bad json", encoding="utf-8")

    bindings = artifacts.load_resolved_bindings(
        str(patent_dir),
        ["03_final_output.json", "_resolved.json"],
    )

    assert [row["compound"] for row in bindings] == ["A", "B"]
    assert {row["patent_number"] for row in bindings} == {"USUNITTESTA1"}


def test_get_patent_directories_finds_nested_resolved_dirs(tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "A_resolved.json").write_text("[]", encoding="utf-8")
    (tmp_path / "B").mkdir()
    nested = tmp_path / "nested" / "C"
    nested.mkdir(parents=True)
    (nested / "C_resolved.json").write_text("[]", encoding="utf-8")

    found = artifacts.get_patent_directories(str(tmp_path))

    assert set(found) == {str(tmp_path / "A"), str(nested)}


def test_load_resolved_proteins_merges_single_and_complex_entries(tmp_path):
    (tmp_path / "single_proteins.json").write_text(
        json.dumps(
            {
                "proteins": [
                    {
                        "protein_name": "Shared target",
                        "organism": "human",
                        "sequence": "SINGLESEQ",
                        "accession": "P00001",
                        "uniprot_id": "SINGLE_HUMAN",
                        "gene": "SINGLE",
                        "organism_scientific": "Homo sapiens",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "protein_complexes.json").write_text(
        json.dumps(
            {
                "complexes": [
                    {
                        "complex_name": "Shared target",
                        "organism": "mouse",
                        "species": "Mus musculus",
                        "proteins": [
                            {
                                "sequence": "COMPLEXSEQ",
                                "accession": "Q00001",
                                "uniprot_id": "COMPLEX_MOUSE",
                                "gene": "COMPLEX",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = proteins.load_resolved_proteins(str(tmp_path))

    assert resolved["Shared target"]["human"]["sequence"] == "SINGLESEQ"
    assert resolved["Shared target"]["mouse"]["sequence"] == "COMPLEXSEQ"


def test_normalize_accepted_bindings_filters_and_drops_bad_rows():
    accepted = {
        "agent2_resolution_source": "skipped_enriched_by_agent1",
        "binding_metric": "IC50",
        "value": "12",
        "unit": "nM",
    }
    rejected_source = {
        "agent2_resolution_source": "llm_unverified",
        "binding_metric": "IC50",
        "value": "99",
        "unit": "nM",
    }
    bad_value = {
        "agent2_resolution_source": "skipped_enriched_by_agent1",
        "binding_metric": "IC50",
        "value": "not-a-number",
        "unit": "nM",
    }

    normalized = rows.normalize_accepted_bindings([accepted, rejected_source, bad_value])

    assert normalized == [accepted]
    assert accepted["binding_metric"] == "IC50"
    assert accepted["value"] == 12.0
    assert accepted["unit"] == "nM"
    assert accepted["relation"] == "="


def test_write_binding_batch_writes_non_empty_parquet(tmp_path):
    batch_file = tmp_path / "batch_0000.parquet"

    written, skipped = batches.write_binding_batch(
        [_parquet_ready_row()],
        str(batch_file),
        {"USUNITTESTA1": 123},
    )

    table = pq.read_table(batch_file)
    assert written == 1
    assert skipped == 0
    assert table.num_rows == 1
    assert table.schema.names == OUTPUT_COLUMNS
    assert table.column("normalized_pub_number").to_pylist() == [123]


def test_write_binding_batch_writes_empty_marker_when_all_rows_filtered(tmp_path):
    batch_file = tmp_path / "batch_0000.parquet"

    written, skipped = batches.write_binding_batch(
        [_parquet_ready_row(**{"Ligand InChI Key": "", "inchi_key_cdx": ""})],
        str(batch_file),
        {},
    )

    table = pq.read_table(batch_file)
    assert (written, skipped) == (0, 1)
    assert table.num_rows == 0
    assert table.schema.names == OUTPUT_COLUMNS


def test_write_binding_batch_writes_empty_marker_for_empty_input(tmp_path):
    batch_file = tmp_path / "batch_0000.parquet"

    written, skipped = batches.write_binding_batch([], str(batch_file), {})

    table = pq.read_table(batch_file)
    assert (written, skipped) == (0, 0)
    assert table.num_rows == 0
    assert table.schema.names == OUTPUT_COLUMNS


def test_merge_parquet_batches_returns_merged_row_count(tmp_path):
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    batches.write_binding_batch(
        [_parquet_ready_row(compound="A")],
        str(batch_dir / "batch_0000.parquet"),
        {},
    )
    batches.write_binding_batch(
        [_parquet_ready_row(compound="B")],
        str(batch_dir / "batch_0001.parquet"),
        {},
    )

    merged_count = batches.merge_parquet_batches(str(batch_dir), str(tmp_path / "out.parquet"))

    assert merged_count == 2
    assert pq.read_table(tmp_path / "out.parquet").num_rows == 2


def test_default_cache_dir_is_project_root_cache():
    cache_dir = Path(processing._default_cache_dir())

    assert cache_dir.name == "cache"
    # Located directly under the project root (sibling of bindingdb_export)
    assert (cache_dir.parent / "bindingdb_export").exists()
    assert "bindingdb_export" not in cache_dir.parts


def test_bindingdb_export_helpers_remain_importable():
    from bindingdb_export.cache import ThreadSafeCache

    assert processing.ThreadSafeCache is ThreadSafeCache
    assert callable(enrichment.process_single_patent)
    assert callable(enrichment.process_one_patent_bindings)
    assert callable(organisms.normalize_organism)


def test_normalize_organism_never_assumes_human():
    assert organisms.normalize_organism(None) == "unspecified"
    assert organisms.normalize_organism("") == "unspecified"
    assert organisms.normalize_organism("unknown") == "unspecified"
    assert organisms.normalize_organism("Unspecified species") == "unspecified"
    assert organisms.normalize_organism(" Human ") == "human"
    assert organisms.normalize_organism("Mus musculus") == "mus musculus"


def test_process_one_patent_bindings_enriches_ligand_and_protein_data():
    ligand_extractor = _FakeLigandExtractor()
    bindings = [
        {
            "molecule_name": "Ethanol",
            "molecule_smiles": "CCO",
            "protein_target_name": "EGFR",
            "organism": "human",
            "value": 10,
            "unit": "nM",
        }
    ]
    resolved_proteins = {
        "EGFR": {
            "human": {
                "sequence": "MSEQ",
                "accession": "P00533",
                "uniprot_id": "EGFR_HUMAN",
                "gene": "EGFR",
                "organism_scientific": "Homo sapiens",
            }
        }
    }

    result = enrichment.process_one_patent_bindings(bindings, ligand_extractor, resolved_proteins)

    assert len(result) == 1
    row = result[0]
    assert row["Ligand InChI Key"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert row["Ligand SMILES"] == "CCO"
    assert row["Sequence"] == "MSEQ"
    assert row["Target accession"] == "P00533"
    assert row["UniProt ID"] == "EGFR_HUMAN"
    assert row["gene"] == "EGFR"
    assert row["organism_scientific"] == "Homo sapiens"
    assert row["organism"] == "human"
    assert "species_source" not in row
    assert ligand_extractor.smiles_requests == ["CCO"]


def test_process_one_patent_bindings_flags_assumed_species_but_keeps_sequence():
    ligand_extractor = _FakeLigandExtractor()
    bindings = [
        {
            "molecule_name": "Ethanol",
            "molecule_smiles": "CCO",
            "protein_target_name": "EGFR",
            "organism": "unspecified",
            "value": 10,
            "unit": "nM",
        }
    ]
    resolved_proteins = {
        "EGFR": {
            "human": {
                "sequence": "MSEQ",
                "accession": "P00533",
                "uniprot_id": "EGFR_HUMAN",
                "gene": "EGFR",
                "organism_scientific": "Homo sapiens",
            }
        }
    }

    result = enrichment.process_one_patent_bindings(
        bindings, ligand_extractor, resolved_proteins
    )
    assert len(result) == 1
    row = result[0]
    assert row["organism"] == "human (default)"
    assert "species_source" not in row
    assert row["gene"] == "EGFR"
    assert row["Sequence"] == "MSEQ"


def test_process_one_patent_bindings_keeps_cdx_fallback_without_ligand_inchikey():
    ligand_extractor = _FakeLigandExtractor()
    bindings = [
        {
            "molecule_name": None,
            "protein_target_name": "EGFR",
            "organism": "unknown",
            "inchi_key_cdx": "CDX-INCHIKEY",
        }
    ]

    result = enrichment.process_one_patent_bindings(bindings, ligand_extractor, {})

    assert len(result) == 1
    assert result[0]["organism"] == "unspecified"
    assert "species_source" not in result[0]
    assert "Ligand InChI Key" not in result[0]
    assert ligand_extractor.smiles_requests == []
    assert ligand_extractor.extract_requests == []


def test_process_one_patent_bindings_extracts_by_molecule_name_when_smiles_missing():
    ligand_extractor = _FakeLigandExtractor()
    bindings = [
        {
            "molecule_name": "Example ligand",
            "protein_target_name": "EGFR",
        }
    ]

    result = enrichment.process_one_patent_bindings(bindings, ligand_extractor, {})

    assert len(result) == 1
    assert result[0]["Ligand InChI Key"] == "EXTRACTED-INCHIKEY"
    assert result[0]["Ligand SMILES"] == "EXTRACTED-SMILES"
    assert ligand_extractor.extract_requests == ["Example ligand"]


def test_process_one_patent_bindings_converts_mass_units_using_molecular_weight():
    ligand_extractor = _FakeLigandExtractor()
    bindings = [
        {
            "molecule_name": "Mass ligand",
            "molecule_smiles": "NCC",
            "protein_target_name": "EGFR",
            "value": 2.0,
            "unit": "ng/mL",
            "needs_mw_conversion": True,
        }
    ]

    result = enrichment.process_one_patent_bindings(bindings, ligand_extractor, {})

    assert len(result) == 1
    assert result[0]["value"] == 4.0
    assert result[0]["unit"] == "nM"
    assert result[0]["original_unit_before_mw_conversion"] == "ng/mL"
    assert result[0]["molecular_weight"] == 500.0
    assert "needs_mw_conversion" not in result[0]
    assert ligand_extractor.mw_requests == ["NCC"]


def test_process_single_patent_loads_legacy_patterns_and_adds_patent_number(tmp_path):
    patent_dir = tmp_path / "USUNITTESTA1"
    patent_dir.mkdir()
    (patent_dir / "USUNITTESTA1_resolved.json").write_text(
        json.dumps(
            [
                {
                    "molecule_smiles": "CCO",
                    "protein_target_name": "EGFR",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = enrichment.process_single_patent(
        str(patent_dir),
        _FakeLigandExtractor(),
        patterns=["_resolved.json"],
    )

    assert len(result) == 1
    assert result[0]["patent_number"] == "USUNITTESTA1"
    assert result[0]["Ligand InChI Key"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_reusable_batches_detects_valid_batches_and_force_ignores(tmp_path):
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    good = str(batch_dir / "batch_0002.parquet")
    batches.write_binding_batch([_parquet_ready_row()], good, {})
    processing.write_batch_manifest(good, ["/results/USUNITTESTA1"], 200)
    (batch_dir / "batch_0003.parquet").write_text("not parquet", encoding="utf-8")

    requested = ["/results/USUNITTESTA1"]
    reused, covered = processing.reusable_batches(str(batch_dir), requested, force=False)

    assert reused == {2}
    assert covered == {"/results/USUNITTESTA1"}
    assert not (batch_dir / "batch_0003.parquet").exists()
    assert processing.reusable_batches(str(batch_dir), requested, force=True) == (set(), set())


def test_ligand_extractor_returns_none_when_rdkit_fails():
    extractor = LigandInfoExtractor(
        opsin=None,
        cache=_DictCache(),
        smiles_cache=_DictCache(),
    )

    assert extractor.get_inchi_key_by_smiles("not a smiles") is None


def test_run_bindingdb_processing_builds_local_only_ligand_extractor(tmp_path, monkeypatch):
    created_extractors = []

    class FakeCache(_DictCache):
        def __init__(self, cache_path, cache_name):
            super().__init__()
            self.cache_path = cache_path
            self.cache_name = cache_name

    class FakeLigandInfoExtractor:
        def __init__(self, opsin, cache, smiles_cache):
            created_extractors.append({
                "opsin": opsin,
                "cache": cache,
                "smiles_cache": smiles_cache,
            })

    monkeypatch.setattr(processing, "ThreadSafeCache", FakeCache)
    monkeypatch.setattr(processing, "LigandInfoExtractor", FakeLigandInfoExtractor)
    monkeypatch.setattr(processing, "load_patent_number_dict", lambda _: {})
    monkeypatch.setattr(processing, "get_patent_directories", lambda _: [str(tmp_path / "USUNITTESTA1")])
    monkeypatch.setattr(processing, "process_patent_worker", lambda args: (args[0], [], None))
    monkeypatch.setattr(processing, "write_binding_batch", lambda batch, path, patent_dict: (0, 0))
    monkeypatch.setattr(processing, "merge_parquet_batches", lambda batches_dir, output_file: 1)

    success, message, _elapsed = processing.run_bindingdb_processing(
        input_dir=str(tmp_path),
        output_file=str(tmp_path / "out.parquet"),
        workers=1,
        cache_dir=str(tmp_path / "cache"),
        batch_size=1,
        force=True,
        use_opsin=False,
    )

    assert success is True
    assert "1 bindings written" in message
    assert created_extractors[0]["opsin"] is None
