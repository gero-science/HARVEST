"""Tests for pure helpers in export_table.py."""
import json
from pathlib import Path

import pandas as pd
import pytest

import export_table


# ---------------------------------------------------------------------------
# extract_chem_num_from_chemical_id
# ---------------------------------------------------------------------------
class TestExtractChemNumFromChemicalId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("CHEM-AB-123", "123"),
            ("CHEM-XY-9", "9"),
            ("CHEM-ZZ-100000", "100000"),
            # The regex uses re.search, so a CHEM token embedded in a longer
            # string is still extracted.
            ("prefix CHEM-AB-42 suffix", "42"),
        ],
    )
    def test_valid_extracts_number(self, raw, expected):
        assert export_table.extract_chem_num_from_chemical_id(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            None,
            "CHEM-A-123",          # only one letter — does not match [A-Z]{2}
            "CHEM-ABC-123",        # three letters
            "chem-ab-123",         # lowercase — regex is case-sensitive
            "CHEM-AB-",            # missing number
            "AB-123",              # missing CHEM prefix
            "random text",
        ],
    )
    def test_invalid_returns_none(self, raw):
        assert export_table.extract_chem_num_from_chemical_id(raw) is None


# ---------------------------------------------------------------------------
# load_patent_number_dict
# ---------------------------------------------------------------------------
class TestLoadPatentNumberDict:
    def test_loads_jsonl(self, tmp_path):
        jsonl = tmp_path / "dict.jsonl"
        records = [
            {"application_number": "US20100001", "patent_number": 1001},
            {"application_number": "US20100002", "patent_number": 1002},
        ]
        with jsonl.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        result = export_table.load_patent_number_dict(str(jsonl))
        assert result == {"US20100001": 1001, "US20100002": 1002}

    def test_missing_file_returns_empty(self, tmp_path):
        assert export_table.load_patent_number_dict(str(tmp_path / "nope.jsonl")) == {}

    def test_loads_csv(self, tmp_path):
        csv_file = tmp_path / "mapping.csv"
        csv_file.write_text(
            "patent_number,application_number\n"
            "US9001,US20200001A1\n"
            "US9002,US20200002A1\n"
        )
        result = export_table.load_patent_number_dict(str(csv_file))
        assert result == {"US20200001A1": 9001, "US20200002A1": 9002}

    def test_none_path_no_default_files_returns_empty(self, tmp_path, monkeypatch):
        # Patch the default mapping path so the repo-shipped CSV is not found.
        monkeypatch.setattr(
            "bindingdb_export.artifacts._DEFAULT_PATENT_MAPPING",
            str(tmp_path / "absent.csv"),
        )
        monkeypatch.chdir(tmp_path)
        assert export_table.load_patent_number_dict(None) == {}

    def test_repo_patent_mapping_csv_loads(self):
        """Smoke test: the repo-shipped curated_data/patent_mapping.csv loads."""
        csv_path = Path(__file__).resolve().parents[3] / "curated_data" / "patent_mapping.csv"
        if not csv_path.exists():
            pytest.skip("curated_data/patent_mapping.csv not present")
        result = export_table.load_patent_number_dict(str(csv_path))
        # ~7 955 rows, ~6 949 unique application_numbers (some share a patent)
        assert len(result) >= 5000
        # spot-check one known mapping
        assert result["US20150266868A1"] == 9447092

    def test_default_none_finds_repo_csv(self):
        """Smoke test: load_patent_number_dict(None) picks up the repo CSV."""
        csv_path = Path(__file__).resolve().parents[3] / "curated_data" / "patent_mapping.csv"
        if not csv_path.exists():
            pytest.skip("curated_data/patent_mapping.csv not present")
        result = export_table.load_patent_number_dict(None)
        assert len(result) >= 5000

    def test_malformed_json_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not valid json at all\n")
        assert export_table.load_patent_number_dict(str(bad)) == {}


# ---------------------------------------------------------------------------
# load_single_proteins
# ---------------------------------------------------------------------------
class TestLoadSingleProteins:
    def test_basic(self, tmp_path):
        payload = {
            "proteins": [
                {
                    "protein_name": "EGFR",
                    "organism": "human",
                    "sequence": "MKTAY",
                    "accession": "P00533",
                    "uniprot_id": "EGFR_HUMAN",
                    "gene": "EGFR",
                    "organism_scientific": "Homo sapiens",
                }
            ]
        }
        (tmp_path / "single_proteins.json").write_text(json.dumps(payload))
        result = export_table.load_single_proteins(str(tmp_path))
        assert "EGFR" in result
        assert "human" in result["EGFR"]
        rec = result["EGFR"]["human"]
        assert rec["sequence"] == "MKTAY"
        assert rec["accession"] == "P00533"
        assert rec["gene"] == "EGFR"

    def test_missing_file_returns_empty(self, tmp_path):
        assert export_table.load_single_proteins(str(tmp_path)) == {}

    def test_unknown_organism_normalized_to_unspecified(self, tmp_path):
        payload = {
            "proteins": [
                {
                    "protein_name": "EGFR",
                    "organism": None,
                    "sequence": "MKTAY",
                }
            ]
        }
        (tmp_path / "single_proteins.json").write_text(json.dumps(payload))
        result = export_table.load_single_proteins(str(tmp_path))
        # normalize_organism: None / "unknown" / "" → "unspecified"
        assert "unspecified" in result["EGFR"]


# ---------------------------------------------------------------------------
# load_protein_complexes
# ---------------------------------------------------------------------------
class TestLoadProteinComplexes:
    def test_basic_complex_with_three_proteins(self, tmp_path):
        # Semicolon join edge cases: test_complex_sequence_join.py
        # Here we exercise the structure.
        payload = {
            "complexes": [
                {
                    "complex_name": "IL-12 receptor signaling",
                    "organism": "human",
                    "species": "Homo sapiens",
                    "proteins": [
                        {"gene": "TYK2", "sequence": "MPRG", "accession": "P29597",
                         "uniprot_id": "TYK2_HUMAN"},
                        {"gene": "JAK2", "sequence": "MGMA", "accession": "O60674",
                         "uniprot_id": "JAK2_HUMAN"},
                        {"gene": "STAT4", "sequence": "MSQW", "accession": "Q14765",
                         "uniprot_id": "STAT4_HUMAN"},
                    ],
                }
            ]
        }
        (tmp_path / "protein_complexes.json").write_text(json.dumps(payload))
        result = export_table.load_protein_complexes(str(tmp_path))
        rec = result["IL-12 receptor signaling"]["human"]
        assert rec["gene"] == "TYK2;JAK2;STAT4"
        assert rec["accession"] == "P29597;O60674;Q14765"
        assert rec["uniprot_id"] == "TYK2_HUMAN;JAK2_HUMAN;STAT4_HUMAN"
        assert rec["organism_scientific"] == "Homo sapiens"

    def test_empty_proteins_skipped(self, tmp_path):
        payload = {"complexes": [{"complex_name": "X", "organism": "human", "proteins": []}]}
        (tmp_path / "protein_complexes.json").write_text(json.dumps(payload))
        assert export_table.load_protein_complexes(str(tmp_path)) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert export_table.load_protein_complexes(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# load_cdx_results
# ---------------------------------------------------------------------------
class TestLoadCdxResults:
    def test_basic(self, tmp_path):
        payload = {
            "compounds": {
                "1": {"smiles": "CCO", "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"},
                "42": {"smiles": "c1ccccc1", "inchikey": "UHOVQNZJYSORNB-UHFFFAOYSA-N"},
            }
        }
        (tmp_path / "cdx_results.json").write_text(json.dumps(payload))
        result = export_table.load_cdx_results(str(tmp_path))
        assert result["1"]["smiles"] == "CCO"
        assert result["42"]["inchikey"] == "UHOVQNZJYSORNB-UHFFFAOYSA-N"

    def test_missing_file_returns_empty(self, tmp_path):
        assert export_table.load_cdx_results(str(tmp_path)) == {}

    def test_malformed_json_returns_empty(self, tmp_path):
        (tmp_path / "cdx_results.json").write_text("{not json", encoding="utf-8")
        assert export_table.load_cdx_results(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# apply_cdx_results_to_bindings
# ---------------------------------------------------------------------------
class TestApplyCdxResultsToBindings:
    def test_adds_smiles_and_inchikey(self):
        bindings = [
            {"chemical_id": "CHEM-AB-1", "value": 100},
            {"chemical_id": "CHEM-AB-42", "value": 200},
        ]
        cdx = {
            "1": {"smiles": "CCO", "inchikey": "ETH1"},
            "42": {"smiles": "c1ccccc1", "inchikey": "BENZ1"},
        }
        export_table.apply_cdx_results_to_bindings(bindings, cdx)
        assert bindings[0]["smiles_cdx"] == "CCO"
        assert bindings[0]["inchi_key_cdx"] == "ETH1"
        assert bindings[1]["smiles_cdx"] == "c1ccccc1"

    def test_unparseable_chemical_id_skipped(self):
        bindings = [{"chemical_id": "GARBAGE", "value": 1}]
        export_table.apply_cdx_results_to_bindings(bindings, {"1": {"smiles": "CCO"}})
        assert "smiles_cdx" not in bindings[0]

    def test_unknown_chem_num_skipped(self):
        bindings = [{"chemical_id": "CHEM-AB-99", "value": 1}]
        export_table.apply_cdx_results_to_bindings(bindings, {"1": {"smiles": "CCO"}})
        assert "smiles_cdx" not in bindings[0]

    def test_empty_cdx_entry_does_not_add_fields(self):
        bindings = [{"chemical_id": "CHEM-AB-1", "value": 1}]
        export_table.apply_cdx_results_to_bindings(bindings, {"1": {"smiles": None, "inchikey": None}})
        assert "smiles_cdx" not in bindings[0]
        assert "inchi_key_cdx" not in bindings[0]


# ---------------------------------------------------------------------------
# binding_to_parquet_row
# ---------------------------------------------------------------------------
def _full_row(**overrides):
    """Build a maximally-complete input row; tests override specific fields."""
    base = {
        "Ligand SMILES": "CCO",
        "Ligand InChI Key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "smiles_source": "table_chemistry_tag",
        "smiles_cdx": "",
        "inchi_key_cdx": "",
        "Sequence": "MKTAY",
        "binding_metric": "IC50",
        "value": 100.0,
        "relation": "=",
        "original_range": "",
        "patent_number": "US20100001A1",
        "chemical_id": "CHEM-AB-1",
        "compound": "Example 1",
        "compound_IUPAC_name": "ethanol",
        "original_alias": "Ex. 1",
        "protein_target_name": "EGFR",
        "gene": "EGFR",
        "organism": "human",
        "organism_scientific": "Homo sapiens",
        "Target accession": "P00533",
        "UniProt ID": "EGFR_HUMAN",
        "is_complex": "N",
        "extreme_conditions": "N",
    }
    base.update(overrides)
    return base


class TestBindingToParquetRow:
    def test_ic50_row(self):
        row = _full_row()
        out = export_table.binding_to_parquet_row(row, patent_dict={"US20100001A1": 12345})
        assert out is not None
        assert out["Ligand SMILES"] == "CCO"
        assert out["Ligand InChI Key"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        assert out["IC50 (nM)"] == "100.0"
        # Other metric columns must stay None.
        assert out["Ki (nM)"] is None
        assert out["Kd (nM)"] is None
        assert out["EC50 (nM)"] is None
        assert out["normalized_pub_number"] == 12345
        assert out["patent_number"] == "US20100001A1"
        assert out["protein_target_name"] == "EGFR"
        assert set(out) == set(export_table.OUTPUT_COLUMNS)

    def test_species_source_is_not_written(self):
        row = _full_row(species_source="human_default")
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert "species_source" not in out
        assert set(out) == set(export_table.OUTPUT_COLUMNS)

    def test_organism_is_passed_through(self):
        out = export_table.binding_to_parquet_row(
            _full_row(organism="human (default)"), patent_dict={}
        )
        assert out["organism"] == "human (default)"

    @pytest.mark.parametrize(
        "metric,column",
        [("IC50", "IC50 (nM)"), ("EC50", "EC50 (nM)"),
         ("Ki", "Ki (nM)"), ("Kd", "Kd (nM)")],
    )
    def test_metric_routes_to_correct_column(self, metric, column):
        row = _full_row(binding_metric=metric, value=42.5)
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out[column] == "42.5"

    def test_missing_inchi_key_returns_none(self):
        row = _full_row(**{"Ligand InChI Key": None, "inchi_key_cdx": ""})
        assert export_table.binding_to_parquet_row(row, patent_dict={}) is None

    def test_inchi_key_from_cdx_alone_is_enough(self):
        row = _full_row(**{"Ligand InChI Key": None, "inchi_key_cdx": "FROM_CDX"})
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out is not None
        assert out["Ligand InChI Key"] == "FROM_CDX"

    def test_smiles_from_cdx_is_used_when_ligand_smiles_missing(self):
        row = _full_row(**{"Ligand SMILES": "", "smiles_cdx": "CDX_SMILES"})
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out is not None
        assert out["Ligand SMILES"] == "CDX_SMILES"

    def test_missing_protein_returns_none(self):
        row = _full_row(Sequence="", gene="")
        assert export_table.binding_to_parquet_row(row, patent_dict={}) is None

    def test_gene_alone_is_enough(self):
        row = _full_row(Sequence="")
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out is not None

    def test_missing_value_keeps_row_with_none_metric(self):
        row = _full_row(value=None)
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out is not None
        assert out["IC50 (nM)"] is None

    def test_inhibition_metric_excluded(self):
        row = _full_row(binding_metric="Inhibition", value=75.0)
        assert export_table.binding_to_parquet_row(row, patent_dict={}) is None

    def test_unknown_metric_excluded(self):
        row = _full_row(binding_metric="HillSlope", value=1.5)
        assert export_table.binding_to_parquet_row(row, patent_dict={}) is None

    def test_iupac_kept_only_for_pyopsin_source(self):
        row = _full_row(smiles_source="py2opsin_full_name", compound_IUPAC_name="ethanol")
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out["compound_IUPAC_name"] == "ethanol"

    def test_iupac_dropped_when_smiles_from_other_source(self):
        row = _full_row(smiles_source="table_chemistry_tag", compound_IUPAC_name="ethanol")
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out["compound_IUPAC_name"] == ""

    def test_normalized_pub_number_defaults_to_zero(self):
        row = _full_row()
        out = export_table.binding_to_parquet_row(row, patent_dict={})  # patent not in dict
        assert out["normalized_pub_number"] == 0

    def test_relation_and_range_propagate(self):
        row = _full_row(relation="<", value=10.0)
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out["relation"] == "<"
        assert out["IC50 (nM)"] == "10.0"

        row = _full_row(relation="range", value=15.0, original_range="10.0 to 20.0")
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out["relation"] == "range"
        assert out["original_range"] == "10.0 to 20.0"

    def test_mutations_list_is_joined(self):
        row = _full_row(mutations=["T790M", "L858R"])
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out["mutations"] == "T790M, L858R"

    def test_mutations_absent_yields_empty_string(self):
        row = _full_row()
        out = export_table.binding_to_parquet_row(row, patent_dict={})
        assert out["mutations"] == ""
