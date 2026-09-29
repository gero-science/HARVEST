import json

import export_table


class DummyLigandExtractor:
    def get_inchi_key_by_smiles(self, smiles):
        return "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"

    def get_molecular_weight(self, smiles):
        return 46.07


def binding_row(source: str, compound: str) -> dict:
    return {
        "compound": compound,
        "molecule_smiles": "CCO",
        "molecule_inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "smiles_source": "py2opsin",
        "agent2_resolution_source": source,
        "protein_target_name": "EGFR",
        "organism": "human",
        "binding_metric": "IC50",
        "value": "12",
        "unit": "nM",
    }


def test_bindingdb_accepts_only_agent1_enriched_rows_from_agent2_contract(tmp_path):
    patent_dir = tmp_path / "USUNITTESTA1"
    patent_dir.mkdir()
    (patent_dir / "USUNITTESTA1_resolved.json").write_text(
        json.dumps(
            [
                binding_row("skipped_enriched_by_agent1", "kept"),
                binding_row("llm_unverified", "filtered"),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (patent_dir / "single_proteins.json").write_text(
        json.dumps(
            {
                "proteins": [
                    {
                        "protein_name": "EGFR",
                        "organism": "human",
                        "sequence": "MTEYK",
                        "accession": "P00533",
                        "uniprot_id": "EGFR_HUMAN",
                        "gene": "EGFR",
                        "organism_scientific": "Homo sapiens",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    _, rows, error = export_table.process_patent_worker((str(patent_dir), DummyLigandExtractor()))

    assert error is None
    assert len(rows) == 1
    assert rows[0]["compound"] == "kept"
    assert rows[0]["agent2_resolution_source"] == "skipped_enriched_by_agent1"
    assert rows[0]["Ligand InChI Key"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert rows[0]["Sequence"] == "MTEYK"
