import llm.extractor as extractor_module
import bioactivity_extraction.structure_enrichment as enrichment_module
from patent_processor.data_structures import ChemistryNode


def fake_py2opsin(*, chemical_name, output_format, allow_bad_stereo=True):
    values = []
    for name in chemical_name:
        if name == "ethanol" and output_format == "SMILES":
            values.append("CCO")
        elif name == "ethanol" and output_format == "StdInChIKey":
            values.append("LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
        else:
            values.append(False)
    return values


def test_py2opsin_has_priority_over_chemistry_node(monkeypatch, make_agent, make_document):
    monkeypatch.setattr(extractor_module, "PY2OPSIN_AVAILABLE", True)
    monkeypatch.setattr(extractor_module, "py2opsin", fake_py2opsin)
    row = {
        "compound": "Example 12",
        "compound_IUPAC_name": "ethanol",
        "chemical_id": "CHEM-US-00012",
    }
    doc = make_document(
        ChemistryNode(
            chemistry_id="CHEM-US-00012",
            chem_num="00012",
            smiles="c1ccccc1",
            inchikey="UHOVQNZJYSORNB-UHFFFAOYSA-N",
        )
    )

    make_agent()._enrich_with_smiles([row], doc)

    assert row["molecule_smiles"] == "CCO"
    assert row["molecule_inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert row["smiles_source"] == "py2opsin"


def test_py2opsin_fix_iupac_variants_are_sorted(monkeypatch, make_agent, make_document):
    """Bracket-fix variants must be tried in stable order, not set iteration order."""

    class Variant(str):
        def __new__(cls, value, hash_value):
            obj = str.__new__(cls, value)
            obj.hash_value = hash_value
            return obj

        def __hash__(self):
            return self.hash_value

    calls = []

    def fake_py2opsin_with_fix_variants(*, chemical_name, output_format, allow_bad_stereo=True):
        names = list(chemical_name)
        calls.append((output_format, names))

        if output_format == "SMILES":
            if names in (["bad original"], ["normalized"]):
                return [False]
            return [f"SMILES-{name[-1].upper()}" for name in names]

        if output_format == "StdInChIKey":
            if names in (["bad original"], ["normalized"]):
                return [False]
            return [f"KEY-{name[-1].upper()}" for name in names]

        raise AssertionError(output_format)

    monkeypatch.setattr(extractor_module, "PY2OPSIN_AVAILABLE", True)
    monkeypatch.setattr(extractor_module, "py2opsin", fake_py2opsin_with_fix_variants)
    monkeypatch.setattr(enrichment_module, "fix_balanced", lambda name: name)
    monkeypatch.setattr(enrichment_module, "normalize_iupac_spelling", lambda name: "normalized")
    monkeypatch.setattr(enrichment_module, "fix_balanced_by_adding", lambda name: Variant("variant-b", 1))
    monkeypatch.setattr(
        enrichment_module,
        "fix_balanced_by_removing_variants",
        lambda name: [Variant("variant-a", 2)],
    )

    row = {
        "compound": "Example 53",
        "compound_IUPAC_name": "bad original",
        "chemical_id": "CHEM-US-00053",
    }

    make_agent()._enrich_with_smiles([row], make_document())

    fix_smiles_call = next(
        names
        for output_format, names in calls
        if output_format == "SMILES" and names == ["variant-a", "variant-b"]
    )
    assert fix_smiles_call == ["variant-a", "variant-b"]
    assert row["molecule_smiles"] == "SMILES-A"
    assert row["molecule_inchikey"] == "KEY-A"
    assert row["smiles_source"] == "py2opsin_fix_iupac"


def test_chemistry_node_is_fallback_when_py2opsin_unavailable(monkeypatch, make_agent, make_document):
    monkeypatch.setattr(extractor_module, "PY2OPSIN_AVAILABLE", False)
    row = {
        "compound": "Example 12",
        "compound_IUPAC_name": "",
        "chemical_id": "CHEM-US-00012",
    }
    doc = make_document(
        ChemistryNode(
            chemistry_id="CHEM-US-00012",
            chem_num="00012",
            smiles="c1ccccc1",
            inchikey="UHOVQNZJYSORNB-UHFFFAOYSA-N",
        )
    )

    make_agent()._enrich_with_smiles([row], doc)

    assert row["molecule_smiles"] == "c1ccccc1"
    assert row["molecule_inchikey"] == "UHOVQNZJYSORNB-UHFFFAOYSA-N"
    assert row["smiles_source"] == "table_chemistry_tag"


def test_chemistry_node_matching_is_asymmetric_for_leading_zeroes(monkeypatch, make_agent, make_document):
    """chemical_id matching with ChemistryNode is asymmetric for leading zeros.

    The node is stored with id 'CHEM-US-00012'. A row whose chemical_id also has leading
    zeros ('CHEM-US-00012') matches. A row with an already normalized id ('CHEM-US-12')
    does NOT match: the guard in _enrich_with_smiles skips lookup
    in the normalized map when chem_id == normalized_chem_id, and the original map has no
    key 'CHEM-US-12'.
    """
    node = ChemistryNode(
        chemistry_id="CHEM-US-00012",
        chem_num="00012",
        smiles="CCN",
        inchikey="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
    )

    # With leading zeros → match exists
    monkeypatch.setattr(extractor_module, "PY2OPSIN_AVAILABLE", False)
    row_with_zeroes = {"compound": "Example 12", "compound_IUPAC_name": "", "chemical_id": "CHEM-US-00012"}
    make_agent()._enrich_with_smiles([row_with_zeroes], make_document(node))
    assert row_with_zeroes["molecule_smiles"] == "CCN"
    assert row_with_zeroes["smiles_source"] == "table_chemistry_tag"

    # Without leading zeros → no match (current asymmetry)
    row_stripped = {"compound": "Example 12", "compound_IUPAC_name": "", "chemical_id": "CHEM-US-12"}
    make_agent()._enrich_with_smiles([row_stripped], make_document(node))
    assert "molecule_smiles" not in row_stripped
    assert "smiles_source" not in row_stripped


def test_scaffold_chemistry_node_is_not_used(monkeypatch, make_agent, make_document):
    monkeypatch.setattr(extractor_module, "PY2OPSIN_AVAILABLE", False)
    row = {
        "compound": "Example 12",
        "compound_IUPAC_name": "",
        "chemical_id": "CHEM-US-00012",
    }
    doc = make_document(
        ChemistryNode(
            chemistry_id="CHEM-US-00012",
            chem_num="00012",
            smiles="C[*]",
            inchikey="",
            is_scaffold=True,
        )
    )

    make_agent()._enrich_with_smiles([row], doc)

    assert "molecule_smiles" not in row
    assert "smiles_source" not in row
