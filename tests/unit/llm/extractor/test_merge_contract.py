def assay(assay_id: str = "A1") -> dict:
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


def bioactivity(compound: str = "Example 1", assay_id: str = "A1") -> dict:
    return {
        "compound": compound,
        "assay_id": assay_id,
        "binding_metric": "IC50",
        "value": "12",
        "unit": "nM",
        "reasoning": "p0002",
    }


def compound(alias: str = "Example 1", chemical_id: str = "CHEM-US-00001") -> dict:
    return {
        "compound": alias,
        "compound_IUPAC_identifier": "ethanol",
        "reasoning": "p0003",
        "chemical_id": chemical_id,
    }


def test_merge_keeps_stage_contract_fields(make_agent, make_document):
    rows = make_agent()._merge_with_assay_id(
        [assay()],
        [bioactivity()],
        [compound()],
        make_document(),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["compound"] == "Example 1"
    assert row["compound_IUPAC_name"] == "ethanol"
    assert row["chemical_id"] == "CHEM-US-00001"
    assert row["assay"] == "biochemical"
    assert row["organism"] == "human"
    assert row["extreme_conditions"] == "N"
    assert row["stage1_reasoning"] == "p0001"
    assert row["stage2_reasoning"] == "p0002"


def test_unknown_assay_id_is_dropped(make_agent, make_document):
    rows = make_agent()._merge_with_assay_id(
        [assay("A1")],
        [bioactivity(assay_id="A404")],
        [compound()],
        make_document(),
    )

    assert rows == []


def test_unknown_compound_is_dropped(make_agent, make_document):
    rows = make_agent()._merge_with_assay_id(
        [assay()],
        [bioactivity(compound="Example 404")],
        [compound("Example 1")],
        make_document(),
    )

    assert rows == []


def test_safe_alias_matching_maps_ex_to_example(make_agent, make_document):
    rows = make_agent()._merge_with_assay_id(
        [assay()],
        [bioactivity(compound="Ex. 1")],
        [compound("Example 1")],
        make_document(),
    )

    assert len(rows) == 1
    assert rows[0]["compound"] == "Example 1"


def test_unsafe_number_matching_does_not_map_compound_5_to_55(make_agent, make_document):
    rows = make_agent()._merge_with_assay_id(
        [assay()],
        [bioactivity(compound="Compound 5")],
        [compound("Compound 55", "CHEM-US-00055")],
        make_document(),
    )

    assert rows == []


def test_scaffold_chemical_id_is_dropped(make_agent, make_document):
    from patent_processor.data_structures import ChemistryNode

    scaffold = ChemistryNode(
        chemistry_id="CHEM-US-00002",
        chem_num="00002",
        smiles="C[*]",
        is_scaffold=True,
    )

    rows = make_agent()._merge_with_assay_id(
        [assay()],
        [bioactivity(compound="Example 2")],
        [compound("Example 2", "CHEM-US-00002")],
        make_document(scaffold),
    )

    assert rows == []
