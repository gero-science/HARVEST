from final_postprocessing.add_final_structure import (
    classify_structure_pair,
    reason_for_conflict_class,
    select_final_structure,
)


def test_only_opsin_selects_opsin_clean_structure():
    result = select_final_structure(
        "OCC",
        None,
        opsin_clean_smiles="CCO",
        opsin_clean_inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    )

    assert result.final_smiles == "CCO"
    assert result.final_inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert result.structure_choice_reason == "only_opsin"


def test_sources_match_uses_opsin_clean_structure():
    result = select_final_structure(
        "OCC",
        "CCO",
        opsin_clean_smiles="CCO",
        cdx_clean_smiles="CCO",
        opsin_clean_inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        cdx_clean_inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    )

    assert result.final_smiles == "CCO"
    assert result.structure_choice_reason == "sources_match"


def test_same_heavy_atoms_different_connectivity_prefers_cdx():
    result = select_final_structure(
        "CCCO",
        "CC(C)O",
        opsin_clean_smiles="CCCO",
        cdx_clean_smiles="CC(C)O",
        opsin_clean_inchikey="BDERNNFJNOPAEC-UHFFFAOYSA-N",
        cdx_clean_inchikey="KFZMGEQAYNKOFK-UHFFFAOYSA-N",
    )

    assert classify_structure_pair("CCCO", "CC(C)O") == "same_heavy_atoms_different_connectivity"
    assert result.final_smiles == "CC(C)O"
    assert result.structure_choice_reason == "prefer_cdx_same_size_graph"


def test_cdx_multifragment_prefers_opsin():
    result = select_final_structure(
        "CCCO",
        "CCO.CC",
        opsin_clean_smiles="CCCO",
        cdx_clean_smiles="CCO.CC",
        opsin_clean_inchikey="BDERNNFJNOPAEC-UHFFFAOYSA-N",
        cdx_clean_inchikey="LCGLNKUTAGEVQW-UHFFFAOYSA-N",
    )

    assert classify_structure_pair("CCCO", "CCO.CC") == "cdx_multifragment_opsin_single"
    assert result.final_smiles == "CCCO"
    assert result.structure_choice_reason == "prefer_opsin_cdx_multifragment"


def test_small_delta_prefers_opsin():
    result = select_final_structure(
        "CCCCO",
        "CCCO",
        opsin_clean_smiles="CCCCO",
        cdx_clean_smiles="CCCO",
        opsin_clean_inchikey="LRHPLDYGYMQRHN-UHFFFAOYSA-N",
        cdx_clean_inchikey="BDERNNFJNOPAEC-UHFFFAOYSA-N",
    )

    assert classify_structure_pair("CCCCO", "CCCO") == "small_size_delta_1_2_atoms"
    assert result.final_smiles == "CCCCO"
    assert result.structure_choice_reason == "prefer_opsin_small_delta"


def test_reason_mapping_matches_analysis_policy():
    assert reason_for_conflict_class("opsin_larger") == (
        "opsin",
        "prefer_opsin_opsin_larger",
    )
    assert reason_for_conflict_class("cdx_larger") == (
        "opsin",
        "prefer_opsin_cdx_larger",
    )
    assert reason_for_conflict_class("same_heavy_atoms_different_connectivity") == (
        "cdx",
        "prefer_cdx_same_size_graph",
    )
