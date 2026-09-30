"""Tests for Stage 3 deduplication of IUPAC names and chemical_id (bioactivity_extraction.stage3_cleanup)."""


def test_check_duplicate_iupac_names_empty_returns_empty_set(make_agent):
    assert make_agent()._check_duplicate_iupac_names([]) == set()


def test_check_duplicate_iupac_names_finds_only_repeated_nonblank(make_agent):
    data = [
        {"compound_IUPAC_name": "ethanol"},
        {"compound_IUPAC_name": "ethanol"},
        {"compound_IUPAC_name": "methanol"},
        {"compound_IUPAC_name": "  "},
        {"compound_IUPAC_name": ""},
        {},
    ]
    assert make_agent()._check_duplicate_iupac_names(data) == {"ethanol"}


def test_check_duplicate_iupac_names_strips_before_counting(make_agent):
    data = [
        {"compound_IUPAC_name": "ethanol"},
        {"compound_IUPAC_name": " ethanol "},
    ]
    assert make_agent()._check_duplicate_iupac_names(data) == {"ethanol"}


def test_remove_duplicate_iupac_names_blanks_matches_and_copies(make_agent):
    data = [
        {"compound": "A", "compound_IUPAC_name": "ethanol"},
        {"compound": "B", "compound_IUPAC_name": "methanol"},
    ]
    cleaned = make_agent()._remove_duplicate_iupac_names(data, {"ethanol"})

    assert cleaned[0]["compound_IUPAC_name"] == ""
    assert cleaned[1]["compound_IUPAC_name"] == "methanol"
    # original is unchanged (method copies elements)
    assert data[0]["compound_IUPAC_name"] == "ethanol"


def test_check_duplicate_chemical_ids_finds_only_repeated_nonblank(make_agent):
    data = [
        {"chemical_id": "CHEM-US-00001"},
        {"chemical_id": "CHEM-US-00001"},
        {"chemical_id": "CHEM-US-00002"},
        {"chemical_id": ""},
        {},
    ]
    assert make_agent()._check_duplicate_chemical_ids(data) == {"CHEM-US-00001"}


def test_remove_duplicate_chemical_ids_blanks_matches_and_copies(make_agent):
    data = [
        {"compound": "A", "chemical_id": "CHEM-US-00001"},
        {"compound": "B", "chemical_id": "CHEM-US-00002"},
    ]
    cleaned = make_agent()._remove_duplicate_chemical_ids(data, {"CHEM-US-00001"})

    assert cleaned[0]["chemical_id"] == ""
    assert cleaned[1]["chemical_id"] == "CHEM-US-00002"
    assert data[0]["chemical_id"] == "CHEM-US-00001"
