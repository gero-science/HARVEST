"""Tests for _validate_item minimal required fields (bioactivity_extraction.validation)."""

import pytest


def _valid_item() -> dict:
    return {
        "compound": "Example 1",
        "value": "10",
        "binding_metric": "IC50",
        "protein_target_name": "EGFR",
    }


def test_validate_item_accepts_complete_item(make_agent):
    assert make_agent()._validate_item(_valid_item()) is True


@pytest.mark.parametrize("not_dict", [None, "x", 42, ["compound"], ("a",)])
def test_validate_item_rejects_non_dict(make_agent, not_dict):
    assert make_agent()._validate_item(not_dict) is False


def test_validate_item_requires_some_identifier(make_agent):
    item = _valid_item()
    del item["compound"]
    assert make_agent()._validate_item(item) is False


def test_validate_item_accepts_chemical_id_as_identifier(make_agent):
    item = _valid_item()
    del item["compound"]
    item["chemical_id"] = "CHEM-US-00001"
    assert make_agent()._validate_item(item) is True


@pytest.mark.parametrize("missing", ["value", "binding_metric", "protein_target_name"])
def test_validate_item_requires_value_metric_target(make_agent, missing):
    item = _valid_item()
    item[missing] = ""
    assert make_agent()._validate_item(item) is False
