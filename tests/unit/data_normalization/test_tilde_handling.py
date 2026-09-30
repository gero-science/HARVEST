"""Tilde (~) handling in normalize_value.

- "~50" (tilde at start, "approximately") → strip the tilde, value is 50.
- "50~500" (tilde between digits) → range separator, value is mean (275),
  relation is "range".

A global `s.replace("~", "")` used to turn "50~500" into "50500" and break
range parsing for typography that uses tilde as a range dash.
"""
import pytest

from data_normalization.normalize_data import normalize_value

pytestmark = pytest.mark.regression


@pytest.mark.parametrize(
    "raw,expected_value,expected_relation",
    [
        ("~50", 50.0, "="),
        ("~5.3", 5.3, "="),
        ("~ 100", 100.0, "="),
        ("~ 1.5e-3", 1.5e-3, "="),
    ],
)
def test_tilde_at_start_means_approximately(raw, expected_value, expected_relation):
    value, relation, _ = normalize_value(raw)
    assert value == pytest.approx(expected_value)
    assert relation == expected_relation


def test_tilde_between_digits_is_range_separator():
    value, relation, original_range = normalize_value("50~500")
    assert relation == "range"
    assert value == pytest.approx(275.0)
    assert original_range


def test_tilde_range_with_decimals():
    value, relation, _ = normalize_value("0.5~1.5")
    assert relation == "range"
    assert value == pytest.approx(1.0)


def test_tilde_does_not_corrupt_numbers():
    """Direct check that "50~500" is NOT silently parsed as 50500."""
    value, _, _ = normalize_value("50~500")
    assert value != 50500
