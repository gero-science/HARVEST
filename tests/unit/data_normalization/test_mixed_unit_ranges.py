"""Ranges that span different units in normalize_value.

">100 nM - 10 µM" used to be parsed by stripping non-digits and averaging bare
numbers (100 and 10 → 55). Each endpoint must be converted to nM before averaging
(e.g. mean of 100 nM and 10000 nM → 5050 nM).
"""
import pytest

from data_normalization.normalize_data import normalize_value

pytestmark = pytest.mark.regression


@pytest.mark.parametrize(
    "raw,expected_avg_nM",
    [
        (">100 nM - 10 µM", 5050.0),
        ("100 nM to 10 µM", 5050.0),
        (">100 nM - 10 μM", 5050.0),
        ("100 nM to 200 nM", 150.0),
        ("0.1 µM to 1 µM", 550.0),
        ("1 nM - 1 mM", (1 + 1_000_000) / 2),
    ],
)
def test_mixed_unit_range_yields_correct_nM_mean(raw, expected_avg_nM):
    value, relation, _ = normalize_value(raw)
    assert relation == "range"
    assert value == pytest.approx(expected_avg_nM, rel=1e-6)


def test_mixed_unit_range_does_not_average_bare_numbers():
    """Guard against averaging (100 + 10) / 2 = 55 without unit conversion."""
    value, _, _ = normalize_value(">100 nM - 10 µM")
    assert value != pytest.approx(55.0)
    assert value > 1000


def test_original_range_includes_unit_for_mixed_units():
    _, _, original_range = normalize_value(">100 nM - 10 µM")
    assert "nM" in original_range
