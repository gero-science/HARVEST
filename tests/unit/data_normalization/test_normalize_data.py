"""Tests for data_normalization.normalize_data — unit conversion is the
single most safety-critical layer in the dataset; if any of these break,
the whole final parquet is silently miscomputed."""
import math

import pytest

from data_normalization.normalize_data import (
    _clean_unit,
    _convert_value_to_nM,
    _extract_value_and_unit,
    convert_mass_to_nM,
    convert_to_nM,
    is_activity_unit,
    normalize_metric_name,
    normalize_unit_name,
    normalize_value,
    process_row,
)


# ---------------------------------------------------------------------------
# _extract_value_and_unit
# ---------------------------------------------------------------------------
class TestExtractValueAndUnit:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("100 nM", (100.0, "nM")),
            ("100nM", (100.0, "nM")),
            ("10 µM", (10.0, "µM")),
            ("10µM", (10.0, "µM")),
            ("10uM", (10.0, "uM")),
            # μ (U+03BC GREEK SMALL LETTER MU) — different codepoint from µ (U+00B5)
            ("10μM", (10.0, "μM")),
            ("0.5 mM", (0.5, "mM")),
            (">100 nM", (100.0, "nM")),
            (">=100nM", (100.0, "nM")),
            ("<= 1 µM", (1.0, "µM")),
            # value alone (no unit) is fine — unit is None
            ("100", (100.0, None)),
            # unit with /mL
            ("5 ng/mL", (5.0, "ng/mL")),
        ],
    )
    def test_extracts(self, raw, expected):
        assert _extract_value_and_unit(raw) == expected

    @pytest.mark.parametrize("raw", ["", "  ", "abc", "1.2.3 nM", "10x100 nM"])
    def test_unparseable_returns_none(self, raw):
        assert _extract_value_and_unit(raw) == (None, None)


# ---------------------------------------------------------------------------
# _convert_value_to_nM
# ---------------------------------------------------------------------------
class TestConvertValueToNM:
    @pytest.mark.parametrize(
        "value,unit,expected",
        [
            (100, "nM", 100),
            (10, "uM", 10_000),
            (10, "µM", 10_000),
            # Greek small mu (U+03BC) treated identically to micro sign (U+00B5)
            (10, "μM", 10_000),
            (0.5, "mM", 500_000),
            (1, "M", 1_000_000_000),
            (5, "pM", 0.005),
            (5, "fM", 0.000_005),
            # case-insensitive
            (1, "NM", 1),
            (1, "Um", 1_000),
        ],
    )
    def test_converts(self, value, unit, expected):
        assert _convert_value_to_nM(value, unit) == pytest.approx(expected)

    def test_none_inputs(self):
        assert _convert_value_to_nM(None, "nM") is None
        assert _convert_value_to_nM(10, None) is None
        assert _convert_value_to_nM(None, None) is None

    def test_unknown_unit(self):
        assert _convert_value_to_nM(10, "kg") is None
        assert _convert_value_to_nM(10, "ng/mL") is None  # mass unit, not handled here


# ---------------------------------------------------------------------------
# normalize_value
# ---------------------------------------------------------------------------
class TestNormalizeValue:
    def test_returns_three_tuple_default_relation(self):
        v, rel, rng = normalize_value("100")
        assert v == 100.0
        assert rel == "="
        assert rng == ""

    @pytest.mark.parametrize(
        "raw,value,relation",
        [
            ("<10", 10.0, "<"),
            (">100", 100.0, ">"),
            ("<=5", 5.0, "<="),
            (">=200", 200.0, ">="),
            ("≤1", 1.0, "<="),
            ("≥1", 1.0, ">="),
            ("≦1", 1.0, "<="),
            ("≧1", 1.0, ">="),
        ],
    )
    def test_relation_operators(self, raw, value, relation):
        v, rel, _ = normalize_value(raw)
        assert v == value
        assert rel == relation

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.5*10^-6", 1.5e-6),
            ("1.5x10^-6", 1.5e-6),
            ("1.5×10^-6", 1.5e-6),
            ("2*10^3", 2_000.0),
            # plain scientific notation
            ("1.2e-3", 1.2e-3),
        ],
    )
    def test_scientific_notation(self, raw, expected):
        v, rel, _ = normalize_value(raw)
        assert v == pytest.approx(expected)
        assert rel == "="

    @pytest.mark.parametrize(
        "raw,expected_avg",
        [
            ("10 to 20", 15.0),
            ("10-20", 15.0),
            ("100 and 200", 150.0),
            # range via "to" with units that get mean-reduced
            ("100 nM to 200 nM", 150.0),
        ],
    )
    def test_simple_ranges(self, raw, expected_avg):
        v, rel, rng = normalize_value(raw)
        assert v == pytest.approx(expected_avg)
        assert rel == "range"
        assert rng != ""

    def test_between_x_and_y_takes_lower(self):
        # Per code: "between X and Y" → keep first (smaller) part.
        v, rel, _ = normalize_value("between 10 and 20")
        assert v == 10.0
        assert rel == "="

    def test_plus_minus_keeps_central_value(self):
        v, rel, _ = normalize_value("7.5±0.3")
        assert v == 7.5
        assert rel == "="

    def test_trailing_plus_value(self):
        # "7.9+0.8" → 7.9
        v, rel, _ = normalize_value("7.9+0.8")
        assert v == 7.9
        assert rel == "="

    def test_about_approx_words_stripped(self):
        v, _, _ = normalize_value("about 50")
        assert v == 50.0
        v, _, _ = normalize_value("approx 50")
        assert v == 50.0
        v, _, _ = normalize_value("at least 50")
        assert v == 50.0

    def test_approx_dot_form_known_buggy(self):
        # "approx. 50" -> 0.5 in the current implementation: the regex
        # `\bapprox\.?\b` strips "approx" but leaves the ".", so the leading
        # dot ends up parsed as "0.50". Pinning the actual behavior so a
        # future fix shows up as a deliberate test change.
        v, _, _ = normalize_value("approx. 50")
        assert v == 0.5

    def test_range_with_metric_inside(self):
        # "20000>=IC50>=500" — code path "range_with_metric"
        v, rel, rng = normalize_value("20000>=IC50>=500")
        assert v == pytest.approx((500 + 20000) / 2)
        assert rel == "range"

    def test_non_string_returns_none(self):
        assert normalize_value(None) == (None, "=", "")
        assert normalize_value(123) == (None, "=", "")
        assert normalize_value(3.14) == (None, "=", "")

    def test_garbage_returns_none(self):
        assert normalize_value("not a number")[0] is None

    # --- CLASS-1 impossible-value regressions (digit glue / sci / sandwich) ---

    def test_sci_notation_with_leading_relation(self):
        # US20170101391A1 / US20200199083A1: ">2.00E-5" must NOT become 2.005
        v, rel, _ = normalize_value(">2.00E-5")
        assert v == pytest.approx(2e-5)
        assert rel == ">"
        v, rel, _ = normalize_value(">2.00E−5")  # unicode minus
        assert v == pytest.approx(2e-5)
        assert rel == ">"

    def test_sci_notation_range_endpoints(self):
        # US20160145297A1: "1.0E-08 to 1.0E-10" must NOT become 1.008..1.01
        v, rel, rng = normalize_value("1.0E-08 to 1.0E-10")
        assert v == pytest.approx(5.05e-9)
        assert rel == "range"
        assert "1.008" not in rng

    def test_sandwich_with_units_and_default_unit(self):
        # US20240066027A1: "20 nM<IC50<10" + uM → ~5010 nM, not 205010000
        v, rel, rng = normalize_value("20 nM<IC50<10", default_unit="uM")
        assert v == pytest.approx(5010.0)
        assert rel == "range"
        assert rng.endswith(" nM")

        v, rel, rng = normalize_value("20 nM < IC50 < 10 uM")
        assert v == pytest.approx(5010.0)
        assert rel == "range"
        assert rng.endswith(" nM")

    def test_slash_multivalue_rejected(self):
        # US20210355104A1: "133/209/375/142/48" must not digit-glue
        assert normalize_value("133/209/375/142/48")[0] is None
        assert normalize_value("7.2/31.4")[0] is None

    def test_extract_sci_notation(self):
        assert _extract_value_and_unit("1.0E-08") == (1e-8, None)
        assert _extract_value_and_unit("2.00e-5 M")[0] == pytest.approx(2e-5)


# ---------------------------------------------------------------------------
# normalize_metric_name
# ---------------------------------------------------------------------------
class TestNormalizeMetricName:
    @pytest.mark.parametrize(
        "raw,name,is_log",
        [
            ("IC50", "IC50", False),
            ("ic50", "IC50", False),
            ("IC-50", "IC50", False),
            ("IC 50", "IC50", False),
            ("IC₅₀", "IC50", False),
            ("cERK IC50", "IC50", False),
            ("aERK2 IC50", "IC50", False),
            ("pIC50", "IC50", True),
            ("logIC50", "IC50", True),
            ("Ki", "Ki", False),
            ("K_i", "Ki", False),
            ("pKi", "Ki", True),
            ("logKi", "Ki", True),
            ("Kd", "Kd", False),
            ("K_d", "Kd", False),
            ("pKd", "Kd", True),
            ("Dissociation Constant", "Kd", False),
            ("EC50", "EC50", False),
            ("EC-50", "EC50", False),
            ("pEC50", "EC50", True),
            ("Inhibition", "Inhibition", False),
            ("% inhibition", "Inhibition", False),
        ],
    )
    def test_known_metrics(self, raw, name, is_log):
        assert normalize_metric_name(raw) == (name, is_log)

    @pytest.mark.parametrize("raw", ["", "foo", "Hill slope", "AC50"])
    def test_unknown_returns_none(self, raw):
        assert normalize_metric_name(raw) == (None, False)

    def test_non_string_input(self):
        assert normalize_metric_name(None) == (None, False)
        assert normalize_metric_name(42) == (None, False)


# ---------------------------------------------------------------------------
# normalize_unit_name + _clean_unit
# ---------------------------------------------------------------------------
class TestNormalizeUnitName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("nM", "nM"),
            ("nm", "nM"),
            ("nanomolar", "nM"),
            ("uM", "uM"),
            ("µM", "uM"),
            ("μM", "uM"),
            ("micromolar", "uM"),
            ("microM", "uM"),
            ("mM", "mM"),
            ("millimolar", "mM"),
            ("pM", "pM"),
            ("fM", "fM"),
            ("M", "M"),
            ("molar", "M"),
            # mass units
            ("ng/mL", "ng/mL"),
            ("mg/mL", "mg/mL"),
            ("µg/mL", "µg/mL"),
            ("pg/mL", "pg/mL"),
            ("%", "%"),
            ("percent", "%"),
            # mol-based aliases
            ("nmol", "nM"),
            ("µmol", "uM"),
            ("mmol", "mM"),
            ("pmol", "pM"),
            ("fmol", "fM"),
        ],
    )
    def test_known_units(self, raw, expected):
        assert normalize_unit_name(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # /mL forms must yield (unit, multiplier=1000) — caller multiplies
            ("nmol/mL", ("nM", 1000)),
            ("umol/mL", ("uM", 1000)),
            ("mmol/mL", ("mM", 1000)),
            ("pmol/mL", ("pM", 1000)),
            ("fmol/mL", ("fM", 1000)),
            ("mol/mL", ("M", 1000)),
        ],
    )
    def test_per_ml_returns_tuple_with_multiplier(self, raw, expected):
        assert normalize_unit_name(raw) == expected

    @pytest.mark.parametrize("raw", ["", "kg", "foobar", "minutes"])
    def test_unknown_returns_none(self, raw):
        assert normalize_unit_name(raw) is None

    def test_non_string_input(self):
        assert normalize_unit_name(None) is None
        assert normalize_unit_name(123) is None


class TestCleanUnit:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("µM", "um"),
            ("μM", "um"),
            ("nmol / L", "nmol/l"),
            ("nmol per L", "nmol/l"),
            ("nmol・L", "nmol/l"),
            ("Litre", "l"),
            ("liter", "l"),
            (" Ca2+ nM", "nm"),
        ],
    )
    def test_clean(self, raw, expected):
        assert _clean_unit(raw) == expected


class TestIsActivityUnit:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("U/min", True),
            ("nmol/min/mg", True),
            ("pmol/sec", True),
            ("mg/h", True),
            ("nM", False),
            ("ng/mL", False),
        ],
    )
    def test_detects(self, raw, expected):
        assert is_activity_unit(raw) is expected


# ---------------------------------------------------------------------------
# convert_mass_to_nM
# ---------------------------------------------------------------------------
class TestConvertMassToNM:
    def test_mg_per_ml_with_known_mw(self):
        # MW=100 g/mol; 1 mg/mL = 1 g/L = 0.01 mol/L = 1e7 nmol/L = 1e7 nM
        assert convert_mass_to_nM(1, "mg/mL", molecular_weight=100) == pytest.approx(
            1e7
        )

    def test_ug_per_ml(self):
        # MW=100; 1 µg/mL = 1e-3 g/L → 1e-5 mol/L → 1e4 nM
        assert convert_mass_to_nM(1, "µg/mL", molecular_weight=100) == pytest.approx(
            1e4
        )

    def test_ng_per_ml(self):
        # MW=100; 1 ng/mL = 1e-6 g/L → 1e-8 mol/L → 10 nM
        assert convert_mass_to_nM(1, "ng/mL", molecular_weight=100) == pytest.approx(
            10
        )

    def test_pg_per_ml(self):
        # MW=100; 1 pg/mL = 1e-9 g/L → 1e-11 mol/L → 0.01 nM
        assert convert_mass_to_nM(1, "pg/mL", molecular_weight=100) == pytest.approx(
            0.01
        )

    @pytest.mark.parametrize("mw", [0, -1, None, 0.0])
    def test_invalid_mw_returns_none(self, mw):
        assert convert_mass_to_nM(1, "mg/mL", molecular_weight=mw) is None

    def test_unknown_unit_returns_none(self):
        assert convert_mass_to_nM(1, "kg/L", molecular_weight=100) is None
        assert convert_mass_to_nM(1, "nM", molecular_weight=100) is None


# ---------------------------------------------------------------------------
# convert_to_nM
# ---------------------------------------------------------------------------
class TestConvertToNM:
    @pytest.mark.parametrize(
        "value,unit,expected",
        [
            (100, "nM", 100),
            (1, "uM", 1000),
            (1, "µM", 1000),
            (1, "mM", 1_000_000),
            (1, "M", 1_000_000_000),
            (10, "pM", 0.01),
        ],
    )
    def test_molar_units(self, value, unit, expected):
        assert convert_to_nM(value, unit) == pytest.approx(expected)

    def test_logarithmic_pIC50(self):
        # pIC50 = 9 → IC50 = 10^-9 M = 1 nM
        assert convert_to_nM(9, "any-unit-ignored", is_logarithmic=True) == pytest.approx(1)
        # pIC50 = 6 → IC50 = 10^-6 M = 1000 nM
        assert convert_to_nM(6, "any-unit-ignored", is_logarithmic=True) == pytest.approx(
            1000
        )

    def test_per_ml_unit_with_multiplier(self):
        # nmol/mL → ("nM", 1000); convert_to_nM(1) → 1 * 1 * 1000 = 1000
        assert convert_to_nM(1, "nmol/mL") == pytest.approx(1000)

    def test_mass_unit_requires_mw(self):
        # MW=100; 1 mg/mL → 1e7 nM
        assert convert_to_nM(1, "mg/mL", molecular_weight=100) == pytest.approx(1e7)
        # without MW → None
        assert convert_to_nM(1, "mg/mL") is None

    def test_percent_returned_as_is(self):
        assert convert_to_nM(50, "%") == 50

    def test_unknown_unit_returns_none(self):
        assert convert_to_nM(1, "kg") is None
        assert convert_to_nM(1, "") is None


# ---------------------------------------------------------------------------
# process_row
# ---------------------------------------------------------------------------
class TestProcessRow:
    def test_simple_ic50_in_nM(self):
        row = {"binding_metric": "IC50", "value": "100", "unit": "nM"}
        result = process_row(row)
        assert isinstance(result, dict)
        assert result["binding_metric"] == "IC50"
        assert result["value"] == 100
        assert result["unit"] == "nM"
        assert result["relation"] == "="
        assert result["original_value"] == "100"
        assert result["original_unit"] == "nM"
        assert result["original_metric"] == "IC50"
        assert result["is_logarithmic"] is False

    def test_uM_to_nM_conversion(self):
        row = {"binding_metric": "Ki", "value": "5", "unit": "µM"}
        result = process_row(row)
        assert result["value"] == pytest.approx(5000)
        assert result["binding_metric"] == "Ki"
        assert result["unit"] == "uM"  # normalized form

    def test_pIC50_logarithmic(self):
        row = {"binding_metric": "pIC50", "value": "9", "unit": "nM"}
        result = process_row(row)
        assert result["binding_metric"] == "IC50"
        assert result["is_logarithmic"] is True
        assert result["value"] == pytest.approx(1)  # 10^-9 M = 1 nM

    def test_inhibition_kept_as_percent(self):
        row = {"binding_metric": "Inhibition", "value": "75", "unit": "%"}
        result = process_row(row)
        assert result["binding_metric"] == "Inhibition"
        assert result["value"] == 75
        assert result.get("is_percentage") is True

    def test_mass_unit_defers_conversion(self):
        row = {"binding_metric": "IC50", "value": "1", "unit": "mg/mL"}
        result = process_row(row)
        # No MW available — flag for later, keep raw value
        assert result.get("needs_mw_conversion") is True
        assert result["value"] == 1

    def test_default_unit_when_missing(self):
        row = {"binding_metric": "IC50", "value": "100"}
        result = process_row(row)
        assert result["original_unit"] == "nM"  # defaulted
        assert result["value"] == 100

    def test_invalid_value_returns_error_string(self):
        row = {"binding_metric": "IC50", "value": "not a number", "unit": "nM"}
        result = process_row(row)
        assert isinstance(result, str)
        assert "Wrong value" in result

    def test_invalid_metric_returns_error_string(self):
        row = {"binding_metric": "Hill slope", "value": "1.5", "unit": "nM"}
        result = process_row(row)
        assert isinstance(result, str)
        assert "Wrong metric" in result

    def test_unconvertible_unit_returns_error_string(self):
        row = {"binding_metric": "IC50", "value": "100", "unit": "kg"}
        result = process_row(row)
        assert isinstance(result, str)
        assert "Cannot convert" in result

    def test_relation_preserved(self):
        row = {"binding_metric": "IC50", "value": "<10", "unit": "nM"}
        result = process_row(row)
        assert result["relation"] == "<"
        assert result["value"] == 10

    def test_range_value(self):
        row = {"binding_metric": "IC50", "value": "10 to 20", "unit": "nM"}
        result = process_row(row)
        assert result["relation"] == "range"
        assert result["value"] == pytest.approx(15)
        assert result["original_range"] != ""

    def test_non_dict_input(self):
        assert process_row("not a dict") is None
        assert process_row(None) is None

    # --- CLASS-1 patent reproductions ---

    def test_sci_with_relation_mol_per_l(self):
        # US20170101391A1 Example 197
        result = process_row(
            {"binding_metric": "IC50", "value": ">2.00E-5", "unit": "mol/l"}
        )
        assert isinstance(result, dict)
        assert result["value"] == pytest.approx(20_000.0)
        assert result["relation"] == ">"

    def test_sci_with_relation_molar(self):
        # US20200199083A1 Example 5
        result = process_row(
            {"binding_metric": "IC50", "value": ">2.00E−5", "unit": "M"}
        )
        assert isinstance(result, dict)
        assert result["value"] == pytest.approx(20_000.0)
        assert result["relation"] == ">"

    def test_sci_range_in_molar(self):
        # US20160145297A1 A3 ASSAY_001
        result = process_row(
            {
                "binding_metric": "IC50",
                "value": "1.0E-08 to 1.0E-10",
                "unit": "M",
            }
        )
        assert isinstance(result, dict)
        assert result["value"] == pytest.approx(5.05)
        assert result["relation"] == "range"

    def test_sandwich_mixed_units(self):
        # US20240066027A1 Example 113
        result = process_row(
            {"binding_metric": "IC50", "value": "20 nM<IC50<10", "unit": "uM"}
        )
        assert isinstance(result, dict)
        assert result["value"] == pytest.approx(5010.0)
        assert result["relation"] == "range"
        assert result["original_range"].endswith(" nM")

    def test_slash_multivalue_rejected_in_process_row(self):
        # US20210355104A1 Example 376
        result = process_row(
            {
                "binding_metric": "IC50",
                "value": "133/209/375/142/48",
                "unit": "nM",
            }
        )
        assert isinstance(result, str)
        assert "Wrong value" in result

    def test_unit_bearing_range_no_double_convert(self):
        result = process_row(
            {"binding_metric": "IC50", "value": "0.1 µM to 1 µM", "unit": "uM"}
        )
        assert isinstance(result, dict)
        assert result["value"] == pytest.approx(550.0)
        assert result["original_range"].endswith(" nM")

    def test_rejects_values_above_1mM(self):
        result = process_row(
            {"binding_metric": "IC50", "value": ">2005", "unit": "mM"}
        )
        assert isinstance(result, str)
        assert "Impossible value" in result

    def test_single_word_adhesion_metric_is_rejected_not_raised(self):
        """"Readhesion" has no second token to take the metric from.

        Indexing blindly raised IndexError, and because the export catches per
        patent rather than per row, one unusable row failed the whole patent.
        """
        result = process_row(
            {"binding_metric": "Readhesion", "value": "100", "unit": "nM"}
        )
        assert isinstance(result, str)
        assert "Wrong metric" in result

    def test_adhesion_metric_still_reads_the_second_token(self):
        # The branch assumes "<adhesion-word> <metric>", so the metric is the
        # second token; anything else was already rejected before this fix.
        result = process_row(
            {"binding_metric": "adhesion IC50", "value": "100", "unit": "nM"}
        )
        assert isinstance(result, dict)
        assert result["binding_metric"] == "IC50"
        assert result["assay_context"] == "adhesion"
