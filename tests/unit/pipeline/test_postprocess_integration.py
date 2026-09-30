"""Tests for pipeline.postprocess_integration."""
import pytest

from pipeline.postprocess_integration import (
    apply_alias_mapping_to_stage2,
    create_fuzzy_alias_mapping,
    extract_compound_number,
    is_safe_fuzzy_match,
    normalize_compound_alias,
    similarity_ratio,
)


class TestNormalizeCompoundAlias:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Example 669", "example669"),
            ("Ex. 669", "example669"),
            ("Ex 669", "example669"),
            ("EXAMPLE 669", "example669"),
            ("Compound 5", "compound5"),
            ("Comp. 5", "compound5"),
            ("Cmpd 5", "compound5"),
            ("Structure 10", "structure10"),
            ("Struct. 10", "structure10"),
            ("Structure-10", "structure10"),
            ("Structure_10", "structure10"),
            ("Compound 5a", "compound5a"),
        ],
    )
    def test_normalizes_known_variants(self, raw, expected):
        assert normalize_compound_alias(raw) == expected

    def test_empty_returns_empty(self):
        assert normalize_compound_alias("") == ""
        assert normalize_compound_alias(None) == ""

    def test_preserves_distinct_compounds(self):
        # Different numbers must yield different normalized forms.
        assert normalize_compound_alias("Compound 5a") != normalize_compound_alias(
            "Compound 5b"
        )
        assert normalize_compound_alias("Example 1") != normalize_compound_alias(
            "Example 11"
        )


class TestExtractCompoundNumber:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("Example 669", "669"),
            ("Ex. 5a", "5a"),
            ("Compound 10", "10"),
            ("Structure-20b", "20b"),
            # leading zeros are stripped
            ("Compound 0042", "42"),
            # all-zero number falls back to "0"
            ("Compound 000", "0"),
            # letters are lowercased
            ("Compound 5A", "5a"),
        ],
    )
    def test_extracts_numbers(self, alias, expected):
        assert extract_compound_number(alias) == expected

    def test_no_number_returns_none(self):
        assert extract_compound_number("Example") is None
        assert extract_compound_number("foo bar baz") is None

    def test_empty_returns_none(self):
        assert extract_compound_number("") is None
        assert extract_compound_number(None) is None


class TestSimilarityRatio:
    def test_identical_strings(self):
        assert similarity_ratio("example1", "example1") == 1.0

    def test_totally_different_strings(self):
        assert similarity_ratio("aaaaa", "zzzzz") < 0.1

    def test_empty_strings(self):
        # SequenceMatcher returns 1.0 for two empty strings (vacuously identical).
        assert similarity_ratio("", "") == 1.0

    def test_case_insensitive(self):
        assert similarity_ratio("Example1", "EXAMPLE1") == 1.0


class TestIsSafeFuzzyMatch:
    def test_same_normalized_form_matches(self):
        assert is_safe_fuzzy_match("Example 1", "Ex. 1") is True
        assert is_safe_fuzzy_match("Compound 5", "Comp. 5") is True
        assert is_safe_fuzzy_match("Compound 5", "Cmpd 5") is True

    def test_different_numbers_do_not_match(self):
        # The most critical safety property: 1 vs 11 must NOT collapse.
        assert is_safe_fuzzy_match("Example 1", "Example 11") is False
        assert is_safe_fuzzy_match("Compound 5a", "Compound 5b") is False
        assert is_safe_fuzzy_match("Compound 10", "Compound 100") is False

    def test_different_types_do_not_match(self):
        # Same number but different "type" prefix → distinct compounds.
        assert is_safe_fuzzy_match("Compound 5a", "Example 5a") is False
        assert is_safe_fuzzy_match("Structure 10", "Compound 10") is False

    def test_bare_number_matches_typed(self):
        # "5a" alone equals "Compound 5a" (special case).
        assert is_safe_fuzzy_match("5a", "Compound 5a") is True
        assert is_safe_fuzzy_match("Compound 5a", "5a") is True
        assert is_safe_fuzzy_match("669", "Example 669") is True

    def test_bare_numbers_with_different_values(self):
        assert is_safe_fuzzy_match("5a", "5b") is False
        assert is_safe_fuzzy_match("1", "11") is False

    def test_empty_inputs(self):
        assert is_safe_fuzzy_match("", "Example 1") is False
        assert is_safe_fuzzy_match("Example 1", "") is False
        assert is_safe_fuzzy_match("", "") is False
        assert is_safe_fuzzy_match(None, "Example 1") is False

    def test_no_extractable_number_when_required(self):
        # Two text-only aliases without numbers — refused (unsafe).
        assert is_safe_fuzzy_match("foo", "foo bar") is False


class TestCreateFuzzyAliasMapping:
    def test_empty_inputs_return_empty_mapping(self):
        assert create_fuzzy_alias_mapping([], []) == {}
        assert create_fuzzy_alias_mapping([{"compound": "x"}], []) == {}
        assert create_fuzzy_alias_mapping([], [{"compound": "x"}]) == {}

    def test_exact_match_skipped_no_mapping_needed(self):
        s2 = [{"compound": "Example 1"}]
        s3 = [{"compound": "Example 1"}]
        assert create_fuzzy_alias_mapping(s2, s3) == {}

    def test_maps_variant_writings(self):
        # "Ex. 1" in stage 2 → "Example 1" in stage 3.
        s2 = [{"compound": "Ex. 1"}, {"compound": "Cmpd 5"}]
        s3 = [{"compound": "Example 1"}, {"compound": "Compound 5"}]
        mapping = create_fuzzy_alias_mapping(s2, s3)
        assert mapping == {"Ex. 1": "Example 1", "Cmpd 5": "Compound 5"}

    def test_does_not_map_distinct_numbers(self):
        # Critical safety: stage 2 "Example 1" must NOT map to "Example 11".
        s2 = [{"compound": "Example 1"}]
        s3 = [{"compound": "Example 11"}]
        assert create_fuzzy_alias_mapping(s2, s3) == {}

    def test_skips_empty_aliases(self):
        s2 = [{"compound": ""}, {"compound": None}, {"compound": "Ex. 1"}]
        s3 = [{"compound": "Example 1"}]
        mapping = create_fuzzy_alias_mapping(s2, s3)
        assert mapping == {"Ex. 1": "Example 1"}

    def test_picks_best_when_multiple_safe_candidates(self):
        # Both "Example 5" and "Compound 5" are theoretically reachable from
        # "5", but bare-number rule allows match to either. Best similarity
        # picks the closest after normalization. Both normalize differently
        # from "5"; the closer one wins.
        s2 = [{"compound": "5"}]
        s3 = [{"compound": "Example 5"}, {"compound": "Compound 5"}]
        mapping = create_fuzzy_alias_mapping(s2, s3)
        # We assert mapping happens; either choice is acceptable for the
        # safety contract — what matters is it doesn't silently drop.
        assert "5" in mapping
        assert mapping["5"] in {"Example 5", "Compound 5"}


class TestApplyAliasMappingToStage2:
    def test_empty_mapping_returns_input_unchanged(self):
        s2 = [{"compound": "Ex. 1", "value": 10}]
        result = apply_alias_mapping_to_stage2(s2, {})
        assert result is s2  # docstring says: returns the input list itself

    def test_replaces_aliases(self):
        s2 = [
            {"compound": "Ex. 1", "value": 10},
            {"compound": "Cmpd 5", "value": 20},
            {"compound": "Compound 99", "value": 30},
        ]
        mapping = {"Ex. 1": "Example 1", "Cmpd 5": "Compound 5"}
        result = apply_alias_mapping_to_stage2(s2, mapping)
        assert [r["compound"] for r in result] == [
            "Example 1",
            "Compound 5",
            "Compound 99",
        ]

    def test_does_not_mutate_original(self):
        s2 = [{"compound": "Ex. 1", "value": 10}]
        mapping = {"Ex. 1": "Example 1"}
        result = apply_alias_mapping_to_stage2(s2, mapping)
        # Original record is untouched (function copies before update).
        assert s2[0]["compound"] == "Ex. 1"
        assert result[0]["compound"] == "Example 1"

    def test_preserves_extra_fields(self):
        s2 = [{"compound": "Ex. 1", "value": 10, "unit": "nM", "metric": "IC50"}]
        result = apply_alias_mapping_to_stage2(s2, {"Ex. 1": "Example 1"})
        assert result[0]["value"] == 10
        assert result[0]["unit"] == "nM"
        assert result[0]["metric"] == "IC50"
