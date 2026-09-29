"""Tests for BioactivityExtractor alias helpers.

The extractor delegates to bioactivity_extraction.compound_alias.
pipeline/postprocess_integration.py has its own implementation of
normalize_compound_alias / extract_compound_number, tested separately.
"""

import pytest


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("", ""),
        ("Example 669", "ex669"),
        ("Ex. 669", "ex669"),
        ("Ex 669", "ex669"),
        ("Compound 5", "cmpd5"),
        ("Comp. 5", "cmpd5"),
        ("Cmpd 5", "cmpd5"),
        ("Structure 10", "struct10"),
    ],
)
def test_normalize_compound_alias(make_agent, alias, expected):
    assert make_agent()._normalize_compound_alias(alias) == expected


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("", ""),
        ("Example 669", "669"),
        ("Ex. 669", "669"),
        ("Compound 5", "5"),
        ("5", "5"),
        ("Comp. 5a", "5a"),
        ("Ex. 00005", "5"),
        ("Ex. (5)", "5"),
        ("Compound [10]", "10"),
        ("Structure-10", "10"),
        ("no number here", ""),
    ],
)
def test_extract_compound_number(make_agent, alias, expected):
    assert make_agent()._extract_compound_number(alias) == expected
