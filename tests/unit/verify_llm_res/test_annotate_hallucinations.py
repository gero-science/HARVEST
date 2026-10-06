"""Tests for verify_llm_res annotation and hallucination_detector
Unicode normalization.
"""

import pytest

from verify_llm_res.annotate import patent_stats_to_hallu_json
from verify_llm_res.hallucination_detector import (
    _build_table_column_texts,
    find_iupac_in_xml,
    normalize_iupac_for_search,
    normalize_unicode_text,
)


class TestPatentStatsToHalluJson:
    def test_none_stats_returns_clean_sidecar(self):
        """No stats (ZIP missing, no TSV) → empty sidecar."""
        result = patent_stats_to_hallu_json(None)
        assert result["detector_version"] == "1.0"
        assert result["flagged_compounds"] == {}
        assert result["flagged_iupac"] == []
        assert result["flagged_values"] == []

    def test_clean_patent_returns_empty_flags(self):
        """Patent with 0 hallucinations → empty flags."""
        stats = {
            "patent_id": "US12345A1",
            "total": 10,
            "hallucinations": 0,
            "types": {"chem_id": 0, "tif": 0, "iupac": 0, "mismatch": 0},
            "hallucination_details": {
                "chem_id": [],
                "tif": [],
                "iupac": [],
                "mismatch": [],
            },
            "stage2": {
                "total": 5,
                "hallucinations": 0,
                "types": {"value": 0},
                "hallucination_details": {"value": []},
            },
        }
        result = patent_stats_to_hallu_json(stats)
        assert result["flagged_compounds"] == {}
        assert result["flagged_iupac"] == []
        assert result["flagged_values"] == []

    def test_chem_id_flag(self):
        stats = _make_stats(chem_id=[{"chem_id": "CHEM-US-00001", "compound": "c1"}])
        result = patent_stats_to_hallu_json(stats)
        assert "CHEM-US-00001" in result["flagged_compounds"]
        assert "chem_id" in result["flagged_compounds"]["CHEM-US-00001"]

    def test_mismatch_flag(self):
        stats = _make_stats(mismatch=[{
            "chem_id": "CHEM-US-00002",
            "chem_number": 2,
            "tif": "FILE.TIF",
            "tif_number": 99,
            "compound": "c2",
        }])
        result = patent_stats_to_hallu_json(stats)
        assert "CHEM-US-00002" in result["flagged_compounds"]
        assert "mismatch" in result["flagged_compounds"]["CHEM-US-00002"]

    def test_iupac_flag(self):
        stats = _make_stats(iupac=[{"iupac": "fake-iupac-name", "compound": "c3"}])
        result = patent_stats_to_hallu_json(stats)
        assert "fake-iupac-name" in result["flagged_iupac"]

    def test_tif_flag_goes_to_flagged_iupac(self):
        """TIF filenames stored in compound_IUPAC_name → flagged_iupac."""
        stats = _make_stats(tif=[{"tif": "US12345A1-C00099.TIF", "compound": "c4"}])
        result = patent_stats_to_hallu_json(stats)
        assert "US12345A1-C00099.TIF" in result["flagged_iupac"]

    def test_stage2_value_flag(self):
        stats = _make_stats(s2_value=[{
            "value": "999",
            "compound": "c5",
            "binding_metric": "IC50",
            "reasoning": "Table 1",
        }])
        result = patent_stats_to_hallu_json(stats)
        assert len(result["flagged_values"]) == 1
        assert result["flagged_values"][0]["value"] == "999"
        assert result["flagged_values"][0]["compound"] == "c5"

    def test_combined_flags(self):
        """Multiple flag types on the same patent."""
        stats = _make_stats(
            chem_id=[{"chem_id": "CHEM-US-00001", "compound": "c1"}],
            iupac=[{"iupac": "fake-iupac", "compound": "c2"}],
            mismatch=[{
                "chem_id": "CHEM-US-00001",
                "chem_number": 1,
                "tif": "FILE.TIF",
                "tif_number": 99,
                "compound": "c1",
            }],
        )
        result = patent_stats_to_hallu_json(stats)
        # CHEM-US-00001 should have both chem_id and mismatch
        assert set(result["flagged_compounds"]["CHEM-US-00001"]) == {
            "chem_id",
            "mismatch",
        }
        assert "fake-iupac" in result["flagged_iupac"]

    def test_no_duplicate_iupac(self):
        """Same IUPAC flagged twice should appear once."""
        stats = _make_stats(iupac=[
            {"iupac": "same-name", "compound": "c1"},
            {"iupac": "same-name", "compound": "c2"},
        ])
        result = patent_stats_to_hallu_json(stats)
        assert result["flagged_iupac"].count("same-name") == 1


class TestNormalizeUnicodeText:
    """Tests for normalize_unicode_text() — category-based Unicode→ASCII."""

    # --- Dashes (Pd category + MINUS/HYPHEN name) ---

    def test_em_dash_to_hyphen(self):
        assert normalize_unicode_text("(S)—N-methyl") == "(S)-N-methyl"

    def test_en_dash_to_hyphen(self):
        assert normalize_unicode_text("1–2") == "1-2"

    def test_minus_sign_to_hyphen(self):
        assert normalize_unicode_text("value−offset") == "value-offset"

    def test_non_breaking_hyphen(self):
        assert normalize_unicode_text("non‑breaking") == "non-breaking"

    def test_unicode_hyphen(self):
        assert normalize_unicode_text("a‐b") == "a-b"

    # --- Spaces (Zs category) ---

    def test_nbsp_to_space(self):
        assert normalize_unicode_text("hello world") == "hello world"

    def test_en_space_to_space(self):
        assert normalize_unicode_text("hello world") == "hello world"

    def test_thin_space_to_space(self):
        assert normalize_unicode_text("hello world") == "hello world"

    # --- Format characters (Cf category) ---

    def test_zero_width_space_removed(self):
        assert normalize_unicode_text("hel​lo") == "hello"

    def test_soft_hyphen_to_hyphen(self):
        """Soft hyphen (U+00AD) has HYPHEN in name → maps to '-'."""
        assert normalize_unicode_text("hel­lo") == "hel-lo"

    # --- Primes (Po category + PRIME in name) ---

    def test_prime_to_apostrophe(self):
        """U+2032 PRIME → apostrophe (chemistry: 3\', 5\'-positions)."""
        assert normalize_unicode_text("[1,1′-BIPHENYL]") == "[1,1\'-BIPHENYL]"

    def test_double_prime(self):
        """U+2033 DOUBLE PRIME → apostrophe(s) via NFKC decomposition."""
        result = normalize_unicode_text("5″")
        # NFKC decomposes ″ to two ′, each mapped to apostrophe
        assert result == "5\'\'",  f"got {result!r}"

    # --- Quotes (Pi/Pf categories) ---

    def test_smart_double_quotes(self):
        assert normalize_unicode_text("“text”") == '"text"'

    def test_smart_single_quotes(self):
        assert normalize_unicode_text("‘text’") == "\'text\'"

    # --- Preservation ---

    def test_plain_ascii_unchanged(self):
        text = "(S)-N-(1-OXO-3-PHENYLPROPAN-2-YL)"
        assert normalize_unicode_text(text) == text

    def test_greek_letters_expanded(self):
        """Greek letters are spelled out for IUPAC matching."""
        assert normalize_unicode_text("α-methyl") == "alpha-methyl"
        assert normalize_unicode_text("β-sheet") == "beta-sheet"
        assert normalize_unicode_text("γ-amino") == "gamma-amino"
        assert normalize_unicode_text("δ ppm") == "delta ppm"

    def test_degree_sign_preserved(self):
        assert "°" in normalize_unicode_text("25°C")

    # --- Combined ---

    def test_combined(self):
        """Multiple Unicode chars in one string."""
        text = "(S)—N–(1−OXO thing)"
        assert normalize_unicode_text(text) == "(S)-N-(1-OXO thing)"


class TestFindIupacEmDashBug:
    """Regression tests for the em dash false-positive bug.

    Patent XML encodes em dashes as &#x2014; in IUPAC headings.  After
    html.unescape() they become U+2014 '—', which did NOT match the LLM's
    standard hyphen '-' (U+002D), causing false-positive hallucination flags.
    """

    IUPAC = "(S)-N-(1-OXO-3-PHENYLPROPAN-2-YL)-1-PHENYL-1H-IMIDAZOLE-5-CARBOXAMIDE"

    def _wrap_xml(self, heading_text):
        return f'<patent><heading id="h-0026">{heading_text} (1)</heading></patent>'

    def test_exact_hyphen_match(self):
        """Baseline: standard hyphen in XML matches."""
        xml = self._wrap_xml(self.IUPAC)
        assert find_iupac_in_xml(xml, self.IUPAC) is True

    def test_em_dash_entity_match(self):
        """The actual bug: &#x2014; in XML should match hyphen in IUPAC."""
        xml_iupac = self.IUPAC.replace("(S)-N", "(S)&#x2014;N")
        xml = self._wrap_xml(xml_iupac)
        assert find_iupac_in_xml(xml, self.IUPAC) is True

    def test_en_dash_entity_match(self):
        """En dash &#x2013; should also match."""
        xml_iupac = self.IUPAC.replace("(S)-N", "(S)&#x2013;N")
        xml = self._wrap_xml(xml_iupac)
        assert find_iupac_in_xml(xml, self.IUPAC) is True

    def test_minus_sign_entity_match(self):
        """Minus sign &#x2212; should also match."""
        xml_iupac = self.IUPAC.replace("(S)-N", "(S)&#x2212;N")
        xml = self._wrap_xml(xml_iupac)
        assert find_iupac_in_xml(xml, self.IUPAC) is True

    def test_prime_entity_match(self):
        """&#x2032; (prime) in XML should match apostrophe in IUPAC."""
        iupac = "(S)-1-([1,1'-BIPHENYL]-3-YL)-N-(4-AMINO-3,4-DIOXO-1-PHENYLBUTAN-2-YL)-3-METHYL-1H-PYRAZOLE-5-CARBOXAMIDE"
        xml_iupac = iupac.replace("1,1'-", "1,1&#x2032;-")
        xml = self._wrap_xml(xml_iupac)
        assert find_iupac_in_xml(xml, iupac) is True

    def test_genuine_hallucination_still_caught(self):
        """A truly absent IUPAC should still return False."""
        xml = self._wrap_xml("COMPLETELY-DIFFERENT-COMPOUND")
        assert find_iupac_in_xml(xml, self.IUPAC) is False


class TestOptionalAlphaHyphen:
    """Regression tests for optional hyphens between letter groups.

    IUPAC names allow stylistic variation in hyphenation between word parts:
    "methylamino-purin" vs "methylaminopurin", "cyclopentyl-amino" vs
    "cyclopentylamino", "tetrahydro-thiophene" vs "tetrahydrothiophene".
    The detector must match either form.
    """

    def _wrap_xml(self, text):
        return f'<patent><p>{text}</p></patent>'

    def test_hyphenated_llm_vs_joined_xml(self):
        """LLM writes hyphens, XML has joined form (US20050256143A1 pattern)."""
        llm = "(2S)-5-(2-chloro-6-methylamino-purin-9-yl)-3,4-dihydroxy-tetrahydro-thiophene-2-carboxylic acid methylamide"
        xml_text = "(2S)-5-(2-chloro-6-methylaminopurin-9-yl)-3,4-dihydroxytetrahydrothiophene-2-carboxylic acid methyl amide"
        assert find_iupac_in_xml(self._wrap_xml(xml_text), llm) is True

    def test_joined_llm_vs_hyphenated_xml(self):
        """LLM writes joined form, XML has hyphen (US20050192324A1 pattern)."""
        llm = "(2S)-1-{2-[(3S,1R)-3-(tetraazol-5-ylaminomethyl)cyclopentylamino]acetyl}-pyrrolidine"
        xml_text = "(2S)-1-{2-[(3S,1R)-3-(tetraazol-5-ylaminomethyl)cyclopentyl-amino]acetyl}-pyrrolidine"
        assert find_iupac_in_xml(self._wrap_xml(xml_text), llm) is True

    def test_structural_hyphens_still_required(self):
        """Hyphens between letter-digit (e.g. purin-9-yl) must still match."""
        llm = "purin-9-yl-methylamine"
        # Removing the '9' entirely should NOT match
        xml_text = "purin-8-yl-methylamine"
        assert find_iupac_in_xml(self._wrap_xml(xml_text), llm) is False


class TestGreekLetterAndCaretMatching:
    """Regression tests for Greek letter and caret-superscript false positives.

    Patent XML uses Greek letters (α, β) and <sup> markup.  LLMs spell out
    Greek letters ("alpha") and use caret notation ("N^6").  After XML tag
    stripping <sup>6</sup> becomes "6", but the LLM's "^6" kept the caret.

    Fixes: normalize_unicode_text() expands Greek letters to names;
    normalize_iupac_for_search() strips caret-before-digit.
    """

    def _wrap_xml(self, text):
        return f'<patent><p>{text}</p></patent>'

    def test_greek_alpha_in_xml_matches_spelled_out_llm(self):
        """XML α-methyl matches LLM alpha-methyl (US20100041041A1 pattern)."""
        llm = "5-(alpha-methyl-nitrobenzyl-oxymethyl)-2'-dUTP"
        xml = "5-(&#x3b1;-methyl-nitrobenzyl-oxymethyl)-2&#x2032;-dUTP"
        assert find_iupac_in_xml(self._wrap_xml(xml), llm) is True

    def test_caret_superscript_matches_sup_tag(self):
        """LLM N^6 matches XML N<sup>6</sup> (US20100041041A1 pattern)."""
        llm = "N^6-(2-nitrobenzyl)-2'-dATP"
        xml = "N<sup>6</sup>-(2-nitrobenzyl)-2&#x2032;-dATP"
        assert find_iupac_in_xml(self._wrap_xml(xml), llm) is True

    def test_combined_greek_and_caret(self):
        """Both Greek alpha and caret in one name."""
        llm = "O^6-(alpha-methyl-2-nitrobenzyl)-2'-dGTP"
        xml = "O<sup>6</sup>-(&#x3b1;-methyl-2-nitrobenzyl)-2&#x2032;-dGTP"
        assert find_iupac_in_xml(self._wrap_xml(xml), llm) is True

    def test_caret_stripped_in_normalize(self):
        """normalize_iupac_for_search strips ^ before digits."""
        assert normalize_iupac_for_search("N^6-methyl") == "N6-methyl"
        assert normalize_iupac_for_search("C^7-amino") == "C7-amino"
        # Caret NOT before digit should stay
        assert normalize_iupac_for_search("x^y") == "x^y"

    def test_genuine_hallucination_still_caught(self):
        """A truly absent name should still be flagged."""
        xml = "N<sup>6</sup>-(2-nitrobenzyl)-2&#x2032;-dATP"
        llm = "5-(alpha-isopropyl-2-chlorobenzyl-oxymethyl)-2'-dCTP"
        assert find_iupac_in_xml(self._wrap_xml(xml), llm) is False


class TestSelfClosingEntryBug:
    """Regression tests for self-closing <entry/> column misalignment.

    Patent tables use <entry/> (self-closing) for empty cells in continuation
    rows.  The old regex ``<entry[^>]*>(.*?)</entry>`` skipped these, shifting
    column indices so that IUPAC fragments split across rows never reunited.
    """

    IUPAC = (
        "[1-(2,3-Dichloro-4-fluoro-phenyl)-1H-tetrazol-5-yl]-"
        "{2-[4-(3,3,3-trifluoro-propyl)-[1,4]diazepan-1-yl]-"
        "pyridin-3-ylmethyl}-amine"
    )

    TABLE_XML = (
        "<tables>"
        "<row>"
        "<entry>Example 295</entry>"
        "<entry>[1-(2,3-Dichloro-4-fluoro-phenyl)-1H-</entry>"
        "<entry>533.3</entry>"
        "<entry>1.75</entry>"
        "</row>"
        "<row>"
        "<entry/>"
        "<entry>tetrazol-5-yl]-{2-[4-(3,3,3-</entry>"
        "</row>"
        "<row>"
        "<entry/>"
        "<entry>trifluoro-propyl)-[1,4]diazepan-1-yl]-</entry>"
        "</row>"
        "<row>"
        "<entry/>"
        "<entry>pyridin-3-ylmethyl}-amine</entry>"
        "</row>"
        "</tables>"
    )

    def test_column_texts_align_with_self_closing_entry(self):
        """<entry/> must be counted to keep column indices aligned."""
        cols = _build_table_column_texts(self.TABLE_XML)
        # Column 1 should contain the full concatenated IUPAC
        iupac_no_ws = self.IUPAC.replace(' ', '').lower()
        found = any(
            iupac_no_ws in c.replace(' ', '').lower() for c in cols
        )
        assert found, f"IUPAC not found in any column: {cols}"

    def test_find_iupac_in_xml_with_self_closing_entry(self):
        """find_iupac_in_xml should find IUPAC split across rows with <entry/>."""
        xml = f"<patent>{self.TABLE_XML}</patent>"
        assert find_iupac_in_xml(xml, self.IUPAC) is True

    def test_self_closing_with_attributes(self):
        """<entry colname='c1'/> should also be counted."""
        xml = (
            "<tables>"
            "<row><entry>Example-compound-label-one</entry>"
            "<entry>A-long-iupac-name-part-one</entry></row>"
            "<row><entry colname='c1'/>"
            "<entry>iupac-name-continued-here</entry></row>"
            "</tables>"
        )
        cols = _build_table_column_texts(xml)
        # "continued" should be in column 1, not column 0
        col1_texts = [c for c in cols if 'continued' in c.lower()]
        assert col1_texts, f"'continued' not found in expected column: {cols}"
        # The label should NOT be concatenated with the IUPAC continuation
        for c in col1_texts:
            assert 'compound-label' not in c


def _make_stats(
    *,
    chem_id=None,
    tif=None,
    iupac=None,
    mismatch=None,
    s2_value=None,
):
    """Build a minimal detector stats dict with the given hallucination details."""
    return {
        "patent_id": "US99999A1",
        "total": 10,
        "hallucinations": 0,
        "types": {"chem_id": 0, "tif": 0, "iupac": 0, "mismatch": 0},
        "hallucination_details": {
            "chem_id": chem_id or [],
            "tif": tif or [],
            "iupac": iupac or [],
            "mismatch": mismatch or [],
        },
        "stage2": {
            "total": 5,
            "hallucinations": 0,
            "types": {"value": 0},
            "hallucination_details": {"value": s2_value or []},
        },
    }
