"""Tests for XMLPatentParser entity handling."""

import pytest
from lxml import etree

from patent_processor.xml_patent_parser import (
    XMLPatentParser, _replace_entities, _SGML_ENTITIES, _PATENT_ENTITIES,
)


class TestReplaceEntities:
    def test_lsqb_rsqb(self):
        assert _replace_entities("&lsqb;0001&rsqb;") == "[0001]"

    def test_greek_letters(self):
        assert _replace_entities("&agr;-subunit") == "α-subunit"
        assert _replace_entities("&bgr;-sheet") == "β-sheet"
        assert _replace_entities("&mgr;M") == "μM"

    def test_math_symbols(self):
        result = _replace_entities("5 &times; 10")
        assert "×" in result
        assert _replace_entities("a &plus; b") == "a + b"

    def test_standard_entities_untouched(self):
        assert _replace_entities("&lt;tag&gt;") == "&lt;tag&gt;"
        assert _replace_entities("&amp;") == "&amp;"

    def test_standard_entity_case_variants_untouched(self):
        assert _replace_entities("&AMP;") == "&AMP;"
        assert _replace_entities("&LT;") == "&LT;"

    def test_no_entities(self):
        text = "plain text without entities"
        assert _replace_entities(text) == text

    def test_multiple_mixed(self):
        text = "&lsqb;0001&rsqb; &mgr;g/mL &mdash; see &agr;&bgr;"
        result = _replace_entities(text)
        assert "[0001]" in result
        assert "μ" in result
        assert "—" in result

    def test_html_entities(self):
        assert "′" in _replace_entities("5&prime;-end")
        assert "±" in _replace_entities("&plusmn;0.5")
        assert " " in _replace_entities("a&emsp;b")
        assert "™" in _replace_entities("&trade;")

    def test_sgml_greek_lowercase(self):
        assert "ε" in _replace_entities("&egr;")
        assert "λ" in _replace_entities("&lgr;")
        assert "π" in _replace_entities("&pgr;")
        assert "σ" in _replace_entities("&sgr;")

    def test_sgml_greek_uppercase(self):
        assert "Δ" in _replace_entities("&Dgr;")
        assert "Ω" in _replace_entities("&OHgr;")
        assert "Ψ" in _replace_entities("&PSgr;")

    def test_unknown_entity_left_in_place(self):
        assert _replace_entities("&completelyunknown;") == "&completelyunknown;"


class TestXMLPatentParser:
    def test_valid_xml_unchanged(self):
        xml = "<root><child>text</child></root>"
        root, err = XMLPatentParser.parse(xml)
        assert root is not None
        assert err is None
        assert root.tag == "root"

    def test_entity_lsqb_rsqb(self):
        xml = "<doc><p>&lsqb;0001&rsqb; Example text</p></doc>"
        root, err = XMLPatentParser.parse(xml)
        assert root is not None
        assert err is None
        text = root.find(".//p").text
        assert text.startswith("[0001]")

    def test_entity_greek(self):
        xml = "<doc><p>&agr;-receptor &bgr;-sheet &mgr;M</p></doc>"
        root, err = XMLPatentParser.parse(xml)
        assert root is not None
        assert err is None

    def test_entity_math(self):
        xml = "<doc><p>5 &times; 10 &plus; 2</p></doc>"
        root, err = XMLPatentParser.parse(xml)
        assert root is not None
        assert err is None

    def test_mixed_standard_and_patent_entities(self):
        xml = "<doc><p>&lsqb;0001&rsqb; a &lt; b &amp; &mgr;g</p></doc>"
        root, err = XMLPatentParser.parse(xml)
        assert root is not None
        assert err is None
        text = root.find(".//p").text
        assert "[0001]" in text

    def test_all_sgml_entities_parseable(self):
        """Every SGML entity in _SGML_ENTITIES produces a parseable document."""
        for name in _SGML_ENTITIES:
            xml = f"<doc><p>&{name};</p></doc>"
            root, err = XMLPatentParser.parse(xml)
            assert root is not None, f"Entity &{name}; failed: {err}"

    def test_common_html_entities_parseable(self):
        """Common HTML entities found in patents parse correctly."""
        common = [
            "prime", "Prime", "plusmn", "emsp", "ensp", "thinsp",
            "trade", "sect", "frac12", "frac14", "frac34",
            "rarr", "larr", "darr", "uarr", "equiv", "ne",
            "lsquo", "rsquo", "bull", "num", "excl", "quest",
            "commat", "eacute", "uuml", "ouml", "auml", "ccedil",
            "copy", "infin", "nabla", "forall", "part", "empty",
        ]
        for name in common:
            xml = f"<doc><p>&{name};</p></doc>"
            root, err = XMLPatentParser.parse(xml)
            assert root is not None, f"Entity &{name}; failed: {err}"

    def test_truly_broken_xml_returns_error(self):
        root, err = XMLPatentParser.parse("<doc><unclosed>")
        assert root is None
        assert "XML Syntax Error" in err

    def test_empty_string(self):
        root, err = XMLPatentParser.parse("")
        assert root is None

    def test_none_input(self):
        root, err = XMLPatentParser.parse(None)
        assert root is None

    def test_unknown_entity_still_fails(self):
        xml = "<doc>&completelyunknown;</doc>"
        root, err = XMLPatentParser.parse(xml)
        assert root is None
        assert "XML Syntax Error" in err
