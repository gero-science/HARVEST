"""Tests for _extract_relevant_xml_sections: title + abstract + description (bioactivity_extraction.xml_sections)."""

from lxml import etree


def _root(xml: str):
    return etree.fromstring(xml.encode("utf-8"))


def test_extract_includes_all_three_sections(make_agent):
    root = _root(
        "<patent>"
        "<invention-title>Foo Title</invention-title>"
        "<abstract><p>Bar Abstract</p></abstract>"
        "<description><p>Baz Description</p></description>"
        "<claims><claim>ignored</claim></claims>"
        "</patent>"
    )

    out = make_agent()._extract_relevant_xml_sections(root)

    assert "Foo Title" in out
    assert "Bar Abstract" in out
    assert "Baz Description" in out
    # sections outside title/abstract/description are excluded
    assert "ignored" not in out


def test_extract_omits_missing_sections(make_agent):
    root = _root("<patent><invention-title>Only Title</invention-title></patent>")

    out = make_agent()._extract_relevant_xml_sections(root)

    assert "Only Title" in out
    assert "abstract" not in out
    assert "description" not in out


def test_extract_empty_when_no_relevant_sections(make_agent):
    root = _root("<patent><claims><claim>x</claim></claims></patent>")

    assert make_agent()._extract_relevant_xml_sections(root) == ""
