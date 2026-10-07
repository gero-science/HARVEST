"""Tests for extract_relevant_xml_sections with old and new USPTO XML schemas."""

from lxml import etree

from bioactivity_extraction.xml_sections import extract_relevant_xml_sections

MODERN_XML = """\
<us-patent-application>
  <us-bibliographic-data-application>
    <invention-title>KINASE INHIBITORS</invention-title>
  </us-bibliographic-data-application>
  <abstract><p>A composition.</p></abstract>
  <description><p id="P-0001">Detailed description here.</p></description>
</us-patent-application>
"""

PRE2005_XML = """\
<patent-application-publication>
  <subdoc-bibliographic-information>
    <title-of-invention>DIAZA COMPOUNDS</title-of-invention>
  </subdoc-bibliographic-information>
  <subdoc-abstract><paragraph>An abstract.</paragraph></subdoc-abstract>
  <subdoc-description>
    <paragraph id="P-0001">Background of the invention.</paragraph>
    <paragraph id="P-0002">More details.</paragraph>
  </subdoc-description>
</patent-application-publication>
"""

MINIMAL_XML = "<root><other>no description</other></root>"


class TestModernSchema:
    def test_extracts_all_sections(self):
        root = etree.fromstring(MODERN_XML.encode())
        result = extract_relevant_xml_sections(root)
        assert "KINASE INHIBITORS" in result
        assert "A composition" in result
        assert "Detailed description" in result

    def test_length_nonzero(self):
        root = etree.fromstring(MODERN_XML.encode())
        result = extract_relevant_xml_sections(root)
        assert len(result) > 0


class TestPre2005Schema:
    def test_extracts_title(self):
        root = etree.fromstring(PRE2005_XML.encode())
        result = extract_relevant_xml_sections(root)
        assert "DIAZA COMPOUNDS" in result

    def test_extracts_abstract(self):
        root = etree.fromstring(PRE2005_XML.encode())
        result = extract_relevant_xml_sections(root)
        assert "An abstract" in result

    def test_extracts_description(self):
        root = etree.fromstring(PRE2005_XML.encode())
        result = extract_relevant_xml_sections(root)
        assert "Background of the invention" in result

    def test_length_nonzero(self):
        root = etree.fromstring(PRE2005_XML.encode())
        result = extract_relevant_xml_sections(root)
        assert len(result) > 0


class TestEdgeCases:
    def test_no_matching_sections(self):
        root = etree.fromstring(MINIMAL_XML.encode())
        result = extract_relevant_xml_sections(root)
        assert result == ""

    def test_logger_reports_sections(self):
        calls = []

        class FakeLogger:
            def info(self, msg):
                calls.append(msg)

        root = etree.fromstring(PRE2005_XML.encode())
        extract_relevant_xml_sections(root, logger=FakeLogger())
        assert len(calls) == 1
        assert "title=Yes" in calls[0]
        assert "abstract=Yes" in calls[0]
        assert "description=Yes" in calls[0]
