"""
Basic XML parser for patent documents.
Handles safe conversion of an XML string into an lxml object.
"""

import html.entities
import re
from typing import Optional, Tuple
from lxml import etree

# ISO 8879 / SGML entities used in older USPTO patent XML that are absent
# from both the XML built-in set and Python's html.entities tables.
_SGML_ENTITIES: dict[str, str] = {
    # ISO 8879 Greek lowercase
    "agr": "α", "bgr": "β", "ggr": "γ", "dgr": "δ",
    "egr": "ε", "zgr": "ζ", "eegr": "η", "thgr": "θ",
    "igr": "ι", "kgr": "κ", "lgr": "λ", "mgr": "μ",
    "ngr": "ν", "xgr": "ξ", "ogr": "ο", "pgr": "π",
    "rgr": "ρ", "sgr": "σ", "tgr": "τ", "ugr": "υ",
    "phgr": "φ", "khgr": "χ", "psgr": "ψ", "ohgr": "ω",
    # ISO 8879 Greek uppercase
    "Agr": "Α", "Bgr": "Β", "Ggr": "Γ", "Dgr": "Δ",
    "Egr": "Ε", "Zgr": "Ζ", "EEgr": "Η", "THgr": "Θ",
    "Igr": "Ι", "Kgr": "Κ", "Lgr": "Λ", "Mgr": "Μ",
    "Ngr": "Ν", "Xgr": "Ξ", "Ogr": "Ο", "Pgr": "Π",
    "Rgr": "Ρ", "Sgr": "Σ", "Tgr": "Τ", "Ugr": "Υ",
    "PHgr": "Φ", "KHgr": "Χ", "PSgr": "Ψ", "OHgr": "Ω",
    # USPTO-specific / rare SGML
    "Circlesolid": "●", "CenterDot": "·",
    "Brketopenst": "⎡", "Brketclosest": "⎤",
    "Parenopenst": "(", "Parenclosest": ")",
    "LeftBracketingBar": "│", "RightBracketingBar": "│",
    "AutoLeftMatch": "", "AutoRightMatch": "",
    "Asteriskpseud": "*",
    "angst": "Å",
    "lsqb": "[", "rsqb": "]", "lcub": "{", "rcub": "}",
    "plus": "+", "minus": "−", "equals": "=",
    "boxH": "═",
}

# Build a unified lookup: XML builtins are excluded so lxml handles them.
_XML_BUILTINS = frozenset({"lt", "gt", "amp", "quot", "apos"})


def _build_entity_map() -> dict[str, str]:
    m: dict[str, str] = {}
    skip = {b.lower() for b in _XML_BUILTINS}
    for name, cp in html.entities.name2codepoint.items():
        if name.lower() not in skip:
            m[name] = chr(cp)
    for name, char in html.entities.html5.items():
        bare = name.rstrip(";")
        if bare.lower() not in skip and bare not in m:
            m[bare] = char
    m.update(_SGML_ENTITIES)
    return m


_PATENT_ENTITIES: dict[str, str] = _build_entity_map()

_ENTITY_RE = re.compile(r"&([a-zA-Z][a-zA-Z0-9]*);")


_XML_BUILTINS_LOWER = {b.lower() for b in _XML_BUILTINS}


def _replace_entities(xml_string: str) -> str:
    """Replace non-standard entities with their Unicode characters."""
    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name.lower() in _XML_BUILTINS_LOWER:
            return m.group(0)
        return _PATENT_ENTITIES.get(name, m.group(0))
    return _ENTITY_RE.sub(_sub, xml_string)


class XMLPatentParser:
    """
    Parses an XML string and returns an lxml object.
    Intended for working with HTML descriptions from Parquet sources.
    """

    @staticmethod
    def parse(xml_string: str) -> Tuple[Optional[etree._Element], Optional[str]]:
        """
        Parses XML from a string.

        Args:
            xml_string: XML/HTML content as a string.

        Returns:
            Tuple[Optional[etree._Element], Optional[str]]:
                (lxml_root, error_message) - root element or None and an error message
        """
        if not xml_string or not isinstance(xml_string, str):
            return None, "Empty or invalid XML string"

        try:
            root = etree.fromstring(xml_string.encode('utf-8'))
            return root, None
        except etree.XMLSyntaxError:
            pass

        # Retry after replacing known patent-XML entities.
        cleaned = _replace_entities(xml_string)
        try:
            root = etree.fromstring(cleaned.encode('utf-8'))
            return root, None
        except etree.XMLSyntaxError as e:
            return None, f"XML Syntax Error: {e}"
        except Exception as e:
            return None, f"Unexpected parsing error: {e}"
