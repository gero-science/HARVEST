"""
Basic XML parser for patent documents.
Handles safe conversion of an XML string into an lxml object.
"""

from typing import Optional, Tuple
from lxml import etree


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
        except etree.XMLSyntaxError as e:
            return None, f"XML Syntax Error: {e}"
        except Exception as e:
            return None, f"Unexpected parsing error: {e}"

