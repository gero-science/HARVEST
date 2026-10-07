from lxml import etree


def _find_first(xml_root, *xpaths):
    """Return the first element matched by any of *xpaths*, or None."""
    for xp in xpaths:
        el = xml_root.find(xp)
        if el is not None:
            return el
    return None


def extract_relevant_xml_sections(xml_root, logger=None) -> str:
    """Extract title, abstract and description XML sections.

    Handles both the modern USPTO XML schema (``invention-title``,
    ``abstract``, ``description``) and the pre-2005 schema
    (``subdoc-bibliographic-information//title-of-invention``,
    ``subdoc-abstract``, ``subdoc-description``).
    """
    sections = []

    title = _find_first(
        xml_root,
        './/invention-title',
        './/title-of-invention',
    )
    if title is not None:
        title_xml = etree.tostring(title, encoding='unicode', pretty_print=True)
        sections.append(title_xml)

    abstract = _find_first(xml_root, './/abstract', './/subdoc-abstract')
    if abstract is not None:
        abstract_xml = etree.tostring(abstract, encoding='unicode', pretty_print=True)
        sections.append(abstract_xml)

    description = _find_first(xml_root, './/description', './/subdoc-description')
    if description is not None:
        description_xml = etree.tostring(description, encoding='unicode', pretty_print=True)
        sections.append(description_xml)

    combined_xml = '\n'.join(sections)

    if logger:
        logger.info(
            f"Extracted XML sections: title={'Yes' if title is not None else 'No'}, "
            f"abstract={'Yes' if abstract is not None else 'No'}, "
            f"description={'Yes' if description is not None else 'No'}, "
            f"total_length={len(combined_xml)} chars"
        )

    return combined_xml
