from lxml import etree


def extract_relevant_xml_sections(xml_root, logger=None) -> str:
    """Extract title, abstract and description XML sections."""
    sections = []

    title = xml_root.find('.//invention-title')
    if title is not None:
        title_xml = etree.tostring(title, encoding='unicode', pretty_print=True)
        sections.append(title_xml)

    abstract = xml_root.find('.//abstract')
    if abstract is not None:
        abstract_xml = etree.tostring(abstract, encoding='unicode', pretty_print=True)
        sections.append(abstract_xml)

    description = xml_root.find('.//description')
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
