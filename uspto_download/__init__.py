"""Fetch USPTO patent archives and turn them into pipeline input.

    download_bulk      USPTO Open Data Portal -> bulk .tar/.zip archives
    extract_patents    bulk archives -> one ZIP per patent
    bioactivity_filter regex used to keep only patents reporting binding data
"""
