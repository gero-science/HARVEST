"""
Hallucination Detector - scans processing results and finds LLM hallucinations.

This module analyzes Stage 3 and Stage 2 TSV files from all patents and detects:

Stage 3:
- Nonexistent CHEM IDs (hallucinated_chem_id)
- Nonexistent TIF files (hallucinated_tif)
- Nonexistent IUPAC names (hallucinated_iupac)
- TIF/CHEM ID number mismatch

Stage 2:
- Nonexistent bioactivity values (hallucinated_value)
- Support for symbolic notation (++, ++++, A/B/C) via table legend

Generates a detailed report for extraction quality analysis.
"""
import csv
import html
import json
import logging
import re
import sys
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def load_tsv_data(tsv_file: Path) -> Tuple[List[str], List[Dict]]:
    """
    Load a TSV file and return the header and data rows.

    Uses rstrip to preserve trailing tabs, pads short rows with empty strings.
    """
    if not tsv_file.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")

    with open(tsv_file, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n\r') for line in f if line.rstrip('\n\r')]

    if not lines:
        return [], []

    # First line is the header
    header = lines[0].split('\t')

    # Remaining lines are data
    data = []
    for line in lines[1:]:
        parts = line.split('\t')
        # Pad short rows with empty strings
        while len(parts) < len(header):
            parts.append('')
        if len(parts) > len(header):
            logging.warning(f"Line has more columns than header ({len(parts)} > {len(header)}): {line[:100]}")
            parts = parts[:len(header)]

        row = dict(zip(header, parts))
        data.append(row)

    return header, data


def extract_xml_from_zip(zip_path: Path, patent_id: str) -> Optional[str]:
    """
    Extract XML content from a patent ZIP file.
    """
    if not zip_path.exists():
        logging.warning(f"ZIP file not found: {zip_path}")
        return None

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            xml_files = [f for f in zip_file.namelist() if f.lower().endswith('.xml')]

            if not xml_files:
                logging.warning(f"No XML files found in ZIP: {zip_path}")
                return None

            matching_xml = None
            for xml_file in xml_files:
                if patent_id in xml_file:
                    matching_xml = xml_file
                    break

            if not matching_xml:
                matching_xml = xml_files[0]

            with zip_file.open(matching_xml) as xml_file:
                xml_content = xml_file.read().decode('utf-8', errors='replace')

            return xml_content

    except Exception as e:
        logging.error(f"Failed to extract XML from {zip_path}: {e}")
        return None


def find_chemical_ids_in_xml(xml_content: str) -> Set[str]:
    """
    Find all CHEM IDs in patent XML.
    """
    chem_ids = set()

    pattern = r'<chemistry\s+id=["\']([^"\']+)["\']'
    matches = re.findall(pattern, xml_content, re.IGNORECASE)
    chem_ids.update(matches)

    pattern2 = r'chemistry\s+id\s*=\s*["\']([^"\']+)["\']'
    matches2 = re.findall(pattern2, xml_content, re.IGNORECASE)
    chem_ids.update(matches2)

    return chem_ids


def find_tif_files_in_xml(xml_content: str) -> Set[str]:
    """
    Find all TIF files in patent XML.
    """
    tif_files = set()

    pattern = r'<img\s+file=["\']([^"\']+\.TIF)["\']'
    matches = re.findall(pattern, xml_content, re.IGNORECASE)
    tif_files.update(matches)

    pattern2 = r'file\s*=\s*["\']([^"\']+\.TIF)["\']'
    matches2 = re.findall(pattern2, xml_content, re.IGNORECASE)
    tif_files.update(matches2)

    return tif_files


def extract_word_and_number_from_alias(alias: str) -> Optional[Tuple[str, int]]:
    """
    Extract word and number from an alias in "word number" format.
    """
    if not alias or not alias.strip():
        return None

    alias_clean = alias.strip()

    pattern = r'^([a-zA-Z]+)\.?\s+(\d+)$'
    match = re.match(pattern, alias_clean, re.IGNORECASE)
    if match:
        word = match.group(1).lower()
        try:
            number = int(match.group(2))
            return (word, number)
        except ValueError:
            return None

    return None


def find_compound_alias_in_xml(xml_content: str, alias: str) -> bool:
    """
    Check whether compound_alias appears in patent XML text.
    """
    if not alias or not alias.strip():
        return False

    alias_clean = alias.strip()
    alias_without_brackets = re.sub(r'^[<\[\(]+|[>\]\)]+$', '', alias_clean)

    word_number = extract_word_and_number_from_alias(alias_without_brackets)

    if word_number:
        word, number = word_number

        word_variants = {
            'exp': ['exp', 'example', 'examples'],
            'example': ['exp', 'example', 'examples'],
            'compound': ['compound', 'comp', 'compounds'],
            'comp': ['compound', 'comp', 'compounds'],
            'ex': ['ex', 'example', 'examples'],
        }

        search_words = word_variants.get(word, [word])

        for search_word in search_words:
            pattern = rf'\b{re.escape(search_word)}\.?\s+{number}\b'
            if re.search(pattern, xml_content, re.IGNORECASE):
                return True

        for search_word in search_words:
            pattern = rf'\b{re.escape(search_word)}{number}\b'
            if re.search(pattern, xml_content, re.IGNORECASE):
                return True

    alias_escaped = re.escape(alias_without_brackets)
    pattern = rf'\b{alias_escaped}\b'
    if re.search(pattern, xml_content, re.IGNORECASE):
        return True

    if alias_clean != alias_without_brackets:
        alias_with_brackets_escaped = re.escape(alias_clean)
        pattern = rf'\b{alias_with_brackets_escaped}\b'
        if re.search(pattern, xml_content, re.IGNORECASE):
            return True

    return False


def strip_xml_tags(text: str) -> str:
    """Remove all XML/HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)


def _normalize_brackets(text: str) -> str:
    """Normalize bracket types and optional hyphens for flexible IUPAC matching.

    Maps { -> [, } -> ], ) -> ], ( -> [, and removes hyphens adjacent to brackets.
    This handles OCR/formatting artifacts in patent XML where bracket types are swapped.
    """
    # Unify all bracket types to square brackets
    text = text.replace('{', '[').replace('}', ']').replace('(', '[').replace(')', ']')
    # Remove hyphens adjacent to brackets: "]-terphenyl" matches "]terphenyl"
    text = re.sub(r'-(?=[\[\]])', '', text)
    text = re.sub(r'(?<=[\[\]])-', '', text)
    return text


def collapse_whitespace_around_punctuation(text: str) -> str:
    """Remove whitespace adjacent to brackets and semicolons in chemical names."""
    return re.sub(r'\s*([;\[\]\(\)\{\}])\s*', r'\1', text)


def normalize_unicode_text(text: str) -> str:
    """Normalize Unicode to ASCII equivalents for text matching.

    Uses Unicode character categories and names to map non-ASCII characters
    to their closest ASCII equivalents.  This is intentionally aggressive
    for the purpose of *matching* IUPAC names between LLM output and patent
    XML -- it is NOT a general-purpose transliterator.

    Category rules:
      Pd  (dash punctuation)               -> '-'
      Sm  with MINUS/HYPHEN in name        -> '-'
      Zs  (space separator)                -> ' '
      Cf  (format chars: ZWSP, soft hyph)  -> removed
      Pi/Pf (quote punctuation)            -> '"' or "'"
      Po  with PRIME in name               -> "'"
      Greek letters (Ll/Lu with GREEK)     -> spelled-out name (alpha, beta, …)

    Greek-letter expansion is needed because LLMs consistently spell out
    Greek letters (``alpha-methyl``) while patent XML uses the actual
    Unicode character (``α-methyl``).
    """
    text = unicodedata.normalize('NFKC', text)

    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue

        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, '')

        if cat == 'Pd' or 'MINUS' in name or 'HYPHEN' in name:
            out.append('-')
        elif cat == 'Zs':
            out.append(' ')
        elif cat == 'Cf':
            pass  # remove invisible formatting characters
        elif 'PRIME' in name and cat == 'Po':
            out.append("'")
        elif cat in ('Pi', 'Pf') or 'QUOTATION' in name:
            if 'SINGLE' in name or 'APOSTROPHE' in name:
                out.append("'")
            else:
                out.append('"')
        # Greek letters: α → "alpha", β → "beta", etc.
        elif 'GREEK' in name and 'LETTER' in name:
            # "GREEK SMALL LETTER ALPHA" → "alpha"
            greek_name = name.split()[-1].lower()
            out.append(greek_name)
        else:
            out.append(ch)

    return ''.join(out)



def normalize_iupac_for_search(iupac: str) -> str:
    """
    Normalize an IUPAC name for search by decoding HTML entities
    and converting Unicode dashes/whitespace to ASCII equivalents.

    Also strips caret-superscript notation (``N^6`` → ``N6``) that
    LLMs produce for ``<sup>`` markup in patent XML.
    """
    if not iupac:
        return ""

    normalized = html.unescape(iupac)
    normalized = normalize_unicode_text(normalized)
    # Strip caret-superscript: N^6 → N6, C^7 → C7
    normalized = re.sub(r'\^(?=\d)', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)

    return normalized.strip()


def _build_table_column_texts(xml_content: str) -> List[str]:
    """
    Builds concatenated text per column across all <row> elements in tables.

    For each table, collects <entry> content per column position and
    concatenates them, producing strings that represent full column content.
    This helps find IUPACs that are split across table rows.

    Returns:
        List of concatenated column texts
    """
    column_texts = []

    # Find all tables
    table_pattern = r'<tables[^>]*>(.*?)</tables>'
    tables = re.findall(table_pattern, xml_content, re.DOTALL | re.IGNORECASE)

    for table_content in tables:
        # Find all rows
        row_pattern = r'<row[^>]*>(.*?)</row>'
        rows = re.findall(row_pattern, table_content, re.DOTALL | re.IGNORECASE)

        if not rows:
            continue

        # Collect entries per column.
        # Match both <entry>...</entry> AND self-closing <entry/> so that
        # empty cells in continuation rows are counted for correct column
        # alignment.  Without this, <entry/> is skipped and subsequent
        # entries land in the wrong column index.
        columns: Dict[int, List[str]] = {}
        for row in rows:
            entry_iter = re.finditer(
                r'<entry(?:\s[^/>]*)?(?:\s*/\s*>|>(.*?)</entry>)',
                row, re.DOTALL | re.IGNORECASE,
            )
            for col_idx, m in enumerate(entry_iter):
                # group(1) is None for self-closing <entry/>
                raw = (m.group(1) or '').strip()
                text = re.sub(r'<[^>]+>', '', raw).strip()
                if text:
                    if col_idx not in columns:
                        columns[col_idx] = []
                    columns[col_idx].append(text)

        # Concatenate each column's texts
        for col_idx in sorted(columns.keys()):
            col_text = ' '.join(columns[col_idx])
            if len(col_text) > 20:  # Only add meaningful columns
                column_texts.append(col_text)

    return column_texts


def find_iupac_in_xml(xml_content: str, iupac: str) -> bool:
    """
    Check whether an IUPAC name appears in patent XML text.

    Search strategy:
    1. Full match (exact, with flexible whitespace, with/without word boundaries)
    2. Partial search (first tokens, prefix)
    3. Search in concatenated table columns (for IUPAC split across rows)
    """
    if not iupac or not iupac.strip():
        return False

    iupac_normalized = normalize_iupac_for_search(iupac)
    xml_decoded = normalize_unicode_text(html.unescape(xml_content))
    xml_tag_free = strip_xml_tags(xml_decoded)

    # Search targets: tag-free first (most likely match), then decoded, then raw
    search_targets = [xml_tag_free, xml_decoded, xml_content]

    # ============================================
    # STAGE 1: FULL MATCH
    # ============================================

    iupac_pattern = re.escape(iupac_normalized)
    iupac_pattern = iupac_pattern.replace(r'\ ', r'\s+')
    pattern = rf'\b{iupac_pattern}\b'
    for target in search_targets:
        if re.search(pattern, target, re.IGNORECASE):
            return True

    escaped_iupac = re.escape(iupac_normalized)
    for target in search_targets:
        if re.search(escaped_iupac, target, re.IGNORECASE):
            return True

    iupac_pattern_flexible = re.escape(iupac_normalized)
    iupac_pattern_flexible = iupac_pattern_flexible.replace(r'\ ', r'\s+')
    for target in search_targets:
        if re.search(iupac_pattern_flexible, target, re.IGNORECASE):
            return True

    # Punctuation-collapsed matching: handles whitespace diffs around brackets
    iupac_collapsed = collapse_whitespace_around_punctuation(iupac_normalized)
    xml_tag_free_collapsed = collapse_whitespace_around_punctuation(xml_tag_free)
    if iupac_collapsed.lower() in xml_tag_free_collapsed.lower():
        return True

    # Whitespace-stripped matching: handles mid-word spaces (e.g. "cyanom ethyl" vs "cyanomethyl")
    iupac_no_ws = iupac_normalized.replace(' ', '')
    xml_tag_free_no_ws = xml_tag_free.replace(' ', '')
    if iupac_no_ws.lower() in xml_tag_free_no_ws.lower():
        return True

    # Hyphen-normalized matching: handles optional hyphens between letter groups.
    # IUPAC names allow stylistic variation: "methylamino-purin" vs
    # "methylaminopurin", "cyclopentyl-amino" vs "cyclopentylamino",
    # "tetrahydro-thiophene" vs "tetrahydrothiophene".
    # Only strip hyphens between two letters — structural hyphens like
    # "purin-9-yl" (letter-digit) and "3,4-dihydroxy" (digit-letter) stay.
    _ALPHA_HYPHEN = re.compile(r'(?<=[a-zA-Z])-(?=[a-zA-Z])')
    iupac_hyph_norm = _ALPHA_HYPHEN.sub('', iupac_no_ws.lower())
    xml_hyph_norm = _ALPHA_HYPHEN.sub('', xml_tag_free_no_ws.lower())
    if iupac_hyph_norm in xml_hyph_norm:
        return True

    # Bracket-normalized matching: handles {/[ and }/] interchange plus optional hyphens
    # Common in patent XML where OCR/formatting swaps bracket types
    iupac_bracket_norm = _normalize_brackets(iupac_no_ws.lower())
    xml_bracket_norm = _normalize_brackets(xml_tag_free_no_ws.lower())
    if iupac_bracket_norm in xml_bracket_norm:
        return True

    # ============================================
    # STAGE 2: PARTIAL SEARCH
    # ============================================

    if len(iupac_normalized) > 100:
        prefix = iupac_normalized[:200]
        prefix_escaped = re.escape(prefix)
        prefix_escaped = prefix_escaped.replace(r'\ ', r'\s+')
        for target in search_targets:
            if re.search(prefix_escaped, target, re.IGNORECASE):
                return True
    elif len(iupac_normalized) > 50:
        prefix = iupac_normalized[:150]
        prefix_escaped = re.escape(prefix)
        prefix_escaped = prefix_escaped.replace(r'\ ', r'\s+')
        for target in search_targets:
            if re.search(prefix_escaped, target, re.IGNORECASE):
                return True

    tokens = re.split(r'[\s\-]+', iupac_normalized)

    if len(tokens) >= 15:
        search_phrase = ' '.join(tokens[:8])
        search_phrase_escaped = re.escape(search_phrase)
        search_phrase_escaped = search_phrase_escaped.replace(r'\ ', r'[\s\-]+')
        pattern = rf'\b{search_phrase_escaped}'
        for target in search_targets:
            if re.search(pattern, target, re.IGNORECASE):
                return True
    elif len(tokens) >= 10:
        search_phrase = ' '.join(tokens[:6])
        search_phrase_escaped = re.escape(search_phrase)
        search_phrase_escaped = search_phrase_escaped.replace(r'\ ', r'[\s\-]+')
        pattern = rf'\b{search_phrase_escaped}'
        for target in search_targets:
            if re.search(pattern, target, re.IGNORECASE):
                return True
    elif len(tokens) >= 4:
        search_phrase = ' '.join(tokens[:4])
        search_phrase_escaped = re.escape(search_phrase)
        search_phrase_escaped = search_phrase_escaped.replace(r'\ ', r'[\s\-]+')
        pattern = rf'\b{search_phrase_escaped}'
        for target in search_targets:
            if re.search(pattern, target, re.IGNORECASE):
                return True

    # ============================================
    # STAGE 3: SEARCH IN TABLE COLUMNS
    # ============================================

    column_texts = _build_table_column_texts(xml_content)
    iupac_no_ws = iupac_normalized.replace(' ', '').lower()
    for col_text in column_texts:
        col_decoded = normalize_unicode_text(html.unescape(col_text))
        if iupac_normalized.lower() in col_decoded.lower():
            return True
        # Also try flexible space matching in column text
        iupac_flex = re.escape(iupac_normalized).replace(r'\ ', r'\s+')
        if re.search(iupac_flex, col_decoded, re.IGNORECASE):
            return True
        # Whitespace-stripped matching for IUPACs split across table rows
        if iupac_no_ws and iupac_no_ws in col_decoded.replace(' ', '').lower():
            return True

    return False


def extract_number_from_tif(tif_filename: str) -> Optional[int]:
    """
    Extract the number from a TIF filename.
    """
    if not tif_filename:
        return None

    pattern = r'C(\d+)\.TIF$'
    match = re.search(pattern, tif_filename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    return None


def extract_number_from_chem_id(chem_id: str) -> Optional[int]:
    """
    Extract the number from a CHEM ID.
    """
    if not chem_id:
        return None

    pattern = r'-(\d+)$'
    match = re.search(pattern, chem_id)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    return None


# ============================================
# Stage 2: Bioactivity value checking
# ============================================

def extract_numeric_values(value_str: str) -> List[str]:
    """
    Extract numeric values from a bioactivity value string.

    Strips operators (<>~=≤≥), handles ranges (X to Y), ±, scientific notation.
    Returns empty list for purely non-numeric values (ND, no inhibition, etc.)

    Examples:
        "<10.5"          -> ["10.5"]
        "0.019 to 0.026" -> ["0.019", "0.026"]
        "10 ± 2"         -> ["10"]
        "1.5e-6"         -> ["1.5"]
        "1.5*10^-6"      -> ["1.5"]
        "++"             -> []
        "ND"             -> []
        "no inhibition"  -> []
        "< 0.025"        -> ["0.025"]
    """
    if not value_str or not value_str.strip():
        return []

    s = value_str.strip()

    # Strip leading operators
    s = re.sub(r'^[<>~=≤≥≈]+\s*', '', s)

    # Check if purely non-numeric (letters only, symbolic like ++, etc.)
    # After stripping operators, if no digits remain, it's non-numeric
    if not re.search(r'\d', s):
        return []

    numbers = []

    # Handle ranges: "X to Y", "X-Y" (but not negative numbers)
    range_match = re.match(r'([\d.]+(?:[eE][+-]?\d+)?)\s*(?:to|-)\s*([\d.]+(?:[eE][+-]?\d+)?)', s)
    if range_match:
        numbers.append(range_match.group(1))
        numbers.append(range_match.group(2))
        return numbers

    # Handle ±: "X ± Y" -> just X
    pm_match = re.match(r'([\d.]+(?:[eE][+-]?\d+)?)\s*[±]\s*[\d.]+', s)
    if pm_match:
        numbers.append(pm_match.group(1))
        return numbers

    # Handle scientific notation with *10^: "1.5*10^-6" -> "1.5"
    sci_match = re.match(r'([\d.]+)\s*[*×x]\s*10\s*\^\s*[+-]?\d+', s)
    if sci_match:
        numbers.append(sci_match.group(1))
        return numbers

    # Handle standard scientific notation: "1.5e-6" -> "1.5"
    std_sci_match = re.match(r'([\d.]+)[eE][+-]?\d+', s)
    if std_sci_match:
        numbers.append(std_sci_match.group(1))
        return numbers

    # Simple number (possibly with trailing text like "nM")
    simple_match = re.match(r'([\d.]+)', s)
    if simple_match:
        numbers.append(simple_match.group(1))
        return numbers

    return numbers


def extract_paragraph_text(xml_content: str, paragraph_id: str) -> Optional[str]:
    """
    Extract text from <p id="p-XXXX">...</p>, stripping XML tags.

    Args:
        xml_content: Full XML content
        paragraph_id: Paragraph ID like "p-0531"

    Returns:
        Plain text content of the paragraph, or None if not found
    """
    if not paragraph_id or not paragraph_id.startswith('p-'):
        return None

    # Search for the paragraph by id
    pattern = rf'<p\s+id="{re.escape(paragraph_id)}"[^>]*>(.*?)</p>'
    match = re.search(pattern, xml_content, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    # Strip XML tags
    text = re.sub(r'<[^>]+>', '', match.group(1))
    # Decode HTML entities
    text = html.unescape(text)
    return text.strip()


def _check_symbolic_value_in_xml(xml_content: str, reasoning: str) -> bool:
    """
    Check if the value was likely decoded from symbolic notation.

    Checks two scenarios:
    1. If reasoning references a Table — check table entries for symbolic patterns
    2. If reasoning references a paragraph — check paragraph text for symbolic patterns
    3. Always check for a legend defining symbolic notation in the document

    Args:
        xml_content: Full XML content
        reasoning: The reasoning field from the TSV record

    Returns:
        True if symbolic notation is used
    """
    if not reasoning:
        return False

    reasoning = reasoning.strip()

    # Check if document has a symbolic legend anywhere (e.g., "+++++" designates IC50 < 25 nM)
    has_legend = False
    if re.search(r'["\u201c]\+{2,}["\u201d]\s+designates', xml_content, re.IGNORECASE):
        has_legend = True
    elif re.search(r'&#x201[cd];?\+{2,}&#x201[cd];?\s+designates', xml_content, re.IGNORECASE):
        has_legend = True

    # Check if reasoning references a table
    table_match = re.match(r'Table\s+(\S+)', reasoning, re.IGNORECASE)
    if table_match:
        # Find all table sections in XML
        table_sections = re.findall(
            r'<tables[^>]*>.*?</tables>',
            xml_content, re.DOTALL | re.IGNORECASE
        )

        for table_section in table_sections:
            # Pattern 1: Plus symbols (++, +++, ++++, +++++)
            plus_entries = re.findall(r'<entry[^>]*>\s*(\+{2,})\s*</entry>', table_section, re.IGNORECASE)
            if plus_entries:
                return True

            # Pattern 2: Single letter grades in entry tags (A, B, C, D, etc.)
            letter_entries = re.findall(r'<entry[^>]*>\s*([A-D])\s*</entry>', table_section, re.IGNORECASE)
            if len(letter_entries) >= 3:
                return True

        # If there's a legend in the document and reasoning references a table, trust it
        if has_legend:
            return True

    # Check if reasoning references a paragraph
    para_match = re.match(r'p-\d+', reasoning)
    if para_match:
        para_text = extract_paragraph_text(xml_content, reasoning)
        if para_text:
            # Check if the paragraph contains symbolic notation like +++++
            if re.search(r'\+{2,}', para_text):
                return True
            # Check for letter grade patterns (single letters as values)
            if re.search(r'\b[A-D]\b', para_text) and has_legend:
                return True

        # Even if paragraph text doesn't have symbols directly,
        # if the document has a legend, the LLM may have decoded it
        if has_legend:
            return True

    return False


def _add_thousands_commas(num_str: str) -> str:
    """Convert a plain numeric string to comma-separated format.

    Examples: "30000" -> "30,000", "100000" -> "100,000", "5.5" -> "5.5"
    """
    if '.' in num_str:
        int_part, dec_part = num_str.split('.', 1)
    else:
        int_part, dec_part = num_str, None

    if len(int_part) <= 3:
        return num_str

    # Insert commas from right
    result = []
    for i, ch in enumerate(reversed(int_part)):
        if i > 0 and i % 3 == 0:
            result.append(',')
        result.append(ch)
    formatted = ''.join(reversed(result))

    if dec_part is not None:
        formatted += '.' + dec_part
    return formatted


def find_value_in_xml(xml_content: str, value_str: str, reasoning: str = None) -> Tuple[bool, str]:
    """
    Check if a bioactivity value can be found in the patent XML.

    Args:
        xml_content: Full XML content
        value_str: The value string from Stage 2 TSV
        reasoning: The reasoning field (paragraph ID or table reference)

    Returns:
        Tuple of (found: bool, reason: str)
        - (True, "found_in_paragraph") - value found in referenced paragraph
        - (True, "found_in_xml") - value found somewhere in XML
        - (True, "non_numeric") - value is non-numeric, not flagged
        - (True, "symbolic_decoded") - value decoded from symbolic notation
        - (False, "not_found") - value not found anywhere
    """
    # Extract numeric values
    numbers = extract_numeric_values(value_str)

    # If no numbers extracted, it's non-numeric (ND, ++, no inhibition, etc.)
    if not numbers:
        return (True, "non_numeric")

    # Precompute comma-formatted variants (e.g. "30000" -> "30,000")
    numbers_comma = [_add_thousands_commas(n) for n in numbers]
    has_comma_variants = any(nc != n for nc, n in zip(numbers_comma, numbers))

    # Try to find in referenced paragraph first
    if reasoning and reasoning.startswith('p-'):
        para_text = extract_paragraph_text(xml_content, reasoning)
        if para_text:
            all_found = all(num in para_text for num in numbers)
            if all_found:
                return (True, "found_in_paragraph")
            if has_comma_variants and all(nc in para_text for nc in numbers_comma):
                return (True, "found_in_paragraph")

    # Search in full XML (decoded)
    xml_decoded = html.unescape(xml_content)
    all_found_xml = all(num in xml_decoded for num in numbers)
    if all_found_xml:
        return (True, "found_in_xml")

    # Also check raw XML
    all_found_raw = all(num in xml_content for num in numbers)
    if all_found_raw:
        return (True, "found_in_xml")

    # Try comma-formatted variants (e.g. "30000" -> "30,000")
    if has_comma_variants:
        if all(nc in xml_decoded for nc in numbers_comma):
            return (True, "found_in_xml")

    # Try space-separated digit matching (e.g. "270" -> "2 7 0" in table entries)
    numbers_spaced = [' '.join(n) for n in numbers if n.isdigit() and len(n) >= 2]
    if numbers_spaced and all(ns in xml_decoded for ns in numbers_spaced):
        return (True, "found_in_xml")

    # Symbolic fallback: check if the value was decoded from symbolic notation
    if _check_symbolic_value_in_xml(xml_content, reasoning):
        return (True, "symbolic_decoded")

    return (False, "not_found")


def check_stage2_record_hallucinations(record: Dict, xml_content: str, patent_id: str) -> Dict:
    """
    Check a single Stage 2 record for hallucinated values.

    Only checks the 'value' field (no alias or metric checks).

    Args:
        record: Row from Stage 2 TSV
        xml_content: Full XML content
        patent_id: Patent ID

    Returns:
        Dict with hallucination info
    """
    hallucinations = {"value": False}
    details = {}

    value_str = record.get("value", "").strip()
    reasoning = record.get("reasoning", "").strip()

    if value_str:
        found, reason = find_value_in_xml(xml_content, value_str, reasoning)
        if not found:
            hallucinations["value"] = True
            details["value"] = value_str
            details["reasoning"] = reasoning
            details["compound"] = record.get("compound", "")
            details["binding_metric"] = record.get("binding_metric", "")

    return {
        "has_hallucination": any(hallucinations.values()),
        "hallucinations": hallucinations,
        "details": details
    }


def check_record_hallucinations(
    record: Dict,
    xml_content: str,
    chem_ids: Set[str],
    tif_files: Set[str],
    patent_id: str
) -> Dict:
    """
    Check a single Stage 3 record for hallucinations.

    Checks: chem_id, tif, iupac, mismatch (no alias check).
    TIF fallback: skip hallucination if reasoning says "Wrote TIF filename".
    """
    hallucinations = {
        "chem_id": False,
        "tif": False,
        "iupac": False,
        "mismatch": False
    }

    details = {}
    reasoning = record.get("reasoning", "").strip()

    # Check chemical_id
    chemical_id = record.get("chemical_id", "").strip()
    if chemical_id:
        if chemical_id not in chem_ids:
            hallucinations["chem_id"] = True
            details["chem_id"] = chemical_id

    # Check compound_IUPAC_identifier
    iupac_identifier = record.get("compound_IUPAC_identifier", "").strip()
    tif_number = None
    if iupac_identifier:
        if iupac_identifier.upper().endswith('.TIF'):
            # This is a TIF file
            if iupac_identifier not in tif_files:
                # TIF fallback: don't flag if reasoning says "Wrote TIF filename"
                if "Wrote TIF filename" in reasoning:
                    pass  # Not a hallucination — LLM explicitly wrote TIF name
                else:
                    hallucinations["tif"] = True
                    details["tif"] = iupac_identifier
            else:
                tif_number = extract_number_from_tif(iupac_identifier)
        else:
            # This is an IUPAC name
            if not find_iupac_in_xml(xml_content, iupac_identifier):
                hallucinations["iupac"] = True
                details["iupac"] = iupac_identifier

    # Check number consistency between TIF and chem_id
    if chemical_id and tif_number is not None:
        chem_number = extract_number_from_chem_id(chemical_id)
        if chem_number is not None:
            if chem_number != tif_number:
                hallucinations["mismatch"] = True
                details["mismatch"] = {
                    "chem_id": chemical_id,
                    "chem_number": chem_number,
                    "tif": iupac_identifier,
                    "tif_number": tif_number
                }

    details["reasoning"] = reasoning

    return {
        "has_hallucination": any(hallucinations.values()),
        "hallucinations": hallucinations,
        "details": details
    }


def process_single_patent(
    patent_dir: Path,
    source_path: Path,
) -> Optional[Dict]:
    """
    Process one patent and return hallucination statistics.
    Processes both Stage 3 and Stage 2 files.
    """
    patent_id = patent_dir.name

    # Locate the patent ZIP file
    zip_file = source_path / f"{patent_id}.zip"
    if not zip_file.exists():
        logging.warning(f"ZIP file not found for {patent_id}: {zip_file}")
        return None

    # Check if at least one stage file exists
    stage3_file = patent_dir / f"{patent_id}_agent1_stage3_compounds.tsv"
    stage2_file = patent_dir / f"{patent_id}_agent1_stage2_bioactivity.tsv"

    if not stage3_file.exists() and not stage2_file.exists():
        logging.debug(f"No Stage 2 or Stage 3 TSV found for {patent_id}")
        return None

    # Extract XML from ZIP
    xml_content = extract_xml_from_zip(zip_file, patent_id)
    if not xml_content:
        logging.warning(f"Failed to extract XML for {patent_id}")
        return None

    # Extract existing CHEM IDs and TIF files from XML
    chem_ids = find_chemical_ids_in_xml(xml_content)
    tif_files = find_tif_files_in_xml(xml_content)

    logging.info(f"Processing {patent_id}: found {len(chem_ids)} CHEM IDs, {len(tif_files)} TIF files")

    # Statistics for this patent
    patent_stats = {
        "patent_id": patent_id,
        "total": 0,
        "hallucinations": 0,
        "types": {
            "chem_id": 0,
            "tif": 0,
            "iupac": 0,
            "mismatch": 0
        },
        "hallucination_details": {
            "chem_id": [],
            "tif": [],
            "iupac": [],
            "mismatch": []
        },
        "stage2": {
            "total": 0,
            "hallucinations": 0,
            "types": {"value": 0},
            "hallucination_details": {"value": []}
        }
    }

    # ============================================
    # Stage 3 processing
    # ============================================
    if stage3_file.exists():
        try:
            header, data = load_tsv_data(stage3_file)
            patent_stats["total"] = len(data)

            for record in data:
                result = check_record_hallucinations(
                    record, xml_content, chem_ids, tif_files, patent_id
                )

                if result["has_hallucination"]:
                    patent_stats["hallucinations"] += 1

                    reasoning = result["details"].get("reasoning", "")

                    if result["hallucinations"].get("chem_id"):
                        patent_stats["hallucination_details"]["chem_id"].append({
                            "chem_id": result["details"].get("chem_id"),
                            "compound": record.get("compound", ""),
                            "reasoning": reasoning,
                        })
                        patent_stats["types"]["chem_id"] += 1

                    if result["hallucinations"].get("tif"):
                        patent_stats["hallucination_details"]["tif"].append({
                            "tif": result["details"].get("tif"),
                            "compound": record.get("compound", ""),
                            "reasoning": reasoning,
                        })
                        patent_stats["types"]["tif"] += 1

                    if result["hallucinations"].get("iupac"):
                        patent_stats["hallucination_details"]["iupac"].append({
                            "iupac": result["details"].get("iupac"),
                            "compound": record.get("compound", ""),
                            "reasoning": reasoning,
                        })
                        patent_stats["types"]["iupac"] += 1

                    if result["hallucinations"].get("mismatch"):
                        mismatch_info = result["details"].get("mismatch", {})
                        patent_stats["hallucination_details"]["mismatch"].append({
                            "chem_id": mismatch_info.get("chem_id", ""),
                            "chem_number": mismatch_info.get("chem_number"),
                            "tif": mismatch_info.get("tif", ""),
                            "tif_number": mismatch_info.get("tif_number"),
                            "compound": record.get("compound", ""),
                            "reasoning": reasoning,
                        })
                        patent_stats["types"]["mismatch"] += 1
        except Exception as e:
            logging.error(f"Failed to process Stage 3 for {patent_id}: {e}")

    # ============================================
    # Stage 2 processing
    # ============================================
    if stage2_file.exists():
        try:
            header2, data2 = load_tsv_data(stage2_file)
            patent_stats["stage2"]["total"] = len(data2)

            for record in data2:
                result = check_stage2_record_hallucinations(
                    record, xml_content, patent_id
                )

                if result["has_hallucination"]:
                    patent_stats["stage2"]["hallucinations"] += 1

                    if result["hallucinations"].get("value"):
                        patent_stats["stage2"]["types"]["value"] += 1
                        patent_stats["stage2"]["hallucination_details"]["value"].append({
                            "value": result["details"].get("value", ""),
                            "compound": result["details"].get("compound", ""),
                            "binding_metric": result["details"].get("binding_metric", ""),
                            "reasoning": result["details"].get("reasoning", ""),
                        })
        except Exception as e:
            logging.error(f"Failed to process Stage 2 for {patent_id}: {e}")

    # Print result immediately
    print_patent_result(patent_stats)
    sys.stdout.flush()

    return patent_stats


def print_patent_result(patent_stats: Dict) -> None:
    """
    Print processing results for a single patent.
    """
    patent_id = patent_stats["patent_id"]
    total = patent_stats["total"]
    hall_count = patent_stats["hallucinations"]
    pct = (hall_count / total * 100) if total > 0 else 0

    s2 = patent_stats.get("stage2", {})
    s2_total = s2.get("total", 0)
    s2_hall = s2.get("hallucinations", 0)
    s2_pct = (s2_hall / s2_total * 100) if s2_total > 0 else 0

    print(f"\n[{patent_id}] Stage3: {total} records, {hall_count} hall ({pct:.1f}%) | "
          f"Stage2: {s2_total} records, {s2_hall} hall ({s2_pct:.1f}%)")

    if hall_count > 0:
        types = patent_stats["types"]
        print(f"  Stage3 - CHEM ID: {types['chem_id']}, TIF: {types['tif']}, "
              f"IUPAC: {types['iupac']}, Mismatch: {types['mismatch']}")

    if s2_hall > 0:
        s2_types = s2.get("types", {})
        print(f"  Stage2 - Value: {s2_types.get('value', 0)}")


def generate_hallucination_report(
    output_dir: str,
    source_dir: str,
    max_workers: int = 1,
    verbose: bool = True
) -> Dict:
    """
    Scan all Stage 3 and Stage 2 TSV files and detect hallucinations.
    """
    stats = {
        "total_records": 0,
        "hallucinated_records": 0,
        "hallucinations": {
            "chem_id": [],
            "tif": [],
            "iupac": [],
            "mismatch": []
        },
        "stage2_total_records": 0,
        "stage2_hallucinated_records": 0,
        "stage2_hallucinations": {
            "value": []
        },
        "per_patent": {},
        # Patents whose analysis raised. Counted rather than only logged: an
        # empty report is indistinguishable from "nothing was wrong" unless the
        # failures are carried out with the results.
        "failed_patents": [],
        "analyzed_patents": 0,
    }

    output_path = Path(output_dir)
    source_path = Path(source_dir)

    patent_dirs = []
    for patent_dir in output_path.iterdir():
        if patent_dir.is_dir():
            patent_dirs.append(patent_dir)

    if not patent_dirs:
        logging.warning(f"No patent directories found in {output_dir}")
        return stats

    logging.info(f"Found {len(patent_dirs)} patents to process")

    if verbose:
        print(f"Processing {len(patent_dirs)} patents with {max_workers} workers...")

    # Process patents using ProcessPoolExecutor
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_patent = {
                executor.submit(process_single_patent, patent_dir, source_path): patent_dir
                for patent_dir in patent_dirs
            }

            for future in as_completed(future_to_patent):
                patent_dir = future_to_patent[future]
                try:
                    patent_stats = future.result()
                    if patent_stats:
                        _aggregate_stats(stats, patent_stats)
                    stats["analyzed_patents"] += 1
                except Exception as e:
                    logging.error(f"Error processing {patent_dir.name}: {e}")
                    stats["failed_patents"].append((patent_dir.name, f"{type(e).__name__}: {e}"))
    else:
        for patent_dir in patent_dirs:
            try:
                patent_stats = process_single_patent(patent_dir, source_path)
                if patent_stats:
                    _aggregate_stats(stats, patent_stats)
                stats["analyzed_patents"] += 1
            except Exception as e:
                logging.error(f"Error processing {patent_dir.name}: {e}")
                stats["failed_patents"].append((patent_dir.name, f"{type(e).__name__}: {e}"))

    failed = len(stats["failed_patents"])
    if failed:
        share = failed / len(patent_dirs)
        logging.error(
            f"{failed:,} of {len(patent_dirs):,} patents ({share:.1%}) could not be analyzed; "
            "the report below covers only the rest"
        )
        # A wholesale failure means the report describes nothing at all. Saying
        # "no hallucinations found" there is worse than saying nothing, so stop.
        if failed == len(patent_dirs):
            first = stats["failed_patents"][0]
            raise RuntimeError(
                f"Hallucination analysis failed for all {failed:,} patents. "
                f"First failure: {first[0]}: {first[1]}"
            )

    return stats


def _aggregate_stats(stats: Dict, patent_stats: Dict) -> None:
    """Aggregate patent stats into global stats."""
    patent_id = patent_stats["patent_id"]

    # Stage 3
    stats["total_records"] += patent_stats["total"]
    stats["hallucinated_records"] += patent_stats["hallucinations"]

    for hall_type in ["chem_id", "tif", "iupac", "mismatch"]:
        for detail in patent_stats["hallucination_details"][hall_type]:
            detail["patent"] = patent_id
            stats["hallucinations"][hall_type].append(detail)

    # Stage 2
    s2 = patent_stats.get("stage2", {})
    stats["stage2_total_records"] += s2.get("total", 0)
    stats["stage2_hallucinated_records"] += s2.get("hallucinations", 0)

    for detail in s2.get("hallucination_details", {}).get("value", []):
        detail["patent"] = patent_id
        stats["stage2_hallucinations"]["value"].append(detail)

    # Per-patent
    has_s3_hall = patent_stats["hallucinations"] > 0
    has_s2_hall = s2.get("hallucinations", 0) > 0
    if has_s3_hall or has_s2_hall:
        stats["per_patent"][patent_id] = {
            "total": patent_stats["total"],
            "hallucinations": patent_stats["hallucinations"],
            "types": patent_stats["types"],
            "stage2_total": s2.get("total", 0),
            "stage2_hallucinations": s2.get("hallucinations", 0),
            "stage2_types": s2.get("types", {})
        }


def print_hallucination_summary(stats: Dict) -> str:
    """
    Generate a text report on hallucinations, split by Stage 3 / Stage 2.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("HALLUCINATION DETECTION REPORT")
    lines.append("=" * 70)

    # ============================================
    # Stage 3 Summary
    # ============================================
    lines.append("")
    lines.append("-" * 40)
    lines.append("STAGE 3 (Compound Resolution)")
    lines.append("-" * 40)

    failed = len(stats.get("failed_patents", []))
    if failed:
        analyzed = stats.get("analyzed_patents", 0)
        lines.append("")
        lines.append(
            f"!! {failed:,} patents FAILED to analyze ({analyzed:,} succeeded). "
            "The figures below describe only the patents that succeeded."
        )
        for name, why in stats["failed_patents"][:5]:
            lines.append(f"     {name}: {why}")
        if failed > 5:
            lines.append(f"     ... and {failed - 5:,} more")
        lines.append("")

    total = stats["total_records"]
    hall_rec = stats["hallucinated_records"]
    pct = (hall_rec / total * 100) if total > 0 else 0

    lines.append(f"Total records analyzed: {total:,}")
    lines.append(f"Records with hallucinations: {hall_rec:,} ({pct:.2f}%)")
    lines.append("")

    lines.append("Breakdown by type:")
    lines.append(f"  - Nonexistent CHEM IDs:        {len(stats['hallucinations']['chem_id']):,}")
    lines.append(f"  - Nonexistent TIF files:       {len(stats['hallucinations']['tif']):,}")
    lines.append(f"  - Nonexistent IUPAC names:     {len(stats['hallucinations']['iupac']):,}")
    lines.append(f"  - TIF/CHEM ID number mismatch: {len(stats['hallucinations']['mismatch']):,}")

    # ============================================
    # Stage 2 Summary
    # ============================================
    lines.append("")
    lines.append("-" * 40)
    lines.append("STAGE 2 (Bioactivity Values)")
    lines.append("-" * 40)

    s2_total = stats["stage2_total_records"]
    s2_hall = stats["stage2_hallucinated_records"]
    s2_pct = (s2_hall / s2_total * 100) if s2_total > 0 else 0

    lines.append(f"Total records analyzed: {s2_total:,}")
    lines.append(f"Records with hallucinations: {s2_hall:,} ({s2_pct:.2f}%)")
    lines.append("")

    lines.append("Breakdown by type:")
    lines.append(f"  - Hallucinated values: {len(stats['stage2_hallucinations']['value']):,}")

    # ============================================
    # Per-patent table
    # ============================================
    lines.append("")
    lines.append("-" * 40)
    lines.append("PER-PATENT BREAKDOWN")
    lines.append("-" * 40)

    if stats["per_patent"]:
        lines.append(f"Patents affected: {len(stats['per_patent']):,}")
        lines.append("")

        sorted_patents = sorted(
            stats["per_patent"].items(),
            key=lambda x: x[1]["hallucinations"] + x[1].get("stage2_hallucinations", 0),
            reverse=True
        )[:20]

        if sorted_patents:
            lines.append(f"{'Patent':<25} {'S3 Hall':>8} {'S3 Tot':>8} {'S2 Hall':>8} {'S2 Tot':>8}")
            lines.append("-" * 60)
            for patent_id, pstats in sorted_patents:
                lines.append(
                    f"{patent_id:<25} "
                    f"{pstats['hallucinations']:>8} "
                    f"{pstats['total']:>8} "
                    f"{pstats.get('stage2_hallucinations', 0):>8} "
                    f"{pstats.get('stage2_total', 0):>8}"
                )
    else:
        lines.append("No patents with hallucinations detected!")

    # ============================================
    # Examples
    # ============================================
    lines.append("")
    lines.append("-" * 40)
    lines.append("EXAMPLES")
    lines.append("-" * 40)

    if stats["hallucinations"]["chem_id"]:
        lines.append("")
        lines.append("Hallucinated CHEM IDs (first 5):")
        for i, item in enumerate(stats["hallucinations"]["chem_id"][:5], 1):
            lines.append(
                f"  {i}. Patent: {item['patent']}, "
                f"CHEM ID: {item['chem_id']}, "
                f"Compound: {item.get('compound', 'N/A')}"
            )

    if stats["hallucinations"]["tif"]:
        lines.append("")
        lines.append("Hallucinated TIF files (first 5):")
        for i, item in enumerate(stats["hallucinations"]["tif"][:5], 1):
            lines.append(
                f"  {i}. Patent: {item['patent']}, "
                f"TIF: {item['tif']}, "
                f"Compound: {item.get('compound', 'N/A')}"
            )

    if stats["hallucinations"]["iupac"]:
        lines.append("")
        lines.append("Hallucinated IUPAC names (first 5):")
        for i, item in enumerate(stats["hallucinations"]["iupac"][:5], 1):
            iupac_display = item['iupac'][:80] + "..." if len(item['iupac']) > 80 else item['iupac']
            lines.append(
                f"  {i}. Patent: {item['patent']}, "
                f"IUPAC: {iupac_display}, "
                f"Compound: {item.get('compound', 'N/A')}"
            )

    if stats["hallucinations"]["mismatch"]:
        lines.append("")
        lines.append("TIF/CHEM ID number mismatches (first 5):")
        for i, item in enumerate(stats["hallucinations"]["mismatch"][:5], 1):
            lines.append(
                f"  {i}. Patent: {item['patent']}, "
                f"CHEM ID: {item.get('chem_id', 'N/A')} (#{item.get('chem_number', '?')}), "
                f"TIF: {item.get('tif', 'N/A')} (#{item.get('tif_number', '?')})"
            )

    if stats["stage2_hallucinations"]["value"]:
        lines.append("")
        lines.append("Hallucinated Stage 2 values (first 5):")
        for i, item in enumerate(stats["stage2_hallucinations"]["value"][:5], 1):
            lines.append(
                f"  {i}. Patent: {item['patent']}, "
                f"Value: {item['value']}, "
                f"Metric: {item.get('binding_metric', 'N/A')}, "
                f"Compound: {item.get('compound', 'N/A')}, "
                f"Reasoning: {item.get('reasoning', 'N/A')}"
            )

    # ============================================
    # Recommendations
    # ============================================
    lines.append("")
    lines.append("-" * 40)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)

    combined_total = total + s2_total
    combined_hall = hall_rec + s2_hall
    combined_pct = (combined_hall / combined_total * 100) if combined_total > 0 else 0

    if combined_pct > 10:
        lines.append(f"WARNING: HIGH hallucination rate: {combined_pct:.1f}%")
        lines.append("   - Review prompts for clarity")
        lines.append("   - Check if chemistry nodes are extracted correctly")
    elif combined_pct > 5:
        lines.append(f"MODERATE hallucination rate: {combined_pct:.1f}%")
        lines.append("   - Review affected patents manually")
    elif combined_pct > 0:
        lines.append(f"LOW hallucination rate: {combined_pct:.1f}% - acceptable")
        lines.append("   - Monitor in production")
    elif failed:
        # Nothing was flagged, but patents failed to analyze -- a clean result
        # here would be indistinguishable from having checked nothing.
        lines.append(
            f"NO hallucinations found, but {failed:,} patents failed to analyze "
            "- treat this as inconclusive, not clean"
        )
    else:
        lines.append("NO hallucinations detected - excellent!")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_report_csv(stats: Dict, csv_path: str) -> None:
    """
    Save hallucination report as CSV files.

    Creates two files:
    - {csv_path}: Per-patent summary
    - {csv_path_stem}_details.csv: Detailed per-record hallucinations
    """
    csv_path = Path(csv_path)
    details_path = csv_path.with_name(csv_path.stem + "_details.csv")

    # Per-patent summary CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'patent_id',
            'stage3_total', 'stage3_hallucinations',
            'stage3_chem_id', 'stage3_tif', 'stage3_iupac', 'stage3_mismatch',
            'stage2_total', 'stage2_hallucinations', 'stage2_value'
        ])

        for patent_id, pstats in sorted(stats["per_patent"].items()):
            writer.writerow([
                patent_id,
                pstats['total'], pstats['hallucinations'],
                pstats['types'].get('chem_id', 0),
                pstats['types'].get('tif', 0),
                pstats['types'].get('iupac', 0),
                pstats['types'].get('mismatch', 0),
                pstats.get('stage2_total', 0),
                pstats.get('stage2_hallucinations', 0),
                pstats.get('stage2_types', {}).get('value', 0),
            ])

    # Details CSV
    with open(details_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'patent_id', 'stage', 'hallucination_type',
            'value', 'compound', 'reasoning', 'extra_info'
        ])

        # Stage 3 details
        for hall_type in ["chem_id", "tif", "iupac", "mismatch"]:
            for item in stats["hallucinations"][hall_type]:
                if hall_type == "chem_id":
                    value = item.get("chem_id", "")
                    extra = ""
                elif hall_type == "tif":
                    value = item.get("tif", "")
                    extra = ""
                elif hall_type == "iupac":
                    value = item.get("iupac", "")
                    extra = ""
                elif hall_type == "mismatch":
                    value = f"{item.get('chem_id', '')} / {item.get('tif', '')}"
                    extra = f"chem#={item.get('chem_number', '')}, tif#={item.get('tif_number', '')}"
                else:
                    value = ""
                    extra = ""

                writer.writerow([
                    item.get("patent", ""),
                    "stage3",
                    hall_type,
                    value,
                    item.get("compound", ""),
                    item.get("reasoning", ""),
                    extra
                ])

        # Stage 2 details
        for item in stats["stage2_hallucinations"]["value"]:
            writer.writerow([
                item.get("patent", ""),
                "stage2",
                "value",
                item.get("value", ""),
                item.get("compound", ""),
                item.get("reasoning", ""),
                f"metric={item.get('binding_metric', '')}"
            ])

    print(f"\nCSV report saved: {csv_path}")
    print(f"CSV details saved: {details_path}")


if __name__ == "__main__":
    """Standalone execution for testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Detect hallucinations in patent processing results")
    parser.add_argument("output_dir", help="Directory with processed patents (contains patent subdirectories)")
    parser.add_argument("source_dir", help="Directory with source ZIP files (contains {patent_number}.zip files)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text summary")
    parser.add_argument("--csv", type=str, default=None,
                       help="Save per-patent summary and details to CSV files")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    parser.add_argument("--workers", type=int, default=16,
                       help="Number of parallel workers (processes) for processing patents (default: 16)")
    parser.add_argument("--no-verbose", action="store_true",
                       help="Don't print results for each patent as they are processed")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    verbose = not args.no_verbose

    report = generate_hallucination_report(
        args.output_dir,
        args.source_dir,
        max_workers=args.workers,
        verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(print_hallucination_summary(report))

    if args.csv:
        save_report_csv(report, args.csv)
