import re


def normalize_compound_alias(alias: str) -> str:
    """Normalize compound aliases exactly as the historical extractor does."""
    if not alias:
        return ""

    normalized = alias.lower().strip()
    normalized = re.sub(r'\bexample\b', 'ex', normalized)
    normalized = re.sub(r'\bex\b', 'ex', normalized)
    normalized = re.sub(r'\bcompound\b', 'cmpd', normalized)
    normalized = re.sub(r'\bcomp\b', 'cmpd', normalized)
    normalized = re.sub(r'\bcmpd\b', 'cmpd', normalized)
    normalized = re.sub(r'\bstructure\b', 'struct', normalized)
    normalized = re.sub(r'\bstruct\b', 'struct', normalized)
    normalized = normalized.replace('.', '')
    normalized = normalized.replace(' ', '')
    normalized = normalized.replace('-', '')
    normalized = normalized.replace('_', '')
    return normalized


def extract_compound_number(alias: str) -> str:
    """Extract the normalized trailing compound number from an alias."""
    if not alias:
        return ""

    cleaned = alias.strip()
    cleaned = cleaned.replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
    cleaned = cleaned.replace('(', '').replace(')', '')
    cleaned = cleaned.replace('[', '').replace(']', '')
    cleaned = cleaned.replace('{', '').replace('}', '')

    match = re.search(r'(\d+[a-zA-Z]*)$', cleaned)
    if match:
        number_str = match.group(1)
        digit_match = re.match(r'^(\d+)([a-zA-Z]*)$', number_str)
        if digit_match:
            digits = digit_match.group(1)
            letters = digit_match.group(2).lower()
            normalized_digits = digits.lstrip('0') or '0'
            return normalized_digits + letters
        return number_str.lower()

    if re.match(r'^\d+[a-zA-Z]*$', cleaned):
        digit_match = re.match(r'^(\d+)([a-zA-Z]*)$', cleaned)
        if digit_match:
            digits = digit_match.group(1)
            letters = digit_match.group(2).lower()
            normalized_digits = digits.lstrip('0') or '0'
            return normalized_digits + letters
        return cleaned.lower()

    return ""
