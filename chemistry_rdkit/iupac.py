"""IUPAC name normalization and bracket-balancing helpers.

These are pure text-processing utilities — no RDKit dependency.
Used by the bioactivity extraction pipeline to clean LLM-produced
IUPAC names before OPSIN conversion.
"""

import re


def is_balanced(text: str) -> list:
    """
    Check whether brackets are balanced in a string.

    Args:
        text: String to check

    Returns:
        Empty list if brackets are balanced, otherwise a list of unbalanced symbols
    """
    stack = []
    bracket_map = {")": "(", "]": "[", "}": "{"}

    for char in text:
        if char in bracket_map.values():
            stack.append(char)
        elif char in bracket_map.keys():
            if not stack or stack[-1] != bracket_map[char]:
                return [char]
            stack.pop()

    return stack


def fix_balanced_by_adding(text: str) -> str:
    """
    Tries to fix unbalanced brackets by adding missing brackets to the beginning/end.

    Args:
        text: String with potentially unbalanced brackets

    Returns:
        Fixed string or original if fix failed
    """
    need_fix = is_balanced(text)
    if len(need_fix) == 0:
        return text

    map_tuples = [('(', ')'), ('[', ']'), ('{', '}')]
    max_tries = 5
    count = 0
    while len(need_fix) > 0 and count < max_tries:
        for missing in need_fix:
            for problem in map_tuples:
                if missing in problem:
                    indx = problem.index(missing)
                    text = text + problem[indx-1]
                    text = text[-indx:] + text[:-indx]
        need_fix = is_balanced(text)
        count += 1
    return text


# Alias for backward compatibility
fix_balanced = fix_balanced_by_adding


def fix_balanced_by_removing(text: str) -> str:
    """
    Tries to fix unbalanced brackets by removing unmatched brackets.

    Args:
        text: String with potentially unbalanced brackets

    Returns:
        Fixed string with unmatched brackets removed
    """
    stack = []
    bracket_map = {")": "(", "]": "[", "}": "{"}
    to_remove = set()

    for i, char in enumerate(text):
        if char in bracket_map.values():
            stack.append((i, char))
        elif char in bracket_map.keys():
            if stack and stack[-1][1] == bracket_map[char]:
                stack.pop()
            else:
                to_remove.add(i)

    for idx, _ in stack:
        to_remove.add(idx)

    result = ''.join(char for i, char in enumerate(text) if i not in to_remove)
    return result


def fix_balanced_by_removing_variants(text: str) -> list[str]:
    """
    Generates multiple variants by trying to remove different combinations of brackets.
    When there are unmatched brackets, tries removing each possible bracket individually.

    Args:
        text: String with potentially unbalanced brackets

    Returns:
        List of variant strings with different bracket removal strategies
    """
    variants = []

    bracket_chars = "()[]{}/"
    bracket_positions = [i for i, char in enumerate(text) if char in bracket_chars]

    if not bracket_positions:
        return [text]

    variants.append(text)
    variants.append(fix_balanced_by_removing(text))

    for pos in bracket_positions:
        variant = text[:pos] + text[pos+1:]
        if len(is_balanced(variant)) == 0:
            variants.append(variant)

    if len(bracket_positions) >= 2:
        for i in range(len(bracket_positions)):
            for j in range(i+1, len(bracket_positions)):
                pos1, pos2 = bracket_positions[i], bracket_positions[j]
                variant = text[:pos1] + text[pos1+1:pos2] + text[pos2+1:]
                if len(is_balanced(variant)) == 0:
                    variants.append(variant)

    return variants


def normalize_iupac_spelling(name: str) -> str:
    """
    Normalizes IUPAC names by fixing common formatting errors.

    This function applies SAFE transformations that don't change molecular identity:
    - Removes spaces (IUPAC names should not contain spaces)
    - Fixes fused ring notation: "pyrido3,2-bpyrazin" -> "pyrido[3,2-b]pyrazin"
    - Fixes locant-letter hyphen: "4,4-a,5,6-tetrahydro" -> "4,4a,5,6-tetrahydro"
    - Fixes azol suffix before brackets: "imidazol[" -> "imidazo["
    - Fixes heterocyclic amino position: "2-dibenzofuranamino" -> "dibenzofuran-2-ylamino"
    - Removes trailing periods
    - Removes unnecessary outer parentheses

    NOTE: For parenthetical "and/or" alternatives, current behavior keeps the
    first branch. Downstream structure fields are single-valued, so preserving
    multiple stereoisomer variants would require a separate schema change.

    This function intentionally does NOT:
    - Remove "doubled" digits like "44-" - this could be "4,4-di" substitution

    Example:
    Input:  "1-(3-isopropyl-1H-pyrazol-5-yl)urea."
    Output: "1-(3-isopropyl-1H-pyrazol-5-yl)urea"

    Input:  "pyrido3,2-bpyrazin"
    Output: "pyrido[3,2-b]pyrazin"

    Input:  "7H-imidazol[4,5-c]pyridazine"
    Output: "7H-imidazo[4,5-c]pyridazine"

    Args:
        name: IUPAC chemical name string

    Returns:
        Normalized IUPAC name with formatting errors fixed
    """
    normalized_name = name

    # Fix common LLM errors with fused ring nomenclature
    normalized_name = re.sub(
        r'(\w+)(\d+,\d+)-([a-z])(?=\w)',
        r'\1[\2-\3]',
        normalized_name,
        flags=re.IGNORECASE
    )

    # Fix erroneous hyphen after locant before letter
    normalized_name = re.sub(
        r'(\d+,\d+)-([a-z])(?=,\d)',
        r'\1\2',
        normalized_name,
        flags=re.IGNORECASE
    )

    # Fix incorrect fused ring suffix: "imidazol[" -> "imidazo["
    normalized_name = re.sub(
        r'(imid|thi|ox|pyr|tri|tetr)azol(\[)',
        r'\1azo\2',
        normalized_name,
        flags=re.IGNORECASE
    )

    # Fix incorrect substituent position for heterocyclic amino groups
    heterocycles_polycyclic = (
        r'dibenzofuran|carbazol|fluoren|xanthen|thioxanthen|acridin|'
        r'phenazin|phenothiazin|phenoxazin|dibenzothiophen|dibenzothiazol'
    )
    normalized_name = re.sub(
        rf'(\d+)-({heterocycles_polycyclic})(e?)(amino|amine)',
        r'\2\3-\1-yl\4',
        normalized_name,
        flags=re.IGNORECASE
    )

    # Keep only first branch of parenthetical "and/or" alternatives
    pattern = r'\(([^)]*?) (and|or) [^)]*\)'
    replacement = r'(\1)'
    normalized_name = re.sub(
        pattern,
        replacement,
        normalized_name,
        flags=re.IGNORECASE
    )

    # remove spaces around hyphens
    normalized_name = re.sub(r'\s*-\s*', '-', normalized_name)

    # Remove outer parentheses if they wrap the entire name
    while normalized_name.startswith('(') and normalized_name.endswith(')'):
        inner = normalized_name[1:-1]
        if len(is_balanced(inner)) == 0:
            normalized_name = inner
        else:
            break

    # remove trailing periods
    normalized_name = normalized_name.rstrip('.')

    # remove leading/trailing whitespace
    normalized_name = normalized_name.strip()

    return normalized_name
