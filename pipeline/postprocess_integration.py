"""
Module for postprocessing and integrating data between Stage 2 and Stage 3.

Main task: fuzzy matching of compound aliases to reconcile different spellings
of the same compound across stages.

Examples:
    "Example 1" ↔ "Ex. 1" ↔ "Ex 1"
    "Compound 5a" ↔ "Comp. 5a" ↔ "Cmpd 5a"
    "Structure-10" ↔ "Struct. 10"
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def normalize_compound_alias(alias: str) -> str:
    """
    Normalize a compound alias for matching.

    Examples:
        "Example 669" → "example669"
        "Ex. 669" → "example669"
        "Ex 669" → "example669"
        "Compound 5" → "compound5"
        "Comp. 5" → "compound5"
        "Cmpd 5" → "compound5"
        "Structure 10" → "structure10"

    Args:
        alias: Original compound alias

    Returns:
        Normalized alias (lowercase, without spaces, dots, or hyphens)
    """
    if not alias:
        return ""

    normalized = alias.lower().strip()

    # Map common abbreviations to full forms
    # "Ex." or "Ex" → "example"
    normalized = re.sub(r'\bex\.?\b', 'example', normalized)

    # "Comp." or "Cmpd" → "compound"
    normalized = re.sub(r'\bcomp\.?\b', 'compound', normalized)
    normalized = re.sub(r'\bcmpd\.?\b', 'compound', normalized)

    # "Struct." → "structure"
    normalized = re.sub(r'\bstruct\.?\b', 'structure', normalized)

    # Remove all dots, spaces, hyphens, and underscores
    normalized = normalized.replace('.', '')
    normalized = normalized.replace(' ', '')
    normalized = normalized.replace('-', '')
    normalized = normalized.replace('_', '')

    return normalized


def extract_compound_number(alias: str) -> Optional[str]:
    """
    Extract the compound number from an alias.

    Examples:
        "Example 669" → "669"
        "Ex. 5a" → "5a"
        "Compound 10" → "10"
        "Structure-20b" → "20b"

    Args:
        alias: Original alias

    Returns:
        Compound number (with letters if present) or None
    """
    if not alias:
        return None

    # Look for a trailing number (digits plus optional letters)
    match = re.search(r'(\d+[a-zA-Z]*)$', alias.replace(' ', '').replace('-', '').replace('.', ''))
    if match:
        number = match.group(1)
        # Normalize: strip leading zeros, lowercase letters
        digit_match = re.match(r'^(\d+)([a-zA-Z]*)$', number)
        if digit_match:
            digits = digit_match.group(1).lstrip('0') or '0'
            letters = digit_match.group(2).lower()
            return digits + letters

    return None


def similarity_ratio(s1: str, s2: str) -> float:
    """
    Compute the similarity ratio of two strings (0.0 - 1.0).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio (1.0 = identical)
    """
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def is_safe_fuzzy_match(alias1: str, alias2: str,
                        min_similarity: float = 0.85,
                        require_same_number: bool = True) -> bool:
    """
    Check whether two aliases can safely be treated as the same compound.

    Safe matching criteria:
    1. High textual similarity (>= min_similarity)
    2. Same compound number (if require_same_number=True)
    3. Same type (Example/Compound/Structure)

    Special case: "bare numbers" (e.g. "5a", "10")
    - If one alias is only a number without a type
    - And the numbers match
    - Then treat them as the same compound (e.g. "Compound 5a" = "5a")

    Args:
        alias1: First alias
        alias2: Second alias
        min_similarity: Minimum similarity ratio (default 0.85)
        require_same_number: Require matching numbers (default True)

    Returns:
        True if the aliases can safely be considered identical
    """
    if not alias1 or not alias2:
        return False

    # Step 1: compare normalized versions
    norm1 = normalize_compound_alias(alias1)
    norm2 = normalize_compound_alias(alias2)

    # Exact match after normalization
    if norm1 == norm2:
        return True

    # Step 2: compare compound numbers
    num1 = extract_compound_number(alias1)
    num2 = extract_compound_number(alias2)

    if require_same_number:
        # Different numbers mean different compounds
        if num1 != num2:
            return False

        # Reject if a number could not be extracted (unsafe)
        if num1 is None or num2 is None:
            return False

    # Step 3: compare compound type (example/compound/structure)
    type1 = None
    type2 = None

    if re.search(r'\bex(ample)?\.?\b', alias1.lower()):
        type1 = 'example'
    elif re.search(r'\b(comp(ound)?|cmpd)\.?\b', alias1.lower()):
        type1 = 'compound'
    elif re.search(r'\bstruct(ure)?\.?\b', alias1.lower()):
        type1 = 'structure'

    if re.search(r'\bex(ample)?\.?\b', alias2.lower()):
        type2 = 'example'
    elif re.search(r'\b(comp(ound)?|cmpd)\.?\b', alias2.lower()):
        type2 = 'compound'
    elif re.search(r'\bstruct(ure)?\.?\b', alias2.lower()):
        type2 = 'structure'

    # If both types are known and differ, these are different compounds
    # BUT: if one type is None (bare number "5a"), allow matching
    if type1 and type2 and type1 != type2:
        return False

    # Step 3.5: special logic for bare numbers
    # If one alias is just a number (no Example/Compound/Structure type)
    # and the numbers match, treat them as the same compound
    is_bare_number1 = type1 is None and num1 is not None
    is_bare_number2 = type2 is None and num2 is not None

    if (is_bare_number1 or is_bare_number2) and num1 == num2:
        # One alias is a bare number and the numbers match
        # Example: "Compound 5a" and "5a" are the same compound
        return True

    # Step 4: compare textual similarity
    similarity = similarity_ratio(norm1, norm2)

    if similarity >= min_similarity:
        return True

    return False


def create_fuzzy_alias_mapping(stage2_data: List[Dict],
                               stage3_data: List[Dict],
                               min_similarity: float = 0.85) -> Dict[str, str]:
    """
    Build an alias mapping between Stage 2 and Stage 3 using fuzzy matching.

    Strategy:
    1. Collect all aliases from Stage 2 and Stage 3
    2. For each Stage 2 alias, find the closest match in Stage 3
    3. Apply strict safety criteria (same number, type, high similarity)
    4. If a match is found, create a Stage2_alias → Stage3_alias mapping

    Args:
        stage2_data: Data from Stage 2 (bioactivities)
        stage3_data: Data from Stage 3 (compounds)
        min_similarity: Minimum similarity ratio (0.85 = very similar)

    Returns:
        Dictionary {stage2_alias: stage3_alias} for replacement
    """
    if not stage2_data or not stage3_data:
        logger.debug("Empty stage2_data or stage3_data, no fuzzy matching needed")
        return {}

    # Collect unique aliases from Stage 2
    stage2_aliases = set()
    for item in stage2_data:
        alias = (item.get("compound") or "").strip()
        if alias:
            stage2_aliases.add(alias)

    # Collect unique aliases from Stage 3
    stage3_aliases = set()
    for item in stage3_data:
        alias = (item.get("compound") or "").strip()
        if alias:
            stage3_aliases.add(alias)

    logger.info(f"Fuzzy matching: {len(stage2_aliases)} aliases in Stage 2, {len(stage3_aliases)} in Stage 3")

    # Build the mapping
    alias_mapping = {}
    matched_count = 0

    for s2_alias in stage2_aliases:
        # Skip if there is already an exact match
        if s2_alias in stage3_aliases:
            continue

        # Find the best match in Stage 3
        best_match = None
        best_similarity = 0.0

        for s3_alias in stage3_aliases:
            # Check whether the match is safe
            if is_safe_fuzzy_match(s2_alias, s3_alias, min_similarity=min_similarity):
                # Compute exact similarity to pick the best candidate
                sim = similarity_ratio(
                    normalize_compound_alias(s2_alias),
                    normalize_compound_alias(s3_alias)
                )

                if sim > best_similarity:
                    best_similarity = sim
                    best_match = s3_alias

        # If a good match was found, add it to the mapping
        if best_match:
            alias_mapping[s2_alias] = best_match
            matched_count += 1
            logger.debug(
                f"Fuzzy match: '{s2_alias}' → '{best_match}' "
                f"(similarity: {best_similarity:.3f})"
            )

    if matched_count > 0:
        logger.info(
            f"Fuzzy matching: mapped {matched_count} aliases "
            f"(threshold: {min_similarity})"
        )
    else:
        logger.debug("Fuzzy matching: no aliases needed mapping")

    return alias_mapping


def apply_alias_mapping_to_stage2(stage2_data: List[Dict],
                                   alias_mapping: Dict[str, str]) -> List[Dict]:
    """
    Apply an alias mapping to Stage 2 data.

    Replaces aliases in the "compound" field according to the mapping.

    Args:
        stage2_data: Data from Stage 2
        alias_mapping: Dictionary {old_alias: new_alias}

    Returns:
        Updated Stage 2 data (new list; the input list is not modified)
    """
    if not alias_mapping:
        return stage2_data

    updated_data = []
    updated_count = 0

    for item in stage2_data:
        item_copy = item.copy()
        old_alias = (item_copy.get("compound") or "").strip()

        if old_alias in alias_mapping:
            new_alias = alias_mapping[old_alias]
            item_copy["compound"] = new_alias
            updated_count += 1
            logger.debug(f"Updated alias in Stage 2: '{old_alias}' → '{new_alias}'")

        updated_data.append(item_copy)

    if updated_count > 0:
        logger.info(f"Applied fuzzy alias mapping to {updated_count} Stage 2 records")

    return updated_data


# Public functions for export
__all__ = [
    'create_fuzzy_alias_mapping',
    'apply_alias_mapping_to_stage2',
    'normalize_compound_alias',
    'extract_compound_number',
    'is_safe_fuzzy_match',
]
