import csv
from io import StringIO
from typing import Dict, List


def merge_tsv_parts(tsv_parts: List[str], stage_name: str, patent_id: str = "", logger=None) -> List[Dict]:
    """Parse and merge multi-part TSV responses."""
    log_prefix = f"{stage_name} [{patent_id}]" if patent_id else stage_name
    merged_data = []
    headers = None

    for part_idx, tsv_part in enumerate(tsv_parts):
        if not tsv_part or not tsv_part.strip():
            if logger:
                logger.debug(f"{log_prefix} part {part_idx + 1}: empty, skipping")
            continue

        part_lines = tsv_part.strip().count('\n') + 1
        if logger:
            logger.debug(f"{log_prefix} part {part_idx + 1}: raw length={len(tsv_part)} chars, lines={part_lines}")

        try:
            if part_idx == 0:
                reader = csv.DictReader(StringIO(tsv_part.strip()), delimiter='\t')
                headers = reader.fieldnames
                part_data = list(reader)
                merged_data.extend(part_data)
                if logger:
                    logger.info(
                        f"{log_prefix} part {part_idx + 1}: parsed {len(part_data)} records, "
                        f"columns: {headers}"
                    )
            else:
                if headers is None:
                    if logger:
                        logger.error(f"{log_prefix} part {part_idx + 1}: no headers available from part 1, skipping")
                    continue

                reader = csv.DictReader(StringIO(tsv_part.strip()), fieldnames=headers, delimiter='\t')
                part_data = list(reader)
                merged_data.extend(part_data)
                if logger:
                    logger.info(
                        f"{log_prefix} part {part_idx + 1}: parsed {len(part_data)} records (continuation)"
                    )
        except Exception as e:
            if logger:
                logger.error(f"Failed to parse {log_prefix} part {part_idx + 1}: {e}")
                logger.debug(f"Failed TSV part (first 500 chars): {tsv_part[:500]}")
                preview_lines = tsv_part.strip().split('\n')[:3]
                logger.debug(f"First 3 lines of failed part: {preview_lines}")

    if logger:
        logger.info(f"{log_prefix}: total {len(merged_data)} records from {len(tsv_parts)} parts")
    return merged_data


def ensure_header_in_tsv(tsv_responses: List[str], expected_header: str, stage_name: str, patent_id: str = "", logger=None) -> List[str]:
    """Ensure the first TSV response has the expected header."""
    log_prefix = f"{stage_name} [{patent_id}]" if patent_id else stage_name

    if not tsv_responses:
        if logger:
            logger.debug(f"{log_prefix}: No responses to check for header")
        return []

    first_response = tsv_responses[0].strip()
    if not first_response:
        if logger:
            logger.debug(f"{log_prefix}: Empty first response, no header check needed")
        return tsv_responses

    first_line = first_response.split('\n')[0].strip()
    normalized_first_line = first_line.lower().replace('_', '').replace('-', '').replace(' ', '')
    normalized_expected = expected_header.lower().replace('_', '').replace('-', '').replace(' ', '')
    expected_keywords = [word.strip() for word in normalized_expected.split('\t') if word.strip()]
    is_header = all(keyword in normalized_first_line for keyword in expected_keywords)

    if is_header:
        if logger:
            logger.debug(f"{log_prefix}: ✓ Header already present")
        return tsv_responses

    if logger:
        logger.warning(
            f"{log_prefix}: ✗ Header missing! First line appears to be data: '{first_line[:80]}...'"
        )
        logger.info(f"{log_prefix}: → Adding expected header")

    tsv_responses[0] = expected_header + '\n' + first_response

    if logger:
        logger.debug(f"{log_prefix}: ✓ Header added successfully")
    return tsv_responses
