#!/usr/bin/env python3
"""
Script to check all TSV files in a directory for column count consistency.
Finds files where data lines have different number of columns than expected by StageFormats.
"""

import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add the repo root to the path BEFORE importing from it: this used to sit
# below the import, so the module only resolved when CWD happened to be the
# repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.stage_formats import StageFormats



def count_tsv_columns(line: str) -> int:
    """Count the number of columns in a TSV line."""
    return len(line.split('\t'))


def extract_patent_id(file_path: Path) -> str:
    """Extract patent ID from file path (parent directory name)."""
    return file_path.parent.name


def get_stage_from_filename(file_path: Path) -> Optional[int]:
    """Extract stage number from filename."""
    filename = file_path.name.lower()
    if 'stage1' in filename:
        return 1
    elif 'stage2' in filename:
        return 2
    elif 'stage3' in filename:
        return 3
    return None


def extract_llm_refusal_text(file_path: Path, expected_columns: int) -> str:
    """
    Extract LLM refusal/reasoning text from TSV file.

    LLM sometimes outputs explanation text mixed with TSV data.
    This text has wrong column count (usually 1 column - just text).
    Collects all lines with 1 column (except header-like lines).
    """
    refusal_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line_stripped = line.rstrip('\n\r')
                if not line_stripped.strip():
                    continue
                # Collect lines with 1 column (refusal text)
                if count_tsv_columns(line_stripped) == 1:
                    refusal_lines.append(line_stripped)
    except Exception:
        pass

    return '\n'.join(refusal_lines)


def get_expected_columns_from_filename(file_path: Path) -> Optional[int]:
    """
    Determine expected column count based on filename.

    Looks for stage1, stage2, stage3 patterns in filename.
    Returns None if stage cannot be determined.
    """
    filename = file_path.name.lower()

    if 'stage1' in filename:
        return StageFormats.STAGE1_COLUMN_COUNT
    elif 'stage2' in filename:
        return StageFormats.STAGE2_COLUMN_COUNT
    elif 'stage3' in filename:
        return StageFormats.STAGE3_COLUMN_COUNT

    return None


def check_tsv_file(file_path: Path) -> Tuple[bool, Dict]:
    """
    Check if a TSV file has consistent column counts.

    Returns:
        (is_consistent, details_dict)
        details_dict contains:
            - expected_columns: expected number of columns from StageFormats (or header if unknown stage)
            - header_columns: number of columns in header
            - total_lines: total number of data lines
            - inconsistent_lines: list of (line_number, column_count) tuples
            - stage: detected stage number (1, 2, 3) or None
    """
    details = {
        'expected_columns': 0,
        'header_columns': 0,
        'total_lines': 0,
        'inconsistent_lines': [],
        'stage': None
    }

    # Determine expected columns from filename
    expected_columns = get_expected_columns_from_filename(file_path)
    if expected_columns is not None:
        if expected_columns == StageFormats.STAGE1_COLUMN_COUNT:
            details['stage'] = 1
        elif expected_columns == StageFormats.STAGE2_COLUMN_COUNT:
            details['stage'] = 2
        elif expected_columns == StageFormats.STAGE3_COLUMN_COUNT:
            details['stage'] = 3

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Read header
            header_line = f.readline()
            if not header_line.strip():
                # Empty file
                return True, details

            header_columns = count_tsv_columns(header_line.rstrip('\n\r'))
            details['header_columns'] = header_columns

            # Use StageFormats expected columns if available, otherwise fall back to header
            if expected_columns is not None:
                details['expected_columns'] = expected_columns
            else:
                details['expected_columns'] = header_columns

            check_columns = details['expected_columns']

            # Check each data line
            line_number = 2  # Start from line 2 (after header)
            for line in f:
                line = line.rstrip('\n\r')
                if not line.strip():  # Skip empty lines
                    continue

                line_columns = count_tsv_columns(line)
                details['total_lines'] += 1

                if line_columns != check_columns:
                    details['inconsistent_lines'].append((line_number, line_columns))

                line_number += 1

        # Determine consistency status
        num_inconsistent = len(details['inconsistent_lines'])
        if num_inconsistent == 0:
            is_consistent = True
        else:
            # Check if inconsistent lines are at the beginning (LLM refusal) or end (truncated)
            # LLM refusal: multiple consecutive lines with 1 column at the START of file
            # Truncated: single line with wrong columns at the END of file

            # Check if all inconsistent lines have 1 column and are consecutive from start
            first_inconsistent_line = details['inconsistent_lines'][0][0]
            all_single_column = all(col_count == 1 for _, col_count in details['inconsistent_lines'])

            # Check if it's just one truncated line at the end
            last_line_number = details['total_lines'] + 1  # +1 because total_lines counts data lines, line_number includes header
            is_truncated_end = (num_inconsistent == 1 and
                               details['inconsistent_lines'][0][0] == last_line_number)

            if is_truncated_end:
                # Single truncated line at end - ignore completely
                is_consistent = True
            elif all_single_column and first_inconsistent_line <= 2:
                # Multiple lines with 1 column starting from beginning - LLM refusal
                is_consistent = True
                details['llm_refusal'] = True
            else:
                is_consistent = False

        return is_consistent, details

    except Exception as e:
        details['error'] = str(e)
        return False, details


def find_all_tsv_files(directory: Path) -> List[Path]:
    """Find all TSV files recursively in the directory."""
    tsv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tsv'):
                tsv_files.append(Path(root) / file)
    return sorted(tsv_files)


def main():
    """Main function to check all TSV files."""
    directory = Path('next_17000')

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist!")
        return

    print(f"Scanning for TSV files in: {directory}")
    print("=" * 80)
    print(f"Expected columns by stage:")
    print(f"  Stage 1: {StageFormats.STAGE1_COLUMN_COUNT} columns")
    print(f"  Stage 2: {StageFormats.STAGE2_COLUMN_COUNT} columns")
    print(f"  Stage 3: {StageFormats.STAGE3_COLUMN_COUNT} columns")
    print("=" * 80)

    tsv_files = find_all_tsv_files(directory)
    print(f"Found {len(tsv_files)} TSV files\n")

    inconsistent_files = []
    llm_refusal_files = []  # Files with LLM refusal text at the beginning
    single_truncated_files = 0  # Count of files with single truncated line (ignored)

    for tsv_file in tsv_files:
        is_consistent, details = check_tsv_file(tsv_file)

        if not is_consistent:
            inconsistent_files.append((tsv_file, details))
        elif details.get('llm_refusal'):
            llm_refusal_files.append((tsv_file, details))
        elif len(details.get('inconsistent_lines', [])) == 1:
            # Single inconsistent line that was ignored (truncated)
            single_truncated_files += 1

    # Report results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if not inconsistent_files:
        print(f"\n✓ All {len(tsv_files)} TSV files have consistent column counts!")
    else:
        print(f"\n✗ Found {len(inconsistent_files)} file(s) with inconsistent column counts:\n")

        for file_path, details in inconsistent_files:
            print(f"File: {file_path}")
            stage_info = f" (Stage {details['stage']})" if details['stage'] else " (unknown stage)"
            print(f"  Expected columns{stage_info}: {details['expected_columns']}")
            print(f"  Header columns: {details['header_columns']}")
            print(f"  Total data lines: {details['total_lines']}")

            if 'error' in details:
                print(f"  ERROR: {details['error']}")
            else:
                print(f"  Inconsistent lines: {len(details['inconsistent_lines'])}")

                # Show first 10 inconsistent lines
                for line_num, col_count in details['inconsistent_lines'][:10]:
                    print(f"    Line {line_num}: {col_count} columns (expected {details['expected_columns']})")

                if len(details['inconsistent_lines']) > 10:
                    print(f"    ... and {len(details['inconsistent_lines']) - 10} more")

            print()

    # Save LLM refusal files to CSV (patent_id, stage, llm_refusal_text, file_path)
    if llm_refusal_files:
        csv_path = Path.cwd() / "llm_refusal_files.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['patent_id', 'stage', 'llm_refusal_text', 'file_path'])
            for file_path, details in llm_refusal_files:
                patent_id = extract_patent_id(file_path)
                stage = details['stage'] or get_stage_from_filename(file_path) or 'unknown'
                refusal_text = extract_llm_refusal_text(file_path, details['expected_columns'])
                writer.writerow([patent_id, stage, refusal_text, str(file_path)])
        print(f"\nSaved {len(llm_refusal_files)} files with LLM refusal to: {csv_path}")

    # Summary
    print("=" * 80)
    print(f"Summary: {len(inconsistent_files)}/{len(tsv_files)} files have column inconsistencies")
    print(f"         {len(llm_refusal_files)}/{len(tsv_files)} files have LLM refusal text (logged to CSV)")
    print(f"         {single_truncated_files}/{len(tsv_files)} files have single truncated line (ignored)")
    print("=" * 80)


if __name__ == '__main__':
    main()
