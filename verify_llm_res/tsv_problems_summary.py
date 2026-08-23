#!/usr/bin/env python3
"""
Combined script to find all TSV problems:
1. Files with inconsistent column counts
2. IUPAC names ending with dash '-'
"""

import os
import csv
from pathlib import Path
from typing import List, Dict


def count_tsv_columns(line: str) -> int:
    """Count the number of columns in a TSV line."""
    return len(line.split('\t'))


def check_column_consistency(file_path: Path) -> Dict:
    """Check if TSV file has consistent column counts."""
    details = {
        'header_columns': 0,
        'total_lines': 0,
        'inconsistent_lines': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            header_line = f.readline()
            if not header_line.strip():
                return details
            
            header_columns = count_tsv_columns(header_line.rstrip('\n\r'))
            details['header_columns'] = header_columns
            
            line_number = 2
            for line in f:
                line = line.rstrip('\n\r')
                if not line.strip():
                    continue
                
                line_columns = count_tsv_columns(line)
                details['total_lines'] += 1
                
                if line_columns != header_columns:
                    details['inconsistent_lines'].append((line_number, line_columns))
                
                line_number += 1
    except Exception as e:
        details['error'] = str(e)
    
    return details


def check_trailing_dash(file_path: Path) -> List[Dict]:
    """Check for compound_IUPAC_name entries ending with '-'."""
    problematic_entries = []
    
    try:
        csv.field_size_limit(10000000)
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader, start=2):
                iupac_name = row.get('compound_IUPAC_name', '')
                
                if iupac_name is None:
                    iupac_name = ''
                else:
                    iupac_name = iupac_name.strip()
                
                if iupac_name and iupac_name.endswith('-'):
                    problematic_entries.append({
                        'row': row_num,
                        'compound': row.get('compound', '') or '',
                        'compound_IUPAC_name': iupac_name,
                    })
    except Exception as e:
        pass  # Silently skip errors
    
    return problematic_entries


def find_tsv_files(directory: Path, pattern: str) -> List[Path]:
    """Find all TSV files matching pattern recursively."""
    tsv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(pattern):
                tsv_files.append(Path(root) / file)
    return sorted(tsv_files)


def main():
    """Main function."""
    directory = Path('bdb100_full_patent7')
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist!")
        return
    
    print("TSV FILES PROBLEMS SUMMARY")
    print("=" * 100)
    print()
    
    # Problem 1: Column count inconsistencies (all TSV files)
    print("1. COLUMN COUNT INCONSISTENCIES")
    print("-" * 100)
    
    all_tsv_files = find_tsv_files(directory, '.tsv')
    print(f"Checking {len(all_tsv_files)} TSV files...\n")
    
    column_problems = []
    for tsv_file in all_tsv_files:
        details = check_column_consistency(tsv_file)
        if details['inconsistent_lines']:
            column_problems.append((tsv_file, details))
    
    if column_problems:
        print(f"Found {len(column_problems)} files with column count issues:\n")
        for file_path, details in column_problems:
            patent_id = file_path.parent.name
            print(f"  {patent_id}/{file_path.name}")
            print(f"    Header columns: {details['header_columns']}, Total lines: {details['total_lines']}")
            print(f"    Inconsistent lines: {len(details['inconsistent_lines'])}")
            if len(details['inconsistent_lines']) <= 5:
                for line_num, col_count in details['inconsistent_lines']:
                    print(f"      Line {line_num}: {col_count} columns (expected {details['header_columns']})")
            else:
                for line_num, col_count in details['inconsistent_lines'][:3]:
                    print(f"      Line {line_num}: {col_count} columns (expected {details['header_columns']})")
                print(f"      ... and {len(details['inconsistent_lines']) - 3} more")
            print()
    else:
        print("  ✓ No column count issues found!\n")
    
    # Problem 2: Trailing dashes in Stage 2 files
    print("\n2. IUPAC NAMES ENDING WITH DASH (Stage 2 files)")
    print("-" * 100)
    
    stage2_files = find_tsv_files(directory, '_agent1_stage2_compounds.tsv')
    print(f"Checking {len(stage2_files)} stage2 files...\n")
    
    dash_problems = []
    for tsv_file in stage2_files:
        problems = check_trailing_dash(tsv_file)
        if problems:
            dash_problems.append((tsv_file, problems))
    
    if dash_problems:
        print(f"Found {sum(len(p) for _, p in dash_problems)} entries with trailing dash:\n")
        for file_path, problems in dash_problems:
            patent_id = file_path.parent.name
            print(f"  {patent_id}/{file_path.name}")
            for problem in problems:
                print(f"    Row {problem['row']}: {problem['compound']} - {problem['compound_IUPAC_name'][:60]}...")
            print()
    else:
        print("  ✓ No trailing dash issues found!\n")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total TSV files checked: {len(all_tsv_files)}")
    print(f"Files with column count issues: {len(column_problems)}")
    print(f"Stage 2 files checked: {len(stage2_files)}")
    print(f"Files with trailing dash issues: {len(dash_problems)}")
    print()
    
    # Files with multiple issues
    column_problem_names = {f.name for f, _ in column_problems}
    dash_problem_names = {f.name for f, _ in dash_problems}
    multiple_issues = column_problem_names & dash_problem_names
    
    if multiple_issues:
        print(f"Files with MULTIPLE issues ({len(multiple_issues)}):")
        for name in sorted(multiple_issues):
            print(f"  - {name}")
    else:
        print("✓ No files have multiple issues")
    
    print("=" * 100)


if __name__ == '__main__':
    main()

