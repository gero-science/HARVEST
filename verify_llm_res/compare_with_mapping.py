#!/usr/bin/env python3
"""
Script for comparing pipeline results with the BindingDB reference table
using the application_number ↔ patent_number mapping from
curated_data/patent_mapping.csv (or legacy bdb_100_dict.json).
"""

import csv
import json
import argparse
from collections import Counter
from typing import Dict, Set, Tuple

import pandas as pd


def load_patent_mapping(mapping_file: str) -> Dict[str, str]:
    """Load patent mapping from CSV or JSONL.

    Accepts ``curated_data/patent_mapping.csv`` (columns
    ``patent_number,application_number``) or legacy JSONL with the same
    two fields.

    Builds a bidirectional mapping: application_number → patent_number and back.

    Returns:
        dict: Mapping {application_number: patent_number, patent_number: patent_number}
    """
    mapping: Dict[str, str] = {}

    def _add(app_num: str, patent_num: str) -> None:
        app_num = app_num.strip()
        patent_num = str(patent_num).strip()
        if not app_num or not patent_num:
            return
        app_norm = app_num.replace('US', '').replace('A1', '').replace('A2', '').replace('B1', '').replace('B2', '')
        mapping[app_num] = patent_num
        mapping[app_norm] = patent_num
        mapping[patent_num] = patent_num

    if mapping_file.endswith('.csv'):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                _add(row.get('application_number', ''), row.get('patent_number', ''))
    else:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    _add(entry.get('application_number', ''), entry.get('patent_number', ''))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:100]}... Error: {e}")

    return mapping


def normalize_patent_number(patent_num: str, mapping: Dict[str, str]) -> str:
    """
    Normalize a patent number using the mapping.

    Args:
        patent_num: Patent number (may be application_number or patent_number)
        mapping: Mapping from load_patent_mapping

    Returns:
        str: Normalized patent number (numeric)
    """
    if pd.isna(patent_num) or not patent_num:
        return ""
    
    patent_num = str(patent_num).strip()
    
    # If already a numeric patent number
    if patent_num.isdigit():
        return patent_num
    
    # Try to find in mapping
    if patent_num in mapping:
        return mapping[patent_num]
    
    # Try to normalize application_number
    normalized = patent_num.replace('US', '').replace('A1', '').replace('A2', '').replace('B1', '').replace('B2', '')
    if normalized in mapping:
        return mapping[normalized]
    
    # If not found, return as-is (may already be normalized)
    return patent_num


def run_comparison_with_mapping(
    test_table_path: str,
    bdb_table_path: str,
    mapping_file: str,
    show_not_found_examples: bool = False,
    show_top_new_patents: bool = False
):
    """
    Compare the test table with the BindingDB reference table
    using the application_number ↔ patent_number mapping.

    Args:
        test_table_path: Path to the test table (CSV)
        bdb_table_path: Path to the reference table (Parquet)
        mapping_file: Path to the mapping file (CSV or JSONL)
        show_not_found_examples: Show examples from only_in_bdb
        show_top_new_patents: Show top patents from only_in_test

    Returns:
        tuple: (results_dict, detailed_output_string)
    """
    print("Loading data...")
    
    # Load tables
    df_test = pd.read_csv(test_table_path)
    df_bdb = pd.read_parquet(bdb_table_path)
    
    print(f"Test table: {len(df_test)} rows")
    print(f"Reference table: {len(df_bdb)} rows")
    
    # Load mapping
    print("Loading patent number mapping...")
    patent_mapping = load_patent_mapping(mapping_file)
    print(f"Loaded {len(patent_mapping)} mapping entries")
    
    # Normalize patent numbers in the test table
    print("Normalizing patent numbers in the test table...")
    
    # If the test table has patent_number, use it for normalization
    if 'patent_number' in df_test.columns:
        df_test['normalized_pub_number'] = df_test['patent_number'].apply(
            lambda x: normalize_patent_number(x, patent_mapping)
        )
    elif 'normalized_pub_number' not in df_test.columns:
        raise ValueError("Test table has neither 'patent_number' nor 'normalized_pub_number'")
    
    # Check required columns
    required_cols_test = ['normalized_pub_number', 'Ligand InChI Key', 'Sequence']
    missing_cols_test = [col for col in required_cols_test if col not in df_test.columns]
    if missing_cols_test:
        raise ValueError(f"Test table is missing columns: {missing_cols_test}")
    
    # Cast to string for correct merge
    df_test['normalized_pub_number'] = df_test['normalized_pub_number'].astype(str)
    df_bdb['normalized_pub_number'] = df_bdb['normalized_pub_number'].astype(str)
    
    # Rename columns for merge convenience
    df_test_renamed = df_test.rename(columns={
        'normalized_pub_number': 'normalized_pub_number',
        'Ligand InChI Key': 'Ligand InChI Key',
        'Sequence': 'Sequence',
        'Ki (nM)': 'Ki (nM)',
        'IC50 (nM)': 'IC50 (nM)',
        'Kd (nM)': 'Kd (nM)',
        'EC50 (nM)': 'EC50 (nM)'
    })
    
    df_bdb_renamed = df_bdb.rename(columns={
        'normalized_pub_number': 'normalized_pub_number',
        'inchi_key_restored': 'Ligand InChI Key',
        'bdb__bindingdb_target_chain_sequence': 'Sequence',
        'bdb__ki (nm)': 'bdb__ki (nm)',
        'bdb__ic50 (nm)': 'bdb__ic50 (nm)',
        'bdb__kd (nm)': 'bdb__kd (nm)',
        'bdb__ec50 (nm)': 'bdb__ec50 (nm)',
    })
    
    # Check required columns in the reference table
    required_cols_bdb = ['normalized_pub_number', 'Ligand InChI Key', 'Sequence']
    missing_cols_bdb = [col for col in required_cols_bdb if col not in df_bdb_renamed.columns]
    if missing_cols_bdb:
        raise ValueError(f"Reference table is missing columns: {missing_cols_bdb}")
    
    print("Merging on keys: normalized_pub_number, Ligand InChI Key, Sequence...")
    
    # Merge on three keys
    merged = pd.merge(
        df_test_renamed,
        df_bdb_renamed,
        on=['normalized_pub_number', 'Ligand InChI Key', 'Sequence'],
        how='inner',
        suffixes=('_test', '_bdb')
    )
    
    print(f"Found {len(merged)} key matches")
    
    def normalize_value(v):
        """Normalize a value for comparison"""
        if pd.isna(v) or v == '':
            return None
        v_str = str(v).strip()
        if v_str.startswith('<') or v_str.startswith('>'):
            v_str = v_str[1:]
        try:
            return float(v_str)
        except (ValueError, TypeError):
            return None
    
    # Metric value comparison function
    def compare_metrics(row):
        """Check whether at least one metric matches (same type only)"""
        metrics = [
            ('Ki (nM)', 'bdb__ki (nm)'),
            ('IC50 (nM)', 'bdb__ic50 (nm)'),
            ('Kd (nM)', 'bdb__kd (nm)'),
            ('EC50 (nM)', 'bdb__ec50 (nm)')
        ]
        
        for test_col, bdb_col in metrics:
            val_test = normalize_value(row.get(test_col))
            val_bdb = normalize_value(row.get(bdb_col))
            
            # Compare only when both values exist (same metric type)
            if val_test is not None and val_bdb is not None:
                if abs(val_test - val_bdb) < 0.01:  # Allowed tolerance
                    return test_col
        
        return None
    
    # Helper to collect matching metrics (same type only)
    def get_matching_metrics(row):
        """Return metrics present in both tables (same type)"""
        metrics = [
            ('Ki (nM)', 'bdb__ki (nm)'),
            ('IC50 (nM)', 'bdb__ic50 (nm)'),
            ('Kd (nM)', 'bdb__kd (nm)'),
            ('EC50 (nM)', 'bdb__ec50 (nm)')
        ]
        
        matching = []
        for test_col, bdb_col in metrics:
            val_test = normalize_value(row.get(test_col))
            val_bdb = normalize_value(row.get(bdb_col))
            
            # Both values present means a matching metric (same type)
            if val_test is not None and val_bdb is not None:
                matching.append((test_col, bdb_col, val_test, val_bdb))
        
        return matching
    
    # Find metric matches
    print("Comparing metric values...")
    matches = merged[merged.apply(compare_metrics, axis=1).notnull()]
    
    # Find key matches with differing metrics
    non_matches = merged[merged.apply(compare_metrics, axis=1).isnull()]
    
    # Pairs present only in the test table
    test_pub_numbers = set(df_test_renamed['normalized_pub_number'].dropna())
    bdb_pub_numbers = set(df_bdb_renamed['normalized_pub_number'].dropna())
    common_pub_numbers = test_pub_numbers & bdb_pub_numbers
    
    print(f"Common patent numbers: {len(common_pub_numbers)}")
    print(f"Unique in test: {len(test_pub_numbers)}")
    print(f"Unique in reference: {len(bdb_pub_numbers)}")
    
    merged_keys = set(zip(
        merged['normalized_pub_number'],
        merged['Ligand InChI Key'],
        merged['Sequence']
    ))
    
    test_keys = set(zip(
        df_test_renamed['normalized_pub_number'],
        df_test_renamed['Ligand InChI Key'],
        df_test_renamed['Sequence']
    ))
    
    bdb_keys = set(zip(
        df_bdb_renamed['normalized_pub_number'],
        df_bdb_renamed['Ligand InChI Key'],
        df_bdb_renamed['Sequence']
    ))
    
    # Bindings only in test (for shared patents)
    only_in_test = [
        key for key in test_keys
        if key[0] in common_pub_numbers and key not in merged_keys
    ]
    
    # Bindings only in reference (for shared patents)
    only_in_bdb = [
        key for key in bdb_keys
        if key[0] in common_pub_numbers and key not in merged_keys
    ]
    
    # Count full duplicates
    full_duplicates = df_test.duplicated(keep='first')
    full_duplicates_count = int(full_duplicates.sum())
    
    results = {
        'total_binding': len(df_test),
        'new_binding': len(only_in_test),
        'exact_match': len(matches),
        'no_exact_match': len(non_matches),
        'not_found_binding': len(only_in_bdb),
        'full_duplicates_count': full_duplicates_count,
        'common_patents': len(common_pub_numbers)
    }
    
    # Prepare data for detailed examples
    def get_metric_value(row, metric_col):
        """Get a metric value from a row"""
        val = row.get(metric_col, '')
        if pd.isna(val) or val == '':
            return 'N/A'
        return str(val)
    
    # Group mismatches by patent
    non_matches_by_patent = {}
    for idx, row in non_matches.iterrows():
        patent = str(row['normalized_pub_number'])
        if patent not in non_matches_by_patent:
            non_matches_by_patent[patent] = []
        non_matches_by_patent[patent].append(row)
    
    # Group new bindings by patent
    only_in_test_by_patent = {}
    for key in only_in_test:
        patent = str(key[0])
        if patent not in only_in_test_by_patent:
            only_in_test_by_patent[patent] = []
        # Find the matching row in the test table
        mask = (
            (df_test_renamed['normalized_pub_number'].astype(str) == key[0]) &
            (df_test_renamed['Ligand InChI Key'] == key[1]) &
            (df_test_renamed['Sequence'] == key[2])
        )
        matching_rows = df_test_renamed[mask]
        if len(matching_rows) > 0:
            only_in_test_by_patent[patent].append(matching_rows.iloc[0])
    
    # Group missing bindings by patent
    only_in_bdb_by_patent = {}
    for key in only_in_bdb:
        patent = str(key[0])
        if patent not in only_in_bdb_by_patent:
            only_in_bdb_by_patent[patent] = []
        # Find the matching row in the reference table
        mask = (
            (df_bdb_renamed['normalized_pub_number'].astype(str) == key[0]) &
            (df_bdb_renamed['Ligand InChI Key'] == key[1]) &
            (df_bdb_renamed['Sequence'] == key[2])
        )
        matching_rows = df_bdb_renamed[mask]
        if len(matching_rows) > 0:
            only_in_bdb_by_patent[patent].append(matching_rows.iloc[0])
    
    # Build detailed output
    detailed_output = []
    detailed_output.append("=" * 80)
    detailed_output.append("TABLE COMPARISON RESULTS WITH MAPPING")
    detailed_output.append("=" * 80)
    detailed_output.append("")
    detailed_output.append(f"Total bindings in test: {len(df_test)}")
    detailed_output.append(f"Total bindings in reference: {len(df_bdb)}")
    detailed_output.append(f"Common patent numbers: {len(common_pub_numbers)}")
    detailed_output.append("")
    detailed_output.append(f"New bindings (test only): {len(only_in_test)}")
    detailed_output.append(f"Exact metric matches: {len(matches)}")
    detailed_output.append(f"Inexact matches (keys matched, metrics differ): {len(non_matches)}")
    detailed_output.append(f"Missing bindings (reference only): {len(only_in_bdb)}")
    detailed_output.append(f"Duplicate count in test: {full_duplicates_count}")
    detailed_output.append("")
    
    # DETAILED EXAMPLES OF EXACT MATCHES (for contrast)
    if len(matches) > 0:
        detailed_output.append("=" * 80)
        detailed_output.append("EXAMPLES OF EXACT MATCHES (keys and metrics matched)")
        detailed_output.append("=" * 80)
        detailed_output.append("")
        
        # Group by patent
        matches_by_patent = {}
        for idx, row in matches.iterrows():
            patent = str(row['normalized_pub_number'])
            if patent not in matches_by_patent:
                matches_by_patent[patent] = []
            matches_by_patent[patent].append(row)
        
        # Show examples from different patents (max 2 patents, 1 example each)
        shown_patents = 0
        for patent in sorted(matches_by_patent.keys()):
            if shown_patents >= 2:
                break
            row = matches_by_patent[patent][0]  # 1 example per patent
            app_number = str(row['patent_number'])

            detailed_output.append(f"Patent: {patent} {app_number} ({len(matches_by_patent[patent])} exact matches)")
            detailed_output.append("-" * 80)
            detailed_output.append(f"  InChI Key: {row['Ligand InChI Key'][:50]}...")
            detailed_output.append(f"  Sequence: {row['Sequence'][:50]}...")
            detailed_output.append(f"  Matching metrics:")
            # Show only metrics present in both tables
            matching_metrics = get_matching_metrics(row)
            for test_col, bdb_col, val_test, val_bdb in matching_metrics:
                if abs(val_test - val_bdb) < 0.01:  # Exact match
                    detailed_output.append(f"    {test_col}: {val_test} (test) = {val_bdb} (reference)")
            detailed_output.append("")
            shown_patents += 1
        
        detailed_output.append("")
    
    # DETAILED EXAMPLES OF INEXACT MATCHES (keys matched, metrics differ)
    if len(non_matches) > 0:
        detailed_output.append("=" * 80)
        detailed_output.append("EXAMPLES OF INEXACT MATCHES (keys matched, metrics differ)")
        detailed_output.append("=" * 80)
        detailed_output.append("")
        
        # Show examples from different patents (max 3 patents, 2 examples each)
        shown_patents = 0
        for patent in sorted(non_matches_by_patent.keys()):
            if shown_patents >= 3:
                break
            rows = non_matches_by_patent[patent][:2]  # 2 examples per patent
            
            # Get app_number from the first row (merged table has _test and _bdb suffixes)
            app_number = ""
            if len(rows) > 0:
                app_number = str(rows[0].get('patent_number_test', rows[0].get('patent_number', '')))
                if app_number and app_number != 'nan' and app_number != '':
                    app_number = f" {app_number}"
                else:
                    # Try to find via mapping
                    for app_num, pat_num in patent_mapping.items():
                        if str(pat_num) == str(patent):
                            app_number = f" {app_num}"
                            break
            
            detailed_output.append(f"Patent: {patent} {app_number} ({len(non_matches_by_patent[patent])} mismatches)")
            detailed_output.append("-" * 80)
            
            for i, row in enumerate(rows, 1):
                detailed_output.append(f"\n  Example {i}:")
                detailed_output.append(f"    InChI Key: {row['Ligand InChI Key'][:50]}...")
                detailed_output.append(f"    Sequence: {row['Sequence'][:50]}...")
                
                # Collect only matching metrics (same type)
                matching_metrics = get_matching_metrics(row)
                
                if matching_metrics:
                    detailed_output.append(f"    Comparison of matching metric types:")
                    for test_col, bdb_col, val_test, val_bdb in matching_metrics:
                        diff = abs(val_test - val_bdb)
                        diff_pct = (diff / val_bdb * 100) if val_bdb != 0 else 0
                        detailed_output.append(f"      {test_col}:")
                        detailed_output.append(f"        TEST: {val_test}")
                        detailed_output.append(f"        REFERENCE: {val_bdb}")
                        detailed_output.append(f"        Difference: {diff:.2f} ({diff_pct:.1f}%)")
                else:
                    detailed_output.append(f"    WARNING: No matching metric types to compare!")
                    detailed_output.append(f"    TEST metrics:")
                    for metric in ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']:
                        val = get_metric_value(row, metric)
                        if val != 'N/A':
                            detailed_output.append(f"      {metric}: {val}")
                    detailed_output.append(f"    REFERENCE metrics:")
                    for metric, bdb_metric in [('Ki (nM)', 'bdb__ki (nm)'), ('IC50 (nM)', 'bdb__ic50 (nm)'), 
                                              ('Kd (nM)', 'bdb__kd (nm)'), ('EC50 (nM)', 'bdb__ec50 (nm)')]:
                        val = get_metric_value(row, bdb_metric)
                        if val != 'N/A':
                            detailed_output.append(f"      {metric}: {val}")
            
            detailed_output.append("")
            shown_patents += 1
        
        if len(non_matches_by_patent) > 3:
            detailed_output.append(f"... and {len(non_matches_by_patent) - 3} more patents with mismatches")
        detailed_output.append("")
    
    # DETAILED EXAMPLES OF NEW BINDINGS (test only)
    if len(only_in_test) > 0:
        detailed_output.append("=" * 80)
        detailed_output.append("EXAMPLES OF NEW BINDINGS (test only)")
        detailed_output.append("=" * 80)
        detailed_output.append("")
        
        # Show examples from different patents (max 3 patents, 2 examples each)
        shown_patents = 0
        for patent in sorted(only_in_test_by_patent.keys()):
            if shown_patents >= 3:
                break
            rows = only_in_test_by_patent[patent][:2]  # 2 examples per patent
            
            # Get app_number from the first row
            app_number = ""
            if len(rows) > 0:
                app_number = str(rows[0].get('patent_number', ''))
                if app_number and app_number != 'nan':
                    app_number = f" {app_number}"
            
            detailed_output.append(f"Patent: {patent} {app_number} ({len(only_in_test_by_patent[patent])} new bindings)")
            detailed_output.append("-" * 80)
            
            for i, row in enumerate(rows, 1):
                detailed_output.append(f"\n  Example {i}:")
                detailed_output.append(f"    InChI Key: {row['Ligand InChI Key'][:50]}...")
                detailed_output.append(f"    Sequence: {row['Sequence'][:50]}...")
                detailed_output.append(f"    Protein: {row.get('protein_target_name', 'N/A')}")
                detailed_output.append(f"    Metrics:")
                for metric in ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']:
                    val = get_metric_value(row, metric)
                    if val != 'N/A':
                        detailed_output.append(f"      {metric}: {val}")
            
            detailed_output.append("")
            shown_patents += 1
        
        if len(only_in_test_by_patent) > 3:
            detailed_output.append(f"... and {len(only_in_test_by_patent) - 3} more patents with new bindings")
        detailed_output.append("")
    
    # DETAILED EXAMPLES OF MISSING BINDINGS (reference only)
    if show_not_found_examples and len(only_in_bdb) > 0:
        detailed_output.append("=" * 80)
        detailed_output.append("EXAMPLES OF MISSING BINDINGS (reference only)")
        detailed_output.append("=" * 80)
        detailed_output.append("")
        
        # Show examples from different patents (max 3 patents, 2 examples each)
        shown_patents = 0
        for patent in sorted(only_in_bdb_by_patent.keys()):
            if shown_patents >= 3:
                break
            rows = only_in_bdb_by_patent[patent][:2]  # 2 examples per patent
            
            # Get app_number from mapping (reverse lookup)
            app_number = ""
            if len(rows) > 0:
                # Try to find application_number via reverse mapping
                for app_num, pat_num in patent_mapping.items():
                    if str(pat_num) == str(patent):
                        app_number = f" {app_num}"
                        break
                # If not found via mapping, try data columns
                if not app_number:
                    app_number_val = rows[0].get('bdb__patent_number', rows[0].get('patent_number', ''))
                    if app_number_val and str(app_number_val) != 'nan':
                        app_number = f" {app_number_val}"
            
            detailed_output.append(f"Patent: {patent} {app_number} ({len(only_in_bdb_by_patent[patent])} missing bindings)")
            detailed_output.append("-" * 80)
            
            for i, row in enumerate(rows, 1):
                detailed_output.append(f"\n  Example {i}:")
                detailed_output.append(f"    InChI Key: {row['Ligand InChI Key'][:50]}...")
                detailed_output.append(f"    Sequence: {row['Sequence'][:50]}...")
                detailed_output.append(f"    Protein: {row.get('bdb__target_name', 'N/A')}")
                detailed_output.append(f"    Metrics:")
                for metric, bdb_metric in [('Ki (nM)', 'bdb__ki (nm)'), ('IC50 (nM)', 'bdb__ic50 (nm)'), 
                                          ('Kd (nM)', 'bdb__kd (nm)'), ('EC50 (nM)', 'bdb__ec50 (nm)')]:
                    val = get_metric_value(row, bdb_metric)
                    if val != 'N/A':
                        detailed_output.append(f"      {metric}: {val}")
            
            detailed_output.append("")
            shown_patents += 1
        
        if len(only_in_bdb_by_patent) > 3:
            detailed_output.append(f"... and {len(only_in_bdb_by_patent) - 3} more patents with missing bindings")
        detailed_output.append("")
    
    # Top patents from only_in_test
    if show_top_new_patents and len(only_in_test) > 0:
        detailed_output.append("=" * 80)
        detailed_output.append("TOP PATENTS FROM NEW BINDINGS (only_in_test)")
        detailed_output.append("=" * 80)
        patent_counts = Counter([key[0] for key in only_in_test])
        top_patents = patent_counts.most_common(10)
        
        detailed_output.append(f"Total unique patents with new bindings: {len(patent_counts)}")
        detailed_output.append("")
        detailed_output.append("Top 10 patents by new binding count:")
        for i, (pub_num, count) in enumerate(top_patents, 1):
            detailed_output.append(f"{i}. {pub_num}: {count} new bindings")
        detailed_output.append("")
    
    detailed_output.append("=" * 80)
    detailed_output.append("SUMMARY STATISTICS")
    detailed_output.append("=" * 80)
    detailed_output.append(f"Total binding: {results['total_binding']}")
    detailed_output.append(f"New binding: {results['new_binding']}")
    detailed_output.append(f"Exact match by metric: {results['exact_match']}")
    detailed_output.append(f"No Exact match by metric: {results['no_exact_match']}")
    detailed_output.append(f"Not found binding: {results['not_found_binding']}")
    detailed_output.append(f"Duplicates count: {results['full_duplicates_count']}")
    detailed_output.append(f"Common patents: {results['common_patents']}")
    
    return results, "\n".join(detailed_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare pipeline results with the BindingDB reference table using patent mapping"
    )
    parser.add_argument(
        "--test-table",
        type=str,
        required=True,
        help="Path to the test table (CSV)"
    )
    parser.add_argument(
        "--bdb-table",
        type=str,
        required=True,
        help="Path to the reference table (Parquet)"
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        required=True,
        help="Path to the mapping file (CSV or JSONL, e.g. curated_data/patent_mapping.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results (optional)"
    )
    parser.add_argument(
        "--show-not-found-examples",
        action='store_true',
        help="Show example rows from only_in_bdb"
    )
    parser.add_argument(
        "--show-top-new-patents",
        action='store_true',
        help="Show top patents from only_in_test"
    )
    
    args = parser.parse_args()
    
    try:
        results, detailed_output = run_comparison_with_mapping(
            test_table_path=args.test_table,
            bdb_table_path=args.bdb_table,
            mapping_file=args.mapping_file,
            show_not_found_examples=args.show_not_found_examples,
            show_top_new_patents=args.show_top_new_patents
        )
        
        print(detailed_output)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(detailed_output)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

