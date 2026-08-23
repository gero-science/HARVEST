#!/usr/bin/env python3
"""
Script to filter parquet files for fragments and Markush structures.

TWO FILTERS:

1. FRAGMENT FILTER (by SMILES):
   - Appear more than N times (--min-count, default: 5)
   - Contain <= M heavy atoms (--heavy-atoms, default: 15)
   -> Removes ALL records from patents with such SMILES

2. TRIPLET FILTER (ligand + assay + patent):
   - Triplet appears >= K times (--triplet-count, default: 5)
   -> Removes only RECORDS with frequent triplets (not whole patents)
   -> Preserves experimental data (different assay_id)

Usage:
    python filter_fragments.py input.parquet output.parquet
    python filter_fragments.py input.parquet output.parquet --triplet-count 0  # without triplet filter
    python filter_fragments.py input.parquet output.parquet --triplet-count 10 # less aggressive
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def get_heavy_atoms(smiles: str) -> int | None:
    """Return the number of heavy atoms in a molecule."""
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol else None
    except:
        return None


def collect_all_smiles_chunked(input_path: Path, chunk_size: int = 100000) -> Counter:
    """Collect SMILES from the clean_smiles column, reading the file in chunks."""
    import pyarrow.parquet as pq
    
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    smiles_counter = Counter()
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size, 
                                                 columns=['clean_smiles']),
                      total=(total_rows // chunk_size) + 1,
                      desc="   Collecting SMILES"):
        df_chunk = batch.to_pandas()
        
        if 'clean_smiles' in df_chunk.columns:
            clean_smiles = df_chunk['clean_smiles'].dropna()
            clean_smiles = clean_smiles[clean_smiles != '']
            smiles_counter.update(clean_smiles.tolist())
    
    return smiles_counter


def create_blacklist(smiles_counts: Counter, heavy_threshold: int, count_threshold: int) -> set:
    """
    Build a fragment blacklist.
    
    Criteria: count > count_threshold AND heavy_atoms <= heavy_threshold
    """
    # Select frequent SMILES
    frequent_smiles = {s: c for s, c in smiles_counts.items() if c > count_threshold}
    
    print(f"  Frequent SMILES (count > {count_threshold}): {len(frequent_smiles):,}")
    
    # Compute heavy atoms for frequent SMILES
    print("  Computing heavy atoms...")
    blacklist = set()
    
    for smi in tqdm(frequent_smiles.keys(), desc="  Heavy atoms"):
        heavy = get_heavy_atoms(smi)
        if heavy is not None and heavy <= heavy_threshold:
            blacklist.add(smi)
    
    return blacklist


def collect_triplets_chunked(input_path: Path, chunk_size: int = 100000) -> Counter:
    """Collect triplets (clean_smiles, assay_id, patent_number)."""
    import pyarrow.parquet as pq
    
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    triplet_counter = Counter()
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size,
                                                 columns=['clean_smiles', 'assay_id', 'patent_number']),
                      total=(total_rows // chunk_size) + 1,
                      desc="   Collecting triplets"):
        df_chunk = batch.to_pandas()
        triplets = (df_chunk['clean_smiles'].fillna('') + '|||' + 
                   df_chunk['assay_id'].fillna('') + '|||' +
                   df_chunk['patent_number'].fillna(''))
        triplet_counter.update(triplets.tolist())
    
    return triplet_counter


def collect_patents_with_fragments(input_path: Path, blacklist: set, chunk_size: int = 100000) -> set:
    """Collect patent_number values for records with blacklisted SMILES."""
    import pyarrow.parquet as pq
    
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    patents = set()
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size,
                                                 columns=['patent_number', 'clean_smiles']),
                      total=(total_rows // chunk_size) + 1,
                      desc="   Finding patents"):
        df_chunk = batch.to_pandas()
        mask = df_chunk['clean_smiles'].isin(blacklist)
        patents.update(df_chunk.loc[mask, 'patent_number'].dropna().unique())
    
    return patents


def filter_dataframe_chunked(input_path: Path, output_path: Path, patents_to_remove: set, 
                             bad_triplets: set = None, chunk_size: int = 100000):
    """
    Filter a parquet file in chunks, writing the result directly to disk.
    
    Two filtering modes:
    1. By patent_number - removes ALL records from patents with fragments
    2. By triplets (assay) - removes only specific records with frequent triplets
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Read metadata
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    removed_count = 0
    removed_by_patent = 0
    removed_by_triplet = 0
    total_written = 0
    writer = None
    
    print(f"   Processing in chunks of {chunk_size:,} records...")
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), 
                      total=(total_rows // chunk_size) + 1,
                      desc="   Filtering"):
        df_chunk = batch.to_pandas()
        original_len = len(df_chunk)
        
        # Filter 1: by patent_number (removes whole patents)
        mask_patent = df_chunk['patent_number'].isin(patents_to_remove)
        
        # Filter 2: by assay triplets (removes only records)
        if bad_triplets:
            triplet = (df_chunk['clean_smiles'].fillna('') + '|||' + 
                      df_chunk['assay_id'].fillna('') + '|||' +
                      df_chunk['patent_number'].fillna(''))
            mask_triplet = triplet.isin(bad_triplets)
        else:
            mask_triplet = pd.Series(False, index=df_chunk.index)
        
        # Combine masks
        mask = mask_patent | mask_triplet
        
        removed_by_patent += (mask_patent & ~mask_triplet).sum()
        removed_by_triplet += mask_triplet.sum()
        
        df_filtered = df_chunk[~mask]
        removed_count += original_len - len(df_filtered)
        
        # Write chunk to file
        if len(df_filtered) > 0:
            table = pa.Table.from_pandas(df_filtered, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_written += len(df_filtered)
    
    if writer:
        writer.close()
    
    return total_written, removed_count, removed_by_patent, removed_by_triplet


def main():
    parser = argparse.ArgumentParser(
        description="Filter parquet files for fragments (R-groups/Markush structures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_fragments.py data.parquet filtered.parquet
  python filter_fragments.py data.parquet filtered.parquet --triplet-count 0   # without triplet filter
  python filter_fragments.py data.parquet filtered.parquet --triplet-count 10  # less aggressive triplet filter
  python filter_fragments.py data.parquet filtered.parquet --heavy-atoms 12 --min-count 3
        """
    )
    
    parser.add_argument("input", type=Path, help="Path to input parquet file")
    parser.add_argument("output", type=Path, help="Path to output parquet file")
    parser.add_argument(
        "--heavy-atoms", type=int, default=15,
        help="Heavy atom threshold (SMILES with <= this value are filtered, default: 15)"
    )
    parser.add_argument(
        "--min-count", type=int, default=5,
        help="Minimum repetition count for filtering (SMILES with > this value are filtered, default: 5)"
    )
    parser.add_argument(
        "--triplet-count", type=int, default=5,
        help="Threshold for ligand+assay+patent triplets (default: 5, 0 = disable). Removes RECORDS, not patents."
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: file {args.input} not found", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60)
    print("FRAGMENT AND FREQUENT TRIPLET FILTERING")
    print("=" * 60)
    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print(f"Parameters:")
    print(f"  Heavy atoms threshold: {args.heavy_atoms}")
    print(f"  Min count threshold:   {args.min_count}")
    print(f"  Triplet count threshold: {args.triplet_count} {'(disabled)' if args.triplet_count == 0 else ''}")
    
    # Load metadata
    import pyarrow.parquet as pq
    
    print("\n1. Reading metadata...")
    parquet_file = pq.ParquetFile(args.input)
    original_len = parquet_file.metadata.num_rows
    schema_names = parquet_file.schema.names
    print(f"   Records in file: {original_len:,}")
    
    # Validate columns
    has_clean = 'clean_smiles' in schema_names
    has_patent = 'patent_number' in schema_names
    has_assay = 'assay_id' in schema_names
    print(f"   Column 'clean_smiles':  {'✓' if has_clean else '✗'}")
    print(f"   Column 'patent_number': {'✓' if has_patent else '✗'}")
    print(f"   Column 'assay_id':      {'✓' if has_assay else '✗'}")
    
    if not has_clean:
        print("Error: column 'clean_smiles' not found", file=sys.stderr)
        sys.exit(1)
    
    if not has_patent:
        print("Error: column 'patent_number' not found", file=sys.stderr)
        sys.exit(1)
    
    if args.triplet_count > 0 and not has_assay:
        print("Error: column 'assay_id' not found (required for triplet filter)", file=sys.stderr)
        print("Use --triplet-count 0 to disable triplet filtering", file=sys.stderr)
        sys.exit(1)
    
    # Collect SMILES and count frequencies (chunked)
    print("\n2. Collecting SMILES and counting frequencies (chunked)...")
    smiles_counts = collect_all_smiles_chunked(args.input)
    total_smiles = sum(smiles_counts.values())
    print(f"   Total SMILES (with duplicates): {total_smiles:,}")
    print(f"   Unique SMILES: {len(smiles_counts):,}")
    
    # Build blacklist
    print("\n3. Creating blacklist...")
    blacklist = create_blacklist(smiles_counts, args.heavy_atoms, args.min_count)
    print(f"   SMILES in blacklist: {len(blacklist):,}")
    
    if len(blacklist) == 0:
        print("\n   Blacklist is empty; no filtering needed.")
        import shutil
        shutil.copy(args.input, args.output)
        print(f"\n   File copied: {args.output}")
        return
    
    # Find patents with fragments
    print("\n4. Finding patents with fragments...")
    patents_fragments = collect_patents_with_fragments(args.input, blacklist)
    print(f"   Patents with fragments: {len(patents_fragments):,}")
    
    # Triplet filter (if enabled)
    bad_triplets = set()
    if args.triplet_count > 0:
        print(f"\n5. Collecting triplets (ligand+assay+patent)...")
        triplet_counts = collect_triplets_chunked(args.input)
        print(f"   Total unique triplets: {len(triplet_counts):,}")
        
        # Build bad_triplets set
        bad_triplets = {t for t, c in triplet_counts.items() if c >= args.triplet_count}
        print(f"   Triplets with count >= {args.triplet_count}: {len(bad_triplets):,}")
    
    # Check whether there is anything to filter
    patents_to_remove = patents_fragments  # From fragment filter only
    print(f"\n   Patents to remove (fragments): {len(patents_to_remove):,}")
    print(f"   Triplets to remove (assay):    {len(bad_triplets):,}")
    
    if len(patents_to_remove) == 0 and len(bad_triplets) == 0:
        print("\n   Nothing to filter.")
        import shutil
        shutil.copy(args.input, args.output)
        print(f"\n   File copied: {args.output}")
        return
    
    # Filter (chunked for memory efficiency, write directly to file)
    step = 6 if args.triplet_count > 0 else 5
    print(f"\n{step}. Filtering and saving...")
    total_written, removed, removed_by_patent, removed_by_triplet = filter_dataframe_chunked(
        args.input, args.output, patents_to_remove, bad_triplets if args.triplet_count > 0 else None
    )
    removed_pct = 100 * removed / original_len
    
    print(f"\n   Before filtering:  {original_len:,}")
    print(f"   After filtering:   {total_written:,}")
    print(f"   Records removed:   {removed:,} ({removed_pct:.2f}%)")
    if args.triplet_count > 0:
        print(f"     - by patents (fragments): {removed_by_patent:,}")
        print(f"     - by triplets (assay):      {removed_by_triplet:,}")
    print(f"   File saved: {args.output}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input file:           {original_len:,} records")
    print(f"Output file:          {total_written:,} records")
    print(f"Blacklist SMILES:     {len(blacklist):,}")
    print(f"Patents (fragments):  {len(patents_fragments):,}")
    if args.triplet_count > 0:
        print(f"Bad triplets (assay): {len(bad_triplets):,}")
    print(f"Records removed:      {removed:,} ({removed_pct:.2f}%)")


if __name__ == '__main__':
    main()

