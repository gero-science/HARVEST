"""
UniRef90-aware Novel Protein Reclassification Report
=====================================================

Queries UniProt API for UniRef90 clusters of "new" proteins (in harvest but
not in BDB by exact accession), checks if any cluster member is in BDB, and
reports reclassifications.

Usage:
    python cluster_split/compare_novel_proteins.py \
      --harvest ../../projects/patents/6k_nov21/final_v1/final_v10_clean_smiles.parquet \
      --bdb ../../projects/patents/full_bdb_chembl_fix.parquet \
      --output cluster_split/uniref90_reclassification_report.csv
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

UNIREF_SEARCH_URL = "https://rest.uniprot.org/uniref/search"
UNIREF_ENTRY_URL = "https://rest.uniprot.org/uniref"

REQUEST_DELAY = 0.5  # seconds between API requests


def load_protein_sets(harvest_path, bdb_path):
    """Load harvest/BDB parquets, return (harvest_proteins, bdb_proteins) sets."""
    df_6k = pd.read_parquet(harvest_path)
    df_6k = df_6k.rename(columns={'Target accession': 'uniprot_acc'})
    df_6k['uniprot_acc_1st'] = [
        str(x).split(';')[0] for x in df_6k['uniprot_acc']
    ]

    df_bd = pd.read_parquet(bdb_path)

    proteins_harvest = set(df_6k['uniprot_acc_1st'].dropna().unique())
    proteins_harvest.discard('')
    proteins_bdb = set(df_bd['uniprot_acc'].dropna().unique())

    return proteins_harvest, proteins_bdb


def _parse_members_from_result(result):
    """Extract member accessions from a single UniRef search result entry.

    With fields=id,members the API returns members as a flat list of accession strings.
    """
    cluster_id = result.get('id', '')
    members = set(result.get('members', []))
    return cluster_id, members


def query_uniref90_batch(accessions, session):
    """Query UniRef90 clusters for a batch of accessions via search endpoint.

    Returns dict: accession -> {cluster_id, members, member_count}
    """
    # Build query: (uniprot_id:P1 OR uniprot_id:P2 ...) AND identity:0.9
    id_parts = ' OR '.join(f'uniprot_id:{acc}' for acc in accessions)
    query = f'({id_parts}) AND identity:0.9'

    params = {
        'query': query,
        'fields': 'id,members',
        'format': 'json',
        'size': 500,  # more than enough for batch of 10
    }

    resp = session.get(UNIREF_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Map each accession to its cluster
    results = {}
    for entry in data.get('results', []):
        cluster_id, members = _parse_members_from_result(entry)
        member_count = entry.get('memberCount', len(members))
        # Figure out which of our queried accessions belong to this cluster
        for acc in accessions:
            if acc in members:
                results[acc] = {
                    'cluster_id': cluster_id,
                    'members': members,
                    'member_count': member_count,
                }

    return results


def fetch_full_members(cluster_id, session):
    """Fetch all members for a cluster via direct endpoint."""
    url = f"{UNIREF_ENTRY_URL}/{cluster_id}"
    params = {'fields': 'id,members', 'format': 'json'}

    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    _, members = _parse_members_from_result(data)
    return members


def check_new_proteins(new_proteins, bdb_proteins, batch_size=10, cache_path=None):
    """Query API for all new proteins, return reclassification results.

    Returns list of dicts with keys:
        uniprot_acc, uniref90_cluster, cluster_member_count,
        matching_bdb_proteins, status
    """
    # Load cache if available
    cache = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded cache with {len(cache)} entries from {cache_path}")

    sorted_proteins = sorted(new_proteins)
    # Split into proteins that need querying vs cached
    to_query = [p for p in sorted_proteins if p not in cache]
    print(f"Total new proteins: {len(sorted_proteins)}, cached: {len(sorted_proteins) - len(to_query)}, to query: {len(to_query)}")

    session = requests.Session()
    session.headers.update({'Accept': 'application/json'})

    # Process in batches
    batches = [to_query[i:i + batch_size] for i in range(0, len(to_query), batch_size)]

    for batch in tqdm(batches, desc="Querying UniRef90"):
        try:
            batch_results = query_uniref90_batch(batch, session)
        except requests.exceptions.RequestException as e:
            print(f"\n  API error for batch starting with {batch[0]}: {e}")
            # Mark all in batch as failed
            for acc in batch:
                if acc not in cache:
                    cache[acc] = {'cluster_id': 'API_ERROR', 'members': [], 'member_count': 0}
            time.sleep(REQUEST_DELAY)
            continue

        # Check for proteins that need full member fetch
        for acc in batch:
            if acc in batch_results:
                info = batch_results[acc]
                members = info['members']
                member_count = info['member_count']

                # Check if any member is in BDB
                bdb_matches = members & bdb_proteins
                if not bdb_matches and member_count > len(members):
                    # Got truncated results, fetch full members
                    try:
                        time.sleep(REQUEST_DELAY)
                        full_members = fetch_full_members(info['cluster_id'], session)
                        bdb_matches = full_members & bdb_proteins
                        members = full_members
                        member_count = len(full_members)
                    except requests.exceptions.RequestException as e:
                        print(f"\n  Failed to fetch full members for {info['cluster_id']}: {e}")

                cache[acc] = {
                    'cluster_id': info['cluster_id'],
                    'members': sorted(members),
                    'member_count': member_count,
                    'bdb_matches': sorted(bdb_matches),
                }
            else:
                # No cluster found for this accession
                cache[acc] = {'cluster_id': 'NOT_FOUND', 'members': [], 'member_count': 0, 'bdb_matches': []}

        time.sleep(REQUEST_DELAY)

    # Save cache
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"Saved cache with {len(cache)} entries to {cache_path}")

    # Build results for ALL new proteins
    results = []
    for acc in sorted_proteins:
        info = cache.get(acc, {})
        cluster_id = info.get('cluster_id', 'NOT_FOUND')

        # Recompute BDB matches from cached members (in case bdb_proteins changed)
        members = set(info.get('members', []))
        bdb_matches = sorted(members & bdb_proteins) if members else info.get('bdb_matches', [])

        member_count = info.get('member_count', 0)

        if cluster_id in ('NOT_FOUND', 'API_ERROR'):
            status = 'no_cluster_found'
        elif bdb_matches:
            status = 'reclassified_to_overlap'
        else:
            status = 'remains_new'

        results.append({
            'uniprot_acc': acc,
            'uniref90_cluster': cluster_id,
            'cluster_member_count': member_count,
            'matching_bdb_proteins': ';'.join(bdb_matches),
            'status': status,
        })

    return results


def save_report(results, output_path):
    """Save CSV report."""
    df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved report: {output_path} ({len(df)} rows)")
    return df


def print_summary(proteins_harvest, proteins_bdb, new_proteins, overlap_proteins, df_report):
    """Print summary statistics."""
    n_reclassified = (df_report['status'] == 'reclassified_to_overlap').sum()
    n_remains_new = (df_report['status'] == 'remains_new').sum()
    n_no_cluster = (df_report['status'] == 'no_cluster_found').sum()

    print(f"\n{'='*50}")
    print(f"Harvest: {len(proteins_harvest):,} | BDB: {len(proteins_bdb):,}")
    print(f"New (exact): {len(new_proteins):,} | Overlap (exact): {len(overlap_proteins):,}")
    print(f"\nUniRef90 reclassification:")
    print(f"  Queried: {len(df_report):,} proteins")
    print(f"  No cluster found: {n_no_cluster:,}")
    print(f"  Remains new: {n_remains_new:,}")
    print(f"  Reclassified to overlap: {n_reclassified:,}")
    print(f"\nFinal counts:")
    print(f"  New: {n_remains_new + n_no_cluster:,} | Overlap: {len(overlap_proteins) + n_reclassified:,}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='UniRef90-aware novel protein reclassification report')
    parser.add_argument('--harvest',
                        default='../../projects/patents/6k_nov21/final_v1/final_v10_clean_smiles.parquet',
                        help='Path to harvest parquet file')
    parser.add_argument('--bdb',
                        default='../../projects/patents/full_bdb_chembl_fix.parquet',
                        help='Path to BDB parquet file')
    parser.add_argument('--output',
                        default='cluster_split/uniref90_reclassification_report.csv',
                        help='Output CSV path')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of proteins per API search request (default: 10)')
    parser.add_argument('--cache',
                        default=None,
                        help='Path to JSON cache file (saves/loads API results)')
    args = parser.parse_args()

    print("Loading data...")
    proteins_harvest, proteins_bdb = load_protein_sets(args.harvest, args.bdb)

    new_proteins = proteins_harvest - proteins_bdb
    overlap_proteins = proteins_harvest & proteins_bdb

    print(f"Harvest proteins: {len(proteins_harvest):,}")
    print(f"BDB proteins:     {len(proteins_bdb):,}")
    print(f"New (exact):      {len(new_proteins):,}")
    print(f"Overlap (exact):  {len(overlap_proteins):,}")

    print(f"\nQuerying UniRef90 for {len(new_proteins):,} new proteins...")
    results = check_new_proteins(
        new_proteins, proteins_bdb,
        batch_size=args.batch_size,
        cache_path=args.cache,
    )

    df_report = save_report(results, args.output)
    print_summary(proteins_harvest, proteins_bdb, new_proteins, overlap_proteins, df_report)


if __name__ == '__main__':
    main()
