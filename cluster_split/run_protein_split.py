"""
Per-Protein Clustering & Split Script
======================================

Processes each protein individually:
  - NEW proteins (harvest-only): cluster and compute scaffold stats
  - OVERLAP proteins (harvest + BDB): run full split_clusters pipeline

Usage:
    python cluster_split/run_protein_split.py --limit 50 --output-dir cluster_split/output_debug
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster_split.split_clusters import split_clusters, save_split_results
from cluster_split.compare_novel_proteins import check_new_proteins


def load_data(harvest_path, bdb_path):
    """Load harvest and BDB parquets, determine protein sets."""
    df_6k = pd.read_parquet(harvest_path)
    df_6k = df_6k.rename(columns={'Target accession': 'uniprot_acc'})
    df_6k['uniprot_acc_1st'] = [
        str(x).split(';')[0] for x in df_6k['uniprot_acc']
    ]

    df_bd = pd.read_parquet(bdb_path)

    proteins_harvest = set(df_6k['uniprot_acc_1st'].dropna().unique())
    proteins_harvest.discard('')
    proteins_bdb = set(df_bd['uniprot_acc'].dropna().unique())

    new_proteins = proteins_harvest - proteins_bdb
    overlap_proteins = proteins_harvest & proteins_bdb

    print(f"Harvest proteins: {len(proteins_harvest)}")
    print(f"BDB proteins:     {len(proteins_bdb)}")
    print(f"New (harvest-only): {len(new_proteins)}")
    print(f"Overlap:            {len(overlap_proteins)}")

    return df_6k, df_bd, new_proteins, overlap_proteins


def reclassify_via_uniref90(new_proteins, proteins_bdb, report_path=None,
                            cache_path=None):
    """Reclassify 'new' proteins that share a UniRef90 cluster with BDB proteins.

    Returns:
        bdb_acc_map: dict mapping harvest accession -> list of matching BDB accessions
        remaining_new: set of proteins that remain 'new'
        reclassified: set of proteins moved to 'overlap'
    """
    if report_path:
        print(f"\nLoading UniRef90 reclassification report: {report_path}")
        df_report = pd.read_csv(report_path)
        df_recl = df_report[df_report['status'] == 'reclassified_to_overlap']
        bdb_acc_map = {}
        reclassified = set()
        for _, row in df_recl.iterrows():
            acc = row['uniprot_acc']
            if acc in new_proteins:
                bdb_accs = [x for x in str(row['matching_bdb_proteins']).split(';') if x]
                bdb_acc_map[acc] = bdb_accs
                reclassified.add(acc)
    else:
        print(f"\nQuerying UniProt API for UniRef90 clusters of {len(new_proteins)} new proteins...")
        results = check_new_proteins(new_proteins, proteins_bdb, cache_path=cache_path)
        bdb_acc_map = {}
        reclassified = set()
        for entry in results:
            if entry['status'] == 'reclassified_to_overlap':
                acc = entry['uniprot_acc']
                bdb_accs = [x for x in entry['matching_bdb_proteins'].split(';') if x]
                bdb_acc_map[acc] = bdb_accs
                reclassified.add(acc)

    remaining_new = new_proteins - reclassified
    print(f"UniRef90 reclassification: {len(reclassified)} proteins moved new -> overlap")
    return bdb_acc_map, remaining_new, reclassified


def detect_smiles_col(df, candidates=('clean_smiles', 'smiles', 'Ligand SMILES')):
    """Return the first matching SMILES column name."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")


def compute_new_protein_stats(result):
    """Compute stats dict from split_clusters result for a single-source protein."""
    clusters = result['clusters']
    cluster_sizes = np.array([len(c) for c in clusters])
    n_compounds = int(cluster_sizes.sum())
    n_clusters = len(clusters)
    big_mask = cluster_sizes >= 10

    return {
        'n_compounds': n_compounds,
        'n_clusters': n_clusters,
        'scaffold_ratio': n_clusters / n_compounds if n_compounds > 0 else 0,
        'n_big_clusters': int(big_mask.sum()),
        'ave_clust_size': float(np.mean(cluster_sizes)) if n_clusters > 0 else 0,
        'ave_big_clust_size': float(np.mean(cluster_sizes[big_mask])) if big_mask.any() else 0,
    }


def compute_overlap_stats(result):
    """Compute overlap stats from split_clusters result for a two-source protein.

    """
    df_valid = result['df_valid']
    clusters = result['clusters']
    final_labels = result['final_cluster_labels']
    # cluster_sizes = np.array([len(c) for c in clusters])

    # TODO
    n_harvest = int((df_valid['source_label'] == 'A').sum())
    n_bdb = int((df_valid['source_label'] == 'B').sum())
    n_harvest_only = int((df_valid['final_label'].isin(['A', 'C'])).sum())  # number of unique compounds in harvest
    n_bdb_only = int((df_valid['final_label'] == 'B').sum())  # number of unique compounds in BDB
    n_compound_overlap = int((df_valid['final_label'].isin(['CB'])).sum())  # number of shared compounds in boader clusters
    n_total_clusters = len(clusters)

    # Final label distribution
    n_label_A = int(sum(1 for v in final_labels.values() if v == 'A'))
    n_label_B = int(sum(1 for v in final_labels.values() if v == 'B'))
    n_label_C = int(sum(1 for v in final_labels.values() if v in ['C', 'CoCB']))

    # big_mask = cluster_sizes >= 10

    return {
        'n_harvest_only_compounds': n_harvest_only,
        'n_bdb_only_compounds': n_bdb_only,
        'n_harvest_compounds': n_harvest,
        'n_bdb_compounds': n_bdb,
        'n_compound_overlap': n_compound_overlap,
        'n_total_clusters': n_total_clusters,
        'n_harvest_only_clusters': n_label_A,
        'n_bdb_only_clusters': n_label_B,
        'n_shared_clusters': n_label_C,  # includes boarder clusters
        # 'n_big_clusters': int(big_mask.sum()),
        # 'ave_clust_size': float(np.mean(cluster_sizes)) if n_total_clusters > 0 else 0,
        # 'ave_big_clust_size': float(np.mean(cluster_sizes[big_mask])) if big_mask.any() else 0,
    }


def process_new_proteins(df_6k, new_proteins, smiles_col, splits_dir,
                         cluster_threshold, fps_type, clustering_method,
                         ligand_threshold=None):
    """Process harvest-only proteins: cluster and collect stats."""
    stats_rows = []
    for protein in tqdm(sorted(new_proteins), desc="New proteins"):
        smiles = df_6k[df_6k['uniprot_acc_1st'] == protein][smiles_col].dropna().unique()
        if len(smiles) < 2:
            stats_rows.append({
                'protein': protein,
                'n_compounds': len(smiles),
                'n_clusters': len(smiles),
                'scaffold_ratio': 1.0,
                'n_big_clusters': 0,
                'ave_clust_size': 1.0 if len(smiles) else 0,
                'ave_big_clust_size': 0,
            })
            continue

        df_in = pd.DataFrame({
            'clean_smiles': smiles,
            'label': 'HARVEST',
        })

        try:
            result = split_clusters(
                df_in,
                smiles_key='clean_smiles',
                label_key='label',
                label_map={'HARVEST': 'A'},
                cluster_threshold=cluster_threshold,
                ligand_threshold=ligand_threshold,
                fps_type=fps_type,
                clustering_method=clustering_method,
            )
        except Exception as e:
            print(f"  ERROR processing {protein}: {e}")
            stats_rows.append({'protein': protein, 'error': str(e)})
            continue

        save_split_results(result['df_valid'],
                           splits_dir / f'split_results_{protein}.csv')

        row = compute_new_protein_stats(result)
        row['protein'] = protein
        stats_rows.append(row)

    return pd.DataFrame(stats_rows)


def process_overlap_proteins(df_6k, df_bd, overlap_proteins, smiles_col_harvest,
                             smiles_col_bdb, splits_dir, cluster_threshold,
                             fps_type, clustering_method,
                             ligand_threshold=None, max_jumps=2,
                             relabel_from='A', bdb_acc_map=None):
    """Process overlap proteins: full split_clusters pipeline."""
    if bdb_acc_map is None:
        bdb_acc_map = {}
    stats_rows = []
    for protein in tqdm(sorted(overlap_proteins), desc="Overlap proteins"):
        smiles_harvest = list(
            df_6k[df_6k['uniprot_acc_1st'] == protein][smiles_col_harvest].dropna().unique()
        )
        bdb_accs = [protein] + bdb_acc_map.get(protein, [])
        smiles_bdb = list(
            df_bd[df_bd['uniprot_acc'].isin(bdb_accs)][smiles_col_bdb].dropna().unique()
        )

        if len(smiles_harvest) + len(smiles_bdb) < 2:
            stats_rows.append({
                'protein': protein,
                'n_harvest': len(smiles_harvest),
                'n_bdb': len(smiles_bdb),
                'n_total_clusters': 0,
                'skipped': True,
            })
            continue

        # Compute compound overlap BEFORE dedup (shared SMILES between sources)
        compound_overlap = len(set(smiles_harvest) & set(smiles_bdb))

        # Build combined DataFrame with unique SMILES
        df_h = pd.DataFrame({'clean_smiles': smiles_harvest, 'label': 'HARVEST'})
        df_b = pd.DataFrame({'clean_smiles': smiles_bdb, 'label': 'BDB'})
        df_in = pd.concat([df_h, df_b], ignore_index=True)  # keep duplicates from harvest and bdb

        try:
            result = split_clusters(
                df_in,
                smiles_key='clean_smiles',
                label_key='label',
                label_map={'HARVEST': 'A', 'BDB': 'B'},
                cluster_threshold=cluster_threshold,
                ligand_threshold=ligand_threshold,
                max_jumps=max_jumps,
                fps_type=fps_type,
                clustering_method=clustering_method,
                relabel_from=relabel_from,
            )
        except Exception as e:
            print(f"  ERROR processing {protein}: {e}")
            stats_rows.append({'protein': protein, 'error': str(e)})
            continue

        # Save only HARVEST rows (source_label 'A') to per-protein CSV
        df_harvest_only = result['df_valid'][result['df_valid']['source_label'] == 'A']
        save_split_results(df_harvest_only,
                           splits_dir / f'split_results_{protein}.csv')

        df_bdb_only = result['df_valid'][result['df_valid']['source_label'] == 'B']
        save_split_results(df_bdb_only,
                           splits_dir / 'BDB_SPLIT' / f'split_results_{protein}.csv')


        row = compute_overlap_stats(result)
        row['protein'] = protein
        stats_rows.append(row)

    return pd.DataFrame(stats_rows)


def main():
    parser = argparse.ArgumentParser(
        description='Per-protein clustering and split analysis')
    parser.add_argument('--harvest', default='../../projects/patents/6k_nov21/final_v1/final_v10_clean_smiles.parquet',
                        help='Path to harvest parquet file')
    parser.add_argument('--bdb', default='../../projects/patents/full_bdb_chembl_fix.parquet',
                        help='Path to BDB parquet file (must have uniprot_acc and smiles columns)')
    parser.add_argument('--output-dir', default='cluster_split/output',
                        help='Output directory for all results')
    parser.add_argument('--cluster-threshold', type=float, default=0.225,
                        help='Tanimoto distance cutoff (default: 0.225)')
    parser.add_argument('--ligand-threshold', type=float, default=0.2,
                        help='Ligand similarity threshold (default: 0.2)')
    parser.add_argument('--max-jumps', type=int, default=2,
                        help='Max graph hops for relabelling (default: 2)')
    parser.add_argument('--relabel-from', choices=['A', 'B', 'both'], default='A',
                        help='Which source to relabel from (default: A)')
    parser.add_argument('--fps-type', default='morgan',
                        help='Fingerprint type (default: morgan)')
    parser.add_argument('--clustering-method', default='complete_linkage',
                        choices=['butina', 'clique', 'dbscan', 'complete_linkage'],
                        help='Clustering method (default: complete_linkage)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of proteins to process (for debugging)')
    parser.add_argument('--reclassification-report', default=None,
                        help='Path to pre-computed UniRef90 reclassification CSV (from compare_novel_proteins.py)')
    parser.add_argument('--uniref90-cache', default=None,
                        help='Path to JSON cache for UniRef90 API queries')
    parser.add_argument('--skip-uniref90', action='store_true',
                        help='Skip UniRef90 reclassification (use exact matching only)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / 'BDB_SPLIT').mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df_6k, df_bd, new_proteins, overlap_proteins = load_data(args.harvest, args.bdb)

    # Filter BDB to bindingdb source if column exists
    if 'source' in df_bd.columns:
        df_bd = df_bd[df_bd['source'] == 'bindingdb']
        print(f"Filtered BDB to bindingdb source: {len(df_bd)} rows")

    smiles_col_harvest = detect_smiles_col(df_6k)
    smiles_col_bdb = detect_smiles_col(df_bd)
    print(f"SMILES columns: harvest='{smiles_col_harvest}', bdb='{smiles_col_bdb}'")

    # UniRef90 reclassification
    bdb_acc_map = {}
    if not args.skip_uniref90:
        proteins_bdb = set(df_bd['uniprot_acc'].dropna().unique())
        bdb_acc_map, new_proteins, reclassified = reclassify_via_uniref90(
            new_proteins, proteins_bdb,
            report_path=args.reclassification_report,
            cache_path=args.uniref90_cache,
        )
        overlap_proteins = overlap_proteins | reclassified
        print(f"After UniRef90: New={len(new_proteins)}, Overlap={len(overlap_proteins)}")

    # Apply limit
    if args.limit:
        new_proteins = set(sorted(new_proteins)[:args.limit])
        overlap_proteins = set(sorted(overlap_proteins)[:args.limit])
        print(f"Limited to {len(new_proteins)} new + {len(overlap_proteins)} overlap proteins")

    # Process new proteins
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(new_proteins)} NEW PROTEINS (harvest-only)")
    print(f"{'='*70}")
    df_new_stats = process_new_proteins(
        df_6k, new_proteins, smiles_col_harvest, splits_dir,
        args.cluster_threshold, args.fps_type, args.clustering_method,
        ligand_threshold=args.ligand_threshold,
    )

    rearrange_cols = ['protein', 'n_compounds', 'n_clusters']
    new_stats_path = output_dir / 'df_new_protein_stats.csv'
    df_new_stats[rearrange_cols].to_csv(new_stats_path, index=False)
    print(f"\nSaved new protein stats: {new_stats_path} ({len(df_new_stats)} rows)")

    # Process overlap proteins
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(overlap_proteins)} OVERLAP PROTEINS")
    print(f"{'='*70}")
    # overlap_proteins = [ 'P21453' ]
    df_overlap_stats = process_overlap_proteins(
        df_6k, df_bd, overlap_proteins, smiles_col_harvest, smiles_col_bdb,
        splits_dir, args.cluster_threshold, args.fps_type, args.clustering_method,
        ligand_threshold=args.ligand_threshold, max_jumps=args.max_jumps,
        relabel_from=args.relabel_from, bdb_acc_map=bdb_acc_map,
    )
    overlap_stats_path = output_dir / 'df_overlap_stats.csv'
    df_overlap_stats['n_bdb_overlap_compounds'] = df_overlap_stats['n_bdb_compounds'] - df_overlap_stats['n_bdb_only_compounds']
    df_overlap_stats['n_harvest_overlap_compounds'] = df_overlap_stats['n_harvest_compounds'] - df_overlap_stats['n_harvest_only_compounds']
    rearrange_cols = ['protein', 'n_harvest_compounds', 'n_harvest_only_compounds', 'n_harvest_overlap_compounds',
                      'n_bdb_compounds', 'n_bdb_only_compounds', 'n_bdb_overlap_compounds',
                      'n_compound_overlap',
                       'n_total_clusters', 'n_harvest_only_clusters', 'n_bdb_only_clusters', 'n_shared_clusters']
    df_overlap_stats[rearrange_cols].to_csv(overlap_stats_path, index=False)
    print(f"\nSaved overlap stats: {overlap_stats_path} ({len(df_overlap_stats)} rows)")

    print(f"\n{'='*70}")
    print("ALL DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
