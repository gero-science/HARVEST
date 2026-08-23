"""
Annotate a HARVEST build with chemical-novelty columns.

For each protein target, HARVEST compounds and BindingDB compounds are pooled,
clustered by Tanimoto distance, and separated by a minimum buffer zone (see
``cluster_split/methods_splitting.md``). Every HARVEST compound then carries:

    cluster_id          per-protein cluster, namespaced as "<uniprot>_<n>"
    novelty_label       Novel / Buffer / Boundary  (categorical)
    nearest_BDB_smiles  most similar BindingDB compound
    tanimoto_sim        similarity to that compound

``novelty_label`` renames the internal A/B/C/CB alphabet used by
``split_clusters`` into the manuscript's terms:

    A  -> Novel      cluster holds only HARVEST compounds
    C  -> Buffer     buffer-zone cluster separating novel from BindingDB
    CB -> Boundary   mixed cluster, chemical space shared with BindingDB
    B  -> dropped    BindingDB-only; the output keeps HARVEST rows only

The rename matters: the same letters mean different things in
``allocate_training.py``, where A marks an exact test-set duplicate. Naming the
categories removes that ambiguity for good.

The script clusters per-protein unique SMILES, writes a compact annotation
table (``novelty_annotations.parquet``) and per-protein CSVs, then merges the
annotations back onto the full input parquet so the output keeps every
original column plus the four novelty columns.

Usage:
    # smoke test on two targets
    python annotate_novelty.py \
        --harvest final_v16.parquet --bdb full_bdb_chembl_fix.parquet \
        --proteins P00918 P07900 -o results/annot

    # all targets in the build
    python annotate_novelty.py \
        --harvest final_v16.parquet --bdb full_bdb_chembl_fix.parquet \
        -o results/annot
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_split.split_clusters import split_clusters

# Internal split label -> released category name.
LABEL_NAMES = {
    'A': 'Novel',
    'C': 'Buffer',
    'CB': 'Boundary',
}

# Fixed category order, so the dtype is identical across runs even when a
# protein happens to produce none of a given class.
NOVELTY_CATEGORIES = ['Novel', 'Buffer', 'Boundary']

ANNOTATION_COLUMNS = [
    'cluster_id', 'novelty_label', 'nearest_BDB_smiles', 'tanimoto_sim',
]

# Defaults from methods_splitting.md; these reproduce the published splits.
DEFAULT_LIGAND_THRESHOLD = 0.2
DEFAULT_CLUSTER_THRESHOLD = 0.225
DEFAULT_MAX_JUMPS = 2


def parquet_columns(path):
    """Column names of a parquet file, read from its schema alone."""
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(path).schema_arrow.names)


def detect_smiles_col(columns, candidates=('clean_smiles', 'smiles', 'SMILES',
                                           'Ligand SMILES')):
    """Return the first SMILES-like column present."""
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"No SMILES column found. Columns: {list(columns)}")


def detect_uniprot_col(columns, candidates=('uniprot_acc', 'Target accession',
                                            'uniprot_id')):
    """Return the first UniProt-accession column present."""
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"No UniProt column found. Columns: {list(columns)}")


def first_accession(value):
    """UniProt fields may hold several ';'-separated accessions; take the first."""
    return str(value).split(';')[0].strip()


def resolve_targets(proteins, harvest_accessions, limit=None):
    """Work out which accessions to annotate, keeping a stable order."""
    if proteins:
        targets = list(dict.fromkeys(proteins))
        missing = [t for t in targets if t not in harvest_accessions]
        if missing:
            print(f"  {len(missing)} target(s) absent from the build, skipped: "
                  f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
        targets = [t for t in targets if t in harvest_accessions]
    else:
        targets = sorted(harvest_accessions)

    if limit:
        targets = targets[:limit]
    return targets


def annotate_one_protein(protein, smiles_harvest, smiles_bdb, args):
    """Run the split for one target, returning its HARVEST-side annotations."""
    df_in = pd.concat([
        pd.DataFrame({'clean_smiles': smiles_harvest, 'label': 'HARVEST'}),
        pd.DataFrame({'clean_smiles': smiles_bdb, 'label': 'BDB'}),
    ], ignore_index=True)

    result = split_clusters(
        df_in,
        smiles_key='clean_smiles',
        label_key='label',
        label_map={'HARVEST': 'A', 'BDB': 'B'},
        cluster_threshold=args.cluster_threshold,
        ligand_threshold=args.ligand_threshold,
        max_jumps=args.max_jumps,
        clustering_method=args.clustering_method,
        relabel_from='A',
        n_jobs=args.workers,
    )

    df = result['df_valid']
    df = df[df['source_label'] == 'A'].copy()

    unmapped = set(df['final_label'].unique()) - set(LABEL_NAMES)
    if unmapped:
        raise ValueError(
            f"{protein}: split produced unexpected label(s) {sorted(unmapped)} "
            f"on the HARVEST side; expected {sorted(LABEL_NAMES)}.")

    df['novelty_label'] = df['final_label'].map(LABEL_NAMES)
    # Cluster ids are per-run, so namespace them to stay unique across targets.
    df['cluster_id'] = protein + '_' + df['cluster_id'].astype(str)
    df = df.rename(columns={'nearest_smiles': 'nearest_BDB_smiles'})
    df['uniprot_acc_1st'] = protein

    return df[['uniprot_acc_1st', 'clean_smiles'] + ANNOTATION_COLUMNS]


def main():
    parser = argparse.ArgumentParser(
        description='Annotate a HARVEST build with novelty/cluster/Tanimoto columns',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--harvest', required=True,
                        help='HARVEST build parquet')
    parser.add_argument('--bdb', required=True,
                        help='BindingDB reference parquet')
    parser.add_argument('--proteins', nargs='+',
                        help='UniProt accessions to annotate (default: all in the build)')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Where annotations and per-protein CSVs are written')
    parser.add_argument('--apply-to',
                        help='Path for the annotated full parquet (default: '
                             '<output-dir>/annotated.parquet)')
    parser.add_argument('--limit', type=int,
                        help='Only process the first N targets (smoke test)')
    parser.add_argument('--ligand-threshold', type=float,
                        default=DEFAULT_LIGAND_THRESHOLD,
                        help='Tanimoto distance for molecule clustering')
    parser.add_argument('--cluster-threshold', type=float,
                        default=DEFAULT_CLUSTER_THRESHOLD,
                        help='Tanimoto distance for the cluster graph')
    parser.add_argument('--max-jumps', type=int, default=DEFAULT_MAX_JUMPS,
                        help='Minimum hop separation between novel and BDB clusters')
    parser.add_argument('--clustering-method', default='complete_linkage',
                        choices=['butina', 'clique', 'dbscan', 'complete_linkage'])
    parser.add_argument('--workers', type=int, default=8,
                        help='Threads for fingerprinting')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ── Load just what the split needs, not the whole build ──
    print(f"Loading HARVEST accessions/SMILES from {args.harvest} ...")
    harvest_cols = parquet_columns(args.harvest)
    h_smiles_col = detect_smiles_col(harvest_cols)
    h_uniprot_col = detect_uniprot_col(harvest_cols)
    df_h = pd.read_parquet(args.harvest, columns=[h_uniprot_col, h_smiles_col])
    df_h['uniprot_acc_1st'] = df_h[h_uniprot_col].map(first_accession)
    print(f"  {len(df_h):,} rows; smiles='{h_smiles_col}', uniprot='{h_uniprot_col}'")

    print(f"Loading BindingDB reference from {args.bdb} ...")
    bdb_cols_all = parquet_columns(args.bdb)
    b_smiles_col = detect_smiles_col(bdb_cols_all)
    b_uniprot_col = detect_uniprot_col(bdb_cols_all)
    bdb_cols = [b_uniprot_col, b_smiles_col]
    if 'source' in bdb_cols_all:
        bdb_cols.append('source')
    df_b = pd.read_parquet(args.bdb, columns=bdb_cols)
    if 'source' in df_b.columns:
        df_b = df_b[df_b['source'] == 'bindingdb']
        print(f"  filtered to source='bindingdb': {len(df_b):,} rows")
    df_b['uniprot_acc_1st'] = df_b[b_uniprot_col].map(first_accession)
    print(f"  {len(df_b):,} rows; smiles='{b_smiles_col}', uniprot='{b_uniprot_col}'")

    targets = resolve_targets(args.proteins,
                              set(df_h['uniprot_acc_1st'].unique()), args.limit)
    print(f"\nAnnotating {len(targets)} target(s).")

    harvest_by_acc = df_h.groupby('uniprot_acc_1st')[h_smiles_col]
    bdb_by_acc = df_b.groupby('uniprot_acc_1st')[b_smiles_col]
    bdb_groups = bdb_by_acc.groups

    annotations, skipped = [], []
    for protein in tqdm(targets, desc='Targets'):
        smiles_harvest = list(harvest_by_acc.get_group(protein).dropna().unique())
        smiles_bdb = (list(df_b.loc[bdb_groups[protein], b_smiles_col].dropna().unique())
                      if protein in bdb_groups else [])

        if not smiles_harvest:
            skipped.append((protein, 'no HARVEST compounds'))
            continue
        if len(smiles_harvest) + len(smiles_bdb) < 2:
            skipped.append((protein, 'fewer than 2 compounds'))
            continue
        # When smiles_bdb is empty every compound is Novel by definition —
        # split_clusters handles the single-source case (sets tanimoto_sim=0,
        # nearest_BDB_smiles='').

        try:
            df_annot = annotate_one_protein(protein, smiles_harvest, smiles_bdb, args)
        except Exception as exc:
            print(f"  ERROR on {protein}: {exc}")
            skipped.append((protein, f'error: {exc}'))
            continue

        df_annot.to_csv(splits_dir / f'{protein}.csv', index=False)
        annotations.append(df_annot)

    if not annotations:
        raise SystemExit("No targets were annotated.")

    df_all = pd.concat(annotations, ignore_index=True)
    df_all['novelty_label'] = pd.Categorical(
        df_all['novelty_label'], categories=NOVELTY_CATEGORIES)

    annotations_path = output_dir / 'novelty_annotations.parquet'
    df_all.to_parquet(annotations_path, index=False)

    print(f"\n{'='*70}")
    print(f"Annotated {len(df_all):,} compound-target pairs "
          f"across {df_all['uniprot_acc_1st'].nunique()} target(s)")
    for name, count in df_all['novelty_label'].value_counts().items():
        print(f"  {name:9s} {count:,}")
    if skipped:
        print(f"  skipped {len(skipped)} target(s):")
        for protein, why in skipped[:10]:
            print(f"    {protein}: {why}")
    print(f"  compact annotations -> {annotations_path}")
    print(f"  per-protein CSVs    -> {splits_dir}")

    # ── Merge annotations onto the full input parquet ──
    apply_to = Path(args.apply_to) if args.apply_to else output_dir / 'annotated.parquet'
    print(f"\nApplying annotations to the full build ...")
    df_full = pd.read_parquet(args.harvest)
    df_full['uniprot_acc_1st'] = df_full[h_uniprot_col].map(first_accession)
    df_full = df_full.merge(
        df_all.rename(columns={'clean_smiles': h_smiles_col}),
        on=['uniprot_acc_1st', h_smiles_col], how='left')
    df_full.drop(columns=['uniprot_acc_1st'], inplace=True)
    df_full['novelty_label'] = pd.Categorical(
        df_full['novelty_label'], categories=NOVELTY_CATEGORIES)
    df_full.to_parquet(apply_to, index=False)
    n_annotated = df_full['novelty_label'].notna().sum()
    print(f"  {n_annotated:,} / {len(df_full):,} rows annotated")
    print(f"  -> {apply_to}")


if __name__ == '__main__':
    main()
