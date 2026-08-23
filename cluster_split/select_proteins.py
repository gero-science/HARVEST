"""
Select 50-60 Representative Proteins and Build Analysis Dataset
===============================================================

This is the target-selection step used to build **H-bench**: it picks the
diverse protein subset that became the released per-protein CSVs. It is kept
for provenance -- so the published benchmark can be traced back to the code
that produced it -- rather than as part of the everyday pipeline, which is why
it is not documented in the README.

Nothing imports it; it runs standalone, after `run_protein_split.py` has
produced the per-protein split directories. It shells out to `mmseqs` for
sequence-identity filtering, so that binary must be on PATH.

Curates a diverse subset of proteins from per-protein split results,
merges harvest data with split labels, computes activity statistics,
and generates tanimoto similarity plots.

Usage:
    python cluster_split/select_proteins.py
    python cluster_split/select_proteins.py --harvest path/to/harvest.parquet
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gaussian_kde
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, inchi as rdkit_inchi
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

warnings.filterwarnings('ignore', category=FutureWarning)
RDLogger.DisableLog('rdApp.*')
_MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)

# ── Paths (set in main() from CLI args) ───────────────────────────────────
SPLITS_DIR = None
OVERLAP_STATS = None
NEW_STATS = None
CHEMBL_INFO = None

MANDATORY_PROTEINS = ['P03372', 'O43613', 'P23458', 'P56373', 'O60341', 'O75874']

# Protein families to exclude (matched case-insensitively against l2)
EXCLUDED_L2_FAMILIES = ['Cytochrome P450']

# Columns to drop from the final selected_proteins.csv
_DROP_COLS = [
    'n_total_clusters', 'n_harvest_only_clusters', 'n_bdb_only_clusters',
    'n_shared_clusters', 'n_big_clusters', 'ave_clust_size',
    'ave_big_clust_size', 'error', 'n_label_B'
]

# Columns to drop from per-protein merged CSVs
_MERGE_DROP_COLS = [
    'Ligand SMILES', 'inchikey', 'smiles_source', 'smiles_cdx',
    'inchi_key_cdx', 'normalized_pub_number', 'molecular_weight',
    'stage1_reasoning', 'stage2_reasoning',
    'uniprot_acc_1st', 'Ligand InChI Key',
    'compound', 'extreme_conditions', 'assay_id', 'organism', 'is_complex'
]

ACTIVITY_COLS = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
ACTIVE_THRESHOLD = 100  # nM

# ── Helpers ────────────────────────────────────────────────────────────────

def _smiles_to_fp(smi):
    """Convert SMILES to Morgan fingerprint (radius=2, 2048 bits)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return _MORGAN_GEN.GetFingerprint(mol)


def _smiles_to_inchikey(smi):
    """Convert SMILES to InChIKey via RDKit. Returns '' on failure."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return ''
    inchi_val = rdkit_inchi.MolToInchi(mol)
    if inchi_val is None:
        return ''
    return rdkit_inchi.InchiToInchiKey(inchi_val) or ''


def _compute_max_tanimoto(fps_query, fps_ref):
    """For each query fp, compute max Tanimoto similarity to any ref fp."""
    if not fps_ref:
        return np.full(len(fps_query), np.nan)
    max_sims = np.empty(len(fps_query))
    for i, fp in enumerate(fps_query):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_ref)
        max_sims[i] = max(sims) if sims else 0.0
    return max_sims


def load_harvest(harvest_path):
    """Load harvest parquet once, derive uniprot_acc_1st. Returns full DataFrame."""
    print("Loading harvest parquet...")
    df = pd.read_parquet(harvest_path)
    df = df.rename(columns={'Target accession': 'uniprot_acc'})
    df['uniprot_acc_1st'] = [str(x).split(';')[0] for x in df['uniprot_acc']]
    print(f"  {len(df)} rows, {df['uniprot_acc_1st'].nunique()} proteins")
    return df


def cluster_proteins_by_sequence(proteins, df_harvest, identity_threshold=0.4):
    """Cluster proteins by sequence similarity using MMseqs2.

    Returns dict {protein_accession: cluster_representative}.
    Falls back to identity mapping (no filtering) if MMseqs2 fails.
    """
    mmseqs_bin = shutil.which('mmseqs') or '/opt/homebrew/bin/mmseqs'
    if not os.path.isfile(mmseqs_bin):
        print("  WARNING: mmseqs not found, skipping sequence diversity filter")
        return {p: p for p in proteins}

    # Extract one sequence per protein from harvest
    seq_map = {}
    for prot in proteins:
        rows = df_harvest.loc[df_harvest['uniprot_acc_1st'] == prot, 'Sequence']
        rows = rows.dropna()
        if len(rows) > 0:
            seq_map[prot] = str(rows.iloc[0])

    if len(seq_map) < 2:
        return {p: p for p in proteins}

    tmp_dir = tempfile.mkdtemp(prefix='mmseqs_seqclust_')
    try:
        # Write FASTA
        fasta_path = os.path.join(tmp_dir, 'proteins.fasta')
        with open(fasta_path, 'w') as f:
            for prot, seq in seq_map.items():
                f.write(f'>{prot}\n{seq}\n')

        result_prefix = os.path.join(tmp_dir, 'clust')
        mmseqs_tmp = os.path.join(tmp_dir, 'mmseqs_tmp')

        cmd = [
            mmseqs_bin, 'easy-cluster',
            fasta_path, result_prefix, mmseqs_tmp,
            '--min-seq-id', str(identity_threshold),
            '-c', '0.6',
            '--cov-mode', '0',
            '-v', '0',
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"  WARNING: mmseqs failed (rc={proc.returncode}): {proc.stderr[:200]}")
            return {p: p for p in proteins}

        # Parse cluster TSV: representative\tmember
        cluster_tsv = result_prefix + '_cluster.tsv'
        cluster_map = {}
        with open(cluster_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    rep, member = parts
                    cluster_map[member] = rep

        # Fill in proteins that had no sequence (map to themselves)
        for p in proteins:
            if p not in cluster_map:
                cluster_map[p] = p

        n_clusters = len(set(cluster_map[p] for p in proteins if p in cluster_map))
        print(f"  Sequence clustering: {len(seq_map)} proteins → {n_clusters} clusters "
              f"(identity={identity_threshold})")
        return cluster_map

    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  WARNING: mmseqs error: {e}, skipping sequence diversity filter")
        return {p: p for p in proteins}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Step 1: Load and filter proteins ───────────────────────────────────────

def load_and_filter(n_cluster_threshold=30, df_harvest=None,
                    min_overlap_harvest=300, min_new_harvest=200,
                    max_seq_length=1200):
    """Load stats CSVs and classification, filter by cluster counts.

    For new proteins, also filters by is_complex != 'Y' from harvest data
    (only non-complex compounds count toward the threshold).
    """
    df_overlap = pd.read_csv(OVERLAP_STATS)
    df_new = pd.read_csv(NEW_STATS)
    df_chembl = pd.read_csv(CHEMBL_INFO, compression='gzip')

    # Deduplicate chembl by accession (keep first)
    df_chembl = df_chembl.drop_duplicates(subset='accession', keep='first')

    # Join with classification + protein name
    chembl_cols = ['accession', 'protein_pref_name', 'gene_symbol', 'l1', 'l2', 'l3']
    df_overlap = df_overlap.merge(
        df_chembl[chembl_cols],
        left_on='protein', right_on='accession', how='left'
    ).drop(columns=['accession'], errors='ignore')
    df_overlap = df_overlap.rename(columns={'protein_pref_name': 'Protein Name'})

    # New proteins: rename columns to match overlap schema
    df_new = df_new.rename(columns={
        'n_compounds': 'n_harvest_only_compounds',
        'n_clusters': 'n_total_clusters',
    })
    df_new = df_new.merge(
        df_chembl[chembl_cols],
        left_on='protein', right_on='accession', how='left'
    ).drop(columns=['accession', 'scaffold_ratio'], errors='ignore')
    df_new = df_new.rename(columns={'protein_pref_name': 'Protein Name'})

    # For new proteins: count non-complex compounds from harvest
    if df_harvest is not None:
        print("  Filtering new proteins by is_complex...")
        df_h_nc = df_harvest[df_harvest['is_complex'] != 'Y']
        nc_counts = df_h_nc.groupby('uniprot_acc_1st').size().rename('n_harvest_non_complex')
        df_new = df_new.merge(nc_counts, left_on='protein', right_index=True, how='left')
        df_new['n_harvest_non_complex'] = df_new['n_harvest_non_complex'].fillna(0).astype(int)
        n_before = len(df_new)
        df_new['n_harvest_only_compounds'] = df_new['n_harvest_non_complex']
        df_new = df_new.drop(columns=['n_harvest_non_complex'])
        print(f"  New proteins with non-complex data: {(df_new['n_harvest_only_compounds'] > 0).sum()} / {n_before}")

    # Filter overlap: both sources independently have enough clusters
    mask_overlap = (
        (df_overlap['n_harvest_only_clusters'] > n_cluster_threshold) &
        (df_overlap['n_bdb_compounds'] > 1000 )   # enoght data in BDB
        # (df_overlap['n_bdb_only_clusters'] > n_cluster_threshold)
    )
    df_overlap_filt = df_overlap[mask_overlap].copy()

    # Filter new: enough clusters
    df_new_filt = df_new[df_new['n_total_clusters'] > n_cluster_threshold].copy()

    # Exclude cytochrome and other blacklisted families
    for excl in EXCLUDED_L2_FAMILIES:
        excl_lower = excl.lower()
        mask_ov = df_overlap_filt['l2'].str.lower().eq(excl_lower).fillna(False)
        mask_new = df_new_filt['l2'].str.lower().eq(excl_lower).fillna(False)
        n_removed = mask_ov.sum() + mask_new.sum()
        df_overlap_filt = df_overlap_filt[~mask_ov]
        df_new_filt = df_new_filt[~mask_new]
        if n_removed:
            print(f"  Excluded {n_removed} '{excl}' proteins")

    # Exclude enzymes without l2 annotation
    mask_enz_no_l2 = (df_overlap_filt['l1'] == 'Enzyme') & df_overlap_filt['l2'].isna()
    n_enz = mask_enz_no_l2.sum()
    df_overlap_filt = df_overlap_filt[~mask_enz_no_l2]
    mask_enz_no_l2_new = (df_new_filt['l1'] == 'Enzyme') & df_new_filt['l2'].isna()
    n_enz += mask_enz_no_l2_new.sum()
    df_new_filt = df_new_filt[~mask_enz_no_l2_new]
    if n_enz:
        print(f"  Excluded {n_enz} enzymes without l2 annotation")

    # Require sufficient A-label molecules (harvest-unique compounds)
    n_before_ov = len(df_overlap_filt)
    df_overlap_filt = df_overlap_filt[df_overlap_filt['n_harvest_only_compounds'] >= min_overlap_harvest]
    print(f"  Harvest_only_compounds > {min_overlap_harvest}: {n_before_ov} -> {len(df_overlap_filt)} overlap")

    # For new proteins: require >= min_new_harvest non-complex compounds
    n_before_new = len(df_new_filt)
    df_new_filt = df_new_filt[df_new_filt['n_harvest_only_compounds'] >= min_new_harvest]
    print(f"  n_harvest >= {min_new_harvest} (non-complex): {n_before_new} -> {len(df_new_filt)} new")

    # Filter by sequence length (first sequence per protein)
    if df_harvest is not None and max_seq_length is not None:
        seq_lengths = (
            df_harvest.dropna(subset=['Sequence'])
            .drop_duplicates(subset='uniprot_acc_1st')
            .set_index('uniprot_acc_1st')['Sequence']
            .str.len()
        )
        long_proteins = set(seq_lengths[seq_lengths > max_seq_length].index)
        n_ov_before = len(df_overlap_filt)
        n_new_before = len(df_new_filt)
        df_overlap_filt = df_overlap_filt[~df_overlap_filt['protein'].isin(long_proteins)]
        df_new_filt = df_new_filt[~df_new_filt['protein'].isin(long_proteins)]
        print(f"  Seq length <= {max_seq_length} AA: overlap {n_ov_before} -> {len(df_overlap_filt)}, "
              f"new {n_new_before} -> {len(df_new_filt)}")

    print(f"Overlap candidates: {len(df_overlap_filt)} / {len(df_overlap)}")
    print(f"New candidates: {len(df_new_filt)} / {len(df_new)}")

    return df_overlap_filt, df_new_filt


# ── Step 2: Select 2-3 per protein family ──────────────────────────────────

def _pick_diverse(group_df, n_pick, sort_col, seq_clusters):
    """Pick up to n_pick proteins from group, at most 1 per sequence cluster.

    Greedy: iterate by sort_col descending, skip if cluster rep already taken.
    """
    if seq_clusters is None:
        return group_df.nlargest(min(n_pick, len(group_df)), sort_col)

    sorted_df = group_df.sort_values(sort_col, ascending=False)
    taken_reps = set()
    indices = []
    for idx, row in sorted_df.iterrows():
        rep = seq_clusters.get(row['protein'], row['protein'])
        if rep not in taken_reps:
            taken_reps.add(rep)
            indices.append(idx)
            if len(indices) >= n_pick:
                break
    return group_df.loc[indices]


def select_from_pool(df, group_col_primary='l2', group_col_fallback='l1',
                     target_n=50, per_group=2, extra_l3_groups=None,
                     mandatory=None, seq_clusters=None):
    """Pick 2-3 proteins per family group, preferring largest n_total_clusters.

    Parameters
    ----------
    extra_l3_groups : dict, optional
        {l3_value: n_extra} — pick additional proteins from specific l3 classes
        to increase diversity within broad l2 groups.
    mandatory : list, optional
        Protein accessions to force-include if present in df.
    seq_clusters : dict, optional
        {protein: cluster_representative} from MMseqs2. When provided, at most
        one protein per sequence cluster is selected within each l2 group.
    """
    sort_col = 'n_total_clusters'

    selected = []

    # Group by l2 first
    has_l2 = df[df[group_col_primary].notna()].copy()
    no_l2 = df[df[group_col_primary].isna()].copy()

    # For proteins with l2 classification
    for group_name, group_df in has_l2.groupby(group_col_primary):
        n_pick = per_group if len(group_df) >= per_group else len(group_df)
        picks = _pick_diverse(group_df, n_pick, sort_col, seq_clusters)
        selected.append(picks)

    # Extra l3 diversity picks
    if extra_l3_groups:
        already_selected = set()
        for s in selected:
            already_selected.update(s['protein'].values)

        for l3_val, n_extra in extra_l3_groups.items():
            pool = has_l2[
                (has_l2['l3'] == l3_val) &
                (~has_l2['protein'].isin(already_selected))
            ]
            if len(pool) > 0:
                picks = _pick_diverse(pool, n_extra, sort_col, seq_clusters)
                selected.append(picks)
                already_selected.update(picks['protein'].values)

    # For proteins with only l1 (fallback) — but skip enzymes (excluded above)
    has_l1 = no_l2[no_l2[group_col_fallback].notna()]
    for group_name, group_df in has_l1.groupby(group_col_fallback):
        n_pick = min(per_group, len(group_df))
        picks = _pick_diverse(group_df, n_pick, sort_col, seq_clusters)
        selected.append(picks)

    # Unclassified: pick top few
    unclassified = no_l2[no_l2[group_col_fallback].isna()]
    if len(unclassified) > 0:
        picks = _pick_diverse(unclassified, per_group, sort_col, seq_clusters)
        selected.append(picks)

    if mandatory is not None:
        m = df['protein'].isin(mandatory)
        if m.sum():
            print(f"  Adding {m.sum()} mandatory proteins")
            selected.append(df[m])

    if not selected:
        return pd.DataFrame()

    result = pd.concat(selected, ignore_index=True)
    result = result.drop_duplicates(subset='protein', keep='first')

    # If we have too many, trim globally
    if len(result) > target_n:
        result = result.nlargest(target_n, sort_col)

    return result


# ── Step 4: Merge harvest data with split results ─────────────────────────

def merge_harvest_with_splits(proteins, df_harvest, output_dir):
    """For each protein, merge harvest slice with split results.

    Drops internal columns, renames clean_smiles→SMILES, computes InChIKey.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_counts = {}
    for protein in proteins:
        split_file = SPLITS_DIR / f'split_results_{protein}.csv'
        if not split_file.exists():
            print(f"  SKIP {protein}: no split file")
            continue

        df_split = pd.read_csv(split_file)
        df_split = df_split[df_split.final_label.isin(['A', 'C'])] # drop duplicates adn border clusters
        df_h = df_harvest[df_harvest['uniprot_acc_1st'] == protein].copy()

        if len(df_h) == 0:
            print(f"  SKIP {protein}: no harvest data")
            continue

        # Merge on clean_smiles, keep split columns except 'label'
        split_cols = ['clean_smiles', 'cluster_id', 'final_label',
                      'nearest_smiles', 'tanimoto_sim']
        df_split_subset = (df_split[split_cols]
                           .drop_duplicates(subset='clean_smiles')
                           .rename(columns={'nearest_smiles': 'nearest_BDB_smiles'}))

        df_merged = df_h.merge(df_split_subset, on='clean_smiles', how='inner')

        # Drop internal columns
        cols_drop = [c for c in _MERGE_DROP_COLS if c in df_merged.columns]
        df_merged = df_merged.drop(columns=cols_drop)

        # Rename clean_smiles → SMILES
        df_merged = df_merged.rename(columns={'clean_smiles': 'SMILES'})

        # Compute InChIKey from SMILES
        df_merged['InChIKey'] = df_merged['SMILES'].apply(_smiles_to_inchikey)

        df_merged.to_csv(output_dir / f'{protein}.csv', index=False)
        merged_counts[protein] = len(df_merged)

    print(f"Merged {len(merged_counts)} proteins")
    return merged_counts


# ── Step 5: Activity statistics ────────────────────────────────────────────

def parse_activity_value(val):
    """Parse activity value from string, handling <, >, =, ranges."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == '' or val.lower() == 'nan':
        return np.nan

    # Remove leading operators
    val = re.sub(r'^[<>=~]+\s*', '', val)

    # Handle ranges: "10-20" or "10 to 20" → mean
    range_match = re.match(r'^([\d.eE+\-]+)\s*(?:to|-)\s*([\d.eE+\-]+)$', val)
    if range_match:
        try:
            a, b = float(range_match.group(1)), float(range_match.group(2))
            return (a + b) / 2
        except ValueError:
            pass

    try:
        return float(val)
    except ValueError:
        return np.nan


def _activity_stats_for_df(df):
    """Compute activity stats dict for a single protein's DataFrame.

    Expects columns: activity cols (Ki/IC50/Kd/EC50 in nM) and 'final_label'.
    """
    # Parse activity values
    for col in ACTIVITY_COLS:
        if col in df.columns:
            df[col + '_parsed'] = df[col].apply(parse_activity_value)

    parsed_cols = [c + '_parsed' for c in ACTIVITY_COLS if c + '_parsed' in df.columns]
    if parsed_cols:
        df['is_active'] = df[parsed_cols].apply(
            lambda r: any(v < ACTIVE_THRESHOLD for v in r if pd.notna(v)),
            axis=1
        )
        df['has_activity'] = df[parsed_cols].notna().any(axis=1)
    else:
        df['is_active'] = False
        df['has_activity'] = False

    n_total = len(df)
    n_with_activity = int(df['has_activity'].sum())
    n_active = int(df['is_active'].sum())
    n_inactive = n_with_activity - n_active
    pct_active = round(n_active / n_with_activity * 100, 1) if n_with_activity > 0 else 0

    row = {
        'n_total': n_total,
        'n_with_activity': n_with_activity,
        'n_active': n_active,
        'n_inactive': n_inactive,
        'pct_active': pct_active,
    }

    for label in ['A', 'C']:
        sub = df[df['final_label'] == label]
        n_sub = len(sub)
        n_sub_activity = int(sub['has_activity'].sum())
        n_sub_active = int(sub['is_active'].sum())
        n_sub_inactive = n_sub_activity - n_sub_active
        pct_sub = round(n_sub_active / n_sub_activity * 100, 1) if n_sub_activity > 0 else 0
        row[f'n_{label}'] = n_sub
        row[f'n_active_{label}'] = n_sub_active
        row[f'n_inactive_{label}'] = n_sub_inactive
        row[f'pct_active_{label}'] = pct_sub

    return row


def compute_activity_stats(proteins, df_harvest):
    """Compute activity stats per protein by joining harvest with split labels.

    Works both pre-merge (for filtering) and post-merge (for final stats).
    Uses the shared harvest DataFrame — no redundant parquet loads.
    """
    rows = []
    for protein in proteins:
        split_file = SPLITS_DIR / f'split_results_{protein}.csv'
        if not split_file.exists():
            continue

        df_split = pd.read_csv(split_file, usecols=['clean_smiles', 'final_label'])
        df_h = df_harvest[df_harvest['uniprot_acc_1st'] == protein].copy()
        if len(df_h) == 0:
            continue

        df_split_dedup = df_split.drop_duplicates(subset='clean_smiles')
        df = df_h.merge(df_split_dedup, on='clean_smiles', how='inner')

        row = _activity_stats_for_df(df)
        row['protein'] = protein
        rows.append(row)

    # Reorder columns so 'protein' is first
    df_result = pd.DataFrame(rows)
    if len(df_result) > 0:
        cols = ['protein'] + [c for c in df_result.columns if c != 'protein']
        df_result = df_result[cols]
    return df_result


# ── Step 6: Tanimoto plots (KDE) ───────────────────────────────────────────

def _load_tanimoto(protein):
    """Load tanimoto_sim for A and C labels, filtering out zeros.

    Returns (sims_a_to_bdb, sims_c_to_bdb, sims_a_to_c):
      - sims_a_to_bdb: precomputed tanimoto_sim for A compounds (to nearest BDB)
      - sims_c_to_bdb: precomputed tanimoto_sim for C compounds (to nearest BDB)
      - sims_a_to_c:   max tanimoto of each A compound to any C compound (computed here)
    """
    split_file = SPLITS_DIR / f'split_results_{protein}.csv'
    if not split_file.exists():
        return None, None, None
    df = pd.read_csv(split_file)

    # Precomputed tanimoto_sim (to nearest BDB), filter zeros
    df_nz = df[df['tanimoto_sim'] > 0]
    sims_a = df_nz[df_nz['final_label'] == 'A']['tanimoto_sim'].dropna().values
    sims_c = df_nz[df_nz['final_label'] == 'C']['tanimoto_sim'].dropna().values

    # Compute A → C similarity from SMILES
    smiles_a = df[df['final_label'] == 'A']['clean_smiles'].dropna().tolist()
    smiles_c = df[df['final_label'] == 'C']['clean_smiles'].dropna().tolist()

    sims_a_to_c = np.array([])
    if smiles_a and smiles_c:
        fps_a = [fp for fp in (_smiles_to_fp(s) for s in smiles_a) if fp is not None]
        fps_c = [fp for fp in (_smiles_to_fp(s) for s in smiles_c) if fp is not None]
        if fps_a and fps_c:
            sims_a_to_c = _compute_max_tanimoto(fps_a, fps_c)

    return sims_a, sims_c, sims_a_to_c


def _plot_kde(ax, data, color, label):
    """Plot a KDE curve with filled area on the given axes."""
    if len(data) < 2:
        return
    kde = gaussian_kde(data, bw_method='scott')
    x = np.linspace(max(0, data.min() - 0.05), min(1, data.max() + 0.05), 300)
    y = kde(x)
    ax.fill_between(x, y, alpha=0.3, color=color)
    ax.plot(x, y, color=color, linewidth=1.5, label=label)
    ax.axvline(np.median(data), color=color, linestyle='--', alpha=0.8,
               label=f'median={np.median(data):.3f}')


def plot_tanimoto_per_protein(proteins, output_dir):
    """Plot KDE of tanimoto similarities per protein.

    Three subplots:
      1. A → BDB (precomputed tanimoto_sim for harvest-unique)
      2. C → BDB (precomputed tanimoto_sim for common/buffer)
      3. A → C  (max tanimoto from each A compound to any C compound)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for protein in proteins:
        sims_a, sims_c, sims_a_to_c = _load_tanimoto(protein)
        if sims_a is None:
            continue
        if len(sims_a) == 0 and len(sims_c) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        _plot_kde(axes[0], sims_a, 'steelblue', f'n={len(sims_a)}')
        axes[0].set_title(f'A → BDB (n={len(sims_a)})')
        axes[0].set_xlabel('Max Tanimoto Similarity')
        axes[0].set_ylabel('Density')
        axes[0].legend()

        _plot_kde(axes[1], sims_c, 'coral', f'n={len(sims_c)}')
        axes[1].set_title(f'C → BDB (n={len(sims_c)})')
        axes[1].set_xlabel('Max Tanimoto Similarity')
        axes[1].set_ylabel('Density')
        axes[1].legend()

        if len(sims_a_to_c) > 1:
            _plot_kde(axes[2], sims_a_to_c, 'mediumseagreen', f'n={len(sims_a_to_c)}')
        axes[2].set_title(f'A → C (n={len(sims_a_to_c)})')
        axes[2].set_xlabel('Max Tanimoto Similarity')
        axes[2].set_ylabel('Density')
        if len(sims_a_to_c) > 1:
            axes[2].legend()

        fig.suptitle(f'{protein} — Tanimoto Similarity KDE', fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / f'{protein}.png', dpi=120)
        plt.close(fig)


def plot_tanimoto_aggregate(proteins, output_path):
    """Aggregate KDE tanimoto plot across all selected proteins.

    Three subplots: A → BDB, C → BDB, A → C.
    """
    all_a, all_c, all_a_to_c = [], [], []

    for protein in proteins:
        sims_a, sims_c, sims_a_to_c = _load_tanimoto(protein)
        if sims_a is None:
            continue
        all_a.extend(sims_a.tolist())
        all_c.extend(sims_c.tolist())
        all_a_to_c.extend(sims_a_to_c.tolist())

    all_a = np.array(all_a)
    all_c = np.array(all_c)
    all_a_to_c = np.array(all_a_to_c)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    _plot_kde(axes[0], all_a, 'steelblue', f'n={len(all_a)}')
    axes[0].set_title(f'Valid → BindingDB')
    axes[0].set_xlabel('Max Tanimoto Similarity')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    _plot_kde(axes[1], all_c, 'coral', f'n={len(all_c)}')
    axes[1].set_title(f'Common → BindingDB')
    axes[1].set_xlabel('Max Tanimoto Similarity')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Aggregate tanimoto plot saved: {output_path}")

    fig, axes = plt.subplots(1, 1, figsize=(4, 3))
    if len(all_a_to_c) > 1:
        _plot_kde(axes, all_a_to_c, 'mediumseagreen', f'n={len(all_a_to_c)}')
    axes.set_title(f'A → C (n={len(all_a_to_c)})')
    axes.set_xlabel('Max Tanimoto Similarity')
    axes.set_ylabel('Density')
    if len(all_a_to_c) > 1:
        axes.legend()

    fig.tight_layout()
    a_to_c_path = output_path.parent / 'tanimoto_A_to_C.png'
    fig.savefig(a_to_c_path, dpi=150)
    plt.close(fig)
    print(f"Aggregate tanimoto A→C plot saved: {a_to_c_path}")


# ── Main ───────────────────────────────────────────────────────────────────
DEFAULT_HARVEST = Path(
    'final_v16_clean.parquet'
)
DEFAULT_SPLIT = Path(__file__).resolve().parent / 'split_v14'

def main():
    parser = argparse.ArgumentParser(description='Select representative proteins')
    parser.add_argument('--harvest', default=str(DEFAULT_HARVEST),
                        help='Path to harvest parquet')
    parser.add_argument('--split-dir', default=str(DEFAULT_SPLIT),
                        help='Path to harvest parquet')
    parser.add_argument('--n-overlap', type=int, default=100,
                        help='Target number of overlap proteins')
    parser.add_argument('--n-per-group', type=int, default=4,
                        help='Target number of proteins for same annotaion')
    parser.add_argument('--n-new', type=int, default=20,
                        help='Target number of new proteins')
    parser.add_argument('--cluster-threshold', type=int, default=30,
                        help='Minimum clusters per source')
    parser.add_argument('--seq-identity', type=float, default=0.2,
                        help='Sequence identity threshold for diversity filtering (0-1)')
    parser.add_argument('--max-seq-length', type=int, default=1200,
                        help='Maximum protein sequence length in AA (default: 1200)')
    args = parser.parse_args()

    global SPLITS_DIR, OVERLAP_STATS, NEW_STATS, CHEMBL_INFO
    SPLIT_DIR = Path(args.split_dir)
    SPLITS_DIR = SPLIT_DIR / 'splits'
    OVERLAP_STATS = SPLIT_DIR / 'df_overlap_stats.csv'
    NEW_STATS = SPLIT_DIR / 'df_new_protein_stats.csv'
    CHEMBL_INFO = Path(__file__).resolve().parent.parent / 'curated_data' / 'target_info_chembl.csv.gz'


    out_dir = SPLIT_DIR / 'selected_proteins'
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = out_dir / 'merged'

    # Load harvest once — reused across all steps
    df_harvest = load_harvest(args.harvest)

    # Step 1: Load and filter
    print("=" * 60)
    print("Step 1: Load and filter proteins")
    df_overlap_filt, df_new_filt = load_and_filter(
        args.cluster_threshold, df_harvest=df_harvest,
        min_overlap_harvest=300, min_new_harvest=100,
        max_seq_length=args.max_seq_length)

    # Step 1b: Sequence-based diversity clustering
    print("\n" + "=" * 60)
    print("Step 1b: Sequence diversity clustering (MMseqs2)")
    all_candidates = set(df_overlap_filt['protein'].tolist() + df_new_filt['protein'].tolist())
    seq_clusters = cluster_proteins_by_sequence(
        list(all_candidates), df_harvest, identity_threshold=args.seq_identity)

    # Step 2: Select per family
    #   Extra GPCR family A diversity: pick 4 more from l3
    print("\n" + "=" * 60)
    print("Step 2: Select per protein family")
    extra_l3 = {'Small molecule receptor (family A GPCR)': max(3, args.n_per_group),
                'Peptide receptor (family A GPCR)': max(3, args.n_per_group)
                }
    sel_overlap = select_from_pool(
        df_overlap_filt, target_n=args.n_overlap, per_group=args.n_per_group,
        extra_l3_groups=extra_l3, mandatory=MANDATORY_PROTEINS,
        seq_clusters=seq_clusters,
    )
    sel_new = select_from_pool(df_new_filt, target_n=args.n_new, per_group=args.n_per_group,
                               mandatory=MANDATORY_PROTEINS,
                               seq_clusters=seq_clusters)
    sel_overlap['source'] = 'overlap'
    sel_new['source'] = 'new'
    print(f"  Selected overlap: {len(sel_overlap)}")
    print(f"  Selected new: {len(sel_new)}")

    # Show GPCR picks
    gpcr = sel_overlap[sel_overlap['l3'].str.contains('family A GPCR', na=False)]
    if len(gpcr):
        print(f"  GPCR family A selected: {len(gpcr)}")
        for _, r in gpcr.iterrows():
            print(f"    {r['protein']}  {r.get('Protein Name', '')}")

    # Combine
    selected = pd.concat([sel_overlap, sel_new], ignore_index=True)
    selected = selected.drop_duplicates(subset='protein', keep='first')
    proteins = selected['protein'].tolist()
    print(f"\n  TOTAL selected: {len(selected)}")
    print(f"  L1 families: {selected['l1'].nunique()}")
    print(f"  L2 families: {selected['l2'].nunique()}")
    tot_man = selected['protein'].isin(MANDATORY_PROTEINS).sum()
    miss = len(MANDATORY_PROTEINS) - tot_man
    print(f"\n  TOTAL mandatory: {tot_man}")
    if miss > 0:
        print(f"  Missing mandatory: {np.setdiff1d(MANDATORY_PROTEINS, proteins)}")

    # Step 5: Activity statistics
    print("\n" + "=" * 60)
    print("Step 5: Activity statistics")
    df_activity = compute_activity_stats(proteins, df_harvest)
    df_activity.to_csv(out_dir / 'activity_summary.csv', index=False)
    print(f"  Proteins with stats: {len(df_activity)}")
    if len(df_activity) > 0:
        print(f"  Mean % active: {df_activity['pct_active'].mean():.1f}%")

    # Step 5b: Filter by activity balance in A subset
    print("\n" + "=" * 60)
    print("Step 5b: Filter by activity balance in A subset (20% <= pct_active_A <= 55%)")
    min_balance = 20.0
    balanced = []
    for _, row in df_activity.iterrows():
        n_act_A = row.get('n_active_A', 0) + row.get('n_inactive_A', 0)
        if n_act_A == 0:
            continue
        pct_active_A = row['n_active_A'] / n_act_A * 100
        if min_balance <= pct_active_A <= 55:
            balanced.append(row['protein'])

    # Always keep mandatory proteins regardless of balance
    for p in MANDATORY_PROTEINS:
        if p not in balanced and p in set(proteins):
            print(f"  Keep {p} mandatory protein, balancing failed")
            balanced.append(p)
        elif p not in set(proteins):
            print(f"  {p} not in selected proteins")

    n_before = len(proteins)
    proteins = balanced
    print(f"  Before: {n_before}, After: {len(proteins)}")
    print(f"  Removed {n_before - len(proteins)} unbalanced proteins")

    # Save selected proteins list (drop internal stats columns)
    selected = selected[selected['protein'].isin(proteins)].copy()
    cols_to_drop = [c for c in _DROP_COLS if c in selected.columns]
    selected_out = selected.drop(columns=cols_to_drop)
    selected_out.to_csv(out_dir / 'selected_proteins.csv', index=False)
    print(f"  Saved: {out_dir / 'selected_proteins.csv'}")

    # Step 6: Merge harvest with splits (after filtering)
    print("\n" + "=" * 60)
    print("Step 6: Merge harvest data with split results")
    merge_harvest_with_splits(proteins, df_harvest, merged_dir)

    # Re-save activity stats (same data, already computed above)
    df_activity_final = df_activity[df_activity['protein'].isin(proteins)].copy()
    df_activity_final.to_csv(out_dir / 'activity_summary.csv', index=False)
    print(f"  Activity summary saved: {out_dir / 'activity_summary.csv'}")

    # Step 7: Tanimoto plots
    plot_tanimoto_aggregate(proteins, out_dir / 'tanimoto_A_C_BDB.png')

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print(f"Output directory: {out_dir}")
    for p in MANDATORY_PROTEINS:
        status = "OK" if p in set(proteins) else "MISSING"
        print(f"  Mandatory {p}: {status}")


if __name__ == '__main__':
    main()
