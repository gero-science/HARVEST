"""
Allocate New Data for Training Without Test Set Leakage
=======================================================

Loads new data (set B), compares against test set (set A) per protein family,
and moves similar molecules to common set (set C) to prevent leakage.

Protein matching supports two modes:
  - UniRef mode (--uniref 50|90|100): groups all uniprots sharing the same
    UniRef cluster via UniProt API, useful when uniprot IDs may differ between datasets.
  - Exact mode (default, no --uniref): matches by exact uniprot ID only.

Split-dir CSV format (one file per protein, named {uniprot_id}.csv):
  Required columns: 'SMILES', 'final_label' (values: 'A', 'B', 'C').

Pipeline per protein family:
  1. Load per-protein split file from --split-dir ({uniprot_id}.csv)
  2. Match new data proteins to split proteins (by cluster or exact uniprot ID)
  3. Combine set A smiles + set B smiles (new data, excluding exact A duplicates)
  4. Run split_clusters with relabel_from='B' (only B molecules can become C)
  5. B molecules relabeled 'C' by split_clusters → common set (leakage risk)
  6. B molecules still labeled 'B' → safe for training

Usage (exact uniprot match):
    python cluster_split/allocate_training.py \\
        --new-data cluster_split/data/new_data.csv.gz \\
        --split-dir cluster_split/data/h_bench/ \\
        --output-dir cluster_split/output_allocation

Usage (with UniRef90 protein grouping):
    python allocate_training.py \\
        --new-data data/new_data.csv.gz \\
        --split-dir data/h_bench/ \\
        -o output_allocation \\
        --uniref 90 --uniref-cache uniref_cache.json
"""

import sys
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path

import math
from collections import defaultdict
from scipy import sparse
from tqdm.auto import trange
import networkx as nx

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


import json
import time
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    radius=2, fpSize=2048, countSimulation=True,
)


def smiles_to_fps(smiles, n_jobs=8):
    """Convert SMILES strings to Morgan fingerprints (radius=2, 2048 bits).

    Parameters
    ----------
    smiles : list[str] | pd.Series | np.ndarray
        SMILES strings.
    n_jobs : int
        Number of parallel jobs for mol conversion.

    Returns
    -------
    list : RDKit fingerprint objects (None for invalid SMILES).
    """
    import joblib

    if isinstance(smiles, pd.Series):
        smiles = smiles.tolist()
    elif isinstance(smiles, np.ndarray):
        smiles = smiles.tolist()

    mols = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(Chem.MolFromSmiles)(smi) for smi in smiles
    )
    fps = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(lambda m: _MORGAN_GEN.GetFingerprint(m) if m is not None else None)(mol)
        for mol in mols
    )
    return fps


def hierarchical_complete_linkage(dist_matrix, threshold):
    """Complete-linkage hierarchical clustering via scipy.
    Converts sparse distance matrix to dense distance matrix for linkage.

    The sparse matrix stores 1-sim for entries where sim >= 1-threshold.
    Missing entries (zeros) represent pairs with sim < 1-threshold,
    i.e. distance > threshold -- these are set to 1.0 (max distance).
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    # Build 1-D condensed distance vector directly from sparse matrix.
    # Missing entries (not stored in sparse) are pairs with sim < 1-threshold,
    # i.e. distance > threshold — set those to 1.0 (max distance).
    n = dist_matrix.shape[0]
    n_pairs = n * (n - 1) // 2
    condensed = np.ones(n_pairs, dtype=np.float32)  # default = max distance

    coo = dist_matrix.tocoo()
    # Only fill upper triangle entries (i < j)
    mask = coo.row < coo.col
    rows, cols, dists = coo.row[mask], coo.col[mask], coo.data[mask]
    # Condensed index for (i, j) with i < j: n*i - i*(i+1)/2 + (j - i - 1)
    idx = (n * rows - rows * (rows + 1) // 2 + (cols - rows - 1)).astype(np.intp)
    condensed[idx] = dists

    Z = linkage(condensed, method='complete')
    labels = fcluster(Z, t=threshold, criterion='distance')
    cluster_dict = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_dict[label].append(i)
    return list(cluster_dict.values())


def compute_sparse_tanimoto_matrix_speed(fps, threshold=0.3, save_path=None):
    """Compute sparse Tanimoto similarity matrix.
    Only stores entries where similarity >= 1 - threshold.
    Returns a sparse DISTANCE matrix (dist = 1 - sim).
    """
    n = len(fps)
    rows = []
    cols = []
    data = []

    threshold_sim = 1.0 - threshold

    for i in trange(n, desc="Computing Tanimoto"):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
        local_valid = np.where(sims >= threshold_sim)[0]

        if local_valid.size > 0:
            global_indices = local_valid + (i + 1)
            rows.append(np.full(local_valid.size, i))
            cols.append(global_indices)
            data.append(sims[local_valid])

    if not data:
        sparse_matrix = sparse.csr_matrix((n, n))
    else:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        U = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        sparse_matrix = U + U.T

    # Convert similarity to distance
    sparse_matrix.data = 1.0 - sparse_matrix.data

    if save_path:
        sparse.save_npz(save_path, sparse_matrix)

    return sparse_matrix



def compute_cluster_centroids(clusters, fps):
    """Compute centroid fingerprint for each cluster (bitwise majority)."""
    centroids = []
    n_bits = fps[0].GetNumBits()
    for cluster in clusters:
        if len(cluster) == 1:
            centroids.append(fps[cluster[0]])
        else:
            avg_bits = np.zeros(n_bits)
            for idx in cluster:
                arr = np.zeros(n_bits)
                DataStructs.ConvertToNumpyArray(fps[idx], arr)
                avg_bits += arr
            avg_bits /= len(cluster)

            centroid = DataStructs.ExplicitBitVect(n_bits)
            on_bits = np.where(avg_bits >= 0.5)[0]
            for bit_idx in on_bits:
                centroid.SetBit(int(bit_idx))
            centroids.append(centroid)
    return centroids


def cluster_fps_butina(fps, cutoff=0.4):
    """Cluster fingerprints using Butina algorithm with Tanimoto distance.
    cutoff: Tanimoto distance threshold (1-similarity). 0.4 means similarity >= 0.6
    """
    from rdkit.ML.Cluster import Butina
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return clusters


def greedy_clique_cover(G):
    """Greedy clique cover: repeatedly find the largest clique, assign it as a
    cluster, remove those nodes, and repeat.
    Ensures every pair in a cluster has similarity >= threshold.
    """
    remaining = set(G.nodes())
    clusters = []

    while remaining:
        subgraph = G.subgraph(remaining)
        best_clique = None
        best_size = 0

        clique_iterator = nx.find_cliques(subgraph)
        max_cliques_to_check = 100000
        checked = 0

        for clique in clique_iterator:
            if len(clique) > best_size:
                best_clique = clique
                best_size = len(clique)
            checked += 1
            if checked >= max_cliques_to_check:
                break

        if best_clique is None or best_size == 0:
            for node in remaining:
                clusters.append([node])
            break

        clusters.append(sorted(best_clique))
        remaining -= set(best_clique)

    return clusters


def build_weighted_cluster_graph(dist_matrix_wide, threshold, cluster_labels, max_edge_length):
    """Build cluster graph with integer edge lengths (no dummy nodes).

    Edge length = ceil(tanimoto_distance / threshold), capped at max_edge_length.
    Only edges with length <= max_edge_length are added.

    Parameters
    ----------
    dist_matrix_wide : scipy.sparse matrix
        Sparse DISTANCE matrix (values = 1-sim).
    threshold : float
        Distance cutoff for clustering (same as cluster_threshold).
    cluster_labels : dict
        {node_id: 'A'|'B'|'C'} for cluster nodes.
    max_edge_length : int
        Maximum edge length to include. Pairs with distance > max_edge_length * threshold
        are not connected.

    Returns
    -------
    G : nx.Graph with 'length' edge attribute
    node_labels : dict (copy of cluster_labels)
    """
    G = nx.Graph()
    n_nodes = dist_matrix_wide.shape[0]
    G.add_nodes_from(range(n_nodes))

    coo = dist_matrix_wide.tocoo()
    # Upper triangle only
    upper_mask = coo.row < coo.col
    rows_ut = coo.row[upper_mask]
    cols_ut = coo.col[upper_mask]
    dists_ut = coo.data[upper_mask]

    n_by_length = defaultdict(int)
    for idx in range(len(rows_ut)):
        i, j = int(rows_ut[idx]), int(cols_ut[idx])
        d = float(dists_ut[idx])
        edge_length = max(1, math.ceil(d / threshold))
        if edge_length > max_edge_length:
            continue
        G.add_edge(i, j, length=edge_length)
        n_by_length[edge_length] += 1

    node_labels = dict(cluster_labels)

    print(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    for length in sorted(n_by_length):
        print(f"    length={length}: {n_by_length[length]} edges")
    return G, node_labels


def find_ab_pairs_within_distance(G, labels, max_jumps):
    """
    Find all (a, b) pairs where labels[a]=='A', labels[b]=='B',
    and weighted shortest-path distance(a,b) <= max_jumps.

    Uses Dijkstra with edge attribute 'length' as weight.
    Falls back to unweighted hop counting if edges have no 'length' attribute.
    """
    a_nodes = [n for n in G.nodes() if labels.get(n) == 'A']
    b_nodes_set = {n for n in G.nodes() if labels.get(n) == 'B'}

    # Check if graph has weighted edges
    has_weights = any('length' in d for _, _, d in G.edges(data=True))

    pairs = []
    for a in a_nodes:
        if has_weights:
            lengths = nx.single_source_dijkstra_path_length(
                G, a, cutoff=max_jumps, weight='length')
        else:
            lengths = nx.single_source_shortest_path_length(G, a, cutoff=max_jumps)
        for node, dist in lengths.items():
            if node in b_nodes_set and dist <= max_jumps:
                pairs.append((a, node, dist))

    return pairs


def _get_changeable_nodes(labels, relabel_from, frozen_nodes):
    """
    Return the set of nodes that are allowed to be relabeled to C.
    A node is changeable if:
      1. Its current label is in the allowed set (per relabel_from)
      2. It is NOT in frozen_nodes
    """
    if relabel_from == 'A':
        allowed_labels = {'A'}
    elif relabel_from == 'B':
        allowed_labels = {'B'}
    else:
        allowed_labels = {'A', 'B'}

    changeable = set()
    for n, l in labels.items():
        if l in allowed_labels and n not in frozen_nodes:
            changeable.add(n)

    return changeable


def relabel_greedy(G, labels, max_jumps=2, relabel_from='both', frozen_nodes=None):
    """
    Greedy vertex cover. Only picks changeable nodes.
    """
    if frozen_nodes is None:
        frozen_nodes = set()

    new_labels = labels.copy()
    c_nodes = set()

    while True:
        pairs = find_ab_pairs_within_distance(G, new_labels, max_jumps)
        if not pairs:
            break

        changeable = _get_changeable_nodes(new_labels, relabel_from, frozen_nodes)

        # Count conflicts per changeable node
        node_count = defaultdict(int)
        for a, b, d in pairs:
            if a in changeable:
                node_count[a] += 1
            if b in changeable:
                node_count[b] += 1

        if not node_count:
            print(f"  WARNING: {len(pairs)} unresolvable (no changeable nodes left)")
            break

        best = max(node_count, key=node_count.get)
        new_labels[best] = 'C'
        c_nodes.add(best)

    return new_labels, c_nodes


def relabel_ilp(G, labels, max_jumps=2, relabel_from='both', frozen_nodes=None):
    """
    Exact minimum via ILP.

    For every (a, b) with a in A, b in B, dist(a,b) <= max_jumps:
        x_a + x_b >= 1   (at least one must become C)

    Nodes in frozen_nodes have x_n fixed to 0 (cannot change).
    relabel_from further restricts which labels may change.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    if frozen_nodes is None:
        frozen_nodes = set()

    pairs = find_ab_pairs_within_distance(G, labels, max_jumps)

    if not pairs:
        print("  No A-B conflicts!")
        return labels.copy(), set()

    changeable = _get_changeable_nodes(labels, relabel_from, frozen_nodes)

    # Check feasibility
    infeasible = [(a, b, d) for a, b, d in pairs
                  if a not in changeable and b not in changeable]
    if infeasible:
        print(f"  WARNING: {len(infeasible)} pairs where neither endpoint is changeable")

    # Candidate nodes = all A/B nodes that appear in conflict pairs
    candidate_set = set()
    for a, b, d in pairs:
        candidate_set.add(a)
        candidate_set.add(b)

    candidate_list = sorted(candidate_set)
    idx_of = {n: i for i, n in enumerate(candidate_list)}
    n_vars = len(candidate_list)

    # Deduplicate pair constraints
    unique_pairs = set()
    for a, b, d in pairs:
        unique_pairs.add((min(a, b), max(a, b)))

    n_changeable_cands = sum(1 for n in candidate_list if n in changeable)
    print(f"  ILP: {n_vars} candidates ({n_changeable_cands} changeable, "
          f"{n_vars - n_changeable_cands} frozen), {len(unique_pairs)} constraints")

    # Constraint matrix: -x_a - x_b <= -1 for each pair
    n_con = len(unique_pairs)
    A_matrix = np.zeros((n_con, n_vars))
    for ci, (a, b) in enumerate(unique_pairs):
        A_matrix[ci, idx_of[a]] = -1.0
        A_matrix[ci, idx_of[b]] = -1.0
    b_upper = np.full(n_con, -1.0)

    # Bounds: frozen/disallowed nodes have ub=0
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    for i, n in enumerate(candidate_list):
        if n not in changeable:
            ub[i] = 0.0

    result = milp(
        c=np.ones(n_vars),
        constraints=LinearConstraint(A_matrix, ub=b_upper),
        integrality=np.ones(n_vars),
        bounds=Bounds(lb=lb, ub=ub),
    )

    if not result.success:
        print(f"  ILP failed: {result.message}, using greedy")
        return relabel_greedy(G, labels, max_jumps, relabel_from, frozen_nodes)

    c_nodes = {candidate_list[i] for i, v in enumerate(result.x) if v > 0.5}
    new_labels = labels.copy()
    for n in c_nodes:
        new_labels[n] = 'C'

    return new_labels, c_nodes


def relabel_minimum_buffer(G, labels, max_jumps=2, relabel_from='both',
                           frozen_nodes=None, method='auto'):
    """
    Main API for minimum buffer relabeling.

    Parameters
    ----------
    G : nx.Graph
        Cluster similarity graph (may include dummy C nodes)
    labels : dict
        {node_id: 'A' | 'B' | 'C'} for all nodes
    max_jumps : int
        No remaining A within this many hops of any remaining B
    relabel_from : str
        'A'    -- only A nodes can become C
        'B'    -- only B nodes can become C
        'both' -- either can become C (fewest total C)
    frozen_nodes : set, optional
        Nodes that must NEVER change label (e.g. pre-placed dummy C nodes).
    method : str
        'ilp' | 'greedy' | 'auto'

    Returns
    -------
    new_labels : dict -- updated labels
    c_nodes : set     -- which nodes were NEWLY changed to C
    """

    assert relabel_from in ('A', 'B', 'both'), \
        f"relabel_from must be 'A', 'B', or 'both', got '{relabel_from}'"

    if frozen_nodes is None:
        frozen_nodes = set()

    n = G.number_of_nodes()
    n_a = sum(1 for l in labels.values() if l == 'A')
    n_b = sum(1 for l in labels.values() if l == 'B')
    n_c_existing = sum(1 for l in labels.values() if l == 'C')

    print(f"\n{'='*60}")
    print(f"  MINIMUM BUFFER RELABELING")
    print(f"  relabel_from='{relabel_from}' | max_jumps={max_jumps}")
    print(f"{'='*60}")
    print(f"  Graph: {n} nodes, {G.number_of_edges()} edges")
    print(f"  Labels: {n_a} A, {n_b} B, {n_c_existing} C (pre-existing)")
    print(f"  Frozen nodes: {len(frozen_nodes)}")

    desc = {'A': 'Only A->C (B frozen)',
            'B': 'Only B->C (A frozen)',
            'both': 'A or B->C'}
    print(f"  Mode: {desc[relabel_from]}")

    pairs = find_ab_pairs_within_distance(G, labels, max_jumps)
    print(f"  A-B pairs within {max_jumps} hops: {len(pairs)}")

    if not pairs:
        print("  Already separated (pre-existing C nodes provide enough buffer)!")
        return labels.copy(), set()

    if method == 'auto':
        method = 'ilp' if n < 1000000 else 'greedy'
    print(f"  Method: {method}")

    if method == 'ilp':
        new_labels, c_nodes = relabel_ilp(G, labels, max_jumps, relabel_from, frozen_nodes)
    else:
        new_labels, c_nodes = relabel_greedy(G, labels, max_jumps, relabel_from, frozen_nodes)

    # Validate + greedy post-fix
    remaining = find_ab_pairs_within_distance(G, new_labels, max_jumps)
    if remaining:
        print(f"  Post-fix: {len(remaining)} residual conflicts...")
        fix_labels, fix_nodes = relabel_greedy(
            G, new_labels, max_jumps, relabel_from, frozen_nodes)
        new_labels = fix_labels
        c_nodes = c_nodes | fix_nodes
        remaining = find_ab_pairs_within_distance(G, new_labels, max_jumps)

    # Stats
    n_a2 = sum(1 for l in new_labels.values() if l == 'A')
    n_b2 = sum(1 for l in new_labels.values() if l == 'B')
    n_c2 = sum(1 for l in new_labels.values() if l == 'C')
    from_a = sum(1 for n in c_nodes if labels[n] == 'A')
    from_b = sum(1 for n in c_nodes if labels[n] == 'B')

    print(f"\n  RESULT:")
    print(f"    A: {n_a} -> {n_a2}  ({from_a} became C)")
    print(f"    B: {n_b} -> {n_b2}  ({from_b} became C)")
    print(f"    C: {n_c_existing} pre-existing + {len(c_nodes)} new = {n_c2} total")
    print(f"    Remaining conflicts: {len(remaining)}")

    print(f"{'='*60}")
    return new_labels, c_nodes




def split_clusters(df, cluster_threshold=0.25, max_jumps=2,
                   relabel_from='both', method='auto',
                   clustering_method='complete_linkage', dbscan_eps=0.2,
                   smiles_key='SMILES', label_key='label', label_map=None,
                   ligand_threshold=None):
    """Main pipeline: cluster SMILES, label A/B/C, build graph, relabel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'SMILES' and 'label' columns.
        Labels should be 'HARVEST' (mapped to A) and 'BDB' (mapped to B).
    clustering_method : str
        'clique' — greedy clique cover on similarity graph (default)
        'butina' — Butina centroid-based clustering
        'dbscan' — DBSCAN on precomputed distance matrix
        'complete_linkage' — hierarchical complete-linkage clustering
    dbscan_eps : float
        DBSCAN eps parameter (only used when clustering_method='dbscan')

    Returns
    -------
    dict with keys: df_valid, clusters, final_cluster_labels, fps,
                    mol_to_cluster, centroids
    """

    # Default ligand_threshold to cluster_threshold for backward compat
    if ligand_threshold is None:
        ligand_threshold = cluster_threshold

    # ── Step 1: Load & Fingerprint ──
    print("=" * 70)
    print("CLUSTER-BASED SMILES SPLIT")
    print("=" * 70)

    if label_map is None:
        label_map = {'HARVEST': 'A', 'BDB': 'B'}

    assert smiles_key in df.columns, f"Missing column '{smiles_key}'"
    assert label_key in df.columns, f"Missing column '{label_key}'"
    df = df[[smiles_key, label_key]].copy()
    df['source_label'] = df[label_key].map(label_map)
    assert df['source_label'].notna().all(), f"Unknown labels: {df[label_key].unique()}"

    # ── Step 1: Fingerprints ──
    print(f"\n[1/6] Loading {len(df)} SMILES...")
    print(f"  HARVEST: {(df['source_label']=='A').sum()}, BDB: {(df['source_label']=='B').sum()}")

    fps_all = smiles_to_fps(df[smiles_key].tolist())
    valid_mask = np.array([fp is not None for fp in fps_all])
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  WARNING: {n_invalid} invalid SMILES skipped")
    df_valid = df[valid_mask].reset_index(drop=True)
    fps = [fp for fp in fps_all if fp is not None]

    source_labels = df_valid['source_label'].values
    print(f"  Valid: {len(fps)} fingerprints")

    # ── Step 2: Cluster molecules ──
    print(f"\n[2/6] Clustering (method={clustering_method}, ligand_threshold={ligand_threshold})...")
    dist_matrix = None
    if clustering_method in ['clique', 'dbscan', 'complete_linkage']:
        dist_matrix = compute_sparse_tanimoto_matrix_speed(fps, threshold=ligand_threshold)

    if clustering_method == 'butina':
        butina_clusters = cluster_fps_butina(fps, cutoff=ligand_threshold)
        clusters = [list(c) for c in butina_clusters]

    elif clustering_method == 'clique':
        G_mol = nx.from_scipy_sparse_array(dist_matrix)
        print(f"  Similarity graph: {G_mol.number_of_nodes()} nodes, {G_mol.number_of_edges()} edges")
        clusters = greedy_clique_cover(G_mol)
    elif clustering_method == 'dbscan':
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=dbscan_eps, min_samples=5, metric='precomputed', n_jobs=-1)
        db_labels = db.fit_predict(dist_matrix)
        cluster_dict = defaultdict(list)
        for idx, lab in enumerate(db_labels):
            cluster_dict[lab].append(idx)
        clusters = []
        for lab, members in cluster_dict.items():
            if lab == -1:
                for m in members:
                    clusters.append([m])
            else:
                clusters.append(members)
    elif clustering_method == 'complete_linkage':
        clusters = hierarchical_complete_linkage(dist_matrix, threshold=ligand_threshold)
    else:
        raise ValueError(f"Unknown clustering_method: {clustering_method}")

    clusters.sort(key=len, reverse=True)

    print(f"  {len(clusters)} clusters found")
    print(f"  Largest: {len(clusters[0])}, Singletons: {sum(1 for c in clusters if len(c)==1)}")

    # ── Step 3: Label clusters A/B/C ──
    print(f"\n[3/6] Labeling clusters...")
    cluster_labels = {}
    mixed_cluster_ids = {}
    for ci, members in enumerate(clusters):
        member_labels = source_labels[members]
        unique = set(member_labels)
        if unique == {'A'}:
            cluster_labels[ci] = 'A'
        elif unique == {'B'}:
            cluster_labels[ci] = 'B'
        elif relabel_from == 'A':  # consider all common as B
            cluster_labels[ci] = 'B'
            mixed_cluster_ids[ci] = 'C'
        elif relabel_from == 'B':  # consider all common as A
            cluster_labels[ci] = 'A'
            mixed_cluster_ids[ci] = 'C'
        else:
            cluster_labels[ci] = 'C'

    label_map_inv = {v: k for k, v in label_map.items()}
    n_a = sum(1 for v in cluster_labels.values() if v == 'A')
    n_b = sum(1 for v in cluster_labels.values() if v == 'B')
    n_c = sum(1 for v in cluster_labels.values() if v == 'C')

    # Frozen = mixed-source clusters only (no more dummy nodes)
    frozen_nodes = {ci for ci, lab in cluster_labels.items() if lab == 'C'}

    print(f"  A (HARVEST-only): {n_a} clusters")
    print(f"  B (BDB-only): {n_b} clusters")
    print(f"  C (mixed): {n_c} clusters")

    # ── Single-source early exit ──
    unique_source_labels = set(df_valid['source_label'].unique())
    if len(unique_source_labels) == 1:
        the_label = unique_source_labels.pop()
        print(f"\n  Single source detected ('{label_map_inv.get(the_label, the_label)}') — "
              f"skipping graph building and relabeling (steps 4-5).")
        final_cluster_labels = {ci: the_label for ci in cluster_labels}

        # ── Step 6 (early): Map back to molecules ──
        print(f"\n[6/6] Mapping to molecules...")
        mol_to_cluster = np.full(len(df_valid), -1, dtype=int)
        for ci, members in enumerate(clusters):
            mol_to_cluster[np.array(members)] = ci

        df_valid = df_valid.copy()
        df_valid['cluster_id'] = mol_to_cluster
        df_valid['final_label'] = the_label
        df_valid['nearest_smiles'] = ''
        df_valid['tanimoto_sim'] = 0.0

        print(f"\n  Molecule counts:")
        print(f"    {the_label} ({label_map_inv.get(the_label, the_label)}): {len(df_valid)}")

        centroids = compute_cluster_centroids(clusters, fps)

        print(f"\n{'='*70}")
        print("DONE")
        print(f"{'='*70}")

        return {
            'df_valid': df_valid,
            'clusters': clusters,
            'final_cluster_labels': final_cluster_labels,
            'fps': fps,
            'mol_to_cluster': mol_to_cluster,
            'centroids': centroids,
        }

    # ── Step 4: Build weighted cluster graph ──
    print(f"\n[4/6] Building cluster graph (threshold={cluster_threshold})...")
    wide_threshold = min(max_jumps * cluster_threshold, 0.95)

    centroids = compute_cluster_centroids(clusters, fps)
    print(f"  Computed {len(centroids)} centroid fingerprints")

    centroid_dist_wide = compute_sparse_tanimoto_matrix_speed(
        centroids, threshold=wide_threshold)

    print(f"  Centroid distance matrix (wide, thresh={wide_threshold:.3f}): "
          f"{centroid_dist_wide.nnz} non-zero entries")

    G, node_labels = build_weighted_cluster_graph(
        centroid_dist_wide, threshold=cluster_threshold,
        cluster_labels=cluster_labels, max_edge_length=max_jumps)

    # ── Step 5: Relabel with buffer zone ──
    print(f"\n[5/6] Relabeling (max_jumps={max_jumps}, relabel_from='{relabel_from}')...")
    new_labels, c_nodes = relabel_minimum_buffer(
        G, node_labels,
        max_jumps=max_jumps,
        relabel_from=relabel_from,
        frozen_nodes=frozen_nodes,
        method=method,
    )

    final_cluster_labels = new_labels

    # Reassign untouched mixed clusters to C
    if relabel_from != 'both':
        for ci in mixed_cluster_ids:
            if ci not in c_nodes:
                final_cluster_labels[ci] = 'CB'  # common clusters on a boarder

    # Stats after relabeling
    n_a2 = sum(1 for v in final_cluster_labels.values() if v == 'A')
    n_b2 = sum(1 for v in final_cluster_labels.values() if v == 'B')
    n_c2 = sum(1 for v in final_cluster_labels.values() if v == 'C')
    n_cb = sum(1 for v in final_cluster_labels.values() if v == 'CB')
    print(f"\n  After relabeling:")
    print(f"    A: {n_a} → {n_a2}")
    print(f"    B: {n_b} → {n_b2}")
    print(f"    C: {n_c} → {n_c2}")
    print(f"    CB: {0} → {n_cb}")

    # ── Step 6: Map back to molecules + nearest other-source molecule ──
    print(f"\n[6/6] Mapping to molecules & computing nearest other-source similarity...")

    # Assign cluster IDs to molecules
    mol_to_cluster = np.full(len(df_valid), -1, dtype=int)
    for ci, members in enumerate(clusters):
        mol_to_cluster[np.array(members)] = ci

    df_valid = df_valid.copy()
    df_valid['cluster_id'] = mol_to_cluster
    df_valid['final_label'] = df_valid['cluster_id'].map(final_cluster_labels)

    # Pairwise centroid similarity (dense, small matrix)
    n_cl = len(clusters)
    centroid_sim_dense = np.zeros((n_cl, n_cl), dtype=np.float32)
    for i in range(n_cl):
        sims = DataStructs.BulkTanimotoSimilarity(centroids[i], centroids)
        centroid_sim_dense[i] = sims

    # Build per-source cluster index
    clusters_with_a = {ci for ci, members in enumerate(clusters)
                       if any(source_labels[m] == 'A' for m in members)}
    clusters_with_b = {ci for ci, members in enumerate(clusters)
                       if any(source_labels[m] == 'B' for m in members)}

    # For each cluster, find nearest cluster containing molecules from the other source.
    # Mixed clusters (both A and B sources) include themselves so that intra-cluster
    # other-source molecules are considered at the molecule level.
    nearest_other_source_cluster = {}
    for ci in range(n_cl):
        has_a = ci in clusters_with_a
        has_b = ci in clusters_with_b
        if has_a and not has_b:
            target_clusters = clusters_with_b  # pure A → look at B clusters
        elif has_b and not has_a:
            target_clusters = clusters_with_a  # pure B → look at A clusters
        else:
            # Mixed cluster: include self so molecule-level loop finds
            # other-source neighbors within the same cluster
            target_clusters = clusters_with_a | clusters_with_b
        if not target_clusters:
            continue
        sims = centroid_sim_dense[ci].copy()
        mask = np.zeros(n_cl, dtype=bool)
        for tc in target_clusters:
            mask[tc] = True
        sims[~mask] = -1
        nearest_other_source_cluster[ci] = int(np.argmax(sims))

    # Per-molecule: find nearest molecule from other source in the nearest other-source cluster
    smiles_list = df_valid[smiles_key].values
    nearest_smiles_arr = np.full(len(df_valid), '', dtype=object)
    nearest_sim_arr = np.full(len(df_valid), 0.0, dtype=np.float32)

    for ci, members in enumerate(clusters):
        if ci not in nearest_other_source_cluster:
            continue
        target_ci = nearest_other_source_cluster[ci]
        target_members = clusters[target_ci]

        for mol_idx in members:
            my_source = source_labels[mol_idx]
            other_mols = [m for m in target_members if source_labels[m] != my_source]
            if not other_mols:
                continue
            other_fps = [fps[m] for m in other_mols]
            sims = DataStructs.BulkTanimotoSimilarity(fps[mol_idx], other_fps)
            best_j = int(np.argmax(sims))
            nearest_smiles_arr[mol_idx] = smiles_list[other_mols[best_j]]
            nearest_sim_arr[mol_idx] = sims[best_j]

    df_valid['nearest_smiles'] = nearest_smiles_arr
    df_valid['tanimoto_sim'] = nearest_sim_arr

    # Summary
    print(f"\n  Molecule counts:")
    for lab, name in [('A', 'HARVEST-unique'), ('B', 'BDB-unique'), ('C', 'Common/Buffer') , ('CB', 'Same/Boarder')]:
        count = (df_valid['final_label'] == lab).sum()
        print(f"    {lab} ({name}): {count}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")

    return {
        'df_valid': df_valid,
        'clusters': clusters,
        'final_cluster_labels': final_cluster_labels,
        'fps': fps,
        'mol_to_cluster': mol_to_cluster,
        'centroids': centroids,
    }




UNIREF_SEARCH_URL = "https://rest.uniprot.org/uniref/search"
REQUEST_DELAY = 0.5


def build_uniref_mapping(accessions, identity='90', cache_path=None):
    """Query UniProt API for UniRef clusters of given accessions.

    Parameters
    ----------
    accessions : list[str]
        UniProt accessions to look up.
    identity : str
        UniRef identity level: '50', '90', or '100'.
    cache_path : str or Path, optional
        Path to JSON cache file (loads/saves API results).

    Returns
    -------
    acc_to_cluster : dict
        {accession: cluster_id} for all queried accessions.
    cluster_to_accs : dict
        {cluster_id: set(accessions)} grouping accessions by cluster.
    """
    cache = {}
    cache_exists = cache_path and Path(cache_path).exists()
    if cache_exists:
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  UniRef cache loaded: {len(cache)} entries from {cache_path}")

    identity_map = {'50': '0.5', '90': '0.9', '100': '1.0'}
    identity_val = identity_map.get(identity, '0.9')

    to_query = [a for a in accessions if a not in cache]
    print(f"  UniRef{identity}: {len(accessions)} accessions, "
          f"{len(accessions) - len(to_query)} cached, {len(to_query)} to query")

    if cache_exists and to_query:
        print(f"  Using existing cache only — {len(to_query)} accessions not in cache will be skipped")
        to_query = []

    if to_query:
        session = requests.Session()
        session.headers.update({'Accept': 'application/json'})

        batch_size = 10
        batches = [to_query[i:i + batch_size] for i in range(0, len(to_query), batch_size)]

        for batch in trange(len(batches), desc=f"Querying UniRef{identity}"):
            accs = batches[batch]
            id_parts = ' OR '.join(f'uniprot_id:{acc}' for acc in accs)
            query = f'({id_parts}) AND identity:{identity_val}'
            params = {
                'query': query,
                'fields': 'id,members',
                'format': 'json',
                'size': 500,
            }
            try:
                resp = session.get(UNIREF_SEARCH_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                for entry in data.get('results', []):
                    cluster_id = entry.get('id', '')
                    members = set(entry.get('members', []))
                    for acc in accs:
                        if acc in members:
                            cache[acc] = cluster_id

            except requests.exceptions.RequestException as e:
                print(f"\n  API error for batch starting with {accs[0]}: {e}")

            # Mark missing as NOT_FOUND
            for acc in accs:
                if acc not in cache:
                    cache[acc] = 'NOT_FOUND'

            time.sleep(REQUEST_DELAY)

        # Save cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"  UniRef cache saved: {len(cache)} entries to {cache_path}")

    # Build mappings
    acc_to_cluster = {a: cache.get(a, 'NOT_FOUND') for a in accessions}
    cluster_to_accs = defaultdict(set)
    for acc, cl in acc_to_cluster.items():
        if cl != 'NOT_FOUND':
            cluster_to_accs[cl].add(acc)

    n_found = sum(1 for v in acc_to_cluster.values() if v != 'NOT_FOUND')
    n_clusters = len(cluster_to_accs)
    print(f"  UniRef{identity}: {n_found}/{len(accessions)} mapped to {n_clusters} clusters")

    return acc_to_cluster, cluster_to_accs


def allocate_training(data_dir, df_new,
                      cluster_threshold,
                      ligand_threshold,
                      clustering_method,
                      max_jumps,
                      uniprot_col_new='uniprot_id',
                      uniref=None,
                      uniref_cache=None,
                      smiles_col_new='smiles_clean',
                      limit=None):

    data_dir = Path(data_dir)
    df_new = df_new.copy()
    df_new['final_label'] = 'B'
    uniprots = [Path(f).stem for f in glob.glob(str(data_dir / '*.csv'))]
    if limit is not None:
        uniprots = uniprots[:limit]

    # Build UniRef cluster mapping if requested
    cluster_to_accs = None
    acc_to_cluster = None
    if uniref is not None:
        # Collect all unique accessions from both h_bench and new data
        all_accs = set(uniprots) | set(df_new[uniprot_col_new].dropna().unique())
        acc_to_cluster, cluster_to_accs = build_uniref_mapping(
            sorted(all_accs), identity=uniref, cache_path=uniref_cache)

    report_rows = []

    # Group h_bench proteins by UniRef cluster to avoid processing the same
    # cluster multiple times (which inflates the report and duplicates work)
    if acc_to_cluster is not None:
        # Build cluster_id -> list of h_bench proteins
        cluster_groups = defaultdict(list)
        no_cluster = []
        for u in uniprots:
            cl = acc_to_cluster.get(u, 'NOT_FOUND')
            if cl != 'NOT_FOUND':
                cluster_groups[cl].append(u)
            else:
                no_cluster.append(u)
        # Each group: process once with combined test set A from all h_bench files in cluster
        protein_jobs = []
        for cl_id, members in cluster_groups.items():
            protein_jobs.append((members, cl_id))
        for u in no_cluster:
            protein_jobs.append(([u], None))
    else:
        protein_jobs = [([u], None) for u in uniprots]

    for h_bench_proteins, cluster_id in protein_jobs:
        # Load and combine test set A from all h_bench files in this group
        smiles_a_parts = []
        for u in h_bench_proteins:
            df_h = pd.read_csv(data_dir / f'{u}.csv')
            smiles_a_parts.append(df_h[df_h.final_label == 'A']['SMILES'].unique())
        smiles_a = np.unique(np.concatenate(smiles_a_parts)) if smiles_a_parts else np.array([])

        # Build protein mask for new data
        if cluster_id is not None:
            cluster_members = cluster_to_accs.get(cluster_id, set())
            protein_mask = df_new[uniprot_col_new].isin(cluster_members)
            matched_uniprots = df_new[protein_mask][uniprot_col_new].unique()
            label = (f'{",".join(h_bench_proteins)} (UniRef{uniref} cluster {cluster_id}, '
                     f'{len(matched_uniprots)} matched uniprots)')
        else:
            protein_mask = df_new[uniprot_col_new] == h_bench_proteins[0]
            label = f'{h_bench_proteins[0]} (exact match)'
        print(f'Processing {label}')

        # Create set B (drop nulls)
        smiles_b = df_new[protein_mask][smiles_col_new].dropna().unique()

        # Remove Set A from new df
        smiles_b = np.setdiff1d(smiles_b, smiles_a)
        # Count exact duplicates: rows in new data whose SMILES match test set A
        mask_exact = protein_mask & df_new[smiles_col_new].isin(smiles_a)
        n_exact_dup = int(mask_exact.sum())
        df_new.loc[mask_exact, 'final_label'] = 'A'

        if len(smiles_b) == 0:
            print(f'  No new molecules, skipping')
            report_rows.append({
                'h_bench_protein': ';'.join(h_bench_proteins),
                'uniref_cluster': cluster_id or '',
                'n_new_total': int(protein_mask.sum()),
                'n_exact_duplicates': n_exact_dup,
                'n_removed_buffer': 0,
                'n_safe_training': 0,
            })
            continue

        df_combined = pd.concat([
            pd.DataFrame({'clean_smiles': smiles_a, 'label': 'SET_A'}),
            pd.DataFrame({'clean_smiles': smiles_b, 'label': 'SET_B'}),
        ], ignore_index=True)

        df_combined = df_combined.drop_duplicates(subset='clean_smiles', keep='first')

        result = split_clusters(
            df_combined,
            smiles_key='clean_smiles',
            label_key='label',
            label_map={'SET_A': 'A', 'SET_B': 'B'},
            relabel_from='B',
            cluster_threshold=cluster_threshold,
            ligand_threshold=ligand_threshold,
            max_jumps=max_jumps,
            clustering_method=clustering_method,
        )

        df_result = result['df_valid']

        # Remove common molecules (B -> C or CB)
        mask_b2c = (df_result.source_label == 'B') & (df_result.final_label.isin(['C', 'CB']))
        smiles_removed = set(df_result[mask_b2c]['clean_smiles'].tolist())
        mask = protein_mask & df_new[smiles_col_new].isin(smiles_removed)
        df_new.loc[mask, 'final_label'] = 'C'

        n_removed = len(smiles_removed)
        n_safe = len(smiles_b) - n_removed

        report_rows.append({
            'h_bench_protein': ';'.join(h_bench_proteins),
            'uniref_cluster': cluster_id or '',
            'n_new_total': int(protein_mask.sum()),
            'n_exact_duplicates': n_exact_dup,
            'n_removed_buffer': n_removed,
            'n_safe_training': n_safe,
        })

        print(f'  => {";".join(h_bench_proteins)}: {n_exact_dup} exact dups, '
              f'{n_removed} removed (buffer), {n_safe} safe for training')

    df_report = pd.DataFrame(report_rows)
    return df_new, df_report

def print_report(df_new_labeled, df_report):
    alloc_counts = df_new_labeled['final_label'].value_counts()
    n_total = len(df_new_labeled)
    n_B = int(alloc_counts.get('B', 0))
    n_A = int(alloc_counts.get('A', 0))
    n_C = int(alloc_counts.get('C', 0))

    print(f"\n{'=' * 60}")
    print("ALLOCATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total molecules:        {n_total:,}")
    print(f"  B (safe for training):  {n_B:,} ({n_B / n_total * 100:.1f}%)")
    print(f"  A (exact test dups):    {n_A:,} ({n_A / n_total * 100:.1f}%)")
    print(f"  C (removed buffer):     {n_C:,} ({n_C / n_total * 100:.1f}%)")
    print(f"  Total removed (A+C):    {n_A + n_C:,} ({(n_A + n_C) / n_total * 100:.1f}%)")


    # Show top proteins by removals
    if len(df_report) > 0 and 'n_removed_buffer' in df_report.columns:
        df_top = df_report.nlargest(10, 'n_removed_buffer')
        df_top = df_top[df_top['n_removed_buffer'] > 0]
        if len(df_top) > 0:
            print(f"\nTop proteins by buffer removals:")
            for _, row in df_top.iterrows():
                print(f"  {row['h_bench_protein']}: "
                      f"{row['n_removed_buffer']} removed, "
                      f"{row['n_safe_training']} safe, "
                      f"{row['n_exact_duplicates']} exact dups")

def main():
    # Data dirs
    parser = argparse.ArgumentParser(
        description='Allocate new data for training without test set leakage')
    parser.add_argument('--new-data', type=Path, help='Path to new data in csv/csv.gz/parquet format',
                        required=True)
    parser.add_argument('-o', '--output-dir', help='Output directory for results', type=Path, required=True)
    parser.add_argument('--split-dir', default='./data/h_bench/', type=Path,
                        help='Path to H-bench directory with per-protein split files named {uniprot_id}.csv. '
                             'Each CSV must contain columns: SMILES, final_label.')

    # New data column names
    parser.add_argument('--smiles-col-new', default='SMILES',
                        help='SMILES column name in new data (default: SMILES)')
    parser.add_argument('--uniprot-col-new', default='uniprot_acc',
                        help='Column name in new data with Uniprot IDs (default: uniprot_acc) '
                             'to match sequences with H-Bench.')
    parser.add_argument('--uniref', default='90', choices=['50', '90', '100', 'none'],
                        help='UniRef identity level for protein grouping (default: 90). '
                             'Proteins sharing a UniRef cluster are grouped together. '
                             'Use "none" for exact uniprot ID matching only.')
    parser.add_argument('--uniref-cache', default=None, type=Path,
                        help='Path to JSON cache for UniRef API queries')

    # Clusterization params
    parser.add_argument('--cluster-threshold', type=float, default=0.225,
                        help='Tanimoto distance cutoff for cluster graph (default: 0.225)')
    parser.add_argument('--ligand-threshold', type=float, default=0.2,
                        help='Tanimoto distance cutoff for molecule clustering (default: 0.2)')
    parser.add_argument('--max-jumps', type=int, default=2,
                        help='Max graph hops for relabelling (default: 2)')
    parser.add_argument('--clustering-method', default='complete_linkage',
                        choices=['butina', 'clique', 'dbscan', 'complete_linkage'],
                        help='Clustering method (default: complete_linkage)')

    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of families to process (for debugging)')
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data...')
    if '.csv' in args.new_data.suffixes:
        df_new = pd.read_csv(args.new_data)
    elif '.parquet' in  args.new_data.suffixes:
        df_new = pd.read_parquet(args.new_data)
    else:
        raise f"File format {args.new_data.suffixes} is not recognized"

    required_cols = [args.smiles_col_new]
    missing = [c for c in required_cols if c not in df_new.columns]
    assert not missing, (
        f"Column(s) {missing} not found in {args.new_data}. "
        f"Available columns: {list(df_new.columns)}"
    )

    uniref_val = args.uniref if args.uniref != 'none' else None

    print('Merge new data with H-BENCH...')
    df_new_labeled, df_report = allocate_training(
        data_dir=args.split_dir,
        df_new=df_new,
        cluster_threshold=args.cluster_threshold,
        ligand_threshold=args.ligand_threshold,
        clustering_method=args.clustering_method,
        max_jumps=args.max_jumps,
        smiles_col_new=args.smiles_col_new,
        uniprot_col_new=args.uniprot_col_new,
        uniref=uniref_val,
        uniref_cache=args.uniref_cache,
        limit=args.limit,
    )
    # Save in same format as input
    input_suffixes = args.new_data.suffixes
    if '.parquet' in input_suffixes:
        out_path = output_dir / "df_with_splits.parquet"
        df_new_labeled.to_parquet(out_path, index=False)
    else:
        out_path = output_dir / "df_with_splits.csv.gz"
        df_new_labeled.to_csv(out_path, index=False, compression='gzip')
    print(f"Labeled data: {out_path}")

    # Save and print allocation report
    print(f"\nPer-protein report: {output_dir / 'allocation_report.csv'}")
    df_report.to_csv(output_dir / "allocation_report.csv", index=False)
    print_report(df_new_labeled, df_report)

if __name__ == '__main__':
    main()
