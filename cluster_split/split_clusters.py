"""
Cluster-Based SMILES Split: A (HARVEST) / B (BDB) / C (Common/Buffer)
=====================================================================

Clusters combined SMILES from two datasets, labels clusters as A/B/C,
builds a cluster connectivity graph with weighted edges encoding
multi-hop distances, and uses relabel_minimum_buffer to refine the buffer zone.

Usage:
    python cluster_split/split_clusters.py --input cluster_split/data.csv
"""

import sys
import math
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy import sparse
from pathlib import Path
from tqdm.auto import trange

from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdFingerprintGenerator import GetMorganFeatureAtomInvGen
RDLogger.logger().setLevel(RDLogger.ERROR)

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprints
# ─────────────────────────────────────────────────────────────────────────────

# FCFP-style Morgan: feature atom invariants + count simulation. These are the
# exact parameters the published splits were built with, so they must not be
# changed without regenerating the released cluster_id/tanimoto_sim columns.
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    atomInvariantsGenerator=GetMorganFeatureAtomInvGen(),
    radius=2, fpSize=2048, countSimulation=True,
)


def smiles_to_fps(smiles, finger_type='morgan', n_jobs=8):
    """SMILES -> RDKit fingerprints, ``None`` for anything unparseable.

    Only 'morgan' is supported; the argument is kept so existing callers and
    the ``--fps-type`` CLI flag keep working.
    """
    if finger_type != 'morgan':
        raise NotImplementedError(
            f"finger_type={finger_type!r} is not supported; use 'morgan'.")

    if isinstance(smiles, pd.Series):
        smiles = smiles.tolist()

    def _one(smi):
        if not isinstance(smi, str) or not smi:
            return None
        mol = Chem.MolFromSmiles(smi)
        return None if mol is None else _MORGAN_GEN.GetFingerprint(mol)

    if n_jobs and n_jobs > 1 and len(smiles) > 1000:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_one)(smi) for smi in smiles)
    return [_one(smi) for smi in smiles]


# ─────────────────────────────────────────────────────────────────────────────
# Functions extracted from paper_harvest.ipynb
# ─────────────────────────────────────────────────────────────────────────────

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


def reduce_cutoff_distance_sparce_matrix(arr, threshold=0.3):
    """Filter sparse distance matrix to keep only entries with sim >= threshold
    (i.e., distance <= 1 - threshold)."""
    arr = arr.copy()
    arr.data = 1. - arr.data  # convert to similarity
    arr.data[arr.data < threshold] = 0.  # zero out below threshold
    arr.eliminate_zeros()
    arr.data = 1. - arr.data  # convert back to distance
    return arr


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


# ─────────────────────────────────────────────────────────────────────────────
# Similarity histogram plotting
# ─────────────────────────────────────────────────────────────────────────────

def compute_max_tanimoto(fps_query, fps_ref):
    """For each query fp, compute max Tanimoto similarity to any ref fp.
    Returns numpy array of max similarities."""
    if len(fps_ref) == 0:
        return np.full(len(fps_query), np.nan)
    max_sims = np.empty(len(fps_query))
    for i, fp in enumerate(fps_query):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_ref)
        max_sims[i] = max(sims) if sims else 0.0
    return max_sims


def plot_similarity_histograms(fps, mol_labels, cluster_ids, final_cluster_labels,
                                output_path, source_labels=None):
    """Plot histograms of max Tanimoto similarity between label groups.

    Parameters
    ----------
    fps : list of fingerprints
    mol_labels : ignored (kept for backward compat)
    cluster_ids : array of cluster IDs per molecule
    final_cluster_labels : dict {cluster_id: 'A'|'B'|'C'}
    output_path : path to save the figure
    source_labels : array-like, optional
        Original source label per molecule ('A' or 'B').
        When provided, enables extra histograms:
          - A → all BDB (B-source + C-from-BDB)
          - A → C-from-BDB (common/buffer molecules originally from BDB)
    """
    # Group molecule indices by final cluster label
    a_indices = [i for i, cid in enumerate(cluster_ids)
                 if final_cluster_labels.get(cid) == 'A']
    b_indices = [i for i, cid in enumerate(cluster_ids)
                 if final_cluster_labels.get(cid) == 'B']
    c_indices = [i for i, cid in enumerate(cluster_ids)
                 if final_cluster_labels.get(cid) == 'C']

    fps_a = [fps[i] for i in a_indices]
    fps_b = [fps[i] for i in b_indices]
    fps_c = [fps[i] for i in c_indices]

    # Source-aware groups (if source_labels provided)
    has_source = source_labels is not None
    if has_source:
        # All molecules originally from BDB (regardless of final cluster label)
        all_bdb_indices = [i for i in range(len(fps)) if source_labels[i] == 'B']
        # C-cluster molecules that originally came from BDB
        c_from_bdb_indices = [i for i in c_indices if source_labels[i] == 'B']
        # C-cluster molecules that originally came from HARVEST
        c_from_harvest_indices = [i for i in c_indices if source_labels[i] == 'A']
        fps_all_bdb = [fps[i] for i in all_bdb_indices]
        fps_c_from_bdb = [fps[i] for i in c_from_bdb_indices]
        fps_c_from_harvest = [fps[i] for i in c_from_harvest_indices]

    print(f"\n  Computing similarity histograms...")
    print(f"    A (HARVEST-unique): {len(fps_a)} molecules")
    print(f"    B (BDB-unique): {len(fps_b)} molecules")
    print(f"    C (common/buffer): {len(fps_c)} molecules")
    if has_source:
        print(f"    All BDB-source: {len(all_bdb_indices)} molecules")
        print(f"    C from BDB: {len(c_from_bdb_indices)} molecules")
        print(f"    C from HARVEST: {len(c_from_harvest_indices)} molecules")

    n_plots = 6 if has_source else 3
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    # 1. A → B (HARVEST-unique → BDB-unique)
    if fps_a and fps_b:
        sims_a_to_b = compute_max_tanimoto(fps_a, fps_b)
        axes[0].hist(sims_a_to_b, bins=51, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(np.median(sims_a_to_b), color='red', linestyle='--',
                        label=f'median={np.median(sims_a_to_b):.3f}')
        axes[0].legend()
    axes[0].set_title('HARVEST-unique (A) → BDB-unique (B)')
    axes[0].set_xlabel('Max Tanimoto Similarity')
    axes[0].set_ylabel('Count')

    # 2. A → C (HARVEST-unique → Common)
    if fps_a and fps_c:
        sims_a_to_c = compute_max_tanimoto(fps_a, fps_c)
        axes[1].hist(sims_a_to_c, bins=51, alpha=0.7, color='orange', edgecolor='black')
        axes[1].axvline(np.median(sims_a_to_c), color='red', linestyle='--',
                        label=f'median={np.median(sims_a_to_c):.3f}')
        axes[1].legend()
    axes[1].set_title('HARVEST-unique (A) → Common (C)')
    axes[1].set_xlabel('Max Tanimoto Similarity')
    axes[1].set_ylabel('Count')

    # 3. B → C (BDB-unique → Common)
    if fps_b and fps_c:
        sims_b_to_c = compute_max_tanimoto(fps_b, fps_c)
        axes[2].hist(sims_b_to_c, bins=51, alpha=0.7, color='green', edgecolor='black')
        axes[2].axvline(np.median(sims_b_to_c), color='red', linestyle='--',
                        label=f'median={np.median(sims_b_to_c):.3f}')
        axes[2].legend()
    axes[2].set_title('BDB-unique (B) → Common (C)')
    axes[2].set_xlabel('Max Tanimoto Similarity')
    axes[2].set_ylabel('Count')

    # 4. A → all BDB (HARVEST-unique → all molecules originally from BDB)
    if has_source:
        if fps_a and fps_all_bdb:
            sims_a_to_all_bdb = compute_max_tanimoto(fps_a, fps_all_bdb)
            axes[3].hist(sims_a_to_all_bdb, bins=51, alpha=0.7, color='purple', edgecolor='black')
            axes[3].axvline(np.median(sims_a_to_all_bdb), color='red', linestyle='--',
                            label=f'median={np.median(sims_a_to_all_bdb):.3f}')
            axes[3].legend()
        axes[3].set_title('HARVEST-unique (A) → All BDB-source')
        axes[3].set_xlabel('Max Tanimoto Similarity')
        axes[3].set_ylabel('Count')

        # 5. A → C-from-BDB (HARVEST-unique → common/buffer molecules from BDB)
        if fps_a and fps_c_from_bdb:
            sims_a_to_c_bdb = compute_max_tanimoto(fps_a, fps_c_from_bdb)
            axes[4].hist(sims_a_to_c_bdb, bins=51, alpha=0.7, color='darkcyan', edgecolor='black')
            axes[4].axvline(np.median(sims_a_to_c_bdb), color='red', linestyle='--',
                            label=f'median={np.median(sims_a_to_c_bdb):.3f}')
            axes[4].legend()
        axes[4].set_title('HARVEST-unique (A) → C from BDB')
        axes[4].set_xlabel('Max Tanimoto Similarity')
        axes[4].set_ylabel('Count')

        # 6. C-from-HARVEST → all BDB (common/buffer HARVEST molecules → all BDB-source)
        if fps_c_from_harvest and fps_all_bdb:
            sims_c_harvest_to_bdb = compute_max_tanimoto(fps_c_from_harvest, fps_all_bdb)
            axes[5].hist(sims_c_harvest_to_bdb, bins=51, alpha=0.7, color='coral', edgecolor='black')
            axes[5].axvline(np.median(sims_c_harvest_to_bdb), color='red', linestyle='--',
                            label=f'median={np.median(sims_c_harvest_to_bdb):.3f}')
            axes[5].legend()
        axes[5].set_title('C from HARVEST → All BDB-source')
        axes[5].set_xlabel('Max Tanimoto Similarity')
        axes[5].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Relabel functions (merged from relabel_clusters.py)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical complete linkage clustering
# ─────────────────────────────────────────────────────────────────────────────

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


def cluster_molecules(smiles_or_fps, method='complete_linkage', threshold=0.2,
                      dbscan_eps=0.2, n_jobs=1):
    """Cluster molecules and return a list of clusters (list of index lists).

    This is the standalone Step 2 extracted from :func:`split_clusters` so it
    can be reused without the full A/B/C relabeling pipeline.

    Parameters
    ----------
    smiles_or_fps : list
        Either SMILES strings or pre-computed RDKit fingerprint objects.
        Strings are fingerprinted via :func:`smiles_to_fps`; invalid SMILES
        are silently dropped (indices in the returned clusters refer to the
        valid subset).
    method : str
        'complete_linkage' (default), 'butina', 'clique', or 'dbscan'.
    threshold : float
        Tanimoto distance threshold for clustering (0.2 = similarity >= 0.8).
    dbscan_eps : float
        DBSCAN eps (only used when method='dbscan').
    n_jobs : int
        Threads for fingerprinting (only used when *smiles_or_fps* contains
        SMILES strings).

    Returns
    -------
    list[list[int]]
        Each inner list holds the indices into the valid fingerprints that
        belong to one cluster, sorted largest-cluster-first.
    """
    if smiles_or_fps and isinstance(smiles_or_fps[0], str):
        fps = [fp for fp in smiles_to_fps(list(smiles_or_fps), n_jobs=n_jobs)
               if fp is not None]
    else:
        fps = list(smiles_or_fps)
    if not fps:
        return []
    dist_matrix = None
    if method in ('clique', 'dbscan', 'complete_linkage'):
        dist_matrix = compute_sparse_tanimoto_matrix_speed(fps, threshold=threshold)

    if method == 'butina':
        butina_clusters = cluster_fps_butina(fps, cutoff=threshold)
        clusters = [list(c) for c in butina_clusters]
    elif method == 'clique':
        G_mol = nx.from_scipy_sparse_array(dist_matrix)
        clusters = greedy_clique_cover(G_mol)
    elif method == 'dbscan':
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
    elif method == 'complete_linkage':
        clusters = hierarchical_complete_linkage(dist_matrix, threshold=threshold)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    clusters.sort(key=len, reverse=True)
    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def split_clusters(df, cluster_threshold=0.25, max_jumps=2,
                   relabel_from='both', method='auto', fps_type='morgan',
                   clustering_method='complete_linkage', dbscan_eps=0.2,
                   smiles_key='SMILES', label_key='label', label_map=None,
                   ligand_threshold=None, n_jobs=8):
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
    print(f"\n[1/6] Loading {len(df)} SMILES ({fps_type} fingerprints)...")
    print(f"  HARVEST: {(df['source_label']=='A').sum()}, BDB: {(df['source_label']=='B').sum()}")

    fps_all = smiles_to_fps(df[smiles_key].tolist(), finger_type=fps_type, n_jobs=n_jobs)
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
    clusters = cluster_molecules(fps, method=clustering_method,
                                 threshold=ligand_threshold,
                                 dbscan_eps=dbscan_eps)

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

    centroids = compute_cluster_centroids(clusters, fps)
    print(f"  Computed {len(centroids)} centroid fingerprints")

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
        'G': G
    }


def save_split_results(df_valid, output_path):
    """Save split results DataFrame to CSV.
    Dynamically selects SMILES/label columns from the DataFrame.
    """
    # Find the smiles and label columns (they vary by caller)
    base_cols = ['cluster_id', 'final_label',
                 'nearest_smiles', 'tanimoto_sim']
    # Include all original columns plus the computed ones
    extra_cols = [c for c in df_valid.columns if c not in base_cols and c != 'source_label']
    out_cols = extra_cols + base_cols
    out_cols = [c for c in out_cols if c in df_valid.columns]
    df_valid[out_cols].to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({len(df_valid)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cluster-based A/B/C split of two SMILES subsets')
    parser.add_argument('--input', '-i', default='cluster_split/data.csv',
                        help='Input CSV with SMILES and label columns')
    parser.add_argument('--cluster-threshold', '-t', type=float, default=0.225,
                        help='Tanimoto distance cutoff for clustering and graph (default: 0.225)')
    parser.add_argument('--max-jumps', type=int, default=2,
                        help='Min hop distance between A and B in cluster graph (default: 2)')
    parser.add_argument('--relabel-from', choices=['A', 'B', 'both'], default='A',
                        help='Which labels can become C (default: A)')
    parser.add_argument('--method', choices=['ilp', 'greedy', 'auto'], default='auto',
                        help='Relabeling method (default: auto)')
    parser.add_argument('--fps-type', default='morgan',
                        help='Fingerprint type for smiles_to_fps (default: morgan)')
    parser.add_argument('--clustering-method', choices=['clique', 'butina', 'dbscan', 'complete_linkage'],
                        default='complete_linkage',
                        help='Clustering method (default: complete_linkage)')
    parser.add_argument('--ligand-threshold', type=float, default=0.2,
                        help='Tanimoto distance cutoff for molecule clustering (default: 0.2)')
    parser.add_argument('--dbscan-eps', type=float, default=0.2,
                        help='DBSCAN eps parameter (default: 0.2)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: <input_dir>/split_results.csv)')

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # df = df.sample(2000, random_state=98888)
    result = split_clusters(
        df=df,
        cluster_threshold=args.cluster_threshold,
        max_jumps=args.max_jumps,
        relabel_from=args.relabel_from,
        method=args.method,
        fps_type=args.fps_type,
        clustering_method=args.clustering_method,
        dbscan_eps=args.dbscan_eps,
        ligand_threshold=args.ligand_threshold,
    )

    output_dir = Path(args.input).parent
    output_path = args.output or str(output_dir / 'split_results.csv')
    save_split_results(result['df_valid'], output_path)

    hist_path = output_dir / 'similarity_histograms.png'
    plot_similarity_histograms(
        result['fps'], None, result['mol_to_cluster'],
        result['final_cluster_labels'], hist_path,
        source_labels=result['df_valid']['source_label'].values)