# Cluster-Based Dataset Splitting

## Full Methods

*(See below for the condensed version)*

### Overview

To rigorously evaluate the novelty of bioactivity data extracted from USPTO patents relative to existing public databases (BindingDB), we developed a cluster-based splitting procedure that partitions compounds into three non-overlapping sets per protein target: **Set A** (patent-unique), **Set B** (database-unique), and **Set C** (common/buffer zone). The procedure ensures that no compound in Set A shares high structural similarity with any compound in Set B, thereby providing an unbiased assessment of chemical novelty contributed by patent mining. The algorithm operates independently for each protein target and consists of six stages: (1) molecular fingerprinting, (2) hierarchical clustering, (3) cluster source assignment, (4) cluster graph construction, (5) minimum buffer zone optimization, and (6) per-molecule label assignment.

### Molecular fingerprinting

For each protein target, SMILES strings from both the patent-harvested dataset ("HARVEST") and the reference database ("BDB") were pooled. When a compound appeared in both sources, only the HARVEST instance was retained. Morgan circular fingerprints (radius 2, 2048 bits) were computed for all valid structures using RDKit (v2023.09). Compounds that failed sanitization or fingerprint generation were excluded from further analysis.

### Pairwise similarity computation

Pairwise Tanimoto similarities were computed between all compounds within each protein target set. To maintain computational tractability, we employed a sparse representation: only pairs with Tanimoto similarity $\geq 1 - d_L$ were stored, where $d_L$ is the ligand distance threshold (default $d_L = 0.2$, corresponding to similarity $\geq 0.8$). The resulting sparse distance matrix stores Tanimoto distances ($d = 1 - \text{sim}$) for qualifying pairs, with all other pairs implicitly treated as maximally distant ($d = 1.0$).

### Hierarchical complete-linkage clustering

Compounds were grouped into structurally homogeneous clusters using agglomerative hierarchical clustering with complete linkage. Under the complete-linkage criterion, the distance between two clusters is defined as the maximum pairwise distance between any member of one cluster and any member of the other:

$$d(C_i, C_j) = \max_{a \in C_i, \, b \in C_j} d_T(a, b)$$

where $d_T(a, b) = 1 - \text{Tanimoto}(a, b)$. The dendrogram was cut at distance $d_L$, ensuring that all pairs of compounds within any single cluster share Tanimoto similarity $\geq 1 - d_L$. This was implemented using `scipy.cluster.hierarchy.linkage` (method `'complete'`) and `fcluster` (criterion `'distance'`, threshold $d_L$). Missing entries in the condensed distance matrix (pairs with similarity below $1 - d_L$) were set to a default distance of 1.0, which is equivalent to treating structurally dissimilar compounds as maximally distant and ensuring they are never merged into the same cluster.

### Initial cluster labeling

Each cluster was assigned a source label based on the composition of its member compounds:

- **Label A**: all members originate from HARVEST (patent-derived)
- **Label B**: all members originate from BDB (reference database)
- **Mixed clusters**: clusters containing compounds from both sources were provisionally assigned to Set B (since their structural scaffolds are already represented in the public database) and flagged as frozen nodes that cannot be reassigned during buffer zone optimization

This conservative assignment ensures that any shared chemical space is attributed to the reference database rather than claimed as novel patent content.

### Cluster connectivity graph

To model inter-cluster structural relationships, a weighted graph was constructed over cluster centroids. The centroid of each cluster was computed by bitwise majority voting: each bit position in the Morgan fingerprint was set to 1 if at least half of the cluster members had that bit activated, and 0 otherwise. For singleton clusters, the member's fingerprint served directly as the centroid.

Pairwise Tanimoto distances between centroids were computed using a wide threshold $d_W = \min(k \cdot d_C, \, 0.95)$, where $d_C$ is the cluster distance threshold (default $d_C = 0.225$) and $k$ is the maximum number of allowed hops (default $k = 2$). This wide threshold captures all centroid pairs potentially relevant for multi-hop path analysis.

Edges were added to the graph with integer-valued lengths encoding the number of "structural hops" between clusters:

$$\ell(i, j) = \left\lceil \frac{d_T(\text{centroid}_i, \, \text{centroid}_j)}{d_C} \right\rceil$$

Only edges with $\ell \leq k$ were retained. An edge of length 1 indicates that the cluster centroids are within one clustering radius of each other; length 2 indicates an intermediate structural gap equivalent to one additional hop. The weighted shortest-path distance between any two clusters in this graph thus represents the minimum number of structural hops required to traverse between them.

### Minimum buffer zone optimization

The key algorithmic contribution is the identification of a minimum-cardinality buffer zone (Set C) that separates Sets A and B by at least $k$ hops in the cluster graph. This was formulated as an integer linear program (ILP):

$$\min \sum_{n \in \mathcal{V}} x_n$$

subject to:

$$x_a + x_b \geq 1 \quad \forall \, (a, b) : \text{label}(a) = A, \, \text{label}(b) = B, \, d_G(a, b) \leq k$$

$$x_n = 0 \quad \forall \, n \in \mathcal{F}$$

$$x_n \in \{0, 1\} \quad \forall \, n \in \mathcal{V}$$

where $x_n = 1$ indicates that cluster $n$ is reassigned to Set C, $d_G(a, b)$ denotes the weighted shortest-path distance between clusters $a$ and $b$ in the graph (computed via Dijkstra's algorithm with the edge length attribute as weight), and $\mathcal{F}$ is the set of frozen nodes (mixed-source clusters that are fixed as C). The `relabel_from` parameter restricts which labels may transition to C; by default, only A-labeled clusters are candidates for reassignment (`relabel_from = 'A'`), preserving the reference database partition intact.

Conflicting A-B pairs within $k$ hops were identified using single-source Dijkstra shortest-path computations with a cutoff of $k$, run from each A-labeled node. The ILP was solved using `scipy.optimize.milp` with binary integrality constraints. For graphs exceeding $10^6$ nodes, a greedy vertex cover heuristic was used instead, iteratively reassigning the most-conflicted node to C until no A-B pair remained within $k$ hops. A post-optimization validation step verified that all conflicts were resolved; any residual violations were eliminated by the greedy heuristic as a fallback.

### Per-molecule label assignment

Final labels were propagated from clusters to individual molecules: each compound inherited the label of its parent cluster. Additionally, for each compound, the most structurally similar compound from the opposing source was identified by (i) locating the nearest cluster (by centroid Tanimoto similarity) containing molecules from the other source, and (ii) computing pairwise Tanimoto similarities to all opposing-source molecules within that cluster. The maximum similarity and corresponding SMILES string were recorded, providing a per-molecule measure of the structural gap between the patent-derived and reference datasets.

### Default parameters

Unless otherwise specified, the following parameters were used: ligand clustering threshold $d_L = 0.2$ (Tanimoto distance), cluster graph threshold $d_C = 0.225$, maximum hop distance $k = 2$, Morgan fingerprints (radius 2, 2048 bits), complete-linkage clustering, and relabeling restricted to A-labeled clusters only. These settings ensure that compounds within the same cluster share at least 80% Tanimoto similarity, while the buffer zone spans a structural gap of at least two clustering radii (effective Tanimoto distance $\geq 0.45$) between patent-unique and database-represented chemical series.

### Per-protein processing

The splitting procedure was applied independently to each protein target appearing in the patent-harvested dataset. Proteins were categorized as:

- **New proteins** (present only in the patent dataset): all compounds were clustered and assigned to Set A. No buffer zone optimization was required, as no reference compounds existed for comparison.
- **Overlap proteins** (present in both datasets): the full six-stage pipeline was executed. Per-protein CSV output files contained only HARVEST-source molecules with their assigned split labels (A or C), enabling direct assessment of which patent-derived compounds represent genuinely novel chemical matter versus structural analogs of known database entries.

Summary statistics were computed for each protein, including the number of clusters, cluster size distributions, scaffold diversity ratios, and the distribution of final A/B/C labels across clusters.

---

## Condensed Version (for main text)

To assess the structural novelty of patent-derived compounds relative to existing public bioactivity databases, we developed a per-protein cluster-based splitting procedure. For each protein target, compounds from both the patent-harvested and reference datasets were pooled, represented as Morgan circular fingerprints, and grouped into structurally homogeneous clusters using agglomerative hierarchical clustering with complete linkage (Tanimoto distance threshold 0.2, ensuring $\geq$ 80% within-cluster similarity). Each cluster was then labeled according to its source composition -- patent-only, database-only, or mixed -- and a weighted connectivity graph was built over cluster centroids, where edge weights encode multi-hop structural distances between clusters.

To guarantee that compounds classified as patent-unique are structurally well-separated from known database compounds, we introduced a buffer zone of intermediate clusters between the two sets. The minimal buffer zone was determined by solving an integer linear program that minimizes the number of clusters reassigned to the buffer while ensuring no patent-unique cluster lies within two structural hops of any database cluster in the connectivity graph. Clusters containing compounds from both sources were conservatively attributed to the reference database. This procedure partitions the chemical space into three non-overlapping regions per protein target: structurally novel patent matter, known database matter, and a buffer zone of structurally intermediate compounds, enabling unbiased quantification of the novel chemical space contributed by patent mining.
