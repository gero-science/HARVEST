#!/usr/bin/env python3
"""Plot HARVEST vs BindingDB diversity figure (2 rows x 3 columns):

Row 1: three pie charts of the cluster-split statistics
    (a) Protein composition, (b) PLI diversity, (c) Scaffold diversity.
Row 2: (d) nearest-neighbour Tanimoto similarity histograms for the three
    cluster-split subsets (Novel = final_label A, Buffer = C, Border = CB)
    aggregated from `split_results_<UNIPROT>.csv` files,
    (e) HARVEST + BindingDB patents binned by PLIs-per-patent,
    (f) hit-to-lead activity-cliff pair count per source. Assays first
    pass the Boltz-2 curation (>=10 samples, std_y >= 0.25, unique-ratio
    >= 0.20, mean pairwise Tanimoto >= 0.25); then within each passing
    assay unique molecules are sorted by potency, paired max-vs-min with
    no ligand reused, and a pair is counted only if |Δlog10 activity|
    >= _CLIFF_LOG_DIFF_THRESHOLD (default 2 = 100-fold).

Also emits an L2-classification summary table alongside the PNG.
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Running this file as a script puts `manuscript/` on sys.path, not the repo
# root, so `cluster_split` would not be importable (same fix as in
# manuscript/cross_validation.py).
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Patent-office prefixes considered "patent-derived" in BindingDB
# (mirrors PATENT_PREFIX_RE in plot_harvest_stats.py).
PATENT_PREFIX_RE = r"^(US|WO|EP)"

# Panel (d): user-facing labels for the three final_label codes used in the
# per-protein `split_results_<UNIPROT>.csv` files.
_TANIMOTO_LABEL_MAP = {
    "A":  "Novel",
    "C":  "Buffer",
    "CB": "Boundary",
}

# Bin edges for the PLIs-per-patent bar chart (panel e).
_PLI_BIN_EDGES = [0, 10, 100, 500, np.inf]
_PLI_BIN_LABELS = ["1-10", "10-100", "100-500", ">500"]

# Panel (f) activity-cliffs (Boltz-2 hit-to-lead curation, Appendix C.2.5).
# Per-assay filter: >=10 samples, std(y)>=0.25, >=10 unique values,
# unique/total>=0.20, mean pairwise Morgan Tanimoto>=0.25.
# Score = IQR of y=-log10(value in uM); assays with high IQR are the ones
# Boltz-2 up-weights during hit-to-lead training.
_CLIFF_LOG_REF_NM      = 1000.0     # 1 uM = 1000 nM
_CLIFF_GT_KEEP_NM      = 10_000.0   # keep '>'/'>=' only if value >= 10 uM
_CLIFF_UNIT_ERROR_NM   = 1e-3       # drop assay if any value < 1e-6 uM
_CLIFF_MIN_POINTS      = 10
_CLIFF_MIN_STD_Y       = 0.25
_CLIFF_MIN_UNIQUE      = 10
_CLIFF_MIN_UNIQUE_RATIO = 0.20
_CLIFF_MIN_MEAN_TANIMOTO = 0.25
_CLIFF_FP_RADIUS       = 2
_CLIFF_FP_NBITS        = 2048
_CLIFF_TANIMOTO_CAP    = 300        # max mols/assay used in Tanimoto estimate
_CLIFF_RANDOM_SEED     = 42
_CLIFF_XMAX            = 4.5
_CLIFF_XTHR            = 2.2        # Boltz-2 "high" cliff-score threshold
# Minimum |Δy| (log10 activity units) between the high- and low-activity
# ligand of a hit-to-lead pair to be counted as an activity cliff. Default
# 2.0 == a 100-fold potency swing. Overridable via --cliff-log-diff (main()
# rebinds this constant so the signature/cache reflect the chosen value).
_CLIFF_LOG_DIFF_THRESHOLD = 2.0
_HARVEST_ASSAY_KEY     = ("patent_number", "assay_id", "activity_type")
# Post-refactor: the BindingDB slice now uses `BindingDB Entry DOI` (populated
# for 99.8% of source='bindingdb' rows, ~57 rows per DOI on average) as the
# real assay-level identifier. `validity_comment` picks which sub-slice.
_BDB_ASSAY_KEY         = ("uniprot_id", "activity_type", "doi")
_BDB_PATENT_VC         = ("US Patent", "WIPO")
_BDB_CHEMPUB_VC        = ("ChEMBL", "PubChem")


def load_data(data_dir: Path, bdb_path: Path):
    """Load stats CSVs and compute protein counts."""
    df_new = pd.read_csv(data_dir / "df_new_protein_stats.csv")
    df_overlap = pd.read_csv(data_dir / "df_overlap_stats.csv")

    # Protein counts
    n_new_proteins = len(df_new)
    n_overlap_proteins = len(df_overlap)

    df_bd = pd.read_parquet(bdb_path, columns=["uniprot_acc"])
    all_bdb_proteins = set(df_bd["uniprot_acc"].dropna().unique())
    overlap_protein_names = set(df_overlap["protein"].dropna().unique())
    n_bdb_only_proteins = len(all_bdb_proteins - overlap_protein_names)

    return df_new, df_overlap, n_new_proteins, n_overlap_proteins, n_bdb_only_proteins


def _first_accession(value):
    """Take the first semicolon-separated UniProt accession."""
    return str(value).split(';')[0].strip()


_CLUSTER_MAX_MOLS = 5000  # O(n²) distance matrix; cap to keep runtime sane


def _count_clusters(smiles_arr, threshold: float = 0.2) -> int:
    """Count clusters for an array of unique SMILES.

    Uses the same FCFP fingerprints and complete-linkage clustering as
    ``annotate_novelty.py`` (via ``cluster_split.split_clusters``).
    For proteins exceeding ``_CLUSTER_MAX_MOLS`` falls back to generic
    Murcko scaffold count to avoid the O(n²) distance matrix.
    """
    n = len(smiles_arr)
    if n <= 1:
        return n
    # Deliberately unguarded: a missing import used to fall back to `return n`,
    # which silently reports one cluster per compound — a plausible-looking
    # number that is wrong by roughly a factor of two.
    from cluster_split.split_clusters import cluster_molecules

    if n > _CLUSTER_MAX_MOLS:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffolds = set()
        for s in smiles_arr:
            mol = Chem.MolFromSmiles(s)
            if mol:
                scaf = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol))
                scaffolds.add(Chem.MolToSmiles(scaf))
        return len(scaffolds) or 1

    return len(cluster_molecules(list(smiles_arr),
                                 method='complete_linkage',
                                 threshold=threshold))


def load_data_from_parquet(harvest_path: Path, bdb_path: Path):
    """Derive split statistics from the annotated parquet's novelty columns.

    This replaces the CSV-based ``load_data`` when the HARVEST parquet already
    carries ``novelty_label`` and ``cluster_id`` columns produced by
    ``annotate_novelty.py``.
    """
    import pyarrow.parquet as pq
    cols = set(pq.ParquetFile(harvest_path).schema_arrow.names)
    smiles_col = next(c for c in ('clean_smiles', 'Ligand SMILES', 'smiles')
                      if c in cols)
    uniprot_col = next(c for c in ('uniprot_acc', 'Target accession')
                       if c in cols)

    df = pd.read_parquet(harvest_path,
                         columns=[uniprot_col, smiles_col,
                                  'novelty_label', 'cluster_id'])
    df['_acc'] = df[uniprot_col].map(_first_accession)
    # Drop rows without a valid UniProt accession
    df = df[df['_acc'].str.match(r'^[A-Z][A-Z0-9]{4,9}$')]

    # BDB proteins
    df_bd = pd.read_parquet(bdb_path, columns=["uniprot_acc"])
    bdb_proteins = set(df_bd["uniprot_acc"].dropna().unique())
    harvest_proteins = set(df['_acc'].unique())

    # ── Classify proteins by BDB membership, not annotation status ──
    # After the cluster-id backfill all proteins carry novelty_label, so
    # annotation presence can no longer distinguish new from overlap.
    overlap_proteins = harvest_proteins & bdb_proteins
    new_proteins = harvest_proteins - bdb_proteins

    # Overlap proteins: those shared with BDB (have Novel/Buffer/Boundary)
    annotated = df[df['_acc'].isin(overlap_proteins)].dropna(
        subset=['novelty_label']).copy()

    overlap_rows = []
    for protein, grp in annotated.groupby('_acc'):
        novel = grp[grp['novelty_label'] == 'Novel']
        buffer = grp[grp['novelty_label'] == 'Buffer']
        non_novel = grp[grp['novelty_label'].isin(['Buffer', 'Boundary'])]
        overlap_rows.append({
            'protein': protein,
            'n_harvest_only_compounds': novel[smiles_col].nunique(),
            'n_harvest_overlap_compounds': non_novel[smiles_col].nunique(),
            'n_harvest_only_clusters': novel['cluster_id'].nunique(),
            'n_shared_clusters': non_novel['cluster_id'].nunique(),
            'n_buffer_compounds': buffer[smiles_col].nunique(),
            'n_buffer_clusters': buffer['cluster_id'].nunique(),
        })
    df_overlap = pd.DataFrame(overlap_rows) if overlap_rows else pd.DataFrame(
        columns=['protein', 'n_harvest_only_compounds',
                 'n_harvest_overlap_compounds',
                 'n_harvest_only_clusters', 'n_shared_clusters',
                 'n_buffer_compounds', 'n_buffer_clusters'])

    # ── New proteins: in HARVEST but not in BDB ──
    new_rows = []
    for protein in sorted(new_proteins):
        grp = df[df['_acc'] == protein]
        smiles_unique = grp[smiles_col].dropna().unique()
        n_compounds = len(smiles_unique)
        n_with_cid = grp['cluster_id'].notna().sum()
        if n_with_cid > 0:
            n_clusters = grp['cluster_id'].nunique()
        else:
            n_clusters = _count_clusters(smiles_unique)
        new_rows.append({
            'protein': protein,
            'n_compounds': n_compounds,
            'n_clusters': n_clusters,
        })
    df_new = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(
        columns=['protein', 'n_compounds', 'n_clusters'])

    n_new_proteins = len(df_new)
    n_overlap_proteins = len(df_overlap)
    n_bdb_only_proteins = len(bdb_proteins - overlap_proteins)

    return df_new, df_overlap, n_new_proteins, n_overlap_proteins, n_bdb_only_proteins


_PANEL_LABEL_X = -0.25
_PANEL_LABEL_Y = 1.075


def _add_panel_label(ax, panel_label, title_fontsize=11):
    """Place a panel letter (a)/(b)/... at the same axes-relative position
    across all subplots so the letters line up in the composite figure.
    """
    ax.text(_PANEL_LABEL_X, _PANEL_LABEL_Y, panel_label,
            transform=ax.transAxes, fontsize=title_fontsize,
            fontweight="bold", va="top")


def _make_legend_pie(ax, sizes, labels, colors, title, panel_label,
                     radius=1.0, fontsize=9, title_fontsize=11,
                     hatches=None):
    """Draw a pie chart with a legend instead of inline labels to avoid overlap.

    Optional `hatches` is a per-slice list of matplotlib hatch strings (or
    None/empty for no hatch) — used in (b)/(c) to distinguish two slices
    that share the same colour.
    """
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90, radius=radius,
        wedgeprops=dict(edgecolor="white", linewidth=0.5),
    )
    # ax.pie() sets aspect='equal' with adjustable='box' → the axes bbox
    # shrinks to a square, which knocks the panel letter out of alignment
    # with row-2 (bar/hist) axes in the same GridSpec column. Keep the
    # bbox the full GridSpec cell and let the pie scale within data
    # limits instead.
    ax.set_aspect("equal", adjustable="datalim")
    if hatches is not None:
        for wedge, h in zip(wedges, hatches):
            if h:
                wedge.set_hatch(h)
                # keep the hatch stroke black regardless of face colour
                wedge.set_edgecolor("black")
                wedge.set_linewidth(0.3)
    # Add percentage text inside wedges
    total = sum(sizes)
    for i, (wedge, size) in enumerate(zip(wedges, sizes)):
        pct = size / total * 100
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = 0.55 * radius * np.cos(np.radians(ang))
        y = 0.55 * radius * np.sin(np.radians(ang))
        ax.text(x, y, f"{pct:.1f}%", ha="center", va="center",
                fontsize=fontsize - 1, fontweight="bold", color="white")

    legend_labels = [f"{lab} ({size:,})" for lab, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.05), fontsize=fontsize + 1,
              frameon=False, ncol=1, handlelength=1.0, handletextpad=0.4)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
    _add_panel_label(ax, panel_label, title_fontsize=title_fontsize)


def plot_pie_proteins(ax, n_new, n_bdb_only, n_overlap, **kw):
    colors = matplotlib.colormaps["Set2"].colors[:3]
    sizes = [n_new, n_bdb_only, n_overlap]
    labels = ["HARVEST-exclusive", "BindingDB-exclusive", "BindingDB & HARVEST shared"]
    _make_legend_pie(ax, sizes, labels, colors,
                     "Proteins composition", "(a)", **kw)


def plot_compounds_diver(ax, split_data: pd.DataFrame,
                          df_new: pd.DataFrame, **kw):
    """(b) PLI pie — four slices.

    * Novel-protein PLIs: every HARVEST compound on a HARVEST-only target
      (df_new). By construction they have no BindingDB counterpart, so
      the cluster label is trivially "A"; we split them out to make
      "target-novel" vs "chemistry-novel" contributions visible.
    * Novel PLIs (on shared targets): "A"-labelled compounds where the
      target itself is also in BindingDB — the cluster contains no BDB
      ligand for that target.
    * Buffer PLIs: "C"-labelled compounds — cluster partially bridges
      a BDB cluster.
    * Boundary PLIs: "CB"-labelled compounds — cluster essentially
      co-located with a BDB cluster.
    """
    palette = matplotlib.colormaps["Set2"].colors
    n_novel_prot = int(df_new["n_compounds"].sum())
    shared = split_data[~split_data["protein"].isin(df_new["protein"])]
    counts = shared["final_label"].value_counts()
    n_novel_pli = int(counts.get("A", 0))
    n_buffer    = int(counts.get("C", 0))
    n_border   = int(counts.get("CB", 0))
    sizes  = [n_novel_prot, n_novel_pli, n_buffer, n_border]
    labels = ["Novel-protein PLIs", "Novel PLIs (shared prot.)",
              "Buffer PLIs", "Boundary PLIs"]
    # Novel-protein slice shares the "Novel" green with A-shared but is
    # distinguished by a hatch pattern.
    colors  = [palette[0], palette[0], palette[1], palette[2]]
    hatches = ["///", "",    "",         ""]
    _make_legend_pie(ax, sizes, labels, colors,
                     "PLI Diversity", "(b)", hatches=hatches, **kw)


def plot_clusters_diver(ax, split_data: pd.DataFrame,
                         df_new: pd.DataFrame, **kw):
    """(c) Scaffold pie — four slices, mirroring (b).

    * Novel-protein scaffolds: HARVEST clusters on HARVEST-only targets
      (from df_new); no BDB comparison possible.
    * Novel scaffolds (shared prot.): label "A" — no BDB ligand in the
      cluster.
    * Buffer scaffolds: label "C" — partial bridge to a BDB cluster.
    * Boundary scaffolds: label "CB" — essentially co-located with a BDB
      cluster.
    """
    palette = matplotlib.colormaps["Set2"].colors
    n_novel_prot = int(df_new["n_clusters"].sum())
    shared = split_data[~split_data["protein"].isin(df_new["protein"])]
    per_cluster = (shared.drop_duplicates(["protein", "cluster_id"])
                   ["final_label"])
    n_novel   = int((per_cluster == "A").sum())
    n_buffer  = int((per_cluster == "C").sum())
    n_border = int((per_cluster == "CB").sum())
    sizes  = [n_novel_prot, n_novel, n_buffer, n_border]
    labels = ["Novel-protein scaffolds", "Novel scaffolds (shared prot.)",
              "Buffer scaffolds", "Boundary scaffolds"]
    colors  = [palette[0], palette[0], palette[1], palette[2]]
    hatches = ["///", "",    "",         ""]
    _make_legend_pie(ax, sizes, labels, colors,
                     "Scaffold Diversity", "(c)", hatches=hatches, **kw)


def build_l2_summary(df_new: pd.DataFrame, df_overlap: pd.DataFrame,
                     chembl_path: str,
                     min_proteins: int = 15,
                     min_l1_proteins: int = 50) -> pd.DataFrame:
    """Build summary of new compounds and clusters by ChEMBL L1/L2 classification.

    Includes both HARVEST-only proteins (all their compounds/clusters are new)
    and overlap proteins (Novel + Buffer compounds/clusters on shared targets).
    L1 classes with fewer than `min_l1_proteins` are collapsed into 'Other'.
    L2 classes with fewer than `min_proteins` are collapsed into 'Other' within
    their L1 group.

    Returns a flat DataFrame with columns:
        Target Class, Proteins, New Compounds, New Clusters
    where L1 names appear as group header rows (numbers are L1 subtotals)
    and L2 names are indented data rows underneath.
    """
    chembl = pd.read_csv(chembl_path)
    chembl_map = chembl.drop_duplicates("accession")[["accession", "l1", "l2"]].copy()
    chembl_map.rename(columns={"accession": "protein"}, inplace=True)

    # Prepare df_new: all compounds/clusters are novel
    new = df_new[["protein", "n_compounds", "n_clusters"]].copy()
    new.rename(columns={"n_compounds": "novel_compounds",
                         "n_clusters": "novel_clusters"}, inplace=True)

    # Prepare df_overlap: Novel + Buffer compounds/clusters
    if "n_buffer_compounds" in df_overlap.columns:
        # From annotated parquet: sum Novel + Buffer
        overlap = df_overlap[["protein"]].copy()
        overlap["novel_compounds"] = (
            df_overlap["n_harvest_only_compounds"]
            + df_overlap["n_buffer_compounds"])
        overlap["novel_clusters"] = (
            df_overlap["n_harvest_only_clusters"]
            + df_overlap["n_buffer_clusters"])
    else:
        # Legacy CSV path: only harvest-only (Novel) available
        overlap = df_overlap[["protein", "n_harvest_only_compounds",
                              "n_harvest_only_clusters"]].copy()
        overlap.rename(columns={"n_harvest_only_compounds": "novel_compounds",
                                 "n_harvest_only_clusters": "novel_clusters"},
                       inplace=True)

    combined = pd.concat([new, overlap], ignore_index=True)
    combined = combined.merge(chembl_map, on="protein", how="left")
    combined["l1"] = combined["l1"].fillna("Unclassified")
    combined["l2"] = combined["l2"].fillna("Unspecified")

    # Aggregate by l1+l2
    agg = combined.groupby(["l1", "l2"]).agg(
        n_proteins=("protein", "count"),
        novel_compounds=("novel_compounds", "sum"),
        novel_clusters=("novel_clusters", "sum"),
    ).reset_index()

    for col in ["n_proteins", "novel_compounds", "novel_clusters"]:
        agg[col] = agg[col].astype(int)

    # L1 subtotals, sorted by novel_compounds desc
    l1_totals = agg.groupby("l1").agg(
        n_proteins=("n_proteins", "sum"),
        novel_compounds=("novel_compounds", "sum"),
        novel_clusters=("novel_clusters", "sum"),
    ).reset_index().sort_values("novel_compounds", ascending=False)

    # Collapse small L1 categories into "Other"
    small_l1 = l1_totals[l1_totals["n_proteins"] < min_l1_proteins]
    big_l1 = l1_totals[l1_totals["n_proteins"] >= min_l1_proteins]
    collapsed_l1_names = set(small_l1["l1"])
    if not small_l1.empty:
        # Merge their L2 rows into a single "Other" L1 group
        other_l1 = pd.DataFrame([{
            "l1": "Other",
            "n_proteins": small_l1["n_proteins"].sum(),
            "novel_compounds": small_l1["novel_compounds"].sum(),
            "novel_clusters": small_l1["novel_clusters"].sum(),
        }])
        l1_totals = pd.concat([big_l1, other_l1], ignore_index=True).sort_values(
            "novel_compounds", ascending=False)
        # Remap small L1 rows in agg
        agg.loc[agg["l1"].isin(collapsed_l1_names), "l1"] = "Other"
        agg.loc[agg["l1"] == "Other", "l2"] = "Other"
        # Re-aggregate after remapping
        agg = agg.groupby(["l1", "l2"]).agg(
            n_proteins=("n_proteins", "sum"),
            novel_compounds=("novel_compounds", "sum"),
            novel_clusters=("novel_clusters", "sum"),
        ).reset_index()

    # Build flat rows: L1 header (no numbers), then L2 data rows
    rows = []
    for _, l1_row in l1_totals.iterrows():
        l1_name = l1_row["l1"]
        children = agg[agg["l1"] == l1_name].sort_values(
            "novel_compounds", ascending=False).copy()

        # Collapse small L2 classes within this L1
        small = children[children["n_proteins"] < min_proteins]
        big = children[children["n_proteins"] >= min_proteins]
        if not small.empty:
            other = pd.DataFrame([{
                "l2": "Other",
                "n_proteins": small["n_proteins"].sum(),
                "novel_compounds": small["novel_compounds"].sum(),
                "novel_clusters": small["novel_clusters"].sum(),
            }])
            children = pd.concat([big, other], ignore_index=True).sort_values(
                "novel_compounds", ascending=False)

        # If only one L2 child, show L1 name as a flat data row (no header)
        if len(children) == 1:
            c = children.iloc[0]
            rows.append({
                "Target Class": l1_name,
                "Proteins": int(c["n_proteins"]),
                "New Compounds": int(c["novel_compounds"]),
                "New Clusters": int(c["novel_clusters"]),
                "_is_header": False,
            })
        else:
            # L1 header row — no numbers, just the group name
            rows.append({
                "Target Class": l1_name,
                "Proteins": None,
                "New Compounds": None,
                "New Clusters": None,
                "_is_header": True,
            })
            for _, c in children.iterrows():
                rows.append({
                    "Target Class": f"  {c['l2']}",
                    "Proteins": int(c["n_proteins"]),
                    "New Compounds": int(c["novel_compounds"]),
                    "New Clusters": int(c["novel_clusters"]),
                    "_is_header": False,
                })

    # Grand total
    rows.append({
        "Target Class": "Total",
        "Proteins": int(agg["n_proteins"].sum()),
        "New Compounds": int(agg["novel_compounds"].sum()),
        "New Clusters": int(agg["novel_clusters"].sum()),
        "_is_header": True,  # special: total row
    })

    return pd.DataFrame(rows)


def save_csv_table(df: pd.DataFrame, path: Path):
    """Save summary as CSV (without internal _is_header flag).
    Header rows get empty numeric cells."""
    out = df.drop(columns=["_is_header"]).copy()
    out = out.fillna("")
    path.write_text(out.to_csv(index=False))


def save_latex_table(df: pd.DataFrame, path: Path):
    """Save summary as a LaTeX table with L1 as merged bold header rows."""
    n_data_cols = 3  # Proteins, New Compounds, New Clusters
    total_cols = 1 + n_data_cols  # Target Class + data cols
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Target Class & Proteins & New Compounds & New Clusters \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        name = row["Target Class"]

        if name == "Total":
            p = f"{int(row['Proteins']):,}"
            c = f"{int(row['New Compounds']):,}"
            cl = f"{int(row['New Clusters']):,}"
            lines.append(r"\midrule")
            lines.append(
                rf"\textbf{{{name}}} & \textbf{{{p}}} & \textbf{{{c}}} & \textbf{{{cl}}} \\")
        elif row["_is_header"]:
            # L1 header: bold name spanning all columns, no numbers
            lines.append(
                rf"\multicolumn{{{total_cols}}}{{l}}{{\textbf{{{name}}}}} \\")
        else:
            clean = name.strip()
            p = f"{int(row['Proteins']):,}"
            c = f"{int(row['New Compounds']):,}"
            cl = f"{int(row['New Clusters']):,}"
            # Indented L2 row (starts with spaces) vs flat L1 row
            prefix = r"\quad " if name.startswith("  ") else ""
            lines.append(
                rf"{prefix}{clean} & {p} & {c} & {cl} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def load_split_data(splits_dir: Path) -> pd.DataFrame:
    """Concatenate per-HARVEST-compound rows from every split_results CSV.

    Reads every `split_results_<UNIPROT>.csv` directly under `splits_dir`
    (HARVEST splits only -- BDB_SPLIT/ is ignored). Adds a `protein`
    column from the filename, keeps only HARVEST rows, and restricts to
    the three cluster labels A / C / CB. Columns returned:
    `[protein, cluster_id, final_label, tanimoto_sim, clean_smiles]`.
    """
    splits_dir = Path(splits_dir)
    csv_paths = sorted(splits_dir.glob("split_results_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No split_results_*.csv files found in {splits_dir}")

    parts = []
    cols = ["clean_smiles", "label", "cluster_id",
            "final_label", "tanimoto_sim"]
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, usecols=cols)
        except (ValueError, pd.errors.EmptyDataError):
            continue
        df["protein"] = csv_path.stem.replace("split_results_", "")
        parts.append(df)

    all_data = pd.concat(parts, ignore_index=True)
    all_data = all_data[all_data["label"] == "HARVEST"]
    all_data = all_data.dropna(subset=["final_label"])
    all_data = all_data[all_data["final_label"].isin(_TANIMOTO_LABEL_MAP)]
    return all_data.reset_index(drop=True)


# Reverse map: novelty_label -> final_label for compatibility with plot functions.
_NOVELTY_TO_FINAL = {v: k for k, v in _TANIMOTO_LABEL_MAP.items()}
# {"Novel": "A", "Buffer": "C", "Boundary": "CB"}


def load_split_data_from_parquet(harvest_path: Path) -> pd.DataFrame:
    """Build the same split-data DataFrame from the annotated parquet.

    The annotated parquet carries ``novelty_label``, ``cluster_id``, and
    ``tanimoto_sim`` columns produced by ``annotate_novelty.py``.  This
    function reads those, maps ``novelty_label`` back to the internal
    final_label codes (A/C/CB), deduplicates per (protein, clean_smiles),
    and returns the same schema as :func:`load_split_data`.
    """
    import pyarrow.parquet as pq
    cols = set(pq.ParquetFile(harvest_path).schema_arrow.names)
    smiles_col = next(c for c in ('clean_smiles', 'Ligand SMILES', 'smiles')
                      if c in cols)
    uniprot_col = next(c for c in ('uniprot_acc', 'Target accession')
                       if c in cols)

    df = pd.read_parquet(harvest_path,
                         columns=[uniprot_col, smiles_col,
                                  'novelty_label', 'cluster_id', 'tanimoto_sim'])
    df = df.dropna(subset=['novelty_label'])
    df['protein'] = df[uniprot_col].astype(str).str.split(';').str[0].str.strip()
    # Same accession filter as load_data_from_parquet — drop NaN/empty accessions
    df = df[df['protein'].str.match(r'^[A-Z][A-Z0-9]{4,9}$')]
    df = df.rename(columns={smiles_col: 'clean_smiles'})

    # Deduplicate per (protein, clean_smiles) — keep first occurrence
    df = df.drop_duplicates(subset=['protein', 'clean_smiles'])

    # Map novelty_label -> final_label
    df['final_label'] = df['novelty_label'].map(_NOVELTY_TO_FINAL)
    df = df.dropna(subset=['final_label'])

    return df[['protein', 'cluster_id', 'final_label',
               'tanimoto_sim', 'clean_smiles']].reset_index(drop=True)


def plot_tanimoto_hist(ax, df: pd.DataFrame,
                       title_fontsize=11, fontsize=9,
                       bins: int = 50):
    """Panel (d): overlapping Tanimoto-similarity histograms per split subset.

    Draws a step-filled histogram of nearest-neighbour Tanimoto similarity
    for the three cluster-split subsets: A -> Novel, C -> Buffer,
    CB -> Boundary. Y axis is linear (not log).
    """
    tab10 = matplotlib.colormaps["Set2"].colors
    colors = {
        "A":  tab10[0],  # green
        "C":  tab10[1],  # orange
        "CB": tab10[2],  # blue
    }
    edges = np.linspace(0.0, 1.0, bins + 1)

    print("\nTanimoto-similarity distribution (per final_label):")
    for code in ("A", "C", "CB"):
        vals = df.loc[df["final_label"] == code, "tanimoto_sim"].to_numpy()
        if not len(vals):
            print(f"  {_TANIMOTO_LABEL_MAP[code]}: 0 rows (skipped)")
            continue
        ax.hist(vals, bins=edges, histtype="step",
                color=colors[code], linewidth=2.2,
                label=f"{_TANIMOTO_LABEL_MAP[code]}")
        print(f"  {_TANIMOTO_LABEL_MAP[code]:>7}: n={len(vals):>8,} | "
              f"median={float(np.median(vals)):.3f} | "
              f"mean={float(np.mean(vals)):.3f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Max Tanimoto similarity", fontsize=fontsize)
    ax.set_ylabel("Compounds", fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_title("Similarity to nearest BindingDB ligand",
                 fontsize=title_fontsize, fontweight="bold")
    _add_panel_label(ax, "(d)", title_fontsize=title_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.legend(fontsize=fontsize - 1, frameon=False, loc="best")


def compute_plis_per_patent(harvest_parquet: Path) -> pd.Series:
    """PLI count per HARVEST patent (unique (uniprot_acc, clean_smiles))."""
    df = pd.read_parquet(harvest_parquet,
                         columns=["patent_number", "Target accession",
                                  "clean_smiles"])
    df = df.rename(columns={"Target accession": "uniprot_acc"})
    df = df.dropna(subset=["patent_number", "uniprot_acc", "clean_smiles"])
    df["uniprot_acc"] = df["uniprot_acc"].astype(str).str.split(";").str[0]
    df = df.drop_duplicates(["patent_number", "uniprot_acc", "clean_smiles"])
    return df.groupby("patent_number").size()


def compute_plis_per_bdb_patent(bdb_parquet: Path) -> pd.Series:
    """PLI count per BindingDB patent — only rows where patent_number is set."""
    df = pd.read_parquet(bdb_parquet,
                         columns=["patent_number", "uniprot_acc",
                                  "clean_smiles"])
    df = df.dropna(subset=["patent_number", "uniprot_acc", "clean_smiles"])
    df = df.drop_duplicates(["patent_number", "uniprot_acc", "clean_smiles"])
    return df.groupby("patent_number").size()


def _bin_plis(plis: pd.Series) -> pd.Series:
    bins = pd.cut(plis, bins=_PLI_BIN_EDGES, labels=_PLI_BIN_LABELS,
                  right=True, include_lowest=True)
    return bins.value_counts().reindex(_PLI_BIN_LABELS).fillna(0).astype(int)


def plot_plis_per_patent(ax, plis: pd.Series, plis_bdb: pd.Series | None = None,
                         title_fontsize=11, fontsize=9):
    """Panel (e): grouped bars — HARVEST vs BindingDB patents per PLI bin."""
    colors = matplotlib.colormaps["Set2"].colors[:3]
    counts_h = _bin_plis(plis)
    counts_b = _bin_plis(plis_bdb) if plis_bdb is not None else None

    x = np.arange(len(_PLI_BIN_LABELS))
    if counts_b is None:
        width = 0.7
        bars_h = ax.bar(x, counts_h.values, width=width, color=colors[0],
                        edgecolor="white", linewidth=0.5, label="HARVEST")
        groups = [(bars_h, counts_h.values)]
        y_max = int(counts_h.max())
    else:
        width = 0.4
        bars_h = ax.bar(x - width / 2, counts_h.values, width=width,
                        color=colors[0], edgecolor="white", linewidth=0.5,
                        label="HARVEST")
        bars_b = ax.bar(x + width / 2, counts_b.values, width=width,
                        color=colors[1], edgecolor="white", linewidth=0.5,
                        label="BindingDB")
        groups = [(bars_h, counts_h.values), (bars_b, counts_b.values)]
        y_max = max(int(counts_h.max()), int(counts_b.max()))

    # for bars, values in groups:
    #     for bar, v in zip(bars, values):
    #         ax.text(bar.get_x() + bar.get_width() / 2,
    #                 bar.get_height(), f"{int(v):,}",
    #                 ha="center", va="bottom", fontsize=fontsize - 1)

    ax.set_xticks(x)
    ax.set_xticklabels(_PLI_BIN_LABELS, fontsize=fontsize)
    ax.set_xlabel("PLIs per patent", fontsize=fontsize)
    ax.set_ylabel("Number of patents", fontsize=fontsize)
    ax.set_title("Patent complexity",
                 fontsize=title_fontsize, fontweight="bold")
    _add_panel_label(ax, "(e)", title_fontsize=title_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=fontsize - 1)
    ax.legend(fontsize=fontsize - 1, frameon=False, loc="best")

    # Headroom for value labels.
    ax.set_ylim(0, y_max * 1.12 if y_max else 1)

    def _report(name, series, counts):
        print(f"\nPLIs-per-patent distribution ({name}):")
        for label, count in counts.items():
            print(f"  {label:>7}: {int(count):>7,} patents")
        print(f"  total patents: {int(counts.sum()):,}")
        print(f"  median PLIs/patent: {float(series.median()):.1f}, "
              f"P95: {float(series.quantile(0.95)):.1f}, "
              f"max: {int(series.max()):,}")

    _report("HARVEST", plis, counts_h)
    if plis_bdb is not None:
        _report("BindingDB", plis_bdb, counts_b)


def _cliff_worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _assay_mean_tanimoto(task):
    """(assay_key, smiles_list, seed) -> (key, mean_pairwise_tanimoto, n_valid)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    key, smiles_list, seed = task
    if len(smiles_list) > _CLIFF_TANIMOTO_CAP:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(smiles_list), _CLIFF_TANIMOTO_CAP, replace=False)
        smiles_list = [smiles_list[i] for i in idx]
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=_CLIFF_FP_RADIUS, fpSize=_CLIFF_FP_NBITS)
    fps = []
    for smi in smiles_list:
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fps.append(gen.GetFingerprint(m))
    n = len(fps)
    if n < 2:
        return key, float("nan"), n
    total, cnt = 0.0, 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += float(np.sum(sims))
        cnt += len(sims)
    return key, (total / cnt if cnt else float("nan")), n


def _normalize_bdb_for_cliffs(bdb_parquet: Path,
                                validity_comments: tuple) -> pd.DataFrame:
    """BindingDB rows filtered to `source == 'bindingdb'` + given
    validity_comment values + a populated `BindingDB Entry DOI` column.

    Assay grouping key: (uniprot_id, activity_type, doi). The DOI is the
    real BindingDB per-entry identifier (~57 rows per DOI on average),
    finer than patent_number / application_number for the patent slice
    and finer than anything else for the ChEMBL/PubChem slice.
    """
    cols = ["source", "validity_comment", "uniprot_id", "activity_type",
            "activity_value", "activity_relation", "BindingDB Entry DOI",
            "clean_smiles", "inchikey_clean"]
    df = pd.read_parquet(bdb_parquet, columns=cols).rename(columns={
        "activity_value": "value_nM", "activity_relation": "relation",
        "clean_smiles": "smiles", "inchikey_clean": "inchikey",
        "BindingDB Entry DOI": "doi",
    })
    df = df[(df["source"] == "bindingdb")
            & df["validity_comment"].isin(validity_comments)
            & df["activity_type"].notna() & df["value_nM"].notna()
            & df["smiles"].notna() & df["doi"].notna()].copy()
    df["value_nM"] = pd.to_numeric(df["value_nM"], errors="coerce")
    df = df[df["value_nM"].notna()]
    df["relation"] = df["relation"].fillna("=")
    return df


def _normalize_harvest_for_cliffs(harvest_parquet: Path) -> pd.DataFrame:
    """HARVEST wide-format -> (patent_number, assay_id, activity_type, value)."""
    aff = ["IC50 (nM)", "Ki (nM)", "Kd (nM)", "EC50 (nM)"]
    type_for = {"IC50 (nM)": "IC50", "Ki (nM)": "Ki",
                "Kd (nM)": "Kd", "EC50 (nM)": "EC50"}
    cols = aff + ["relation", "assay_id", "patent_number", "clean_smiles",
                  "best_smiles", "best_inchikey", "Ligand InChI Key"]
    df = pd.read_parquet(harvest_parquet, columns=cols)

    def _populated(c):
        s = df[c].astype("string").str.strip()
        return df[c].notna() & (s != "") & (s.str.lower() != "nan")

    masks = [_populated(c) for c in aff]
    df["activity_type"] = np.select(
        masks, [type_for[c] for c in aff], default=None)
    valstr = np.select(masks, [df[c].astype("string") for c in aff],
                       default=pd.NA)
    df["value_nM"] = pd.to_numeric(pd.Series(valstr, index=df.index),
                                   errors="coerce")
    df["smiles"] = df["clean_smiles"].fillna(df["best_smiles"])
    df["inchikey"] = df["best_inchikey"].fillna(df["Ligand InChI Key"])
    df = df[df["activity_type"].notna() & df["value_nM"].notna()
            & df["smiles"].notna()].copy()
    # HARVEST 'range' rows already carry the midpoint in value_nM, treat as '='
    rel = df["relation"].astype("string")
    df["relation"] = rel.where(rel != "range", "=")
    return df


def _curate_assays_for_cliffs(df: pd.DataFrame, key_cols) -> pd.DataFrame:
    """Apply the Boltz-2 hit-to-lead curation.

    Returns per-passing-assay DataFrame with columns
    ["cliff_score_iqr", "n_points"] indexed by assay_key. Multiprocessed
    Morgan-Tanimoto stage only runs on the assays that already survived the
    cheap n / std / uniqueness filters.
    """
    rel = df["relation"].astype("string")
    v = df["value_nM"]
    keep = (v > 0) & ((rel == "=") | (rel.isin([">", ">="])
                                       & (v >= _CLIFF_GT_KEEP_NM)))
    df = df[keep].copy()

    df["y"] = np.log10(_CLIFF_LOG_REF_NM) - np.log10(df["value_nM"].to_numpy())
    parts = [df[c].astype("string").fillna("__NA__") for c in key_cols]
    df["assay_key"] = parts[0].str.cat(parts[1:], sep="||")

    amin = df.groupby("assay_key")["value_nM"].transform("min")
    df = df[amin >= _CLIFF_UNIT_ERROR_NM].copy()

    g = df.groupby("assay_key")
    at = g["y"].agg(n_points="count", std_y="std")
    qs = g["y"].agg(q25=lambda s: np.percentile(s.to_numpy(), 25),
                    q75=lambda s: np.percentile(s.to_numpy(), 75))
    at = at.join(qs)
    at["n_unique"] = g["value_nM"].nunique()
    at["unique_ratio"] = at["n_unique"] / at["n_points"]
    at["cliff_score_iqr"] = at["q75"] - at["q25"]

    cheap = ((at["n_points"] >= _CLIFF_MIN_POINTS)
             & (at["std_y"] >= _CLIFF_MIN_STD_Y)
             & (at["n_unique"] >= _CLIFF_MIN_UNIQUE)
             & (at["unique_ratio"] >= _CLIFF_MIN_UNIQUE_RATIO))
    n_cheap = int(cheap.sum())
    print(f"    cheap-pass assays (n/std/unique): {n_cheap:,} / {len(at):,}")
    cand = set(at.index[cheap])

    sub = df[df["assay_key"].isin(cand)].copy()
    sub["dedup"] = sub["inchikey"].astype("string").fillna(
        sub["smiles"].astype("string"))
    # Per-molecule y (mean over replicate measurements of the same InChIKey
    # within one assay) — used both for pair counting and, via first(), for
    # the SMILES list that feeds the Tanimoto worker.
    per_mol = (sub.groupby(["assay_key", "dedup"])
               .agg(y_mean=("y", "mean"),
                    smiles=("smiles", "first")).reset_index())
    n_unique_mols = per_mol.groupby("assay_key").size()
    grp = per_mol.groupby("assay_key")["smiles"].apply(list)
    tasks = [(k, grp[k], _CLIFF_RANDOM_SEED + i)
             for i, k in enumerate(grp.index)]
    tan = {}
    if tasks:
        n_workers = max(1, min(12, (os.cpu_count() or 2) - 2))
        print(f"    computing mean Tanimoto for {len(tasks):,} assays on "
              f"{n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers,
                                  initializer=_cliff_worker_init) as ex:
            for k, mt, _n in ex.map(_assay_mean_tanimoto, tasks, chunksize=64):
                tan[k] = mt
    at["mean_tanimoto"] = pd.Series(tan)
    passed = (cheap & (at["mean_tanimoto"] >= _CLIFF_MIN_MEAN_TANIMOTO)
              & at["cliff_score_iqr"].notna())
    print(f"    final passing assays: {int(passed.sum()):,}")

    # Cliff-pair counting: within each passing assay, sort unique molecules
    # by y, split at the median, and pair i-th highest with i-th lowest. No
    # ligand is used twice. Count pairs whose |Δy| clears the threshold
    # (default 2 log10-units = 100-fold potency swing).
    cliff_pairs = {}
    passing_keys = set(at.index[passed])
    y_by_assay = per_mol[per_mol["assay_key"].isin(passing_keys)] \
        .groupby("assay_key")["y_mean"].apply(lambda s: np.sort(s.to_numpy()))
    for k, ys in y_by_assay.items():
        half = len(ys) // 2
        if half == 0:
            cliff_pairs[k] = 0
            continue
        low  = ys[:half]
        high = ys[-half:][::-1]        # descending
        cliff_pairs[k] = int(np.sum((high - low) >= _CLIFF_LOG_DIFF_THRESHOLD))
    at["n_cliff_pairs"] = pd.Series(cliff_pairs)

    out = at.loc[passed, ["cliff_score_iqr", "n_points"]].copy()
    out["n_unique_mols"] = n_unique_mols.reindex(out.index).fillna(0).astype(int)
    out["n_cliff_pairs"] = at.loc[passed, "n_cliff_pairs"].fillna(0).astype(int)
    return out


def _cliff_signature(bdb_path: Path, harvest_path: Path) -> str:
    parts = ["v7-cliff-pair-threshold"]  # bump when the curation semantics change
    for p in (bdb_path, harvest_path):
        p = Path(p)
        parts.append(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}")
    parts.append(
        f"min_points={_CLIFF_MIN_POINTS}|min_std={_CLIFF_MIN_STD_Y}|"
        f"min_unique={_CLIFF_MIN_UNIQUE}|min_uratio={_CLIFF_MIN_UNIQUE_RATIO}|"
        f"min_tan={_CLIFF_MIN_MEAN_TANIMOTO}|"
        f"cliff_dy={_CLIFF_LOG_DIFF_THRESHOLD}|"
        f"fp={_CLIFF_FP_RADIUS}/{_CLIFF_FP_NBITS}|cap={_CLIFF_TANIMOTO_CAP}")
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:12]


_CLIFF_SLICES = ("harvest", "bdb_patents", "bdb_chempub")


def compute_or_load_cliff_scores(bdb_path: Path, harvest_path: Path,
                                  cache_path: Path) -> dict:
    """Return {slice: DataFrame} for the three source slices.

    Slices:
      * harvest      -- HARVEST patents keyed on (patent, assay_id, activity_type)
      * bdb_patents  -- BindingDB validity_comment in {US Patent, WIPO},
                        keyed on (uniprot_id, activity_type, DOI)
      * bdb_chempub  -- BindingDB validity_comment in {ChEMBL, PubChem},
                        keyed on (uniprot_id, activity_type, DOI)

    Rows with a null `BindingDB Entry DOI` are dropped. All slices use the
    Boltz-2 strict rule ('=' only, plus inactive-only '>' / '>=' >= 10 uM).
    """
    sig = _cliff_signature(bdb_path, harvest_path)
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as npz:
                if str(npz["signature"]) == sig:
                    print(f"  Cliff-scores cache hit ({cache_path}) - "
                          f"signature {sig}")
                    return {
                        s: pd.DataFrame({
                            "cliff_score_iqr": npz[f"{s}_cliff"],
                            "n_points":        npz[f"{s}_npts"],
                            "n_unique_mols":   npz[f"{s}_nmols"],
                            "n_cliff_pairs":   npz[f"{s}_pairs"],
                        }) for s in _CLIFF_SLICES
                    }
                print(f"  Cliff-scores cache signature mismatch "
                      f"(cached {npz['signature']} != {sig}); recomputing")
        except Exception as exc:
            print(f"  Cliff-scores cache read failed ({exc}); recomputing")

    print("  Normalizing HARVEST for cliffs...")
    df_harvest = _normalize_harvest_for_cliffs(harvest_path)
    print(f"    rows after normalize: {len(df_harvest):,}")
    print("  Curating HARVEST assays...")
    harvest_out = _curate_assays_for_cliffs(df_harvest,
                                              list(_HARVEST_ASSAY_KEY))

    print("  Normalizing BDB patents "
          f"(validity_comment in {_BDB_PATENT_VC})...")
    df_bpat = _normalize_bdb_for_cliffs(bdb_path, _BDB_PATENT_VC)
    print(f"    rows after normalize: {len(df_bpat):,}")
    print("  Curating BDB patent assays (keyed on DOI)...")
    bdb_patents_out = _curate_assays_for_cliffs(df_bpat,
                                                  list(_BDB_ASSAY_KEY))

    print("  Normalizing BDB ChEMBL/PubChem "
          f"(validity_comment in {_BDB_CHEMPUB_VC})...")
    df_bcp = _normalize_bdb_for_cliffs(bdb_path, _BDB_CHEMPUB_VC)
    print(f"    rows after normalize: {len(df_bcp):,}")
    print("  Curating BDB ChEMBL/PubChem assays (keyed on DOI)...")
    bdb_chempub_out = _curate_assays_for_cliffs(df_bcp,
                                                  list(_BDB_ASSAY_KEY))

    out = {
        "harvest":     harvest_out.reset_index(drop=True),
        "bdb_patents": bdb_patents_out.reset_index(drop=True),
        "bdb_chempub": bdb_chempub_out.reset_index(drop=True),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, signature=np.array(sig),
             **{f"{s}_cliff": out[s]["cliff_score_iqr"].to_numpy()
                for s in _CLIFF_SLICES},
             **{f"{s}_npts": out[s]["n_points"].to_numpy()
                for s in _CLIFF_SLICES},
             **{f"{s}_nmols": out[s]["n_unique_mols"].to_numpy()
                for s in _CLIFF_SLICES},
             **{f"{s}_pairs": out[s]["n_cliff_pairs"].to_numpy()
                for s in _CLIFF_SLICES})
    print(f"  Saved cliff-scores cache to {cache_path} (signature {sig})")
    return out


def plot_activity_cliffs(ax, cliffs: dict,
                          title_fontsize=11, fontsize=9):
    """Panel (f): one bar per source of hit-to-lead activity-cliff pairs.

    A "cliff pair" = two distinct molecules from the same passing assay,
    matched max-vs-min after sorting by activity, whose |Δy| in log10
    units is at least `_CLIFF_LOG_DIFF_THRESHOLD` (default 2.0 == 100-fold
    potency swing). Each ligand contributes to at most one pair (no reuse).
    HARVEST (green), BindingDB patents (orange), BindingDB ChEMBL/PubChem
    (pink).
    """
    palette = matplotlib.colormaps["Set2"].colors[:3]

    sources = [
        ("harvest",     "HARVEST",              palette[0]),
        ("bdb_patents", "BindingDB\n(Patents US)",  palette[1]),
        ("bdb_chempub", "BindingDB\n(ChEMBL/PubChem)", palette[2]),
    ]

    def _pair_count(df):
        if not len(df):
            return 0
        return int(df["n_cliff_pairs"].sum())

    values = [_pair_count(cliffs[key]) for key, _, _ in sources]
    labels = [name for _, name, _ in sources]
    colors = [c for _, _, c in sources]

    x = np.arange(len(sources))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(v):,}", ha="center", va="bottom",
                fontsize=fontsize - 1)

    y_max = max(values) if values else 1
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=10)
    ax.set_ylabel(f"Activity-cliff pairs "
                  f"(|Δlog$_{{10}}$| ≥ {_CLIFF_LOG_DIFF_THRESHOLD:g}",
                  fontsize=fontsize)
    ax.set_ylim(0, y_max * 1.15 if y_max else 1)
    ax.set_title("Hit-to-lead activity cliffs",
                 fontsize=title_fontsize, fontweight="bold")
    _add_panel_label(ax, "(f)", title_fontsize=title_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=fontsize - 1)

    print(f"\nHit-to-lead availability (Boltz-2 curation, "
          f"|Δlog10| >= {_CLIFF_LOG_DIFF_THRESHOLD:g}):")
    for (key, name, _), n_pairs in zip(sources, values):
        df = cliffs[key]
        n = len(df)
        plis = int(df["n_points"].sum()) if n else 0
        n_pot_pairs = int((df["n_unique_mols"] // 2).sum()) if n else 0
        med = float(df["cliff_score_iqr"].median()) if n else float("nan")
        print(f"  {name:<26}: assays = {n:>7,} | cliff pairs = {n_pairs:>8,} "
              f"| candidate pairs = {n_pot_pairs:>8,} | "
              f"PLIs = {plis:>10,} | median IQR = {med:.3f}")


def plot_reserved(ax, title_fontsize=11, fontsize=9):
    """Panel (f) fallback (--skip-cliffs)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, "cliffs skipped\n(--skip-cliffs)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, style="italic", color="0.55")
    _add_panel_label(ax, "(f)", title_fontsize=title_fontsize)


def main():
    global _CLIFF_LOG_DIFF_THRESHOLD

    parser = argparse.ArgumentParser(description="Plot split statistics pie charts")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--harvest",
                            help="Annotated HARVEST parquet with novelty_label column "
                                 "(replaces --data-dir)")
    data_group.add_argument("--data-dir", default="cluster_split/split_v14",
                            help="Directory containing df_new_protein_stats.csv and "
                                 "df_overlap_stats.csv (legacy)")
    parser.add_argument("--bdb", default="../../projects/patents/full_bdb_chembl_fix.parquet",
                        help="Path to BDB parquet file")
    parser.add_argument("--chembl", default="curated_data/target_info_chembl.csv.gz",
                        help="Path to ChEMBL target info CSV")
    parser.add_argument("--output", default='manuscript/stats_pies.png',
                        help="Output PNG path (default: manuscript/stats_pies.png)")
    parser.add_argument("--min-proteins", type=int, default=40,
                        help="Min proteins in L2 class before collapsing to Other")
    parser.add_argument("--min-l1-proteins", type=int, default=50,
                        help="Min proteins in L1 class before collapsing to Other")
    parser.add_argument("--harvest-parquet", default="final_v16_clean.parquet",
                        help="HARVEST parquet (used for PLIs/patent panel)")
    parser.add_argument("--splits-dir",
                        default="cluster_split/split_v14/splits",
                        help="Directory with per-protein split_results_<UNIPROT>.csv "
                             "(HARVEST splits) — used for panel (d) Tanimoto "
                             "histograms.")
    parser.add_argument("--tanimoto-bins", type=int, default=50,
                        help="Number of histogram bins in [0, 1] for panel (d)")
    parser.add_argument("--cliffs-cache",
                        default="manuscript/cliff_scores_cache.npz",
                        help="Path for cached per-assay cliff scores (.npz). "
                             "Re-renders reuse the cache on signature match.")
    parser.add_argument("--skip-cliffs", action="store_true",
                        help="Skip panel (f) cliffs compute; draw a placeholder")
    parser.add_argument("--cliff-log-diff", type=float,
                        default=_CLIFF_LOG_DIFF_THRESHOLD,
                        help="Min |Δlog10 activity| between the hi- and lo-"
                             "activity ligand of a pair for it to count as an "
                             "activity cliff (default: %(default)s = 100-fold).")
    args = parser.parse_args()

    _CLIFF_LOG_DIFF_THRESHOLD = float(args.cliff_log_diff)

    bdb_path = Path(args.bdb)
    output_path = Path(args.output) if args.output else Path("manuscript/stats_pies.png")

    # When --harvest is used, panels (e)/(f) should read the same parquet
    # unless --harvest-parquet is explicitly given.
    if args.harvest:
        harvest_path = Path(args.harvest)
    else:
        harvest_path = Path(args.harvest_parquet)

    if args.harvest:
        df_new, df_overlap, n_new_proteins, n_overlap_proteins, n_bdb_only_proteins = \
            load_data_from_parquet(Path(args.harvest), bdb_path)
    else:
        data_dir = Path(args.data_dir)
        df_new, df_overlap, n_new_proteins, n_overlap_proteins, n_bdb_only_proteins = \
            load_data(data_dir, bdb_path)

    print(f"Proteins — new: {n_new_proteins}, overlap: {n_overlap_proteins}, "
          f"bdb-only: {n_bdb_only_proteins}")

    # --- Figure: 2 rows x 3 columns ---
    fig = plt.figure(figsize=(14, 8))
    title_fontsize = 11
    fontsize = 9

    gs = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.55,
                           height_ratios=[1, 1.15])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    plot_pie_proteins(ax_a, n_new_proteins, n_bdb_only_proteins, n_overlap_proteins,
                      fontsize=fontsize, title_fontsize=title_fontsize)

    # Panels (b), (c), (d) need per-compound split data (final_label + tanimoto).
    print("Loading split data (final_label + cluster_id + tanimoto_sim) ...")
    if args.harvest:
        split_df = load_split_data_from_parquet(Path(args.harvest))
    else:
        split_df = load_split_data(Path(args.splits_dir))
    print(f"  loaded {len(split_df):,} HARVEST-compound rows across "
          f"{split_df['protein'].nunique():,} proteins; "
          f"final_label counts: "
          f"{split_df['final_label'].value_counts().to_dict()}")

    plot_compounds_diver(ax_b, split_df, df_new,
                         fontsize=fontsize, title_fontsize=title_fontsize)
    plot_clusters_diver(ax_c, split_df, df_new,
                        fontsize=fontsize, title_fontsize=title_fontsize)
    plot_tanimoto_hist(ax_d, split_df, bins=args.tanimoto_bins,
                       title_fontsize=title_fontsize, fontsize=fontsize)

    # Panel (e): PLIs per patent — HARVEST + BindingDB
    print("Computing PLIs per HARVEST patent...")
    plis = compute_plis_per_patent(harvest_path)
    print("Computing PLIs per BindingDB patent...")
    plis_bdb = compute_plis_per_bdb_patent(bdb_path)
    plot_plis_per_patent(ax_e, plis, plis_bdb=plis_bdb,
                         title_fontsize=title_fontsize, fontsize=fontsize)

    # Panel (f): activity-cliffs (Boltz-2 hit-to-lead curation)
    if args.skip_cliffs:
        print("Skipping cliffs (--skip-cliffs); drawing placeholder in (f)")
        plot_reserved(ax_f, title_fontsize=title_fontsize, fontsize=fontsize)
    else:
        print("Computing activity-cliff scores per assay...")
        cliffs = compute_or_load_cliff_scores(
            bdb_path, harvest_path, Path(args.cliffs_cache))
        plot_activity_cliffs(ax_f, cliffs,
                              title_fontsize=title_fontsize,
                              fontsize=fontsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=600, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")

    # --- Summary table by L2 classification ---
    summary = build_l2_summary(df_new, df_overlap, args.chembl,
                               min_proteins=args.min_proteins,
                               min_l1_proteins=args.min_l1_proteins)

    csv_path = output_path.with_name("l2_harvest_only_summary.csv")
    save_csv_table(summary, csv_path)
    print(f"Saved CSV  to {csv_path}")

    tex_path = output_path.with_name("l2_harvest_only_summary.tex")
    save_latex_table(summary, tex_path)
    print(f"Saved LaTeX to {tex_path}")

    print("\nSummary table:")
    print(summary.drop(columns=["_is_header"]).fillna("").to_string(index=False))


if __name__ == "__main__":
    main()