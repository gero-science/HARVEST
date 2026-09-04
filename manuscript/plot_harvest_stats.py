#!/usr/bin/env python3
"""Plot 3 subplots comparing HARVEST and BindingDB data:
(a) Documents per year, (b) Records per document per year,
(c) Data source comparison (unique PLIs by source)."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def load_data(harvest_path: str, bdb_path: str, patent_mapping_path: str):
    """Load and prepare HARVEST, BDB, and patent mapping data."""
    df_harvest = pd.read_parquet(harvest_path)
    df_harvest = df_harvest.rename(columns={'Target accession': 'uniprot_acc'})
    df_harvest["year"] = pd.to_numeric(df_harvest["year"], errors="coerce")
    df_bdb = pd.read_parquet(bdb_path)
    patent_mapping = pd.read_csv(patent_mapping_path)

    # Merge patent mapping into BDB
    df_bdb = df_bdb.merge(patent_mapping, how="left", on="patent_number")

    # Derive year columns for BDB
    df_bdb["publication_date"] = pd.to_datetime(
        df_bdb["publication_date"], format="mixed")
    df_bdb["year"] = df_bdb["publication_date"].dt.year
    df_bdb["year_application"] = [
        int(x[2:6]) if isinstance(x, str) else np.nan
        for x in df_bdb["application_number"]
    ]

    return df_harvest, df_bdb


# Patent-office prefixes considered "patent-derived" in BindingDB.
# US covers US and USRE (US Reissue); WO covers WIPO filings; EP covers EPO.
PATENT_PREFIX_RE = r"^(US|WO|EP)"


def _row(level, h_only, b_only, shared):
    union = len(h_only) + len(b_only) + len(shared)
    def cell(n):
        return f"{n:>10,} ({100 * n / union:5.2f}%)"
    return f"  {level:<9} {cell(len(h_only))}  {cell(len(b_only))}  {cell(len(shared))}"


def print_patent_overlap_stats(df_harvest, df_bdb):
    """Print HARVEST vs BindingDB-patent-subset overlap at two granularities.

    Isolates the patent-derived slice of BindingDB (rows whose patent_number
    prefix is US / WO / EP), then reports a 4-column table with rows for
    PLI-level and document-level overlap. HARVEST's `patent_number` is the
    patent application number; BindingDB's application number comes from the
    merged patent_mapping table.
    """
    is_patent = df_bdb["patent_number"].fillna("").str.match(
        PATENT_PREFIX_RE, na=False)
    bdb_patent = df_bdb.loc[is_patent, ["patent_number", "application_number",
                                        "uniprot_acc", "clean_smiles"]].copy()
    bdb_patent["pfx"] = bdb_patent["patent_number"].str.extract(
        r"^(US|WO|EP)", expand=False)
    prefix_counts = bdb_patent["pfx"].value_counts()

    # PLI level
    harvest_plis = set(map(tuple, df_harvest[["uniprot_acc", "clean_smiles"]]
                           .drop_duplicates().to_numpy()))
    bdb_patent_plis = set(map(tuple, bdb_patent[["uniprot_acc", "clean_smiles"]]
                              .drop_duplicates().to_numpy()))
    h_only_p = harvest_plis - bdb_patent_plis
    b_only_p = bdb_patent_plis - harvest_plis
    shared_p = harvest_plis & bdb_patent_plis

    # Document level (HARVEST patent_number == application number)
    harvest_docs = set(df_harvest["patent_number"].dropna().unique())
    unmapped = int(bdb_patent["application_number"].isna().sum())
    bdb_docs = set(bdb_patent["application_number"].dropna().unique())
    h_only_d = harvest_docs - bdb_docs
    b_only_d = bdb_docs - harvest_docs
    shared_d = harvest_docs & bdb_docs

    print("\n=== BindingDB patent-derived subset ===")
    print(f"Rows: {int(is_patent.sum()):,}  (prefixes: " + ", ".join(
        f"{k}={int(v):,}" for k, v in prefix_counts.items()) + ")")
    print(f"Unmapped to application_number: {unmapped:,} rows "
          f"(excluded from document-level overlap only)")
    print(f"HARVEST unique PLIs {len(harvest_plis):,} | documents {len(harvest_docs):,}")
    print(f"BDB-patent unique PLIs {len(bdb_patent_plis):,} | documents {len(bdb_docs):,}")

    print("\n=== HARVEST vs BindingDB-patent overlap ===")
    header = (f"  {'Level':<9} {'HARVEST only':>18}  "
              f"{'BindingDB only':>18}  {'Shared':>18}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(_row("PLI",      h_only_p, b_only_p, shared_p))
    print(_row("Document", h_only_d, b_only_d, shared_d))
    print()


def plot_documents_per_year(ax, df_harvest, df_bdb, colors,
                            title_fontsize=11, fontsize=9):
    """Subplot (a): Number of unique patents per year."""
    harvest_docs = df_harvest.groupby("year")["patent_number"].nunique()
    bdb_us = df_bdb[df_bdb["validity_comment"] == "US Patent"]
    bdb_docs = bdb_us.groupby("year_application")["patent_number"].nunique()

    ax.plot(harvest_docs.index, harvest_docs.values,
            label="HARVEST", color=colors[0])
    ax.plot(bdb_docs.index, bdb_docs.values,
            label="BindingDB", color=colors[1])

    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_ylabel("Number of documents", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.set_title("Documents per year", fontsize=title_fontsize, fontweight="bold")
    ax.text(-0.15, 1.15, "(a)", transform=ax.transAxes,
            fontsize=title_fontsize, fontweight="bold", va="top")


def plot_records_per_document(ax, df_harvest, df_bdb, colors,
                              title_fontsize=11, fontsize=9):
    """Subplot (b): Records per document per year."""
    # HARVEST
    h_count = df_harvest.groupby("year")["patent_number"].count()
    h_nunique = df_harvest.groupby("year")["patent_number"].nunique()
    h_ratio = h_count / h_nunique
    ax.plot(h_ratio.index, h_ratio.values, label="HARVEST", color=colors[0])

    # BindingDB (US Patent only, filter years with >10 docs)
    bdb_us = df_bdb[df_bdb["validity_comment"] == "US Patent"]
    b_count = bdb_us.groupby("year_application")["patent_number"].count()
    b_nunique = bdb_us.groupby("year_application")["patent_number"].nunique()
    m = b_nunique > 10
    b_ratio = b_count[m] / b_nunique[m]
    ax.plot(b_ratio.index, b_ratio.values, label="BindingDB", color=colors[1])

    # HARVEST ∩ BindingDB
    bdb_app_numbers = df_bdb["application_number"].dropna().unique()
    mask = df_harvest["patent_number"].isin(bdb_app_numbers)
    hi_count = df_harvest[mask].groupby("year")["patent_number"].count()
    hi_nunique = df_harvest[mask].groupby("year")["patent_number"].nunique()
    m2 = hi_nunique > 10
    hi_ratio = hi_count[m2] / hi_nunique[m2]
    ax.plot(hi_ratio.index, hi_ratio.values,
            label="HARVEST ∩ BindingDB", color=colors[2])

    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_ylabel("Records per document", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.set_title("Records per document per year",
                 fontsize=title_fontsize, fontweight="bold")
    ax.text(-0.15, 1.15, "(b)", transform=ax.transAxes,
            fontsize=title_fontsize, fontweight="bold", va="top")


def plot_data_source_comparison(ax, df_harvest, df_bdb, colors,
                                pli_threshold=500_000,
                                title_fontsize=11, fontsize=9):
    """Subplot (c): Horizontal bar chart — HARVEST vs stacked BindingDB sources."""
    # Deduplicate both datasets by (uniprot_acc, clean_smiles)
    harvest_plis = df_harvest.drop_duplicates(
        subset=["uniprot_acc", "clean_smiles"]
    ).shape[0]

    # BDB: group by validity_comment, count unique PLIs per source
    bdb_dedup = df_bdb.drop_duplicates(subset=["uniprot_acc", "clean_smiles"])
    source_counts = bdb_dedup.groupby("validity_comment").size()

    # Collapse small sources into "Other"
    big = source_counts[source_counts >= pli_threshold].sort_values(ascending=False)
    small = source_counts[source_counts < pli_threshold]
    if not small.empty:
        big["Other"] = small.sum()

    # Two rows: HARVEST (top) and BindingDB (bottom, stacked by source)
    y_harvest = 1
    y_bdb = 0
    bar_height = 0.5

    # HARVEST bar with count inside
    ax.barh(y_harvest, harvest_plis, height=bar_height, color=colors[0])
    ax.text(harvest_plis / 2, y_harvest, f"{harvest_plis:,.0f}",
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color="white")

    print("HARVEST bigger x times:\n", harvest_plis / big)

    # BindingDB stacked bar
    left = 0
    bdb_colors = list(colors[1:len(big) + 1])
    handles = []
    for i, (source, count) in enumerate(big.items()):
        bar = ax.barh(y_bdb, count, left=left, height=bar_height,
                       color=bdb_colors[i % len(bdb_colors)])
        handles.append((bar, source))
        left += count

    # Total BDB count inside bar
    bdb_total = big.sum()
    ax.text(bdb_total / 2, y_bdb, f"{bdb_total:,.0f}",
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color="white")

    ax.set_yticks([y_bdb, y_harvest])
    ax.set_yticklabels(["BindingDB", "HARVEST"], fontsize=fontsize)
    ax.set_xlim(right=max(harvest_plis, bdb_total) * 1.05)
    ax.set_xlabel("Unique PLIs", fontsize=fontsize)

    # Legend for BDB sub-sources inside plot
    ax.legend([h[0] for h in handles], [h[1] for h in handles],
              loc="center right", fontsize=fontsize - 1, frameon=True,
              framealpha=0.9)
    ax.set_title("Data source comparison",
                 fontsize=title_fontsize, fontweight="bold")
    ax.text(-0.15, 1.15, "(c)", transform=ax.transAxes,
            fontsize=title_fontsize, fontweight="bold", va="top")


def main():
    parser = argparse.ArgumentParser(
        description="Plot HARVEST vs BindingDB comparison statistics")
    parser.add_argument(
        "--harvest",
        default="final_v16_clean.parquet",
        help="Path to HARVEST parquet file")
    parser.add_argument(
        "--bdb",
        default="full_bdb_chembl_fix.parquet",
        help="Path to BDB parquet file")
    parser.add_argument(
        "--patent-mapping",
        default="curated_data/patent_mapping.csv",
        help="Path to patent mapping CSV")
    parser.add_argument(
        "--output",
        default="manuscript/harvest_stats.png",
        help="Output PNG path")
    parser.add_argument(
        "--pli-threshold", type=int, default=500_000,
        help="PLI count threshold for collapsing small BDB sources into Other")
    args = parser.parse_args()

    print("Loading data...")
    df_harvest, df_bdb = load_data(args.harvest, args.bdb, args.patent_mapping)
    print(f"  HARVEST: {len(df_harvest):,} rows")
    print(f"  BDB:     {len(df_bdb):,} rows")

    print_patent_overlap_stats(df_harvest, df_bdb)

    colors = matplotlib.colormaps["Set2"].colors
    title_fontsize = 11
    fontsize = 9

    fig = plt.figure(figsize=(12, 3.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    plot_documents_per_year(ax1, df_harvest, df_bdb, colors,
                            title_fontsize=title_fontsize, fontsize=fontsize)
    plot_records_per_document(ax2, df_harvest, df_bdb, colors,
                              title_fontsize=title_fontsize, fontsize=fontsize)
    plot_data_source_comparison(ax3, df_harvest, df_bdb, colors,
                                pli_threshold=args.pli_threshold,
                                title_fontsize=title_fontsize, fontsize=fontsize)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
