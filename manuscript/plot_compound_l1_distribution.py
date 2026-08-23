#!/usr/bin/env python3
"""Diverging ("tornado") bar chart of compound distribution by ChEMBL L1 class.

One row per L1 target class: BindingDB compounds extend left, HARVEST extend
right, on a mirrored log x-axis. Each side is a single bar whose length is that
source's total compounds; the inner segment (nearest the center) is the
shared/overlap portion, the outer segment is the source-only portion.

Data comes only from the precomputed per-protein split-stats CSVs.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


def load_and_aggregate(data_dir: Path, chembl_path: str,
                       top_n: int = 10,
                       force_others: tuple = ()) -> pd.DataFrame:
    """Aggregate compound counts per ChEMBL L1 class from the split-stats CSVs.

    The `top_n` largest classes (by total compounds) are kept individually;
    the remaining smaller classes are collapsed into a single "Others" row.
    Any class named in `force_others` is always collapsed into "Others",
    even if it ranks within the top `top_n`.

    Returns a DataFrame indexed by L1 with columns:
        harvest_total, harvest_shared, harvest_only,
        bdb_total, bdb_shared, bdb_only
    Rows are ordered for bottom-up plotting (smallest first; "Others" pinned
    to the bottom).
    """
    df_new = pd.read_csv(data_dir / "df_new_protein_stats.csv")
    df_overlap = pd.read_csv(data_dir / "df_overlap_stats.csv")

    chembl = pd.read_csv(chembl_path)
    chembl_map = chembl.drop_duplicates("accession")[["accession", "l1"]].copy()
    chembl_map.rename(columns={"accession": "protein"}, inplace=True)

    # HARVEST-only proteins: every compound is HARVEST-only, no BindingDB data.
    new = pd.DataFrame({
        "protein": df_new["protein"],
        "harvest_total": df_new["n_compounds"],
        "harvest_shared": 0,
        "bdb_total": 0,
        "bdb_shared": 0,
        "bdb_only": 0,
    })

    # Shared proteins: split into source-only and overlap portions.
    overlap = pd.DataFrame({
        "protein": df_overlap["protein"],
        "harvest_total": df_overlap["n_harvest_compounds"],
        "harvest_shared": df_overlap["n_harvest_overlap_compounds"],
        "bdb_total": df_overlap["n_bdb_compounds"],
        "bdb_shared": df_overlap["n_bdb_overlap_compounds"],
        "bdb_only": df_overlap["n_bdb_only_compounds"],
    })

    combined = pd.concat([new, overlap], ignore_index=True)
    combined = combined.merge(chembl_map, on="protein", how="left")
    combined["l1"] = combined["l1"].fillna("Unclassified")

    agg = combined.groupby("l1").agg(
        harvest_total=("harvest_total", "sum"),
        harvest_shared=("harvest_shared", "sum"),
        bdb_total=("bdb_total", "sum"),
        bdb_shared=("bdb_shared", "sum"),
        bdb_only=("bdb_only", "sum"),
    )
    agg["harvest_only"] = agg["harvest_total"] - agg["harvest_shared"]

    total = agg["harvest_total"] + agg["bdb_total"]
    ranked = total.sort_values(ascending=False).index.tolist()
    force = set(force_others)
    kept = [c for c in ranked[:top_n] if c not in force]
    rest = [c for c in ranked if c not in kept]

    result = agg.loc[kept].copy()
    if rest:
        result.loc["Others"] = agg.loc[rest].sum()

    # Plot bottom-up: smallest kept class first, "Others" pinned to the bottom.
    kept_asc = total.loc[kept].sort_values().index.tolist()
    plot_order = (["Others"] + kept_asc) if rest else kept_asc
    return result.loc[plot_order]


def plot_diverging(ax, df: pd.DataFrame):
    """Draw the diverging log bar chart on `ax` (classes ordered bottom-up).

    Each side has two bars anchored at the center: a full-height source-only
    bar and a thinner nested 'shared' bar. Both tips sit at their true log
    value, so the chart stays honest on a log axis (an additive total/split
    cannot be drawn faithfully as adjacent stacked segments in log space).
    """
    set2 = matplotlib.colormaps["Set2"].colors
    color_harvest_only = set2[0]
    color_bdb_only = set2[1]
    color_shared = set2[2]

    y = np.arange(len(df))
    bar_h = 0.66
    inner_h = bar_h * 0.46

    def log_pos(v):
        return np.log10(max(int(v), 1))

    for i, (_, row) in enumerate(df.iterrows()):
        # HARVEST side (right, positive): only bar, then nested shared bar
        ax.barh(i, log_pos(row["harvest_only"]), height=bar_h, left=0,
                color=color_harvest_only, edgecolor="white", linewidth=0.5)
        ax.barh(i, log_pos(row["harvest_shared"]), height=inner_h, left=0,
                color=color_shared, edgecolor="white", linewidth=0.5)
        # BindingDB side (left, negative)
        ax.barh(i, -log_pos(row["bdb_only"]), height=bar_h, left=0,
                color=color_bdb_only, edgecolor="white", linewidth=0.5)
        ax.barh(i, -log_pos(row["bdb_shared"]), height=inner_h, left=0,
                color=color_shared, edgecolor="white", linewidth=0.5)

        # Count labels: the source-only value, placed past the outer tip
        if row["harvest_only"] > 0:
            tip = max(log_pos(row["harvest_only"]), log_pos(row["harvest_shared"]))
            ax.text(tip + 0.12, i, f"{int(row['harvest_only']):,}",
                    ha="left", va="center", fontsize=7.5)
        if row["bdb_only"] > 0:
            tip = max(log_pos(row["bdb_only"]), log_pos(row["bdb_shared"]))
            ax.text(-tip - 0.12, i, f"{int(row['bdb_only']):,}",
                    ha="right", va="center", fontsize=7.5)

    max_log = max(
        log_pos(df[["harvest_only", "harvest_shared"]].max().max()),
        log_pos(df[["bdb_only", "bdb_shared"]].max().max()),
    )
    limit = np.ceil(max_log) + 0.85
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.7, len(df) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.axvline(0, color="0.3", linewidth=1.0)

    ticks = np.arange(0, np.ceil(max_log) + 1)
    ax.set_xticks(np.concatenate([-ticks[::-1], ticks[1:]]))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "0" if abs(x) < 1e-9
                      else f"$10^{{{int(abs(x))}}}$"))
    ax.set_xlabel("Compounds (log scale)", fontsize=10)

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Side headers
    ax.text(-limit * 0.5, len(df) - 0.15, "BindingDB", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
    ax.text(limit * 0.5, len(df) - 0.15, "HARVEST", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

    legend_handles = [
        Patch(facecolor=color_shared, label="Shared (BindingDB & HARVEST)"),
        Patch(facecolor=color_bdb_only, label="BindingDB only"),
        Patch(facecolor=color_harvest_only, label="HARVEST only"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)


def main():
    parser = argparse.ArgumentParser(
        description="Diverging bar chart of compound distribution by ChEMBL L1")
    parser.add_argument("--data-dir", default="cluster_split/split_v14",
                        help="Directory with df_new_protein_stats.csv and "
                             "df_overlap_stats.csv")
    parser.add_argument("--chembl", default="curated_data/target_info_chembl.csv.gz",
                        help="Path to ChEMBL target info CSV")
    parser.add_argument("--output", default="manuscript/compound_l1_distribution.png",
                        help="Output PNG path")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of largest L1 classes to show individually; "
                             "the rest are collapsed into 'Others'")
    parser.add_argument("--force-others", nargs="*",
                        default=["Unclassified", "Unclassified protein",
                                 "Other cytosolic protein"],
                        help="L1 classes always collapsed into 'Others', "
                             "regardless of rank")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    df = load_and_aggregate(data_dir, args.chembl, top_n=args.top_n,
                            force_others=tuple(args.force_others))

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(df) + 2.0))
    plot_diverging(ax, df)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    # Summary table (largest first) + CSV
    out = df.iloc[::-1][[
        "bdb_total", "bdb_only", "bdb_shared",
        "harvest_total", "harvest_only", "harvest_shared",
    ]].copy()
    out.index.name = "ChEMBL L1"
    csv_path = output_path.with_name("compound_l1_distribution.csv")
    out.to_csv(csv_path)
    print(f"Saved CSV  to {csv_path}")

    print("\nCompounds by ChEMBL L1 class:")
    print(out.to_string())
    print(f"\nGrand totals — BindingDB: {int(out['bdb_total'].sum()):,}, "
          f"HARVEST: {int(out['harvest_total'].sum()):,}")


if __name__ == "__main__":
    main()
