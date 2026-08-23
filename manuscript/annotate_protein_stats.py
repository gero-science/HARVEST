#!/usr/bin/env python3
"""Annotate the per-protein split-stats CSVs with ChEMBL + UniProt info.

Joins `df_new_protein_stats.csv` (HARVEST-only proteins) and
`df_overlap_stats.csv` (proteins shared with BindingDB) into a single
harmonized table, then adds:
    * ChEMBL annotation: l1 / l2 / l3 target class, gene symbol, protein name
    * UniProt Cellular Component (GO_C), joined with "; " when multiple

UniProt fields are fetched in batches from the public REST API
(`https://rest.uniprot.org/uniprotkb/accessions`) and cached to a local
JSON file so re-runs do not re-hit the network.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


GO_ID_RE = re.compile(r"\s*\[GO:\d+\]")


def strip_go_ids(value: str) -> str:
    """Drop the ' [GO:NNNNNNN]' suffix from each '; '-joined term."""
    if not value:
        return ""
    return "; ".join(GO_ID_RE.sub("", t).strip() for t in value.split(";") if t.strip())


def classify_major(cellular_component: str) -> str:
    """Bucket a protein into Secretory / Intracellular / Membrane-Bound.

    Heuristic on the GO Cellular Component string:
      * any 'membrane' term  -> Membrane-Bound
      * else any 'extracellular' / 'secreted' term -> Secretory
      * else                                       -> Intracellular
    Empty annotation -> "" (unknown).
    """
    if not cellular_component:
        return ""
    s = cellular_component.lower()
    if "membrane" in s:
        return "Membrane-Bound"
    if "extracellular" in s or "secreted" in s:
        return "Secretory"
    return "Intracellular"


UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"
UNIPROT_FIELDS = "accession,go_c"
UNIPROT_BATCH = 100


def harmonize(data_dir: Path) -> pd.DataFrame:
    """Concatenate the two per-protein CSVs into one harmonized table.

    HARVEST-only rows are expanded so they share the same column schema as
    the overlap rows (BindingDB / overlap counts filled with 0).
    """
    df_new = pd.read_csv(data_dir / "df_new_protein_stats.csv")
    df_overlap = pd.read_csv(data_dir / "df_overlap_stats.csv")

    new = pd.DataFrame({
        "protein": df_new["protein"],
        "status": "harvest_only",
        "n_harvest_compounds": df_new["n_compounds"],
        "n_harvest_only_compounds": df_new["n_compounds"],
        "n_harvest_overlap_compounds": 0,
        "n_bdb_compounds": 0,
        "n_bdb_only_compounds": 0,
        "n_bdb_overlap_compounds": 0,
        "n_compound_overlap": 0,
        "n_total_clusters": df_new["n_clusters"],
        "n_harvest_only_clusters": df_new["n_clusters"],
        "n_bdb_only_clusters": 0,
        "n_shared_clusters": 0,
    })
    overlap = df_overlap.copy()
    overlap.insert(1, "status", "shared")

    cols = new.columns.tolist()
    return pd.concat([new[cols], overlap[cols]], ignore_index=True)


def add_chembl(df: pd.DataFrame, chembl_path: Path) -> pd.DataFrame:
    """Attach ChEMBL L1/L2/L3, gene symbol, and preferred protein name."""
    chembl = pd.read_csv(chembl_path)
    chembl = chembl.drop_duplicates("accession")[[
        "accession", "gene_symbol", "protein_pref_name", "l1", "l2", "l3",
    ]].rename(columns={
        "accession": "protein",
        "protein_pref_name": "protein_name",
        "l1": "chembl_l1",
        "l2": "chembl_l2",
        "l3": "chembl_l3",
    })
    return df.merge(chembl, on="protein", how="left")


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with cache_path.open() as f:
            return json.load(f)
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(cache, f, indent=0, sort_keys=True)


def _fetch_batch(accessions: list[str]) -> dict[str, str]:
    """One request to UniProt; returns {accession: cellular_component_str}."""
    params = {
        "accessions": ",".join(accessions),
        "fields": UNIPROT_FIELDS,
        "format": "tsv",
    }
    resp = requests.get(UNIPROT_ACCESSIONS_URL, params=params, timeout=60)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    if not lines:
        return {}
    out: dict[str, str] = {}
    # TSV: header line "Entry\tGene Ontology (cellular component)"
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            acc = parts[0] if parts else ""
            if acc:
                out[acc] = ""
            continue
        acc, raw = parts[0], parts[1]
        terms = [t.strip() for t in raw.split(";") if t.strip()] if raw else []
        out[acc] = "; ".join(terms)
    return out


def fetch_uniprot_cellular_component(
    accessions: list[str], cache_path: Path,
    batch_size: int = UNIPROT_BATCH, sleep: float = 0.2,
) -> dict[str, str]:
    """Fetch GO Cellular Component for each accession, with on-disk caching."""
    cache = _load_cache(cache_path)
    missing = sorted({a for a in accessions if a and a not in cache})
    if missing:
        print(f"Fetching {len(missing)} UniProt entries "
              f"({(len(missing) + batch_size - 1) // batch_size} batches)...")
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            fetched = _fetch_batch(batch)
            # Mark every requested accession (use "" for ones UniProt skipped,
            # e.g. obsolete / demerged entries) so we don't retry forever.
            for acc in batch:
                cache[acc] = fetched.get(acc, "")
            _save_cache(cache_path, cache)
            print(f"  {min(i + batch_size, len(missing))}/{len(missing)}")
            time.sleep(sleep)
    return cache


def main():
    parser = argparse.ArgumentParser(
        description="Annotate per-protein split-stats with ChEMBL + UniProt")
    parser.add_argument("--data-dir", default="cluster_split/split_v14",
                        help="Directory with df_new_protein_stats.csv and "
                             "df_overlap_stats.csv")
    parser.add_argument("--chembl", default="curated_data/target_info_chembl.csv.gz",
                        help="Path to ChEMBL target info CSV")
    parser.add_argument("--output", default="manuscript/protein_stats_annotated.csv",
                        help="Output CSV path")
    parser.add_argument("--uniprot-cache",
                        default="manuscript/uniprot_cellular_component_cache.json",
                        help="JSON cache for UniProt GO Cellular Component lookups")
    parser.add_argument("--batch-size", type=int, default=UNIPROT_BATCH,
                        help="UniProt accessions per REST request")
    parser.add_argument("--skip-uniprot", action="store_true",
                        help="Skip UniProt fetch (e.g. offline); leaves the "
                             "cellular_component column blank")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = harmonize(data_dir)
    print(f"Harmonized: {len(df)} proteins "
          f"({(df['status'] == 'shared').sum()} shared, "
          f"{(df['status'] == 'harvest_only').sum()} HARVEST-only)")

    df = add_chembl(df, Path(args.chembl))
    matched = df["chembl_l1"].notna().sum()
    print(f"ChEMBL annotated: {matched}/{len(df)}")

    if args.skip_uniprot:
        df["cellular_component"] = ""
    else:
        cache = fetch_uniprot_cellular_component(
            df["protein"].tolist(),
            Path(args.uniprot_cache),
            batch_size=args.batch_size,
        )
        # Cache keeps the raw '... [GO:NNNNNNN]' strings for traceability;
        # the exported column drops the IDs.
        df["cellular_component"] = (
            df["protein"].map(cache).fillna("").map(strip_go_ids)
        )
        with_cc = (df["cellular_component"] != "").sum()
        print(f"UniProt cellular component: {with_cc}/{len(df)} non-empty")

    df["major_class"] = df["cellular_component"].map(classify_major)
    counts = df["major_class"].value_counts(dropna=False)
    print("Major class breakdown:")
    for k, v in counts.items():
        print(f"  {k or '(unknown)'}: {v}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved annotated table to {output_path}")


if __name__ == "__main__":
    main()
