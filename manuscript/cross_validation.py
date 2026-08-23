#!/usr/bin/env python3
"""
Cross-validation of HARVEST against BindingDB (paper Table I).

Method:
  - compound key  = InChIKey truncated to its first block (ik14, connectivity layer)
  - protein match = intersection of UniProt-accession sets (handles HARVEST complexes "P1;P2")
  - patent match  = BDB granted patent -> application number (patent_mapping.csv)
  - scope         = US patents present in BOTH databases
  - de-duplication by triplet (ik14, uniprot, patent)

Each record from one database is classified against the other on the same patent:
  Match        - compound (ik14) AND protein found
  Ligand only  - compound found, protein differs        (paper: "Target mismatch")
  Protein only - protein found, compound differs         (paper: "Compound mismatch")
  Not found    - neither compound nor protein found

Usage:
  python manuscript/cross_validation.py --harvest merged_clean.parquet --bdb full_bdb.parquet
"""
import argparse, os, sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

CATS = ['Match', 'Ligand only', 'Protein only', 'Not found']
LATEX_LABELS = ['Full match', 'Target mismatch', 'Compound mismatch', 'No overlap']


def _batch_inchi_keys(smiles_series):
    """Compute InChI Keys for a pandas Series, operating on unique values only.

    Returns a Series aligned with the input.
    """
    from chemistry_rdkit import smiles_to_inchi_key

    raw = smiles_series.fillna('')
    uniq = [s for s in raw.unique() if s]
    n = len(uniq)
    print(f"  Computing InChI Keys for {n:,} unique SMILES ...")

    ik_map = {}
    step = max(n // 10, 1)
    for i, s in enumerate(uniq):
        if i % step == 0:
            print(f"    {i:>10,} / {n:,} ({100 * i / n:4.1f}%)", end='\r', flush=True)
        ik_map[s] = smiles_to_inchi_key(s)
    print(f"    {n:>10,} / {n:,} (done)       ")
    return raw.map(ik_map).fillna('')


def load_bdb(path, mapping_path):
    b = pd.read_parquet(path, columns=['source', 'uniprot_acc', 'inchikey_clean', 'patent_number'])
    b = b[(b['source'] == 'bindingdb') & b['patent_number'].fillna('').str.match(r'^US\d')]
    p2app = dict(zip(*[pd.read_csv(mapping_path)[c] for c in ('patent_number', 'application_number')]))
    b = b.rename(columns={'inchikey_clean': 'inchi_key_clean', 'uniprot_acc': 'uniprot',
                          'patent_number': 'bdb_patent'})
    b['h_patent'] = b['bdb_patent'].map(p2app)
    miss = b['h_patent'].isna()
    b.loc[miss, 'h_patent'] = b.loc[miss, 'bdb_patent'] + 'A1'
    b = b[b['inchi_key_clean'].fillna('').str.len() > 0]
    b = b[b['uniprot'].fillna('').str.len() > 0]
    b['ik14'] = b['inchi_key_clean'].str.split('-').str[0].fillna('')
    return b


def load_harvest(path):
    """Load HARVEST parquet, using clean_inchi_key if present, else computing from clean_smiles."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    available = set(schema.names)

    if 'clean_inchi_key' in available:
        print("  Using existing 'clean_inchi_key' column")
        cols = ['patent_number', 'Target accession', 'clean_inchi_key']
        h = pd.read_parquet(path, columns=cols)
        h = h.rename(columns={'patent_number': 'h_patent'})
        h['inchi_key_clean'] = h['clean_inchi_key'].fillna('')
    elif 'clean_smiles' in available:
        print("  Column 'clean_inchi_key' not found, computing from 'clean_smiles'")
        cols = ['patent_number', 'Target accession', 'clean_smiles']
        h = pd.read_parquet(path, columns=cols)
        h = h.rename(columns={'patent_number': 'h_patent'})
        h['inchi_key_clean'] = _batch_inchi_keys(h['clean_smiles'])
    else:
        raise ValueError(f"Parquet has neither 'clean_inchi_key' nor 'clean_smiles': {path}")

    h['uniprot'] = h['Target accession'].fillna('').str.split(';').str[0].str.strip()
    h['acc_set'] = h['Target accession'].fillna('').str.split(';').apply(
        lambda x: frozenset(a.strip() for a in x if a.strip()))
    h['ik14'] = h['inchi_key_clean'].str.split('-').str[0].fillna('')
    return h[(h['ik14'].str.len() > 0) & (h['acc_set'].apply(len) > 0)]


def classify(src, tgt, src_has_acc_set):
    """Return (Match, Ligand only, Protein only, Not found) for src records against tgt."""
    key = 'ik14'
    if src_has_acc_set:                 # src = HARVEST (acc_set), tgt = BDB (single uniprot)
        by_ik_pat, by_pat = {}, {}
        for ik, up, pat in zip(tgt[key], tgt['uniprot'], tgt['h_patent']):
            by_ik_pat.setdefault((ik, pat), set()).add(up)
            by_pat.setdefault(pat, set()).add(up)
        m = l = pr = nf = 0
        for ik, pat, acc in zip(src[key], src['h_patent'], src['acc_set']):
            ups_c = by_ik_pat.get((ik, pat), set())
            if ups_c & acc:                 m += 1
            elif ups_c:                     l += 1
            elif by_pat.get(pat, set()) & acc: pr += 1
            else:                           nf += 1
    else:                               # src = BDB (single uniprot), tgt = HARVEST (acc_set)
        by_ik_pat, by_pat = {}, {}
        for ik, pat, acc in zip(tgt[key], tgt['h_patent'], tgt['acc_set']):
            by_ik_pat.setdefault((ik, pat), []).append(acc)
            by_pat.setdefault(pat, set()).update(acc)
        m = l = pr = nf = 0
        for ik, up, pat in zip(src[key], src['uniprot'], src['h_patent']):
            accs = by_ik_pat.get((ik, pat), [])
            if any(up in a for a in accs):   m += 1
            elif accs:                       l += 1
            elif up in by_pat.get(pat, set()): pr += 1
            else:                            nf += 1
    return m, l, pr, nf


def report(name, counts):
    n = sum(counts)
    print(f"\n  {name} (n={n:,})")
    for cat, c in zip(CATS, counts):
        print(f"    {cat:13} {c:>10,} ({100*c/n:5.1f}%)")


def _latex_num(n: int) -> str:
    """Format an integer with LaTeX thousand separators: 486{,}950."""
    s = f"{n:,}"
    return s.replace(",", "{,}")


def _latex_pct(c: int, n: int) -> str:
    """Format count and percentage for a LaTeX table cell."""
    pct = f"{100 * c / n:.1f}"
    num = _latex_num(c)
    if float(pct) < 10:
        return f"{num} \\ ({pct}\\%)"
    return f"{num} ({pct}\\%)"


def latex_table(h_counts, b_counts):
    """Print the LaTeX cross-validation table."""
    nh = sum(h_counts)
    nb = sum(b_counts)
    lines = [
        r"\begin{tabular*}{\columnwidth}{l@{\extracolsep{\fill}}rr}",
        r"\toprule",
        r"  & \makecell{\textbf{HARVEST}\\\textbf{($n{=}" + _latex_num(nh)
            + r"$)}\\\textbf{in BindingDB}}",
        r"  & \makecell{\textbf{BindingDB}\\\textbf{($n{=}" + _latex_num(nb)
            + r"$)}\\\textbf{in HARVEST}} \\",
        r"\midrule",
    ]
    for label, hc, bc in zip(LATEX_LABELS, h_counts, b_counts):
        h_cell = _latex_pct(hc, nh)
        b_cell = _latex_pct(bc, nb)
        lines.append(f"{label:19s} & {h_cell} & {b_cell} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    print("\n".join(lines))


def _default_mapping():
    """Return path to curated_data/patent_mapping.csv relative to this script's repo root."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'curated_data', 'patent_mapping.csv')


def main():
    ap = argparse.ArgumentParser(description="HARVEST vs BindingDB cross-validation (paper Table I)")
    ap.add_argument('--harvest', required=True, help="HARVEST dataset parquet")
    ap.add_argument('--bdb', required=True,
                    help="BindingDB parquet with columns source/uniprot_acc/inchikey_clean/patent_number")
    ap.add_argument('--mapping', default=None, help="patent_number,application_number CSV "
                    "(default: curated_data/patent_mapping.csv in repo)")
    a = ap.parse_args()
    if a.mapping is None:
        a.mapping = _default_mapping()

    print("Loading BindingDB reference ...")
    bdb = load_bdb(a.bdb, a.mapping)

    print(f"\nLoading HARVEST: {a.harvest}")
    h = load_harvest(a.harvest)
    shared = set(h['h_patent'].unique()) & set(bdb['h_patent'].unique())
    print(f"  shared patents: {len(shared):,}")
    h = h[h['h_patent'].isin(shared)].drop_duplicates(subset=['ik14', 'uniprot', 'h_patent'])
    b = bdb[bdb['h_patent'].isin(shared)].drop_duplicates(subset=['ik14', 'uniprot', 'h_patent'])

    h_counts = classify(h, b, True)
    b_counts = classify(b, h, False)
    report("HARVEST -> BindingDB", h_counts)
    report("BindingDB -> HARVEST", b_counts)
    print(f"\n--- LaTeX table ---\n")
    latex_table(h_counts, b_counts)


if __name__ == '__main__':
    main()
