#!/usr/bin/env python3
"""
HARVEST quality figures — single self-contained public-release script.

Reproduces, from one hand-curated reference table + the HARVEST/BindingDB datasets:
  fig_quality.png        (6 panels): (a) molecular weight, (b) affinity, (c) synthetic
                         accessibility [HARVEST vs BindingDB]; (d,e) activity-value
                         residuals vs BindingDB (US patents / articles, broken y-axis);
                         (f) extraction fidelity vs the manual reference (by field).
  fig_fidelity_supp.png  (2 panels): (a) compound+target fidelity inside vs outside
                         BindingDB; (b) fidelity by reference datapoints per patent.
  per_patent_fidelity.csv  per-patent recall/precision (intermediate, written on the way).

It first SCORES the reference against HARVEST and BindingDB (compound / UniProt target /
compound+target / +value-within-2x, per-patent recall and per-record precision), then draws
-- by default the value criterion compares all reference values; set env HARVEST_VALUE_EQ_ONLY=1
to restrict it to exact '=' reference rows (inequalities/ranges excluded from num+denom) --
the figures. Matching key ("config C"): TARGET = UniProt entry-name (per-patent, ';'-complex
expansion); COMPOUND = InChIKey connectivity[:14], row-level OR (gold over its 4 arms, HARVEST
InChIKey(clean_smiles), BindingDB inchi_cut).

Inputs (set $HARVEST_DATA_DIR; defaults to the repo root): final_v16_clean.parquet,
full_bdb_chembl_fix.parquet. Patent map: curated_data/patent_mapping.csv.
Reference: curated_data/manual_reference.csv.
Molecular weight / synthetic accessibility are cached to parquet on first run.

Run:  python make_figures.py
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
import cycler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

# ----------------------------------------------------------------------------- paths / constants
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA = os.environ.get('HARVEST_DATA_DIR', REPO_ROOT)
REFERENCE = os.environ.get('HARVEST_REFERENCE', f'{REPO_ROOT}/curated_data/manual_reference.csv')
HARV = os.environ.get('HARVEST_PARQUET', f'{DATA}/final_v16_clean.parquet')
BDB = os.environ.get('HARVEST_BDB_PARQUET', f'{DATA}/full_bdb_chembl_fix.parquet')
PMAP = os.environ.get('HARVEST_PATENT_MAP', f'{REPO_ROOT}/curated_data/patent_mapping.csv')
IKCACHE = f'{DATA}/harvest_inchikey_clean_cache.parquet'     # optional (clean_smiles -> InChIKey); regenerable
FIGS = os.environ.get('HARVEST_FIG_DIR', f'{HERE}/figs')
PROPCACHE = f'{HERE}/quality_fig_props_cache.parquet'         # smiles -> mw, sa (built on first run)
PERPATENT_OUT = os.environ.get('HARVEST_PERPATENT_OUT', f'{HERE}/per_patent_fidelity.csv')
VALUE_EQ_ONLY = os.environ.get('HARVEST_VALUE_EQ_ONLY', '0') == '1'   # opt-in: restrict value criterion to exact '=' reference rows (off by default; the value bar then compares all values)
os.makedirs(FIGS, exist_ok=True)

MW_H, MW_B = '#1f77b4', '#ff7f0e'   # MW / SA lines (HARVEST / BindingDB)
AF_H, AF_B = '#0066ff', '#ff4500'   # affinity fill
RES_C = '#4878CF'                    # residual histograms
C_R, C_P = '#4878CF', '#8172B3'     # recall / precision
WH, WR = 0.854, 0.146               # corpus strata weights (HARVEST-only / BDB-overlap)
EMPTIES = ['US20120077856A1', 'US20150072995A1', 'US20160256437A1', 'US20180318310A1',
           'US20180369249A1', 'US20220356257A1', 'US20230102192A1']  # no-PLI patents (over-extraction; precision 0)
CONNCOLS = ['inchikey_connectivity', 'clean_conn_iupac', 'clean_conn_mol', 'clean_conn_cdx']  # gold compound arms
FAM = {'IC50 (nM)': 'IC50', 'EC50 (nM)': 'EC50', 'Ki (nM)': 'KI', 'Kd (nM)': 'KD'}


def set_style():
    matplotlib.rcdefaults()
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': 'arial',
        'axes.facecolor': '#ffffff', 'axes.edgecolor': 'k', 'axes.linewidth': 1.0,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'axes.labelcolor': 'k', 'xtick.color': 'k', 'ytick.color': 'k',
        'xtick.major.width': 1.4, 'ytick.major.width': 1.4, 'lines.linewidth': 2.0,
        'figure.facecolor': '#ffffff', 'savefig.dpi': 300, 'legend.frameon': False})
    plt.rc('axes', prop_cycle=cycler.cycler('color', matplotlib.cm.tab10.colors))


# ----------------------------------------------------------------------------- load HARVEST/BindingDB ONCE
def load_all():
    dfh = pd.read_parquet(HARV, columns=['patent_number', 'UniProt ID', 'clean_smiles', 'relation',
                                         'Target accession', 'IC50 (nM)', 'Ki (nM)', 'Kd (nM)', 'EC50 (nM)'])
    dfb = pd.read_parquet(BDB, columns=['source', 'patent_number', 'smiles', 'inchi_cut', 'uniprot_id',
                                        'uniprot_acc', 'inchikey_clean', 'activity_type', 'activity_value',
                                        'activity_relation'])
    dfb = dfb[dfb['source'] == 'bindingdb'].copy()
    dfb['is_us_patent'] = dfb['patent_number'].fillna('').astype(str).str.startswith('US')
    return dfh, dfb


def cut(ik):
    return str(ik).split('-')[0][:14] if ik and str(ik) not in ('nan', 'None') else None


def clean_cut(smi, _c={}):
    if not smi or pd.isna(smi):
        return None
    if smi in _c:
        return _c[smi]
    try:
        _c[smi] = cut(Chem.MolToInchiKey(Chem.MolFromSmiles(smi)))
    except Exception:
        _c[smi] = None
    return _c[smi]


def expand(series):
    """Set of upper-cased UniProt entry-names, splitting ';'-joined complexes."""
    s = set()
    for v in series.dropna():
        for x in str(v).split(';'):
            x = x.strip().upper()
            if x and x != 'NAN':
                s.add(x)
    return s


def row_conns(r):
    vs = set()
    for c in CONNCOLS:
        v = cut(r.get(c))
        if v:
            vs.add(v)
    return vs


_RANGE_RE = re.compile(r'\d\s*(?:-|to)\s*\d')


def is_exact(rep):
    """True iff value_reported is a single point value (no operator / range) — used to
    restrict the value comparison to '=' measurements, mirroring the residual panels."""
    if not isinstance(rep, str):
        return False
    r = rep.strip()
    if any(t in r for t in ('<', '>', '≤', '≥', '~', '–', '—')):
        return False
    if _RANGE_RE.search(r):
        return False
    return bool(re.match(r'^[\d.]+', r))


# ----------------------------------------------------------------------------- fidelity scoring
def _build_system_index(rows_iter):
    """Build compound/target/value lookup structures from a system's rows.

    Each row provides (connectivity_set, uniprot_entries_set) and optional
    activity values keyed by (metric, connectivity, entry).  Values carry an
    ``is_eq`` flag so we can also build an eq-only value index for the
    VALUE_EQ_ONLY precision denominator.

    Returns:
        targets:          set of UniProt entry names
        compounds:        set of InChI connectivity keys
        compound_to_entries: {conn: set(entries)} for pair matching
        pairs:            set of (conn, entry)
        value_index:      {(metric, conn, entry): [float]} — all values
        value_eq_index:   {(metric, conn, entry): [float]} — only '=' values
        records:          [(conns, entries), ...] for precision scoring
    """
    targets = set()
    compounds = set()
    compound_to_entries = {}
    pairs = set()
    value_index = {}
    value_eq_index = {}
    records = []
    for conns, entries, values in rows_iter:
        records.append((conns, entries))
        targets |= entries
        compounds |= conns
        for conn in conns:
            compound_to_entries.setdefault(conn, set()).update(entries)
            for entry in entries:
                pairs.add((conn, entry))
        for metric_key, val, val_is_eq in values:
            for conn in conns:
                for entry in entries:
                    value_index.setdefault((metric_key, conn, entry), []).append(val)
                    if val_is_eq:
                        value_eq_index.setdefault((metric_key, conn, entry), []).append(val)
    return targets, compounds, compound_to_entries, pairs, value_index, value_eq_index, records


def _precision_pairs(records, gold_pair_set):
    """Precision: fraction of system rows whose (compound, target) pair appears in the gold set."""
    total = hits = 0
    for conns, entries in records:
        if not conns or not entries:
            continue
        total += 1
        if any((c, e) in gold_pair_set for c in conns for e in entries):
            hits += 1
    return hits / total if total else np.nan


def _precision_field(records, gold_set, use_entries=False):
    """Precision: fraction of system rows whose compound (or target) appears in the gold set."""
    total = hits = 0
    for conns, entries in records:
        items = entries if use_entries else conns
        if not items:
            continue
        total += 1
        if any(item in gold_set for item in items):
            hits += 1
    return hits / total if total else np.nan


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def _value_within_2x(candidate, gold):
    """True if candidate and gold are both positive and within a 2× ratio."""
    return candidate > 0 and gold > 0 and 0.5 <= candidate / gold <= 2


def compute_fidelity(dfh, dfb):
    """Score the manual reference against HARVEST and BindingDB per patent.

    Recall  = fraction of gold-reference data points found in the system.
    Precision = fraction of system rows that match a gold-reference data point.

    Matching key:
      COMPOUND = InChI Key connectivity[:14], row-level OR over the gold's
                 4 arms (inchikey_connectivity, clean_conn_iupac/mol/cdx).
      TARGET   = UniProt entry-name (upper-cased), ';'-complex expansion.
      VALUE    = within 2× ratio for the same (metric, compound, target).

    Writes per_patent_fidelity.csv and returns the DataFrame.
    """
    # --- Load gold reference ---
    sep = '\t' if str(REFERENCE).endswith('.tsv') else ','
    gold = pd.read_csv(REFERENCE, sep=sep, dtype=str)
    patcol = 'patent_number' if 'patent_number' in gold.columns else 'patent'
    gold = gold.rename(columns={patcol: 'patent_number'})
    gold['uniprot_entry'] = gold['uniprot_id'].str.upper()
    gold['value_numeric'] = pd.to_numeric(gold['value_nM'], errors='coerce')
    gold['metric_upper'] = gold['metric'].astype(str).str.upper().str.strip()
    in_bdb_map = (gold.groupby('patent_number')['in_bindingdb'].first().to_dict()
                  if 'in_bindingdb' in gold.columns else {})
    patents = sorted(gold['patent_number'].dropna().unique())

    # --- Prepare HARVEST rows for matching patents ---
    harvest = dfh[dfh['patent_number'].isin(patents)].copy()
    harvest['uniprot_entries'] = harvest['UniProt ID'].fillna('').apply(
        lambda s: set(x.strip().upper() for x in s.split(';') if x.strip()))
    harvest['connectivity_set'] = [
        set(filter(None, [clean_cut(smi)])) for smi in harvest['clean_smiles']
    ]

    # --- Prepare BDB lookup (application → granted patent mapping) ---
    patent_map = pd.read_csv(PMAP, dtype=str)
    app_to_granted = {}
    for row in patent_map.itertuples():
        app_to_granted.setdefault(str(row.application_number), set()).add(str(row.patent_number))
    bdb_by_patent = {}
    for row in dfb.itertuples(index=False):
        bdb_by_patent.setdefault(str(row.patent_number), []).append(row)

    def get_bdb_rows(patent_app):
        """Look up BDB rows by application number, trying all number forms."""
        forms = {patent_app, patent_app + 'A1', patent_app.rstrip('A1')}
        forms |= app_to_granted.get(patent_app, set())
        rows = []
        for form in forms:
            rows += bdb_by_patent.get(form, [])
        return rows

    # --- Score each patent ---
    patent_records = []
    skipped_absent = []
    for patent in patents:
        gold_patent = gold[gold['patent_number'] == patent]
        harvest_patent = harvest[harvest['patent_number'] == patent]
        # Only score patents that HARVEST actually extracted. A reference patent with
        # zero HARVEST rows means this build didn't cover it (scope/coverage change) —
        # excluding it keeps fidelity a measure of extraction QUALITY, not coverage,
        # so the reference need not be re-edited whenever HARVEST's scope changes.
        if len(harvest_patent) == 0:
            skipped_absent.append(patent)
            continue

        # Build HARVEST index for this patent
        def harvest_rows_iter():
            for _, row in harvest_patent.iterrows():
                row_is_eq = str(row.get('relation', '')).strip() == '='
                values = []
                for parquet_col, metric_key in FAM.items():
                    val = pd.to_numeric(row[parquet_col], errors='coerce')
                    if pd.notna(val):
                        values.append((metric_key, float(val), row_is_eq))
                yield row['connectivity_set'], row['uniprot_entries'], values

        (harv_targets, harv_compounds, harv_compound_to_entries,
         harv_pairs, harv_value_index, harv_value_eq_index, harv_records) = _build_system_index(harvest_rows_iter())

        # Build BDB index for this patent
        def bdb_rows_iter():
            for bdb_row in get_bdb_rows(patent):
                conns = set(filter(None, [cut(bdb_row.inchi_cut)]))
                entries = expand(pd.Series([bdb_row.uniprot_id]))
                val = pd.to_numeric(bdb_row.activity_value, errors='coerce')
                metric_key = str(bdb_row.activity_type).upper().strip()
                bdb_is_eq = str(getattr(bdb_row, 'activity_relation', '')).strip() == '='
                values = [(metric_key, float(val), bdb_is_eq)] if pd.notna(val) else []
                yield conns, entries, values

        (bdb_targets, bdb_compounds, bdb_compound_to_entries,
         bdb_pairs, bdb_value_index, bdb_value_eq_index, bdb_records) = _build_system_index(bdb_rows_iter())

        # --- Recall: iterate gold rows, check if found in each system ---
        recall_counts = {
            sys_name: dict(target_n=0, target_hit=0,
                           compound_n=0, compound_hit=0,
                           pair_n=0, pair_hit=0,
                           value_n=0, value_hit=0)
            for sys_name in ('harvest', 'bdb')
        }
        gold_pair_set = set()          # for precision: all gold (compound, target) pairs
        gold_compound_set = set()      # for precision: all gold compound connectivities
        gold_target_set = expand(gold_patent['uniprot_entry'])
        gold_value_index = {}          # for precision: {(metric, conn, entry): [float]}

        for _, gold_row in gold_patent.iterrows():
            gold_entries = expand(pd.Series([gold_row['uniprot_entry']]))
            gold_conns = row_conns(gold_row)
            gold_val = gold_row['value_numeric']
            gold_metric = gold_row['metric_upper']
            has_any_value = pd.notna(gold_val) and gold_metric and gold_metric != 'NAN'
            has_value = has_any_value and (not VALUE_EQ_ONLY or is_exact(gold_row.get('value_reported')))

            # Accumulate gold sets for precision scoring
            if gold_conns:
                gold_compound_set |= gold_conns
                for conn in gold_conns:
                    for entry in gold_entries:
                        gold_pair_set.add((conn, entry))
                if has_any_value:   # precision target = ALL gold values
                    for conn in gold_conns:
                        for entry in gold_entries:
                            gold_value_index.setdefault(
                                (gold_metric, conn, entry), []).append(gold_val)

            # Score recall against each system
            for sys_name, sys_targets, sys_compounds, sys_c2e, sys_value_idx in [
                ('harvest', harv_targets, harv_compounds, harv_compound_to_entries, harv_value_index),
                ('bdb', bdb_targets, bdb_compounds, bdb_compound_to_entries, bdb_value_index),
            ]:
                counts = recall_counts[sys_name]

                # Target recall
                if gold_entries:
                    counts['target_n'] += 1
                    counts['target_hit'] += bool(gold_entries & sys_targets)

                if not gold_conns:
                    continue

                # Compound recall
                compound_found = bool(gold_conns & sys_compounds)
                counts['compound_n'] += 1
                counts['compound_hit'] += compound_found

                # Pair recall: compound found AND its target matches
                matched_entries = set()
                for conn in gold_conns:
                    matched_entries |= sys_c2e.get(conn, set())
                pair_found = compound_found and bool(gold_entries & matched_entries)
                counts['pair_n'] += 1
                counts['pair_hit'] += pair_found

                # Value recall: pair found AND value within 2×
                if has_value:
                    counts['value_n'] += 1
                    if pair_found:
                        candidates = [
                            v for conn in gold_conns for entry in gold_entries
                            for v in sys_value_idx.get((gold_metric, conn, entry), [])
                        ]
                        if any(_value_within_2x(c, gold_val) for c in candidates):
                            counts['value_hit'] += 1

        # --- Precision (value): iterate system rows, check if found in gold ---
        # In eq-mode, restrict precision denominator to '=' values (symmetric with exact-only gold)
        precision_value_counts = {'harvest': dict(total=0, hit=0), 'bdb': dict(total=0, hit=0)}
        if VALUE_EQ_ONLY:
            prec_sources = [('harvest', harv_value_eq_index), ('bdb', bdb_value_eq_index)]
        else:
            prec_sources = [('harvest', harv_value_index), ('bdb', bdb_value_index)]
        for sys_name, sys_value_idx in prec_sources:
            for key, sys_values in sys_value_idx.items():
                gold_values = gold_value_index.get(key, [])
                for sys_val in sys_values:
                    precision_value_counts[sys_name]['total'] += 1
                    if any(_value_within_2x(sys_val, gv) for gv in gold_values):
                        precision_value_counts[sys_name]['hit'] += 1

        # --- Assemble per-patent record ---
        rc = recall_counts
        pvc = precision_value_counts
        patent_records.append(dict(
            patent=patent,
            in_bindingdb=in_bdb_map.get(patent, ''),
            n_reference_datapoints=len(gold_patent),
            # HARVEST recall (fraction of gold found in HARVEST)
            harvest_recall_target=_safe_ratio(rc['harvest']['target_hit'], rc['harvest']['target_n']),
            harvest_recall_compound=_safe_ratio(rc['harvest']['compound_hit'], rc['harvest']['compound_n']),
            harvest_recall_pair=_safe_ratio(rc['harvest']['pair_hit'], rc['harvest']['pair_n']),
            harvest_recall_value=_safe_ratio(rc['harvest']['value_hit'], rc['harvest']['value_n']),
            # HARVEST precision (fraction of HARVEST rows matching gold)
            harvest_precision_target=_precision_field(harv_records, gold_target_set, use_entries=True),
            harvest_precision_compound=_precision_field(harv_records, gold_compound_set),
            harvest_precision_pair=_precision_pairs(harv_records, gold_pair_set),
            harvest_precision_value=_safe_ratio(pvc['harvest']['hit'], pvc['harvest']['total']),
            # BDB recall
            bdb_recall_target=_safe_ratio(rc['bdb']['target_hit'], rc['bdb']['target_n']),
            bdb_recall_compound=_safe_ratio(rc['bdb']['compound_hit'], rc['bdb']['compound_n']),
            bdb_recall_pair=_safe_ratio(rc['bdb']['pair_hit'], rc['bdb']['pair_n']),
            bdb_recall_value=_safe_ratio(rc['bdb']['value_hit'], rc['bdb']['value_n']),
            # BDB precision
            bdb_precision_target=_precision_field(bdb_records, gold_target_set, use_entries=True),
            bdb_precision_compound=_precision_field(bdb_records, gold_compound_set),
            bdb_precision_pair=_precision_pairs(bdb_records, gold_pair_set),
            bdb_precision_value=_safe_ratio(pvc['bdb']['hit'], pvc['bdb']['total']),
        ))

    per_patent = pd.DataFrame(patent_records)
    per_patent.to_csv(PERPATENT_OUT, index=False)
    msg = f'  scored {len(per_patent)} patents -> {PERPATENT_OUT}'
    if skipped_absent:
        msg += f'  (skipped {len(skipped_absent)} absent from HARVEST: {", ".join(skipped_absent)})'
    print(msg)
    return per_patent


# ----------------------------------------------------------------------------- properties (a, c)
def property_cache(unique_smiles):
    import sys
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    have = pd.read_parquet(PROPCACHE) if os.path.exists(PROPCACHE) else pd.DataFrame(columns=['smiles', 'mw', 'sa'])
    known = set(have['smiles'])
    todo = [s for s in unique_smiles if s not in known]
    if todo:
        print(f'  computing MW+SA for {len(todo):,} new SMILES (cached: {len(have):,}) ...')
        rows = []
        for s in todo:
            try:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    rows.append((s, Descriptors.MolWt(m), sascorer.calculateScore(m)))
            except Exception:
                continue
        have = pd.concat([have, pd.DataFrame(rows, columns=['smiles', 'mw', 'sa'])], ignore_index=True)
        have.to_parquet(PROPCACHE)
    return dict(zip(have['smiles'], have['mw'])), dict(zip(have['smiles'], have['sa']))


def property_series(dfh, dfb):
    h = dfh['clean_smiles'].dropna().unique().tolist()
    b = dfb.loc[dfb['is_us_patent'], 'smiles'].dropna().unique().tolist()
    mw, sa = property_cache(list(set(h) | set(b)))
    ser = lambda smis, d: pd.Series([d.get(s) for s in smis]).dropna()
    return ser(h, mw), ser(b, mw), ser(h, sa), ser(b, sa)


# ----------------------------------------------------------------------------- affinity (b)
def affinity(dfh, dfb):
    h = dfh[dfh['relation'] == '=']
    hv = []
    for c in ['IC50 (nM)', 'Ki (nM)', 'Kd (nM)']:
        v = pd.to_numeric(h[c], errors='coerce').dropna()
        hv.append(v[(v >= 1e-6) & (v <= 1e8)])
    pa_h = -np.log10(pd.concat(hv) * 1e-9)
    b = dfb[dfb['is_us_patent'] & (dfb['activity_relation'] == '=')]
    bv = []
    for at in ['IC50', 'Ki', 'Kd', 'EC50']:
        v = pd.to_numeric(b[b['activity_type'] == at]['activity_value'], errors='coerce').dropna()
        bv.append(v[(v >= 1e-6) & (v <= 1e8)])
    pa_b = -np.log10(pd.concat(bv) * 1e-9)
    return pa_h, pa_b


# ----------------------------------------------------------------------------- residuals (d, e)
def _ik14(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToInchiKey(m).split('-')[0] if m else None
    except Exception:
        return None


def residuals(dfh, dfb):
    """ΔpActivity (BindingDB - HARVEST) for matched exact+deduped pairs, split US-patents / articles.
    Match key = (first UniProt accession, clean-InChIKey connectivity[:14])."""
    h = dfh.copy()

    def norm(v):
        try:
            s = str(v)
            return float(s[1:] if s[:1] in '<>' else s)
        except Exception:
            return np.nan
    for c in ['IC50 (nM)', 'Ki (nM)', 'Kd (nM)', 'EC50 (nM)']:
        h[c] = h[c].map(norm)
    h['norm_activity'] = h[['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']].bfill(axis=1).iloc[:, 0]
    h['acc'] = h['Target accession'].fillna('').str.split(';').str[0]
    ikmap = {}
    if os.path.exists(IKCACHE):
        hc = pd.read_parquet(IKCACHE)
        ikmap = {s: (str(k).split('-')[0] if isinstance(k, str) and k else None)
                 for s, k in zip(hc['clean_smiles'], hc['inchi_key_clean'])}
    for s in h['clean_smiles'].dropna().unique():
        if s not in ikmap:
            ikmap[s] = _ik14(s)
    h['ik'] = h['clean_smiles'].map(ikmap)
    h = h[(h['relation'] == '=')].dropna(subset=['acc', 'ik', 'norm_activity'])
    h = h[h['acc'] != '']
    hh = h.sort_values('norm_activity').drop_duplicates(['acc', 'ik'])[['acc', 'ik', 'norm_activity']]

    b = dfb[dfb['activity_relation'] == '='].copy()
    b['acc'] = b['uniprot_acc']
    b['ik'] = b['inchikey_clean'].fillna('').str.split('-').str[0]
    b = b.dropna(subset=['acc', 'ik', 'activity_value'])
    b = b[(b['acc'] != '') & (b['ik'] != '')]
    pn = b['patent_number'].fillna('').astype(str)
    b_pat = b[pn.str.match(r'^US\d')]
    b_art = b[~pn.str.match(r'^[A-Z]{2}\d')]

    def resid(bsub):
        bd = bsub.sort_values('activity_value').drop_duplicates(['acc', 'ik'])[['acc', 'ik', 'activity_value']]
        m = hh.merge(bd, on=['acc', 'ik'], how='inner')
        m = m[(m['activity_value'] > 0) & (m['norm_activity'] > 0)]
        return (-np.log10(m['activity_value'].values * 1e-9)) - (-np.log10(m['norm_activity'].values * 1e-9))
    return resid(b_pat), resid(b_art)


def broken_hist(fig, cell, res, label, tag, top_ylim, bot_ylim, bins, show_ylabel):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=cell, height_ratios=[1, 2], hspace=0.08)
    ax_t = fig.add_subplot(inner[0]); ax_b = fig.add_subplot(inner[1])
    wt = np.ones_like(res) * (100.0 / len(res))
    for ax in (ax_t, ax_b):
        ax.hist(res, bins=bins, weights=wt, color=RES_C, alpha=0.85, edgecolor='none')
        ax.set_xlim(-4, 4)
    ax_t.set_ylim(*top_ylim); ax_b.set_ylim(*bot_ylim)
    ax_b.set_xticks([-3, -2, -1, 0, 1, 2, 3]); ax_t.set_xticks([])
    ax_t.spines['bottom'].set_visible(False); ax_b.spines['top'].set_visible(False)
    ax_t.tick_params(bottom=False)
    d = 0.015; kw = dict(color='k', clip_on=False, lw=1)
    for x0 in (0.0, 0.5):
        ax_t.plot((x0 - d, x0 + d), (-d, d), transform=ax_t.transAxes, **kw)
        ax_b.plot((x0 - d, x0 + d), (1 - d, 1 + d), transform=ax_b.transAxes, **kw)
    ax_b.set_xlabel(r'$\Delta$pActivity (BDB $-$ HARVEST)', fontsize=14)
    ax_t.set_title(label, fontsize=15, fontweight='bold', pad=6)
    ax_t.text(-0.13, 1.14, tag, transform=ax_t.transAxes, fontsize=18, fontweight='bold', va='bottom')
    ax_t.tick_params(labelsize=12); ax_b.tick_params(labelsize=12)
    if show_ylabel:
        ax_b.set_ylabel('Percent of pairs', fontsize=14)


# ----------------------------------------------------------------------------- fidelity figure helpers
_rng = np.random.RandomState(1)


def sboot(v, N=5000):
    v = np.array([x for x in v if x == x])
    e = [v[_rng.randint(0, len(v), len(v))].mean() for _ in range(N)]
    return v.mean(), v.mean() - np.percentile(e, 2.5), np.percentile(e, 97.5) - v.mean()


def wboot(hv, rv, N=5000):
    hv = np.array([x for x in hv if x == x]); rv = np.array([x for x in rv if x == x])
    e = [WH * hv[_rng.randint(0, len(hv), len(hv))].mean() + WR * rv[_rng.randint(0, len(rv), len(rv))].mean() for _ in range(N)]
    m = WH * hv.mean() + WR * rv.mean()
    return m, m - np.percentile(e, 2.5), np.percentile(e, 97.5) - m


def grouped(ax, x, series, w=0.30, lab_fs=13.5):
    for j, (lab, means, errs, cc) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * w
        ax.bar(x + off, means, w, label=lab, color=cc, yerr=errs, capsize=3, error_kw=dict(lw=1.1, alpha=.7))
        for xi, m, hi in zip(x, means, errs[1]):
            if m == m:
                ax.text(xi + off, m + hi + 0.02, f'{m:.2f}', ha='center', fontsize=lab_fs)


# ----------------------------------------------------------------------------- fig_quality
def make_fig_quality(dfh, dfb, ho, rest):
    import seaborn as sns
    mw_h, mw_b, sa_h, sa_b = property_series(dfh, dfb)
    pa_h, pa_b = affinity(dfh, dfb)
    res_pat, res_art = residuals(dfh, dfb)

    METS = [('Compound', 'harvest_recall_compound', 'harvest_precision_compound'),
            ('Target', 'harvest_recall_target', 'harvest_precision_target'),
            ('Compound\n+ target', 'harvest_recall_pair', 'harvest_precision_pair'),
            ('Compound\n+ target\n+ value', 'harvest_recall_value', 'harvest_precision_value')]
    rm, re_, pm, pe = [], [[], []], [], [[], []]
    for _, rc, pc in METS:
        r = wboot(ho[rc].tolist(), rest[rc].tolist()); rm.append(r[0]); re_[0].append(r[1]); re_[1].append(r[2])
        hp = ho[pc].tolist() + [0.0] * len(EMPTIES)
        p = wboot(hp, rest[pc].tolist()); pm.append(p[0]); pe[0].append(p[1]); pe[1].append(p[2])
    labels = [m[0] for m in METS]

    fig = plt.figure(figsize=(16, 9.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
    TS, LS, TK = 15, 14, 12

    def titled(ax, letter, title):
        ax.set_title(title, fontsize=TS, fontweight='bold')
        ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')
        ax.tick_params(labelsize=TK)

    sns.kdeplot(mw_h[mw_h <= 1500], color=MW_H, bw_adjust=0.8, cut=0, clip=(0, 1500), label='HARVEST', ax=axA)
    sns.kdeplot(mw_b[mw_b <= 1500], color=MW_B, bw_adjust=0.8, cut=0, clip=(0, 1500), ls='--', label='BDB', ax=axA)
    axA.set_xlim(0, 1500); axA.set_xlabel('Molecular weight (Da)', fontsize=LS); axA.set_ylabel('Density', fontsize=LS)
    axA.legend(loc='upper right', fontsize=13); titled(axA, '(a)', 'Molecular weight')

    bins = np.linspace(1, 15, 91)
    axB.hist(pa_b, bins=bins, density=True, alpha=0.6, color=AF_B, edgecolor='none', label='BDB')
    axB.hist(pa_h, bins=bins, density=True, alpha=0.6, color=AF_H, edgecolor='none', label='HARVEST')
    axB.set_xlabel(r'$-\log_{10}$(IC$_{50}$, K$_i$, K$_d$, EC$_{50}$) [M]', fontsize=LS)
    axB.set_ylabel('Density', fontsize=LS); axB.legend(loc='upper right', fontsize=13)
    titled(axB, '(b)', 'Affinity')

    sns.kdeplot(sa_h, color=MW_H, label='HARVEST', ax=axC)
    sns.kdeplot(sa_b, color=MW_B, ls='--', label='BDB', ax=axC)
    axC.set_xlim(0, 10); axC.set_xlabel('Synthetic accessibility', fontsize=LS); axC.set_ylabel('Density', fontsize=LS)
    axC.legend(loc='upper right', fontsize=13); titled(axC, '(c)', 'Synthetic accessibility')

    rbins = np.linspace(-4, 4, 160)
    tails = []
    for res in (res_pat, res_art):
        cnt, edg = np.histogram(res, bins=rbins); pct = cnt * 100 / len(res)
        cen = 0.5 * (edg[:-1] + edg[1:]); tails.append(pct[np.abs(cen) > 0.5].max())
    top_ylim = (max(tails) * 3, 100); bot_ylim = (0, max(tails) * 2.5)
    broken_hist(fig, gs[1, 0], res_pat, 'Value agreement (BDB patents)', '(d)', top_ylim, bot_ylim, rbins, True)
    broken_hist(fig, gs[1, 1], res_art, 'Value agreement (BDB articles)', '(e)', top_ylim, bot_ylim, rbins, False)

    axF = fig.add_subplot(gs[1, 2])
    x = np.arange(len(labels)); w = 0.38
    for j, (lab, m, e, cc) in enumerate([('Recall', rm, re_, C_R), ('Precision', pm, pe, C_P)]):
        off = (j - 0.5) * w
        axF.bar(x + off, m, w, label=lab, color=cc, yerr=e, capsize=3, error_kw=dict(lw=1.1, alpha=.7))
        for xi, mi, hi in zip(x, m, e[1]):
            axF.text(xi + off, mi + hi + 0.02, f'{mi:.2f}', ha='center', fontsize=11)
    axF.set_xticks(x); axF.set_xticklabels(labels, fontsize=12); axF.set_ylim(0, 1.18)
    axF.set_ylabel('Agreement with manual\nreference (per-patent, 95% CI)', fontsize=LS)
    axF.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=12,
               frameon=True, framealpha=0.9, edgecolor='0.8')
    titled(axF, '(f)', 'Fidelity vs manual reference')

    fig.savefig(f'{FIGS}/fig_quality.png', dpi=400, bbox_inches='tight')
    fig.savefig(f'{FIGS}/fig_quality.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', f'{FIGS}/fig_quality.png')
    print(f'  (f) recall  : {[round(v,3) for v in rm]}   precision: {[round(v,3) for v in pm]}')
    print(f'  residuals n : patents={len(res_pat):,}  articles={len(res_art):,}')


# ----------------------------------------------------------------------------- fig_fidelity_supp
def make_fig_fidelity_supp(ho, rest, ho_full):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 6.2), gridspec_kw={'width_ratios': [1.0, 1.0]})

    he = ho['harvest_precision_pair'].tolist() + [0.0] * len(EMPTIES)
    grp = [('HARVEST\n(outside\nBDB)', sboot(ho['harvest_recall_pair'].tolist()), sboot(he)),
           ('HARVEST\n(shared\npatents)', sboot(rest['harvest_recall_pair'].tolist()), sboot(rest['harvest_precision_pair'].tolist())),
           ('BDB\n(shared\npatents)', sboot(rest['bdb_recall_pair'].tolist()), sboot(rest['bdb_precision_pair'].tolist()))]
    xb = np.arange(3)
    rmn = [g[1][0] for g in grp]; rer = [[g[1][1] for g in grp], [g[1][2] for g in grp]]
    pmn = [g[2][0] for g in grp]; per = [[g[2][1] for g in grp], [g[2][2] for g in grp]]
    grouped(ax1, xb, [('Recall', rmn, rer, C_R), ('Precision', pmn, per, C_P)])
    ax1.set_xticks(xb); ax1.set_xticklabels([g[0] for g in grp], fontsize=13.5)
    ax1.set_ylim(0, 1.12); ax1.set_ylabel('Compound + target (mean, 95% CI)', fontsize=16)
    ax1.set_title('Inside vs. outside BDB', fontsize=15.5, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=14.5)

    allp = pd.concat([ho_full[['patent', 'harvest_recall_pair', 'harvest_precision_pair', 'n_reference_datapoints']],
                      rest[['patent', 'harvest_recall_pair', 'harvest_precision_pair', 'n_reference_datapoints']]], ignore_index=True)
    allp['bk'] = allp['n_reference_datapoints'].astype(int).map(lambda n: 'easy' if n < 10 else ('medium' if n <= 100 else 'hard'))
    BK = ['easy', 'medium', 'hard']; DISP = {'easy': '< 10', 'medium': '10–100', 'hard': '> 100'}
    rm3, re3, pm3, pe3, ns = [], [[], []], [], [[], []], []
    for b in BK:
        s = allp[allp.bk == b]; ns.append(len(s))
        r = sboot(s['harvest_recall_pair'].tolist()); p = sboot(s['harvest_precision_pair'].tolist())
        rm3.append(r[0]); re3[0].append(r[1]); re3[1].append(r[2]); pm3.append(p[0]); pe3[0].append(p[1]); pe3[1].append(p[2])
    xc = np.arange(3)
    grouped(ax2, xc, [('Recall', rm3, re3, C_R), ('Precision', pm3, pe3, C_P)])
    ax2.set_xticks(xc); ax2.set_xticklabels([f'{DISP[b]}\n$n$={n}' for b, n in zip(BK, ns)], fontsize=14)
    ax2.set_ylim(0, 1.12); ax2.set_ylabel('Compound + target (mean, 95% CI)', fontsize=16)
    ax2.set_xlabel('Reference datapoints per patent', fontsize=15)
    ax2.set_title('Fidelity by number of records', fontsize=15.5, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14.5)

    for ax, lett in zip((ax1, ax2), ['(a)', '(b)']):
        ax.set_title(lett, loc='left', fontweight='bold', fontsize=18.5)
    ax1.legend(loc='upper left', ncol=1, fontsize=13, frameon=True, framealpha=0.95, edgecolor='0.8')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f'{FIGS}/fig_fidelity_supp.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'{FIGS}/fig_fidelity_supp.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', f'{FIGS}/fig_fidelity_supp.png')
    print(f'  (a) recall {[round(v,3) for v in rmn]}  prec {[round(v,3) for v in pmn]}')
    print(f'  (b) buckets n {ns}  recall {[round(v,3) for v in rm3]}  prec {[round(v,3) for v in pm3]}')


def main():
    set_style()
    print('loading HARVEST + BindingDB (once) ...'); dfh, dfb = load_all()
    print('scoring fidelity vs the manual reference ...'); per = compute_fidelity(dfh, dfb)
    ho = per[per['in_bindingdb'] == 'no']
    rest = per[per['in_bindingdb'] == 'yes']
    emp = pd.DataFrame({'patent': EMPTIES, 'harvest_recall_pair': np.nan,
                        'harvest_precision_pair': 0.0, 'n_reference_datapoints': 0})
    ho_full = pd.concat([ho[['patent', 'harvest_recall_pair', 'harvest_precision_pair', 'n_reference_datapoints']],
                         emp], ignore_index=True)
    make_fig_quality(dfh, dfb, ho, rest)
    make_fig_fidelity_supp(ho, rest, ho_full)


if __name__ == '__main__':
    main()
