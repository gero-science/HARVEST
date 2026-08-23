# verify_llm_res

Hallucination annotation and LLM output quality checks.

## Pipeline stage (`python -m verify_llm_res`)

Writes `_hallu.json` sidecars next to `_resolved.json` files. The `export`
stage reads them to drop flagged rows at build time.

```bash
python -m verify_llm_res --results-dir results/run1 --source-dir data/patents --workers 16
```

## Standalone scripts

### 🔍 tsv_problems_summary.py

Aggregates TSV quality issues per patent directory.

Checks:
- ✅ Column count consistency with the header (all TSV files in the directory)
- ✅ IUPAC names ending with a trailing dash `-` (stage2 files)

```bash
python verify_llm_res/tsv_problems_summary.py results/run1
```

### 📊 check_tsv_columns.py

Checks structural integrity of TSV files against the expected stage format.

- Compares column counts in the header and data rows
- Finds rows with incorrect column counts
- Shows line numbers of problematic rows

### 📈 compare_with_mapping.py

Compares pipeline output with a BindingDB reference table using the
application_number ↔ patent_number mapping.

```bash
python verify_llm_res/compare_with_mapping.py \
  --test-table results.csv \
  --bdb-table reference.parquet \
  --mapping-file curated_data/patent_mapping.csv
```
