# Final Parquet post-processing

Optional steps after `export_table.py` produces `main_res.parquet`. Run them
separately when you want a `final_v16_clean`-style table, or as the `postprocess`
stage of `pipeline.py` (`--stages all`, or `postprocess` in a run config). They
are **not** part of the default `extract,proteins,verify,export` run.

`add_final_structure.py` is intentionally **not** included in the one-command
runner (it adds a newer `final_smiles` / `final_inchikey` layer).

Hallucination filtering is handled earlier in the pipeline: the `verify` stage
writes `_hallu.json` sidecars, and the `export` stage reads them to drop
flagged rows at build time (see `verify_llm_res/`).

## One command

```bash
python -m final_postprocessing \
  results/my_run/main_res.parquet \
  results/my_run/main_res_clean.parquet
```

Default chain:

1. `add_clean_best` — `year`, `clean_smiles`, `clean_inchi_key`
2. `filter_phenotypic_assays` — drop phenotypic / off-target assay rows
3. `filter_fragments` — fragment SMILES + frequent ligand/assay/patent triplets
4. `filter_duplicates` — bad activity quadruplets / noisy patents

Useful flags:

```bash
# Only add clean columns (closest to a light enrichment step)
python -m final_postprocessing in.parquet out.parquet --enrichment-only --workers 8

# Shared SQLite cache for clean_smiles (speeds up re-runs / multi-corpus)
python -m final_postprocessing in.parquet out.parquet \
  --workers 14 \
  --cache-path /path/to/clean_smiles_cache.sqlite

# Filters only (no year/clean_smiles). Without clean_smiles, fragments is skipped.
python -m final_postprocessing in.parquet out.parquet --skip-add-clean-best

# Skip selected filters
python -m final_postprocessing in.parquet out.parquet --skip-duplicates --triplet-count 0
```

`add_clean_best` standardizes each unique `Ligand SMILES` once (process pool +
optional SQLite cache). Default `--workers` is `cpu_count - 2`.

## Individual scripts

Each step can still be run alone:

```bash
python final_postprocessing/add_clean_best.py in.parquet out.parquet --workers 8
python final_postprocessing/filter_phenotypic_assays.py in.parquet out.parquet
python final_postprocessing/filter_fragments.py in.parquet out.parquet
python final_postprocessing/filter_duplicates.py --input in.parquet --output out.parquet
```

Optional separate structure layer (not in the default chain):

```bash
python final_postprocessing/add_final_structure.py in.parquet out.parquet --workers 4
```
