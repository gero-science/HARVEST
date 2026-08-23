# Pipeline Module

Modular structure of the asynchronous pipeline for extracting and processing patent data.

Protein resolution, Parquet export and final cleanup are **not** part of the
extraction core: they are separate stages orchestrated by
[`full_run.py`](full_run.py) (see [Run stages](#run-stages)).

## Module structure

```
pipeline/
├── __init__.py                 # Public module API
├── run_config.py               # YAML run config, CLI overrides, stage resolution
├── full_run.py                 # Stage orchestration (extract/proteins/verify/export/postprocess)
├── core.py                     # Main run_pipeline_async function
├── producer.py                 # Producer for generating tasks from patents
├── workers.py                  # Agent 1 workers and Agent 2 aggregator
├── agent1_processing.py        # Single-patent Agent 1 helpers
├── agent2_orchestration.py     # Per-patent provenance tagging and saves
├── agent2_processing.py        # Alias-resolution accounting / debug payloads
├── alias_resolution.py         # resolve_aliases / bypass path
├── accounting.py               # Usage / global stats helpers
├── artifacts.py                # Per-patent result JSON writing
├── aggregator_state.py         # Aggregator state
├── debug_artifacts.py          # Debug artifact helpers
├── finalization.py             # Pipeline finalization
├── postprocess_integration.py  # Fuzzy mapping helpers used in merge
├── statistics.py               # Statistics and report generation
├── error_handling.py           # Error classification
└── utils.py                    # Logging, list reading, usage helpers
```

## Run stages

A run is a sequence of stages over one output directory, in this canonical order:

| Stage | What it does | Main artifacts |
| --- | --- | --- |
| `extract` | Agent 1 extraction | `{patent_id}_resolved.json`, stage TSV, `cdx_results.json` |
| `proteins` | [`protein_postprocessor`](../protein_postprocessor/) | `single_proteins.json`, `protein_complexes.json` |
| `verify` | [`verify_llm_res`](../verify_llm_res/) hallucination annotation | `{patent_id}_hallu.json` sidecars |
| `export` | [`export_table.py`](../export_table.py) export | `main_res.parquet` |
| `postprocess` | [`final_postprocessing`](../final_postprocessing/) chain | `main_res_clean.parquet` |

Stage selection precedence: `--stages` > `--from-stage` > `stages` in the config
file > the default, `extract,proteins,verify,export`. `--from-stage X` runs X and
every later stage, `postprocess` included.

Before anything runs, artifacts of stages that are *not* part of the run are
checked: `proteins`/`export` need `*_resolved.json` in the output directory, and
`postprocess` needs its input Parquet. A missing prerequisite fails the run
instead of producing an empty table.

A failing stage stops the run with exit code 1; `--continue-on-error` keeps going
through the remaining stages while still exiting non-zero. Every run writes
`pipeline_run_summary.json` with per-stage status, duration, artifacts, and the
effective configuration.

`add_final_structure` is intentionally not part of the `postprocess` stage, same
as in `python -m final_postprocessing`.

## Main components

### run_config.py / full_run.py
- `load_run_config()` reads a YAML config (see
  [`configs/pipeline.example.yaml`](../configs/pipeline.example.yaml)); unknown
  keys and stage names are errors, not silent typos
- `apply_cli_overrides()` puts CLI arguments on top of the file; store_true flags
  (`--debug`, `--resume`, `--continue-on-error`) can only switch a setting on
- `run_full_pipeline()` validates, runs the stages in canonical order and writes
  the run summary
- LLM credentials and extraction worker limits stay in `.env` / [`config.py`](../config.py)

### core.py
Contains the main `run_pipeline_async` function, which:
- Initializes all pipeline components
- Creates asynchronous queues for coordination between agents
- Starts producer, workers, and aggregator in parallel
- Manages the pipeline lifecycle

### producer.py
- `producer_chunks()` - asynchronously reads patent ZIP archives
- Generates one task per patent from its XML/chemistry data
- Enqueues tasks for Agent 1

### workers.py
- `worker_agent1()` - extracts bioactivity data via SimpleAgent1
- `worker_agent2_aggregator()` - aggregates Agent 1 output per patent, tags row provenance, and saves results
- Delegates orchestration/accounting/artifact writing to sibling modules

### statistics.py / error_handling.py / utils.py
Unchanged roles: reports, error classification, logging and list helpers.

## Usage

### As a library
```python
from pipeline import run_pipeline_async, setup_logging

setup_logging("INFO")
await run_pipeline_async(
    input_paths=["patents.zip"],
    output_dir="results",
    limit=None,
    resume=False,
    debug_mode=False
)
```

Whole run with stages:

```python
from pipeline import load_run_config, resolve_stages, run_full_pipeline

config = load_run_config("configs/pipeline.example.yaml")
exit_code = await run_full_pipeline(config, resolve_stages(config))
```

### As CLI
```bash
# extraction only
python pipeline.py --input-path patents.zip --output-dir results --stages extract

# everything, including final_postprocessing
python pipeline.py --input-path data/hallu_100 --output-dir results/run1 --stages all

# catch up on an existing results directory, no LLM extraction
python pipeline.py --output-dir results/run1 --from-stage export

# from a config file
python pipeline.py --config configs/pipeline.example.yaml
```

## Pipeline architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Producer   │───▶│  Agent 1    │───▶│  Aggregator │
│  (patents)  │    │  (extract)  │    │  (Agent 2)  │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
                                    *_resolved.json          stage: extract
                                              │
                                              ▼
                         protein_postprocessor              stage: proteins
                                              │
                                              ▼
                          verify_llm_res annotate            stage: verify
                                              │
                                              ▼
                            export_table.py export          stage: export
                                              │
                                              ▼
                        final_postprocessing chain          stage: postprocess
```

## Refactoring

This package was split out of a monolithic `pipeline.py` while keeping the CLI
facade. Prefer the current code over older line-count notes in historical docs.
