# Tests

HARVEST uses [pytest](https://docs.pytest.org/). Configuration lives in [`pyproject.toml`](../pyproject.toml) (`testpaths = ["tests"]`).

## Setup

```bash
pip install -r requirements-dev.txt
```

Runtime dependencies include RDKit and git-pinned chemistry packages from `requirements.txt`; there is no supported “tests-only” install without them.

## Run

From the repository root:

```bash
pytest -q                  # full suite
pytest tests/unit -q       # unit tests only
pytest -m regression -q  # targeted bug-fix regressions (under tests/unit/)
pytest --collect-only -q   # import/collection check
```

CI is not configured in this repository; maintainers run the suite locally (or in their own CI) before releases.

## Layout

`tests/unit/` mirrors the project packages; `@pytest.mark.regression` cases live next to the code they cover.

| Path | Covers |
|------|--------|
| `tests/unit/pipeline/` | `pipeline/` (agents, workers, aggregator, run config, full run) |
| `tests/unit/llm/` | `llm/` (credentials); `llm/extractor/` covers `BioactivityExtractor` and the `bioactivity_extraction/` helpers it delegates to |
| `tests/unit/bindingdb_export/` | `bindingdb_export/` (artifacts, batches, enrichment, organisms, hallucination sidecar) |
| `tests/unit/export_table/` | [`export_table.py`](../export_table.py) entry point (CLI, re-exports, helpers) |
| `tests/unit/data_normalization/` | `data_normalization/` (units, ranges, relations) |
| `tests/unit/chemistry_rdkit/` | `chemistry_rdkit/` (IUPAC normalization) |
| `tests/unit/enrich_data/` | `enrich_data/` (RDKit wrappers) |
| `tests/unit/patent_processor/` | `patent_processor/` (ZIP source, CDX) |
| `tests/unit/final_postprocessing/` | `final_postprocessing/` |
| `tests/unit/verify_llm_res/` | `verify_llm_res/` (hallucination annotation) |
| `tests/unit/uspto_download/` | `uspto_download/` |
| `tests/conftest.py` | Adds project root to `sys.path` |

## Fixtures and data

- Most tests build minimal data in `tmp_path` and do not need API keys.
- [`tests/unit/llm/test_credentials.py`](unit/llm/test_credentials.py) checks that imports work without real LLM credentials.
- Some BindingDB/export tests expect [`curated_data/patent_mapping.csv`](../curated_data/patent_mapping.csv); they `pytest.skip` if the file is missing.
- [`tests/unit/pipeline/test_deterministic_replay_chain.py`](unit/pipeline/test_deterministic_replay_chain.py) exercises a short merge → Agent 2 bypass → Parquet export chain on synthetic patent folders.
- [`tests/unit/llm/extractor/conftest.py`](unit/llm/extractor/conftest.py) provides `make_agent` / `make_document` fixtures that build a `BioactivityExtractor` without an LLM client.
