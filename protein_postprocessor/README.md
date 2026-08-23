# Protein Postprocessor

Module for resolving proteins from pipeline results. Uses an LLM to map protein names to official gene symbols, then fetches sequences from the local UniProt database (FastaGeneResolver).

## Purpose

The module processes proteins from pipeline results:

1. Reads `*_resolved.json` from each patent folder
2. Splits records into individual proteins (`is_complex=N`) and complexes (`is_complex=Y`)
3. Sends them to the LLM to obtain gene symbols and scientific species
4. Looks up sequences in `uniprot_sprot.fasta` by gene + species
5. Saves results:
   - `single_proteins.json` — individual proteins
   - `protein_complexes.json` — protein complexes (all genes in the complex)

## Species / organism policy

This module resolves whatever organism label it receives and does not decide
whether the species was actually stated in the patent. That audit lives in the
export stage: `bindingdb_export/organisms.py` classifies every row internally
(`stated`, `human_default`, `inferred_from_target`, `unstated`,
`human_fallback`) using the Stage 1 `organism` column. Assumed human rows are
written as `organism = "human (default)"`; there is no separate
`species_source` column. See "Species policy" in [AGENTS.md](../AGENTS.md).

`FastaGeneResolver` does not substitute a human ortholog when the requested
species is missing from the FASTA, so a resolved species is never silently
replaced by Homo sapiens here.

## Installation

The module uses the main project dependencies. Make sure:

1. Dependencies from `requirements.txt` are installed
2. `.env` is configured with an OpenRouter API key
3. The file `data/protein_data/uniprot_sprot.fasta` is downloaded

## Usage

### Basic run

```bash
# Process both proteins and complexes (default)
python -m protein_postprocessor results/bdb_100
```

### With options

```bash
# 100 parallel workers
python -m protein_postprocessor results/bdb_100 --workers 100

# Individual proteins only (no complexes)
python -m protein_postprocessor results/bdb_100 --singles-only

# Complexes only (no individual proteins)
python -m protein_postprocessor results/bdb_100 --complexes-only

# Without Stage 1 context (enabled by default)
python -m protein_postprocessor results/bdb_100 --no-stage1-context

# Only proteins without sequence in _resolved.json (default processes all)
python -m protein_postprocessor results/bdb_100 --no-reprocess-all

# Reprocess all patents (ignore existing files)
python -m protein_postprocessor results/bdb_100 --force

# Debug mode
python -m protein_postprocessor results/bdb_100 --debug
```

## Command-line arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `results_dir` | Pipeline results directory | (required) |
| `--workers` | Number of parallel workers | 5 |
| `--singles-only` | Process individual proteins only | False |
| `--complexes-only` | Process complexes only | False |
| `--with-stage1-context` | Pass Stage 1 assay context to the LLM | **True** |
| `--no-stage1-context` | Do not pass Stage 1 context to the LLM | - |
| `--reprocess-all` | Process all proteins, including already resolved | **True** |
| `--no-reprocess-all` | Process only proteins without sequence | - |
| `--force` | Reprocess patents with existing output files | False |
| `--debug` | Enable debug mode | False |
| `--log-level` | Logging level | INFO |

## Input data

The module expects this folder structure:

```
results/
├── US20140296271A1/
│   ├── US20140296271A1_resolved.json              # Main data source
│   └── US20140296271A1_agent1_stage1_targets.tsv  # For --with-stage1-context
├── US20140364410A1/
│   ├── US20140364410A1_resolved.json
│   └── ...
└── ...
```

### `*_resolved.json` format

```json
[
  {
    "compound": "101",
    "protein_target_name": "ORL-1 receptor",
    "organism": "human",
    "is_complex": "N",
    "protein_sequence": null,
    "protein_accession": null
  },
  {
    "compound": "102",
    "protein_target_name": "IL-12 receptor signaling",
    "organism": "human",
    "is_complex": "Y",
    "protein_sequence": null,
    "protein_accession": null
  }
]
```

## Output data

### `single_proteins.json` (individual proteins)

```json
{
  "patent_id": "US20140296271A1",
  "proteins": [
    {
      "protein_name": "EGFR",
      "organism": "human",
      "organism_scientific": "Homo sapiens",
      "gene": "EGFR",
      "sequence": "MKTAY...",
      "accession": "P00533",
      "uniprot_id": "EGFR_HUMAN"
    }
  ]
}
```

### `protein_complexes.json` (complexes)

```json
{
  "patent_id": "US20140296271A1",
  "complexes": [
    {
      "complex_name": "IL-12 receptor signaling",
      "organism": "human",
      "species": "Homo sapiens",
      "proteins": [
        {
          "gene": "TYK2",
          "sequence": "MPRG...",
          "accession": "P29597",
          "uniprot_id": "TYK2_HUMAN"
        },
        {
          "gene": "JAK2",
          "sequence": "MGMACLT...",
          "accession": "O60674",
          "uniprot_id": "JAK2_HUMAN"
        },
        {
          "gene": "STAT4",
          "sequence": "...",
          "accession": "Q14765",
          "uniprot_id": "STAT4_HUMAN"
        }
      ]
    }
  ]
}
```

## How it works

1. **Scanning** — finds patent folders with `*_resolved.json`
2. **Skip processed** — patents with existing files are skipped (unless `--force`)
3. **Splitting** — proteins and complexes are processed separately
4. **Parallel processing** — N workers process patents concurrently
5. **LLM mapping**:
   - For proteins: obtains gene symbol and species
   - For complexes: obtains ALL genes in the complex
6. **FastaGeneResolver** — looks up sequence in FASTA by (gene, species)
7. **Saving** — results are written to the corresponding files

## Fault tolerance

- **Independent processing** — each patent is processed and saved separately
- **Separate skip logic** — proteins and complexes are skipped independently
- **Resume** — on restart, continues with unprocessed patents
- **Retry** — retries on API errors

If the script fails, data is not lost. You can:
1. Restart — it continues with unprocessed patents
2. Use `--force` for full reprocessing
3. Use `--singles-only` or `--complexes-only` for partial reprocessing

## Configuration

Settings in `config_postprocess.py`:

| Parameter | Description | Value |
|-----------|-------------|-------|
| `MODEL_NAME` | LLM model | `openai/gpt-5.1` |
| `API_RETRY_ATTEMPTS` | API retry attempts | 5 |
| `MAX_TOKENS_RESPONSE` | Maximum response tokens | 20000 |
| `UNIPROT_FASTA_PATH` | Path to FASTA file | `data/protein_data/uniprot_sprot.fasta` |

## Logging

Logs are written to:
- Console (level via `--log-level`)
- File `{results_dir}/protein_postprocessor.log`

## Module structure

```
protein_postprocessor/
├── __init__.py
├── __main__.py              # CLI interface
├── protein_postprocessor.py # Main logic
├── config_postprocess.py    # Configuration and prompts
└── README.md
```
