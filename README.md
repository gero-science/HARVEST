# HARVEST

**HARVEST** (High-throughput Automated extRaction of bioactivity from patentS using agenTic LLMs) is a multi-agent LLM system for automated extraction of protein-ligand bioactivity data from USPTO patents. The pipeline processes patent documents through specialized agents for assay identification, bioactivity extraction, compound name resolution, chemical structure resolution, and protein mapping to UniProt identifiers.

From 43,187 USPTO patents, HARVEST extracted 3.15 million activity records, identifying 326,342 novel molecular scaffolds and 967 protein targets absent from existing databases like BindingDB.

![HARVEST Pipeline](imgs/harvest_pipeline.png)

## Links

- **Paper**: [biorxiv:10.64898/2026.03.15.711910](https://www.biorxiv.org/content/10.64898/2026.03.15.711910)
- **Dataset**: [h-bench](./data/h_bench)

## Dataset Structure

The released H-bench dataset (`data/h_bench/`) contains one CSV per protein target (48 targets), each a curated subset of HARVEST results filtered for valid activity values and compounds absent from BindingDB.

| Column | Description                                                                                          |
|---|------------------------------------------------------------------------------------------------------|
| `Sequence` | Full amino acid sequence of the protein target, multiple entries separated by `;`                    |
| `Ki (nM)` | Inhibition constant in nanomolar (if reported)                                                       |
| `IC50 (nM)` | Half-maximal inhibitory concentration in nanomolar (if reported)                                     |
| `Kd (nM)` | Dissociation constant in nanomolar (if reported)                                                     |
| `EC50 (nM)` | Half-maximal effective concentration in nanomolar (if reported)                                      |
| `relation` | Activity value relation operator (`=`, `<`, `>`, `~`)                                                |
| `original_range` | Original range string if the patent reported a range (e.g. "10-20 nM")                               |
| `patent_number` | USPTO patent identifier (e.g. US20200299264A1)                                                       |
| `chemical_id` | Internal chemical identifier from the patent XML                                                     |
| `compound_IUPAC_name` | IUPAC name of the compound (when available in the patent text)                                       |
| `original_alias` | Compound alias as used in the patent (e.g. "Example 1", "Compound 160a")                             |
| `protein_target_name` | Name of the protein target as stated in the patent                                                   |
| `gene` | Gene symbol (e.g. ESR1, JAK1)                                                                        |
| `organism_scientific` | Scientific name of the source organism (e.g. Homo sapiens)                                           |
| `uniprot_acc` | UniProt accession of the protein target, multiple entries separated by `;`                                                              |
| `UniProt ID` | UniProt entry name (e.g. ESR1_HUMAN), multiple entries separated by `;`                                                                 |
| `mutations` | Protein mutations noted in the assay (if any)                                                        |
| `protein_modification` | Protein modification type (e.g. Degradation, Inhibition)                                             |
| `assay` | Assay type classification (e.g. Functional, Binding)                                                 |
| `assay_description` | Full assay description extracted from the patent                                                     |
| `year` | Publication year of the patent                                                                       |
| `SMILES` | Canonical SMILES representation of the compound                                                      |
| `cluster_id` | Tanimoto-based cluster identifier from scaffold clustering                                           |
| `final_label` | Split label: `A` = novel (harvest-unique cluster), `C` = buffer region between HARVEST and BindingDB |
| `nearest_BDB_smiles` | SMILES of the most similar compound in BindingDB                                                     |
| `tanimoto_sim` | Tanimoto similarity to the nearest BindingDB compound                                                |
| `InChIKey` | InChIKey identifier for the compound                                                                 |

## Limitations

- **English USPTO patents only**: The current release covers only US patents in English. EPO, WIPO, and non-English patents are not included.
- **h_bench is a raw subset, not a ready-to-use benchmark**: h_bench contains molecules from HARVEST that are absent from BindingDB, but it is not a fully curated validation set. To create a high-quality benchmark for a specific ML task (e.g. virtual screening, binding affinity prediction, activity cliff detection), additional processing is required — such as filtering by activity type, selecting active/inactive thresholds, deduplicating by scaffold, and ensuring temporal or structural separation from training data.

## Training Data Allocation

When using H-bench as a test set, your training data must be checked for leakage. The `allocate_training.py` script compares your training dataset against H-bench and adds a `final_label` column indicating whether each molecule is safe to use:

- **`B`** — safe for training, no leakage risk
- **`A`** — exact duplicate of an H-bench test compound, must be excluded
- **`C`** — too structurally similar to test compounds (within the Tanimoto buffer zone), must be excluded

**Only keep rows with `final_label == "B"` in your training set.**

Protein matching uses UniRef90 clusters by default: if your training data contains a protein that shares a UniRef90 cluster (>=90% sequence identity) with an H-bench protein, the script will check for compound-level leakage across all cluster members.

### Usage

```bash
# Basic usage (UniRef90 protein grouping, recommended)
python allocate_training.py \
    --new-data your_training_data.parquet \
    --split-dir data/h_bench/ \
    -o output/ \
    --uniref-cache uniref_cache.json

# With custom column names
python allocate_training.py \
    --new-data your_training_data.csv.gz \
    --split-dir data/h_bench/ \
    -o output/ \
    --smiles-col-new SMILES \
    --uniprot-col-new uniprot_id

# Without UniRef grouping (exact UniProt match only)
python allocate_training.py \
    --new-data your_training_data.parquet \
    --split-dir data/h_bench/ \
    -o output/ \
    --uniref none
```

The script outputs:
- Labeled copy of your data with the `final_label` column (same format as input: parquet or csv.gz)
- `allocation_report.csv` — per-protein breakdown of removed molecules

