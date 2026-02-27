# HARVEST

**HARVEST** (High-throughput Automated extRaction of bioactivity from patentS using agenTic LLMs) is a multi-agent LLM system for automated extraction of protein-ligand bioactivity data from USPTO patents. The pipeline processes patent documents through specialized agents for assay identification, bioactivity extraction, compound name resolution, chemical structure resolution, and protein mapping to UniProt identifiers.

From 43,187 USPTO patents, HARVEST extracted 3.68 million activity records, identifying 365,713 novel molecular scaffolds and 961 protein targets absent from existing databases like BindingDB.

![HARVEST Pipeline](imgs/harvest_pipeline.png)

## Links

- **Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- **Dataset**: [Zenodo](https://zenodo.org/records/XXXXXXX)

## Dataset Structure

The released dataset is a curated subset filtered for valid activity values and selected protein targets. <!-- TODO: add specific filtering criteria -->

| Column | Description |
|--------|-------------|
| `patent_id` | USPTO patent identifier |
| `protein_target_name` | Name of the protein target |
| `organism` | Source organism of the protein |
| `binding_metric` | Activity measurement type (IC50, Ki, Kd, EC50) |
| `value` | Numeric activity value |
| `unit` | Measurement unit (nM, μM, etc.) |
| `smiles` | SMILES representation of the compound |
| `compound` | Compound name or alias from the patent |
| `inchikey` | InChIKey identifier for the compound |
| `scaffold` | Murcko scaffold of the compound |
| `protein_class` | Protein classification (kinase, GPCR, etc.) |

## Limitations

- **English USPTO patents only**: The current release covers only US patents in English. EPO, WIPO, and non-English patents are not included.

