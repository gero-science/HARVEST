"""Parquet schema for BindingDB-style export."""

import pyarrow as pa


PARQUET_SCHEMA = pa.schema([
    ("Ligand SMILES", pa.string()),
    ("Ligand InChI Key", pa.string()),
    ("smiles_source", pa.string()),
    ("smiles_cdx", pa.string()),
    ("inchi_key_cdx", pa.string()),
    ("Sequence", pa.string()),
    ("Ki (nM)", pa.string()),
    ("IC50 (nM)", pa.string()),
    ("Kd (nM)", pa.string()),
    ("EC50 (nM)", pa.string()),
    ("relation", pa.string()),
    ("original_range", pa.string()),
    ("patent_number", pa.string()),
    ("chemical_id", pa.string()),
    ("compound", pa.string()),
    ("compound_IUPAC_name", pa.string()),
    ("original_alias", pa.string()),
    ("protein_target_name", pa.string()),
    ("gene", pa.string()),
    ("organism", pa.string()),
    ("organism_scientific", pa.string()),
    ("Target accession", pa.string()),
    ("UniProt ID", pa.string()),
    ("is_complex", pa.string()),
    ("extreme_conditions", pa.string()),
    ("normalized_pub_number", pa.int64()),
    ("mutations", pa.string()),
    ("protein_modification", pa.string()),
    ("molecular_weight", pa.string()),
    ("assay", pa.string()),
    ("assay_id", pa.string()),
    ("assay_description", pa.string()),
    ("stage1_reasoning", pa.string()),
    ("stage2_reasoning", pa.string()),
])

OUTPUT_COLUMNS = [field.name for field in PARQUET_SCHEMA]
