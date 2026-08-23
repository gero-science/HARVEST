"""Protein artifact loading for BindingDB export."""

import json
import logging
import os
from typing import Any, Dict

from .organisms import normalize_organism


def load_single_proteins(patent_dir: str) -> Dict[str, Any]:
    """Load proteins from single_proteins.json in the patent directory."""
    single_proteins_path = os.path.join(patent_dir, "single_proteins.json")
    if not os.path.exists(single_proteins_path):
        return {}

    try:
        with open(single_proteins_path, "r") as f:
            data = json.load(f)

        resolved = {}
        for protein in data.get("proteins", []):
            name = protein.get("protein_name")
            org = normalize_organism(protein.get("organism"))
            if name:
                if name not in resolved:
                    resolved[name] = {}
                resolved[name][org] = {
                    "sequence": protein.get("sequence"),
                    "accession": protein.get("accession"),
                    "uniprot_id": protein.get("uniprot_id"),
                    "gene": protein.get("gene"),
                    "organism_scientific": protein.get("organism_scientific"),
                }

        return resolved
    except Exception as e:
        logging.debug(f"Error loading single_proteins.json from {patent_dir}: {e}")
        return {}


def load_protein_complexes(patent_dir: str) -> Dict[str, Any]:
    """Load complexes from protein_complexes.json in the patent directory.

    For complexes:
    - Sequences are concatenated (without a separator)
    - gene, accession, uniprot_id are joined with ';'
    """
    complexes_path = os.path.join(patent_dir, "protein_complexes.json")
    if not os.path.exists(complexes_path):
        return {}

    try:
        with open(complexes_path, "r") as f:
            data = json.load(f)

        resolved = {}
        for cpx in data.get("complexes", []):
            name = cpx.get("complex_name")
            org = normalize_organism(cpx.get("organism"))
            proteins = cpx.get("proteins", [])

            if name and proteins:
                sequences = ";".join(p.get("sequence", "") for p in proteins if p.get("sequence"))
                genes = ";".join(p.get("gene", "") for p in proteins if p.get("gene"))
                accessions = ";".join(p.get("accession", "") for p in proteins if p.get("accession"))
                uniprot_ids = ";".join(p.get("uniprot_id", "") for p in proteins if p.get("uniprot_id"))

                if name not in resolved:
                    resolved[name] = {}
                resolved[name][org] = {
                    "sequence": sequences or None,
                    "accession": accessions or None,
                    "uniprot_id": uniprot_ids or None,
                    "gene": genes or None,
                    "organism_scientific": cpx.get("species"),
                }

        return resolved
    except Exception as e:
        logging.debug(f"Error loading protein_complexes.json from {patent_dir}: {e}")
        return {}


def load_resolved_proteins(patent_dir: str) -> Dict[str, Any]:
    resolved_proteins = load_single_proteins(patent_dir)

    complexes = load_protein_complexes(patent_dir)
    for name, orgs in complexes.items():
        if name not in resolved_proteins:
            resolved_proteins[name] = {}
        resolved_proteins[name].update(orgs)

    return resolved_proteins
