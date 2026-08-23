"""Row filtering and conversion for BindingDB export."""

import math
from typing import Any, Dict, Optional

from data_normalization.normalize_data import process_row


ACCEPTED_AGENT2_SOURCE = "skipped_enriched_by_agent1"


def _normalized_pub_number(patent_dict: Dict[str, int], patent_number: str) -> int:
    """Look up the granted publication number, falling back to 0.

    The dictionary is built from an application table in which every pending or
    abandoned application carries a NaN patent number -- a third of the entries
    in a typical dump. Those keys are present, so a plain ``dict.get(key, 0)``
    returns the NaN rather than the default and ``int()`` then raises, killing
    the whole export over patents that simply were never granted. Treat a
    non-finite value the same as a missing key.
    """
    value = patent_dict.get(patent_number, 0)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 0
    return int(value)


def normalize_accepted_bindings(bindings: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    normalize_bindings = []
    for row in bindings:
        if row.get("agent2_resolution_source") != ACCEPTED_AGENT2_SOURCE:
            continue

        processed = process_row(row)
        if isinstance(processed, dict):
            normalize_bindings.append(processed)

    return normalize_bindings


def _inchi_connectivity(inchi_key: str) -> str:
    """First 14-char block of an InChI Key (molecular graph identity)."""
    return inchi_key.split("-")[0] if inchi_key else ""


def select_ligand_smiles(row: Dict[str, Any]) -> tuple[str, str, str]:
    """Pick best (smiles, inchi_key, smiles_source) using hallu flags + CDX comparison.

    When both OPSIN- and CDX-derived structures are available:
    - Same InChI connectivity → prefer OPSIN (canonical IUPAC-derived SMILES).
    - Different connectivity  → prefer CDX (structural drawing is ground truth).

    CDX-based sources (table_chemistry_tag and similar) always use CDX
    structures (smiles_cdx / inchi_key_cdx) when available.

    Hallucination flags disable a source:
    - IUPAC flagged  → OPSIN tainted (name was hallucinated, SMILES unreliable).
    - chem_id flagged → CDX tainted (lookup may have matched the wrong compound).
    """
    opsin_smiles = row.get("Ligand SMILES") or ""
    opsin_inchi = row.get("Ligand InChI Key") or ""
    cdx_smiles = row.get("smiles_cdx") or ""
    cdx_inchi = row.get("inchi_key_cdx") or ""
    source = row.get("smiles_source") or ""
    hallu = row.get("_hallu_masked", set())

    is_opsin = source.startswith("py2opsin")

    if not is_opsin:
        # table_chemistry_tag or other CDX-based source — prefer CDX structures
        if "chemical_id" in hallu:
            return "", "", source
        # Use CDX SMILES/InChI when available; fall back to Ligand fields
        smiles = cdx_smiles or opsin_smiles
        inchi = cdx_inchi or opsin_inchi
        return smiles, inchi, source

    # OPSIN source path
    opsin_tainted = "compound_IUPAC_name" in hallu
    cdx_tainted = "chemical_id" in hallu
    opsin_ok = bool(opsin_smiles) and not opsin_tainted
    cdx_ok = bool(cdx_smiles) and not cdx_tainted

    if opsin_ok and cdx_ok:
        o_conn = _inchi_connectivity(opsin_inchi)
        c_conn = _inchi_connectivity(cdx_inchi)
        if o_conn and c_conn and o_conn == c_conn:
            return opsin_smiles, opsin_inchi, source        # same → OPSIN
        else:
            return cdx_smiles, cdx_inchi, "cdx_over_opsin"  # differ → CDX
    elif opsin_ok:
        return opsin_smiles, opsin_inchi, source
    elif cdx_ok:
        return cdx_smiles, cdx_inchi, "cdx_fallback"
    else:
        return "", "", source


def binding_to_parquet_row(row: Dict[str, Any], patent_dict: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """
    Convert a binding to a Parquet table row.

    Returns:
        Dict with Parquet fields, or None if the row should be skipped
    """
    ligand_smiles, ligand_inchi_key, smiles_source = select_ligand_smiles(row)
    has_inchi_key = bool(ligand_inchi_key)
    has_protein_info = row.get("Sequence") or row.get("gene")
    if not (has_inchi_key and has_protein_info and row.get("binding_metric")):
        return None

    metric = row["binding_metric"].lower().strip()
    if metric not in ["ic50", "ec50", "kd", "ki"]:
        return None

    out_row = {
        "Ligand SMILES": ligand_smiles,
        "Ligand InChI Key": ligand_inchi_key,
        "smiles_source": smiles_source,
        "smiles_cdx": row.get("smiles_cdx", ""),
        "inchi_key_cdx": row.get("inchi_key_cdx", ""),
        "Sequence": row.get("Sequence", ""),
        "Ki (nM)": None,
        "IC50 (nM)": None,
        "Kd (nM)": None,
        "EC50 (nM)": None,
        "relation": row.get("relation", "="),
        "original_range": row.get("original_range", ""),
        "patent_number": row.get("patent_number", ""),
        "chemical_id": row.get("chemical_id", ""),
        "compound": row.get("compound", ""),
        "compound_IUPAC_name": row.get("compound_IUPAC_name", "") if smiles_source.startswith("py2opsin") else "",
        "original_alias": row.get("original_alias", ""),
        "protein_target_name": row.get("protein_target_name", ""),
        "gene": row.get("gene", ""),
        "organism": row.get("organism", ""),
        "organism_scientific": row.get("organism_scientific", ""),
        "Target accession": row.get("Target accession", ""),
        "UniProt ID": row.get("UniProt ID", ""),
        "is_complex": str(row.get("is_complex", "")) if row.get("is_complex") else "",
        "extreme_conditions": row.get("extreme_conditions", ""),
        "normalized_pub_number": _normalized_pub_number(patent_dict, row.get("patent_number", "")),
        "mutations": ", ".join(row.get("mutations", [])) if row.get("mutations") else "",
        "protein_modification": row.get("protein_modification", ""),
        "molecular_weight": str(row.get("molecular_weight", "")) if row.get("molecular_weight") else "",
        "assay": row.get("assay", ""),
        "assay_id": row.get("assay_id", ""),
        "assay_description": row.get("assay_description", ""),
        "stage1_reasoning": row.get("stage1_reasoning", ""),
        "stage2_reasoning": row.get("stage2_reasoning", ""),
    }

    value = row.get("value")
    if metric == "ic50":
        out_row["IC50 (nM)"] = str(value) if value is not None else None
    elif metric == "ec50":
        out_row["EC50 (nM)"] = str(value) if value is not None else None
    elif metric == "kd":
        out_row["Kd (nM)"] = str(value) if value is not None else None
    elif metric == "ki":
        out_row["Ki (nM)"] = str(value) if value is not None else None

    return out_row
