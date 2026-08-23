"""CDX helpers for BindingDB export."""

import re
from typing import Any, Dict, List, Optional


def extract_chem_num_from_chemical_id(chemical_id: str) -> Optional[str]:
    """Extract compound number from chemical_id."""
    if not chemical_id:
        return None
    match = re.search(r"CHEM-[A-Z]{2}-(\d+)", chemical_id)
    return match.group(1) if match else None


def apply_cdx_results_to_bindings(
    bindings: List[Dict[str, Any]],
    cdx_results: Dict[str, Dict],
) -> None:
    """Apply CDX parsing results to bindings with chemical_id."""
    for binding in bindings:
        chem_num = extract_chem_num_from_chemical_id(binding.get("chemical_id"))
        if not chem_num or chem_num not in cdx_results:
            continue

        cdx_data = cdx_results[chem_num]
        if cdx_data.get("smiles"):
            binding["smiles_cdx"] = cdx_data["smiles"]
        if cdx_data.get("inchikey"):
            binding["inchi_key_cdx"] = cdx_data["inchikey"]
