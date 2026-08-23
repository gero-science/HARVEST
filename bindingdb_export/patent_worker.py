"""Per-patent worker for BindingDB export."""

import logging
from typing import Any, Dict, Optional

from .artifacts import load_cdx_results, load_mol_results, load_hallu_flags, load_resolved_bindings, mask_hallucinated_fields
from .cdx import apply_cdx_results_to_bindings
from .enrichment import process_single_patent
from .proteins import load_resolved_proteins
from .rows import normalize_accepted_bindings


RESOLVED_PATTERNS = ["03_final_output.json", "_resolved.json"]


def process_patent_worker(args: tuple) -> tuple[str, list[Dict[str, Any]], Optional[str]]:
    """Worker function for processing one patent.

    *args* is ``(patent_dir, ligand_extractor)`` or
    ``(patent_dir, ligand_extractor, structure_source)`` where
    *structure_source* is ``"cdx"`` (default), ``"mol"``, or ``"none"``.
    """
    if len(args) == 3:
        patent_dir, ligand_extractor, structure_source = args
    else:
        patent_dir, ligand_extractor = args
        structure_source = "cdx"

    try:
        bindings = load_resolved_bindings(patent_dir, RESOLVED_PATTERNS)

        if not bindings:
            return patent_dir, [], None

        if structure_source == "mol":
            struct_results = load_mol_results(patent_dir)
        elif structure_source == "cdx":
            struct_results = load_cdx_results(patent_dir)
        else:
            struct_results = {}
        if struct_results:
            apply_cdx_results_to_bindings(bindings, struct_results)

        normalize_bindings = normalize_accepted_bindings(bindings)

        if not normalize_bindings:
            return patent_dir, [], None

        # Mask hallucinated fields when a sidecar annotation is present.
        # Instead of dropping the entire binding, individual fields
        # (chemical_id, compound_IUPAC_name, value) are blanked so the
        # remaining data is preserved in the Parquet.
        hallu_flags = load_hallu_flags(patent_dir)
        if hallu_flags:
            for b in normalize_bindings:
                masked = mask_hallucinated_fields(b, hallu_flags)
                b["_hallu_masked"] = masked

        resolved_proteins = load_resolved_proteins(patent_dir)

        updated_bindings = process_single_patent(
            patent_dir,
            ligand_extractor,
            resolved_proteins,
            RESOLVED_PATTERNS,
            bindings=normalize_bindings,
        )

        return patent_dir, updated_bindings, None

    except Exception as e:
        error_msg = f"Error processing {patent_dir}: {str(e)}"
        logging.error(error_msg)
        return patent_dir, [], error_msg
