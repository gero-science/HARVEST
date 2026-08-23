from typing import Dict, List

from patent_processor import PatentDocument
from chemistry_rdkit import (
    fix_balanced,
    normalize_iupac_spelling,
    fix_balanced_by_adding,
    fix_balanced_by_removing_variants,
)


def enrich_with_smiles(
    valid_data: List[Dict],
    doc: PatentDocument,
    py2opsin_available: bool,
    py2opsin_func,
    logger=None,
) -> None:
    """Enrich merged rows with SMILES/InChIKey, mutating valid_data in place."""
    if not valid_data:
        return

    enriched_py2opsin = 0
    enriched_py2opsin_normalized = 0
    enriched_py2opsin_fix_iupac = 0
    enriched_chemistry_node = 0

    if py2opsin_available:
        iupac_data = []

        for i, item in enumerate(valid_data):
            iupac_name = item.get("compound_IUPAC_name", "").strip()

            if not iupac_name:
                continue

            if iupac_name.upper().endswith('.TIF') or iupac_name.upper().endswith('.PNG') or iupac_name.upper().endswith('.JPG'):
                if logger:
                    logger.debug(f"Skipping image file for py2opsin: {iupac_name}")
                continue

            if len(iupac_name) < 3:
                if logger:
                    logger.debug(f"Skipping too short name for py2opsin: {iupac_name}")
                continue

            normalized = normalize_iupac_spelling(fix_balanced(iupac_name))

            if not normalized or not normalized.strip():
                if logger:
                    logger.debug(f"IUPAC name became empty after normalization: {iupac_name}")
                continue

            iupac_data.append((iupac_name, normalized, i))

        if iupac_data:
            try:
                if logger:
                    logger.debug(f"Processing {len(iupac_data)} IUPAC names with py2opsin (two-stage strategy)")

                original_names = [item[0] for item in iupac_data]

                smiles_results_orig = py2opsin_func(
                    chemical_name=original_names,
                    output_format="SMILES",
                    allow_bad_stereo=True
                )

                inchikey_results_orig = py2opsin_func(
                    chemical_name=original_names,
                    output_format="StdInChIKey",
                    allow_bad_stereo=True
                )

                if not isinstance(smiles_results_orig, list):
                    smiles_results_orig = [smiles_results_orig]
                if not isinstance(inchikey_results_orig, list):
                    inchikey_results_orig = [inchikey_results_orig]

                failed_indices = []

                for idx, (original_name, normalized_name, valid_data_idx) in enumerate(iupac_data):
                    smiles = smiles_results_orig[idx] if idx < len(smiles_results_orig) else False
                    inchikey = inchikey_results_orig[idx] if idx < len(inchikey_results_orig) else False

                    if smiles and smiles is not False:
                        item = valid_data[valid_data_idx]
                        item["molecule_smiles"] = smiles
                        item["smiles_source"] = "py2opsin"
                        enriched_py2opsin += 1

                        if inchikey and inchikey is not False:
                            item["molecule_inchikey"] = inchikey

                        if logger:
                            logger.debug(
                                f"py2opsin (original) resolved '{original_name[:50]}...' to SMILES: {smiles[:50]}..."
                            )
                    else:
                        failed_indices.append(idx)

                if failed_indices:
                    if logger:
                        logger.debug(f"Retrying {len(failed_indices)} failed names with normalized versions")

                    normalized_names = [iupac_data[idx][1] for idx in failed_indices]

                    smiles_results_norm = py2opsin_func(
                        chemical_name=normalized_names,
                        output_format="SMILES",
                        allow_bad_stereo=True
                    )

                    inchikey_results_norm = py2opsin_func(
                        chemical_name=normalized_names,
                        output_format="StdInChIKey",
                        allow_bad_stereo=True
                    )

                    if not isinstance(smiles_results_norm, list):
                        smiles_results_norm = [smiles_results_norm]
                    if not isinstance(inchikey_results_norm, list):
                        inchikey_results_norm = [inchikey_results_norm]

                    for retry_idx, orig_idx in enumerate(failed_indices):
                        smiles = smiles_results_norm[retry_idx] if retry_idx < len(smiles_results_norm) else False
                        inchikey = inchikey_results_norm[retry_idx] if retry_idx < len(inchikey_results_norm) else False

                        if smiles and smiles is not False:
                            original_name, normalized_name, valid_data_idx = iupac_data[orig_idx]
                            item = valid_data[valid_data_idx]
                            item["molecule_smiles"] = smiles
                            item["smiles_source"] = "py2opsin_normalized"
                            enriched_py2opsin_normalized += 1

                            if inchikey and inchikey is not False:
                                item["molecule_inchikey"] = inchikey

                            if logger:
                                logger.debug(
                                    f"py2opsin (normalized) resolved '{original_name[:50]}...' "
                                    f"(normalized: '{normalized_name[:50]}...') to SMILES: {smiles[:50]}..."
                                )

                still_failed_indices = []
                for idx, (original_name, normalized_name, valid_data_idx) in enumerate(iupac_data):
                    item = valid_data[valid_data_idx]
                    if "molecule_smiles" not in item:
                        still_failed_indices.append(idx)

                if still_failed_indices:
                    if logger:
                        logger.debug(f"Stage 3: Generating bracket-fix variants for {len(still_failed_indices)} remaining failed names")

                    all_variants = []
                    variant_mapping = []

                    for still_idx, orig_idx in enumerate(still_failed_indices):
                        original_name, normalized_name, valid_data_idx = iupac_data[orig_idx]
                        variants_set = set()

                        fixed_add = fix_balanced_by_adding(normalized_name)
                        if fixed_add and fixed_add != normalized_name:
                            variants_set.add(fixed_add)

                        removal_variants = fix_balanced_by_removing_variants(normalized_name)
                        for variant in removal_variants:
                            if variant and variant != normalized_name:
                                variants_set.add(variant)

                        for variant in sorted(variants_set, key=str):
                            all_variants.append(variant)
                            variant_mapping.append((still_idx, variant))

                    if all_variants:
                        if logger:
                            logger.debug(f"Stage 3: Generated {len(all_variants)} variants for {len(still_failed_indices)} names, sending batch to py2opsin")

                        smiles_results_fix = py2opsin_func(
                            chemical_name=all_variants,
                            output_format="SMILES",
                            allow_bad_stereo=True
                        )

                        if not isinstance(smiles_results_fix, list):
                            smiles_results_fix = [smiles_results_fix]

                        resolved_still_indices = set()
                        successful_variants = []

                        for var_idx, (still_idx, variant_text) in enumerate(variant_mapping):
                            if still_idx in resolved_still_indices:
                                continue

                            smiles = smiles_results_fix[var_idx] if var_idx < len(smiles_results_fix) else False
                            if smiles and smiles is not False:
                                successful_variants.append((still_idx, variant_text, smiles))
                                resolved_still_indices.add(still_idx)

                        if successful_variants:
                            successful_variant_names = [v[1] for v in successful_variants]
                            inchikey_results_fix = py2opsin_func(
                                chemical_name=successful_variant_names,
                                output_format="StdInChIKey",
                                allow_bad_stereo=True
                            )
                            if not isinstance(inchikey_results_fix, list):
                                inchikey_results_fix = [inchikey_results_fix]

                            for succ_idx, (still_idx, variant_text, smiles) in enumerate(successful_variants):
                                orig_idx = still_failed_indices[still_idx]
                                original_name, normalized_name, valid_data_idx = iupac_data[orig_idx]

                                item = valid_data[valid_data_idx]
                                item["molecule_smiles"] = smiles
                                item["smiles_source"] = "py2opsin_fix_iupac"
                                enriched_py2opsin_fix_iupac += 1

                                inchikey = inchikey_results_fix[succ_idx] if succ_idx < len(inchikey_results_fix) else False
                                if inchikey and inchikey is not False:
                                    item["molecule_inchikey"] = inchikey

                                if logger:
                                    logger.debug(
                                        f"fix_iupac (batch) resolved '{original_name[:50]}...' "
                                        f"(variant: '{variant_text[:50]}...') to SMILES: {smiles[:50]}..."
                                    )

                        if logger:
                            logger.debug(f"Stage 3: Resolved {len(successful_variants)} of {len(still_failed_indices)} names")

            except Exception as e:
                if logger:
                    logger.warning(f"py2opsin batch processing failed: {e}")
        else:
            if logger:
                logger.debug("No valid IUPAC names found for py2opsin processing")

    if doc.chemistry_nodes:
        chem_nodes_map_original = {}
        chem_nodes_map_normalized = {}

        for node in doc.chemistry_nodes:
            if not (node.chemistry_id and node.smiles and not getattr(node, 'is_scaffold', False)):
                continue

            chem_nodes_map_original[node.chemistry_id] = node

            normalized_id = node.chemistry_id
            if normalized_id.startswith("CHEM-US-"):
                number_part = normalized_id[8:]
                normalized_number = number_part.lstrip('0') or '0'
                normalized_id = f"CHEM-US-{normalized_number}"

                if normalized_id != node.chemistry_id:
                    chem_nodes_map_normalized[normalized_id] = node

        if chem_nodes_map_original or chem_nodes_map_normalized:
            for item in valid_data:
                if "molecule_smiles" in item:
                    continue

                chem_id = item.get("chemical_id")
                if not chem_id:
                    continue

                node = None
                search_method = None

                normalized_chem_id = chem_id
                if normalized_chem_id.startswith("CHEM-US-"):
                    number_part = normalized_chem_id[8:]
                    normalized_number = number_part.lstrip('0') or '0'
                    normalized_chem_id = f"CHEM-US-{normalized_number}"

                    if normalized_chem_id != chem_id:
                        node = chem_nodes_map_normalized.get(normalized_chem_id)
                        if node:
                            search_method = "normalized"

                if not node:
                    node = chem_nodes_map_original.get(chem_id)
                    if node:
                        search_method = "original"

                if node and node.smiles:
                    item["molecule_smiles"] = node.smiles
                    if hasattr(node, 'inchikey') and node.inchikey:
                        item["molecule_inchikey"] = node.inchikey
                    item["smiles_source"] = "table_chemistry_tag"
                    enriched_chemistry_node += 1

                    if logger:
                        logger.debug(
                            f"Enriched '{item.get('compound') or item.get('compound_IUPAC_name')}' with SMILES "
                            f"from ChemistryNode {chem_id} (found via {search_method} ID)"
                        )

    total_enriched = enriched_py2opsin + enriched_py2opsin_normalized + enriched_py2opsin_fix_iupac + enriched_chemistry_node
    if total_enriched > 0 and logger:
        logger.info(
            f"Simple Agent 1 enriched {total_enriched}/{len(valid_data)} "
            f"measures with SMILES (py2opsin: {enriched_py2opsin}, "
            f"py2opsin_normalized: {enriched_py2opsin_normalized}, "
            f"py2opsin_fix_iupac: {enriched_py2opsin_fix_iupac}, "
            f"ChemistryNode: {enriched_chemistry_node})"
        )
