import re
from typing import Dict, List

from patent_processor import PatentDocument

from .compound_alias import extract_compound_number, normalize_compound_alias


def merge_with_assay_id(
    stage1_data: List[Dict],
    stage2_data: List[Dict],
    stage3_data: List[Dict],
    doc: PatentDocument,
    logger=None,
) -> List[Dict]:
    """Merge Stage 1/2/3 rows using the extractor's established behavior."""
    assay_map = {}
    for row in stage1_data:
        aid = (row.get("assay_id") or "").strip()
        if aid:
            assay_map[aid] = {
                "protein_target_name": row.get("protein_target_name", ""),
                "is_complex": row.get("is_complex", "N"),
                "protein_modification": row.get("protein_modification", ""),
                "organism": row.get("organism", ""),
                "assay": row.get("assay", ""),
                "assay_description": row.get("assay_description", ""),
                "stage1_reasoning": row.get("reasoning", ""),
                "extreme_conditions": row.get("extreme_conditions", "U")
            }

    if logger:
        logger.info(f"Created assay map with {len(assay_map)} unique assays from Stage 1")

    try:
        from pipeline.postprocess_integration import create_fuzzy_alias_mapping, apply_alias_mapping_to_stage2

        alias_mapping = create_fuzzy_alias_mapping(stage2_data, stage3_data)
        if alias_mapping:
            stage2_data = apply_alias_mapping_to_stage2(stage2_data, alias_mapping)
            if logger:
                logger.info(f"Applied fuzzy alias mapping: {len(alias_mapping)} aliases updated")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to apply fuzzy alias mapping: {e}")

    scaffold_map = {}
    scaffold_chem_num_map = {}
    if doc and hasattr(doc, 'chemistry_nodes') and doc.chemistry_nodes:
        for node in doc.chemistry_nodes:
            if getattr(node, 'is_scaffold', False):
                if node.chemistry_id:
                    scaffold_map[node.chemistry_id] = True
                    if logger:
                        logger.debug(f"Found scaffold node: {node.chemistry_id}")
                if node.chem_num:
                    scaffold_chem_num_map[node.chem_num] = True
                    normalized_chem_num = node.chem_num.lstrip('0') or '0'
                    if normalized_chem_num != node.chem_num:
                        scaffold_chem_num_map[normalized_chem_num] = True
                    if logger:
                        logger.debug(f"Found scaffold node with chem_num: {node.chem_num} (normalized: {normalized_chem_num})")

    compounds_map = {}
    compounds_map_normalized = {}
    compounds_map_by_number = {}
    number_conflicts = set()

    for item in stage3_data:
        alias = (item.get("compound") or "").strip()
        if alias:
            compound_data = {
                "compound": item.get("compound") or "",
                "compound_IUPAC_name": item.get("compound_IUPAC_identifier") or "",
                "chemical_id": item.get("chemical_id") or ""
            }
            compounds_map[alias] = compound_data
            normalized = normalize_compound_alias(alias)
            compounds_map_normalized[normalized] = compound_data

            number = extract_compound_number(alias)
            if number:
                if number in compounds_map_by_number:
                    number_conflicts.add(number)
                    if logger:
                        logger.debug(f"Number conflict detected: '{number}' appears in multiple compounds")
                else:
                    has_conflict = False
                    for existing_number in list(compounds_map_by_number.keys()):
                        if number.isdigit() and existing_number.isdigit():
                            if (number.startswith(existing_number) or existing_number.startswith(number)) and number != existing_number:
                                number_conflicts.add(number)
                                number_conflicts.add(existing_number)
                                has_conflict = True
                                if logger:
                                    logger.debug(f"Number substring conflict: '{number}' vs '{existing_number}'")

                    if not has_conflict:
                        compounds_map_by_number[number] = compound_data

    for conflict_number in number_conflicts:
        if conflict_number in compounds_map_by_number:
            del compounds_map_by_number[conflict_number]

    if logger:
        logger.info(
            f"Created compounds map with {len(compounds_map)} entries (exact), "
            f"{len(compounds_map_normalized)} normalized, and {len(compounds_map_by_number)} by number "
            f"({len(number_conflicts)} numbers excluded due to conflicts)"
        )

    merged = []
    skipped_compound_count = 0
    skipped_compound_aliases = []
    scaffold_skipped_count = 0
    hallucinated_assay_count = 0

    for bioactivity in stage2_data:
        compound_alias = (bioactivity.get("compound") or "").strip()
        assay_id = (bioactivity.get("assay_id") or "").strip()

        compound_data = None

        if compound_alias:
            if compound_alias in compounds_map:
                compound_data = compounds_map[compound_alias]
            else:
                normalized_alias = normalize_compound_alias(compound_alias)
                if normalized_alias in compounds_map_normalized:
                    compound_data = compounds_map_normalized[normalized_alias]
                    if logger:
                        logger.debug(f"Matched compound alias '{compound_alias}' using normalization (normalized: '{normalized_alias}')")
                else:
                    number = extract_compound_number(compound_alias)
                    if number and number in compounds_map_by_number:
                        compound_data = compounds_map_by_number[number]
                        if logger:
                            logger.debug(f"Matched compound alias '{compound_alias}' by number (number: '{number}')")

        if not compound_data:
            skipped_compound_aliases.append(compound_alias)
            skipped_compound_count += 1
            continue

        if not assay_id or assay_id not in assay_map:
            if logger:
                logger.warning(f"Assay ID '{assay_id}' from Stage 2 not found in Stage 1 (hallucination!), skipping")
            hallucinated_assay_count += 1
            continue

        assay_data = assay_map[assay_id]
        chemical_id = compound_data.get("chemical_id", "").strip()

        is_scaffold = False
        if chemical_id:
            if chemical_id in scaffold_map:
                is_scaffold = True
            else:
                chem_num_match = re.search(r'(\d+)$', chemical_id)
                if chem_num_match:
                    extracted_num = chem_num_match.group(1)
                    normalized_num = extracted_num.lstrip('0') or '0'
                    if (extracted_num in scaffold_chem_num_map or
                        normalized_num in scaffold_chem_num_map):
                        is_scaffold = True

        if is_scaffold:
            if logger:
                logger.debug(f"Skipping scaffold (incomplete molecule) with chemical_id='{chemical_id}'")
            scaffold_skipped_count += 1
            continue

        merged_item = {
            "compound": compound_data["compound"],
            "compound_IUPAC_name": compound_data["compound_IUPAC_name"],
            "chemical_id": compound_data["chemical_id"],
            "protein_target_name": assay_data["protein_target_name"],
            "is_complex": assay_data["is_complex"],
            "protein_modification": assay_data["protein_modification"],
            "organism": assay_data["organism"],
            "assay": assay_data["assay"],
            "assay_description": assay_data["assay_description"],
            "stage1_reasoning": assay_data["stage1_reasoning"],
            "extreme_conditions": assay_data["extreme_conditions"],
            "assay_id": assay_id,
            "binding_metric": bioactivity.get("binding_metric", ""),
            "value": bioactivity.get("value", ""),
            "unit": bioactivity.get("unit", ""),
            "stage2_reasoning": bioactivity.get("reasoning", "")
        }
        merged.append(merged_item)

    if skipped_compound_aliases and logger:
        logger.warning(
            f"{len(skipped_compound_aliases)} compound aliases from Stage 2 not found in Stage 3: "
            f"{', '.join(skipped_compound_aliases[:5])}{'...' if len(skipped_compound_aliases) > 5 else ''}"
        )

    if logger:
        logger.info(
            f"Merged {len(merged)} bioactivity records from Stage 1+2+3 "
            f"(skipped: {skipped_compound_count} unmatched compounds, "
            f"{hallucinated_assay_count} hallucinated assay_ids, "
            f"{scaffold_skipped_count} scaffolds)"
        )
    return merged
