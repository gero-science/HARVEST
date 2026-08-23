"""Ligand and protein enrichment for BindingDB export rows."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict

from data_normalization.normalize_data import convert_mass_to_nM

from .artifacts import load_stage1_organisms
from .organisms import (
    HUMAN_DEFAULT_FLAG,
    HUMAN_DEFAULT_ORGANISM,
    HUMAN_FALLBACK_FLAG,
    align_human_default_scientific,
    classify_species_source,
    lookup_protein_data,
    normalize_organism,
    strip_human_protein_parts,
)

if TYPE_CHECKING:
    from enrich_data.ligand_info_extractor import LigandInfoExtractor


def process_one_patent_bindings(
    bindings: list[Dict[str, Any]],
    ligand_extractor: LigandInfoExtractor,
    resolved_proteins: Dict[str, str],
    stage1_organisms: Dict[str, str] | None = None,
) -> list[Dict[str, Any]]:
    """
    Process bindings for one patent.

    Proteins come from resolved_proteins (single_proteins.json);
    no online UniProt requests are made.

    stage1_organisms maps assay_id to the organism as Stage 1 extracted it and
    takes precedence over the row's own label, which legacy runs overwrote.
    """
    stage1_organisms = stage1_organisms or {}
    updated_bindings = []

    for result in bindings:
        molecule_name = result.get("molecule_name")
        target_name = (result.get("protein_target_name") or "").replace("\\", "")

        if not target_name:
            continue

        # Process ligand
        smiles = result.get("molecule_smiles", None)
        inchi_key = result.get("molecule_inchikey", None)

        if smiles:
            if not inchi_key:
                inchi_key = ligand_extractor.get_inchi_key_by_smiles(smiles)
        elif molecule_name:
            inchi_key, smiles = ligand_extractor.extract(molecule_name)

        # Do not drop records with inchi_key_cdx; they will be used as fallback
        if not inchi_key and not result.get("inchi_key_cdx"):
            continue

        if inchi_key:
            result["Ligand InChI Key"] = inchi_key
        if smiles:
            result["Ligand SMILES"] = smiles

        # Convert mass units to nM
        if result.get("needs_mw_conversion"):
            molecular_weight = result.get("molecular_weight")
            mw_source = "chemistry file" if molecular_weight else None

            if not molecular_weight and smiles:
                molecular_weight = ligand_extractor.get_molecular_weight(smiles)
                if molecular_weight:
                    result["molecular_weight"] = molecular_weight
                    mw_source = "SMILES calculation"
                    logging.debug(f"Calculated MW from SMILES for {molecule_name}: {molecular_weight:.2f}")

            if molecular_weight:
                value = result.get("value")
                unit = result.get("unit")

                converted_value = convert_mass_to_nM(value, unit, molecular_weight)
                if converted_value:
                    result["value"] = converted_value
                    result["original_unit_before_mw_conversion"] = unit
                    result["unit"] = "nM"
                    logging.info(
                        f"Converted {value} {unit} to {converted_value} nM "
                        f"(MW: {molecular_weight:.2f} g/mol from {mw_source})"
                    )
                else:
                    logging.warning(f"Failed to convert {value} {unit} to nM for molecule {molecule_name}")
            else:
                logging.warning(f"No molecular weight available for {molecule_name} (no MW in data, SMILES: {smiles})")

            del result["needs_mw_conversion"]

        # Process protein from resolved_proteins
        result["protein_sequence"] = None
        result["protein_accession"] = None
        result["protein_uniprot_id"] = None
        result["gene"] = None
        result["organism_scientific"] = None

        assay_id = (result.get("assay_id") or "").strip()
        if assay_id in stage1_organisms:
            text_organism = stage1_organisms[assay_id]
        else:
            text_organism = result.get("organism")

        if target_name in resolved_proteins:
            protein_data = lookup_protein_data(resolved_proteins[target_name], text_organism)
            if protein_data is not None:
                result["protein_sequence"] = protein_data.get("sequence")
                result["protein_accession"] = protein_data.get("accession")
                result["protein_uniprot_id"] = protein_data.get("uniprot_id")
                result["gene"] = protein_data.get("gene")
                result["organism_scientific"] = protein_data.get("organism_scientific")

        result["organism"] = normalize_organism(text_organism)
        species_source = classify_species_source(
            text_organism,
            result.get("organism_scientific"),
            result.get("protein_uniprot_id"),
        )

        if species_source == HUMAN_FALLBACK_FLAG:
            # The text named another species, so a human sequence is wrong here.
            # Only the gene survives when nothing non-human is left.
            (
                result["gene"],
                result["protein_uniprot_id"],
                result["protein_accession"],
                result["protein_sequence"],
            ) = strip_human_protein_parts(
                result.get("gene"),
                result.get("protein_uniprot_id"),
                result.get("protein_accession"),
                result.get("protein_sequence"),
            )
            species_source = classify_species_source(
                text_organism,
                result.get("organism_scientific"),
                result.get("protein_uniprot_id"),
            )

        if species_source == HUMAN_DEFAULT_FLAG:
            result["organism"] = HUMAN_DEFAULT_ORGANISM
            result["organism_scientific"] = align_human_default_scientific(
                result.get("organism_scientific"),
                result.get("protein_uniprot_id"),
            )
        updated_bindings.append(result)

    # Final processing - populate output fields
    final_bindings = []
    for result in updated_bindings:
        sequence = result.get("protein_sequence", None)
        accession = result.get("protein_accession", None)
        uniprot_id = result.get("protein_uniprot_id", None)

        if sequence:
            result["Sequence"] = sequence
        if accession:
            result["Target accession"] = accession
        if uniprot_id:
            result["UniProt ID"] = uniprot_id

        final_bindings.append(result)

    return final_bindings


def printable_result(result: Dict[str, Any]) -> Dict[str, Any]:
    printable_result = dict(result)
    for key in ["binding_context", "name_context", "agent1_response"]:
        if key in printable_result:
            del printable_result[key]
    return printable_result


def process_single_patent(
    patent_dir: str,
    ligand_extractor: LigandInfoExtractor,
    resolved_proteins: Dict[str, Any] = None,
    patterns: list[str] = None,
    bindings: list[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    """
    Process one patent by loading and enriching its bindings.

    Args:
        patent_dir: Path to the patent directory
        ligand_extractor: LigandInfoExtractor instance with caches
        resolved_proteins: Dictionary of resolved proteins
        patterns: List of file patterns to search for
        bindings: Preloaded normalized data (if None, loaded from files)

    Returns:
        List of processed bindings
    """
    if patterns is None:
        patterns = ["03_final_output.json", "_resolved.json"]

    if resolved_proteins is None:
        resolved_proteins = {}

    # If data was not provided, load it from the patent directory
    if bindings is None:
        bindings = []
        for root, dirs, files in os.walk(patent_dir):
            for file in files:
                if any(file.endswith(pat) for pat in patterns):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for dat in data:
                                    dat["patent_number"] = os.path.basename(patent_dir)
                                bindings.extend(data)
                    except Exception as e:
                        logging.error(f"Error loading {file_path}: {e}")

    if not bindings:
        return []

    # Process bindings with already-open extractors
    updated_bindings = process_one_patent_bindings(
        bindings,
        ligand_extractor,
        resolved_proteins,
        stage1_organisms=load_stage1_organisms(patent_dir),
    )

    return updated_bindings
