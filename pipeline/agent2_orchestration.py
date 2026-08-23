"""Orchestration for processing one patent through Agent 2."""

import asyncio
import logging
import time

from .agent2_processing import (
    append_error_detail,
    build_agent2_debug_results,
    build_agent2_error_detail,
    build_agent2_preprocess_debug_info,
    build_resolved_details,
    calculate_molecule_resolution_stats,
    count_valid_chemistry_nodes,
    empty_agent2_usage,
    log_agent2_usage,
    resolve_aliases_for_patent,
    update_agent2_data_counts,
    update_agent2_usage_stats,
)
from .artifacts import save_patent_result_artifacts
from .debug_artifacts import save_debug_data
from .statistics import save_patent_statistics


def _empty_agent2_usage_on_error():
    return empty_agent2_usage()


def _log_preprocess_debug(patent_id, measures_list, chemistry_nodes, valid_nodes, debug_mode):
    if not debug_mode:
        return

    debug_info = build_agent2_preprocess_debug_info(
        patent_id,
        measures_list,
        chemistry_nodes,
        valid_nodes,
    )
    logging.debug(f"[DEBUG] Alias resolution pre-process info for {patent_id}: {debug_info}")


async def _resolve_and_account_aliases(
    patent_id,
    measures_list,
    global_stats,
    debug_mode,
):
    resolved, unresolved, processing_time, agent2_usage = await resolve_aliases_for_patent(
        measures_list=measures_list,
    )

    log_agent2_usage(patent_id, agent2_usage)
    update_agent2_usage_stats(global_stats, agent2_usage)

    molecule_stats = calculate_molecule_resolution_stats(resolved)
    update_agent2_data_counts(global_stats, resolved, unresolved, molecule_stats)

    molecules_with_smiles = molecule_stats["molecules_with_smiles"]
    verified_molecules = molecule_stats["verified_molecules"]
    unverified_molecules = molecule_stats["unverified_molecules"]

    if molecules_with_smiles > 0 or verified_molecules > 0:
        logging.info(
            f"Patent {patent_id}: Resolved {len(resolved)} aliases, "
            f"{molecules_with_smiles} with SMILES, {verified_molecules} verified, "
            f"{unverified_molecules} unverified"
        )

    if debug_mode:
        debug_results = build_agent2_debug_results(
            patent_id,
            resolved,
            unresolved,
            processing_time,
            molecule_stats,
        )
        logging.debug(f"[DEBUG] Alias resolution results for {patent_id}: {debug_results}")

        resolved_details = build_resolved_details(resolved)
        if resolved_details:
            logging.debug(f"[DEBUG] Sample resolved aliases for {patent_id}: {resolved_details}")

    return resolved, unresolved, processing_time, agent2_usage, molecule_stats


async def _save_outputs_and_patent_stats(
    patent_id,
    resolved,
    unresolved,
    processing_time,
    agent2_usage,
    molecule_stats,
    document,
    output_dir,
    processed_patents_file,
    global_stats,
    agent1_usage_for_patent,
    debug_mode,
):
    await save_patent_result_artifacts(
        patent_id,
        resolved,
        unresolved,
        output_dir,
        processed_patents_file,
        to_thread=asyncio.to_thread,
        cdx_data=getattr(document, "cdx_data", None),
    )

    if debug_mode:
        await save_debug_data(patent_id, document, output_dir)

    await save_patent_statistics(
        output_dir=output_dir,
        patent_id=patent_id,
        agent1_usage=agent1_usage_for_patent,
        agent2_usage=agent2_usage,
        resolved_count=len(resolved),
        unresolved_count=len(unresolved),
        molecules_with_smiles=molecule_stats["molecules_with_smiles"],
        verified_molecules=molecule_stats["verified_molecules"],
        unverified_molecules=molecule_stats["unverified_molecules"],
        processing_time=processing_time,
        timestamp=time.time(),
    )


def _handle_agent2_error(patent_id, error, detailed_error_log):
    logging.error(f"Error during asynchronous patent processing {patent_id}: {error}")

    error_detail = build_agent2_error_detail(patent_id, error)
    append_error_detail(detailed_error_log, error_detail)

    return patent_id, 0, 0, _empty_agent2_usage_on_error()


async def process_patent_agent2(
    patent_id,
    measures_list,
    patent_text,
    document,
    output_dir,
    processed_patents_file,
    global_stats,
    detailed_error_log,
    agent1_usage_for_patent,
    debug_mode=False,
):
    """
    Tag one patent's rows with extractor provenance and run the downstream saves.

    The signature and return contract intentionally mirror pipeline.workers.
    """
    del patent_text  # Preserved in the public signature for compatibility.

    try:
        logging.info(f"Starting ASYNCHRONOUS Agent 2 for patent {patent_id}")

        chemistry_nodes, valid_nodes = count_valid_chemistry_nodes(document)
        if chemistry_nodes:
            logging.info(
                f"Validating chemistry_nodes for {patent_id}: "
                f"{valid_nodes}/{len(chemistry_nodes)} nodes with SMILES"
            )

        _log_preprocess_debug(
            patent_id,
            measures_list,
            chemistry_nodes,
            valid_nodes,
            debug_mode,
        )

        resolved, unresolved, processing_time, agent2_usage, molecule_stats = await _resolve_and_account_aliases(
            patent_id=patent_id,
            measures_list=measures_list,
            global_stats=global_stats,
            debug_mode=debug_mode,
        )

        await _save_outputs_and_patent_stats(
            patent_id=patent_id,
            resolved=resolved,
            unresolved=unresolved,
            processing_time=processing_time,
            agent2_usage=agent2_usage,
            molecule_stats=molecule_stats,
            document=document,
            output_dir=output_dir,
            processed_patents_file=processed_patents_file,
            global_stats=global_stats,
            agent1_usage_for_patent=agent1_usage_for_patent,
            debug_mode=debug_mode,
        )

        logging.info(
            f"ASYNCHRONOUS Patent {patent_id} completed. "
            f"Resolved: {len(resolved)}, Unresolved: {len(unresolved)}"
        )
        return patent_id, len(resolved), len(unresolved), agent2_usage

    except Exception as e:
        return _handle_agent2_error(patent_id, e, detailed_error_log)
