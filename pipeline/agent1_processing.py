"""Small helpers for Agent 1 worker task processing."""

import logging
import time
import traceback

from .accounting import collect_agent1_usage, empty_agent1_usage
from .error_handling import classify_error


def enrich_agent1_data_from_document(extracted_data, document):
    if not document:
        return 0

    chem_nodes_map = {
        node.chem_num: node
        for node in document.chemistry_nodes
        if node.chem_num and node.smiles and not getattr(node, "is_scaffold", False)
    }

    enriched_count = 0
    for item in extracted_data:
        chem_id = item.get("chemical_id")
        if chem_id:
            normalized_id = str(chem_id).lstrip("0")

            node = chem_nodes_map.get(chem_id) or chem_nodes_map.get(normalized_id)

            if node and node.smiles:
                item["molecule_smiles"] = node.smiles
                if hasattr(node, "inchikey") and node.inchikey:
                    item["molecule_inchikey"] = node.inchikey
                item["smiles_source"] = "table_chemistry_tag"
                enriched_count += 1

                logging.debug(
                    f"Enriched '{item.get('molecule_name')}' with SMILES "
                    f"from chemical_id {chem_id}"
                )

    return enriched_count


async def process_agent1_task(extractor, task, output_dir=None):
    result = await extractor.async_process_document(task.chunk, output_dir=output_dir)
    extracted_data = result.get("final_data", [])

    document = task.document if hasattr(task, "document") else None
    enriched_count = enrich_agent1_data_from_document(extracted_data, document)

    if enriched_count > 0:
        logging.info(
            f"Agent 1 enriched {enriched_count}/{len(extracted_data)} "
            f"measures with SMILES from table chemistry tags"
        )

    agent1_usage = collect_agent1_usage(result)

    logging.debug(
        f"[Agent1] Patent {task.patent_id}: requests={agent1_usage.get('request_count', 0)}, "
        f"tokens={agent1_usage.get('total_tokens', 0)}, cost=${agent1_usage.get('cost', 0):.4f}"
    )

    zip_file_path = getattr(task, "zip_file_path", "")
    payload = (
        task.patent_id,
        extracted_data,
        task.total_chunks_in_patent,
        task.patent_text,
        agent1_usage,
        task.document,
        {},
        None,
        zip_file_path,
    )

    logging.info(f"Extracted {len(extracted_data)} data points from 1 chunks.")
    return payload


def build_agent1_error_detail(task, error):
    error_type = classify_error(str(error))
    return error_type, {
        "timestamp": time.time(),
        "agent": "agent1",
        "patent_id": task.patent_id,
        "error_type": error_type,
        "error_class": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc() if logging.getLogger().isEnabledFor(logging.DEBUG) else None,
    }


def build_agent1_error_payload(task, worker_error_stats, error_detail):
    zip_file_path = getattr(task, "zip_file_path", "")
    return (
        task.patent_id,
        [],
        task.total_chunks_in_patent,
        task.patent_text,
        empty_agent1_usage(),
        task.document,
        worker_error_stats.copy(),
        error_detail,
        zip_file_path,
    )
