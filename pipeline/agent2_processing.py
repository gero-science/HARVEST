"""Small helpers for Agent 2 patent processing."""

import json
import logging
import time
import traceback

from .accounting import add_usage_to_global_agent
from .alias_resolution import (
    ACCEPTED_EXTRACTOR_SOURCE,
    UNRESOLVED_MISSING_SMILES_REASON,
    bypass_agent2_alias_resolution,
    empty_agent2_usage,
    resolve_aliases_for_patent,
)
from .error_handling import classify_error


def count_valid_chemistry_nodes(document):
    chemistry_nodes = document.chemistry_nodes if hasattr(document, "chemistry_nodes") else []
    valid_nodes = 0
    for node in chemistry_nodes:
        if hasattr(node, "molecule_smiles") and node.molecule_smiles:
            valid_nodes += 1
    return chemistry_nodes, valid_nodes


def build_agent2_preprocess_debug_info(patent_id, measures_list, chemistry_nodes, valid_nodes):
    return {
        "patent_id": patent_id,
        "measures_count": len(measures_list),
        "chemistry_nodes_count": len(chemistry_nodes),
        "chemistry_nodes_with_smiles": valid_nodes,
        "timestamp": time.time(),
    }


def log_agent2_usage(patent_id, agent2_usage):
    logging.debug(
        f"[Agent2] Patent {patent_id}: requests={agent2_usage.get('request_count', 0)}, "
        f"tokens={agent2_usage.get('total_tokens', 0)}, cost=${agent2_usage.get('cost', 0):.4f}"
    )
    logging.debug(
        f"[Agent2] Patent {patent_id}: "
        f"completion_tokens_per_request={agent2_usage.get('completion_tokens_per_request', [])[:5]}..."
    )
    logging.debug(
        f"[Agent2] Patent {patent_id}: "
        f"max_completion_tokens_per_request={agent2_usage.get('max_completion_tokens_per_request', 0)}"
    )
    logging.debug(f"[Agent2] Patent {patent_id}: pyopsin_filtered={agent2_usage.get('pyopsin_filtered_count', 0)}")


def update_agent2_usage_stats(global_stats, agent2_usage):
    add_usage_to_global_agent(global_stats, "agent2", agent2_usage)
    global_stats["agent2"]["pyopsin_filtered_count"] = (
        global_stats["agent2"].get("pyopsin_filtered_count", 0)
        + agent2_usage.get("pyopsin_filtered_count", 0)
    )


def calculate_molecule_resolution_stats(resolved):
    molecules_with_smiles = 0
    verified_molecules = 0
    unverified_molecules = 0

    for molecule in resolved:
        if molecule.get("molecule_smiles"):
            molecules_with_smiles += 1

        resolution_source = molecule.get("agent2_resolution_source", "unresolved")
        if resolution_source.startswith("verified_"):
            verified_molecules += 1
        elif resolution_source == "llm_unverified":
            unverified_molecules += 1

    return {
        "molecules_with_smiles": molecules_with_smiles,
        "verified_molecules": verified_molecules,
        "unverified_molecules": unverified_molecules,
    }


def update_agent2_data_counts(global_stats, resolved, unresolved, molecule_stats):
    global_stats["data_counts"]["patents_processed"] += 1
    global_stats["data_counts"]["aliases_resolved"] += len(resolved)
    global_stats["data_counts"]["aliases_unresolved"] += len(unresolved)
    global_stats["data_counts"]["molecules_resolved_with_smiles"] += molecule_stats["molecules_with_smiles"]
    global_stats["data_counts"]["molecules_resolved_verified"] += molecule_stats["verified_molecules"]
    global_stats["data_counts"]["molecules_resolved_unverified"] += molecule_stats["unverified_molecules"]


def build_agent2_debug_results(patent_id, resolved, unresolved, processing_time, molecule_stats):
    return {
        "patent_id": patent_id,
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "molecules_with_smiles": molecule_stats["molecules_with_smiles"],
        "verified_molecules": molecule_stats["verified_molecules"],
        "unverified_molecules": molecule_stats["unverified_molecules"],
        "processing_time": processing_time,
        "resolution_sources": {
            source: count
            for source, count in [
                ("verified_both", sum(1 for r in resolved if r.get("agent2_resolution_source") == "verified_both")),
                ("verified_smiles", sum(1 for r in resolved if r.get("agent2_resolution_source") == "verified_smiles")),
                ("verified_name", sum(1 for r in resolved if r.get("agent2_resolution_source") == "verified_name")),
                ("llm_unverified", sum(1 for r in resolved if r.get("agent2_resolution_source") == "llm_unverified")),
            ]
        },
    }


def build_resolved_details(resolved, limit=5):
    resolved_details = []
    for item in resolved[:limit]:
        resolved_details.append({
            "original_name": item.get("original_name", "N/A"),
            "resolved_name": item.get("resolved_name", "N/A"),
            "molecule_smiles": item.get("molecule_smiles", "N/A"),
            "resolution_source": item.get("agent2_resolution_source", "N/A"),
        })
    return resolved_details


def build_agent2_error_detail(patent_id, error):
    return {
        "timestamp": time.time(),
        "agent": "agent2",
        "patent_id": patent_id,
        "error_type": classify_error(str(error)),
        "error_class": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc() if logging.getLogger().isEnabledFor(logging.DEBUG) else None,
    }


def append_error_detail(error_log_file, error_detail):
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(error_detail, ensure_ascii=False, default=str) + "\n")
