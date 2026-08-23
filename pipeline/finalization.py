"""Finalization helpers for the async pipeline aggregator."""

import json
import logging
import os
import time


async def save_error_logs(output_dir, error_log_file, error_count, debug_mode):
    """Process JSONL error logs and create debug_problematic_patents.json."""
    if not debug_mode:
        return

    if not error_log_file.exists() or error_count == 0:
        logging.info("No errors recorded - error log files were not created")
        return

    logging.info(f"Recorded {error_count} errors in {error_log_file}")

    problematic_patents = {}
    with open(error_log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    error_entry = json.loads(line)
                    patent_id = error_entry.get("patent_id", "unknown")
                    if patent_id not in problematic_patents:
                        problematic_patents[patent_id] = {
                            "patent_id": patent_id,
                            "error_count": 0,
                            "error_types": set(),
                            "error_classes": set(),
                            "agents_with_errors": set(),
                            "errors": [],
                        }

                    problematic_patents[patent_id]["error_count"] += 1
                    problematic_patents[patent_id]["error_types"].add(error_entry.get("error_type", "unknown"))
                    problematic_patents[patent_id]["error_classes"].add(error_entry.get("error_class", "unknown"))
                    problematic_patents[patent_id]["agents_with_errors"].add(error_entry.get("agent", "unknown"))
                    problematic_patents[patent_id]["errors"].append({
                        "timestamp": error_entry.get("timestamp"),
                        "agent": error_entry.get("agent"),
                        "error_type": error_entry.get("error_type"),
                        "error_class": error_entry.get("error_class"),
                        "error_message": error_entry.get("error_message", "")[:200]
                        + ("..." if len(error_entry.get("error_message", "")) > 200 else ""),
                    })
                except json.JSONDecodeError:
                    continue

    for patent_data in problematic_patents.values():
        patent_data["error_types"] = list(patent_data["error_types"])
        patent_data["error_classes"] = list(patent_data["error_classes"])
        patent_data["agents_with_errors"] = list(patent_data["agents_with_errors"])

    if problematic_patents:
        problematic_file = os.path.join(output_dir, "debug_problematic_patents.json")
        with open(problematic_file, "w", encoding="utf-8") as f:
            json.dump(list(problematic_patents.values()), f, ensure_ascii=False, indent=2, default=str)
        logging.info(f"Saved list of {len(problematic_patents)} problematic patents to {problematic_file}")


def print_final_statistics(global_stats, start_time):
    """Log final token and data statistics."""
    logging.info("=" * 70)
    logging.info("FINAL TOKEN AND DATA USAGE STATISTICS")
    logging.info("=" * 70)
    logging.info("FOUND DATA:")
    logging.info(f"  - Patents processed: {global_stats['data_counts']['patents_processed']}")
    logging.info(f"  - Bioactivity data points found: {global_stats['data_counts']['bioactivity_data_found']}")
    logging.info(f"  - Aliases resolved: {global_stats['data_counts']['aliases_resolved']}")
    logging.info(f"  - Aliases unresolved: {global_stats['data_counts']['aliases_unresolved']}")
    if global_stats["data_counts"]["aliases_resolved"] + global_stats["data_counts"]["aliases_unresolved"] > 0:
        resolve_rate = (
            global_stats["data_counts"]["aliases_resolved"]
            / (global_stats["data_counts"]["aliases_resolved"] + global_stats["data_counts"]["aliases_unresolved"])
            * 100
        )
        logging.info(f"  - Alias resolution rate: {resolve_rate:.1f}%")

    logging.info(f"  - Molecules with SMILES strings: {global_stats['data_counts']['molecules_resolved_with_smiles']}")
    if global_stats["data_counts"]["aliases_resolved"] > 0:
        smiles_rate = (
            global_stats["data_counts"]["molecules_resolved_with_smiles"]
            / global_stats["data_counts"]["aliases_resolved"]
            * 100
        )
        logging.info(f"  - Structure resolution rate: {smiles_rate:.1f}%")
    logging.info(f"  - Verified resolutions: {global_stats['data_counts']['molecules_resolved_verified']}")
    logging.info(f"  - Unverified resolutions: {global_stats['data_counts']['molecules_resolved_unverified']}")
    if (
        global_stats["data_counts"]["molecules_resolved_verified"]
        + global_stats["data_counts"]["molecules_resolved_unverified"]
        > 0
    ):
        verified_rate = (
            global_stats["data_counts"]["molecules_resolved_verified"]
            / (
                global_stats["data_counts"]["molecules_resolved_verified"]
                + global_stats["data_counts"]["molecules_resolved_unverified"]
            )
            * 100
        )
        logging.info(f"  - Verified resolution rate: {verified_rate:.1f}%")
    logging.info("")
    logging.info("TOKEN USAGE:")

    logging.info("Agent 1 (Data extraction):")
    logging.info(f"  - LLM requests: {global_stats['agent1']['request_count']:,}")
    logging.info(f"  - Prompt tokens: {global_stats['agent1']['prompt_tokens']:,}")
    logging.info(f"  - Completion tokens: {global_stats['agent1']['completion_tokens']:,}")
    logging.info(f"  - Total tokens: {global_stats['agent1']['total_tokens']:,}")
    if global_stats["agent1"]["request_count"] > 0:
        avg_tokens = global_stats["agent1"]["total_tokens"] / global_stats["agent1"]["request_count"]
        avg_completion_tokens = global_stats["agent1"]["completion_tokens"] / global_stats["agent1"]["request_count"]
        logging.info(f"  - Average tokens per request: {avg_tokens:.1f}")
        logging.info(f"  - Average completion tokens per request: {avg_completion_tokens:.1f}")
    logging.info(f"  - Maximum tokens per request: {global_stats['agent1']['max_tokens_per_request']:,}")
    logging.info(f"  - Maximum completion tokens per request: {global_stats['agent1']['max_completion_tokens_per_request']:,}")
    logging.info(f"  - Cost: ${global_stats['agent1']['cost']:.4f}")
    logging.info("")

    logging.info("Agent 2 (Alias resolution):")
    logging.info(f"  - LLM requests: {global_stats['agent2']['request_count']:,}")
    logging.info(f"  - Prompt tokens: {global_stats['agent2']['prompt_tokens']:,}")
    logging.info(f"  - Completion tokens: {global_stats['agent2']['completion_tokens']:,}")
    logging.info(f"  - Total tokens: {global_stats['agent2']['total_tokens']:,}")
    if global_stats["agent2"]["request_count"] > 0:
        avg_tokens = global_stats["agent2"]["total_tokens"] / global_stats["agent2"]["request_count"]
        avg_completion_tokens = global_stats["agent2"]["completion_tokens"] / global_stats["agent2"]["request_count"]
        logging.info(f"  - Average tokens per request: {avg_tokens:.1f}")
        logging.info(f"  - Average completion tokens per request: {avg_completion_tokens:.1f}")
    logging.info(f"  - Maximum tokens per request: {global_stats['agent2']['max_tokens_per_request']:,}")
    logging.info(f"  - Maximum completion tokens per request: {global_stats['agent2']['max_completion_tokens_per_request']:,}")
    logging.info(f"  - Cost: ${global_stats['agent2']['cost']:.4f}")
    logging.info(f"  - PyOpsin filtered: {global_stats['agent2']['pyopsin_filtered_count']:,}")
    logging.info("")

    total_requests = global_stats["agent1"]["request_count"] + global_stats["agent2"]["request_count"]
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    logging.info("OVERALL TOTAL:")
    logging.info(f"  - Total LLM requests: {total_requests:,}")
    logging.info(f"  - Total tokens: {global_stats['total']['total_tokens']:,}")
    logging.info(f"  - Total cost: ${global_stats['total']['cost']:.4f}")
    if total_requests > 0:
        avg_cost_per_request = global_stats["total"]["cost"] / total_requests
        logging.info(f"  - Average cost per request: ${avg_cost_per_request:.6f}")
    logging.info(f"  - Total elapsed time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")
    if total_time > 0 and total_requests > 0:
        avg_time_per_request = total_time / total_requests
        logging.info(f"  - Average time per request: {avg_time_per_request:.3f} sec")
    logging.info("")

    resolved_aliases = global_stats["data_counts"]["aliases_resolved"]
    logging.info("KEY EFFICIENCY METRICS:")
    if resolved_aliases > 0:
        tokens_per_resolved = global_stats["total"]["total_tokens"] / resolved_aliases
        time_per_resolved = total_time / resolved_aliases
        cost_per_resolved = global_stats["total"]["cost"] / resolved_aliases
        logging.info(f"  - Tokens per resolved connection: {tokens_per_resolved:.1f}")
        logging.info(f"  - Time per connection: {time_per_resolved:.3f} sec")
        logging.info(f"  - Cost per connection: ${cost_per_resolved:.6f}")
        if time_per_resolved > 0:
            connections_per_hour = 3600 / time_per_resolved
            logging.info(f"  - Connections per hour (theoretical): {connections_per_hour:.0f}")
    else:
        logging.info("  - No connections found - cannot compute efficiency metrics")
    logging.info("=" * 70)
