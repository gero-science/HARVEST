"""Usage accounting helpers for pipeline workers."""

from collections import defaultdict

from .utils import aggregate_usage_stats


USAGE_TOTAL_KEYS = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "request_count"]


def empty_usage_stats():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "request_count": 0,
        "token_per_request": [],
        "max_tokens_per_request": 0,
        "completion_tokens_per_request": [],
        "max_completion_tokens_per_request": 0,
    }


def empty_agent1_usage():
    return empty_usage_stats()


def collect_agent1_usage(result):
    agent1_usage_list = []
    for artifact in result.get("debug_artifacts", []):
        usage_list = artifact.get("usage_stats_list")
        if usage_list:
            agent1_usage_list.extend(usage_list)
        else:
            usage = artifact.get("usage_stats")
            if usage:
                agent1_usage_list.append(usage)

    return aggregate_usage_stats(agent1_usage_list)


def add_usage_totals(target, usage):
    for key in USAGE_TOTAL_KEYS:
        target[key] += usage.get(key, 0)


def merge_request_token_lists(target, usage):
    target["token_per_request"].extend(usage.get("token_per_request", []))
    target["completion_tokens_per_request"].extend(usage.get("completion_tokens_per_request", []))


def update_usage_maxima(target, usage):
    if usage.get("max_tokens_per_request", 0) > target["max_tokens_per_request"]:
        target["max_tokens_per_request"] = usage.get("max_tokens_per_request", 0)
    if usage.get("max_completion_tokens_per_request", 0) > target["max_completion_tokens_per_request"]:
        target["max_completion_tokens_per_request"] = usage.get("max_completion_tokens_per_request", 0)


def add_agent1_usage_for_patent(agent1_usage_per_patent, patent_id, agent1_usage):
    patent_usage = agent1_usage_per_patent[patent_id]
    add_usage_totals(patent_usage, agent1_usage)
    merge_request_token_lists(patent_usage, agent1_usage)
    update_usage_maxima(patent_usage, agent1_usage)


def create_agent1_usage_store():
    return defaultdict(empty_agent1_usage)


def create_global_stats():
    return {
        "agent1": empty_usage_stats(),
        "agent2": {
            **empty_usage_stats(),
            "pyopsin_filtered_count": 0,
        },
        "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
        "data_counts": {
            "patents_processed": 0,
            "bioactivity_data_found": 0,
            "aliases_resolved": 0,
            "aliases_unresolved": 0,
            "molecules_resolved_with_smiles": 0,
            "molecules_resolved_verified": 0,
            "molecules_resolved_unverified": 0,
        },
    }


def add_usage_to_global_agent(global_stats, agent_name, usage):
    add_usage_totals(global_stats[agent_name], usage)
    update_usage_maxima(global_stats[agent_name], usage)


def finalize_total_usage(global_stats):
    for key in ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]:
        global_stats["total"][key] = global_stats["agent1"][key] + global_stats["agent2"][key]
