"""Usage aggregation helpers for staged bioactivity extraction."""

from typing import Dict, List, Optional


def prepare_usage_stats_list(
    stage0_usage: Optional[Dict],
    stage1_usage: Optional[Dict],
    stage2_usage_list: List[Dict],
    stage3_usage_list: List[Dict],
    logger=None,
) -> List[Dict]:
    """
    Build per-stage usage stats for pipeline aggregation.
    """
    usage_list = []

    if stage0_usage:
        stage0_copy = {
            "prompt_tokens": stage0_usage.get("prompt_tokens", 0),
            "completion_tokens": stage0_usage.get("completion_tokens", 0),
            "total_tokens": stage0_usage.get("total_tokens", 0),
            "cost": stage0_usage.get("cost", 0),
            "_stage_name": "Stage 0: Cache XML",
        }
        usage_list.append(stage0_copy)

    if stage1_usage:
        stage1_copy = {
            "prompt_tokens": stage1_usage.get("prompt_tokens", 0),
            "completion_tokens": stage1_usage.get("completion_tokens", 0),
            "total_tokens": stage1_usage.get("total_tokens", 0),
            "cost": stage1_usage.get("cost", 0),
            "_stage_name": "Stage 1: Extract assays",
        }
        usage_list.append(stage1_copy)

    if stage2_usage_list:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0
        total_cached_tokens = 0

        for usage in stage2_usage_list:
            cached_tokens = 0
            if "prompt_tokens_details" in usage:
                details = usage["prompt_tokens_details"]
                cached_tokens = details.get("cached_tokens", 0)

            prompt_tokens = usage.get("prompt_tokens", 0) - cached_tokens
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_cost += usage.get("cost", 0)
            total_cached_tokens += cached_tokens

        stage2_copy = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cost": total_cost,
            "_stage_name": f"Stage 2: Extract bioactivity ({len(stage2_usage_list)} parts)",
            "_cached_tokens": total_cached_tokens,
        }
        usage_list.append(stage2_copy)

    if stage3_usage_list:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0
        total_cached_tokens = 0

        for usage in stage3_usage_list:
            cached_tokens = 0
            if "prompt_tokens_details" in usage:
                details = usage["prompt_tokens_details"]
                cached_tokens = details.get("cached_tokens", 0)

            prompt_tokens = usage.get("prompt_tokens", 0) - cached_tokens
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_cost += usage.get("cost", 0)
            total_cached_tokens += cached_tokens

        stage3_copy = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cost": total_cost,
            "_stage_name": f"Stage 3: Extract compounds ({len(stage3_usage_list)} parts)",
            "_cached_tokens": total_cached_tokens,
        }
        usage_list.append(stage3_copy)

    total_cached = sum(s.get("_cached_tokens", 0) for s in usage_list)
    if total_cached > 0 and logger:
        total_prompt_raw = sum(s.get("prompt_tokens", 0) for s in usage_list) + total_cached
        cache_savings_percent = round(total_cached / total_prompt_raw * 100, 1) if total_prompt_raw > 0 else 0
        logger.info(f"Cache hit! Saved {total_cached} tokens ({cache_savings_percent}%)")

    return usage_list
