"""Continuation request helpers for staged LLM extraction."""

from typing import Dict, List, Optional


async def request_continuation(
    llm,
    messages: List[Dict],
    patent_id: str,
    stage_name: str,
    continuation_prompt: str,
    max_continuations: int,
    logger,
) -> tuple[List[str], List[Dict]]:
    """
    Request continuation when a staged LLM response is truncated.

    The messages list is intentionally mutated in-place to preserve the current
    the extractor's behavior.
    """
    responses = []
    usage_stats = []

    logger.info(f"{stage_name}: making initial request for patent {patent_id}")
    response, usage = await llm.async_call_llm(
        patent_id=patent_id,
        messages=messages,
    )

    if not response:
        logger.error(f"{stage_name}: initial request failed for patent {patent_id}")
        raise RuntimeError(f"{stage_name} failure: No response received for patent {patent_id}")

    responses.append(response)
    if usage:
        usage_stats.append(usage)

    messages.append({"role": "assistant", "content": response})

    finish_reason = usage.get("finish_reason", "stop") if usage else "stop"
    response_lines = response.count("\n") + 1 if response else 0
    logger.info(
        f"{stage_name}: initial response received, finish_reason={finish_reason}, "
        f"length={len(response)} chars, lines={response_lines}"
    )

    continuation_count = 0
    while finish_reason == "length" and continuation_count < max_continuations:
        continuation_count += 1
        logger.warning(
            f"{stage_name}: response was truncated (finish_reason='length'), "
            f"requesting continuation {continuation_count}/{max_continuations}. "
            f"Prompt caching should apply to the existing context."
        )

        messages.append({"role": "user", "content": continuation_prompt})

        logger.debug(f"{stage_name}: continuation request, total messages in context: {len(messages)}")

        response, usage = await llm.async_call_llm(
            patent_id=patent_id,
            messages=messages,
        )

        if not response:
            logger.warning(f"{stage_name}: continuation {continuation_count} failed, stopping")
            messages.pop()
            break

        responses.append(response)
        if usage:
            usage_stats.append(usage)

        messages.append({"role": "assistant", "content": response})

        finish_reason = usage.get("finish_reason", "stop") if usage else "stop"
        response_lines = response.count("\n") + 1 if response else 0
        logger.info(
            f"{stage_name}: continuation {continuation_count} received, "
            f"finish_reason={finish_reason}, length={len(response)} chars, lines={response_lines}"
        )

    if finish_reason == "length":
        logger.warning(f"{stage_name}: reached max continuations ({max_continuations}), response may still be incomplete")

    logger.info(f"{stage_name}: completed with {len(responses)} total responses")
    return responses, usage_stats
