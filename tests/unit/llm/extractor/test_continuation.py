"""Tests for continuation control flow (bioactivity_extraction.continuation)."""

import asyncio

import pytest

from llm.extractor import BioactivityExtractor


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def async_call_llm(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_request_continuation_stops_without_length_finish_reason(make_agent):
    agent = make_agent()
    agent.llm = FakeLLM([
        ("row1\n", {"finish_reason": "stop", "prompt_tokens": 10}),
    ])
    messages = [{"role": "user", "content": "prompt"}]

    responses, usage = asyncio.run(
        agent._request_continuation(messages, "US1", "Stage 2", max_continuations=1)
    )

    assert responses == ["row1\n"]
    assert usage == [{"finish_reason": "stop", "prompt_tokens": 10}]
    assert messages == [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "row1\n"},
    ]


def test_request_continuation_appends_prompt_and_response_for_truncated_output(make_agent):
    agent = make_agent()
    agent.llm = FakeLLM([
        ("part1\n", {"finish_reason": "length", "prompt_tokens": 10}),
        ("part2\n", {"finish_reason": "stop", "prompt_tokens": 20}),
    ])
    messages = [{"role": "user", "content": "prompt"}]

    responses, usage = asyncio.run(
        agent._request_continuation(messages, "US1", "Stage 2", max_continuations=1)
    )

    assert responses == ["part1\n", "part2\n"]
    assert usage == [
        {"finish_reason": "length", "prompt_tokens": 10},
        {"finish_reason": "stop", "prompt_tokens": 20},
    ]
    assert messages == [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "part1\n"},
        {"role": "user", "content": BioactivityExtractor.CONTINUATION_PROMPT},
        {"role": "assistant", "content": "part2\n"},
    ]


def test_request_continuation_removes_failed_continuation_prompt(make_agent):
    agent = make_agent()
    agent.llm = FakeLLM([
        ("part1\n", {"finish_reason": "length", "prompt_tokens": 10}),
        (None, {"finish_reason": "stop", "prompt_tokens": 20}),
    ])
    messages = [{"role": "user", "content": "prompt"}]

    responses, usage = asyncio.run(
        agent._request_continuation(messages, "US1", "Stage 2", max_continuations=1)
    )

    assert responses == ["part1\n"]
    assert usage == [{"finish_reason": "length", "prompt_tokens": 10}]
    assert messages == [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "part1\n"},
    ]


def test_request_continuation_raises_on_missing_initial_response(make_agent):
    agent = make_agent()
    agent.llm = FakeLLM([(None, {"finish_reason": "stop"})])
    messages = [{"role": "user", "content": "prompt"}]

    with pytest.raises(RuntimeError, match="Stage 2 failure"):
        asyncio.run(agent._request_continuation(messages, "US1", "Stage 2"))

    assert messages == [{"role": "user", "content": "prompt"}]


def test_improved_continuation_wrapper_reports_malformed_continuation():
    llm = FakeLLM([
        ("a\tb\n1\t2\n", {"finish_reason": "length", "completion_tokens": 5}),
        ("wrong\tcolumn\tcount\n", {"finish_reason": "stop", "completion_tokens": 5}),
    ])
    agent = BioactivityExtractor(config=object(), llm=llm)
    messages = [{"role": "user", "content": "prompt"}]

    responses, usage, diagnostics = asyncio.run(
        agent.request_continuation(
            messages=messages,
            patent_id="US1",
            stage_name="Stage 2",
            expected_columns=2,
        )
    )

    assert responses == ["a\tb\n1\t2\n"]
    assert usage == [{"finish_reason": "length", "completion_tokens": 5}]
    assert diagnostics["stopped_early"] is True
    assert diagnostics["failed_responses"] == [
        {
            "continuation_num": 1,
            "full_response": "wrong\tcolumn\tcount\n",
            "reason": "No valid lines found (all 1 lines have wrong column count)",
        }
    ]
