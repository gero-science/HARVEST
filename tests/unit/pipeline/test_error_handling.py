"""Tests for pipeline.error_handling."""
import pytest

from pipeline.error_handling import classify_error, create_empty_error_stats


@pytest.mark.parametrize(
    "message,expected",
    [
        # llm_failure_errors — has highest priority
        ("LLM Failure", "llm_failure_errors"),
        ("llm failure: model returned empty", "llm_failure_errors"),
        ("No response received after all retry attempts", "llm_failure_errors"),
        ("NO RESPONSE RECEIVED AFTER ALL RETRY ATTEMPTS for patent X", "llm_failure_errors"),
        # rate_limit_errors
        ("Rate limit exceeded", "rate_limit_errors"),
        ("HTTP 429 Too Many Requests", "rate_limit_errors"),
        ("Too Many Requests, retry later", "rate_limit_errors"),
        # timeout_errors
        ("Connection timed out", "timeout_errors"),
        ("Request timeout after 30s", "timeout_errors"),
        ("HTTP 408 request timeout", "timeout_errors"),
        # api_errors — http codes or generic 'api'
        ("HTTP 500 Internal Server Error", "api_errors"),
        ("502 Bad Gateway", "api_errors"),
        ("503 service unavailable", "api_errors"),
        ("API returned 401 Unauthorized", "api_errors"),
        ("API returned 403 Forbidden", "api_errors"),
        # parse_errors
        ("JSON decode error: unexpected token", "parse_errors"),
        ("Failed to parse response", "parse_errors"),
        ("decode error in stream", "parse_errors"),
        # other_errors
        ("Random unexpected condition", "other_errors"),
        ("Disk full", "other_errors"),
        ("", "other_errors"),
    ],
)
def test_classify_error(message, expected):
    assert classify_error(message) == expected


def test_classify_error_priority_llm_failure_over_rate_limit():
    """LLM failure phrase wins even if 'rate limit' is also present."""
    assert (
        classify_error("LLM Failure: rate limit exceeded too many requests")
        == "llm_failure_errors"
    )


def test_classify_error_priority_rate_limit_over_api():
    """'rate limit' wins over generic 'api' / http code mention."""
    assert (
        classify_error("rate limit on the API endpoint")
        == "rate_limit_errors"
    )


def test_classify_error_case_insensitive():
    assert classify_error("RATE LIMIT EXCEEDED") == "rate_limit_errors"
    assert classify_error("Timeout") == "timeout_errors"


def test_create_empty_error_stats_keys_and_values():
    stats = create_empty_error_stats()
    expected_keys = {
        "api_errors",
        "timeout_errors",
        "rate_limit_errors",
        "parse_errors",
        "llm_failure_errors",
        "other_errors",
    }
    assert set(stats.keys()) == expected_keys
    assert all(v == 0 for v in stats.values())


def test_create_empty_error_stats_returns_fresh_dict():
    """Two calls must return independent dicts (no shared state)."""
    a = create_empty_error_stats()
    b = create_empty_error_stats()
    a["api_errors"] = 5
    assert b["api_errors"] == 0
