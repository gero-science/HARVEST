"""Tests for TSV part parsing/merging and header guarantees (bioactivity_extraction.tsv)."""


def test_merge_tsv_parts_uses_header_from_first_part_for_continuations(make_agent):
    parts = ["a\tb\n1\t2", "3\t4"]
    merged = make_agent()._merge_tsv_parts(parts, "stage2")

    assert merged == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_merge_tsv_parts_empty_list_returns_empty(make_agent):
    assert make_agent()._merge_tsv_parts([], "stage2") == []


def test_merge_tsv_parts_skips_blank_parts(make_agent):
    parts = ["a\tb\n1\t2", "   ", "3\t4"]
    merged = make_agent()._merge_tsv_parts(parts, "stage2")

    assert merged == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_merge_tsv_parts_drops_continuation_when_first_part_blank(make_agent):
    # First part empty → header not preserved → continuation is dropped
    parts = ["", "3\t4"]
    assert make_agent()._merge_tsv_parts(parts, "stage2") == []


def test_ensure_header_keeps_existing_header(make_agent):
    responses = ["compound\tvalue\nAspirin\t10"]
    out = make_agent()._ensure_header_in_tsv(responses, "compound\tvalue", "stage2")

    assert out[0].splitlines()[0] == "compound\tvalue"
    assert len(out[0].splitlines()) == 2


def test_ensure_header_prepends_missing_header(make_agent):
    responses = ["Aspirin\t10"]
    out = make_agent()._ensure_header_in_tsv(responses, "compound\tvalue", "stage2")

    assert out[0].splitlines()[0] == "compound\tvalue"
    assert out[0].splitlines()[1] == "Aspirin\t10"


def test_ensure_header_empty_responses_returns_empty(make_agent):
    assert make_agent()._ensure_header_in_tsv([], "compound\tvalue", "stage2") == []


def test_ensure_header_blank_first_response_unchanged(make_agent):
    responses = ["   "]
    out = make_agent()._ensure_header_in_tsv(responses, "compound\tvalue", "stage2")

    assert out == ["   "]
