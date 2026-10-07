"""Tests for TSV part parsing/merging and header guarantees (bioactivity_extraction.tsv)."""

from bioactivity_extraction.tsv import strip_markdown_fences


class TestStripMarkdownFences:
    def test_removes_tsv_fence(self):
        text = "```tsv\nheader1\theader2\nval1\tval2\n```"
        assert strip_markdown_fences([text]) == ["header1\theader2\nval1\tval2"]

    def test_removes_plain_fence(self):
        text = "```\nheader1\theader2\nval1\tval2\n```"
        assert strip_markdown_fences([text]) == ["header1\theader2\nval1\tval2"]

    def test_no_fence_unchanged(self):
        text = "header1\theader2\nval1\tval2"
        assert strip_markdown_fences([text]) == [text]

    def test_empty_string(self):
        assert strip_markdown_fences([""]) == [""]

    def test_multiple_responses(self):
        texts = [
            "```tsv\nA\tB\n1\t2\n```",
            "3\t4",
            "```\n5\t6\n```",
        ]
        result = strip_markdown_fences(texts)
        assert result == ["A\tB\n1\t2", "3\t4", "5\t6"]

    def test_fence_with_language_tag(self):
        text = "```csv\nA,B\n1,2\n```"
        assert strip_markdown_fences([text]) == ["A,B\n1,2"]

    def test_only_fences_returns_empty(self):
        text = "```tsv\n```"
        assert strip_markdown_fences([text]) == [""]

    def test_nested_backticks_in_data_preserved(self):
        text = "```tsv\nname\tvalue\n`compound`\t10\n```"
        result = strip_markdown_fences([text])
        assert "`compound`" in result[0]


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
