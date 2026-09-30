"""Tests for _filter_stage1_for_stage2: slice of 5 required columns + dedup (bioactivity_extraction.stage1_filter)."""

NEEDED = ["assay_id", "protein_target_name", "protein_modification", "assay_description", "organism"]

FULL_HEADER = (
    "assay_id\tprotein_target_name\tis_complex\tprotein_modification"
    "\tassay_description\treasoning\tassay\torganism\textreme_conditions"
)


def _row(*cols: str) -> str:
    return "\t".join(cols)


def test_empty_input_returns_empty(make_agent):
    assert make_agent()._filter_stage1_for_stage2("") == ""
    assert make_agent()._filter_stage1_for_stage2("   ") == ""


def test_keeps_only_needed_columns_in_order(make_agent):
    data = _row("A1", "EGFR", "N", "wild type", "inhibition", "p1", "biochemical", "human", "N")
    out = make_agent()._filter_stage1_for_stage2(FULL_HEADER + "\n" + data)

    lines = out.split("\n")
    assert lines[0] == "\t".join(NEEDED)
    assert lines[1] == "A1\tEGFR\twild type\tinhibition\thuman"


def test_deduplicates_identical_filtered_rows(make_agent):
    data = _row("A1", "EGFR", "N", "wild type", "inhibition", "p1", "biochemical", "human", "N")
    out = make_agent()._filter_stage1_for_stage2(FULL_HEADER + "\n" + data + "\n" + data)

    lines = out.split("\n")
    assert len(lines) == 2  # header + 1 unique row


def test_missing_column_becomes_empty_value(make_agent):
    # Header without organism → its value is empty in output, header still has 5 columns
    header = (
        "assay_id\tprotein_target_name\tis_complex\tprotein_modification"
        "\tassay_description\treasoning\tassay\textreme_conditions"
    )
    data = _row("A1", "EGFR", "N", "wild type", "inhibition", "p1", "biochemical", "N")
    out = make_agent()._filter_stage1_for_stage2(header + "\n" + data)

    lines = out.split("\n")
    assert lines[0] == "\t".join(NEEDED)
    assert lines[1].split("\t") == ["A1", "EGFR", "wild type", "inhibition", ""]
