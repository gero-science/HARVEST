from llm.stage_formats import StageFormats


def read_first_line(path):
    return path.read_text(encoding="utf-8").splitlines()[0]


def test_stage_artifacts_are_written_with_stage_format_headers(tmp_path, make_agent):
    patent_id = "USUNITTESTA1"

    make_agent()._save_stage_responses(
        patent_id=patent_id,
        stage1_response=StageFormats.STAGE1_HEADER + "\nA1\tEGFR\tN\t\tassay\tp1\tbiochemical\thuman\tN",
        stage2_responses=[StageFormats.STAGE2_HEADER + "\nExample 1\tA1\tIC50\t10\tnM\tp2"],
        stage3_responses=[StageFormats.STAGE3_HEADER + "\nExample 1\tethanol\tp3\tCHEM-US-00001"],
        output_dir=str(tmp_path),
    )

    patent_dir = tmp_path / patent_id
    assert read_first_line(patent_dir / f"{patent_id}_agent1_stage1_targets.tsv") == StageFormats.STAGE1_HEADER
    assert read_first_line(patent_dir / f"{patent_id}_agent1_stage2_bioactivity.tsv") == StageFormats.STAGE2_HEADER
    assert read_first_line(patent_dir / f"{patent_id}_agent1_stage3_compounds.tsv") == StageFormats.STAGE3_HEADER


def test_stage3_cleaned_artifact_is_written_only_when_cleaned_data_is_passed(tmp_path, make_agent):
    patent_id = "USUNITTESTA1"
    agent = make_agent()

    agent._save_stage_responses(
        patent_id=patent_id,
        stage1_response=StageFormats.STAGE1_HEADER,
        stage2_responses=[StageFormats.STAGE2_HEADER],
        stage3_responses=[StageFormats.STAGE3_HEADER],
        output_dir=str(tmp_path),
    )
    cleaned_path = tmp_path / patent_id / f"{patent_id}_agent1_stage3_compounds_cleaned.tsv"
    assert not cleaned_path.exists()

    agent._save_stage_responses(
        patent_id=patent_id,
        stage1_response=StageFormats.STAGE1_HEADER,
        stage2_responses=[StageFormats.STAGE2_HEADER],
        stage3_responses=[StageFormats.STAGE3_HEADER],
        stage3_compounds_cleaned=[
            {
                "compound": "Example 1",
                "compound_IUPAC_identifier": "ethanol",
                "reasoning": "p3",
                "chemical_id": "CHEM-US-00001",
            }
        ],
        output_dir=str(tmp_path),
    )

    assert cleaned_path.exists()
    assert read_first_line(cleaned_path) == StageFormats.STAGE3_HEADER


def test_failed_artifacts_are_written_only_for_failed_continuation_rows(tmp_path, make_agent):
    patent_id = "USUNITTESTA1"

    make_agent()._save_stage_responses(
        patent_id=patent_id,
        stage1_response=StageFormats.STAGE1_HEADER,
        stage2_responses=[StageFormats.STAGE2_HEADER],
        stage3_responses=[StageFormats.STAGE3_HEADER],
        output_dir=str(tmp_path),
    )
    patent_dir = tmp_path / patent_id
    assert not (patent_dir / f"{patent_id}_agent1_stage2_failed.tsv").exists()
    assert not (patent_dir / f"{patent_id}_agent1_stage3_failed.tsv").exists()

    make_agent()._save_stage_responses(
        patent_id=patent_id,
        stage1_response=StageFormats.STAGE1_HEADER,
        stage2_responses=[StageFormats.STAGE2_HEADER],
        stage3_responses=[StageFormats.STAGE3_HEADER],
        stage2_failed=[{"continuation_num": 1, "reason": "bad columns", "invalid_lines": ["bad"]}],
        stage3_failed=[{"continuation_num": 1, "reason": "bad columns", "invalid_lines": ["bad"]}],
        output_dir=str(tmp_path),
    )

    assert (patent_dir / f"{patent_id}_agent1_stage2_failed.tsv").exists()
    assert (patent_dir / f"{patent_id}_agent1_stage3_failed.tsv").exists()


def test_empty_stage2_still_writes_stage_artifacts_without_resolved_json(tmp_path, make_agent):
    patent_id = "USUNITTESTA1"

    make_agent()._save_stage_responses(
        patent_id=patent_id,
        stage1_response=StageFormats.STAGE1_HEADER + "\nA1\tEGFR\tN\t\tassay\tp1\tbiochemical\thuman\tN",
        stage2_responses=[StageFormats.STAGE2_HEADER],
        stage3_responses=[StageFormats.STAGE3_HEADER],
        output_dir=str(tmp_path),
    )

    patent_dir = tmp_path / patent_id
    assert (patent_dir / f"{patent_id}_agent1_stage1_targets.tsv").exists()
    assert (patent_dir / f"{patent_id}_agent1_stage2_bioactivity.tsv").exists()
    assert (patent_dir / f"{patent_id}_agent1_stage3_compounds.tsv").exists()
    assert not (patent_dir / f"{patent_id}_resolved.json").exists()
