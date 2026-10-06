"""Tests for usage stats aggregation (bioactivity_extraction.usage)."""


def test_prepare_usage_stats_keeps_stage0_and_stage1_as_individual_requests(make_agent):
    usage = make_agent()._prepare_usage_stats_list(
        stage0_usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11, "cost": 0.1},
        stage1_usage={"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22, "cost": 0.2},
        stage2_usage_list=[],
        stage3_usage_list=[],
    )

    assert usage == [
        {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
            "cost": 0.1,
            "_stage_name": "Stage 0: Cache XML",
        },
        {
            "prompt_tokens": 20,
            "completion_tokens": 2,
            "total_tokens": 22,
            "cost": 0.2,
            "_stage_name": "Stage 1: Extract assays",
        },
    ]


def test_prepare_usage_stats_sums_stage2_and_stage3_minus_cached_tokens(make_agent):
    usage = make_agent()._prepare_usage_stats_list(
        stage0_usage=None,
        stage1_usage=None,
        stage2_usage_list=[
            {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "cost": 1.0,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
            {
                "prompt_tokens": 80,
                "completion_tokens": 8,
                "cost": 0.8,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        ],
        stage3_usage_list=[
            {"prompt_tokens": 70, "completion_tokens": 7, "cost": 0.7},
        ],
    )

    assert usage == [
        {
            "prompt_tokens": 120,
            "completion_tokens": 18,
            "total_tokens": 138,
            "cost": 1.8,
            "_stage_name": "Stage 2: Extract bioactivity (2 parts)",
            "_cached_tokens": 60,
        },
        {
            "prompt_tokens": 70,
            "completion_tokens": 7,
            "total_tokens": 77,
            "cost": 0.7,
            "_stage_name": "Stage 3: Extract compounds (1 parts)",
            "_cached_tokens": 0,
        },
    ]


def test_prepare_usage_stats_handles_missing_usage_fields_as_zero(make_agent):
    usage = make_agent()._prepare_usage_stats_list(
        stage0_usage={},
        stage1_usage={},
        stage2_usage_list=[{}],
        stage3_usage_list=[{}],
    )

    assert usage == [
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0,
            "_stage_name": "Stage 2: Extract bioactivity (1 parts)",
            "_cached_tokens": 0,
        },
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0,
            "_stage_name": "Stage 3: Extract compounds (1 parts)",
            "_cached_tokens": 0,
        },
    ]
