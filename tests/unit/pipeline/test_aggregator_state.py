from types import SimpleNamespace

from pipeline.aggregator_state import AggregatorState, parse_agent1_payload


def usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0, request_count=0):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "request_count": request_count,
        "token_per_request": [total_tokens] if total_tokens else [],
        "max_tokens_per_request": total_tokens,
        "completion_tokens_per_request": [completion_tokens] if completion_tokens else [],
        "max_completion_tokens_per_request": completion_tokens,
    }


def payload(
    patent_id="USUNITTESTA1",
    extracted_data=None,
    total_chunks=1,
    agent1_usage=None,
    zip_file_path="patent.zip",
    error_detail=None,
):
    return (
        patent_id,
        extracted_data if extracted_data is not None else [{"row": 1}],
        total_chunks,
        "patent text",
        agent1_usage or usage(total_tokens=10, request_count=1),
        SimpleNamespace(patent_id=patent_id),
        {},
        error_detail,
        zip_file_path,
    )


def error_payload(patent_id="USUNITTESTA1", total_chunks=1):
    """A chunk that came back as an Agent 1 failure: no data, an error detail."""
    return payload(
        patent_id=patent_id,
        extracted_data=[],
        total_chunks=total_chunks,
        error_detail={"error_type": "api_error", "detail": "boom"},
    )


def test_parse_agent1_payload_preserves_queue_tuple_contract():
    parsed = parse_agent1_payload(payload())

    assert parsed.patent_id == "USUNITTESTA1"
    assert parsed.extracted_data == [{"row": 1}]
    assert parsed.total_chunks_in_patent == 1
    assert parsed.zip_file_path == "patent.zip"


def test_aggregator_state_returns_completed_patent_and_clears_pending():
    state = AggregatorState()
    patent_id = state.ingest_payload(payload())

    assert state.is_complete(patent_id)
    completed = state.pop_completed_patent(patent_id)

    assert completed.patent_id == "USUNITTESTA1"
    assert completed.measures_list == [{"row": 1}]
    assert completed.received_chunks == 1
    assert completed.total_chunks_in_patent == 1
    assert completed.agent1_usage["total_tokens"] == 10
    assert patent_id not in state.pending_data


def test_aggregator_state_supports_empty_measures_as_completed_patent():
    state = AggregatorState()
    patent_id = state.ingest_payload(payload(extracted_data=[]))

    completed = state.pop_completed_patent(patent_id)

    assert completed.measures_list == []


def test_failed_chunks_is_zero_when_every_chunk_answered():
    state = AggregatorState()
    patent_id = state.ingest_payload(payload(extracted_data=[]))

    assert state.pop_completed_patent(patent_id).failed_chunks == 0


def test_failed_chunks_counts_only_the_chunks_that_errored():
    state = AggregatorState()
    state.ingest_payload(payload(extracted_data=[{"row": 1}], total_chunks=3))
    state.ingest_payload(error_payload(total_chunks=3))
    state.ingest_payload(error_payload(total_chunks=3))

    completed = state.pop_completed_patent("USUNITTESTA1")

    # An errored chunk still counts as received, which is exactly why the
    # failure count has to be tracked separately.
    assert completed.received_chunks == 3
    assert completed.failed_chunks == 2
    assert completed.measures_list == [{"row": 1}]


def test_all_chunks_failing_is_distinguishable_from_a_patent_with_no_data():
    """The distinction that keeps --resume from skipping unprocessed patents."""
    state = AggregatorState()
    state.ingest_payload(error_payload(total_chunks=2))
    state.ingest_payload(error_payload(total_chunks=2))

    completed = state.pop_completed_patent("USUNITTESTA1")

    assert completed.measures_list == []
    assert completed.failed_chunks == 2


def test_failed_chunk_counts_are_isolated_per_patent_and_cleared_on_pop():
    state = AggregatorState()
    state.ingest_payload(error_payload(patent_id="USFAILEDA1"))
    state.ingest_payload(payload(patent_id="USCLEANA1"))

    assert state.pop_completed_patent("USFAILEDA1").failed_chunks == 1
    assert state.pop_completed_patent("USCLEANA1").failed_chunks == 0
    assert "USFAILEDA1" not in state.failed_chunks_count


def test_aggregator_state_reports_incomplete_chunks():
    state = AggregatorState()
    state.ingest_payload(payload(total_chunks=2, zip_file_path="/tmp/patent.zip"))

    assert state.build_incomplete_patents() == [{
        "patent_id": "USUNITTESTA1",
        "zip_file_path": "/tmp/patent.zip",
        "reason": "Not all chunks received: 1/2",
        "stage": "agent2_aggregator_pending_chunks",
    }]


def test_aggregator_state_aggregates_agent1_usage_across_chunks():
    state = AggregatorState()
    state.ingest_payload(payload(extracted_data=[{"row": 1}], total_chunks=2, agent1_usage=usage(total_tokens=10, request_count=1)))
    state.ingest_payload(payload(extracted_data=[{"row": 2}], total_chunks=2, agent1_usage=usage(total_tokens=5, request_count=1)))

    completed = state.pop_completed_patent("USUNITTESTA1")

    assert completed.measures_list == [{"row": 1}, {"row": 2}]
    assert completed.agent1_usage["total_tokens"] == 15
    assert completed.agent1_usage["request_count"] == 2
    assert completed.agent1_usage["max_tokens_per_request"] == 10
