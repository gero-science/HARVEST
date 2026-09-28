"""Tests for Agent 1 per-payload error increments."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if "pipeline" not in sys.modules:
    _pipeline_pkg = types.ModuleType("pipeline")
    _pipeline_pkg.__path__ = [str(_ROOT / "pipeline")]
    sys.modules["pipeline"] = _pipeline_pkg

agent1_processing = importlib.import_module("pipeline.agent1_processing")
error_handling = importlib.import_module("pipeline.error_handling")
aggregator_state = importlib.import_module("pipeline.aggregator_state")

build_agent1_error_payload = agent1_processing.build_agent1_error_payload
create_empty_error_stats = error_handling.create_empty_error_stats
parse_agent1_payload = aggregator_state.parse_agent1_payload


def aggregate_agent1_errors(global_errors, payload):
    for error_type, count in payload.worker_errors.items():
        if count > 0:
            global_errors[error_type] += count


class Agent1ErrorAggregationTest(unittest.TestCase):
    def test_error_payload_carries_single_increment(self):
        task = MagicMock()
        task.patent_id = "US-TEST"
        task.total_chunks_in_patent = 1
        task.patent_text = ""
        task.document = None
        task.zip_file_path = ""

        error_detail = {"error_type": "api_errors", "agent": "agent1"}
        payload_tuple = build_agent1_error_payload(task, "api_errors", error_detail)
        payload = parse_agent1_payload(payload_tuple)

        self.assertEqual(payload.worker_errors, {"api_errors": 1})

    def test_three_increments_sum_to_three_not_six(self):
        global_errors = create_empty_error_stats()
        task = MagicMock()
        task.patent_id = "US-TEST"
        task.total_chunks_in_patent = 3
        task.patent_text = ""
        task.document = None
        task.zip_file_path = ""

        for _ in range(3):
            error_detail = {"error_type": "api_errors", "agent": "agent1"}
            payload = parse_agent1_payload(
                build_agent1_error_payload(task, "api_errors", error_detail),
            )
            aggregate_agent1_errors(global_errors, payload)

        self.assertEqual(global_errors["api_errors"], 3)
        self.assertEqual(sum(global_errors.values()), 3)


if __name__ == "__main__":
    unittest.main()
