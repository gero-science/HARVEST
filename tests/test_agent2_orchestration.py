"""Tests for Agent 2 orchestration error propagation."""

import asyncio
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if "pipeline" not in sys.modules:
    _pipeline_pkg = types.ModuleType("pipeline")
    _pipeline_pkg.__path__ = [str(_ROOT / "pipeline")]
    sys.modules["pipeline"] = _pipeline_pkg

process_patent_agent2 = importlib.import_module(
    "pipeline.agent2_orchestration",
).process_patent_agent2


class ProcessPatentAgent2ErrorPropagationTest(unittest.IsolatedAsyncioTestCase):
    async def test_reraises_on_save_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            patent_id = "US-TEST-001"
            error_log = base / "errors.jsonl"
            error_log.write_text("")

            document = MagicMock(chemistry_nodes=[])
            global_stats = {
                "data_counts": {
                    "patents_processed": 0,
                    "aliases_resolved": 0,
                    "aliases_unresolved": 0,
                    "molecules_resolved_with_smiles": 0,
                    "molecules_resolved_verified": 0,
                    "molecules_resolved_unverified": 0,
                },
                "agent2": {},
            }

            with patch(
                "pipeline.agent2_orchestration.resolve_aliases_for_patent",
                new_callable=AsyncMock,
                return_value=([], [], 0.0, {"request_count": 0, "total_tokens": 0, "cost": 0}),
            ), patch(
                "pipeline.agent2_orchestration.save_patent_result_artifacts",
                new_callable=AsyncMock,
                side_effect=OSError("disk full"),
            ):
                task = asyncio.create_task(
                    process_patent_agent2(
                        patent_id,
                        [],
                        "",
                        document,
                        str(base),
                        str(base / "processed_patents.txt"),
                        global_stats,
                        error_log,
                        {},
                        debug_mode=False,
                    )
                )
                task.patent_id = patent_id

                results = await asyncio.gather(task, return_exceptions=True)

            self.assertEqual(len(results), 1)
            self.assertIsInstance(results[0], OSError)
            self.assertIn("disk full", str(results[0]))
            self.assertTrue(error_log.read_text(encoding="utf-8").strip())

            processed_file = base / "processed_patents.txt"
            if processed_file.exists():
                self.assertNotIn(patent_id, processed_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
