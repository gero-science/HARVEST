"""Async pipeline settings read from the environment.

Split out of the former top-level `config.py`: these values are consumed only
by the pipeline (`core.py`, `producer.py`) and share nothing with the LLM
settings, which now live in `llm/config.py`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Explicit path rather than a bare load_dotenv(), which resolves .env by
# walking up from the calling frame.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Defaults match .env.example. These are concurrency knobs for the extraction
# stage only, so a missing .env must not stop the LLM-free stages -- this module
# is imported by pipeline/__init__.py, which every entry point goes through.
PIPELINE_MAX_WORKERS = int(os.environ.get("PIPELINE_MAX_WORKERS") or 100)
PIPELINE_QUEUE_SIZE = int(os.environ.get("PIPELINE_QUEUE_SIZE") or 200)


class ConfigPipeline:
    """Configuration for async pipeline execution"""

    MAX_WORKERS = PIPELINE_MAX_WORKERS
    QUEUE_SIZE = PIPELINE_QUEUE_SIZE
