"""
Asynchronous pipeline module for extracting and processing patent data.

This module implements a fully asynchronous process for:
1. Extracting bioactivity data from patent tables (Agent 1).
2. Resolving chemical compound aliases (Agent 2).
3. Enriching data with protein sequences (Agent 3).

Key features:
- 100% asynchrony: all operations are non-blocking
- Parallel processing: Agent 2 processes patents in parallel
- Maximum performance: up to 50 workers for maximum API utilization
- Reliability: an error in one component does not stop the entire pipeline
"""

from .core import run_pipeline_async
from .full_run import run_full_pipeline
from .run_config import (
    STAGES,
    PipelineRunConfig,
    apply_cli_overrides,
    load_run_config,
    resolve_stages,
)
from .utils import setup_logging

__all__ = [
    'STAGES',
    'PipelineRunConfig',
    'apply_cli_overrides',
    'load_run_config',
    'resolve_stages',
    'run_full_pipeline',
    'run_pipeline_async',
    'setup_logging',
]


