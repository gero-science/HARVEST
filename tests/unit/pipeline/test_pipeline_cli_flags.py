import subprocess
import sys

import pytest


def run_help():
    return subprocess.run(
        [sys.executable, "pipeline.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--config",
        "--stages",
        "--from-stage",
        "--continue-on-error",
    ],
)
def test_pipeline_cli_exposes_flag(flag):
    assert flag in run_help().stdout


def test_pipeline_cli_lists_available_stages():
    stdout = run_help().stdout

    for stage in ("extract", "proteins", "export", "postprocess"):
        assert stage in stdout
