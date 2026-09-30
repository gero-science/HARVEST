"""Credentials must be required only when an LLM is actually built.

`llm.config` is reachable from every pipeline entry point, so importing it must
not demand a key -- otherwise the LLM-free `--stages export,postprocess`
rebuild of an existing results directory cannot run without inventing one.
"""

import importlib
import logging
from types import SimpleNamespace

import pytest


def test_llm_config_imports_without_any_credentials(monkeypatch):
    for var in ("LLM_KEY", "LLM_URL", "LLM_MODEL", "LLM_TEMPERATURE",
                "LLM_API_RETRY_ATTEMPTS", "LLM_API_RETRY_DELAY"):
        monkeypatch.delenv(var, raising=False)
    # Stop python-dotenv from putting the developer's real .env back.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    config = importlib.reload(importlib.import_module("llm.config"))

    assert config.ConfigLLM.API_KEY == ""
    # Defaults still come through, so nothing downstream sees None.
    assert config.LLM_API_RETRY_ATTEMPTS == 5
    assert config.LLM_API_RETRY_DELAY == 1


def test_pipeline_package_imports_without_credentials(monkeypatch):
    """pipeline/__init__.py pulls in core and workers, which reach llm.config."""
    for var in ("LLM_KEY", "LLM_URL", "LLM_MODEL", "PIPELINE_MAX_WORKERS",
                "PIPELINE_QUEUE_SIZE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    for name in ("pipeline.config", "llm.config", "llm.extractor", "pipeline.core"):
        importlib.reload(importlib.import_module(name))

    pipeline_config = importlib.import_module("pipeline.config")
    assert pipeline_config.ConfigPipeline.MAX_WORKERS == 100
    assert pipeline_config.ConfigPipeline.QUEUE_SIZE == 200


@pytest.mark.parametrize("missing", ["API_KEY", "API_BASE_URL", "MODEL_NAME"])
def test_from_config_rejects_incomplete_credentials(missing):
    from llm.call_llm import LLM

    values = {
        "API_KEY": "k",
        "API_BASE_URL": "https://example.invalid",
        "MODEL_NAME": "m",
        "TEMPERATURE": 0,
        "MAX_TOKENS_RESPONSE": None,
        "PROVIDER": None,
    }
    values[missing] = ""

    with pytest.raises(RuntimeError) as excinfo:
        LLM.from_config(SimpleNamespace(**values), logging.getLogger("t"))

    assert "Missing LLM configuration" in str(excinfo.value)


def test_from_config_accepts_complete_credentials():
    from llm.call_llm import LLM

    client = LLM.from_config(
        SimpleNamespace(
            API_KEY="k",
            API_BASE_URL="https://example.invalid",
            MODEL_NAME="m",
            TEMPERATURE=0,
            MAX_TOKENS_RESPONSE=None,
            PROVIDER=None,
        ),
        logging.getLogger("t"),
    )

    assert client.model_name == "m"
