"""LLM settings read from the environment.

Values are read at import time but a missing one is not fatal here: importing
this module must not require credentials, because it is reachable from every
pipeline entry point -- including the LLM-free `--stages export,postprocess`
rebuild of an existing results directory, which never calls an API.

Missing credentials are still caught before any work starts, in
`LLM.from_config`, which is the single funnel through which every client is
built. `llm/__init__.py` stays empty so `from llm.call_llm import LLM` does not
drag this module in at all.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Explicit path rather than a bare load_dotenv(): the bare form discovers .env
# by walking up from the calling frame, which is fragile once this module is
# imported from other packages.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Google API direct access
USE_GOOGLE_DIRECT = os.environ.get("USE_GOOGLE_DIRECT", "False").lower() in ("true", "1", "yes")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_API_URL = os.environ.get("GOOGLE_API_URL", "")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "")

# Environment variables. Absent values stay empty rather than raising, so that
# importing this module is harmless; LLM.from_config rejects them before use.
LLM_KEY = os.environ.get("LLM_KEY", "")
LLM_URL = os.environ.get("LLM_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE") or 0)

# OpenRouter specific settings
LLM_PROVIDER = os.environ.get("LLM_PROVIDER")
LLM_PROVIDER_SORT = os.environ.get("LLM_PROVIDER_SORT")
LLM_PROVIDER_QUANTIZATIONS = os.environ.get("LLM_PROVIDER_QUANTIZATIONS")
LLM_PROVIDER_IGNORE = os.environ.get("LLM_PROVIDER_IGNORE")

# API retry configuration. Defaults match .env.example.
LLM_API_RETRY_ATTEMPTS = int(os.environ.get("LLM_API_RETRY_ATTEMPTS") or 5)
LLM_API_RETRY_DELAY = int(os.environ.get("LLM_API_RETRY_DELAY") or 1)


class ConfigLLM:
    # --- API Selection Logic ---
    if USE_GOOGLE_DIRECT and GOOGLE_API_KEY:
        # Google API direct
        API_BASE_URL = GOOGLE_API_URL
        API_KEY = GOOGLE_API_KEY
        MODEL_NAME = GOOGLE_MODEL
        TEMPERATURE = LLM_TEMPERATURE

        # Provider settings are not used for Google API
        PROVIDER = None
        PROVIDER_SORT = None
        PROVIDER_QUANTIZATIONS = None
        PROVIDER_IGNORE = None
    else:
        # OpenRouter
        API_BASE_URL = LLM_URL
        API_KEY = LLM_KEY
        MODEL_NAME = LLM_MODEL
        TEMPERATURE = LLM_TEMPERATURE

        # OpenRouter specific settings
        PROVIDER = LLM_PROVIDER
        PROVIDER_SORT = LLM_PROVIDER_SORT
        PROVIDER_QUANTIZATIONS = LLM_PROVIDER_QUANTIZATIONS
        PROVIDER_IGNORE = LLM_PROVIDER_IGNORE

    # --- Error Handling Configuration ---
    API_RETRY_ATTEMPTS = LLM_API_RETRY_ATTEMPTS
    API_RETRY_DELAY = LLM_API_RETRY_DELAY


class ConfigExtractor(ConfigLLM):
    """Configuration for the bioactivity extractor (historically Agent 1)."""

    # --- LLM Response Configuration ---
    MAX_TOKENS_RESPONSE = None  # No limit on response tokens
