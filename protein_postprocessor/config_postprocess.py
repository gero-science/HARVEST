import os
from pathlib import Path

from llm.config import ConfigLLM
from llm.prompts import load_prompt_file

# Where the UniProt FASTA lives when nothing overrides it. This is a data path,
# not LLM configuration: callers pass an explicit path (see
# ProteinPostprocessor(fasta_path=...)), and `pipeline.py --protein-data-path`
# feeds it. Fetch the file with scripts/download_protein_data.py.
DEFAULT_PROTEIN_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "protein_data"
DEFAULT_FASTA_FILENAME = "uniprot_sprot.fasta"
DEFAULT_FASTA_PATH = DEFAULT_PROTEIN_DATA_DIR / DEFAULT_FASTA_FILENAME


class ConfigProteinLLM(ConfigLLM):
    # --- Prompt Templates ---
    # Kept as .txt beside the extractor's prompts so all prompt text lives in
    # one place and can be reviewed without reading Python.
    SYSTEM_PROMPT = load_prompt_file("protein_system_prompt.txt")
    USER_PROMPT_SIMPLE = load_prompt_file("protein_user_prompt.txt")
    USER_PROMPT_WITH_CONTEXT = load_prompt_file("protein_user_prompt_with_context.txt")

    SYSTEM_PROMPT_COMPLEX = load_prompt_file("protein_complex_system_prompt.txt")
    USER_PROMPT_COMPLEX_SIMPLE = load_prompt_file("protein_complex_user_prompt.txt")
    USER_PROMPT_COMPLEX_WITH_CONTEXT = load_prompt_file(
        "protein_complex_user_prompt_with_context.txt"
    )

    # --- LLM Configuration ---
    API_RETRY_DELAY = 15
    API_RETRY_ATTEMPTS = 5
    MAX_TOKENS_RESPONSE = 20000
    # This stage runs on a different model from the extractor, so it does not
    # inherit LLM_MODEL. Override with PROTEIN_LLM_MODEL to control its cost.
    MODEL_NAME = os.environ.get("PROTEIN_LLM_MODEL", "openai/gpt-5.1")
