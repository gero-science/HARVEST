"""
Utility functions for loading prompts from files.
"""

from pathlib import Path


def load_prompt_file(filename: str) -> str:
    prompts_dir = Path(__file__).parent
    prompt_path = prompts_dir / filename
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

