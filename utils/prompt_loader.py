from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = BASE_DIR / "prompts"


def load_prompt(name: str) -> str:
    """
    Load a prompt text file from /prompts.
    Example: load_prompt("researcher_system") -> prompts/researcher_system.txt
    """
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
