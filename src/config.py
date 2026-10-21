"""Central configuration. All settings come from environment variables
(loaded from .env) so secrets and tunable limits stay out of code."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str
    llm_model: str
    embedding_model: str
    retrieval_top_k: int
    max_retrieval_iterations: int

    guidelines_dir: Path = PROJECT_ROOT / "data" / "guidelines"
    test_cases_dir: Path = PROJECT_ROOT / "data" / "test_cases"
    annotations_dir: Path = PROJECT_ROOT / "data" / "annotations"
    index_dir: Path = PROJECT_ROOT / "indexes"
    outputs_dir: Path = PROJECT_ROOT / "outputs"


def _positive_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        retrieval_top_k=_positive_int("RETRIEVAL_TOP_K", "5"),
        max_retrieval_iterations=_positive_int("MAX_RETRIEVAL_ITERATIONS", "3"),
    )


def api_key_is_configured(settings: Settings) -> bool:
    """True only if a real-looking key is set (not empty, not the placeholder)."""
    return bool(settings.llm_api_key) and settings.llm_api_key != "your-api-key-here"