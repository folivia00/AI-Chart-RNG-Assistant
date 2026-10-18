"""Application settings read from environment variables (optionally via a `.env` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


@dataclass(frozen=True)
class Settings:
    """Runtime settings. API keys are never stored here; providers read them when needed."""

    llm_provider: str


def load_settings() -> Settings:
    """Load `.env` if present (existing environment variables win) and return settings."""
    load_dotenv(ENV_FILE, override=False)
    return Settings(llm_provider=os.getenv("LLM_PROVIDER", "rule_based"))
