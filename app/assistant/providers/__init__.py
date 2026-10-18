"""Chat providers and provider selection."""

from __future__ import annotations

from app.assistant.providers.base import Provider
from app.assistant.providers.gemini import GeminiProvider, gemini_settings_from_env
from app.assistant.providers.rule_based import RuleBasedProvider

PLANNED_PROVIDERS = {"openai", "ollama"}


def select_provider(name: str | None) -> tuple[Provider, str | None]:
    """Return the provider for `name` and an optional notice for the user.

    Unknown, not yet implemented or misconfigured providers fall back to the
    offline rule-based provider with a visible notice.
    """
    normalized = (name or "rule_based").strip().lower()
    if normalized in ("", "rule_based"):
        return RuleBasedProvider(), None
    if normalized == "gemini":
        api_key, model = gemini_settings_from_env()
        if not api_key:
            return RuleBasedProvider(), (
                "Gemini unavailable: GEMINI_API_KEY is not set. Using the offline rule-based "
                "mode instead."
            )
        return GeminiProvider(api_key, model), None
    if normalized in PLANNED_PROVIDERS:
        notice = (
            f"The '{normalized}' provider is not available yet. "
            "Using the offline rule-based mode instead."
        )
    else:
        notice = f"Unknown LLM_PROVIDER '{name}'. Using the offline rule-based mode instead."
    return RuleBasedProvider(), notice
