"""Gemini provider (google-genai SDK): the model only picks a tool and its arguments.

The model receives the user message, a short text history, the tool declarations and a
compact description of the app state (column names, date range, selection). It never
receives data values, and automatic function calling is disabled: the router validates
and executes every tool call itself.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.assistant.providers.base import (
    ChatTurn,
    ProviderContext,
    ProviderError,
    ProviderResponse,
    TextReply,
    ToolCall,
)
from app.assistant.providers.gemini_schema import to_gemini_schema

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"
HISTORY_TURNS = 6
MAX_TURN_CHARS = 500
TIMEOUT_MS = 20_000
EMPTY_REPLY = "I could not match that to an action. Try rephrasing, or ask what I can do."

SYSTEM_PROMPT = """You are the assistant of a data app with a time series chart and a random \
number generator. You act only by calling the provided tools.

Rules:
- If the request matches a tool, call exactly one tool with valid arguments.
- Otherwise answer briefly in plain text (at most two sentences) and suggest what you can do.
- Use only the column names listed below. Dates are YYYY-MM-DD.
- "Last N days" is counted back from the latest date in the data. For relative periods use \
last_n_days: week = 7, month = 30, quarter = 90, year = 365.
- You cannot see the data values. For questions about values, trends or changes, call \
summarize; the app shows the statistics.
- The random number tool is a generator: never claim to predict anything."""


def gemini_settings_from_env() -> tuple[str, str]:
    """Return (api_key, model) from the environment; the key may be empty."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    return api_key, model


class GeminiProvider:
    """Provider backed by the Gemini API."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, client: Any | None = None) -> None:
        self._api_key = api_key
        self.model = model
        self._client = client

    def __repr__(self) -> str:  # never include the key
        return f"GeminiProvider(model={self.model!r})"

    @property
    def label(self) -> str:
        return f"Gemini · {self.model}"

    def respond(
        self,
        message: str,
        tools: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderResponse:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ProviderError(
                "Gemini unavailable: the google-genai package is not installed."
            ) from exc

        config = types.GenerateContentConfig(
            system_instruction=build_system_instruction(context),
            tools=[types.Tool(function_declarations=build_declarations(tools))],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0,
        )
        contents = build_contents(message, context.history)
        try:
            response = self._get_client().models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise to_provider_error(exc, self.model) from None
        return parse_response(response)

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise ProviderError("Gemini unavailable: GEMINI_API_KEY is not set.")
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    timeout=TIMEOUT_MS,
                    # Retry briefly on transient server errors, not on quota errors (429).
                    retry_options=types.HttpRetryOptions(
                        attempts=2,
                        initial_delay=1.0,
                        max_delay=4.0,
                        http_status_codes=[500, 502, 503, 504],
                    ),
                ),
            )
        return self._client


def build_declarations(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn registry tool specs into Gemini function declarations."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": to_gemini_schema(tool["parameters"]),
        }
        for tool in tools
    ]


def build_system_instruction(context: ProviderContext) -> str:
    """System prompt plus a compact, data-free summary of the app state."""
    lines = [SYSTEM_PROMPT, "", "Current app state:"]
    if context.columns:
        lines.append(f"- Numeric columns: {', '.join(context.columns)}")
        if context.data_range:
            lines.append(f"- Data covers {context.data_range[0]} to {context.data_range[1]}")
        chart_range = context.current_range or context.data_range
        if chart_range:
            lines.append(f"- Chart range: {chart_range[0]} to {chart_range[1]}")
        if context.selected_series:
            lines.append(f"- Selected series: {', '.join(context.selected_series)}")
        if context.aggregation:
            lines.append(f"- Aggregation: {context.aggregation[1]} per {context.aggregation[0]}")
        else:
            lines.append("- Aggregation: none (raw values)")
    else:
        lines.append("- No data loaded yet (chart tools will ask the user to load data).")
    if context.last_numbers_params:
        lines.append(f"- Last random numbers request: {context.last_numbers_params}")
    return "\n".join(lines)


def build_contents(message: str, history: list[ChatTurn]) -> list[dict[str, Any]]:
    """Short text history plus the new message, in the Gemini `contents` format."""
    contents = [
        {
            "role": "user" if turn.role == "user" else "model",
            "parts": [{"text": turn.text[:MAX_TURN_CHARS]}],
        }
        for turn in history[-HISTORY_TURNS:]
        if turn.text
    ]
    contents.append({"role": "user", "parts": [{"text": message}]})
    return contents


def parse_response(response: Any) -> ProviderResponse:
    """Return the first function call as a `ToolCall`, else the text as a `TextReply`."""
    calls = getattr(response, "function_calls", None) or []
    if calls:
        call = calls[0]
        if len(calls) > 1:
            logger.info("Gemini returned %d function calls; using the first", len(calls))
        args = call.args if call.args is not None else {}
        # Non-dict arguments are passed on under a dummy key so validation rejects them.
        arguments = dict(args) if isinstance(args, dict) else {"_invalid_arguments": str(args)}
        return ToolCall(call.name or "", arguments)
    text = (getattr(response, "text", None) or "").strip()
    return TextReply(text or EMPTY_REPLY)


def to_provider_error(exc: Exception, model: str) -> ProviderError:
    """Map SDK and network exceptions to a friendly `ProviderError` (no raw details)."""
    reason = _classify(exc, model)
    return ProviderError(f"Gemini unavailable: {reason}.")


def _classify(exc: Exception, model: str) -> str:
    try:
        from google.genai import errors
    except ImportError:  # pragma: no cover - the SDK is present whenever a client exists
        errors = None

    if errors is not None and isinstance(exc, errors.APIError):
        code = exc.code or 0
        logger.warning("Gemini API error: code=%s status=%s", code, exc.status)
        details = f"{exc.status} {exc.message} {exc.details}".lower()
        if code in (401, 403) or (code == 400 and "api key" in details) or "api_key" in details:
            return "the API key was rejected (check GEMINI_API_KEY)"
        if code == 429:
            return "rate limit or quota reached, try again later"
        if code == 404:
            return f"model '{model}' is not available (check GEMINI_MODEL)"
        if code >= 500:
            return "the service is temporarily unavailable"
        return f"the request failed (HTTP {code})"

    try:
        import httpx

        network_errors: tuple[type[BaseException], ...] = (httpx.HTTPError, OSError)
    except ImportError:  # pragma: no cover
        network_errors = (OSError,)
    if isinstance(exc, network_errors):
        logger.warning("Gemini network error: %s", type(exc).__name__)
        return "network error or timeout"

    logger.error("Unexpected Gemini error: %s", type(exc).__name__)
    return "unexpected error"
