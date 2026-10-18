"""Route a chat message: provider -> tool call -> pydantic validation -> execution -> result.

Nothing in here raises into the UI: every failure becomes a `ChatResult` with a
friendly error message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.assistant.providers.base import (
    ChatTurn,
    Provider,
    ProviderContext,
    ProviderError,
    TextReply,
    ToolCall,
)
from app.assistant.providers.rule_based import RuleBasedProvider
from app.assistant.tools import TOOLS, ToolExecutionError, ToolOutcome, tool_schemas
from app.data.errors import DataError
from app.rng.generator import RngError
from app.state import AppState

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Outcome of one chat turn, ready for display."""

    reply: str
    tool_name: str | None = None
    raw_arguments: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    outcome: ToolOutcome | None = None
    error: str | None = None
    provider: str | None = None
    notice: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def build_context(state: AppState, history: list[ChatTurn] | None = None) -> ProviderContext:
    """Collect the non-sensitive state hints a provider may use. No data values are included."""
    history = list(history or [])
    if state.dataset is None:
        return ProviderContext(last_numbers_params=state.numbers_params, history=history)
    first, last = state.bounds()
    current = state.date_range
    return ProviderContext(
        columns=list(state.dataset.numeric_cols),
        last_numbers_params=state.numbers_params,
        data_range=(first.isoformat(), last.isoformat()),
        current_range=(current[0].isoformat(), current[1].isoformat()) if current else None,
        selected_series=list(state.selected_series),
        aggregation=state.aggregation,
        history=history,
    )


def format_validation_error(error: ValidationError) -> str:
    """Turn a pydantic error into short, readable English sentences."""
    lines = []
    for item in error.errors():
        message = item["msg"].removeprefix("Value error, ")
        field = ".".join(str(part) for part in item["loc"])
        lines.append(f"{field}: {message}" if field else message)
    return "Invalid parameters: " + "; ".join(lines)


def handle_message(
    message: str,
    provider: Provider,
    state: AppState,
    history: list[ChatTurn] | None = None,
) -> ChatResult:
    """Process one user message end to end.

    If an LLM provider fails, the offline rule-based provider answers this message
    instead and the result carries a visible notice.
    """
    context = build_context(state, history)
    tools = tool_schemas()
    used, notice = provider, None
    try:
        response = provider.respond(message, tools, context)
    except Exception as exc:
        if isinstance(exc, ProviderError):
            reason = str(exc)
        else:
            logger.exception("Provider %s failed", getattr(provider, "name", "unknown"))
            reason = "The assistant provider failed unexpectedly."
        if isinstance(provider, RuleBasedProvider):
            return ChatResult(
                reply="The assistant could not process that message.",
                error="Please try again or rephrase.",
                provider=provider.name,
            )
        used = RuleBasedProvider()
        notice = f"{reason} Used offline mode for this message."
        response = used.respond(message, tools, context)

    if isinstance(response, TextReply):
        result = ChatResult(reply=response.text)
    else:
        result = execute_tool_call(response, state)
    result.provider = used.name
    result.notice = notice
    return result


def execute_tool_call(call: ToolCall, state: AppState) -> ChatResult:
    """Validate a tool call's arguments and run the tool."""
    result = ChatResult(reply="", tool_name=call.name, raw_arguments=dict(call.arguments))
    tool = TOOLS.get(call.name)
    if tool is None:
        result.error = f"Unknown tool '{call.name}'."
        result.reply = "I tried to use a tool that does not exist."
        return result

    try:
        params = tool.params_model.model_validate(call.arguments)
    except ValidationError as exc:
        result.error = format_validation_error(exc)
        result.reply = "I could not run that request."
        return result
    result.params = params.model_dump(mode="json")

    try:
        outcome = tool.executor(params, state)
    except (ToolExecutionError, DataError, RngError) as exc:
        result.error = str(exc)
        result.reply = "I could not run that request."
        return result
    except Exception:
        logger.exception("Tool %s failed", call.name)
        result.error = f"Something went wrong while running '{call.name}'."
        result.reply = "I could not run that request."
        return result

    result.outcome = outcome
    result.reply = outcome.message
    return result
