"""Provider interface shared by the offline parser and future LLM backends.

A provider never touches data: it only turns a user message into either a
`ToolCall` (tool name + raw, unvalidated arguments) or a plain `TextReply`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolCall:
    """A request to run a tool. Arguments are validated later by the router."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextReply:
    """A plain text answer when no tool fits the message."""

    text: str


ProviderResponse = ToolCall | TextReply


@dataclass(frozen=True)
class ChatTurn:
    """One earlier message of the conversation, as plain text."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class ProviderContext:
    """Non-sensitive hints about the app state that help map a message to a tool.

    It never contains the data itself: only column names, dates and settings.
    """

    columns: list[str] = field(default_factory=list)
    last_numbers_params: dict[str, Any] | None = None
    data_range: tuple[str, str] | None = None
    current_range: tuple[str, str] | None = None
    selected_series: list[str] = field(default_factory=list)
    aggregation: tuple[str, str] | None = None
    history: list[ChatTurn] = field(default_factory=list)


class ProviderError(Exception):
    """The provider cannot answer right now. `str(error)` is safe to show to the user."""


class Provider(Protocol):
    """Turns a user message into a tool call or a text reply."""

    name: str
    label: str

    def respond(
        self,
        message: str,
        tools: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderResponse:
        """Pick a tool and its arguments for `message`, or answer with text."""
        ...
