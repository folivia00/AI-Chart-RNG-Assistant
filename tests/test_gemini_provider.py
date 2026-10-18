"""Gemini provider with a mocked SDK client: no network, no real key."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.assistant.providers.base import ChatTurn, ProviderContext, TextReply, ToolCall
from app.assistant.providers.gemini import EMPTY_REPLY, GeminiProvider
from app.assistant.router import build_context, handle_message
from app.assistant.tools import TOOLS, tool_schemas
from app.state import AppState

genai_errors = pytest.importorskip("google.genai.errors")
httpx = pytest.importorskip("httpx")

PLACEHOLDER_KEY = "placeholder-not-a-real-key"


@dataclass
class FakeCall:
    name: str
    args: Any


@dataclass
class FakeResponse:
    function_calls: list[FakeCall] | None = None
    text: str | None = None


@dataclass
class FakeModels:
    result: Any = None
    requests: list[dict[str, Any]] = field(default_factory=list)

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.requests.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@dataclass
class FakeClient:
    models: FakeModels


def make_provider(result: Any) -> tuple[GeminiProvider, FakeModels]:
    models = FakeModels(result)
    return GeminiProvider(PLACEHOLDER_KEY, "test-model", client=FakeClient(models)), models


def api_error(cls: type, code: int, status: str, message: str = "error") -> Exception:
    return cls(code, {"error": {"code": code, "status": status, "message": message}})


def test_function_call_becomes_tool_call() -> None:
    provider, _ = make_provider(
        FakeResponse(function_calls=[FakeCall("filter_date_range", {"last_n_days": 30})])
    )
    response = provider.respond("show the last 30 days", tool_schemas(), ProviderContext())
    assert response == ToolCall("filter_date_range", {"last_n_days": 30})


def test_request_shape(state: AppState) -> None:
    provider, models = make_provider(FakeResponse(text="Hello"))
    history = [ChatTurn("user", "hi"), ChatTurn("assistant", "Hello!")]
    provider.respond("summarize", tool_schemas(), build_context(state, history))

    request = models.requests[0]
    config = request["config"]
    assert request["model"] == "test-model"
    assert config.automatic_function_calling.disable is True
    assert config.temperature == 0
    names = [decl.name for decl in config.tools[0].function_declarations]
    assert names == list(TOOLS)
    # Context: column names and dates only, never data values.
    assert "sales, visitors" in config.system_instruction
    assert "2024-01-01 to 2024-04-09" in config.system_instruction
    assert "49.5" not in config.system_instruction
    assert [c["role"] for c in request["contents"]] == ["user", "model", "user"]
    assert request["contents"][-1]["parts"][0]["text"] == "summarize"


def test_text_reply_and_empty_reply() -> None:
    provider, _ = make_provider(FakeResponse(text="  I can filter the chart.  "))
    assert provider.respond("hi", [], ProviderContext()) == TextReply("I can filter the chart.")
    provider, _ = make_provider(FakeResponse())
    assert provider.respond("hi", [], ProviderContext()) == TextReply(EMPTY_REPLY)


def test_valid_call_runs_through_router(state: AppState) -> None:
    call = FakeCall("aggregate", {"period": "week", "method": "sum"})
    provider, _ = make_provider(FakeResponse(function_calls=[call]))
    result = handle_message("weekly totals please", provider, state)
    assert result.ok
    assert result.provider == "gemini"
    assert result.params == {"period": "week", "method": "sum"}
    assert state.aggregation == ("week", "sum")


def test_unknown_tool_is_a_friendly_error(state: AppState) -> None:
    provider, _ = make_provider(FakeResponse(function_calls=[FakeCall("drop_table", {})]))
    result = handle_message("delete everything", provider, state)
    assert not result.ok
    assert "Unknown tool 'drop_table'" in result.error
    assert result.notice is None


@pytest.mark.parametrize(
    "args",
    [
        {"count": "many", "low": 1, "high": 10},
        {"count": 5, "low": 10, "high": 1},
        {"count": 5, "low": 1, "high": 10, "colour": "red"},
        "not a dict",
    ],
)
def test_malformed_args_are_friendly_errors(state: AppState, args: Any) -> None:
    provider, _ = make_provider(FakeResponse(function_calls=[FakeCall("generate_numbers", args)]))
    result = handle_message("numbers", provider, state)
    assert not result.ok
    assert result.error.startswith("Invalid parameters:")
    assert state.numbers is None


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (
            api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "API key not valid."),
            "API key",
        ),
        (api_error(genai_errors.ClientError, 403, "PERMISSION_DENIED"), "API key"),
        (api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"), "rate limit"),
        (api_error(genai_errors.ClientError, 404, "NOT_FOUND"), "test-model"),
        (api_error(genai_errors.ServerError, 503, "UNAVAILABLE"), "temporarily unavailable"),
        (httpx.ConnectError("connection refused"), "network error"),
        (TimeoutError("timed out"), "network error"),
        (RuntimeError("boom"), "unexpected error"),
    ],
)
def test_sdk_errors_fall_back_to_offline_mode(
    state: AppState, error: Exception, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    provider, _ = make_provider(error)
    result = handle_message("show only the last 30 days", provider, state)

    assert result.provider == "rule_based"
    assert result.notice.startswith("Gemini unavailable:")
    assert reason in result.notice
    assert result.notice.endswith("Used offline mode for this message.")
    # The offline provider still handled the message.
    assert result.ok and result.tool_name == "filter_date_range"
    assert PLACEHOLDER_KEY not in result.notice
    assert PLACEHOLDER_KEY not in caplog.text
    assert "Traceback" not in result.notice


def test_missing_key_falls_back(state: AppState) -> None:
    provider = GeminiProvider("", "test-model")
    result = handle_message("monthly average", provider, state)
    assert "GEMINI_API_KEY is not set" in result.notice
    assert result.provider == "rule_based"
    assert state.aggregation == ("month", "mean")
