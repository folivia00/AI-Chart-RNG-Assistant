"""Tool validation, router behaviour and provider selection."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.assistant.providers import select_provider
from app.assistant.providers.base import ToolCall
from app.assistant.providers.rule_based import RuleBasedProvider
from app.assistant.router import execute_tool_call, handle_message
from app.assistant.tools import (
    TOOLS,
    AggregateParams,
    FilterDateRangeParams,
    GenerateNumbersParams,
    tool_schemas,
)
from app.state import AppState


def test_registry_has_required_tools() -> None:
    assert set(TOOLS) == {
        "filter_date_range",
        "select_series",
        "aggregate",
        "summarize",
        "generate_numbers",
        "export",
    }


def test_schemas_are_generated_from_models() -> None:
    schemas = {schema["name"]: schema for schema in tool_schemas()}
    period = schemas["aggregate"]["parameters"]["properties"]["period"]
    assert period["enum"] == ["day", "week", "month"]
    assert "count" in schemas["generate_numbers"]["parameters"]["required"]


@pytest.mark.parametrize(
    ("model", "arguments"),
    [
        (FilterDateRangeParams, {}),
        (FilterDateRangeParams, {"last_n_days": 0}),
        (FilterDateRangeParams, {"last_n_days": 5, "start": "2024-01-01"}),
        (FilterDateRangeParams, {"start": "2024-02-01", "end": "2024-01-01"}),
        (FilterDateRangeParams, {"start": "not a date"}),
        (AggregateParams, {"period": "year"}),
        (AggregateParams, {"period": "day", "method": "median"}),
        (GenerateNumbersParams, {"count": 5, "low": 10, "high": 1}),
        (GenerateNumbersParams, {"count": 5, "low": 1, "high": 3, "unique": True}),
        (GenerateNumbersParams, {"count": 5, "low": 1, "high": 3, "extra": 1}),
    ],
)
def test_invalid_params_are_rejected(model, arguments: dict) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(arguments)


def test_valid_params() -> None:
    params = FilterDateRangeParams.model_validate({"start": "2024-01-01", "end": "2024-01-31"})
    assert params.start == date(2024, 1, 1)


def test_router_turns_bad_args_into_friendly_error(state: AppState) -> None:
    call = ToolCall("generate_numbers", {"count": 5, "low": 9, "high": 1})
    result = execute_tool_call(call, state)
    assert not result.ok
    assert "Min must be less than or equal to max" in result.error
    assert result.params is None


def test_router_unknown_tool(state: AppState) -> None:
    result = execute_tool_call(ToolCall("delete_everything", {}), state)
    assert not result.ok


def test_tools_need_data() -> None:
    result = execute_tool_call(ToolCall("summarize", {}), AppState())
    assert not result.ok
    assert "No data is loaded" in result.error


def test_chat_updates_state(state: AppState) -> None:
    provider = RuleBasedProvider()
    revision = state.revision

    handle_message("show only the last 30 days", provider, state)
    assert state.date_range == (date(2024, 3, 11), date(2024, 4, 9))

    handle_message("monthly average", provider, state)
    assert state.aggregation == ("month", "mean")

    handle_message("show visitors", provider, state)
    assert state.selected_series == ["visitors"]

    result = handle_message("10 unique numbers from 1 to 20 seed 7", provider, state)
    assert result.ok
    assert len(set(state.numbers)) == 10
    assert state.revision > revision


def test_unknown_column_is_friendly(state: AppState) -> None:
    result = execute_tool_call(ToolCall("select_series", {"columns": ["profit"]}), state)
    assert not result.ok
    assert "Available: sales, visitors" in result.error


def test_summarize_returns_table(state: AppState) -> None:
    result = handle_message("stats for last 10 days", RuleBasedProvider(), state)
    assert result.ok
    assert list(result.outcome.table["series"]) == ["sales", "visitors"]


def test_csv_exports(state: AppState) -> None:
    provider = RuleBasedProvider()
    data = handle_message("download the data as csv", provider, state)
    assert data.outcome.file.data.startswith(b"date,sales,visitors")

    missing = handle_message("export numbers as csv", provider, state)
    assert not missing.ok

    handle_message("give me 3 random integers between 1 and 6", provider, state)
    numbers = handle_message("export numbers as csv", provider, state)
    assert numbers.outcome.file.data.startswith(b"value")


def test_garbage_input_gets_help(state: AppState) -> None:
    result = handle_message("what is the weather like?", RuleBasedProvider(), state)
    assert result.tool_name is None
    assert "Try for example" in result.reply


def test_provider_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider, notice = select_provider(None)
    assert provider.name == "rule_based"
    assert notice is None
    provider, notice = select_provider("gemini")
    assert provider.name == "rule_based"
    assert "GEMINI_API_KEY is not set" in notice
    provider, notice = select_provider("openai")
    assert provider.name == "rule_based"
    assert "openai" in notice
    provider, notice = select_provider("nonsense")
    assert provider.name == "rule_based"
    assert "Unknown" in notice


def test_gemini_selected_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-placeholder")
    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    provider, notice = select_provider("gemini")
    assert provider.name == "gemini"
    assert notice is None
    assert provider.label == "Gemini · some-model"
    assert "test-key" not in repr(provider)
