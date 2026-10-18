"""Offline rule-based parser on the supported phrase list."""

from __future__ import annotations

import pytest

from app.assistant.providers.base import ProviderContext, TextReply, ToolCall
from app.assistant.providers.rule_based import RuleBasedProvider

CONTEXT = ProviderContext(columns=["sales", "visitors", "avg_price"])


def parse(message: str, context: ProviderContext = CONTEXT):
    return RuleBasedProvider().respond(message, [], context)


@pytest.mark.parametrize(
    ("message", "name", "arguments"),
    [
        ("show only the last 30 days", "filter_date_range", {"last_n_days": 30}),
        ("last 2 weeks", "filter_date_range", {"last_n_days": 14}),
        ("Show the past 3 months", "filter_date_range", {"last_n_days": 90}),
        (
            "from 2024-01-01 to 2024-06-30",
            "filter_date_range",
            {"start": "2024-01-01", "end": "2024-06-30"},
        ),
        ("since 2024-03-01", "filter_date_range", {"start": "2024-03-01"}),
        ("show sales and visitors", "select_series", {"columns": ["sales", "visitors"]}),
        ("plot avg price", "select_series", {"columns": ["avg_price"]}),
        ("monthly average", "aggregate", {"period": "month", "method": "mean"}),
        ("weekly sum", "aggregate", {"period": "week", "method": "sum"}),
        ("max per day", "aggregate", {"period": "day", "method": "max"}),
        ("summarize", "summarize", {}),
        ("stats for last 90 days", "summarize", {"last_n_days": 90}),
        (
            "give me 50 random integers between 10 and 100",
            "generate_numbers",
            {"count": 50, "low": 10, "high": 100},
        ),
        (
            "10 unique numbers from 1 to 20",
            "generate_numbers",
            {"count": 10, "low": 1, "high": 20, "unique": True},
        ),
        (
            "5 random floats between 0 and 1 with 3 decimals",
            "generate_numbers",
            {"count": 5, "low": 0, "high": 1, "kind": "float", "decimals": 3},
        ),
        (
            "20 random numbers between 1 and 6 seed 42",
            "generate_numbers",
            {"count": 20, "low": 1, "high": 6, "seed": 42},
        ),
        (
            "3 secure random integers from 1 to 1000",
            "generate_numbers",
            {"count": 3, "low": 1, "high": 1000, "secure": True},
        ),
        ("export chart as png", "export", {"target": "chart_png"}),
        ("save the chart as PDF", "export", {"target": "chart_pdf"}),
        ("download the data as csv", "export", {"target": "data_csv"}),
        ("export numbers as csv", "export", {"target": "numbers_csv"}),
    ],
)
def test_phrases(message: str, name: str, arguments: dict) -> None:
    response = parse(message)
    assert isinstance(response, ToolCall)
    assert response.name == name
    assert response.arguments == arguments


def test_seed_alone_reuses_last_params() -> None:
    last = {"count": 5, "low": 1, "high": 6, "kind": "int", "decimals": 2, "unique": False}
    response = parse("seed 42", ProviderContext(last_numbers_params=last))
    assert response.arguments == {**last, "seed": 42, "secure": False}


def test_seed_alone_without_history_uses_defaults() -> None:
    response = parse("use seed 7")
    assert response.name == "generate_numbers"
    assert response.arguments["seed"] == 7


def test_series_without_context_columns() -> None:
    response = parse("show sales and visitors", ProviderContext())
    assert response.arguments == {"columns": ["sales", "visitors"]}


@pytest.mark.parametrize("message", ["", "hello there", "what is the meaning of life"])
def test_unknown_input_gets_help(message: str) -> None:
    response = parse(message)
    assert isinstance(response, TextReply)
    assert "Try for example" in response.text
