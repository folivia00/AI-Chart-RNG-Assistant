"""Offline provider: maps common English phrasings to tool calls with regular expressions.

It needs no network and no API key. It returns the same `ToolCall` objects an LLM
provider would, so the router validates and executes them the same way.
"""

from __future__ import annotations

import re
from typing import Any

from app.assistant.providers.base import ProviderContext, ProviderResponse, TextReply, ToolCall

HELP_TEXT = """I can drive the chart and the random number generator. Try for example:

- **Chart range:** "show only the last 30 days", "last 2 weeks", "from 2024-01-01 to 2024-06-30"
- **Series:** "show sales and visitors"
- **Aggregation:** "monthly average", "weekly sum"
- **Statistics:** "summarize", "stats for last 90 days"
- **Random numbers:** "give me 50 random integers between 10 and 100",
  "10 unique numbers from 1 to 20", "5 random floats between 0 and 1 with 3 decimals", "seed 42"
- **Export:** "export chart as png", "download the data as csv", "export numbers as csv"
"""

DEFAULT_COUNT = 10
DEFAULT_LOW = 1
DEFAULT_HIGH = 100

NUMBER = r"-?\d+(?:\.\d+)?"
DATE = r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

EXPORT_RE = re.compile(r"\b(export|download|save)\b")
NUMBER_NOUN_RE = re.compile(r"\b(integers?|ints?|numbers?|floats?|decimals?|values?)\b")
RANDOM_RE = re.compile(r"\b(random|generate|pick|draw|roll)\b")
COUNT_RE = re.compile(
    r"(\d[\d,]*)\s+(?:(?:unique|distinct|random|secure|different|whole)\s+)*"
    r"(?:integers?|ints?|numbers?|floats?|decimals?|values?)\b"
)
RANGE_RE = re.compile(
    rf"(?:between|from|in)\s+({NUMBER})\s+(?:and|to|-)\s+({NUMBER})|({NUMBER})\s*(?:-|\.\.)\s*({NUMBER})"
)
DECIMALS_RE = re.compile(
    r"(?:with|to|using|,)\s*(\d+)\s+(?:decimals?|decimal places?|digits)\b"
    r"|(\d+)\s+decimal places?\b|decimals?\s*[=:]\s*(\d+)"
)
FLOAT_RE = re.compile(r"\b(floats?|decimals?|real numbers?|fractional)\b")
UNIQUE_RE = re.compile(
    r"\b(unique|distinct|no repeats?|without (?:repeats?|repetitions?|duplicates))\b"
)
SEED_RE = re.compile(r"\bseed\s*(?:=|:|of|is|to)?\s*(\d+)")
SECURE_RE = re.compile(r"\b(secure|securely|cryptographic\w*|crypto)\b")

SUMMARY_RE = re.compile(r"\b(summari[sz]e|summary|stats|statistics|describe|overview)\b")
LAST_RE = re.compile(r"\b(?:last|past|previous)\s+(\d+\s*)?(days?|weeks?|months?|years?)\b")
DATE_RE = re.compile(DATE)

PERIOD_RE = re.compile(r"\b(daily|weekly|monthly)\b|\b(?:by|per|each|every)\s+(day|week|month)\b")
PERIOD_WORDS = {"daily": "day", "weekly": "week", "monthly": "month"}
METHOD_PATTERNS = (
    ("mean", re.compile(r"\b(average|averages|avg|mean)\b")),
    ("sum", re.compile(r"\b(sum|sums|total|totals)\b")),
    ("min", re.compile(r"\b(min|minimum|lowest|smallest)\b")),
    ("max", re.compile(r"\b(max|maximum|highest|largest|peak)\b")),
)
SHOW_RE = re.compile(r"\b(?:show|plot|display|select|chart|graph)\s+(?:only\s+|me\s+)?(.+)")


class RuleBasedProvider:
    """Offline provider based on regular expressions."""

    name = "rule_based"
    label = "Offline rule-based mode — no API key needed"

    def respond(
        self,
        message: str,
        tools: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderResponse:
        text = " ".join(message.lower().split())
        if not text:
            return TextReply(HELP_TEXT)
        for parser in (
            _parse_export,
            _parse_generate,
            _parse_seed_only,
            _parse_summarize,
            _parse_filter,
            _parse_aggregate,
            _parse_series,
        ):
            call = parser(text, context)
            if call is not None:
                return call
        return TextReply("Sorry, I did not understand that.\n\n" + HELP_TEXT)


def _parse_export(text: str, context: ProviderContext) -> ToolCall | None:
    if not EXPORT_RE.search(text):
        return None
    mentions_numbers = re.search(r"\b(numbers?|random)\b", text) is not None
    if "png" in text or re.search(r"\b(image|picture)\b", text):
        target = "chart_png"
    elif "pdf" in text:
        target = "chart_pdf"
    elif "csv" in text or re.search(r"\b(data|table|numbers?)\b", text):
        target = "numbers_csv" if mentions_numbers else "data_csv"
    elif re.search(r"\b(chart|graph|plot)\b", text):
        target = "chart_png"
    else:
        return None
    return ToolCall("export", {"target": target})


def _to_number(raw: str) -> int | float:
    value = float(raw)
    return int(value) if value.is_integer() and "." not in raw else value


def _parse_generate(text: str, context: ProviderContext) -> ToolCall | None:
    if not NUMBER_NOUN_RE.search(text):
        return None
    range_match = RANGE_RE.search(DATE_RE.sub(" ", text))
    if not (RANDOM_RE.search(text) or range_match):
        return None

    args: dict[str, Any] = {"count": DEFAULT_COUNT, "low": DEFAULT_LOW, "high": DEFAULT_HIGH}
    count_match = COUNT_RE.search(text)
    if count_match:
        args["count"] = int(count_match.group(1).replace(",", ""))
    if range_match:
        low_raw, high_raw = [group for group in range_match.groups() if group is not None]
        args["low"], args["high"] = _to_number(low_raw), _to_number(high_raw)

    decimals_match = DECIMALS_RE.search(text)
    is_float = (
        FLOAT_RE.search(text) is not None
        or decimals_match is not None
        or isinstance(args["low"], float)
        or isinstance(args["high"], float)
    )
    if is_float:
        args["kind"] = "float"
        if decimals_match:
            args["decimals"] = int(next(g for g in decimals_match.groups() if g is not None))
    if UNIQUE_RE.search(text):
        args["unique"] = True
    seed_match = SEED_RE.search(text)
    if seed_match:
        args["seed"] = int(seed_match.group(1))
    if SECURE_RE.search(text):
        args["secure"] = True
    return ToolCall("generate_numbers", args)


def _parse_seed_only(text: str, context: ProviderContext) -> ToolCall | None:
    """'seed 42' alone repeats the last generation (or a default one) with that seed."""
    seed_match = SEED_RE.search(text)
    if not seed_match:
        return None
    args = dict(context.last_numbers_params or {})
    if not args:
        args = {"count": DEFAULT_COUNT, "low": DEFAULT_LOW, "high": DEFAULT_HIGH}
    args["seed"] = int(seed_match.group(1))
    args["secure"] = False
    return ToolCall("generate_numbers", args)


def _date_args(text: str) -> dict[str, Any]:
    """Extract last_n_days or start/end dates. Dates stay strings; the router validates them."""
    last = LAST_RE.search(text)
    if last:
        amount = int(last.group(1)) if last.group(1) else 1
        unit = last.group(2).rstrip("s")
        return {"last_n_days": amount * UNIT_DAYS[unit]}

    dates = [re.sub(r"[/.]", "-", raw) for raw in DATE_RE.findall(text)]
    if len(dates) >= 2:
        return {"start": dates[0], "end": dates[1]}
    if len(dates) == 1:
        if re.search(r"\b(until|till|to|before|up to)\b", text):
            return {"end": dates[0]}
        if re.search(r"\b(since|from|after|starting)\b", text):
            return {"start": dates[0]}
        return {"start": dates[0], "end": dates[0]}
    return {}


def _parse_summarize(text: str, context: ProviderContext) -> ToolCall | None:
    if not SUMMARY_RE.search(text):
        return None
    return ToolCall("summarize", _date_args(text))


def _parse_filter(text: str, context: ProviderContext) -> ToolCall | None:
    args = _date_args(text)
    return ToolCall("filter_date_range", args) if args else None


def _parse_aggregate(text: str, context: ProviderContext) -> ToolCall | None:
    period_match = PERIOD_RE.search(text)
    if not period_match:
        return None
    word, unit = period_match.groups()
    period = PERIOD_WORDS[word] if word else unit
    method = next((name for name, pattern in METHOD_PATTERNS if pattern.search(text)), "mean")
    return ToolCall("aggregate", {"period": period, "method": method})


def _parse_series(text: str, context: ProviderContext) -> ToolCall | None:
    mentioned = [column for column in context.columns if _mentions(text, column)]
    if mentioned:
        # Keep the order in which the user named the columns.
        mentioned.sort(key=lambda column: _position(text, column))
        return ToolCall("select_series", {"columns": mentioned})

    show = SHOW_RE.search(text)
    if not show:
        return None
    names = [
        name.strip(" .!?") for name in re.split(r",|\band\b|&", show.group(1)) if name.strip(" .!?")
    ]
    return ToolCall("select_series", {"columns": names}) if names else None


def _variants(column: str) -> list[str]:
    lowered = column.lower()
    return list(dict.fromkeys([lowered, lowered.replace("_", " ")]))


def _mentions(text: str, column: str) -> bool:
    return _position(text, column) >= 0


def _position(text: str, column: str) -> int:
    positions = [
        match.start()
        for variant in _variants(column)
        if (match := re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", text))
    ]
    return min(positions) if positions else -1
