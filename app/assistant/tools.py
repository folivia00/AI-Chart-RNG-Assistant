"""Tool registry: the single source of truth for what the assistant can do.

Each tool has a name, a description, a pydantic model for its parameters and an
executor that works on an explicit `AppState`. LLM providers receive JSON schemas
generated from the same models (`tool_schemas()`).
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.charts.export import ExportError, export_figure
from app.charts.figures import build_line_chart
from app.data.transform import (
    compute_stats,
    filter_date_range,
    last_n_days_range,
    stats_table,
)
from app.rng.generator import MAX_COUNT, MAX_DECIMALS, generate_numbers, validate_params
from app.state import AppState

MAX_DAYS = 36_500
PREVIEW_SIZE = 20


class ToolExecutionError(Exception):
    """A tool could not run in the current state. `str(error)` is safe to show to the user."""


@dataclass
class ExportFile:
    """A file produced by a tool, ready to be offered as a download."""

    file_name: str
    mime: str
    data: bytes


@dataclass
class ToolOutcome:
    """What a tool returns: a short message plus optional table and file."""

    message: str
    table: pd.DataFrame | None = None
    file: ExportFile | None = None


class ToolParams(BaseModel):
    """Base class for tool parameters: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


# --- Parameter models -------------------------------------------------------


class FilterDateRangeParams(ToolParams):
    """Show only a date range: either the last N days, or start and/or end dates."""

    last_n_days: int | None = Field(
        default=None, ge=1, le=MAX_DAYS, description="Keep the last N days of the data."
    )
    start: date | None = Field(default=None, description="First date to keep (YYYY-MM-DD).")
    end: date | None = Field(default=None, description="Last date to keep (YYYY-MM-DD).")

    @model_validator(mode="after")
    def _check_range(self) -> FilterDateRangeParams:
        has_dates = self.start is not None or self.end is not None
        if self.last_n_days is None and not has_dates:
            raise ValueError("Give either last_n_days or a start and/or end date.")
        if self.last_n_days is not None and has_dates:
            raise ValueError("Use either last_n_days or start/end dates, not both.")
        if self.start and self.end and self.start > self.end:
            raise ValueError("The start date must be on or before the end date.")
        return self


class SelectSeriesParams(ToolParams):
    """Choose which numeric columns are plotted."""

    columns: list[str] = Field(min_length=1, description="Names of the columns to show.")


class AggregateParams(ToolParams):
    """Resample the chart data by period."""

    period: Literal["day", "week", "month"] = Field(description="Bucket size.")
    method: Literal["mean", "sum", "min", "max"] = Field(
        default="mean", description="How values in a bucket are combined."
    )


class SummarizeParams(ToolParams):
    """Statistics for a period; defaults to the current chart range."""

    start: date | None = Field(default=None, description="First date (YYYY-MM-DD).")
    end: date | None = Field(default=None, description="Last date (YYYY-MM-DD).")
    last_n_days: int | None = Field(
        default=None, ge=1, le=MAX_DAYS, description="Summarize the last N days instead."
    )

    @model_validator(mode="after")
    def _check_range(self) -> SummarizeParams:
        if self.last_n_days is not None and (self.start or self.end):
            raise ValueError("Use either last_n_days or start/end dates, not both.")
        if self.start and self.end and self.start > self.end:
            raise ValueError("The start date must be on or before the end date.")
        return self


class GenerateNumbersParams(ToolParams):
    """Generate random numbers (a generator, not a predictor)."""

    count: int = Field(ge=1, le=MAX_COUNT, description="How many numbers to generate.")
    low: int | float = Field(description="Minimum value (inclusive).")
    high: int | float = Field(description="Maximum value (inclusive).")
    kind: Literal["int", "float"] = Field(default="int", description="Integers or floats.")
    decimals: int = Field(
        default=2, ge=0, le=MAX_DECIMALS, description="Decimal places (floats only)."
    )
    unique: bool = Field(default=False, description="Disallow repeated values.")
    seed: int | None = Field(
        default=None, ge=0, description="Seed for reproducible results (ignored if secure)."
    )
    secure: bool = Field(default=False, description="Use the OS cryptographically secure source.")

    @model_validator(mode="after")
    def _check_values(self) -> GenerateNumbersParams:
        # RngError is a ValueError, so pydantic reports it as a validation error.
        validate_params(self.count, self.low, self.high, self.kind, self.decimals, self.unique)
        return self


class ExportParams(ToolParams):
    """Export the chart, the chart data or the generated numbers."""

    target: Literal["chart_png", "chart_pdf", "data_csv", "numbers_csv"] = Field(
        description="What to export."
    )


# --- Executors --------------------------------------------------------------


def _require_data(state: AppState) -> None:
    if not state.has_data:
        raise ToolExecutionError(
            "No data is loaded yet. Open the Graph tab and upload a file or load the demo data."
        )


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def _format_range(start: date, end: date) -> str:
    return f"{start.isoformat()} to {end.isoformat()}"


def run_filter_date_range(params: FilterDateRangeParams, state: AppState) -> ToolOutcome:
    _require_data(state)
    first, last = state.bounds()
    if params.last_n_days is not None:
        start, end = last_n_days_range(
            state.dataset.frame, state.dataset.date_col, params.last_n_days
        )
    else:
        start = max(params.start or first, first)
        end = min(params.end or last, last)
    if start > end:
        raise ToolExecutionError(
            f"That range is outside the data, which covers {_format_range(first, last)}."
        )
    state.date_range = (start, end)
    state.touch()
    rows = len(filter_date_range(state.dataset.frame, state.dataset.date_col, start, end))
    return ToolOutcome(f"Showing {_format_range(start, end)} ({_plural(rows, 'row')}).")


def run_select_series(params: SelectSeriesParams, state: AppState) -> ToolOutcome:
    _require_data(state)
    available = state.dataset.numeric_cols
    by_lower = {column.lower(): column for column in available}
    selected: list[str] = []
    unknown: list[str] = []
    for name in params.columns:
        match = by_lower.get(name.strip().lower())
        if match is None:
            unknown.append(name)
        elif match not in selected:
            selected.append(match)
    if unknown:
        raise ToolExecutionError(
            f"Unknown column(s): {', '.join(unknown)}. Available: {', '.join(available)}."
        )
    state.selected_series = selected
    state.touch()
    return ToolOutcome(f"Showing series: {', '.join(selected)}.")


def run_aggregate(params: AggregateParams, state: AppState) -> ToolOutcome:
    _require_data(state)
    state.aggregation = (params.period, params.method)
    state.touch()
    rows = len(state.view())
    return ToolOutcome(
        f"Chart now shows the {params.method} per {params.period} ({_plural(rows, 'point')})."
    )


def run_summarize(params: SummarizeParams, state: AppState) -> ToolOutcome:
    _require_data(state)
    frame, date_col = state.dataset.frame, state.dataset.date_col
    if params.last_n_days is not None:
        start, end = last_n_days_range(frame, date_col, params.last_n_days)
    elif params.start or params.end:
        first, last = state.bounds()
        start, end = params.start or first, params.end or last
    else:
        start, end = state.date_range or state.bounds()
    period = filter_date_range(frame, date_col, start, end)
    if period.empty:
        raise ToolExecutionError(f"No data between {_format_range(start, end)}.")
    columns = state.selected_series or state.dataset.numeric_cols
    table = stats_table(compute_stats(period, columns))
    return ToolOutcome(
        f"Statistics for {_format_range(start, end)} ({_plural(len(period), 'row')}).", table=table
    )


def run_generate_numbers(params: GenerateNumbersParams, state: AppState) -> ToolOutcome:
    values = generate_numbers(**params.model_dump())
    state.numbers = values
    state.numbers_params = params.model_dump()
    state.touch()
    kind = "integers" if params.kind == "int" else f"floats ({params.decimals} decimals)"
    flags = [flag for flag, on in (("unique", params.unique), ("secure", params.secure)) if on]
    if params.seed is not None and not params.secure:
        flags.append(f"seed {params.seed}")
    extra = f" [{', '.join(flags)}]" if flags else ""
    preview = ", ".join(str(value) for value in values[:PREVIEW_SIZE])
    more = f", ... ({len(values) - PREVIEW_SIZE:,} more)" if len(values) > PREVIEW_SIZE else ""
    low, high = _format_number(params.low), _format_number(params.high)
    return ToolOutcome(
        f"Generated {len(values):,} {kind} between {low} and {high}{extra}: {preview}{more}"
    )


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def numbers_to_csv(values: list[int] | list[float]) -> bytes:
    """Serialize generated numbers as a one-column CSV."""
    return dataframe_to_csv(pd.DataFrame({"value": values}))


def dataframe_to_csv(frame: pd.DataFrame) -> bytes:
    """Serialize a dataframe as CSV without the index."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def run_export(params: ExportParams, state: AppState) -> ToolOutcome:
    if params.target == "numbers_csv":
        if not state.numbers:
            raise ToolExecutionError(
                "No numbers generated yet. Try: 10 random integers from 1 to 100."
            )
        file = ExportFile("random_numbers.csv", "text/csv", numbers_to_csv(state.numbers))
        return ToolOutcome(f"Prepared {_plural(len(state.numbers), 'number')} as CSV.", file=file)

    _require_data(state)
    view = state.view()
    if params.target == "data_csv":
        file = ExportFile("chart_data.csv", "text/csv", dataframe_to_csv(view))
        return ToolOutcome(f"Prepared {_plural(len(view), 'row')} as CSV.", file=file)

    fmt = "png" if params.target == "chart_png" else "pdf"
    fig = build_line_chart(
        view,
        state.dataset.date_col,
        state.selected_series,
        state.source_name,
        secondary_axis=state.auto_secondary_axis,
    )
    try:
        data = export_figure(fig, fmt, owner=state.export_owner)
    except ExportError as exc:
        raise ToolExecutionError(str(exc)) from exc
    mime = "image/png" if fmt == "png" else "application/pdf"
    return ToolOutcome(
        f"Prepared the chart as {fmt.upper()}.", file=ExportFile(f"chart.{fmt}", mime, data)
    )


# --- Registry ---------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """A tool the assistant can call."""

    name: str
    description: str
    params_model: type[ToolParams]
    executor: Callable[[Any, AppState], ToolOutcome] = field(repr=False)


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            "filter_date_range",
            "Limit the chart to the last N days or to a start/end date range.",
            FilterDateRangeParams,
            run_filter_date_range,
        ),
        Tool(
            "select_series",
            "Choose which numeric columns are shown on the chart.",
            SelectSeriesParams,
            run_select_series,
        ),
        Tool(
            "aggregate",
            "Resample the chart data by day, week or month using mean, sum, min or max.",
            AggregateParams,
            run_aggregate,
        ),
        Tool(
            "summarize",
            "Show min, max, mean and % change for a period (default: the current chart range).",
            SummarizeParams,
            run_summarize,
        ),
        Tool(
            "generate_numbers",
            "Generate random numbers with optional uniqueness, seed and secure mode.",
            GenerateNumbersParams,
            run_generate_numbers,
        ),
        Tool(
            "export",
            "Export the chart (PNG/PDF), the chart data (CSV) or the generated numbers (CSV).",
            ExportParams,
            run_export,
        ),
    )
}


def tool_schemas() -> list[dict[str, Any]]:
    """Return JSON schemas of all tools, generated from their pydantic models."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.params_model.model_json_schema(),
        }
        for tool in TOOLS.values()
    ]
