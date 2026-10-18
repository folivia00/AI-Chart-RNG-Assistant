"""Filtering, aggregation and summary statistics for time series dataframes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd

Period = Literal["day", "week", "month"]
Method = Literal["mean", "sum", "min", "max"]

# Weekly buckets end on Sunday ("W"); monthly buckets are labelled by month start ("MS").
RESAMPLE_RULES: dict[str, str] = {"day": "D", "week": "W", "month": "MS"}


def date_bounds(frame: pd.DataFrame, date_col: str) -> tuple[date, date]:
    """Return the first and last calendar date in the dataframe."""
    return frame[date_col].min().date(), frame[date_col].max().date()


def last_n_days_range(frame: pd.DataFrame, date_col: str, days: int) -> tuple[date, date]:
    """Return the (start, end) range covering the last `days` days of the data.

    The range is anchored to the latest date in the data, not to today, so it
    works for historical datasets.
    """
    if days < 1:
        raise ValueError("The number of days must be at least 1.")
    first, last = date_bounds(frame, date_col)
    start = max(first, last - timedelta(days=days - 1))
    return start, last


def filter_date_range(
    frame: pd.DataFrame,
    date_col: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Return rows whose date falls within [start, end] (both inclusive, both optional)."""
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame[date_col] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame[date_col] < pd.Timestamp(end) + pd.Timedelta(days=1)
    return frame.loc[mask]


def aggregate(
    frame: pd.DataFrame,
    date_col: str,
    columns: list[str],
    period: Period,
    method: Method,
) -> pd.DataFrame:
    """Resample `columns` by day, week or month using mean, sum, min or max."""
    if period not in RESAMPLE_RULES:
        raise ValueError(f"Unknown period '{period}'.")
    if method not in ("mean", "sum", "min", "max"):
        raise ValueError(f"Unknown method '{method}'.")
    resampled = frame.set_index(date_col)[columns].resample(RESAMPLE_RULES[period])
    result = getattr(resampled, method)()
    if method != "sum":
        result = result.dropna(how="all")
    return result.reset_index()


def build_view(
    frame: pd.DataFrame,
    date_col: str,
    series: list[str],
    date_range: tuple[date, date] | None = None,
    aggregation: tuple[Period, Method] | None = None,
) -> pd.DataFrame:
    """Apply date filter, series selection and optional aggregation in one step."""
    view = frame[[date_col, *series]]
    if date_range is not None:
        view = filter_date_range(view, date_col, *date_range)
    if aggregation is not None and not view.empty:
        view = aggregate(view, date_col, series, *aggregation)
    return view.reset_index(drop=True)


@dataclass
class SeriesStats:
    """Summary statistics of one series over a period."""

    series: str
    count: int
    min: float | None
    max: float | None
    mean: float | None
    first: float | None
    last: float | None
    change_pct: float | None


def _clean(value: float) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def compute_stats(frame: pd.DataFrame, columns: list[str]) -> list[SeriesStats]:
    """Compute min, max, mean and % change (first -> last non-empty value) per column."""
    stats: list[SeriesStats] = []
    for column in columns:
        values = frame[column].dropna()
        if values.empty:
            stats.append(SeriesStats(column, 0, None, None, None, None, None, None))
            continue
        first, last = float(values.iloc[0]), float(values.iloc[-1])
        change = None if first == 0 else (last - first) / abs(first) * 100
        stats.append(
            SeriesStats(
                series=column,
                count=int(values.size),
                min=_clean(values.min()),
                max=_clean(values.max()),
                mean=_clean(values.mean()),
                first=first,
                last=last,
                change_pct=None if change is None or math.isnan(change) else change,
            )
        )
    return stats


def stats_table(stats: list[SeriesStats]) -> pd.DataFrame:
    """Convert stats to a display-friendly dataframe (values rounded to 2 decimals)."""
    table = pd.DataFrame([asdict(item) for item in stats])
    return table.rename(columns={"change_pct": "change_%"}).round(2)
