"""Detect date and numeric columns and turn a raw table into a clean time series dataset."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

from app.data.errors import EmptyFileError, NoDateColumnError, NoNumericColumnsError

DATE_NAME_HINTS = ("date", "time", "day", "timestamp", "period", "month")
MIN_PARSE_RATIO = 0.8
SAMPLE_SIZE = 200


@dataclass
class Dataset:
    """A cleaned dataset: parsed dates (sorted ascending) plus numeric value columns."""

    frame: pd.DataFrame
    date_col: str
    numeric_cols: list[str]


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _parse_dates(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        # Mixed or ambiguous formats make pandas warn; unparseable values become NaT.
        warnings.simplefilter("ignore")
        # utc=True handles mixed time zones; naive values keep their wall-clock time.
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_localize(None)


def _date_parse_ratio(series: pd.Series) -> float:
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0
    if not _is_text(series):
        return 0.0
    sample = series.dropna().astype(str).str.strip().head(SAMPLE_SIZE)
    sample = sample[sample != ""]
    if sample.empty:
        return 0.0
    # Plain numbers like "2024" or "15" are not treated as dates.
    if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
        return 0.0
    return float(_parse_dates(sample).notna().mean())


def detect_date_column(frame: pd.DataFrame) -> str | None:
    """Return the most likely date column, or None.

    Columns that parse as dates for at least 80% of sampled values qualify;
    among them, a column whose name looks like a date (e.g. "date", "timestamp") wins.
    """
    candidates = [
        column for column in frame.columns if _date_parse_ratio(frame[column]) >= MIN_PARSE_RATIO
    ]
    if not candidates:
        return None
    for column in candidates:
        if any(hint in column.lower() for hint in DATE_NAME_HINTS):
            return column
    return candidates[0]


def _numeric_ratio(series: pd.Series) -> float:
    if pd.api.types.is_bool_dtype(series):
        return 0.0
    if pd.api.types.is_numeric_dtype(series):
        return 1.0
    if not _is_text(series):
        return 0.0
    values = series.dropna()
    if values.empty:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").notna().mean())


def detect_numeric_columns(frame: pd.DataFrame, exclude: str | None = None) -> list[str]:
    """Return columns whose values are (at least 90%) numeric, excluding `exclude`."""
    return [
        column
        for column in frame.columns
        if column != exclude and _numeric_ratio(frame[column]) >= 0.9
    ]


def prepare_dataset(
    raw: pd.DataFrame,
    date_col: str | None = None,
    numeric_cols: list[str] | None = None,
) -> Dataset:
    """Build a `Dataset` from a raw table.

    `date_col` and `numeric_cols` override auto-detection. Rows with an unparseable
    date are dropped, values are coerced to numbers, and rows are sorted by date.
    """
    if raw.empty:
        raise EmptyFileError()

    date_col = date_col or detect_date_column(raw)
    if date_col is None or date_col not in raw.columns:
        raise NoDateColumnError()

    if numeric_cols is None:
        numeric_cols = detect_numeric_columns(raw, exclude=date_col)
    numeric_cols = [
        column for column in numeric_cols if column in raw.columns and column != date_col
    ]
    if not numeric_cols:
        raise NoNumericColumnsError()

    frame = pd.DataFrame({date_col: _parse_dates(raw[date_col])})
    for column in numeric_cols:
        frame[column] = pd.to_numeric(raw[column], errors="coerce")

    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        raise NoDateColumnError(
            f"Column '{date_col}' does not contain any valid dates. Pick another date column."
        )
    frame = frame.dropna(subset=numeric_cols, how="all")
    if frame.empty:
        raise NoNumericColumnsError(
            "The selected numeric columns contain no numbers. Pick other columns."
        )

    frame = frame.sort_values(date_col).reset_index(drop=True)
    return Dataset(frame=frame, date_col=date_col, numeric_cols=list(numeric_cols))
