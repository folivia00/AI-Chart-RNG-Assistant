"""Loading, column detection, filtering and statistics."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.data.columns import detect_date_column, detect_numeric_columns, prepare_dataset
from app.data.errors import (
    EmptyFileError,
    NoDateColumnError,
    NoNumericColumnsError,
    UnreadableFileError,
)
from app.data.loader import DEMO_DATA_PATH, load_path, load_table
from app.data.transform import (
    aggregate,
    compute_stats,
    filter_date_range,
    last_n_days_range,
)


def test_detects_date_and_numeric_columns(daily_frame: pd.DataFrame) -> None:
    assert detect_date_column(daily_frame) == "date"
    assert detect_numeric_columns(daily_frame, exclude="date") == ["sales", "visitors"]


def test_date_column_detected_by_content_not_only_name() -> None:
    frame = pd.DataFrame({"when": ["2024-01-01", "2024-01-02"], "year": [2024, 2025]})
    assert detect_date_column(frame) == "when"


def test_numbers_are_not_mistaken_for_dates() -> None:
    frame = pd.DataFrame({"id": ["1", "2", "3"], "value": [1.0, 2.0, 3.0]})
    assert detect_date_column(frame) is None


def test_manual_override(daily_frame: pd.DataFrame) -> None:
    dataset = prepare_dataset(daily_frame, date_col="date", numeric_cols=["visitors"])
    assert dataset.numeric_cols == ["visitors"]


def test_demo_data_loads() -> None:
    dataset = prepare_dataset(load_path(DEMO_DATA_PATH))
    assert dataset.date_col == "date"
    assert dataset.numeric_cols == ["sales", "visitors", "avg_price"]
    assert len(dataset.frame) > 1000


def test_csv_encodings_and_semicolon_separator() -> None:
    text = "datum;umsatz\n2024-01-01;10\n2024-01-02;12\n"
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        frame = load_table(text.encode(encoding), "data.csv")
        assert list(frame.columns) == ["datum", "umsatz"]


def test_cp1252_bytes_are_readable() -> None:
    content = "date,caf\xe9\n2024-01-01,1\n".encode("cp1252")
    assert list(load_table(content, "x.csv").columns) == ["date", "caf\xe9"]


def test_drops_fully_empty_rows() -> None:
    frame = load_table(b"date,value\n2024-01-01,1\n,\n2024-01-03,3\n", "x.csv")
    assert len(frame) == 2


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (b"", EmptyFileError),
        (b"date,value\n", EmptyFileError),
        (b"name,value\nfoo,1\nbar,2\n", NoDateColumnError),
        (b"date,name\n2024-01-01,foo\n2024-01-02,bar\n", NoNumericColumnsError),
    ],
)
def test_friendly_errors(content: bytes, error: type[Exception]) -> None:
    with pytest.raises(error) as info:
        prepare_dataset(load_table(content, "bad.csv"))
    assert str(info.value)


def test_unsupported_and_broken_files() -> None:
    with pytest.raises(UnreadableFileError):
        load_table(b"data", "file.json")
    with pytest.raises(UnreadableFileError):
        load_table(b"not an excel file", "file.xlsx")


def test_filter_date_range(state) -> None:
    frame = state.dataset.frame
    result = filter_date_range(frame, "date", date(2024, 1, 10), date(2024, 1, 19))
    assert len(result) == 10
    assert result["date"].min() == pd.Timestamp("2024-01-10")
    assert result["date"].max() == pd.Timestamp("2024-01-19")


def test_last_n_days_is_anchored_to_data_end(state) -> None:
    start, end = last_n_days_range(state.dataset.frame, "date", 30)
    assert end == date(2024, 4, 9)
    assert (end - start).days == 29


def test_aggregate_monthly_sum(state) -> None:
    result = aggregate(state.dataset.frame, "date", ["sales"], "month", "sum")
    assert len(result) == 4
    assert result["sales"].iloc[0] == sum(range(31))


def test_stats(state) -> None:
    [sales, _] = compute_stats(state.dataset.frame, ["sales", "visitors"])
    assert (sales.min, sales.max, sales.mean) == (0, 99, 49.5)
    assert sales.change_pct is None  # first value is 0
    [visitors] = compute_stats(state.dataset.frame, ["visitors"])
    assert visitors.change_pct == 0
