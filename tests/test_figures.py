"""Chart building: automatic secondary y-axis."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.charts.figures import (
    RIGHT_AXIS_SUFFIX,
    build_line_chart,
    series_magnitude,
    split_axes,
)
from app.data.columns import prepare_dataset
from app.data.loader import DEMO_DATA_PATH, load_path
from app.state import AppState


def test_series_magnitude_is_median_of_absolute_values() -> None:
    assert series_magnitude(pd.Series([-3.0, 1.0, 2.0, 1000.0])) == 2.5
    assert series_magnitude(pd.Series([None, 4.0, None])) == 4.0
    assert series_magnitude(pd.Series([], dtype=float)) == 0.0
    assert series_magnitude(pd.Series([None, None], dtype=float)) == 0.0


@pytest.mark.parametrize(
    ("magnitudes", "expected"),
    [
        ({}, ([], [])),
        ({"a": 5.0}, (["a"], [])),
        ({"a": 1000.0, "b": 500.0}, (["a", "b"], [])),
        # Exactly 10x apart stays on one axis; just over 10x moves to the right.
        ({"a": 100.0, "b": 10.0}, (["a", "b"], [])),
        ({"a": 100.0, "b": 9.99}, (["a"], ["b"])),
        # The largest series defines the main group, wherever it is in the list.
        ({"price": 24.0, "sales": 2000.0, "visitors": 1500.0}, (["sales", "visitors"], ["price"])),
        # Several small series all go to the right axis, order preserved.
        ({"a": 1e6, "b": 50.0, "c": 1.0, "d": 2e5}, (["a", "d"], ["b", "c"])),
        # Zero and undefined (empty series) magnitudes stay on the left.
        ({"a": 1000.0, "zeros": 0.0, "empty": math.nan}, (["a", "zeros", "empty"], [])),
        ({"zeros": 0.0, "also": 0.0}, (["zeros", "also"], [])),
    ],
)
def test_split_axes(magnitudes: dict[str, float], expected: tuple[list[str], list[str]]) -> None:
    assert split_axes(magnitudes) == expected


def test_split_axes_custom_ratio() -> None:
    assert split_axes({"a": 100.0, "b": 30.0}, ratio=3.0) == (["a"], ["b"])
    assert split_axes({"a": 100.0, "b": 30.0}) == (["a", "b"], [])


@pytest.fixture
def demo() -> AppState:
    """App state with the bundled demo data (sales, visitors ~2,000; avg_price ~25)."""
    raw = load_path(DEMO_DATA_PATH)
    app_state = AppState()
    app_state.set_dataset(raw, prepare_dataset(raw), DEMO_DATA_PATH.name)
    return app_state


def _axes(fig) -> dict[str, str]:
    return {trace.name: trace.yaxis for trace in fig.data}


def test_demo_chart_puts_avg_price_on_right_axis(demo: AppState) -> None:
    fig = build_line_chart(demo.view(), "date", demo.selected_series)
    assert _axes(fig) == {"sales": "y", "visitors": "y", "avg_price" + RIGHT_AXIS_SUFFIX: "y2"}
    assert fig.layout.yaxis.title.text == "sales, visitors"
    assert fig.layout.yaxis2.title.text == "avg_price"
    assert fig.layout.yaxis2.overlaying == "y"
    assert fig.layout.yaxis2.side == "right"


def test_secondary_axis_follows_filtering_and_aggregation(demo: AppState) -> None:
    demo.date_range = (pd.Timestamp("2025-06-01").date(), pd.Timestamp("2025-06-30").date())
    demo.aggregation = ("week", "sum")
    fig = build_line_chart(demo.view(), "date", demo.selected_series)
    assert _axes(fig)["avg_price" + RIGHT_AXIS_SUFFIX] == "y2"


def test_single_axis_when_disabled_or_similar_scales(demo: AppState) -> None:
    fig = build_line_chart(demo.view(), "date", demo.selected_series, secondary_axis=False)
    assert _axes(fig) == {"sales": "y", "visitors": "y", "avg_price": "y"}
    assert "yaxis2" not in fig.to_dict()["layout"]

    fig = build_line_chart(demo.view(), "date", ["sales", "visitors"])
    assert _axes(fig) == {"sales": "y", "visitors": "y"}
    assert fig.layout.yaxis.title.text is None
