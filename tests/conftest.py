"""Shared fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.columns import prepare_dataset
from app.state import AppState


@pytest.fixture
def daily_frame() -> pd.DataFrame:
    """100 days of simple data starting 2024-01-01."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "sales": range(100),
            "visitors": [10.0] * 100,
            "region": ["north"] * 100,
        }
    )


@pytest.fixture
def state(daily_frame: pd.DataFrame) -> AppState:
    """App state with the daily frame loaded."""
    app_state = AppState()
    app_state.set_dataset(daily_frame, prepare_dataset(daily_frame), "test.csv")
    return app_state
