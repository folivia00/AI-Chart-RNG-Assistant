"""Explicit application state shared by the UI tabs and the assistant tools."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from app.data.columns import Dataset
from app.data.transform import Method, Period, build_view, date_bounds


@dataclass
class AppState:
    """Everything the Graph, Random and Chat tabs share.

    `revision` is increased whenever a tool changes the state, so the UI knows to
    refresh its widgets from the state instead of keeping their old values.
    """

    raw: pd.DataFrame | None = None
    source_name: str = ""
    dataset: Dataset | None = None
    selected_series: list[str] = field(default_factory=list)
    date_range: tuple[date, date] | None = None
    aggregation: tuple[Period, Method] | None = None
    auto_secondary_axis: bool = True
    numbers: list[int] | list[float] | None = None
    numbers_params: dict[str, Any] | None = None
    revision: int = 0
    # Identifies this session for the one-image-render-at-a-time limit.
    export_owner: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def has_data(self) -> bool:
        return self.dataset is not None

    def set_dataset(self, raw: pd.DataFrame, dataset: Dataset, source_name: str) -> None:
        """Replace the loaded data and reset the view settings."""
        self.raw = raw
        self.dataset = dataset
        self.source_name = source_name
        self.selected_series = list(dataset.numeric_cols)
        self.date_range = None
        self.aggregation = None
        self.touch()

    def bounds(self) -> tuple[date, date] | None:
        """First and last date of the loaded data, if any."""
        if self.dataset is None:
            return None
        return date_bounds(self.dataset.frame, self.dataset.date_col)

    def view(self) -> pd.DataFrame:
        """Return the data as currently shown: filtered, selected and aggregated."""
        if self.dataset is None:
            raise ValueError("No data loaded.")
        return build_view(
            self.dataset.frame,
            self.dataset.date_col,
            self.selected_series,
            self.date_range,
            self.aggregation,
        )

    def touch(self) -> None:
        """Mark the state as changed outside of the UI widgets."""
        self.revision += 1
