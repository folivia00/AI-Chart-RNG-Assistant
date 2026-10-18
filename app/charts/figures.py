"""Build interactive Plotly figures."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go

SECONDARY_AXIS_RATIO = 10.0
RIGHT_AXIS_SUFFIX = " (right axis)"


def series_magnitude(values: pd.Series) -> float:
    """Typical size of a series: the median absolute value, ignoring missing values.

    The median is robust to single spikes. Returns 0.0 for an empty or all-missing series.
    """
    clean = pd.to_numeric(values, errors="coerce").abs().dropna()
    return float(clean.median()) if not clean.empty else 0.0


def split_axes(
    magnitudes: Mapping[str, float], ratio: float = SECONDARY_AXIS_RATIO
) -> tuple[list[str], list[str]]:
    """Split series into (left axis, right axis) by magnitude, keeping their order.

    Rule: the largest magnitude defines the main (left) group. A series goes to the
    right axis when its magnitude is positive and the largest magnitude is more than
    `ratio` times bigger. Series with zero or undefined magnitude stay on the left.
    With fewer than two series, or when nothing qualifies, everything stays on the left.
    """
    finite = {name: m for name, m in magnitudes.items() if math.isfinite(m) and m > 0}
    if len(magnitudes) < 2 or not finite:
        return list(magnitudes), []
    reference = max(finite.values())
    right = [name for name, m in finite.items() if reference > ratio * m]
    left = [name for name in magnitudes if name not in right]
    return left, right


def build_line_chart(
    frame: pd.DataFrame,
    date_col: str,
    series: list[str],
    title: str | None = None,
    secondary_axis: bool = True,
) -> go.Figure:
    """Return a multi-series line chart with unified hover, zoom and a range slider.

    With `secondary_axis`, series much smaller than the rest (see `split_axes`) are drawn
    on a right-hand y-axis and marked in the legend, so they do not sit flat at zero.
    """
    if secondary_axis:
        left, right = split_axes({column: series_magnitude(frame[column]) for column in series})
    else:
        left, right = list(series), []

    fig = go.Figure()
    for column in series:
        on_right = column in right
        fig.add_trace(
            go.Scatter(
                x=frame[date_col],
                y=frame[column],
                mode="lines",
                name=column + RIGHT_AXIS_SUFFIX if on_right else column,
                yaxis="y2" if on_right else "y",
                hovertemplate="%{y:,.2f}",
            )
        )
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 80, "b": 20},
        height=480,
    )
    if right:
        fig.update_layout(
            yaxis={"title": {"text": ", ".join(left)}},
            yaxis2={
                "title": {"text": ", ".join(right)},
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                # Overlaying axes default to "sync" (ticks aligned to the left grid), which
                # gives odd tick values; independent round ticks read better.
                "tickmode": "auto",
            },
            margin={"l": 60, "r": 60},
        )
    fig.update_xaxes(rangeslider={"visible": True}, title_text=date_col)
    return fig
