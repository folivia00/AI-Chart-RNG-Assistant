"""Build interactive Plotly figures."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_line_chart(
    frame: pd.DataFrame,
    date_col: str,
    series: list[str],
    title: str | None = None,
) -> go.Figure:
    """Return a multi-series line chart with unified hover, zoom and a range slider."""
    fig = go.Figure()
    for column in series:
        fig.add_trace(
            go.Scatter(
                x=frame[date_col],
                y=frame[column],
                mode="lines",
                name=column,
                hovertemplate="%{y:,.2f}",
            )
        )
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 60, "b": 20},
        height=480,
    )
    fig.update_xaxes(rangeslider={"visible": True}, title_text=date_col)
    return fig
