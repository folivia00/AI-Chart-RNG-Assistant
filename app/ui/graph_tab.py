"""Graph tab: load data, choose series/range/aggregation, view chart and stats, export."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.assistant.tools import dataframe_to_csv
from app.charts.export import (
    UNAVAILABLE_MESSAGE,
    ExportError,
    ExportFailedError,
    ImageFormat,
    cached_image,
    export_figure,
    export_unavailable,
    figure_key,
)
from app.charts.figures import build_line_chart
from app.data.columns import prepare_dataset
from app.data.errors import DataError
from app.data.loader import DEMO_DATA_PATH, load_table
from app.data.transform import compute_stats, filter_date_range, stats_table
from app.state import AppState

PERIOD_LABELS = {None: "No aggregation", "day": "Daily", "week": "Weekly", "month": "Monthly"}
METHODS = ["mean", "sum", "min", "max"]

logger = logging.getLogger(__name__)


def render(state: AppState) -> None:
    """Render the Graph tab."""
    _render_loader(state)
    if not state.has_data:
        st.info("Upload a CSV/XLSX file or click **Load demo data** to get started.")
        return
    _render_column_override(state)
    if not _render_controls(state):
        return

    dataset = state.dataset
    view = state.view()
    if view.empty:
        st.warning("No data in the selected date range.")
        return

    fig = build_line_chart(
        view,
        dataset.date_col,
        state.selected_series,
        state.source_name,
        secondary_axis=state.auto_secondary_axis,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistics for the selected period")
    period = filter_date_range(
        dataset.frame, dataset.date_col, *(state.date_range or state.bounds())
    )
    st.dataframe(stats_table(compute_stats(period, state.selected_series)), hide_index=True)

    _render_exports(fig, view)


def _load(state: AppState, content: bytes, name: str) -> None:
    try:
        raw = load_table(content, name)
        state.set_dataset(raw, prepare_dataset(raw), name)
        st.session_state.pop("load_error", None)
    except DataError as exc:
        st.session_state.load_error = f"Could not load '{name}': {exc}"
    except Exception:
        logger.exception("Unexpected error while loading a file")
        st.session_state.load_error = (
            f"Could not load '{name}': the file has an unexpected structure. "
            "Please check it is a table with a header row."
        )


def _render_loader(state: AppState) -> None:
    upload_col, demo_col = st.columns([4, 1], vertical_alignment="bottom")
    uploaded = upload_col.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
    if demo_col.button("Load demo data", width="stretch"):
        _load(state, DEMO_DATA_PATH.read_bytes(), DEMO_DATA_PATH.name)
    # Load each uploaded file once; reruns must not reset the user's view settings.
    if uploaded is not None and st.session_state.get("loaded_file_id") != uploaded.file_id:
        st.session_state.loaded_file_id = uploaded.file_id
        _load(state, uploaded.getvalue(), uploaded.name)
    if st.session_state.get("load_error"):
        st.error(st.session_state.load_error)


def _render_column_override(state: AppState) -> None:
    columns = list(state.raw.columns)
    dataset = state.dataset
    with (
        st.expander("Columns (auto-detected, change if needed)"),
        st.form(f"columns_form_{state.revision}"),
    ):
        date_col = st.selectbox("Date column", columns, index=columns.index(dataset.date_col))
        numeric = st.multiselect(
            "Numeric columns",
            [column for column in columns if column != date_col],
            default=[column for column in dataset.numeric_cols if column != date_col],
        )
        if st.form_submit_button("Apply columns"):
            try:
                state.set_dataset(
                    state.raw, prepare_dataset(state.raw, date_col, numeric), state.source_name
                )
                st.rerun()
            except DataError as exc:
                st.error(str(exc))


def _render_controls(state: AppState) -> bool:
    """Render series/range/aggregation widgets and write their values to the state.

    Widget keys include `state.revision`, so when the assistant changes the state the
    widgets are recreated with the new values. Returns False if nothing can be plotted.
    """
    dataset = state.dataset
    rev = state.revision
    first, last = state.bounds()
    series_col, range_col, period_col, method_col = st.columns([3, 2, 1.3, 1])

    series = series_col.multiselect(
        "Series", dataset.numeric_cols, default=state.selected_series, key=f"series_{rev}"
    )
    picked = range_col.date_input(
        "Date range",
        value=state.date_range or (first, last),
        min_value=first,
        max_value=last,
        key=f"range_{rev}",
    )
    current_period, current_method = state.aggregation or (None, "mean")
    period = period_col.selectbox(
        "Aggregation",
        list(PERIOD_LABELS),
        index=list(PERIOD_LABELS).index(current_period),
        format_func=PERIOD_LABELS.get,
        key=f"period_{rev}",
    )
    method = method_col.selectbox(
        "Method",
        METHODS,
        index=METHODS.index(current_method),
        disabled=period is None,
        key=f"method_{rev}",
    )

    state.selected_series = series
    # While the user is still picking the end date, the widget returns a single date.
    if isinstance(picked, tuple) and len(picked) == 2:
        state.date_range = None if picked == (first, last) else (picked[0], picked[1])
    state.aggregation = None if period is None else (period, method)
    caption_col, axis_col = st.columns([4, 1], vertical_alignment="center")
    caption_col.caption(
        f'"Last N days" in the chat is counted back from the latest date in the dataset ({last}).'
    )
    state.auto_secondary_axis = axis_col.toggle(
        "Auto secondary axis",
        value=state.auto_secondary_axis,
        help=(
            "Series more than 10x smaller than the largest one (by median absolute value) "
            "are drawn on a right-hand axis."
        ),
        key="auto_secondary_axis",
    )

    if not series:
        st.warning("Select at least one series to plot.")
        return False
    return True


def _render_exports(fig: go.Figure, view: pd.DataFrame) -> None:
    st.subheader("Export")
    csv_col, png_col, pdf_col = st.columns(3)
    csv_col.download_button(
        "Download data (CSV)",
        dataframe_to_csv(view),
        file_name="chart_data.csv",
        mime="text/csv",
        width="stretch",
    )
    key = figure_key(fig)
    with png_col:
        _render_image_export(fig, key, "png")
    with pdf_col:
        _render_image_export(fig, key, "pdf")
    if export_unavailable():
        st.info(UNAVAILABLE_MESSAGE)


def _render_image_export(fig: go.Figure, key: str, fmt: ImageFormat) -> None:
    """Render the image only after an explicit click; afterwards offer the cached bytes."""
    label = fmt.upper()
    data = cached_image(key, fmt)
    prepare = data is None and st.button(
        f"Prepare chart ({label})",
        key=f"prepare_{fmt}",
        disabled=export_unavailable(),
        width="stretch",
    )
    if prepare:
        with st.spinner(f"Rendering {label}..."):
            try:
                data = export_figure(fig, fmt, key)
            except ExportFailedError as exc:
                st.error(str(exc))
            except ExportError:
                # Chrome is missing: rerun so both buttons are disabled and the message shown.
                st.rerun()
    if data is not None:
        st.download_button(
            f"Download chart ({label})",
            data,
            file_name=f"chart.{fmt}",
            mime="image/png" if fmt == "png" else "application/pdf",
            key=f"download_{fmt}",
            width="stretch",
        )
