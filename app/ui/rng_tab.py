"""Random tab: generator form, copy-friendly result and CSV download."""

from __future__ import annotations

import streamlit as st

from app.assistant.providers.base import ToolCall
from app.assistant.router import execute_tool_call
from app.assistant.tools import numbers_to_csv
from app.rng.generator import MAX_COUNT, MAX_DECIMALS
from app.state import AppState

DEFAULTS = {
    "count": 10,
    "low": 1,
    "high": 100,
    "kind": "int",
    "decimals": 2,
    "unique": False,
    "seed": None,
    "secure": False,
}
DISPLAY_LIMIT = 10_000


def render(state: AppState) -> None:
    """Render the Random tab."""
    st.caption(
        "A random number generator: every draw is independent, so past results "
        "say nothing about future ones."
    )
    params = {**DEFAULTS, **(state.numbers_params or {})}

    # The form key includes the revision, so values set from the chat show up here.
    with st.form(f"rng_form_{state.revision}"):
        count_col, low_col, high_col = st.columns(3)
        count = count_col.number_input("Count", 1, MAX_COUNT, value=int(params["count"]), step=1)
        low = low_col.number_input("Min", value=float(params["low"]), step=1.0)
        high = high_col.number_input("Max", value=float(params["high"]), step=1.0)

        kind_col, decimals_col, seed_col = st.columns(3)
        kind = kind_col.radio(
            "Type",
            ["int", "float"],
            index=["int", "float"].index(params["kind"]),
            format_func={"int": "Integers", "float": "Floats"}.get,
            horizontal=True,
        )
        decimals = decimals_col.number_input(
            "Decimals (floats only)", 0, MAX_DECIMALS, value=int(params["decimals"]), step=1
        )
        seed = seed_col.number_input(
            "Seed (optional)", min_value=0, value=params["seed"], step=1, placeholder="Random"
        )

        unique = st.checkbox("Unique values (no repeats)", value=params["unique"])
        secure = st.checkbox(
            "Cryptographically secure mode",
            value=params["secure"],
            help="Uses the operating system's secure random source.",
        )
        st.caption("The seed is ignored in secure mode: the OS random source cannot be seeded.")
        submitted = st.form_submit_button("Generate", type="primary")

    if submitted:
        arguments = {
            "count": int(count),
            "low": int(low) if kind == "int" and low.is_integer() else low,
            "high": int(high) if kind == "int" and high.is_integer() else high,
            "kind": kind,
            "decimals": int(decimals),
            "unique": unique,
            "seed": None if seed is None else int(seed),
            "secure": secure,
        }
        result = execute_tool_call(ToolCall("generate_numbers", arguments), state)
        if result.ok:
            st.rerun()
        st.error(result.error)

    _render_result(state)


def _render_result(state: AppState) -> None:
    if not state.numbers:
        return
    values = state.numbers
    st.subheader(f"Result ({len(values):,} numbers)")
    shown = values[:DISPLAY_LIMIT]
    st.code(", ".join(str(value) for value in shown), language=None, wrap_lines=True)
    if len(values) > DISPLAY_LIMIT:
        st.caption(f"Showing the first {DISPLAY_LIMIT:,}. Download the CSV for all values.")
    st.download_button(
        "Download numbers (CSV)",
        numbers_to_csv(values),
        file_name="random_numbers.csv",
        mime="text/csv",
    )
