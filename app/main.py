"""Streamlit entry point: `streamlit run app/main.py`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# `streamlit run app/main.py` puts only the app/ folder on sys.path; add the project root
# so the `app` package can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.assistant.providers import select_provider  # noqa: E402
from app.assistant.providers.base import Provider  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.state import AppState  # noqa: E402
from app.ui import chat_tab, graph_tab, rng_tab  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.INFO)


def get_state() -> AppState:
    """Return the per-session application state, creating it on first use."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    return st.session_state.app_state


@st.cache_resource
def get_provider(name: str) -> tuple[Provider, str | None]:
    """Create the configured provider once per process (its SDK client is reused)."""
    return select_provider(name)


def main() -> None:
    st.set_page_config(page_title="AI Chart & RNG Assistant", layout="wide")
    state = get_state()
    provider, notice = get_provider(load_settings().llm_provider)

    with st.sidebar:
        st.header("AI Chart & RNG Assistant")
        st.caption("Assistant provider")
        st.success(provider.label)
        if notice:
            st.warning(notice)
        st.caption(
            "The assistant only picks a tool and its parameters; the app validates and runs "
            "it. Data values are never sent to a model."
        )
        if state.has_data:
            first, last = state.bounds()
            st.caption("Loaded data")
            st.write(f"**{state.source_name}**")
            st.write(f"{len(state.dataset.frame):,} rows · {first} to {last}")

    st.title("AI Chart & RNG Assistant")
    graph, random_tab, chat = st.tabs(
        ["Graph", "Random", "Chat"], key="main_tabs", on_change="rerun"
    )
    with graph:
        graph_tab.render(state)
    with random_tab:
        rng_tab.render(state)
    with chat:
        chat_tab.render(state, provider)


main()
