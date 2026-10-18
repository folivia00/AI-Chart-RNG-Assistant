"""Chat tab: message history with a visible tool call block for every assistant turn."""

from __future__ import annotations

import streamlit as st

from app.assistant.providers.base import ChatTurn, Provider
from app.assistant.router import ChatResult, handle_message
from app.state import AppState

WELCOME = (
    "Hi! Ask me to change the chart or generate random numbers, for example "
    '"show only the last 30 days", "monthly average" or '
    '"give me 50 random integers between 10 and 100".'
)


def render(state: AppState, provider: Provider) -> None:
    """Render the Chat tab."""
    history: list[dict] = st.session_state.setdefault("chat_history", [])

    with st.chat_message("assistant"):
        st.markdown(WELCOME)
    for index, entry in enumerate(history):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("result") is not None:
                _render_result(entry["result"], index)

    prompt = st.chat_input("Ask the assistant...")
    if prompt:
        turns = _to_turns(history)
        history.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            result = handle_message(prompt, provider, state, turns)
        history.append({"role": "assistant", "content": result.reply, "result": result})
        # Rerun so the Graph and Random tabs pick up state changes made by the tool.
        st.rerun()


def _to_turns(history: list[dict]) -> list[ChatTurn]:
    """Plain-text history for the provider (tool errors included, tables and files not)."""
    turns = []
    for entry in history:
        text = entry["content"]
        result = entry.get("result")
        if result is not None and result.error:
            text = f"{text} ({result.error})"
        turns.append(ChatTurn(entry["role"], text))
    return turns


def _render_result(result: ChatResult, index: int) -> None:
    if result.notice:
        st.warning(result.notice)
    if result.error:
        st.error(result.error)
    outcome = result.outcome
    if outcome is not None and outcome.table is not None:
        st.dataframe(outcome.table, hide_index=True)
    if outcome is not None and outcome.file is not None:
        st.download_button(
            f"Download {outcome.file.file_name}",
            outcome.file.data,
            file_name=outcome.file.file_name,
            mime=outcome.file.mime,
            key=f"chat_download_{index}",
        )
    if result.tool_name:
        with st.expander(f"Tool call: {result.tool_name}"):
            st.json(
                {
                    "provider": result.provider,
                    "tool": result.tool_name,
                    "params": result.params if result.params is not None else result.raw_arguments,
                    "validated": result.params is not None,
                    "status": "ok" if result.ok else "error",
                    "result": result.reply if result.ok else result.error,
                }
            )
