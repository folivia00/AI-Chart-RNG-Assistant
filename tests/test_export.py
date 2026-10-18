"""Static image export: rendered only on demand, cached, never on ordinary reruns."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import plotly.graph_objects as go
import pytest
from kaleido.errors import ChromeNotFoundError
from streamlit.testing.v1 import AppTest

from app.charts import export
from app.charts.figures import build_line_chart

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


@pytest.fixture
def render_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace Kaleido rendering with a counter (no Chrome needed)."""
    calls: list[str] = []

    def fake_to_image(self: go.Figure, format: str, **kwargs: object) -> bytes:
        calls.append(format)
        return f"fake-{format}".encode()

    monkeypatch.setattr(go.Figure, "to_image", fake_to_image)
    export.clear_cache()
    yield calls
    export.clear_cache()


def test_same_figure_is_rendered_once(render_calls: list[str], state) -> None:
    fig = build_line_chart(state.view(), "date", state.selected_series)
    first = export.export_figure(fig, "png")
    again = export.export_figure(
        build_line_chart(state.view(), "date", state.selected_series), "png"
    )
    assert first == again == b"fake-png"
    assert render_calls == ["png"]

    export.export_figure(fig, "pdf")
    assert render_calls == ["png", "pdf"]


def test_changed_figure_is_rendered_again(render_calls: list[str], state) -> None:
    export.export_figure(build_line_chart(state.view(), "date", ["sales"]), "png")
    export.export_figure(build_line_chart(state.view(), "date", ["visitors"]), "png")
    assert render_calls == ["png", "png"]


def chrome_missing(self: go.Figure, format: str, **kwargs: object) -> bytes:
    """Mimic Plotly: RuntimeError raised while handling Kaleido's ChromeNotFoundError."""
    try:
        raise ChromeNotFoundError("Chrome not found")
    except ChromeNotFoundError:
        raise RuntimeError("Kaleido requires Google Chrome to be installed.") from None


def test_missing_chrome_is_remembered(monkeypatch: pytest.MonkeyPatch, state) -> None:
    calls: list[str] = []

    def failing(self: go.Figure, format: str, **kwargs: object) -> bytes:
        calls.append(format)
        return chrome_missing(self, format)

    monkeypatch.setattr(go.Figure, "to_image", failing)
    export.clear_cache()
    fig = build_line_chart(state.view(), "date", state.selected_series)
    with pytest.raises(export.ExportUnavailableError):
        export.export_figure(fig, "png")
    assert export.export_unavailable()
    with pytest.raises(export.ExportUnavailableError):
        export.export_figure(fig, "pdf")
    assert calls == ["png"]
    export.clear_cache()


def test_transient_failure_can_be_retried(monkeypatch: pytest.MonkeyPatch, state) -> None:
    outcomes = iter([TimeoutError("render timed out"), b"fake-png"])

    def flaky(self: go.Figure, format: str, **kwargs: object) -> bytes:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(go.Figure, "to_image", flaky)
    export.clear_cache()
    fig = build_line_chart(state.view(), "date", state.selected_series)
    with pytest.raises(export.ExportFailedError):
        export.export_figure(fig, "png")
    assert not export.export_unavailable()
    assert export.export_figure(fig, "png") == b"fake-png"
    export.clear_cache()


class Hang:
    """Simulated Kaleido hang: the first `count` renders block until `release()`."""

    def __init__(self, count: int = 1) -> None:
        self.count = count
        self.calls: list[str] = []
        self._event = threading.Event()

    def to_image(self, fig: go.Figure, format: str, **kwargs: object) -> bytes:
        self.calls.append(format)
        if len(self.calls) <= self.count:
            self._event.wait(10)
        return f"fake-{format}".encode()

    def release(self) -> None:
        """Let blocked renders finish and wait until their threads are gone."""
        self._event.set()
        deadline = time.monotonic() + 5
        while export.renders_in_flight() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert export.renders_in_flight() == 0


@pytest.fixture
def hang(monkeypatch: pytest.MonkeyPatch):
    controller = Hang()

    def to_image(self: go.Figure, format: str, **kwargs: object) -> bytes:
        return controller.to_image(self, format, **kwargs)

    monkeypatch.setattr(go.Figure, "to_image", to_image)
    monkeypatch.setattr(export, "RENDER_TIMEOUT", 0.2)
    export.clear_cache()
    yield controller
    controller.release()
    export.clear_cache()


def test_hanging_render_times_out(hang: Hang, state) -> None:
    fig = build_line_chart(state.view(), "date", state.selected_series)
    with pytest.raises(export.ExportTimeoutError) as info:
        export.export_figure(fig, "png")
    assert str(info.value) == export.TIMEOUT_MESSAGE
    assert isinstance(info.value, export.ExportFailedError)
    assert not export.export_unavailable()
    assert export.export_figure(fig, "png") == b"fake-png"
    assert hang.calls == ["png", "png"]


def test_one_render_per_owner_until_abandoned_render_ends(hang: Hang, state) -> None:
    fig = build_line_chart(state.view(), "date", state.selected_series)
    with pytest.raises(export.ExportTimeoutError):
        export.export_figure(fig, "png", owner="session-a")
    # The abandoned render still runs: the same session is refused, another one is not.
    assert export.renders_in_flight() == 1
    assert export.busy_message("session-a") == export.OWNER_BUSY_MESSAGE
    with pytest.raises(export.ExportBusyError) as info:
        export.export_figure(fig, "pdf", owner="session-a")
    assert str(info.value) == export.OWNER_BUSY_MESSAGE
    assert export.busy_message("session-b") is None
    assert export.export_figure(fig, "pdf", owner="session-b") == b"fake-pdf"
    assert hang.calls == ["png", "pdf"]

    hang.release()
    assert export.busy_message("session-a") is None
    assert export.export_figure(fig, "png", owner="session-a") == b"fake-png"


def test_global_cap_on_running_renders(hang: Hang, state) -> None:
    hang.count = 2
    fig = build_line_chart(state.view(), "date", state.selected_series)
    for owner in ("a", "b"):
        with pytest.raises(export.ExportTimeoutError):
            export.export_figure(fig, "png", owner=owner)
    assert export.renders_in_flight() == export.MAX_CONCURRENT_RENDERS == 2
    assert export.busy_message("c") == export.SERVER_BUSY_MESSAGE
    with pytest.raises(export.ExportBusyError) as info:
        export.export_figure(fig, "pdf", owner="c")
    assert str(info.value) == export.SERVER_BUSY_MESSAGE
    with pytest.raises(export.ExportBusyError):
        export.export_figure(fig, "pdf")  # callers without an owner count too
    assert hang.calls == ["png", "png"]  # refused renders never reach Kaleido

    hang.release()
    assert export.export_figure(fig, "pdf", owner="c") == b"fake-pdf"


def test_cached_image_is_served_while_busy(hang: Hang, state) -> None:
    hang.count = 0
    fig = build_line_chart(state.view(), "date", state.selected_series)
    assert export.export_figure(fig, "png", owner="a") == b"fake-png"
    hang.count = 2
    with pytest.raises(export.ExportTimeoutError):
        export.export_figure(fig, "pdf", owner="a")
    assert export.export_figure(fig, "png", owner="a") == b"fake-png"


def test_missing_dependency_detection() -> None:
    try:
        chrome_missing(go.Figure(), "png")
    except RuntimeError as exc:
        assert export.is_missing_dependency(exc)
    assert not export.is_missing_dependency(RuntimeError("JavaScript error in plotly.js"))
    assert not export.is_missing_dependency(TimeoutError())


def _button(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)


def _download_labels(at: AppTest) -> list[str]:
    return [element.proto.label for element in at.get("download_button")]


def test_ui_renders_images_only_on_request(render_calls: list[str]) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _button(at, "Load demo data").click().run()
    assert render_calls == []

    # Ordinary interactions: series change, date range, aggregation, chat.
    at.multiselect[1].set_value(["sales"]).run()
    at.date_input[0].set_value((at.date_input[0].min, at.date_input[0].max)).run()
    at.selectbox[1].set_value("week").run()
    at.chat_input[0].set_value("show only the last 30 days").run()
    at.chat_input[0].set_value("give me 5 random integers between 1 and 6").run()
    assert render_calls == []
    assert "Download chart (PNG)" not in _download_labels(at)

    _button(at, "Prepare chart (PNG)").click().run()
    assert render_calls == ["png"]
    assert "Download chart (PNG)" in _download_labels(at)

    # Further reruns reuse the cached bytes.
    at.run()
    at.chat_input[0].set_value("summarize").run()
    assert render_calls == ["png"]
    assert not at.exception


def _load_demo() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _button(at, "Load demo data").click().run()
    return at


def test_ui_retry_after_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter([TimeoutError("render timed out"), b"fake-png"])

    def flaky(self: go.Figure, format: str, **kwargs: object) -> bytes:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(go.Figure, "to_image", flaky)
    export.clear_cache()
    at = _load_demo()

    _button(at, "Prepare chart (PNG)").click().run()
    assert [e.value for e in at.error] == [export.FAILED_MESSAGE]
    assert not _button(at, "Prepare chart (PNG)").disabled

    _button(at, "Prepare chart (PNG)").click().run()
    assert not at.error
    assert "Download chart (PNG)" in _download_labels(at)
    export.clear_cache()


def test_ui_blocks_export_while_abandoned_render_runs(hang: Hang) -> None:
    at = _load_demo()

    _button(at, "Prepare chart (PNG)").click().run()
    assert [e.value for e in at.error] == [export.TIMEOUT_MESSAGE]
    # The timed-out render is still running: both buttons wait for it.
    assert _button(at, "Prepare chart (PNG)").disabled
    assert _button(at, "Prepare chart (PDF)").disabled
    assert export.OWNER_BUSY_MESSAGE in [c.value for c in at.caption]

    at.run()  # still running: still disabled
    assert _button(at, "Prepare chart (PNG)").disabled

    hang.release()
    _button(at, "Check again").click().run()
    assert not at.error
    assert not _button(at, "Prepare chart (PNG)").disabled
    assert "Check again" not in [b.label for b in at.button]

    _button(at, "Prepare chart (PNG)").click().run()
    assert "Download chart (PNG)" in _download_labels(at)
    assert hang.calls == ["png", "png"]
    assert not at.exception


def test_ui_disables_export_when_server_is_busy(hang: Hang, state) -> None:
    hang.count = 2
    fig = build_line_chart(state.view(), "date", state.selected_series)
    for owner in ("other-1", "other-2"):
        with pytest.raises(export.ExportTimeoutError):
            export.export_figure(fig, "png", owner=owner)

    at = _load_demo()
    assert _button(at, "Prepare chart (PNG)").disabled
    assert export.SERVER_BUSY_MESSAGE in [c.value for c in at.caption]
    assert not at.exception


def test_ui_disables_export_without_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(go.Figure, "to_image", chrome_missing)
    export.clear_cache()
    at = _load_demo()

    _button(at, "Prepare chart (PNG)").click().run()
    assert [i.value for i in at.info] == [export.UNAVAILABLE_MESSAGE]
    assert _button(at, "Prepare chart (PNG)").disabled
    assert _button(at, "Prepare chart (PDF)").disabled
    assert "Download data (CSV)" in _download_labels(at)
    assert not at.exception
    export.clear_cache()


def test_chat_export_follows_axis_toggle_and_session_limit(
    monkeypatch: pytest.MonkeyPatch, hang: Hang, state
) -> None:
    from app.assistant.providers.base import ToolCall
    from app.assistant.router import execute_tool_call

    figures: list[go.Figure] = []
    original = go.Figure.to_image

    def capture(self: go.Figure, format: str, **kwargs: object) -> bytes:
        figures.append(self)
        return original(self, format, **kwargs)

    monkeypatch.setattr(go.Figure, "to_image", capture)
    state.dataset.frame["visitors"] = 0.5  # sales median 49.5: about 100x apart
    state.selected_series = ["sales", "visitors"]
    state.auto_secondary_axis = False

    result = execute_tool_call(ToolCall("export", {"target": "chart_png"}), state)
    assert result.error == export.TIMEOUT_MESSAGE  # the first render hangs
    assert {trace.yaxis for trace in figures[0].data} == {"y"}

    busy = execute_tool_call(ToolCall("export", {"target": "chart_pdf"}), state)
    assert busy.error == export.OWNER_BUSY_MESSAGE
    hang.release()

    state.auto_secondary_axis = True
    done = execute_tool_call(ToolCall("export", {"target": "chart_pdf"}), state)
    assert done.ok and done.outcome.file.data == b"fake-pdf"
    assert [trace.yaxis for trace in figures[-1].data] == ["y", "y2"]
