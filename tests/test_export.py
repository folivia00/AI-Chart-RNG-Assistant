"""Static image export: rendered only on demand, cached, never on ordinary reruns."""

from __future__ import annotations

import threading
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


@pytest.fixture
def hanging_render(monkeypatch: pytest.MonkeyPatch):
    """Make the first render hang until the test ends, later renders succeed."""
    release = threading.Event()
    calls: list[str] = []

    def hang_once(self: go.Figure, format: str, **kwargs: object) -> bytes:
        calls.append(format)
        if len(calls) == 1:
            release.wait(10)
        return f"fake-{format}".encode()

    monkeypatch.setattr(go.Figure, "to_image", hang_once)
    monkeypatch.setattr(export, "RENDER_TIMEOUT", 0.2)
    export.clear_cache()
    yield calls
    release.set()
    export.clear_cache()


def test_hanging_render_times_out(hanging_render: list[str], state) -> None:
    fig = build_line_chart(state.view(), "date", state.selected_series)
    with pytest.raises(export.ExportTimeoutError) as info:
        export.export_figure(fig, "png")
    assert str(info.value) == export.TIMEOUT_MESSAGE
    assert isinstance(info.value, export.ExportFailedError)
    assert not export.export_unavailable()
    assert export.export_figure(fig, "png") == b"fake-png"
    assert hanging_render == ["png", "png"]


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


def test_ui_retry_after_timeout(hanging_render: list[str]) -> None:
    at = _load_demo()

    _button(at, "Prepare chart (PNG)").click().run()
    assert [e.value for e in at.error] == [export.TIMEOUT_MESSAGE]
    assert not _button(at, "Prepare chart (PNG)").disabled

    _button(at, "Prepare chart (PNG)").click().run()
    assert not at.error
    assert "Download chart (PNG)" in _download_labels(at)
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
