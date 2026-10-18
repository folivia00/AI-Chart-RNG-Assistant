"""Static PNG/PDF export through Kaleido: on demand, cached, with graceful degradation.

Kaleido 1.x renders images with a locally installed Chrome/Chromium, which takes a few
seconds per image. Images are therefore rendered only when explicitly requested, and
the bytes are cached per figure state so repeated downloads are instant.

Two kinds of failure are distinguished:

* Chrome/Chromium (or Kaleido) is missing: `ExportUnavailableError`. This is remembered
  for the rest of the process, so the UI can disable image export.
* Anything else (timeout, render error): `ExportFailedError`. Nothing is remembered and
  the user can simply try again.

Kaleido's own per-image timeout (90 s) cannot be changed through `Figure.to_image` and
does not cover the browser start, so rendering runs in a worker thread and the caller
stops waiting after `RENDER_TIMEOUT` seconds.

A render that timed out keeps running in the background until Kaleido gives up, so
renders are limited: one at a time per owner (a browser session) and at most
`MAX_CONCURRENT_RENDERS` in the whole process. A render counts as running until its
worker thread really ends, abandoned ones included.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import Counter, OrderedDict
from typing import Literal

import plotly.graph_objects as go

logger = logging.getLogger(__name__)

ImageFormat = Literal["png", "pdf"]

UNAVAILABLE_MESSAGE = (
    "PNG/PDF export is unavailable: it needs Google Chrome or Chromium installed on the "
    "machine running the app. The interactive chart and CSV export still work."
)
CACHE_SIZE = 16
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 600
FAILED_MESSAGE = "Rendering the chart image failed. Please try again."
TIMEOUT_MESSAGE = "Rendering the chart image took too long. Please try again."
RENDER_TIMEOUT = 30.0  # seconds
MAX_CONCURRENT_RENDERS = 2
OWNER_BUSY_MESSAGE = (
    "A chart image from this session is still rendering. Try again in a few seconds."
)
SERVER_BUSY_MESSAGE = "Other chart images are still rendering. Try again in a few seconds."


class ExportError(RuntimeError):
    """Base class for static export failures. `str(error)` is safe to show to the user."""


class ExportUnavailableError(ExportError):
    """Static image export is not possible in this environment (no Chrome or Kaleido)."""


class ExportFailedError(ExportError):
    """A single export attempt failed; trying again may work."""


class ExportTimeoutError(ExportFailedError):
    """Rendering did not finish within `RENDER_TIMEOUT` seconds."""


class ExportBusyError(ExportFailedError):
    """A render limit is reached (per owner or process-wide); try again later."""


_cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_lock = threading.Lock()
_unavailable = False
_running: Counter[str | None] = Counter()  # owner -> renders whose thread is still alive


def figure_key(fig: go.Figure) -> str:
    """Return a stable hash of the figure's content, used as the cache key."""
    return hashlib.sha256(fig.to_json().encode("utf-8")).hexdigest()


def cached_image(key: str, fmt: ImageFormat) -> bytes | None:
    """Return previously rendered bytes for this figure key and format, if any."""
    with _lock:
        data = _cache.get((key, fmt))
        if data is not None:
            _cache.move_to_end((key, fmt))
        return data


def is_missing_dependency(exc: BaseException) -> bool:
    """True if the exception (or anything it was raised from) means Chrome/Kaleido is missing."""
    try:
        from kaleido.errors import ChromeNotFoundError
    except ImportError:  # Kaleido itself is missing
        return True
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ChromeNotFoundError | ImportError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def busy_message(owner: str | None = None) -> str | None:
    """Why a new render cannot start right now, or None if it can."""
    with _lock:
        if owner is not None and _running[owner]:
            return OWNER_BUSY_MESSAGE
        if _running.total() >= MAX_CONCURRENT_RENDERS:
            return SERVER_BUSY_MESSAGE
    return None


def renders_in_flight() -> int:
    """Number of render threads still running, including abandoned ones."""
    with _lock:
        return _running.total()


def _reserve(owner: str | None) -> None:
    with _lock:
        if owner is not None and _running[owner]:
            raise ExportBusyError(OWNER_BUSY_MESSAGE)
        if _running.total() >= MAX_CONCURRENT_RENDERS:
            raise ExportBusyError(SERVER_BUSY_MESSAGE)
        _running[owner] += 1


def _release(owner: str | None) -> None:
    with _lock:
        _running[owner] -= 1
        if _running[owner] <= 0:
            del _running[owner]


def _render(fig: go.Figure, fmt: ImageFormat, owner: str | None) -> bytes:
    """Render with Kaleido in a daemon thread and wait at most `RENDER_TIMEOUT` seconds.

    A thread that is still running after the timeout cannot be stopped; it is abandoned
    and ends on its own when Kaleido returns or fails. Its slot is released only then.
    """
    outcome: dict[str, object] = {}

    def work() -> None:
        try:
            outcome["data"] = fig.to_image(format=fmt, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        except BaseException as exc:  # handed over to the waiting thread
            outcome["error"] = exc
        finally:
            _release(owner)

    _reserve(owner)
    worker = threading.Thread(target=work, name="chart-export", daemon=True)
    try:
        worker.start()
    except BaseException:
        _release(owner)
        raise
    worker.join(RENDER_TIMEOUT)
    if worker.is_alive():
        raise ExportTimeoutError(TIMEOUT_MESSAGE)
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["data"]  # type: ignore[return-value]


def export_unavailable() -> bool:
    """True once an export attempt has shown that static export does not work here."""
    return _unavailable


def export_figure(
    fig: go.Figure, fmt: ImageFormat, key: str | None = None, owner: str | None = None
) -> bytes:
    """Render the figure to PNG or PDF bytes, reusing cached bytes for the same figure.

    `owner` identifies the caller (a browser session) for the one-render-per-owner limit.
    Raises `ExportUnavailableError` if Chrome/Kaleido is missing (remembered), or
    `ExportFailedError` for any other failure, including `ExportTimeoutError` and
    `ExportBusyError` (not remembered, safe to retry).
    """
    global _unavailable
    if fmt not in ("png", "pdf"):
        raise ValueError(f"Unsupported image format '{fmt}'.")
    if _unavailable:
        raise ExportUnavailableError(UNAVAILABLE_MESSAGE)

    key = key or figure_key(fig)
    data = cached_image(key, fmt)
    if data is not None:
        return data

    try:
        data = _render(fig, fmt, owner)
    except ExportTimeoutError:
        logger.warning("Static export timed out after %g s", RENDER_TIMEOUT)
        raise
    except ExportBusyError:
        logger.info("Static export refused: render limit reached")
        raise
    except Exception as exc:  # Kaleido/Chrome failures surface as several exception types
        if is_missing_dependency(exc):
            logger.warning("Static export unavailable: Chrome/Chromium or Kaleido not found")
            _unavailable = True
            raise ExportUnavailableError(UNAVAILABLE_MESSAGE) from exc
        logger.warning("Static export failed: %s", type(exc).__name__)
        raise ExportFailedError(FAILED_MESSAGE) from exc

    logger.info("Rendered chart as %s (%d bytes)", fmt.upper(), len(data))
    with _lock:
        _cache[(key, fmt)] = data
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return data


def clear_cache() -> None:
    """Forget cached images and any remembered export failure (used by tests).

    Running renders are not touched: their slots are released when their threads end.
    """
    global _unavailable
    with _lock:
        _cache.clear()
    _unavailable = False
