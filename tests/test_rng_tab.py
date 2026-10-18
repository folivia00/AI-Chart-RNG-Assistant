"""Random tab: integer mode shows whole numbers, including values filled from the chat."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


def _number(at: AppTest, label: str):
    return next(widget for widget in at.number_input if widget.label == label)


def _type_radio(at: AppTest):
    return next(widget for widget in at.radio if widget.label == "Type")


def test_integer_mode_shows_whole_numbers() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    low, high = _number(at, "Min"), _number(at, "Max")
    assert (low.value, high.value) == (1, 100)
    assert isinstance(low.value, int) and isinstance(high.value, int)
    assert _number(at, "Decimals (floats only)").disabled

    _type_radio(at).set_value("float").run()
    low = _number(at, "Min")
    assert isinstance(low.value, float) and low.value == 1.0
    assert not _number(at, "Decimals (floats only)").disabled
    assert not at.exception


def test_chat_filled_form_shows_integers() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.chat_input[0].set_value("give me 50 random integers between 10 and 100").run()
    low, high = _number(at, "Min"), _number(at, "Max")
    assert (low.value, high.value) == (10, 100)
    assert isinstance(low.value, int) and isinstance(high.value, int)
    assert _number(at, "Count").value == 50
    assert _type_radio(at).value == "int"
    assert not at.exception


def test_chat_filled_float_form_keeps_decimals() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.chat_input[0].set_value("5 random floats between 0 and 1 with 3 decimals").run()
    low, high = _number(at, "Min"), _number(at, "Max")
    assert isinstance(low.value, float) and (low.value, high.value) == (0.0, 1.0)
    assert _type_radio(at).value == "float"
    assert _number(at, "Decimals (floats only)").value == 3


def test_generate_from_form_in_integer_mode() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _number(at, "Min").set_value(5)
    _number(at, "Max").set_value(9)
    _number(at, "Count").set_value(20)
    next(b for b in at.button if b.label == "Generate").click().run()
    assert not at.error
    result = at.code[0]
    values = [int(v) for v in result.value.split(", ")]
    assert len(values) == 20 and all(5 <= v <= 9 for v in values)
    assert not at.exception
