"""Conversion of pydantic tool schemas into Gemini-compatible declarations."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel

from app.assistant.providers.gemini_schema import ALLOWED_KEYS, to_gemini_schema
from app.assistant.tools import TOOLS

TOOL_NAMES = sorted(TOOLS)


def walk(node: Any):
    """Yield every schema node (dict) in a converted schema."""
    if isinstance(node, dict):
        yield node
        for key in ("properties",):
            for child in node.get(key, {}).values():
                yield from walk(child)
        if "items" in node:
            yield from walk(node["items"])
        for branch in node.get("anyOf", []):
            yield from walk(branch)


def converted(name: str) -> dict[str, Any]:
    return to_gemini_schema(TOOLS[name].params_model.model_json_schema())


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_only_supported_keys_remain(name: str) -> None:
    for node in walk(converted(name)):
        assert set(node) <= ALLOWED_KEYS, set(node) - ALLOWED_KEYS
        assert "type" in node or "anyOf" in node
        assert node.get("type") != "null"


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_top_level_is_object_with_same_fields(name: str) -> None:
    original = TOOLS[name].params_model.model_json_schema()
    schema = converted(name)
    assert schema["type"] == "object"
    assert list(schema["properties"]) == list(original["properties"])
    assert schema.get("required", []) == original.get("required", [])


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_accepted_by_sdk_schema_model(name: str) -> None:
    types = pytest.importorskip("google.genai.types")
    # The SDK model rejects unknown keys, so this checks the declaration shape offline.
    types.Schema.model_validate(converted(name))
    types.FunctionDeclaration.model_validate(
        {"name": name, "description": TOOLS[name].description, "parameters": converted(name)}
    )


def test_optional_fields_become_nullable() -> None:
    props = converted("filter_date_range")["properties"]
    assert props["last_n_days"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 36500,
        "nullable": True,
        "description": "Keep the last N days of the data.",
    }
    assert props["start"]["type"] == "string"
    assert props["start"]["nullable"] is True
    assert "YYYY-MM-DD" in props["start"]["description"]


def test_numeric_union_collapses_to_number() -> None:
    props = converted("generate_numbers")["properties"]
    assert props["low"]["type"] == "number"
    assert "anyOf" not in props["low"]


def test_defaults_move_into_description() -> None:
    props = converted("generate_numbers")["properties"]
    assert props["kind"]["enum"] == ["int", "float"]
    assert props["kind"]["description"].endswith("Default: int.")
    assert props["unique"]["description"].endswith("Default: false.")
    assert "default" not in props["kind"]


def test_array_and_enum_tools() -> None:
    assert converted("select_series")["properties"]["columns"] == {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "description": "Names of the columns to show.",
    }
    assert converted("export")["properties"]["target"]["enum"] == [
        "chart_png",
        "chart_pdf",
        "data_csv",
        "numbers_csv",
    ]


def test_refs_are_inlined() -> None:
    class Window(BaseModel):
        unit: Literal["day", "week"]

    class Params(BaseModel):
        window: Window
        backup: Window | None = None

    raw = Params.model_json_schema()
    assert "$defs" in raw
    schema = to_gemini_schema(raw)
    assert schema["properties"]["window"]["properties"]["unit"]["enum"] == ["day", "week"]
    assert schema["properties"]["backup"]["nullable"] is True
    assert schema["properties"]["backup"]["type"] == "object"
    assert "$ref" not in str(schema) and "$defs" not in str(schema)
