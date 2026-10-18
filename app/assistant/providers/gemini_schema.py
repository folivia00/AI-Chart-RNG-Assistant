"""Convert pydantic JSON Schemas into the OpenAPI subset accepted by Gemini function declarations.

Pydantic emits constructs that Gemini's `Schema` does not support or handles poorly:
`$defs`/`$ref`, `anyOf` with `{"type": "null"}` for optional fields, `anyOf` of integer and
number, `title`, `default`, `additionalProperties` and string formats such as `date`.
`to_gemini_schema()` inlines references, turns optional fields into `nullable`, collapses
numeric unions to `number`, moves defaults and date formats into the description, and
drops everything else that is not on the allowlist.
"""

from __future__ import annotations

from typing import Any

ALLOWED_KEYS = {
    "type",
    "description",
    "enum",
    "nullable",
    "properties",
    "required",
    "items",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "anyOf",
}
FORMAT_HINTS = {
    "date": "Date in YYYY-MM-DD format.",
    "date-time": "Date and time in ISO 8601 format.",
}
NUMERIC_TYPES = {"integer", "number"}


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a Gemini-compatible copy of a pydantic `model_json_schema()` result."""
    return _convert(schema, schema.get("$defs", {}))


def _resolve(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name not in defs:
            raise ValueError(f"Unresolvable schema reference: {node['$ref']}")
        # Keep sibling keys such as "description" next to the reference.
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        node = {**defs[name], **siblings}
    return node


def _merge_union(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Simplify `anyOf`: drop the null branch (-> nullable) and collapse numeric unions."""
    branches = [_resolve(branch, defs) for branch in node["anyOf"]]
    non_null = [branch for branch in branches if branch.get("type") != "null"]
    merged = {key: value for key, value in node.items() if key != "anyOf"}
    if len(non_null) < len(branches):
        merged["nullable"] = True

    if len(non_null) == 1:
        return {**non_null[0], **merged}
    if non_null and all(branch.get("type") in NUMERIC_TYPES for branch in non_null):
        return {**merged, "type": "number"}
    merged["anyOf"] = non_null
    return merged


def _describe(node: dict[str, Any]) -> str | None:
    parts = [node.get("description")]
    hint = FORMAT_HINTS.get(node.get("format", ""))
    if hint and hint not in (node.get("description") or ""):
        parts.append(hint)
    if node.get("default") is not None:
        parts.append(f"Default: {_format_default(node['default'])}.")
    text = " ".join(part for part in parts if part)
    return text or None


def _format_default(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _convert(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    node = _resolve(node, defs)
    if "anyOf" in node:
        node = _merge_union(node, defs)
    if "const" in node and "enum" not in node:
        node = {**node, "enum": [node["const"]]}

    result: dict[str, Any] = {}
    for key, value in node.items():
        if key not in ALLOWED_KEYS or key == "description":
            continue
        if key == "properties":
            result[key] = {name: _convert(child, defs) for name, child in value.items()}
        elif key == "items":
            result[key] = _convert(value, defs)
        elif key == "anyOf":
            result[key] = [_convert(branch, defs) for branch in value]
        else:
            result[key] = value

    description = _describe(node)
    if description:
        result["description"] = description
    if "type" not in result and "properties" in result:
        result["type"] = "object"
    return result
