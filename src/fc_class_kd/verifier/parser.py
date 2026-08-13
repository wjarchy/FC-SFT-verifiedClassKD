"""Parse model output / JSON into structured FunctionCall."""

from __future__ import annotations

import json
import re
from typing import Any

from fc_class_kd.types import FunctionCall

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\{\[].*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_text(text: str) -> str:
    return text.strip()


def _try_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_function_call_surface(text: str) -> str | None:
    """Return the exact first decodable call-shaped JSON object in model text."""
    text = _strip_text(text)
    direct = _try_json_loads(text)
    if isinstance(direct, dict):
        return text

    for match in _JSON_BLOCK_RE.finditer(text):
        surface = match.group(1).strip()
        obj = _try_json_loads(surface)
        if isinstance(obj, dict) and "name" in obj:
            return surface

    # Model completions normally wrap the JSON in <tool_call> tags.  A brace regex is
    # unsafe here: braces may be nested arbitrarily or appear inside quoted strings.
    # Let the standard decoder find the end of each candidate object instead.
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            return text[start : start + end]

    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    surface = extract_function_call_surface(text)
    if surface is None:
        return None
    obj = _try_json_loads(surface)
    if isinstance(obj, dict):
        return obj

    return None


def parse_function_call(text: str) -> tuple[FunctionCall | None, str | None]:
    """
    Parse a function call from raw model output or JSON string.

    Returns (call, error_message). On failure call is None.
    """
    if not text or not text.strip():
        return None, "empty_input"

    obj = _extract_json_object(text)
    if obj is None:
        return None, "json_parse_failed"

    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, "missing_or_invalid_name"

    raw_args = obj.get("arguments", {})
    if raw_args is None:
        raw_args = {}
    if isinstance(raw_args, str):
        inner = _try_json_loads(raw_args)
        if not isinstance(inner, dict):
            return None, "arguments_not_object"
        raw_args = inner
    if not isinstance(raw_args, dict):
        return None, "arguments_not_object"

    return FunctionCall(name=name.strip(), arguments=dict(raw_args)), None


def serialize_function_call(call: FunctionCall, sort_keys: bool = True) -> str:
    payload = {"name": call.name, "arguments": call.arguments}
    return json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys)
