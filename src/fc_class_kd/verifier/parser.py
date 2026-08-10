"""Parse model output / JSON into structured FunctionCall."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from fc_class_kd.types import FunctionCall


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\{\[].*?)\s*```", re.DOTALL | re.IGNORECASE)
_BRACE_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def _strip_text(text: str) -> str:
    return text.strip()


def _try_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_text(text)
    direct = _try_json_loads(text)
    if isinstance(direct, dict):
        return direct

    for match in _JSON_BLOCK_RE.finditer(text):
        obj = _try_json_loads(match.group(1))
        if isinstance(obj, dict):
            return obj

    for match in _BRACE_RE.finditer(text):
        obj = _try_json_loads(match.group(0))
        if isinstance(obj, dict) and "name" in obj:
            return obj

    return None


def parse_function_call(text: str) -> Tuple[Optional[FunctionCall], Optional[str]]:
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
