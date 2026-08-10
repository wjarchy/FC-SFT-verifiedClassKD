"""High-precision canonicalization rules (PRD §4.3)."""

from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from fc_class_kd.types import FunctionCall, ToolSchema
from fc_class_kd.verifier.schema_checker import find_tool_schema, normalize_param_type


_DATE_PATTERNS = [
    ("%Y-%m-%d", r"^\d{4}-\d{2}-\d{2}$"),
    ("%Y/%m/%d", r"^\d{4}/\d{2}/\d{2}$"),
]


def _sort_dict_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_dict_keys(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_dict_keys(x) for x in obj]
    return obj


def _normalize_number(value: Any, param_type: str) -> Any:
    t = normalize_param_type(param_type)
    if t == "integer" and isinstance(value, float) and value == int(value):
        return int(value)
    if t == "number" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _normalize_date_string(value: str) -> str:
    for fmt, pattern in _DATE_PATTERNS:
        if re.match(pattern, value):
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return value


def _normalize_bool_string(value: Any, param_type: str) -> Any:
    t = normalize_param_type(param_type)
    if t == "boolean" and isinstance(value, str):
        low = value.lower().strip()
        if low in ("true", "false"):
            return low == "true"
    return value


def _sort_unordered_list(value: List[Any]) -> List[Any]:
    try:
        return sorted(value, key=lambda x: json_key(x))
    except TypeError:
        return value


def json_key(x: Any) -> str:
    import json

    return json.dumps(x, sort_keys=True, default=str)


def apply_schema_defaults(
    call: FunctionCall,
    tool: ToolSchema,
) -> FunctionCall:
    args = dict(call.arguments)
    for key, spec in tool.parameters.items():
        if key not in args and spec.default is not None:
            args[key] = copy.deepcopy(spec.default)
    return FunctionCall(name=call.name, arguments=args)


def canonicalize_call(
    call: FunctionCall,
    tools: List[ToolSchema],
    gold_call: Optional[FunctionCall] = None,
) -> FunctionCall:
    """
    Apply deterministic canonicalization. Idempotent: C(C(y)) = C(y).
    """
    tool = find_tool_schema(tools, call.name)
    if tool is None:
        return FunctionCall(name=call.name, arguments=_sort_dict_keys(call.arguments))

    normalized = apply_schema_defaults(call, tool)
    new_args: Dict[str, Any] = {}
    for key, value in normalized.arguments.items():
        if key not in tool.parameters:
            new_args[key] = _sort_dict_keys(value)
            continue
        spec = tool.parameters[key]
        v = _normalize_bool_string(value, spec.type)
        v = _normalize_number(v, spec.type)
        if isinstance(v, str):
            v = _normalize_date_string(v)
        if spec.unordered and isinstance(v, list):
            v = _sort_unordered_list(v)
        if normalize_param_type(spec.type) == "object" and isinstance(v, dict):
            v = _sort_dict_keys(v)
        new_args[key] = v

    new_args = _sort_dict_keys(new_args)
    return FunctionCall(name=normalized.name, arguments=new_args)


def calls_equivalent(
    candidate: FunctionCall,
    gold: FunctionCall,
    tools: List[ToolSchema],
) -> bool:
    c_canon = canonicalize_call(candidate, tools, gold)
    g_canon = canonicalize_call(gold, tools, gold)
    return (
        c_canon.name == g_canon.name
        and c_canon.arguments == g_canon.arguments
    )
