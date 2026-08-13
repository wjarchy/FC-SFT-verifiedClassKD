"""Schema validation against tool definitions."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from fc_class_kd.types import FunctionCall, ToolSchema

_TYPE_ALIASES = {
    "str": "string",
    "string": "string",
    "text": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "double": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}


def normalize_param_type(raw: str) -> str:
    cleaned = raw.lower().strip()
    # xLAM stores Python-style annotations and may append ", optional" or a textual
    # default to the type string.  The verifier only needs the top-level JSON type.
    if cleaned.startswith(("list[", "tuple[")) or cleaned in {"list", "tuple", "set"}:
        return "array"
    if cleaned.startswith("dict[") or cleaned == "dict":
        return "object"
    base = cleaned.split(",", 1)[0].strip()
    return _TYPE_ALIASES.get(base, base)


def _is_type_match(value: Any, expected: str) -> bool:
    if "|" in expected:
        return any(_is_type_match(value, member) for member in expected.split("|"))
    expected = normalize_param_type(expected)
    if expected == "null":
        return value is None
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True  # Production adapters reject unknown types before this point.


def find_tool_schema(tools: List[ToolSchema], name: str) -> Optional[ToolSchema]:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def check_schema(
    call: FunctionCall,
    tools: List[ToolSchema],
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Validate call against tool schemas.

    Returns (is_valid_structure, issues, primary_issue_kind).
    primary_issue_kind: wrong_function | wrong_arguments | invalid | None
    """
    issues: List[str] = []
    tool = find_tool_schema(tools, call.name)
    if tool is None:
        return False, ["function_not_in_schema"], "wrong_function"

    params = tool.parameters
    arg_keys = set(call.arguments.keys())
    schema_keys = set(params.keys())

    missing_required = [
        k for k, p in params.items() if p.required and k not in call.arguments
    ]
    if missing_required:
        issues.append(f"missing_required:{','.join(sorted(missing_required))}")

    unknown_keys = [k for k in arg_keys if k not in schema_keys]
    if unknown_keys:
        issues.append(f"unknown_keys:{','.join(sorted(unknown_keys))}")

    wrong_type_keys: List[str] = []
    enum_errors: List[str] = []
    for key, value in call.arguments.items():
        if key not in params:
            continue
        spec = params[key]
        if not _is_type_match(value, spec.type):
            wrong_type_keys.append(key)
        if spec.enum is not None and value not in spec.enum:
            enum_errors.append(key)

    if wrong_type_keys:
        issues.append(f"wrong_type:{','.join(sorted(wrong_type_keys))}")
    if enum_errors:
        issues.append(f"wrong_enum:{','.join(sorted(enum_errors))}")

    if issues:
        if any(i.startswith("missing_required") or i.startswith("unknown_keys") for i in issues):
            kind = "wrong_arguments"
        elif any(i.startswith("wrong_type") or i.startswith("wrong_enum") for i in issues):
            kind = "wrong_arguments"
        else:
            kind = "wrong_arguments"
        return False, issues, kind

    return True, [], None
