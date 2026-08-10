"""Dataset adapters for xLAM and ToolACE."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fc_class_kd.types import FCRecord, FunctionCall, ToolParameter, ToolSchema


def _parse_tool_parameters(raw_params: Dict[str, Any]) -> Dict[str, ToolParameter]:
    out: Dict[str, ToolParameter] = {}
    for name, spec in raw_params.items():
        if not isinstance(spec, dict):
            continue
        out[name] = ToolParameter(
            type=str(spec.get("type", "string")),
            description=spec.get("description"),
            required=bool(spec.get("required", False)),
            enum=spec.get("enum"),
            default=spec.get("default"),
            unordered=bool(spec.get("unordered", False)),
        )
    return out


def _flat_tool_to_schema(tool: Dict[str, Any]) -> Optional[ToolSchema]:
    name = tool.get("name")
    if not name:
        return None
    params_raw = tool.get("parameters", {})
    parameters: Dict[str, ToolParameter] = {}
    if isinstance(params_raw, dict):
        for pname, pspec in params_raw.items():
            if not isinstance(pspec, dict):
                continue
            parameters[pname] = ToolParameter(
                type=str(pspec.get("type", "string")),
                description=pspec.get("description"),
                required=bool(pspec.get("required", False)),
                enum=pspec.get("enum"),
                default=pspec.get("default"),
                unordered=bool(pspec.get("unordered", False)),
            )
    return ToolSchema(name=name, description=tool.get("description"), parameters=parameters)


def _is_flat_parameters(params: Dict[str, Any]) -> bool:
    if not params:
        return False
    return all(isinstance(v, dict) and "type" in v for v in params.values())


def _openai_tools_to_schemas(tools: List[Dict[str, Any]]) -> List[ToolSchema]:
    schemas: List[ToolSchema] = []
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name")
        if not name:
            continue

        params_raw = fn.get("parameters", {})
        if isinstance(params_raw, dict) and _is_flat_parameters(params_raw):
            flat = _flat_tool_to_schema(fn)
            if flat:
                schemas.append(flat)
            continue

        properties = params_raw.get("properties", params_raw) if isinstance(params_raw, dict) else {}
        required_set = set(params_raw.get("required", [])) if isinstance(params_raw, dict) else set()
        parameters: Dict[str, ToolParameter] = {}
        if isinstance(properties, dict):
            for pname, pspec in properties.items():
                if not isinstance(pspec, dict):
                    continue
                parameters[pname] = ToolParameter(
                    type=str(pspec.get("type", "string")),
                    description=pspec.get("description"),
                    required=pname in required_set,
                    enum=pspec.get("enum"),
                    default=pspec.get("default"),
                    unordered=bool(pspec.get("unordered", False)),
                )
        schemas.append(
            ToolSchema(
                name=name,
                description=fn.get("description"),
                parameters=parameters,
            )
        )
    return schemas


def xlam_row_to_record(row: Dict[str, Any], idx: int) -> Optional[FCRecord]:
    query = row.get("query")
    if not query:
        return None

    tools_raw = row.get("tools")
    answers_raw = row.get("answers")
    if isinstance(tools_raw, str):
        tools_raw = json.loads(tools_raw)
    if isinstance(answers_raw, str):
        answers_raw = json.loads(answers_raw)

    if not isinstance(tools_raw, list) or not isinstance(answers_raw, list):
        return None
    if len(answers_raw) != 1:
        return None

    tools: List[ToolSchema] = []
    for t in tools_raw:
        if not isinstance(t, dict) or "name" not in t:
            continue
        params = t.get("parameters", {})
        tools.append(
            ToolSchema(
                name=t["name"],
                description=t.get("description"),
                parameters=_parse_tool_parameters(params if isinstance(params, dict) else {}),
            )
        )
    if len(tools) < 2:
        return None

    ans = answers_raw[0]
    if not isinstance(ans, dict) or "name" not in ans:
        return None
    args = ans.get("arguments", {})
    if not isinstance(args, dict):
        return None

    return FCRecord(
        id=f"xlam_{idx}",
        source="xlam",
        query=str(query),
        tools=tools,
        gold_call=FunctionCall(name=ans["name"], arguments=dict(args)),
        metadata={"original_index": idx},
    )


def toolace_openai_row_to_record(row: Dict[str, Any], idx: int) -> Optional[FCRecord]:
    """OpenAI-normalized ToolACE (tuandunghcmut/toolace, ToolACE-Qwen-cleaned)."""
    messages_raw = row.get("messages") or row.get("conversations")
    tools_raw = row.get("tools_json") or row.get("tools")
    if isinstance(messages_raw, str):
        messages = json.loads(messages_raw)
    else:
        messages = messages_raw
    if isinstance(tools_raw, str):
        tools_list = json.loads(tools_raw)
    else:
        tools_list = tools_raw

    if not isinstance(messages, list) or not isinstance(tools_list, list):
        return None

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if len(user_msgs) != 1:
        return None

    assistant_tool_calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            assistant_tool_calls.extend(m["tool_calls"])
    if len(assistant_tool_calls) != 1:
        return None

    tc = assistant_tool_calls[0]
    fn = tc.get("function", {})
    name = fn.get("name")
    args_raw = fn.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            return None
    else:
        args = args_raw
    if not name or not isinstance(args, dict):
        return None

    tools = _openai_tools_to_schemas(tools_list)
    if len(tools) < 2:
        return None

    return FCRecord(
        id=f"toolace_{idx}",
        source="toolace",
        query=str(user_msgs[0].get("content", "")),
        tools=tools,
        gold_call=FunctionCall(name=name, arguments=dict(args)),
        metadata={"original_index": idx},
    )


def load_toolace_qwen_records(limit: int | None = None) -> list:
    """Load tryumanshow/ToolACE-Qwen-cleaned (ungated on HF Hub)."""
    from datasets import load_dataset

    ds = load_dataset("tryumanshow/ToolACE-Qwen-cleaned", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    records = []
    for i, row in enumerate(ds):
        rec = toolace_openai_row_to_record(dict(row), i)
        if rec:
            records.append(rec)
    return records
