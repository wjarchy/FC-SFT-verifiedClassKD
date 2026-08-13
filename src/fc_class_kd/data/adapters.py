"""Dataset adapters for xLAM and ToolACE."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fc_class_kd.types import FCRecord, FunctionCall, ToolParameter, ToolSchema
from fc_class_kd.verifier.schema_checker import normalize_param_type

XLAM_REPO = "Salesforce/xlam-function-calling-60k"
XLAM_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
TOOLACE_QWEN_REPO = "tryumanshow/ToolACE-Qwen-cleaned"
TOOLACE_QWEN_REVISION = "74befe8a24c760ccad2b5d58bcf35d89b3bbd2be"
TOOLACE_SCHEMA_REPO = "Team-ACE/ToolACE"
TOOLACE_SCHEMA_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"


_SUPPORTED_JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def _schema_type_to_string(raw_type: Any) -> str:
    """Preserve JSON Schema union types in a verifier-readable form."""
    if isinstance(raw_type, list):
        return "|".join(str(item) for item in raw_type)
    return str(raw_type)


def _parse_tool_parameters(
    raw_params: Dict[str, Any],
) -> Optional[Dict[str, ToolParameter]]:
    """Parse xLAM's actual flat schema, not the outdated dataset-card example.

    The released JSON omits the documented ``required`` boolean. Optionality is encoded
    in strings such as ``str, optional`` and by the presence of a default.  Defaults are
    intentionally not used for equivalence because the file contains many type-invalid
    placeholder defaults.  This keeps omission validity while avoiding fake equivalence.
    """
    out: Dict[str, ToolParameter] = {}
    for name, spec in raw_params.items():
        if not isinstance(spec, dict):
            continue
        raw_type = _schema_type_to_string(spec.get("type", "string"))
        normalized_type = normalize_param_type(raw_type)
        if normalized_type not in _SUPPORTED_JSON_TYPES:
            return None
        explicit_required = spec.get("required")
        if isinstance(explicit_required, bool):
            required = explicit_required
        else:
            required = "optional" not in raw_type.lower() and "default" not in spec
        out[name] = ToolParameter(
            type=normalized_type,
            description=spec.get("description"),
            required=required,
            enum=spec.get("enum"),
            # Do not trust xLAM defaults for equivalence: many numeric/bool parameters
            # contain empty strings or other mismatched values in the released file.
            default=None,
            unordered=normalized_type == "array" and raw_type.lower().strip() == "set",
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
                type=_schema_type_to_string(pspec.get("type", "string")),
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
                    type=_schema_type_to_string(pspec.get("type", "string")),
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
        parsed_params = _parse_tool_parameters(params if isinstance(params, dict) else {})
        if parsed_params is None:
            return None
        tools.append(
            ToolSchema(
                name=t["name"],
                description=t.get("description"),
                parameters=parsed_params,
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
        metadata={
            "original_index": idx,
            "schema_policy": "xlam_infer_required_ignore_untrusted_defaults",
        },
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


def toolace_original_system_to_schemas(system: str) -> List[ToolSchema]:
    """Recover authoritative JSON Schemas from a Team-ACE/ToolACE system prompt."""
    marker = "Here is a list of functions in JSON format that you can invoke:"
    if not isinstance(system, str) or marker not in system:
        raise ValueError("ToolACE system prompt does not contain the function-list marker")
    payload = system.split(marker, 1)[1].lstrip()
    try:
        tools_raw, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("ToolACE system prompt contains invalid tool JSON") from exc
    if not isinstance(tools_raw, list):
        raise ValueError("ToolACE system tool payload must be a list")
    schemas = _openai_tools_to_schemas(tools_raw)
    if not schemas:
        raise ValueError("ToolACE system prompt produced no usable schemas")
    return schemas


def load_toolace_qwen_records(
    limit: int | None = None,
    revision: str = TOOLACE_QWEN_REVISION,
    schema_revision: str | None = TOOLACE_SCHEMA_REVISION,
) -> list:
    """Load cleaned calls and restore schemas from uniquely matched original rows."""
    if schema_revision is None:
        raise ValueError(
            "ToolACE-Qwen-cleaned has lossy required/default fields and must not be "
            "loaded as an experiment schema; provide a pinned Team-ACE/ToolACE "
            "schema_revision"
        )

    from datasets import load_dataset

    ds = load_dataset(
        TOOLACE_QWEN_REPO,
        split="train",
        revision=revision,
    )
    schema_ds = load_dataset(
        TOOLACE_SCHEMA_REPO,
        split="train",
        revision=schema_revision,
    )
    schemas_by_query: dict[str, list[tuple[int, List[ToolSchema]]]] = defaultdict(list)
    for original_index, original_row in enumerate(schema_ds):
        try:
            schemas = toolace_original_system_to_schemas(original_row["system"])
        except ValueError:
            continue
        users = [
            str(message.get("value", "")).strip()
            for message in original_row["conversations"]
            if message.get("from") == "user"
        ]
        if len(users) == 1:
            schemas_by_query[users[0]].append((original_index, schemas))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    records = []
    for i, row in enumerate(ds):
        rec = toolace_openai_row_to_record(dict(row), i)
        if rec:
            matches = schemas_by_query.get(rec.query.strip(), [])
            if len(matches) != 1:
                continue
            original_index, original_schemas = matches[0]
            if len(original_schemas) != len(rec.tools):
                continue
            if any(
                set(original.parameters) != set(cleaned.parameters)
                or (original.description or "").strip()
                != (cleaned.description or "").strip()
                for original, cleaned in zip(original_schemas, rec.tools)
            ):
                continue
            # Cleaned ToolACE occasionally strips a namespace from a function name.
            # Preserve the name used by the cleaned gold call while restoring all
            # parameter semantics from the structurally identical original schema.
            schemas = [
                ToolSchema(
                    name=cleaned.name,
                    description=original.description,
                    parameters=original.parameters,
                )
                for original, cleaned in zip(original_schemas, rec.tools)
            ]
            rec = rec.model_copy(
                update={
                    "tools": schemas,
                    "metadata": {
                        **rec.metadata,
                        "schema_source": TOOLACE_SCHEMA_REPO,
                        "schema_revision": schema_revision,
                        "schema_original_index": original_index,
                        "schema_match": "unique_query_and_ordered_structure",
                    },
                }
            )
            records.append(rec)
    return records
