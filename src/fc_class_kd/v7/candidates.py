"""Deterministic V7 K=4 candidate pools."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from fc_class_kd.types import CandidateRecord, FCRecord, FunctionCall, ToolParameter
from fc_class_kd.v7.equivalence import build_strict_equivalent, canonical_signature
from fc_class_kd.verifier.parser import serialize_function_call
from fc_class_kd.verifier.schema_checker import find_tool_schema, normalize_param_type


def _candidate(record: FCRecord, source: str, call: FunctionCall) -> CandidateRecord:
    return CandidateRecord(
        sample_id=record.id,
        source=source,
        serialized_call=serialize_function_call(call, sort_keys=False),
        parsed_call=call,
    )


def _different_value(value: Any, spec: ToolParameter) -> Any:
    if spec.enum:
        for item in spec.enum:
            if item != value:
                return copy.deepcopy(item)
    kind = normalize_param_type(spec.type)
    if kind == "string":
        return f"{value}__changed" if isinstance(value, str) else "__changed"
    if kind == "integer":
        return value + 1 if isinstance(value, int) and not isinstance(value, bool) else 1
    if kind == "number":
        return (
            float(value) + 1.0
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else 1.0
        )
    if kind == "boolean":
        return not value if isinstance(value, bool) else True
    if kind == "array":
        return [*value, "__changed"] if isinstance(value, list) else ["__changed"]
    if kind == "object":
        changed = copy.deepcopy(value) if isinstance(value, dict) else {}
        changed["__changed"] = True
        return changed
    return "__changed"


def _wrong_argument(record: FCRecord) -> FunctionCall:
    tool = find_tool_schema(record.tools, record.gold_call.name)
    if tool is None:
        raise ValueError(f"{record.id}: gold tool missing")
    for key, value in record.gold_call.arguments.items():
        spec = tool.parameters.get(key)
        if spec is None:
            continue
        args = copy.deepcopy(record.gold_call.arguments)
        args[key] = _different_value(value, spec)
        call = FunctionCall(name=record.gold_call.name, arguments=args)
        if canonical_signature(record, call) != canonical_signature(record, record.gold_call):
            return call
    for key, spec in tool.parameters.items():
        if spec.required and key in record.gold_call.arguments:
            args = copy.deepcopy(record.gold_call.arguments)
            del args[key]
            return FunctionCall(name=record.gold_call.name, arguments=args)
    args = copy.deepcopy(record.gold_call.arguments)
    args["__wrong_argument__"] = None
    return FunctionCall(name=record.gold_call.name, arguments=args)


def _representative(spec: ToolParameter) -> Any:
    if spec.enum:
        return copy.deepcopy(spec.enum[0])
    kind = normalize_param_type(spec.type)
    return {
        "string": "placeholder",
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "array": [],
        "object": {},
    }.get(kind)


def _wrong_function(record: FCRecord) -> FunctionCall:
    for tool in record.tools:
        if tool.name == record.gold_call.name:
            continue
        args = {
            key: _representative(spec) for key, spec in tool.parameters.items() if spec.required
        }
        return FunctionCall(name=tool.name, arguments=args)
    raise ValueError(f"{record.id}: K=4 requires at least two tools")


def build_k4_pool(record: FCRecord) -> list[CandidateRecord]:
    variant = build_strict_equivalent(record)
    if variant is None:
        raise ValueError(f"{record.id}: no non-trivial strict equivalent")
    calls = [
        _candidate(record, "gold", record.gold_call),
        _candidate(record, f"equivalent_{variant.category}:{variant.rule}", variant.call),
        _candidate(record, "wrong_argument", _wrong_argument(record)),
        _candidate(record, "wrong_function", _wrong_function(record)),
    ]
    surfaces = [candidate.serialized_call for candidate in calls]
    if len(set(surfaces)) != 4:
        raise ValueError(f"{record.id}: K=4 surfaces are not unique")
    class_ids = candidate_class_ids(record, calls)
    if class_ids[0] != class_ids[1]:
        raise ValueError(f"{record.id}: Gold and Equivalent must share a class")
    if class_ids[2] == class_ids[0] or class_ids[3] == class_ids[0]:
        raise ValueError(f"{record.id}: negative candidate collapsed into the gold class")
    return calls


def candidate_class_ids(
    record: FCRecord,
    candidates: Sequence[CandidateRecord],
) -> list[str]:
    ids: list[str] = []
    for candidate in candidates:
        if candidate.parsed_call is None:
            raise ValueError(f"{record.id}: V7 K=4 candidates must be parseable")
        ids.append(canonical_signature(record, candidate.parsed_call))
    return ids
